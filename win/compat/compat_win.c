/*
 * compat_win.c - POSIX-on-Windows helpers for the Bend-generated guard.c.
 *
 * Compiled TOGETHER with guard.c (guard.c itself is never edited); the headers
 * next to this file declare what is defined here. Everything is small and
 * only covers what the generated runtime references. Per function, whether
 * the guard program reaches it at runtime:
 *
 *  mmap / munmap / mprotect   REACHED  address-space reservation for the
 *                                      corpus (8 GiB, grown in place) and the
 *                                      per-thread term stacks (2 GiB + guard)
 *  vectored exception handler REACHED  commits 64 KiB chunks on first touch
 *                                      (Windows has no MAP_NORESERVE), and
 *                                      forwards real faults to the runtime's
 *                                      SIGSEGV trap ("memory fault" message)
 *  sigaction / sigaltstack    REACHED  install the trap; sigaltstack no-op
 *  compat_signal              REACHED  SIGPIPE ignore (no such signal here)
 *  sysconf                    REACHED  processor count
 *  compat_pipe/read/write/close
 *  fcntl / select             REACHED  wake pipe between the file-helper
 *                                      threads and the main loop; read/close
 *                                      also forward to the CRT for real files
 *  readlink                   NOT reached (GPU program path only)
 *  constructor compat_init    REACHED  binary mode for stdio and open(), so
 *                                      "\n" is written as "\n" (no CRLF) and
 *                                      file sizes match the bytes read
 */
#define WIN32_LEAN_AND_MEAN
#define _CRT_DECLARE_NONSTDC_NAMES 1
#include <windows.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/time.h>
#include <signal.h>
#include "compat_win.h"
#undef signal

/* ---- address space: reserve now, commit on first touch ------------------ */

#define COMMIT_CHUNK ((size_t)64 << 10)
#define MAX_REGS     4096
#define MAX_GUARDS   4096

typedef struct { char *lo; char *hi; } Range;

static SRWLOCK g_lock = SRWLOCK_INIT;
static Range   g_regs[MAX_REGS];   /* one per VirtualAlloc reservation */
static int     g_nregs;
static Range   g_guards[MAX_GUARDS]; /* PROT_NONE ranges: never committed */
static int     g_nguards;
static void  (*g_segv)(int);

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  (void)off;
  if (!(flags & MAP_ANON) || fd != -1) { errno = ENOSYS; return MAP_FAILED; }
  DWORD pr = prot == PROT_NONE ? PAGE_NOACCESS : PAGE_READWRITE;
  /* Like POSIX without MAP_FIXED, addr is only a hint. */
  void *p = VirtualAlloc(addr, len, MEM_RESERVE, pr);
  if (p == NULL && addr != NULL) p = VirtualAlloc(NULL, len, MEM_RESERVE, pr);
  if (p == NULL) { errno = ENOMEM; return MAP_FAILED; }
  AcquireSRWLockExclusive(&g_lock);
  int stored = g_nregs < MAX_REGS;
  if (stored) {
    g_regs[g_nregs].lo = p;
    g_regs[g_nregs].hi = (char *)p + len;
    g_nregs++;
  }
  ReleaseSRWLockExclusive(&g_lock);
  if (!stored) { /* table full: an untracked region would fault as a real crash, so fail instead */
    VirtualFree(p, 0, MEM_RELEASE);
    errno = ENOMEM;
    return MAP_FAILED;
  }
  return p;
}

int munmap(void *addr, size_t len) {
  (void)len; /* only whole reservations are ever unmapped */
  int found = -1;
  AcquireSRWLockExclusive(&g_lock);
  for (int i = 0; i < g_nregs; i++) {
    if (g_regs[i].lo == (char *)addr) { found = i; break; }
  }
  if (found >= 0) {
    Range r = g_regs[found];
    g_regs[found] = g_regs[--g_nregs];
    for (int i = 0; i < g_nguards;) {
      if (g_guards[i].lo >= r.lo && g_guards[i].hi <= r.hi) g_guards[i] = g_guards[--g_nguards];
      else i++;
    }
  }
  ReleaseSRWLockExclusive(&g_lock);
  if (found < 0) { errno = EINVAL; return -1; }
  return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : -1;
}

int mprotect(void *addr, size_t len, int prot) {
  AcquireSRWLockExclusive(&g_lock);
  if (prot == PROT_NONE) {
    if (g_nguards >= MAX_GUARDS) { /* cannot record the guard: report it, never run with an untracked one */
      ReleaseSRWLockExclusive(&g_lock);
      errno = ENOMEM;
      return -1;
    }
    g_guards[g_nguards].lo = addr;
    g_guards[g_nguards].hi = (char *)addr + len;
    g_nguards++;
  } else {
    for (int i = 0; i < g_nguards;) {
      if (g_guards[i].lo == (char *)addr) g_guards[i] = g_guards[--g_nguards];
      else i++;
    }
  }
  ReleaseSRWLockExclusive(&g_lock);
  if (prot == PROT_NONE) VirtualFree(addr, len, MEM_DECOMMIT); /* fails harmlessly if never committed */
  return 0;
}

static LONG CALLBACK compat_veh(EXCEPTION_POINTERS *ep) {
  DWORD code = ep->ExceptionRecord->ExceptionCode;
  if (code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2) {
    char *a = (char *)ep->ExceptionRecord->ExceptionInformation[1];
    char *lo = NULL, *hi = NULL;
    AcquireSRWLockExclusive(&g_lock);
    for (int i = 0; i < g_nregs; i++) {
      if (a >= g_regs[i].lo && a < g_regs[i].hi) { lo = g_regs[i].lo; hi = g_regs[i].hi; break; }
    }
    if (lo != NULL) {
      for (int i = 0; i < g_nguards; i++) {
        if (a >= g_guards[i].lo && a < g_guards[i].hi) { lo = NULL; break; }
      }
    }
    if (lo != NULL) {
      char *c0 = (char *)((uintptr_t)a & ~(uintptr_t)(COMMIT_CHUNK - 1));
      char *c1 = c0 + COMMIT_CHUNK;
      if (c0 < lo) c0 = lo;
      if (c1 > hi) c1 = hi;
      for (int i = 0; i < g_nguards; i++) { /* never commit across a guard */
        if (g_guards[i].hi <= a && g_guards[i].hi > c0) c0 = g_guards[i].hi;
        if (g_guards[i].lo > a && g_guards[i].lo < c1) c1 = g_guards[i].lo;
      }
      ReleaseSRWLockExclusive(&g_lock);
      if (VirtualAlloc(c0, (size_t)(c1 - c0), MEM_COMMIT, PAGE_READWRITE) != NULL) {
        return EXCEPTION_CONTINUE_EXECUTION;
      }
    } else {
      ReleaseSRWLockExclusive(&g_lock);
    }
  }
  if ((code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_STACK_OVERFLOW) && g_segv != NULL) {
    g_segv(SIGSEGV); /* the runtime's err_trap: prints a message, _exit(1) */
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

/* ---- signals ------------------------------------------------------------ */

int sigaction(int sig, const struct sigaction *act, struct sigaction *old) {
  if (old != NULL) memset(old, 0, sizeof *old);
  if (act != NULL && (sig == SIGSEGV || sig == SIGBUS)) g_segv = act->sa_handler;
  return 0;
}

int sigaltstack(const stack_t *ss, stack_t *old) {
  (void)ss;
  if (old != NULL) memset(old, 0, sizeof *old);
  return 0;
}

void (*compat_signal(int sig, void (*h)(int)))(int) {
  switch (sig) {
    case SIGINT: case SIGILL: case SIGFPE: case SIGSEGV: case SIGTERM: case SIGABRT:
      return (signal)(sig, h);
    default: /* SIGPIPE, SIGBUS: not raised on Windows */
      return SIG_DFL;
  }
}

/* ---- system information ------------------------------------------------- */

long sysconf(int name) {
  (void)name; /* only _SC_NPROCESSORS_ONLN is ever asked */
  DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  return n > 0 ? (long)n : 1;
}

ssize_t readlink(const char *path, char *buf, size_t n) {
  (void)path; (void)buf; (void)n;
  errno = ENOSYS;
  return -1;
}

/* ---- emulated pipe: a byte queue plus an event, for the wake channel ---- */

#define PIPE_BASE 2048 /* fds >= this are emulated; CRT fds stay far below */
#define PIPE_MAX  8
#define PIPE_CAP  65536

typedef struct {
  int              used;
  int              nonblock;
  int              r_open, w_open;
  CRITICAL_SECTION cs;
  HANDLE           ev; /* manual-reset, signalled while data or EOF is pending */
  size_t           head, count;
  unsigned char    buf[PIPE_CAP];
} Pipe;

static Pipe            g_pipes[PIPE_MAX];
static CRITICAL_SECTION g_pipe_table;
static INIT_ONCE       g_pipe_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK pipe_table_init(PINIT_ONCE o, PVOID p, PVOID *c) {
  (void)o; (void)p; (void)c;
  InitializeCriticalSection(&g_pipe_table);
  return TRUE;
}

static Pipe *pipe_of(int fd, int *is_read) {
  if (fd < PIPE_BASE || fd >= PIPE_BASE + 2 * PIPE_MAX) return NULL;
  Pipe *p = &g_pipes[(fd - PIPE_BASE) / 2];
  if (!p->used) return NULL;
  *is_read = (fd - PIPE_BASE) % 2 == 0;
  return p;
}

static void pipe_sync_event(Pipe *p) { /* caller holds p->cs */
  if (p->count > 0 || !p->w_open) SetEvent(p->ev); else ResetEvent(p->ev);
}

int compat_pipe(int fds[2]) {
  InitOnceExecuteOnce(&g_pipe_once, pipe_table_init, NULL, NULL);
  EnterCriticalSection(&g_pipe_table);
  for (int i = 0; i < PIPE_MAX; i++) {
    Pipe *p = &g_pipes[i];
    if (p->used) continue;
    memset(p, 0, sizeof *p);
    InitializeCriticalSection(&p->cs);
    p->ev = CreateEventA(NULL, TRUE, FALSE, NULL);
    p->used = p->r_open = p->w_open = 1;
    fds[0] = PIPE_BASE + 2 * i;
    fds[1] = PIPE_BASE + 2 * i + 1;
    LeaveCriticalSection(&g_pipe_table);
    return 0;
  }
  LeaveCriticalSection(&g_pipe_table);
  errno = EMFILE;
  return -1;
}

ssize_t compat_read(int fd, void *buf, size_t n) {
  int rd;
  Pipe *p = pipe_of(fd, &rd);
  if (p == NULL) return _read(fd, buf, n > 0x7fffffffu ? 0x7fffffffu : (unsigned)n);
  if (!rd) { errno = EBADF; return -1; }
  for (;;) {
    EnterCriticalSection(&p->cs);
    if (p->count > 0) {
      size_t k = n < p->count ? n : p->count, first = PIPE_CAP - p->head;
      if (first > k) first = k;
      memcpy(buf, p->buf + p->head, first);
      memcpy((char *)buf + first, p->buf, k - first);
      p->head = (p->head + k) % PIPE_CAP;
      p->count -= k;
      pipe_sync_event(p);
      LeaveCriticalSection(&p->cs);
      return (ssize_t)k;
    }
    int eof = !p->w_open;
    LeaveCriticalSection(&p->cs);
    if (eof) return 0;
    if (p->nonblock) { errno = EAGAIN; return -1; }
    WaitForSingleObject(p->ev, INFINITE);
  }
}

ssize_t compat_write(int fd, const void *buf, size_t n) {
  int rd;
  Pipe *p = pipe_of(fd, &rd);
  if (p == NULL) return _write(fd, buf, n > 0x7fffffffu ? 0x7fffffffu : (unsigned)n);
  if (rd) { errno = EBADF; return -1; }
  EnterCriticalSection(&p->cs);
  if (!p->r_open) { LeaveCriticalSection(&p->cs); errno = EPIPE; return -1; }
  if (PIPE_CAP - p->count < n) { LeaveCriticalSection(&p->cs); errno = EAGAIN; return -1; }
  size_t tail = (p->head + p->count) % PIPE_CAP, first = PIPE_CAP - tail;
  if (first > n) first = n;
  memcpy(p->buf + tail, buf, first);
  memcpy(p->buf, (const char *)buf + first, n - first);
  p->count += n;
  pipe_sync_event(p);
  LeaveCriticalSection(&p->cs);
  return (ssize_t)n;
}

int compat_close(int fd) {
  int rd;
  Pipe *p = pipe_of(fd, &rd);
  if (p == NULL) return _close(fd);
  EnterCriticalSection(&p->cs);
  if (rd) p->r_open = 0; else p->w_open = 0;
  pipe_sync_event(p);
  LeaveCriticalSection(&p->cs);
  return 0;
}

int fcntl(int fd, int cmd, ...) {
  int rd;
  Pipe *p = pipe_of(fd, &rd);
  va_list ap;
  va_start(ap, cmd);
  int arg = cmd == F_SETFL ? va_arg(ap, int) : 0;
  va_end(ap);
  if (p != NULL && cmd == F_SETFL) p->nonblock = (arg & O_NONBLOCK) != 0;
  return 0;
}

/* select over the bitset layout guard.c builds (bit fd%8 of byte fd/8).
 * Emulated pipe read ends are ready when data/EOF is queued; every other fd
 * (regular files, write sets) is always ready, as on POSIX. */
static int bit_get(const fd_set *s, int fd) { return s != NULL && (s->fds_bits[fd >> 3] >> (fd & 7) & 1); }
static void bit_clr(fd_set *s, int fd) { if (s != NULL) s->fds_bits[fd >> 3] &= (unsigned char)~(1u << (fd & 7)); }

int select(int nfds, fd_set *r, fd_set *w, fd_set *x, struct timeval *tv) {
  (void)x;
  ULONGLONG t0 = GetTickCount64();
  ULONGLONG limit = tv == NULL ? 0 : (ULONGLONG)tv->tv_sec * 1000 + (ULONGLONG)(tv->tv_usec + 999) / 1000;
  static unsigned char keep_r[1 << 16], keep_w[1 << 16]; /* select is called from the main loop thread only */
  size_t bytes = ((size_t)nfds + 7) / 8;
  if (bytes > sizeof keep_r) { errno = EINVAL; return -1; }
  for (;;) {
    HANDLE evs[PIPE_MAX];
    int nev = 0, ready = 0;
    memset(keep_r, 0, bytes);
    memset(keep_w, 0, bytes);
    for (int fd = 0; fd < nfds; fd++) {
      if (bit_get(r, fd)) {
        int rd;
        Pipe *p = pipe_of(fd, &rd);
        int ok = 1;
        if (p != NULL && rd) {
          EnterCriticalSection(&p->cs);
          ok = p->count > 0 || !p->w_open;
          LeaveCriticalSection(&p->cs);
          if (!ok && nev < PIPE_MAX) evs[nev++] = p->ev;
        }
        if (ok) { keep_r[fd >> 3] |= (unsigned char)(1u << (fd & 7)); ready++; }
      }
      if (bit_get(w, fd)) { keep_w[fd >> 3] |= (unsigned char)(1u << (fd & 7)); ready++; }
    }
    ULONGLONG used = GetTickCount64() - t0;
    if (ready > 0 || (tv != NULL && used >= limit)) {
      for (int fd = 0; fd < nfds; fd++) {
        if (!(keep_r[fd >> 3] >> (fd & 7) & 1)) bit_clr(r, fd);
        if (!(keep_w[fd >> 3] >> (fd & 7) & 1)) bit_clr(w, fd);
      }
      return ready;
    }
    DWORD ms = tv == NULL ? INFINITE : (DWORD)(limit - used);
    if (nev > 0) WaitForMultipleObjects((DWORD)nev, evs, FALSE, ms);
    else if (ms != INFINITE) Sleep(ms);
    else { errno = EINVAL; return -1; }
  }
}

/* ---- startup ------------------------------------------------------------ */

__attribute__((constructor)) static void compat_init(void) {
  _set_fmode(_O_BINARY); /* open() gives byte-exact files */
  _setmode(0, _O_BINARY);
  _setmode(1, _O_BINARY); /* no LF -> CRLF on stdout */
  _setmode(2, _O_BINARY);
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  AddVectoredExceptionHandler(1, compat_veh);
}
