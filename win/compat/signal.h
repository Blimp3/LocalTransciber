/*
 * signal.h shim (Windows): adds sigaction/sigaltstack/SIGBUS/SIGPIPE on top of
 * the CRT's signal.h (via #include_next).
 *
 * guard.c installs an error trap for SIGSEGV/SIGBUS (memory fault = the Bend
 * term stack ran into its guard page) and sets SIGPIPE to ignored.
 *  sigaction   - CALLED per pool thread; records the handler, which the
 *                vectored exception handler in compat_win.c invokes on a
 *                fault that is not lazy-commit.
 *  sigaltstack - CALLED per pool thread; a no-op (the fault is in the Bend
 *                stack, not the C stack).
 *  signal      - CALLED once (SIGPIPE, SIG_IGN); the CRT aborts on unknown
 *                signal numbers, so this is wrapped to ignore them.
 */
#ifndef COMPAT_SIGNAL_H
#define COMPAT_SIGNAL_H
#include_next <signal.h>
#include <stddef.h>

#ifndef SIGBUS
#define SIGBUS 7
#endif
#ifndef SIGPIPE
#define SIGPIPE 13
#endif
#define SA_ONSTACK 0x08000000
#define SIGSTKSZ   16384

typedef struct { void *ss_sp; size_t ss_size; int ss_flags; } stack_t;
struct sigaction {
  void (*sa_handler)(int);
  unsigned long sa_flags;
  unsigned long sa_mask;
};
int sigaction(int sig, const struct sigaction *act, struct sigaction *old);
int sigaltstack(const stack_t *ss, stack_t *old);
void (*compat_signal(int sig, void (*h)(int)))(int);
#define signal(s, h) compat_signal((s), (h))
#endif
