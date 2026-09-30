/*
 * sys/mman.h shim (Windows): anonymous mmap/munmap/mprotect on VirtualAlloc.
 *
 * The Bend runtime reserves huge address ranges (8 GiB corpus at a high hint
 * address, doubled in place; 2 GiB per-thread term stacks with a PROT_NONE
 * guard page) using MAP_NORESERVE and relies on lazy page allocation.
 * Windows has no overcommit, so compat_win.c reserves with MEM_RESERVE and
 * commits 64 KiB chunks on first touch from a vectored exception handler.
 *
 *  mmap     - CALLED at runtime (anonymous only; file mappings are CUDA-only,
 *             not compiled for this program and return MAP_FAILED).
 *  munmap   - CALLED only when a hint address is not honoured (corpus_map
 *             retry) or growth in place fails; whole-allocation release only.
 *  mprotect - CALLED once per pool thread, only with PROT_NONE (stack guard).
 */
#ifndef COMPAT_SYS_MMAN_H
#define COMPAT_SYS_MMAN_H
#include <stddef.h>
#include <sys/types.h>

#define PROT_NONE     0
#define PROT_READ     1
#define PROT_WRITE    2
#define PROT_EXEC     4
#define MAP_SHARED    1
#define MAP_PRIVATE   2
#define MAP_ANON      0x20
#define MAP_ANONYMOUS MAP_ANON
#define MAP_NORESERVE 0x4000
#define MAP_FAILED    ((void *)-1)

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off);
int   munmap(void *addr, size_t len);
int   mprotect(void *addr, size_t len, int prot);
#endif
