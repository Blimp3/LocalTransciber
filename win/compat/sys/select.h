/*
 * sys/select.h shim (Windows): select() over POSIX-style fd bitsets.
 *
 * guard.c builds its own byte-array bitsets (bit fd%8 of byte fd/8, the Linux
 * fd_set layout) and casts them to fd_set*. Winsock's fd_set is a different
 * structure, so this header defines a plain bitset fd_set and compat_win.c
 * implements select() for it. Only the emulated wake pipe is ever waited on
 * by the guard; any other fd is reported ready. CALLED at runtime: yes
 * (io_wait, waits for the file-helper thread to signal completion).
 */
#ifndef COMPAT_SYS_SELECT_H
#define COMPAT_SYS_SELECT_H
#include <sys/time.h>

typedef struct { unsigned char fds_bits[1 << 16]; } fd_set;
int select(int nfds, fd_set *r, fd_set *w, fd_set *x, struct timeval *tv);
#endif
