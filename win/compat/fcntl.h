/*
 * fcntl.h shim (Windows): adds fcntl(F_SETFL, O_NONBLOCK) for the wake pipe
 * (via #include_next). CALLED once at startup. Real files are opened O_BINARY
 * by default (compat_win.c sets _fmode) so byte counts match file_size.
 */
#ifndef COMPAT_FCNTL_H
#define COMPAT_FCNTL_H
#include_next <fcntl.h>
#ifndef O_NONBLOCK
#define O_NONBLOCK 0x4000
#endif
#ifndef F_SETFL
#define F_GETFL 3
#define F_SETFL 4
#endif
int fcntl(int fd, int cmd, ...);
#endif
