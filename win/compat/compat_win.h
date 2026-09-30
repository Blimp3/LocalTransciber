/*
 * compat_win.h - shared declarations for the Windows compatibility layer.
 *
 * WHY THIS EXISTS: Bend 2.0.32 emits one C file (guard.c) that is written for
 * Linux/macOS: it calls mmap, mprotect, sigaction, pipe, select, sysconf ...
 * and has no _WIN32 branch. We must not edit generated C, so this directory
 * is put first on the include path (-I win/compat) and supplies the POSIX
 * headers that mingw-w64 lacks, plus one small C file (compat_win.c) that is
 * compiled together with guard.c.
 *
 * Only what guard.c references is provided. Each header says whether the
 * proven checker (guard_cli) reaches the function at runtime.
 */
#ifndef COMPAT_WIN_H
#define COMPAT_WIN_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

/* Wake-pipe emulation and fd dispatch (compat_win.c). */
int     compat_pipe(int fds[2]);
ssize_t compat_read(int fd, void *buf, size_t n);
ssize_t compat_write(int fd, const void *buf, size_t n);
int     compat_close(int fd);

#endif
