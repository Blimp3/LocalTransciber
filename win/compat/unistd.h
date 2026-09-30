/*
 * unistd.h shim (Windows): adds sysconf, readlink, pipe and routes
 * read/write/close through compat_win.c so the emulated wake pipe works
 * (via #include_next of the mingw unistd.h).
 *
 *  sysconf(_SC_NPROCESSORS_ONLN) - CALLED once (thread count).
 *  readlink                      - compiled (gpu_path), NOT called (GPU only).
 *  pipe / read / write / close   - CALLED: file-helper threads wake the main
 *                                  loop through a pipe; read/close also hit
 *                                  real file descriptors (open file, read it).
 */
#ifndef COMPAT_UNISTD_H
#define COMPAT_UNISTD_H
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <fcntl.h>
#include <process.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include_next <unistd.h>
#include "compat_win.h"

#define _SC_NPROCESSORS_ONLN 84
long    sysconf(int name);
ssize_t readlink(const char *path, char *buf, size_t n);

#define pipe(f)        compat_pipe(f)
#define read(f, b, n)  compat_read((f), (b), (n))
#define write(f, b, n) compat_write((f), (b), (n))
#define close(f)       compat_close(f)
#endif
