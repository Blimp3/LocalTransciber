/*
 * poll.h shim (Windows): event constants only. guard.c uses POLLIN/POLLOUT as
 * tags for its own select-based loop and never calls poll(). NOT called.
 */
#ifndef COMPAT_POLL_H
#define COMPAT_POLL_H
#define POLLIN  0x001
#define POLLOUT 0x004
#endif
