/*
 * sys/socket.h shim (Windows): only the types io_sys_addr() names.
 * Deliberately does NOT pull in winsock, so the .exe imports no ws2_32.dll.
 * io_sys_addr is compiled but only reachable from TCP/UDP effects, which the
 * guard program does not use. inet_pton always fails. NOT called.
 */
#ifndef COMPAT_SYS_SOCKET_H
#define COMPAT_SYS_SOCKET_H
#include <stdint.h>
#include <errno.h>

#define AF_INET 2
struct in_addr { uint32_t s_addr; };
struct sockaddr_in {
  uint16_t       sin_family;
  uint16_t       sin_port;
  struct in_addr sin_addr;
  char           sin_zero[8];
};
static inline uint16_t htons(uint16_t v) { return (uint16_t)((v << 8) | (v >> 8)); }
static inline int inet_pton(int af, const char *src, void *dst) {
  (void)af; (void)src; (void)dst;
  errno = ENOSYS;
  return -1;
}
#endif
