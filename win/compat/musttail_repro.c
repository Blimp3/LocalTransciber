/*
 * Minimal reproducer of the clang `musttail` + 16-byte struct bug on Windows x64
 * (clang 21 as bundled in zig 0.16.0). See the comment in abi_fix.h for why it matters.
 *
 * Build and run (from the repository root):
 *   zig cc -target x86_64-windows-gnu -O2 win/compat/musttail_repro.c -o musttail_bad.exe
 *   zig cc -target x86_64-windows-gnu -O2 -DUSE_SYSV win/compat/musttail_repro.c -o musttail_ok.exe
 *
 * Expected output is 1000107 (mem[3] + alc[1] + the loop count: 100 + 7 + 1000000 steps of r4).
 *   musttail_bad.exe   the Env struct (16 bytes) is passed by hidden pointer under the Win64 ABI; clang copies it
 *                      in the caller's frame and then tail-jumps, so the callee reads a dead frame: it crashes
 *                      or prints garbage instead of 1000107.
 *   musttail_ok.exe    with __attribute__((sysv_abi)) the struct travels in two registers, musttail is sound
 *                      and it prints 1000107. This is what abi_fix.h does for the generated guard.c.
 * Nothing here is used by the build; it only documents the bug.
 */
#include <stdio.h>
#include <stdint.h>
typedef uint64_t u64;
typedef struct { u64* mem; u64* alc; } Env;
#ifdef USE_SYSV
#define CC __attribute__((sysv_abi))
#else
#define CC
#endif
CC __attribute__((noinline)) u64 b(Env e, u64* sp, unsigned seq, unsigned rn, u64 r0, u64 r1, u64 r2, u64 r3, u64 r4);
CC __attribute__((noinline)) u64 a(Env e, u64* sp, unsigned seq, unsigned rn, u64 r0, u64 r1, u64 r2, u64 r3, u64 r4) {
  if (r0 == 0) return e.mem[3] + e.alc[1] + r4;
  __attribute__((musttail)) return b(e, sp, seq, rn, r0 - 1, r1, r2, r3, r4 + 1);
}
CC __attribute__((noinline)) u64 b(Env e, u64* sp, unsigned seq, unsigned rn, u64 r0, u64 r1, u64 r2, u64 r3, u64 r4) {
  Env f = e; f.mem[0]++;
  __attribute__((musttail)) return a(f, sp, seq, rn, r0, r1, r2, r3, r4);
}
int main(void) {
  u64 m[8] = {0,0,0,100}, al[4] = {0,7};
  Env e = { m, al };
  printf("%llu\n", (unsigned long long)a(e, m, 0, 0, 1000000, 0, 0, 0, 0));
  return 0;
}
