/*
 * abi_fix.h - force-included (-include) BEFORE guard.c. Works around a clang
 * code-generation bug on Windows x64; guard.c itself is not edited.
 *
 * THE BUG (clang 21.1.0 as bundled in zig 0.16.0, target x86_64-windows-*):
 * guard.c's work loop is a chain of `__attribute__((musttail))` calls between
 * functions taking `Env` (a 16-byte struct) by value. The Windows x64 ABI
 * passes such a struct indirectly (pointer to a caller-side copy). For a
 * musttail call clang still makes that copy in the caller's frame and then
 * jumps, so the callee reads a pointer into a dead frame: garbage `Env`,
 * wild writes, "memory fault". A 20-line reproducer (musttail + 16-byte
 * struct argument, see musttail_repro.c next to this file, with build instructions) segfaults with and without
 * preserve_none, and works with sysv_abi. On Linux/macOS
 * the struct travels in two registers and the bug does not exist.
 *
 * THE WORKAROUND: guard.c spells its calling convention with the attribute
 * names `preserve_none` (work-loop functions and their function-pointer type)
 * and `preserve_most` (cold helpers), through `PRESERVE(preserve_none)`.
 * Macro arguments are expanded before substitution, so mapping the attribute
 * NAME to `sysv_abi` gives every work-loop function the System V x86-64
 * convention, where the 16-byte struct goes in two registers and musttail is
 * sound (the same convention the proven Linux/macOS build uses for it).
 * preserve_most becomes `cold` (already on those helpers); they are
 * ordinary calls and never tail-called.
 *
 * Only calling conventions change; no logic. The full test suite (fixed
 * cases plus 300 random paragraphs against a Python reference) runs against
 * the resulting binary.
 */
#ifndef COMPAT_ABI_FIX_H
#define COMPAT_ABI_FIX_H
#define preserve_none sysv_abi
#define preserve_most cold
#endif
