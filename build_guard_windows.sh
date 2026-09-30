#!/usr/bin/env bash
# Build the proven Bend checker (guard.bend, via guard_cli.bend) into bin/guard-windows-x64.exe.
# Windows twin of build_guard.sh (run it from Git Bash). Refuses to build unless every law in LAWS.bend is proven.
#
# UNOFFICIAL BRIDGE: Bend 2 does not support Windows. This script runs Bend anyway by cutting the compiler bundle
# (one JavaScript file) out of the official Linux release (win/extract_bend_main.py), running it on Bun for Windows
# with a small path preload (win/bend-win-preload.js), and compiling the C it emits with zig's clang plus POSIX
# shims (win/compat/). Nothing in the proofs or in the emitted guard.c is edited; the proofs are checked here, and
# the emitted guard.c is meant to be identical to the macOS build's (compare the "guard.c sha256" printed at the
# end with the one from a macOS build; the tools above are pinned so this is repeatable). When an official Windows Bend exists it replaces win/extract_bend_main.py and
# win/bend-win-preload.js, and this script shrinks to the build_guard.sh steps plus the C compiler call.
#
# Tools (nothing is installed, PATH is not touched). Get them with win/fetch_toolchain.sh, then either set
#   BEND_TOOLCHAIN  a folder holding bun/ (bun.exe), bend/ (bend2/main.js, guide/) and zig/ (zig.exe, lib/)
# or set each of BEND_BUN (bun.exe), BEND_DIR (the bend/ folder) and ZIG (zig.exe). There is no default location.
#
# What is checked at build time (sha256 pinned below, each one printed):
#   always      bun.exe, zig.exe (from the pinned archives) and bend2/main.js (extracted from the pinned tarball),
#               i.e. the programs that are actually run, and LAWS.bend (the specification).
#   if present  the three downloaded archives in $BEND_TOOLCHAIN/dl/ (kept there by win/fetch_toolchain.sh).
#   not hashed  the other files of the Bend tarball (base.bend, effs/, guide/) and zig's lib/ folder; they are
#               covered only by the archive hashes above, so keep dl/ if you want that to be checked at build time.
#   afterwards  the proofs must print ALL PROOFS CHECK, and guard.c and the exe hashes are printed for comparison.
set -euo pipefail
cd "$(dirname "$0")"

BEND_TGZ_SHA=5c365ddb12954d0933cef751802e0f7d9875f842edcb80f9661f89cd1a9ff7b6   # bend-2.0.32-linux-x64.tar.gz
BUN_ZIP_SHA=ce4c17497b2f29712a99d3d53f028de28cd42e3bacb8589599e7f000e49b6405    # bun-windows-x64.zip (bun 1.4.2)
ZIG_ZIP_SHA=68659eb5f1e4eb1437a722f1dd889c5a322c9954607f5edcf337bc3684a75a7e    # zig-x86_64-windows-0.16.0.zip
MAIN_JS_SHA=06151d6f68b8abff5e8d487684b57cc7800f958dbc8dc915dddb15021002e728    # bend2/main.js cut out of the tarball
BUN_EXE_SHA=15277c59ccd6c6c20f8dc9716c2b59c1776320d606b6a8658f70be8799519ca4    # bun.exe inside the bun zip
ZIG_EXE_SHA=086ce9d47ba42f33a514e1a6e04eb1d4a8fa1d75e0868e0213caad447c91e864    # zig.exe inside the zig zip
LAWS_SHA=d3ef92b7e8023e3c09b5eb45ac6a4ee35e9f4b2387e88d9c00ebaa0e3c5ee74d

die() { echo "build_guard_windows: $*" >&2; exit 1; }
# verify LABEL FILE SHA256 [strip-cr]: print the hash and stop if it is not the pinned one.
verify() {
  local got
  if [ "${4:-}" = strip-cr ]; then got=$(tr -d '\015' < "$2" | sha256sum | cut -d' ' -f1); else got=$(sha256sum "$2" | cut -d' ' -f1); fi
  [ "$got" = "$3" ] || die "$1 has sha256 $got, expected $3"
  echo "ok  $1  $3"
}

TC="${BEND_TOOLCHAIN:-}"
if [ -n "$TC" ]; then
  BEND_BUN="${BEND_BUN:-$TC/bun/bun.exe}"; BEND_DIR="${BEND_DIR:-$TC/bend}"; ZIG="${ZIG:-$TC/zig/zig.exe}"
fi
[ -n "${BEND_BUN:-}" ] && [ -n "${BEND_DIR:-}" ] && [ -n "${ZIG:-}" ] ||
  die "tools not set. Set BEND_TOOLCHAIN to a folder with bun/, bend/ and zig/ (win/fetch_toolchain.sh creates it), or set BEND_BUN, BEND_DIR and ZIG"
export BEND_NO_TELEMETRY=1

for f in "$BEND_BUN" "$BEND_DIR/bend2/main.js" "$ZIG"; do
  [ -e "$f" ] || die "missing $f (see win/fetch_toolchain.sh)"
done

if [ -n "$TC" ]; then  # archives, when fetch_toolchain.sh left them next to the tools
  for pair in "bend-2.0.32-linux-x64.tar.gz $BEND_TGZ_SHA" "bun-windows-x64.zip $BUN_ZIP_SHA" "zig-x86_64-windows-0.16.0.zip $ZIG_ZIP_SHA"; do
    set -- $pair
    if [ -e "$TC/dl/$1" ]; then verify "$1" "$TC/dl/$1" "$2"; else echo "--  $1 not in dl/, archive check skipped"; fi
  done
fi
verify "bun.exe" "$BEND_BUN" "$BUN_EXE_SHA"
verify "zig.exe" "$ZIG" "$ZIG_EXE_SHA"
verify "bend2/main.js" "$BEND_DIR/bend2/main.js" "$MAIN_JS_SHA"

# LAWS.bend is the maintainer's specification: refuse to build if it is not byte-for-byte the approved one.
# (CR is ignored: a Git checkout with autocrlf turns the LF file into CRLF; the compiler output is the same.)
verify "LAWS.bend (CR stripped)" LAWS.bend "$LAWS_SHA" strip-cr
# Inside a Git work tree it must also be exactly the committed file (hash-object applies the same autocrlf rules).
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  h_file=$(git hash-object LAWS.bend)
  h_head=$(git rev-parse HEAD:LAWS.bend 2>/dev/null) || die "LAWS.bend is not in HEAD"
  [ "$h_file" = "$h_head" ] || die "LAWS.bend differs from HEAD:LAWS.bend (git object $h_file, HEAD has $h_head)"
  echo "ok  LAWS.bend == HEAD:LAWS.bend  $h_head"
fi

# The Bend compiler assumes POSIX real paths; win/bend-win-preload.js fixes only how paths are spelled.
bend() { "$BEND_BUN" --preload ./win/bend-win-preload.js "$BEND_DIR/bend2/main.js" "$@"; }

out=$(bend PROOF.bend 2>&1) || true
echo "$out" | grep -q "ALL PROOFS CHECK" || { echo "$out" >&2; echo "build_guard_windows: proofs do not check, not building" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
tmpw=$(cygpath -m "$tmp")

bend guard_cli.bend -o "$tmpw/guard.c"

mkdir -p bin
# -O3 and C11 as bend's own clang call; pthreads come from zig's built-in winpthreads, linked statically.
# abi_fix.h works around a clang musttail bug on Windows x64 (see the comment in it).
"$ZIG" cc -target x86_64-windows-gnu -std=c11 -O3 -D_FILE_OFFSET_BITS=64 \
  -Iwin/compat -include win/compat/abi_fix.h \
  "$tmpw/guard.c" win/compat/compat_win.c -lm -o bin/guard-windows-x64.exe

rm -f bin/guard-windows-x64.pdb  # the linker writes a debug database next to the exe; not needed
# The linker stamps the time and a random GUID into the exe; zero them so two builds are identical.
PYTHON="${PYTHON:-python}"
"$PYTHON" win/pe_reproducible.py bin/guard-windows-x64.exe || echo "build_guard_windows: no Python, exe is not byte-reproducible" >&2

echo "1 0 -;2 1 7,8;3 1 9|5 8 4" > "$tmp/in.txt"
got=$(bin/guard-windows-x64.exe "$tmpw/in.txt")  # exact bytes: a CR from text-mode stdout would fail here
[ "$got" = "1 8 3" ] || { echo "build_guard_windows: smoke test printed '$got', expected '1 8 3'" >&2; exit 1; }

echo "built bin/guard-windows-x64.exe ($(wc -c < bin/guard-windows-x64.exe | tr -d ' ') bytes), smoke test OK"
echo "sha256 $(sha256sum bin/guard-windows-x64.exe | cut -d' ' -f1)"
echo "guard.c sha256 $(sha256sum "$tmp/guard.c" | cut -d' ' -f1)  (compare with the macOS build)"
echo "bun $("$BEND_BUN" --version), bend $(grep -o 'VERSION = "[0-9.]*"' "$BEND_DIR/bend2/main.js" | head -1 | cut -d'"' -f2), zig $("$ZIG" version)"
