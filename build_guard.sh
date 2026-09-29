#!/usr/bin/env bash
# Build the proven Bend checker (guard.bend, via guard_cli.bend) into bin/guard-macos-arm64.
# Refuses to build unless every law in LAWS.bend is proven. Needs bend (~/.bend/bin) and clang.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.bend/bin:$PATH"

out=$(bend PROOF.bend 2>&1) || true
echo "$out" | grep -q "ALL PROOFS CHECK" || { echo "$out" >&2; echo "build_guard: proofs do not check, not building" >&2; exit 1; }

mkdir -p bin
bend guard_cli.bend -o bin/guard-macos-arm64
chmod +x bin/guard-macos-arm64

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
echo "1 0 -;2 1 7,8;3 1 9|5 8 4" > "$tmp"
got=$(bin/guard-macos-arm64 "$tmp")
[ "$got" = "1 8 3" ] || { echo "build_guard: smoke test printed '$got', expected '1 8 3'" >&2; exit 1; }
echo "built bin/guard-macos-arm64 ($(wc -c < bin/guard-macos-arm64 | tr -d ' ') bytes), smoke test OK"
