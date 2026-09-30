#!/usr/bin/env bash
# Maintainers only: fetch the tools that build_guard_windows.sh needs into $BEND_TOOLCHAIN (a folder you choose).
#   BEND_TOOLCHAIN=<dir> bash win/fetch_toolchain.sh            download, verify, extract
#   BEND_TOOLCHAIN=<dir> bash win/fetch_toolchain.sh --check    only verify what is already in <dir>/dl (no network)
# BEND_MIRROR=<folder> copies the archives from a local folder instead of downloading (same file names as below).
# Result: <dir>/dl/ (the archives, kept for the build's checks), <dir>/bun/, <dir>/bend/, <dir>/zig/, and
# <dir>/bend/bend2/main.js cut out of the Linux tarball. UNOFFICIAL BRIDGE: Bend 2 has no Windows release, see the
# header of build_guard_windows.sh. Every archive's sha256 is checked BEFORE it is extracted; the hashes are the
# same ones pinned in build_guard_windows.sh.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
TC="${BEND_TOOLCHAIN:?set BEND_TOOLCHAIN to the folder to fill}"
PYTHON="${PYTHON:-python}"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1

# name | sha256 | official URL
ITEMS=(
  "bend-2.0.32-linux-x64.tar.gz|5c365ddb12954d0933cef751802e0f7d9875f842edcb80f9661f89cd1a9ff7b6|https://github.com/bendlang/bend/releases/download/v2.0.32/bend-2.0.32-linux-x64.tar.gz"
  "bun-windows-x64.zip|ce4c17497b2f29712a99d3d53f028de28cd42e3bacb8589599e7f000e49b6405|https://github.com/oven-sh/bun/releases/download/bun-v1.4.2/bun-windows-x64.zip"
  "zig-x86_64-windows-0.16.0.zip|68659eb5f1e4eb1437a722f1dd889c5a322c9954607f5edcf337bc3684a75a7e|https://ziglang.org/download/0.16.0/zig-x86_64-windows-0.16.0.zip"
)
MAIN_JS_SHA=06151d6f68b8abff5e8d487684b57cc7800f958dbc8dc915dddb15021002e728
mkdir -p "$TC/dl"

sha_of() { sha256sum "$1" | cut -d' ' -f1; }
for item in "${ITEMS[@]}"; do
  IFS='|' read -r name sha url <<< "$item"
  f="$TC/dl/$name"
  if [ ! -e "$f" ] && [ $CHECK = 0 ]; then
    if [ -n "${BEND_MIRROR:-}" ]; then cp "$BEND_MIRROR/$name" "$f"; else curl -fL --retry 3 -o "$f" "$url"; fi
  fi
  [ -e "$f" ] || { echo "fetch_toolchain: $f is missing" >&2; exit 1; }
  got=$(sha_of "$f")
  [ "$got" = "$sha" ] || { echo "fetch_toolchain: $name has sha256 $got, expected $sha" >&2; exit 1; }
  echo "ok  $name  $sha"
done
[ $CHECK = 1 ] && exit 0

# Extract (only after the checks above): drop the archive's top folder; from the Bend tarball keep bend2/ and guide/.
"$PYTHON" - "$TC" <<'PY'
import os, sys, tarfile, zipfile
tc = sys.argv[1]
def put(dest, rel, data_or_none, is_dir):
    rel = rel.split("/", 1)[1] if "/" in rel else ""
    if not rel or ".." in rel.split("/"): return
    path = os.path.join(dest, *rel.split("/"))
    if is_dir: os.makedirs(path, exist_ok=True); return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f: f.write(data_or_none())
for name, dest in (("bun-windows-x64.zip", "bun"), ("zig-x86_64-windows-0.16.0.zip", "zig")):
    with zipfile.ZipFile(os.path.join(tc, "dl", name)) as z:
        for i in z.infolist(): put(os.path.join(tc, dest), i.filename, lambda i=i: z.read(i), i.is_dir())
with tarfile.open(os.path.join(tc, "dl", "bend-2.0.32-linux-x64.tar.gz")) as t:
    for m in t.getmembers():
        if m.isfile() and m.name.split("/")[1:2] in (["bend2"], ["guide"]):
            put(os.path.join(tc, "bend"), m.name, lambda m=m: t.extractfile(m).read(), False)
PY
"$PYTHON" "$here/extract_bend_main.py" "$TC/dl/bend-2.0.32-linux-x64.tar.gz" "$TC/bend"
got=$(sha_of "$TC/bend/bend2/main.js")
[ "$got" = "$MAIN_JS_SHA" ] || { echo "fetch_toolchain: main.js has sha256 $got, expected $MAIN_JS_SHA" >&2; exit 1; }
echo "ok  bend2/main.js  $MAIN_JS_SHA"
echo "toolchain ready: BEND_TOOLCHAIN=$TC bash build_guard_windows.sh"
