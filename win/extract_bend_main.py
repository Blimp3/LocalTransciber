"""Get the Bend 2.0.32 compiler (one JavaScript file) out of the official Linux release.

UNOFFICIAL BRIDGE: Bend 2 does not support Windows, so this cuts the compiler out of the official Linux release
instead. An official Windows Bend would make this file (and bend-win-preload.js) unnecessary. The result is
checked by sha256 in build_guard_windows.sh and win/fetch_toolchain.sh.

Bend ships no Windows build and its tarball has no compiler source: bin/bend is a Bun single-file executable
with the compiler bundle embedded (base.bend, effs/ and guide/ are separate files in the tarball).
This reads bin/bend as data (nothing is executed) and writes the bundle to <BEND_DIR>/bend2/main.js, which
Bun then runs directly. Usage:
  python extract_bend_main.py bend-2.0.32-linux-x64.tar.gz <toolchain>/bend
The tarball's bend2/ and guide/ folders must already be extracted into that directory."""
import hashlib
import os
import sys
import tarfile

tgz, dest = sys.argv[1], sys.argv[2]
with tarfile.open(tgz) as t:
    blob = t.extractfile("bend/bin/bend").read()
start = blob.index(b"#!/usr/bin/env bun\n// @bun\n")
js = blob[start:blob.index(b"\0", start)]
out = os.path.join(dest, "bend2", "main.js")
open(out, "wb").write(js)
print(out, len(js), "bytes, sha256", hashlib.sha256(js).hexdigest())
