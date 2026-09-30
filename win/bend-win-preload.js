// UNOFFICIAL BRIDGE: Bend 2 does not support Windows; this is one of the two pieces (with extract_bend_main.py)
// that run the official Linux release's compiler on Windows. An official Windows Bend would replace both.
//
// Bun preload for running the Bend 2.0.32 compiler (bend2/main.js) on Windows.
//
// The compiler assumes POSIX real paths: it slices them at "/" and resolves
// imports with path.posix. On Windows fs.realpathSync returns "C:\dir\file",
// which breaks every `import ./X.bend`. This shim makes realpathSync return
// forward-slash, drive-less paths ("/Tools/x/file"), which Windows APIs read as
// drive-relative (so run it with the current drive = the drive of the sources).
// It changes only how paths are spelled; the compiler and its output are
// untouched. Nothing here writes files or touches the network.
const fs = require("fs");
const fix = (p) => (typeof p === "string" ? p.replace(/\\/g, "/").replace(/^[A-Za-z]:(?=\/)/, "") : p);
const real = fs.realpathSync;
const wrapped = (p, o) => fix(real(p, o));
wrapped.native = (p, o) => fix(real.native(p, o));
fs.realpathSync = wrapped;
