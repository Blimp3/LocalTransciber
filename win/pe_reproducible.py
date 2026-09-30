"""Make a linked Windows .exe byte-for-byte reproducible: zero the PE link timestamp and the build GUID.

The linker stamps the wall-clock time into the PE header and the debug directory and draws a random CodeView
GUID; everything else in the file is already deterministic. Usage: python pe_reproducible.py file.exe"""
import struct
import sys

path = sys.argv[1]
d = bytearray(open(path, "rb").read())
pe = struct.unpack_from("<I", d, 0x3C)[0]
assert d[pe:pe + 4] == b"PE\0\0"
struct.pack_into("<I", d, pe + 8, 0)  # FileHeader.TimeDateStamp
nsec, = struct.unpack_from("<H", d, pe + 6)
optsz, = struct.unpack_from("<H", d, pe + 20)
opt = pe + 24
magic, = struct.unpack_from("<H", d, opt)
dirs = opt + (112 if magic == 0x20B else 96)  # data directories


def rva_to_off(rva):
    for i in range(nsec):
        vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", d, opt + optsz + 40 * i + 8)
        if va <= rva < va + max(vsize, rawsize):
            return rawptr + rva - va
    raise SystemExit("rva not in a section")


dbg_rva, dbg_size = struct.unpack_from("<II", d, dirs + 6 * 8)  # IMAGE_DIRECTORY_ENTRY_DEBUG
if dbg_rva:
    base = rva_to_off(dbg_rva)
    for k in range(dbg_size // 28):
        e = base + 28 * k
        struct.pack_into("<I", d, e + 4, 0)  # IMAGE_DEBUG_DIRECTORY.TimeDateStamp
        kind, = struct.unpack_from("<I", d, e + 12)
        rawptr, = struct.unpack_from("<I", d, e + 24)
        if kind == 2 and d[rawptr:rawptr + 4] == b"RSDS":  # CodeView: zero the 16-byte GUID
            d[rawptr + 4:rawptr + 20] = bytes(16)
open(path, "wb").write(d)
