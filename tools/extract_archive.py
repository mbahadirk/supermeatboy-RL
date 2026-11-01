"""Extract files from Super Meat Boy's gamedata.dat archive.

Clean-room parser derived from reverse-engineering the archive mount function.
On-disk format (little-endian):

    u32 N1                         # group count (29)
    N1 x { u32 groupIdx; u32 nameOffA }
    u32 N2                         # file count (2770)
    N2 x { u32 offset; u32 size; u32 flag }   # offset/size are ABSOLUTE in file
    u32 sizeA                      # group-name blob size
    u32 sizeB                      # file-name blob size
    byte[sizeA]                    # NUL-terminated group names
    byte[sizeB]                    # NUL-terminated file names (parallel to records)
    ... payload (file bytes at their absolute offsets) ...

OUTPUT: extracted originals are copyrighted; this writes them OUTSIDE the repo
(default D:\\SMB-RL\\imported) and they are gitignored. For personal/educational use.

Usage:
    python smbrl/tools/extract_archive.py <gamedata.dat> [--out DIR] [--prefix Levels/]
"""
from __future__ import annotations
import argparse
import os
import struct


def parse_archive(data: bytes):
    u32 = lambda o: struct.unpack_from("<I", data, o)[0]
    off = 0
    n1 = u32(off); off += 4
    off += n1 * 8                      # skip group (idx, nameOffA) pairs
    n2 = u32(off); off += 4
    rec_off = off
    records = [(u32(rec_off + i * 12),
                u32(rec_off + i * 12 + 4),
                u32(rec_off + i * 12 + 8)) for i in range(n2)]
    off += n2 * 12
    size_a = u32(off); off += 4
    size_b = u32(off); off += 4
    off += size_a                      # skip group-name blob
    name_blob = data[off: off + size_b]
    # split NUL-terminated names in order (parallel to records)
    names, start = [], 0
    for _ in range(n2):
        end = name_blob.find(b"\x00", start)
        if end < 0:
            break
        names.append(name_blob[start:end].decode("latin1"))
        start = end + 1
    return n1, n2, records, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--out", default=r"D:\SMB-RL\imported")
    ap.add_argument("--prefix", default="Levels/")
    args = ap.parse_args()

    data = open(args.archive, "rb").read()
    n1, n2, records, names = parse_archive(data)
    print(f"groups={n1} files={n2} names_parsed={len(names)}")

    sel = [(i, nm) for i, nm in enumerate(names) if nm.startswith(args.prefix)]
    print(f"matching '{args.prefix}': {len(sel)}")
    os.makedirs(args.out, exist_ok=True)
    mags = {}
    written = 0
    for i, nm in sel:
        off, size, flag = records[i]
        blob = data[off: off + size]
        mags[blob[:4]] = mags.get(blob[:4], 0) + 1
        # safe filename: strip the group prefix, keep the rest
        rel = nm[len(args.prefix):] if nm.startswith(args.prefix) else nm
        rel = rel.replace("/", "_")
        sub = args.prefix.rstrip("/").replace("/", "_")
        d = os.path.join(args.out, sub)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, rel), "wb") as f:
            f.write(blob)
        written += 1
    print(f"written={written} -> {args.out}")
    print("first-4-byte signatures:", {k.decode('latin1', 'replace'): v for k, v in mags.items()})


if __name__ == "__main__":
    main()
