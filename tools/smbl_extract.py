"""Static extractor for Super Meat Boy level files (.smbl, versions 4-7) -> clone levels.

Reverse-engineered from the game's loader (FUN_004e9020 / FUN_00468030), its tile shape pass
(FUN_004cccd0 -> FUN_004c9fe0 / FUN_004ca450 / FUN_004ca9a0) and checked against live in-game
captures of 8 levels (collision layer 100% identical, tile shapes 249/249).

Layout (little-endian):
    0x000  "20LT"/"10LT", u32 version (7), u8 255, char[256] palette name ("hell.lp")
           (versions 4-5 without the magic: every later offset is 4 lower)
    0x109  f32 spawn x, f32 spawn y            (player start, game units, y up)
    0x111  u16 W, u16 H                        (tiles of 20 units; the map is centred on (0, 0))
    0x115  160 bytes of level settings (lighting / colours), not used here
    0x1b5  u32 n0, n1, n2                      record counts of the three tile layers
    0x1c1  layer 0 (collision), layer 1, layer 2: run-length u32 records, row-major from the bottom
           row: cell value = rec & 0x3fffff, repeated rec >> 22 times
           cell value: bits 0-7 tile id (solid when 0 < id < 100), bit 8 crumble block (with ids 64, 96-100),
           12-13 rotation (90 deg steps),
           14-15 flip x/y, 18-19 shape: 0 block, 1 half block, 2/3 45-degree slope
    then   5 x u32 (unknown), then 50-byte object records until the tail:
           +12 f32 x, +16 f32 y, +24 f32 scale x, +28 f32 scale y, +32 f32 angle, +36 u16 type
           type high byte 0x3X = obstacle from the palette's obstacles .am (low byte = animation index),
           other high bytes = decorative set pieces
    tail   per-object extra data (e.g. movement paths), not decoded yet

Usage (from the project root; reads the files extracted by extract_archive.py):
    python smbrl/tools/smbl_extract.py smbrl/imported/smbl_raw smbrl/imported/smb_static
Writes <level>_semantic.json files in the format levels_smb.py loads. Personal use only (gitignored).
"""
from __future__ import annotations

import os
import sys
import json
import math
import struct
from collections import Counter

TILE_UNITS = 20.0
NUMTILES = 100

# palette -> obstacle animation package (from LevelPalettes/*.lp "obstacles { animationpackage=... }")
PALETTE_OBSTACLES = {
    "forest.lp": "obstacles.am", "hospital.lp": "hobstacles.am", "factory.lp": "sfobstacles.am",
    "hell.lp": "hellobstacles.am", "rapture.lp": "robstacles.am", "end.lp": "eobstacles.am",
    "retro1.lp": "retroobjects.am", "glitch.lp": "retroobjects.am", "4bit1.lp": "4bitobjects.am",
    "gb.lp": "gbbjects.am",
}

# obstacle animation name -> clone kind. Verified against the game's type table (goal 0x06, bandage
# 0x0c, saws 0x121) and live deaths; everything not listed (effects, ghosts, boss markers, pickups'
# break/ghost frames, decorations) has no collision in the clone.
LETHAL_PREFIXES = ("saw", "needle", "spike", "lava", "fire", "blood", "forest - river", "salt", "bigsalt", "missile",
                   "missle", "littledemon", "buttholekiller", "buttholeammo", "spinner", "movin", "lazeron",
                   "hominglazeron", "bullet", "thwamp", "maggots", "jet fire", "superfuck", "shock", "ammo1",
                   "ammo2", "orb")
SOLID_NAMES = {"breakblock1", "lock", "platform", "onoffappear"}
WIND_NAMES = {"fan"}

# lethal extent (half sizes in game units at scale 1, along the object's own axes) for hazard families
# that are not round. Sprite bounds are not decoded from the .am animations yet: these come from how the
# level files tile them (hell lava sprites repeat every ~120 units x scale, lavafalls every ~33-45).
# Everything else (saws, needles, salt, ...) is a disc of 0.9 tiles x scale.
HAZARD_BOXES = (("lava", 60.0, 14.0), ("forest - river", 22.0, 14.0), ("fire - large", 20.0, 20.0),
                ("fire", 14.0, 14.0), ("spike", 10.0, 10.0))


def hazard_box(name: str):
    n = name.lower()
    return next(((hx, hy) for p, hx, hy in HAZARD_BOXES if n.startswith(p)), None)


def classify(name: str | None) -> str:
    if name is None:
        return "unknown"
    n = name.lower()
    if n == "levelend1":
        return "goal"
    if n == "bandaid":
        return "bandage"
    if n in WIND_NAMES:
        return "wind"
    if n == "portalidle":
        return "portal"
    if n in SOLID_NAMES:
        return "solid"
    if n.endswith("ghost") or n.endswith("break") or n.endswith("explosion"):
        return "none"
    if n.startswith(LETHAL_PREFIXES):
        return "lethal"
    return "none"


def am_names(path: str) -> list[str]:
    """Animation names of an .am package ("101F" header, u16 count at 6, NUL-separated names at 0x10)."""
    b = open(path, "rb").read()
    n = struct.unpack_from("<H", b, 6)[0]
    off, names = 0x10, []
    for _ in range(n):
        e = b.index(b"\x00", off)
        names.append(b[off:e].decode("latin1"))
        off = e + 1
    return names


ANGLES = (0.0, math.pi / 2, math.pi, -math.pi / 2)


def tile_shape(v: int):
    """Clone shape (kind, a, b) of a solid tile, or None for a full block. The game derives an orientation
    by flipping then rotating a reference vector (FUN_004ca450 half blocks, FUN_004ca9a0 slopes)."""
    typ = (v >> 18) & 3
    if typ == 0:
        return None
    rot, flip = (v >> 12) & 3, (v >> 14) & 3
    sx = -1.0 if flip in (1, 3) else 1.0
    sy = -1.0 if flip in (2, 3) else 1.0
    x, y = (0.0, 0.25) if typ == 1 else (-0.5, 0.5)
    x, y = x * sx, y * sy
    c, s = math.cos(ANGLES[rot]), math.sin(ANGLES[rot])
    x, y = round(c * x - s * y, 6), round(s * x + c * y, 6)
    if typ == 1:   # half block: the solid half faces the reference direction (y up)
        o = (2 + (x > 0)) if y == 0 else (0 if y > 0 else 1)
        return [("C", 0.5, 0.5), ("F", 0.5, 0.5), ("XL", 0.5, 0.5), ("XR", 0.5, 0.5)][o]
    # slope: the solid corner (y up): 0 bottom-left, 1 bottom-right, 2 top-left, 3 top-right
    o = int(x > 0) if y < 0 else (3 if x > 0 else 2)
    return [("F", 1.0, 0.0), ("F", 0.0, 1.0), ("C", 1.0, 0.0), ("C", 0.0, 1.0)][o]


def parse(path: str) -> dict:
    b = open(path, "rb").read()
    # "TL01"/"TL02" magic then u32 version (5-7), or (older files) the version alone (4-5)
    hdr = 8 if b[:4] in (b"20LT", b"10LT") else 4
    version = struct.unpack_from("<I", b, hdr - 4)[0]
    if not 4 <= version <= 7:
        raise ValueError(f"unsupported version {version}")
    pal = hdr + 1 + b[hdr] + 1          # u8 length, then the name and its NUL (always 256 bytes so far)
    palette = b[hdr + 1:pal].split(b"\x00")[0].decode("latin1")
    sx, sy = struct.unpack_from("<ff", b, pal)
    W, H = struct.unpack_from("<HH", b, pal + 8)
    base = pal + 8 + 164 - 0x1b5         # the rest is laid out like version 7 at 0x1b5
    counts = struct.unpack_from("<III", b, base + 0x1b5)
    off, layers = base + 0x1c1, []
    for n in counts:
        cells = []
        for rec in struct.unpack_from("<%dI" % n, b, off):
            cells += [rec & 0x3fffff] * (rec >> 22)
        off += 4 * n
        if len(cells) != W * H:
            raise ValueError(f"layer size {len(cells)} != {W}x{H}")
        layers.append(cells)
    objs, s = [], off + 20
    rec = 50 if hdr == 8 else 52   # magic-less (version 4-5) files: 2 more bytes per record
    while s + rec <= len(b):
        x, y, _z, scx, scy, ang = struct.unpack_from("<6f", b, s + 12)
        # the record list has no count we know of: it ends where the per-object tail data starts, which
        # never parses as six sane floats
        if not (all(math.isfinite(v) and abs(v) < 1e5 for v in (x, y, _z, scx, scy, ang))
                and 1e-4 < abs(scx) < 1e4 and 1e-4 < abs(scy) < 1e4
                and struct.unpack_from("<H", b, s + 36)[0] >> 12 in (0, 1, 3, 4)):
            break
        objs.append({"type": struct.unpack_from("<H", b, s + 36)[0], "x": x, "y": y,
                     "scale": scx, "scale_y": scy, "angle": ang})
        s += rec
    paths = parse_paths(b, s, len(objs))
    for i, keys in (paths or {}).items():
        objs[i]["path"] = keys
    return {"palette": palette, "spawn": [sx, sy], "W": W, "H": H, "layers": layers, "objects": objs,
            "tail_bytes": len(b) - s, "paths_decoded": paths is not None}


def _paths_at(b: bytes, p: int, end: int, n_objs: int):
    """Movement path list at p: u32 count, then per path u32 object index, u8 key count and keys of
    7 f32 (x, y, z, scale x, scale y, angle, time); it must end exactly at end."""
    if p + 4 > end:
        return None
    count = struct.unpack_from("<I", b, p)[0]
    if count > n_objs:
        return None
    p += 4
    out = {}
    for _ in range(count):
        if p + 5 > end:
            return None
        idx, n = struct.unpack_from("<IB", b, p)
        p += 5
        if idx >= n_objs or n == 0 or idx in out or p + 28 * n > end:
            return None
        keys, last_t = [], -1.0
        for k in range(n):
            x, y, _z, _sx, _sy, _a, t = struct.unpack_from("<7f", b, p + 28 * k)
            if not all(math.isfinite(v) and abs(v) < 1e5 for v in (x, y, t)) or t < 0:
                return None
            last_t = max(t, last_t)   # a few editor keys have time 0 mid-path: keep the order
            keys.append([last_t, x, y])
        p += 28 * n
        out[idx] = keys
    return out if p == end else None


def parse_paths(b: bytes, start: int, n_objs: int):
    """The tail (lights and other per-level data, then the movement paths) ends with the path list,
    followed by "TEND" in newer files; the path list is the suffix that parses exactly. None when not found."""
    end = len(b) - 4 if b[-4:] == b"DNET" else len(b)
    for p in range(end - 4, start - 1, -1):
        r = _paths_at(b, p, end, n_objs)
        if r is not None:
            return r
    return None


def to_semantic(name: str, lv: dict, names: list[str] | None) -> dict:
    W, H = lv["W"], lv["H"]
    col = lv["layers"][0]
    # tiles 1..99 are solid; tiles flagged with bit 8 (ids 64, 96, 97, 99, 100) are the special tiles the
    # game's tile pass (FUN_004cccd0) gives a per-level variant: crumble blocks. Treating them as breakable
    # opens 126 chapter levels whose goal is otherwise sealed in.
    def crumbles(v):
        return bool(v & 0x100) and 0 < (v & 0xff) <= NUMTILES
    solid = [[1 if 0 < (col[gy * W + gx] & 0xff) < NUMTILES or crumbles(col[gy * W + gx]) else 0
              for gx in range(W)] for gy in range(H)]
    crumble = [[gx, gy] for gy in range(H) for gx in range(W) if crumbles(col[gy * W + gx])]
    shapes = []
    for gy in range(H):
        for gx in range(W):
            v = col[gy * W + gx]
            if solid[gy][gx]:
                sh = tile_shape(v)
                if sh:
                    shapes.append([gx, gy, *sh])
    objects, kinds, wind = [], Counter(), []
    for o in lv["objects"]:
        if (o["type"] >> 12) != 0x3:   # decorative set piece
            continue
        idx = o["type"] & 0xff
        nm = names[idx] if names and idx < len(names) else None
        k = classify(nm)
        kinds[k] += 1
        if k == "none":
            continue
        ob = {"type": idx, "name": nm, "kind": k, "x": o["x"], "y": o["y"], "angle": o["angle"],
              "scale": o["scale"], "flags": o["type"] >> 8}
        if len(o.get("path", ())) > 1:
            ob["path"] = o["path"]
        box = hazard_box(nm) if k == "lethal" else None
        if box:
            ob["half"] = [box[0] * abs(o["scale"]), box[1] * abs(o["scale_y"] or o["scale"])]
        objects.append(ob)
        if k == "wind":   # fan updraft column (370 units, from the hospital fan's object data)
            half = 20.0 * max(0.5, o["scale"])
            wind.append([o["x"] - half, o["x"] + half, o["y"], o["y"] + 370.0])
    return {"name": name, "palette": lv["palette"], "W": W, "H": H, "tile": TILE_UNITS,
            "origin": [-W * TILE_UNITS / 2, -H * TILE_UNITS / 2], "solid": solid, "shapes": shapes,
            "crumble": crumble,
            "spawn": lv["spawn"], "objects": objects, "wind_zones": wind, "source": "static",
            "object_kinds": dict(kinds), "tail_bytes": lv["tail_bytes"], "paths_decoded": lv["paths_decoded"]}


LEVEL_KINDS = ("normlevel", "altlevel", "warp1level", "warp2level", "warp3level", "warp4level", "warp5level",
               "bosslevel")


# chapter files in the game's world order (titles are not unique: Bandage.chapter is also "Factory");
# Debug and demo only repeat Forest levels
CHAPTER_ORDER = ("Forest", "Hospital", "SaltFactory", "Hell", "Rapture", "DrFetus", "Bandage", "Internets")
CHAPTER_SKIP = ("Debug", "demo")


def chapters(lvl_dir: str) -> list[dict]:
    r"""Level order from the game's .chapter files (chapter { name } then normlevel / altlevel (dark world) /
    warpNlevel / bosslevel { file, name, par, id } blocks), in CHAPTER_ORDER then the rest by file name."""
    import re
    out = []
    for f in os.listdir(lvl_dir):
        if not f.endswith(".chapter"):
            continue
        text = open(os.path.join(lvl_dir, f), "rb").read().decode("latin1")
        title = re.search(r'chapter\s*\{[^}]*?name\s*=\s*"([^"]*)"', text)
        for m in re.finditer(r"(\w+)\s*\{([^}]*)\}", text):
            kind, body = m.group(1), m.group(2)
            fm = re.search(r'file\s*=\s*"([^"]+)"', body)
            if kind not in LEVEL_KINDS or not fm:
                continue
            get = lambda k: (re.search(k + r'\s*=\s*"?([^";]*)"?;', body) or [None, None])[1]
            par = get("par")
            out.append({"chapter": title.group(1) if title else f[:-8], "chapter_file": f[:-8], "kind": kind,
                        "file": fm.group(1), "name": get("name"), "par": float(par) if par else None})

    kind_rank = {k: i for i, k in enumerate(LEVEL_KINDS)}
    order = {f: i for i, f in enumerate(CHAPTER_ORDER)}
    out = [c for c in out if c["chapter_file"] not in CHAPTER_SKIP]
    return sorted(out, key=lambda c: (order.get(c["chapter_file"], len(order)), c["chapter_file"].lower(),
                                      kind_rank[c["kind"]]))


def main(src: str, dst: str):
    lvl_dir, pal_dir = os.path.join(src, "Levels"), os.path.join(src, "palettes")
    os.makedirs(dst, exist_ok=True)
    am_cache, report = {}, Counter()
    for f in sorted(os.listdir(lvl_dir)):
        if not f.endswith(".smbl"):
            continue
        name = f[:-5]
        try:
            lv = parse(os.path.join(lvl_dir, f))
        except (ValueError, struct.error) as e:
            report[f"skipped: {e}"] += 1
            continue
        pkg = PALETTE_OBSTACLES.get(lv["palette"])
        if pkg not in am_cache:
            p = os.path.join(pal_dir, "pc_LevelPalettes_" + pkg) if pkg else None
            am_cache[pkg] = am_names(p) if p and os.path.exists(p) else None
        sem = to_semantic(name, lv, am_cache[pkg])
        has_goal = any(o["kind"] == "goal" for o in sem["objects"])
        report["with goal" if has_goal else "no goal object"] += 1
        with open(os.path.join(dst, name + "_semantic.json"), "w") as fh:
            json.dump(sem, fh)
    order = chapters(lvl_dir)
    for c in order:
        c["imported"] = os.path.exists(os.path.join(dst, c["file"] + "_semantic.json"))
    with open(os.path.join(dst, "chapters.json"), "w") as fh:
        json.dump(order, fh, indent=0)
    print(dict(report))
    print(f"{len(order)} chapter levels, {sum(c['imported'] for c in order)} imported -> chapters.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
