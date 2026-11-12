"""Import levels from the user's own copy of Super Meat Boy (personal use only).

Reads imported/smb_static/<name>_semantic.json, written by tools/smbl_extract.py straight from the
game's .smbl level files (all levels, exact tile shapes, obstacles named by the palette's animation
package). Falls back to imported/smb/, the older live-capture pipeline (Frida dump of 8 levels).
These files are gitignored (see .gitignore) and must never be shipped or committed.

Coordinates: the game uses +Y up and 20-unit tiles; the engine uses +Y down and
TILE-pixel cells. Cells map 1:1 (game cell -> engine cell, rows flipped).

Approximations (not decoded from the game yet):
  - hazard collision shapes: a rotated box for lava / fire / lavafalls (sizes in smbl_extract.HAZARD_BOXES),
    else a disc of hazard_radius_tiles() around each lethal object
  - moving saws (level.movers) and platforms (level.platforms) follow the file's keyframe paths, assumed to
    wrap after the last key; platform size is a guess (60 x 20 units x scale)
  - crumble blocks (tile bit 8) break crumble_frames after a touch; the game's break speed is not decoded
  - portals pair up in file order; break blocks, locks and static platforms are plain solid cells
  - wind columns: a fixed-height box above each fan
"""
from __future__ import annotations

import json
import math
import os
from collections import deque

from levels import Level

_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imported")
IMPORT_DIR = os.environ.get("SMBRL_IMPORTED") or next(
    (d for d in (os.path.join(_HERE, "smb_static"), os.path.join(_HERE, "smb")) if os.path.isdir(d)),
    os.path.join(_HERE, "smb"))
SUFFIX = "_semantic.json"


def hazard_radius_tiles(obj: dict) -> float:
    """Approximate lethal radius in tiles; real shapes come from the palette and are not decoded."""
    return max(0.5, 0.9 * (obj.get("scale") or 1.0))


def shape_of(bits: int, left_bits: int = 0, right_bits: int = 0):
    """Collision shape for a solid tile from the game's shape bits (top byte of the tile word), or None = full.

    Measured from captured play (player centres deep inside flagged cells) and matched to the visible
    corner cuts in 1-1: 0x40 = floor-type, 0x80 = ceiling-type, low bits = which side is cut; 0x40 or 0x43
    alone is a half-height block. A ramp next to a half block rises only to half height.
    """
    kind, side = bits & 0xc0, bits & 0x03
    half = lambda b: (b & 0xc0) == 0x40 and (b & 0x03) in (0, 3)
    if kind == 0x40:
        if side == 2:
            return ("F", 0.0, 0.5 if half(right_bits) else 1.0)   # ramp rising to the right
        if side == 1:
            return ("F", 0.5 if half(left_bits) else 1.0, 0.0)    # ramp rising to the left
        return ("F", 0.5, 0.5)                                    # half block
    if kind in (0x80, 0xc0):
        if side == 1:
            return ("C", 1.0, 0.0)   # solid top-left, bottom-right corner cut
        if side == 2:
            return ("C", 0.0, 1.0)   # solid top-right, bottom-left corner cut
        if kind == 0x80:
            return ("C", 0.5, 0.5)   # hanging half block
    return None


def chapters() -> list[dict]:
    """The game's level list (tools/smbl_extract.py -> chapters.json), in game order: dicts with chapter_file
    (Forest, Hospital, SaltFactory, Hell, Rapture, DrFetus, Bandage, ...), kind (normlevel, altlevel = dark
    world, warpNlevel, bosslevel), file, name, par, imported."""
    p = os.path.join(IMPORT_DIR, "chapters.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


_PAR = None


def a_plus_time(name: str | None) -> float | None:
    """The game's par time (seconds) of a level: the `par` its .chapter file gives it (chapters.json), the time
    the game asks for an A+ grade. None: unknown (no chapters.json, or a level without one). Our finish time is
    the wave's updates x the locked step (agent.py on_finish); whether the game's own timer starts on the same
    frame, and whether A+ means at most par exactly, is not verified against the game."""
    global _PAR
    if _PAR is None:
        _PAR = {}
        for c in chapters():
            if c.get("par") is not None and c["file"] not in _PAR:
                _PAR[c["file"]] = float(c["par"])
    return _PAR.get(name) if name else None


def a_plus(name: str | None, seconds: float | None) -> bool | None:
    """Is a finish in `seconds` within the level's A+ time (a_plus_time)? None: unknown."""
    par = a_plus_time(name)
    return None if par is None or seconds is None else bool(float(seconds) <= par)


_GROUPS = {"light": ("normlevel",), "dark": ("altlevel",), "both": ("normlevel", "altlevel"),
           "warp": tuple(f"warp{i}level" for i in range(1, 6)),
           "all": ("normlevel", "altlevel") + tuple(f"warp{i}level" for i in range(1, 6))}


def chapter_levels(chapter: str | None = None, group: str = "light") -> list[str]:
    """Trainable level names of one chapter (file name, title or world number 1-7, case-insensitive) or of
    every chapter (None), in game order. group: light (default), dark, both, warp or all."""
    kinds = _GROUPS[group]
    want = None if chapter is None else chapter.lower()
    world = {str(i + 1): f for i, f in enumerate(("forest", "hospital", "saltfactory", "hell", "rapture",
                                                     "drfetus", "bandage"))}
    want = world.get(want, want)
    out = []
    for c in chapters():
        if c["kind"] not in kinds or not c["imported"]:
            continue
        if want and want not in (c["chapter_file"].lower(), (c["chapter"] or "").lower()):
            continue
        if c["file"] not in out and _has_goal(c["file"]):
            out.append(c["file"])
    return out


def level_title(name: str) -> str:
    """'4-1 Boilermaker (Hell)' for a chapter level, else the file name."""
    c = next((c for c in chapters() if c["file"] == name), None)
    return f"{name} {c['name']} ({c['chapter_file']})" if c and c.get("name") else name


def _has_goal(name: str) -> bool:
    with open(os.path.join(IMPORT_DIR, name + SUFFIX)) as f:
        return any(o["kind"] == "goal" for o in json.load(f)["objects"])


def _order(name: str):
    """Game order: chapter-level (1-1 .. 7-20), then everything else alphabetically."""
    a, _, b = name.partition("-")
    return (0, int(a), int(b)) if a.isdigit() and b.isdigit() else (1, 0, 0)


def available(solvable_only: bool = False) -> list[str]:
    """Imported level names that have a goal (boss fights and editor tests have none): the game's chapter
    levels in game order, then other files; solvable_only also drops levels whose goal the clone cannot reach."""
    if not os.path.isdir(IMPORT_DIR):
        return []
    names = [f[:-len(SUFFIX)] for f in os.listdir(IMPORT_DIR) if f.endswith(SUFFIX)]
    rank = {}
    for i, c in enumerate(chapters()):
        rank.setdefault(c["file"], i)
    names = sorted((n for n in names if _has_goal(n)), key=lambda n: (rank.get(n, len(rank)), _order(n), n))
    return [n for n in names if load(n).meta["goal_reachable"]] if solvable_only else names


def reachable_cells(level: Level) -> dict[tuple[int, int], int]:
    """4-connected flood from spawn -> {cell: steps}. Hazards can be jumped and shaped cells are partly open,
    so neither blocks; crumble blocks break and portals connect their two ends."""
    from engine import TILE as _T
    links = {}
    for a, b in level.portals:
        ca, cb = (int(a[0] // _T), int(a[1] // _T)), (int(b[0] // _T), int(b[1] // _T))
        links.setdefault(ca, []).append(cb)
        links.setdefault(cb, []).append(ca)
    seen, q = {level.spawn_cell: 0}, deque([level.spawn_cell])
    while q:
        c, r = q.popleft()
        for nc, nr in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1), *links.get((c, r), ())):
            if (0 <= nc < level.cols and 0 <= nr < level.rows and (nc, nr) not in seen
                    and ((nc, nr) not in level.solids or (nc, nr) in level.shapes or (nc, nr) in level.crumble)):
                seen[(nc, nr)] = seen[(c, r)] + 1
                q.append((nc, nr))
    return seen


def goal_reachable(level: Level) -> bool:
    return level.goal_cell in reachable_cells(level)


def load(name: str) -> Level:
    path = os.path.join(IMPORT_DIR, name + SUFFIX)
    with open(path) as f:
        d = json.load(f)
    W, H, T = d["W"], d["H"], d["tile"]
    ox, oy = d["origin"]

    def cell(x: float, y: float) -> tuple[int, int]:
        """Game world position -> engine (col, row), row 0 at the top."""
        return int((x - ox) // T), H - 1 - int((y - oy) // T)

    def inside(c: int, r: int) -> bool:
        return 0 <= c < W and 0 <= r < H

    grid = [[" "] * W for _ in range(H)]
    for gy, row in enumerate(d["solid"]):
        for gx, v in enumerate(row):
            if v:
                grid[H - 1 - gy][gx] = "#"

    for x0, x1, y0, y1 in d.get("wind_zones", []):
        c0, r1 = cell(x0, y0)
        c1, r0 = cell(x1, y1)
        for r in range(max(0, r0), min(H - 1, r1) + 1):
            for c in range(max(0, c0), min(W - 1, c1) + 1):
                if grid[r][c] == " ":
                    grid[r][c] = "w"

    bandages, movers, platforms = [], [], []
    for o in d["objects"]:
        c, r = cell(o["x"], o["y"])
        if o["kind"] == "bandage":
            bandages.append((c, r))
        if o["kind"] == "solid" and len(o.get("path", ())) > 1:   # moving platform: level.platforms
            platforms.append(o)
            continue
        if o["kind"] == "solid":   # break block / lock: one tile; platform: three tiles wide
            w = 1 if o.get("name") != "platform" else 3
            for cc in range(c - w // 2, c + w // 2 + 1):
                if inside(cc, r) and grid[r][cc] in (" ", "w"):
                    grid[r][cc] = "#"
            continue
        if o["kind"] != "lethal":
            continue
        if len(o.get("path", ())) > 1:   # moving hazard: handled by the engine (level.movers)
            movers.append(o)
            continue
        if "half" in o:   # rotated box: mark cells whose centre lies inside it
            hx, hy = o["half"]
            ca, sa = math.cos(o["angle"]), math.sin(o["angle"])
            ext = math.hypot(hx, hy)
            c0, r1 = cell(o["x"] - ext, o["y"] - ext)
            c1, r0 = cell(o["x"] + ext, o["y"] + ext)
            for rr in range(max(0, r0), min(H - 1, r1) + 1):
                for cc in range(max(0, c0), min(W - 1, c1) + 1):
                    px = ox + (cc + 0.5) * T - o["x"]
                    py = oy + (H - 1 - rr + 0.5) * T - o["y"]
                    lx, ly = ca * px + sa * py, -sa * px + ca * py
                    if abs(lx) <= hx + T / 2 and abs(ly) <= hy + T / 2 and grid[rr][cc] in (" ", "w"):
                        grid[rr][cc] = "^"
            continue
        rad = hazard_radius_tiles(o)
        k = math.ceil(rad)
        for rr in range(r - k, r + k + 1):
            for cc in range(c - k, c + k + 1):
                if inside(cc, rr) and math.hypot(cc - c, rr - r) <= rad and grid[rr][cc] in (" ", "w"):
                    grid[rr][cc] = "^"

    goals = [o for o in d["objects"] if o["kind"] == "goal"]
    if not goals:
        raise ValueError(f"{name}: no goal object in {path}")
    gc, gr = cell(goals[0]["x"], goals[0]["y"])   # placeholder; the reachable goal is chosen below
    grid[gr][gc] = "G"

    sc, sr = cell(*d["spawn"])
    while inside(sc, sr) and grid[sr][sc] == "#":  # spawn sits on the floor edge; step up into air
        sr -= 1
    grid[sr][sc] = "S"

    level = Level(f"smb:{name}", ["".join(row) for row in grid])
    # the game's own spawn point (player centre, game units) in engine px, used by World.reset
    from engine import TILE as _T
    k = _T / T
    # Live first frame after the level intro (in-game trace): Meat Boy is still at the captured spawn point
    # (a few units above the floor), can jump at once, and faces left (constructor flags 0x50081, bit0).
    level.spawn_center = ((d["spawn"][0] - ox) * k, (H * T - (d["spawn"][1] - oy)) * k)
    level.spawn_grounded = True
    level.spawn_facing = -1
    for o in platforms:
        # platform sprite size is not decoded: 60 x 20 game units x scale (4-2 starts with its top flush
        # under the spawn floor, which bounds the half height by 10)
        s = abs(o.get("scale") or 1.0)
        keys = [(t, (x - ox) * k, (H * T - (y - oy)) * k) for t, x, y in o["path"]]
        level.platforms.append({"keys": keys, "period": keys[-1][0], "hw": 30.0 * s * k, "hh": 10.0 * s * k})
    for o in movers:
        # disc radius: the hazard's own size (boxes: their smaller half-extent); the path is in game units
        r = (min(o["half"]) / T if "half" in o else hazard_radius_tiles(o)) * _T
        keys = [(t, (x - ox) * k, (H * T - (y - oy)) * k) for t, x, y in o["path"]]
        level.movers.append({"keys": keys, "period": keys[-1][0], "r": r, "name": o.get("name")})
    for gx, gy in d.get("crumble", ()):
        level.crumble.add((gx, H - 1 - gy))
    ports = [((o["x"] - ox) * k, (H * T - (o["y"] - oy)) * k) for o in d["objects"] if o["kind"] == "portal"]
    level.portals = list(zip(ports[0::2], ports[1::2]))   # the game pairs portals in file order
    if "shapes" in d:   # exact shapes decoded from the level file
        exact = {(gx, H - 1 - gy): (k, a, b) for gx, gy, k, a, b in d["shapes"]}
        for (c, r), (k, a, b) in exact.items():
            if (c, r) not in level.solids:
                continue
            if k == "F" and a != b:   # a ramp ending at a half block rises only to half height (measured)
                hi = exact.get((c + (1 if b > a else -1), r))
                if hi == ("F", 0.5, 0.5):
                    a, b = min(a, 0.5), min(b, 0.5)
            level.shapes[(c, r)] = (k, a, b)
    bits = d.get("shape_bits") if "shapes" not in d else None
    if bits:
        for gy, row in enumerate(bits):
            for gx, b in enumerate(row):
                if b and (gx, H - 1 - gy) in level.solids:
                    sh = shape_of(b, row[gx - 1] if gx > 0 else 0, row[gx + 1] if gx + 1 < W else 0)
                    if sh:
                        level.shapes[(gx, H - 1 - gy)] = sh
    level.meta = {"palette": d.get("palette"), "bandages": bandages, "source": path,
                  "wind_cells": len(level.wind), "hazard_cells": len(level.hazards), "movers": len(level.movers), "shape_cells": len(level.shapes)}
    # several goal objects: touching any ends the level; the clone uses the nearest one reachable from spawn
    reach = reachable_cells(level)
    cells = [cell(o["x"], o["y"]) for o in goals]
    ok = [c for c in cells if c in reach]
    level.goal_cell = min(ok, key=reach.get) if ok else cells[0]
    level.meta["goal_reachable"] = bool(ok)
    level.meta.update(goals=len(goals), crumble=len(level.crumble), portals=len(level.portals),
                      platforms=len(level.platforms))
    return level


if __name__ == "__main__":
    for n in available():
        lv = load(n)
        print(f"{lv.name:10s} {lv.cols}x{lv.rows} solids={len(lv.solids)} hazards={len(lv.hazards)} "
              f"wind={len(lv.wind)} spawn={lv.spawn_cell} goal={lv.goal_cell} palette={lv.meta['palette']} "
              f"goal_reachable={lv.meta['goal_reachable']}")
