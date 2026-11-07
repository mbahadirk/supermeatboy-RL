"""Where Meat Boy is supported (standing on something or holding a wall), from the level geometry.

The game's flag bit 0x2 (wm.features.AIR) is NOT a reliable "airborne" bit: it is also set while standing.
So support is derived from the tiles around the body (BODY = 6, realpath.BODY: the body is a 12 x 12 box):

- on the ground: solid at GROUND_PROBE = BODY + 1.5 units below the centre, under either foot (x - FOOT_DX or
  x + FOOT_DX, FOOT_DX = BODY - 0.5: half a unit inside the body's edges), in the tile map or a moving
  platform. On a 45-degree slope the body rests on one bottom corner; a probe 1.5 below and 0.5 inside the
  corner is 1 unit inside the slope (the brief's 1 below / 1 inside lies exactly on the slope's surface);
- on a wall: solid WALL_PROBE = BODY + 1 units beside the centre (1 unit past the body's edge) at the body's
  mid height; side -1 = wall on the left, +1 = on the right (left wins when both).

"Solid" is tested per point with the tile's shape, not per cell: a half block is the half of its cell given by
its rotation / flip bits (reference: the upper half, flipped then rotated as the game does for slopes,
realpath.slope_open_corner; checked on recorded standing heights: agents stand 6.01 units above the half's
surface in every orientation seen), a 45-degree slope is the triangle opposite its open corner. Treating half
blocks and slopes as whole cells called 10 to 40% of the rule's "ground" frames ground while the agent was
in the air above the open part of such a cell.

Special tiles (crumble / lock / switch) are solid only while their word is the level's start word
(game.level.Level.apply_tiles). Outside the map is not solid (nothing to stand on there; leaving the level is
safety's business).

Checked on recorded play: `python -m plan.geometry <wm_data> --levels 1-5 1-13` (measure() below); the
numbers are in the docstring of plan/planner.py.
"""
from __future__ import annotations

import numpy as np

from game.level import tile_solid, TILE
from realpath import BODY, slope_open_corner, tile_classes

GROUND_PROBE = BODY + 1.5
FOOT_DX = BODY - 0.5
WALL_PROBE = BODY + 1.0

SHAPE_OPEN, SHAPE_FULL, SHAPE_HALF, SHAPE_SLOPE = 0, 1, 2, 3
_DC, _DR = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])     # box_overlap: the 2 x 2 cells a box can touch
SUB = TILE / 2                                                 # travel: half blocks are 2 of 4 such squares


def solid_now(words0: np.ndarray, words: np.ndarray) -> np.ndarray:
    """Solid cells of the live tile words (H x W, row 0 = bottom): game.level.Level.apply_tiles' rule."""
    solid = tile_solid(words.astype(np.uint32))
    w0 = words0.astype(np.uint32)
    special = ((w0 >> 8) & 1).astype(bool) & tile_solid(w0)
    solid[special] = (words.astype(np.uint32) == w0)[special]
    return solid


def half_normal(words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per cell, the direction (nx, ny) from the cell centre to its solid half for half blocks: the upper half
    (0, +1) flipped (flip bits 14-15: 1 / 3 mirror x, 2 / 3 mirror y), then rotated by the rotation bits 12-13
    (as realpath.slope_open_corner)."""
    t = words.astype(np.uint32)
    rot, flip = (t >> 12) & 3, (t >> 14) & 3
    x = np.zeros(t.shape, np.int64)
    y = np.where((flip == 2) | (flip == 3), -1, 1)
    xr = np.select([rot == 0, rot == 1, rot == 2, rot == 3], [x, -y, -x, y])
    yr = np.select([rot == 0, rot == 1, rot == 2, rot == 3], [y, x, -y, -x])
    return xr.astype(np.int8), yr.astype(np.int8)


class Terrain:
    """The live tile map as point-solid shapes. Rebuild (or `update`) when tiles change. slopes=False: slope
    cells are open (the planner's collision guard: real agents are recorded inside some slope cells' modelled
    triangles, plan/planner.py)."""

    def __init__(self, words0: np.ndarray, ox: float, oy: float, words: np.ndarray | None = None,
                 slopes: bool = True):
        self.slopes = bool(slopes)
        self.words0 = np.asarray(words0, np.uint32)
        self.ox, self.oy = float(ox), float(oy)
        self.H, self.W = self.words0.shape
        self.update(self.words0 if words is None else words)

    def update(self, words: np.ndarray):
        w = np.asarray(words, np.uint32)
        solid = solid_now(self.words0, w)
        k = tile_classes(w)
        shape = np.where(solid, SHAPE_FULL, SHAPE_OPEN).astype(np.int8)
        shape[solid & k["half"]] = SHAPE_HALF
        shape[solid & k["slope"]] = SHAPE_SLOPE if self.slopes else SHAPE_OPEN
        self.shape = shape
        self.nx, self.ny = half_normal(w)
        self.open_corner = slope_open_corner(w)
        self._line_form()

    def point_solid(self, x, y) -> np.ndarray:
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        fc, fr = (x - self.ox) / TILE, (y - self.oy) / TILE
        c, r = np.floor(fc).astype(np.int64), np.floor(fr).astype(np.int64)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        out = np.zeros(x.shape, bool)
        if not ok.any():
            return out
        cc, rr = c[ok], r[ok]
        u, v = fc[ok] - cc, fr[ok] - rr                     # 0..1 inside the cell, v up
        sh = self.shape[rr, cc]
        nx, ny = self.nx[rr, cc], self.ny[rr, cc]
        half = (nx * (u - 0.5) + ny * (v - 0.5)) >= 0       # on the solid half's side of the centre line
        oc = self.open_corner[rr, cc]
        slope = np.select([oc == 0, oc == 1, oc == 2, oc == 3], [u + v >= 1, v >= u, u >= v, u + v <= 1], False)
        out[ok] = (sh == SHAPE_FULL) | ((sh == SHAPE_HALF) & half) | ((sh == SHAPE_SLOPE) & slope)
        return out

    def box_overlap(self, x, y, half) -> np.ndarray:
        """Does a square box (centre x, y, half size `half`, < TILE / 2) overlap a solid shape with positive
        area? Exact for the shapes point_solid uses: each is its cell cut by one line (a half block: its centre
        line, a slope: its diagonal, a full block: none; `lin`: solid where a u + b v + c > 0, u, v 0..1 in the
        cell), so the box's part inside a cell overlaps the solid when its corner furthest into the solid side
        is strictly inside. Touching is not overlapping. Outside the map: not solid."""
        x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
        h = np.broadcast_to(np.asarray(half, np.float64), x.shape)
        shape = x.shape
        x, y, h = x.ravel(), y.ravel(), h.ravel()
        x0, x1, y0, y1 = x - h, x + h, y - h, y + h
        c = np.floor((x0[:, None] - self.ox) / TILE).astype(np.int64) + _DC      # the 2 x 2 cells a box touches
        r = np.floor((y0[:, None] - self.oy) / TILE).astype(np.int64) + _DR
        idx = np.clip(r + 1, 0, self.H + 1) * (self.W + 2) + np.clip(c + 1, 0, self.W + 1)   # border: open
        ua = np.maximum((x0[:, None] - self.ox) / TILE - c, 0.0)
        ub = np.minimum((x1[:, None] - self.ox) / TILE - c, 1.0)      # the box inside each cell, 0..1
        va = np.maximum((y0[:, None] - self.oy) / TILE - r, 0.0)
        vb = np.minimum((y1[:, None] - self.oy) / TILE - r, 1.0)
        L = self.lin.reshape(-1, 3)[idx]
        A, B = L[..., 0], L[..., 1]
        val = np.where(A > 0, A * ub, A * ua) + np.where(B > 0, B * vb, B * va) + L[..., 2]
        return ((val > 0) & (ub > ua) & (vb > va)).any(1).reshape(shape)

    def _line_form(self):
        """`lin` (H + 2, W + 2, 3): per cell (a, b, c), solid where a u + b v + c > 0; a 1-cell open border.
        Without slopes (rects_only) also travel's surface tables."""
        sh, nx, ny, oc = self.shape, self.nx.astype(np.float64), self.ny.astype(np.float64), self.open_corner
        lin = np.zeros((self.H + 2, self.W + 2, 3))
        lin[..., 2] = -1.0
        L = lin[1:-1, 1:-1]
        L[sh == SHAPE_FULL] = (0.0, 0.0, 1.0)
        hm = sh == SHAPE_HALF
        L[hm] = np.stack([nx[hm], ny[hm], -0.5 * (nx[hm] + ny[hm])], -1)
        for k, abc in enumerate(((1.0, 1.0, -1.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, -1.0, 1.0))):
            L[(sh == SHAPE_SLOPE) & (oc == k)] = abc
        self.lin = lin
        self.rects_only = not (sh == SHAPE_SLOPE).any()
        if self.rects_only:
            self._surfaces()

    def _surfaces(self):
        """travel's tables. Full and half blocks are unions of SUB x SUB squares (half a tile), so the map is a
        grid G of those (rows along y). For each direction of motion, a table over the grid (along x
        transposed): the nearest solid face a box's leading edge meets moving that way from each square,
        cumulated along the motion, as a "falling along -axis" lookup in a mirrored frame for the + directions.
        Each table has one -inf row before the map (nothing there) and -inf columns beside it."""
        sh, nx, ny = self.shape, self.nx, self.ny
        hm = sh == SHAPE_HALF
        full = sh == SHAPE_FULL
        G = np.zeros((2 * self.H, 2 * self.W), bool)                        # [row, col] of half-tile squares
        for dr in (0, 1):
            for dc in (0, 1):
                part = hm & np.where(ny != 0, ny == (2 * dr - 1), nx == (2 * dc - 1))   # the half's two squares
                G[dr::2, dc::2] = full | part
        tabs, origins = [], []
        for grid, oP, oQ in ((G, self.oy, self.ox), (G.T, self.ox, self.oy)):   # along y, along x
            N = grid.shape[0]
            for mirror in (False, True):                                    # -axis, +axis (mirrored)
                g = grid[::-1] if mirror else grid
                o = -(oP + SUB * N) if mirror else oP                       # the mirrored frame's origin
                top = np.where(g, o + SUB * (np.arange(N)[:, None] + 1), -np.inf)
                t = np.full((N + 1, g.shape[1] + 2), -np.inf)
                t[1:, 1:-1] = np.maximum.accumulate(top, axis=0)            # the highest face at or below
                tabs.append(t)
                origins.append((o, oQ))
        self._tab_flat = np.concatenate([t.ravel() for t in tabs])
        base = np.cumsum([0] + [t.size for t in tabs[:-1]])
        # per table: origin along, origin across, last row, last column, offset, row length (floats: one gather)
        self._tab_par = np.array([(o, q, t.shape[0] - 1, t.shape[1] - 1, b, t.shape[1])
                                  for (o, q), t, b in zip(origins, tabs, base)], np.float64)

    def travel(self, px, py, d, along_y, half: float) -> np.ndarray:
        """How far (0..1) a box (centre px, py, half size `half`) can move by d along one axis (x, or y where
        along_y) before it overlaps a solid shape with positive area: the nearest solid face across the box's
        path at or beyond its leading edge; a solid the box already overlaps that way stops it at once (0). No
        move (d 0): 1. Exact for full and half blocks (`_surfaces`); only for a map without slopes
        (rects_only)."""
        assert self.rects_only, "travel: a map with slopes"
        ay, d = np.asarray(along_y, bool), np.asarray(d, np.float64)
        up = d > 0
        par = np.take(self._tab_par, np.where(ay, 0, 2) + up, axis=0)     # table y-, y+, x-, x+: its parameters
        oP, oQ, top_i, last_j, base, nc = par.T
        P = np.where(up, -1.0, 1.0) * np.where(ay, py, px)                  # along (mirrored for + moves)
        lead = P - half
        i = np.minimum(np.maximum(np.ceil((lead - oP) / SUB), 0.0), top_i)  # the square below the edge, +1
        u = (np.where(ay, px, py) - oQ) / SUB                               # across, in squares
        hs = half / SUB
        j0, j1 = np.floor(u - hs), np.ceil(u + hs) - 1                       # the columns the box overlaps
        js = (j0, j1) if half <= SUB / 2 else (j0, j1, np.floor(0.5 * (j0 + j1)))   # 2, or 3 for a wider box
        j = np.minimum(np.maximum(np.stack(js) + 1, 0.0), last_j)            # -inf columns beside the map
        face = np.take(self._tab_flat, (base + i * nc + j).astype(np.int64)).max(0)
        t = np.minimum(np.maximum((lead - face) / np.maximum(np.abs(d), 1e-12), 0.0), 1.0)
        return np.where(d == 0, 1.0, t)

    def opened(self, cells) -> "Terrain":
        """A copy with these cells [(row, col)] open (lock tiles an agent's keys opened)."""
        t = Terrain.__new__(Terrain)
        t.words0, t.ox, t.oy, t.H, t.W, t.slopes = self.words0, self.ox, self.oy, self.H, self.W, self.slopes
        t.shape, t.nx, t.ny, t.open_corner = self.shape.copy(), self.nx, self.ny, self.open_corner
        for r, c in cells:
            if 0 <= r < self.H and 0 <= c < self.W:
                t.shape[r, c] = SHAPE_OPEN
        t._line_form()
        return t

    def support(self, x, y, plats=None):
        """(ground bool, wall_side int8: -1 left, +1 right, 0 none) for positions x, y (any shape). plats:
        moving platform boxes (x, y, half w, half h, ...) (game.level.Level.plats), optional."""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        yb = y - GROUND_PROBE
        ground = self.point_solid(x - FOOT_DX, yb) | self.point_solid(x + FOOT_DX, yb)
        left = self.point_solid(x - WALL_PROBE, y)
        right = self.point_solid(x + WALL_PROBE, y)
        if plats:
            ground |= _in_plats(plats, x - FOOT_DX, yb) | _in_plats(plats, x + FOOT_DX, yb)
            left |= _in_plats(plats, x - WALL_PROBE, y)
            right |= _in_plats(plats, x + WALL_PROBE, y)
        side = np.where(left, -1, np.where(right, 1, 0)).astype(np.int8)
        return ground, side

    def supported(self, x, y, plats=None) -> np.ndarray:
        """On the ground or on a wall."""
        g, s = self.support(x, y, plats)
        return g | (s != 0)


def _in_plats(plats, x, y) -> np.ndarray:
    out = np.zeros(np.shape(x), bool)
    for px, py, hw, hh, *_ in plats or ():
        out |= (np.abs(x - px) <= hw) & (np.abs(y - py) <= hh)
    return out


# ---- check on recorded play -------------------------------------------------------------------------------
def measure(d: str, levels=("1-5", "1-13"), shards: int = 4, log=print) -> dict:
    """Agreement of the rule with the recorded movement states, vy and the flag bits, on the first `shards`
    waves of each level in a wm_data folder (tile changes applied as they came). Returns per level the counts."""
    import glob
    import gzip
    import json
    import os
    from collections import defaultdict
    from wm.buffer import load_level
    from wm.features import AIR, WALL
    out = {}
    for name in levels:
        words0, meta = load_level(d, name)
        words0 = words0.astype(np.uint32)
        ox, oy = float(meta["ox"]), float(meta["oy"])
        tot = defaultdict(int)
        per_state = defaultdict(lambda: defaultdict(int))
        n = 0
        for p in sorted(glob.glob(os.path.join(d, "wave_*.npz"))):
            z = np.load(p)
            if json.loads(str(z["meta"])).get("level") != name:
                continue
            n += 1
            if n > shards:
                break
            rows = np.sort(z["rows"], order=["agent", "frame"])
            world = json.load(gzip.open(p[:-4] + "_world.json.gz", "rt", encoding="utf-8"))
            ter = Terrain(words0, ox, oy)
            words = words0.copy()
            H, W = words.shape
            tc_at = {w["f"]: w["tc"] for w in world if w.get("tc")}
            alive = rows[rows["status"] == 0]
            nxt = np.ones(len(alive), bool)
            nxt[:-1] = (alive["agent"][1:] == alive["agent"][:-1]) & (alive["frame"][1:] == alive["frame"][:-1] + 1)
            vy_next = np.append(alive["vy"][1:], np.nan)
            vy_next[~nxt] = np.nan
            upto = -1
            for f in np.unique(alive["frame"]):
                changed = False
                for ff in range(upto + 1, int(f) + 1):     # tiles as the game had them at this frame
                    for idx, word in tc_at.get(ff, ()):
                        r, c = divmod(int(idx), W)
                        if 0 <= r < H:
                            words[r, c] = word
                            changed = True
                upto = int(f)
                if changed:
                    ter.update(words)
                m = alive["frame"] == f
                a = alive[m]
                g, side = ter.support(a["x"], a["y"])
                vy0 = np.abs(a["vy"]) < 1e-3
                rest = vy0 & (np.abs(vy_next[m]) < 1e-3)
                air = (a["flags"] & AIR) != 0
                wall = (a["flags"] & WALL) != 0
                for key, v in (("n", np.ones(len(a), bool)), ("ground", g), ("wall", side != 0), ("vy0", vy0),
                               ("rest", rest), ("rest_ground", rest & g), ("ground_vy0", g & vy0),
                               ("air_flag", air), ("ground_air_flag", g & air), ("noair_ground", ~air & g),
                               ("noair", ~air), ("wall_flag", wall), ("wall_flag_geom", wall & (side != 0)),
                               ("geom_wall_flag", (side != 0) & wall), ("ground_or_wall", g | (side != 0)),
                               ("rest_sup", rest & (g | (side != 0)))):
                    tot[key] += int(v.sum())
                for st in np.unique(a["state"]):
                    k = a["state"] == st
                    ps = per_state[int(st)]
                    ps["n"] += int(k.sum())
                    ps["ground"] += int((g & k).sum())
                    ps["wall"] += int(((side != 0) & k).sum())
                    ps["vy0"] += int((vy0 & k).sum())
                    ps["air_flag"] += int((air & k).sum())
                    ps["wall_flag"] += int((wall & k).sum())
        out[name] = {"shards": min(n, shards), "tot": dict(tot), "state": {s: dict(v) for s, v in per_state.items()}}
        t = tot
        pc = lambda a, b: 100.0 * a / max(1, b)
        log(f"{name}: {t['n']} alive rows from {out[name]['shards']} waves")
        log(f"  resting (vy == 0 now and next frame): {pc(t['rest'], t['n']):.1f}% of rows; of them on the ground "
            f"by the rule {pc(t['rest_ground'], t['rest']):.1f}%, supported (ground or wall) {pc(t['rest_sup'], t['rest']):.1f}%")
        log(f"  rule says ground: {pc(t['ground'], t['n']):.1f}% of rows; of them vy == 0: {pc(t['ground_vy0'], t['ground']):.1f}%")
        log(f"  flag 0x2 set: {pc(t['air_flag'], t['n']):.1f}% of rows; of the rule's ground rows it is set on "
            f"{pc(t['ground_air_flag'], t['ground']):.1f}% (so it is not 'airborne'); flag 0x2 clear -> rule ground "
            f"{pc(t['noair_ground'], t['noair']):.1f}%")
        log(f"  wall: rule {pc(t['wall'], t['n']):.1f}% of rows, flag 0x200 {pc(t['wall_flag'], t['n']):.1f}%; flag 0x200 "
            f"rows the rule calls wall {pc(t['wall_flag_geom'], t['wall_flag']):.1f}%, rule's wall rows with flag 0x200 "
            f"{pc(t['geom_wall_flag'], t['wall']):.1f}%")
        log("  movement state: rows, rule ground %, rule wall %, vy == 0 %, flag 0x2 %, flag 0x200 %")
        for s_, ps in sorted(out[name]["state"].items(), key=lambda kv: -kv[1]["n"]):
            if ps["n"] < 0.002 * t["n"]:
                continue
            log(f"    {s_:3d}: {ps['n']:7d}  {pc(ps['ground'], ps['n']):5.1f}  {pc(ps['wall'], ps['n']):5.1f}  "
                f"{pc(ps['vy0'], ps['n']):5.1f}  {pc(ps['air_flag'], ps['n']):5.1f}  {pc(ps['wall_flag'], ps['n']):5.1f}")
    return out


def measure_overlap(dirs, tols=(0.5, 1.0, 1.5, 2.0), shards: int = 6, log=print, slopes: bool = True) -> dict:
    """How often real recorded states overlap solid tiles by more than each tolerance (the planner's
    collision guard, plan/planner.py, fires on imagined states that do): per level of each wm_data folder, the
    first `shards` waves, alive rows, tiles as the game had them at that frame, lock tiles open for the keys
    the agent held. Returns {level: {"rows": n, "fired": {tol: count}}} summed over the folders."""
    import glob
    import gzip
    import json
    import os
    from collections import defaultdict
    from plan.measure_progress import level_from_data
    out = defaultdict(lambda: {"rows": 0, "fired": defaultdict(int)})
    for d in dirs:
        seen = defaultdict(int)
        for p in sorted(glob.glob(os.path.join(d, "wave_*.npz"))):
            z = np.load(p)
            name = json.loads(str(z["meta"])).get("level")
            if name is None or seen[name] >= shards:
                continue
            seen[name] += 1
            lv = level_from_data(d, name)
            words0 = lv.words0.astype(np.uint32)
            ter = Terrain(words0, lv.ox, lv.oy, slopes=slopes)
            words = words0.copy()
            H, W = words.shape
            world = json.load(gzip.open(p[:-4] + "_world.json.gz", "rt", encoding="utf-8"))
            tc_at = {w["f"]: w["tc"] for w in world if w.get("tc")}
            alive = z["rows"][z["rows"]["status"] == 0]
            upto = -1
            for f in np.unique(alive["frame"]):
                changed = False
                for ff in range(upto + 1, int(f) + 1):
                    for idx, word in tc_at.get(ff, ()):
                        r, c = divmod(int(idx), W)
                        if 0 <= r < H:
                            words[r, c] = word
                            changed = True
                upto = int(f)
                if changed:
                    ter.update(words)
                a = alive[alive["frame"] == f]
                for mask in np.unique(a["keys"]):
                    m = a["keys"] == mask
                    t = ter.opened(lv.opened_cells(int(mask))) if mask else ter
                    o = out[name]
                    o["rows"] += int(m.sum())
                    for tol in tols:
                        o["fired"][tol] += int(t.box_overlap(a["x"][m], a["y"][m], BODY - tol).sum())
    res = {k: {"rows": v["rows"], "fired": dict(v["fired"])} for k, v in sorted(out.items())}
    log("level     rows    " + "  ".join(f"tol {t:g}" for t in tols))
    tot = {"rows": 0, "fired": defaultdict(int)}
    for k, v in res.items():
        tot["rows"] += v["rows"]
        for t in tols:
            tot["fired"][t] += v["fired"][t]
        log(f"{k:6s} {v['rows']:8d}  " + "  ".join(f"{v['fired'][t]:7d}" for t in tols))
    log(f"all    {tot['rows']:8d}  " + "  ".join(f"{tot['fired'][t]:7d}" for t in tols)
        + "   (" + ", ".join(f"{100 * tot['fired'][t] / max(1, tot['rows']):.3f}%" for t in tols) + ")")
    return res


if __name__ == "__main__":
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description="check the support rule (or, --overlap, the planner's collision "
                                             "guard tolerance) on recorded play")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--levels", nargs="+", default=["1-5", "1-13"])
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--overlap", action="store_true", help="real states overlapping solid tiles, per tolerance")
    ap.add_argument("--no-slopes", action="store_true", help="--overlap: slope cells open (as the guard)")
    a = ap.parse_args()
    if a.overlap:
        measure_overlap(a.data, shards=a.shards, slopes=not a.no_slopes)
    else:
        measure(a.data[0], a.levels, a.shards)
