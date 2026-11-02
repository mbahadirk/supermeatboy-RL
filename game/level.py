"""The level and hazard model of the real game (realgame.js reports the facts): the collision tile layer kept
live from the game's changes, goals, spawn, keys, locks, portals, fans, orbs, moving platforms, sealed pockets,
the agents' cell codes and surface codes, the lethal pieces as rotated boxes, and which extracted level file the
game shows. Moved out of realgame.py (the PPO trainer), which re-exports every name here, so the planner can use
it without importing the trainer. Needs the smbrl folder on sys.path (realpath, hazard_reach)."""
from __future__ import annotations

import math
import os
import sys
import json
from collections import deque

import numpy as np

from realpath import box_distance, tile_classes
import hazard_reach

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the smbrl folder (tools/, imported/)

HZ = 60
TILE = 20.0
# surface codes (second local map): which sides of a solid cell are open
S_FLOOR, S_WALL, S_LEDGE, S_CEIL, S_CEIL_WALL, S_INSIDE = 1.0, -1.0, 0.5, -0.5, -0.75, 0.25
C_EMPTY, C_SOLID, C_HAZARD, C_GOAL = 0.0, 1.0, -1.0, 0.75
C_WIND, C_ORB, C_ACCEL = 0.3, -0.4, 0.55  # map codes: in a fan's wind column, inside an orb, accelerator tile
C_PLATFORM = 0.85                         # map code: a moving platform (solid, but it moves)
PLAT_HALF = (30.0, 10.0)                  # a platform is a 60 x 20 bar (20 x 60 when upright), realgame.js platforms()
PLAT_STILL = 240                          # frames: a platform seen in place this long is a fixed one (a wall for routes)
SEALED_MAX = 150                          # cells: a pocket the player cannot get into larger than this is left open
                                          # (a moving saw at its start position could look like a seal)
ACCEL_TILE = 98                           # Salt Factory surface tile (no special bit); what it does is not traced


# ---- level ------------------------------------------------------------------------------------------
def tile_solid(words: np.ndarray) -> np.ndarray:
    """Cells the player cannot enter: ids 1-99, and special tiles (bit 8: crumble, lock, switch blocks)."""
    tid = words & 0xff
    return ((tid > 0) & (tid < 100)) | (((words >> 8) & 1).astype(bool) & (tid > 0) & (tid <= 100))


# pieces with no type-table row that are only decoration (the game's "no row = lethal" rule would make them
# hazards): names from the level's palette (tools/smbl_extract.py; imported files, personal use)
DECOR = ("particle", "smoke", "quake", "portal", "sign", "bosspos", "square", "circle", "xxxxx", "poof",
         "flap", "blablabla", "bip", "temp", "explosion", "warp", "pickup", "ghost", "break")   # not "Symbol 111": needles
_names_cache: dict = {}


def piece_names(level: str) -> list[str] | None:
    """Obstacle names of the level's palette, by piece type (low byte), from the extracted game files."""
    if level in _names_cache:
        return _names_cache[level]
    out = None
    try:
        sys.path.insert(0, os.path.join(HERE, "tools"))
        import smbl_extract as X
        raw = os.path.join(HERE, "imported", "smbl_raw")
        pal = X.parse(os.path.join(raw, "Levels", level + ".smbl"))["palette"]
        out = X.am_names(os.path.join(raw, "palettes", "pc_LevelPalettes_" + X.PALETTE_OBSTACLES[pal]))
    except Exception:
        out = None
    _names_cache[level] = out
    return out


class Level:
    """Facts the game reported for the current level: the collision tile layer (kept live from the game's
    changes), goals, spawn, keys, bandages."""

    def __init__(self, info: dict, tiles: bytes):
        self.name = info["name"]
        self.W, self.H, self.ox, self.oy = info["W"], info["H"], info["ox"], info["oy"]
        self.words0 = np.frombuffer(tiles, dtype="<u4").reshape(self.H, self.W).copy()   # row 0 = bottom
        self.words = self.words0.copy()
        self.solid0 = tile_solid(self.words0)
        self.solid = self.solid0.copy()
        self.special = ((self.words0 >> 8) & 1).astype(bool) & self.solid0
        self.tile_ver, self.tile_log = 0, 0
        self.goals = [tuple(g) for g in info["goals"]]
        self.spawn = tuple(info["spawn"])
        self.names = piece_names(self.name)
        nm = lambda t: (self.names[t].lower() if self.names and 0 <= t < len(self.names) else None)
        objs = info.get("objs", [])
        self.keys = [(x, y) for t, x, y, fl in objs if (fl >= 0 and fl & 0x40) or nm(t) == "key"]
        self.bandages = [(x, y) for t, x, y, fl in objs if (fl >= 0 and fl & 0x08) or nm(t) == "bandaid"]
        # which lethal pieces (by the game's rule) are really decoration; ids = piece addresses (stable)
        self.set_lethal(info.get("lethalTypes", []), info.get("lethalIds"))
        self.hz_level_ids = set(self.hz_ids)       # pieces there from the start (shots come later)
        self.lock_groups = self._lock_groups()
        # portals: the level build pairs portal pieces in their order (FUN_004c8c10)
        pt = set(info.get("portalTypes", []))
        ends = [(x, y) for t, x, y, fl in objs if t in pt]
        self.portals = list(zip(ends[0::2], ends[1::2]))
        self.accel = ((self.words0 & 0xff) == ACCEL_TILE) & ~(((self.words0 >> 8) & 1).astype(bool))
        self.fans, self.orbs = [], []
        self.wind = np.zeros((self.H, self.W), dtype=bool)
        self.orb_cells = np.zeros((self.H, self.W), dtype=bool)
        self.base = np.where(self.solid, C_SOLID, C_EMPTY).astype(np.float32)
        self.base[self.accel & self.solid] = C_ACCEL
        for gx, gy in self.goals:
            c, r = self.cell(gx, gy)
            if 0 <= c < self.W and 0 <= r < self.H:
                self.base[r, c] = C_GOAL
        self.set_fields(info.get("fx"))
        self.plats, self.plat_vel, self.plat_cells, self.plat_set = [], [], [], set()
        self.plat_seen = np.zeros((self.H, self.W), dtype=bool)   # every cell a platform has covered
        self.plat_owner = {}                                       # (r, c) -> platforms that covered it
        self.sealed_haz = np.zeros((self.H, self.W), dtype=bool)  # pockets inside hazards (lava pits): hazard
        self.sealed_wall = np.zeros((self.H, self.W), dtype=bool) # pockets inside walls: solid
        self.set_platforms(info.get("pf"))
        self.dist = self._bfs()
        self.diag = float(np.hypot(self.W, self.H))

    def set_fields(self, fx):
        """Fans [x, y, up x, up y, half width, range, strength] and orbs [x, y, r1, r2, s1, s2] from the game:
        which open cells are in a wind column or inside an orb (agents' map, routes)."""
        if not fx:
            return
        fans, orbs = fx.get("fans", []), fx.get("orbs", [])
        if fans == self.fans and orbs == self.orbs:
            return
        self.fans, self.orbs = fans, orbs
        cx = self.ox + (np.arange(self.W) + 0.5) * TILE
        cy = self.oy + (np.arange(self.H) + 0.5) * TILE
        X, Y = np.meshgrid(cx, cy)
        self.wind[:] = False
        for x, y, ux, uy, hw, rng, st in fans:
            if rng <= 0:
                continue
            along = (X - x) * ux + (Y - y) * uy
            lat = (X - x) * uy - (Y - y) * ux
            self.wind |= (along >= 0) & (along <= rng) & (np.abs(lat) <= hw + 6.0)
        self.orb_cells[:] = False
        for x, y, r1, r2, s1, s2 in orbs:
            self.orb_cells |= np.hypot(X - x, Y - y) <= max(r1, r2)
        self.fx_ver = getattr(self, "fx_ver", 0) + 1
        self.tile_ver += 1

    def set_platforms(self, pf) -> bool:
        """Moving platforms now ([x, y, upright] from the game): boxes, velocities (by order, from the last
        frame), the cells they cover, every cell they have covered, and which of them move (a platform seen in
        place for PLAT_STILL frames is a fixed one). True when the routes should be rebuilt: a platform
        covered new cells, started moving, or turned out to be fixed."""
        if not pf:
            return False
        boxes = [(x, y) + ((PLAT_HALF[1], PLAT_HALF[0]) if up else PLAT_HALF) + (bool(up),) for x, y, up in pf]
        if len(self.plats) == len(boxes):
            self.plat_vel = [((b[0] - a[0]) * HZ, (b[1] - a[1]) * HZ) for a, b in zip(self.plats, boxes)]
        else:
            self.plat_vel = [(0.0, 0.0)] * len(boxes)
        changed = False
        if len(getattr(self, "plat_start", [])) != len(boxes):   # (the level's platforms)
            self.plat_start = [b[:2] for b in boxes]
            self.plat_moved = [False] * len(boxes)
            self.plat_frames = 0
        self.plat_frames += 1
        for k, b in enumerate(boxes):
            if not self.plat_moved[k] and math.hypot(b[0] - self.plat_start[k][0], b[1] - self.plat_start[k][1]) > 2.0:
                self.plat_moved[k] = changed = True
        if self.plat_frames == PLAT_STILL and not all(self.plat_moved):
            changed = True                             # the ones still in place now count as fixed
        self.plats = boxes
        cells = []
        for k, (x, y, hw, hh, up) in enumerate(boxes):   # cells the bar covers (more than 5 units of each side)
            c0, c1 = int((x - hw + 5 - self.ox) // TILE), int((x + hw - 5 - self.ox) // TILE)
            r0, r1 = int((y - hh + 5 - self.oy) // TILE), int((y + hh - 5 - self.oy) // TILE)
            for r in range(max(0, r0), min(self.H, r1 + 1)):
                for c in range(max(0, c0), min(self.W, c1 + 1)):
                    cells.append((r, c, up))
                    if not self.plat_seen[r, c]:
                        self.plat_seen[r, c] = changed = True
                    self.plat_owner.setdefault((r, c), set()).add(k)
        self.plat_cells = cells
        self.plat_set = {(r, c) for r, c, _ in cells}
        return changed

    def hazard_mask(self, boxes) -> np.ndarray:
        """Open cells a hazard box reaches (the body would touch it there), as in the agents' map."""
        out = np.zeros((self.H, self.W), dtype=bool)
        cx = self.ox + (np.arange(self.W) + 0.5) * TILE
        cy = self.oy + (np.arange(self.H) + 0.5) * TILE
        for b in boxes:
            x, y, hw, hh, _ = b
            far = float(np.hypot(hw, hh)) + 6.0
            c0, c1 = max(0, int((x - far - self.ox) // TILE)), min(self.W, int((x + far - self.ox) // TILE) + 1)
            r0, r1 = max(0, int((y - far - self.oy) // TILE)), min(self.H, int((y + far - self.oy) // TILE) + 1)
            if c0 < c1 and r0 < r1:
                out[r0:r1, c0:c1] |= box_distance(cx[None, c0:c1], cy[r0:r1, None], b) < 6.0
            c, r = self.cell(x, y)
            if 0 <= c < self.W and 0 <= r < self.H:
                out[r, c] = True
        return out & ~self.solid0

    def set_sealed(self, boxes, seeds) -> int:
        """Open cells the player cannot get into: not reachable from the spawn, goals, keys or portals without
        going through a wall or a hazard (boxes: the hazards that stay put). Lava pits are drawn as a few
        stacked lava pieces whose boxes leave gaps between them; in the game the pit is all lava. A pocket
        that touches a hazard counts as hazard, one closed in by walls as solid (agents' map, routes,
        monitor). Pockets over SEALED_MAX cells are left open. Returns the number of sealed cells."""
        haz = self.hazard_mask(boxes)
        passable = (~self.solid0 | self.special) & ~haz
        seen = np.zeros((self.H, self.W), dtype=bool)
        todo = []
        for x, y in seeds:
            c, r = self.cell(x, y)
            if 0 <= c < self.W and 0 <= r < self.H and not seen[r, c]:
                seen[r, c] = True
                todo.append((r, c))
        while todo:
            r, c = todo.pop()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < self.H and 0 <= nc < self.W and not seen[nr, nc] and passable[nr, nc]:
                    seen[nr, nc] = True
                    todo.append((nr, nc))
        pocket = ~self.solid0 & ~seen & ~haz
        self.sealed_haz[:] = False
        self.sealed_wall[:] = False
        done = np.zeros_like(pocket)
        for r0, c0 in zip(*np.nonzero(pocket)):
            if done[r0, c0]:
                continue
            comp, todo, touch = [], [(r0, c0)], False
            done[r0, c0] = True
            while todo:
                r, c = todo.pop()
                comp.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < self.H and 0 <= nc < self.W:
                        if haz[nr, nc]:
                            touch = True
                        elif pocket[nr, nc] and not done[nr, nc]:
                            done[nr, nc] = True
                            todo.append((nr, nc))
            if len(comp) > SEALED_MAX:
                continue
            rr, cc = zip(*comp)
            (self.sealed_haz if touch else self.sealed_wall)[list(rr), list(cc)] = True
        self.tile_ver += 1
        return int(self.sealed_haz.sum() + self.sealed_wall.sum())

    def plat_fixed(self, k: int) -> bool:
        return self.plat_frames >= PLAT_STILL and not self.plat_moved[k]

    def plat_blocks(self) -> np.ndarray:
        """Cells of the fixed platforms: walls for the routes."""
        out = np.zeros((self.H, self.W), dtype=bool)
        for (r, c), ks in self.plat_owner.items():
            if all(self.plat_fixed(k) for k in ks):
                out[r, c] = True
        return out

    def plat_doors(self) -> np.ndarray:
        """Every cell a moving (or not yet known) platform has covered: the routes cross them only when there
        is no way around, like a moving saw's sweep (it may be in the way, or carry the player)."""
        return self.plat_seen & ~self.plat_blocks()

    def plat_support(self) -> np.ndarray:
        """Open cells a route may stand in thanks to a platform: just above anywhere one has been (riding
        it), and beside anywhere one has been (a wall to jump off)."""
        seen = self.plat_seen
        out = np.zeros_like(seen)
        out[1:, :] |= seen[:-1, :]
        out[:, 1:] |= seen[:, :-1]
        out[:, :-1] |= seen[:, 1:]
        return out & ~self.solid & ~seen

    def surfaces(self) -> np.ndarray:
        """Per cell, what a solid tile offers the player (live tiles): a floor to run on (open above), a wall to
        jump off (open left or right), both (a ledge corner), a ceiling (open below), or nothing (inside)."""
        if getattr(self, "_surf_ver", None) == self.tile_ver:
            return self._surf
        S = self.solid
        H, W = S.shape
        pad = np.pad(S, 1, constant_values=True)
        up, down = ~pad[2:, 1:-1], ~pad[:-2, 1:-1]            # row 0 = bottom: "above" is row + 1
        side = ~pad[1:-1, :-2] | ~pad[1:-1, 2:]
        out = np.zeros((H, W), dtype=np.float32)
        out[S] = S_INSIDE
        out[S & down & ~up & ~side] = S_CEIL
        out[S & down & side & ~up] = S_CEIL_WALL
        out[S & side & ~up & ~down] = S_WALL
        out[S & up & ~side] = S_FLOOR
        out[S & up & side] = S_LEDGE
        self._surf, self._surf_ver = out, self.tile_ver
        return out

    def wind_at(self, x: float, y: float):
        """The wind force on a character here, as the game computes it (FUN_004c40e0)."""
        fx = fy = 0.0
        for fxp, fyp, ux, uy, hw, rng, st in self.fans:
            if rng <= 0:
                continue
            along = (x - fxp) * ux + (y - fyp) * uy
            lat = (x - fxp) * uy - (y - fyp) * ux
            if along < 0 or along > rng or abs(lat) > hw + 6.0:
                continue
            c = 1.0 - along / rng
            if c >= 0.5:
                c *= 5.0
            fx += ux * st * c
            fy += uy * st * c
        return fx, fy

    def set_lethal(self, types: list, ids=None):
        nm = lambda t: (self.names[t].lower() if self.names and 0 <= t < len(self.names) else None)
        self.hz_keep = [not (nm(t) and nm(t).startswith(DECOR)) for t in types]
        self.hz_grow = hazard_reach.growth([nm(t) for t in types])   # measured kill reach (hazard_reach.py)
        self.hz_ids = list(ids) if ids is not None else [str(k) for k in range(len(types))]

    def _lock_groups(self) -> list[list[int]]:
        """Per key: the lock tiles it opens (cell indices row * W + col). Lock tiles in touch form a group;
        each group is opened by the key nearest to it."""
        lock = tile_classes(self.words0)["lock"]
        out = [[] for _ in self.keys]
        if not self.keys or not lock.any():
            return out
        seen = np.zeros_like(lock)
        for r0, c0 in zip(*np.nonzero(lock)):
            if seen[r0, c0]:
                continue
            comp, q = [], [(r0, c0)]
            seen[r0, c0] = True
            while q:
                r, c = q.pop()
                comp.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.H and 0 <= cc < self.W and lock[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            q.append((rr, cc))
            cx = [self.ox + (c + 0.5) * TILE for _, c in comp]
            cy = [self.oy + (r + 0.5) * TILE for r, _ in comp]
            k = min(range(len(self.keys)), key=lambda k: min(np.hypot(np.array(cx) - self.keys[k][0],
                                                                        np.array(cy) - self.keys[k][1])))
            out[k] += [int(r) * self.W + int(c) for r, c in comp]
        return out

    def opened_cells(self, mask: int) -> list[tuple[int, int]]:
        """Lock tiles open for an agent holding keys `mask` (bit k = key k)."""
        out = []
        for k, g in enumerate(self.lock_groups):
            if (mask >> k) & 1:
                out += [divmod(ci, self.W) for ci in g]
        return out

    def apply_tiles(self, changes: list):
        """Cells the game changed this frame. A special tile (crumble / lock / switch block) whose word differs
        from the level's start is taken as gone; back to its start word, it is there again."""
        for idx, w in changes:
            r, c = divmod(int(idx), self.W)
            if not (0 <= r < self.H):
                continue
            old = int(self.words[r, c])
            self.words[r, c] = w
            if self.special[r, c]:
                solid = int(w) == int(self.words0[r, c])
            else:
                solid = bool(tile_solid(np.array([w], dtype=np.uint32))[0])
            if self.tile_log < 12:             # how the game marks a broken / opened tile (for the record)
                self.tile_log += 1
                print(f"[tiles] {self.name}: cell ({c},{r}) 0x{old:08x} -> 0x{int(w):08x}"
                      f" ({'solid' if solid else 'open'})", flush=True)
            if solid != self.solid[r, c]:
                self.solid[r, c] = solid
                if self.base[r, c] != C_GOAL:
                    self.base[r, c] = C_SOLID if solid else C_EMPTY
                self.tile_ver += 1

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.ox) // TILE), int((y - self.oy) // TILE)

    def _bfs(self) -> np.ndarray:
        """Path length (tiles) from each open cell to the nearest goal, 4-connected; inf where unreachable."""
        d = np.full((self.H, self.W), np.inf, dtype=np.float32)
        q = deque()
        for gx, gy in self.goals:
            c, r = self.cell(gx, gy)
            if 0 <= c < self.W and 0 <= r < self.H:
                d[r, c] = 0.0
                q.append((c, r))
        while q:
            c, r = q.popleft()
            for nc, nr in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                if 0 <= nc < self.W and 0 <= nr < self.H and d[nr, nc] == np.inf and not self.solid[nr, nc]:
                    d[nr, nc] = d[r, c] + 1.0
                    q.append((nc, nr))
        return d

    def goal_dist(self, x: float, y: float) -> float:
        c, r = self.cell(x, y)
        if 0 <= c < self.W and 0 <= r < self.H and np.isfinite(self.dist[r, c]):
            return float(self.dist[r, c])
        gx, gy = min(self.goals, key=lambda g: (g[0] - x) ** 2 + (g[1] - y) ** 2) if self.goals else self.spawn
        return float(np.hypot(gx - x, gy - y) / TILE) + self.diag   # off the known paths: worse than any path

    def grid_with_hazards(self, boxes: list) -> np.ndarray:
        """Cell codes now: tiles, goals, and the lethal pieces at their current positions (every open cell
        whose centre is within the body's half size of a hazard box, and the cell the hazard is in)."""
        g = self.base.copy()
        if self.wind.any():
            g[self.wind & (g == C_EMPTY)] = C_WIND
        if self.orb_cells.any():
            g[self.orb_cells & (g == C_EMPTY)] = C_ORB
        g[self.sealed_haz & (g != C_SOLID)] = C_HAZARD   # pockets the player cannot get into (lava pits)
        g[self.sealed_wall & (g != C_SOLID)] = C_SOLID
        for r, c, _ in self.plat_cells:              # moving platforms where they are now
            if g[r, c] != C_SOLID:
                g[r, c] = C_PLATFORM
        cx = self.ox + (np.arange(self.W) + 0.5) * TILE
        cy = self.oy + (np.arange(self.H) + 0.5) * TILE
        for b in boxes:
            x, y, hw, hh, _ = b
            far = float(np.hypot(hw, hh)) + 6.0
            c0, c1 = max(0, int((x - far - self.ox) // TILE)), min(self.W, int((x + far - self.ox) // TILE) + 1)
            r0, r1 = max(0, int((y - far - self.oy) // TILE)), min(self.H, int((y + far - self.oy) // TILE) + 1)
            if c0 < c1 and r0 < r1:
                m = (box_distance(cx[None, c0:c1], cy[r0:r1, None], b) < 6.0) & (g[r0:r1, c0:c1] != C_SOLID)
                g[r0:r1, c0:c1][m] = C_HAZARD
            c, r = self.cell(x, y)
            if 0 <= c < self.W and 0 <= r < self.H and g[r, c] != C_SOLID:
                g[r, c] = C_HAZARD
        return g


def hazard_boxes(hz: list, keep=None, grow=None) -> tuple[list, list]:
    """Lethal pieces [x, y, size w, size h, scale x, scale y, angle] (realgame.js; size from the game's
    FUN_004ca860 at scale 1) as rotated boxes (x, y, half w, half h, angle). keep: which ones are real hazards
    (not decoration). grow: per piece (x, y) added to the half sizes, how much further than the body size the
    piece really kills (hazard_reach.py, measured from deaths; Level.hz_grow). Returns the boxes and the
    indices of the entries kept."""
    out, kept = [], []
    for k, h in enumerate(hz):
        if keep is not None and k < len(keep) and not keep[k]:
            continue
        x, y, w, hgt, sx, sy, ang = h
        sx, sy = abs(sx or 1.0), abs(sy or 1.0)
        if w <= 0 and hgt <= 0:
            w = hgt = 24.0
        gx, gy = grow[k] if grow is not None and k < len(grow) else (0.0, 0.0)
        out.append((x, y, 0.5 * w * sx + gx, 0.5 * hgt * sy + gy, ang))
        kept.append(k)
    return out, kept


def identify_level(info: dict, tiles: bytes) -> str | None:
    """Name of the level the game shows when attached mid-level: the extracted level file (tools/smbl_extract.py)
    with the same size, origin and collision tiles."""
    d = os.path.join(HERE, "imported", "smb_static")
    if not os.path.isdir(d):
        return None
    t = np.frombuffer(tiles, dtype="<u4").reshape(info["H"], info["W"]) & 0xff
    live = ((t > 0) & (t < 100)).astype(np.uint8)
    try:   # prefer the files the game's chapters use (there are editor copies such as 4-1_TEST)
        with open(os.path.join(d, "chapters.json")) as fh:
            used = [c["file"] for c in json.load(fh)]
    except (OSError, ValueError):
        used = []
    names = sorted(f[:-len("_semantic.json")] for f in os.listdir(d) if f.endswith("_semantic.json"))
    names.sort(key=lambda n: (n not in used, len(n), n))
    for n in names:
        f = n + "_semantic.json"
        with open(os.path.join(d, f)) as fh:
            j = json.load(fh)
        if (j["W"], j["H"]) != (info["W"], info["H"]) or tuple(j["origin"]) != (info["ox"], info["oy"]):
            continue
        tid = np.array(j["solid"], dtype=np.uint8)
        if (tid == live).mean() > 0.98:
            return n
    return None
