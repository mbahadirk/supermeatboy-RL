"""The level and hazard model of the real game (realgame.js reports the facts): the collision tile layer kept
live from the game's changes, goals, spawn, keys, locks, portals, fans, orbs, moving platforms, sealed pockets,
the agents' cell codes and surface codes, the lethal pieces as rotated boxes, and which extracted level file the
game shows. Moved out of realgame.py (the PPO trainer), which re-exports every name here, so the planner can use
it without importing the trainer. Needs the smbrl folder on sys.path (realpath, contact_reach)."""
from __future__ import annotations

import math
import os
import sys
import json
from collections import deque

import numpy as np

from realpath import BODY, box_distance, tile_classes, tile_shape
import contact_reach

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
# off screen: the player dies when its centre leaves the camera's view x 1.15 (FUN_00520d30, mode 0 only); the
# camera is clamped to the level's camera bounds, so for a followed player that is a fixed kill box per level
# (realgame.js killBox, sent as info["kb"]). Without one (no camera bounds, unreadable): this far past the
# map's left / right / top edge, or below its bottom, is dead (the older realgame.js rule).
OUT_SIDE, OUT_BOTTOM = 10.0, 60.0


# ---- level ------------------------------------------------------------------------------------------
def parse_kill_box(kb):
    """(left, right, bottom, top) from realgame.js `kb`, or None when absent or not a finite, non-empty box."""
    try:
        left, right, bottom, top = (float(v) for v in kb)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (left, right, bottom, top)) or left >= right or bottom >= top:
        return None
    return left, right, bottom, top


def alive_box(ox: float, oy: float, W: int, H: int, kill_box=None) -> tuple[float, float, float, float]:
    """(left, right, bottom, top) lines a centre must stay within to live: the kill box when there is one,
    else the map's edges widened by OUT_SIDE (left, right, top) and OUT_BOTTOM (bottom)."""
    if kill_box is not None:
        return tuple(float(v) for v in kill_box)
    return (float(ox) - OUT_SIDE, float(ox + W * TILE) + OUT_SIDE, float(oy) - OUT_BOTTOM,
            float(oy + H * TILE) + OUT_SIDE)


def tile_solid(words: np.ndarray) -> np.ndarray:
    """Cells the player cannot enter: ids 1-99, and special tiles (bit 8: crumble, lock, switch blocks)."""
    tid = words & 0xff
    return ((tid > 0) & (tid < 100)) | (((words >> 8) & 1).astype(bool) & (tid > 0) & (tid <= 100))


# pieces with no type-table row that are only decoration (the game's "no row = lethal" rule would make them
# hazards): names from the level's palette (tools/smbl_extract.py; imported files, personal use). Only for
# what the game's kill shapes do not decide: the world model's inputs (trained with this rule) and hazard
# rows recorded before realgame.js sent the shapes (hazard_boxes); a piece's measured shapes decide the rest
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


LOCK_CONTEXT_PROTOCOL = "private-locks-v1"
KEY_PICKUP_RADIUS = 22.0                 # realgame.js KEY_R: strict post-update endpoint distance


def key_pickup_policy(step=1):
    return {"rule": "strict-native-endpoint-circle-v1", "radius": KEY_PICKUP_RADIUS,
            "frames_per_model_step": int(step), "interpolated_integer_endpoints": int(step) > 1,
            "guard_mask": "before_incoming_model_step", "new_mask_usable": "next_model_step",
            "maximum_opening_delay_frames": max(0, int(step) - 1),
            "delay_reference": "own_interpolated_pickup_not_observed_game" if int(step) > 1 else "predicted_endpoint",
            "approximation": "linear_integer_samples_no_midstep_physics" if int(step) > 1 else "post_update_endpoint"}


def key_lock_groups(words, ox, oy, keys):
    """The existing game's adapter rule: connected lock cells belong to their nearest supplied key."""
    lock = tile_classes(words)["lock"]
    H, W = words.shape
    out = [[] for _ in keys]
    if not keys or not lock.any():
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
                    if 0 <= rr < H and 0 <= cc < W and lock[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
        cx = [ox + (c + 0.5) * TILE for _, c in comp]
        cy = [oy + (r + 0.5) * TILE for r, _ in comp]
        k = min(range(len(keys)), key=lambda k: min(np.hypot(np.array(cx) - keys[k][0],
                                                           np.array(cy) - keys[k][1])))
        out[k] += [int(r) * W + int(c) for r, c in comp]
    return out


def opened_lock_cells(groups, mask, width):
    return [divmod(int(ci), width) for k, g in enumerate(groups) if (int(mask) >> k) & 1 for ci in g]


class LockContext:
    """Immutable private lock association and deterministic endpoint pickup rule; no learned pickup head."""

    def __init__(self, words, meta):
        self.width = words.shape[1]
        self.keys = tuple(tuple(k) for k in meta.get("keys", []))
        context = meta.get("lock_context")
        if context is not None:
            if context.get("protocol") != LOCK_CONTEXT_PROTOCOL or context.get("source") != "python-go-payload" or \
                    tuple(tuple(k) for k in context.get("keys", [])) != self.keys:
                raise ValueError("invalid applied key/lock context")
            groups = context.get("groups")
            if not isinstance(groups, list) or len(groups) != len(self.keys):
                raise ValueError("key/lock context has no matching per-key groups")
            cells = [ci for group in groups for ci in group]
            if any(not isinstance(ci, (int, np.integer)) or not 0 <= ci < words.size for ci in cells) or \
                    len(set(cells)) != len(cells):
                raise ValueError("invalid or overlapping lock cells")
            self.groups = tuple(tuple(g) for g in groups)
            self.source = "python-go-payload"
        elif len(self.keys) <= 1:
            self.groups = tuple(tuple(g) for g in key_lock_groups(words, meta["ox"], meta["oy"], self.keys))
            self.source = "legacy-single-key-reconstruction" if self.keys else "no-keys"
        else:
            self.groups, self.source = None, "unknown-legacy-multi-key"

    def opened_cells(self, mask):
        mask = int(mask)
        if mask == 0:
            return []                         # preserves legacy/no-key inputs without guessing a mapping
        if mask < 0 or mask >> len(self.keys):
            raise ValueError("key mask contains an unknown key index")
        if self.groups is None:
            raise ValueError("key-owned legacy multi-key data needs a recorded applied lock context")
        return opened_lock_cells(self.groups, mask, self.width)

    def pickup_path(self, xs, ys, initial=0, step=1, upto=None):
        """Private masks after each integer-frame endpoint; no continuous-segment pickup.

        step1 matches realgame.js pickKeys after updateChar (even if its status changed). Coarse paths
        linearly interpolate integer endpoints, an approximation only: callers expose new masks to
        physics/guard only at the NEXT model step. Geometry and caller arrays are never mutated.
        """
        xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        if xs.shape != ys.shape or xs.ndim < 1 or xs.shape[-1] < 2 or int(step) != step or step < 1:
            raise ValueError("key pickup needs matching paths and a positive integer frame step")
        prefix = xs.shape[:-1]
        mask = np.broadcast_to(np.asarray(initial, np.uint32), prefix).copy()
        count = (xs.shape[-1] - 1) * int(step)
        limit = np.broadcast_to(count if upto is None else upto, prefix)
        masks = np.empty(prefix + (count + 1,), np.uint32)
        first = np.full(prefix + (len(self.keys),), -1, np.int64)
        masks[..., 0] = mask
        for frame in range(1, count + 1):
            col, frac = divmod(frame - 1, int(step))
            alpha = (frac + 1) / int(step)
            x = xs[..., col] + alpha * (xs[..., col + 1] - xs[..., col])
            y = ys[..., col] + alpha * (ys[..., col + 1] - ys[..., col])
            for k, (kx, ky) in enumerate(self.keys):
                bit = np.uint32(1 << k)
                hit = (frame <= limit) & ((mask & bit) == 0) & (np.hypot(x - kx, y - ky) < KEY_PICKUP_RADIUS)
                first[..., k] = np.where(hit, frame, first[..., k])
                mask = mask | np.where(hit, bit, np.uint32(0))
            masks[..., frame] = mask
        # Legacy multi-key geometry cannot silently invent which private locks a newly held key opens.
        for owned in np.unique(masks[..., -1]):
            self.opened_cells(owned)
        return masks, first

    def pickup_after_step(self, previous, following, step=1):
        masks, _ = self.pickup_path(np.stack([previous["x"], following["x"]], -1),
                                    np.stack([previous["y"], following["y"]], -1), previous["keys"], step)
        following["keys"] = masks[..., -1]


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
        # the camera-bounds kill box (realgame.js; raw inputs kept for the record)
        self.cam_bounds, self.view_half = info.get("cam_bounds"), info.get("view_half")
        self.kill_box = None
        self.outside_cells = np.zeros((self.H, self.W), dtype=bool)
        self.set_kill_box(info.get("kb"))

    def set_kill_box(self, kb) -> bool:
        """The kill box (left, right, bottom, top) from realgame.js, or None: the old map-edge rule. Recomputes
        outside_cells (cells whose centre is outside it: hazard in the agents' map, closed for the routes and
        the sealed pockets). True when it changed."""
        kb = parse_kill_box(kb) if kb is not None else None
        if kb == self.kill_box:
            return False
        self.kill_box = kb
        cx = self.ox + (np.arange(self.W) + 0.5) * TILE
        cy = self.oy + (np.arange(self.H) + 0.5) * TILE
        if kb is None:
            self.outside_cells = np.zeros((self.H, self.W), dtype=bool)
        else:
            self.outside_cells = self.out_of_level(cx[None, :], cy[:, None])
        self.tile_ver += 1
        return True

    def alive_box(self) -> tuple[float, float, float, float]:
        return alive_box(self.ox, self.oy, self.W, self.H, self.kill_box)

    def out_of_level(self, x, y) -> np.ndarray:
        """Where a centre (x, y) (arrays broadcast) is dead for being off screen: outside the kill box, or
        without one past the map edges (left / right / top + OUT_SIDE, bottom - OUT_BOTTOM)."""
        left, right, bottom, top = self.alive_box()
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        return (x < left) | (x > right) | (y < bottom) | (y > top)

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
        haz = self.hazard_mask(boxes) | self.outside_cells   # off screen is death, like a hazard
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
        self.hz_keep = [not (nm(t) and nm(t).startswith(DECOR)) for t in types]   # see DECOR
        self.hz_ids = list(ids) if ids is not None else [str(k) for k in range(len(types))]

    def _lock_groups(self) -> list[list[int]]:
        """Per key: the lock tiles it opens (cell indices row * W + col). Lock tiles in touch form a group;
        each group is opened by the key nearest to it."""
        return key_lock_groups(self.words0, self.ox, self.oy, self.keys)

    def opened_cells(self, mask: int) -> list[tuple[int, int]]:
        """Lock tiles open for an agent holding keys `mask` (bit k = key k)."""
        return opened_lock_cells(self.lock_groups, mask, self.W)

    def wm_geometry_metadata(self):
        """The same key-index association passed to realgame.js by the Python go payload."""
        return {"ox": self.ox, "oy": self.oy, "keys": [list(k) for k in self.keys],
                "lock_context": {"protocol": LOCK_CONTEXT_PROTOCOL, "source": "python-go-payload",
                                 "keys": [list(k) for k in self.keys], "groups": [list(g) for g in self.lock_groups]}}

    def open_dirs(self, pad: int) -> np.ndarray:
        """(2, H + 2 pad, W + 2 pad) float32: the open direction (x, y) of every cell's live tile word
        (realpath.tile_shape: a half block's open side, a slope's open corner; 0 elsewhere and outside the map),
        padded by pad cells. Cached until a tile word changes (apply_tiles)."""
        cache = self.__dict__.setdefault("_open_dirs", {})
        if pad not in cache:
            t = tile_shape(self.words)
            cache[pad] = np.pad(np.stack([t.ox, t.oy]).astype(np.float32), ((0, 0), (pad, pad), (pad, pad)))
        return cache[pad]

    def apply_tiles(self, changes: list):
        """Cells the game changed this frame. A special tile (crumble / lock / switch block) whose word differs
        from the level's start is taken as gone; back to its start word, it is there again."""
        for idx, w in changes:
            r, c = divmod(int(idx), self.W)
            if not (0 <= r < self.H):
                continue
            old = int(self.words[r, c])
            self.words[r, c] = w
            if old != int(w):
                self.__dict__.pop("_open_dirs", None)     # open_dirs reads the live words
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

    def goal_boxes(self) -> list:
        """Contact boxes of the goals, (x, y, half w, half h, 0) in game units: touched when the agent's centre
        is within BODY (6 units) of a box, the convention plan/safety.py uses for hazards (the box excludes the
        body). Reach measured from recorded touches (contact_reach.json); empty without that table."""
        return [b for b in (contact_reach.box("goal", x, y) for x, y in self.goals) if b is not None]

    def key_boxes(self) -> list:
        """Contact boxes of the keys in the order of self.keys (bit k of the keys mask), as goal_boxes. The key
        reach is unmeasured while no recorded level has keys (contact_reach.json says so): a fallback from
        realgame.js's pickup rule. Empty without the table."""
        out = [contact_reach.box("key", x, y) for x, y in self.keys]
        return out if all(b is not None for b in out) else []

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
        if self.kill_box is not None:                # off screen (outside the camera-bounds kill box): death
            g[self.outside_cells & (g != C_SOLID) & (g != C_GOAL)] = C_HAZARD
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


def shape_box(s) -> tuple:
    """One kill shape of a piece as the game tests it (realgame.js killShapes: [1, x, y, r, r, 0] a circle,
    [2, x, y, half w, half h, angle] a turned box, both against Meat Boy's axis-aligned box of half size BODY) as
    a box in the contact convention (x, y, half w, half h, angle): the centre dies within BODY of it.
    A circle: its bounding square (covers the circle grown by the body's box for r >= BODY). A turned box: its
    half sizes grow by how much further the body's box reaches along the box's axes than BODY
    (BODY (|cos| + |sin|) - BODY: 0 when upright, 2.5 units at 45 degrees), so its sides are exact."""
    kind, x, y, a, b, ang = s[:6]
    if int(kind) == 1:
        return (float(x), float(y), float(a), float(a), 0.0)
    e = BODY * (abs(math.cos(ang)) + abs(math.sin(ang))) - BODY
    return (float(x), float(y), float(a) + e, float(b) + e, float(ang))


def hazard_boxes(hz: list, keep=None) -> tuple[list, list]:
    """Lethal pieces (realgame.js hazards rows) as the boxes that kill (x, y, half w, half h, angle) and, per
    box, the index of its piece (a piece can have several boxes, or none now).

    A row [x, y, size w, size h, scale x, scale y, angle, shapes] has the game's own kill shapes (shape_box);
    [] = this piece kills nothing at the moment (decoration, or a phase of its animation without a shape).
    Rows without shapes (recorded before the shapes were sent, or None: not measured) fall back to the piece's
    size box from FUN_004ca860 at its scale, with keep (Level.hz_keep: not decoration) deciding which count."""
    out, kept = [], []
    for k, h in enumerate(hz):
        if len(h) > 7 and h[7] is not None:
            for s in h[7]:
                out.append(shape_box(s))
                kept.append(k)
            continue
        if keep is not None and k < len(keep) and not keep[k]:
            continue
        x, y, w, hgt, sx, sy, ang = h[:7]
        sx, sy = abs(sx or 1.0), abs(sy or 1.0)
        if w <= 0 and hgt <= 0:
            w = hgt = 24.0
        out.append((x, y, 0.5 * w * sx, 0.5 * hgt * sy, ang))
        kept.append(k)
    return out, kept


def box_ids(kept: list, names=None, prefix: str = "h") -> list[str]:
    """A stable, unique id per box of hazard_boxes: its piece's name (names[k], else prefix + k), and for a
    piece's second, third ... box that name + "#1", "#2" ... (box_base gives the piece's name back)."""
    out, seen = [], {}
    for k in kept:
        base = str(names[k]) if names is not None and k < len(names) else f"{prefix}{k}"
        j = seen.get(k, 0)
        seen[k] = j + 1
        out.append(base if j == 0 else f"{base}#{j}")
    return out


def box_base(box_id: str) -> str:
    """The piece's name of a box_ids id."""
    return str(box_id).split("#", 1)[0]


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
