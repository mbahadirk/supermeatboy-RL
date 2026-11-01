"""Train agents inside the real Super Meat Boy (personal use). No clone: every agent is a real Meat Boy
character in the running game (realgame.js), so it collides with, dies on and finishes exactly like Meat Boy.

Start the game, enter a level, then from the project root:

    python smbrl/realgame.py --agents 40                     # train on whatever level the game shows
    python smbrl/realgame.py --agents 40 --load smbrl/runs/real_4-1   # continue a saved agent
    python smbrl/realgame.py --agents 8 --test               # no learning: scripted inputs (checks the mod)

Waves: all agents spawn together after the level (saws, platforms, crumble blocks) is reset; each has one
life and waits where it died; the next wave starts when every agent is dead or finished, when --end-after
agents (default 5) have finished, or after --wave-seconds. The camera follows the agent furthest along its
path. The timer sign shows the best finishing time on this level. S in the game (window focused): save the agent and finish the level the game's way;
training continues on the next level the game loads. Ctrl+C (or a file runs/<name>/STOP): save and detach.
Agents: the game keeps a pool of 40 extra Meat Boys per level; with --agents above 40, more are built the same
way when a level loads (up to 200; the level already on screen when the trainer attaches has only 40).

Reward paths (realpath.py): when a level loads, up to --paths different short routes from the spawn to the
goal (picking up the level's keys on the way) that stay clear of solid tiles and lethal objects (rotated
boxes with the game's own sizes; moving hazards everywhere they were seen), plus one through the bandage,
are found and drawn into the game (green, blue, orange; bandage pink). Agents are split over the routes;
each is rewarded for new progress (share of the route's length) along whichever route it is furthest
along (only points it can see count), so it can leave a hard route for an easier one; it sees the direction
of that route ahead and is lightly pulled toward it. A route nobody gets far on for many waves keeps only a
couple of agents. A teleport seen 3 times from the same place to the same place (a portal) is remembered
and the routes are rebuilt through it.

Keys are per agent: an agent picks up a key by touching it (+R_KEY), and only that agent can then pass the
locks that key opens (realgame.js opens them during that agent's update only). Lock groups: connected lock
tiles, each opened by the nearest key.

The level map is live: tiles that change in the game (crumble blocks breaking, locks opening) change in the
agents' view and in the monitor. Agents also see the two nearest moving hazards (where, how fast).

Phases per level: FINISH until some agent reaches the goal (progress and the finish matter most), then
SPEED: every frame costs more, a finish pays by how close it is to the best time, a new record pays extra,
and half the agents move to the route of the fastest finisher.

Hops: starting a jump from the ground costs a little (R_HOP) and running on the ground toward the route
pays a little (R_RUN), so agents run instead of hopping all the time.

Standing still: an agent that stays within a tile of one spot for STILL_KILL seconds is killed (a death,
R_DEATH; the game script stops it like any dead agent).

Progress pays both ways: every frame the change of the agent's progress (closeness to the routes) is paid,
so going back costs what going forward paid and turning back is never free. Fitness uses the best progress.

Stuck: an agent with no new best progress and no newly visited block for more than STALL_GRACE seconds pays
R_STALL every frame (rising to twice that), and every jump from the ground after the first HOP_FREE costs
R_HOP_SPAM more (both together at most STALL_CAP per stuck stretch, less than a death). Exploring: the first visit in a life to each 2x2-tile block pays
R_EXPLORE, less for blocks many agents have already been to on this level (1 / sqrt(1 + visits per agent)), so
unvisited corners and other ways pay most. Agents see how long they have been stuck and how often they jumped
there (the last two inputs).

The monitor window (monitor.py; --no-monitor to skip) shows the level, routes and agents, the leader's
policy output, learning curves (finishes, progress, fitness, time, hops, losses, entropy, KL, explained
variance) and the network (the previous wave's best agent). Its sliders set the number of agents and how
they are split over the routes (from the next wave).

Saved in runs/<name>/: model.pt (latest), levels/<level>/best.pt + best.json (fastest finish).
"""
from __future__ import annotations

import math
import os
import sys
import json
import time
import argparse
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from realpath import Planner, fitness, PATH_RGB, box_distance, tile_classes

HERE = os.path.dirname(os.path.abspath(__file__))
JS = os.path.join(HERE, "realgame.js")
GAME_EXE = "SuperMeatBoy.exe"
HZ = 60
TILE = 20.0
MAX_AGENTS = 200        # the game keeps 40 spare Meat Boys per level; more are built at the level load
GAME_POOL = 40

# actions: (left, right, jump, run)
ACTIONS = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0),
           (0, 1, 0, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 0, 1, 1)]
K = 5                                   # local map radius (cells): (2K+1)^2 values
N_STATE = 10
N_PATH = 6                              # own route: points 2 and 6 tiles ahead, progress, distance from it
N_HAZ = 8                               # the two nearest moving hazards: offset and velocity
N_WALL = 3                              # wall touching on the left, on the right (tile map), sliding down a wall
N_FX = 6                                # wind here (x, y), in a wind column, nearest orb (dx, dy), inside an orb
N_SURF = (2 * K + 1) ** 2               # second local map: what kind of surface each solid cell offers
N_STUCK = 2                             # time stuck in one spot (/ STALL_GRACE + STALL_RAMP), jumps there (/ 5)
OBS = N_STATE + (2 * K + 1) ** 2 + N_PATH + N_HAZ + N_WALL + N_FX + N_SURF + N_STUCK
# surface codes (second local map): which sides of a solid cell are open
S_FLOOR, S_WALL, S_LEDGE, S_CEIL, S_CEIL_WALL, S_INSIDE = 1.0, -1.0, 0.5, -0.5, -0.75, 0.25
WALL_SLIDE = 9                          # character state while sliding down a wall
C_EMPTY, C_SOLID, C_HAZARD, C_GOAL = 0.0, 1.0, -1.0, 0.75
C_WIND, C_ORB, C_ACCEL = 0.3, -0.4, 0.55  # map codes: in a fan's wind column, inside an orb, accelerator tile
ACCEL_TILE = 98                           # Salt Factory surface tile (no special bit); what it does is not traced

# rewards
R_PROGRESS = 1.0       # per tile of progress toward the goal, both ways: going back costs what going forward paid
                       # (so turning back is never free and back-and-forth earns nothing). Progress is measured by
                       # closeness to the routes (realpath FIELD_*; the furthest route at the agent's spot): anywhere
                       # near a route counts, not only on the line
R_TETHER = -0.003      # per tile away from the route beyond TETHER_FREE tiles (up to 6 more), every frame
TETHER_FREE = 6.0      # tiles: within this an agent may leave the line freely
R_DEATH = -15.0        # dying (also: killed for standing still, STILL_KILL)
R_WIN = 50.0
R_FAST = 20.0          # FINISH phase: + this x (1 - finish time / wave length)
R_STEP = -0.002        # FINISH phase, every frame
R_SPEED = 60.0         # SPEED phase: + this x (best time / finish time), at most 1.5x
R_RECORD = 25.0        # SPEED phase: a new best time
R_STEP_SPEED = -0.01   # SPEED phase, every frame
R_HOP = -0.08          # starting a jump from the ground: stops constant short hops (a jump that is needed still
                       # pays for itself through progress)
R_RUN = 0.004          # every frame on the ground: x horizontal speed toward the route (/400 units/s, up to 1.5)
R_KEY = 10.0           # picking up a key (each key once per life)
STILL_KILL = 1.0       # seconds: an agent that stays within STILL_R of one spot this long is killed (R_DEATH)
STILL_R = 1.0 * 20.0   # game units (one tile)
WAYPOINT_R = 1.5 * 20.0  # touching a key / bandage (game units): the route after it starts counting
R_STALL = -0.02        # every frame stuck: no new best progress and no new block (exploring) for STALL_GRACE.
                       # Stepping away does not end it (it used to, which made walking back the way to wait)
STALL_GRACE = 1.5      # seconds (waiting for a saw to pass is fine); the cost doubles over the next STALL_RAMP
STALL_RAMP = 3.0       # seconds
R_HOP_SPAM = -0.15     # extra for every jump from the ground while stuck, after the first HOP_FREE
HOP_FREE = 2
STALL_CAP = 4.0        # at most this much (stuck + jumping) per stuck stretch: less than R_DEATH, so dying is never the
                       # cheaper way out
R_EXPLORE = 0.25       # first visit this life to a 2x2-tile block, / sqrt(1 + visits there per agent on this level)
EXPLORE_BLOCK = 2      # tiles


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


def hazard_boxes(hz: list, keep=None) -> tuple[list, list]:
    """Lethal pieces [x, y, size w, size h, scale x, scale y, angle] (realgame.js; size from the game's
    FUN_004ca860 at scale 1) as rotated boxes (x, y, half w, half h, angle). keep: which ones are real hazards
    (not decoration). Returns the boxes and the indices of the entries kept."""
    out, kept = [], []
    for k, h in enumerate(hz):
        if keep is not None and k < len(keep) and not keep[k]:
            continue
        x, y, w, hgt, sx, sy, ang = h
        sx, sy = abs(sx or 1.0), abs(sy or 1.0)
        if w <= 0 and hgt <= 0:
            w = hgt = 24.0
        out.append((x, y, 0.5 * w * sx, 0.5 * hgt * sy, ang))
        kept.append(k)
    return out, kept


def route_rgb(k: int, p) -> tuple:
    """Route colour: green / blue / orange for the main routes, pink for the bandage route."""
    return PATH_RGB[3] if getattr(p, "kind", "main") == "bandage" else PATH_RGB[k % 3]


def argb(rgb, alpha=0xD8) -> int:
    r, g, b = rgb
    return (alpha << 24) | (r << 16) | (g << 8) | b


def other_trainers() -> list[str]:
    """PIDs of other realgame.py processes (two trainers in one game break each other)."""
    import subprocess
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                              "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    me = str(os.getpid())
    return [ln.split("|", 1)[0] for ln in out.splitlines()
            if "realgame.py" in ln and "|" in ln and ln.split("|", 1)[0].strip() != me]


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


def observe(lv: Level, grid: np.ndarray, s: list, path, prog: float, off: float, haz=None, opened=(),
            stuck=(0.0, 0.0)) -> np.ndarray:
    x, y, vx, vy, flags = s[:5]
    gx, gy = lv.goals[0] if lv.goals else lv.spawn
    d = lv.goal_dist(x, y)
    c, r = lv.cell(x, y)
    pad = np.pad(grid, K, constant_values=C_SOLID)
    rr, cc = min(max(r, -K), lv.H + K - 1) + K, min(max(c, -K), lv.W + K - 1) + K
    patch = pad[rr - K:rr + K + 1, cc - K:cc + K + 1]
    if patch.shape != (2 * K + 1, 2 * K + 1):
        patch = np.full((2 * K + 1, 2 * K + 1), C_SOLID, dtype=np.float32)
    if opened:                                     # locks this agent's keys opened (open for it alone)
        patch = patch.copy()
        for lr, lc in opened:
            pr, pc = lr + K - (rr - K), lc + K - (cc - K)
            if 0 <= pr < 2 * K + 1 and 0 <= pc < 2 * K + 1:
                patch[pr, pc] = C_EMPTY
    vec = np.array([
        (x - lv.ox) / (lv.W * TILE), (y - lv.oy) / (lv.H * TILE), vx / 400.0, vy / 600.0,
        0.0 if flags & 2 else 1.0,                 # on the ground (bit 1 = airborne)
        1.0 if flags & 0x200 else 0.0,             # on a wall
        -1.0 if flags & 1 else 1.0,                # facing
        (gx - x) / (lv.W * TILE), (gy - y) / (lv.H * TILE), min(d / lv.diag, 2.0),
    ], dtype=np.float32)
    if path is not None:                           # where the agent's own route goes next
        ax, ay = path.point_at(prog + 2 * TILE)
        bx, by = path.point_at(prog + 6 * TILE)
        pv = np.clip(np.array([(ax - x) / (10 * TILE), (ay - y) / (10 * TILE), (bx - x) / (10 * TILE),
                               (by - y) / (10 * TILE), prog / path.length, off / (5 * TILE)], np.float32), -2, 2)
    else:
        pv = np.zeros(N_PATH, np.float32)
    hv = haz if haz is not None else np.zeros(N_HAZ, np.float32)
    wfx, wfy = lv.wind_at(x, y)
    fv = np.zeros(N_FX, np.float32)
    fv[0], fv[1], fv[2] = np.clip(wfx / 2000.0, -2, 2), np.clip(wfy / 2000.0, -2, 2), 1.0 if (wfx or wfy) else 0.0
    if lv.orbs:
        ox_, oy_, r1, r2 = min(lv.orbs, key=lambda o: (o[0] - x) ** 2 + (o[1] - y) ** 2)[:4]
        fv[3], fv[4] = np.clip((ox_ - x) / 200.0, -2, 2), np.clip((oy_ - y) / 200.0, -2, 2)
        fv[5] = 1.0 if np.hypot(ox_ - x, oy_ - y) <= max(r1, r2) else 0.0
    wv = np.array([wall_touch(lv, x, y, -1, opened), wall_touch(lv, x, y, 1, opened),
                   1.0 if len(s) > 5 and s[5] == WALL_SLIDE else 0.0], np.float32)
    sp = np.pad(lv.surfaces(), K, constant_values=S_INSIDE)[rr - K:rr + K + 1, cc - K:cc + K + 1]
    if sp.shape != (2 * K + 1, 2 * K + 1):
        sp = np.full((2 * K + 1, 2 * K + 1), S_INSIDE, dtype=np.float32)
    return np.concatenate([vec, patch[::-1].ravel(), pv, hv, wv, fv, sp[::-1].ravel(),   # patch rows top-down
                           np.asarray(stuck, np.float32)])


def wall_touch(lv: Level, x: float, y: float, side: int, opened=()) -> float:
    """1 if a solid tile is right beside the body (within 2 units of its edge) on that side, at its middle,
    upper or lower part: the wall a wall jump pushes off."""
    for dy in (-4.0, 0.0, 4.0):
        c, r = lv.cell(x + side * 8.0, y + dy)
        if 0 <= c < lv.W and 0 <= r < lv.H and lv.solid[r, c] and (r, c) not in opened:
            return 1.0
    return 0.0


def moving_features(x: float, y: float, pos: np.ndarray, vel: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """The two nearest moving hazards within 10 tiles: offset (/200) and velocity (/300) each, zeros if none."""
    out = np.zeros(N_HAZ, np.float32)
    if pos is None or not len(pos) or not moving.any():
        return out
    d = np.hypot(pos[:, 0] - x, pos[:, 1] - y)
    d = np.where(moving & (d < 10 * TILE), d, np.inf)
    for j, k in enumerate(np.argsort(d)[:2]):
        if not np.isfinite(d[k]):
            break
        out[4 * j:4 * j + 4] = [(pos[k, 0] - x) / 200.0, (pos[k, 1] - y) / 200.0, vel[k, 0] / 300.0, vel[k, 1] / 300.0]
    return np.clip(out, -2.0, 2.0)


# ---- policy + PPO ---------------------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self, obs=OBS, n=len(ACTIONS), h=256):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n))
        self.vf = nn.Sequential(nn.Linear(obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def dist(self, o):
        return torch.distributions.Categorical(logits=self.pi(o))


def load_policy(policy: Policy, path: str):
    """Load a saved model; one from before the route inputs existed keeps its weights for the old inputs and
    starts the new input columns small."""
    sd = torch.load(path, map_location="cpu")["policy"]
    own = policy.state_dict()
    widened = False
    for k, v in sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif v.dim() == 2 and own[k].shape[0] == v.shape[0] and v.shape[1] < own[k].shape[1]:
            own[k][:, :v.shape[1]] = v
            own[k][:, v.shape[1]:] *= 0.1
            widened = True
    policy.load_state_dict(own)
    return widened


class PPO:
    def __init__(self, policy: Policy, lr=3e-4, gamma=0.995, lam=0.95, clip=0.2, epochs=4, mb=2048, ent=0.01):
        self.p = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma, self.lam, self.clip, self.epochs, self.mb, self.ent = gamma, lam, clip, epochs, mb, ent
        self.batch = []          # finished trajectories: dicts of arrays
        self.steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy=False):
        o = torch.as_tensor(obs)
        d = self.p.dist(o)
        a = d.probs.argmax(-1) if greedy else d.sample()
        return a.numpy(), d.log_prob(a).numpy(), self.p.vf(o).squeeze(-1).numpy(), d.probs.numpy()

    def add(self, traj: dict, last_value: float):
        r, v, done = np.array(traj["r"], np.float32), np.array(traj["v"], np.float32), traj["done"]
        n = len(r)
        adv = np.zeros(n, np.float32)
        nxt, gae = (0.0 if done else last_value), 0.0
        for t in range(n - 1, -1, -1):
            delta = r[t] + self.gamma * nxt - v[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t], nxt = gae, v[t]
        self.batch.append({"o": np.array(traj["o"], np.float32), "a": np.array(traj["a"]),
                           "lp": np.array(traj["lp"], np.float32), "adv": adv, "ret": adv + v, "v": v})
        self.steps += n

    def update(self) -> dict:
        cat = lambda k: torch.as_tensor(np.concatenate([b[k] for b in self.batch]))
        o, a, lp0, adv, ret, v0 = cat("o"), cat("a"), cat("lp"), cat("adv"), cat("ret"), cat("v")
        ev = float(1.0 - torch.var(ret - v0) / (torch.var(ret) + 1e-8))      # how much of the return the critic explained
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(a)
        acc = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": [], "gnorm": []}
        for _ in range(self.epochs):
            for idx in torch.randperm(n).split(self.mb):
                d = self.p.dist(o[idx])
                logr = d.log_prob(a[idx]) - lp0[idx]
                ratio = torch.exp(logr)
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                vl = ((self.p.vf(o[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                en = d.entropy().mean()
                loss = pg + 0.5 * vl - self.ent * en
                self.opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.p.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    acc["pg"].append(pg.item()); acc["vf"].append(vl.item()); acc["ent"].append(en.item())
                    acc["kl"].append(((ratio - 1) - logr).mean().item())
                    acc["clip"].append(((ratio - 1).abs() > self.clip).float().mean().item())
                    acc["gnorm"].append(float(gn))
        self.batch = []
        return {k: float(np.mean(v)) for k, v in acc.items()} | {"ev": ev, "samples": n}


# ---- session --------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, args):
        self.args = args
        self.run_dir = os.path.join(HERE, "runs", args.name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.policy = Policy()
        if args.load:
            p = args.load if args.load.endswith(".pt") else os.path.join(args.load, "model.pt")
            widened = load_policy(self.policy, p)
            print(f"loaded {p}" + (" (older model: the route inputs start untrained)" if widened else ""))
        self.ppo = PPO(self.policy)
        self.level: Level | None = None
        self.traj, self.best_d, self.acts, self.alive = [], [], [], []
        self.visits = {}
        self.best_time = None
        self.waves = 0
        self.t0 = time.time()
        self.script = None
        self.stopping = False
        self.stopped = threading.Event()
        # shown by the monitor
        self.paths, self.path_of, self.boxes, self.last_states = [], [], [], []
        self.links, self.route_stats = [], []
        self.phase, self.best_route, self.leader, self.wave_no, self.frame_no = "finish", None, 0, 0, 0
        self.leader_probs, self.leader_value = None, None
        self.history, self.updates = deque(maxlen=400), deque(maxlen=400)
        self.best_net = None             # the network and the previous wave's best agent's last input (monitor)
        self.game_paths_on = True
        # set from the monitor: agents per wave (applied from the next wave) and route shares (None = automatic)
        self.pool = max(GAME_POOL, min(args.agents, MAX_AGENTS))   # characters per level (game's 40 + extra)
        self.want_agents = max(1, min(args.agents, self.pool))
        self.path_share = None

    # level / waves
    def on_info(self, info: dict, tiles: bytes):
        if info["W"] * info["H"] == 0 or len(tiles or b"") != info["W"] * info["H"] * 4:
            print(f"[level] ignoring invalid level data {info['W']}x{info['H']}", flush=True)
            return
        if info["name"] in (None, "unknown"):
            info["name"] = identify_level(info, tiles) or "unknown"
        self.level = Level(info, tiles)
        self.best_time = None
        self.phase, self.best_route = "finish", None
        self.paths, self.hz_start, self.moving, self.sweep, self.path_dirty = [], None, set(), set(), True
        self.paths_built, self.rebuilds, self.built_wave = False, 0, 0
        self.best_fit = None
        self.links, self.route_stats, self.jumps_seen = [], [], {}
        self.hz_prev, self.hz_vel = {}, None
        self.visits = {}                   # 2x2-tile block -> agent visits on this level (exploration bonus)
        n = info["agents"]
        lv = self.level
        decor = sum(1 for k in lv.hz_keep if not k)
        print(f"[level] {info['name']}: {lv.W}x{lv.H}, {len(lv.goals)} goal(s), {info['lethal']} lethal pieces"
              + (f" ({decor} of them decoration)" if decor else "")
              + (f", {len(lv.keys)} key(s)" if lv.keys else "") + (f", {len(lv.bandages)} bandage(s)" if lv.bandages else "")
              + (f", {int(lv.special.sum())} crumble/lock/switch tiles" if lv.special.any() else "")
              + (f", {len(lv.fans)} fan(s)" if lv.fans else "") + (f", {len(lv.orbs)} orb(s)" if lv.orbs else "")
              + (f", {len(lv.portals)} portal pair(s)" if lv.portals else "")
              + (f", {int(lv.accel.sum())} accelerator tiles" if lv.accel.any() else "")
              + f", {n} agents", flush=True)
        self.new_wave(n)
        if lv.keys:
            print(f"[keys] {lv.name}: " + ", ".join(f"key {k + 1} opens {len(g)} lock tile(s)"
                                                   for k, g in enumerate(lv.lock_groups)), flush=True)
        if "hz" in info:                     # the game waits for this (no agent moves meanwhile)
            self.boxes, kept = hazard_boxes(info["hz"], lv.hz_keep)
            self.track_hazards(self.boxes, [lv.hz_ids[k] for k in kept])
            self.build_paths(reset=False)

    def assign_paths(self, n: int) -> list[int]:
        k = len(self.paths)
        if k == 0:
            return [0] * n
        if self.path_share is not None:      # set in the monitor: agents per route in proportion to the shares
            w = [max(0.0, float(self.path_share[j])) if j < len(self.path_share) else 0.0 for j in range(k)]
            if sum(w) <= 0:
                w = [1.0] * k
            exact = [n * x / sum(w) for x in w]
            cnt = [int(e) for e in exact]
            for j in sorted(range(k), key=lambda j: exact[j] - cnt[j], reverse=True)[:n - sum(cnt)]:
                cnt[j] += 1
            return [j for j in range(k) for _ in range(cnt[j])]
        # automatic: main routes 1 each, the bandage route 0.5; a route nobody has got far on for many waves
        # (stalled) keeps only a little; in the SPEED phase half the agents take the fastest finisher's route
        w = [(0.5 if p.kind == "bandage" else 1.0) * (0.15 if self.stalled(j) else 1.0) for j, p in enumerate(self.paths)]
        out = []
        if self.phase == "speed" and self.best_route is not None and self.best_route < k:
            out = [self.best_route] * (n // 2)
        m = n - len(out)
        exact = [m * x / sum(w) for x in w]
        cnt = [int(e) for e in exact]
        for j in sorted(range(k), key=lambda j: exact[j] - cnt[j], reverse=True)[:m - sum(cnt)]:
            cnt[j] += 1
        if m >= k:                            # every route keeps at least one agent
            for j in range(k):
                if cnt[j] == 0:
                    cnt[max(range(k), key=lambda q: cnt[q])] -= 1
                    cnt[j] = 1
        mix = [j for j in range(k) for _ in range(cnt[j])]
        mix = [mix[(i * 7) % len(mix)] for i in range(len(mix))] if len(mix) and len(mix) % 7 else mix
        return out + mix

    def stalled(self, j: int) -> bool:
        if j >= len(self.route_stats) or len(self.route_stats) < 2:
            return False
        st = self.route_stats[j]
        top = max(r["best"] for r in self.route_stats)
        return self.waves - st["improved"] >= 12 and st["best"] < top - 0.15

    def new_wave(self, n: int):
        self.traj = [{"o": [], "a": [], "lp": [], "v": [], "r": [], "done": False} for _ in range(n)]
        self.best_d = [None] * n
        self.best_s = [None] * n       # best progress along the agent's route this life (game units)
        self.cur_s = [0.0] * n
        self.cur_off = [0.0] * n
        self.frames = [0] * n
        self.fit = [None] * n
        self.acts = [[] for _ in range(n)]
        self.alive = [True] * n
        self.ground_at_act = [False] * n   # on the ground when the last action was chosen
        self.hops = [0] * n
        self.prev_xy = [None] * n          # last frame's position (teleport = portal)
        self.best_k = [[None] * len(self.paths) for _ in range(n)]   # best progress on each route
        self.frac = [None] * n             # best share of any route's length so far (what progress pays for)
        self.keys_of = [0] * n
        self.anchor = [None] * n           # frame of the agent's last new best progress / new block (stuck timer)
        self.hops_here = [0] * n           # jumps from the ground since then
        self.phi = [None] * n              # progress at the agent's spot last frame (share of the shortest route)
        self.stall_paid = [0.0] * n        # stuck costs paid there (up to STALL_CAP)
        self.seen = [set() for _ in range(n)]   # blocks visited this life
        self.reached = [set() for _ in range(n)]   # key / bandage positions touched this life (route legs)
        self.still_at = [None] * n         # (x, y, frame) where the agent last stood still (STILL_KILL)
        self.killed = 0
        self.stuck_frames, self.new_blocks = 0, 0
        self.path_of = self.assign_paths(n)

    def route(self, i: int):
        return self.paths[self.path_of[i]] if self.paths and i < len(self.path_of) else None

    def on_frame(self, m: dict) -> dict:
        lv, states = self.level, m["s"]
        if lv is None:
            return {"a": [[0, 0, 0, 0]] * len(states), "leader": 0}
        n = len(states)
        if len(self.traj) != n:
            self.new_wave(n)
        if m.get("tc"):
            lv.apply_tiles(m["tc"])
        if m.get("fx"):
            lv.set_fields(m["fx"])
        if m.get("lt"):                      # pieces came or went (shots): the hazard list changed
            before = len(lv.hz_ids)
            lv.set_lethal(m["lt"]["types"], m["lt"]["ids"])
            lv.hz_changes = getattr(lv, "hz_changes", 0) + 1
            if lv.hz_changes <= 5:
                print(f"[hazards] {lv.name}: pieces came or went, {before} -> {len(lv.hz_ids)} lethal "
                      f"(frame {m['frame']})", flush=True)
        boxes, kept = hazard_boxes(m["hz"], lv.hz_keep)
        ids = [lv.hz_ids[k] if k < len(lv.hz_ids) else str(k) for k in kept]
        for sid, x, y, rad in m.get("sh", []):   # flying shots (seen by the agents, not planned around)
            boxes.append((x, y, rad, rad, 0.0))
            ids.append(f"shot{sid}")
        if m.get("sh") and not getattr(lv, "shots_seen", False):
            lv.shots_seen = True
            print(f"[hazards] {lv.name}: shots in the air (thrown saws / missiles) are now seen by the agents", flush=True)
        self.boxes = boxes
        self.track_hazards(boxes, ids)
        pos = np.array([b[:2] for b in boxes], np.float32).reshape(-1, 2)
        self.hz_vel = np.zeros_like(pos)
        if m["frame"] > 0:                   # velocity of each piece from its last position (by id)
            for k, hid in enumerate(ids):
                q = self.hz_prev.get(hid)
                if q is not None:
                    self.hz_vel[k] = (pos[k] - q) * HZ
        self.hz_prev = {hid: pos[k] for k, hid in enumerate(ids)}
        moving = np.hypot(self.hz_vel[:, 0], self.hz_vel[:, 1]) > 8.0 if len(pos) else np.zeros(0, bool)
        if not self.paths_built:             # (normally built when the level info came)
            self.build_paths(reset=m["frame"] != 0)
        grid = lv.grid_with_hazards(boxes)
        self.last_states, self.frame_no, self.wave_no = states, m["frame"], m["wave"]
        if self.args.debug_spawn and m["frame"] in (0, 1, 30):
            sx, sy = lv.spawn
            far = [(i, s) for i, s in enumerate(states) if abs(s[0] - sx) > 30 or abs(s[1] - sy) > 30]
            print(f"  [spawn f{m['frame']}] spawn=({sx:.0f},{sy:.0f}) {len(far)} away: " + " ".join(
                f"a{i}=({s[0]:.0f},{s[1]:.0f} st{s[5]} fl{s[4]:x})" for i, s in far[:6]), flush=True)
        if self.args.test and m["frame"] % 120 == 0:
            print(f"  f{m['frame']}: " + " ".join(f"({s[0]:.0f},{s[1]:.0f} v{s[2]:.0f},{s[3]:.0f} st{s[5]} {s[6]})"
                                               for s in states[:4]), flush=True)
        # where every agent is on the routes (only points it can see count): its own route every frame, the
        # others every 6th; it follows whichever route it is furthest along
        new_frac = [None] * n             # best share of any route so far (fitness, stuck timer)
        new_phi = [None] * n              # share at the agent's spot now (the progress reward, both ways)
        for i, s in enumerate(states):
            if not self.alive[i]:
                continue
            if self.prev_xy[i] is not None and s[6] == 0 and m["frame"] > 1:
                self.check_teleport(self.prev_xy[i], (s[0], s[1]))
            self.prev_xy[i] = (s[0], s[1]) if s[6] == 0 else None
            p = self.route(i)
            if p is None:
                continue
            bk, own = self.best_k[i], self.path_of[i]
            if len(bk) != len(self.paths):
                bk = self.best_k[i] = [None] * len(self.paths)
            for w in p.pl.keys + p.pl.bandages:               # keys / bandages touched: the next leg of the routes
                if w not in self.reached[i] and math.hypot(s[0] - w[0], s[1] - w[1]) < WAYPOINT_R:
                    self.reached[i].add(w)
            rc = self.reached[i]
            vals = [q.value(s[0], s[1], rc) for q in self.paths]   # progress on every route at this spot
            for j, v in enumerate(vals):
                if v is not None:
                    bk[j] = v if bk[j] is None else max(bk[j], v)
            fr = [v / q.length if v is not None else -1.0 for v, q in zip(vals, self.paths)]
            j = int(np.argmax(fr))
            if j != own and fr[j] > fr[own] + 0.03:          # further along another route here: follow that one
                self.path_of[i] = own = j
            if vals[own] is not None:
                self.cur_s[i] = vals[own]
            self.cur_off[i] = self.paths[own].offset(s[0], s[1])
            new_phi[i] = max(fr) if max(fr) >= 0.0 else None   # progress here (None: no route near)
            new_frac[i] = max((b or 0.0) / q.length for b, q in zip(bk, self.paths))
            self.best_s[i] = bk[own]
        # finish the previous step: rewards for what happened since the last frame
        step_cost = R_STEP_SPEED if self.phase == "speed" else R_STEP
        kills = []                                             # agents killed for standing still (sent to the game)
        for i, s in enumerate(states):
            tr = self.traj[i]
            if not self.alive[i] or not tr["a"]:
                continue
            x, y, st = s[0], s[1], s[6]
            sa = self.still_at[i]                              # standing still for STILL_KILL: killed (a death)
            if sa is None or math.hypot(x - sa[0], y - sa[1]) > STILL_R:
                self.still_at[i] = (x, y, m["frame"])
            elif st == 0 and m["frame"] - sa[2] >= STILL_KILL * HZ:
                st = 1
                kills.append(i)
            r = step_cost
            self.frames[i] += 1
            p = self.route(i)
            moved = False                                      # new progress this frame (ends being stuck)
            if p is not None:
                if new_phi[i] is not None:                     # progress both ways (share x shortest route)
                    if self.phi[i] is not None:
                        r += R_PROGRESS * (new_phi[i] - self.phi[i]) * self.paths[0].length / TILE
                    self.phi[i] = new_phi[i]
                if new_frac[i] is not None:                    # best so far: fitness, the stuck timer
                    moved = self.frac[i] is not None and new_frac[i] - self.frac[i] > 1e-4
                    self.frac[i] = new_frac[i] if self.frac[i] is None else max(self.frac[i], new_frac[i])
                r += R_TETHER * min(max(0.0, self.cur_off[i] / TILE - TETHER_FREE), 6.0)
                if not (s[4] & 2):                             # on the ground: running toward the route pays
                    ax, _ = p.point_at(self.cur_s[i] + 2 * TILE)
                    r += R_RUN * min(1.5, max(0.0, s[2] * (1.0 if ax >= x else -1.0) / 400.0))
            else:                                              # no route: closest path distance to the goal
                d = lv.goal_dist(x, y)
                if self.phi[i] is not None:                    # both ways here too
                    r += R_PROGRESS * (self.phi[i] - d)
                self.phi[i] = d
                if self.best_d[i] is None or d < self.best_d[i]:
                    moved = self.best_d[i] is not None and d < self.best_d[i] - 0.5
                    self.best_d[i] = d
            got = (s[7] if len(s) > 7 else 0) & ~self.keys_of[i]  # keys picked up since the last frame
            if got:
                r += R_KEY * bin(got).count("1")
                self.keys_of[i] |= got
            # exploring: the first visit this life to a block, more where few agents have been
            blk = (int((x - lv.ox) // (EXPLORE_BLOCK * TILE)), int((y - lv.oy) // (EXPLORE_BLOCK * TILE)))
            if blk not in self.seen[i]:
                if self.seen[i]:                               # (not the spawn block)
                    v = self.visits.get(blk, 0)
                    r += R_EXPLORE / math.sqrt(1.0 + v / max(1, n))
                    self.new_blocks += 1
                    moved = True
                self.seen[i].add(blk)
                self.visits[blk] = self.visits.get(blk, 0) + 1
            # stuck: no new best progress and no new block for STALL_GRACE (stepping away does not end it)
            if self.anchor[i] is None or moved:
                self.anchor[i], self.hops_here[i], self.stall_paid[i] = m["frame"], 0, 0.0
            pen = 0.0
            t = (m["frame"] - self.anchor[i]) / HZ - STALL_GRACE
            if t > 0:
                pen += -R_STALL * (1.0 + min(1.0, t / STALL_RAMP))
                self.stuck_frames += 1
            a_now = tr["a"][-1]                                # a jump started from the ground (a hop) costs
            if ACTIONS[a_now][2] and self.ground_at_act[i] and not (len(tr["a"]) > 1 and ACTIONS[tr["a"][-2]][2]):
                r += R_HOP
                self.hops[i] += 1
                self.hops_here[i] += 1
                if self.hops_here[i] > HOP_FREE:               # jumping again and again, getting nowhere
                    pen += -R_HOP_SPAM
            pen = min(pen, STALL_CAP - self.stall_paid[i])
            if pen > 0:
                r -= pen
                self.stall_paid[i] += pen
            if st == 1:
                r += R_DEATH
            elif st == 2:
                r += self.finish_reward(m["frame"])
            tr["r"].append(r)
            if st != 0:
                tr["done"] = True
                self.alive[i] = False
                self.ppo.add(tr, 0.0)
                self.fit[i] = self.agent_fitness(i, won=st == 2, dead=st == 1)
        live = [i for i, s in enumerate(states) if s[6] == 0 and self.alive[i]]
        acts = [[0, 0, 0, 0]] * n
        leader = 0
        if live:
            obs = np.stack([observe(lv, grid, states[i], self.route(i), self.cur_s[i], self.cur_off[i],
                                    moving_features(states[i][0], states[i][1], pos, self.hz_vel, moving),
                                    lv.opened_cells(self.keys_of[i]) if self.keys_of[i] else (),
                                    self.stuck_inputs(i, m["frame"])) for i in live])
            if self.args.test:
                a = np.array([self.scripted(m["frame"], i) for i in live])
                lp = v = np.zeros(len(live), np.float32)
                probs = None
            else:
                a, lp, v, probs = self.ppo.act(obs)
            for j, i in enumerate(live):
                tr = self.traj[i]
                if self.best_d[i] is None:
                    self.best_d[i] = lv.goal_dist(states[i][0], states[i][1])
                tr["o"].append(obs[j]); tr["a"].append(int(a[j])); tr["lp"].append(float(lp[j])); tr["v"].append(float(v[j]))
                acts[i] = list(ACTIONS[int(a[j])])
                self.ground_at_act[i] = not (states[i][4] & 2)
                self.acts[i].append(int(a[j]))
            if self.paths:
                leader = max(live, key=lambda i: self.frac[i] or 0.0)
            else:
                leader = min(live, key=lambda i: lv.goal_dist(states[i][0], states[i][1]))
            j = live.index(leader)
            self.leader_probs = probs[j] if probs is not None else None
            self.leader_value = float(v[j])
        won = [i for i, s in enumerate(states) if s[6] == 2]
        if won:
            leader = won[0]
        self.leader = leader
        self.killed += len(kills)
        return {"a": acts, "leader": leader, "kill": kills}

    def stuck_inputs(self, i: int, frame: int) -> tuple[float, float]:
        an = self.anchor[i]
        if an is None:
            return 0.0, 0.0
        return (min(1.0, (frame - an) / HZ / (STALL_GRACE + STALL_RAMP)), min(1.0, self.hops_here[i] / 5.0))

    def finish_reward(self, frame: int) -> float:
        t = (frame + 1) / HZ
        if self.phase == "speed" and self.best_time:
            return R_WIN + R_SPEED * min(1.5, self.best_time / t) + (R_RECORD if t < self.best_time - 1e-6 else 0.0)
        return R_WIN + R_FAST * max(0.0, 1.0 - frame / (self.args.wave_seconds * HZ))

    def scripted(self, f: int, i: int) -> int:
        """--test: run right, jumping at agent-specific intervals (checks deaths, walls, goal, waves)."""
        period = 20 + 7 * i
        return 7 if (f % period) < 10 else 6

    def on_wave_end(self, m: dict):
        lv = self.level
        self.waves += 1
        n = len(self.traj)
        for i in range(n):           # outcomes of the wave's last frame; agents alive at the time limit are cut
            tr = self.traj[i]
            if not (self.alive[i] and tr["a"]):
                continue
            st = m["status"][i] if i < len(m["status"]) else 0
            self.fit[i] = self.agent_fitness(i, won=st == 2, dead=st == 1)
            step_cost = R_STEP_SPEED if self.phase == "speed" else R_STEP
            if st == 1:
                tr["r"].append(step_cost + R_DEATH)
                tr["done"] = True
                self.ppo.add(tr, 0.0)
            elif st == 2:
                tr["r"].append(step_cost + self.finish_reward(m["finish"][i] if m["finish"][i] >= 0 else m["frame"]))
                tr["done"] = True
                self.ppo.add(tr, 0.0)
            else:
                tr["r"].append(step_cost)
                with torch.no_grad():
                    lv_ = float(self.policy.vf(torch.as_tensor(tr["o"][-1])).item())
                self.ppo.add(tr, lv_)
        rets = [sum(tr["r"]) for tr in self.traj if tr["r"]]
        hops = 60.0 * sum(self.hops) / max(1, sum(self.frames))         # hops per agent-second
        self.snapshot_best(m["wave"])
        fin = [(f, i) for i, (st, f) in enumerate(zip(m["status"], m["finish"])) if st == 2 and f >= 0]
        dead = sum(1 for st in m["status"] if st == 1)
        line = f"[wave {m['wave']}] {lv.name if lv else '?'}: {len(fin)}/{n} finished, {dead} dead"
        line += f" ({self.killed} for standing still)" if self.killed else ""
        fastest = None
        if fin:
            f, i = min(fin)
            fastest = (f + 1) / HZ
            line += f", fastest {fastest:.2f}s (agent {i}, path {self.path_of[i] + 1})"
            if self.best_time is None or fastest < self.best_time:
                self.best_time = fastest
                self.best_route = self.path_of[i] if self.paths else None
                self.save_best(lv.name, fastest, i)
                self.script.post({"type": "best", "payload": fastest})
            if self.phase != "speed":
                self.phase = "speed"
                print(f"[phase] {lv.name}: an agent finished - now rewarding speed (best {self.best_time:.2f}s, "
                      f"half the agents on path {self.best_route + 1 if self.best_route is not None else '-'})", flush=True)
        prog = [100.0 * (self.frac[i] or 0.0) for i in range(n)] if self.paths else []
        self.update_route_stats(fin)
        fits = [f for f in self.fit if f is not None]
        best_fit = max(fits) if fits else None
        line += f", {hops:.1f} hops/s, stuck {100.0 * self.stuck_frames / max(1, sum(self.frames)):.0f}%, " \
                f"{self.new_blocks} new blocks"
        if best_fit is not None:
            line += f", fitness best {best_fit:.0f}" + (f", path {max(prog):.0f}% max / {np.mean(prog):.0f}% mean" if prog else "")
        self.history.append({"wave": m["wave"], "finished": len(fin), "dead": dead, "fastest": fastest, "fit": best_fit,
                             "path_max": max(prog) if prog else None, "path_mean": float(np.mean(prog)) if prog else None,
                             "ret": float(np.mean(rets)) if rets else None, "hops": hops})
        if not self.args.test and self.ppo.steps >= self.args.batch:
            st = self.ppo.update()
            self.ppo.steps = 0
            self.updates.append(st)
            line += (f"  | PPO {st['samples']} samples: policy {st['pg']:+.3f}, value {st['vf']:.2f}, "
                     f"entropy {st['ent']:.2f}, KL {st['kl']:.3f}, EV {st['ev']:.2f}")
            self.save()
        el = (time.time() - self.t0) / 60
        print(line + f"  [{el:.1f} min]", flush=True)
        self.new_wave(n)
        # moving hazards seen in new places: routes around their sweep (every 3rd wave at most, 8 times per level,
        # so the routes settle and agents keep the same ones)
        if self.path_dirty and self.rebuilds < 8 and self.waves - self.built_wave >= 3:
            self.rebuilds += 1
            self.build_paths(reset=False)

    # reward paths
    def track_hazards(self, boxes: list, ids: list):
        """Remember where each lethal piece of the level started and every place a moving one has been (shots,
        pieces that appear later, are seen by the agents but not planned around)."""
        if self.hz_start is None:
            self.hz_start = {}
        level_ids = self.level.hz_level_ids if self.level is not None else set()
        for b, hid in zip(boxes, ids):
            if hid not in level_ids:
                continue
            if hid not in self.hz_start:
                self.hz_start[hid] = b
                continue
            x, y, hw, hh, a = b
            x0, y0, _, _, a0 = self.hz_start[hid]
            if abs(x - x0) > 2 or abs(y - y0) > 2 or abs(a - a0) > 0.05:
                if hid not in self.moving:
                    self.moving.add(hid)
                    self.sweep.add((round(x0 / 4) * 4, round(y0 / 4) * 4, hw, hh, round(a0, 1)))
                key = (round(x / 4) * 4, round(y / 4) * 4, hw, hh, round(a, 1))
                if key not in self.sweep:
                    self.sweep.add(key)
                    self.path_dirty = True

    def check_teleport(self, a, b):
        """An agent that moved further than anything can move in one frame went through a portal: remember
        the link (rounded to tiles) and rebuild the routes through it."""
        if np.hypot(b[0] - a[0], b[1] - a[1]) < 90:
            return
        if np.hypot(b[0] - self.level.spawn[0], b[1] - self.level.spawn[1]) < 40:
            return                         # the game put it back on the spawn (a death), not a portal
        ka, kb = (round(a[0] / (2 * TILE)), round(a[1] / (2 * TILE))), (round(b[0] / (2 * TILE)), round(b[1] / (2 * TILE)))
        if any((round(p[0] / (2 * TILE)), round(p[1] / (2 * TILE))) == ka for p, _ in self.links):
            return
        if len(self.links) >= 12:
            return
        n = self.jumps_seen[(ka, kb)] = self.jumps_seen.get((ka, kb), 0) + 1
        if n < 3:                          # a hitch moves agents far too, but never 3 times the same way
            return
        self.links.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
        self.path_dirty = True
        print(f"[portal] {self.level.name}: ({a[0]:.0f},{a[1]:.0f}) -> ({b[0]:.0f},{b[1]:.0f}); routes rebuilt "
              f"after this wave", flush=True)

    def update_route_stats(self, fin):
        """Per route: the furthest any of its agents got (share of its length) and the last wave that improved."""
        k = len(self.paths)
        if len(self.route_stats) != k:
            self.route_stats = [{"best": 0.0, "improved": self.waves, "finished": 0, "shown": False} for _ in range(k)]
        for j in range(k):
            fr = [self.frac[i] or 0.0 for i in range(len(self.path_of)) if self.path_of[i] == j]
            won = sum(1 for _, i in fin if i < len(self.path_of) and self.path_of[i] == j)
            st = self.route_stats[j]
            st["finished"] += won
            best = max(fr + [1.0 if won else 0.0]) if fr else 0.0
            if best > st["best"] + 0.02:
                st["best"], st["improved"] = best, self.waves
        for j in range(k):                  # report each change of a route's stalled state once
            now = self.stalled(j)
            if now != self.route_stats[j]["shown"]:
                self.route_stats[j]["shown"] = now
                print(f"[routes] route {j + 1} " + ("stalled at {:.0f}%: fewer agents".format(100 * self.route_stats[j]["best"])
                                                    if now else "moving again: back to its share"), flush=True)

    def build_paths(self, reset: bool):
        lv = self.level
        static = [b for hid, b in (self.hz_start or {}).items() if hid not in self.moving]
        t = time.time()
        sweeps = [(x, y, hw, hh, a) for x, y, hw, hh, a in sorted(self.sweep)]
        self.paths = Planner(lv.words0, lv.ox, lv.oy, lv.spawn, lv.goals, static, sweeps, lv.keys, lv.bandages,
                             list(lv.portals) + list(self.links), lift=lv.wind).find(self.args.paths)
        self.route_stats = []
        if hasattr(self, "best_k"):          # new routes: progress on them starts again (no reward for that)
            self.best_k = [[None] * len(self.paths) for _ in self.best_k]
            self.phi = [None] * len(self.best_k)   # the progress reward picks up from the new routes' values
        self.paths_built, self.path_dirty, self.built_wave = True, False, self.waves
        print(f"[path] {lv.name}: " + (", ".join(f"path {k + 1} {p.length / TILE:.0f} tiles ({p.mode})" for k, p in enumerate(self.paths))
                                       if self.paths else "no route found")
              + f"  [{len(static)} fixed hazards, {len(self.moving)} moving; {time.time() - t:.2f}s]", flush=True)
        self.path_of = self.assign_paths(len(self.traj))
        if reset:                   # mid-wave (attached mid-level): progress so far was on other routes
            self.best_s = [None] * len(self.traj)
        self.post_paths()

    def post_paths(self):
        payload = [{"pts": pc.tolist(), "col": argb(route_rgb(k, p)), "w": 3}
                   for k, p in enumerate(self.paths) for pc in p.pieces()] if self.game_paths_on else []
        try:
            self.script.post({"type": "path", "payload": payload})
        except Exception:
            pass

    def toggle_game_paths(self):
        self.game_paths_on = not self.game_paths_on
        self.post_paths()

    def agent_fitness(self, i: int, won: bool, dead: bool) -> float:
        return fitness(won, dead, self.frac[i] or 0.0, 1.0, self.frames[i])

    def snapshot_best(self, wave: int):
        """The policy network as it played this wave (before learning from it) and the last input of the agent
        with the best fitness, for the monitor's network view."""
        cand = [i for i in range(len(self.fit)) if self.fit[i] is not None and self.traj[i]["o"]]
        if not cand:
            return
        k = max(cand, key=lambda i: self.fit[i])
        with torch.no_grad():
            layers = [self.policy.pi[0], self.policy.pi[2], self.policy.pi[4]]
            self.best_net = {"wave": wave, "agent": k, "fit": self.fit[k], "path": self.path_of[k] if self.paths else None,
                             "obs": np.array(self.traj[k]["o"][-1], np.float32),
                             "W": [l.weight.detach().numpy().copy() for l in layers],
                             "b": [l.bias.detach().numpy().copy() for l in layers]}

    def set_agents(self, n: int):
        self.want_agents = max(1, min(int(n), self.pool))
        print(f"[monitor] {self.want_agents} agents from the next wave", flush=True)

    def set_path_share(self, share):
        self.path_share = None if share is None else [float(x) for x in share]
        print("[monitor] route shares: " + ("automatic" if share is None else " / ".join(f"{x:.0f}" for x in share))
              + " (from the next wave)", flush=True)

    # saving
    def save(self, path=None):
        torch.save({"policy": self.policy.state_dict(), "obs": OBS, "actions": ACTIONS},
                   path or os.path.join(self.run_dir, "model.pt"))

    def save_best(self, level: str, t: float, agent: int):
        d = os.path.join(self.run_dir, "levels", level)
        os.makedirs(d, exist_ok=True)
        self.save(os.path.join(d, "best.pt"))
        with open(os.path.join(d, "best.json"), "w") as f:
            json.dump({"level": level, "seconds": t, "agent": agent, "wave": self.waves,
                       "path": self.path_of[agent] + 1 if self.paths else None,
                       "fitness": self.fit[agent], "actions": self.acts[agent]}, f)
        print(f"[best] {level}: {t:.2f}s -> {d}", flush=True)

    # frida
    def on_message(self, msg, data):
        if msg["type"] == "error":
            print("[script error]", msg.get("description"), msg.get("stack", ""), flush=True)
            return
        p = msg["payload"]
        ev = p.get("ev")
        reply = None
        try:
            if ev == "frame":
                reply = self.on_frame(p)
            elif ev == "info":
                self.on_info(p, data)
            elif ev == "wave_end":
                self.on_wave_end(p)
            elif ev == "level":
                print(f"[game] level {p['name']}", flush=True)
                self.level = None
                self.paths = []
            elif ev == "save":
                lv = p.get("level")
                self.save(os.path.join(self.run_dir, f"{lv}_model.pt"))
                print(f"[S] saved the agent for {lv}; finishing the level", flush=True)
                self.script.post({"type": "finish"})
            elif ev == "stopped":
                self.stopped.set()
            elif ev in ("ready", "wave", "continue", "level_end"):
                if ev != "wave":
                    print(f"[game] {ev} {p}", flush=True)
            else:
                print(f"[game] {p}", flush=True)
        except Exception:  # never leave the game waiting
            import traceback
            traceback.print_exc()
            reply = None
        if ev in ("info", "wave_end"):         # the game waits for this between waves / at the level start
            go = {"agents": self.want_agents}
            if ev == "info" and self.level is not None and self.level.keys:
                go["keys"] = [list(k) for k in self.level.keys]
                go["locks"] = self.level.lock_groups
            self.script.post({"type": "go", "payload": go})
        if ev == "frame":
            if reply is None:
                reply = {"a": [[0, 0, 0, 0]] * len(p["s"]), "leader": 0}
            if self.stopping:
                reply["stop"] = True
            self.script.post({"type": "act", "payload": reply})

    def run(self):
        import frida
        other = other_trainers()
        if other:
            raise SystemExit(f"another realgame.py is already attached to the game (pid {', '.join(other)}): "
                             f"stop it first (Ctrl+C in its terminal, or a file runs/<name>/STOP)")
        if self.args.agents > MAX_AGENTS:
            print(f"--agents {self.args.agents}: at most {MAX_AGENTS}, using {MAX_AGENTS}")
        if self.pool > GAME_POOL:
            print(f"{self.pool} agents: {self.pool - GAME_POOL} extra Meat Boys are built when a level loads (the level "
                  f"on screen now, if any, has the game's {GAME_POOL}: enter or restart a level for all)", flush=True)
        cfg = {"agents": self.want_agents, "pool": self.pool, "wave_frames": int(self.args.wave_seconds * HZ),
               "finish_frames": 60, "debug": bool(self.args.debug_spawn),
               "finish_quota": max(0, self.args.end_after)}
        with open(JS, encoding="utf-8") as f:
            src = f.read().replace("__CONFIG__", json.dumps(cfg))
        try:
            session = frida.attach(GAME_EXE)
        except frida.ProcessNotFoundError:
            raise SystemExit(f"{GAME_EXE} is not running: start the game and enter a level")
        mon = None
        if not self.args.no_monitor:
            try:
                from monitor import Monitor
                mon = Monitor(self)
            except Exception as e:
                print(f"monitor off: {e}")
        self.script = session.create_script(src)
        self.script.on("message", self.on_message)
        self.script.load()
        print(f"attached: {cfg['agents']} agents, waves of {self.args.wave_seconds:g}s, up to {self.args.paths} paths"
              + (" (TEST: scripted inputs, no learning)" if self.args.test else "") + "; Ctrl+C to stop", flush=True)
        stop_file = os.path.join(self.run_dir, "STOP")   # create it to stop a background run safely
        if os.path.exists(stop_file):
            os.remove(stop_file)
        # Ctrl+C only asks for a stop: a second Ctrl+C in the middle of stopping used to leave the game waiting
        # for an answer that never came (the game froze and had to be restarted)
        import signal
        want_stop = threading.Event()

        def on_sigint(*_):
            if not want_stop.is_set():
                print("stopping (saving and handing the characters back; Ctrl+C again is ignored)...", flush=True)
            want_stop.set()
        old_sigint = signal.signal(signal.SIGINT, on_sigint)
        try:
            while not os.path.exists(stop_file) and not want_stop.is_set():
                if mon is not None and not mon.closed:
                    mon.tick()
                    time.sleep(1 / 20)
                else:
                    time.sleep(0.5)
            if os.path.exists(stop_file):
                os.remove(stop_file)
        finally:
            if not self.args.test:
                self.save()
            # stop through the next frame reply: the game thread is never left waiting for an answer
            self.stopping = True
            if not self.stopped.wait(3.0):
                try:   # no frames are coming (menu / level change): nothing is waiting, RPC is safe
                    self.script.exports_sync.shutdown()
                except Exception:
                    pass
            time.sleep(0.2)
            try:
                self.script.unload()
            except Exception:
                pass
            session.detach()
            if mon is not None:
                mon.close()
            signal.signal(signal.SIGINT, old_sigint)
            print(f"saved in {self.run_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=40, help=f"agents per wave (1-{MAX_AGENTS}; above {GAME_POOL}, extra Meat Boys are built "
                                                          "at each level load)")
    ap.add_argument("--wave-seconds", type=float, default=20.0, help="wave time limit (default 20)")
    ap.add_argument("--end-after", type=int, default=5,
                    help="end the wave once this many agents finished (default 5; 0 = wait for all)")
    ap.add_argument("--batch", type=int, default=16384, help="samples per PPO update (updates between waves)")
    ap.add_argument("--paths", type=int, default=3, help="different reward paths per level (1-3)")
    ap.add_argument("--load", default=None, help="run folder or .pt to continue from")
    ap.add_argument("--name", default=None, help="run folder name under runs/ (default real_<time>)")
    ap.add_argument("--test", action="store_true", help="scripted inputs, no learning (checks the mod)")
    ap.add_argument("--no-monitor", action="store_true", help="no monitor window")
    ap.add_argument("--debug-spawn", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.paths = max(1, min(args.paths, 3))
    args.name = args.name or ("real_test" if args.test else time.strftime("real_%Y%m%d-%H%M%S"))
    Trainer(args).run()


if __name__ == "__main__":
    main()
