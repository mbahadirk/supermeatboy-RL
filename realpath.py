"""Reward path for the real game (realgame.py): the shortest route from the spawn to the goal that keeps
clear of every solid tile and every lethal object, plus the measures rewards and fitness use.

Inputs are the game's own facts for the current level (realgame.js): the live collision tile map, the
spawn, the goal pieces, and the lethal pieces as circles (centre, radius = scale x half the diagonal of the
piece's size from the game's FUN_004ca860, the bound the game tests first). Moving hazards are added at
every position they are seen at, so the path avoids their whole sweep.

Search: 8-connected cells, no corner cutting, step length 1 or sqrt(2) tiles. Cells the player's body
(about 12 x 12 units) would share with a solid tile or a hazard circle are closed. Cells close to a hazard
cost a little more, and cells with nothing to stand on or push against cost a little more, so among
equally short routes the path keeps a margin from danger and follows surfaces Meat Boy can use.

Coordinates: game units, y up; cell (c, r) has its lower-left corner at (ox + 20c, oy + 20r).
"""
from __future__ import annotations

import heapq
import math

import numpy as np

TILE = 20.0
BODY = 6.0           # half size of Meat Boy's box (game units)
MARGIN = 3.0         # extra clearance from hazards
NEAR = 1.5           # tiles: cells this close to a hazard cost NEAR_COST more
NEAR_COST = 0.6
AIR_COST = 0.25      # extra per step through a cell with no surface next to it


class RewardPath:
    def __init__(self, solid: np.ndarray, ox: float, oy: float, spawn, goals, hazards, sweeps=()):
        """solid: H x W bool (row 0 = bottom), the tiles the player cannot enter (not slopes, half blocks or
        crumble blocks); hazards: [(x, y, radius)] of objects that stay put; sweeps: the same for every
        position a moving hazard was seen at (game units)."""
        self.solid, self.ox, self.oy = solid, ox, oy
        self.H, self.W = solid.shape
        self.spawn, self.goals = tuple(spawn), [tuple(g) for g in goals]
        self.hazards, self.sweeps = list(hazards), list(sweeps)
        self.blocked_static, near_static = self._hazard_cells(self.hazards)
        self.blocked_sweep, near_sweep = self._hazard_cells(self.sweeps)
        self.near = near_static | near_sweep
        self.mode = None
        self.cells = self._search()
        self.found = bool(self.cells) and self.cells[-1] in self._goal_cells()
        pts = [self.center(c) for c in self.cells] if self.cells else [self.spawn] + self.goals[:1]
        self.points = np.array(_simplify(pts), dtype=np.float64)
        seg = np.diff(self.points, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1]) if len(seg) else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = max(float(self.cum[-1]), 1.0)

    # ---- geometry --------------------------------------------------------------------------------
    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.ox) // TILE), int((y - self.oy) // TILE)

    def center(self, c) -> tuple[float, float]:
        return self.ox + (c[0] + 0.5) * TILE, self.oy + (c[1] + 0.5) * TILE

    def _goal_cells(self):
        return {self.cell(*g) for g in self.goals}

    def _hazard_cells(self, hazards):
        """Cells the body cannot occupy without touching a hazard circle, and cells near one."""
        blocked = np.zeros((self.H, self.W), dtype=bool)
        near = np.zeros((self.H, self.W), dtype=bool)
        cx = self.ox + (np.arange(self.W) + 0.5) * TILE
        cy = self.oy + (np.arange(self.H) + 0.5) * TILE
        for x, y, rad in hazards:
            reach = rad + BODY + MARGIN
            far = reach + NEAR * TILE
            c0, c1 = max(0, int((x - far - self.ox) // TILE)), min(self.W, int((x + far - self.ox) // TILE) + 1)
            r0, r1 = max(0, int((y - far - self.oy) // TILE)), min(self.H, int((y + far - self.oy) // TILE) + 1)
            if c0 >= c1 or r0 >= r1:
                continue
            # the path runs through cell centres: a player centred there must stay clear of the circle
            d = np.hypot((cx[c0:c1] - x)[None, :], (cy[r0:r1] - y)[:, None])
            blocked[r0:r1, c0:c1] |= d < reach
            near[r0:r1, c0:c1] |= d < far
        return blocked, near

    # ---- search ------------------------------------------------------------------------------------
    def _search(self) -> list[tuple[int, int]]:
        """Clear of everything; else clear of the hazards that stay put, crossing a moving hazard's sweep
        (passable with the right timing) at a cost; else solid tiles only."""
        start, goals = self.cell(*self.spawn), self._goal_cells()
        if not goals:
            return []
        for mode, closed, sweep_cost in (("clear", self.blocked_static | self.blocked_sweep, 0.0),
                                         ("timed", self.blocked_static, 3.0),
                                         ("tiles only", np.zeros_like(self.solid), 3.0)):
            path = self._dijkstra(start, goals, closed, sweep_cost)
            if path:
                self.mode = mode
                return path
        return []

    def _open(self, c: int, r: int, closed, ends) -> bool:
        if not (0 <= c < self.W and 0 <= r < self.H) or self.solid[r, c]:
            return False
        return not closed[r, c] or (c, r) in ends

    def _dijkstra(self, start, goals, closed, sweep_cost):
        ends = goals | {start}
        if not self._open(*start, closed, ends):
            return []
        solid = self.solid
        support = np.zeros_like(solid)                  # a surface to stand on or push against
        support[1:, :] |= solid[:-1, :]                 # floor below
        support[:, 1:] |= solid[:, :-1]                 # wall left
        support[:, :-1] |= solid[:, 1:]                 # wall right
        support[0, :] = True
        dist = {start: 0.0}
        prev = {}
        pq = [(0.0, start)]
        while pq:
            d, (c, r) = heapq.heappop(pq)
            if (c, r) in goals:
                path = [(c, r)]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                return path[::-1]
            if d > dist[(c, r)]:
                continue
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nc, nr = c + dc, r + dr
                    if not self._open(nc, nr, closed, ends):
                        continue
                    if dc and dr and (solid[r, nc] or solid[nr, c]):   # no squeezing past a corner
                        continue
                    step = math.sqrt(2.0) if dc and dr else 1.0
                    step *= (1.0 + (0.0 if support[nr, nc] else AIR_COST) + (NEAR_COST if self.near[nr, nc] else 0.0)
                             + (sweep_cost if self.blocked_sweep[nr, nc] else 0.0))
                    nd = d + step
                    if nd < dist.get((nc, nr), math.inf):
                        dist[(nc, nr)] = nd
                        prev[(nc, nr)] = (c, r)
                        heapq.heappush(pq, (nd, (nc, nr)))
        return []

    # ---- measures -----------------------------------------------------------------------------------
    def project(self, x: float, y: float, best: float = math.inf, ahead: float = 3 * TILE):
        """(progress along the path, distance from it) for a point. Progress is searched only up to
        best + ahead, so a point next to a later stretch of a winding path (behind a wall) cannot jump to it."""
        if len(self.points) < 2:
            return 0.0, float(math.hypot(x - self.points[0][0], y - self.points[0][1]))
        p0 = self.points[:-1]
        v = self.points[1:] - p0
        L2 = np.maximum((v ** 2).sum(1), 1e-9)
        t = np.clip(((x - p0[:, 0]) * v[:, 0] + (y - p0[:, 1]) * v[:, 1]) / L2, 0.0, 1.0)
        px, py = p0[:, 0] + t * v[:, 0], p0[:, 1] + t * v[:, 1]
        d = np.hypot(x - px, y - py)
        s = self.cum[:-1] + t * self.seg_len
        if math.isfinite(best):
            d = np.where(self.cum[:-1] <= best + ahead, d, np.inf)
        k = int(np.argmin(d))
        return float(min(s[k], best + ahead if math.isfinite(best) else s[k])), float(d[k])

    def markers(self, spacing: float = 1.5 * TILE, limit: int = 60) -> list[list[float]]:
        """Evenly spaced points along the path (for drawing it in the game)."""
        n = max(2, min(limit, int(self.length / spacing) + 1))
        out = []
        for s in np.linspace(0.0, self.length, n):
            k = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.seg_len) - 1)
            f = (s - self.cum[k]) / max(self.seg_len[k], 1e-9)
            out.append([float(self.points[k][0] + f * (self.points[k + 1][0] - self.points[k][0])),
                        float(self.points[k][1] + f * (self.points[k + 1][1] - self.points[k][1]))])
        return out


def _simplify(pts):
    """Drop points in the middle of straight runs."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[1:-1], pts[2:]):
        o = out[-1]
        if (a[0] - o[0]) * (b[1] - a[1]) != (a[1] - o[1]) * (b[0] - a[0]):
            out.append(a)
    out.append(pts[-1])
    return out


def fitness(won: bool, dead: bool, progress: float, length: float, frames: int) -> float:
    """One number to compare agents: finishing dominates, then how far along the path, then speed."""
    return (1000.0 if won else 0.0) + 500.0 * min(1.0, progress / max(length, 1.0)) - 0.05 * frames \
        - (50.0 if dead else 0.0)
