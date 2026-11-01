"""Reward paths for the real game (realgame.py): up to three different short routes from the spawn to the
goal that keep clear of every solid tile and every lethal object, plus the measures rewards and fitness use.

Inputs are the game's own facts for the current level (realgame.js): the live collision tile map, the
spawn, the goal pieces, and the lethal pieces as circles (centre, radius = scale x half the diagonal of the
piece's size from the game's FUN_004ca860, the bound the game tests first). Moving hazards are added at
every position they are seen at, so the routes avoid their whole sweep.

Search: 8-connected cells, no corner cutting, step length 1 or sqrt(2) tiles. Cells the player's body
(about 12 x 12 units) would share with a solid tile or a hazard circle are closed. Cells close to a hazard
cost a little more, and cells with nothing to stand on or push against cost a little more, so among
equally short routes a path keeps a margin from danger and follows surfaces Meat Boy can use.

Jumps: a route may leave a floor or wall in any direction, but in the air it can only rise and drift so far
(AIR_BUDGET) and, once it starts falling, only go down until it touches a floor or wall again. So a route
cannot fly up a shaft or cross the bottom of a pit. If that finds no route (an unusual level), the search
runs again without the jump limits.

Different routes: after each route is found, the cells around it cost more and the search runs again; a
new route is kept when most of it (away from the spawn and the goal, which every route shares) does not
run along an earlier one and it is not much longer than the shortest.

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
SPREAD_COST = 4.0    # extra per step next to an earlier route (pushes the next search elsewhere)
SPREAD_RADIUS = 2    # cells
SHARED_ENDS = 3.0    # tiles around the spawn and the goal that every route may share
MAX_OVERLAP = 0.6    # a new route may run along earlier ones for at most this share of its length
MAX_STRETCH = 2.5    # and be at most this many times as long as the shortest
AIR_BUDGET = 14      # jump model: after leaving a floor or wall, up to this much "air" (a step up costs 2,
                     # sideways 1): about 7 tiles up or 14 across; once falling, only down (straight or diagonal)
PATH_RGB = [(70, 240, 110), (70, 200, 255), (255, 160, 50)]   # route colours (game and monitor)


class Planner:
    def __init__(self, solid: np.ndarray, ox: float, oy: float, spawn, goals, hazards, sweeps=()):
        """solid: H x W bool (row 0 = bottom), the tiles the player cannot enter (not slopes, half blocks or
        crumble blocks); hazards: [(x, y, radius)] of objects that stay put; sweeps: the same for every
        position a moving hazard was seen at (game units)."""
        self.solid, self.ox, self.oy = solid, ox, oy
        self.H, self.W = solid.shape
        self.spawn, self.goals = tuple(spawn), [tuple(g) for g in goals]
        self.blocked_static, near_static = self._hazard_cells(hazards)
        self.blocked_sweep, near_sweep = self._hazard_cells(sweeps)
        self.near = near_static | near_sweep
        support = np.zeros_like(solid)                  # a surface to stand on or push against
        support[1:, :] |= solid[:-1, :]                 # floor below
        support[:, 1:] |= solid[:, :-1]                 # wall left
        support[:, :-1] |= solid[:, 1:]                 # wall right
        self.support = support
        # below the map is death (the player falls out): the bottom row is only open where it is not air over
        # the edge, i.e. never; a route may not run along it
        self.pit = np.zeros_like(solid)
        self.pit[0, :] = ~solid[0, :]

    # ---- geometry --------------------------------------------------------------------------------
    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.ox) // TILE), int((y - self.oy) // TILE)

    def center(self, c) -> tuple[float, float]:
        return self.ox + (c[0] + 0.5) * TILE, self.oy + (c[1] + 0.5) * TILE

    def solid_at(self, x: float, y: float) -> bool:
        c, r = self.cell(x, y)
        return 0 <= c < self.W and 0 <= r < self.H and bool(self.solid[r, c])

    def visible(self, x0, y0, x1, y1) -> bool:
        """No solid tile on the straight line between two points (sampled every 5 units)."""
        n = int(math.hypot(x1 - x0, y1 - y0) / 5.0)
        for k in range(1, n):
            f = k / n
            if self.solid_at(x0 + f * (x1 - x0), y0 + f * (y1 - y0)):
                return False
        return True

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
            # a path runs through cell centres: a player centred there must stay clear of the circle
            d = np.hypot((cx[c0:c1] - x)[None, :], (cy[r0:r1] - y)[:, None])
            blocked[r0:r1, c0:c1] |= d < reach
            near[r0:r1, c0:c1] |= d < far
        return blocked, near

    # ---- search ------------------------------------------------------------------------------------
    def find(self, k: int = 3) -> list["RewardPath"]:
        """Up to k different routes, shortest first. Clear of everything if possible; else clear of the
        hazards that stay put, crossing a moving hazard's sweep (passable with the right timing) at a cost;
        else solid tiles only."""
        start = self.cell(*self.spawn)
        goals = {self.cell(*g) for g in self.goals}
        if not goals:
            return []
        first = None
        for jumps in (True, False):
            for mode, closed, sweep_cost in (("clear", self.blocked_static | self.blocked_sweep, 0.0),
                                             ("timed", self.blocked_static, 3.0),
                                             ("tiles only", np.zeros_like(self.solid), 3.0)):
                first = self._dijkstra(start, goals, closed, sweep_cost, None, jumps)
                if first:
                    break
            if first:
                break
        if not first:
            return []
        if not jumps:
            mode += ", no jump limits"
        routes = [RewardPath(self, first, mode)]
        extra = np.zeros((self.H, self.W), dtype=np.float32)
        last = first
        for _ in range(3 * k):
            if len(routes) >= k:
                break
            self._spread(extra, last)
            cells = self._dijkstra(start, goals, closed, sweep_cost, extra, jumps)
            if not cells:
                break
            p = RewardPath(self, cells, mode)
            if p.length > MAX_STRETCH * routes[0].length:
                break
            last = cells
            if self._overlap(cells, routes) <= MAX_OVERLAP:
                routes.append(p)
        return routes

    def _spread(self, extra, cells):
        R = SPREAD_RADIUS
        for c, r in cells:
            extra[max(0, r - R):r + R + 1, max(0, c - R):c + R + 1] += SPREAD_COST

    def _overlap(self, cells, routes) -> float:
        """Share of a route's cells (away from the shared ends) within one cell of an earlier route."""
        taken = set()
        for p in routes:
            for c, r in p.cells:
                for dc in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        taken.add((c + dc, r + dr))
        ends = [self.spawn] + self.goals
        mid = [c for c in cells
               if min(math.hypot(self.center(c)[0] - x, self.center(c)[1] - y) for x, y in ends) > SHARED_ENDS * TILE]
        if not mid:
            return 1.0
        return sum(1 for c in mid if c in taken) / len(mid)

    def _open(self, c: int, r: int, closed, ends) -> bool:
        if not (0 <= c < self.W and 0 <= r < self.H) or self.solid[r, c] or self.pit[r, c]:
            return False
        return not closed[r, c] or (c, r) in ends

    def _dijkstra(self, start, goals, closed, sweep_cost, extra, jumps=True):
        """Cheapest cell route. With jumps the state is (cell, air used, falling); without, just the cell."""
        ends = goals | {start}
        if not self._open(*start, closed, ends):
            return []
        solid, support, near, sweep = self.solid, self.support, self.near, self.blocked_sweep
        gl = list(goals)

        def h(c, r):                                     # A*: octile distance to the nearest goal (never more than the cost)
            best = math.inf
            for gc, gr in gl:
                dx, dy = abs(gc - c), abs(gr - r)
                best = min(best, max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy))
            return best

        s0 = (start[0], start[1], 0, 0)
        dist = {s0: 0.0}
        prev = {}
        front = {}                                       # (c, r, falling) -> [(air, dist)] not beaten by another
        pq = [(h(*start), s0)]
        while pq:
            f, st = heapq.heappop(pq)
            c, r, air, falling = st
            d = dist[st]
            if f > d + h(c, r) + 1e-9:
                continue
            if (c, r) in goals:
                path = [st]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                return [(q[0], q[1]) for q in path[::-1]]
            grounded = support[r, c] or (c, r) == start
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nc, nr = c + dc, r + dr
                    if not self._open(nc, nr, closed, ends):
                        continue
                    if dc and dr and (solid[r, nc] or solid[nr, c]):   # no squeezing past a corner
                        continue
                    if not jumps or support[nr, nc]:
                        nst = (nc, nr, 0, 0)                 # on (or against) something again
                    else:
                        if falling and not grounded and dr >= 0:
                            continue                         # falling: only down
                        a = (0 if grounded else air) + (2 if dr > 0 else 1 if dr == 0 else 0)
                        if a > AIR_BUDGET:
                            continue
                        nst = (nc, nr, a, 1 if dr < 0 else 0)
                    step = math.sqrt(2.0) if dc and dr else 1.0
                    step *= (1.0 + (0.0 if support[nr, nc] else AIR_COST) + (NEAR_COST if near[nr, nc] else 0.0)
                             + (sweep_cost if sweep[nr, nc] else 0.0) + (float(extra[nr, nc]) if extra is not None else 0.0))
                    nd = d + step
                    if nd >= dist.get(nst, math.inf):
                        continue
                    if jumps:                            # skip a state both later and with less air left than a known one
                        key = (nst[0], nst[1], nst[3])
                        fr = front.setdefault(key, [])
                        if any(a2 <= nst[2] and d2 <= nd for a2, d2 in fr):
                            continue
                        fr[:] = [(a2, d2) for a2, d2 in fr if not (nst[2] <= a2 and nd <= d2)] + [(nst[2], nd)]
                    dist[nst] = nd
                    prev[nst] = st
                    heapq.heappush(pq, (nd + h(nc, nr), nst))
        return []


class RewardPath:
    """One route as a polyline from the spawn to the goal, with arc-length progress."""

    def __init__(self, planner: Planner, cells: list, mode: str):
        self.pl, self.cells, self.mode = planner, list(cells), mode
        goal = min(planner.goals, key=lambda g: math.hypot(*np.subtract(planner.center(cells[-1]), g)))
        pts = [planner.spawn] + [planner.center(c) for c in cells[1:-1]] + [goal]
        self.points = np.array(_simplify(pts), dtype=np.float64)
        seg = np.diff(self.points, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1]) if len(seg) else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = max(float(self.cum[-1]), 1.0)

    def project(self, x: float, y: float, best: float = 0.0, ahead: float = 3 * TILE):
        """(progress along the path, distance from it) for a point. Only path points the player can see
        (no solid tile in between) and at most `ahead` past the best progress so far count, so standing under
        a platform the path crosses, or next to a later stretch behind a wall, earns nothing."""
        if len(self.points) < 2:
            return 0.0, float(math.hypot(x - self.points[0][0], y - self.points[0][1]))
        p0 = self.points[:-1]
        v = self.points[1:] - p0
        L2 = np.maximum((v ** 2).sum(1), 1e-9)
        t = np.clip(((x - p0[:, 0]) * v[:, 0] + (y - p0[:, 1]) * v[:, 1]) / L2, 0.0, 1.0)
        px, py = p0[:, 0] + t * v[:, 0], p0[:, 1] + t * v[:, 1]
        d = np.hypot(x - px, y - py)
        s = self.cum[:-1] + t * self.seg_len
        reach = np.where(self.cum[:-1] <= best + ahead, d, np.inf)
        for k in np.argsort(reach)[:4]:
            if not np.isfinite(reach[k]) or reach[k] > 8 * TILE:
                break
            if self.pl.visible(x, y, px[k], py[k]):
                return float(min(s[k], best + ahead)), float(d[k])
        return float(best), float(d.min())

    def point_at(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.length)
        k = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.seg_len) - 1)
        f = (s - self.cum[k]) / max(self.seg_len[k], 1e-9)
        a, b = self.points[k], self.points[k + 1]
        return float(a[0] + f * (b[0] - a[0])), float(a[1] + f * (b[1] - a[1]))

    def overlap_with(self, xs, ys) -> float:
        """Mean distance (tiles) from a trajectory to this path (which route a finisher took)."""
        return float(np.mean([self.project(x, y, best=math.inf)[1] for x, y in zip(xs, ys)])) / TILE if len(xs) else 1e9


def _simplify(pts):
    """Drop points in the middle of straight runs."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[1:-1], pts[2:]):
        o = out[-1]
        if abs((a[0] - o[0]) * (b[1] - a[1]) - (a[1] - o[1]) * (b[0] - a[0])) > 1e-6:
            out.append(a)
    out.append(pts[-1])
    return out


def fitness(won: bool, dead: bool, progress: float, length: float, frames: int) -> float:
    """One number to compare agents: finishing dominates, then how far along the path, then speed."""
    return (1000.0 if won else 0.0) + 500.0 * min(1.0, progress / max(length, 1.0)) - 0.05 * frames \
        - (50.0 if dead else 0.0)
