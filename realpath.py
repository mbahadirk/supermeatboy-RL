"""Reward paths for the real game (realgame.py): up to three different short routes from the spawn to the
goal (collecting the level's keys on the way), plus one through the bandage, and the measures rewards and
fitness use.

Inputs are the game's own facts for the current level (realgame.js): the collision tile layer (raw tile
words), the spawn, the goal pieces, the keys and bandages, and every lethal piece as a rotated box (the
game's size for the piece x its scale, turned by its angle). Moving hazards are added at every position
they were seen at, so the routes avoid their whole sweep.

Tiles (bits 0-7 id, bit 8 special, bits 18-19 shape): ids 1-99 are solid; a shape of 1 is a half block and
2/3 a 45-degree slope. Meat Boy's body (12 x 12 units) does not fit beside half a tile, so half blocks and
slopes close their cell like a full block (they used to count as open, and routes slipped through walls
made of them). Special tiles (bit 8): 100 (64) crumble blocks, 99 lock blocks (opened by keys), 96/97
switch blocks (Rapture). Crumble blocks can be crossed at a small cost; switch blocks at a large one; lock
blocks stay closed until the route has picked up a key, cost a lot until it has all keys, and are open
after that.

Hazards: a cell whose centre is closer than BODY + 1 to a hazard box is closed (the body would touch it);
closer than HAZ_SOFT costs HAZ_COST more, closer than NEAR tiles NEAR_COST more. So a route keeps a margin
where there is room and still threads a gap a body fits through.

Search: 8-connected cells, no corner cutting past full or half blocks. Jumps: a route may leave a floor or
wall in any direction, but in the air it can rise at most JUMP[0] tiles and move JUMP[1] tiles sideways and,
once it starts falling, only go down until it touches a floor or wall again (a wall resets this: climbing
by wall jumps). Open air costs more than cells along a surface. Modes, first that works: hazards and moving
sweeps closed; sweeps crossable at a cost (passable with the right timing); every hazard crossable at a high
cost (never ignored). Then the same with higher jumps, and last without jump limits.

Keys: the route visits the keys one by one, each time the nearest one it can reach, then the goal.

Different routes: after each route is found, the cells around it cost more and the search runs again; a
new route is kept when most of it (away from the spawn and the goal) does not run the same way as an earlier
one (within SAME_WAY tiles with no wall between: a parallel copy beside the first route does not count) and
it is not much longer than the shortest. The bandage route goes spawn -> bandage -> keys -> goal.

Coordinates: game units, y up; cell (c, r) has its lower-left corner at (ox + 20c, oy + 20r).
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

TILE = 20.0
BODY = 6.0           # half size of Meat Boy's box (game units)
HAZ_HARD = BODY + 1.0            # a cell centre this close to a hazard box: the body touches it
HAZ_SOFT = BODY + 12.0           # this close: HAZ_COST extra per step
HAZ_COST = 3.0
NEAR = 1.5           # tiles: cells this close to a hazard cost NEAR_COST more
NEAR_COST = 0.6
RISK_COST = 15.0     # last resort: crossing a hazard cell
SWEEP_COST = 3.0     # crossing a moving hazard's sweep (timed mode)
AIR_COST = 0.5       # extra per step through a cell with no surface next to it (routes follow walls and floors)
CRUMBLE_COST = 1.0   # extra per step through a crumble block (it breaks after a touch)
SWITCH_COST = 6.0    # extra per step through a switch block (open only while switched)
HALF_COST = 10.0     # last resort ("squeeze"): extra per step through half a tile
LOCK_COST = 6.0      # extra per step through a lock block once some (not all) keys are picked up
SPREAD_COST = 4.0    # extra per step next to an earlier route (pushes the next search elsewhere)
SPREAD_RADIUS = 4    # cells (as wide as SAME_WAY: the next search is pushed past what would count as the same way)
SHARED_ENDS = 3.0    # tiles around the spawn and the goal that every route may share
MAX_OVERLAP = 0.6    # a new route may run the same way as earlier ones for at most this share of its length
SAME_WAY = 4         # tiles: a route cell this close to an earlier route, with no wall between, runs the same way
                     # (a copy a few cells beside the first route is not another way through the level)
MAX_STRETCH = 2.5    # and be at most this many times as long as the shortest
JUMP = (5, 12)       # jump model: after leaving a floor or wall a route may rise at most 5 tiles and move at most
                     # 12 tiles sideways before it starts falling (the game: jump speed ~300, gravity 550 ->
                     # about 4 tiles up); falling it only goes down (straight or diagonal). A cell beside a wall
                     # counts as a surface, so climbing by wall jumps is allowed.
JUMP_RELAXED = (8, 18)   # if nothing fits: higher jumps (wind, wall-jump chains the model misses); last: no limits
TIME_BUDGET = 6.0    # seconds: no more alternative routes after this (the game waits while routes are built)
MAX_POPS = 250000    # a search that expands this many states gives up (the next, looser mode is tried)
# reward by closeness to a route (RewardPath.project): progress at a point = route progress s of a nearby route
# point - FIELD_DECAY x the walking distance to it (open cells only), for route points within FIELD_RADIUS tiles
FIELD_DECAY = 1.0    # a tile of progress lost per tile away from the route (walking distance). Not less: with less,
                     # an agent standing still beside a long stretch (or behind a wall next to one) was credited
                     # further along it every frame (up to ~11 tiles). Being off the line still costs only once
                     # (rewards pay new best progress), and moving toward the goal beside the line pays fully
FIELD_RADIUS = 10    # tiles
FIELD_UP = 2.0       # climbing a tile counts this many tiles of distance (sitting under a high stretch is far from it)
WAYPOINT_NEAR = 2 * TILE   # a key this close to a route is on it (the route picks it up there)
PATH_RGB = [(70, 240, 110), (70, 200, 255), (255, 160, 50), (240, 110, 200)]   # route colours (bandage last)
CRUMBLE_IDS, LOCK_IDS, SWITCH_IDS = (64, 100), (99,), (96, 97, 98)


def tile_classes(words: np.ndarray) -> dict:
    """Boolean masks of the collision tile words (H x W, row 0 = bottom)."""
    t = words.astype(np.uint32)
    tid, special, shape = t & 0xff, ((t >> 8) & 1).astype(bool), (t >> 18) & 3
    sp = special & (tid > 0) & (tid <= 100)
    plain = (tid > 0) & (tid < 100) & ~sp
    return {"full": plain & (shape == 0), "half": plain & (shape == 1), "slope": plain & (shape >= 2),
            "crumble": sp & np.isin(tid, CRUMBLE_IDS), "lock": sp & np.isin(tid, LOCK_IDS),
            "switch": sp & np.isin(tid, SWITCH_IDS),
            "other": sp & ~np.isin(tid, CRUMBLE_IDS + LOCK_IDS + SWITCH_IDS)}


def slope_open_corner(words: np.ndarray) -> np.ndarray:
    """For 45-degree slope tiles, the corner of the cell their open triangle has (0 bottom-left, 1 bottom-right,
    2 top-left, 3 top-right; -1 for other cells). The game derives the solid corner by flipping, then rotating a
    reference corner (FUN_004ca9a0; same decode as tools/smbl_extract.tile_shape)."""
    t = words.astype(np.uint32)
    slope = (((t >> 18) & 3) >= 2) & ((t & 0xff) > 0) & ((t & 0xff) < 100)
    rot, flip = (t >> 12) & 3, (t >> 14) & 3
    x = np.where((flip == 1) | (flip == 3), 1, -1)          # reference solid corner (-1, +1), flipped
    y = np.where((flip == 2) | (flip == 3), -1, 1)
    xr =np.select([rot == 0, rot == 1, rot == 2, rot == 3], [x, -y, -x, y])
    yr = np.select([rot == 0, rot == 1, rot == 2, rot == 3], [y, x, -y, -x])
    solid_corner = (xr > 0).astype(np.int8) + 2 * (yr > 0).astype(np.int8)
    return np.where(slope, 3 - solid_corner, -1).astype(np.int8)


CORNER_SIGN = ((-1, -1), (1, -1), (-1, 1), (1, 1))
SQRT2 = math.sqrt(2.0)


def box_distance(px, py, box) -> np.ndarray:
    """Distance from points to a rotated box (x, y, half w, half h, angle); 0 inside."""
    x, y, hw, hh, ang = box
    c, s = math.cos(ang), math.sin(ang)
    dx, dy = px - x, py - y
    u, v = dx * c + dy * s, -dx * s + dy * c
    return np.hypot(np.maximum(np.abs(u) - hw, 0.0), np.maximum(np.abs(v) - hh, 0.0))


class Planner:
    def __init__(self, words: np.ndarray, ox: float, oy: float, spawn, goals, hazards, sweeps=(), keys=(),
                 bandages=(), portals=(), lift=None):
        """words: H x W collision tile words; hazards: [(x, y, half w, half h, angle)] of pieces that stay put;
        sweeps: the same for every position a moving hazard was seen at; keys / bandages: [(x, y)]; portals:
        [((x, y), (x, y))] pairs (touching one end puts the player at the other)."""
        self.cls = tile_classes(words)
        k = self.cls
        self.ox, self.oy = ox, oy
        self.H, self.W = words.shape
        self.wall = k["full"] | k["half"]                          # never open
        self.corner = self.wall                                    # no diagonal step past these
        self.sopen = slope_open_corner(words)                      # slopes: open only toward one corner
        self.spawn, self.goals = tuple(spawn), [tuple(g) for g in goals]
        self.keys, self.bandages = [tuple(p) for p in keys], [tuple(p) for p in bandages]
        self.portals = [(tuple(a), tuple(b)) for a, b in portals]
        self.lift = lift if lift is not None else np.zeros(words.shape, dtype=bool)   # wind: rising is free
        self.hard_static, self.soft_static, near_static = self._hazard_cells(hazards)
        self.hard_sweep, self.soft_sweep, near_sweep = self._hazard_cells(sweeps)
        self.near = near_static | near_sweep
        # below the map is death (the player falls out): a route may not run along the bottom row
        self.pit = np.zeros((self.H, self.W), dtype=bool)
        self.pit[0, :] = True

    # ---- geometry --------------------------------------------------------------------------------
    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.ox) // TILE), int((y - self.oy) // TILE)

    def center(self, c) -> tuple[float, float]:
        return self.ox + (c[0] + 0.5) * TILE, self.oy + (c[1] + 0.5) * TILE

    def solid_at(self, x: float, y: float) -> bool:
        c, r = self.cell(x, y)
        return 0 <= c < self.W and 0 <= r < self.H and bool(self.wall[r, c] or self.cls["lock"][r, c])

    def _slope_ok(self, c, r, dc, dr) -> bool:
        """A step leaving (or entering) slope cell (c, r) toward offset (dc, dr): only through its open sides
        (the two edges at its open corner, or that corner)."""
        oc = self.sopen[r, c]
        if oc < 0:
            return True
        sx, sy = CORNER_SIGN[oc]
        return (dc == 0 or dc == sx) and (dr == 0 or dr == sy)

    def _corner_ok(self, c, r, crossing, corner) -> bool:
        """A diagonal step passes corner `crossing` (signs) of side cell (c, r): blocked by a full or half block,
        and by a slope unless that corner is its open one."""
        if not (0 <= c < self.W and 0 <= r < self.H):
            return False
        if corner[r, c]:
            return False
        oc = self.sopen[r, c]
        return oc < 0 or CORNER_SIGN[oc] == crossing

    def visible(self, x0, y0, x1, y1) -> bool:
        """No solid tile on the straight line between two points (sampled every 5 units)."""
        n = int(math.hypot(x1 - x0, y1 - y0) / 5.0)
        for k in range(1, n):
            f = k / n
            if self.solid_at(x0 + f * (x1 - x0), y0 + f * (y1 - y0)):
                return False
        return True

    def _hazard_cells(self, boxes):
        """Cells the body cannot occupy without touching a hazard, cells with little margin, cells near one."""
        hard = np.zeros((self.H, self.W), dtype=bool)
        soft = np.zeros((self.H, self.W), dtype=bool)
        near = np.zeros((self.H, self.W), dtype=bool)
        cx = self.ox + (np.arange(self.W) + 0.5) * TILE
        cy = self.oy + (np.arange(self.H) + 0.5) * TILE
        for b in boxes:
            x, y, hw, hh, _ = b
            far = math.hypot(hw, hh) + BODY + NEAR * TILE
            c0, c1 = max(0, int((x - far - self.ox) // TILE)), min(self.W, int((x + far - self.ox) // TILE) + 1)
            r0, r1 = max(0, int((y - far - self.oy) // TILE)), min(self.H, int((y + far - self.oy) // TILE) + 1)
            if c0 >= c1 or r0 >= r1:
                continue
            d = box_distance(cx[None, c0:c1], cy[r0:r1, None], b)
            hard[r0:r1, c0:c1] |= d < HAZ_HARD
            soft[r0:r1, c0:c1] |= d < HAZ_SOFT
            near[r0:r1, c0:c1] |= d < BODY + NEAR * TILE
        return hard, soft, near

    # ---- search ------------------------------------------------------------------------------------
    MODES = (("clear", "closed", "closed", False), ("timed", "closed", "cost", False), ("risky", "cost", "cost", False),
             ("squeeze", "cost", "cost", True))

    def _layers(self, mode, locks: str):
        """(closed cells, extra cost per step, support, corner cells, wall) for a hazard mode and a lock state
        ('closed' / 'cost' / 'open')."""
        _, hz, sw, squeeze = mode
        k = self.cls
        wall = k["full"] if squeeze else self.wall
        closed = wall | self.pit
        cost = np.zeros((self.H, self.W), dtype=np.float32)
        cost += np.where(self.soft_static | self.soft_sweep, HAZ_COST, 0.0).astype(np.float32)
        cost += np.where(self.near, NEAR_COST, 0.0).astype(np.float32)
        if hz == "closed":
            closed = closed | self.hard_static
        else:
            cost += np.where(self.hard_static, RISK_COST, 0.0).astype(np.float32)
        if sw == "closed":
            closed = closed | self.hard_sweep
        else:
            cost += np.where(self.hard_sweep, SWEEP_COST, 0.0).astype(np.float32)
        cost += np.where(k["crumble"] | k["other"], CRUMBLE_COST, 0.0).astype(np.float32)
        cost += np.where(k["switch"], SWITCH_COST, 0.0).astype(np.float32)
        if squeeze:
            cost += np.where(k["half"], HALF_COST, 0.0).astype(np.float32)
        if locks == "closed":
            closed = closed | k["lock"]
        elif locks == "cost":
            cost += np.where(k["lock"], LOCK_COST, 0.0).astype(np.float32)
        solid = self.wall | k["slope"] | k["crumble"] | k["switch"] | k["other"] | \
            (k["lock"] if locks != "open" else False)
        support = k["slope"].copy()                     # a surface to stand on or push against (a slope cell has one)
        support[1:, :] |= solid[:-1, :]                 # floor below
        support[:, 1:] |= solid[:, :-1]                 # wall left
        support[:, :-1] |= solid[:, 1:]                 # wall right
        support |= self.lift                            # a fan's wind column carries the player up
        return closed, cost, support, wall, wall

    def find(self, k: int = 3) -> list["RewardPath"]:
        """Up to k different routes, shortest first, then the bandage route when the level has one."""
        if not self.goals:
            return []
        start = self._snap(self.cell(*self.spawn))
        goals = {self._snap(self.cell(*g)) for g in self.goals}
        self.key_cells = [self._snap(self.cell(*p)) for p in self.keys]
        self.links = {}
        for a, b in self.portals:
            ca, cb = self._snap(self.cell(*a)), self._snap(self.cell(*b))
            self.links[ca], self.links[cb] = cb, ca
        self._cache = {}
        first, used = None, None
        self.t0 = time.time()
        order = [(m, j) for m in self.MODES[:3] for j in (JUMP, JUMP_RELAXED)] +                 [(self.MODES[3], JUMP_RELAXED)] + [(m, None) for m in self.MODES]
        for mode, jumps in order:
            first = self._route(start, goals, mode, None, jumps)
            if first:
                used = (mode, jumps)
                break
        if not first:
            return []
        mode, jumps = used
        label = mode[0] + ("" if jumps == JUMP else ", high jumps" if jumps else ", no jump limits") + (f", {len(self.keys)} key(s)" if self.keys else "")
        routes = [RewardPath(self, first, label)]
        extra = np.zeros((self.H, self.W), dtype=np.float32)
        last = first
        for _ in range(3 * k):
            if len(routes) >= k or time.time() - self.t0 > TIME_BUDGET:
                break
            self._spread(extra, last)
            cells = self._route(start, goals, mode, extra, jumps)
            if not cells:
                break
            if time.time() - self.t0 < TIME_BUDGET:
                cells = self._straighten(cells, start, goals, mode, jumps)
            p = RewardPath(self, cells, label)
            if p.length > MAX_STRETCH * routes[0].length:
                break
            last = cells
            if self._overlap(cells, routes) <= MAX_OVERLAP:
                routes.append(p)
        if self.bandages:                   # spawn -> the nearest bandage it can reach -> keys -> goal
            for jm in ((mode, jumps),) + tuple((m, jumps) for m in self.MODES if m != mode):
                cells = self._route(start, goals, jm[0], None, jm[1],
                                    via={self._snap(self.cell(*b)) for b in self.bandages})
                if cells:
                    routes.append(RewardPath(self, cells, jm[0][0] + ", bandage", kind="bandage"))
                    break
        for p in routes:                    # closeness fields now, while the game waits for the routes
            p._build_field()
        return routes

    def _route(self, start, goals, mode, extra, jumps, via=None):
        """Cells from the start through an optional waypoint (nearest of `via`), every key (each time the
        nearest one still reachable), to the goal. [] if some leg has no way."""
        legs, cur, keys_left, n_keys = [], start, set(self.key_cells), 0
        total = len(keys_left)
        if via:
            cells = self._leg(cur, via, mode, extra, jumps, "closed" if total else "cost")
            if not cells:
                return []
            legs.append(cells)
            cur = cells[-1]
            keys_left.discard(cur)
        while keys_left:
            locks = "closed" if n_keys == 0 else "cost"
            cells = self._leg(cur, keys_left, mode, extra, jumps, locks)
            if not cells and locks == "closed":
                cells = self._leg(cur, keys_left, mode, extra, jumps, "cost")
            if not cells:
                break                      # keys it cannot reach: go for the goal anyway
            legs.append(cells)
            cur = cells[-1]
            keys_left.discard(cur)
            n_keys += 1
        locks = "open" if total and n_keys == total else "cost"
        cells = self._leg(cur, goals, mode, extra, jumps, locks)
        if not cells:
            return []
        legs.append(cells)
        out = list(legs[0])
        for leg in legs[1:]:
            out += leg[1:]
        return out

    def _leg(self, start, goals, mode, extra, jumps, locks):
        key = (mode, locks)
        if key not in self._cache:
            closed, cost, support, corner, wall = self._layers(mode, locks)
            self._cache[key] = (closed.tolist(), cost, support.tolist(), self._moves(mode[3], corner, wall), wall)
        closed, cost, support, moves, wall = self._cache[key]
        cost = (cost + extra) if extra is not None else cost
        return self._dijkstra(start, set(goals), closed, cost.tolist(), support, moves, wall, jumps)

    def _moves(self, squeeze, corner, wall):
        """Per cell, the steps the geometry allows (no wall, slopes only through their open side, no squeezing
        past a corner): [(nc, nr, dc, dr, length)]. The same for every lock state and hazard mode."""
        key = ("moves", squeeze)
        if key in self._cache:
            return self._cache[key]
        H, W = self.H, self.W
        sopen = self.sopen
        out = [[[] for _ in range(W)] for _ in range(H)]
        for r in range(H):
            for c in range(W):
                if wall[r, c]:
                    continue
                lst = out[r][c]
                for dc in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        if dc == 0 and dr == 0:
                            continue
                        nc, nr = c + dc, r + dr
                        if not (0 <= nc < W and 0 <= nr < H) or wall[nr, nc]:
                            continue
                        if (sopen[r, c] >= 0 and not self._slope_ok(c, r, dc, dr)) or \
                                (sopen[nr, nc] >= 0 and not self._slope_ok(nc, nr, -dc, -dr)):
                            continue
                        if dc and dr and not (self._corner_ok(nc, r, (-dc, dr), corner) and
                                              self._corner_ok(c, nr, (dc, -dr), corner)):
                            continue
                        lst.append((nc, nr, dc, dr, SQRT2 if dc and dr else 1.0))
        self._cache[key] = out
        return out

    def _snap(self, cell):
        """A spawn, goal or key inside a block (a goal standing on a half block): the nearest open cell,
        looking up first."""
        c, r = cell
        ok = lambda q, p: 0 <= q < self.W and 0 <= p < self.H and not self.wall[p, q]
        if ok(c, r):
            return cell
        for dc, dr in ((0, 1), (-1, 0), (1, 0), (-1, 1), (1, 1), (0, -1), (0, 2), (-1, -1), (1, -1)):
            if ok(c + dc, r + dr):
                return c + dc, r + dr
        return cell

    def _straighten(self, cells, start, goals, mode, jumps):
        """The spread cost bends an alternative route away from earlier ones cell by cell (bumps along a
        ceiling). Search again without it, but only inside the route's own corridor (its cells and their
        neighbours), so it stays a different way through the level, minus the bumps."""
        corridor = np.zeros((self.H, self.W), dtype=bool)
        for c, r in cells:
            corridor[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
        out = self._route(start, goals, mode, np.where(corridor, 0.0, 50.0).astype(np.float32), jumps)
        return out or cells

    def _spread(self, extra, cells):
        R = SPREAD_RADIUS
        for c, r in cells:
            extra[max(0, r - R):r + R + 1, max(0, c - R):c + R + 1] += SPREAD_COST

    def _overlap(self, cells, routes) -> float:
        """Share of a route's cells (away from the shared ends) that run the same way as an earlier route: an
        earlier route's cell within SAME_WAY tiles with no wall between. Running beside an earlier route a few
        cells away is the same way; the other side of a wall, or another shaft further off, is not."""
        taken = set(c for p in routes for c in p.cells)
        ends = [self.spawn] + self.goals
        mid = [c for c in cells
               if min(math.hypot(self.center(c)[0] - x, self.center(c)[1] - y) for x, y in ends) > SHARED_ENDS * TILE]
        if not mid:
            return 1.0
        R = SAME_WAY
        near = sorted(((dc, dr) for dc in range(-R, R + 1) for dr in range(-R, R + 1) if dc * dc + dr * dr <= R * R),
                      key=lambda o: o[0] * o[0] + o[1] * o[1])
        same = 0
        for c, r in mid:
            x, y = self.center((c, r))
            for dc, dr in near:
                q = (c + dc, r + dr)
                if q in taken and self.visible(x, y, *self.center(q)):
                    same += 1
                    break
        return same / len(mid)

    def _dijkstra(self, start, goals, closed, cost, support, moves, wall, jumps=JUMP):
        """Cheapest cell route (A*). With jumps the state is (cell, rise used, sideways used, falling); without,
        just the cell. The start and goal cells are open even when a hazard is close (a spawn next to a saw).
        Gives up (no route) after MAX_POPS expansions."""
        ends = goals | {start}
        sc, sr = start
        if not (0 <= sc < self.W and 0 <= sr < self.H) or wall[sr, sc]:
            return []
        gl = list(goals)
        links = self.links
        diag = SQRT2 - 1.0

        def h(c, r):                                     # octile distance to the nearest goal (never more than the cost)
            best = math.inf
            for gc, gr in gl:
                dx, dy = abs(gc - c), abs(gr - r)
                v = (dx + diag * dy) if dx > dy else (dy + diag * dx)
                if v < best:
                    best = v
            return best

        s0 = (sc, sr, 0, 0, 0)
        dist = {s0: 0.0}
        prev = {}
        front = {}                                       # (c, r, falling) -> [(up, side, dist)] not beaten by another
        max_up, max_side = jumps if jumps else (0, 0)
        pq = [(h(sc, sr), s0)]
        pops = 0
        while pq:
            f, st = heapq.heappop(pq)
            c, r, up, side, falling = st
            d = dist[st]
            if f > d + h(c, r) + 1e-9:
                continue
            pops += 1
            if pops > MAX_POPS:
                return []
            if (c, r) in goals:
                path = [st]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                return [(q[0], q[1]) for q in path[::-1]]
            grounded = support[r][c] or (c, r) == start
            came = prev.get(st)
            if (c, r) in links and (c, r) != start and not (came and (came[0], came[1]) == links[(c, r)]):
                # a portal (entered on foot, not just arrived through it): the only way on is out of the other end
                nst = (*links[(c, r)], 0, 0, 0)
                nd = d + 1.0
                if nd < dist.get(nst, math.inf):
                    dist[nst], prev[nst] = nd, st
                    heapq.heappush(pq, (nd + h(nst[0], nst[1]), nst))
                continue
            if grounded:
                up0 = side0 = 0
            else:
                up0, side0 = up, side
            for nc, nr, dc, dr, ln in moves[r][c]:
                if closed[nr][nc] and (nc, nr) not in ends:
                    continue
                sup = support[nr][nc]
                if not jumps or sup:
                    nst = (nc, nr, 0, 0, 0)                  # on (or against) something again
                else:
                    if falling and not grounded and dr >= 0:
                        continue                             # falling: only down
                    u1 = up0 + (1 if dr > 0 else 0)
                    s1 = side0 + (1 if dc and dr >= 0 else 0)
                    if u1 > max_up or s1 > max_side:
                        continue
                    nst = (nc, nr, u1, s1, 1 if dr < 0 else 0)
                nd = d + ln * (1.0 + (0.0 if sup else AIR_COST) + cost[nr][nc])
                if nd >= dist.get(nst, math.inf):
                    continue
                if jumps and not sup:                    # skip a state later and with less jump left than a known one
                    key = (nc, nr, nst[4])
                    fr = front.get(key)
                    if fr is None:
                        front[key] = [(nst[2], nst[3], nd)]
                    else:
                        if any(u2 <= nst[2] and s2 <= nst[3] and d2 <= nd for u2, s2, d2 in fr):
                            continue
                        fr[:] = [(u2, s2, d2) for u2, s2, d2 in fr
                                 if not (nst[2] <= u2 and nst[3] <= s2 and nd <= d2)] + [(nst[2], nst[3], nd)]
                dist[nst] = nd
                prev[nst] = st
                heapq.heappush(pq, (nd + h(nc, nr), nst))
        return []


class RewardPath:
    """One route as a polyline from the spawn to the goal, with arc-length progress. A portal jump is a
    segment of no length (`warp`): it adds no progress, nothing projects onto it, and it is not drawn."""

    def __init__(self, planner: Planner, cells: list, mode: str, kind: str = "main"):
        self.pl, self.cells, self.mode, self.kind = planner, list(cells), mode, kind
        goal = min(planner.goals, key=lambda g: math.hypot(*np.subtract(planner.center(cells[-1]), g)))
        pts = [planner.spawn] + [planner.center(c) for c in cells[1:-1]] + [goal]
        runs, cur = [], [pts[0]]                       # split where consecutive cells are not neighbours (portals)
        for a, b, p in zip(cells[:-1], cells[1:], pts[1:]):
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1:
                runs.append(cur)
                cur = []
            cur.append(p)
        runs.append(cur)
        points, warp = [], []
        for run in runs:
            run = _simplify(run)
            if points:
                warp.append(len(points) - 1)           # segment from the last point to this run's first
            points += run
        self.points = np.array(points, dtype=np.float64)
        seg = np.diff(self.points, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1]) if len(seg) else np.zeros(0)
        self.warp = np.zeros(len(self.seg_len), dtype=bool)
        self.warp[warp] = True
        self.seg_len[self.warp] = 0.0
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = max(float(self.cum[-1]), 1.0)
        # where the route picks up keys and bandages (progress along it), for the monitor
        cs = set(self.cells)
        self.marks = [(x, y, kd) for kd, ps in (("key", planner.keys), ("bandage", planner.bandages))
                      for x, y in ps if planner.cell(x, y) in cs]

    def _build_field(self):
        """The closeness field, one per leg of the route (legs end at the keys and the bandage it picks up, in
        route order): for every open cell, the best s - FIELD_DECAY x d over route points with progress s up to
        the end of that leg, d the way from the cell to the point through open cells (no walls; climbing a tile
        counts FIELD_UP tiles), at most FIELD_RADIUS tiles. Leg j only knows the route up to its waypoint, so
        where the route doubles back from a key the way out is not credited before the key is touched. A cell's
        value does not depend on the agent's best so far, so standing still earns nothing."""
        pl = self.pl
        H, W = pl.H, pl.W
        k = pl.cls                                     # every solid tile is a wall here (slopes, crumble, switch too)
        self._blocked = pl.wall | k["lock"] | k["slope"] | k["crumble"] | k["switch"] | k["other"]
        blocked = self._blocked.tolist()
        # waypoints: the keys this route passes (within WAYPOINT_NEAR), and on the bandage route the bandage, at the
        # progress of the route point nearest to each
        wps = list(pl.keys) + (list(pl.bandages) if self.kind == "bandage" else [])
        self.waypoints = sorted((self._closest_s(x, y), (x, y)) for x, y in wps
                                if self._route_dist(x, y) < WAYPOINT_NEAR)
        bounds = [w[0] for w in self.waypoints] + [self.length]
        # route points every half tile (no portal jumps) with their progress
        src = []
        for j in range(len(self.seg_len)):
            if self.warp[j]:
                continue
            a, b, L = self.points[j], self.points[j + 1], self.seg_len[j]
            m = max(1, int(L / (TILE / 2)))
            for q in range(m + 1):
                f = q / m
                c, r = pl.cell(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
                if 0 <= c < W and 0 <= r < H and not blocked[r][c]:
                    src.append((self.cum[j] + f * L, c, r))
        # (dc, dr, way) for the field spreading from a cell to its neighbour (dc, dr): the agent would go the
        # other way, so a neighbour below (dr < 0) means climbing
        steps = [(dc, dr, (SQRT2 if dc and dr else 1.0) * TILE + (FIELD_UP - 1.0) * TILE * (dr < 0))
                 for dc in (-1, 0, 1) for dr in (-1, 0, 1) if dc or dr]
        rad = FIELD_RADIUS * TILE
        out = np.full((len(bounds), H, W), -np.inf, dtype=np.float32)
        for leg, bound in enumerate(bounds):
            best = {}                                  # cell -> value
            pq = []
            for sv, c, r in src:
                if sv <= bound + 1e-6 and sv > best.get((c, r), -math.inf):
                    best[(c, r)] = sv
                    heapq.heappush(pq, (-sv, 0.0, c, r))
            while pq:
                v, way, c, r = heapq.heappop(pq)
                v = -v
                if v < best[(c, r)]:
                    continue
                for dc, dr, ln in steps:
                    nc, nr = c + dc, r + dr
                    if not (0 <= nc < W and 0 <= nr < H) or blocked[nr][nc]:
                        continue
                    if dc and dr and (blocked[r][nc] or blocked[nr][c]):
                        continue                       # no cutting a corner
                    nw = way + ln
                    if nw > rad:
                        continue
                    nv = v - FIELD_DECAY * ln
                    if nv > best.get((nc, nr), -math.inf):
                        best[(nc, nr)] = nv
                        heapq.heappush(pq, (-nv, nw, nc, nr))
            for (c, r), v in best.items():
                out[leg, r, c] = v
        self._field = out

    def _nearest(self, x: float, y: float):
        """(segment, position along it 0-1, distance) of the route point nearest to (x, y) (straight distance)."""
        p0 = self.points[:-1]
        v = self.points[1:] - p0
        L2 = np.maximum((v ** 2).sum(1), 1e-9)
        t = np.clip(((x - p0[:, 0]) * v[:, 0] + (y - p0[:, 1]) * v[:, 1]) / L2, 0.0, 1.0)
        d = np.where(self.warp, np.inf, np.hypot(x - (p0[:, 0] + t * v[:, 0]), y - (p0[:, 1] + t * v[:, 1])))
        k = int(np.argmin(d))
        return k, float(t[k]), float(d[k])

    def _closest_s(self, x: float, y: float) -> float:
        k, t, _ = self._nearest(x, y)
        return float(self.cum[k] + t * self.seg_len[k])

    def _route_dist(self, x: float, y: float) -> float:
        return self._nearest(x, y)[2]

    def leg(self, reached) -> int:
        """How many of this route's waypoints (in route order) an agent has touched: `reached` holds the
        positions of the keys / bandages it touched this life."""
        n = 0
        for _, pos in self.waypoints:
            if pos not in reached:
                break
            n += 1
        return n

    def value(self, x: float, y: float, reached=()):
        """Progress at a point, by closeness: the closeness field of the agent's leg (the route up to the next
        key / bandage it has not touched) at the point. Moving toward the goal pays anywhere near the route, not
        only on the line; walls make the way longer, climbing counts double, and a spot's value is fixed (no
        gain from waiting there). None where the field does not reach (more than FIELD_RADIUS away)."""
        if len(self.points) < 2:
            return 0.0
        if getattr(self, "_field", None) is None:
            self._build_field()
        f = self._field[min(self.leg(reached), len(self._field) - 1)]
        pl = self.pl
        c0, r0 = pl.cell(x, y)
        bl = self._blocked
        H, W = bl.shape
        free = lambda c, r: 0 <= c < W and 0 <= r < H and not bl[r, c]
        val = -math.inf
        for c in (c0 - 1, c0, c0 + 1):                     # the value at the nearby cell centres, minus the way there
            for r in (r0 - 1, r0, r0 + 1):
                if not free(c, r):
                    continue
                if c != c0 and r != r0 and not (free(c, r0) or free(c0, r)):
                    continue                               # not diagonally through a wall corner
                fv = f[r, c]
                if fv > -math.inf:
                    cx, cy = pl.center((c, r))
                    way = math.hypot(x - cx, y - cy) + (FIELD_UP - 1.0) * max(0.0, cy - y)
                    val = max(val, float(fv) - FIELD_DECAY * way)
        if val == -math.inf:
            return None
        return float(min(max(val, 0.0), self.length))

    def offset(self, x: float, y: float) -> float:
        """Straight distance from a point to the nearest route point."""
        if len(self.points) < 2:
            return float(math.hypot(x - self.points[0][0], y - self.points[0][1]))
        return self._route_dist(x, y)

    def pieces(self) -> list[np.ndarray]:
        """The polyline split at portal jumps (for drawing)."""
        out, start = [], 0
        for k in np.nonzero(self.warp)[0]:
            out.append(self.points[start:k + 1])
            start = k + 1
        out.append(self.points[start:])
        return [p for p in out if len(p) >= 2]

    def point_at(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.length)
        k = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.seg_len) - 1)
        f = (s - self.cum[k]) / max(self.seg_len[k], 1e-9)
        a, b = self.points[k], self.points[k + 1]
        return float(a[0] + f * (b[0] - a[0])), float(a[1] + f * (b[1] - a[1]))


def _simplify(pts):
    """Drop points in the middle of straight runs (not where the route turns back)."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[1:-1], pts[2:]):
        o = out[-1]
        if abs((a[0] - o[0]) * (b[1] - a[1]) - (a[1] - o[1]) * (b[0] - a[0])) > 1e-6 \
                or (a[0] - o[0]) * (b[0] - a[0]) + (a[1] - o[1]) * (b[1] - a[1]) <= 0:
            out.append(a)
    out.append(pts[-1])
    return out


def fitness(won: bool, dead: bool, progress: float, length: float, frames: int) -> float:
    """One number to compare agents: finishing dominates, then how far along the path, then speed."""
    return (1000.0 if won else 0.0) + 500.0 * min(1.0, progress / max(length, 1.0)) - 0.05 * frames \
        - (50.0 if dead else 0.0)
