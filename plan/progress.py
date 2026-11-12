"""The progress counter of the planning agent: a jump-aware cost-to-go field V (PLAN_PLANNER_AGENT.md, decision 2).

For every cell, the cheapest cost from there to the goal (through the level's keys, in the route's key order)
under exactly the move rules and step costs realpath.Planner uses for its routes. Cells from which the goal
cannot be reached hold +inf: an explicit dead end.

Units: the Planner's cost units. One step to a side neighbour along a surface costs 1 (a diagonal sqrt 2), a
step into a cell with no surface costs AIR_COST more, plus the cell's extra cost (hazard margins, crumble,
switch, lock blocks, and in the looser modes hazard or sweep crossings). On open ground that is about tiles.

How it is built (one backward search per leg):
- The Planner's search state is (cell, rise used, sideways used, falling). Every cell gets the full set of
  (JUMP[0] + 1) x (JUMP[1] + 1) x 2 states (one state without jump limits); a cell with a floor below or a wall
  beside (Planner "support") only ever holds its fresh state, since arriving there resets the jump. Portal
  cells get one extra node, "arrived through the portal" (the Planner's walk-in rule: a portal entered on foot
  only leads out of the other end, one arrived through moves on normally).
- Every state's moves are expanded once with the Planner's rules (realpath.Planner._dijkstra: the geometry of
  Planner._moves, closed cells, the falling and jump-limit rules, the step costs), vectorised with numpy,
  into an explicit edge list. Dijkstra then runs on the reversed edges from the leg's target states (scipy's
  C implementation when scipy is there, else a heap in Python): the cost-to-go of every state at once.
- Legs: with keys, the route's key order is found as Planner.find does (from the spawn, each time the
  nearest key still reachable, with the same lock states per leg). Leg j holds while keys 0..j-1 of that order
  are taken: V_leg[j] = cost to key j + V_leg[j+1] at key j (that leg starting there). The last leg ends at
  the goal. Without keys there is one leg.
- Mode and jump limits: the first (mode, jumps) of Planner.find's order that finds a route from the spawn, so
  V agrees with the routes. If none does, the loosest one (squeeze, no jump limits) and found_route False.

Outputs per leg (n_legs x H x W, float64, +inf where the goal cannot be reached, and in walls):
- V_support: from a cell as a fresh start (standing on a floor or against a wall: rise and sideways used 0,
  not falling). This is the Planner's start rule (its start cell counts as grounded, and may be a closed
  cell, such as one beside a hazard), so V_support at the spawn equals the cost of Planner.find's first route.
- V_air: the best cost-to-go over the airborne states of the cell that can occur (reachable from some supported
  cell under the same rules). Optimistic: it assumes the cheapest such jump state. In cells with support it
  is the fresh state's value, in closed cells (never entered by a route) the V_support value.

The time field T (frames to the goal; the planner's score, plan/planner.py): the same graph (states, moves,
mode and jump limits, closed cells, key legs) with a second weight per edge, the frames of the step:
max(|dc| x fx, |dr| x fy), fx = FX_GROUND from a supported source state else FX_AIR, fy = FY_UP for a step up
and FY_DOWN for a step down (measured: plan/measure_time.py); a portal crossing PORTAL_FRAMES. None of V's extras
(hazard margins, near, risk, sweep, crumble, switch, half, lock costs). A second Dijkstra on these weights, key
legs like V (T_leg[j] = T to key j + T_leg[j+1] at key j). Outputs per leg, like V's:
- T_support: the start rule (as V_support), in frames;
- T_rise / T_fall: the best reachable airborne state of the cell that is rising (falling bit 0: not yet moving
  down since the last support) / falling (bit 1). Where the graph reaches only one of the two in a cell, both
  take that one; without jump limits (no falling bit) both are the cell's value; cells with support and closed
  cells as V_air.
Lookups (values() for V, times() for T, which takes the body's vertical velocity: rising iff vy > 0) share one
code path (_lookup: points in walls and in the open half of half blocks).
V stays for display, comparisons and the failure memory's expectations.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from realpath import AIR_COST, JUMP, JUMP_RELAXED, SQRT2, TILE, Planner

try:                                        # Dijkstra in C; the fallback below gives the same result, slower
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as _sp_dijkstra
except ImportError:                         # pragma: no cover - scipy is in the training environment
    csr_matrix = _sp_dijkstra = None

DIRS = tuple((dc, dr) for dc in (-1, 0, 1) for dr in (-1, 0, 1) if dc or dr)
# frames per tile, measured on recorded play (python -m plan.measure_time runs/plan_2026110*/wm_data: 548 shards,
# 2.07 M locked steps): TILE / the 90th percentile of |dx| per frame on the ground / in the air (both 4.0 u/f),
# TILE / the mean rise speed, TILE / the mean speed of falls of more than 2 tiles
FX_GROUND, FX_AIR, FY_UP, FY_DOWN = 5.00, 5.00, 5.41, 7.74
PORTAL_FRAMES = 1.0                         # a portal crossing (from one end to the other)
DIR_INDEX = {d: k for k, d in enumerate(DIRS)}


def search_order():
    """(mode, jumps) in Planner.find's order: the first that finds a route is used."""
    M = Planner.MODES
    return [(m, j) for m in M[:3] for j in (JUMP, JUMP_RELAXED)] + [(M[3], JUMP_RELAXED)] + [(m, None) for m in M]


def prepare(pl: Planner):
    """What Planner.find sets up before its searches (snapped start, goals, key cells, portal links, the
    layer cache). Returns (start cell, goal cells)."""
    start = pl._snap(pl.cell(*pl.spawn))
    goals = {pl._snap(pl.cell(*g)) for g in pl.goals}
    pl.key_cells = [pl._snap(pl.cell(*p)) for p in pl.keys]
    pl.links = {}
    for a, b in pl.portals:
        ca, cb = pl._snap(pl.cell(*a)), pl._snap(pl.cell(*b))
        pl.links[ca], pl.links[cb] = cb, ca
    pl._cache = {}
    pl.t0 = time.time()
    return start, goals


def route_legs(pl: Planner, start, goals, mode, jumps):
    """Planner._route (no waypoint, no extra cost) split into its legs: [(target key cell or None for the
    goal, lock state, cells)], or None when some leg has no way. Same searches, same order, same result."""
    legs, cur, keys_left, n_keys = [], start, set(pl.key_cells), 0
    total = len(keys_left)
    while keys_left:
        locks = "closed" if n_keys == 0 else "cost"
        cells = pl._leg(cur, keys_left, mode, None, jumps, locks)
        if not cells and locks == "closed":
            locks = "cost"
            cells = pl._leg(cur, keys_left, mode, None, jumps, locks)
        if not cells:
            break                              # keys it cannot reach: go for the goal anyway
        legs.append((cells[-1], locks, cells))
        cur = cells[-1]
        keys_left.discard(cur)
        n_keys += 1
    locks = "open" if total and n_keys == total else "cost"
    cells = pl._leg(cur, goals, mode, None, jumps, locks)
    if not cells:
        return None
    legs.append((None, locks, cells))
    return legs


def path_cost(cells, cost, support) -> float:
    """The Planner's cost of a cell path (a step: length x (1 + AIR_COST off a surface + the cell's extra);
    a portal jump between cells that are not neighbours: 1)."""
    total = 0.0
    for (c, r), (nc, nr) in zip(cells[:-1], cells[1:]):
        dc, dr = nc - c, nr - r
        if max(abs(dc), abs(dr)) > 1:
            total += 1.0
            continue
        ln = SQRT2 if dc and dr else 1.0
        total += ln * (1.0 + (0.0 if support[nr, nc] else AIR_COST) + float(cost[nr, nc]))
    return total


class ProgressField:
    """V for one level. Built from the same inputs as realpath.Planner (or from a Planner: from_planner)."""

    def __init__(self, words, ox, oy, spawn, goals, hazards, sweeps=(), keys=(), bandages=(), portals=(),
                 lift=None, blocks=None, doors=None, outside=None):
        self._build(Planner(words, ox, oy, spawn, goals, hazards, sweeps, keys, bandages, portals, lift=lift,
                            blocks=blocks, doors=doors, outside=outside))

    @classmethod
    def from_planner(cls, pl: Planner) -> "ProgressField":
        self = cls.__new__(cls)
        self._build(pl)
        return self

    # ---- build ------------------------------------------------------------------------------------------
    def _build(self, pl: Planner):
        t0 = time.time()
        self.pl = pl
        self.ox, self.oy, self.H, self.W = pl.ox, pl.oy, pl.H, pl.W
        start, goals = prepare(pl)
        self.start, self.goal_cells = start, goals
        found = None
        if pl.goals:
            for mode, jumps in search_order():
                legs = route_legs(pl, start, goals, mode, jumps)
                if legs:
                    found = (mode, jumps, legs)
                    break
        self.found_route = found is not None
        if found is None:                      # no route in any mode: the loosest rules, one leg to the goal
            mode, jumps = Planner.MODES[3], None
            legs = [(None, "cost", [])]
        else:
            mode, jumps, legs = found
        self.mode, self.jumps = mode, jumps
        self.label = mode[0] + ("" if jumps == JUMP else ", high jumps" if jumps else ", no jump limits")
        self.t_route = time.time() - t0
        # key order: per key leg, the indices of the keys at that cell (two keys may share a cell)
        self.key_order = [[k for k, kc in enumerate(pl.key_cells) if kc == tgt] for tgt, _, _ in legs[:-1]]
        self.leg_locks = [lk for _, lk, _ in legs]
        self.route_cells = [c for c in legs[0][2]]
        for _, _, cells in legs[1:]:
            self.route_cells += cells[1:]
        self.n_legs = len(legs)
        H, W = self.H, self.W
        self._V = np.full((2, self.n_legs, H, W), np.inf)        # the lookup's tables (views below)
        self._T = np.full((3, self.n_legs, H, W), np.inf)
        self.V_support, self.V_air = self._V
        self.T_support, self.T_rise, self.T_fall = self._T
        self.n_states, self.n_edges, self.leg_route_cost = [], [], []
        seed = seed_t = 0.0
        self._t_time = 0.0                     # the time field's own Dijkstra (seconds, all legs)
        for j in range(self.n_legs - 1, -1, -1):      # backward: the goal leg first
            tgt, locks, cells = legs[j]
            targets = np.zeros((H, W), bool)
            for c, r in ([tgt] if tgt is not None else goals):
                if 0 <= c < W and 0 <= r < H:
                    targets[r, c] = True
            leg_start = start if j == 0 else legs[j - 1][0]
            lay = self._layer(locks)
            vs, va, ts, trise, tfall, ns, ne = self._leg_field(lay, targets, seed, leg_start, seed_t)
            self.V_support[j], self.V_air[j] = vs, va
            self.T_support[j], self.T_rise[j], self.T_fall[j] = ts, trise, tfall
            self.n_states.insert(0, ns)
            self.n_edges.insert(0, ne)
            self.leg_route_cost.insert(0, path_cost(cells, lay["cost"], lay["support"]) if cells else math.inf)
            if j > 0:
                kc, kr = legs[j - 1][0]
                seed, seed_t = float(vs[kr, kc]), float(ts[kr, kc])
        self.route_cost = float(sum(self.leg_route_cost))
        self.t_build = time.time() - t0
        self.t_time = self._t_time
        self.t_field = self.t_build - self.t_route

    def _layer(self, locks: str) -> dict:
        """The Planner's layers for the mode and a lock state, and the allowed moves as 8 direction masks."""
        pl, mode = self.pl, self.mode
        closed, cost, support, corner, wall = pl._layers(mode, locks)
        key = ("allowed", mode[3])
        if key not in pl._cache:
            moves = pl._moves(mode[3], corner, wall)
            allowed = np.zeros((8, self.H, self.W), bool)
            for r in range(self.H):
                for c in range(self.W):
                    for nc, nr, dc, dr, ln in moves[r][c]:
                        allowed[DIR_INDEX[(dc, dr)], r, c] = True
            pl._cache[key] = allowed
        return {"closed": closed, "cost": cost, "cost64": cost.astype(np.float64), "support": support,
                "wall": wall, "allowed": pl._cache[key]}

    def _leg_field(self, lay, targets, seed, leg_start, seed_t=0.0):
        H, W = self.H, self.W
        wall, closed, support = lay["wall"], lay["closed"], lay["support"]
        jumps = self.jumps
        U, S = jumps if jumps else (0, 0)
        K = (U + 1) * (S + 1) * 2 if jumps else 1
        cid = np.full((H, W), -1, np.int64)
        open_ = ~wall
        rr, cc = np.nonzero(open_)
        cid[rr, cc] = np.arange(len(rr))
        N = len(rr)
        links = [(q, l) for q, l in self.pl.links.items()
                 if 0 <= q[0] < W and 0 <= q[1] < H and 0 <= l[0] < W and 0 <= l[1] < H and open_[q[1], q[0]]
                 and open_[l[1], l[0]]]
        warp_id = {q: N * K + p for p, (q, _) in enumerate(links)}
        n = N * K + len(links)
        enter = open_ & (~closed | targets)
        portal = np.zeros((H, W), bool)
        for (qc, qr), _ in links:
            portal[qr, qc] = True
        ctx = {"cid": cid, "K": K, "U": U, "S": S, "enter": enter, "support": support, "targets": targets,
               "allowed": lay["allowed"], "cost64": lay["cost64"]}
        # sources: states that move on (not at a target; closed cells are never entered)
        src_cells = enter & ~targets
        ju = np.arange(K) // (2 * (S + 1))
        js = (np.arange(K) // 2) % (S + 1)
        jf = np.arange(K) % 2
        parts = []
        normal = src_cells & ~portal
        r1, c1 = np.nonzero(normal & support)                  # standing: the fresh state only
        if len(r1):
            z = np.zeros(len(r1), np.int64)
            parts.append(_step(ctx, r1, c1, z, z, z, np.ones(len(r1), bool), cid[r1, c1] * K))
        r2, c2 = np.nonzero(normal & ~support)                 # in the air: every jump state
        if len(r2):
            rep = lambda a: np.repeat(a, K)
            tile = lambda a: np.tile(a, len(r2))
            parts.append(_step(ctx, rep(r2), rep(c2), tile(ju), tile(js), tile(jf), np.zeros(len(r2) * K, bool),
                               rep(cid[r2, c2] * K) + tile(np.arange(K))))
        src_l, dst_l, w_l, t_l = ([p[i] for p in parts] for i in range(4))
        for (qc, qr), (lc, lr) in links:                        # walked into a portal: out of the other end
            if not src_cells[qr, qc]:
                continue
            ids = cid[qr, qc] * K + (np.zeros(1, np.int64) if support[qr, qc] else np.arange(K))
            src_l.append(ids)
            dst_l.append(np.full(len(ids), warp_id[(lc, lr)], np.int64))
            w_l.append(np.ones(len(ids)))
            t_l.append(np.full(len(ids), PORTAL_FRAMES))
        for (qc, qr), _ in links:                               # arrived through it: moves on as a fresh state
            if targets[qr, qc]:
                continue
            one = np.array([qr]), np.array([qc])
            z = np.zeros(1, np.int64)
            s_, d_, w_, t_ = _step(ctx, one[0], one[1], z, z, z, np.array([bool(support[qr, qc])]),
                                   np.array([warp_id[(qc, qr)]], np.int64))
            src_l.append(s_)
            dst_l.append(d_)
            w_l.append(w_)
            t_l.append(t_)
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)
        src, dst, w, wt = cat(src_l, np.int64), cat(dst_l, np.int64), cat(w_l, np.float64), cat(t_l, np.float64)
        # target states: the fresh state of a target cell (arrivals there are collapsed to it), its portal node
        tr, tc = np.nonzero(targets & open_)
        seeds = np.array(list(cid[tr, tc] * K) + [warp_id[q] for q in warp_id if targets[q[1], q[0]]], np.int64)
        V = _cost_to_go(n, src, dst, w, seeds) + seed
        t_t = time.time()
        T = _cost_to_go(n, src, dst, wt, seeds) + seed_t
        self._t_time += time.time() - t_t
        # the support values: the Planner's start rule from every cell (grounded, fresh jump state)
        ro, co = np.nonzero(open_ & ~targets)
        z = np.zeros(len(ro), np.int64)
        s_, d_, w_, t_ = _step(ctx, ro, co, z, z, z, np.ones(len(ro), bool), np.arange(len(ro)))

        def start_rule(val, wts, at_target):
            out = np.full((H, W), np.inf)
            best = np.full(len(ro), np.inf)
            np.minimum.at(best, s_, wts + val[d_])
            out[ro, co] = best
            out[tr, tc] = at_target
            return out
        vs, ts = start_rule(V, w_, seed), start_rule(T, t_, seed_t)
        # airborne: the best reachable airborne state (V: any; T: rising = falling bit 0, falling = bit 1); the
        # fresh state where there is support; closed cells: the start rule
        seeds_up = cid[np.nonzero(enter & support)] * K
        sc, sr = leg_start
        if 0 <= sc < W and 0 <= sr < H and open_[sr, sc] and not targets[sr, sc]:
            k = np.nonzero((ro == sr) & (co == sc))[0]
            if len(k):
                seeds_up = np.concatenate([seeds_up, d_[s_ == k[0]]])
        reach = _reach(n, src, dst, seeds_up)
        air_cells = enter & ~support
        ra, ca = np.nonzero(air_cells)
        rs, cs_ = np.nonzero(enter & support)
        shut = open_ & ~enter

        def air(val, states, sup_val, at_target):
            vr = np.where(reach, val, np.inf)
            cells = vr[:N * K].reshape(N, K)[:, states].min(1)
            for q, i in warp_id.items():
                cells[cid[q[1], q[0]]] = min(cells[cid[q[1], q[0]]], vr[i])
            out = np.full((H, W), np.inf)
            out[ra, ca] = cells[cid[ra, ca]]
            out[rs, cs_] = val[cid[rs, cs_] * K]
            out[shut] = sup_val[shut]
            out[tr, tc] = at_target
            return out
        every = np.arange(K)
        va = air(V, every, vs, seed)
        if K == 1:                                    # no jump limits: no falling bit
            t_rise = t_fall = air(T, every, ts, seed_t)
        else:
            t_rise, t_fall = air(T, every[every % 2 == 0], ts, seed_t), air(T, every[every % 2 == 1], ts, seed_t)
            # a cell where the graph reaches only one of the two (e.g. rising in a shaft it only falls through):
            # the other one's value (the body is there, so the graph's rule is the one that is short)
            t_rise, t_fall = np.where(np.isfinite(t_rise), t_rise, t_fall), np.where(np.isfinite(t_fall), t_fall, t_rise)
        return vs, va, ts, t_rise, t_fall, n, len(src)

    # ---- lookup -----------------------------------------------------------------------------------------
    def leg_of(self, keys_mask) -> np.ndarray:
        """The leg of agents holding keys `keys_mask` (bit k = key k of the level's list): how many of the
        route's key legs, in order, have been picked up (as RewardPath.leg)."""
        m = np.asarray(keys_mask, np.int64)
        out = np.zeros(m.shape, np.int64)
        alive = np.ones(m.shape, bool)
        for ks in self.key_order:
            got = np.zeros(m.shape, bool)
            for k in ks:
                got |= ((m >> k) & 1).astype(bool)
            alive &= got
            out += alive
        return out

    def values(self, xs, ys, leg=0, supported=True) -> np.ndarray:
        """V at points (game units): V_support where `supported` is True, else V_air, on leg `leg` (an int or
        one per point; clamped to the last leg). Points inside walls and in the open half of half blocks: as
        _lookup. +inf outside the map and where the goal cannot be reached."""
        return self._lookup(self._V, np.where(np.asarray(supported, bool), 0, 1), xs, ys, leg, 1.0)

    def times(self, xs, ys, leg=0, supported=True, vy=None) -> np.ndarray:
        """T at points (frames to the goal): T_support where `supported`, else T_rise for a rising body (vy > 0,
        any unit: only the sign is used, y up) and T_fall otherwise; vy None: the smaller of the two (as V_air).
        Leg, points inside walls and in the open half of half blocks: as values() (the same _lookup; the
        distance to the neighbour taken costs FX_GROUND frames per tile)."""
        sup = np.asarray(supported, bool)
        if vy is None:
            return np.minimum(self._lookup(self._T, np.where(sup, 0, 1), xs, ys, leg, FX_GROUND),
                              self._lookup(self._T, np.where(sup, 0, 2), xs, ys, leg, FX_GROUND))
        which = np.where(sup, 0, np.where(np.asarray(vy, np.float64) > 0, 1, 2))
        return self._lookup(self._T, which, xs, ys, leg, FX_GROUND)

    def _lookup(self, stack, which, xs, ys, leg, per_tile) -> np.ndarray:
        """stack[which] (stack: tables x n_legs x H x W) at points (game units), on leg `leg` (an int or one per point;
        clamped to the last leg). A point inside a wall takes its best open neighbour of the 3 x 3 around it (not
        diagonally through a wall corner) plus the distance to that cell's centre (tiles x per_tile). A point in
        the OPEN half of a half-block cell (a wall cell of the field; orientation as realpath.half_normal
        decodes it) takes only the neighbours on its open side: the cell that side faces and the two diagonal
        ones beside it (not through a wall corner), plus the distance; never a neighbour across the solid half (a
        body standing on a floor of half blocks is not valued by the cell below the floor). +inf outside the
        map."""
        xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        shape = np.broadcast(xs, ys, which).shape
        xs, ys = np.broadcast_to(xs, shape).ravel(), np.broadcast_to(ys, shape).ravel()
        leg = np.clip(np.broadcast_to(np.asarray(leg, np.int64), shape).ravel(), 0, self.n_legs - 1)
        tab = np.broadcast_to(np.asarray(which, np.int64), shape).ravel()
        wall = self.pl.cls["full"] if self.mode[3] else self.pl.wall      # the cells the field covers
        H, W = self.H, self.W
        c0 = np.floor((xs - self.ox) / TILE).astype(np.int64)
        r0 = np.floor((ys - self.oy) / TILE).astype(np.int64)

        def free(c, r):
            ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            out = np.zeros(c.shape, bool)
            out[ok] = ~wall[r[ok], c[ok]]
            return out

        def field(c, r, ok, k=None):
            v = np.full(c.shape, np.inf)
            g = np.nonzero(ok)[0]
            p = g if k is None else k[g]
            v[g] = stack[tab[p], leg[p], r[g], c[g]]
            return v
        inside = (c0 >= 0) & (c0 < W) & (r0 >= 0) & (r0 < H)
        own = free(c0, r0)
        out = field(c0, r0, own)
        solid = inside & ~own
        if solid.any():
            k = np.nonzero(solid)[0]
            ck, rk, xk, yk = c0[k], r0[k], xs[k], ys[k]
            # half blocks: the direction (nx, ny) to the solid half; a point on the other side is in the open half
            nx, ny = self._half_n
            nx, ny = nx[rk, ck], ny[rk, ck]
            u, v = (xk - self.ox) / TILE - ck - 0.5, (yk - self.oy) / TILE - rk - 0.5
            open_half = self.pl.cls["half"][rk, ck] & (nx * u + ny * v < 0)
            best = np.full(len(k), np.inf)
            for dc, dr in DIRS:
                c, r = ck + dc, rk + dr
                ok = free(c, r) & (~open_half | (dc * nx + dr * ny < 0))     # open half: its open side only
                if dc and dr:                                  # not diagonally through a wall corner
                    ok &= free(c, rk) | free(ck, r)
                cx, cy = self.ox + (c + 0.5) * TILE, self.oy + (r + 0.5) * TILE
                best = np.minimum(best, field(c, r, ok, k) + per_tile * np.hypot(xk - cx, yk - cy) / TILE)
            out[k] = best
        return out.reshape(shape)

    def _in(self, cell) -> bool:
        return 0 <= cell[0] < self.W and 0 <= cell[1] < self.H

    @property
    def _half_n(self):
        """realpath.half_normal of the level's tile words, once."""
        if getattr(self, "_half_n_cache", None) is None:
            from realpath import half_normal
            self._half_n_cache = tuple(a.astype(np.int64) for a in half_normal(self.pl.words))
        return self._half_n_cache

    def summary(self) -> str:
        return (f"{self.label}, {self.n_legs} leg(s); route cost {self.route_cost:.1f}, "
                f"T at the spawn {self.T_support[0][self.start[1], self.start[0]] if self._in(self.start) else math.inf:.0f} f; "
                f"{max(self.n_states)} states, {max(self.n_edges)} edges per leg; build {self.t_build:.2f} s "
                f"(route search {self.t_route:.2f} s, field {self.t_field:.2f} s)")


def measure_build(pl: Planner):
    """(ProgressField, peak traced memory in bytes) of a build (tracemalloc slows it down: timing apart)."""
    import tracemalloc
    tracemalloc.start()
    try:
        pf = ProgressField.from_planner(pl)
        return pf, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


# ---- graph helpers ----------------------------------------------------------------------------------------
def _step(ctx, r, c, u, s, f, grounded, src):
    """The Planner's moves (realpath.Planner._dijkstra) from source states (cell r, c; rise used u, sideways
    used s, falling f; standing `grounded`), ids `src`: (source ids, target state ids, step costs, step frames).
    Step frames: max(|dc| x fx, |dr| x fy), fx = FX_GROUND from a supported (standing) source state else FX_AIR,
    fy = FY_UP for a step up, FY_DOWN for a step down; none of the cost extras."""
    cid, K, U, S = ctx["cid"], ctx["K"], ctx["U"], ctx["S"]
    enter, support, targets, allowed, cost = ctx["enter"], ctx["support"], ctx["targets"], ctx["allowed"], ctx["cost64"]
    out_s, out_d, out_w, out_t = [], [], [], []
    for d, (dc, dr) in enumerate(DIRS):
        k = np.nonzero(allowed[d, r, c])[0]
        if not len(k):
            continue
        nr, nc = r[k] + dr, c[k] + dc
        ok = enter[nr, nc]                                     # closed cells only when they are the target
        k, nr, nc = k[ok], nr[ok], nc[ok]
        sup = support[nr, nc]
        if K == 1:                                             # no jump limits: every state is the cell
            j = np.zeros(len(k), np.int64)
            g = grounded[k]
        else:
            g = grounded[k]
            up0, side0 = np.where(g, 0, u[k]), np.where(g, 0, s[k])
            u1 = up0 + (1 if dr > 0 else 0)
            s1 = side0 + (1 if dc and dr >= 0 else 0)
            air_ok = (u1 <= U) & (s1 <= S)
            if dr >= 0:
                air_ok &= ~((f[k] == 1) & ~g)                  # falling: only down
            keep = sup | air_ok                                # onto (or against) something: always
            k, nr, nc, sup, u1, s1, g = k[keep], nr[keep], nc[keep], sup[keep], u1[keep], s1[keep], g[keep]
            j = np.where(sup, 0, (u1 * (S + 1) + s1) * 2 + (1 if dr < 0 else 0))
        j = np.where(targets[nr, nc], 0, j)                    # any arrival at a target ends the leg
        ln = SQRT2 if dc and dr else 1.0
        out_s.append(src[k])
        out_d.append(cid[nr, nc] * K + j)
        out_w.append(ln * ((1.0 + np.where(sup, 0.0, AIR_COST)) + cost[nr, nc]))
        fy = 0.0 if dr == 0 else (FY_UP if dr > 0 else FY_DOWN)          # rows go up (y up)
        out_t.append(np.maximum(abs(dc) * np.where(g, FX_GROUND, FX_AIR), fy))
    if not out_s:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    return np.concatenate(out_s), np.concatenate(out_d), np.concatenate(out_w), np.concatenate(out_t)


def _cost_to_go(n, src, dst, w, seeds) -> np.ndarray:
    """Per state, the cheapest cost to any seed state along the edges src -> dst (Dijkstra on the reversed
    edges; +inf where no seed can be reached)."""
    if not len(seeds):
        return np.full(n, np.inf)
    if _sp_dijkstra is not None:
        g = csr_matrix((w, (dst, src)), shape=(n, n))           # no duplicate (src, dst): one per move direction
        return np.asarray(_sp_dijkstra(g, directed=True, indices=seeds, min_only=True), np.float64)
    order = np.argsort(dst, kind="stable")                    # pragma: no cover - fallback without scipy
    rs, rw = src[order].tolist(), w[order].tolist()
    ptr = np.searchsorted(dst[order], np.arange(n + 1)).tolist()
    dist = [math.inf] * n
    pq = []
    for q in seeds.tolist():
        dist[q] = 0.0
        pq.append((0.0, q))
    heapq.heapify(pq)
    while pq:
        d, q = heapq.heappop(pq)
        if d > dist[q]:
            continue
        for e in range(ptr[q], ptr[q + 1]):
            p, nd = rs[e], d + rw[e]
            if nd < dist[p]:
                dist[p] = nd
                heapq.heappush(pq, (nd, p))
    return np.array(dist)


def _reach(n, src, dst, seeds) -> np.ndarray:
    """States reachable from the seeds along src -> dst (breadth first, vectorised)."""
    order = np.argsort(src, kind="stable")
    fd = dst[order]
    ptr = np.searchsorted(src[order], np.arange(n + 1))
    seen = np.zeros(n, bool)
    front = np.unique(np.asarray(seeds, np.int64))
    seen[front] = True
    while len(front):
        a, b = ptr[front], ptr[front + 1]
        cnt = b - a
        tot = int(cnt.sum())
        if not tot:
            break
        idx = np.repeat(a - np.cumsum(cnt) + cnt, cnt) + np.arange(tot)
        nxt = fd[idx]
        nxt = np.unique(nxt[~seen[nxt]])
        seen[nxt] = True
        front = nxt
    return seen


# ---- display (viz/viewer.py: the P key) -------------------------------------------------------------------------
def best_times(field: ProgressField) -> np.ndarray:
    """(n_legs, H, W) float32: per cell the smallest of T_support, T_rise and T_fall, the frames to the goal from
    the cell's best state (+inf in walls and where the goal cannot be reached). What the viewer draws."""
    return np.minimum(field.T_support, np.minimum(field.T_rise, field.T_fall)).astype(np.float32)


def leg_of_keys(key_order, keys_mask: int) -> int:
    """ProgressField.leg_of for one agent, from the key order alone (the viewer has no field object)."""
    leg = 0
    for ks in key_order:
        if not any((int(keys_mask) >> int(k)) & 1 for k in ks):
            break
        leg += 1
    return leg


def descent_path(tbest, ox: float, oy: float, x: float, y: float, leg: int = 0, max_steps: int = 2000) -> list:
    """The line the field leads along from a point: from its cell, step by step to the neighbour (8 around, not
    diagonally between two walls) with the smallest best_times value, while that is smaller; at a leg's key
    cell the next leg's table goes on to the goal. Every cell's best state has a move to a neighbour state whose
    value is lower by the move's frames, so each step lowers the value and the line ends at the goal (or a key
    the last leg cannot leave). [(x, y, frames to go)] at cell centres, game units; [] where the point is
    outside the map or in a cell with no way to the goal."""
    tb = np.asarray(tbest)
    n_legs, H, W = tb.shape
    leg = int(np.clip(leg, 0, n_legs - 1))
    c = int(math.floor((x - ox) / TILE))
    r = int(math.floor((y - oy) / TILE))
    if not (0 <= c < W and 0 <= r < H):
        return []
    t = tb[leg]
    if not np.isfinite(t[r, c]):                     # in a wall or a dead end: the best neighbour to start
        best = None
        for dc, dr in DIRS:
            nc, nr = c + dc, r + dr
            if 0 <= nc < W and 0 <= nr < H and np.isfinite(t[nr, nc]) and (best is None or t[nr, nc] < t[best[1], best[0]]):
                best = (nc, nr)
        if best is None:
            return []
        c, r = best
    out = [(ox + (c + 0.5) * TILE, oy + (r + 0.5) * TILE, float(t[r, c]))]
    for _ in range(max_steps):
        cur = t[r, c]
        best, bv = None, cur
        for dc, dr in DIRS:
            nc, nr = c + dc, r + dr
            if not (0 <= nc < W and 0 <= nr < H):
                continue
            if dc and dr and not (np.isfinite(t[r, nc]) or np.isfinite(t[nr, c])):
                continue
            if t[nr, nc] < bv:
                best, bv = (nc, nr), t[nr, nc]
        if best is None:                             # a key cell (or the goal): the next leg, if any
            if leg + 1 < n_legs and np.isfinite(tb[leg + 1][r, c]):
                leg += 1
                t = tb[leg]
                continue
            break
        c, r = best
        out.append((ox + (c + 0.5) * TILE, oy + (r + 0.5) * TILE, float(bv)))
    return out
