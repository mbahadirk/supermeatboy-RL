"""Wave search in the real game (PLAN_GAME_SEARCH.md section 5; agent.py --search): the game is the simulator.

With the time-step lock a wave starts from the same map and the same actions from the wave start give the same
states, so a wave of N agents tests N action sequences with exact physics. No world model.

- Node: the actions from the wave start, the frame they end at (the state in that frame's message is the result
  of actions[0 .. end-1]) and the real state there (x, y, vx, vy, keys, supported: plan.geometry.Terrain on the
  live tiles). Kept per cell key (tile cell, supported, sign of vx / vy beyond plan.planner.KEY_SPEED, keys:
  plan.planner.Planner.state_key plus the keys); a cell keeps the node that reaches it in the fewest frames.
- Selection: frontier nodes ordered by T (frames to the goal, plan/progress.py, on the field agent.py built),
  plus EXPAND_W frames per branch already played from the node and DEATH_W frames x the share of those branches
  that died. Nodes whose replay plus a branch would not fit in the game's wave are left out.
- Expansion: agent 0 is the verifier (below); the others are dealt round-robin over the best AGENTS_PER_NODE-sized
  share of picks. Each replays its node's actions exactly, then plays a chain of 1-3 library moves
  (plan/moves.py), or (PERTURB_P, when the archive holds a longer sequence through the node) a perturbed copy
  (plan.demos.perturb) of that sequence's rest. After the first trusted finish FINISH_SHARE of the agents play
  perturbed copies of the fastest finish instead; the fastest trusted finish is kept.
- Insertion (after the wave): an agent's states at its move ends (perturbed copies: every SEG frames) become nodes
  when their cell key is new or reached sooner. Not inserted: a state NEAR_DEATH frames or less before the
  agent's death (or before the last frame seen, for an agent alive when the wave was cut) and an untrusted one.
- Shared tiles (the guard). Measured (step 2, 196 waves of 41-50 agents): crumble blocks are shared, the first
  agent that touches one breaks it for everyone CRUMBLE_L frames later; one agent's replay reproduces exactly.
  A state is trusted (self-contained) only when every special tile (crumble, lock, switch, other: bit 8) that
  changed by its frame and that its agent touched by then (body centre within TOUCH units of the cell, so touched,
  entered or stood next to) changed at that agent's own first touch + CRUMBLE_L, crumble only. A change caused by
  another agent's earlier touch, a crumble block the agent reached after it broke, and a change of a non-crumble
  special tile (no clear toucher) make it untrusted. Tile words are compared without the open-neighbour flags
  (bits 22-25, EDGE_BITS), which change when a neighbour breaks. Levels without special tiles need no check.
- Verification: agent 0 replays the current best node (the fastest trusted finish, else the frontier node with
  the lowest T) and its state at the node's end is compared with the record (VERIFY_POS units, VERIFY_VEL units / s,
  same keys; a finish: the same finish frame). A mismatch marks the node invalid (out of the archive; logged,
  "interference" when another agent first touched a special tile on the node's recorded path before the node's
  agent had, else "mismatch"). Then agent 0 expands that node like any agent.
- Wave length: every agent's action sequence (prefix + branch) runs out, then SETTLE frames (so each insertion
  point has NEAR_DEATH frames seen after it): agent.py ends the wave there ("search done"); earlier when all are
  done. After an agent's sequence it waits (action 0).

The archive is per level for the run, kept in memory only (never saved): held-out levels can be searched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from plan.demos import perturb
from plan.geometry import Terrain
from plan.moves import LIBRARY, MAX_LEN
from plan.planner import Planner
from realpath import tile_classes

TILE = 20.0
CRUMBLE_L = (29, 30)          # frames from the first touch to the break (measured, locked, step 2)
TOUCH = 7.0                   # units from the body centre to the cell box: a touch (step 2's definition)
EDGE_BITS = 0x3C00000         # bits 22-25 of a tile word: which neighbours are open (change when a neighbour breaks)
NEAR_DEATH = 10               # frames: a state this close before a death is not inserted
SETTLE = NEAR_DEATH           # frames played (waiting) after the longest sequence ran out
SEG = 8                       # perturbed copies: an insertion point every SEG frames
AGENTS_PER_NODE = 5           # agents per picked node
CHAIN = (1, 3)                # library moves per branch (inclusive)
PERTURB_P = 0.25              # branch = a perturbed copy of a longer known sequence's rest (when there is one)
FINISH_SHARE = 0.5            # after the first finish: share of the agents (verifier aside) on finish copies
EXPAND_W = 1.0                # selection: frames added per branch already played from a node
DEATH_W = 60.0                # selection: frames added x the share of the node's branches that died
T_UNREACHABLE = 1e6           # T of a node the field cannot route from (inf)
VERIFY_POS = 0.5              # units
VERIFY_VEL = 30.0             # units / s (0.5 units / frame)
KEEP_FINISHES = 20            # trusted finishes kept per level (the fastest)
MAX_BRANCH = max(CHAIN) * MAX_LEN


@dataclass
class Node:
    id: int
    actions: list
    end: int
    state: tuple                  # (x, y, vx, vy, keys); vx, vy in units / s as the game sends them
    supported: bool
    key: tuple
    parent: int | None
    wave: int
    agent: int
    path: dict = dc_field(default_factory=dict)       # special cell index -> its agent's first touch frame (<= end)
    finish: bool = False
    expanded: int = 0
    died: int = 0
    verified: int = 0
    invalid: str | None = None
    children: list = dc_field(default_factory=list)

    def brief(self) -> dict:
        return {"node": self.id, "end": self.end, "x": round(self.state[0], 1), "y": round(self.state[1], 1)}


class Archive:
    """One level's nodes: cell key -> the soonest node; the trusted finishes, fastest first."""

    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.cells: dict[tuple, int] = {}
        self.finishes: list[int] = []
        self.next_id = 0
        self.root: Node | None = None

    def new(self, **kw) -> Node:
        n = Node(id=self.next_id, **kw)
        self.next_id += 1
        self.nodes[n.id] = n
        if n.parent is not None and n.parent in self.nodes:
            self.nodes[n.parent].children.append(n.id)
        return n

    def insert(self, **kw) -> str | None:
        """'new' (a new cell), 'sooner' (replaced a slower node there) or None (a node there is as fast)."""
        cur = self.cells.get(kw["key"])
        if cur is not None and self.nodes[cur].end <= kw["end"]:
            return None
        n = self.new(**kw)
        self.cells[n.key] = n.id
        return "new" if cur is None else "sooner"

    def add_finish(self, **kw) -> Node | None:
        """A trusted finish; kept when among the KEEP_FINISHES fastest. Returns the node (None: not kept)."""
        ends = [self.nodes[i].end for i in self.finishes]
        if len(ends) >= KEEP_FINISHES and kw["end"] >= max(ends):
            return None
        if kw["end"] in ends:
            return None                                       # the same finish frame again: one record
        n = self.new(finish=True, **kw)
        self.finishes = sorted(self.finishes + [n.id], key=lambda i: self.nodes[i].end)[:KEEP_FINISHES]
        return n

    def frontier(self) -> list[Node]:
        """The nodes holding a cell (the root's cell keeps the root: nothing reaches it sooner than frame 0)."""
        return [self.nodes[i] for i in self.cells.values()]

    def best_finish(self) -> Node | None:
        return self.nodes[self.finishes[0]] if self.finishes else None

    def invalidate(self, n: Node, why: str):
        n.invalid = why
        if self.cells.get(n.key) == n.id:
            del self.cells[n.key]
        if n.id in self.finishes:
            self.finishes.remove(n.id)

    def longest_through(self, n: Node) -> Node | None:
        """The deepest valid node whose actions extend n's (a descendant), None without one."""
        best, todo = None, list(n.children)
        while todo:
            c = self.nodes[todo.pop()]
            todo += c.children
            if c.invalid is None and c.end > n.end and (best is None or c.end > best.end):
                best = c
        return best


class Search:
    def __init__(self, n_agents: int, seed: int = 0, wave_frames: int = 3600):
        self.n = int(n_agents)
        self.rng = np.random.default_rng(seed)
        self.wave_frames = int(wave_frames)
        self.archives: dict[str, Archive] = {}
        self.arch: Archive | None = None
        self.level = None
        self.active = False
        self.skipped = None

    # ---- level ---------------------------------------------------------------------------------------------
    def set_level(self, lv):
        """A level (re)loaded: its archive (kept for the run, per level name), the special cells and the support
        geometry. A wave in progress is dropped."""
        self.level = lv
        self.arch = self.archives.setdefault(lv.name, Archive())
        w0 = np.asarray(lv.words0, np.uint32)
        k = tile_classes(w0)
        special = k["crumble"] | k["lock"] | k["switch"] | k["other"]
        rr, cc = np.nonzero(special)
        self.cell_rc = list(zip(rr.tolist(), cc.tolist()))
        self.cell_index = {lv.W * r + c: j for j, (r, c) in enumerate(self.cell_rc)}
        self.cell_x0 = lv.ox + cc * TILE
        self.cell_y0 = lv.oy + rr * TILE
        self.crumble = k["crumble"][rr, cc] if len(rr) else np.zeros(0, bool)
        self.word0 = (w0[rr, cc] & ~np.uint32(EDGE_BITS)) if len(rr) else np.zeros(0, np.uint32)
        self.guard = bool(len(rr))
        self.terrain = Terrain(lv.words0, lv.ox, lv.oy)
        self.abort_wave()

    def abort_wave(self):
        self.active = False
        self.skipped = None

    # ---- wave start: assignments --------------------------------------------------------------------------
    def node_t(self, nodes, field) -> np.ndarray:
        """T (frames to the goal) of nodes on the current field; T_UNREACHABLE where it is inf."""
        if not nodes or field is None:
            return np.full(len(nodes), T_UNREACHABLE)
        x = np.array([n.state[0] for n in nodes])
        y = np.array([n.state[1] for n in nodes])
        vy = np.array([n.state[3] for n in nodes])
        sup = np.array([n.supported for n in nodes])
        leg = field.leg_of(np.array([int(n.state[4]) for n in nodes], np.int64))
        t = field.times(x, y, leg, sup, vy)
        return np.where(np.isfinite(t), t, T_UNREACHABLE)

    def _chain(self) -> tuple[list, list, list]:
        """A chain of library moves: (actions, move names, the frames after each move, from 0)."""
        acts, names, ends = [], [], []
        for _ in range(int(self.rng.integers(CHAIN[0], CHAIN[1] + 1))):
            m = LIBRARY[int(self.rng.integers(len(LIBRARY)))]
            acts += list(m.actions)
            names.append(m.name)
            ends.append(len(acts))
        return acts, names, ends

    def _expand(self, node: Node) -> dict:
        """One agent's branch from node: a library chain or a perturbed copy of a longer sequence's rest."""
        P = node.end
        longer = self.arch.longest_through(node) if not node.finish else None
        if longer is not None and longer.end + SETTLE < self.wave_frames and self.rng.random() < PERTURB_P:
            rest, ch = perturb(longer.actions[P:], self.rng)
            seq = list(node.actions) + list(rest)
            pts = list(range(P + SEG, len(seq), SEG)) + [len(seq)]
            return {"node": node.id, "prefix": P, "seq": seq, "points": pts, "kind": "perturbed",
                    "from": longer.id, "changes": ch}
        acts, names, ends = self._chain()
        return {"node": node.id, "prefix": P, "seq": list(node.actions) + acts, "points": [P + e for e in ends],
                "kind": "library", "moves": names}

    def begin_wave(self, wave: int, frame: int, states, field):
        """The first frame message of a wave (frame 0: every agent at the spawn): the root on the level's first
        wave, then each agent's action sequence."""
        n = len(states)
        self.wave, self.field = int(wave), field
        self.active = True
        if frame != 0 or self.arch is None:
            self.skipped = "joined mid-wave" if frame != 0 else "no level"
            self.plan = [{"node": None, "prefix": 0, "seq": [], "points": [], "kind": "idle"} for _ in range(n)]
            self.end_frame = 0
            return
        self.skipped = None
        A = self.arch
        if A.root is None:
            s = states[0]
            st = self._state(s)
            sup = bool(self.terrain.supported(np.array([st[0]]), np.array([st[1]]), self.level.plats or None)[0])
            A.root = A.new(actions=[], end=0, state=st, supported=sup, key=self.key(st, sup), parent=None,
                           wave=self.wave, agent=0)
            A.cells[A.root.key] = A.root.id
        fin = A.best_finish()
        self.replay = fin is not None                 # a recorded finish is replayed this wave (verifier, copies)
        front = [x for x in A.frontier() if x.end + MAX_BRANCH + SETTLE < self.wave_frames]
        t = self.node_t(front, field)
        self.t_of = {x.id: float(v) for x, v in zip(front, t)}
        best = fin
        if best is None and front:
            j = min(range(len(front)), key=lambda k: (t[k], front[k].end))
            best = front[j] if front[j].end > 0 else None
        self.verify = best
        score = [float(v) + EXPAND_W * x.expanded + DEATH_W * (x.died / x.expanded if x.expanded else 0.0)
                 for x, v in zip(front, t)]
        order = [front[k] for k in sorted(range(len(front)), key=lambda k: (score[k], front[k].end, front[k].id))]
        plan = [None] * n
        if n and best is not None:
            plan[0] = ({"node": best.id, "prefix": best.end, "seq": list(best.actions), "points": [], "kind": "verify"}
                       if best.finish else dict(self._expand(best), kind="verify"))
        rest = [i for i in range(n) if plan[i] is None]
        n_copy = int(round(FINISH_SHARE * len(rest))) if fin is not None else 0
        for i in rest[:n_copy]:
            seq, ch = perturb(fin.actions, self.rng)
            first = min((c["frame"] for c in ch), default=0)
            p0 = max(SEG, first - first % SEG)
            plan[i] = {"node": fin.id, "prefix": first, "seq": seq, "kind": "finish copy", "changes": ch,
                       "points": [f for f in range(p0, len(seq), SEG) if f > first]}
        rest = rest[n_copy:]
        k = max(1, math.ceil(len(rest) / AGENTS_PER_NODE)) if rest else 0
        picks = order[:k] or ([A.root] if A.root is not None else [])
        for j, i in enumerate(rest):
            plan[i] = self._expand(picks[j % len(picks)])
        self.plan = plan
        self.points = [set(p["points"]) for p in plan]
        self.picked = {}
        for p in plan:
            if p["node"] is not None and p["kind"] != "finish copy":
                self.picked[p["node"]] = self.picked.get(p["node"], 0) + 1
        self.end_frame = max((len(p["seq"]) for p in plan), default=0) + SETTLE
        self.touch = np.full((n, len(self.cell_rc)), -1, np.int64)
        self.change = np.full(len(self.cell_rc), -1, np.int64)
        self.cap = [{} for _ in range(n)]                 # agent -> {frame: (state, supported)}
        self.death = [None] * n
        self.fin = [None] * n
        self.fin_xy = [None] * n
        self.last_frame = 0

    # ---- every frame ---------------------------------------------------------------------------------------
    @staticmethod
    def _state(s) -> tuple:
        return (float(s[0]), float(s[1]), float(s[2]), float(s[3]), int(s[7]) if len(s) > 7 else 0)

    def key(self, st: tuple, supported: bool) -> tuple:
        lv = self.level
        cell = (int((st[0] - lv.ox) // TILE), int((st[1] - lv.oy) // TILE))
        to_f = 1.0 / 60.0                                  # units / s -> units / frame (the lock's step)
        return Planner.state_key(cell, supported, st[2] * to_f, st[3] * to_f) + (int(st[4]),)

    def action(self, i: int, frame: int) -> int:
        if not self.active or i >= len(self.plan):
            return 0
        s = self.plan[i]["seq"]
        return int(s[frame]) if 0 <= frame < len(s) else 0

    def observe(self, frame: int, states, tc=None):
        """A frame message (or the wave end's final states): tile changes of special cells, first touches,
        deaths and the states at the agents' insertion points."""
        if not self.active or self.skipped:
            return
        frame = int(frame)
        self.last_frame = max(self.last_frame, frame)
        if tc:
            for idx, w in tc:
                j = self.cell_index.get(int(idx))
                if j is not None and self.change[j] < 0 and (int(w) & ~EDGE_BITS) != int(self.word0[j]):
                    self.change[j] = frame
            self.terrain.update(self.level.words)          # agent.py applied the changes to the level's words
        n = min(len(states), len(self.plan))
        alive = [i for i in range(n) if int(states[i][6]) == 0]
        for i in range(n):
            if int(states[i][6]) == 1 and self.death[i] is None:
                self.death[i] = frame
        if not alive:
            return
        if len(self.cell_rc):
            x = np.array([float(states[i][0]) for i in alive])[:, None]
            y = np.array([float(states[i][1]) for i in alive])[:, None]
            dx = np.maximum(np.maximum(self.cell_x0[None, :] - x, 0.0), x - self.cell_x0[None, :] - TILE)
            dy = np.maximum(np.maximum(self.cell_y0[None, :] - y, 0.0), y - self.cell_y0[None, :] - TILE)
            hit = np.hypot(dx, dy) <= TOUCH
            sub = self.touch[alive]
            sub[(sub < 0) & hit] = frame
            self.touch[alive] = sub
        want = [i for i in alive if frame in self.points[i] or
                (i == 0 and self.verify is not None and not self.verify.finish and frame == self.verify.end)]
        if want:
            xs = np.array([float(states[i][0]) for i in want])
            ys = np.array([float(states[i][1]) for i in want])
            sup = self.terrain.supported(xs, ys, self.level.plats or None)
            for i, s in zip(want, sup):
                self.cap[i][frame] = (self._state(states[i]), bool(s))

    def finished(self, i: int, frames: int, x=None, y=None):
        """Agent i touched the goal at (x, y): its finish takes `frames` updates (actions[0 .. frames-1])."""
        if self.active and not self.skipped and 0 <= i < len(self.fin) and self.fin[i] is None:
            self.fin[i] = int(frames)
            self.fin_xy[i] = (float(x) if x is not None else math.nan, float(y) if y is not None else math.nan)

    # ---- the guard -----------------------------------------------------------------------------------------
    def trusted(self, i: int, end: int) -> tuple[bool, str | None]:
        """Agent i's state at frame `end` is self-contained: every special tile that changed by then and that i
        touched by then changed at i's own first touch + CRUMBLE_L (crumble only)."""
        if not self.guard:
            return True, None
        t, c = self.touch[i], self.change
        for j in np.nonzero((c >= 0) & (c <= end) & (t >= 0) & (t <= end))[0]:
            if not self.crumble[j]:
                return False, "special tile changed (no clear toucher)"
            lag = int(c[j] - t[j])
            if lag < 0:
                return False, "crumble broken before its touch"
            if lag < CRUMBLE_L[0]:
                return False, "crumble broken by another agent"
            if lag > CRUMBLE_L[1]:
                return False, "crumble break late"
        return True, None

    def path_of(self, i: int, end: int) -> dict:
        t = self.touch[i]
        return {int(j): int(t[j]) for j in np.nonzero((t >= 0) & (t <= end))[0]}

    def _verify(self) -> dict | None:
        b = self.verify
        if b is None:
            return None
        rec = {"node": b.id, "end": b.end, "finish": b.finish}
        if b.finish:
            got = self.fin[0]
            ok = got == b.end
            rec["got"] = {"finish_frames": got}
        else:
            c = self.cap[0].get(b.end)
            ok = c is not None and c[0][4] == b.state[4] and \
                abs(c[0][0] - b.state[0]) <= VERIFY_POS and abs(c[0][1] - b.state[1]) <= VERIFY_POS and \
                abs(c[0][2] - b.state[2]) <= VERIFY_VEL and abs(c[0][3] - b.state[3]) <= VERIFY_VEL
            rec["got"] = None if c is None else {"x": round(c[0][0], 2), "y": round(c[0][1], 2), "vx": round(c[0][2], 2),
                                                "vy": round(c[0][3], 2), "keys": c[0][4]}
            rec["want"] = {"x": round(b.state[0], 2), "y": round(b.state[1], 2), "vx": round(b.state[2], 2),
                           "vy": round(b.state[3], 2), "keys": b.state[4]}
        rec["ok"] = bool(ok)
        if ok:
            b.verified += 1
            return rec
        others = [j for j, t0 in b.path.items() if any(0 <= self.touch[a, j] < t0 for a in range(1, len(self.plan)))]
        rec["label"] = "interference" if others else "mismatch"
        rec["cells"] = [list(self.cell_rc[j][::-1]) for j in others]      # (column, row)
        self.arch.invalidate(b, rec["label"])
        return rec

    # ---- wave end: insertion -------------------------------------------------------------------------------
    def wave_end(self, frame: int) -> dict:
        """After the wave's last update (and its final states, if any, went through observe): verification,
        finishes, new nodes and the selection statistics. Returns the search.jsonl record."""
        if not self.active:
            return {"skipped": "no wave"}
        self.active = False
        if self.skipped:
            return {"skipped": self.skipped}
        A, frame = self.arch, int(frame)
        n = len(self.plan)
        ver = self._verify()
        counts = {"new": 0, "sooner": 0, "untrusted": 0, "near_death": 0, "unseen": 0}
        why = {}
        fins = {"trusted": [], "untrusted": 0, "new_best": False}
        best_before = A.best_finish()
        for i in range(n):
            p = self.plan[i]
            d, f = self.death[i], self.fin[i]
            if p["node"] is not None and p["kind"] != "finish copy" and p["node"] in A.nodes:
                nd = A.nodes[p["node"]]
                nd.expanded += 1
                nd.died += int(d is not None)
            if f is not None:
                ok, r = self.trusted(i, f)
                if ok:
                    fins["trusted"].append(f)
                    A.add_finish(actions=list(p["seq"][:f]), end=f, state=self.fin_xy[i] + (0.0, 0.0, 0), supported=False,
                                 key=("finish", f), parent=p["node"], wave=self.wave, agent=i, path=self.path_of(i, f))
                else:
                    fins["untrusted"] += 1
                    why[r] = why.get(r, 0) + 1
            limit = f if f is not None else (d - NEAR_DEATH - 1 if d is not None else self.last_frame - NEAR_DEATH)
            for e in p["points"]:
                if f is not None and e >= f:
                    continue
                if e > limit:
                    counts["near_death" if d is not None else "unseen"] += 1
                    continue
                c = self.cap[i].get(e)
                if c is None:
                    counts["unseen"] += 1
                    continue
                ok, r = self.trusted(i, e)
                if not ok:
                    counts["untrusted"] += 1
                    why[r] = why.get(r, 0) + 1
                    continue
                st, sup = c
                got = A.insert(actions=list(p["seq"][:e]), end=e, state=st, supported=sup, key=self.key(st, sup),
                               parent=p["node"], wave=self.wave, agent=i, path=self.path_of(i, e))
                if got:
                    counts[got] += 1
        bf = A.best_finish()
        fins["new_best"] = bf is not None and (best_before is None or bf.end < best_before.end)
        front = A.frontier()
        t = self.node_t(front, self.field)
        j = int(np.argmin(t)) if len(t) else None
        kinds = {}
        for p in self.plan:
            kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
        return {"picked": [dict(A.nodes[k].brief(), agents=v, T=0.0 if A.nodes[k].finish else round(self.t_of.get(k, T_UNREACHABLE), 1))
                           for k, v in self.picked.items() if k in A.nodes],
                "branches": kinds, "inserted": counts, "untrusted_why": why, "guard": self.guard,
                "breaks": int((self.change >= 0).sum()), "verification": ver,
                "finishes": {"trusted": sorted(fins["trusted"]), "untrusted": fins["untrusted"],
                             "new_best": fins["new_best"]},
                "best_finish_frames": bf.end if bf is not None else None,
                "best_T": round(float(t[j]), 1) if j is not None else None,
                "best_node": front[j].brief() if j is not None else None,
                "archive": len(A.cells), "nodes": len(A.nodes), "replay": self.replay, "frames": frame,
                "end_frame": self.end_frame}


def text(level: str, wave: int, r: dict, dt_ms: float = 1000.0 / 60) -> str:
    """The console line of a search wave."""
    if r.get("skipped"):
        return f"[search] {level} wave {wave}: skipped ({r['skipped']})"
    ins, v, f = r["inserted"], r["verification"], r["finishes"]
    ts = [p["T"] for p in r["picked"]]
    vt = "verify -" if v is None else (f"verify node {v['node']} " + ("ok" if v["ok"] else f"FAILED ({v['label']})"))
    bf = r["best_finish_frames"]
    return (f"[search] {level} wave {wave}: picked {len(r['picked'])} nodes"
            + (f" (T {min(ts):.0f}..{max(ts):.0f})" if ts else "")
            + ", branches " + ", ".join(f"{k} {n}" for k, n in sorted(r["branches"].items()))
            + f" | new cells {ins['new']}, sooner {ins['sooner']}, untrusted {ins['untrusted']}, near death "
              f"{ins['near_death']} | best T {r['best_T'] if r['best_T'] is not None else '-'} f, archive {r['archive']}"
            + f" | {vt} | finishes {len(f['trusted'])}" + (f" (+{f['untrusted']} untrusted)" if f["untrusted"] else "")
            + (f", best {bf * dt_ms / 1000.0:.2f} s" + (" NEW" if f["new_best"] else "") if bf is not None else ""))
