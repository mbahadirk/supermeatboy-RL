"""The planning agent's decision: beam search over named move primitives (plan/moves.py), the world model for
movement, geometric safety for death, the progress field for progress, and a Plan that records what it
predicts and why it was chosen (plan/explain.py). No policy network, no PPO, no learned value.

Search, per decision (Planner.plan):
1. Stage 1 (one ensemble member): every library move (73, 8 to 24 frames; rolled out as one batch to 24
   frames with its last action held, and judged on its own frames) plus the safety check of those frames.
   Scored (below, without the uncertainty term); the best B1 = 8 not vetoed are kept (all vetoed: the ones
   with the latest veto frame).
2. Stage 2 (one member): from each kept end state (the world model's own end rows), each of 16 second moves
   (a fixed diverse core, moves.CORE_SECOND, plus the best-scoring first moves of this decision not in it),
   padded to the horizon H = 48 (last action held, jump released). Only the continuation is rolled out (the
   prefix is shared); hazards for the model inputs at each candidate's own time (offset rows).
3. Final: the top S = 16 full sequences and the previous plan shifted by the frames played (always included)
   are re-scored from the start with ALL ensemble members and the full safety check (plan.safety.check); the
   best is chosen.
A time budget (120 ms) gives stage 1 a deadline at 30% of it and stage 2 one at the budget minus the final
stage's measured cost (a running average; at least 45% of the budget); a cut stage is recorded as cut
(candidates and steps actually evaluated in Plan.counts), never passed off as complete, and its candidates are
ranked on the frames they have. The final stage always runs to the horizon, so a decision can exceed the
budget by what the final stage takes beyond its estimate.

Score of a candidate (higher is better; units of V, about tiles):
- progress = V(start) - V(end) on the agent's current leg (ProgressField.leg_of(keys)). V(end) is V_support
  where the end position is supported (plan/geometry.py: ground or wall), else V_air. An end with V = +inf
  (a dead end) counts DEAD_END below the start. Final stage: the mean over members (each member's own end).
- death: contact within the first 16 frames (plan.safety.VETO_FRAMES) in any member is a hard veto; contact
  or near contact later is a soft risk in [0, 1] (safety's margin from the measured position error), charged
  RISK_WEIGHT x risk.
- uncertainty (final stage only): positive progress is multiplied by exp(-max(0, 2 x spread - 20) / 20), spread
  = the members' position std at the end (units).
- failure memory: FAIL_COST x the number of times this first move failed from this start cell (FailureMemory).
total = progress credit - RISK_WEIGHT x risk - failure penalty; vetoed candidates sort below every other one, by
their veto frame (later is better). No hand-tuned shaping (no hop, explore, stuck or tether terms).

Supported rule (plan/geometry.py, checked by `python -m plan.geometry runs/real_20261102-165351/wm_data --shards 3`
on 3 waves each of 1-5 and 1-13, 331,000 rows): of the frames at rest (vy == 0 now and the next frame) the rule
says ground on 99.7% (1-5) and 98.6% (1-13); flag 0x2 is set on 74% / 92% of all rows, so it is not "airborne".
Ground movement states 5, 10, 15, 4: rule ground 97-99.9% (1-5), 88-99% (1-13); airborne states 6, 7, 8:
0-2%; wall-slide state 9: rule wall 86% / 95%.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from mpc.candidates import ACTIONS
from plan import safety as S
from plan.explain import Plan, SurpriseTracker, HORIZONS
from plan.geometry import Terrain
from plan.moves import LIBRARY, BY_NAME, CORE_SECOND, MAX_LEN, padded
from plan.progress import ProgressField
from wm.buffer import ROW
from wm.features import (FEAT_DIM, N_HAZ, RES_SCALE, EVENTS, LevelGeom, featurize, hazard_features,
                         platform_features, prior)
from wm.model import WorldModel
from wm.rollout import World, advance, _stacked

H_DEFAULT, B1_DEFAULT, BRANCH_DEFAULT, S_DEFAULT, BUDGET_MS = 48, 8, 16, 16, 120.0
STAGE1_SHARE, STAGE2_SHARE = 0.30, 0.45      # of the budget: stage 1's deadline, stage 2's earliest one
RISK_WEIGHT = 10.0
FAIL_COST = 1.0
DEAD_END = 100.0                             # V units charged for an end state with V = +inf
UNC_KNEE, UNC_SCALE = 20.0, 20.0             # uncertainty factor exp(-max(0, 2 spread - KNEE) / SCALE)
NO_PROGRESS = 0.05                           # V units: "no progress" below this
MIN_EXPECT = 0.5                             # FailureMemory: an expected V drop below this expects nothing
VETO_SCORE = -1e6
BOX_MARGIN = 70.0                            # units around the candidates' paths: hazards further are not checked
N_ALTERNATIVES = 5
HAZ_FAR = 300.0                              # wm.features.hazard_features clips offsets and distances here
FINAL_MS0 = 60.0                             # first guess of the final stage's cost (then measured, averaged)


# ---- world model rollouts with per-row time offsets -------------------------------------------------------
@torch.no_grad()
def imagine(model, members: slice, geom: LevelGeom, s0: np.ndarray, actions: np.ndarray, dt_ms: float,
            worlds: list, offsets=None, deadline: float | None = None, keep_rows: bool = False) -> dict:
    """wm.rollout.rollout_batched for a subset of ensemble members (`members`, a slice) whose rows may start at
    different future frames: row i at step k sees the level's moving parts worlds[offsets[i] + k] (clamped to
    the last). s0 (n,) ROW, actions (n, K). Returns x, y (E, n, K+1) float64, alive (E, n, K+1) (the learned
    death head: a second opinion only), rows (E, n, K+1) when keep_rows, and steps (how many steps ran before
    the deadline; positions after that are held at the last one)."""
    Wl, Bl = _stacked(model)
    Wl, Bl = [w[members] for w in Wl], [b[members] for b in Bl]
    E = Wl[0].shape[0]
    n, K = actions.shape
    s = np.concatenate([s0] * E)
    a_all = np.tile(actions, (E, 1))
    off = np.zeros(n, np.int64) if offsets is None else np.asarray(offsets, np.int64)
    off_all = np.tile(off, E)
    groups = [(int(o), np.nonzero(off_all == o)[0]) for o in np.unique(off_all)]
    xs = np.zeros((E, n, K + 1))
    ys = np.zeros((E, n, K + 1))
    alive = np.ones((E, n, K + 1), np.float32)
    rows = np.zeros((E, n, K + 1), s0.dtype) if keep_rows else None
    xs[:, :, 0], ys[:, :, 0] = s["x"].reshape(E, n), s["y"].reshape(E, n)
    if rows is not None:
        rows[:, :, 0] = s.reshape(E, n)
    rs = torch.as_tensor(RES_SCALE)
    X = np.empty((E * n, FEAT_DIM), np.float32)
    al = np.ones(E * n, np.float32)
    done = K
    last = len(worlds) - 1
    for k in range(K):
        features_into(X, groups, s, a_all[:, k], dt_ms, geom, [worlds[min(o + k, last)] for o, _ in groups])
        z = ((torch.as_tensor(X) - model.mu) / model.sd).view(E, n, -1)
        for i in range(len(Wl) - 1):
            z = torch.nn.functional.silu(torch.baddbmm(Bl[i], z, Wl[i]))
        h = model.split(torch.baddbmm(Bl[-1], z, Wl[-1]).reshape(E * n, -1))
        move = (h["mean"] * rs).numpy() + prior(s, dt_ms)
        pb = torch.sigmoid(h["bin"]).numpy()
        s = advance(s, a_all[:, k], move, pb, h["state"].argmax(-1).numpy())
        al = al * (1.0 - pb[:, EVENTS["death"]])
        xs[:, :, k + 1], ys[:, :, k + 1] = s["x"].reshape(E, n), s["y"].reshape(E, n)
        alive[:, :, k + 1] = al.reshape(E, n)
        if rows is not None:
            rows[:, :, k + 1] = s.reshape(E, n)
        if deadline is not None and time.perf_counter() > deadline and k + 1 < K:
            done = k + 1
            break
    if done < K:
        xs[:, :, done + 1:] = xs[:, :, done:done + 1]
        ys[:, :, done + 1:] = ys[:, :, done:done + 1]
        alive[:, :, done + 1:] = alive[:, :, done:done + 1]
    return {"x": xs, "y": ys, "alive": alive, "rows": rows, "steps": done}


HAZ_COLS = slice(FEAT_DIM - 6 - 5 * N_HAZ, FEAT_DIM - 6)     # wm.features.featurize: [..., hazards, platform]
PLAT_COLS = slice(FEAT_DIM - 6, FEAT_DIM)


def features_into(X, groups, s, a, dt_ms, geom, worlds):
    """wm.features.featurize for rows in groups [(offset, row indices)] that see different worlds: one call for
    every row with the first group's world, then only the hazard and platform columns (the parts that depend on
    the world besides wind) per other group. A group whose wind (fans) differs gets a full call."""
    w0 = worlds[0]
    if len(groups) == 1:
        idx = groups[0][1]
        X[idx] = featurize(s[idx], a[idx], dt_ms, geom, _model_boxes(w0.boxes, s["x"][idx], s["y"][idx]), w0.pf,
                           w0.pf_prev, w0.dt_prev_s, w0.fans)
        return
    X[:] = featurize(s, a, dt_ms, geom, np.zeros((0, 5)), w0.pf, w0.pf_prev, w0.dt_prev_s, w0.fans)
    for (o, idx), w in zip(groups, worlds):
        x, y = s["x"][idx].astype(np.float64), s["y"][idx].astype(np.float64)
        if w.fans != w0.fans:
            X[idx] = featurize(s[idx], a[idx], dt_ms, geom, _model_boxes(w.boxes, x, y), w.pf, w.pf_prev,
                               w.dt_prev_s, w.fans)
            continue
        X[idx, HAZ_COLS] = hazard_features(x, y, _model_boxes(w.boxes, x, y))
        X[idx, PLAT_COLS] = platform_features(x, y, w.pf, w.pf_prev, w.dt_prev_s)


def _model_boxes(boxes: np.ndarray, x, y) -> np.ndarray:
    """The hazards the model can tell apart for rows at x, y: wm.features gives each row its 3 nearest boxes,
    with offsets and distance clipped at HAZ_FAR units. Kept: boxes within HAZ_FAR (plus their size) of the
    rows' bounding box, and always the 3 nearest to its centre (as mpc.controller._relevant). Only a row with
    fewer than 3 hazards within HAZ_FAR can see a different (far, clipped) third box than with all of them."""
    if len(boxes) <= 3:
        return boxes
    x0, x1, y0, y1 = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    size = np.hypot(boxes[:, 2], boxes[:, 3])
    dx = np.maximum(np.maximum(x0 - boxes[:, 0], boxes[:, 0] - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - boxes[:, 1], boxes[:, 1] - y1), 0.0)
    keep = np.hypot(dx, dy) - size <= HAZ_FAR
    if keep.all():
        return boxes
    keep[np.argsort(np.hypot(boxes[:, 0] - 0.5 * (x0 + x1), boxes[:, 1] - 0.5 * (y0 + y1)))[:3]] = True
    return boxes[keep]


# ---- hazards per future frame ---------------------------------------------------------------------------
def future_inputs(future, frame: int, H: int, dt_ms: float, level_name: str | None = None, deaths=None):
    """From an mpc.future.FutureWorld that observed this frame: the model inputs per step (H World, index k =
    frame + k) and the deadly boxes per frame (H + 1 plan.safety.FrameBoxes, index 0 unused) with their
    measured kill reach and the future's uncertainty radius, flying shots, and remembered deaths (a
    plan.safety.DeathMemory) at their wave frames. Also how many steps came from the recording."""
    worlds, check, recorded = future.worlds(frame, H, dt_ms)
    _, hz0, _, sh0, _ = future.last
    keep = future.keep
    ids_h = [f"h{k}" for k in range(len(hz0)) if keep is None or k >= len(keep) or keep[k]]
    ids_s = [f"shot{s[0]}" for s in sh0]
    frames = [None]
    for k, (b, r) in enumerate(check, start=1):
        b = np.asarray(b, np.float64).reshape(-1, 5).copy()
        if len(b):
            b[:, 2:4] += np.asarray(r, np.float64)[:, None]
        n_sh = len(ids_s) if len(b) >= len(ids_s) else 0
        nh = len(b) - n_sh
        ids = (ids_h[:nh] if len(ids_h) >= nh else [f"h{j}" for j in range(nh)]) + ids_s[:n_sh]
        kind = [S.KIND_HAZARD] * nh + [S.KIND_SHOT] * n_sh
        if deaths is not None and level_name is not None:
            mem = deaths.boxes(level_name, frame + k)
            if len(mem):
                b = np.concatenate([b, mem])
                kind += [S.KIND_MEMORY] * len(mem)
                ids += [f"death@{x:.0f},{y:.0f}" for x, y in mem[:, :2]]
        frames.append(S.FrameBoxes(b, np.asarray(kind, np.int8), ids))
    return worlds, frames, recorded


def _near_frames(frames, xs, ys, margin=BOX_MARGIN):
    """The frames' boxes that can matter for these paths: centres within the paths' bounding box grown by the
    box size + BODY + margin (the soft-risk margin reaches 60 units at 48 frames)."""
    x0, x1, y0, y1 = float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys))
    out = []
    for fb in frames:
        if fb is None or not len(fb):
            out.append(fb)
            continue
        b = fb.boxes
        reach = np.hypot(b[:, 2], b[:, 3]) + S.BODY + margin
        k = (b[:, 0] >= x0 - reach) & (b[:, 0] <= x1 + reach) & (b[:, 1] >= y0 - reach) & (b[:, 1] <= y1 + reach)
        if k.all():
            out.append(fb)
        else:
            idx = np.nonzero(k)[0]
            out.append(S.FrameBoxes(b[idx], fb.kind[idx], [fb.ids[i] for i in idx]))
    return out


def _verdict(sw: S.Sweep, upto: np.ndarray):
    """The decision rule (plan.safety.assess) for candidates judged on their own first `upto` frames (one or
    more members): veto, veto frame, what was hit, soft risk and its frame."""
    E, n, K = sw.step_clear.shape
    cf = sw.contact_frame
    lim = np.minimum(upto, S.VETO_FRAMES)[None, :]
    early = (cf >= 1) & (cf <= lim)
    veto = early.any(0)
    vf = np.where(early, cf, np.iinfo(np.int64).max)
    m = vf.argmin(0)
    veto_frame = np.where(veto, vf[m, np.arange(n)], -1)
    veto_kind = np.where(veto, sw.contact_kind[m, np.arange(n)], S.KIND_NONE)
    veto_id = [None if not veto[c] else (S.OUT_ID if veto_kind[c] == S.KIND_OUT else sw.ids[sw.contact_box[m[c], c]])
               for c in range(n)]
    soft = np.zeros(n)
    soft_frame = np.full(n, -1, np.int64)
    if K > S.VETO_FRAMES:
        ks = np.arange(S.VETO_FRAMES + 1, K + 1)
        c = sw.step_clear[:, :, S.VETO_FRAMES:]
        with np.errstate(invalid="ignore", divide="ignore"):
            risk = np.where(c < 0, 1.0, np.clip(1.0 - c / S.margin(ks), 0.0, 1.0))
        risk = np.nan_to_num(risk, nan=0.0)
        risk = np.where(ks[None, None, :] <= upto[None, :, None], risk, 0.0).max(0)
        j = risk.argmax(1)
        soft = risk[np.arange(n), j]
        soft_frame = np.where(soft > 0, ks[j], -1)
    return veto, veto_frame, veto_kind, veto_id, soft, soft_frame


def hit_reason(kind, hid, frame) -> str:
    if kind == S.KIND_OUT:
        return f"out of level at frame {frame}"
    if kind == S.KIND_MEMORY:
        return f"hits remembered {hid} at frame {frame}"
    return f"hits {hid} at frame {frame}"


# ---- failure memory --------------------------------------------------------------------------------------
@dataclass
class _Expect:
    start: int
    end: int
    move: str
    cell: tuple
    actions: np.ndarray
    v_start: float
    drop: float


class FailureMemory:
    """Moves that did not deliver the progress their plan predicted, per (start cell, move name).

    Planner.plan calls expect() with every chosen plan: its first move, the start cell, V at the start and the
    predicted V drop by the move's end. The expectation stays while later plans keep playing the same move
    (their actions continue it); a plan that changes the move drops it (abandoned, not failed). update() (the
    runner, or Planner.observe) gives the real V each frame; once the move has been played to its end, a
    real drop below half the expected one (or a death) counts a failure. Expected drops below MIN_EXPECT expect
    nothing. Penalty: FAIL_COST x count for plans starting with that move from that cell."""

    def __init__(self, fail_cost: float = FAIL_COST):
        self.fail_cost = fail_cost
        self.counts: dict[tuple, int] = {}
        self.pending: dict[int, _Expect] = {}

    def count(self, cell, move: str) -> int:
        return self.counts.get((tuple(int(v) for v in cell), move), 0)

    def penalty(self, cell, move: str) -> float:
        return self.fail_cost * self.count(cell, move)

    def reason(self, cell, move: str) -> str:
        return f"{move} from ({int(cell[0])},{int(cell[1])}) failed {self.count(cell, move)} times"

    def expect(self, agent: int, frame: int, move: str, move_frames: int, actions, cell, v_start, v_end_pred):
        p = self.pending.get(agent)
        actions = np.asarray(actions, np.int64)
        if p is not None and frame < p.end:
            k = p.end - frame
            if np.array_equal(actions[:k], p.actions[frame - p.start:]):
                return                                   # still playing the same move
        if v_start is None or v_end_pred is None or not (math.isfinite(v_start) and math.isfinite(v_end_pred)) \
                or v_start - v_end_pred < MIN_EXPECT or move_frames <= 0:
            self.pending.pop(agent, None)
            return
        self.pending[agent] = _Expect(int(frame), int(frame) + int(move_frames), move,
                                      tuple(int(v) for v in cell), actions[:move_frames].copy(), float(v_start),
                                      float(v_start - v_end_pred))

    def update(self, agent: int, frame: int, v_real: float | None, dead: bool = False) -> list[dict]:
        """The agent's real V at this frame (None / inf when unknown). Returns the failures counted now."""
        p = self.pending.get(agent)
        if p is None or (frame < p.end and not dead):
            return []
        del self.pending[agent]
        real = None if dead or v_real is None or not math.isfinite(v_real) else p.v_start - float(v_real)
        if real is not None and real >= p.drop / 2:
            return []
        key = (p.cell, p.move)
        self.counts[key] = self.counts.get(key, 0) + 1
        return [{"kind": "failed move", "agent": int(agent), "frame": int(frame), "move": p.move, "cell": list(p.cell),
                 "expected_drop": round(p.drop, 3), "real_drop": None if real is None else round(real, 3),
                 "dead": bool(dead), "count": self.counts[key], "reason": self.reason(p.cell, p.move)}]


# ---- the planner ------------------------------------------------------------------------------------------
def state_row(x, y, vx, vy, flags, state, keys=0, held=0) -> np.ndarray:
    """One agent row (wm.buffer.ROW) from the game's state values."""
    r = np.zeros(1, ROW)
    r["x"], r["y"], r["vx"], r["vy"] = x, y, vx, vy
    r["flags"], r["state"], r["status"], r["keys"], r["held"] = int(flags) & 0xffffffff, state, 0, int(keys), held
    return r


class Planner:
    """One planner per level (several agents may share it; the previous plan is kept per agent)."""

    def __init__(self, model, level, field: ProgressField, H: int = H_DEFAULT, B1: int = B1_DEFAULT,
                 branch: int = BRANCH_DEFAULT, S_final: int = S_DEFAULT, budget_ms: float = BUDGET_MS,
                 member: int = 0, memory: FailureMemory | None = None):
        self.model = WorldModel.load(model).eval() if isinstance(model, str) else model
        self.lv = level
        self.field = field
        self.H, self.B1, self.branch, self.S, self.budget = H, B1, branch, S_final, budget_ms
        self.member = member
        self.geom = LevelGeom(level.words0, {"ox": level.ox, "oy": level.oy})
        self.words = np.asarray(level.words0, np.uint32).copy()
        self.terrain = Terrain(level.words0, level.ox, level.oy)
        self.bounds = S.level_bounds(level)
        self.memory = memory if memory is not None else FailureMemory()
        self.surprise = SurpriseTracker()
        self.prev: dict[int, Plan] = {}
        self.final_ms = FINAL_MS0
        self.lib = list(LIBRARY)
        self.lib_len = np.array([len(m) for m in self.lib])
        self.lib_actions = np.stack([padded(m.actions, MAX_LEN) for m in self.lib])

    # ---- level state -------------------------------------------------------------------------------------
    def tiles(self, changes):
        """Tile changes from the game [(cell index, word)]: the model's tile inputs and the support geometry."""
        if not changes:
            return
        for idx, word in changes:
            self.geom.set_tile(idx, word)
            r, c = divmod(int(idx), self.terrain.W)
            if 0 <= r < self.terrain.H:
                self.words[r, c] = word
        self.terrain.update(self.words)

    def new_wave(self):
        self.prev = {}
        self.memory.pending = {}

    def cell(self, x, y) -> tuple:
        return int((x - self.lv.ox) // 20.0), int((y - self.lv.oy) // 20.0)

    def value(self, x, y, keys=0):
        """V at positions (arrays) for an agent holding `keys`: (V, supported, ground, wall side, leg)."""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        g, side = self.terrain.support(x, y, self.lv.plats or None)
        sup = g | (side != 0)
        leg = int(self.field.leg_of(np.int64(int(keys))))
        return self.field.values(x, y, leg, sup), sup, g, side, leg

    def observe(self, agent: int, frame: int, x: float, y: float, keys: int = 0, dead: bool = False) -> list[dict]:
        """The runner calls this with every real state: surprise events (SurpriseTracker) and failed moves
        (FailureMemory.update)."""
        ev = self.surprise.observe(agent, frame, x, y, dead)
        v = None if dead else float(self.value(np.array([x]), np.array([y]), keys)[0][0])
        return ev + self.memory.update(agent, frame, v, dead)

    # ---- one decision ------------------------------------------------------------------------------------
    def plan(self, row, frame: int, worlds: list, frames: list, dt_ms: float = 16.667, agent: int = 0) -> Plan:
        """row: the agent's state (wm.buffer.ROW, one row; state_row builds one). worlds: at least H World
        (model inputs, index k = frame + k); frames: H + 1 plan.safety.FrameBoxes (index k = frame + k; index 0
        unused). Returns the chosen Plan (also kept as this agent's previous plan)."""
        t0 = time.perf_counter()
        H = self.H
        row = np.asarray(row, ROW).reshape(1)
        x0, y0, keys = float(row["x"][0]), float(row["y"][0]), int(row["keys"][0])
        v0a, sup0, g0, side0, leg = self.value(np.array([x0]), np.array([y0]), keys)
        v0 = float(v0a[0])
        v0_ok = math.isfinite(v0)
        cell0 = self.cell(x0, y0)
        mem = slice(self.member, self.member + 1)
        timing, counts = {}, {}

        def progress_of(vend):
            vend = np.asarray(vend, np.float64)
            base = v0 if v0_ok else 0.0
            return np.where(np.isfinite(vend), base - vend, -DEAD_END if v0_ok else -1e3)

        def ends_value(xe, ye):
            return self.value(xe, ye, keys)[0]

        # ---- stage 1: every library move, one member ------------------------------------------------------
        n1 = len(self.lib)
        K1 = MAX_LEN
        out1 = imagine(self.model, mem, self.geom, np.repeat(row, n1), self.lib_actions, dt_ms, worlds,
                       deadline=t0 + STAGE1_SHARE * self.budget / 1000.0, keep_rows=True)
        steps1 = out1["steps"]
        up1 = np.minimum(self.lib_len, steps1)
        fr1 = _near_frames(frames[:steps1 + 1], out1["x"][:, :, :steps1 + 1], out1["y"][:, :, :steps1 + 1])
        sw1 = S.sweep(out1["x"][:, :, :steps1 + 1], out1["y"][:, :, :steps1 + 1], fr1, self.bounds)
        veto1, vf1, vk1, vid1, soft1, sf1 = _verdict(sw1, up1)
        xe1, ye1 = out1["x"][0, np.arange(n1), up1], out1["y"][0, np.arange(n1), up1]
        vend1 = ends_value(xe1, ye1)
        prog1 = progress_of(vend1)
        fail1 = np.array([self.memory.penalty(cell0, m.name) for m in self.lib])
        raw1 = prog1 - RISK_WEIGHT * soft1 - fail1
        score1 = np.where(veto1, VETO_SCORE + vf1 + 1e-3 * prog1, raw1)
        order1 = np.argsort(-score1, kind="stable")
        keep1 = order1[:self.B1]
        t1 = time.perf_counter()
        timing["stage1"] = 1000 * (t1 - t0)
        counts["stage1"] = {"candidates": n1, "steps": int(steps1), "complete": int((self.lib_len <= steps1).sum()),
                            "vetoed": int(veto1.sum())}

        # ---- stage 2: second moves from the kept end states, one member ------------------------------------
        seconds = [BY_NAME[nm] for nm in CORE_SECOND]
        for i in order1:
            if len(seconds) >= self.branch:
                break
            if self.lib[i].name not in CORE_SECOND and not veto1[i]:
                seconds.append(self.lib[i])
        seconds = seconds[:self.branch]
        cands = []                                       # (segments, actions H)
        stage2 = steps1 >= self.lib_len[keep1].max() if len(keep1) else False
        if stage2:
            K2 = H - int(self.lib_len[keep1].min())
            s0, acts2, offs, firsts = [], [], [], []
            for i in keep1:
                L = int(self.lib_len[i])
                for sm in seconds:
                    full = padded(np.concatenate([self.lib[i].array, sm.array]), H)
                    cont = padded(full[L:], K2)
                    s0.append(out1["rows"][0, i, L])
                    acts2.append(cont)
                    offs.append(L)
                    firsts.append(i)
                    L2 = min(H, L + len(sm))
                    segs = [(self.lib[i].name, 0, L), (sm.name, L, L2)] + ([("pad", L2, H)] if L2 < H else [])
                    cands.append((segs, full))
            s0 = np.array(s0, ROW)
            dl2 = t0 + max(STAGE2_SHARE * self.budget, self.budget - self.final_ms) / 1000.0
            out2 = imagine(self.model, mem, self.geom, s0, np.stack(acts2), dt_ms, worlds, offsets=offs, deadline=dl2)
            steps2 = out2["steps"]
            n2 = len(cands)
            xs2, ys2 = np.zeros((1, n2, H + 1)), np.zeros((1, n2, H + 1))
            up2 = np.zeros(n2, np.int64)
            for c in range(n2):
                i, L = firsts[c], offs[c]
                m = min(steps2, H - L)
                xs2[0, c, :L + 1], ys2[0, c, :L + 1] = out1["x"][0, i, :L + 1], out1["y"][0, i, :L + 1]
                xs2[0, c, L + 1:L + m + 1], ys2[0, c, L + 1:L + m + 1] = out2["x"][0, c, 1:m + 1], out2["y"][0, c, 1:m + 1]
                xs2[0, c, L + m + 1:], ys2[0, c, L + m + 1:] = xs2[0, c, L + m], ys2[0, c, L + m]
                up2[c] = L + m
            K = int(up2.max())
            fr2 = _near_frames(frames[:K + 1], xs2[:, :, :K + 1], ys2[:, :, :K + 1])
            sw2 = S.sweep(xs2[:, :, :K + 1], ys2[:, :, :K + 1], fr2, self.bounds)
            veto2, vf2, _, _, soft2, _ = _verdict(sw2, up2)
            vend2 = ends_value(xs2[0, np.arange(n2), up2], ys2[0, np.arange(n2), up2])
            prog2 = progress_of(vend2)
            fail2 = np.array([fail1[i] for i in firsts])
            score2 = np.where(veto2, VETO_SCORE + vf2 + 1e-3 * prog2, prog2 - RISK_WEIGHT * soft2 - fail2)
            pick = np.argsort(-score2, kind="stable")[:self.S]
            final = [cands[c] for c in pick]
            counts["stage2"] = {"candidates": n2, "steps": int(steps2), "of_steps": int(K2),
                                "complete": int((up2 >= H).sum()), "vetoed": int(veto2.sum()),
                                "second_moves": [sm.name for sm in seconds]}
        else:
            final = []
            for i in order1[:self.S]:
                L = int(self.lib_len[i])
                final.append(([(self.lib[i].name, 0, L)] + ([("pad", L, H)] if L < H else []),
                              padded(self.lib[i].array, H)))
            counts["stage2"] = {"candidates": 0, "steps": 0, "skipped": "stage 1 was cut by the deadline"}
        t2 = time.perf_counter()
        timing["stage2"] = 1000 * (t2 - t1)

        # ---- final: re-score with all members and the full check -------------------------------------------
        prev = self.prev.get(agent)
        prev_idx = -1
        if prev is not None and 0 < frame - prev.frame < H:
            shift = frame - prev.frame
            segs = [(nm, max(0, a - shift), b - shift) for nm, a, b in prev.segments if b - shift > 0 and nm != "pad"]
            acts = padded(np.asarray(prev.actions, np.int64)[shift:], H)
            end = segs[-1][2] if segs else 0
            if end < H:
                segs.append(("pad", end, H))
            dup = [k for k, (_, a) in enumerate(final) if np.array_equal(a, acts)]
            if dup:
                prev_idx = dup[0]
            else:
                final.append((segs, acts))
                prev_idx = len(final) - 1
        n3 = len(final)
        acts3 = np.stack([a for _, a in final])
        out3 = imagine(self.model, slice(None), self.geom, np.repeat(row, n3), acts3, dt_ms, worlds)
        E = out3["x"].shape[0]
        fr3 = _near_frames(frames[:H + 1], out3["x"], out3["y"])
        saf = S.check(out3["x"], out3["y"], fr3, self.bounds)
        xe, ye = out3["x"][:, :, H], out3["y"][:, :, H]
        vend_e = ends_value(xe, ye)                                  # (E, n3)
        prog_e = progress_of(vend_e)
        prog3 = prog_e.mean(0)
        spread_end = np.sqrt(xe.var(0) + ye.var(0))
        unc = np.exp(-np.maximum(0.0, 2 * spread_end - UNC_KNEE) / UNC_SCALE)
        credit = np.where(prog3 > 0, prog3 * unc, prog3)
        first_name = [next((nm for nm, a, b in segs if nm != "pad"), "pad") for segs, _ in final]
        fail3 = np.array([self.memory.penalty(cell0, nm) for nm in first_name])
        risk3 = RISK_WEIGHT * saf.soft_risk
        raw3 = credit - risk3 - fail3
        total3 = np.where(saf.veto, VETO_SCORE + saf.veto_frame + 1e-3 * raw3, raw3)
        best = int(np.argmax(total3))
        t3 = time.perf_counter()
        timing["final"] = 1000 * (t3 - t2)
        self.final_ms = 0.7 * self.final_ms + 0.3 * timing["final"]
        counts["final"] = {"candidates": n3, "members": E, "steps": H, "vetoed": int(saf.veto.sum()),
                           "prev_included": prev_idx >= 0}

        # ---- explanation -----------------------------------------------------------------------------------
        segs, acts = final[best]
        mx, my = out3["x"][:, best].mean(0), out3["y"][:, best].mean(0)
        spread = np.sqrt(out3["x"][:, best].var(0) + out3["y"][:, best].var(0))
        g, side = self.terrain.support(mx, my, self.lv.plats or None)
        sup = g | (side != 0)
        landings = [{"frame": k, "x": float(mx[k]), "y": float(my[k]), "on": "ground" if g[k] else "wall"}
                    for k in range(1, H + 1) if sup[k] and not sup[k - 1]]
        jump = np.array([ACTIONS[a][2] for a in acts], bool)
        wall_jumps = [{"frame": k, "x": float(mx[k]), "y": float(my[k]), "side": "L" if side[k] < 0 else "R"}
                      for k in range(H) if side[k] != 0 and not g[k] and jump[k] and (k == 0 or not jump[k - 1])]
        fin_e = np.isfinite(vend_e[:, best])
        vend_mean = float(np.mean(vend_e[fin_e, best])) if fin_e.any() else None
        chosen_parts = {"progress": float(prog3[best]), "credit": float(credit[best]), "risk": float(risk3[best]),
                        "fail": float(fail3[best]), "unc": float(unc[best]), "spread": float(spread_end[best])}
        alts = self._alternatives(final, best, saf, prog3, credit, risk3, fail3, unc, spread_end, total3, vend_e,
                                  v0, cell0, first_name, veto1, vf1, vk1, vid1, prog1, vend1, keep1)
        veto = None
        if saf.veto[best]:
            veto = {"frame": int(saf.veto_frame[best]), "id": saf.veto_id[best],
                    "kind": S.KIND_NAMES[int(saf.veto_kind[best])]}
        first_frames = next((b for nm, a, b in segs if nm != "pad"), 0)
        plan = Plan(
            frame=int(frame), agent=int(agent),
            state={"x": x0, "y": y0, "vx": float(row["vx"][0]), "vy": float(row["vy"][0]),
                   "flags": int(row["flags"][0]), "state": int(row["state"][0]), "held": int(row["held"][0]),
                   "keys": keys, "ground": bool(g0[0]), "wall_side": int(side0[0]), "supported": bool(sup0[0]),
                   "cell": list(cell0)},
            chosen=[nm for nm, a, b in segs if nm != "pad"], actions=[int(a) for a in acts],
            pad=int(sum(b - a for nm, a, b in segs if nm == "pad")), from_prev=best == prev_idx,
            path=[[float(a), float(b)] for a, b in zip(mx, my)], spread=[float(v) for v in spread],
            landings=landings, wall_jumps=wall_jumps, nearest=saf.nearest[best], veto=veto,
            soft_risk=float(saf.soft_risk[best]), soft_frame=int(saf.soft_frame[best]),
            score={"progress": chosen_parts["progress"], "progress_credit": chosen_parts["credit"],
                   "risk": chosen_parts["risk"], "uncertainty": chosen_parts["unc"],
                   "spread_end": chosen_parts["spread"], "failure": chosen_parts["fail"],
                   "total": float(total3[best])},
            v_start=v0 if v0_ok else None, v_end=vend_mean, dead_end_members=int((~fin_e).sum()), leg=leg,
            end_supported=bool(sup[H]),
            alternatives=alts,
            predictions={h: [float(mx[h]), float(my[h])] for h in HORIZONS if h <= H},
            death_model=float(1.0 - out3["alive"][:, best, H].min()),
            timing={**{k: round(v, 2) for k, v in timing.items()}, "total": round(1000 * (time.perf_counter() - t0), 2)},
            counts=counts, segments=[[nm, int(a), int(b)] for nm, a, b in segs])
        plan.first_move_frames = int(first_frames)
        self.prev[agent] = plan
        self.surprise.add(plan)
        if first_frames > 0 and v0_ok:
            vm = self.value(np.array([mx[first_frames]]), np.array([my[first_frames]]), keys)[0][0]
            self.memory.expect(agent, frame, plan.chosen[0], first_frames, acts, cell0, v0, float(vm))
        return plan

    def _alternatives(self, final, best, saf, prog3, credit, risk3, fail3, unc, spread_end, total3, vend_e, v0,
                      cell0, first_name, veto1, vf1, vk1, vid1, prog1, vend1, keep1) -> list[dict]:
        """The top N_ALTERNATIVES rejected plans, each with the one reason that separates it from the chosen one:
        the final stage's next best by score (up to N_ALTERNATIVES - 2 of them first), then the stage-1 moves the
        safety check vetoed with the most progress (up to 2), then more of the final stage."""
        names = lambda segs: [nm for nm, a, b in segs if nm != "pad"]
        vstr = lambda v: "inf" if not math.isfinite(v) else f"{v:.1f}"

        def final_alt(c):
            vend = float(vend_e[:, c].mean())
            if saf.veto[c]:
                reason = hit_reason(int(saf.veto_kind[c]), saf.veto_id[c], int(saf.veto_frame[c]))
            elif prog3[c] <= NO_PROGRESS:
                reason = f"no progress (V {vstr(v0)} -> {vstr(vend)})"
            else:
                gaps = {"progress": credit[best] - credit[c], "risk": risk3[c] - risk3[best],
                        "fail": fail3[c] - fail3[best]}
                why = max(gaps, key=gaps.get)
                if why == "fail" and fail3[c] > 0:
                    reason = self.memory.reason(cell0, first_name[c])
                elif why == "risk" and risk3[c] > 0:
                    near = saf.nearest[c][0]["id"] if saf.nearest[c] else "?"
                    reason = (f"risk {saf.soft_risk[c]:.2f} near {near} at frame {int(saf.soft_frame[c])} "
                              f"(chosen {saf.soft_risk[best]:.2f})")
                else:
                    reason = f"lower score than chosen: progress {prog3[c]:.1f} vs {prog3[best]:.1f}"
                    if unc[c] < 0.9:
                        reason += f" (uncertain: spread {spread_end[c]:.0f}u, credit x{unc[c]:.2f})"
            return {"moves": names(final[c][0]), "total": float(total3[c]), "progress": float(prog3[c]),
                    "stage": "final", "reason": reason}

        def stage1_alt(i):
            return {"moves": [self.lib[i].name], "total": None, "progress": float(prog1[i]), "stage": "stage 1",
                    "reason": hit_reason(int(vk1[i]), vid1[i], int(vf1[i]))}

        fin = [int(c) for c in np.argsort(-total3, kind="stable") if c != best]
        vet = [int(i) for i in np.argsort(-prog1, kind="stable") if veto1[i]]
        n_fin = min(len(fin), max(N_ALTERNATIVES - min(2, len(vet)), 0))
        n_vet = min(len(vet), N_ALTERNATIVES - n_fin)
        return [final_alt(c) for c in fin[:n_fin]] + [stage1_alt(i) for i in vet[:n_vet]]
