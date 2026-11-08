"""Commitment to the chosen plan (PLAN_PLANNER_ROUND2.md C3, synchronous): a planned agent plays its committed
plan and gets a new decision only when there is a reason (a trigger). agent.py and the offline closed loop
(plan/offline.py --closed-loop) use the same rules.

Triggers, checked every frame (Committer.triggers), each recorded with the decision it causes:
- start: no committed plan (level or wave start, first frame alive), or "start: tiles" (the tile map changed);
- interval: --replan frames since the last decision (default 8);
- move end: the committed plan's first move (not "pad") has been played to its end;
- surprise: the real position is off the committed plan's prediction for this frame by more than
  max(tube x the 90% error of followed predictions at that horizon, MIN_TUBE) (ErrorTube: the run's own
  followed errors once a horizon has MIN_SAMPLES of them, else plan.explain.ERR90_AT);
- support: the real support (plan/geometry.py rule on the real position: ground, wall or air) matches none of
  the plan's predicted ones for this frame and the frames just before and after (landed early, not landed,
  lost the wall, on a wall);
- danger: the committed plan's remaining path (its predicted mean path), re-checked against this frame's
  deadly boxes (the planner's own: plan.planner.future_inputs), touches something within plan.safety.VETO_FRAMES
  frames that the plan did not already accept at its decision (a plan chosen while vetoed fires again only
  when the contact comes earlier), or its soft risk rose by more than DANGER_RISE over its value at the
  decision. Checked only when no other trigger fired (it is the only one that costs a little: about 1-2 ms);
- short: fewer than SHORT_FRAMES frames of the plan remain;
- missed: the plan touched the princess or picked a key in its prediction and the touch frame (+1 frame for
  the game's report) passed without it.
With triggers off (agent.py --triggers off) only the fixed interval counts: the old behaviour (with --replan 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from plan import safety as S
from plan.explain import ERR90_AT, HORIZONS

REPLAN_DEFAULT = 8
TUBE_DEFAULT = 1.5
SHORT_FRAMES = 16
DANGER_RISE = 0.3            # soft risk rise (0..1) over the decision's own value that counts as new danger
MIN_SAMPLES = 200            # followed errors at a horizon before the run's own 90% replaces ERR90_AT
KEEP_SAMPLES = 5000          # followed errors kept per horizon (the newest)
# units: the surprise tube is never narrower. Well under the body's 6-unit half size (realpath.BODY), above the
# live followed 90% error at +1 (0.9, runs/plan_20261103-162219) and about the one at +4 (3.4); without it the
# early +1 tube from ERR90_AT (1.5 x 0.43 = 0.65) fires on ordinary frames until the run has its own errors
MIN_TUBE = 3.0
SOFT_REACH = 60.0            # units: boxes further from the path than this (plus their size) cannot matter
KINDS = ("start", "interval", "move end", "surprise", "support", "danger", "short", "missed")


def kind_of(trigger: str) -> str:
    """'surprise +8 err 14.2 > 11.0' -> 'surprise'."""
    for k in KINDS:
        if trigger.startswith(k):
            return k
    return trigger.split()[0] if trigger else "?"


def tube_horizons(step: int = 1) -> tuple:
    """The tube's measured horizons for a world model of `step` frames per step: plan.explain.HORIZONS that are
    whole steps (1-frame model: 1, 4, 8, 16; step-4 model: 4, 8, 16)."""
    return tuple(h for h in HORIZONS if h % max(1, int(step)) == 0)


class ErrorTube:
    """90% position error (units) of followed predictions per horizon (frames after a decision).

    Measured horizons are plan.explain.HORIZONS (1, 4, 8, 16): the run's own followed errors
    (SurpriseTracker.err, added at every wave end with add(), plus the current wave's with refresh()) once a
    horizon has MIN_SAMPLES of them, else ERR90_AT. Between horizons linear; beyond 16 frames the model's
    measured table plan.safety.ERR90 (32, 48 frames). horizons: the measured ones (a step-4 world model:
    4, 8, 16, the frames its predictions are made at; below the first the first one's error holds)."""

    def __init__(self, horizons=HORIZONS):
        self.horizons = tuple(horizons)
        self.samples = {h: [] for h in self.horizons}
        self.table = {}
        self.refresh()

    def add(self, err: dict):
        """A finished wave's followed errors {horizon: [errors]}."""
        for h in self.horizons:
            s = self.samples[h]
            s.extend(float(e) for e in (err.get(h) or []))
            if len(s) > KEEP_SAMPLES:
                del s[:len(s) - KEEP_SAMPLES]
        self.refresh()

    def refresh(self, current: dict | None = None):
        """Recompute the table from the run's samples plus the current wave's (not kept)."""
        self.table = {}
        for h in self.horizons:
            e = self.samples[h] + [float(v) for v in ((current or {}).get(h) or [])]
            if len(e) >= MIN_SAMPLES:
                self.table[h] = (float(np.percentile(e, 90)), "run", len(e))
            else:
                self.table[h] = (float(ERR90_AT[h]), "ERR90_AT", len(e))
        far = [(f, e) for f, e in S.ERR90 if f > max(self.horizons)]
        self._f = np.array([h for h in self.horizons] + [f for f, _ in far], np.float64)
        self._e = np.array([self.table[h][0] for h in self.horizons] + [e for _, e in far], np.float64)

    def at(self, k: int) -> float:
        return float(np.interp(float(k), self._f, self._e))

    def summary(self) -> dict:
        """{horizon: [90% error, source, samples]} for logs."""
        return {str(h): [round(v, 2), src, n] for h, (v, src, n) in self.table.items()}


def support_kind(ground, side) -> np.ndarray:
    """0 air, 1 ground, 2 wall (on a wall and not on the ground)."""
    g = np.asarray(ground, bool)
    return np.where(g, 1, np.where(np.asarray(side) != 0, 2, 0))


SUPPORT_TEXT = {(1, 0): "landed early", (0, 1): "not on the ground", (0, 2): "lost the wall", (1, 2): "lost the wall",
                (2, 0): "on a wall", (2, 1): "on a wall"}


def path_check(xs, ys, frames, bounds=None):
    """plan.safety's rule for one path (frame 0 = now, K more): each step k-1 -> k sampled at most
    plan.safety.SUB_STEP apart against frames[k]'s boxes (and the level's edge), in one vectorized pass.
    Returns (contact frame or -1, id of what was hit or None, soft risk in [0, 1] over frames after
    VETO_FRAMES)."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    K = min(len(xs), len(frames)) - 1
    if K < 1:
        return -1, None, 0.0
    xs, ys = xs[:K + 1], ys[:K + 1]
    seg = np.hypot(np.diff(xs), np.diff(ys))
    nk = np.maximum(1, np.ceil(seg / S.SUB_STEP)).astype(np.int64)    # samples per step (as safety.sweep)
    n = int(nk.max())
    j = np.arange(1, n + 1, dtype=np.float64)
    t = np.minimum(j[None, :], nk[:, None]) / nk[:, None]             # short steps repeat their end point
    px = xs[:-1, None] + np.diff(xs)[:, None] * t                     # (K, n): step k-1 -> k in row k-1
    py = ys[:-1, None] + np.diff(ys)[:, None] * t
    step = np.full(K, np.inf)
    hit_id = [None] * K
    parts = [(k, fb) for k in range(1, K + 1) for fb in (frames[k],) if fb is not None and len(fb)]
    if parts:
        B = np.concatenate([fb.boxes for _, fb in parts])
        fk = np.concatenate([np.full(len(fb), k) for k, fb in parts])
        ids = [i for _, fb in parts for i in fb.ids]
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        reach = np.hypot(B[:, 2], B[:, 3]) + S.BODY + SOFT_REACH
        near = np.nonzero((B[:, 0] >= x0 - reach) & (B[:, 0] <= x1 + reach) & (B[:, 1] >= y0 - reach)
                          & (B[:, 1] <= y1 + reach))[0]
        if len(near):
            B, fk = B[near], fk[near]
            c, s = np.cos(B[:, 4])[:, None], np.sin(B[:, 4])[:, None]
            dx = px[fk - 1] - B[:, 0:1]
            dy = py[fk - 1] - B[:, 1:2]
            u, v = dx * c + dy * s, -dx * s + dy * c
            clear = (np.hypot(np.maximum(np.abs(u) - B[:, 2:3], 0.0),
                              np.maximum(np.abs(v) - B[:, 3:4], 0.0)) - S.BODY).min(1)
            np.minimum.at(step, fk - 1, clear)
            for j in np.nonzero(clear < 0)[0]:
                k = int(fk[j]) - 1
                if clear[j] <= step[k] and hit_id[k] is None:
                    hit_id[k] = ids[near[j]]
    if bounds is not None:
        oc = S.out_clearance(bounds, px, py).min(1)
        for k in np.nonzero((oc < 0) & ~(step < 0))[0]:      # a box contact is reported before the edge
            hit_id[k] = S.OUT_ID
        step = np.minimum(step, oc)
    hits = np.nonzero(step < 0)[0]
    cf = int(hits[0]) + 1 if len(hits) else -1
    risk = 0.0
    if K > S.VETO_FRAMES:
        ks = np.arange(S.VETO_FRAMES + 1, K + 1)
        c = step[S.VETO_FRAMES:]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(c < 0, 1.0, np.clip(1.0 - c / S.margin(ks), 0.0, 1.0))
        risk = float(np.nan_to_num(r, nan=0.0).max())
    return cf, (hit_id[cf - 1] if cf > 0 else None), risk


@dataclass
class Committed:
    """A planned agent's committed plan and what was known at its decision: the predicted support per frame
    (plan/geometry.py rule on the predicted mean path) and the mean path's own safety at the decision (the
    contact it already accepted, as an absolute frame, and its soft risk)."""
    plan: object
    move_end: int = 0
    support: np.ndarray | None = None
    base_contact: int | None = None
    base_risk: float = 0.0
    xs: np.ndarray = field(default=None, repr=False)
    ys: np.ndarray = field(default=None, repr=False)


class Committer:
    """Per planned agent: the committed plan and when to decide again (the module's rules)."""

    def __init__(self, replan: int = REPLAN_DEFAULT, tube: float = TUBE_DEFAULT, on: bool = True, step: int = 1):
        self.replan = max(1, int(replan))
        self.tube_factor = float(tube)
        self.on = bool(on)
        self.tube = ErrorTube(tube_horizons(step))
        self.cur: dict[int, Committed] = {}

    def reset(self):
        """A new wave or level: no committed plans."""
        self.cur = {}

    def drop(self, agent: int):
        self.cur.pop(int(agent), None)

    def decided(self, agent: int, plan, frame: int, fboxes=None, planner=None):
        """A new decision for this agent at `frame` (made with these deadly boxes fboxes, index k = frame + k)."""
        c = Committed(plan, move_end=int(getattr(plan, "first_move_frames", 0) or 0))
        if self.on and planner is not None:
            xy = np.asarray(plan.path, np.float64)
            c.xs, c.ys = xy[:, 0], xy[:, 1]
            g, side = planner.terrain.support(c.xs, c.ys, planner.lv.plats or None)
            c.support = support_kind(g, side)
            if fboxes is not None:
                cf, _, risk = path_check(c.xs, c.ys, fboxes, planner.bounds)
                c.base_contact = frame + cf if 0 < cf <= S.VETO_FRAMES else None
                c.base_risk = risk
        self.cur[int(agent)] = c

    def triggers(self, agent: int, frame: int, x: float, y: float, keys: int = 0, tiles: bool = False,
                 planner=None, fboxes_fn=None) -> list[str]:
        """Why this agent needs a decision now ([] = keep playing the committed plan). fboxes_fn(): this
        frame's deadly boxes (called only for the danger check)."""
        c = self.cur.get(int(agent))
        if c is None:
            return ["start"]
        p = c.plan
        k = frame - p.frame
        n = len(p.actions)
        if not self.on:                                           # the old fixed interval
            return ["interval"] if (k < 0 or k >= self.replan or k >= n) else []
        if k < 0 or k >= n:
            return ["start"]
        out = ["start: tiles"] if tiles else []
        if k >= self.replan:
            out.append("interval")
        if 0 < c.move_end <= k:
            out.append("move end")
        if c.xs is not None and 1 <= k < len(c.xs):
            err = math.hypot(x - c.xs[k], y - c.ys[k])
            thr = max(self.tube_factor * self.tube.at(k), MIN_TUBE)
            if err > thr:
                out.append(f"surprise +{k} err {err:.1f} > {thr:.1f}")
        if c.support is not None and planner is not None and 1 <= k < len(c.support):
            g, side = planner.terrain.support(np.array([x]), np.array([y]), planner.lv.plats or None)
            real = int(support_kind(g, side)[0])
            near = c.support[max(0, k - 1):k + 2]
            if real not in near:
                pred = int(c.support[k])
                out.append(f"support: {SUPPORT_TEXT.get((real, pred), 'changed')} +{k}")
        if n - k < SHORT_FRAMES:
            out.append("short")
        gt = getattr(p, "goal_touch", None)
        if gt and k > int(gt["frame"]) + 1:
            out.append(f"missed: princess f{int(gt['frame'])}")
        for kt in getattr(p, "key_touches", None) or []:
            if k > int(kt["frame"]) + 1 and not (int(keys) >> int(kt["key"])) & 1:
                out.append(f"missed: key {int(kt['key'])} f{int(kt['frame'])}")
                break
        if not out and fboxes_fn is not None and c.xs is not None and planner is not None:
            fb = fboxes_fn()
            if fb is not None:
                cf, hid, risk = path_check(c.xs[k:], c.ys[k:], fb, planner.bounds)
                if 0 < cf <= S.VETO_FRAMES and (c.base_contact is None or frame + cf < c.base_contact - 1):
                    out.append(f"danger {hid} f{cf}")
                elif risk - c.base_risk > DANGER_RISE:
                    out.append(f"danger: risk {c.base_risk:.2f} -> {risk:.2f}")
        return out
