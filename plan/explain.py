"""What the planner decided and why (the Plan object), and the check of its predictions against what really
happened (SurpriseTracker).

A Plan is written once per decision by plan.planner.Planner.plan and is JSON-serialisable (to_dict) for the
plan log, the viewer and the in-game overlay. Every number in it comes from the decision's own computation:
nothing is filled in after the fact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

HORIZONS = (1, 4, 8, 16)
# the world model's measured 90% position error (units) at those horizons (PROJECT_STATE.md 7.2)
ERR90_AT = {1: 0.43, 4: 2.0, 8: 5.0, 16: 13.9}
SURPRISE_FACTOR = 1.5          # a surprise: position error over this x ERR90_AT at that horizon


def _plain(v):
    """numpy scalars / arrays / tuples -> plain Python for json.dumps; non-finite floats -> None."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return round(f, 3) if math.isfinite(f) else None
    return v


@dataclass
class Plan:
    """One decision.

    frame, agent; state: the start state (x, y, vx, vy, flags, movement state, jump key held, keys, and the
    geometric support: ground, wall side, supported).
    chosen: names of the moves (e.g. ["run-jump R hold 10", "wall-jump L hold 8"]); pad: frames after the moves,
    their last action held (no jump); from_prev: the chosen plan is the previous one, shifted; actions: H
    ACTIONS indices.
    path: predicted mean (x, y) per frame 0..H over the ensemble members; spread: their position std per frame.
    landings: frames where predicted support starts after being airborne {frame, x, y, on: ground | wall};
    wall_jumps: frames on a wall (not on the ground) where a new jump press follows {frame, x, y, side: the side
    the wall is on, L | R}.
    nearest: the closest hazards {id, kind, clearance, frame} (plan.safety.Safety.nearest; "out" = the level's
    edge); veto: None or {frame, id, kind} (contact within 16 frames, any member); soft_risk / soft_frame.
    score: {progress, progress_credit, risk, uncertainty (factor), spread_end, failure, total}; v_start, v_end
    (mean over the members whose end is not a dead end), dead_end_members (ends with V = +inf), leg;
    end_supported.
    alternatives: the top rejected plans {moves, total, progress, stage, reason}, and from Planner.plan also
    path (predicted [[x, y], ...]: final-stage ones the members' mean over frames 0..H, stage-1 ones member 0
    over their move's frames) and vetoed (bool).
    predictions: {1: [x, y], 4: ..., 8: ..., 16: ...} for the surprise check.
    death_model: the learned death output (probability of dying within the plan, max over members): a second
    opinion only, never used for the decision.
    goal_touch: None, or where the plan touches the goal {frame (the members' median), x, y (mean touch point),
    members (how many touch it safely)}: a success (plan/planner.py, goal contact). key_touches: the route's
    next keys the plan picks up [{frame, key (index in the level's key list), x, y}].
    finish_model: the learned finish head along the plan (max probability over members and frames): a
    cross-check of goal_touch only, never used for the decision.
    timing: ms per stage; counts: candidates and rolled-out steps actually evaluated per stage, and whether the
    stage ran to the end (never pretended). segments: [[move, from frame, to frame]] of the H frames.
    switch: None without a previous plan to keep, else the switch rule's outcome (plan/planner.py) {kept: the
    previous plan (shifted) was kept, gain: the best other candidate's advantage over it (V units, or frames
    earlier at the goal: unit), margin: the margin it had to beat, unit: "V" | "frames", reason: e.g.
    "kept: best other +0.31 < 0.5", "switched: +1.80", "switched: previous plan vetoed (h3 f9)"}.
    """
    frame: int
    agent: int
    state: dict
    chosen: list
    actions: list
    pad: int
    from_prev: bool
    path: list
    spread: list
    landings: list
    wall_jumps: list
    nearest: list
    veto: dict | None
    soft_risk: float
    soft_frame: int
    score: dict
    v_start: float | None
    v_end: float | None
    dead_end_members: int
    leg: int
    end_supported: bool
    alternatives: list
    predictions: dict
    death_model: float
    timing: dict
    counts: dict
    segments: list = field(default_factory=list)      # [[move name or "pad", first frame, end frame], ...]
    first_move_frames: int = 0
    goal_touch: dict | None = None
    key_touches: list = field(default_factory=list)
    finish_model: float | None = None
    switch: dict | None = None

    @property
    def name(self) -> str:
        return " > ".join(self.chosen) + (f" (+{self.pad}f)" if self.pad else "")

    def to_dict(self) -> dict:
        return _plain(asdict(self))

    def summary(self) -> str:
        """One or two lines: chosen moves, progress, nearest hazard clearance, landing, top rejected reason."""
        sc = self.score
        v0 = "?" if self.v_start is None else f"{self.v_start:.1f}"
        v1 = ("inf" if self.v_end is None else f"{self.v_end:.1f}") +             (f" ({self.dead_end_members} members: dead end)" if self.v_end is not None and self.dead_end_members else "")
        near = self.nearest[0] if self.nearest else None
        near_s = f"nearest {near['id']} {near['clearance']:.0f}u at f{near['frame']}" if near else "no hazard near"
        land = self.landings[0] if self.landings else None
        land_s = f"lands ({land['x']:.0f},{land['y']:.0f}) f{land['frame']} on {land['on']}" if land else "no landing"
        wj = f", wall-jump f{self.wall_jumps[0]['frame']} {self.wall_jumps[0]['side']}" if self.wall_jumps else ""
        veto = f" VETO {self.veto['id']} f{self.veto['frame']}" if self.veto else ""
        alt = self.alternatives[0] if self.alternatives else None
        alt_s = f" | rejected '{' > '.join(alt['moves'])}': {alt['reason']}" if alt else ""
        gt = self.goal_touch
        touch = (f"touches the princess at f{gt['frame']} ({gt['members']} members); " if gt else "") + "".join(
            f"takes key {k['key']} at f{k['frame']}; " for k in self.key_touches or [])
        sw = f" [{self.switch['reason']}]" if self.switch else ""
        return (f"{'[prev] ' if self.from_prev else ''}{self.name}{sw}: {touch}progress {sc['progress']:+.2f} (V {v0} -> {v1}, "
                f"leg {self.leg}), risk {sc['risk']:.2f}, unc x{sc['uncertainty']:.2f}, total {sc['total']:+.2f}; "
                f"{near_s}; {land_s}{wj}{veto}{alt_s}")


class SurpriseTracker:
    """Compares each real state with what earlier plans predicted for that frame (+1, +4, +8, +16) and emits
    surprise events: a position error over SURPRISE_FACTOR x the measured 90% error at that horizon, or a death
    the plan did not predict (no veto and no soft risk).

    A prediction at horizon h of a plan made at frame f counts only if the plan was followed that far: the
    actions really played on frames f .. f+h-1 (played()) equal the plan's actions[0:h]. A plan replaced
    before then predicted a future that was never played. The followed predictions' position errors are kept
    per horizon until new_wave() (errors()). An unpredicted death is judged by the latest plan covering the
    frame, followed or not."""

    def __init__(self, factor: float = SURPRISE_FACTOR, err90: dict | None = None):
        self.factor = factor
        self.err90 = dict(err90 or ERR90_AT)
        self.plans: dict[int, list] = {}        # agent -> [(frame, predictions, veto, soft risk, H, name, actions)]
        self.played_at: dict[int, dict] = {}    # agent -> {frame: action index played on that frame}
        self.err: dict[int, list] = {h: [] for h in HORIZONS}

    def add(self, plan: Plan):
        lst = self.plans.setdefault(int(plan.agent), [])
        lst.append((int(plan.frame), {int(k): v for k, v in plan.predictions.items()}, plan.veto,
                    float(plan.soft_risk), len(plan.actions), plan.name, [int(a) for a in plan.actions]))

    def played(self, agent: int, frame: int, action: int):
        """The action index the runner really played for this agent on this frame."""
        self.played_at.setdefault(int(agent), {})[int(frame)] = int(action)

    def followed(self, agent: int, f0: int, actions: list, h: int) -> bool:
        """The plan made at f0 was played unchanged for its first h frames."""
        pl = self.played_at.get(int(agent), {})
        return h <= len(actions) and all(pl.get(f0 + k) == actions[k] for k in range(h))

    def observe(self, agent: int, frame: int, x: float, y: float, dead: bool = False) -> list[dict]:
        events = []
        lst = self.plans.get(int(agent), [])
        for f0, preds, veto, risk, H, name, acts in lst:
            h = frame - f0
            if dead or h not in preds or preds[h] is None or not self.followed(agent, f0, acts, h):
                continue
            px, py = preds[h]
            err = math.hypot(x - px, y - py)
            if h in self.err:
                self.err[h].append(err)
            thr = self.factor * self.err90.get(h, max(self.err90.values()))
            if err > thr:
                events.append({"kind": "position", "agent": int(agent), "frame": int(frame), "plan_frame": f0,
                               "horizon": h, "predicted": [round(px, 2), round(py, 2)],
                               "real": [round(float(x), 2), round(float(y), 2)], "error": round(err, 2),
                               "threshold": round(thr, 2), "plan": name})
        if dead:
            cover = [p for p in lst if 0 < frame - p[0] <= p[4]]
            if cover:
                f0, _, veto, risk, H, name, _ = max(cover, key=lambda p: p[0])     # the latest plan
                if veto is None and risk <= 0.0:
                    events.append({"kind": "unpredicted death", "agent": int(agent), "frame": int(frame),
                                   "plan_frame": f0, "x": round(float(x), 2), "y": round(float(y), 2), "plan": name})
            self.plans[int(agent)] = []
            self.played_at.pop(int(agent), None)
        else:
            keep = max(HORIZONS)
            self.plans[int(agent)] = [p for p in lst if frame - p[0] < keep]
            pl = self.played_at.get(int(agent))
            if pl and len(pl) > keep:
                self.played_at[int(agent)] = {f: a for f, a in pl.items() if f >= frame - keep}
        return events

    def errors(self) -> dict:
        """Position error of the followed predictions since new_wave(), per horizon:
        {"1": [median, 90%, n], ...} (median / 90% None when n is 0)."""
        out = {}
        for h in HORIZONS:
            e = self.err.get(h) or []
            out[str(h)] = ([round(float(np.median(e)), 2), round(float(np.percentile(e, 90)), 2), len(e)]
                           if e else [None, None, 0])
        return out

    def new_wave(self):
        """Frames start again at 0: old predictions and played actions must not match."""
        self.plans.clear()
        self.played_at.clear()
        self.err = {h: [] for h in HORIZONS}
