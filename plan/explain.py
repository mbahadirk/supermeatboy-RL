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
    alternatives: the top rejected plans {moves, total, reason}.
    predictions: {1: [x, y], 4: ..., 8: ..., 16: ...} for the surprise check.
    death_model: the learned death output (probability of dying within the plan, max over members): a second
    opinion only, never used for the decision.
    timing: ms per stage; counts: candidates and rolled-out steps actually evaluated per stage, and whether the
    stage ran to the end (never pretended). segments: [[move, from frame, to frame]] of the H frames.
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
        return (f"{'[prev] ' if self.from_prev else ''}{self.name}: progress {sc['progress']:+.2f} (V {v0} -> {v1}, "
                f"leg {self.leg}), risk {sc['risk']:.2f}, unc x{sc['uncertainty']:.2f}, total {sc['total']:+.2f}; "
                f"{near_s}; {land_s}{wj}{veto}{alt_s}")


class SurpriseTracker:
    """Compares each real state with what earlier plans predicted for that frame (+1, +4, +8, +16) and emits
    surprise events: a position error over SURPRISE_FACTOR x the measured 90% error at that horizon, or a death
    the plan did not predict (no veto and no soft risk)."""

    def __init__(self, factor: float = SURPRISE_FACTOR, err90: dict | None = None):
        self.factor = factor
        self.err90 = dict(err90 or ERR90_AT)
        self.plans: dict[int, list] = {}        # agent -> [(frame, predictions, veto, soft risk, H)]

    def add(self, plan: Plan):
        lst = self.plans.setdefault(int(plan.agent), [])
        lst.append((int(plan.frame), {int(k): v for k, v in plan.predictions.items()}, plan.veto,
                    float(plan.soft_risk), len(plan.actions), plan.name))

    def observe(self, agent: int, frame: int, x: float, y: float, dead: bool = False) -> list[dict]:
        events = []
        lst = self.plans.get(int(agent), [])
        for f0, preds, veto, risk, H, name in lst:
            h = frame - f0
            if h in preds and preds[h] is not None and not dead:
                px, py = preds[h]
                err = math.hypot(x - px, y - py)
                thr = self.factor * self.err90.get(h, max(self.err90.values()))
                if err > thr:
                    events.append({"kind": "position", "agent": int(agent), "frame": int(frame), "plan_frame": f0,
                                   "horizon": h, "predicted": [round(px, 2), round(py, 2)],
                                   "real": [round(float(x), 2), round(float(y), 2)], "error": round(err, 2),
                                   "threshold": round(thr, 2), "plan": name})
        if dead:
            cover = [p for p in lst if 0 < frame - p[0] <= p[4]]
            if cover:
                f0, _, veto, risk, H, name = max(cover, key=lambda p: p[0])     # the latest plan
                if veto is None and risk <= 0.0:
                    events.append({"kind": "unpredicted death", "agent": int(agent), "frame": int(frame),
                                   "plan_frame": f0, "x": round(float(x), 2), "y": round(float(y), 2), "plan": name})
            self.plans[int(agent)] = []
        else:
            keep = max(HORIZONS)
            self.plans[int(agent)] = [p for p in lst if frame - p[0] < keep]
        return events
