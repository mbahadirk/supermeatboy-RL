"""Geometric safety of imagined trajectories: the planner's death check instead of the world model's learned
death output (PLAN_PLANNER_AGENT.md, "Decisions after review", item 1).

Hazards per frame are the lethal pieces as boxes grown by their measured kill reach (game.level.hazard_boxes,
hazard_reach.json), plus flying shots as boxes (x, y, rad, rad, 0) as realgame.Trainer.on_frame builds them,
plus remembered deaths the geometry did not explain (DeathMemory). Contact is the agents' map rule: the
agent's centre within BODY (realpath.BODY) of a box. Leaving the level is the realgame.js rule (outOfLevel).

A trajectory is positions (members E, candidates n, frames K+1); frame 0 is now. Each step k -> k+1 is tested
as a movement segment (a fall covers about 50 units per frame and would skip thin hazards between frames),
sampled every SUB_STEP units or less, against the hazards at frame k+1. The segment's start is not re-tested
(it was the previous step's end).

Decision rule: contact within the first VETO_FRAMES frames is a hard veto. Later contact or near contact is a
soft risk, with a margin from the world model's measured 90% position error at that horizon (ERR90),
counted once (not on top of the members' spread).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from realpath import BODY
from game.level import hazard_boxes
from wm.features import box_distances

VETO_FRAMES = 16            # contact in frames 1..VETO_FRAMES vetoes the candidate
SUB_STEP = 3.0              # units: at most this far between the samples of one movement segment
OUT_SIDE, OUT_BOTTOM = 10.0, 60.0    # realgame.js: dead this far past the left / right / top edge, or below
# world model's 90% position error (units) by horizon (frames): the soft-risk margin
ERR90 = ((1, 0.43), (4, 2.0), (8, 5.0), (16, 13.9), (32, 38.1), (48, 60.0))
NEAREST = 3                 # boxes per candidate in the clearance summary
WINDOW = 30                 # frames: a remembered death is a hazard this long before and after its wave frame
DEATH_HALF = 8.0            # half size of a remembered death's box (contact adds BODY)

KIND_NONE, KIND_HAZARD, KIND_SHOT, KIND_OUT, KIND_MEMORY = -1, 0, 1, 2, 3
KIND_NAMES = {KIND_NONE: "none", KIND_HAZARD: "hazard", KIND_SHOT: "shot", KIND_OUT: "out of level",
              KIND_MEMORY: "remembered death"}
OUT_ID = "out"


# ---- hazards of one frame -------------------------------------------------------------------------------
@dataclass
class FrameBoxes:
    """The deadly boxes of one frame: (M, 5) rotated boxes (x, y, half w, half h, angle), what each is
    (KIND_HAZARD / KIND_SHOT / KIND_MEMORY) and a stable id (hazard index in the frame's list, shot id, or the
    remembered death's position)."""
    boxes: np.ndarray
    kind: np.ndarray
    ids: list

    def __len__(self):
        return len(self.ids)


def frame_boxes(hz, keep=None, grow=None, sh=(), memory=None, hz_ids=None) -> FrameBoxes:
    """One frame's deadly boxes: lethal pieces hz (realgame.js rows, grown by `grow`, decoration dropped by
    `keep`: game.level.hazard_boxes), flying shots sh [sid, x, y, rad], and remembered deaths `memory` (M, 5)
    (DeathMemory.boxes). hz_ids: names of the hz rows (default "h<index>")."""
    boxes, kept = hazard_boxes(hz or [], keep, grow)
    kind = [KIND_HAZARD] * len(boxes)
    ids = [str(hz_ids[k]) if hz_ids is not None and k < len(hz_ids) else f"h{k}" for k in kept]
    for sid, x, y, rad in sh or ():
        boxes.append((x, y, rad, rad, 0.0))
        kind.append(KIND_SHOT)
        ids.append(f"shot{sid}")
    if memory is not None:
        for x, y, hw, hh, ang in np.asarray(memory, np.float64).reshape(-1, 5):
            boxes.append((x, y, hw, hh, ang))
            kind.append(KIND_MEMORY)
            ids.append(f"death@{x:.0f},{y:.0f}")
    return FrameBoxes(np.asarray(boxes, np.float64).reshape(-1, 5), np.asarray(kind, np.int8), ids)


def record_boxes(rec: dict, keep=None, grow=None, memory=None) -> FrameBoxes:
    """frame_boxes of a recorded world frame (wm_data *_world.json.gz) or a live frame message."""
    return frame_boxes((rec or {}).get("hz", []), keep, grow, (rec or {}).get("sh", []), memory)


# ---- level bounds ---------------------------------------------------------------------------------------
def level_bounds(lv) -> tuple[float, float, float, float]:
    """(left, right, bottom, top) of the tile map (realgame.js levelBox / levelOY)."""
    return float(lv.ox), float(lv.ox + lv.W * 20), float(lv.oy), float(lv.oy + lv.H * 20)


def out_clearance(bounds, x, y) -> np.ndarray:
    """How far inside the alive area a centre is (negative: out of the level, realgame.js outOfLevel)."""
    left, right, bottom, top = bounds
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return np.minimum(np.minimum(x - (left - OUT_SIDE), (right + OUT_SIDE) - x),
                      np.minimum((top + OUT_SIDE) - y, y - (bottom - OUT_BOTTOM)))


def out_of_level(lv, x, y) -> np.ndarray:
    """realgame.js outOfLevel: x < left-10, x > right+10, y > top+10 or y < bottom-60."""
    return out_clearance(level_bounds(lv), x, y) < 0


# ---- one step: movement segments against one frame's boxes ----------------------------------------------
def segment_clearance(x0, y0, x1, y1, fb: FrameBoxes, bounds=None, sub: float = SUB_STEP):
    """Segments (x0, y0) -> (x1, y1) (same shapes, P after flattening) sampled at t in (0, 1] at most `sub`
    units apart. Returns the smallest clearance (distance - BODY) to each box over the segment (P, M) and the
    smallest out-of-level clearance (P,) (+inf without bounds). Negative = contact."""
    x0, y0, x1, y1 = (np.asarray(a, np.float64).ravel() for a in (x0, y0, x1, y1))
    P = len(x0)
    seg = np.hypot(x1 - x0, y1 - y0)
    S = max(1, int(math.ceil(float(seg.max(initial=0.0)) / sub)))
    t = np.arange(1, S + 1, dtype=np.float64) / S
    px = x0[:, None] + (x1 - x0)[:, None] * t
    py = y0[:, None] + (y1 - y0)[:, None] * t
    M = len(fb)
    if M:
        clear = (box_distances(px.ravel(), py.ravel(), fb.boxes) - BODY).reshape(P, S, M).min(1)
    else:
        clear = np.zeros((P, 0))
    oc = out_clearance(bounds, px, py).min(1) if bounds is not None else np.full(P, np.inf)
    return clear, oc


def segment_contact(x0, y0, x1, y1, fb: FrameBoxes, bounds=None):
    """Which segments touch a box or leave the level. Returns per segment the kind (KIND_NONE when safe;
    a box contact is reported before leaving the level) and the index of the box hit (-1 when none / out)."""
    clear, oc = segment_clearance(x0, y0, x1, y1, fb, bounds)
    P = len(oc)
    box = np.full(P, -1, np.int64)
    kind = np.full(P, KIND_NONE, np.int8)
    if clear.shape[1]:
        j = clear.argmin(1)
        hit = clear[np.arange(P), j] < 0
        box[hit] = j[hit]
        kind[hit] = fb.kind[j[hit]]
    out = (kind == KIND_NONE) & (oc < 0)
    kind[out] = KIND_OUT
    return kind, box


def explain_death(lv, frame_record: dict, keep, grow, x0, y0, x1, y1) -> bool:
    """Does the geometry explain a real death: the move from the previous position (x0, y0) to the death
    position (x1, y1) against the hazards and shots of the death frame, or leaving the level."""
    kind, _ = segment_contact([x0], [y0], [x1], [y1], record_boxes(frame_record, keep, grow), level_bounds(lv))
    return bool(kind[0] != KIND_NONE)


# ---- whole trajectories ---------------------------------------------------------------------------------
@dataclass
class Sweep:
    """Swept check of trajectories (E members, n candidates, K steps).

    contact_frame (E, n): first frame 1..K in contact, -1 if none; contact_kind (E, n): KIND_*;
    contact_box (E, n): index into ids of the box hit (-1: none or out of level).
    ids / kinds (U,): every box seen over frames 1..K; clear_min / clear_frame (E, n, U): smallest clearance
    to each box over the trajectory and the frame of it (+inf / -1 for boxes absent then);
    out_min / out_frame (E, n): the same for the level's edge; step_clear (E, n, K): smallest clearance to
    anything at each step (index k-1 = step k-1 -> k)."""
    contact_frame: np.ndarray
    contact_kind: np.ndarray
    contact_box: np.ndarray
    ids: list
    kinds: np.ndarray
    clear_min: np.ndarray
    clear_frame: np.ndarray
    out_min: np.ndarray
    out_frame: np.ndarray
    step_clear: np.ndarray


def sweep(xs, ys, frames, bounds=None) -> Sweep:
    """xs, ys: (E, n, K+1) positions, frame 0 = now. frames: K+1 FrameBoxes (or None for none), frames[k] =
    the hazards at trajectory frame k (frames[0] is not used: the start is a fact). bounds: level_bounds(lv)
    for the out-of-level rule, or None."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    E, n, K1 = xs.shape
    K = K1 - 1
    if len(frames) != K1:
        raise ValueError(f"need {K1} frames of boxes (frame 0 .. {K}), got {len(frames)}")
    empty = FrameBoxes(np.zeros((0, 5)), np.zeros(0, np.int8), [])
    frames = [fb if fb is not None else empty for fb in frames]
    col, ids, kinds = {}, [], []
    for fb in frames[1:]:
        for i, k in zip(fb.ids, fb.kind):
            if i not in col:
                col[i] = len(ids)
                ids.append(i)
                kinds.append(int(k))
    U, P = len(ids), E * n
    clear_min = np.full((P, U), np.inf)
    clear_frame = np.full((P, U), -1, np.int64)
    out_min = np.full(P, np.inf)
    out_frame = np.full(P, -1, np.int64)
    step_clear = np.full((P, K), np.inf)
    contact_frame = np.full(P, -1, np.int64)
    contact_kind = np.full(P, KIND_NONE, np.int8)
    contact_box = np.full(P, -1, np.int64)
    X, Y = xs.reshape(P, K1), ys.reshape(P, K1)
    for k in range(1, K1):
        fb = frames[k]
        clear, oc = segment_clearance(X[:, k - 1], Y[:, k - 1], X[:, k], Y[:, k], fb, bounds)
        if len(fb):
            idx = np.fromiter((col[i] for i in fb.ids), np.int64, len(fb))
            better = clear < clear_min[:, idx]
            clear_min[:, idx] = np.where(better, clear, clear_min[:, idx])
            clear_frame[:, idx] = np.where(better, k, clear_frame[:, idx])
            j = clear.argmin(1)
            cbest = clear[np.arange(P), j]
        else:
            j = np.zeros(P, np.int64)
            cbest = np.full(P, np.inf)
        better = oc < out_min
        out_min[better], out_frame[better] = oc[better], k
        step_clear[:, k - 1] = np.minimum(cbest, oc)
        new = contact_frame < 0
        hit_box = new & (cbest < 0)
        hit_out = new & ~hit_box & (oc < 0)
        contact_frame[hit_box | hit_out] = k
        if hit_box.any():
            contact_box[hit_box] = idx[j[hit_box]]
            contact_kind[hit_box] = fb.kind[j[hit_box]]
        contact_kind[hit_out] = KIND_OUT
    sh2, sh3 = (E, n), (E, n, -1)
    return Sweep(contact_frame.reshape(sh2), contact_kind.reshape(sh2), contact_box.reshape(sh2), ids,
                 np.asarray(kinds, np.int8), clear_min.reshape(sh3), clear_frame.reshape(sh3),
                 out_min.reshape(sh2), out_frame.reshape(sh2), step_clear.reshape(E, n, K))


# ---- decision rule --------------------------------------------------------------------------------------
def margin(k) -> np.ndarray:
    """Soft-risk margin (units) at horizon k frames: the world model's 90% position error, interpolated
    linearly in ERR90 and held at its ends."""
    f, e = zip(*ERR90)
    return np.interp(np.asarray(k, np.float64), f, e)


@dataclass
class Safety:
    """Per candidate (n,): veto (contact within VETO_FRAMES frames in any member), veto_frame (-1: none),
    veto_kind / veto_id (what was hit first), soft_risk in [0, 1] (max over members and frames after
    VETO_FRAMES) and soft_frame (its frame, -1 when 0), nearest: per candidate the NEAREST closest boxes as
    dicts {id, kind, clearance, frame} (min over members; the level edge counts as id "out"). sweep: the
    full Sweep."""
    veto: np.ndarray
    veto_frame: np.ndarray
    veto_kind: np.ndarray
    veto_id: list
    soft_risk: np.ndarray
    soft_frame: np.ndarray
    nearest: list
    sweep: Sweep = field(repr=False)


def assess(sw: Sweep) -> Safety:
    """The horizon-dependent decision rule on a Sweep."""
    E, n, K = sw.step_clear.shape
    cf = sw.contact_frame
    early = (cf >= 1) & (cf <= VETO_FRAMES)
    veto = early.any(0)
    vf = np.where(early, cf, np.iinfo(np.int64).max)
    first_m = vf.argmin(0)                               # member that hits first
    veto_frame = np.where(veto, vf[first_m, np.arange(n)], -1)
    veto_kind = np.where(veto, sw.contact_kind[first_m, np.arange(n)], KIND_NONE).astype(np.int8)
    veto_id = []
    for c in range(n):
        b = sw.contact_box[first_m[c], c]
        veto_id.append(None if not veto[c] else (OUT_ID if veto_kind[c] == KIND_OUT else sw.ids[b]))
    soft_risk = np.zeros(n)
    soft_frame = np.full(n, -1, np.int64)
    if K > VETO_FRAMES:
        ks = np.arange(VETO_FRAMES + 1, K + 1)
        c = sw.step_clear[:, :, VETO_FRAMES:]            # (E, n, K - VETO_FRAMES)
        with np.errstate(invalid="ignore"):
            risk = np.where(c < 0, 1.0, np.clip(1.0 - c / margin(ks), 0.0, 1.0))
        risk = np.nan_to_num(risk, nan=0.0)
        r = risk.max(0)                                  # (n, steps)
        j = r.argmax(1)
        soft_risk = r[np.arange(n), j]
        soft_frame = np.where(soft_risk > 0, ks[j], -1)
    # clearance summary: per box, min over members (with the frame of that member's minimum)
    allc = np.concatenate([sw.clear_min, sw.out_min[:, :, None]], 2)        # (E, n, U+1)
    allf = np.concatenate([sw.clear_frame, sw.out_frame[:, :, None]], 2)
    names = list(sw.ids) + [OUT_ID]
    kinds = list(sw.kinds) + [KIND_OUT]
    m = allc.argmin(0)                                                       # (n, U+1)
    cmin = np.take_along_axis(allc, m[None], 0)[0]
    fmin = np.take_along_axis(allf, m[None], 0)[0]
    nearest = []
    for c in range(n):
        order = [b for b in np.argsort(cmin[c], kind="stable") if np.isfinite(cmin[c, b])][:NEAREST]
        nearest.append([{"id": names[b], "kind": KIND_NAMES[int(kinds[b])], "clearance": float(cmin[c, b]),
                         "frame": int(fmin[c, b])} for b in order])
    return Safety(veto, veto_frame, veto_kind, veto_id, soft_risk, soft_frame, nearest, sw)


def check(xs, ys, frames, bounds=None) -> Safety:
    """sweep + assess: the planner's safety verdict for (E, n, K+1) trajectories."""
    return assess(sweep(xs, ys, frames, bounds))


# ---- deaths the geometry did not predict ----------------------------------------------------------------
class DeathMemory:
    """Real deaths the geometry did not explain, as (level, wave frame, x, y). Waves repeat exactly (time-step
    lock), so each becomes a hazard box (half size DEATH_HALF, centred on the death point) for wave frames
    within WINDOW of the recorded one."""

    def __init__(self, entries=()):
        self.entries: list[tuple[str, int, float, float]] = []
        self._by_level: dict[str, np.ndarray] = {}
        for e in entries:
            self.add(*e)

    def __len__(self):
        return len(self.entries)

    def add(self, level: str, frame: int, x: float, y: float):
        self.entries.append((str(level), int(frame), float(x), float(y)))
        self._by_level.pop(str(level), None)

    def boxes(self, level: str, frame: int) -> np.ndarray:
        """(M, 5) boxes of the remembered deaths of this level active at this wave frame."""
        a = self._by_level.get(level)
        if a is None:
            a = np.array([(f, x, y) for lvl, f, x, y in self.entries if lvl == level], np.float64).reshape(-1, 3)
            self._by_level[level] = a
        on = a[np.abs(a[:, 0] - frame) <= WINDOW]
        out = np.zeros((len(on), 5))
        out[:, 0], out[:, 1] = on[:, 1], on[:, 2]
        out[:, 2] = out[:, 3] = DEATH_HALF
        return out

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"about": "real deaths the geometric safety check did not explain (plan/safety.py)",
                       "window": WINDOW, "half": DEATH_HALF,
                       "deaths": [{"level": l, "frame": fr, "x": x, "y": y} for l, fr, x, y in self.entries]},
                      f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "DeathMemory":
        """A saved memory; an empty one when there is no file yet (a malformed file raises)."""
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls((d["level"], d["frame"], d["x"], d["y"]) for d in raw["deaths"])
