"""Geometric safety of imagined trajectories: the planner's death check instead of the world model's learned
death output (PLAN_PLANNER_AGENT.md, "Decisions after review", item 1).

Hazards per frame are the lethal pieces' kill shapes as the game's narrow phase tests them, as boxes
(game.level.hazard_boxes: realgame.js killShapes, which move inside a piece's animation), plus flying shots as boxes (x, y, rad, rad, 0) as realgame.Trainer.on_frame builds them,
plus remembered deaths the geometry did not explain (DeathMemory). Contact is the agents' map rule: the
agent's centre within BODY (realpath.BODY) of a box. Leaving the level is the realgame.js rule (outOfLevel):
outside the level's camera-bounds kill box (Level.kill_box), or without one past the map edges.

A trajectory is positions (members E, candidates n, frames K+1); frame 0 is now. Each step k -> k+1 is tested
as a movement segment (a fall covers about 50 units per frame and would skip thin hazards between frames),
sampled every SUB_STEP units or less, against the hazards at frame k+1. The segment's start is not re-tested
(it was the previous step's end). A step through a game portal (jumps: game.level.Jumps, the rollout's
teleports or a recorded path's one-frame jumps) is never tested along the straight line between its ends: its
own motion up to where it entered the trigger and from the exit to its end (a recorded jump: its end only).

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
from game.level import OUT_BOTTOM, OUT_SIDE, alive_box, box_ids, hazard_boxes   # noqa: F401 (OUT_*: fallback rule)
from game.level import (SHAPE_BOX, SHAPE_CIRCLE, SHAPE_GAMEBOX, hazard_shapes, kill_clearance,   # noqa: F401
                        kill_clearance_paired, kill_reach)
from game.level import Jumps, teleport_jumps   # noqa: F401 (teleport_jumps: re-exported for the callers)
from wm.features import box_distances

VETO_FRAMES = 16            # contact in frames 1..VETO_FRAMES vetoes the candidate
SUB_STEP = 3.0              # units: at most this far between the samples of one movement segment
# world model's 90% position error (units) by horizon (frames): the soft-risk margin
ERR90 = ((1, 0.43), (4, 2.0), (8, 5.0), (16, 13.9), (32, 38.1), (48, 60.0))
# The margin stops growing after this many frames: a plan is corrected long before its far frames are played (a
# decision every 8 frames, made 8 ahead), so the error that matters there is the one that builds up between two
# corrections, not the open-loop error at +32 / +48 (38 and 60 units: with those every pass within 2-3 tiles of
# a hazard was "risky" and waiting, which costs only its frames, always won: 2-1, 2026-11-13).
MARGIN_HORIZON = 16
NEAREST = 3                 # boxes per candidate in the clearance summary
WINDOW = 30                 # frames: a remembered death is a hazard this long before and after its wave frame
STATIC_NEAR = 16.0          # units: remembered deaths this close at wave frames more than WINDOW apart are one
                            # place that always kills (DeathMemory)
DEATH_HALF = 8.0            # half size of a remembered death's box (contact adds BODY)
# alive samples (wave frame, x, y) of real agents kept per level for the run (DeathMemory.observe_alive), so a
# death added later can count the visits made before it: about 73 minutes of one agent at 60 frames a second,
# 3 MB as float32; past it the oldest chunk is dropped
TRAIL_KEEP = 1 << 18
TRAIL_CHUNK = 4096          # samples gathered in a list before they become one float32 chunk of the trail

KIND_NONE, KIND_HAZARD, KIND_SHOT, KIND_OUT, KIND_MEMORY, KIND_FLOOR = -1, 0, 1, 2, 3, 4
KIND_NAMES = {KIND_NONE: "none", KIND_HAZARD: "hazard", KIND_SHOT: "shot", KIND_OUT: "out of level",
              KIND_MEMORY: "remembered death", KIND_FLOOR: "out of level (map bottom)"}
OUT_ID = "out"
FLOOR_ID = "map bottom"


def contact_id(kind, ids, box):
    """The id of what a contact hit: OUT_ID / FLOOR_ID for leaving the level, else the box's id."""
    if kind == KIND_OUT:
        return OUT_ID
    if kind == KIND_FLOOR:
        return FLOOR_ID
    return ids[box]


# ---- hazards of one frame -------------------------------------------------------------------------------
@dataclass
class FrameBoxes:
    """The deadly shapes of one frame: (M, 5) geometry (x, y, half w, half h, angle), what each is (KIND_HAZARD /
    KIND_SHOT / KIND_MEMORY), a stable id (hazard index in the frame's list, shot id, or the remembered death's
    position), its shape type (game.level SHAPE_*: how the game tests it; None: all SHAPE_BOX, the contact
    convention) and grow (M,), a future's uncertainty radius (None: 0). game.level.kill_clearance is the rule."""
    boxes: np.ndarray
    kind: np.ndarray
    ids: list
    shape: np.ndarray | None = None
    grow: np.ndarray | None = None

    def __len__(self):
        return len(self.ids)

    def shapes(self) -> np.ndarray:
        return np.zeros(len(self.ids), np.int8) if self.shape is None else np.asarray(self.shape, np.int8)

    def grows(self) -> np.ndarray:
        return np.zeros(len(self.ids)) if self.grow is None else np.asarray(self.grow, np.float64)

    def take(self, idx) -> "FrameBoxes":
        idx = np.asarray(idx, np.int64)
        return FrameBoxes(self.boxes[idx], self.kind[idx], [self.ids[i] for i in idx], self.shapes()[idx],
                          self.grows()[idx])

    def clearance(self, px, py) -> np.ndarray:
        """(n points, M): game.level.kill_clearance of character centres against these shapes."""
        return kill_clearance(px, py, self.boxes, self.shape, self.grow)

    def reach(self) -> np.ndarray:
        return kill_reach(self.boxes, self.shape, self.grow)


def frame_boxes(hz, keep=None, sh=(), memory=None, hz_ids=None) -> FrameBoxes:
    """One frame's deadly boxes: lethal pieces hz (realgame.js rows: their kill shapes, game.level.hazard_boxes;
    keep only for rows without shapes), flying shots sh [sid, x, y, rad], and remembered deaths `memory` (M, 5)
    (DeathMemory.boxes). hz_ids: names of the hz rows (default "h<index>"; game.level.box_ids)."""
    boxes, kept, shape = hazard_shapes(hz or [], keep)
    kind = [KIND_HAZARD] * len(boxes)
    ids = box_ids(kept, hz_ids)
    for sid, x, y, rad in sh or ():                  # flying shots: circles (FUN_004cd350)
        boxes.append((x, y, rad, rad, 0.0))
        kind.append(KIND_SHOT)
        ids.append(f"shot{sid}")
        shape.append(SHAPE_CIRCLE)
    if memory is not None:
        for x, y, hw, hh, ang in np.asarray(memory, np.float64).reshape(-1, 5):
            boxes.append((x, y, hw, hh, ang))
            kind.append(KIND_MEMORY)
            ids.append(f"death@{x:.0f},{y:.0f}")
            shape.append(SHAPE_BOX)
    return FrameBoxes(np.asarray(boxes, np.float64).reshape(-1, 5), np.asarray(kind, np.int8), ids,
                      np.asarray(shape, np.int8))


def record_boxes(rec: dict, keep=None, memory=None) -> FrameBoxes:
    """frame_boxes of a recorded world frame (wm_data *_world.json.gz) or a live frame message."""
    return frame_boxes((rec or {}).get("hz", []), keep, (rec or {}).get("sh", []), memory)


# ---- level bounds ---------------------------------------------------------------------------------------
def level_bounds(lv) -> tuple[float, float, float, float]:
    """(left, right, bottom, top) lines a centre must stay within (realgame.js outOfLevel): the level's
    kill box (lv.kill_box: the camera bounds widened by the view x 0.15) when it has one, else the map edges
    widened by OUT_SIDE (left / right / top) and OUT_BOTTOM (bottom) (game.level.alive_box)."""
    return alive_box(lv.ox, lv.oy, lv.W, lv.H, getattr(lv, "kill_box", None))


def out_clearance(bounds, x, y) -> np.ndarray:
    """How far inside the alive area a centre is (negative: out of the level, realgame.js outOfLevel).
    bounds: level_bounds(lv)."""
    left, right, bottom, top = bounds
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return np.minimum(np.minimum(x - left, right - x), np.minimum(top - y, y - bottom))


def out_of_level(lv, x, y) -> np.ndarray:
    """realgame.js outOfLevel: outside the kill box, or without one x < left-10, x > right+10, y > top+10 or
    y < bottom-60 of the map."""
    return out_clearance(level_bounds(lv), x, y) < 0


# ---- one step: movement segments against one frame's boxes ----------------------------------------------
def segment_clearance(x0, y0, x1, y1, fb: FrameBoxes, bounds=None, sub: float = SUB_STEP):
    """Segments (x0, y0) -> (x1, y1) (same shapes, P after flattening) sampled at t in (0, 1] at most `sub`
    units apart. Returns the smallest clearance (game.level.kill_clearance: per shape type, as the game tests it)
    to each shape over the segment (P, M) and the smallest out-of-level clearance (P,) (+inf without bounds).
    Negative = contact."""
    x0, y0, x1, y1 = (np.asarray(a, np.float64).ravel() for a in (x0, y0, x1, y1))
    P = len(x0)
    seg = np.hypot(x1 - x0, y1 - y0)
    S = max(1, int(math.ceil(float(seg.max(initial=0.0)) / sub)))
    t = np.arange(1, S + 1, dtype=np.float64) / S
    px = x0[:, None] + (x1 - x0)[:, None] * t
    py = y0[:, None] + (y1 - y0)[:, None] * t
    M = len(fb)
    if M:
        clear = fb.clearance(px.ravel(), py.ravel()).reshape(P, S, M).min(1)
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


def _cut(x0, y0, x1, y1, jumps: Jumps | None, k: int):
    """A step's segments across a teleport (flat (P,) rows of step k-1 -> k; jumps flat (P, K+1)): the starts
    with each jumped row's start moved to its exit (its end when unknown), and the jumped rows whose motion up
    to the entry is a second segment (rows, start x, y, entry x, y). Without a jump at k: the inputs, None."""
    if jumps is None or not jumps.at[:, k].any():
        return x0, y0, None
    at = jumps.at[:, k]
    ex, ey = jumps.exit[:, k, 0], jumps.exit[:, k, 1]
    sx = np.where(at, np.where(np.isnan(ex), x1, ex), x0)
    sy = np.where(at, np.where(np.isnan(ey), y1, ey), y0)
    r = np.nonzero(at & ~np.isnan(jumps.entry[:, k, 0]))[0]
    pre = None if not len(r) else (r, x0[r], y0[r], jumps.entry[r, k, 0], jumps.entry[r, k, 1])
    return sx, sy, pre


def _flat(jumps: Jumps | None, P: int, K1: int) -> Jumps | None:
    """jumps of (E, n, K+1) trajectories as (P, K+1) rows; None (also without any jump) stays None."""
    if jumps is None or not jumps.any():
        return None
    if jumps.at.size != P * K1:
        raise ValueError(f"jumps of shape {jumps.at.shape} for {P} paths of {K1} columns")
    return Jumps(jumps.at.reshape(P, K1), jumps.frame.reshape(P, K1), jumps.entry.reshape(P, K1, 2),
                 jumps.exit.reshape(P, K1, 2))


def explain_death(lv, frame_record: dict, keep, x0, y0, x1, y1) -> bool:
    """Does the geometry explain a real death: the move from the previous position (x0, y0) to the death
    position (x1, y1) against the hazards and shots of the death frame, or leaving the level."""
    kind, _ = segment_contact([x0], [y0], [x1], [y1], record_boxes(frame_record, keep), level_bounds(lv))
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
    anything at each step (index k-1 = step k-1 -> k). step: frames per column (a step-k world model's
    positions); every frame above (contact_frame, clear_frame, out_frame) is in frames (column x step)."""
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
    step: int = 1

    def frames_of(self) -> np.ndarray:
        """The frame of each step_clear column (1..K x step)."""
        return np.arange(1, self.step_clear.shape[2] + 1) * self.step


def sweep(xs, ys, frames, bounds=None, floor=None, step: int = 1, jumps: Jumps | None = None) -> Sweep:
    """xs, ys: (E, n, K+1) positions, frame 0 = now. frames: K+1 FrameBoxes (or None for none), frames[k] =
    the hazards at trajectory frame k (frames[0] is not used: the start is a fact). bounds: level_bounds(lv)
    for the out-of-level rule, or None. floor: a plan.geometry.Terrain (or None): a step ending on the padding
    below the map (Terrain.on_padding: the world model's false floor) has left the level too, a contact of kind
    KIND_FLOOR (clearance -1 at that step: the same verdict and risk as the kill box). A floor with at(frame)
    (plan.planner's per-step rule, its moving platforms where the column's frame has them) gives each column's
    rule: at(k x step). step: frames per column (frames[k] then holds the boxes of frame k x step; the segment
    test covers the motion between columns). jumps (game.level.Jumps of xs' shape, or None): steps through a
    game portal, tested as the module docstring says."""
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
    J = _flat(jumps, P, K1)
    floor_at = getattr(floor, "at", None)
    for k in range(1, K1):
        fb = frames[k]
        sx, sy, pre = _cut(X[:, k - 1], Y[:, k - 1], X[:, k], Y[:, k], J, k)
        clear, oc = segment_clearance(sx, sy, X[:, k], Y[:, k], fb, bounds)
        if pre is not None:                             # a teleport's motion up to its entry
            r, ax, ay, bx, by = pre
            c2, o2 = segment_clearance(ax, ay, bx, by, fb, bounds)
            clear[r], oc[r] = np.minimum(clear[r], c2), np.minimum(oc[r], o2)
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
        fl = floor_at(k * step) if floor_at is not None else floor
        pad = fl.on_padding(X[:, k], Y[:, k], dy=Y[:, k] - sy) if fl is not None else None
        if pad is not None and pad.any():
            oc = np.where(pad, np.minimum(oc, -1.0), oc)
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
        if pad is not None:
            contact_kind[hit_out & pad] = KIND_FLOOR
    sh2, sh3 = (E, n), (E, n, -1)
    if step != 1:                                       # columns -> frames (-1 stays -1)
        for a in (contact_frame, clear_frame, out_frame):
            a[a > 0] *= step
    return Sweep(contact_frame.reshape(sh2), contact_kind.reshape(sh2), contact_box.reshape(sh2), ids,
                 np.asarray(kinds, np.int8), clear_min.reshape(sh3), clear_frame.reshape(sh3),
                 out_min.reshape(sh2), out_frame.reshape(sh2), step_clear.reshape(E, n, K), int(step))


def first_contact(xs, ys, boxes, first=None, upto=None, step: int = 1, jumps: Jumps | None = None):
    """The same swept segment test against boxes that are touched rather than deadly (the goal, a key): fixed
    boxes (M, 5) in the contact convention (centre within BODY: game.level.Level.goal_boxes / key_boxes).
    xs, ys (E, n, K+1). Per trajectory the first frame k (step k-1 -> k) that touches a box, only frames
    first..upto (first, upto: broadcast to (E, n); None: 1 and K), and the index of the box; -1 when none.
    A box the trajectory starts in contact with (frame 0) is not counted for it: the game said it was not
    touched there. step: frames per column; first, upto and the frames returned are in frames (a frame inside
    a column counts as that column's end frame). jumps: as sweep's (a portal step: never the line across)."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    E, n, K1 = xs.shape
    K = K1 - 1
    frame = np.full((E, n), -1, np.int64)
    which = np.full((E, n), -1, np.int64)
    b = np.asarray(boxes, np.float64).reshape(-1, 5)
    if not len(b) or K < 1:
        return frame, which
    reach = np.hypot(b[:, 2], b[:, 3]) + BODY
    near = ((b[:, 0] >= xs.min() - reach) & (b[:, 0] <= xs.max() + reach) & (b[:, 1] >= ys.min() - reach)
            & (b[:, 1] <= ys.max() + reach))
    if not near.any():
        return frame, which
    idx = np.nonzero(near)[0]
    fb = FrameBoxes(b[idx], np.zeros(len(idx), np.int8), [str(i) for i in idx])   # goal / key: SHAPE_BOX
    P = E * n
    X, Y = xs.reshape(P, K1), ys.reshape(P, K1)
    def lim(v, default):
        if v is None:
            return np.full(P, default, np.int64)
        v = np.broadcast_to(np.asarray(v, np.int64), (E, n)).reshape(P)
        return v if step == 1 else -(-v // step)
    lo, hi = lim(first, 1), np.minimum(lim(upto, K), K)
    # only the paths that come near a box, every step at once
    b = fb.boxes
    r = np.hypot(b[:, 2], b[:, 3]) + BODY
    cand = np.nonzero(((X.min(1)[:, None] <= b[:, 0] + r) & (X.max(1)[:, None] >= b[:, 0] - r)
                       & (Y.min(1)[:, None] <= b[:, 1] + r) & (Y.max(1)[:, None] >= b[:, 1] - r)).any(1)
                      & (lo <= hi))[0]
    if not len(cand):
        return frame, which
    Xc, Yc = X[cand], Y[cand]
    J = _flat(jumps, P, K1)
    Sx, Sy = Xc[:, :-1], Yc[:, :-1]
    pre = []
    if J is not None and J.at[cand].any():
        Sx, Sy = Sx.copy(), Sy.copy()
        for k in np.nonzero(J.at[cand].any(0))[0]:
            sx, sy, pk = _cut(Xc[:, k - 1], Yc[:, k - 1], Xc[:, k], Yc[:, k], J[cand], k)
            Sx[:, k - 1], Sy[:, k - 1] = sx, sy
            if pk is not None:
                pre.append((k, pk))
    clear, _ = segment_clearance(Sx, Sy, Xc[:, 1:], Yc[:, 1:], fb)
    clear = clear.reshape(len(cand), K, len(b))
    for k, (r, ax, ay, bx, by) in pre:                  # a teleport's motion up to its entry
        c2, _ = segment_clearance(ax, ay, bx, by, fb)
        clear[r, k - 1] = np.minimum(clear[r, k - 1], c2)
    start_in = box_distances(Xc[:, 0], Yc[:, 0], b) - BODY < 0              # (paths, M)
    clear = np.where(start_in[:, None, :], np.inf, clear)
    j = clear.argmin(2)                                                      # (paths, K)
    ks = np.arange(1, K + 1)
    hit = ((np.take_along_axis(clear, j[:, :, None], 2)[:, :, 0] < 0) & (ks >= lo[cand][:, None])
           & (ks <= hi[cand][:, None]))
    got = hit.any(1)
    k1 = hit.argmax(1)
    out_f, out_w = frame.reshape(P), which.reshape(P)
    out_f[cand[got]] = (k1[got] + 1) * step
    out_w[cand[got]] = idx[j[np.arange(len(cand)), k1][got]]
    return out_f.reshape(E, n), out_w.reshape(E, n)


# ---- decision rule --------------------------------------------------------------------------------------
def margin(k) -> np.ndarray:
    """Soft-risk margin (units) at horizon k frames: the world model's 90% position error, interpolated
    linearly in ERR90, held at its ends and from MARGIN_HORIZON frames on."""
    f, e = zip(*ERR90)
    return np.interp(np.minimum(np.asarray(k, np.float64), MARGIN_HORIZON), f, e)


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
        veto_id.append(None if not veto[c] else contact_id(veto_kind[c], sw.ids, b))
    soft_risk = np.zeros(n)
    soft_frame = np.full(n, -1, np.int64)
    fk = sw.frames_of()
    late = fk > VETO_FRAMES
    if late.any():
        ks = fk[late]
        c = sw.step_clear[:, :, late]                    # (E, n, columns after VETO_FRAMES)
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


def check(xs, ys, frames, bounds=None, floor=None, step: int = 1, jumps: Jumps | None = None) -> Safety:
    """sweep + assess: the planner's safety verdict for (E, n, K+1) trajectories (step: frames per column;
    jumps: their teleports, sweep's)."""
    return assess(sweep(xs, ys, frames, bounds, floor, step, jumps))


# ---- deaths the geometry did not predict ----------------------------------------------------------------
class DeathMemory:
    """Real deaths the geometry did not explain, as (level, wave frame, x, y). Waves repeat exactly (time-step
    lock), so each becomes a hazard box (half size DEATH_HALF, centred on the death point) for wave frames
    within WINDOW of the recorded one.

    The window stays short on purpose: repeated deaths at one place at other wave frames came from kill shapes
    moving inside a piece's animation (1-14's saws), which the hazard rows now carry for every frame; the wave
    reset restarts those animations, so a remembered death is a fact about its own wave frame only. A longer
    window would block the place at times it is safe.

    The exception: deaths within STATIC_NEAR units of each other at wave frames more than WINDOW apart. The agent
    came back when the window had passed and died there again, so the place kills at any time (2-1: a pile whose
    kill shape reaches further than the one read from the game; the agent died there once per wave, 48 frames
    later each time, and no agent was ever there alive). Those boxes are on at every frame (one per place: its
    first death's box).

    Unless a real agent survived there: a place stops being static once an agent was observed alive in contact
    with its box (the safety check's contact: game.level.kill_clearance of a SHAPE_BOX < 0, the centre within BODY
    of the DEATH_HALF box) at a wave frame more than WINDOW from every death at the place (its deaths and every
    death within STATIC_NEAR of its first one). It then falls back to its deaths' own windowed boxes. Measured on
    3-2 (runs/online_check03): deaths at (25.19, 216.01) frame 772 and (18.68, 217.95) frame 680 made a static
    box on the take-off of the only way on (a run-jump over the saw row h28-h30); the stalls there grew from
    60-100 frames to 170-345, and 44 of 110 rejected alternatives in one stall hit it. Yet agent 0 was alive in
    contact with that box for 132 frames of wave 1 (580-771, before the death at 772), 17 frames in each of waves
    2 and 3 (246-311) and 23 in wave 8 (456-679).

    Visits are kept per death as the wave frames an agent was alive in contact with its box (`visits`, saved with
    it), not as a refuted flag: whether a frame is outside WINDOW of every death at the place changes when a
    death is added. They come from observe_alive, every real alive position: each is tested against the level's
    remembered deaths (one vectorised contact test) and also kept in a bounded trail per level for the run
    (TRAIL_KEEP samples), which add() replays for the new death. So the visits before a death count too (3-2's
    wave 1 above, made before any death was remembered there). A previous run's positions are not kept: only
    the visits saved with its deaths.

    Levels in `transient` (agent.py: the held-out levels, which with --online practise after their first life)
    keep their deaths in memory for the run, but save() never writes them."""

    def __init__(self, entries=()):
        self.entries: list[tuple[str, int, float, float]] = []
        self.visits: list[set[int]] = []                # per entry: wave frames an agent was alive in its box
        self._by_level: dict[str, tuple] = {}           # level -> (rows f x y static, live static places)
        self._pts: dict[str, tuple] = {}                # level -> (entry indices, their boxes)
        self._trail: dict[str, list] = {}               # level -> [float32 (n, 3) chunks, pending samples]
        self.transient: set[str] = set()
        self.dirty = False                              # visits added since the last save()
        self.version = 0                                # changes whenever boxes() can: add() and a refuted static
                                                        # place (plan.planner.future_inputs keeps boxes by it)
        for e in entries:
            self.add(*e)

    def __len__(self):
        return len(self.entries)

    def add(self, level: str, frame: int, x: float, y: float, visits=()):
        """A remembered death; its visits: the given ones (a saved file) and the level's trail in contact."""
        level = str(level)
        self.entries.append((level, int(frame), float(x), float(y)))
        v = set(int(f) for f in visits)
        t = self.trail(level)
        if len(t):
            box = np.array([[float(x), float(y), DEATH_HALF, DEATH_HALF, 0.0]])
            v.update(int(f) for f in np.unique(t[kill_clearance(t[:, 1], t[:, 2], box)[:, 0] < 0, 0]))
        self.visits.append(v)
        self._by_level.pop(level, None)
        self._pts.pop(level, None)
        self.version += 1

    def trail(self, level: str) -> np.ndarray:
        """(n, 3) the level's kept alive samples (wave frame, x, y), oldest first."""
        t = self._trail.get(level)
        if t is None:
            return np.zeros((0, 3), np.float32)
        chunks, pending = t
        if pending:
            chunks.append(np.asarray(pending, np.float32).reshape(-1, 3))
            pending.clear()
        return np.concatenate(chunks) if chunks else np.zeros((0, 3), np.float32)

    def observe_alive(self, level: str, frame: int, xs, ys) -> bool:
        """Real agents alive at (xs, ys) at this wave frame of this level: kept in the level's trail and counted as
        visits of the remembered deaths whose box they touch. True when that refuted a static place (its box is
        now windowed: worth saving)."""
        xs, ys = np.atleast_1d(np.asarray(xs, np.float64)), np.atleast_1d(np.asarray(ys, np.float64))
        if not len(xs):
            return False
        frame = int(frame)
        chunks, pending = self._trail.setdefault(level, ([], []))
        pending.extend((frame, x, y) for x, y in zip(xs.tolist(), ys.tolist()))
        if len(pending) >= TRAIL_CHUNK:
            chunks.append(np.asarray(pending, np.float32).reshape(-1, 3))
            pending.clear()
            while len(chunks) > 1 and sum(len(c) for c in chunks) > TRAIL_KEEP:
                chunks.pop(0)
        pts = self._pts.get(level)
        if pts is None:
            idx = np.array([i for i, e in enumerate(self.entries) if e[0] == level], np.int64)
            b = np.zeros((len(idx), 5))
            if len(idx):
                b[:, :2] = [self.entries[i][2:] for i in idx]
                b[:, 2:4] = DEATH_HALF
            pts = self._pts[level] = (idx, b)
        idx, b = pts
        if not len(idx):
            return False
        hit = idx[(kill_clearance(xs, ys, b) < 0).any(0)]
        if not len(hit):
            return False
        live = self._table(level)[1]                    # before these visits: static places' first deaths
        refuted = False
        for i in hit.tolist():
            if frame in self.visits[i]:
                continue
            self.visits[i].add(frame)
            self.dirty = True
            g = live.get(i)                             # a static place's first death: its deaths' frames
            if g is not None and (np.abs(g - frame) > WINDOW).all():
                refuted = True
        if refuted:
            self._by_level.pop(level, None)
            self.version += 1
        return refuted

    def _table(self, level: str) -> tuple:
        """The level's (rows (n, 4): wave frame, x, y, static; {first death's entry index: its place's death
        frames} of the static places not refuted)."""
        t = self._by_level.get(level)
        if t is not None:
            return t
        idx = [i for i, e in enumerate(self.entries) if e[0] == level]
        a = np.array([self.entries[i][1:] for i in idx], np.float64).reshape(-1, 3)
        n = len(a)
        static = np.zeros(n, bool)
        live = {}
        if n > 1:
            near = np.hypot(a[:, None, 1] - a[None, :, 1], a[:, None, 2] - a[None, :, 2]) <= STATIC_NEAR
            always = (near & (np.abs(a[:, None, 0] - a[None, :, 0]) > WINDOW)).any(1)
            place = np.arange(n)                        # a static death joins the place of the first static
            for i in np.nonzero(always)[0]:             # death near it (none: it is a place's first death)
                j = np.nonzero(near[i, :i] & always[:i])[0]
                if len(j):
                    place[i] = place[j[0]]
            keep = np.ones(n, bool)
            for r in np.nonzero(always & (place == np.arange(n)))[0]:
                members = always & (place == r)
                g = a[members | near[r], 0]             # every death at the place
                v = np.fromiter(self.visits[idx[r]], np.float64)
                if len(v) and (np.abs(v[:, None] - g[None, :]) > WINDOW).all(1).any():
                    continue                            # refuted: its deaths keep their own windows
                static[r] = True
                keep &= ~members
                keep[r] = True
                live[idx[r]] = g
            a, static = a[keep], static[keep]
        t = self._by_level[level] = (np.column_stack([a, static]), live)
        return t

    def boxes(self, level: str, frame: int) -> np.ndarray:
        """(M, 5) boxes of the remembered deaths of this level active at this wave frame."""
        a = self._table(level)[0]
        on = a[(a[:, 3] > 0) | (np.abs(a[:, 0] - frame) <= WINDOW)]
        out = np.zeros((len(on), 5))
        out[:, 0], out[:, 1] = on[:, 1], on[:, 2]
        out[:, 2] = out[:, 3] = DEATH_HALF
        return out

    def save(self, path: str):
        """Every non-transient death with its visits as wave-frame ranges ("alive": [[first, last], ...])."""
        tmp = path + ".tmp"
        deaths = []
        for (lv, fr, x, y), v in zip(self.entries, self.visits):
            if lv in self.transient:
                continue
            d = {"level": lv, "frame": fr, "x": x, "y": y}
            if v:
                d["alive"] = frame_ranges(v)
            deaths.append(d)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"about": "real deaths the geometric safety check did not explain (plan/safety.py); alive: "
                                "wave frames a real agent was alive in contact with the death's box",
                       "window": WINDOW, "half": DEATH_HALF, "deaths": deaths}, f, indent=1)
        os.replace(tmp, path)
        self.dirty = False

    @classmethod
    def load(cls, path: str) -> "DeathMemory":
        """A saved memory; an empty one when there is no file yet (a malformed file raises). Files from before
        the visits ("alive") load without them."""
        m = cls()
        if not os.path.exists(path):
            return m
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for d in raw["deaths"]:
            m.add(d["level"], d["frame"], d["x"], d["y"],
                  [f for a, b in d.get("alive", ()) for f in range(int(a), int(b) + 1)])
        return m


def frame_ranges(frames) -> list:
    """Sorted wave frames as [[first, last], ...] runs of consecutive frames."""
    out = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out
