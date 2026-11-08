"""The planning agent's decision: beam search over named move primitives (plan/moves.py), the world model for
movement, geometric safety for death, the progress field for progress, and a Plan that records what it
predicts and why it was chosen (plan/explain.py). No policy network, no PPO, no learned value.

World model step (wm/model.py step, read from the model file): the default model (runs/wm_step4) predicts 4
frames per step from the 4 per-frame actions, so a decision runs a quarter of the sequential model steps. Every
horizon stays in frames (H = 48, completion 45: 12 steps each, rounded up); a move whose length is not a multiple
of 4 is judged at the next step boundary (Planner.lib_end; its padded frames are the move's last action held);
the safety sweep, goal / key contact and the collision guard work on the step positions (the swept segment test
covers the motion between them; hazard boxes at each step's frame); every frame reported (veto, soft risk,
touches, landings) is a frame. The chosen plan's path is interpolated back to frames (commit tube, overlay,
viewer); the commit tube is measured at whole steps (4, 8, 16: plan.commit.tube_horizons). A 1-frame model
(runs/wm_locked) works as before.

Planning modes (Planner(modes=True): agent.py and plan/offline.py --modes on, the default there; off in the
library and the golden tests). The 5 members are copies of the same physics predictor with roles:
- LONG-SAFE (member 0, blue): when the agent is supported and its last long-safe plan is older than LONG_EVERY =
  32 frames or no longer valid, stages 1 and 2 with member 0 to H_LONG = 96 frames (stage 2 rolled from the start:
  the cost is the steps, not the batch); only candidates with no veto and soft risk <= LONG_RISK = 0.05 qualify;
  its best (shifted by the frames since) joins the final stage. It stays valid while the final stage finds no
  veto and soft risk <= 0.05 on it.
- NORMAL (member 1, green): stages 1 and 2 below, every decision (Planner.member = 1).
- REFLEX (member 2, red): when the commit "danger" trigger fired (plan(danger=True)) or a deadly box (a kill
  shape) is within REFLEX_NEAR = 40 units (2 tiles) of the current plan's path in the next 16 frames: the
  library moves of at most 16 frames with member 2, the least risk (a veto counts 1, else early_risk: frames
  1..16 by the soft rule), then the best estimate. Its plan is re-scored in the final stage for the record but
  chosen only by its override: whenever the chosen plan has a veto or any early risk in its first 16 frames.
- Final stage: all 5 members re-score normal's top S, long-safe's best, the reflex plan and the previous plan as
  below; the switch rule is unchanged (the reflex plan is not a switch candidate). Plan.mode records the mode
  that produced the chosen plan (a kept previous plan keeps its mode); counts["modes"], timing long_safe / reflex.
- EXPLORE (members 3 and 4, purple; Planner.explore, agent.py's --explore agents every EXPLORE_EVERY = 16
  frames): stage 1 only (73 moves padded to 48 frames) with members 3 and 4; among the plans without a veto the
  one whose two members' end positions disagree most, ties by estimate.

Search, per decision (Planner.plan):
1. Stage 1 (one ensemble member): every library move (73, 8 to 24 frames; rolled out as one batch to 24
   frames with its last action held) plus the safety check. Every move is ranked at the same end frame (24:
   the batch's length; goal and key contact over those frames): its estimate is 24 + T there, so a short move
   is followed by its held last action, not credited for stopping early. Safety covers the frames the
   estimate credits, 1..24: the veto and the margin rule on the move's own frames, contact only (soft risk 1,
   no margin) on the held tail, even before frame VETO_FRAMES (a success: 1..its touch, margin rule). The best
   B1 = 8 not vetoed are kept (all vetoed: the ones with the latest veto frame).
2. Stage 2 (one member): from each kept end state (the world model's own end rows), each of 16 second moves
   (a fixed diverse core, moves.CORE_SECOND, plus the best-scoring first moves of this decision not in it),
   padded to the horizon H = 48 (last action held, jump released). Only the continuation is rolled out (the
   prefix is shared); hazards for the model inputs at each candidate's own time (offset rows).
3. Final: the top S = 16 full sequences and the previous plan shifted by the frames played (always included)
   are re-scored from the start with ALL ensemble members and the full safety check (plan.safety.check); the
   best is chosen, unless the switch rule below keeps the previous plan. One window for all: every candidate,
   the shifted previous plan included, ends at H (its unplanned tail is the pad: last action held, jump
   released) for T(end), the completion and goal / key contact, and safety counts the same frames 1..H: the
   veto on 1..VETO_FRAMES, the margin rule on the later planned frames (pads excluded; the previous plan: its
   remaining planned frames), contact only (soft risk 1, no margin) on the never-planned pad tail (a success:
   1..its touch, margin rule). (Judging the kept plan on H - shift frames made standing still cheaper by
   construction: frames + T(end) over a shorter window. No tail check credited a pad tail through a hazard
   with the progress of passing it and no risk; the margin on the tail refused passages near hazards.)

Switch rule (hysteresis, Planner(hysteresis=...), agent.py / plan/offline.py --hysteresis, frames, default
HYSTERESIS_FRAMES = 3 there, from closed loops): the shifted previous plan is kept unless the best other candidate's estimate is lower
by more than `hysteresis` frames (both successes: SUCCESS_MARGIN frames earlier at the goal); a success always
beats a non-success; a vetoed previous plan never blocks a switch; a previous plan whose soft risk is over the
best's by more than RISK_SWITCH is dropped. hysteresis 0 (the default here): no rule, the best total is chosen.
Plan.switch records the outcome {kept, gain (frames), margin, unit "frames", reason}. A kept plan is registered
once: SurpriseTracker.add only for a plan not taken from the previous one (its predictions stay those of its
adoption); FailureMemory.expect keeps the expectation of a move while the same move goes on.
A time budget (120 ms) gives stage 1 a deadline at 30% of it and stage 2 one at the budget minus the final
stage's measured cost (a running average, completion included; at least 45% of the budget); a cut stage is
recorded as cut (candidates and steps actually evaluated in Plan.counts), never passed off as complete, and its
candidates are ranked on the frames they have. The final stage (and its completion) always runs to the end, so
a decision can exceed the budget by what the final stage takes beyond its estimate. budget_ms=None: no
deadlines, every stage runs complete (tests and offline checks whose result must not depend on the machine's
load).

Score of a candidate: the estimated frames to reach the goal (lower is better; total = -estimate):
  estimate = frames used + T(end) + DEATH_FRAMES x (soft risk, or the point-of-no-return rate) + failure frames.
  DEATH_FRAMES = max(T_support(spawn) - T(start), DEAD_END_FRAMES): one death costs the progress it throws away
  (at least a dead end's charge), the same everywhere: soft risk, a completion where every continuation dies, and
  the point-of-no-return memory.
- frames used: stage 1 the common end (24, or the frames rolled before its deadline), stage 2 H (or the frames
  rolled before its deadline), final H, or H + k with the airborne completion (below).
- T(end): plan/progress.py's time field (frames to the goal on realpath's graph, measured frames per tile) at the
  end on the agent's leg after the path's key pickups: T_support where the end is supported (Planner.support:
  ground, or a wall while the downward speed is at most CLING_SPEED, measured inf), else T_rise / T_fall by the
  end's vertical velocity (imagine's per-step vy). T = +inf (a dead end): T(start) + DEAD_END_FRAMES (DEAD_FAR
  when the start is a dead end too). Final stage: the mean over members (each member's own end and leg).
- death: contact within the first 16 frames (plan.safety.VETO_FRAMES) in any member is a hard veto; contact or near
  contact later is a soft risk in [0, 1] (safety's margin from the measured position error), DEATH_FRAMES each;
  on a never-planned tail (pad, stage 1's held last action) only contact counts (1).
- failure memory: FAIL_FRAMES x the times this first move failed from this start cell (FailureMemory).
- point-of-no-return memory: deaths / (tries + 1) x DEATH_FRAMES for this first move from this start cell
  (MoveMemory).
FAIL_FRAMES and DEAD_END_FRAMES are the former V-unit weights (FAIL_COST, DEAD_END) x FX_GROUND frames per tile (on
open ground 1 V unit is about 1 tile). (Until step 4's follow-up a soft risk cost RISK_FRAMES = 50 frames, about
what waiting out the window costs, so plans ran into hazards after frame 16.) Vetoed candidates sort below every other one, by their veto frame (later is
better), then by estimate. Waiting costs its frames; no uncertainty factor (the members' end spread is logged:
Plan.score["spread_end"]); no hand-tuned shaping.
Airborne completion (final stage only, Planner.complete): a candidate lands when a majority of members are
supported at H (each member's own end vy): estimate = H + mean T(end). Otherwise it is completed from the main
member's (Planner.member) end row at H with three continuations, no jump: keep (the plan's last horizontal input
held, with its run flag), neutral, reverse (after a plan ending without a horizontal input: neutral, left,
right), up to COMPLETE_MAX = 45 frames in batches of COMPLETE_CHUNK, stopping when every row has landed or died.
Landed: the first supported frame (the cling rule) or a goal touch (T = 0); died: a deadly contact by the same
safety sweep (hazard boxes, remembered deaths, the level's edge) against the deadly boxes of those frames (the
decision's inputs reach H + COMPLETE_MAX: Planner.future_frames; beyond what is given, the last frame's boxes).
Score: the best continuation that lands before dying: H + k + T_support(landing); none lands: all died, a death
(soft risk 1, i.e. DEATH_FRAMES; with the window's soft risk at most one death) and H + T_air(end); some still airborne after COMPLETE_MAX: H + COMPLETE_MAX + their best T_air there. Deaths in completion are
risk, never veto. Plan.completion records the chosen plan's (continuation, frames, x, y, on, died, path);
counts["final"]["completions"] every candidate's, counts["final"]["safe_alternatives"] the other candidates not
vetoed, with soft risk < SAFE_RISK (the window's soft risk plus the completion's death, as scored), that land
(majority supported, completion landed, or a success).
Known limit (measured, 1-14): the world model's tile input pads outside the map with solid, so imagined bodies do
not fall past the map's bottom edge (they hover about 6 units above it); a fall out of a level whose kill box is
below the map is not seen by the safety check or the completion.

Goal and key contact (every stage, every candidate; the final stage per member): the swept segment test of
plan.safety.first_contact against the goal boxes and the next key's box (game.level.Level.goal_boxes /
key_boxes, reach measured by contact_reach.py), on the candidate's judged frames.
- Goal touched at frame t before any deadly contact of that path: a success, total = SUCCESS_SCORE - t (a tiny
  tie-break, under one frame, by the penalty frames up to t). Every success ranks above every non-success, an earlier
  touch above a later one; frames after t count nothing (veto and risk are judged on frames 1..t only).
  Final stage: a success when a majority of members touch the goal safely, t = the (lower) median of their
  touch frames; otherwise the plan is scored as without contact.
- Next key of the current leg touched at frame t (before any deadly contact): the path goes on with the next
  leg (and may take the key after that), so T(end) is taken on the leg after the pickups. The legs are
  comparable: progress.ProgressField builds T_leg[j] = T to key j + T_leg[j+1] at key j (as V), the frames to
  the goal through the remaining keys. Known limit: the lock tiles a key opens stay
  closed in the imagined tile map (the world model's tile inputs and the support geometry) after the pickup.
- The world model's finish head along the chosen path is logged (Plan.finish_model), never scored.
Plans that touch no goal and no key score exactly as without contact.

Collision guard (every stage, every imagined step: imagine(..., terrain=Planner.guard_terrain(keys)), collide):
the world model's tile input marks a half block without saying which half is solid, so its imagined bodies
went through floors of half blocks (1-14 row 19: 62% of the chosen plans of two runs had the centre in solid).
After each step a body box (half size BODY) overlapping solid tiles by more than COLLIDE_TOL = 1 unit is put back
per axis to contact (collide's docstring). The tolerance from recorded real states (`python -m plan.geometry
--overlap --no-slopes <wm_data>...`, 5 runs, 19 levels, 2.09 M alive rows): rows the guard would move at 0.5 /
1 / 1.5 / 2 units: 342 / 182 / 114 / 63 (0.019 / 0.010 / 0.006 / 0.004%) on every level but 1-7, whose
2,305 rows (at 1 unit, as many at any tolerance) are deep inside full blocks, nearly all from one wave recorded
without the time-step lock (real_20261102-035612, dt up to 64 ms).
Slope cells are left out: real agents are recorded inside the modelled triangles of some (1-13: 5,900 rows,
1-7, 1-6). Moving platforms are not tiles: not in the guard. Plan.model_clipped: the chosen plan's clipped
frames; counts[stage]["clipped"]: candidates clipped. Planner(collide_tol=None): no guard (as before).

Supported rule (plan/geometry.py, checked by `python -m plan.geometry runs/real_20261102-165351/wm_data --shards 3`
on 3 waves each of 1-5 and 1-13, 331,000 rows): of the frames at rest (vy == 0 now and the next frame) the rule
says ground on 99.7% (1-5) and 98.6% (1-13); flag 0x2 is set on 74% / 92% of all rows, so it is not "airborne".
Ground movement states 5, 10, 15, 4: rule ground 97-99.9% (1-5), 88-99% (1-13); airborne states 6, 7, 8:
0-2%; wall-slide state 9: rule wall 86% / 95%.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from mpc.candidates import ACTIONS, act
from plan import safety as S
from plan.explain import Plan, SurpriseTracker, HORIZONS
from plan.geometry import Terrain
from realpath import BODY
from plan.moves import LIBRARY, BY_NAME, CORE_SECOND, MAX_LEN, padded
from plan.progress import FX_GROUND, ProgressField
from wm.buffer import ROW
from wm.features import (BIN_NAMES, FEAT_DIM, N_HAZ, N_STATES, EVENTS, LevelGeom, actions_into, box_distances,
                         featurize, hazard_features, platform_features, prior)
from wm.model import N_CONT
from wm.model import WorldModel
from wm.rollout import World, advance, step_actions, _stacked

H_DEFAULT, B1_DEFAULT, BRANCH_DEFAULT, S_DEFAULT, BUDGET_MS = 48, 8, 16, 16, 120.0
STAGE1_SHARE, STAGE2_SHARE = 0.30, 0.45      # of the budget: stage 1's deadline, stage 2's earliest one
# penalties in frames, from the former V-unit weights by FX_GROUND frames per tile (on open ground 1 V unit ~ 1 tile)
FAIL_COST, DEAD_END = 1.0, 100.0                          # the former weights (V units)
FAIL_FRAMES = FAIL_COST * FX_GROUND          # per counted failure of the first move from the start cell
DEAD_END_FRAMES = DEAD_END * FX_GROUND       # an end with T = +inf: T(start) + this
DEAD_FAR = 1e4                               # frames: T of a dead end when the start is one too
# wall contact is support (for scoring) only up to this downward speed (units / frame). Measured (plan/measure_time.py,
# M2: 13,385 wall jump presses, runs/plan_2026110*): a press on a wall gives a wall jump in 93-100% of the cases in
# every downward-speed bucket up to 10 u/f (8+ u/f: 16 presses only), so no bucket drops below 80%: inf (no cut)
CLING_SPEED = math.inf
# frames: airborne final plans are completed this far at most. 45 (was 90): measured on 30 recorded decisions per
# level (1-14, 1-13, 1-11; no deadlines), the chosen first move was the same with 45 as with 90 in all of them, and
# the completion's cost at the 90th percentile fell 71 -> 41 ms (1-14), 42 -> 39 (1-13), 63 -> 50 (1-11)
COMPLETE_MAX = 45
COMPLETE_CHUNK = 10                          # frames rolled out per completion batch (early stop between them)
NO_PROGRESS = 0.0                            # frames: "no progress" when T(start) - T(end) is not above this
SAFE_RISK = 0.5                              # soft risk (completion's deaths included) under which a landing alternative counts as safe
MIN_EXPECT = 0.5                             # FailureMemory: an expected V drop below this expects nothing
VETO_SCORE = -1e6
VETO_TIE = 1e-6                              # vetoed: by veto frame, then (tiny) by estimate
SUCCESS_SCORE = 1e5                          # a plan touching the goal at frame t: SUCCESS_SCORE - t
TIE = 1e-3                                   # successes at the same frame: tie-break by their penalty frames
SUCCESS_MARGIN = 3                           # frames: a success replaces a successful previous plan only if earlier
# agent.py / plan/offline.py --hysteresis default (frames). After every final candidate is judged at the same end
# frame H, the kept plan's re-score noise (interval-only decisions, closed loops at 3) is |noise| median 11, 75% 21,
# 90% 75 frames (141 pairs; signed median +10: a kept plan pays for the frames it stood). Closed loops
# (offline.closed_loop, 300 frames, no deadlines, modes on; 9 starts on 1-13, 1-8, 1-14, 3 on 1-11, 1-1 from the
# demo) at 3 / 8 / 11 / 16 frames: finished 4 / 5 / 4 / 2, died 3 / 4 / 5 / 4, switch rate 0.58 / 0.50 / 0.43 / 0.44;
# 3 is the smallest value (and the only one tried) with no more deaths and no fewer finishes than 3.
HYSTERESIS_FRAMES = 3.0
RISK_SWITCH = 0.2                            # soft risk: a previous plan riskier than the best by more is dropped
BOX_MARGIN = 70.0                            # units around the candidates' paths: hazards further are not checked
N_ALTERNATIVES = 5
HAZ_FAR = 300.0                              # wm.features.hazard_features clips offsets and distances here
FINAL_MS0 = 60.0                             # first guess of the final stage's cost (then measured, averaged)
COLLIDE_TOL = 1.0                            # units: the collision guard moves imagined bodies overlapping more
# planning modes (Planner(modes=True), module docstring): the ensemble members' roles
MEMBERS = {"long-safe": 0, "normal": 1, "reflex": 2, "explore": (3, 4)}
MODES = ("long-safe", "normal", "reflex", "explore")
H_LONG = 96                                  # frames: the long-safe horizon
LONG_EVERY = 32                              # frames: a long-safe plan older than this is searched again (supported)
LONG_RISK = 0.05                             # soft risk at most this (and no veto) for a long-safe plan
REFLEX_MAX = 16                              # frames: reflex library moves of at most this
REFLEX_WINDOW = 16                           # frames of the current plan's path checked for a near hazard
REFLEX_NEAR = 40.0                           # units beyond a deadly box (2 tiles): a hazard this near is danger
EXPLORE_EVERY = 16                           # frames between an explore agent's decisions
CONTACT_EPS = 0.01                           # units: a body put back to contact is this clear of the solid


# ---- world model rollouts with per-row time offsets -------------------------------------------------------
@torch.inference_mode()
def imagine(model, members: slice, geom: LevelGeom, s0: np.ndarray, actions: np.ndarray, dt_ms: float,
            worlds: list, offsets=None, deadline: float | None = None, keep_rows: bool = False,
            terrain: Terrain | None = None, tol: float = None) -> dict:
    """wm.rollout.rollout_batched for a subset of ensemble members (`members`, a slice) whose rows may start at
    different future frames: row i at step k sees the level's moving parts worlds[offsets[i] + k] (clamped to
    the last). s0 (n,) ROW, actions (n, K). Returns x, y (E, n, K+1) float64; vx, vy (E, n, K+1): the velocity
    per step as position differences (units / frame, y up; step 0: the start rows' vx, vy x dt); last (E, n)
    ROW: the rows after the last step run; alive (E, n, K+1) (the learned
    death head: a second opinion only), finish (E, n, K+1) (the learned finish head's probability per step:
    a cross-check only), rows (E, n, K+1) when keep_rows, and steps (how many steps ran before the deadline;
    positions after that are held at the last one). terrain: the collision guard (collide) after every step
    against these tiles with tolerance tol (default COLLIDE_TOL); clip (E, n, K+1) says which steps it moved
    (None without a terrain).
    A step-k model (model.step = f > 1): actions, dt_ms, offsets and worlds stay per frame; every array is per
    model step instead (ceil(K / f) steps, column j = frame j x f, the last action repeated to fill the last
    step), step j sees worlds[offset + j x f]; vx, vy stay units per frame (the step's difference / f); steps
    counts model steps; "step": f."""
    Wl, Bl = _stacked(model)
    Wl, Bl = [w[members] for w in Wl], [b[members] for b in Bl]
    E = Wl[0].shape[0]
    f = int(getattr(model, "step", 1))
    if f > 1:
        actions = step_actions(actions, f)
    n, K = actions.shape[:2]
    dt_step = dt_ms * f
    s = np.concatenate([s0] * E)
    a_all = np.tile(actions, (E, 1) + (1,) * (actions.ndim - 2))
    off = np.zeros(n, np.int64) if offsets is None else np.asarray(offsets, np.int64)
    off_all = np.tile(off, E)
    groups = [(int(o), np.nonzero(off_all == o)[0]) for o in np.unique(off_all)]
    xs = np.zeros((E, n, K + 1))
    ys = np.zeros((E, n, K + 1))
    alive = np.ones((E, n, K + 1), np.float32)
    fin = np.zeros((E, n, K + 1), np.float32)
    rows = np.zeros((E, n, K + 1), s0.dtype) if keep_rows else None
    clip = np.zeros((E, n, K + 1), bool) if terrain is not None else None
    tol = COLLIDE_TOL if tol is None else tol
    xs[:, :, 0], ys[:, :, 0] = s["x"].reshape(E, n), s["y"].reshape(E, n)
    if rows is not None:
        rows[:, :, 0] = s.reshape(E, n)
    rs = torch.as_tensor(model.res_scale)
    # per-step buffers, allocated once (the same arithmetic as (X - mu) / sd and silu(baddbmm(...)) per step)
    n_in = model.mu.numel()
    X = np.empty((E * n, n_in), np.float32)
    Xf = X if n_in == FEAT_DIM else X[:, :FEAT_DIM]
    Xt = torch.from_numpy(X)
    Z0 = torch.empty((E * n, n_in))
    hid = [torch.empty((E, n, w.shape[2])) for w in Wl]
    al = np.ones(E * n, np.float32)
    done = K
    last = len(worlds) - 1
    for k in range(K):
        if f > 1:
            features_into(Xf, groups, s, a_all[:, k, 0], dt_step, geom, [worlds[min(o + k * f, last)] for o, _ in groups])
            actions_into(X, a_all[:, k])
        else:
            features_into(X, groups, s, a_all[:, k], dt_ms, geom, [worlds[min(o + k, last)] for o, _ in groups])
        torch.sub(Xt, model.mu, out=Z0)
        z = Z0.div_(model.sd).view(E, n, -1)
        for i in range(len(Wl) - 1):
            z = torch.nn.functional.silu(torch.baddbmm(Bl[i], z, Wl[i], out=hid[i]), inplace=True)
        o = torch.baddbmm(Bl[-1], z, Wl[-1], out=hid[-1]).view(E * n, -1)     # model.split's heads, sliced
        move = (o[:, :N_CONT] * rs).numpy() + prior(s, dt_step, f)
        pb = torch.sigmoid(o[:, _BIN]).numpy()
        s2 = advance(s, a_all[:, k], move, pb, o[:, _STATE].argmax(-1).numpy())
        if clip is not None:
            clip[:, :, k + 1] = collide(terrain, s, s2, tol).reshape(E, n)
        s = s2
        al = al * (1.0 - pb[:, EVENTS["death"]])
        xs[:, :, k + 1], ys[:, :, k + 1] = s["x"].reshape(E, n), s["y"].reshape(E, n)
        alive[:, :, k + 1] = al.reshape(E, n)
        fin[:, :, k + 1] = pb[:, EVENTS["finish"]].reshape(E, n)
        if rows is not None:
            rows[:, :, k + 1] = s.reshape(E, n)
        if deadline is not None and time.perf_counter() > deadline and k + 1 < K:
            done = k + 1
            break
    if done < K:
        xs[:, :, done + 1:] = xs[:, :, done:done + 1]
        ys[:, :, done + 1:] = ys[:, :, done:done + 1]
        alive[:, :, done + 1:] = alive[:, :, done:done + 1]
    vx, vy = np.empty_like(xs), np.empty_like(ys)
    vx[:, :, 1:], vy[:, :, 1:] = np.diff(xs, axis=2), np.diff(ys, axis=2)
    if f > 1:
        vx[:, :, 1:] /= f
        vy[:, :, 1:] /= f
    vx[:, :, 0] = np.tile(s0["vx"], E).reshape(E, n) * (dt_ms / 1000.0)
    vy[:, :, 0] = np.tile(s0["vy"], E).reshape(E, n) * (dt_ms / 1000.0)
    return {"x": xs, "y": ys, "vx": vx, "vy": vy, "last": s.reshape(E, n), "alive": alive, "finish": fin,
            "rows": rows, "steps": done, "clip": clip, "step": f}


def collide(ter: Terrain, prev: np.ndarray, s: np.ndarray, tol: float = COLLIDE_TOL) -> np.ndarray:
    """The collision guard: imagined rows s (just stepped from prev) whose body box (half size BODY) would go
    into solid tiles by more than `tol` are put back to contact, per axis as a platformer resolves it: if
    moving only along x (from the previous position) is free, x is kept and y goes back to where the box
    touches (vy = 0); else if moving only along y is free, y is kept and x goes back to contact (vx = 0); else
    x then y, each to contact (a cut axis' velocity 0). "Free" and "touches": Terrain.travel of the box shrunk
    by `tol` (exact on the guard's map of full and half blocks, slopes open: ter.rects_only); a cut move is
    then backed off to CONTACT_EPS of overlap (never behind its start). A body already deeper in a solid
    than that stays where it was. Flags, movement state and the jump key stay what the model predicted (the
    next step sees the corrected position and velocity). Cost per step: one travel pass of every row (x
    alone, y after x, y alone, x after y: a table lookup each), a second only for rows blocked both ways. In
    place; returns which rows were moved."""
    m = len(s)
    x0, y0, x1, y1 = prev["x"], prev["y"], s["x"], s["y"]
    dx, dy = x1 - x0, y1 - y0
    D = np.concatenate([dx, dy, dy, dx])
    q = ter.travel(np.concatenate([x0, x1, x0, x0]), np.concatenate([y0, y0, y0, y1]), D, _AXES4(m),
                   BODY - tol)                           # x alone / y after x / y alone / x after y
    e = np.where(q < 1, np.maximum(q - (tol - CONTACT_EPS) / np.maximum(np.abs(D), 1e-12), 0.0), 1.0)
    q, e = q.reshape(4, m), e.reshape(4, m)              # a cut move backed off to CONTACT_EPS of overlap
    a = q[0] == 1                                        # x free: x kept, y after x
    b = ~a & (q[2] == 1)                                 # else y free: y kept, x after y
    c = ~a & ~b                                          # else x, then y from there
    nx = np.where(a, x1, x0 + np.where(b, e[3], e[0]) * dx)
    ny = np.where(a, y0 + e[1] * dy, y1)
    cut_x = ~a & (c | (q[3] < 1))
    cut_y = (a & (q[1] < 1)) | c
    if c.any():
        k = np.nonzero(c)[0]
        t2 = ter.travel(nx[k], y0[k], dy[k], np.ones(len(k), bool), BODY - tol)
        e2 = np.where(t2 < 1, np.maximum(t2 - (tol - CONTACT_EPS) / np.maximum(np.abs(dy[k]), 1e-12), 0.0), 1.0)
        ny[k] = y0[k] + e2 * dy[k]
        cut_y[k] = t2 < 1
    moved = cut_x | cut_y
    np.copyto(s["x"], nx, casting="unsafe", where=moved)
    np.copyto(s["y"], ny, casting="unsafe", where=moved)
    np.copyto(s["vx"], 0.0, casting="unsafe", where=cut_x)
    np.copyto(s["vy"], 0.0, casting="unsafe", where=cut_y)
    return moved


_AXES = {}


def _AXES4(n: int) -> np.ndarray:
    """travel's axis flags for [x alone, y after x, y alone, x after y] of n rows (cached per n)."""
    a = _AXES.get(n)
    if a is None:
        a = _AXES[n] = np.repeat(np.array([False, True, True, False]), n)
    return a


_BIN = slice(2 * N_CONT, 2 * N_CONT + len(BIN_NAMES))      # wm.model.WorldModel.split's output layout
_STATE = slice(_BIN.stop, _BIN.stop + N_STATES)
HAZ_COLS = slice(FEAT_DIM - 6 - 5 * N_HAZ, FEAT_DIM - 6)     # wm.features.featurize: [..., hazards, platform]
PLAT_COLS = slice(FEAT_DIM - 6, FEAT_DIM)


def features_into(X, groups, s, a, dt_ms, geom, worlds):
    """wm.features.featurize for rows in groups [(offset, row indices)] that see different worlds: one call for
    every row with the first group's world, then only the hazard and platform columns (the parts that depend on
    the world besides wind) per other group. A group whose wind (fans) differs gets a full call."""
    w0 = worlds[0]
    if len(groups) == 1:
        idx = groups[0][1]
        if len(idx) == len(X):                       # every row (idx is arange): straight into the buffer
            featurize(s, a, dt_ms, geom, _model_boxes(w0.boxes, s["x"], s["y"]), w0.pf, w0.pf_prev, w0.dt_prev_s,
                      w0.fans, out=X)
        else:
            X[idx] = featurize(s[idx], a[idx], dt_ms, geom, _model_boxes(w0.boxes, s["x"][idx], s["y"][idx]), w0.pf,
                               w0.pf_prev, w0.dt_prev_s, w0.fans)
        return
    featurize(s, a, dt_ms, geom, np.zeros((0, 5)), w0.pf, w0.pf_prev, w0.dt_prev_s, w0.fans, out=X)
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
    frame + k) and the deadly boxes per frame (H + 1 plan.safety.FrameBoxes, index 0 unused): the game's kill
    shapes with the future's uncertainty radius, flying shots, and remembered deaths (a
    plan.safety.DeathMemory) at their wave frames. Also how many steps came from the recording."""
    worlds, check, recorded = future.worlds(frame, H, dt_ms)
    frames = [None]
    for k, (b, r, ids) in enumerate(check, start=1):
        b = np.asarray(b, np.float64).reshape(-1, 5).copy()
        if len(b):
            b[:, 2:4] += np.asarray(r, np.float64)[:, None]
        ids = list(ids)
        kind = [S.KIND_SHOT if i.startswith("shot") else S.KIND_HAZARD for i in ids]
        if deaths is not None and level_name is not None:
            mem = deaths.boxes(level_name, frame + k)
            if len(mem):
                b = np.concatenate([b, mem])
                kind += [S.KIND_MEMORY] * len(mem)
                ids += [f"death@{x:.0f},{y:.0f}" for x, y in mem[:, :2]]
        frames.append(S.FrameBoxes(b, np.asarray(kind, np.int8), ids))
    return worlds, frames, recorded


def _cols(frames, n: int, f: int) -> list:
    """The FrameBoxes of columns 0..n-1 of a step-f rollout: frames[j x f] (the last given one beyond them)."""
    if f == 1:
        return frames[:n]
    last = len(frames) - 1
    return [frames[min(j * f, last)] for j in range(n)]


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


def _verdict(sw: S.Sweep, upto: np.ndarray, veto_upto=None, planned=None):
    """The decision rule (plan.safety.assess) for candidates judged on their own first `upto` frames (one or
    more members): veto, veto frame, what was hit, soft risk and its frame (all in frames, also for a sweep of
    a step-k model's columns). The veto counts contact in frames 1..min(veto_upto, VETO_FRAMES) (veto_upto:
    default upto); the later frames up to `planned` (n,; default upto) are soft risk by the margin rule
    (contact 1, else 1 - clearance / margin(frame)); the frames after `planned` up to upto (a never-planned
    tail: the pad, or stage 1's held last action) count contact only (1), no margin. Defaults: assess on the
    first upto frames."""
    E, n, K = sw.step_clear.shape
    cf = sw.contact_frame
    upto = np.asarray(upto)
    pl_n = upto if planned is None else np.minimum(np.asarray(planned), upto)
    lim_n = np.minimum(upto if veto_upto is None else np.asarray(veto_upto), S.VETO_FRAMES)
    lim = lim_n[None, :]
    early = (cf >= 1) & (cf <= lim)
    veto = early.any(0)
    vf = np.where(early, cf, np.iinfo(np.int64).max)
    m = vf.argmin(0)
    veto_frame = np.where(veto, vf[m, np.arange(n)], -1)
    veto_kind = np.where(veto, sw.contact_kind[m, np.arange(n)], S.KIND_NONE)
    veto_id = [None if not veto[c] else S.contact_id(veto_kind[c], sw.ids, sw.contact_box[m[c], c]) for c in range(n)]
    soft = np.zeros(n)
    soft_frame = np.full(n, -1, np.int64)
    fk = sw.frames_of()                              # frames of the columns (a step-k model: every k-th)
    late = fk > lim_n.min() if n else fk > S.VETO_FRAMES     # columns after some candidate's veto window
    if late.any():
        ks = fk[late]
        c = sw.step_clear[:, :, late]
        with np.errstate(invalid="ignore", divide="ignore"):
            near = np.where(c < 0, 1.0, np.clip(1.0 - c / S.margin(ks), 0.0, 1.0))
        near = np.nan_to_num(near, nan=0.0)
        after = ks[None, :] > lim_n[:, None]                                            # (n, columns)
        margin_win = after & (ks[None, :] <= pl_n[:, None])                             # planned frames
        tail_win = after & (ks[None, :] > pl_n[:, None]) & (ks[None, :] <= upto[:, None])  # never planned
        risk = np.where(margin_win[None], near, np.where(tail_win[None] & (c < 0), 1.0, 0.0)).max(0)
        j = risk.argmax(1)
        soft = risk[np.arange(n), j]
        soft_frame = np.where(soft > 0, ks[j], -1)
    return veto, veto_frame, veto_kind, veto_id, soft, soft_frame


def early_risk(sw: S.Sweep, upto=None) -> np.ndarray:
    """Risk of frames 1..VETO_FRAMES by plan.safety's soft rule (contact 1, else 1 - clearance / margin(frame),
    clipped to [0, 1]), max over members and those frames (only frames <= upto (n,) when given): REFLEX's
    measure, and its override test on the chosen plan."""
    E, n, K = sw.step_clear.shape
    fk = sw.frames_of()
    m = fk <= S.VETO_FRAMES
    if not m.any():
        return np.zeros(n)
    c, ks = sw.step_clear[:, :, m], fk[m]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(c < 0, 1.0, np.clip(1.0 - c / S.margin(ks), 0.0, 1.0))
    r = np.nan_to_num(r, nan=0.0)
    if upto is not None:
        r = np.where(ks[None, None, :] <= np.asarray(upto)[None, :, None], r, 0.0)
    return r.max(axis=(0, 2))


def hit_reason(kind, hid, frame) -> str:
    if kind == S.KIND_OUT:
        return f"out of level at frame {frame}"
    if kind == S.KIND_FLOOR:
        return f"out of level (map bottom) at frame {frame}"
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
    real drop below half the expected one counts a failure. A death counts nothing here (the point-of-no-return
    memory, MoveMemory, owns deaths: no double charge). Expected drops below MIN_EXPECT expect nothing.
    Penalty: fail_cost (default FAIL_FRAMES) x count, in frames, for plans starting with that move from that
    cell."""

    def __init__(self, fail_cost: float = FAIL_FRAMES):
        self.fail_cost = fail_cost
        self.counts: dict[tuple, int] = {}
        self.pending: dict[int, _Expect] = {}

    def count(self, cell, move: str) -> int:
        return self.counts.get((tuple(int(v) for v in cell), move), 0)

    def penalty(self, cell, move: str) -> float:
        return self.fail_cost * self.count(cell, move)

    def reason(self, cell, move: str) -> str:
        return (f"{move} from ({int(cell[0])},{int(cell[1])}) failed {self.count(cell, move)} times "
                f"(+{self.penalty(cell, move):.0f} frames)")

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
        """The agent's real V at this frame (None / inf when unknown). Returns the failures counted now (a death
        drops the expectation and counts nothing)."""
        p = self.pending.get(agent)
        if p is None or (frame < p.end and not dead):
            return []
        del self.pending[agent]
        if dead:
            return []
        real = None if v_real is None or not math.isfinite(v_real) else p.v_start - float(v_real)
        if real is not None and real >= p.drop / 2:
            return []
        key = (p.cell, p.move)
        self.counts[key] = self.counts.get(key, 0) + 1
        return [{"kind": "failed move", "agent": int(agent), "frame": int(frame), "move": p.move, "cell": list(p.cell),
                 "expected_drop": round(p.drop, 3), "real_drop": None if real is None else round(real, 3),
                 "dead": False, "count": self.counts[key], "reason": self.reason(p.cell, p.move)}]


# ---- point-of-no-return memory ---------------------------------------------------------------------------
class MoveMemory:
    """Deaths charged to the move taken at the point of no return, per level, (start cell, first move name).

    Planner.plan calls decided() with every decision of a planned agent: its frame, start cell, chosen first
    move and how many safe alternatives it had (counts["final"]["safe_alternatives"]: other final candidates
    not vetoed, soft risk < SAFE_RISK counting the completion's deaths too (an option whose every continuation
    dies is not safe), that land or reach the princess). A plan adopted (not kept from the
    previous one) counts a try of its first move from its start cell. On the agent's death (explained or not:
    Planner.observe(dead=True)) the point of no return is the latest decision of the wave with a safe
    alternative; its (cell, first move) gets a death, and a "point of no return" event says where (decision
    frame, cell, move, death frame, cause); none found: the event says so and nothing is charged.
    Penalty, frames, for candidates whose first move matches from the start cell: deaths / (tries + 1) x
    max(T_support(spawn) - T(start), DEAD_END_FRAMES) (the progress the death throws away, at least a dead
    end's charge). Keyed by cell and move only, not by the wave frame (moving hazards' timing is not part of
    the key). Persisted per level in a JSON file (save(): atomic; load(): a malformed file raises). learn False
    (agent.py on held-out evaluation levels): decisions are kept, nothing is charged or counted."""

    VERSION = 1

    def __init__(self, path: str | None = None):
        self.path = path
        self.levels: dict[str, dict[tuple, list]] = {}     # level -> {(c, r, move): [deaths, tries]}
        self.level: str | None = None
        self.wave: dict[int, list] = {}                     # agent -> [(frame, cell, move, safe alternatives)]
        self.dirty = False
        self.learn = True                                   # False (held-out levels): nothing charged or counted

    @classmethod
    def load(cls, path: str) -> "MoveMemory":
        m = cls(path)
        if not os.path.exists(path):
            return m
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict) or d.get("version") != cls.VERSION or not isinstance(d.get("levels"), dict):
            raise ValueError(f"{path}: not a move memory (version {cls.VERSION})")
        for name, ents in d["levels"].items():
            lv = m.levels.setdefault(str(name), {})
            for e in ents:
                c, r, move, deaths, tries = e
                if not (isinstance(move, str) and int(deaths) >= 0 and int(tries) >= 0):
                    raise ValueError(f"{path}: bad entry {e!r} for {name}")
                lv[(int(c), int(r), move)] = [int(deaths), int(tries)]
        return m

    def save(self, path: str | None = None):
        path = path or self.path
        if path is None:
            return
        d = {"version": self.VERSION, "levels": {
            name: [[c, r, mv, n[0], n[1]] for (c, r, mv), n in sorted(ents.items())] for name, ents in
            sorted(self.levels.items())}}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, separators=(",", ":"))
        os.replace(tmp, path)
        self.dirty = False

    def set_level(self, name: str | None):
        self.level = name
        self.wave = {}

    def new_wave(self):
        self.wave = {}

    def _ent(self, cell, move: str, make: bool = False):
        lv = self.levels.setdefault(self.level, {}) if make else self.levels.get(self.level, {})
        key = (int(cell[0]), int(cell[1]), move)
        if make and key not in lv:
            lv[key] = [0, 0]
        return lv.get(key)

    def stats(self, cell, move: str) -> tuple[int, int]:
        e = self._ent(cell, move)
        return (0, 0) if e is None else (e[0], e[1])

    def penalty(self, cell, move: str, scale: float) -> float:
        d, t = self.stats(cell, move)
        return d / (t + 1.0) * scale if d else 0.0

    def reason(self, cell, move: str, scale: float) -> str:
        d, t = self.stats(cell, move)
        return (f"{move} from ({int(cell[0])},{int(cell[1])}) died {d} of {t} tries "
                f"(+{self.penalty(cell, move, scale):.0f} frames)")

    def decided(self, agent: int, frame: int, cell, move: str, safe: int, adopted: bool):
        self.wave.setdefault(int(agent), []).append((int(frame), (int(cell[0]), int(cell[1])), move, int(safe)))
        if adopted and move != "pad" and self.learn:
            self._ent(cell, move, make=True)[1] += 1
            self.dirty = True

    def died(self, agent: int, frame: int, cause=None) -> dict:
        """The agent died at `frame`: charge its point of no return. Returns the event."""
        decs = self.wave.pop(int(agent), [])
        pnr = next((d for d in reversed(decs) if d[3] > 0 and d[2] != "pad"), None)
        ev = {"kind": "point of no return", "agent": int(agent), "frame": int(frame), "death_frame": int(frame),
              "cause": cause, "decisions": len(decs)}
        if pnr is None or not self.learn:
            ev.update(decision_frame=None if pnr is None else pnr[0], cell=None if pnr is None else list(pnr[1]),
                      move=None if pnr is None else pnr[2], charged=False,
                      reason="no decision of this wave had a safe alternative: nothing charged" if pnr is None
                      else "held-out level: nothing charged")
            return ev
        f0, cell, move, safe = pnr
        e = self._ent(cell, move, make=True)
        e[0] += 1
        self.dirty = True
        ev.update(decision_frame=f0, cell=list(cell), move=move, safe_alternatives=safe, charged=True,
                  deaths=e[0], tries=e[1])
        return ev


# ---- the planner ------------------------------------------------------------------------------------------
def state_row(x, y, vx, vy, flags, state, keys=0, held=0) -> np.ndarray:
    """One agent row (wm.buffer.ROW) from the game's state values."""
    r = np.zeros(1, ROW)
    r["x"], r["y"], r["vx"], r["vy"] = x, y, vx, vy
    r["flags"], r["state"], r["status"], r["keys"], r["held"] = int(flags) & 0xffffffff, state, 0, int(keys), held
    return r


def continuations(a_last: int) -> list[tuple[str, int]]:
    """The completion fan after a plan whose last action is a_last (no jump in any): keep its horizontal input
    (and run), neutral, reverse. A plan ending without a horizontal input: neutral, left, right (keep and
    reverse would both be neutral)."""
    L, R, _, run = ACTIONS[int(a_last)]
    d = int(R) - int(L)
    if d == 0:
        return [("neutral", act(0)), ("left", act(-1)), ("right", act(1))]
    return [("keep", act(d, run=bool(run))), ("neutral", act(0)), ("reverse", act(-d, run=bool(run)))]


class _Floor:
    """The safety sweep's map-bottom rule (plan.geometry.Terrain.on_padding) on the planner's live tiles and
    moving platforms (a body on a platform at the map's bottom has ground)."""

    def __init__(self, planner):
        self.pl = planner

    def on_padding(self, x, y, dy=None):
        return self.pl.terrain.on_padding(x, y, self.pl.lv.plats or None, dy=dy)


class Planner:
    """One planner per level (several agents may share it; the previous plan is kept per agent)."""

    def __init__(self, model, level, field: ProgressField, H: int = H_DEFAULT, B1: int = B1_DEFAULT,
                 branch: int = BRANCH_DEFAULT, S_final: int = S_DEFAULT, budget_ms: float | None = BUDGET_MS,
                 member: int = 0, memory: FailureMemory | None = None, hysteresis: float = 0.0,
                 collide_tol: float | None = COLLIDE_TOL, moves: "MoveMemory | None | str" = "new",
                 complete_max: int = COMPLETE_MAX, modes: bool = False):
        self.model = WorldModel.load(model).eval() if isinstance(model, str) else model
        self.H, self.B1, self.branch, self.S, self.budget = H, B1, branch, S_final, budget_ms
        self.hysteresis = float(hysteresis)          # frames (switch rule); 0: the best total is chosen
        self.collide_tol = collide_tol               # the collision guard's tolerance; None: no guard
        self.complete_max = int(complete_max)        # airborne completion frames (0: none)
        self.modes = bool(modes) and len(self.model.members) >= 5     # long-safe / normal / reflex on own members
        self.member = MEMBERS["normal"] if self.modes else member
        self.final_ms = FINAL_MS0
        self.f = int(getattr(self.model, "step", 1))   # frames per model step (a step-4 model: 4)
        self.lib = list(LIBRARY)
        self.lib_len = np.array([len(m) for m in self.lib])
        self.lib_end = -(-self.lib_len // self.f) * self.f    # judged at the next step boundary (frames)
        self.lib_actions = np.stack([padded(m.actions, MAX_LEN) for m in self.lib])
        self.moves = MoveMemory() if isinstance(moves, str) and moves == "new" else moves
        self.floor = _Floor(self)                     # the sweep's map-bottom rule (plan.geometry.Terrain.on_padding)
        self.last_stage1 = None                      # the last decision's stage-1 estimates (tests, offline checks)
        self._dead_t = DEAD_FAR                      # T charged for a dead end (set per decision from T(start))
        self.last_final = self.last_completion = None    # the last decision's final stage / completion (checks)
        self.set_level(level, field, memory)

    @property
    def future_frames(self) -> int:
        """How many future frames a decision's inputs should cover (worlds; frames + 1): H + completion (the
        long-safe horizon with modes)."""
        return max(self.H + self.complete_max, H_LONG if self.modes else 0)

    # ---- level state -------------------------------------------------------------------------------------
    def set_level(self, level, field: ProgressField, memory: FailureMemory | None = None):
        """A (new) level for this planner; the loaded model and the final stage's cost estimate stay. The level
        geometry, the progress field and every per-level memory (previous plans, failed moves, the surprise
        tracker's predictions) start again; the point-of-no-return memory switches to this level's entries."""
        self.lv = level
        self.set_field(field)
        self.geom = LevelGeom(level.words0, {"ox": level.ox, "oy": level.oy})
        self.words = np.asarray(level.words0, np.uint32).copy()
        self.terrain = Terrain(level.words0, level.ox, level.oy)
        self.guard = Terrain(level.words0, level.ox, level.oy, slopes=False)    # the collision guard's tiles
        self._guard_keys = {}
        self.bounds = S.level_bounds(level)
        # what the agent touches to finish / pick a key (contact boxes; empty without contact_reach.json)
        self.goal_boxes = np.asarray(getattr(level, "goal_boxes", list)(), np.float64).reshape(-1, 5)
        self.key_boxes = list(getattr(level, "key_boxes", list)())
        self.memory = memory if memory is not None else FailureMemory()
        if self.moves is not None:
            self.moves.set_level(getattr(level, "name", None))
        self.surprise = SurpriseTracker()
        self.prev: dict[int, Plan] = {}
        self.long: dict[int, dict] = {}               # agent -> the last long-safe plan {frame, actions, segs, valid}

    def set_field(self, field: ProgressField):
        """A (rebuilt) progress field: also T_support at the spawn (the point-of-no-return charge's scale)."""
        self.field = field
        sc, sr = field.start
        ok = 0 <= sc < field.W and 0 <= sr < field.H
        self.t_spawn = float(field.T_support[0, sr, sc]) if ok else math.inf

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
        self.guard.update(self.words)
        self._guard_keys = {}

    def guard_terrain(self, keys: int) -> Terrain | None:
        """The collision guard's tiles for an agent holding `keys`: the live tiles (plan.geometry.solid_now's
        rule: crumble / lock / switch blocks solid while their word is the start one), slope cells open (real
        agents are recorded inside some slopes' modelled triangles), the lock tiles of the held keys open (the
        game opens them for that agent only; keys picked up during an imagined path do not open their locks
        there). Moving platforms are not tiles: not in the guard. None when the guard is off."""
        if self.collide_tol is None:
            return None
        keys = int(keys)
        if not keys or not getattr(self.lv, "lock_groups", None):
            return self.guard
        t = self._guard_keys.get(keys)
        if t is None:
            t = self._guard_keys[keys] = self.guard.opened(self.lv.opened_cells(keys))
        return t

    def new_wave(self):
        self.prev = {}
        self.long = {}                               # agent -> the last long-safe plan {frame, actions, segs, valid}
        self.memory.pending = {}
        if self.moves is not None:
            self.moves.new_wave()
        self.surprise.new_wave()

    def cell(self, x, y) -> tuple:
        return int((x - self.lv.ox) // 20.0), int((y - self.lv.oy) // 20.0)

    def support(self, x, y, vy=None):
        """The scoring's support rule at positions (arrays): (supported, ground, wall side). Ground, or a wall
        (plan/geometry.py's probe) while the downward speed -vy (units / frame; None: unknown, any speed) is at
        most CLING_SPEED. The explanation's landings / wall jumps and the safety check use the geometry alone."""
        g, side = self.terrain.support(x, y, self.lv.plats or None)
        wall = side != 0
        if vy is not None and math.isfinite(CLING_SPEED):
            wall = wall & (-np.asarray(vy, np.float64) <= CLING_SPEED)
        return g | wall, g, side

    def value(self, x, y, keys=0, leg=None, vy=None):
        """V at positions (arrays) for an agent holding `keys`, or on leg `leg` (an int or one per position):
        (V, supported, ground, wall side, leg). Supported: Planner.support (vy: the bodies' vertical velocity in
        units / frame, for the cling rule).

        Points inside solid: ProgressField.values gives a point in a wall cell its best open 3 x 3 neighbour,
        which can be across a floor. With the collision guard the score's points (every stage's end points, per
        member) are practically never inside solid geometry: measured on recorded states (no deadlines; 1-14 on
        the half-block floor 40 decisions, 1-13 and 1-12 30 each): 0 / 11,280, 0 / 8,460 and 1 / 8,452 (a slope)
        points, against 1,888 / 599 / 111 without the guard; so the rule is unchanged. Not the same thing: a
        body standing on a lower half block has its centre in the cell's open upper half, a wall cell of the
        field, so it still takes the best neighbour, which there is the cell below the floor (1-14: V 13.1 on the
        floor at x 110 from the cell below, 22.4 just above it)."""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        sup, g, side = self.support(x, y, vy)
        if leg is None:
            leg = int(self.field.leg_of(np.int64(int(keys))))
        return self.field.values(x, y, leg, sup), sup, g, side, leg

    def time_to_go(self, x, y, keys=0, leg=None, vy=None):
        """T (frames to the goal, plan/progress.py) at positions for an agent holding `keys` or on leg `leg`:
        (T, supported, ground, wall side, leg). Supported by Planner.support (cling rule with vy); airborne
        bodies take T_rise when vy > 0 (units / frame), else T_fall (vy None: the better of the two)."""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        sup, g, side = self.support(x, y, vy)
        if leg is None:
            leg = int(self.field.leg_of(np.int64(int(keys))))
        return self.field.times(x, y, leg, sup, vy), sup, g, side, leg

    def _contacts(self, xs, ys, upto, death, leg: int):
        """Goal and key contact of paths xs, ys (E, n, K+1 columns of self.f frames; every frame in and out is a
        frame) on their frames 1..upto ((n,), None: all K), given
        each path's first deadly contact frame `death` (E, n) (plan.safety.Sweep.contact_frame, -1 none).
        Returns goal (E, n): the first frame touching a goal before any deadly contact (-1 none); legs (E, n):
        the leg after the route's next keys taken in order before any deadly contact; keys: per key leg tried
        (leg j, pickup frame (E, n) or -1, key index (E, n) or -1)."""
        E, n = xs.shape[:2]
        f = self.f
        safe = lambda t: (t > 0) & ((death < 0) | (death > t))
        g, _ = S.first_contact(xs, ys, self.goal_boxes, upto=upto, step=f)
        goal = np.where(safe(g), g, -1)
        legs = np.full((E, n), leg, np.int64)
        keys = []
        on = np.ones((E, n), bool)                     # every earlier key leg taken
        last = np.zeros((E, n), np.int64)              # frame of the last pickup
        never = (xs.shape[2] + 1) * f
        for j in range(leg, len(self.field.key_order)):
            ks = [k for k in self.field.key_order[j] if k < len(self.key_boxes)]
            if not ks or not on.any():
                break
            t, w = S.first_contact(xs, ys, [self.key_boxes[k] for k in ks], first=np.where(on, last + 1, never),
                                   upto=upto, step=f)
            ok = on & safe(t)
            if not ok.any():
                break
            legs = np.where(ok, j + 1, legs)
            keys.append((j, np.where(ok, t, -1), np.where(ok, np.asarray(ks)[np.maximum(w, 0)], -1)))
            on, last = ok, np.where(ok, t, last)
        return goal, legs, keys

    def observe(self, agent: int, frame: int, x: float, y: float, keys: int = 0, dead: bool = False,
                cause=None) -> list[dict]:
        """The runner calls this with every real state: surprise events (SurpriseTracker), failed moves
        (FailureMemory.update) and, on a death (cause: what killed it, for the log), the point of no return
        (MoveMemory.died)."""
        ev = self.surprise.observe(agent, frame, x, y, dead)
        v = None if dead else float(self.value(np.array([x]), np.array([y]), keys)[0][0])
        ev = ev + self.memory.update(agent, frame, v, dead)
        if dead and self.moves is not None:
            ev.append(self.moves.died(agent, frame, cause))
        return ev

    def played(self, agent: int, frame: int, action: int):
        """The runner calls this with the action index it really played for the agent on this frame (the
        surprise check counts a prediction only while its plan was followed)."""
        self.surprise.played(agent, frame, action)

    # ---- airborne completion -----------------------------------------------------------------------------
    def complete(self, rows, a_last, legs, worlds, frames, dt_ms, keys, H):
        """C1b: from each candidate's main-member end row at H (rows (m,) ROW, its last action a_last (m,), its
        leg there legs (m,)), the continuation fan (continuations()) rolled out with the main member up to
        complete_max frames in batches of COMPLETE_CHUNK (rows that landed or died stop; all stopped: early
        end). Landed: the first frame the cling rule calls supported, or a goal touch (T = 0 there). Died: the
        safety sweep's deadly contact (hazard boxes, remembered deaths, the level's edge) against frames[H + k]
        (the last given frame's boxes beyond the inputs). Returns per candidate (estimate frames after H,
        completion risk 0 / 1, the record {continuation, frames, x, y, on, died}, its path [[x, y], ...]) and
        the frames rolled out. A step-f model: complete_max and the batches round up to whole steps; the paths
        are per step, every frame reported is a frame."""
        m_all = len(rows)
        f = self.f
        fan = [continuations(a) for a in a_last]
        nf = len(fan[0])
        cand = np.repeat(np.arange(m_all), nf)
        names = [nm for f in fan for nm, _ in f]
        acts = np.array([a for f in fan for _, a in f], np.int64)
        n = len(cand)
        s = np.repeat(np.asarray(rows, ROW), nf)
        leg = np.asarray(legs, np.int64)[cand]
        Kmax = -(-self.complete_max // f) * f            # frames (whole steps)
        chunk = -(-COMPLETE_CHUNK // f) * f
        xs = np.zeros((n, Kmax // f + 1))
        ys = np.zeros((n, Kmax // f + 1))
        xs[:, 0], ys[:, 0] = s["x"], s["y"]
        state = np.zeros(n, np.int8)                   # 0 flying, 1 landed, 2 goal, 3 died
        at = np.full(n, -1, np.int64)
        hit = [None] * n
        vy0 = s["vy"].astype(np.float64) * dt_ms / 1000.0
        sup0, g0, side0 = self.support(s["x"].astype(np.float64), s["y"].astype(np.float64), vy0)
        state[sup0] = 1
        at[sup0] = 0
        on = np.where(g0, "ground", "wall").astype(object)
        last_vy = vy0.copy()
        mem = slice(self.member, self.member + 1)
        ck = {"terrain": self.guard_terrain(keys), "tol": self.collide_tol}
        done, rolled = 0, 0
        clipped = np.zeros(n, bool)
        while done < Kmax:
            act_i = np.nonzero(state == 0)[0]
            if not len(act_i):
                break
            K = min(chunk, Kmax - done)
            Kc, dc = K // f, done // f                   # columns of this batch, columns done
            out = imagine(self.model, mem, self.geom, s[act_i], np.repeat(acts[act_i, None], K, 1), dt_ms, worlds,
                          offsets=np.full(len(act_i), H + done), **ck)
            rolled += len(act_i) * K
            if out["clip"] is not None:
                clipped[act_i] |= out["clip"][0].any(1)
            x, y, vy = out["x"][0], out["y"][0], out["vy"][0]               # (a, Kc+1)
            fr = [frames[min(H + done + k * f, len(frames) - 1)] if k else None for k in range(Kc + 1)]
            fr = _near_frames(fr, x[None], y[None])
            sw = S.sweep(x[None], y[None], fr, self.bounds, self.floor, f)
            cf = sw.contact_frame[0]
            tg = S.first_contact(x[None], y[None], self.goal_boxes, step=f)[0][0] if len(self.goal_boxes) else \
                np.full(len(act_i), -1)
            sup, g, side = self.support(x[:, 1:], y[:, 1:], vy[:, 1:])
            land = np.where(sup.any(1), (sup.argmax(1) + 1) * f, -1)
            for j, r in enumerate(act_i):
                ev = [(t, kd) for t, kd in ((int(land[j]), 1), (int(tg[j]), 2), (int(cf[j]), 3)) if t > 0]
                if not ev:
                    continue
                t, kd = min(ev, key=lambda e: (e[0], -e[1]))     # same frame: a deadly contact, then the goal, then support
                state[r], at[r] = kd, done + t
                tc = t // f
                if kd == 1:
                    on[r] = "ground" if g[j, tc - 1] else "wall"
                elif kd == 2:
                    on[r] = "goal"
                else:
                    kind = int(sw.contact_kind[0, j])
                    hit[r] = S.contact_id(kind, sw.ids, sw.contact_box[0, j])
                xs[r, dc + 1:dc + tc + 1], ys[r, dc + 1:dc + tc + 1] = x[j, 1:tc + 1], y[j, 1:tc + 1]
                xs[r, dc + tc + 1:], ys[r, dc + tc + 1:] = x[j, tc], y[j, tc]
            fly = state[act_i] == 0
            fi = act_i[fly]
            xs[fi, dc + 1:dc + Kc + 1], ys[fi, dc + 1:dc + Kc + 1] = x[fly, 1:], y[fly, 1:]
            last_vy[fi] = vy[fly, -1]
            s[fi] = out["last"][0][fly]
            done += K
        # the scores
        end_c = np.where(at >= 0, at, done) // f
        xe, ye = xs[np.arange(n), end_c], ys[np.arange(n), end_c]
        t_land = np.where(state == 2, 0.0, np.inf)
        k = np.nonzero(state == 1)[0]
        if len(k):
            t_land[k] = self.field.times(xe[k], ye[k], leg[k], True)
        t_fly = np.full(n, np.inf)
        k = np.nonzero(state == 0)[0]
        if len(k):
            t_fly[k] = self.field.times(xe[k], ye[k], leg[k], False, last_vy[k])
        res = []
        for c in range(m_all):
            idx = np.arange(c * nf, (c + 1) * nf)
            died = [names[i] for i in idx if state[i] == 3]
            ok = [i for i in idx if state[i] in (1, 2)]
            if ok:
                i = min(ok, key=lambda i: at[i] + self._tt(t_land[i]))
                est, risk = at[i] + self._tt(t_land[i]), 0
            elif len(died) == nf:
                i = max(idx, key=lambda i: at[i])                       # the one that lived longest (shown)
                xm, ym = float(rows["x"][c]), float(rows["y"][c])
                vm = float(rows["vy"][c]) * dt_ms / 1000.0
                t_end = self.field.times(np.array([xm]), np.array([ym]), int(legs[c]), False, np.array([vm]))[0]
                est, risk = self._tt(t_end), 1
            else:
                fly_i = [i for i in idx if state[i] == 0]
                i = min(fly_i, key=lambda i: self._tt(t_fly[i]))
                est, risk = done + self._tt(t_fly[i]), 0
            kk = int(at[i]) if at[i] >= 0 else done
            rec = {"continuation": names[i] if ok or risk == 0 else None, "frames": kk if (ok or not risk) else None,
                   "x": round(float(xe[i]), 2), "y": round(float(ye[i]), 2),
                   "on": on[i] if ok else ("air" if not risk else None), "died": died,
                   "hits": {names[j]: hit[j] for j in idx if state[j] == 3}}
            res.append((float(est), risk, rec, [[float(a), float(b)] for a, b in zip(xs[i, :kk // f + 1], ys[i, :kk // f + 1])]))
        self.completion_clipped = getattr(self, "completion_clipped", 0) + int(clipped.sum())
        self.last_completion = {"frames": done, "rows": n, "flying_at_end": int((state == 0).sum()),
                                "xs": xs, "ys": ys, "names": names,
                                "states": state.copy(), "at": at.copy()}
        return res, rolled

    def _tt(self, T):
        """T with dead ends (+inf) charged: T(start) + DEAD_END_FRAMES (DEAD_FAR when the start is one too)."""
        T = np.asarray(T, np.float64)
        out = np.where(np.isfinite(T), T, self._dead_t)
        return float(out) if out.ndim == 0 else out

    # ---- one decision ------------------------------------------------------------------------------------
    def plan(self, row, frame: int, worlds: list, frames: list, dt_ms: float = 16.667, agent: int = 0,
             danger: bool = False) -> Plan:
        """row: the agent's state (wm.buffer.ROW, one row; state_row builds one). worlds: at least H World
        (model inputs, index k = frame + k); frames: H + 1 plan.safety.FrameBoxes (index k = frame + k; index 0
        unused); for the airborne completion they should cover H + complete_max (Planner.future_frames: the
        last given ones are repeated beyond them). Returns the chosen Plan (also kept as this agent's previous
        plan). danger: the commit "danger" trigger fired (REFLEX runs; modes only)."""
        t0 = time.perf_counter()
        f = self.f                                   # frames per model step; arrays are per step (column)
        H = -(-self.H // f) * f                      # frames, whole steps
        Hc = H // f
        row = np.asarray(row, ROW).reshape(1)
        x0, y0, keys = float(row["x"][0]), float(row["y"][0]), int(row["keys"][0])
        vy0 = np.array([float(row["vy"][0]) * dt_ms / 1000.0])
        t0a, sup0, g0, side0, leg = self.time_to_go(np.array([x0]), np.array([y0]), keys, vy=vy0)
        T0 = float(t0a[0])
        T0_ok = math.isfinite(T0)
        self._dead_t = T0 + DEAD_END_FRAMES if T0_ok else DEAD_FAR
        v0 = float(self.value(np.array([x0]), np.array([y0]), keys, vy=vy0)[0][0])
        v0_ok = math.isfinite(v0)
        cell0 = self.cell(x0, y0)
        scale = max(self.t_spawn - T0, DEAD_END_FRAMES) if T0_ok and math.isfinite(self.t_spawn) else DEAD_END_FRAMES
        death_of = (lambda nm: self.moves.penalty(cell0, nm, scale)) if self.moves is not None else (lambda nm: 0.0)
        self.death_frames = scale                    # DEATH_FRAMES of this decision (module docstring)
        mem = slice(self.member, self.member + 1)
        timing, counts = {}, {}
        gt = self.guard_terrain(keys)
        ck = {"terrain": gt, "tol": self.collide_tol}
        n_clip = lambda c, upto=None: 0 if c is None else int(
            (c if upto is None else c & (np.arange(c.shape[2])[None, None, :] <= np.asarray(upto)[None, :, None]))
            .any(axis=(0, 2)).sum())

        def ends_time(xe, ye, vye, legs=None):
            return self.time_to_go(xe, ye, keys, legs, vye)[0]

        def success_total(t, pen):
            # earlier touch first; the penalty frames (up to the touch) only break ties (< 1 frame)
            return SUCCESS_SCORE - t - TIE * np.minimum(pen, 400.0)

        def vetoed_total(vf, est):
            return VETO_SCORE + vf - VETO_TIE * np.minimum(est, 1e5)

        start_in_goal = bool(len(self.goal_boxes)) and bool(
            (box_distances(np.array([x0]), np.array([y0]), self.goal_boxes) < S.BODY).any())

        # ---- stage 1: every library move, one member ------------------------------------------------------
        n1 = len(self.lib)
        out1 = imagine(self.model, mem, self.geom, np.repeat(row, n1), self.lib_actions, dt_ms, worlds,
                       deadline=None if self.budget is None else t0 + STAGE1_SHARE * self.budget / 1000.0,
                       keep_rows=True, **ck)
        steps1 = out1["steps"]                       # model steps (columns); f frames each
        up1 = np.minimum(self.lib_end, steps1 * f)   # the move's frames (a step-f model: the next step boundary)
        uc1 = up1 // f
        end1 = steps1 * f                            # one end frame for every move (rolled to the same length)
        fr1 = _near_frames(_cols(frames, steps1 + 1, f), out1["x"][:, :, :steps1 + 1], out1["y"][:, :, :steps1 + 1])
        sw1 = S.sweep(out1["x"][:, :, :steps1 + 1], out1["y"][:, :, :steps1 + 1], fr1, self.bounds, self.floor, f)
        tg1, legs1, _ = self._contacts(out1["x"][:, :, :steps1 + 1], out1["y"][:, :, :steps1 + 1],
                                       None, sw1.contact_frame, leg)
        tg1, legs1 = tg1[0], legs1[0]
        succ1 = tg1 > 0
        # safety over every frame the score credits: the estimate is taken at the common end (a success: its goal
        # touch), so soft risk counts frames 1..end1 (1..t), the held last action included; the veto and the
        # margin rule stay on the move's own frames (or up to its touch); the held tail counts contact only
        own1 = np.where(succ1, np.minimum(up1, tg1), up1)
        veto1, vf1, vk1, vid1, soft1, sf1 = _verdict(sw1, np.where(succ1, tg1, end1), veto_upto=own1,
                                                     planned=np.where(succ1, tg1, up1))
        xe1, ye1, vye1 = out1["x"][0, :, steps1], out1["y"][0, :, steps1], out1["vy"][0, :, steps1]
        tend1 = ends_time(xe1, ye1, vye1, legs1)
        fail1 = np.array([self.memory.penalty(cell0, m.name) for m in self.lib])
        death1 = np.array([death_of(m.name) for m in self.lib])
        pen1 = scale * soft1 + fail1 + death1
        est1 = end1 + self._tt(tend1) + pen1
        score1 = np.where(veto1, vetoed_total(vf1, est1), -est1)
        score1 = np.where(succ1, success_total(tg1, pen1), score1)
        order1 = np.argsort(-score1, kind="stable")
        keep1 = order1[:self.B1]
        self.last_stage1 = {"names": [m.name for m in self.lib], "frames": up1, "t_end": tend1, "estimate": est1}
        t1 = time.perf_counter()
        timing["stage1"] = 1000 * (t1 - t0)
        counts["stage1"] = {"candidates": n1, "steps": int(steps1), "complete": int((self.lib_end <= steps1 * f).sum()),
                            "vetoed": int(veto1.sum()), "goal": int(succ1.sum()), "clipped": n_clip(out1["clip"], uc1)}

        # ---- stage 2: second moves from the kept end states, one member ------------------------------------
        seconds = [BY_NAME[nm] for nm in CORE_SECOND]
        for i in order1:
            if len(seconds) >= self.branch:
                break
            if self.lib[i].name not in CORE_SECOND and not veto1[i]:
                seconds.append(self.lib[i])
        seconds = seconds[:self.branch]
        cands = []                                       # (segments, actions H)
        stage2 = steps1 * f >= self.lib_end[keep1].max() if len(keep1) else False
        if stage2:
            K2 = H - int(self.lib_end[keep1].min())
            s0, acts2, offs, firsts = [], [], [], []
            for i in keep1:
                L0, L = int(self.lib_len[i]), int(self.lib_end[i])     # the move; its step boundary (f = 1: same)
                first = self.lib[i].array if L == L0 else padded(self.lib[i].array, L)
                for sm in seconds:
                    full = padded(np.concatenate([first, sm.array]), H)
                    cont = padded(full[L:], K2)
                    s0.append(out1["rows"][0, i, L // f])
                    acts2.append(cont)
                    offs.append(L)
                    firsts.append(i)
                    L2 = min(H, L + len(sm))
                    segs = ([(self.lib[i].name, 0, L0)] + ([("pad", L0, L)] if L > L0 else []) + [(sm.name, L, L2)]
                            + ([("pad", L2, H)] if L2 < H else []))
                    cands.append((segs, full))
            s0 = np.array(s0, ROW)
            dl2 = (None if self.budget is None else
                   t0 + max(STAGE2_SHARE * self.budget, self.budget - self.final_ms) / 1000.0)
            out2 = imagine(self.model, mem, self.geom, s0, np.stack(acts2), dt_ms, worlds, offsets=offs, deadline=dl2,
                           **ck)
            steps2 = out2["steps"]
            n2 = len(cands)
            xs2, ys2 = np.zeros((1, n2, Hc + 1)), np.zeros((1, n2, Hc + 1))
            vye2 = np.zeros(n2)
            up2 = np.zeros(n2, np.int64)
            for c in range(n2):
                i, L = firsts[c], offs[c] // f                           # columns
                m = min(steps2, Hc - L)
                xs2[0, c, :L + 1], ys2[0, c, :L + 1] = out1["x"][0, i, :L + 1], out1["y"][0, i, :L + 1]
                xs2[0, c, L + 1:L + m + 1], ys2[0, c, L + 1:L + m + 1] = out2["x"][0, c, 1:m + 1], out2["y"][0, c, 1:m + 1]
                xs2[0, c, L + m + 1:], ys2[0, c, L + m + 1:] = xs2[0, c, L + m], ys2[0, c, L + m]
                up2[c] = (L + m) * f
                vye2[c] = out2["vy"][0, c, m] if m > 0 else out1["vy"][0, i, L]
            K = int(up2.max()) // f
            fr2 = _near_frames(_cols(frames, K + 1, f), xs2[:, :, :K + 1], ys2[:, :, :K + 1])
            sw2 = S.sweep(xs2[:, :, :K + 1], ys2[:, :, :K + 1], fr2, self.bounds, self.floor, f)
            tg2, legs2, _ = self._contacts(xs2[:, :, :K + 1], ys2[:, :, :K + 1], up2, sw2.contact_frame, leg)
            tg2, legs2 = tg2[0], legs2[0]
            succ2 = tg2 > 0
            veto2, vf2, _, _, soft2, _ = _verdict(sw2, np.where(succ2, np.minimum(up2, tg2), up2))
            tend2 = ends_time(xs2[0, np.arange(n2), up2 // f], ys2[0, np.arange(n2), up2 // f], vye2, legs2)
            pen2 = scale * soft2 + np.array([fail1[i] + death1[i] for i in firsts])
            est2 = up2 + self._tt(tend2) + pen2
            score2 = np.where(veto2, vetoed_total(vf2, est2), -est2)
            score2 = np.where(succ2, success_total(tg2, pen2), score2)
            pick = np.argsort(-score2, kind="stable")[:self.S]
            final = [cands[c] for c in pick]
            counts["stage2"] = {"candidates": n2, "steps": int(steps2), "of_steps": int(-(-K2 // f)),
                                "complete": int((up2 >= H).sum()), "vetoed": int(veto2.sum()),
                                "second_moves": [sm.name for sm in seconds], "goal": int(succ2.sum()),
                                "clipped": n_clip(out2["clip"])}
        else:
            final = []
            for i in order1[:self.S]:
                L = int(self.lib_len[i])
                final.append(([(self.lib[i].name, 0, L)] + ([("pad", L, H)] if L < H else []),
                              padded(self.lib[i].array, H)))
            counts["stage2"] = {"candidates": 0, "steps": 0, "skipped": "stage 1 was cut by the deadline"}
        t2 = time.perf_counter()
        timing["stage2"] = 1000 * (t2 - t1)

        # ---- modes: long-safe (member 0) and reflex (member 2) candidates ------------------------------------
        modes3 = ["normal"] * len(final)
        li = ri = None                               # final indices of the long-safe / reflex candidates
        ri_new = False
        if self.modes:
            pen_first = fail1 + death1
            tl = time.perf_counter()
            ls = self.long.get(agent)
            age = frame - ls["frame"] if ls is not None else None
            usable = ls is not None and ls["valid"] and 0 <= age < len(ls["actions"]) - 1
            ran = bool(sup0[0]) and (not usable or age > LONG_EVERY)
            cl = {"ran": ran}
            if ran:
                res, cnt_l = self._long_safe(row, worlds, frames, dt_ms, ck, leg, keys, pen_first, scale)
                cl.update(cnt_l)
                if res is not None:
                    ls = self.long[agent] = {"frame": int(frame), "segs": res[0], "actions": [int(a) for a in res[1]],
                                             "valid": True}
                    age, usable = 0, True
                else:
                    self.long.pop(agent, None)
                    usable = False
            if usable:
                segs = [(nm, max(0, a - age), min(b - age, H)) for nm, a, b in ls["segs"]
                        if b - age > 0 and a - age < H and nm != "pad"]
                acts_l = padded(np.asarray(ls["actions"], np.int64)[age:], H)
                end = segs[-1][2] if segs else 0
                if end < H:
                    segs.append(("pad", end, H))
                dup = [k for k, (_, a) in enumerate(final) if np.array_equal(a, acts_l)]
                if dup:
                    li = dup[0]
                    modes3[li] = "long-safe"
                else:
                    final.append((segs, acts_l))
                    modes3.append("long-safe")
                    li = len(final) - 1
                cl["age"] = int(age)
            cl["candidate"] = li is not None
            timing["long_safe"] = 1000 * (time.perf_counter() - tl)
            tr = time.perf_counter()
            near = danger or self._danger_near(agent, frame, x0, y0, frames)
            cr = {"ran": bool(near), "commit_danger": bool(danger)}
            if near:
                k, r_risk, r_est = self._reflex(row, worlds, frames, dt_ms, ck, leg, keys, pen_first)
                L = int(self.lib_len[k])
                acts_r = padded(self.lib[k].array, H)
                cr.update(move=self.lib[k].name, risk=round(r_risk, 3), estimate=round(r_est, 1))
                dup = [c for c, (_, a) in enumerate(final) if np.array_equal(a, acts_r)]
                if dup:
                    ri = dup[0]
                else:
                    final.append(([(self.lib[k].name, 0, L)] + ([("pad", L, H)] if L < H else []), acts_r))
                    modes3.append("reflex")
                    ri, ri_new = len(final) - 1, True
            timing["reflex"] = 1000 * (time.perf_counter() - tr)
            counts["modes"] = {"long_safe": cl, "reflex": cr}

        # ---- final: re-score with all members and the full check -------------------------------------------
        prev = self.prev.get(agent)
        prev_idx, shift = -1, 0
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
                modes3.append(getattr(prev, "mode", "normal"))
                prev_idx = len(final) - 1
        n3 = len(final)
        acts3 = np.stack([a for _, a in final])
        # every candidate, the shifted previous plan included, is judged at the same end frame H (its unplanned
        # tail is the pad: last action held, jump released): estimates of one stage compare over one window.
        # Safety covers every frame the score credits: a non-success is scored by its state at H (or its
        # completion from H), so it is judged on frames 1..H: the veto on 1..VETO_FRAMES, the margin rule on its
        # later planned frames (pads excluded; the previous plan: its remaining planned frames), contact only
        # (soft risk 1, no margin) on the never-planned pad tail up to H; a success on frames 1..t (its goal
        # touch), margin rule throughout. (No tail check credited a pad tail through a hazard with the progress
        # of passing it and no risk: bench_20261108-221922, 1-8. The margin on the tail refused passages: a pad
        # passing a saw at 40 units cost most of a death; bench_20261109-001206, 1-13.)
        planned3 = np.array([max((b for nm, a, b in segs if nm != "pad"), default=0) for segs, _ in final], np.int64)
        planned3 = np.minimum(np.maximum(-(-planned3 // f) * f, S.VETO_FRAMES), H)   # step-f model: step boundary
        out3 = imagine(self.model, slice(None), self.geom, np.repeat(row, n3), acts3, dt_ms, worlds, **ck)
        E = out3["x"].shape[0]
        fr3 = _near_frames(_cols(frames, Hc + 1, f), out3["x"], out3["y"])
        saf = S.check(out3["x"], out3["y"], fr3, self.bounds, self.floor, f)
        tg_e, legs_e, keys_e = self._contacts(out3["x"], out3["y"], None, saf.sweep.contact_frame, leg)
        n_goal = (tg_e > 0).sum(0)                                   # members touching the goal safely
        succ3 = 2 * n_goal > E
        tg3 = np.full(n3, -1, np.int64)
        for c in np.nonzero(succ3)[0]:
            ts = np.sort(tg_e[tg_e[:, c] > 0, c])
            tg3[c] = ts[(len(ts) - 1) // 2]                          # median (the lower one of an even count)
        win3 = np.where(succ3, tg3, H)                               # successes: frames 1..t; others: 1..H
        vv, vfr, vkd, vid, vsoft, vsf = _verdict(saf.sweep, win3, planned=np.where(succ3, tg3, planned3))
        saf = replace(saf, veto=vv, veto_frame=vfr, veto_kind=vkd, veto_id=vid, soft_risk=vsoft, soft_frame=vsf)
        xe, ye, vye = out3["x"][:, :, Hc], out3["y"][:, :, Hc], out3["vy"][:, :, Hc]
        tend_e, sup_e, _, _, _ = self.time_to_go(xe, ye, keys, legs_e, vye)      # (E, n3), each member's own leg
        vend_e = self.value(xe, ye, keys, legs_e, vye)[0]                         # V: display only
        lands = 2 * sup_e.sum(0) > E                                 # a majority of members supported at H
        tend3 = self._tt(tend_e).mean(0)
        base3 = H + tend3                                            # frames used + T at the end
        frames_used = np.full(n3, H, np.int64)
        t_end3 = tend3.copy()
        comp_risk = np.zeros(n3)
        comp_rec, comp_path = [None] * n3, [None] * n3
        tc = time.perf_counter()
        need = np.nonzero(~lands & ~succ3 & ~saf.veto)[0] if self.complete_max > 0 else np.zeros(0, np.int64)
        rolled = 0
        self.completion_clipped = 0
        if len(need):
            mi = min(self.member, E - 1)
            res, rolled = self.complete(out3["last"][mi, need], acts3[need, H - 1], legs_e[mi, need], worlds,
                                        frames, dt_ms, keys, H)
            for c, (est, risk, rec, path) in zip(need, res):
                base3[c] = H + est
                comp_risk[c], comp_rec[c], comp_path[c] = risk, rec, path
                k = rec["frames"] if rec["frames"] is not None else 0
                frames_used[c] = H + k
                t_end3[c] = est - k
        timing["completion"] = 1000 * (time.perf_counter() - tc)
        first_name = [next((nm for nm, a, b in segs if nm != "pad"), "pad") for segs, _ in final]
        fail3 = np.array([self.memory.penalty(cell0, nm) for nm in first_name])
        death3 = np.array([death_of(nm) for nm in first_name])
        soft3 = np.minimum(saf.soft_risk + comp_risk, 1.0)          # one death at most
        risk3 = scale * soft3
        est3 = base3 + risk3 + fail3 + death3
        total3 = np.where(saf.veto, vetoed_total(saf.veto_frame, est3), -est3)
        total3 = np.where(succ3, success_total(tg3, risk3 + fail3 + death3), total3)
        landing3 = lands | succ3 | np.array([r is not None and r["continuation"] is not None and r["on"] != "air"
                                             for r in comp_rec])
        sel = total3
        if ri_new and ri != prev_idx:                # the reflex plan is chosen only by its override
            sel = total3.copy()
            sel[ri] = -np.inf
        best, switch = self.switch_rule(int(np.argmax(sel)), prev_idx, sel, succ3, tg3, saf, soft3)
        mode = getattr(prev, "mode", "normal") if best == prev_idx else modes3[best]
        if ri is not None:
            er = early_risk(saf.sweep)
            if saf.veto[best] or er[best] > 0:       # REFLEX overrides a chosen plan with early danger
                why = (f"vetoed f{int(saf.veto_frame[best])}" if saf.veto[best] else f"early risk {er[best]:.2f}")
                switch = {"kept": False, "gain": None, "margin": None, "unit": "frames",
                          "reason": f"reflex override: the chosen plan was {why}"}
                best, mode = ri, "reflex"
            counts["modes"]["reflex"]["override"] = mode == "reflex"
        if li is not None and (saf.veto[li] or soft3[li] > LONG_RISK):
            self.long[agent]["valid"] = False        # no longer valid: searched again when supported
        if self.modes:
            counts["modes"]["candidates"] = modes3
        safe_alts = int(sum(1 for c in range(n3) if c != best and not saf.veto[c] and soft3[c] < SAFE_RISK
                            and landing3[c]))            # soft3: the window's risk and the completion's deaths
        self.last_final = {"estimate": est3, "veto": saf.veto, "soft": soft3, "landing": landing3, "completion": comp_rec,
                           "prev": prev_idx, "first": first_name, "best": best, "t_end": t_end3,
                           "frames_used": frames_used, "risk": risk3, "fail": fail3, "death": death3, "lands": lands,
                           "sup_e": sup_e, "tend_e": tend_e, "end_xy": (xe, ye), "end_vy": vye,
                           "soft_frame": saf.soft_frame, "comp_risk": comp_risk}  # offline checks
        t3 = time.perf_counter()
        timing["final"] = 1000 * (t3 - t2)
        self.final_ms = 0.7 * self.final_ms + 0.3 * timing["final"]
        counts["final"] = {"candidates": n3, "members": E, "steps": Hc, "vetoed": int(saf.veto.sum()),
                           "prev_included": prev_idx >= 0, "goal": int(succ3.sum()), "start_in_goal": start_in_goal,
                           "clipped": n_clip(out3["clip"]), "landing": int(lands.sum()), "completed": int(len(need)),
                           "completion_rows_frames": int(rolled), "completion_clipped": self.completion_clipped,
                           "completion_inputs": int(min(len(worlds), len(frames) - 1)),
                           "safe_alternatives": safe_alts,
                           "completions": [None if r is None else {k: r[k] for k in ("continuation", "frames", "on", "died")}
                                           for r in comp_rec]}

        # ---- explanation -----------------------------------------------------------------------------------
        segs, acts = final[best]
        mx, my = out3["x"][:, best].mean(0), out3["y"][:, best].mean(0)            # per column (step)
        spread = np.sqrt(out3["x"][:, best].var(0) + out3["y"][:, best].var(0))
        g, side = self.terrain.support(mx, my, self.lv.plats or None)
        sup = g | (side != 0)
        landings = [{"frame": k * f, "x": float(mx[k]), "y": float(my[k]), "on": "ground" if g[k] else "wall"}
                    for k in range(1, Hc + 1) if sup[k] and not sup[k - 1]]
        jump = np.array([ACTIONS[a][2] for a in acts], bool)
        wall_jumps = [{"frame": k * f, "x": float(mx[k]), "y": float(my[k]), "side": "L" if side[k] < 0 else "R"}
                      for k in range(Hc) if side[k] != 0 and not g[k] and jump[k * f] and (k == 0 or not jump[k * f - 1])]
        if f > 1:                                    # the chosen path per frame (commit tube, overlay, viewer)
            fr_c, fr_all = np.arange(Hc + 1) * f, np.arange(H + 1)
            mx, my, spread = (np.interp(fr_all, fr_c, v) for v in (mx, my, spread))
        fin_e = np.isfinite(vend_e[:, best])
        vend_mean = float(np.mean(vend_e[fin_e, best])) if fin_e.any() else None
        spread_end = np.sqrt(xe.var(0) + ye.var(0))
        goal_touch, key_touches = self._touches(out3["x"][:, best], out3["y"][:, best], tg_e[:, best],
                                                [(j, t[:, best], w[:, best]) for j, t, w in keys_e], E, f)
        parts = {"base": base3, "t_end": t_end3, "comp_death": comp_risk, "risk": risk3, "fail": fail3, "death": death3, "est": est3, "soft": soft3,
                 "first": first_name}
        alts = self._alternatives(final, best, saf, parts, total3, T0, cell0, scale, veto1, vf1, vk1, vid1, est1,
                                  paths=(out3["x"].mean(0), out3["y"].mean(0), out1["x"][0], out1["y"][0], uc1),
                                  goal=(succ3, tg3, n_goal, E), switch=switch)
        veto = None
        if saf.veto[best]:
            veto = {"frame": int(saf.veto_frame[best]), "id": saf.veto_id[best],
                    "kind": S.KIND_NAMES[int(saf.veto_kind[best])]}
        first_frames = next((b for nm, a, b in segs if nm != "pad"), 0)
        completion = None
        if comp_rec[best] is not None:
            completion = {**comp_rec[best], "risk": int(comp_risk[best]), "path": comp_path[best]}
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
            soft_risk=float(soft3[best]), soft_frame=int(saf.soft_frame[best]),
            score={"frames_used": int(frames_used[best]), "t_start": T0 if T0_ok else None,
                   "t_end": float(t_end3[best]), "estimate": float(est3[best]),
                   "risk_frames": float(risk3[best]), "failure_frames": float(fail3[best]),
                   "death_frames": float(death3[best]), "total": float(total3[best]),
                   "completion": None if completion is None else completion["frames"],
                   "landing": bool(landing3[best]), "spread_end": float(spread_end[best])},
            v_start=v0 if v0_ok else None, v_end=vend_mean, dead_end_members=int((~fin_e).sum()), leg=leg,
            end_supported=bool(sup[Hc]),
            alternatives=alts,
            predictions={h: [float(mx[h]), float(my[h])] for h in HORIZONS if h <= H},
            death_model=float(1.0 - out3["alive"][:, best, Hc].min()),
            goal_touch=goal_touch, key_touches=key_touches,
            finish_model=float(out3["finish"][:, best].max()),
            timing={**{k: round(v, 2) for k, v in timing.items()}, "total": round(1000 * (time.perf_counter() - t0), 2)},
            counts=counts, segments=[[nm, int(a), int(b)] for nm, a, b in segs], switch=switch,
            model_clipped=None if out3["clip"] is None else int(out3["clip"][:, best].any(0).sum()),
            completion=completion, mode=mode)
        plan.first_move_frames = int(first_frames)
        self.prev[agent] = plan
        if not plan.from_prev:                       # a kept plan keeps the predictions of its adoption
            self.surprise.add(plan)
        if self.moves is not None:
            self.moves.decided(agent, frame, cell0, plan.chosen[0] if plan.chosen else "pad", safe_alts,
                               adopted=not plan.from_prev)
        if first_frames > 0 and v0_ok:
            vm = self.value(np.array([mx[first_frames]]), np.array([my[first_frames]]), keys)[0][0]
            self.memory.expect(agent, frame, plan.chosen[0], first_frames, acts, cell0, v0, float(vm))
        return plan

    # ---- planning modes -----------------------------------------------------------------------------------
    def _judge(self, out, frames, upto, leg, keys):
        """Sweep, goal / key contact and the decision rule for an imagine() result (any members) judged on
        frames 1..upto (n,): {veto, veto_frame, soft, succ, tg, tend (T at the end, members' mean, dead ends
        charged), sweep}. A success: every member touches the goal before any deadly contact."""
        f = self.f
        x, y = out["x"], out["y"]
        E, n, K1 = x.shape
        fr = _near_frames(_cols(frames, K1, f), x, y)
        sw = S.sweep(x, y, fr, self.bounds, self.floor, f)
        tg_e, legs_e, _ = self._contacts(x, y, upto, sw.contact_frame, leg)
        succ = (tg_e > 0).all(0)
        tg = np.where(succ, tg_e.max(0), -1)
        veto, vf, _, _, soft, _ = _verdict(sw, np.where(succ, np.minimum(upto, tg), upto))
        uc = np.asarray(upto) // f
        ar = np.arange(n)
        xe, ye, vye = x[:, ar, uc], y[:, ar, uc], out["vy"][:, ar, uc]
        tend = self._tt(self.time_to_go(xe, ye, keys, legs_e, vye)[0]).mean(0)
        return {"veto": veto, "veto_frame": vf, "soft": soft, "succ": succ, "tg": tg, "tend": tend, "sweep": sw}

    def _long_safe(self, row, worlds, frames, dt_ms, ck, leg, keys, pen_first, scale):
        """LONG-SAFE (member 0): stage 1 (every library move on its own frames) and stage 2 (from the best
        B1 qualifying first moves, the second moves padded to H_LONG, rolled from the start: the cost is
        the number of steps, not the batch) with member 0. Only candidates with no veto and soft risk <=
        LONG_RISK qualify; the best estimate (successes first) is returned as (segments, actions of H_LONG
        frames, estimate, soft risk), or None when none qualifies; and the counts."""
        f = self.f
        HL = -(-H_LONG // f) * f
        mem = slice(MEMBERS["long-safe"], MEMBERS["long-safe"] + 1)
        n1 = len(self.lib)
        out1 = imagine(self.model, mem, self.geom, np.repeat(row, n1), self.lib_actions, dt_ms, worlds, **ck)
        j1 = self._judge(out1, frames, self.lib_end, leg, keys)
        est1 = self.lib_end + j1["tend"] + scale * j1["soft"] + pen_first
        ok1 = ~j1["veto"] & (j1["soft"] <= LONG_RISK)
        key1 = np.where(j1["succ"], SUCCESS_SCORE - j1["tg"], -est1)
        firsts = [int(i) for i in np.argsort(-key1, kind="stable") if ok1[i]][:self.B1]
        cnt = {"stage1_qualified": int(ok1.sum()), "candidates": 0, "qualified": 0}
        if not firsts:
            return None, cnt
        seconds = [BY_NAME[nm] for nm in CORE_SECOND]
        for i in firsts:
            if len(seconds) >= self.branch:
                break
            if self.lib[i].name not in CORE_SECOND:
                seconds.append(self.lib[i])
        cands, acts = [], []
        for i in firsts:
            L0, L = int(self.lib_len[i]), int(self.lib_end[i])
            first = self.lib[i].array if L == L0 else padded(self.lib[i].array, L)
            for sm in seconds[:self.branch]:
                L2 = min(HL, L + len(sm))
                cands.append(([(self.lib[i].name, 0, L0)] + ([("pad", L0, L)] if L > L0 else []) + [(sm.name, L, L2)]
                              + ([("pad", L2, HL)] if L2 < HL else []), i))
                acts.append(padded(np.concatenate([first, sm.array]), HL))
        acts = np.stack(acts)
        out2 = imagine(self.model, mem, self.geom, np.repeat(row, len(acts)), acts, dt_ms, worlds, **ck)
        up = np.full(len(acts), HL, np.int64)
        j2 = self._judge(out2, frames, up, leg, keys)
        est2 = HL + j2["tend"] + scale * j2["soft"] + pen_first[[i for _, i in cands]]
        ok2 = ~j2["veto"] & (j2["soft"] <= LONG_RISK)
        cnt.update(candidates=len(acts), qualified=int(ok2.sum()))
        if not ok2.any():
            return None, cnt
        key2 = np.where(j2["succ"], SUCCESS_SCORE - j2["tg"], -est2)
        b = int(np.argmax(np.where(ok2, key2, -np.inf)))
        return (cands[b][0], acts[b], float(est2[b]), float(j2["soft"][b])), cnt

    def _danger_near(self, agent: int, frame: int, x0: float, y0: float, frames) -> bool:
        """REFLEX's trigger besides the commit "danger" one: a deadly box (a kill shape, game.level.hazard_boxes)
        within REFLEX_NEAR units of the current plan's path (its own predicted path, shifted to now; without
        one, the agent's position) in the next REFLEX_WINDOW frames."""
        prev = self.prev.get(agent)
        pts = [[x0, y0]]
        if prev is not None and 0 <= frame - prev.frame < len(prev.path):
            k0 = frame - prev.frame
            pts += [list(q) for q in prev.path[k0 + 1:k0 + REFLEX_WINDOW + 1]]
        pts += [pts[-1]] * (REFLEX_WINDOW + 1 - len(pts))
        p = np.asarray(pts, np.float64)
        for k in range(1, REFLEX_WINDOW + 1):
            fb = frames[min(k, len(frames) - 1)]
            if fb is None or not len(fb):
                continue
            clear, _ = S.segment_clearance(p[k - 1:k, 0], p[k - 1:k, 1], p[k:k + 1, 0], p[k:k + 1, 1], fb)
            if clear.size and float(clear.min()) < REFLEX_NEAR:
                return True
        return False

    def _reflex(self, row, worlds, frames, dt_ms, ck, leg, keys, pen_first):
        """REFLEX (member 2): the library moves of at most REFLEX_MAX frames on their own frames; the least
        risk (a veto counts 1, else early_risk: frames 1..16 by the soft rule), then the best estimate.
        Returns (library index, its risk, its estimate)."""
        idx = np.nonzero(self.lib_len <= REFLEX_MAX)[0]
        mem = slice(MEMBERS["reflex"], MEMBERS["reflex"] + 1)
        acts = self.lib_actions[idx][:, :REFLEX_MAX]
        upto = self.lib_end[idx]
        out = imagine(self.model, mem, self.geom, np.repeat(row, len(idx)), acts, dt_ms, worlds, **ck)
        j = self._judge(out, frames, upto, leg, keys)
        risk = np.where(j["veto"], 1.0, early_risk(j["sweep"], upto))
        est = upto + j["tend"] + pen_first[idx]
        b = int(np.lexsort((est, risk, j["veto"]))[0])
        return int(idx[b]), float(risk[b]), float(est[b])

    def explore(self, row, frame: int, worlds: list, frames: list, dt_ms: float = 16.667, agent: int = 0) -> Plan:
        """EXPLORE (members 3 and 4): stage 1 only, every library move padded to H frames (last action held),
        rolled with members 3 and 4; among the plans without a veto, the one whose two members' end positions
        disagree most (the most informative for the model), ties by estimate (all vetoed: the latest veto).
        Returns a Plan with mode "explore" (no previous plan, memories or surprise tracking)."""
        t0 = time.perf_counter()
        f = self.f
        H = -(-self.H // f) * f
        row = np.asarray(row, ROW).reshape(1)
        x0, y0, keys = float(row["x"][0]), float(row["y"][0]), int(row["keys"][0])
        vy0 = np.array([float(row["vy"][0]) * dt_ms / 1000.0])
        t0a, sup0, g0, side0, leg = self.time_to_go(np.array([x0]), np.array([y0]), keys, vy=vy0)
        T0 = float(t0a[0])
        self._dead_t = T0 + DEAD_END_FRAMES if math.isfinite(T0) else DEAD_FAR
        if getattr(self, "_explore_acts", None) is None or self._explore_acts.shape[1] != H:
            self._explore_acts = np.stack([padded(m.array, H) for m in self.lib])
        acts = self._explore_acts
        n = len(acts)
        E_all = len(self.model.members)
        a, b = MEMBERS["explore"]
        mem = slice(min(a, E_all - 2), min(b, E_all - 1) + 1)
        ck = {"terrain": self.guard_terrain(keys), "tol": self.collide_tol}
        out = imagine(self.model, mem, self.geom, np.repeat(row, n), acts, dt_ms, worlds, **ck)
        up = np.full(n, H, np.int64)
        j = self._judge(out, frames, up, leg, keys)
        x, y = out["x"], out["y"]
        dis = np.hypot(x[0, :, -1] - x[-1, :, -1], y[0, :, -1] - y[-1, :, -1])
        est = H + j["tend"]
        if (~j["veto"]).any():
            c = int(np.lexsort((est, -dis, j["veto"]))[0])
        else:
            c = int(np.lexsort((est, -j["veto_frame"]))[0])
        Hc = H // f
        mx, my = x[:, c].mean(0), y[:, c].mean(0)
        if f > 1:
            mx, my = (np.interp(np.arange(H + 1), np.arange(Hc + 1) * f, v) for v in (mx, my))
        name = self.lib[c].name
        L = int(self.lib_len[c])
        ms = 1000 * (time.perf_counter() - t0)
        veto = None
        if j["veto"][c]:
            veto = {"frame": int(j["veto_frame"][c]), "id": None, "kind": "explore"}
        return Plan(
            frame=int(frame), agent=int(agent),
            state={"x": x0, "y": y0, "vx": float(row["vx"][0]), "vy": float(row["vy"][0]), "keys": keys,
                   "supported": bool(sup0[0])},
            chosen=[name], actions=[int(v) for v in acts[c]], pad=H - L, from_prev=False,
            path=[[float(p), float(q)] for p, q in zip(mx, my)], spread=[], landings=[], wall_jumps=[], nearest=[],
            veto=veto, soft_risk=float(j["soft"][c]), soft_frame=-1,
            score={"estimate": float(est[c]), "disagreement": float(dis[c]),
                   "t_start": T0 if math.isfinite(T0) else None},
            v_start=None, v_end=None, dead_end_members=0, leg=int(leg), end_supported=False, alternatives=[],
            predictions={}, death_model=0.0, timing={"explore": round(ms, 2), "total": round(ms, 2)},
            counts={"explore": {"candidates": n, "members": mem.stop - mem.start, "vetoed": int(j["veto"].sum())}},
            segments=[[name, 0, L]] + ([["pad", L, H]] if L < H else []), first_move_frames=L, mode="explore")

    def switch_rule(self, best: int, p: int, total3, succ3, tg3, saf, soft=None):
        """The switch rule (module docstring) between the shifted previous plan (final index p, -1 none) and
        the best other candidate. best: the best total. soft: the candidates' soft risk (default the safety
        check's). Returns (chosen index, Plan.switch record or None). Unit: frames (the totals are minus the
        estimated frames to the goal; successes: their touch frames)."""
        if p < 0:
            return best, None
        soft = saf.soft_risk if soft is None else soft
        others = np.array([c for c in range(len(total3)) if c != p], np.int64)
        if not len(others):
            return p, {"kept": True, "gain": None, "margin": None, "unit": "frames", "reason": "kept: no other candidate"}
        b = int(others[np.argmax(total3[others])])
        h = self.hysteresis
        both = bool(succ3[p] and succ3[b])
        gain = float(tg3[p] - tg3[b]) if both else float(total3[b] - total3[p])
        margin = float(SUCCESS_MARGIN) if both else h
        g = f"{gain:+.0f} frames"

        def rec(kept, why):
            return {"kept": kept, "gain": round(gain, 3), "margin": margin, "unit": "frames", "reason": why}
        if h <= 0:                                   # no rule: the best total (only described)
            return best, rec(best == p, f"kept: best other {g}" if best == p else f"switched: {g}")
        if best == p:
            return p, rec(True, f"kept: best other {g}")
        if saf.veto[p]:
            return b, rec(False, f"switched: previous plan vetoed ({saf.veto_id[p]} f{int(saf.veto_frame[p])})")
        if succ3[b] and not succ3[p]:
            return b, rec(False, "switched: reaches the princess, the previous plan does not")
        if not both and soft[p] - soft[b] > RISK_SWITCH:
            return b, rec(False, f"switched: previous plan's risk {soft[p]:.2f} vs {soft[b]:.2f}")
        if gain > margin:
            return b, rec(False, f"switched: {g}")
        return p, rec(True, f"kept: best other {g} < {margin:g}")

    @staticmethod
    def _touches(xs, ys, tg, keys, E, f: int = 1):
        """The chosen plan's goal touch {frame, x, y, members} (None unless a majority of members touch it)
        and key pickups [{frame, key, x, y}] (those a majority of members make), from its members' paths
        xs, ys (E, H+1), goal touch frames tg (E,) and per key leg (leg, frames (E,), key index (E,)). frame: the
        (lower) median of the members' touch frames, x / y: the mean of their touch points."""
        def at(t):
            m = np.nonzero(t > 0)[0]
            if 2 * len(m) <= E:
                return None
            ts = np.sort(t[m])
            return int(ts[(len(ts) - 1) // 2]), float(np.mean(xs[m, t[m] // f])), float(np.mean(ys[m, t[m] // f])), m
        g = at(tg)
        goal = None if g is None else {"frame": g[0], "x": g[1], "y": g[2], "members": int(len(g[3]))}
        out = []
        for j, t, w in keys:
            k = at(t)
            if k is None:
                break
            vals, cnt = np.unique(w[k[3]], return_counts=True)
            out.append({"frame": k[0], "key": int(vals[cnt.argmax()]), "x": k[1], "y": k[2]})
        return goal, out

    def _alternatives(self, final, best, saf, parts, total3, T0, cell0, scale, veto1, vf1, vk1, vid1, est1,
                      paths=None, goal=None, switch=None) -> list[dict]:
        """The top N_ALTERNATIVES rejected plans, each with the one reason that separates it from the chosen one
        (in frames): the final stage's next best by score (up to N_ALTERNATIVES - 2 of them first), then the
        stage-1 moves the safety check vetoed with the lowest estimate (up to 2), then more of the final stage.
        parts: the final candidates' frames {base (frames used + T at the end), risk, fail, death, est (the
        estimate), soft (soft risk), first (first move names)}.

        With `paths` (final mean x, y (n3, H+1); stage-1 member-0 x, y (n1, K1+1); stage-1 frames judged (n1,)),
        each alternative also carries its predicted path [[x, y], ...] (final: the members' mean over frames
        0..H; stage 1: frames 0..its move's length) and whether it was vetoed. "progress": T(start) minus its
        estimate (frames gained beyond the time it takes; None without T(start))."""
        names = lambda segs: [nm for nm, a, b in segs if nm != "pad"]
        est = parts["est"]
        gained = lambda e: None if not math.isfinite(T0) else float(T0 - e)

        def with_path(d, xs, ys, vetoed):
            if paths is not None:
                d["path"] = [[float(a), float(b)] for a, b in zip(xs, ys)]
                d["vetoed"] = bool(vetoed)
            return d

        succ, tg, n_goal, E = goal if goal is not None else (np.zeros(len(final), bool), None, None, None)
        kept_over = switch is not None and switch["kept"]

        def final_alt(c):
            if kept_over and total3[c] > total3[best]:          # better, but within the switch margin
                if succ[best] and succ[c]:
                    reason = f"reaches the princess {int(tg[best] - tg[c])} frames earlier"
                else:
                    reason = f"estimate {est[c]:.0f} vs {est[best]:.0f} frames"
                reason += f", within the switch margin: the committed plan is kept ({switch['reason']})"
            elif succ[best] and succ[c]:
                if tg[c] > tg[best]:
                    reason = f"reaches the princess later (f{tg[c]} vs f{tg[best]})"
                elif total3[c] == total3[best]:
                    reason = f"reaches the princess at f{tg[c]} too (a tie: the first one is kept)"
                else:
                    reason = (f"reaches the princess at f{tg[c]} too, with more risk or failures before it "
                              f"(risk {saf.soft_risk[c]:.2f} vs {saf.soft_risk[best]:.2f})")
            elif saf.veto[c]:
                reason = hit_reason(int(saf.veto_kind[c]), saf.veto_id[c], int(saf.veto_frame[c]))
            elif succ[best]:
                reason = "does not reach the princess" + (f" ({n_goal[c]} of {E} members do)" if n_goal[c] else "")
            else:
                gaps = {"time": parts["base"][c] - parts["base"][best], "risk": parts["risk"][c] - parts["risk"][best],
                        "fail": parts["fail"][c] - parts["fail"][best], "death": parts["death"][c] - parts["death"][best]}
                why = max(gaps, key=gaps.get)
                if why == "time" and math.isfinite(T0) and T0 - parts["t_end"][c] <= NO_PROGRESS:
                    reason = (f"no progress (T {T0:.0f} -> {parts['t_end'][c]:.0f} frames; estimate {est[c]:.0f} "
                              f"vs {est[best]:.0f})")
                elif why == "fail" and parts["fail"][c] > 0:
                    reason = self.memory.reason(cell0, parts["first"][c])
                elif why == "death" and parts["death"][c] > 0 and self.moves is not None:
                    reason = self.moves.reason(cell0, parts["first"][c], scale)
                elif why == "risk" and parts["comp_death"][c] > 0:
                    reason = f"every continuation of its completion dies (+{parts['risk'][c]:.0f} frames: a death)"
                elif why == "risk" and parts["risk"][c] > 0:
                    near = saf.nearest[c][0]["id"] if saf.nearest[c] else "?"
                    reason = (f"risk {parts['soft'][c]:.2f} near {near} at frame {int(saf.soft_frame[c])} "
                              f"(chosen {parts['soft'][best]:.2f}, +{parts['risk'][c]:.0f} frames)")
                else:
                    reason = f"estimate {est[c]:.0f} vs {est[best]:.0f} frames"
            d = {"moves": names(final[c][0]), "total": float(total3[c]), "estimate": float(est[c]),
                 "progress": gained(est[c]), "stage": "final", "reason": reason}
            return with_path(d, paths[0][c], paths[1][c], saf.veto[c]) if paths is not None else d

        def stage1_alt(i):
            d = {"moves": [self.lib[i].name], "total": None, "estimate": float(est1[i]), "progress": gained(est1[i]),
                 "stage": "stage 1", "reason": hit_reason(int(vk1[i]), vid1[i], int(vf1[i]))}
            if paths is None:
                return d
            n = int(paths[4][i]) + 1
            return with_path(d, paths[2][i][:n], paths[3][i][:n], True)

        fin = [int(c) for c in np.argsort(-total3, kind="stable") if c != best]
        vet = [int(i) for i in np.argsort(est1, kind="stable") if veto1[i]]
        n_fin = min(len(fin), max(N_ALTERNATIVES - min(2, len(vet)), 0))
        n_vet = min(len(vet), N_ALTERNATIVES - n_fin)
        return [final_alt(c) for c in fin[:n_fin]] + [stage1_alt(i) for i in vet[:n_vet]]
