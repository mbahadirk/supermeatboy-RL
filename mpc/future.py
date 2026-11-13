"""Where the level's moving parts will be over the next frames.

With the game's time step locked, hazards sit at the same place at the same wave frame in every wave
(measured: 0.0 units off on 1-13 and 1-14). So the first wave on a level is recorded, and later waves look
the future up in it ("recorded"), but only while the current positions keep matching the recording. A new
level starts a new recording: a wave cut by the level switch (no wave end) belongs to the old level and is
dropped. Otherwise (first wave, a mismatch, or frames past the recording) positions are extrapolated with their
current velocity ("extrapolated"), with an uncertainty radius that grows with time: the planner keeps that
much more distance, so a guessed future is never treated as known. A kill box whose own motion has repeated in
this wave (the last P frames equal the P before within MATCH_TOL, and it moved in them: _BoxRun.period) is
looked up P frames back instead, with REC_RADIUS: on recorded first-life waves this missed nothing where the
linear guess missed 24-67% at +8..+48 (runs/experiments/shot_future_v1); before a period is seen the linear guess
stays (the alternatives measured there either missed as often or needed 2.5-15 times the area).

Flying shots (realgame.js shots: [id, x, y, radius], not pieces, never looked up in the recording) are predicted
from their own flight in this wave (FutureWorld.shot_track, shot_future): a constant-acceleration fit to the
flight's last frames, with the margin the fit can be wrong by, and none from the frame its centre would surely be
inside a solid tile (the game removes it there). Measured on recorded flights (runs/experiments/shot_future_v1):
lava balls (4-4, 4-6) fall with a constant -0.15 units / frame^2, bullets (1-15, 1-16, 2-3, 3-1) fly straight at
6 units / frame; a slot's id is reused by its next flight.

Two views of each frame's hazards: the model inputs (wm.features.hazard_boxes: each piece's own box, as the
world model was trained) and the boxes that kill (game.level.hazard_boxes: the game's kill shapes, which move
inside a piece's animation while the piece stays put). The recording is trusted while both match.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from game.level import SHAPE_CIRCLE, box_ids, hazard_shapes
from wm.features import hazard_boxes
from wm.rollout import World

MATCH_TOL = 1.0          # game units: the recording still holds if every hazard is this close to it
STILL = 5.0              # units/s: slower than this counts as not moving (radius 0 when extrapolated)
GROW = 0.5               # extrapolated radius: 2 units + this x speed x time ahead
REC_RADIUS = 1.0         # recorded positions: this much margin
PERIOD_MIN = 20          # frames: the shortest period looked for (a box's own repetition in this wave)
LAUNCH_KEY = 2.0         # units: shots born within this of one point come from one launcher (1-15: exact spawns)
LAUNCH_P_MIN = 10        # frames: two births closer than this are not a launcher's period
LAUNCH_P_TOL = 1         # frames: a launcher silent for longer than its period + this has stopped firing
PERIOD_TRIES = 16        # periods verified per search at most (the shortest candidates first)

# Flying shots, per frame k ahead (frames: the time step is locked). Measured on 149 recorded flights in 4-4, 4-6,
# 3-1, 1-15, 1-16, 2-3 (positions recorded to 0.01 units; runs/experiments/shot_future_v1):
SHOT_FIT = 8             # frames of the current flight fitted (a quadratic per axis, least squares)
SHOT_BASE = 1.0          # units always added: the largest error not explained by rounding was 0.23 (a 4-4 lava
                         # ball's last frame before it vanished: its speed changed by 0.08 against the fit)
SHOT_NOISE = 0.01        # units: the least position error assumed (recorded rounding 0.005; fit error / sum |w(k)|
                         # never exceeded 0.0043 otherwise); margin = SHOT_BASE + max(this, residual) x sum |w(k)|
SHOT_MODEL_TOL = 0.05    # units: a larger fit residual is not constant acceleration (largest seen 0.0064)
SHOT_V_MAX = 10.0        # units / frame: fastest shot seen 9.67 (a falling 4-4 lava ball); bounds one-frame flights
SHOT_A_MAX = 0.2         # units / frame^2: largest |second difference| seen 0.17; bounds two-frame flights
SHOT_JUMP = 1.0          # units / frame^2: a larger second difference starts a new flight (a respawn in the slot)
SHOT_STOP_MARGIN = 10.0  # units (TILE / 2): a shot stops only where its centre is surely in a solid tile: margin
                         # at most this, and the centre and 8 points at the margin around it all solid
_RING = np.array([(0.0, 0.0), (1, 0), (-1, 0), (0, 1), (0, -1)]
                 + [(a * 0.5 ** 0.5, b * 0.5 ** 0.5) for a in (1, -1) for b in (1, -1)], np.float64)


class FutureWorld:
    def __init__(self):
        self.new_level()

    def new_level(self):
        self.ref: dict[int, tuple] = {}       # wave frame -> (hz raw, pf, sh) of the reference recording
        self.cur: dict[int, tuple] = {}       # a wave cut by the level switch belongs to the old level: dropped
        self.keep, self.fans = None, []
        self.terrain = None                   # the live tiles (plan.geometry.Terrain) shots stop in; None unknown
        self.launchers: dict[tuple, Launcher] = {}   # spawn point -> what fires there (Launcher)
        self.new_wave()

    def new_wave(self) -> dict:
        """The wave ended: it extends the recording when it is the level's first or matched it. Returns what this
        wave did: {frames: frames observed, recording: frames recorded before it (0: it ran extrapolated),
        trusted: it still matched at its end (None without a recording), mismatch: [frame, units off] of the
        first mismatch or None, extended: frames it added to the recording}."""
        cur = getattr(self, "cur", {})
        trusted, mismatch = getattr(self, "trusted", True), getattr(self, "mismatch", None)
        before, added = len(self.ref), 0
        if cur and (not self.ref or trusted):  # the first wave, or one that matched: extends the recording
            top = max(self.ref) if self.ref else -1
            for f, v in cur.items():
                if f > top:
                    self.ref[f] = v
                    added += 1
        info = {"frames": len(cur), "recording": before, "trusted": bool(trusted) if before else None,
                "mismatch": None if mismatch is None else [int(mismatch[0]), round(float(mismatch[1]), 1)],
                "extended": added}
        self.cur: dict[int, tuple] = {}
        self.runs: dict[str, _BoxRun] = {}    # kill box id -> its consecutive boxes up to now in this wave
        self.trusted = True                   # this wave still matches the recording
        self.shots_trusted = True             # its flying shots too (aimed shots would differ: then predicted)
        self.mismatch = None                  # (frame, units off) of the first mismatch
        self.flights: dict = {}               # shot id -> its current flight {key, birth, pts, last}
        for L in getattr(self, "launchers", {}).values():
            L.last = None                     # wave frames start again: the phase is seen anew in each wave
        self.last = None                      # (frame, hz, pf, sh, dt_ms) now
        self.prev = None                      # the frame before
        return info

    def observe(self, frame: int, hz: list, pf, sh, keep, fans, dt_ms: float, terrain=None):
        """This frame's hazards rows, platforms, flying shots, and the live tiles (a plan.geometry.Terrain, held by
        reference: its owner keeps it current; None: unknown, shots are then not stopped by walls)."""
        self.terrain = terrain
        if keep is not None:
            self.keep = keep
        if fans is not None:
            self.fans = fans
        rec = (hz or [], pf or [], sh or [])
        self.cur[frame] = rec
        ref = self.ref.get(frame)
        if ref is not None and self.trusted:
            off = _offset(ref[0], rec[0])
            if off > MATCH_TOL:
                self.trusted = False
                self.mismatch = (frame, off)
        if ref is not None and self.shots_trusted and _shot_offset(ref[2], rec[2]) > MATCH_TOL:
            self.shots_trusted = False
        self._shots_seen(frame, rec[2])
        kb, kept, ksh = _kill(rec[0], self.keep)
        ids = box_ids(kept)
        runs = self.runs
        for j, i in enumerate(ids):
            run = runs.get(i)
            if run is None or not run.add(frame, kb[j]):
                runs[i] = _BoxRun(frame, kb[j])
        self.kill_now = (frame, kb, kept, ids)
        self.kill_shape = dict(zip(ids, ksh.tolist()))   # each kill box's shape type (game.level SHAPE_*)
        self.periodic = None                  # this frame's periodic boxes (worlds), once
        self.prev, self.last = self.last, (frame, rec[0], rec[1], rec[2], dt_ms)

    def _shots_seen(self, frame: int, sh: list):
        """Flights and launchers from this frame's shots: a shot whose flight starts here (shot_track has only this
        frame) is a birth at its launcher (keyed by the spawn point, LAUNCH_KEY units); a flight that ends (its id
        gone, or a new flight in the slot) becomes the launcher's template when it was seen from its birth."""
        seen = set()
        for s in sh:
            sid = s[0]
            seen.add(sid)
            fl = self.flights.get(sid)
            if fl is None or fl["last"] != frame - 1 or len(self.shot_track(frame, sid)) == 1:
                if fl is not None:
                    self._flight_end(fl)
                key = _launch_key(s[1], s[2])
                fl = self.flights[sid] = {"key": key, "birth": frame, "pts": [], "last": frame,
                                          "first": len(self.cur) > 1 and (frame - 1) in self.cur}
                if fl["first"]:                       # seen from its first frame (not already flying at the start)
                    self.launchers.setdefault(key, Launcher()).born(frame, sid, float(s[3]))
            fl["pts"].append((float(s[1]), float(s[2])))
            fl["last"] = frame
        for sid in [i for i in self.flights if i not in seen]:
            self._flight_end(self.flights.pop(sid))

    def _flight_end(self, fl: dict):
        L = self.launchers.get(fl["key"])
        if L is not None and fl["first"] and len(fl["pts"]) >= 3:
            L.template = np.array(fl["pts"], np.float64)
            L.complete = True

    def launcher_shots(self, frame: int, H: int) -> list:
        """Shots not fired yet: each launcher with a period seen in this wave fires again at last + m x P; such a
        shot follows the launcher's template flight by age (its last whole flight, else the current one, then
        extrapolated with shot_future), with margin SHOT_BASE + its speed (one frame of timing). Returns
        [(id, radius, centre (H, 2), margin (H,), first k, last k)] for steps 1..H (k: frames ahead)."""
        out = []
        for key, L in self.launchers.items():
            if L.P is None or L.last is None or L.radius is None or frame - L.last > L.P + LAUNCH_P_TOL:
                continue                              # no period yet, or it stopped firing
            tpl, complete = L.template, L.complete
            if tpl is None:
                fl = next((f for f in self.flights.values() if f["key"] == key and f["first"]), None)
                if fl is None or len(fl["pts"]) < 3:
                    continue
                tpl, complete = np.array(fl["pts"], np.float64), False
            n = len(tpl)
            step = np.r_[np.hypot(*np.diff(tpl, axis=0).T), 0.0]
            step[-1] = step[-2] if n > 1 else 0.0
            if complete:
                path, marg, life = tpl, SHOT_BASE + step, n
            else:                                     # the flight so far, then its own extrapolation
                c, m, alive = shot_future(tpl[-SHOT_FIT:], H, self.terrain)
                path = np.concatenate([tpl, c])
                marg = np.concatenate([SHOT_BASE + step, m + step[-1]])
                life = n + alive
            b = L.last + L.P
            while b - frame <= H:
                if b > frame:
                    age = frame + np.arange(1, H + 1) - b          # the shot's age at steps 1..H
                    ok = (age >= 0) & (age < life)
                    if ok.any():
                        centre = np.zeros((H, 2))
                        margin = np.zeros(H)
                        centre[ok], margin[ok] = path[age[ok]], marg[age[ok]]
                        k = np.nonzero(ok)[0]
                        out.append((f"{L.sid}+{b}", L.radius, centre, margin, int(k[0]) + 1, int(k[-1]) + 1))
                b += L.P
        return out

    def shot_track(self, frame: int, sid) -> np.ndarray:
        """Shot `sid`'s positions in its current flight (n, 2), oldest first, ending at `frame`: at most SHOT_FIT
        consecutive frames of this wave, cut where the slot was empty or jumped (a new flight with a reused id:
        a step over SHOT_V_MAX, or a second difference over SHOT_JUMP)."""
        out = []
        for f in range(frame, frame - SHOT_FIT, -1):
            rec = self.cur.get(f)
            q = next((s for s in rec[2] if s[0] == sid), None) if rec is not None else None
            if q is None:
                break
            p = np.array((float(q[1]), float(q[2])))
            if out and np.hypot(*(p - out[-1])) > SHOT_V_MAX:
                break
            out.append(p)
            if len(out) >= 3 and np.max(np.abs(out[-3] - 2 * out[-2] + out[-1])) > SHOT_JUMP:
                # the jump is the longer of the two steps: out[-1] belongs to the slot's previous flight, or (only
                # decidable when these are the first three) out[-2] already did
                if len(out) == 3 and np.hypot(*(out[0] - out[1])) > np.hypot(*(out[1] - out[2])):
                    del out[1:]
                else:
                    out.pop()
                break
        return np.array(out[::-1], np.float64).reshape(-1, 2)

    def worlds(self, frame: int, H: int, dt_ms: float, deadly_only: bool = False):
        """World (model inputs) for steps 0..H-1 (frames frame..frame+H-1), the boxes that kill at steps 1..H
        as (boxes (M, 5), extra radius (M,), ids (M,): game.level.box_ids, then "shot<id>" of the shots still
        flying then: shot_future), and how many steps came from the recording. deadly_only: the boxes that kill
        only (worlds None; the same boxes): the per-frame danger check needs no model inputs."""
        if self.last is None or self.last[0] != frame:
            raise RuntimeError("observe() this frame before planning")
        _, hz0, pf0, sh0, _ = self.last
        boxes0 = np.zeros((0, 5)) if deadly_only else hazard_boxes(hz0, self.keep)
        _, kill0, kept0, ids0 = self.kill_now
        vel = np.zeros((len(boxes0), 2))
        kvel = np.zeros((len(kill0), 2))
        pf_vel = np.zeros((len(pf0), 2))
        if self.prev is not None and self.prev[0] == frame - 1:
            dt_s = max(1e-3, self.prev[4] / 1000.0)
            b1 = boxes0 if deadly_only else hazard_boxes(self.prev[1], self.keep)
            if len(b1) == len(boxes0):
                vel = (boxes0[:, :2] - b1[:, :2]) / dt_s
            k1, kept1, _ = _kill(self.prev[1], self.keep)
            if kept1 == kept0:                # the same shapes of the same pieces: their own motion
                kvel = (kill0[:, :2] - k1[:, :2]) / dt_s
            if len(self.prev[2]) == len(pf0) and len(pf0):
                pf_vel = (np.array(pf0, float)[:, :2] - np.array(self.prev[2], float)[:, :2]) / dt_s
        dt_s = dt_ms / 1000.0
        kspeed = np.hypot(kvel[:, 0], kvel[:, 1])
        if self.periodic is None:
            self.periodic = []
            for j, i in enumerate(ids0):
                run = self.runs[i]
                P = run.period() if run.moved else None
                if P is not None:
                    self.periodic.append((j, run, P))
        per_j = np.array([j for j, _, _ in self.periodic], np.int64)
        per_boxes = np.stack([run.ahead(H, P) for _, run, P in self.periodic], 1) if len(per_j) else None
        use_rec = self.trusted and bool(self.ref)
        recorded = 0
        hz_k, pf_k, kill_k = [], [], []
        rec_shots = [None] * (H + 1)                  # recorded flying shots per step (trusted recording only)
        for k in range(H + 1):
            rec = self.ref.get(frame + k) if use_rec else None
            if k and rec is not None and self.shots_trusted:
                rec_shots[k] = rec[2]
            if k == 0:
                b, p = boxes0, pf0
                kb = (kill0, np.zeros(len(kill0)), ids0)
            elif rec is not None:
                b, p = (None, None) if deadly_only else (hazard_boxes(rec[0], self.keep), rec[1])
                kr, kept_r, _ = _kill(rec[0], self.keep)
                kb = (kr, np.full(len(kr), REC_RADIUS), box_ids(kept_r))
                recorded += 1
            else:
                use_rec = False               # once off the recording, stay off it for this plan
                t = k * dt_s
                b = boxes0 + np.c_[vel * t, np.zeros((len(boxes0), 3))] if not deadly_only else None
                kr = kill0.copy()
                kr[:, :2] += kvel * t
                rad = np.where(kspeed > STILL, 2.0 + GROW * kspeed * t, 0.0)
                if per_boxes is not None:     # their own repetition in this wave: looked up P frames back
                    kr[per_j] = per_boxes[k - 1]
                    rad[per_j] = REC_RADIUS
                kb = (kr, rad, ids0)
                p = None
                if not deadly_only:
                    p = [list(q) for q in pf0]
                    for j, q in enumerate(p):
                        q[0], q[1] = q[0] + pf_vel[j, 0] * t, q[1] + pf_vel[j, 1] * t
            hz_k.append(b)
            pf_k.append(p)
            kill_k.append(kb)
        # flying shots: only kept away from (the world model was not trained with them). Recorded (trusted, the
        # game repeats them) where the recording is used; otherwise each from its own flight, and the shots its
        # launcher will fire (launcher_shots: none before a launcher's period has been seen)
        shots = [(sh[0], sh[3], *shot_future(self.shot_track(frame, sh[0]), H, self.terrain)) for sh in sh0]
        launch = self.launcher_shots(frame, H)
        worlds = None if deadly_only else [
            World(hz_k[k], pf_k[k] or None, (pf_k[k - 1] if k else (self.prev[2] if self.prev else None)) or None,
                  dt_s, list(self.fans)) for k in range(H)]
        check = []
        for k in range(1, H + 1):
            b, r, ids = kill_k[k]
            if rec_shots[k] is not None:
                live = [(s[0], float(s[3]), (float(s[1]), float(s[2])), REC_RADIUS) for s in rec_shots[k]]
            else:
                live = [(sid, rad, c[k - 1], m[k - 1]) for sid, rad, c, m, alive in shots if k <= alive]
                live += [(lid, rad, c[k - 1], m[k - 1]) for lid, rad, c, m, k0, k1 in launch if k0 <= k <= k1]
            shape = np.array([self.kill_shape.get(i, 0) for i in ids], np.int8)
            if live:
                sb = np.array([(c[0], c[1], rad, rad, 0.0) for _, rad, c, _ in live])
                b, r = np.concatenate([b, sb]), np.concatenate([r, [m for *_, m in live]])
                ids = list(ids) + [f"shot{sid}" for sid, *_ in live]
                shape = np.concatenate([shape, np.full(len(live), SHAPE_CIRCLE, np.int8)])
            check.append((b, r, list(ids), shape))
        return worlds, check, recorded


class _BoxRun:
    """One kill box's consecutive frames in this wave (its boxes (x, y, half w, half h, angle)), and its own period:
    the shortest P >= PERIOD_MIN with the last P frames' centres within MATCH_TOL of the P frames before and
    spread over more than 2 MATCH_TOL (it moved: a box at rest repeats any P). Kept while every new frame still
    matches the one P before; searched again (at most once per frame) otherwise."""

    def __init__(self, frame: int, box):
        self.start, self.n = frame, 1
        self.arr = np.empty((64, 5), np.float64)
        self.arr[0] = box
        self.x0, self.y0 = float(self.arr[0, 0]), float(self.arr[0, 1])
        self.moved = False                    # ever more than MATCH_TOL from its first centre
        self.P, self.searched = None, -1

    @property
    def end(self) -> int:
        return self.start + self.n - 1

    def add(self, frame: int, box) -> bool:
        """Append the box of the next frame (False: not the next frame, the caller starts a new run)."""
        if frame != self.start + self.n:
            return False
        if self.n == len(self.arr):
            self.arr = np.concatenate([self.arr, np.empty_like(self.arr)])
        b = self.arr[self.n]
        b[:] = box
        self.n += 1
        x, y = b[0], b[1]
        if not self.moved and (x - self.x0) ** 2 + (y - self.y0) ** 2 > MATCH_TOL ** 2:
            self.moved = True
        if self.P is not None:
            q = self.arr[self.n - 1 - self.P]
            if (x - q[0]) ** 2 + (y - q[1]) ** 2 > MATCH_TOL ** 2:
                self.P = None
        return True

    def period(self):
        if self.P is not None or self.searched == self.n or not self.moved:
            return self.P
        self.searched = L = self.n
        if L < 2 * PERIOD_MIN:
            return None
        c = self.arr[L - 1::-1, :2]                                      # newest first
        P = np.arange(PERIOD_MIN, L // 2 + 1)
        spread = np.hypot(*(np.maximum.accumulate(c, 0) - np.minimum.accumulate(c, 0)).T)  # over c[:j + 1]
        cand = P[(np.hypot(*(c[P] - c[0]).T) <= MATCH_TOL) & (spread[P - 1] > 2 * MATCH_TOL)]
        for p in cand[:PERIOD_TRIES]:
            if np.max(np.hypot(*(c[:p] - c[p:2 * p]).T)) <= MATCH_TOL:
                self.P = int(p)
                break
        return self.P

    def ahead(self, H: int, P: int) -> np.ndarray:
        """(H, 5): the boxes 1..H frames after the run's end by the period, each the one seen P x ceil(k / P)
        frames before it."""
        k = np.arange(1, H + 1)
        return self.arr[self.n - 1 + k - P * (-(-k // P))]


@lru_cache(maxsize=256)
class Launcher:
    """What fires shots at one spawn point (FutureWorld._shots_seen): its period P (frames between two births seen
    in one wave; kept for the level), the last birth in this wave, the shot's id and radius, and its template
    flight (positions by age from the birth: the last whole flight, complete=True, the shot gone after it)."""

    def __init__(self):
        self.P: int | None = None
        self.last: int | None = None
        self.sid = None
        self.radius: float | None = None
        self.template: np.ndarray | None = None
        self.complete = False

    def born(self, frame: int, sid, radius: float):
        if self.last is not None and frame - self.last >= LAUNCH_P_MIN:
            self.P = int(frame - self.last)
        self.last, self.sid, self.radius = int(frame), sid, float(radius)


def _launch_key(x: float, y: float) -> tuple:
    return int(round(float(x) / LAUNCH_KEY)), int(round(float(y) / LAUNCH_KEY))


def _shot_offset(a: list, b: list) -> float:
    """How far two frames' flying shots are apart (inf when their ids differ)."""
    if len(a) != len(b):
        return float("inf")
    pb = {s[0]: s for s in b}
    off = 0.0
    for s in a:
        q = pb.get(s[0])
        if q is None:
            return float("inf")
        off = max(off, math.hypot(float(s[1]) - float(q[1]), float(s[2]) - float(q[2])))
    return off


def _fit_weights(n: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares quadratic through n positions at frames -n+1..0: (H, n) weights giving its value at frames
    1..H, and the (n, n) matrix giving its value at the fitted frames (for the residual). Cached: read-only."""
    A = np.vander(np.arange(-n + 1, 1, dtype=np.float64), 3, increasing=True)
    P = np.linalg.pinv(A)
    W, fitted = np.vander(np.arange(1, H + 1, dtype=np.float64), 3, increasing=True) @ P, A @ P
    W.flags.writeable = fitted.flags.writeable = False
    return W, fitted


def shot_future(track: np.ndarray, H: int, terrain=None):
    """A flying shot's centre (H, 2) and margin (H,) at frames 1..H ahead, from its flight so far (track (n, 2),
    oldest first, n >= 1: FutureWorld.shot_track), and how many of those frames it still flies (shot_alive).

    n >= 3 and the flight fits constant acceleration (residual <= SHOT_MODEL_TOL): the fitted quadratic, margin
    SHOT_BASE + max(SHOT_NOISE, residual) x sum |w(k)| (the most a position error of that size moves the fit).
    Otherwise: velocity from the last two frames with |acceleration| <= SHOT_A_MAX, margin SHOT_BASE + SHOT_A_MAX
    k (k + 1) / 2 + 2 SHOT_NOISE k; a shot seen once stays where it is with margin SHOT_BASE + SHOT_V_MAX k."""
    track = np.asarray(track, np.float64).reshape(-1, 2)
    n = len(track)
    k = np.arange(1, H + 1, dtype=np.float64)
    if n >= 3:
        W, fitted = _fit_weights(n, H)
        resid = float(np.max(np.abs(fitted @ track - track)))
        if resid <= SHOT_MODEL_TOL:
            centre, margin = W @ track, SHOT_BASE + max(SHOT_NOISE, resid) * np.abs(W).sum(1)
            return centre, margin, shot_alive(centre, margin, terrain)
    if n >= 2:
        centre = track[-1] + (track[-1] - track[-2]) * k[:, None]
        margin = SHOT_BASE + 0.5 * SHOT_A_MAX * k * (k + 1) + 2 * SHOT_NOISE * k
    else:
        centre = np.repeat(track[-1:], H, 0)
        margin = SHOT_BASE + SHOT_V_MAX * k
    return centre, margin, shot_alive(centre, margin, terrain)


def shot_alive(centre: np.ndarray, margin: np.ndarray, terrain) -> int:
    """How many of the predicted frames 1..H a shot still flies: those before the first whose centre is surely in
    a solid tile (margin <= SHOT_STOP_MARGIN and the centre and 8 points at the margin around it solid in
    terrain.point_solid). Measured: a flying shot's centre was never in a solid tile, and bullets and 4-4 lava
    balls were gone by the frame their centre would have entered one; 4-6 lava balls vanish in the lava (a
    hazard, not a tile) and are not stopped. No terrain: all H frames."""
    H = len(margin)
    if terrain is None or not H:
        return H
    idx = np.nonzero(margin <= SHOT_STOP_MARGIN)[0]
    if not len(idx):
        return H
    pts = centre[idx, None, :] + margin[idx, None, None] * _RING[None, :, :]        # (m, 9, 2)
    solid = np.asarray(terrain.point_solid(pts[..., 0], pts[..., 1]), bool).all(1)
    return int(idx[np.argmax(solid)]) if solid.any() else H


def _kill(hz: list, keep):
    """The shapes that kill (game.level.hazard_shapes: the game's own geometry) as an (M, 5) array, their pieces'
    indices and their shape types (M,)."""
    b, kept, shape = hazard_shapes(hz, keep)
    return np.asarray(b, np.float64).reshape(-1, 5), kept, np.asarray(shape, np.int8)


def _offset(a: list, b: list) -> float:
    """How far two frames' hazards are apart: the pieces' positions and their kill shapes' centres (inf when
    the pieces or their shapes differ in number)."""
    if len(a) != len(b):
        return float("inf")
    if not a:
        return 0.0
    pa, pb = np.array([h[:2] for h in a], float), np.array([h[:2] for h in b], float)
    off = float(np.max(np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1])))
    (ka, ia, _), (kb, ib, _) = _kill(a, None), _kill(b, None)
    if ia != ib:
        return float("inf")
    if len(ka):
        off = max(off, float(np.max(np.hypot(ka[:, 0] - kb[:, 0], ka[:, 1] - kb[:, 1]))))
    return off
