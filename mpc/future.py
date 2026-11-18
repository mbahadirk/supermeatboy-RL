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

Homing laser eyes (plan/lasers.py, set_lasers: an EyeTracker on levels that have them) aim at the agent, so their
recorded or extrapolated beam segments are wrong for any other path: their rows' kill shapes are left out of the
deadly boxes and of the recording's match (_offset), and beam_context gives the decision's eye state instead (the
planner rolls each trajectory's own beams). Plain lasers stay as recorded. Without a tracker nothing changes.

Each recorded wave frame is worked out once per wave (FRAME_CACHE, _RecFrame): a sliding window asks for the same
wave frame about 48 times (the per-frame danger check) and parsing its rows was most of the planner's per-frame
time (runs/online_check04, 3-2: 12.8 ms per danger check, 37 ms per decision's full window). What depends on the
asking frame is never kept: the step-0 and step-1 World (the live platforms), extrapolated frames (their guess
and radius grow from the current frame), flying shots predicted from the current flight, and the kill shapes'
types (looked up in the asking frame's kill_shape: a kept check is reused only while that is unchanged).
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
FRAME_CACHE = True       # keep each recorded wave frame's boxes, World and check for the wave (_RecFrame); False:
                         # recompute them on every call (tests/test_future_cache.py compares the two)

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
        self._keep_seen = None                # tuple(keep) the kept recorded frames were parsed with
        self.terrain = None                   # the live tiles (plan.geometry.Terrain) shots stop in; None unknown
        self.launchers: dict[tuple, Launcher] = {}   # spawn point -> what fires there (Launcher)
        self.lasers = None                    # plan.lasers.EyeTracker of a level with homing eyes (set_lasers)
        self._skip = frozenset()              # hz rows whose kill shapes are not deadly boxes (homing beams)
        self.new_wave()

    def set_lasers(self, tracker):
        """The level's homing eyes (plan.lasers.EyeTracker, or None: none). Set after new_level()."""
        self.lasers = tracker
        self._skip = frozenset()
        self._rec = {}

    def beam_context(self, frame: int, H: int):
        """The homing eyes at this observed frame for a decision of H frames (plan.lasers.Beams), or None."""
        return None if self.lasers is None else self.lasers.context(frame, H)

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
        for fl in getattr(self, "flights", {}).values():   # flights cut by the wave's end: a partial template
            L = getattr(self, "launchers", {}).get(fl["key"])   # (extrapolated past its end) until a whole one
            if L is not None and fl["first"] and len(fl["pts"]) >= 3 and not L.complete and                     (L.template is None or len(fl["pts"]) > len(L.template)):
                L.template = np.array(fl["pts"], np.float64)
        self.cur: dict[int, tuple] = {}
        self.runs: dict[str, _BoxRun] = {}    # kill box id -> its consecutive boxes up to now in this wave
        self.trusted = True                   # this wave still matches the recording
        self.shots_trusted = True             # its flying shots too (aimed shots would differ: then predicted)
        self.mismatch = None                  # (frame, units off) of the first mismatch
        self.flights: dict = {}               # shot id -> its current flight {key, birth, pts, last}
        for L in getattr(self, "launchers", {}).values():
            L.last = None                     # wave frames start again: the phase is seen anew in each wave
        if getattr(self, "lasers", None) is not None:
            self.lasers.new_wave()
        self.last = None                      # (frame, hz, pf, sh, dt_ms) now
        self.prev = None                      # the frame before
        self._rec: dict[int, _RecFrame] = {}  # wave frame -> what the recording gives every window asking for it
        self._pruned = None                   # the asking frame _rec was last cut to (worlds)
        self._now: dict = {}                  # this frame's values shared by its calls (reset by observe)
        self._kill_prev = None                # the previous frame's (kill boxes, kept) when keep is unchanged
        return info

    def observe(self, frame: int, hz: list, pf, sh, keep, fans, dt_ms: float, terrain=None, lz=None, target=None):
        """This frame's hazards rows, platforms, flying shots, and the live tiles (a plan.geometry.Terrain, held by
        reference: its owner keeps it current; None: unknown, shots are then not stopped by walls). lz: realgame.js's
        laser states; target: the leader's position (the homing eyes' target), both for set_lasers' tracker."""
        self.terrain = terrain
        same_keep = True
        if self.lasers is not None:
            self.lasers.observe(frame, hz or [], lz, target)
            skip = self.lasers.beam_rows()
            if skip != self._skip:            # the kept frames were parsed with the other rows
                self._skip = skip
                self._rec = {}
                same_keep = False
        if keep is not None:
            self.keep = keep
            seen = tuple(keep)
            if seen != self._keep_seen:       # the pieces kept changed: the kept frames were parsed without it
                same_keep = False
                self._keep_seen = seen
                self._rec = {}
        if fans is not None:
            self.fans = fans
        rec = (hz or [], pf or [], sh or [])
        self.cur[frame] = rec
        ref = self.ref.get(frame)
        if ref is not None and self.trusted:
            off = _offset(ref[0], rec[0], self._skip)
            if off > MATCH_TOL:
                self.trusted = False
                self.mismatch = (frame, off)
                self._rec = {}                # the recording is not used again in this wave
        if ref is not None and self.shots_trusted and _shot_offset(ref[2], rec[2]) > MATCH_TOL:
            self.shots_trusted = False
            for e in self._rec.values():      # their kept checks hold the recorded shots
                e.check = e.derived = None
        self._shots_seen(frame, rec[2])
        if same_keep and self.last is not None:   # the previous frame's kill shapes, parsed as worlds() would now
            self._kill_prev = self.kill_now[1:3]
        else:
            self._kill_prev = None
        self._now = {}
        kb, kept, ksh = _kill(rec[0], self.keep, self._skip)
        ids = box_ids(kept)
        # the previous observed frame's kill boxes, comparable with this frame's by id (same pieces kept, same
        # number of pieces: box_ids are their indices) for hazard_motion
        prev_k = getattr(self, "kill_now", None)
        self._kill_cmp = (prev_k[1], prev_k[3]) if (same_keep and self.last is not None and prev_k is not None
                                                     and len(self.last[1]) == len(rec[0])) else None
        runs = self.runs
        for j, i in enumerate(ids):
            run = runs.get(i)
            if run is None or not run.add(frame, kb[j]):
                runs[i] = _BoxRun(frame, kb[j])
        self.kill_now = (frame, kb, kept, ids)
        shapes = dict(zip(ids, ksh.tolist()))  # each kill box's shape type (game.level SHAPE_*)
        if shapes != getattr(self, "kill_shape", None):
            self.kill_shape = shapes          # kept when equal: its identity tells the kept checks it is unchanged
        self.periodic = None                  # this frame's periodic boxes (periods), once
        self.prev, self.last = self.last, (frame, rec[0], rec[1], rec[2], dt_ms)

    def hazard_motion(self) -> tuple[float, str | None]:
        """How far a kill box moved from the previous observed frame to this one (units, its centre), matched by
        id, and that box's id: the largest such move; (0.0, None) when no box is comparable (the first frame of
        a wave, the pieces kept or their number changed). agent.py --async-wait auto: more than 0.5 units is a
        moving hazard."""
        cmp_ = getattr(self, "_kill_cmp", None)
        if cmp_ is None:
            return 0.0, None
        kb0, ids0 = cmp_
        _, kb, _, ids = self.kill_now
        if not len(kb) or not len(kb0):
            return 0.0, None
        if ids == ids0:
            d = np.hypot(kb[:, 0] - kb0[:, 0], kb[:, 1] - kb0[:, 1])
            k = int(np.argmax(d))
            return float(d[k]), ids[k]
        at = {b: n for n, b in enumerate(ids0)}
        pairs = [(n, at[b]) for n, b in enumerate(ids) if b in at]
        if not pairs:
            return 0.0, None
        a, b = np.array(pairs).T
        d = np.hypot(kb[a, 0] - kb0[b, 0], kb[a, 1] - kb0[b, 1])
        k = int(np.argmax(d))
        return float(d[k]), ids[int(a[k])]

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
            flight = self._launcher_flight(key, L, H)
            if flight is None:
                continue
            path, marg, life = flight
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

    def _launcher_flight(self, key, L: "Launcher", H: int):
        """A launcher's flight by age from a birth: (positions (life, 2), margins (life,), life frames), from its
        template (its last whole flight: margin SHOT_BASE + the speed, one frame of timing), else a flight of it
        seen from its birth with at least 3 frames so far, then that flight's own extrapolation; None unknown."""
        tpl, complete = L.template, L.complete
        if tpl is None:
            fl = next((f for f in self.flights.values() if f["key"] == key and f["first"] and len(f["pts"]) >= 3), None)
            if fl is None:
                return None
            tpl, complete = np.array(fl["pts"], np.float64), False
        n = len(tpl)
        step = np.r_[np.hypot(*np.diff(tpl, axis=0).T), 0.0]
        step[-1] = step[-2] if n > 1 else 0.0
        if complete:
            return tpl, SHOT_BASE + step, n
        c, m, alive = shot_future(tpl[-SHOT_FIT:], H, self.terrain)     # the flight so far, then extrapolated
        return np.concatenate([tpl, c]), np.concatenate([SHOT_BASE + step, m + step[-1]]), n + alive

    def flying_shot(self, frame: int, sid, H: int):
        """One flying shot's (centre (H, 2), margin (H,), frames it still flies) at frames 1..H: shot_future of its
        own flight; a shot born less than 3 frames ago from a launcher whose flight is known (_launcher_flight)
        follows that flight by age instead. (Its own 1-2 frames only bound it by SHOT_V_MAX k in every direction,
        through walls: 3-1's ceiling shot, born every 119 frames 144 units above the floor behind a solid ceiling,
        looked 160 units wide at +16 and sent the planner into reflex waits and early jumps at each birth.)"""
        track = self.shot_track(frame, sid)
        fl = self.flights.get(sid)
        if len(track) < 3 and fl is not None and fl["first"] and fl["birth"] == frame - len(track) + 1:
            L = self.launchers.get(fl["key"])
            flight = self._launcher_flight(fl["key"], L, H) if L is not None else None
            if flight is not None and len(flight[0]) > len(track) - 1:
                path, marg, life = flight
                a0 = len(track) - 1                       # its age now
                if np.hypot(*(path[a0] - track[-1])) <= LAUNCH_KEY + SHOT_BASE:   # it is on that flight so far
                    age = np.minimum(a0 + np.arange(1, H + 1), life - 1)
                    return path[age], marg[age], int(np.clip(life - 1 - a0, 0, H))
        return shot_future(track, H, self.terrain)

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

    def periods(self) -> list:
        """This frame's kill boxes repeating with their own period [(index in kill_now, _BoxRun, P)], searched once
        per frame. The search keeps state (_BoxRun.period: the window last searched, the period kept while it
        holds), so it must run on the frames it always ran on: every frame worlds() is asked for, and the frames
        agent.py's telemetry skips (its frame message used to ask worlds() on every frame)."""
        if self.periodic is None:
            self.periodic = []
            for j, i in enumerate(self.kill_now[3]):
                run = self.runs[i]
                P = run.period() if run.moved else None
                if P is not None:
                    self.periodic.append((j, run, P))
        return self.periodic

    def _frame_value(self, key: str, make):
        """A value of this frame shared by its calls (FRAME_CACHE; observe resets them)."""
        if not FRAME_CACHE:
            return make()
        v = self._now.get(key)
        if v is None:
            v = self._now[key] = make()
        return v

    def _recorded(self, f: int, rec: tuple) -> "_RecFrame":
        """Recorded wave frame f's entry, kept for the wave (FRAME_CACHE). Only the frames of windows still to come
        are kept (worlds() drops those at or before the asking frame): at most the longest window asked (plan.planner
        future_frames, 101, plus agent.py's async lead), not a whole wave (about 1,200 frames)."""
        e = self._rec.get(f) if FRAME_CACHE else None
        if e is None:
            kr, kept_r, _ = _kill(rec[0], self.keep, self._skip)
            e = _RecFrame((_ro(kr), _ro(np.full(len(kr), REC_RADIUS)), box_ids(kept_r)))
            if FRAME_CACHE:
                self._rec[f] = e
        return e

    def _rec_boxes(self, e: "_RecFrame", rec: tuple) -> np.ndarray:
        """A recorded frame's model boxes (wm.features.hazard_boxes), once."""
        if e.hb is None:
            e.hb = _ro(hazard_boxes(rec[0], self.keep))
        return e.hb

    def _rec_shape(self, e: "_RecFrame") -> np.ndarray:
        """A recorded frame's kill shapes' types as the asking frame's kill_shape has them (0 for an id it does not
        have), kept while kill_shape is the same object (observe keeps it while its content is unchanged)."""
        if e.shape is None or e.shape_ks is not self.kill_shape:
            e.shape = _ro(np.array([self.kill_shape.get(i, 0) for i in e.kill[2]], np.int8))
            e.shape_ks = self.kill_shape
            e.check = e.derived = None
        return e.shape

    def _rec_check(self, e: "_RecFrame", shots: list) -> tuple:
        """A recorded frame's deadly check with these flying shots, kept with the frame: its recorded ones while they
        are trusted, else none (observe drops the kept checks when they stop being trusted; new_wave every entry)."""
        shape = self._rec_shape(e)
        if e.check is None:
            b, r, ids = e.kill
            live = [(s[0], float(s[3]), (float(s[1]), float(s[2])), REC_RADIUS) for s in shots]
            if live:
                sb = np.array([(c[0], c[1], rad, rad, 0.0) for _, rad, c, _ in live])
                b, r = _ro(np.concatenate([b, sb])), _ro(np.concatenate([r, [m for *_, m in live]]))
                ids = list(ids) + [f"shot{sid}" for sid, *_ in live]
                shape = _ro(np.concatenate([shape, np.full(len(live), SHAPE_CIRCLE, np.int8)]))
            e.check, e.derived = (b, r, list(ids), shape), None
        return e.check

    def derived(self, f: int, check: tuple, key, make, *args):
        """A consumer's value made from one of worlds()'s deadly checks (plan.planner.future_inputs: its
        FrameBoxes): make(*args), kept with wave frame f while `check` is that frame's kept check and `key`
        (everything else the value depends on, compared with ==) is unchanged; made anew for any other check
        (extrapolated frames, predicted shots, FRAME_CACHE off)."""
        e = self._rec.get(f)
        if e is None or e.check is not check:
            return make(*args)
        if e.derived is not None and e.derived[0] == key:
            return e.derived[1]
        v = make(*args)
        e.derived = (key, v)
        return v

    def worlds(self, frame: int, H: int, dt_ms: float, deadly_only: bool = False):
        """World (model inputs) for steps 0..H-1 (frames frame..frame+H-1), the boxes that kill at steps 1..H
        as (boxes (M, 5), extra radius (M,), ids (M,): game.level.box_ids, then "shot<id>" of the shots still
        flying then: shot_future), and how many steps came from the recording. deadly_only: the boxes that kill
        only (worlds None; the same boxes): the per-frame danger check needs no model inputs. Recorded frames'
        parts are shared between calls (FRAME_CACHE): their arrays are read-only."""
        if self.last is None or self.last[0] != frame:
            raise RuntimeError("observe() this frame before planning")
        if FRAME_CACHE and self._pruned != frame:  # windows only move on: frames up to this one are not asked again
            self._pruned = frame
            for f in [f for f in self._rec if f <= frame]:
                del self._rec[f]
        _, hz0, pf0, sh0, _ = self.last
        boxes0 = np.zeros((0, 5)) if deadly_only else \
            self._frame_value("boxes0", lambda: _ro(hazard_boxes(hz0, self.keep)))
        _, kill0, kept0, ids0 = self.kill_now
        vel = np.zeros((len(boxes0), 2))
        kvel = np.zeros((len(kill0), 2))
        pf_vel = np.zeros((len(pf0), 2))
        if self.prev is not None and self.prev[0] == frame - 1:
            dt_s = max(1e-3, self.prev[4] / 1000.0)
            b1 = boxes0 if deadly_only else \
                self._frame_value("b1", lambda: _ro(hazard_boxes(self.prev[1], self.keep)))
            if len(b1) == len(boxes0):
                vel = (boxes0[:, :2] - b1[:, :2]) / dt_s
            if FRAME_CACHE and self._kill_prev is not None:
                k1, kept1 = self._kill_prev   # observe kept them (the same keep)
            else:
                k1, kept1, _ = _kill(self.prev[1], self.keep, self._skip)
            if kept1 == kept0:                # the same shapes of the same pieces: their own motion
                kvel = (kill0[:, :2] - k1[:, :2]) / dt_s
            if len(self.prev[2]) == len(pf0) and len(pf0):
                pf_vel = (np.array(pf0, float)[:, :2] - np.array(self.prev[2], float)[:, :2]) / dt_s
        dt_s = dt_ms / 1000.0
        kspeed = np.hypot(kvel[:, 0], kvel[:, 1])
        periodic = self.periods()
        per_j = np.array([j for j, _, _ in periodic], np.int64)
        per_boxes = np.stack([run.ahead(H, P) for _, run, P in periodic], 1) if len(per_j) else None
        use_rec = self.trusted and bool(self.ref)
        recorded = 0
        hz_k, pf_k, kill_k = [], [], []
        ent = [None] * (H + 1)                        # the recorded steps' entries (_recorded)
        rec_shots = [None] * (H + 1)                  # recorded flying shots per step (trusted recording only)
        for k in range(H + 1):
            rec = self.ref.get(frame + k) if use_rec else None
            if k and rec is not None and self.shots_trusted:
                rec_shots[k] = rec[2]
            if k == 0:
                b, p = boxes0, pf0
                kb = (kill0, np.zeros(len(kill0)), ids0)
            elif rec is not None:
                e = ent[k] = self._recorded(frame + k, rec)
                b, p = (None, None) if deadly_only else (self._rec_boxes(e, rec), rec[1])
                kb = e.kill
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
        worlds = None
        if not deadly_only:
            fans = list(self.fans)
            worlds = []
            for k in range(H):
                e = ent[k] if k >= 2 else None    # steps 0 and 1 hold the live platforms (now, the frame before)
                w = e.world if e is not None else None
                if w is None or w.dt_prev_s != dt_s or w.fans != fans:
                    w = World(hz_k[k], pf_k[k] or None,
                              (pf_k[k - 1] if k else (self.prev[2] if self.prev else None)) or None, dt_s, list(fans))
                    if e is not None and FRAME_CACHE:
                        e.world = w           # a recorded step k >= 2: the platforms before it are recorded too
                worlds.append(w)
        # flying shots: only kept away from (the world model was not trained with them). Recorded (trusted, the
        # game repeats them) where the recording is used; otherwise each from its own flight, and the shots its
        # launcher will fire (launcher_shots: none before a launcher's period has been seen), predicted when a
        # step first needs them (they change nothing: the same whenever made)
        shots = launch = ids_x = None
        check = []
        for k in range(1, H + 1):
            e = ent[k]
            if e is not None and rec_shots[k] is not None:    # all from the recording, its shots too
                check.append(self._rec_check(e, rec_shots[k]))
                continue
            b, r, ids = kill_k[k]
            if shots is None:
                shots = [(sh[0], sh[3], *self.flying_shot(frame, sh[0], H)) for sh in sh0]
                launch = self.launcher_shots(frame, H)
            live = [(sid, rad, c[k - 1], m[k - 1]) for sid, rad, c, m, alive in shots if k <= alive]
            live += [(lid, rad, c[k - 1], m[k - 1]) for lid, rad, c, m, k0, k1 in launch if k0 <= k <= k1]
            if e is not None and not live:    # a recorded frame and no shot predicted then: its kept check
                check.append(self._rec_check(e, []))
                continue
            if e is not None:
                shape = self._rec_shape(e)
            else:                             # the pieces now (ids0): the same types at every such step
                shape = self._frame_value("shape0", lambda: _ro(np.array([self.kill_shape.get(i, 0) for i in ids0],
                                                                        np.int8)))
            if live:
                sb = np.array([(c[0], c[1], rad, rad, 0.0) for _, rad, c, _ in live])
                b, r = np.concatenate([b, sb]), np.concatenate([r, [m for *_, m in live]])
                ids = list(ids) + [f"shot{sid}" for sid, *_ in live]
                shape = np.concatenate([shape, np.full(len(live), SHAPE_CIRCLE, np.int8)])
            elif FRAME_CACHE:                 # one list for all such steps (plan.planner.future_inputs: kinds once)
                if ids_x is None:
                    ids_x = list(ids0)
                check.append((b, r, ids_x, shape))
                continue
            check.append((b, r, list(ids), shape))
        return worlds, check, recorded


class _RecFrame:
    """What one recorded wave frame gives every window that asks for it (FutureWorld._recorded): its kill boxes
    (game.level.hazard_shapes), their REC_RADIUS and ids; once asked for, the model's boxes (hb), its World as a
    step k >= 2 (the platforms before it are recorded too; dt and fans compared on use), its shapes' types under
    the kill_shape `shape_ks`, its deadly check with the recorded shots, and one consumer's value made from that
    check (FutureWorld.derived: (key, value))."""
    __slots__ = ("kill", "hb", "world", "shape", "shape_ks", "check", "derived")

    def __init__(self, kill: tuple):
        self.kill = kill
        self.hb = self.world = self.shape = self.shape_ks = self.check = self.derived = None


def _ro(a: np.ndarray) -> np.ndarray:
    """a, read-only: it is shared between calls (a consumer writing into it would change every later window)."""
    a.flags.writeable = False
    return a


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


@lru_cache(maxsize=256)
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


def _kill(hz: list, keep, skip=frozenset()):
    """The shapes that kill (game.level.hazard_shapes: the game's own geometry) as an (M, 5) array, their pieces'
    indices and their shape types (M,). skip: rows whose kill shapes are left out (homing beams)."""
    if skip:
        hz = [list(h[:7]) + [[]] if k in skip else h for k, h in enumerate(hz)]
    b, kept, shape = hazard_shapes(hz, keep)
    return np.asarray(b, np.float64).reshape(-1, 5), kept, np.asarray(shape, np.int8)


def _offset(a: list, b: list, skip=frozenset()) -> float:
    """How far two frames' hazards are apart: the pieces' positions and their kill shapes' centres (inf when
    the pieces or their shapes differ in number). skip: rows left out (homing beams: they follow the agent)."""
    if len(a) != len(b):
        return float("inf")
    if skip:
        a = [h for k, h in enumerate(a) if k not in skip]
        b = [h for k, h in enumerate(b) if k not in skip]
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
