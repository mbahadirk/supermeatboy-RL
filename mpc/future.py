"""Where the level's moving parts will be over the next frames.

With the game's time step locked, hazards sit at the same place at the same wave frame in every wave
(measured: 0.0 units off on 1-13 and 1-14). So the first wave on a level is recorded, and later waves look
the future up in it ("recorded"), but only while the current positions keep matching the recording. A new
level starts a new recording: a wave cut by the level switch (no wave end) belongs to the old level and is
dropped. Otherwise (first wave, a mismatch, or frames past the recording) positions are extrapolated with their
current velocity ("extrapolated"), with an uncertainty radius that grows with time: the planner keeps that
much more distance, so a guessed future is never treated as known.

Two views of each frame's hazards: the model inputs (wm.features.hazard_boxes: each piece's own box, as the
world model was trained) and the boxes that kill (game.level.hazard_boxes: the game's kill shapes, which move
inside a piece's animation while the piece stays put). The recording is trusted while both match.
"""
from __future__ import annotations

import numpy as np

from game.level import box_ids, hazard_boxes as kill_boxes
from wm.features import hazard_boxes
from wm.rollout import World

MATCH_TOL = 1.0          # game units: the recording still holds if every hazard is this close to it
STILL = 5.0              # units/s: slower than this counts as not moving (radius 0 when extrapolated)
GROW = 0.5               # extrapolated radius: 2 units + this x speed x time ahead
REC_RADIUS = 1.0         # recorded positions: this much margin


class FutureWorld:
    def __init__(self):
        self.new_level()

    def new_level(self):
        self.ref: dict[int, tuple] = {}       # wave frame -> (hz raw, pf, sh) of the reference recording
        self.cur: dict[int, tuple] = {}       # a wave cut by the level switch belongs to the old level: dropped
        self.keep, self.fans = None, []
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
        self.trusted = True                   # this wave still matches the recording
        self.mismatch = None                  # (frame, units off) of the first mismatch
        self.last = None                      # (frame, hz, pf, sh, dt_ms) now
        self.prev = None                      # the frame before
        return info

    def observe(self, frame: int, hz: list, pf, sh, keep, fans, dt_ms: float):
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
        self.prev, self.last = self.last, (frame, rec[0], rec[1], rec[2], dt_ms)

    def worlds(self, frame: int, H: int, dt_ms: float):
        """World (model inputs) for steps 0..H-1 (frames frame..frame+H-1), the boxes that kill at steps 1..H
        as (boxes (M, 5), extra radius (M,), ids (M,): game.level.box_ids, then "shot<id>"), and how many steps
        came from the recording."""
        if self.last is None or self.last[0] != frame:
            raise RuntimeError("observe() this frame before planning")
        _, hz0, pf0, sh0, _ = self.last
        boxes0 = hazard_boxes(hz0, self.keep)
        kill0, kept0 = _kill(hz0, self.keep)
        ids0 = box_ids(kept0)
        vel = np.zeros((len(boxes0), 2))
        kvel = np.zeros((len(kill0), 2))
        pf_vel = np.zeros((len(pf0), 2))
        shots_prev = {}
        if self.prev is not None and self.prev[0] == frame - 1:
            dt_s = max(1e-3, self.prev[4] / 1000.0)
            b1 = hazard_boxes(self.prev[1], self.keep)
            if len(b1) == len(boxes0):
                vel = (boxes0[:, :2] - b1[:, :2]) / dt_s
            k1, kept1 = _kill(self.prev[1], self.keep)
            if kept1 == kept0:                # the same shapes of the same pieces: their own motion
                kvel = (kill0[:, :2] - k1[:, :2]) / dt_s
            if len(self.prev[2]) == len(pf0) and len(pf0):
                pf_vel = (np.array(pf0, float)[:, :2] - np.array(self.prev[2], float)[:, :2]) / dt_s
            shots_prev = {s[0]: s for s in self.prev[3]}
        dt_s = dt_ms / 1000.0
        kspeed = np.hypot(kvel[:, 0], kvel[:, 1])
        use_rec = self.trusted and bool(self.ref)
        recorded = 0
        hz_k, pf_k, kill_k = [], [], []
        for k in range(H + 1):
            rec = self.ref.get(frame + k) if use_rec else None
            if k == 0:
                b, p = boxes0, pf0
                kb = (kill0, np.zeros(len(kill0)), ids0)
            elif rec is not None:
                b, p = hazard_boxes(rec[0], self.keep), rec[1]
                kr, kept_r = _kill(rec[0], self.keep)
                kb = (kr, np.full(len(kr), REC_RADIUS), box_ids(kept_r))
                recorded += 1
            else:
                use_rec = False               # once off the recording, stay off it for this plan
                t = k * dt_s
                b = boxes0.copy()
                b[:, :2] += vel * t
                kr = kill0.copy()
                kr[:, :2] += kvel * t
                kb = (kr, np.where(kspeed > STILL, 2.0 + GROW * kspeed * t, 0.0), ids0)
                p = [list(q) for q in pf0]
                for j, q in enumerate(p):
                    q[0], q[1] = q[0] + pf_vel[j, 0] * t, q[1] + pf_vel[j, 1] * t
            hz_k.append(b)
            pf_k.append(p)
            kill_k.append(kb)
        # flying shots: only kept away from (the world model was not trained with them), always extrapolated
        shots = []
        for s in sh0:
            q = shots_prev.get(s[0])
            v = ((s[1] - q[1]) / max(1e-3, self.prev[4] / 1000.0), (s[2] - q[2]) / max(1e-3, self.prev[4] / 1000.0))                 if q is not None else (0.0, 0.0)
            shots.append((s[0], s[1], s[2], s[3], v))
        worlds = [World(hz_k[k], pf_k[k] or None, (pf_k[k - 1] if k else (self.prev[2] if self.prev else None)) or None,
                        dt_s, list(self.fans)) for k in range(H)]
        check = []
        for k in range(1, H + 1):
            b, r, ids = kill_k[k]
            if shots:
                t = k * dt_s
                sb = np.array([(x + vx * t, y + vy * t, rad, rad, 0.0) for _, x, y, rad, (vx, vy) in shots])
                sr = np.array([2.0 + GROW * np.hypot(*v) * t for *_, v in shots])
                b, r = np.concatenate([b, sb]), np.concatenate([r, sr])
                ids = list(ids) + [f"shot{sid}" for sid, *_ in shots]
            check.append((b, r, list(ids)))
        return worlds, check, recorded


def _kill(hz: list, keep):
    """The boxes that kill (game.level.hazard_boxes) as an (M, 5) array, and their pieces' indices."""
    b, kept = kill_boxes(hz, keep)
    return np.asarray(b, np.float64).reshape(-1, 5), kept


def _offset(a: list, b: list) -> float:
    """How far two frames' hazards are apart: the pieces' positions and their kill shapes' centres (inf when
    the pieces or their shapes differ in number)."""
    if len(a) != len(b):
        return float("inf")
    if not a:
        return 0.0
    pa, pb = np.array([h[:2] for h in a], float), np.array([h[:2] for h in b], float)
    off = float(np.max(np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1])))
    (ka, ia), (kb, ib) = _kill(a, None), _kill(b, None)
    if ia != ib:
        return float("inf")
    if len(ka):
        off = max(off, float(np.max(np.hypot(ka[:, 0] - kb[:, 0], ka[:, 1] - kb[:, 1]))))
    return off
