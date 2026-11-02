"""Where the level's moving parts will be over the next frames.

With the game's time step locked, hazards sit at the same place at the same wave frame in every wave
(measured: 0.0 units off on 1-13 and 1-14). So the first wave on a level is recorded, and later waves look
the future up in it ("recorded"), but only while the current positions keep matching the recording.
Otherwise (first wave, a mismatch, or frames past the recording) positions are extrapolated with their
current velocity ("extrapolated"), with an uncertainty radius that grows with time: the planner keeps that
much more distance, so a guessed future is never treated as known.
"""
from __future__ import annotations

import numpy as np

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
        self.keep, self.fans, self.grow = None, [], None
        self.new_wave()

    def new_wave(self):
        cur = getattr(self, "cur", {})
        if cur and (not self.ref or self.trusted):  # the first wave, or one that matched: extends the recording
            top = max(self.ref) if self.ref else -1
            for f, v in cur.items():
                if f > top:
                    self.ref[f] = v
        self.cur: dict[int, tuple] = {}
        self.trusted = True                   # this wave still matches the recording
        self.mismatch = None                  # (frame, units off) of the first mismatch
        self.last = None                      # (frame, hz, pf, sh, dt_ms) now
        self.prev = None                      # the frame before

    def observe(self, frame: int, hz: list, pf, sh, keep, fans, dt_ms: float, grow=None):
        if keep is not None:
            self.keep = keep
        if grow is not None:
            self.grow = grow
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
        """World (model inputs) for steps 0..H-1 (frames frame..frame+H-1), the hazard boxes to keep away
        from at steps 1..H with their extra radius, and how many steps came from the recording."""
        if self.last is None or self.last[0] != frame:
            raise RuntimeError("observe() this frame before planning")
        _, hz0, pf0, sh0, _ = self.last
        boxes0 = hazard_boxes(hz0, self.keep)
        vel = np.zeros((len(boxes0), 2))
        pf_vel = np.zeros((len(pf0), 2))
        shots_prev = {}
        if self.prev is not None and self.prev[0] == frame - 1:
            dt_s = max(1e-3, self.prev[4] / 1000.0)
            b1 = hazard_boxes(self.prev[1], self.keep)
            if len(b1) == len(boxes0):
                vel = (boxes0[:, :2] - b1[:, :2]) / dt_s
            if len(self.prev[2]) == len(pf0) and len(pf0):
                pf_vel = (np.array(pf0, float)[:, :2] - np.array(self.prev[2], float)[:, :2]) / dt_s
            shots_prev = {s[0]: s for s in self.prev[3]}
        dt_s = dt_ms / 1000.0
        speed = np.hypot(vel[:, 0], vel[:, 1])
        use_rec = self.trusted and bool(self.ref)
        recorded = 0
        hz_k, pf_k = [], []
        radius = []
        for k in range(H + 1):
            rec = self.ref.get(frame + k) if use_rec else None
            if k == 0:
                b, p, r = boxes0, pf0, np.zeros(len(boxes0))
            elif rec is not None:
                b, p = hazard_boxes(rec[0], self.keep), rec[1]
                r = np.full(len(b), REC_RADIUS)
                recorded += 1
            else:
                use_rec = False               # once off the recording, stay off it for this plan
                t = k * dt_s
                b = boxes0.copy()
                b[:, :2] += vel * t
                r = np.where(speed > STILL, 2.0 + GROW * speed * t, 0.0)
                p = [list(q) for q in pf0]
                for j, q in enumerate(p):
                    q[0], q[1] = q[0] + pf_vel[j, 0] * t, q[1] + pf_vel[j, 1] * t
            hz_k.append(b)
            pf_k.append(p)
            radius.append(r)
        # flying shots: only kept away from (the world model was not trained with them), always extrapolated
        shots = []
        for s in sh0:
            q = shots_prev.get(s[0])
            v = ((s[1] - q[1]) / max(1e-3, self.prev[4] / 1000.0), (s[2] - q[2]) / max(1e-3, self.prev[4] / 1000.0)) \
                if q is not None else (0.0, 0.0)
            shots.append((s[1], s[2], s[3], v))
        worlds = [World(hz_k[k], pf_k[k] or None, (pf_k[k - 1] if k else (self.prev[2] if self.prev else None)) or None,
                        dt_s, list(self.fans)) for k in range(H)]
        # the check keeps the measured kill reach of each kind of piece (hazard_reach.py); the model inputs
        # above keep the game's sizes
        g = np.array([self.grow[j] for j in range(len(hz0)) if self.keep is None or j >= len(self.keep) or self.keep[j]]
                     if self.grow is not None and len(self.grow) >= len(hz0) else np.zeros((0, 2))).reshape(-1, 2)
        check = []
        for k in range(1, H + 1):
            b, r = hz_k[k], radius[k]
            if len(g) == len(b) and len(b):
                b = b.copy()
                b[:, 2:4] += g
            if shots:
                t = k * dt_s
                sb = np.array([(x + vx * t, y + vy * t, rad, rad, 0.0) for x, y, rad, (vx, vy) in shots])
                sr = np.array([2.0 + GROW * np.hypot(*v) * t for *_, v in shots])
                b, r = np.concatenate([b, sb]), np.concatenate([r, sr])
            check.append((b, r))
        return worlds, check, recorded


def _offset(a: list, b: list) -> float:
    if len(a) != len(b):
        return float("inf")
    if not a:
        return 0.0
    pa, pb = np.array([h[:2] for h in a], float), np.array([h[:2] for h in b], float)
    return float(np.max(np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1])))
