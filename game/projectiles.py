"""Observed projectile facts shared by PPO and the planner's existing shot history.

Raw sh=[id,x,y,radius], shx=[kind,vx,vy], game units and seconds unchanged.
No dynamics model, future trajectories or action selection live here.
"""
from __future__ import annotations
import math
import numpy as np

LAUNCH_KEY, LAUNCH_P_MIN, LAUNCH_P_TOL = 2.0, 10, 1
SHOT_FIT, SHOT_V_MAX, SHOT_JUMP = 8, 10.0, 1.0
N_LAUNCHERS, LAUNCHER_WIDTH = 2, 13
N_PROJECTILE = N_LAUNCHERS * LAUNCHER_WIDTH
PROJECTILE_SCHEMA = 'observed-shot-clock-v1'


def parse_shx(sh, shx):
    """Aligned native metadata; missing/malformed rows remain explicitly unknown."""
    sh = sh or []
    if not shx or len(shx) != len(sh):
        return [(None, None, None)] * len(sh)
    out = []
    for row in shx:
        try:
            k, vx, vy = row
            k, vx, vy = int(k), float(vx), float(vy)
            if not math.isfinite(vx) or not math.isfinite(vy):raise ValueError('nonfinite velocity')
            out.append((k, vx, vy))
        except (TypeError, ValueError, OverflowError):
            out.append((None, None, None))
    return out


def launch_key(x, y):
    return int(round(float(x) / LAUNCH_KEY)), int(round(float(y) / LAUNCH_KEY))


class Launcher:
    """The existing observed-birth contract: period survives waves; last birth does not.

    template/complete retain the planner's past-flight API, without predicting anything.
    """
    def __init__(self):
        self.P = self.last = self.sid = self.radius = self.template = None
        self.complete = False

    def born(self, frame, sid, radius):
        if self.last is not None and frame - self.last >= LAUNCH_P_MIN:
            self.P = int(frame - self.last)
        self.last, self.sid, self.radius = int(frame), sid, float(radius)


def shot_track(history, frame, sid, fit=SHOT_FIT, vmax=SHOT_V_MAX, jump=SHOT_JUMP):
    """Current observed flight, identical slot-reuse cuts to FutureWorld.shot_track."""
    out = []
    for f in range(frame, frame - fit, -1):
        rec = history.get(f)
        q = next((s for s in rec[2] if s[0] == sid), None) if rec is not None else None
        if q is None:break
        p = np.array((float(q[1]), float(q[2])))
        if out and np.hypot(*(p - out[-1])) > vmax:break
        out.append(p)
        if len(out) >= 3 and np.max(np.abs(out[-3] - 2 * out[-2] + out[-1])) > jump:
            if len(out) == 3 and np.hypot(*(out[0] - out[1])) > np.hypot(*(out[1] - out[2])):
                del out[1:]
            else:out.pop()
            break
    return np.array(out[::-1], np.float64).reshape(-1, 2)


class ShotClock:
    """Observed births/periods, using the same rules as the main branch's shot history."""
    def __init__(self):
        self.new_level()

    def new_level(self):
        self.launchers, self.facts = {}, {}
        self.new_wave()

    def new_wave(self):
        self.cur, self.flights = {}, {}
        self.frame = None
        for launcher in self.launchers.values():launcher.last = None

    def observe(self, frame, sh, shx=None):
        if self.frame is not None and frame <= self.frame:self.new_wave()
        self.frame = int(frame)
        sh = sh or []
        self.cur[frame] = ([], [], sh)
        seen = set()
        for s, ext in zip(sh, parse_shx(sh, shx)):
            sid, x, y, radius = s
            seen.add(sid)
            fl = self.flights.get(sid)
            track = shot_track(self.cur, frame, sid)
            if fl is None or fl['last'] != frame - 1 or len(track) == 1:
                key = launch_key(x, y)
                fl = self.flights[sid] = dict(key=key, last=frame,
                    first=len(self.cur) > 1 and frame - 1 in self.cur)
                if fl['first']:
                    self.launchers.setdefault(key, Launcher()).born(frame, sid, radius)
            key = fl['key'];fl['last'] = frame
            if fl['first']:
                previous = self.facts.get(key)
                origin = (float(x), float(y)) if len(track) == 1 else (previous['x'], previous['y'])
                velocity = ext[1:] if ext[0] is not None else tuple((track[-1] - track[-2]) * 60) if len(track)>1 else (0.,0.)
                self.facts[key] = dict(x=origin[0], y=origin[1], vx=float(velocity[0]), vy=float(velocity[1]),
                                       velocity_known=ext[0] is not None or len(track)>1,
                                       kind=ext[0], radius=float(radius))
        self.flights = {sid:fl for sid,fl in self.flights.items() if sid in seen}
        self.cur = {f:r for f,r in self.cur.items() if f > frame - SHOT_FIT}

    def features(self, x, y):
        """Two nearest sources: [present,dx,dy,vx,vy,r,P,age,eta,P_known,phase_known,v_known,kind].

        Position /200, velocity /600, radius /50, times /120 frames. Missing phase
        is zero with phase_known=0; a missed expected firing invalidates phase.
        """
        out = np.zeros((N_LAUNCHERS, LAUNCHER_WIDTH), np.float32)
        keys = [k for k in self.launchers if k in self.facts and self.facts[k]['kind'] != 16]
        # Main's future shots use established periods only. Short/reused-slot tracks
        # must not displace a verified source with an unconfirmed nearby birth.
        keys.sort(key=lambda k: (self.launchers[k].P is None,
                                 (self.facts[k]['x']-x)**2+(self.facts[k]['y']-y)**2,k))
        for row, key in zip(out, keys):
            L, f = self.launchers[key], self.facts[key]
            known = L.P is not None
            phase = known and L.last is not None and self.frame is not None and self.frame-L.last <= L.P+LAUNCH_P_TOL
            age = self.frame-L.last if phase else 0
            eta = max(0,L.P-age) if phase else 0
            row[:] = [1,(f['x']-x)/200,(f['y']-y)/200,f['vx']/600,f['vy']/600,
                      f['radius']/50,(L.P or 0)/120,age/120,eta/120,known,phase,
                      f['velocity_known'],f['kind']/5 if f['kind'] is not None else -1]
        return np.clip(out.ravel(),-8,8)
