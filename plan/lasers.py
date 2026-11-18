"""Homing laser eyes in planning: the eye's aim, beam and kill test from the game's code
(analysis/reactive_hazards_20261117/rules.md, section 1), rolled forward along each imagined trajectory.

A homing eye (laser kind 1, piece hominglazeron/off/wait) turns toward the game's main character every frame. With
realgame.js that character is hidden and parked on the leading agent, so the beam follows the agent: a recorded or
extrapolated beam from an earlier wave is wrong for any other path (4-16, runs/online_check11: every candidate looked
deadly). Here every trajectory gets its own beam future:

- aim (rules.md 1.1, lag 0): the world update of frame f turns the eye toward the target's position in that frame's
  message (the trajectory's frame f position, before its step f): a = acos(D . F), the side chosen by comparing sines
  rounded up to 1/1000 (the game's quirk, kept: aim_step reproduces verify.py's float32 rule), turn k x a with
  k = 1 / piece[+0x48] (0.1 when that is 0; K_UNKNOWN when the params were not recorded).
- beam (1.2): from the eye centre along the angle (turned round by the piece's flip bits), the game's sampled ray cast
  (ceil(L / 10) samples, the first whose cell word's low byte is not 0 is the end), L = the level's diagonal. The
  length per beam direction comes from a table per eye (BeamTable, NBINS directions, each the exact sampled cast on the
  live tile words; between two directions the longer one, with live ray casts across discontinuities).
- kill (1.4): the beam kills in state 1 only; FUN_004cbd40 tests the character's centre within the beam's length and
  in front of the eye, then the swept leading-edge x / y segments and the static cross of half size BODY (6).
- frames: step i of a path (its frame-i position to its frame-i+1 position: the agent's update of frame i) is tested
  against the beam after the world update of frame i, aimed at the frame-i position. Frame 0's eye angle is the
  observed one (already aimed at the current position).
- lethal schedule (1.3): with the recorded laser state ("lz": state, timer) the off -> wait -> on -> off cycle is
  stepped with the piece's off / on times and the wait length measured on this level (frames in state 2), else the
  level's first lethal wave frame seen in earlier waves; unknown: lethal (conservative). Without "lz": lethal now
  stays lethal (the on time is unknown), not lethal now becomes lethal at that first lethal wave frame, else at once.

Clearance (soft risk, nearest logs): outside the length / behind the eye the distance past that gate, else the
centre's distance to the beam's line less the static cross's reach (BODY x max(|cos|, |sin|) of the beam); a frame the
exact test kills is negative, one it does not is >= 0, so contact is exactly the game's test.

The eye targets the leader only. The planner assumes the planning agent is the leader (one planned agent: agent.py's
leader is the first live planned one), so a candidate's own trajectory is the target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from game.level import HOMING, LASER_TYPE, SHRAPNEL_ID, SHRAPNEL_KIND, parse_lz, parse_lzp, parse_shx  # noqa: F401
from realpath import BODY

f32 = np.float32
TILE = 20.0
STEP = 10.0               # ray-cast sample spacing (FUN_004663f0: 20 x 0.5)
K_DEFAULT = 0.1           # the game's turn factor when the piece's divisor (+0x48) is 0
K_UNKNOWN = 1.0 / 18.0    # without the recorded params: the divisor of every homing eye in the level files (4-16, 4-20)
NBINS = 16384             # beam directions per eye in the length table (2 pi / NBINS = 0.38 mrad)
DT = 1.0 / 60.0           # the locked game step (s)
SCHEDULE_PAD = 160        # lethal schedule frames beyond the asked horizon (the completion, plans made ahead)
REFINE_TOL = 0.006        # rad: a recorded (0.01-rounded) eye angle is refined from its beam row within this
NO_WAIT = 0               # wait frames when unknown and nothing else is known: lethal at once (conservative)


# ---- the game's arithmetic, vectorised ---------------------------------------------------------------------------
def c1000(x) -> np.ndarray:
    """ceil(float32(x) x 1000) / 1000 in float32 (the aim code's rounding; verify.py c1000)."""
    v = np.asarray(x, np.float64).astype(f32) * f32(1000.0)
    return np.ceil(v) / f32(1000.0)


def aim_step(theta, ex, ey, tx, ty, k) -> np.ndarray:
    """One frame of homing eyes (rules.md 1.1; verify.py aim_step, vectorised, float32 like the game): angles theta,
    eye centres (ex, ey), targets (tx, ty), turn factors k (broadcast arrays). Returns the new angles (float32)."""
    theta = np.asarray(theta, f32)
    dx = (np.asarray(tx, f32) - np.asarray(ex, f32)).astype(f32)
    dy = (np.asarray(ty, f32) - np.asarray(ey, f32)).astype(f32)
    ln = np.sqrt((dx * dx + dy * dy).astype(f32).astype(np.float64)).astype(f32)
    pos = ln > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(pos, f32(1.0) / np.where(pos, ln, f32(1.0)), f32(1.0)).astype(f32)
    dx = np.where(pos, dx * inv, dx).astype(f32)
    dy = np.where(pos, dy * inv, dy).astype(f32)
    th64 = theta.astype(np.float64)
    fx, fy = np.cos(th64).astype(f32), np.sin(th64).astype(f32)
    dot = dx.astype(np.float64) * fx.astype(np.float64) + dy.astype(np.float64) * fy.astype(np.float64)
    ok = (dot >= -1.0) & (dot <= 1.0)
    a = np.arccos(np.clip(dot, -1.0, 1.0)).astype(f32)
    x1 = c1000(np.sin((a + theta).astype(f32).astype(np.float64)))
    x2 = c1000(np.sin((theta - a).astype(f32).astype(np.float64)))
    t = c1000(np.sin(np.arctan2(dy.astype(np.float64), dx.astype(np.float64)).astype(f32).astype(np.float64)))
    ka = (np.asarray(k, f32) * a).astype(f32)
    out = np.where(ok & (x1 == t), theta + ka, np.where(ok & (x2 == t), theta - ka, theta))
    return out.astype(f32)


def level_length(W: int, H: int) -> float:
    """The beam's length before the ray cast: the level's diagonal (rules.md 1.2), float32 as the game."""
    w, h = f32(W * 20), f32(H * 20)
    return float(f32(math.sqrt(float(f32(w * w + h * h)))))


def cast_lengths(ex: float, ey: float, phis, solid: np.ndarray, ox: float, oy: float, L: float) -> np.ndarray:
    """FUN_004663f0's sampled ray cast from (ex, ey) along the beam directions phis (n,): the beam's length per
    direction (n,). solid (H, W): the cell words' low byte != 0 (row 0 = bottom); samples at i x len / ceil(len / 10)
    from the eye on, the first solid one is the end (overshooting into the wall by up to one sample); none: L."""
    H, W = solid.shape
    phis = np.asarray(phis, np.float64)
    sx, sy = f32(ex), f32(ey)
    Lf = f32(L)
    endx = (sx + (np.cos(phis).astype(f32) * Lf).astype(f32)).astype(f32)
    endy = (sy + (np.sin(phis).astype(f32) * Lf).astype(f32)).astype(f32)
    dx, dy = (endx - sx).astype(f32), (endy - sy).astype(f32)
    ln = np.sqrt((dx * dx + dy * dy).astype(f32).astype(np.float64)).astype(f32)
    n = np.ceil((ln / f32(STEP)).astype(np.float64)).astype(np.int64)
    s = (ln / n.astype(f32)).astype(f32)
    inv = (f32(1.0) / ln).astype(f32)
    ux, uy = (dx * inv).astype(f32), (dy * inv).astype(f32)
    N = int(n.max(initial=1))
    i = np.arange(N, dtype=np.float64)
    out = ln.astype(np.float64).copy()
    for a in range(0, len(phis), 1024):
        b = slice(a, a + 1024)
        # The game repeatedly adds a float32 step, rather than multiplying by i.
        # At a wall corner that rounding can change which sample hits by many cells.
        px = np.broadcast_to((ux[b] * s[b])[:, None], (len(ux[b]), N)).copy()
        py = np.broadcast_to((uy[b] * s[b])[:, None], (len(uy[b]), N)).copy()
        px[:, 0], py[:, 0] = sx, sy
        np.cumsum(px, axis=1, dtype=f32, out=px)
        np.cumsum(py, axis=1, dtype=f32, out=py)
        cx = np.clip(np.floor((px - f32(ox)) / f32(TILE)).astype(np.int64), 0, W - 1)
        cy = np.clip(np.floor((py - f32(oy)) / f32(TILE)).astype(np.int64), 0, H - 1)
        hit = solid[cy, cx] & (i[None, :] < n[b, None])
        anyh = hit.any(1)
        first = hit.argmax(1)
        j = np.arange(len(px))
        hitlen = np.hypot(px[j, first].astype(np.float64) - sx, py[j, first].astype(np.float64) - sy)
        out[b] = np.where(anyh, hitlen, out[b])
    return out


def beam_solid(words: np.ndarray) -> np.ndarray:
    """The ray cast's solid test on the live cell words: the low byte is not 0 (rules.md 1.2). Not Level.solid: a
    crumble block that started to break (4-16: 0x03000164 -> 0x03400164) still blocks the beam until its word is
    0x100 (runs/online_check11: the recorded beam lengths follow the live words exactly)."""
    return (np.asarray(words) & 0xFF) != 0


def beam_clearance(ex, ey, phi, blen, ax, ay, bx, by, vx=None, vy=None, h: float = BODY):
    """FUN_004cbd40 (rules.md 1.4) for broadcast arrays: beams from the eye (ex, ey) along phi with length blen; the
    character's centre now (ax, ay) and before its move (bx, by), its velocity (vx, vy) (default the move). Returns
    (clearance, hit): hit is the game's test; clearance < 0 exactly where it hits (module docstring)."""
    ux, uy = np.cos(phi), np.sin(phi)
    rx, ry = ax - ex, ay - ey
    dist = np.hypot(rx, ry)
    along = rx * ux + ry * uy
    s_a = uy * rx - ux * ry                             # side of the line (d_y, -d_x) . (p - S), per unit length
    s_b = uy * (bx - ex) - ux * (by - ey)
    vx = ax - bx if vx is None else vx
    vy = ay - by if vy is None else vy
    sx, sy = h * np.sign(vx), h * np.sign(vy)

    def cross(p, q):
        return ((p > 0) & (q < 0)) | ((p < 0) & (q > 0))
    hit = ((dist <= blen) & (along >= 0)
           & (cross(s_b + uy * sx, s_a + uy * sx) | cross(s_b - ux * sy, s_a - ux * sy)
              | cross(s_a + uy * h, s_a - uy * h) | cross(s_a - ux * h, s_a + ux * h)))
    clear = np.maximum(np.maximum(dist - blen, -along), np.abs(s_a) - h * np.maximum(np.abs(ux), np.abs(uy)))
    clear = np.where(hit, np.minimum(clear, -1e-3), np.maximum(clear, 0.0))
    return clear, hit


# ---- per eye: the beam length table ------------------------------------------------------------------------------
class BeamTable:
    """Beam length per direction (NBINS over the turn) for one eye on the live tiles, updated where tiles changed."""

    def __init__(self, ex: float, ey: float, lv, nbins: int = NBINS):
        self.ex, self.ey, self.nbins = float(ex), float(ey), int(nbins)
        self.ox, self.oy = float(lv.ox), float(lv.oy)
        self.L = level_length(lv.W, lv.H)
        self.solid = beam_solid(lv.words)
        self.phis = 2.0 * math.pi * np.arange(self.nbins) / self.nbins
        self.len = cast_lengths(self.ex, self.ey, self.phis, self.solid, self.ox, self.oy, self.L).astype(f32)

    def update(self, words) -> int:
        """The live tile words: the directions whose ray can meet a changed cell are cast again. Returns how many."""
        solid = beam_solid(words)
        ch = np.argwhere(solid != self.solid)
        if not len(ch):
            return 0
        self.solid = solid.copy()
        redo = np.zeros(self.nbins, bool)
        for r, c in ch:
            x0, y0 = self.ox + c * TILE, self.oy + r * TILE
            cx, cy = x0 + TILE / 2 - self.ex, y0 + TILE / 2 - self.ey
            near = math.hypot(max(abs(cx) - TILE / 2, 0.0), max(abs(cy) - TILE / 2, 0.0))
            if near < TILE:
                redo[:] = True
                break
            a0 = math.atan2(cy, cx)
            hw = max(abs((math.atan2(y - self.ey, x - self.ex) - a0 + math.pi) % (2 * math.pi) - math.pi)
                     for x in (x0, x0 + TILE) for y in (y0, y0 + TILE)) + 2.0 / self.nbins * 2 * math.pi
            b = np.arange(int(math.floor((a0 - hw) / (2 * math.pi) * self.nbins)),
                          int(math.ceil((a0 + hw) / (2 * math.pi) * self.nbins)) + 1) % self.nbins
            redo[b[self.len[b] >= near - 1.5 * STEP]] = True
        idx = np.nonzero(redo)[0]
        if len(idx):
            self.len[idx] = cast_lengths(self.ex, self.ey, self.phis[idx], self.solid, self.ox, self.oy, self.L)
        return int(len(idx))


def table_length(table: np.ndarray, phi) -> np.ndarray:
    """Beam lengths at directions phi (..., nE) from the eyes' tables (nE, NBINS): the longer of the two neighbouring
    directions (a conservative approximation, not a geometric bound at occlusion discontinuities)."""
    nb = table.shape[-1]
    b = np.mod(np.asarray(phi, np.float64), 2 * math.pi) * (nb / (2 * math.pi))
    b0 = np.floor(b).astype(np.int64) % nb
    b1 = (b0 + 1) % nb
    e = np.arange(table.shape[0])
    return np.maximum(table[e, b0], table[e, b1])


# ---- one decision's eye future ---------------------------------------------------------------------------------
@dataclass
class Beams:
    """The homing eyes at one decision frame (EyeTracker.context): ids, centres, angles now (float32, whole turns
    kept), turn factors, flip (the beam turned round), length tables (nE, NBINS), lethal (nE, S) per frame offset from
    the decision frame (beyond S: the last value), the decision's wave frame, and the leader's position now (target
    of the frames before a path that starts later; None unknown). Picklable (agent.py sends it with the frames)."""
    ids: list
    ex: np.ndarray
    ey: np.ndarray
    theta: np.ndarray
    k: np.ndarray
    flip: np.ndarray
    tables: np.ndarray
    lethal: np.ndarray
    frame: int = 0
    start: tuple | None = None
    solid: np.ndarray | None = None
    origin: tuple | None = None
    length: float | None = None

    def lengths(self, phi, on):
        """Use table lookups except across an occlusion edge; cast those rays on the actual tiles.

        A direction at a wall corner can jump hundreds of units between neighbouring bins.
        Choosing the longer bin there incorrectly shines through the wall.
        """
        tables = self.tables[on]
        result = table_length(tables, phi)
        if self.solid is None:
            return result
        b = np.mod(phi, 2 * math.pi) * (tables.shape[1] / (2 * math.pi))
        b0 = np.floor(b).astype(np.int64) % tables.shape[1]
        e = np.arange(len(on))
        edge = np.abs(tables[e, b0] - tables[e, (b0 + 1) % tables.shape[1]]) > 1.0
        for j, eye in enumerate(on):
            mask = edge[..., j]
            if mask.any():
                result[..., j][mask] = cast_lengths(self.ex[eye], self.ey[eye], phi[..., j][mask],
                                                    self.solid, *self.origin, self.length)
        return result

    def __len__(self):
        return len(self.ids)

    def lethal_at(self, k0: int, F: int) -> np.ndarray:
        """(F, nE): lethal at frame offsets k0 .. k0 + F - 1."""
        S = self.lethal.shape[1]
        idx = np.clip(np.arange(k0, k0 + F), 0, S - 1)
        return self.lethal[:, idx].T

    def seed(self, x0, y0, k0: int) -> np.ndarray:
        """(P, nE) angles at offset k0 > 0 for paths starting at (x0, y0) (P,) there, the eyes rolled from now along
        the straight line from the leader's position now (start; unknown: the path's start) to each path's start."""
        P = len(x0)
        th = np.broadcast_to(self.theta, (P, len(self))).astype(f32)
        if k0 <= 0:
            return th.copy()
        sx, sy = (np.asarray(x0, np.float64), np.asarray(y0, np.float64)) if self.start is None else \
            (np.full(P, float(self.start[0])), np.full(P, float(self.start[1])))
        for i in range(1, k0 + 1):
            t = i / k0
            tx, ty = sx + (x0 - sx) * t, sy + (y0 - sy) * t
            th = aim_step(th, self.ex[None], self.ey[None], tx[:, None], ty[:, None], self.k[None])
        return th

    def columns(self, X, Y, step: int = 1, k0: int = 0, theta=None, jumps=None, want_theta: bool = False):
        """Beam contact of paths X, Y (P, K+1) of columns (column c = frame offset k0 + c x step; jumps: their
        teleports, game.level.Jumps of that shape, or None). theta (P, nE): the angles at offset k0 (None: now's for
        k0 = 0, else seed). Returns (clearance (P, K, nE): per column the least over its frames' steps (+inf where no
        beam is lethal), theta at the last column (P, nE) or None (want_theta False and no beam lethal)). Each frame's
        beam is aimed at that path's own position then."""
        from game.level import frame_path
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        P, K1 = X.shape
        K, nE = K1 - 1, len(self)
        F = K * int(step)
        if K < 1 or nE == 0:
            return np.full((P, max(K, 0), nE), np.inf), None
        leth = self.lethal_at(k0, F)                                   # (F, nE)
        if not leth.any() and not want_theta:
            return np.full((P, K, nE), np.inf), None
        px, py = frame_path(X, Y, step, jumps)                          # (P, F + 1)
        th = self.seed(px[:, 0], py[:, 0], k0) if theta is None else np.asarray(theta, f32).reshape(P, nE)
        ths = np.empty((P, F, nE), f32)
        ths[:, 0] = th
        ex, ey, kk = self.ex[None], self.ey[None], self.k[None]
        txf, tyf = px.astype(f32), py.astype(f32)
        for i in range(1, F):
            th = aim_step(th, ex, ey, txf[:, i, None], tyf[:, i, None], kk)
            ths[:, i] = th
        th_end = aim_step(th, ex, ey, txf[:, F, None], tyf[:, F, None], kk) if want_theta else None
        clear = np.full((P, F, nE), np.inf)
        on = np.nonzero(leth.any(0))[0]
        if len(on):
            phi = ths[:, :, on].astype(np.float64) + np.where(self.flip[on], math.pi, 0.0)
            blen = self.lengths(phi, on)                             # (P, F, eyes on)
            e_x, e_y = self.ex[on].astype(np.float64), self.ey[on].astype(np.float64)
            bx, by = px[:, :-1].copy(), py[:, :-1].copy()
            pre = None
            if jumps is not None and jumps.any():
                r, col = np.nonzero(jumps.at[:, 1:])
                col = col + 1
                t = (col - 1) * int(step) + jumps.frame[r, col] - 1
                t = t.astype(np.int64)
                ent, ext = jumps.entry[r, col], jumps.exit[r, col]
                pre = (r, t, bx[r, t].copy(), by[r, t].copy(), ent)
                bx[r, t] = np.where(np.isnan(ext[:, 0]), px[r, t + 1], ext[:, 0])
                by[r, t] = np.where(np.isnan(ext[:, 1]), py[r, t + 1], ext[:, 1])
            c, _ = beam_clearance(e_x, e_y, phi, blen, px[:, 1:, None], py[:, 1:, None], bx[..., None], by[..., None])
            if pre is not None:
                r, t, sx, sy, ent = pre
                valid = ~np.isnan(ent).any(1)
                r, t, sx, sy, ent = r[valid], t[valid], sx[valid], sy[valid], ent[valid]
                pc, _ = beam_clearance(e_x, e_y, phi[r, t], blen[r, t], ent[:, :1], ent[:, 1:],
                                       sx[:, None], sy[:, None])
                c[r, t] = np.minimum(c[r, t], pc)
            clear[:, :, on] = np.where(leth[None, :, on], c, np.inf)
        return clear.reshape(P, K, int(step), nE).min(2), th_end


# ---- per level: what the eyes do now --------------------------------------------------------------------------
@dataclass
class Eye:
    id: str                    # the beam's lethal id ("laser0x..." when known, else "laser@h<row>")
    beam_row: int | None       # the beam's row in the frame's hz list (its recorded segment: dropped)
    eye_row: int | None        # the eye piece's row (its angle, without "lz")
    lz: int | None             # its index in "lz" / "lzp"
    x: float
    y: float
    k: float
    flip: bool
    off_t: float | None
    on_t: float | None
    wait_t: float | None = None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class EyeTracker:
    """The level's homing eyes (FutureWorld.set_lasers): identified from realgame.js "lzp" (Level.lzp) or, without
    it, from the hz rows (beam rows of type LASER_TYPE paired with the nearest eye piece named hominglazer*), observed
    every frame (angle, state, lethal now; the level's first lethal wave frame and wait length learned), and the
    decision's Beams (context)."""

    def __init__(self, lv):
        self.lv = lv
        self.eyes: list[Eye] = []
        self._key = None
        self.tables: dict[str, BeamTable] = {}
        self.first_on: dict[str, int] = {}          # eye id -> the earliest wave frame its beam was lethal
        self.wait: dict[str, int] = {}              # eye id -> frames in state 2 (wait) before lethal, measured
        self.new_wave()

    @staticmethod
    def candidate(lv) -> bool:
        """The level may have homing eyes: "lzp" with a homing laser, or a beam row and an eye piece type named
        hominglazer* (the tracker is made only then: levels without them run exactly as before)."""
        lzp = getattr(lv, "lzp", None)
        if lzp:
            return any(L.get("kind") == HOMING for L in lzp["lasers"])
        types = list(getattr(lv, "hz_types", []) or [])
        names = getattr(lv, "names", None) or []
        nm = lambda t: names[t].lower() if 0 <= t < len(names) else ""
        return LASER_TYPE in types and any(nm(t).startswith("hominglazer") for t in types if t != LASER_TYPE)

    @classmethod
    def for_level(cls, lv):
        return cls(lv) if cls.candidate(lv) else None

    def new_wave(self):
        self.wave_start = None                      # first frame observed in this wave
        self.on_seen: set = set()                   # eyes already lethal in this wave
        self.prev_state: dict = {}
        self.wait_from: dict = {}
        self.cur: dict = {}                         # eye id -> (theta, state, timer, lethal now)
        self.animation: dict = {}                   # eye id -> observed wait animation time (not the state timer)
        self.frame = None
        self.target = None

    # identification
    def _identify(self, hz):
        lv = self.lv
        ids = list(getattr(lv, "hz_ids", []) or [])
        lzp = getattr(lv, "lzp", None)
        key = (tuple(ids), id(lzp), len(hz))
        if key == self._key:
            return
        self._key = key
        eyes = []
        if lzp:
            for j, L in enumerate(lzp["lasers"]):
                if L["kind"] != HOMING or L["x"] is None or L["y"] is None:
                    continue
                p = L["params"]
                k = K_UNKNOWN if p[5] is None else (1.0 / p[5] if p[5] else K_DEFAULT)
                wait_t = None if lzp.get("wait_len") is None or p[0] is None else lzp["wait_len"] * p[0]
                eyes.append(Eye(L["id"], ids.index(L["id"]) if L["id"] in ids else None,
                                ids.index(L["piece"]) if L["piece"] in ids else None, j, L["x"], L["y"], k,
                                L["flip"] in (1, 3), p[2], p[3], wait_t))
        else:
            types = list(getattr(lv, "hz_types", []) or [])
            names = getattr(lv, "names", None) or []
            nm = lambda t: names[t].lower() if 0 <= t < len(names) else ""
            beams = [r for r in range(min(len(hz), len(types))) if types[r] == LASER_TYPE]
            lazers = [r for r in range(min(len(hz), len(types))) if "lazer" in nm(types[r])]
            for r in beams:
                h = hz[r]
                bx, by, bl, ang = float(h[0]), float(h[1]), float(h[2]), float(h[6])
                sx, sy = bx - 0.5 * bl * math.cos(ang), by - 0.5 * bl * math.sin(ang)    # the beam's start
                if not lazers:
                    continue
                e = min(lazers, key=lambda q: math.hypot(float(hz[q][0]) - sx, float(hz[q][1]) - sy))
                if math.hypot(float(hz[e][0]) - sx, float(hz[e][1]) - sy) > 3.0 or \
                        not nm(types[e]).startswith("hominglazer"):
                    continue                          # a plain lazer's beam (or unpaired): kept as recorded
                bid = ids[r] if r < len(ids) and str(ids[r]).startswith("laser") else f"laser@h{r}"
                eyes.append(Eye(bid, r, e, None, float(hz[e][0]), float(hz[e][1]), K_UNKNOWN, False, None, None))
        self.eyes = eyes
        for e in eyes:
            t = self.tables.get(e.id)
            if t is None or (t.ex, t.ey) != (float(e.x), float(e.y)):
                self.tables[e.id] = BeamTable(e.x, e.y, lv)

    def beam_rows(self) -> frozenset:
        """hz rows of the homing beams (their recorded / extrapolated segments are stale for planning)."""
        return frozenset(e.beam_row for e in self.eyes
                         if e.beam_row is not None and self.cur.get(e.id, (None,))[0] is not None)

    def progress_ids(self) -> frozenset:
        """Reactive beam identities excluded from permanent route obstacles (not from live safety).

        Level parameters identify them before the first observation; legacy captures
        use the tracker's existing piece/beam pairing. Plain lasers and eye bodies remain.
        """
        ids = {e.id for e in self.eyes}
        params = getattr(self.lv, "lzp", None)
        if params:
            ids.update(e["id"] for e in params["lasers"] if e["kind"] == HOMING)
        return frozenset(ids)

    def observe(self, frame: int, hz: list, lz=None, target=None):
        """This frame's hz rows, realgame.js "lz" (or None) and the leader's position (the eyes' target; None
        unknown)."""
        self._identify(hz)
        self._obs = getattr(self, "_obs", 0) + 1
        frame = int(frame)
        if self.wave_start is None:
            self.wave_start = frame
        self.frame, self.target = frame, (None if target is None else (float(target[0]), float(target[1])))
        lzs = parse_lz(lz) if lz is not None else None
        for e in self.eyes:
            st = tm = ang = animation = None
            if lzs is not None and e.lz is not None and e.lz < len(lzs):
                st, tm, ang, animation = lzs[e.lz]
            th = ang if ang is not None else self._angle(e, hz, frame)
            if st is not None:
                lethal = st == 1
            else:
                h = hz[e.beam_row] if e.beam_row is not None and e.beam_row < len(hz) else None
                lethal = bool(h is not None and len(h) > 7 and h[7])
            self.cur[e.id] = (th, st, tm, lethal)
            self.animation[e.id] = animation
            # learned: the first lethal wave frame (waves seen from their reset) and the wait length
            if lethal and e.id not in self.on_seen:
                self.on_seen.add(e.id)
                if self.wave_start <= 1 and frame >= 1:
                    self.first_on[e.id] = min(self.first_on.get(e.id, frame), frame)
            ps = self.prev_state.get(e.id)
            if st == 2 and ps != 2:
                self.wait_from[e.id] = frame if ps is not None else None
            if st == 1 and ps == 2 and self.wait_from.get(e.id) is not None:
                self.wait[e.id] = frame - self.wait_from[e.id]
            self.prev_state[e.id] = st
        for e in self.eyes:                         # the live words (crumble blocks breaking): the tables follow
            t = self.tables.get(e.id)               # (every word change, not Level.tile_ver: a breaking crumble
            if t is not None:                       # block is open for Level.solid but blocks a beam)
                t.update(self.lv.words)

    def _angle(self, e: Eye, hz, frame: int):
        """The eye's angle without "lz": its piece row's (exact live; recorded rounded to 0.01), refined from the beam
        row's direction (good to about 1e-4 rad) when the angle looks rounded and the beam row is fresh (frame >= 1)."""
        if e.eye_row is None or e.eye_row >= len(hz):
            return self.cur.get(e.id, (None,))[0]
        a = float(hz[e.eye_row][6])
        if frame < 1 or abs(a * 100 - round(a * 100)) > 1e-6 or e.beam_row is None or e.beam_row >= len(hz):
            return a
        h = hz[e.beam_row]
        if len(h) > 7 and h[7]:
            _, x0, y0, x1, y1 = h[7][0][:5]
            d = math.atan2(y1 - y0, x1 - x0)
        else:
            if float(h[2]) < 1.0:
                return a
            d = math.atan2(float(h[1]) - e.y, float(h[0]) - e.x)
        if e.flip:
            d += math.pi
        off = _wrap(d - a)
        return a + off if abs(off) <= REFINE_TOL else a

    def schedule(self, e: Eye, n: int) -> np.ndarray:
        """(n,) lethal at frame offsets 0..n-1 from now (module docstring)."""
        th, st, tm, lethal = self.cur.get(e.id, (None, None, None, True))
        out = np.zeros(n, bool)
        fo = self.first_on.get(e.id)
        if st is None:                                   # no recorded state
            if lethal:
                out[:] = True
            else:
                start = 1 if fo is None else max(1, fo - self.frame)
                out[start:] = True
            return out
        t = float(tm or 0.0)
        s = st
        w = None if s != 2 else int(round(t / DT))
        W = self.wait.get(e.id)
        anim = self.animation.get(e.id)
        for i in range(n):
            if i:
                t += DT
                if s == 0:
                    if e.off_t is None or t > e.off_t:
                        s, t, w, anim = 2, 0.0, 0, 0.0
                elif s == 2:
                    w += 1
                    if anim is not None:
                        anim += DT
                elif s == 1 and e.on_t is not None and t > e.on_t:
                    s, t = 0, 0.0
                if s == 2:
                    if e.wait_t is not None and anim is not None:
                        go = anim >= e.wait_t
                    elif W is not None:
                        go = w >= W
                    elif fo is not None:
                        go = self.frame + i >= fo
                    else:
                        go = w >= NO_WAIT
                    if go:
                        s, t = 1, 0.0
            out[i] = s == 1
        return out

    def context(self, frame: int, H: int) -> Beams | None:
        """The eyes at this frame for a decision (H: frames asked; the schedule covers H + SCHEDULE_PAD), or None when
        no eye is known (or not observed this frame)."""
        if not self.eyes or self.frame != int(frame):
            return None
        eyes = [e for e in self.eyes if self.cur.get(e.id, (None,))[0] is not None]
        if not eyes:
            return None
        n = int(H) + SCHEDULE_PAD
        c = getattr(self, "_ctx", None)                 # asked again this frame (danger check, then a decision)
        if c is not None and c[0] == self.frame and c[1] >= n and c[2] == self._obs:
            return c[3]
        ctx = self._context(eyes, n)
        self._ctx = (self.frame, n, self._obs, ctx)
        return ctx

    def _context(self, eyes, n: int) -> Beams:
        return Beams(ids=[e.id for e in eyes], ex=np.array([e.x for e in eyes], f32),
                     ey=np.array([e.y for e in eyes], f32),
                     theta=np.array([self.cur[e.id][0] for e in eyes], f32),
                     k=np.array([e.k for e in eyes], f32), flip=np.array([e.flip for e in eyes], bool),
                     tables=np.stack([self.tables[e.id].len for e in eyes]),
                     lethal=np.stack([self.schedule(e, n) for e in eyes]), frame=int(self.frame), start=self.target,
                     solid=beam_solid(self.lv.words), origin=(self.lv.ox, self.lv.oy),
                     length=level_length(self.lv.W, self.lv.H))
