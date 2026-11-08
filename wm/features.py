"""World-model inputs and targets from recorded gameplay (wm/buffer.py shards).

Everything is relative to the agent, so the model learns how Meat Boy moves, not where a level's walls are:
the level is looked up (tiles around the agent, nearby hazards and platforms, wind), never memorised.

Input (FEAT_DIM floats):
  own state   vx/400, vy/600, time step / 33 ms, jump held / 20, position inside its tile (x, y; -0.5..0.5),
              wind here (/2000, the game's formula)                                                    8
  flags       the character's 32 flag bits (+0x7b4: airborne, on a wall, facing, input bits ...)     32
  state       movement state (+0x7b0) one-hot                                                         24
  action      one-hot (realgame.ACTIONS)                                                              10
  tiles       7x7 cells around the agent, 4 channels: full block, half block, slope ((open corner+1)/4),
              special (crumble / lock / switch), rows bottom-up                                       196
  hazards     the 3 nearest (box edge distance): dx, dy, distance (/100), half w, half h (/50)         15
  platform    the nearest: dx, dy (/100), upright, vx, vy (/300), present                              6

Targets: the next flag bits that change during play (FLAG_BITS), events, and position change and new
velocity as the correction to plain physics (`prior`: the same velocity
with gravity in the air, moved by velocity x time step), scaled by RES_SCALE; next flags (airborne, on a wall, facing);
death, finish; landing (airborne -> not); blocked (horizontal speed cut by a wall); next movement state;
reward of the step (/5).
"""
from __future__ import annotations

import math
import os

import numpy as np

from realpath import tile_classes, slope_open_corner, box_distance
from wm.buffer import transitions, chains, load_level, load_shards

TILE = 20.0
P = 3                                  # tile patch radius: 7x7
N_STATES = 24
N_ACTIONS = 10
N_HAZ = 3
AIR, WALL, FACE = 0x2, 0x200, 0x1
CONT_NAMES = ["vx", "vy", "dt", "held", "in-tile x", "in-tile y", "wind x", "wind y"]
FEAT_DIM = len(CONT_NAMES) + 32 + N_STATES + N_ACTIONS + 4 * (2 * P + 1) ** 2 + 5 * N_HAZ + 6
# the flag bits that change during play (measured on real data), predicted so rollouts can carry them on
FLAG_BITS = [0x2, 0x200, 0x1, 0x4, 0x20, 0x800, 0x2000, 0x8000000]
BIN_NAMES = ["airborne", "on wall", "facing", "flag 0x4", "flag 0x20", "flag 0x800", "flag 0x2000", "flag 0x8000000",
             "death", "finish", "landing", "blocked"]
EVENTS = {n: BIN_NAMES.index(n) for n in ("death", "finish", "landing", "blocked")}
DT_DEFAULT = 33.0                      # ms (median with 100 agents)
RES_SCALE = np.array([2.0, 2.0, 40.0, 60.0], np.float32)   # dx, dy, vx, vy corrections (game units)
GRAVITY = 550.0


class LevelGeom:
    """A level's tiles as 4 channels (padded with solid outside the map), kept up to date with tile changes."""

    def __init__(self, words: np.ndarray, meta: dict):
        self.ox, self.oy = float(meta["ox"]), float(meta["oy"])
        self.words0 = words.astype(np.uint32)
        self.H, self.W = self.words0.shape
        self.words = self.words0.copy()
        k = tile_classes(self.words0)
        self.special0 = k["crumble"] | k["lock"] | k["switch"] | k["other"]
        self.ch = self._channels(self.words0)
        self.pad = np.pad(self.ch, ((0, 0), (P, P), (P, P)))
        self.pad[0, :, :P] = self.pad[0, :, -P:] = 1.0     # outside the map: solid
        self.pad[0, :P, :] = self.pad[0, -P:, :] = 1.0
        self._table = None                                 # every padded cell's patch, built on first use

    @staticmethod
    def _channels(words):
        k = tile_classes(words)
        so = slope_open_corner(words)
        ch = np.zeros((4,) + words.shape, np.float32)
        ch[0] = k["full"]
        ch[1] = k["half"]
        ch[2] = np.where(k["slope"], (so + 1) / 4.0, 0.0)
        ch[3] = k["crumble"] | k["lock"] | k["switch"] | k["other"]
        return ch

    def reset(self):
        self.words[:] = self.words0
        self.pad[:, P:-P, P:-P] = self.ch
        self._table = None

    def set_tile(self, idx: int, word: int):
        r, c = divmod(int(idx), self.W)
        if not (0 <= r < self.H):
            return
        self.words[r, c] = word
        if self.special0[r, c]:                # a crumble / lock / switch block: gone once its word changed
            v = self.ch[:, r, c] if int(word) == int(self.words0[r, c]) else np.zeros(4, np.float32)
        else:
            v = self._channels(np.array([[word]], np.uint32))[:, 0, 0]
        self.pad[:, r + P, c + P] = v
        if self._table is not None:                        # the patches that contain this cell
            rr, cc = np.mgrid[r:r + 2 * P + 1, c:c + 2 * P + 1]
            self._table[rr, cc] = self._gather(rr.ravel(), cc.ravel()).reshape(rr.shape + (-1,))

    def patches(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The 4 x (2P+1) x (2P+1) tile patch around each point (channel, then row, then column), flattened: one
        lookup in a table of every padded cell's patch (built once, kept up to date by set_tile / reset)."""
        c = np.clip(np.floor((x - self.ox) / TILE).astype(int), -P, self.W + P - 1) + P
        r = np.clip(np.floor((y - self.oy) / TILE).astype(int), -P, self.H + P - 1) + P
        if self._table is None:
            rr, cc = np.mgrid[0:self.H + 2 * P, 0:self.W + 2 * P]
            self._table = self._gather(rr.ravel(), cc.ravel()).reshape(rr.shape + (-1,))
        return self._table[r, c]

    def _gather(self, r: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Patches of padded cells (r, c) from the padded channels (edge cells repeat the outermost padding)."""
        d = np.arange(-P, P + 1)
        rr = np.clip(r[:, None, None] + d[None, :, None], 0, self.H + 2 * P - 1)
        cc = np.clip(c[:, None, None] + d[None, None, :], 0, self.W + 2 * P - 1)
        return self.pad[:, rr, cc].transpose(1, 0, 2, 3).reshape(len(r), -1)

    def in_tile(self, x, y):
        return ((x - self.ox) / TILE) % 1.0 - 0.5, ((y - self.oy) / TILE) % 1.0 - 0.5


def hazard_boxes(hz: list, keep=None) -> np.ndarray:
    """The model's hazard inputs, as it was trained: each piece's own box [x, y, size w, size h, scale x,
    scale y, angle] (the first 7 fields of a realgame.js hazards row) -> (x, y, half w, half h, angle), with
    keep (game.level.Level.hz_keep) dropping decoration. Not what kills: game.level.hazard_boxes reads the
    rows' kill shapes for that."""
    out = []
    for k, h in enumerate(hz):
        if keep is not None and k < len(keep) and not keep[k]:
            continue
        x, y, w, hgt, sx, sy, ang = h[:7]
        sx, sy = abs(sx or 1.0), abs(sy or 1.0)
        if w <= 0 and hgt <= 0:
            w = hgt = 24.0
        out.append((x, y, 0.5 * w * sx, 0.5 * hgt * sy, ang))
    return np.array(out, np.float64).reshape(-1, 5)


def box_distances(x, y, boxes) -> np.ndarray:
    """realpath.box_distance for every box at once: (n points, M boxes)."""
    b = np.asarray(boxes, np.float64).reshape(-1, 5)
    c, s = np.cos(b[:, 4]), np.sin(b[:, 4])
    dx = np.asarray(x, np.float64)[:, None] - b[None, :, 0]
    dy = np.asarray(y, np.float64)[:, None] - b[None, :, 1]
    u, v = dx * c + dy * s, -dx * s + dy * c
    return np.hypot(np.maximum(np.abs(u) - b[:, 2], 0.0), np.maximum(np.abs(v) - b[:, 3], 0.0))


def hazard_features(x, y, boxes) -> np.ndarray:
    n = len(x)
    out = np.zeros((n, N_HAZ, 5), np.float32)
    out[:, :, 2] = 3.0                         # no hazard: far away
    if not len(boxes):
        return out.reshape(n, -1)
    d = box_distances(x, y, boxes)                                   # (n, boxes)
    order = np.argsort(d, 1)[:, :N_HAZ]
    for j in range(order.shape[1]):
        b = boxes[order[:, j]]
        dj = d[np.arange(n), order[:, j]]
        out[:, j] = np.stack([(b[:, 0] - x) / 100, (b[:, 1] - y) / 100, dj / 100, b[:, 2] / 50, b[:, 3] / 50], 1)
    out[:, :, :2] = np.clip(out[:, :, :2], -3, 3)
    out[:, :, 2] = np.minimum(out[:, :, 2], 3.0)
    return out.reshape(n, -1)


def platform_features(x, y, pf, pf_prev, dt_prev_s) -> np.ndarray:
    n = len(x)
    out = np.zeros((n, 6), np.float32)
    if not pf:
        return out
    p = np.array(pf, np.float64)
    d = np.hypot(p[None, :, 0] - x[:, None], p[None, :, 1] - y[:, None])
    k = np.argmin(d, 1)
    vel = np.zeros((len(p), 2))
    if pf_prev and len(pf_prev) == len(pf) and dt_prev_s:
        vel = (p[:, :2] - np.array(pf_prev, np.float64)[:, :2]) / dt_prev_s
    out[:, 0] = np.clip((p[k, 0] - x) / 100, -3, 3)
    out[:, 1] = np.clip((p[k, 1] - y) / 100, -3, 3)
    out[:, 2] = p[k, 2]
    out[:, 3:5] = np.clip(vel[k] / 300, -3, 3)
    out[:, 5] = (d[np.arange(n), k] < 10 * TILE).astype(np.float32)
    return out


def wind(x, y, fans) -> tuple[np.ndarray, np.ndarray]:
    """The game's wind force (FUN_004c40e0), as realgame.Level.wind_at."""
    fx, fy = np.zeros(len(x)), np.zeros(len(x))
    for fxp, fyp, ux, uy, hw, rng, st in fans or []:
        if rng <= 0:
            continue
        along = (x - fxp) * ux + (y - fyp) * uy
        lat = (x - fxp) * uy - (y - fyp) * ux
        inside = (along >= 0) & (along <= rng) & (np.abs(lat) <= hw + 6.0)
        c = 1.0 - along / rng
        c = np.where(c >= 0.5, c * 5.0, c)
        fx += np.where(inside, ux * st * c, 0.0)
        fy += np.where(inside, uy * st * c, 0.0)
    return fx, fy


_SHIFTS = np.arange(32, dtype=np.uint64)
_C_FLAGS = len(CONT_NAMES)
_C_ST = _C_FLAGS + 32
_C_ACT = _C_ST + N_STATES
_C_PATCH = _C_ACT + N_ACTIONS
_C_HAZ = _C_PATCH + 4 * (2 * P + 1) ** 2
_C_PLAT = _C_HAZ + 5 * N_HAZ


def featurize(s, a, dt_ms, geom: LevelGeom, boxes, pf, pf_prev, dt_prev_s, fans, out=None) -> np.ndarray:
    """Inputs for agent rows s (wm.buffer.ROW) taking actions a with this frame's step and level state. out: an
    (n, FEAT_DIM) float32 array to write them into (the planner's per-step buffer; the same values as a new
    array), returned."""
    n = len(s)
    X = np.empty((n, FEAT_DIM), np.float32) if out is None else out
    x, y = s["x"].astype(np.float64), s["y"].astype(np.float64)
    tx, ty = geom.in_tile(x, y)
    wx, wy = wind(x, y, fans)
    X[:, :_C_FLAGS] = np.stack([s["vx"] / 400.0, s["vy"] / 600.0, np.full(n, dt_ms / DT_DEFAULT), s["held"] / 20.0,
                                tx, ty, np.clip(wx / 2000, -3, 3), np.clip(wy / 2000, -3, 3)], 1)
    X[:, _C_FLAGS:_C_ST] = (s["flags"][:, None].astype(np.uint64) >> _SHIFTS) & 1
    X[:, _C_ST:_C_PATCH] = 0.0
    r = np.arange(n)
    X[r, _C_ST + np.clip(s["state"], 0, N_STATES - 1)] = 1.0
    X[r, _C_ACT + np.clip(a, 0, N_ACTIONS - 1)] = 1.0
    X[:, _C_PATCH:_C_HAZ] = geom.patches(x, y)
    X[:, _C_HAZ:_C_PLAT] = hazard_features(x, y, boxes)
    X[:, _C_PLAT:] = platform_features(x, y, pf, pf_prev, dt_prev_s)
    return X


def prior(s, dt_ms, sub: int = 1) -> np.ndarray:
    """Plain physics [dx, dy, vx', vy'] (game units): the same velocity, gravity in the air, moved by the new
    velocity x the time step. The model predicts only the correction to this (input, collisions, walls).
    sub: the step is `sub` frames of dt_ms / sub each (a step-k model: gravity and movement per frame)."""
    vx, vy = s["vx"].astype(np.float64), s["vy"].astype(np.float64)
    air = (s["flags"] & AIR) != 0
    if sub == 1:
        dt = dt_ms / 1000.0
        vy2 = np.where(air, vy - GRAVITY * dt, vy)
        return np.stack([vx * dt, vy2 * dt, vx, vy2], 1).astype(np.float32)
    h = np.asarray(dt_ms, np.float64) / 1000.0 / sub
    dy = np.zeros_like(vy)
    for _ in range(sub):
        vy = np.where(air, vy - GRAVITY * h, vy)
        dy = dy + vy * h
    return np.stack([vx * h * sub, dy, vx, vy], 1).astype(np.float32)


def featurize_k(s, acts, dt_ms, geom: LevelGeom, boxes, pf, pf_prev, dt_prev_s, fans) -> np.ndarray:
    """Inputs of a step-k model (wm.model): featurize with the first of the k actions acts (n, k) and the whole
    step's dt, then the other k - 1 actions one-hot (N_ACTIONS each) after all other inputs."""
    n, k = acts.shape
    X = np.zeros((n, FEAT_DIM + (k - 1) * N_ACTIONS), np.float32)
    featurize(s, acts[:, 0], dt_ms, geom, boxes, pf, pf_prev, dt_prev_s, fans, out=X[:, :FEAT_DIM])
    actions_into(X, acts)
    return X


def actions_into(X, acts):
    """The extra one-hot actions 2..k of a step-k model's inputs (columns after FEAT_DIM), in place."""
    n, k = acts.shape
    if k > 1:
        X[:, FEAT_DIM:] = 0.0
        r = np.arange(n)
        for j in range(1, k):
            X[r, FEAT_DIM + (j - 1) * N_ACTIONS + np.clip(acts[:, j], 0, N_ACTIONS - 1)] = 1.0


def targets(s, s2, r, pri):
    """(continuous: correction to `prior` / RES_SCALE, binary [BIN_NAMES], next state class, reward/5)."""
    raw = np.stack([s2["x"] - s["x"], s2["y"] - s["y"], s2["vx"], s2["vy"]], 1).astype(np.float32)
    cont = (raw - pri) / RES_SCALE
    air0, air1 = (s["flags"] & AIR) != 0, (s2["flags"] & AIR) != 0
    blocked = (np.abs(s["vx"]) > 40) & (np.abs(s2["vx"]) < 0.25 * np.abs(s["vx"]))
    binary = np.stack([(s2["flags"] & b) != 0 for b in FLAG_BITS] +
                      [s2["status"] == 1, s2["status"] == 2, air0 & ~air1, blocked], 1)
    rew = np.clip(np.nan_to_num(r) / 5.0, -10, 10)
    return cont.astype(np.float32), binary.astype(np.float32), np.clip(s2["state"], 0, N_STATES - 1), \
        rew.astype(np.float32)


def baselines(s, dt_ms):
    """Raw-unit predictions [dx, dy, vx', vy'] of two simple guesses: the same velocity, and the same velocity
    with gravity in the air (no input, no collisions)."""
    dt = dt_ms / 1000.0
    vx, vy = s["vx"].astype(np.float64), s["vy"].astype(np.float64)
    const = np.stack([vx * dt, vy * dt, vx, vy], 1)
    air = (s["flags"] & AIR) != 0
    vy2 = np.where(air, vy - GRAVITY * dt, vy)
    grav = np.stack([vx * dt, vy2 * dt, vx, vy2], 1)
    return const, grav


def group_of(s) -> np.ndarray:
    """0 on the ground, 1 in the air, 2 on a wall."""
    wall = (s["flags"] & WALL) != 0
    air = (s["flags"] & AIR) != 0
    return np.where(wall, 2, np.where(air, 1, 0)).astype(np.int8)


def demo_key(d: str) -> str:
    """A wm_data folder's identity for demo lookup (relative paths from the smbrl folder)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normcase(os.path.normpath(d if os.path.isabs(d) else os.path.join(here, d)))


def build(dirs: list, levels=None, max_samples=None, seed=0, log=print, frac=None, exclude=(), step: int = 1,
          demos: dict | None = None) -> dict:
    """Features, targets and baselines of every transition in these wm_data folders. frac: keep this share of
    each wave's transitions (all deaths and finishes) before the inputs are built, for runs too long to
    featurize whole. exclude: levels left out (the held-out ones). step k > 1: samples of a step-k model
    (wm.buffer.chains: k frames, the k actions, the summed dt, events of any of the frames; "full" False where a
    death / finish cut the chain short; the targets' continuous part in game units minus the prior, / RES_SCALE
    as for step 1: wm.train rescales it to the model's res_scale). demos: {normalized folder (demo_key):
    {shard index: [(agent, first frame, last frame)]}}: those agents' transitions are marked in "demo" and kept by
    frac and max_samples (wm.train --demos)."""
    rng_f = np.random.default_rng(seed + 1)
    X, YC, YB, YS, YR, G, SH, LV, BC, BG, RAW, PR, FU, DM = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    shard_no = 0
    for d in dirs:
        geoms = {}
        dem = (demos or {}).get(demo_key(d), {})
        for meta, rows, world in load_shards(d):
            lv = meta.get("level")
            shard_no += 1
            if (levels and lv not in levels) or lv in exclude:
                continue
            if lv not in geoms:
                try:
                    geoms[lv] = LevelGeom(*load_level(d, lv))
                except (OSError, KeyError):
                    log(f"[wm] no level data for {lv}, skipped")
                    geoms[lv] = None
            geom = geoms[lv]
            if geom is None:
                continue
            geom.reset()
            if step > 1:
                tr = chains(rows, step)
                s, s2, a, r = tr["s"], tr["s2"], tr["a"], tr["r"]
                ext = {"full": tr["full"], "landing": tr["landing"], "blocked": tr["blocked"]}
            else:
                tr = transitions(rows)
                s, s2, a, r = tr["s"], tr["s2"], tr["a"], tr["r"]
                ext = None
            is_demo = np.zeros(len(s), bool)
            for ag, f0, f1 in (dem.get(int(rows["wave"][0]), []) if len(rows) else []):
                is_demo |= (s["agent"] == ag) & (s["frame"] >= f0) & (s["frame"] <= f1)
            if frac is not None and frac < 1.0 and len(s):
                k = (s2["status"] != 0) | (rng_f.random(len(s)) < frac) | is_demo
                s, s2, a, r, is_demo = s[k], s2[k], a[k], r[k], is_demo[k]
                if ext is not None:
                    ext = {kk: v[k] for kk, v in ext.items()}
            if not len(s):
                continue
            wf = {w["f"]: w for w in world}
            frames = np.unique(s["frame"])
            keep, fans, upto = None, [], -1
            for f in frames:                   # frame by frame: tiles, hazards, platforms, wind as the game had them
                w = wf.get(int(f), {})
                for ff in range(upto + 1, int(f) + 1):     # everything the game reported up to this frame
                    for idx, word in wf.get(ff, {}).get("tc", []) or []:
                        geom.set_tile(idx, word)
                    if "keep" in wf.get(ff, {}):
                        keep = wf[ff]["keep"]
                    if "fx" in wf.get(ff, {}):
                        fans = wf[ff]["fx"].get("fans", [])
                upto = int(f)
                m = np.nonzero(s["frame"] == f)[0]
                dt = w.get("dt") or DT_DEFAULT
                pw = wf.get(int(f) - 1, {})
                if ext is None:
                    X.append(featurize(s[m], a[m], dt, geom, hazard_boxes(w.get("hz", []), keep), w.get("pf"),
                                       pw.get("pf"), (pw.get("dt") or DT_DEFAULT) / 1000.0, fans).astype(np.float16))
                    pri = prior(s[m], dt)
                    yc, yb, ys_, yr = targets(s[m], s2[m], r[m], pri)
                else:
                    dt = sum((wf.get(int(f) + j, {}).get("dt") or dt) for j in range(step))
                    X.append(featurize_k(s[m], a[m], dt, geom, hazard_boxes(w.get("hz", []), keep), w.get("pf"),
                                         pw.get("pf"), (pw.get("dt") or DT_DEFAULT) / 1000.0, fans).astype(np.float16))
                    pri = prior(s[m], dt, step)
                    yc, yb, ys_, yr = targets(s[m], s2[m], r[m], pri)
                    yb[:, EVENTS["landing"]] = ext["landing"][m]
                    yb[:, EVENTS["blocked"]] = ext["blocked"][m]
                    FU.append(ext["full"][m])
                PR.append(pri)
                YC.append(yc); YB.append(yb); YS.append(ys_); YR.append(yr)
                G.append(group_of(s[m]))
                SH.append(np.full(len(m), shard_no, np.int32))
                DM.append(is_demo[m])
                LV.append(np.array([lv] * len(m)))
                bc, bg = baselines(s[m], dt)
                BC.append(bc.astype(np.float32)); BG.append(bg.astype(np.float32))
                RAW.append(np.stack([s2["x"][m] - s["x"][m], s2["y"][m] - s["y"][m], s2["vx"][m], s2["vy"][m]], 1))
            log(f"[wm] {d} shard {shard_no} ({lv}): {len(s)} transitions")
    if not X:
        raise SystemExit("no transitions found")
    out = {"X": np.concatenate(X), "yc": np.concatenate(YC), "yb": np.concatenate(YB), "ys": np.concatenate(YS),
           "yr": np.concatenate(YR), "group": np.concatenate(G), "shard": np.concatenate(SH),
           "level": np.concatenate(LV), "base_const": np.concatenate(BC), "base_grav": np.concatenate(BG),
           "raw": np.concatenate(RAW).astype(np.float32), "prior": np.concatenate(PR)}
    out["full"] = np.concatenate(FU) if FU else np.ones(len(out["X"]), bool)
    out["demo"] = np.concatenate(DM)
    n = len(out["X"])
    if max_samples and n > max_samples:        # keep every rare event (deaths, finishes), sample the rest
        rng = np.random.default_rng(seed)
        rare = (out["yb"][:, EVENTS["death"]] > 0) | (out["yb"][:, EVENTS["finish"]] > 0) | out["demo"]
        idx = np.concatenate([np.nonzero(rare)[0],
                              rng.choice(np.nonzero(~rare)[0], max_samples - int(rare.sum()), replace=False)])
        idx.sort()
        out = {k: v[idx] for k, v in out.items()}
    return out
