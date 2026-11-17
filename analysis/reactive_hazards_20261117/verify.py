"""Check the decoded homing-laser rules (rules.md) against recorded 4-16 waves. Read-only: no game, no training.

Run from smbrl:  python analysis/reactive_hazards_20261117/verify.py [run_dir ...]
Default run: runs/online_check11. Prints error statistics for
  (a) the eye's per-frame turn (free-run from the placed angle; target = the leading agent, lags 0/1/2;
      turn factor k = 1/18 (the eye's param) and 0.1 (the game's default when the param is 0)),
  (b) the beam end vs the game's sampled ray cast on the level tiles,
  (c) the off -> wait -> on timing (first lethal frame after the reset),
  (d) the beam kill test on every recorded step (deaths explained / hits without a death),
  (e) the shared shot spawn / integration code on recorded straight shots (chapter 3, runs/online_check01):
      spawn 20 units from the shooter centre, not moved on its spawn frame, then speed x dt per frame.
      launcher2 missiles are never recorded (realgame.js shots() has no radius for kind 5), so their homing
      law is code-only.
Recorded values are rounded to 0.01 (positions, eye angle) by realgame.js, so beam directions are good to
about 0.01 / (beam half length) rad.
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
f32 = np.float32
TILE = f32(20.0)
STEP = f32(10.0)                  # ray-cast sample spacing: 20 * 0.5 ([0x6f6ab4] * [0x6ba1c8])
BODY = 6.0                        # Meat Boy's box half size (realpath.BODY)


# ---- the game's arithmetic (float32 where the game uses SSE single precision) -------------------------------
def c1000(x: float) -> f32:
    """ceil(x * 1000) / 1000 the way the aim code rounds (float32 multiply, double ceil, float32 divide)."""
    return f32(f32(math.ceil(float(f32(f32(x) * f32(1000.0))))) / f32(1000.0))


def aim_step(theta: f32, eye: tuple, target: tuple, k: f32) -> f32:
    """One frame of a homing eye (FUN_004d4550, 0x4d8573..0x4d894d)."""
    dx, dy = f32(f32(target[0]) - f32(eye[0])), f32(f32(target[1]) - f32(eye[1]))
    ln = f32(math.sqrt(float(f32(dx * dx + dy * dy))))
    if ln > 0:
        inv = f32(f32(1.0) / ln)
        dx, dy = f32(dx * inv), f32(dy * inv)
    fx, fy = f32(math.cos(float(theta))), f32(math.sin(float(theta)))          # M(theta) * (1, 0)
    dot = float(dx) * float(fx) + float(dy) * float(fy)                         # x87, stored as double
    if not -1.0 <= dot <= 1.0:
        return theta                                                            # acos -> NaN: no match, no turn
    a = f32(math.acos(dot))
    x1 = c1000(math.sin(float(f32(a + theta))))
    x2 = c1000(math.sin(float(f32(theta - a))))
    t = c1000(math.sin(float(f32(math.atan2(float(dy), float(dx))))))
    if x1 == t:
        return f32(theta + f32(k * a))
    if x2 == t:
        return f32(theta - f32(k * a))
    return theta


def ray_cast(start, theta: f32, lvl) -> tuple:
    """FUN_004663f0 from the eye along theta (beam end before the cast: start + R(theta) (L, 0))."""
    W, H, ox, oy, solid = lvl
    L = f32(math.sqrt(float(f32(f32(W * 20) * f32(W * 20) + f32(H * 20) * f32(H * 20)))))
    sx, sy = f32(start[0]), f32(start[1])
    ex = f32(sx + f32(f32(math.cos(float(theta))) * L))
    ey = f32(sy + f32(f32(math.sin(float(theta))) * L))
    dx, dy = f32(ex - sx), f32(ey - sy)
    ln = f32(math.sqrt(float(f32(dx * dx + dy * dy))))
    inv = f32(f32(1.0) / ln)
    dx, dy = f32(dx * inv), f32(dy * inv)
    n = int(math.ceil(float(f32(ln / STEP))))
    s = f32(ln / f32(n))
    px, py = sx, sy
    for _ in range(n):
        tx = min(max(int(math.floor(float(f32(f32(px - f32(ox)) / TILE)))), 0), W - 1)
        ty = min(max(int(math.floor(float(f32(f32(py - f32(oy)) / TILE)))), 0), H - 1)
        if solid[ty, tx]:
            return float(px), float(py)
        px, py = f32(px + f32(dx * s)), f32(py + f32(dy * s))
    return float(ex), float(ey)


def beam_kills(S, E, A, B, v, h=BODY) -> bool:
    """FUN_004cbd40: character centre A (now), B (+0x8c, before the step), velocity v; box half size h."""
    d = (E[0] - S[0], E[1] - S[1])
    n = (d[1], -d[0])
    if math.hypot(A[0] - S[0], A[1] - S[1]) > math.hypot(*d):
        return False
    if (A[0] - S[0]) * d[0] + (A[1] - S[1]) * d[1] < 0:
        return False
    side = lambda p: n[0] * (p[0] - S[0]) + n[1] * (p[1] - S[1])
    cross = lambda p, q: (side(p) > 0 > side(q)) or (side(p) < 0 < side(q))
    sx = h if v[0] > 0 else -h if v[0] < 0 else 0.0                              # swept, leading edge x
    if cross((B[0] + sx, B[1]), (A[0] + sx, A[1])):
        return True
    sy = h if v[1] > 0 else -h if v[1] < 0 else 0.0                              # swept, leading edge y
    if cross((B[0], B[1] + sy), (A[0], A[1] + sy)):
        return True
    return cross((A[0] + h, A[1]), (A[0] - h, A[1])) or cross((A[0], A[1] + h), (A[0], A[1] - h))


def wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def stats(v) -> str:
    v = np.abs(np.asarray(v, float))
    if not len(v):
        return "n=0"
    return (f"n={len(v)} median={np.median(v):.5f} p95={np.percentile(v, 95):.5f} max={v.max():.5f} "
            f"<=0.001: {np.mean(v <= 0.001) * 100:.1f}%")


# ---- data ----------------------------------------------------------------------------------------------------
def placed_angles(level: str) -> dict:
    """Exact placed (x, y) -> angle of the level's objects from the .smbl (hz rows round to 0.01)."""
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import smbl_extract as X
        objs = X.parse(os.path.join(ROOT, "imported", "smbl_raw", "Levels", level + ".smbl"))["objects"]
        return {(round(o["x"], 2), round(o["y"], 2)): f32(o["angle"]) for o in objs}
    except Exception:
        return {}


def load_level(run: str, name: str):
    z = np.load(os.path.join(run, "wm_data", "levels", name + ".npz"), allow_pickle=True)
    m = json.loads(str(z["meta"]))
    words = z["words"]                                     # row 0 = bottom (game/level.py)
    solid = (words & 0xFF) != 0                            # the ray cast's test: the cell word's low byte
    crumble = solid & (((words >> 8) & 1) == 1) & ((words & 0xFF) == 0x64)   # crumble blocks (break away)
    return (m["W"], m["H"], m["ox"], m["oy"], solid), (m["W"], m["H"], m["ox"], m["oy"], solid & ~crumble), m


def waves(run: str, level: str):
    for f in sorted(glob.glob(os.path.join(run, "wm_data", "wave_*.npz"))):
        w = np.load(f, allow_pickle=True)
        meta = json.loads(str(w["meta"]))
        if meta.get("level") != level:
            continue
        world = json.load(gzip.open(f.replace(".npz", "_world.json.gz")))
        yield f, meta, w["rows"], world


def main(runs):
    for run in runs:
        level = "4-16"
        if not os.path.exists(os.path.join(run, "wm_data", "levels", level + ".npz")):
            continue
        lvl, lvl_nc, lmeta = load_level(run, level)
        placed = placed_angles(level)
        print(f"== {run}  level {level}")
        err = {(lag, k): [] for lag in (0, 1, 2) for k in ("1/18", "0.1")}
        err_eye = []
        one_step = []
        len_err, end_err, len_crumble = [], [], 0
        first_on, offs, on_frames = Counter(), 0, 0
        deaths, deaths_beam, hits_alive, steps_checked = 0, 0, 0, 0
        f0_eye_ok, f0_n = 0, 0
        n_waves = 0
        for fn, meta, rows, world in waves(run, level):
            n_waves += 1
            ids = world[0]["ids"]
            beams = [i for i, s in enumerate(ids) if s.startswith("laser")]
            eyes = [i for i, h in enumerate(world[0]["hz"]) if i not in beams and h[2] == 16.85]
            if len(beams) != len(eyes):
                continue
            # pair each beam with its eye: the beam row's segment starts at the eye centre (checked on frame 1)
            pairs = []
            for b in beams:
                hb = world[1]["hz"][b]
                e = min(eyes, key=lambda i: abs(math.hypot(hb[0] - world[1]["hz"][i][0], hb[1] - world[1]["hz"][i][1]) - hb[2] / 2))
                pairs.append((e, b))
            r0 = rows[rows["agent"] == 0]
            pos = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in r0}
            vel = {int(r["frame"]): (float(r["vx"]), float(r["vy"])) for r in r0}
            frames = [fr["f"] for fr in world]
            for e, b in pairs:
                ex, ey = world[0]["hz"][e][0], world[0]["hz"][e][1]
                th0 = placed.get((round(ex, 2), round(ey, 2)))
                if th0 is None:
                    th0 = f32(world[0]["hz"][e][6])
                f0_n += 1
                f0_eye_ok += abs(world[0]["hz"][e][6] - float(th0)) < 0.006
                # (a) free-run turn
                for lag in (0, 1, 2):
                    for kname, k in (("1/18", f32(1.0) / f32(18)), ("0.1", f32(0.1))):
                        th = th0
                        for fr in world[1:]:
                            f = fr["f"]
                            tgt = pos.get(max(f - lag, 0))
                            if tgt is None:
                                break
                            th = aim_step(th, (ex, ey), tgt, k)
                            hb = fr["hz"][b]
                            meas = math.atan2(hb[1] - ey, hb[0] - ex)
                            err[(lag, kname)].append(wrap(meas - float(th)))
                            if lag == 0 and kname == "1/18":
                                err_eye.append(wrap(fr["hz"][e][6] - float(th)))
                # one step from the measured previous beam direction (lag 0, k 1/18)
                for j in range(2, len(world)):
                    f = world[j]["f"]
                    hp, hb = world[j - 1]["hz"][b], world[j]["hz"][b]
                    prev = f32(math.atan2(hp[1] - ey, hp[0] - ex))
                    th = aim_step(prev, (ex, ey), pos[f], f32(1.0) / f32(18))
                    one_step.append(wrap(math.atan2(hb[1] - ey, hb[0] - ex) - float(th)))
                # (b) ray cast at the measured direction
                for fr in world[1:]:
                    hb = fr["hz"][b]
                    th = f32(math.atan2(hb[1] - ey, hb[0] - ex))
                    px, py = ray_cast((ex, ey), th, lvl)
                    plen = math.hypot(px - ex, py - ey)
                    if abs(plen - hb[2]) > 0.05:           # a crumble block on the way may have broken this wave
                        qx, qy = ray_cast((ex, ey), th, lvl_nc)
                        if abs(math.hypot(qx - ex, qy - ey) - hb[2]) <= 0.05:
                            len_crumble += 1
                            continue
                    len_err.append(plen - hb[2])
                    mx, my = 2 * hb[0] - ex, 2 * hb[1] - ey
                    end_err.append(math.hypot(px - mx, py - my))
                # (c) timing: first frame whose beam row carries a kill shape, and any on -> off afterwards
                on = [bool(fr["hz"][b][7]) for fr in world]
                if any(on[1:]):
                    j = on.index(True, 1)
                    first_on[frames[j]] += 1
                    on_frames += sum(on[j:])
                    offs += sum(1 for q in range(j + 1, len(on)) if on[q - 1] and not on[q])
                else:
                    first_on[f"none<={frames[-1]}"] += 1
            # (d) kill test on every step: world update of frame f, then the agent's step f (pos f -> f+1)
            last = max(pos)
            died = (meta.get("status") or [0])[0] == 1
            for fr in world[1:]:
                f = fr["f"]
                if f + 1 not in pos:
                    continue
                hit = False
                for e, b in pairs:
                    hb = fr["hz"][b]
                    if not hb[7]:
                        continue
                    _, sx0, sy0, sx1, sy1, _ = hb[7][0]
                    hit |= beam_kills((sx0, sy0), (sx1, sy1), pos[f + 1], pos[f], vel[f + 1])
                steps_checked += 1
                is_death = died and f + 1 == last
                deaths += is_death
                deaths_beam += is_death and hit
                hits_alive += hit and not is_death
        print(f"waves: {n_waves}")
        print(f"frame 0 eye angle == placed angle (to 0.005): {f0_eye_ok}/{f0_n}  (the reset restores it)")
        print("(a) beam direction - free-run prediction from the placed angle, target = agent row at f - lag:")
        for (lag, k), v in err.items():
            print(f"    lag {lag} k {k:>4}: {stats(v)}")
        print(f"    recorded eye angle (0.01 rounding) - prediction (lag 0, 1/18): {stats(err_eye)}")
        print(f"    one step from the measured previous direction (lag 0, 1/18): {stats(one_step)}")
        odd = [v for v in one_step if abs(v) > 0.003]
        print(f"    one-step frames off by more than 0.003 rad (a turn the game skipped or made at the 1/1000 sine "
              f"comparison): {len(odd)} of {len(one_step)}")
        print(f"(b) beam length: predicted - recorded: {stats(len_err)}  signed mean {np.mean(len_err):+.4f}")
        print(f"    frames whose recorded length matches only with the crumble cells gone (excluded above): {len_crumble}")
        print(f"    beam end distance predicted vs recorded: {stats(end_err)}")
        print(f"(c) first lethal (kill shape) frame per eye-wave: {dict(first_on)}; on -> off seen: {offs} "
              f"(over {on_frames} on frames)")
        print(f"(d) kill test: {steps_checked} steps; deaths {deaths}, explained by a beam hit {deaths_beam}; "
              f"beam hits without a death {hits_alive}")


def shots_check(run: str, levels=("3-1", "3-4")):
    """(e) Straight shots (kinds 0/1: 360 units/s): per-frame displacement and birth distance from the shooter."""
    steps, births = [], Counter()
    for lv in levels:
        for fn, meta, rows, world in waves(run, lv):
            prev = {}
            for fr in world:
                cur = {s[0]: (s[1], s[2]) for s in fr.get("sh", [])}
                for k, p in cur.items():
                    if k in prev:
                        steps.append(math.hypot(p[0] - prev[k][0], p[1] - prev[k][1]))
                    elif fr["f"] > 0:
                        births[round(min(math.hypot(p[0] - h[0], p[1] - h[1]) for h in fr["hz"]), 1)] += 1
                prev = cur
    if not steps:
        return
    s = np.asarray(steps)
    print(f"== {run}  levels {', '.join(levels)}: straight shots")
    print(f"(e) per-frame displacement: n={len(s)} median={np.median(s):.4f} (360 units/s x 1/60 s = 6.0); "
          f"within 0.02 of 6.0: {np.mean(np.abs(s - 6.0) < 0.02) * 100:.2f}%")
    top = births.most_common(3)
    print(f"    birth distance from the nearest hazard centre (most common): {top} of {sum(births.values())} births")


def shot_count(run: str, levels=("4-15", "4-16")):
    """launcher2 levels: recorded shot entries (realgame.js shots() skips kind 5, so 0 is expected)."""
    for lv in levels:
        n_w = n_sh = 0
        for fn, meta, rows, world in waves(run, lv):
            n_w += 1
            n_sh += sum(len(fr.get("sh", [])) for fr in world)
        print(f"    {lv}: {n_w} waves, {n_sh} recorded shot entries")


if __name__ == "__main__":
    os.chdir(ROOT)
    runs = sys.argv[1:] or [os.path.join("runs", "online_check11")]
    main(runs)
    for r in runs:
        print(f"== {r}  launcher2 levels")
        shot_count(r)
    if not sys.argv[1:]:
        shots_check(os.path.join("runs", "online_check01"))
