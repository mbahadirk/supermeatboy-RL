"""Frames per tile and the wall cling speed, measured on recorded play (read-only): the constants of the
time-to-go field T (plan/progress.py) and of the cling rule (plan/planner.py, Planner.value).

    python -m plan.measure_time <wm_data dirs...> [--json out.json] [--t-field]

Data: every alive recorded agent row (planned and explore agents) of the given wm_data folders whose frame
and the next one ran with the time-step lock (the world record's dt within LOCK_TOL of 16.667 ms), with the
level's tile words (wm_data/levels/<level>.npz) and its tile changes applied as they came. Velocities are
position differences per frame (units / frame, y up) between an agent's consecutive frames, never the rows'
vx / vy fields (the ratio of those fields to the differences is reported, as a check of their unit).
Support is plan.geometry.Terrain.support's rule (ground; wall: side != 0 and not ground; air: neither).

M1 frames per tile (TILE = 20 units):
- fx_ground = TILE / the 90th percentile of |dx| per frame on frames on the ground (this frame and the next)
  while moving (|dx| > MOVING); fx_air: the same in the air;
- fy_up = TILE / the mean upward speed (dy > 0) on frames in the air;
- fy_down = TILE / the mean downward speed on the frames of falls (consecutive airborne frames with dy < 0) that
  drop more than FALL_MIN (2 tiles) in all.
Medians: the same with the median speed (reported, not used).
M2 cling: frames on a wall with a new jump press (jump in the action, the key not held before), bucketed by
the downward speed arriving there (-dy of the frame before); per bucket the share followed within JUMP_WINDOW
frames by an upward (dy > 0), away-from-the-wall (dx * side < 0) step: a wall jump. CLING_SPEED = the lower
edge of the first bucket (slowest first) from which on the share stays below CLING_SHARE (80%); none: inf.
Buckets with fewer than MIN_PRESSES presses are reported but not used.
M3 (--t-field): T_support at the spawn (plan/progress.py, the level's first recorded wave's hazards as
plan/offline.py builds them) against the recorded best finish frames (best_times.json). Reported, not fitted.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game.level import TILE                                              # noqa: E402
from mpc.candidates import ACTIONS                                       # noqa: E402
from plan.geometry import Terrain                                        # noqa: E402

LOCK_DT, LOCK_TOL = 16.667, 0.05
MOVING = 0.05                      # units / frame: slower is standing
FALL_MIN = 2 * TILE                # a fall counts for fy_down when it drops more than this
JUMP_WINDOW = 3
CLING_SHARE = 0.8
MIN_PRESSES = 30
SPEED_EDGES = (-np.inf, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, np.inf)   # downward units / frame
ACT_JUMP = np.array([a[2] for a in ACTIONS], bool)
BEST_TIMES = os.path.join(HERE, "best_times.json")


def _level_words(d: str, name: str, cache: dict):
    key = (d, name)
    if key not in cache:
        cands = [os.path.join(d, "levels", f"{name}.npz")] + sorted(
            glob.glob(os.path.join(HERE, "runs", "*", "wm_data", "levels", f"{name}.npz")),
            key=lambda p: -os.path.getmtime(p))
        cache[key] = None
        for p in cands:
            if os.path.exists(p):
                z = np.load(p)
                meta = json.loads(str(z["meta"]))
                cache[key] = (z["words"].astype(np.uint32), float(meta["ox"]), float(meta["oy"]))
                break
    return cache[key]


def shards(dirs):
    """Per readable shard: (level, rows sorted by agent and frame, ground, wall side (both for every row;
    alive rows only meaningful), locked (the row's frame and the next ran with the lock))."""
    cache = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "wave_*.npz"))):
            try:
                z = np.load(p)
                meta = json.loads(str(z["meta"]))
                rows = np.sort(z["rows"], order=["agent", "frame"])
                with gzip.open(p[:-4] + "_world.json.gz", "rt", encoding="utf-8") as f:
                    world = json.load(f)
            except Exception:                             # a shard still being written is skipped
                continue
            name = meta.get("level")
            lw = _level_words(d, name, cache) if name else None
            if lw is None or not len(rows):
                continue
            words0, ox, oy = lw
            Hh, W = words0.shape
            dt = {int(w["f"]): float(w.get("dt") or 0.0) for w in world}
            tc = {int(w["f"]): w["tc"] for w in world if w.get("tc")}
            fr = rows["frame"].astype(np.int64)
            lock_f = np.array([abs(dt.get(int(f), 0.0) - LOCK_DT) < LOCK_TOL for f in range(int(fr.max()) + 2)])
            locked = lock_f[fr] & lock_f[fr + 1]
            g = np.zeros(len(rows), bool)
            side = np.zeros(len(rows), np.int8)
            cuts = sorted(tc)
            bounds = [-1] + cuts + [int(fr.max()) + 1]
            words = words0.copy()
            ter = Terrain(words0, ox, oy)
            for a, b in zip(bounds[:-1], bounds[1:]):
                if a >= 0:
                    for idx, word in tc[a]:
                        r, c = divmod(int(idx), W)
                        if 0 <= r < Hh:
                            words[r, c] = word
                    ter.update(words)
                m = np.nonzero((fr >= max(a, 0)) & (fr < b))[0]
                if len(m):
                    g[m], side[m] = ter.support(rows["x"][m], rows["y"][m])
            yield name, rows, g, side, locked


def measure(dirs) -> dict:
    sp = defaultdict(list)                       # "ground" / "air" |dx|, "up" dy, "down" -dy
    press_speed, press_ok = [], []
    ratio_x, ratio_y = [], []
    n_rows = n_shards = 0
    levels = set()
    for name, rows, g, side, locked in shards(dirs):
        n_shards += 1
        levels.add(name)
        n = len(rows)
        alive = rows["status"] == 0
        nxt = np.arange(1, n + 1).clip(max=n - 1)
        same = (rows["agent"][nxt] == rows["agent"]) & (rows["frame"][nxt] == rows["frame"] + 1) & \
            (np.arange(n) < n - 1)
        ok = alive & alive[nxt] & same & locked          # a step frame -> frame + 1, both alive, locked
        n_rows += int(ok.sum())
        dx = np.where(ok, rows["x"][nxt].astype(np.float64) - rows["x"], 0.0)
        dy = np.where(ok, rows["y"][nxt].astype(np.float64) - rows["y"], 0.0)
        air = ~g & (side == 0)
        gg, aa = ok & g & g[nxt], ok & air & air[nxt]
        mv = np.abs(dx) > MOVING
        sp["ground"].append(np.abs(dx[gg & mv]))
        sp["air"].append(np.abs(dx[aa & mv]))
        sp["up"].append(dy[aa & (dy > 0)])
        # falls: runs of consecutive airborne descending steps
        desc = aa & (dy < 0)
        idx = np.nonzero(desc)[0]
        if len(idx):
            brk = np.nonzero(np.diff(idx) != 1)[0]
            starts, ends = np.r_[0, brk + 1], np.r_[brk + 1, len(idx)]
            for s0, e0 in zip(starts, ends):
                k = idx[s0:e0]
                if -dy[k].sum() > FALL_MIN:
                    sp["down"].append(-dy[k])
        # field units: vx / vy against the position differences (moving frames)
        big = ok & (np.abs(dx) > 1.0)
        ratio_x.append(rows["vx"][big] / dx[big])
        big = ok & (np.abs(dy) > 1.0)
        ratio_y.append(rows["vy"][nxt][big] / dy[big])
        # cling: a new jump press on a wall, the downward speed arriving there
        prv = np.arange(-1, n - 1).clip(min=0)
        came = (rows["agent"][prv] == rows["agent"]) & (rows["frame"][prv] == rows["frame"] - 1) & alive[prv] & \
            locked[prv] & (np.arange(n) > 0)
        act = rows["action"].astype(np.int64)
        press = (act >= 0) & ACT_JUMP[np.clip(act, 0, len(ACTIONS) - 1)] & (rows["held"] == 0)
        wall = alive & ~g & (side != 0) & came & press
        if wall.any():
            w = np.nonzero(wall)[0]
            down = -(rows["y"][w].astype(np.float64) - rows["y"][prv[w]])
            jumped = np.zeros(len(w), bool)
            cur, chain = w.copy(), np.ones(len(w), bool)
            for _ in range(JUMP_WINDOW):
                nx_ = nxt[cur]
                chain &= ok[cur]
                ddx = rows["x"][nx_].astype(np.float64) - rows["x"][cur]
                ddy = rows["y"][nx_].astype(np.float64) - rows["y"][cur]
                jumped |= chain & (ddy > 0) & (ddx * side[w] < 0)
                cur = nx_
            press_speed.append(down)
            press_ok.append(jumped)
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in sp.items()}
    out = {"shards": n_shards, "steps": n_rows, "levels": sorted(levels)}

    def per_tile(v, how):
        if not len(v):
            return None
        s = {"p90": np.percentile(v, 90), "mean": np.mean(v), "median": np.median(v)}[how]
        return float(TILE / s) if s > 0 else None
    out["fx_ground"] = per_tile(cat["ground"], "p90")
    out["fx_air"] = per_tile(cat["air"], "p90")
    out["fy_up"] = per_tile(cat["up"], "mean")
    out["fy_down"] = per_tile(cat["down"], "mean")
    out["median"] = {"fx_ground": per_tile(cat["ground"], "median"), "fx_air": per_tile(cat["air"], "median"),
                     "fy_up": per_tile(cat["up"], "median"), "fy_down": per_tile(cat["down"], "median")}
    out["speeds"] = {k: {"n": int(len(v)), "p90": float(np.percentile(v, 90)) if len(v) else None,
                         "mean": float(np.mean(v)) if len(v) else None,
                         "median": float(np.median(v)) if len(v) else None} for k, v in cat.items()}
    rx = np.concatenate(ratio_x) if ratio_x else np.zeros(0)
    ry = np.concatenate(ratio_y) if ratio_y else np.zeros(0)
    out["field_per_difference"] = {"vx": float(np.median(rx)) if len(rx) else None,
                                   "vy_next": float(np.median(ry)) if len(ry) else None}
    spd = np.concatenate(press_speed) if press_speed else np.zeros(0)
    okk = np.concatenate(press_ok) if press_ok else np.zeros(0, bool)
    buckets = []
    for lo, hi in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:]):
        m = (spd >= lo) & (spd < hi)
        n = int(m.sum())
        buckets.append({"from": lo, "to": hi, "presses": n, "wall_jumps": int(okk[m].sum()),
                        "share": float(okk[m].mean()) if n else None})
    out["cling"] = buckets
    cling = np.inf
    used = [b for b in buckets if b["presses"] >= MIN_PRESSES and b["from"] >= 0]
    for i, b in enumerate(used):
        if all(u["share"] < CLING_SHARE for u in used[i:]):
            cling = b["from"]
            break
    out["CLING_SPEED"] = cling
    return out


def t_field(dirs) -> dict:
    """M3: T_support at the spawn per level (first recorded wave's hazards) vs best_times.json frames."""
    from plan.offline import waves_of
    from plan.measure_progress import level_from_data, planner_inputs
    from plan.progress import ProgressField
    best = {}
    if os.path.exists(BEST_TIMES):
        with open(BEST_TIMES, encoding="utf-8") as f:
            best = json.load(f)
    out = {}
    for d in dirs:
        names = []
        for p in sorted(glob.glob(os.path.join(d, "wave_*.npz"))):
            try:
                nm = json.loads(str(np.load(p)["meta"])).get("level")
            except Exception:
                continue
            if nm and nm not in names:
                names.append(nm)
        for nm in names:
            if nm in out:
                continue
            _, world = next(waves_of(d, nm))
            lv = level_from_data(d, nm)
            inp = planner_inputs(level_from_data(d, nm), world)
            pf = ProgressField(lv.words0, lv.ox, lv.oy, lv.spawn, lv.goals, inp["hazards"], inp["sweeps"],
                               inp["keys"], inp["bandages"], inp["portals"], lift=inp["lift"], blocks=inp["blocks"],
                               doors=inp["doors"])
            sx, sy = lv.spawn
            t = float(pf.times(np.array([sx]), np.array([sy]), 0, True)[0])
            v = float(pf.values(np.array([sx]), np.array([sy]), 0, True)[0])
            b = (best.get(nm) or {}).get("planned")
            out[nm] = {"T_support_spawn": t, "V_support_spawn": v, "best_frames": b["frames"] if b else None,
                       "ratio": round(b["frames"] / t, 3) if b and t > 0 and np.isfinite(t) else None,
                       "build_s": round(pf.t_build, 3), "time_field_s": round(pf.t_time, 3), "dir": d}
    return out


def report(r: dict) -> str:
    f = lambda v: "-" if v is None else f"{v:.2f}"
    s = r["speeds"]
    lines = [f"{r['shards']} shards, {r['steps']} locked alive steps, levels {', '.join(r['levels'])}",
             f"M1 frames per tile: fx_ground {f(r['fx_ground'])} (|dx| p90 {f(s['ground']['p90'])} u/f, n "
             f"{s['ground']['n']}), fx_air {f(r['fx_air'])} (p90 {f(s['air']['p90'])}, n {s['air']['n']}), fy_up "
             f"{f(r['fy_up'])} (mean dy {f(s['up']['mean'])}, n {s['up']['n']}), fy_down {f(r['fy_down'])} (mean "
             f"-dy {f(s['down']['mean'])}, n {s['down']['n']} frames of falls > {FALL_MIN:.0f} u)",
             "   medians: " + ", ".join(f"{k} {f(v)}" for k, v in r["median"].items()),
             f"   field / position difference (median): vx {f(r['field_per_difference']['vx'])}, vy (next frame) "
             f"{f(r['field_per_difference']['vy_next'])} (fields in units / s at 60 fps would be 60)",
             f"M2 wall jump share by downward speed arriving at the wall (u/f; presses >= {MIN_PRESSES} used):"]
    for b in r["cling"]:
        lines.append(f"   [{b['from']:g}, {b['to']:g}): {b['wall_jumps']}/{b['presses']}"
                     + (f" ({100 * b['share']:.0f}%)" if b["share"] is not None else ""))
    c = r["CLING_SPEED"]
    lines.append(f"   CLING_SPEED = {c:g} u/f" + (" (no bucket drops below 80%: inf)" if not np.isfinite(c) else ""))
    if r.get("t_field"):
        lines.append("M3 T_support(spawn) vs the best finish (frames; reported, not fitted):")
        for nm, t in r["t_field"].items():
            lines.append(f"   {nm}: T {f(t['T_support_spawn'])} (V {f(t['V_support_spawn'])}), best "
                         f"{t['best_frames'] if t['best_frames'] is not None else '-'}, best / T "
                         f"{f(t['ratio'])}; build {t['build_s']:.2f} s (time field {t['time_field_s']:.2f} s)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--t-field", action="store_true", help="also M3 (needs plan/progress.py's time field)")
    a = ap.parse_args(argv)
    r = measure(a.dirs)
    if a.t_field:
        r["t_field"] = t_field(a.dirs)
    print(report(r))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, default=float)
    return r


if __name__ == "__main__":
    main()
