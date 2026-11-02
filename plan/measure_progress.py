"""How well the progress field V (plan/progress.py) follows the recorded lives that FINISHED a level.

    python -m plan.measure_progress [wm_data dirs...]      (from the smbrl folder; default: the two runs below)

Per level, V is built once from the level data and the hazards of the level's longest recorded wave in that
folder (replayed as realgame.py does before it builds its routes: pieces that stay put, the sweeps of the
moving ones, fans, platforms, sealed pockets). Then, along every life that ended in a finish (a row with
status 2), V is looked up at each frame: V_support where the agent stands or holds a wall (flags bit 0x2
clear, or bit 0x200 set), else V_air; the leg from the keys it holds (the rows' `keys` bitmask).

Reported per level:
- **inf**: share of the frames where V is +inf (the field calls the cell a dead end, yet the agent was there
  and then finished); split by supported / airborne frames.
- **no drop in 48 frames**: share of the 48-frame windows (one starting at every frame) in which V did not
  drop (V at the end >= V at the start). Also with airborne frames scored at their landing (V at the next
  supported frame), the way the planner will score plan ends, and on the fastest quarter of the lives only
  (the most direct finishes: what is left there is the field, not wandering).
- V at the life's first frame and at its finish (medians), and V_support at the spawn (the route's cost).
- the field's build: mode, legs, states and edges per leg, build time, peak traced memory.
Writes runs/progress_report.md and prints it.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game.level import Level, hazard_boxes                        # noqa: E402
from realpath import Planner                                       # noqa: E402
from plan.progress import ProgressField, measure_build             # noqa: E402
from wm.buffer import load_level, STATUS_FINISHED                  # noqa: E402

DEFAULT = [os.path.join(HERE, "runs", "real_20261102-165351", "wm_data"),
           os.path.join(HERE, "runs", "real_20261103-012157", "wm_data")]
WINDOW = 48
REPORT = os.path.join(HERE, "runs", "progress_report.md")
SUPPORTED_CLEAR, SUPPORTED_SET = 0x2, 0x200


def level_from_data(d: str, name: str) -> Level:
    """As mpc/offline.level_from_data, without importing the trainer."""
    words, meta = load_level(d, name)
    info = dict(meta)
    info["name"] = name
    return Level(info, words.astype("<u4").tobytes())


def planner_inputs(lv: Level, world: list) -> dict:
    """Planner keyword inputs as realgame.py builds them (track_hazards, build_paths), from one recorded wave:
    the level's pieces that never moved, every position the moving ones were seen at (rounded to 4 units),
    fans, platforms (fixed ones are walls, moving ones doors) and the sealed pockets."""
    keep, ids, level_ids = list(lv.hz_keep), None, None
    start, moving, sweep = {}, set(), set()
    for rec in world:
        if "keep" in rec:
            keep = rec["keep"]
        if "ids" in rec:
            ids = rec["ids"]
        if rec.get("fx"):
            lv.set_fields(rec["fx"])
        if rec.get("pf"):
            lv.set_platforms(rec["pf"])
        hz = rec.get("hz", [])
        boxes, kept = hazard_boxes(hz, keep, lv.hz_grow)
        bid = [str(ids[k]) if ids is not None and k < len(ids) else str(k) for k in kept]
        if level_ids is None:
            level_ids = {str(ids[k]) if ids is not None and k < len(ids) else str(k) for k in range(len(hz))}
        for b, hid in zip(boxes, bid):
            if hid not in level_ids:
                continue
            if hid not in start:
                start[hid] = b
                continue
            x, y, hw, hh, a = b
            x0, y0, _, _, a0 = start[hid]
            if abs(x - x0) > 2 or abs(y - y0) > 2 or abs(a - a0) > 0.05:
                if hid not in moving:
                    moving.add(hid)
                    sweep.add((round(x0 / 4) * 4, round(y0 / 4) * 4, hw, hh, round(a0, 1)))
                sweep.add((round(x / 4) * 4, round(y / 4) * 4, hw, hh, round(a, 1)))
    static = [b for hid, b in start.items() if hid not in moving]
    seeds = [lv.spawn] + list(lv.goals) + list(lv.keys) + list(lv.bandages) + [p for pr in lv.portals for p in pr]
    lv.set_sealed(static, seeds)
    return {"hazards": static, "sweeps": sorted(sweep), "keys": lv.keys, "bandages": lv.bandages,
            "portals": list(lv.portals), "lift": lv.wind | lv.plat_support(),
            "blocks": lv.plat_blocks() | lv.sealed_haz | lv.sealed_wall, "doors": lv.plat_doors(),
            "n_static": len(static), "n_moving": len(moving)}


def build_field(d: str, name: str, world: list):
    lv = level_from_data(d, name)
    inp = planner_inputs(lv, world)
    args = (lv.words0, lv.ox, lv.oy, lv.spawn, lv.goals, inp["hazards"], inp["sweeps"], inp["keys"],
            inp["bandages"], inp["portals"])
    kw = {"lift": inp["lift"], "blocks": inp["blocks"], "doors": inp["doors"]}
    pf = ProgressField(*args, **kw)                                  # timed build
    _, peak = measure_build(Planner(*args, **kw))                    # the same build again, for its memory
    pf.mem_peak = peak
    pf.inputs = inp
    return lv, pf


def life_values(pf: ProgressField, life: np.ndarray):
    x, y = life["x"].astype(np.float64), life["y"].astype(np.float64)
    fl = life["flags"].astype(np.int64)
    sup = ((fl & SUPPORTED_CLEAR) == 0) | ((fl & SUPPORTED_SET) != 0)
    leg = pf.leg_of(life["keys"].astype(np.int64))
    return pf.values(x, y, leg, sup), sup


def no_drop(v: np.ndarray, frames: np.ndarray) -> tuple[int, int]:
    """(windows, windows where V did not drop): every frame t with a row at t + WINDOW."""
    j = np.searchsorted(frames, frames + WINDOW)
    ok = j < len(frames)
    ok[ok] &= frames[j[ok]] == frames[ok] + WINDOW
    a, b = v[ok], v[j[ok]]
    return int(ok.sum()), int((b >= a - 1e-9).sum())


def new_stats():
    return {"lives": 0, "frames": 0, "inf": 0, "inf_out": 0, "sup": 0, "inf_sup": 0, "inf_air": 0, "win": 0, "flat": 0,
            "win_s": 0, "flat_s": 0, "v0": [], "v1": [], "per_life": [], "field": None}


def measure(dirs: list, log=print) -> dict:
    per = defaultdict(new_stats)
    for d in dirs:
        t0 = time.time()
        shards = defaultdict(list)                  # level -> [(path, frames, finished)]
        for p in sorted(glob.glob(os.path.join(d, "wave_*.npz"))):
            meta = json.loads(str(np.load(p)["meta"]))
            shards[meta.get("level")].append((p, meta.get("frames") or 0, STATUS_FINISHED in (meta.get("status") or [])))
        for name, lst in shards.items():
            fins = [p for p, _, f in lst if f]
            if not fins:
                continue
            if not os.path.exists(os.path.join(d, "levels", f"{name}.npz")):
                log(f"[progress] {d}: no level data for {name}, skipped")
                continue
            ref = max(lst, key=lambda e: e[1])[0]    # the longest wave: the moving hazards' whole cycle
            wp = ref[:-4] + "_world.json.gz"
            world = json.load(gzip.open(wp, "rt", encoding="utf-8")) if os.path.exists(wp) else []
            lv, pf = build_field(d, name, world)
            st = per[name]
            if st["field"] is None:
                st["field"] = pf
            log(f"[progress] {name}: {pf.summary()}; {pf.inputs['n_static']} fixed hazards, "
                f"{pf.inputs['n_moving']} moving")
            for p in fins:
                rows = np.load(p)["rows"]
                for a in np.unique(rows["agent"][rows["status"] == STATUS_FINISHED]):
                    life = np.sort(rows[rows["agent"] == a], order="frame")
                    end = int(np.nonzero(life["status"] == STATUS_FINISHED)[0][0])
                    life = life[:end + 1]
                    v, sup = life_values(pf, life)
                    inf = ~np.isfinite(v)
                    c0 = np.floor((life["x"] - pf.ox) / 20.0)
                    r0 = np.floor((life["y"] - pf.oy) / 20.0)
                    st["inf_out"] += int((inf & ((c0 < 0) | (c0 >= pf.W) | (r0 < 0) | (r0 >= pf.H))).sum())
                    st["lives"] += 1
                    st["frames"] += len(v)
                    st["inf"] += int(inf.sum())
                    st["sup"] += int(sup.sum())
                    st["inf_sup"] += int((inf & sup).sum())
                    st["inf_air"] += int((inf & ~sup).sum())
                    fr = life["frame"].astype(np.int64)
                    n, flat = no_drop(v, fr)
                    st["win"] += n
                    st["flat"] += flat
                    nf = len(v)                                 # airborne frames scored at their landing:
                    idx = np.where(sup, np.arange(nf), nf)      # V at the next supported frame
                    idx = np.minimum.accumulate(idx[::-1])[::-1]
                    vl = np.where(idx < nf, v[np.minimum(idx, nf - 1)], v)
                    n, flat = no_drop(vl, fr)
                    st["win_s"] += n
                    st["flat_s"] += flat
                    n, flat = no_drop(v, fr)
                    st["per_life"].append((nf, n, flat))
                    st["v0"].append(v[0])
                    st["v1"].append(v[-1])
        log(f"[progress] {os.path.relpath(d, HERE)}: {time.time() - t0:.0f} s")
    return dict(per)


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "-"


def med(xs):
    xs = np.asarray(xs, np.float64)
    if not len(xs):
        return "-"
    m = float(np.median(xs))
    return "inf" if not np.isfinite(m) else f"{m:.1f}"


def report(per: dict, dirs: list, seconds: float) -> str:
    lines = ["# Progress field V on recorded finishes (plan/measure_progress.py)", "",
             f"Data: {', '.join(os.path.relpath(d, HERE) for d in dirs)}; {seconds:.0f} s.", "",
             "- V: the cost-to-go to the goal under realpath.Planner's jump model and step costs (about tiles), "
             "per leg; +inf: the field says the goal cannot be reached from there.",
             "- Lookup: V_support at supported frames (flags bit 0x2 clear or 0x200 set), else V_air (best "
             "airborne jump state of the cell).",
             f"- **No drop**: share of the {WINDOW}-frame windows (one per frame) where V at the end >= V at the "
             "start; **landing**: the same with an airborne frame scored at the next supported frame (its landing); "
             "**fastest quarter**: only the lives no longer than the level's 25th percentile life.",
             "- V start / finish: medians over the lives; route cost: V_support at the spawn.", "",
             "| Level | Mode | Lives | Frames | Supported frames | inf | inf outside the map | inf supported | inf air | No drop 48 | No drop 48 (landing) | "
             "No drop 48 (fastest quarter) | V start | V finish | Route cost |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    key = lambda n: tuple(int(p) if p.isdigit() else 0 for p in str(n).split("-")) + (str(n),)
    tot = new_stats()
    for name in sorted(per, key=key):
        st, pf = per[name], per[name]["field"]
        for k in ("lives", "frames", "inf", "inf_out", "sup", "inf_sup", "inf_air", "win", "flat", "win_s", "flat_s"):
            tot[k] += st[k]
        pl_ = np.array(st["per_life"], np.int64).reshape(-1, 3)
        fast = pl_[pl_[:, 0] <= np.percentile(pl_[:, 0], 25)] if len(pl_) else pl_
        fw, ff = int(fast[:, 1].sum()), int(fast[:, 2].sum())
        tot["fw"], tot["ff"] = tot.get("fw", 0) + fw, tot.get("ff", 0) + ff
        lines.append(f"| {name} | {pf.label} | {st['lives']} | {st['frames']} | {pct(st['sup'], st['frames'])} | "
                     f"{pct(st['inf'], st['frames'])} | "
                     f"{pct(st['inf_out'], st['frames'])} | "
                     f"{pct(st['inf_sup'], st['sup'])} | {pct(st['inf_air'], st['frames'] - st['sup'])} | "
                     f"{pct(st['flat'], st['win'])} | {pct(st['flat_s'], st['win_s'])} | {pct(ff, fw)} | {med(st['v0'])} | "
                     f"{med(st['v1'])} | {pf.route_cost:.1f} |")
    lines.append(f"| **all** | | {tot['lives']} | {tot['frames']} | {pct(tot['sup'], tot['frames'])} | "
                 f"{pct(tot['inf'], tot['frames'])} | "
                 f"{pct(tot['inf_out'], tot['frames'])} | "
                 f"{pct(tot['inf_sup'], tot['sup'])} | {pct(tot['inf_air'], tot['frames'] - tot['sup'])} | "
                 f"{pct(tot['flat'], tot['win'])} | {pct(tot['flat_s'], tot['win_s'])} | {pct(tot.get('ff', 0), tot.get('fw', 0))} "
                 "| | | |")
    lines += ["", "## Field builds", "",
              "| Level | Grid (H x W) | Legs | States per leg | Edges per leg | Build (s) | Route search (s) | "
              "Peak traced memory (MiB) | Fixed / moving hazards |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(per, key=key):
        pf = per[name]["field"]
        lines.append(f"| {name} | {pf.H} x {pf.W} | {pf.n_legs} | {max(pf.n_states)} | {max(pf.n_edges)} | "
                     f"{pf.t_build:.2f} | {pf.t_route:.2f} | {pf.mem_peak / 2 ** 20:.0f} | "
                     f"{pf.inputs['n_static']} / {pf.inputs['n_moving']} |")
    return "\n".join(lines) + "\n"


def main():
    dirs = sys.argv[1:] or DEFAULT
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f"not a folder: {', '.join(missing)}")
    dirs = [os.path.abspath(d) for d in dirs]
    t0 = time.time()
    per = measure(dirs)
    text = report(per, dirs, time.time() - t0)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
