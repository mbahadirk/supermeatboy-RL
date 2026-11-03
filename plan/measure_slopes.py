"""Can Meat Boy wall-slide / wall-jump on 45-degree slope faces and half blocks, and on which faces? Measured on
recorded play (read-only), no assumptions.

    python -m plan.measure_slopes [--runs smbrl/runs] [--json out.json] [--max-shards N]

Data: every runs/*/wm_data shard (real_* and plan_* runs; runs/fake_wm skipped), with that run's level tile
words (wm_data/levels/<level>.npz, else another run's recording of the level), tile changes applied as they
came (crumble blocks breaking).

Method: for every recorded alive agent frame where plan.geometry.Terrain says "wall" on a side and not
"ground", the solid point the wall probe hit (WALL_PROBE beside the centre at mid height; left wins when both)
is classified by its tile:
- full block; crumble; other special (lock / switch);
- half block: "outer face" (its solid half is on the agent's side: the cell edge, a full-height vertical
  face), "mid-cell face" (the solid half is on the far side: its full-height face at the cell's middle, the
  body partly inside the cell), "upper side" / "lower side" (a top / bottom half: a half-height vertical face);
- slope (by its open corner relative to the agent): "diagonal up" (open corner on the agent's side, at the
  top: a ramp face, facing up), "diagonal down" (open on the agent's side, at the bottom: an overhang face),
  "vertical back" (open corner away from the agent: the slope's full vertical side faces the agent).
Per class: frames; the share in each movement state (the wall-slide state is taken from the data: the state
most airborne frames against full blocks are in, checked to be rare in free fall; the game's flag 0x200 is
reported too, but in this data it is set on few rows, in ground states, so it is no wall flag); jump presses there
(jump in the action with the key not held before) and how many were followed within 3 frames by vy > 0 and
vx away from the wall (a wall jump); the vertical speed while touching (median, and of the descending frames
the mean), against free fall (frames with no support at all) as reference.
Also: slope and half cells per recorded level (and wall-probe contacts with them), and, from the game's own
level files when present (imported/smbl_raw/Levels, tools/smbl_extract.py), which levels have slope cells.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game.level import TILE                                              # noqa: E402
from mpc.candidates import ACTIONS                                       # noqa: E402
from plan.geometry import Terrain, WALL_PROBE, SHAPE_FULL, SHAPE_HALF, SHAPE_SLOPE   # noqa: E402
from realpath import tile_classes                                        # noqa: E402
from wm.features import WALL                                             # noqa: E402

RUNS = os.path.join(HERE, "runs")
GAME_LEVELS = os.path.join(HERE, "imported", "smbl_raw", "Levels")
SKIP_RUNS = ("fake_wm",)
JUMP_WINDOW = 3                    # frames after a press to see a wall jump
MIN_FRAMES, MIN_PRESSES = 300, 30  # below this a class cannot answer the question
CLASSES = ("full", "crumble", "special", "half outer face", "half upper side", "half lower side", "half mid-cell face", "slope diagonal up",
           "slope diagonal down", "slope vertical back", "other")
SLOPE_CLASSES = ("slope diagonal up", "slope diagonal down", "slope vertical back")
HALF_CLASSES = ("half outer face", "half upper side", "half lower side", "half mid-cell face")
ACT_JUMP = np.array([a[2] for a in ACTIONS], bool)
VY_BINS = np.arange(-1500.0, 1501.0, 1.0)  # units/s, for vy percentiles


def probe_class(ter: Terrain, cls: dict, x: np.ndarray, y: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Class index (CLASSES) of the cell the wall probe hit, for rows with side != 0."""
    px = x + side * WALL_PROBE
    c = np.floor((px - ter.ox) / TILE).astype(np.int64)
    r = np.floor((y - ter.oy) / TILE).astype(np.int64)
    ok = (c >= 0) & (c < ter.W) & (r >= 0) & (r < ter.H)
    out = np.full(len(x), CLASSES.index("other"), np.int8)
    cc, rr, sd = c[ok], r[ok], side[ok]
    sh = ter.shape[rr, cc]
    k = np.full(len(cc), CLASSES.index("other"), np.int8)
    k[sh == SHAPE_FULL] = CLASSES.index("full")
    nx, ny = ter.nx[rr, cc], ter.ny[rr, cc]
    half = sh == SHAPE_HALF
    k[half & (nx == -sd)] = CLASSES.index("half outer face")         # agent right of the cell (side -1): right half
    k[half & (nx == sd)] = CLASSES.index("half mid-cell face")
    k[half & (nx == 0) & (ny > 0)] = CLASSES.index("half upper side")
    k[half & (nx == 0) & (ny < 0)] = CLASSES.index("half lower side")
    oc = ter.open_corner[rr, cc]
    slope = sh == SHAPE_SLOPE
    open_right, open_top = (oc == 1) | (oc == 3), oc >= 2
    near = np.where(sd < 0, open_right, ~open_right)           # the open corner is on the agent's side
    k[slope & near & open_top] = CLASSES.index("slope diagonal up")
    k[slope & near & ~open_top] = CLASSES.index("slope diagonal down")
    k[slope & ~near] = CLASSES.index("slope vertical back")
    crumble, special = cls["crumble"][rr, cc], (cls["lock"] | cls["switch"] | cls["other"])[rr, cc]
    k[(sh != 0) & crumble] = CLASSES.index("crumble")
    k[(sh != 0) & special & ~crumble] = CLASSES.index("special")
    out[ok] = k
    return out


class Acc:
    """Sums per class (and the free-fall reference)."""

    def __init__(self):
        self.frames = Counter()
        self.state = defaultdict(Counter)
        self.wall_flag = Counter()
        self.presses = defaultdict(Counter)       # class -> movement state at the press -> presses
        self.wall_jumps = defaultdict(Counter)
        self.vy = defaultdict(lambda: np.zeros(len(VY_BINS) - 1, np.int64))   # histogram of vy
        self.desc_sum, self.desc_n = Counter(), Counter()
        self.by_level = defaultdict(Counter)      # level -> class -> frames
        self.flag_state = Counter()               # states of alive rows with flag 0x200 (any support)
        self.alive = 0

    def add(self, name, rows, press, jumped, level):
        self.frames[name] += len(rows)
        self.by_level[level][name] += len(rows)
        for s, n in zip(*np.unique(rows["state"], return_counts=True)):
            self.state[name][int(s)] += int(n)
        self.wall_flag[name] += int(((rows["flags"] & WALL) != 0).sum())
        if press.any():
            for s, n in zip(*np.unique(rows["state"][press], return_counts=True)):
                self.presses[name][int(s)] += int(n)
            for s, n in zip(*np.unique(rows["state"][press & jumped], return_counts=True)):
                self.wall_jumps[name][int(s)] += int(n)
        vy = rows["vy"].astype(np.float64)
        d = vy < 0
        self.desc_sum[name] += float(vy[d].sum())
        self.desc_n[name] += int(d.sum())
        self.vy[name] += np.histogram(np.clip(vy, VY_BINS[0], VY_BINS[-1] - 1), VY_BINS)[0]


def hist_pct(h: np.ndarray, p: float, lo: float = -np.inf, hi: float = np.inf):
    """The p-th percentile of a VY_BINS histogram, over the bins within [lo, hi) (bin left edges)."""
    left = VY_BINS[:-1]
    m = (left >= lo) & (left < hi)
    c = np.cumsum(h * m)
    if not c[-1]:
        return None
    return float(left[np.searchsorted(c, p / 100.0 * c[-1])])


def shard_paths(runs: str):
    for d in sorted(glob.glob(os.path.join(runs, "*", "wm_data"))):
        if os.path.basename(os.path.dirname(d)) in SKIP_RUNS:
            continue
        for p in sorted(glob.glob(os.path.join(d, "wave_*.npz"))):
            yield d, p


def level_words(runs: str, d: str, name: str, cache: dict):
    """(words, ox, oy) of a level: this run's recording, else the newest other one."""
    key = (d, name)
    if key in cache:
        return cache[key]
    cands = [os.path.join(d, "levels", f"{name}.npz")] + sorted(
        glob.glob(os.path.join(runs, "*", "wm_data", "levels", f"{name}.npz")), key=lambda p: -os.path.getmtime(p))
    out = None
    for p in cands:
        if os.path.exists(p) and os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p)))) not in SKIP_RUNS:
            z = np.load(p)
            meta = json.loads(str(z["meta"]))
            out = (z["words"].astype(np.uint32), float(meta["ox"]), float(meta["oy"]))
            break
    cache[key] = out
    return out


def after(arr, k):
    """arr shifted k rows up (row j gets row j + k), padded."""
    out = np.empty_like(arr)
    out[:len(arr) - k] = arr[k:]
    out[len(arr) - k:] = arr[-1] if len(arr) else 0
    return out


def measure(runs: str = RUNS, max_shards: int | None = None, game_levels: str = GAME_LEVELS) -> dict:
    acc = Acc()
    cache, used = {}, Counter()
    levels_seen = {}
    n_shards = n_rows = 0
    missing = Counter()
    unreadable = 0
    for d, p in shard_paths(runs):
        if max_shards is not None and n_shards >= max_shards:
            break
        try:                                          # a shard still being written is skipped
            z = np.load(p)
            meta = json.loads(str(z["meta"]))
            rows = np.sort(z["rows"], order=["agent", "frame"])
        except Exception:
            unreadable += 1
            continue
        name = meta.get("level")
        lw = level_words(runs, d, name, cache) if name else None
        if lw is None:
            missing[name] += 1
            continue
        words0, ox, oy = lw
        if not len(rows):
            continue
        H, W = words0.shape
        cls0 = tile_classes(words0)
        # tile changes only matter where special tiles exist (crumble / lock / switch)
        tc = {}
        if (((words0 >> 8) & 1).astype(bool) & ((words0 & 0xff) > 0)).any():
            wp = p[:-4] + "_world.json.gz"
            if os.path.exists(wp):
                try:
                    with gzip.open(wp, "rt", encoding="utf-8") as f:
                        tc = {int(w["f"]): w["tc"] for w in json.load(f) if w.get("tc")}
                except Exception:
                    unreadable += 1
                    continue
        levels_seen[name] = (words0, ox, oy)
        n_shards += 1
        n_rows += len(rows)
        used[os.path.basename(os.path.dirname(d))] += 1
        # the next rows of the same agent (consecutive frames) for the wall-jump check
        same = [(after(rows["agent"], k) == rows["agent"]) & (after(rows["frame"], k) == rows["frame"] + k)
                for k in range(1, JUMP_WINDOW + 1)]
        same = [s & (np.arange(len(rows)) < len(rows) - k) for k, s in enumerate(same, 1)]
        vx_n = [after(rows["vx"], k) for k in range(1, JUMP_WINDOW + 1)]
        vy_n = [after(rows["vy"], k) for k in range(1, JUMP_WINDOW + 1)]
        act = rows["action"].astype(np.int64)
        press = (act >= 0) & ACT_JUMP[np.clip(act, 0, len(ACTIONS) - 1)] & (rows["held"] == 0)
        alive = rows["status"] == 0
        # spans of frames with one tile state
        cuts = sorted(f for f in tc)
        bounds = [-1] + cuts + [int(rows["frame"].max()) + 1]
        words = words0.copy()
        ter = Terrain(words0, ox, oy)
        for a, b in zip(bounds[:-1], bounds[1:]):
            if a >= 0:
                for idx, word in tc[a]:
                    r, c = divmod(int(idx), W)
                    if 0 <= r < H:
                        words[r, c] = word
                ter.update(words)
            # tile changes in a frame's message apply to that frame's state (as geometry.measure does)
            m = alive & (rows["frame"] >= max(a, 0)) & (rows["frame"] < b)
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            sub = rows[idx]
            g, side = ter.support(sub["x"], sub["y"])
            acc.alive += len(sub)
            fl = (sub["flags"] & WALL) != 0
            for s, n in zip(*np.unique(sub["state"][fl], return_counts=True)):
                acc.flag_state[int(s)] += int(n)
            free = ~g & (side == 0)
            nf = np.zeros(int(free.sum()), bool)          # (no wall: no wall jumps counted)
            acc.add("free fall (no support)", sub[free], nf, nf, name)
            wall = ~g & (side != 0)
            if not wall.any():
                continue
            wi = idx[wall]
            sd = side[wall].astype(np.int64)
            k = probe_class(ter, cls0, sub["x"][wall].astype(np.float64), sub["y"][wall].astype(np.float64), sd)
            jumped = np.zeros(len(wi), bool)
            for j in range(JUMP_WINDOW):
                jumped |= same[j][wi] & (vy_n[j][wi] > 0) & (vx_n[j][wi] * sd < 0)
            for ci, cname in enumerate(CLASSES):
                sel = k == ci
                if sel.any():
                    acc.add(cname, rows[wi[sel]], press[wi[sel]], jumped[sel], name)
    # the wall-slide state: the state most airborne frames against full blocks are in
    full = acc.state["full"]
    slide_state = full.most_common(1)[0][0] if full else None
    ff, nff = acc.state["free fall (no support)"], acc.frames["free fall (no support)"]
    flag_total = sum(acc.flag_state.values())
    table = {}
    for name in CLASSES + ("free fall (no support)",):
        n = acc.frames[name]
        if not n:
            continue
        h = acc.vy[name]
        pr, wj = sum(acc.presses[name].values()), sum(acc.wall_jumps[name].values())
        prs, wjs = acc.presses[name][slide_state], acc.wall_jumps[name][slide_state]
        table[name] = {
            "frames": n,
            "slide_state_share": round(acc.state[name][slide_state] / n, 4) if slide_state is not None else None,
            "wall_flag_share": round(acc.wall_flag[name] / n, 4),
            "top_states": [[s, round(c / n, 3)] for s, c in acc.state[name].most_common(3)],
            "jump_presses": pr, "wall_jumps": wj, "wall_jump_share": round(wj / pr, 4) if pr else None,
            "presses_in_slide_state": prs, "wall_jumps_from_slide_state": wjs,
            "slide_wall_jump_share": round(wjs / prs, 4) if prs else None,
            "vy_median": hist_pct(h, 50),
            "descending_share": round(acc.desc_n[name] / n, 3),
            "vy_desc_mean": round(acc.desc_sum[name] / acc.desc_n[name], 1) if acc.desc_n[name] else None,
            "vy_desc_median": hist_pct(h, 50, hi=0.0), "vy_desc_10": hist_pct(h, 10, hi=0.0)}
    # slope / half cells per recorded level, and wall-probe contacts with them
    per_level = {}
    for name, (w, _, _) in sorted(levels_seen.items()):
        k = tile_classes(w)
        bl = acc.by_level.get(name, Counter())
        per_level[name] = {"slope_cells": int(k["slope"].sum()), "half_cells": int(k["half"].sum()),
                           "slope_contact_frames": int(sum(bl[c] for c in SLOPE_CLASSES)),
                           "half_contact_frames": int(sum(bl[c] for c in HALF_CLASSES))}
    enough = {c: bool(acc.frames[c] >= MIN_FRAMES and sum(acc.presses[c].values()) >= MIN_PRESSES)
              for c in SLOPE_CLASSES + HALF_CLASSES}
    return {"runs": dict(used), "shards": n_shards, "rows": n_rows, "levels_without_tiles": dict(missing),
            "unreadable_shards": unreadable,
            "slide_state": slide_state,
            "slide_state_share_full": round(full[slide_state] / acc.frames["full"], 4) if full else None,
            "slide_state_share_free_fall": round(ff[slide_state] / nff, 4) if nff and full else None,
            "wall_flag_rows": round(flag_total / acc.alive, 4) if acc.alive else None,
            "wall_flag_states": [[s, c] for s, c in acc.flag_state.most_common(5)],
            "classes": table, "levels": per_level, "enough_data": enough,
            "game_levels_with_slopes": game_levels_with_slopes(game_levels)}


def game_levels_with_slopes(lvl_dir: str = GAME_LEVELS) -> dict | None:
    """{level file: slope cells} of the game's chapter levels (collision layer), or None without the files."""
    if not os.path.isdir(lvl_dir):
        return None
    sys.path.insert(0, os.path.join(HERE, "tools"))
    try:
        import smbl_extract as X
    except ImportError:
        return None
    out = {}
    for c in X.chapters(lvl_dir):
        p = os.path.join(lvl_dir, c["file"] + ".smbl")
        if c["file"] in out or not os.path.exists(p):
            continue
        try:
            lv = X.parse(p)
        except Exception:                                 # an unreadable file is skipped, not guessed
            continue
        w = np.asarray(lv["layers"][0], np.uint32).reshape(lv["H"], lv["W"])
        n = int(tile_classes(w)["slope"].sum())
        if n:
            out[c["file"]] = n
    return out


def report(r: dict) -> str:
    out = []
    w = out.append
    w(f"{r['shards']} shards, {r['rows']} agent rows from {len(r['runs'])} runs"
      + (f"; no tile data for {r['levels_without_tiles']}" if r["levels_without_tiles"] else "")
      + (f"; {r['unreadable_shards']} unreadable shards skipped" if r["unreadable_shards"] else ""))
    w(f"wall-slide state (most airborne frames against full blocks): {r['slide_state']} "
      f"({100 * (r['slide_state_share_full'] or 0):.0f}% of them, {100 * (r['slide_state_share_free_fall'] or 0):.1f}% "
      f"of free-fall frames); flag 0x200 on {100 * (r['wall_flag_rows'] or 0):.2f}% of alive rows, states "
      f"{r['wall_flag_states']}")
    w("airborne frames with a wall probe hit, by what the probe hit:")
    w(f"  {'class':22s} {'frames':>9s} {'slide':>6s} {'presses':>8s} {'wall jumps':>14s} {'from slide':>16s} "
      f"{'vy med':>7s} {'desc':>5s} {'desc mean/med/10%':>18s}  top states")

    def nj(a, b, sh):
        return f"{a}/{b} ({100 * sh:.0f}%)" if sh is not None else "-"

    def v(x):
        return "-" if x is None else f"{x:.0f}"
    for name, t in r["classes"].items():
        w(f"  {name:22s} {t['frames']:9d} {100 * (t['slide_state_share'] or 0):5.0f}% {t['jump_presses']:8d} "
          f"{nj(t['wall_jumps'], t['jump_presses'], t['wall_jump_share']):>14s} "
          f"{nj(t['wall_jumps_from_slide_state'], t['presses_in_slide_state'], t['slide_wall_jump_share']):>16s} "
          f"{v(t['vy_median']):>7s} {100 * t['descending_share']:4.0f}% "
          f"{v(t['vy_desc_mean']) + '/' + v(t['vy_desc_median']) + '/' + v(t['vy_desc_10']):>18s}  {t['top_states']}")
    w("  (slide: share of frames in the wall-slide state; presses: new jump presses there; wall jumps: followed "
      f"within {JUMP_WINDOW} frames by vy > 0 and vx away from the wall; from slide: the presses made in the "
      "wall-slide state; vy in units/s, y up; desc: share descending, their mean / median / fastest 10%)")
    w("recorded levels: slope cells / half cells; airborne wall-probe frames on slopes / halves:")
    w("  " + ", ".join(f"{n} {L['slope_cells']}/{L['half_cells']}: {L['slope_contact_frames']}/{L['half_contact_frames']}"
                       for n, L in r["levels"].items()))
    few = [c for c, ok in r["enough_data"].items() if not ok]
    if few:
        w(f"too few recorded frames (< {MIN_FRAMES}) or jump presses (< {MIN_PRESSES}) to answer for: {', '.join(few)}")
        g = r["game_levels_with_slopes"]
        if g is None:
            w("  (the game's level files are not here: cannot list which levels have slopes)")
        else:
            rec = set(r["levels"])
            other = sorted(((k, v) for k, v in g.items() if k not in rec), key=lambda kv: -kv[1])
            w(f"  game levels with slope cells not recorded yet ({len(other)} of {len(g)} with slopes), most first: "
              + ", ".join(f"{k} ({v})" for k, v in other[:30]) + (" ..." if len(other) > 30 else ""))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=RUNS, help="the runs folder (default smbrl/runs)")
    ap.add_argument("--json", default=None, help="also write the numbers to this JSON file")
    ap.add_argument("--max-shards", type=int, default=None, help="stop after this many shards (a quick look)")
    a = ap.parse_args(argv)
    r = measure(a.runs, a.max_shards)
    print(report(r))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
    return r


if __name__ == "__main__":
    main()
