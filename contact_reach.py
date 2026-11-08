"""How close Meat Boy's centre must come to the goal (the princess) and to a key to touch it, measured from
recorded play (wm_data).

Contact convention (the same as plan/safety.py for hazards): a box (x, y, half w, half h, angle) is touched
when the agent's centre comes within BODY (realpath.BODY, 6 units) of it. A reach (rx, ry) is how far from
the piece's position the centre is when touching, along x and y; the box is (x, y, rx - BODY, ry - BODY, 0).
So the contact region is a rectangle of half size (rx, ry) with corners rounded by BODY.

Goal, measured:
- Touch positions: the agent row with status 2 (wm.buffer). realgame.js counts frame++ after each update,
  so the status-2 row at frame f is the state right after update f - 1, the update in which the goal was
  touched (realgame.js finishFrame, kept in the shard's meta "finish"; rows where the two disagree are
  skipped). Finishes on the wave's last update have no status-2 row and are not used.
- Positions without a touch: every alive row after the wave's first update (frame >= 1) is the state right
  after an update that did not touch; those within NEAR units of a goal are kept. Among them the frame before
  each touch (the pre-touch check) and agents at rest (|vx| <= REST_VX, vy == 0).
- Offsets are relative to the nearest goal piece (the level data's goals).
- Reach per axis: the largest |offset| of the touch positions (percentile 100), else the 99th, 98th, 95th,
  90th, 75th, the first whose contact region holds no position at rest without a touch and at most
  SAFE_INSIDE x (touch positions inside) moving positions without a touch. An agent at rest inside a too big
  region would wait there, believing it touches; a moving one touches an update later anyway.
Keys, measured the same way from the rows where an agent's keys mask gains bit k (relative to key k of the
level data; realgame.js picks keys after each update, by the end position). When no recorded level has keys
the table keeps the fallback: realgame.js's own rule (the centre within KEY_R = 22 units of the key piece)
as the largest rounded box inside that circle (half size (KEY_R - BODY) / sqrt 2), marked unmeasured.

    python smbrl/contact_reach.py --all          # every recorded run under runs/ (not fake_wm)
    python smbrl/contact_reach.py smbrl/runs/plan_20261103-152013/wm_data      # or chosen wm_data folders

The table is written only when the goal was measured from at least MIN_TOUCHES touches (an empty or
incomplete table is refused and the old one left as it is). A key measurement from earlier data is kept when
the given data has no key pickups.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "contact_reach.json")
BODY = 6.0                     # realpath.BODY
KEY_R = 22.0                   # realgame.js KEY_R: a key is picked when the centre is within this of it
PERCENTILES = (100, 99, 98, 95, 90, 75)
SAFE_INSIDE = 0.02             # moving positions without a touch inside, per touch position inside
REST_VX = 30.0                 # units per second: at rest when |vx| <= this and vy == 0
NEAR = 40.0                    # units: positions without a touch further from the piece are not kept
MIN_TOUCHES = 30

_cache = {"mtime": None, "table": {}}


def load() -> dict:
    """{"goal": (rx, ry), "key": (rx, ry)} from the table; empty without one. Re-read when the file changes."""
    try:
        mt = os.path.getmtime(TABLE)
    except OSError:
        return {}
    if _cache["mtime"] != mt:
        with open(TABLE, encoding="utf-8") as f:
            raw = json.load(f)
        _cache["table"] = {k: (float(raw[k]["reach_x"]), float(raw[k]["reach_y"])) for k in ("goal", "key")
                           if k in raw}
        _cache["mtime"] = mt
    return _cache["table"]


def box(kind: str, x: float, y: float):
    """The contact box (x, y, half w, half h, 0) of a goal or key piece at (x, y) (touched within BODY of it), or
    None without a table entry."""
    r = load().get(kind)
    if r is None:
        return None
    return (float(x), float(y), max(0.0, r[0] - BODY), max(0.0, r[1] - BODY), 0.0)


def inside(dx, dy, rx: float, ry: float) -> np.ndarray:
    """Offsets (dx, dy) from a piece within its contact region of reach (rx, ry) (the rounded box)."""
    ex = np.maximum(np.abs(np.asarray(dx, np.float64)) - (rx - BODY), 0.0)
    ey = np.maximum(np.abs(np.asarray(dy, np.float64)) - (ry - BODY), 0.0)
    return np.hypot(ex, ey) < BODY


def _nearest(x, y, pts):
    """Offsets of points (x, y) from the nearest of pts (M, 2)."""
    dx = x[:, None] - pts[None, :, 0]
    dy = y[:, None] - pts[None, :, 1]
    k = np.argmin(dx * dx + dy * dy, 1)
    i = np.arange(len(x))
    return dx[i, k], dy[i, k]


def collect(dirs: list, log=print) -> dict:
    """Touch positions, positions without a touch near the piece (with their speed) and the pre-touch positions,
    for the goal and the keys, from wm_data folders."""
    from wm.buffer import STATUS_ALIVE, STATUS_FINISHED, load_level, load_shards
    out = {k: {"touch": [], "free": [], "pre": []} for k in ("goal", "key")}
    n = {"mismatch": 0, "no_row": 0, "shards": 0}
    for d in dirs:
        levels = {}
        for meta, rows, world in load_shards(d):
            name = meta.get("level")
            if name not in levels:
                try:
                    m = load_level(d, name)[1]
                    levels[name] = (np.asarray(m.get("goals") or [], np.float64).reshape(-1, 2),
                                    np.asarray(m.get("keys") or [], np.float64).reshape(-1, 2))
                except OSError:
                    levels[name] = None
            if levels[name] is None or not len(rows):
                continue
            n["shards"] += 1
            goals, keys = levels[name]
            rows = np.sort(rows, order=["agent", "frame"])
            x, y = rows["x"].astype(np.float64), rows["y"].astype(np.float64)
            vx, vy = rows["vx"].astype(np.float64), rows["vy"].astype(np.float64)
            alive = (rows["status"] == STATUS_ALIVE) & (rows["frame"] >= 1)
            if len(goals):
                dx, dy = _nearest(x, y, goals)
                fin = meta.get("finish") or []
                touch = rows["status"] == STATUS_FINISHED
                for j in np.nonzero(touch)[0]:
                    i, f = int(rows["agent"][j]), int(rows["frame"][j])
                    u = fin[i] if i < len(fin) and fin[i] is not None and fin[i] >= 0 else f - 1
                    if u != f - 1:
                        n["mismatch"] += 1
                        touch[j] = False
                        continue
                    if j > 0 and rows["agent"][j - 1] == i and rows["frame"][j - 1] == f - 1:
                        out["goal"]["pre"].append((dx[j - 1], dy[j - 1]))
                n["no_row"] += sum(1 for i, u in enumerate(fin) if u is not None and u >= 0
                                   and not ((rows["agent"] == i) & (rows["status"] == STATUS_FINISHED)).any())
                out["goal"]["touch"] += list(zip(dx[touch], dy[touch]))
                near = alive & (np.abs(dx) < NEAR) & (np.abs(dy) < NEAR)
                out["goal"]["free"] += list(zip(dx[near], dy[near], vx[near], vy[near]))
            if len(keys):
                kk = rows["keys"].astype(np.int64)
                same = (rows["agent"][1:] == rows["agent"][:-1]) & (rows["frame"][1:] == rows["frame"][:-1] + 1)
                got = np.where(same, kk[1:] & ~kk[:-1], 0)
                hit = np.zeros(len(rows), bool)
                for j in np.nonzero(got)[0]:
                    for k in range(len(keys)):
                        if (got[j] >> k) & 1:
                            out["key"]["touch"].append((x[j + 1] - keys[k, 0], y[j + 1] - keys[k, 1]))
                            out["key"]["pre"].append((x[j] - keys[k, 0], y[j] - keys[k, 1]))
                            hit[j + 1] = True
                dx, dy = _nearest(x, y, keys)
                near = alive & ~hit & (np.abs(dx) < NEAR) & (np.abs(dy) < NEAR)
                out["key"]["free"] += list(zip(dx[near], dy[near], vx[near], vy[near]))
        log(f"[contact] {d}: {len(out['goal']['touch'])} goal touches, {len(out['key']['touch'])} key pickups so far")
    res = {k: {kk: np.asarray(vv, np.float64).reshape(-1, 4 if kk == "free" else 2) for kk, vv in v.items()}
           for k, v in out.items()}
    res["counts"] = n
    return res


def choose(touch: np.ndarray, free: np.ndarray, pre: np.ndarray) -> dict:
    """The reach (the decision rule in the module text) and its evidence."""
    ax, ay = np.abs(touch[:, 0]), np.abs(touch[:, 1])
    rest = (np.abs(free[:, 2]) <= REST_VX) & (free[:, 3] == 0)
    q = lambda a: {str(p): round(float(np.percentile(a, p)), 2) for p in (50, 75, 90, 95, 98, 99, 100)}
    row = {"touches": int(len(touch)), "dist_abs_dx": q(ax), "dist_abs_dy": q(ay),
           "dx_range": [round(float(touch[:, 0].min()), 2), round(float(touch[:, 0].max()), 2)],
           "dy_range": [round(float(touch[:, 1].min()), 2), round(float(touch[:, 1].max()), 2)],
           "free_near": int(len(free)), "candidates": []}
    pick = None
    for p in PERCENTILES:
        rx, ry = float(np.percentile(ax, p)), float(np.percentile(ay, p))
        t_in = int(inside(touch[:, 0], touch[:, 1], rx, ry).sum())
        f_in = inside(free[:, 0], free[:, 1], rx, ry)
        c = {"percentile": p, "reach_x": round(rx, 2), "reach_y": round(ry, 2), "touches_inside": t_in,
             "free_inside": int(f_in.sum()), "rest_inside": int((f_in & rest).sum()),
             "pre_inside": int(inside(pre[:, 0], pre[:, 1], rx, ry).sum()), "pre": int(len(pre))}
        row["candidates"].append(c)
        if pick is None and c["rest_inside"] == 0 and c["free_inside"] <= SAFE_INSIDE * max(t_in, 1):
            pick = c
    if pick is None:
        raise ValueError("no candidate reach keeps resting agents without a touch outside")
    row.update(reach_x=pick["reach_x"], reach_y=pick["reach_y"], percentile=pick["percentile"], measured=True,
               half_w=round(pick["reach_x"] - BODY, 2), half_h=round(pick["reach_y"] - BODY, 2))
    return row


def key_fallback() -> dict:
    h = (KEY_R - BODY) / math.sqrt(2.0)
    return {"measured": False, "reach_x": round(h + BODY, 2), "reach_y": round(h + BODY, 2), "half_w": round(h, 2),
            "half_h": round(h, 2), "rule": f"no key pickups in the data: realgame.js KEY_R = {KEY_R:g} (a circle at "
            "the end of each update), as the largest rounded box inside it"}


def measure(dirs: list, log=print) -> dict:
    """The table's goal and key rows from wm_data folders (raises ValueError when the goal cannot be measured)."""
    sys.path.insert(0, HERE)
    c = collect(dirs, log)
    g = c["goal"]
    if len(g["touch"]) < MIN_TOUCHES:
        raise ValueError(f"only {len(g['touch'])} goal touches with a status-2 row (need {MIN_TOUCHES})")
    goal = choose(g["touch"], g["free"], g["pre"])
    goal["skipped"] = {"finish_frame_mismatch": c["counts"]["mismatch"], "no_status2_row": c["counts"]["no_row"]}
    k = c["key"]
    key = choose(k["touch"], k["free"], k["pre"]) if len(k["touch"]) >= MIN_TOUCHES else None
    return {"goal": goal, "key": key, "key_pickups": int(len(k["touch"])), "shards": c["counts"]["shards"]}


def all_data() -> list[str]:
    """Every recorded world-model data folder under runs/ except the fake one."""
    return sorted(d for d in glob.glob(os.path.join(HERE, "runs", "*", "wm_data*"))
                  if glob.glob(os.path.join(d, "wave_*.npz")) and os.path.basename(os.path.dirname(d)) != "fake_wm")


def write(res: dict, dirs: list, path: str = TABLE) -> dict:
    """Write the table (atomically). Refuses (ValueError) an incomplete goal row; keeps an earlier measured key
    row when this data has none."""
    goal = res.get("goal") or {}
    for k in ("reach_x", "reach_y", "half_w", "half_h"):
        v = goal.get(k)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"goal row incomplete ({k} = {v!r}): the table is left as it was")
    if goal.get("touches", 0) < MIN_TOUCHES:
        raise ValueError("goal measured from too few touches: the table is left as it was")
    old = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    key = res.get("key")
    if key is None:
        key = old["key"] if (old.get("key") or {}).get("measured") else key_fallback()
    out = {"about": "goal and key contact reach (contact_reach.py): touched when the agent's centre is within "
                    "body of the box (x, y, reach_x - body, reach_y - body), as plan/safety.py for hazards",
           "body": BODY, "rule": {"percentiles": list(PERCENTILES), "safe_inside": SAFE_INSIDE, "rest_vx": REST_VX,
                                  "near": NEAR},
           "measured_from": sorted(os.path.relpath(d, HERE).replace("\\", "/") for d in dirs),
           "shards": res.get("shards"), "key_pickups": res.get("key_pickups", 0), "goal": goal, "key": key}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)
    return out


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    dirs = all_data() if args == ["--all"] else args
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing or not dirs:
        raise SystemExit(f"not a folder: {', '.join(missing) or '(none)'} (give real wm_data folders, or --all)")
    try:
        res = measure(dirs)
        out = write(res, dirs)
    except ValueError as e:
        raise SystemExit(f"{e}: {TABLE} not written")
    g, k = out["goal"], out["key"]
    print(f"goal: reach x {g['reach_x']} y {g['reach_y']} (p{g['percentile']} of {g['touches']} touches; box half "
          f"{g['half_w']} x {g['half_h']} + body {BODY:g})")
    for c in g["candidates"]:
        print(f"  p{c['percentile']:>3}: reach {c['reach_x']:5.2f} x {c['reach_y']:5.2f}  touches inside "
              f"{c['touches_inside']}/{g['touches']}, without touch inside {c['free_inside']} (at rest "
              f"{c['rest_inside']}), frame before the touch inside {c['pre_inside']}/{c['pre']}")
    print(f"key: reach x {k['reach_x']} y {k['reach_y']} " + ("(measured)" if k.get("measured") else f"({k['rule']})"))
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
