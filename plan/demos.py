"""The PPO agents' best finishes (realgame.py's runs/real_*/levels/<level>/best.json) as demos for the world model.

    python -m plan.demos inventory      # runs/demos/inventory.json + a short table
    python -m plan.demos recorded       # runs/demos/recorded.json: where the valid demos' rows already are

Each best.json holds the fastest finish of one run on one level: its per-frame action indices (mpc.candidates.ACTIONS),
the agent index and the run's wave count (the wm_data shard index is wave - 1; len(actions) = finish frame + 1;
the agent's status-2 row is at frame len(actions)). Verdicts:
- VALID_RECORDED: the run had the time-step lock, its wm_data holds that agent's rows of that wave, the path is
  continuous (frames consecutive, no jump over 2 tiles) and its finish row touches the level's goal box
  (contact_reach) at the recorded frame count (+-2).
- REPLAYABLE: locked, no recorded rows, actions present (the user can replay it with agent.py --demo).
- SUSPICIOUS: unlocked or lock unknown, S pressed on that level in that run (realgame.py's S saves
  <level>_model.pt and finishes the level the game's way), a finish away from the goal box, a teleport, frames
  far below every other run's finish on that level, or anything that cannot be checked.
- HELD_OUT: a held-out level (eval_levels.json), never used.

Lock evidence: the world frames' time step in the run's wm_data (16.667 ms every frame: locked); else runs
started before the lock existed (commit 154f3fb, 2026-11-02 16:51) were unlocked; else unknown.
"""
from __future__ import annotations

import datetime as _dt
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import contact_reach                                       # noqa: E402
from wm.buffer import eval_levels, STATUS_FINISHED        # noqa: E402

RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(RUNS, "demos")
HZ = 60
DT_LOCK = 1000.0 / HZ
DT_TOL = 0.05                 # ms
LOCK_SINCE = _dt.datetime(2026, 11, 2, 16, 51, 51)        # 154f3fb "feat(realgame): lock the game's time step"
JUMP_MAX = 40.0               # 2 tiles: more between consecutive frames is a teleport
FRAME_TOL = 2                 # finish row frame vs len(actions)
FAR_BELOW = 0.5               # frames < this x every other run's finish on the level: suspicious

R_HELD = "held-out level"
R_UNLOCKED = "unlocked (run predates the time-step lock)"
R_UNLOCKED_DT = "unlocked (wm_data time step varies)"
R_LOCK_UNKNOWN = "lock unknown (no wm_data, run after the lock existed)"
R_S = "S pressed on this level in this run"
R_NO_GOAL = "finish row away from the goal box"
R_NO_FINISH_ROW = "no finish row (cannot check the goal touch)"
R_FRAMES = "finish row frame differs from the action count"
R_TELEPORT = "teleport (jump over 2 tiles or missing frames)"
R_FAR_BELOW = "frames far below every other run's finish"
R_NO_ACTIONS = "no actions"
R_NO_GOAL_DATA = "no goal data (cannot check the goal touch)"


def _rel(p: str) -> str:
    return os.path.relpath(p, HERE).replace("\\", "/")


def run_start(run_dir: str) -> _dt.datetime | None:
    m = re.match(r"real_(\d{8}-\d{6})$", os.path.basename(run_dir.rstrip("/\\")))
    if m:
        return _dt.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
    times = [os.path.getmtime(p) for p in glob.glob(os.path.join(run_dir, "**", "*"), recursive=True)]
    return _dt.datetime.fromtimestamp(min(times)) if times else None


def shard_meta(path: str) -> dict:
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


def world_dts(npz_path: str) -> list:
    wp = npz_path[:-4] + "_world.json.gz"
    if not os.path.exists(wp):
        return []
    with gzip.open(wp, "rt", encoding="utf-8") as f:
        return [w["dt"] for w in json.load(f) if w.get("dt") is not None]


def level_goals(wm_dir: str, level: str):
    p = os.path.join(wm_dir, "levels", f"{level}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        g = json.loads(str(z["meta"])).get("goals") or []
    return np.array(g, np.float64).reshape(-1, 2) if g else None


def check_rows(r: np.ndarray, n_actions: int, goals, reach) -> list:
    """Reasons the agent's rows (one wave, sorted by frame) are not a clean finish by movement; [] if clean."""
    why = []
    if len(r) < 2 or (np.diff(r["frame"]) != 1).any() or \
            (np.hypot(np.diff(r["x"]), np.diff(r["y"])) > JUMP_MAX).any():
        why.append(R_TELEPORT)
    fin = np.nonzero(r["status"] == STATUS_FINISHED)[0]
    if not len(fin):
        return why + [R_NO_FINISH_ROW]
    last = r[fin[0]]
    if abs(int(last["frame"]) - n_actions) > FRAME_TOL:
        why.append(R_FRAMES)
    if goals is None or reach is None:
        return why + [R_NO_GOAL_DATA]
    dx, dy = last["x"] - goals[:, 0], last["y"] - goals[:, 1]
    if not contact_reach.inside(dx, dy, reach[0], reach[1]).any():
        why.append(R_NO_GOAL)
    return why


def inventory(runs_dir: str = RUNS) -> list:
    held = eval_levels()
    reach = contact_reach.load().get("goal")
    entries = []
    for bp in sorted(glob.glob(os.path.join(runs_dir, "real_*", "levels", "*", "best.json"))):
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(bp)))
        with open(bp, encoding="utf-8") as f:
            b = json.load(f)
        lv, acts = b.get("level") or os.path.basename(os.path.dirname(bp)), b.get("actions") or []
        e = {"level": lv, "run": os.path.basename(run_dir), "best": _rel(bp), "seconds": b.get("seconds"),
             "frames": len(acts), "wave": b.get("wave"), "agent": b.get("agent"), "locked": "unknown",
             "lock_evidence": None, "rows": False, "s_pressed": os.path.exists(os.path.join(run_dir, f"{lv}_model.pt")),
             "reasons": []}
        wm = os.path.join(run_dir, "wm_data")
        shards = sorted(glob.glob(os.path.join(wm, "wave_*.npz")))
        lv_shards = [p for p in shards if shard_meta(p).get("level") == lv] if shards else []
        match = None
        if lv_shards and isinstance(e["wave"], int) and isinstance(e["agent"], int):
            k = e["wave"] - 1                                  # realgame.py: shard index = run wave count - 1
            cand = [shards[k]] if 0 <= k < len(shards) else []
            cand += [p for p in lv_shards if p not in cand]
            for p in cand:
                m = shard_meta(p)
                st, fi, a = m.get("status") or [], m.get("finish") or [], e["agent"]
                if m.get("level") == lv and a < len(st) and st[a] == STATUS_FINISHED and a < len(fi) \
                        and fi[a] + 1 == e["frames"]:
                    match = p
                    break
        if lv_shards:                                          # lock from the recorded time step
            dts = world_dts(match) if match else [d for p in lv_shards[:3] for d in world_dts(p)]
            if dts:
                ok = np.abs(np.array(dts) - DT_LOCK) <= DT_TOL
                e["locked"] = "yes" if ok.all() else "no"
                e["lock_evidence"] = f"wm_data dt: {ok.mean():.1%} of {len(dts)} frames at {DT_LOCK:.3f} ms"
        if e["lock_evidence"] is None:
            t0 = run_start(run_dir)
            if t0 is not None and t0 < LOCK_SINCE:
                e["locked"], e["lock_evidence"] = "no", f"run started {t0:%Y-%m-%d %H:%M}, before the lock (154f3fb)"
            else:
                e["lock_evidence"] = "no wm_data for this level and no log"
        if match:
            with np.load(match) as z:
                r = z["rows"]
            r = np.sort(r[r["agent"] == e["agent"]], order="frame")
            e["rows"] = True
            e["wm_data"], e["shard"] = _rel(wm), os.path.basename(match)
            e["frame_range"] = [int(r["frame"][0]), int(r["frame"][-1])]
            e["reasons"] += check_rows(r, e["frames"], level_goals(wm, lv), reach)
        entries.append(e)

    for e in entries:                                          # comparisons and verdicts
        if e["level"] in held:
            e["verdict"], e["reasons"] = "HELD_OUT", [R_HELD]
            continue
        others = [o["frames"] for o in entries if o["level"] == e["level"] and o is not e and o["frames"]]
        if not e["frames"]:
            e["reasons"].append(R_NO_ACTIONS)
        if others and e["frames"] < FAR_BELOW * min(others):
            e["reasons"].append(R_FAR_BELOW)
        if e["locked"] == "no":
            e["reasons"].append(R_UNLOCKED_DT if e["rows"] or "dt" in (e["lock_evidence"] or "") else R_UNLOCKED)
        elif e["locked"] == "unknown":
            e["reasons"].append(R_LOCK_UNKNOWN)
        if e["s_pressed"] and not e["rows"]:                   # with rows the goal touch decides
            e["reasons"].append(R_S)
        if e["reasons"]:
            e["verdict"] = "SUSPICIOUS"
        else:
            e["verdict"] = "VALID_RECORDED" if e["rows"] else "REPLAYABLE"
    return entries


def _level_key(lv: str):
    m = re.match(r"(\d+)-(\d+)", lv)
    return (int(m.group(1)), int(m.group(2)), lv) if m else (99, 99, lv)


def table(entries: list) -> str:
    verdicts = ["VALID_RECORDED", "REPLAYABLE", "SUSPICIOUS", "HELD_OUT"]
    lines = [f"{'level':6} {'best valid s':>12}  " + "  ".join(f"{v[:5]:>5}" for v in verdicts)]
    for lv in sorted({e["level"] for e in entries}, key=_level_key):
        es = [e for e in entries if e["level"] == lv]
        ok = [e["seconds"] for e in es if e["verdict"] in ("VALID_RECORDED", "REPLAYABLE")]
        lines.append(f"{lv:6} {(f'{min(ok):.2f}' if ok else '-'):>12}  "
                     + "  ".join(f"{sum(e['verdict'] == v for e in es):>5}" for v in verdicts))
    reasons = {}
    for e in entries:
        if e["verdict"] == "SUSPICIOUS":
            for r in e["reasons"]:
                reasons[r] = reasons.get(r, 0) + 1
    lines += ["", "verdicts: " + ", ".join(f"{v} {sum(e['verdict'] == v for e in entries)}" for v in verdicts),
              "suspicious reasons (an entry can have several): "
              + ("; ".join(f"{r}: {n}" for r, n in sorted(reasons.items(), key=lambda t: -t[1])) or "-")]
    return "\n".join(lines)


def load_inventory(path: str = os.path.join(OUT, "inventory.json")) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["entries"]


def best_for_level(level: str, entries: list | None = None) -> dict | None:
    """The fastest VALID_RECORDED / REPLAYABLE entry of a level (agent.py --demo-level), or None."""
    es = [e for e in (entries if entries is not None else load_inventory())
          if e["level"] == level and e["verdict"] in ("VALID_RECORDED", "REPLAYABLE")]
    return min(es, key=lambda e: e["seconds"]) if es else None


def recorded(entries: list) -> list:
    """Where each VALID_RECORDED demo's rows are (no rows copied)."""
    return [{"wm_data": e["wm_data"], "shard": e["shard"], "agent": e["agent"], "frames": e["frame_range"],
             "level": e["level"], "seconds": e["seconds"], "run": e["run"]}
            for e in entries if e["verdict"] == "VALID_RECORDED"]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inventory", "recorded"])
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if args.cmd == "inventory":
        entries = inventory()
        with open(os.path.join(OUT, "inventory.json"), "w", encoding="utf-8") as f:
            json.dump({"about": __doc__.split("\n\n")[1], "entries": entries}, f, indent=1)
        print(table(entries))
        print(f"-> {_rel(os.path.join(OUT, 'inventory.json'))}")
    else:
        rec = recorded(load_inventory())
        with open(os.path.join(OUT, "recorded.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        print(f"{len(rec)} recorded demos on {len({r['level'] for r in rec})} levels -> "
              f"{_rel(os.path.join(OUT, 'recorded.json'))}")


if __name__ == "__main__":
    main()
