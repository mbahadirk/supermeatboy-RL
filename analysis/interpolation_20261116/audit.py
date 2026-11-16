"""Replay fixed recorded actions to audit prediction resolution; no training, planning or game control.

Run from smbrl: python analysis/interpolation_20261116/audit.py
Scope is online_check05 waves 1..57, even if that run later grows. Windows overlap: counts are diagnostic
checks, not saved decisions or an FPS estimate. Old behavior is the same Committer with every path sample
declared direct. Both sides use the same step-4 error tube and recorded geometry. Danger is tested in the
regression suite, not reconstructed here. No alternate actions or counterfactual survival are simulated.
"""
from __future__ import annotations

import ctypes
import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.level import platform_boxes
from plan.commit import Committer, kind_of
from plan.explain import Plan
from plan.geometry import Terrain

RUN = ROOT / "runs/online_check05"
OUT = Path(__file__).resolve().parent
MAX_WAVE = 57


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_memory():
    if os.name != "nt":
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak_wset", "wset", "peak_paged", "paged",
                                               "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak_wset


def main():
    # This is the historical audit of the first, now-reverted optimization. Never overwrite its evidence
    # by silently replaying a different controller under the old before/after labels.
    saved = OUT / "report.json"
    if saved.exists():
        expected = json.loads(saved.read_text(encoding="utf8"))["source_sha256"]
        if any(digest(ROOT / name) != value for name, value in expected.items()):
            raise SystemExit("Historical controller changed; use analysis/interpolation_safety_20261116/audit.py. "
                             "The original report.json is retained.")
    wall, cpu = time.perf_counter(), time.process_time()
    plans = []
    with (RUN / "plans.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            p = json.loads(line)
            if p["level"] != "4-4" or p["wave"] > MAX_WAVE:
                break                              # the original 4-4 visit only; later visits can reuse wave IDs
            plans.append(p)
    plan_digest = hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()
    by_wave = {}
    for p in plans:
        by_wave.setdefault(p["wave"], []).append(p)
    level_path = RUN / "wm_data/levels/4-4.npz"
    with np.load(level_path) as z:
        words, meta = z["words"], json.loads(str(z["meta"]))
    assert not meta.get("keys"), "This bounded audit has no private key geometry."
    terrain = Terrain(words, meta["ox"], meta["oy"])
    plats = []
    def support(x, y, vy=None, keys=0):
        assert not np.asarray(keys).any()
        ground, side = terrain.support(x, y, plats)
        return ground | (side != 0), ground, side
    geometry = SimpleNamespace(support=support, key_pickup=False, bounds=None)
    fields = {f.name for f in dataclasses.fields(Plan)}
    inputs, totals, errors, oracle, examples = [], {}, [], [], []
    def count(key, n=1):
        totals[key] = totals.get(key, 0) + int(n)
    for path in sorted((RUN / "wm_data").glob("wave_*.npz")):
        with np.load(path) as z:
            info = json.loads(str(z["meta"]))
            if info["level"] != "4-4" or info["wave"] > MAX_WAVE:
                break
            rows = z["rows"]
        assert info["level"] == "4-4"
        world_path = path.with_name(path.stem + "_world.json.gz")
        with gzip.open(world_path, "rt", encoding="utf8") as stream:
            world = {r["f"]: r for r in json.load(stream)}
        assert not any(r.get("tc") for r in world.values()), "Audit requires unchanged tiles."
        inputs.append({"wave": info["wave"], "shard": path.name, "sha256": digest(path),
                       "world_sha256": digest(world_path)})
        by = {(int(r["agent"]), int(r["frame"])): r for r in rows}
        for agent in np.unique(rows["agent"]):
            a = rows[(rows["agent"] == agent) & (rows["status"] == 0)]
            xy = np.stack([a["x"], a["y"]], axis=1)
            for i in range(len(a) - 4):
                if a["frame"][i + 4] - a["frame"][i] != 4:
                    continue
                truth = xy[i:i + 5]
                line = truth[0] + np.arange(5)[:, None] / 4 * (truth[4] - truth[0])
                oracle.append(float(np.linalg.norm(line - truth, axis=1)[1:4].max()))
        for raw in by_wave.get(info["wave"], []):
            p = Plan(**{k: v for k, v in raw.items() if k in fields})
            route = raw.get("best_route") or {}
            # Old logs lack path_step. Use the producer's rule: only a chosen, recorded real route is dense.
            p.path_step = raw.get("path_step", 1 if p.mode == "best-route" and route.get("judged") == "real"
                                  and route.get("used") else 4)
            start = by.get((p.agent, p.frame))
            if start is None or start["status"] != 0 or not p.path:
                continue
            if np.linalg.norm(np.array(p.path[0]) - [start["x"], start["y"]]) > 0.1:
                continue
            old, new = Committer(step=4), Committer(step=4)
            plats = platform_boxes(world.get(p.frame, {}).get("pf"))
            old.decided(p.agent, dataclasses.replace(p, path_step=1), p.frame, planner=geometry)
            new.decided(p.agent, p, p.frame, planner=geometry)
            count("plans"); count("dense_plans" if p.path_step == 1 else "coarse_plans")
            first_four = []
            for k in range(1, min(4, len(p.actions), len(p.path) - 1) + 1):
                before = by.get((p.agent, p.frame + k - 1))
                actual = by.get((p.agent, p.frame + k))
                if before is None or actual is None or actual["status"] != 0 or before["action"] != p.actions[k - 1]:
                    break
                if p.frame + k not in world:
                    break
                plats = platform_boxes(world[p.frame + k].get("pf"))
                xy = [float(actual["x"]), float(actual["y"])]
                if k <= 4:
                    first_four.append(float(np.linalg.norm(np.array(p.path[k]) - xy)))
                a = old.triggers(p.agent, p.frame + k, *xy, keys=int(actual["keys"]), planner=geometry)
                b = new.triggers(p.agent, p.frame + k, *xy, keys=int(actual["keys"]), planner=geometry)
                other = lambda tr: [t for t in tr if kind_of(t) != "surprise"]
                assert other(a) == other(b)
                count("followed_frame_checks")
                count("old_surprise", any(kind_of(t) == "surprise" for t in a))
                count("new_surprise", any(kind_of(t) == "surprise" for t in b))
                count("old_any_trigger", bool(a)); count("new_any_trigger", bool(b))
                count("checks_with_support", any(kind_of(t) == "support" for t in a))
                if k % p.path_step == 0:
                    assert a == b
                    count("direct_sample_checks_identical")
            if len(first_four) == 4 and p.path_step == 4:
                errors.append(first_four)
                if first_four[3] < 1 and max(first_four[:3]) > 3 and len(examples) < 8:
                    examples.append({"wave": info["wave"], "frame": p.frame, "errors": first_four,
                                     "chosen": p.chosen})
        if info["wave"] % 10 == 0:
            print(f"Audited wave {info['wave']}/{MAX_WAVE}", flush=True)
    e, o = np.asarray(errors), np.asarray(oracle)
    report = {"scope": "online_check05 original 4-4 visit, completed waves 1..57; first four recorded actions per plan",
              "limitations": ["Not a closed-loop evaluation, FPS measurement or forecast of saved decisions.",
                              "Danger omitted in replay; per-frame danger checked by regression tests.",
                              "Does not establish improved model physics or success rate.",
                              "Position-only errors between direct samples can be detected up to 3 frames later."],
              "plans_sha256_canonical": plan_digest, "level_sha256": digest(level_path), "inputs": inputs,
              "source_sha256": {name: digest(ROOT / name) for name in
                                 ["plan/commit.py", "plan/planner.py", "plan/explain.py"]},
              "counts": totals, "coarse_followed_four_frame_windows": len(e),
              "endpoint_under_1_intermediate_over_3": int(np.sum((e[:, 3] < 1) & (e[:, :3].max(1) > 3))),
              "errors_by_horizon_1_2_3_4": {"median": np.median(e, axis=0).tolist(),
                                          "p90": np.percentile(e, 90, axis=0).tolist()},
              "oracle_interpolation": {"windows": len(o), "median": float(np.median(o)),
                                       "p90": float(np.percentile(o, 90)), "fraction_over_3": float(np.mean(o > 3))},
              "examples": examples,
              "resources": {"wall_s": time.perf_counter() - wall, "cpu_s": time.process_time() - cpu,
                            "peak_process_bytes_including_imports": peak_memory(),
                            "children": 0, "timing_scope": "one audit run after imports; no planner benchmark"}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: report[k] for k in ["counts", "coarse_followed_four_frame_windows",
                                           "endpoint_under_1_intermediate_over_3", "resources"]}, indent=2))


if __name__ == "__main__":
    main()
