"""Replay online_check06 observations, not actions of a counterfactual agent.

Three trigger rules, the same committed plans and causal hazard forecasts: repository HEAD (before the
optimization), unconditional step gating (the first optimization), and restored per-frame feedback with
the current observed-position danger check.
Only the selected lives are timed/compared. Previous lives build FutureWorld's reference just as live.
No model inference, fitting, game attachment or memory writes. DeathMemory is omitted: the two selected
deaths are explained by native hazards. This cannot establish a changed win rate or FPS.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.level import platform_boxes
from mpc.future import FutureWorld
from plan.commit import Committer
from plan.explain import Plan
from plan.geometry import Terrain
from plan.measure_progress import level_from_data
from plan.planner import future_inputs
from plan.safety import level_bounds

RUN = ROOT / "runs/online_check06"
OUT = Path(__file__).resolve().parent
CASES = {("4-1", 5), ("4-2", 50)}
BASE = "a1e7f343e6f610dfff9d4ca5c575f500eef8b70b"


def module(source, name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(source, name, "exec"), mod.__dict__)
    return mod


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source = subprocess.check_output(["git", "show", f"{BASE}:plan/commit.py"], cwd=ROOT).decode()
    original = module(source, "original_commit").Committer
    needle = "if c.xs is not None and 1 <= k < len(c.xs):"
    assert source.count(needle) == 1
    gated_source = source.replace(needle, needle[:-1] + ' and k % getattr(p, "path_step", 1) == 0:')
    gated = module(gated_source, "unconditional_commit").Committer
    constructors = {"original": original, "unconditional": gated, "corrected": Committer}
    plans = {}
    with (RUN / "plans.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            d = json.loads(line)
            key = d["level"], d["wave"]
            if key in CASES:
                plans.setdefault(key, {})[d["frame"]] = d
    fields = {f.name for f in dataclasses.fields(Plan)}
    events = [json.loads(line) for line in (RUN / "events.jsonl").read_text().splitlines()]
    level_events = {e["level"]: e for e in events if e["kind"] == "level"}
    deaths = {(e["level"], e["wave"]): e for e in events if e["kind"] == "death"}
    fw, level, results, inputs = FutureWorld(), None, [], []
    totals = {name: {"checks": 0, "trigger_frames": 0, "surprise": 0, "danger": 0, "cpu_s": 0., "wall_s": 0.}
              for name in constructors}
    for shard in sorted((RUN / "wm_data").glob("wave_*.npz")):
        with np.load(shard) as z:
            meta = json.loads(str(z["meta"]))
            rows = z["rows"]
        key = meta["level"], meta["wave"]
        if level != key[0]:
            fw.new_level()
            level = key[0]
        else:
            fw.new_wave()
        lv = level_from_data(str(RUN / "wm_data"), level)
        if level_events[level].get("kill_box"):
            lv.set_kill_box(level_events[level]["kill_box"])
        assert not lv.keys, "Audit adapter has no private key geometry."
        terrain = Terrain(lv.words0, lv.ox, lv.oy)
        def support(x, y, vy=None, keys=0):
            ground, side = terrain.support(x, y, lv.plats)
            return ground | (side != 0), ground, side
        geometry = types.SimpleNamespace(support=support, key_pickup=False, bounds=level_bounds(lv))
        world_path = shard.with_name(shard.stem + "_world.json.gz")
        with gzip.open(world_path, "rt", encoding="utf8") as stream:
            world = json.load(stream)
        inputs.append({"file": shard.name, "sha256": digest(shard.read_bytes()),
                       "world_sha256": digest(world_path.read_bytes())})
        cms = {name: cls(step=4) for name, cls in constructors.items()}
        for cm in cms.values():
            # The second level inherited the measured error tube from the first. Keep it identical.
            tube = level_events[level]["tube"]
            cm.tube._e = np.array([tube[str(h)][0] for h in cm.tube.horizons] + list(cm.tube._e[len(tube):]))
        by = {int(r["frame"]): r for r in rows if r["agent"] == 0}
        active, followed = None, True
        for rec in world:
            f = rec["f"]
            if rec.get("tc"):
                lv.apply_tiles(rec["tc"])
                terrain.update(lv.words)
            if rec.get("pf"):
                lv.set_platforms(rec["pf"])
            dt = rec.get("dt") or 16.667
            fw.observe(f, rec.get("hz"), rec.get("pf"), rec.get("sh"), rec.get("keep"),
                       rec.get("fx", {}).get("fans"), dt, terrain)
            if key not in CASES:
                continue
            row = by[f]
            cached = []
            def boxes_now():
                if not cached:
                    cached.append(future_inputs(fw, f, 48, dt, deadly_only=True)[1])
                return cached[0]
            if f in plans[key]:
                active = Plan(**{k: v for k, v in plans[key][f].items() if k in fields})
                followed = True
                for cm in cms.values():
                    cm.decided(0, active, f, boxes_now(), geometry)
            if active is None:
                continue
            k = f - active.frame
            if k > 0:
                followed = followed and k <= len(active.actions) and by[f - 1]["action"] == active.actions[k - 1]
            if not followed or k <= 0 or k >= len(active.actions) or row["status"] != 0:
                continue
            boxes = boxes_now()                     # same causal forecast, excluded from rule timings
            record = {"level": level, "wave": key[1], "frame": f, "plan_frame": active.frame,
                      "age": k, "death_in": deaths[key]["frame"] - f,
                      "position_error": float(np.hypot(row["x"] - active.path[k][0], row["y"] - active.path[k][1]))}
            for name, cm in cms.items():
                cpu, wall = time.process_time(), time.perf_counter()
                tr = cm.triggers(0, f, float(row["x"]), float(row["y"]), keys=int(row["keys"]),
                                 tiles=bool(rec.get("tc")), planner=geometry, fboxes_fn=lambda: boxes)
                totals[name]["wall_s"] += time.perf_counter() - wall
                totals[name]["cpu_s"] += time.process_time() - cpu
                totals[name]["checks"] += 1
                totals[name]["trigger_frames"] += bool(tr)
                totals[name]["surprise"] += any(t.startswith("surprise") for t in tr)
                totals[name]["danger"] += any(t.startswith("danger") for t in tr)
                record[name] = tr
            results.append(record)
        if key == ("4-2", 50):
            break
    restored = [r for r in results if any(t.startswith("surprise") for t in r["corrected"])
                and not any(t.startswith("surprise") for t in r["unconditional"])]
    report = {"scope": "online_check06 4-1 wave 5 and 4-2 wave 50; fixed committed plans and actual actions",
              "limitations": ["No counterfactual actions, model retraining, success-rate or FPS claim.",
                              "Causal hazard forecast reconstructed; DeathMemory omitted.",
                              "Rule timings exclude forecast construction and planner inference; single pass."],
              "baseline_commit": BASE, "unconditional_source_sha256": digest(gated_source.encode()),
              "corrected_source_sha256": digest((ROOT / "plan/commit.py").read_bytes()),
              "inputs": inputs, "totals": totals, "restored_early_surprises": restored, "checks": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"totals": totals, "restored_early_surprises": restored}, indent=2))


if __name__ == "__main__":
    main()
