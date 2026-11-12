"""Read-only bounded 2-10 progress graph audit; two field builds, no model inference/game/training."""
import os
for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_var] = "1"
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wm.buffer import load_level, load_shard, shard_files
from game.level import Level, hazard_boxes
from plan import progress as P

RUN = ROOT / "runs/collect_2-10_fan_keys_01"
OUT = Path(__file__).parent


def main():
    t0 = time.perf_counter()
    folder = RUN / "wm_data"
    words, meta = load_level(str(folder), "2-10")
    smeta, rows, world = load_shard(shard_files(str(folder))[0], 0)
    w0 = world[0]
    lv = Level(meta, words.astype("<u4").tobytes())
    lv.set_fields(w0.get("fx"))
    lv.set_platforms(w0.get("pf"))
    static, _ = hazard_boxes(w0.get("hz", []), w0.get("keep", lv.hz_keep))
    lv.set_sealed(static, [lv.spawn] + lv.goals + lv.keys + lv.bandages + [p for pair in lv.portals for p in pair])
    captures, current = [], {}
    orig_step, orig_cost = P._step, P._cost_to_go
    def step(ctx, *args):
        current["ctx"] = ctx
        return orig_step(ctx, *args)
    def cost(n, src, dst, w, seeds):
        value = orig_cost(n, src, dst, w, seeds)
        captures.append(dict(ctx=current["ctx"], n=n, src=src, dst=dst, w=w, seeds=seeds, value=value))
        return value
    P._step, P._cost_to_go = step, cost
    kw = dict(lift=lv.wind | lv.plat_support(), blocks=lv.plat_blocks() | lv.sealed_haz | lv.sealed_wall,
              doors=lv.plat_doors(), outside=lv.outside_cells if lv.kill_box is not None else None)
    field = P.ProgressField(words, lv.ox, lv.oy, lv.spawn, lv.goals, static, (), lv.keys, lv.bandages, lv.portals, **kw)
    P._step, P._cost_to_go = orig_step, orig_cost
    # Legs are built backwards, V then T. Final captured T graph belongs to key leg zero.
    graph = captures[-1]
    ctx, K = graph["ctx"], graph["ctx"]["K"]
    inv = {int(ctx["cid"][r, c]): (int(c), int(r)) for r, c in zip(*np.nonzero(ctx["cid"] >= 0))}
    def decode(node):
        cell = inv.get(int(node) // K)
        if cell is None:
            return {"node": int(node), "portal": True}
        c, r = cell
        state = int(node) % K
        up = state // (2 * (ctx["S"] + 1)) if K > 1 else 0
        side = (state // 2) % (ctx["S"] + 1) if K > 1 else 0
        return {"node": int(node), "cell": [c, r], "xy": list(field.pl.center(cell)),
                "up": up, "side": side, "falling": bool(state % 2),
                "support": bool(ctx["support"][r, c]), "lift": bool(lv.wind[r, c]),
                "time_without_later_leg": float(graph["value"][node])}
    def trace(x, y):
        c, r = field.pl.cell(x, y)
        node = int(ctx["cid"][r, c]) * K
        if not ctx["support"][r, c]:
            # Read rising/falling lookup's actual reachable-min choice below separately.
            return {"query": [x, y], "unsupported": True}
        path = []
        for _ in range(100):
            rec = decode(node)
            ids = np.flatnonzero(graph["src"] == node)
            if not len(ids):
                path.append(rec)
                break
            best = ids[np.argmin(graph["w"][ids] + graph["value"][graph["dst"][ids]])]
            nxt = int(graph["dst"][best])
            rec["edge_frames"] = float(graph["w"][best])
            rec["to"] = decode(nxt)["cell"]
            path.append(rec)
            node = nxt
        return {"query": [x, y], "T_support": float(field.T_support[0, r, c]), "path": path}
    probes = []
    for y in (-154, -114, -74, -34, 6, 46, 86, 126):
        for x in (30, 70, 110, 150, 190, 230):
            c, r = field.pl.cell(x, y)
            probes.append(dict(x=x, y=y, cell=[c, r], support=bool(ctx["support"][r, c]), lift=bool(lv.wind[r, c]),
                               T_support=float(field.T_support[0, r, c]), T_rise=float(field.T_rise[0, r, c]),
                               T_fall=float(field.T_fall[0, r, c])))
    traces = [trace(x, -154) for x in (30, 110, 190, 230)]
    # Controlled counterfactual only: remove wind-support, preserving all static geometry/hazards/outside.
    kw["lift"] = lv.plat_support()
    no_lift = P.ProgressField(words, lv.ox, lv.oy, lv.spawn, lv.goals, static, (), lv.keys, lv.bandages, lv.portals, **kw)
    map_text = []
    for r in range(lv.H - 1, -1, -1):
        line = []
        for c in range(lv.W):
            ch = "#" if field.pl.wall[r, c] else "."
            if lv.wind[r, c] and ch == ".": ch = "F"
            if ctx["support"][r, c] and ch == ".": ch = "s"
            if field.pl.cls["lock"][r, c]: ch = "L"
            if (c, r) == field.pl.cell(30, -154): ch = "X"
            if (c, r) == field.pl.cell(*lv.keys[0]): ch = "K"
            line.append(ch)
        map_text.append(f"r{r:02d} y={lv.oy + (r+.5)*20:6.0f} " + "".join(line))
    report = {"runtime_reconstruction": {"summary": field.summary(), "static_hazards": len(static),
              "world_record_frame": w0.get("f"), "platforms": len(lv.plats), "lift_cells": int(lv.wind.sum()),
              "sealed_hazard": int(lv.sealed_haz.sum()), "sealed_wall": int(lv.sealed_wall.sum()),
              "mode": field.mode, "jumps": field.jumps, "leg_locks": field.leg_locks,
              "route_cells": field.route_cells, "keys": lv.keys, "fans": lv.fans},
              "traces": traces, "probes": probes, "map": map_text,
              "no_wind_support_counterfactual": {"summary": no_lift.summary(), "jumps": no_lift.jumps,
                  "floor_T": {str(x): float(no_lift.T_support[0, 7, field.pl.cell(x,-154)[0]]) for x in (30,70,110,150,190,230)}},
              "field_builds": 2, "elapsed_seconds": time.perf_counter()-t0}
    target = OUT / "graph_audit.json"
    with target.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"summary": report["runtime_reconstruction"], "traces": traces,
                      "no_lift": report["no_wind_support_counterfactual"], "elapsed": report["elapsed_seconds"]}, indent=2))
    print("\n".join(map_text))

if __name__ == "__main__":
    main()
