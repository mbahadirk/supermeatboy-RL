"""How well the geometric safety check (plan/safety.py) explains the real recorded deaths, and how often it
flags agents that lived.

    python -m plan.measure_safety [wm_data dirs...]      (from the smbrl folder; default: every recorded run)

For every death transition (s -> s2, s2 dead): is the move s -> s2 in contact with the hazards and shots of
s2's frame (swept), or out of the level? For a random SURVIVOR_SAMPLE of the transitions that lived: is the
end point in contact (falsely flagged)? Per level and overall, with the cells of the deaths left unexplained.
Writes runs/safety_report.md and prints it.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import hazard_reach                                                   # noqa: E402
from game.level import Level, TILE                                    # noqa: E402
from plan import safety as S                                          # noqa: E402
from wm.buffer import load_level, load_shards, transitions, STATUS_ALIVE, STATUS_DEAD   # noqa: E402

SURVIVOR_SAMPLE = 0.10
TOP_CELLS = 10
REPORT = os.path.join(HERE, "runs", "safety_report.md")


def level_from_data(d: str, name: str) -> Level:
    """As mpc/offline.level_from_data, without importing the trainer."""
    words, meta = load_level(d, name)
    info = dict(meta)
    info["name"] = name
    return Level(info, words.astype("<u4").tobytes())


def new_stats():
    return {"deaths": 0, "explained": 0, "end": 0, "swept_only": 0, "shot": 0, "out": 0, "no_record": 0,
            "surv": 0, "surv_contact": 0, "surv_out": 0, "cells": Counter()}


def measure(dirs: list, log=print) -> dict:
    per = defaultdict(new_stats)
    rng = np.random.default_rng(0)
    for d in dirs:
        levels, t0, n_shards = {}, time.time(), 0
        for meta, rows, world in load_shards(d):
            name = meta.get("level")
            if name not in levels:
                try:
                    levels[name] = level_from_data(d, name)
                except OSError:
                    levels[name] = None
                    log(f"[safety] {d}: no level data for {name}, skipped")
            lv = levels[name]
            if lv is None or not len(rows):
                continue
            n_shards += 1
            st = per[name]
            bounds = S.level_bounds(lv)
            tr = transitions(rows)
            s, s2 = tr["s"], tr["s2"]
            dead = s2["status"] == STATUS_DEAD
            surv = (s2["status"] == STATUS_ALIVE) & (rng.random(len(s2)) < SURVIVOR_SAMPLE)
            pick = dead | surv
            if not pick.any():
                continue
            s, s2, dead = s[pick], s2[pick], dead[pick]
            fr = s2["frame"]
            order = np.argsort(fr, kind="stable")
            s, s2, dead, fr = s[order], s2[order], dead[order], fr[order]
            keep, grow = lv.hz_keep, lv.hz_grow
            wf = sorted(world, key=lambda w: w["f"])
            wi = 0
            ufr, start = np.unique(fr, return_index=True)
            stops = list(start[1:]) + [len(fr)]
            for f, a, b in zip(ufr, start, stops):
                rec = None
                while wi < len(wf) and wf[wi]["f"] <= f:      # keep updates up to this frame
                    if "keep" in wf[wi]:
                        keep = wf[wi]["keep"]
                    if wf[wi]["f"] == f:
                        rec = wf[wi]
                    wi += 1
                x0, y0 = s["x"][a:b].astype(np.float64), s["y"][a:b].astype(np.float64)
                x1, y1 = s2["x"][a:b].astype(np.float64), s2["y"][a:b].astype(np.float64)
                dm = dead[a:b]
                fb = S.record_boxes(rec, keep, grow)
                if dm.any():
                    st["deaths"] += int(dm.sum())
                    if rec is None:
                        st["no_record"] += int(dm.sum())
                    kind, _ = S.segment_contact(x0[dm], y0[dm], x1[dm], y1[dm], fb, bounds)
                    endk, _ = S.segment_contact(x1[dm], y1[dm], x1[dm], y1[dm], fb, bounds)
                    ok = kind != S.KIND_NONE
                    st["explained"] += int(ok.sum())
                    st["end"] += int((endk != S.KIND_NONE).sum())
                    st["swept_only"] += int((ok & (endk == S.KIND_NONE)).sum())
                    st["shot"] += int((kind == S.KIND_SHOT).sum())
                    st["out"] += int((kind == S.KIND_OUT).sum())
                    for x, y in zip(x1[dm][~ok], y1[dm][~ok]):
                        st["cells"][lv.cell(x, y)] += 1
                sm = ~dm
                if sm.any():
                    st["surv"] += int(sm.sum())
                    clear, oc = S.segment_clearance(x1[sm], y1[sm], x1[sm], y1[sm], fb, bounds)
                    hit = clear.min(1) < 0 if clear.shape[1] else np.zeros(int(sm.sum()), bool)
                    st["surv_contact"] += int(hit.sum())
                    st["surv_out"] += int((~hit & (oc < 0)).sum())
        tot = sum(v["deaths"] for v in per.values())
        log(f"[safety] {os.path.relpath(d, HERE)}: {n_shards} waves in {time.time() - t0:.0f} s; "
            f"{tot} deaths so far")
    return dict(per)


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "-"


def report(per: dict, dirs: list, seconds: float) -> str:
    total = new_stats()
    for st in per.values():
        for k, v in st.items():
            total[k] = total[k] + v
    lines = ["# Geometric safety check on recorded deaths (plan/measure_safety.py)", "",
             f"Data: {len(dirs)} wm_data folders ({', '.join(os.path.relpath(d, HERE) for d in dirs)}); "
             f"{seconds / 60:.1f} min.", "",
             "- **explained**: the move into the death position touches a hazard or shot of the death frame "
             "(swept, BODY contact, grown boxes) or leaves the level.",
             "- **end point**: the death position alone touches; **swept only**: only the segment does.",
             f"- **survivors flagged**: a random {SURVIVOR_SAMPLE:.0%} of the transitions that lived; end point "
             "in contact with a box, or out of the level (counted apart).", "",
             "| Level | Deaths | Explained | End point | Swept only | Shot | Out of level | No frame record | "
             "Survivors sampled | Flagged (contact) | Flagged (out) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, st):
        d = st["deaths"]
        return (f"| {name} | {d} | {pct(st['explained'], d)} | {pct(st['end'], d)} | {pct(st['swept_only'], d)} | "
                f"{pct(st['shot'], d)} | {pct(st['out'], d)} | {st['no_record']} | {st['surv']} | "
                f"{pct(st['surv_contact'], st['surv'])} | {pct(st['surv_out'], st['surv'])} |")

    key = lambda n: tuple(int(p) if p.isdigit() else 0 for p in str(n).split("-")) + (str(n),)
    for name in sorted(per, key=key):
        lines.append(row(name, per[name]))
    lines.append(row("**all**", total))
    lines += ["", "## Unexplained deaths: top cells per level (col, row: count)", ""]
    for name in sorted(per, key=key):
        st = per[name]
        un = st["deaths"] - st["explained"]
        if not un:
            continue
        cells = ", ".join(f"({c},{r}): {k}" for (c, r), k in st["cells"].most_common(TOP_CELLS))
        lines.append(f"- **{name}** ({un} unexplained): {cells}")
    return "\n".join(lines) + "\n"


def main():
    dirs = sys.argv[1:] or hazard_reach.all_data()
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f"not a folder: {', '.join(missing)}")
    t0 = time.time()
    per = measure([os.path.abspath(d) for d in dirs])
    text = report(per, [os.path.abspath(d) for d in dirs], time.time() - t0)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
