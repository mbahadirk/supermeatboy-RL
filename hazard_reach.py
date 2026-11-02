"""How far each kind of lethal piece really kills, measured from recorded deaths (wm_data).

The game reports a piece's size, but what kills is not that box plus Meat Boy's half size (realpath.BODY,
6 units) for every kind: lava kills about 19 units above its strip and 40+ units past its ends (the floor
between strips is lava too), fire falls 17 units above and below but only 6 to the side. A map built with
6 units everywhere leaves open cells where agents die, and routes go through them.

Measured per piece kind (the palette name), along the piece's own axes: how far beyond its box the agent's
centre was when it died (deaths are put on the nearest real hazard, not on decoration), against how close
surviving agents got. A reach is accepted at the 90th, else 75th, else 50th percentile of the deaths, the
first that fewer than SAFE_INSIDE of the survivors' positions fall inside. Kinds with fewer than MIN_DEATHS
deaths, or no accepted reach, keep BODY. Reach is never below BODY (only ever more caution).

    python smbrl/hazard_reach.py <run>/wm_data [more wm_data]     # writes smbrl/hazard_reach.json

realgame.hazard_boxes grows every piece's box by (reach - BODY) along each axis, so the agents' map, the
sealed pockets, the routes, the monitor and the planner's hazard check all use the measured reach. The world
model keeps the game's sizes (it learned from them).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "hazard_reach.json")
BODY = 6.0               # realpath.BODY
MIN_DEATHS = 30
SAFE_INSIDE = 0.01       # at most this share of survivors' positions inside an accepted reach
MAX_REACH = 60.0         # game units (3 tiles)
SAFE_SAMPLE = 0.15       # share of frames without a death whose survivors are counted

_cache = {"mtime": None, "table": {}}


def load() -> dict:
    """{piece name (lower case): (reach x, reach y)}; empty without a table. Re-read when the file changes."""
    try:
        mt = os.path.getmtime(TABLE)
    except OSError:
        return {}
    if _cache["mtime"] != mt:
        with open(TABLE, encoding="utf-8") as f:
            raw = json.load(f).get("pieces", {})
        _cache["table"] = {k: (float(v["reach_x"]), float(v["reach_y"])) for k, v in raw.items()}
        _cache["mtime"] = mt
    return _cache["table"]


def growth(names: list) -> list[tuple[float, float]]:
    """Per piece: how much to add to its half width and half height (0 for unknown kinds)."""
    t = load()
    out = []
    for nm in names:
        rx, ry = t.get((nm or "").lower(), (BODY, BODY))
        out.append((max(0.0, rx - BODY), max(0.0, ry - BODY)))
    return out


def _excess(x, y, boxes):
    """For each point, the nearest box and how far beyond it along the box's own axes (ex, ey)."""
    from wm.features import box_distances
    d = box_distances(x, y, boxes)
    k = d.argmin(1)
    b = boxes[k]
    c, s = np.cos(b[:, 4]), np.sin(b[:, 4])
    u = (x - b[:, 0]) * c + (y - b[:, 1]) * s
    v = -(x - b[:, 0]) * s + (y - b[:, 1]) * c
    return k, np.abs(u) - b[:, 2], np.abs(v) - b[:, 3]


def measure(dirs: list, log=print) -> dict:
    import realgame as rg
    from wm.buffer import load_level, load_shards, transitions
    died, safe = defaultdict(list), defaultdict(list)
    rng = np.random.default_rng(0)
    for d in dirs:
        lv_info = {}
        for meta, rows, world in load_shards(d):
            name = meta.get("level")
            if name not in lv_info:
                try:
                    _, m = load_level(d, name)
                except OSError:
                    lv_info[name] = None
                    continue
                pal = rg.piece_names(name)
                types = m.get("lethalTypes", [])
                lv_info[name] = (pal, types)
            if lv_info[name] is None:
                continue
            pal, types = lv_info[name]
            nm_of = lambda t: (pal[t].lower() if pal and 0 <= t < len(pal) else f"type {t}")
            keep = [not nm_of(t).startswith(rg.DECOR) for t in types]
            wf = {w["f"]: w for w in world}
            tr = transitions(rows)
            s2 = tr["s2"]
            for f in np.unique(s2["frame"]):
                w = wf.get(int(f))
                if "keep" in (w or {}):
                    keep = w["keep"]
                if not w or not w.get("hz"):
                    continue
                m = s2["frame"] == f
                dead = s2["status"][m] == 1
                if not dead.any() and rng.random() > SAFE_SAMPLE:
                    continue
                boxes, kept = rg.hazard_boxes(w["hz"], keep)
                if not boxes:
                    continue
                k, ex, ey = _excess(s2["x"][m].astype(float), s2["y"][m].astype(float), np.array(boxes, float))
                for i in range(len(k)):
                    j = kept[k[i]]
                    nm = nm_of(types[j]) if j < len(types) else f"piece {j}"
                    (died if dead[i] else safe)[nm].append((ex[i], ey[i]))
        log(f"[reach] {d}: {sum(len(v) for v in died.values())} deaths so far")
    out = {}
    for nm, dd in died.items():
        dd = np.array(dd)
        ss = np.array(safe.get(nm, [(1e9, 1e9)]))
        row = {"deaths": int(len(dd)), "safe": int(len(ss)), "reach_x": BODY, "reach_y": BODY, "percentile": None}
        if len(dd) >= MIN_DEATHS:
            for q in (90, 75, 50):
                rx, ry = np.percentile(dd[:, 0], q), np.percentile(dd[:, 1], q)
                inside = float(((ss[:, 0] < rx) & (ss[:, 1] < ry)).mean())
                if inside < SAFE_INSIDE:
                    row.update(reach_x=round(float(np.clip(rx, BODY, MAX_REACH)), 1),
                               reach_y=round(float(np.clip(ry, BODY, MAX_REACH)), 1), percentile=q,
                               safe_inside=round(inside, 4))
                    break
        out[nm] = row
    return out


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return
    sys.path.insert(0, HERE)
    pieces = measure(dirs)
    with open(TABLE, "w", encoding="utf-8") as f:
        json.dump({"about": "how far each kind of lethal piece kills beyond its box (hazard_reach.py)",
                   "measured_from": dirs, "body": BODY, "pieces": dict(sorted(pieces.items()))}, f, indent=1)
    for nm, r in sorted(pieces.items(), key=lambda kv: -kv[1]["deaths"]):
        print(f"{nm[:28]:28s} deaths {r['deaths']:5d}  reach x {r['reach_x']:5.1f}  y {r['reach_y']:5.1f}"
              + (f"  (p{r['percentile']}, survivors inside {r['safe_inside']:.4f})" if r["percentile"] else "  (kept at BODY)"))
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
