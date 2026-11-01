"""Batch evidence analysis over every level captured by smb_level_dumper.js.

For each captured level it collects, per set-piece type:
  deaths  : recorded lethal contacts whose nearest piece (<=50u) is this type
  goal    : run ends before a respawn teleport, away from any death, nearest piece (<=70u)
  updraft : sustained rises (>=45 frames, >=120u; a normal jump is ~33 frames / ~82u)
            starting just above a piece of this type
and infers a kind per (palette, type): goal > fan > lethal > unknown.
Writes palette_evidence.json and <level>_semantic.json files.

Usage: python smb_batch.py <capture_dir> <gamedata.dat> [semantic_out_dir]
  e.g. python smbrl/tools/capture/smb_batch.py smbrl/imported/capture D:/Games/SuperMeatBoy/gamedata.dat smbrl/imported/smb
"""
import glob
import json
import math
import os
import struct
import sys
from collections import Counter, defaultdict

DEATH_R, GOAL_R, FAN_DX, FAN_DY = 50.0, 70.0, 45.0, 150.0


def palettes_from_gamedata(path):
    """Map level name -> palette file named in its .smbl header (offset 9, inside a 256-byte field)."""
    f = open(path, "rb")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]
    n1 = u32(); f.read(8 * n1)
    n2 = u32(); entries = [struct.unpack("<III", f.read(12)) for _ in range(n2)]
    b1 = u32(); b2 = u32(); f.read(b1)
    names = [p.decode("latin1") for p in f.read(b2).split(b"\x00") if p]
    out = {}
    for name, (off, size, _) in zip(names, entries):
        if name.startswith("Levels/") and name.endswith(".smbl"):
            f.seek(off); head = f.read(min(size, 0x109))
            if head[:4] in (b"20LT", b"10LT"):
                out[name[7:-5]] = head[9:].split(b"\x00")[0].decode("latin1").removesuffix(".lp")
    return out


def tracks(rows):
    """Split each character's rows into continuous segments at respawn teleports."""
    by = defaultdict(list)
    for r in rows:
        by[r["p"]].append(r)
    segs = []
    for rs in by.values():
        cur = [rs[0]]
        for a, b in zip(rs, rs[1:]):
            if math.dist((a["x"], a["y"]), (b["x"], b["y"])) > 60:
                segs.append(cur); cur = []
            cur.append(b)
        segs.append(cur)
    return [s for s in segs if len(s) > 10]


def rises(seg):
    """Sustained, impulse-free upward runs from position deltas (works for live play and replay ghosts).

    A wall-climb chain also rises monotonically but shows a velocity jump at every jump;
    wind lifts change speed smoothly, so runs with more than one impulse are discarded.
    """
    moved = [seg[0]] + [b for a, b in zip(seg, seg[1:]) if (a["x"], a["y"]) != (b["x"], b["y"])]
    out, run = [], []

    def close():
        # Heuristic thresholds from 2-4 (wind) vs 3-1/3-2/4-1 (wall-jump climbs): wind lifts last >=75 frames,
        # rise >=250u and contain almost no wall-slide (state 9); climbs top out at the ~344 wall-jump speed.
        if len(run) >= 75 and run[-1]["y"] - run[0]["y"] >= 250:
            dy = [b["y"] - a["y"] for a, b in zip(run, run[1:])]
            wall = sum(1 for q in run if q.get("st") == 9) / len(run)
            if wall < 0.05 and sum(1 for u, v in zip(dy, dy[1:]) if v - u > 1.5) <= 1:
                out.append((run[0], run[-1], len(run)))

    for a, b in zip(moved, moved[1:]):
        if b["y"] - a["y"] > 0.01:
            run = run or [a]
            run.append(b)
        elif b["y"] - a["y"] < -0.5:
            close(); run = []
    close()
    return out


def type_table(level):
    table = {}
    for row in level["typeTable"]:
        w = struct.unpack("<10I", bytes.fromhex(row))
        table[w[0] & 0xff] = w[9]
    return table


def kind_from_table(t8, table):
    """Code-derived (FUN_004cd350) with live verification:
    0x06 level end (end events in factory/hell, run ends in forest/hospital); bit0 or no row = lethal on contact;
    0x0c bandage (sets global 0x6f0758 at the piece, cleared on death; hell 4-2);
    0x44 portal (0x40 handler fired only on the chapter map; never seen in a level); 0x14 not observed yet."""
    if t8 not in table:
        return "lethal"
    f = table[t8]
    if f == 0x06:
        return "goal"
    if f & 1:
        return "lethal"
    return {0x0c: "bandage", 0x44: "portal_unverified"}.get(f, f"special_{f:#x}")


def wind_zones(rises_list, merge_dx=40.0, half_w=20.0):
    """Cluster wind lifts into vertical columns keyed by the x where the lift starts: [x0, x1, y0, y1].

    The player drifts sideways while carried, so the start x locates the column better than the full path.
    """
    cols = []
    for a, b, _ in rises_list:
        for c in cols:
            if abs(a["x"] - c["x"]) <= merge_dx and a["y"] <= c["y1"] + 40 and c["y0"] <= b["y"] + 40:
                c["xs"].append(a["x"]); c["x"] = sum(c["xs"]) / len(c["xs"])
                c["y0"], c["y1"] = min(c["y0"], a["y"]), max(c["y1"], b["y"])
                break
        else:
            cols.append({"x": a["x"], "xs": [a["x"]], "y0": a["y"], "y1": b["y"]})
    return [[round(min(c["xs"]) - half_w, 1), round(max(c["xs"]) + half_w, 1), round(c["y0"], 1), round(c["y1"], 1)] for c in cols]


def traj_evidence(level, rows, death_pos):
    """Goal ends, rises and respawn points from one capture's player rows."""
    pieces = [(p["type"] & 0xff, p["x"], p["y"]) for p in level["pieces"]]
    ev, rises_all, spawns = defaultdict(Counter), [], Counter()
    for s in tracks(rows):
        spawns[(round(s[0]["x"], 1), round(s[0]["y"], 1))] += 1
        end = s[-1]
        if pieces and all(math.dist((end["x"], end["y"]), p) > 40 for p in death_pos):
            dist, t = min((math.dist((px, py), (end["x"], end["y"])), t) for t, px, py in pieces)
            if dist <= GOAL_R: ev[t]["goal"] += 1
        rises_all += rises(s)
    return ev, rises_all, spawns


def main(d, gamedata, out_dir=None):
    out_dir = out_dir or d
    pal = palettes_from_gamedata(gamedata)
    jl = lambda n: [json.loads(l) for l in open(os.path.join(d, n)) if l.strip()] if os.path.exists(os.path.join(d, n)) else []
    traj, deaths, events = jl("traj.jsonl"), jl("deaths.jsonl"), jl("events.jsonl")
    captures = defaultdict(list)
    for path in glob.glob(os.path.join(d, "level_*.json")):
        lv = json.load(open(path))
        if lv.get("name") and lv.get("tiles", {}).get("layers"):
            captures[lv["name"]].append((os.path.getmtime(path), lv))
    reports, counts, kinds = [], defaultdict(lambda: defaultdict(Counter)), defaultdict(dict)
    for name, caps in sorted(captures.items()):
        caps.sort(key=lambda c: c[0])
        latest = caps[-1][1]
        table = type_table(latest)
        pieces = [(p["type"] & 0xff, p["x"], p["y"]) for p in latest["pieces"]]
        dl = [x for x in deaths if x["name"] == name]
        ev = defaultdict(Counter)
        for x in dl:
            if pieces:
                dist, t = min((math.dist((px, py), x["pos"]), t) for t, px, py in pieces)
                if dist <= DEATH_R: ev[t]["deaths"] += 1
        # special-handler events (smb_level_dumper.js): attribute each hit to the nearest piece within 80u
        for x in events:
            if x["name"] != name or not x.get("pos") or not pieces:
                continue
            kind = x["ev"]
            if kind == "bit08_global":  # 0->1 happens at the pickup; 1->0 is the reset when the player dies
                if x.get("to") != 1: continue
                kind = "bit08_set"
            dist, t = min((math.dist((px, py), x["pos"]), t) for t, px, py in pieces)
            if dist <= 80: ev[t]["ev_" + kind] += 1
        rises_all, spawns = [], Counter()
        for _, lv in caps:  # trajectory evidence accumulates over every capture of this level
            rows = [r for r in traj if r.get("k") == "pl" and r.get("r") == lv.get("run") and r["s"] == lv["seq"]]
            e, rs, sp = traj_evidence(lv, rows, [x["pos"] for x in dl])
            for t, c in e.items(): ev[t].update(c)
            rises_all += rs; spawns.update(sp)
        spawn = list(spawns.most_common(1)[0][0]) if spawns else latest["spawn"]
        palette = pal.get(name, "?")
        objects = []
        for p in latest["pieces"]:
            t8 = p["type"] & 0xff
            k = kind_from_table(t8, table)
            support = ev[t8]["goal"] if k == "goal" else ev[t8]["deaths"] if k == "lethal" else 0
            scale = struct.unpack_from("<f", bytes.fromhex(p["raw"]), 0x24)[0]  # set-piece scale (f32 @+0x24)
            objects.append({"type": t8, "kind": k, "x": p["x"], "y": p["y"], "angle": p["angle"], "scale": scale,
                            "observed_support": support})
        t = latest["tiles"]; W, H = t["W"], t["H"]; raw = bytes.fromhex(t["layers"][0])
        # Solid = drawn tileset tile. Every palette declares numtiles=100, so ids >= 100 are markers;
        # the player passes through id 100 into 3-1's goal pocket.
        NUMTILES = 100
        zones = wind_zones(rises_all)
        json.dump({"name": name, "palette": palette, "W": W, "H": H, "tile": t["tileSize"], "origin": [t["originX"], t["originY"]],
                   "solid": [[1 if 0 < raw[(gy * W + gx) * 4] < NUMTILES else 0 for gx in range(W)] for gy in range(H)],
                   # top byte of each solid tile word: the game's per-level edge/shape bits (FUN_004cccd0 fills
                   # bits 22-31 from neighbours); levels_smb.shape_of() maps them to slopes and half blocks
                   "shape_bits": [[raw[(gy * W + gx) * 4 + 3] if 0 < raw[(gy * W + gx) * 4] < NUMTILES else 0
                                   for gx in range(W)] for gy in range(H)],
                   "spawn": spawn, "objects": objects, "wind_zones": zones},
                  open(os.path.join(out_dir, f"{name}_semantic.json"), "w"))
        types = Counter(p[0] for p in pieces)
        reports.append({"name": name, "palette": palette, "captures": len(caps), "deaths": len(dl), "spawn": spawn,
                        "types": dict(types), "evidence": {k: dict(v) for k, v in ev.items()}, "wind_zones": zones})
        for t8, n in types.items():
            counts[palette][t8].update({"instances": n, **ev[t8]})
            kinds[palette][t8] = (kind_from_table(t8, table), hex(table[t8]) if t8 in table else None)
    mapping = {p: {t8: {"kind": kinds[p][t8][0], "row_flags": kinds[p][t8][1], **c} for t8, c in ts.items()} for p, ts in counts.items()}
    json.dump({"levels": reports, "mapping": mapping}, open(os.path.join(d, "palette_evidence.json"), "w"), indent=1)
    for r in reports:
        print(f'{r["name"]:>5} {r["palette"]:<9} captures={r["captures"]} deaths={r["deaths"]:3d} spawn={r["spawn"]} wind_zones={r["wind_zones"]}')
    print("\npalette type mapping (kind from type table; counts = observed evidence):")
    for p, ts in mapping.items():
        print(" ", p)
        for t8, v in sorted(ts.items()):
            k, sup = v["kind"], v.get("goal", 0) if v["kind"] == "goal" else v.get("deaths", 0)
            flag = ""
            if v["kind"] == "goal" and v.get("deaths", 0): flag = "  <- deaths near goal"
            if v["kind"] != "goal" and v.get("goal", 0) >= 2: flag = "  <- runs end here"
            evs = {key[3:]: n for key, n in v.items() if key.startswith("ev_")}
            print(f"     type {t8:3d} row={v['row_flags']!s:6} kind={k:<12} instances={v['instances']:3d} "
                  f"deaths={v.get('deaths', 0):3d} goal_ends={v.get('goal', 0):3d}{' events=' + str(evs) if evs else ''}{flag}")
    unattributed = Counter(x["ev"] for x in events if not x.get("pos"))
    print("\nevents:", dict(Counter((x["name"], x["ev"]) for x in events)), "| without position:", dict(unattributed))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
