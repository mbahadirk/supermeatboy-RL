"""Measure Super Meat Boy movement constants from live player captures (traj.jsonl).

Rows come from the character update hook (one row per 60 Hz frame, game units, +Y up).
Inputs were not recorded, so every constant is read from velocity changes at
identifiable events rather than by replaying trajectories:
  gravity / wall-slide / wind   per-frame dvy distributions by phase
  jump / wall jump              vy (and vx) right after an upward impulse
  run speed / max fall          velocity plateaus
  accel / decel / turn          per-frame dvx while speeding up, slowing down, reversing
  jump cut                      extra vy loss while rising (early jump release)

Usage: python smbrl/tools/capture/smb_calibrate.py smbrl/imported/capture/traj.jsonl
"""
import json
import sys
from collections import Counter, defaultdict

FPS = 60.0
WALL = 9  # character state while wall-sliding (dvy = 0.6 g observed)


def live_tracks(path):
    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line) if line.strip() else None
        if r and r.get("k") == "pl" and r.get("vx") is not None:
            by[(r.get("r"), r["s"], r["p"])].append(r)
    # replay ghosts have vx = vy = 0 throughout; keep tracks with real velocities
    return [t for t in by.values() if any(abs(r["vx"]) > 1 or abs(r["vy"]) > 1 for r in t)]


def frames(tracks):
    """Consecutive frame pairs (a, b) with derived per-frame deltas, skipping respawn teleports."""
    for t in tracks:
        for a, b in zip(t, t[1:]):
            if abs(b["x"] - a["x"]) > 60 or abs(b["y"] - a["y"]) > 60:
                continue
            grounded = a["vy"] == 0 and b["vy"] == 0 and a["y"] == b["y"]
            yield a, b, b["vx"] - a["vx"], b["vy"] - a["vy"], grounded


def mode(values, nd=1):
    c = Counter(round(v, nd) for v in values)
    return c.most_common(1)[0] if c else (None, 0)


def main(path):
    tracks = live_tracks(path)
    fr = list(frames(tracks))
    print(f"live tracks={len(tracks)} frame pairs={len(fr)}\n")
    est = {}

    # gravity: airborne, off-wall, no impulse
    g = [-dvy * FPS for a, b, dvx, dvy, gr in fr if not gr and b["st"] != WALL and -30 < dvy < 0]
    est["gravity"] = mode(g)
    wall = [-dvy * FPS for a, b, dvx, dvy, gr in fr if b["st"] == WALL and a["st"] == WALL and -30 < dvy < 0]
    est["wall_slide_gravity"] = mode(wall)
    wind = [dvy * FPS for a, b, dvx, dvy, gr in fr if not gr and b["st"] == 6 and 0 < dvy < 30]
    est["wind_net_up_accel"] = mode(wind, 0)

    # jumps: upward impulse (dvy > 100 units/s in one frame)
    ground_j, wall_j = [], []
    for t in tracks:
        for p, a, b in zip(t, t[1:], t[2:]):
            # upward impulse that leaves the player moving up (b.vy > 50 excludes landings, where vy jumps to 0)
            if b["vy"] - a["vy"] > 100 and b["vy"] > 50 and abs(b["y"] - a["y"]) < 60:
                (wall_j if a["st"] == WALL or p["st"] == WALL else ground_j).append((b["vx"], b["vy"], a["vx"]))
    est["jump_vy"] = mode([vy for vx, vy, _ in ground_j])
    est["jump_vy_top3"] = Counter(round(vy, 1) for vx, vy, _ in ground_j).most_common(3)
    est["wall_jump_vy"] = mode([vy for vx, vy, _ in wall_j])
    est["wall_jump_vx"] = mode([abs(vx) for vx, vy, _ in wall_j])
    est["wall_jump_vx_top3"] = Counter(round(abs(vx)) for vx, vy, _ in wall_j).most_common(3)

    # horizontal plateaus and changes
    run = [abs(b["vx"]) for a, b, dvx, dvy, gr in fr if gr and abs(dvx) < 1e-3 and abs(b["vx"]) > 50]
    est["run_max_ground"] = mode(run)
    air_run = [abs(b["vx"]) for a, b, dvx, dvy, gr in fr if not gr and abs(dvx) < 1e-3 and abs(b["vx"]) > 50]
    est["run_max_air"] = mode(air_run)
    est["run_plateaus_top3"] = Counter(round(v) for v in run + air_run).most_common(3)
    fall = [-b["vy"] for a, b, dvx, dvy, gr in fr if not gr and b["st"] != WALL and abs(dvy) < 1e-3 and b["vy"] < -50]
    est["max_fall"] = mode(fall)
    est["max_fall_top3"] = Counter(round(v) for v in fall).most_common(3)
    est["fastest_fall_seen"] = (round(max(-b["vy"] for a, b, *_ in fr), 1), None)

    def dvx_set(grounded, speeding):
        out = []
        for a, b, dvx, dvy, gr in fr:
            if gr != grounded or abs(dvx) < 1e-3 or abs(dvx) > 60:
                continue
            same = (dvx > 0) == (a["vx"] > 0) or a["vx"] == 0
            if speeding == "up" and same and abs(b["vx"]) > abs(a["vx"]):
                out.append(abs(dvx) * FPS)
            if speeding == "down" and not same and abs(b["vx"]) < abs(a["vx"]) and (b["vx"] > 0) == (a["vx"] > 0):
                out.append(abs(dvx) * FPS)
            if speeding == "cross" and a["vx"] != 0 and b["vx"] != 0 and (a["vx"] > 0) != (b["vx"] > 0):
                out.append(abs(dvx) * FPS)
        return out

    for grounded, tag in ((True, "ground"), (False, "air")):
        for kind in ("up", "down"):
            vals = dvx_set(grounded, kind)
            est[f"{tag}_accel_{kind}"] = mode(vals, 0)
            est[f"{tag}_accel_{kind}_top3"] = Counter(round(v) for v in vals).most_common(3)

    # jump cut: while rising off-wall, dvy beyond gravity (more negative than -g/FPS)
    gval = est["gravity"][0] or 550.0
    # b.vy > 0 excludes ceiling bumps (vy snaps to 0)
    cut = [b["vy"] / a["vy"] for a, b, dvx, dvy, gr in fr
           if not gr and a["vy"] > 30 and b["vy"] > 0 and b["st"] != WALL and dvy < -(gval / FPS) - 5]
    est["jump_cut_ratio"] = mode(cut, 2)
    est["jump_cut_ratio_top3"] = Counter(round(v, 2) for v in cut).most_common(3)

    print(f"{'quantity':24s} {'mode':>10s} {'count':>7s}   (game units, per second)")
    for k, v in est.items():
        if k.endswith("_top3"):
            print(f"{'':26s}top: {v}")
        else:
            print(f"{k:24s} {v[0]!s:>10s} {v[1]!s:>7s}")
    return est


if __name__ == "__main__":
    main(sys.argv[1])
