"""Compare the clean-room engine (smb_measured preset) against live Super Meat Boy captures.

Game-side numbers come only from recorded live frames (traj.jsonl); clone-side numbers come from
simulating the same manoeuvre in engine.World and converting px back to game units (/k).

Usage (from the project root):
  python smbrl/tools/capture/validate_physics.py smbrl/imported/capture/traj.jsonl [input_probe_*.jsonl ...]

Probe files (smb_input_probe.py) hold scripted pad trials with the exact buttons per frame; each trial is
replayed in the clone from rest on a flat floor and compared frame by frame.
"""
import os
import statistics as st
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # smbrl/
sys.path.insert(0, HERE)
from engine import World, Action, TILE, smb_measured  # noqa: E402
from levels import Level  # noqa: E402
import smb_calibrate as C  # noqa: E402

K = TILE / 20.0


def game_jumps(tracks, g=550.0):
    """Unobstructed ground jumps: every rising frame shows pure gravity (no ceiling, wall or wind).

    Returns (rising, round_trip): rising = (first vy, apex height, frames to apex) for every such jump;
    round_trip = airtime in frames for the subset that lands back at the takeoff height.
    """
    step = g / C.FPS
    rising, round_trip = [], []
    for t in tracks:
        for i in range(1, len(t) - 150):
            a, b = t[i - 1], t[i]
            if not (b["vy"] - a["vy"] > 100 and b["vy"] > 50 and a["vy"] == 0 and a["st"] != C.WALL):
                continue
            y0, ok, apex = a["y"], True, None
            for j in range(i + 1, i + 150):
                p, q = t[j - 1], t[j]
                if abs(q["x"] - p["x"]) > 60 or q["st"] == C.WALL:
                    ok = False
                    break
                if apex is None:
                    if abs((q["vy"] - p["vy"]) + step) > 0.1:  # anything but pure gravity while rising
                        ok = False
                        break
                    if q["vy"] <= 0:
                        apex = (b["vy"], max(p["y"], q["y"]) - y0, j - i + 1)
                        rising.append(apex)
                elif q["vy"] == 0:  # landed
                    if abs(q["y"] - y0) < 0.5:
                        round_trip.append(j - i + 1)
                    break
            if not ok:
                continue
    return rising, round_trip


def clone_world(grid):
    return World(Level("cal", grid), smb_measured())


FLOOR = ["  " + " " * 60, " S" + " " * 58 + "G", "#" * 62, "#" * 62]
WALL = ["#  " + " " * 8] + ["#  " + " " * 8 for _ in range(40)] + ["#S " + " " * 7 + "G", "#" * 11]


def settle(w, n=20):
    for _ in range(n):
        w.step(Action())


def clone_accel(run):
    w = clone_world(FLOOR); settle(w)
    vmax = (w.p.run_max if run else w.p.walk_max)
    for f in range(1, 200):
        s = w.step(Action(right=True, run=run))
        if abs(s["vx"] - vmax) < 1e-6:
            return f, s["vx"] / K


def clone_turn_and_release():
    w = clone_world(FLOOR); settle(w)
    while w.step(Action(right=True, run=True))["vx"] < w.p.run_max:
        pass
    for f in range(1, 200):
        if w.step(Action(left=True, run=True))["vx"] <= -w.p.run_max + 1e-6:
            turn = f
            break
    for f in range(1, 200):
        if abs(w.step(Action(run=True))["vx"]) < 1e-6:
            return turn, f


def clone_jump():
    w = clone_world(FLOOR); settle(w)
    y0 = w.state()["y"]
    s = w.step(Action(jump=True))
    first = -s["vy"] / K
    peak, apex_f = 0.0, None
    for f in range(2, 200):
        s = w.step(Action(jump=True))
        peak = max(peak, (y0 - s["y"]) / K)
        if apex_f is None and s["vy"] >= 0:
            apex_f = f
        if s["on_ground"]:
            return first, peak, apex_f, f


def clone_wall():
    """Start mid-air against the left wall, hold left: per-frame gravity on the wall, then a wall jump."""
    w = clone_world(WALL)
    w.x, w.y, w.vx, w.vy = float(TILE), 10.0 * TILE, -1.0, 0.0
    dv, prev = [], None
    for _ in range(12):
        s = w.step(Action(left=True))
        if prev is not None and s["on_wall"] and not s["on_ground"]:
            dv.append((s["vy"] - prev) / K * 60)  # +Y down: falling speeds up
        prev = s["vy"]
    s = w.step(Action(left=True, jump=True))
    return st.median(dv) if dv else None, -s["vy"] / K, abs(s["vx"]) / K


def probe_trials(paths):
    """Replay scripted pad trials (buttons per frame known) in the clone; count frames whose vx/vy match."""
    import json
    from collections import defaultdict
    trials, facing_before, last_fl = defaultdict(list), {}, None
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            key = (path, r["trial"])
            if key not in trials and last_fl is not None:
                facing_before[key] = -1 if last_fl & 1 else 1   # flags bit0 = facing left
            trials[key].append(r)
            last_fl = r["fl"]
    print()
    print(f"{'scripted trial':28s} {'frames':>6s} {'vx ok':>6s} {'vy ok':>6s}  first mismatch (k: game vs clone vx/vy)")
    for key, rows in trials.items():
        name = key[1]
        if name.startswith("setup") or key not in facing_before:
            continue
        w = clone_world(["  " + " " * 200, " S" + " " * 198 + "G", "#" * 202, "#" * 202])
        settle(w)
        w.x = 100 * TILE                                   # open floor, far from walls
        w.facing = facing_before[key]
        ok_x = ok_y = 0
        first = None
        for r in rows:
            b = r["applied"]
            s = w.step(Action(left=bool(b & 4), right=bool(b & 8), jump=bool(b & 0x1000), run=bool(b & 0x4000)))
            cvx, cvy = s["vx"] / K, -s["vy"] / K
            gx_ok, gy_ok = abs(cvx - r["vx"]) < 0.6, abs(cvy - r["vy"]) < 0.6
            ok_x += gx_ok
            ok_y += gy_ok
            if first is None and not (gx_ok and gy_ok):
                first = f"{r['k']}: {r['vx']:.0f}/{r['vy']:.0f} vs {cvx:.0f}/{cvy:.0f}"
        print(f"{name:28s} {len(rows):6d} {ok_x:6d} {ok_y:6d}  {first or '-'}")


def main(path, probes=()):
    tr = C.live_tracks(path)
    rising, round_trip = game_jumps(tr)
    print(f"\ngame unobstructed jumps: rising={len(rising)} round-trip={len(round_trip)}")
    g_first = st.median(j[0] for j in rising); g_peak = st.median(j[1] for j in rising)
    g_apex = st.median(j[2] for j in rising); g_air = st.median(round_trip) if round_trip else float("nan")
    c_first, c_peak, c_apex, c_air = clone_jump()
    walk_f, walk_v = clone_accel(False); run_f, run_v = clone_accel(True)
    turn_f, rel_f = clone_turn_and_release()
    wall_g, wj_vy, wj_vx = clone_wall()

    rows = [
        ("jump: vy after takeoff frame", g_first, c_first),
        ("jump: apex height (units)", g_peak, c_peak),
        ("jump: frames to apex", g_apex, c_apex),
        ("jump: airtime to same height (frames)", g_air, c_air),
        ("walk: top speed", 240.0, walk_v),
        ("walk: frames 0 -> top (game 240/8)", 30, walk_f),
        ("run: top speed", 390.0, run_v),
        ("run: frames 0 -> top (game 390/13)", 30, run_f),
        ("run: frames to reverse (scripted: 1 lag + 390/26 + 390/13)", 46, turn_f),
        ("run: frames to stop on release (scripted trials)", 1, rel_f),
        ("wall: gravity while on wall", 330.0, wall_g),
        ("wall jump: vy after takeoff frame", 344.5, wj_vy),
        ("wall jump: vx", 300.0, wj_vx),
    ]
    print(f"\n{'behaviour':42s} {'game':>9s} {'clone':>9s} {'diff%':>7s}")
    for name, g, c in rows:
        d = "" if not g or c is None else f"{100 * (c - g) / g:+.1f}"
        print(f"{name:42s} {g:9.2f} {c if c is not None else float('nan'):9.2f} {d:>7s}")


    if probes:
        probe_trials(probes)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
