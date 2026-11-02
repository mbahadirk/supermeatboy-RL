"""Phase 2 probes on collected world-model data (wm/buffer.py), all from real gameplay:

  repeat  Do hazards, shots and platforms sit at the same place at the same wave frame in every wave? If yes,
          the planner can look their future up instead of predicting it.
  step    The game's time step (ms per update) and how the game uses it: does a move equal velocity x the
          real step, or velocity / 60 whatever the step was?
  char    Which bytes of the character change from frame to frame, and which of them behave like the
          hidden timers (reset on a jump press / release, a landing, a wall touch). Needs --probe-char data.

    python -m wm.probes all <run>/wm_data
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from wm.buffer import load_shards, transitions

AIR, WALL = 0x2, 0x200
ACT_JUMP = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], bool)        # realgame.ACTIONS: which actions hold jump


# ---- repeat ------------------------------------------------------------------------------------------
def _timeline(world):
    """frame -> ({hazard id: (x, y)}, [platform (x, y)])"""
    out, ids = {}, []
    for w in world:
        if "ids" in w:
            ids = w["ids"]
        hz = {}
        for k, h in enumerate(w.get("hz", [])):
            hz[ids[k] if k < len(ids) else f"#{k}"] = (h[0], h[1])
        out[w["f"]] = (hz, [(p[0], p[1]) for p in w.get("pf", [])])
    return out


def repeat(d: str):
    shards = load_shards(d)
    by_level = {}
    for meta, _, world in shards:
        by_level.setdefault(meta.get("level"), []).append(_timeline(world))
    for lv, tls in by_level.items():
        if len(tls) < 2:
            print(f"[repeat] {lv}: one wave only, nothing to compare")
            continue
        ref = tls[0]
        dh, dp, moving = [], [], set()
        for tl in tls[1:]:
            for f, (hz, pf) in tl.items():
                if f not in ref:
                    continue
                rhz, rpf = ref[f]
                for hid, (x, y) in hz.items():
                    if hid in rhz:
                        dh.append(np.hypot(x - rhz[hid][0], y - rhz[hid][1]))
                for a, b in zip(pf, rpf):
                    dp.append(np.hypot(a[0] - b[0], a[1] - b[1]))
        for f, (hz, _) in ref.items():                 # which hazards move at all (in the first wave)
            if 0 in ref:
                for hid, (x, y) in hz.items():
                    if hid in ref[0][0] and np.hypot(x - ref[0][0][hid][0], y - ref[0][0][hid][1]) > 2:
                        moving.add(hid)
        line = f"[repeat] {lv}: {len(tls)} waves, {len(moving)} moving hazard(s)"
        if dh:
            dh = np.array(dh)
            line += f"; hazard offset vs wave 1 at the same frame: median {np.median(dh):.2f}, " \
                    f"99% {np.percentile(dh, 99):.2f}, max {dh.max():.1f} units"
        if dp:
            dp = np.array(dp)
            line += f"; platforms: median {np.median(dp):.2f}, max {dp.max():.1f}"
        print(line)


# ---- step --------------------------------------------------------------------------------------------
def step(d: str):
    shards = load_shards(d)
    dts, ex_real, ex_fixed = [], [], []
    for meta, rows, world in shards:
        dt_of = {w["f"]: w.get("dt") for w in world}
        dts += [v for v in dt_of.values() if v is not None]
        tr = transitions(rows)
        s, n = tr["s"], tr["s2"]
        # free flight: in the air both frames, no wall, alive: the move should be the new velocity x the step
        ok = ((s["flags"] & AIR) != 0) & ((n["flags"] & AIR) != 0) & ((s["flags"] & WALL) == 0) & \
             ((n["flags"] & WALL) == 0) & (n["status"] == 0)
        for a, b in zip(s[ok], n[ok]):
            dt = dt_of.get(int(a["frame"]))
            if dt is None or abs(b["vx"]) < 30:
                continue
            ex_real.append((b["x"] - a["x"]) - b["vx"] * dt / 1000.0)
            ex_fixed.append((b["x"] - a["x"]) - b["vx"] / 60.0)
    if not dts:
        print("[step] no time steps recorded (data from before the game sent them)")
        return
    dts = np.array(dts, np.float32)
    print(f"[step] {len(dts)} frames: time step median {np.median(dts):.1f} ms, 90% {np.percentile(dts, 90):.1f}, "
          f"99% {np.percentile(dts, 99):.1f}, max {dts.max():.0f}; steps over 20 ms {100 * (dts > 20).mean():.1f}%")
    if ex_real:
        er, ef = np.abs(ex_real), np.abs(ex_fixed)
        print(f"[step] free-flight moves ({len(er)}): |x change - vx x real step| median {np.median(er):.2f} units; "
              f"|x change - vx / 60| median {np.median(ef):.2f} units  -> the game uses "
              + ("the real step" if np.median(er) < np.median(ef) else "a fixed 1/60 s step"))


# ---- char --------------------------------------------------------------------------------------------
def char(d: str, top: int = 25):
    files = sorted(glob.glob(os.path.join(d, "wave_*_char.npy")))
    if not files:
        print("[char] no character bytes (run with --probe-char N)")
        return
    shards = {os.path.basename(m_p)[:10]: m_p for m_p in glob.glob(os.path.join(d, "wave_*.npz"))}
    V, F, ev = [], [], []
    for fp in files:
        key = os.path.basename(fp)[:10]
        B = np.load(fp)
        frames = np.load(fp.replace("_char.npy", "_char_frames.npy"))
        rows = np.load(shards[key])["rows"]
        r0 = {int(r["frame"]): r for r in rows[rows["agent"] == 0]}
        n4 = B.shape[1] // 4
        V.append(B[:, :n4 * 4].view("<u4"))
        F.append(B[:, :n4 * 4].view("<f4"))
        for f in frames:                              # what happened to the agent between this frame and the next
            a, b = r0.get(int(f)), r0.get(int(f) + 1)
            if a is None or b is None or a["action"] < 0:
                ev.append(None)
                continue
            jump = bool(ACT_JUMP[a["action"]])
            ev.append({"press": jump and a["held"] == 0, "release": (not jump) and a["held"] > 0,
                       "land": bool(a["flags"] & AIR) and not (b["flags"] & AIR),
                       "takeoff": not (a["flags"] & AIR) and bool(b["flags"] & AIR),
                       "wall": bool(b["flags"] & WALL) and not (a["flags"] & WALL)})
    V, F = np.concatenate(V), np.concatenate(F)
    if len(V) < 3:
        print("[char] too few frames")
        return
    changed = (V[1:] != V[:-1])
    rate = changed.mean(0)
    print(f"[char] {len(V)} frames, {V.shape[1]} 4-byte fields; {int((rate > 0).sum())} change at least once")
    known = {0x84: "x", 0x88: "y", 0x9c: "vx", 0xa0: "vy", 0x7b0: "state", 0x7b4: "flags", 0x824: "render x",
             0x828: "render y"}
    names = ["press", "release", "land", "takeoff", "wall"]
    evm = {k: np.array([bool(e and e[k]) for e in ev[:-1]]) for k in names}
    out = []
    for o in np.nonzero(rate > 0)[0]:
        ch = changed[:, o]
        # how much more often the field changes right after an event than on other frames
        lifts = {}
        for k in names:
            m = evm[k][:len(ch)]
            if m.sum() >= 2:
                lifts[k] = ch[m].mean() - ch[~m].mean()
        best = max(lifts.items(), key=lambda kv: kv[1]) if lifts else ("-", 0.0)
        f = F[:, o]
        fl = np.isfinite(f).all() and np.abs(f).max() < 1e7 and (np.abs(f) > 1e-6).any()
        diffs = np.diff(f) if fl else np.diff(V[:, o].astype(np.int64))
        counter = (np.median(diffs) != 0) and (np.mean(np.isclose(diffs, np.median(diffs))) > 0.6)
        out.append((best[1], o, rate[o], best[0], fl, counter, f if fl else V[:, o]))
    out.sort(key=lambda t: -t[0])
    print(f"[char] fields that change most right after an event (top {top}):")
    print("  offset   name       changes  event (lift)     kind      values")
    for lift, o, r, evn, fl, counter, vals in out[:top]:
        kind = ("counter " if counter else "") + ("float" if fl else "int")
        sample = ", ".join(f"{v:.3g}" if fl else str(int(v)) for v in vals[:8])
        print(f"  +0x{o * 4:03x}  {known.get(o * 4, ''):9s}  {100 * r:5.1f}%  {evn:8s} {lift:+.2f}   {kind:14s} {sample}")


if __name__ == "__main__":
    cmd, d = (sys.argv[1], sys.argv[2]) if len(sys.argv) >= 3 else ("", "")
    if cmd in ("repeat", "all"):
        repeat(d)
    if cmd in ("step", "all"):
        step(d)
    if cmd in ("char", "all"):
        char(d)
    if cmd not in ("repeat", "step", "char", "all"):
        print(__doc__)
