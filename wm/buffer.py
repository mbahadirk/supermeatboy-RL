"""Transition data from real gameplay, for the world model.

Writing (Collector, called by realgame.py with --wm-collect): per wave, one shard in <run>/wm_data/:
  wave_<n>.npz        agent rows: one per agent per frame while it is alive (and the frame it dies / finishes)
  wave_<n>_world.json.gz   per frame: the game's time step and the level's moving parts
  wave_<n>_char.npy   (--probe-char) the first agent's character object, one row of bytes per frame
  levels/<name>.npz   once per level: tile words, origin, spawn, goals, keys, bandages, portals

An agent row holds the state the trainer saw at that frame (before the agent's input), the jump key as held
before it, the action chosen for it (-1 when it chose none: dead / finished), and the reward its previous
action earned (NaN on its first row). Rows are raw game values, not network inputs, so the world model can
build its own features from them and the level data.

Reading (load_shards / transitions): consecutive rows of one agent give the transitions
(state, action, next state, reward, terminated, truncated).

    python -m wm.buffer stats <wm_data dir>
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

import numpy as np

ROW = np.dtype([("wave", "<i4"), ("frame", "<i4"), ("agent", "<i2"),
                ("x", "<f4"), ("y", "<f4"), ("vx", "<f4"), ("vy", "<f4"),
                ("flags", "<u4"), ("state", "<i2"), ("status", "<i1"), ("keys", "<u4"),
                ("held", "<i2"), ("action", "<i1"), ("reward", "<f4")])
STATUS_ALIVE, STATUS_DEAD, STATUS_FINISHED = 0, 1, 2


def _r(v, nd=2):
    """Round nested lists of numbers (smaller JSON)."""
    if isinstance(v, (list, tuple)):
        return [_r(x, nd) for x in v]
    return round(v, nd) if isinstance(v, float) else v


class Collector:
    def __init__(self, out_dir: str, char_bytes: bool = False):
        self.dir = out_dir
        os.makedirs(os.path.join(out_dir, "levels"), exist_ok=True)
        self.char_bytes = char_bytes
        self.level_name = None
        self.n_shards = len(glob.glob(os.path.join(out_dir, "wave_*.npz")))
        self._reset()

    def _reset(self):
        self.rows, self.world, self.chars, self.char_frames = [], [], [], []
        self.ids_sent = None

    def level(self, lv, info: dict):
        """A new level: its static data, once per level name."""
        self.level_name = lv.name
        self._reset()
        p = os.path.join(self.dir, "levels", f"{lv.name}.npz")
        if os.path.exists(p):
            return
        meta = {"name": lv.name, "W": lv.W, "H": lv.H, "ox": lv.ox, "oy": lv.oy, "spawn": list(lv.spawn),
                "goals": [list(g) for g in lv.goals], "keys": [list(k) for k in lv.keys],
                "bandages": [list(b) for b in lv.bandages], "portals": [[list(a), list(b)] for a, b in lv.portals],
                "objs": info.get("objs", []), "lethalTypes": info.get("lethalTypes", [])}
        np.savez_compressed(p, words=lv.words0.astype(np.uint32), meta=json.dumps(meta))

    def frame(self, m: dict, hz_ids: list, hz_keep: list | None = None):
        """The level's moving parts this frame (and the game's time step for the update that follows)."""
        rec = {"f": m["frame"], "dt": m.get("dt"), "hz": _r(m.get("hz", [])), "sh": _r(m.get("sh", []))}
        if m.get("pf"):
            rec["pf"] = _r(m["pf"])
        if m.get("tc"):
            rec["tc"] = m["tc"]
        if m.get("fx"):
            rec["fx"] = _r(m["fx"])
        if hz_ids != self.ids_sent:             # which hazard is which (changes when pieces come or go), and
            rec["ids"] = list(hz_ids)           # which are real (the rest is decoration the trainer skips)
            if hz_keep is not None:
                rec["keep"] = [bool(k) for k in hz_keep]
            self.ids_sent = list(hz_ids)
        self.world.append(rec)

    def agents(self, wave: int, frame: int, states: list, alive: list, held: list, actions: list, rewards):
        """One row per agent that was alive when this frame came in. The row's wave is the shard number (the
        game's wave count starts again on every level; it is kept in the shard's meta)."""
        idx = [i for i in range(len(states)) if i < len(alive) and alive[i]]
        if not idx:
            return
        r = np.zeros(len(idx), ROW)
        for k, i in enumerate(idx):
            s = states[i]
            r[k] = (self.n_shards, frame, i, s[0], s[1], s[2], s[3], int(s[4]) & 0xffffffff, s[5], s[6],
                    int(s[7]) if len(s) > 7 else 0, held[i], actions[i], rewards[i])
        self.rows.append(r)

    def char(self, frame: int, data: bytes):
        if self.char_bytes and data:
            self.chars.append(np.frombuffer(data, np.uint8).copy())
            self.char_frames.append(frame)

    def wave_end(self, wave: int, m: dict):
        """Write this wave's shard (the game waits for the trainer here anyway)."""
        if self.rows:
            k = self.n_shards
            self.n_shards += 1
            base = os.path.join(self.dir, f"wave_{k:05d}")
            meta = {"level": self.level_name, "wave": wave, "why": m.get("why"), "frames": m.get("frame"),
                    "status": m.get("status"), "finish": m.get("finish")}
            np.savez_compressed(base + ".npz", rows=np.concatenate(self.rows), meta=json.dumps(meta))
            with gzip.open(base + "_world.json.gz", "wt", encoding="utf-8") as f:
                json.dump(self.world, f, separators=(",", ":"))
            if self.chars:
                np.save(base + "_char.npy", np.stack(self.chars))
                np.save(base + "_char_frames.npy", np.array(self.char_frames, np.int32))
        self._reset()


# ---- reading --------------------------------------------------------------------------------------
def load_shards(d: str, level: str | None = None):
    """[(meta, rows, world frames)] of every shard in a wm_data folder (optionally one level only)."""
    out = []
    for k, p in enumerate(sorted(glob.glob(os.path.join(d, "wave_*.npz")))):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        if level and meta.get("level") != level:
            continue
        wp = p[:-4] + "_world.json.gz"
        world = json.load(gzip.open(wp, "rt", encoding="utf-8")) if os.path.exists(wp) else []
        rows = z["rows"].copy()
        rows["wave"] = k                     # one number per shard (older data stored the game's wave count)
        out.append((meta, rows, world))
    return out


def load_level(d: str, name: str):
    z = np.load(os.path.join(d, "levels", f"{name}.npz"))
    return z["words"], json.loads(str(z["meta"]))


def transitions(rows: np.ndarray, wave_ended_by_time: bool = True) -> dict:
    """Pairs of consecutive rows of one agent in one wave: state -> action -> next state. terminated: the
    next row is a death / finish; truncated: the agent's last row with the wave cut by time."""
    rows = np.sort(rows, order=["wave", "agent", "frame"])
    same = (rows["wave"][1:] == rows["wave"][:-1]) & (rows["agent"][1:] == rows["agent"][:-1]) & \
        (rows["frame"][1:] == rows["frame"][:-1] + 1)
    ok = same & (rows["action"][:-1] >= 0)
    s, n = rows[:-1][ok], rows[1:][ok]
    last = np.ones(len(rows), bool)
    last[:-1] = ~((rows["wave"][1:] == rows["wave"][:-1]) & (rows["agent"][1:] == rows["agent"][:-1]))
    trunc_rows = rows[last & (rows["status"] == STATUS_ALIVE)] if wave_ended_by_time else rows[:0]
    return {"s": s, "a": s["action"].astype(np.int64), "s2": n, "r": n["reward"],
            "terminated": n["status"] != STATUS_ALIVE, "truncated_rows": trunc_rows}


def stats(d: str):
    shards = load_shards(d)
    if not shards:
        print("no shards in", d)
        return
    rows = np.concatenate([r for _, r, _ in shards])
    tr = transitions(rows)
    lv = sorted({m.get("level") for m, _, _ in shards})
    print(f"{len(shards)} waves, levels {', '.join(map(str, lv))}")
    print(f"{len(rows)} agent rows, {len(tr['a'])} transitions, {int(tr['terminated'].sum())} ends "
          f"({int((tr['s2']['status'] == STATUS_DEAD).sum())} deaths, "
          f"{int((tr['s2']['status'] == STATUS_FINISHED).sum())} finishes), {len(tr['truncated_rows'])} cut by time")
    st, cnt = np.unique(tr["s"]["state"], return_counts=True)
    print("movement states:", ", ".join(f"{a}:{b}" for a, b in zip(st, cnt)))
    acts = np.bincount(tr["a"], minlength=10)
    print("actions:", " ".join(str(int(x)) for x in acts))
    dts = np.array([w["dt"] for _, _, world in shards for w in world if w.get("dt") is not None], np.float32)
    if len(dts):
        print(f"time step (ms): median {np.median(dts):.1f}, 90% {np.percentile(dts, 90):.1f}, "
              f"99% {np.percentile(dts, 99):.1f}, max {dts.max():.0f}; over 20 ms: {100 * (dts > 20).mean():.1f}%")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "stats":
        stats(sys.argv[2])
    else:
        print(__doc__)
