"""One frozen-model, one-life evaluation; reports never equate a declared split with unseen training data.

    python -m plan.first_eval report runs/first_eval_1-18_v1

Only agent.py --eval-first runs have the evidence required by this report. Older multi-wave logs are not
silently relabelled as first-encounter evaluations. Same-life observations and replanning are allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import time
from pathlib import Path

from levels_smb import a_plus_time

PROTOCOL = "first-life-v1"
SPAWN_TOL = 1.0
DT_MS = 16.667
DT_TOL = 0.01


def validate_args(args):
    if not getattr(args, "eval_first", None):
        return
    conflicts = [name for name in ("learn", "search", "demo", "demo_level", "wm_collect")
                 if getattr(args, name, False)]
    if getattr(args, "demo_play", None) is not None:
        conflicts.append("demo_play")
    if args.agents != 1 or args.explore != 0:
        conflicts.append("agents must be 1 and explore must be 0")
    if args.no_lock_step:
        conflicts.append("no_lock_step")
    if not math.isfinite(args.wave_seconds) or args.wave_seconds < 1 / 60:
        conflicts.append("wave_seconds must be finite and at least one frame (1/60 second)")
    if conflicts:
        raise SystemExit("--eval-first: incompatible options: " + ", ".join(conflicts))


def load_checkpoint(path):
    """Hash and load the very same bytes, even if the original path is replaced during startup."""
    from wm.model import WorldModel
    data = Path(path).read_bytes()
    model = WorldModel.load(io.BytesIO(data)).eval()
    from wm.protocol import feature_usage
    return model, {"path": os.path.abspath(path), "sha256": hashlib.sha256(data).hexdigest(),
                   "bytes": len(data), "step": model.step, "members": len(model.members),
                   "metadata": getattr(model, "info", {}), "features": feature_usage(model.info)}


def _write(path, value):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _git(root):
    def run(*args):
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                           timeout=5, creationflags=0x08000000 if os.name == "nt" else 0)
        return p.stdout.strip() if p.returncode == 0 else None
    try:
        return {"head": run("rev-parse", "HEAD"), "status": run("status", "--porcelain")}
    except (OSError, subprocess.TimeoutExpired):
        return {"head": None, "status": None}


class FirstEvaluation:
    def __init__(self, folder, args, identity, declared_holdout, root):
        self.folder = Path(folder)
        self.target = args.eval_first
        self.limit = int(args.wave_seconds * 60)
        self.started = False
        self.info_seen = False
        self.last_frame = None
        self.wave = None
        self.t0 = time.monotonic()
        self.started_at = None
        self.result = None
        levels = identity.get("metadata", {}).get("levels", [])
        overlap = "seen" if isinstance(levels, list) and self.target in levels else "unknown"
        effective = vars(args).copy()
        effective.update(replan=max(1, args.replan), threads=max(1, args.threads), model=identity["path"],
                         death_memory=None, move_memory=None, best_times=None)
        effective.update(wave_frames=self.limit, finish_quota=max(0, args.end_after),
                         modes_active=args.modes == "on" and identity["members"] >= 5)
        manifest = {"protocol": PROTOCOL, "target_level": self.target, "requested_config": vars(args).copy(),
                    "effective_config": effective,
                    "model": identity, "git": _git(root), "declared_holdout": bool(declared_holdout),
                    "training_overlap": overlap,
                    "unseen_evidence": "checkpoint lists target in its data" if overlap == "seen" else
                                       "not established; declared holdout alone is insufficient",
                    "rules": {"lives": 1, "planned_agents": 1, "explore_agents": 0,
                              "model_updates": False, "persistent_memory": False,
                              "prior_wave_future": False, "same_life_observations": True}}
        _write(self.folder / "run.json", manifest)
        self._save("incomplete", None, "no completed attempt", final=False)

    def _save(self, validity, outcome, reason, final=True, **kw):
        r = {"protocol": PROTOCOL, "level": self.target, "validity": validity, "outcome": outcome,
             "reason": reason, "start_verified": self.started, "wave": self.wave,
             "last_observed_frame": self.last_frame, "wall_seconds": round(time.monotonic() - self.t0, 6),
             "attempt_wall_seconds": round(time.monotonic() - self.started_at, 6)
                                     if self.started_at is not None else None, **kw}
        _write(self.folder / "evaluation.json", r)
        if final:
            self.result = r

    def invalidate(self, reason):
        if self.result is None:
            self._save("invalid", None, reason)

    def interrupt(self, reason):
        if self.result is None:
            self._save("incomplete", None, reason)

    def info(self, level, agents):
        if self.info_seen:
            self.interrupt("level was reloaded before the attempt completed")
        elif level != self.target or agents != 1:
            self.invalidate(f"expected level {self.target} with one agent, got {level} with {agents}")
        self.info_seen = True
        return self.result is None

    def observe(self, m, spawn, lock):
        if self.result is not None:
            return False
        f, w, states = m.get("frame"), m.get("wave"), m.get("s", [])
        dt = float(m.get("dt", 0))
        if lock is not True or not math.isfinite(dt) or abs(dt - DT_MS) > DT_TOL:
            self.invalidate("fixed time step was not confirmed")
        elif not self.info_seen or len(states) != 1 or not isinstance(f, int):
            self.invalidate("missing level info, frame or single-agent state")
        elif not self.started:
            s = states[0]
            if f != 0 or w != 1 or int(s[6]) != 0 or not all(math.isfinite(v) for v in s[:4]) or \
                    math.hypot(s[0] - spawn[0], s[1] - spawn[1]) > SPAWN_TOL:
                self.invalidate("attempt did not start alive at frame 0, wave 1 and the level spawn")
            else:
                self.started, self.wave, self.started_at = True, w, time.monotonic()
        elif w != self.wave or f != self.last_frame + 1:
            self.invalidate("frame sequence skipped or a second life started")
        if self.result is None:
            self.last_frame = f
            return True
        return False

    def finish(self, m, lock, metrics=None):
        if self.result is not None:
            return
        f, status, why = m.get("frame"), m.get("status", []), m.get("why")
        if not self.started or lock is not True or len(status) != 1 or f != self.last_frame + 1 or \
                m.get("wave") != self.wave:
            self.invalidate("terminal message lacks a verified, contiguous first life")
            return
        if why not in ("done", "quota", "time"):
            self.interrupt(f"attempt ended by {why!r}")
            return
        s = status[0]
        if s == 2:
            fin = m.get("finish", [])
            if len(fin) != 1 or not isinstance(fin[0], int) or fin[0] != f - 1:
                self.invalidate("finish frame is missing or inconsistent with the single-agent terminal frame")
                return
            outcome = "success"
        elif s == 1:
            outcome = "death"
        elif s == 0 and why == "time" and f >= self.limit:
            outcome = "timeout"
        else:
            self.invalidate("terminal status and reason are inconsistent")
            return
        secs = round(f * DT_MS / 1000.0, 6)
        par = a_plus_time(self.target)              # the game's A+ time: the target (levels_smb.a_plus_time)
        self._save("valid", outcome, "first life completed", frames=f, game_seconds=secs, a_plus_seconds=par,
                   a_plus=(outcome == "success" and secs <= par) if par is not None else None, metrics=metrics or {})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("report",))
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args(argv)
    print("level | validity | outcome | game seconds | A+ seconds | A+ | attempt wall seconds | training overlap | "
          "declared holdout")
    for folder in args.runs:
        p = Path(folder)
        manifest = json.loads((p / "run.json").read_text(encoding="utf-8"))
        result = json.loads((p / "evaluation.json").read_text(encoding="utf-8"))
        if manifest.get("protocol") != PROTOCOL or result.get("protocol") != PROTOCOL:
            raise SystemExit(f"{p}: no {PROTOCOL} evidence; legacy runs cannot establish this protocol")
        print(f"{result['level']} | {result['validity']} | {result['outcome']} | "
              f"{result.get('game_seconds', '-')} | {result.get('a_plus_seconds', '-')} | "
              f"{ {True: 'yes', False: 'no'}.get(result.get('a_plus'), '-')} | {result.get('attempt_wall_seconds')} | "
              f"{manifest['training_overlap']} | "
              f"{manifest['declared_holdout']}")
        if result['validity'] != 'valid':
            print("  " + result['reason'])


if __name__ == "__main__":
    main()
