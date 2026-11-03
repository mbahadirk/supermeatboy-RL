"""Play the real Super Meat Boy with the explainable planner (plan/planner.py) instead of PPO (personal use).

Start the game, enter a level, then from the project root:

    python smbrl/agent.py --agents 1 --wm-collect
    python smbrl/agent.py --agents 1 --explore 40 --wm-collect

Same game side as realgame.py (realgame.js unchanged, the time step always locked): every agent is a real Meat
Boy; waves start together, one life each, and end when all are dead or finished, when --end-after agents have
finished, or after --wave-seconds. Agents 0..--agents-1 are planned; --explore K more agents play sticky random
actions mixed with random chains of library moves (plan/moves.py), only for world-model data.

Per level: game.level.Level (with the camera-bounds kill box), hazard boxes with their measured kill reach,
sealed pockets, the progress field V (plan/progress.py, cells outside the kill box closed), mpc.future.FutureWorld
(where the moving parts will be) and the beam planner. V is rebuilt between waves when moving hazards are seen
in new places, the kill box changes or a platform reaches new cells (at most FIELD_REBUILDS times per level).
Every frame: the level's live state is updated (tile changes, fans, platforms, lethal list, kill box), each
planned agent replans every --replan frames (otherwise it plays the next action of its current plan), the real
states are checked against earlier predictions (surprises) and against the moves' expected progress (failed
moves). A death of a planned agent the geometry does not explain (plan.safety.explain_death) is remembered
(plan.safety.DeathMemory, saved to --death-memory): the planner avoids that place at that wave frame from then on.

Logs in runs/<name>/: plans.jsonl (one Plan.to_dict() per decision, with the level, wave and the real state),
events.jsonl (surprises, deaths, failed moves, level / kill box / hazard / field messages, wave summaries),
states.jsonl (per frame: frame, wave, every planned agent's [i, x, y, vx, vy, status]), times.jsonl (every
finish, planned and explore: level, wave, agent, source, frames, seconds, lock, time of day); --wm-collect:
world-model data in runs/<name>/wm_data (shards' source: planner, explore or planner+explore, with agent_sources).
Best finish times per level (planned and explore apart, time-step lock on only): --best-times (default
smbrl/best_times.json; the old best.json is not used).
In the game (unless --no-overlay): the first planned agent's chosen path (green), predicted landings (yellow),
wall jumps (cyan) and the top rejected alternatives (faint red when vetoed, grey otherwise).
Telemetry (--telemetry host:port, default 127.0.0.1:47800, "off" to disable): viz/telemetry.py (schema there).
Stop: Ctrl+C or a file runs/<name>/STOP (the characters are handed back to the game).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game.level import Level, hazard_boxes, identify_level, HZ          # noqa: E402
from mpc.candidates import ACTIONS                                       # noqa: E402
from mpc.future import FutureWorld                                       # noqa: E402
from plan import safety as S                                             # noqa: E402
from plan.moves import LIBRARY, MAX_LEN                                  # noqa: E402
from plan.planner import Planner, future_inputs, state_row              # noqa: E402
from plan.progress import ProgressField                                  # noqa: E402
from viz import telemetry as T                                           # noqa: E402
from wm.rollout import next_held                                         # noqa: E402

JS = os.path.join(HERE, "realgame.js")
GAME_EXE = "SuperMeatBoy.exe"
MAX_AGENTS = 200                 # realgame.py: the game's 40 spare Meat Boys, more built at the level load
GAME_POOL = 40
DEFAULT_MODEL = os.path.join("runs", "wm_locked", "wm_model.pt")
DEFAULT_DEATHS = os.path.join(HERE, "death_memory.json")
DEFAULT_BEST = os.path.join(HERE, "best_times.json")
FIELD_REBUILDS = 4               # progress field rebuilds per level (between waves)
STICKY = 0.75                    # explore: keep the previous action with this probability
CHAIN_P = 0.03                   # explore: per free frame, chance to start a chain of 1-3 library moves
FLUSH_LINES = 200                # log lines between flushes (and at every wave end)
LEVEL_EVERY_S = 2.0              # telemetry: the level message again after this long
NOOP = [0, 0, 0, 0]

# overlay colours (0xAARRGGBB) and widths (pixels, realgame.js drawPaths)
COL_PLAN, COL_LAND, COL_WALL = 0xE020E040, 0xF0FFE020, 0xF020E0FF
COL_ALT_VETO, COL_ALT = 0x60FF4040, 0x50B0B0B0
OVERLAY_ALTS, PLAN_STEP, ALT_STEP = 3, 3, 4


def other_runners() -> list[str]:
    """PIDs of other realgame.py / agent.py processes (two runners in one game break each other)."""
    import subprocess
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                              "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    me = str(os.getpid())
    pat = re.compile(r"realgame\.py|(^|[\s\"'\\/])agent\.py")
    return [ln.split("|", 1)[0] for ln in out.splitlines()
            if "|" in ln and ln.split("|", 1)[0].strip() != me and pat.search(ln.split("|", 1)[1])
            and "pytest" not in ln]


def _sub(pts, step: int) -> list:
    """Every step-th point (and the last), rounded: a small overlay payload."""
    pts = list(pts)
    if not pts:
        return []
    out = pts[::step]
    if (len(pts) - 1) % step:
        out.append(pts[-1])
    return [[round(float(x), 1), round(float(y), 1)] for x, y in out]


def overlay_payload(plan) -> list:
    """realgame.js 'path' payload [{pts, col, w}] for one plan: the top rejected alternatives faintly (red when
    vetoed, grey otherwise), the chosen path (green), landings (yellow) and wall jumps (cyan) as dots."""
    if plan is None:
        return []
    out = []
    for a in [a for a in plan.alternatives if a.get("path")][:OVERLAY_ALTS]:
        if len(a["path"]) >= 2:
            out.append({"pts": _sub(a["path"], ALT_STEP), "col": COL_ALT_VETO if a.get("vetoed") else COL_ALT,
                        "w": 1})
    out.append({"pts": _sub(plan.path, PLAN_STEP), "col": COL_PLAN, "w": 3})
    for p in plan.landings[:4]:
        out.append({"pts": [[round(p["x"], 1), round(p["y"], 1)]], "col": COL_LAND, "w": 4})
    for p in plan.wall_jumps[:4]:
        out.append({"pts": [[round(p["x"], 1), round(p["y"], 1)]], "col": COL_WALL, "w": 4})
    return out


def followed_text(err: dict) -> str:
    """'followed error +1 0.2/0.4, +4 ... (median/90%, n 120/110/90/60)' from SurpriseTracker.errors()."""
    hs = [h for h in err if err[h][2]]
    if not hs:
        return "followed error: none"
    return ("followed error " + ", ".join(f"+{h} {err[h][0]:.1f}/{err[h][1]:.1f}" for h in hs)
            + " (median/90%, n " + "/".join(str(err[h][2]) for h in hs) + ")")


class BestTimes:
    """Best finish time per level, planned and explore agents apart:
    {level: {"planned": {seconds, frames, run, wave, agent, date}, "explore": {...}}}. Written atomically (temp
    file + replace); re-read before each update. A file that cannot be read is left alone (no updates)."""

    def __init__(self, path: str):
        self.path = path
        self.data, self.ok = self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return {}, True
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                raise ValueError("not a JSON object")
            return d, True
        except (OSError, ValueError) as e:
            print(f"[times] cannot read {self.path} ({e}): best times are not updated", flush=True)
            return {}, False

    def best(self, level: str, source: str) -> dict | None:
        return (self.data.get(level) or {}).get(source)

    def offer(self, level: str, source: str, rec: dict) -> bool:
        """rec becomes the level's best for source if it is faster; True when it did (and was saved)."""
        if not self.ok:
            return False
        d, self.ok = self._load()
        if not self.ok:
            return False
        self.data = d
        cur = (d.get(level) or {}).get(source)
        if cur is not None and cur.get("seconds") is not None and float(cur["seconds"]) <= float(rec["seconds"]):
            return False
        d.setdefault(level, {})[source] = rec
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[times] could not save {self.path}: {e}", flush=True)
            return False
        return True


def best_text(seconds, best: dict | None, new_best: bool, lock) -> str:
    """'4.25 s (best 4.25 s, new best)' for the console."""
    if lock is not True:
        return f"{seconds:.2f} s (time-step lock not confirmed: not a best time)"
    if best is None:
        return f"{seconds:.2f} s"
    return f"{seconds:.2f} s (best {float(best['seconds']):.2f} s" + (", new best)" if new_best else ")")


class Runner:
    def __init__(self, args, telemetry=None):
        self.args = args
        self.run_dir = os.path.join(HERE, "runs", args.name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.n_plan, self.n_explore = max(0, int(args.agents)), max(0, int(args.explore))
        self.n_total = self.n_plan + self.n_explore
        if self.n_total < 1:
            raise SystemExit("no agents: --agents and --explore are both 0")
        if self.n_total > MAX_AGENTS:
            raise SystemExit(f"{self.n_total} agents: at most {MAX_AGENTS} (--agents + --explore)")
        self.pool = max(GAME_POOL, self.n_total)
        self.replan = max(1, int(args.replan))
        self.model = None
        if self.n_plan:
            import torch
            from wm.model import WorldModel
            torch.set_num_threads(max(1, int(args.threads)))
            path = args.model if os.path.exists(args.model) else os.path.join(HERE, args.model)
            self.model = WorldModel.load(path).eval()
            self.model_path = path
        self.deaths_path = args.death_memory
        self.deaths = S.DeathMemory.load(self.deaths_path)
        self.f_plans = open(os.path.join(self.run_dir, "plans.jsonl"), "a", encoding="utf-8")
        self.f_events = open(os.path.join(self.run_dir, "events.jsonl"), "a", encoding="utf-8")
        self.f_states = open(os.path.join(self.run_dir, "states.jsonl"), "a", encoding="utf-8")
        self.f_times = open(os.path.join(self.run_dir, "times.jsonl"), "a", encoding="utf-8")
        self.best = BestTimes(args.best_times)
        self.lock = None                        # the time-step lock: True / False once realgame.js said so
        self.dt_ms = 16.667                     # the frame time of the last frame message (ms)
        self.lines = 0
        self.wm = None
        if args.wm_collect:
            from wm.buffer import Collector
            src = "planner" if not self.n_explore else ("explore" if not self.n_plan else "planner+explore")
            self.wm = Collector(os.path.join(self.run_dir, "wm_data"), source=src)
            self.wm.meta_extra = {"agent_sources": {"planner": list(range(self.n_plan)),
                                                    "explore": list(range(self.n_plan, self.n_total))}}
        if telemetry is not None:
            self.tel = telemetry
        else:
            addr = T.parse_addr(args.telemetry)
            self.tel = T.Telemetry(addr) if addr else None
        self.rng = np.random.default_rng(args.seed)
        self.level: Level | None = None
        self.planner: Planner | None = None
        self.planner_level = None
        self.field_summary = None
        self.fw = FutureWorld()
        self.protected = False
        self.script = None
        self.stopping = False
        self.stopped = threading.Event()
        self.t0 = time.time()
        self.wave_no = 0
        self.level_sent = 0.0
        self.new_wave(0)

    # ---- logs ------------------------------------------------------------------------------------------------
    def _write(self, f, obj):
        f.write(json.dumps(T._plain(obj), separators=(",", ":")) + "\n")
        self.lines += 1
        if self.lines >= FLUSH_LINES:
            self.flush()

    def flush(self):
        self.lines = 0
        for f in (self.f_plans, self.f_events, self.f_states, self.f_times):
            try:
                f.flush()
            except (OSError, ValueError):
                pass

    def event(self, kind: str, frame=None, **kw):
        ev = {"kind": kind, "frame": frame, **kw}
        self._write(self.f_events, {"time": round(time.time(), 3), "level": self.level.name if self.level else None,
                                    "wave": self.wave_no, **ev})
        if self.tel is not None:
            self.tel.send(T.event_message(self.level.name if self.level else None, self.wave_no, ev))

    # ---- level / waves ---------------------------------------------------------------------------------------
    def new_wave(self, n: int):
        self.alive = [True] * n
        self.held = [0] * n
        self.prev_xy = [None] * n
        self.last_state = [None] * n
        self.cur_plan = {}
        self.ex_queue = [[] for _ in range(n)]
        self.ex_prev = [-1] * n
        self.last_msg = None
        self.overlay_frame = None
        self.st = {"finished": 0, "died": 0, "explained": 0, "unexplained": 0, "estimated": 0, "memory": 0,
                   "surprise_pos": 0, "surprise_death": 0, "decisions": 0, "ms": [], "cut1": 0, "cut2": 0,
                   "failed": 0, "errors": 0, "ex_finished": 0, "ex_died": 0}
        self.fin = {"planned": [], "explore": []}     # this wave's finish times (s)
        self.new_best = False

    def on_info(self, info: dict, tiles: bytes):
        if info["W"] * info["H"] == 0 or len(tiles or b"") != info["W"] * info["H"] * 4:
            print(f"[level] ignoring invalid level data {info['W']}x{info['H']}", flush=True)
            return
        if info["name"] in (None, "unknown"):
            info["name"] = identify_level(info, tiles) or "unknown"
        self.level = lv = Level(info, tiles)
        from wm.buffer import eval_levels
        self.protected = info["name"] in eval_levels()
        if self.wm and not self.protected:
            self.wm.level(lv, info)
        self.fw.new_level()
        self.hz_start, self.moving, self.sweep = None, set(), set()
        self.field_built, self.field_dirty, self.rebuilds = False, False, 0
        self.lt_changes = 0
        self.wave_no = 0
        kb = lv.kill_box
        print(f"[level] {lv.name}: {lv.W}x{lv.H}, {len(lv.goals)} goal(s), {info.get('lethal', '?')} lethal pieces, "
              f"{len(lv.keys)} key(s), {info['agents']} agents (planned {self.n_plan}, explore {self.n_explore}); "
              + (f"kill box x {kb[0]:.0f}..{kb[1]:.0f}, y {kb[2]:.0f}..{kb[3]:.0f}" if kb else "no kill box")
              + (" - held-out evaluation level: no world-model data, no death memory" if self.protected else ""),
              flush=True)
        self.event("level", None, name=lv.name, W=lv.W, H=lv.H, kill_box=list(kb) if kb else None,
                   protected=self.protected, agents=info["agents"])
        self.new_wave(info["agents"])
        if "hz" in info:                       # the game waits for the 'go' (no agent moves meanwhile)
            boxes, kept = hazard_boxes(info["hz"], lv.hz_keep, lv.hz_grow)
            self.track_hazards(boxes, [lv.hz_ids[k] for k in kept])
            self.build_field("level start")
        self.send_level()

    def track_hazards(self, boxes: list, ids: list):
        """As realgame.Trainer.track_hazards: each level piece's start box and every place a moving one was seen
        (rounded to 4 units); a new place marks the field for a rebuild."""
        if self.hz_start is None:
            self.hz_start = {}
        level_ids = self.level.hz_level_ids
        for b, hid in zip(boxes, ids):
            if hid not in level_ids:
                continue
            if hid not in self.hz_start:
                self.hz_start[hid] = b
                continue
            x, y, hw, hh, a = b
            x0, y0, _, _, a0 = self.hz_start[hid]
            if abs(x - x0) > 2 or abs(y - y0) > 2 or abs(a - a0) > 0.05:
                if hid not in self.moving:
                    self.moving.add(hid)
                    self.sweep.add((round(x0 / 4) * 4, round(y0 / 4) * 4, hw, hh, round(a0, 1)))
                key = (round(x / 4) * 4, round(y / 4) * 4, hw, hh, round(a, 1))
                if key not in self.sweep:
                    self.sweep.add(key)
                    self.field_dirty = True

    def build_field(self, why: str):
        """The progress field from the pieces that stay put, the sweeps of the moving ones, fans, platforms,
        sealed pockets and the cells outside the kill box (as realgame.Trainer.build_paths builds its routes);
        the planner gets it (a new planner on the first level, set_level on a new one)."""
        lv = self.level
        t = time.time()
        static = [b for hid, b in (self.hz_start or {}).items() if hid not in self.moving]
        sweeps = sorted(self.sweep)
        seeds = [lv.spawn] + list(lv.goals) + list(lv.keys) + list(lv.bandages) + [p for pr in lv.portals for p in pr]
        lv.set_sealed(static, seeds)
        field = ProgressField(lv.words0, lv.ox, lv.oy, lv.spawn, lv.goals, static, sweeps, lv.keys, lv.bandages,
                              list(lv.portals), lift=lv.wind | lv.plat_support(),
                              blocks=lv.plat_blocks() | lv.sealed_haz | lv.sealed_wall, doors=lv.plat_doors(),
                              outside=lv.outside_cells if lv.kill_box is not None else None)
        self.field_summary = field.summary()
        if self.model is not None:
            if self.planner is None:
                self.planner = Planner(self.model, lv, field, budget_ms=self.args.budget_ms)
            elif self.planner_level is not lv:
                self.planner.set_level(lv, field)
            else:
                self.planner.field = field
                self.planner.bounds = S.level_bounds(lv)
            self.planner_level = lv
        self.field_built, self.field_dirty = True, False
        dt = time.time() - t
        print(f"[field] {lv.name} ({why}): {self.field_summary}; {len(static)} fixed hazards, {len(self.moving)} "
              f"moving, {len(sweeps)} sweep boxes [{dt:.2f}s]", flush=True)
        self.event("field", None, why=why, summary=self.field_summary, fixed=len(static), moving=len(self.moving),
                   seconds=round(dt, 3))

    def send_level(self):
        if self.tel is None or self.level is None:
            return
        self.level_sent = time.time()
        self.tel.send(T.level_message(self.level, self.wave_no, self.n_plan, self.n_explore, self.field_summary))

    # ---- one frame ---------------------------------------------------------------------------------------------
    def on_frame(self, m: dict, data=None) -> dict:
        lv, states = self.level, m["s"]
        n = len(states)
        if lv is None:
            return {"a": [NOOP] * n, "leader": 0}
        frame = int(m["frame"])
        self.wave_no = int(m.get("wave", self.wave_no))
        if len(self.alive) != n:
            self.new_wave(n)
        pl = self.planner
        # the level's live state (as realgame.Trainer.on_frame)
        if m.get("tc"):
            lv.apply_tiles(m["tc"])
            if pl is not None:
                pl.tiles(m["tc"])
        if m.get("fx"):
            lv.set_fields(m["fx"])
        if m.get("kb") and lv.set_kill_box(m["kb"]):
            kb = lv.kill_box
            print(f"[level] {lv.name}: kill box now x {kb[0]:.0f}..{kb[1]:.0f}, y {kb[2]:.0f}..{kb[3]:.0f} "
                  f"(frame {frame})", flush=True)
            if pl is not None:
                pl.bounds = S.level_bounds(lv)
            self.field_dirty = True
            self.event("kill box", frame, kill_box=list(kb))
        if m.get("pf") and lv.set_platforms(m["pf"]) and self.field_built:
            self.field_dirty = True
        if m.get("lt"):
            before = len(lv.hz_ids)
            lv.set_lethal(m["lt"]["types"], m["lt"]["ids"])
            self.lt_changes += 1
            if self.lt_changes <= 5:
                print(f"[hazards] {lv.name}: pieces came or went, {before} -> {len(lv.hz_ids)} lethal (frame {frame})",
                      flush=True)
                self.event("hazards", frame, before=before, after=len(lv.hz_ids))
        boxes, kept = hazard_boxes(m.get("hz", []), lv.hz_keep, lv.hz_grow)
        ids = [lv.hz_ids[k] if k < len(lv.hz_ids) else str(k) for k in kept]
        self.track_hazards(boxes, ids)
        if not self.field_built:
            self.build_field("first frame")
            pl = self.planner
        dt = float(m.get("dt") or 16.667)
        self.dt_ms = dt
        self.fw.observe(frame, m.get("hz", []), m.get("pf"), m.get("sh", []), lv.hz_keep, lv.fans, dt, lv.hz_grow)
        if self.wm and not self.protected:
            self.wm.frame(m, list(lv.hz_ids), list(lv.hz_keep))

        # what happened since the last frame: deaths, finishes, surprises, failed moves
        alive_before, held_before = list(self.alive), list(self.held)
        for i, s in enumerate(states):
            if not self.alive[i]:
                continue
            x, y, st = float(s[0]), float(s[1]), int(s[6])
            keys = int(s[7]) if len(s) > 7 else 0
            if i < self.n_plan and pl is not None:
                if st == 1:
                    self.on_death(i, frame, self.prev_xy[i], (x, y), keys, m, None, estimated=False)
                else:
                    self.observe(i, frame, x, y, keys, dead=False)
            if st != 0:
                self.alive[i] = False
                planned = i < self.n_plan
                if st == 2:
                    # realgame.js counts frame++ after each update: status 2 first shown in the message of
                    # frame f means the goal was touched in update f - 1
                    self.on_finish(i, frame - 1, x, y)
                elif planned and pl is None:
                    self.st["died"] += 1
                elif not planned:
                    self.st["ex_died"] += 1
            self.prev_xy[i] = (x, y)
            self.last_state[i] = s

        # decisions
        acts, act_idx = [NOOP] * n, [-1] * n
        fboxes = None
        live_plan = [i for i in range(min(self.n_plan, n)) if self.alive[i]] if pl is not None else []
        need = [i for i in live_plan if self.need_plan(i, frame)]
        if need:
            worlds, fboxes, _ = future_inputs(self.fw, frame, pl.H, dt, lv.name, self.deaths)
            for i in need:
                s = states[i]
                row = state_row(s[0], s[1], s[2], s[3], s[4], s[5], s[7] if len(s) > 7 else 0, held_before[i])
                try:
                    plan = pl.plan(row, frame, worlds, fboxes, dt, agent=i)
                except Exception as e:                  # never leave the agent without an answer
                    self.st["errors"] += 1
                    if self.st["errors"] <= 3:
                        import traceback
                        traceback.print_exc()
                    self.event("plan error", frame, agent=i, error=f"{type(e).__name__}: {e}")
                    continue
                self.cur_plan[i] = plan
                self.on_plan(plan, s)
        for i in live_plan:
            p = self.cur_plan.get(i)
            k = frame - p.frame if p is not None else -1
            a = int(p.actions[k]) if p is not None and 0 <= k < len(p.actions) else 0
            acts[i], act_idx[i] = list(ACTIONS[a]), a
            pl.played(i, frame, a)
        for i in range(self.n_plan, n):
            if self.alive[i] and int(states[i][6]) == 0:
                a = self.explore_action(i)
                acts[i], act_idx[i] = list(ACTIONS[a]), a
        for i in range(n):
            if act_idx[i] >= 0:
                self.held[i] = int(next_held(np.array([self.held[i]]), np.array([act_idx[i]]),
                                             np.array([int(states[i][4]) & 0xffffffff], np.int64))[0])
            else:
                self.held[i] = 0
        self.last_msg = m
        self.post_overlay(frame)
        won = [i for i, s in enumerate(states) if int(s[6]) == 2]
        live = [i for i, s in enumerate(states) if int(s[6]) == 0 and self.alive[i]]
        leader = won[0] if won else (live_plan[0] if live_plan else (live[0] if live else 0))
        self._write(self.f_states, {"frame": frame, "wave": self.wave_no, "level": lv.name,
                                    "agents": [[i, round(float(states[i][0]), 2), round(float(states[i][1]), 2),
                                                round(float(states[i][2]), 2), round(float(states[i][3]), 2),
                                                int(states[i][6])] for i in range(min(self.n_plan, n))]})
        if self.tel is not None:
            self.send_frame(m, frame, states, act_idx, boxes, kept, fboxes, dt)
        if self.wm and not self.protected:
            self.wm.agents(self.wave_no, frame, states, alive_before, held_before, act_idx, [float("nan")] * n)
        return {"a": acts, "leader": int(leader)}

    def need_plan(self, i: int, frame: int) -> bool:
        p = self.cur_plan.get(i)
        if p is None:
            return True
        k = frame - p.frame
        return k < 0 or k >= self.replan or k >= len(p.actions)

    def on_plan(self, plan, s):
        st = self.st
        st["decisions"] += 1
        st["ms"].append(plan.timing.get("total", 0.0))
        c = plan.counts
        if c.get("stage1", {}).get("steps", MAX_LEN) < MAX_LEN:
            st["cut1"] += 1
        s2 = c.get("stage2", {})
        if "skipped" in s2 or s2.get("steps", 0) < s2.get("of_steps", 0):
            st["cut2"] += 1
        d = plan.to_dict()
        d.update({"level": self.level.name, "wave": self.wave_no,
                  "real": {"x": s[0], "y": s[1], "vx": s[2], "vy": s[3], "flags": int(s[4]), "state": int(s[5]),
                           "status": int(s[6]), "keys": int(s[7]) if len(s) > 7 else 0}})
        self._write(self.f_plans, d)
        if self.tel is not None:
            self.tel.send(T.plan_message(self.level.name, self.wave_no, d))

    def observe(self, i, frame, x, y, keys, dead):
        """Planner.observe: surprises and failed moves, counted and logged."""
        for ev in self.planner.observe(i, frame, x, y, keys, dead):
            k = ev.get("kind")
            if k == "position":
                self.st["surprise_pos"] += 1
            elif k == "unpredicted death":
                self.st["surprise_death"] += 1
            elif k == "failed move":
                self.st["failed"] += 1
            self.event(k, ev.get("frame", frame), **{a: b for a, b in ev.items() if a not in ("kind", "frame")})

    def on_death(self, i, frame, xy0, xy1, keys, rec, extra_boxes, estimated: bool):
        """A planned agent died at wave frame `frame` at xy1 (from xy0 the frame before): does the geometry
        explain it (the move against the death frame's hazards and shots, or leaving the level)? If not, the
        place goes into the death memory (not on held-out levels) and is saved."""
        lv = self.level
        x1, y1 = xy1
        x0, y0 = xy0 if xy0 is not None else xy1
        explained = bool(S.explain_death(lv, rec or {}, lv.hz_keep, lv.hz_grow, x0, y0, x1, y1))
        fb = S.record_boxes(rec or {}, lv.hz_keep, lv.hz_grow)
        kind, box = S.segment_contact([x0], [y0], [x1], [y1], fb, S.level_bounds(lv))
        cause = S.KIND_NAMES[int(kind[0])] + (f" {fb.ids[int(box[0])]}" if box[0] >= 0 else "")
        if not explained and extra_boxes is not None and len(extra_boxes):   # wave-end deaths: also the predicted
            k2, b2 = S.segment_contact([x0], [y0], [x1], [y1], extra_boxes)  # boxes of the death frame
            if k2[0] != S.KIND_NONE:
                explained = True
                cause = "predicted " + S.KIND_NAMES[int(k2[0])] + (f" {extra_boxes.ids[int(b2[0])]}" if b2[0] >= 0 else "")
        added = False
        if not explained and not self.protected:
            self.deaths.add(lv.name, frame, x1, y1)
            try:
                d = os.path.dirname(os.path.abspath(self.deaths_path))
                os.makedirs(d, exist_ok=True)
                self.deaths.save(self.deaths_path)
                added = True
            except OSError as e:
                print(f"[deaths] could not save {self.deaths_path}: {e}", flush=True)
            self.st["memory"] += 1
        self.st["died"] += 1
        self.st["explained" if explained else "unexplained"] += 1
        self.st["estimated"] += int(estimated)
        if self.planner is not None:
            self.observe(i, frame, x1, y1, keys, dead=True)
        self.event("death", frame, agent=i, x=x1, y=y1, explained=explained, cause=cause if explained else "none",
                   estimated=estimated, memory_added=added)
        if not explained:
            print(f"[death] {lv.name} wave {self.wave_no} frame {frame}: agent {i} died at ({x1:.0f},{y1:.0f}) "
                  f"unexplained{' (position predicted)' if estimated else ''}"
                  + (" - remembered as a hazard" if added else ""), flush=True)

    def on_finish(self, i: int, finish_frame: int, x, y):
        """Agent i touched the goal in update `finish_frame` (0 = the wave's first update). Its time is the
        updates played up to and including that one, frames = finish_frame + 1, x the frame time (the
        messages' dt; 16.667 ms with the time-step lock) / 1000. Every finish goes to times.jsonl; with the
        lock on it is offered as the level's best (planned and explore apart); planned finishes are events."""
        planned = i < self.n_plan
        source = "planned" if planned else "explore"
        self.st["finished" if planned else "ex_finished"] += 1
        lv = self.level.name if self.level else None
        frames = int(finish_frame) + 1
        seconds = round(frames * self.dt_ms / 1000.0, 3)
        rec = {"level": lv, "wave": self.wave_no, "agent": int(i), "source": source, "frame": int(finish_frame),
               "frames": frames, "seconds": seconds, "dt_ms": round(self.dt_ms, 3), "lock": self.lock,
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._write(self.f_times, rec)
        self.fin[source].append(seconds)
        new = False
        if self.lock is True and lv is not None:
            new = self.best.offer(lv, source, {"seconds": seconds, "frames": frames, "run": self.args.name,
                                               "wave": self.wave_no, "agent": int(i), "date": rec["time"]})
        if planned:
            self.new_best |= new
            best = self.best.best(lv, "planned") if lv is not None else None
            self.event("finish", int(finish_frame), agent=i, x=x, y=y, frames=frames, seconds=seconds,
                       dt_ms=rec["dt_ms"], lock=self.lock, best=float(best["seconds"]) if best else None,
                       new_best=new)

    def explore_action(self, i: int) -> int:
        """Sticky random actions mixed with random chains of library moves (world-model data only)."""
        q = self.ex_queue[i]
        if not q and self.rng.random() < CHAIN_P:
            for _ in range(int(self.rng.integers(1, 4))):
                q.extend(LIBRARY[int(self.rng.integers(len(LIBRARY)))].actions)
        if q:
            a = int(q.pop(0))
        elif self.ex_prev[i] >= 0 and self.rng.random() < STICKY:
            a = self.ex_prev[i]
        else:
            a = int(self.rng.integers(len(ACTIONS)))
        self.ex_prev[i] = a
        return a

    def post_overlay(self, frame: int, force: bool = False):
        if self.args.no_overlay or self.script is None or not self.n_plan:
            return
        if not force and self.overlay_frame is not None and 0 <= frame - self.overlay_frame < self.replan:
            return
        self.overlay_frame = frame
        p = self.cur_plan.get(0) if self.alive and self.alive[0] else None
        try:
            self.script.post({"type": "path", "payload": overlay_payload(p)})
        except Exception:
            pass

    def send_frame(self, m, frame, states, act_idx, boxes, kept, fboxes, dt):
        lv = self.level
        agents = [{"i": i, "role": "planner" if i < self.n_plan else "explore", "x": float(s[0]), "y": float(s[1]),
                   "vx": float(s[2]), "vy": float(s[3]), "status": int(s[6]), "flags": int(s[4]) & 0xffffffff,
                   "state": int(s[5]), "keys": int(s[7]) if len(s) > 7 else 0, "action": int(act_idx[i])}
                  for i, s in enumerate(states)]
        now = T.box_rows(boxes, S.KIND_HAZARD)
        sh = m.get("sh", [])
        if sh:
            now += T.box_rows([(x, y, r, r, 0.0) for _, x, y, r in sh], S.KIND_SHOT)
        future = {}
        try:
            if fboxes is None or len(fboxes) <= max(T.FUTURE_AT):
                fboxes = future_inputs(self.fw, frame, max(T.FUTURE_AT), dt, lv.name, self.deaths)[1]
            future = {str(k): T.box_rows(fboxes[k].boxes, fboxes[k].kind) for k in T.FUTURE_AT}
        except Exception:
            future = {}
        mem = self.deaths.boxes(lv.name, frame)
        self.tel.send(T.frame_message(lv.name, self.wave_no, frame, agents, now, future,
                                      T.box_rows(mem, S.KIND_MEMORY)))
        if time.time() - self.level_sent >= LEVEL_EVERY_S:
            self.send_level()

    def on_wave_end(self, m: dict):
        lv = self.level
        if self.wm and not self.protected:
            self.wm.wave_end(m["wave"], m)
        self.wave_no = int(m.get("wave", self.wave_no))
        status = m.get("status", [])
        frame = int(m.get("frame", 0))
        # agents that died or finished on the wave's last update: no frame message shows them. realgame.js
        # sends their final states `s` with the hazards / shots of that moment: the death is checked exactly
        # as in on_frame. Without `s` (older script) the death position is the plan's prediction for that
        # frame, checked against the last frame's hazards and the ones predicted for the death frame
        final = m.get("s") or None
        box1 = None
        if final is None and self.last_msg is not None and self.fw.last is not None \
                and self.fw.last[0] == self.last_msg["frame"]:
            try:
                box1 = future_inputs(self.fw, int(self.last_msg["frame"]), 1, float(self.last_msg.get("dt") or 16.667),
                                     lv.name, self.deaths)[1][1] if lv is not None else None
            except Exception:
                box1 = None
        for i, st in enumerate(status):
            if i >= len(self.alive) or not self.alive[i] or st == 0:
                continue
            planned = i < self.n_plan
            if st == 2:
                # wave_end's `finish` is realgame.js finishFrame: the update (0-based) in which the agent
                # touched the goal; `frame` is the count after the last update (finish = frame - 1 when it
                # finished on the last update, as with a 'quota' end)
                fin = m.get("finish") or []
                ff = int(fin[i]) if i < len(fin) and fin[i] is not None and int(fin[i]) >= 0 else frame - 1
                last = final[i] if final is not None and i < len(final) else self.last_state[i]
                fx, fy = (float(last[0]), float(last[1])) if last is not None else (None, None)
                self.on_finish(i, ff, fx, fy)
            elif not planned:
                self.st["ex_died"] += 1
            elif self.planner is None or lv is None:
                self.st["died"] += 1
            elif final is not None and i < len(final):
                f = final[i]
                self.on_death(i, frame, self.prev_xy[i], (float(f[0]), float(f[1])),
                              int(f[7]) if len(f) > 7 else 0, m, None, estimated=False)
            elif self.last_state[i] is None:
                self.st["died"] += 1
            else:
                s = self.last_state[i]
                p = self.cur_plan.get(i)
                k = frame - p.frame if p is not None else -1
                if p is not None and 0 <= k < len(p.path):
                    x1, y1 = p.path[k]              # the plan's prediction for the death frame
                else:
                    x1, y1 = float(s[0]) + float(s[2]) / HZ, float(s[1]) + float(s[3]) / HZ
                self.on_death(i, frame, (float(s[0]), float(s[1])), (float(x1), float(y1)),
                              int(s[7]) if len(s) > 7 else 0, self.last_msg, box1, estimated=True)
            self.alive[i] = False
        st = self.st
        ms = np.array(st["ms"]) if st["ms"] else None
        el = (time.time() - self.t0) / 60
        err_f = self.planner.surprise.errors() if self.planner is not None else None
        fin_p = min(self.fin["planned"]) if self.fin["planned"] else None
        fin_x = min(self.fin["explore"]) if self.fin["explore"] else None
        best_p = self.best.best(lv.name, "planned") if lv is not None else None
        line = (f"[wave {self.wave_no}] {lv.name if lv else '?'} ({m.get('why')}, {frame} frames): planned "
                f"{st['finished']}/{self.n_plan} finished"
                + (f" in {best_text(fin_p, best_p, self.new_best, self.lock)}" if fin_p is not None else "")
                + f", {st['died']} died (geometry {st['explained']}, "
                f"unexplained {st['unexplained']}" + (f", {st['estimated']} at the wave end: position predicted"
                                                      if st['estimated'] else "") + ")"
                f" | surprises {st['surprise_pos'] + st['surprise_death']} (position {st['surprise_pos']}, death "
                f"{st['surprise_death']})" + (f" | {followed_text(err_f)}" if err_f else "")
                + f" | {st['decisions']} decisions"
                + (f", {np.median(ms):.0f}/{np.percentile(ms, 95):.0f} ms (median/95%)" if ms is not None else "")
                + f" | cut short: stage 1 {st['cut1']}, stage 2 {st['cut2']} | failed moves {st['failed']}"
                + (f" | explore {st['ex_finished']}/{self.n_explore} finished"
                   + (f" (fastest {fin_x:.2f} s)" if fin_x is not None else "") + f", {st['ex_died']} died"
                   if self.n_explore else "")
                + (f" | {st['errors']} plan errors" if st["errors"] else "")
                + (f" | death memory +{st['memory']} ({len(self.deaths)})" if st["memory"] else "")
                + f"  [{el:.1f} min]")
        print(line, flush=True)
        self.event("wave", frame, why=m.get("why"), frames=frame,
                   **{k: v for k, v in st.items() if k != "ms"},
                   ms_median=float(np.median(ms)) if ms is not None else None,
                   ms_95=float(np.percentile(ms, 95)) if ms is not None else None, err_followed=err_f,
                   finish_s=fin_p, explore_finish_s=fin_x, best_s=float(best_p["seconds"]) if best_p else None,
                   new_best=self.new_best, lock=self.lock)
        self.fw.new_wave()
        if self.planner is not None:
            self.planner.new_wave()                 # also clears the surprise check's predictions and errors
        if lv is not None and self.field_dirty and self.rebuilds < FIELD_REBUILDS:
            self.rebuilds += 1
            self.build_field(f"rebuild {self.rebuilds}/{FIELD_REBUILDS}: moving hazards / kill box / platforms")
        self.new_wave(len(status) if status else len(self.alive))
        self.post_overlay(0, force=True)
        self.flush()

    # ---- frida -------------------------------------------------------------------------------------------------
    def on_message(self, msg, data):
        if msg["type"] == "error":
            print("[script error]", msg.get("description"), msg.get("stack", ""), flush=True)
            return
        p = msg["payload"]
        ev = p.get("ev")
        reply = None
        try:
            if ev == "frame":
                reply = self.on_frame(p, data)
            elif ev == "info":
                self.on_info(p, data)
            elif ev == "wave_end":
                self.on_wave_end(p)
            elif ev == "level":
                print(f"[game] level {p['name']}", flush=True)
                self.level = None
                self.cur_plan = {}
            elif ev == "save":
                print(f"[S] finishing {p.get('level')} the game's way", flush=True)
                self.script.post({"type": "finish"})
            elif ev == "stopped":
                self.stopped.set()
            elif ev == "error":
                print(f"[game error] {p.get('msg')}", flush=True)
                if "time step lock" in str(p.get("msg", "")):
                    print("the time step could not be locked: the planner needs it (it assumes every future step is "
                          "the game's fixed step); stopping", flush=True)
                    self.lock = False
                    self.event("lock", None, ok=False, msg=p.get("msg"))
                    self.stopping = True
            elif ev == "debug":
                if "time step locked" in str(p.get("msg", "")):
                    self.lock = True
                    self.event("lock", None, ok=True)
                print(f"[game] {p.get('msg')}", flush=True)
            elif ev in ("ready", "wave", "continue", "level_end"):
                if ev != "wave":
                    print(f"[game] {ev} {p}", flush=True)
            else:
                print(f"[game] {p}", flush=True)
        except Exception:  # never leave the game waiting
            import traceback
            traceback.print_exc()
            reply = None
        if ev in ("info", "wave_end"):          # the game waits for this between waves / at the level start
            go = {"agents": self.n_total}
            if ev == "info" and self.level is not None and self.level.keys:
                go["keys"] = [list(k) for k in self.level.keys]
                go["locks"] = self.level.lock_groups
            self.script.post({"type": "go", "payload": go})
        if ev == "frame":
            if reply is None:
                reply = {"a": [NOOP] * len(p.get("s", [])), "leader": 0}
            if self.stopping:
                reply["stop"] = True
            self.script.post({"type": "act", "payload": reply})

    def close(self):
        self.flush()
        for f in (self.f_plans, self.f_events, self.f_states, self.f_times):
            try:
                f.close()
            except OSError:
                pass
        if self.tel is not None:
            print(f"telemetry: {self.tel.sent} sent, {self.tel.dropped} dropped, {self.tel.too_big} too big", flush=True)
            self.tel.close()

    def run(self):
        import frida
        import signal
        other = other_runners()
        if other:
            raise SystemExit(f"another realgame.py / agent.py is already attached to the game (pid "
                             f"{', '.join(other)}): stop it first (Ctrl+C in its terminal, or a file runs/<name>/STOP)")
        if self.pool > GAME_POOL:
            print(f"{self.pool} agents: {self.pool - GAME_POOL} extra Meat Boys are built when a level loads (the level "
                  f"on screen now, if any, has the game's {GAME_POOL}: enter or restart a level for all)", flush=True)
        cfg = {"agents": self.n_total, "pool": self.pool, "wave_frames": int(self.args.wave_seconds * HZ),
               "finish_frames": 60, "debug": False, "finish_quota": max(0, self.args.end_after), "probe_frames": 0,
               "lock_step": True}
        with open(JS, encoding="utf-8") as f:
            src = f.read().replace("__CONFIG__", json.dumps(cfg))
        try:
            session = frida.attach(GAME_EXE)
        except frida.ProcessNotFoundError:
            raise SystemExit(f"{GAME_EXE} is not running: start the game and enter a level")
        self.script = session.create_script(src)
        self.script.on("message", self.on_message)
        self.script.load()
        print(f"attached: {self.n_plan} planned + {self.n_explore} explore agents, waves of {self.args.wave_seconds:g}s, "
              f"replan every {self.replan} frames, budget {self.args.budget_ms:g} ms"
              + (f", model {self.model_path}" if self.model is not None else "")
              + f", death memory {self.deaths_path} ({len(self.deaths)} entries); logs in {self.run_dir}; "
                f"Ctrl+C to stop", flush=True)
        stop_file = os.path.join(self.run_dir, "STOP")   # create it to stop a background run safely
        if os.path.exists(stop_file):
            os.remove(stop_file)
        # Ctrl+C only asks for a stop: a second Ctrl+C in the middle of stopping must not leave the game waiting
        want_stop = threading.Event()

        def on_sigint(*_):
            if not want_stop.is_set():
                print("stopping (handing the characters back; Ctrl+C again is ignored)...", flush=True)
            want_stop.set()
        old_sigint = signal.signal(signal.SIGINT, on_sigint)
        try:
            while not os.path.exists(stop_file) and not want_stop.is_set() and not self.stopping:
                time.sleep(0.5)
            if os.path.exists(stop_file):
                os.remove(stop_file)
        finally:
            # stop through the next frame reply: the game thread is never left waiting for an answer
            self.stopping = True
            if not self.stopped.wait(3.0):
                try:   # no frames are coming (menu / level change): nothing is waiting, RPC is safe
                    self.script.exports_sync.shutdown()
                except Exception:
                    pass
            time.sleep(0.2)
            try:
                self.script.unload()
            except Exception:
                pass
            session.detach()
            self.close()
            signal.signal(signal.SIGINT, old_sigint)
            print(f"logs in {self.run_dir}; death memory {self.deaths_path} ({len(self.deaths)} entries)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=1, help="planned agents (default 1)")
    ap.add_argument("--explore", type=int, default=0,
                    help="extra agents with sticky random actions and random library move chains (world-model data)")
    ap.add_argument("--wave-seconds", type=float, default=20.0, help="wave time limit (default 20)")
    ap.add_argument("--end-after", type=int, default=1,
                    help="end the wave once this many agents finished (default 1; 0 = wait for all)")
    ap.add_argument("--replan", type=int, default=2, help="frames between a planned agent's decisions (default 2)")
    ap.add_argument("--budget-ms", type=float, default=120.0, help="planner time budget per decision (default 120)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="world model (default runs/wm_locked/wm_model.pt)")
    ap.add_argument("--name", default=None, help="run folder under runs/ (default plan_<time>)")
    ap.add_argument("--wm-collect", action="store_true", help="save world-model data to runs/<name>/wm_data")
    ap.add_argument("--no-overlay", action="store_true", help="do not draw the plan in the game")
    ap.add_argument("--telemetry", default="127.0.0.1:47800",
                    help="host:port for the live telemetry (viz/telemetry.py), or off (default 127.0.0.1:47800)")
    ap.add_argument("--death-memory", default=DEFAULT_DEATHS,
                    help="unexplained deaths remembered as hazards (loaded at start, saved on every new one; "
                         "default smbrl/death_memory.json)")
    ap.add_argument("--best-times", default=DEFAULT_BEST,
                    help="best finish times per level (default smbrl/best_times.json; time-step lock on only)")
    ap.add_argument("--threads", type=int, default=2, help="torch threads for the world model (default 2)")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the explore agents")
    ap.add_argument("--no-lock-step", action="store_true", help=argparse.SUPPRESS)
    return ap


def main():
    args = build_parser().parse_args()
    if args.no_lock_step:
        raise SystemExit("the planner needs the game's time step locked (it assumes every future step is the game's "
                         "fixed step): run without --no-lock-step")
    args.name = args.name or time.strftime("plan_%Y%m%d-%H%M%S")
    Runner(args).run()


if __name__ == "__main__":
    main()
