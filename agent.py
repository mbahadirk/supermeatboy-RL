"""Play the real Super Meat Boy with the explainable planner (plan/planner.py) instead of PPO (personal use).

Start the game, enter a level, then from the project root:

    python smbrl/agent.py --agents 1 --wm-collect
    python smbrl/agent.py --agents 1 --explore 40 --wm-collect

Same game side as realgame.py (realgame.js unchanged, the time step always locked): every agent is a real Meat
Boy; waves start together, one life each, and end when all are dead or finished, when --end-after agents have
finished, or after --wave-seconds. Agents 0..--agents-1 are planned; --explore K more agents, for world-model
data: with planning modes on (--modes, default) each plans every 16 frames with Planner.explore (members 3 and 4;
the safe plan they disagree on most), with --modes off (or without a planned agent, so without a model) they play
sticky random actions mixed with random chains of library moves (plan/moves.py).
World model: runs/wm_step4 by default (4 frames per model step; --model runs/wm_locked/wm_model.pt for the 1-frame
one). Planning modes (plan/planner.py): long-safe (member 0), normal (member 1), reflex (member 2, also on the
commit "danger" trigger); agent 0's chosen path is drawn in the game in its mode's colour (MODE_COL).

Per level: game.level.Level (with the camera-bounds kill box), hazard boxes from the game's kill shapes,
sealed pockets, the progress field V (plan/progress.py, cells outside the kill box closed), mpc.future.FutureWorld
(where the moving parts will be) and the beam planner. V is rebuilt between waves when moving hazards are seen
in new places, the kill box changes or a platform reaches new cells (at most FIELD_REBUILDS times per level).
Every frame: the level's live state is updated (tile changes, fans, platforms, lethal list, kill box), each
planned agent plays its committed plan and gets a new decision only when a trigger fires (plan/commit.py: start,
--replan frames since the last decision, the plan's first move ended, the real state left the prediction's
error tube (--tube) or its support, new danger on the remaining path, fewer than 16 frames left, a predicted
goal or key touch that did not happen; --triggers off: every --replan frames only, the old behaviour with
--replan 2); each decision's triggers go into its plans.jsonl line ("trigger"). The real
states are checked against earlier predictions (surprises) and against the moves' expected progress (failed
moves). Plans are scored in estimated frames to the goal (plan/planner.py); --hysteresis is in frames (default 3, chosen from closed loops);
each decision's inputs reach H + the airborne completion's 45 frames. A death of a planned agent the geometry does
not explain (plan.safety.explain_death) is remembered (plan.safety.DeathMemory, saved to --death-memory): the
planner avoids that place at that wave frame from then on. Every death of a planned agent (explained or not)
charges the move taken at its point of no return (the latest decision of the wave with a safe alternative;
plan.planner.MoveMemory, per level in --move-memory, default smbrl/move_memory.json, saved at every wave end;
none: off; held-out levels: nothing charged) and logs a "point of no return" event.

Logs in runs/<name>/: plans.jsonl (one Plan.to_dict() per decision, with the level, wave and the real state),
events.jsonl (surprises, deaths, failed moves, points of no return, level / kill box / hazard / field messages,
wave summaries),
states.jsonl (per frame: frame, wave, every planned agent's [i, x, y, vx, vy, status]; --demo: every agent's),
times.jsonl (every finish, planned and explore: level, wave, agent, source, frames, seconds, lock, time of day);
--wm-collect: world-model data in runs/<name>/wm_data (shards' source: planner, explore or planner+explore, with
agent_sources). A wave cut by S, a level change or Ctrl+C / STOP is still written, as a shard marked partial.
Best finish times per level (planned and explore apart, time-step lock on only): --best-times (default
smbrl/best_times.json; the old best.json is not used); the [level] line and the game's timer sign
("BEST x.xx", at the level start and on every new best) show the planned agents' best.
S in the game (focused) finishes the level the game's way and writes the planned agents' best for it as the
level's time (the game keeps it only if better than its record); 999.99 without one. The save file
(UserData/savegame.dat next to the game) is copied to smbrl/backups/save_<date-time>/ before the first write.
In the game (unless --no-overlay): the first planned agent's chosen path (green), its airborne completion (fainter
green, dashed) with its landing (yellow), predicted landings (yellow),
wall jumps (cyan), where it touches the princess (pink) or picks a key (gold), and the top rejected
alternatives (faint red when vetoed, grey otherwise).
Demo replay (--demo <best.json> or --demo-level <level>, plan/demos.py; run  python -m plan.demos inventory  first):
no planner and no model; agent 0 plays a PPO agent's best finish (realgame.py's levels/<level>/best.json) frame by
frame from the wave start with the time-step lock, --demo-copies K (default 20) more agents play it with 1-3 small
seeded changes each (an action change shifted 1-4 frames, a jump released 1-4 frames early or late, 4-12 frames of
a random action; new per wave, logged in events.jsonl and the shard meta "perturbations"). A wave ends when every
agent finished or died, or 1 s after the demo's actions ran out (plan.demos.DemoPlay.end_frame: copies left
standing count as done; why "demo done"), or at --wave-seconds (raised to the demo + 2 s). Each wave checks that agent 0 finished at the demo's
frame (+-1) and warns when it did not (the demo is not reproducible, e.g. recorded unlocked). Held-out levels,
the levels the user excluded (plan.demos.USER_EXCLUDED: 1-16, 4-6), SUSPICIOUS inventory entries and another level
loaded in the game are refused. Run folder runs/demo_<level>_<time>;
--wm-collect shards have source "demo" (agent 0) / "demo-noisy" (the copies), as agent_sources. Overlay: the
demo's path in white (recorded positions, else agent 0's path as it plays). No best times are offered.
    python smbrl/agent.py --demo-level 1-12 --wm-collect
Train mode (--learn; turns on --wm-collect, not with --demo): wm.learner runs as a child process (below-normal
priority, 2 threads; from the default model, or from --live-dir's published one when that is newer; stopped on
exit). It fine-tunes on every locked wm_data folder plus the demos whenever 20 new shards or a new self-demo exist
and publishes runs/wm_live/wm_model.pt only when its error at +16 and +48 frames is not worse (runs/wm_live/
learner.log). Between waves (never mid-wave) a newer published model replaces the planner's: "[learn] model v<N>
loaded (+16 error a -> b)". A planned agent's new best time (lock on, not a held-out level) appends its wave rows
to runs/demos/self.json (--self-demos), which the learner trains on as a demo.
    python smbrl/agent.py --learn
Wave search (--search, --search-agents N, default 50; plan/search.py, PLAN_GAME_SEARCH.md section 5): no planner,
no model ("none (search)" in the level event); the game is the simulator. Agent 0 replays the current best node
(verification), the others replay picked nodes' actions exactly from the wave start and branch (library move chains,
perturbed copies of longer known sequences or, after the first finish, of the fastest finish); the states at the
move ends become nodes (soonest per cell key; crumble guard: only states whose agent broke its own blocks). A wave
ends when every sequence ran out (+ plan.search.SETTLE frames: why "search done") or every agent is done; --end-after
is 0 and --wave-seconds at least 60 (the game's limit). Logs: search.jsonl (per wave: picks, branches, insertions,
verification, finishes, best T, archive size), states.jsonl with every agent, finish and death events and
times.jsonl rows of every agent (source "search"), best_times.json under "search", the wave event's replay_route
(a trusted finish was replayed) and all_agent_frames; --wm-collect shards have source "search". Not with --agents /
--explore (set to 0), --demo or --learn. Held-out levels are searched too (no shards, no memory; the archive is
in memory for the run only). S writes the planned agents' best as before (none in a search run: 999.99).
    python smbrl/agent.py --search --wm-collect
Telemetry (--telemetry host:port, default 127.0.0.1:47800, "off" to disable): viz/telemetry.py (schema there).
The wave summary line also says how often the planned agents decided: decisions per 100 alive frames, the
median frames between an agent's decisions, the triggers by kind, the switch rate (decisions that switched
plans / all decisions), the planning ms per game frame (the decisions' own time, plus the trigger checks) and
the share of decisions whose chosen plan the planner's collision guard clipped ("chosen plans clipped").
Per wave, after the summary line, a [cost] line (FrameCost): ms per frame (mean, median, 95%) spent planning,
observing (surprises, failed moves, deaths), on the explore agents, world-model data, telemetry, file logging, the
overlay, the level's live state, the rest, and waiting for the game (from the reply to the next frame message),
plus the game speed (game seconds per wall-clock second); the same numbers go into the wave event.
Stop: Ctrl+C or a file runs/<name>/STOP (the characters are handed back to the game, then the process exits).
A second Ctrl+C, or no clean stop within 10 s, unloads the script and exits at once (shutdown.py).
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

from game.level import Level, box_base, box_ids, hazard_boxes, identify_level, HZ          # noqa: E402
from mpc.candidates import ACTIONS                                       # noqa: E402
from mpc.future import FutureWorld                                       # noqa: E402
from plan import safety as S                                             # noqa: E402
from plan.commit import Committer, REPLAN_DEFAULT, TUBE_DEFAULT, kind_of  # noqa: E402
from plan.moves import LIBRARY                                           # noqa: E402
from plan.planner import (EXPLORE_EVERY, HYSTERESIS_FRAMES, MoveMemory, Planner, future_inputs,   # noqa: E402
                          state_row)
from plan.progress import ProgressField                                  # noqa: E402
from shutdown import Shutdown, bounded                                   # noqa: E402
from viz import telemetry as T                                           # noqa: E402
from wm.rollout import next_held                                         # noqa: E402

JS = os.path.join(HERE, "realgame.js")
GAME_EXE = "SuperMeatBoy.exe"
MAX_AGENTS = 200                 # realgame.py: the game's 40 spare Meat Boys, more built at the level load
GAME_POOL = 40
DEFAULT_MODEL = os.path.join("runs", "wm_step4", "wm_model.pt")      # 4 frames per step (wm.train --step 4)
DEFAULT_DEATHS = os.path.join(HERE, "death_memory.json")
DEFAULT_MOVES = os.path.join(HERE, "move_memory.json")
DEFAULT_BEST = os.path.join(HERE, "best_times.json")
LIVE_DIR = os.path.join(HERE, "runs", "wm_live")       # --learn: wm.learner's published model and log
SELF_DEMOS = os.path.join(HERE, "runs", "demos", "self.json")   # --learn: the planned agents' new bests as demos
BACKUPS = os.path.join(HERE, "backups")         # S: the game's save is copied here once per run before a time write
SAVE_FILE = os.path.join("UserData", "savegame.dat")   # next to the game's executable
FIELD_REBUILDS = 4               # progress field rebuilds per level (between waves)
STICKY = 0.75                    # explore: keep the previous action with this probability
CHAIN_P = 0.03                   # explore: per free frame, chance to start a chain of 1-3 library moves
FLUSH_LINES = 200                # log lines between flushes (and at every wave end)
LEVEL_EVERY_S = 2.0              # telemetry: the level message again after this long
NOOP = [0, 0, 0, 0]

# overlay colours (0xAARRGGBB) and widths (pixels, realgame.js drawPaths)
COL_PLAN, COL_LAND, COL_WALL = 0xE020E040, 0xF0FFE020, 0xF020E0FF
COL_ALT_VETO, COL_ALT = 0x60FF4040, 0x50B0B0B0
COL_GOAL, COL_KEY = 0xF0FF78B9, 0xF0FFE628      # where the plan touches the princess / picks a key
COL_COMP = 0x9020E040                             # the airborne completion after the plan (fainter, dashed)
# the chosen path in its planning mode's colour (plan/planner.py MODES; viz/viewer.py uses the same colours)
MODE_COL = {"long-safe": 0xE04080FF, "normal": COL_PLAN, "reflex": 0xE0FF3030, "explore": 0xE0B040F0,
            "best-route": 0xE0FFC020}
OVERLAY_ALTS, PLAN_STEP, ALT_STEP = 3, 3, 4
COL_DEMO = 0xE0FFFFFF                             # --demo: the demo's path (white)
DEMO_TRAIL_EVERY = 15                             # --demo without recorded positions: redraw agent 0's path this often
SEARCH_AGENTS = 50                                # --search: agents per wave (the user: 50 is enough)
SEARCH_WAVE_SECONDS = 60.0                        # --search: the game's wave limit at least this (search waves end
                                                  # themselves when every sequence ran out: plan/search.py)


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


def game_exe() -> str | None:
    """The running game's executable path (a Win32_Process query, read-only), None when not found."""
    import subprocess
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"Get-CimInstance Win32_Process -Filter \"Name='{GAME_EXE}'\" | "
                              "ForEach-Object { $_.ExecutablePath }"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    paths = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return paths[0] if paths else None


def locate_save(exe: str | None = None) -> tuple[str | None, str | None]:
    """(the game's save file, None) or (None, why not): UserData/savegame.dat next to the running game's
    executable (the game keeps its level records there)."""
    exe = exe or game_exe()
    if not exe:
        return None, f"the running {GAME_EXE} was not found"
    p = os.path.join(os.path.dirname(exe), SAVE_FILE)
    return (p, None) if os.path.isfile(p) else (None, f"no save file at {p}")


def backup_save(save: str, root: str = BACKUPS) -> str:
    """Copies the save file into root/save_<date-time>/ (a new folder); returns that folder."""
    import shutil
    stamp = time.strftime("save_%Y%m%d-%H%M%S")
    d, k = os.path.join(root, stamp), 1
    while os.path.exists(d):
        k += 1
        d = os.path.join(root, f"{stamp}_{k}")
    os.makedirs(d)
    shutil.copy2(save, os.path.join(d, os.path.basename(save)))
    return d


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
    """realgame.js 'path' payload [{pts, col, w, dash}] for one plan: the top rejected alternatives faintly (red
    when vetoed, grey otherwise), the chosen path (green), its airborne completion (Plan.completion: fainter
    green, dashed) with its landing, landings (yellow) and wall jumps (cyan) as dots, and where it touches the
    princess (pink) or picks a key (gold) as larger dots."""
    if plan is None:
        return []
    out = []
    for a in [a for a in plan.alternatives if a.get("path")][:OVERLAY_ALTS]:
        if len(a["path"]) >= 2:
            out.append({"pts": _sub(a["path"], ALT_STEP), "col": COL_ALT_VETO if a.get("vetoed") else COL_ALT,
                        "w": 1})
    out.append({"pts": _sub(plan.path, PLAN_STEP), "col": MODE_COL.get(getattr(plan, "mode", "normal"), COL_PLAN),
                "w": 3})
    comp = getattr(plan, "completion", None)
    if comp and len(comp.get("path") or []) >= 2:
        out.append({"pts": _sub(comp["path"], PLAN_STEP), "col": COL_COMP, "w": 2, "dash": True})
        if comp.get("on") in ("ground", "wall", "goal"):
            out.append({"pts": [[round(comp["x"], 1), round(comp["y"], 1)]], "col": COL_LAND, "w": 4})
    for p in plan.landings[:4]:
        out.append({"pts": [[round(p["x"], 1), round(p["y"], 1)]], "col": COL_LAND, "w": 4})
    for p in plan.wall_jumps[:4]:
        out.append({"pts": [[round(p["x"], 1), round(p["y"], 1)]], "col": COL_WALL, "w": 4})
    touches = [(p, COL_KEY) for p in (getattr(plan, "key_touches", None) or [])[:2]]
    if getattr(plan, "goal_touch", None):
        touches.append((plan.goal_touch, COL_GOAL))
    for p, col in touches:
        out.append({"pts": [[round(p["x"], 1), round(p["y"], 1)]], "col": col, "w": 6})
    return out


def followed_text(err: dict) -> str:
    """'followed error +1 0.2/0.4, +4 ... (median/90%, n 120/110/90/60)' from SurpriseTracker.errors()."""
    hs = [h for h in err if err[h][2]]
    if not hs:
        return "followed error: none"
    return ("followed error " + ", ".join(f"+{h} {err[h][0]:.1f}/{err[h][1]:.1f}" for h in hs)
            + " (median/90%, n " + "/".join(str(err[h][2]) for h in hs) + ")")


class FrameCost:
    """Where each frame's wall time goes (perf_counter only). A frame runs from its message's arrival to the
    reply being posted; go(cat) charges the time since the last switch to the current category and switches
    (returns the previous one, to switch back after a nested part). wait: from posting a reply to the next
    frame message (the game's own work and the message transport)."""
    CATS = ("planning", "observe", "explore", "wm_data", "telemetry", "logging", "overlay", "level", "other")
    SHORT = {"planning": "plan", "wm_data": "wm", "telemetry": "tel", "logging": "log"}

    def __init__(self):
        self.cat = None
        self.reset()

    def reset(self):
        """A new wave: the per-frame rows start again (the wait after a wave end is not a frame's wait)."""
        self.rows, self.wait = [], []
        self.t_first = self.sent = None

    def begin(self):
        now = time.perf_counter()
        if self.sent is not None:
            self.wait.append(now - self.sent)
        if self.t_first is None:
            self.t_first = now
        self.acc = dict.fromkeys(self.CATS, 0.0)
        self.cat, self.t = "other", now

    def go(self, cat):
        if self.cat is None or cat is None:         # outside a frame (level start, wave end)
            return None
        now = time.perf_counter()
        self.acc[self.cat] += now - self.t
        prev, self.cat, self.t = self.cat, cat, now
        return prev

    def end(self):
        if self.cat is None:
            return
        now = time.perf_counter()
        self.acc[self.cat] += now - self.t
        self.rows.append([self.acc[c] for c in self.CATS])
        self.cat, self.sent = None, now

    def summary(self, frames: int, dt_ms: float) -> dict | None:
        """{cat: [mean, median, 95%] ms per frame, ..., "python", "wait"}, game_speed (game s per wall s)."""
        if not self.rows:
            return None
        a = 1000.0 * np.asarray(self.rows)
        stat = lambda v: [round(float(np.mean(v)), 2), round(float(np.median(v)), 2),
                          round(float(np.percentile(v, 95)), 2)]
        out = {c: stat(a[:, k]) for k, c in enumerate(self.CATS)}
        out["python"] = stat(a.sum(1))
        if self.wait:
            out["wait"] = stat(1000.0 * np.asarray(self.wait))
        wall = time.perf_counter() - self.t_first
        game = frames * dt_ms / 1000.0
        return {"cost_ms": out, "wall_s": round(wall, 2), "game_s": round(game, 2),
                "game_speed": round(game / wall, 3) if wall > 0 else None}

    def text(self, s: dict) -> str:
        c = s["cost_ms"]
        f = lambda v: f"{v[0]:.1f} ({v[1]:.1f}/{v[2]:.0f})"
        return ("[cost] ms/frame mean (median/95%): " + " ".join(f"{self.SHORT.get(k, k)} {f(c[k])}" for k in self.CATS)
                + f" | python {f(c['python'])}" + (f" | wait {f(c['wait'])}" if "wait" in c else "")
                + f" | {s['game_s']:.1f} game s in {s['wall_s']:.1f} s: speed {s['game_speed']}x")


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

    def reload(self):
        """Read the file again (another run may have written it); a file that cannot be read keeps the data."""
        d, ok = self._load()
        if ok:
            self.data = d

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


def commit_summary(st: dict, frames: int) -> dict:
    """How often the planned agents decided in a wave (Runner.st): decisions per 100 alive agent frames, the
    median frames between an agent's decisions, the switch rate (switched / decisions), the planning ms per
    game frame (the decisions' own time), the trigger checks' ms per game frame and the share of decisions
    whose chosen plan the collision guard clipped (plan/planner.py collide: the world model's imagined body
    went into solid tiles)."""
    d, af = st["decisions"], st["agent_frames"]
    return {"decisions_per_100": round(100.0 * d / af, 2) if af else None,
            "gap_median": float(np.median(st["gaps"])) if st["gaps"] else None,
            "switch_rate": round(st["switched"] / d, 4) if d else None,
            "plan_ms_per_frame": round(sum(st["ms"]) / frames, 2) if frames else None,
            "check_ms_per_frame": round(1000.0 * st["check_s"] / frames, 3) if frames else None,
            "clipped_share": round(st.get("clipped", 0) / d, 4) if d else None}


def commit_text(c: dict, triggers: dict | None = None) -> str:
    """'7.1 decisions/100 frames, every 8 f (median), switched 20%, plan 9.8 ms/frame (+checks 1.2); triggers
    interval 30, move end 12'."""
    f = lambda v, spec: "-" if v is None else format(v, spec)
    tr = ", ".join(f"{k} {v}" for k, v in sorted((triggers or {}).items(), key=lambda kv: -kv[1]))
    return (f"{f(c['decisions_per_100'], '.1f')} decisions/100 frames, every {f(c['gap_median'], '.0f')} f "
            f"(median), switched {f(None if c['switch_rate'] is None else 100 * c['switch_rate'], '.0f')}%, plan "
            f"{f(c['plan_ms_per_frame'], '.1f')} ms/frame (+checks {f(c['check_ms_per_frame'], '.1f')})"
            + (f", chosen plans clipped {100 * c['clipped_share']:.0f}%" if c.get("clipped_share") is not None else "")
            + (f"; triggers {tr}" if tr else ""))


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
        self.demo = getattr(args, "demo_play", None)       # plan.demos.DemoPlay (--demo / --demo-level)
        self.demo_fin, self.demo_trail = None, []
        if self.demo is not None:
            args.agents, args.explore = 0, 0
        self.search = None                                   # plan.search.Search (--search)
        n_search = 0
        if getattr(args, "search", False):
            if self.demo is not None:
                raise SystemExit("--search and --demo / --demo-level: give one")
            from plan.search import Search
            args.agents, args.explore, args.end_after = 0, 0, 0      # every agent plays its sequence out
            args.wave_seconds = max(float(args.wave_seconds), SEARCH_WAVE_SECONDS)
            n_search = int(args.search_agents)
            if not 1 <= n_search <= MAX_AGENTS:
                raise SystemExit(f"--search-agents {n_search}: 1 to {MAX_AGENTS}")
            self.search = Search(n_search, args.seed, int(args.wave_seconds * HZ))
        self.n_plan, self.n_explore = max(0, int(args.agents)), max(0, int(args.explore))
        self.n_total = self.n_plan + self.n_explore + (self.demo.n if self.demo is not None else 0) + n_search
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
        self.learn = bool(getattr(args, "learn", False)) and self.model is not None
        self.learner = None                     # --learn: the wm.learner child process
        self.live_mtime = os.path.getmtime(self.model_path) if self.model is not None else 0.0
        self.commit = Committer(self.replan, args.tube, args.triggers == "on",
                                step=getattr(self.model, "step", 1) if self.model is not None else 1)
        self.deaths_path = args.death_memory
        self.deaths = S.DeathMemory.load(self.deaths_path)
        mm = getattr(args, "move_memory", DEFAULT_MOVES)
        self.moves = None if mm is None or str(mm).lower() == "none" else MoveMemory.load(mm)
        self.f_plans = open(os.path.join(self.run_dir, "plans.jsonl"), "a", encoding="utf-8")
        self.f_events = open(os.path.join(self.run_dir, "events.jsonl"), "a", encoding="utf-8")
        self.f_states = open(os.path.join(self.run_dir, "states.jsonl"), "a", encoding="utf-8")
        self.f_times = open(os.path.join(self.run_dir, "times.jsonl"), "a", encoding="utf-8")
        self.f_search = open(os.path.join(self.run_dir, "search.jsonl"), "a", encoding="utf-8") \
            if self.search is not None else None
        self.best = BestTimes(args.best_times)
        self.find_save = locate_save            # S: where the game's save is (tests replace it)
        self.backup_root = BACKUPS
        self.save_backup = None                 # the folder the save was copied to (once per run), None not yet
        self.lock = None                        # the time-step lock: True / False once realgame.js said so
        self.dt_ms = 16.667                     # the frame time of the last frame message (ms)
        self.cost = FrameCost()
        self.lines = 0
        self.wm = None
        if args.wm_collect:
            from wm.buffer import Collector
            if self.demo is not None:
                self.wm = Collector(os.path.join(self.run_dir, "wm_data"),
                                    source="demo" if not self.demo.copies else "demo+demo-noisy")
                self.wm.meta_extra = {"agent_sources": {"demo": [0], "demo-noisy": list(range(1, self.n_total))},
                                      "demo": self.demo.demo["best"]}
            elif self.search is not None:
                self.wm = Collector(os.path.join(self.run_dir, "wm_data"), source="search")
                self.wm.meta_extra = {"agent_sources": {"search": list(range(self.n_total))}}
            else:
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
        self.field = None                       # the level's progress field (the search's T)
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
        prev = self.cost.go("logging")
        f.write(json.dumps(T._plain(obj), separators=(",", ":")) + "\n")
        self.lines += 1
        if self.lines >= FLUSH_LINES:
            self.flush()
        self.cost.go(prev)

    def flush(self):
        self.lines = 0
        for f in (self.f_plans, self.f_events, self.f_states, self.f_times, self.f_search):
            if f is None:
                continue
            try:
                f.flush()
            except (OSError, ValueError):
                pass

    def event(self, kind: str, frame=None, **kw):
        ev = {"kind": kind, "frame": frame, **kw}
        self._write(self.f_events, {"time": round(time.time(), 3), "level": self.level.name if self.level else None,
                                    "wave": self.wave_no, **ev})
        if self.tel is not None:
            prev = self.cost.go("telemetry")
            self.tel.send(T.event_message(self.level.name if self.level else None, self.wave_no, ev))
            self.cost.go(prev)

    # ---- level / waves ---------------------------------------------------------------------------------------
    def new_wave(self, n: int):
        self.wave_shard = self.wm.n_shards if self.wm is not None else None   # this wave's shard number
        self.alive = [True] * n
        self.held = [0] * n
        self.prev_xy = [None] * n
        self.last_state = [None] * n
        self.cur_plan = {}
        self.commit.reset()
        self.last_decision = {}
        self.ex_queue = [[] for _ in range(n)]
        self.ex_prev = [-1] * n
        self.ex_plan = {}                      # explore agent -> its current plan (Planner.explore)
        self.last_msg = None
        self.overlay_frame = ()                # the (frame, id) of agent 0's plan last drawn; () nothing yet
        self.st = {"finished": 0, "died": 0, "explained": 0, "unexplained": 0, "estimated": 0, "memory": 0,
                   "surprise_pos": 0, "surprise_death": 0, "decisions": 0, "ms": [], "cut1": 0, "cut2": 0,
                   "failed": 0, "errors": 0, "ex_finished": 0, "ex_died": 0, "triggers": {}, "gaps": [],
                   "agent_frames": 0, "check_s": 0.0, "switched": 0, "kept": 0, "clipped": 0, "pnr": 0,
                   "pnr_none": 0, "completed": 0, "all_agent_frames": 0}
        self.fin = {"planned": [], "explore": [], "demo": [], "search": []}     # this wave's finish times (s)
        self.new_best = False
        if self.demo is not None:                 # this wave's perturbed copies (same for repeated calls)
            self.demo.new_wave()
            self.demo_fin, self.demo_trail = None, []
            if self.wm is not None:
                self.wm.meta_extra["perturbations"] = self.demo.changes

    def on_info(self, info: dict, tiles: bytes):
        if info["W"] * info["H"] == 0 or len(tiles or b"") != info["W"] * info["H"] * 4:
            print(f"[level] ignoring invalid level data {info['W']}x{info['H']}", flush=True)
            return
        if info["name"] in (None, "unknown"):
            info["name"] = identify_level(info, tiles) or "unknown"
        if self.demo is not None and info["name"] != self.demo.demo["level"]:
            print(f"[demo] refused: the game has level {info['name']} loaded, the demo is for "
                  f"{self.demo.demo['level']} (enter {self.demo.demo['level']} and start again); stopping", flush=True)
            self.level = None
            self.stopping = True
            return
        self.level = lv = Level(info, tiles)
        self.cost.reset()
        from wm.buffer import eval_levels
        self.protected = info["name"] in eval_levels()
        if self.wm and not self.protected:
            self.wm.level(lv, info)
        self.fw.new_level()
        self.commit.tube.refresh()                 # the surprise threshold, fixed for this level
        self.hz_start, self.moving, self.sweep = None, set(), set()
        self.field_built, self.field_dirty, self.rebuilds = False, False, 0
        self.lt_changes = 0
        self.wave_no = 0
        kb = lv.kill_box
        best = self.best.best(lv.name, "planned")
        print(f"[level] {lv.name}: {lv.W}x{lv.H}, {len(lv.goals)} goal(s), {info.get('lethal', '?')} lethal pieces, "
              f"{len(lv.keys)} key(s), {info['agents']} agents (planned {self.n_plan}, explore {self.n_explore}); "
              + (f"agent best {float(best['seconds']):.2f} s; " if best and best.get("seconds") is not None
                 else "no agent best yet; ")
              + (f"kill box x {kb[0]:.0f}..{kb[1]:.0f}, y {kb[2]:.0f}..{kb[3]:.0f}" if kb else "no kill box")
              + (" - held-out evaluation level: no world-model data, no death memory" if self.protected else ""),
              flush=True)
        self.event("level", None, name=lv.name, W=lv.W, H=lv.H, kill_box=list(kb) if kb else None,
                   protected=self.protected, agents=info["agents"],
                   tube=self.commit.tube.summary() if self.commit.on else None,
                   model=self.model_path if self.model is not None else ("none (search)" if self.search else None))
        if self.search is not None:
            self.search.set_level(lv)
        self.new_wave(info["agents"])
        if "hz" in info:                       # the game waits for the 'go' (no agent moves meanwhile)
            boxes, kept = hazard_boxes(info["hz"], lv.hz_keep)
            self.track_hazards(boxes, box_ids(kept, lv.hz_ids))
            self.build_field("level start")
        self.send_level()
        self.post_best(lv.name)

    def post_best(self, level):
        """The planned agents' best for the level (best_times.json, lock-on times only) on the game's timer sign
        (realgame.js 'best': it only changes the sign's text, "BEST x.xx"); nothing without one (realgame.js
        resets the sign to "BEST --.--" at every level load)."""
        b = self.best.best(level, "planned") if level else None
        if self.script is None or not b or b.get("seconds") is None:
            return
        try:
            self.script.post({"type": "best", "payload": float(b["seconds"])})
        except Exception:
            pass

    def track_hazards(self, boxes: list, ids: list):
        """As realgame.Trainer.track_hazards: each level piece's start box and every place a moving one was seen
        (rounded to 4 units); a new place marks the field for a rebuild."""
        if self.hz_start is None:
            self.hz_start = {}
        level_ids = self.level.hz_level_ids
        for b, hid in zip(boxes, ids):
            if box_base(hid) not in level_ids:
                continue
            if hid not in self.hz_start:
                self.hz_start[hid] = b
                continue
            x, y, hw, hh, a = b
            x0, y0, hw0, hh0, a0 = self.hz_start[hid]
            if abs(x - x0) > 2 or abs(y - y0) > 2 or abs(a - a0) > 0.05 or abs(hw - hw0) > 1 or abs(hh - hh0) > 1:
                if hid not in self.moving:
                    self.moving.add(hid)
                    self.sweep.add((round(x0 / 4) * 4, round(y0 / 4) * 4, hw0, hh0, round(a0, 1)))
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
        self.field = field
        if self.model is not None:
            if self.planner is None:
                self.planner = Planner(self.model, lv, field, budget_ms=self.args.budget_ms,
                                       hysteresis=self.args.hysteresis, moves=self.moves,
                                       modes=self.args.modes == "on")
            elif self.planner_level is not lv:
                self.planner.set_level(lv, field)
            else:
                self.planner.set_field(field)
                self.planner.bounds = S.level_bounds(lv)
            if self.moves is not None:
                self.moves.learn = not self.protected      # held-out levels: no charges, no tries
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
        C = self.cost
        # the level's live state (as realgame.Trainer.on_frame)
        C.go("level")
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
        boxes, kept = hazard_boxes(m.get("hz", []), lv.hz_keep)
        ids = box_ids(kept, lv.hz_ids, prefix="")
        self.track_hazards(boxes, ids)
        if not self.field_built:
            self.build_field("first frame")
            pl = self.planner
        dt = float(m.get("dt") or 16.667)
        self.dt_ms = dt
        self.fw.observe(frame, m.get("hz", []), m.get("pf"), m.get("sh", []), lv.hz_keep, lv.fans, dt)
        if self.wm and not self.protected:
            C.go("wm_data")
            self.wm.frame(m, list(lv.hz_ids), list(lv.hz_keep))

        # what happened since the last frame: deaths, finishes, surprises, failed moves
        C.go("observe")
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
                    self.observe(i, frame, x, y, keys, dead=False, vx=float(s[2]), vy=float(s[3]))
            if st != 0:
                self.alive[i] = False
                planned = i < self.n_plan
                if st == 2:
                    # realgame.js counts frame++ after each update: status 2 first shown in the message of
                    # frame f means the goal was touched in update f - 1
                    self.on_finish(i, frame - 1, x, y)
                elif planned and pl is None:
                    self.st["died"] += 1
                elif self.search is not None:
                    self.search_death(i, frame, x, y)
                elif not planned:
                    self.st["ex_died"] += 1
            self.prev_xy[i] = (x, y)
            self.last_state[i] = s

        # decisions
        C.go("planning")
        acts, act_idx = [NOOP] * n, [-1] * n
        fboxes = None
        live_plan = [i for i in range(min(self.n_plan, n)) if self.alive[i]] if pl is not None else []
        need = {}
        fut = {}

        def fboxes_now():                           # this frame's model inputs and deadly boxes, once
            if "fb" not in fut:
                fut["w"], fut["fb"], _ = future_inputs(self.fw, frame, pl.H, dt, lv.name, self.deaths)
            return fut["fb"]
        if live_plan:
            t_chk = time.perf_counter()
            tiles = bool(m.get("tc"))
            for i in live_plan:
                s = states[i]
                tr = self.commit.triggers(i, frame, float(s[0]), float(s[1]), int(s[7]) if len(s) > 7 else 0, tiles,
                                          pl, fboxes_now)
                if tr:
                    need[i] = tr
            self.st["agent_frames"] += len(live_plan)
            self.st["check_s"] += time.perf_counter() - t_chk
        fboxes = fut.get("fb")                      # computed for a check or a decision: telemetry reuses it
        if need:                                    # decisions: inputs to H + the airborne completion's frames
            fut["w"], fut["fb"], self.fut_rec = future_inputs(self.fw, frame, pl.future_frames, dt, lv.name,
                                                              self.deaths)
            worlds, fboxes = fut["w"], fut["fb"]
            for i, trig in need.items():
                s = states[i]
                row = state_row(s[0], s[1], s[2], s[3], s[4], s[5], s[7] if len(s) > 7 else 0, held_before[i])
                try:
                    plan = pl.plan(row, frame, worlds, fboxes, dt, agent=i,
                                   danger=any(kind_of(t) == "danger" for t in trig), recorded=self.fut_rec)
                except Exception as e:                  # never leave the agent without an answer
                    self.st["errors"] += 1
                    if self.st["errors"] <= 3:
                        import traceback
                        traceback.print_exc()
                    self.event("plan error", frame, agent=i, error=f"{type(e).__name__}: {e}")
                    continue
                self.cur_plan[i] = plan
                self.commit.decided(i, plan, frame, fboxes, pl)
                self.on_plan(plan, s, trig)
        for i in live_plan:
            p = self.cur_plan.get(i)
            k = frame - p.frame if p is not None else -1
            a = int(p.actions[k]) if p is not None and 0 <= k < len(p.actions) else 0
            acts[i], act_idx[i] = list(ACTIONS[a]), a
            pl.played(i, frame, a)
        C.go("explore")
        self.st["all_agent_frames"] += sum(1 for i in range(n) if self.alive[i] and int(states[i][6]) == 0)
        if self.search is not None:
            if not self.search.active:
                self.search.begin_wave(self.wave_no, frame, states, self.field)
            self.search.observe(frame, states, m.get("tc"))
            for i in range(n):
                if self.alive[i] and int(states[i][6]) == 0:
                    a = self.search.action(i, frame)
                    acts[i], act_idx[i] = list(ACTIONS[a]), a
        if self.demo is not None:
            if self.alive and self.alive[0] and int(states[0][6]) == 0:
                self.demo_trail.append((float(states[0][0]), float(states[0][1])))
            for i in range(n):
                if self.alive[i] and int(states[i][6]) == 0:
                    a = self.demo.action(i, frame)
                    acts[i], act_idx[i] = list(ACTIONS[a]), a
        for i in range(self.n_plan, n if self.demo is None and self.search is None else 0):
            if self.alive[i] and int(states[i][6]) == 0:
                a = self.explore_plan(i, frame, states[i], held_before[i], dt, fut) if pl is not None and \
                    pl.modes else self.explore_action(i)
                acts[i], act_idx[i] = list(ACTIONS[a]), a
        C.go("other")
        for i in range(n):
            if act_idx[i] >= 0:
                self.held[i] = int(next_held(np.array([self.held[i]]), np.array([act_idx[i]]),
                                             np.array([int(states[i][4]) & 0xffffffff], np.int64))[0])
            else:
                self.held[i] = 0
        self.last_msg = m
        C.go("overlay")
        self.post_overlay(frame)
        C.go("other")
        won = [i for i, s in enumerate(states) if int(s[6]) == 2]
        live = [i for i, s in enumerate(states) if int(s[6]) == 0 and self.alive[i]]
        leader = won[0] if won else (live_plan[0] if live_plan else (live[0] if live else 0))
        self._write(self.f_states, {"frame": frame, "wave": self.wave_no, "level": lv.name,
                                    "agents": [[i, round(float(states[i][0]), 2), round(float(states[i][1]), 2),
                                                round(float(states[i][2]), 2), round(float(states[i][3]), 2),
                                                int(states[i][6])]
                                               for i in range(n if self.demo is not None or self.search is not None
                                                              else min(self.n_plan, n))]})
        if self.tel is not None:
            C.go("telemetry")
            self.send_frame(m, frame, states, act_idx, boxes, kept, fboxes, dt)
        if self.wm and not self.protected:
            C.go("wm_data")
            self.wm.agents(self.wave_no, frame, states, alive_before, held_before, act_idx, [float("nan")] * n)
        C.go("other")
        if self.demo is not None and frame >= self.demo.end_frame:
            # the demo's actions ran out a second ago: copies still standing count as done (they would keep the
            # wave going to --wave-seconds); realgame.js ends the wave on "end" before stepping anyone
            return {"a": acts, "leader": int(leader), "end": "demo done"}
        if self.search is not None and self.search.active and frame >= self.search.end_frame:
            # every agent's sequence ran out SETTLE frames ago: the wave is done (realgame.js ends it on "end")
            return {"a": acts, "leader": int(leader), "end": "search done"}
        return {"a": acts, "leader": int(leader)}

    def on_plan(self, plan, s, trigger=()):
        st = self.st
        st["decisions"] += 1
        for t in trigger:
            kd = kind_of(t)
            st["triggers"][kd] = st["triggers"].get(kd, 0) + 1
        sw = getattr(plan, "switch", None)
        if sw is not None:
            st["kept" if sw["kept"] else "switched"] += 1
        last = self.last_decision.get(plan.agent)
        if last is not None and plan.frame > last:
            st["gaps"].append(plan.frame - last)
        self.last_decision[plan.agent] = plan.frame
        st["ms"].append(plan.timing.get("total", 0.0))
        st["clipped"] += int((getattr(plan, "model_clipped", None) or 0) > 0)
        st["completed"] += int(getattr(plan, "completion", None) is not None)
        c = plan.counts
        if c.get("stage1", {}).get("complete", 0) < c.get("stage1", {}).get("candidates", 0):   # moves cut short
            st["cut1"] += 1
        s2 = c.get("stage2", {})
        if "skipped" in s2 or s2.get("steps", 0) < s2.get("of_steps", 0):
            st["cut2"] += 1
        prev = self.cost.go("logging")
        d = plan.to_dict()
        # future_rec: frames of the hazards' future taken from the recording (0: all extrapolated)
        d.update({"level": self.level.name, "wave": self.wave_no, "trigger": list(trigger),
                  "future_rec": int(getattr(self, "fut_rec", 0)),
                  "real": {"x": s[0], "y": s[1], "vx": s[2], "vy": s[3], "flags": int(s[4]), "state": int(s[5]),
                           "status": int(s[6]), "keys": int(s[7]) if len(s) > 7 else 0}})
        self._write(self.f_plans, d)
        if self.tel is not None:
            self.cost.go("telemetry")
            self.tel.send(T.plan_message(self.level.name, self.wave_no, d))
        self.cost.go(prev)

    def observe(self, i, frame, x, y, keys, dead, cause=None, vx=None, vy=None):
        """Planner.observe: surprises, failed moves and (deaths) the point of no return, counted and logged."""
        for ev in self.planner.observe(i, frame, x, y, keys, dead, cause=cause, vx=vx, vy=vy, dt_ms=self.dt_ms):
            k = ev.get("kind")
            if k == "position":
                self.st["surprise_pos"] += 1
            elif k == "unpredicted death":
                self.st["surprise_death"] += 1
            elif k == "failed move":
                self.st["failed"] += 1
            elif k == "point of no return":
                self.st["pnr" if ev.get("charged") else "pnr_none"] += 1
            self.event(k, ev.get("frame", frame), **{a: b for a, b in ev.items() if a not in ("kind", "frame")})

    def on_death(self, i, frame, xy0, xy1, keys, rec, extra_boxes, estimated: bool):
        """A planned agent died at wave frame `frame` at xy1 (from xy0 the frame before): does the geometry
        explain it (the move against the death frame's hazards and shots, or leaving the level)? If not, the
        place goes into the death memory (not on held-out levels) and is saved."""
        lv = self.level
        x1, y1 = xy1
        x0, y0 = xy0 if xy0 is not None else xy1
        explained = bool(S.explain_death(lv, rec or {}, lv.hz_keep, x0, y0, x1, y1))
        fb = S.record_boxes(rec or {}, lv.hz_keep)
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
            self.observe(i, frame, x1, y1, keys, dead=True, cause=cause if explained else "unexplained")
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
        search = self.search is not None
        source = "planned" if planned else ("demo" if self.demo is not None else ("search" if search else "explore"))
        self.st["finished" if planned or search else "ex_finished"] += 1
        if search:
            self.search.finished(i, int(finish_frame) + 1, x, y)
        if self.demo is not None and i == 0:
            self.demo_fin = int(finish_frame)
        lv = self.level.name if self.level else None
        frames = int(finish_frame) + 1
        seconds = round(frames * self.dt_ms / 1000.0, 3)
        rec = {"level": lv, "wave": self.wave_no, "agent": int(i), "source": source, "frame": int(finish_frame),
               "frames": frames, "seconds": seconds, "dt_ms": round(self.dt_ms, 3), "lock": self.lock,
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._write(self.f_times, rec)
        self.fin[source].append(seconds)
        if planned and self.planner is not None:
            r = self.planner.finished(i, int(finish_frame))
            if r["best_route"]:
                self.event("best route", int(finish_frame), agent=int(i), frames=frames, seconds=seconds)
            if r["failures_cleared"]:
                self.event("forgiven", int(finish_frame), agent=int(i), failures=r["failures_cleared"])
        new = False
        if self.lock is True and lv is not None and self.demo is None:
            new = self.best.offer(lv, source, {"seconds": seconds, "frames": frames, "run": self.args.name,
                                               "wave": self.wave_no, "agent": int(i), "date": rec["time"]})
        if planned:
            self.new_best |= new
            if new:
                self.post_best(lv)
                if self.learn and self.wm is not None and not self.protected:
                    self.add_self_demo(i, int(finish_frame), lv, seconds)
            best = self.best.best(lv, "planned") if lv is not None else None
            self.event("finish", int(finish_frame), agent=i, x=x, y=y, frames=frames, seconds=seconds,
                       dt_ms=rec["dt_ms"], lock=self.lock, best=float(best["seconds"]) if best else None,
                       new_best=new)
        elif search:
            best = self.best.best(lv, "search") if lv is not None else None
            self.event("finish", int(finish_frame), agent=i, x=x, y=y, frames=frames, seconds=seconds,
                       dt_ms=rec["dt_ms"], lock=self.lock, best=float(best["seconds"]) if best else None,
                       new_best=new, source="search")

    def search_death(self, i: int, frame: int, x, y):
        """A search agent died (seen at wave frame `frame`): counted and logged (place and frame: the benchmark's
        distinct outcomes); no death memory, no geometry check (nothing is learned from it)."""
        self.st["died"] += 1
        self.event("death", frame, agent=i, x=x, y=y, explained=None, cause="search", estimated=False,
                   memory_added=False)

    def explore_plan(self, i: int, frame: int, s, held: int, dt: float, fut: dict) -> int:
        """EXPLORE (planning modes on): the explore agent's action from its plan (Planner.explore: members 3 and 4,
        the safe plan they disagree on most), decided every EXPLORE_EVERY frames; on a planner error the old
        sticky random action."""
        pl, lv = self.planner, self.level
        p = self.ex_plan.get(i)
        if p is None or not 0 <= frame - p.frame < min(EXPLORE_EVERY, len(p.actions)):
            if "w" not in fut:
                fut["w"], fut["fb"], _ = future_inputs(self.fw, frame, pl.H, dt, lv.name, self.deaths)
            row = state_row(s[0], s[1], s[2], s[3], s[4], s[5], s[7] if len(s) > 7 else 0, held)
            try:
                p = self.ex_plan[i] = pl.explore(row, frame, fut["w"], fut["fb"], dt, agent=i)
            except Exception as e:
                self.st["errors"] += 1
                self.event("explore error", frame, agent=i, error=f"{type(e).__name__}: {e}")
                self.ex_plan.pop(i, None)
                return self.explore_action(i)
            self.st["explore_ms"] = self.st.get("explore_ms", 0.0) + p.timing["total"]
            self.st["explore_decisions"] = self.st.get("explore_decisions", 0) + 1
        return int(p.actions[frame - p.frame])

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
        if self.demo is not None:
            if self.args.no_overlay or self.script is None:
                return
            pts = self.demo.demo.get("path") or self.demo_trail
            key = ("rec",) if self.demo.demo.get("path") else ("trail", len(pts) // DEMO_TRAIL_EVERY)
            if (not force and key == self.overlay_frame) or len(pts) < 2:
                return
            self.overlay_frame = key
            try:
                self.script.post({"type": "path", "payload": [{"pts": _sub(pts, PLAN_STEP), "col": COL_DEMO, "w": 3}]})
            except Exception:
                pass
            return
        if self.args.no_overlay or self.script is None or not self.n_plan:
            return
        p = self.cur_plan.get(0) if self.alive and self.alive[0] else None
        key = None if p is None else (p.frame, id(p))
        if not force and key == self.overlay_frame:
            return                                  # agent 0's plan did not change since the last post
        self.overlay_frame = key
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
            elif self.search is not None:
                last = final[i] if final is not None and i < len(final) else self.last_state[i]
                self.search_death(i, frame, *((float(last[0]), float(last[1])) if last is not None else (None, None)))
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
        srec = None
        if self.search is not None:
            if final is not None:
                self.search.observe(frame, final)
            srec = self.search.wave_end(frame)
            self._write(self.f_search, {"time": round(time.time(), 3), "level": lv.name if lv else None,
                                        "wave": self.wave_no, **srec})
        ms = np.array(st["ms"]) if st["ms"] else None
        el = (time.time() - self.t0) / 60
        err_f = self.planner.surprise.errors() if self.planner is not None else None
        cost = self.cost.summary(frame, self.dt_ms)
        com = commit_summary(st, frame)
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
                + f" | {commit_text(com, st['triggers'])}"
                + f" | cut short: stage 1 {st['cut1']}, stage 2 {st['cut2']} | failed moves {st['failed']}"
                + (f" | points of no return charged {st['pnr']}, none found {st['pnr_none']}"
                   if st["pnr"] + st["pnr_none"] else "")
                + f" | chosen plans completed {st['completed']}"
                + (f" | explore {st['ex_finished']}/{self.n_explore} finished"
                   + (f" (fastest {fin_x:.2f} s)" if fin_x is not None else "") + f", {st['ex_died']} died"
                   if self.n_explore else "")
                + (f" | {st['errors']} plan errors" if st["errors"] else "")
                + (f" | death memory +{st['memory']} ({len(self.deaths)})" if st["memory"] else "")
                + f"  [{el:.1f} min]")
        if srec is not None:
            from plan.search import text as search_text
            fin_s = min(self.fin["search"]) if self.fin["search"] else None
            line = (f"[wave {self.wave_no}] {lv.name if lv else '?'} ({m.get('why')}, {frame} frames): search "
                    f"{st['finished']}/{self.n_total} finished" + (f" (fastest {fin_s:.2f} s)" if fin_s is not None else "")
                    + f", {st['died']} died, {st['all_agent_frames']} agent frames  [{el:.1f} min]\n"
                    + search_text(lv.name if lv else "?", self.wave_no, srec, self.dt_ms))
        print(line, flush=True)
        if self.demo is not None and lv is not None:
            ok, why = self.demo.check(self.demo_fin)
            fins = self.fin["demo"]
            print(f"[demo] wave {self.wave_no}: {why}" + ("" if ok else " - WARNING: the demo is not reproducible "
                  "here (recorded without the time-step lock, or the level does not replay the same way)")
                  + f"; {len(fins)}/{self.n_total} agents finished" + (f" (fastest {min(fins):.2f} s)" if fins else ""),
                  flush=True)
            self.event("demo check", frame, ok=ok, why=why, finish_frame=self.demo_fin,
                       demo_finish_frame=self.demo.finish_frame, finished=len(fins), demo=self.demo.demo["best"],
                       perturbations=self.demo.changes, demo_wave=self.demo.waves)
            self.demo.wave_done()
        if cost is not None:
            print(self.cost.text(cost), flush=True)
        fut = self.fw.new_wave()                     # the hazard recording: what this wave used and added
        self.event("wave", frame, **(cost or {}), why=m.get("why"), frames=frame, future=fut,
                   **{k: v for k, v in st.items() if k not in ("ms", "gaps", "check_s")},
                   ms_median=float(np.median(ms)) if ms is not None else None,
                   ms_95=float(np.percentile(ms, 95)) if ms is not None else None, err_followed=err_f, **com,
                   tube=self.commit.tube.summary() if self.commit.on else None,
                   finish_s=fin_p, explore_finish_s=fin_x, best_s=float(best_p["seconds"]) if best_p else None,
                   new_best=self.new_best, lock=self.lock,
                   **({"replay_route": bool(srec.get("replay")), "search_finish_s": min(self.fin["search"])
                       if self.fin["search"] else None} if srec is not None else {}))
        if self.moves is not None and self.moves.dirty:
            try:
                self.moves.save()
            except OSError as e:
                print(f"[moves] could not save {self.moves.path}: {e}", flush=True)
        if self.planner is not None:
            if self.commit.on:
                self.commit.tube.add(self.planner.surprise.err)     # the run's followed errors, for the tube
            self.planner.new_wave()                 # also clears the surprise check's predictions and errors
        if lv is not None and self.field_dirty and self.rebuilds < FIELD_REBUILDS:
            self.rebuilds += 1
            self.build_field(f"rebuild {self.rebuilds}/{FIELD_REBUILDS}: moving hazards / kill box / platforms")
        if self.learn:
            self.load_live()                        # between waves only: a newer published model
        self.new_wave(len(status) if status else len(self.alive))
        self.post_overlay(0, force=True)
        self.flush()
        self.cost.reset()

    # ---- frida -------------------------------------------------------------------------------------------------
    def on_message(self, msg, data):
        if msg["type"] == "error":
            print("[script error]", msg.get("description"), msg.get("stack", ""), flush=True)
            return
        p = msg["payload"]
        ev = p.get("ev")
        if ev == "frame":
            self.cost.begin()
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
                if self.wm is not None:
                    self.wm.cut("level change")     # the wave in progress is kept as a partial shard
                self.level = None
                self.cur_plan = {}
                self.commit.reset()
                if self.search is not None:
                    self.search.abort_wave()
            elif ev == "save":
                self.on_save(p.get("level"))
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
            self.cost.go("other")
            self.script.post({"type": "act", "payload": reply})
            self.cost.end()

    def on_save(self, level):
        """S in the game: realgame.js walks the main character into the goal (the game's own level end) and
        writes the level time it records (hud+0x384) on every frame of that finish. The time sent is the planned
        agents' best for the level (best_times.json, lock on only); the game keeps it as the level's record only
        if it is better than the stored one (FUN_004ba960). Before the first such write of the run the save file
        is copied to backups/save_<date-time>/; without a best, or when the save cannot be found or copied, the
        time is 999.99 (realgame.js's default: slower than any real run)."""
        self.best.reload()
        b = self.best.best(level, "planned") if level else None
        secs = float(b["seconds"]) if b and b.get("seconds") is not None else None
        why = None
        if secs is not None and self.save_backup is None:
            save, why = self.find_save()
            if save is not None:
                try:
                    self.save_backup = backup_save(save, self.backup_root)
                except OSError as e:
                    why = f"could not back up {save}: {e}"
        if secs is not None and self.save_backup is not None:
            print(f"[S] finishing {level} the game's way with the agent's best {secs:.2f} s (save backed up to "
                  f"{self.save_backup})", flush=True)
        else:
            print(f"[S] finishing {level} the game's way, " + (f"no agent best for {level}" if secs is None else
                  f"the agent's best {secs:.2f} s is not written ({why})") + ": 999.99", flush=True)
            secs = None
        self.event("save", None, level=level, seconds=secs, backup=self.save_backup)
        if self.wm is not None:
            self.wm.cut("S")                        # the wave in progress is kept as a partial shard
        self.script.post({"type": "finish", "payload": {"time": secs}})

    # ---- train mode (--learn) ---------------------------------------------------------------------------------
    def add_self_demo(self, i: int, finish_frame: int, lv: str, seconds: float):
        """A planned agent's new best (lock on, not held-out): its rows of this wave become a demo for wm.learner
        (runs/demos/self.json, recorded.json's form; the shard is written at the wave end or when the wave is cut)."""
        from wm.learner import add_self_demo, _rel
        e = {"wm_data": _rel(self.wm.dir), "shard": f"wave_{self.wave_shard:05d}.npz", "agent": int(i),
             "frames": [0, finish_frame + 1], "level": lv, "seconds": seconds, "run": self.args.name,
             "wave": self.wave_no, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            n = add_self_demo(e, self.args.self_demos)
        except OSError as err:
            print(f"[learn] could not record the self-demo: {err}", flush=True)
            return
        print(f"[learn] self-demo {n}: {lv} {seconds:.2f} s (agent {i}, {e['shard']})", flush=True)
        self.event("self demo", finish_frame, **e)

    def load_live(self):
        """Between waves: the learner's published model when it is newer than the one in use."""
        p = os.path.join(self.args.live_dir, "wm_model.pt")
        try:
            if not os.path.exists(p) or os.path.getmtime(p) <= self.live_mtime:
                return
            mt = os.path.getmtime(p)
            from wm.model import WorldModel
            m = WorldModel.load(p).eval()
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            print(f"[learn] could not load {p}: {e}", flush=True)
            return
        self.live_mtime = mt
        if m.step != self.model.step or len(m.members) != len(self.model.members):
            print(f"[learn] {p}: step {m.step} / {len(m.members)} members, the planner has {self.model.step} / "
                  f"{len(self.model.members)}: not loaded", flush=True)
            return
        info = (m.info or {}).get("learner", {})
        old = (info.get("parent_errors") or {}).get("16", [None])[0]
        new = (info.get("errors") or {}).get("16", [None])[0]
        self.model, self.model_path = m, p
        if self.planner is not None:
            self.planner.model = m
        print(f"[learn] model v{info.get('version', '?')} loaded (+16 error {old} -> {new})", flush=True)
        self.event("learn model", None, version=info.get("version"), err16=[old, new], path=p)

    def start_learner(self):
        """wm.learner as a child (below-normal priority), from the default model unless --live-dir has a newer one."""
        import subprocess
        live = os.path.join(self.args.live_dir, "wm_model.pt")
        newer = os.path.exists(live) and os.path.getmtime(live) > os.path.getmtime(self.model_path)
        init = live if newer else self.model_path
        os.makedirs(self.args.live_dir, exist_ok=True)
        out = open(os.path.join(self.args.live_dir, "learner_output.txt"), "a", encoding="utf-8")
        flags = 0x4000 if sys.platform == "win32" else 0          # BELOW_NORMAL_PRIORITY_CLASS
        self.learner = subprocess.Popen([sys.executable, "-m", "wm.learner", "--init", init, "--out",
                                         self.args.live_dir, "--self-demos", self.args.self_demos],
                                        cwd=HERE, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
        out.close()
        print(f"[learn] learner started (pid {self.learner.pid}, from {init}; log "
              f"{os.path.join(self.args.live_dir, 'learner.log')})", flush=True)

    def stop_learner(self):
        if self.learner is None:
            return
        try:
            self.learner.terminate()
            self.learner.wait(5)
        except Exception:
            try:
                self.learner.kill()
            except Exception:
                pass
        print("[learn] learner stopped", flush=True)
        self.learner = None

    def close(self):
        self.stop_learner()
        if self.wm is not None:
            try:
                self.wm.cut("stop")                 # Ctrl+C / STOP: the wave in progress is kept (partial)
            except OSError as e:
                print(f"[wm] could not write the cut wave: {e}", flush=True)
        self.flush()
        for f in (self.f_plans, self.f_events, self.f_states, self.f_times, self.f_search):
            if f is None:
                continue
            try:
                f.close()
            except OSError:
                pass
        if self.tel is not None:
            print(f"telemetry: {self.tel.sent} sent, {self.tel.dropped} dropped, {self.tel.too_big} too big", flush=True)
            self.tel.close()

    def run(self):
        import frida
        other = other_runners()
        if other:
            raise SystemExit(f"another realgame.py / agent.py is already attached to the game (pid "
                             f"{', '.join(other)}): stop it first (Ctrl+C in its terminal, or a file runs/<name>/STOP)")
        if self.pool > GAME_POOL:
            print(f"{self.pool} agents: {self.pool - GAME_POOL} extra Meat Boys are built when a level loads (the level "
                  f"on screen now, if any, has the game's {GAME_POOL}: enter or restart a level for all)", flush=True)
        if self.learn:
            self.start_learner()
        cfg = {"agents": self.n_total, "pool": self.pool, "wave_frames": int(self.args.wave_seconds * HZ),
               "finish_frames": 60, "debug": False, "finish_quota": max(0, self.args.end_after), "probe_frames": 0,
               "lock_step": True}
        with open(JS, encoding="utf-8") as f:
            src = f.read().replace("__CONFIG__", json.dumps(cfg))
        # first Ctrl+C: the clean stop below; second Ctrl+C or no clean stop after 10 s: unload, leave at once
        sd = Shutdown(force=lambda: self.script is not None and self.script.unload()).install()
        try:
            session = frida.attach(GAME_EXE)
        except frida.ProcessNotFoundError:
            raise SystemExit(f"{GAME_EXE} is not running: start the game and enter a level")
        self.script = session.create_script(src)
        self.script.on("message", self.on_message)
        self.script.load()
        print(f"attached: {self.n_plan} planned + {self.n_explore} explore agents"
              + (f" + {self.n_total} search agents (plan/search.py)" if self.search is not None else "")
              + f", waves of {self.args.wave_seconds:g}s, "
              f"replan every {self.replan} frames" + (f" or on a trigger (tube x{self.args.tube:g})" if self.commit.on
                                                       else " (triggers off)") + f", budget {self.args.budget_ms:g} ms"
              + (f", model {self.model_path}" if self.model is not None else "")
              + f", death memory {self.deaths_path} ({len(self.deaths)} entries); logs in {self.run_dir}; "
                f"Ctrl+C to stop", flush=True)
        stop_file = os.path.join(self.run_dir, "STOP")   # create it to stop a background run safely
        if os.path.exists(stop_file):
            os.remove(stop_file)
        code = 0
        try:
            while not os.path.exists(stop_file) and not sd.requested.is_set() and not self.stopping:
                time.sleep(0.5)
            if os.path.exists(stop_file):
                os.remove(stop_file)
        except BaseException:
            import traceback
            traceback.print_exc()
            code = 1
        sd.request()                                  # the watchdog covers the clean stop from here
        self.stopping = True
        clean_stop(self.script, session, self.stopped, self.close)
        print(f"logs in {self.run_dir}; death memory {self.deaths_path} ({len(self.deaths)} entries)", flush=True)
        sd.finish(code)


def clean_stop(script, session, stopped: threading.Event, close, wait_s: float = 3.0):
    """Stop through the next frame reply (the game thread is never left waiting for an answer), then unload the
    script (its hooks go and a waiting game thread is released; the time-step lock bytes are put back by the
    shutdown export, not by unload), detach and close the logs. Every Frida call is bounded: at most ~8 s."""
    if not stopped.wait(wait_s):
        # no frames are coming (menu / level change): nothing is waiting, the RPC is safe
        bounded(lambda: script.exports_sync.shutdown(), 1.5)
    time.sleep(0.2)
    bounded(script.unload, 2.0)
    bounded(session.detach, 1.5)
    close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=1, help="planned agents (default 1)")
    ap.add_argument("--explore", type=int, default=0,
                    help="extra agents with sticky random actions and random library move chains (world-model data)")
    ap.add_argument("--wave-seconds", type=float, default=20.0, help="wave time limit (default 20)")
    ap.add_argument("--end-after", type=int, default=1,
                    help="end the wave once this many agents finished (default 1; 0 = wait for all)")
    ap.add_argument("--replan", type=int, default=REPLAN_DEFAULT,
                    help=f"frames between a planned agent's decisions at most (default {REPLAN_DEFAULT}); with "
                         "--triggers off exactly (--replan 2 --triggers off: the old behaviour)")
    ap.add_argument("--tube", type=float, default=TUBE_DEFAULT,
                    help=f"replan when the real position is off the committed plan's prediction by more than this x "
                         f"the 90%% error of followed predictions at that horizon (default {TUBE_DEFAULT:g})")
    ap.add_argument("--hysteresis", type=float, default=HYSTERESIS_FRAMES,
                    help="keep the committed plan unless another's estimated frames to the goal are lower by more "
                         "than this (frames; successes: 3 frames earlier at the goal); 0 = always the best score "
                         f"(default {HYSTERESIS_FRAMES:g})")
    ap.add_argument("--triggers", choices=("on", "off"), default="on",
                    help="replan on triggers (start, interval, move end, surprise, support, danger, short, missed: "
                         "plan/commit.py) or only every --replan frames (off) (default on)")
    ap.add_argument("--budget-ms", type=float, default=120.0,
                    help="target planner time per decision, reported only: no stage is cut by time, the search is "
                         "deterministic (default 120)")
    ap.add_argument("--modes", choices=("on", "off"), default="on",
                    help="planning modes (plan/planner.py): long-safe (member 0, blue), normal (member 1, green), "
                         "reflex (member 2, red); explore agents plan on members 3 and 4 (purple) instead of random "
                         "actions (default on)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="world model (default runs/wm_step4/wm_model.pt, 4 frames per step; the 1-frame "
                         "runs/wm_locked/wm_model.pt works too)")
    ap.add_argument("--name", default=None, help="run folder under runs/ (default plan_<time>)")
    ap.add_argument("--wm-collect", action="store_true", help="save world-model data to runs/<name>/wm_data")
    ap.add_argument("--learn", action="store_true",
                    help="train mode: wm.learner in the background (published models loaded between waves), new "
                         "best times recorded as self-demos; turns on --wm-collect")
    ap.add_argument("--live-dir", default=LIVE_DIR, help="--learn: the learner's folder (default runs/wm_live)")
    ap.add_argument("--self-demos", default=SELF_DEMOS, help="--learn: self-demo list (default runs/demos/self.json)")
    ap.add_argument("--no-overlay", action="store_true", help="do not draw the plan in the game")
    ap.add_argument("--telemetry", default="127.0.0.1:47800",
                    help="host:port for the live telemetry (viz/telemetry.py), or off (default 127.0.0.1:47800)")
    ap.add_argument("--death-memory", default=DEFAULT_DEATHS,
                    help="unexplained deaths remembered as hazards (loaded at start, saved on every new one; "
                         "default smbrl/death_memory.json)")
    ap.add_argument("--move-memory", default=DEFAULT_MOVES,
                    help="deaths charged to the move taken at the point of no return, per level (plan/planner.py "
                         "MoveMemory; loaded at start, a malformed file stops the start; saved at every wave end; "
                         "default smbrl/move_memory.json; none: no point-of-no-return charging)")
    ap.add_argument("--best-times", default=DEFAULT_BEST,
                    help="best finish times per level (default smbrl/best_times.json; time-step lock on only)")
    ap.add_argument("--threads", type=int, default=2, help="torch threads for the world model (default 2)")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the explore agents")
    ap.add_argument("--demo", default=None, metavar="BEST_JSON",
                    help="replay this PPO best finish (runs/real_*/levels/<level>/best.json; plan.demos inventory)")
    ap.add_argument("--demo-level", default=None, help="replay the level's fastest non-suspicious PPO best finish")
    ap.add_argument("--demo-copies", type=int, default=20, help="--demo: perturbed copies besides the exact one")
    ap.add_argument("--search", action="store_true",
                    help="wave search (plan/search.py): no planner, no model; every agent replays a known action "
                         "prefix exactly, then branches; not with --agents / --explore / --demo / --learn")
    ap.add_argument("--search-agents", type=int, default=SEARCH_AGENTS,
                    help=f"--search: agents per wave, agent 0 the verifier (default {SEARCH_AGENTS})")
    ap.add_argument("--no-lock-step", action="store_true", help=argparse.SUPPRESS)
    return ap


def setup_demo(args, entries=None):
    """--demo / --demo-level: resolve (refusals: SystemExit), then the agents, wave length and run name."""
    from plan.demos import resolve_demo, DemoPlay
    if args.demo and args.demo_level:
        raise SystemExit("--demo and --demo-level: give one")
    demo = resolve_demo(best=args.demo, level=args.demo_level, entries=entries)
    if not 0 <= args.demo_copies <= MAX_AGENTS - 1:
        raise SystemExit(f"--demo-copies {args.demo_copies}: 0 to {MAX_AGENTS - 1}")
    args.demo_play = DemoPlay(demo, args.demo_copies, args.seed)
    args.end_after = 0                              # every copy plays the wave out
    need = demo["seconds"] + 2.0
    if args.wave_seconds < need:
        print(f"[demo] --wave-seconds {args.wave_seconds:g} -> {need:.1f} (the demo takes {demo['seconds']:.2f} s)")
        args.wave_seconds = need
    args.name = args.name or time.strftime(f"demo_{demo['level']}_%Y%m%d-%H%M%S")
    print(f"[demo] {demo['best']}: {demo['level']} in {demo['seconds']:.2f} s ({len(demo['actions'])} frames, "
          f"{demo['entry']['verdict']}), {args.demo_copies} perturbed copies (seed {args.seed})", flush=True)
    return args


def main():
    args = build_parser().parse_args()
    if args.no_lock_step:
        raise SystemExit("the planner needs the game's time step locked (it assumes every future step is the game's "
                         "fixed step): run without --no-lock-step")
    if args.demo or args.demo_level:
        if args.learn:
            raise SystemExit("--learn needs planned agents: not with --demo / --demo-level")
        setup_demo(args)
    if args.search:
        if args.learn:
            raise SystemExit("--learn needs planned agents: not with --search")
        if args.demo or args.demo_level:
            raise SystemExit("--search and --demo / --demo-level: give one")
        args.name = args.name or time.strftime("search_%Y%m%d-%H%M%S")
    if args.learn:
        args.wm_collect = True                      # the learner trains on this run's shards too
    args.name = args.name or time.strftime("plan_%Y%m%d-%H%M%S")
    Runner(args).run()


if __name__ == "__main__":
    main()
