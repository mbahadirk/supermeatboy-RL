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
planner avoids that place at that wave frame from then on (at every frame where deaths repeat at other frames,
until a real agent is seen alive there at another time: every alive position is fed to it). Every death of a planned agent (explained or not)
charges the move taken at its point of no return (the latest decision of the wave with a safe alternative;
plan.planner.MoveMemory, per level in --move-memory, default smbrl/move_memory.json, saved at every wave end;
none: off) and logs a "point of no return" event. Held-out levels (wm.buffer.eval_levels): the first life
charges nothing and adds no death; with --online, after this run has played one full wave of the level (its
first life is over) the level practises: moves are charged and unexplained deaths remembered as on training
levels, for this run only (both memories keep held-out levels transient: never written to their files, also not
by the planner process), so a later run's first life stays clean. Without --online, and in a first-life
evaluation, nothing is learned on them.

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
    python smbrl/agent.py --learn --legacy-split

The command above explicitly selects historical training compatibility. --learn without a split choice is
rejected before attaching. --dataset-manifest FILE selects a frozen trace-split-v1 learner instead; its model
must already belong to that exact snapshot. A strict learner does not absorb new gameplay automatically.
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
Telemetry (--telemetry host:port, default 127.0.0.1:47800, "off" to disable): viz/telemetry.py (schema there). The
frame message goes out every --telemetry-every frames (default 3: 20 Hz) and on every frame that matters (a decision,
a death, a finish, a wave's first and last frame, a new level); the planned agents' positions of the frames between
ride along (its "trail"). The run's files are written every frame as before.
The wave summary line also says how often the planned agents decided: decisions per 100 alive frames, the
median frames between an agent's decisions, the triggers by kind, the switch rate (decisions that switched
plans / all decisions), the planning ms per game frame (the decisions' own time, plus the trigger checks) and
the share of decisions whose chosen plan the planner's collision guard clipped ("chosen plans clipped").
Per wave, after the summary line, a [cost] line (FrameCost): ms per frame (mean, median, 95%) spent planning,
observing (surprises, failed moves, deaths), on the explore agents, world-model data, telemetry, file logging, the
overlay, the level's live state, the rest, and waiting for the game (from the reply to the next frame message),
the frames by python + wait time (<=17, <=34, <=50, >50 ms: hitches; FrameCost.frame_hist), plus the game speed
(game seconds per wall-clock second); the same numbers go into the wave event.
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
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from game.level import Level, box_base, box_ids, hazard_boxes, identify_level, HZ, TELEPORT_JUMP   # noqa: E402
from mpc.candidates import ACTIONS                                       # noqa: E402
from mpc.future import FutureWorld                                       # noqa: E402
from plan import safety as S                                             # noqa: E402
from plan.commit import Committer, MIN_TUBE, REPLAN_DEFAULT, TUBE_DEFAULT, kind_of  # noqa: E402
from levels_smb import a_plus_time                                       # noqa: E402
from plan.moves import LIBRARY                                           # noqa: E402
from plan.planner import (STALL_FREE, DEATH_MIN_FRAMES, EXPLORE_EVERY, HYSTERESIS_FRAMES, TRIAL_GAIN, MoveMemory, Planner,   # noqa: E402
                          future_inputs, state_row)
from plan.progress import ProgressField                                  # noqa: E402
from shutdown import Shutdown, bounded                                   # noqa: E402
from viz import telemetry as T                                           # noqa: E402
from wm.rollout import next_held                                         # noqa: E402
from wm.protocol import feature_usage                                  # noqa: E402

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
TELEMETRY_EVERY = 3              # frames: the live frame message every 3rd frame (20 Hz at the locked 60 frames/s) and
                                 # on frames that matter (Runner.tel_due). Sent every frame it cost 9-11 ms of the
                                 # 16.7 ms of a real-time frame on 3-2 (runs/online_check04; 6-8 ms median on an
                                 # offline replay of two 3-2 waves), all of it for the live viewer only
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


def _pieces(pts, jumps=None, per_frame: bool = True) -> list:
    """A drawn path split where it goes through a game portal: at the point indices `jumps` (a per-step path's
    teleports: Plan.alternatives / completion "jumps"), else, for a per-frame path, where one frame moves more
    than TELEPORT_JUMP. One piece without."""
    pts = list(pts)
    if len(pts) < 2:
        return [pts]
    if jumps:
        cut = {int(k) for k in jumps if 0 < int(k) < len(pts)}
    elif per_frame:
        cut = {k for k in range(1, len(pts))
               if math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]) > TELEPORT_JUMP}
    else:
        cut = set()
    if not cut:
        return [pts]
    out, start = [], 0
    for k in sorted(cut):
        out.append(pts[start:k])
        start = k
    out.append(pts[start:])
    return [p for p in out if p]


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
            for p in _pieces(a["path"], a.get("jumps"), per_frame=False):
                out.append({"pts": _sub(p, ALT_STEP), "col": COL_ALT_VETO if a.get("vetoed") else COL_ALT, "w": 1})
    for p in _pieces(plan.path):
        out.append({"pts": _sub(p, PLAN_STEP), "col": MODE_COL.get(getattr(plan, "mode", "normal"), COL_PLAN),
                    "w": 3})
    comp = getattr(plan, "completion", None)
    if comp and len(comp.get("path") or []) >= 2:
        for p in _pieces(comp["path"], comp.get("jumps"), per_frame=False):
            out.append({"pts": _sub(p, PLAN_STEP), "col": COL_COMP, "w": 2, "dash": True})
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


ASYNC_LEAD = 8          # frames: --async-plan starts the next timed decision this far ahead (whole model steps)
# a plan made ahead is used only from a start this close to the real one: within the surprise tube's floor
# (commit.MIN_TUBE units) and this speed (units / frame); farther, its path broke the surprise check within 2-4
# frames in the simulated 1-15 runs (runs/experiments/async_plan_v1)
ASYNC_DV = 0.5
# --async-wait events (the default): the game waits for a decision, made at once from the real state, on the
# trigger kinds WAIT_KINDS["events"] and when the agent has no plan; a plan being made ahead is then dropped.
# Every other trigger's decision (interval, move end, short, missed) is made in the background for ASYNC_LEAD
# frames later while the committed plan plays on, and is used from a start within the surprise tube at that many
# frames (the error the surprise rule itself allows a prediction of that age). Surprise (the prediction broke)
# and support (landed early, not on the ground, on a wall) wait too: planned ahead (--async-wait danger, about
# 95% of decisions) they were answered up to ASYNC_LEAD frames late, by a plan made before the trigger from a
# predicted start, and a trigger before the ahead plan's frame was absorbed by it. --async-wait danger keeps that
# old rule (danger and start only) for A/B; all: every decision not planned ahead is waited for. After
# ASYNC_MISSES plans made ahead in a row started too far off, the next decision is made at once, from the real
# state.
# --async-wait react (new, not the default): the game never stops for a decision except danger, start (the
# wave's / level's first decision, "start: tiles") and no committed plan; every other decision is planned in the
# background from the committed plan's next L frames while that plan plays on (Runner.route_react):
# - L (frames, whole model steps) adapts to the planner process: the smallest multiple of the model step >= the
#   90% latency + 1 of the last REACT_WINDOW jobs (frames from a job's start to the first frame its reply was
#   seen; a job cancelled late gives its age then, a lower bound; other cancelled jobs give none), clamped to REACT_LEAD_MIN..REACT_LEAD_MAX, REACT_LEAD_INIT before any;
# - timed (interval, move end, short): a job starts once the next timed decision is due within L frames, for L
#   frames on (so it is made at max(due, start + L)); timed triggers while a job is in flight are absorbed;
# - events (REACT_EVENT_KINDS: surprise, support, missed): a routine job in flight (made for a timed decision
#   from before the event) is cancelled and a reactive one started from the real state now; with a reactive job
#   in flight the event is absorbed (a surprise fires every frame until the new plan is committed);
# - at or after its target frame a job is never waited for: not ready, the committed plan plays on (cancelled
#   and restarted more than REACT_LATE_MAX frames late); ready k frames late, it is used when the k actions
#   played since its target are its own first k and the real state is within the tube of its path at k
#   (lead + k frames old), else dropped and a new reactive job started. Too few committed actions left for a
#   job (not even one model step): the decision is made at once. ASYNC_MISSES never forces a decision at once.
# --planners N (N >= 2, react only): N planner processes (plan.worker.PlannerPool, every state call broadcast,
#   every committed plan adopted by the others). The first job is as above; with a job in flight a free process
#   starts a speculative one for max(the latest in-flight target + S, frame + L) frames (S = max(f, L / 2 in whole
#   model steps), rounded up to whole model steps) from the committed plan's actions up to it. Jobs are taken in
#   target order (react_take); a commit restarts every later job whose prefix over [its target, theirs) is not the
#   new plan's actions (same target when it still fits, else frame + L), a drop keeps them; an event cancels every
#   job and starts a reactive one, the other processes then get staggered speculative ones; danger, start and
#   pl_sync drop every job (the decision is made on the primary). Runner.route_pool.
# --async-wait auto (new, not the default): a level plays react until the agent first sees something move in it
#   (a moving platform, a kill box more than AUTO_MOVE units from where it was the frame before, a shot); from
#   then on, and on every later visit of that level in the run, events (Runner.switch_wait). Live, react made 3-5
#   decisions per 100 frames against events' 12-15 and failed on chapter 4's moving parts (4-6 0-1 of its first
#   11 waves against 8), while static chapter-3 levels were as good in react.
AUTO_MOVE = 0.5         # units a kill box's centre moves between two frames to count as moving (--async-wait auto)
ASYNC_MISSES = 2
WAIT_KINDS = {"danger": ("danger", "start"), "events": ("danger", "start", "surprise", "support"),
              "react": ("danger", "start")}
REACT_EVENT_KINDS = ("surprise", "support", "missed")
REACT_WINDOW = 16       # jobs whose latency sets the react lead
REACT_LEAD_MIN, REACT_LEAD_INIT, REACT_LEAD_MAX = 8, 12, 24      # frames
REACT_LATE_MAX = 8      # frames after its target a job not yet ready is still waited for (playing on)


def react_lead(latencies, f: int = 4) -> int:
    """--async-wait react: frames ahead a job is started for (module comment above): the smallest multiple of the
    model step f >= the 90% latency (frames) + 1, within REACT_LEAD_MIN..REACT_LEAD_MAX; REACT_LEAD_INIT
    without samples."""
    lat = list(latencies)
    if not lat:
        return REACT_LEAD_INIT
    f = max(1, int(f))
    lead = int(math.ceil((float(np.percentile(lat, 90)) + 1.0) / f - 1e-9)) * f
    return int(min(max(lead, REACT_LEAD_MIN), REACT_LEAD_MAX))


def react_span(want: int, left: int, f: int) -> int:
    """--async-wait react: the frames a job plays from the committed plan before its decision: `want` (the lead)
    or, with only `left` committed actions left, the most whole model steps of f that fit; 0 when not even one."""
    f = max(1, int(f))
    d = min(int(want), max(0, int(left)) // f * f)
    return d if d >= f else 0


class AheadJob:
    """A decision being planned ahead in the planner process (agent.py --async-plan): the call's id, the agent,
    the frame it started at, the frame it is for (target) and the deadly boxes from the target on. reactive
    (--async-wait react): started for an event (or to replace a dropped job), not for a timed decision;
    ready_at: the first frame its reply was seen (react polls once per frame), None before."""

    def __init__(self, mid: int, agent: int, start: int, target: int, fboxes, why=(), reactive: bool = False,
                 proc: int = 0, prefix=None, spec: bool = False, base=None):
        self.mid, self.agent, self.start, self.target, self.fboxes = int(mid), int(agent), int(start), int(target), fboxes
        self.why = list(why)                    # the triggers it answers ([]: the next timed decision)
        self.reactive = bool(reactive)
        self.ready_at = None
        # --planners N: the process it runs in, the committed actions it plays from its start to its target, whether
        # it was started speculatively (behind another job) and the committed plan it was started from
        self.proc, self.prefix, self.spec, self.base = int(proc), prefix, bool(spec), base


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
        return {"cost_ms": out, "frame_hist": self.frame_hist(), "wall_s": round(wall, 2), "game_s": round(game, 2),
                "game_speed": round(game / wall, 3) if wall > 0 else None}

    HIST = (17.0, 34.0, 50.0)                   # ms: about 1, 2, 3 screen refreshes of 16.7 ms

    def frame_hist(self) -> dict:
        """Frames by their python + wait time (ms; the time from one frame message to the next, which snaps to
        16.7 ms screen refreshes: a hitch is a frame over 17): {"<=17": n, "<=34": n, "<=50": n, ">50": n}. Only
        complete frames count: a frame's wait is known at the next frame's message, so the wave's last frame
        (no next message) is left out."""
        keys = [f"<={b:g}" for b in self.HIST] + [f">{self.HIST[-1]:g}"]
        n = min(len(self.rows), len(self.wait))
        hist = dict.fromkeys(keys, 0)
        if n:
            tot = 1000.0 * (np.asarray(self.rows[:n]).sum(1) + np.asarray(self.wait[:n]))
            idx = np.searchsorted(np.asarray(self.HIST), tot, side="left")     # <= bound: that bucket
            for k, c in zip(*np.unique(idx, return_counts=True)):
                hist[keys[int(k)]] = int(c)
        return hist

    def text(self, s: dict) -> str:
        c = s["cost_ms"]
        f = lambda v: f"{v[0]:.1f} ({v[1]:.1f}/{v[2]:.0f})"
        h = s.get("frame_hist")
        return ("[cost] ms/frame mean (median/95%): " + " ".join(f"{self.SHORT.get(k, k)} {f(c[k])}" for k in self.CATS)
                + f" | python {f(c['python'])}" + (f" | wait {f(c['wait'])}" if "wait" in c else "")
                + (" | frames (python + wait ms) " + " ".join(f"{k} {v}" for k, v in h.items()) if h else "")
                + f" | {s['game_s']:.1f} game s in {s['wall_s']:.1f} s: speed {s['game_speed']}x")


class BestTimes:
    """Best finish time per level, planned and explore agents apart:
    {level: {"planned": {seconds, frames, run, wave, agent, date}, "explore": {...}}}. Written atomically (temp
    file + replace); re-read before each update. A file that cannot be read is left alone (no updates)."""

    def __init__(self, path: str):
        self.path = path
        self.data, self.ok = self._load()

    def _load(self):
        if self.path is None:                 # isolated evaluation: no persistent best-time reads or writes
            return {}, True
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
        if self.path is None or not self.ok:
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


def react_summary(st: dict) -> dict:
    """A wave's --async-wait react leads and job latencies (Runner.st react_leads / react_latency, frames) for
    the wave event: {"lead": {median, max}, "latency_frames": {p50, p90}} (None without any)."""
    leads, lat = st.get("react_leads") or [], st.get("react_latency") or []
    return {"lead": {"median": float(np.median(leads)), "max": int(max(leads))} if leads else None,
            "latency_frames": {"p50": float(np.percentile(lat, 50)), "p90": float(np.percentile(lat, 90))}
            if lat else None}


def react_text(st: dict, r: dict) -> str:
    """'[react] waited for 2 decisions (410 ms) | ahead used 40 (5 late), dropped 2, restarted 6, late dropped 1,
    absorbed 30 | lead 12/16 f (median/max) | latency 9/11 f (50%/90%)'."""
    g = lambda k: st.get(k, 0)
    return (f"[react] waited for {g('sync_decisions')} decisions ({g('sync_ms'):.0f} ms) | ahead used "
            f"{g('ahead_used')} ({g('ahead_late_used')} late), dropped {g('ahead_dropped')}, restarted "
            f"{g('ahead_restarted')}, late dropped {g('ahead_late_dropped')}, absorbed {g('ahead_absorbed')}"
            + (f" | speculative started {g('ahead_spec_started')}, used {g('ahead_spec_used')}, cancelled "
               f"{g('ahead_spec_cancelled')}" if g("ahead_spec_started") else "")
            + (f" | lead {r['lead']['median']:g}/{r['lead']['max']} f (median/max)" if r["lead"] else "")
            + (f" | latency {r['latency_frames']['p50']:g}/{r['latency_frames']['p90']:g} f (50%/90%)"
               if r["latency_frames"] else ""))


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
        from plan.first_eval import validate_args
        validate_args(args)
        self.args = args
        self.run_dir = os.path.join(HERE, "runs", args.name)
        first = bool(getattr(args, "eval_first", None))
        if first and os.path.exists(self.run_dir) and os.listdir(self.run_dir):
            raise SystemExit("--eval-first needs a new empty run folder (existing evidence is never appended)")
        os.makedirs(self.run_dir, exist_ok=True)
        self.evaluation = None
        self.demo = getattr(args, "demo_play", None)       # plan.demos.DemoPlay (--demo / --demo-level)
        self.demo_fin, self.demo_trail = None, []
        if self.demo is not None:
            args.agents, args.explore = 0, 0
        self.search = None                                   # plan.search.Search (--search)
        n_search = 0
        self.human = None                                    # plan.human.HumanInput (--human): agent 0 is you
        if getattr(args, "human", False):
            if self.demo is not None or getattr(args, "search", False) or getattr(args, "learn", False):
                raise SystemExit("--human plays alone: not with --demo / --search / --learn")
            from plan.human import HumanInput
            self.human = HumanInput()
            args.agents, args.explore, args.end_after = 0, 0, 1
            n_search = 1                                     # (counted as the one agent below)
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
            if first:
                from plan.first_eval import load_checkpoint, FirstEvaluation
                from wm.buffer import eval_levels
                self.model, identity = load_checkpoint(path)
                self.evaluation = FirstEvaluation(self.run_dir, args, identity,
                                                  args.eval_first in eval_levels(), HERE)
            else:
                self.model = WorldModel.load(path).eval()
                if getattr(args, "online", False):      # the online learner's replay must be of this model's encoding
                    from wm.online import replay_mismatch
                    why = replay_mismatch(json.loads(str(np.load(args.online_replay)["meta"])), self.model)
                    if why:
                        raise SystemExit(f"--online: {args.online_replay} cannot be trained with {path}: {why}")
            self.model_path = path
        self.learn = bool(getattr(args, "learn", False)) and self.model is not None
        self.learner = None                     # --learn: the wm.learner child process
        self.live_mtime = os.path.getmtime(self.model_path) if self.model is not None else 0.0
        self.commit = Committer(self.replan, args.tube, args.triggers == "on",
                                step=getattr(self.model, "step", 1) if self.model is not None else 1)
        # --async-plan: the next timed decision is planned in a background thread while the game plays
        # (AheadJob); only for one planned agent with nothing else driven by the planner
        self.async_on = bool(getattr(args, "async_plan", False))
        why_not = ("--eval-first" if first else "not one planned agent" if self.n_plan != 1 else
                   "explore agents" if self.n_explore else "--search" if self.search is not None else
                   "--demo" if self.demo is not None else "no world model" if self.model is None else None)
        if self.async_on and why_not:
            print(f"[async] --async-plan is off: {why_not}", flush=True)
            self.async_on = False
        self.async_lead = int(getattr(args, "async_lead", ASYNC_LEAD))
        wait_mode = getattr(args, "async_wait", "events")
        # --async-wait auto: each level starts in react, events once something moved in it (switch_wait)
        self.auto_wait = self.async_on and wait_mode == "auto"
        self.dyn_levels: dict[str, dict] = {}   # auto: level -> what was first seen moving there (events from then)
        self.react_lat = deque(maxlen=REACT_WINDOW)     # react: the last jobs' latencies (frames), for the lead
        # wait_kinds (trigger kinds decided at once), nowait, react (route_react) and, for react, the committer's
        # danger check on event triggers too (a surprise is no longer decided at once, so it must not hide a
        # danger: Committer.danger_always)
        self.set_wait_mode("react" if self.auto_wait else wait_mode)
        self.ahead_miss = 0                     # plans made ahead in a row that started too far off
        self.ajob = None                        # the plan being made ahead (AheadJob) or None
        self.jobs = []                          # --planners N >= 2 in react: the plans being made ahead, by target
        n_pl = max(1, int(getattr(args, "planners", 1) or 1))
        if n_pl > 1 and not (self.async_on and wait_mode in ("react", "auto")):
            print("[async] --planners is for --async-plan with --async-wait react or auto: one planner process",
                  flush=True)
            n_pl = 1
        self.n_planners = n_pl
        self.proc = None                        # plan.worker.PlannerProcess (PlannerPool): the planner, its memories
        self.proc_ready = False                 # the process has its Planner (build_field: init)
        if self.async_on:
            from plan.worker import PlannerPool, PlannerProcess
            self.proc = PlannerPool(n_pl, args.threads) if n_pl > 1 else PlannerProcess(args.threads)
            pids = [p.proc.pid for p in self.proc.procs] if n_pl > 1 else [self.proc.proc.pid]
            print(f"[async] planner process{'es' if n_pl > 1 else ''} started (pid {', '.join(map(str, pids))}); "
                  "decisions planned "
                  + (f"{REACT_LEAD_MIN}-{REACT_LEAD_MAX} frames ahead (react: the planner's own latency, "
                     f"{REACT_LEAD_INIT} at first)" if self.react else f"{self.async_lead} frames ahead")
                  + "; the game waits "
                  + (f"only for {', '.join(self.wait_kinds)} decisions" if self.nowait
                     else "for every decision not planned ahead")
                  + ("; auto: react on each level until something moves there, then events" if self.auto_wait
                     else "")
                  + (f"; {n_pl} planner processes (react: speculative plans ahead)" if n_pl > 1 else ""), flush=True)
        self.deaths_path = args.death_memory
        self.deaths = S.DeathMemory() if first else S.DeathMemory.load(self.deaths_path)
        mm = getattr(args, "move_memory", DEFAULT_MOVES)
        self.moves = None if first or mm is None or str(mm).lower() == "none" else MoveMemory.load(mm)
        from wm.buffer import eval_levels
        self.held_out = set(eval_levels())      # never saved in either memory (practice is for this run only)
        self.deaths.transient = set(self.held_out)
        if self.moves is not None:
            self.moves.transient = set(self.held_out)
        self.level_waves: dict[str, int] = {}   # level -> full waves played in this run (on_wave_end)
        self.practising: set[str] = set()       # held-out levels whose practice was announced
        self.f_plans = open(os.path.join(self.run_dir, "plans.jsonl"), "a", encoding="utf-8")
        self.f_events = open(os.path.join(self.run_dir, "events.jsonl"), "a", encoding="utf-8")
        self.f_states = open(os.path.join(self.run_dir, "states.jsonl"), "a", encoding="utf-8")
        self.f_times = open(os.path.join(self.run_dir, "times.jsonl"), "a", encoding="utf-8")
        self.f_search = open(os.path.join(self.run_dir, "search.jsonl"), "a", encoding="utf-8") \
            if self.search is not None else None
        self.best = BestTimes(None if first else args.best_times)
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
            if self.human is not None:
                self.wm = Collector(os.path.join(self.run_dir, "wm_data"), source="human")
                self.wm.meta_extra = {"agent_sources": {"human": [0]}}
            elif self.demo is not None:
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
        self.wm_held = None                     # --online: the held-out levels' play (runs/<name>/wm_heldout: never in
        if self.wm is not None and getattr(args, "online", False) and self.evaluation is None:   # offline training)
            from wm.buffer import Collector
            self.wm_held = Collector(os.path.join(self.run_dir, "wm_heldout"), source=self.wm.source)
            self.wm_held.meta_extra = dict(self.wm.meta_extra, heldout=True)
        self.online = None                      # --online: the wm.online child process
        self.online_mtime = 0.0
        if telemetry is not None:
            self.tel = telemetry
        else:
            addr = T.parse_addr(args.telemetry)
            self.tel = T.Telemetry(addr) if addr else None
        self.tel_every = max(1, int(getattr(args, "telemetry_every", TELEMETRY_EVERY)))
        self.tel_last = None                    # (level, wave, frame) of the last frame message sent
        self.tel_force = False                  # this frame's message is sent (a decision, a death, a finish)
        self.tel_trail = []                     # [frame, agent, x, y] of the planned agents on frames not sent
        self.tel_pending = None                 # the last frame's send_frame arguments when it was not sent
        self.rng = np.random.default_rng(args.seed)
        self.level: Level | None = None
        self.planner: Planner | None = None
        self.planner_level = None
        self.field = None                       # the level's progress field (the search's T)
        self.field_summary = None
        self.field_builds = 0
        self.progress_msg = None                # telemetry "progress" (viz/telemetry.py) of the current field
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
                   "pnr_none": 0, "completed": 0, "all_agent_frames": 0,
                   # decisions the game waited for (plan_now / an in-process plan) and their ms; --async-wait
                   # react's job counts; the react leads used and job latencies (frames) of this wave
                   "sync_decisions": 0, "sync_ms": 0.0, "ahead_restarted": 0, "ahead_late_used": 0,
                   "ahead_late_dropped": 0, "react_leads": [], "react_latency": [],
                   # --planners N: speculative jobs started / committed / cancelled (restarted, events, drops)
                   "ahead_spec_started": 0, "ahead_spec_used": 0, "ahead_spec_cancelled": 0}
        self.fin = {"planned": [], "explore": [], "demo": [], "search": []}     # this wave's finish times (s)
        self.new_best = False
        if self.demo is not None:                 # this wave's perturbed copies (same for repeated calls)
            self.demo.new_wave()
            self.demo_fin, self.demo_trail = None, []
            if self.wm is not None:
                self.wm.meta_extra["perturbations"] = self.demo.changes

    def on_info(self, info: dict, tiles: bytes):
        self.pl_sync(drop="new level data")
        if info["W"] * info["H"] == 0 or len(tiles or b"") != info["W"] * info["H"] * 4:
            if self.evaluation is not None:
                self.evaluation.invalidate("invalid level data")
                self.stopping = True
            print(f"[level] ignoring invalid level data {info['W']}x{info['H']}", flush=True)
            return
        if info["name"] in (None, "unknown"):
            info["name"] = identify_level(info, tiles) or "unknown"
        if self.evaluation is not None and not self.evaluation.info(info["name"], info["agents"]):
            self.stopping = True
            return
        if self.demo is not None and info["name"] != self.demo.demo["level"]:
            print(f"[demo] refused: the game has level {info['name']} loaded, the demo is for "
                  f"{self.demo.demo['level']} (enter {self.demo.demo['level']} and start again); stopping", flush=True)
            self.level = None
            self.stopping = True
            return
        self.level = lv = Level(info, tiles)
        if getattr(self, "auto_wait", False):  # react until something moves here (events if it already did)
            self.level_wait(lv.name)
        if self.n_total == 1 and lv.keys:      # one agent: the game's own locks (they dissolve in their time)
            lv.use_native_locks()
            print(f"[keys] {lv.name}: one agent, so the game opens the locks itself (no per-agent locks)", flush=True)
        self.cost.reset()
        from wm.buffer import eval_levels
        self.protected = self.evaluation is not None or info["name"] in eval_levels()
        if self.practice_on(info["name"]):         # back on a level whose first life this run already played
            self.start_practice(info["name"])
        for c in (self.wm, self.wm_held):           # the other collector's wave in progress ends with its level
            if c is not None and c is not self.rec_for(self.protected):
                c.cut("level change")
        if self.rec is not None:
            self.rec.level(lv, info)
        self.fw.new_level()
        self.commit.tube.refresh()                 # the surprise threshold, fixed for this level
        self.hz_start, self.moving, self.sweep = None, set(), set()
        self.field_built, self.field_dirty, self.rebuilds = False, False, 0
        self.progress_msg = None
        self.lt_changes = 0
        self.wave_no = 0
        kb = lv.kill_box
        best = self.best.best(lv.name, "planned")
        print(f"[level] {lv.name}: {lv.W}x{lv.H}, {len(lv.goals)} goal(s), {info.get('lethal', '?')} lethal pieces, "
              f"{len(lv.keys)} key(s), {info['agents']} agents (planned {self.n_plan}, explore {self.n_explore}); "
              + (f"agent best {float(best['seconds']):.2f} s; " if best and best.get("seconds") is not None
                 else "no agent best yet; ")
              + (f"kill box x {kb[0]:.0f}..{kb[1]:.0f}, y {kb[2]:.0f}..{kb[3]:.0f}" if kb else "no kill box")
              + (" - first-life evaluation: no training or persistent memories" if self.evaluation is not None else
                 (" - held-out evaluation level: " + ("practising (move and death memory on, not saved)"
                                                       if self.practice_on(lv.name) else
                                                       "first life clean; recorded to wm_heldout, after it move and "
                                                       "death memory on for this run (not saved)"
                                                       if getattr(self.args, "online", False) else
                                                       "no world-model data, no death memory")
                  if self.protected else "")),
              flush=True)
        self.event("level", None, name=lv.name, W=lv.W, H=lv.H, kill_box=list(kb) if kb else None,
                   protected=self.protected, practice=self.practice_on(lv.name), agents=info["agents"],
                   features=feature_usage(self.model.info)
                            if self.model is not None else None,
                   tube=self.commit.tube.summary() if self.commit.on else None,
                   model=self.model_path if self.model is not None else ("none (search)" if self.search else None),
                   wait_mode=getattr(self, "wait_mode", None) if getattr(self, "async_on", False) else None)
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
        self.field_builds += 1
        self.progress_msg = None
        try:                                    # the viewer's progress line (P): sent with the level, kept for replay
            from plan.progress import best_times
            tb = best_times(field)
            self.progress_msg = T.progress_payload(tb, field.route_cells, field.key_order, self.field_builds)
            fdir = os.path.join(self.run_dir, "fields")
            os.makedirs(fdir, exist_ok=True)
            np.savez_compressed(os.path.join(fdir, f"{lv.name}.npz"), T=tb.astype(np.float16),
                                route=np.asarray(field.route_cells, np.int64).reshape(-1, 2),
                                key_order=json.dumps(field.key_order), ox=lv.ox, oy=lv.oy)
        except Exception as e:                  # display only: never stops the agent
            print(f"[field] progress display not saved: {type(e).__name__}: {e}", flush=True)
        if self.model is not None:
            if self.planner is None:
                self.planner = Planner(self.model, lv, field, budget_ms=self.args.budget_ms,
                                       hysteresis=self.args.hysteresis, moves=self.moves,
                                       veto_members=getattr(self.args, "veto_members", 1),
                                       death_min=getattr(self.args, "death_min", DEATH_MIN_FRAMES),
                                       risk_members=getattr(self.args, "risk_members", "second"),
                                       corner=getattr(self.args, "corner_risk", "on") == "on",
                                       early_margin=getattr(self.args, "reflex", "veto") == "veto",
                                       stall_free=getattr(self.args, "stall_free", STALL_FREE),
                                       modes=self.args.modes == "on",
                                       route_trial_gain=getattr(self.args, "route_trial_gain", TRIAL_GAIN))
                cm = self.commit                    # route trials: deviation = the surprise trigger's tube
                self.planner.trial_tube = lambda k: max(cm.tube_factor * cm.tube.at(k), MIN_TUBE)
            elif self.planner_level is not lv:
                self.planner.set_level(lv, field)
            else:
                self.planner.set_field(field)
                self.planner.bounds = S.level_bounds(lv)
            if self.moves is not None:
                self.moves.learn = self.memory_on()       # held-out first life: no charges, no tries
            if self.proc is not None:               # the planner process: the same level, field and settings
                from plan.worker import TubeFn
                tube = TubeFn(self.commit.tube, self.commit.tube_factor, MIN_TUBE)
                if not self.proc_ready:
                    mm = getattr(self.args, "move_memory", DEFAULT_MOVES)
                    opts = {"budget_ms": self.args.budget_ms, "hysteresis": self.args.hysteresis,
                            "veto_members": getattr(self.args, "veto_members", 1),
                            "death_min": getattr(self.args, "death_min", DEATH_MIN_FRAMES),
                            "risk_members": getattr(self.args, "risk_members", "second"),
                            "corner": getattr(self.args, "corner_risk", "on") == "on",
                            "early_margin": getattr(self.args, "reflex", "veto") == "veto",
                            "stall_free": getattr(self.args, "stall_free", STALL_FREE),
                            "modes": self.args.modes == "on",
                            "route_trial_gain": getattr(self.args, "route_trial_gain", TRIAL_GAIN)}
                    self.proc.call("init", self.model_path, lv, field, opts,
                                   None if self.moves is None else mm, tube)
                    self.proc_ready = True
                elif self.planner_level is not lv:
                    self.proc.call("set_level", lv, field, tube)
                else:
                    self.proc.call("set_field", field, S.level_bounds(lv))
                self.proc.call("moves_transient", sorted(self.held_out))   # idempotent: never saved there either
                self.proc.call("moves_learn", self.memory_on())
            self.planner_level = lv
        self.field_built, self.field_dirty = True, False
        dt = time.time() - t
        print(f"[field] {lv.name} ({why}): {self.field_summary}; {len(static)} fixed hazards, {len(self.moving)} "
              f"moving, {len(sweeps)} sweep boxes [{dt:.2f}s]", flush=True)
        self.event("field", None, why=why, summary=self.field_summary, fixed=len(static), moving=len(self.moving),
                   seconds=round(dt, 3))

    def practice_on(self, name) -> bool:
        """A held-out level (eval_levels) practises with --online once this run has played a full wave of it (its
        first life is over); never in a first-life evaluation or without --online."""
        return (self.evaluation is None and bool(getattr(self.args, "online", False)) and name in self.held_out
                and self.level_waves.get(name, 0) >= 1)

    def memory_on(self) -> bool:
        """The current level's move charges / tries and death memory are on: a training level, or a held-out level
        in practice (their entries then stay transient: not saved)."""
        return not self.protected or (self.level is not None and self.practice_on(self.level.name))

    def start_practice(self, name: str):
        """Practice starts (between waves, or at the level's load): the move memory learns in both processes."""
        if self.moves is not None:
            self.moves.learn = True
        if self.proc is not None and self.proc_ready:
            self.proc.call("moves_learn", True)
        if name in self.practising:
            return
        self.practising.add(name)
        print(f"[practice] {name}: first life over; move and death memory on for this run (not saved)", flush=True)
        self.event("practice", None, name=name, waves=self.level_waves.get(name, 0))

    def send_level(self):
        if self.tel is None or self.level is None:
            return
        self.level_sent = time.time()
        self.tel.send(T.level_message(self.level, self.wave_no, self.n_plan, self.n_explore, self.field_summary,
                                    progress=self.progress_msg))

    # ---- one frame ---------------------------------------------------------------------------------------------
    def on_frame(self, m: dict, data=None) -> dict:
        lv, states = self.level, m["s"]
        n = len(states)
        if self.evaluation is not None:
            if self.stopping or lv is None or not self.evaluation.observe(m, lv.spawn, self.lock):
                if lv is None and self.evaluation.result is None:
                    self.evaluation.invalidate("frame arrived without the target level")
                self.stopping = True
                return {"a": [NOOP] * n, "leader": 0, "stop": True}
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
        if self.proc is not None and self.field_built:
            upd = {k: m[k] for k in ("tc", "fx", "kb", "pf", "lt") if m.get(k)}
            if upd:
                self.proc.send("frame_level", upd)
        boxes, kept = hazard_boxes(m.get("hz", []), lv.hz_keep)
        ids = box_ids(kept, lv.hz_ids, prefix="")
        self.track_hazards(boxes, ids)
        if not self.field_built:
            self.build_field("first frame")
            pl = self.planner
        dt = float(m.get("dt") or 16.667)
        self.dt_ms = dt
        self.fw.observe(frame, m.get("hz", []), m.get("pf"), m.get("sh", []), lv.hz_keep, lv.fans, dt,
                        terrain=pl.terrain if pl is not None else None)     # live tiles: where shots stop
        if getattr(self, "auto_wait", False):       # --async-wait auto: the first moving part seen switches to events
            self.auto_check(frame, m, lv)
        if self.rec is not None:
            C.go("wm_data")
            self.rec.frame(m, list(lv.hz_ids), list(lv.hz_keep))

        # what happened since the last frame: deaths, finishes, surprises, failed moves
        C.go("observe")
        if self.proc is not None:
            self.proc_events()
        alive_before, held_before = list(self.alive), list(self.held)
        alive_xy = []
        for i, s in enumerate(states):
            if not self.alive[i]:
                continue
            x, y, st = float(s[0]), float(s[1]), int(s[6])
            if st == 0:
                alive_xy.append((x, y))
            keys = int(s[7]) if len(s) > 7 else 0
            if i < self.n_plan and pl is not None:
                if st == 1:
                    self.on_death(i, frame, self.prev_xy[i], (x, y), keys, m, None, estimated=False)
                else:
                    self.observe(i, frame, x, y, keys, dead=False, vx=float(s[2]), vy=float(s[3]),
                                 held=held_before[i])
            if st != 0:
                self.alive[i] = False
                planned = i < self.n_plan
                if st == 2:
                    # realgame.js counts frame++ after each update: status 2 first shown in the message of
                    # frame f means the goal was touched in update f - 1
                    self.on_finish(i, frame - 1, x, y)
                elif planned and pl is None:
                    self.st["died"] += 1
                elif self.search is not None and self.human is None:
                    self.search_death(i, frame, x, y)
                elif not planned:
                    self.st["ex_died"] += 1
            self.prev_xy[i] = (x, y)
            self.last_state[i] = s
        if alive_xy and pl is not None:             # where real agents survive: a remembered place's refutation
            self.observe_alive(lv.name, frame, alive_xy)

        # decisions
        C.go("planning")
        acts, act_idx = [NOOP] * n, [-1] * n
        fboxes = None
        live_plan = [i for i in range(min(self.n_plan, n)) if self.alive[i]] if pl is not None else []
        need = {}
        fut = {}

        def fboxes_now():                           # this frame's deadly boxes (the danger check), once
            if "fb" not in fut:
                _, fut["fb"], _ = future_inputs(self.fw, frame, pl.H, dt, lv.name, self.deaths, deadly_only=True)
            return fut["fb"]
        if live_plan:
            t_chk = time.perf_counter()
            tiles = bool(m.get("tc"))
            for i in live_plan:
                s = states[i]
                tr = self.commit.triggers(i, frame, float(s[0]), float(s[1]), int(s[7]) if len(s) > 7 else 0, tiles,
                                          pl, fboxes_now)
                tr = self.route_triggers(i, frame, s, tr, held_before[i], dt)
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
                t_sync = time.perf_counter()
                try:
                    danger = any(kind_of(t) == "danger" for t in trig)
                    if self.proc is not None:
                        plan, trial_ev = self.proc.call("plan_now", row, frame, worlds, fboxes, dt, agent=i,
                                                        danger=danger, recorded=self.fut_rec)
                        if trial_ev:
                            plan.route_trial = trial_ev
                    else:
                        plan = pl.plan(row, frame, worlds, fboxes, dt, agent=i, danger=danger, recorded=self.fut_rec)
                except Exception as e:                  # never leave the agent without an answer
                    self.count_sync(t_sync)
                    self.st["errors"] += 1
                    if self.st["errors"] <= 3:
                        import traceback
                        traceback.print_exc()
                    self.event("plan error", frame, agent=i, error=f"{type(e).__name__}: {e}")
                    if self.evaluation is not None:
                        self.evaluation.invalidate(f"planner error: {type(e).__name__}: {e}")
                        self.stopping = True
                        return {"a": [NOOP] * n, "leader": 0, "stop": True}
                    continue
                self.count_sync(t_sync)
                self.cur_plan[i] = plan
                self.ahead_miss = 0
                self.commit.decided(i, plan, frame, fboxes, pl)
                self.on_plan(plan, s, trig)
                ev = getattr(plan, "route_trial", None)
                if ev:                                  # RouteTrials: a better plan than the best route, tried once
                    self.event(ev["kind"], frame, **{a: b for a, b in ev.items() if a not in ("kind", "frame")})
        for i in live_plan:
            p = self.cur_plan.get(i)
            k = frame - p.frame if p is not None else -1
            a = int(p.actions[k]) if p is not None and 0 <= k < len(p.actions) else 0
            acts[i], act_idx[i] = list(ACTIONS[a]), a
            self.pl_call("played", i, frame, a)
        if self.async_on and live_plan and self.pool_react():
            self.pool_fill(live_plan[0], frame, states[live_plan[0]], held_before[live_plan[0]], dt)
        elif self.async_on and self.ajob is None and live_plan:
            self.ahead_start(live_plan[0], frame, states[live_plan[0]], held_before[live_plan[0]], dt)
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
        if self.human is not None:
            for i in range(n):
                if self.alive[i] and int(states[i][6]) == 0:
                    a = self.human.action() if i == 0 else 0
                    acts[i], act_idx[i] = list(ACTIONS[a]), a
        for i in range(self.n_plan, n if self.demo is None and self.search is None and self.human is None else 0):
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
                                                              or self.human is not None else min(self.n_plan, n))]})
        if self.tel is not None:
            C.go("telemetry")
            if self.tel_due(frame):
                self.send_frame(m, frame, states, act_idx, boxes, kept, fboxes, dt)
            else:
                self.skip_frame(m, frame, states, act_idx, boxes, kept, fboxes, dt)
        if self.rec is not None:
            C.go("wm_data")
            self.rec.agents(self.wave_no, frame, states, alive_before, held_before, act_idx, [float("nan")] * n)
        C.go("other")
        if self.demo is not None and frame >= self.demo.end_frame:
            # the demo's actions ran out a second ago: copies still standing count as done (they would keep the
            # wave going to --wave-seconds); realgame.js ends the wave on "end" before stepping anyone
            return {"a": acts, "leader": int(leader), "end": "demo done"}
        if self.search is not None and self.search.active and frame >= self.search.end_frame:
            # every agent's sequence ran out SETTLE frames ago: the wave is done (realgame.js ends it on "end")
            return {"a": acts, "leader": int(leader), "end": "search done"}
        return {"a": acts, "leader": int(leader)}

    def on_plan(self, plan, s, trigger=(), ahead=None):
        st = self.st
        self.tel_force = True                       # the viewer gets the frame of every decision
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
                  "future_rec": int(getattr(self, "fut_rec", 0)), "ahead": ahead,
                  "real": {"x": s[0], "y": s[1], "vx": s[2], "vy": s[3], "flags": int(s[4]), "state": int(s[5]),
                           "status": int(s[6]), "keys": int(s[7]) if len(s) > 7 else 0}})
        self._write(self.f_plans, d)
        if self.tel is not None:
            self.cost.go("telemetry")
            self.tel.send(T.plan_message(self.level.name, self.wave_no, d))
        self.cost.go(prev)

    def observe(self, i, frame, x, y, keys, dead, cause=None, vx=None, vy=None, held=None):
        """Planner.observe: surprises, failed moves, the route book's record (held: the buttons held before this
        frame's action) and (deaths) the point of no return, counted and logged (held while the planner plans
        ahead in the background: pl_call)."""
        if self.proc is not None:                   # in order behind a plan being made; events come back later
            self.proc.send("observe", i, frame, x, y, keys, dead, cause=cause, vx=vx, vy=vy, dt_ms=self.dt_ms,
                           held=held)
            return
        self.observe_events(self.planner.observe(i, frame, x, y, keys, dead, cause=cause, vx=vx, vy=vy,
                                                 dt_ms=self.dt_ms, held=held), frame)

    def observe_events(self, events, frame):
        for ev in events:
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

    # ---- planning ahead (--async-plan) ----------------------------------------------------------------------
    def pl_call(self, name: str, *a):
        """played(): to the planner process when there is one (in order behind a plan being made), else here."""
        if self.proc is not None:
            self.proc.send(name, *a)
        else:
            getattr(self.planner, name)(*a)

    def proc_events(self):
        """The planner process's replies to the calls sent without waiting: their events logged, errors too."""
        for name, ok, res in self.proc.take_events():
            if not ok:
                self.event("planner process error", None, call=name, error=str(res)[:2000])
                print(f"[async] planner process, {name}: {str(res).splitlines()[0]}", flush=True)
            elif name == "observe" and res:
                self.observe_events(res, None)

    def pl_sync(self, drop: str | None = None):
        """Before the planner's state is used here (death, finish, wave end, level change, stop): the plan being
        made ahead is called off (drop: the reason, logged), every sent call is done and its events logged."""
        if self.proc is None:
            return
        j = self.ajob
        if j is not None and drop is not None:
            t = time.perf_counter()
            self.proc.cancel(j.mid)
            self.ajob = None
            st = getattr(self, "st", None)
            if st is not None:
                st["ahead_wait_ms"] = st.get("ahead_wait_ms", 0.0) + 1000 * (time.perf_counter() - t)
                st["ahead_dropped"] = st.get("ahead_dropped", 0) + 1
            self.event("ahead dropped", j.target, agent=j.agent, reason=drop, started=j.start)
        if getattr(self, "jobs", None) and drop is not None:    # --planners N: every job, every process
            self.pool_cancel(None, drop, list(self.jobs), count="ahead_dropped")
        self.proc.call("ping")
        self.proc_events()

    def route_triggers(self, i: int, frame: int, s, tr: list, held: int, dt: float) -> list:
        """What agent i's triggers of this frame lead to: the triggers to decide on now, from the real state, or []
        (nothing now). With --async-plan: a trigger of a waiting kind (--async-wait: self.wait_kinds), no plan, or
        too many misses waits: a plan being made ahead is dropped (ahead_take) and the decision is made now. Any
        other trigger before the ahead plan's frame is absorbed by it (the committed plan plays on), else its
        decision starts in the background for async_lead frames on. --async-wait react: route_react. Without
        --async-plan: tr as it is."""
        if getattr(self, "react", False):
            if self.pool_react():
                return self.route_pool(i, frame, s, tr, held, dt)
            return self.route_react(i, frame, s, tr, held, dt)
        j = self.ajob
        wait = (not self.nowait or self.cur_plan.get(i) is None or self.ahead_miss >= ASYNC_MISSES
                or any(kind_of(t) in self.wait_kinds for t in tr))
        if j is not None and j.agent == i:
            if frame >= j.target or (tr and wait):
                tr = self.ahead_take(i, frame, s, tr)
            elif tr:                                    # the plan being made answers it: the committed one plays on
                self.st["ahead_absorbed"] = self.st.get("ahead_absorbed", 0) + 1
                tr = []
        if tr and not wait and self.ajob is None and self.ahead_miss < ASYNC_MISSES and \
                self.ahead_start(i, frame, s, held, dt, frame + self.async_lead, tr):
            tr = []                                     # decided in the background, for async_lead frames on
        return tr

    def ahead_start(self, i: int, frame: int, s, held: int, dt: float, target: int | None = None, why=(),
                    reactive: bool = False, proc: int | None = None, spec: bool = False, exact: bool = False) -> bool:
        """Start planning a decision of agent i in the planner process: its next timed one (Committer.due), or
        the one for frame `target` that answers the triggers `why` (a kind not waited for). Only a whole number of
        model steps away, at most --async-lead frames, within the committed plan's actions: from the state those
        actions are imagined to reach then (Planner.predict_start), with this frame's view of the future shifted.
        --async-wait react: the timed one once it is due within the lead L (react_lead), for L frames on (never
        before its due frame); `target` asks for target - frame frames on; either shrinks to the whole model steps
        the committed plan still has (react_span). reactive: AheadJob.reactive. True when it started.
        --planners N: proc = the process it runs in (the job goes into self.jobs, by target; None: the primary,
        self.ajob); spec: a speculative job (behind another one, AheadJob.spec); exact: `target` rounded up to
        whole model steps from now, never shrunk (False when the committed plan has too few actions)."""
        p, pl, lv = self.cur_plan.get(i), self.planner, self.level
        if getattr(self, "react", False):
            if p is None or pl is None or lv is None or self.proc is None:
                return False
            k0 = frame - p.frame
            due = None
            if target is None:                          # the next timed decision, once due within the lead
                due, want = self.commit.due(i), self.lead_now()
                if due is None or due - frame > want:
                    return False
            else:
                want = target - frame
            if exact:
                d = -(-int(want) // pl.f) * pl.f
                if d < pl.f or k0 < 0 or d > (len(p.actions) - k0) // pl.f * pl.f:
                    d = 0
            else:
                d = react_span(want, len(p.actions) - k0, pl.f) if k0 >= 0 else 0
            if d == 0 or (due is not None and frame + d < due):
                return False
            due = frame + d
            if not spec:
                self.st.setdefault("react_leads", []).append(d)
        else:
            due = self.commit.due(i) if target is None else target
            if p is None or due is None or pl is None or lv is None or self.proc is None:
                return False
            d = due - frame
            k0 = frame - p.frame
            if d < pl.f or d > self.async_lead or d % pl.f or k0 < 0 or k0 + d > len(p.actions):
                return False
        worlds, fboxes, rec = future_inputs(self.fw, frame, pl.future_frames + d, dt, lv.name, self.deaths)
        row = state_row(s[0], s[1], s[2], s[3], s[4], s[5], s[7] if len(s) > 7 else 0, held)
        prefix = np.asarray(p.actions[k0:k0 + d], np.int64)
        if proc is None:
            mid = self.proc.send("plan_ahead", row, prefix, worlds, fboxes, d, due, dt, i, rec, keep=True)
            self.ajob = AheadJob(mid, i, frame, due, fboxes[d:], why, reactive)
            return True
        mid = self.proc.send("plan_ahead", row, prefix, worlds, fboxes, d, due, dt, i, rec, keep=True, to=int(proc))
        self.jobs.append(AheadJob(mid, i, frame, due, fboxes[d:], why, reactive, proc=proc, prefix=prefix, spec=spec,
                                  base=p))
        self.jobs.sort(key=lambda x: x.target)
        if spec:
            self.st["ahead_spec_started"] = self.st.get("ahead_spec_started", 0) + 1
        return True

    def ahead_take(self, i: int, frame: int, s, tr: list) -> list:
        """At the job's frame: its plan becomes the committed one when it was made, no trigger other than a timed
        one (interval, move end, short) fired, and the real state is within MIN_TUBE units and ASYNC_DV units /
        frame of the predicted start (its path is then moved onto the real start); returns [] then. Otherwise (or a
        trigger before its frame) the job is dropped and the triggers that need a decision now are returned; the
        result is waited for (proc.result). --async-wait danger / events / all only: react never waits for a job
        and takes it with react_take (route_react)."""
        j = self.ajob
        if frame != j.target:                       # named by the trigger that waits (danger, surprise, support)
            why = next((t_ for t_ in tr if kind_of(t_) in self.wait_kinds), tr[0] if tr else "frame")
            self.pl_sync(drop=f"{why} at {frame}, before its frame {j.target}")
            return tr or ["start"]
        t = time.perf_counter()
        ok, res = self.proc.result(j.mid)
        self.st["ahead_wait_ms"] = self.st.get("ahead_wait_ms", 0.0) + 1000 * (time.perf_counter() - t)
        self.ajob = None
        # --async-wait events / danger: the plan was made as the answer to every trigger but the waiting kinds
        # (self.wait_kinds); its start may be as far off as the surprise rule lets a prediction of that age be
        urgent = [t_ for t_ in tr if (kind_of(t_) in self.wait_kinds if self.nowait else
                                      kind_of(t_) not in ("interval", "move end", "short"))]
        lead = j.target - j.start
        tube = max(self.commit.tube_factor * self.commit.tube.at(lead), MIN_TUBE) if self.nowait else MIN_TUBE
        off = None
        if not ok:
            reason = f"planning error: {str(res).splitlines()[0]}"
        elif urgent:
            reason = urgent[0]
        else:
            plan, px, py, spread, pvx, pvy = res
            off = math.hypot(float(s[0]) - px, float(s[1]) - py)
            dv = math.hypot(float(s[2]) - pvx, float(s[3]) - pvy) * self.dt_ms / 1000.0     # units / frame
            reason = (f"start off by {off:.1f} > {tube:.1f} units" if off > tube else
                      f"start speed off by {dv:.2f} > {ASYNC_DV:g} units/frame" if dv > ASYNC_DV else None)
        if reason is not None:
            if ok:
                self.proc.send("discard", j.mid)
            self.st["ahead_dropped"] = self.st.get("ahead_dropped", 0) + 1
            if not urgent:
                self.ahead_miss += 1
            self.event("ahead dropped", frame, agent=i, reason=reason, started=j.start)
            return tr or j.why or [f"ahead dropped: {reason}"]
        trial_ev = self.proc.call("commit", j.mid)
        if trial_ev:
            plan.route_trial = trial_ev
        dx, dy = float(s[0]) - px, float(s[1]) - py   # its predictions from the real start (the surprise and danger
        plan.path = [[x + dx, y + dy] for x, y in plan.path]           # checks here follow the real agent)
        plan.predictions = {h: [v[0] + dx, v[1] + dy] for h, v in plan.predictions.items()}
        self.st["ahead_used"] = self.st.get("ahead_used", 0) + 1
        self.ahead_miss = 0
        self.cur_plan[i] = plan
        self.commit.decided(i, plan, frame, j.fboxes, self.planner)
        self.on_plan(plan, s, j.why or tr or ["due"], ahead={"lead": lead, "start_off": round(off, 2),
                                                    "speed_off": round(dv, 3), "spread": round(spread, 2)})
        if trial_ev:
            self.event(trial_ev["kind"], frame, **{a: b for a, b in trial_ev.items() if a not in ("kind", "frame")})
        return []

    # ---- --async-wait react: the game waits only for danger and start (the module comment at WAIT_KINDS) ------
    def stale_prev(self, i: int, j: AheadJob, plan) -> str | None:
        """--planners N, react_take: a speculative job's process planned before an earlier job's plan was
        committed (the adoption waits behind the plan in its pipe), so its previous plan is the one committed when
        it started (j.base). A plan that keeps that one (from_prev) with other actions than the plan committed
        since, from its frame on, would play the replaced plan again (and the surprise check would keep the
        committed plan's predictions): the reason it is dropped, else None."""
        cur = self.cur_plan.get(i)
        if not getattr(plan, "from_prev", False) or cur is None or j.base is cur:
            return None
        theirs = [int(cur.actions[f - cur.frame]) for f in range(j.target, j.target + len(plan.actions))
                  if 0 <= f - cur.frame < len(cur.actions)]
        if [int(a) for a in plan.actions[:len(theirs)]] == theirs:
            return None
        return "it keeps the plan replaced since it started (other actions than the committed one)"

    # ---- --async-wait auto: react until something moves in the level (the module comment at WAIT_KINDS) ---------
    def set_wait_mode(self, mode: str):
        """The decision rule now (--async-wait, or auto's choice for this level: react / events): the trigger
        kinds decided at once, nowait, react, the committer's danger check on events (react only); misses reset."""
        self.wait_mode = mode
        self.wait_kinds = WAIT_KINDS.get(mode, ())
        self.nowait = self.async_on and mode in WAIT_KINDS
        self.react = self.async_on and mode == "react"
        self.commit.danger_always = self.react
        self.ahead_miss = 0

    def level_wait(self, name: str):
        """--async-wait auto, a level starts: react, or events when something moved in it earlier in the run."""
        d = self.dyn_levels.get(name)
        self.set_wait_mode("react" if d is None else "events")
        if d is not None:
            print(f"[wait] {name}: events (seen before: {d['why']} at frame {d['frame']} of wave {d['wave']})",
                  flush=True)

    def auto_check(self, frame: int, m: dict, lv):
        """--async-wait auto, every frame after FutureWorld.observe while the level is in react: the first moving
        part seen switches it to events (switch_wait), once."""
        if self.react:
            why = self.dynamic_seen(m, lv)
            if why:
                self.switch_wait(frame, why)

    def dynamic_seen(self, m: dict, lv) -> str | None:
        """--async-wait auto: what in this frame moves (a moving platform, a kill box more than AUTO_MOVE units from
        the frame before: FutureWorld.hazard_motion, a shot), None when nothing does."""
        pf = m.get("pf") or lv.plats
        if pf:
            return f"{len(pf)} moving platform{'s' if len(pf) != 1 else ''}"
        if m.get("sh"):
            return f"{len(m['sh'])} shot{'s' if len(m['sh']) != 1 else ''}"
        d, hid = self.fw.hazard_motion()
        if d > AUTO_MOVE:
            return f"hazard {hid} moved {d:.2f} units in a frame"
        return None

    def switch_wait(self, frame: int, why: str):
        """--async-wait auto: something moves in this level: every plan being made ahead is dropped and the level
        is decided as with --async-wait events from this frame on, for the rest of the run (later visits start in
        events); logged ("wait mode"), counted in the wave event (wait_mode_switch: the frame)."""
        lv = self.level
        self.dyn_levels[lv.name] = {"frame": int(frame), "wave": self.wave_no, "why": why}
        self.pl_sync(drop=f"wait mode events: {why} at {frame}")
        self.set_wait_mode("events")
        self.st["wait_mode_switch"] = int(frame)
        self.event("wait mode", frame, level=lv.name, mode="events", why=why)
        print(f"[wait] {lv.name}: {why} (wave {self.wave_no}, frame {frame}): events from now on in this level",
              flush=True)

    def lead_now(self) -> int:
        """The react lead L (frames) for a job started now: react_lead of the last REACT_WINDOW latencies."""
        pl = getattr(self, "planner", None)
        return react_lead(self.react_lat, pl.f if pl is not None else 4)

    def count_sync(self, t0: float):
        """A decision the game waited for (plan_now, or the in-process planner), started at perf_counter t0."""
        st = self.st
        st["sync_decisions"] = st.get("sync_decisions", 0) + 1
        st["sync_ms"] = st.get("sync_ms", 0.0) + 1000.0 * (time.perf_counter() - t0)

    def ahead_poll(self, frame: int, j: AheadJob | None = None):
        """Once per frame while a job is in flight: its reply seen for the first time (proc.ready, a non-blocking
        read) gives its latency, frame - its start frame (react_lat, this wave's react_latency). j: that job
        (--planners N), else self.ajob."""
        j = self.ajob if j is None else j
        if j is None or j.ready_at is not None or not self.proc.ready(j.mid):
            return
        j.ready_at = int(frame)
        self.react_lat.append(int(frame) - j.start)
        self.st.setdefault("react_latency", []).append(int(frame) - j.start)

    def ahead_cancel(self, frame: int, reason: str, count: str, j: AheadJob | None = None) -> AheadJob:
        """Cancel the job in flight (proc.cancel: it stops at its next model step; the one short wait left in
        react, charged to ahead_wait_ms), count it under `count`, log it; returns it. j: that job of self.jobs
        (--planners N: pool_cancel), else self.ajob."""
        if j is not None:
            self.pool_cancel(frame, reason, [j], count=count)
            return j
        j = self.ajob
        t = time.perf_counter()
        self.proc.cancel(j.mid)
        self.ajob = None
        st = self.st
        st["ahead_wait_ms"] = st.get("ahead_wait_ms", 0.0) + 1000.0 * (time.perf_counter() - t)
        st[count] = st.get(count, 0) + 1
        self.event("ahead dropped", frame, agent=j.agent, reason=reason, started=j.start, target=j.target)
        return j

    # ---- --planners N (N >= 2) with react: several plans made ahead at once (module comment at WAIT_KINDS) ------
    def pool_react(self) -> bool:
        """react with more than one planner process: route_pool / pool_fill instead of route_react's one job."""
        return getattr(self, "react", False) and getattr(self, "n_planners", 1) > 1

    def free_procs(self) -> list:
        """The planner processes with no job in flight (primary first)."""
        busy = {j.proc for j in self.jobs}
        return [k for k in range(self.n_planners) if k not in busy]

    def pool_cancel(self, frame, reason: str, jobs: list, count: str | None = None, first: str | None = None):
        """Cancel these jobs of self.jobs at once (PlannerPool.cancel_many: each process stops at its next model
        step, the wait is the slowest one's, charged to ahead_wait_ms); each is counted under `count` (the
        earliest one under `first` instead when given), a speculative one also under ahead_spec_cancelled, and
        logged (frame None: at its target)."""
        if not jobs:
            return
        t = time.perf_counter()
        self.proc.cancel_many([j.mid for j in jobs])
        gone = set(map(id, jobs))
        self.jobs = [j for j in self.jobs if id(j) not in gone]
        st = getattr(self, "st", None)
        if st is not None:
            st["ahead_wait_ms"] = st.get("ahead_wait_ms", 0.0) + 1000.0 * (time.perf_counter() - t)
        for n, j in enumerate(jobs):
            c = first if (n == 0 and first) else count
            if st is not None:
                if c:
                    st[c] = st.get(c, 0) + 1
                if j.spec:
                    st["ahead_spec_cancelled"] = st.get("ahead_spec_cancelled", 0) + 1
            self.event("ahead dropped", j.target if frame is None else frame, agent=j.agent, reason=reason,
                       started=j.start, target=j.target, planner=j.proc, spec=j.spec)

    def pool_fill(self, i: int, frame: int, s, held: int, dt: float):
        """Once per frame after the decisions (--planners N, react): no job in flight, the next timed decision
        starts as with one process (ahead_start); with jobs in flight, every free process starts a speculative
        one for max(the latest target + S, frame + L) (S = max(f, L / 2 in whole model steps)), rounded up to
        whole model steps, while the committed plan has the actions up to it."""
        if not self.jobs:
            self.ahead_start(i, frame, s, held, dt, proc=self.free_procs()[0])
        f = self.planner.f if self.planner is not None else 4
        while self.jobs:
            free = self.free_procs()
            if not free:
                return
            L = self.lead_now()
            S = max(f, (L // 2) // f * f)
            target = max(self.jobs[-1].target + S, frame + L, self.trial_end(i))
            if not self.ahead_start(i, frame, s, held, dt, target, (), proc=free[0], spec=True, exact=True):
                return

    def trial_end(self, i: int) -> int:
        """The first frame a new decision may be for: the committed plan's frame + its route trial's frames (a
        route trial plays its actions with no new decision: Plan.trial_frames, Committer.due), else 0."""
        p = self.cur_plan.get(i)
        return int(p.frame) + int(getattr(p, "trial_frames", 0) or 0) if p is not None else 0

    def pool_reconcile(self, i: int, frame: int, s, held: int, dt: float, j0: AheadJob):
        """j0's plan was just committed (cur_plan, its frame j0.target): every later job whose prefix over
        [j0.target, its target) (from its start when that is later: the frames before were checked by react_take)
        is not the new plan's actions there is cancelled and restarted on its process from the real state now,
        for the same target (whole model steps up) when it still fits at least one model step from now, else for
        frame + L. Jobs whose prefix matches stay. A route trial (the new plan's trial_frames) takes no decision
        while it plays: later jobs for a frame before its end are cancelled, not restarted, and no restart is for
        a frame before it."""
        p = self.cur_plan.get(i)
        hold = self.trial_end(i)
        inside = [jl for jl in self.jobs if jl.target < hold]
        if inside:
            self.pool_cancel(frame, f"within the route trial committed for {j0.target} (to {hold})", inside)
        bad = []
        for jl in self.jobs:
            lo = max(j0.target, jl.start)
            mine = [int(a) for a in jl.prefix[lo - jl.start:jl.target - jl.start]]
            new = [int(p.actions[f_ - p.frame]) if 0 <= f_ - p.frame < len(p.actions) else -1
                   for f_ in range(lo, jl.target)]
            if mine != new:
                bad.append(jl)
        if not bad:
            return
        self.pool_cancel(frame, f"its actions up to its frame are not those of the plan committed for {j0.target}: "
                                "restarted", bad, count="ahead_restarted")
        f = self.planner.f
        for jl in bad:
            spec = any(x.target < jl.target for x in self.jobs)
            if jl.target >= frame + f and self.ahead_start(i, frame, s, held, dt, jl.target, jl.why, proc=jl.proc,
                                                           spec=spec, exact=True):
                continue
            t2 = max(frame + self.lead_now(), hold)
            self.ahead_start(i, frame, s, held, dt, t2, jl.why, proc=jl.proc, exact=t2 != frame + self.lead_now(),
                             spec=any(x.target < t2 for x in self.jobs))

    def route_pool(self, i: int, frame: int, s, tr: list, held: int, dt: float) -> list:
        """route_react with --planners N >= 2 (several jobs in flight, self.jobs in target order): every job is
        polled (latency); a waiting kind (danger, start) or no plan drops them all (pl_sync) and decides now (on
        the primary, adopted by the others). Then, from the earliest job on:
        - at or after its target with its reply here: react_take; used: the later jobs are checked against the
          new plan (pool_reconcile) and nothing is decided now; dropped: the later jobs stay, the next is looked at;
        - an event (surprise, support, missed) with no reactive job in flight: every job is cancelled (the earliest
          counted as ahead_restarted) and a reactive one started from the real state now (pool_fill staggers
          speculative ones after it on the free processes);
        - more than REACT_LATE_MAX frames late and not ready: cancelled (ahead_late_dropped), the next looked at;
        - else the triggers are absorbed (the committed plan plays on).
        With no job left (none in flight, or all dropped), a trigger or a drop starts one for L frames on (reactive
        for an event or after a drop, as route_react); none fits: every job dropped, decided now."""
        for j in self.jobs:
            self.ahead_poll(frame, j)
        waiting = [t_ for t_ in tr if kind_of(t_) in self.wait_kinds]
        if waiting or self.cur_plan.get(i) is None:
            if self.jobs:
                self.pl_sync(drop=f"{waiting[0] if waiting else 'no plan'} at {frame} (their frames "
                                  f"{[j.target for j in self.jobs]})")
            return tr or ["start"]
        events = [t_ for t_ in tr if kind_of(t_) in REACT_EVENT_KINDS]
        reactive, why, old, had = bool(events), None, (), bool(self.jobs)
        while self.jobs:
            j = self.jobs[0]
            if frame >= j.target and j.ready_at is not None:    # its reply is here: judged on the real state
                why = self.react_take(i, frame, s, tr, j)
                if why is None:
                    self.pool_reconcile(i, frame, s, held, dt, j)
                    return []
                reactive, old = True, j.why
                continue
            if events and not any(x.reactive for x in self.jobs):
                why = (f"{events[0]} at {frame}, " + (f"before its frame {j.target}" if frame < j.target else
                                                      f"its frame {j.target}, not ready")
                       + ": restarted from the real state")
                old = j.why
                self.pool_cancel(frame, why, list(self.jobs), count=None, first="ahead_restarted")
                break
            if frame - j.target > REACT_LATE_MAX:
                why = f"late: not ready {frame - j.target} frames after its frame {j.target}"
                if j.ready_at is None:                  # its age: a lower bound of its latency (else L never grows)
                    self.react_lat.append(int(frame) - j.start)
                    self.st.setdefault("react_latency", []).append(int(frame) - j.start)
                self.ahead_cancel(frame, why, "ahead_late_dropped", j)
                reactive, old = reactive or j.reactive, j.why
                continue
            if tr:                                      # the jobs answer it: the committed plan plays on
                self.st["ahead_absorbed"] = self.st.get("ahead_absorbed", 0) + 1
            return []
        if not had and not tr:
            return []
        if self.ahead_start(i, frame, s, held, dt, frame + self.lead_now(), list(tr) or list(old), reactive=reactive,
                            proc=self.free_procs()[0]):
            return []
        return tr or [f"ahead dropped: {why}"]          # no job fits: decided now (no job in flight)

    def route_react(self, i: int, frame: int, s, tr: list, held: int, dt: float) -> list:
        """route_triggers with --async-wait react: [] (the committed plan plays on) unless a waiting kind fired
        (danger, start), the agent has no plan, or no job fits in the committed plan's actions left; then the
        triggers to decide on now (a job in flight is dropped). Otherwise, with a job in flight:
        - at or after its target with its reply here: react_take (used, its start judged on the real state, even
          with an event this frame; or dropped and a reactive job started);
        - an event (surprise, support, missed) while it is routine (and not taken above): it is cancelled and a
          reactive one started from the real state now (ahead_restarted); while it is reactive: absorbed;
        - any other trigger is absorbed (ahead_absorbed); not ready at its target, the committed plan plays on,
          and more than REACT_LATE_MAX frames late it is cancelled and restarted (ahead_late_dropped).
        With no job in flight, a trigger starts one (reactive for an event) for L frames on (lead_now)."""
        j = self.ajob if self.ajob is not None and self.ajob.agent == i else None
        if j is not None:
            self.ahead_poll(frame)
        waiting = [t_ for t_ in tr if kind_of(t_) in self.wait_kinds]
        if waiting or self.cur_plan.get(i) is None:
            if j is not None:
                self.pl_sync(drop=f"{waiting[0] if waiting else 'no plan'} at {frame} (its frame {j.target})")
            return tr or ["start"]
        events = [t_ for t_ in tr if kind_of(t_) in REACT_EVENT_KINDS]
        reactive, why, old = bool(events), None, ()     # why: the job just dropped here (a new one starts now)
        if j is not None:
            if frame >= j.target and j.ready_at is not None:    # its reply is here: judged on the real state
                why = self.react_take(i, frame, s, tr)
                if why is None:
                    return []
                reactive = True
            elif events and not j.reactive:
                why = (f"{events[0]} at {frame}, " + (f"before its frame {j.target}" if frame < j.target else
                                                      f"its frame {j.target}, not ready")
                       + ": restarted from the real state")
                self.ahead_cancel(frame, why, "ahead_restarted")
            elif frame - j.target > REACT_LATE_MAX:
                why = f"late: not ready {frame - j.target} frames after its frame {j.target}"
                if j.ready_at is None:                  # its age: a lower bound of its latency (else L never grows)
                    self.react_lat.append(int(frame) - j.start)
                    self.st.setdefault("react_latency", []).append(int(frame) - j.start)
                self.ahead_cancel(frame, why, "ahead_late_dropped")
                reactive = reactive or j.reactive
            else:                                       # the job answers it: the committed plan plays on
                if tr:
                    self.st["ahead_absorbed"] = self.st.get("ahead_absorbed", 0) + 1
                return []
            old = j.why
        elif not tr:
            return []
        if self.ahead_start(i, frame, s, held, dt, frame + self.lead_now(), list(tr) or list(old), reactive=reactive):
            return []
        return tr or [f"ahead dropped: {why}"]          # no job fits: decided now

    def react_take(self, i: int, frame: int, s, tr: list, j: AheadJob | None = None) -> str | None:
        """--async-wait react: the job's reply is here (ready) at frame = its target + k (k >= 0). Its plan is
        committed when it was made, for k > 0 the actions played on frames target .. target + k - 1 are its own
        first k, and the real state is within the tube (lead + k frames) and ASYNC_DV units / frame of its
        predicted state at k (k = 0: the predicted start; else its path at k, the speed from the path around k);
        its path and predictions are moved onto the real state, its frame stays the target (the agent plays its
        action k now). Returns None then, else the reason it was dropped (discarded, ahead_dropped).
        j: that job of self.jobs (--planners N), else self.ajob; such a job is also dropped when its plan keeps
        its process's previous plan (from_prev) while an earlier job's plan, with other actions from its frame
        on, was committed after it started (it would play the replaced plan again)."""
        pool = j is not None
        if pool:
            self.jobs.remove(j)
        else:
            j = self.ajob
            self.ajob = None
        ok, res = self.proc.result(j.mid)               # ready: no wait
        k = frame - j.target
        lead = j.target - j.start
        tube = max(self.commit.tube_factor * self.commit.tube.at(lead + k), MIN_TUBE)
        uf = self.dt_ms / 1000.0                        # units / s -> units / frame
        off = dv = None
        if not ok:
            reason = f"planning error: {str(res).splitlines()[0]}"
        else:
            plan, px, py, spread, pvx, pvy = res
            reason = self.stale_prev(i, j, plan) if pool else None
            if reason is None and k == 0:
                qx, qy, qvx, qvy = px, py, pvx * uf, pvy * uf
            elif reason is None:
                p = self.cur_plan.get(i)
                played = [int(p.actions[f - p.frame]) if p is not None and 0 <= f - p.frame < len(p.actions) else 0
                          for f in range(j.target, frame)]
                path = plan.path
                if played != [int(a) for a in plan.actions[:k]]:
                    reason = f"{k} frames late: the actions played since its frame are not its first {k}"
                elif k + 1 >= len(path):
                    reason = f"{k} frames late: past its path"
                else:
                    qx, qy = float(path[k][0]), float(path[k][1])
                    qvx = (float(path[k + 1][0]) - float(path[k - 1][0])) / 2.0
                    qvy = (float(path[k + 1][1]) - float(path[k - 1][1])) / 2.0
            if reason is None:
                off = math.hypot(float(s[0]) - qx, float(s[1]) - qy)
                dv = math.hypot(float(s[2]) * uf - qvx, float(s[3]) * uf - qvy)
                at = "start" if k == 0 else f"state {k} frames late"
                reason = (f"{at} off by {off:.1f} > {tube:.1f} units" if off > tube else
                          f"{at} speed off by {dv:.2f} > {ASYNC_DV:g} units/frame" if dv > ASYNC_DV else None)
        if reason is not None:
            if ok:
                self.proc.send("discard", j.mid)
            self.st["ahead_dropped"] = self.st.get("ahead_dropped", 0) + 1
            self.ahead_miss += 1                        # counted and logged; never forces a decision now (react)
            self.event("ahead dropped", frame, agent=i, reason=reason, started=j.start, target=j.target, late=k,
                       **({"planner": j.proc, "spec": j.spec} if pool else {}))
            return reason
        trial_ev = self.proc.call("commit", j.mid)      # --planners N: the other processes adopt it (PlannerPool)
        if trial_ev:
            plan.route_trial = trial_ev
        dx, dy = float(s[0]) - qx, float(s[1]) - qy   # its predictions from the real state (as ahead_take)
        plan.path = [[x + dx, y + dy] for x, y in plan.path]
        plan.predictions = {h: [v[0] + dx, v[1] + dy] for h, v in plan.predictions.items()}
        self.st["ahead_used"] = self.st.get("ahead_used", 0) + 1
        if k:
            self.st["ahead_late_used"] = self.st.get("ahead_late_used", 0) + 1
        if pool and j.spec:
            self.st["ahead_spec_used"] = self.st.get("ahead_spec_used", 0) + 1
        self.ahead_miss = 0
        self.cur_plan[i] = plan                         # plan.frame = j.target: its action k plays now
        self.commit.decided(i, plan, j.target, j.fboxes, self.planner)     # fboxes: index 0 = the target
        self.on_plan(plan, s, j.why or tr or ["due"], ahead={"lead": lead, "late": k, "start_off": round(off, 2),
                                                             "speed_off": round(dv, 3), "spread": round(spread, 2),
                                                             **({"planner": j.proc, "spec": j.spec} if pool else {})})
        if trial_ev:
            self.event(trial_ev["kind"], frame, **{a: b for a, b in trial_ev.items() if a not in ("kind", "frame")})
        return None

    def observe_alive(self, level: str, frame: int, alive_xy: list):
        """Real agents alive at alive_xy [(x, y), ...] this wave frame: DeathMemory.observe_alive. When that
        refutes a static place, an event, and the memory is saved (not on a transient level, nor with the
        memory off)."""
        xy = np.asarray(alive_xy, np.float64).reshape(-1, 2)
        if self.deaths.observe_alive(level, frame, xy[:, 0], xy[:, 1]):
            self.event("death place refuted", frame, deaths=len(self.deaths))
            if self.memory_on() and level not in self.deaths.transient:
                self.save_deaths()

    def save_deaths(self) -> bool:
        """The death memory to --death-memory (its transient levels left out); False when it could not be."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.deaths_path)), exist_ok=True)
            self.deaths.save(self.deaths_path)
            return True
        except OSError as e:
            print(f"[deaths] could not save {self.deaths_path}: {e}", flush=True)
            return False

    def on_death(self, i, frame, xy0, xy1, keys, rec, extra_boxes, estimated: bool):
        """A planned agent died at wave frame `frame` at xy1 (from xy0 the frame before): does the geometry
        explain it (the move against the death frame's hazards and shots, or leaving the level)? If not, the
        place goes into the death memory and is saved (held-out levels: not in the first life; with --online
        after it, kept for this run only: DeathMemory.transient)."""
        self.pl_sync(drop="the agent died")
        self.tel_force = True
        lv = self.level
        x1, y1 = xy1
        x0, y0 = xy0 if xy0 is not None else xy1
        if getattr(lv, "portals", None) and math.hypot(x1 - x0, y1 - y0) > TELEPORT_JUMP:
            x0, y0 = x1, y1                          # came out of a game portal: the death frame's end only
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
        if not explained and self.memory_on():
            self.deaths.add(lv.name, frame, x1, y1)
            # practice on a held-out level: this run only, nothing to save
            added = lv.name in self.deaths.transient or self.save_deaths()
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
        self.tel_force = True
        planned = i < self.n_plan
        search = self.search is not None
        human = self.human is not None
        source = "planned" if planned else ("demo" if self.demo is not None else ("search" if search else
                                                                                 "human" if human else "explore"))
        self.st["finished" if planned or (search and not human) else "ex_finished"] += 1
        if search and not human:
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
        self.fin.setdefault(source, []).append(seconds)
        if human:
            par = a_plus_time(lv)
            ok = par is not None and seconds <= par
            print(f"[human] {lv} wave {self.wave_no}: finished in {seconds:.2f} s"
                  + (f" (A+ {par:g} s: {'UNDER A+, kept as an A+ example' if ok else 'over A+'})" if par is not None
                     else " (no A+ time known)") + ("" if self.lock is True else "; the time step is NOT locked"),
                  flush=True)
            self.event("finish", int(finish_frame), agent=i, x=x, y=y, frames=frames, seconds=seconds, source=source,
                       dt_ms=rec["dt_ms"], lock=self.lock, a_plus_seconds=par, a_plus=ok)
        if planned and self.planner is not None:
            self.pl_sync(drop="the agent finished")
            r = (self.proc.call("finished", i, int(finish_frame)) if self.proc is not None
                 else self.planner.finished(i, int(finish_frame)))
            if r["best_route"]:                     # the fastest real finish of the run on this level so far
                self.event("best route", int(finish_frame), agent=int(i), frames=frames, seconds=seconds)
            rb = r.get("route_book")
            if rb:                                  # every finish joins the route book: its composite rebuilt
                self.event("route book", int(finish_frame), agent=int(i), **rb)
                print(f"[route book] {lv}: {rb['records']} finished waves"
                      + (f" (+{rb['duplicates']} duplicates)" if rb.get("duplicates") else "")
                      + f", {rb['portals']} portals, sum of best {rb['finish']} f (best real {rb['best_real']} f), "
                      f"{len(rb['segments'])} segments [{rb['build_ms'] / 1000.0:.2f} s]", flush=True)
            if r["failures_cleared"]:
                self.event("forgiven", int(finish_frame), agent=int(i), failures=r["failures_cleared"])
            ev = r.get("route_trial")
            if ev:
                self.event(ev["kind"], int(finish_frame), **{a: b for a, b in ev.items() if a not in ("kind", "frame")})
        new = False
        if self.lock is True and lv is not None and self.demo is None and not human:
            new = self.best.offer(lv, source, {"seconds": seconds, "frames": frames, "run": self.args.name,
                                               "wave": self.wave_no, "agent": int(i), "date": rec["time"]})
        if planned:
            self.new_best |= new
            if new:
                self.post_best(lv)
                if self.learn and self.wm is not None and not self.protected:
                    self.add_self_demo(i, int(finish_frame), lv, seconds)
            best = self.best.best(lv, "planned") if lv is not None else None
            par = a_plus_time(lv)                      # the game's A+ time: the target (levels_smb)
            self.event("finish", int(finish_frame), agent=i, x=x, y=y, frames=frames, seconds=seconds,
                       dt_ms=rec["dt_ms"], lock=self.lock, best=float(best["seconds"]) if best else None,
                       new_best=new, a_plus_seconds=par, a_plus=None if par is None else seconds <= par)
            if par is not None:
                print(f"[finish] {lv} wave {self.wave_no}: {seconds:.2f} s, A+ {par:g} s: "
                      f"{'yes' if seconds <= par else f'no (+{seconds - par:.2f} s)'}", flush=True)
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

    def tel_due(self, frame: int) -> bool:
        """This frame's message is sent: a decision, death or finish in it (tel_force), the first frame of a wave
        or level, or tel_every frames since the last one."""
        last = self.tel_last
        return (self.tel_force or last is None or last[:2] != (self.level.name, self.wave_no) or
                not 0 < frame - last[2] < self.tel_every)

    def skip_frame(self, m, frame, states, act_idx, boxes, kept, fboxes, dt):
        """A frame whose message is not sent: the planned agents' positions go into the next one's trail (the
        viewer compares predictions with them at their own frames), the frame is kept to be sent at the wave's end
        if it is the last, and the hazards' future runs its period search as send_frame's future_inputs would have
        (mpc.future.FutureWorld.periods: it keeps state, so it runs on the same frames as before)."""
        self.tel_trail.extend([frame, i, float(s[0]), float(s[1])] for i, s in enumerate(states) if i < self.n_plan)
        self.tel_pending = (m, frame, states, act_idx, boxes, kept, fboxes, dt)
        fw = self.fw
        if (fboxes is None or len(fboxes) <= max(T.FUTURE_AT)) and fw.last is not None and fw.last[0] == frame:
            fw.periods()
        if time.time() - self.level_sent >= LEVEL_EVERY_S:
            self.send_level()

    def send_frame(self, m, frame, states, act_idx, boxes, kept, fboxes, dt):
        lv = self.level
        same = self.tel_last is not None and self.tel_last[:2] == (lv.name, self.wave_no)
        trail = self.tel_trail if same else []          # frames skipped since the last message, if of this wave
        self.tel_trail, self.tel_pending, self.tel_force = [], None, False
        self.tel_last = (lv.name, self.wave_no, frame)
        agents = [{"i": i, "role": "planner" if i < self.n_plan else "explore", "x": float(s[0]), "y": float(s[1]),
                   "vx": float(s[2]), "vy": float(s[3]), "status": int(s[6]), "flags": int(s[4]) & 0xffffffff,
                   "state": int(s[5]), "keys": int(s[7]) if len(s) > 7 else 0, "action": int(act_idx[i])}
                  for i, s in enumerate(states)]
        fb_now = S.frame_boxes(m.get("hz", []), lv.hz_keep, m.get("sh", []))   # the game's own shapes
        now = T.box_rows(fb_now.boxes, fb_now.kind, fb_now.shapes())
        future = {}
        try:
            if fboxes is None or len(fboxes) <= max(T.FUTURE_AT):
                fboxes = future_inputs(self.fw, frame, max(T.FUTURE_AT), dt, lv.name, self.deaths)[1]
            future = {str(k): T.box_rows(fboxes[k].boxes, fboxes[k].kind, fboxes[k].shapes(), fboxes[k].grows())
                      for k in T.FUTURE_AT}
        except Exception:
            future = {}
        mem = self.deaths.boxes(lv.name, frame)
        self.tel.send(T.frame_message(lv.name, self.wave_no, frame, agents, now, future,
                                      T.box_rows(mem, S.KIND_MEMORY), T.platform_rows(lv.plats), trail))
        if time.time() - self.level_sent >= LEVEL_EVERY_S:
            self.send_level()

    def on_wave_end(self, m: dict):
        if self.tel is not None and self.tel_pending is not None and self.level is not None:
            self.send_frame(*self.tel_pending)     # the wave's last frame was not sent: the viewer gets it now
        self.tel_pending, self.tel_trail = None, []
        self.pl_sync(drop="the wave ended")
        lv = self.level
        if self.rec is not None:
            self.rec.wave_end(m["wave"], m, held=list(self.held))
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
            elif self.search is not None and self.human is None:
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
        if self.planner is None:
            surprise_err = err_f = None
        elif self.proc is not None:
            surprise_err, err_f = self.proc.call("surprise")
        else:
            surprise_err, err_f = self.planner.surprise.err, self.planner.surprise.errors()
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
        fut = self.fw.new_wave() if self.evaluation is None else {
            "frames": len(self.fw.cur), "recording": 0, "trusted": None, "mismatch": None, "extended": 0}
        react = react_summary(st)
        if getattr(self, "react", False) or st.get("wait_mode_switch") is not None:
            print(react_text(st, react), flush=True)
        self.event("wave", frame, **(cost or {}), why=m.get("why"), frames=frame, future=fut,
                   **{k: v for k, v in st.items() if k not in ("ms", "gaps", "check_s", "react_leads", "react_latency")},
                   **react, planners=getattr(self, "n_planners", 1),
                   wait_mode=getattr(self, "wait_mode", None) if getattr(self, "async_on", False) else None,
                   ms_median=float(np.median(ms)) if ms is not None else None,
                   ms_95=float(np.percentile(ms, 95)) if ms is not None else None, err_followed=err_f, **com,
                   tube=self.commit.tube.summary() if self.commit.on else None,
                   finish_s=fin_p, explore_finish_s=fin_x, best_s=float(best_p["seconds"]) if best_p else None,
                   new_best=self.new_best, lock=self.lock,
                   **({"replay_route": bool(srec.get("replay")), "search_finish_s": min(self.fin["search"])
                       if self.fin["search"] else None} if srec is not None else {}))
        if self.evaluation is not None:
            self.evaluation.finish(m, self.lock, metrics={"decisions": st["decisions"],
                                   "planning_ms_median": float(np.median(ms)) if ms is not None else None,
                                   "planning_ms_95": float(np.percentile(ms, 95)) if ms is not None else None,
                                   "cost": cost})
            self.stopping = True
            self.flush()
            return                              # never promote this life's recording or start another plan
        if lv is not None:                          # a full wave of this level (its deaths above: still the old rule)
            self.level_waves[lv.name] = self.level_waves.get(lv.name, 0) + 1
            if self.practice_on(lv.name):           # a held-out level's first life is over: practice from the next
                self.start_practice(lv.name)
        if self.proc is not None:                   # the process holds the move memory: it saves it
            try:
                self.proc.call("moves_save")
            except RuntimeError as e:
                print(f"[moves] could not save: {e}", flush=True)
        elif self.moves is not None and self.moves.dirty:
            try:
                self.moves.save()
            except OSError as e:
                print(f"[moves] could not save {self.moves.path}: {e}", flush=True)
        if self.deaths.dirty and self.memory_on():  # the wave's visits of remembered deaths (DeathMemory.visits)
            self.save_deaths()
        if self.planner is not None:
            if self.commit.on:
                self.commit.tube.add(surprise_err)  # the run's followed errors, for the tube
            new_wave_ev = self.proc.call("new_wave") if self.proc is not None else self.planner.new_wave()
            for ev in new_wave_ev:                  # also clears the surprise check's predictions and errors
                self.event(ev["kind"], ev.get("end_frame", ev.get("frame")),
                           **{a: b for a, b in ev.items() if a not in ("kind", "frame")})
        if lv is not None and self.field_dirty and self.rebuilds < FIELD_REBUILDS:
            self.rebuilds += 1
            self.build_field(f"rebuild {self.rebuilds}/{FIELD_REBUILDS}: moving hazards / kill box / platforms")
        if self.learn:
            self.load_live()                        # between waves only: a newer published model
        if self.online is not None:
            self.load_online()                      # between waves only: the online learner's newer model
        self.new_wave(len(status) if status else len(self.alive))
        self.post_overlay(0, force=True)
        self.flush()
        self.cost.reset()

    # ---- frida -------------------------------------------------------------------------------------------------
    def on_message(self, msg, data):
        if msg["type"] == "error":
            if self.evaluation is not None:
                self.evaluation.invalidate("game script error: " + str(msg.get("description")))
                self.stopping = True
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
                if self.evaluation is not None and self.evaluation.info_seen:
                    self.evaluation.interrupt("level changed before the attempt completed")
                    self.stopping = True
                print(f"[game] level {p['name']}", flush=True)
                if self.wm is not None:
                    self.wm.cut("level change")     # the wave in progress is kept as a partial shard
                self.pl_sync(drop="level change")
                self.level = None
                self.cur_plan = {}
                self.commit.reset()
                if self.search is not None:
                    self.search.abort_wave()
            elif ev == "save":
                if self.evaluation is not None:
                    self.evaluation.interrupt("manual level finish requested")
                    self.stopping = True
                else:
                    self.on_save(p.get("level"))
            elif ev == "stopped":
                self.stopped.set()
            elif ev == "error":
                if self.evaluation is not None:
                    self.evaluation.invalidate("game error: " + str(p.get("msg")))
                    self.stopping = True
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
            if self.evaluation is not None:
                self.evaluation.invalidate("runner failed while handling " + str(ev))
                self.stopping = True
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
        self.pl_sync(drop="model swap")
        p = os.path.join(self.args.live_dir, "wm_model.pt")
        try:
            if not os.path.exists(p) or os.path.getmtime(p) <= self.live_mtime:
                return
            mt = os.path.getmtime(p)
            from wm.model import WorldModel
            m = WorldModel.load(p).eval()
            if getattr(self.args, "dataset_manifest", None):
                from wm.dataset import Dataset
                Dataset.load(self.args.dataset_manifest).require_model(m)
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            print(f"[learn] could not load {p}: {e}", flush=True)
            return
        self.live_mtime = mt
        if not self.compatible(m, p, "learn"):
            return
        info = (m.info or {}).get("learner", {})
        old = (info.get("parent_errors") or {}).get("16", [None])[0]
        new = (info.get("errors") or {}).get("16", [None])[0]
        self.swap_model(m, p)
        print(f"[learn] model v{info.get('version', '?')} loaded (+16 error {old} -> {new})", flush=True)
        self.event("learn model", None, version=info.get("version"), err16=[old, new], path=p)

    @property
    def rec(self):
        """The collector this level's play goes to: wm_data (not held out), wm_heldout (held out, --online), or None
        (no recording: --wm-collect off, a first-life evaluation, or a held-out level without --online)."""
        return self.rec_for(self.protected)

    def rec_for(self, protected: bool):
        if self.evaluation is not None:
            return None
        return self.wm_held if protected else self.wm

    def swap_model(self, m, p: str):
        """Between waves: the planner's world model (here and in the planner process) becomes m (loaded from p)."""
        self.model, self.model_path = m, p
        if self.planner is not None:
            self.planner.model = m
        if self.proc is not None:
            self.proc.call("set_model", p)

    def compatible(self, m, p: str, tag: str) -> bool:
        if m.step != self.model.step or len(m.members) != len(self.model.members):
            print(f"[{tag}] {p}: step {m.step} / {len(m.members)} members, the planner has {self.model.step} / "
                  f"{len(self.model.members)}: not loaded", flush=True)
            return False
        from wm.features import model_layout
        try:
            same_inputs = model_layout(m) == model_layout(self.model)     # the planner's tile encoding stays
        except ValueError as e:
            print(f"[{tag}] {p}: {e}: not loaded", flush=True)
            return False
        if not same_inputs:
            print(f"[{tag}] {p}: feature protocol {model_layout(m).protocol}, the planner uses "
                  f"{model_layout(self.model).protocol}: not loaded", flush=True)
        return same_inputs

    def start_online(self):
        """wm.online as a child (below-normal priority): it learns from this run's waves (held-out levels too) and
        publishes runs/<name>/online/wm_model.pt when a model is better on the newest wave and no worse on the base
        data's replay sample. The base model is never written."""
        import subprocess
        out_dir = os.path.join(self.run_dir, "online")
        os.makedirs(out_dir, exist_ok=True)
        data = [self.wm.dir] + ([self.wm_held.dir] if self.wm_held is not None else [])
        log = open(os.path.join(out_dir, "online_output.txt"), "a", encoding="utf-8")
        flags = 0x4000 if sys.platform == "win32" else 0          # BELOW_NORMAL_PRIORITY_CLASS
        self.online = subprocess.Popen([sys.executable, "-m", "wm.online", "--init", self.model_path, "--replay",
                                        self.args.online_replay, "--out", out_dir, "--data", *data],
                                       cwd=HERE, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        log.close()
        self.online_path = os.path.join(out_dir, "wm_model.pt")
        print(f"[online] learner started (pid {self.online.pid}; log {os.path.join(out_dir, 'online.log')})", flush=True)

    def load_online(self):
        """Between waves: the online learner's published model when it is newer than the one in use."""
        p = self.online_path
        try:
            if not os.path.exists(p) or os.path.getmtime(p) <= self.online_mtime:
                return
            mt = os.path.getmtime(p)
            from wm.model import WorldModel
            m = WorldModel.load(p).eval()
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            print(f"[online] could not load {p}: {e}", flush=True)
            return
        self.online_mtime = mt
        if not self.compatible(m, p, "online"):
            return
        self.pl_sync(drop="model swap")
        self.swap_model(m, p)
        info = (m.info or {}).get("online", {})
        print(f"[online] model v{info.get('version', '?')} loaded (losses {info.get('losses')})", flush=True)
        self.event("online model", None, version=info.get("version"), losses=info.get("losses"), path=p)

    def stop_online(self):
        if self.online is None:
            return
        try:
            self.online.terminate()
            self.online.wait(5)
        except Exception:
            try:
                self.online.kill()
            except Exception:
                pass
        print("[online] learner stopped", flush=True)
        self.online = None

    def start_learner(self):
        """wm.learner as a child (below-normal priority), from the default model unless --live-dir has a newer one."""
        import subprocess
        validate_learning_args(self.args)
        live = os.path.join(self.args.live_dir, "wm_model.pt")
        newer = os.path.exists(live) and os.path.getmtime(live) > os.path.getmtime(self.model_path)
        init = live if newer else self.model_path
        if self.args.dataset_manifest:
            from wm.dataset import Dataset, load_model
            Dataset.load(self.args.dataset_manifest).require_model(load_model(init)[0])
        os.makedirs(self.args.live_dir, exist_ok=True)
        out = open(os.path.join(self.args.live_dir, "learner_output.txt"), "a", encoding="utf-8")
        flags = 0x4000 if sys.platform == "win32" else 0          # BELOW_NORMAL_PRIORITY_CLASS
        split = ["--manifest", self.args.dataset_manifest] if self.args.dataset_manifest else ["--legacy-split"]
        self.learner = subprocess.Popen([sys.executable, "-m", "wm.learner", "--init", init, "--out",
                                         self.args.live_dir, "--self-demos", self.args.self_demos, *split],
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
        if self.proc is not None:
            try:
                self.pl_sync(drop="run stopped")
            except (RuntimeError, OSError, EOFError):
                pass
            self.proc.close()
        self.stop_learner()
        self.stop_online()
        if self.evaluation is not None:
            self.evaluation.interrupt("run stopped before the first life completed")
        for c in (self.wm, self.wm_held):
            if c is None:
                continue
            try:
                c.cut("stop")                       # Ctrl+C / STOP: the wave in progress is kept (partial)
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
        validate_learning_args(self.args)
        if self.learn and self.args.dataset_manifest:
            from wm.dataset import Dataset
            Dataset.load(self.args.dataset_manifest).require_model(self.model)
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
        if getattr(self.args, "online", False) and self.model is not None and self.wm is not None \
                and self.evaluation is None:
            self.start_online()
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
        if self.human is not None:
            from plan.human import KEYS
            print("[human] you play agent 0: keyboard arrows or A / D, jump Space / Up / W, run Shift / Ctrl (never "
                  "S: it finishes the level); gamepad d-pad or stick, jump A / B, run X / Y / bumpers / triggers. "
                  f"Input counts only while the game is in front. Gamepad {'found' if self.human.xinput else 'API missing'}.",
                  flush=True)
        print(f"attached: {self.n_plan} planned + {self.n_explore} explore agents"
              + (f" + {self.n_total} search agents (plan/search.py)" if self.search is not None and self.human is None
                 else "")
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
    ap.add_argument("--eval-first", metavar="LEVEL", default=None,
                    help="one frozen-model life on this level, with no persistent memories, training or replay; "
                         "auto-stops after success/death/timeout; writes run.json and evaluation.json")
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
    ap.add_argument("--async-plan", action="store_true",
                    help="plan the next timed decision (interval, move end, short) in a background thread while the "
                         "game plays, from the state the committed plan is imagined to reach; adopted when the real "
                         "state is within the surprise tube of it and no other trigger fired, else decided as before "
                         "(one planned agent, no explore agents, not with --eval-first / --search / --demo)")
    ap.add_argument("--death-min", type=float, default=DEATH_MIN_FRAMES,
                    help="frames one death costs at least in a plan's estimate (it costs the progress it throws "
                         f"away when that is more; default {DEATH_MIN_FRAMES:g}). Lower: bolder near hazards; 500 was "
                         "the old value")
    ap.add_argument("--risk-members", choices=("worst", "second", "mean"), default="second",
                    help="a plan's soft risk over the world model's members: second (default) = the second worst "
                         "member's, so one member alone grazing a hazard does not cost a whole death; worst = the "
                         "old, most cautious rule; mean = their average (boldest)")
    ap.add_argument("--corner-risk", choices=("on", "off"), default="on",
                    help="soft risk for passing a ledge's corner closer than the world model's error when missing it "
                         "slides into a hazard (plan.planner.corner_risk; 3-1's salt pit). off: the old planner")
    ap.add_argument("--stall-free", type=float, default=STALL_FREE,
                    help="frames an agent may go without progress (its time to the goal not falling) before plans that "
                         f"make none are charged the frames already spent so (default {STALL_FREE}): the longer it has "
                         "stood, the more risk the way on may carry. -1: off (it may wait for ever)")
    ap.add_argument("--reflex", choices=("veto", "any-risk"), default="veto",
                    help="when the reflex move replaces the chosen plan: veto (default) = only when the plan touches a "
                         "hazard within 16 frames; a near pass in those frames costs soft risk in the estimate "
                         "instead. any-risk = the old rule: any hazard within the margin in the first 16 frames "
                         "(stalls in narrow passages with hazards on the walls)")
    ap.add_argument("--veto-members", type=int, default=1,
                    help="world-model members that must touch a hazard within the first 16 frames for a plan to be "
                         "vetoed (default 1: any of the 5; 3: a majority). With more than 1, a touch by fewer members "
                         "is soft risk (their share). Bolder near hazards: tight passes are no longer forbidden")
    ap.add_argument("--async-wait", choices=("events", "danger", "all", "react", "auto"), default="events",
                    help="--async-plan: events (default) = the game waits for a danger, surprise or support decision "
                         "(and the first one), made at once from the real state (a plan being made ahead is dropped); "
                         "an interval, move end or short decision is made in the background for --async-lead frames "
                         "later while the committed plan plays on. danger = the old rule: only danger waits (surprise "
                         "and support are planned ahead or absorbed). all = every decision not planned ahead is "
                         "waited for. react (new, not the default) = the game waits only for danger and start (the "
                         "first decision, tiles changed, no plan) or when the committed plan has no whole model step "
                         "left: every other decision is planned in the background from the committed plan's next L "
                         f"frames, L adapting to the planner's 90%% latency ({REACT_LEAD_MIN}-{REACT_LEAD_MAX} frames, "
                         f"{REACT_LEAD_INIT} at first; --async-lead is not used); a surprise, support or missed "
                         "trigger restarts a plan being made for a timed decision from the real state; a plan not "
                         f"ready at its frame is never waited for (used up to {REACT_LATE_MAX} frames late when the "
                         "actions played meanwhile are its own and the real state is on its path). auto (new, not "
                         "the default) = react on each level until the agent first sees something move there (a "
                         f"moving platform, a hazard moving more than {AUTO_MOVE:g} units in a frame, a shot), then "
                         "events for the rest of the run on that level (later visits start in events)")
    ap.add_argument("--planners", type=int, default=1,
                    help="--async-plan with --async-wait react (or auto while a level is in react): planner "
                         "processes (default 1). With N >= 2 a free process plans the next decision speculatively "
                         "while another is being made (for about half the lead later, from the committed plan's "
                         "actions); a commit restarts the later plans whose actions up to their frame changed. "
                         "Every process gets every state call (the same memories); 2 torch threads each "
                         "(--threads)")
    ap.add_argument("--async-lead", type=int, default=ASYNC_LEAD,
                    help=f"--async-plan: frames ahead a decision starts (whole model steps; default {ASYNC_LEAD}; "
                         "not with --async-wait react, whose lead adapts)")
    ap.add_argument("--route-trial-gain", type=float, default=TRIAL_GAIN,
                    help="where the run's best route would be played, an unvetoed plan whose estimate is better by "
                         "more than this many frames is played once with its exact actions (one agent per wave, "
                         "never the same plan from the same state again, nor a failed move family near where it "
                         "failed, nor riskier than the route; plan.planner.RouteTrials); 0 = never "
                         f"(default {TRIAL_GAIN:g})")
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
    split = ap.add_mutually_exclusive_group()
    split.add_argument("--dataset-manifest", help="--learn: immutable world-model dataset snapshot")
    split.add_argument("--legacy-split", action="store_true", help="--learn: explicit old global-modulo compatibility")
    ap.add_argument("--live-dir", default=LIVE_DIR, help="--learn: the learner's folder (default runs/wm_live)")
    ap.add_argument("--online", action="store_true",
                    help="learn between waves: wm.online fine-tunes a copy of the world model on this run's own waves "
                         "(held-out levels too, recorded to runs/<name>/wm_heldout, never used offline; after a "
                         "held-out level's first life the move and death memory are on there for this run, never "
                         "saved) mixed with a "
                         "replay of the base training data, and the planner takes it between waves when it predicts "
                         "the newest wave better and the replay no worse (runs/<name>/online). Turns on --wm-collect")
    ap.add_argument("--online-replay", default=os.path.join(HERE, "runs", "experiments", "human_v3", "replay.npz"),
                    help="--online: the base data's replay sample (runs/experiments/human_v3/make_replay.py)")
    ap.add_argument("--self-demos", default=SELF_DEMOS, help="--learn: self-demo list (default runs/demos/self.json)")
    ap.add_argument("--no-overlay", action="store_true", help="do not draw the plan in the game")
    ap.add_argument("--telemetry", default="127.0.0.1:47800",
                    help="host:port for the live telemetry (viz/telemetry.py), or off (default 127.0.0.1:47800)")
    ap.add_argument("--telemetry-every", type=int, default=TELEMETRY_EVERY,
                    help=f"frames between two live frame messages (default {TELEMETRY_EVERY}: 20 Hz; 1 = every "
                         "frame); a frame with a decision, death or finish and a wave's first and last always go")
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
    ap.add_argument("--human", action="store_true",
                    help="you play: one agent driven by your keyboard / gamepad (plan/human.py), no planner, no model; "
                         "with --wm-collect your play is recorded as the agents' is (source 'human'); finishes under "
                         "the level's A+ time are reported. Not with --demo / --search / --learn")
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


def validate_learning_args(args):
    if getattr(args, "learn", False) and not (getattr(args, "dataset_manifest", None) or
                                             getattr(args, "legacy_split", False)):
        raise SystemExit("--learn requires --dataset-manifest FILE or explicit --legacy-split before attaching")


def main():
    args = build_parser().parse_args()
    from plan.first_eval import validate_args
    validate_args(args)
    validate_learning_args(args)
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
    if args.online:
        if args.learn or args.demo or args.demo_level or args.search or args.human:
            raise SystemExit("--online learns from the planned agents' own play: not with --learn / --demo / "
                             "--search / --human")
        if not os.path.exists(args.online_replay):
            raise SystemExit(f"--online needs its replay sample {args.online_replay} "
                             "(python runs/experiments/human_v3/make_replay.py)")
        args.wm_collect = True                      # the online learner trains on this run's waves
    if args.human:
        args.name = args.name or time.strftime("human_%Y%m%d-%H%M%S")
    args.name = args.name or time.strftime("first_%Y%m%d-%H%M%S" if args.eval_first else "plan_%Y%m%d-%H%M%S")
    Runner(args).run()


if __name__ == "__main__":
    main()
