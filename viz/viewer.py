"""The planning agent's reasoning, visible: a separate-process pygame viewer of agent.py's decisions.

    python smbrl/viz/viewer.py                                  live: listens for agent.py's UDP telemetry
    python smbrl/viz/viewer.py --listen 127.0.0.1:47800         (the same, explicit address)
    python smbrl/viz/viewer.py --replay smbrl/runs/plan_20261103-120000   replay of a run's logs
    python smbrl/viz/viewer.py --replay latest                  the newest runs/plan_* folder

Live: a non-blocking UDP socket (viz/telemetry.py schema); every drawn frame reads all pending datagrams, keeps
the latest level / frame message for drawing and every plan / event for the history and the running stats.
Never blocks, never sends anything (the agent cannot notice the viewer). Draws at most 30 frames per second.

Replay: runs/<name>/plans.jsonl, events.jsonl and states.jsonl, one step per logged frame. The level map comes
from the run's own wm_data/levels (--wm-collect) or else any other recording of that level under runs/*/wm_data;
hazards from the run's wm_data world records (the recorded hazards at +8 / +16 frames stand in for the
planner's expectation, which is not logged; waves repeat exactly under the time-step lock); remembered deaths
from the run's death events (memory_added). Without wm_data the replay has no hazards (said in the panel).

Drawn (map, y up): tiles (solid, special, gone), the kill box / alive box, goal and keys with the planner's
contact region (outline: what counts as touching them), spawn; hazards now (red,
rotated, already grown by their kill reach), predicted at +8 / +16 (faint outlines), remembered deaths
(purple); planned agents (white) and explore agents (small dim dots) with a velocity arrow (0.1 s); deaths
(X: red unexplained, orange explained). For the selected agent: the chosen path (green; dashed and faded where
the ensemble spread is over 20 units; the part already played thinner), its airborne completion (plans.jsonl
"completion": pale green, dashed, from the plan's end to where it lands, landing yellow, or a red X where every
continuation died), landings (yellow) and wall jumps (cyan)
with their frame, where it touches the princess (pink ring) or picks a key (yellow ring), the nearest hazards' clearance (a thin line from the path point at that frame to the closest
box of that kind, labelled with the planner's own clearance number), the rejected alternatives (red vetoed,
grey otherwise; hover for the reason), the plan's predictions at +1 / +4 / +8 / +16 (hollow circles) and, for
each horizon, the latest prediction whose frame has arrived with a line to the real position and the error
(green within the surprise threshold, red over it). The info panel also shows the last planned finish time and
the level's best (from the finish events; before any, from smbrl/best_times.json), and for the current decision
what triggered it and whether the committed plan was kept or switched, with the margin (plans.jsonl "trigger"
and "switch"; older runs without them show "-"). The replay's timeline marks every decision of the planned
agents (short ticks at its bottom: orange where the plan was switched, grey otherwise).

Keys: Tab next planned agent, H help, A alternatives, F future hazards, Z zoom on the selected agent (mouse wheel or
[ / ] zoom in / out), S screenshot (replay: into the run folder; live: the current folder), Esc / Q quit. Replay: Space play / pause, Left / Right one step, + / - speed,
N next event, D next surprise or unexplained death, Home / End, click the timeline to seek.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import glob
import gzip
import json
import math
import os
import socket
import sys
import time
import zlib
from collections import deque
from dataclasses import dataclass, field, replace

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_WINDOW_NO_ACTIVATION_WHEN_SHOWN", "1")   # do not take focus from the game

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # the smbrl folder
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from viz import telemetry as T                                            # noqa: E402
from plan.explain import HORIZONS, ERR90_AT, SURPRISE_FACTOR             # noqa: E402

RUNS = os.path.join(HERE, "runs")
BEST_TIMES = os.path.join(HERE, "best_times.json")      # agent.py --best-times default
TILE = 20.0
BODY = 6.0                    # realpath.BODY: a goal / key box is touched within this of it
FPS = 30
HZ = 60                       # game frames per second (replay speed 1x)
PANEL_W = 480
TIMELINE_H = 22
SPREAD_FAR = 20.0             # units: path segments with a larger ensemble spread are drawn dashed and faded
ARROW_S = 0.1                 # seconds of velocity per arrow
MATURED_MAX_AGE = 8           # frames: a matured prediction is shown this long after its frame
NEAREST_DRAWN = 3
FOLLOW_ZOOM, MAX_ZOOM = 3.0, 12.0
HOVER_PX = 8
EVENTS_SHOWN = 12
MEMORY_WINDOW, MEMORY_HALF = 30, 8.0      # plan/safety.py DeathMemory WINDOW / DEATH_HALF (replay boxes)
KIND_HAZARD, KIND_SHOT, KIND_MEMORY = 0, 1, 3
KIND_OF_NAME = {"hazard": KIND_HAZARD, "shot": KIND_SHOT, "remembered death": KIND_MEMORY}
SURPRISE_KINDS = ("position", "unpredicted death")

# colours (monitor.py's palette)
BG = (16, 17, 23)
PANEL = (24, 25, 34)
SOLID = (64, 68, 88)
SPECIAL = (150, 128, 70)
ACCEL = (190, 70, 200)
GONE = (34, 36, 48)           # a solid tile that crumbled / opened (the words say solid, the live map does not)
TEXT = (222, 222, 236)
DIM = (140, 142, 162)
FAINT = (80, 82, 100)
HAZ = (215, 60, 60)
SHOT = (255, 120, 60)
MEMORY = (170, 90, 255)
GOAL = (245, 120, 185)
SPAWN = (90, 210, 120)
KEY = (255, 230, 40)
KILLBOX = (220, 90, 90)
ALIVEBOX = (90, 92, 110)
PLAN = (40, 225, 70)
PLAN_DONE = (30, 120, 45)
COMPLETION = (110, 170, 110)       # the airborne completion after the plan (fainter, dashed)
LAND = (255, 225, 40)
WALL = (40, 225, 255)
ALT = (150, 150, 150)
ALT_VETO = (230, 70, 70)
PRED = (235, 120, 255)
GOOD, BAD, WARN = (90, 220, 110), (240, 80, 70), (255, 170, 60)
SELECT = (255, 225, 40)


# ---- small helpers -------------------------------------------------------------------------------------------
def unpack(s, dtype, shape):
    """base64(zlib(raw)) -> array of that shape, or None when absent / malformed."""
    if not s:
        return None
    try:
        a = np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype)
        return a.reshape(shape) if a.size == shape[0] * shape[1] else None
    except (ValueError, zlib.error, TypeError):
        return None


def box_corners(b):
    x, y, hw, hh, a = (float(v) for v in b[:5])
    c, s = math.cos(a), math.sin(a)
    return [(x + u * c - v * s, y + u * s + v * c) for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def box_closest(px, py, b):
    """The point of a rotated box [x, y, hw, hh, angle, ...] closest to (px, py) (telemetry's box convention)."""
    x, y, hw, hh, a = (float(v) for v in b[:5])
    c, s = math.cos(a), math.sin(a)
    dx, dy = px - x, py - y
    u = min(max(dx * c + dy * s, -hw), hw)
    v = min(max(-dx * s + dy * c, -hh), hh)
    return x + u * c - v * s, y + u * s + v * c


def edge_closest(px, py, box):
    """The point of the alive box's edge (left, right, bottom, top) closest to (px, py)."""
    left, right, bottom, top = box
    cands = [(abs(px - left), (left, py)), (abs(right - px), (right, py)), (abs(py - bottom), (px, bottom)),
             (abs(top - py), (px, top))]
    return min(cands, key=lambda t: t[0])[1]


def threshold(h: int) -> float:
    """plan.explain.SurpriseTracker: a surprise is a position error over this at horizon h."""
    return SURPRISE_FACTOR * ERR90_AT.get(h, max(ERR90_AT.values()))


def plan_name(p: dict) -> str:
    pad = int(p.get("pad") or 0)
    return " > ".join(p.get("chosen") or []) + (f" (+{pad}f)" if pad else "")


def matured_predictions(plans: list, real_at, now: int, max_age: int = MATURED_MAX_AGE) -> list[dict]:
    """Prediction vs reality. plans: one agent's plans of one wave, in frame order. For each horizon h: the
    latest plan whose +h frame has arrived (plan frame + h <= now, at most max_age frames ago), its predicted
    position, the real one at that frame (real_at(frame) -> (x, y) or None) and the error in units (None when
    the real position is unknown)."""
    out = []
    for h in HORIZONS:
        for p in reversed(plans):
            pr = (p.get("predictions") or {}).get(str(h))
            t = int(p["frame"]) + h
            if pr is None or t > now:
                continue
            if now - t <= max_age:
                real = real_at(t)
                err = math.hypot(real[0] - pr[0], real[1] - pr[1]) if real is not None else None
                out.append({"h": h, "plan_frame": int(p["frame"]), "frame": t, "pred": (float(pr[0]), float(pr[1])),
                            "real": real, "error": err, "threshold": threshold(h)})
            break
    return out


def resolve_run(arg: str, runs_dir: str = RUNS) -> str:
    """--replay argument -> run folder: 'latest' = the newest runs/plan_* folder (by name: plan_<date>-<time>);
    else a path as given, or a folder name under runs/."""
    if arg == "latest":
        cands = sorted(d for d in glob.glob(os.path.join(runs_dir, "plan_*")) if os.path.isdir(d))
        if not cands:
            raise SystemExit(f"--replay latest: no plan_* run folder in {runs_dir}")
        return cands[-1]
    for p in (arg, os.path.join(runs_dir, arg), os.path.join(HERE, arg)):
        if os.path.isdir(p):
            return os.path.abspath(p)
    raise SystemExit(f"--replay: no run folder {arg!r} (looked as given, under {runs_dir} and under {HERE})")


def read_jsonl(path: str) -> list[dict]:
    """A JSON-lines log; a missing file is empty, a malformed (e.g. half-written last) line is skipped."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
    return out


# ---- the level ---------------------------------------------------------------------------------------------
@dataclass
class LevelView:
    """A telemetry level message, decoded for drawing."""
    name: str
    W: int
    H: int
    ox: float
    oy: float
    solid: np.ndarray            # (H, W) uint8, row 0 = bottom
    words: np.ndarray | None     # (H, W) uint32 or None
    kill_box: tuple | None
    alive_box: tuple
    goals: list
    spawn: tuple | None
    keys: list
    bandages: list
    portals: list
    planned: int
    explore: int
    field: str | None
    key: tuple                   # changes when the drawn tiles change
    goal_boxes: list = field(default_factory=list)    # [x, y, hw, hh, angle]: touched within BODY (planner)
    key_boxes: list = field(default_factory=list)

    @classmethod
    def from_message(cls, m: dict) -> "LevelView | None":
        try:
            W, H = int(m["W"]), int(m["H"])
            solid = unpack(m.get("solid"), np.uint8, (H, W))
            if solid is None or W <= 0 or H <= 0:
                return None
            words = unpack(m.get("words"), "<u4", (H, W))
            ag = m.get("agents") or {}
            kb = m.get("kill_box")
            ox, oy = float(m["ox"]), float(m["oy"])
            ab = m.get("alive_box") or [ox - 10, ox + W * TILE + 10, oy - 60, oy + H * TILE + 10]
            return cls(str(m.get("name") or m.get("level")), W, H, ox, oy, solid, words,
                       tuple(kb) if kb else None, tuple(float(v) for v in ab), [tuple(g) for g in m.get("goals") or []],
                       tuple(m["spawn"]) if m.get("spawn") else None, [tuple(k) for k in m.get("keys") or []],
                       [tuple(b) for b in m.get("bandages") or []], m.get("portals") or [],
                       int(ag.get("planned", 0)), int(ag.get("explore", 0)), m.get("field"),
                       (m.get("name"), W, H, ox, oy, hash(m.get("solid")), m.get("words") is not None),
                       [list(b) for b in m.get("goal_boxes") or []], [list(b) for b in m.get("key_boxes") or []])
        except (KeyError, TypeError, ValueError):
            return None

    def bounds(self):
        x0, x1, y0, y1 = self.ox, self.ox + self.W * TILE, self.oy, self.oy + self.H * TILE
        if self.kill_box:
            l, r, b, t = self.kill_box
            x0, x1, y0, y1 = min(x0, l), max(x1, r), min(y0, b), max(y1, t)
        return x0, x1, y0, y1

    def colours(self) -> np.ndarray:
        """(W, H, 3) RGB for pygame.surfarray, x first, the top row first."""
        rgb = np.zeros((self.H, self.W, 3), np.uint8)
        rgb[:] = BG
        solid = self.solid.astype(bool)
        if self.words is not None:
            tid = self.words & 0xff
            special = ((self.words >> 8) & 1).astype(bool)
            was = ((tid > 0) & (tid < 100)) | (special & (tid > 0) & (tid <= 100))
            rgb[was & ~solid] = GONE
            rgb[solid] = SOLID
            rgb[solid & (tid == 98) & ~special] = ACCEL
            rgb[solid & special] = SPECIAL
        else:
            rgb[solid] = SOLID
        return np.ascontiguousarray(rgb[::-1].transpose(1, 0, 2))


# ---- what one drawn frame shows ------------------------------------------------------------------------------
@dataclass
class Scene:
    mode: str                                    # "live" | "replay"
    title: str
    level: LevelView | None = None
    level_name: str | None = None
    wave: int = 0
    frame: int = 0
    agents: list = field(default_factory=list)   # [{i, role, x, y, vx, vy, status}]
    hazards: list = field(default_factory=list)  # telemetry boxes [x, y, hw, hh, a, kind]
    future: dict = field(default_factory=dict)   # {"8": boxes, "16": boxes}
    future_note: str = "predicted"
    memory: list = field(default_factory=list)
    plans: dict = field(default_factory=dict)    # agent -> its latest plan (dict) at or before this frame
    history: dict = field(default_factory=dict)  # agent -> its recent plans this wave, frame order
    real_at: object = None                       # (agent, frame) -> (x, y) or None
    boxes_at: object = None                      # (absolute frame) -> boxes then, or None
    deaths: list = field(default_factory=list)   # death events this wave so far
    events: list = field(default_factory=list)   # recent events, newest last
    stats: dict = field(default_factory=dict)
    status: list = field(default_factory=list)   # extra header lines
    planned: list = field(default_factory=list)  # planned agent indices
    timeline: dict | None = None                 # replay: {n, idx, marks: [(idx, colour)], decisions: [(idx, switched)]}


def event_colour(e: dict):
    k = e.get("kind")
    if k == "death":
        return BAD if not e.get("explained") else WARN
    if k == "unpredicted death":
        return BAD
    if k == "position":
        return WARN
    if k == "failed move":
        return (230, 130, 230)
    if k == "finish":
        return GOOD
    if k in ("plan error", "lock"):
        return BAD
    return DIM


def event_text(e: dict) -> str:
    k, f = e.get("kind"), e.get("frame")
    fs = f"f{f} " if f is not None else ""
    if k == "position":
        return (f"{fs}surprise a{e.get('agent')} +{e.get('horizon')}: off by {e.get('error', 0):.1f}u "
                f"(> {e.get('threshold', 0):.1f})")
    if k == "unpredicted death":
        return f"{fs}SURPRISE death a{e.get('agent')} at ({e.get('x', 0):.0f},{e.get('y', 0):.0f}) [{e.get('plan')}]"
    if k == "death":
        how = f"explained: {e.get('cause')}" if e.get("explained") else "UNEXPLAINED" + \
            (" -> remembered" if e.get("memory_added") else "")
        return f"{fs}death a{e.get('agent')} ({e.get('x', 0):.0f},{e.get('y', 0):.0f}) {how}" + \
            (" (est.)" if e.get("estimated") else "")
    if k == "failed move":
        return f"{fs}failed move a{e.get('agent')}: {e.get('reason') or e.get('move')}"
    if k == "finish":
        t = e.get("seconds")
        return f"{fs}finish a{e.get('agent')}" + (f" in {t:.2f} s" if t is not None else "") + \
            (" NEW BEST" if e.get("new_best") else "")
    if k == "wave":
        return f"{fs}wave end ({e.get('why')}): {e.get('finished')} finished, {e.get('died')} died"
    if k == "plan error":
        return f"{fs}plan error a{e.get('agent')}: {e.get('error')}"
    if k == "field":
        return f"field ({e.get('why')})"
    if k == "level":
        return f"level {e.get('name') or e.get('level')}"
    if k == "kill box":
        return f"{fs}kill box {e.get('kill_box')}"
    return f"{fs}{k}"


def is_surprise_or_unexplained(e: dict) -> bool:
    return e.get("kind") in SURPRISE_KINDS or (e.get("kind") == "death" and not e.get("explained"))


class Stats:
    """Running stats: decisions, ms per decision (mean / 95%), surprises per 100 decisions, unexplained deaths."""

    def __init__(self):
        self.ms = []
        self.surprises = self.unexplained = self.deaths = self.failed = 0
        self.switched = self.kept = 0             # decisions that switched / kept the committed plan
        self.last_finish = None                  # the last planned finish event with a time
        self.best = {}                           # level -> best planned time (s) as the finish events said

    def plan(self, p: dict):
        self.ms.append(float((p.get("timing") or {}).get("total") or 0.0))
        sw = p.get("switch")
        self.switched += int(bool(sw) and not sw.get("kept"))
        self.kept += int(bool(sw) and bool(sw.get("kept")))

    def event(self, e: dict):
        k = e.get("kind")
        if k in SURPRISE_KINDS:
            self.surprises += 1
        elif k == "death":
            self.deaths += 1
            self.unexplained += int(not e.get("explained"))
        elif k == "failed move":
            self.failed += 1
        elif k == "finish" and e.get("seconds") is not None:
            self.last_finish = e
            if e.get("best") is not None:
                self.best[e.get("level")] = float(e["best"])

    def summary(self) -> dict:
        n = len(self.ms)
        ms = np.asarray(self.ms) if n else None
        return {"decisions": n, "ms_mean": float(ms.mean()) if n else None,
                "ms_95": float(np.percentile(ms, 95)) if n else None,
                "surprises": self.surprises, "surprises_per_100": 100.0 * self.surprises / n if n else None,
                "deaths": self.deaths, "unexplained": self.unexplained, "failed": self.failed,
                "switched": self.switched, "kept": self.kept, "last_finish": self.last_finish,
                "best": dict(self.best)}


def score_lines(plan: dict) -> list[str]:
    """The score's lines for a plan record: frames ("estimate N frames = H + completion k + T to go", then the
    penalties), or the older V-unit fields for older runs."""
    sc = plan.get("score") or {}
    v0 = fmt(plan.get("v_start"), ".1f")
    v1 = "inf" if plan.get("v_end") is None else fmt(plan.get("v_end"), ".1f")
    vtxt = f"V {v0} -> {v1}" + (f" ({plan.get('dead_end_members')} members dead end)"
                                 if plan.get("dead_end_members") else "") + f", leg {plan.get('leg', 0)}"
    if "estimate" not in sc:
        return [f"progress {fmt(sc.get('progress'), '+.2f')} (credit {fmt(sc.get('progress_credit'), '+.2f')}), " + vtxt,
                f"risk {fmt(sc.get('risk'), '.2f')}  uncertainty x{fmt(sc.get('uncertainty'), '.2f')} "
                f"(spread end {fmt(sc.get('spread_end'), '.1f')}u)  failure {fmt(sc.get('failure'), '.2f')}  "
                f"total {fmt(sc.get('total'), '+.2f')}"]
    comp = plan.get("completion")
    k = sc.get("completion")
    used = sc.get("frames_used")
    h = (used - k) if used is not None and k is not None else used
    head = f"estimate {fmt(sc.get('estimate'), '.0f')} frames = {fmt(h, '.0f')}"
    if comp is not None and k is None:
        head += " + completion: every continuation dies"
    elif k is not None:
        head += f" + completion {k} ({comp.get('continuation') if comp else '?'}, {comp.get('on') if comp else '?'})"
    head += f" + T to go {fmt(sc.get('t_end'), '.0f')}"
    pen = (f"+ risk {fmt(sc.get('risk_frames'), '.0f')}, failures {fmt(sc.get('failure_frames'), '.0f')}, deaths "
           f"{fmt(sc.get('death_frames'), '.0f')}; T at the start {fmt(sc.get('t_start'), '.0f')}; "
           f"{'lands' if sc.get('landing') else 'does not land'}")
    return [head, pen, vtxt + f"; spread end {fmt(sc.get('spread_end'), '.1f')}u"]


def decision_lines(plan: dict) -> tuple[str, str]:
    """('trigger: interval, move end', 'kept: best other +0.31 < 0.5') for a plan record; '-' for older runs
    without these fields, 'no previous plan' when there was none to keep."""
    tr = plan.get("trigger")
    sw = plan.get("switch")
    if "switch" not in plan:
        swt = "kept / switched: -"
    elif sw is None:
        swt = "no previous plan to keep"
    else:
        swt = str(sw.get("reason") or ("kept" if sw.get("kept") else "switched"))
        if sw.get("margin") is not None and sw.get("unit"):
            swt += f" (margin {sw['margin']:g} {sw['unit']})"
    return "trigger: " + (", ".join(tr) if tr else "-"), swt


_best_cache = {"mtime": None, "data": {}}


def file_best(level: str | None, path: str = BEST_TIMES) -> float | None:
    """The planned agents' best time (s) for a level from best_times.json (re-read when it changes)."""
    try:
        mt = os.path.getmtime(path)
        if mt != _best_cache["mtime"]:
            with open(path, encoding="utf-8") as f:
                _best_cache["data"], _best_cache["mtime"] = json.load(f), mt
        b = ((_best_cache["data"].get(level) or {}).get("planned") or {}).get("seconds")
        return float(b) if b is not None else None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def finish_lines(stats: dict, level: str | None) -> list[str]:
    """The info panel's finish lines: the last planned finish time and the level's best."""
    out = []
    e = stats.get("last_finish")
    if e:
        out.append(f"last finish: {e['seconds']:.2f} s ({e.get('frames')} frames) {e.get('level')} wave "
                   f"{e.get('wave')} a{e.get('agent')}" + (", NEW BEST" if e.get("new_best") else "")
                   + ("" if e.get("lock") else " (lock not confirmed)"))
    b = (stats.get("best") or {}).get(level)
    if b is not None:
        out.append(f"level best: {b:.2f} s")
    else:
        fb = file_best(level)
        out.append(f"level best: {fb:.2f} s (best_times.json)" if fb is not None else "level best: -")
    return out


# ---- live source ---------------------------------------------------------------------------------------------
class LiveSource:
    """UDP telemetry listener. poll() reads every pending datagram without blocking."""
    MAX_PER_POLL = 5000
    HIST_PLANS = 64
    HIST_FRAMES = 600

    def __init__(self, host: str = "127.0.0.1", port: int = 47800, bind: bool = True):
        self.addr = (host, port)
        self.sock = None
        if bind:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
            except OSError:
                pass
            s.bind(self.addr)
            s.setblocking(False)
            self.sock = s
            self.addr = s.getsockname()
        self.level_msg = None
        self.level: LevelView | None = None
        self.frame_msg = None
        self.seg = None                                 # (level, wave) of the histories
        self.plans: dict[int, deque] = {}
        self.real: dict[int, dict] = {}
        self.deaths: list = []
        self.events = deque(maxlen=200)
        self.stats = Stats()
        self.received = self.bad = 0
        self.last_rx = None

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def poll(self) -> int:
        """Handle every datagram waiting now; returns how many."""
        n = 0
        while self.sock is not None and n < self.MAX_PER_POLL:
            try:
                data = self.sock.recv(65536)
            except (BlockingIOError, InterruptedError):
                break
            except ConnectionResetError:          # Windows reports an earlier ICMP error here: keep reading
                continue
            except OSError:
                break
            n += 1
            self.datagram(data)
        return n

    def datagram(self, data: bytes):
        try:
            msg = T.decode(data)
        except Exception:
            self.bad += 1
            return
        self.received += 1
        self.last_rx = time.time()
        self.handle(msg)

    def _segment(self, msg):
        seg = (msg.get("level"), int(msg.get("wave") or 0))
        if seg != self.seg:
            self.seg = seg
            self.plans, self.real, self.deaths = {}, {}, []

    def handle(self, msg: dict):
        t = msg.get("t")
        if t == "level":
            if self.level_msg is None or msg.get("solid") != self.level_msg.get("solid") or \
                    msg.get("name") != self.level_msg.get("name") or msg.get("kill_box") != self.level_msg.get("kill_box") \
                    or (msg.get("words") and not self.level_msg.get("words")):
                lv = LevelView.from_message(msg)
                if lv is not None:
                    self.level = lv
            self.level_msg = msg
        elif t == "frame":
            self._segment(msg)
            self.frame_msg = msg
            f = int(msg.get("frame", 0))
            for a in msg.get("agents") or []:
                if a.get("role") == "planner":
                    h = self.real.setdefault(int(a["i"]), {})
                    h[f] = (float(a["x"]), float(a["y"]))
                    if len(h) > self.HIST_FRAMES:
                        for k in sorted(h)[:len(h) - self.HIST_FRAMES]:
                            del h[k]
        elif t == "plan":
            self._segment(msg)
            self.plans.setdefault(int(msg.get("agent", 0)), deque(maxlen=self.HIST_PLANS)).append(msg)
            self.stats.plan(msg)
        elif t == "event":
            self.events.append(msg)
            self.stats.event(msg)
            if msg.get("kind") == "death" and (msg.get("level"), int(msg.get("wave") or 0)) == self.seg:
                self.deaths.append(msg)

    def scene(self) -> Scene:
        fm = self.frame_msg or {}
        lv = self.level
        frame = int(fm.get("frame", 0))
        agents = [dict(a) for a in fm.get("agents") or []]
        planned = sorted(a["i"] for a in agents if a.get("role") == "planner")
        if not planned and lv is not None:
            planned = list(range(lv.planned))
        plans = {i: d[-1] for i, d in self.plans.items() if d}
        hist = {i: list(d) for i, d in self.plans.items()}
        sets = [(frame, list(fm.get("hazards") or []) + list(fm.get("memory") or []))]
        for k, v in (fm.get("future") or {}).items():
            sets.append((frame + int(k), v))

        def boxes_at(f):
            return min(sets, key=lambda s: abs(s[0] - f))[1] if fm else None

        age = time.time() - self.last_rx if self.last_rx else None
        status = [f"LIVE  listening on {self.addr[0]}:{self.addr[1]}  ({self.received} messages"
                  + (f", {self.bad} unreadable" if self.bad else "") + ")"]
        if age is None:
            status.append("waiting for agent.py telemetry (no sender yet)")
        elif age > 2.0:
            status.append(f"no telemetry for {age:.0f} s")
        return Scene("live", "live", lv, fm.get("level") or (lv.name if lv else None), int(fm.get("wave") or 0), frame,
                     agents, list(fm.get("hazards") or []), dict(fm.get("future") or {}), "predicted",
                     list(fm.get("memory") or []), plans, hist,
                     lambda i, f: self.real.get(int(i), {}).get(int(f)), boxes_at, list(self.deaths),
                     list(self.events)[-EVENTS_SHOWN:], self.stats.summary(), status, planned)


# ---- replay source --------------------------------------------------------------------------------------------
class ReplaySource:
    """A run's logs as a timeline of steps (one per logged frame)."""

    def __init__(self, run_dir: str, runs_dir: str = RUNS):
        self.run_dir = os.path.abspath(run_dir)
        self.runs_dir = runs_dir
        self.name = os.path.basename(self.run_dir.rstrip("/\\"))
        plans = read_jsonl(os.path.join(self.run_dir, "plans.jsonl"))
        events = read_jsonl(os.path.join(self.run_dir, "events.jsonl"))
        states = read_jsonl(os.path.join(self.run_dir, "states.jsonl"))
        if not states:      # no state log: one step per decision frame
            seen = []
            for p in plans:
                k = (p.get("level"), p.get("wave"), p.get("frame"))
                if not seen or seen[-1]["k"] != k:
                    seen.append({"k": k, "frame": p.get("frame"), "wave": p.get("wave"), "level": p.get("level"),
                                 "agents": []})
            states = seen
        if not states:
            raise SystemExit(f"{self.run_dir}: no plans.jsonl / states.jsonl lines to replay")
        self.steps = states
        self.n = len(states)
        # segments: a new one when the level or wave changes or the frame goes back
        self.seg = []
        s, prev = 0, None
        for st in states:
            k = (st.get("level"), st.get("wave"))
            if prev is not None and (k != prev[0] or int(st.get("frame") or 0) < prev[1]):
                s += 1
            self.seg.append(s)
            prev = (k, int(st.get("frame") or 0))
        self.real: dict[tuple, tuple] = {}
        for i, st in enumerate(states):
            for a in st.get("agents") or []:
                self.real[(self.seg[i], int(st["frame"]), int(a[0]))] = (float(a[1]), float(a[2]))
        self.planned = sorted({int(a[0]) for st in states for a in st.get("agents") or []} |
                              {int(p.get("agent", 0)) for p in plans})
        # plans and events -> steps (file order: each is written before / at its frame's state line)
        self.plan_step = self._assign(plans)
        self.plans = plans
        self.by_agent: dict[int, list] = {}
        for k, p in enumerate(plans):
            self.by_agent.setdefault(int(p.get("agent", 0)), []).append((self.plan_step[k], p))
        self.agent_steps = {i: [s for s, _ in lst] for i, lst in self.by_agent.items()}
        self._cache = None                                 # (idx, Scene) of the last step drawn
        self.event_step = self._assign(events)
        self.events = events
        self.ev_steps = list(self.event_step)
        # cumulative stats per step
        self._plan_steps = list(self.plan_step)
        # remembered deaths and kill boxes from the events
        self.mem = [(self.event_step[k], e.get("level"), int(e["frame"]), float(e["x"]), float(e["y"]))
                    for k, e in enumerate(events)
                    if e.get("kind") == "death" and e.get("memory_added") and e.get("frame") is not None]
        self.kb = [(self.event_step[k], e.get("level") or e.get("name"), e.get("kill_box"))
                   for k, e in enumerate(events) if e.get("kind") in ("level", "kill box")]
        # level maps and recorded hazards
        self.levels: dict[str, object] = {}
        self.level_note: dict[str, str] = {}
        self.world = self._load_world()
        self._boxes_cache: dict = {}
        self.marks = [(self.event_step[k], event_colour(e)) for k, e in enumerate(events)
                      if e.get("kind") in SURPRISE_KINDS + ("death", "failed move", "finish")]
        self.decision_marks = [(self.plan_step[k], bool(p.get("switch")) and not p["switch"].get("kept"))
                               for k, p in enumerate(plans)]

    def _assign(self, rows: list) -> list[int]:
        out, ptr = [], 0
        for r in rows:
            k = (r.get("level"), r.get("wave"), r.get("frame"))
            hit = None
            if r.get("frame") is not None:
                for j in range(ptr, min(self.n, ptr + 4000)):
                    st = self.steps[j]
                    if (st.get("level"), st.get("wave"), st.get("frame")) == k:
                        hit = j
                        break
            if hit is not None:
                ptr = hit
            out.append(hit if hit is not None else ptr)
        return out

    # -- level map and hazards --
    def _level_files(self, name: str) -> list[str]:
        own = os.path.join(self.run_dir, "wm_data", "levels", f"{name}.npz")
        others = sorted(glob.glob(os.path.join(self.runs_dir, "*", "wm_data", "levels", f"{name}.npz")),
                        key=lambda p: -os.path.getmtime(p))
        return ([own] if os.path.exists(own) else []) + [p for p in others if os.path.abspath(p) != own]

    def level_obj(self, name: str):
        """game.level.Level of a level name from a recording (cached), or None."""
        if name in self.levels:
            return self.levels[name]
        lv = None
        for p in self._level_files(name):
            try:
                from game.level import Level
                z = np.load(p)
                meta = json.loads(str(z["meta"]))
                lv = Level(dict(meta, name=name), np.asarray(z["words"], "<u4").tobytes())
                self.level_note[name] = f"map: {os.path.relpath(p, self.run_dir) if p.startswith(self.run_dir) else p}"
                break
            except Exception as e:           # a damaged file: try the next recording
                self.level_note[name] = f"map file unreadable: {type(e).__name__}"
        if lv is None:
            self.level_note.setdefault(name, f"no map of {name}: run without --wm-collect and no other recording")
        self.levels[name] = lv
        return lv

    def level_view(self, name: str, idx: int):
        lv = self.level_obj(name) if name else None
        if lv is None:
            return None
        kb = None
        for s, lname, k in self.kb:
            if s <= idx and lname == name:
                kb = k
        from game.level import parse_kill_box, alive_box
        kb = parse_kill_box(kb) if kb is not None else lv.kill_box
        key = (name, kb)
        cache = getattr(self, "_lv_cache", {})
        self._lv_cache = cache
        if key not in cache:
            m = T.level_message(lv, 0, len(self.planned), 0)
            m["kill_box"] = list(kb) if kb else None
            m["alive_box"] = list(alive_box(lv.ox, lv.oy, lv.W, lv.H, kb))
            cache[key] = LevelView.from_message(T.decode(T.encode(m)))
        return cache[key]

    def _load_world(self) -> dict:
        """(level, wave) -> [world records by frame dict, ...] from the run's wm_data shards, in shard order."""
        out: dict = {}
        for p in sorted(glob.glob(os.path.join(self.run_dir, "wm_data", "wave_*.npz"))):
            wp = p[:-4] + "_world.json.gz"
            if not os.path.exists(wp):
                continue
            try:
                meta = json.loads(str(np.load(p)["meta"]))
                world = json.load(gzip.open(wp, "rt", encoding="utf-8"))
            except Exception:
                continue
            keep, recs = None, {}
            for w in world:
                if w.get("keep") is not None:
                    keep = w["keep"]
                recs[int(w["f"])] = (w, keep)
            out.setdefault((meta.get("level"), int(meta.get("wave") or 0)), []).append(recs)
        # which occurrence of (level, wave) each segment is
        self.seg_world = {}
        count: dict = {}
        for i, st in enumerate(self.steps):
            s = self.seg[i]
            if s in self.seg_world:
                continue
            k = (st.get("level"), int(st.get("wave") or 0))
            c = count.get(k, 0)
            count[k] = c + 1
            lst = out.get(k, [])
            self.seg_world[s] = lst[c] if c < len(lst) else None
        return out

    def boxes(self, seg: int, frame: int, level: str) -> list | None:
        """Recorded hazard boxes (grown, telemetry rows) at a wave frame of a segment, None when not recorded."""
        key = (seg, frame)
        if key in self._boxes_cache:
            return self._boxes_cache[key]
        recs = self.seg_world.get(seg)
        out = None
        if recs is not None and frame in recs:
            lv = self.level_obj(level)
            if lv is not None:
                from plan.safety import record_boxes
                rec, keep = recs[frame]
                fb = record_boxes(rec, keep if keep is not None else lv.hz_keep, lv.hz_grow)
                out = T.box_rows(fb.boxes, fb.kind)
        if len(self._boxes_cache) > 4000:
            self._boxes_cache.clear()
        self._boxes_cache[key] = out
        return out

    def memory_boxes(self, idx: int, level: str, frame: int) -> list:
        return [[x, y, MEMORY_HALF, MEMORY_HALF, 0.0, KIND_MEMORY] for s, lname, f, x, y in self.mem
                if s <= idx and lname == level and abs(f - frame) <= MEMORY_WINDOW]

    # -- navigation --
    def next_event(self, idx: int, pred=None) -> int | None:
        for k, s in enumerate(self.event_step):
            if s > idx and (pred is None or pred(self.events[k])):
                return s
        return None

    # -- one step --
    def scene(self, idx: int, speed: float = 1.0, playing: bool = False) -> Scene:
        idx = max(0, min(self.n - 1, int(idx)))
        head = f"REPLAY {self.name}  step {idx + 1}/{self.n}  x{speed:g} {'playing' if playing else 'paused'}"
        if self._cache is None or self._cache[0] != idx:
            self._cache = (idx, self._scene(idx))
        sc = self._cache[1]
        return replace(sc, status=[head] + sc.status[1:])

    def _scene(self, idx: int) -> Scene:
        st = self.steps[idx]
        seg, frame, level, wave = self.seg[idx], int(st.get("frame") or 0), st.get("level"), int(st.get("wave") or 0)
        lv = self.level_view(level, idx)
        agents = [{"i": int(a[0]), "role": "planner", "x": float(a[1]), "y": float(a[2]), "vx": float(a[3]),
                   "vy": float(a[4]), "status": int(a[5])} for a in st.get("agents") or []]
        plans, hist = {}, {}
        for i, lst in self.by_agent.items():
            j = bisect.bisect_right(self.agent_steps[i], idx)
            recent = [p for s, p in lst[max(0, j - 40):j] if self.seg[s] == seg]
            if recent:
                plans[i], hist[i] = recent[-1], recent
        mem = self.memory_boxes(idx, level, frame)
        now = self.boxes(seg, frame, level)
        future = {}
        for h in T.FUTURE_AT:
            b = self.boxes(seg, frame + h, level)
            if b is not None:
                future[str(h)] = b + self.memory_boxes(idx, level, frame + h)

        def boxes_at(f):
            b = self.boxes(seg, f, level)
            return (b or []) + self.memory_boxes(idx, level, f) if b is not None or self.mem else None

        j = bisect.bisect_right(self.ev_steps, idx)
        evs = self.events[:j]
        deaths = [e for k, e in enumerate(evs) if e.get("kind") == "death" and self.seg[self.event_step[k]] == seg]
        stats = Stats()
        for k in range(bisect.bisect_right(self._plan_steps, idx)):
            stats.plan(self.plans[k])
        for e in evs:
            stats.event(e)
        status = ["", self.level_note.get(level, "") if level else ""]
        if now is None:
            status.append("no recorded hazards (the run has no wm_data for this wave)")
        return Scene("replay", self.name, lv, level, wave, frame, agents, now or [], future,
                     "recorded (not the planner's prediction)", mem, plans, hist,
                     lambda i, f: self.real.get((seg, int(f), int(i))), boxes_at, deaths,
                     [e for e in evs if e.get("kind") != "field"][-EVENTS_SHOWN:], stats.summary(), status,
                     list(self.planned), {"n": self.n, "idx": idx, "marks": self.marks,
                                          "decisions": self.decision_marks})


# ---- drawing --------------------------------------------------------------------------------------------------
class Mapper:
    """World (y up) -> screen: the bounds fitted into rect, times zoom, around centre (default the bounds'
    centre; kept so the view does not leave the bounds where it can)."""

    def __init__(self, rect, bounds, zoom: float = 1.0, centre=None):
        x0, x1, y0, y1 = bounds
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        self.sc = min((rect[2] - 20) / w, (rect[3] - 20) / h) * max(1.0, zoom)
        cx, cy = centre if centre is not None else ((x0 + x1) / 2, (y0 + y1) / 2)
        hw, hh = rect[2] / 2 / self.sc, rect[3] / 2 / self.sc
        cx = (x0 + x1) / 2 if hw * 2 >= w else min(max(cx, x0 + hw), x1 - hw)
        cy = (y0 + y1) / 2 if hh * 2 >= h else min(max(cy, y0 + hh), y1 - hh)
        self.cx, self.cy = cx, cy
        self.rcx, self.rcy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2

    def pt(self, x, y):
        return (self.rcx + (float(x) - self.cx) * self.sc, self.rcy - (float(y) - self.cy) * self.sc)

    def world(self, px, py):
        return self.cx + (px - self.rcx) / self.sc, self.cy - (py - self.rcy) / self.sc


class Renderer:
    """Draws a Scene onto a surface (a window, or an off-screen surface in tests)."""

    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.font.init()
        self.font = self._font(14)
        self.small = self._font(12)
        self.big = self._font(17, bold=True)
        self.show_alts = True
        self.show_future = True
        self.show_help = False
        self.sel = None
        self.mouse = None
        self.zoom = 1.0                         # 1 = the whole level
        self.follow = False                     # zoomed view centred on the selected agent
        self._small = None                      # (level key, the level's tiles, 1 pixel per tile)
        self._tiles = None                      # (key, scaled visible part)
        self.alt_hits = []                      # [(screen points, alternative)] of the last draw
        self.timeline_rect = None

    def _font(self, size, bold=False):
        pg = self.pg
        for name in ("consolas", "couriernew", "dejavusansmono", "monospace"):
            try:
                f = pg.font.SysFont(name, size, bold=bold)
                if f is not None:
                    return f
            except Exception:
                continue
        return pg.font.Font(None, size + 4)

    def text(self, surf, s, x, y, col=TEXT, font=None):
        img = (font or self.font).render(str(s), True, col)
        surf.blit(img, (x, y))
        return img.get_width()

    # -- selection --
    def selected(self, scene: Scene):
        if not scene.planned:
            return None
        if self.sel not in scene.planned:
            self.sel = scene.planned[0]
        return self.sel

    def next_agent(self, scene: Scene):
        if scene.planned:
            cur = self.selected(scene)
            self.sel = scene.planned[(scene.planned.index(cur) + 1) % len(scene.planned)]

    # -- the whole frame --
    def render(self, surf, scene: Scene):
        pg = self.pg
        W, H = surf.get_size()
        map_w = max(200, W - PANEL_W)
        map_h = H - (TIMELINE_H if scene.timeline else 0)
        surf.fill(PANEL)
        surf.fill(BG, (0, 0, map_w, map_h))
        rect = (0, 0, map_w, map_h)
        bounds = self._bounds(scene)
        sel = self.selected(scene)
        centre = None
        if self.follow:
            a = next((a for a in scene.agents if a["i"] == sel), None)
            centre = (a["x"], a["y"]) if a is not None else None
        m = Mapper(rect, bounds, self.zoom, centre) if bounds else None
        self.alt_hits = []
        if m is not None:
            clip = surf.get_clip()
            surf.set_clip(pg.Rect(rect))
            self.draw_map(surf, scene, m, rect)
            plan = scene.plans.get(sel) if sel is not None else None
            if plan is not None and plan.get("path"):
                self.draw_plan(surf, scene, m, plan, sel)
            self.draw_agents(surf, scene, m, sel)
            self.draw_hover(surf)
            surf.set_clip(clip)
        else:
            self.text(surf, "waiting for a level / frame...", 30, 30, DIM, self.big)
        self.text(surf, "green plan  yellow landing  cyan wall jump  red hazard  purple remembered death  "
                        "o prediction", 8, 4, FAINT, self.small)
        if scene.timeline:
            self.draw_timeline(surf, scene, (0, map_h, map_w, TIMELINE_H))
        self.draw_panel(surf, scene, sel, map_w + 10, W - map_w - 18)
        if self.show_help:
            self.draw_help(surf, scene)

    def _bounds(self, scene: Scene):
        if scene.level is not None:
            return scene.level.bounds()
        xs = [a["x"] for a in scene.agents] + [p[0] for pl in scene.plans.values() for p in pl.get("path") or []]
        ys = [a["y"] for a in scene.agents] + [p[1] for pl in scene.plans.values() for p in pl.get("path") or []]
        if not xs:
            return None
        return min(xs) - 100, max(xs) + 100, min(ys) - 100, max(ys) + 100

    # -- map --
    def draw_tiles(self, surf, lv: LevelView, m: Mapper, rect):
        """The level's tiles: one small surface per level (1 pixel per tile), its visible part scaled (cached
        while the view and the tiles stay the same)."""
        pg = self.pg
        if self._small is None or self._small[0] != lv.key:
            self._small = (lv.key, pg.surfarray.make_surface(lv.colours()))
            self._tiles = None
        wx0, wy1 = m.world(rect[0], rect[1])
        wx1, wy0 = m.world(rect[0] + rect[2], rect[1] + rect[3])
        c0, c1 = max(0, math.floor((wx0 - lv.ox) / TILE)), min(lv.W, math.ceil((wx1 - lv.ox) / TILE))
        r0, r1 = max(0, math.floor((wy0 - lv.oy) / TILE)), min(lv.H, math.ceil((wy1 - lv.oy) / TILE))
        if c0 >= c1 or r0 >= r1:
            return
        ax, ay = m.pt(lv.ox + c0 * TILE, lv.oy + r1 * TILE)
        bx, by = m.pt(lv.ox + c1 * TILE, lv.oy + r0 * TILE)
        ax, ay, bx, by = (int(round(v)) for v in (ax, ay, bx, by))
        key = (lv.key, c0, c1, r0, r1, bx - ax, by - ay)
        if self._tiles is None or self._tiles[0] != key:
            sub = self._small[1].subsurface((c0, lv.H - r1, c1 - c0, r1 - r0))
            self._tiles = (key, pg.transform.scale(sub, (max(1, bx - ax), max(1, by - ay))))
        surf.blit(self._tiles[1], (ax, ay))

    def poly(self, surf, m, b, col, width=0, alpha=None):
        pts = [m.pt(x, y) for x, y in box_corners(b)]
        if alpha is None:
            self.pg.draw.polygon(surf, col, pts, width)
        else:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x0, y0 = int(min(xs)) - 2, int(min(ys)) - 2
            w, h = int(max(xs)) - x0 + 3, int(max(ys)) - y0 + 3
            if w <= 0 or h <= 0 or w > 4000 or h > 4000:
                return
            s = self.pg.Surface((w, h), self.pg.SRCALPHA)
            self.pg.draw.polygon(s, col + (alpha,), [(x - x0, y - y0) for x, y in pts], width)
            surf.blit(s, (x0, y0))

    def draw_map(self, surf, scene: Scene, m: Mapper, rect):
        pg = self.pg
        lv = scene.level
        if lv is not None:
            self.draw_tiles(surf, lv, m, rect)
            l, r, b, t = lv.alive_box
            (x0, y0), (x1, y1) = m.pt(l, t), m.pt(r, b)
            pg.draw.rect(surf, KILLBOX if lv.kill_box else ALIVEBOX, pg.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0)), 1)
            if lv.spawn:
                pg.draw.circle(surf, SPAWN, m.pt(*lv.spawn[:2]), min(14, max(4, int(8 * m.sc))), 2)
            for b in lv.goal_boxes:          # the planner's contact region: the box grown by the body
                self.poly(surf, m, [b[0], b[1], b[2] + BODY, b[3] + BODY, b[4]], GOAL, 1)
            for b in lv.key_boxes:
                self.poly(surf, m, [b[0], b[1], b[2] + BODY, b[3] + BODY, b[4]], KEY, 1)
            for g in lv.goals:
                pg.draw.circle(surf, GOAL, m.pt(*g[:2]), min(16, max(5, int(10 * m.sc))))
            for kx, ky in [k[:2] for k in lv.keys]:
                px, py = m.pt(kx, ky)
                pg.draw.polygon(surf, KEY, [(px, py - 7), (px + 5, py), (px, py + 7), (px - 5, py)])
            for bx, by in [b_[:2] for b_ in lv.bandages]:
                pg.draw.circle(surf, (255, 255, 255), m.pt(bx, by), 6, 2)
            for pr in lv.portals:
                try:
                    a, b_ = pr
                    pg.draw.line(surf, MEMORY, m.pt(*a[:2]), m.pt(*b_[:2]), 1)
                except (TypeError, ValueError):
                    pass
        if self.show_future:
            for key, alpha in (("16", 60), ("8", 110)):
                for b in scene.future.get(key) or []:
                    col = MEMORY if len(b) > 5 and b[5] == KIND_MEMORY else HAZ
                    self.poly(surf, m, b, col, 1, alpha)
        for b in scene.hazards:
            col = SHOT if len(b) > 5 and b[5] == KIND_SHOT else HAZ
            self.poly(surf, m, b, col, 0, 120)
            self.poly(surf, m, b, col, 1)
        for b in scene.memory:
            self.poly(surf, m, b, MEMORY, 0, 90)
            self.poly(surf, m, b, MEMORY, 1)
        for e in scene.deaths:
            if e.get("x") is None:
                continue
            px, py = m.pt(e["x"], e["y"])
            col = WARN if e.get("explained") else BAD
            pg.draw.line(surf, col, (px - 5, py - 5), (px + 5, py + 5), 2)
            pg.draw.line(surf, col, (px - 5, py + 5), (px + 5, py - 5), 2)

    def draw_agents(self, surf, scene: Scene, m: Mapper, sel):
        pg = self.pg
        for a in sorted(scene.agents, key=lambda a: a.get("role") == "planner"):
            p = m.pt(a["x"], a["y"])
            planned = a.get("role") == "planner"
            st = int(a.get("status", 0))
            col = (240, 240, 245) if planned else (110, 112, 130)
            if st == 0:
                end = m.pt(a["x"] + a.get("vx", 0.0) * ARROW_S, a["y"] + a.get("vy", 0.0) * ARROW_S)
                if math.hypot(end[0] - p[0], end[1] - p[1]) > 2:
                    self.arrow(surf, col if planned else FAINT, p, end, 2 if planned else 1)
            if planned:
                if st == 1:
                    pg.draw.line(surf, BAD, (p[0] - 5, p[1] - 5), (p[0] + 5, p[1] + 5), 2)
                    pg.draw.line(surf, BAD, (p[0] - 5, p[1] + 5), (p[0] + 5, p[1] - 5), 2)
                elif st == 2:
                    pg.draw.circle(surf, GOOD, p, 7, 2)
                else:
                    pg.draw.circle(surf, col, p, 5)
                if a["i"] == sel:
                    pg.draw.circle(surf, SELECT, p, 9, 1)
                self.text(surf, str(a["i"]), p[0] + 7, p[1] - 16, col, self.small)
            elif st == 0:
                pg.draw.circle(surf, col, p, 2)

    def arrow(self, surf, col, a, b, w=1):
        pg = self.pg
        pg.draw.line(surf, col, a, b, w)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        for d in (2.6, -2.6):
            pg.draw.line(surf, col, b, (b[0] + 6 * math.cos(ang + d), b[1] + 6 * math.sin(ang + d)), w)

    def draw_plan(self, surf, scene: Scene, m: Mapper, plan: dict, sel):
        pg = self.pg
        path = plan.get("path") or []
        spread = plan.get("spread") or []
        k_now = scene.frame - int(plan.get("frame", scene.frame))
        # alternatives first (under the chosen path)
        if self.show_alts:
            for alt in plan.get("alternatives") or []:
                ap = alt.get("path") or []
                if len(ap) < 2:
                    continue
                pts = [m.pt(x, y) for x, y in ap]
                col = ALT_VETO if alt.get("vetoed") else ALT
                pg.draw.lines(surf, col, False, pts, 1)
                self.alt_hits.append((pts, alt))
        # the chosen path: dashed and faded where the members disagree, thinner where already played
        pts = [m.pt(x, y) for x, y in path]
        for k in range(len(pts) - 1):
            far = k + 1 < len(spread) and spread[k + 1] is not None and spread[k + 1] > SPREAD_FAR
            done = k < k_now
            if far:
                if k % 2 == 0:
                    pg.draw.line(surf, PLAN_DONE, pts[k], pts[k + 1], 2)
            else:
                pg.draw.line(surf, PLAN_DONE if done else PLAN, pts[k], pts[k + 1], 2 if done else 3)
        # the airborne completion: dashed, fainter, from the plan's end to its landing (or death)
        comp = plan.get("completion") or {}
        cp = [m.pt(x, y) for x, y in comp.get("path") or []]
        for k in range(0, len(cp) - 1, 2):
            pg.draw.line(surf, COMPLETION, cp[k], cp[k + 1], 2)
        if cp:
            if comp.get("on") in ("ground", "wall", "goal"):
                pg.draw.circle(surf, LAND, cp[-1], 5, 2)
                self.text(surf, f"lands +{comp.get('frames')}f {comp.get('on')}", cp[-1][0] + 6, cp[-1][1] + 2, LAND,
                          self.small)
            elif comp.get("frames") is None:
                p = cp[-1]
                pg.draw.line(surf, BAD, (p[0] - 4, p[1] - 4), (p[0] + 4, p[1] + 4), 2)
                pg.draw.line(surf, BAD, (p[0] - 4, p[1] + 4), (p[0] + 4, p[1] - 4), 2)
        # landings and wall jumps
        for ld in plan.get("landings") or []:
            p = m.pt(ld["x"], ld["y"])
            pg.draw.circle(surf, LAND, p, 5)
            self.text(surf, f"f{ld['frame']}" + ("" if ld.get("on") == "ground" else " wall"), p[0] + 6, p[1] + 2, LAND,
                      self.small)
        for wj in plan.get("wall_jumps") or []:
            p = m.pt(wj["x"], wj["y"])
            pg.draw.circle(surf, WALL, p, 6, 2)
            self.text(surf, f"WJ f{wj['frame']} {wj.get('side', '')}", p[0] + 7, p[1] - 14, WALL, self.small)
        # where the plan touches the princess / picks a key
        gt = plan.get("goal_touch")
        if gt:
            p = m.pt(gt["x"], gt["y"])
            pg.draw.circle(surf, GOAL, p, 8, 2)
            self.text(surf, f"princess f{gt['frame']}", p[0] + 9, p[1] - 16, GOAL, self.small)
        for kt in plan.get("key_touches") or []:
            p = m.pt(kt["x"], kt["y"])
            pg.draw.circle(surf, KEY, p, 7, 2)
            self.text(surf, f"key {kt.get('key')} f{kt['frame']}", p[0] + 8, p[1] + 2, KEY, self.small)
        # the nearest hazards: path point at that frame -> the closest box of that kind then
        lv = scene.level
        for nh in (plan.get("nearest") or [])[:NEAREST_DRAWN]:
            f = int(nh.get("frame", 0))
            if not path:
                break
            px, py = path[max(0, min(f, len(path) - 1))]
            tgt = None
            hid, kind = str(nh.get("id")), nh.get("kind")
            if hid == "out" or kind == "out of level":
                if lv is not None:
                    tgt = edge_closest(px, py, lv.alive_box)
            else:
                boxes = scene.boxes_at(int(plan.get("frame", 0)) + f) if scene.boxes_at else None
                want = KIND_OF_NAME.get(kind)
                if hid.startswith("death@"):
                    try:
                        dx, dy = (float(v) for v in hid[6:].split(","))
                        tgt = box_closest(px, py, [dx, dy, MEMORY_HALF, MEMORY_HALF, 0.0])
                    except ValueError:
                        tgt = None
                elif boxes:
                    cand = [b for b in boxes if want is None or len(b) < 6 or b[5] == want] or boxes
                    tgt = min((box_closest(px, py, b) for b in cand),
                              key=lambda q: math.hypot(q[0] - px, q[1] - py))
            c = float(nh.get("clearance", 0.0))
            col = BAD if c < 10 else WARN if c < 30 else (150, 200, 150)
            a = m.pt(px, py)
            if tgt is not None:
                b = m.pt(*tgt)
                pg.draw.line(surf, col, a, b, 1)
                pg.draw.circle(surf, col, b, 2)
                lx, ly = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            else:
                lx, ly = a
            pg.draw.circle(surf, col, a, 3, 1)
            self.text(surf, f"{hid} {c:.0f}u f{f}", lx + 4, ly - 6, col, self.small)
        # predictions: this plan's (hollow), and the matured ones against reality
        for h, pr in (plan.get("predictions") or {}).items():
            if pr is None:
                continue
            p = m.pt(*pr)
            pg.draw.circle(surf, PRED, p, 4, 1)
            self.text(surf, f"+{h}", p[0] + 5, p[1] + 3, PRED, self.small)
        for r in matured_predictions(scene.history.get(sel, []), lambda f: scene.real_at(sel, f), scene.frame):
            p = m.pt(*r["pred"])
            pg.draw.circle(surf, PRED, p, 3, 1)
            if r["real"] is not None:
                q = m.pt(*r["real"])
                col = GOOD if r["error"] <= r["threshold"] else BAD
                pg.draw.line(surf, col, p, q, 1)
                lab = self.small.render(f"+{r['h']}: {r['error']:.1f}u", True, col)
                surf.blit(lab, (p[0] - lab.get_width() - 5, p[1] - 7))

    def draw_hover(self, surf):
        if not self.mouse or not self.alt_hits:
            return
        mx, my = self.mouse
        best, bd = None, HOVER_PX
        for pts, alt in self.alt_hits:
            for k in range(len(pts) - 1):
                d = seg_dist(mx, my, pts[k], pts[k + 1])
                if d < bd:
                    best, bd = alt, d
        if best is None:
            return
        lines = [" > ".join(best.get("moves") or []) + ("  VETOED" if best.get("vetoed") else ""),
                 (f"estimate {fmt(best.get('estimate'), '.0f')} frames  " if "estimate" in best else
                  f"total {fmt(best.get('total'), '+.2f')}  ") + f"progress {fmt(best.get('progress'), '+.2f')}  "
                 f"stage {best.get('stage')}"] + wrap(str(best.get("reason", "")), 70)
        w = max(self.small.size(s)[0] for s in lines) + 12
        h = 16 * len(lines) + 8
        x, y = min(mx + 14, surf.get_width() - PANEL_W - w - 4), my + 14
        self.pg.draw.rect(surf, (10, 10, 14), (x, y, w, h))
        self.pg.draw.rect(surf, ALT_VETO if best.get("vetoed") else ALT, (x, y, w, h), 1)
        for k, s in enumerate(lines):
            self.text(surf, s, x + 6, y + 4 + 16 * k, TEXT, self.small)

    def draw_timeline(self, surf, scene: Scene, rect):
        pg = self.pg
        x, y, w, h = rect
        self.timeline_rect = pg.Rect(rect)
        pg.draw.rect(surf, (30, 32, 44), rect)
        tl = scene.timeline
        n = max(1, tl["n"] - 1)
        drawn = {}
        for s, switched in tl.get("decisions") or []:                 # one tick per pixel column
            px = int(x + 6 + (w - 12) * s / n)
            drawn[px] = drawn.get(px, False) or switched
        for px, switched in drawn.items():
            pg.draw.line(surf, WARN if switched else FAINT, (px, y + h - 7), (px, y + h - 2), 1)
        for s, col in tl["marks"]:
            px = x + 6 + (w - 12) * s / n
            pg.draw.line(surf, col, (px, y + 4), (px, y + h - 4), 1)
        cx = x + 6 + (w - 12) * tl["idx"] / n
        pg.draw.line(surf, TEXT, (cx, y + 1), (cx, y + h - 1), 2)

    # -- side panel --
    def draw_panel(self, surf, scene: Scene, sel, x, w):
        pg = self.pg
        y = 6
        bottom = surf.get_height() - 4

        def line(s, col=TEXT, font=None, step=17):
            nonlocal y
            font = font or self.font
            n = max(20, w // max(1, font.size("m")[0]))
            for k, part in enumerate(wrap(s, n - 2)):
                if y + step > bottom:
                    return
                self.text(surf, part if k == 0 else "  " + part, x, y, col, font)
                y += step

        def small(s, col=DIM):
            line(s, col, self.small, 15)

        for s in scene.status:
            if s:
                small(s, DIM)
        line(f"{scene.level_name or '?'}  wave {scene.wave}  frame {scene.frame}", TEXT, self.big, 22)
        plan = scene.plans.get(sel) if sel is not None else None
        line(f"agent {sel if sel is not None else '-'}" + (f"  ({scene.planned.index(sel) + 1} of {len(scene.planned)} "
                                                           f"planned, Tab)" if sel in scene.planned else ""), SELECT)
        if plan is None:
            line("no decision yet", DIM)
        else:
            age = scene.frame - int(plan.get("frame", 0))
            line(f"decision at f{plan.get('frame')} ({age} frames ago)" + ("  [previous plan, shifted]"
                                                                           if plan.get("from_prev") else ""), DIM)
            trig, swt = decision_lines(plan)
            sw = plan.get("switch")
            small(trig, TEXT)
            small(swt, WARN if sw and not sw.get("kept") else DIM)
            line(plan_name(plan), PLAN)
            for t in score_lines(plan):
                small(t, TEXT)
            veto = plan.get("veto")
            if veto:
                small(f"VETO: {veto.get('kind')} {veto.get('id')} at f{veto.get('frame')} (every option hit something)",
                      BAD)
            else:
                small(f"no veto; soft risk {fmt(plan.get('soft_risk'), '.2f')}"
                      + (f" at f{plan.get('soft_frame')}" if (plan.get("soft_frame") or -1) >= 0 else "")
                      + f"; death model {fmt(plan.get('death_model'), '.2f')}", GOOD if not plan.get("soft_risk") else WARN)
            lands = ", ".join(f"f{d['frame']} {d.get('on')} ({d['x']:.0f},{d['y']:.0f})" for d in plan.get("landings") or [])
            wjs = ", ".join(f"f{d['frame']} {d.get('side')} ({d['x']:.0f},{d['y']:.0f})" for d in plan.get("wall_jumps") or [])
            gt = plan.get("goal_touch")
            fm = plan.get("finish_model")
            small((f"princess: touched at f{gt['frame']} ({gt.get('members')} members)" if gt else
                   "princess: not reached") + (f"; finish model {fmt(fm, '.2f')}" if fm is not None else "")
                  + "".join(f"; key {k.get('key')} at f{k['frame']}" for k in plan.get("key_touches") or []),
                  GOAL if gt else DIM)
            small("lands: " + (lands or "-"), LAND)
            small("wall jumps: " + (wjs or "-"), WALL)
            near = plan.get("nearest") or []
            small("nearest: " + ("; ".join(f"{d.get('id')} ({d.get('kind')}) {fmt(d.get('clearance'), '.0f')}u at "
                                          f"f{d.get('frame')}" for d in near) or "-"), TEXT)
            mat = matured_predictions(scene.history.get(sel, []), lambda f: scene.real_at(sel, f), scene.frame)
            if mat:
                small("prediction vs reality: " + "  ".join(
                    f"+{r['h']} {fmt(r['error'], '.1f')}u" + ("!" if r["error"] is not None and r["error"] > r["threshold"]
                                                               else "") for r in mat), TEXT)
            y += 4
            alts = plan.get("alternatives") or []
            line(f"alternatives ({len(alts)})" + ("" if self.show_alts else "  [hidden: A]"), TEXT)
            for k, a in enumerate(alts[:6]):
                col = ALT_VETO if a.get("vetoed") else ALT
                val = f"{fmt(a.get('estimate'), '.0f')} f" if "estimate" in a else fmt(a.get('total'), '+.2f')
                small(f"{k + 1}. {' > '.join(a.get('moves') or [])}  {val} [{a.get('stage')}]", col)
                small("   " + str(a.get("reason", "")), FAINT)
            y += 4
            tm, ct = plan.get("timing") or {}, plan.get("counts") or {}
            small("ms: " + "  ".join(f"{k} {fmt(tm.get(k), '.0f')}" for k in ("stage1", "stage2", "final", "completion", "total")),
                  TEXT)
            small(counts_text(ct), DIM)
        y += 4
        line("events", TEXT)
        for e in scene.events:
            small(event_text(e), event_colour(e))
        y += 4
        st = scene.stats
        line("running", TEXT)
        small(f"decisions {st.get('decisions', 0)}, ms mean {fmt(st.get('ms_mean'), '.0f')} / 95% "
              f"{fmt(st.get('ms_95'), '.0f')}; switched {st.get('switched', 0)}, kept {st.get('kept', 0)}", TEXT)
        small(f"surprises {st.get('surprises', 0)} ({fmt(st.get('surprises_per_100'), '.1f')} per 100 decisions), "
              f"deaths {st.get('deaths', 0)} (unexplained {st.get('unexplained', 0)}), failed moves "
              f"{st.get('failed', 0)}", TEXT)
        for s in finish_lines(st, scene.level_name):
            small(s, GOOD)
        small(f"future boxes: {scene.future_note}" + ("" if self.show_future else " [hidden: F]") +
              "   H: help", FAINT)

    def draw_help(self, surf, scene: Scene):
        lines = ["Keys", "Tab   next planned agent", "H     this help", "A     alternatives on / off",
                 "F     future hazards (+8 / +16) on / off", "Z     zoom on the selected agent on / off",
                 "wheel or [ ]  zoom in / out", "S     screenshot (PNG)", "Esc Q quit"]
        if scene.mode == "replay":
            lines += ["Space play / pause", "Left / Right  one step back / forward", "+ / -  speed",
                      "N     next event", "D     next surprise or unexplained death", "Home / End  first / last step",
                      "click the timeline to seek"]
        lines += ["", "Hover a thin line: why that alternative lost.",
                  "Clearance lines: path point at the hazard's closest frame -> the box (planner's number).",
                  "o: prediction; line to the real position, green within the surprise threshold."]
        w = max(self.font.size(s)[0] for s in lines) + 30
        h = 19 * len(lines) + 20
        x, y = 40, 40
        s = self.pg.Surface((w, h), self.pg.SRCALPHA)
        s.fill((8, 8, 12, 230))
        surf.blit(s, (x, y))
        for k, t in enumerate(lines):
            self.text(surf, t, x + 15, y + 10 + 19 * k, TEXT if k else SELECT)


def zoom_step(z: float, d: int) -> float:
    return min(MAX_ZOOM, max(1.0, z * (1.25 if d > 0 else 0.8)))


def counts_text(ct: dict) -> str:
    out = []
    s1 = ct.get("stage1") or {}
    if s1:
        cut = s1.get("complete", s1.get("candidates", 0)) < s1.get("candidates", 0)
        out.append(f"stage 1: {s1.get('candidates')} moves, {s1.get('steps')} steps" + (" CUT SHORT" if cut else ""))
    s2 = ct.get("stage2") or {}
    if s2:
        if "skipped" in s2:
            out.append(f"stage 2 SKIPPED ({s2['skipped']})")
        else:
            cut = s2.get("steps", 0) < s2.get("of_steps", 0)
            out.append(f"stage 2: {s2.get('candidates')} pairs, {s2.get('steps')}/{s2.get('of_steps')} steps"
                       + (" CUT SHORT" if cut else ""))
    f = ct.get("final") or {}
    if f:
        out.append(f"final: {f.get('candidates')} x {f.get('members')} members, {f.get('vetoed')} vetoed")
    return "; ".join(out) or "counts: -"


def fmt(v, spec: str) -> str:
    try:
        return format(float(v), spec) if v is not None else "-"
    except (TypeError, ValueError):
        return "-"


def wrap(s: str, n: int) -> list[str]:
    n = max(10, int(n))
    words, out, cur = str(s).split(" "), [], ""
    for wd in words:
        while len(wd) > n:
            if cur:
                out.append(cur)
                cur = ""
            out.append(wd[:n])
            wd = wd[n:]
        if not cur:
            cur = wd
        elif len(cur) + 1 + len(wd) <= n:
            cur += " " + wd
        else:
            out.append(cur)
            cur = wd
    if cur or not out:
        out.append(cur)
    return out


def seg_dist(px, py, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L <= 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


# ---- the window ------------------------------------------------------------------------------------------------
class App:
    def __init__(self, source, size=(1500, 900), speed=1.0, agent=None):
        import pygame
        self.pg = pygame
        self.src = source
        self.replay = isinstance(source, ReplaySource)
        self.size = size
        self.speed = speed
        self.idx = 0
        self.playing = False
        self.acc = 0.0
        self.r = Renderer()
        self.r.sel = agent
        self.scene = None
        self.note, self.note_t = None, 0.0

    def current(self) -> Scene:
        if self.replay:
            return self.src.scene(self.idx, self.speed, self.playing)
        self.src.poll()
        return self.src.scene()

    def screenshot(self, surf):
        sc = self.scene
        d = self.src.run_dir if self.replay else os.getcwd()
        name = f"viewer_{(sc.level_name or 'level').replace('/', '_')}_w{sc.wave}_f{sc.frame}_{time.strftime('%H%M%S')}.png"
        path = os.path.join(d, name)
        self.pg.image.save(surf, path)
        self.note, self.note_t = f"saved {path}", time.time()
        print(self.note, flush=True)

    def key(self, e, surf):
        pg = self.pg
        k = e.key
        if k in (pg.K_ESCAPE, pg.K_q):
            return False
        if k == pg.K_TAB:
            self.r.next_agent(self.scene)
        elif k == pg.K_h:
            self.r.show_help = not self.r.show_help
        elif k == pg.K_a:
            self.r.show_alts = not self.r.show_alts
        elif k == pg.K_f:
            self.r.show_future = not self.r.show_future
        elif k == pg.K_s:
            self.screenshot(surf)
        elif k == pg.K_z:
            self.r.follow = not self.r.follow
            if self.r.follow and self.r.zoom <= 1.0:
                self.r.zoom = FOLLOW_ZOOM
            elif not self.r.follow:
                self.r.zoom = 1.0
        elif k in (pg.K_RIGHTBRACKET, pg.K_LEFTBRACKET):
            self.r.zoom = zoom_step(self.r.zoom, 1 if k == pg.K_RIGHTBRACKET else -1)
        elif self.replay:
            n = self.src.n
            if k == pg.K_SPACE:
                self.playing = not self.playing
                if self.playing and self.idx >= n - 1:
                    self.idx = 0
            elif k == pg.K_RIGHT:
                self.playing, self.idx = False, min(n - 1, self.idx + 1)
            elif k == pg.K_LEFT:
                self.playing, self.idx = False, max(0, self.idx - 1)
            elif k in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                self.speed = min(16.0, self.speed * 2)
            elif k in (pg.K_MINUS, pg.K_KP_MINUS):
                self.speed = max(1 / 16, self.speed / 2)
            elif k == pg.K_HOME:
                self.idx = 0
            elif k == pg.K_END:
                self.idx = n - 1
            elif k in (pg.K_n, pg.K_d):
                nxt = self.src.next_event(self.idx, None if k == pg.K_n else is_surprise_or_unexplained)
                if nxt is None:
                    self.note, self.note_t = "no later " + ("event" if k == pg.K_n else "surprise / unexplained death"), \
                        time.time()
                else:
                    self.playing, self.idx = False, nxt
        return True

    def run(self):
        pg = self.pg
        pg.init()
        screen = pg.display.set_mode(self.size, pg.RESIZABLE)
        pg.display.set_caption("Super Meat Boy planner - " + ("replay " + self.src.name if self.replay else "live"))
        clock = pg.time.Clock()
        running = True
        self.scene = self.current()
        while running:
            dt = clock.tick(FPS) / 1000.0
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running = False
                elif e.type == pg.KEYDOWN:
                    running = self.key(e, screen) and running
                elif e.type == pg.MOUSEMOTION:
                    self.r.mouse = e.pos
                elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1 and self.replay and self.r.timeline_rect \
                        and self.r.timeline_rect.collidepoint(e.pos):
                    tr = self.r.timeline_rect
                    self.idx = int(round((e.pos[0] - tr.x - 6) / max(1, tr.w - 12) * (self.src.n - 1)))
                    self.idx = max(0, min(self.src.n - 1, self.idx))
                elif e.type == pg.MOUSEWHEEL:
                    self.r.zoom = zoom_step(self.r.zoom, 1 if e.y > 0 else -1)
                elif e.type == pg.VIDEORESIZE:
                    screen = pg.display.set_mode((max(800, e.w), max(500, e.h)), pg.RESIZABLE)
            if self.replay and self.playing:
                self.acc += dt * HZ * self.speed
                step = int(self.acc)
                self.acc -= step
                self.idx = min(self.src.n - 1, self.idx + step)
                if self.idx >= self.src.n - 1:
                    self.playing = False
            self.scene = self.current()
            self.r.render(screen, self.scene)
            if self.note and time.time() - self.note_t < 3:
                self.r.text(screen, self.note, 10, screen.get_height() - 44, SELECT)
            pg.display.flip()
        if not self.replay:
            self.src.close()
        pg.quit()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default=f"{T.DEFAULT_ADDR[0]}:{T.DEFAULT_ADDR[1]}",
                    help="host:port to listen on for agent.py telemetry (default 127.0.0.1:47800)")
    ap.add_argument("--replay", default=None,
                    help="a run folder (runs/plan_<date>-<time>, the name agent.py prints) or 'latest'")
    ap.add_argument("--size", default="1500x900", help="window size WxH (default 1500x900)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = 60 game frames per second)")
    ap.add_argument("--agent", type=int, default=None, help="agent to follow first (default the first planned)")
    args = ap.parse_args(argv)
    w, _, h = args.size.lower().partition("x")
    size = (int(w), int(h))
    if args.replay:
        src = ReplaySource(resolve_run(args.replay))
        print(f"replay {src.run_dir}: {src.n} steps, {len(src.plans)} decisions, {len(src.events)} events", flush=True)
    else:
        addr = T.parse_addr(args.listen)
        if addr is None:
            raise SystemExit("--listen wants host:port")
        try:
            src = LiveSource(*addr)
        except OSError as e:
            raise SystemExit(f"cannot listen on {args.listen}: {e} (another viewer already running?)")
        print(f"listening on {addr[0]}:{addr[1]} for agent.py telemetry", flush=True)
    App(src, size, args.speed, args.agent).run()


if __name__ == "__main__":
    main()
