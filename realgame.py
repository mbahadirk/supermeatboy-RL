"""Train agents inside the real Super Meat Boy (personal use). No clone: every agent is a real Meat Boy
character in the running game (realgame.js), so it collides with, dies on and finishes exactly like Meat Boy.

Start the game, enter a level, then from the project root:

    python smbrl/realgame.py --agents 40                     # train on whatever level the game shows
    python smbrl/realgame.py --agents 40 --load smbrl/runs/real_4-1   # continue a saved agent
    python smbrl/realgame.py --agents 8 --test               # no learning: scripted inputs (checks the mod)

Waves: all agents spawn together after the level (saws, platforms, crumble blocks) is reset; each has one
life and waits where it died; the next wave starts when every agent is dead or finished, or after
--wave-seconds. The camera follows the agent furthest along its path. The timer sign shows the best finishing
time on this level. S in the game (window focused): save the agent and finish the level the game's way;
training continues on the next level the game loads. Ctrl+C (or a file runs/<name>/STOP): save and detach.
At most 40 agents: the game keeps a pool of 40 extra Meat Boys per level, and the agents are those.

Reward paths (realpath.py): when a level loads, up to --paths different short routes from the spawn to the
goal that stay clear of every solid tile and lethal object (sizes from the game itself; moving hazards
everywhere they were seen) are found and drawn into the game (green, blue, orange). Agents are split over
the routes; each is rewarded for new progress along its own route (only points it can see count), lightly
for staying near it, and sees the direction of its route ahead.

Phases per level: FINISH until some agent reaches the goal (progress and the finish matter most), then
SPEED: every frame costs more, a finish pays by how close it is to the best time, a new record pays extra,
and half the agents move to the route of the fastest finisher.

The monitor window (monitor.py; --no-monitor to skip) shows the level, routes and agents, the leader's
policy output, learning curves (finishes, progress, fitness, time, losses, entropy, KL, explained variance)
and the policy's weights.

Saved in runs/<name>/: model.pt (latest), levels/<level>/best.pt + best.json (fastest finish).
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from realpath import Planner, fitness, PATH_RGB

HERE = os.path.dirname(os.path.abspath(__file__))
JS = os.path.join(HERE, "realgame.js")
GAME_EXE = "SuperMeatBoy.exe"
HZ = 60
TILE = 20.0
MAX_AGENTS = 40

# actions: (left, right, jump, run)
ACTIONS = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0),
           (0, 1, 0, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 0, 1, 1)]
K = 5                                   # local map radius (cells): (2K+1)^2 values
N_STATE = 10
N_PATH = 6                              # own route: points 2 and 6 tiles ahead, progress, distance from it
OBS = N_STATE + (2 * K + 1) ** 2 + N_PATH
C_EMPTY, C_SOLID, C_HAZARD, C_GOAL = 0.0, 1.0, -1.0, 0.75

# rewards
R_PROGRESS = 1.0       # per tile of new progress along the agent's route (best so far this life)
R_TETHER = -0.003      # per tile away from the route (up to 6 tiles), every frame
R_DEATH = -5.0
R_WIN = 50.0
R_FAST = 20.0          # FINISH phase: + this x (1 - finish time / wave length)
R_STEP = -0.002        # FINISH phase, every frame
R_SPEED = 60.0         # SPEED phase: + this x (best time / finish time), at most 1.5x
R_RECORD = 25.0        # SPEED phase: a new best time
R_STEP_SPEED = -0.01   # SPEED phase, every frame


# ---- level ------------------------------------------------------------------------------------------
class Level:
    """Static facts the game reported for the current level (live tile layer 0, goals, spawn)."""

    def __init__(self, info: dict, tiles: bytes):
        self.name = info["name"]
        self.W, self.H, self.ox, self.oy = info["W"], info["H"], info["ox"], info["oy"]
        t = np.frombuffer(tiles, dtype="<u4").reshape(self.H, self.W)        # row 0 = bottom
        tid = t & 0xff
        self.solid = ((tid > 0) & (tid < 100)) | (((t >> 8) & 1).astype(bool) & (tid > 0) & (tid <= 100))
        # tiles a reward path cannot go through: full blocks only (slopes and half blocks have room,
        # crumble blocks - ids 64, 96, 97, 99, 100 - break)
        shaped = ((t >> 18) & 3) != 0
        crumble = np.isin(tid, (64, 96, 97, 99, 100))
        self.path_solid = (tid > 0) & (tid <= 100) & ~shaped & ~crumble
        self.goals = [tuple(g) for g in info["goals"]]
        self.spawn = tuple(info["spawn"])
        self.base = np.where(self.solid, C_SOLID, C_EMPTY).astype(np.float32)
        for gx, gy in self.goals:
            c, r = self.cell(gx, gy)
            if 0 <= c < self.W and 0 <= r < self.H:
                self.base[r, c] = C_GOAL
        self.dist = self._bfs()
        self.diag = float(np.hypot(self.W, self.H))

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.ox) // TILE), int((y - self.oy) // TILE)

    def _bfs(self) -> np.ndarray:
        """Path length (tiles) from each open cell to the nearest goal, 4-connected; inf where unreachable."""
        d = np.full((self.H, self.W), np.inf, dtype=np.float32)
        q = deque()
        for gx, gy in self.goals:
            c, r = self.cell(gx, gy)
            if 0 <= c < self.W and 0 <= r < self.H:
                d[r, c] = 0.0
                q.append((c, r))
        while q:
            c, r = q.popleft()
            for nc, nr in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                if 0 <= nc < self.W and 0 <= nr < self.H and d[nr, nc] == np.inf and not self.solid[nr, nc]:
                    d[nr, nc] = d[r, c] + 1.0
                    q.append((nc, nr))
        return d

    def goal_dist(self, x: float, y: float) -> float:
        c, r = self.cell(x, y)
        if 0 <= c < self.W and 0 <= r < self.H and np.isfinite(self.dist[r, c]):
            return float(self.dist[r, c])
        gx, gy = min(self.goals, key=lambda g: (g[0] - x) ** 2 + (g[1] - y) ** 2) if self.goals else self.spawn
        return float(np.hypot(gx - x, gy - y) / TILE) + self.diag   # off the known paths: worse than any path

    def grid_with_hazards(self, circles: list) -> np.ndarray:
        """Cell codes now: tiles, goals, and the lethal pieces at their current positions."""
        g = self.base.copy()
        for x, y, radius in circles:
            rad = max(0.5, radius / TILE)
            c, r = self.cell(x, y)
            k = int(np.ceil(rad))
            r0, r1, c0, c1 = max(0, r - k), min(self.H, r + k + 1), max(0, c - k), min(self.W, c + k + 1)
            if r0 >= r1 or c0 >= c1:
                continue
            rr, cc = np.mgrid[r0:r1, c0:c1]
            m = (np.hypot(rr - r, cc - c) <= rad) & (g[r0:r1, c0:c1] != C_SOLID)
            g[r0:r1, c0:c1][m] = C_HAZARD
        return g


def hazard_circles(hz: list) -> list[tuple[float, float, float]]:
    """Lethal pieces [x, y, size w, size h, scale] (realgame.js) as circles: the game first tests a circle around
    the piece of half its scaled diagonal (FUN_004cd350 with sizes from FUN_004ca860)."""
    out = []
    for x, y, w, h, s in hz:
        s = abs(s or 1.0)
        r = 0.5 * s * float(np.hypot(w, h)) if (w > 0 or h > 0) else 18.0 * s
        out.append((x, y, r))
    return out


def argb(rgb, alpha=0xD8) -> int:
    r, g, b = rgb
    return (alpha << 24) | (r << 16) | (g << 8) | b


def other_trainers() -> list[str]:
    """PIDs of other realgame.py processes (two trainers in one game break each other)."""
    import subprocess
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                              "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    me = str(os.getpid())
    return [ln.split("|", 1)[0] for ln in out.splitlines()
            if "realgame.py" in ln and "|" in ln and ln.split("|", 1)[0].strip() != me]


def identify_level(info: dict, tiles: bytes) -> str | None:
    """Name of the level the game shows when attached mid-level: the extracted level file (tools/smbl_extract.py)
    with the same size, origin and collision tiles."""
    d = os.path.join(HERE, "imported", "smb_static")
    if not os.path.isdir(d):
        return None
    t = np.frombuffer(tiles, dtype="<u4").reshape(info["H"], info["W"]) & 0xff
    live = ((t > 0) & (t < 100)).astype(np.uint8)
    try:   # prefer the files the game's chapters use (there are editor copies such as 4-1_TEST)
        with open(os.path.join(d, "chapters.json")) as fh:
            used = [c["file"] for c in json.load(fh)]
    except (OSError, ValueError):
        used = []
    names = sorted(f[:-len("_semantic.json")] for f in os.listdir(d) if f.endswith("_semantic.json"))
    names.sort(key=lambda n: (n not in used, len(n), n))
    for n in names:
        f = n + "_semantic.json"
        with open(os.path.join(d, f)) as fh:
            j = json.load(fh)
        if (j["W"], j["H"]) != (info["W"], info["H"]) or tuple(j["origin"]) != (info["ox"], info["oy"]):
            continue
        tid = np.array(j["solid"], dtype=np.uint8)
        if (tid == live).mean() > 0.98:
            return n
    return None


def observe(lv: Level, grid: np.ndarray, s: list, path, prog: float, off: float) -> np.ndarray:
    x, y, vx, vy, flags, _state, _st = s
    gx, gy = lv.goals[0] if lv.goals else lv.spawn
    d = lv.goal_dist(x, y)
    c, r = lv.cell(x, y)
    pad = np.pad(grid, K, constant_values=C_SOLID)
    rr, cc = min(max(r, -K), lv.H + K - 1) + K, min(max(c, -K), lv.W + K - 1) + K
    patch = pad[rr - K:rr + K + 1, cc - K:cc + K + 1]
    if patch.shape != (2 * K + 1, 2 * K + 1):
        patch = np.full((2 * K + 1, 2 * K + 1), C_SOLID, dtype=np.float32)
    vec = np.array([
        (x - lv.ox) / (lv.W * TILE), (y - lv.oy) / (lv.H * TILE), vx / 400.0, vy / 600.0,
        0.0 if flags & 2 else 1.0,                 # on the ground (bit 1 = airborne)
        1.0 if flags & 0x200 else 0.0,             # on a wall
        -1.0 if flags & 1 else 1.0,                # facing
        (gx - x) / (lv.W * TILE), (gy - y) / (lv.H * TILE), min(d / lv.diag, 2.0),
    ], dtype=np.float32)
    if path is not None:                           # where the agent's own route goes next
        ax, ay = path.point_at(prog + 2 * TILE)
        bx, by = path.point_at(prog + 6 * TILE)
        pv = np.clip(np.array([(ax - x) / (10 * TILE), (ay - y) / (10 * TILE), (bx - x) / (10 * TILE),
                               (by - y) / (10 * TILE), prog / path.length, off / (5 * TILE)], np.float32), -2, 2)
    else:
        pv = np.zeros(N_PATH, np.float32)
    return np.concatenate([vec, patch[::-1].ravel(), pv])   # patch rows top-down


# ---- policy + PPO ---------------------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self, obs=OBS, n=len(ACTIONS), h=256):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n))
        self.vf = nn.Sequential(nn.Linear(obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def dist(self, o):
        return torch.distributions.Categorical(logits=self.pi(o))


def load_policy(policy: Policy, path: str):
    """Load a saved model; one from before the route inputs existed keeps its weights for the old inputs and
    starts the new input columns small."""
    sd = torch.load(path, map_location="cpu")["policy"]
    own = policy.state_dict()
    widened = False
    for k, v in sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif v.dim() == 2 and own[k].shape[0] == v.shape[0] and v.shape[1] < own[k].shape[1]:
            own[k][:, :v.shape[1]] = v
            own[k][:, v.shape[1]:] *= 0.1
            widened = True
    policy.load_state_dict(own)
    return widened


class PPO:
    def __init__(self, policy: Policy, lr=3e-4, gamma=0.995, lam=0.95, clip=0.2, epochs=4, mb=2048, ent=0.01):
        self.p = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma, self.lam, self.clip, self.epochs, self.mb, self.ent = gamma, lam, clip, epochs, mb, ent
        self.batch = []          # finished trajectories: dicts of arrays
        self.steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy=False):
        o = torch.as_tensor(obs)
        d = self.p.dist(o)
        a = d.probs.argmax(-1) if greedy else d.sample()
        return a.numpy(), d.log_prob(a).numpy(), self.p.vf(o).squeeze(-1).numpy(), d.probs.numpy()

    def add(self, traj: dict, last_value: float):
        r, v, done = np.array(traj["r"], np.float32), np.array(traj["v"], np.float32), traj["done"]
        n = len(r)
        adv = np.zeros(n, np.float32)
        nxt, gae = (0.0 if done else last_value), 0.0
        for t in range(n - 1, -1, -1):
            delta = r[t] + self.gamma * nxt - v[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t], nxt = gae, v[t]
        self.batch.append({"o": np.array(traj["o"], np.float32), "a": np.array(traj["a"]),
                           "lp": np.array(traj["lp"], np.float32), "adv": adv, "ret": adv + v, "v": v})
        self.steps += n

    def update(self) -> dict:
        cat = lambda k: torch.as_tensor(np.concatenate([b[k] for b in self.batch]))
        o, a, lp0, adv, ret, v0 = cat("o"), cat("a"), cat("lp"), cat("adv"), cat("ret"), cat("v")
        ev = float(1.0 - torch.var(ret - v0) / (torch.var(ret) + 1e-8))      # how much of the return the critic explained
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(a)
        acc = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": [], "gnorm": []}
        for _ in range(self.epochs):
            for idx in torch.randperm(n).split(self.mb):
                d = self.p.dist(o[idx])
                logr = d.log_prob(a[idx]) - lp0[idx]
                ratio = torch.exp(logr)
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                vl = ((self.p.vf(o[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                en = d.entropy().mean()
                loss = pg + 0.5 * vl - self.ent * en
                self.opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.p.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    acc["pg"].append(pg.item()); acc["vf"].append(vl.item()); acc["ent"].append(en.item())
                    acc["kl"].append(((ratio - 1) - logr).mean().item())
                    acc["clip"].append(((ratio - 1).abs() > self.clip).float().mean().item())
                    acc["gnorm"].append(float(gn))
        self.batch = []
        return {k: float(np.mean(v)) for k, v in acc.items()} | {"ev": ev, "samples": n}


# ---- session --------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, args):
        self.args = args
        self.run_dir = os.path.join(HERE, "runs", args.name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.policy = Policy()
        if args.load:
            p = args.load if args.load.endswith(".pt") else os.path.join(args.load, "model.pt")
            widened = load_policy(self.policy, p)
            print(f"loaded {p}" + (" (older model: the route inputs start untrained)" if widened else ""))
        self.ppo = PPO(self.policy)
        self.level: Level | None = None
        self.traj, self.best_d, self.acts, self.alive = [], [], [], []
        self.best_time = None
        self.waves = 0
        self.t0 = time.time()
        self.script = None
        self.stopping = False
        self.stopped = threading.Event()
        # shown by the monitor
        self.paths, self.path_of, self.circles, self.last_states = [], [], [], []
        self.phase, self.best_route, self.leader, self.wave_no, self.frame_no = "finish", None, 0, 0, 0
        self.leader_probs, self.leader_value = None, None
        self.history, self.updates = deque(maxlen=400), deque(maxlen=400)
        self.weights, self.wver = None, 0
        self.game_paths_on = True
        self.snapshot_weights()

    # level / waves
    def on_info(self, info: dict, tiles: bytes):
        if info["W"] * info["H"] == 0 or len(tiles or b"") != info["W"] * info["H"] * 4:
            print(f"[level] ignoring invalid level data {info['W']}x{info['H']}", flush=True)
            return
        if info["name"] in (None, "unknown"):
            info["name"] = identify_level(info, tiles) or "unknown"
        self.level = Level(info, tiles)
        self.best_time = None
        self.phase, self.best_route = "finish", None
        self.paths, self.hz_start, self.moving, self.sweep, self.path_dirty = [], None, set(), set(), True
        self.paths_built, self.rebuilds, self.built_wave = False, 0, 0
        self.best_fit = None
        n = info["agents"]
        print(f"[level] {info['name']}: {self.level.W}x{self.level.H}, {len(self.level.goals)} goal(s), "
              f"{info['lethal']} lethal pieces, {n} agents", flush=True)
        self.new_wave(n)
        if "hz" in info:                     # the game waits for this (no agent moves meanwhile)
            self.circles = hazard_circles(info["hz"])
            self.track_hazards(self.circles)
            self.build_paths(reset=False)

    def assign_paths(self, n: int) -> list[int]:
        k = len(self.paths)
        if k == 0:
            return [0] * n
        if self.phase == "speed" and self.best_route is not None and self.best_route < k:
            return [self.best_route if i % 2 == 0 else (i // 2) % k for i in range(n)]
        return [i % k for i in range(n)]

    def new_wave(self, n: int):
        self.traj = [{"o": [], "a": [], "lp": [], "v": [], "r": [], "done": False} for _ in range(n)]
        self.best_d = [None] * n
        self.best_s = [None] * n       # best progress along the agent's route this life (game units)
        self.cur_s = [0.0] * n
        self.cur_off = [0.0] * n
        self.frames = [0] * n
        self.fit = [None] * n
        self.acts = [[] for _ in range(n)]
        self.alive = [True] * n
        self.path_of = self.assign_paths(n)

    def route(self, i: int):
        return self.paths[self.path_of[i]] if self.paths and i < len(self.path_of) else None

    def on_frame(self, m: dict) -> dict:
        lv, states = self.level, m["s"]
        if lv is None:
            return {"a": [[0, 0, 0, 0]] * len(states), "leader": 0}
        n = len(states)
        if len(self.traj) != n:
            self.new_wave(n)
        circles = hazard_circles(m["hz"])
        self.circles = circles
        self.track_hazards(circles)
        if not self.paths_built:             # (normally built when the level info came)
            self.build_paths(reset=m["frame"] != 0)
        grid = lv.grid_with_hazards(circles)
        self.last_states, self.frame_no, self.wave_no = states, m["frame"], m["wave"]
        if self.args.debug_spawn and m["frame"] in (0, 1, 30):
            sx, sy = lv.spawn
            far = [(i, s) for i, s in enumerate(states) if abs(s[0] - sx) > 30 or abs(s[1] - sy) > 30]
            print(f"  [spawn f{m['frame']}] spawn=({sx:.0f},{sy:.0f}) {len(far)} away: " + " ".join(
                f"a{i}=({s[0]:.0f},{s[1]:.0f} st{s[5]} fl{s[4]:x})" for i, s in far[:6]), flush=True)
        if self.args.test and m["frame"] % 120 == 0:
            print(f"  f{m['frame']}: " + " ".join(f"({s[0]:.0f},{s[1]:.0f} v{s[2]:.0f},{s[3]:.0f} st{s[5]} {s[6]})"
                                               for s in states[:4]), flush=True)
        # where every agent is on its route (only points it can see count)
        for i, s in enumerate(states):
            if not self.alive[i]:
                continue
            p = self.route(i)
            if p is None:
                continue
            prog, off = p.project(s[0], s[1], self.best_s[i] if self.best_s[i] is not None else 0.0)
            self.cur_s[i], self.cur_off[i] = prog, off
        # finish the previous step: rewards for what happened since the last frame
        step_cost = R_STEP_SPEED if self.phase == "speed" else R_STEP
        for i, s in enumerate(states):
            tr = self.traj[i]
            if not self.alive[i] or not tr["a"]:
                continue
            x, y, st = s[0], s[1], s[6]
            r = step_cost
            self.frames[i] += 1
            p = self.route(i)
            if p is not None:
                prog = self.cur_s[i]
                if self.best_s[i] is None:
                    self.best_s[i] = prog
                elif prog > self.best_s[i]:
                    r += R_PROGRESS * (prog - self.best_s[i]) / TILE
                    self.best_s[i] = prog
                r += R_TETHER * min(self.cur_off[i] / TILE, 6.0)
            else:                                              # no route: closest path distance to the goal
                d = lv.goal_dist(x, y)
                if self.best_d[i] is None or d < self.best_d[i]:
                    r += R_PROGRESS * ((self.best_d[i] if self.best_d[i] is not None else d) - d)
                    self.best_d[i] = d
            if st == 1:
                r += R_DEATH
            elif st == 2:
                r += self.finish_reward(m["frame"])
            tr["r"].append(r)
            if st != 0:
                tr["done"] = True
                self.alive[i] = False
                self.ppo.add(tr, 0.0)
                self.fit[i] = self.agent_fitness(i, won=st == 2, dead=st == 1)
        live = [i for i, s in enumerate(states) if s[6] == 0 and self.alive[i]]
        acts = [[0, 0, 0, 0]] * n
        leader = 0
        if live:
            obs = np.stack([observe(lv, grid, states[i], self.route(i), self.cur_s[i], self.cur_off[i]) for i in live])
            if self.args.test:
                a = np.array([self.scripted(m["frame"], i) for i in live])
                lp = v = np.zeros(len(live), np.float32)
                probs = None
            else:
                a, lp, v, probs = self.ppo.act(obs)
            for j, i in enumerate(live):
                tr = self.traj[i]
                if self.best_d[i] is None:
                    self.best_d[i] = lv.goal_dist(states[i][0], states[i][1])
                tr["o"].append(obs[j]); tr["a"].append(int(a[j])); tr["lp"].append(float(lp[j])); tr["v"].append(float(v[j]))
                acts[i] = list(ACTIONS[int(a[j])])
                self.acts[i].append(int(a[j]))
            if self.paths:
                leader = max(live, key=lambda i: self.cur_s[i] / self.route(i).length)
            else:
                leader = min(live, key=lambda i: lv.goal_dist(states[i][0], states[i][1]))
            j = live.index(leader)
            self.leader_probs = probs[j] if probs is not None else None
            self.leader_value = float(v[j])
        won = [i for i, s in enumerate(states) if s[6] == 2]
        if won:
            leader = won[0]
        self.leader = leader
        return {"a": acts, "leader": leader}

    def finish_reward(self, frame: int) -> float:
        t = (frame + 1) / HZ
        if self.phase == "speed" and self.best_time:
            return R_WIN + R_SPEED * min(1.5, self.best_time / t) + (R_RECORD if t < self.best_time - 1e-6 else 0.0)
        return R_WIN + R_FAST * max(0.0, 1.0 - frame / (self.args.wave_seconds * HZ))

    def scripted(self, f: int, i: int) -> int:
        """--test: run right, jumping at agent-specific intervals (checks deaths, walls, goal, waves)."""
        period = 20 + 7 * i
        return 7 if (f % period) < 10 else 6

    def on_wave_end(self, m: dict):
        lv = self.level
        self.waves += 1
        n = len(self.traj)
        for i in range(n):           # outcomes of the wave's last frame; agents alive at the time limit are cut
            tr = self.traj[i]
            if not (self.alive[i] and tr["a"]):
                continue
            st = m["status"][i] if i < len(m["status"]) else 0
            self.fit[i] = self.agent_fitness(i, won=st == 2, dead=st == 1)
            step_cost = R_STEP_SPEED if self.phase == "speed" else R_STEP
            if st == 1:
                tr["r"].append(step_cost + R_DEATH)
                tr["done"] = True
                self.ppo.add(tr, 0.0)
            elif st == 2:
                tr["r"].append(step_cost + self.finish_reward(m["finish"][i] if m["finish"][i] >= 0 else m["frame"]))
                tr["done"] = True
                self.ppo.add(tr, 0.0)
            else:
                tr["r"].append(step_cost)
                with torch.no_grad():
                    lv_ = float(self.policy.vf(torch.as_tensor(tr["o"][-1])).item())
                self.ppo.add(tr, lv_)
        rets = [sum(tr["r"]) for tr in self.traj if tr["r"]]
        fin = [(f, i) for i, (st, f) in enumerate(zip(m["status"], m["finish"])) if st == 2 and f >= 0]
        dead = sum(1 for st in m["status"] if st == 1)
        line = f"[wave {m['wave']}] {lv.name if lv else '?'}: {len(fin)}/{n} finished, {dead} dead"
        fastest = None
        if fin:
            f, i = min(fin)
            fastest = (f + 1) / HZ
            line += f", fastest {fastest:.2f}s (agent {i}, path {self.path_of[i] + 1})"
            if self.best_time is None or fastest < self.best_time:
                self.best_time = fastest
                self.best_route = self.path_of[i] if self.paths else None
                self.save_best(lv.name, fastest, i)
                self.script.post({"type": "best", "payload": fastest})
            if self.phase != "speed":
                self.phase = "speed"
                print(f"[phase] {lv.name}: an agent finished - now rewarding speed (best {self.best_time:.2f}s, "
                      f"half the agents on path {self.best_route + 1 if self.best_route is not None else '-'})", flush=True)
        prog = [100.0 * (self.best_s[i] or 0.0) / self.route(i).length for i in range(n)] if self.paths else []
        fits = [f for f in self.fit if f is not None]
        best_fit = max(fits) if fits else None
        if best_fit is not None:
            line += f", fitness best {best_fit:.0f}" + (f", path {max(prog):.0f}% max / {np.mean(prog):.0f}% mean" if prog else "")
        self.history.append({"wave": m["wave"], "finished": len(fin), "dead": dead, "fastest": fastest, "fit": best_fit,
                             "path_max": max(prog) if prog else None, "path_mean": float(np.mean(prog)) if prog else None,
                             "ret": float(np.mean(rets)) if rets else None})
        if not self.args.test and self.ppo.steps >= self.args.batch:
            st = self.ppo.update()
            self.ppo.steps = 0
            self.updates.append(st)
            self.snapshot_weights()
            line += (f"  | PPO {st['samples']} samples: policy {st['pg']:+.3f}, value {st['vf']:.2f}, "
                     f"entropy {st['ent']:.2f}, KL {st['kl']:.3f}, EV {st['ev']:.2f}")
            self.save()
        el = (time.time() - self.t0) / 60
        print(line + f"  [{el:.1f} min]", flush=True)
        self.new_wave(n)
        # moving hazards seen in new places: routes around their sweep (every 3rd wave at most, 8 times per level,
        # so the routes settle and agents keep the same ones)
        if self.path_dirty and self.rebuilds < 8 and self.waves - self.built_wave >= 3:
            self.rebuilds += 1
            self.build_paths(reset=False)

    # reward paths
    def track_hazards(self, circles: list):
        """Remember where each lethal piece started and every place a moving one has been."""
        if self.hz_start is None or len(self.hz_start) != len(circles):
            self.hz_start = list(circles)
            return
        for k, (x, y, r) in enumerate(circles):
            x0, y0, _ = self.hz_start[k]
            if abs(x - x0) > 2 or abs(y - y0) > 2:
                if k not in self.moving:
                    self.moving.add(k)
                    self.sweep.add((round(x0 / 4) * 4, round(y0 / 4) * 4, r))
                key = (round(x / 4) * 4, round(y / 4) * 4, r)
                if key not in self.sweep:
                    self.sweep.add(key)
                    self.path_dirty = True

    def build_paths(self, reset: bool):
        lv = self.level
        static = [c for k, c in enumerate(self.hz_start or []) if k not in self.moving]
        t = time.time()
        self.paths = Planner(lv.path_solid, lv.ox, lv.oy, lv.spawn, lv.goals, static, sorted(self.sweep)).find(self.args.paths)
        self.paths_built, self.path_dirty, self.built_wave = True, False, self.waves
        print(f"[path] {lv.name}: " + (", ".join(f"path {k + 1} {p.length / TILE:.0f} tiles ({p.mode})" for k, p in enumerate(self.paths))
                                       if self.paths else "no route found")
              + f"  [{len(static)} fixed hazards, {len(self.moving)} moving; {time.time() - t:.2f}s]", flush=True)
        self.path_of = self.assign_paths(len(self.traj))
        if reset:                   # mid-wave (attached mid-level): progress so far was on other routes
            self.best_s = [None] * len(self.traj)
        self.post_paths()

    def post_paths(self):
        payload = [{"pts": p.points.tolist(), "col": argb(PATH_RGB[k % 3]), "w": 3} for k, p in enumerate(self.paths)] \
            if self.game_paths_on else []
        try:
            self.script.post({"type": "path", "payload": payload})
        except Exception:
            pass

    def toggle_game_paths(self):
        self.game_paths_on = not self.game_paths_on
        self.post_paths()

    def agent_fitness(self, i: int, won: bool, dead: bool) -> float:
        p = self.route(i)
        return fitness(won, dead, self.best_s[i] or 0.0, p.length if p is not None else 1.0, self.frames[i])

    def snapshot_weights(self):
        with torch.no_grad():
            names = {"pi1": self.policy.pi[0], "pi2": self.policy.pi[2], "pi3": self.policy.pi[4],
                     "vf1": self.policy.vf[0], "vf2": self.policy.vf[2], "vf3": self.policy.vf[4]}
            self.wver += 1
            self.weights = {"ver": self.wver, "w0": self.policy.pi[0].weight.detach().numpy().copy(),
                            "norms": {k: float(l.weight.norm()) for k, l in names.items()}}

    # saving
    def save(self, path=None):
        torch.save({"policy": self.policy.state_dict(), "obs": OBS, "actions": ACTIONS},
                   path or os.path.join(self.run_dir, "model.pt"))

    def save_best(self, level: str, t: float, agent: int):
        d = os.path.join(self.run_dir, "levels", level)
        os.makedirs(d, exist_ok=True)
        self.save(os.path.join(d, "best.pt"))
        with open(os.path.join(d, "best.json"), "w") as f:
            json.dump({"level": level, "seconds": t, "agent": agent, "wave": self.waves,
                       "path": self.path_of[agent] + 1 if self.paths else None,
                       "fitness": self.fit[agent], "actions": self.acts[agent]}, f)
        print(f"[best] {level}: {t:.2f}s -> {d}", flush=True)

    # frida
    def on_message(self, msg, data):
        if msg["type"] == "error":
            print("[script error]", msg.get("description"), msg.get("stack", ""), flush=True)
            return
        p = msg["payload"]
        ev = p.get("ev")
        reply = None
        try:
            if ev == "frame":
                reply = self.on_frame(p)
            elif ev == "info":
                self.on_info(p, data)
            elif ev == "wave_end":
                self.on_wave_end(p)
            elif ev == "level":
                print(f"[game] level {p['name']}", flush=True)
                self.level = None
                self.paths = []
            elif ev == "save":
                lv = p.get("level")
                self.save(os.path.join(self.run_dir, f"{lv}_model.pt"))
                print(f"[S] saved the agent for {lv}; finishing the level", flush=True)
                self.script.post({"type": "finish"})
            elif ev == "stopped":
                self.stopped.set()
            elif ev in ("ready", "wave", "continue", "level_end"):
                if ev != "wave":
                    print(f"[game] {ev} {p}", flush=True)
            else:
                print(f"[game] {p}", flush=True)
        except Exception:  # never leave the game waiting
            import traceback
            traceback.print_exc()
            reply = None
        if ev in ("info", "wave_end"):         # the game waits for this between waves / at the level start
            self.script.post({"type": "go"})
        if ev == "frame":
            if reply is None:
                reply = {"a": [[0, 0, 0, 0]] * len(p["s"]), "leader": 0}
            if self.stopping:
                reply["stop"] = True
            self.script.post({"type": "act", "payload": reply})

    def run(self):
        import frida
        other = other_trainers()
        if other:
            raise SystemExit(f"another realgame.py is already attached to the game (pid {', '.join(other)}): "
                             f"stop it first (Ctrl+C in its terminal, or a file runs/<name>/STOP)")
        if self.args.agents > MAX_AGENTS:
            print(f"--agents {self.args.agents}: the game has {MAX_AGENTS} spare Meat Boys per level, using {MAX_AGENTS}")
        cfg = {"agents": max(1, min(self.args.agents, MAX_AGENTS)), "wave_frames": int(self.args.wave_seconds * HZ),
               "finish_frames": 60, "debug": bool(self.args.debug_spawn)}
        with open(JS, encoding="utf-8") as f:
            src = f.read().replace("__CONFIG__", json.dumps(cfg))
        try:
            session = frida.attach(GAME_EXE)
        except frida.ProcessNotFoundError:
            raise SystemExit(f"{GAME_EXE} is not running: start the game and enter a level")
        mon = None
        if not self.args.no_monitor:
            try:
                from monitor import Monitor
                mon = Monitor(self)
            except Exception as e:
                print(f"monitor off: {e}")
        self.script = session.create_script(src)
        self.script.on("message", self.on_message)
        self.script.load()
        print(f"attached: {cfg['agents']} agents, waves of {self.args.wave_seconds:g}s, up to {self.args.paths} paths"
              + (" (TEST: scripted inputs, no learning)" if self.args.test else "") + "; Ctrl+C to stop", flush=True)
        stop_file = os.path.join(self.run_dir, "STOP")   # create it to stop a background run safely
        if os.path.exists(stop_file):
            os.remove(stop_file)
        try:
            while not os.path.exists(stop_file):
                if mon is not None and not mon.closed:
                    mon.tick()
                    time.sleep(1 / 20)
                else:
                    time.sleep(0.5)
            os.remove(stop_file)
        except KeyboardInterrupt:
            print("stopping...", flush=True)
        finally:
            if not self.args.test:
                self.save()
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
            if mon is not None:
                mon.close()
            print(f"saved in {self.run_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=40, help=f"agents per wave (1-{MAX_AGENTS})")
    ap.add_argument("--wave-seconds", type=float, default=20.0, help="wave time limit (default 20)")
    ap.add_argument("--batch", type=int, default=16384, help="samples per PPO update (updates between waves)")
    ap.add_argument("--paths", type=int, default=3, help="different reward paths per level (1-3)")
    ap.add_argument("--load", default=None, help="run folder or .pt to continue from")
    ap.add_argument("--name", default=None, help="run folder name under runs/ (default real_<time>)")
    ap.add_argument("--test", action="store_true", help="scripted inputs, no learning (checks the mod)")
    ap.add_argument("--no-monitor", action="store_true", help="no monitor window")
    ap.add_argument("--debug-spawn", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.paths = max(1, min(args.paths, 3))
    args.name = args.name or ("real_test" if args.test else time.strftime("real_%Y%m%d-%H%M%S"))
    Trainer(args).run()


if __name__ == "__main__":
    main()
