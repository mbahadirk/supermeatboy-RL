"""Train agents inside the real Super Meat Boy (personal use). No clone: every agent is a real Meat Boy
character in the running game (realgame.js), so it collides with, dies on and finishes exactly like Meat Boy.

Start the game, enter a level, then from the project root:

    python smbrl/realgame.py --agents 32                     # train on whatever level the game shows
    python smbrl/realgame.py --agents 32 --load runs/real_4-1   # continue a saved agent
    python smbrl/realgame.py --agents 8 --test               # no learning: scripted inputs (checks the mod)

Waves: all agents spawn together after the level (saws, platforms, crumble blocks) is reset; each has one
life and waits where it died; the next wave starts when every agent is dead or finished, or after
--wave-seconds. The camera follows the agent closest to the goal. The timer sign shows the best finishing
time on this level. S in the game (window focused): save the agent and finish the level the game's way;
training continues on the next level the game loads. Ctrl+C (or a file runs/<name>/STOP): save and detach.

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

HERE = os.path.dirname(os.path.abspath(__file__))
JS = os.path.join(HERE, "realgame.js")
GAME_EXE = "SuperMeatBoy.exe"
HZ = 60
TILE = 20.0

# actions: (left, right, jump, run)
ACTIONS = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0),
           (0, 1, 0, 1), (0, 1, 1, 1), (1, 0, 0, 1), (1, 0, 1, 1)]
K = 5                                   # local map radius (cells): (2K+1)^2 values
N_STATE = 10
OBS = N_STATE + (2 * K + 1) ** 2
C_EMPTY, C_SOLID, C_HAZARD, C_GOAL = 0.0, 1.0, -1.0, 0.75

# rewards
R_PROGRESS = 1.0       # per tile of new closest path distance to the goal
R_WIN = 50.0
R_FAST = 20.0           # + this x (1 - finish time / wave length)
R_DEATH = -5.0
R_STEP = -0.002


# ---- level ------------------------------------------------------------------------------------------
class Level:
    """Static facts the game reported for the current level (live tile layer 0, goals, spawn)."""

    def __init__(self, info: dict, tiles: bytes):
        self.name = info["name"]
        self.W, self.H, self.ox, self.oy = info["W"], info["H"], info["ox"], info["oy"]
        t = np.frombuffer(tiles, dtype="<u4").reshape(self.H, self.W)        # row 0 = bottom
        tid = t & 0xff
        self.solid = ((tid > 0) & (tid < 100)) | (((t >> 8) & 1).astype(bool) & (tid > 0) & (tid <= 100))
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

    def grid_with_hazards(self, hz: list) -> np.ndarray:
        """Cell codes now: tiles, goals, and the lethal pieces at their current positions."""
        g = self.base.copy()
        for x, y, _a, _b, s in hz:
            rad = max(0.6, 0.9 * abs(s or 1.0))           # tiles; the piece shapes are not read yet
            c, r = self.cell(x, y)
            k = int(np.ceil(rad))
            r0, r1, c0, c1 = max(0, r - k), min(self.H, r + k + 1), max(0, c - k), min(self.W, c + k + 1)
            if r0 >= r1 or c0 >= c1:
                continue
            rr, cc = np.mgrid[r0:r1, c0:c1]
            m = (np.hypot(rr - r, cc - c) <= rad) & (g[r0:r1, c0:c1] != C_SOLID)
            g[r0:r1, c0:c1][m] = C_HAZARD
        return g


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


def observe(lv: Level, grid: np.ndarray, s: list) -> np.ndarray:
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
    return np.concatenate([vec, patch[::-1].ravel()])   # patch rows top-down


# ---- policy + PPO ---------------------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self, obs=OBS, n=len(ACTIONS), h=256):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n))
        self.vf = nn.Sequential(nn.Linear(obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def dist(self, o):
        return torch.distributions.Categorical(logits=self.pi(o))


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
        return a.numpy(), d.log_prob(a).numpy(), self.p.vf(o).squeeze(-1).numpy()

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
                           "lp": np.array(traj["lp"], np.float32), "adv": adv, "ret": adv + v})
        self.steps += n

    def update(self) -> dict:
        o = torch.as_tensor(np.concatenate([b["o"] for b in self.batch]))
        a = torch.as_tensor(np.concatenate([b["a"] for b in self.batch]))
        lp0 = torch.as_tensor(np.concatenate([b["lp"] for b in self.batch]))
        adv = torch.as_tensor(np.concatenate([b["adv"] for b in self.batch]))
        ret = torch.as_tensor(np.concatenate([b["ret"] for b in self.batch]))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n, stats = len(a), {}
        for _ in range(self.epochs):
            for idx in torch.randperm(n).split(self.mb):
                d = self.p.dist(o[idx])
                ratio = torch.exp(d.log_prob(a[idx]) - lp0[idx])
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                vl = ((self.p.vf(o[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                en = d.entropy().mean()
                loss = pg + 0.5 * vl - self.ent * en
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.p.parameters(), 0.5)
                self.opt.step()
                stats = {"pg": pg.item(), "vf": vl.item(), "ent": en.item()}
        self.batch = []
        return stats | {"samples": n}


# ---- session --------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, args):
        self.args = args
        self.run_dir = os.path.join(HERE, "runs", args.name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.policy = Policy()
        if args.load:
            p = args.load if args.load.endswith(".pt") else os.path.join(args.load, "model.pt")
            self.policy.load_state_dict(torch.load(p, map_location="cpu")["policy"])
            print(f"loaded {p}")
        self.ppo = PPO(self.policy)
        self.level: Level | None = None
        self.traj, self.best_d, self.acts, self.alive = [], [], [], []
        self.best_time = None
        self.waves = 0
        self.t0 = time.time()
        self.stats = deque(maxlen=20)
        self.script = None
        self.stopping = False
        self.stopped = threading.Event()

    # level / waves
    def on_info(self, info: dict, tiles: bytes):
        if info["W"] * info["H"] == 0 or len(tiles or b"") != info["W"] * info["H"] * 4:
            print(f"[level] ignoring invalid level data {info['W']}x{info['H']}", flush=True)
            return
        if info["name"] in (None, "unknown"):
            info["name"] = identify_level(info, tiles) or "unknown"
        self.level = Level(info, tiles)
        self.best_time = None
        n = info["agents"]
        print(f"[level] {info['name']}: {self.level.W}x{self.level.H}, {len(self.level.goals)} goal(s), "
              f"{info['lethal']} lethal pieces, {n} agents", flush=True)
        self.new_wave(n)

    def new_wave(self, n: int):
        self.traj = [{"o": [], "a": [], "lp": [], "v": [], "r": [], "done": False} for _ in range(n)]
        self.best_d = [None] * n
        self.acts = [[] for _ in range(n)]
        self.alive = [True] * n
        self.stagnation = [0] * n

    def on_frame(self, m: dict) -> dict:
        lv, states = self.level, m["s"]
        if lv is None:
            return {"a": [[0, 0, 0, 0]] * len(states), "leader": 0}
        n = len(states)
        if len(self.traj) != n:
            self.new_wave(n)
        grid = lv.grid_with_hazards(m["hz"])
        if self.args.debug_spawn and m["frame"] in (0, 1, 30):
            sx, sy = lv.spawn
            far = [(i, s) for i, s in enumerate(states) if abs(s[0] - sx) > 30 or abs(s[1] - sy) > 30]
            prev = getattr(self, "last_pos", {})
            print(f"  [spawn f{m['frame']}] spawn=({sx:.0f},{sy:.0f}) {len(far)} away: " + " ".join(
                f"a{i}=({s[0]:.0f},{s[1]:.0f} st{s[5]} fl{s[4]:x}, died at {prev.get(i, ('?', '?'))[0]},{prev.get(i, ('?', '?'))[1]})"
                for i, s in far[:6]), flush=True)
        self.last_pos = {i: (round(s[0]), round(s[1])) for i, s in enumerate(states)}
        if self.args.test and m["frame"] == 0 and m["hz"]:
            print("  hazards at wave start: " + " ".join(f"({h[0]:.0f},{h[1]:.0f})" for h in m["hz"][:6]), flush=True)
        if self.args.test and m["frame"] % 120 == 0:
            print(f"  f{m['frame']}: " + " ".join(f"({s[0]:.0f},{s[1]:.0f} v{s[2]:.0f},{s[3]:.0f} st{s[5]} {s[6]})"
                                               for s in states[:4]), flush=True)
        # finish the previous step: rewards for what happened since the last frame
        for i, s in enumerate(states):
            tr = self.traj[i]
            if not self.alive[i] or not tr["a"]:
                continue
            x, y, st = s[0], s[1], s[6]
            r = R_STEP
            d = lv.goal_dist(x, y)
            if self.best_d[i] is None or d < self.best_d[i]:
                r += R_PROGRESS * ((self.best_d[i] if self.best_d[i] is not None else d) - d)
                self.best_d[i] = d
            if hasattr(self, "last_pos") and i in self.last_pos:
                px, py = self.last_pos[i]
                if abs(x - px) < 2.0 and abs(y - py) < 2.0:
                    self.stagnation[i] += 1
                    if self.stagnation[i] > 40:
                        r -= 0.02
                    if self.stagnation[i] > 120:
                        r -= 0.06
                else:
                    self.stagnation[i] = max(0, self.stagnation[i] - 2)
            if st == 1:
                r += R_DEATH
            elif st == 2:
                r += R_WIN + R_FAST * max(0.0, 1.0 - m["frame"] / (self.args.wave_seconds * HZ))
            tr["r"].append(r)
            if st != 0:
                tr["done"] = True
                self.alive[i] = False
                self.ppo.add(tr, 0.0)
        live = [i for i, s in enumerate(states) if s[6] == 0 and self.alive[i]]
        acts = [[0, 0, 0, 0]] * n
        leader = 0
        if live:
            obs = np.stack([observe(lv, grid, states[i]) for i in live])
            if self.args.test:
                a = np.array([self.scripted(m["frame"], i) for i in live])
                lp = v = np.zeros(len(live), np.float32)
            else:
                a, lp, v = self.ppo.act(obs)
            for j, i in enumerate(live):
                tr = self.traj[i]
                if self.best_d[i] is None:
                    self.best_d[i] = lv.goal_dist(states[i][0], states[i][1])
                tr["o"].append(obs[j]); tr["a"].append(int(a[j])); tr["lp"].append(float(lp[j])); tr["v"].append(float(v[j]))
                acts[i] = list(ACTIONS[int(a[j])])
                self.acts[i].append(int(a[j]))
            leader = min(live, key=lambda i: lv.goal_dist(states[i][0], states[i][1]))
        won = [i for i, s in enumerate(states) if s[6] == 2]
        if won:
            leader = won[0]
        return {"a": acts, "leader": leader}

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
            if st == 1:
                tr["r"].append(R_STEP + R_DEATH)
                tr["done"] = True
                self.ppo.add(tr, 0.0)
            elif st == 2:
                tr["r"].append(R_STEP + R_WIN + R_FAST * max(0.0, 1.0 - m["frame"] / (self.args.wave_seconds * HZ)))
                tr["done"] = True
                self.ppo.add(tr, 0.0)
            else:
                tr["r"].append(R_STEP)
                with torch.no_grad():
                    lv_ = float(self.policy.vf(torch.as_tensor(tr["o"][-1])).item())
                self.ppo.add(tr, lv_)
        fin = [(f, i) for i, (st, f) in enumerate(zip(m["status"], m["finish"])) if st == 2 and f >= 0]
        dead = sum(1 for st in m["status"] if st == 1)
        line = f"[wave {m['wave']}] {lv.name if lv else '?'}: {len(fin)}/{n} finished, {dead} dead"
        if fin:
            f, i = min(fin)
            t = (f + 1) / HZ
            line += f", fastest {t:.2f}s (agent {i})"
            if self.best_time is None or t < self.best_time:
                self.best_time = t
                self.save_best(lv.name, t, i)
                self.script.post({"type": "best", "payload": t})
        if not self.args.test and self.ppo.steps >= self.args.batch:
            st = self.ppo.update()
            self.ppo.steps = 0
            line += f"  | PPO {st['samples']} samples, entropy {st['ent']:.2f}"
            self.save()
        el = (time.time() - self.t0) / 60
        print(line + f"  [{el:.1f} min]", flush=True)
        self.new_wave(n)

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
                       "actions": self.acts[agent]}, f)
        print(f"[best] {level}: {t:.2f}s -> {d}", flush=True)

    # frida
    def on_message(self, msg, data):
        if msg["type"] == "error":
            print("[script error]", msg.get("description"), msg.get("stack", ""), flush=True)
            return
        p = msg["payload"]
        ev = p.get("ev")
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
        except Exception as e:  # never leave the game waiting
            import traceback
            traceback.print_exc()
            reply = None
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
        cfg = {"agents": max(1, min(self.args.agents, 40)), "wave_frames": int(self.args.wave_seconds * HZ),
               "finish_frames": 60}
        with open(JS, encoding="utf-8") as f:
            src = f.read().replace("__CONFIG__", json.dumps(cfg))
        try:
            session = frida.attach(GAME_EXE)
        except frida.ProcessNotFoundError:
            raise SystemExit(f"{GAME_EXE} is not running: start the game and enter a level")
        self.script = session.create_script(src)
        self.script.on("message", self.on_message)
        self.script.load()
        print(f"attached: {cfg['agents']} agents, waves of {self.args.wave_seconds:g}s"
              + (" (TEST: scripted inputs, no learning)" if self.args.test else "") + "; Ctrl+C to stop", flush=True)
        stop_file = os.path.join(self.run_dir, "STOP")   # create it to stop a background run safely
        if os.path.exists(stop_file):
            os.remove(stop_file)
        try:
            while not os.path.exists(stop_file):
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
            print(f"saved in {self.run_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=32, help="agents per wave (1-40)")
    ap.add_argument("--wave-seconds", type=float, default=20.0, help="wave time limit (default 20)")
    ap.add_argument("--batch", type=int, default=16384, help="samples per PPO update (updates between waves)")
    ap.add_argument("--load", default=None, help="run folder or .pt to continue from")
    ap.add_argument("--name", default=None, help="run folder name under runs/ (default real_<time>)")
    ap.add_argument("--test", action="store_true", help="scripted inputs, no learning (checks the mod)")
    ap.add_argument("--debug-spawn", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.name = args.name or ("real_test" if args.test else time.strftime("real_%Y%m%d-%H%M%S"))
    Trainer(args).run()


if __name__ == "__main__":
    main()
