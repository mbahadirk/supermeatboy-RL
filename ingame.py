"""Arena mode for the real Super Meat Boy (your own running copy): trained agents play every level.

Run from the project root, with the game open (windowed is easiest to watch) and a level entered:

    python smbrl/ingame.py runs/<run> --agents 16           # watch a saved agent's policy
    python smbrl/train.py --level smb:1-2 --game ...         # live: the agents follow training

There is no controllable main Meat Boy: the game's own character is hidden and frozen and only steers
the camera. N agents run in the calibrated clone in real time (actions sampled from the current policy)
and the game draws each of them with its own character renderer; the agent closest to the goal is drawn
solid and followed by the camera, the others translucent. Every agent keeps its own best finishing time;
the best of all is shown on the game's timer sign ("BEST 2.43").

S (in the game window): save the best agent of this level (its exact policy weights, its action sequence
and time) under <run>/arena/, then finish the level through the game's own flow; the game loads the next
level, the agents follow, and with train.py --game training continues there.

Personal use only: attaches Frida to the user's own game process; nothing is written to game files.
"""
from __future__ import annotations

import os
import json
import time
import argparse
import threading

import numpy as np

import levels_smb
from engine import TILE
from env import MeatBoyEnv, obs_grid_of

HERE = os.path.dirname(os.path.abspath(__file__))
ARENA_JS = os.path.join(HERE, "ingame_arena.js")
GAME_EXE = "SuperMeatBoy.exe"
FPS = 60


def _write_atomic(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def export_levels(path: str):
    """Grid size, origin and goal point (game units) of every imported level, keyed by the game's name."""
    out = {"levels": {}}
    for name in levels_smb.available():
        with open(os.path.join(levels_smb.IMPORT_DIR, name + levels_smb.SUFFIX)) as f:
            d = json.load(f)
        goal = next(o for o in d["objects"] if o["kind"] == "goal")
        out["levels"][name] = {"cols": d["W"], "rows": d["H"], "T": d["tile"], "origin": d["origin"],
                               "goal_xy": [goal["x"], goal["y"]]}
    _write_atomic(path, json.dumps(out))


def export_policy(model, out_dir: str, label: str):
    """Write the policy MLP (policy_net layers + action_net) as policy.json + policy.ver."""
    import torch.nn as nn
    pol = model.policy
    linears = [m for m in pol.mlp_extractor.policy_net if isinstance(m, nn.Linear)] + [pol.action_net]
    acts = {type(m) for m in pol.mlp_extractor.policy_net if not isinstance(m, nn.Linear)}
    if acts != {nn.Tanh}:
        raise ValueError(f"arena agents implement Tanh MLPs only, got {acts}")
    k = obs_grid_of(linears[0].in_features)
    layers = [{"nin": m.in_features, "nout": m.out_features,
               "W": [round(float(v), 6) for v in m.weight.detach().cpu().numpy().ravel()],
               "b": [round(float(v), 6) for v in m.bias.detach().cpu().numpy()]} for m in linears]
    _write_atomic(os.path.join(out_dir, "policy.json"), json.dumps({"obs_grid": k, "label": label, "layers": layers}))
    _write_atomic(os.path.join(out_dir, "policy.ver"), f"{time.time():.3f}")


class Agents(threading.Thread):
    """N clone agents on the game's current level at 60 Hz, actions sampled from policy.json. Tracks each
    agent's best finishing time and the best run overall, and posts positions / camera / best to the game."""

    LEADER_HOLD = 90   # frames the camera stays on a leader before switching

    def __init__(self, arena: "Arena", n: int, seed: int = 0):
        super().__init__(daemon=True)
        self.arena, self.n = arena, n
        self.rng = np.random.default_rng(seed)
        self.level, self.envs, self.obs = None, [], None
        self.layers, self.ver, self.k, self.policy = None, None, None, None
        self.best = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    # ---- policy -----------------------------------------------------------
    def _load_policy(self):
        try:
            with open(os.path.join(self.arena.out_dir, "policy.ver")) as f:
                ver = f.read().strip()
            if ver == self.ver:
                return
            with open(os.path.join(self.arena.out_dir, "policy.json")) as f:
                text = f.read()
            p = json.loads(text)
        except (OSError, ValueError):
            return  # not written yet / mid-write
        self.layers = [(np.array(l["W"], np.float32).reshape(l["nout"], l["nin"]), np.array(l["b"], np.float32))
                       for l in p["layers"]]
        if p["obs_grid"] != self.k:
            self.k, self.level = p["obs_grid"], None   # rebuild the envs for the new observation layout
        self.ver, self.policy = ver, {"label": p.get("label"), "text": text}

    def _sample(self, obs: np.ndarray) -> np.ndarray:
        h = obs
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        p = np.exp(h - h.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return np.array([self.rng.choice(len(row), p=row) for row in p])

    # ---- level ------------------------------------------------------------
    def _start_level(self, level: str):
        self.level = level
        self.envs = [MeatBoyEnv("smb:" + level, obs_grid=self.k, explore_starts=0.0) for _ in range(self.n)]
        self.obs = np.stack([e.reset()[0] for e in self.envs])
        self.ep_actions = [[] for _ in range(self.n)]
        self.agent_best = [None] * self.n          # frames
        self.best = None                           # {"agent", "frames", "actions", "policy"}
        self.leader, self.leader_frames = 0, self.LEADER_HOLD

    def _to_game(self, env: MeatBoyEnv):
        w, lv = env.world, self.arena.levels[self.level]
        k = TILE / lv["T"]
        cx, cy = w.x + w.p.player_w / 2, w.y + w.p.player_h / 2
        return [cx / k + lv["origin"][0], lv["rows"] * lv["T"] - cy / k + lv["origin"][1], w.facing > 0]

    def _rail_to_game(self, env: MeatBoyEnv):
        if not hasattr(env, "rail") or not env.rail:
            return []
        lv = self.arena.levels[self.level]
        k = TILE / lv["T"]
        return [[wp[0] / k + lv["origin"][0], lv["rows"] * lv["T"] - wp[1] / k + lv["origin"][1]] for wp in env.rail.waypoints]

    def _pick_leader(self):
        self.leader_frames += 1
        if self.leader_frames < self.LEADER_HOLD:
            return
        d = [e._goal_dist(e.world.state()) for e in self.envs]
        self.leader, self.leader_frames = int(np.argmin(d)), 0

    def snapshot_best(self) -> dict | None:
        with self.lock:
            if self.best is None:
                return None
            return {"level": self.level, "agent": self.best["agent"], "frames": self.best["frames"],
                    "seconds": round(self.best["frames"] / FPS, 3), "policy_label": self.best["policy"]["label"],
                    "actions": list(self.best["actions"]), "policy_text": self.best["policy"]["text"],
                    "agent_best_seconds": [None if b is None else round(b / FPS, 3) for b in self.agent_best]}

    # ---- loop -------------------------------------------------------------
    def run(self):
        t_next = time.perf_counter()
        while not self.stop_flag.is_set():
            self._load_policy()
            level = self.arena.current_level
            if self.layers is None or level not in self.arena.levels:
                time.sleep(0.2)
                t_next = time.perf_counter()
                continue
            with self.lock:
                if level != self.level:
                    self._start_level(level)
                acts = self._sample(self.obs)
                for i, (env, a) in enumerate(zip(self.envs, acts)):
                    o, _, term, trunc, info = env.step(int(a))
                    self.ep_actions[i].append(int(a))
                    if info["won"]:
                        f = info["steps"]
                        if self.agent_best[i] is None or f < self.agent_best[i]:
                            self.agent_best[i] = f
                        if self.best is None or f < self.best["frames"]:
                            self.best = {"agent": i, "frames": f, "actions": self.ep_actions[i], "policy": self.policy}
                            self.arena.log(f"[arena] {level}: new best {f / FPS:.2f}s by agent {i} ({self.policy['label']})")
                    if term or trunc:
                        o, _ = env.reset()
                        self.ep_actions[i] = []
                    self.obs[i] = o
                self._pick_leader()
                ghosts = [self._to_game(e) for e in self.envs]
                payload = {"level": level, "g": ghosts, "cam": ghosts[self.leader][:2], "leader": self.leader,
                           "best": None if self.best is None else self.best["frames"] / FPS,
                           "rail": self._rail_to_game(self.envs[0])}
            try:
                self.arena.script.post({"type": "ghosts", "payload": payload})
            except Exception:
                return  # script unloaded
            t_next += 1 / FPS
            time.sleep(max(0.0, t_next - time.perf_counter()))
            if time.perf_counter() - t_next > 0.5:   # fell behind (machine busy): resync
                t_next = time.perf_counter()


class Arena:
    """Attach to the running game, load ingame_arena.js and run the agents.

    out_dir holds policy.json/policy.ver (written by export_policy) and levels.json; saved best agents go
    to save_dir/<level>_best.json + <level>_best_policy.json. on_save(level) runs after a save (train.py
    saves its model there); current_level follows the game."""

    def __init__(self, out_dir: str, save_dir: str, agents: int = 16, log=print, on_save=None, debug=False):
        import frida
        self.out_dir = os.path.abspath(out_dir)   # the game resolves relative paths against its own folder
        self.save_dir, self.log, self.on_save = os.path.abspath(save_dir), log, on_save
        os.makedirs(self.out_dir, exist_ok=True)
        export_levels(os.path.join(self.out_dir, "levels.json"))
        with open(os.path.join(self.out_dir, "levels.json")) as f:
            self.levels = json.load(f)["levels"]
        self.current_level = None
        self.advancing, self.advanced_to = False, None   # set when S moved the game to another level
        cfg = {"levels": os.path.join(self.out_dir, "levels.json").replace("\\", "/"), "finish_frames": 60,
               "debug": debug}
        with open(ARENA_JS, encoding="utf-8") as f:
            src = f.read().replace("__CONFIG__", json.dumps(cfg))
        try:
            self.session = frida.attach(GAME_EXE)
        except frida.ProcessNotFoundError:
            raise SystemExit(f"{GAME_EXE} is not running: start the game first")
        self.script = self.session.create_script(src)
        self.script.on("message", self._on_message)
        self.script.load()
        self.agents = Agents(self, max(1, min(agents, 64)))
        self.agents.start()

    def save_best(self, level: str) -> dict | None:
        b = self.agents.snapshot_best()
        if b is None or b["level"] != level:
            return None
        os.makedirs(self.save_dir, exist_ok=True)
        _write_atomic(os.path.join(self.save_dir, f"{level}_best_policy.json"), b.pop("policy_text"))
        _write_atomic(os.path.join(self.save_dir, f"{level}_best.json"), json.dumps(b))
        return b

    def request_next(self, level: str | None = None) -> bool:
        """What S does: save the level's best agent, then finish the level so the game loads the next one.
        Returns False (nothing happens) while no agent has finished the level."""
        level = level or self.current_level
        b = self.save_best(level)
        if b is None:
            self.log(f"[arena] S: no agent has finished {level} yet; nothing saved")
            return False
        self.log(f"[arena] saved best agent of {level}: agent {b['agent']}, {b['seconds']:.2f}s "
                 f"({b['policy_label']}) -> {self.save_dir}")
        if self.on_save:
            self.on_save(level)
        self.advancing = True
        self.script.post({"type": "finish"})
        return True

    def _on_message(self, msg, _data):
        if msg["type"] == "error":
            self.log(f"[arena] script error: {msg.get('description')}")
            return
        p = msg["payload"]
        ev = p.get("ev")
        if ev == "ready":
            self.log("[arena] attached; enter a level (S in the game: save best agent + next level)")
        elif ev == "level":
            self.current_level = p["name"]
            if self.advancing:
                self.advancing, self.advanced_to = False, p["name"]
            self.log(f"[arena] level {p['name']}" + ("" if p["known"] else " (not imported: no agents here)"))
        elif ev == "save":
            self.request_next(p["level"])
        elif ev == "level_end":
            self.log(f"[arena] {p['level']} finished; continuing to the next level")
        elif ev == "debug":
            self.log(f"[arena] debug {p}")
        elif ev == "ghost_error":
            self.log(f"[arena] agent drawing disabled: {p['msg']}")

    def close(self):
        self.agents.stop_flag.set()
        self.agents.join(timeout=2)
        try:   # hand the character back to the game first, let a few frames pass, then remove the hooks
            self.script.exports_sync.shutdown()
            time.sleep(0.3)
        except Exception:
            pass
        for fn in (self.script.unload, self.session.detach):
            try:
                fn()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="run folder under runs/ or a .zip")
    ap.add_argument("--agents", type=int, default=16, help="agents on the level at once (max 64)")
    args = ap.parse_args()

    from train import resolve_model, load_model
    path = resolve_model(args.model)
    run_dir = os.path.dirname(path)
    arena = Arena(os.path.join(run_dir, "arena_live"), os.path.join(run_dir, "arena"), agents=args.agents)
    export_policy(load_model(path), arena.out_dir, os.path.basename(path))
    print("running; Ctrl+C to detach (the game keeps running)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    arena.close()


if __name__ == "__main__":
    main()
