"""PPO training for MeatBoyEnv (Stable-Baselines3): live view, best-model saving, continue on another map.

Run from the project root:

    python smbrl/train.py --list-levels
    python smbrl/train.py --level smb:1-1 --n-envs 16 --watch --save-best
    python smbrl/train.py --level smb:3-1 --minutes 20 --save-best
    python smbrl/train.py --load runs/smb_1-1_<time> --level smb:3-1 --watch --save-best
    python smbrl/watch.py runs/smb_1-1_<time>                 # watch a saved agent play
    python smbrl/train.py --levels 4-1,4-2 --level-minutes 15 --watch       # campaign: level after level
    python smbrl/train.py --chapter hell --level-minutes 10                  # all 20 Hell levels
    python smbrl/train.py --chapter hell --world dark --game                 # dark world, in the real game
    python smbrl/train.py --all --level-minutes 10                           # every chapter, in game order

Options:
    --level        map to train on (built-ins: flat, gap, shaft; imports: smb:<name>)
    --n-envs       how many agents play the same map at once (parallel environments)
    --save-best    greedy-evaluate every --eval-every steps and keep the best policy as best.zip
    --load         continue from a saved agent (run folder -> its best.zip, else model.zip; or a .zip)
                   on any map; the observation layout (--obs-grid) comes from the loaded model
    --watch        open a window showing every agent live
                   keys: S real time on/off, V drawing on/off, Esc stop and save
    --game         arena in the running Super Meat Boy (ingame.py): --game-agents agents play the level
                   with the policy being trained (updated every --game-sync s); the best time is on the
                   game's timer sign; S in the game saves the best agent, finishes the level and training
                   continues on the next level

Campaign (--levels / --chapter / --all): one agent trains on each level in turn, up to --level-minutes per
level; once its greedy run finishes the level (plus --after-win-minutes to get faster) it moves to the next.
With --game the move goes through the real game: the best in-game agent is saved and the level is finished
(like S), so the game loads the next level and training follows it. --save-best is always on.
    --world          light (default), dark, both, warp or all: which levels of a chapter
    --list-chapters  print the chapters and their levels and exit

Each run writes runs/<name>/ : model.zip (last), best.zip (--save-best), meta.json (settings, level history).
The engine is deterministic, so one greedy episode fully describes the greedy policy; the final report
adds a sampled-action success rate to show how reliable the policy is.
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import argparse
import shutil
from collections import deque, Counter

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.save_util import load_from_zip_file

from engine import TILE
from env import MeatBoyEnv, obs_grid_of
import levels

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")

ROLLOUT_SIZE = 8192        # transitions per PPO update, split across the agents
REALTIME_HZ = 60.0         # game frame rate (S key in the live view)
VIEW_FPS = 30.0
PRINT_EVERY_S = 15.0


# ---- saved agents -----------------------------------------------------------
def resolve_model(path: str) -> str:
    """Run folder -> best.zip (else model.zip); a .zip path (extension optional) is used as given."""
    if os.path.isdir(path):
        for name in ("best.zip", "model.zip"):
            if os.path.exists(os.path.join(path, name)):
                return os.path.join(path, name)
        raise SystemExit(f"no best.zip or model.zip in {path}")
    if not path.endswith(".zip") and os.path.exists(path + ".zip"):
        path += ".zip"
    if not os.path.exists(path):
        raise SystemExit(f"model not found: {path}")
    return path


def _obs_size(path: str) -> int:
    """Observation length from the policy weights (avoids unpickling, which can fail across numpy majors)."""
    params = load_from_zip_file(path, load_data=False, device="cpu")[1]["policy"]
    return int(params["mlp_extractor.policy_net.0.weight"].shape[1])


def load_model(path: str, env=None, **kw) -> PPO:
    """PPO.load; if the pickled numpy objects came from another numpy major version (numpy 2 saves opened
    under numpy 1.26), rebuild the spaces from MeatBoyEnv's layout and drop the last-rollout caches
    (learn() re-creates them by resetting the env)."""
    try:
        return PPO.load(path, env=env, device="cpu", **kw)
    except (ModuleNotFoundError, ValueError, AttributeError) as e:
        import numpy as np
        from gymnasium import spaces
        from env import ACTION_TABLE
        n = _obs_size(path)
        custom = {"observation_space": spaces.Box(-np.inf, np.inf, (n,), np.float32),
                  "action_space": spaces.Discrete(len(ACTION_TABLE)),
                  "_last_obs": None, "_last_episode_starts": None}
        print(f"[load] {os.path.basename(path)} was saved with another numpy version ({type(e).__name__}); "
              f"rebuilt its observation/action spaces")
        return PPO.load(path, env=env, device="cpu", custom_objects=custom, **kw)


def model_obs_grid(path: str) -> int:
    """Local-map radius k the saved model was trained with, from its observation length."""
    try:
        return obs_grid_of(_obs_size(path))
    except ValueError as e:
        raise SystemExit(f"{path}: {e}")


def read_meta(model_path: str) -> dict:
    p = os.path.join(os.path.dirname(model_path), "meta.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def all_levels() -> list[str]:
    names = list(levels.LEVELS)
    try:
        import levels_smb
        names += ["smb:" + n for n in levels_smb.available(solvable_only=True)]
    except Exception:  # imports are optional (personal-use data, gitignored)
        pass
    return names


# ---- evaluation ---------------------------------------------------------------
def rollout(model, env: MeatBoyEnv, deterministic: bool = True) -> dict:
    """One episode. Returns outcome, steps, closest goal distance (tiles) and the centre path."""
    obs, _ = env.reset()
    path, closest = [], math.inf
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(int(action))
        s = env.world.state()
        path.append((s["cx"], s["cy"]))
        closest = min(closest, env._goal_dist(s) / TILE)
        if term or trunc:
            outcome = "won" if info["won"] else "dead" if info["dead"] else "timeout"
            return {"outcome": outcome, "steps": info["steps"], "closest_tiles": round(closest, 1), "path": path}


def score(r: dict) -> tuple:
    """Higher is better: reaching the goal beats everything, then fewer steps; otherwise get closer."""
    return (1, -r["steps"]) if r["outcome"] == "won" else (0, -r["closest_tiles"])


def describe(r: dict | None) -> str:
    if r is None:
        return "-"
    if r["outcome"] == "won":
        return f"won in {r['steps']} steps ({r['steps'] / REALTIME_HZ:.1f}s)"
    return f"{r['outcome']} after {r['steps']} steps, {r['closest_tiles']:.0f} tiles from goal"


# ---- callbacks ------------------------------------------------------------------
class TimeLimit(BaseCallback):
    def __init__(self, max_seconds: float):
        super().__init__()
        self.max_seconds = max_seconds

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self) -> bool:
        return time.time() - self.t0 < self.max_seconds


class BestModel(BaseCallback):
    """Greedy-evaluate every `every` steps; save best.zip when the score improves."""

    def __init__(self, every: int, run_dir: str, env: MeatBoyEnv):
        super().__init__()
        self.every, self.run_dir, self.env = every, run_dir, env
        self.best, self._next = None, 0

    def _on_training_start(self):
        self._next = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next = self.num_timesteps + self.every
            self.check()
        return True

    def check(self):
        r = rollout(self.model, self.env)
        if self.best is None or score(r) > score(self.best):
            self.best = dict(r, timesteps=self.num_timesteps)
            self.model.save(os.path.join(self.run_dir, "best.zip"))
            with open(os.path.join(self.run_dir, "best.json"), "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in self.best.items() if k != "path"} | {"level": self.env.level_name}, f, indent=1)
            print(f"[best] {self.num_timesteps:,} steps: {describe(r)} -> best.zip")


class GameSync(BaseCallback):
    """Push the current policy to the in-game arena (ingame.py) after PPO updates, and stop this training
    segment when the game moves to another imported level (S in the game) so training follows it."""

    def __init__(self, arena, every_s: float, level: str):
        super().__init__()
        self.arena, self.every_s, self.level, self._last = arena, every_s, level, 0.0
        self.next_level = None

    def push(self):
        from ingame import export_policy
        export_policy(self.model, self.arena.out_dir, f"{self.num_timesteps / 1e6:.2f}M steps")
        self._last = time.time()

    def _on_training_start(self):
        self.push()

    def _on_rollout_end(self):
        if time.time() - self._last >= self.every_s:
            self.push()

    def _on_step(self) -> bool:
        nxt = self.arena.advanced_to   # only levels reached through S, not wherever the game happens to be
        if nxt:
            self.arena.advanced_to = None
            if nxt in self.arena.levels and "smb:" + nxt != self.level:
                self.next_level = "smb:" + nxt
                return False
            print(f"[arena] the game moved to {nxt}, which is not imported: training stays on {self.level}")
        return True


class CampaignAdvance(BaseCallback):
    """Campaign: end this level once the greedy policy has finished it (and after_win_s more seconds of
    training passed). With the game arena, ask the game to move on instead (GameSync then follows it)."""

    def __init__(self, best: "BestModel", after_win_s: float, game, level: str):
        super().__init__()
        self.best, self.after_win_s, self.game, self.level = best, after_win_s, game, level
        self.t_win, self._last_request, self.solved = None, 0.0, False

    def _on_step(self) -> bool:
        b = self.best.best
        if not b or b["outcome"] != "won":
            return True
        now = time.time()
        if self.t_win is None:
            self.t_win = now
            print(f"[campaign] {self.level} solved: {describe(b)}"
                  + (f"; {self.after_win_s / 60:.1f} more min to get faster" if self.after_win_s > 0 else ""),
                  flush=True)
        if now - self.t_win < self.after_win_s:
            return True
        self.solved = True
        if self.game is None:
            return False
        if now - self._last_request > 5.0:   # until an in-game agent has finished too
            self._last_request = now
            self.game.request_next(self.level.split(":", 1)[-1])
        return True


class Monitor(BaseCallback):
    """Episode outcomes, periodic progress lines and (optionally) the live map view."""

    def __init__(self, level_name: str, minutes: float, best: BestModel | None, watch: bool):
        super().__init__()
        self.level_name, self.limit_s, self.best = level_name, minutes * 60, best
        self.recent = deque(maxlen=100)
        self.total = Counter()
        self.watch, self.view = watch, None
        self.draw_on, self.realtime, self.stop = True, False, False

    def _on_training_start(self):
        self.t0 = self._last_print = time.time()
        self._last_draw = 0.0
        self._steps0 = self.num_timesteps
        self.envs = [e.unwrapped for e in self.training_env.envs]
        if self.watch:
            from viewer import MapView
            self.view = MapView("SMB-RL training")
            self.view.set_level(self.envs[0].level, self.level_name)

    def _curriculum(self) -> str:
        from env import DEMOS
        d = DEMOS.get(self.level_name)
        return f"  curriculum: starts {len(d['states']) - d['start']}/{len(d['states'])} frames from goal" if d else ""

    def _rates(self) -> str:
        n = len(self.recent) or 1
        c = Counter(self.recent)
        return f"won {100 * c['won'] / n:.0f}%  dead {100 * c['dead'] / n:.0f}%  timeout {100 * c['timeout'] / n:.0f}%"

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                outcome = "won" if info["won"] else "dead" if info["dead"] else "timeout"
                self.recent.append(outcome)
                self.total[outcome] += 1
                if self.view:
                    env, o = self.envs[i], info["terminal_observation"]
                    self.view.end_episode(i, o[0] * env.level_w, o[1] * env.level_h, outcome)

        now = time.time()
        if now - self._last_print >= PRINT_EVERY_S:
            self._last_print = now
            sps = (self.num_timesteps - self._steps0) / max(now - self.t0, 1e-6)
            print(f"[{(now - self.t0) / 60:5.1f} min] {self.num_timesteps:,} steps ({sps:,.0f}/s)  "
                  f"episodes {sum(self.total.values()):,}  last 100: {self._rates()}  "
                  f"best: {describe(self.best.best) if self.best else '-'}{self._curriculum()}", flush=True)

        if self.view:
            if self.realtime:  # every agent at game speed: one step per frame
                time.sleep(max(0.0, 1.0 / REALTIME_HZ - (now - getattr(self, "_last_step", 0.0))))
                self._last_step = time.time()
            for i, env in enumerate(self.envs):
                w = env.world
                self.view.track(i, w.x + w.p.player_w / 2, w.y + w.p.player_h / 2)
            if time.time() - self._last_draw >= 1.0 / VIEW_FPS:
                self._last_draw = time.time()
                self._handle(self.view.poll())
                if self.draw_on:
                    self._draw()
        return not self.stop

    def _on_rollout_end(self):
        if self.view and self.draw_on:
            self._draw(banner="PPO update...")

    def _handle(self, cmds):
        for c in cmds:
            if c == "quit":
                self.stop = True
            elif c == "slow":
                self.realtime = not self.realtime
            elif c == "draw":
                self.draw_on = not self.draw_on
                if not self.draw_on:
                    self._draw(banner="drawing off (V)")

    def _draw(self, banner=None):
        if self.best and self.best.best:
            self.view.best_path = self.best.best["path"]
        el = time.time() - self.t0
        sps = (self.num_timesteps - self._steps0) / max(el, 1e-6)
        agents = [(i, e.world.x, e.world.y, e.world.p.player_w, e.world.p.player_h) for i, e in enumerate(self.envs)]
        lines = [
            f"agents (parallel envs): {len(self.envs)}",
            f"steps: {self.num_timesteps:,}  ({sps:,.0f}/s)",
            f"time: {el / 60:.1f} / {self.limit_s / 60:.1f} min",
            f"episodes: {sum(self.total.values()):,}",
            "",
            "last 100 episodes:",
            ("  " + self._rates(), (200, 230, 200)),
            "",
            "best greedy run (gold line):" if self.best else "best model: off (--save-best)",
            (f"  {describe(self.best.best)}" if self.best else "", (255, 210, 60)),
            "",
            (f"[S] real time: {'on' if self.realtime else 'off'}", (150, 150, 165)),
            (f"[V] drawing: {'on' if self.draw_on else 'off'}", (150, 150, 165)),
            ("[Esc] stop and save", (150, 150, 165)),
        ]
        lead_idx = 0
        best_prog = -1.0
        for i, env in enumerate(self.envs):
            p = getattr(env, "_rail_best_dist", 0.0)
            if p > best_prog:
                best_prog = p
                lead_idx = i

        if self.view and getattr(self, "model", None) is not None:
            try:
                import torch
                lead_env = self.envs[lead_idx]
                obs = lead_env._obs(lead_env.world.state())
                with torch.no_grad():
                    obs_t = self.model.policy.obs_to_tensor(obs)[0]
                    dist = self.model.policy.get_distribution(obs_t)
                    probs = dist.distribution.probs.detach().cpu().numpy()[0]
                    val = float(self.model.policy.predict_values(obs_t).detach().cpu().numpy()[0, 0])
                    ent = float(dist.entropy().detach().cpu().numpy()[0])

                self.view.set_telemetry({
                    "action_probs": probs,
                    "value": val,
                    "entropy": ent,
                    "rail_ratio": lead_env.rail.progress_ratio(lead_env.world.state()["cx"], lead_env.world.state()["cy"]) if hasattr(lead_env, "rail") else 0.0,
                    "fitness": getattr(lead_env, "_latest_fitness", 0.0),
                    "stagnation": getattr(lead_env, "_stagnation", 0),
                    "keys_collected": len(getattr(lead_env, "_collected_keys", ())),
                })
            except Exception:
                pass

        self.view.frame(agents, lines, highlight=lead_idx, banner=banner, world=self.envs[0].world)

    def _on_training_end(self):
        if self.view:
            self.view.close()


# ---- main ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default=None, help="map to train on (default: flat, or the loaded agent's last map)")
    ap.add_argument("--list-levels", action="store_true", help="print the available maps and exit")
    ap.add_argument("--list-chapters", action="store_true", help="print the chapters and their levels and exit")
    ap.add_argument("--levels", default=None, help="campaign: comma-separated levels, e.g. 4-1,4-2")
    ap.add_argument("--chapter", default=None,
                    help="campaign: forest, hospital, saltfactory, hell, rapture, drfetus, bandage, ... or 1-7")
    ap.add_argument("--world", default="light", choices=["light", "dark", "both", "warp", "all"],
                    help="campaign: which levels of the chapter(s) (default light)")
    ap.add_argument("--all", action="store_true", help="campaign: every chapter in game order")
    ap.add_argument("--start-at", default=None, help="campaign: skip the levels before this one")
    ap.add_argument("--level-minutes", type=float, default=10.0, help="campaign: budget per level (default 10)")
    ap.add_argument("--after-win-minutes", type=float, default=1.0,
                    help="campaign: keep training this long after the first win to get faster (default 1)")
    ap.add_argument("--n-envs", type=int, default=8, help="agents playing the same map at once (default 8)")
    ap.add_argument("--minutes", type=float, default=5.0, help="wall-clock budget (default 5)")
    ap.add_argument("--timesteps", type=int, default=100_000_000, help="step budget; whichever limit hits first")
    ap.add_argument("--save-best", action="store_true", help="keep the best greedy policy as best.zip")
    ap.add_argument("--eval-every", type=int, default=25_000, help="steps between best-model checks")
    ap.add_argument("--load", default=None, help="saved agent (run folder or .zip) to continue training")
    ap.add_argument("--watch", action="store_true", help="live window with every agent")
    ap.add_argument("--game", action="store_true",
                    help="arena in the running Super Meat Boy: agents with the policy being trained (see ingame.py)")
    ap.add_argument("--game-sync", type=float, default=5.0, help="seconds between policy pushes to the game")
    ap.add_argument("--game-agents", type=int, default=16,
                    help="with --game: agents playing in the game at once (clone-simulated, max 64)")
    ap.add_argument("--name", default=None, help="run folder name under runs/ (default: <level>_<time>)")
    ap.add_argument("--obs-grid", type=int, default=None,
                    help="local map radius k for new agents (default 5 for smb: maps, 0 for built-ins)")
    ap.add_argument("--demo-seconds", type=float, default=None,
                    help="smb: maps: seconds of policy-free search for a winning path to start a backward "
                         "curriculum from (default 30, campaigns 180; found paths are cached in runs/demos)")
    ap.add_argument("--max-steps", type=int, default=1800, help="episode length cap (1800 = 30s of game time)")
    ap.add_argument("--reward-mode", default="rail", choices=["rail", "explore", "dense"],
                    help="reward mode: rail (Dijkstra optimal path, default), explore, dense")
    ap.add_argument("--eval-episodes", type=int, default=20, help="sampled-action episodes in the final report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print SB3's PPO tables")
    args = ap.parse_args()

    if args.list_levels:
        print("\n".join(all_levels()))
        return
    if args.list_chapters:
        import levels_smb
        groups = {}
        for c in levels_smb.chapters():
            groups.setdefault((c["chapter_file"], c["kind"]), []).append(
                c["file"] + ("" if c["imported"] else "(missing)"))
        for (ch, kind), files in groups.items():
            print(f"{ch:22s} {kind:11s} {' '.join(files)}")
        return

    queue = campaign_levels(args)
    if args.demo_seconds is None:
        args.demo_seconds = 180.0 if queue else 30.0
    if queue:
        args.save_best = True
        if args.level is None:
            args.level = queue[0]
        elif args.level not in queue:
            queue.insert(0, args.level)
        queue = queue[queue.index(args.level):]
        if "--minutes" not in sys.argv:
            args.minutes = args.level_minutes * len(queue)
        print(f"campaign: {len(queue)} levels, up to {args.level_minutes:g} min each: "
              f"{' '.join(q[4:] for q in queue)}")

    load_path, meta = None, {"history": []}
    if args.load:
        load_path = resolve_model(args.load)
        meta = read_meta(load_path) or meta
        k = model_obs_grid(load_path)
        if args.obs_grid is not None and args.obs_grid != k:
            raise SystemExit(f"--obs-grid {args.obs_grid} does not match the loaded agent (trained with {k})")
        args.obs_grid = k
    level = args.level or (meta["history"][-1]["level"] if meta["history"] else "flat")
    if args.obs_grid is None:
        args.obs_grid = 5 if level.startswith("smb:") else 0

    env_kwargs = {"max_steps": args.max_steps, "obs_grid": args.obs_grid, "reward_mode": args.reward_mode}
    name = args.name or f"{level.replace(':', '_')}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, name)
    os.makedirs(run_dir, exist_ok=True)

    vec_env = make_vec_env(MeatBoyEnv, n_envs=args.n_envs, seed=args.seed,
                           env_kwargs={"level_name": level, **env_kwargs})
    if load_path:
        model = load_model(load_path, env=vec_env, seed=args.seed)
        model.verbose = int(args.verbose)
        print(f"loaded {load_path} ({model.num_timesteps:,} steps so far)")
    else:
        model = PPO("MlpPolicy", vec_env, verbose=int(args.verbose), seed=args.seed,
                    n_steps=max(64, ROLLOUT_SIZE // args.n_envs), batch_size=512, gae_lambda=0.95,
                    gamma=0.99, ent_coef=0.01, learning_rate=3e-4, device="cpu")

    game = None
    if args.game:
        from ingame import Arena
        game = Arena(os.path.join(run_dir, "arena_live"), os.path.join(run_dir, "arena"), agents=args.game_agents,
                     on_save=lambda lv: model.save(os.path.join(run_dir, "arena", f"{lv}_model.zip")))
    deadline = time.time() + args.minutes * 60
    history, steps_total0, t_run0 = list(meta["history"]), model.num_timesteps, time.time()
    while True:
        seg_deadline = min(deadline, time.time() + args.level_minutes * 60) if queue else deadline
        entry, next_level = train_segment(args, model, vec_env, level, env_kwargs, run_dir, seg_deadline, game,
                                          load_path, campaign=bool(queue))
        history.append(entry)
        vec_env.close()
        if queue and next_level is None and time.time() < deadline:
            i = queue.index(level) if level in queue else -1
            if i + 1 < len(queue):
                next_level = queue[i + 1]
                if game:
                    print(f"[campaign] the game did not move on from {level}; training continues on {next_level} "
                          f"(load it in the game to see the agents there)")
        if next_level is None:
            break
        # S in the game finished this level and the game loaded the next one: continue training there
        level, load_path = next_level, None
        print(f"\n=== {name}: " + ("next campaign level" if queue and not game else "the game moved on")
              + f", continuing on '{level}' ===", flush=True)
        vec_env = make_vec_env(MeatBoyEnv, n_envs=args.n_envs, seed=args.seed,
                               env_kwargs={"level_name": level, **env_kwargs})
        model.set_env(vec_env)
    if game:
        game.close()

    model.save(os.path.join(run_dir, "model.zip"))
    meta = {"obs_grid": args.obs_grid, "max_steps": args.max_steps, "history": history}
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"\n*** {name}: {(time.time() - t_run0) / 60:.1f} min, {model.num_timesteps - steps_total0:,} steps ***")
    print(f"  saved in {run_dir} (model.zip; best.zip = best of the last level, per level under levels/)")


def campaign_levels(args) -> list[str]:
    """Levels of a campaign (--levels / --chapter / --all), as smb:<name>, in game order."""
    if not (args.levels or args.chapter or args.all):
        return []
    import levels_smb
    if args.levels:
        names = [n.strip().split(":", 1)[-1] for n in args.levels.split(",") if n.strip()]
        missing = [n for n in names if n not in levels_smb.available()]
        if missing:
            raise SystemExit(f"not imported (or no goal): {', '.join(missing)}")
    else:
        names = levels_smb.chapter_levels(None if args.all else args.chapter, args.world)
        if not names:
            raise SystemExit(f"no levels for chapter {args.chapter!r} / world {args.world!r} (see --list-chapters)")
    if args.start_at:
        s = args.start_at.split(":", 1)[-1]
        if s not in names:
            raise SystemExit(f"--start-at {s} is not in the campaign")
        names = names[names.index(s):]
    return ["smb:" + n for n in names]


def train_segment(args, model, vec_env, level, env_kwargs, run_dir, deadline, game, load_path, campaign=False):
    """Train on one level until the time/step budget ends or the game moves to another imported level.
    Returns (history entry, next level or None)."""
    eval_env = MeatBoyEnv(level, explore_starts=0.0, **env_kwargs)
    if eval_env.reward_mode == "explore":
        from env import prepare_level
        prepare_level(level, seconds=args.demo_seconds)
    level_dir = os.path.join(run_dir, "levels", level.replace(":", "_"))
    os.makedirs(level_dir, exist_ok=True)
    print(f"=== training on '{level}' with {args.n_envs} agents, up to {(deadline - time.time()) / 60:.1f} min "
          f"(obs_grid={args.obs_grid}, reward={eval_env.reward_mode}, save_best={args.save_best}, "
          f"watch={args.watch}, game={bool(game)}) ===", flush=True)
    best = BestModel(args.eval_every, level_dir, eval_env) if args.save_best else None
    monitor = Monitor(level, (deadline - time.time()) / 60, best, args.watch)
    callbacks = [TimeLimit(deadline - time.time()), monitor] + ([best] if best else [])
    sync = None
    if game:
        sync = GameSync(game, args.game_sync, level)
        callbacks.append(sync)
    if campaign and best:
        callbacks.append(CampaignAdvance(best, args.after_win_minutes * 60, game, level))
    steps_before, t0 = model.num_timesteps, time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks),
                    reset_num_timesteps=(not load_path) and steps_before == 0, progress_bar=False)
    except KeyboardInterrupt:
        print("interrupted: saving")
    elapsed = time.time() - t0
    if best:
        best.check()  # the final policy may be the best one
        if best.best:   # run folder best.zip = best agent of the latest level (what --load picks up)
            shutil.copyfile(os.path.join(level_dir, "best.zip"), os.path.join(run_dir, "best.zip"))

    greedy = rollout(model, eval_env)
    sampled = Counter(rollout(model, eval_env, deterministic=False)["outcome"] for _ in range(args.eval_episodes))
    entry = {
        "level": level, "n_envs": args.n_envs, "seconds": round(elapsed), "loaded_from": load_path,
        "reward_mode": eval_env.reward_mode,
        "steps_added": model.num_timesteps - steps_before, "total_steps": model.num_timesteps,
        "final_greedy": {k: v for k, v in greedy.items() if k != "path"},
        "final_sampled_won": f"{sampled['won']}/{args.eval_episodes}",
        "best": {k: v for k, v in best.best.items() if k != "path"} if best and best.best else None,
        "solved": bool(best and best.best and best.best["outcome"] == "won"),
    }
    print(f"\n*** '{level}' after {elapsed / 60:.1f} min, {model.num_timesteps:,} total steps ***")
    print(f"  final greedy : {describe(greedy)}")
    print(f"  final sampled: won {sampled['won']}/{args.eval_episodes} episodes")
    if best and best.best:
        print(f"  best         : {describe(best.best)}  (at {best.best['timesteps']:,} steps) -> best.zip")
    return entry, (sync.next_level if sync else None)


if __name__ == "__main__":
    main()
