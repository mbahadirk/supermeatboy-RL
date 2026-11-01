"""Gymnasium environment wrapping the clean-room platformer `World`.

Observation and reward shaping are controlled by the module-level constants
below; tune them there. Headless-safe (imports no display).

Run modules from the project root:  python smbrl/env.py
"""
from __future__ import annotations

import os
import json
import math
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from engine import World, Action, TILE, Physics, smb_measured
import levels
from reward_rail import RewardRail

# local-grid observation cell codes (obs_grid > 0)
GRID_EMPTY, GRID_SOLID, GRID_HAZARD, GRID_WIND, GRID_GOAL = 0.0, 1.0, -1.0, 0.5, 0.75
GRID_SHAPED = 0.9   # ramp / half block / cut corner (level.shapes): partly solid
GRID_CRUMBLE = 0.8  # crumble block: solid until it breaks after a touch

# ---- observation normalization scales -----------------------------------
VX_SCALE = 600.0
VY_SCALE = 1700.0

# ---- reward shaping (tunable) --------------------------------------------
W_PROG = 0.05          # reward per pixel of goal-distance reduction
R_WON = 10.0           # terminal bonus on reaching the goal
R_DEAD = -1.0          # terminal penalty on death
R_TIME = -0.001        # per-step time penalty
REWARD_CLIP = 10.0     # clip shaped reward to +/- this each step

# ---- "explore" reward mode (default for smb: levels) ------------------------
# The goal-distance field ignores gravity, so on levels like 1-2 it points straight up a face the player
# cannot climb and agents settle into hopping under the goal. This mode keeps progress as a guide but
# stops it from being a trap:
R_NEW_CELL = 0.02      # first visit of a cell in this episode
R_RARE_CELL = 0.05     # / sqrt(visits of that cell over all training on this level)
R_STALL = -0.5         # episode cut after STALL_FRAMES with neither a new cell nor a new closest distance
STALL_FRAMES = 360
# Archive of reached cells per level (shared by all envs in the process): explore_starts of the resets
# begin from a stored state of a rarely-chosen cell instead of the spawn (Go-Explore style), so the
# policy also practises the far parts of a level. Evaluation and in-game agents always start at spawn.
ARCHIVES: dict[str, dict] = {}
ARCHIVE_MAX = 4000
# Backward curriculum from one demonstration (Salimans & Chen 2018): a policy-free Go-Explore search finds
# a winning trajectory in the clone (prepare_level); demo_starts of the resets begin on that trajectory,
# first near its end, moving back towards the spawn as episodes started there keep finishing.
DEMOS: dict[str, dict] = {}
DEMO_WINDOW, DEMO_PASS = 16, 0.6
# found winning action lists are kept here (gitignored, like runs/) and reused when their replay still wins
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "demos")


def _demo_file(level_name: str) -> str:
    return os.path.join(DEMO_DIR, level_name.replace(":", "_") + ".json")


def _replay_demo(w, acts: list[int]) -> list[dict] | None:
    """Per-frame states of an action list from spawn, or None when it no longer wins (level or physics changed)."""
    s = w.reset()
    states = [w.snapshot()]
    for a in acts:
        s = w.step(Action(*ACTION_TABLE[a]))
        if s["won"] or s["dead"]:
            break
        states.append(w.snapshot())
    return states if s["won"] else None


def prepare_level(level_name: str, seconds: float = 30.0, seed: int = 0, log=print) -> dict | None:
    """Policy-free Go-Explore on the clone: random action chunks from archived cells until the goal is
    reached or time runs out. Stores the winning trajectory's per-frame states in DEMOS[level_name]."""
    import time as _time
    if level_name in DEMOS:
        return DEMOS[level_name]
    probe = MeatBoyEnv(level_name, explore_starts=0.0)   # its goal-distance field ranks the cells
    w = probe.world
    try:
        with open(_demo_file(level_name)) as f:
            cached = json.load(f)["actions"]
    except (OSError, ValueError, KeyError):
        cached = None
    states = _replay_demo(w, cached) if cached else None
    if states:
        demo = {"states": states, "start": max(0, len(states) - 30), "results": []}
        DEMOS[level_name] = demo
        log(f"[demo] {level_name}: cached winning path of {len(cached)} frames ({len(cached) / 60:.1f}s)")
        return demo
    rng = np.random.default_rng(seed)
    if probe.level.platforms:
        # moving platforms: where the player is only matters together with when (and what it is riding);
        # moving saws alone do not need this (and it slows the search there)
        cell = lambda s: (int(s["cx"] // TILE), int(s["cy"] // TILE), w._on_plat, w.clock // 20)
    else:
        cell = lambda s: (int(s["cx"] // TILE), int(s["cy"] // TILE))
    s = w.reset()
    # cell -> [state, times chosen, actions from spawn, goal distance (tiles)]
    arch = {cell(s): [w.snapshot(), 0, [], probe._goal_dist(s) / TILE]}
    t0, frames, win = _time.time(), 0, None
    while win is None and _time.time() - t0 < seconds:
        keys = list(arch)
        # rarely chosen cells first, and (domain knowledge) cells closer to the goal
        wts = np.array([1.0 / np.sqrt(arch[k][1] + 1.0) / (1.0 + arch[k][3] / 6.0) for k in keys])
        k = keys[int(rng.choice(len(keys), p=wts / wts.sum()))]
        arch[k][1] += 1
        w.restore(arch[k][0])
        acts = list(arch[k][2])
        for _ in range(12):
            a = int(rng.integers(len(ACTION_TABLE)))
            for _ in range(int(rng.integers(4, 31))):
                s = w.step(Action(*ACTION_TABLE[a]))
                acts.append(a)
                frames += 1
                if s["dead"] or s["won"]:
                    break
                c = cell(s)
                if c not in arch:
                    arch[c] = [w.snapshot(), 0, list(acts), probe._goal_dist(s) / TILE]
            if s["won"]:
                win = acts
            if s["dead"] or s["won"]:
                break
    if win is None:
        log(f"[demo] {level_name}: no winning path found in {seconds:.0f}s ({len(arch)} cells)")
        return None
    states = _replay_demo(w, win)   # deterministic engine: the replay wins again
    demo = {"states": states, "start": max(0, len(states) - 30), "results": []}
    DEMOS[level_name] = demo
    try:
        os.makedirs(DEMO_DIR, exist_ok=True)
        with open(_demo_file(level_name), "w") as f:
            json.dump({"level": level_name, "actions": win}, f)
    except OSError as e:
        log(f"[demo] could not cache the path: {e}")
    log(f"[demo] {level_name}: winning path of {len(win)} frames found in {_time.time() - t0:.1f}s "
        f"({frames:,} frames explored, {len(arch)} cells); backward curriculum starts {len(win) - demo['start']} frames from the goal")
    return demo

# Discrete action id -> (left, right, jump, run)
ACTION_TABLE = [
    (False, False, False, False),  # 0 noop
    (True,  False, False, False),  # 1 left
    (False, True,  False, False),  # 2 right
    (False, False, True,  False),  # 3 jump
    (True,  False, True,  False),  # 4 left+jump
    (False, True,  True,  False),  # 5 right+jump
    (False, True,  False, True),   # 6 right+run
    (False, True,  True,  True),   # 7 right+run+jump
    (True,  False, False, True),   # 8 left+run
]


N_STATE = 12  # state values before the local cell map (see _vector_obs)


def obs_size(obs_grid: int) -> int:
    """Observation length: N_STATE state values + (2k+1)^2 local cells."""
    return N_STATE + ((2 * obs_grid + 1) ** 2 if obs_grid > 0 else 0)


def obs_grid_of(n: int) -> int:
    """Local-map radius k for an observation length; inverse of obs_size."""
    for k in range(0, 16):
        if obs_size(k) == n:
            return k
    raise ValueError(f"observation size {n} is not a MeatBoyEnv layout")


class MeatBoyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level_name: str = "flat", max_steps: int = 1800,
                 physics: str | Physics | None = None, obs_grid: int = 0, progress: str | None = None,
                 reward_mode: str | None = None, explore_starts: float | None = None,
                 demo_starts: float | None = None):
        """physics: "original" | "smb_measured" | Physics (default: smb_measured for "smb:" levels).
        obs_grid: k > 0 appends a (2k+1)^2 local cell map around the player to the 10-dim vector.
        progress: "euclid" | "bfs" goal distance for the progress reward (default: bfs for "smb:" levels).
        reward_mode: "dense" (progress +/- every step) | "explore" (record progress only, novelty, stall cut;
        default for "smb:" levels). explore_starts: share of resets that start from the archive
        (explore mode; default 0.25, use 0 for evaluation).
        Defaults keep the original observation and behaviour for the built-in levels."""
        super().__init__()
        self.level_name = level_name
        self.max_steps = max_steps
        self.level = levels.get(level_name)
        smb = level_name.startswith("smb:")
        if physics is None:
            physics = "smb_measured" if smb else "original"
        if isinstance(physics, str):
            physics = {"original": Physics, "smb_measured": smb_measured}[physics]()
        self.world = World(self.level, physics)
        self.obs_grid = int(obs_grid)
        self.progress = progress or ("bfs" if smb else "euclid")
        self.reward_mode = reward_mode or "rail"
        if self.reward_mode not in ("rail", "dense", "explore"):
            raise ValueError("reward_mode must be 'rail', 'dense', or 'explore'")
        self.rail = RewardRail(self.level)
        self.explore_starts = (0.25 if self.reward_mode == "explore" else 0.0) if explore_starts is None else float(explore_starts)
        self.archive = ARCHIVES.setdefault(level_name, {})
        # share of resets on the demonstration (only when prepare_level found one); 0 for evaluation
        self.demo_starts = (0.5 if self.reward_mode == "explore" else 0.0) if demo_starts is None else float(demo_starts)
        if explore_starts is not None and float(explore_starts) == 0.0 and demo_starts is None:
            self.demo_starts = 0.0   # evaluation / in-game agents: always from the spawn
        self._demo_ep = False

        self.level_w = self.level.cols * TILE
        self.level_h = self.level.rows * TILE
        self.diag = float(np.hypot(self.level_w, self.level_h))
        self._cells = self._cell_map()
        self._bfs = self._goal_bfs() if self.progress == "bfs" else None

        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        # normalized observation; bounds are generous (values may slightly
        # exceed [-1, 1] early in an episode, so we do not hard-clip observations).
        n = obs_size(self.obs_grid)
        high = np.full(n, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._prev_goal_dist = None

    # ---- level-derived maps ----------------------------------------------
    def _cell_map(self) -> np.ndarray:
        """rows x cols array of GRID_* codes."""
        m = np.full((self.level.rows, self.level.cols), GRID_EMPTY, dtype=np.float32)
        for code, cells in ((GRID_WIND, self.level.wind), (GRID_HAZARD, self.level.hazards),
                            (GRID_SOLID, self.level.solids)):
            for c, r in cells:
                m[r, c] = code
        for c, r in self.level.shapes:
            m[r, c] = GRID_SHAPED
        for c, r in self.level.crumble:
            m[r, c] = GRID_CRUMBLE
        gc, gr = self.level.goal_cell
        m[gr, gc] = GRID_GOAL
        return m

    def _goal_bfs(self) -> np.ndarray:
        """4-connected path length (cells) from the goal through non-solid cells; inf if unreachable.
        Hazards do not block: the player jumps over them, and this only shapes progress reward."""
        rows, cols = self._cells.shape
        dist = np.full((rows, cols), np.inf, dtype=np.float32)
        gc, gr = self.level.goal_cell
        dist[gr, gc] = 0.0
        q = deque([(gc, gr)])
        while q:
            c, r = q.popleft()
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = c + dc, r + dr
                if (0 <= nc < cols and 0 <= nr < rows and dist[nr, nc] == np.inf
                        and self._cells[nr, nc] != GRID_SOLID):
                    dist[nr, nc] = dist[r, c] + 1.0
                    q.append((nc, nr))
        return dist

    def _goal_dist(self, s: dict) -> float:
        if self._bfs is None:
            return s["goal_dist"]
        c, r = int(s["cx"] // TILE), int(s["cy"] // TILE)
        rows, cols = self._bfs.shape
        if 0 <= c < cols and 0 <= r < rows and np.isfinite(self._bfs[r, c]):
            return float(self._bfs[r, c]) * TILE
        return s["goal_dist"]  # outside the reachable map (e.g. inside a wall corner): fall back

    def _grid_patch(self, s: dict) -> np.ndarray:
        k = self.obs_grid
        rows, cols = self._cells.shape
        # clamp: the player can be briefly outside the map (e.g. falling out) before the death check
        c = min(max(int(s["cx"] // TILE), 0), cols - 1)
        r = min(max(int(s["cy"] // TILE), 0), rows - 1)
        padded = np.pad(self._cells, k + 1, constant_values=GRID_SOLID)  # outside the level reads as solid
        patch = padded[r + 1:r + 2 * k + 2, c + 1:c + 2 * k + 2].copy()
        for bx0, by0, bx1, by1 in self.world._plat_boxes:   # moving platforms where they are now
            for rr in range(int(by0 // TILE), int((by1 - 1e-6) // TILE) + 1):
                for cc in range(int(bx0 // TILE), int((bx1 - 1e-6) // TILE) + 1):
                    pr, pc = rr - (r - k), cc - (c - k)
                    if 0 <= pr <= 2 * k and 0 <= pc <= 2 * k:
                        patch[pr, pc] = GRID_SOLID
        for gc_, gr_ in self.world._gone:   # broken crumble blocks
            pr, pc = gr_ - (r - k), gc_ - (c - k)
            if 0 <= pr <= 2 * k and 0 <= pc <= 2 * k:
                patch[pr, pc] = GRID_EMPTY
        for mx, my, mr in self.world.mover_positions() if self.level.movers else ():
            # moving hazards: every cell the disc touches, where it is now
            n = int(mr // TILE) + 1
            mc, mrow = int(mx // TILE), int(my // TILE)
            for rr in range(mrow - n, mrow + n + 1):
                for cc in range(mc - n, mc + n + 1):
                    pr, pc = rr - (r - k), cc - (c - k)
                    if (0 <= pr <= 2 * k and 0 <= pc <= 2 * k and patch[pr, pc] != GRID_SOLID
                            and math.hypot((cc + 0.5) * TILE - mx, (rr + 0.5) * TILE - my) < mr + TILE / 2):
                        patch[pr, pc] = GRID_HAZARD
        return patch.ravel()

    def _obs(self, s: dict) -> np.ndarray:
        vec = self._vector_obs(s)
        return np.concatenate([vec, self._grid_patch(s)]) if self.obs_grid > 0 else vec

    def _vector_obs(self, s: dict) -> np.ndarray:
        w = self.world
        return np.array([
            s["cx"] / self.level_w,
            s["cy"] / self.level_h,
            s["vx"] / VX_SCALE,
            s["vy"] / VY_SCALE,
            1.0 if s["on_ground"] else 0.0,
            float(s["on_wall"]),
            float(s["facing"]),
            s["goal_dx"] / self.level_w,
            s["goal_dy"] / self.level_h,
            s["goal_dist"] / self.diag,
            # input memory the game's jump rules depend on (engine World): a press only counts after 2
            # released frames, and releasing cuts a rising jump once its minimum frames have passed
            min(w._jump_up, 2) / 2.0,
            min(w._jump_frames, 10) / 10.0 if (w._jump_frames >= 0 and w.vy < 0) else 0.0,
        ], dtype=np.float32)

    def _cell_of(self, s: dict) -> tuple[int, int]:
        return int(s["cx"] // TILE), int(s["cy"] // TILE)

    def _pick_archived(self):
        """A stored state, favouring cells chosen rarely so far and cells closer to the goal."""
        cells = list(self.archive.values())
        w = np.array([(1.0 / np.sqrt(e["chosen"] + 1.0)) / (1.0 + e["dist"] / (8 * TILE)) for e in cells])
        e = cells[int(self.np_random.choice(len(cells), p=w / w.sum()))]
        e["chosen"] += 1
        return e

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = self.world.reset()
        demo = DEMOS.get(self.level_name)
        self._demo_ep = False
        if demo and self.demo_starts > 0 and self.np_random.random() < self.demo_starts:
            i = int(self.np_random.integers(demo["start"], len(demo["states"])))
            s = self.world.restore(demo["states"][i])
            self.world.steps = 0
            s = self.world.state()
            self._demo_ep = True
        elif (self.explore_starts > 0 and len(self.archive) > 8
                and self.np_random.random() < self.explore_starts):
            s = self.world.restore(self._pick_archived()["state"])
            self.world.steps = 0
            s = self.world.state()
        self._prev_goal_dist = self._goal_dist(s)
        self._best_dist = self._prev_goal_dist
        self._seen = {self._cell_of(s)}
        self._last_gain = 0
        self._rail_best_dist = 0.0
        self._prev_x = s["cx"]
        self._prev_y = s["cy"]
        self._stagnation = 0
        self._prev_action = 0
        self._collected_keys = set()
        self._latest_fitness = 0.0
        return self._obs(s), {"won": False, "dead": False, "steps": 0, "rail_ratio": 0.0, "fitness": 0.0}

    def _demo_result(self, won: bool):
        """Backward curriculum: once episodes started on the demo finish often enough, start earlier."""
        demo = DEMOS[self.level_name]
        demo["results"].append(won)
        if len(demo["results"]) >= DEMO_WINDOW:
            if np.mean(demo["results"][-DEMO_WINDOW:]) >= DEMO_PASS and demo["start"] > 0:
                demo["start"] = max(0, demo["start"] - max(15, len(demo["states"]) // 20))
            demo["results"] = []

    def _explore_reward(self, s: dict, d: float) -> tuple[float, bool]:
        """Record-progress + novelty reward; returns (reward, stalled)."""
        r = 0.0
        if d < self._best_dist:
            r += W_PROG * (self._best_dist - d)
            self._best_dist = d
            self._last_gain = s["steps"]
        cell = self._cell_of(s)
        if cell not in self._seen:
            self._seen.add(cell)
            self._last_gain = s["steps"]
            r += R_NEW_CELL
            e = self.archive.get(cell)
            if e is None and not (s["dead"] or s["won"]) and len(self.archive) < ARCHIVE_MAX:
                e = self.archive[cell] = {"state": self.world.snapshot(), "dist": d, "chosen": 0, "visits": 0}
            if e is not None:
                e["visits"] += 1
                r += R_RARE_CELL / np.sqrt(e["visits"])
        return r, s["steps"] - self._last_gain > STALL_FRAMES

    def step(self, action: int):
        left, right, jump, run = ACTION_TABLE[int(action)]
        s = self.world.step(Action(left=left, right=right, jump=jump, run=run))

        d = self._goal_dist(s)
        stalled = False
        if self.reward_mode == "rail":
            eval_res = self.rail.evaluate_step(
                self._prev_x, self._prev_y, s["cx"], s["cy"],
                self._rail_best_dist, self._stagnation, int(action), self._prev_action,
                bool(s["won"]), bool(s["dead"]), int(s["steps"]), self.max_steps
            )
            reward = eval_res["reward"]
            stalled = eval_res["stalled"]
            self._rail_best_dist = eval_res["new_best_progress"]
            self._stagnation = eval_res["stagnation_counter"]
            self._latest_fitness = eval_res["fitness"]
            for idx, kp in enumerate(self.rail.keys_pts):
                if idx not in self._collected_keys and math.hypot(s["cx"] - kp[0], s["cy"] - kp[1]) < 1.2 * TILE:
                    self._collected_keys.add(idx)
                    reward += 25.0
                    self._latest_fitness += 100.0
        elif self.reward_mode == "explore":
            reward, stalled = self._explore_reward(s, d)
            reward += R_TIME
        else:
            reward = W_PROG * (self._prev_goal_dist - d) + R_TIME
        self._prev_goal_dist = d
        self._prev_x, self._prev_y = s["cx"], s["cy"]
        self._prev_action = int(action)

        terminated = False
        if s["won"]:
            if self.reward_mode != "rail":
                reward += R_WON
            terminated = True
        elif s["dead"]:
            if self.reward_mode != "rail":
                reward += R_DEAD
            terminated = True

        reward = float(np.clip(reward, -REWARD_CLIP, 60.0))

        truncated = (not terminated) and (s["steps"] >= self.max_steps or stalled)
        if self._demo_ep and (terminated or truncated):
            self._demo_result(bool(s["won"]))
        if stalled and not terminated and self.reward_mode != "rail":
            reward += R_STALL

        info = {
            "won": bool(s["won"]),
            "dead": bool(s["dead"]),
            "steps": int(s["steps"]),
            "fitness": float(self._latest_fitness),
            "rail_progress": float(self._rail_best_dist),
            "rail_ratio": float(self.rail.progress_ratio(s["cx"], s["cy"])),
            "keys_collected": len(self._collected_keys),
            "stagnation": self._stagnation,
        }
        return self._obs(s), reward, terminated, truncated, info


if __name__ == "__main__":
    # Smoke test: hold RIGHT on flat and confirm it eventually wins.
    env = MeatBoyEnv("flat", max_steps=1800)
    obs, info = env.reset(seed=0)
    print("obs dim:", obs.shape, "action n:", env.action_space.n)
    total = 0.0
    for _ in range(1800):
        obs, r, term, trunc, info = env.step(6)  # right+run
        total += r
        if term or trunc:
            break
    print(f"hold-right rollout: reward={total:.3f} info={info}")
