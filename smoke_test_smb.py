"""Headless smoke test for personal-use Super Meat Boy imports (imported/smb, gitignored).

Run:  python smbrl/smoke_test_smb.py   (from the project root)
Skips cleanly when no imports are present.
"""
import time

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from engine import World, Action, TILE, smb_measured
import levels
import levels_smb
from env import MeatBoyEnv, GRID_SOLID


def place(world: World, cell):
    """Put the player AABB centred in an engine cell, at rest."""
    p = world.p
    c, r = cell
    world.x = c * TILE + (TILE - p.player_w) / 2
    world.y = r * TILE + (TILE - p.player_h) / 2
    world.vx = world.vy = 0.0


def main():
    names = levels_smb.available()
    if not names:
        print("no imports under", levels_smb.IMPORT_DIR, "- SKIP")
        return 0
    ok = True

    # 1) every level: spawn in free space, valid observation; goal reachability reported separately.
    #    Unreachable goals are known geometry gaps in the clone (tile collision shapes not decoded),
    #    reported as WARN and excluded from training lists via levels_smb.available(solvable_only=True).
    gaps = []
    for n in names:
        env = MeatBoyEnv("smb:" + n, obs_grid=5)
        obs, _ = env.reset(seed=0)
        sc, sr = env.level.spawn_cell
        d = env._bfs[sr, sc]
        good = env._cells[sr, sc] != GRID_SOLID and obs.shape == env.observation_space.shape
        reach = bool(np.isfinite(d))
        if not reach:
            gaps.append(n)
        print(f"[{n:4s}] {env.level.cols}x{env.level.rows} obs={obs.shape[0]} bfs_spawn_to_goal={d:.0f} cells "
              f"hazards={len(env.level.hazards)} wind={len(env.level.wind)} -> "
              f"{'FAIL' if not good else 'ok' if reach else 'WARN goal unreachable in clone'}")
        ok &= bool(good) and reach == env.level.meta["goal_reachable"]
    solvable = [n for n in names if n not in gaps]
    print(f"[levels] solvable={solvable} geometry_gaps={gaps}")
    ok &= bool(solvable)

    # 2) measured preset: a jump peaks at the game's measured 79.3 units (= 95 px at TILE 24)
    w = World(levels.get("flat"), smb_measured())
    for _ in range(30):
        w.step(Action())
    y0, peak = w.state()["cy"], 0.0
    for t in range(90):
        s = w.step(Action(jump=True))
        peak = max(peak, y0 - s["cy"])
    print(f"[jump] smb_measured peak={peak:.1f}px (expect ~95)")
    ok &= 85.0 <= peak <= 110.0

    # 3) wind lifts, hazards kill, goal wins (2-4 has all three)
    if "2-4" in names:
        lv = levels.get("smb:2-4")
        w = World(lv, smb_measured())
        cell = next(c for c in sorted(lv.wind) if (c[0], c[1] + 1) in lv.wind and (c[0], c[1] - 1) in lv.wind)
        place(w, cell)
        y_start = w.y
        for _ in range(10):
            w.step(Action())
        lifted = w.y < y_start and w.vy < 0
        print(f"[wind] start cell={cell} dy={w.y - y_start:.1f}px vy={w.vy:.0f} -> {'ok' if lifted else 'FAIL'}")
        ok &= lifted

        w = World(lv, smb_measured())
        place(w, sorted(lv.hazards)[0])
        dead = w.step(Action())["dead"]
        print(f"[hazard] dead={dead}")
        ok &= dead

        w = World(lv, smb_measured())
        place(w, lv.goal_cell)
        won = w.step(Action())["won"]
        print(f"[goal] won={won}")
        ok &= won

    # 4) Gymnasium API conformance
    check_env(MeatBoyEnv("smb:" + names[0], obs_grid=5), skip_render_check=True)
    print("[check_env] passed")

    # 5) many agents: one vectorised batch over every imported level, random actions
    k = 8
    fns = [lambda n=solvable[i % len(solvable)]: MeatBoyEnv("smb:" + n, obs_grid=5, max_steps=600) for i in range(k)]
    venv = gym.vector.SyncVectorEnv(fns)
    venv.reset(seed=0)
    rng = np.random.default_rng(0)
    steps, t0, outcomes = 2000, time.perf_counter(), {"won": 0, "dead": 0}
    for _ in range(steps):
        _, _, term, trunc, info = venv.step(rng.integers(0, venv.single_action_space.n, size=k))
        for key in outcomes:
            if key in info:
                outcomes[key] += int(np.sum(info[key] & (term | trunc)))
    dt = time.perf_counter() - t0
    print(f"[vector] {k} envs x {steps} steps in {dt:.1f}s = {k * steps / dt:,.0f} env-steps/s; random outcomes={outcomes}")
    venv.close()

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
