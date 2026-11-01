"""Watch a saved agent play in real time, on its own map or any other.

Run from the project root:

    python smbrl/watch.py runs/<run>                          # map from the run's meta.json
    python smbrl/watch.py runs/<run> --level smb:3-1 --agents 8
    python smbrl/watch.py runs/ppo_smb_1-1.zip --level smb:1-1

A run folder loads best.zip (else model.zip); use --file model to pick the last policy instead.
Agent 1 (white outline) plays greedily (always its most likely action). Extra agents sample actions
from the policy, which shows how reliable it is. The engine is deterministic, so greedy replays repeat.

Keys: Space pause, R restart, + / - speed, N / P next / previous map, Esc quit.
"""
from __future__ import annotations

import os
import argparse
from collections import Counter

import numpy as np

from env import MeatBoyEnv
from train import resolve_model, model_obs_grid, read_meta, all_levels, load_model, REALTIME_HZ

SPEEDS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="run folder under runs/ or a .zip")
    ap.add_argument("--file", choices=["best", "model"], default=None,
                    help="in a run folder: best.zip or model.zip (default: best if present)")
    ap.add_argument("--level", default=None, help="map to play (default: the run's last training map)")
    ap.add_argument("--agents", type=int, default=1, help="agents on the map at once (default 1)")
    ap.add_argument("--speed", type=float, default=1.0, help="game speed multiplier (default 1 = real time)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)       # headless self-check
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    path = args.model
    if args.file and os.path.isdir(path):
        path = os.path.join(path, args.file + ".zip")
    path = resolve_model(path)
    meta = read_meta(path)
    k = model_obs_grid(path)
    level = args.level or (meta["history"][-1]["level"] if meta.get("history") else None)
    if level is None:
        raise SystemExit("this model has no meta.json: pass --level")
    max_steps = args.max_steps or meta.get("max_steps", 1800)
    model = load_model(path)

    import pygame
    from viewer import MapView

    maps = all_levels()
    view = MapView(f"SMB-RL watch: {os.path.relpath(path)}")
    speed = min(SPEEDS, key=lambda s: abs(s - args.speed))
    paused, acc, frames = False, 0.0, 0
    clock = pygame.time.Clock()

    def start(name):
        envs = [MeatBoyEnv(name, obs_grid=k, max_steps=max_steps, explore_starts=0.0) for _ in range(args.agents)]
        view.set_level(envs[0].level, name)
        return name, envs, np.stack([e.reset()[0] for e in envs]), [Counter() for _ in envs], [None] * len(envs)

    level, envs, obs, tallies, last = start(level)
    while True:
        cmds = view.poll()
        if "quit" in cmds:
            break
        if "pause" in cmds:
            paused = not paused
        if "faster" in cmds or "slower" in cmds:
            i = SPEEDS.index(speed) + (1 if "faster" in cmds else -1)
            speed = SPEEDS[min(max(i, 0), len(SPEEDS) - 1)]
        if "restart" in cmds:
            level, envs, obs, tallies, last = start(level)
        if ("next" in cmds or "prev" in cmds) and maps:
            i = maps.index(level) if level in maps else -1
            level, envs, obs, tallies, last = start(maps[(i + (1 if "next" in cmds else -1)) % len(maps)])

        acc += 0.0 if paused else speed
        while acc >= 1.0:
            acc -= 1.0
            greedy, _ = model.predict(obs[:1], deterministic=True)
            sampled, _ = model.predict(obs[1:], deterministic=False) if len(envs) > 1 else (np.empty(0, int), None)
            for i, (env, a) in enumerate(zip(envs, np.concatenate([greedy, sampled]))):
                o, _, term, trunc, info = env.step(int(a))
                w = env.world
                cx, cy = w.x + w.p.player_w / 2, w.y + w.p.player_h / 2
                if term or trunc:
                    outcome = "won" if info["won"] else "dead" if info["dead"] else "timeout"
                    tallies[i][outcome] += 1
                    last[i] = (outcome, info["steps"])
                    view.end_episode(i, cx, cy, outcome)
                    o, _ = env.reset()
                else:
                    view.track(i, cx, cy)
                obs[i] = o

        g = tallies[0]
        lines = [
            f"model: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
            f"trained on: {', '.join(h['level'] for h in meta.get('history', [])) or '?'}",
            f"agents: {len(envs)}  (1 greedy" + (f", {len(envs) - 1} sampled)" if len(envs) > 1 else ")"),
            f"speed: x{speed:g}" + ("   PAUSED" if paused else ""),
            f"step: {envs[0].world.steps}",
            "",
            "greedy agent (white outline):",
            (f"  won {g['won']}  dead {g['dead']}  timeout {g['timeout']}", (200, 230, 200)),
            (f"  last: {last[0][0]} at step {last[0][1]} ({last[0][1] / REALTIME_HZ:.1f}s)" if last[0] else "", (200, 230, 200)),
        ]
        if len(envs) > 1:
            s = sum(tallies[1:], Counter())
            n = sum(s.values())
            lines += ["", "sampled agents:",
                      (f"  won {s['won']}  dead {s['dead']}  timeout {s['timeout']}"
                       + (f"  ({100 * s['won'] / n:.0f}% won)" if n else ""), (200, 230, 200))]
        lines += ["", ("Space pause   R restart", (150, 150, 165)), ("+/- speed   N/P map   Esc quit", (150, 150, 165))]
        view.frame([(i, e.world.x, e.world.y, e.world.p.player_w, e.world.p.player_h) for i, e in enumerate(envs)],
                   lines, highlight=0, world=envs[0].world)

        frames += 1
        if args.frames and frames >= args.frames:
            if args.screenshot:
                view.screenshot(args.screenshot)
            break
        clock.tick(0 if args.frames else REALTIME_HZ)
    view.close()


if __name__ == "__main__":
    main()
