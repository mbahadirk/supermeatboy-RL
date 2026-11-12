"""One MPC decision: candidates -> world-model rollouts (all members) -> scores -> the best plan's first action.

Receding horizon: every decision plans H frames ahead and only the first action is played (or the first
`every` actions, with --mpc-every). The game's time step is locked, so every future step is the step the game
reports now (checked: a different step is logged).

A plan that could not be finished is never passed off as evaluated: with a time budget the rollout stops at
the deadline and the plan is scored on the steps it has (info["steps"]); in mpc+ppo mode a plan with fewer
than MIN_STEPS steps, one the members already disagree on after one step, or a decision where every
candidate looks fatal (ALL_FATAL) hands the decision to PPO (info["fallback"] says why). In mpc mode the
planner still picks (the latest death wins, mpc.score) and those decisions are counted. Every decision is written to the plan log (one JSON line).
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import torch

from mpc import candidates
from mpc.future import FutureWorld
from mpc.score import Weights, score, TILE
from wm.buffer import ROW
from wm.features import LevelGeom, model_layout
from wm.model import WorldModel
from wm.rollout import rollout_batched

MIN_STEPS = 4            # mpc+ppo: a plan with fewer rolled-out steps than this goes to PPO
UNSURE_STEP1 = TILE      # mpc+ppo: members already this far apart after one step: PPO decides
ALL_FATAL = 0.9          # every candidate at least this likely to die: counted; in mpc+ppo PPO decides (the
                         # model is likely wrong about this level's hazards, or there is no way out it can see)
REACH = 700.0            # units/s: hazards further than this x the horizon (+ their size) are not checked
HAZ_FAR = 300.0          # wm.features.hazard_features: offsets and distances are clipped at 3 x 100 units


class MPC:
    def __init__(self, model_path: str, n_cand: int = 32, horizon: int = 16, every: int = 1,
                 budget_ms: float = 0.0, weights: Weights | None = None, seed: int = 0, fallback: bool = False,
                 log_path: str | None = None):
        self.model = WorldModel.load(model_path).eval()
        self.model_path = model_path
        self.n, self.H, self.every, self.budget = n_cand, horizon, max(1, every), budget_ms
        self.w = weights or Weights()
        self.seed, self.fallback_on = seed, fallback
        self.future = FutureWorld()
        self.geom = None
        self.plans: dict[int, tuple[int, np.ndarray]] = {}    # agent -> (frame planned, its action sequence)
        self.rng = np.random.default_rng(seed)
        self.dt = None
        self.dt_warned = False
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None
        self.stats = self._new_stats()
        self.wave = 0

    @staticmethod
    def _new_stats():
        return {"plans": 0, "reused": 0, "ms": [], "fallbacks": {}, "short": 0, "all_fatal": 0}

    # ---- level / wave / frame ------------------------------------------------------------------------
    def set_level(self, lv):
        self.geom = LevelGeom(lv.words0, lv.wm_geometry_metadata(),               # the model's tile encoding
                              protocol=model_layout(self.model).protocol)
        self.future.new_level()
        self.plans = {}
        self.level_name = lv.name

    def new_wave(self, wave: int):
        self.future.new_wave()
        self.plans = {}
        self.wave = wave
        self.rng = np.random.default_rng(self.seed * 100003 + wave)   # reproducible per wave
        st, self.stats = self.stats, self._new_stats()
        return st

    def frame(self, m: dict, keep):
        """Every frame before planning: tile changes, the level's moving parts, the time step. keep: per piece,
        not decoration (Level.hz_keep: the model inputs, and rows without kill shapes)."""
        if self.geom is None:
            return
        for idx, word in m.get("tc") or []:
            self.geom.set_tile(idx, word)
        fans = m["fx"].get("fans", []) if m.get("fx") else None
        dt = float(m.get("dt") or 16.667)
        if self.dt is not None and abs(dt - self.dt) > 0.5 and not self.dt_warned:
            self.dt_warned = True
            print(f"[mpc] the game's time step changed: {self.dt:.2f} -> {dt:.2f} ms (is the step lock off?)", flush=True)
        self.dt = dt
        self.future.observe(m["frame"], m.get("hz", []), m.get("pf"), m.get("sh", []), keep, fans, dt)

    # ---- one decision --------------------------------------------------------------------------------
    def plan(self, agent: int, s: list, held: int, frame: int, route, reached, lv, probs=None, value_fn=None,
             ppo_action: int | None = None) -> tuple[int, dict]:
        """s: the agent's state as the game sent it [x, y, vx, vy, flags, state, status, keys]."""
        t0 = time.perf_counter()
        old = self.plans.get(agent)
        if old is not None and frame - old[0] < self.every and frame - old[0] < len(old[1]):
            self.stats["reused"] += 1
            return int(old[1][frame - old[0]]), {"reused": True, "frame": frame, "agent": agent}
        prev = old[1][frame - old[0]:] if old is not None and frame - old[0] < len(old[1]) else None
        H, dt = self.H, self.dt or 16.667
        seqs, labels = candidates.make(self.n, H, self.rng, prev, probs)
        t1 = time.perf_counter()
        worlds, check, recorded = self.future.worlds(frame, H, dt)
        x0, y0 = float(s[0]), float(s[1])
        reach = REACH * H * dt / 1000.0
        check = [(b[_near(b, x0, y0, reach)], r[_near(b, x0, y0, reach)]) for b, r, _ in check]
        # the model sees the 3 nearest hazards and anything 300+ units away as "far" (wm.features): hazards
        # that cannot come that close during the plan are left out of its inputs (the 3 nearest always stay)
        worlds = [replace(w, boxes=w.boxes[_relevant(w.boxes, x0, y0, reach + HAZ_FAR)]) for w in worlds]
        row = np.zeros(1, ROW)
        row["x"], row["y"], row["vx"], row["vy"] = s[0], s[1], s[2], s[3]
        row["flags"], row["state"], row["status"] = int(s[4]) & 0xffffffff, s[5], 0
        row["keys"], row["held"] = (int(s[7]) if len(s) > 7 else 0), held
        n = len(seqs)
        t2 = time.perf_counter()
        deadline = t0 + self.budget / 1000.0 if self.budget > 0 else None
        out = rollout_batched(self.model, self.geom, np.repeat(row, n), seqs, np.full(H, dt), worlds, deadline,
                              keep_rows=value_fn is not None)
        t3 = time.perf_counter()
        leg = route.leg(reached) if route is not None else 0
        wp = route.waypoints[leg][1] if route is not None and leg < len(route.waypoints) else None
        rem_fn = _remaining(route, leg, lv)
        rem_next = _remaining(route, leg + 1, lv) if wp is not None else None
        value_end = None
        if value_fn is not None and out["steps"] > 0:
            end = out["rows"][:, :, out["steps"]]                  # (E, n): the members' mean end state
            mean = end[0].copy()
            for name in ("x", "y", "vx", "vy"):
                mean[name] = end[name].mean(0)
            value_end = np.broadcast_to(np.asarray(value_fn(mean), np.float64), (end.shape[0], n))
        t4 = time.perf_counter()
        sc = score(out, rem_fn, self.w, check, rem_next, wp, value_end)
        best = int(np.argmax(sc["total"]))
        t5 = time.perf_counter()
        action = int(seqs[best, 0])
        all_fatal = bool(sc["death"].min() >= ALL_FATAL)
        if all_fatal:
            self.stats["all_fatal"] += 1
        fallback = None
        if self.fallback_on and ppo_action is not None:
            if sc["steps"] < MIN_STEPS:
                fallback = "budget"
            elif sc["spread"][best, 1] > UNSURE_STEP1:
                fallback = "unsure"
            elif all_fatal:
                fallback = "all fatal"
            if fallback:
                action = int(ppo_action)
                self.stats["fallbacks"][fallback] = self.stats["fallbacks"].get(fallback, 0) + 1
        if sc["steps"] < H:
            self.stats["short"] += 1
        self.plans[agent] = (frame, seqs[best] if not fallback else np.concatenate([[action], seqs[best][1:]]))
        ms = 1000 * (time.perf_counter() - t0)
        self.stats["plans"] += 1
        self.stats["ms"].append(ms)
        order = np.argsort(sc["total"])[::-1]
        info = {"frame": frame, "wave": self.wave, "agent": agent, "x": x0, "y": y0, "vx": float(s[2]),
                "vy": float(s[3]), "n": n, "H": H, "steps": sc["steps"], "recorded": recorded,
                "future_trusted": self.future.trusted, "all_fatal": all_fatal, "chosen": best, "label": labels[best], "action": action,
                "fallback": fallback, "seq": seqs[best][:sc["steps"]].tolist(),
                "ms": {"total": round(ms, 2), "candidates": round(1000 * (t1 - t0), 2),
                       "future": round(1000 * (t2 - t1), 2), "rollout": round(1000 * (t3 - t2), 2),
                       "value": round(1000 * (t4 - t3), 2), "score": round(1000 * (t5 - t4), 2)},
                "path": [[round(float(a), 1), round(float(b), 1)] for a, b in
                         zip(sc["mean_x"][best], sc["mean_y"][best])],
                "spread": [round(float(v), 2) for v in sc["spread"][best]],
                "landing_step": int(sc["landing_step"][best]),
                "top": [{"i": int(i), "label": labels[i], **{k: round(float(sc[k][i]), 3) for k in
                         ("total", "progress", "key", "land", "value", "unc", "death", "death_cost", "death_step",
                                    "death_model", "death_geom")}}
                        for i in order[:5]]}
        info["candidates"] = {"x": sc["mean_x"], "y": sc["mean_y"], "spread": sc["spread"], "death": sc["death"],
                              "total": sc["total"], "labels": labels}    # (not logged: the viewer's)
        if self.log is not None:
            self.log.write(json.dumps({k: v for k, v in info.items() if k != "candidates"}) + "\n")
            if self.stats["plans"] % 50 == 0:
                self.log.flush()
        return action, info

    def summary(self, st: dict) -> str:
        if not st["plans"]:
            return ""
        ms = np.array(st["ms"])
        fb = sum(st["fallbacks"].values())
        return (f"MPC {st['plans']} plans (+{st['reused']} reused), {np.median(ms):.0f} ms median / "
                f"{np.percentile(ms, 95):.0f} ms 95%" + (f", {st['short']} cut short by the budget" if st["short"] else "")
                + (f", every option looked fatal {st['all_fatal']} times" if st["all_fatal"] else "")
                + (f", {fb} to PPO ({', '.join(f'{k} {v}' for k, v in st['fallbacks'].items())})" if fb else ""))

    def close(self):
        if self.log is not None:
            self.log.close()
            self.log = None


def _near(boxes: np.ndarray, x: float, y: float, reach: float) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0, bool)
    size = np.hypot(boxes[:, 2], boxes[:, 3])
    return np.hypot(boxes[:, 0] - x, boxes[:, 1] - y) < reach + size + 40.0


def _relevant(boxes: np.ndarray, x: float, y: float, reach: float) -> np.ndarray:
    """Hazards within reach (+ their size), and always the 3 nearest (the model's 3 hazard slots)."""
    keep = _near(boxes, x, y, reach)
    if len(boxes) > 3:
        keep[np.argsort(np.hypot(boxes[:, 0] - x, boxes[:, 1] - y))[:3]] = True
    else:
        keep[:] = True
    return keep


def _remaining(route, leg: int, lv):
    """f(x, y) -> the way left to the goal (game units): along the route's closeness field; outside it the
    walking distance to the goal plus the route's length (worse than any point the field reaches)."""
    def goal_way(x, y):
        c = np.floor((x - lv.ox) / TILE).astype(np.int64)
        r = np.floor((y - lv.oy) / TILE).astype(np.int64)
        ok = (c >= 0) & (c < lv.W) & (r >= 0) & (r < lv.H)
        d = np.full(x.shape, np.inf)
        d[ok] = lv.dist[r[ok], c[ok]]
        if lv.goals:
            gx, gy = lv.goals[0]
            far = np.hypot(gx - x, gy - y) / TILE + lv.diag
        else:
            far = np.full(x.shape, lv.diag)
        return np.where(np.isfinite(d), d, far) * TILE

    if route is None:
        return goal_way
    L = route.length

    def f(x, y):
        v = route.values(x.ravel(), y.ravel(), leg).reshape(x.shape)
        return np.where(np.isnan(v), L + goal_way(x, y), L - v)
    return f
