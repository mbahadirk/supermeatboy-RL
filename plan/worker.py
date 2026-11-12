"""The planner in a child process (agent.py --async-plan): it plans at full speed on its own core while the game
loop keeps answering the game, and the next timed decision can be planned ahead.

PlannerProcess (the game loop's side) sends every planner call through one pipe, so the child runs them in the
order they were made: observations and played actions sent while a plan is being made wait for it, exactly as
if the plan had been made at its frame. Calls are either kept (call(): the loop waits for the reply; send(...,
keep=True) then result()) or fire-and-forget (send(): their replies, the planner's events, are collected by
drain() for the loop to log). A plan made ahead runs with plan.planner.CANCEL set to a shared event, so cancel()
stops it at its next model step. Plans come back without their deferred effects (Plan.commit_fn stays in the
child under the call's id until "commit" or "discard").

The child holds its own Level copy (frame_level applies the frame's tile, field, kill box, platform and lethal
changes as agent.py does), the Planner with its memories, the MoveMemory (saved by the child) and the route /
trial state. The game loop keeps its own Planner for geometry only (support, bounds, terrain, H, f)."""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
import sys
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TubeFn:
    """A picklable deviation threshold for RouteTrials (agent.py's committer tube): max(factor x tube(k), floor)."""

    def __init__(self, tube, factor: float, floor: float):
        self.tube, self.factor, self.floor = tube, float(factor), float(floor)

    def __call__(self, k):
        return max(self.factor * self.tube.at(k), self.floor)


def _plain_plan(plan):
    """The Plan as sent back: its fields only (no commit_fn), plus the route trial event if it has one."""
    return dataclasses.replace(plan), getattr(plan, "route_trial", None)


def serve(conn, cancel, threads: int):
    """The child's loop: (id, name, args, kwargs) in, (id, ok, result or error text) out; None stops it."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl+C reaches the whole console: the loop stops us
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import torch
    torch.set_num_threads(max(1, int(threads)))
    import plan.planner as P
    from plan import safety as S
    planner, lv, moves, pending = None, None, None, {}

    def frame_level(m):
        if m.get("tc"):
            lv.apply_tiles(m["tc"])
            planner.tiles(m["tc"])
        if m.get("fx"):
            lv.set_fields(m["fx"])
        if m.get("kb") and lv.set_kill_box(m["kb"]):
            planner.bounds = S.level_bounds(lv)
        if m.get("pf"):
            lv.set_platforms(m["pf"])
        if m.get("lt"):
            lv.set_lethal(m["lt"]["types"], m["lt"]["ids"])

    while True:
        msg = conn.recv()
        if msg is None:
            break
        mid, name, a, kw = msg
        try:
            res = None
            if name == "init":                                   # model path, level, field, Planner kwargs, moves
                model, lv, field, opts, moves_path, tube = a
                moves = None if moves_path is None else P.MoveMemory.load(moves_path)
                planner = P.Planner(model, lv, field, moves=moves, **opts)
                planner.trial_tube = tube
            elif name == "set_level":
                lv, field, tube = a
                planner.set_level(lv, field)
                planner.trial_tube = tube
            elif name == "set_field":
                field, bounds = a
                planner.set_field(field)
                planner.bounds = bounds
            elif name == "frame_level":
                frame_level(a[0])
            elif name == "moves_learn":
                if moves is not None:
                    moves.learn = bool(a[0])
            elif name == "moves_save":
                if moves is not None and moves.dirty:
                    moves.save()
            elif name == "plan_now":                             # a decision now: its effects at once
                plan = planner.plan(*a, **kw)
                res = _plain_plan(plan)
            elif name == "plan_ahead":                           # row, prefix, worlds, frames, d, target, dt, agent, rec
                row, prefix, worlds, frames, d, target, dt, agent, rec = a
                P.CANCEL = cancel
                try:
                    pred, spread = planner.predict_start(row, prefix, worlds[:d], dt)
                    plan = planner.plan(pred, target, worlds[d:], frames[d:], dt, agent=agent,
                                        recorded=None if rec is None else max(0, int(rec) - d), defer=True)
                finally:
                    P.CANCEL = None
                pending[mid] = plan
                res = (dataclasses.replace(plan), float(pred["x"][0]), float(pred["y"][0]), spread,
                       float(pred["vx"][0]), float(pred["vy"][0]))
            elif name == "commit":
                plan = pending.pop(a[0])
                pending.clear()
                plan.commit_fn()
                res = getattr(plan, "route_trial", None)
            elif name == "discard":
                pending.pop(a[0], None)
            elif name == "surprise":
                res = (planner.surprise.err, planner.surprise.errors())
            elif name == "ping":
                pass
            else:                                                # observe, played, finished, new_wave
                res = getattr(planner, name)(*a, **kw)
            conn.send((mid, True, res))
        except P.PlanCancelled:
            conn.send((mid, False, "cancelled"))
        except BaseException as e:                               # reported to the loop, the child goes on
            conn.send((mid, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        finally:
            if name == "plan_ahead":
                cancel.clear()


class PlannerProcess:
    """The game loop's handle on the child (module docstring)."""

    def __init__(self, threads: int = 2):
        ctx = mp.get_context("spawn")
        self.cancel_event = ctx.Event()
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=serve, args=(child, self.cancel_event, threads), name="planner", daemon=True)
        self.proc.start()
        child.close()
        self.next_id = 0
        self.kinds: dict[int, tuple] = {}          # id -> (call name, kept)
        self.results: dict[int, tuple] = {}        # kept replies: id -> (ok, result)
        self.events: list[tuple] = []              # fire-and-forget replies in order: (name, ok, result)

    def send(self, name: str, *a, keep: bool = False, **kw) -> int:
        mid = self.next_id
        self.next_id += 1
        self.kinds[mid] = (name, keep)
        self.conn.send((mid, name, a, kw))
        return mid

    def drain(self, until: int | None = None):
        """Read the replies there are (until: also wait for this call's reply)."""
        while self.conn.poll() or (until is not None and until not in self.results):
            mid, ok, res = self.conn.recv()
            name, keep = self.kinds.pop(mid)
            if keep:
                self.results[mid] = (ok, res)
            else:
                self.events.append((name, ok, res))

    def ready(self, mid: int) -> bool:
        self.drain()
        return mid in self.results

    def result(self, mid: int):
        """Wait for a kept call's reply: (ok, result or error text)."""
        self.drain(mid)
        return self.results.pop(mid)

    def call(self, name: str, *a, **kw):
        ok, res = self.result(self.send(name, *a, keep=True, **kw))
        if not ok:
            raise RuntimeError(f"planner process, {name}: {res}")
        return res

    def cancel(self, mid: int):
        """Stop a plan made ahead (at its next model step) and wait for its reply; its plan is dropped."""
        self.cancel_event.set()
        self.result(mid)
        self.cancel_event.clear()               # (it may have finished before the request: nothing else to stop)
        self.send("discard", mid)

    def take_events(self) -> list:
        self.drain()
        ev, self.events = self.events, []
        return ev

    def close(self):
        try:
            self.conn.send(None)
            self.proc.join(5)
        except (OSError, EOFError, BrokenPipeError):
            pass
        if self.proc.is_alive():                   # our own child only (never the game)
            self.proc.terminate()
            self.proc.join(2)
