"""The planner in a child process (agent.py --async-plan): it plans at full speed on its own core while the game
loop keeps answering the game, and the next timed decision can be planned ahead.

PlannerProcess (the game loop's side) sends every planner call through one pipe (read at once by the child's
reader thread; the replies come back through another), so the child runs them in the order they were made: observations and played actions sent while a plan is being made wait for it, exactly as
if the plan had been made at its frame. Calls are either kept (call(): the loop waits for the reply; send(...,
keep=True) then result()) or fire-and-forget (send(): their replies, the planner's events, are collected by
drain() for the loop to log). A plan made ahead runs with plan.planner.CANCEL set to a shared event, so cancel()
stops it at its next model step. Plans come back without their deferred effects (Plan.commit_fn stays in the
child under the call's id until "commit" or "discard").

The child holds its own Level copy (frame_level applies the frame's tile, field, kill box, platform and lethal
changes as agent.py does), the Planner with its memories, the MoveMemory (saved by the child; "moves_learn" turns
its charging on / off, "moves_transient" names the levels it never saves: agent.py's held-out levels) and the route /
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


def plain_plan(plan):
    """The Plan as sent between processes: its fields and its effects as data (Plan._effects, for
    Planner.adopt), not its commit_fn."""
    p = dataclasses.replace(plan)
    if hasattr(plan, "_effects"):
        p._effects = plan._effects
    return p


def _plain_plan(plan):
    """The Plan as sent back: plain_plan, plus the route trial event if it has one."""
    return plain_plan(plan), getattr(plan, "route_trial", None)


def _reader(conn_in, q):
    """The child's reader thread: every call read from the loop's pipe at once, in order, into q (None: stop), so
    the loop's sends never wait for the child's planning. Without it the two sides could block each other: the loop
    sending a big call (an adopted plan) to a child busy planning, the child then sending its big reply to a loop
    not reading (seen with --planners 2: all three processes idle for good)."""
    while True:
        try:
            msg = conn_in.recv()
        except (EOFError, OSError):
            msg = None
        q.put(msg)
        if msg is None:
            return


def serve(conn_in, conn_out, cancel, threads: int):
    """The child's loop: (id, name, args, kwargs) in (conn_in, read by _reader), (id, ok, result or error text) out
    (conn_out); None stops it."""
    import queue
    import signal
    import threading
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl+C reaches the whole console: the loop stops us
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import torch
    torch.set_num_threads(max(1, int(threads)))
    import plan.planner as P
    from plan import safety as S
    planner, lv, moves, pending = None, None, None, {}
    inbox = queue.Queue()
    threading.Thread(target=_reader, args=(conn_in, inbox), name="planner-reader", daemon=True).start()
    conn = conn_out

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
        msg = inbox.get()
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
            elif name == "set_model":                          # a newer world model (agent.py --online / --learn)
                from wm.model import WorldModel
                planner.model = WorldModel.load(a[0]).eval()
            elif name == "moves_learn":
                if moves is not None:
                    moves.learn = bool(a[0])
            elif name == "moves_transient":                    # level names never saved (agent.py: held-out)
                if moves is not None:
                    moves.transient = set(a[0])
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
                    pred, spread = planner.predict_start(row, prefix, worlds[:d], dt, frame=int(target) - int(d))
                    plan = planner.plan(pred, target, worlds[d:], frames[d:], dt, agent=agent,
                                        recorded=None if rec is None else max(0, int(rec) - d), defer=True)
                finally:
                    P.CANCEL = None
                pending[mid] = plan
                res = (plain_plan(plan), float(pred["x"][0]), float(pred["y"][0]), spread,
                       float(pred["vx"][0]), float(pred["vy"][0]))
            elif name == "commit":
                plan = pending.pop(a[0])
                pending.clear()
                plan.commit_fn()                                 # Planner.adopt of its own plan
                res = getattr(plan, "route_trial", None)
            elif name == "adopt":                                # a plan another process made and committed
                res = planner.adopt(a[0])                        # (pending plans here stay: PlannerPool)
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
        child_in, self._out = ctx.Pipe(duplex=False)       # calls: the loop writes, the child's reader reads
        self.conn, child_out = ctx.Pipe(duplex=False)      # replies: the child writes, the loop reads
        self.proc = ctx.Process(target=serve, args=(child_in, child_out, self.cancel_event, threads), name="planner",
                                daemon=True)
        self.proc.start()
        child_in.close()
        child_out.close()
        self.next_id = 0
        self.kinds: dict[int, tuple] = {}          # id -> (call name, kept)
        self.results: dict[int, tuple] = {}        # kept replies: id -> (ok, result)
        self.events: list[tuple] = []              # fire-and-forget replies in order: (name, ok, result)

    def send(self, name: str, *a, keep: bool = False, **kw) -> int:
        mid = self.next_id
        self.next_id += 1
        self.kinds[mid] = (name, keep)
        self._out.send((mid, name, a, kw))
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
        """Stop a plan made ahead (at its next model step) and wait for its reply; its plan is dropped. Only this
        process is waited for (each process has its own cancel event)."""
        self.cancel_begin()
        self.cancel_end(mid)

    def cancel_begin(self):
        """cancel(), first half: ask the plan being made to stop (PlannerPool.cancel_many asks several at once)."""
        self.cancel_event.set()

    def cancel_end(self, mid: int):
        """cancel(), second half: wait for the plan's reply, clear the request, drop the plan."""
        self.result(mid)
        self.cancel_event.clear()               # (it may have finished before the request: nothing else to stop)
        self.send("discard", mid)

    def take_events(self) -> list:
        self.drain()
        ev, self.events = self.events, []
        return ev

    def close(self):
        try:
            self._out.send(None)
            self.proc.join(5)
        except (OSError, EOFError, BrokenPipeError):
            pass
        if self.proc.is_alive():                   # our own child only (never the game)
            self.proc.terminate()
            self.proc.join(2)


class PlannerPool:
    """N planner processes (agent.py --planners N) behind PlannerProcess's interface; process 0 is the primary.

    Every call that changes planner state (init, set_level, set_field, frame_level, set_model, moves_learn,
    moves_transient, observe, played, finished, new_wave, ping and any other name not listed below) is broadcast
    to every process in the order it is made, so every process runs the same calls in the same order. Only the
    primary's replies are used (a kept call waits for the primary alone); the other processes' replies are
    drained by take_events and dropped, their errors kept as events naming the process ("observe (planner 1)").
    Plans: plan_ahead goes to the process `to` (default the primary), plan_now to the primary, commit / discard /
    cancel to the process that made the plan. A plan committed by its process ("commit", after its result was
    read here) or made at once ("plan_now") goes to every other process as "adopt" (Planner.adopt: the same
    deferred effects, from Plan._effects), at the same place in their call order, so the processes' planner states
    stay identical. moves_save and surprise go to the primary only: it alone saves the move memory (every process
    charges it identically). Ids returned by send are the pool's own."""
    PRIMARY = ("moves_save", "surprise")

    def __init__(self, n: int, threads: int = 2, make=None):
        make = make or PlannerProcess
        self.procs = [make(threads) for _ in range(max(1, int(n)))]
        self.next_id = 0
        self.route: dict[int, tuple] = {}         # pool id -> (process index, its id there, call name)
        self.plans: dict[int, object] = {}        # plan_ahead ids whose result was read -> the Plan (for adopt)

    @property
    def n(self) -> int:
        return len(self.procs)

    @property
    def proc(self):
        """The primary's multiprocessing.Process (agent.py prints its pid)."""
        return self.procs[0].proc

    def owner(self, mid: int) -> int:
        """The process a kept call (a plan made ahead) runs in."""
        return self.route[mid][0]

    def _id(self, k: int, local: int, name: str) -> int:
        mid = self.next_id
        self.next_id += 1
        self.route[mid] = (k, local, name)
        return mid

    def adopt_others(self, plan, k: int):
        for j, p in enumerate(self.procs):
            if j != k:
                p.send("adopt", plan)

    def send(self, name: str, *a, keep: bool = False, to: int | None = None, **kw) -> int:
        if name == "plan_ahead":
            k = 0 if to is None else int(to)
            return self._id(k, self.procs[k].send(name, *a, keep=keep, **kw), name)
        if name in ("commit", "discard"):
            k, local, _ = self.route.pop(a[0])
            plan = self.plans.pop(a[0], None)
            if name == "commit" and plan is None:
                raise RuntimeError("PlannerPool: commit of a plan whose result was not read here")
            mid = self._id(k, self.procs[k].send(name, local, *a[1:], keep=keep, **kw), name)
            if name == "commit":                  # the others adopt it at this place in their call order
                self.adopt_others(plan, k)
            if not keep:
                self.route.pop(mid, None)
            return mid
        if name == "plan_now" or name in self.PRIMARY:
            mid = self._id(0, self.procs[0].send(name, *a, keep=keep, **kw), name)
        else:                                     # broadcast; the others' replies are dropped (take_events)
            mid = self._id(0, self.procs[0].send(name, *a, keep=keep, **kw), name)
            for p in self.procs[1:]:
                p.send(name, *a, **kw)
        if not keep:
            self.route.pop(mid, None)
        return mid

    def ready(self, mid: int) -> bool:
        k, local, _ = self.route[mid]
        return self.procs[k].ready(local)

    def result(self, mid: int):
        """Wait for a kept call's reply: (ok, result or error text). A plan made ahead stays known here until it is
        committed, discarded or cancelled; a plan made now is adopted by the other processes at once."""
        k, local, name = self.route[mid]
        ok, res = self.procs[k].result(local)
        if name == "plan_ahead" and ok:
            self.plans[mid] = res[0]
        else:
            self.route.pop(mid, None)
        if name == "plan_now" and ok:
            self.adopt_others(res[0], 0)
        return ok, res

    def call(self, name: str, *a, **kw):
        ok, res = self.result(self.send(name, *a, keep=True, **kw))
        if not ok:
            raise RuntimeError(f"planner process, {name}: {res}")
        return res

    def cancel(self, mid: int):
        """Stop a plan made ahead in its own process (only that one is waited for); its plan is dropped."""
        self.cancel_many([mid])

    def cancel_many(self, mids):
        """Stop several plans made ahead: every process is asked first, then each reply is waited for, so the wait
        is the slowest one's, not their sum."""
        jobs = []
        for mid in mids:
            k, local, _ = self.route.pop(mid)
            self.plans.pop(mid, None)
            self.procs[k].cancel_begin()
            jobs.append((k, local))
        for k, local in jobs:
            self.procs[k].cancel_end(local)

    def take_events(self) -> list:
        ev = self.procs[0].take_events()
        for k, p in enumerate(self.procs[1:], 1):
            for name, ok, res in p.take_events():
                if not ok:
                    ev.append((f"{name} (planner {k})", ok, res))
        return ev

    def close(self):
        for p in self.procs:
            p.close()
