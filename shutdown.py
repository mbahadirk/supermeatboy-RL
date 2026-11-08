"""Ctrl+C that really ends the process.

agent.py (Shutdown): the first Ctrl+C (or the STOP file, through request()) asks for the clean stop; the run calls
finish() when it is done, which leaves the process at once with os._exit (Frida's own teardown at interpreter
exit can block on its native main loop, which kept the process alive after a clean stop). A second Ctrl+C, or the
clean stop not finished `grace` seconds after it was asked for, runs the `force` callback (best effort, bounded
to `force_s` seconds: agent.py unloads the Frida script so the game thread is never left waiting) and exits with
code 1 at once.

Tools without outside state (viewer, offline planner, model training): immediate() makes Ctrl+C leave at once.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
from typing import Callable


def bounded(fn: Callable[[], object], timeout: float) -> bool:
    """Run fn in a daemon thread for at most `timeout` s; exceptions are swallowed. True if it returned in time."""
    done = threading.Event()

    def body():
        try:
            fn()
        except BaseException:   # best effort: a failing cleanup step must not stop the next one
            pass
        finally:
            done.set()
    threading.Thread(target=body, daemon=True, name="bounded").start()
    return done.wait(timeout)


def _flush():
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:
            pass


class Shutdown:
    def __init__(self, force: Callable[[], object] | None = None, grace: float = 10.0, force_s: float = 2.0,
                 exit_fn: Callable[[int], object] = os._exit, say: Callable[[str], object] | None = None):
        self.force = force
        self.grace = grace
        self.force_s = force_s
        self._exit = exit_fn
        self.say = say or (lambda t: print(t, flush=True))
        self.requested = threading.Event()
        self._timer: threading.Timer | None = None
        self._old = None
        self._forcing = threading.Lock()

    def install(self):
        self._old = signal.signal(signal.SIGINT, self._on_sigint)
        return self

    def _on_sigint(self, *_):
        if self.requested.is_set():
            self.force_exit("second Ctrl+C")
        else:
            self.say("stopping (handing the characters back; Ctrl+C again leaves at once)...")
            self.request()

    def request(self):
        """Ask for the clean stop (Ctrl+C or the STOP file); the watchdog forces the exit after `grace` s."""
        if self.requested.is_set():
            return
        self.requested.set()
        self._timer = threading.Timer(self.grace, self.force_exit,
                                      (f"the clean stop did not finish in {self.grace:g} s",))
        self._timer.daemon = True
        self._timer.start()

    def force_exit(self, why: str):
        if not self._forcing.acquire(blocking=False):
            return
        self.say(f"{why}: unloading the script and leaving now")
        if self.force is not None:
            bounded(self.force, self.force_s)
        _flush()
        self._exit(1)

    def finish(self, code: int = 0):
        """The clean stop is done: leave the process now (no waiting on Frida's exit-time teardown)."""
        if self._timer is not None:
            self._timer.cancel()
        _flush()
        self._exit(code)


def immediate(code: int = 130):
    """Ctrl+C leaves the process at once (also where a GUI library installed its own SIGINT handler)."""
    def on_sigint(*_):
        print("\nCtrl+C: leaving", flush=True)
        _flush()
        os._exit(code)
    signal.signal(signal.SIGINT, on_sigint)
