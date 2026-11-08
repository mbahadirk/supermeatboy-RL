"""Ctrl+C exits: the clean stop finishes with no non-daemon threads left, the force path unloads then exits."""
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent                     # noqa: E402
from shutdown import Shutdown    # noqa: E402


class FakeScript:
    def __init__(self, hang=False):
        self.calls = []
        self.hang = hang

        class Exports:
            def shutdown(_):
                self.calls.append("shutdown")
                if self.hang:
                    time.sleep(60)
        self.exports_sync = Exports()

    def unload(self):
        self.calls.append("unload")
        if self.hang:
            time.sleep(60)


class FakeSession:
    def __init__(self, hang=False):
        self.detached = False
        self.hang = hang

    def detach(self):
        self.detached = True
        if self.hang:
            time.sleep(60)


def test_clean_stop_finishes_and_leaves_no_non_daemon_threads():
    exits = []
    sd = Shutdown(exit_fn=exits.append, say=lambda t: None)
    before = {t.ident for t in threading.enumerate()}
    script, session, closed = FakeScript(), FakeSession(), []
    stopped = threading.Event()
    stopped.set()                              # the frame reply carried the stop
    sd.request()
    t0 = time.time()
    agent.clean_stop(script, session, stopped, lambda: closed.append(1))
    sd.finish(0)
    assert time.time() - t0 < 5
    assert script.calls == ["unload"] and session.detached and closed == [1]
    assert exits == [0]
    extra = [t for t in threading.enumerate() if t.ident not in before and not t.daemon]
    assert extra == []


def test_clean_stop_is_bounded_when_frida_hangs():
    script, session = FakeScript(hang=True), FakeSession(hang=True)
    t0 = time.time()
    agent.clean_stop(script, session, threading.Event(), lambda: None, wait_s=0.1)
    assert time.time() - t0 < 6                # every Frida call is bounded
    assert script.calls == ["shutdown", "unload"] and session.detached
    assert all(t.daemon for t in threading.enumerate() if t is not threading.main_thread())


def test_second_ctrl_c_unloads_then_exits():
    order = []
    script = FakeScript()
    sd = Shutdown(force=lambda: (script.unload(), order.append("unload")), exit_fn=lambda c: order.append(("exit", c)),
                  say=lambda t: None)
    old = signal.getsignal(signal.SIGINT)
    try:
        sd.install()
        sd._on_sigint()                        # first: only a request
        assert sd.requested.is_set() and order == []
        sd._on_sigint()                        # second: unload, exit 1
        assert order == ["unload", ("exit", 1)]
    finally:
        signal.signal(signal.SIGINT, old)
        sd._timer.cancel()


def test_watchdog_forces_exit_when_the_clean_stop_hangs():
    order = []
    sd = Shutdown(force=lambda: order.append("unload"), grace=0.2, exit_fn=lambda c: order.append(("exit", c)),
                  say=lambda t: None)
    sd.request()
    time.sleep(0.6)
    assert order == ["unload", ("exit", 1)]
