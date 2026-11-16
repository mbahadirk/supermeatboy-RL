"""plan.planner.RouteBook: every finished wave of a level, joined at portals into one composite route (synthetic
records, no world model)."""
import types

import numpy as np
import pytest

from plan import planner as P
from plan import safety as S
from plan.planner import PORTAL_SPEED, RouteBook
from test_key_pickup import synthetic
from wm.buffer import ROW


def feed(book, states, finish=None, agent=0, held=0, keys=0, tiles=None, action=6):
    """One wave: states [(x, y, vx, vy), ...] per frame from 0 (`action` each), per-frame overrides as dicts
    {frame: value} for held / keys, tiles {frame: changes} applied before that frame's state. Returns finished()."""
    book.new_wave()
    for f, (x, y, vx, vy) in enumerate(states):
        if tiles and f in tiles:
            book.tiles(tiles[f])
        h = held.get(f, 0) if isinstance(held, dict) else held
        k = keys.get(f, 0) if isinstance(keys, dict) else keys
        book.observe(agent, f, x, y, vx, vy, keys=k, held=h)
        book.played(agent, f, action)
    return book.finished(agent, len(states) - 1 if finish is None else finish)


def line(n, x0=0.0, y=0.0, vx=1.0):
    return [(x0 + vx * f, y, vx, 0.0) for f in range(n)]


# A: fast to the shared state S (x 100, vx 5) at its frame 4, then slow to its finish at 24;
# B: slow to S at its frame 14, then fast to its finish at 18. Only S is shared (velocities differ elsewhere).
SHARED = (100.0, 0.0, 5.0, 0.0)


def wave_a():
    return [(25.0 * f, 0.0, 25.0, 0.0) for f in range(4)] + [SHARED] + \
           [(100.0 + 5.0 * k, 0.0, 5.0, 0.0) for k in range(1, 21)]          # frames 5..24


def wave_b(shared=SHARED):
    return [(100.0 * f / 14, 0.0, 7.0, 0.0) for f in range(14)] + [shared] + \
           [(100.0 + 50.0 * k, 0.0, 50.0, 0.0) for k in range(1, 5)]         # frames 15..18


def test_a_single_record_is_the_composite():
    book = RouteBook()
    assert book.match(0.0, 0.0, 0.0, 0.0, 0) is None and book.best is None
    assert feed(book, line(11, vx=3.0))
    b = book.best
    assert b["finish"] == 10 and b["f"].tolist() == list(range(11)) and b["w"].tolist() == list(range(11))
    assert b["x"].tolist() == [3.0 * f for f in range(11)] and b["a"].tolist() == [6] * 11
    assert b["segments"] == [[0, 0, 10]] and not b["jump"].any() and not book.composite_at(5)
    assert book.summary["records"] == 1 and book.summary["portals"] == 0 and book.summary["best_real"] == 10


def test_b_two_records_join_at_their_shared_state():
    book = RouteBook()
    assert feed(book, wave_a())                                       # A: record 0, finish 24
    assert feed(book, wave_b())                                       # B: record 1, finish 18 (a new best real)
    b = book.best
    assert b["finish"] == 4 + 4                                       # A's 4 frames to S + B's 4 from S
    assert b["segments"] == [[0, 0, 3], [1, 14, 18]]
    assert b["f"].tolist() == list(range(9)) and b["w"].tolist() == [0, 1, 2, 3, 14, 15, 16, 17, 18]
    assert b["rec"].tolist() == [0] * 4 + [1] * 5
    assert b["x"][4] == SHARED[0] and b["jump"][4] == 0.0                # an exact portal: no jump
    s = book.summary
    assert s["finish"] == 8 and s["best_real"] == 18 and s["records"] == 2 and s["portals"] == 1
    assert book.composite_at(0) and not book.composite_at(5)          # A's rows are not the fastest real record
    a, n = book.actions(0, 48)
    assert n == 9 and a.tolist() == [6] * 9


@pytest.mark.parametrize("change", ["keys", "held", "tiles", "speed", "unknown held", "unknown keys"])
def test_c_no_portal_across_keys_held_tiles_or_speed(change):
    level = types.SimpleNamespace(keys=[[0, 0]], plats=None, words0=np.zeros((2, 5), np.uint32))
    book = RouteBook(level)
    assert feed(book, wave_a())
    shared = SHARED
    kw = {}
    if change == "keys":
        kw["keys"] = {14: 1}
    elif change == "held":
        kw["held"] = {14: 2}
    elif change == "tiles":
        kw["tiles"] = {14: [(3, 7)], 15: [(3, 0)]}                    # cell 3 broken at S only, then whole again
    elif change == "speed":
        shared = (SHARED[0], SHARED[1], SHARED[2] + PORTAL_SPEED + 0.01, SHARED[3])
    elif change == "unknown held":
        kw["held"] = {14: None}
    elif change == "unknown keys":
        kw["keys"] = {14: None}
    assert feed(book, wave_b(shared), **kw)
    assert book.summary["portals"] == 0 and book.best["finish"] == 18 and book.best["segments"] == [[1, 0, 18]]


def test_c_tile_signature_follows_the_tile_state():
    level = types.SimpleNamespace(keys=(), plats=None, words0=np.zeros((2, 5), np.uint32))
    book = RouteBook(level)
    book.tiles([(3, 7), (4, 9)])
    s = book.sig
    assert s != 0
    book.tiles([(3, 0)])
    book.tiles([(3, 7)])
    assert book.sig == s                                              # the same tiles: the same signature
    book.tiles([(4, 0), (3, 0)])                                      # restored (the next wave's frame 0)
    assert book.sig == 0


def test_d_moving_platforms_join_only_at_the_same_wave_frame():
    level = types.SimpleNamespace(keys=(), plats=[object()], words0=None)
    book = RouteBook(level)
    assert feed(book, wave_a())
    assert feed(book, wave_b())                                       # S at frames 4 and 14: no portal
    assert book.summary["portals"] == 0 and book.best["finish"] == 18
    book = RouteBook(level)
    assert feed(book, wave_a())
    c = [(25.0 * f, 40.0, 25.0, 0.0) for f in range(4)] + [SHARED] + [(100.0 + 9.0 * k, 0.0, 9.0, 0.0)
                                                                      for k in range(1, 13)]
    assert feed(book, c)                                              # S at frame 4 too (finish 16)
    assert book.summary["portals"] == 1
    book = RouteBook(types.SimpleNamespace(keys=(), plats=None, words0=None))
    assert feed(book, wave_a())
    assert feed(book, wave_b())                                       # without platforms the same pair joins
    assert book.summary["portals"] == 1 and book.best["finish"] == 8


def test_e_the_book_drops_the_slowest_finish(monkeypatch):
    monkeypatch.setattr(P, "ROUTE_BOOK_MAX", 3)
    book = RouteBook()
    for n in (10, 20, 15, 30):
        feed(book, line(n + 1, y=float(n)), held=None)                 # held unknown: no portals
    assert [r["finish"] for r in book.records] == [10, 20, 15]       # the 4th (slowest) not kept
    feed(book, line(13, y=12.0), held=None)
    assert [r["finish"] for r in book.records] == [10, 15, 12]
    assert book.summary["records"] == 3


def test_e_records_the_composite_uses_are_kept_over_faster_unused_ones(monkeypatch):
    monkeypatch.setattr(P, "ROUTE_BOOK_MAX", 2)
    book = RouteBook()
    assert feed(book, wave_a()) and feed(book, wave_b())             # A (24) and B (18): composite 8 from both
    assert not feed(book, line(21, y=500.0), held=None)              # C (20): faster than A, but no portal
    assert sorted(r["finish"] for r in book.records) == [18, 24]     # C went, not the slowest (A)
    assert book.best["finish"] == 8 and book.best["segments"] == [[0, 0, 3], [1, 14, 18]]
    assert book.summary["records"] == 2 and book.summary["portals"] == 1
    monkeypatch.setattr(P, "ROUTE_BOOK_MAX", 1)                       # every record contributes: the slowest goes
    assert not feed(book, line(31, y=600.0), held=None)
    assert [r["finish"] for r in book.records] == [18] and book.best["finish"] == 18
    assert book.summary["portals"] == 0


def test_identical_waves_are_counted_not_added():
    book = RouteBook()
    assert feed(book, wave_a())
    s = book.summary
    assert not feed(book, wave_a()) and len(book.records) == 1      # the same finish and actions: a duplicate
    assert book.summary["duplicates"] == 1 and book.summary is not s and book.best["finish"] == 24
    assert not feed(book, wave_a(), action=7) and len(book.records) == 2     # other actions: a record
    assert not feed(book, wave_a(), finish=25) and len(book.records) == 3    # another finish frame: a record
    assert book.summary["duplicates"] == 1 and [r["id"] for r in book.records] == [0, 1, 2]


def test_f_finished_is_true_only_for_a_new_best_real_finish():
    book = RouteBook()
    assert feed(book, line(21, y=0.0), held=None)                     # 20: the first
    assert not feed(book, line(26, y=1.0), held=None)                 # 25: slower
    assert not feed(book, line(21, y=2.0), held=None, action=7)       # 20: equal, not faster
    assert feed(book, line(16, y=3.0), held=None)                     # 15: faster
    assert book.best_real == 15 and len(book.records) == 4
    book.new_wave()
    book.observe(0, 0, 0.0, 0.0, None, None)                          # no velocities: not recorded
    assert not book.finished(0, 0) and len(book.records) == 4


def test_g_died_at_is_reset_only_when_the_composite_finish_changes():
    book = RouteBook()
    assert feed(book, wave_a())
    comp = book.best
    book.using[0] = (10, 2)                                           # adopted at wave frame 10 from row 2
    assert book.dead(0, 13) == 5 and book.died_at == 5
    assert not feed(book, [(x, y + 50.0, vx, vy) for x, y, vx, vy in line(31)])    # slower, no portal
    assert book.best is comp and book.died_at == 5                    # the same composite: still marked
    assert feed(book, wave_b())                                       # joins A: composite 8 frames
    assert book.best["finish"] == 8 and book.died_at is None


# ---- the planner's use of a composite ------------------------------------------------------------------------
def _composite(n=21, jump_at=None, w=None):
    b = {"finish": 400, "f": np.arange(n), "x": np.full(n, 100.0), "y": np.full(n, 30.0), "vx": np.zeros(n),
         "vy": np.zeros(n), "a": np.full(n, 6), "keys": np.zeros(n, np.uint32), "keys_known": np.ones(n, bool),
         "w": np.arange(n) if w is None else np.asarray(w), "rec": np.zeros(n, np.int64),
         "jump": np.zeros(n), "jump_v": np.zeros(n), "segments": [[0, 0, n - 1]]}
    if jump_at is not None:
        b["jump"][jump_at], b["jump_v"][jump_at] = 0.5, 0.25
    return b


def test_real_path_margin_grows_with_a_crossed_portal():
    """An exact match is judged on contact only; past a portal the margin is the jump (+ its speed x frames),
    so a box 2 units from the recorded path costs soft risk there."""
    _, _, pl = synthetic()
    pl.route = RouteBook()
    def box_at(d):
        return S.FrameBoxes(np.array([[100.0, 30.0 + d, 4.0, 4.0, 0.0]]), np.zeros(1, np.int8), ["h0"])

    def clear(fb):
        return float(fb.clearance(np.array([100.0]), np.array([30.0]))[0, 0])
    box = box_at(20.0 - (clear(box_at(20.0)) - 2.0))
    assert clear(box) == pytest.approx(2.0)
    frames = [box] * 21
    pl.route.best = _composite()
    real, why = pl._route_real(0, 0.0, 0.0, 0.0, 0, frames, 20, 0, 0)
    assert why is None and real["soft"] == 0.0 and real["portals"] == 0
    pl.route.best = _composite(jump_at=5)
    real, why = pl._route_real(0, 0.0, 0.0, 0.0, 0, frames, 20, 0, 0)
    assert why is None and real["portals"] == 1
    assert real["soft"] == pytest.approx(1.0 - 2.0 / (0.5 + 0.25 * 15)) and real["soft_frame"] == 20


def test_real_path_needs_every_state_at_its_wave_frame_when_tiles_changed():
    _, _, pl = synthetic()
    pl.route = RouteBook()
    pl.route.best = _composite(w=list(range(10)) + list(range(40, 51)))   # a later segment from wave frames 40..
    pl.tiles_changed = False
    real, why = pl._route_real(0, 0.0, 0.0, 0.0, 0, [None] * 21, 20, 0, 0)
    assert why is None
    pl.tiles_changed = True
    real, why = pl._route_real(0, 0.0, 0.0, 0.0, 0, [None] * 21, 20, 0, 0)
    assert real is None and why.startswith("route state +10 is -30 frames away")


def test_ahead_start_never_crosses_a_portal():
    _, _, pl = synthetic()
    pl.route, pl.tiles_changed = RouteBook(), False
    row = np.zeros(1, ROW)
    row["x"], row["y"] = 100.0, 30.0
    acts = np.full(8, 6)
    pl.route.best = _composite()
    assert pl._route_ahead(row, acts, 4, 16.667) == 12
    pl.route.best = _composite(jump_at=9)
    assert pl._route_ahead(row, acts, 4, 16.667) is None             # the portal lies in frames 5..12
    assert pl._route_ahead(row, acts, 9, 16.667) == 17               # from the portal's own state on: fine
    pl.route.best = _composite(w=np.arange(21) + 30)                  # recorded at other wave frames
    assert pl._route_ahead(row, acts, 4, 16.667) == 12
    pl.tiles_changed = True
    assert pl._route_ahead(row, acts, 4, 16.667) is None


def test_planner_finished_reports_the_route_book():
    _, _, pl = synthetic()
    pl.route, pl.moves = RouteBook(), None
    pl.memory = types.SimpleNamespace(finished=lambda agent: [])
    for f in range(5):
        pl.route.observe(0, f, 10.0 * f, 30.0, 10.0, 0.0, keys=0, held=0)
        pl.route.played(0, f, 6)
    r = P.Planner.finished(pl, 0, 4)
    assert r["best_route"] and r["route_book"]["records"] == 1 and r["route_book"]["finish"] == 4
    assert r["route_book"]["best_real"] == 4 and r["route_book"]["segments"] == [[0, 0, 4]]
    assert set(r["route_book"]) == {"records", "portals", "finish", "best_real", "segments", "duplicates",
                                   "build_ms"}
    assert P.Planner.finished(pl, 0, 9)["route_book"] is None         # nothing recorded for the agent
