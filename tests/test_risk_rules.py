"""The risk rules of 2026-11-13: the soft-risk margin stops growing at MARGIN_HORIZON, a death costs at least
DEATH_MIN_FRAMES (not a dead end's 500), and remembered deaths that repeat at one place at other wave frames are
on at every frame, until a real agent is seen alive there at another time (3-2, 2026-11-16)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import safety as S                                               # noqa: E402
from plan import planner as P                                              # noqa: E402


def test_margin_is_held_from_the_margin_horizon_on():
    m = S.margin(np.array([1, 8, S.MARGIN_HORIZON, 32, 48]))
    assert m[0] < m[1] < m[2] and m[2] == m[3] == m[4]
    assert m[2] == dict(S.ERR90)[16]


def test_a_death_costs_less_than_a_dead_end_by_default():
    assert P.DEATH_MIN_FRAMES < P.DEAD_END_FRAMES


def test_one_remembered_death_is_a_fact_about_its_own_wave_frame():
    dm = S.DeathMemory([("L", 100, 50.0, 60.0)])
    assert len(dm.boxes("L", 100)) == 1 and len(dm.boxes("L", 100 + S.WINDOW)) == 1
    assert len(dm.boxes("L", 100 + S.WINDOW + 1)) == 0 and len(dm.boxes("other", 100)) == 0


def test_deaths_repeating_at_one_place_at_other_frames_are_always_on():
    # the 2-1 pattern: the same point, 48 frames later each wave; and one unrelated death elsewhere
    dm = S.DeathMemory([("L", 329, -111.8, -193.3), ("L", 381, -111.8, -193.3), ("L", 429, -111.8, -193.3),
                        ("L", 128, -397.0, -194.0)])
    for frame in (0, 329, 700, 5000):
        b = dm.boxes("L", frame)
        here = b[np.hypot(b[:, 0] + 111.8, b[:, 1] + 193.3) < 1]
        assert len(here) == 1 and here[0, 2] == S.DEATH_HALF          # one box for the place, at every frame
    assert len(dm.boxes("L", 128)) == 2 and len(dm.boxes("L", 300)) == 1   # the single death keeps its window
    # two deaths at one place within the window of each other are still one moment, not a place
    dm2 = S.DeathMemory([("L", 100, 0.0, 0.0), ("L", 100 + S.WINDOW, 0.0, 0.0)])
    assert len(dm2.boxes("L", 1000)) == 0 and len(dm2.boxes("L", 100)) == 2
    # added later: the cache is rebuilt
    dm2.add("L", 400, 3.0, 0.0)
    assert len(dm2.boxes("L", 1000)) == 1


D32 = (("3-2", 772, 25.19, 216.01), ("3-2", 680, 18.68, 217.95))   # runs/online_check03: the 3-2 take-off


@pytest.mark.parametrize("seen_first", [False, True])
def test_a_place_where_an_agent_survived_is_not_always_deadly(seen_first):
    """3-2: two deaths at one place 92 frames apart, but an agent stood there alive at frame 300 (wave 1 of
    online_check03 was there alive before either death: seen_first, replayed from the trail by add())."""
    dm = S.DeathMemory()
    if seen_first:
        assert not dm.observe_alive("3-2", 300, [30.0], [220.0])        # no remembered death there yet
    for e in D32:
        dm.add(*e)
    if not seen_first:
        assert dm.observe_alive("3-2", 300, [30.0], [220.0])            # this visit refutes it (cold cache)
        static = S.DeathMemory(D32)
        assert len(static.boxes("3-2", 300)) == 1                       # static until a visit
        assert static.observe_alive("3-2", 300, [30.0], [220.0])        # (warm cache)
        assert len(static.boxes("3-2", 300)) == 0
    assert len(dm.boxes("3-2", 300)) == 0
    b = dm.boxes("3-2", 700)
    assert len(b) == 1 and b[0, :2].tolist() == [18.68, 217.95]         # the frame-680 death's own window
    assert len(dm.boxes("3-2", 772)) == 1 and dm.boxes("3-2", 772)[0, 0] == 25.19


def test_visits_inside_a_deaths_window_or_out_of_contact_do_not_refute():
    dm = S.DeathMemory(D32)
    assert not dm.observe_alive("3-2", 680 - S.WINDOW, [25.0], [216.0])    # within WINDOW of the 680 death
    assert not dm.observe_alive("3-2", 760, [25.0], [216.0])               # within WINDOW of the 772 death
    # out of contact: the box's half size plus BODY along x (the safety check's rule, kill_clearance < 0)
    assert not dm.observe_alive("3-2", 300, [25.19 + S.DEATH_HALF + S.BODY + 0.01], [216.01])
    assert not dm.observe_alive("3-2", 300, [25.19], [216.01 - S.DEATH_HALF - S.BODY - 0.01])
    assert not dm.observe_alive("4-4", 300, [25.0], [216.0])               # another level
    for f in (0, 300, 700, 5000):
        assert len(dm.boxes("3-2", f)) == 1                             # still on at every frame
    assert dm.observe_alive("3-2", 300, [25.19 + S.DEATH_HALF + S.BODY - 0.01], [216.01])   # just in contact


def test_the_2_1_place_where_nobody_lived_stays_always_on():
    dm = S.DeathMemory([("2-1", 100, -111.8, -193.3), ("2-1", 148, -111.8, -193.3)])
    dm.observe_alive("2-1", 300, [-300.0], [-193.3])                     # alive elsewhere
    for f in (0, 100, 124, 300, 5000):
        assert len(dm.boxes("2-1", f)) == 1


def test_the_2_1_agent_alive_just_before_its_next_death_does_not_refute_for_long():
    """As on 2-1 live: the agent stands in the box for the frames before each death. Before the third death
    those frames are more than WINDOW from the first two, so the place is refuted for a moment (an event and a
    save in agent.py); the death that follows puts them inside its window and the place is static again."""
    p = (-111.8, -193.3)
    dm = S.DeathMemory()
    for f in range(300, 329):
        dm.observe_alive("2-1", f, [p[0] + 1.0], [p[1]])
    dm.add("2-1", 329, *p)
    for f in range(352, 381):
        assert not dm.observe_alive("2-1", f, [p[0] + 1.0], [p[1]])    # one death: no static place yet
    dm.add("2-1", 381, *p)
    assert all(len(dm.boxes("2-1", f)) == 1 for f in (0, 200, 600))
    flips = [dm.observe_alive("2-1", f, [p[0] + 1.0], [p[1]]) for f in (426, 427, 428)]
    assert flips == [True, False, False] and len(dm.boxes("2-1", 600)) == 0
    dm.add("2-1", 429, *p)
    assert all(len(dm.boxes("2-1", f)) == 1 for f in (0, 200, 600, 5000))


def test_the_runner_saves_and_logs_a_refutation_only_where_memory_is_on(tmp_path):
    import agent
    for transient, memory, saves in ((False, True, 1), (True, True, 0), (False, False, 0)):
        r = object.__new__(agent.Runner)
        r.deaths = S.DeathMemory(D32)
        if transient:
            r.deaths.transient = {"3-2"}
        r.memory_on = lambda m=memory: m
        saved, events = [], []
        r.save_deaths = lambda: saved.append(1) or True
        r.event = lambda kind, frame, **kw: events.append((kind, frame))
        r.observe_alive("3-2", 250, [(500.0, 0.0)])                     # nowhere near: nothing
        assert not events and not saved
        r.observe_alive("3-2", 300, [(500.0, 0.0), (30.0, 220.0)])
        assert events == [("death place refuted", 300)] and len(saved) == saves
        r.observe_alive("3-2", 301, [(30.0, 220.0)])                    # already refuted: no second event
        assert len(events) == 1


def test_visits_are_saved_and_old_files_still_load(tmp_path):
    p = str(tmp_path / "deaths.json")
    dm = S.DeathMemory(D32 + (("4-4", 41, -225.4, -114.0),))
    dm.observe_alive("3-2", 300, [30.0], [220.0])
    dm.observe_alive("3-2", 301, [30.0], [220.0])
    dm.observe_alive("3-2", 305, [30.0], [220.0])
    assert dm.dirty
    dm.save(p)
    assert not dm.dirty
    raw = json.load(open(p, encoding="utf-8"))
    assert raw["deaths"][0]["alive"] == [[300, 301], [305, 305]] and "alive" not in raw["deaths"][2]
    again = S.DeathMemory.load(p)
    assert again.entries == dm.entries and again.visits == dm.visits
    assert len(again.boxes("3-2", 300)) == 0 and len(again.boxes("3-2", 700)) == 1
    # a file from before the visits: the place is static as before
    old = {"window": S.WINDOW, "half": S.DEATH_HALF,
           "deaths": [{"level": l, "frame": f, "x": x, "y": y} for l, f, x, y in D32]}
    (tmp_path / "deaths.json").write_text(json.dumps(old), encoding="utf-8")
    old_dm = S.DeathMemory.load(p)
    assert len(old_dm) == 2 and old_dm.visits == [set(), set()] and len(old_dm.boxes("3-2", 300)) == 1


def test_the_trail_is_bounded(monkeypatch):
    monkeypatch.setattr(S, "TRAIL_KEEP", 8)
    monkeypatch.setattr(S, "TRAIL_CHUNK", 4)
    dm = S.DeathMemory()
    for f in range(20):
        dm.observe_alive("L", f, [float(f)], [0.0])
    t = dm.trail("L")
    assert len(t) <= 8 + 4 and t[-1, 0] == 19 and t[0, 0] > 0
