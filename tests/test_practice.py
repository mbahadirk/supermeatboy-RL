"""Practice on held-out levels (agent.py --online): after a held-out level's first life the move and death memory
learn for the run, but their entries are transient (MoveMemory / DeathMemory.transient): never saved, so a later
run's first life on the level stays clean."""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import planner as P                                              # noqa: E402
from plan import safety as S                                               # noqa: E402


def filled_moves(path) -> P.MoveMemory:
    m = P.MoveMemory(str(path))
    for name in ("1-1", "1-4"):
        m.set_level(name)
        m.decided(0, 10, (3, 4), "run_r", 1, adopted=True)
        m.decided(0, 20, (5, 4), "jump_r", 2, adopted=True)
        m.died(0, 30, x=50.0, y=40.0)
    return m


def test_move_memory_keeps_a_transient_level_in_memory_and_never_saves_it(tmp_path):
    base = tmp_path / "moves.json"
    filled_moves(base).save()
    before = base.read_bytes()
    m = P.MoveMemory.load(str(base))
    m.save()                                         # a plain round trip writes the same file
    assert base.read_bytes() == before
    m.transient = {"2-6"}
    m.set_level("2-6")
    m.decided(0, 12, (7, 2), "run_r", 1, adopted=True)
    ev = m.died(0, 40, x=70.0, y=20.0)
    assert ev["charged"] and m.stats((7, 2), "run_r") == (1, 1)
    assert m.dirty
    m.save()
    assert base.read_bytes() == before               # the other levels' entries only, byte for byte
    assert m.stats((7, 2), "run_r") == (1, 1)        # kept in memory for the run
    again = P.MoveMemory.load(str(base))
    assert "2-6" not in again.levels and set(again.levels) == {"1-1", "1-4"}
    assert again.levels == P.MoveMemory.load(str(base)).levels


def test_the_first_life_message_stays_and_practice_charges_normally(tmp_path):
    m = P.MoveMemory(str(tmp_path / "moves.json"))
    m.transient = {"2-6"}
    m.set_level("2-6")
    m.learn = False                                  # the first life
    m.decided(0, 12, (7, 2), "run_r", 1, adopted=True)
    ev = m.died(0, 40)
    assert not ev["charged"] and ev["reason"] == "held-out level: nothing charged"
    m.learn = True                                   # practice
    m.decided(0, 12, (7, 2), "run_r", 1, adopted=True)
    ev = m.died(0, 40)
    assert ev["charged"] and "reason" not in ev and ev["deaths"] == 1


def test_death_memory_keeps_a_transient_level_in_memory_and_never_saves_it(tmp_path):
    base = tmp_path / "deaths.json"
    S.DeathMemory([("1-1", 10, 1.0, 2.0), ("1-4", 20, 3.5, 4.0)]).save(str(base))
    before = base.read_bytes()
    d = S.DeathMemory.load(str(base))
    d.save(str(base))
    assert base.read_bytes() == before               # a plain round trip writes the same file
    d.transient = {"2-6"}
    d.add("2-6", 30, 5.0, 6.0)
    d.add("1-4", 40, 7.0, 8.0)
    assert len(d) == 4 and len(d.boxes("2-6", 30)) == 1          # kept in memory: the planner avoids it this run
    d.save(str(base))
    saved = json.loads(base.read_text(encoding="utf-8"))["deaths"]
    assert [e["level"] for e in saved] == ["1-1", "1-4", "1-4"]
    again = S.DeathMemory.load(str(base))
    assert len(again.boxes("2-6", 30)) == 0 and len(again) == 3


class FakeProc:
    def __init__(self):
        self.calls = []

    def call(self, name, *a):
        self.calls.append((name, *a))


def bare_runner(online=True, evaluation=None, level="2-6", held=("2-6",)):
    import agent
    r = agent.Runner.__new__(agent.Runner)
    r.args = types.SimpleNamespace(online=online)
    r.evaluation = evaluation
    r.held_out = set(held)
    r.level_waves = {}
    r.practising = set()
    r.level = types.SimpleNamespace(name=level)
    r.protected = evaluation is not None or level in r.held_out
    r.moves = P.MoveMemory()
    r.moves.learn = r.memory_on()                    # as build_field sets it
    r.proc, r.proc_ready = FakeProc(), True
    r.events, r.event = [], lambda kind, frame=None, **kw: r.events.append((kind, kw))
    return r


def wave_end(r):
    """on_wave_end's step: one more full wave of the level, then practice when it is now on."""
    name = r.level.name
    r.level_waves[name] = r.level_waves.get(name, 0) + 1
    if r.practice_on(name):
        r.start_practice(name)


def test_a_held_out_level_practises_from_its_second_wave_with_learning_between_waves():
    r = bare_runner()
    assert not r.practice_on("2-6") and not r.memory_on() and not r.moves.learn     # the first life: clean
    wave_end(r)
    assert r.practice_on("2-6") and r.memory_on()
    assert r.moves.learn and ("moves_learn", True) in r.proc.calls                   # both processes, between waves
    assert r.events == [("practice", {"name": "2-6", "waves": 1})]
    r.moves.learn = r.memory_on()                    # a field rebuild between waves keeps it on
    assert r.moves.learn
    wave_end(r)
    assert len(r.events) == 1                        # announced once per level and run


def test_no_practice_in_a_first_life_evaluation_or_without_learning_between_waves():
    for r in (bare_runner(evaluation=object()), bare_runner(online=False)):
        for _ in range(3):
            wave_end(r)
            assert not r.practice_on("2-6") and not r.memory_on() and not r.moves.learn
        assert r.events == [] and r.proc.calls == []


def test_training_levels_learn_as_before():
    for online in (True, False):
        r = bare_runner(online=online, level="1-4")
        assert not r.practice_on("1-4") and r.memory_on() and r.moves.learn
        wave_end(r)
        assert not r.practice_on("1-4") and r.memory_on() and r.events == []
