"""Required-key routes survive conservative hazard modes and reactive beam histories."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realpath import Planner
from plan.progress import ProgressField, leg_of_keys


def flat():
    w = np.zeros((10, 20), np.uint32)
    w[:2] = 1
    return w


def test_key_cannot_be_skipped_when_a_sweep_blocks_the_clear_route():
    # Exit is near the spawn, key is across a moving hazard's sweep. Timed mode is required.
    args = (flat(), 0., 0., (50., 50.), [(90., 50.)], [], [(200., 100., 10., 120., 0.)],
            [(330., 50.)], [], [])
    field = ProgressField(*args)
    assert field.found_route and field.mode[0] == "timed"
    assert field.key_order == [[0]] and field.n_legs == 2
    assert (16, 2) in field.route_cells
    assert field.leg_of([0, 1]).tolist() == [0, 1]
    route = Planner(*args).find(1)[0]
    assert route.cells.index((16, 2)) < len(route.cells) - 1
    assert field.times([50.], [50.])[0] > 100


def test_unreachable_key_is_explicit_and_kept_before_the_goal():
    w = flat()
    w[:, 10] = 1
    args = (w, 0., 0., (50., 50.), [(90., 50.)], [], [], [(330., 50.)])
    field = ProgressField(*args)
    assert not field.found_route and "no complete route" in field.summary()
    assert field.key_order == [[0]] and field.n_legs == 2
    assert field.leg_of([0, 1]).tolist() == [0, 1]
    assert np.isinf(field.times([50.], [50.])[0])
    assert not Planner(*args).find(1)             # never offer the nearby exit as a completed route


def test_all_keys_sharing_a_route_cell_are_required():
    field = ProgressField(flat(), 0., 0., (50., 50.), [(90., 50.)], [], [], [(330., 50.), (330., 50.)])
    assert field.key_order == [[0, 1]]
    assert field.leg_of([0, 1, 2, 3]).tolist() == [0, 0, 0, 1]
    assert [leg_of_keys(field.key_order, k) for k in range(4)] == [0, 0, 0, 1]


def test_reactive_beams_are_not_permanent_but_plain_lasers_still_are():
    from agent import Runner
    from game.level import Level, box_ids, hazard_boxes
    from plan.lasers import EyeTracker
    from mpc.future import FutureWorld
    from plan.planner import future_inputs
    from plan import safety as S
    params = {"lasers": [dict(id="laser_homing", kind=1, x=100., y=100., angle=0.,
                              params=[1., 0., 0., 999., 0, 18]),
                         dict(id="laser_plain", kind=0, x=200., y=100., angle=0., params=[])]}
    lv = Level(dict(name="test-required-keys", W=20, H=10, ox=0., oy=0., spawn=[50., 50.], goals=[[90., 50.]],
                    lethalTypes=[2000, 2000], lethalIds=["laser_homing", "laser_plain"], lzp=params), flat().tobytes())
    fw = FutureWorld()
    fw.new_level()
    fw.set_lasers(EyeTracker.for_level(lv))
    r = SimpleNamespace(level=lv, fw=fw, hz_start=None, moving=set(), sweep=set(), field_dirty=False)
    def hz(end):
        return [[150., 100., end-100., 0., 1., 1., 0., [[3, 100., 100., end, 100., 0]]],
                [250., 100., 100., 0., 1., 1., 0., [[3, 200., 100., 300., 100., 0]]]]
    for f, end in [(0, 190.), (1, 180.)]:
        boxes, kept = hazard_boxes(hz(end), lv.hz_keep)
        Runner.track_hazards(r, boxes, box_ids(kept, lv.hz_ids))
        fw.observe(f, hz(end), None, [], lv.hz_keep, [], 16.667,
                   lz=[[1, .1, 0., 0.], [1, .1, 0., 0.]], target=(150., 100.))
    assert set(r.hz_start) == {"laser_plain"}
    assert not r.moving and not r.sweep
    _, frames, _ = future_inputs(fw, 1, 2, 16.667)
    check = S.check(np.full((1, 1, 3), 150.), np.full((1, 1, 3), 100.), frames)
    assert check.veto[0] and check.veto_id[0] == "laser_homing"  # candidate safety still rejects contact


DATA = Path(__file__).resolve().parents[1] / "runs/online_check12/wm_data"


@pytest.mark.skipif(not (DATA / "levels/4-12.npz").exists(), reason="recorded online_check12 not present")
def test_recorded_4_12_rebuild_retains_the_key():
    from plan.measure_progress import level_from_data, planner_inputs
    from agent import Runner
    from game.level import hazard_boxes, box_ids
    from plan.lasers import EyeTracker
    lv = level_from_data(str(DATA), "4-12")
    r = SimpleNamespace(level=lv, fw=SimpleNamespace(lasers=EyeTracker.for_level(lv)),
                        hz_start=None, moving=set(), sweep=set(), field_dirty=False)
    world = []
    for wave in [1, 2]:
        with gzip.open(DATA / f"wave_{wave:05d}_world.json.gz", "rt") as f:
            world.extend(json.load(f))
    for rec in world:
        if "ids" in rec:
            lv.hz_ids = list(rec["ids"])
            lv.hz_level_ids = set(lv.hz_ids)
        boxes, kept = hazard_boxes(rec["hz"], lv.hz_keep)
        Runner.track_hazards(r, boxes, box_ids(kept, lv.hz_ids))
    assert not r.moving and not r.sweep
    inp = planner_inputs(lv, world)
    assert inp["n_moving"] == 0 and not inp["sweeps"]
    field = ProgressField(lv.words0, lv.ox, lv.oy, lv.spawn, lv.goals, inp["hazards"], inp["sweeps"],
                          lv.keys, lv.bandages, lv.portals, lift=inp["lift"], blocks=inp["blocks"],
                          doors=inp["doors"], outside=inp["outside"])
    assert field.found_route and field.key_order == [[0]] and field.n_legs == 2
    assert (32, 7) in field.route_cells
    assert field.times([lv.spawn[0]], [lv.spawn[1]])[0] > 400
