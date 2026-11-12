"""Native endpoint convention and private imagined masks; synthetic fixed outputs, no learned checkpoint."""
import types

import numpy as np
import pytest
import torch

from game.level import LockContext, LOCK_CONTEXT_PROTOCOL
from plan.geometry import Terrain
from plan import planner as P
from plan.progress import ProgressField
from wm.buffer import ROW
from wm import features as F
from wm import rollout as R
from wm.model import WorldModel
from wm.protocol import KEY_TILES_PROTOCOL


def context(keys, groups=None):
    words = np.zeros((7, 12), np.uint32)
    words[1, 4] = 0xc00163
    meta = {"ox": 0, "oy": 0, "keys": keys}
    if groups is not None:
        meta["lock_context"] = {"protocol": LOCK_CONTEXT_PROTOCOL, "source": "python-go-payload",
                                "keys": keys, "groups": groups}
    return words, meta


def synthetic(step=1, key_x=90):
    words, meta = context([[key_x, 30]], [[16]])
    geom = F.LevelGeom(words, meta)
    # Zero weights and constant output biases are a deterministic fixture, not trained physics.
    model = WorldModel(n_in=F.layout(KEY_TILES_PROTOCOL).n_in(step), members=2, layers=0, step=step,
                       res_scale=np.ones(4))
    model.info = {"feature_protocol": KEY_TILES_PROTOCOL}
    with torch.no_grad():
        for i, member in enumerate(model.members):
            member[0].weight.zero_()
            member[0].bias.zero_()
            member[0].bias[0] = 110 if i == 0 else 0
            member[0].bias[8:8 + len(F.BIN_NAMES)] = -100
    pl = P.Planner.__new__(P.Planner)
    pl.f, pl.key_pickup, pl.collide_tol, pl.geom = step, True, 1., geom
    pl.model, pl.member, pl.complete_max = model, 0, step * 2
    pl.lv = types.SimpleNamespace(plats=[], keys=meta["keys"], ox=0, oy=0,
                                  lock_groups=[[16]], opened_cells=geom.lock_context.opened_cells)
    pl.terrain, pl.guard = Terrain(words, 0, 0), Terrain(words, 0, 0, slopes=False)
    pl._guard_keys, pl._support_keys = {}, {}
    field = ProgressField.__new__(ProgressField)
    field.key_order = [[0]]
    field.times = lambda x, y, leg, supported=True, vy=None: np.full(np.shape(x), 100.) - 50 * np.asarray(leg)
    pl.field, pl._dead_t = field, 1000
    pl.goal_boxes = np.empty((0, 5))
    pl.bounds, pl.floor = (-100, 1000, -100, 1000), None
    pl.trials, pl.trial_tube = P.RouteTrials(), None
    return model, geom, pl


def state():
    s = np.zeros(2, ROW)
    s["x"], s["y"] = 0, [30, 60]
    return s


def test_strict_endpoint_radius_and_no_continuous_segment_rule():
    words, meta = context([[0, 0]])
    c = LockContext(words, meta)
    masks, times = c.pickup_path(np.array([[40., 22.], [40., 21.999], [-30., 30.]]), np.zeros((3, 2)))
    assert masks[:, -1].tolist() == [0, 1, 0]
    assert times[:, 0].tolist() == [-1, 1, -1]
    # A continuous segment crossing the circle does not imply any native endpoint was inside it.
    assert masks[2, -1] == 0


def test_coarse_interpolation_integer_samples_and_out_of_order_keys():
    words, meta = context([[30, 0], [70, 0]], [[], []])
    c = LockContext(words, meta)
    masks, times = c.pickup_path(np.array([[100., 0.]]), np.array([[0., 0.]]), step=4)
    assert masks.tolist() == [[0, 2, 3, 3, 3]] and times.tolist() == [[2, 1]]
    field = ProgressField.__new__(ProgressField)
    field.key_order = [[0], [1]]
    assert field.leg_of(masks[0]).tolist() == [0, 0, 2, 2, 2]
    # Near-circle chord with endpoints (including interpolated integer samples) all outside: no pickup.
    c2 = LockContext(*context([[0, 0]]))
    masks, _ = c2.pickup_path(np.array([[-120., 120.]]), np.array([[21., 21.]]), step=3)
    assert masks[0, -1] == 0


def test_unknown_multi_key_mapping_is_not_invented_by_pickup():
    c = LockContext(*context([[0, 0], [100, 0]]))
    with pytest.raises(ValueError, match="multi-key"):
        c.pickup_path(np.array([[40., 0.]]), np.zeros((1, 2)))


def test_post_update_pickup_does_not_gate_on_new_terminal_status():
    c = LockContext(*context([[90, 30]]))
    before, after = np.zeros(2, ROW), np.zeros(2, ROW)
    after["x"], after["y"], after["status"] = 90, 30, [1, 2]
    c.pickup_after_step(before, after)
    assert after["keys"].tolist() == [1, 1] and not before["keys"].any()


@pytest.mark.parametrize("step,key_x", [(1, 90), (4, 30)])
def test_guard_closed_incoming_then_private_next_step_open_and_features(step, key_x, monkeypatch):
    model, geom, pl = synthetic(step, key_x)
    s0 = state()
    geom.patches(s0["x"], s0["y"], s0["keys"])
    snapshot = [a.copy() for a in (geom.words, geom.pad, geom._table, pl.guard.shape, pl.terrain.shape)]
    seen = []
    original = P.features_into
    def observe(X, groups, s, a, dt, g, worlds):
        original(X, groups, s, a, dt, g, worlds)
        seen.append(s["keys"].copy())
    monkeypatch.setattr(P, "features_into", observe)
    out = P.imagine(model, slice(None), geom, s0, np.zeros((2, 2 * step), int), 16.667,
                    [R.World()] * (2 * step), keep_rows=True, **pl.imagine_options())
    assert 73 < out["x"][0, 0, 1] < 80              # pickup cannot open an incoming step's locked wall
    assert out["x"][0, 0, 2] > 100                 # next model step sees the newly owned mask
    assert out["keys"][:, :, -1].tolist() == [[1, 0], [0, 0]]
    assert seen[0].tolist() == [0, 0, 0, 0] and seen[1].tolist() == [1, 0, 0, 0]
    assert out["rows"]["keys"].tobytes() == out["keys"].tobytes()
    assert out["key_pickup"]["maximum_opening_delay_frames"] == step - 1
    if step == 4:
        _, first = geom.lock_context.pickup_path(out["x"][0, :1, :2], out["y"][0, :1, :2], step=4)
        assert first[0, 0] == 1                     # interpolated pickup, but guard stays closed through frame4
    for before, after in zip(snapshot, (geom.words, geom.pad, geom._table, pl.guard.shape, pl.terrain.shape)):
        assert before.tobytes() == after.tobytes()
    assert not s0["keys"].any()


def test_initial_owned_guard_and_support_lookup_use_row_masks():
    _, _, pl = synthetic()
    x, y = np.array([74., 74.]), np.array([30., 30.])
    sup, ground, side = pl.support(x, y, keys=[0, 1])
    assert sup.tolist() == [True, False] and side.tolist() == [1, 0]
    _, sup, _, _, leg = pl.time_to_go(x, y, keys=[0, 1])
    assert sup.tolist() == [True, False] and leg.tolist() == [0, 1]
    assert pl.guard_terrain(0).shape[1, 4] != pl.guard_terrain(1).shape[1, 4]
    assert pl.support_terrain(1) is pl.support_terrain(1)


@pytest.mark.parametrize("step", [1, 4])
def test_raw_batched_rollout_opt_in_and_frozen_default(step):
    model, geom, _ = synthetic(step, 90 if step == 1 else 30)
    s = state()[:1]
    acts, dt, world = np.zeros((1, step * 2), int), np.full(step * 2, 16.667), [R.World()] * (step * 2)
    default = R.rollout_batched(model, geom, s, acts, dt, world, keep_rows=True)
    frozen = R.rollout_batched(model, geom, s, acts, dt, world, keep_rows=True, key_pickup=False)
    for key in default:
        if isinstance(default[key], np.ndarray):
            assert default[key].tobytes() == frozen[key].tobytes()
    assert "keys" not in default and not default["rows"]["keys"].any()
    enabled = R.rollout_batched(model, geom, s, acts, dt, world, keep_rows=True, key_pickup=True)
    if step == 1:                                   # endpoint110 is within22 of key90
        assert enabled["rows"]["keys"][0, 0, 1] == 1
    else:                                           # first interpolated endpoint27.5 is near key30
        assert enabled["rows"]["keys"][0, 0, 1] == 1
    assert not enabled["rows"]["keys"][1].any()   # fixed member1 predicts no movement


def test_mean_and_unbatched_consumers_keep_private_postupdate_keys():
    model, geom, _ = synthetic(1, 90)
    model.res_scale = F.RES_SCALE.copy()
    with torch.no_grad():
        model.members[0][0].bias[0] = 55
    s, acts, dt, world = state()[:1], np.zeros((1, 2), int), np.full(2, 16.667), [R.World()] * 2
    mean_features, states = R.rollout_mean(model, geom, s, acts, dt, world, key_pickup=True)
    # Ensemble mean dx55 lands within22 of key90 only on the second update.
    assert states[1]["keys"].tolist() == [0] and states[2]["keys"].tolist() == [1]
    out = R.rollout(model, geom, s, acts, dt, world, key_pickup=True)
    assert out["keys"][:, 0, -1].tolist() == [1, 0]


@pytest.mark.parametrize("step", [1, 4])
def test_completion_does_not_land_on_a_lock_opened_by_its_private_pickup(step):
    _, _, pl = synthetic(step, 90 if step == 1 else 30)
    result, _ = pl.complete(state()[:1], np.array([0]), np.array([0]), [R.World()] * 20,
                             [None] * 21, 16.667, 0, 0)
    _, _, rec, _ = result[0]
    assert rec["keys"] == 1 and rec["x"] > 100 and rec["on"] == "air"


def test_contacts_use_native_circle_all_keys_and_suppress_after_geometric_death():
    _, _, pl = synthetic()
    pl.geom = F.LevelGeom(*context([[30, 0], [70, 0]], [[], []]))
    pl.field.key_order = [[0], [1]]
    x, y = np.array([[[100., 75., 50., 25.]]]), np.zeros((1, 1, 4))
    _, legs, touches = pl._contacts(x, y, None, np.array([[-1]]), 0)
    assert legs.tolist() == [[2]] and [t[1][0, 0] for t in touches] == [2, 1]
    _, legs, _ = pl._contacts(x, y, None, np.array([[2]]), 0)
    assert legs.tolist() == [[0]]                    # key1 first, key0 at death excluded: route leg0 remains


def test_committed_support_uses_known_and_coarse_predicted_private_masks():
    from plan.commit import Committer
    _, _, pl = synthetic(4, 30)
    path = [[x, 30.] for x in [0., 18.5, 37., 55.5, 74., 100., 130., 160., 184.]]
    plan = types.SimpleNamespace(frame=0, path=path, actions=[0] * 8, first_move_frames=0,
                                 state={"keys": 0}, goal_touch=None, key_touches=[])
    c = Committer(replan=48)
    c.decided(0, plan, 0, planner=pl)
    assert c.cur[0].support[4] == 0                  # at the closed wall's contact, newly owned means no support
    assert not any(t.startswith("support:") for t in c.triggers(0, 4, 74., 30., keys=1, planner=pl))
    plan.state["keys"] = 1
    c.decided(0, plan, 0, planner=pl)
    assert c.cur[0].support[4] == 0


def test_best_route_producer_records_known_and_unknown_keys_without_changing_actions():
    route = P.BestRoute()
    for frame, keys in enumerate((None, 0, 1)):
        route.observe(0, frame, frame * 10., 30., 1., 0., keys=keys)
        route.played(0, frame, 6)
    assert route.finished(0, 2)
    assert route.best["keys_known"].tolist() == [False, True, True]
    assert route.best["keys"].tolist() == [0, 0, 1] and route.best["a"].tolist() == [6, 6, 6]


def test_planner_observation_forwards_native_keys_to_route():
    _, _, pl = synthetic()
    pl.route = P.BestRoute()
    pl.surprise = types.SimpleNamespace(observe=lambda *a: [])
    pl.memory = types.SimpleNamespace(update=lambda *a, **kw: [])
    P.Planner.observe(pl, 0, 0, 0., 30., keys=1, vx=0., vy=0.)
    pl.route.played(0, 0, 6)
    assert pl.route.finished(0, 0)
    assert pl.route.best["keys_known"].tolist() == [True] and pl.route.best["keys"].tolist() == [1]


def _recorded_route(pl, masks):
    pl.route = P.BestRoute()
    for f, (x, mask) in enumerate(zip((0., 74., 74.), masks)):
        pl.route.observe(0, f, x, 30., 0., 0., keys=mask)
        pl.route.played(0, f, 6)
    assert pl.route.finished(0, 100)
    return [None] * 3


def test_trusted_route_requires_matching_known_initial_mask_and_uses_endpoint_keys():
    _, _, pl = synthetic()
    frames = _recorded_route(pl, (0, 1, 1))
    real, why = pl._route_real(0, 0., 0., 0., 0, frames, 2, 0, 0)
    assert why is None and real["t_end"] == 50 and not real["supported"]
    rejected, why = pl._route_real(0, 0., 0., 0., 0, frames, 2, 1, 1)
    assert rejected is None and "key mask 0 differs from current 1" in why
    frames = _recorded_route(pl, (1, 1, 1))
    rejected, why = pl._route_real(0, 0., 0., 0., 0, frames, 2, 0, 0)
    assert rejected is None and "key mask 1 differs from current 0" in why


def test_legacy_unknown_route_keys_fall_back_only_on_key_maps():
    _, _, pl = synthetic()
    frames = _recorded_route(pl, (None, None, None))
    rejected, why = pl._route_real(0, 0., 0., 0., 0, frames, 2, 0, 0)
    assert rejected is None and "unknown on a key map" in why
    # Genuine legacy dictionaries have neither masks nor a knownness array.
    del pl.route.best["keys"], pl.route.best["keys_known"]
    rejected, why = pl._route_real(0, 0., 0., 0., 0, frames, 2, 0, 0)
    assert rejected is None and "unknown on a key map" in why
    pl.lv.keys = []
    real, why = pl._route_real(0, 0., 0., 0., 0, frames, 2, 0, 0)
    assert real is not None and why is None
