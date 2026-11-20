"""PPO perception parity, checkpoint migration, live-message integration and dependency isolation."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import realgame as R
from perception_cases import fixture, snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_current_ppo_observation_parity():
    expected = json.loads((Path(__file__).parent / 'data/perception_parity.json').read_text())
    expected.pop('source_hashes')
    assert snapshot(R) == expected


def test_enemy_laser_and_multiple_kill_regions():
    info, words = fixture()
    lv = R.Level(info, words.tobytes())
    boxes, kept = R.hazard_boxes(info['hz'], lv.hz_keep)
    assert kept == [0, 1, 3, 4]                       # harmless row not a hazard
    assert 'enemy1' in R.box_ids(kept, lv.hz_ids)
    assert boxes[-1][2] > 60                        # laser reaches along its segment
    duplicate = info['hz'][0].copy()
    duplicate[-1] = [duplicate[-1][0], [1, -10, -30, 8, 8, 0]]
    boxes, kept = R.hazard_boxes([duplicate], [False])
    assert len(boxes) == 2                          # measured regions override palette decoration
    assert len(set(R.box_ids(kept, ['piece']))) == 2
    off = info['hz'][-1].copy()
    off[-1] = []
    assert R.hazard_boxes([off])[0] == []
    assert len(R.hazard_boxes([info['hz'][0][:7]], [True])[0]) == 1  # older captures still work


def test_live_tile_keys_boundaries_and_platform_velocity():
    info, words = fixture()
    lv = R.Level(info, words.tobytes())
    before = lv.open_dirs(0).copy()
    lv.apply_tiles([[2 * lv.W + 4, 0]])
    assert np.any(before[:, 2, 4]) and not lv.open_dirs(0)[:, 2, 4].any()
    assert lv.opened_cells(1) == [(3, 8), (4, 8)] and not lv.opened_cells(0)
    assert lv.out_of_level(-131, 0) and not lv.out_of_level(0, 0)
    lv.set_kill_box([-120., 120., -100., 100.])
    assert lv.grid_with_hazards([])[1, 0] == R.C_HAZARD
    lv.set_platforms([[-68., -43., 0], [90., -8., 1]])
    assert lv.plat_vel[0] == (120., 120.)


def test_legacy_checkpoint_and_shape_inputs_roundtrip(tmp_path):
    old = R.Policy()
    path = tmp_path / 'old.pt'
    torch.save({'policy': old.state_dict(), 'obs': R.OBS}, path)
    same = R.Policy()
    assert not R.load_policy(same, str(path))
    for key, value in old.state_dict().items():
        assert torch.equal(same.state_dict()[key], value)
    wide = R.Policy(R.obs_size(True))
    assert R.load_policy(wide, str(path))
    for key, value in old.state_dict().items():
        actual = wide.state_dict()[key]
        assert torch.equal(actual[:, :value.shape[1]] if actual.shape != value.shape else actual, value)
    torch.save({'policy': wide.state_dict(), 'obs': R.obs_size(True), 'shape_inputs': True}, path)
    restored = R.Policy(R.obs_size(True))
    assert not R.load_policy(restored, str(path))
    with pytest.raises(ValueError, match='shape-inputs'):
        R.load_policy(R.Policy(), str(path))


@pytest.mark.parametrize('shape_inputs', [False, True])
def test_game_messages_reach_ppo_and_train_without_other_controllers(tmp_path, shape_inputs):
    torch.set_num_threads(2)
    args = SimpleNamespace(name=str(tmp_path), agents=2, load=None, shape_inputs=shape_inputs,
                           paths=1, wave_seconds=20., test=False, batch=999999, debug_spawn=False)
    trainer = R.Trainer(args)
    posted = []
    trainer.script = SimpleNamespace(post=posted.append)
    info, words = fixture()
    trainer.on_message({'type': 'send', 'payload': dict(info, ev='info')}, words.tobytes())
    assert posted[-1]['type'] == 'go' and posted[-1]['payload']['locks']
    for frame in range(3):
        s = [[-110. + frame, -90., 60., 0., 0, 0, 0, keys] for keys in (0, 1)]
        msg = dict(ev='frame', wave=1, frame=frame, s=s, hz=info['hz'],
                   sh=[[1, -90. + frame, -70., 4.]], pf=[[-70. + frame, -45., 0]],
                   kb=[-125., 125., -110., 110.])
        trainer.on_message({'type': 'send', 'payload': msg}, None)
        assert posted[-1]['type'] == 'act' and len(posted[-1]['payload']['a']) == 2
    assert all(len(t['o']) == 3 for t in trainer.traj)
    assert all(len(o) == R.obs_size(shape_inputs) for t in trainer.traj for o in t['o'])
    assert len(trainer.boxes) == 5                    # four hazard regions plus one flying shot
    trainer.on_message({'type': 'send', 'payload': dict(ev='wave_end', wave=1, frame=3,
                        status=[1, 1], finish=[-1, -1])}, None)
    assert trainer.ppo.steps == 6
    trainer.ppo.epochs, trainer.ppo.mb = 1, 8
    before = next(trainer.policy.parameters()).detach().clone()
    trainer.ppo.update()
    assert not torch.equal(before, next(trainer.policy.parameters()))
    trainer.save()
    saved = torch.load(tmp_path / 'model.pt', map_location='cpu')
    assert saved['obs'] == R.obs_size(shape_inputs)
    assert saved.get('shape_inputs', False) is shape_inputs


def test_no_model_planner_or_online_dependency_even_when_importing():
    forbidden = {'wm', 'mpc', 'plan', 'agent'}
    for name in forbidden:
        assert not (ROOT / name).exists() and not (ROOT / (name + '.py')).exists()
    for path in [ROOT / 'realgame.py', ROOT / 'realtest.py', ROOT / 'levels_smb.py', ROOT / 'realpath.py', ROOT / 'monitor.py', *ROOT.glob('game/*.py')]:
        for n in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(n, ast.Import):
                assert not any(a.name.split('.')[0] in forbidden for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                assert n.module.split('.')[0] not in forbidden
    script = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'wm', 'mpc', 'plan', 'agent'}:
            raise RuntimeError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
import realgame, game.level, monitor
assert not any(k.split('.')[0] in {'wm', 'mpc', 'plan', 'agent'} for k in sys.modules)
'''
    subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True, capture_output=True, text=True)


def test_native_observation_producers_and_frame_wire_format():
    result = subprocess.run(['node', str(Path(__file__).with_name('native_perception.cjs'))],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
