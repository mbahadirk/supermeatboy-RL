"""Deterministic observed-game cases, usable against either branch in a separate Python process."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def fixture():
    words = np.zeros((12, 14), np.uint32)
    words[0] = 1
    words[1:8, 8] = 1
    words[2, 4], words[2, 5] = 1 | (1 << 18), 1 | (2 << 18) | (1 << 12)
    words[3, 8], words[4, 8] = 0xc00163, 0xc00163  # one lock group
    words[1, 5], words[1, 6] = 0x800062, 0x2001062
    words[2, 6] = 0x101                         # crumble
    hz = [[-30, -30, 20, 20, 1, 1, 0, [[1, -30, -30, 10, 10, 0]]],
          [55, 45, 16, 12, 1, 1, .6, [[2, 55, 45, 8, 6, .6]]],
          [90, 30, 20, 20, 1, 1, 0, []],       # harmless phase/decoration
          [20, 30, 20, 20, 1, 1, 0, [[1, 20, 30, 8, 8, 0]]],  # enemy
          [0, 0, 0, 0, 1, 1, 0, [[3, -40, 10, 80, 60, 0]]]]   # active laser
    info = dict(name='perception-fixture', W=14, H=12, ox=-140., oy=-120., spawn=[-110., -90.],
                goals=[[110., 90.]], objs=[[1, -60., -70., 0x40], [2, 70., -70., 8]],
                lethalTypes=[3, 4, 5, 1000, 2000], lethalIds=['saw', 'needle', 'decor', 'enemy1', 'laser1'],
                hz=hz, lethal=5, agents=2, kb=[-130., 130., -110., 110.],
                pf=[[-70., -45., 0], [90., -10., 1]],
                fx={'fans': [[-100., -100., 0., 1., 30., 180., 900.]],
                    'orbs': [[50., 50., 15., 25., 100., 150.]]},
                portals=[[[-100., 50.], [90., 70.]]])
    return info, words


def digest(a):
    return hashlib.sha256(np.asarray(a, dtype='<f4').tobytes()).hexdigest()


def snapshot(rg):
    info, words = fixture()
    lv = rg.Level(info, words.astype('<u4').tobytes())
    boxes, kept = rg.hazard_boxes(info['hz'], lv.hz_keep)
    pos = np.array([b[:2] for b in boxes], np.float32)
    vel = np.tile([30., -60.], (len(pos), 1)).astype(np.float32)
    results = {}
    for name, x, y, keys in [('spawn', -110., -90., 0), ('shapes', -50., -70., 0),
                             ('key_holder', 10., -50., 1), ('hazards', 10., 30., 0),
                             ('outside', -190., 180., 0), ('changed', -50., -70., 0)]:
        if name == 'changed':
            lv.apply_tiles([[2 * lv.W + 4, 1 | (1 << 18) | (1 << 12)], [2 * lv.W + 6, 0]])
            lv.set_platforms([[-68., -43., 0], [90., -8., 1]])
            lv.set_kill_box([-120., 120., -100., 100.])
        grid = lv.grid_with_hazards(boxes)
        hazard = rg.moving_features(x, y, pos, vel, np.ones(len(pos), bool))
        state = [x, y, 80., -40., 2, 9, 0, keys]
        obs = [rg.observe(lv, grid, state, None, 0., 0., hazard, lv.opened_cells(keys), (.3, .4), 5,
                          shape_inputs=enabled) for enabled in (False, True)]
        results[name] = {'obs': [digest(o) for o in obs], 'sizes': [len(o) for o in obs],
                         'grid': digest(grid), 'surfaces': digest(lv.surfaces())}
    return {'cases': results, 'boxes': [list(b) for b in boxes],
            'ids': rg.box_ids(kept, lv.hz_ids), 'lock_groups': lv.lock_groups}


if __name__ == '__main__':
    # Explicit source root keeps the baseline independent of the worktree under test.
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    import realgame
    result = snapshot(realgame)
    result['source_hashes'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('realgame.py', 'game/level.py', 'realpath.py')}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
