"""Native shape -> PPO map/reward route/monitor regressions (no live game access)."""
from types import SimpleNamespace as NS
import math
import numpy as np
import realgame as R
from game.hazards import (hazard_shapes,hazard_reaches,SHAPE_BOX,SHAPE_CIRCLE,
                          SHAPE_GAMEBOX,SHAPE_SEGMENT)
from realpath import Planner


def level_info():
    words=np.zeros((9,9),dtype='<u4')
    hz=[[0.,0.,60.,60.,1.,1.,0.,[[1,0.,0.,30.,30.,0.]]]]
    info=dict(name='native-saw',W=9,H=9,ox=-60.,oy=-60.,spawn=[-50.,-50.],goals=[[110.,110.]],
              objs=[],lethalTypes=[1],lethalIds=['saw'],lethal=1,agents=2,hz=hz)
    return info,words


def test_saw_corner_is_open_in_ppo_map_sealing_and_reward_routes():
    info,words=level_info();lv=R.Level(info,words.tobytes())
    boxes,kept,kinds=hazard_shapes(info['hz'],lv.hz_keep)
    assert kept==[0] and kinds==[SHAPE_CIRCLE] and boxes[0][2]==30.
    corner=lv.cell(30.,30.)[::-1];axis=lv.cell(30.,10.)[::-1]
    assert not lv.hazard_mask(boxes,kinds)[corner]
    assert lv.hazard_mask(boxes,kinds)[axis]
    assert lv.grid_with_hazards(boxes,kinds)[corner]==R.C_EMPTY
    assert lv.grid_with_hazards(boxes,kinds)[axis]==R.C_HAZARD
    assert lv.grid_with_hazards(boxes)[corner]==R.C_HAZARD  # old covering-square representation
    lv.set_sealed(boxes,[lv.spawn,*lv.goals],kinds)
    assert not lv.sealed_haz[corner] and not lv.sealed_wall[corner]
    route=Planner(words,lv.ox,lv.oy,lv.spawn,lv.goals,boxes,hazard_types=kinds)
    assert not route.hard_static[corner] and route.hard_static[axis]


def test_circle_retains_character_body_and_native_gamebox_and_laser_contacts():
    saw=(0.,0.,30.,30.,0.)
    assert not hazard_reaches(30.,30.,saw,SHAPE_CIRCLE)
    assert hazard_reaches(26.,26.,saw,SHAPE_CIRCLE)  # the player's square still touches the circle
    needle=(0.,0.,8.,6.,0.)
    assert hazard_reaches(20.,0.,needle,SHAPE_GAMEBOX)
    assert not hazard_reaches(20.,0.,needle,SHAPE_BOX)
    laser=(0.,0.,math.hypot(30.,30.),0.,math.pi/4)
    assert hazard_reaches(20.,20.,laser,SHAPE_SEGMENT)
    assert not hazard_reaches(20.,-20.,laser,SHAPE_SEGMENT)


def test_real_trainer_feeds_round_saw_map_to_ppo_and_keeps_shape_when_moving(tmp_path):
    args=NS(name=str(tmp_path),agents=2,load=None,shape_inputs=False,paths=1,
            wave_seconds=20.,test=False,batch=999999,debug_spawn=False,realtest=False)
    t=R.Trainer(args);t.script=NS(post=lambda m:None)
    info,words=level_info();t.on_info(info,words.tobytes())
    state=[30.,30.,0.,0.,0,0,0,0]
    t.on_frame(dict(wave=1,frame=0,s=[state,state],hz=info['hz'],sh=[]))
    patch=t.traj[0]['o'][-1][10:10+(2*R.K+1)**2].reshape(2*R.K+1,-1)
    assert patch[R.K,R.K]==R.C_EMPTY
    assert t.hazard_types==[SHAPE_CIRCLE] and t.hz_start['saw'][-1]==SHAPE_CIRCLE
    shifted=[[20.,0.,60.,60.,1.,1.,0.,[[1,20.,0.,30.,30.,0.]]]]
    t.on_frame(dict(wave=1,frame=1,s=[state,state],hz=shifted,sh=[[7,60.,40.,4.]]))
    assert t.hazard_types==[SHAPE_CIRCLE,SHAPE_CIRCLE]
    assert len(t.sweep)==2 and all(s[-1]==SHAPE_CIRCLE for s in t.sweep)


def test_multiple_shapes_and_inactive_phase_preserve_native_identity():
    row=[0,0,60,60,1,1,0,[[1,0,0,30,30,0],[2,50,0,8,6,.2],[3,-20,0,20,40,0]]]
    boxes,kept,kinds=hazard_shapes([row,[0,0,60,60,1,1,0,[]]],[False,True])
    assert kept==[0,0,0] and kinds==[SHAPE_CIRCLE,SHAPE_GAMEBOX,SHAPE_SEGMENT]
    assert len(set(R.box_ids(kept,['piece','off'])))==3
    assert hazard_shapes([row[:7]])[2]==[SHAPE_BOX]


def test_monitor_draws_round_saw_not_its_bounding_square(monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy');monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    import pygame
    import monitor as M
    m=M.Monitor.__new__(M.Monitor);m.sc=1.;m.screen=pygame.Surface((100,100));m.screen.fill(M.BG)
    m.pt=lambda x,y:(round(x+50),round(50-y))
    m.t=NS(boxes=[(0.,0.,30.,30.,0.)],hazard_types=[SHAPE_CIRCLE],paths=[],last_states=[],path_of=[],links=[])
    m.draw_map(NS())
    assert m.screen.get_at((75,50))[:3]==M.HAZ
    assert m.screen.get_at((75,25))[:3]==M.BG
