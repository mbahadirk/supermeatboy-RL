from ppo_rewards import RepeatHops, physical_jump, REPEAT_HOP_COST, REPEAT_HOP_CAP
import pytest


def test_waiting_and_walking_off_ledges_never_count_as_jumps():
    grounded=[0,0,0,0,0,0,0]
    air=[0,10,0,200,2,0,0]
    assert not physical_jump((0,0,0,0),True,air)
    assert not physical_jump((0,0,1,0),True,grounded)
    assert not physical_jump((0,0,1,0),False,air)
    assert physical_jump((0,0,1,0),True,air)  # held jump can still produce a real takeoff


def test_small_cost_only_after_repeated_stalled_takeoffs_and_capped_per_life():
    r=RepeatHops()
    assert [r.takeoff((0,0),True) for _ in range(3)]==[0.,0.,0.]
    assert r.takeoff((0,0),True)==REPEAT_HOP_COST
    for _ in range(1000):r.takeoff((0,0),True)
    assert r.paid==pytest.approx(REPEAT_HOP_CAP)
    r.progress()
    for _ in range(10):assert r.takeoff((0,0),True)==0.


def test_progress_and_moving_platform_launch_positions_do_not_pay_repeat_cost():
    r=RepeatHops()
    for n in range(100):
        assert r.takeoff((n*25.,0),True)==0.
    for n in range(100):
        assert r.takeoff((0,n*25.),True)==0.  # climbing up, not a repeated launch height
    for n in range(100):
        r.progress()
        assert r.takeoff((0,0),True)==0.
    assert r.paid==0.


def test_initial_jumps_have_no_cost_during_progress_grace():
    r=RepeatHops()
    for _ in range(20):assert r.takeoff((0,0),False)==0.


def test_real_trainer_charges_repeated_physical_takeoffs_with_jump_held(tmp_path):
    from types import SimpleNamespace as NS
    import numpy as np
    import torch
    import realgame as R
    from test_hazard_shapes import level_info
    torch.set_num_threads(2)
    args=NS(name=str(tmp_path),agents=1,load=None,shape_inputs=False,paths=1,wave_seconds=20.,
            test=False,batch=999999,debug_spawn=False,realtest=False)
    t=R.Trainer(args);t.script=NS(post=lambda m:None)
    info,words=level_info();info['hz']=[];info['lethalTypes']=[];info['lethalIds']=[];info['lethal']=0
    t.on_info(info,words.tobytes())
    t.ppo.act=lambda obs:(np.array([3]),np.zeros(1),np.zeros(1),None)  # hold jump throughout
    for f in range(162):
        airborne=f%40==1
        s=[30.,31. if airborne else 30.,0.,200. if airborne else 0.,2 if airborne else 0,0,0,0]
        t.on_frame(dict(wave=1,frame=f,s=[s],hz=[],sh=[]))
    assert t.repeat_hops[0].paid==pytest.approx(2*REPEAT_HOP_COST)
    assert len(t.traj[0]['r'])==161
