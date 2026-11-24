"""Raw WM-compatible projectile contract, observed phases and checkpoint preservation."""
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import realgame as R
from game.projectiles import ShotClock, parse_shx, N_PROJECTILE, LAUNCHER_WIDTH, PROJECTILE_SCHEMA
from perception_cases import fixture


def shot(f,period=20,life=5):
    age=(f-1)%period
    return [[7,100.+3*age,0.,4.]] if f>=1 and age<life else []


def test_shared_module_matches_main_exactly():
    root=Path(__file__).resolve().parents[1]
    assert (root/'game/projectiles.py').read_bytes()==(root.parents[1]/'game/projectiles.py').read_bytes()


def test_raw_shx_alignment_and_explicit_unknowns():
    sh=[[1,0,0,4],[2,1,0,4]]
    assert parse_shx(sh,[[0,360,0],[5,10,-50]])==[(0,360.,0.),(5,10.,-50.)]
    assert parse_shx(sh,None)==[(None,None,None)]*2
    assert parse_shx(sh,[[0,360,0]])==[(None,None,None)]*2
    assert parse_shx(sh,[[0,float('nan'),0],None])==[(None,None,None)]*2


def test_phase_requires_two_observed_births_and_expires_when_firing_stops():
    clock=ShotClock()
    for f in range(22):clock.observe(f,shot(f),[[0,180,0]] if shot(f) else None)
    L=next(iter(clock.launchers.values()))
    assert (L.P,L.last)==(20,21)
    v=clock.features(100,0).reshape(2,LAUNCHER_WIDTH)[0]
    assert v[9]==1 and v[10]==1 and v[8]==pytest.approx(20/120)
    assert v[3]==pytest.approx(.3) and v[11]==1
    for f in range(22,44):clock.observe(f,[])
    assert clock.features(100,0).reshape(2,LAUNCHER_WIDTH)[0,10]==0
    clock.new_wave()
    v=clock.features(100,0).reshape(2,LAUNCHER_WIDTH)[0]
    assert v[9]==1 and v[10]==0
    clock.observe(0,[]);clock.observe(1,shot(1),[[0,180,0]])
    assert clock.features(100,0).reshape(2,LAUNCHER_WIDTH)[0,10]==1
    clock.new_level()
    assert not clock.features(100,0).any()


def test_midflight_attachment_is_not_misidentified_as_a_birth():
    c=ShotClock()
    for f in range(5):c.observe(f,[[7,100.+3*f,0.,4.]],[[0,180,0]])
    assert not c.launchers
    c.observe(5,[])
    c.observe(6,[[7,100.,0.,4.]],[[0,180,0]])
    assert len(c.launchers)==1
    assert next(iter(c.launchers.values())).P is None


def test_shrapnel_is_not_exposed_as_a_periodic_launcher():
    c=ShotClock();c.observe(0,[]);c.observe(1,[[9000,0,0,10]],[[16,300,0]])
    assert not c.features(0,0).any()


def test_unconfirmed_nearby_track_does_not_hide_verified_clock_or_mutate_raw_rows():
    import copy
    c=ShotClock()
    for f in range(22):c.observe(f,shot(f),[[0,180,0]] if shot(f) else None)
    c.observe(22,shot(22),[[0,180,0]])
    raw=[[8,0.,0.,4.],*shot(23)];ext=[[0,180.,0.],[0,180.,0.]]
    original=copy.deepcopy((raw,ext));c.observe(23,raw,ext)
    assert (raw,ext)==original
    first=c.features(0,0).reshape(2,LAUNCHER_WIDTH)[0]
    assert first[9]==1 and first[1]==pytest.approx(.5)


@pytest.mark.parametrize('shape',[False,True])
def test_widening_preserves_actor_and_critic_with_arbitrary_new_inputs(tmp_path,shape):
    old=R.Policy(R.obs_size(shape));file=tmp_path/'old.pt'
    torch.save({'policy':old.state_dict()},file)
    new=R.Policy(R.obs_size(shape,True));assert R.load_policy(new,str(file))
    x=torch.randn(8,R.obs_size(shape));extra=torch.randn(8,N_PROJECTILE)
    with torch.no_grad():
        assert torch.allclose(old.pi(x),new.pi(torch.cat([x,extra],1)),atol=1e-6)
        assert torch.allclose(old.vf(x),new.vf(torch.cat([x,extra],1)),atol=1e-6)


def test_adding_shape_maps_relocates_existing_projectile_columns(tmp_path):
    old=R.Policy(R.obs_size(False,True));file=tmp_path/'clock.pt'
    torch.save({'policy':old.state_dict()},file)
    new=R.Policy(R.obs_size(True,True));assert R.load_policy(new,str(file))
    x=torch.randn(8,R.OBS);clock=torch.randn(8,N_PROJECTILE);shape=torch.randn(8,R.N_SHAPE)
    with torch.no_grad():
        assert torch.allclose(old.pi(torch.cat([x,clock],1)),new.pi(torch.cat([x,shape,clock],1)),atol=1e-6)
        assert torch.allclose(old.vf(torch.cat([x,clock],1)),new.vf(torch.cat([x,shape,clock],1)),atol=1e-6)
    with pytest.raises(ValueError):R.load_policy(R.Policy(R.obs_size(True)),str(file))


def test_native_newborn_velocity_reaches_real_ppo_input_and_checkpoint(tmp_path):
    args=R.build_parser().parse_args(['--name',str(tmp_path),'--agents','2','--shape-inputs','--no-monitor'])
    assert args.projectile_inputs
    t=R.Trainer(args);t.learning=False;t.script=NS(post=lambda m:None)
    info,words=fixture();t.on_info(info,words.tobytes())
    t.on_frame(dict(wave=1,frame=0,s=[[-110.,-90.,0.,0.,0,0,0,0]]*2,
                    hz=info['hz'],sh=[[7,-100.,-80.,4.]],shx=[[0,360.,0.]]))
    assert np.array_equal(t.hz_vel[-1],[360.,0.])
    assert t.traj[0]['o'][0].shape==(550,)
    t.save();saved=torch.load(tmp_path/'model.pt',weights_only=True)
    assert saved['obs']==550 and saved['projectile_schema']==PROJECTILE_SCHEMA
    t.log.close()
