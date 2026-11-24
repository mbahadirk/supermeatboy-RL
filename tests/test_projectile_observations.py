"""The PPO observation clock follows main's existing observed-launcher contract."""
from game.projectiles import ShotClock,Launcher,launch_key,parse_shx
from game.level import parse_shx as public_parser
from mpc.future import FutureWorld


def test_observed_launch_period_and_phase_match_future_world():
    clock=ShotClock();world=FutureWorld()
    assert public_parser is parse_shx
    for wave in range(2):
        if wave:clock.new_wave();world.new_wave()
        for f in range(90):
            age=(f-1)%20
            sh=[[7,100.+3*age,0.,4.]] if f>=1 and age<5 else []
            clock.observe(f,sh,[[0,180.,0.]] if sh else None)
            world.observe(f,[],[],sh,None,None,1000/60)
            assert set(clock.launchers)==set(world.launchers)
            for key,L in clock.launchers.items():
                old=world.launchers[key]
                assert (L.P,L.last,L.sid,L.radius)==(old.P,old.last,old.sid,old.radius)
    assert isinstance(next(iter(world.launchers.values())),Launcher)


def test_wm_collector_retains_existing_raw_projectile_schema(tmp_path):
    from wm.buffer import Collector
    collector=Collector(str(tmp_path))
    payload=dict(frame=7,dt=1000/60,hz=[],sh=[[10,430.5,-49.5,10.]],shx=[[0,-360.,0.]])
    collector.frame(payload,[])
    assert collector.world[0]['sh']==payload['sh']
    assert collector.world[0]['shx']==payload['shx']
