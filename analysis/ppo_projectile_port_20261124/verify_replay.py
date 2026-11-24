"""Verify shared observed phases against the existing planner on a native 1-20 wave."""
import json
import sys
import zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from game.projectiles import ShotClock, LAUNCHER_WIDTH
from mpc.future import FutureWorld

run=ROOT/'.worktrees/onlyppo/runs/aplus_v2_batch2048'
waves=[json.loads(s) for s in (run/'waves.jsonl').read_text().splitlines()]
wave=next(r['wave'] for r in waves if r['level']=='1-20' and r['frames']>600)
world=FutureWorld();clock=ShotClock();checked=0
with (run/'native_frames.jsonl.gz').open('rb') as f:
    limit=f.seek(0,2);f.seek(0);dec=zlib.decompressobj(31);tail=b''
    done=False
    while f.tell()<limit and not done:
        raw=tail+dec.decompress(f.read(min(1024*1024,limit-f.tell())))
        lines=raw.split(b'\n');tail=lines.pop()
        for line in lines:
            if not line:continue
            row=json.loads(line);m=row['payload']
            if row.get('resolved_level')!='1-20' or m.get('ev')!='frame' or m.get('wave')!=wave:continue
            frame=m['frame'];sh=m.get('sh',[])
            clock.observe(frame,sh,m.get('shx'))
            world.observe(frame,[],[],sh,None,None,m.get('dt',1000/60))
            assert set(clock.launchers)==set(world.launchers)
            for key,L in clock.launchers.items():
                a=world.launchers[key]
                assert (L.P,L.last,L.sid,L.radius)==(a.P,a.last,a.sid,a.radius)
            checked+=1
            if frame>=600:done=True;break
assert checked==601
verified=[L for L in clock.launchers.values() if L.P is not None]
assert len(verified)==4
assert all(L.P==119 for L in verified)
result=dict(wave=wave,frames=checked,observed_sources=len(clock.launchers),verified_launchers=len(verified),period_frames=119,
            endpoint_features=clock.features(420,-40).reshape(2,LAUNCHER_WIDTH).tolist(),
            scope='Offline recorded native sh/shx; no gameplay or learning. Phase fields matched existing FutureWorld every observed frame.')
(Path(__file__).parent/'replay_parity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
