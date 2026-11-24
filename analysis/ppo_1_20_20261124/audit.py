"""Offline audit of early repeated 1-20 deaths. Does not control the game or train."""
import base64
import collections
import io
import json
import math
import sys
import zlib
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import realgame as R
HERE.mkdir(exist_ok=True)
run=ROOT/'runs/aplus_v2_1_14_continue'
allwaves=[json.loads(s) for s in (run/'waves.jsonl').read_text().splitlines()]
waves=[r for r in allwaves if r['level']=='1-20'];complete={r['wave'] for r in waves}
selected={1,50,100,200,max(complete)}
out=dict(waves=len(waves),finished=sum(r['finished'] for r in waves),
    dead=sum(r['dead'] for r in waves),timeout=sum(r['truncated'] for r in waves),
    agents=sum(r['agents'] for r in waves),native_waves={},checkpoints={})
for label,q in [('first20',waves[:20]),('last20',waves[-20:])]:
    out[label]=dict(median_frames=float(np.median([r['frames'] for r in q])),
                   min_frames=min(r['frames'] for r in q),max_frames=max(r['frames'] for r in q))
models=dict(current=run/'model.pt',before_1_20=run/'levels/1-19/best.pt',
            before_long_1_14=ROOT/'runs/aplus_v2/levels/1-13/best.pt')
weights={}
for name,path in models.items():
    b=path.read_bytes();sd=torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)['policy'];weights[name]=sd
    import hashlib
    out['checkpoints'][name]=dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),
        all_finite=all(bool(torch.isfinite(t).all()) for t in sd.values()),
        max_abs=max(float(t.abs().max()) for t in sd.values()))
    if name=='current':(HERE/'current_snapshot.pt').write_bytes(b)

info=None;frames={k:[] for k in selected};endings={};previous={};deaths={};counters={}
with (run/'native_frames.jsonl.gz').open('rb') as f:
    limit=f.seek(0,2);f.seek(0);out['telemetry_byte_limit']=limit
    dec=zlib.decompressobj(31);tail=b''
    while f.tell()<limit:
        raw=tail+dec.decompress(f.read(min(1024*1024,limit-f.tell())))
        lines=raw.split(b'\n');tail=lines.pop()
        for line in lines:
            if not line:continue
            row=json.loads(line);m=row['payload']
            if row.get('resolved_level')!='1-20':continue
            if m['ev']=='info':info=row;continue
            wave=m.get('wave')
            if wave not in complete:continue
            if m['ev']=='wave_end':endings[wave]=m;continue
            if m['ev']!='frame':continue
            if wave in selected:frames[wave].append(row)
            c=counters.setdefault(wave,dict(actions=np.zeros(4,np.int64),deaths=[],positions=collections.Counter(),death_frames=[]))
            acts=row.get('policy_response',{}).get('a',[])
            for i,s in enumerate(m['s']):
                key=(wave,i)
                if s[6]==0 and i<len(acts):
                    a=acts[i];c['actions']+=np.array([1,a[0],a[1],a[2]])
                    c['positions'][(round(s[0]/20)*20,round(s[1]/20)*20)]+=1
                elif s[6]==1 and previous.get(key,0)==0:
                    c['deaths'].append(s[:2]);c['death_frames'].append(m['frame'])
                previous[key]=s[6]
assert info is not None
(HERE/'native_info.json').write_text(json.dumps(info),encoding='utf-8')
for wave,c in counters.items():
    a=c['actions'];hist=collections.Counter((round(x/20)*20,round(y/20)*20) for x,y in c['deaths'])
    out['native_waves'][str(wave)]=dict(action_samples=int(a[0]),left=float(a[1]/max(1,a[0])),
        right=float(a[2]/max(1,a[0])),jump=float(a[3]/max(1,a[0])),
        death_frame_median=float(np.median(c['death_frames'])) if c['death_frames'] else None,
        death_points=c['deaths'],death_bins=[[x,y,n] for (x,y),n in hist.most_common(4)])
steps=sum(c['actions'][0] for c in counters.values());batch=16384
out['native_action_samples']=int(steps)
def update_count(carry):
    count=0
    for wave in sorted(counters):
        carry+=int(counters[wave]['actions'][0])
        if carry>=batch:count+=1;carry=0  # production consumes the whole batch including overshoot
    return count
out['update_count_bound']=dict(min=update_count(0),max=update_count(batch-1),
    scope='Derived from recorded action counts, end-of-wave threshold and full batch consumption; previous-level carry unknown, not logged update count.')

args=R.build_parser().parse_args(['--agents','40','--shape-inputs','--no-monitor',
    '--load',str(HERE/'current_snapshot.pt'),'--name',str(HERE/'offline_replay'),
    '--wave-seconds','30','--end-after','0'])
t=R.Trainer(args);t.learning=False;t.script=type('Sink',(),{'post':lambda self,m:None})()
probe=[];meta=[];returns={}
for wave in sorted(selected):
    if not frames[wave] or wave not in endings:continue
    t.on_info(info['payload'].copy(),base64.b64decode(info['tiles_base64']))
    for row in frames[wave]:
        m=row['payload'];recorded=row['policy_response']['a']
        def replay_act(obs):
            ids=[i for i,s in enumerate(m['s']) if t.alive[i] and s[6]==0]
            if m['frame']%3==0:
                for j,k in enumerate(ids):
                    if k%5==0:probe.append(obs[j].copy());meta.append([wave,m['frame'],k,*m['s'][k][:2]])
            acts=np.array([R.ACTIONS.index(tuple(recorded[i])) for i in ids])
            return acts,np.zeros(len(ids),np.float32),np.zeros(len(ids),np.float32),None
        t.ppo.act=replay_act;t.on_frame(m)
    # Complete final terminal rewards, retaining trajectories for this offline diagnostic.
    new=t.new_wave;t.new_wave=lambda n:None
    t.on_wave_end(endings[wave]);t.new_wave=new
    rs=[sum(tr['r']) for tr in t.traj]
    returns[str(wave)]=dict(min=float(min(rs)),median=float(np.median(rs)),max=float(max(rs)),
        action_count=sum(len(tr['a']) for tr in t.traj))
t.log.close()
o=np.array(probe,np.float32);meta=np.array(meta)
np.savez_compressed(HERE/'probes.npz',obs=o,meta=meta)
out['reconstructed_returns']=returns
out['reconstruction_scope']='Initial native tiles/hazards and recorded selected-wave actions. Routes recomputed without earlier-wave hazard sweeps; not original logged rewards/logits.'
out['probe_count']=len(o)
for name,sd in weights.items():
    p=R.Policy(obs=524);p.load_state_dict(sd);p.eval()
    with torch.no_grad():
        dist=p.dist(torch.as_tensor(o));prob=dist.probs.numpy()
    groups=dict(all=np.ones(len(o),bool),spawn=(meta[:,1]<=3),
                early=(meta[:,1]<15),late_recording=(meta[:,0]>=200))
    stats={}
    for label,mask in groups.items():
        q=prob[mask];jump=q[:,[i for i,a in enumerate(R.ACTIONS) if a[2]]].sum(1)
        stats[label]=dict(n=int(mask.sum()),jump=float(jump.mean()),
            right=float(q[:,[i for i,a in enumerate(R.ACTIONS) if a[1]]].sum(1).mean()),
            left=float(q[:,[i for i,a in enumerate(R.ACTIONS) if a[0]]].sum(1).mean()),
            maxp=float(q.max(1).mean()),entropy=float((-(q*np.log(q+1e-30)).sum(1)).mean()),
            action_means=q.mean(0).tolist())
    out['checkpoints'][name]['probes']=stats
(HERE/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print('SUMMARY',{k:v for k,v in out.items() if k not in ['native_waves','checkpoints']})
for name,s in out['checkpoints'].items():print('MODEL',name,s)
for wave in [1,50,100,200,max(complete)]:
    s=out['native_waves'].get(str(wave),{}).copy();s.pop('death_points',None);print('WAVE',wave,s)
