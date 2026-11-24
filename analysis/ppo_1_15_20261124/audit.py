"""Read-only checkpoint and native replay audit. Never attaches to or controls the game."""
import base64
import collections
import hashlib
import io
import json
import sys
import zlib
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import realgame as R

RUN=ROOT/'runs/aplus_v2_1_14_continue'
HERE.mkdir(exist_ok=True)
models={
    'old_282':ROOT.parents[1]/'runs/collect_2-10_ppo50_01/levels/1-9/best.pt',
    'recovery':ROOT/'runs/ppo_recovery_1_9_20261124_a/model.pt',
    'before_long_1_14':ROOT/'runs/aplus_v2/levels/1-13/best.pt',
    'after_long_1_14':ROOT/'runs/aplus_v2/model.pt',
    'after_visibility_test':ROOT/'runs/aplus_v2_1_14_visibility_fix/model.pt',
    'passed_1_14':RUN/'levels/1-14/best.pt',
    'current_1_15':RUN/'model.pt',
}
weights={};out={'checkpoints':{},'wave_summary':{},'native_waves':{}}
for name,path in models.items():
    data=path.read_bytes()
    ck=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    sd=ck['policy'];weights[name]=sd
    obs=sd['pi.0.weight'].shape[1]
    stats={}
    for k,t in sd.items():
        stats[k]=dict(shape=list(t.shape),finite=bool(torch.isfinite(t).all()),
                      norm=float(t.norm()),max_abs=float(t.abs().max()),std=float(t.std(unbiased=False)))
    out['checkpoints'][name]=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),obs=obs,layers=stats)
    if name=='current_1_15':(HERE/'current_snapshot.pt').write_bytes(data)
waves=[json.loads(line) for line in (RUN/'waves.jsonl').read_text().splitlines()]
for level in dict.fromkeys(r['level'] for r in waves):
    q=[r for r in waves if r['level']==level]
    out['wave_summary'][level]=dict(waves=len(q),finished=sum(r['finished'] for r in q),
        dead=sum(r['dead'] for r in q),timeout=sum(r['truncated'] for r in q),agents=sum(r['agents'] for r in q))

def rows():
    # Fixed byte boundary: the active trainer may still be appending its gzip stream.
    with (RUN/'native_frames.jsonl.gz').open('rb') as f:
        limit=f.seek(0,2);f.seek(0)
        out['telemetry_snapshot_bytes']=limit
        dec=zlib.decompressobj(31);tail=b''
        while f.tell()<limit:
            buf=f.read(min(1024*1024,limit-f.tell()))
            raw=tail+dec.decompress(buf);lines=raw.split(b'\n');tail=lines.pop()
            for line in lines:
                if line:yield json.loads(line)

first_info=None;first_frames=[];first_end=None
action_counts={};positions={};prev={};takeoffs={}
for row in rows():
    f=row['payload'];level=row.get('resolved_level')
    if level!='1-15':continue
    if f['ev']=='info' and first_info is None:first_info=row
    if f['ev']=='wave_end' and f['wave']==1:first_end=f
    if f['ev']!='frame':continue
    if f['wave']==1:first_frames.append(row)
    wave=f['wave'];counts=action_counts.setdefault(wave,np.zeros(4,np.int64))
    pos=positions.setdefault(wave,collections.Counter())
    for k,(s,a) in enumerate(zip(f['s'],row.get('policy_response',{}).get('a',[]))):
        if s[6]!=0:continue
        counts+=np.array([1,a[0],a[1],a[2]])
        pos[(round(s[0]/20)*20,round(s[1]/20)*20)]+=1
        key=(wave,k);old=prev.get(key)
        if old is not None and not (old[4]&2) and (s[4]&2) and s[3]>0:
            takeoffs[wave]=takeoffs.get(wave,0)+1
        prev[key]=s
for wave,c in action_counts.items():
    out['native_waves'][str(wave)]=dict(live_action_samples=int(c[0]),left_fraction=float(c[1]/max(1,c[0])),
        right_fraction=float(c[2]/max(1,c[0])),jump_fraction=float(c[3]/max(1,c[0])),
        physical_takeoffs_sampled=takeoffs.get(wave,0),
        top_positions=[dict(x=k[0],y=k[1],samples=n) for k,n in positions[wave].most_common(4)])

# Reconstruct first-wave observations with recorded actions, never optimizing or drawing.
# Full initial native info/tiles and every first-wave frame are replayed through the real
# producer. Later learned hazard sweeps from other waves are deliberately not invented.
assert first_info is not None and first_frames,'No recorded 1-15 inputs'
args=R.build_parser().parse_args(['--agents','40','--shape-inputs','--no-monitor',
    '--load',str(HERE/'current_snapshot.pt'),'--name',str(HERE/'offline_reconstruction')])
t=R.Trainer(args);t.learning=False;t.script=type('Sink',(),{'post':lambda self,m:None})()
t.on_info(first_info['payload'],base64.b64decode(first_info['tiles_base64']))
observations=[];frame_meta=[]
for row in first_frames:
    f=row['payload'];response=row['policy_response'];recorded=response['a']
    def replay_act(obs):
        ids=[i for i,s in enumerate(f['s']) if t.alive[i] and s[6]==0]
        if f['frame']%30==0:
            for j,k in enumerate(ids[::4]):
                observations.append(obs[ids.index(k)].copy())
                frame_meta.append([f['frame'],k,*f['s'][k][:2]])
        actions=np.array([R.ACTIONS.index(tuple(recorded[i])) for i in ids])
        return actions,np.zeros(len(ids),np.float32),np.zeros(len(ids),np.float32),None
    t.ppo.act=replay_act
    t.on_frame(f)
out['reconstruction']=dict(first_wave_frames=len(first_frames),probe_observations=len(observations),
    description='First native 1-15 wave replayed offline with recorded actions; static route construction is recomputed, not original logged policy logits.')
o=np.array(observations,np.float32)
np.savez_compressed(HERE/'probe_inputs.npz',obs=o,meta=np.array(frame_meta))
for name,sd in weights.items():
    policy=R.Policy(obs=sd['pi.0.weight'].shape[1]);policy.load_state_dict(sd);policy.eval()
    with torch.no_grad():
        x=torch.as_tensor(o[:,:policy.pi[0].in_features])
        h1=policy.pi[1](policy.pi[0](x));h2=policy.pi[3](policy.pi[2](h1))
        dist=policy.dist(x);p=dist.probs.numpy();v=policy.vf(x).numpy().ravel()
        jp=p[:,[i for i,a in enumerate(R.ACTIONS) if a[2]]].sum(1)
        out['checkpoints'][name]['probe']=dict(entropy_mean=float(dist.entropy().mean()),
            max_action_prob_mean=float(p.max(1).mean()),jump_prob_mean=float(jp.mean()),
            jump_prob_p10_p50_p90=np.percentile(jp,[10,50,90]).tolist(),
            p_jump_above_99_fraction=float((jp>.99).mean()),
            mean_action_probs=p.mean(0).tolist(),std_action_probs=p.std(0).tolist(),
            layer1_tanh_saturation_fraction=float((h1.abs()>.99).float().mean()),
            layer2_tanh_saturation_fraction=float((h2.abs()>.99).float().mean()),
            value_mean=float(v.mean()),value_min=float(v.min()),value_max=float(v.max()))
out['parameter_relative_changes']={}
for a,b in [('before_long_1_14','after_long_1_14'),('after_long_1_14','passed_1_14'),('passed_1_14','current_1_15')]:
    out['parameter_relative_changes'][a+' -> '+b]={}
    for branch in ['pi','vf']:
        keys=[k for k in weights[a] if k.startswith(branch+'.')]
        numerator=sum(float((weights[a][k]-weights[b][k]).square().sum()) for k in keys)
        denominator=sum(float(weights[a][k].square().sum()) for k in keys)
        out['parameter_relative_changes'][a+' -> '+b][branch]=float((numerator/max(denominator,1e-20))**.5)
(HERE/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('wave summary',out['wave_summary'])
for name,ck in out['checkpoints'].items():print(name,'finite',all(s['finite'] for s in ck['layers'].values()),'probe',ck['probe'])
print('relative changes',out['parameter_relative_changes'])
t.log.close()
