"""Compare checkpoint decisions on the recorded wave-26 plateau, without game control."""
import base64
import io
import json
import sys
import zlib
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import realgame as R

results=json.loads((HERE/'results.json').read_text())
run=ROOT/'runs/aplus_v2_1_14_continue'
info=None;frames=[]
with (run/'native_frames.jsonl.gz').open('rb') as f:
    limit=results['telemetry_snapshot_bytes'];dec=zlib.decompressobj(31);tail=b''
    while f.tell()<limit:
        raw=tail+dec.decompress(f.read(min(1024*1024,limit-f.tell())))
        lines=raw.split(b'\n');tail=lines.pop()
        for line in lines:
            if not line:continue
            row=json.loads(line);m=row['payload']
            if row.get('resolved_level')!='1-15':continue
            if m['ev']=='info':info=row
            if m['ev']=='frame' and m['wave']==26:frames.append(row)

assert info is not None and len(frames)==1800
(HERE/'native_info.json').write_text(json.dumps(info),encoding='utf-8')
args=R.build_parser().parse_args(['--agents','40','--shape-inputs','--no-monitor',
    '--load',str(HERE/'current_snapshot.pt'),'--name',str(HERE/'offline_wave26'),
    '--wave-seconds','30','--end-after','0'])
t=R.Trainer(args);t.learning=False;t.script=type('Sink',(),{'post':lambda self,m:None})()
t.on_info(info['payload'],base64.b64decode(info['tiles_base64']))
observations=[];meta=[];takeoffs=[];deaths=[]
for row in frames:
    m=row['payload'];recorded=row['policy_response']['a']
    for i,s in enumerate(m['s']):
        if s[6]==1 and t.alive[i]:deaths.append([m['frame'],i,*s[:2]])
    def replay_act(obs):
        ids=[i for i,s in enumerate(m['s']) if t.alive[i] and s[6]==0]
        if m['frame']%30==0:
            for j,k in enumerate(ids):
                if k%4==0:
                    observations.append(obs[j].copy());meta.append([m['frame'],k,*m['s'][k][:2]])
        acts=np.array([R.ACTIONS.index(tuple(recorded[i])) for i in ids])
        return acts,np.zeros(len(ids),np.float32),np.zeros(len(ids),np.float32),None
    t.ppo.act=replay_act;t.on_frame(m)
t.log.close()
o=np.array(observations,np.float32);meta=np.array(meta)
np.savez_compressed(HERE/'wave26_probes.npz',obs=o,meta=meta)
results['wave26_probe']=dict(observations=len(o),deaths=deaths,
    reconstruction_scope='Wave26 native states/actions; initial native level routes recomputed without prior-wave hazard sweep memory. Same probe inputs across checkpoints; not original policy logits or live success rates.',
    groups={})
groups=dict(all=np.ones(len(o),bool),lower_plateau=(meta[:,2]>90)&(meta[:,3]>-285)&(meta[:,3]<-190),
            after_5_seconds=meta[:,0]>300)
for name,ck in results['checkpoints'].items():
    path=HERE/'current_snapshot.pt' if name=='current_1_15' else Path(ck['path'])
    sd=torch.load(path,map_location='cpu',weights_only=False)['policy']
    policy=R.Policy(obs=sd['pi.0.weight'].shape[1]);policy.load_state_dict(sd);policy.eval()
    with torch.no_grad():
        x=torch.as_tensor(o[:,:policy.pi[0].in_features]);p=policy.dist(x).probs.numpy()
    for group,mask in groups.items():
        q=p[mask];jp=q[:,[i for i,a in enumerate(R.ACTIONS) if a[2]]].sum(1)
        left=q[:,[i for i,a in enumerate(R.ACTIONS) if a[0]]].sum(1)
        right=q[:,[i for i,a in enumerate(R.ACTIONS) if a[1]]].sum(1)
        results['wave26_probe']['groups'].setdefault(group,{})[name]=dict(n=int(mask.sum()),
            jump_mean=float(jp.mean()),left_mean=float(left.mean()),right_mean=float(right.mean()),
            entropy_mean=float((-(q*np.log(q+1e-30)).sum(1)).mean()),
            maxp_mean=float(q.max(1).mean()),jump_above_99_fraction=float((jp>.99).mean()),
            action_means=q.mean(0).tolist())
    for k,s in ck['layers'].items():
        if np.prod(s['shape'])==1:s['std']=0.0
(HERE/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
print('native info', {k:info['payload'].get(k) for k in ['W','H','spawn','goals','lethal','kb']})
for group,g in results['wave26_probe']['groups'].items():
    print('group',group)
    for name,s in g.items():print(name, {k:round(v,4) if isinstance(v,float) else v for k,v in s.items() if k!='action_means'})
