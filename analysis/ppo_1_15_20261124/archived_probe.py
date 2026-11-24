"""Compare archived 1-9 action agreement; this is not a live success-rate test."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
report=json.loads((HERE/'results.json').read_text())
archive=ROOT.parents[1]/'analysis/ppo_1_9_compare_20261124'
x=np.load(archive/'current_inputs.npz')['obs']
a=np.load(archive/'old_inputs.npz')['actions']
out=dict(scope='335 archived successful states with historical reconstructed current inputs, before visibility fix. Action agreement is not a live success rate.',models={})
for name,ck in report['checkpoints'].items():
    path=HERE/'current_snapshot.pt' if name=='current_1_15' else Path(ck['path'])
    sd=torch.load(path,map_location='cpu',weights_only=False)['policy']
    o=torch.from_numpy(x[:,:ck['obs']].copy())
    for k in (0,2):o=torch.tanh(F.linear(o,sd[f'pi.{k}.weight'],sd[f'pi.{k}.bias']))
    p=F.softmax(F.linear(o,sd['pi.4.weight'],sd['pi.4.bias']),-1).numpy()
    out['models'][name]=dict(action_agreement=float((p.argmax(1)==a).mean()),
        recorded_action_probability=float(p[np.arange(len(a)),a].mean()),
        entropy=float(-(p*np.log(p+1e-30)).sum(1).mean()))
report['archived_1_9_probe']=out
(HERE/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(out,indent=2))
