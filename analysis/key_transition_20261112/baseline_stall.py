"""Two bounded fresh planner decisions on the recorded stall; no game or persistent memories."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from game.level import Level,hazard_boxes
from wm.buffer import load_level
from wm.dataset import file_hash
from wm.rollout import World
from plan import planner as P,safety as S
from plan.progress import ProgressField
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent
DATA=ROOT/'runs/collect_2-10_fan_keys_01/wm_data'
def plain(x):
    if isinstance(x,dict):return {k:plain(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v)for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float)and not np.isfinite(x):return None
    return x
words,meta=load_level(str(DATA),'2-10')
with np.load(DATA/'wave_00001.npz',allow_pickle=False)as z:rows=z['rows']
world=json.loads(gzip.decompress((DATA/'wave_00001_world.json.gz').read_bytes()))
f=200;start=rows[(rows['agent']==0)&(rows['frame']==f)];assert len(start)==1
wf={int(w['f']):w for w in world}
models={'old':ROOT/'runs/wm_step4/wm_model.pt','epoch12':ROOT/'runs/experiments/tile_shapes_epochs_v1/epoch_12/model/wm_model.pt'}
report=dict(frame=f,state={n:plain(start[n][0])for n in start.dtype.names},
            source_hashes={p:file_hash(ROOT/p)for p in ['plan/planner.py','wm/rollout.py','game/level.py','plan/geometry.py']},
            input_hashes={p:file_hash(DATA/p)for p in ['wave_00001.npz','wave_00001_world.json.gz','levels/2-10.npz']},
            limitations=['Recorded frame0 geometry, not missing historical pre-frame info snapshot.',
                         'Recorded future worlds are oracle inputs; fresh memories/no replay, not original accumulated planner state.',
                         'Cross-model comparison also changes input encoding/training data; not a causal encoding test.'],models={})
for label,path in models.items():
    lv=Level(meta,words.astype('<u4').tobytes());w0=world[0]
    lv.set_fields(w0.get('fx'));lv.set_platforms(w0.get('pf'))
    static,_=hazard_boxes(w0.get('hz',[]),w0.get('keep',lv.hz_keep))
    lv.set_sealed(static,[lv.spawn]+lv.goals+lv.keys+lv.bandages+[p for pair in lv.portals for p in pair])
    field=ProgressField(words,lv.ox,lv.oy,lv.spawn,lv.goals,static,(),lv.keys,lv.bandages,lv.portals,
                        lift=lv.wind|lv.plat_support(),blocks=lv.plat_blocks()|lv.sealed_haz|lv.sealed_wall,
                        doors=lv.plat_doors(),outside=lv.outside_cells if lv.kill_box is not None else None)
    pl=P.Planner(str(path),lv,field,budget_ms=None,modes=True,moves=None,hysteresis=3)
    fw=[];fb=[None];fans=[];keep=lv.hz_keep
    for t in sorted(wf):
        w=wf[t];keep=w.get('keep',keep)
        if isinstance(w.get('fx'),dict):fans=w['fx'].get('fans',fans)
        if f<=t<f+pl.future_frames:fw.append(World.from_record(w,wf.get(t-1),keep,fans))
        if f<t<=f+pl.future_frames:fb.append(S.frame_boxes(w.get('hz'),keep,w.get('sh',[])))
    assert len(fw)==pl.future_frames and len(fb)==pl.future_frames+1
    original=P.imagine;captured=[];arrays={}
    def spy(*args,**kwargs):
        out=original(*args,**kwargs)
        i=len(captured);x,y=out['x'],out['y']
        captured.append(dict(call=i,shape=list(x.shape),actions_shape=list(args[4].shape),
                             candidate_max_x=x.max(axis=(0,2)),candidate_max_y=y.max(axis=(0,2)),
                             end_x=x[:,:,-1],end_y=y[:,:,-1]))
        arrays[f'call{i}_x']=x;arrays[f'call{i}_y']=y;arrays[f'call{i}_actions']=args[4]
        return out
    P.imagine=spy;t=time.monotonic()
    try:plan=pl.plan(start.copy(),f,fw,fb,16.667,agent=0,recorded=None)
    finally:P.imagine=original
    with(OUT/f'baseline_{label}_paths.npz').open('xb')as h:np.savez_compressed(h,**arrays)
    report['models'][label]=plain(dict(checkpoint_sha256=file_hash(path),field=field.summary(),seconds=time.monotonic()-t,
                                      plan=plan.to_dict(),stage1=pl.last_stage1,final=pl.last_final,calls=captured))
    print(json.dumps(dict(model=label,chosen=report['models'][label]['plan']['chosen'],
                          score=report['models'][label]['plan']['score'],seconds=time.monotonic()-t)),flush=True)
with(OUT/'baseline_stall.json').open('x',encoding='utf-8')as h:json.dump(report,h,indent=2,allow_nan=False)
