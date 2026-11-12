"""Paired opt-in/off recorded validation windows; no fitting, game, or historical-report overwrite."""
import gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from game.level import Level,LockContext,hazard_boxes
from plan.planner import Planner,imagine,collide
from plan.progress import ProgressField
from wm.dataset import Dataset,canonical,digest,file_hash,load_model
from wm.features import LevelGeom,model_layout
from wm.rollout import World,rollout_batched
BASE=Path(__file__).resolve().parent
OUTPUT=BASE/(sys.argv[1]if len(sys.argv)>1 else'evaluation.json')
assert OUTPUT.name in ('evaluation.json','evaluation_final.json')
if OUTPUT.exists():raise FileExistsError(OUTPUT)
torch.set_num_threads(2)
def load_document(name):
    d=json.loads((BASE/name).read_text())
    assert d['sha256']==digest(canonical({k:v for k,v in d.items()if k!='sha256'}))
    return d
pickup=load_document('cases.json');gate=load_document('gate_cases.json')
ds=Dataset.load(ROOT/'runs/experiments/tile_shapes_epochs_v1/dataset.json')
assert ds.hash==pickup['manifest_sha256']==gate['manifest_sha256']
model,identity=load_model(ROOT/'runs/experiments/tile_shapes_epochs_v1/epoch_12/model/wm_model.pt')
assert identity['sha256']==pickup['model_sha256'];ds.require_model(model);model.eval()
code_files=['game/level.py','wm/rollout.py','plan/planner.py','plan/geometry.py','wm/features.py']
source_hashes={p:file_hash(ROOT/p)for p in code_files}
requested=pickup['cases']+gate['cases']
wanted={c['entry']for c in requested};snapshots={}
for meta,rows,world in ds.iter_shards(pickup['source'],split='validation',max_shards=160,max_rows=1500000,max_seconds=90):
    if meta.get('level')=='2-10':snapshots[meta['_dataset_index']]=(meta,rows,world)
words,lm=ds.level(pickup['source'],'2-10')
context=LockContext(words,lm)
records=[];carry=[];guard_tests=[];start=time.monotonic()
native_check=dict(transitions=0,pickups=0,mask_mismatches=0)
for meta,rows,world in snapshots.values():
    owned=ds.rows_for(meta,rows,'validation')
    for agent in np.unique(owned['agent']):
        r=np.sort(owned[owned['agent']==agent],order='frame')
        if len(r)<2:continue
        masks,_=context.pickup_path(r['x'],r['y'],int(r['keys'][0]),step=1)
        native_check['transitions']+=len(r)-1
        native_check['pickups']+=int(((r['keys'][1:]==1)&(r['keys'][:-1]==0)).sum())
        native_check['mask_mismatches']+=int((masks!=r['keys']).sum())
def summary(x):return dict(n=len(x),median=float(np.median(x))if len(x)else None,p90=float(np.percentile(x,90))if len(x)else None)
for ci,c in enumerate(requested):
    assert time.monotonic()-start<180
    meta,rows,world=snapshots[c['entry']]
    assert ds.entries[c['entry']]['agents'][str(c['agent'])]['split']=='validation'
    r=np.sort(rows[rows['agent']==c['agent']],order='frame')
    by={int(v['frame']):v for v in r};wf={int(w['f']):w for w in world}
    F=c['frame'];initial=np.array([by[F]])
    assert int(initial['keys'][0])==c['initial_keys']
    actions=[]
    for j in range(48):
        a,b=by.get(F+j),by.get(F+j+1)
        if a is None or b is None or a['status']!=0 or b['status']!=0 or a['action']<0 or F+j not in wf:break
        actions.append(int(a['action']))
    H=len(actions)//4*4
    assert H>=4
    acts=np.array([actions[:H]],np.int64)
    worlds=[];dts=[];fans=[];keep=None
    for f in sorted(wf):
        w=wf[f];keep=w.get('keep',keep)
        if isinstance(w.get('fx'),dict):fans=w['fx'].get('fans',fans)
        if F<=f<F+H:
            worlds.append(World.from_record(w,wf.get(f-1),keep,fans));dts.append(float(w['dt']))
    assert len(worlds)==H and np.allclose(dts,16.667)
    # Fresh private geometry/guards for this window; recorded changes before start only.
    lv=Level(lm,words.astype('<u4').tobytes());w0=world[0]
    lv.set_fields(w0.get('fx'));lv.set_platforms(w0.get('pf'))
    static,_=hazard_boxes(w0.get('hz',[]),w0.get('keep',lv.hz_keep))
    lv.set_sealed(static,[lv.spawn]+lv.goals+lv.keys)
    field=ProgressField(words,lv.ox,lv.oy,lv.spawn,lv.goals,static,(),lv.keys,lv.bandages,lv.portals,
                        lift=lv.wind|lv.plat_support(),blocks=lv.plat_blocks()|lv.sealed_haz|lv.sealed_wall,
                        doors=lv.plat_doors(),outside=lv.outside_cells if lv.kill_box is not None else None)
    pl=Planner(model,lv,field,budget_ms=None,moves=None)
    for f in sorted(wf):
        if f>F:break
        if wf[f].get('tc'):pl.tiles(wf[f]['tc'])
    base_words=pl.geom.words.copy();terrain_words=pl.guard.words.copy()if hasattr(pl.guard,'words')else None
    rec=dict(case=c,horizon=H,modes={})
    for guarded in [False,True]:
        for enabled in [False,True]:
            name=('guarded'if guarded else'raw')+('_pickup'if enabled else'_fixed')
            if guarded:
                out=imagine(model,slice(None),pl.geom,initial.copy(),acts,16.667,worlds,keep_rows=True,
                            terrain=pl.guard_terrain(c['initial_keys']),tol=pl.collide_tol,
                            terrain_for_mask=pl.guard_terrain if enabled else None,key_pickup=enabled)
            else:
                out=rollout_batched(model,pl.geom,initial.copy(),acts,dts,worlds,keep_rows=True,key_pickup=enabled)
            assert out['steps']*4==H
            key_masks=out['rows']['keys'][:,0]
            first=[]
            for member in key_masks:
                ix=np.flatnonzero((member & ~np.uint32(c['initial_keys']))!=0)
                first.append(int(F+4*ix[0])if len(ix)else None)
            majority=np.sum((key_masks & ~np.uint32(c['initial_keys']))!=0,axis=0)>=3
            ix=np.flatnonzero(majority)
            errors={}
            for h in [8,16,32,48]:
                if h>H:continue
                x,y=out['x'][:,0,h//4].mean(),out['y'][:,0,h//4].mean()
                errors[str(h)]=float(np.hypot(x-by[F+h]['x'],y-by[F+h]['y']))
            rec['modes'][name]=dict(first_new_key_coarse_frame_by_member=first,
                                    first_new_key_coarse_frame_majority=int(F+4*ix[0])if len(ix)else None,
                                    end_key_masks=key_masks[:,-1].astype(int).tolist(),position_errors=errors,
                                    guard_clipped_steps=int(out['clip'].sum())if guarded else None,
                                    policy=out.get('key_pickup'))
            assert np.array_equal(pl.geom.words,base_words)
    if c in gate['cases']:
        # Carry only geometric/key logic along the OBSERVED path, not 260+ frame model dynamics.
        path=r[(r['frame']>=c['carried_mask_start'])&(r['frame']<=c['observed_lock_frame'])]
        masks,_=context.pickup_path(path['x'],path['y'],int(path['keys'][0]),step=1)
        carry.append(dict(agent=c['agent'],entry=c['entry'],frames=len(path),
                          observed_pickup=c['observed_pickup_frame'],observed_gate=c['observed_lock_frame'],
                          key_mask_mismatches=int((masks!=path['keys']).sum())))
        gf=c['observed_lock_frame'];prev=np.array([by[gf-1]])
        results={}
        for mask in [0,1]:
            following=np.array([by[gf]]);incoming=prev.copy()
            incoming['keys']=mask;following['keys']=mask
            results[str(mask)]=dict(
                clipped=bool(collide(pl.guard_terrain(mask),incoming,following)[0]),
                resulting_xy=[float(following['x'][0]),float(following['y'][0])])
        guard_tests.append(dict(agent=c['agent'],entry=c['entry'],frame=gf,comparison=results))
    records.append(rec)
    if (ci+1)%8==0:print(json.dumps(dict(cases=ci+1,seconds=time.monotonic()-start)),flush=True)
panels={}
for mode in records[0]['modes']:
    main=records[:len(pickup['cases'])];positives=[r for r in main if r['case']['observed_pickup_frame']is not None]
    negatives=[r for r in main if r['case']['observed_pickup_frame']is None]
    timing=[r['modes'][mode]['first_new_key_coarse_frame_majority']-r['case']['observed_pickup_frame']for r in positives
            if r['modes'][mode]['first_new_key_coarse_frame_majority']is not None]
    panels[mode]=dict(positive_cases=len(positives),negative_cases=len(negatives),
                      detected=sum(r['modes'][mode]['first_new_key_coarse_frame_majority']is not None for r in positives),
                      false_positive=sum(r['modes'][mode]['first_new_key_coarse_frame_majority']is not None for r in negatives),
                      coarse_frame_error_detected=summary(timing),
                      pickup_windows_position={str(h):summary([r['modes'][mode]['position_errors'][str(h)]for r in main
                                                               if str(h)in r['modes'][mode]['position_errors']])for h in [8,16,32,48]})
assert all(file_hash(ROOT/p)==h for p,h in source_hashes.items())
result=dict(protocol='paired-imagined-key-eval-v1',manifest_sha256=ds.hash,checkpoint=identity,
            cases_sha256=pickup['sha256'],gate_cases_sha256=gate['sha256'],source_hashes=source_hashes,
            panels=panels,native_endpoint_replay=native_check,observed_path_carry=carry,guard_counterfactuals=guard_tests,records=records,
            seconds=time.monotonic()-start,game_connected=False,training=False,
            limits=pickup['limits']+gate['limits'])
with OUTPUT.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(panels=panels,carry_mismatches=sum(c['key_mask_mismatches']for c in carry),seconds=result['seconds'])))
