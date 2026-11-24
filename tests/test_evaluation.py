import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest
import torch
import realgame as R
from perception_cases import fixture


def make(tmp_path,limit=1,seed=7):
    torch.set_num_threads(2)
    checkpoint=tmp_path/'source.pt'
    if not checkpoint.exists():torch.save({'policy':R.Policy().state_dict(),'obs':R.OBS},checkpoint)
    args=NS(name=str(tmp_path/'eval_run'),agents=2,load=str(checkpoint),shape_inputs=False,paths=1,
            wave_seconds=30.,test=False,batch=1,debug_spawn=False,realtest=False,eval=True,
            eval_waves=limit,seed=seed,record_states=True,greedy=False,no_lock_step=False)
    t=R.Trainer(args);posted=[];t.script=NS(post=posted.append)
    return t,posted,checkpoint


def test_evaluation_plays_without_learning_or_checkpoint_writes_and_records_results(tmp_path):
    t,posted,checkpoint=make(tmp_path)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    weights={k:v.clone() for k,v in t.policy.state_dict().items()}
    def forbidden(*args,**kwargs):raise AssertionError('evaluation must never train')
    t.ppo.add=forbidden;t.ppo.update=forbidden
    info,words=fixture()
    t.on_message({'type':'send','payload':dict(info,ev='info')},words.tobytes())
    for f in range(3):
        t.on_message({'type':'send','payload':dict(ev='frame',wave=1,frame=f,dt=16.667,
            s=[[-110.+f,-90.,60.,0.,0,0,0,0]]*2,hz=info['hz'],sh=[])},None)
    t.on_message({'type':'send','payload':dict(ev='wave_end',wave=1,frame=3,
        status=[1,2],finish=[-1,2],finish_seconds=[None,2.3])},None)
    assert t.stopping and t.eval_completed==1 and not t.learning
    assert t.ppo.steps==0 and not t.ppo.batch
    t.save();t.save(str(tmp_path/'must_not_exist.pt'))
    t.log.close()
    assert all(torch.equal(v,weights[k]) for k,v in t.policy.state_dict().items())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    assert not list(Path(t.run_dir).rglob('*.pt')) and not (tmp_path/'must_not_exist.pt').exists()
    summary=json.loads(Path(t.run_dir,'evaluation_summary.json').read_text())
    assert summary['agents']==2 and summary['finished']==1 and summary['waves']==1
    row=json.loads(Path(t.run_dir,'evaluation.jsonl').read_text())
    assert row['dt_ms']['mean']==pytest.approx(16.667) and row['fastest_native_seconds']==2.3
    cfg=json.loads(Path(t.run_dir,'run_config.json').read_text())
    assert cfg['lock_step'] and cfg['checkpoint']['sha256']==digest and cfg['seed']==7
    with gzip.open(Path(t.run_dir,'native_frames.jsonl.gz'),'rt') as f:events=[json.loads(x) for x in f]
    assert base64.b64decode(events[0]['tiles_base64'])==words.tobytes()
    assert events[-1]['payload']['ev']=='wave_end'
    assert 'a' in events[1]['policy_response']


def test_evaluation_sampling_is_repeatable_for_policy_seed_and_unlimited_mode(tmp_path):
    t,_,_=make(tmp_path,limit=0)
    info,words=fixture();t.on_info(info,words.tobytes())
    msg=dict(wave=1,frame=0,s=[[-110.,-90.,0.,0.,0,0,0,0]]*2,hz=info['hz'],sh=[])
    a=t.on_frame(msg)['a'];t.log.close()
    second,_,_=make(tmp_path,limit=0)
    second.on_info(info,words.tobytes());b=second.on_frame(msg)['a']
    assert a==b
    second.on_wave_end(dict(wave=1,frame=1,status=[1,1],finish=[-1,-1],finish_seconds=[None,None]))
    assert not second.stopping
    second.log.close()


def test_flags_and_invalid_evaluation_configuration():
    a=R.build_parser().parse_args(['--eval','--load','model.pt','--eval-waves','10','--record-states'])
    assert a.eval and a.eval_waves==10 and a.record_states and not a.no_lock_step
    with pytest.raises(ValueError,match='requires --load'):
        R.Trainer(NS(eval=True,test=False,load=None))


def test_native_fixed_step_lifecycle():
    result=subprocess.run(['node',str(Path(__file__).with_name('native_timing.cjs'))],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_midlevel_identity_uses_same_breakable_tile_decoding_as_live_map(tmp_path,monkeypatch):
    import numpy as np
    import levels_smb
    from game.level import identify_level,tile_solid
    words=np.zeros((10,10),dtype='<u4');words[0:3,:]=0x100|100
    info=dict(W=10,H=10,ox=-100.,oy=-100.)
    (tmp_path/'chapters.json').write_text(json.dumps([dict(file='1-9')]))
    (tmp_path/'1-9_semantic.json').write_text(json.dumps(dict(W=10,H=10,origin=[-100.,-100.],solid=tile_solid(words).astype(int).tolist())))
    monkeypatch.setattr(levels_smb,'IMPORT_DIR',str(tmp_path))
    assert identify_level(info,words.tobytes())=='1-9'
