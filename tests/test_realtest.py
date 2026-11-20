"""PPO-only realtest: native timing, terminal processing, backup and progression contracts."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import realgame as R
import realtest as RT
import levels_smb as L
from perception_cases import fixture


def message(seconds=9.,status=2):
    return dict(wave=3,frame=1000,status=[status],finish=[999],finish_seconds=[seconds])


def trainer(tmp_path,monkeypatch):
    monkeypatch.setattr(RT,'a_plus_time',lambda level:10. if level=='4-1' else None)
    t=R.Trainer.__new__(R.Trainer);t.args=SimpleNamespace(realtest=True)
    t.level=SimpleNamespace(name='4-1');t.stopping=False;t.realtest_finishing=False;t.save_backup=None
    t.realtest_progress=RT.APlusProgress()
    t.run_dir=str(tmp_path/'run');Path(t.run_dir).mkdir()
    t.backup_root=str(tmp_path/'backups');t.saved=[];t.save=lambda:t.saved.append(True)
    save=tmp_path/'native.dat';save.write_bytes(b'original save')
    monkeypatch.setattr(R,'locate_save',lambda:str(save))
    return t,save


def test_parser_and_scripted_mode_rejection():
    assert R.build_parser().parse_args(['--realtest']).realtest
    assert not R.build_parser().parse_args([]).realtest
    with pytest.raises(ValueError,match='PPO'):
        R.Trainer(SimpleNamespace(realtest=True,test=True))


@pytest.mark.parametrize('seconds',[None,-1,0,10.1,float('inf'),float('nan'),True])
def test_invalid_or_non_aplus_time_never_advances(tmp_path,monkeypatch,seconds):
    t,save=trainer(tmp_path,monkeypatch)
    assert t.prepare_realtest(message(seconds)) is None
    assert not t.saved and not t.realtest_finishing and t.save_backup is None


def test_native_seconds_determine_aplus_not_frame_count(tmp_path,monkeypatch):
    t,save=trainer(tmp_path,monkeypatch)
    m=message(10.);m.update(status=[2,2],finish=[100,900],finish_seconds=[11.,9.])
    assert t.prepare_realtest(m) is None
    for wave in range(4,8):
        assert t.prepare_realtest(dict(message(11.),wave=wave)) is None
    result=t.prepare_realtest(dict(message(11.),wave=8))
    assert result['after_wave']==8 and result['first_wave']==3
    assert result['agent']==1 and result['seconds']==9. and result['wave']==3
    assert Path(t.save_backup,'native.dat').read_bytes()==b'original save'
    assert save.read_bytes()==b'original save' and t.saved==[True]
    assert t.prepare_realtest(dict(message(),wave=8)) is None and len(t.saved)==1
    assert len(Path(t.run_dir,'realtest.jsonl').read_text().splitlines())==1


def test_missing_clock_par_death_stop_and_backup_failure_skip_progression(tmp_path,monkeypatch):
    t,save=trainer(tmp_path,monkeypatch)
    assert t.prepare_realtest(message(9.,status=1)) is None
    t.level.name='unknown';assert t.prepare_realtest(message()) is None
    t.level.name='4-1';t.stopping=True;assert t.prepare_realtest(message()) is None
    t.stopping=False
    def fail():raise FileNotFoundError('no native save')
    monkeypatch.setattr(R,'locate_save',fail)
    t.realtest_progress=RT.APlusProgress()
    assert t.prepare_realtest(message()) is None
    assert t.prepare_realtest(dict(message(),wave=8)) is None
    assert not t.saved and not t.realtest_finishing


def test_chapter_metadata_par_lookup(tmp_path,monkeypatch):
    (tmp_path/'chapters.json').write_text(json.dumps([dict(file='4-1',par=10.5),dict(file='boss',par=0)]))
    monkeypatch.setattr(L,'IMPORT_DIR',str(tmp_path));monkeypatch.setattr(L,'_PAR',None)
    assert L.a_plus_time('4-1')==10.5
    assert L.a_plus_time('boss') is None and L.a_plus_time('missing') is None


def test_terminal_wave_reaches_ppo_before_native_finish_directive(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'a_plus_time',lambda level:10.)
    monkeypatch.setattr(RT,'a_plus_time',lambda level:10.)
    save=tmp_path/'savegame.dat';save.write_bytes(b'native')
    monkeypatch.setattr(R,'locate_save',lambda:str(save))
    args=SimpleNamespace(name=str(tmp_path/'run'),agents=2,load=None,shape_inputs=False,paths=1,
                         wave_seconds=20.,test=False,batch=999999,debug_spawn=False,realtest=True)
    t=R.Trainer(args);t.backup_root=str(tmp_path/'backups');posted=[]
    t.script=SimpleNamespace(post=posted.append)
    info,words=fixture()
    t.on_message(dict(type='send',payload=dict(info,ev='info')),words.tobytes())
    assert posted[-1]['payload']['realtest_par']==10.
    for f in range(2):
        t.on_message(dict(type='send',payload=dict(ev='frame',wave=1,frame=f,
            s=[[-110.+f,-90.,60.,0.,0,0,0,0]]*2,hz=info['hz'],sh=[])),None)
    t.on_message(dict(type='send',payload=dict(ev='wave_end',wave=1,frame=2,
        status=[2,1],finish=[1,-1],finish_seconds=[7.5,None])),None)
    assert t.ppo.steps==4 and len(t.ppo.batch)==2
    assert 'finish' not in posted[-1]['payload']
    for wave in range(2,7):
        t.on_message(dict(type='send',payload=dict(ev='wave_end',wave=wave,frame=2,
            status=[1,1],finish=[-1,-1],finish_seconds=[None,None])),None)
    go=posted[-1]
    assert go['type']=='go' and go['payload']['finish']['seconds']==7.5
    assert go['payload']['finish']['level']==info['name'] and t.realtest_finishing
    assert (tmp_path/'run/model.pt').is_file()


def test_native_realtest_contract():
    script=Path(__file__).with_name('native_realtest.cjs')
    result=subprocess.run(['node',str(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_midlevel_identification_uses_configured_assets_and_replies_with_identity(tmp_path,monkeypatch):
    import numpy as np
    from game.level import identify_level
    info,words=fixture()
    words=np.asarray(words,dtype='<u4')
    tiles=words.tobytes()
    low=words.reshape(info['H'],info['W']) & 0xff
    semantic=dict(W=info['W'],H=info['H'],origin=[info['ox'],info['oy']],
                  solid=((low>0)&(low<100)).astype(int).tolist())
    (tmp_path/'4-1_semantic.json').write_text(json.dumps(semantic))
    (tmp_path/'chapters.json').write_text(json.dumps([dict(file='4-1',par=10.)]))
    monkeypatch.setattr(L,'IMPORT_DIR',str(tmp_path));monkeypatch.setattr(L,'_PAR',None)
    assert identify_level(info,tiles)=='4-1'
    args=SimpleNamespace(name=str(tmp_path/'run'),agents=2,load=None,shape_inputs=False,paths=1,
                         wave_seconds=20.,test=False,batch=999999,debug_spawn=False,realtest=True)
    t=R.Trainer(args);posted=[];t.script=SimpleNamespace(post=posted.append)
    t.on_message(dict(type='send',payload=dict(info,ev='info',name='unknown')),tiles)
    assert t.level.name=='4-1'
    assert posted[-1]['payload']['level_name']=='4-1'
    assert posted[-1]['payload']['realtest_par']==10.


def test_five_extra_waves_keep_best_without_restarting_countdown():
    p=RT.APlusProgress()
    original=RT.a_plus_time
    RT.a_plus_time=lambda level:10.
    try:
        assert p.observe('4-1',dict(message(9.),wave=10)) is None
        assert p.observe('4-1',dict(message(8.),wave=11)) is None
        assert p.observe('4-1',dict(message(1.),wave=11)) is None  # duplicate ignored
        for wave in (12,13,14):
            assert p.observe('4-1',dict(message(None,status=1),wave=wave)) is None
        d=p.observe('4-1',dict(message(None,status=1),wave=15))
        assert d['seconds']==8. and d['wave']==11 and d['first_wave']==10 and d['after_wave']==15
    finally:RT.a_plus_time=original
