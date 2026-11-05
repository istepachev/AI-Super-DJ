"""Configured storage is a destination allowlist, independent of old SDK settings."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from superdj import storage_budget as storage

@pytest.fixture
def selected(tmp_path,monkeypatch):
    from superdj.resource_limits import runtime_path
    monkeypatch.setattr('superdj.cues.runtime_config',lambda:json.loads(runtime_path().read_text()))
    root=tmp_path/'selected'
    runtime_path().write_text(json.dumps({'storage_root':str(root),'storage_max_gb':64,'cache_mode':'off'}))
    return root,tmp_path/'old-root'

@pytest.mark.parametrize('entry',['temporary','cleanup','cache-budget','object','lock','session','execution-cache','execution-scratch','analysis','decode-f32','decode-wave','prepare','worker','pipeline','stems','cues','master-scratch','render-output','master-output','cli-work','cli-output'])
def test_old_destinations_refused_before_file_creation(selected,tmp_path,monkeypatch,entry):
    root,outside=selected
    from superdj import execution as engine, audio, adapters, analysis, cues, stems
    monkeypatch.setattr(audio,'ffmpeg_executable',lambda *_:'must-not-run')
    source=tmp_path/'source.wav';source.write_bytes(b'original')
    def invoke():
        if entry=='temporary':return storage.temporary(outside,'run-')
        if entry=='cleanup':return storage.cleanup_stale(outside)
        if entry=='cache-budget':return storage.CacheBudget(outside,10000)
        if entry=='object':return engine.Objects(outside).put('clip','a'*64,{},tmp_path)
        if entry=='lock':
            with engine.object_lock(outside/'key.lock'):pass
        if entry=='session':return engine.Session(outside,scratch_root=root/'scratch')
        if entry=='execution-cache':
            with engine.execution(outside,scratch_root=root/'scratch'):pass
        if entry=='execution-scratch':
            with engine.execution(scratch_root=outside):pass
        if entry=='analysis':return analysis._cache_directory(outside)
        if entry=='decode-f32':return audio.decode_f32_file(source,outside/'decoded.f32')
        if entry=='decode-wave':return audio.decode_wave_file(source,outside/'decoded.wav',sample_rate=8000)
        if entry=='prepare':return adapters.prepare_tracks([],outside,audamcp='must-not-run')
        if entry=='worker':return adapters.run_worker({'action':'prepare','output':str(outside/'pcm.wav')},root/'jobs','bad')
        if entry=='pipeline':
            from superdj.pipeline import PreparationPipeline
            return PreparationPipeline(SimpleNamespace(work=outside),None)
        if entry=='stems':return stems.separate(source,'hash',outside,{},'must-not-run')
        if entry=='cues':return cues.infer(source,outside,{})
        if entry=='master-scratch':
            from superdj.stream_master import master
            return master(source,root/'master.wav',None,scratch=outside)
        if entry=='master-output':return adapters.run_worker({'action':'master','output':str(outside/'master.wav')},root/'jobs','bad')
        if entry in ('cli-work','cli-output'):
            from superdj.cli import render
            return render(SimpleNamespace(work=outside if entry=='cli-work' else root/'render',output=outside if entry=='cli-output' else root/'out'))
        if entry=='render-output':
            from superdj.render import render_plan
            record={'id':'a','path':str(source),'duration':90,'bpm':120,'beats':[i*.5 for i in range(180)],'downbeats':[i*2 for i in range(45)],'grid_confidence':1,'grid_status':'reviewed','grid':{'meter':4,'human_reviewed':True}}
            return render_plan({'order':['a'],'edges':[]},[record],{},outside/'premix.wav',root/'render')
    with pytest.raises(storage.StorageLimitError,match='configured working folder'):invoke()
    assert not outside.exists()
    assert source.read_bytes()==b'original'
    if entry in ('worker','render-output'):assert not (root/'jobs').exists() and not (root/'render').exists()


def test_old_runtime_override_cannot_bypass_saved_policy(selected,monkeypatch):
    root,outside=selected
    from superdj import cues
    from superdj.execution import execution
    monkeypatch.setattr(cues,'runtime_config',lambda:{'scratch_root':str(outside),'storage_root':str(outside),'storage_max_gb':100000})
    with pytest.raises(storage.StorageLimitError):
        with execution():pass
    assert not outside.exists()


def test_defaults_use_selected_root_and_one_decimal_limit(selected):
    root,_=selected
    from superdj.execution import execution
    from superdj.analysis import default_cache_dir
    assert storage.temporary_parent()==root/'scratch'
    assert default_cache_dir()==root/'cache'
    with execution(work_max_bytes=100*1024**3) as session:
        assert session.work.is_relative_to(root)
        assert session.work_budget.maximum==64_000_000_000
        assert root in session.work_budget.roots
        work=session.work
    assert not work.exists()


def test_new_work_and_watch_cannot_adopt_outside_directory(selected):
    root,outside=selected
    from superdj.execution import execution
    with execution(scratch_root=root/'scratch') as session:
        for action in (session.new_work,session.watch_work):
            with pytest.raises(storage.StorageLimitError):action(outside)
    assert not outside.exists()


def test_resolved_link_escape_refused(selected,tmp_path):
    root,outside=selected
    root.mkdir();outside.mkdir();link=root/'escape'
    try:link.symlink_to(outside,target_is_directory=True)
    except OSError:pytest.skip('symlink unavailable')
    with pytest.raises(storage.StorageLimitError):storage.temporary(link,'run-')
    assert list(outside.iterdir())==[]


def test_explicit_reconfiguration_changes_root_atomically(selected,tmp_path):
    from superdj.resource_limits import runtime_path
    root,outside=selected
    storage.configure('off',storage_root=outside,storage_max_gb=64)
    storage.validate_storage_path(outside/'scratch')
    with pytest.raises(storage.StorageLimitError):storage.validate_storage_path(root/'scratch')
    before=runtime_path().read_bytes()
    with pytest.raises(storage.StorageLimitError):storage.configure('configure',scratch_root=root/'scratch')
    assert runtime_path().read_bytes()==before


def test_forbidden_root_remains_forbidden_during_reconfiguration(selected):
    from superdj.resource_limits import runtime_path
    root,outside=selected
    policy=json.loads(runtime_path().read_text());policy['forbidden_storage_roots']=[str(outside)]
    runtime_path().write_text(json.dumps(policy));before=runtime_path().read_bytes()
    with pytest.raises(storage.StorageLimitError):storage.configure('off',storage_root=outside/'new')
    assert not outside.exists() and runtime_path().read_bytes()==before


def test_installed_model_outside_audio_root_is_read_without_new_writes(selected,tmp_path):
    root,_=selected
    from superdj.analysis import prepare_beat_model
    model_root=tmp_path/'installed';target=model_root/'models/final0.ckpt'
    target.parent.mkdir(parents=True);target.write_bytes(b'installed model')
    assert prepare_beat_model(cache_dir=model_root)==target
    assert not root.exists()


def test_direct_budgets_cannot_increase_or_split_workstation_total(selected):
    root,_=selected
    root.mkdir();(root/'reports').mkdir();(root/'reports/existing.txt').write_bytes(b'x'*5000)
    from superdj.resource_limits import runtime_path
    value=json.loads(runtime_path().read_text());value['storage_max_gb']=.00001
    runtime_path().write_text(json.dumps(value))
    work=storage.WorkBudget(999999,[root/'scratch'])
    assert work.maximum==10000 and work.usage()==5000
    budget=storage.CacheBudget(root/'cache',999999)
    try:
        assert budget.maximum==10000
        assert not budget.make_room(6000)
    finally:budget.close()
