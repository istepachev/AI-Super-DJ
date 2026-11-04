import json
from pathlib import Path
import pytest
from superdj.execution import Objects, cached, execution
from superdj.storage_budget import CacheBudget, WorkBudget, StorageLimitError, configure, tree_bytes


def store(objects, owned, key, data):
    owned.mkdir(parents=True, exist_ok=True)
    path=owned/'audio.wav';path.write_bytes(data)
    objects.put('clip', key*64, {'path':str(path)}, owned)
    return objects.get('clip', key*64)


def test_lru_evicts_old_unleased_objects_only(tmp_path):
    root=tmp_path/'cache'
    first=CacheBudget(root,10000);a=Objects(root,first)
    old=store(a,tmp_path/'a','a',b'a'*3500);first.close()
    second=CacheBudget(root,10000);b=Objects(root,second)
    keep=store(b,tmp_path/'b','b',b'b'*3500);second.close()
    third=CacheBudget(root,10000);c=Objects(root,third)
    new=store(c,tmp_path/'c','c',b'c'*3500)
    assert not Path(old['path']).exists()
    assert Path(keep['path']).read_bytes()==b'b'*3500
    assert Path(new['path']).read_bytes()==b'c'*3500
    assert tree_bytes(root)<=10000
    third.close()


def test_another_live_session_pin_prevents_eviction(tmp_path):
    root=tmp_path/'cache';first=CacheBudget(root,10000);a=Objects(root,first)
    value=store(a,tmp_path/'a','a',b'a'*6000)
    second=CacheBudget(root,10000);b=Objects(root,second)
    assert store(b,tmp_path/'b','b',b'b'*6000) is None
    assert Path(value['path']).read_bytes()==b'a'*6000
    assert second.skipped_writes==1
    second.close();first.close()


def test_oversized_cache_item_falls_back_without_changing_pcm(tmp_path):
    pcm=bytes(range(256))*16
    with execution(tmp_path/'cache',cache_max_bytes=2500,scratch_root=tmp_path/'scratch') as session:
        def render():
            path=session.work/'render.wav';path.write_bytes(pcm);return {'path':str(path)}
        value=cached('clip', {'source':'input'}, render, session.work)
        assert Path(value['path']).read_bytes()==pcm
        assert session.cache_budget.skipped_writes==1
        assert not list((tmp_path/'cache').rglob('pending-*'))
    assert not Path(value['path']).exists()


def test_work_limit_refusal_cleans_only_owned_scratch(tmp_path):
    original=tmp_path/'source.wav';original.write_bytes(b'keep')
    with pytest.raises(StorageLimitError):
        with execution(scratch_root=tmp_path/'scratch',work_max_bytes=200) as session:
            work=session.work;(work/'large.wav').write_bytes(b'x'*201)
            cached('clip',{},lambda:None,work)
    assert not work.exists()
    assert original.read_bytes()==b'keep'


def test_explicit_parent_work_has_owned_child_and_is_cleaned(tmp_path):
    parent=tmp_path/'work';parent.mkdir();(parent/'user.txt').write_text('keep')
    with execution(scratch_root=tmp_path/'scratch') as session:
        child=session.new_work(parent);(child/'audio.wav').write_bytes(b'x')
    assert not child.exists()
    assert (parent/'user.txt').read_text()=='keep'


def test_directory_link_is_refused_without_touching_target(tmp_path):
    source=tmp_path/'outside';source.mkdir();(source/'original.wav').write_bytes(b'keep')
    root=tmp_path/'cache';root.mkdir()
    try:(root/'link').symlink_to(source,target_is_directory=True)
    except OSError:pytest.skip('symlink unavailable')
    with pytest.raises(StorageLimitError):tree_bytes(root)
    assert (source/'original.wav').read_bytes()==b'keep'


@pytest.mark.parametrize('number',[0,-1,float('inf'),float('nan'),True])
def test_invalid_limit_preserves_runtime_settings(tmp_path,number):
    path=tmp_path/'runtime.json';before={'resource_limits':{'enabled':True},'default_style':'zhilin'}
    path.write_text(json.dumps(before))
    with pytest.raises(ValueError):configure('configure',path=path,cache_max_gib=number)
    assert json.loads(path.read_text())==before


def test_storage_settings_are_field_scoped_and_default_off(tmp_path):
    path=tmp_path/'runtime.json';path.write_text(json.dumps({'resource_limits':{'enabled':True}}))
    assert configure(path=path)['cache_mode']=='off'
    configure('configure',path=path,cache_root=tmp_path/'cache',cache_max_gib=7,scratch_max_gib=19)
    configure('off',path=path)
    value=json.loads(path.read_text())
    assert value['cache_mode']=='off' and value['cache_max_gib']==7 and value['scratch_max_gib']==19
    assert value['resource_limits']=={'enabled':True}


def test_setup_accepts_empty_cache_choice_as_off(tmp_path):
    from superdj.storage_budget import setup
    answers=iter([str(tmp_path/'work'),'64',''])
    value=setup(path=tmp_path/'runtime.json',ask=lambda _:next(answers))
    assert value['cache_mode']=='off' and value['storage_max_gb']==64


def test_orphan_cleanup_keeps_active_and_unmarked_data(tmp_path,monkeypatch):
    from superdj import storage_budget as m
    dead=tmp_path/'run-dead';dead.mkdir()
    (dead/'.superdj-run.json').write_text(json.dumps({'schema':'superdj-temporary-v1','pid':99999}))
    (dead/'pcm.wav').write_bytes(b'derived')
    live=tmp_path/'run-live';live.mkdir()
    (live/'.superdj-run.json').write_text(json.dumps({'schema':'superdj-temporary-v1','pid':123}))
    user=tmp_path/'user';user.mkdir();(user/'song.wav').write_bytes(b'original')
    monkeypatch.setattr(m,'_alive',lambda pid:pid==123)
    m.cleanup_stale(tmp_path)
    assert not dead.exists() and live.exists() and (user/'song.wav').read_bytes()==b'original'


def test_multiple_owned_workspaces_share_one_limit(tmp_path):
    from superdj.storage_budget import temporary
    a=temporary(tmp_path,'run-');b=temporary(tmp_path,'run-')
    try:
        (Path(a.name)/'pcm.wav').write_bytes(b'x'*100)
        (Path(b.name)/'pcm.wav').write_bytes(b'x'*100)
        limit=WorkBudget(250);limit.watch_parent(tmp_path)
        with pytest.raises(StorageLimitError):limit.check()
    finally:a.cleanup();b.cleanup()


def test_music_folder_cannot_be_adopted_as_cache(tmp_path):
    (tmp_path/'original.mp3').write_bytes(b'user music')
    with pytest.raises(StorageLimitError):CacheBudget(tmp_path,10000)
    assert (tmp_path/'original.mp3').read_bytes()==b'user music'


def test_models_are_never_eviction_targets(tmp_path):
    root=tmp_path/'cache';(root/'models').mkdir(parents=True)
    model=root/'models/model.ckpt';model.write_bytes(b'model'*3000)
    budget=CacheBudget(root,10000);objects=Objects(root,budget)
    assert store(objects,tmp_path/'work','a',b'pcm') is None
    assert model.read_bytes()==b'model'*3000
    budget.close()


def test_failed_copy_removes_only_pending_cache_object(tmp_path,monkeypatch):
    import superdj.execution as m
    root=tmp_path/'cache';budget=CacheBudget(root,10000);objects=Objects(root,budget)
    original=tmp_path/'work';original.mkdir();source=original/'audio.wav';source.write_bytes(b'source PCM')
    def fail(a,b):
        Path(b).write_bytes(b'partial');raise OSError('copy interrupted')
    monkeypatch.setattr(m.shutil,'copyfile',fail)
    with pytest.raises(OSError):objects.put('clip','a'*64,{'path':str(source)},original)
    assert not list(root.rglob('pending-*')) and source.read_bytes()==b'source PCM'
    budget.close()


def test_cli_exposes_per_run_storage_choices(monkeypatch):
    import superdj.cli as cli
    import superdj.make as make
    seen=[];monkeypatch.setattr(make,'make',lambda args:seen.append(args))
    cli.main(['make','folder','--no-cache','--scratch-root','D:/audio-temp','--scratch-max-gib','30','--cache-max-gib','5'])
    assert seen[0].no_cache and seen[0].scratch_max_gib==30 and seen[0].cache_max_gib==5


def test_process_liveness_probe_does_not_signal_child():
    import subprocess,sys
    from superdj.storage_budget import _alive
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.5)'])
    assert _alive(child.pid) and child.poll() is None
    child.wait(timeout=5)
    assert not _alive(child.pid)


def test_worker_shutdown_failure_still_cleans_owned_pcm(tmp_path,monkeypatch):
    with pytest.raises(RuntimeError,match='shutdown failed'):
        with execution(scratch_root=tmp_path/'scratch') as session:
            work=session.work;(work/'pcm.wav').write_bytes(b'derived')
            def fail():raise RuntimeError('shutdown failed')
            monkeypatch.setattr(session,'release_workers',fail)
    assert not work.exists()


def test_direct_sdk_cache_call_cannot_hide_large_external_work(tmp_path):
    outside=tmp_path/'explicit-work';outside.mkdir();(outside/'pcm.wav').write_bytes(b'x'*300)
    with execution(scratch_root=tmp_path/'scratch',work_max_bytes=250):
        with pytest.raises(StorageLimitError):cached('clip',{},lambda:None,outside)
    # Explicit external directories are inspected, never recursively deleted.
    assert (outside/'pcm.wav').read_bytes()==b'x'*300


def test_existing_one_argument_cache_adapter_still_works(tmp_path,monkeypatch):
    import superdj.execution as module
    original=module.Objects
    class LegacyAdapter(original):
        def __init__(self,root):super().__init__(root)
    monkeypatch.setattr(module,'Objects',LegacyAdapter)
    with module.execution(tmp_path/'cache',scratch_root=tmp_path/'scratch') as session:
        assert session.objects.budget is session.cache_budget
        def produce():
            target=session.work/'pcm.wav';target.write_bytes(b'same PCM');return {'path':str(target)}
        result=module.cached('clip',{},produce,session.work)
        assert Path(result['path']).read_bytes()==b'same PCM'


def test_cache_and_work_share_the_total_folder_limit(tmp_path):
    root=tmp_path/'store';root.mkdir();(root/'work.wav').write_bytes(b'w'*5000)
    budget=CacheBudget(root/'cache',10000);budget.shared_budget=WorkBudget(10000,[root]);objects=Objects(root/'cache',budget)
    assert store(objects,tmp_path/'generated','a',b'a'*6000) is None
    assert tree_bytes(root)<=10000
    assert (root/'work.wav').read_bytes()==b'w'*5000
    budget.close()


def test_total_folder_setting_and_forbidden_location(tmp_path,monkeypatch):
    from superdj import storage_budget as module
    blocked=tmp_path/'synced';base=tmp_path/'local';path=tmp_path/'runtime.json'
    monkeypatch.setattr(module,'workstation_policy',lambda:{'forbidden_storage_roots':[str(blocked)]})
    result=configure('off',path=path,storage_root=base,storage_max_gb=64)
    assert result['storage_max_gb']==64 and Path(result['scratch_root'])==base/'scratch'
    assert Path(result['cache_root'])==base/'cache' and result['cache_mode']=='off'
    with pytest.raises(StorageLimitError):configure('configure',path=path,storage_root=blocked/'cache')
    assert not blocked.exists()


def test_configured_total_budget_covers_persistent_and_temporary_data(tmp_path,monkeypatch):
    from superdj import storage_budget as module
    base=tmp_path/'store'
    monkeypatch.setattr(module,'workstation_policy',lambda:{'storage_root':str(base),'storage_max_gb':.00001})
    with execution(base/'cache',scratch_root=base/'scratch',cache_max_bytes=20000,work_max_bytes=20000) as session:
        assert session.work_budget.maximum==10000
        assert session.cache_budget.shared_budget is session.work_budget
        (session.work/'pcm.wav').write_bytes(b'x'*10001)
        with pytest.raises(StorageLimitError):session.check_storage()
