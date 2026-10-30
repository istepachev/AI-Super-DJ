from pathlib import Path
import pytest
from superdj.execution import execution,cached
from superdj.render_index import enable,RenderSearchBudgetExceeded
from superdj.render_index import indexed_compensate
from superdj.render import SourceWindowError
from test_render_index import deps


def test_recovery_clock_survives_restart_and_stops_only_additional_dsp(tmp_path,monkeypatch):
    clock=[0.];monkeypatch.setattr('superdj.render_index.time.monotonic',lambda:clock[0])
    path=tmp_path/'a';path.write_bytes(b'a')
    for i,cost in enumerate((.1,2.)):
        with execution(tmp_path/'cache')as session:
            enable(session,'clock',max_group_candidates=10,recovery_seconds=1.)
            def work():clock[0]+=cost;return {'verified':True}
            cached('clip',deps(path,start=float(i)),work,session.work)
    with execution(tmp_path/'cache')as session:
        enable(session,'clock',max_group_candidates=10,recovery_seconds=1.)
        with pytest.raises(RenderSearchBudgetExceeded):
            cached('clip',deps(path,start=2.),lambda:pytest.fail('Time-exhausted DSP must not run'),session.work)


def test_programme_extra_budget_preserves_first_attempt_for_other_sources(tmp_path):
    paths=[]
    for name in 'abc':
        p=tmp_path/name;p.write_bytes(name.encode());paths.append(p)
    with execution(tmp_path/'cache')as session:
        enable(session,'programme',max_extra_candidates=1)
        for i in (0,1):cached('clip',deps(paths[i],identity=str(i)),lambda:{'ok':True},session.work)
        cached('clip',deps(paths[0],identity='0',start=1.),lambda:{'ok':True},session.work)
        with pytest.raises(RenderSearchBudgetExceeded):
            cached('clip',deps(paths[1],identity='1',start=1.),lambda:pytest.fail('Extra budget must be shared'),session.work)
        assert cached('clip',deps(paths[2],identity='2'),lambda:{'ok':True},session.work)['ok']


def test_temporary_index_keeps_failures_bounded_without_persistent_storage(tmp_path):
    path=tmp_path/'source';path.write_bytes(b'source')
    with execution()as session:
        root=session.root;index=enable(session,'small RAM',max_group_candidates=2)
        index.memory_limit=1024
        for _ in range(2):
            with pytest.raises(SourceWindowError):
                cached('clip',deps(path),lambda:(_ for _ in()).throw(SourceWindowError('a',['head'],'unrecognized',{'large':'x'*2048})),session.work)
        assert index.memory_bytes<=1024
        with pytest.raises(RenderSearchBudgetExceeded):
            cached('clip',deps(path),lambda:pytest.fail('Eviction must not reset the attempt budget'),session.work)
    assert not root.exists()


def test_backend_corrections_are_bounded_without_accepting_an_unverified_render(tmp_path):
    path=tmp_path/'source';path.write_bytes(b'source');calls=[]
    def correction():calls.append(1)
    with execution()as session:
        enable(session,'corrections',max_backend_corrections=2)
        def produce():
            indexed_compensate(correction);indexed_compensate(correction)
            with pytest.raises(ValueError,match='budget exhausted'):indexed_compensate(correction)
            raise SourceWindowError('a',['head'],'actual audit still refuses',{'verified':False})
        with pytest.raises(SourceWindowError,match='actual audit still refuses'):
            cached('clip',deps(path),produce,session.work)
    assert len(calls)==2


@pytest.mark.parametrize('outcomes',[(False,False),(False,True),(True,False)])
def test_rejected_map_proposals_do_not_consume_the_stretch_budget(tmp_path,outcomes):
    path=tmp_path/'source';path.write_bytes(b'source');calls=[]
    def correction(passed):
        calls.append(passed)
        if not passed:raise ValueError('backend rejected the proposed map')
        return 'measured candidate'
    with execution()as session:
        enable(session,'failed corrections',max_backend_corrections=2)
        def produce():
            for passed in outcomes:
                if passed:assert indexed_compensate(correction,True)=='measured candidate'
                else:
                    with pytest.raises(ValueError,match='backend rejected'):
                        indexed_compensate(correction,False)
            for _ in range(2-sum(outcomes)):
                assert indexed_compensate(correction,True)=='measured candidate'
            with pytest.raises(ValueError,match='budget exhausted'):
                indexed_compensate(correction,True)
            raise SourceWindowError('a',['head'],'actual audit still refuses',{'verified':False})
        with pytest.raises(SourceWindowError,match='actual audit still refuses'):
            cached('clip',deps(path),produce,session.work)
    assert calls==[*outcomes,*([True]*(2-sum(outcomes)))]
    # Leaving the failed indexed producer must restore the ordinary API context.
    assert indexed_compensate(correction,True)=='measured candidate'
    assert calls==[*outcomes,*([True]*(3-sum(outcomes)))]


def test_changed_neighbour_does_not_exhaust_a_working_deck_but_shares_extra_budget(tmp_path):
    path=tmp_path/'source';path.write_bytes(b'source');calls=[]
    def item(clock):
        value=deps(path);value['next']={'from':'a','to':'b','target_bpm':clock};return value
    def produce():calls.append(1);return {'verified':True}
    for identity,clock in [('old neighbour',120.),('new neighbour',121.)]:
        with execution(tmp_path/'cache')as session:
            enable(session,'neighbours',max_group_candidates=1,max_extra_candidates=1,
                source_identities={'b':identity})
            assert cached('clip',item(clock),produce,session.work)['verified']
    assert len(calls)==2
    with execution(tmp_path/'cache')as session:
        enable(session,'neighbours',max_group_candidates=1,max_extra_candidates=1,
            source_identities={'b':'new neighbour'})
        with pytest.raises(RenderSearchBudgetExceeded):
            cached('clip',item(122.),lambda:pytest.fail('Same-neighbour search stays bounded'),session.work)
    with execution(tmp_path/'cache')as session:
        enable(session,'neighbours',max_group_candidates=1,max_extra_candidates=1,
            source_identities={'b':'third neighbour'})
        with pytest.raises(RenderSearchBudgetExceeded):
            cached('clip',item(123.),lambda:pytest.fail('Changing neighbours cannot bypass the programme budget'),session.work)
    assert len(calls)==2


def test_metadata_lru_eviction_during_artifact_check_is_safe(tmp_path,monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import superdj.execution as module
    with execution()as session:
        index=enable(session,'concurrent LRU');artifact=session.work/'artifact';artifact.write_bytes(b'checked PCM')
        value={'path':str(artifact)};index._remember('a',value)
        entered=threading.Event();release=threading.Event();original=module.sha256
        def digest(path):
            entered.set();assert release.wait(5);return original(path)
        monkeypatch.setattr(module,'sha256',digest)
        with ThreadPoolExecutor(max_workers=1)as pool:
            future=pool.submit(index._memory_result,'a');assert entered.wait(5)
            index.result_limit=1;index._remember('b',0,metadata_only=True)
            assert 'a'not in index.results
            release.set();assert future.result()==value
        assert index.result_bytes==sum(item['bytes']for item in index.results.values())
