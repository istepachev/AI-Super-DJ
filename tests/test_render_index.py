from copy import deepcopy
from pathlib import Path
import json,os,threading,time
import numpy as np
import pytest
import soundfile as sf
from superdj.execution import execution,cached,key_for
from superdj.render_index import enable,clip_dependencies,cached_phrase_repeat,RenderSearchBudgetExceeded
from superdj.render_index import indexed_source_call
from superdj.render import SourceWindowError
from superdj.transition_quality import TransitionFailure


def deps(path,*,tag='tag',start=0.,identity='a'):
    return {'record':{'id':identity,'path':str(path),'render_evidence_timeline_signature':tag,
                      'vocal_stem':{'profile':[{'time':start,'foreground_risk':.1}]}},
            'section':{'start':start,'end':10.},'controls':{'sample_rate':8000}}


def test_38_path_only_tags_reuse_one_verified_pcm_and_changed_source_does_not(tmp_path):
    source=tmp_path/'native';source.write_bytes(b'native source');calls=[]
    with execution(tmp_path/'cache')as session:
        enable(session,'programme',track_count=2)
        def produce():
            calls.append(1);out=session.work/'clip.wav';out.write_bytes(b'exact PCM')
            return {'path':str(out),'verified':True}
        paths=[cached('clip',deps(source,tag=str(i)),produce,session.work)['path']for i in range(38)]
        assert len(calls)==1 and len(set(paths))==1
        assert Path(paths[0]).read_bytes()==b'exact PCM'
        source.write_bytes(b'changed source')
        cached('clip',deps(source,tag='38'),produce,session.work)
        assert len(calls)==2


def test_actual_rebased_feature_changes_and_input_metadata_are_retained(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'source')
    a=deps(source);before=deepcopy(a);b=deps(source,tag='other',start=1.)
    assert key_for(clip_dependencies(a))!=key_for(clip_dependencies(b))
    assert a==before and a['record']['render_evidence_timeline_signature']=='tag'


@pytest.mark.parametrize('kind',['source','transition'])
def test_exact_rejections_are_preflighted_across_restarts_without_rerendering(tmp_path,kind):
    source=tmp_path/'source';source.write_bytes(b'source');calls=[]
    for turn in range(2):
        with execution(tmp_path/'cache')as session:
            index=enable(session,'same programme')
            def fail():
                calls.append(1)
                if kind=='source':raise SourceWindowError('a',['head'],'late',{'delay':.004})
                raise TransitionFailure('a','b','late',{'delay':.004})
            expected=SourceWindowError if kind=='source'else TransitionFailure
            with pytest.raises(expected)as error:cached('clip',deps(source),fail,session.work)
            assert error.value.evidence['delay']==.004
            if turn:assert error.value.evidence['render_index_replayed_rejection']
    assert calls==[1]


def test_changed_windows_consume_persistent_budget_but_successful_cache_hits_do_not(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'source')
    for turn in range(2):
        with execution(tmp_path/'cache')as session:
            enable(session,'programme',max_group_candidates=2)
            def ok():
                out=session.work/f'clip{turn}.wav';out.write_bytes(b'PCM');return {'path':str(out)}
            cached('clip',deps(source,start=float(turn)),ok,session.work)
    with execution(tmp_path/'cache')as session:
        enable(session,'programme',max_group_candidates=2)
        with pytest.raises(RenderSearchBudgetExceeded)as error:
            cached('clip',deps(source,start=2.),lambda:pytest.fail('Third DSP trial must not run'),session.work)
        assert error.value.info['source_ids']==['a']
        cached('clip',deps(source,start=0.),lambda:pytest.fail('Existing valid PCM must be reused'),session.work)


def test_accepted_index_does_not_approve_corrupt_audio_and_io_errors_are_not_memoized(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'source');calls=[]
    with execution(tmp_path/'cache')as session:
        enable(session,'programme')
        def good():
            calls.append(1);p=session.work/'audio';p.write_bytes(b'good PCM');return {'path':str(p)}
        value=cached('clip',deps(source),good,session.work)
        Path(value['path']).write_bytes(b'bad PCM!')
        assert Path(cached('clip',deps(source),good,session.work)['path']).read_bytes()==b'good PCM'
        assert len(calls)==2
        for _ in range(2):
            with pytest.raises(OSError):cached('clip',deps(source,start=1.),lambda:(_ for _ in()).throw(OSError('disk')),session.work)
        assert session.render_index.metrics['replayed_rejections']==0


def test_no_cache_index_is_memory_only_and_independent_jobs_remain_parallel(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'source');active=maximum=0;lock=threading.Lock();barrier=threading.Barrier(2)
    with execution(jobs=2,cpu_jobs=2)as session:
        index=enable(session,'temporary');assert index.root is None
        def one(i):
            def produce():
                nonlocal active,maximum
                with lock:active+=1;maximum=max(maximum,active)
                barrier.wait(timeout=5)
                with lock:active-=1
                return {'value':i}
            return cached('clip',deps(source,identity=str(i)),produce,session.work)
        assert session.map(one,[1,2])==[{'value':1},{'value':2}]
        assert maximum==2


def test_cached_phrase_repeat_is_bit_exact_and_context_paths_are_canonical(tmp_path,monkeypatch):
    import superdj.loops as loops
    sr=8000;t=np.arange(sr*14)/sr
    pcm=np.column_stack([.07*np.sin(2*np.pi*80*t)+.03*np.sin(2*np.pi*400*t)]*2).astype('float32')
    source=tmp_path/'native.wav';sf.write(source,pcm,sr,subtype='FLOAT')
    edit={'method':'verified phrase repeat with complementary-band join','loop_start_seconds':4.,
          'insertion_source_seconds':8.,'loop_seconds':4.,'repeat_count':2,'added_seconds':8.,
          'bars_repeated':2,'join_frames':sr,'beat_seconds':.5,'sample_rate':sr,
          'original_source_duration_seconds':14.,'aligned_preparation_streams':['path']}
    loops.repeat_file(source,tmp_path/'baseline.wav',edit,context_path=tmp_path/'baseline.npy')
    original=loops.repeat_file;calls=[]
    def counted(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(loops,'repeat_file',counted)
    contexts=[]
    for i in range(2):
        with execution(tmp_path/'cache')as session:
            folder=session.work/str(i);folder.mkdir()
            out=folder/'repeat.wav'
            repeated,context=cached_phrase_repeat(source,out,edit,context_path=folder/'join.npy');contexts.append(context)
            a,ar=sf.read(repeated,dtype='float32',always_2d=True);b,br=sf.read(tmp_path/'baseline.wav',dtype='float32',always_2d=True)
            assert ar==br and a.tobytes()==b.tobytes()
            if i:assert not out.exists(),'Reusing PCM must not rewrite the full repeated track'
    assert len(calls)==1 and contexts[0]==contexts[1]


def test_real_indexed_render_retains_all_pcm_and_reuses_verified_prefix(tmp_path):
    from test_execution import _tools
    from test_sync import RenderedSyncTests
    from superdj.render import render_plan
    from superdj.planner import plan_tracks
    ffmpeg,rubber=_tools();rows=RenderedSyncTests().fixtures(tmp_path,[120,122,124])
    plan=plan_tracks(rows,mode='fixed');options=dict(ffmpeg=ffmpeg,rubberband=rubber,sample_rate=16000,tempo_conflict_policy='sync')
    render_plan(plan,rows,{},tmp_path/'baseline.wav',tmp_path/'baseline',**options)
    for i in range(2):
        with execution(tmp_path/'cache')as session:
            index=enable(session,'real programme',track_count=3)
            render_plan(plan,rows,{},tmp_path/f'indexed{i}.wav',session.work,**options)
            a,ar=sf.read(tmp_path/f'indexed{i}.wav',dtype='float32',always_2d=True);b,br=sf.read(tmp_path/'baseline.wav',dtype='float32',always_2d=True)
            assert ar==br and a.tobytes()==b.tobytes()
            if i:assert index.metrics['fresh_candidates']==0


@pytest.mark.parametrize('persistent',[False,True])
def test_native_evidence_is_indexed_before_stretch_and_changed_map_invalidates_it(tmp_path,persistent):
    path=tmp_path/'native.wav';sf.write(path,np.zeros((8000,2)),8000,subtype='FLOAT');calls=[]
    def native(path,warp,rate):
        calls.append(1);warp['measured_native_events']=[.5,1.]
        return {'native':True,'actual_blend_verified':False}
    with execution(tmp_path/'cache'if persistent else None)as session:
        enable(session,'native')
        for _ in range(3):
            warp={'anchors':[1.,2.]}
            result=indexed_source_call('source_controls',native,path,warp,8000)
            assert result['native']and not result['actual_blend_verified']
            assert warp['measured_native_events']==[.5,1.]
        assert len(calls)==1
        indexed_source_call('source_controls',native,path,{'anchors':[2.,3.]},8000)
        assert len(calls)==2
