from copy import deepcopy
import pytest
from test_reliability import source_plan
from superdj import execution
from superdj.recovery import render_with_recovery
from superdj.transition_quality import TransitionFailure


@pytest.fixture(autouse=True)
def indexed_session(tmp_path):
    from superdj.render_index import enable
    with execution.execution(scratch_root=tmp_path/'scratch') as session:
        enable(session,'bounded-recovery-test',track_count=4)
        yield session


def indexed(monkeypatch):
    return execution.current()


def test_distinct_failed_pairs_do_not_share_two_window_alternatives(monkeypatch,tmp_path):
    indexed(monkeypatch);rows,plan=source_plan(4);calls=[]
    pairs=[('0','1'),('1','2'),('2','3')]
    def render(p,*args,**kwargs):
        calls.append(deepcopy(p))
        if len(calls)<=3:raise TransitionFailure(*pairs[len(calls)-1],'local rhythm coverage')
        return {'passed':True}
    result,effective=render_with_recovery(plan,rows,{},tmp_path/'out.wav',tmp_path,renderer=render,sample_rate=48000)
    assert result['passed'] and len(calls)==4 and effective['order']==plan['order']
    assert result['window_recovery']['shared_render_index_budget_retained']


def test_repeated_pair_still_stops_after_two_window_alternatives(monkeypatch,tmp_path):
    indexed(monkeypatch);rows,plan=source_plan(4);calls=[];error=TransitionFailure('0','1','still unverified')
    def render(*args,**kwargs):calls.append(1);raise error
    with pytest.raises(TransitionFailure)as caught:
        render_with_recovery(plan,rows,{},tmp_path/'out.wav',tmp_path,renderer=render,sample_rate=48000)
    assert caught.value is error and len(calls)==3


def profile_failure():
    return TransitionFailure('0','1','stretched attacks refused',{'failed_track':'1','failed_roles':['head','tail'],
        'warp':{'rubberband_engine':'r2','rubberband_options':['--fast','--detector-perc']}})


def test_default_output_refusal_tries_short_on_identical_cues_without_mutating_inputs(monkeypatch,tmp_path):
    indexed(monkeypatch);rows,plan=source_plan(3);before=deepcopy(rows);calls=[]
    plan['sections']['1']['locked_start']=plan['sections']['1']['locked_end']=True
    def render(p,r,*args,**kwargs):
        calls.append(deepcopy(r))
        assert p==plan
        if len(calls)==1:raise profile_failure()
        assert r[1]['rubberband_profile_override']=='short'
        return {'passed':True}
    result,_=render_with_recovery(plan,rows,{},tmp_path/'out.wav',tmp_path,renderer=render,sample_rate=48000)
    assert result['passed'] and len(calls)==2 and rows==before


def test_explicit_profile_is_never_replaced_by_short(monkeypatch,tmp_path):
    indexed(monkeypatch);rows,plan=source_plan(3);rows[1]['rubberband_profile_override']='crisp6'
    plan['sections']['1']['locked_start']=plan['sections']['1']['locked_end']=True
    calls=[]
    def render(p,r,*args,**kwargs):calls.append(r[1]['rubberband_profile_override']);raise profile_failure()
    with pytest.raises(TransitionFailure):render_with_recovery(plan,rows,{},tmp_path/'out.wav',tmp_path,renderer=render,sample_rate=48000)
    assert calls==['crisp6']


def test_fixed_single_attempt_remains_fixed_with_an_index(monkeypatch,tmp_path):
    indexed(monkeypatch);rows,plan=source_plan(3);calls=[]
    def render(*args,**kwargs):calls.append(1);raise profile_failure()
    with pytest.raises(TransitionFailure):render_with_recovery(plan,rows,{},tmp_path/'out.wav',tmp_path,renderer=render,maximum_attempts=1,sample_rate=48000)
    assert calls==[1]


def test_passing_renderer_retains_original_record_object_and_does_no_recovery(monkeypatch,tmp_path):
    indexed(monkeypatch);rows,plan=source_plan(3);calls=[]
    def render(p,r,*args,**kwargs):assert r is rows;calls.append(1);return {'pcm_digest':'unchanged'}
    result,effective=render_with_recovery(plan,rows,{},tmp_path/'out.wav',tmp_path,renderer=render,sample_rate=48000)
    assert result['pcm_digest']=='unchanged' and calls==[1] and effective==plan
