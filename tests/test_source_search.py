from copy import deepcopy
from pathlib import Path
import sys
import pytest

from test_reliability import source_plan
from superdj.transition_quality import TransitionFailure
from superdj.render import RenderError
from superdj.source_search import fresh_arrangement_alternative as fresh_section


def test_failed_exit_retries_distinct_exits_without_shuffling_intros():
    rows,plan=source_plan()
    before=deepcopy(plan);ends=[];excluded={}
    for n in range(4):
        ends.append(plan['sections']['0']['end'])
        error=TransitionFailure('0','1','unresolved',{'failed_track':'0','failed_roles':['tail']})
        plan,report=fresh_section(plan,rows,error,excluded,n+1,sample_rate=48000,ffmpeg=None)
        assert plan['sections']['0']['start']==before['sections']['0']['start']
        assert plan['sections']['0']['end'] not in ends
        assert plan['sections']['1']==before['sections']['1']
        assert plan['sections']['2']==before['sections']['2']
        assert report['same_failed_boundary_cannot_be_retried']


def test_failed_entrance_keeps_exit_and_neighbours():
    rows,plan=source_plan();before=deepcopy(plan)
    error=TransitionFailure('0','1','unresolved',{'failed_track':'1','failed_roles':['head']})
    updated,_=fresh_section(plan,rows,error,{},1,sample_rate=48000,ffmpeg=None)
    assert updated['sections']['1']['start']!=before['sections']['1']['start']
    assert updated['sections']['1']['end']==before['sections']['1']['end']
    assert updated['sections']['0']==before['sections']['0']


def test_user_locked_bad_exit_cannot_be_replaced_by_an_intro_change():
    rows,plan=source_plan();plan['sections']['0']['locked_end']=True
    error=TransitionFailure('0','1','unresolved',{'failed_track':'0','failed_roles':['tail']})
    with pytest.raises(RenderError,match='locked'):
        fresh_section(plan,rows,error,{},1,sample_rate=48000,ffmpeg=None)


def test_two_failed_ends_cannot_hide_one_user_locked_failure():
    rows,plan=source_plan();plan['sections']['1']['locked_end']=True
    error=TransitionFailure('0','1','unresolved',{'failed_track':'1','failed_roles':['head','tail']})
    with pytest.raises(RenderError,match='locked'):
        fresh_section(plan,rows,error,{},1,sample_rate=48000,ffmpeg=None)


def test_native_timing_refusal_survives_recovery_of_the_other_source_edge():
    rows,plan=source_plan();before=deepcopy(plan);excluded={}
    old_end=plan['sections']['1']['end'];points=[old_end-8.,old_end-6.]
    error=TransitionFailure('1','2','native timing variation',{
        'failed_track':'1','failed_role':'tail','stage':'source-only timing preflight',
        'source_timing_evidence':{'short_term_timing_continuity':{
            'passed':False,'worst_source_seconds':points}}})
    changed,_=fresh_section(plan,rows,error,excluded,1,sample_rate=48000,ffmpeg=None)
    assert changed['rejected_attack_window_events']['1']['tail']==points
    head_error=TransitionFailure('0','1','local entry not recognized',{
        'failed_track':'1','failed_roles':['head']})
    updated,_=fresh_section(changed,rows,head_error,excluded,2,sample_rate=48000,ffmpeg=None)
    assert plan==before
    assert updated['rejected_attack_window_events']['1']['tail']==points
    outgoing=next(e for e in updated['edges']if e['from']=='1')
    period=60/rows[1]['bpm'];ending=updated['sections']['1']['end']
    assert not any(ending-outgoing['overlap_bars']*4*period<=t<=ending for t in points)
    assert updated['order']==plan['order']
    assert [e['overlap_bars']for e in updated['edges']]==[e['overlap_bars']for e in plan['edges']]


def test_native_timing_positions_belong_to_the_reported_role_only():
    from superdj.recovery import unsupported_attack_events
    evidence={'failed_role':'tail','source_timing_evidence':{
        'short_term_timing_continuity':{'passed':False,'worst_source_seconds':[337.377,337.873]}}}
    assert unsupported_attack_events(evidence,'tail')==[337.377,337.873]
    assert unsupported_attack_events(evidence,'head')==[]


def test_explicit_single_attempt_profile_trial_cannot_move_source_windows(tmp_path,monkeypatch):
    import superdj.recovery as recovery
    rows,plan=source_plan();before=deepcopy(plan);seen=[]
    failure=TransitionFailure('0','1','profile did not pass')
    def refused(candidate,*args,**kwargs):
        seen.append(deepcopy(candidate));raise failure
    def unexpected_search(*args,**kwargs):
        raise AssertionError('A fixed-window profile trial must not search other cues')
    monkeypatch.setattr(recovery,'alternative',unexpected_search)
    with pytest.raises(TransitionFailure)as caught:
        recovery.render_with_recovery(plan,rows,{},tmp_path/'out.wav',tmp_path/'work',
            renderer=refused,maximum_attempts=1)
    assert caught.value is failure
    assert seen==[before]and plan==before
