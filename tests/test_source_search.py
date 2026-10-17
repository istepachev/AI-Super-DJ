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
