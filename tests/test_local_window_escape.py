from copy import deepcopy
import pytest
from test_reliability import source_plan
from superdj import compact,source_search
from superdj.recovery import alternative
from superdj.render import RenderError
from superdj.transition_quality import TransitionFailure


def test_local_radius_refusal_can_use_existing_wide_native_search(monkeypatch):
    rows,plan=source_plan();before=deepcopy(plan);calls=[]
    real=compact.arrange_sync
    def local_refused(*args,**kwargs):
        calls.append('local')
        raise RenderError('No window fits the local radius')
    # The native-wide route uses its independently imported original solver.
    assert source_search.arrange_sync is real
    monkeypatch.setattr(compact,'arrange_sync',local_refused)
    failure=TransitionFailure('0','1','mandatory stretched head hit not recognized',
        {'failed_track':'1','failed_roles':['head']})
    updated,report=alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert calls and plan==before and updated['order']==plan['order']
    assert updated['sections']['1']['start']!=before['sections']['1']['start']
    assert updated['sections']['1']['end']==before['sections']['1']['end']
    assert updated['sections']['0']==before['sections']['0']
    assert updated['sections']['2']==before['sections']['2']
    assert report['same_failed_boundary_cannot_be_retried']
    assert [e['overlap_bars']for e in updated['edges']]==[e['overlap_bars']for e in plan['edges']]


def test_local_escape_preserves_user_lock(monkeypatch):
    rows,plan=source_plan();plan['sections']['1']['locked_start']=True;before=deepcopy(plan)
    failure=TransitionFailure('0','1','head refused',{'failed_track':'1','failed_roles':['head']})
    with pytest.raises(RenderError,match='locked'):
        alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert plan==before


def test_successful_local_window_does_not_call_wider_search(monkeypatch):
    rows,plan=source_plan()
    def unexpected(*args,**kwargs):raise AssertionError('Passing local search must remain unchanged')
    monkeypatch.setattr(source_search,'fresh_arrangement_alternative',unexpected)
    failure=TransitionFailure('0','1','head refused',{'failed_track':'1','failed_roles':['head']})
    updated,_=alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert updated['sections']['1']['start']!=plan['sections']['1']['start']
