"""A known bad coverage view must not trigger edits to its good neighbour."""
from copy import deepcopy
import numpy as np
import pytest

from superdj.transition_quality import (TransitionFailure,coverage_failure_context,verify_overlap)
from superdj.recovery import alternative
from test_reliability import source_plan


def report(outgoing=False,incoming=True):
    return {'passed':False,'energy':{'passed':True},'vocals':{'passed':True},
        'reliability':{'passed':outgoing and incoming,'decks':[
            {'id':'0','role':'tail','sufficient_for_automatic_window':outgoing,'recognized_beats':20,'unknown_beats':107},
            {'id':'1','role':'head','sufficient_for_automatic_window':incoming,'recognized_beats':49,'unknown_beats':78}]}}


@pytest.mark.parametrize('outgoing,incoming,identity,role',[(False,True,'0','tail'),(True,False,'1','head')])
def test_a_single_insufficient_source_view_retains_its_owner_and_role(outgoing,incoming,identity,role):
    metrics=report(outgoing,incoming);before=deepcopy(metrics)
    context=coverage_failure_context(metrics,{'from':'0','to':'1'})
    assert context['failed_track']==identity and context['failed_roles']==[role]
    assert metrics==before


@pytest.mark.parametrize('outgoing,incoming',[(False,False),(True,True)])
def test_two_failed_views_or_unrelated_quality_errors_do_not_invent_a_single_owner(outgoing,incoming):
    metrics=report(outgoing,incoming)
    assert coverage_failure_context(metrics,{'from':'0','to':'1'})=={}


def test_verify_overlap_passes_coverage_owner_to_real_window_recovery(monkeypatch):
    import superdj.transition_quality as module
    metrics=report()
    monkeypatch.setattr(module,'audit_energy',lambda *a:deepcopy(metrics['energy']))
    monkeypatch.setattr(module,'audit_vocals',lambda *a,**k:deepcopy(metrics['vocals']))
    monkeypatch.setattr(module,'reliability',lambda *a:deepcopy(metrics['reliability']))
    rows,plan=source_plan(2);before=deepcopy(plan);edge=plan['edges'][0]
    audio=np.zeros((100,2))
    with pytest.raises(TransitionFailure) as captured:
        verify_overlap(audio,audio,audio,48000,[],edge)
    error=captured.value
    assert error.evidence['reliability']==metrics['reliability']
    assert error.evidence['failed_track']=='0'
    revised,selection=alternative(plan,rows,error,{},1,sample_rate=48000,ffmpeg=None)
    assert selection['changed_track']=='0' and selection['boundary']=='end'
    assert revised['sections']['1']['start']==before['sections']['1']['start']
    assert revised['sections']['1']['end']==before['sections']['1']['end']
    assert plan==before and revised['order']==plan['order']
    assert [e['overlap_bars'] for e in revised['edges']]==[e['overlap_bars'] for e in plan['edges']]


def test_inconsistent_source_role_is_not_used_as_a_recovery_target():
    metrics=report();metrics['reliability']['decks'][0]['role']='head'
    assert coverage_failure_context(metrics,{'from':'0','to':'1'})=={}
