from copy import deepcopy
import pytest
from superdj.native_control_recovery import refine_with_native_clock_recovery
from superdj.timing_safety import SourceWindowFailure


def scenario(monkeypatch, *, supported=True, fail_retry=False, change_events=False):
    calls=[]
    error=SourceWindowFailure('rate rejected', role='head', evidence={'passed':False})
    def refine(path, warp, rate, evidence):
        calls.append(deepcopy(evidence))
        if len(calls)==1:
            warp['source_kick_alignment']={'head':{
                'stable_native_primary_clock':supported, 'events':[{'time':1}],
                'native_rhythm_observations':[{'time':1.1}]}}
            raise error
        warp['anchors']=[{'source_frame':123}]
        if fail_retry:raise SourceWindowFailure('retry rejected')
        if change_events:warp['source_kick_alignment']['head']['events']=[]
        return warp['source_kick_alignment']
    monkeypatch.setattr('superdj.kick_grid.refine_source_controls',refine)
    return calls,error


@pytest.mark.parametrize('enabled,supported',[(False,True),(True,False)])
def test_default_or_unsupported_clock_cannot_retry(monkeypatch,enabled,supported):
    calls,error=scenario(monkeypatch,supported=supported)
    with pytest.raises(SourceWindowFailure) as caught:
        refine_with_native_clock_recovery('source',{},44100,{'head':{'path':'drums'}},enabled=enabled)
    assert caught.value is error and len(calls)==1


@pytest.mark.parametrize('fail_retry,change_events',[(True,False),(False,True)])
def test_failed_or_reference_changing_retry_cannot_replace_safe_state(monkeypatch,fail_retry,change_events):
    calls,error=scenario(monkeypatch,fail_retry=fail_retry,change_events=change_events)
    warp={'anchors':[{'source_frame':0}]}
    with pytest.raises(SourceWindowFailure) as caught:
        refine_with_native_clock_recovery('source',warp,44100,{'head':{'path':'drums'}},enabled=True)
    assert caught.value is error and warp['anchors']==[{'source_frame':0}]


def test_successful_retry_retains_observations_and_original_evidence(monkeypatch):
    calls,_=scenario(monkeypatch)
    warp={};evidence={'head':{'path':'drums'}}
    result=refine_with_native_clock_recovery('source',warp,44100,evidence,enabled=True)
    assert result['head']['events']==[{'time':1}]
    assert result['head']['native_rhythm_observations']==[{'time':1.1}]
    assert evidence=={'head':{'path':'drums'}}
    assert calls[1]['head']['stable_clock_controls']
    assert warp['supported_native_control_recovery']['timing_gates_unchanged']
