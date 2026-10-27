import numpy as np
import pytest
from superdj import orchestration as module
from superdj.transition_quality import TransitionFailure


def setup_call(tmp_path,monkeypatch,evidence=None,controls=None):
    a=np.arange(128,dtype='float32').reshape(64,2)/1000;b=a[::-1].copy()
    phase=np.linspace(0,1,len(a));error=TransitionFailure('a','b','quality refusal',evidence or{'energy':{'passed':False}})
    edge={'from':'a','to':'b','beats':128,'shared_clock':{'identity':'keep'},**(controls or{})}
    events=[];quality=lambda pcm,selected:{'passed':True}
    def legacy(*args,**kw):events.append('legacy');raise kw['initial_error']
    def drums():events.append('estimates');return {'drums_out':a*.2,'drums_in':b*.2},{'exact_timed_inputs':True}
    def gradual(*args,**kw):
        events.append('gradual');assert args[0]is a and args[1]is b and kw['beat_phase']is phase
        assert args[5]['shared_clock']is edge['shared_clock']
        assert kw['quality_check']is quality and kw['allow_pitch_changing_residual']is False
        return a+b,{**args[5],'quality_gate':quality(a+b,args[5]),'blended_front_audit':{'verified':True}}
    def whole(*args,**kw):events.append('whole');return a+b,{**args[5],'quality_gate':quality(a+b,args[5])}
    monkeypatch.setattr(module,'legacy_mid_alternative',legacy)
    monkeypatch.setattr(module,'gradual_alternative',gradual)
    monkeypatch.setattr(module,'whole_spectrum_alternative',whole)
    kwargs=dict(style='zhilin',beat_phase=phase,percussion_args={},initial_error=error,
        quality_check=quality,percussion_factory=drums)
    return a,b,edge,error,events,kwargs


@pytest.mark.parametrize('evidence',[{'energy':{'passed':False}},{'local_rhythm_coverage':{'passed':False}}])
def test_quality_refusal_tries_original_then_existing_soft_positions(tmp_path,monkeypatch,evidence):
    a,b,e,error,events,kw=setup_call(tmp_path,monkeypatch,evidence)
    original_a=a.copy();original_b=b.copy()
    pcm,selected=module.quality_alternative(a,b,44100,{}, {},e,tmp_path,**kw)
    assert events==['legacy','estimates','gradual']
    assert selected['quality_gate_recovery']['source_windows_clock_and_overlap_unchanged']
    assert selected['quality_gate']['passed']and selected['percussion_estimates']['exact_timed_inputs']
    np.testing.assert_array_equal(a,original_a);np.testing.assert_array_equal(b,original_b)


def test_successful_legacy_route_is_returned_without_new_estimates(tmp_path,monkeypatch):
    a,b,e,error,events,kw=setup_call(tmp_path,monkeypatch)
    expected=(a,{'already_passed':True})
    monkeypatch.setattr(module,'legacy_mid_alternative',lambda *args,**options:expected)
    assert module.quality_alternative(a,b,44100,{}, {},e,tmp_path,**kw)is expected
    assert events==[]


def test_whole_spectrum_variants_remain_after_gradual_failure(tmp_path,monkeypatch):
    a,b,e,error,events,kw=setup_call(tmp_path,monkeypatch)
    def fail(*args,**options):events.append('gradual');raise error
    monkeypatch.setattr(module,'gradual_alternative',fail)
    _,selected=module.quality_alternative(a,b,44100,{}, {},e,tmp_path,**kw)
    assert events==['legacy','estimates','gradual','whole']
    assert selected['quality_gate_recovery']['strategy']=='existing_whole_spectrum_variants'


@pytest.mark.parametrize('controls',[{'bass_handoff_beats':32},{'handoff_windows':{'bass':[8,40]}},
    {'whole_spectrum_handoff':{'start_beat':4,'end_beat':36}}])
def test_explicit_route_is_not_replaced_after_quality_refusal(tmp_path,monkeypatch,controls):
    a,b,e,error,events,kw=setup_call(tmp_path,monkeypatch,controls=controls)
    with pytest.raises(TransitionFailure)as result:module.quality_alternative(a,b,44100,{}, {},e,tmp_path,**kw)
    assert result.value is error and events==['legacy']


def test_foreground_only_refusal_keeps_previous_source_window_policy(tmp_path,monkeypatch):
    a,b,e,error,events,kw=setup_call(tmp_path,monkeypatch,{'energy':{'passed':True},'vocals':{'passed':False}})
    with pytest.raises(TransitionFailure):module.quality_alternative(a,b,44100,{}, {},e,tmp_path,**kw)
    assert events==['legacy']


def test_existing_exact_estimates_are_reused(tmp_path,monkeypatch):
    a,b,e,error,events,kw=setup_call(tmp_path,monkeypatch)
    kw['percussion_args']={'drums_out':a*.1,'drums_in':b*.1}
    module.quality_alternative(a,b,44100,{}, {},e,tmp_path,**kw)
    assert events==['legacy','gradual']


def test_failed_existing_variants_stay_failed(tmp_path,monkeypatch):
    a,b,e,error,events,kw=setup_call(tmp_path,monkeypatch)
    def fail(*args,**options):raise TransitionFailure('a','b','still not verified')
    monkeypatch.setattr(module,'gradual_alternative',fail);monkeypatch.setattr(module,'whole_spectrum_alternative',fail)
    with pytest.raises(TransitionFailure,match='still not verified'):
        module.quality_alternative(a,b,44100,{}, {},e,tmp_path,**kw)


def test_missing_one_final_estimate_is_not_accepted(tmp_path,monkeypatch):
    a,b,e,error,events,kw=setup_call(tmp_path,monkeypatch)
    kw['percussion_factory']=lambda:({'drums_out':a},{})
    with pytest.raises(ValueError,match='both exact'):
        module.quality_alternative(a,b,44100,{}, {},e,tmp_path,**kw)
    assert events==['legacy']
