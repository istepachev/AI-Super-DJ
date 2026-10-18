"""Coarse beat labels must not produce an audible slow/fast timing pulse."""
from copy import deepcopy
import numpy as np
import pytest

from superdj.timing_safety import audit_warp_rates, classify_native_phase, initialize_rate_reference
from superdj.sync import shared_clock


def test_audio_consensus_recovers_a_true_beat_behind_a_bad_coarse_label():
    centres=.4+np.arange(128)*60/113
    centres[121]-=.06325
    references=[{'index':i,'time':float(.4+i*60/113+.003*np.sin(i))}
                for i in range(128) if i not in (80,121)]
    true_beat={'index':121,'time':float(.4+121*60/113+.00117)}
    native_fill={'index':80,'time':float(.4+80*60/113+.10)}
    rows=deepcopy(references+[true_beat,native_fill])
    accepted,displaced=classify_native_phase(rows,centres,references)
    assert 121 in {r['index'] for r in accepted}
    assert 80 not in {r['index'] for r in accepted}
    assert next(r for r in accepted if r['index']==121)['time']==true_beat['time']
    assert next(r for r in displaced if r['index']==80)['time']==native_fill['time']
    assert rows==references+[true_beat,native_fill]


def test_unrecognized_beat_uses_observed_source_time_instead_of_bad_grid_offsets(tmp_path):
    from test_kick_grid import fixture
    from superdj.kick_grid import refine_source_controls
    path,warp,sr=fixture(tmp_path,missing=12)
    mapping=warp['input_output_beat_mapping']['head']
    mapping[12]['source_seconds']-=.065
    for anchor in warp['anchors']:
        if anchor['role']=='head_sync_control':
            index=(anchor['output_seconds']-.3)/.5
            source=float(np.interp(index,np.arange(len(mapping)),[r['source_seconds'] for r in mapping]))
            anchor['source_seconds']=source
            anchor['source_frame']=round(source*sr)
    clock=shared_clock(120,120,len(mapping)-1,sr)
    initialize_rate_reference(warp,sr,native_bpm=60/.503,windows=[('head',clock)])
    report=refine_source_controls(path,warp,sr)['head']
    assert 12 not in {e['beat'] for e in report['events']}
    point=min(warp['anchors'],key=lambda p:abs(p['output_seconds']-6.3))
    neighbours={e['beat']:e['source_seconds'] for e in report['events']}
    expected=(neighbours[11]+neighbours[13])/2
    assert point['source_frame']/sr==pytest.approx(expected,abs=2/sr)
    audit=audit_warp_rates(warp,sr,stage='corrected missing-beat controls')
    assert audit['short_term_timing_continuity']['passed']


@pytest.mark.parametrize('dense',[False,True],ids=['reported-frame-slopes','densified-controls'])
def test_sub_fifteen_percent_slow_fast_pulses_fail_even_with_correct_duration(dense):
    sr=44100
    if dense:
        rates=np.r_[np.linspace(1,.865,30),np.linspace(.865,1.135,60),np.linspace(1.135,1,30)]
        output=np.arange(len(rates)+1)*441
        source=np.r_[0,np.cumsum(rates*441)].round().astype(int)
    else:
        output=np.arange(4)*sr
        source=np.array([0,38132,87980,132300])
    warp={'anchors':[{'source_frame':int(t),'output_frame':int(t)} for t in output],
          'input_output_beat_mapping':{'head':[],'tail':[]}}
    initialize_rate_reference(warp,sr,limit=.15)
    for anchor,frame in zip(warp['anchors'],source):anchor['source_frame']=int(frame)
    with pytest.raises(ValueError,match='Short-term acoustic timing'):
        audit_warp_rates(warp,sr,stage='reported sub-limit rate pulse')
    audit=warp['actual_timing_rate_audit']
    assert audit['maximum_timing_correction_change']<.15
    assert not audit['short_term_timing_continuity']['passed']
    assert audit['short_term_timing_continuity']['maximum_correction_rate_span']>.25


def test_a_smooth_musical_tempo_change_is_not_a_correction_speed_pulse():
    sr=44100;clock=shared_clock(90,130,128,sr)
    source=np.rint(np.asarray(clock['phase'])*60/110*sr).astype(int)
    warp={'anchors':[{'source_frame':int(a),'output_frame':int(b)} for a,b in zip(source,clock['output_frames'])],
          'input_output_beat_mapping':{'head':[{'output_seconds':0.},{'output_seconds':clock['duration_seconds']}],'tail':[]}}
    before=deepcopy(warp['anchors'])
    initialize_rate_reference(warp,sr,native_bpm=110,windows=[('head',clock)])
    audit=audit_warp_rates(warp,sr,stage='deliberate smooth musical curve')
    assert audit['short_term_timing_continuity']['passed']
    assert warp['anchors']==before


def test_rate_recovery_rejects_native_positions_only_on_the_failed_side():
    from superdj.recovery import unsupported_attack_events
    evidence={'failed_role':'head','source_timing_evidence':{'short_term_timing_continuity':{
        'passed':False,'worst_output_seconds':[65.17,65.70],'worst_source_seconds':[93.997,94.455]}}}
    before=deepcopy(evidence)
    assert unsupported_attack_events(evidence,'head')==[93.997,94.455]
    assert unsupported_attack_events(evidence,'tail')==[]
    evidence['canonical_attack_rejections']={'head':[93.997,95.053]}
    assert unsupported_attack_events(evidence,'head')==[93.997,95.053,94.455]
    assert evidence['source_timing_evidence']==before['source_timing_evidence']


def test_source_preflight_single_role_reaches_the_first_window_retry(monkeypatch):
    from superdj import recovery
    from superdj.transition_quality import TransitionFailure
    from test_reliability import source_plan
    records,plan=source_plan()
    source_time=plan['sections']['1']['start']+2
    evidence={'stage':'source-only timing preflight','failed_track':'1','failed_role':'head',
        'source_timing_evidence':{'short_term_timing_continuity':{
            'passed':False,'worst_source_seconds':[source_time]}}}
    visited=[]
    def solver(proposal,*args,**kwargs):
        assert proposal['rejected_attack_window_events']['1']['head']==[source_time]
        visited.append(True)
        updated=deepcopy(plan)
        updated['sections']['1']['start']+=2
        return updated
    monkeypatch.setattr(recovery,'arrange_sync',solver)
    updated,_=recovery.alternative(plan,records,TransitionFailure('0','1','pulse',evidence),{},1,
                                   sample_rate=48000,ffmpeg=None)
    assert visited
    assert updated['sections']['0']==plan['sections']['0']
