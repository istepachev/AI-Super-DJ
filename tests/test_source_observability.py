"""Unknown native candidates cannot become primary clocks; known errors survive."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest
import soundfile as sf

from superdj.source_observability import bass_candidate_support
from superdj.timing_safety import SourceWindowFailure
from superdj.percussion import needs_handoff,ownership_policy
from superdj.attack_references import retain_mixture_references
from superdj.kick_grid import rendered_attack_audit
from test_attack_reference_signal import fixture


def test_estimated_kick_hidden_by_continuous_native_tone_remains_unknown(tmp_path):
    sr,source,warp=fixture(tmp_path,bed=0.)
    drums,_=sf.read(source,dtype='float32',always_2d=True)
    t=np.arange(len(drums))/sr
    native=np.column_stack([.3*np.sin(2*np.pi*80*t)]*2).astype('float32')
    sf.write(source,native,sr,subtype='FLOAT');before=hashlib.sha256(source.read_bytes()).hexdigest()
    candidates=[{'index':i,'time':float(.5+i*.5)} for i in (4,9,14)]
    accepted,unknown,proof=bass_candidate_support(source,warp,candidates,drums,sr,0.,role='head')
    assert not accepted and {p['beat'] for p in unknown}=={5,10,15}
    assert all(not p['source_control_supported'] for p in proof['checks'])
    assert proof['unverified_is_not_percussion_absence'] and not proof['audio_edited']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before


def test_visible_native_kick_remains_required_when_the_output_is_late(tmp_path):
    sr,source,warp=fixture(tmp_path,bed=0.)
    drums,_=sf.read(source,dtype='float32',always_2d=True)
    candidates=[{'index':9,'time':5.}]
    accepted,unknown,proof=bass_candidate_support(source,warp,candidates,drums,sr,0.,role='head')
    assert accepted==candidates and not unknown and proof['checks'][0]['source_control_supported']
    retain_mixture_references(source,warp,sr)
    delayed=np.zeros_like(drums);delayed[64:]=drums[:-64]
    path=tmp_path/'late.wav';sf.write(path,delayed,sr,subtype='FLOAT')
    audit=rendered_attack_audit(path,warp,sr)
    assert any(p['beat']==10 and p['view']=='bass_body' and abs(p['delay_seconds'])>.007 for p in audit['measured'])
    # Source classification cannot change merely because a later output failed.
    again,_,_=bass_candidate_support(source,warp,candidates,drums,sr,0.,role='head')
    assert again==candidates


def test_strong_native_contradiction_is_not_dropped_as_uncertainty(tmp_path):
    sr,source,warp=fixture(tmp_path,bed=0.)
    drums,_=sf.read(source,dtype='float32',always_2d=True)
    delayed=np.zeros_like(drums);delayed[64:]=drums[:-64]
    sf.write(source,delayed,sr,subtype='FLOAT')
    with pytest.raises(SourceWindowFailure,match='contradicts') as exc:
        bass_candidate_support(source,warp,[{'index':9,'time':5.}],drums,sr,0.,role='head')
    assert exc.value.role=='head'
    assert exc.value.evidence['native_estimated_percussion_reference']['delay_seconds']>.007


def test_native_uncertainty_uses_gradual_drums_and_preserves_harmonic_overlap():
    warp={'source_kick_alignment':{'head':{'unverified_native_bass_candidates':[{'source_seconds':5.}]}}}
    saved=deepcopy(warp)
    assert needs_handoff({},warp,style='zhilin')
    assert ownership_policy({},warp)=='smooth'
    assert needs_handoff({},warp,style='default')
    assert warp==saved


@pytest.mark.parametrize('signal',['native','estimate'])
def test_corrupt_audio_is_a_data_failure_never_an_uncertain_candidate(tmp_path,signal):
    sr,source,warp=fixture(tmp_path,bed=0.)
    drums,_=sf.read(source,dtype='float32',always_2d=True)
    if signal=='native':
        native=drums.copy();native[5*sr,0]=np.nan;sf.write(source,native,sr,subtype='FLOAT')
    else:drums[5*sr,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite') as exc:
        bass_candidate_support(source,warp,[{'index':9,'time':5.}],drums,sr,0.,role='head')
    assert not isinstance(exc.value,SourceWindowFailure)


def test_unobservable_candidate_never_enters_primary_source_controls(tmp_path,monkeypatch):
    from test_kick_grid import fixture as kick_fixture
    from superdj import kick_grid
    native,warp,sr=kick_fixture(tmp_path,missing=12)
    folder=tmp_path/'estimate';folder.mkdir()
    estimate,_,_=kick_fixture(folder)
    original=deepcopy(warp)
    repeated=kick_grid.repeated_attacks
    def omit_candidate(rows,*args,**kwargs):
        return [r for r in repeated(rows,*args,**kwargs) if r['index']!=11]
    monkeypatch.setattr(kick_grid,'repeated_attacks',omit_candidate)
    start=warp['input_output_beat_mapping']['head'][0]['source_seconds']
    base=round((start-.2)*sr)/sr
    fake_time=float(np.arange(.3,17.4,.503)[12]+.025)
    def discovery(audio,rate,centres,accepted,attack_band_hz=(100,1400)):
        return list(accepted)+([{'index':11,'time':fake_time-base,'eligible':True,
            'source_template_correlation':.99,'source_template_margin':.1}] if tuple(attack_band_hz)==(35,180) else [])
    monkeypatch.setattr(kick_grid,'augment_source_attacks',discovery)
    evidence={'head':{'path':str(estimate),'source_start_seconds':0.}}
    report=kick_grid.refine_source_controls(native,warp,sr,evidence)['head']
    assert 12 not in report['low_frequency_gap_recovery']['recovered_beats']
    assert 12 not in {e['beat'] for e in report['events']}
    assert report['unverified_native_bass_candidates'][0]['beat']==12
    target=warp['input_output_beat_mapping']['head'][12]['output_seconds']
    anchor=next(a for a in warp['anchors'] if abs(a['output_seconds']-target)<1e-7)
    assert abs(anchor['source_seconds']-fake_time)>.015
    # Bypassing native support used to move this point. The independent rate
    # continuity guard must now reject that damaging edit and restore anchors.
    monkeypatch.setattr('superdj.source_observability.bass_candidate_support',
        lambda path,warp,candidates,*args,**kwargs:(candidates,[],{'executed':False}))
    legacy=deepcopy(original)
    with pytest.raises(SourceWindowFailure,match='Short-term acoustic timing') as exc:
        kick_grid.refine_source_controls(native,legacy,sr,evidence)
    assert not exc.value.evidence['short_term_timing_continuity']['passed']
    assert legacy['anchors']==original['anchors']
