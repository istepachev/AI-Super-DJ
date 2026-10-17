from pathlib import Path
import numpy as np
import pytest
from scipy.signal import butter,sosfiltfilt

from superdj.percussion import (_foreground_identity_ambiguities,_paired_front_audit,
    verify_blended_fronts)


def test_a_dominant_other_waveform_does_not_identify_a_quiet_source():
    sr=8000;t=np.arange(sr*3)/sr
    own=np.column_stack([.005*np.sin(2*np.pi*90*t)]*2)
    other=np.column_stack([.3*np.sin(2*np.pi*140*t)]*2)
    raw=[{'eligible':True,'time':1.214,'template_correlation':.96}]
    aligned=[{'eligible':True,'time':1.2,'template_correlation':.99}]
    explanation=_foreground_identity_ambiguities(own+other,own,other,sr,[1.],raw,aligned,[35,180])
    assert explanation[0]['raw_delay_seconds']==pytest.approx(.014)
    assert explanation[0]['own_delay_seconds']==pytest.approx(0)
    assert explanation[0]['raw_candidate_other_waveform_cosine']>.98
    late=[{'eligible':True,'time':1.207,'template_correlation':.99}]
    assert not _foreground_identity_ambiguities(own+other,own,other,sr,[1.],raw,late,[35,180])
    assert not _foreground_identity_ambiguities(own*20+other,own*20,other,sr,[1.],raw,aligned,[35,180])


def fixture(tmp_path,late_side=None):
    sr=8000;n=sr*12;centres=.5+np.arange(20)*.5
    time=np.arange(round(.1*sr))/sr
    pulse=.15*np.sin(2*np.pi*400*time)*np.exp(-time/.012)+.05*np.random.default_rng(8).normal(size=len(time))*np.exp(-time/.002)
    other=np.zeros_like(pulse);other[:len(pulse)-112]=.08*pulse[:len(pulse)-112]
    other[112:]+=pulse[:-112]*15
    decks=[];warps=[]
    for side,wave in [('tail',pulse),('head',other)]:
        native=np.zeros((n,2),dtype='float32')
        for stamp in centres:
            i=round(stamp*sr);native[i:i+len(wave)]+=wave[:,None]
        filtered=sosfiltfilt(butter(3,(180,1200),btype='bandpass',fs=sr,output='sos'),native[:,0])
        templates=np.array([filtered[round(t*sr)-96:round(t*sr)+360] for t in centres])
        path=tmp_path/(side+'.npz');np.savez(path,templates=templates)
        source={'attack_templates_path':str(path),'attack_band_hz':[180,1200],
            'events':[{'beat':i+1,'output_seconds':float(t)} for i,t in enumerate(centres)]}
        warps.append({'source_kick_alignment':{side:source},'input_output_beat_mapping':{side:[{'output_seconds':0.}]}})
        if side==late_side:
            native=np.pad(native,((64,0),(0,0)))[:n]
        decks.append(native)
    return sr,decks,warps


def test_actual_other_identity_can_be_disambiguated_without_modifying_pcm(tmp_path,monkeypatch):
    sr,(own,other),warps=fixture(tmp_path)
    mixed=own+other;gains={'bass_out':np.ones((len(own),1)),'bass_in':np.ones((len(own),1))}
    # No separation is needed when the known own contribution identifies its
    # native attack; this remains a direct PCM, not a guessed model, view.
    monkeypatch.setattr('superdj.percussion.final_drum_estimates',lambda *a,**k:pytest.fail('Unexpected guessed separation'))
    checked,report=verify_blended_fronts(mixed,sr,*warps,gains,tmp_path,components={'tail':own,'head':other})
    np.testing.assert_array_equal(checked,mixed)
    assert any(c['raw_mixture_identity_ambiguities'] for c in report['checks'])
    assert report['paired_source_front_audit']['maximum_pair_error_seconds']<=.003


@pytest.mark.parametrize('side',['tail','head'])
def test_known_component_with_a_real_late_hit_is_never_erased(tmp_path,monkeypatch,side):
    sr,(own,other),warps=fixture(tmp_path,side)
    gains={'bass_out':np.ones((len(own),1)),'bass_in':np.ones((len(own),1))}
    monkeypatch.setattr('superdj.percussion.final_drum_estimates',lambda a,b,*x,**k:({'drums_out':a,'drums_in':b},[]))
    with pytest.raises(ValueError,match='misses the shared clock'):
        verify_blended_fronts(own+other,sr,*warps,gains,tmp_path,components={'tail':own,'head':other})


def test_two_individually_small_errors_cannot_hide_a_large_blended_pair_error():
    checks=[{'role':'tail','recognized':[{'seconds':1.,'delay_seconds':.002}]},
            {'role':'head:estimated_bass','recognized':[{'seconds':1.,'delay_seconds':-.002}]}]
    with pytest.raises(ValueError,match='paired percussion'):_paired_front_audit(checks,8000)
    checks[1]['recognized'][0]['delay_seconds']=-.001
    report=_paired_front_audit(checks,8000)
    assert report['coincident_source_beats']==1 and report['maximum_pair_error_seconds']==pytest.approx(.003)
