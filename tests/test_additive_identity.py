from copy import deepcopy
import numpy as np
import pytest
from superdj.percussion import _foreground_identity_ambiguities,_strengthened_own_rows,_known_native_fronts


def case():
    sr=8000;t=np.arange(sr*2)/sr
    own=np.column_stack([np.sin(2*np.pi*90*t)]*2)
    other=np.column_stack([np.sin(2*np.pi*130*t)]*2)
    raw=[{'eligible':True,'time':1.2045,'template_correlation':.95,'alternative_peak_margin':.4}]
    isolated=[{'eligible':True,'time':1.199875,'template_correlation':.998,'alternative_peak_margin':.46}]
    front={'eligible':True,'delay_seconds':.001,'template_correlation':.999,'alternative_peak_margin':.5}
    return sr,own,other,raw,isolated,{sr:[front]}


def explain(values):
    sr,own,other,raw,isolated,fronts=values
    return _foreground_identity_ambiguities(own+other,own,other,sr,[1.],raw,isolated,[35,180],fronts)


def test_two_exact_native_contributions_identify_additive_peak_without_pcm_changes():
    values=case();proof=explain(values)[0]
    assert proof['classification'].startswith('additive waveform interference')
    assert proof['raw_delay_seconds']==pytest.approx(.0045)
    assert proof['own_delay_seconds']==pytest.approx(-.000125)
    assert proof['other_native_delay_seconds']==.001
    assert proof['raw_timing_contradiction_retained']


@pytest.mark.parametrize('mutation',['own_late','other_late','weak_own','weak_other','ambiguous_own','ambiguous_other','missing_other'])
def test_additive_identity_cannot_hide_late_weak_unknown_or_ambiguous_sources(mutation):
    values=list(case());own=values[4][0];other=values[5][values[0]][0]
    if mutation=='own_late':own['time']=1.206
    if mutation=='other_late':other['delay_seconds']=.006
    if mutation=='weak_own':own['template_correlation']=.97
    if mutation=='weak_other':other['template_correlation']=.97
    if mutation=='ambiguous_own':own['alternative_peak_margin']=.01
    if mutation=='ambiguous_other':other['alternative_peak_margin']=.01
    if mutation=='missing_other':values[5]={}
    assert not explain(values)


def test_nonreconstructing_components_cannot_authorize_additive_identity():
    sr,own,other,raw,isolated,fronts=case()
    assert not _foreground_identity_ambiguities(own+other+.001,own,other,sr,[1.],raw,isolated,[35,180],fronts)


def test_individual_bounds_cannot_hide_a_large_source_pair_error():
    values=list(case());values[4][0]['time']=1.202
    values[5][values[0]][0]['delay_seconds']=-.002
    assert not explain(values)


def test_fresh_view_only_strengthens_an_independently_on_time_own_identity():
    direct=[{'eligible':True,'time':1.2,'template_correlation':.94,'alternative_peak_margin':.4}]
    fresh=[{'eligible':True,'time':1.2,'template_correlation':.997,'alternative_peak_margin':.5}]
    rows,proof=_strengthened_own_rows([1.],direct,fresh)
    assert rows==fresh and proof[0]['own_actual_pcm_template_correlation']==.94
    for changes in ({'time':1.206},{'eligible':False},{'template_correlation':.85},{'alternative_peak_margin':.01}):
        invalid=[{**direct[0],**changes}]
        rows,proof=_strengthened_own_rows([1.],invalid,fresh)
        assert rows==invalid and not proof


def test_bracketed_native_clock_does_not_invent_an_absent_other_attack():
    sr,own,other,raw,isolated,_=case()
    fronts={round(t*sr):[{'eligible':True,'delay_seconds':.001,
        'template_correlation':.999,'alternative_peak_margin':.5,'native_reference_seconds':t}]
        for t in (0.,.1,.2,.3,.5,1.5,1.7,1.8)}
    proof=_foreground_identity_ambiguities(own+other,own,other,sr,[1.],raw,isolated,[35,180],fronts,.55)[0]
    assert proof['other_native_delay_seconds'] is None
    assert proof['directly_verified_native_pair_error_seconds'] is None
    assert not proof['missing_other_attack_invented']
    assert len(proof['other_native_clock_bracketing_references'])==2
    fronts[round(.5*sr)][0]['delay_seconds']=.006
    assert not _foreground_identity_ambiguities(own+other,own,other,sr,[1.],raw,isolated,[35,180],fronts,.55)


def test_other_native_identity_uses_its_own_observed_band(monkeypatch):
    def recognize(audio,sr,centres,templates,**kwargs):
        assert kwargs['attack_band_hz']==[180,1200]
        return [{'eligible':True,'time':centres[0],'template_correlation':.999,'alternative_peak_margin':.5}]
    monkeypatch.setattr('superdj.kick_grid.template_attack_rows',recognize)
    refs=[('head',[1.],np.zeros((1,400)),[180,1200],set())]
    result=_known_native_fronts(np.zeros((16000,2)),8000,refs,{},'head',[100,1400])
    assert result[8000][0]['native_attack_band_hz']==[180,1200]
    assert result[8000][0]['delay_seconds']==pytest.approx(0.)


@pytest.mark.parametrize('side',['tail','head'])
def test_full_verification_keeps_real_late_component_failures_with_native_clock_proofs(tmp_path,monkeypatch,side):
    from test_percussion_identity import fixture
    from superdj.percussion import verify_blended_fronts
    sr,(own,other),warps=fixture(tmp_path,side)
    for warp,role in zip(warps,('tail','head')):
        warp['source_kick_alignment'][role]['stable_native_primary_clock']=True
    gains={'bass_out':np.ones((len(own),1)),'bass_in':np.ones((len(own),1))}
    monkeypatch.setattr('superdj.percussion.final_drum_estimates',lambda a,b,*x,**k:({'drums_out':a,'drums_in':b},[]))
    with pytest.raises(ValueError,match='misses the shared clock'):
        verify_blended_fronts(own+other,sr,*warps,gains,tmp_path,components={'tail':own,'head':other})
