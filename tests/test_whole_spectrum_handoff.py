from copy import deepcopy
import numpy as np
import pytest
from superdj.mixing import blend,envelopes
from superdj.percussion import verification_components
from superdj.transition_quality import audit_vocals

@pytest.mark.parametrize('spec',[{'bias':3.},{'bias':4.},{'start_beat':4.,'end_beat':36.}])
def test_whole_spectrum_retains_timbre_endpoints_and_exact_contributions(monkeypatch,spec):
    import superdj.mixing as mixing
    sr=8000;n=sr*6;phase=np.linspace(0.,1.,n)**1.2
    rng=np.random.default_rng(781);a=rng.normal(0,.02,(n,2)).astype('float32');b=rng.normal(0,.03,(n,2)).astype('float32');before=[x.copy()for x in (a,b)]
    # This mode must not call a filter or estimate subtraction, even temporarily.
    original=mixing.split_bands;monkeypatch.setattr(mixing,'split_bands',lambda *a,**k:pytest.fail('Whole-spectrum path filtered source timbre'))
    y,report=blend(a,b,sr,beats=128,beat_phase=phase,style='zhilin',whole_spectrum_handoff=spec)
    g,_=envelopes(n,128,beat_phase=phase,style='zhilin',whole_spectrum_handoff=spec)
    np.testing.assert_array_equal(y,(a*g['mid_out']+b*g['mid_in']).astype('float32'))
    assert np.array_equal(a,before[0])and np.array_equal(b,before[1])
    assert np.array_equal(y[0],a[0])and np.array_equal(y[-1],b[-1])
    assert report['actual_10_90_shared_beats']>=16
    assert not report['frequency_filtering_performed']
    for band in ('bass','mid','high'):
        np.testing.assert_array_equal(g[band+'_in'],g['mid_in'])
        np.testing.assert_array_equal(g[band+'_in']+g[band+'_out'],np.ones((n,1)))
    monkeypatch.setattr(mixing,'split_bands',original)
    parts=verification_components(a,b,sr,g)
    np.testing.assert_allclose(parts['head']+parts['tail'],y,atol=2e-8)


def test_same_recording_is_transparent_through_the_entire_crossfade():
    sr=8000;t=np.arange(sr*8)/sr;x=np.column_stack([.12*np.sin(2*np.pi*55*t)+.06*np.sin(2*np.pi*1100*t)]*2).astype('float32')
    y,_=blend(x,x,sr,beats=128,style='zhilin',whole_spectrum_handoff={'start_beat':4.,'end_beat':36.})
    np.testing.assert_array_equal(y,x)

@pytest.mark.parametrize('spec',[{'start_beat':4.,'end_beat':5.},{'bias':100.},{'bias':float('nan')},{'bias':True},{'start_beat':-1.},{'end_beat':129.},{'unknown':1.}])
def test_invalid_or_abrupt_whole_handoffs_fail_without_modifying_sources(spec):
    x=np.ones((32000,2),dtype='float32')*.01;before=x.copy()
    with pytest.raises(ValueError):blend(x,x,8000,beats=128,style='zhilin',whole_spectrum_handoff=spec)
    np.testing.assert_array_equal(x,before)


def test_explicit_full_spectrum_cannot_silently_mix_with_other_routing():
    x=np.zeros((32000,2),dtype='float32')
    for kw in [{'drums_out':x,'drums_in':x},{'handoff_windows':{'bass':[20,40]}},{'bass_handoff_beats':16}]:
        with pytest.raises(ValueError):blend(x,x,8000,beats=128,style='zhilin',whole_spectrum_handoff={'bias':3.},**kw)


def test_voice_gate_uses_the_same_whole_spectrum_curve():
    sr=8000;x=np.ones((sr*10,2),dtype='float32')*.1;phase=np.linspace(0,1,len(x));transition={'beats':128,'whole_spectrum_handoff':{'start_beat':4.,'end_beat':36.}}
    # Both equal voice estimates are prominent only while the real faders coexist.
    actual,_=blend(x,x,sr,beats=128,style='zhilin',beat_phase=phase,whole_spectrum_handoff=transition['whole_spectrum_handoff'])
    measured=audit_vocals(x,x,actual,sr,x,x,transition,style='zhilin',beat_phase=phase)
    assert measured['passed'] and measured['sustained_competing_estimates_seconds']<2
