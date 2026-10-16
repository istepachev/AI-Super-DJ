import numpy as np
import pytest
from superdj.mixing import envelopes,blend
from superdj.percussion import percussion_gains,verification_components
from superdj.accompaniment import residual_rhythm_audit,accompaniment_gains


def figures(sr=8000):
    t=np.arange(sr*16)/sr
    def pattern(offset,frequency):
        value=np.zeros(len(t))
        for stamp in np.arange(.5,15.5,.5)+offset:
            local=t-stamp;mask=(local>=0)&(local<.12)
            value[mask]+=.2*np.sin(2*np.pi*frequency*local[mask])*np.exp(-local[mask]/.04)
        return np.column_stack([value,value]).astype('float32')
    drums=pattern(0,80);a=drums+pattern(.13,330);b=drums+pattern(.28,550)
    phase=np.linspace(0,1,len(t));gains,_=envelopes(len(t),32,beat_phase=phase,style='zhilin')
    gains=percussion_gains(gains,phase,32,'sequential')
    return sr,a,b,drums,phase,gains


def test_aligned_drums_do_not_hide_distinct_residual_figures():
    sr,a,b,drums,phase,gains=figures()
    check=residual_rhythm_audit(a,b,sr,gains,drums_out=drums,drums_in=drums)
    assert check['competing_rhythmic_accompaniment']
    routed=accompaniment_gains(gains,'sequential_low_mid')
    after=residual_rhythm_audit(a,b,sr,routed,drums_out=drums,drums_in=drums)
    assert not after['competing_rhythmic_accompaniment']
    for band in ('bass','mid'):assert np.max(routed[band+'_out']*routed[band+'_in'])==0
    for side in ('out','in'):assert np.array_equal(routed['high_'+side],gains['high_'+side])


def test_tonal_beds_and_single_audible_figure_do_not_select_short_handoff():
    sr,a,b,drums,phase,gains=figures();t=np.arange(len(a))/sr
    tone=np.column_stack([.05*np.sin(2*np.pi*440*t)]*2).astype('float32')
    for x,y in ((tone,tone),(a,drums)):
        report=residual_rhythm_audit(x,y,sr,gains,drums_out=drums if x is a else np.zeros_like(x),drums_in=drums if y is drums else np.zeros_like(y))
        assert not report['competing_rhythmic_accompaniment']


def test_routed_known_components_exactly_reconstruct_new_mix():
    sr,a,b,drums,phase,gains=figures()
    mixed,report=blend(a,b,sr,beats=32,beat_phase=phase,style='zhilin',drums_out=drums,drums_in=drums,
        percussion_policy='sequential',accompaniment_policy='sequential_low_mid')
    routed=accompaniment_gains(gains,'sequential_low_mid')
    parts=verification_components(a,b,sr,routed,drums_out=drums,drums_in=drums)
    assert np.max(abs(parts['head']+parts['tail']-mixed))<1e-7
    assert np.array_equal(mixed[0],a[0]) and np.array_equal(mixed[-1],b[-1])
    assert report['residual_bass_mid_gain_product_max']=={'bass':0.,'mid':0.}
    with pytest.raises(ValueError,match='single percussion'):
        accompaniment_gains({**gains,'drums_in':gains['drums_out']},'sequential_low_mid')
