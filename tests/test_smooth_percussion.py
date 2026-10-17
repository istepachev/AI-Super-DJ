import numpy as np
import pytest
from superdj.mixing import blend,envelopes,validate_handoff_continuity
from superdj.percussion import percussion_gains,verification_components


def test_uncertain_rhythm_transfer_cannot_silently_become_a_one_beat_cut():
    sr=8000;t=np.arange(sr*20)/sr;phase=np.linspace(0,1,len(t));n=len(t)
    a=np.column_stack([.1*np.sin(2*np.pi*90*t)]*2).astype('float32')
    b=np.column_stack([.1*np.sin(2*np.pi*120*t)]*2).astype('float32')
    with pytest.raises(ValueError,match='Percussion handoff is too short'):
        blend(a,b,sr,beats=32,beat_phase=phase,style='zhilin',drums_out=a*.7,drums_in=b*.6,percussion_policy='sequential')
    mixed,report=blend(a,b,sr,beats=32,beat_phase=phase,style='zhilin',drums_out=a*.7,drums_in=b*.6,percussion_policy='smooth')
    base,_=envelopes(n,32,beat_phase=phase,style='zhilin')
    gains=percussion_gains(base,phase,32,'smooth')
    assert all(np.array_equal(gains[k],base[k]) for k in base)
    assert np.max(abs(gains['drums_out']+gains['drums_in']-1))<1e-12
    assert validate_handoff_continuity(gains,phase,32)['effective_10_90_shared_beats']['percussion']>=4
    parts=verification_components(a,b,sr,gains,drums_out=a*.7,drums_in=b*.6)
    np.testing.assert_allclose(parts['tail']+parts['head'],mixed,atol=2e-8)
    assert report['percussion_processing']['handoff_beats']>=7.99
    assert report['endpoint_max_error']<1e-7


def test_gradual_rhythm_policy_retains_an_already_broad_bass_transfer():
    phase=np.linspace(0,1,128001)**1.2
    base,_=envelopes(len(phase),128,beat_phase=phase,style='zhilin')
    gains=percussion_gains(base,phase,128,'smooth')
    report=validate_handoff_continuity(gains,phase,128)
    assert report['passed'] and report['effective_10_90_shared_beats']['percussion']>8
