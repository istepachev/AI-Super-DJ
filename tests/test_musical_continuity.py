import numpy as np
import pytest
from superdj.mixing import envelopes,blend,smooth,validate_handoff_continuity
from superdj.percussion import percussion_gains,verification_components


def test_a_sample_smooth_core_cut_is_not_a_musical_transition():
    phase=np.linspace(0,1,64001);gains,_=envelopes(len(phase),128,beat_phase=phase,style='zhilin')
    fast=smooth((phase*128-69)/.9)[:,None]
    for band in ('bass','mid'):gains[band+'_in']=fast;gains[band+'_out']=1-fast
    assert np.max(abs(np.diff(fast[:,0])))<.004  # No digital discontinuity; still an audible core cut.
    with pytest.raises(ValueError,match='energy handoff is too short'):validate_handoff_continuity(gains,phase,128)


def test_a_short_percussion_transfer_does_not_move_the_entire_accompaniment():
    phase=np.linspace(0,1,64001);base,_=envelopes(len(phase),128,beat_phase=phase,style='zhilin')
    gains=percussion_gains(base,phase,128,'sequential')
    with pytest.raises(ValueError,match='Percussion handoff is too short'):
        validate_handoff_continuity(gains,phase,128)
    assert validate_handoff_continuity(gains,phase,128,allow_abrupt_percussion=True)['passed']
    assert all(np.array_equal(base[k],gains[k]) for k in base)


def test_phrase_positioned_smooth_handoffs_keep_highs_and_shared_clock():
    phase=np.linspace(0,1,64001);base,_=envelopes(len(phase),128,beat_phase=phase,style='zhilin')
    gains,report=envelopes(len(phase),128,beat_phase=phase,style='zhilin',handoff_windows={'mid':[24,64],'bass':[48,64]})
    assert report['mid_handoff_fraction']==[24/128,64/128]
    assert validate_handoff_continuity(gains,phase,128)['passed']
    for side in ('out','in'):assert np.array_equal(gains['high_'+side],base['high_'+side])
    assert gains['mid_in'][round(.5*(len(phase)-1)),0]==1
    with pytest.raises(ValueError,match='one middle'):envelopes(len(phase),128,style='zhilin',mid_handoff_bars=8,handoff_windows={'mid':[24,64]})


def test_final_smooth_mix_reconstructs_known_contributions():
    sr=8000;t=np.arange(sr*12)/sr;phase=np.linspace(0,1,len(t));a=np.column_stack([.1*np.sin(2*np.pi*90*t)]*2).astype('float32');b=np.column_stack([.1*np.sin(2*np.pi*110*t)]*2).astype('float32')
    windows={'bass':[12,20],'mid':[4,20]};gains,_=envelopes(len(t),32,beat_phase=phase,style='zhilin',handoff_windows=windows)
    mixed,report=blend(a,b,sr,beats=32,beat_phase=phase,style='zhilin',handoff_windows=windows,drums_out=a*.3,drums_in=b*.4)
    parts=verification_components(a,b,sr,gains,drums_out=a*.3,drums_in=b*.4)
    assert np.max(abs(parts['head']+parts['tail']-mixed))<1e-7
    assert report['musical_handoff_continuity']['passed']
    with pytest.raises(ValueError,match='too short'):
        blend(a,b,sr,beats=32,beat_phase=phase,style='zhilin',handoff_windows={'bass':[16,17],'mid':[16,17]})


def test_foreground_audit_uses_the_actual_phrase_positioned_window():
    from superdj.transition_quality import audit_vocals
    sr=8000;t=np.arange(sr*16)/sr;phase=np.linspace(0,1,len(t));voice=np.column_stack([.1*np.sin(2*np.pi*440*t)]*2).astype('float32')
    broad=audit_vocals(voice,voice,voice,sr,voice,voice,{'beats':128},style='zhilin',beat_phase=phase)
    positioned=audit_vocals(voice,voice,voice,sr,voice,voice,{'beats':128,'handoff_windows':{'mid':[0,8]}},style='zhilin',beat_phase=phase)
    assert not broad['passed'] and positioned['passed']
