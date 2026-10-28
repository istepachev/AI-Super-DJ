"""Observable energy/routing regressions, independent of report wording."""
import math
import numpy as np
import pytest
from scipy.signal import butter,sosfiltfilt
from superdj.mid_energy import decide,apply_report,LAW,PREVIOUS_LAW
from superdj.mixing import envelopes

def beds(seconds=12):
    sr=8000;rng=np.random.default_rng(384)
    sos=butter(3,(200,3000),btype='bandpass',fs=sr,output='sos')
    a,b=[sosfiltfilt(sos,rng.normal(0,.08,(sr*seconds,2)),axis=0)for _ in range(2)]
    return sr,a,b

def report(a,b,sr):
    g,_=envelopes(len(a),128,style='zhilin')
    return g,decide(a,b,sr,g,vocal_out=a*0,vocal_in=b*0)

def test_independent_broad_shared_spectra_receive_full_power_support():
    sr,a,b=beds();g,r=report(a,b,sr)
    assert r['spectral_overlap_90th_percentile']>.8
    assert r['applied']and r['strength']==1
    updated=apply_report(g,{'adaptive_mid_energy':r})
    mixed=a*updated['mid_out']+b*updated['mid_in']
    position=g['mid_in']/(g['mid_in']+g['mid_out'])
    expected=(1-position)*a*a+position*b*b
    # The former spectral veto left a 3 dB valley on this independent material.
    core=(position[:,0]>.4)&(position[:,0]<.6)
    assert abs(10*np.log10(np.mean(mixed[core]**2)/np.mean(expected[core])))<.12

def test_partly_coherent_music_uses_fixed_partial_support_without_local_swell():
    sr,a,c=beds();b=.5*a+math.sqrt(.75)*c;g,r=report(a,b,sr)
    assert r['applied']and 0<r['strength']<1
    assert r['maximum_predicted_active_window_excess_db']<=1
    u=apply_report(g,{'adaptive_mid_energy':r});mixed=a*u['mid_out']+b*u['mid_in']
    p=g['mid_in']/(g['mid_in']+g['mid_out'])
    for lo in range(sr,len(a)-sr,sr//2):
        sl=slice(lo,lo+sr//2);expected=np.mean((1-p[sl])*a[sl]**2+p[sl]*b[sl]**2)
        assert 10*np.log10(np.mean(mixed[sl]**2)/expected)<=1.12
    assert np.all(np.diff(u['mid_in'][:,0])>=-1e-12)
    assert np.all(np.diff(u['mid_out'][:,0])<=1e-12)

def test_matching_broad_waveform_keeps_legacy():
    sr,a,_=beds();g,r=report(a,a,sr)
    assert not r['applied']
    assert apply_report(g,{'adaptive_mid_energy':r})is g

def test_nearby_narrow_tones_are_not_mistaken_for_independent_music():
    sr=8000;t=np.arange(sr*8)/sr
    a=np.repeat((.1*np.sin(2*np.pi*420*t))[:,None],2,axis=1)
    b=np.repeat((.1*np.sin(2*np.pi*426*t))[:,None],2,axis=1)
    _,r=report(a,b,sr)
    assert not r['applied']and r['sustained_shared_narrow_tone_seconds']>=2

def test_short_early_coherence_cannot_hide_behind_later_independent_music():
    sr,a,b=beds(24);b[sr:sr*3]=a[sr:sr*3];g,r=report(a,b,sr)
    if r['applied']:
        assert r['maximum_predicted_active_window_excess_db']<=1
        assert r['strength']<1

def test_previous_v1_evidence_replays_exactly():
    sr,a,b=beds(4);g,r=report(a,b,sr);assert r['applied']
    v1={**r,'law':PREVIOUS_LAW}
    v2=apply_report(g,{'adaptive_mid_energy':r});old=apply_report(g,{'adaptive_mid_energy':v1})
    for key in g:assert np.array_equal(v2[key],old[key])

@pytest.mark.parametrize('style',['default','zhilin'])
def test_only_mid_gains_change_and_each_stays_bounded(style):
    sr,a,b=beds(12);g,_=envelopes(len(a),64,style=style)
    r=decide(a,b,sr,g,vocal_out=a*0,vocal_in=b*0);assert r['applied']
    u=apply_report(g,{'adaptive_mid_energy':r})
    for key in ('bass_in','bass_out','high_in','high_out'):assert np.array_equal(g[key],u[key])
    for side in ('in','out'):assert np.min(u['mid_'+side])>=0 and np.max(u['mid_'+side])<=1
    assert np.max(u['mid_in']+u['mid_out'])<=math.sqrt(2)+1e-12

def test_routed_foregrounds_are_screened_at_the_new_audibility():
    sr,a,b=beds(8);g,_=envelopes(len(a),128,style='zhilin')
    r=decide(a,b,sr,g,vocal_out=a*.9,vocal_in=b*.9)
    assert not r['applied']and r['sustained_competing_foreground_seconds']>=2

def test_energy_evidence_includes_newly_audible_entry_windows():
    sr,a,b=beds(12);n=len(a)
    position=np.linspace(0,1,n)[:,None]**4
    g={'mid_out':1-position,'mid_in':position}
    r=decide(a,b,sr,g,vocal_out=a*0,vocal_in=b*0)
    assert r['applied']
    indices=np.minimum(np.rint((np.arange(24)+.5)*sr*.5).astype(int),n-1)
    before=((1-position[indices,0])>=.05)&(position[indices,0]>=.05)
    assert r['local_energy_windows_checked']>np.count_nonzero(before)
