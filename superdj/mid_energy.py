"""Bounded, signal-dependent mid-bed support; never a timing correction."""
import math
import numpy as np
from scipy.signal import resample_poly

LAW='conditional_equal_power_mid_v1'

def apply_report(gains, frequency):
    """Replay exactly the gains used by the mixer in every independent audit."""
    report=(frequency or {}).get('adaptive_mid_energy')
    if not report or not report.get('applied'):return gains
    n=len(gains['mid_in']);strength=report.get('strength')
    if (report.get('law')!=LAW or report.get('frames')!=n
            or isinstance(strength,bool) or not isinstance(strength,(int,float))
            or not np.isfinite(strength) or not 0<=strength<=1
            or report.get('maximum_source_gain')!=1.
            or abs(report.get('maximum_pair_gain_sum',float('inf'))-math.sqrt(2.))>1e-9):
        raise ValueError('Adaptive middle evidence does not match the actual overlap')
    a,b=gains['mid_out'],gains['mid_in']
    position=b/np.maximum(a+b,1e-12)
    return {**gains,'mid_out':(1-strength)*a+strength*np.sqrt(1-position),
        'mid_in':(1-strength)*b+strength*np.sqrt(position)}

def _powers(audio,block):
    count=len(audio)//block
    return np.mean(audio[:count*block].reshape(count,block,-1)**2,axis=(1,2),dtype=np.float64)

def _longest(values):
    longest=run=0
    for value in values:run=run+1 if value else 0;longest=max(longest,run)
    return longest*.5

def decide(bed_out,bed_in,sample_rate,gains,*,vocal_out=None,vocal_in=None,vocal_processing=None,mode='auto'):
    """Retain the old law unless actual mid spectra and voice estimates support it.

    One fixed decision per overlap prevents a pumping loudness follower. The
    accepted law interprets the same monotonic handoff position as energy,
    changes no bass/high/drum gains and never raises either mid above its
    native full level. The pair's maximum amplitude sum is sqrt(2).
    Spectral overlap is a conservative engineering proxy, not a taste model.
    """
    if mode not in ('auto','legacy'):raise ValueError('Middle energy mode must be auto or legacy')
    n=len(bed_out);report={'mode':mode,'applied':False,'law':LAW,'frames':n,
        'maximum_source_gain':1.,'maximum_pair_gain_sum':math.sqrt(2.),'strength':0.,'timing_or_pitch_changed':False,
        'classification':'estimated frequency occupancy and foreground activity; not certified musical compatibility'}
    if mode=='legacy':return {**report,'reason':'explicit legacy complementary middle retained'}
    if vocal_out is None or vocal_in is None:
        return {**report,'reason':'aligned foreground estimates unavailable; retain legacy gains'}
    if bed_out.shape!=bed_in.shape or any(x.shape!=bed_out.shape or not np.isfinite(x).all()for x in (bed_out,bed_in,vocal_out,vocal_in)):
        raise ValueError('Middle energy analysis needs both exact stereo beds and aligned foreground estimates')
    divisor=max(1,int(sample_rate//8000));rate=sample_rate/divisor;block=round(rate*.5)
    if n/sample_rate<2:return {**report,'reason':'insufficient actual overlap for spectral evidence'}
    voices=[np.asarray(vocal_out),np.asarray(vocal_in)]
    if (vocal_processing or {}).get('executed'):
        from .mixing import smooth
        phase=np.linspace(0,1,n)[:,None]
        ranges=[vocal_processing.get('outgoing_voice_exit_fraction'),vocal_processing.get('incoming_voice_entry_fraction')]
        for limits in ranges:
            if not isinstance(limits,(list,tuple)) or len(limits)!=2 or not 0<=limits[0]<limits[1]<=1:
                raise ValueError('Missing actual foreground handoff fractions')
        a,b=ranges;voices=[voices[0]*(1-smooth((phase-a[0])/(a[1]-a[0]))),voices[1]*smooth((phase-b[0])/(b[1]-b[0]))]
    aa,bb,va,vb=[resample_poly(x,1,divisor,axis=0)for x in (bed_out,bed_in,*voices)]
    pa,pb,pva,pvb=[_powers(x,block)for x in (aa,bb,va,vb)];count=min(map(len,(pa,pb,pva,pvb)))
    indices=np.minimum(np.rint((np.arange(count)+.5)*block*divisor).astype(int),n-1)
    ga,gb=[gains['mid_'+side][indices,0]for side in ('out','in')]
    reference=pa[:count]*ga**2+pb[:count]*gb**2
    foreground=(pva[:count]*ga**2>np.maximum(reference*.10,1e-10))&(pvb[:count]*gb**2>np.maximum(reference*.10,1e-10))
    competing_seconds=_longest(foreground)
    report['sustained_competing_foreground_seconds']=competing_seconds
    if competing_seconds>=2:return {**report,'reason':'simultaneous foreground estimates; retain complementary middle'}
    floor_a=max(1e-12,float(np.quantile(pa,.75))*.01);floor_b=max(1e-12,float(np.quantile(pb,.75))*.01)
    active=(pa[:count]>floor_a)&(pb[:count]>floor_b)&(ga>=.05)&(gb>=.05)
    fft_size=1<<(block-1).bit_length();frequencies=np.fft.rfftfreq(fft_size,1/rate)
    edges=np.geomspace(180,min(3500,rate*.48),49);bins=np.searchsorted(edges,frequencies)-1
    selected=(bins>=0)&(bins<48);window=np.hanning(block)[:,None];scores=[];scored=[]
    for i in range(count):
        if not active[i]:scored.append(0.);continue
        spectra=[]
        for audio in (aa,bb):
            power=np.mean(abs(np.fft.rfft(audio[i*block:(i+1)*block]*window,n=fft_size,axis=0))**2,axis=1)
            values=np.bincount(bins[selected],weights=power[selected],minlength=48)
            # Adjacent spectral bins cooperate; tiny detuning is not evidence
            # that two foregrounds occupy different frequency space.
            values=np.convolve(values,[.25,.5,.25],mode='same');values/=max(float(values.sum()),1e-30);spectra.append(values)
        score=float(np.minimum(*spectra).sum());scores.append(score);scored.append(score)
    high=float(np.quantile(scores,.9))if scores else 0.
    competing_spectrum_seconds=_longest(np.asarray(scored)>.5)
    report.update(simultaneously_active_half_second_windows=int(np.count_nonzero(active)),spectral_overlap_90th_percentile=high,
        sustained_shared_spectral_occupancy_seconds=competing_spectrum_seconds)
    if high>.35 or competing_spectrum_seconds>=2:
        return {**report,'reason':'shared or uncertain mid occupancy; retain complementary middle'}
    return {**report,'applied':True,'strength':1.,'reason':'separate estimated mid occupancy without sustained competing foreground',
        'bass_high_and_percussion_gains_unchanged':True,'original_monotonic_handoff_position_preserved':True,
        'maximum_mid_pair_gain_sum':math.sqrt(2.),'absence_of_all_musical_competition_certified':False}
