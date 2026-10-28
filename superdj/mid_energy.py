"""Bounded, signal-dependent mid-bed support; never a timing correction."""
import math
import numpy as np
from scipy.signal import resample_poly

LAW='bounded_independent_power_mid_v2'
PREVIOUS_LAW='conditional_equal_power_mid_v1'

def apply_report(gains, frequency):
    """Replay exactly the gains used by the mixer in every independent audit."""
    report=(frequency or {}).get('adaptive_mid_energy')
    if not report or not report.get('applied'):return gains
    n=len(gains['mid_in']);strength=report.get('strength')
    if (report.get('law') not in (LAW,PREVIOUS_LAW) or report.get('frames')!=n
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
    """Prefer independent power, bounded by local coherent energy and foreground.

    One fixed decision per overlap prevents a pumping loudness follower. The
    accepted law interprets the same monotonic handoff position as energy,
    changes no bass/high/drum gains and never raises either mid above its
    native full level. The pair's maximum amplitude sum is sqrt(2).
    Shared frequency occupancy alone does not establish coherent or competing
    parts. Retain foreground/narrow-tone guards; choose one fixed strength
    against every active half-second, never an automatic gain follower.
    """
    if mode not in ('auto','legacy','power'):raise ValueError('Middle energy mode must be auto or legacy, or explicit power')
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
    position=gb/np.maximum(ga+gb,1e-12)
    independent=pa[:count]*(1-position)+pb[:count]*position
    # The fully supported candidate can make an earlier voice audible. Screen
    # that candidate, rather than certifying it with the quieter old faders.
    foreground=(pva[:count]*(1-position)>np.maximum(independent*.10,1e-10))&(pvb[:count]*position>np.maximum(independent*.10,1e-10))
    competing_seconds=_longest(foreground)
    report['sustained_competing_foreground_seconds']=competing_seconds
    if competing_seconds>=2:return {**report,'reason':'simultaneous foreground estimates; retain complementary middle'}
    floor_a=max(1e-12,float(np.quantile(pa,.75))*.01);floor_b=max(1e-12,float(np.quantile(pb,.75))*.01)
    active=(pa[:count]>floor_a)&(pb[:count]>floor_b)&(1-position>=.05**2)&(position>=.05**2)
    fft_size=1<<(block-1).bit_length();frequencies=np.fft.rfftfreq(fft_size,1/rate)
    edges=np.geomspace(180,min(3500,rate*.48),49);bins=np.searchsorted(edges,frequencies)-1
    selected=(bins>=0)&(bins<48);window=np.hanning(block)[:,None];scores=[];scored=[];narrow=[]
    for i in range(count):
        if not active[i]:scored.append(0.);narrow.append(False);continue
        spectra=[];compact=[]
        for audio in (aa,bb):
            power=np.mean(abs(np.fft.rfft(audio[i*block:(i+1)*block]*window,n=fft_size,axis=0))**2,axis=1)
            values=np.bincount(bins[selected],weights=power[selected],minlength=48)
            # Adjacent spectral bins cooperate; tiny detuning is not evidence
            # that two foregrounds occupy different frequency space.
            values=np.convolve(values,[.25,.5,.25],mode='same');values/=max(float(values.sum()),1e-30);spectra.append(values)
            compact.append(float(np.sort(values)[-6:].sum())>=.9)
        score=float(np.minimum(*spectra).sum());scores.append(score);scored.append(score)
        narrow.append(score>.8 and all(compact))
    high=float(np.quantile(scores,.9))if scores else 0.
    competing_spectrum_seconds=_longest(np.asarray(scored)>.5)
    report.update(simultaneously_active_half_second_windows=int(np.count_nonzero(active)),spectral_overlap_90th_percentile=high,
        sustained_shared_spectral_occupancy_seconds=competing_spectrum_seconds)
    if np.count_nonzero(active)<4:
        return {**report,'reason':'insufficient simultaneous active evidence; retain complementary middle'}
    narrow_seconds=_longest(narrow)
    report['sustained_shared_narrow_tone_seconds']=narrow_seconds
    if narrow_seconds>=2:
        return {**report,'reason':'sustained shared narrow tones; retain complementary middle'}
    count_frames=count*block
    covariance=np.mean((aa[:count_frames]*bb[:count_frames]).reshape(count,block,-1),axis=(1,2),dtype=np.float64)
    coherence=covariance/np.maximum(np.sqrt(pa[:count]*pb[:count]),1e-20)
    report['positive_waveform_coherence_90th_percentile']=float(np.quantile(np.maximum(coherence[active],0),.9))
    if report['positive_waveform_coherence_90th_percentile']>=.85:
        return {**report,'reason':'coherent shared waveform; retain complementary middle'}
    # Positive covariance may add energy; negative covariance is not permission
    # to exceed native unity and amplify a cancellation. Check every active
    # block, including the beginning, with no percentile hiding a local swell.
    strength=0.;ceiling_db=1.;excess=None
    for value in (1.,.875,.75,.625,.5,.375,.25,.125):
        out=(1-value)*ga+value*np.sqrt(1-position)
        inc=(1-value)*gb+value*np.sqrt(position)
        conservative=out**2*pa[:count]+inc**2*pb[:count]+2*out*inc*np.maximum(covariance,0)
        worst=float(np.max(10*np.log10(np.maximum(conservative[active],1e-20)/np.maximum(independent[active],1e-20))))
        if worst<=ceiling_db:
            strength=value;excess=worst;break
    report.update(maximum_predicted_active_window_excess_db=excess,predicted_excess_ceiling_db=ceiling_db,
        energy_reference='handoff-weighted native stereo powers; equal independent parts at full strength',
        local_energy_window_seconds=.5,local_energy_windows_checked=int(np.count_nonzero(active)),
        compensation_is_fixed_per_overlap=True,negative_correlation_not_boosted=True)
    if strength==0:
        return {**report,'reason':'local coherent energy ceiling; retain complementary middle'}
    return {**report,'applied':True,'strength':strength,'reason':'bounded independent-power support; shared broad spectra alone are not a veto',
        'bass_high_and_percussion_gains_unchanged':True,'original_monotonic_handoff_position_preserved':True,
        'maximum_mid_pair_gain_sum':math.sqrt(2.),'absence_of_all_musical_competition_certified':False}
