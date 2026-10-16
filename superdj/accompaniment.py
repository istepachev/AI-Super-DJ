"""Expose competing residual figures that a primary-kick audit cannot certify."""
import math
import numpy as np
from scipy.signal import resample_poly,find_peaks
from scipy.ndimage import gaussian_filter1d


def residual_rhythm_audit(outgoing,incoming,sample_rate,gains,*,drums_out,drums_in):
    """Conservative signal proxy on final routed accompaniment, not a taste score.

    This audit does not assign peaks to beats. It looks for repeated distinct
    attacks in two audible residual beds around the percussion handoff.
    """
    from .mixing import split_bands
    if outgoing.shape!=incoming.shape or any(x.shape!=outgoing.shape or not np.isfinite(x).all() for x in (outgoing,incoming,drums_out,drums_in)):
        raise ValueError('Residual rhythm audit requires aligned finite final decks')
    n=len(outgoing);rate=1000;divisor=math.gcd(sample_rate,rate)
    audible=(gains['drums_out'][:,0]>.01)&(gains['drums_out'][:,0]<.99)
    center=int(np.median(np.flatnonzero(audible))) if audible.any() else int(n*.55)
    first=max(0,center-round(4*sample_rate));last=min(n,center+round(5*sample_rate))
    beds={side:split_bands(pcm-drums,sample_rate)[:2] for side,pcm,drums in (('out',outgoing,drums_out),('in',incoming,drums_in))}
    full=np.mean((outgoing[first:last].astype('float64')**2+incoming[first:last].astype('float64')**2)/2)
    checks=[]
    for index,band in enumerate(('bass','mid')):
        events={};energies={}
        for side in ('out','in'):
            pcm=beds[side][index][first:last]*gains[band+'_'+side][first:last]
            energy=float(np.mean(pcm.astype('float64')**2));energies[side]=10*math.log10(max(energy,1e-20)/max(full,1e-20))
            # Preserve stereo energy; opposite channel phases cannot erase an attack.
            power=np.mean(pcm.astype('float64')**2,axis=1)
            envelope=resample_poly(power,rate//divisor,sample_rate//divisor)
            envelope=np.maximum(gaussian_filter1d(envelope,rate*.008),0)
            delay=round(.025*rate);novelty=np.maximum(0,envelope-np.r_[np.repeat(envelope[0],delay),envelope[:-delay]])
            threshold=max(float(np.quantile(novelty,.94))*.6,full*.0001)
            peaks,_=find_peaks(novelty,height=threshold,prominence=threshold*.65,distance=round(.07*rate))
            frames=np.clip(first+np.rint(peaks*sample_rate/rate).astype(int),first,last-1)
            joint=(gains[band+'_out'][frames,0]>=.05)&(gains[band+'_in'][frames,0]>=.05)
            peaks=peaks[joint]
            events[side]=peaks/rate+first/sample_rate if energies[side]>=-24 else np.array([])
        distinct={side:[float(t) for t in events[side] if not len(events[other]) or np.min(abs(events[other]-t))>.040] for side,other in (('out','in'),('in','out'))}
        competing=all(len(distinct[side])>=3 for side in ('out','in'))
        checks.append({'band':band,'routed_energy_relative_to_unweighted_decks_db':energies,
            'observed_residual_attack_seconds':{k:v.tolist() for k,v in events.items()},
            'distinct_residual_attack_seconds':distinct,'competing_rhythmic_beds':competing})
    return {'executed':True,'region_seconds':[first/sample_rate,last/sample_rate],
        'checks':checks,'competing_rhythmic_accompaniment':any(c['competing_rhythmic_beds'] for c in checks),
        'method':'distinct audible residual envelope attacks after final drum subtraction and actual band gains',
        'limitations':'Uncalibrated conflict proxy; source separation has leakage and different figures may be intentional; no metrical or subjective-quality certification'}


def accompaniment_gains(gains,policy='shared'):
    if policy not in ('shared','sequential_low_mid'):raise ValueError('Unknown accompaniment ownership policy')
    if policy=='shared':return gains
    if 'drums_out' not in gains or 'drums_in' not in gains:raise ValueError('Sequential accompaniment requires sequential percussion gains')
    if np.max(gains['drums_out']*gains['drums_in'])>1e-12:raise ValueError('Accompaniment ownership must share a single percussion handoff')
    return {**gains,**{band+'_'+side:gains['drums_'+side] for band in ('bass','mid') for side in ('out','in')}}
