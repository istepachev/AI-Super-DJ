"""Gentle frequency-specific support of dominant accompaniment, never drums."""
import numpy as np
from scipy.signal import butter,sosfiltfilt,resample_poly

LAW='dominant_accompaniment_cells_v1'
CUTS=np.geomspace(180.,3500.,9)

def cells(audio,sample_rate):
    """Telescoping zero-phase cells; exact sum is the existing middle band."""
    previous=sosfiltfilt(butter(4,CUTS[0],fs=sample_rate,output='sos'),audio,axis=0)
    for cutoff in CUTS[1:]:
        following=sosfiltfilt(butter(4,cutoff,fs=sample_rate,output='sos'),audio,axis=0)
        yield following-previous
        previous=following

def decide(outgoing,incoming,sample_rate,*,beat_phase,beats,enabled):
    report={'law':LAW,'applied':False,'frames':len(outgoing),'cutoffs_hz':CUTS.tolist(),
        'estimated_accompaniment_only':True,'timing_or_pitch_changed':False}
    if not enabled:return {**report,'reason':'retain existing power/legacy or whole-spectrum route'}
    if (outgoing.shape!=incoming.shape or outgoing.ndim!=2 or outgoing.shape[1]!=2
            or not np.isfinite(outgoing).all()or not np.isfinite(incoming).all()):
        raise ValueError('Selective middle requires aligned stereo accompaniment')
    n=len(outgoing)
    if n/sample_rate<4:return {**report,'reason':'insufficient accompaniment context'}
    phase=np.linspace(0,1,n)if beat_phase is None else np.asarray(beat_phase)
    if phase.shape!=(n,)or not np.isfinite(phase).all()or np.any(np.diff(phase)<0):
        raise ValueError('Selective middle requires the unchanged shared phase')
    divisor=max(1,int(sample_rate//8000));rate=sample_rate/divisor;block=round(rate*.5)
    a,b=[resample_poly(x,1,divisor,axis=0)for x in (outgoing,incoming)]
    count=min(len(a),len(b))//block
    at=np.minimum(np.rint((np.arange(count)+.5)*block*divisor).astype(int),n-1)
    centres=phase[at];knots=np.linspace(0,1,max(3,int(np.ceil(beats/4))+1))
    strengths=[]
    for aa,bb in zip(cells(a,rate),cells(b,rate)):
        pa,pb=[np.mean(x[:count*block].reshape(count,block,2)**2,axis=(1,2),dtype=np.float64)for x in (aa,bb)]
        floor=max(float(np.quantile(np.maximum(pa,pb),.75))*.01,1e-12)
        ratio=10*np.log10(np.maximum(pa,floor)/np.maximum(pb,floor))
        # Start supporting a dominant part above 3 dB separation; full support
        # needs 12 dB. Neighbouring half-seconds cooperate over 16 shared beats,
        # so source transients do not become a fast equalizer/AGC follower.
        pair=[]
        for sign in (1,-1):
            value=np.clip((sign*ratio-3)/9,0,1);value=value*value*(3-2*value)
            values=[]
            for point in knots:
                distance=np.abs(centres-point)*beats
                weights=np.maximum(0,1-distance/8)
                # A convex mean stays in [0, 1]; floating reduction can round
                # a saturated region one ulp above one. Canonicalize the
                # producer's result without relaxing strict report replay.
                support=np.dot(value,weights)/weights.sum()if weights.sum()else 0.
                values.append(float(np.clip(support,0.,1.)))
            pair.append(values)
        strengths.append(pair)
    if max(np.max(x)for x in strengths)<.05:
        return {**report,'reason':'no supported dominant accompaniment cells'}
    return {**report,'applied':True,'phase_knots':knots.tolist(),'strengths':strengths,
        'maximum_source_cell_gain':1.,'support_smoothing_shared_beats':16.,
        'source_envelopes_monotonic':True,'reason':'retain dominant frequency regions; shared regions keep existing power handoff',
        'limitations':'coarse spectral dominance of estimated accompaniment is not certified musical compatibility'}

def apply_report(gains,frequency,*,position,beat_phase=None):
    report=(frequency or{}).get('selective_mid_energy')or{}
    if not report.get('applied'):return gains
    n=len(gains['mid_in']);knots=np.asarray(report.get('phase_knots'),dtype=float)
    strengths=np.asarray(report.get('strengths'),dtype=float)
    if knots.ndim!=1:raise ValueError('Selective middle evidence does not match actual overlap')
    if (report.get('law')!=LAW or report.get('frames')!=n or report.get('cutoffs_hz')!=CUTS.tolist()
            or report.get('maximum_source_cell_gain')!=1. or strengths.shape!=(8,2,len(knots))
            or knots.ndim!=1 or len(knots)<3 or not np.isfinite(knots).all()
            or abs(knots[0])>1e-9 or abs(knots[-1]-1)>1e-9 or np.any(np.diff(knots)<=0)
            or not np.isfinite(strengths).all()or np.any(strengths<0)or np.any(strengths>1)):
        raise ValueError('Selective middle evidence does not match actual overlap')
    phase=np.linspace(0,1,n)if beat_phase is None else np.asarray(beat_phase)
    p=np.asarray(position).reshape(n,1)
    from .mixing import smooth
    targets={'out':1-smooth((p-.65)/.35),'in':smooth(p/.35)}
    result=dict(gains);maximum={side:np.zeros((n,1))for side in ('out','in')}
    for i,pair in enumerate(strengths):
        for j,side in enumerate(('out','in')):
            support=np.interp(phase,knots,pair[j])[:,None]
            base=gains['mid_'+side]
            candidate=base+support*np.maximum(targets[side]-base,0)
            # Conservative monotone projection: never reopen the outgoing part
            # or take back a previously opened incoming part. These projections
            # can only lower the proposed gain, never introduce extra support.
            actual=np.minimum.accumulate(candidate,axis=0)if side=='out'else np.minimum.accumulate(candidate[::-1],axis=0)[::-1]
            result[f'mid_cell_{i}_{side}']=actual
            maximum[side]=np.maximum(maximum[side],actual)
    # Conservative audibility/voice/continuity envelope. Actual PCM and native
    # attack components always use the individual cells, never this upper bound.
    for side in ('out','in'):result['mid_'+side]=maximum[side]
    return result

def mixed_middle(accompaniment,sample_rate,gains,side):
    if 'mid_cell_0_'+side not in gains:raise ValueError('Missing recorded selective middle cells')
    value=np.zeros_like(accompaniment,dtype=np.float64)
    for i,cell in enumerate(cells(accompaniment,sample_rate)):
        value+=cell*gains[f'mid_cell_{i}_{side}']
    return value
