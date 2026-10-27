"""Independent frequency envelopes with a gentle, uncompensated mid entry."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from .derived_audio import derived


def smooth(value):
    x = np.clip(value, 0, 1)
    return x * x * (3 - 2 * x)


def split_bands(audio, sample_rate, bass_hz=180.0, high_hz=3500.0):
    """Complementary zero-phase decomposition; bands sum to the source."""
    def calculate():
        low = sosfiltfilt(butter(4, bass_hz, fs=sample_rate, output="sos"), audio, axis=0)
        below_high = sosfiltfilt(butter(4, high_hz, fs=sample_rate, output="sos"), audio, axis=0)
        return low, below_high - low, audio - below_high
    return derived('frequency_bands', audio, (sample_rate, bass_hz, high_hz), calculate)


def envelopes(n, beats, *, tonal_conflict=False, beat_phase=None, style='default', profile=None,
              mid_handoff_bars=None, bass_handoff_beats=None,handoff_windows=None,whole_spectrum_handoff=None):
    """Complementary, gradual per-band transfer for every ordinary style.

    Both bass lines and dense mid parties are never held at full level at once.
    Highs change over the whole transition. Use the renderer's actual shared
    beat phase when supplied, rather than approximating phase by elapsed time.
    """
    if beats <= 0 or n < 2:
        raise ValueError("A transition needs positive beats and at least two samples")
    p = np.linspace(0., 1., n) if beat_phase is None else np.asarray(beat_phase, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or np.any(np.diff(p) < 0) or abs(p[0]) > 1e-9 or abs(p[-1] - 1) > 1e-9:
        raise ValueError("Envelope phase must run monotonically from zero to one")
    p = p[:, None]
    if style not in ('default','zhilin'):
        raise ValueError('Unknown frequency mixing style')
    if whole_spectrum_handoff is not None:
        if not isinstance(whole_spectrum_handoff,dict) or set(whole_spectrum_handoff)-{'start_beat','end_beat','bias'}:
            raise ValueError('Whole-spectrum handoff supports start_beat, end_beat and bias')
        if mid_handoff_bars is not None or bass_handoff_beats is not None or handoff_windows:
            raise ValueError('Choose a whole-spectrum handoff or separate band overrides')
        first=whole_spectrum_handoff.get('start_beat',0.)
        last=whole_spectrum_handoff.get('end_beat',float(beats))
        bias=whole_spectrum_handoff.get('bias',1.)
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in (first,last,bias)) or not 0<=first<last<=beats or not 1<=bias<=32:
            raise ValueError('Whole-spectrum handoff needs increasing shared beats and a finite bias from 1 to 32')
        transfer=1-(1-smooth((p*beats-first)/(last-first)))**bias
        span=[int(np.flatnonzero(transfer[:,0]>=value)[0]) for value in (.1,.9)]
        actual=float((p[span[1],0]-p[span[0],0])*beats)
        minimum=min(float(beats)*.5,16. if style=='zhilin' else 8.)
        if actual+1e-5<minimum:
            raise ValueError('Whole-spectrum handoff is too short for a smooth mix')
        gains={band+'_'+side:value for band in ('bass','mid','high') for side,value in (('out',1-transfer),('in',transfer))}
        return gains,{'style':style,'whole_spectrum_handoff':{'start_beat':first,'end_beat':last,'bias':bias},
            'incoming_mid_opening_beats':actual,'incoming_initial_attenuation_db':0.,
            'bass_handoff_beats':last-first,'bass_handoff_fraction':[first/beats,last/beats],
            'mid_handoff_fraction':[first/beats,last/beats],'high_handoff_fraction':[first/beats,last/beats],
            'mid_10_90_fraction':[float(p[i,0]) for i in span],
            'actual_10_90_shared_beats':actual,'band_pair_gain_sum_maximum':{band:1. for band in ('bass','mid','high')},
            'phase_source':'shared integrated clock' if beat_phase is not None else 'linear phase',
            'frequency_filtering_performed':False,'automatic_recovery_shortened_overlap':False}
    for label,value,limit in (('mid_handoff_bars',mid_handoff_bars,beats/4),('bass_handoff_beats',bass_handoff_beats,beats)):
        if value is not None and (isinstance(value,bool) or not np.isfinite(value) or not 0<value<=limit):
            raise ValueError(f'{label} must be positive and within the overlap')
    if handoff_windows is not None:
        if not isinstance(handoff_windows,dict) or set(handoff_windows)-{'bass','mid'}:
            raise ValueError('Handoff windows support bass and middle only')
        for span in handoff_windows.values():
            if not isinstance(span,(list,tuple)) or len(span)!=2 or not all(np.isfinite(x) for x in span) or not 0<=span[0]<span[1]<=beats:
                raise ValueError('Handoff windows need two increasing shared-beat positions within the overlap')
    if style!='zhilin' and (mid_handoff_bars is not None or bass_handoff_beats is not None or 'mid' in (handoff_windows or {})):
        raise ValueError('Per-transition handoff overrides require DJ Zhilin')
    if style=='zhilin':
        from .zhilin import reference_profile
        reference=reference_profile(profile) if not isinstance(profile,dict) else profile
        grid=np.asarray(reference['phase'])
        mid=np.interp(p[:,0],grid,reference['mid_in'])[:,None]
        high=np.interp(p[:,0],grid,reference['high_in'])[:,None]
        start,end=reference['bass_transfer_fraction']
        if bass_handoff_beats is None and not (handoff_windows or {}).get('bass'):
            # Preserve the reference's long handoff while keeping shortened
            # source overlaps from scaling it down to a one-bar switch.
            width=min(1.,max(end-start,32./beats))
            center=(start+end)/2
            start=max(0.,min(center-width/2,1-width));end=start+width
        if bass_handoff_beats is not None:
            center=(start+end)/2;width=bass_handoff_beats/beats
            start=max(0.,min(center-width/2,1-width));end=start+width
        bass=np.interp(np.clip((p[:,0]-start)/(end-start),0,1),grid,reference['bass_shape'])[:,None]
        mid_fraction=[0.,1.]
        if mid_handoff_bars is not None and mid_handoff_bars<beats/4:
            width=mid_handoff_bars*4/beats
            start_mid=max(0.,min(.55-width/2,1-width));end_mid=start_mid+width
            mid=smooth((p-start_mid)/width);mid_fraction=[start_mid,end_mid]
        if handoff_windows is not None:
            if not isinstance(handoff_windows,dict) or set(handoff_windows)-{'bass','mid'}:raise ValueError('Handoff windows support bass and middle only')
            for band,span in handoff_windows.items():
                if not isinstance(span,(list,tuple)) or len(span)!=2 or not all(np.isfinite(x) for x in span) or not 0<=span[0]<span[1]<=beats:
                    raise ValueError('Handoff windows need two increasing shared-beat positions within the overlap')
                if band=='mid':
                    if mid_handoff_bars is not None:raise ValueError('Choose one middle handoff override')
                    mid_fraction=[span[0]/beats,span[1]/beats];mid=smooth((p-mid_fraction[0])/(mid_fraction[1]-mid_fraction[0]))
                else:
                    if bass_handoff_beats is not None:raise ValueError('Choose one bass handoff override')
                    start,end=span[0]/beats,span[1]/beats;bass=smooth((p-start)/(end-start))
        gains={'bass_out':1-bass,'bass_in':bass,'mid_out':1-mid,'mid_in':mid,'high_out':1-high,'high_in':high}
        values=np.asarray(reference['mid_in'])
        span=[float(grid[np.searchsorted(values,.1)]),float(grid[np.searchsorted(values,.9)])]
        if mid_fraction!=[0.,1.]:
            span=[mid_fraction[0]+float(grid[np.searchsorted(smooth(grid),v)])*(mid_fraction[1]-mid_fraction[0]) for v in (.1,.9)]
        return gains,{'style':'zhilin','profile_name':reference['name'],'profile_sha256':reference.get('profile_sha256'),
            'profile_method':reference['method'],'incoming_mid_opening_beats':beats*(span[1]-span[0]),
            'incoming_initial_attenuation_db':0,'bass_handoff_beats':beats*(end-start),
            'mid_handoff_fraction':mid_fraction,'mid_10_90_fraction':span,'bass_handoff_fraction':[start,end],
            'mid_handoff_bars_override':mid_handoff_bars,'bass_handoff_beats_override':bass_handoff_beats,
            'high_handoff_fraction':[0.,1.],
            'band_pair_gain_sum_maximum':{band:float(np.max(gains[band+'_out']+gains[band+'_in'])) for band in ('bass','mid','high')},
            'phase_source':'shared integrated clock' if beat_phase is not None else 'linear phase',
            'handoff_windows_shared_beats':handoff_windows,
            'foreground_conflict_policy':'Complementary accompaniment; separate aligned vocal handoff; dense instrumental conflicts need cue review'}
    mid_beats = min(float(beats), 8 if tonal_conflict else 16)
    mid_width = mid_beats / beats
    mid_start=max(0.,min(.43-mid_width/2,1-mid_width))
    mid = smooth((p - mid_start) / mid_width)
    bass_beats = min(float(beats), max(16., beats / 8))
    bass_width=bass_beats/beats
    bass_start=max(0.,min(.55-bass_width/2,1-bass_width))
    if (handoff_windows or {}).get('bass'):
        first,last=handoff_windows['bass']
        bass_start=first/beats;bass_width=(last-first)/beats;bass_beats=last-first
    bass = smooth((p - bass_start) / bass_width)
    high = smooth(p)
    quiet_db = 2 if tonal_conflict else 1
    quiet = 10 ** (-quiet_db * (1 - mid) / 20)
    gains = {'bass_out': 1 - bass, 'bass_in': bass,
             'mid_out': 1 - mid, 'mid_in': mid * quiet,
             'high_out': 1 - high, 'high_in': high}
    report = {'incoming_mid_opening_beats': mid_beats,
              'incoming_initial_attenuation_db': quiet_db,
              'bass_handoff_beats': bass_beats,
              'mid_handoff_fraction': [mid_start, mid_start+mid_width],
              'bass_handoff_fraction': [bass_start, bass_start+bass_width],
              'high_handoff_fraction': [0., 1.],
              'band_pair_gain_sum_maximum': {band: float(np.max(gains[band + '_out'] + gains[band + '_in']))
                                             for band in ('bass', 'mid', 'high')},
              'phase_source': 'shared integrated clock' if beat_phase is not None else 'linear phase'}
    return gains, report


def recorded_envelopes(n,beats,transition,*,style,beat_phase):
    """Use recorded effective routing, including older default widths."""
    frequency=transition.get('frequency_processing')or{}
    controls={k:transition[k]for k in ('mid_handoff_bars','bass_handoff_beats','handoff_windows','whole_spectrum_handoff')if transition.get(k)is not None}
    if controls.get('whole_spectrum_handoff')is None:
        if not controls.get('handoff_windows')and frequency.get('handoff_windows_shared_beats'):
            controls['handoff_windows']=frequency['handoff_windows_shared_beats']
        if not(controls.get('handoff_windows')or{}).get('bass')and controls.get('bass_handoff_beats')is None and frequency.get('bass_handoff_beats')is not None:
            if style=='zhilin':controls['bass_handoff_beats']=frequency['bass_handoff_beats']
            elif frequency.get('bass_handoff_fraction'):
                controls['handoff_windows']={**controls.get('handoff_windows',{}),'bass':[v*beats for v in frequency['bass_handoff_fraction']]}
        if controls.get('mid_handoff_bars')is None and frequency.get('mid_handoff_bars_override')is not None:
            controls['mid_handoff_bars']=frequency['mid_handoff_bars_override']
    gains,curves=envelopes(n,beats,style=style,beat_phase=beat_phase,
        tonal_conflict=frequency.get('tonal_conflict_risk',False),**controls)
    from .mid_energy import apply_report
    return apply_report(gains,frequency),curves


def blend(outgoing, incoming, sample_rate, *, bass_hz=180.0,
          beats=64, tonal_conflict=False, beat_phase=None, style='default', profile=None,
          mid_handoff_bars=None, bass_handoff_beats=None,drums_out=None,drums_in=None,percussion_policy='shared',handoff_windows=None,
          allow_abrupt_percussion=False,whole_spectrum_handoff=None,
          mid_energy_mode='auto',vocal_out=None,vocal_in=None,vocal_processing=None):
    if outgoing.shape != incoming.shape or outgoing.ndim != 2 or outgoing.shape[1] != 2:
        raise ValueError("Frequency blend needs equally sized stereo blocks")
    if not np.isfinite(outgoing).all() or not np.isfinite(incoming).all():
        raise ValueError("Nonfinite overlap audio")
    if mid_energy_mode not in ('auto','legacy'):raise ValueError('Middle energy mode must be auto or legacy')
    n = len(incoming)
    if whole_spectrum_handoff is not None:
        if drums_out is not None or drums_in is not None or percussion_policy!='shared':
            raise ValueError('Whole-spectrum handoff does not subtract percussion estimates')
        gains,curves=envelopes(n,beats,tonal_conflict=tonal_conflict,beat_phase=beat_phase,style=style,
            mid_handoff_bars=mid_handoff_bars,bass_handoff_beats=bass_handoff_beats,
            handoff_windows=handoff_windows,whole_spectrum_handoff=whole_spectrum_handoff)
        continuity=validate_handoff_continuity(gains,beat_phase,beats,style=style)
        mixed=(outgoing*gains['mid_out']+incoming*gains['mid_in']).astype(np.float32)
        return mixed,{'method':'complementary whole-spectrum faders on unchanged timed decks',
            **curves,'musical_handoff_continuity':continuity,'tonal_conflict_risk':bool(tonal_conflict),
            'adaptive_mid_energy':{'mode':mid_energy_mode,'applied':False,'reason':'explicit whole-spectrum variant retained unchanged'},
            'entry_gain_compensated':False,
            'percussion_processing':{'executed':False,'reason':'explicit whole-spectrum handoff'},
            'endpoint_max_error':float(max(np.max(np.abs(mixed[0]-outgoing[0])),np.max(np.abs(mixed[-1]-incoming[-1]))))}
    if drums_out is not None or drums_in is not None:
        if drums_out is None or drums_in is None:raise ValueError('Both aligned drum estimates are required')
        if drums_out.shape!=outgoing.shape or drums_in.shape!=incoming.shape or not np.isfinite(drums_out).all() or not np.isfinite(drums_in).all():
            raise ValueError('Drum estimates must share the final deck PCM timing and shape')
        if bass_hz<=0 or n<128:raise ValueError('Percussion routing requires verified frequency envelopes')
    if n < 2:
        return incoming.copy(), {"method": "short endpoint"}
    position = np.linspace(0., 1., n)[:, None]
    out_gain, in_gain = np.cos(position * np.pi / 2), np.sin(position * np.pi / 2)
    if bass_hz <= 0 or n < 128:
        return (outgoing * out_gain + incoming * in_gain).astype(np.float32), {"method": "unverified equal-power fade"}
    low_a, mid_a, high_a = split_bands(outgoing, sample_rate, bass_hz)
    low_b, mid_b, high_b = split_bands(incoming, sample_rate, bass_hz)
    gains, curves = envelopes(n, beats, tonal_conflict=tonal_conflict, beat_phase=beat_phase,style=style,profile=profile,
                             mid_handoff_bars=mid_handoff_bars,bass_handoff_beats=bass_handoff_beats,handoff_windows=handoff_windows)
    if percussion_policy!='shared' and drums_out is None:raise ValueError('Percussion routing requires both final drum estimates')
    from .percussion import percussion_gains
    gains=percussion_gains(gains,beat_phase,beats,percussion_policy)
    drum_bands_a=split_bands(drums_out,sample_rate,bass_hz)if drums_out is not None else None
    drum_bands_b=split_bands(drums_in,sample_rate,bass_hz)if drums_in is not None else None
    from .mid_energy import decide,apply_report
    mid_energy=decide(mid_a-(drum_bands_a[1]if drum_bands_a is not None else 0),
        mid_b-(drum_bands_b[1]if drum_bands_b is not None else 0),sample_rate,gains,
        vocal_out=vocal_out,vocal_in=vocal_in,vocal_processing=vocal_processing,mode=mid_energy_mode)
    gains=apply_report(gains,{'adaptive_mid_energy':mid_energy})
    if mid_energy['applied']:
        actual_phase=np.linspace(0,1,n)if beat_phase is None else np.asarray(beat_phase)
        span=[float(actual_phase[np.flatnonzero(gains['mid_in'][:,0]>=value)[0]])for value in (.1,.9)]
        mid_energy['base_mid_10_90_fraction']=curves.get('mid_10_90_fraction')
        curves['mid_10_90_fraction']=span;curves['incoming_mid_opening_beats']=beats*(span[1]-span[0])
    curves['band_pair_gain_sum_maximum']['mid']=float(np.max(gains['mid_out']+gains['mid_in']))
    continuity=validate_handoff_continuity(gains,beat_phase,beats,allow_abrupt_percussion=allow_abrupt_percussion,
        style=style,reviewed_band_override=bass_handoff_beats is not None or bool((handoff_windows or {}).get('bass')))
    mixed = low_a * gains['bass_out'] + low_b * gains['bass_in']
    mixed += mid_a * gains['mid_out'] + mid_b * gains['mid_in']
    mixed += high_a * gains['high_out'] + high_b * gains['high_in']
    percussion=None
    if drums_out is not None or drums_in is not None:
        # Keep wide accompaniment curves, but transfer the whole percussion
        # spectrum together. An old kick's upper attack must not remain at
        # the mid gain after its fundamental has been handed to the next deck.
        for drum_a,drum_b,label in zip(drum_bands_a,drum_bands_b,('bass','mid','high')):
            mixed+=drum_a*(gains.get('drums_out',gains['bass_out'])-gains[label+'_out'])+drum_b*(gains.get('drums_in',gains['bass_in'])-gains[label+'_in'])
        percussion_phase=np.linspace(0.,1.,n) if beat_phase is None else np.asarray(beat_phase)
        percussion={'executed':True,'method':'Transfer the full estimated percussion spectrum with the bass clock, while accompaniment retains its independent broad frequency curves',
                    'handoff_fraction':curves['bass_handoff_fraction'],'handoff_beats':curves['bass_handoff_beats'],'source':'time-aligned estimates of final rendered decks; not isolated ground truth',
                    'ownership_policy':percussion_policy,
                    'maximum_simultaneous_estimated_drum_gain_product':float(np.max(gains.get('drums_out',gains['bass_out'])*gains.get('drums_in',gains['bass_in']))),
                    'maximum_outgoing_drum_gain_after_handoff':float(np.max(gains.get('drums_out',gains['bass_out'])[percussion_phase>=curves['bass_handoff_fraction'][1]])) if np.any(percussion_phase>=curves['bass_handoff_fraction'][1]) else None}
        if percussion_policy in ('sequential','smooth'):
            first=np.flatnonzero(gains['drums_out'][:,0]<1)[0];last=np.flatnonzero(gains['drums_in'][:,0]>=1)[0]
            percussion.update(method=('Explicit sequential percussion effect' if percussion_policy=='sequential' else 'Gradual complementary estimated percussion transfer; independent harmonic band curves retained'),
                handoff_fraction=[float(percussion_phase[first]),float(percussion_phase[last])],
                handoff_beats=float((percussion_phase[last]-percussion_phase[first])*beats))
    report = {"method": "three complementary zero-phase bands with independent bass/mid/high envelopes",
              "bass_hz": bass_hz, "high_hz": 3500., "beats": beats,
              **curves, "incoming_mid_start_multiplier": 0.,
              "tonal_conflict_risk": bool(tonal_conflict), "vocal_overlap_certified_absent": False,
              "entry_gain_compensated": False,
              "adaptive_mid_energy":mid_energy,
              "endpoint_max_error": float(max(np.max(np.abs(mixed[0] - outgoing[0])), np.max(np.abs(mixed[-1] - incoming[-1]))))}
    if mid_energy['applied']:report['method']='independent band handoff with energy-preserving noncompeting middle beds'
    if percussion:report['percussion_processing']=percussion
    if continuity:report['musical_handoff_continuity']=continuity
    return mixed.astype(np.float32), report


def validate_handoff_continuity(gains,beat_phase,beats,*,allow_abrupt_percussion=False,style=None,reviewed_band_override=False):
    """A sample-smooth envelope can still switch the musical core too quickly.

    Check effective accompaniment gains after routing. A short drum transfer
    must not silently become a short transfer of the entire low/mid party.
    """
    phase=np.asarray(beat_phase) if beat_phase is not None else np.linspace(0,1,len(gains['mid_in']))
    spans={}
    bands=['bass','mid']+(['percussion'] if 'drums_in' in gains else [])
    for band in bands:
        transfer=np.asarray(gains['drums_in' if band=='percussion' else band+'_in'])[:,0]
        if transfer.shape!=phase.shape or not np.isfinite(transfer).all() or np.any(np.diff(transfer)<-1e-9):raise ValueError('Accompaniment handoff must be finite and monotonic')
        first=np.flatnonzero(transfer>=.1);last=np.flatnonzero(transfer>=.9)
        if not len(first) or not len(last):raise ValueError('Incomplete accompaniment handoff')
        spans[band]=float((phase[last[0]]-phase[first[0]])*beats)
    required={'bass':min(2.,beats/4),'mid':min(4.,beats/4)}
    if style is not None and not reviewed_band_override:required['bass']=min(16. if style=='zhilin'else 8.,beats/2)
    if 'percussion' in spans:required['percussion']=0. if allow_abrupt_percussion else min(2.,beats/4)
    if 'percussion'in spans and style is not None and not reviewed_band_override and not allow_abrupt_percussion:required['percussion']=min(16. if style=='zhilin'else 8.,beats/2)
    if spans.get('percussion',float('inf'))+1e-5<required.get('percussion',0.):
        raise ValueError('Percussion handoff is too short for a smooth mix; select a gradual transfer or another source window')
    if any(spans[k]+1e-5<required[k] for k in spans):
        raise ValueError('Accompaniment energy handoff is too short; select compatible phrases and a gradual band transfer instead of a core cut')
    return {'executed':True,'effective_10_90_shared_beats':spans,'minimum_10_90_shared_beats':required,
        'passed':True,'abrupt_percussion_explicitly_allowed':bool(allow_abrupt_percussion),'method':'actual accompaniment and estimated percussion gain progression after routing; separate from primary-attack timing',
        'limitations':'A minimum envelope span does not certify source phrasing, every residual figure or subjective quality'}
