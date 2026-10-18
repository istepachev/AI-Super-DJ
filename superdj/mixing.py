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
              mid_handoff_bars=None, bass_handoff_beats=None,handoff_windows=None):
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
            width=min(1.,max(end-start,16./beats))
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
    bass_beats = min(float(beats), max(8., beats / 8))
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


def blend(outgoing, incoming, sample_rate, *, bass_hz=180.0,
          beats=64, tonal_conflict=False, beat_phase=None, style='default', profile=None,
          mid_handoff_bars=None, bass_handoff_beats=None,drums_out=None,drums_in=None,percussion_policy='shared',handoff_windows=None,
          allow_abrupt_percussion=False):
    if outgoing.shape != incoming.shape or outgoing.ndim != 2 or outgoing.shape[1] != 2:
        raise ValueError("Frequency blend needs equally sized stereo blocks")
    if not np.isfinite(outgoing).all() or not np.isfinite(incoming).all():
        raise ValueError("Nonfinite overlap audio")
    n = len(incoming)
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
    continuity=validate_handoff_continuity(gains,beat_phase,beats,allow_abrupt_percussion=allow_abrupt_percussion)
    mixed = low_a * gains['bass_out'] + low_b * gains['bass_in']
    mixed += mid_a * gains['mid_out'] + mid_b * gains['mid_in']
    mixed += high_a * gains['high_out'] + high_b * gains['high_in']
    percussion=None
    if drums_out is not None or drums_in is not None:
        # Keep wide accompaniment curves, but transfer the whole percussion
        # spectrum together. An old kick's upper attack must not remain at
        # the mid gain after its fundamental has been handed to the next deck.
        for drum_a,drum_b,label in zip(split_bands(drums_out,sample_rate,bass_hz),split_bands(drums_in,sample_rate,bass_hz),('bass','mid','high')):
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
              "endpoint_max_error": float(max(np.max(np.abs(mixed[0] - outgoing[0])), np.max(np.abs(mixed[-1] - incoming[-1]))))}
    if percussion:report['percussion_processing']=percussion
    if continuity:report['musical_handoff_continuity']=continuity
    return mixed.astype(np.float32), report


def validate_handoff_continuity(gains,beat_phase,beats,*,allow_abrupt_percussion=False):
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
    if 'percussion' in spans:required['percussion']=0. if allow_abrupt_percussion else min(2.,beats/4)
    if spans.get('percussion',float('inf'))+1e-5<required.get('percussion',0.):
        raise ValueError('Percussion handoff is too short for a smooth mix; select a gradual transfer or another source window')
    if any(spans[k]+1e-5<required[k] for k in spans):
        raise ValueError('Accompaniment energy handoff is too short; select compatible phrases and a gradual band transfer instead of a core cut')
    return {'executed':True,'effective_10_90_shared_beats':spans,'minimum_10_90_shared_beats':required,
        'passed':True,'abrupt_percussion_explicitly_allowed':bool(allow_abrupt_percussion),'method':'actual accompaniment and estimated percussion gain progression after routing; separate from primary-attack timing',
        'limitations':'A minimum envelope span does not certify source phrasing, every residual figure or subjective quality'}
