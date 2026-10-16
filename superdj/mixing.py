"""Independent frequency envelopes with a gentle, uncompensated mid entry."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def smooth(value):
    x = np.clip(value, 0, 1)
    return x * x * (3 - 2 * x)


def split_bands(audio, sample_rate, bass_hz=180.0, high_hz=3500.0):
    """Complementary zero-phase decomposition; bands sum to the source."""
    low = sosfiltfilt(butter(4, bass_hz, fs=sample_rate, output="sos"), audio, axis=0)
    below_high = sosfiltfilt(butter(4, high_hz, fs=sample_rate, output="sos"), audio, axis=0)
    return low, below_high - low, audio - below_high


def envelopes(n, beats, *, tonal_conflict=False, beat_phase=None, style='default', profile=None,
              mid_handoff_bars=None, bass_handoff_beats=None):
    """Complementary per-band transfer, with a brief soft mid handoff.

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
    if style!='zhilin' and (mid_handoff_bars is not None or bass_handoff_beats is not None):
        raise ValueError('Per-transition handoff overrides require DJ Zhilin')
    if style=='zhilin':
        from .zhilin import reference_profile
        reference=reference_profile(profile) if not isinstance(profile,dict) else profile
        grid=np.asarray(reference['phase'])
        mid=np.interp(p[:,0],grid,reference['mid_in'])[:,None]
        high=np.interp(p[:,0],grid,reference['high_in'])[:,None]
        start,end=reference['bass_transfer_fraction']
        if bass_handoff_beats is not None:
            center=(start+end)/2;width=bass_handoff_beats/beats
            start=max(0.,min(center-width/2,1-width));end=start+width
        bass=np.interp(np.clip((p[:,0]-start)/(end-start),0,1),grid,reference['bass_shape'])[:,None]
        mid_fraction=[0.,1.]
        if mid_handoff_bars is not None and mid_handoff_bars<beats/4:
            width=mid_handoff_bars*4/beats
            start_mid=max(0.,min(.55-width/2,1-width));end_mid=start_mid+width
            mid=smooth((p-start_mid)/width);mid_fraction=[start_mid,end_mid]
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
            'foreground_conflict_policy':'Complementary accompaniment; separate aligned vocal handoff; dense instrumental conflicts need cue review'}
    mid_beats = min(beats * .4, 8 if tonal_conflict else 16)
    mid_width = mid_beats / beats
    mid = smooth((p - (.43 - mid_width / 2)) / mid_width)
    bass_beats = min(4, beats * .2)
    bass = smooth((p - (.55 - bass_beats / beats / 2)) / (bass_beats / beats))
    high = smooth(p)
    quiet_db = 2 if tonal_conflict else 1
    quiet = 10 ** (-quiet_db * (1 - mid) / 20)
    gains = {'bass_out': 1 - bass, 'bass_in': bass,
             'mid_out': 1 - mid, 'mid_in': mid * quiet,
             'high_out': 1 - high, 'high_in': high}
    report = {'incoming_mid_opening_beats': mid_beats,
              'incoming_initial_attenuation_db': quiet_db,
              'bass_handoff_beats': bass_beats,
              'mid_handoff_fraction': [.43 - mid_width / 2, .43 + mid_width / 2],
              'bass_handoff_fraction': [.55 - bass_beats / beats / 2, .55 + bass_beats / beats / 2],
              'high_handoff_fraction': [0., 1.],
              'band_pair_gain_sum_maximum': {band: float(np.max(gains[band + '_out'] + gains[band + '_in']))
                                             for band in ('bass', 'mid', 'high')},
              'phase_source': 'shared integrated clock' if beat_phase is not None else 'linear phase'}
    return gains, report


def blend(outgoing, incoming, sample_rate, *, bass_hz=180.0,
          beats=64, tonal_conflict=False, beat_phase=None, style='default', profile=None,
          mid_handoff_bars=None, bass_handoff_beats=None,drums_out=None,drums_in=None,percussion_policy='shared',accompaniment_policy='shared'):
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
                             mid_handoff_bars=mid_handoff_bars,bass_handoff_beats=bass_handoff_beats)
    if percussion_policy!='shared' and drums_out is None:raise ValueError('Sequential percussion requires both final drum estimates')
    from .percussion import percussion_gains
    gains=percussion_gains(gains,beat_phase,beats,percussion_policy)
    from .accompaniment import accompaniment_gains
    gains=accompaniment_gains(gains,accompaniment_policy)
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
        if percussion_policy=='sequential':
            first=np.flatnonzero(gains['drums_out'][:,0]<1)[0];last=np.flatnonzero(gains['drums_in'][:,0]>=1)[0]
            percussion.update(method='Sequential estimated percussion ownership for unresolved leading fronts or displaced native figures; harmonic band curves remain unchanged',
                handoff_fraction=[float(percussion_phase[first]),float(percussion_phase[last])],
                handoff_beats=float((percussion_phase[last]-percussion_phase[first])*beats))
    report = {"method": "three complementary zero-phase bands with independent bass/mid/high envelopes",
              "bass_hz": bass_hz, "high_hz": 3500., "beats": beats,
              **curves, "incoming_mid_start_multiplier": 0.,
              "tonal_conflict_risk": bool(tonal_conflict), "vocal_overlap_certified_absent": False,
              "entry_gain_compensated": False,
              "endpoint_max_error": float(max(np.max(np.abs(mixed[0] - outgoing[0])), np.max(np.abs(mixed[-1] - incoming[-1]))))}
    if percussion:report['percussion_processing']=percussion
    report['accompaniment_ownership_policy']=accompaniment_policy
    if accompaniment_policy!='shared':
        report['residual_bass_mid_gain_product_max']={band:float(np.max(gains[band+'_out']*gains[band+'_in'])) for band in ('bass','mid')}
        report['accompaniment_handoff_note']='Bass and middle residual figures follow the single percussion handoff; the approved overlap, high-band curves and deck timing remain intact'
    return mixed.astype(np.float32), report
