"""Native evidence is required before an estimated bass candidate controls time."""
import math
import numpy as np
import soundfile as sf
from scipy.signal import butter,resample_poly,sosfiltfilt


def bass_candidate_support(path,warp,candidates,drum_audio,drum_rate,source_base,*,role):
    """Classify source candidates before rendering, never from a failed output.

    A repeated shape in a separator estimate can be hidden or ambiguous in
    the actual native mixture. Such a candidate is unknown, not a certified
    primary kick and not permission to move that source point onto a beat.
    Known contradictory native matches still stop automatic source selection.
    """
    from .kick_grid import template_attack_rows
    from .timing_safety import SourceWindowFailure
    if not candidates:return [],[],{'executed':False,'reason':'no new estimated bass candidates'}
    if drum_audio.ndim!=2 or not len(drum_audio) or not np.isfinite(drum_audio).all():
        raise ValueError('Invalid/nonfinite estimated percussion input for native source evidence')
    stamps=[source_base+r['time']-warp['trim_start_seconds'] for r in candidates]
    # Use the same complete input and resampling origin as the retained native
    # references, so certification and later mandatory-view checks agree.
    with sf.SoundFile(path) as f:
        native=f.read(dtype='float32',always_2d=True);rate=f.samplerate
    if not len(native) or not np.isfinite(native).all():
        raise ValueError('Invalid/nonfinite native mixture for bass candidate evidence')
    centres=stamps
    def bass(audio,sr):
        divisor=math.gcd(sr,8000)
        mono=resample_poly(np.mean(audio,axis=1),8000//divisor,sr//divisor)
        return sosfiltfilt(butter(3,(35,180),btype='bandpass',fs=8000,output='sos'),mono)
    native_band=bass(native,rate);estimated_band=bass(drum_audio,drum_rate)
    def templates(signal,times):
        rows=[]
        for t in times:
            center=round(t*8000)
            if center<96 or center+360>len(signal):
                raise SourceWindowFailure('Native bass candidate lacks reference context',role=role)
            rows.append(signal[center-96:center+360])
        return np.asarray(rows,dtype='float32')
    native_refs=templates(native_band,centres)
    estimated_refs=templates(estimated_band,[r['time'] for r in candidates])
    baseline=template_attack_rows(native,rate,centres,native_refs,minimum_correlation=.90,attack_band_hz=(35,180))
    independent=template_attack_rows(native,rate,centres,estimated_refs,minimum_correlation=.90,attack_band_hz=(35,180))
    accepted=[];unknown=[];checks=[]
    for candidate,t,raw,estimate in zip(candidates,centres,baseline,independent):
        def view(row):
            return {'recognized':bool(row['eligible']),'delay_seconds':row['time']-t,
                    'correlation':row.get('template_correlation'),'alternative_peak_margin':row.get('alternative_peak_margin')}
        a,b=view(raw),view(estimate)
        point={'beat':candidate['index']+1,'source_seconds':source_base+candidate['time'],
               'native_mixture_reference':a,'native_estimated_percussion_reference':b}
        if any(v['recognized'] and abs(v['delay_seconds'])>.003 for v in (a,b)):
            raise SourceWindowFailure('Recognized native bass reference contradicts the estimated attack position',role=role,evidence=point)
        supported=a['recognized'] or b['recognized']
        checks.append({**point,'source_control_supported':supported})
        if supported:accepted.append(candidate)
        else:unknown.append({**candidate,'beat':candidate['index']+1,'source_seconds':source_base+candidate['time'],
            'reason':'estimated candidate not uniquely observable in the actual native mixture; no primary timing control authorized'})
    return accepted,unknown,{'executed':True,'scope':'native source evidence before the first stretch; no output-based reclassification',
        'comparison_signal':'exact prepared mixture fed to the pitch-preserving backend',
        'minimum_correlation':.90,'minimum_alternative_peak_margin':.02,'timing_tolerance_seconds':.003,
        'checks':checks,'unverified_is_not_percussion_absence':True,'audio_edited':False}


def required_bass_views(source):
    """Require every certified recovered hit in each observable native view."""
    recovered=set((source.get('low_frequency_gap_recovery') or {}).get('recovered_beats',[]))
    contract=source.get('native_mixture_reference') or {}
    body=recovered & set(contract['mixture_bass_visible_beats']) if 'mixture_bass_visible_beats' in contract else recovered
    estimate=recovered & set(contract.get('bass_visible_beats',[]))
    if 'mixture_bass_visible_beats' in contract and recovered-(body|estimate):
        raise ValueError('Certified recovered bass hit lacks an observable native reference')
    return {'mixture':body,'estimate':estimate}
