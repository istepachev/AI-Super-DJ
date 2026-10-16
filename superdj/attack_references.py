"""Compare rendered mixtures with native mixtures, retaining visible drum views."""
from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy.signal import butter,sosfiltfilt,resample_poly


def retain_mixture_references(path,warp,sample_rate):
    """Change measurement references only; never edit audio or attack positions.

    A separator estimate is useful for discovery. It is not the waveform fed
    to the stretcher. Its signature can already be masked in the native mix.
    Keep independently visible signatures as additional contradiction checks.
    """
    from .kick_grid import template_attack_rows
    with sf.SoundFile(path) as source:
        if source.samplerate!=sample_rate:raise ValueError('Native mixture reference rate mismatch')
        audio=source.read(dtype='float32',always_2d=True)
    divisor=math.gcd(sample_rate,8000)
    mono=resample_poly(audio.mean(axis=1),8000//divisor,sample_rate//divisor)
    filtered={}
    for role,source in warp.get('source_kick_alignment',{}).items():
        if source.get('evidence')!='estimated drums' or not source.get('attack_templates_path') or not source.get('events'):continue
        events=source['events'];centres=[p['source_seconds']-warp['trim_start_seconds'] for p in events]
        contract={'comparison_signal':'the exact prepared native mixture fed to the stretcher',
            'estimated_drum_discovery_retained':True,'audio_edited':False,
            'scope':'mixture context timing; masked isolated percussion remains uncertain',
            'primary_visible_beats':[],'bass_visible_beats':[],'visibility':{}}
        for label,key,band,minimum in [('primary','attack_templates_path',source.get('attack_band_hz',[100,1400]),.65),
                                      ('bass','bass_attack_templates_path',[35,180],.90)]:
            if not source.get(key):continue
            previous=source[key]
            with np.load(previous) as archive:drum_templates=archive['templates']
            baseline=template_attack_rows(audio,sample_rate,centres,drum_templates,
                minimum_correlation=minimum,attack_band_hz=band)
            contract['visibility'][label]=[{'beat':p['beat'],'recognized_in_native_mixture':bool(row['eligible']),
                'template_correlation':row['template_correlation'],'alternative_peak_margin':row['alternative_peak_margin'],
                'native_reference_delay_seconds':row['time']-t}
                for p,t,row in zip(events,centres,baseline)]
            contract[label+'_visible_beats']=[p['beat'] for p,t,row in zip(events,centres,baseline)
                if row['eligible'] and abs(row['time']-t)<=.003]
            frequency=tuple(band)
            if frequency not in filtered:filtered[frequency]=sosfiltfilt(butter(3,band,btype='bandpass',fs=8000,output='sos'),mono)
            signal=filtered[frequency];templates=[]
            for t in centres:
                center=round(t*8000)
                if center<96 or center+360>len(signal):raise ValueError('Native mixture attack reference lacks source context')
                templates.append(signal[center-96:center+360])
            target=Path(path).with_name(Path(path).stem+f'-{role}-native-mixture-{label}.npz')
            np.savez_compressed(target,templates=np.asarray(templates,dtype='float32'))
            source['estimated_drum_'+key]=previous
            source[key]=str(target)
        contract['unresolved_estimated_bass_beats']=[p['beat'] for p in events if p['beat'] not in contract['bass_visible_beats']]
        source['native_mixture_reference']=contract
    return warp
