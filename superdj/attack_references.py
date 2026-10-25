"""Compare rendered mixtures with native mixtures, retaining visible drum views."""
from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy.signal import butter,sosfiltfilt,resample_poly


def retain_initial_front_references(path,audio,warp,role,source,sample_rate):
    """Keep a natively observable short onset separate from its louder body.

    The 12 ms leading context and 12 ms post-context identify the same source
    event. Native-ambiguous short windows never become timing controls or
    required output observations. Audio and the existing long templates stay intact.
    """
    from .kick_grid import template_attack_rows
    original=source.get('attack_templates_path')
    if not original or not source.get('events'):return
    with np.load(original) as archive:templates=archive['templates']
    if templates.ndim!=2 or templates.shape[1]<192:return
    initial=templates[:,:192].copy()
    centres=[e['source_seconds']-warp['trim_start_seconds'] for e in source['events']]
    band=source.get('attack_band_hz',[100,1400])
    if band[0]<100 and band[1]<1000:
        # A 24 ms fragment of a bass-focused clock does not establish a
        # distinct leading click. Coherent earlier fronts are selected before
        # this stage; retain the bass views and explicit leading uncertainty.
        contract=source.setdefault('native_mixture_reference',{})
        contract['initial_front_visible_beats']=[]
        contract['initial_front_required_beats']=[]
        contract['initial_front_status']='no independently established leading family in the selected bass-focused band'
        return
    native=template_attack_rows(audio,sample_rate,centres,initial,minimum_correlation=.90,attack_band_hz=band)
    visible=[e['beat'] for e,t,row in zip(source['events'],centres,native)
        if row['eligible'] and abs(row['time']-t)<=.003 and row.get('alternative_peak_margin',0)>=.05]
    target=Path(path).with_name(Path(path).stem+f'-{role}-initial-front.npz')
    np.savez_compressed(target,templates=initial.astype('float32'))
    source['initial_front_attack_templates_path']=str(target)
    contract=source.setdefault('native_mixture_reference',{})
    contract['initial_front_visible_beats']=visible
    # Fix a distributed native support set before seeing any backend output.
    # All recognized fronts, including those outside this set, still enter
    # the maximum-error audit. Unrecognized output shapes remain explicit.
    candidates=[i for i,e in enumerate(source['events'])if e['beat'] in set(visible)]
    required=[]
    if candidates:
        times=np.asarray([source['events'][i]['source_seconds'] for i in candidates])
        boundaries=np.linspace(float(times.min()),float(times.max())+1e-8,min(8,len(candidates))+1)
        for first,last in zip(boundaries[:-1],boundaries[1:]):
            group=[i for i in candidates if first<=source['events'][i]['source_seconds']<last]
            if not group:continue
            def strength(i):
                short=float(np.mean(np.square(initial[i,96:],dtype='float64')))
                full=float(np.mean(np.square(templates[i],dtype='float64')))
                return short/max(full,1e-20)*native[i].get('alternative_peak_margin',0.)
            chosen=max(group,key=strength);required.append(source['events'][chosen]['beat'])
        if len(required)<min(8,len(candidates)):
            remaining=sorted((i for i in candidates if source['events'][i]['beat'] not in required),key=lambda i:native[i].get('alternative_peak_margin',0.),reverse=True)
            required.extend(source['events'][i]['beat']for i in remaining[:min(8,len(candidates))-len(required)])
    contract['initial_front_required_beats']=sorted(required)
    contract['initial_front_status']='native support set fixed before stretching; every recognized output front retains the unchanged timing bound'
    contract['visibility']['initial_front']=[{'beat':e['beat'],'recognized_in_native_mixture':e['beat'] in visible,
        'template_correlation':row.get('template_correlation'),'alternative_peak_margin':row.get('alternative_peak_margin'),
        'native_reference_delay_seconds':row['time']-t}for e,t,row in zip(source['events'],centres,native)]


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
        if source.get('evidence') not in ('estimated drums','original mixture') or not source.get('attack_templates_path') or not source.get('events'):continue
        events=source['events'];centres=[p['source_seconds']-warp['trim_start_seconds'] for p in events]
        contract={'comparison_signal':'the exact prepared native mixture fed to the stretcher',
            'estimated_drum_discovery_retained':source.get('evidence')=='estimated drums','audio_edited':False,
            'scope':'mixture context timing; masked isolated percussion remains uncertain',
            'primary_visible_beats':[],'bass_visible_beats':[],'visibility':{}}
        if source.get('evidence')=='original mixture':
            # These references already come from the native mixture. Calibrate
            # bass observability without labelling a second copy independent.
            if source.get('bass_attack_templates_path'):
                with np.load(source['bass_attack_templates_path']) as saved:templates=saved['templates']
                native=template_attack_rows(audio,sample_rate,centres,templates,
                    minimum_correlation=.90,attack_band_hz=(35,180))
                contract['mixture_bass_visible_beats']=[p['beat'] for p,t,row in zip(events,centres,native)
                    if row['eligible'] and abs(row['time']-t)<=.003]
                contract['visibility']['mixture_bass']=[{'beat':p['beat'],'recognized_in_native_mixture':bool(row['eligible']),
                    'template_correlation':row.get('template_correlation'),'alternative_peak_margin':row.get('alternative_peak_margin'),
                    'native_reference_delay_seconds':row['time']-t} for p,t,row in zip(events,centres,native)]
            source['native_mixture_reference']=contract
            retain_initial_front_references(path,audio,warp,role,source,sample_rate)
            continue
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
            if label=='bass':
                native=template_attack_rows(audio,sample_rate,centres,np.asarray(templates,dtype='float32'),
                    minimum_correlation=.90,attack_band_hz=band)
                contract['mixture_bass_visible_beats']=[p['beat'] for p,t,row in zip(events,centres,native)
                    if row['eligible'] and abs(row['time']-t)<=.003]
                contract['visibility']['mixture_bass']=[{'beat':p['beat'],'recognized_in_native_mixture':bool(row['eligible']),
                    'template_correlation':row.get('template_correlation'),'alternative_peak_margin':row.get('alternative_peak_margin'),
                    'native_reference_delay_seconds':row['time']-t} for p,t,row in zip(events,centres,native)]
            source['estimated_drum_'+key]=previous
            source[key]=str(target)
        contract['unresolved_estimated_bass_beats']=[p['beat'] for p in events if p['beat'] not in contract['bass_visible_beats']]
        source['native_mixture_reference']=contract
        retain_initial_front_references(path,audio,warp,role,source,sample_rate)
    return warp
