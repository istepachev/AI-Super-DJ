"""Conditional percussion ownership on exact, final overlap PCM."""
from pathlib import Path
import hashlib
import numpy as np
import soundfile as sf

def needs_handoff(outgoing_warp,incoming_warp,*,style='default'):
    if style!='zhilin':return False
    return any((warp.get('source_kick_alignment',{}).get(role,{}).get('attack_front_comparison') or {}).get('upper_front_selected',False)
               for warp,role in ((outgoing_warp or {},'tail'),(incoming_warp or {},'head')))

def final_drum_estimates(outgoing,incoming,sample_rate,directory,*,ffmpeg='ffmpeg'):
    from .stems import configuration,separate
    config=configuration()
    if config is None:raise RuntimeError('Percussion handoff requires the configured local Demucs runtime')
    from .cues import runtime_config
    shared_cache=runtime_config().get('cache')
    cache=Path(shared_cache)/'final-percussion' if shared_cache else Path(directory)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);estimates=[];reports=[]
    for role,pcm in (('out',outgoing),('in',incoming)):
        if pcm.ndim!=2 or pcm.shape[1]!=2 or not np.isfinite(pcm).all():raise ValueError('Invalid final deck PCM')
        digest=hashlib.sha256(str(sample_rate).encode()+pcm.astype('float32').tobytes()).hexdigest();source=directory/f'{role}-{digest}.wav'
        if not source.exists():sf.write(source,pcm,sample_rate,subtype='FLOAT')
        estimate=separate(source,digest,cache,config,ffmpeg);drums,rate=sf.read(estimate['drums_path'],dtype='float32',always_2d=True)
        if rate!=sample_rate:
            from scipy.signal import resample_poly
            import math
            divisor=math.gcd(rate,sample_rate);drums=resample_poly(drums,sample_rate//divisor,rate//divisor,axis=0)
        if len(drums)>len(pcm):drums=drums[:len(pcm)]
        if drums.shape!=pcm.shape or not np.isfinite(drums).all():raise ValueError('Final drum estimate changed deck geometry')
        estimates.append(drums);reports.append({'role':role,'final_deck_pcm_sha256':digest,'estimate_path':estimate['drums_path'],'provenance':estimate['provenance'],'estimated_not_ground_truth':True})
    return {'drums_out':estimates[0],'drums_in':estimates[1]},reports
