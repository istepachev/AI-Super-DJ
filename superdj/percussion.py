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
    from .execution import current
    shared_cache=current().root if current() else None
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

def verify_blended_fronts(audio,sample_rate,outgoing_warp,incoming_warp,gains,directory,*,ffmpeg='ffmpeg'):
    """Check identified source fronts in the actual blend; bound local repair."""
    from .kick_grid import template_attack_rows
    references=[];n=len(audio);pad=round(.2*sample_rate)
    for warp,role,gain_key in ((outgoing_warp or {},'tail','bass_out'),(incoming_warp or {},'head','bass_in')):
        source=warp.get('source_kick_alignment',{}).get(role,{})
        if not (source.get('attack_front_comparison') or {}).get('upper_front_selected'):continue
        mapping=warp['input_output_beat_mapping'][role];start=mapping[0]['output_seconds'];events=source['events'];centres=[e['output_seconds']-start for e in events]
        with np.load(source['attack_templates_path']) as f:templates=f['templates']
        audible=[i for i,t in enumerate(centres) if 0<=round(t*sample_rate)<n and gains[gain_key][round(t*sample_rate),0]>=.5]
        references.append((role,[centres[i] for i in audible],templates[audible],source['attack_band_hz']))
    if not references:return audio,{'executed':False,'reason':'no selected leading-front source family'}
    original=audio.copy();corrected=audio.copy();refinements=[];checks=[]
    for attempt in range(2):
        checks=[];residuals=[]
        for role,centres,templates,band in references:
            rows=template_attack_rows(np.pad(corrected,((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band)
            count=sum(r['eligible'] for r in rows);maximum=max((abs(r['time']-.2-t) for t,r in zip(centres,rows) if r['eligible']),default=None);fresh=None
            if count<min(8,len(centres)) or maximum is None or maximum>.003:
                estimates,_=final_drum_estimates(corrected,corrected,sample_rate,Path(directory)/f'view-{attempt}',ffmpeg=ffmpeg);fresh=template_attack_rows(np.pad(estimates['drums_out'],((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band)
                for i,(old,new) in enumerate(zip(rows,fresh)):
                    if new['eligible'] and (not old['eligible'] or (abs(old['time']-.2-centres[i])>.003 and new['template_correlation']>old.get('template_correlation',0)+.08)):rows[i]=new
            measured=[{'seconds':t,'delay_seconds':round(r['time']-.2-t,9),'correlation':r['template_correlation']} for t,r in zip(centres,rows) if r['eligible']]
            if not measured or len(measured)<min(8,len(centres)):raise ValueError('Combined percussion fronts remain unverified; review the source window')
            for row in measured:
                if abs(row['delay_seconds'])>.003:
                    if attempt or row['correlation']<.9 or abs(row['delay_seconds'])>.006:raise ValueError('Combined percussion attack misses the shared clock; review the rendered overlap')
                    residuals.append(row)
            checks.append({'role':role,'audible_source_events':len(centres),'recognized':measured,'fresh_percussion_view_used':fresh is not None,'maximum_error_seconds':max(abs(r['delay_seconds']) for r in measured)})
        if not residuals:break
        intervals=[]
        reference_times=sorted(set(t for _,centres,_,_ in references for t in centres))
        for row in residuals:
            distances=[abs(t-row['seconds']) for t in reference_times if abs(t-row['seconds'])>1e-5]
            support=min(.5,.95*min(distances)) if distances else .5
            support=min(support,.95*row['seconds'],.95*((n-1)/sample_rate-row['seconds']))
            center=round(row['seconds']*sample_rate);radius=round(support*sample_rate);first=center-radius;last=center+radius+1
            if first<0 or last>n:raise ValueError('Final front correction lacks endpoint context')
            if any(first<b and last>a for a,b in intervals):raise ValueError('Final front correction contexts overlap')
            intervals.append((first,last));t=np.arange(last-first);p=np.clip(1-abs(t-radius)/radius,0,1);shape=p*p*(3-2*p);indices=t+row['delay_seconds']*sample_rate*shape
            rate_change=float(np.max(abs(np.diff(indices)-1)))
            if np.any(np.diff(indices)<=0) or rate_change>=.03:raise ValueError('Final front correction exceeds gentle local rate bounds')
            block=original[first:last];corrected[first:last]=np.column_stack([np.interp(indices,t,block[:,ch]) for ch in (0,1)]).astype('float32');refinements.append({**row,'start_frame':first,'end_frame':last,'context_seconds':2*support,'maximum_local_rate_change':rate_change})
    return corrected,{'executed':True,'recognized_fronts_only':True,'checks':checks,'local_combined_waveform_refinements':refinements,'other_pcm_retained':True,'measurement_limit_seconds':.003}
