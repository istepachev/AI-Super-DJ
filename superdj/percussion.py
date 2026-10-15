"""Conditional percussion ownership on exact, final overlap PCM."""
from pathlib import Path
import hashlib
import numpy as np
import soundfile as sf

def needs_handoff(outgoing_warp,incoming_warp,*,style='default'):
    if style!='zhilin':return False
    return any((warp.get('source_kick_alignment',{}).get(role,{}).get('attack_front_comparison') or {}).get('upper_front_selected',False)
               or (warp.get('source_kick_alignment',{}).get(role,{}).get('low_frequency_gap_recovery') or {}).get('recovered_beats')
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

def verification_components(outgoing,incoming,sample_rate,gains,*,bass_hz=180.,drums_out=None,drums_in=None):
    """Independently reconstruct each known final-deck contribution.

    This permits source-identity checks at the crossover, where two different
    correctly aligned kicks can mask one another in a single template view.
    It does not separate unknown audio or replace the raw-mixture checks.
    """
    from .mixing import split_bands
    result={}
    if (drums_out is None)!=(drums_in is None):raise ValueError('Verification requires both drum contributions')
    for role,pcm,drums in [('tail',outgoing,drums_out),('head',incoming,drums_in)]:
        side='out' if role=='tail' else 'in'
        value=sum(band*gains[name+'_'+side] for name,band in zip(('bass','mid','high'),split_bands(pcm,sample_rate,bass_hz)))
        if drums is not None:
            value+=sum(band*(gains['bass_'+side]-gains[name+'_'+side]) for name,band in zip(('bass','mid','high'),split_bands(drums,sample_rate,bass_hz)))
        result[role]=value.astype('float32')
    return result


def verify_blended_fronts(audio,sample_rate,outgoing_warp,incoming_warp,gains,directory,*,ffmpeg='ffmpeg',components=None):
    """Check identified source fronts in the actual blend; bound local repair."""
    from .kick_grid import template_attack_rows
    references=[];n=len(audio);pad=round(.2*sample_rate)
    for warp,role,gain_key in ((outgoing_warp or {},'tail','bass_out'),(incoming_warp or {},'head','bass_in')):
        source=warp.get('source_kick_alignment',{}).get(role,{})
        if not source.get('attack_templates_path') or not source.get('events'):continue
        mapping=warp['input_output_beat_mapping'][role];start=mapping[0]['output_seconds'];events=source['events'];centres=[e['output_seconds']-start for e in events]
        with np.load(source['attack_templates_path']) as f:templates=f['templates']
        audible=[i for i,t in enumerate(centres) if 0<=round(t*sample_rate)<n and gains[gain_key][round(t*sample_rate),0]>=.5]
        if not audible:continue
        references.append((role,[centres[i] for i in audible],templates[audible],source['attack_band_hz'],set()))
        if source.get('bass_attack_templates_path'):
            protected=warp.get('native_attack_preservation',{}).get('bass_mixture_attack_templates',{}).get(role)
            with np.load(protected or source['bass_attack_templates_path']) as f:low=f['templates']
            recovered=set((source.get('low_frequency_gap_recovery') or {}).get('recovered_beats',[]))
            required={centres[i] for i in audible if events[i]['beat'] in recovered}
            references.append((role+':bass_body',[centres[i] for i in audible],low[audible],[35,180],required))
    if not references:return audio,{'executed':False,'reason':'no audible recognized source attack family'}
    if components is not None:
        if set(components)!={'head','tail'} or any(p.shape!=audio.shape or not np.isfinite(p).all() for p in components.values()):raise ValueError('Invalid final-deck verification components')
        if np.max(abs(components['head']+components['tail']-audio))>2e-6:raise ValueError('Verification components do not reconstruct actual mixed PCM')
    original_components={k:v.copy() for k,v in (components or {}).items()}
    current_components={k:v.copy() for k,v in original_components.items()}
    original=audio.copy();corrected=audio.copy();refinements=[];checks=[]
    for attempt in range(2):
        checks=[];residuals=[]
        for role,centres,templates,band,required in references:
            minimum=.90 if band==[35,180] else .65
            rows=template_attack_rows(np.pad(corrected,((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band,minimum_correlation=minimum)
            raw_rows=[dict(r) for r in rows];component_used=[]
            if current_components:
                other='head' if role.startswith('tail') else 'tail'
                isolated=corrected-current_components[other]
                recovered=template_attack_rows(np.pad(isolated,((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band,minimum_correlation=minimum)
                for i,(old,new) in enumerate(zip(rows,recovered)):
                    # Keep strong contradictory raw timing evidence. A weak
                    # match to the other kick is not evidence of this source.
                    if old['eligible'] and old.get('template_correlation',0)>=.90 and abs(old['time']-.2-centres[i])>.003:continue
                    if new['eligible'] and (not old['eligible'] or new['template_correlation']>old.get('template_correlation',0)+.08):
                        rows[i]=new;component_used.append(centres[i])
            count=sum(r['eligible'] for r in rows);maximum=max((abs(r['time']-.2-t) for t,r in zip(centres,rows) if r['eligible']),default=None);fresh=None
            if count<min(8,len(centres)) or maximum is None or maximum>.003:
                estimates,_=final_drum_estimates(corrected,corrected,sample_rate,Path(directory)/f'view-{attempt}',ffmpeg=ffmpeg);fresh=template_attack_rows(np.pad(estimates['drums_out'],((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band,minimum_correlation=minimum)
                for i,(old,new) in enumerate(zip(rows,fresh)):
                    if new['eligible'] and (not old['eligible'] or (abs(old['time']-.2-centres[i])>.003 and new['template_correlation']>old.get('template_correlation',0)+.08)):rows[i]=new
            measured=[{'seconds':t,'delay_seconds':round(r['time']-.2-t,9),'correlation':r['template_correlation']} for t,r in zip(centres,rows) if r['eligible']]
            if required-{r['seconds'] for r in measured}:raise ValueError('Recovered bass attacks remain unverified in the actual blend; review the source window')
            if not measured or len(measured)<min(8,len(centres)):raise ValueError('Combined percussion fronts remain unverified; review the source window')
            for row in measured:
                if abs(row['delay_seconds'])>.003:
                    if attempt or row['correlation']<.9 or abs(row['delay_seconds'])>.006:raise ValueError('Combined percussion attack misses the shared clock; review the rendered overlap')
                    residuals.append(row)
            checks.append({'role':role,'audible_source_events':len(centres),'recognized':measured,'unverified_source_seconds':[t for t,r in zip(centres,rows) if not r['eligible']],'attack_band_hz':band,'fresh_percussion_view_used':fresh is not None,'known_other_deck_subtraction_seconds':component_used,'raw_mixture_unverified_seconds':[t for t,r in zip(centres,raw_rows) if not r['eligible']],'maximum_error_seconds':max(abs(r['delay_seconds']) for r in measured)})
        if not residuals:break
        intervals=[]
        reference_times=sorted(set(t for _,centres,_,_,_ in references for t in centres))
        grouped={}
        for row in residuals:grouped.setdefault(round(row['seconds']*sample_rate),[]).append(row)
        residuals=[]
        for points in grouped.values():
            delays=[p['delay_seconds'] for p in points]
            if max(delays)-min(delays)>.003:raise ValueError('Front and bass evidence disagree; review the source window')
            residuals.append({**points[0],'delay_seconds':(min(delays)+max(delays))/2})
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
            for key,value in original_components.items():current_components[key][first:last]=np.column_stack([np.interp(indices,t,value[first:last,ch]) for ch in (0,1)]).astype('float32')
    return corrected,{'executed':True,'recognized_fronts_only':True,'checks':checks,'local_combined_waveform_refinements':refinements,'other_pcm_retained':True,'measurement_limit_seconds':.003}
