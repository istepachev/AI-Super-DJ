"""Refine coarse beat labels against repeated percussive attacks, then audit PCM.

This estimates attack time, not waveform phase or a calibrated kick probability.
Unrecognised beats remain explicit gaps in coverage.
"""
from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly,correlate
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import map_coordinates
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom

def attack_events(audio, sample_rate, centres, search_seconds=.12,attack_band_hz=(100,1400)):
    rate=8000
    divisor=math.gcd(sample_rate,rate)
    mono=resample_poly(np.mean(audio,axis=1),rate//divisor,sample_rate//divisor)
    filtered=sosfiltfilt(butter(3,attack_band_hz,btype='bandpass',fs=rate,output='sos'),mono)
    energy=uniform_filter1d(filtered**2,max(1,round(.002*rate)))
    lag=round(.008*rate)
    novelty=np.maximum(energy-np.r_[np.zeros(lag),energy[:-lag]],0)
    low=sosfiltfilt(butter(3,(35,180),btype='bandpass',fs=rate,output='sos'),mono)
    low_energy=uniform_filter1d(low**2,round(.010*rate))
    rows=[]
    for i,centre in enumerate(centres):
        lo=max(0,round((centre-search_seconds)*rate));hi=min(len(mono),round((centre+search_seconds)*rate))
        if hi-lo<8:continue
        peak=lo+int(np.argmax(novelty[lo:hi]));value=float(novelty[peak]);onset=peak
        while onset>lo and novelty[onset-1]>.2*value:onset-=1
        ratio=value/max(float(np.percentile(novelty[lo:hi],75)),1e-20)
        low_power=float(np.max(low_energy[lo:hi]))
        rows.append({'index':i,'time':onset/rate,'centre':float(centre),'peak_time':peak/rate,'strength':value,
                     'prominence':ratio,'low_power':low_power,'window_edge':onset<=lo+2 or peak>=hi-2})
    reference=float(np.percentile([r['strength'] for r in rows],70)) if rows else 0
    low_reference=float(np.percentile([r['low_power'] for r in rows],70)) if rows else 0
    for r in rows:
        r['eligible']=bool(reference>1e-10 and not r['window_edge'] and r['prominence']>=6
                           and r['strength']>=.15*reference and r['low_power']>=.05*low_reference)
    return rows

def repeated_attacks(rows, minimum_fraction=.10,tolerance_seconds=.010):
    supported=[r for r in rows if r['eligible']]
    if len(supported)<8:return []
    indices=np.array([r['index'] for r in supported],dtype=float)
    times=np.array([r['time'] for r in supported])
    # Reject a nearby snare/hat that wins one local peak search. Fit audio
    # attack time, not the quantised neural beat times.
    # Sparse/half-time kicks and a competing snare family must not pull the
    # initial least-squares line halfway between two real attack sequences.
    # Select a repeatable consensus first; excluded events stay unverified.
    initial=np.polyfit(indices,times,1)
    proposals=[initial]
    rng=np.random.default_rng(0)
    for _ in range(256):
        a,b=rng.choice(len(times),2,replace=False)
        if abs(indices[b]-indices[a])<max(4,(indices.max()-indices.min())/3):continue
        slope=(times[b]-times[a])/(indices[b]-indices[a])
        proposals.append(np.array([slope,times[a]-slope*indices[a]]))
    best=None
    for proposal in proposals:
        residual=np.abs(times-np.polyval(proposal,indices));mask=residual<=tolerance_seconds
        score=(int(mask.sum()),-float(np.sum(residual[mask])))
        if best is None or score>best[0]:best=(score,mask)
    keep=best[1]
    if keep.sum()<8:return []
    for _ in range(4):
        line=np.polyfit(indices[keep],times[keep],1)
        residual=times-np.polyval(line,indices)
        new=np.abs(residual-np.median(residual[keep]))<=tolerance_seconds
        if new.sum()<8:break
        keep=new
    result=[r for r,k in zip(supported,keep) if k]
    # A sparse kick pattern may occupy only every fourth grid beat. Require
    # substantial agreement among eligible attacks rather than pretending
    # that every quarter note contains a kick. Random offset peaks do not
    # form a sufficiently dominant family.
    # Uniform offsets in the +/-120 ms source window would hit a +/-10 ms
    # line by chance about 1/12 of the time. Account conservatively for all
    # consensus proposals; this is a rejection heuristic, not a calibrated
    # musical confidence score. Sparse real bar-level kicks can still qualify.
    chance=float(binom.sf(len(result)-1,len(supported),min(.5,tolerance_seconds/.12)))*len(proposals)
    return result if len(result)>=max(minimum_fraction*len(rows),8) and chance<1e-4 else []

def isolated_events(audio,sample_rate,centres):
    """Preserve the temporal centre of a genuinely isolated sharp impulse."""
    envelope=uniform_filter1d(np.max(np.abs(audio),axis=1),max(1,round(.0005*sample_rate)))
    rows=[]
    for index,centre in enumerate(centres):
        first=max(0,round((centre-.035)*sample_rate));last=min(len(envelope),round((centre+.035)*sample_rate)+1)
        region=envelope[first:last]
        if not len(region):continue
        peak=int(np.argmax(region));value=float(region[peak]);outer=region[np.abs(np.arange(len(region))-peak)>round(.010*sample_rate)]
        if len(outer) and value>1e-5 and value>4*float(np.mean(region)) and value>3*float(np.max(outer)):
            rows.append({'index':index,'time':(first+peak)/sample_rate})
    return rows


def repeated_waveform_family(audio,sample_rate,rows,attack_band_hz=(100,1400)):
    """Optional evidence for a variable live pulse in a trusted native meter.

    Require eight closely repeated attack shapes and substantial source
    coverage. This does not infer a constant BPM or certify the supplied meter.
    Callers must explicitly authorize this evidence path for a reviewed grid.
    """
    candidates=[r for r in rows if r['eligible']]
    if len(candidates)<8:return []
    divisor=math.gcd(sample_rate,8000);band=resample_poly(np.mean(audio,axis=1),8000//divisor,sample_rate//divisor)
    band=sosfiltfilt(butter(3,attack_band_hz,btype='bandpass',fs=8000,output='sos'),band)
    candidates=[r for r in candidates if round(r['time']*8000)>=96 and round(r['time']*8000)+360<=len(band)]
    if len(candidates)<8:return []
    templates=np.asarray([band[round(r['time']*8000)-96:round(r['time']*8000)+360] for r in candidates])
    templates=templates/np.sqrt(np.maximum(np.sum(templates**2,axis=1),1e-30))[:,None]
    similarities=templates@templates.T;column=int(np.argmax(np.sum(similarities>=.85,axis=0)));family=[r for r,v in zip(candidates,similarities[:,column]) if v>=.85]
    return family if len(family)>=max(8,.35*len(candidates)) else []


def augment_source_attacks(audio,sample_rate,centres,accepted,attack_band_hz=(100,1400)):
    """Find an established primary shape behind a louder nearby fill.

    Search the native beat vicinity with same-source attack references. Missing
    or ambiguous matches remain missing; neither a rhythm fill nor a bass tone
    is promoted just because a beat-model point exists.
    """
    if len(accepted)<8:return accepted
    divisor=math.gcd(sample_rate,8000);band=resample_poly(np.mean(audio,axis=1),8000//divisor,sample_rate//divisor)
    band=sosfiltfilt(butter(3,attack_band_hz,btype='bandpass',fs=8000,output='sos'),band)
    references=[]
    for r in accepted:
        centre=round(r['time']*8000)
        if centre>=96 and centre+360<=len(band):references.append((r,band[centre-96:centre+360]))
    if len(references)<8:return accepted
    reference_power=float(np.median([np.sum(t*t) for _,t in references]));known={r['index'] for r in accepted};result=list(accepted)
    for index,centre in enumerate(centres):
        if index in known:continue
        target=round(centre*8000);first=target-96-960;last=target+360+960
        if first<0 or last>len(band):continue
        segment=band[first:last];power=np.convolve(segment**2,np.ones(456),mode='valid');best=None
        for reference,template in references:
            scores=correlate(segment,template,mode='valid',method='direct')/np.sqrt(np.maximum(power*np.sum(template**2),1e-30));peak=int(np.argmax(scores));outside=np.abs(np.arange(len(scores))-peak)>24;margin=float(scores[peak]-np.max(scores[outside]))
            if scores[peak]<.90 or margin<.05 or power[peak]<.01*reference_power or not 2<peak<len(scores)-3:continue
            if best is None or scores[peak]>best['source_template_correlation']:
                best={'index':index,'time':(first+peak+96)/8000,'eligible':True,'source_template_correlation':float(scores[peak]),'source_template_margin':margin,'reference_index':reference['index'],'recognition':'established source attack waveform behind a stronger different transient'}
        if best:result.append(best)
    return sorted(result,key=lambda r:r['index'])

def refine_source_controls(path,warp,sample_rate,evidence=None):
    from copy import deepcopy
    from .timing_safety import initialize_rate_reference,audit_warp_rates,classify_native_phase,SourceWindowFailure
    initialize_rate_reference(warp,sample_rate)
    original_anchors=deepcopy(warp['anchors'])
    reports={}
    for role in ('head','tail'):
        mapping=warp['input_output_beat_mapping'][role]
        if len(mapping)<10 or not any(p['role']==role+'_sync_control' for p in warp['anchors']):continue
        start=mapping[0]['source_seconds'];end=mapping[-1]['source_seconds']
        context=.2
        stem=(evidence or {}).get(role)
        input_path=Path(stem['path']) if stem else Path(path)
        offset=stem.get('source_start_seconds',0) if stem else warp['trim_start_seconds']
        with sf.SoundFile(input_path) as f:
            first=max(0,round((start-offset-context)*f.samplerate))
            last=min(f.frames,round((end-offset+context)*f.samplerate))
            f.seek(first);audio=f.read(last-first,dtype='float32',always_2d=True);rate=f.samplerate
        base=offset+first/rate
        centres=[p['source_seconds']-base for p in mapping[1:-1]]
        impulses=isolated_events(audio,rate,centres)
        isolated=len(impulses)>=.9*len(centres)
        band_hz=tuple((stem or {}).get('attack_band_hz',(100,1400)));comparison=None
        rows=attack_events(audio,rate,centres,attack_band_hz=band_hz);accepted=impulses if isolated else repeated_attacks(rows)
        phase_references=list(accepted)
        if not isolated:
            residual_rows=[{**r,'time':r['time']-r['centre']} for r in rows]
            native_family=repeated_attacks(residual_rows,tolerance_seconds=.0125)
            if len(native_family)>len(accepted):
                accepted=[rows[r['index']] for r in native_family]
            if not accepted and stem and stem.get('allow_variable_attack_family'):
                accepted=repeated_waveform_family(audio,rate,rows,attack_band_hz=band_hz)
            if stem and 'attack_band_hz' not in stem and len(accepted)>=8:
                # A bass-heavy 100-180 Hz body can win the old novelty search
                # after the audible click has already happened. Independently
                # locate a coherent upper attack family instead of calling
                # body-template coincidence a complete attack SYNC check.
                upper_rows=attack_events(audio,rate,centres,attack_band_hz=(180,1200))
                upper=repeated_attacks(upper_rows)
                upper_native=repeated_attacks([{**r,'time':r['time']-r['centre']} for r in upper_rows],tolerance_seconds=.0125)
                if len(upper_native)>len(upper):upper=[upper_rows[r['index']] for r in upper_native]
                by_index={r['index']:r for r in upper};pairs=[r['time']-by_index[r['index']]['time'] for r in accepted if r['index'] in by_index]
                if len(pairs)>=8:
                    median=float(np.median(pairs));spread=float(np.percentile(pairs,90)-np.percentile(pairs,10))
                    selected=.008<median<.085 and spread<.018
                    comparison={'matched_source_events':len(pairs),'body_minus_upper_median_seconds':median,'body_minus_upper_10_90_spread_seconds':spread,'upper_front_selected':selected}
                    if selected:band_hz=(180,1200);rows=upper_rows;accepted=upper
            phase_references=list(accepted)
            accepted=augment_source_attacks(audio,rate,centres,accepted,attack_band_hz=band_hz)
        # An upper click family can miss a real kick behind a different fill.
        # Recover only the already-established same-source waveform in bass;
        # do not refit a straight line that would discard these events again.
        bass_recovered=[];unverified_bass=[];native_bass_proof={'executed':False,'reason':'no new estimated bass candidates'}
        if stem and not isolated and len(accepted)>=8:
            known={r['index'] for r in accepted}
            recovered=augment_source_attacks(audio,rate,centres,accepted,attack_band_hz=(35,180))
            bass_recovered=[r for r in recovered if r['index'] not in known]
            from .source_observability import bass_candidate_support
            bass_recovered,unverified_bass,native_bass_proof=bass_candidate_support(
                path,warp,bass_recovered,audio,rate,base,role=role)
            supported={r['index'] for r in bass_recovered}
            accepted=[r for r in recovered if r['index'] in known or r['index'] in supported]
        displaced=[]
        if not isolated and not (stem or {}).get('allow_variable_attack_family'):
            try:accepted,displaced=classify_native_phase(accepted,centres,phase_references)
            except SourceWindowFailure as error:
                error.role=role
                raise
        retained={r['index'] for r in accepted}
        bass_recovered=[r for r in bass_recovered if r['index'] in retained]
        reports[role]={'method':'repeated band-limited attack onset; 20% novelty rise, not amplitude/body maximum',
                       'evidence':'estimated drums' if stem else 'original mixture',
                       'detected':len(rows),'supported':len(accepted),'unsupported':len(rows)-len(accepted),'events':[]}
        reports[role]['attack_band_hz']=list(band_hz)
        reports[role]['native_rhythm_observations']=[{**r,'beat':r['index']+1,'source_seconds':base+r['time']} for r in displaced]
        reports[role]['native_phase_policy']='Waveform identity alone cannot assign a fill/ghost to a primary beat; displaced figures remain at native source timing'
        reports[role]['unverified_native_bass_candidates']=unverified_bass
        if comparison:reports[role]['attack_front_comparison']=comparison
        reports[role]['low_frequency_gap_recovery']={'executed':bool(stem and not isolated),
            'recovered_source_events':len(bass_recovered),'attack_band_hz':[35,180],
            'recovered_beats':[r['index']+1 for r in bass_recovered],
            'native_source_observability':native_bass_proof,
            'method':'same-source waveform recognition; correlation >=0.90, alternative margin >=0.05; no grid-only events'}
        if stem and stem.get('allow_variable_attack_family'):
            reports[role]['variable_native_meter_evidence_authorized']=True
        reports[role]['attack_kind']='isolated_peak' if isolated else 'percussive_onset'
        if not accepted:continue
        actual_by_index={r['index']+1:base+r['time'] for r in accepted}
        known_indices=sorted(actual_by_index)
        known_offsets=[actual_by_index[i]-mapping[i]['source_seconds'] for i in known_indices]
        times=[start];offsets=[0.]
        for index,point in enumerate(mapping[1:-1],1):
            actual=actual_by_index.get(index,point['source_seconds']+float(np.interp(index,known_indices,known_offsets)))
            times.append(point['source_seconds']);offsets.append(actual-point['source_seconds'])
        for row in accepted:
            point=mapping[row['index']+1]
            actual=base+row['time'];old=point['source_seconds']
            reports[role]['events'].append({'beat':point['beat'],'source_seconds':actual,'nominal_source_seconds':old,
                                            'output_seconds':point['output_seconds'],'offset_seconds':actual-old,
                                            **{k:row[k] for k in ('source_template_correlation','source_template_margin','recognition') if k in row}})
        if accepted and not isolated:
            divisor=math.gcd(rate,8000)
            band=resample_poly(np.mean(audio,axis=1),8000//divisor,rate//divisor)
            band=sosfiltfilt(butter(3,band_hz,btype='bandpass',fs=8000,output='sos'),band)
            templates=[]
            for row in accepted:
                centre=round(row['time']*8000);first=centre-96;last=centre+360
                if first<0 or last>len(band):raise SourceWindowFailure('Source attack template lacks context',role=role)
                templates.append(band[first:last])
            template_path=Path(path).with_name(Path(path).stem+f'-{role}-attack-templates.npz')
            np.savez_compressed(template_path,templates=np.asarray(templates,dtype='float32'))
            reports[role]['attack_templates_path']=str(template_path)
            # Independently retain the bass-body view of every recognized
            # source attack, including those recovered behind stronger fills.
            bass=sosfiltfilt(butter(3,(35,180),btype='bandpass',fs=8000,output='sos'),resample_poly(np.mean(audio,axis=1),8000//divisor,rate//divisor))
            low_templates=[bass[round(row['time']*8000)-96:round(row['time']*8000)+360] for row in accepted]
            low_path=Path(path).with_name(Path(path).stem+f'-{role}-bass-attack-templates.npz')
            np.savez_compressed(low_path,templates=np.asarray(low_templates,dtype='float32'))
            reports[role]['bass_attack_templates_path']=str(low_path)
        times.append(end);offsets.append(0.)
        reports[role]['inferred_grid_points']=len(mapping)-2-len(accepted)
        for anchor in warp['anchors']:
            if anchor['role']!=role+'_sync_control':continue
            old=anchor['source_seconds'];delta=float(np.interp(old,times,offsets))
            anchor['nominal_source_frame']=anchor['source_frame']
            anchor['source_frame']+=round(delta*sample_rate)
            anchor['source_seconds']=warp['trim_start_seconds']+anchor['source_frame']/sample_rate
    if any(b['source_frame']<=a['source_frame'] for a,b in zip(warp['anchors'],warp['anchors'][1:])):
        if any((r or {}).get('allow_variable_attack_family') for r in (evidence or {}).values()):
            warp['anchors']=original_anchors
            warp['source_attack_controls_unapplied']='Variable attack offsets would make quarter controls nonmonotonic; native attack protection and a fresh rendered audit are required'
        else:raise SourceWindowFailure('Acoustic source grid would make the map nonmonotonic')
    warp['source_kick_alignment']=reports
    try:audit_warp_rates(warp,sample_rate,stage='after source attack refinement')
    except ValueError as error:
        rejected=deepcopy(warp['actual_timing_rate_audit']);warp['anchors']=original_anchors
        audit_warp_rates(warp,sample_rate,stage='restored before source attack refinement')
        warp.setdefault('rejected_timing_rate_audits',[]).append(rejected)
        if not rejected.get('passed',True):
            roles=[role for role,report in reports.items() if report.get('events')]
            raise SourceWindowFailure(str(error),role=roles[0] if len(roles)==1 else None,evidence=rejected) from error
        raise
    return reports

def template_attack_rows(audio,sample_rate,centres,templates,minimum_correlation=.65,attack_band_hz=(100,1400)):
    """Find the same source kick shape, rather than any nearby novelty peak."""
    rate=8000;divisor=math.gcd(sample_rate,rate)
    band=resample_poly(np.mean(audio,axis=1),rate//divisor,sample_rate//divisor)
    band=sosfiltfilt(butter(3,attack_band_hz,btype='bandpass',fs=rate,output='sos'),band)
    pre=96;search=280;rows=[]
    for index,(centre,template) in enumerate(zip(centres,templates)):
        post=len(template)-pre
        if post<16:raise ValueError('Attack template lacks leading-front context')
        target=round(centre*rate);first=target-pre-search;last=target+post+search
        if first<0 or last>len(band):rows.append({'index':index,'time':centre,'eligible':False});continue
        segment=band[first:last];cross=correlate(segment,template,mode='valid',method='direct')
        power=np.convolve(segment**2,np.ones(len(template)),mode='valid')
        scores=cross/np.sqrt(np.maximum(power*np.sum(template**2),1e-30));peak=int(np.argmax(scores))
        outside=np.abs(np.arange(len(scores))-peak)>24
        margin=float(scores[peak]-np.max(scores[outside])) if outside.any() else 1.
        rows.append({'index':index,'time':centre+(peak-search)/rate,'eligible':bool(scores[peak]>=minimum_correlation and margin>=.02 and 2<peak<len(scores)-3),'template_correlation':float(scores[peak]),'alternative_peak_margin':margin})
    return rows


def preserve_rendered_attacks(path, source_path, warp, sample_rate, gain=1., source_origin_seconds=0., roles=('head','tail'),record_templates=True):
    """Protect recognized native attacks when stretching has changed their shape.

    Copy short unpitched native islands into the stretched deck, with cosine
    joins outside the measurement template. This is an explicit fallback,
    not evidence of alignment: audit the resulting PCM independently afterwards.
    Source events must already be supported; bodies and overlap endpoints stay
    unchanged. Apply the identical islands to any aligned foreground estimate.
    """
    with sf.SoundFile(path) as f:
        if f.samplerate!=sample_rate:raise ValueError('Attack preservation rate mismatch')
        audio=f.read(dtype='float32',always_2d=True)
    events=sorted(((role,e) for role,r in warp.get('source_kick_alignment',{}).items() if role in roles for e in r['events']),key=lambda item:item[1]['output_seconds'])
    islands=[];templates_by_role={};bass_templates_by_role={}
    with sf.SoundFile(source_path) as source:
        if source.samplerate!=sample_rate:raise ValueError('Native attack source rate mismatch')
        for i,(role,event) in enumerate(events):
            pre=round((.085 if warp.get('source_attack_controls_unapplied') else .050)*sample_rate);post=round(.085*sample_rate);fade=round(.015*sample_rate)
            if i and event['output_seconds']-events[i-1][1]['output_seconds']<(pre+post)/sample_rate:raise ValueError('Overlapping protected attacks need review')
            first=round((event['source_seconds']-source_origin_seconds)*sample_rate)-pre
            dest=round(event['output_seconds']*sample_rate)-pre;count=pre+post
            if first<0 or first+count>source.frames or dest<0 or dest+count>len(audio):raise ValueError('Protected attack lacks source/output context')
            source.seek(first);raw=source.read(count,dtype='float32',always_2d=True)*gain
            weight=np.ones(count);weight[:fade]=.5-.5*np.cos(np.linspace(0,np.pi,fade));weight[-fade:]=weight[:fade][::-1]
            audio[dest:dest+count]=audio[dest:dest+count]*(1-weight[:,None])+raw*weight[:,None]
            islands.append({'beat':event['beat'],'source_start_frame':first,'output_start_frame':dest,'frames':count,'fade_frames':fade})
            if record_templates:
                centre=(event['source_seconds']-source_origin_seconds)*sample_rate;lo=max(0,round(centre-.25*sample_rate));hi=min(source.frames,round(centre+.25*sample_rate));source.seek(lo)
                context=source.read(hi-lo,dtype='float32',always_2d=True);divisor=math.gcd(sample_rate,8000);band=resample_poly(context.mean(axis=1),8000//divisor,sample_rate//divisor)
                band_hz=warp['source_kick_alignment'][role].get('attack_band_hz',(100,1400))
                low=sosfiltfilt(butter(3,(35,180),btype='bandpass',fs=8000,output='sos'),band)
                band=sosfiltfilt(butter(3,band_hz,btype='bandpass',fs=8000,output='sos'),band);c=round((centre-lo)*8000/sample_rate)
                if c<96 or c+360>len(band):raise ValueError('Protected attack template lacks context')
                templates_by_role.setdefault(role,[]).append(band[c-96:c+360])
                bass_templates_by_role.setdefault(role,[]).append(low[c-96:c+360])
    if not islands:raise ValueError('No supported source attacks to preserve')
    sf.write(path,audio,sample_rate,subtype='FLOAT')
    if record_templates:
        paths={}
        for role,templates in templates_by_role.items():
            template_path=Path(path).with_name(Path(path).stem+f'-protected-{role}-mixture-templates.npz');np.savez_compressed(template_path,templates=np.asarray(templates,dtype='float32'));paths[role]=str(template_path)
        warp['native_attack_preservation']={'method':'unpitched native attack islands with cosine joins outside the attack template; rendered PCM still requires independent verification','events':islands,'maximum_island_seconds':max(p['frames']/sample_rate for p in islands),'mixture_attack_templates':paths,'template_evidence':'Prepared native mixture at the already identified source percussion attack; require strong waveform identity in actual output PCM'}
        low_paths={}
        for role,templates in bass_templates_by_role.items():
            target=Path(path).with_name(Path(path).stem+f'-protected-{role}-bass-mixture-templates.npz')
            np.savez_compressed(target,templates=np.asarray(templates,dtype='float32'));low_paths[role]=str(target)
        warp['native_attack_preservation']['bass_mixture_attack_templates']=low_paths
    return islands


def rendered_attack_audit(path,warp,sample_rate):
    measured=[];pending=[]
    for role,source in warp.get('source_kick_alignment',{}).items():
        mapping=warp['input_output_beat_mapping'][role]
        if not source['events']:
            pending.extend({'role':role,'beat':p['beat'],'reason':'source lacks a reliable repeated attack sequence'} for p in mapping[1:-1]);continue
        known={e['beat'] for e in source['events']}
        pending.extend({'role':role,'beat':p['beat'],'reason':'source attack not certified; grid inferred from neighbouring repeated attacks'} for p in mapping[1:-1] if p['beat'] not in known)
        first=max(0,round((mapping[0]['output_seconds']-.2)*sample_rate))
        with sf.SoundFile(path) as f:
            last=min(f.frames,round((mapping[-1]['output_seconds']+.2)*sample_rate));f.seek(first)
            audio=f.read(last-first,dtype='float32',always_2d=True)
        centres=[e['output_seconds']-first/sample_rate for e in source['events']]
        protected=warp.get('native_attack_preservation',{}).get('mixture_attack_templates',{}).get(role)
        if protected or source.get('attack_templates_path'):
            with np.load(protected or source['attack_templates_path']) as saved:templates=saved['templates']
            rows=template_attack_rows(audio,sample_rate,centres,templates,minimum_correlation=.90 if protected else .65,attack_band_hz=source.get('attack_band_hz',(100,1400)))
        else:rows=isolated_events(audio,sample_rate,centres) if source.get('attack_kind')=='isolated_peak' else attack_events(audio,sample_rate,centres,search_seconds=.035)
        if source.get('attack_kind')=='isolated_peak':
            by_index={r['index']:r for r in rows}
            rows=[{**by_index.get(i,{'time':float(centre)}),'eligible':i in by_index} for i,centre in enumerate(centres)]
        views=[('primary',rows)]
        if source.get('bass_attack_templates_path'):
            protected_bass=warp.get('native_attack_preservation',{}).get('bass_mixture_attack_templates',{}).get(role)
            with np.load(protected_bass or source['bass_attack_templates_path']) as saved:low_templates=saved['templates']
            views.append(('bass_body',template_attack_rows(audio,sample_rate,centres,low_templates,minimum_correlation=.90,attack_band_hz=(35,180))))
        contract=source.get('native_mixture_reference',{})
        for label,key,band,minimum in [('primary','attack_templates_path',source.get('attack_band_hz',(100,1400)),.65),
                                      ('bass','bass_attack_templates_path',(35,180),.90)]:
            original=source.get('estimated_drum_'+key)
            if not original:continue
            with np.load(original) as archive:drum_templates=archive['templates']
            independent=template_attack_rows(audio,sample_rate,centres,drum_templates,minimum_correlation=minimum,attack_band_hz=band)
            visible=set(contract.get(label+'_visible_beats',[]))
            for row,event in zip(independent,source['events']):
                row['eligible']=row['eligible'] and event['beat'] in visible
            views.append(('estimated_'+label,independent))
        for view,view_rows in views:
          for row,event in zip(view_rows,source['events']):
            if not row['eligible']:
                pending.append({'role':role,'beat':event['beat'],'view':view,'reason':'output attack ambiguous'});continue
            actual=first/sample_rate+row['time'];target=event['output_seconds']
            measured.append({'role':role,'beat':event['beat'],'view':view,'expected_seconds':target,'observed_seconds':actual,
                             'grid_output_seconds':target,'delay_seconds':actual-target,
                             **{k:row[k] for k in ('template_correlation','alternative_peak_margin') if k in row}})
    return {'measured':measured,'unverified':pending,
            'maximum_measured_drift_seconds':max((abs(p['delay_seconds']) for p in measured),default=None),
            'all_interior_beats_verified':bool(measured and not pending),
            'measurement':'recognized source primary attack template against the common output clock; source offsets are corrected, not retained'}

def validate_overlap_attacks(clips,transitions,tolerance_seconds=.003):
    """Fail on any measured flam, not a mean, and never certify zero coverage."""
    reports=[]
    for index,edge in enumerate(transitions):
        if edge.get('tempo_technique')!='shared_phase_sync':continue
        sides=[];observations=[]
        for clip,role in ((clips[index],'tail'),(clips[index+1],'head')):
            audit=(clip.get('warp') or {}).get('post_render_transient_audit',{}).get('final',{})
            recovered=((clip.get('warp') or {}).get('source_kick_alignment',{}).get(role,{}).get('low_frequency_gap_recovery') or {}).get('recovered_beats',[])
            observed_bass={p['beat'] for p in audit.get('measured',[]) if p['role']==role and p.get('view')=='bass_body'}
            if set(recovered)-observed_bass:
                raise ValueError(f"{clip['id']}: recovered source bass attacks remain unverified in rendered PCM; review the source window")
            source=(clip.get('warp') or {}).get('source_kick_alignment',{}).get(role,{})
            required=set(recovered)&set((source.get('native_mixture_reference') or {}).get('bass_visible_beats',[]))
            observed_estimate={p['beat'] for p in audit.get('measured',[]) if p['role']==role and p.get('view')=='estimated_bass'}
            if required-observed_estimate:
                raise ValueError(f"{clip['id']}: independently visible recovered bass signatures remain unverified")
            # Legacy latency checks retained source offsets. Subtract the
            # actual shared grid here, so two individually accurate warps
            # cannot hide different original attack offsets.
            side={};points={}
            for p in audit.get('measured',[]):
                if p['role']!=role:continue
                error=p['observed_seconds']-p['grid_output_seconds']
                points.setdefault(p['beat'],[]).append(error)
                if p['beat'] not in side or abs(error)>abs(side[p['beat']]):side[p['beat']]=error
            sides.append(side)
            observations.append(points)
        common=sorted(set(sides[0])&set(sides[1]));beats=int(edge['beats'])
        side_maxima=[max((abs(v) for v in side.values()),default=None) for side in sides]
        sparse=len(common)<min(8,beats-1)
        if sparse and (not common or min(map(len,sides))<8 or any(v is None or v>tolerance_seconds/2 for v in side_maxima)):
            raise ValueError(f"{edge['from']} → {edge['to']}: no adequate shared acoustic attack evidence")
        if any(v is not None and v>tolerance_seconds for v in side_maxima):
            raise ValueError(f"{edge['from']} → {edge['to']}: an individual rendered attack misses the shared clock by {max(v for v in side_maxima if v is not None)*1000:.2f} ms")
        errors=[{'beat':k,'error_seconds':max((a-b for a in observations[0][k] for b in observations[1][k]),key=abs)} for k in common]
        maximum=max(abs(e['error_seconds']) for e in errors)
        report={'verified_common_beats':len(common),'unverified_beat_indices':[k for k in range(1,beats) if k not in common],
                'maximum_pair_error_seconds':maximum,'tolerance_seconds':tolerance_seconds,'events':errors,
                'all_interior_attacks_verified':len(common)==beats-1,
                'verification_kind':'sparse direct coincidences plus individually verified shared-clock attacks' if sparse else 'direct paired attack coincidence',
                'recognized_attacks_per_side':[len(s) for s in sides],
                'maximum_individual_clock_error_seconds':side_maxima,
                'conservative_pair_error_bound_seconds':sum(side_maxima) if sparse else maximum,
                'measurement':'rendered primary attack coincidence between decks; maximum over shared measured events'}
        edge['paired_attack_audit']=report;reports.append(report)
        if maximum>tolerance_seconds:raise ValueError(f"{edge['from']} → {edge['to']}: rendered attacks differ by {maximum*1000:.2f} ms; cannot export as aligned")
    return reports


def apply_fine_timing(source_path,output_path,warp,sample_rate):
    """Apply the same bounded continuous residual delay to audio or voice.

    This final interpolation addresses backend transient snapping. It is
    confined to the overlaps and records its small instantaneous rate change.
    """
    with sf.SoundFile(source_path) as source:
        audio=source.read(dtype='float32',always_2d=True)
    result=audio.copy()
    for role,controls in warp.get('fine_attack_timing',{}).items():
        times=np.array(controls['times']);offsets=np.array(controls['offsets'])
        first=max(0,round(times[0]*sample_rate));last=min(len(audio),round(times[-1]*sample_rate))
        if last<=first:continue
        t=np.arange(first,last)/sample_rate
        delta=PchipInterpolator(times,offsets)(t)
        positions=t*sample_rate+delta*sample_rate
        rates=np.diff(positions)
        if np.max(np.abs(delta))>.035 or rates.min()<.85 or rates.max()>1.15:
            raise ValueError('Fine attack correction exceeds its bounded local delay/rate')
        controls['maximum_delay_seconds']=float(np.max(np.abs(delta)))
        controls['minimum_rate']=float(rates.min());controls['maximum_rate']=float(rates.max())
        # Include context for cubic interpolation; no repeated processing of
        # an earlier candidate. Native bodies outside these windows are exact.
        pad=round(.08*sample_rate);lo=max(0,first-pad);hi=min(len(audio),last+pad)
        for channel in range(audio.shape[1]):
            result[first:last,channel]=map_coordinates(audio[lo:hi,channel],[positions-lo],order=3,mode='nearest',prefilter=True)
    sf.write(output_path,result,sample_rate,subtype='FLOAT')


def finalize_attack_timing(path,warp,sample_rate,*,allow_pitch_changing_residual=False):
    """Do not turn residual attack delay into unintended pitch modulation.

    Fractional-delay interpolation changes playback speed and pitch. A 15%
    timing-map bound is not a pitch guarantee. Keep the pitch-preserving
    backend PCM; failed attacks require recovery of a compatible window.
    """
    if allow_pitch_changing_residual:
        return fine_align_rendered(path,warp,sample_rate)
    if warp.get('fine_attack_timing'):
        raise ValueError('Pitch-changing residual timing requires explicit selection')
    return rendered_attack_audit(path,warp,sample_rate)


def fine_align_rendered(path,warp,sample_rate,attempts=4,tolerance_seconds=.0015,audit_function=None):
    """Refine measured residual attack delay using one immutable backend PCM."""
    import shutil
    from copy import deepcopy
    path=Path(path);base=path.with_name(path.stem+'-fine-base.wav');best_path=path.with_name(path.stem+'-fine-best.wav')
    shutil.copy2(path,base)
    measure=audit_function or rendered_attack_audit
    audit=measure(path,warp,sample_rate);history=[audit]
    maximum=audit['maximum_measured_drift_seconds']
    if maximum is None:base.unlink();return audit
    best=(maximum,deepcopy(warp),deepcopy(audit));shutil.copy2(path,best_path)
    for _ in range(attempts):
        if audit['maximum_measured_drift_seconds']<=tolerance_seconds:break
        previous=deepcopy(warp.get('fine_attack_timing',{}));controls={}
        for role in ('head','tail'):
            grouped={}
            for point in audit['measured']:
                if point['role']==role:grouped.setdefault(point['grid_output_seconds'],[]).append(point)
            # Primary and bass are independent observations of one physical
            # event. Use one minimax timing control, then recheck both views.
            events=[]
            for stamp,points in sorted(grouped.items()):
                delays=[p['observed_seconds']-stamp for p in points]
                events.append({'grid_output_seconds':stamp,'observed_seconds':stamp+(min(delays)+max(delays))/2})
            if not events:continue
            mapping=warp['input_output_beat_mapping'][role]
            start=mapping[0]['output_seconds'];end=mapping[-1]['output_seconds']
            times=[start]+[p['grid_output_seconds'] for p in events]+[end]
            offsets=[0.]+[p['observed_seconds']-p['grid_output_seconds'] for p in events]+[0.]
            if role in previous:
                old=PchipInterpolator(previous[role]['times'],previous[role]['offsets'],extrapolate=False)
                offsets=[float(e+old(np.clip(t+e,start,end))) for t,e in zip(times,offsets)]
            controls[role]={'times':times,'offsets':offsets,'method':'bounded continuous cubic interpolation of residual primary attack delays; fixed overlap endpoints'}
        warp['fine_attack_timing']=controls
        apply_fine_timing(base,path,warp,sample_rate)
        audit=measure(path,warp,sample_rate);history.append(audit)
        maximum=audit['maximum_measured_drift_seconds']
        if maximum is not None and maximum<best[0]:best=(maximum,deepcopy(warp),deepcopy(audit));shutil.copy2(path,best_path)
        if maximum is None:break
    warp.clear();warp.update(best[1]);shutil.copy2(best_path,path)
    warp['fine_attack_iterations']=history
    base.unlink();best_path.unlink()
    return best[2]
