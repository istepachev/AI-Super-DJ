"""Conditional percussion ownership on exact, final overlap PCM."""
from pathlib import Path
import hashlib
import numpy as np
import soundfile as sf
from .derived_audio import derived_scope

def needs_handoff(outgoing_warp,incoming_warp,*,style='default'):
    if style not in ('default','sync','zhilin'):raise ValueError('Unknown percussion mixing style')
    return any((warp.get('source_kick_alignment',{}).get(role,{}).get('attack_front_comparison') or {}).get('upper_front_selected',False)
               or (warp.get('source_kick_alignment',{}).get(role,{}).get('low_frequency_gap_recovery') or {}).get('recovered_beats')
               or warp.get('source_kick_alignment',{}).get(role,{}).get('native_rhythm_observations')
               or warp.get('source_kick_alignment',{}).get(role,{}).get('unverified_native_bass_candidates')
               or ambiguous_front(warp.get('source_kick_alignment',{}).get(role,{}))
               for warp,role in ((outgoing_warp or {},'tail'),(incoming_warp or {},'head')))

def ambiguous_front(source):
    view=source.get('attack_front_comparison') or {}
    return bool(view.get('matched_source_events',0)>=8 and not view.get('upper_front_selected')
        and view.get('body_minus_upper_median_seconds',0)>.008)

def ownership_policy(outgoing_warp,incoming_warp):
    sources=[w.get('source_kick_alignment',{}).get(role,{}) for w,role in ((outgoing_warp or {},'tail'),(incoming_warp or {},'head'))]
    uncertain=any(ambiguous_front(s) or s.get('native_rhythm_observations') or s.get('unverified_native_bass_candidates') for s in sources)
    return 'smooth' if uncertain else 'shared'

def percussion_gains(gains,beat_phase,beats,policy='shared'):
    if policy not in ('shared','smooth','sequential'):raise ValueError('Unknown percussion ownership policy')
    if policy=='shared':return gains
    from .mixing import smooth
    phase=np.asarray(beat_phase) if beat_phase is not None else np.linspace(0,1,len(gains['bass_in']))
    both=(gains['bass_in'][:,0]>=.49)&(gains['bass_in'][:,0]<=.51)
    center=float(np.median(phase[both])) if both.any() else .55
    if policy=='smooth':
        # Preserve a gradual rhythm handoff even when the bass envelope is
        # brief. These are gain curves, never an audio timing/pitch repair.
        changing=np.flatnonzero((gains['bass_in'][:,0]>0)&(gains['bass_in'][:,0]<1))
        width=(phase[changing[-1]]-phase[changing[0]])*beats if len(changing) else 0.
        width=min(float(beats),max(8.,float(width)))
        first=max(0.,min(center*beats-width/2,beats-width))
        incoming=smooth((phase*beats-first)/width)[:,None]
        return {**gains,'drums_out':1-incoming,'drums_in':incoming}
    beat=min(beats-1,max(1,int(np.ceil(center*beats))));clock=phase*beats
    old=(1-smooth((clock-(beat-1))/.9))[:,None]
    new=smooth((clock-(beat-.1))/.1)[:,None]
    assert np.max(old*new)==0
    return {**gains,'drums_out':old,'drums_in':new}

def final_drum_estimates(outgoing,incoming,sample_rate,directory,*,ffmpeg='ffmpeg'):
    from .stems import configuration,separate
    config=configuration()
    if config is None:raise RuntimeError('Percussion handoff requires the configured local Demucs runtime')
    from .execution import current
    session=current()
    shared_cache=session.root if session else None
    cache=Path(shared_cache)/'final-percussion' if shared_cache else Path(directory)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def one(item):
        role,pcm=item
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
        return drums,{'role':role,'final_deck_pcm_sha256':digest,'estimate_path':estimate['drums_path'],'provenance':estimate['provenance'],'estimated_not_ground_truth':True}
    sides=(('out',outgoing),('in',incoming))
    results=session.device_pair(one,sides) if session and config.get('device')!='cpu' else list(map(one,sides))
    return {'drums_out':results[0][0],'drums_in':results[1][0]},[r[1] for r in results]

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
            value+=sum(band*(gains.get('drums_'+side,gains['bass_'+side])-gains[name+'_'+side]) for name,band in zip(('bass','mid','high'),split_bands(drums,sample_rate,bass_hz)))
        result[role]=value.astype('float32')
    return result


def _known_native_fronts(audio,sample_rate,references,coverage,side,band):
    """Independent native identities in one exact known contribution."""
    from .kick_grid import template_attack_rows
    pad=round(.2*sample_rate);result={}
    for role,centres,templates,reference_band,_ in references:
        if role not in (side,side+':bass_body',side+':initial_front'):continue
        if role==side+':bass_body' and list(band)!=[35,180]:continue
        observable=coverage.get(role,set(centres))
        rows=template_attack_rows(np.pad(audio,((pad,pad),(0,0))),sample_rate,
            [t+.2 for t in centres],templates,attack_band_hz=reference_band,
            minimum_correlation=.90 if list(reference_band)==[35,180] else .65)
        for t,row in zip(centres,rows):
            if t not in observable or not row['eligible']:continue
            result.setdefault(round(t*sample_rate),[]).append({**row,'delay_seconds':row['time']-.2-t,
                'native_reference_seconds':t,'native_attack_band_hz':list(reference_band)})
    return result


def _strengthened_own_rows(centres,direct,fresh):
    """Confirm an already on-time native identity; never erase a late own hit."""
    result=[dict(r) for r in direct];proof={}
    for i,(t,a,b) in enumerate(zip(centres,direct,fresh)):
        if (not a['eligible'] or abs(a['time']-.2-t)>.003
                or a.get('template_correlation',0)<.90 or a.get('alternative_peak_margin',0)<.05
                or not b['eligible'] or abs(b['time']-.2-t)>.003
                or b.get('template_correlation',0)<.98 or b.get('alternative_peak_margin',0)<.05
                or abs(a['time']-b['time'])>.0015):continue
        result[i]=b
        proof[i]={'own_actual_pcm_delay_seconds':a['time']-.2-t,
            'own_actual_pcm_template_correlation':a['template_correlation'],
            'own_fresh_confirmed_delay_seconds':b['time']-.2-t,
            'own_fresh_confirmed_template_correlation':b['template_correlation'],
            'eligible_late_own_reference_erased':False}
    return result,proof


def _foreground_identity_ambiguities(audio,own,other,sample_rate,centres,raw,isolated,band,other_fronts=None,neighbor_radius_seconds=None):
    """A matching dominant other deck cannot identify a quiet source's attack.

    The known contributions must already reconstruct the actual PCM. Only an
    eligible native-template match in the actual own contribution can supply
    timing. This never substitutes a fresh, guessed separation for a late hit.
    """
    candidates=[i for i,(t,a,b) in enumerate(zip(centres,raw,isolated))
        if a['eligible'] and abs(a['time']-.2-t)>.003
        and not (b['eligible'] and abs(b['time']-.2-t)>.003)]
    if not candidates:return {}
    from scipy.signal import butter,sosfiltfilt
    filtered=[sosfiltfilt(butter(3,band,btype='bandpass',fs=sample_rate,output='sos'),x.mean(axis=1))
              for x in (audio,own,other)]
    result={}
    for i in candidates:
        stamp=raw[i]['time']-.2
        first=max(0,round((stamp-.012)*sample_rate));last=min(len(audio),round((stamp+.045)*sample_rate))
        a,b,c=(x[first:last] for x in filtered)
        powers=[float(np.dot(x,x)) for x in (a,b,c)]
        if min(powers)<=1e-20:continue
        similarity=float(np.dot(a,c)/np.sqrt(powers[0]*powers[2]))
        dominance=powers[2]/powers[1]
        additive=False;certificate=None
        if similarity<.98 or dominance<4:
            b=isolated[i];frame=round(centres[i]*sample_rate)
            available=(other_fronts or {}).get(frame,[]);bracket=None
            # A native rhythm may deliberately have no attack on this beat.
            # Certify its actual neighbouring fronts, without inventing the
            # absent event or claiming a coincident pair measurement there.
            if not available and neighbor_radius_seconds and len(other_fronts or {})>=8:
                before=[k for k in other_fronts if k<frame]
                after=[k for k in other_fronts if k>frame]
                if before and after:
                    left,right=max(before),min(after);radius=neighbor_radius_seconds*sample_rate
                    if frame-left<=radius and right-frame<=radius:
                        bracket=[other_fronts[left],other_fronts[right]]
                        available=[r for side in bracket for r in side]
            # A sum of different aligned waveforms need not peak at either
            # native template's correlation maximum. Both exact contributions
            # must independently identify their original native fronts. This
            # never rescues an eligible late own match or an unknown source.
            own_delay=b['time']-.2-centres[i]
            if (not b['eligible'] or abs(own_delay)>.003
                    or b.get('template_correlation',0)<.98
                    or b.get('alternative_peak_margin',0)<.05
                    or not available or any(abs(r['delay_seconds'])>.003 for r in available)):
                continue
            valid=lambda r: abs(r['delay_seconds'])<=.003 and r.get('template_correlation',0)>=.98 and r.get('alternative_peak_margin',0)>=.05
            if bracket and not all(any(valid(r) for r in side) for side in bracket):continue
            certificate=next((r for r in available if abs(r['delay_seconds'])<=.003
                and abs(r['delay_seconds']-own_delay)<=.003
                and r.get('template_correlation',0)>=.98
                and r.get('alternative_peak_margin',0)>=.05),None)
            if certificate is None or np.max(abs(audio-own-other))>2e-6:continue
            additive=True
        result[i]={'seconds':centres[i],'raw_delay_seconds':raw[i]['time']-.2-centres[i],
            'raw_template_correlation':raw[i].get('template_correlation'),
            'own_delay_seconds':isolated[i]['time']-.2-centres[i] if isolated[i]['eligible'] else None,
            'own_unverified_candidate_delay_seconds':isolated[i]['time']-.2-centres[i],
            'own_template_correlation':isolated[i].get('template_correlation'),
            'other_to_own_band_power_ratio':dominance,'raw_candidate_other_waveform_cosine':similarity,
            'classification':'raw peak is not uniquely attributable to this source; known other foreground explains it',
            'timing_view':'own native-template timing still required; raw contradiction retained'}
        if additive:
            result[i].update(classification='additive waveform interference; both known native fronts independently verified',
                exact_mixture_reconstruction_verified=True,
                other_native_delay_seconds=certificate['delay_seconds'] if bracket is None else None,
                other_native_template_correlation=certificate['template_correlation'],
                other_native_alternative_peak_margin=certificate['alternative_peak_margin'],
                independently_verified_other_native_band_hz=certificate.get('native_attack_band_hz'),
                independent_source_timing_budget_seconds=.003,
                directly_verified_native_pair_error_seconds=abs(certificate['delay_seconds']-(isolated[i]['time']-.2-centres[i])) if bracket is None else None,
                other_native_clock_bracketing_references=[{'seconds':r.get('native_reference_seconds'),
                    'delay_seconds':r['delay_seconds'],'correlation':r['template_correlation']} for side in (bracket or []) for r in side if valid(r)],
                missing_other_attack_invented=False,
                raw_timing_contradiction_retained=True)
    return result


def _paired_front_audit(checks,sample_rate):
    """Retain the worst source-aware primary/bass difference on each beat."""
    from .integrity import _timing_limit_exceeded
    groups={}
    for check in checks:
        side=check['role'].split(':')[0]
        for point in check['recognized']:
            frame=round(point['seconds']*sample_rate)
            groups.setdefault(frame,{}).setdefault(side,[]).append(point['delay_seconds'])
    pairs=[]
    for frame,views in groups.items():
        if not views.get('head') or not views.get('tail'):continue
        worst=max(abs(a-b) for a in views['tail'] for b in views['head'])
        if _timing_limit_exceeded(worst,.003,reference_seconds=frame/sample_rate):
            refusal=ValueError('Combined paired percussion attacks miss the shared clock; review the rendered overlap')
            refusal.evidence={'stage':'actual combined paired percussion fronts',
                'coincident_output_seconds':frame/sample_rate,
                'maximum_pair_error_seconds':float(worst),'tolerance_seconds':.003,
                'observed_delays_by_deck':{side:[float(value)for value in values]for side,values in views.items()},
                'source_failure_ownership_assigned':False,
                'scope':'worst recognized primary/bass difference in the actual mix; no independent failed source is inferred'}
            raise refusal
        pairs.append(worst)
    return {'coincident_source_beats':len(pairs),'maximum_pair_error_seconds':max(pairs) if pairs else None,
        'tolerance_seconds':.003,'scope':'worst recognized primary/bass difference across actual audible source contributions; prior coverage gates retained'}


def _audible_band_reference_indices(components,role,sample_rate,centres,templates,band,gains,bass_audible,gaps=None,minimum=.90):
    """Do not treat a silent bass fader as proof that an upper click is gone.

    Band envelopes supply a conservative bound. Exact known own PCM can rule
    out a genuinely quiet routed view, without using the other deck's energy.
    Unknown nonquiet shapes remain coverage gaps, never invented observations.
    """
    side='out' if role=='tail' else 'in';low,high=band
    keys=['drums_'+side]
    if low<180:keys.append('bass_'+side)
    if high>180 and low<3500:keys.append('mid_'+side)
    if high>3500:keys.append('high_'+side)
    curves=[np.asarray(gains[k])[:,0]for k in keys if k in gains]
    if not curves:return list(bass_audible)
    bound=np.maximum.reduce(curves);indices=[i for i,t in enumerate(centres)if 0<=round(t*sample_rate)<len(bound)and bound[round(t*sample_rate)]>=.05]
    if not components or role not in components:return indices
    from scipy.signal import butter,sosfiltfilt,resample_poly
    import math
    own=np.asarray(components[role]);divisor=math.gcd(sample_rate,8000)
    signal=resample_poly(own.mean(axis=1),8000//divisor,sample_rate//divisor)
    signal=sosfiltfilt(butter(3,band,btype='bandpass',fs=8000,output='sos'),signal)
    result=[]
    for i in indices:
        center=round(centres[i]*8000);first=center-96;last=first+len(templates[i])
        if first<0 or last>len(signal):continue
        native=float(np.mean(np.square(templates[i],dtype='float64')))
        actual=float(np.mean(np.square(signal[first:last],dtype='float64')))
        if native>1e-20 and actual>=native*.05**2:result.append(i)
    # The added post-bass views need an actual own-source identity. A weak
    # match in the combined waveform alone can belong to the other track.
    extra=[i for i in result if i not in set(bass_audible)]
    if extra:
        from .kick_grid import template_attack_rows
        pad=round(.2*sample_rate)
        rows=template_attack_rows(np.pad(own,((pad,pad),(0,0))),sample_rate,
            [centres[i]+.2 for i in extra],templates[extra],minimum_correlation=minimum,attack_band_hz=band)
        visible={i for i,row in zip(extra,rows) if row['eligible']}
        if gaps is not None:
            gaps.extend({'role':role,'seconds':centres[i],'band_hz':list(band),
                'reason':'nonquiet routed contribution does not identify this native signature; not a timed source observation'}
                for i,row in zip(extra,rows) if not row['eligible'])
        result=[i for i in result if i in set(bass_audible) or i in visible]
    return result


def _unfiltered_reference_required(indices,centres,sample_rate,gains,side):
    """A native short template is mandatory when routing preserves its shape.

    Frequency/percussion subtraction changes the native mixture shape. Such
    audible views retain maximum-error and coverage checks, with unresolved
    templates explicit, while the unprocessed full-deck gate stays mandatory.
    """
    keys=[k+'_'+side for k in ('bass','mid','high','drums')if k+'_'+side in gains]
    required=set()
    for i in indices:
        frame=round(centres[i]*sample_rate);values=[float(gains[k][frame,0])for k in keys]
        if values and max(values)-min(values)<=1e-9:required.add(centres[i])
    return required


@derived_scope()
def verify_blended_fronts(audio,sample_rate,outgoing_warp,incoming_warp,gains,directory,*,ffmpeg='ffmpeg',components=None,allow_pitch_changing_residual=False):
    """Check identified source fronts in the actual blend; bound local repair."""
    from .kick_grid import template_attack_rows
    from .source_observability import required_bass_views
    if components is not None:
        if set(components)!={'head','tail'} or any(p.shape!=audio.shape or not np.isfinite(p).all() for p in components.values()):raise ValueError('Invalid final-deck verification components')
        if np.max(abs(components['head']+components['tail']-audio))>2e-6:raise ValueError('Verification components do not reconstruct actual mixed PCM')
    references=[];source_coverage={};audibility_gaps=[];n=len(audio);pad=round(.2*sample_rate)
    for warp,role,gain_key in ((outgoing_warp or {},'tail','bass_out'),(incoming_warp or {},'head','bass_in')):
        source=warp.get('source_kick_alignment',{}).get(role,{})
        if not source.get('attack_templates_path') or not source.get('events'):continue
        mapping=warp['input_output_beat_mapping'][role];start=mapping[0]['output_seconds'];events=source['events'];centres=[e['output_seconds']-start for e in events]
        with np.load(source['attack_templates_path']) as f:templates=f['templates']
        side='out' if role=='tail' else 'in';audibility=gains.get('drums_'+side,gains[gain_key])
        audible=[i for i,t in enumerate(centres) if 0<=round(t*sample_rate)<n and audibility[round(t*sample_rate),0]>=.05]
        bass_audible=list(audible)
        audible=_audible_band_reference_indices(components,role,sample_rate,centres,templates,source.get('attack_band_hz',[100,1400]),gains,bass_audible,audibility_gaps)
        if not audible and not bass_audible:continue
        if audible:references.append((role,[centres[i] for i in audible],templates[audible],source.get('attack_band_hz',[100,1400]),set()))
        contract=source.get('native_mixture_reference',{})
        required_views=required_bass_views(source)
        if source.get('initial_front_attack_templates_path'):
            with np.load(source['initial_front_attack_templates_path']) as archive:initial=archive['templates']
            visible=set(contract.get('initial_front_visible_beats',[]))
            initial_audible=_audible_band_reference_indices(components,role,sample_rate,centres,initial,source.get('attack_band_hz',[100,1400]),gains,bass_audible,audibility_gaps,minimum=.90)
            # A 24 ms unfiltered native shape is not the same signal after
            # separate-band routing or estimated-drum subtraction. Retain
            # those as explicit uncertainty; never infer a timing displacement
            # by correlating different short attack shapes. Full-deck initial
            # checks and the existing primary/bass blend views still apply.
            compatible=_unfiltered_reference_required(initial_audible,centres,sample_rate,gains,side)
            for i in initial_audible:
                if events[i]['beat'] in visible and centres[i] not in compatible:
                    audibility_gaps.append({'role':role+':initial_front','seconds':centres[i],
                        'status':'short native reference changed by spectral routing; independent full-deck initial-front and ordinary blend checks retained'})
            selected=[i for i in initial_audible if events[i]['beat'] in visible and centres[i] in compatible]
            if selected:
                required={centres[i] for i in selected if events[i]['beat'] in set(contract.get('initial_front_required_beats',[]))}
                references.append((role+':initial_front',[centres[i] for i in selected],initial[selected],source.get('attack_band_hz',[100,1400]),required))
                source_coverage[role+':initial_front']={centres[i] for i in selected}
        if source.get('bass_attack_templates_path'):
            protected=warp.get('native_attack_preservation',{}).get('bass_mixture_attack_templates',{}).get(role)
            with np.load(protected or source['bass_attack_templates_path']) as f:low=f['templates']
            required={centres[i] for i in bass_audible if events[i]['beat'] in required_views['mixture']}
            references.append((role+':bass_body',[centres[i] for i in bass_audible],low[bass_audible],[35,180],required))
            if 'mixture_bass_visible_beats' in contract:
                visible=set(contract['mixture_bass_visible_beats'])
                source_coverage[role+':bass_body']={centres[i] for i in bass_audible if events[i]['beat'] in visible}
        for label,key,band in [('primary','attack_templates_path',source.get('attack_band_hz',[100,1400])),
                               ('bass','bass_attack_templates_path',[35,180])]:
            original=source.get('estimated_drum_'+key)
            visible=set(contract.get(label+'_visible_beats',[]))
            selected=[i for i in bass_audible if events[i]['beat'] in visible]
            if not original or not selected:continue
            with np.load(original) as archive:independent=archive['templates']
            required={centres[i] for i in selected if label=='bass' and events[i]['beat'] in required_views['estimate']}
            references.append((role+':estimated_'+label,[centres[i] for i in selected],independent[selected],band,required))
    if not references:return audio,{'executed':False,'reason':'no audible recognized source attack family','unverified_routed_source_signatures':audibility_gaps}
    original_components={k:v.copy() for k,v in (components or {}).items()}
    current_components={k:v.copy() for k,v in original_components.items()}
    original=audio.copy();corrected=audio.copy();refinements=[];checks=[]
    for attempt in range(2):
        checks=[];residuals=[]
        for role,centres,templates,band,required in references:
            observable=source_coverage.get(role,set(centres));minimum_count=min(8,len(observable))
            unknown_candidates=[]
            def qualify_native(rows,view):
                # This support is fixed before the first stretch. A newly
                # unique output correlation cannot establish a native clock
                # for an originally ambiguous periodic bass waveform.
                for t,row in zip(centres,rows):
                    if t in observable:continue
                    if row['eligible']:unknown_candidates.append({'seconds':t,'candidate_delay_seconds':row['time']-.2-t,
                        'correlation':row.get('template_correlation'),'view':view,'status':'native source timing unverified'})
                    row['eligible']=False
            minimum=.90 if band==[35,180] or role.endswith(':initial_front') else .65
            rows=template_attack_rows(np.pad(corrected,((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band,minimum_correlation=minimum)
            raw_rows=[dict(r) for r in rows];component_used=[];identity_ambiguities={};own_fresh=None
            if current_components:
                other='head' if role.startswith('tail') else 'tail'
                isolated=corrected-current_components[other]
                recovered=template_attack_rows(np.pad(isolated,((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band,minimum_correlation=minimum)
                own_warp=outgoing_warp if role.startswith('tail') else incoming_warp
                other_warp=incoming_warp if other=='head' else outgoing_warp
                own_source=(own_warp or {}).get('source_kick_alignment',{}).get(role.split(':')[0],{})
                other_source=(other_warp or {}).get('source_kick_alignment',{}).get(other,{})
                other_fronts=None
                neighbor_radius=None
                if own_source.get('stable_native_primary_clock') and other_source.get('stable_native_primary_clock'):
                    other_fronts=_known_native_fronts(current_components[other],sample_rate,references,source_coverage,other,band)
                    clock=(other_warp or {}).get('input_output_beat_mapping',{}).get(other,[])
                    if len(clock)>1:neighbor_radius=1.1*max(np.diff([r['output_seconds'] for r in clock]))
                identity_ambiguities=_foreground_identity_ambiguities(corrected,isolated,current_components[other],sample_rate,centres,rows,recovered,band,other_fronts,neighbor_radius)
                if any(not recovered[i]['eligible'] for i in identity_ambiguities):
                    own_estimate,_=final_drum_estimates(isolated,isolated,sample_rate,Path(directory)/f'own-view-{attempt}',ffmpeg=ffmpeg)
                    own_fresh=template_attack_rows(np.pad(own_estimate['drums_out'],((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band,minimum_correlation=minimum)
                strengthen=other_fronts and any(a['eligible'] and abs(a['time']-.2-t)>.003
                    and b['eligible'] and abs(b['time']-.2-t)<=.003
                    and b.get('template_correlation',0)>=.90
                    for t,a,b in zip(centres,rows,recovered))
                if strengthen:
                    if own_fresh is None:
                        own_estimate,_=final_drum_estimates(isolated,isolated,sample_rate,Path(directory)/f'own-confirm-{attempt}',ffmpeg=ffmpeg)
                        own_fresh=template_attack_rows(np.pad(own_estimate['drums_out'],((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band,minimum_correlation=minimum)
                    strengthened,agreements=_strengthened_own_rows(centres,recovered,own_fresh)
                    additional=_foreground_identity_ambiguities(corrected,isolated,current_components[other],sample_rate,centres,rows,strengthened,band,other_fronts,neighbor_radius)
                    for i,explanation in additional.items():
                        if i in agreements and i not in identity_ambiguities:
                            identity_ambiguities[i]={**explanation,**agreements[i]}
                for i,(old,new) in enumerate(zip(rows,recovered)):
                    if i in identity_ambiguities:
                        candidate=new if new['eligible'] else own_fresh[i]
                        if candidate['eligible'] and abs(candidate['time']-.2-centres[i])<=.003:
                            rows[i]=candidate;component_used.append(centres[i])
                            identity_ambiguities[i].update(own_delay_seconds=candidate['time']-.2-centres[i],
                                own_template_correlation=candidate.get('template_correlation'),
                                timing_view='native template in actual own contribution'+(' via fresh percussion view' if not new['eligible'] else ''),
                                fresh_own_percussion_view_used=not new['eligible'])
                            continue
                    # Keep strong contradictory raw timing evidence. A weak
                    # match to the other kick is not evidence of this source.
                    if old['eligible'] and old.get('template_correlation',0)>=.90 and abs(old['time']-.2-centres[i])>.003:continue
                    if new['eligible'] and (not old['eligible'] or new['template_correlation']>old.get('template_correlation',0)+.08):
                        rows[i]=new;component_used.append(centres[i])
            qualify_native(rows,'raw/known-component')
            count=sum(r['eligible'] for r in rows);covered=sum(r['eligible'] and t in observable for t,r in zip(centres,rows))
            maximum=max((abs(r['time']-.2-t) for t,r in zip(centres,rows) if r['eligible']),default=None);fresh=None
            if covered<minimum_count or (maximum is None and minimum_count) or (maximum is not None and maximum>.003):
                estimates,_=final_drum_estimates(corrected,corrected,sample_rate,Path(directory)/f'view-{attempt}',ffmpeg=ffmpeg);fresh=template_attack_rows(np.pad(estimates['drums_out'],((pad,pad),(0,0))),sample_rate,[t+.2 for t in centres],templates,attack_band_hz=band,minimum_correlation=minimum)
                for i,(old,new) in enumerate(zip(rows,fresh)):
                    if old['eligible'] and old.get('template_correlation',0)>=.90 and abs(old['time']-.2-centres[i])>.003:continue
                    if new['eligible'] and (not old['eligible'] or (abs(old['time']-.2-centres[i])>.003 and new['template_correlation']>old.get('template_correlation',0)+.08)):rows[i]=new
                qualify_native(rows,'fresh estimated percussion')
            measured=[{'seconds':t,'delay_seconds':round(r['time']-.2-t,9),'correlation':r['template_correlation']} for t,r in zip(centres,rows) if r['eligible']]
            if required-{r['seconds'] for r in measured}:raise ValueError('Known native initial fronts remain unverified in the actual blend; review the source window' if role.endswith(':initial_front') else 'Recovered bass attacks remain unverified in the actual blend; review the source window')
            verified_native={r['seconds'] for r in measured}&observable
            if len(verified_native)<minimum_count:raise ValueError('Combined percussion fronts remain unverified; review the source window')
            for row in measured:
                if abs(row['delay_seconds'])>.003:
                    if not allow_pitch_changing_residual or attempt or row['correlation']<.9 or abs(row['delay_seconds'])>.006:raise ValueError('Combined percussion attack misses the shared clock; review the rendered overlap')
                    residuals.append(row)
            checks.append({'role':role,'audible_source_events':len(centres),'recognized':measured,'unverified_source_seconds':[t for t,r in zip(centres,rows) if not r['eligible']],'attack_band_hz':band,'fresh_percussion_view_used':fresh is not None,'known_other_deck_subtraction_seconds':component_used,'raw_mixture_unverified_seconds':[t for i,(t,r) in enumerate(zip(centres,raw_rows)) if not r['eligible'] or t not in observable or i in identity_ambiguities],
                'raw_mixture_identity_ambiguities':list(identity_ambiguities.values()),
                'fresh_known_own_percussion_view_used':own_fresh is not None,
                'native_observable_source_events':len(observable),'coverage_minimum_matches':minimum_count,'observed_native_unique_events':len(verified_native),
                'native_ambiguous_source_seconds':[t for t in centres if t not in observable],'required_source_seconds':sorted(required),
                'native_ambiguous_output_candidates':unknown_candidates,
                'coverage_status':'additional native bass view unverified; primary checks remain mandatory' if not minimum_count else 'native-observable coverage checked',
                'maximum_error_seconds':max((abs(r['delay_seconds']) for r in measured),default=None)})
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
    paired=_paired_front_audit(checks,sample_rate)
    return corrected,{'executed':True,'recognized_fronts_only':True,'checks':checks,'paired_source_front_audit':paired,'local_combined_waveform_refinements':refinements,'other_pcm_retained':True,'measurement_limit_seconds':.003,'unverified_routed_source_signatures':audibility_gaps}
