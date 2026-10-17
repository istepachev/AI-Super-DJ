"""Bounded alternative-window search; explicit montage constraints stay fixed."""
from copy import deepcopy
from pathlib import Path

from .render import RenderError
from .transition_quality import TransitionFailure


def unsupported_attack_events(evidence,role):
    """Retain physical source positions across every required timing view."""
    if role in evidence.get('canonical_attack_rejections',{}):
        return list(evidence['canonical_attack_rejections'][role])
    source=evidence.get('warp',{}).get('source_kick_alignment',{}).get(role,{})
    points=[p for p in evidence.get('final_attack',{}).get('measured',[]) if p['role']==role]
    from .source_observability import required_bass_views
    required=required_bass_views(source)
    observed={p['beat'] for p in points if p.get('view')=='bass_body'}
    independent={p['beat'] for p in points if p.get('view')=='estimated_bass'}
    failed=(required['mixture']-observed)|(required['estimate']-independent)
    failed.update(p['beat'] for p in points if abs(p.get('delay_seconds',0.))>.003)
    return [p['source_seconds'] for p in source.get('events',[]) if p['beat'] in failed]


def alternative(plan, records, failure, excluded, attempt, *, sample_rate, ffmpeg):
    from .compact import arrange_sync
    sections = plan.get('sections') or {}
    by_id = {row['id']: row for row in records}
    pair = (failure.outgoing, failure.incoming)
    edge = next((item for item in plan['edges'] if (item['from'],item['to'])==pair), None)
    if edge is None or not sections:
        raise RenderError('No automatic source windows are available for recovery')
    options = [(failure.incoming, 'start', 'locked_start'), (failure.outgoing, 'end', 'locked_end')]
    source_role=failure.evidence.get('failed_role') if failure.evidence.get('stage')=='source-only timing preflight' else None
    if source_role=='head':
        options=options[:1]
    elif source_role=='tail':
        options=options[1:]
    elif attempt%2 == 0:
        options.reverse()
    failed_track=failure.evidence.get('failed_track')
    failed_roles=failure.evidence.get('failed_roles',[])
    if failed_track and failed_roles:
        options=[item for item in options if item[0]==failed_track
                 and ('head' if item[1]=='start' else 'tail') in failed_roles]
    for identity, boundary, lock in options:
        section = sections[identity]
        if section.get(lock, True):
            continue
        proposal = deepcopy(plan)
        proposal.pop('sections', None)
        proposal.pop('rhythm_profiles', None)
        # A retry must not select the same unverified mandatory source event
        # under a different relative beat number in a nearly identical window.
        evidence=failure.evidence
        role='head' if boundary=='start' else 'tail'
        if identity==failed_track and role in failed_roles:
            rejected=unsupported_attack_events(evidence,role)
            if rejected:
                prior=proposal.setdefault('rejected_attack_window_events',{}).setdefault(identity,{}).setdefault(role,[])
                prior.extend(t for t in rejected if t not in prior)
        boundaries=proposal.setdefault('rejected_recovery_boundaries',{}).setdefault(identity,{})
        prior_boundaries=boundaries.setdefault(role,[])
        if section[boundary] not in prior_boundaries:prior_boundaries.append(section[boundary])
        for item in proposal['edges']:
            if boundary=='start' and item['to']==identity:
                item.setdefault('cue_locks', {}).pop('incoming_start_seconds', None)
            if boundary=='end' and item['from']==identity:
                item.setdefault('cue_locks', {}).pop('outgoing_end_seconds', None)
        previous = (section['start'],section['end'])
        blocked = deepcopy(excluded)
        blocked.setdefault(identity, []).append(previous)
        radius = 16*4*60/float(by_id[identity]['bpm'])
        bars = list(by_id[identity]['downbeats'])
        index = min(range(len(bars)),key=lambda i:abs(bars[i]-section[boundary]))
        choices = [bars[index+offset] for offset in (-16,-8,8,16) if 0<=index+offset<len(bars)]
        choices += [float(c.get('time',c.get('seconds',-1))) for c in by_id[identity].get('cue_candidates',[])
                    if c.get('reviewed') or float(c.get('confidence') or 0)>=.5]
        choices=[t for t in choices if all(abs(t-prior)>=1/sample_rate for prior in prior_boundaries)]
        # Native opener/ending boundaries have no edge cue lock. Holding only
        # edge locks allowed the solver to move the other end and retry the
        # same failed overlap while reporting that its boundary had changed.
        ranges={key:{b:[s[b],s[b]] for b in ('start','end')} for key,s in sections.items()}
        ranges[identity].update({boundary:[max(0.,section[boundary]-radius),section[boundary]+radius],
                                 boundary+'_choices':choices})
        preferences = proposal.get('mixing_preferences') or {}
        try:
            candidate = arrange_sync(proposal, records, sample_rate=sample_rate,
                zhilin=plan.get('style')=='zhilin', ffmpeg=ffmpeg,
                target_play_seconds=preferences.get('target_play_seconds',210.),
                maximum_play_seconds=preferences.get('maximum_play_seconds',300.),
                allow_long_play=bool(plan.get('long_play_explicit')),
                excluded_sections=blocked, section_ranges=ranges)
        except RenderError:
            continue
        if candidate['order'] != plan['order']:
            raise RuntimeError('Window recovery changed the approved order')
        if abs(candidate['sections'][identity][boundary]-section[boundary])<1/sample_rate:
            raise RuntimeError('Window recovery retained the failed source boundary')
        # Holding other automatic endpoints during this one trial must not turn
        # them into user locks in the next trial or a future revision.
        for key, value in candidate['sections'].items():
            for flag in ('locked_start','locked_end'):
                value[flag] = sections[key].get(flag,False)
        excluded.setdefault(identity, []).append(previous)
        return candidate, {'changed_track':identity,'boundary':boundary,'previous_section':list(previous),
            'new_section':[candidate['sections'][identity]['start'],candidate['sections'][identity]['end']],
            'order_preserved':True,'overlap_lengths_preserved':True}
    raise RenderError('No alternative window satisfies the same overlap, source, playback and lock constraints')


def render_with_recovery(plan, records, prepared, output, work, *, renderer=None, maximum_attempts=3, **controls):
    from .render import render_plan
    from .storage import write_json
    renderer = renderer or render_plan
    original = deepcopy(plan)
    candidate = deepcopy(plan)
    excluded, history = {}, []
    work = Path(work)
    for attempt in range(maximum_attempts):
        directory = work if attempt==0 else work/f'alternative-{attempt}'
        try:
            rendered = renderer(candidate, records, prepared, output, directory, **controls)
        except TransitionFailure as error:
            history.append({'attempt':attempt+1,'from':error.outgoing,'to':error.incoming,
                'reason':str(error),'evidence':error.evidence})
            write_json(work/'window_recovery.json', {'status':'searching','attempts':history})
            if attempt+1>=maximum_attempts:
                write_json(work/'window_recovery.json', {'status':'failed','attempts':history})
                raise
            try:
                candidate, selection = alternative(candidate, records, error, excluded, attempt+1,
                    sample_rate=controls.get('sample_rate',44100), ffmpeg=controls.get('ffmpeg'))
            except RenderError:
                write_json(work/'window_recovery.json', {'status':'failed','attempts':history})
                raise error
            history[-1]['selection'] = selection
            continue
        if candidate['order'] != original['order']:
            raise RuntimeError('Recovery changed track order')
        rendered['window_recovery'] = {'attempts':history,'alternatives_used':len(history),
            'constraints_relaxed':False,'maximum_attempts':maximum_attempts}
        return rendered, candidate
    raise RuntimeError('Unreachable recovery state')
