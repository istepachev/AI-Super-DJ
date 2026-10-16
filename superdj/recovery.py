"""Bounded alternative-window search; explicit montage constraints stay fixed."""
from copy import deepcopy
from pathlib import Path

from .render import RenderError
from .transition_quality import TransitionFailure


def alternative(plan, records, failure, excluded, attempt, *, sample_rate, ffmpeg):
    from .compact import arrange_sync
    sections = plan.get('sections') or {}
    by_id = {row['id']: row for row in records}
    pair = (failure.outgoing, failure.incoming)
    edge = next((item for item in plan['edges'] if (item['from'],item['to'])==pair), None)
    if edge is None or not sections:
        raise RenderError('No automatic source windows are available for recovery')
    options = [(failure.incoming, 'start', 'locked_start'), (failure.outgoing, 'end', 'locked_end')]
    if attempt%2 == 0:
        options.reverse()
    for identity, boundary, lock in options:
        section = sections[identity]
        if section.get(lock, True):
            continue
        proposal = deepcopy(plan)
        proposal.pop('sections', None)
        proposal.pop('rhythm_profiles', None)
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
        ranges = {identity: {boundary: [max(0., section[boundary]-radius), section[boundary]+radius],
                             boundary+'_choices':choices}}
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
