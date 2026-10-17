"""Full rebuild search: a different intro cannot repair the same failed exit."""
from copy import deepcopy

from .compact import arrange_sync
from .recovery import unsupported_attack_events
from .render import RenderError


def fresh_arrangement_alternative(plan, records, error, excluded, attempt, *, sample_rate, ffmpeg):
    evidence = error.evidence
    identity = evidence.get('failed_track') or (error.incoming if attempt % 2 else error.outgoing)
    sections = plan['sections']
    old = sections[identity]
    roles = set(evidence.get('failed_roles') or
                (['head'] if identity == error.incoming else ['tail']))
    position = plan['order'].index(identity)
    roles &= ({'head'} if position else set()) | ({'tail'} if position < len(plan['order'])-1 else set())
    boundaries = [('head', 'start', 'locked_start'), ('tail', 'end', 'locked_end')]
    movable = [(r,b) for r,b,lock in boundaries if r in roles and not old.get(lock,False)]
    if any(old.get(lock,False) for role,_,lock in boundaries if role in roles):
        raise RenderError('A failed source boundary is user locked') from error
    if not movable:
        raise RenderError('The failed source boundary is locked or has no overlap') from error
    proposal = deepcopy(plan)
    proposal.pop('sections', None)
    proposal.pop('rhythm_profiles', None)
    rejected = proposal.setdefault('rejected_recovery_boundaries', {}).setdefault(identity, {})
    for role,boundary in movable:
        rejected.setdefault(role, []).append(old[boundary])
        events = proposal.setdefault('rejected_attack_window_events', {}).setdefault(identity, {}).setdefault(role, [])
        events.extend(t for t in unsupported_attack_events(evidence,role) if t not in events)
        for edge in proposal['edges']:
            if role == 'head' and edge['to'] == identity:
                edge.setdefault('cue_locks', {}).pop('incoming_start_seconds',None)
            if role == 'tail' and edge['from'] == identity:
                edge.setdefault('cue_locks', {}).pop('outgoing_end_seconds',None)
    blocked = deepcopy(excluded)
    blocked.setdefault(identity, []).append([old['start'],old['end']])
    # Hold every other source boundary, including the native opener/ending
    # which have no edge lock. Temporary search bounds are never user locks.
    ranges = {key:{b:[s[b],s[b]] for b in ('start','end')} for key,s in sections.items()}
    row = next(r for r in records if r['id'] == identity)
    choices = [float(t) for t in row['downbeats']]
    for role,boundary in movable:
        ranges[identity].pop(boundary)
        ranges[identity][boundary+'_choices'] = [t for t in choices
            if all(abs(t-prior) >= 1/sample_rate for prior in rejected[role])]
        if not ranges[identity][boundary+'_choices']:
            raise RenderError('All distinct source boundaries are exhausted') from error
    preferences = plan.get('mixing_preferences') or {}
    for _ in range(16):
        candidate = arrange_sync(proposal,records,sample_rate=sample_rate,zhilin=True,ffmpeg=ffmpeg,
            target_play_seconds=preferences.get('target_play_seconds',210.),
            maximum_play_seconds=preferences.get('maximum_play_seconds',300.),
            excluded_sections=blocked,section_ranges=ranges)
        selected = candidate['sections'][identity]
        if selected['predicted_play_seconds'] >= 120:
            break
        blocked[identity].append([selected['start'],selected['end']])
    else:
        raise RenderError('No distinct failed-boundary candidate retains 120 seconds') from error
    assert candidate['order'] == plan['order']
    assert [e.get('overlap_bars') for e in candidate['edges']] == [e.get('overlap_bars') for e in plan['edges']]
    for key,value in candidate['sections'].items():
        for flag in ('locked_start','locked_end'):
            value[flag] = sections[key].get(flag,False)
    for role,boundary in movable:
        assert abs(selected[boundary]-old[boundary]) >= 1/sample_rate
    excluded.setdefault(identity, []).append([old['start'],old['end']])
    return candidate, {'changed_track':identity,'boundary':','.join(b for _,b in movable),
        'previous_section':[old['start'],old['end']], 'new_section':[selected['start'],selected['end']],
        'order_preserved':True,'overlap_lengths_preserved':True,'actual_user_locks_preserved':True,
        'same_failed_boundary_cannot_be_retried':True}
