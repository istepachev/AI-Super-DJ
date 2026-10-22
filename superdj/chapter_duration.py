"""Retain chapter duration while choosing a genuinely different source edge."""
from copy import deepcopy

def retain_chapter_target(candidate, records, identity, changed_role, *, sample_rate, ffmpeg):
    from .compact import arrange_sync
    from .render import RenderError
    preferences=candidate.get('mixing_preferences') or {}
    section=candidate['sections'][identity]
    if (preferences.get('duration_budget_basis')!='chapter interval'
            or section.get('playback_intent') or section['predicted_chapter_seconds']>=180):
        return candidate, None
    # Retain the newly chosen entrance/exit. Search its other automatic boundary
    # before concluding that this recording must have a shorter chapter.
    boundary, role, lock=('end','tail','locked_end') if changed_role=='head' else ('start','head','locked_start')
    position=candidate['order'].index(identity)
    if section.get(lock):
        return candidate, None
    proposal=deepcopy(candidate)
    proposal.pop('sections',None);proposal.pop('rhythm_profiles',None)
    for edge in proposal['edges']:
        if boundary=='end' and edge['from']==identity:
            edge.setdefault('cue_locks',{}).pop('outgoing_end_seconds',None)
        if boundary=='start' and edge['to']==identity:
            edge.setdefault('cue_locks',{}).pop('incoming_start_seconds',None)
    ranges={k:{b:[s[b],s[b]] for b in ('start','end')} for k,s in candidate['sections'].items()}
    row=next(r for r in records if r['id']==identity)
    rejected=(proposal.get('rejected_recovery_boundaries') or {}).get(identity,{}).get(role,[])
    ranges[identity].pop(boundary)
    choices=list(row['downbeats'])
    if boundary=='start' and position==0:choices.append(0.)
    if boundary=='end' and position==len(records)-1:
        choices.append(float(row.get('duration',row.get('duration_seconds'))))
    ranges[identity][boundary+'_choices']=[float(t) for t in choices
        if all(abs(t-old)>=1/sample_rate for old in rejected)]
    try:
        result=arrange_sync(proposal,records,sample_rate=sample_rate,ffmpeg=ffmpeg,
            zhilin=candidate.get('style')=='zhilin',target_play_seconds=preferences.get('target_play_seconds',210.),
            maximum_play_seconds=preferences.get('maximum_play_seconds',300.),
            allow_long_play=bool(candidate.get('long_play_explicit')),section_ranges=ranges)
    except RenderError:
        return candidate, None
    selected=result['sections'][identity]
    if selected['predicted_chapter_seconds']<=section['predicted_chapter_seconds']+1/sample_rate:
        return candidate, None
    assert result['order']==candidate['order']
    assert [(e.get('overlap_bars'),e.get('overlap_beats'))for e in result['edges']]==[
        (e.get('overlap_bars'),e.get('overlap_beats'))for e in candidate['edges']]
    for k,s in result['sections'].items():
        for flag in ('locked_start','locked_end'):s[flag]=candidate['sections'][k].get(flag,False)
        for b in ('start','end'):
            if k!=identity or b!=boundary:
                assert abs(s[b]-candidate['sections'][k][b])<1/sample_rate
    return result,{'changed_other_automatic_boundary':boundary,
        'previous_chapter_seconds':section['predicted_chapter_seconds'],
        'new_chapter_seconds':selected['predicted_chapter_seconds'],
        'previous_section':[section['start'],section['end']],
        'new_section':[selected['start'],selected['end']],
        'known_failed_boundary_preserved':True,'overlap_lengths_preserved':True}
