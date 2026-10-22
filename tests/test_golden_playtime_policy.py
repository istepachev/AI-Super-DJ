"""A few short/featured tracks cannot turn into a mostly short set."""
from copy import deepcopy
import pytest

from superdj.compact import arrange_sync
from superdj.planner import plan_tracks
from superdj.playtime_policy import audit_playtime, playback_intent
from superdj.render import _validate_playback_budget
from test_new_workflow import record


def fixture(seconds, native_short=()):
    rows=[record(str(i),150 if i in native_short else 600) for i in range(len(seconds))]
    sections={str(i):{'start':0 if i in native_short else 100.,
        'end':150 if i in native_short else 100.+v+64.,
        'predicted_chapter_seconds':v,'predicted_play_seconds':v+64.,
        'chapter_duration_exception_reason':None if 180<=v<=240 else 'long compatible windows were unavailable',
        'locked_start':False,'locked_end':False} for i,v in enumerate(seconds)}
    plan={'order':[r['id'] for r in rows],'sections':sections,
          'mixing_preferences':{'duration_budget_basis':'chapter interval'}}
    return plan,rows


@pytest.mark.parametrize('normal', [8,9,10])
def test_eighty_percent_is_the_floor_and_ninety_is_the_preference(normal):
    plan,rows=fixture([210]*normal+[120]*(10-normal))
    result=audit_playtime(plan,rows)
    assert result['passed'] and result['normal_fraction']==normal/10
    assert result['minimum_normal_count']==8 and result['preferred_normal_count']==9


@pytest.mark.parametrize('short', [120,90])
def test_a_majority_of_two_or_one_and_half_minute_chapters_is_rejected(short):
    plan,rows=fixture([short]*10)
    with pytest.raises(ValueError,match='Golden 3–4 minute rule'):
        audit_playtime(plan,rows)


def test_a_third_automatic_exception_cannot_silently_break_a_ten_track_set():
    plan,rows=fixture([210]*7+[120]*3)
    with pytest.raises(ValueError,match='7/10'):
        audit_playtime(plan,rows)


def test_required_short_sources_are_named_without_aborting_or_exempting_long_tracks():
    plan,rows=fixture([210]*7+[110]*3,native_short=(7,8,9))
    result=audit_playtime(plan,rows)
    assert result['status']=='constrained_by_supplied_material'
    assert result['unavoidable_source_constraints']==['7','8','9']
    assert result['required_normal_count_for_supplied_material']==7
    plan['sections']['6'].update(predicted_chapter_seconds=120.,end=284.,
        chapter_duration_exception_reason='failed automatic windows')
    with pytest.raises(ValueError,match='6/10'):
        audit_playtime(plan,rows)


def test_actual_cue_frames_override_reassuring_full_deck_or_planned_durations():
    plan,rows=fixture([210]*10)
    for section in plan['sections'].values():
        section['chapter_duration_exception_reason']='automatic transition recovery'
    tracks=[{'id':str(i),'set_start_frame':i*120*8000,'output_frames':210*8000} for i in range(10)]
    with pytest.raises(ValueError,match='0/10'):
        audit_playtime(plan,rows,tracks=tracks,total_frames=10*120*8000,sample_rate=8000)


def test_actual_one_minute_floor_remains_separate_from_the_soft_three_minute_target():
    plan,rows=fixture([45],native_short=(0,))
    assert audit_playtime(plan,rows)['status']=='constrained_by_supplied_material'
    with pytest.raises(ValueError,match='one-minute floor'):
        audit_playtime(plan,rows,tracks=[{'id':'0','set_start_frame':0}],total_frames=45*8000,sample_rate=8000)


def test_temporary_recovery_bounds_do_not_turn_a_long_source_into_a_short_source():
    plan,rows=fixture([120]*10)
    plan['rejected_recovery_boundaries']={r['id']:{'head':[0,20,40]} for r in rows}
    result=audit_playtime(plan,rows,require=False)
    assert not result['passed'] and not result['unavoidable_source_constraints']


def test_optional_loop_does_not_erase_the_original_short_source_constraint():
    plan,rows=fixture([150])
    rows[0]['duration']=250
    plan['sections']['0'].update(start=0.,end=210.,minimum_repeat={
        'method':'verified phrase repeat with complementary-band join',
        'loop_start_seconds':20.,'insertion_source_seconds':40.,'added_seconds':60.,
        'original_source_duration_seconds':190.,'repeat_count':3,'sample_rate':8000,'bars_repeated':8})
    assert audit_playtime(plan,rows)['status']=='constrained_by_supplied_material'


@pytest.mark.parametrize('position,role',[(0,'tail'),(2,'head')])
def test_recovery_can_retain_long_chapters_at_the_first_and_last_track(position,role):
    from superdj.chapter_duration import retain_chapter_target
    rows=[record(str(i),600) for i in range(3)]
    plan=plan_tracks(rows,mode='fixed')
    plan['mixing_preferences']={'duration_budget_basis':'chapter interval'}
    plan=arrange_sync(plan,rows)
    section=plan['sections'][str(position)]
    if role=='tail':
        # The failed exit was moved earlier; its automatic opener may move
        # earlier too. It has no incoming edge lock, but is still searchable.
        plan['edges'][0]['cue_locks']['outgoing_end_seconds']=section['end']-80
    else:
        plan['edges'][1]['cue_locks']['incoming_start_seconds']=section['start']+80
    ranges={k:{b:[v[b],v[b]] for b in ('start','end')} for k,v in plan['sections'].items()}
    boundary='end' if role=='tail' else 'start'
    moved=plan['edges'][0]['cue_locks']['outgoing_end_seconds'] if role=='tail' else plan['edges'][1]['cue_locks']['incoming_start_seconds']
    ranges[str(position)][boundary]=[moved,moved]
    short=arrange_sync(plan,rows,section_ranges=ranges)
    for value in short['sections'].values():
        value['locked_start']=value['locked_end']=False
    assert short['sections'][str(position)]['predicted_chapter_seconds']<180
    result,proof=retain_chapter_target(short,rows,str(position),role,sample_rate=8000,ffmpeg=None)
    assert proof and 180<=result['sections'][str(position)]['predicted_chapter_seconds']<=240
    assert all(e['overlap_bars']==32 for e in result['edges'])


def test_explicit_brief_and_highlight_leave_eighty_percent_ordinary_without_shortening_sync():
    rows=[record(str(i),600 if i==8 else 360) for i in range(10)]
    rows[8]['playback_intent']={'kind':'highlight','reason':'feature this recording'}
    rows[9]['playback_intent']={'kind':'brief','reason':'brief supporting passage'}
    plan=plan_tracks(rows,mode='fixed')
    plan['mixing_preferences']={'duration_budget_basis':'chapter interval'}
    result=arrange_sync(plan,rows,sample_rate=8000)
    audit=audit_playtime(result,rows)
    assert audit['normal_count']==8 and audit['passed']
    assert all(e['overlap_bars']==32 for e in result['edges'])
    for i in range(8):
        assert 180<=result['sections'][str(i)]['predicted_chapter_seconds']<=240
        assert result['sections'][str(i)]['predicted_play_seconds']<=300
    highlight=result['sections']['8'];brief=result['sections']['9']
    assert 270<=highlight['predicted_chapter_seconds']<=300
    assert highlight['predicted_play_seconds']>300 and highlight['allow_long_play']
    assert 90<=brief['predicted_chapter_seconds']<=150
    _validate_playback_budget({'id':'8','output_frames':round(highlight['predicted_play_seconds']*8000)},highlight,8000)
    assert not result['long_play_explicit']


@pytest.mark.parametrize('intent',[{}, {'kind':'brief','reason':''}, {'kind':'whatever','reason':'x'}, 'brief'])
def test_unjustified_or_ambiguous_musical_exceptions_are_refused(intent):
    with pytest.raises(ValueError,match='playback_intent'):
        playback_intent({'playback_intent':intent})
