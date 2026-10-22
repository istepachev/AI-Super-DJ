"""Set-wide chapter pacing; short performances cannot become the default."""
from math import ceil, isfinite

MINIMUM_FRACTION = .80
PREFERRED_FRACTION = .90
TOLERANCE_SECONDS = .05


def playback_intent(record):
    """A musical exception is selected explicitly, never guessed from energy."""
    value = record.get('playback_intent')
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'kind', 'reason'}
            or value['kind'] not in {'brief', 'highlight'}
            or not isinstance(value['reason'], str) or not value['reason'].strip()):
        raise ValueError('playback_intent requires kind brief/highlight and a nonempty reason')
    return dict(value)


def audit_playtime(plan, records, *, tracks=None, total_frames=None, sample_rate=None,
                   require=True):
    """Count actual CUE intervals, not overlapping deck time or solo time.

    Truly insufficient supplied material/user locks remain named constraints.
    They never authorize avoidable shortening of the remaining long sources.
    A whole-set explicit long-play request retains its separate contract.
    """
    preferences = plan.get('mixing_preferences') or {}
    if preferences.get('duration_budget_basis') != 'chapter interval' or plan.get('long_play_explicit'):
        return {'applicable': False, 'reason': 'legacy duration basis or explicit whole-set long play'}
    order = plan['order']
    by_id = {str(r['id']): r for r in records}
    if not order or len(set(order)) != len(order) or set(order) != set(by_id):
        raise ValueError('Chapter pacing requires every unique selected source exactly once')
    actual = None
    if tracks is not None:
        if [t['id'] for t in tracks] != order or not sample_rate or total_frames is None:
            raise ValueError('Actual chapter audit requires ordered rendered frames')
        starts = [t['set_start_frame'] for t in tracks] + [total_frames]
        if starts[0] != 0 or any(b <= a for a,b in zip(starts,starts[1:])):
            raise ValueError('CUE starts must be strictly increasing from the first sample')
        actual = [(b-a)/sample_rate for a,b in zip(starts,starts[1:])]
    normal = 0
    exceptions = []
    unavoidable = []
    for position, identity in enumerate(order):
        section = plan['sections'][identity]
        seconds = actual[position] if actual is not None else float(section['predicted_chapter_seconds'])
        if not isfinite(seconds) or seconds<=0 or (actual is not None and seconds < 60-TOLERANCE_SECONDS):
            raise ValueError(f'{identity}: chapter is below the one-minute floor or nonfinite')
        intent = playback_intent(by_id[identity])
        if 180-TOLERANCE_SECONDS <= seconds <= 240+TOLERANCE_SECONDS:
            normal += 1
            continue
        # Optimistic remaining native capacity: automatic ends may move all
        # the way to the native source bounds. Temporary search ranges are
        # not user locks. This cannot classify a long, merely cropped track
        # as a physically short source.
        native = float(by_id[identity].get('duration',by_id[identity].get('duration_seconds')))
        capacity = float(section['predicted_chapter_seconds'])
        if not section.get('locked_start'):
            capacity += max(0.,float(section['start']))
        if not section.get('locked_end'):
            capacity += max(0.,native-float(section['end']))
        if section.get('minimum_repeat'):
            from .minimum_timeline import recipe
            _,_,added,_ = recipe(section['minimum_repeat'])
            capacity -= added
        constrained = seconds < 180 and capacity < 180-TOLERANCE_SECONDS and intent is None
        reason = section.get('chapter_duration_exception_reason')
        if intent:
            reason = f"selected {intent['kind']}: {intent['reason']}"
        elif constrained:
            reason = ('supplied native source/user-locked span cannot cover a three-minute chapter')
        if not reason:
            raise ValueError(f'{identity}: chapter outside 3–4 minutes has no named exception')
        row = {'id':identity,'chapter_seconds':seconds,'reason':reason,
               'unavoidable_supplied_material_constraint':constrained,
               'optimistic_native_chapter_capacity_seconds':capacity}
        exceptions.append(row)
        if constrained:
            unavoidable.append(identity)
    count = len(order)
    minimum = ceil(MINIMUM_FRACTION*count-1e-9)
    preferred = ceil(PREFERRED_FRACTION*count-1e-9)
    # Retain explicitly supplied short recordings. Reduce only the impossible
    # part of the requirement; do not exempt every other automatic exception.
    required = min(minimum,count-len(unavoidable))
    passed = normal >= required
    result = {'applicable':True,'passed':passed,'measurement':'actual CUE frames' if actual is not None else 'predicted CUE chapters',
        'track_count':count,'normal_count':normal,'normal_fraction':normal/count,
        'minimum_fraction':MINIMUM_FRACTION,'preferred_fraction':PREFERRED_FRACTION,
        'minimum_normal_count':minimum,'preferred_normal_count':preferred,
        'required_normal_count_for_supplied_material':required,
        'status':('constrained_by_supplied_material' if passed and normal<minimum else 'satisfied' if passed else 'too_many_automatic_exceptions'),
        'unavoidable_source_constraints':unavoidable,'exceptions':exceptions}
    if require and not passed:
        raise ValueError(f'Golden 3–4 minute rule: {normal}/{count} chapters meet the target; '
                         f'at least {required} are required (80%, preferred 90%). '
                         'Choose longer compatible source windows or revise automatic selection; '
                         'do not publish a mostly two-minute set.')
    return result
