"""Complete-route musical safety for explicitly selected ordering presets."""
from .harmonize_modes import validate_mode_ranker_profile
from .harmonize_modes_sequence import sequence_mode_tracks
from .presets import ordering_profile


def sequence_ordering_preset(records, name, **settings):
    """Retain the primary route unless a full alternative has fewer bad links.

    Both models receive the same actual musical pool and explicit constraints.
    Relative utilities from different fitted models are not comparable scores.
    Reference orders play no part in generation or selection.
    """
    records = sorted(list(records), key=lambda row: str(row.get('id') or row.get('path')))
    profile = ordering_profile(name)
    primary = sequence_mode_tracks(records, profile, **settings)
    if name != 'harmonic-route-v2':
        return primary
    safety_profile = validate_mode_ranker_profile(profile['safety_profile'])
    safety = sequence_mode_tracks(records, safety_profile, **settings)
    expected = {str(row.get('id') or row.get('path')) for row in records}
    for candidate in (primary, safety):
        if len(candidate['order']) != len(records) or set(candidate['order']) != expected:
            raise RuntimeError('A preset route changed input occurrence membership')
    bad = lambda candidate: len(candidate['edges']) - candidate['harmonic_matches']
    primary_bad, safety_bad = bad(primary), bad(safety)
    use_safety = safety_bad < primary_bad
    result = dict(safety if use_safety else primary)
    result['ordering_preset'] = name
    result['preset_route_selection'] = {
        'policy_version': 'native-harmony-safety-1',
        'selected': 'safety' if use_safety else 'primary',
        'criterion': 'Strictly fewer outside-mode edges across the complete route; ties retain primary',
        'teacher_order_used': False,
        'cross_model_utility_compared': False,
        'primary_outside_mode_edges': primary_bad,
        'safety_outside_mode_edges': safety_bad,
        'candidates': [
            {'role': 'primary', 'order': primary['order'], 'outside_mode_edges': primary_bad},
            {'role': 'safety', 'order': safety['order'], 'outside_mode_edges': safety_bad},
        ],
        'scope': 'Two bounded complete routes; musical safety only, no global-optimum or project archive identity claim',
    }
    return result
