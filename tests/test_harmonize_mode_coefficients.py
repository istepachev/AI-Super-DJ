"""Independent mode updates retain the other mode's exact float calculation."""
import copy
import json

import numpy as np
import pytest

from superdj.harmonize_modes import (rank_mode_candidates, remaining_features,
                                    validate_mode_ranker_profile)
from superdj.harmonize_modes_search import DynamicModePool
from superdj.presets import ordering_profile


def records():
    return [dict(id=name, bpm=bpm, key_camelot=key) for name, bpm, key in (
        ('a', 128, '8A'), ('b', 130, '8A'), ('c', 120, '9A'),
        ('d', 128, '8B'), ('e', 160, '2A'), ('f', 132, '9A'), ('g', 128, '7A'))]


def model_pair():
    base = ordering_profile('harmonic-next-v1')
    base.pop('mode_coefficients', None)
    base.update(schema_version=3, model_version='superdj-mode-priority-ranker-3')
    updated = copy.deepcopy(base)
    updated.update(schema_version=4, model_version='superdj-mode-priority-ranker-4')
    values = [0.] * len(base['feature_names'])
    values[base['feature_names'].index('abs_bpm_delta/10')] = -10.
    updated['mode_coefficients'] = {'fuzzy': values}
    return base, updated


def test_mode_update_preserves_exact_mood_and_uses_fuzzy_vector():
    rows = records()
    base, updated = model_pair()
    before = copy.deepcopy((rows, base, updated))
    for priority, tolerance in (('unknown', None), ('key', None), ('bpm', 6.)):
        kwargs = dict(priority=priority, bpm_tolerance=tolerance)
        old = rank_mode_candidates(rows[0], rows[1:], base, mode='mood', **kwargs)
        new = rank_mode_candidates(rows[0], rows[1:], updated, mode='mood', **kwargs)
        assert old['candidates'] == new['candidates']
        assert old['shortlist_ids'] == new['shortlist_ids']
        fuzzy = rank_mode_candidates(rows[0], rows[1:], updated, mode='fuzzy', **kwargs)
        assert fuzzy['shortlist_ids'] == ['d', 'g', 'b']
        assert len(fuzzy['shortlist_ids']) == 3
        assert next(r for r in fuzzy['candidates'] if r['id'] == 'e')['eligible'] is False
    assert (rows, base, updated) == before


@pytest.mark.parametrize('mode', ['fuzzy', 'mood'])
@pytest.mark.parametrize('priority,tolerance', [('key', None), ('bpm', 6.), ('unknown', None)])
def test_dynamic_route_and_literal_rank_use_same_mode_weights(mode, priority, tolerance):
    rows = records()
    _, updated = model_pair()
    updated['mode_coefficients']['fuzzy'] = np.linspace(-.21, .13, len(updated['feature_names'])).tolist()
    engine = DynamicModePool(rows, updated, mode=mode, priority=priority, bpm_tolerance=tolerance)
    for mask in (0b1111110, 0b1011010):
        indices, values, _ = engine.scores(0, mask)
        pool = [rows[int(i)] for i in indices]
        matrix = remaining_features(rows[0], pool, mode=mode, family=updated['feature_family'],
            sharing=updated['sharing'], priority_aware=True, priority=priority, bpm_tolerance=tolerance)
        weights = updated['mode_coefficients'].get(mode, updated['coefficients']['all'])
        np.testing.assert_allclose(values, matrix @ np.asarray(weights), atol=1e-11, rtol=1e-12)
        result = rank_mode_candidates(rows[0], pool, updated, mode=mode, priority=priority, bpm_tolerance=tolerance)
        actual = {r['id']:r['utility'] for r in result['candidates']}
        assert [actual[r['id']] for r in pool] == list(matrix @ np.asarray(weights))


@pytest.mark.parametrize('overrides', [None, {}, [], {'classic': []}, {'fuzzy': []},
    {'fuzzy': [0.]}, {'fuzzy': 'bad'}])
def test_invalid_override_maps_and_widths_are_rejected(overrides):
    _, updated = model_pair()
    updated['mode_coefficients'] = overrides
    with pytest.raises(ValueError, match='[Mm]ode coefficient'):
        validate_mode_ranker_profile(updated)


@pytest.mark.parametrize('bad', [True, None, float('nan'), float('inf'), -float('inf')])
def test_mode_weights_must_be_finite_numbers(bad):
    _, updated = model_pair()
    updated['mode_coefficients']['fuzzy'][0] = bad
    with pytest.raises(ValueError):
        validate_mode_ranker_profile(updated)


def test_version_boundaries_and_json_round_trip():
    base, updated = model_pair()
    base['mode_coefficients'] = updated['mode_coefficients']
    with pytest.raises(ValueError, match='version 4'):
        validate_mode_ranker_profile(base)
    wrong = copy.deepcopy(updated)
    wrong['schema_version'] = 3
    with pytest.raises(ValueError, match='version'):
        validate_mode_ranker_profile(wrong)
    validated = validate_mode_ranker_profile(json.loads(json.dumps(updated, allow_nan=False)))
    assert validated == updated


def test_packaged_fuzzy_vector_and_original_mood_survive_portable_loading():
    published = ordering_profile('harmonic-next-v1')
    assert published['schema_version'] == 4
    assert set(published['mode_coefficients']) == {'fuzzy'}
    original = copy.deepcopy(published)
    del original['mode_coefficients']
    original.update(schema_version=3, model_version='superdj-mode-priority-ranker-3')
    rows = records()
    for mode in ('fuzzy', 'mood'):
        expected = copy.deepcopy(original)
        if mode == 'fuzzy':
            expected['coefficients']['all'] = published['mode_coefficients']['fuzzy']
        before = rank_mode_candidates(rows[0], rows[1:], expected, mode=mode, priority='bpm', bpm_tolerance=6.)
        after = rank_mode_candidates(rows[0], rows[1:], published, mode=mode, priority='bpm', bpm_tolerance=6.)
        assert before['candidates'] == after['candidates']
        assert before['shortlist_ids'] == after['shortlist_ids']
