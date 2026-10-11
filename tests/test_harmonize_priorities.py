"""Compatibility cannot be bought away by tempo convenience; all-source path tests."""
import itertools

import pytest

from superdj.harmonize import harmonize_tracks, summarize_order


def records(values):
    return [dict(id=str(i), bpm=bpm, key_camelot=key) for i, (bpm, key) in enumerate(values)]


def objective(rows, order, **settings):
    summary = summarize_order(rows, order, **settings)
    return (len(summary['edges']) - summary['harmonic_matches'],
            summary['adventurous_matches'], summary['total_cost'])


def test_default_finds_true_lexicographic_optimum_not_old_weighted_minimum():
    rows = records([(98, '8A'), (125, '8A'), (100, '2A'),
                    (140, '3A'), (121, '3B'), (130, '9A')])
    settings = dict(mode='fuzzy', priority=.1, direction='ascending')
    result = harmonize_tracks(rows, first_id='0', last_id='4', **settings)
    candidates = [['0', *middle, '4'] for middle in itertools.permutations(['1', '2', '3', '5'])]
    optimum = min(objective(rows, order, **settings) for order in candidates)
    actual = objective(rows, result['order'], **settings)
    assert actual[:2] == optimum[:2]
    assert actual[2] == pytest.approx(optimum[2])
    assert result['order_objective'] == 'compatible_first'
    assert result['version'] == 'superdj-harmonize-4'
    assert result['search_cost'] <= result['baseline_search_cost'] + 1e-9
    old = harmonize_tracks(rows, first_id='0', last_id='4', order_objective='weighted', **settings)
    assert objective(rows, result['order'], **settings)[:2] < objective(rows, old['order'], **settings)[:2]


def test_preferred_route_keeps_all_tracks_and_chain_constant_edges():
    rows = records([(110, '8A'), (115, '2A'), (120, '3A'), (118, '3B'), (124, '9A')])
    result = harmonize_tracks(rows, first_id='0', locked_chains=[['0', '1']], last_id='4')
    optimum = min(objective(rows, ['0', '1', *middle, '4']) for middle in itertools.permutations(['2', '3']))
    assert result['order'][:2] == ['0', '1']
    assert result['order'][-1] == '4'
    assert objective(rows, result['order'])[:2] == optimum[:2]
    assert result['edges'][0]['locked_adjacency']
    assert result['edges'][0]['harmonic']['accepted'] is False


def test_mood_does_not_categorically_suppress_its_expressive_relations():
    rows = records([(120, '8B'), (120, '10B'), (155, '9B'), (120, '8A')])
    settings = dict(mode='mood', priority=.1, direction='none')
    candidates = [['0', *middle] for middle in itertools.permutations(['1', '2', '3'])]
    def mood_objective(order):
        summary = summarize_order(rows, order, **settings)
        return len(summary['edges'])-summary['harmonic_matches'], summary['total_cost']
    result = harmonize_tracks(rows, first_id='0', **settings)
    optimum = min(mood_objective(order) for order in candidates)
    assert mood_objective(result['order'])[0] == optimum[0]
    assert mood_objective(result['order'])[1] == pytest.approx(optimum[1])
    assert result['objective_details']['accepted_review_weight'] == 0
    assert result['objective_tuple'][1] == 0
    assert result['objective_details']['priority_order'] == [
        'fewest unaccepted/unknown key edges', 'lowest declared weighted BPM/key cost']


def test_milp_integrates_group_locks_and_reports_its_objective_proof():
    rows = records([(120, '8A'), (124, '9A'), (118, '8B'), (122, '9B'), (126, '10B')])
    result = harmonize_tracks(rows, first_id='0', last_id='4',
                             locked_chains=[['1', '3']], exact_limit=1, beam_width=5)
    assert result['order'][0] == '0' and result['order'][-1] == '4'
    i = result['order'].index('1')
    assert result['order'][i:i+2] == ['1', '3']
    assert len(set(result['order'])) == len(rows)
    assert 'MILP' in result['method']
    assert result['solver']['cost'] == pytest.approx(result['search_cost'])
    assert result['optimal_for_declared_objective']
    assert result['search_improvement'] >= -1e-9


def test_solver_without_incumbent_keeps_complete_fallback(monkeypatch):
    monkeypatch.setattr('superdj.harmonize_route.solve_hamiltonian_path', lambda *a, **kw: None)
    rows = records([(120, '8A'), (124, '9A'), (118, '8B'), (126, '10B')])
    result = harmonize_tracks(rows, first_id='0', exact_limit=1, beam_width=5)
    assert result['order'][0] == '0'
    assert set(result['order']) == {'0', '1', '2', '3'}
    assert not result['optimal_for_declared_objective']
    assert result['solver'] is None
    assert any('no valid incumbent' in text for text in result['uncertainty']['warnings'])


@pytest.mark.parametrize('kwargs', [dict(order_objective='unknown'), dict(search_seconds=0),
                                     dict(search_seconds=float('nan')), dict(search_seconds=121),
                                     dict(search_seconds=True)])
def test_search_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        harmonize_tracks(records([(120, '8A')]), **kwargs)
