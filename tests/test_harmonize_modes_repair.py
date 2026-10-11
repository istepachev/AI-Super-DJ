import random

import pytest

from superdj.harmonize_modes_repair import _repair_route, repair_mode_order


def tracks(keys):
    return [dict(id=str(i), bpm=120 + i, key_camelot=key) for i, key in enumerate(keys)]


def test_repair_fixes_directed_path_trap_preserving_chain_and_endpoints():
    rows = tracks(['8B', '9B', '10B', '7B', '8B'])
    result = repair_mode_order(rows, ['0', '1', '3', '4', '2'], first_id='0',
                               last_id='2', locked_chains=[['3', '4']])
    assert result['order'] == ['0', '3', '4', '1', '2']
    assert result['unaccepted_before'] == 2
    assert result['unaccepted_after'] == 0
    assert result['changed'] and result['all_inputs_once']
    assert result['constraints']['locked_chains'] == [['3', '4']]
    assert result['accepted_moves'][0]['unaccepted_boundaries_after'] == 0


def test_compatible_incumbent_unchanged_despite_large_tempo_gains():
    rows = [dict(id=str(i), bpm=bpm, key_camelot='8B')
            for i, bpm in enumerate([120, 160, 121, 159, 122])]
    order = ['0', '1', '2', '3', '4']
    result = repair_mode_order(rows, order, first_id='0')
    assert result['order'] == order
    assert not result['changed']
    assert result['evaluated_candidates'] == 0
    assert result['incumbent_track_edges_changed'] == 0


def test_bad_user_lock_is_preserved_and_counted_separately():
    rows = tracks(['8B', '2B', '3B'])
    result = repair_mode_order(rows, ['0', '1', '2'], first_id='0',
                               locked_chains=[['0', '1']])
    assert result['order'] == ['0', '1', '2']
    assert result['locked_unaccepted_constant'] == 1
    assert result['unaccepted_boundaries_before'] == 0
    assert result['unaccepted_after'] == 1


def test_unknown_track_stays_present_and_no_feasibility_claim_is_made():
    result = repair_mode_order(tracks(['8B', None, '9B']), ['0', '1', '2'], first_id='0')
    assert result['order'] == ['0', '2', '1']
    assert result['unaccepted_before'] == 2 and result['unaccepted_after'] == 1
    assert 'audio' in result['scope']
    assert not result['optimal_for_declared_objective']


def test_strict_budget_counts_every_examined_move_and_zero_budget_preserves_order():
    rows = tracks(['8B', '9B', '10B', '7B', '8B'])
    order = ['0', '1', '3', '4', '2']
    stopped = repair_mode_order(rows, order, first_id='0', max_candidates=0)
    assert stopped['order'] == order and stopped['evaluated_candidates'] == 0
    limited = repair_mode_order(rows, order, first_id='0', max_candidates=3)
    assert limited['evaluated_candidates'] <= 3
    assert limited['budget_exhausted']
    assert limited['unaccepted_after'] <= limited['unaccepted_before']


def test_static_boundary_tie_cost_cannot_buy_an_extra_forbidden_edge():
    rows = tracks(['8B', '9B', '10B', '7B', '8B'])
    order = ['0', '1', '3', '4', '2']
    # Every changed boundary is deliberately far more expensive than retaining
    # the two bad incumbent links; fewer forbidden edges still take precedence.
    costs = [[10**12]*5 for _ in range(5)]
    for i in range(4):
        costs[i][i+1] = 0
    result = repair_mode_order(rows, order, first_id='0', last_id='2', boundary_costs=costs)
    assert result['unaccepted_before'] == 2 and result['unaccepted_after'] == 0
    assert result['boundary_cost_source'] == 'custom static matrix'


def test_long_stage_repairs_suffix_trap_that_short_moves_cannot_fix():
    # The initial 8B is locked. A long 10B tail cannot connect to 7B; moving
    # fewer than four of its tracks leaves the same blocked boundary behind.
    keys = ['8B', '9B', '10B', '11B', '12B', '1B', '2B', '3B', '4B', '5B', '6B', '7B']
    rows = tracks(keys + ['10B']*4)
    order = [row['id'] for row in rows]
    result = repair_mode_order(rows, order, first_id='0')
    assert result['unaccepted_before'] == 1 and result['unaccepted_after'] == 0
    assert result['stages'][0]['accepted_moves'] == []
    long = result['stages'][1]
    assert long['accepted_moves'][0]['length_blocks'] == 4
    assert result['order'][3:7] == ['12', '13', '14', '15']
    assert result['incumbent_track_edges_changed'] == 2
    assert result['evaluated_candidates'] == sum(stage['evaluated_candidates'] for stage in result['stages'])
    assert result['scanned_candidates'] == sum(stage['scanned_candidates'] for stage in result['stages'])
    assert result['evaluated_candidates'] <= result['candidate_budget']
    assert result['scanned_candidates'] <= result['scan_budget']


def test_separate_scan_budget_bounds_even_pruned_moves():
    result = repair_mode_order(tracks(['8B', '9B', '10B', '7B', '8B']),
                               ['0', '1', '2', '3', '4'], first_id='0', max_scans=3)
    assert result['scanned_candidates'] == 3
    assert result['evaluated_candidates'] <= 3
    assert result['pruned_candidates'] == result['scanned_candidates'] - result['evaluated_candidates']
    assert result['budget_exhausted']
    assert len(result['stages']) == 1


def test_mood_directions_are_not_treated_as_symmetric_edges():
    rows = tracks(['8B', '9A', '7A', '8A'])
    first = repair_mode_order(rows, ['0', '2', '1', '3'], mode='mood', first_id='0')
    assert first['unaccepted_before'] >= 1
    assert first['unaccepted_after'] < first['unaccepted_before']
    assert first['order'][0] == '0'


def test_random_directed_deltas_never_break_locks_or_lose_vertices():
    generator = random.Random(1771)
    for _ in range(100):
        n = generator.randrange(3, 10)
        blocked = [[generator.random() < .5 for _ in range(n)] for _ in range(n)]
        costs = [[generator.random() * 5 for _ in range(n)] for _ in range(n)]
        order = list(range(n))
        generator.shuffle(order)
        chosen, audit = _repair_route(order, blocked, costs, first=order[0], last=order[-1],
                                      max_candidates=5000)
        assert chosen[0] == order[0] and chosen[-1] == order[-1]
        assert len(chosen) == n and set(chosen) == set(order)
        assert sum(blocked[a][b] for a, b in zip(chosen, chosen[1:])) <= sum(
            blocked[a][b] for a, b in zip(order, order[1:]))
        assert audit['evaluated_candidates'] <= 5000
        values = [sum(blocked[a][b] for a, b in zip(order, order[1:]))]
        values.extend(move['unaccepted_boundaries_after'] for move in audit['accepted_moves'])
        assert all(a > b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize('kwargs', [dict(first_id='x'), dict(last_id='x'),
    dict(locked_chains=[['0', '0']]), dict(locked_chains=[['0', '2']]),
    dict(locked_chains=[['0', '1'], ['1', '2']]), dict(passes=True),
    dict(passes=-1), dict(max_candidates=2_000_001), dict(max_scans=True),
    dict(max_scans=-1), dict(max_scans=20_000_001), dict(boundary_costs=[[1]]),
    dict(boundary_costs=[[float('nan')]*3]*3)])
def test_invalid_locks_order_budgets_and_costs_rejected(kwargs):
    with pytest.raises(ValueError):
        repair_mode_order(tracks(['8B', '9B', '10B']), ['0', '1', '2'], **kwargs)


def test_missing_or_duplicate_input_rejected_and_single_track_supported():
    with pytest.raises(ValueError, match='every input'):
        repair_mode_order(tracks(['8B', '9B']), ['0'])
    with pytest.raises(ValueError, match='Duplicate'):
        repair_mode_order([dict(id='x', bpm=120)]*2, ['x'])
    result = repair_mode_order(tracks(['8B']), ['0'], first_id='0', last_id='0')
    assert result['order'] == ['0'] and result['unaccepted_after'] == 0
