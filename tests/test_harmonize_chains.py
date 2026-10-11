import pytest

from superdj.harmonize_compare import chain_agreement, compare_orders, OrderMatchError


def test_directional_contiguous_chains_and_reference_recurrence():
    result = chain_agreement(list('ABCDE'), [list('ABCDE'), list('ABDEC')])
    pairs, triples, fours = result['lengths']
    assert pairs['matches_per_reference'] == [4, 2]
    assert pairs['any_reference_count'] == 4
    assert pairs['repeated_reference_window_count'] == 2
    assert pairs['repeated_reference_hit_count'] == 2
    assert pairs['reference_frequency_weighted_hits'] == 6
    assert triples['matches_per_reference'] == [3, 0]
    assert triples['track_count'] == 3 and triples['edge_count'] == 2
    assert triples['repeated_reference_window_count'] == 0
    assert fours['matches_per_reference'] == [2, 0]
    assert fours['track_count'] == 4 and fours['edge_count'] == 3


def test_reversed_or_interrupted_chains_are_not_counted():
    reversed_ = chain_agreement(list('ABCDE'), [list('EDCBA')])
    assert all(row['any_reference_count'] == 0 for row in reversed_['lengths'])
    interrupted = chain_agreement(list('ABCDE'), [list('ACEBD')], lengths=(3, 4))
    assert all(row['any_reference_count'] == 0 for row in interrupted['lengths'])


def test_short_pools_have_no_trivial_chain_success_rate():
    result = chain_agreement(['a', 'b'], [['a', 'b']])
    assert result['lengths'][0]['any_reference_rate'] == 1
    for row in result['lengths'][1:]:
        assert row['our_window_count'] == row['any_reference_count'] == 0
        assert row['any_reference_rate'] is None


@pytest.mark.parametrize('our,refs,lengths', [
    (['a', 'a'], [['a', 'b']], (3,)),
    (['a', 'b'], [['a', 'c']], (3,)),
    (['a', 'b'], [], (3,)),
    (['a', 'b'], [['a', 'b']], (True,)),
    (['a', 'b'], [['a', 'b']], (1,)),
])
def test_invalid_chain_comparisons_fail_explicitly(our, refs, lengths):
    with pytest.raises(OrderMatchError):
        chain_agreement(our, refs, lengths=lengths)


def test_standard_comparison_includes_chain_counts_only_for_complete_pools():
    records = [dict(id=value, bpm=120, key='8A') for value in 'ABCD']
    complete = compare_orders(records, list('ABCD'), list('ABDC'))
    assert complete['chain_agreement']['lengths'][0]['matches_per_reference'] == [1]
    assert complete['chain_agreement']['lengths'][1]['matches_per_reference'] == [0]
    incomplete = compare_orders(records, list('ABCD'), list('ABC'))
    assert incomplete['chain_agreement'] is None
