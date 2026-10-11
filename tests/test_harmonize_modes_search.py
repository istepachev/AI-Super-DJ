import copy
import itertools
import math

import numpy as np
import pytest

from superdj.harmonize_modes import (FEATURE_VERSION, MODEL_VERSION, RELATIONS, UTILITY_WARNING,
    _pair, feature_names, rank_mode_candidates)
from superdj.harmonize_modes_search import DynamicModePool, search_mode_route
from superdj.harmonize_modes_sequence import sequence_mode_tracks


def profile(family="pool", sharing="shared"):
    names = feature_names(family, sharing)
    rng = np.random.default_rng(123)
    return dict(schema_version=1, model_version=MODEL_VERSION, feature_version=FEATURE_VERSION,
        experimental=True, identifier_features=False, default_changed=False,
        utility_warning=UTILITY_WARNING, feature_family=family, sharing=sharing,
        feature_names=names, coefficients={mode:list(rng.normal(size=len(names))) for mode in
            (('fuzzy', 'mood') if sharing == 'separate' else ('all',))},
        l2_penalty=.1, min_candidates=5, training_provenance=[dict(cohort_group=g,
            mode='fuzzy', metadata_sha256='a'*64) for g in ('a', 'b')])


def tracks(keys=("8B", "9B", "8A", "2B", None)):
    return [dict(id=str(i), bpm=110+7*i, key_camelot=key) for i,key in enumerate(keys)]


@pytest.mark.parametrize('family', ('relation', 'circle', 'tonal', 'pool', 'tempo_pool',
    'tempo_rank_density', 'tempo_rank_graph'))
@pytest.mark.parametrize('sharing', ('shared', 'interactions', 'separate'))
@pytest.mark.parametrize('mode', ('fuzzy', 'mood'))
def test_reused_features_match_public_ranker_for_every_actual_small_pool(family, sharing, mode):
    rows, p = tracks(), profile(family, sharing)
    engine = DynamicModePool(rows, p, mode=mode)
    for current in range(len(rows)):
        for mask in range(1, 1 << len(rows)):
            if mask >> current & 1:
                continue
            indices, utilities, allowed = engine.scores(current, mask)
            public = rank_mode_candidates(rows[current], [rows[i] for i in indices], p, mode=mode)
            expected = {row['id']:row for row in public['candidates']}
            for index, utility, accepted in zip(indices, utilities, allowed):
                assert utility == pytest.approx(expected[rows[index]['id']]['utility'], abs=1e-11)
                assert accepted == expected[rows[index]['id']]['eligible']


def test_tempo_pool_reuses_relation_specific_tempo_terms_and_actual_context():
    rows = [dict(id=str(i), bpm=bpm, key_camelot=key) for i, (bpm, key) in
        enumerate(((120, '8B'), (130, '9B'), (140, '9B'), (110, '9B'), (130, '8A')))]
    p = profile('tempo_pool')
    p['coefficients']['all'] = [0.] * len(p['feature_names'])
    pair_count = len(_pair(rows[0], rows[1], 'fuzzy', 'pool'))
    # One pair's tempo effect depends on its relation, including tempo sign.
    relation = RELATIONS.index('perfect_boost')
    p['coefficients']['all'][pair_count + 2 * relation] = 2.
    p['coefficients']['all'][pair_count + 2 * relation + 1] = 3.
    p['coefficients']['all'][p['feature_names'].index('log_same_key_count')] = 4.
    engine = DynamicModePool(rows, p)
    indices, utilities, _ = engine.scores(0, 0b11110)
    scores = dict(zip(indices, utilities))
    assert scores[1] == pytest.approx(5 + 4 * math.log(3))
    assert scores[2] == pytest.approx(10 + 4 * math.log(3))
    assert scores[3] == pytest.approx(-1 + 4 * math.log(3))
    assert scores[4] == pytest.approx(0.)
    reduced, values, _ = engine.scores(0, 0b10010)
    assert dict(zip(reduced, values))[1] == pytest.approx(5.)
    assert engine.context_weights is not None and len(engine.context_weights) == 12


@pytest.mark.parametrize('family', ('tempo_rank_density', 'tempo_rank_graph'))
def test_cached_rank_pool_recomputes_current_allowed_ranks_and_removed_peers(family):
    rows = [dict(id=str(i), bpm=bpm, key_camelot=key) for i, (bpm, key) in enumerate((
        (120, '8B'), (128, '10B'), (122, '9B'), (120, '8A'),
        (124, '10A'), (110, '2B'), (120, None)))]
    p = profile(family)
    p['coefficients']['all'] = [0.] * len(p['feature_names'])
    p['coefficients']['all'][p['feature_names'].index('allowed_bpm_signed_percentile')] = 1.
    engine = DynamicModePool(rows, p)
    mask = 0b1111100  # Both possible current tracks are outside this same pool.
    idx, first, _ = engine.scores(0, mask)
    _, second, _ = engine.scores(1, mask)
    assert dict(zip(idx, first))[2] == pytest.approx(1.)
    assert dict(zip(idx, second))[2] == pytest.approx(-1.)
    assert mask in engine.rank_cache
    reduced, values, _ = engine.scores(0, mask ^ (1 << 3))
    assert dict(zip(reduced, values))[2] == pytest.approx(0.)
    assert engine.rank_weights.shape == (19,)
    assert (engine.graph_weights is not None) == (family == 'tempo_rank_graph')


@pytest.mark.parametrize('family', ('tempo_rank_density', 'tempo_rank_graph'))
def test_dense_rank_features_match_public_with_aliases_noise_boundaries_and_no_allowed_heads(family):
    rows = [dict(id=str(i), bpm=bpm, key_camelot=key) for i, (bpm, key) in enumerate((
        (120, '8B'), (122, '9B'), (122, '9B'), (122.005, '9B'),
        (124, '9A'), (127, '10B'), (115, '2B'), (120, None)))]
    p = profile(family, 'interactions')
    for mode in ('mood', 'fuzzy'):
        engine = DynamicModePool(rows, p, mode=mode, cache_limit=2)
        for current, mask in ((0, 0b11111110), (0, 0b11101110), (6, 0b10111111),
                              (7, 0b00111111), (0, 0b10000010)):
            idx, utility, allowed = engine.scores(current, mask)
            expected = {r['id']:r for r in rank_mode_candidates(rows[current],
                [rows[i] for i in idx], p, mode=mode)['candidates']}
            for i, value, eligible in zip(idx, utility, allowed):
                assert value == pytest.approx(expected[rows[i]['id']]['utility'], abs=1e-11)
                assert eligible == expected[rows[i]['id']]['eligible']
            if 1 in idx and 2 in idx:
                assert utility[list(idx).index(1)] == utility[list(idx).index(2)]
            assert len(engine.cache) <= 2 and len(engine.rank_cache) <= 2


def test_finite_graph_coefficients_cannot_overflow_actual_continuation_features():
    rows = [dict(id=str(i), bpm=bpm, key_camelot='8B') for i, bpm in enumerate((100, 120, 160))]
    p = profile('tempo_rank_graph')
    p['coefficients']['all'] = [0.] * len(p['feature_names'])
    p['coefficients']['all'][p['feature_names'].index('successor_mean_abs_bpm_jump')] = 1e308
    engine = DynamicModePool(rows, p)
    with pytest.raises(ValueError, match='tempo-rank utility is non-finite'):
        engine.scores(0, 0b110)


@pytest.mark.parametrize('support', ('all_candidates', 'allowed_candidates'))
def test_training_support_is_declared_without_changing_constrained_policy_or_diagnostic(support):
    rows, p = tracks(('8B', '9B', '2B', None)), profile()
    legacy = DynamicModePool(rows, p)
    p['training_support'] = support
    declared = DynamicModePool(rows, p)
    assert declared.normalized_choices(0, 0b1110, [1, 2, 3]) == legacy.normalized_choices(0, 0b1110, [1, 2, 3])
    result = declared.rescore_order(['0', '1', '2', '3'])
    assert result['normalized_musical_loss'] == legacy.rescore_order(['0', '1', '2', '3'])['normalized_musical_loss']
    assert result['training_support'] == support
    assert 'all actual unplayed candidates' in result['normalization']
    assert 'regardless of fitting support' in result['normalization']
    assert 'same support as fitting' not in result['normalization']
    assert legacy.rescore_order(['0', '1', '2', '3'])['training_support'] == 'all_candidates'


def test_dynamic_pool_changes_after_consumption_and_matches_new_normalization():
    rows, p = tracks(('8B', '9B', '10B', '9A')), profile()
    p['coefficients']['all'] = [0.] * len(p['feature_names'])
    p['coefficients']['all'][p['feature_names'].index('log_same_key_count')] = 3
    rows[3]['key_camelot'] = '9B'
    engine = DynamicModePool(rows, p)
    full, reduced = 0b1110, 0b0110
    all_idx, all_u, _ = engine.scores(0, full)
    reduced_idx, reduced_u, _ = engine.scores(0, reduced)
    assert all_u[list(all_idx).index(1)] > reduced_u[list(reduced_idx).index(1)]
    choices = engine.normalized_choices(0, reduced, [1, 2])
    public = rank_mode_candidates(rows[0], rows[1:3], p)
    allowed = [r for r in public['candidates'] if r['eligible']]
    values = {r['id']:r['utility'] for r in allowed}
    z = math.log(sum(math.exp(x) for x in values.values()))
    for index, log_weight, fallback in choices:
        assert not fallback
        assert log_weight == pytest.approx(values[rows[index]['id']]-z)


def test_complete_beam_matches_exhaustive_dynamic_path_objective_on_small_allowed_graph():
    rows, p = tracks(('8B',)*5), profile()
    engine = DynamicModePool(rows, p)
    brute = []
    for tail in itertools.permutations(range(1, len(rows))):
        current, mask, loss = 0, 0b11110, 0.
        for chosen in tail:
            row = next(r for r in engine.normalized_choices(current, mask,
                [i for i in range(len(rows)) if mask >> i & 1]) if r[0] == chosen)
            loss -= row[1]
            mask ^= 1 << chosen
            current = chosen
        brute.append(loss)
    result = search_mode_route(rows, p, first_id='0', beam_width=128, branch_limit=7)
    assert -result['normalized_prefix_log_weight'] == pytest.approx(min(brute), abs=1e-8)
    assert result['full_route_reached_by_beam'] and not result['budget_exhausted']
    assert result['evaluated_expansions'] > len(rows)-1


@pytest.mark.parametrize('mode', ('fuzzy', 'mood'))
def test_both_modes_preserve_locked_blocks_endpoints_and_unknown_sources(mode):
    rows, p = tracks(('8B', '9B', '2B', None, '3B', '4B')), profile()
    snap_rows, snap_p = copy.deepcopy(rows), copy.deepcopy(p)
    result = search_mode_route(rows, p, mode=mode, first_id='0', last_id='5',
        locked_chains=[['1', '2'], ['3', '4']], seed=42)
    order = result['order']
    assert order[0] == '0' and order[-1] == '5'
    assert set(order) == {r['id'] for r in rows} and len(order) == len(set(order))
    for chain in (['1', '2'], ['3', '4']):
        assert order[order.index(chain[0]):order.index(chain[0])+2] == chain
    assert result['preserved_locked_internal_forbidden'] == 2
    assert rows == snap_rows and p == snap_p


def test_title_and_identity_renaming_do_not_change_seeded_music_or_tie_positions():
    rows, p = tracks(('8B',)*8), profile('relation')
    p['coefficients']['all'] = [0.] * len(p['feature_names'])
    first = search_mode_route(rows, p, first_id='0', seed=42)
    renamed = [dict(r, id='new'+r['id'], title='irrelevant') for r in rows]
    second = search_mode_route(renamed, p, first_id='new0', seed=42)
    assert second['order'] == ['new'+i for i in first['order']]
    assert second['normalized_prefix_log_weight'] == first['normalized_prefix_log_weight']
    assert search_mode_route(rows, p, first_id='0', seed=42)['order'] == first['order']


@pytest.mark.parametrize('budget', (0, 1, 8))
def test_finite_budget_exhaustion_keeps_every_track_and_finishes_declared_suffix(budget):
    rows = tracks(('8B',)*9)
    result = search_mode_route(rows, profile(), first_id='0', last_id='8',
        max_expansions=budget, locked_chains=[['3', '4']])
    assert result['evaluated_expansions'] <= budget
    assert result['budget_exhausted'] and result['greedy_completion_blocks'] > 0
    assert len(result['order']) == len(set(result['order'])) == len(rows)
    assert result['order'][0] == '0' and result['order'][-1] == '8'
    assert result['order'][result['order'].index('3')+1] == '4'


def test_one_complete_locked_block_and_all_unknown_keys_are_retained_and_disclosed():
    rows = tracks((None,)*4)
    result = search_mode_route(rows, profile(), first_id='0', last_id='3',
        locked_chains=[['0', '1', '2', '3']])
    assert result['order'] == ['0', '1', '2', '3']
    assert result['evaluated_expansions'] == result['forbidden_boundary_count'] == 0
    assert result['preserved_locked_internal_forbidden'] == 3
    assert result['final_before_repair']['harmonic_violations'] == 3
    assert 'not calibrated confidence' in result['prefix_loss_scope']


def test_search_rejects_duplicate_identity_or_historical_source_path():
    rows = tracks(('8B', '9B'))
    with pytest.raises(ValueError, match='Duplicate'):
        search_mode_route([rows[0], rows[0]], profile(), first_id='0')
    duplicate_path = [dict(row, path='E:/same-source.flac') for row in rows]
    with pytest.raises(ValueError, match='Duplicate'):
        search_mode_route(duplicate_path, profile(), first_id='0')


def test_source_path_identity_without_explicit_id_is_supported():
    rows = [dict(path='E:/a.flac', bpm=120, key_camelot='8B'),
            dict(path='E:/b.flac', bpm=122, key_camelot='9B')]
    result = search_mode_route(rows, profile(), first_id='E:/a.flac')
    assert result['order'] == ['E:/a.flac', 'E:/b.flac']


@pytest.mark.parametrize('kwargs', (dict(beam_width=0), dict(branch_limit=0),
    dict(max_expansions=-1), dict(max_expansions=True), dict(first_id='missing'),
    dict(last_id='missing'), dict(locked_chains=[['1', '1']]),
    dict(locked_chains=[['1', '2']], last_id='1')))
def test_invalid_budgets_or_endpoint_locks_are_rejected(kwargs):
    arguments = dict(first_id='0')
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        search_mode_route(tracks(), profile(), **arguments)


def test_default_sequence_keeps_original_local_choice_evidence_and_rescores_final_route():
    rows, p = tracks(('8B', '9B', '10B', '7B', '8B')), profile()
    result = sequence_mode_tracks(rows, p, first_id='0')
    assert result['version'] == 'superdj-mode-sequence-4'
    assert result['route_search']['version'] == 'superdj-mode-route-search-1'
    assert len(result['search_candidates']) == 2
    chosen = next(r for r in result['search_candidates'] if r['selected'])
    assert chosen['order'] == result['order']
    local = next(r for r in result['search_candidates'] if r['method'].startswith('local'))
    assert chosen['repaired_harmonic_violations'] <= local['repaired_harmonic_violations']
    for position, row in enumerate(result['choices']):
        assert row['current'] == result['incumbent_order'][position]
        assert row['chosen'] == result['incumbent_order'][position+1]
        assert row['candidate_count'] == len(rows)-position-1
    score = DynamicModePool(rows, p).rescore_order(result['order'])
    assert chosen['final_route_score']['normalized_musical_loss'] == pytest.approx(score['normalized_musical_loss'])
    assert 'searched or repaired' in result['choices_scope']


def test_hidden_local_mode_keeps_before_policy_and_skips_search(monkeypatch):
    import superdj.harmonize_modes_sequence as module
    monkeypatch.setattr(module, 'search_mode_route', lambda *a, **k: pytest.fail('local audit must skip beam'))
    result = sequence_mode_tracks(tracks(('8B', '9B', '10B')), profile(), first_id='0', route_search='local')
    assert not result['route_search']['enabled']
    assert result['order'] == result['local_repaired_order']
    assert not result['search_candidates']


def test_finite_coefficients_that_overflow_pair_features_are_rejected():
    p = profile('relation')
    p['coefficients']['all'] = [0.] * len(p['feature_names'])
    p['coefficients']['all'][p['feature_names'].index('abs_bpm_delta/10')] = 1e308
    rows = [dict(id='a', bpm=1, key_camelot='8B'), dict(id='b', bpm=1000, key_camelot='8B')]
    with pytest.raises(ValueError, match='non-finite'):
        DynamicModePool(rows, p)


def test_finite_interaction_coefficients_cannot_overflow_combined_mode_weights():
    p = profile('relation', 'interactions')
    p['coefficients']['all'] = [1e308] * len(p['feature_names'])
    with pytest.raises(ValueError, match='Combined mode coefficients'):
        DynamicModePool(tracks(), p, mode='mood')


def test_large_common_utility_does_not_erase_conditional_normalization():
    p = profile('relation')
    p['coefficients']['all'] = [0.] * len(p['feature_names'])
    p['coefficients']['all'][p['feature_names'].index('relation:same_key')] = 1e100
    rows = tracks(('8B',)*4)
    engine = DynamicModePool(rows, p)
    choices = engine.normalized_choices(0, 0b1110, [1, 2, 3])
    assert all(row[1] == pytest.approx(-math.log(3)) for row in choices)
    score = engine.rescore_order(['0', '1', '2', '3'])
    assert score['normalized_musical_loss'] == pytest.approx(math.log(3)+math.log(2))


def test_mask_scoring_rejects_current_consumed_membership_or_noninput_bits():
    engine = DynamicModePool(tracks(), profile())
    for current, mask in ((0, 1), (0, 1 << len(engine.records)), (-1, 2), (True, 2)):
        with pytest.raises(ValueError):
            engine.scores(current, mask)


def test_harmonic_safety_retains_released_local_route_when_new_candidate_is_worse(monkeypatch):
    import superdj.harmonize_modes_sequence as module
    # Repair cannot fix the test candidate without moving locked endpoints;
    # zero search budgets retain the candidate so the independent safety branch
    # itself, rather than a successful repair, must protect the old route.
    actual_repair = module.repair_mode_order
    def no_repair(records, order, **kwargs):
        return actual_repair(records, order, **kwargs, passes=0, max_candidates=0, max_scans=0)
    monkeypatch.setattr(module, 'repair_mode_order', no_repair)
    p = profile('relation')
    p['coefficients']['all'] = [0.] * len(p['feature_names'])
    rows = tracks(('8B', '9B', '10B', '8B', '7B'))
    original_search = module.search_mode_route
    def worse_candidate(*a, **k):
        result = original_search(*a, **k)
        result['order'] = ['0', '1', '2', '3', '4']
        return result
    monkeypatch.setattr(module, 'search_mode_route', worse_candidate)
    result = sequence_mode_tracks(rows, p, first_id='0')
    assert result['route_search']['selected_method'] == 'local_suffix_guard_and_repair'
    assert result['order'] == result['local_repaired_order']
    assert result['route_search']['beam_repaired_harmonic_violations'] > result['route_search']['local_safety_harmonic_violations']
