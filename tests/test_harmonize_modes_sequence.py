import copy
import itertools
import random

import pytest

from superdj.harmonize_modes import MODEL_VERSION, FEATURE_VERSION, UTILITY_WARNING, feature_names, rank_mode_candidates
from superdj.harmonize_modes_sequence import sequence_mode_tracks, _minimum_forbidden_tail


def profile():
    names = feature_names('relation', 'shared')
    return dict(schema_version=1, model_version=MODEL_VERSION, feature_version=FEATURE_VERSION,
        experimental=True, identifier_features=False, default_changed=False, utility_warning=UTILITY_WARNING,
        feature_family='relation', sharing='shared', feature_names=names,
        coefficients={'all': [0.] * len(names)}, l2_penalty=.1, min_candidates=5,
        training_provenance=[dict(cohort_group=g, mode='fuzzy', metadata_sha256='a'*64) for g in ('a', 'b')])


def tracks(keys):
    return [dict(id=str(i), bpm=120+i, key_camelot=key) for i, key in enumerate(keys)]


def test_model_utility_cannot_choose_forbidden_head_when_allowed_one_exists():
    p = profile()
    p['coefficients']['all'][p['feature_names'].index('relation:incompatible')] = 100
    rows = tracks(['8B', '2B', '9B'])
    result = sequence_mode_tracks(rows, p, first_id='0')
    assert result['order'][:2] == ['0', '2']
    assert not result['choices'][0]['fallback']
    assert result['choices'][1]['fallback']
    assert result['fallback_count'] == 1
    assert not result['optimal_for_declared_objective']
    assert 'explicit fallback' in result['uncertainty']['warnings'][0]


def test_locked_chains_and_endpoints_are_retained_and_bad_locks_disclosed():
    rows = tracks(['8B', '9B', '2B', '3B', '4B'])
    result = sequence_mode_tracks(rows, profile(), first_id='0', last_id='4', locked_chains=[['1', '2']])
    order = result['order']
    assert order[0] == '0' and order[-1] == '4'
    assert order[order.index('1'):order.index('1')+2] == ['1', '2']
    assert set(order) == set(r['id'] for r in rows)
    assert next(e for e in result['edges'] if e['from'] == '1')['locked_adjacency']
    assert any('preserved locked adjacency' in w for w in result['uncertainty']['warnings'])


def test_unknown_key_is_not_dropped_or_called_a_harmonic_recommendation():
    result = sequence_mode_tracks(tracks(['8B', None]), profile(), first_id='0')
    assert result['order'] == ['0', '1']
    assert result['fallback_count'] == result['harmonic_unknown'] == 1


def test_each_choice_uses_only_actual_unplayed_pool(monkeypatch):
    import superdj.harmonize_modes_sequence as module
    actual = module.rank_mode_candidates
    calls = []
    def recording(current, pool, *args, **kwargs):
        calls.append((current['id'], {r['id'] for r in pool}))
        return actual(current, pool, *args, **kwargs)
    monkeypatch.setattr(module, 'rank_mode_candidates', recording)
    rows = tracks(['8B', '8B', '9B', '9A'])
    result = sequence_mode_tracks(rows, profile(), first_id='0')
    for position, (current, pool) in enumerate(calls):
        assert current == result['incumbent_order'][position]
        assert pool == set(result['incumbent_order'][position+1:])


def test_seeded_ties_do_not_use_title_or_identity_as_musical_features():
    rows = tracks(['8B']*7)
    first = sequence_mode_tracks(rows, profile(), first_id='0', seed=42)
    renamed = [dict(r, id='new'+r['id'], title='different') for r in rows]
    second = sequence_mode_tracks(renamed, profile(), first_id='new0', seed=42)
    assert second['order'] == ['new'+i for i in first['order']]


@pytest.mark.parametrize('kwargs', [dict(first_id='x'), dict(last_id='x'),
    dict(locked_chains=[['0', '0']]), dict(locked_chains=[['1', '2']], first_id='2'),
    dict(locked_chains=[['1', '2']], last_id='1'), dict(first_id='0', last_id='0')])
def test_inconsistent_locks_are_rejected(kwargs):
    with pytest.raises(ValueError):
        sequence_mode_tracks(tracks(['8B', '9B', '10B']), profile(), **kwargs)


def test_single_source_and_duplicate_ids():
    assert sequence_mode_tracks(tracks(['8B']), profile())['order'] == ['0']
    with pytest.raises(ValueError, match='Duplicate'):
        sequence_mode_tracks([dict(id='x', bpm=120)]*2, profile())


def test_complete_route_repair_keeps_original_choice_evidence_separate(monkeypatch):
    import superdj.harmonize_modes_sequence as module
    # Force the initial policy down the far side of the wheel. A later bridge
    # can be repaired by reopening an earlier adjacency, without inventing a
    # new learned decision or rewriting the original remaining-pool evidence.
    monkeypatch.setattr(module, '_tail_guard', lambda *args: None)
    p = profile()
    p['coefficients']['all'][p['feature_names'].index('relation:perfect_boost')] = 100
    rows = tracks(['8B', '9B', '7B', '10B', '8B'])
    for row in rows:
        row['bpm'] = 130 if row['id'] == '4' else 120
    result = sequence_mode_tracks(rows, p, first_id='0', route_search='local')
    assert result['incumbent_order'][:3] == ['0', '1', '3']
    assert result['route_repair']['changed']
    assert result['order'] != result['incumbent_order']
    assert result['order'][0] == '0'
    assert set(result['order']) == {r['id'] for r in rows}
    assert result['fallback_count'] == 0
    assert result['incumbent_fallback_count'] > 0
    assert result['harmonic_matches'] == len(rows) - 1
    assert result['choices'][0]['chosen'] == '1'
    assert result['order'][1] != '1'
    assert 'incumbent_order' in result['choices_scope']
    assert not result['uncertainty']['warnings']
    assert result['incumbent_uncertainty']['warnings']


@pytest.mark.parametrize('count', range(1, 7))
def test_suffix_bound_matches_exhaustive_directed_permutations_and_fixed_last(count):
    generator = random.Random(173 + count)
    bad = [[generator.randrange(2) for _ in range(count)] for _ in range(count)]
    for last in (None, count-1):
        exact = _minimum_forbidden_tail(bad, last=last)
        for first in range(count):
            possible = [p for p in itertools.permutations(range(count))
                        if p[0] == first and (last is None or p[-1] == last)]
            expected = min((sum(bad[a][b] for a, b in zip(p, p[1:])) for p in possible), default=None)
            assert exact[first] == expected


@pytest.mark.parametrize('matrix,last', [([], None), ([[0, 1]], None),
    ([[0]*13]*13, None), ([[0, 2], [1, 0]], None), ([[float('nan')]], None),
    ([['0']], None), ([[0]], -1), ([[0]], 1), ([[0]], True)])
def test_suffix_bound_rejects_invalid_graph_or_endpoint(matrix, last):
    with pytest.raises(ValueError):
        _minimum_forbidden_tail(matrix, last=last)


def test_suffix_guard_avoids_stranding_key_chain_before_following_utility():
    rows = tracks(['8B', '9B', '10B', '8B', '7B'])
    p = profile()
    p['coefficients']['all'][p['feature_names'].index('relation:perfect_boost')] = 100
    # The unmodified ranker prefers 9B; that first choice needs a later bridge.
    assert rank_mode_candidates(rows[0], rows[1:], p)['best_candidate_ids'] == ['1']
    result = sequence_mode_tracks(rows, p, first_id='0')
    assert result['order'] == ['0', '4', '3', '1', '2']
    assert result['fallback_count'] == 0
    proof = result['choices'][0]['tail_guard']
    costs = {r['head']: r['minimum_later_forbidden'] for r in proof['candidates']}
    assert costs['1'] == 1 and costs['4'] == 0
    assert proof['chosen_minimum_total_forbidden_remaining'] == 0
    assert 'unrestricted suffix' in proof['scope']
    assert not result['optimal_for_declared_objective']


def test_guard_uses_locked_block_tail_to_next_head_not_its_first_track():
    rows = tracks(['8B', '9B', '10B', '8B', '7B'])
    p = profile()
    p['coefficients']['all'][p['feature_names'].index('relation:perfect_boost')] = 100
    result = sequence_mode_tracks(rows, p, first_id='0', locked_chains=[['1', '2']])
    assert result['order'] == ['0', '4', '3', '1', '2']
    proof = result['choices'][0]['tail_guard']
    locked = next(r for r in proof['candidates'] if r['head'] == '1')
    assert locked['tail'] == '2' and locked['minimum_later_forbidden'] == 1
    assert result['fallback_count'] == 0


def test_suffix_guard_keeps_fixed_last_only_at_end_and_counts_bad_internal_locks():
    rows = tracks(['8B', '9B', '2B', '3B', '4B'])
    result = sequence_mode_tracks(rows, profile(), first_id='0', last_id='4', locked_chains=[['1', '2']])
    assert result['order'][0] == '0' and result['order'][-1] == '4'
    proof = result['choices'][0]['tail_guard']
    last = next(r for r in proof['candidates'] if r['head'] == '4')
    assert last['minimum_later_forbidden'] is None and not last['playable_now']
    assert proof['remaining_locked_internal_forbidden'] == 1
    assert proof['fixed_last_head'] == '4'
    assert proof['chosen_minimum_total_forbidden_remaining'] == proof['chosen_minimum_later_forbidden']+1


def test_no_allowed_head_fallback_minimizes_future_bridges_before_pair_cost():
    rows = [dict(id=str(i), bpm=bpm, key_camelot=key) for i, (bpm, key) in
            enumerate([(120, '2B'), (120, '9B'), (170, '10B'), (180, '8B'), (190, '7B')])]
    result = sequence_mode_tracks(rows, profile(), first_id='0')
    first = result['choices'][0]
    assert first['fallback'] and first['chosen'] != '1'
    assert first['tail_guard']['chosen_minimum_later_forbidden'] == 0
    assert first['tail_guard']['chosen_minimum_total_forbidden_remaining'] == 1
    assert first['tail_guard']['considered_head_scope'] == 'all playable fallback heads'
    assert result['fallback_count'] == 1
    assert len(set(result['order'])) == len(rows)


def test_tail_cap_is_number_of_remaining_blocks_and_never_changes_rank_profile():
    p = profile()
    original = copy.deepcopy(p)
    rows = tracks(['8B']*14)
    ranked = rank_mode_candidates(rows[0], rows[1:], p)
    result = sequence_mode_tracks(rows, p, first_id='0')
    assert not result['choices'][0]['tail_guard']['applied']
    assert result['choices'][1]['tail_guard']['applied']
    assert result['choices'][1]['tail_guard']['remaining_block_count'] == 12
    locked = sequence_mode_tracks(rows, p, first_id='0', locked_chains=[[r['id'] for r in rows[1:]]])
    assert locked['choices'][0]['candidate_count'] == 13
    assert locked['choices'][0]['tail_guard']['remaining_block_count'] == 1
    assert locked['choices'][0]['tail_guard']['applied']
    assert p == original
    assert rank_mode_candidates(rows[0], rows[1:], p) == ranked


def test_cli_uses_mode_profile_for_ranks_and_complete_pack(tmp_path):
    import json
    from superdj.cli import main
    rows = tracks(['8B', '9B', '2B'])
    rows = [dict(r, path=f'E:/Synthetic/{r["id"]}.flac') for r in rows]
    source, model = tmp_path/'records.json', tmp_path/'profile.json'
    source.write_text(json.dumps(rows), encoding='utf-8')
    model.write_text(json.dumps(profile()), encoding='utf-8')
    rank_dir, order_dir = tmp_path/'ranks', tmp_path/'order'
    main(['harmonize-rank', '--records', str(source), '--profile', str(model),
          '--current', '0', '--work', str(rank_dir)])
    ranked = json.loads((rank_dir/'next_track_rankings.json').read_text(encoding='utf-8'))
    assert ranked['model_version'] == MODEL_VERSION
    assert ranked['best_candidate_ids'] == ['1']
    main(['harmonize', '--records', str(source), '--ranking-profile', str(model),
          '--first', '0', '--work', str(order_dir)])
    plan = json.loads((order_dir/'harmonize_plan.json').read_text(encoding='utf-8'))
    assert plan['order'] == ['0', '1', '2']
    assert plan['order_objective'] == 'mode_pool'
    assert plan['fallback_count'] == 1
    assert (order_dir/'harmonized_tracks.m3u8').is_file()
    page = (order_dir/'harmonize_report.html').read_text(encoding='utf-8')
    assert 'глобальный оптимум полного порядка не заявлен' in page
    assert 'Все треки оптимизированы вместе' not in page


def test_cli_mode_learning_holds_out_source_cohorts(tmp_path):
    import json
    from superdj.cli import main
    examples = []
    for group in range(3):
        rows = [dict(r, id=f'{group}-{r["id"]}') for r in tracks(['8B', '9B', '8A', '9A'])]
        for mode in ('fuzzy', 'mood'):
            examples.append(dict(name=f'{group}-{mode}', cohort_group=str(group), mode=mode,
                                 records=rows, order=[r['id'] for r in rows]))
    manifest = tmp_path/'examples.json'
    manifest.write_text(json.dumps(examples), encoding='utf-8')
    work = tmp_path/'learn'
    main(['harmonize-learn', '--learner', 'modes', '--examples', str(manifest),
          '--min-candidates', '2', '--work', str(work)])
    cv = json.loads((work/'mode_ranker_validation.json').read_text(encoding='utf-8'))
    assert cv['cohort_group_count'] == 3
    for fold in cv['folds']:
        trained = {r['name'] for r in fold['profile']['training_provenance']}
        assert not trained.intersection(fold['heldout_examples'])
        assert len(fold['heldout_examples']) == 2
    assert (work/'mode_ranker_profile.json').is_file()
