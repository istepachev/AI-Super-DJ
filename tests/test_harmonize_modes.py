from copy import deepcopy

import numpy as np
import pytest

from superdj.harmonize_modes import (
    MODEL_VERSION, UTILITY_WARNING, _aggregate, _allowed, _prepare, cross_validate_mode_ranker,
    feature_names, fit_mode_ranker, rank_mode_candidates, remaining_features,
    validate_mode_ranker_profile,
)
from superdj.harmonize import harmonic_relation


def track(identity, key="8B", bpm=120, **extra):
    return dict(id=identity, key_camelot=key, bpm=bpm, **extra)


def example(name, *, mode="fuzzy", records=None, group=None):
    records = records or [track(f"{name}:{i}", key=key, bpm=120+i)
        for i, key in enumerate(("8B", "9B", "8A", "7B", "10B", "9A"))]
    result = dict(name=name, mode=mode, records=records, order=[r["id"] for r in records])
    if group is not None:
        result["cohort_group"] = group
    return result


def synthetic():
    result = []
    for cohort in range(3):
        records = [track(f"cohort{cohort}:{i}", key=key, bpm=115+cohort*2+i,
                         source_hash=f"source{cohort}:{i}")
                   for i,key in enumerate(("8B", "9B", "8A", "7B", "10B", "9A"))]
        result += [example(f"{cohort}:fuzzy", records=records),
                   example(f"{cohort}:mood", mode="mood", records=records)]
    return result


@pytest.fixture(scope="module")
def profile():
    return fit_mode_ranker(synthetic(), sharing_options=("shared",),
                           l2_candidates=(.1,), min_candidates=2)


def test_features_permutation_equivariance_and_no_identity_features():
    current = track("current")
    pool = [track("a", "9B", 123), track("b", "8A", 125), track("c", "7B", 120)]
    original = remaining_features(current, pool, mode="mood", sharing="interactions")
    permutation = [2, 0, 1]
    changed = [dict(pool[i], id="renamed"+str(i), title="unrelated", path="private") for i in permutation]
    matrix = remaining_features(dict(current, id="different"), changed, mode="mood", sharing="interactions")
    np.testing.assert_allclose(matrix, original[permutation])
    assert original.shape[1] == len(feature_names("pool", "interactions"))


def test_identical_musical_candidates_remain_tied(profile):
    pool = [track("a", "9B", 125), track("b", "9B", 125)]
    np.testing.assert_array_equal(remaining_features(track("current"), pool)[0],
                                  remaining_features(track("current"), pool)[1])
    ranked = rank_mode_candidates(track("current"), pool, profile)
    assert ranked["best_candidate_ids"] == ["a", "b"]
    assert [row["tie_count"] for row in ranked["candidates"]] == [2, 2]


def test_mode_gate_cannot_be_overridden_by_utility(profile):
    current, candidate = track("source", "8B"), track("target", "7A", 120)
    assert rank_mode_candidates(current, [candidate], profile, mode="fuzzy")["eligible_count"] == 1
    mood = rank_mode_candidates(current, [candidate], profile, mode="mood")
    assert mood["eligible_count"] == 0
    assert mood["candidates"][0]["rank"] is None
    assert mood["candidates"][0]["block_reasons"]


def test_pool_connectivity_gate_matches_public_rules_for_all_keys():
    keys = [(number, letter) for letter in ("A", "B") for number in range(1,13)]
    for mode in ("fuzzy", "mood"):
        for source in keys:
            for target in keys:
                assert _allowed(source, target, mode) == harmonic_relation(
                    f"{source[0]}{source[1]}", f"{target[0]}{target[1]}", mode=mode)["accepted"]


def test_unknown_keys_explicitly_blocked(profile):
    unknown = track("unknown", key=None)
    pool = [unknown, track("known", "8B", 121)]
    assert np.isfinite(remaining_features(track("source"), pool)).all()
    ranked = rank_mode_candidates(track("source"), pool, profile)
    assert ranked["blocked_count"] == 1
    assert next(row for row in ranked["candidates"] if row["id"] == "unknown")["harmonic"]["relation"] == "unknown"
    assert rank_mode_candidates(unknown, [track("a")], profile)["eligible_count"] == 0


def test_cohort_components_merge_transitively_despite_explicit_split_labels():
    a = [track(f"a{i}", source_hash=f"a{i}") for i in range(6)]
    b = [track(f"b{i}", source_hash=("a0" if i == 0 else f"b{i}")) for i in range(6)]
    c = [track(f"c{i}", source_hash=("b1" if i == 0 else f"c{i}")) for i in range(6)]
    prepared = _prepare([example("a", records=a, group="split-a"),
                         example("b", records=b, group="split-b"),
                         example("c", records=c, group="split-c")], 2)
    assert len({row["group"] for row in prepared}) == 1
    with pytest.raises(ValueError, match="independent source cohorts"):
        fit_mode_ranker([example("a", records=a), example("b", records=b)], min_candidates=2)


def test_source_hashes_distinguish_generic_local_ids():
    records_a = [track(str(i), source_hash=f"one:{i}") for i in range(6)]
    records_b = [track(str(i), source_hash=f"two:{i}") for i in range(6)]
    prepared = _prepare([example("a", records=records_a), example("b", records=records_b)], 2)
    assert len({row["group"] for row in prepared}) == 2


def test_nested_validation_holds_out_all_order_and_mode_variants():
    result = cross_validate_mode_ranker(synthetic(), sharing_options=("shared",),
                                        l2_candidates=(.1,), min_candidates=2)
    assert result["cohort_group_count"] == 3
    assert len(result["folds"]) == 3
    assert result["raw_summary"]["steps"] == 24
    for fold in result["folds"]:
        heldout = set(fold["heldout_examples"])
        assert len(heldout) == 2
        assert not heldout.intersection(row["name"] for row in fold["profile"]["training_provenance"])
        assert fold["heldout_cohort_group"] not in {row["cohort_group"] for row in fold["profile"]["training_provenance"]}


def test_blocked_choices_do_not_bias_likelihood_denominator():
    row = dict(steps=4, best_count=1, unique_best_count=1, top3_count=2,
               guaranteed_top3_count=2, demonstrated_next_blocked=2,
               fractional_top1_credit=1., reciprocal_midrank_sum=1., nll_sum=6., unique_feature_ceiling=4)
    result = _aggregate([row])
    assert result["mean_nll"] is None
    assert result["conditional_nll_steps"] == 2
    assert result["conditional_mean_nll"] == 3.
    assert result["unique_best_rate"] == .25  # blocked choices remain in ranking denominator
    all_supported = _aggregate([dict(row, demonstrated_next_blocked=0)])
    assert all_supported["mean_nll"] == all_supported["conditional_mean_nll"] == 1.5
    no_support = _aggregate([dict(row, demonstrated_next_blocked=4, nll_sum=0.)])
    assert no_support["mean_nll"] is no_support["conditional_mean_nll"] is None


def test_top7_boundary_ties_and_small_final_pools_are_reported_separately():
    import numpy as np
    from superdj.harmonize_modes import _metrics
    current = track('current')
    pools = [[track(f'early-{i}') for i in range(10)], [track(f'late-{i}') for i in range(7)]]
    prepared = dict(name='top7', group='group', mode='fuzzy',
        queries=[(current, pool) for pool in pools],
        matrices=[np.asarray([[x] for x in [5, 10, 9, 8, 7, 6, 5, 5, 5, 4]]), np.zeros((7, 1))])
    row = _metrics([prepared], {'all': np.ones(1)}, 'shared')[0]
    assert row['top7_count'] == 2
    assert row['guaranteed_top7_count'] == 1
    assert row['top7_informative_steps'] == row['top7_small_pool_steps'] == 1
    assert row['top7_informative_count'] == 1
    assert row['guaranteed_top7_informative_count'] == 0
    assert row['guaranteed_top7_small_pool_count'] == 1
    summary = _aggregate([row])
    assert summary['top7_informative_rate'] == 1
    assert summary['guaranteed_top7_informative_rate'] == 0


def test_separate_models_require_both_modes_in_each_inner_training_split():
    with pytest.raises(ValueError, match="No model option"):
        fit_mode_ranker([example("fuzzy-only"), example("mood-only", mode="mood")],
                        sharing_options=("separate",), l2_candidates=(.1,), min_candidates=2)


@pytest.mark.parametrize("mutation", [
    lambda p:p.update(model_version="different"),
    lambda p:p.update(identifier_features=True),
    lambda p:p.update(default_changed=True),
    lambda p:p.update(utility_warning="confidence"),
    lambda p:p.update(feature_names=list(reversed(p["feature_names"]))),
    lambda p:p["coefficients"]["all"].__setitem__(0, float("nan")),
    lambda p:p.update(coefficients={"mood":p["coefficients"]["all"]}),
    lambda p:p.update(l2_penalty=0),
    lambda p:p.update(training_provenance=[]),
])
def test_rejects_tampered_profiles(profile, mutation):
    changed = deepcopy(profile)
    mutation(changed)
    with pytest.raises(ValueError):
        validate_mode_ranker_profile(changed)


def test_profile_json_roundtrip(profile):
    import json
    changed = json.loads(json.dumps(profile))
    assert validate_mode_ranker_profile(changed)["model_version"] == MODEL_VERSION
    assert changed["utility_warning"] == UTILITY_WARNING
    assert changed["identifier_features"] is False


@pytest.mark.parametrize("pool", [[track("same")], [track("a"),track("a")], [track("a",bpm=-1)]])
def test_invalid_candidate_pools_rejected(profile, pool):
    with pytest.raises(ValueError):
        rank_mode_candidates(track("same"), pool, profile)


def test_empty_pool_has_explicit_no_recommendation(profile):
    result = rank_mode_candidates(track("current"), [], profile)
    assert result["candidate_count"] == result["eligible_count"] == 0
    assert result["candidates"] == []
    assert result["no_recommendation_reason"]
    assert result['shortlist_ids'] == []


def test_shortlist_caps_at_three_and_discloses_a_tied_boundary(profile):
    tied = [track(f'candidate-{i}', bpm=120) for i in range(12)]
    ranked = rank_mode_candidates(track('current', bpm=120), tied, profile)
    assert len(ranked['shortlist_ids']) == 3
    assert ranked['shortlist_limit'] == 3
    assert ranked['shortlist_boundary_tied']
    legacy = rank_mode_candidates(track('current', bpm=120), tied, profile,
                                  shortlist_limit=7)
    assert len(legacy['shortlist_ids']) == 7
    assert legacy['shortlist_boundary_tied']
    small = rank_mode_candidates(track('current', bpm=120), tied[:3], profile)
    assert len(small['shortlist_ids']) == 3
    assert not small['shortlist_boundary_tied']


@pytest.mark.parametrize('limit', [0, 8, -1, True, 3.0, None])
def test_shortlist_rejects_invalid_limits(profile, limit):
    with pytest.raises(ValueError, match='shortlist_limit'):
        rank_mode_candidates(track('current'), [track('next')], profile,
                             shortlist_limit=limit)


def test_main_top3_excludes_trivial_final_pools(profile):
    from superdj.harmonize_modes import _matrices, _metrics
    prepared = _prepare([example('one')], 2)
    matrices = _matrices(prepared, profile['feature_family'], profile['sharing'])
    row = _metrics(matrices, profile['coefficients'], profile['sharing'])[0]
    assert row['top3_informative_steps'] == 2
    assert row['top3_small_pool_steps'] == 2
    assert row['steps'] == 4
    aggregate = _aggregate([row])
    assert aggregate['primary_shortlist_size'] == 3
    assert aggregate['top3_informative_rate'] == row['top3_informative_count'] / 2
    assert row['top3_small_pool_count'] == row['top3_small_pool_steps']


def test_literal_validation_does_not_credit_teacher_first_at_a_tied_boundary(profile):
    from superdj.harmonize_modes import _matrices, _metrics
    neutral = deepcopy(profile)
    neutral['coefficients']['all'] = [0.] * len(neutral['feature_names'])
    current = track('current')
    # The teacher target is first in the query, but outside the actual list.
    pool = [track(identity) for identity in ('z-target', 'a', 'b', 'c', 'd')]
    prepared = dict(name='tied', group='g', mode='fuzzy', queries=[(current, pool)])
    matrices = _matrices([prepared], neutral['feature_family'], neutral['sharing'])
    row = _metrics(matrices, neutral['coefficients'], neutral['sharing'], gated=True)[0]
    ranked = rank_mode_candidates(current, pool, neutral)
    assert ranked['shortlist_ids'] == ['a', 'b', 'c']
    assert row['top3_informative_count'] == 1  # optimistic rank, five tied tracks
    assert row['guaranteed_top3_informative_count'] == 0
    assert row['literal_top3_informative_count'] == row['literal_top3_count'] == 0
    summary = _aggregate([row])
    assert summary['literal_top3_metrics_available']
    assert summary['primary_top3_metric'] == 'literal_top3_informative_rate'
    assert summary['literal_top3_informative_rate'] == 0


def test_literal_validation_matches_capped_ranker_and_keeps_blocked_targets(profile):
    from superdj.harmonize_modes import _matrices, _metrics
    current = track('current')
    pools = [
        [track('target', '9B', 123), track('a', '8A', 126), track('b', '7B', 121), track('c', '8B', 120)],
        [track('blocked', None), track('a', '8B'), track('b', '9B'), track('c', '7B')],
        [track('short-target', '8B'), track('last', '8B')],
    ]
    prepared = dict(name='literal', group='g', mode='fuzzy', queries=[(current, p) for p in pools])
    matrices = _matrices([prepared], profile['feature_family'], profile['sharing'])
    row = _metrics(matrices, profile['coefficients'], profile['sharing'], gated=True)[0]
    hits = [p[0]['id'] in rank_mode_candidates(current, p, profile)['shortlist_ids'] for p in pools]
    assert row['literal_top3_count'] == sum(hits)
    assert row['literal_top3_informative_count'] == sum(hits[:2])
    assert row['literal_top3_small_pool_count'] == hits[2]
    assert row['demonstrated_next_blocked'] == 1
    assert row['top3_informative_steps'] == 2
    assert _aggregate([row])['literal_top3_informative_rate'] == sum(hits[:2]) / 2


def test_legacy_metric_rows_do_not_invent_literal_coverage():
    old = dict(steps=4, best_count=1, unique_best_count=1, top3_count=2,
        guaranteed_top3_count=2, demonstrated_next_blocked=0, fractional_top1_credit=1.,
        reciprocal_midrank_sum=1., nll_sum=2., unique_feature_ceiling=4,
        top3_informative_steps=4, top3_informative_count=2)
    summary = _aggregate([old])
    assert not summary['literal_top3_metrics_available']
    assert summary['literal_top3_rate'] is summary['literal_top3_informative_rate'] is None


def test_unknown_training_keys_allowed_without_inventing_key():
    examples = synthetic()
    for example_ in examples:
        example_["records"][0]["key_camelot"] = None
    result = fit_mode_ranker(examples, sharing_options=("shared",), l2_candidates=(.1,), min_candidates=2)
    assert validate_mode_ranker_profile(result)


def test_supported_training_skips_only_fit_queries_and_keeps_validation_denominators():
    examples = synthetic()
    for item in examples:
        item['records'][0]['key_camelot'] = None
    options = dict(feature_families=('pool',), sharing_options=('shared',),
                   l2_candidates=(.1,), training_supports=('allowed_candidates',), min_candidates=2)
    profile = fit_mode_ranker(examples, **options)
    assert profile['training_support'] == 'allowed_candidates'
    assert all(row['training_skipped_queries'] >= 1 for row in profile['training_provenance'])
    validation = cross_validate_mode_ranker(examples, **options)
    assert validation['gated_summary']['steps'] == 24
    assert validation['gated_summary']['demonstrated_next_blocked'] >= 6
    assert validation['gated_summary']['literal_top3_metrics_available']
    assert validation['gated_summary']['mean_nll'] is None


def test_supported_only_option_with_no_legal_targets_is_not_fitted():
    examples = synthetic()
    for item in examples:
        for record in item['records']:
            record['key_camelot'] = None
    with pytest.raises(ValueError, match='No model option'):
        fit_mode_ranker(examples, feature_families=('pool',), sharing_options=('shared',),
            l2_candidates=(.1,), training_supports=('allowed_candidates',), min_candidates=2)
    # The explicit all-candidate diagnostic learner still accepts unknown keys.
    profile = fit_mode_ranker(examples, feature_families=('pool',), sharing_options=('shared',),
        l2_candidates=(.1,), training_supports=('all_candidates',), min_candidates=2)
    assert profile['training_support'] == 'all_candidates'
    assert not any(p['training_skipped_queries'] for p in profile['training_provenance'])


@pytest.mark.parametrize('support', [None, '', 'invalid', True])
def test_profile_rejects_unknown_fit_support(profile, support):
    changed = deepcopy(profile)
    changed['training_support'] = support
    with pytest.raises(ValueError, match='training support'):
        validate_mode_ranker_profile(changed)


def test_legacy_profile_without_training_support_remains_loadable(profile):
    legacy = deepcopy(profile)
    legacy.pop('training_support', None)
    assert validate_mode_ranker_profile(legacy)['model_version'] == MODEL_VERSION
    assert rank_mode_candidates(track('current'), [track('next')], legacy)['shortlist_ids'] == ['next']


@pytest.mark.parametrize('supports', [(), ('invalid',), 'allowed_candidates'])
def test_fit_rejects_invalid_training_support_options(supports):
    with pytest.raises(ValueError):
        fit_mode_ranker(synthetic(), training_supports=supports)


def test_selection_uses_actual_gated_capped_lists_and_records_its_scope():
    profile = fit_mode_ranker(synthetic(), feature_families=('pool', 'tempo_pool'),
        sharing_options=('interactions',), l2_candidates=(.01, .1), min_candidates=2)
    selected = profile['selection']['selected']
    scores = selected['inner_validation']
    assert scores['primary_top3_metric'] == 'literal_top3_informative_rate'
    assert scores['literal_top3_informative_rate'] <= scores['top3_informative_rate']
    assert selected['training_support'] == profile['training_support']
    assert all(row['inner_validation']['demonstrated_next_blocked'] > 0
               for row in profile['selection']['options'])
    assert len(profile['feature_names']) == len(feature_names(profile['feature_family'], profile['sharing']))
