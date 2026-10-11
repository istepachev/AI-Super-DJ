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


def test_shortlist_caps_at_seven_and_discloses_a_tied_boundary(profile):
    tied = [track(f'candidate-{i}', bpm=120) for i in range(12)]
    ranked = rank_mode_candidates(track('current', bpm=120), tied, profile)
    assert len(ranked['shortlist_ids']) == 7
    assert ranked['shortlist_boundary_tied']
    small = rank_mode_candidates(track('current', bpm=120), tied[:4], profile)
    assert len(small['shortlist_ids']) == 4
    assert not small['shortlist_boundary_tied']


def test_unknown_training_keys_allowed_without_inventing_key():
    examples = synthetic()
    for example_ in examples:
        example_["records"][0]["key_camelot"] = None
    result = fit_mode_ranker(examples, sharing_options=("shared",), l2_candidates=(.1,), min_candidates=2)
    assert validate_mode_ranker_profile(result)
