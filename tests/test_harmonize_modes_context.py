"""Observable unordered-pool tempo/continuation descriptors, without IDs."""
from copy import deepcopy

import numpy as np
import pytest

from superdj.harmonize_modes import (
    DEFAULT_FEATURE_FAMILIES, GRAPH_CONTEXT_NAMES, RANK_CONTEXT_NAMES,
    _musical, _rank_current_context, _rank_pool_context, feature_names,
    remaining_features,
)


def track(identity, bpm, key="8A"):
    return dict(id=identity, bpm=bpm, key_camelot=key)


@pytest.mark.parametrize("family,size", [("pool", 75), ("tempo_pool", 97),
    ("tempo_rank_density", 116), ("tempo_rank_graph", 121)])
def test_feature_columns_preserve_legacy_and_extend_named_suffix(family, size):
    current = track("current", 105)
    pool = [track("a", 100), track("b", 108, "8B"), track("c", 120, "9A")]
    matrix = remaining_features(current, pool, family=family, mode="mood")
    assert matrix.shape == (3, size)
    assert len(feature_names(family)) == size
    assert len(feature_names(family, "interactions")) == size * 2
    if family.startswith("tempo_rank"):
        np.testing.assert_array_equal(matrix[:, :97],
            remaining_features(current, pool, family="tempo_pool", mode="mood"))
        assert feature_names(family)[97:116] == list(RANK_CONTEXT_NAMES)
        if family == "tempo_rank_graph":
            assert feature_names(family)[116:] == list(GRAPH_CONTEXT_NAMES)
    assert DEFAULT_FEATURE_FAMILIES == ("pool", "tempo_pool", "tempo_rank_density", "tempo_rank_graph")


def test_signed_rank_same_key_minimum_and_fixed_density_have_direct_meaning():
    current = track("current", 105)
    pool = [track("a", 100), track("b", 100), track("c", 108, "8B"), track("d", 120, "9A")]
    values = remaining_features(current, pool, mode="mood", family="tempo_rank_density")[:, 97:]
    np.testing.assert_allclose(values[:, 0], [-2/3, -2/3, 1/3, 1])
    np.testing.assert_array_equal(values[:, 2], [0, 0, 0, 0])
    np.testing.assert_array_equal(values[:, 3], [1, 1, 1, 1])
    np.testing.assert_allclose(values[:, 4], [1/3, 1/3, 1/6, .5])
    np.testing.assert_allclose(values[:, 5], [1/3, 1/3, 0, 0])
    np.testing.assert_allclose(values[:, 6], [1/3, 1/3, 0, 0])


def test_fixed_bpm_tolerance_is_not_exact_float_comparison():
    pool = [track("a", 100), track("b", 100.005), track("c", 100.02)]
    values = remaining_features(track("current", 100), pool, family="tempo_rank_density")[:, 97:]
    np.testing.assert_allclose(values[:, 0], [-.5, -.5, 1])
    np.testing.assert_array_equal(values[:, 3], [1, 1, 0])


@pytest.mark.parametrize("mode", ["fuzzy", "mood"])
@pytest.mark.parametrize("family", ["tempo_rank_density", "tempo_rank_graph"])
def test_descriptors_ignore_ids_titles_paths_and_candidate_order(mode, family):
    current = track("current", 115)
    pool = [track("a", 100), track("b", 100), track("c", 120, "8B"), track("d", 133, "10A"), track("e", 150, None)]
    original = remaining_features(current, pool, mode=mode, family=family, sharing="interactions")
    permutation = [4, 2, 0, 3, 1]
    altered = [dict(pool[i], id="renamed"+str(p), title="unrelated", path="not-audio", source_hash="changed") for p,i in enumerate(permutation)]
    changed = remaining_features(dict(current, id="renamed-current"), altered, mode=mode, family=family, sharing="interactions")
    np.testing.assert_allclose(changed, original[permutation], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(original[0], original[1])


def test_unknown_key_has_no_same_key_rank_minimum_or_allowed_current_evidence():
    current = track("current", 110, None)
    pool = [track("unknown-a", 100, None), track("unknown-b", 120, None), track("known", 115)]
    values = remaining_features(current, pool, family="tempo_rank_graph")[:, 97:]
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[:2, 2:4], np.zeros((2, 2)))
    np.testing.assert_array_equal(values[:, 1], np.zeros(3))
    np.testing.assert_array_equal(values[:, 7], np.zeros(3))
    np.testing.assert_array_equal(values[:2, 19:], np.zeros((2, 5)))


def test_continuation_degrees_remove_candidate_before_successor_aggregates():
    pool = [track("a", 100), track("b", 110), track("c", 120)]
    values = remaining_features(track("current", 105), pool, family="tempo_rank_graph")[:, 116:]
    # Three same-key nodes have degree2. After choosing each candidate,
    # either successor has one remaining exit, normalized by n-2=1.
    np.testing.assert_array_equal(values[:, 2:], np.ones((3, 3)))
    np.testing.assert_allclose(values[:, 0], [.5, 0, -.5])
    np.testing.assert_allclose(values[:, 1], [1.5, 1, 1.5])


@pytest.mark.parametrize("include_graph", [False, True])
def test_mask_cached_plus_current_contributions_match_full_matrix(include_graph):
    pool = [track("a", 100), track("b", 101, "8B"), track("c", 120, "9A"), track("d", 140, None)]
    bpms = [_musical(r)[1] for r in pool]
    keys = [_musical(r)[2] for r in pool]
    fixed = _rank_pool_context(bpms, keys, "mood", include_graph=include_graph)
    original = deepcopy(fixed)
    for current in (track("one", 105), track("two", 155, "9B"), track("unknown", 125, None)):
        _, bpm, key = _musical(current)
        additive = _rank_current_context(bpm, key, bpms, keys, "mood", fixed)
        family = "tempo_rank_graph" if include_graph else "tempo_rank_density"
        np.testing.assert_array_equal(fixed + additive,
            remaining_features(current, pool, mode="mood", family=family)[:, 97:])
        np.testing.assert_array_equal(fixed, original)


@pytest.mark.parametrize("family", ["tempo_rank_density", "tempo_rank_graph"])
def test_single_empty_and_unknown_only_pools_are_finite(family):
    current = track("current", 100)
    assert remaining_features(current, [], family=family).shape == (0, len(feature_names(family)))
    assert np.isfinite(remaining_features(current, [track("a", 120)], family=family)).all()
    assert np.isfinite(remaining_features(current, [track("a", 120, None)], family=family)).all()
