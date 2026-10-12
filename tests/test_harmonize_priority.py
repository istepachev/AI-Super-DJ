"""Opt-in controls, legacy compatibility and actual-pool routing contracts."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from superdj.cli import _routing_priority
from superdj.harmonize_modes import (FEATURE_VERSION, MODEL_VERSION, UTILITY_WARNING,
    feature_names, fit_mode_ranker, fit_priority_mode_ranker_fixed, rank_mode_candidates,
    remaining_features, validate_mode_ranker_profile)
from superdj.harmonize_modes_search import DynamicModePool
from superdj.harmonize_modes_sequence import sequence_mode_tracks
from superdj.harmonize_priority import FEATURE_VERSION as PRIORITY_FEATURE_VERSION
from superdj.harmonize_priority import MODEL_VERSION as PRIORITY_MODEL_VERSION


def records():
    return [dict(id="a", bpm=128, key_camelot="8A"),
            dict(id="b", bpm=130, key_camelot="8A"),
            dict(id="c", bpm=120, key_camelot="9A"),
            dict(id="d", bpm=128, key_camelot="8B"),
            dict(id="e", bpm=160, key_camelot="2A")]


def profile(*, aware=True, sharing="shared"):
    family = "tempo_rank_density"
    parameters = dict(feature_family=family, sharing=sharing, l2_penalty=.1,
                      min_candidates=2, training_support="allowed_candidates")
    names = feature_names(family, sharing, priority_aware=aware)
    coefficients = {key:[0.]*len(names) for key in (("fuzzy", "mood") if sharing == "separate" else ("all",))}
    provenance = [dict(name="teacher", cohort_group="cohort_one", mode="mood", metadata_sha256="a"*64)]
    value = dict(schema_version=3 if aware else 1,
        model_version=PRIORITY_MODEL_VERSION if aware else MODEL_VERSION,
        feature_version=PRIORITY_FEATURE_VERSION if aware else FEATURE_VERSION,
        experimental=True, identifier_features=False, default_changed=False,
        utility_warning=UTILITY_WARNING, feature_names=names, coefficients=coefficients,
        training_provenance=provenance, **parameters)
    if aware:
        value.update(priority_aware=True, training_only=True, independent_validation=False,
            selection=dict(method="fixed-parameters", parameters=parameters, grid_search_performed=False))
        provenance[0]["routing_controls"] = dict(priority="unknown", bpm_tolerance=None)
    else:
        provenance.append(dict(name="other", cohort_group="cohort_two", mode="fuzzy", metadata_sha256="b"*64))
    return value


def test_control_features_are_opt_in_and_names_stable():
    rows = records()
    old = remaining_features(rows[0], rows[1:], family="tempo_rank_density")
    assert old.shape[1] == 116
    assert len(feature_names("tempo_rank_density", "interactions")) == 232
    controlled = remaining_features(rows[0], rows[1:], family="tempo_rank_density", priority_aware=True)
    assert controlled.shape[1] == 159
    assert len(feature_names("tempo_rank_density", "interactions", priority_aware=True)) == 318
    np.testing.assert_array_equal(controlled[:, :116], old)
    np.testing.assert_array_equal(controlled[:, 116:], 0)
    with pytest.raises(ValueError, match="sharing"):
        remaining_features(rows[0], rows[1:], priority_aware=True, sharing="invalid")


def test_control_changes_preferences_without_changing_gate():
    rows, model = records(), profile()
    names = model["feature_names"]
    model["coefficients"]["all"][names.index("priority:key*relation:same_key")] = 4
    model["coefficients"]["all"][names.index("priority:bpm*abs_bpm_delta/10")] = -2
    key = rank_mode_candidates(rows[0], rows[1:], model, mode="mood", priority="key")
    bpm = rank_mode_candidates(rows[0], rows[1:], model, mode="mood", priority="bpm", bpm_tolerance=4)
    assert key["shortlist_ids"][0] == "b"
    assert bpm["shortlist_ids"][0] == "d"
    assert {row["id"]:row["eligible"] for row in key["candidates"]} == {
        row["id"]:row["eligible"] for row in bpm["candidates"]}
    assert next(row for row in bpm["candidates"] if row["id"] == "e")["eligible"] is False


def test_nullable_tolerance_is_soft_and_explicit():
    rows, model = records(), profile()
    model["coefficients"]["all"][model["feature_names"].index("explicit_tolerance:within")] = 5
    outputs = [rank_mode_candidates(rows[0], rows[1:], model, mode="mood", priority="bpm", bpm_tolerance=tolerance)
               for tolerance in (None, 4, 10)]
    c_utilities = [next(row["utility"] for row in result["candidates"] if row["id"] == "c") for result in outputs]
    assert c_utilities == [0, 0, 5]
    assert len({tuple(sorted((row["id"], row["eligible"]) for row in result["candidates"])) for result in outputs}) == 1


def test_legacy_controls_leave_utility_and_default_result_unchanged():
    rows, model = records(), profile(aware=False)
    model["coefficients"]["all"][11] = -.3
    baseline = rank_mode_candidates(rows[0], rows[1:], model, mode="mood")
    explicit = rank_mode_candidates(rows[0], rows[1:], model, mode="mood", priority="key", bpm_tolerance=4)
    assert "routing_controls" not in baseline
    assert baseline["candidates"] == explicit["candidates"]
    assert baseline["shortlist_ids"] == explicit["shortlist_ids"]
    assert explicit["priority_aware"] is False


@pytest.mark.parametrize("sharing", ["shared", "interactions", "separate"])
def test_dynamic_actual_future_pools_match_public_controls(sharing):
    rows, model = records(), profile(sharing=sharing)
    for values in model["coefficients"].values():
        values[:] = np.linspace(-.13, .21, len(values)).tolist()
    before = copy.deepcopy((rows, model))
    engine = DynamicModePool(rows, model, mode="mood", priority="key", bpm_tolerance=4)
    old_b = None
    for mask in (0b11110, 0b11010):
        indices, utility, eligible = engine.scores(0, mask)
        pool = [rows[int(index)] for index in indices]
        expected = remaining_features(rows[0], pool, mode="mood", family="tempo_rank_density",
            sharing=sharing, priority_aware=True, priority="key", bpm_tolerance=4)
        coefficients = np.asarray(model["coefficients"]["mood" if sharing == "separate" else "all"])
        np.testing.assert_allclose(utility, expected @ coefficients, atol=1e-11, rtol=1e-12)
        if old_b is None:
            old_b = utility[0]
        else:
            assert abs(old_b-utility[0]) > 1e-6
    assert (rows, model) == before


def test_complete_route_retains_controls_locks_and_all_sources_at_budget_stop():
    rows, model = records(), profile()
    plan = sequence_mode_tracks(rows, model, mode="mood", first_id="a", last_id="e",
        locked_chains=[["b", "c"]], routing_priority="bpm", bpm_tolerance=4,
        search_max_expansions=0, seed=2)
    assert set(plan["order"]) == {row["id"] for row in rows}
    assert len(plan["order"]) == len(set(plan["order"])) == 5
    assert plan["order"][0] == "a" and plan["order"][-1] == "e"
    assert plan["order"][plan["order"].index("b")+1] == "c"
    assert plan["routing_controls"] == dict(priority="bpm", bpm_tolerance=4)
    assert plan["route_search"]["budget_exhausted"]
    assert plan["objective_details"]["model_version"] == PRIORITY_MODEL_VERSION


def test_fixed_single_cohort_training_has_explicit_scope_and_no_filename_guess():
    rows = records()[:4]
    example = dict(name="key_in_filename_is_not_a_control", mode="mood", records=rows,
                   order=[row["id"] for row in rows])
    with pytest.raises(ValueError, match="explicit priority"):
        fit_priority_mode_ranker_fixed([example], feature_family="relation", sharing="shared", min_candidates=2)
    example.update(priority="unknown", training_eligible=False, priority_aware_training_eligible=True)
    fitted = fit_priority_mode_ranker_fixed([example], feature_family="relation", sharing="shared", min_candidates=2)
    assert fitted["training_only"] and fitted["independent_validation"] is False
    assert fitted["selection"]["method"] == "fixed-parameters"
    assert fitted["training_provenance"][0]["routing_controls"] == dict(priority="unknown", bpm_tolerance=None)
    assert validate_mode_ranker_profile(fitted)["schema_version"] == 3
    with pytest.raises(ValueError, match="two independent"):
        fit_mode_ranker([example], min_candidates=2)
    example["priority_aware_training_eligible"] = False
    with pytest.raises(ValueError, match="not eligible"):
        fit_priority_mode_ranker_fixed([example], min_candidates=2)


@pytest.mark.parametrize("numeric,explicit,expected", [(0,None,"bpm"),(1,None,"key"),(.5,None,"unknown"),
    (None,None,"unknown"),(0,"key","key"),(.5,"bpm","bpm")])
def test_cli_endpoint_mapping_and_categorical_override(numeric, explicit, expected):
    assert _routing_priority(SimpleNamespace(priority=numeric, routing_priority=explicit, bpm_tolerance=None)) == expected


@pytest.mark.parametrize("priority,tolerance", [("made_up",None),("bpm",0),("bpm",float("nan")),("bpm",True)])
def test_invalid_controls_rejected(priority, tolerance):
    rows = records()
    with pytest.raises(ValueError):
        rank_mode_candidates(rows[0], rows[1:], profile(), priority=priority, bpm_tolerance=tolerance)


def test_single_cohort_permission_does_not_accept_claimed_independent_validation():
    model = profile()
    model["independent_validation"] = True
    with pytest.raises(ValueError, match="training-only"):
        validate_mode_ranker_profile(model)
