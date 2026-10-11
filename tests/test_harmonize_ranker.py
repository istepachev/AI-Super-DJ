"""Experimental fitting keeps identity and held-out orders out of its features."""
from copy import deepcopy
import json

import pytest

from superdj.harmonize_ranker import (
    MODEL_VERSION, UTILITY_WARNING, cross_validate_ranker, feature_names,
    fit_ranker, pair_features, rank_remaining_candidates, validate_ranker_profile,
)


def track(identity, bpm=120, key="8A", **extra):
    return dict(id=identity, bpm=bpm, key=key, **extra)


def example(name, offset=0):
    records = [track(f"{name}{i}", 120+offset+2*i) for i in range(6)]
    return dict(name=name, records=records, order=[r["id"] for r in records])


def profile(family="circle"):
    return dict(schema_version=1, model_version=MODEL_VERSION, experimental=True,
                feature_family=family, feature_names=feature_names(family),
                coefficients=[0.]*len(feature_names(family)), l2_penalty=1., min_candidates=2,
                identifier_features=False, default_changed=False,
                utility_warning=UTILITY_WARNING,
                training_provenance=[dict(name="one",metadata_sha256="a"*64),
                                     dict(name="two",metadata_sha256="b"*64)])


def test_feature_values_ignore_identifier_title_path_artist_and_pool_order():
    current = track("current",120,"8A", title="first",path="private/current.mp3")
    candidate = track("candidate",125,"9B", title="second",path="private/candidate.mp3")
    expected = pair_features(current,candidate)
    assert expected == pair_features(dict(current,id="new",title="unrelated",path="elsewhere"),
                                    dict(candidate,id="renamed",title="unrelated",path="elsewhere"))
    assert len(expected)==len(feature_names("circle"))
    assert len(pair_features(current,candidate,"relation"))==len(feature_names("relation"))
    # Transposing both keys around the wheel preserves pair relationships.
    assert expected == pair_features(track("a",120,"10A"),track("b",125,"11B"))


def test_serialized_profile_keeps_warning_and_feature_order():
    fitted=fit_ranker([example("a"),example("b",3)],
                     feature_families=("relation",),l2_candidates=(1.,),min_candidates=2)
    loaded=validate_ranker_profile(json.loads(json.dumps(fitted,allow_nan=False)))
    assert loaded["feature_names"]==feature_names("relation")
    assert loaded["utility_warning"]==UTILITY_WARNING
    assert loaded["identifier_features"] is False and loaded["default_changed"] is False
    assert {p["name"] for p in loaded["training_provenance"]}=={"a","b"}
    assert all(len(p["metadata_sha256"])==64 for p in loaded["training_provenance"])
    assert loaded["convergence"]["training_steps"]==8
    ranked=rank_remaining_candidates(track("current",124),
                                    [track("near",126),track("far",150)],loaded)
    assert ranked["best_candidate_ids"]==["near"]


def test_fit_is_invariant_to_record_order_and_identity_metadata_changes():
    examples=[example("a"),example("b",4)]
    fitted=fit_ranker(examples,feature_families=("circle",),l2_candidates=(1.,),min_candidates=2)
    renamed=deepcopy(examples)
    for value in renamed:
        names={identity:"renamed-"+identity for identity in value["order"]}
        for record in value["records"]:
            record.update(id=names[record["id"]],title="changed",artist="changed",path="changed")
        value["order"]=[names[identity] for identity in value["order"]]
        value["records"].reverse()
    changed=fit_ranker(renamed,feature_families=("circle",),l2_candidates=(1.,),min_candidates=2)
    assert changed["coefficients"]==pytest.approx(fitted["coefficients"],abs=1e-12)
    assert changed["training_provenance"]==fitted["training_provenance"]


def test_candidate_order_and_equal_utilities_do_not_create_false_unique_winner():
    model=profile()
    candidates=[track("z",122),track("a",124),track("b",121)]
    first=rank_remaining_candidates(track("current"),candidates,model)
    second=rank_remaining_candidates(track("current"),list(reversed(candidates)),model)
    assert first==second
    assert first["best_candidate_ids"]==["a","b","z"]
    assert {r["rank"] for r in first["candidates"]}=={1}
    assert {r["tie_count"] for r in first["candidates"]}=={3}


def test_tolerance_ties_are_anchored_instead_of_chaining_between_groups():
    model=profile("relation")
    model["coefficients"][model["feature_names"].index("abs_bpm_delta/10")]=-1.
    result=rank_remaining_candidates(track("current",120),
                    [track("a",120),track("b",120.006),track("c",120.012)],model,
                    tie_tolerance=.001)
    assert [row["rank"] for row in result["candidates"]]==[1,1,3]
    assert [row["tie_count"] for row in result["candidates"]]==[2,2,1]


def test_harmony_gate_cannot_be_overridden_by_learned_utility_or_profile_flags():
    model=profile()
    # Learned raw utility strongly favors an incompatible key.
    model["coefficients"][model["feature_names"].index("relation:incompatible")]=1000.
    model["disable_harmony_gate"]=True
    result=rank_remaining_candidates(track("current",120,"8A"),
                                     [track("bad",120,"4A"),track("good",120,"9A")],model)
    assert result["best_candidate_ids"]==["good"]
    blocked=next(r for r in result["candidates"] if r["id"]=="bad")
    assert blocked["utility"]>result["candidates"][0]["utility"]
    assert blocked["rank"] is None and blocked["eligible"] is False
    assert blocked["block_reasons"]
    assert result["blocked_count"]==1


def test_no_harmonically_allowed_candidate_yields_no_recommendation():
    result=rank_remaining_candidates(track("current",120,"8A"),[track("bad",120,"3A")],profile())
    assert result["best_candidate_ids"]==[] and result["eligible_count"]==0
    assert result["no_recommendation_reason"]
    assert rank_remaining_candidates(track("current"),[],profile())["candidate_count"]==0


def test_mood_adventurous_acceptance_remains_explicit_in_evidence():
    model=profile()
    current,candidate=track("a",120,"8A"),track("b",120,"3A")
    fuzzy=rank_remaining_candidates(current,[candidate],model,mode="fuzzy")
    mood=rank_remaining_candidates(current,[candidate],model,mode="mood")
    assert fuzzy["eligible_count"]==0
    assert mood["eligible_count"]==1
    assert mood["candidates"][0]["harmonic"]["risk"]=="high"


def test_outer_heldout_metadata_and_order_cannot_change_its_trained_profile():
    examples=[example("a"),example("b",3),example("c",6)]
    controls=dict(feature_families=("relation","circle"),l2_candidates=(.1,1.),min_candidates=2)
    first=cross_validate_ranker(examples,**controls)
    changed=deepcopy(examples)
    changed[2]["order"].reverse()
    changed[2]["records"][0]["key"]="3B"
    second=cross_validate_ranker(changed,**controls)
    heldout1,heldout2=first["folds"][2],second["folds"][2]
    assert heldout1["heldout"]==heldout2["heldout"]=="c"
    assert heldout1["profile"]==heldout2["profile"]
    assert heldout1["heldout_metadata_sha256"]!=heldout2["heldout_metadata_sha256"]
    assert {p["name"] for p in heldout1["profile"]["training_provenance"]}=={"a","b"}
    assert "ungated" in heldout1["raw_local_audit"]["scope"]
    assert "blocked demonstrations" in heldout1["gated_recommendation"]["scope"]


def cohort_variants(group, offset=0):
    original = example(group, offset)
    fuzzy = dict(deepcopy(original), name=group+"-fuzzy", cohort_group=group, mode="fuzzy")
    mood = dict(deepcopy(original), name=group+"-mood", cohort_group=group, mode="mood")
    mood["order"].reverse()
    return [fuzzy, mood]


def test_all_variants_of_outer_cohort_are_held_out_together():
    examples = cohort_variants("a") + cohort_variants("b", 3) + cohort_variants("c", 6)
    controls = dict(feature_families=("relation",), l2_candidates=(1.,), min_candidates=2)
    result = cross_validate_ranker(examples, **controls)
    assert result["validation"] == "nested leave-one-whole-cohort-out"
    assert result["cohort_group_count"] == 3
    assert result["fallback_example_names"] == []
    for fold in result["folds"]:
        own_group = fold["heldout_cohort_group"]
        training = fold["profile"]["training_provenance"]
        assert len(training) == 4
        assert own_group not in {row["cohort_group"] for row in training}
        assert set(fold["heldout_cohort_examples"]) == {own_group+"-fuzzy", own_group+"-mood"}
        assert fold["harmonic_mode"] == ("fuzzy" if fold["heldout"].endswith("fuzzy") else "mood")
        assert fold["profile"]["selection"]["method"] == "leave-one-training-cohort-out"
    assert result["folds"][0]["profile"] == result["folds"][1]["profile"]
    changed = deepcopy(examples)
    changed[1]["order"].reverse()
    changed[1]["records"][0].update(bpm=160, key="3B")
    second = cross_validate_ranker(changed, **controls)
    # Changing another held-out mode/order variant cannot leak into either fold.
    assert second["folds"][0]["profile"] == result["folds"][0]["profile"]
    assert second["folds"][1]["profile"] == result["folds"][1]["profile"]


def test_inner_selection_never_splits_variants_of_a_training_cohort(monkeypatch):
    from superdj import harmonize_ranker as module
    examples = cohort_variants("a") + cohort_variants("b", 3)
    fitted_pools, evaluated_pools = [], []
    fit, metrics = module._fit_weights, module._metrics
    def fitted(values, *args):
        fitted_pools.append({row["name"] for row in values})
        return fit(values, *args)
    def evaluated(values, *args):
        evaluated_pools.append({row["name"] for row in values})
        return metrics(values, *args)
    monkeypatch.setattr(module, "_fit_weights", fitted)
    monkeypatch.setattr(module, "_metrics", evaluated)
    module._select(module._examples(examples), ["relation"], [1.], 2)
    assert fitted_pools == [{"b-fuzzy", "b-mood"}, {"a-fuzzy", "a-mood"}]
    assert evaluated_pools == [{"a-fuzzy", "a-mood"}, {"b-fuzzy", "b-mood"}]


def test_group_requirements_count_pools_not_number_of_variants():
    with pytest.raises(ValueError, match="two whole training cohorts"):
        fit_ranker(cohort_variants("a"), min_candidates=2)
    with pytest.raises(ValueError, match="three whole track-pool cohorts"):
        cross_validate_ranker(cohort_variants("a") + cohort_variants("b"), min_candidates=2)


def test_legacy_name_grouping_is_disclosed_and_ineligible_examples_are_skipped():
    values = [example("a"), example("b", 3), dict(name="excluded", training_eligible=False)]
    fitted = fit_ranker(values, feature_families=("relation",), l2_candidates=(1.,), min_candidates=2)
    assert fitted["cohort_grouping"]["fallback_example_names"] == ["a", "b"]
    assert fitted["cohort_grouping"]["training_group_count"] == 2
    assert {row["cohort_group_source"] for row in fitted["training_provenance"]} == {"example_name_fallback"}
    assert {row["name"] for row in fitted["training_provenance"]} == {"a", "b"}


@pytest.mark.parametrize("group", [True, 12, "", " "])
def test_malformed_explicit_cohort_group_is_rejected(group):
    malformed = dict(example("a"), cohort_group=group)
    with pytest.raises(ValueError, match="cohort_group"):
        fit_ranker([malformed, example("b")], min_candidates=2)


def test_loaded_profile_cannot_claim_independent_training_from_one_cohort():
    invalid = profile()
    for row in invalid["training_provenance"]:
        row["cohort_group"] = "one-track-pool"
    with pytest.raises(ValueError, match="two track-pool cohorts"):
        validate_ranker_profile(invalid)


def test_old_coefficients_cannot_silently_use_changed_relation_semantics():
    invalid = dict(profile(), model_version="superdj-conditional-ranker-1")
    with pytest.raises(ValueError, match="version"):
        validate_ranker_profile(invalid)


def test_gated_validation_counts_blocked_demonstrations_in_denominator():
    values=[example("a"),example("b",3),example("c",6)]
    values[2]["records"][1]["key"]="3A"
    result=cross_validate_ranker(values,feature_families=("relation",),l2_candidates=(1.,),min_candidates=2)
    heldout=result["folds"][2]
    assert heldout["raw_local_audit"]["steps"]==4
    assert heldout["gated_recommendation"]["steps"]==4
    assert heldout["gated_recommendation"]["demonstrated_next_blocked"]==2


@pytest.mark.parametrize("change",[
    {"model_version":"other"}, {"schema_version":2}, {"schema_version":True}, {"experimental":False},
    {"default_changed":True}, {"identifier_features":True}, {"utility_warning":"probability"},
    {"feature_family":"unknown"}, {"feature_names":["wrong"]}, {"coefficients":[1.]},
    {"l2_penalty":0}, {"min_candidates":True}, {"training_provenance":[]},
    {"training_provenance":[{"metadata_sha256":"x"*64},{"metadata_sha256":"b"*64}]},
])
def test_invalid_profile_is_rejected(change):
    invalid=dict(profile(),**change)
    with pytest.raises(ValueError):
        validate_ranker_profile(invalid)


def test_nonfinite_coefficients_and_inputs_are_rejected():
    invalid=profile()
    invalid["coefficients"][0]=float("nan")
    with pytest.raises(ValueError,match="finite"):
        validate_ranker_profile(invalid)
    for value in (None,0,-120,float("nan"),float("inf"),True,1001):
        with pytest.raises(ValueError):
            pair_features(track("a",value),track("b"))
    with pytest.raises(ValueError,match="musical key"):
        pair_features(track("a",120,"unmapped"),track("b"))


def test_invalid_training_and_pools_are_rejected():
    with pytest.raises(ValueError,match="two whole"):
        fit_ranker([example("a")])
    with pytest.raises(ValueError,match="three whole"):
        cross_validate_ranker([example("a"),example("b")])
    malformed=example("a")
    malformed["order"].pop()
    with pytest.raises(ValueError,match="every track once"):
        fit_ranker([malformed,example("b")])
    with pytest.raises(ValueError,match="enough remaining"):
        fit_ranker([example("a"),example("b")],min_candidates=10)
    with pytest.raises(ValueError,match="positive"):
        fit_ranker([example("a"),example("b")],l2_candidates=(0,))
    with pytest.raises(ValueError,match="distinct IDs"):
        rank_remaining_candidates(track("a"),[track("b"),track("b")],profile())
    with pytest.raises(ValueError,match="exclude the current"):
        rank_remaining_candidates(track("a"),[track("a")],profile())
