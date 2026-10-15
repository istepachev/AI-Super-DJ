"""Experimental local successor learning with an independent harmony gate.

The model learns relative choice utility from BPM and musical-key pairs.
Neither identifiers nor playlist positions enter its features. Inner model
selection holds out whole track-pool cohorts, including every repeated order
and mode variant; outer validation never selects on its held-out cohort. This
is not a global path optimizer, an audible
quality model, or a reproduction of project archive's proprietary algorithm.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .harmonize import _number, _track, harmonic_relation


MODEL_VERSION = "superdj-conditional-ranker-2"
FEATURE_FAMILIES = ("relation", "circle")
RELATIONS = ("same_key", "relative_major_minor", "perfect_boost", "perfect_drop",
             "diagonal", "energy_boost", "parallel_major_minor", "semitone_lift",
             "incompatible", "unknown")
UTILITY_WARNING = "Relative choice utility; not calibrated confidence, probability or musical quality"
DEFAULT_L2_CANDIDATES = (.001, .01, .1, 1.)


def _family(family):
    if family not in FEATURE_FAMILIES:
        raise ValueError("feature_family must be relation or circle")
    return family


def feature_names(feature_family="circle"):
    """Return the versioned feature order used by serialized coefficients."""
    _family(feature_family)
    names = ["relation:" + relation for relation in RELATIONS]
    names += ["abs_bpm_delta/10", "signed_bpm_delta/10", "positive_delta/10",
              "negative_delta/10", "clipped_abs_delta/10", "large_delta/10",
              "squared_delta/1000", "same_bpm", "absolute_log_bpm_ratio*10"]
    if feature_family == "circle":
        names += [f"circle_delta:{interval};mode_change:{flipped}"
                  for flipped in (0, 1) for interval in range(12)]
        names.append("minor_to_major")
    return names


def _musical(record):
    if not isinstance(record, Mapping):
        raise ValueError("Every track must be a mapping")
    _, bpm, key = _track(record)
    if bpm is None or key is None:
        raise ValueError("Ranker requires a positive BPM and a supported musical key")
    if bpm > 1000:
        raise ValueError("Ranker BPM must not exceed 1000")
    return bpm, key


def pair_features(current, candidate, feature_family="circle"):
    """Compute features from BPM/key only, never titles, IDs or source paths."""
    _family(feature_family)
    bpm_a, key_a = _musical(current)
    bpm_b, key_b = _musical(candidate)
    delta = (bpm_b - bpm_a) / 10
    absolute = abs(delta)
    relation = harmonic_relation(f"{key_a[0]}{key_a[1]}",
                                 f"{key_b[0]}{key_b[1]}", mode="mood")["relation"]
    values = [float(relation == name) for name in RELATIONS]
    values += [absolute, delta, max(delta, 0), max(-delta, 0), min(absolute, 1),
               max(absolute - 1, 0), absolute ** 2 / 10, float(absolute < .001),
               abs(math.log(bpm_b) - math.log(bpm_a)) * 10]
    if feature_family == "circle":
        interval = (key_b[0] - key_a[0]) % 12
        mode_change = key_a[1] != key_b[1]
        values += [float(mode_change == flipped and interval == difference)
                   for flipped in (False, True) for difference in range(12)]
        values.append(float(key_a[1] == "A" and key_b[1] == "B"))
    return values


def _sequence(value, label):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return list(value)


def _minimum(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("min_candidates must be a positive integer")
    return value


def _examples(examples):
    examples = _sequence(examples, "Examples")
    names, prepared = set(), []
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError("Every example requires records and order")
        if example.get("training_eligible") is False:
            continue
        name = str(example.get("name", index))
        if not name.strip():
            raise ValueError("Example names must be nonempty")
        if name in names:
            raise ValueError("Example names must be unique")
        names.add(name)
        group = example.get("cohort_group", example.get("cohortgroup"))
        if group is None:
            group = name
            group_source = "example_name_fallback"
        else:
            if not isinstance(group, str) or not group.strip():
                raise ValueError("cohort_group must be a nonempty string")
            group_source = "explicit_cohort_group"
        mode = example.get("mode")
        if mode is not None and mode not in {"fuzzy", "mood"}:
            raise ValueError("Example mode must be fuzzy or mood")
        records = _sequence(example.get("records"), "Example records")
        order = _sequence(example.get("order"), "Example order")
        by_id = {}
        for record in records:
            _musical(record)
            identity = _track(record)[0]
            if identity in by_id:
                raise ValueError("Duplicate track ids are not allowed")
            by_id[identity] = record
        if not by_id or any(not isinstance(i, str) or not i for i in order):
            raise ValueError("Example order requires nonempty string ids")
        if len(order) != len(by_id) or len(set(order)) != len(order) or set(order) != set(by_id):
            raise ValueError("Example order must contain every track once")
        # Provenance uses the demonstrated BPM/key sequence, not private paths.
        metadata = [dict(bpm=_musical(by_id[i])[0],
                         key="%s%s" % _musical(by_id[i])[1]) for i in order]
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True,
                allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        pairs = [(by_id[current], [by_id[i] for i in order[position+1:]])
                 for position, current in enumerate(order[:-1])]
        prepared.append(dict(name=name, records=records, order=order, pairs=pairs,
                             metadata_sha256=fingerprint, cohort_group=group,
                             cohort_group_source=group_source, mode=mode,
                             mode_provenance=example.get("mode_provenance")))
    if not prepared:
        raise ValueError("At least one example is required")
    return prepared


def _cohorts(examples):
    """Stable grouping uses provenance labels, never an identity feature."""
    return list(dict.fromkeys(example["cohort_group"] for example in examples))


def _queries(examples, family, minimum):
    return [np.asarray([pair_features(current, candidate, family) for candidate in pool])
            for example in examples for current, pool in example["pairs"]
            if len(pool) >= minimum]


def _fit_weights(examples, family, l2, minimum):
    queries = _queries(examples, family, minimum)
    if not queries:
        raise ValueError("No training steps have enough remaining candidates")
    count, dimension = len(queries), len(feature_names(family))
    def objective(weights):
        loss = .5 * l2 * np.dot(weights, weights)
        gradient = l2 * weights
        for matrix in queries:
            logits = matrix @ weights
            denominator = logsumexp(logits)
            loss += (denominator - logits[0]) / count
            gradient = gradient + (matrix.T @ np.exp(logits-denominator) - matrix[0]) / count
        return loss, gradient
    fitted = minimize(objective, np.zeros(dimension), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=400, ftol=1e-10, gtol=1e-7))
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"Ranker fitting did not converge: {fitted.message}")
    return fitted.x, dict(converged=True, iterations=int(fitted.nit),
                         training_steps=count, penalized_loss=float(fitted.fun))


def _rank(logits, tolerance=1e-8):
    """Anchored competition ties, without transitive tolerance chaining."""
    ordered = sorted(range(len(logits)), key=lambda i: -logits[i])
    ranks, ties = {}, {}
    start = 0
    while start < len(ordered):
        end, anchor = start+1, logits[ordered[start]]
        while end < len(ordered) and anchor-logits[ordered[end]] <= tolerance:
            end += 1
        for index in ordered[start:end]:
            ranks[index], ties[index] = start+1, end-start
        start = end
    return ranks, ties


def _metrics(examples, weights, family, minimum):
    queries = _queries(examples, family, minimum)
    if not queries:
        raise ValueError("No evaluation steps have enough remaining candidates")
    best = unique = top3 = guaranteed = 0
    credit = reciprocal = nll = 0.
    for matrix in queries:
        logits = matrix @ weights
        ranks, ties = _rank(logits)
        rank, tie_count = ranks[0], ties[0]
        best += rank == 1
        unique += rank == tie_count == 1
        top3 += rank <= 3
        guaranteed += rank+tie_count-1 <= 3
        credit += (1/tie_count) if rank == 1 else 0
        reciprocal += 1/(rank+(tie_count-1)/2)
        nll += float(logsumexp(logits)-logits[0])
    count = len(queries)
    return dict(steps=count, best_count=best, unique_best_count=unique,
                top3_count=top3, guaranteed_top3_count=guaranteed,
                fractional_top1_credit=credit, mean_reciprocal_midrank=reciprocal/count,
                mean_nll=nll/count, scope="ungated immediate-edge audit, not a recommendation or whole-path score")


def _options(feature_families, l2_candidates):
    families = _sequence(feature_families, "Feature families")
    penalties = _sequence(l2_candidates, "L2 candidates")
    if not families or not penalties:
        raise ValueError("At least one feature family and L2 candidate are required")
    families = list(dict.fromkeys(_family(family) for family in families))
    penalties = list(dict.fromkeys(_number(penalty, "L2 penalty") for penalty in penalties))
    if any(penalty <= 0 for penalty in penalties):
        raise ValueError("L2 penalties must be positive")
    return families, penalties


def _select(examples, families, penalties, minimum):
    groups = _cohorts(examples)
    if len(groups) < 2:
        raise ValueError("Model selection requires at least two whole training cohorts")
    options = []
    for family in families:
        for penalty in penalties:
            validations = []
            for heldout in groups:
                training = [e for e in examples if e["cohort_group"] != heldout]
                validation = [e for e in examples if e["cohort_group"] == heldout]
                weights, _ = _fit_weights(training, family, penalty, minimum)
                validations.append(_metrics(validation, weights, family, minimum))
            steps = sum(row["steps"] for row in validations)
            options.append(dict(feature_family=family, l2_penalty=penalty,
                inner_mean_reciprocal_midrank=sum(row["steps"]*row["mean_reciprocal_midrank"]
                                                for row in validations)/steps,
                inner_mean_nll=sum(row["steps"]*row["mean_nll"] for row in validations)/steps))
    winner = max(options, key=lambda option:(option["inner_mean_reciprocal_midrank"],
                    -option["inner_mean_nll"], option["l2_penalty"]))
    return winner, options


def _profile(examples, selected, options, minimum):
    family, penalty = selected["feature_family"], selected["l2_penalty"]
    weights, convergence = _fit_weights(examples, family, penalty, minimum)
    return dict(schema_version=1, model_version=MODEL_VERSION, experimental=True,
                ranking_scope="local successor utility; does not optimize a complete playlist",
                feature_family=family, feature_names=feature_names(family),
                coefficients=list(map(float,weights)), l2_penalty=penalty,
                min_candidates=minimum, utility_warning=UTILITY_WARNING,
                feature_inputs=["BPM", "musical key"], identifier_features=False,
                selection=dict(method="leave-one-training-cohort-out",
                    holdout_scope="all order and mode variants of one track-pool cohort",
                    training_cohort_groups=_cohorts(examples),
                    inner_splits=[dict(heldout_cohort_group=group,
                        heldout_examples=[e["name"] for e in examples if e["cohort_group"] == group],
                        training_cohort_groups=[g for g in _cohorts(examples) if g != group])
                        for group in _cohorts(examples)],
                    objective="reciprocal midpoint rank", selected=selected, options=options),
                training_provenance=[dict(name=e["name"], track_count=len(e["records"]),
                    metadata_sha256=e["metadata_sha256"], cohort_group=e["cohort_group"],
                    cohort_group_source=e["cohort_group_source"], mode=e["mode"],
                    mode_provenance=e["mode_provenance"]) for e in examples],
                cohort_grouping=dict(training_group_count=len(_cohorts(examples)),
                    fallback_example_names=[e["name"] for e in examples
                        if e["cohort_group_source"] == "example_name_fallback"],
                    warning="Missing cohort_group falls back to the example name; callers must group repeated track pools explicitly"),
                convergence=convergence, default_changed=False,
                harmony_gate="Caller harmonic mode is enforced independently of learned utility")


def fit_ranker(examples, *, feature_families=FEATURE_FAMILIES,
               l2_candidates=DEFAULT_L2_CANDIDATES, min_candidates=5):
    """Fit a JSON-compatible experimental profile; training-only model selection.

    ``examples`` contain ``name``, ``records``, a complete ``order`` of IDs and
    ``cohort_group`` for repeated orders of one track pool. At least two cohorts
    are required; older manifests fall back to names and disclose that limit.
    Each demonstration supplies
    only its actual remaining pool; late forced choices are excluded by the
    requested minimum. Calling this function never installs a default.
    """
    prepared = _examples(examples)
    minimum = _minimum(min_candidates)
    families, penalties = _options(feature_families, l2_candidates)
    selected, options = _select(prepared, families, penalties, minimum)
    return _profile(prepared, selected, options, minimum)


def validate_ranker_profile(profile):
    """Validate a loaded JSON profile without accepting executable model data."""
    if not isinstance(profile, Mapping):
        raise ValueError("Ranker profile must be a mapping")
    if (type(profile.get("schema_version")) is not int or profile.get("schema_version") != 1
            or profile.get("model_version") != MODEL_VERSION):
        raise ValueError("Unsupported ranker profile version")
    if profile.get("experimental") is not True or profile.get("identifier_features") is not False:
        raise ValueError("Ranker must be experimental and use no identifier features")
    if profile.get("default_changed") is not False or profile.get("utility_warning") != UTILITY_WARNING:
        raise ValueError("Ranker profile is missing its scope or utility warning")
    family = _family(profile.get("feature_family"))
    if profile.get("feature_names") != feature_names(family):
        raise ValueError("Ranker profile feature names or order do not match this version")
    coefficients = _sequence(profile.get("coefficients"), "Coefficients")
    if len(coefficients) != len(feature_names(family)):
        raise ValueError("Ranker coefficient count does not match features")
    coefficients = [_number(value,"Ranker coefficient") for value in coefficients]
    if _number(profile.get("l2_penalty"), "L2 penalty") <= 0:
        raise ValueError("L2 penalty must be positive")
    _minimum(profile.get("min_candidates"))
    provenance = _sequence(profile.get("training_provenance"), "Training provenance")
    if len(provenance)<2 or any(not isinstance(row,Mapping) for row in provenance):
        raise ValueError("Training provenance requires at least two playlist records")
    for row in provenance:
        digest = row.get("metadata_sha256")
        if (not isinstance(digest,str) or len(digest)!=64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("Training provenance requires SHA256 metadata fingerprints")
    groups = [row.get("cohort_group", row.get("name")) for row in provenance]
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("Training provenance requires nonempty cohort groups or legacy example names")
    if len(set(groups)) < 2:
        raise ValueError("Training provenance requires at least two track-pool cohorts")
    return dict(profile, coefficients=coefficients)


def rank_remaining_candidates(current, candidates, profile, *, mode="fuzzy", tie_tolerance=1e-8):
    """Rank only remaining harmonically allowed candidates; keep blocked evidence.

    Learned utility cannot override ``harmonic_relation(..., mode=mode)``.
    Blocked candidates remain visible with ``rank=None``. Utility is available
    for inspection but is not a recommendation. Caller must also enforce its
    source-analysis quality, phrase, vocal, rhythm and render constraints.
    """
    profile = validate_ranker_profile(profile)
    _musical(current)
    if mode not in {"fuzzy", "mood"}:
        raise ValueError("mode must be fuzzy or mood")
    tolerance = _number(tie_tolerance,"tie_tolerance")
    if tolerance < 0:
        raise ValueError("tie_tolerance must be nonnegative")
    candidates = _sequence(candidates,"Candidates")
    current_id = _track(current)[0]
    seen, eligible, blocked = set(), [], []
    for candidate in candidates:
        _musical(candidate)
        identity = _track(candidate)[0]
        if identity==current_id or identity in seen:
            raise ValueError("Candidates must have distinct IDs and exclude the current track")
        seen.add(identity)
        values = pair_features(current,candidate,profile["feature_family"])
        utility = float(np.dot(values,profile["coefficients"]))
        if not math.isfinite(utility):
            raise ValueError("Ranker produced a non-finite utility")
        relation = harmonic_relation(current.get("key_camelot") or current.get("key"),
                    candidate.get("key_camelot") or candidate.get("key"),mode=mode)
        row = dict(id=identity, utility=utility, utility_warning=UTILITY_WARNING,
                   eligible=relation["accepted"], harmonic=relation, rank=None, tie_count=None,
                   block_reasons=[] if relation["accepted"] else ["harmonic relation is not allowed by caller mode"])
        (eligible if row["eligible"] else blocked).append(row)
    eligible.sort(key=lambda row:(-row["utility"],row["id"]))
    blocked.sort(key=lambda row:(-row["utility"],row["id"]))
    ranks, ties = _rank([row["utility"] for row in eligible],tolerance)
    for index,row in enumerate(eligible):
        row.update(rank=ranks[index],tie_count=ties[index])
    return dict(model_version=MODEL_VERSION, experimental=True,
                ranking_scope="local remaining candidates with independent harmonic gate",
                utility_warning=UTILITY_WARNING, harmonic_mode=mode,
                candidate_count=len(candidates), eligible_count=len(eligible), blocked_count=len(blocked),
                best_candidate_ids=[row["id"] for row in eligible if row["rank"]==1],
                candidates=eligible+blocked,
                no_recommendation_reason=None if eligible else "No remaining candidate passes the harmonic gate")


def _gated_metrics(example, profile, mode, minimum):
    steps = best = unique = top3 = blocked = 0
    for current,pool in example["pairs"]:
        if len(pool)<minimum:
            continue
        rows = rank_remaining_candidates(current,pool,profile,mode=mode)["candidates"]
        actual = next(row for row in rows if row["id"] == _track(pool[0])[0])
        steps += 1
        if actual["rank"] is None:
            blocked += 1
            continue
        best += actual["rank"]==1
        unique += actual["rank"]==actual["tie_count"]==1
        top3 += actual["rank"]<=3
    return dict(steps=steps, demonstrated_next_blocked=blocked, best_count=best,
                unique_best_count=unique, top3_count=top3,
                scope="gated recommendation; blocked demonstrations remain in denominator")


def cross_validate_ranker(examples, *, feature_families=FEATURE_FAMILIES,
                          l2_candidates=DEFAULT_L2_CANDIDATES, min_candidates=5, mode="fuzzy"):
    """Outer whole-cohort holdout with model selection confined to inner train.

    Raw immediate rankings and harmony-gated recommendations are reported
    separately. No profile is trained on any variant of the held-out track pool.
    The returned profiles are private experiment artifacts, not defaults.
    """
    prepared = _examples(examples)
    groups = _cohorts(prepared)
    if len(groups)<3:
        raise ValueError("Nested validation requires at least three whole track-pool cohorts")
    if mode not in {"fuzzy","mood"}:
        raise ValueError("mode must be fuzzy or mood")
    minimum = _minimum(min_candidates)
    families,penalties = _options(feature_families,l2_candidates)
    folds=[]
    profiles = {}
    for example in prepared:
        group = example["cohort_group"]
        if group not in profiles:
            training=[e for e in prepared if e["cohort_group"]!=group]
            selected,options=_select(training,families,penalties,minimum)
            profiles[group]=_profile(training,selected,options,minimum)
        profile=profiles[group]
        weights=np.asarray(profile["coefficients"])
        metrics=_metrics([example],weights,profile["feature_family"],minimum)
        harmonic_mode = example["mode"] or mode
        folds.append(dict(heldout=example["name"], heldout_metadata_sha256=example["metadata_sha256"],
                          heldout_cohort_group=group,
                          heldout_cohort_examples=[e["name"] for e in prepared if e["cohort_group"]==group],
                          harmonic_mode=harmonic_mode,
                          profile=profile, raw_local_audit=metrics,
                          gated_recommendation=_gated_metrics(example,profile,harmonic_mode,minimum)))
    return dict(schema_version=1, model_version=MODEL_VERSION, experimental=True,
                validation="nested leave-one-whole-cohort-out",
                holdout_scope="all order and mode variants of one track-pool cohort",
                cohort_group_count=len(groups), cohort_groups=groups,
                fallback_example_names=[e["name"] for e in prepared
                    if e["cohort_group_source"] == "example_name_fallback"],
                min_candidates=minimum, harmonic_mode=mode, folds=folds,
                harmonic_mode_scope="explicit per-example mode, with caller mode as fallback",
                utility_warning=UTILITY_WARNING, default_changed=False,
                scope="Local next-track learning only; no whole-order or audible quality claim")
