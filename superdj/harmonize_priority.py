"""Opt-in, fixed-parameter priority-aware successor prototype.

Controls are explicit caller inputs. This single-cohort training API is separate
from the legacy cohort-selected learner and never claims independent validation.
The caller-mode harmonic gate remains independent of every fitted coefficient.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

from .harmonize import _number
from .harmonize_modes import (FAMILIES, SHARING_OPTIONS, TRAINING_SUPPORTS, UTILITY_WARNING,
    _allowed, _coefficients_for_mode, _fit_sharing, _list, _minimum, _mode, _musical, _prepare)


MODEL_VERSION = "superdj-mode-priority-ranker-3"
MODE_COEFFICIENT_MODEL_VERSION = "superdj-mode-priority-ranker-4"
FEATURE_VERSION = "musical-priority-1"
PRIORITIES = ("key", "bpm", "unknown")
TOLERANCE_NAMES = ("explicit_tolerance:within", "explicit_tolerance:clipped_gap_ratio",
                   "explicit_tolerance:clipped_excess_ratio")


def routing_controls(priority="unknown", bpm_tolerance=None):
    if priority not in PRIORITIES:
        raise ValueError("Routing priority must be key, bpm or unknown")
    if bpm_tolerance is not None:
        bpm_tolerance = _number(bpm_tolerance, "BPM tolerance")
        if bpm_tolerance <= 0:
            raise ValueError("BPM tolerance must be positive or null")
    return dict(priority=priority, bpm_tolerance=bpm_tolerance)


def feature_names(family="tempo_rank_density", sharing="interactions"):
    from .harmonize_modes import feature_names as base_names
    base = base_names(family, "shared")
    names = base + [f"priority:{priority}*{name}" for priority in ("key", "bpm")
                    for name in base[:20]] + list(TOLERANCE_NAMES)
    if sharing == "interactions":
        names += ["mood*" + name for name in list(names)]
    elif sharing not in SHARING_OPTIONS:
        raise ValueError("Unsupported mode sharing")
    return names


def tolerance_features(current_bpm, candidate_bpms, bpm_tolerance=None):
    """Explicit soft preference columns; these never filter candidate support."""
    if bpm_tolerance is None:
        return np.zeros((len(candidate_bpms), 3))
    routing_controls("unknown", bpm_tolerance)
    gaps = np.abs(np.asarray(candidate_bpms, dtype=float)-current_bpm)
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = np.minimum(gaps / bpm_tolerance, 3.)
        excess = np.minimum(np.maximum(gaps-bpm_tolerance, 0.) / bpm_tolerance, 3.)
    return np.column_stack((gaps <= bpm_tolerance+1e-6, ratio, excess))


def extend_features(base, current_bpm, candidate_bpms, *, mode="fuzzy",
                    sharing="shared", priority="unknown", bpm_tolerance=None):
    if sharing not in SHARING_OPTIONS:
        raise ValueError("Unsupported mode sharing")
    controls = routing_controls(priority, bpm_tolerance)
    matrix = np.column_stack((base, base[:, :20]*float(priority == "key"),
        base[:, :20]*float(priority == "bpm"),
        tolerance_features(current_bpm, candidate_bpms, controls["bpm_tolerance"])))
    if sharing == "interactions":
        matrix = np.column_stack((matrix, matrix*float(mode == "mood")))
    return matrix


def effective_weights(profile, mode, *, priority="unknown", bpm_tolerance=None):
    """Factor explicit pair-control interactions for the dynamic pool engine."""
    from .harmonize_modes import feature_names as base_names
    routing_controls(priority, bpm_tolerance)
    count = len(base_names(profile["feature_family"], "shared"))
    width = count+43
    values = np.asarray(_coefficients_for_mode(profile, mode))
    if profile["sharing"] == "interactions":
        with np.errstate(over="ignore", invalid="ignore"):
            values = values[:width]+float(mode == "mood")*values[width:]
    base = values[:count].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        if priority == "key":
            base[:20] += values[count:count+20]
        elif priority == "bpm":
            base[:20] += values[count+20:count+40]
    tolerance = values[count+40:count+43]
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(tolerance)):
        raise ValueError("Combined priority coefficients are non-finite")
    return base, tolerance


def _example_controls(example):
    if not isinstance(example, Mapping):
        raise ValueError("Every priority example must be an object")
    controls = example.get("routing_controls", {})
    if not isinstance(controls, Mapping):
        raise ValueError("Explicit routing_controls must be an object")
    if "priority" not in controls and "priority" not in example:
        raise ValueError("Priority-aware fitting requires explicit priority; filenames are not controls")
    priority = controls.get("priority", example.get("priority"))
    tolerance = controls.get("bpm_tolerance", example.get("bpm_tolerance"))
    chosen = routing_controls(priority, tolerance)
    if "priority" in controls and "priority" in example and controls["priority"] != example["priority"]:
        raise ValueError("Conflicting explicit priority controls")
    if "bpm_tolerance" in controls and "bpm_tolerance" in example:
        if routing_controls(priority, example["bpm_tolerance"]) != chosen:
            raise ValueError("Conflicting explicit BPM tolerance")
    if example.get("priority_aware_training_eligible") is False or (
            example.get("training_eligible") is False and example.get("priority_aware_training_eligible") is not True):
        raise ValueError("Example is not eligible for priority-aware fitting")
    return chosen


def fit_priority_mode_ranker_fixed(examples, *, feature_family="tempo_rank_density",
        sharing="interactions", l2_penalty=.1, min_candidates=4,
        training_support="allowed_candidates"):
    """Fit one declared option; allow one cohort only with training-only v3 scope.

    Each matrix retains features of the complete actual remaining pool. Allowed
    support is sliced only AFTER feature extraction and before accumulation.
    Unsupported teacher targets are skipped in fitting, never in evaluation.
    """
    from .harmonize_modes import remaining_features
    if feature_family not in FAMILIES or sharing not in SHARING_OPTIONS:
        raise ValueError("Unsupported fixed feature family or sharing")
    if training_support not in TRAINING_SUPPORTS:
        raise ValueError("Unsupported fixed training support")
    penalty = _number(l2_penalty, "L2 penalty")
    if penalty <= 0:
        raise ValueError("L2 penalty must be positive")
    examples = _list(examples, "Examples")
    explicit = [_example_controls(example) for example in examples]
    minimum = _minimum(min_candidates)
    prepared = _prepare(examples, minimum)
    training, provenance = [], []
    for example, controls in zip(prepared, explicit):
        queries, matrices = [], []
        skipped = 0
        for current, pool in example["queries"]:
            eligible = np.asarray([_allowed(_musical(current)[2], _musical(record)[2], example["mode"]) for record in pool])
            if training_support == "allowed_candidates" and not eligible[0]:
                skipped += 1
                continue
            matrix = remaining_features(current, pool, mode=example["mode"], family=feature_family,
                sharing=sharing, priority_aware=True, **controls)
            if training_support == "allowed_candidates":
                matrix = matrix[eligible]
                fit_pool = [record for record, allowed in zip(pool, eligible) if allowed]
            else:
                fit_pool = pool
            queries.append((current, fit_pool))
            matrices.append(matrix)
        training.append(dict(example, queries=queries, matrices=matrices))
        digest = hashlib.sha256(json.dumps(dict(musical_metadata_sha256=example["metadata_sha256"],
            controls=controls), sort_keys=True, allow_nan=False).encode()).hexdigest()
        provenance.append(dict(name=example["name"], cohort_group=example["group"], mode=example["mode"],
            routing_controls=controls, track_count=len(example["records"]), metadata_sha256=digest,
            queries=len(example["queries"]), training_skipped_queries=skipped))
    # Support has already been sliced, so the existing numerical optimizer can
    # consume all stored rows without copying each mask a second time.
    coefficients = _fit_sharing(training, penalty, sharing, training_support="all_candidates")
    parameters = dict(feature_family=feature_family, sharing=sharing, l2_penalty=penalty,
                      min_candidates=minimum, training_support=training_support)
    profile = dict(schema_version=3, model_version=MODEL_VERSION, feature_version=FEATURE_VERSION,
        priority_aware=True, experimental=True, identifier_features=False, default_changed=False,
        training_only=True, independent_validation=False, utility_warning=UTILITY_WARNING,
        feature_names=feature_names(feature_family, sharing), **parameters,
        coefficients={name:list(map(float, values)) for name, values in coefficients.items()},
        selection=dict(method="fixed-parameters", parameters=parameters, grid_search_performed=False),
        training_provenance=provenance,
        feature_inputs=["BPM", "musical key", "actual remaining musical pool", "explicit priority", "nullable explicit BPM tolerance"],
        grouping_scope="Connected shared source components; no cohort holdout claimed",
        ranking_scope="Fixed-option training prototype; local successor utility is not full-path or audible quality",
        harmony_gate="Caller mode remains independent; blocked targets remain in every evaluation denominator",
        training_support_rows_prefiltered=True)
    return validate_priority_profile(profile)


def validate_priority_profile(profile):
    versions = {3: MODEL_VERSION, 4: MODE_COEFFICIENT_MODEL_VERSION}
    if (not isinstance(profile, Mapping) or type(profile.get("schema_version")) is not int
            or profile["schema_version"] not in versions):
        raise ValueError("Unsupported priority profile schema")
    if profile.get("model_version") != versions[profile["schema_version"]] or profile.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Unsupported priority feature version")
    required = dict(priority_aware=True, experimental=True, identifier_features=False,
                    default_changed=False, training_only=True, independent_validation=False)
    if any(profile.get(name) is not value for name,value in required.items()) or profile.get("utility_warning") != UTILITY_WARNING:
        raise ValueError("Priority profile training-only scope is incomplete")
    names = feature_names(profile.get("feature_family"), profile.get("sharing"))
    if profile.get("feature_names") != names:
        raise ValueError("Priority profile feature names or ordering differ")
    expected = {"fuzzy", "mood"} if profile["sharing"] == "separate" else {"all"}
    coefficients = profile.get("coefficients")
    if not isinstance(coefficients, Mapping) or set(coefficients) != expected:
        raise ValueError("Priority profile coefficients are incomplete")
    validated = {}
    for name, values in coefficients.items():
        values = _list(values, "Coefficients")
        if len(values) != len(names):
            raise ValueError("Priority profile coefficient count differs")
        validated[name] = [_number(value, "Coefficient") for value in values]
    mode_coefficients = None
    if profile["schema_version"] == 4:
        overrides = profile.get("mode_coefficients")
        if not isinstance(overrides, Mapping) or not overrides or not set(overrides) <= {"fuzzy", "mood"}:
            raise ValueError("Mode coefficient overrides need known modes and at least one vector")
        mode_coefficients = {}
        for mode, values in overrides.items():
            values = _list(values, "Mode coefficients")
            if len(values) != len(names):
                raise ValueError("Mode coefficient override count differs")
            mode_coefficients[mode] = [_number(value, "Mode coefficient") for value in values]
    elif "mode_coefficients" in profile:
        raise ValueError("Mode coefficient overrides require priority profile version 4")
    if _number(profile.get("l2_penalty"), "L2 penalty") <= 0 or profile.get("training_support") not in TRAINING_SUPPORTS:
        raise ValueError("Invalid fixed training parameters")
    _minimum(profile.get("min_candidates"))
    parameters = {name:profile[name] for name in ("feature_family", "sharing", "l2_penalty", "min_candidates", "training_support")}
    selection = profile.get("selection")
    if not isinstance(selection, Mapping) or selection.get("method") != "fixed-parameters" or selection.get("parameters") != parameters or selection.get("grid_search_performed") is not False:
        raise ValueError("Priority profile needs explicit fixed-parameter selection scope")
    provenance = _list(profile.get("training_provenance"), "Training provenance")
    if not provenance:
        raise ValueError("Priority profile needs training provenance")
    for row in provenance:
        if not isinstance(row, Mapping) or not isinstance(row.get("cohort_group"), str) or not row["cohort_group"]:
            raise ValueError("Priority provenance needs source cohort groups")
        _mode(row.get("mode"))
        controls = row.get("routing_controls")
        if not isinstance(controls, Mapping) or set(controls) != {"priority", "bpm_tolerance"}:
            raise ValueError("Priority provenance needs explicit controls")
        routing_controls(**controls)
        digest = row.get("metadata_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Priority provenance needs SHA256 metadata fingerprints")
    result = dict(profile, coefficients=validated)
    if mode_coefficients is not None:
        result["mode_coefficients"] = mode_coefficients
    return result
