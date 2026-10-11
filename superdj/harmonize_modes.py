"""Experimental mode-aware successor learning with remaining-pool context.

Music features contain BPM, tonal relations and residual key connectivity.
Track identifiers are used only for matching demonstrations and grouping
source cohorts. Every repeated order and mode variant stays in one holdout.
The independently defined harmonic-mode gate cannot be overridden by utility.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .harmonize import _number, _track, harmonic_relation
from .harmonize_ranker import _rank


MODEL_VERSION = "superdj-mode-pool-ranker-2"
FEATURE_VERSION = "musical-pool-2"
SHARING_OPTIONS = ("shared", "interactions", "separate")
FAMILIES = ("relation", "circle", "tonal", "pool", "tempo_pool")
TRAINING_SUPPORTS = ("all_candidates", "allowed_candidates")
RELATIONS = ("same_key", "relative_major_minor", "perfect_boost", "perfect_drop",
             "diagonal", "reverse_diagonal", "energy_boost", "parallel_major_minor",
             "semitone_lift", "incompatible", "unknown")
UTILITY_WARNING = "Relative musical choice utility; not confidence, probability or audible quality"


def _list(value, label):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return list(value)


def _mode(mode):
    if mode not in {"fuzzy", "mood"}:
        raise ValueError("mode must be fuzzy or mood")
    return mode


def _musical(record):
    if not isinstance(record, Mapping):
        raise ValueError("Every track must be a mapping")
    identity, bpm, key = _track(record)
    if bpm is None or bpm > 1000:
        raise ValueError("Mode ranker requires a positive BPM no greater than 1000")
    return identity, bpm, key


def _tonal(key):
    tonic = (7 * (key[0] - 8) + (9 if key[1] == "A" else 0)) % 12
    offsets = (0, 2, 3, 5, 7, 8, 10) if key[1] == "A" else (0, 2, 4, 5, 7, 9, 11)
    return tonic, {(tonic + note) % 12 for note in offsets}


def _relation(a, b, mode):
    if a is None or b is None:
        return "unknown"
    delta, same = (b[0] - a[0]) % 12, a[1] == b[1]
    if a == b:
        return "same_key"
    if not same and delta == 0:
        return "relative_major_minor"
    if same and delta in {1, 11}:
        return "perfect_boost" if delta == 1 else "perfect_drop"
    if not same and delta in {1, 11}:
        directed = delta == (11 if a[1] == "A" else 1)
        return "diagonal" if mode == "fuzzy" or directed else "reverse_diagonal"
    if same and delta == 2:
        return "energy_boost"
    if not same and delta == (3 if a[1] == "A" else 9):
        return "parallel_major_minor"
    if same and delta == 7:
        return "semitone_lift"
    return "incompatible"


def _allowed(a, b, mode):
    relation = _relation(a, b, mode)
    return (relation in RELATIONS[:5]
            or mode == "mood" and relation in RELATIONS[6:9])


def feature_names(family="pool", sharing="shared"):
    if family not in FAMILIES or sharing not in SHARING_OPTIONS:
        raise ValueError("Unsupported feature family or mode sharing")
    names = ["relation:" + relation for relation in RELATIONS]
    names += ["abs_bpm_delta/10", "signed_delta/10", "positive_delta/10",
              "negative_delta/10", "clipped_abs_delta/10", "large_delta/10",
              "squared_delta/1000", "same_bpm", "absolute_log_ratio*10"]
    if family in {"circle", "tonal", "pool", "tempo_pool"}:
        names += [f"circle:{interval};mode_change:{flipped}"
                  for flipped in (0, 1) for interval in range(12)]
        names.append("minor_to_major")
    if family in {"tonal", "pool", "tempo_pool"}:
        names += [f"tonic_interval:{interval}" for interval in range(12)]
        names += ["scale_overlap", "source_tonic_in_target", "target_tonic_in_source",
                  "source_minor", "target_minor", "mode_change"]
    if family == "tempo_pool":
        names += [f"relation:{relation}*{movement}" for relation in RELATIONS
                  for movement in ("abs_bpm_delta/10", "signed_delta/10")]
    if family in {"pool", "tempo_pool"}:
        names += ["same_key_fraction", "log_same_key_count", "out_degree_fraction",
                  "in_degree_fraction", "inverse_out_degree", "no_successor",
                  "residual_stranded_fraction", "sole_successor_fraction",
                  "nearest_successor_delta/10", "pool_median_distance/10",
                  "pool_min_bpm", "pool_max_bpm"]
    if sharing == "interactions":
        names += ["mood*" + name for name in list(names)]
    return names


def _pair(a, b, mode, family):
    _, bpm_a, key_a = _musical(a)
    _, bpm_b, key_b = _musical(b)
    delta = (bpm_b - bpm_a) / 10
    relation = _relation(key_a, key_b, mode)
    values = [float(relation == name) for name in RELATIONS]
    absolute = abs(delta)
    values += [absolute, delta, max(delta, 0), max(-delta, 0), min(absolute, 1),
               max(absolute - 1, 0), delta ** 2 / 10, float(absolute < .001),
               abs(math.log(bpm_b / bpm_a)) * 10]
    if family in {"circle", "tonal", "pool", "tempo_pool"}:
        if key_a and key_b:
            interval, flip = (key_b[0] - key_a[0]) % 12, key_a[1] != key_b[1]
            values += [float(flip == f and interval == i)
                       for f in (0, 1) for i in range(12)]
            values.append(float(key_a[1] == "A" and key_b[1] == "B"))
        else:
            values += [0.] * 25
    if family in {"tonal", "pool", "tempo_pool"}:
        if key_a and key_b:
            tonic_a, scale_a = _tonal(key_a)
            tonic_b, scale_b = _tonal(key_b)
            interval = (tonic_b - tonic_a) % 12
            values += [float(interval == i) for i in range(12)]
            values += [len(scale_a & scale_b) / 7, float(tonic_a in scale_b),
                       float(tonic_b in scale_a), float(key_a[1] == "A"),
                       float(key_b[1] == "A"), float(key_a[1] != key_b[1])]
        else:
            values += [0.] * 18
    if family == "tempo_pool":
        values += [float(relation == name) * movement for name in RELATIONS
                   for movement in (absolute, delta)]
    return values


def remaining_features(current, candidates, *, mode="fuzzy", family="pool", sharing="shared"):
    """Musical feature matrix; permutation equivariant, no identity columns."""
    mode = _mode(mode)
    feature_names(family, sharing)
    current_id, _, _ = _musical(current)
    candidates = _list(candidates, "Candidates")
    musical = [_musical(candidate) for candidate in candidates]
    ids = [row[0] for row in musical]
    if current_id in ids or len(set(ids)) != len(ids):
        raise ValueError("Candidates must be distinct and exclude the current track")
    keys, bpms = [row[2] for row in musical], np.asarray([row[1] for row in musical])
    count = len(candidates)
    if not count:
        return np.empty((0, len(feature_names(family, sharing))))
    adjacency = None
    if family in {"pool", "tempo_pool"}:
        adjacency = np.asarray([[_allowed(a, b, mode) for b in keys] for a in keys], dtype=float)
        np.fill_diagonal(adjacency, 0)
        outgoing, incoming = adjacency.sum(axis=1), adjacency.sum(axis=0)
        total_degree = outgoing + incoming
        sole_successors = adjacency[outgoing == 1].sum(axis=0)
        indices = np.arange(count)
        median, minimum, maximum = float(np.median(bpms)), bpms.min(), bpms.max()
    values = []
    for index, candidate in enumerate(candidates):
        row = _pair(current, candidate, mode, family)
        if family in {"pool", "tempo_pool"}:
            degree, in_degree = outgoing[index], incoming[index]
            same = sum(k == keys[index] for i, k in enumerate(keys) if i != index)
            residual_degree = total_degree - adjacency[index] - adjacency[:, index]
            stranded = ((residual_degree == 0) & (indices != index)).sum()
            viable = np.flatnonzero(adjacency[index])
            next_gap = min((abs(bpms[i] - bpms[index]) / 10 for i in viable), default=3.)
            sole = sole_successors[index]
            denominator = max(1, count - 1)
            row += [same / denominator, math.log1p(same), degree / denominator,
                    in_degree / denominator, 1 / (1 + degree), float(degree == 0),
                    stranded / denominator, sole / denominator, min(next_gap, 3),
                    abs(bpms[index] - median) / 10,
                    float(bpms[index] == minimum), float(bpms[index] == maximum)]
        if sharing == "interactions":
            row += [value * float(mode == "mood") for value in list(row)]
        values.append(row)
    return np.asarray(values, dtype=float)


def _minimum(value):
    if type(value) is not int or value < 1:
        raise ValueError("min_candidates must be a positive integer")
    return value


def _prepare(examples, minimum):
    """Merge explicit cohort labels and connected shared source identities."""
    examples = _list(examples, "Examples")
    prepared, names, tokens = [], set(), []
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError("Every example requires records, order and mode")
        name = str(example.get("name", index))
        if not name or name in names:
            raise ValueError("Example names must be nonempty and unique")
        names.add(name)
        mode = _mode(example.get("mode"))
        records, order = _list(example.get("records"), "Records"), _list(example.get("order"), "Order")
        by_id, sources = {}, set()
        for record in records:
            identity, _, _ = _musical(record)
            if identity in by_id:
                raise ValueError("Track IDs must be unique within an example")
            by_id[identity] = record
            # Identity is grouping evidence only, never a ranking feature.
            source = record.get("source_hash") or record.get("file_hash")
            if source:
                sources.add("source:" + str(source))
            else:
                sources.add("id:" + identity)
        if (not by_id or any(not isinstance(i, str) or not i for i in order)
                or len(order) != len(by_id) or len(set(order)) != len(order) or set(order) != set(by_id)):
            raise ValueError("Order must include every input track exactly once")
        group = example.get("cohort_group", example.get("cohortgroup"))
        if group is not None:
            if not isinstance(group, str) or not group.strip():
                raise ValueError("cohort_group must be a nonempty string")
            sources.add("explicit:" + group)
        metadata = [dict(bpm=_musical(by_id[i])[1], key=_musical(by_id[i])[2]) for i in order]
        digest = hashlib.sha256(json.dumps(dict(mode=mode, sequence=metadata), sort_keys=True,
            allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        queries = [(by_id[current], [by_id[i] for i in order[position + 1:]])
                   for position, current in enumerate(order[:-1]) if len(order) - position - 1 >= minimum]
        if not queries:
            raise ValueError("Every example needs at least one eligible remaining-pool query")
        prepared.append(dict(name=name, mode=mode, records=records, order=order, queries=queries,
                             metadata_sha256=digest, explicit_cohort_group=group))
        tokens.append(sources)
    if not prepared:
        raise ValueError("At least one example is required")
    parent = list(range(len(prepared)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    seen = {}
    for index, sources in enumerate(tokens):
        for source in sources:
            if source in seen:
                parent[find(index)] = find(seen[source])
            else:
                seen[source] = index
    components = {}
    for index in range(len(prepared)):
        components.setdefault(find(index), []).append(index)
    for indices in components.values():
        members = sorted(prepared[i]["name"] for i in indices)
        group = "cohort_" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:12]
        for index in indices:
            prepared[index]["group"] = group
    return prepared


def _matrices(examples, family, sharing):
    return [dict(example, matrices=[remaining_features(current, pool, mode=example["mode"],
            family=family, sharing=sharing) for current, pool in example["queries"]]) for example in examples]


def _fit(examples, penalty, *, training_support="all_candidates"):
    if training_support not in TRAINING_SUPPORTS:
        raise ValueError("Unsupported mode-ranker training support")
    supported = []
    for example in examples:
        queries = []
        for (current, pool), matrix in zip(example["queries"], example["matrices"]):
            if training_support == "allowed_candidates":
                key = _musical(current)[2]
                eligible = np.asarray([_allowed(key, _musical(r)[2], example["mode"]) for r in pool])
                if not eligible[0]:
                    continue
                matrix = matrix[eligible]
            queries.append(matrix)
        if queries:
            supported.append((example, queries))
    queries = [matrix for _, matrices in supported for matrix in matrices]
    if not queries:
        raise ValueError("No supported mode-specific training queries are available")
    matrix = np.vstack(queries)
    starts = np.r_[0, np.cumsum([len(query) for query in queries])[:-1]]
    query_index = np.repeat(np.arange(len(queries)), [len(query) for query in queries])
    masses = {}
    for example, _ in supported:
        token = example["group"], example["mode"]
        masses[token] = masses.get(token, 0) + 1
    weights = np.asarray([1 / (masses[example["group"], example["mode"]] * len(matrices))
                          for example, matrices in supported for _ in matrices])
    weights /= weights.sum()
    def objective(coefficients):
        logits = matrix @ coefficients
        maxima = np.maximum.reduceat(logits, starts)
        exponentials = np.exp(logits - maxima[query_index])
        denominators = np.add.reduceat(exponentials, starts)
        loss = float(weights @ (maxima + np.log(denominators) - logits[starts]))
        loss += .5 * penalty * float(coefficients @ coefficients)
        gradient = matrix.T @ (exponentials / denominators[query_index] * weights[query_index])
        gradient -= matrix[starts].T @ weights
        gradient += penalty * coefficients
        return loss, gradient
    fitted = minimize(objective, np.zeros(matrix.shape[1]), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=300, ftol=1e-9, gtol=1e-6))
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"Mode ranker fitting did not converge: {fitted.message}")
    return fitted.x


def _fit_sharing(examples, penalty, sharing, *, training_support="all_candidates"):
    if sharing == "separate":
        return {mode: _fit([e for e in examples if e["mode"] == mode], penalty, training_support=training_support)
                for mode in ("fuzzy", "mood")}
    return {"all": _fit(examples, penalty, training_support=training_support)}


def _metrics(examples, coefficients, sharing, *, gated=False):
    rows = []
    for example in examples:
        weights = coefficients[example["mode"] if sharing == "separate" else "all"]
        best = unique = top3 = guaranteed = blocked = feature_ceiling = 0
        top7 = guaranteed7 = early_steps = early7 = early_guaranteed7 = 0
        late_steps = late7 = late_guaranteed7 = 0
        early3_steps = early3 = early_guaranteed3 = 0
        late3_steps = late3 = late_guaranteed3 = 0
        literal3 = early_literal3 = late_literal3 = 0
        reciprocal = credit = nll = 0.
        for (current, pool), matrix in zip(example["queries"], example["matrices"]):
            informative = len(pool) >= 8
            early_steps += informative
            late_steps += not informative
            informative3 = len(pool) >= 4
            early3_steps += informative3
            late3_steps += not informative3
            logits = matrix @ weights
            feature_ceiling += int(np.sum(np.max(abs(matrix - matrix[0]), axis=1) < 1e-10) == 1)
            if gated:
                key = _musical(current)[2]
                eligible = np.asarray([_allowed(key, _musical(candidate)[2], example["mode"]) for candidate in pool])
                logits = np.where(eligible, logits, -np.inf)
            if not np.isfinite(logits[0]):
                blocked += 1
                continue
            # Blocked candidates have -inf utility; rank finite rows only.
            ranks, ties = _rank(logits[np.isfinite(logits)])
            rank, tie_count = ranks[0], ties[0]
            # The demonstrated successor is pool[0]. Do not break a tie by
            # its teacher position: use the same utility/ID ordering as the
            # public ranker's actual capped list. IDs resolve ties only.
            finite = np.flatnonzero(np.isfinite(logits))
            shortlist = sorted(finite, key=lambda i: (-float(logits[i]), _musical(pool[i])[0]))[:3]
            literal_hit = 0 in shortlist
            literal3 += literal_hit
            early_literal3 += informative3 and literal_hit
            late_literal3 += not informative3 and literal_hit
            best += rank == 1
            unique += rank == tie_count == 1
            top3 += rank <= 3
            guaranteed += rank + tie_count - 1 <= 3
            early3 += informative3 and rank <= 3
            early_guaranteed3 += informative3 and rank + tie_count - 1 <= 3
            late3 += not informative3 and rank <= 3
            late_guaranteed3 += not informative3 and rank + tie_count - 1 <= 3
            in7, guaranteed_in7 = rank <= 7, rank + tie_count - 1 <= 7
            top7 += in7
            guaranteed7 += guaranteed_in7
            early7 += informative and in7
            early_guaranteed7 += informative and guaranteed_in7
            late7 += not informative and in7
            late_guaranteed7 += not informative and guaranteed_in7
            reciprocal += 1 / (rank + (tie_count - 1) / 2)
            credit += 1 / tie_count if rank == 1 else 0.
            maximum = logits.max()
            nll += float(maximum + np.log(np.exp(logits - maximum).sum()) - logits[0])
        count = len(example["matrices"])
        rows.append(dict(name=example["name"], cohort_group=example["group"], mode=example["mode"],
                         steps=count, best_count=int(best), unique_best_count=int(unique), top3_count=int(top3),
                         guaranteed_top3_count=int(guaranteed), demonstrated_next_blocked=blocked,
                         literal_top3_count=int(literal3),
                         literal_top3_informative_count=int(early_literal3),
                         literal_top3_small_pool_count=int(late_literal3),
                         top3_informative_steps=int(early3_steps), top3_informative_count=int(early3),
                         guaranteed_top3_informative_count=int(early_guaranteed3),
                         top3_small_pool_steps=int(late3_steps), top3_small_pool_count=int(late3),
                         guaranteed_top3_small_pool_count=int(late_guaranteed3),
                         top7_count=int(top7), guaranteed_top7_count=int(guaranteed7),
                         top7_informative_steps=int(early_steps), top7_informative_count=int(early7),
                         guaranteed_top7_informative_count=int(early_guaranteed7),
                         top7_small_pool_steps=int(late_steps), top7_small_pool_count=int(late7),
                         guaranteed_top7_small_pool_count=int(late_guaranteed7),
                         fractional_top1_credit=float(credit), reciprocal_midrank_sum=float(reciprocal),
                         nll_sum=float(nll), unique_feature_ceiling=feature_ceiling))
    return rows


def _aggregate(rows):
    literal_available = bool(rows) and all("literal_top3_count" in row for row in rows)
    keys = ("steps", "best_count", "unique_best_count", "top3_count", "guaranteed_top3_count",
            "literal_top3_count", "literal_top3_informative_count", "literal_top3_small_pool_count",
            "demonstrated_next_blocked", "fractional_top1_credit", "reciprocal_midrank_sum", "nll_sum", "unique_feature_ceiling",
            "top3_informative_steps", "top3_informative_count", "guaranteed_top3_informative_count",
            "top3_small_pool_steps", "top3_small_pool_count", "guaranteed_top3_small_pool_count",
            "top7_count", "guaranteed_top7_count", "top7_informative_steps", "top7_informative_count",
            "guaranteed_top7_informative_count", "top7_small_pool_steps", "top7_small_pool_count", "guaranteed_top7_small_pool_count")
    result = {key: sum(row.get(key, 0) for row in rows) for key in keys}
    count = result["steps"]
    likelihood_count = count - result["demonstrated_next_blocked"]
    result.update(conditional_nll_steps=likelihood_count,
                  conditional_mean_nll=result["nll_sum"] / likelihood_count if likelihood_count else None,
                  mean_nll=result["nll_sum"] / count if count and not result["demonstrated_next_blocked"] else None,
                  nll_scope="Full mean NLL is undefined when demonstrated choices are blocked; conditional mean uses supported choices only")
    early = result["top7_informative_steps"]
    early3 = result["top3_informative_steps"]
    result.update(top3_informative_rate=result["top3_informative_count"] / early3 if early3 else None,
                  literal_top3_informative_rate=result["literal_top3_informative_count"] / early3 if literal_available and early3 else None,
                  guaranteed_top3_informative_rate=result["guaranteed_top3_informative_count"] / early3 if early3 else None,
                  primary_shortlist_size=3,
                  primary_top3_metric="literal_top3_informative_rate",
                  literal_top3_metrics_available=literal_available,
                  literal_top3_scope="Actual list capped at three; descending utility then lexical ID resolves ties, as in rank_mode_candidates; no teacher-position tie credit",
                  top3_scope="Legacy top3 fields use optimistic competition rank, which can include more than three tied tracks; main literal comparison requires at least four remaining candidates; final pools of at most three are separate")
    result.update(top7_informative_rate=result["top7_informative_count"] / early if early else None,
                  guaranteed_top7_informative_rate=result["guaranteed_top7_informative_count"] / early if early else None,
                  top7_scope="Main Top7 comparison requires at least eight remaining candidates; smaller pools reported separately")
    if count:
        result.update(unique_best_rate=result["unique_best_count"] / count, top3_rate=result["top3_count"] / count,
                      literal_top3_rate=result["literal_top3_count"] / count if literal_available else None,
                      mean_reciprocal_midrank=result["reciprocal_midrank_sum"] / count)
    return result


def _options(families, sharing_options, penalties, training_supports):
    families = list(dict.fromkeys(_list(families, "Feature families")))
    sharing_options = list(dict.fromkeys(_list(sharing_options, "Sharing options")))
    penalties = list(dict.fromkeys(_number(value, "L2 penalty") for value in _list(penalties, "L2 penalties")))
    supports = list(dict.fromkeys(_list(training_supports, "Training supports")))
    if (not families or any(family not in FAMILIES for family in families) or not sharing_options
            or any(sharing not in SHARING_OPTIONS for sharing in sharing_options)
            or not penalties or any(penalty <= 0 for penalty in penalties)
            or not supports or any(support not in TRAINING_SUPPORTS for support in supports)):
        raise ValueError("Unsupported feature, sharing or penalty options")
    return families, sharing_options, penalties, supports


def _select(examples, families, sharings, penalties, supports):
    groups = sorted({example["group"] for example in examples})
    if len(groups) < 2:
        raise ValueError("Inner selection requires two independent source cohorts")
    choices = []
    cache = {}
    for family in families:
        shared = _matrices(examples, family, "shared")
        for sharing in sharings:
            prepared = shared
            if sharing == "interactions":
                prepared = [dict(example, matrices=[np.c_[matrix, matrix * float(example["mode"] == "mood")]
                    for matrix in example["matrices"]]) for example in shared]
            cache[family, sharing] = prepared
            for support, penalty in ((s, p) for s in supports for p in penalties):
                rows, available = [], True
                for group in groups:
                    training = [e for e in prepared if e["group"] != group]
                    if sharing == "separate" and {e["mode"] for e in training} != {"fuzzy", "mood"}:
                        available = False
                        break
                    try:
                        coefficients = _fit_sharing(training, penalty, sharing, training_support=support)
                    except ValueError as error:
                        if support != "allowed_candidates" or "No supported" not in str(error):
                            raise
                        available = False
                        break
                    rows += _metrics([e for e in prepared if e["group"] == group], coefficients, sharing, gated=True)
                if available:
                    choices.append(dict(feature_family=family, sharing=sharing, l2_penalty=penalty, training_support=support,
                                        inner_validation=_aggregate(rows)))
    if not choices:
        raise ValueError("No model option has sufficient independent training cohorts")
    selected = max(choices, key=lambda option:(option["inner_validation"]["literal_top3_informative_rate"],
        option["inner_validation"]["unique_best_rate"], option["inner_validation"]["mean_reciprocal_midrank"],
        -(option["inner_validation"]["conditional_mean_nll"]
          if option["inner_validation"]["conditional_mean_nll"] is not None else math.inf),
        option["l2_penalty"]))
    prepared = cache[selected["feature_family"], selected["sharing"]]
    coefficients = _fit_sharing(prepared, selected["l2_penalty"], selected["sharing"], training_support=selected["training_support"])
    return selected, choices, coefficients


def _profile(examples, selected, options, coefficients, minimum):
    return dict(schema_version=1, model_version=MODEL_VERSION, feature_version=FEATURE_VERSION,
                experimental=True, identifier_features=False, default_changed=False,
                utility_warning=UTILITY_WARNING, feature_family=selected["feature_family"],
                sharing=selected["sharing"], feature_names=feature_names(selected["feature_family"], selected["sharing"]),
                coefficients={mode:list(map(float, values)) for mode, values in coefficients.items()},
                l2_penalty=selected["l2_penalty"], min_candidates=minimum,
                training_support=selected["training_support"],
                selection=dict(method="leave-one-training-source-cohort-out", selected=selected, options=options,
                    criterion="gated literal Top3 on at least four remaining candidates, unique Top1, reciprocal rank, supported NLL; blocked targets remain in every validation denominator"),
                training_provenance=[dict(name=e["name"], cohort_group=e["group"], mode=e["mode"],
                                         track_count=len(e["records"]), metadata_sha256=e["metadata_sha256"],
                                         queries=len(e["queries"]), training_skipped_queries=(sum(not _allowed(_musical(c)[2], _musical(p[0])[2], e["mode"]) for c,p in e["queries"])
                                             if selected["training_support"] == "allowed_candidates" else 0)) for e in examples],
                grouping_scope="shared track IDs/source hashes and explicit cohort labels form connected components",
                feature_inputs=["BPM", "musical key", "remaining-pool musical connectivity"],
                ranking_scope="teacher-forced local successor utility, not a complete path or audible quality model",
                harmony_gate="Caller mode is enforced independently; blocked demonstrations stay in validation denominator")


def fit_mode_ranker(examples, *, feature_families=("pool", "tempo_pool"), sharing_options=SHARING_OPTIONS,
                    l2_candidates=(.01, .1, 1.), min_candidates=4, training_supports=TRAINING_SUPPORTS):
    """Select and fit experimental JSON coefficients using inner cohorts only."""
    minimum = _minimum(min_candidates)
    prepared = _prepare(examples, minimum)
    selected, options, coefficients = _select(prepared, *_options(feature_families, sharing_options, l2_candidates, training_supports))
    return _profile(prepared, selected, options, coefficients, minimum)


def cross_validate_mode_ranker(examples, *, feature_families=("pool", "tempo_pool"), sharing_options=SHARING_OPTIONS,
                              l2_candidates=(.01, .1, 1.), min_candidates=4, training_supports=TRAINING_SUPPORTS):
    """Outer source cohort holdout, with all variants excluded from inner fit."""
    minimum = _minimum(min_candidates)
    prepared = _prepare(examples, minimum)
    groups = sorted({example["group"] for example in prepared})
    if len(groups) < 3:
        raise ValueError("Nested validation requires three independent source cohorts")
    options = _options(feature_families, sharing_options, l2_candidates, training_supports)
    folds = []
    for group in groups:
        training = [e for e in prepared if e["group"] != group]
        test = [e for e in prepared if e["group"] == group]
        selected, choices, coefficients = _select(training, *options)
        matrices = _matrices(test, selected["feature_family"], selected["sharing"])
        folds.append(dict(heldout_cohort_group=group, heldout_examples=[e["name"] for e in test],
                          profile=_profile(training, selected, choices, coefficients, minimum),
                          raw_local_audit=_metrics(matrices, coefficients, selected["sharing"]),
                          gated_recommendation=_metrics(matrices, coefficients, selected["sharing"], gated=True)))
    return dict(schema_version=1, model_version=MODEL_VERSION, validation="nested connected-source-cohort holdout",
                min_candidates=minimum, cohort_group_count=len(groups), folds=folds,
                raw_summary=_aggregate([row for fold in folds for row in fold["raw_local_audit"]]),
                gated_summary=_aggregate([row for fold in folds for row in fold["gated_recommendation"]]),
                default_changed=False, experimental=True, utility_warning=UTILITY_WARNING)


def validate_mode_ranker_profile(profile):
    """Reject incompatible/tampered JSON profiles; no executable model format."""
    if not isinstance(profile, Mapping) or type(profile.get("schema_version")) is not int or profile.get("schema_version") != 1:
        raise ValueError("Unsupported mode-ranker profile schema")
    if profile.get("model_version") != MODEL_VERSION or profile.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Unsupported mode-ranker feature version")
    if (profile.get("experimental") is not True or profile.get("identifier_features") is not False
            or profile.get("default_changed") is not False or profile.get("utility_warning") != UTILITY_WARNING):
        raise ValueError("Mode-ranker profile scope is incomplete")
    names = feature_names(profile.get("feature_family"), profile.get("sharing"))
    if profile.get("feature_names") != names:
        raise ValueError("Mode-ranker feature names or ordering differ")
    coefficients = profile.get("coefficients")
    required = {"fuzzy", "mood"} if profile["sharing"] == "separate" else {"all"}
    if not isinstance(coefficients, Mapping) or set(coefficients) != required:
        raise ValueError("Mode-ranker coefficient modes are incomplete")
    validated = {}
    for mode, values in coefficients.items():
        values = _list(values, "Coefficients")
        if len(values) != len(names):
            raise ValueError("Mode-ranker coefficient count differs")
        validated[mode] = [_number(value, "Coefficient") for value in values]
    if _number(profile.get("l2_penalty"), "L2 penalty") <= 0:
        raise ValueError("L2 penalty must be positive")
    if profile.get("training_support", "all_candidates") not in TRAINING_SUPPORTS:
        raise ValueError("Unsupported mode-ranker training support")
    _minimum(profile.get("min_candidates"))
    provenance = _list(profile.get("training_provenance"), "Training provenance")
    groups = set()
    for row in provenance:
        if not isinstance(row, Mapping) or not isinstance(row.get("cohort_group"), str) or not row["cohort_group"]:
            raise ValueError("Training provenance needs cohort groups")
        groups.add(row["cohort_group"])
        _mode(row.get("mode"))
        digest = row.get("metadata_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Training provenance needs SHA256 metadata fingerprints")
    if len(groups) < 2:
        raise ValueError("Training provenance needs two independent source cohorts")
    return dict(profile, coefficients=validated)


def rank_mode_candidates(current, candidates, profile, *, mode="fuzzy", tie_tolerance=1e-8,
                         shortlist_limit=3):
    """Rank remaining candidates with pool context and an independent mode gate."""
    profile = validate_mode_ranker_profile(profile)
    mode = _mode(mode)
    tolerance = _number(tie_tolerance, "tie_tolerance")
    if tolerance < 0:
        raise ValueError("tie_tolerance must be nonnegative")
    if type(shortlist_limit) is not int or not 1 <= shortlist_limit <= 7:
        raise ValueError("shortlist_limit must be an integer from one to seven")
    candidates = _list(candidates, "Candidates")
    matrix = remaining_features(current, candidates, mode=mode, family=profile["feature_family"], sharing=profile["sharing"])
    coefficients = profile["coefficients"][mode if profile["sharing"] == "separate" else "all"]
    utilities = matrix @ np.asarray(coefficients)
    if not np.all(np.isfinite(utilities)):
        raise ValueError("Mode-ranker utility is non-finite")
    eligible, blocked = [], []
    for candidate, utility in zip(candidates, utilities):
        relation = harmonic_relation(current.get("key_camelot") or current.get("key"),
                                    candidate.get("key_camelot") or candidate.get("key"), mode=mode)
        row = dict(id=_musical(candidate)[0], utility=float(utility), eligible=relation["accepted"],
                   harmonic=relation, rank=None, tie_count=None,
                   block_reasons=[] if relation["accepted"] else ["harmonic relation is outside caller mode"])
        (eligible if row["eligible"] else blocked).append(row)
    eligible.sort(key=lambda row:(-row["utility"], row["id"]))
    blocked.sort(key=lambda row:(-row["utility"], row["id"]))
    ranks, ties = _rank([row["utility"] for row in eligible], tolerance)
    for index, row in enumerate(eligible):
        row.update(rank=ranks[index], tie_count=ties[index])
    limit = min(shortlist_limit, len(eligible))
    boundary_tied = bool(limit and limit < len(eligible)
                        and eligible[limit-1]["rank"] == eligible[limit]["rank"])
    return dict(model_version=MODEL_VERSION, experimental=True, mode=mode, utility_warning=UTILITY_WARNING,
                ranking_scope="local pool-aware musical utility with independent mode gate",
                candidate_count=len(candidates), eligible_count=len(eligible), blocked_count=len(blocked),
                best_candidate_ids=[row["id"] for row in eligible if row["rank"] == 1],
                shortlist_ids=[row["id"] for row in eligible[:limit]], shortlist_limit=shortlist_limit,
                shortlist_boundary_tied=boundary_tied,
                shortlist_scope=f"At most {shortlist_limit} allowed choices, fewer when the pool is small; boundary ties are illustrative, not unique predictions",
                candidates=eligible + blocked,
                no_recommendation_reason=None if eligible else "No remaining candidate passes the caller mode")
