"""Fit explicit successor-score controls and validate on held-out playlists.

Only the declared BPM/key controls are fitted. Track IDs, names and paths are
not features. The result is an experimental *local ranking* profile; it does
not change whole-playlist defaults or certify audible harmonic compatibility.
"""
from __future__ import annotations

from .harmonize import _settings, score_pair
from .harmonize_successors import evaluate_successors


def control_grid():
    return [dict(mode=mode, priority=i / 20, direction=direction, bpm_tolerance=None)
            for mode in ("fuzzy", "mood") for i in range(21)
            for direction in ("ascending", "none", "descending")]


def _examples(examples):
    if not examples:
        raise ValueError("At least one training example is required")
    result = []
    names = set()
    for i, example in enumerate(examples):
        name = str(example.get("name", i))
        if name in names:
            raise ValueError("Example names must be unique")
        names.add(name)
        records, order = example["records"], example["order"]
        audit = evaluate_successors(records, order)
        by_id = {str(r.get("id") or r.get("path")): r for r in records}
        pairs = []
        for position, edge in enumerate(audit["edges"]):
            pairs.append((by_id[edge["current"]],
                          [by_id[identity] for identity in order[position + 1:]]))
        result.append((name, pairs))
    return result


def _candidates(candidates):
    candidates = control_grid() if candidates is None else list(candidates)
    if not candidates:
        raise ValueError("At least one candidate setting is required")
    normalized = []
    for item in candidates:
        mode, priority, direction, tolerance = _settings(
            item.get("mode", "fuzzy"), item.get("priority", .5),
            item.get("direction", "ascending"), item.get("bpm_tolerance"))
        row = dict(mode=mode, priority=priority, direction=direction, bpm_tolerance=tolerance)
        if row not in normalized:
            normalized.append(row)
    return normalized


def _metrics(prepared, settings, minimum, cache):
    # Every pair's cost is affine in priority. Cache both ends of that line,
    # using the public scorer, so fitting has the exact same objective.
    signature = (settings["mode"], settings["direction"], settings["bpm_tolerance"])
    if signature not in cache:
        at0 = dict(settings, priority=0)
        at1 = dict(settings, priority=1)
        cache[signature] = [
            (name, [[(score_pair(current, candidate, **at0)["cost"],
                      score_pair(current, candidate, **at1)["cost"])
                     for candidate in pool] for current, pool in pairs])
            for name, pairs in prepared]
    rows = []
    for name, pools in cache[signature]:
        count = best = strict_best = top3 = guaranteed_top3 = 0
        reciprocal = credit = 0.
        for pool in pools:
            if len(pool) < minimum:
                continue
            p = settings["priority"]
            costs = [a + p * (b - a) for a, b in pool]
            ranked = sorted(enumerate(costs), key=lambda item: item[1])
            start = 0
            while start < len(ranked):
                end = start + 1
                while end < len(ranked) and ranked[end][1] - ranked[start][1] <= 1e-9:
                    end += 1
                if any(index == 0 for index, _ in ranked[start:end]):
                    rank, ties = start + 1, end - start
                    break
                start = end
            count += 1
            best += rank == 1
            strict_best += rank == 1 and ties == 1
            top3 += rank <= 3
            guaranteed_top3 += rank + ties - 1 <= 3
            reciprocal += 1 / (rank + (ties - 1) / 2)
            credit += (1 / ties) if rank == 1 else 0
        rows.append(dict(name=name, edge_count=count, best_count=best,
                         strict_best_count=strict_best, top3_count=top3,
                         guaranteed_top3_count=guaranteed_top3,
                         reciprocal_midrank_sum=reciprocal, fractional_top1_sum=credit))
    count = sum(row["edge_count"] for row in rows)
    return dict(edge_count=count, best_count=sum(r["best_count"] for r in rows),
                top3_count=sum(r["top3_count"] for r in rows),
                mean_midpoint_reciprocal_rank=sum(r["reciprocal_midrank_sum"] for r in rows) / count if count else None,
                fractional_top1_fraction=sum(r["fractional_top1_sum"] for r in rows) / count if count else None,
                per_example=rows)


def fit_successor_controls(examples, *, candidates=None, min_candidates=5):
    """Return a fitted local profile, with no ID-specific rules or auto-install.

    The training criterion is the reciprocal midpoint rank, so a broad tie
    does not count as a confident top-one prediction. Steps with too few
    remaining choices are excluded. Settings are engineering controls,
    never probabilities or recovered proprietary parameters.
    """
    if not isinstance(min_candidates, int) or isinstance(min_candidates, bool) or min_candidates < 1:
        raise ValueError("min_candidates must be a positive integer")
    prepared = _examples(examples)
    candidates = _candidates(candidates)
    cache, evaluated = {}, []
    for settings in candidates:
        metrics = _metrics(prepared, settings, min_candidates, cache)
        if not metrics["edge_count"]:
            raise ValueError("No steps have enough remaining candidates to fit")
        evaluated.append(dict(settings=settings, metrics=metrics))
    def preference(row):
        settings, metrics = row["settings"], row["metrics"]
        # Equal metric values keep the closest profile to existing defaults.
        distance = (abs(settings["priority"] - .5) + .1 * (settings["mode"] != "fuzzy")
                    + .05 * (settings["direction"] != "ascending"))
        return metrics["mean_midpoint_reciprocal_rank"], metrics["top3_count"] / metrics["edge_count"], -distance
    winner = max(evaluated, key=preference)
    return dict(schema_version=1, settings=winner["settings"], training=winner["metrics"],
                candidate_count=len(evaluated), min_candidates=min_candidates,
                selection_scope="local pair ranking; whole-path defaults unchanged",
                objective="mean reciprocal midpoint rank",
                features=["BPM", "musical key"], installs_default=False)


def cross_validate_successor_controls(examples, *, candidates=None, min_candidates=5):
    """Leave one entire playlist out; never select controls on its results."""
    examples = list(examples)
    if len(examples) < 2:
        raise ValueError("At least two playlists are required for held-out validation")
    _examples(examples)  # Validate identities and names across all folds.
    candidates = _candidates(candidates)
    folds = []
    for i, heldout in enumerate(examples):
        training_examples = [e for j, e in enumerate(examples) if j != i]
        fitted = fit_successor_controls(training_examples, candidates=candidates, min_candidates=min_candidates)
        test = _metrics(_examples([heldout]), fitted["settings"], min_candidates, {})
        train_pairs = {tuple(pair) for e in training_examples for pair in zip(e["order"], e["order"][1:])}
        test_pairs = list(zip(heldout["order"], heldout["order"][1:]))
        folds.append(dict(heldout=str(heldout.get("name", i)), fitted=fitted, test=test,
                          previously_seen_directed_pairs=sum(pair in train_pairs for pair in test_pairs)))
    return dict(schema_version=1, validation="leave-one-playlist-out", folds=folds,
                full_training=fit_successor_controls(examples, candidates=candidates, min_candidates=min_candidates),
                defaults_changed=False, audio_compatibility_verified=False)
