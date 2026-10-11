"""Audit demonstrated next-track choices against our local BPM/key score.

This is a teacher-forced comparison: at each step only tracks that have not
already played are candidates. It does not claim that the cheapest immediate
edge is the first edge of the cheapest complete playlist. Ties are reported
rather than treating our deterministic ID tie break as a musical preference.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .harmonize import VERSION, _number, _settings, _track, score_pair


def _identity(record: Mapping[str, Any]) -> str:
    if not isinstance(record, Mapping):
        raise ValueError("Every track must be a record with an id or path")
    return _track(record)[0]


def _controls(mode, priority, direction, bpm_tolerance, tie_tolerance):
    mode, priority, direction, bpm_tolerance = _settings(
        mode, priority, direction, bpm_tolerance)
    tie_tolerance = _number(tie_tolerance, "tie_tolerance")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be nonnegative")
    settings = dict(mode=mode, priority=priority, direction=direction,
                    bpm_tolerance=bpm_tolerance)
    return settings, tie_tolerance


def rank_candidates(current_record: Mapping[str, Any],
                    candidate_records: Sequence[Mapping[str, Any]], *,
                    mode="fuzzy", priority=.5, direction="ascending",
                    bpm_tolerance=None, tie_tolerance=1e-9):
    """Return candidates ordered by local edge cost, with competition ranks.

    Rows contain ``id``, ``rank``, ``tie_count``, ``gap_from_best`` and the full
    ``score_pair`` evidence in ``score``. Equal costs have the same rank (for
    example 1, 1, 3). Within a tolerance group every cost must be within
    ``tie_tolerance`` of the group's lowest cost; near-ties do not accumulate
    transitively. ID sorting is only a reproducible display tie break. An empty
    candidate pool is valid and returns an empty list.
    """
    settings, tie_tolerance = _controls(
        mode, priority, direction, bpm_tolerance, tie_tolerance)
    current_id = _identity(current_record)
    if isinstance(candidate_records, (str, bytes, Mapping)):
        raise ValueError("Candidates must be a sequence of track records")
    rows = []
    identities = set()
    for candidate in candidate_records:
        identity = _identity(candidate)
        if identity == current_id:
            raise ValueError("The current track must not be in the candidate pool")
        if identity in identities:
            raise ValueError("Duplicate candidate ids are not allowed")
        identities.add(identity)
        rows.append({"id": identity,
                     "score": score_pair(current_record, candidate, **settings)})
    rows.sort(key=lambda row: (row["score"]["cost"], row["id"]))
    if not rows:
        return rows
    best_cost = rows[0]["score"]["cost"]
    index = 0
    while index < len(rows):
        end = index + 1
        group_cost = rows[index]["score"]["cost"]
        while end < len(rows) and rows[end]["score"]["cost"] - group_cost <= tie_tolerance:
            end += 1
        for row in rows[index:end]:
            row.update(rank=index + 1, tie_count=end - index,
                       gap_from_best=max(0., row["score"]["cost"] - best_cost))
        index = end
    return rows


def _summary(edges):
    count = len(edges)
    best = sum(edge["is_best"] for edge in edges)
    top3 = sum(edge["in_top3"] for edge in edges)
    top5 = sum(edge["in_top5"] for edge in edges)
    strict_best = sum(edge["is_strict_best"] for edge in edges)
    top1_credit = sum(edge["fractional_top1_credit"] for edge in edges)
    guaranteed_top3 = sum(edge["guaranteed_in_top3"] for edge in edges)
    guaranteed_top5 = sum(edge["guaranteed_in_top5"] for edge in edges)
    return {
        "edge_count": count,
        "best_count": best, "top3_count": top3, "top5_count": top5,
        "best_fraction": best / count if count else None,
        "top3_fraction": top3 / count if count else None,
        "top5_fraction": top5 / count if count else None,
        "strict_best_count": strict_best,
        "strict_best_fraction": strict_best / count if count else None,
        "fractional_top1_credit": top1_credit,
        "fractional_top1_fraction": top1_credit / count if count else None,
        "guaranteed_top3_count": guaranteed_top3,
        "guaranteed_top3_fraction": guaranteed_top3 / count if count else None,
        "guaranteed_top5_count": guaranteed_top5,
        "guaranteed_top5_fraction": guaranteed_top5 / count if count else None,
        "mean_rank": sum(edge["actual_rank"] for edge in edges) / count if count else None,
        "mean_reciprocal_rank": sum(1 / edge["actual_rank"] for edge in edges) / count if count else None,
        "mean_midpoint_reciprocal_rank": sum(
            1 / (edge["actual_rank"] + (edge["actual_tie_count"] - 1) / 2)
            for edge in edges) / count if count else None,
        "mean_gap_from_best": sum(edge["actual_gap_from_best"] for edge in edges) / count if count else None,
        "mean_candidate_count": sum(edge["candidate_count"] for edge in edges) / count if count else None,
        "rank_semantics": "competition rank including ties; thresholds use rank, not display position",
    }


def evaluate_successors(records: Sequence[Mapping[str, Any]],
                        demonstrated_order: Sequence[str], *, mode="fuzzy",
                        priority=.5, direction="ascending", bpm_tolerance=None,
                        tie_tolerance=1e-9):
    """Audit every demonstrated successor using the actual already-played prefix.

    The demonstrated order must contain every input track once. One track is a
    valid zero-edge audit. The result separates the full playlist from steps
    with at least five remaining choices, since late forced choices otherwise
    inflate apparent agreement. ``top_candidates`` includes ranks up to five,
    retaining every tied candidate at that boundary.
    """
    settings, tie_tolerance = _controls(
        mode, priority, direction, bpm_tolerance, tie_tolerance)
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("Records must be a nonempty sequence of tracks")
    by_id = {}
    for record in records:
        identity = _identity(record)
        if identity in by_id:
            raise ValueError("Duplicate track ids are not allowed")
        by_id[identity] = record
    if not by_id:
        raise ValueError("At least one track is required")
    if isinstance(demonstrated_order, (str, bytes, Mapping)):
        raise ValueError("Demonstrated order must be a sequence of track ids")
    order = list(demonstrated_order)
    if any(not isinstance(identity, str) or not identity for identity in order):
        raise ValueError("Demonstrated order requires nonempty string ids")
    if len(order) != len(by_id) or len(set(order)) != len(order) or set(order) != set(by_id):
        raise ValueError("Demonstrated order must contain every input id exactly once")

    remaining = dict(by_id)
    edges = []
    for position, (current_id, actual_next) in enumerate(zip(order, order[1:]), 1):
        remaining.pop(current_id)
        ranked = rank_candidates(by_id[current_id], list(remaining.values()),
                                 tie_tolerance=tie_tolerance, **settings)
        actual = next(row for row in ranked if row["id"] == actual_next)
        rank_end = actual["rank"] + actual["tie_count"] - 1
        best_ids = [row["id"] for row in ranked if row["rank"] == 1]
        edges.append({
            "position": position, "current": current_id, "actual_next": actual_next,
            "candidate_count": len(ranked), "actual_rank": actual["rank"],
            "actual_rank_end": rank_end,
            "actual_tie_count": actual["tie_count"],
            "actual_gap_from_best": actual["gap_from_best"],
            "actual_score": actual["score"], "is_best": actual["rank"] == 1,
            "is_strict_best": actual["rank"] == 1 and actual["tie_count"] == 1,
            "fractional_top1_credit": 1 / actual["tie_count"] if actual["rank"] == 1 else 0.,
            "in_top3": actual["rank"] <= 3, "in_top5": actual["rank"] <= 5,
            "guaranteed_in_top3": rank_end <= 3,
            "guaranteed_in_top5": rank_end <= 5,
            "best_candidate_ids": best_ids,
            "top_candidates": [row for row in ranked if row["rank"] <= 5],
        })
    return {
        "schema_version": 1, "version": VERSION, "track_count": len(by_id),
        "demonstrated_order": order, "settings": {**settings, "tie_tolerance": tie_tolerance},
        "ranking_scope": "local BPM/key edge cost after the demonstrated prefix; not a global path recommendation",
        "tie_semantics": "competition rank; tolerance groups are anchored to their lowest cost",
        "edges": edges,
        "summary": {
            "all": _summary(edges),
            "at_least_five_candidates": _summary(
                [edge for edge in edges if edge["candidate_count"] >= 5]),
        },
    }
