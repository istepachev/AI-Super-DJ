"""Behavioral sequencing checks, independent exhaustive objective and locks."""
import itertools
import json

import pytest

from superdj.harmonize import (
    _relocate_blocks, harmonic_relation, harmonize_tracks, score_pair,
    summarize_order,
)


def track(identity, bpm=120, key="8A", **extra):
    return dict(id=identity, bpm=bpm, key_camelot=key, **extra)


def test_wrapped_fifths_relative_and_risky_directional_moves():
    assert harmonic_relation("12A", "1A")["relation"] == "perfect_boost"
    assert harmonic_relation("1A", "12A")["relation"] == "perfect_drop"
    assert harmonic_relation("8A", "8B")["relation"] == "relative_major_minor"
    assert harmonic_relation("8A", "9B")["relation"] == "diagonal"
    assert harmonic_relation("8A", "10A")["accepted"] is False
    assert harmonic_relation("8A", "10A", mode="mood")["relation"] == "energy_boost"
    assert harmonic_relation("10A", "8A", mode="mood")["accepted"] is False
    assert harmonic_relation("8A", "11B", mode="mood")["relation"] == "parallel_major_minor"
    assert harmonic_relation("11B", "8A", mode="mood")["relation"] == "parallel_major_minor"
    assert harmonic_relation("8A", "3A", mode="mood")["risk"] == "high"


def test_mood_diagonal_is_directional_and_fuzzy_is_wider():
    # The displayed Mood wheel for C major selects E minor, not D minor.
    assert harmonic_relation("8B", "9A", mode="mood")["relation"] == "diagonal"
    assert harmonic_relation("9A", "8B", mode="mood")["relation"] == "diagonal"
    assert harmonic_relation("8A", "7B", mode="mood")["relation"] == "diagonal"
    assert harmonic_relation("8B", "7A", mode="mood")["accepted"] is False
    assert harmonic_relation("8B", "7A", mode="fuzzy")["accepted"] is True
    assert harmonic_relation("8A", "9B", mode="mood")["accepted"] is False
    assert harmonic_relation("8A", "9B", mode="fuzzy")["accepted"] is True
    # Directional expressive moves are not accepted in reverse by accident.
    assert harmonic_relation("8B", "10B", mode="mood")["relation"] == "energy_boost"
    assert harmonic_relation("8B", "5A", mode="mood")["relation"] == "parallel_major_minor"
    assert harmonic_relation("8B", "3B", mode="mood")["relation"] == "semitone_lift"


@pytest.mark.parametrize("source", [f"{n}{letter}" for letter in "AB" for n in range(1, 13)])
def test_every_key_has_six_fuzzy_and_eight_mood_destinations(source):
    keys = [f"{n}{letter}" for letter in "AB" for n in range(1, 13)]
    fuzzy = {key for key in keys if harmonic_relation(source, key, mode="fuzzy")["accepted"]}
    mood = {key for key in keys if harmonic_relation(source, key, mode="mood")["accepted"]}
    assert len(fuzzy) == 6
    assert len(mood) == 8
    assert len(fuzzy & mood) == 5
    assert len(fuzzy - mood) == 1
    assert len(mood - fuzzy) == 3


def test_direction_and_tolerance_are_explicit_soft_costs_not_stretch_limits():
    a, b = track("a", 130), track("b", 120)
    up = score_pair(a, b, direction="ascending", bpm_tolerance=4)
    down = score_pair(a, b, direction="descending", bpm_tolerance=4)
    assert up["cost"] > down["cost"]
    assert up["tempo"]["tolerance_excess_bpm"] == 6
    assert up["tempo"]["direction_reversal_bpm"] == 10
    assert up["feasible"]


def test_bpm_priority_changes_tradeoff_and_ignores_nonobjective_features():
    a = track("a", 120)
    good_key = track("b", 140)
    good_bpm = track("c", 121, "2B")
    assert score_pair(a, good_key, priority=1)["cost"] < score_pair(a, good_bpm, priority=1)["cost"]
    assert score_pair(a, good_key, priority=0)["cost"] > score_pair(a, good_bpm, priority=0)["cost"]
    plain = score_pair(a, good_key)
    changed = score_pair({**a, "energy": 0, "key_confidence": .01, "boundary_features": {"outro_vocal": 1}},
                         {**good_key, "energy": 1, "key_confidence": 0})
    assert plain["cost"] == changed["cost"]


def test_exact_result_matches_exhaustive_cost_with_directed_rules():
    records = [track("a", 120, "8A"), track("b", 123, "10A"),
               track("c", 119, "8B"), track("d", 125, "11B"), track("e", 124, "9A")]
    settings = dict(mode="mood", priority=.65, direction="ascending", bpm_tolerance=4)
    result = harmonize_tracks(records, first_id="a", last_id="e", order_objective="weighted", **settings)
    minimum = min(summarize_order(records, ["a", *middle, "e"], **settings)["total_cost"]
                  for middle in itertools.permutations(["b", "c", "d"]))
    assert result["total_cost"] == pytest.approx(minimum)
    assert result["optimal_for_declared_objective"] is True


def test_chain_and_endpoints_survive_beam_and_all_tracks_retained():
    records = [track(str(i), 110+i, f"{i%12+1}A") for i in range(25)]
    result = harmonize_tracks(records, first_id="2", last_id="23",
                             locked_chains=[["2", "3"], ["7", "8", "9"], ["22", "23"]],
                             beam_width=30, order_objective="weighted")
    assert result["order"][:2] == ["2", "3"]
    assert result["order"][-2:] == ["22", "23"]
    middle = result["order"].index("7")
    assert result["order"][middle:middle+3] == ["7", "8", "9"]
    assert len(set(result["order"])) == 25
    assert result["total_cost"] <= result["baseline_cost"]+1e-9
    assert not result["optimal_for_declared_objective"]
    by_id = {record["id"]: record for record in records}
    independent_cost = sum(score_pair(by_id[a], by_id[b])["cost"]
                           for a, b in zip(result["order"], result["order"][1:]))
    assert result["total_cost"] == pytest.approx(independent_cost)


def test_relocation_recovers_directed_move_missing_from_segment_reversals():
    records = [track(str(i), bpm, key) for i, (bpm, key) in enumerate([
        (114, "7A"), (121, "8B"), (106, "2B"), (138, "4B"),
        (132, "4B"), (108, "1B"), (147, "3B"), (107, "4A"),
        (105, "9A"), (138, "6A"), (147, "4A"), (121, "6B"),
        (133, "5A"), (147, "7B"), (130, "3B"), (142, "12A"),
        (140, "3B"), (106, "7B"),
    ])]
    # The previous beam plus segment-reversal search stops at 22.00 for this
    # fixture. A directed relocation reaches at least 20.55 without any new
    # relation weights, identity features or change to the fixed first track.
    result = harmonize_tracks(records, first_id="0", beam_width=512, order_objective="weighted")
    assert result["order"][0] == "0"
    assert set(result["order"]) == {r["id"] for r in records}
    assert len(result["order"]) == len(records)
    assert result["total_cost"] <= 20.55 + 1e-9
    assert result["version"] == "superdj-harmonize-2"
    assert "relocation" in result["method"]
    by_id = {record["id"]: record for record in records}
    independent_cost = sum(score_pair(by_id[a], by_id[b])["cost"]
                           for a, b in zip(result["order"], result["order"][1:]))
    assert result["total_cost"] == pytest.approx(independent_cost)


def test_relocation_respects_locked_ends_and_full_asymmetric_cost():
    costs = [[9.] * 5 for _ in range(5)]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        costs[a][b] = 3.
    for a, b in [(0, 2), (3, 1), (1, 4)]:
        costs[a][b] = 1.
    order = [0, 1, 2, 3, 4]
    chosen = _relocate_blocks(order, costs, 0, 4)
    assert chosen == [0, 2, 3, 1, 4]
    assert sum(costs[a][b] for a, b in zip(chosen, chosen[1:])) == 6.
    assert _relocate_blocks(chosen, costs, 0, 4) == chosen
    assert order == [0, 1, 2, 3, 4]
    assert _relocate_blocks(order, costs, 0, 4, passes=0) == order


def test_seed_is_reproducible_and_unknown_is_not_invented():
    records = [{"id": "unknown"}, track("b"), track("c", 122)]
    result = harmonize_tracks(records, seed=123)
    assert result["order"] == harmonize_tracks(records, seed=123)["order"]
    assert result["uncertainty"]["warnings"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs", [dict(priority=2), dict(direction="up"), dict(bpm_tolerance=0),
                                     dict(first_id="missing"), dict(locked_chains=[["a", "b"], ["b", "c"]]),
                                     dict(first_id="b", locked_chains=[["a", "b"]])])
def test_invalid_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        harmonize_tracks([track("a"), track("b"), track("c")], **kwargs)
