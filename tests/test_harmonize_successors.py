"""Next-track ranking uses the demonstrated prefix and preserves ambiguity."""
import json

import pytest

from superdj.harmonize import score_pair
from superdj.harmonize_successors import evaluate_successors, rank_candidates


def track(identity, bpm=120, key="8A"):
    return dict(id=identity, bpm=bpm, key_camelot=key)


def test_competition_rank_keeps_ties_and_pair_evidence():
    current = track("current")
    rows = rank_candidates(current, [track("z"), track("a"), track("other", 130)])
    assert [row["id"] for row in rows] == ["a", "z", "other"]
    assert [row["rank"] for row in rows] == [1, 1, 3]
    assert [row["tie_count"] for row in rows] == [2, 2, 1]
    assert rows[0]["gap_from_best"] == 0
    assert rows[2]["gap_from_best"] == pytest.approx(1)
    assert rows[2]["score"] == score_pair(current, track("other", 130))


def test_tolerance_groups_do_not_chain_across_more_than_tolerance():
    # Costs are 0, .0006, .0012; pairwise chaining would incorrectly tie all 3.
    rows = rank_candidates(track("current"),
                           [track("a"), track("b", 120.006), track("c", 120.012)],
                           tie_tolerance=.001)
    assert [row["rank"] for row in rows] == [1, 1, 3]
    assert rows[1]["gap_from_best"] > 0
    assert rows[2]["tie_count"] == 1


def test_remaining_pool_excludes_every_previously_played_track():
    records = [track("a", 120), track("b", 130), track("c", 121), track("d", 132)]
    audit = evaluate_successors(records, ["a", "c", "b", "d"])
    assert [edge["candidate_count"] for edge in audit["edges"]] == [3, 2, 1]
    assert audit["edges"][0]["actual_next"] == "c"
    assert audit["edges"][0]["actual_rank"] == 1
    assert audit["edges"][1]["best_candidate_ids"] == ["b"]
    assert {row["id"] for row in audit["edges"][1]["top_candidates"]} == {"b", "d"}
    assert audit["edges"][2]["actual_rank"] == 1
    assert audit["summary"]["at_least_five_candidates"]["edge_count"] == 0
    assert audit["summary"]["at_least_five_candidates"]["best_fraction"] is None


def test_nontrivial_summary_does_not_use_trivial_last_choices():
    records = [track(str(i), 120 + i) for i in range(7)]
    audit = evaluate_successors(records, ["0", "6", "1", "2", "3", "4", "5"])
    summary = audit["summary"]
    assert summary["all"]["edge_count"] == 6
    assert summary["at_least_five_candidates"]["edge_count"] == 2
    assert summary["all"]["best_count"] == 4
    assert summary["at_least_five_candidates"]["best_count"] == 0
    assert audit["edges"][0]["actual_rank"] == 6
    assert not audit["edges"][0]["in_top5"]
    assert summary["at_least_five_candidates"]["mean_rank"] == pytest.approx(5.5)


def test_display_boundary_does_not_drop_equally_ranked_candidates():
    records = [track(str(i)) for i in range(8)]
    audit = evaluate_successors(records, [str(i) for i in range(8)])
    assert audit["edges"][0]["actual_tie_count"] == 7
    assert audit["edges"][0]["actual_rank_end"] == 7
    assert not audit["edges"][0]["is_strict_best"]
    assert audit["edges"][0]["fractional_top1_credit"] == pytest.approx(1/7)
    assert not audit["edges"][0]["guaranteed_in_top3"]
    assert not audit["edges"][0]["guaranteed_in_top5"]
    assert len(audit["edges"][0]["top_candidates"]) == 7
    assert audit["summary"]["all"]["best_count"] == 7
    assert audit["summary"]["all"]["strict_best_count"] == 1
    assert audit["summary"]["all"]["fractional_top1_credit"] == pytest.approx(sum(1/i for i in range(1, 8)))
    assert audit["summary"]["all"]["guaranteed_top3_count"] == 3
    assert audit["summary"]["all"]["guaranteed_top5_count"] == 5
    assert audit["summary"]["all"]["mean_midpoint_reciprocal_rank"] == pytest.approx(
        sum(1 / (1 + (i-1)/2) for i in range(1, 8)) / 7)


def test_guaranteed_topk_uses_end_of_tie_group():
    records = [track("a", 120), track("b", 121), track("c", 122),
               track("d", 123), track("e", 123), track("f", 123)]
    audit = evaluate_successors(records, ["a", "d", "b", "c", "e", "f"])
    edge = audit["edges"][0]
    assert edge["actual_rank"] == 3
    assert edge["actual_rank_end"] == 5
    assert edge["in_top3"]
    assert not edge["guaranteed_in_top3"]
    assert edge["guaranteed_in_top5"]


def test_controls_are_applied_and_unknown_metadata_remains_unknown():
    current = track("current", 120)
    candidates = [track("key", 130), track("tempo", 121, "2B")]
    assert rank_candidates(current, candidates, priority=1)[0]["id"] == "key"
    assert rank_candidates(current, candidates, priority=0)[0]["id"] == "tempo"
    rows = rank_candidates({"id": "a"}, [{"id": "b"}])
    assert rows[0]["score"]["harmonic"]["relation"] == "unknown"
    assert rows[0]["score"]["tempo"]["delta_bpm"] is None
    json.dumps(rows, allow_nan=False)


def test_one_track_and_empty_candidate_pool_are_valid():
    assert rank_candidates(track("only"), []) == []
    audit = evaluate_successors([track("only")], ["only"])
    assert audit["edges"] == []
    assert audit["summary"]["all"]["edge_count"] == 0
    assert audit["summary"]["all"]["best_fraction"] is None


@pytest.mark.parametrize("records,order", [
    ([], []), ([{}], ["a"]), ([track("a"), track("a")], ["a", "a"]),
    ([track("a"), track("b")], ["a"]),
    ([track("a"), track("b")], ["a", "a"]),
    ([track("a"), track("b")], ["a", "missing"]),
    ([track("a")], "a"), ([track("a")], [None]),
])
def test_invalid_libraries_and_orders_fail(records, order):
    with pytest.raises(ValueError):
        evaluate_successors(records, order)


@pytest.mark.parametrize("candidates", [[{}], [track("current")], [track("a"), track("a")], "abc"])
def test_invalid_candidate_pools_fail(candidates):
    with pytest.raises(ValueError):
        rank_candidates(track("current"), candidates)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf"), True, "oops"])
def test_invalid_tie_tolerance_fails_even_with_no_edges(tolerance):
    with pytest.raises(ValueError):
        evaluate_successors([track("only")], ["only"], tie_tolerance=tolerance)


def test_invalid_scoring_controls_fail_even_with_no_candidates():
    with pytest.raises(ValueError):
        rank_candidates(track("only"), [], priority=2)


def test_path_only_identity_uses_existing_harmonize_convention():
    audit = evaluate_successors([{"path": "a.flac"}, {"path": "b.flac"}],
                                ["a.flac", "b.flac"])
    assert audit["edges"][0]["actual_next"] == "b.flac"
