"""Behavioral planner tests use only synthetic musical features."""

import itertools
import json
import unittest

from superdj.planner import plan_tracks, transition_score


def track(identity, bpm=124.0, key="8A", energy=0.5, confidence=1.0, **extra):
    return {"id": identity, "bpm": bpm, "key_camelot": key, "energy": energy,
            "key_confidence": confidence, "bpm_confidence": 1.0, **extra}


class PlannerTests(unittest.TestCase):
    def test_fixed_order_is_preserved_even_when_costlier(self):
        records = [track("a", 124), track("b", 126), track("c", 120), track("d", 128)]
        expected = ["a", "b", "c", "d"]
        result = plan_tracks(records, mode="fixed", first_id="a", last_id="d")
        self.assertEqual(result["order"], expected)
        automatic = plan_tracks(records, first_id="a", last_id="d")
        self.assertGreater(result["total_cost"], automatic["total_cost"])
        self.assertIsNone(result["optimal_for_declared_objective"])

    def test_fixed_id_list_and_lock_conflict(self):
        records = [track("a"), track("b"), track("c")]
        result = plan_tracks(records, mode="fixed", fixed_order=["c", "b", "a"])
        self.assertEqual(result["order"], ["c", "b", "a"])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            plan_tracks(records, mode="fixed", first_id="c")
        with self.assertRaisesRegex(ValueError, "exactly once"):
            plan_tracks(records, mode="fixed", fixed_order=["a", "a", "c"])

    def test_global_path_beats_nearest_greedy_with_same_constraints(self):
        # Nearest from 124 chooses 126; leaving 120 until next creates a
        # needless detour before the locked 128 ending. Global search sees it.
        records = [track("a", 124), track("b", 126), track("c", 120), track("d", 128)]
        result = plan_tracks(records, first_id="a", last_id="d", energy_arc="none")
        self.assertEqual(result["order"], ["a", "c", "b", "d"])
        self.assertEqual(result["baseline_order"], ["a", "b", "c", "d"])
        self.assertGreater(result["improvement"], 0.1)
        self.assertTrue(result["optimal_for_declared_objective"])
        self.assertEqual(result["edges"][0]["selected_immediate_rank"], 2)

    def test_exact_matches_exhaustive_directional_position_objective(self):
        records = [track("a", 120, "8A", .1), track("b", 125, "8B", .9),
                   track("c", 128, "9A", .5), track("d", 122, "1A", .2),
                   track("e", 126, "10A", .7)]
        result = plan_tracks(records, first_id="a", last_id="e", energy_arc="wave")
        all_costs = []
        for middle in itertools.permutations(["b", "c", "d"]):
            candidate = plan_tracks(records, mode="fixed", fixed_order=["a", *middle, "e"],
                                    first_id="a", last_id="e", energy_arc="wave")
            all_costs.append(candidate["total_cost"])
        self.assertAlmostEqual(result["total_cost"], min(all_costs), places=6)

    def test_uncertain_perfect_key_is_not_rewarded_as_safe_match(self):
        known = transition_score(track("a"), track("b"))
        uncertain = transition_score(track("a"), track("b", confidence=.05))
        self.assertGreater(uncertain["cost"], known["cost"])
        self.assertTrue(any("uncertain" in reason for reason in uncertain["reasons"]))

    def test_missing_analysis_is_reported_not_invented(self):
        records = [{"id": "unknown"}, track("known")]
        result = plan_tracks(records, mode="fixed")
        self.assertEqual(result["tempo_reset_count"], 1)
        self.assertIn("unknown", result["uncertainty"]["low_confidence_tracks"])
        self.assertTrue(any("BPM unknown" in warning for warning in result["uncertainty"]["warnings"]))
        json.dumps(result, allow_nan=False)

    def test_stretch_cap_uses_intersection_not_midpoint_approximation(self):
        # sqrt(120*135)/120 is just over 1.06, but feasible common tempos
        # do exist in [126.9, 127.2], so clipping into that interval matters.
        edge = transition_score(track("a", 120), track("b", 135))
        self.assertTrue(edge["tempo"]["beatmatch_possible"])
        self.assertLessEqual(edge["tempo"]["max_change"], .06)
        self.assertGreaterEqual(edge["tempo"]["outgoing_rate"], .94)
        self.assertLessEqual(edge["tempo"]["outgoing_rate"], 1.06)
        self.assertGreaterEqual(edge["tempo"]["incoming_rate"], .94)
        self.assertLessEqual(edge["tempo"]["incoming_rate"], 1.06)

    def test_impossible_tempo_becomes_explicit_reset_or_error(self):
        records = [track("a", 90), track("b", 150)]
        result = plan_tracks(records, mode="fixed")
        tempo = result["edges"][0]["tempo"]
        self.assertEqual(result["tempo_reset_count"], 1)
        self.assertIsNone(tempo["target_bpm"])
        self.assertEqual(result["render_readiness"], "needs_transition_review")
        self.assertTrue(result["edges"][0]["transition_review_required"])
        self.assertEqual(tempo["outgoing_rate"], 1.0)
        self.assertEqual(tempo["incoming_rate"], 1.0)
        with self.assertRaisesRegex(ValueError, "No complete path"):
            plan_tracks(records, allow_tempo_reset=False)
        with self.assertRaisesRegex(ValueError, "strict tempo"):
            plan_tracks(records, mode="fixed", allow_tempo_reset=False)

    def test_seed_reproducible_and_no_duplicate_records(self):
        records = [track(str(i)) for i in range(7)]
        first = plan_tracks(records, seed=27)
        second = plan_tracks(records, seed=27)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first["order"])), len(records))
        with self.assertRaisesRegex(ValueError, "Duplicate track ids"):
            plan_tracks([track("a"), track("a")])
        with self.assertRaisesRegex(ValueError, "Duplicate source paths"):
            plan_tracks([track("a", path="same.wav"), track("b", path="same.wav")])

    def test_beam_respects_endpoints_and_retains_feasible_baseline(self):
        records = [track(str(i), 120 + i * .5, energy=i / 19) for i in range(20)]
        result = plan_tracks(records, first_id="0", last_id="19", exact_limit=5, beam_width=16)
        self.assertEqual(result["order"][0], "0")
        self.assertEqual(result["order"][-1], "19")
        self.assertEqual(len(set(result["order"])), 20)
        self.assertGreaterEqual(result["improvement"], 0)
        self.assertFalse(result["optimal_for_declared_objective"])

    def test_similarity_and_vocal_boundaries_change_pair_costs(self):
        a = track("a", embedding=[1, 0], boundary_features={"outro_vocal": .9})
        similar = track("b", embedding=[1, 0])
        opposing = track("c", embedding=[-1, 0], boundary_features={"intro_vocal": .9})
        good = transition_score(a, similar)
        poor = transition_score(a, opposing)
        self.assertGreater(poor["components"]["similarity"], good["components"]["similarity"])
        self.assertGreater(poor["components"]["vocal_overlap"], good["components"]["vocal_overlap"])

    def test_invalid_features_and_single_track(self):
        self.assertEqual(plan_tracks([track("solo")], first_id="solo", last_id="solo")["order"], ["solo"])
        with self.assertRaises(ValueError):
            plan_tracks([track("a", bpm=float("nan"))])
        with self.assertRaises(ValueError):
            plan_tracks([track("a", energy=1.3)])
        with self.assertRaises(ValueError):
            plan_tracks([track("a")], weights={"imaginary": 1})


if __name__ == "__main__":
    unittest.main()
