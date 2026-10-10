"""Independent rendered-signal checks for adaptive transition policy.

Generated pulses exercise the real FFmpeg/Rubber Band executables. Measuring
their peaks in the output is separate from trusting the renderer's timemap.
These tests do not certify musical quality or predicted phrase boundaries.
"""

import copy
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

from superdj.planner import plan_tracks
from superdj.render import RenderError, render_plan


FFMPEG = os.getenv("SUPERDJ_FFMPEG") or shutil.which("ffmpeg")
RUBBERBAND = os.getenv("SUPERDJ_RUBBERBAND") or shutil.which("rubberband")


@unittest.skipUnless(FFMPEG and RUBBERBAND,
                     "Set SUPERDJ_FFMPEG and SUPERDJ_RUBBERBAND for signal tests")
class AdaptiveForwardSignalTests(unittest.TestCase):
    sample_rate = 16000

    def setUp(self):
        self.private = tempfile.TemporaryDirectory()
        self.addCleanup(self.private.cleanup)
        self.directory = Path(self.private.name)

    def fixtures(self, first_bpm, second_bpm):
        records = []
        sr = self.sample_rate
        for identity, bpm in (("first", first_bpm), ("second", second_bpm)):
            duration = 90.0
            beats = np.arange(0.25, duration - 0.05, 60 / bpm)
            # A known pulse every beat, with a small pitched background. Pulse
            # centres are ground truth supplied by this generated fixture.
            clock = np.arange(round(duration * sr)) / sr
            audio = (0.006 * np.sin(2 * math.pi * 440 * clock)).astype(np.float32)
            half_width = round(0.008 * sr)
            pulse = (0.25 * np.exp(-0.5 * (np.arange(-half_width, half_width + 1)
                                         / (0.0015 * sr)) ** 2)).astype(np.float32)
            for beat in beats:
                centre = round(beat * sr)
                audio[centre - half_width:centre + half_width + 1] += pulse
            path = self.directory / (identity + ".wav")
            sf.write(path, np.column_stack((audio, audio)), sr, subtype="FLOAT")
            records.append({"id": identity, "path": str(path), "duration": duration,
                            "bpm": bpm, "bpm_confidence": 1.0,
                            "key_camelot": "8A", "key_confidence": 1.0,
                            "energy": 0.5, "beats": beats.tolist(),
                            "downbeats": beats[::4].tolist(),
                            "grid_status": "reviewed", "grid_confidence": 1.0,
                            "grid": {"meter": 4, "human_reviewed": True,
                                     "source": "generated pulse fixture"}})
        return records

    def render(self, planned, records, **settings):
        return render_plan(planned, records, {}, self.directory / "set.wav",
                           self.directory / "work", ffmpeg=FFMPEG,
                           rubberband=RUBBERBAND, sample_rate=self.sample_rate,
                           transition_bars=8, **settings)

    def actual_pulses(self, path, expected_times):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        energy = np.abs(audio).max(axis=1)
        # A local energy envelope makes this insensitive to carrier phase.
        envelope = np.convolve(energy, np.ones(9) / 9, mode="same")
        positions = []
        for expected in expected_times:
            begin = max(0, round((expected - 0.040) * sr))
            end = min(len(envelope), round((expected + 0.040) * sr))
            self.assertGreater(end, begin)
            self.assertGreater(float(envelope[begin:end].max()), 0.015,
                               "Expected output pulse was lost")
            positions.append((begin + int(np.argmax(envelope[begin:end]))) / sr)
        return np.asarray(positions)

    def check_matched_output(self, result):
        transition = result["transitions"][0]
        self.assertTrue(transition["beatmatched"])
        self.assertEqual(transition["mode"], "beatmapped_bass_swap")
        self.assertGreater(transition["overlap_frames"], 0)
        first, second = result["tracks"]
        first_expected = [point["output_seconds"] for point in
                          first["warp"]["input_output_beat_mapping"]["tail"]][3:-3]
        second_expected = [point["output_seconds"] for point in
                           second["warp"]["input_output_beat_mapping"]["head"]][3:-3]
        first_actual = self.actual_pulses(first["audio_path"], first_expected)
        second_actual = self.actual_pulses(second["audio_path"], second_expected)
        first_error = np.abs(first_actual - np.asarray(first_expected))
        second_error = np.abs(second_actual - np.asarray(second_expected))
        self.assertLess(float(first_error.max()), 0.035)
        self.assertLess(float(second_error.max()), 0.035)
        phase_error = np.abs(first_actual + first["set_start_seconds"]
                             - second_actual - second["set_start_seconds"])
        self.assertLess(float(phase_error.max()), 0.035)
        mixed_expected = np.asarray(first_expected) + first["set_start_seconds"]
        mixed_actual = self.actual_pulses(result["output_path"], mixed_expected)
        self.assertLess(float(np.max(np.abs(mixed_actual - mixed_expected))), 0.035)
        expected_period = 60 / transition["target_bpm"]
        for observed in (first_actual, second_actual, mixed_actual):
            self.assertLess(float(np.max(np.abs(np.diff(observed) - expected_period))), 0.035)
        for track in result["tracks"]:
            ramps = track["warp"]["tempo_ramps"]
            self.assertTrue(ramps)
            for ramp in ramps:
                self.assertAlmostEqual(ramp["output_duration_seconds"], 15.0, places=6)
                self.assertFalse(ramp["shortened_for_available_body"])

    def test_normal_blend_has_measured_pulses_and_ordinary_limit(self):
        records = self.fixtures(120.0, 122.0)
        planned = plan_tracks(records, mode="fixed")
        result = self.render(planned, records)
        self.check_matched_output(result)
        self.assertEqual(result["transitions"][0]["tempo_technique"], "standard_blend")
        self.assertEqual(result["transitions"][0]["musical_tempo_limit"], 0.06)

    def test_adaptive_blend_and_legacy_plan_keep_order_locks_and_sources(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                if legacy:
                    self.directory = self.directory / "legacy"
                    self.directory.mkdir()
                records = self.fixtures(174.0, 128.0)
                planned = plan_tracks(records, mode="fixed",
                                      tempo_ramp_max_change=None if legacy else 0.25)
                if legacy:
                    # Plans predating the adaptive feature have no ramp cap;
                    # an explicit modern null cap instead disables adaptation.
                    planned["constraints"].pop("tempo_ramp_max_change_per_deck", None)
                outgoing_end = records[0]["downbeats"][-1]
                incoming_start = records[1]["downbeats"][8]
                planned["edges"][0]["cue_locks"] = {
                    "outgoing_end_seconds": outgoing_end,
                    "incoming_start_seconds": incoming_start,
                }
                frozen = copy.deepcopy((records, planned))
                hashes = {r["path"]: hashlib.sha256(Path(r["path"]).read_bytes()).digest()
                          for r in records}
                result = self.render(planned, records)
                self.check_matched_output(result)
                self.assertEqual(result["order"], ["first", "second"])
                self.assertEqual(result["transitions"][0]["tempo_technique"], "tempo_ramp")
                self.assertEqual(result["transitions"][0]["musical_tempo_limit"], 0.25)
                self.assertAlmostEqual(result["tracks"][0]["trim_end_seconds"], outgoing_end,
                                       delta=1 / self.sample_rate)
                self.assertAlmostEqual(result["tracks"][1]["trim_start_seconds"], incoming_start,
                                       delta=1 / self.sample_rate)
                self.assertEqual((records, planned), frozen)
                for path, original_hash in hashes.items():
                    self.assertEqual(hashlib.sha256(Path(path).read_bytes()).digest(), original_hash)

    def test_unreachable_tempo_uses_positive_echo_without_warp(self):
        records = self.fixtures(100.0, 180.0)
        result = self.render(plan_tracks(records, mode="fixed"), records)
        self.assertEqual(result["transitions"][0]["mode"], "free_time_echo_bridge")
        self.assertGreater(result["transitions"][0]["overlap_frames"], 0)
        self.assertFalse(result["transitions"][0]["beatmatched"])
        self.assertTrue(all(track["warp"] is None for track in result["tracks"]))

    def test_deficient_grid_rejected_without_automatic_cut(self):
        records = self.fixtures(174.0, 128.0)
        records[1]["grid_status"] = "unverified"
        with self.assertRaises(RenderError):
            self.render(plan_tracks(records, mode="fixed"), records)
        self.assertFalse((self.directory / "set.wav").exists())

    def test_offgrid_mandatory_cue_rejected_instead_of_relocated(self):
        records = self.fixtures(174.0, 128.0)
        planned = plan_tracks(records, mode="fixed")
        planned["edges"][0]["cue_locks"] = {
            "incoming_start_seconds": records[1]["downbeats"][8] + 0.020,
        }
        with self.assertRaisesRegex(RenderError, "instead of moving"):
            self.render(planned, records)
        self.assertFalse((self.directory / "set.wav").exists())

    def test_strict_native_fallback_constraint_rejected(self):
        records = self.fixtures(100.0, 180.0)
        planned = plan_tracks(records, mode="fixed")
        planned["constraints"]["allow_tempo_reset"] = False
        with self.assertRaisesRegex(RenderError, "forbids"):
            self.render(planned, records)
        self.assertFalse((self.directory / "set.wav").exists())

    def test_stricter_declared_ramp_cap_is_not_silently_expanded(self):
        records = self.fixtures(174.0, 128.0)
        planned = plan_tracks(records, mode="fixed", tempo_ramp_max_change=0.10)
        result = self.render(planned, records)
        transition = result["transitions"][0]
        self.assertFalse(transition["beatmatched"])
        self.assertEqual(transition["mode"], "free_time_echo_bridge")
        self.assertTrue(all(track["warp"] is None for track in result["tracks"]))


if __name__ == "__main__":
    unittest.main()
