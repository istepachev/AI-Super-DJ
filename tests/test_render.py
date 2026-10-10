"""Timing and signal invariants, including an optional real Rubber Band test."""

import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

from superdj.render import RenderError, _Window, _choose_window, _crossfade, _grid, _local_validation_possible, _warp_anchors, render_plan


def record(identity, path="unused.wav", bpm=120, seconds=90, first=0.25):
    beats = np.arange(first, seconds - 0.1, 60 / bpm)
    return {"id": identity, "path": str(path), "duration": seconds, "bpm": bpm,
            "beats": beats.tolist(), "downbeats": beats[::4].tolist(),
            "grid_status": "reviewed", "grid_confidence": 1.0,
            "grid": {"meter": 4, "human_reviewed": True}}


def plan(a, b, target=122):
    return {"order": [a, b], "edges": [{"from": a, "to": b,
            "tempo": {"beatmatch_possible": True, "target_bpm": target},
            "tempo_reset": False, "harmonic_relation": "same estimated key"}]}


class TimingTests(unittest.TestCase):
    def test_unresolved_tempo_reset_refused_before_files_or_backend_calls(self):
        planned = plan("a", "b")
        planned["edges"][0]["tempo_reset"] = True
        planned["edges"][0]["tempo"]["beatmatch_possible"] = False
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaisesRegex(RenderError, "automatic hard cuts are disabled"):
                render_plan(planned, [record("a", bpm=174), record("b", bpm=128)], {},
                            directory / "set.wav", directory / "work",
                            ffmpeg="must-not-run", rubberband="must-not-run")
            self.assertFalse((directory / "set.wav").exists())
            self.assertFalse((directory / "work").exists())

    def test_variable_native_tempos_share_exact_overlap_anchors(self):
        sr = 8000
        track = record("a")
        times = np.asarray(track["beats"])
        head = _Window(times[:65], 16, 4, 122, round(64 * 60 / 122 * sr))
        tail_times = times[112:177]
        tail = _Window(tail_times, 16, 4, 118, round(64 * 60 / 118 * sr))
        start, end = times[0], tail_times[-1]
        frames = round((end - start) * sr)
        result = _warp_anchors(track, trim_start=start, trim_end=end, source_frames=frames,
                               sample_rate=sr, head=head, tail=tail,
                               max_tempo_change=0.06, max_local_warp_change=0.15)
        for role, window in (("head", head), ("tail", tail)):
            mapped = result["input_output_beat_mapping"][role]
            self.assertEqual(len(mapped), 65)
            self.assertAlmostEqual(mapped[-1]["output_seconds"] - mapped[0]["output_seconds"],
                                   window.frames / sr, places=10)
            gaps = np.diff([point["output_seconds"] for point in mapped])
            np.testing.assert_allclose(gaps, window.frames / sr / 64, rtol=0, atol=1e-12)
        self.assertLessEqual(result["max_musical_rate_change"], 0.06)
        anchors = result["anchors"]
        self.assertEqual(anchors[0]["source_frame"], 0)
        self.assertEqual(anchors[-1]["source_frame"], frames)
        self.assertTrue(all(b["source_frame"] > a["source_frame"] and
                            b["output_frame"] > a["output_frame"]
                            for a, b in zip(anchors, anchors[1:])))

    def test_reject_large_microtiming_correction(self):
        sr = 8000
        track = record("a")
        times = np.asarray(track["beats"][:65])
        times[20] += 0.2
        head = _Window(times, 16, 4, 120, 64 * sr // 2)
        with self.assertRaisesRegex(RenderError, "Local timing correction"):
            _warp_anchors(track, trim_start=times[0], trim_end=90,
                          source_frames=round((90 - times[0]) * sr), sample_rate=sr,
                          head=head, tail=None, max_tempo_change=0.06,
                          max_local_warp_change=0.15)

    def test_identical_low_band_is_not_doubled(self):
        sr = 8000
        t = np.arange(sr * 3) / sr
        mono = (0.2 * np.sin(2 * math.pi * 40 * t)).astype(np.float32)
        audio = np.column_stack((mono, mono))
        mixed = _crossfade(audio, audio, sr, bass_hz=180)
        centre = mixed[sr:sr * 2]
        original = audio[sr:sr * 2]
        self.assertLess(float(np.max(np.abs(centre))), 0.203)
        self.assertLess(float(np.sqrt(np.mean((centre - original) ** 2))), 0.001)

    def test_missing_downbeats_are_not_fabricated(self):
        a, b = record("a"), record("b")
        a["downbeats"] = []
        a["grid_status"] = "estimated_beats_only"
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RenderError, "no measured/predicted downbeats"):
                render_plan(plan("a", "b"), [a, b], {}, Path(temp) / "set.wav", temp,
                            ffmpeg="not-invoked", rubberband="not-invoked")

    def test_search_avoids_unstable_first_and_last_windows(self):
        head_record = record("head")
        head_record["beats"][10] += 0.20
        head_record["cue_candidates"] = [{"time": 6.25, "kind": "intro", "confidence": 0.9,
                                           "novelty": 0.8, "sustained_midband_proxy": 0.1}]
        head = _choose_window(head_record, _grid(head_record, 0.75), region="head", count=16,
                              target_bpm=120, sample_rate=8000, max_trim_seconds=12,
                              max_local_warp_change=0.15)
        self.assertAlmostEqual(head.times[0], 6.25)
        self.assertAlmostEqual(head.cue_provenance["cue_confidence"], 0.9)
        tail_record = record("tail")
        tail_record["beats"][170] += 0.20
        tail = _choose_window(tail_record, _grid(tail_record, 0.75), region="tail", count=16,
                              target_bpm=120, sample_rate=8000, max_trim_seconds=45,
                              max_local_warp_change=0.15)
        self.assertLess(tail.times[-1], tail_record["beats"][170])
        self.assertEqual(len(tail.times), 65)

    def test_mild_quantised_jitter_keeps_a_valid_window(self):
        item = record("a")
        # 20 ms model quantisation may alternate around a stable true tempo.
        for index in range(1, len(item["beats"]), 4):
            item["beats"][index] += 0.02
        head = _choose_window(item, _grid(item, 0.75), region="head", count=16,
                              target_bpm=122, sample_rate=8000, max_trim_seconds=12,
                              max_local_warp_change=0.15)
        self.assertEqual(len(head.times), 65)

    def test_imported_cue_accepts_absent_neural_probability(self):
        item = record("a")
        item["cue_candidates"] = [{"time": 0.25, "kind": "intro", "confidence": 1.0,
                                    "downbeat_model_probability": None,
                                    "sustained_midband_proxy": None}]
        window = _choose_window(item, _grid(item, 0.75), region="head", count=16,
                                target_bpm=120, sample_rate=8000, max_trim_seconds=12,
                                max_local_warp_change=0.15)
        self.assertEqual(window.cue_provenance["downbeat_model_probability"], 0)

    def test_low_whole_confidence_needs_strong_local_neural_evidence(self):
        item = record("a")
        item["grid_confidence"] = 0.60
        item["grid_status"] = "model_predicted"
        count = len(item["beats"])
        item["grid"].update({"metrical_decode": {"accepted": True, "phase_contrast": 0.8},
                            "neural_beat_probability_by_beat": [0.95] * count,
                            "neural_downbeat_probability_by_beat": [0.90 if i % 4 == 0 else 0.05
                                                                       for i in range(count)]})
        grid = _grid(item, 0.75)
        self.assertFalse(grid[0])
        self.assertTrue(_local_validation_possible(item, grid))
        window = _choose_window(item, grid, region="head", count=16, target_bpm=120,
                                sample_rate=8000, max_trim_seconds=12, max_local_warp_change=0.15)
        evidence = window.cue_provenance["local_validation"]
        self.assertGreater(evidence["local_timing_confidence"], 0.75)
        self.assertEqual(evidence["whole_track_confidence"], 0.60)
        item["grid"]["neural_beat_probability_by_beat"] = [0.1] * count
        with self.assertRaisesRegex(RenderError, "No stable"):
            _choose_window(item, grid, region="head", count=16, target_bpm=120,
                           sample_rate=8000, max_trim_seconds=12, max_local_warp_change=0.15)

    def test_duplicate_or_missing_tracks_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RenderError, "exactly once"):
                render_plan({"order": ["a", "a"]}, [record("a"), record("b")], {},
                            Path(temp) / "set.wav", temp)


class ExternalTimingTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("SUPERDJ_FFMPEG") or shutil.which("ffmpeg"),
                         "Set SUPERDJ_FFMPEG for the real native tempo-reset test")
    def test_strong_local_neural_grid_allows_native_bar_reset(self):
        sr = 8000
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = []
            for identity, bpm in (("a", 120), ("b", 124)):
                path = directory / (identity + ".wav")
                item = record(identity, path, bpm)
                audio = np.zeros((90 * sr, 2), dtype=np.float32)
                audio[:, :] = (0.1 * np.sin(2 * math.pi * 220 * np.arange(90 * sr) / sr))[:, None]
                sf.write(path, audio, sr, subtype="FLOAT")
                records.append(item)
            low = records[0]
            low["grid_confidence"], low["grid_status"] = 0.60, "model_predicted"
            count = len(low["beats"])
            low["grid"].update({"metrical_decode": {"accepted": True, "phase_contrast": 0.8},
                               "neural_beat_probability_by_beat": [0.95] * count,
                               "neural_downbeat_probability_by_beat": [0.90 if i % 4 == 0 else 0.05
                                                                          for i in range(count)]})
            planned = plan("a", "b")
            planned["edges"][0]["tempo"]["beatmatch_possible"] = False
            planned["edges"][0]["tempo_reset"] = True
            result = render_plan(planned, records, {}, directory / "set.wav", directory / "work",
                                 sample_rate=sr, ffmpeg=os.getenv("SUPERDJ_FFMPEG") or shutil.which("ffmpeg"),
                                 rubberband="not-invoked", allow_hard_cut=True)
            transition = result["transitions"][0]
            self.assertEqual(transition["mode"], "bar_tempo_reset")
            self.assertFalse(transition["beatmatched"])
            self.assertEqual(transition["overlap_frames"], 0)
            evidence = transition["outgoing_cue"]["local_validation"]
            self.assertLess(evidence["whole_track_confidence"], 0.75)
            self.assertGreater(evidence["local_timing_confidence"], 0.75)
            self.assertEqual(sf.info(directory / "set.wav").subtype, "FLOAT")
            faded = render_plan(planned, records, {}, directory / "fade.wav", directory / "fade-work",
                                sample_rate=sr, ffmpeg=os.getenv("SUPERDJ_FFMPEG") or shutil.which("ffmpeg"),
                                rubberband="not-invoked", allow_unverified=True)
            self.assertEqual(faded["transitions"][0]["mode"], "unverified_short_crossfade")
            self.assertGreater(faded["transitions"][0]["overlap_frames"], 0)

    @unittest.skipUnless((os.getenv("SUPERDJ_FFMPEG") or shutil.which("ffmpeg")) and
                         (os.getenv("SUPERDJ_RUBBERBAND") or shutil.which("rubberband")),
                         "Set SUPERDJ_FFMPEG and SUPERDJ_RUBBERBAND for the real audio timing test")
    def test_real_cli_preserves_float_and_places_transients_on_shared_beats(self):
        sr = 8000
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = []
            for identity, bpm in (("a", 120), ("b", 124)):
                path = directory / (identity + ".wav")
                item = record(identity, path, bpm)
                audio = np.zeros((90 * sr, 2), dtype=np.float32)
                pulse = np.array([0.0, 0.2, 0.5, 0.2, 0.0], dtype=np.float32)
                for beat in item["beats"]:
                    centre = round(beat * sr)
                    audio[centre - 2:centre + 3] = pulse[:, None]
                sf.write(path, audio, sr, subtype="FLOAT")
                records.append(item)
            result = render_plan(plan("a", "b"), records, {}, directory / "set.wav",
                                 directory / "work", sample_rate=sr, bass_hz=0,
                                 ffmpeg=os.getenv("SUPERDJ_FFMPEG") or shutil.which("ffmpeg"),
                                 rubberband=os.getenv("SUPERDJ_RUBBERBAND") or shutil.which("rubberband"))
            self.assertEqual(sf.info(directory / "set.wav").subtype, "FLOAT")
            self.assertEqual(result["track_count"], 2)
            self.assertEqual(len(result["transitions"]), 1)
            transition = result["transitions"][0]
            self.assertTrue(transition["beatmatched"])
            self.assertLess(transition["mapped_beat_alignment_max_error_seconds"], 2 / sr)
            self.assertEqual(sf.info(directory / "set.wav").frames,
                             sum(track["output_frames"] for track in result["tracks"]) -
                             transition["overlap_frames"])
            # Check rendered PCM independently of the mathematical map. Exclude
            # the boundary transients, where an intentional clip trim applies.
            found = []
            for track, role in ((result["tracks"][0], "tail"), (result["tracks"][1], "head")):
                audio, _ = sf.read(track["audio_path"], dtype="float32", always_2d=True)
                actual = []
                for anchor in track["warp"]["input_output_beat_mapping"][role][2:-2]:
                    expected = anchor["output_seconds"]
                    lo, hi = round((expected - 0.05) * sr), round((expected + 0.05) * sr)
                    segment = np.abs(audio[lo:hi, 0])
                    self.assertGreater(float(np.max(segment)), 0.05)
                    observed = (lo + int(np.argmax(segment))) / sr
                    self.assertLess(abs(observed - expected), 0.025)
                    actual.append(observed + track["set_start_seconds"])
                found.append(actual)
            # At 8 kHz the real R2 CLI's transient alignment in this fixture is
            # within 15 ms; anchor arithmetic itself is sample-frame accurate.
            np.testing.assert_allclose(found[0], found[1], rtol=0, atol=0.015)


if __name__ == "__main__":
    unittest.main()
