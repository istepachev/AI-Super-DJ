"""Shared clock integration and independent real-render pulse phase checks."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

from superdj.sync import shared_clock
from superdj.planner import plan_tracks
from superdj.render import render_plan, RenderError

FFMPEG = os.getenv("SUPERDJ_FFMPEG") or shutil.which("ffmpeg")
RUBBERBAND = os.getenv("SUPERDJ_RUBBERBAND") or shutil.which("rubberband")


class ClockTests(unittest.TestCase):
    def test_curve_integrates_to_common_complete_bars(self):
        for b0, b1 in ((174, 128), (128, 174), (120, 122), (128, 128)):
            c = shared_clock(b0, b1, 16, 44100)
            t = np.asarray(c["output_frames"]) / 44100
            z = t / c["duration_seconds"]
            phase = c["duration_seconds"] / 60 * (c["start_bpm"] * z +
                    (c["end_bpm"] - c["start_bpm"]) * (z**3 - z**4 / 2))
            np.testing.assert_allclose(phase, c["phase"], atol=0.00004)
            self.assertLess(abs(c["end_bpm"] - b1), .001)

    def test_invalid_clock_is_rejected(self):
        for settings in ((0, 128, 16, 44100), (174, float("nan"), 16, 44100),
                         (174, 128, 15, 44100)):
            with self.assertRaises(ValueError):
                shared_clock(*settings)


@unittest.skipUnless(FFMPEG and RUBBERBAND, "Actual audio tools needed for SYNC tests")
class RenderedSyncTests(unittest.TestCase):
    def fixtures(self, directory, bpms):
        sr, duration = 16000, 90
        records = []
        for identity, bpm in zip(("a", "b", "c"), bpms):
            beats = np.arange(.25, duration - .05, 60 / bpm)
            audio = np.zeros(duration * sr, dtype=np.float32)
            width = 128
            pulse = .25 * np.exp(-.5 * (np.arange(-width, width + 1) / (sr * .0015))**2)
            for beat in beats:
                centre = round(beat * sr)
                audio[centre-width:centre+width+1] += pulse
            path = directory / (identity + ".wav")
            sf.write(path, np.column_stack((audio, audio)), sr, subtype="FLOAT")
            records.append({"id": identity, "path": str(path), "duration": duration,
                "bpm": bpm, "bpm_confidence": 1., "energy": .5,
                "key_camelot": "8A", "key_confidence": 1., "beats": beats.tolist(),
                "downbeats": beats[::4].tolist(), "grid_status": "reviewed",
                "grid_confidence": 1., "grid": {"meter": 4, "human_reviewed": True}})
        return records

    def render(self, directory, records, **settings):
        return render_plan(plan_tracks(records, mode="fixed"), records, {}, directory / "set.wav",
            directory / "work", ffmpeg=FFMPEG, rubberband=RUBBERBAND,
            sample_rate=16000, transition_bars=8, **settings)

    def pulse_positions(self, path, times):
        x, sr = sf.read(path, always_2d=True)
        energy = np.abs(x).max(axis=1)
        result = []
        for t in times:
            begin, end = round((t-.035)*sr), round((t+.035)*sr)
            self.assertGreater(energy[max(0, begin):end].max(), .02)
            result.append((max(0, begin) + np.argmax(energy[max(0, begin):end])) / sr)
        return np.asarray(result)

    def test_default_both_decks_follow_changing_clock_in_both_directions(self):
        for bpms in ((174, 128), (128, 174), (120, 122)):
            with self.subTest(bpms=bpms), tempfile.TemporaryDirectory() as temporary:
                d = Path(temporary)
                result = self.render(d, self.fixtures(d, bpms))
                edge = result["transitions"][0]
                self.assertEqual(edge["tempo_technique"], "shared_phase_sync")
                self.assertEqual(edge["shared_clock"]["beats"], edge["beats"])
                if abs(bpms[0]-bpms[1]) > 20:
                    self.assertLess(edge["duration_seconds"], 7)
                first, second = result["tracks"]
                observed = []
                for clip, role in ((first, "tail"), (second, "head")):
                    self.assertFalse(clip["warp"]["tempo_ramps"])
                    times = np.array([b["output_seconds"] for b in
                        clip["warp"]["input_output_beat_mapping"][role]])[1:-1]
                    actual = self.pulse_positions(clip["audio_path"], times)
                    self.assertLess(np.max(np.abs(actual - times)), .005)
                    audit = clip["warp"]["post_render_transient_audit"]["final"]
                    self.assertTrue(audit["all_interior_beats_verified"])
                    self.assertLessEqual(audit["maximum_measured_drift_seconds"], .003)
                    observed.append(actual + clip["set_start_seconds"])
                self.assertLess(np.max(np.abs(observed[0]-observed[1])), .005)
                self.assertLess(edge["mapped_beat_alignment_max_error_seconds"], 1/16000)

    def test_long_overlaps_preserve_rendered_phase_across_all_beats(self):
        for bars in (16, 32):
            with self.subTest(bars=bars), tempfile.TemporaryDirectory() as temporary:
                d = Path(temporary)
                result = self.render(d, self.fixtures(d, (138, 174)), sync_transition_bars=bars)
                edge = result["transitions"][0]
                self.assertEqual(edge["beats"], bars * 4)
                self.assertAlmostEqual(edge["duration_seconds"], bars * 4 * 120 / 312, places=3)
                observed = []
                for clip, role in zip(result["tracks"], ("tail", "head")):
                    self.assertFalse(clip["warp"]["tempo_ramps"])
                    times = np.array([b["output_seconds"] for b in
                        clip["warp"]["input_output_beat_mapping"][role]])[1:-1]
                    actual = self.pulse_positions(clip["audio_path"], times)
                    # Long Rubber Band warps can broaden an isolated pulse;
                    # inspect every audible maximum independently, including
                    # attacks the conservative audit declines to certify.
                    self.assertLess(np.max(np.abs(actual-times)), .007)
                    audit = clip["warp"]["post_render_transient_audit"]["final"]
                    self.assertGreaterEqual(len(audit["measured"]), .9 * len(times))
                    self.assertEqual(len(audit["measured"]) + len(audit["unverified"]), len(times))
                    self.assertLessEqual(audit["maximum_measured_drift_seconds"], .003)
                    observed.append(actual + clip["set_start_seconds"])
                self.assertLess(np.max(np.abs(observed[0]-observed[1])), .007)

    def test_temporary_rate_limit_refuses_before_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            d = Path(temporary)
            with self.assertRaisesRegex(RenderError, "temporary rate"):
                self.render(d, self.fixtures(d, (174, 128)), sync_max_change=.1)
            self.assertFalse((d / "set.wav").exists())

    def test_middle_track_has_two_independent_shared_overlaps_and_native_body(self):
        with tempfile.TemporaryDirectory() as temporary:
            d = Path(temporary)
            result = self.render(d, self.fixtures(d, (174, 128, 174)))
            middle = result["tracks"][1]["warp"]
            self.assertFalse(middle["tempo_ramps"])
            self.assertEqual(len(result["transitions"]), 2)
            self.assertTrue(all(e["beatmatched"] for e in result["transitions"]))
