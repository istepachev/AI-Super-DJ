"""Output-time tempo-ramp durations and independent overlap-grid invariants."""

import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import soundfile as sf

from superdj.render import RenderError, _Window, _warp_anchors


class OutputTempoRampTests(unittest.TestCase):
    sample_rate = 8000

    def track(self, bpm=128.0, seconds=180.0):
        beats = np.arange(0.0, seconds + 1e-6, 60 / bpm)
        return {"id": "generated", "bpm": bpm, "duration": seconds,
                "beats": beats.tolist(), "downbeats": beats[::4].tolist()}

    def window(self, track, target_bpm, start_beat, bars=8):
        count = bars * 4
        times = np.asarray(track["beats"][start_beat:start_beat + count + 1])
        return _Window(times, bars, 4, target_bpm,
                       round(count * 60 / target_bpm * self.sample_rate))

    def warp(self, track, *, head=None, tail=None, ramp=15.0, seconds=None,
             timing_cap=0.15):
        duration = track["duration"] if seconds is None else seconds
        return _warp_anchors(track, trim_start=0.0, trim_end=duration,
                             source_frames=round(duration * self.sample_rate),
                             sample_rate=self.sample_rate, head=head, tail=tail,
                             max_tempo_change=0.25, max_local_warp_change=timing_cap,
                             tempo_ramp_seconds=ramp)

    def test_requested_seconds_are_output_seconds_at_both_native_tempos(self):
        # A common overlap BPM requires substantial rate changes on both decks.
        # The arithmetic mean of the ramp's endpoint rates independently fixes
        # source consumption; it is not its duration in rendered audio.
        common = math.sqrt(174.0 * 128.0)
        for native in (174.0, 128.0):
            for seconds in (10.0, 15.0, 20.0):
                with self.subTest(native=native, seconds=seconds):
                    track = self.track(native)
                    head = self.window(track, common, 0)
                    result = self.warp(track, head=head, ramp=seconds, timing_cap=0.01)
                    ramp = result["tempo_ramps"][0]
                    self.assertEqual(ramp["role"], "entry")
                    self.assertAlmostEqual(ramp["output_duration_seconds"], seconds, places=7)
                    source_consumed = ramp["source_end_seconds"] - ramp["source_start_seconds"]
                    self.assertAlmostEqual(source_consumed,
                                           seconds * (common / native + 1) / 2, places=7)
                    self.assertGreater(abs(source_consumed - seconds), 0.5)
                    self.assertAlmostEqual(ramp["output_start_seconds"],
                                           head.frames / self.sample_rate, places=7)
                    self.assertFalse(ramp["shortened_for_available_body"])
                    self.assertGreater(result["max_local_warp_rate_change"], 0.14)
                    self.assertLess(result["max_timing_correction_change"], 0.001)

    def test_head_and_tail_keep_exact_shared_beats_with_separate_ramps(self):
        track = self.track(128.0)
        head = self.window(track, 147.5, 0)
        tail = self.window(track, 112.0, 320)
        result = self.warp(track, head=head, tail=tail, ramp=15.0, timing_cap=0.01)
        self.assertEqual([item["role"] for item in result["tempo_ramps"]], ["entry", "exit"])
        entry, exit = result["tempo_ramps"]
        self.assertAlmostEqual(entry["output_duration_seconds"], 15.0, places=7)
        self.assertAlmostEqual(exit["output_duration_seconds"], 15.0, places=7)
        self.assertLess(entry["output_end_seconds"], exit["output_start_seconds"])
        for role, window in (("head", head), ("tail", tail)):
            mapping = result["input_output_beat_mapping"][role]
            self.assertEqual(len(mapping), len(window.times))
            intervals = np.diff([point["output_seconds"] for point in mapping])
            np.testing.assert_allclose(intervals,
                                       window.frames / self.sample_rate / (len(window.times) - 1),
                                       rtol=0, atol=1e-12)
            for beat in mapping:
                source_frame = round(beat["source_seconds"] * self.sample_rate)
                anchor = next(point for point in result["anchors"]
                              if point["source_frame"] == source_frame)
                self.assertLessEqual(abs(anchor["output_seconds"] - beat["output_seconds"]),
                                     0.5 / self.sample_rate + 1e-12)
        self.assertAlmostEqual(exit["output_end_seconds"],
                               result["input_output_beat_mapping"]["tail"][0]["output_seconds"],
                               delta=0.5 / self.sample_rate)

    def test_ramp_map_has_monotonic_smooth_rates_and_native_plateau(self):
        track = self.track(128.0)
        head = self.window(track, 147.5, 0)
        result = self.warp(track, head=head)
        ramp = result["tempo_ramps"][0]
        anchors = result["anchors"]
        actual_rates = []
        spacings = []
        plateau_rates = []
        for a, b in zip(anchors, anchors[1:]):
            self.assertGreater(b["source_frame"], a["source_frame"])
            self.assertGreater(b["output_frame"], a["output_frame"])
            rate = (b["source_frame"] - a["source_frame"]) / (b["output_frame"] - a["output_frame"])
            if (a["output_seconds"] >= ramp["output_start_seconds"] and
                    b["output_seconds"] <= ramp["output_end_seconds"]):
                actual_rates.append(rate)
                spacings.append(b["output_seconds"] - a["output_seconds"])
            if a["output_seconds"] > ramp["output_end_seconds"] + 1:
                plateau_rates.append(rate)
        # Quantisation can slightly perturb a subinterval, but not introduce
        # a jump comparable to the intended 15% musical rate change.
        self.assertLess(float(np.max(np.abs(np.diff(actual_rates)))), 0.02)
        self.assertGreater(actual_rates[0], 1.14)
        self.assertLess(actual_rates[-1], 1.01)
        self.assertLessEqual(max(spacings), 0.25 + 1 / self.sample_rate)
        np.testing.assert_allclose(plateau_rates, 1.0, rtol=0, atol=0.001)

    def test_short_body_reports_actual_shortened_durations(self):
        track = self.track(128.0)
        head = self.window(track, 147.5, 0)
        tail = self.window(track, 112.0, 48)
        end = float(tail.times[-1])
        result = self.warp(track, head=head, tail=tail, seconds=end)
        self.assertTrue(result["tempo_ramps_shortened_for_available_body"])
        for ramp in result["tempo_ramps"]:
            self.assertTrue(ramp["shortened_for_available_body"])
            self.assertEqual(ramp["requested_output_duration_seconds"], 15.0)
            self.assertGreater(ramp["output_duration_seconds"], 0)
            self.assertLess(ramp["output_duration_seconds"], 15.0)
        self.assertAlmostEqual(result["tempo_ramps"][0]["output_end_seconds"],
                               result["tempo_ramps"][1]["output_start_seconds"], places=6)
        self.assertEqual(result["anchors"][-1]["source_frame"], round(end * self.sample_rate))

    def test_near_coincident_bar_and_dense_controls_do_not_create_rate_spikes(self):
        # These non-integer grids put body bars very close to dense ramp
        # controls. Rounding a tiny interval must not create a fictitious
        # 1.5x local rate when the intended change is only a few percent.
        for native, entry, exit in ((155.6622033126625, 160.17896375905653, 162.86113155548483),
                                   (154.45899958421234, 126.95433271442408, 180.0483421068027),
                                   (153.60227554351223, 176.9255171441516, 175.40490024679536)):
            with self.subTest(native=native):
                track = self.track(native, 240.0)
                head = self.window(track, entry, 0)
                start = (len(track["beats"]) - 33) // 4 * 4
                tail = self.window(track, exit, start)
                result = self.warp(track, head=head, tail=tail, timing_cap=0.01)
                body_rates = [(b["source_frame"] - a["source_frame"]) /
                              (b["output_frame"] - a["output_frame"])
                              for a, b in zip(result["anchors"], result["anchors"][1:])
                              if "body_" in b["role"]]
                self.assertLess(max(body_rates), max(entry / native, exit / native, 1.0) + 0.01)
                self.assertGreater(min(body_rates), min(entry / native, exit / native, 1.0) - 0.01)
                self.assertLess(float(np.max(np.abs(np.diff(body_rates)))), 0.02)

    def test_missing_body_refuses_instantaneous_tempo_jump(self):
        track = self.track(128.0)
        head = self.window(track, 147.5, 0)
        tail = self.window(track, 112.0, 32)
        with self.assertRaisesRegex(RenderError, "continuous tempo ramp"):
            self.warp(track, head=head, tail=tail, seconds=float(tail.times[-1]))

    def test_no_automation_invents_a_ramp_when_native_rate_is_unchanged(self):
        track = self.track()
        result = self.warp(track, head=self.window(track, 128.0, 0))
        self.assertEqual(result["tempo_ramps"], [])
        self.assertFalse(result["tempo_ramps_shortened_for_available_body"])
        self.assertEqual(result["output_frames"], round(track["duration"] * self.sample_rate))

    def test_invalid_duration_and_real_overlap_jitter_rejected(self):
        track = self.track()
        head = self.window(track, 147.5, 0)
        for duration in (0, -1, np.nan, np.inf):
            with self.subTest(duration=duration):
                with self.assertRaises(RenderError):
                    self.warp(track, head=head, ramp=duration)
        head.times[12] += 0.2
        with self.assertRaisesRegex(RenderError, "Local timing correction"):
            self.warp(track, head=head, timing_cap=0.15)

    @unittest.skipUnless(os.getenv("SUPERDJ_RUBBERBAND") or shutil.which("rubberband"),
                         "Rubber Band required to check actual PCM tempo ramps")
    def test_real_pcm_keeps_shared_overlap_and_reaches_native_pulse_period(self):
        native = 174.0
        target = math.sqrt(native * 128.0)
        track = self.track(native, 90.0)
        head = self.window(track, target, 0)
        result = self.warp(track, head=head, ramp=15.0, timing_cap=0.01)
        sr = self.sample_rate
        source = np.zeros((90 * sr, 2), dtype=np.float32)
        pulse = np.array([0.0, 0.2, 0.5, 0.2, 0.0], dtype=np.float32)
        for beat in track["beats"][1:]:
            frame = round(beat * sr)
            if frame + 3 < len(source):
                source[frame - 2:frame + 3] = pulse[:, None]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            sf.write(directory / "input.wav", source, sr, subtype="FLOAT")
            (directory / "time-map.txt").write_text(
                "".join(f"{item['source_frame']} {item['output_frame']}\n"
                        for item in result["anchors"]), encoding="ascii")
            executable = os.getenv("SUPERDJ_RUBBERBAND") or shutil.which("rubberband")
            process = subprocess.run(
                [executable, "--fast", "--detector-perc", "--centre-focus",
                 "--ignore-clipping", "--quiet", "--time",
                 f"{result['output_frames'] / len(source):.12f}",
                 "--timemap", "time-map.txt", "input.wav", "output.wav"],
                cwd=directory, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            info = sf.info(directory / "output.wav")
            self.assertEqual(info.subtype, "FLOAT")
            self.assertLessEqual(abs(info.frames - result["output_frames"]), 128)
            audio, _ = sf.read(directory / "output.wav", dtype="float32", always_2d=True)
        anchors = result["anchors"]
        mapped = np.interp(track["beats"],
                           [item["source_frame"] / sr for item in anchors],
                           [item["output_seconds"] for item in anchors])
        found = []
        for expected in mapped[2:-2]:
            start, end = round((expected - 0.04) * sr), round((expected + 0.04) * sr)
            region = np.abs(audio[start:end, 0])
            self.assertGreater(float(np.max(region)), 0.05)
            found.append((start + int(np.argmax(region))) / sr)
        found = np.asarray(found)
        np.testing.assert_allclose(found, mapped[2:-2], rtol=0, atol=0.025)
        ramp = result["tempo_ramps"][0]
        intervals = np.diff(found)
        middle = (found[:-1] + found[1:]) / 2
        shared = intervals[middle < ramp["output_start_seconds"] - 1]
        early = intervals[(middle > ramp["output_start_seconds"] + 1) &
                          (middle < ramp["output_start_seconds"] + 3)]
        late = intervals[(middle > ramp["output_end_seconds"] - 3) &
                         (middle < ramp["output_end_seconds"] - 1)]
        plateau = intervals[middle > ramp["output_end_seconds"] + 3]
        self.assertAlmostEqual(float(np.median(shared)), 60 / target, delta=0.015)
        self.assertGreater(float(np.median(early) - np.median(late)), 0.035)
        self.assertAlmostEqual(float(np.median(plateau)), 60 / native, delta=0.01)


if __name__ == "__main__":
    unittest.main()
