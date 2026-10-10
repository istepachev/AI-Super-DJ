"""Native incoming timing, outgoing pitch/speed and continuous dual-deck DSP."""

import json
import math
import unittest

import numpy as np

from superdj.creative import (CreativeError, mix_tape_exit, tape_exit_curve,
                              tape_exit_overlap, tape_exit_source_frames, tape_speed_warp)


class VinylExitTests(unittest.TestCase):
    sr = 8000
    frames = 15 * sr
    out_bpm = 174.066
    in_bpm = 128.06

    def curve(self):
        return tape_exit_curve(self.frames, self.sr, ramp_seconds=12,
                                end_rate=self.in_bpm / self.out_bpm)

    def sources(self):
        required = tape_exit_source_frames(self.frames, self.sr, outgoing_bpm=self.out_bpm,
                                            incoming_bpm=self.in_bpm, ramp_seconds=12)
        ta = np.arange(required) / self.sr
        tb = np.arange(self.frames) / self.sr
        a = np.column_stack((0.12 * np.sin(2 * math.pi * 440 * ta),
                             0.10 * np.sin(2 * math.pi * 1100 * ta))).astype(np.float32)
        b = np.column_stack((0.1 * np.sin(2 * math.pi * 700 * tb),
                             0.12 * np.sin(2 * math.pi * 60 * tb))).astype(np.float32)
        return a, b

    def test_curve_integrates_output_seconds_and_reports_pitch_truthfully(self):
        positions, manifest = self.curve()
        ratio = self.in_bpm / self.out_bpm
        expected = 12 * (1 + ratio) / 2 + 3 * ratio
        self.assertAlmostEqual(manifest["source_consumption_seconds"], expected, places=10)
        self.assertEqual(manifest["source_frames_required"], math.ceil(expected * self.sr) + 1)
        self.assertEqual(positions[0], 0)
        self.assertTrue(np.all(np.diff(positions) > 0))
        self.assertLessEqual(float(np.max(np.diff(positions))), 1)
        self.assertAlmostEqual(float(np.diff(positions)[0]), 1.0, places=8)
        self.assertAlmostEqual(float(np.diff(positions)[-1]), ratio, places=8)
        self.assertEqual(manifest["ramp_output_seconds"], 12)
        self.assertEqual(manifest["incoming_rate"], 1)
        self.assertFalse(manifest["outgoing_pitch_preserved"])
        self.assertAlmostEqual(manifest["outgoing_pitch_change_semitones"], 12 * math.log2(ratio))

    def test_sinc_warp_preserves_start_and_matches_analytic_sine_phase(self):
        a, _ = self.sources()
        positions, _ = self.curve()
        warped = tape_speed_warp(a, positions)
        expected = 0.12 * np.sin(2 * math.pi * 440 * positions / self.sr)
        self.assertEqual(warped.shape, (self.frames, 2))
        self.assertEqual(warped.dtype, np.float32)
        np.testing.assert_allclose(warped[32:-32, 0], expected[32:-32], rtol=0, atol=0.0002)
        np.testing.assert_allclose(warped[0], a[0], rtol=0, atol=1e-7)

    def test_outgoing_frequency_falls_only_on_its_exit(self):
        a, _ = self.sources()
        positions, _ = self.curve()
        warped = tape_speed_warp(a, positions)
        early = warped[self.sr // 2:self.sr + self.sr // 2, 0]
        late = warped[13 * self.sr:14 * self.sr, 0]

        def frequency(block):
            spectrum = np.abs(np.fft.rfft(block * np.hanning(len(block))))
            return int(np.argmax(spectrum)) * self.sr / len(block)

        self.assertGreater(frequency(early), 430)
        self.assertAlmostEqual(frequency(late), 440 * self.in_bpm / self.out_bpm, delta=1.0)

    def test_two_decks_overlap_early_and_keep_native_incoming_pulses(self):
        _, b = self.sources()
        # Broad transients test the incoming sample clock independently of the
        # speed-map metadata. Zero-phase filtering must not remap these events.
        b[:] = 0
        native_times = np.arange(0.25, 14.9, 60 / self.in_bpm)
        pulse = np.array([0.0, 0.04, 0.2, 0.04, 0.0], dtype=np.float32)
        for time in native_times:
            frame = round(time * self.sr)
            b[frame - 2:frame + 3] = pulse[:, None]
        zero = np.zeros_like(b)
        incoming_only, manifest = mix_tape_exit(zero, b, self.sr)
        self.assertGreater(manifest["both_decks_nonzero_upper_gain_fraction"], 0.97)
        self.assertEqual(manifest["whole_overlap_normalization_gain"], 1)
        for time in native_times:
            frame = round(time * self.sr)
            section = np.abs(incoming_only[frame - 8:frame + 9, 0])
            found = frame - 8 + int(np.argmax(section))
            self.assertLessEqual(abs(found - frame), 1)
        # Both sources contribute well before the last release, rather than
        # waiting for the outgoing deck to go silent.
        t = np.arange(self.frames) / self.sr
        outgoing = np.column_stack((0.1 * np.sin(2 * math.pi * 900 * t),
                                    0.1 * np.sin(2 * math.pi * 1300 * t))).astype(np.float32)
        outgoing_only, _ = mix_tape_exit(outgoing, zero, self.sr)
        for second in (1, 5, 9, 11):
            region = slice(second * self.sr, (second + 1) * self.sr)
            self.assertGreater(float(np.max(np.abs(incoming_only[region]))), 0.03)
            self.assertGreater(float(np.max(np.abs(outgoing_only[region]))), 0.02)

    def test_bass_is_transferred_without_doubling_identical_low_bands(self):
        t = np.arange(self.frames) / self.sr
        sine = (0.3 * np.sin(2 * math.pi * 40 * t)).astype(np.float32)
        audio = np.column_stack((sine, sine))
        mixed, manifest = mix_tape_exit(audio, audio, self.sr)
        middle = mixed[4 * self.sr:6 * self.sr]
        self.assertLess(float(np.max(np.abs(middle))), 0.305)
        self.assertEqual(manifest["bass_swap_start_seconds"], 2.25)
        self.assertEqual(manifest["bass_swap_end_seconds"], 7.5)

    def test_wrapper_boundaries_immutability_and_honest_provenance(self):
        a, b = self.sources()
        original_a, original_b = a.copy(), b.copy()
        mixed, manifest = tape_exit_overlap(a, b, self.sr, outgoing_bpm=self.out_bpm,
                                            incoming_bpm=self.in_bpm)
        np.testing.assert_array_equal(mixed[0], a[0])
        np.testing.assert_array_equal(mixed[-1], b[-1])
        np.testing.assert_array_equal(a, original_a)
        np.testing.assert_array_equal(b, original_b)
        self.assertFalse(manifest["incoming_time_stretched"])
        self.assertFalse(manifest["beatmatched_throughout_overlap"])
        self.assertFalse(manifest["human_auditioned"])
        self.assertFalse(manifest["rubberband_used"])
        self.assertLess(manifest["output_sample_peak"], 1)
        json.dumps(manifest, allow_nan=False)

    def test_short_eight_second_exit_keeps_native_incoming_and_early_bass(self):
        incoming_frames = 8 * self.sr
        required = tape_exit_source_frames(incoming_frames, self.sr, outgoing_bpm=self.out_bpm,
                                            incoming_bpm=self.in_bpm, ramp_seconds=6)
        ta = np.arange(required) / self.sr
        tb = np.arange(incoming_frames) / self.sr
        a = np.column_stack((0.1 * np.sin(2 * math.pi * 440 * ta),
                             0.1 * np.sin(2 * math.pi * 1100 * ta))).astype(np.float32)
        b = np.column_stack((0.1 * np.sin(2 * math.pi * 800 * tb),
                             0.1 * np.sin(2 * math.pi * 55 * tb))).astype(np.float32)
        mixed, manifest = tape_exit_overlap(
            a, b, self.sr, outgoing_bpm=self.out_bpm, incoming_bpm=self.in_bpm,
            ramp_seconds=6, outgoing_release_seconds=2,
            bass_swap_start=1.5 / 8, bass_swap_end=3.74 / 8)
        self.assertEqual(manifest["overlap_seconds"], 8)
        self.assertEqual(manifest["outgoing_speed_curve"]["ramp_output_seconds"], 6)
        self.assertAlmostEqual(manifest["bass_swap_start_seconds"], 1.5)
        self.assertAlmostEqual(manifest["bass_swap_end_seconds"], 3.74)
        self.assertGreater(manifest["both_decks_nonzero_upper_gain_fraction"], 0.97)
        self.assertEqual(manifest["incoming_rate"], 1)
        np.testing.assert_array_equal(mixed[0], a[0])
        np.testing.assert_array_equal(mixed[-1], b[-1])

    def test_invalid_maps_audio_and_parameters_fail_explicitly(self):
        a, b = self.sources()
        positions, _ = self.curve()
        with self.assertRaisesRegex(CreativeError, "needs"):
            tape_exit_overlap(a[:-2], b, self.sr, outgoing_bpm=self.out_bpm,
                              incoming_bpm=self.in_bpm)
        with self.assertRaisesRegex(CreativeError, "shorter"):
            tape_speed_warp(a[:100], positions)
        with self.assertRaises(CreativeError):
            tape_speed_warp(a, np.arange(self.frames) * 1.2)
        with self.assertRaises(CreativeError):
            tape_speed_warp(a, np.full(self.frames, np.nan))
        with self.assertRaises(CreativeError):
            tape_exit_curve(self.frames, self.sr, end_rate=1.2)
        with self.assertRaises(CreativeError):
            tape_exit_curve(self.frames, self.sr, end_rate=0.7, ramp_seconds=-1)
        with self.assertRaises(CreativeError):
            mix_tape_exit(np.zeros_like(b), b, self.sr, incoming_entry_seconds=14)
        with self.assertRaises(CreativeError):
            mix_tape_exit(np.zeros_like(b), b, self.sr, bass_swap_start=0.8, bass_swap_end=0.5)


if __name__ == "__main__":
    unittest.main()
