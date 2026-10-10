"""Signal invariants for explicitly labelled partial rhythm experiments."""

import unittest

import numpy as np

from superdj.polyrhythm import PartialRhythmError, partial_rhythm_mix, rational_beat_anchors


class PartialRhythmTests(unittest.TestCase):
    def grids(self):
        return np.arange(33) * 60 / 174.066, np.arange(25) * 60 / 128.06

    def test_four_to_three_anchors_keep_incoming_clock_and_small_outgoing_change(self):
        outgoing, incoming = self.grids()
        original = incoming.copy()
        result = rational_beat_anchors(outgoing, incoming)
        expected = incoming[::3]
        np.testing.assert_allclose(result["shared_output_seconds"], expected, atol=1e-12)
        self.assertAlmostEqual(result["outgoing_mean_rate"], (128.06 * 4 / 3) / 174.066)
        self.assertLess(abs(result["outgoing_mean_rate"] - 1), 0.02)
        self.assertFalse(result["one_to_one_beatmatched"])
        self.assertFalse(result["incoming_time_stretch_applied"])
        np.testing.assert_array_equal(incoming, original)

    def test_observed_incoming_jitter_is_retained_without_quantizing_its_audio_clock(self):
        outgoing, incoming = self.grids()
        incoming[4] += 0.012
        result = rational_beat_anchors(outgoing, incoming)
        np.testing.assert_allclose(result["shared_output_seconds"], incoming[::3], atol=1e-12)
        self.assertAlmostEqual(result["outgoing_output_seconds"][4], incoming[3])

    def test_rejects_incomplete_cycles_or_excessive_outgoing_stretch(self):
        outgoing, incoming = self.grids()
        with self.assertRaises(PartialRhythmError):
            rational_beat_anchors(outgoing[:-1], incoming)
        with self.assertRaises(PartialRhythmError):
            rational_beat_anchors(outgoing, incoming * 1.20)
        with self.assertRaises(PartialRhythmError):
            rational_beat_anchors(outgoing, incoming, outgoing_cycle=3, incoming_cycle=3)

    def test_mean_rate_does_not_hide_an_unsafe_local_beat_jump(self):
        outgoing, incoming = self.grids()
        outgoing[8] += 0.20
        with self.assertRaisesRegex(PartialRhythmError, "local timing correction"):
            rational_beat_anchors(outgoing, incoming)

    def test_mix_has_continuous_joins_and_preserves_inputs(self):
        sr = 16000
        clock = np.arange(sr * 4) / sr
        a = np.column_stack((0.06 * np.sin(2 * np.pi * 900 * clock),
                             0.06 * np.sin(2 * np.pi * 900 * clock)))
        b = np.column_stack((0.07 * np.sin(2 * np.pi * 500 * clock),
                             0.07 * np.sin(2 * np.pi * 500 * clock)))
        original_a, original_b = a.copy(), b.copy()
        mixed, evidence = partial_rhythm_mix(a, b, sr, shared_seconds=[0, 1, 2, 3, 4])
        np.testing.assert_allclose(mixed[0], a[0], atol=1e-8)
        np.testing.assert_allclose(mixed[-1], b[-1], atol=1e-8)
        np.testing.assert_array_equal(a, original_a)
        np.testing.assert_array_equal(b, original_b)
        self.assertFalse(evidence["sequential_mute_applied"])
        self.assertFalse(evidence["incoming_time_stretch_applied"])
        self.assertLess(evidence["sample_peak"], 0.20)

    def test_outgoing_off_cycle_high_band_is_softened(self):
        sr = 16000
        clock = np.arange(sr * 4) / sr
        mono = 0.10 * np.sin(2 * np.pi * 4000 * clock)
        a = np.column_stack((mono, mono))
        mixed, _ = partial_rhythm_mix(a, np.zeros_like(a), sr,
                                     shared_seconds=[0, 1, 2, 3, 4])
        def rms(begin, end):
            return np.sqrt(np.mean(mixed[int(begin * sr):int(end * sr)] ** 2))
        self.assertLess(rms(1.25, 1.30), rms(0.99, 1.01) * 0.40)
        self.assertGreater(rms(1.25, 1.30), 0)

    def test_invalid_audio_and_peak_headroom_fail(self):
        a = np.full((1000, 2), 0.8, dtype=np.float32)
        with self.assertRaises(PartialRhythmError):
            partial_rhythm_mix(a, a, 8000, shared_seconds=[0, 0.1], peak_ceiling=0.50)
        with self.assertRaises(PartialRhythmError):
            partial_rhythm_mix(a[:, 0], a, 8000, shared_seconds=[0, 0.1])
        with self.assertRaises(PartialRhythmError):
            partial_rhythm_mix(a, a, 8000, shared_seconds=[0, float("nan")])


if __name__ == "__main__":
    unittest.main()
