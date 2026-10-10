"""Native timing, gain, boundary and filtering invariants for echo bridges."""

import math
import unittest

import numpy as np

from superdj.bridges import BridgeError, echo_bridge


class EchoBridgeTests(unittest.TestCase):
    sr = 8000
    out_bpm = 174.0
    in_bpm = 128.0

    def sources(self, *, amplitude=0.5):
        count = round(8 * 60 / self.in_bpm * self.sr)
        t = np.arange(count) / self.sr
        # Deliberately different rhythms and stereo content. No external music.
        outgoing = np.column_stack((np.cos(2 * math.pi * 60 * t),
                                    np.sin(2 * math.pi * 1200 * t))) * amplitude
        incoming = np.column_stack((np.sin(2 * math.pi * 93 * t),
                                    np.cos(2 * math.pi * 1700 * t))) * amplitude
        return outgoing.astype(np.float32), incoming.astype(np.float32)

    def bridge(self, outgoing, incoming, **kwargs):
        return echo_bridge(outgoing, incoming, self.sr,
                           outgoing_bpm=self.out_bpm, incoming_bpm=self.in_bpm, **kwargs)

    def test_endpoints_and_native_incoming_samples_are_preserved(self):
        a, b = self.sources()
        mixed, report = self.bridge(a, b)
        np.testing.assert_array_equal(mixed[0], a[0])
        np.testing.assert_array_equal(mixed[-1], b[-1])
        end = math.ceil(report["echo_end_seconds"] * self.sr) + 2
        np.testing.assert_array_equal(mixed[end:], b[end:])
        self.assertEqual(mixed.dtype, np.float32)
        self.assertEqual(mixed.shape, a.shape)
        self.assertFalse(report["time_stretch_applied"])
        self.assertFalse(report["beatmatched"])
        self.assertFalse(report["phase_alignment_verified"])
        self.assertFalse(report["phrase_alignment_verified"])
        self.assertFalse(report["meter_inferred"])

    def test_inputs_and_views_are_not_modified(self):
        a, b = self.sources()
        original_a, original_b = a.copy(), b.copy()
        a.flags.writeable = False
        b.flags.writeable = False
        self.bridge(a, b)
        np.testing.assert_array_equal(a, original_a)
        np.testing.assert_array_equal(b, original_b)

    def test_convex_gains_keep_loud_dc_and_echo_bounded(self):
        a, b = self.sources(amplitude=1.0)
        # Loud aligned/transient content exercises the finite delay sum, not
        # merely a quiet sine wave that could never expose clipping.
        a[:, 0] = 1.0
        a[:self.sr // 2, 1] = np.where(np.arange(self.sr // 2) % 13 < 6, 1, -1)
        b[:, 0] = 1.0
        mixed, report = self.bridge(a, b, echo_gain=0.8, echo_decay=0.95,
                                    echo_repeats=16)
        self.assertLessEqual(float(np.max(np.abs(mixed))), 1.000001)
        self.assertLessEqual(report["headroom"]["max_gain_sum"], 1.000001)
        self.assertLess(report["headroom"]["echo_attenuation_gain"], 1.0)
        self.assertEqual(report["headroom"]["whole_region_normalization_gain"], 1.0)

    def test_incoming_waits_until_outgoing_dry_bass_is_removed(self):
        a, b = self.sources()
        both, report = self.bridge(a, b)
        outgoing_only, _ = self.bridge(a, np.zeros_like(b))
        incoming_only, _ = self.bridge(np.zeros_like(a), b)
        start = round(report["incoming_dry_start_seconds"] * self.sr)
        np.testing.assert_array_equal(incoming_only[:start + 1], 0)
        np.testing.assert_array_equal(both[:start + 1], outgoing_only[:start + 1])
        after = round((report["incoming_dry_start_seconds"] + 0.1) * self.sr)
        self.assertGreater(float(np.max(np.abs(incoming_only[after:]))), 0.01)

    def test_echo_contains_source_but_attenuates_its_bass(self):
        a, b = self.sources()
        silence = np.zeros_like(a)
        mixed, report = self.bridge(a, silence)
        begin = round(report["dry_out_end_seconds"] * self.sr) + 20
        end = round(report["echo_end_seconds"] * self.sr) - 20
        tail = mixed[begin:end]
        low_rms = float(np.sqrt(np.mean(tail[:, 0] ** 2)))
        high_rms = float(np.sqrt(np.mean(tail[:, 1] ** 2)))
        self.assertGreater(high_rms, 0.002)
        self.assertLess(low_rms, high_rms * 0.1)
        self.assertEqual(report["echo_taps"][0]["delay_frames"],
                         round(0.5 * 60 / self.out_bpm * self.sr))

    def test_constant_input_gain_envelopes_are_continuous(self):
        a, _ = self.sources()
        a[:] = 0.5
        # Smooth dry gains must not create a click even for a nonzero signal.
        mixed, _ = self.bridge(a, a)
        self.assertLess(float(np.max(np.abs(np.diff(mixed[:, 0])))), 0.002)
        np.testing.assert_array_equal(mixed[0], a[0])
        np.testing.assert_array_equal(mixed[-1], a[-1])

    def test_silent_sources_stay_silent_without_nan(self):
        a, _ = self.sources()
        mixed, report = self.bridge(np.zeros_like(a), np.zeros_like(a))
        np.testing.assert_array_equal(mixed, 0)
        self.assertTrue(np.all(np.isfinite(mixed)))
        self.assertEqual(report["headroom"]["echo_attenuation_gain"], 1.0)

    def test_malformed_audio_and_mismatched_duration_rejected(self):
        a, b = self.sources()
        for broken in (a[:, 0], a[:10], a.astype(np.int16),
                       np.full_like(a, np.nan), np.full_like(a, np.inf)):
            with self.subTest(shape=broken.shape, dtype=broken.dtype):
                with self.assertRaises(BridgeError):
                    self.bridge(broken, b)
        with self.assertRaisesRegex(BridgeError, "same number"):
            self.bridge(a[:-2], b)
        with self.assertRaisesRegex(BridgeError, "frame count"):
            self.bridge(a[:-20], b[:-20])
        with self.assertRaisesRegex(BridgeError, "headroom"):
            self.bridge(a * 3, b)

    def test_invalid_controls_rejected(self):
        a, b = self.sources()
        for options in ({"overlap_beats": -1}, {"outgoing_bpm": np.nan},
                        {"incoming_bpm": 0}, {"echo_decay": 1},
                        {"echo_repeats": 17}, {"echo_repeats": 1.5},
                        {"echo_gain": 2}, {"dry_out_beats": 7},
                        {"echo_delay_beats": 100}, {"highpass_hz": 5000},
                        {"peak_ceiling": 2}):
            with self.subTest(options=options):
                parameters = {"outgoing_bpm": self.out_bpm, "incoming_bpm": self.in_bpm}
                parameters.update(options)
                with self.assertRaises(BridgeError):
                    echo_bridge(a, b, self.sr, **parameters)

    def test_deterministic_result_and_json_ready_provenance(self):
        import json

        a, b = self.sources()
        mixed1, report1 = self.bridge(a, b)
        mixed2, report2 = self.bridge(a, b)
        np.testing.assert_array_equal(mixed1, mixed2)
        self.assertEqual(report1, report2)
        json.dumps(report1, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
