"""Protect native incoming tempo from a following edge's early automation."""

import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

from superdj.render import RenderError, _Window, _warp_anchors, render_plan


class NativeIncomingChainTests(unittest.TestCase):
    sample_rate = 8000

    def record(self, identity, bpm, seconds, path=None):
        beats = np.arange(0.25, seconds - 0.1, 60 / bpm)
        return {"id": identity, "path": str(path or "unused.wav"),
                "duration": seconds, "bpm": bpm,
                "beats": beats.tolist(), "downbeats": beats[::4].tolist(),
                "grid_status": "reviewed", "grid_confidence": 1.0,
                "grid": {"meter": 4, "human_reviewed": True}}

    def plan(self, records):
        return {"order": [item["id"] for item in records],
                "constraints": {"allow_tempo_reset": True},
                "edges": [
                    {"from": "fast", "to": "native", "tempo_reset": True,
                     "tempo": {"beatmatch_possible": False, "target_bpm": 128.0,
                               "technique": "outgoing_only_vinyl"},
                     "cue_locks": {"outgoing_end_seconds": records[0]["downbeats"][-1],
                                   "incoming_start_seconds": records[1]["downbeats"][8]}},
                    {"from": "native", "to": "next", "tempo_reset": False,
                     "tempo": {"beatmatch_possible": True, "target_bpm": 129.0},
                     "cue_locks": {"outgoing_end_seconds": records[1]["downbeats"][-1],
                                   "incoming_start_seconds": records[2]["downbeats"][0]}},
                ]}

    def test_short_body_can_pass_duration_checks_while_breaking_native_entry(self):
        # A mathematically valid middle clip still allows its future shortened
        # ramp to begin inside the protected first 8 s. This is distinct
        # from overlaps consuming the entire clip.
        track = self.record("native", 128.0, 45.0)
        beats = np.asarray(track["beats"])
        last = int(np.argmin(np.abs(beats - track["downbeats"][-1])))
        times = beats[last - 32:last + 1]
        tail = _Window(times, 8, 4, 129.0, round(32 * 60 / 129 * self.sample_rate))
        start, end = track["downbeats"][8], float(times[-1])
        result = _warp_anchors(track, trim_start=start, trim_end=end,
                               source_frames=round((end - start) * self.sample_rate),
                               sample_rate=self.sample_rate, head=None, tail=tail,
                               max_tempo_change=0.06, max_local_warp_change=0.15)
        self.assertLess(8 * self.sample_rate + tail.frames, result["output_frames"])
        self.assertLess(result["tempo_ramps"][0]["output_start_seconds"], 8)

    @unittest.skipUnless(os.getenv("SUPERDJ_FFMPEG") or shutil.which("ffmpeg"),
                         "FFmpeg required for the real three-track chain")
    def test_three_track_render_refuses_future_ramp_inside_native_incoming_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = []
            for identity, bpm, seconds in (("fast", 174.0, 90),
                                           ("native", 128.0, 45),
                                           ("next", 130.0, 90)):
                path = directory / (identity + ".wav")
                t = np.arange(seconds * self.sample_rate) / self.sample_rate
                signal = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
                sf.write(path, np.column_stack((signal, signal)), self.sample_rate, subtype="FLOAT")
                records.append(self.record(identity, bpm, seconds, path))
            output = directory / "set.wav"
            with self.assertRaisesRegex(RenderError,
                                        "Next tempo automation intersects the native incoming overlap"):
                render_plan(self.plan(records), records, {}, output, directory / "work",
                            sample_rate=self.sample_rate, transition_bars=16,
                            tempo_conflict_policy="adaptive", native_exit_seconds=8,
                            ffmpeg=os.getenv("SUPERDJ_FFMPEG") or shutil.which("ffmpeg"),
                            rubberband="must-not-run-before-native-entry-validation")
            self.assertFalse(output.exists())
            self.assertFalse((directory / "set.partial.wav").exists())


if __name__ == "__main__":
    unittest.main()
