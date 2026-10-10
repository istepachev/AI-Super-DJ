"""One integrated beat phase for both decks during a tempo-changing overlap."""

from __future__ import annotations

import math
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d


def shared_clock(start_bpm: float, end_bpm: float, beats: int,
                 sample_rate: int) -> dict:
    """Smoothstep BPM in output time, inverted at shared quarter-beat controls.

    Duration is chosen for an integer number of beats. Both decks must consume
    the same controls, including the same integer output frames. A tempo value
    alone does not supply this phase lock. Input grids still need validation.
    """
    if (not all(math.isfinite(v) and v > 0 for v in (start_bpm, end_bpm))
            or beats < 4 or beats % 4 or sample_rate < 8000):
        raise ValueError("SYNC needs positive BPM, complete 4/4 bars and a valid sample rate")
    frames = round(beats * 120 / (start_bpm + end_bpm) * sample_rate)
    duration = frames / sample_rate
    # Quantisation changes the requested endpoints by far less than 0.001 BPM.
    scale = beats * 120 / (duration * (start_bpm + end_bpm))
    b0, b1 = start_bpm * scale, end_bpm * scale
    phase = np.arange(beats * 4 + 1, dtype=float) / 4
    low, high = np.zeros_like(phase), np.full_like(phase, duration)
    for _ in range(55):
        middle = (low + high) / 2
        z = middle / duration
        accumulated = duration / 60 * (b0 * z + (b1 - b0) * (z**3 - z**4 / 2))
        low = np.where(accumulated < phase, middle, low)
        high = np.where(accumulated >= phase, middle, high)
    output_frames = np.rint((low + high) / 2 * sample_rate).astype(np.int64)
    output_frames[0], output_frames[-1] = 0, frames
    if np.any(np.diff(output_frames) <= 0):
        raise ValueError("SYNC controls quantise to non-increasing frames")
    return {"start_bpm": b0, "end_bpm": b1, "beats": beats,
            "frames": frames, "duration_seconds": duration,
            "phase": phase.tolist(), "output_frames": output_frames.tolist(),
            "beat_output_frames": output_frames[::4].tolist(),
            "curve": "one integrated cubic smoothstep BPM curve, shared sample-frame beat phase",
            "pitch_preserved": True,
            "acoustic_alignment_verified": False}


def deck_controls(source_beats: np.ndarray, clock: dict, sample_rate: int):
    """Map measured source beats to the common phase, including subbeat controls."""
    if len(source_beats) != clock["beats"] + 1:
        raise ValueError("Both decks must supply exactly the shared beat count")
    source = np.interp(clock["phase"], np.arange(len(source_beats)), source_beats)
    return source, np.asarray(clock["output_frames"], dtype=float) / sample_rate


def transient_drift(source_path, output_path, warp: dict, sample_rate: int) -> dict:
    """Conservative isolated-attack check independent of map arithmetic.

    Dense music often lacks an isolated peak at a predicted beat. Such beats
    remain unverified; never mistake the loudest nearby synth for a kick. This
    deliberately covers only isolated, prominent source/output transients.
    """
    measured, pending = [], []
    with sf.SoundFile(source_path) as source, sf.SoundFile(output_path) as rendered:
        def peak(file, centre):
            radius = round(.035 * sample_rate)
            centre_frame = round(centre * sample_rate)
            first = max(0, centre_frame - radius)
            file.seek(first)
            audio = file.read(2 * radius + 1, dtype="float32", always_2d=True)
            envelope = uniform_filter1d(np.max(np.abs(audio), axis=1),
                                        max(1, round(.0005 * sample_rate)))
            index = int(np.argmax(envelope))
            maximum = float(envelope[index])
            outer = envelope[np.abs(np.arange(len(envelope)) - index) > round(.010 * sample_rate)]
            isolated = bool(len(outer) and maximum > 1e-5 and
                            maximum > 4 * float(np.mean(envelope)) and
                            maximum > 3 * float(np.max(outer)))
            return (first + index) / sample_rate, isolated

        for role in ("head", "tail"):
            mapping = warp["input_output_beat_mapping"][role]
            for i, point in enumerate(mapping[1:-1], 1):
                source_centre = point["source_seconds"] - warp["trim_start_seconds"]
                actual_source, good_source = peak(source, source_centre)
                # Source attack offset is carried through the intended local rate.
                rate = ((mapping[i+1]["source_seconds"] - mapping[i-1]["source_seconds"]) /
                        (mapping[i+1]["output_seconds"] - mapping[i-1]["output_seconds"]))
                expected = point["output_seconds"] + (actual_source - source_centre) / rate
                actual, good_output = peak(rendered, expected)
                item = {"role": role, "beat": i, "expected_seconds": expected,
                        "observed_seconds": actual, "delay_seconds": actual - expected,
                        "grid_output_seconds": point["output_seconds"]}
                if good_source and good_output:
                    measured.append(item)
                else:
                    pending.append({"role": role, "beat": i, "reason": "no unambiguous isolated source/output attack"})
    return {"measured": measured, "unverified": pending,
            "maximum_measured_drift_seconds": max((abs(p["delay_seconds"]) for p in measured), default=None),
            "measurement": "isolated PCM amplitude-envelope peaks versus mapped source attacks; not a kick/stem detector",
            "all_interior_beats_verified": bool(measured and not pending)}


def compensate_backend_drift(warp: dict, drift: dict, sample_rate: int, *, strength: float = 1.0) -> None:
    """Precompensate measured backend latency, retaining the intended beat map.

    Render again from the original float clip, never stretch an already warped
    result. Missing or ambiguous attacks receive no new timing correction.
    """
    for role in ("head", "tail"):
        mapping = warp["input_output_beat_mapping"][role]
        measured = [p for p in drift["measured"] if p["role"] == role]
        if not mapping or not measured:
            continue
        times = np.array([mapping[0]["output_seconds"]] +
                         [p["grid_output_seconds"] for p in measured] +
                         [mapping[-1]["output_seconds"]])
        errors = np.array([0.] + [p["delay_seconds"] for p in measured] + [0.])
        for anchor in warp["anchors"]:
            if anchor["role"] == role + "_sync_control":
                intended = anchor.setdefault("intended_output_frame", anchor["output_frame"])
                adjustment = round(np.interp(intended / sample_rate, times, errors) * sample_rate * strength)
                anchor["output_frame"] -= adjustment
                anchor["output_seconds"] = anchor["output_frame"] / sample_rate
    if any(b["output_frame"] <= a["output_frame"] for a, b in zip(warp["anchors"], warp["anchors"][1:])):
        raise ValueError("Acoustic compensation would make the timing map nonmonotonic")
