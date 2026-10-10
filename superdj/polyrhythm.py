"""Exploratory rational rhythm overlaps, without claiming 1:1 beatmatching.

The incoming sample clock is the reference. A caller time-maps only the
outgoing track to the returned anchors. Filtering acts on whole frequency
bands; it is not source separation or a professionally reviewed groove.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


class PartialRhythmError(ValueError):
    pass


def rational_beat_anchors(outgoing_times, incoming_times, *, outgoing_cycle=4,
                          incoming_cycle=3, maximum_tempo_change=0.06,
                          maximum_timing_correction=0.15) -> dict[str, Any]:
    """Map four observed outgoing beats to three unmodified incoming beats.

    Arrays include the ending beat. Shared anchors recur every cycle; beat
    observations between them are interpolated on the incoming sample clock.
    No incoming resampling or octave reinterpretation is performed.
    """
    a, b = np.asarray(outgoing_times, dtype=float), np.asarray(incoming_times, dtype=float)
    for array in (a, b):
        if (array.ndim != 1 or len(array) < 5 or not np.all(np.isfinite(array))
                or np.any(np.diff(array) <= 0)):
            raise PartialRhythmError("Beat arrays must be increasing finite observations")
    if (isinstance(outgoing_cycle, bool) or isinstance(incoming_cycle, bool)
            or not isinstance(outgoing_cycle, int) or not isinstance(incoming_cycle, int)
            or not 1 <= outgoing_cycle <= 8 or not 1 <= incoming_cycle <= 8):
        raise PartialRhythmError("Cycle counts must be integers from 1 to 8")
    if outgoing_cycle == incoming_cycle:
        raise PartialRhythmError("Use the ordinary renderer for 1:1 beatmatching")
    limit = float(maximum_tempo_change)
    if not math.isfinite(limit) or not 0 <= limit < 0.5:
        raise PartialRhythmError("Tempo cap must be finite and below 0.5")
    outgoing_count, incoming_count = len(a) - 1, len(b) - 1
    if (outgoing_count % outgoing_cycle or incoming_count % incoming_cycle
            or outgoing_count // outgoing_cycle != incoming_count // incoming_cycle):
        raise PartialRhythmError("Windows must contain the same number of complete rational cycles")
    positions = np.arange(len(a)) * incoming_cycle / outgoing_cycle
    target = np.interp(positions, np.arange(len(b)), b - b[0])
    musical_rate = (a[-1] - a[0]) / target[-1]
    if abs(musical_rate - 1) > limit + 1e-9:
        raise PartialRhythmError("Rational overlap would exceed the outgoing tempo cap")
    timing_cap = float(maximum_timing_correction)
    if not math.isfinite(timing_cap) or not 0 <= timing_cap < 0.5:
        raise PartialRhythmError("Timing correction cap must be finite and below 0.5")
    local_correction = np.diff(a) / np.diff(target) / musical_rate
    maximum_correction = float(np.max(np.abs(local_correction - 1)))
    if maximum_correction > timing_cap + 1e-9:
        raise PartialRhythmError("Irregular beat observations exceed the local timing correction cap")
    shared = list(range(0, len(a), outgoing_cycle))
    return {
        "mode": "partial_rhythm_overlap",
        "beatmatched": False,
        "one_to_one_beatmatched": False,
        "ratio": {"outgoing_beats": outgoing_cycle, "incoming_beats": incoming_cycle},
        "incoming_time_stretch_applied": False,
        "outgoing_source_seconds": (a - a[0]).tolist(),
        "outgoing_output_seconds": target.tolist(),
        "shared_output_seconds": target[shared].tolist(),
        "outgoing_mean_rate": float(musical_rate),
        "maximum_local_timing_correction": maximum_correction,
        "incoming_duration_seconds": float(b[-1] - b[0]),
        "phase_alignment_verified": False,
        "phrase_alignment_verified": False,
        "human_auditioned": False,
        "alignment_evidence": "Mathematical anchors from supplied beats; not post-render acoustic proof",
    }


def partial_rhythm_mix(outgoing, incoming, sample_rate, *, shared_seconds,
                       bass_hz=180.0, gated_high_hz=2400.0,
                       gate_width_seconds=0.10, off_hit_floor=0.15,
                       mid_gate_floor=0.50, bass_swap_start=0.05, bass_swap_end=0.40,
                       peak_ceiling=1.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Continuous band mix: incoming clock intact, outgoing off-hits softened.

    Both tracks overlap throughout; no sequential mute or zero-overlap cut.
    Outgoing high frequencies are gated near shared rational anchors, while
    its middle frequencies retain a softer positive gate floor continuously.
    No processing is applied to the incoming audio's timing or pitch.
    """
    a, b = np.array(outgoing, dtype=np.float32, copy=True), np.array(incoming, dtype=np.float32, copy=True)
    if (a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape or len(a) < 64
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
        raise PartialRhythmError("Equal finite stereo arrays with at least 64 frames are required")
    sr = float(sample_rate)
    if not math.isfinite(sr) or sr != int(sr) or not 8000 <= sr <= 192000:
        raise PartialRhythmError("Sample rate must be an integer from 8000 to 192000")
    if not 0 < bass_hz < gated_high_hz < sr / 2:
        raise PartialRhythmError("Band cutoffs must lie inside the Nyquist range")
    if (not 0 < gate_width_seconds < 1 or not 0 < off_hit_floor < 1
            or not 0 < mid_gate_floor <= 1 or not 0 <= bass_swap_start < bass_swap_end <= 1
            or not 0 < peak_ceiling <= 1):
        raise PartialRhythmError("Invalid gate or headroom controls")
    duration = (len(a) - 1) / sr
    shared = np.asarray(shared_seconds, dtype=float)
    if (shared.ndim != 1 or len(shared) < 2 or not np.all(np.isfinite(shared))
            or np.any(np.diff(shared) <= 0) or shared[0] < -1 / sr
            or shared[-1] > duration + 2 / sr):
        raise PartialRhythmError("Shared anchors must lie within the overlap")
    clock = np.arange(len(a)) / sr
    progress = np.linspace(0, 1, len(a))
    outgoing_gain = np.cos(progress * math.pi / 2)
    incoming_gain = np.sin(progress * math.pi / 2)
    swap_position = np.clip((progress - bass_swap_start) / (bass_swap_end - bass_swap_start), 0, 1)
    bass_swap = 0.5 - 0.5 * np.cos(swap_position * math.pi)
    shared_gate = np.zeros(len(a))
    for anchor in shared:
        distance = np.abs(clock - anchor) / gate_width_seconds
        event = np.where(distance < 1, 0.5 + 0.5 * np.cos(np.minimum(distance, 1) * math.pi), 0)
        shared_gate = np.maximum(shared_gate, event)
    gate = off_hit_floor + (1 - off_hit_floor) * shared_gate
    mid_gate = mid_gate_floor + (1 - mid_gate_floor) * shared_gate
    low = butter(4, bass_hz, fs=sr, output="sos")
    mid = butter(4, gated_high_hz, fs=sr, output="sos")
    low_a = sosfiltfilt(low, a, axis=0)
    low_b = sosfiltfilt(low, b, axis=0)
    mid_a = sosfiltfilt(mid, a - low_a, axis=0)
    high_a = a - low_a - mid_a
    result = (low_a * (1 - bass_swap[:, None]) + low_b * bass_swap[:, None]
              + mid_a * (outgoing_gain * mid_gate)[:, None]
              + high_a * (outgoing_gain * gate)[:, None]
              + (b - low_b) * incoming_gain[:, None]).astype(np.float32)
    # Exact reconstruction at the joins avoids a normalization-induced jump.
    result[0], result[-1] = a[0], b[-1]
    peak = float(np.max(np.abs(result)))
    if peak > peak_ceiling + 1e-7:
        raise PartialRhythmError("Output exceeds sample headroom; level-match full clips before mixing")
    return result, {
        "mode": "partial_rhythm_band_overlap", "one_to_one_beatmatched": False,
        "incoming_time_stretch_applied": False,
        "sequential_mute_applied": False, "whole_region_normalization_gain": 1.0,
        "bass_handoff_start_seconds": float(duration * bass_swap_start),
        "bass_full_incoming_seconds": float(duration * bass_swap_end),
        "gated_band": "outgoing whole middle/high frequency bands; not drum stems",
        "controls": {"bass_hz": bass_hz, "gated_high_hz": gated_high_hz,
                     "gate_width_seconds": gate_width_seconds, "off_hit_floor": off_hit_floor,
                     "mid_gate_floor": mid_gate_floor,
                     "bass_swap_start": bass_swap_start, "bass_swap_end": bass_swap_end},
        "sample_peak": peak, "true_peak_measured": False,
        "human_auditioned": False,
        "limitations": ["No source separation", "No verified phrasing", "Not 1:1 beatmatching"],
    }
