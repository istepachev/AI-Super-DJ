"""Explicit outgoing-only vinyl-speed transitions with native incoming audio.

The selected outgoing exit deliberately changes pitch with playback speed.
This is neither pitch-preserving time stretch nor verified beat/phrase matching.
The incoming recording stays on its original sample clock throughout.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


class CreativeError(ValueError):
    """An effect cannot honour its declared timing or audio constraints."""


def _positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CreativeError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise CreativeError(f"{name} must be a positive finite number")
    return number


def _rate(value: Any) -> int:
    sr = _positive(value, "sample_rate")
    if sr != int(sr) or not 1000 <= sr <= 384000:
        raise CreativeError("sample_rate must be an integer between 1000 and 384000")
    return int(sr)


def _stereo(value: np.ndarray, name: str) -> np.ndarray:
    audio = np.asarray(value)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < 32:
        raise CreativeError(f"{name} must have at least 32 stereo frames")
    if audio.dtype.kind != "f" or not np.all(np.isfinite(audio)):
        raise CreativeError(f"{name} must contain finite FLOAT samples")
    return np.array(audio, dtype=np.float32, copy=True)


def _smooth(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def tape_exit_curve(frame_count: int, sample_rate: int, *, ramp_seconds: float = 12.0,
                    end_rate: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Return exact fractional source-frame positions and their provenance.

    ``end_rate`` is incoming native BPM / outgoing native BPM. Rates must be
    at most one: this sampler is an outgoing slowdown, with no up-speed alias
    case hidden in its implementation. Positions integrate a cubic smoothstep
    speed curve over output seconds; their finite differences retain its timing.
    ``source_frames_required`` includes the interpolation boundary sample.
    """
    sr = _rate(sample_rate)
    if (isinstance(frame_count, bool) or not isinstance(frame_count, (int, np.integer))
            or frame_count < 32):
        raise CreativeError("frame_count must be an integer of at least 32")
    duration = int(frame_count) / sr
    if duration > 30:
        raise CreativeError("creative overlap is limited to 30 output seconds")
    ratio = _positive(end_rate, "end_rate")
    if not 0.25 <= ratio <= 1.0:
        raise CreativeError("end_rate must be between 0.25 and 1.0; this is a slowdown only")
    requested = _positive(ramp_seconds, "ramp_seconds")
    ramp = min(requested, duration)

    def integral(t: np.ndarray | float) -> np.ndarray:
        values = np.asarray(t, dtype=np.float64)
        u = np.minimum(values, ramp)
        z = u / ramp
        source_seconds = u + (ratio - 1) * ramp * (z ** 3 - 0.5 * z ** 4)
        return source_seconds + np.maximum(values - ramp, 0.0) * ratio

    times = np.arange(int(frame_count), dtype=np.float64) / sr
    positions = integral(times) * sr
    consumption = float(integral(duration))
    required = math.ceil(consumption * sr) + 1
    sample_frames = sorted(set([0, int(frame_count) - 1, round(ramp * sr)] +
                               [round(t * sr) for t in np.arange(0, duration, 0.5)]))
    sample_frames = [frame for frame in sample_frames if frame < frame_count]
    manifest = {
        "method": "outgoing-only vinyl playback-speed slowdown",
        "output_frames": int(frame_count), "sample_rate": sr,
        "output_duration_seconds": duration,
        "ramp_output_seconds": ramp, "requested_ramp_output_seconds": requested,
        "ramp_shortened_to_overlap": requested > duration,
        "start_rate": 1.0, "end_rate": ratio,
        "source_frames_required": required, "used_source_frames": required,
        "source_consumption_seconds": consumption,
        "source_consumption_fractional_frames": consumption * sr,
        "source_window_interpolation_support_frames": required - consumption * sr,
        "outgoing_pitch_change_semitones": 12 * math.log2(ratio),
        "outgoing_pitch_preserved": False,
        "incoming_rate": 1.0, "incoming_pitch_change_semitones": 0.0,
        "curve": "cubic smoothstep speed in output time; analytically integrated source positions",
        "source_map": [{"output_frame": frame, "output_seconds": frame / sr,
                        "source_fractional_frame": float(positions[frame]),
                        "source_seconds": float(positions[frame] / sr),
                        "playback_rate": float(1 + (ratio - 1) * _smooth(np.array(frame / sr / ramp)))}
                       for frame in sample_frames],
        "beat_phase_alignment_verified": False, "phrase_alignment_verified": False,
        "human_auditioned": False,
    }
    return positions, manifest


def tape_exit_source_frames(frame_count: int, sample_rate: int, *, outgoing_bpm: float,
                            incoming_bpm: float, ramp_seconds: float = 12.0) -> int:
    """Exact source-window frame count expected by ``tape_exit_overlap``."""
    out_bpm = _positive(outgoing_bpm, "outgoing_bpm")
    in_bpm = _positive(incoming_bpm, "incoming_bpm")
    _, manifest = tape_exit_curve(frame_count, sample_rate, ramp_seconds=ramp_seconds,
                                  end_rate=in_bpm / out_bpm)
    return manifest["source_frames_required"]


def tape_speed_warp(outgoing: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Sample an outgoing slowdown using a bounded 32-tap Lanczos sinc kernel.

    For rates ≤1 the source bandwidth is shifted downward, so this operation
    has no rate-increase alias case. The windowed-sinc interpolation reduces
    fractional-sample interpolation error; it is not Rubber Band processing.
    Edge samples are extended at the two source boundaries. A caller fades
    the selected outgoing exit before its interpolation support runs out.
    """
    audio = _stereo(outgoing, "outgoing")
    query = np.asarray(positions, dtype=np.float64)
    if (query.ndim != 1 or len(query) < 32 or not np.all(np.isfinite(query)) or
            query[0] < 0 or np.any(np.diff(query) <= 0)):
        raise CreativeError("positions must be a strictly increasing finite nonnegative frame array")
    if float(np.max(np.diff(query))) > 1.0 + 1e-7:
        raise CreativeError("tape_speed_warp supports source steps at most one sample")
    if math.ceil(float(query[-1])) >= len(audio):
        raise CreativeError("outgoing source window is shorter than the declared position map")
    radius = 16
    offsets = np.arange(-radius + 1, radius + 1, dtype=np.int64)
    warped = np.empty((len(query), 2), dtype=np.float32)
    # No full [overlap × taps × channels] allocation; working kernels fit in
    # 4096-frame blocks regardless of the source collection's size.
    for begin in range(0, len(query), 4096):
        values = query[begin:begin + 4096]
        indices = np.floor(values).astype(np.int64)[:, None] + offsets[None, :]
        distances = values[:, None] - indices
        weights = np.sinc(distances) * np.sinc(distances / radius)
        weights[np.abs(distances) >= radius] = 0.0
        weights /= np.sum(weights, axis=1)[:, None]
        samples = audio[np.clip(indices, 0, len(audio) - 1)]
        warped[begin:begin + len(values)] = np.einsum("bt,btc->bc", weights, samples)
    if query[0] == 0:
        warped[0] = audio[0]
    return warped


def mix_tape_exit(outgoing_prewarped: np.ndarray, incoming: np.ndarray, sample_rate: int, *,
                  outgoing_release_seconds: float = 3.0, incoming_entry_seconds: float = 1.5,
                  bass_swap_start: float = 0.15, bass_swap_end: float = 0.50,
                  bass_hz: float = 180.0, outgoing_release_filter_hz: float = 1200.0,
                  ) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend both audible decks; alter incoming levels/bands, never its time.

    Upper-band gains are complementary equal-power gains with early incoming
    entry. Low-band gains make a complementary bass transfer. Only the outgoing
    upper band is progressively high-passed during its final release. No stem
    separation or global normalisation is applied. FLOAT peaks are reported;
    the renderer must allocate headroom and master the complete set afterwards.
    """
    sr = _rate(sample_rate)
    a = _stereo(outgoing_prewarped, "outgoing_prewarped")
    b = _stereo(incoming, "incoming")
    if a.shape != b.shape:
        raise CreativeError("prewarped outgoing and incoming must have the same stereo frame count")
    duration = len(b) / sr
    if duration > 30:
        raise CreativeError("creative overlap is limited to 30 output seconds")
    release = _positive(outgoing_release_seconds, "outgoing_release_seconds")
    entry = _positive(incoming_entry_seconds, "incoming_entry_seconds")
    if entry + release >= duration:
        raise CreativeError("incoming entry and outgoing release need distinct regions inside the overlap")
    swap_start = _positive(bass_swap_start, "bass_swap_start")
    swap_end = _positive(bass_swap_end, "bass_swap_end")
    if not 0 < swap_start < swap_end < 1:
        raise CreativeError("bass swap fractions must satisfy 0 < start < end < 1")
    cutoff = _positive(bass_hz, "bass_hz")
    release_cutoff = _positive(outgoing_release_filter_hz, "outgoing_release_filter_hz")
    if not 0 < cutoff < release_cutoff < sr / 2:
        raise CreativeError("bass and outgoing release cutoffs must increase below Nyquist")
    # Endpoint coordinates, rather than N/sr, give exact dry source gains at
    # the join boundaries. The resulting adjustment is at most one sample.
    t = np.arange(len(b), dtype=np.float64) / sr
    endpoint = (len(b) - 1) / sr
    release_begin = endpoint - release
    first_phase = 2 / math.pi * math.asin(0.55)
    release_phase = 2 / math.pi * math.asin(0.85)
    phase = first_phase * _smooth(t / entry)
    phase += (release_phase - first_phase) * _smooth((t - entry) / (release_begin - entry))
    phase += (1 - release_phase) * _smooth((t - release_begin) / release)
    out_upper_gain = np.cos(phase * math.pi / 2)
    in_upper_gain = np.sin(phase * math.pi / 2)
    bass_transfer = _smooth((t - duration * swap_start) / (duration * (swap_end - swap_start)))
    out_bass_gain = 1 - bass_transfer
    in_bass_gain = bass_transfer
    release_filter = _smooth((t - release_begin) / release)

    low_filter = butter(4, cutoff, fs=sr, output="sos")
    out_low = sosfiltfilt(low_filter, a, axis=0).astype(np.float32)
    in_low = sosfiltfilt(low_filter, b, axis=0).astype(np.float32)
    out_high = a - out_low
    in_high = b - in_low
    released_high = sosfiltfilt(butter(2, release_cutoff, "highpass", fs=sr, output="sos"),
                               out_high, axis=0).astype(np.float32)
    filtered_out_high = out_high * (1 - release_filter[:, None]) + released_high * release_filter[:, None]
    mixed = (filtered_out_high * out_upper_gain[:, None] + in_high * in_upper_gain[:, None] +
             out_low * out_bass_gain[:, None] + in_low * in_bass_gain[:, None]).astype(np.float32)
    # Floating-point low+high summation can differ by one ulp. Preserve the
    # two dry boundary samples exactly when joining adjacent untouched audio.
    mixed[0] = a[0]
    mixed[-1] = b[-1]
    both = (out_upper_gain > 1e-4) & (in_upper_gain > 1e-4)
    peak = float(np.max(np.abs(mixed)))
    manifest = {
        "mode": "outgoing_only_vinyl_native_incoming_overlap",
        "overlap_frames": len(b), "overlap_seconds": duration,
        "incoming_rate": 1.0, "incoming_pitch_change_semitones": 0.0,
        "incoming_time_stretched": False,
        "incoming_entry_seconds": entry,
        "incoming_upper_gain_after_entry": 0.55,
        "outgoing_upper_gain_after_incoming_entry": math.sqrt(1 - 0.55 ** 2),
        "outgoing_release_start_seconds": release_begin,
        "outgoing_release_seconds": release,
        "bass_swap_start_seconds": duration * swap_start,
        "bass_swap_end_seconds": duration * swap_end,
        "both_decks_nonzero_upper_gain_frames": int(np.sum(both)),
        "both_decks_nonzero_upper_gain_fraction": float(np.mean(both)),
        "upper_band_mix": "complementary equal-power gains; incoming enters immediately",
        "bass_mix": "complementary low-band transfer; no doubled full-gain bass",
        "outgoing_release_filter_hz": release_cutoff,
        "bass_hz": cutoff,
        "filter_timing": "zero-phase offline filters; no incoming sample-clock remap",
        "whole_overlap_normalization_gain": 1.0,
        "output_sample_peak": peak,
        "additional_headroom_needed_db_to_fit_sample_peak": max(0.0, 20 * math.log10(max(peak, 1e-30))),
        "true_peak_measured": False,
        "beatmatched_throughout_overlap": False,
        "phase_alignment_verified": False, "phrase_alignment_verified": False,
        "human_auditioned": False,
        "limitations": ["Outgoing exit deliberately changes pitch with playback speed",
                        "Earlier overlap contains different moving/native rhythms",
                        "No vocal or drum stem separation", "No listening verification"],
    }
    return mixed, manifest


def tape_exit_overlap(outgoing: np.ndarray, incoming: np.ndarray, sample_rate: int, *,
                      outgoing_bpm: float, incoming_bpm: float, ramp_seconds: float = 12.0,
                      outgoing_release_seconds: float = 3.0, incoming_entry_seconds: float = 1.5,
                      bass_swap_start: float = 0.15, bass_swap_end: float = 0.50,
                      bass_hz: float = 180.0, outgoing_release_filter_hz: float = 1200.0,
                      ) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare outgoing-only speed automation and mix a native incoming deck."""
    b = _stereo(incoming, "incoming")
    out_bpm = _positive(outgoing_bpm, "outgoing_bpm")
    in_bpm = _positive(incoming_bpm, "incoming_bpm")
    positions, curve = tape_exit_curve(len(b), sample_rate, ramp_seconds=ramp_seconds,
                                       end_rate=in_bpm / out_bpm)
    a = _stereo(outgoing, "outgoing")
    required = curve["source_frames_required"]
    if len(a) < required:
        raise CreativeError(f"outgoing needs {required} source frames for this declared exit curve")
    warped = tape_speed_warp(a[:required], positions)
    mixed, manifest = mix_tape_exit(
        warped, b, sample_rate, outgoing_release_seconds=outgoing_release_seconds,
        incoming_entry_seconds=incoming_entry_seconds, bass_swap_start=bass_swap_start,
        bass_swap_end=bass_swap_end, bass_hz=bass_hz,
        outgoing_release_filter_hz=outgoing_release_filter_hz)
    manifest.update({"outgoing_native_bpm": out_bpm, "incoming_native_bpm": in_bpm,
                     "outgoing_end_effective_bpm": in_bpm,
                     "outgoing_speed_curve": curve,
                     "outgoing_sampler": "32-tap Lanczos-windowed sinc; bounded 4096-frame blocks; edge extension",
                     "rubberband_used": False,
                     "unused_provided_outgoing_frames": len(a) - required})
    return mixed, manifest
