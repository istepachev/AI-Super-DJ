"""Explicit, free-time transition effects for incompatible native tempos.

These effects do not time-stretch music, infer bars, or verify musical phrases.
Callers select the source regions and review the result before accepting it.
Only the supplied stereo overlap is processed; no whole-track allocation or
unbounded feedback delay is used.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt


class BridgeError(ValueError):
    """A requested bridge cannot preserve its declared signal constraints."""


def _positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BridgeError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise BridgeError(f"{name} must be a positive finite number")
    return number


def _audio(value: np.ndarray, name: str) -> np.ndarray:
    audio = np.asarray(value)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < 16:
        raise BridgeError(f"{name} must be stereo audio with at least 16 frames")
    if audio.dtype.kind != "f" or not np.all(np.isfinite(audio)):
        raise BridgeError(f"{name} must contain finite FLOAT samples")
    # Never mutate a caller's views, including read-only decoded source arrays.
    return np.array(audio, dtype=np.float32, copy=True)


def _ramp(frames: np.ndarray, start: int, end: int) -> np.ndarray:
    """Raised-cosine 0→1 ramp with exact, flat endpoints."""
    progress = np.clip((frames - start) / (end - start), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(math.pi * progress)).astype(np.float32)


def echo_bridge(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sample_rate: int,
    *,
    outgoing_bpm: float,
    incoming_bpm: float,
    overlap_beats: float = 8.0,
    echo_delay_beats: float = 0.5,
    dry_out_beats: float = 2.0,
    incoming_fade_beats: float = 1.0,
    echo_repeats: int = 6,
    echo_decay: float = 0.58,
    echo_gain: float = 0.35,
    highpass_hz: float = 400.0,
    peak_ceiling: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bridge a tempo conflict using a filtered, finite outgoing echo tail.

    Both arrays must cover the same region of ``overlap_beats`` incoming beats
    (within one sample). Their sample clocks stay unchanged. The outgoing dry
    signal fades during ``dry_out_beats`` incoming beats; only after that does
    the incoming dry signal enter, fading over ``incoming_fade_beats``. This
    avoids simultaneous full bass lines without pretending to separate stems.

    Delay taps use the outgoing native tempo, with no tempo/phase lock to the
    incoming music. The caller must explicitly permit this free-time technique.
    Source peaks must already fit ``peak_ceiling``: global level preparation
    belongs to the renderer so this overlap can join its neighbours seamlessly.
    A convex gain budget and attenuation of the filtered echo prevent sample
    clipping. This is not an oversampled true-peak measurement or a limiter.
    """
    sr = _positive(sample_rate, "sample_rate")
    if sr != int(sr) or not 1000 <= sr <= 384000:
        raise BridgeError("sample_rate must be an integer between 1000 and 384000")
    sr = int(sr)
    out_bpm = _positive(outgoing_bpm, "outgoing_bpm")
    in_bpm = _positive(incoming_bpm, "incoming_bpm")
    if not 20 <= out_bpm <= 400 or not 20 <= in_bpm <= 400:
        raise BridgeError("native BPM must lie between 20 and 400")
    beats = _positive(overlap_beats, "overlap_beats")
    delay_beats = _positive(echo_delay_beats, "echo_delay_beats")
    out_beats = _positive(dry_out_beats, "dry_out_beats")
    fade_beats = _positive(incoming_fade_beats, "incoming_fade_beats")
    cutoff = _positive(highpass_hz, "highpass_hz")
    ceiling = _positive(peak_ceiling, "peak_ceiling")
    if not 0 < ceiling <= 1:
        raise BridgeError("peak_ceiling must be at most 1.0")
    if not 0 < cutoff < sr / 2:
        raise BridgeError("highpass_hz must be below the Nyquist frequency")
    if not 0 < beats <= 32 or beats * 60 / in_bpm > 30:
        raise BridgeError("bridge must be at most 32 incoming beats and 30 seconds")
    if out_beats + fade_beats >= beats:
        raise BridgeError("dry fade and incoming fade must finish before the overlap ends")
    if (isinstance(echo_repeats, bool) or not isinstance(echo_repeats, (int, np.integer))
            or not 1 <= echo_repeats <= 16):
        raise BridgeError("echo_repeats must be an integer between 1 and 16")
    decay = _positive(echo_decay, "echo_decay")
    gain = _positive(echo_gain, "echo_gain")
    if decay >= 1 or gain > 1:
        raise BridgeError("echo_decay must be below 1 and echo_gain at most 1")

    a, b = _audio(outgoing, "outgoing"), _audio(incoming, "incoming")
    if a.shape != b.shape:
        raise BridgeError("outgoing and incoming must contain the same number of stereo frames")
    count = len(a)
    expected_frames = round(beats * 60 / in_bpm * sr)
    if abs(count - expected_frames) > 1:
        raise BridgeError("overlap frame count does not match overlap_beats at incoming native BPM")
    source_peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if source_peak > ceiling + 1e-7:
        raise BridgeError("source sample peak exceeds peak_ceiling; prepare whole-track headroom first")

    dry_end = round(out_beats * 60 / in_bpm * sr)
    incoming_full = round((out_beats + fade_beats) * 60 / in_bpm * sr)
    delay_frames = round(delay_beats * 60 / out_bpm * sr)
    if not 1 <= dry_end < incoming_full < count - 1:
        raise BridgeError("fade durations are too short or too long at this sample rate")
    if not 1 <= delay_frames < dry_end:
        raise BridgeError("echo delay must fit inside the outgoing dry fade")

    frames = np.arange(count, dtype=np.float64)
    dry_out = 1 - _ramp(frames, 0, dry_end)
    dry_in = _ramp(frames, dry_end, incoming_full)

    # Taper both capture boundaries before a causal high-pass. Delaying a
    # nonzero first sample would otherwise make each tap introduce a click.
    attack_frames = max(1, min(round(0.020 * sr), dry_end // 4))
    capture_gate = dry_out * _ramp(frames, 0, attack_frames)
    capture = a * capture_gate[:, None]
    filtered = sosfilt(butter(2, cutoff, "highpass", fs=sr, output="sos"),
                       capture, axis=0).astype(np.float32)
    # Gate residual IIR ringing at the capture boundary; repeats are finite.
    filtered *= dry_out[:, None]
    echo = np.zeros_like(a)
    taps = []
    for repeat in range(1, int(echo_repeats) + 1):
        offset = repeat * delay_frames
        if offset >= count - 1:
            break
        tap_gain = decay ** (repeat - 1)
        echo[offset:] += filtered[:-offset] * tap_gain
        taps.append({"repeat": repeat, "delay_frames": offset,
                     "delay_seconds": offset / sr, "gain": tap_gain})

    echo_raw_peak = float(np.max(np.abs(echo)))
    echo_attenuation = min(1.0, ceiling / max(echo_raw_peak, 1e-30))
    echo *= echo_attenuation
    echo_end = min(count - 1, dry_end + taps[-1]["delay_frames"])
    # A smooth attack and release also make the headroom allocated to the echo
    # continuous. At each boundary the unprocessed neighbour retains gain 1.
    release_start = min(dry_end, echo_end - 1)
    echo_envelope = gain * _ramp(frames, delay_frames,
                                 min(delay_frames + attack_frames, echo_end - 1))
    echo_envelope *= 1 - _ramp(frames, release_start, echo_end)
    out_gain = dry_out * (1 - echo_envelope)
    in_gain = dry_in * (1 - echo_envelope)
    mixed = (a * out_gain[:, None] + b * in_gain[:, None]
             + echo * echo_envelope[:, None]).astype(np.float32)

    manifest = {
        "mode": "free_time_echo_bridge",
        "timing_verification": "effect timing only; musical cue selection needs review",
        "beatmatched": False,
        "phase_alignment_verified": False,
        "phrase_alignment_verified": False,
        "human_auditioned": False,
        "time_stretch_applied": False,
        "meter_inferred": False,
        "sample_rate": sr,
        "frames": count,
        "duration_seconds": count / sr,
        "outgoing_native_bpm": out_bpm,
        "incoming_native_bpm": in_bpm,
        "overlap_incoming_beats": beats,
        "dry_out_end_seconds": dry_end / sr,
        "incoming_dry_start_seconds": dry_end / sr,
        "incoming_fade_complete_seconds": incoming_full / sr,
        "echo_end_seconds": echo_end / sr,
        "controls": {"echo_delay_outgoing_beats": delay_beats,
                     "dry_out_incoming_beats": out_beats,
                     "incoming_fade_beats": fade_beats,
                     "echo_gain": gain, "echo_decay": decay,
                     "echo_repeats_requested": int(echo_repeats),
                     "highpass_hz": cutoff, "peak_ceiling": ceiling},
        "echo_taps": taps,
        "headroom": {"method": "convex source gains; echo-only peak attenuation",
                     "source_sample_peak": source_peak,
                     "echo_raw_sample_peak": echo_raw_peak,
                     "echo_attenuation_gain": echo_attenuation,
                     "output_sample_peak": float(np.max(np.abs(mixed))),
                     "max_gain_sum": float(np.max(out_gain + in_gain + echo_envelope)),
                     "whole_region_normalization_gain": 1.0,
                     "true_peak_measured": False},
        "limitations": ["No vocal or drum stem separation",
                        "No tempo or phase synchronisation between the two rhythms",
                        "No verified bar or phrase alignment",
                        "No listening verification"],
    }
    return mixed, manifest
