"""Offline, bounded-memory DJ rendering from measured beat/downbeat grids.

The renderer uses an independently installed Rubber Band executable. It does
not implement or claim a proprietary harmonisation algorithm. Predicted bars
are reported as predictions; a group of sixteen bars is not proof of a musical
phrase. Source samples, exact warp anchors and transition decisions are retained
in the returned manifest so that timing can be checked independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


class RenderError(ValueError):
    """A plan cannot be rendered under the declared timing constraints."""


def validate_transition_policy(plan: Mapping[str, Any], *, allow_hard_cut: bool = False,
                               allow_unverified: bool = False,
                               tempo_conflict_policy: str = "review") -> None:
    """Reject unresolved boundaries before audio preparation or output creation.

    A tempo reset is a review condition, not permission to stop one deck and
    start the other. Deliberate cuts and unverified fades have separate opt-ins.
    """
    if tempo_conflict_policy not in {"review", "echo_bridge"}:
        raise RenderError("Tempo conflict policy must be review or echo_bridge")
    if allow_hard_cut or allow_unverified or tempo_conflict_policy == "echo_bridge":
        return
    for edge in plan.get("edges", []):
        if edge.get("tempo_reset") or not (edge.get("tempo") or {}).get("beatmatch_possible"):
            raise RenderError(
                f"{edge.get('from')} → {edge.get('to')}: transition needs review; "
                "automatic hard cuts are disabled. Choose a reviewed blend/bridge, "
                "or explicitly allow a deliberate hard cut or unverified fade."
            )


@dataclass
class _Window:
    times: np.ndarray
    bars: int
    meter: int
    target_bpm: float
    frames: int
    cue_provenance: dict[str, Any] | None = None


def _run(args: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(list(map(str, args)), cwd=cwd, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RenderError(f"Audio tool failed ({result.returncode}): {result.stderr[-3000:]}")
    return result.stderr


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RenderError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise RenderError(f"{name} must be a finite number")
    return number


def _array(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values if values is not None else [], dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0):
        raise RenderError(f"{name} must be a strictly increasing finite timing array")
    return result


def _grid(record: Mapping[str, Any], confidence: float) -> tuple[bool, str, np.ndarray, np.ndarray, int]:
    identity = str(record["id"])
    beats = _array(record.get("beats"), f"{identity}.beats")
    bars = _array(record.get("downbeats"), f"{identity}.downbeats")
    details = record.get("grid") or {}
    meter = int(details.get("meter") or details.get("meter_declared") or record.get("meter") or 0)
    if not len(bars):
        return False, "no measured/predicted downbeats", beats, bars, meter
    if meter != 4:
        return False, "a stable 4/4 meter was not established", beats, bars, meter
    if details.get("meter_consistency") is not None and float(details["meter_consistency"]) < 0.80:
        return False, "downbeat meter consistency is below 0.80", beats, bars, meter
    if (details.get("downbeat_alignment_max_seconds") or 0) > 0.10:
        return False, "a downbeat is more than 100 ms from its detected beat", beats, bars, meter
    if record.get("grid_status") not in {"reviewed", "model_predicted", "imported_unreviewed"}:
        return False, "unsupported or unverified grid status", beats, bars, meter
    if _finite(record.get("grid_confidence", 0), f"{identity}.grid_confidence") < confidence:
        return False, "timing confidence below the declared render threshold", beats, bars, meter
    if len(beats) < 2 or len(bars) < 10:
        return False, "not enough timing anchors for an eight-bar transition", beats, bars, meter
    return True, "usable predicted/reviewed bar grid", beats, bars, meter


def _beat_window(beats: np.ndarray, bars: np.ndarray, begin_bar: int,
                 count: int, bpm: float, sample_rate: int, meter: int = 4) -> _Window:
    """Retain real beat observations, including the two real bar endpoints."""
    if begin_bar < 0 or begin_bar + count >= len(bars):
        raise RenderError("Insufficient bars for this transition")
    start, end = bars[begin_bar], bars[begin_bar + count]
    first = int(np.argmin(np.abs(beats - start)))
    last = int(np.argmin(np.abs(beats - end)))
    tolerance = min(0.10, 60 / bpm * 0.20)
    if max(abs(beats[first] - start), abs(beats[last] - end)) > tolerance:
        raise RenderError("Downbeat does not coincide with its detected beat")
    if last - first != count * meter:
        raise RenderError("Detected beats and downbeats disagree on the number of beats in the window")
    bar_times = bars[begin_bar:begin_bar + count + 1]
    bar_beats = np.array([int(np.argmin(np.abs(beats - value))) for value in bar_times])
    if np.any(np.diff(bar_beats) != meter) or np.max(np.abs(beats[bar_beats] - bar_times)) > tolerance:
        raise RenderError("A phase correction or non-4/4 bar lies inside this transition window")
    times = beats[first:last + 1].copy()
    times[0], times[-1] = start, end
    if np.any(np.diff(times) <= 0):
        raise RenderError("Nonmonotonic beat timing inside a transition")
    frames = round(count * meter * 60 / bpm * sample_rate)
    return _Window(times, count, meter, bpm, frames)


def _local_validation_possible(record: Mapping[str, Any], grid: tuple[Any, ...]) -> bool:
    details = record.get("grid") or {}
    decoded = details.get("metrical_decode") or {}
    return bool(record.get("grid_status") == "model_predicted" and grid[4] == 4
                and "confidence below" in grid[1] and decoded.get("accepted") is True
                and float(decoded.get("phase_contrast") or 0) >= 0.25
                and len(details.get("neural_downbeat_probability_by_beat") or []) == len(grid[2])
                and len(details.get("neural_beat_probability_by_beat") or []) == len(grid[2]))


def _local_window_evidence(record: Mapping[str, Any], beats: np.ndarray,
                           window: _Window) -> dict[str, Any]:
    intervals = np.diff(window.times)
    span = min(8, len(window.times) - 1)
    period = float(np.median((window.times[span:] - window.times[:-span]) / span))
    jitter = float(np.median(np.abs(intervals - period)))
    stable_fraction = float(np.mean(np.abs(intervals - period) <= max(0.08, period * 0.15)))
    regularity = float(np.exp(-6 * max(0, jitter - 0.025) / max(period, 1e-6)) * stable_fraction)
    details = record.get("grid") or {}
    downbeat_probability = np.asarray(details.get("neural_downbeat_probability_by_beat") or [], dtype=float)
    beat_probability = np.asarray(details.get("neural_beat_probability_by_beat") or [], dtype=float)
    first = int(np.argmin(np.abs(beats - window.times[0])))
    last = int(np.argmin(np.abs(beats - window.times[-1])))
    evidence = {"whole_track_confidence": float(record.get("grid_confidence") or 0),
                "local_native_bpm": 60 / period, "local_timing_jitter_seconds": jitter,
                "stable_beat_interval_fraction": stable_fraction,
                "bar_beat_count_consistency": 1.0, "human_reviewed": False,
                "confidence_kind": "heuristic timing regularity and neural event separation; not a calibrated probability"}
    if len(downbeat_probability) == len(beats) and len(beat_probability) == len(beats):
        probabilities = downbeat_probability[first:last + 1]
        selected = np.arange(len(probabilities)) % window.meter == 0
        downbeat_mean = float(np.mean(probabilities[selected]))
        other_mean = float(np.mean(probabilities[~selected]))
        beat_mean = float(np.mean(beat_probability[first:last + 1]))
        confidence = regularity * (0.45 + 0.55 * beat_mean)
        evidence.update({"local_downbeat_probability_mean": downbeat_mean,
                         "local_other_beat_downbeat_probability_mean": other_mean,
                         "local_phase_contrast": downbeat_mean - other_mean,
                         "local_beat_probability_mean": beat_mean,
                         "local_timing_confidence": confidence,
                         "strong_local_neural_evidence": downbeat_mean >= 0.60 and
                          downbeat_mean - other_mean >= 0.25 and beat_mean >= 0.70})
    else:
        evidence.update({"local_timing_confidence": regularity,
                         "strong_local_neural_evidence": False})
    return evidence


def _choose_window(record: Mapping[str, Any], grid: tuple[Any, ...], *,
                   region: str, count: int, target_bpm: float, sample_rate: int,
                   max_trim_seconds: float, max_local_warp_change: float,
                   minimum_grid_confidence: float = 0.75) -> _Window:
    """Find a stable nearby complete bar window, avoiding meter corrections.

    Musical cue evidence has priority within bounded trims; unmarked bars remain
    fallbacks. No candidate is labelled a known phrase unless a separately
    reviewed cue declares that.
    """
    beats, downbeats, meter = grid[2], grid[3], grid[4]
    if region == "head":
        possible = [index for index, time in enumerate(downbeats)
                    if index + count < len(downbeats) and time <= max_trim_seconds]
    else:
        latest = len(downbeats) - 1 - count
        duration = float(record.get("duration", record.get("duration_seconds", downbeats[-1])))
        possible = [index for index in range(latest + 1)
                    if duration - downbeats[index + count] <= max_trim_seconds]
    candidates = []
    for begin in possible:
        try:
            window = _beat_window(beats, downbeats, begin, count, target_bpm, sample_rate, meter)
            local = np.diff(window.times) / (60 / target_bpm)
            if np.max(np.abs(local - 1)) > max_local_warp_change:
                continue
            evidence = _local_window_evidence(record, beats, window)
            if float(record.get("grid_confidence") or 0) < minimum_grid_confidence:
                if not _local_validation_possible(record, grid) or not evidence["strong_local_neural_evidence"]:
                    continue
                if evidence["local_timing_confidence"] < minimum_grid_confidence:
                    continue
        except RenderError:
            continue
        time = float(downbeats[begin])
        nearby = [cue for cue in record.get("cue_candidates", [])
                  if abs(float(cue.get("time", cue.get("seconds", -999))) - time) <= 0.05
                  and cue.get("kind", "intro" if region == "head" else "outro") == ("intro" if region == "head" else "outro")]
        cue = max(nearby, key=lambda item: float(item.get("confidence") or 0), default={})
        confidence = max(0.0, min(1.0, float(cue.get("confidence") or 0)))
        probability = max(0.0, min(1.0, float(cue.get("downbeat_model_probability") or 0)))
        novelty = max(0.0, min(1.0, float(cue.get("novelty") or 0)))
        sustained_value = cue.get("sustained_midband_proxy")
        sustained = max(0.0, min(1.0, float(sustained_value if sustained_value is not None else 1)))
        discarded = time if region == "head" else float(duration - downbeats[begin + count])
        score = (0.25 * discarded / max(max_trim_seconds, 1e-6) - 0.45 * confidence
                 - 0.15 * novelty - 0.15 * (1 - sustained) - 0.20 * probability)
        window.cue_provenance = {"region": region, "bar_index": begin,
                                 "cue_status": cue.get("status", "estimated_bar_grouping"),
                                 "cue_confidence": confidence, "downbeat_model_probability": probability,
                                 "novelty": novelty, "sustained_midband_proxy": sustained,
                                 "local_validation": evidence,
                                 "selection": "stable complete beat/bar window scored for structural cue evidence, within bounded trims",
                                 "musical_phrase_verified": bool(cue.get("reviewed") and cue.get("status") == "reviewed")}
        candidates.append((score, discarded, window))
    if not candidates:
        raise RenderError(f"No stable {count}-bar {region} window within the permitted trim/correction limits")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _select_window(record: Mapping[str, Any], grid: tuple[Any, ...], *,
                   locked_cue_seconds: float | None = None, **kwargs: Any) -> _Window:
    """An explicit boundary lock is mandatory, never merely a score bonus."""
    if locked_cue_seconds is None:
        return _choose_window(record, grid, **kwargs)
    cue = _finite(locked_cue_seconds, "locked cue")
    downbeats = grid[3]
    if not len(downbeats):
        raise RenderError("A locked cue requires a downbeat grid")
    index = int(np.argmin(np.abs(downbeats - cue)))
    if abs(float(downbeats[index]) - cue) > 1 / kwargs["sample_rate"] + 1e-6:
        raise RenderError("Locked cue is not on the supplied downbeat grid; review it instead of moving it")
    count = kwargs["count"]
    first = index if kwargs["region"] == "head" else index - count
    if first < 0 or first + count >= len(downbeats):
        raise RenderError("Locked cue does not leave a complete transition window")
    constrained = (*grid[:3], downbeats[first:first + count + 1], grid[4])
    window = _choose_window(record, constrained, **kwargs)
    window.cue_provenance["bar_index"] += first
    window.cue_provenance["mandatory_boundary"] = {
        "region": kwargs["region"], "seconds": cue,
        "human_reviewed": bool((record.get("grid") or {}).get("human_reviewed")),
        "selection": "explicit boundary lock; no relocation permitted",
    }
    return window


def _warp_anchors(record: Mapping[str, Any], *, trim_start: float,
                  trim_end: float, source_frames: int, sample_rate: int,
                  head: _Window | None, tail: _Window | None,
                  max_tempo_change: float, max_local_warp_change: float,
                  ramp_bars: int = 32) -> dict[str, Any]:
    """Make exact overlap beat anchors and smooth bar-wise body automation.

    Musical rate is the target BPM divided by the measured native BPM. Local
    map slopes additionally correct timing jitter and are bounded separately.
    """
    native_bpm = _finite(record.get("bpm"), "bpm")
    if native_bpm <= 0:
        raise RenderError("A warped track needs positive measured BPM")
    entry_rate = head.target_bpm / native_bpm if head else 1.0
    exit_rate = tail.target_bpm / native_bpm if tail else 1.0
    if max(abs(entry_rate - 1), abs(exit_rate - 1)) > max_tempo_change + 1e-6:
        raise RenderError("Planned musical tempo change exceeds the per-deck limit")
    duration = source_frames / sample_rate
    points: list[tuple[float, float, str]] = [(0.0, 0.0, "clip_start")]
    beat_maps: dict[str, list[dict[str, Any]]] = {"head": [], "tail": []}

    def add(source_seconds: float, target_seconds: float, role: str) -> None:
        if source_seconds < -1e-5 or source_seconds > duration + 0.02:
            raise RenderError("Warp anchor lies outside the decoded track")
        source_seconds = min(duration, max(0.0, source_seconds))
        if abs(source_seconds - points[-1][0]) < 1 / sample_rate:
            if abs(target_seconds - points[-1][1]) > 2 / sample_rate:
                raise RenderError("Two timing constraints disagree at a shared boundary")
            return
        if source_seconds <= points[-1][0] or target_seconds <= points[-1][1]:
            raise RenderError("Warp map must be strictly monotonic")
        points.append((source_seconds, target_seconds, role))

    if head:
        prefix = float(head.times[0] - trim_start)
        if prefix > 1 / sample_rate:
            add(prefix, prefix / entry_rate, "intro_before_overlap")
        head_base = points[-1][1]
        for index, source_time in enumerate(head.times):
            output = head_base + index * head.frames / sample_rate / (len(head.times) - 1)
            add(float(source_time - trim_start), output, "head_beat")
            beat_maps["head"].append({"beat": index, "source_seconds": float(source_time),
                                      "output_seconds": output})
        body_begin = float(head.times[-1] - trim_start)
    else:
        body_begin = 0.0
    body_end = float(tail.times[0] - trim_start) if tail else duration
    if body_end < body_begin - 1 / sample_rate:
        raise RenderError("Head and tail overlaps intersect; shorter transitions are required")
    downbeats = _array(record.get("downbeats"), "downbeats") - trim_start
    boundaries = [body_begin]
    boundaries.extend(float(value) for value in downbeats if body_begin + 1e-5 < value < body_end - 1e-5)
    if body_end > body_begin + 1 / sample_rate:
        boundaries.append(body_end)
    body_span = max(body_end - body_begin, 1e-6)
    ramp_seconds = min(ramp_bars * 4 * 60 / native_bpm, body_span / 2)

    def smooth(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3 - 2 * value)

    for left, right in zip(boundaries, boundaries[1:]):
        middle = (left + right) / 2
        entering = (entry_rate - 1) * (1 - smooth((middle - body_begin) / max(ramp_seconds, 1e-6)))
        leaving = (exit_rate - 1) * smooth((middle - (body_end - ramp_seconds)) / max(ramp_seconds, 1e-6))
        rate = 1 + entering + leaving
        add(right, points[-1][1] + (right - left) / rate, "body_bar_automation")
    if tail:
        tail_base = points[-1][1]
        for index, source_time in enumerate(tail.times):
            output = tail_base + index * tail.frames / sample_rate / (len(tail.times) - 1)
            add(float(source_time - trim_start), output, "tail_beat")
            beat_maps["tail"].append({"beat": index, "source_seconds": float(source_time),
                                      "output_seconds": output})
        suffix = duration - float(tail.times[-1] - trim_start)
        if suffix > 1 / sample_rate:
            add(duration, points[-1][1] + suffix / exit_rate, "outro_after_overlap")
    elif points[-1][0] < duration - 1 / sample_rate:
        add(duration, points[-1][1] + duration - points[-1][0], "clip_end")
    # Sample-frame constraints are the values actually passed to Rubber Band.
    anchors = []
    for source_seconds, output_seconds, role in points:
        source_frame = min(source_frames, round(source_seconds * sample_rate))
        output_frame = round(output_seconds * sample_rate)
        if anchors and source_frame == anchors[-1]["source_frame"]:
            continue
        anchors.append({"source_frame": source_frame, "output_frame": output_frame,
                        "source_seconds": source_frame / sample_rate + trim_start,
                        "output_seconds": output_frame / sample_rate, "role": role})
    if len(anchors) < 2 or anchors[-1]["source_frame"] != source_frames:
        raise RenderError("Warp map does not cover the complete decoded clip")
    rates = np.array([(b["source_frame"] - a["source_frame"]) /
                      (b["output_frame"] - a["output_frame"])
                      for a, b in zip(anchors, anchors[1:])])
    max_local = float(np.max(np.abs(rates - 1)))
    if max_local > max_local_warp_change + 1e-5:
        raise RenderError(f"Local timing correction {max_local:.1%} exceeds {max_local_warp_change:.1%}")
    return {"anchors": anchors, "output_frames": anchors[-1]["output_frame"],
            "entry_musical_rate": entry_rate, "exit_musical_rate": exit_rate,
            "max_musical_rate_change": max(abs(entry_rate - 1), abs(exit_rate - 1)),
            "max_local_warp_rate_change": max_local, "input_output_beat_mapping": beat_maps,
            "body_automation": "smooth entry-to-native and native-to-exit rates at measured bar anchors",
            "trim_start_seconds": trim_start, "trim_end_seconds": trim_end}


def _measure(path: Path, block: int = 262144) -> dict[str, float]:
    square_sum = 0.0
    count = 0
    peak = 0.0
    with sf.SoundFile(path) as source:
        for audio in source.blocks(blocksize=block, dtype="float32", always_2d=True):
            if not np.all(np.isfinite(audio)):
                raise RenderError("Audio contains nonfinite samples")
            square_sum += float(np.sum(audio.astype(np.float64) ** 2))
            count += audio.size
            peak = max(peak, float(np.max(np.abs(audio))))
    if not count:
        raise RenderError("Decoded track is empty")
    return {"rms_dbfs": 10 * math.log10(max(square_sum / count, 1e-24)), "sample_peak": peak}


def _copy_region(source: sf.SoundFile, target: sf.SoundFile, frames: int,
                 gain: float = 1.0, *, fade_in: bool = False, fade_out: bool = False) -> None:
    original = frames
    written = 0
    fade_frames = min(round(source.samplerate * 0.005), max(1, frames // 2))
    while frames:
        audio = source.read(min(frames, 262144), dtype="float32", always_2d=True)
        if not len(audio):
            raise RenderError("Unexpected end of PCM stream")
        audio *= gain
        indices = np.arange(written, written + len(audio))
        if fade_in:
            audio *= np.minimum(1.0, indices / fade_frames)[:, None]
        if fade_out:
            audio *= np.minimum(1.0, (original - 1 - indices) / fade_frames)[:, None]
        target.write(audio)
        written += len(audio)
        frames -= len(audio)


def _crossfade(outgoing: np.ndarray, incoming: np.ndarray, sample_rate: int,
               bass_hz: float = 180.0) -> np.ndarray:
    """Equal-power highs, complementary smooth bass exchange at the centre.

    Zero-phase filtering ensures low/high decomposition sums to the unchanged
    input. The overlap only (normally 8/16 bars) is loaded, never the whole set.
    """
    if outgoing.shape != incoming.shape or outgoing.ndim != 2 or outgoing.shape[1] != 2:
        raise RenderError("Crossfade needs equally sized stereo blocks")
    length = len(outgoing)
    if length < 2:
        return incoming.copy()
    position = np.linspace(0.0, 1.0, length, dtype=np.float64)
    a = np.cos(position * math.pi / 2).astype(np.float32)[:, None]
    b = np.sin(position * math.pi / 2).astype(np.float32)[:, None]
    if bass_hz <= 0 or length < 128:
        return outgoing * a + incoming * b
    filt = butter(4, bass_hz, fs=sample_rate, output="sos")
    low_out = sosfiltfilt(filt, outgoing, axis=0).astype(np.float32)
    low_in = sosfiltfilt(filt, incoming, axis=0).astype(np.float32)
    swap = np.clip((position - 0.35) / 0.30, 0.0, 1.0)
    swap = (swap * swap * (3 - 2 * swap)).astype(np.float32)[:, None]
    return (outgoing - low_out) * a + (incoming - low_in) * b + low_out * (1 - swap) + low_in * swap


def render_plan(plan: Mapping[str, Any], records: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
                prepared_files: Mapping[str, Any] | None, output_path: str | Path,
                work_dir: str | Path, *, ffmpeg: str = "ffmpeg", rubberband: str = "rubberband",
                rubberband_engine: str = "r2",
                sample_rate: int = 44100, transition_bars: int = 16,
                max_tempo_change: float = 0.06, max_local_warp_change: float = 0.15,
                minimum_grid_confidence: float = 0.75, max_intro_trim_seconds: float = 32.0,
                max_outro_trim_seconds: float = 45.0,
                strict_grid: bool = True, allow_unverified: bool = False,
                allow_hard_cut: bool = False,
                tempo_conflict_policy: str = "review", bridge_overlap_beats: float = 8.0,
                target_rms_dbfs: float = -20.0, target_lufs: float = -18.0,
                track_peak_ceiling: float = 0.50,
                max_gain_db: float = 4.0, bass_hz: float = 180.0,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Render a fixed or automatic order to stereo FLOAT WAV, without mastering.

    `prepared_files` maps track id (or source path) to a path, or to a dictionary
    with `path` and optional `provenance`. Processing must preserve audio timing.
    Strict mode refuses deficient grids. Opt-in `allow_unverified=True` allows
    explicit short unverified fades instead of calling them beatmatched blends.
    A musical BPM change limit and a separate microtiming correction limit are
    applied and reported. Output is never overwritten, and no audio is uploaded.
    Zero-overlap tempo resets require the separate `allow_hard_cut` opt-in.
    `echo_bridge` explicitly selects a native-tempo effect bridge for unresolved
    boundaries; it is not beatmatching. Edge `cue_locks` can mandate the incoming
    start and outgoing end in source seconds without claiming human verification.
    """
    if sample_rate < 8000 or transition_bars not in {8, 16, 32}:
        raise RenderError("Use a valid sample rate and 8, 16 or 32 transition bars")
    if rubberband_engine not in {"r2", "r3"}:
        raise RenderError("Rubber Band engine must be r2 or r3")
    if not 0 < max_tempo_change < 0.5 or not 0 < max_local_warp_change < 0.5:
        raise RenderError("Tempo limits must be positive fractions below 0.5")
    if not 0 < track_peak_ceiling < 1 or not 0 <= minimum_grid_confidence <= 1:
        raise RenderError("Invalid peak ceiling or grid confidence")
    bridge_overlap_beats = _finite(bridge_overlap_beats, "bridge overlap beats")
    if not 4 <= bridge_overlap_beats <= 32:
        raise RenderError("Echo bridge must span 4 to 32 incoming beats")
    by_id = {str(r["id"]): r for r in (records.values() if isinstance(records, Mapping) else records)}
    order = [str(item) for item in plan.get("order", [])]
    if not order or len(set(order)) != len(order) or set(order) != set(by_id):
        raise RenderError("The plan must use every unique analysis record exactly once")
    edges = list(plan.get("edges", []))
    if len(edges) != len(order) - 1:
        raise RenderError("The plan needs exactly one edge per adjacent pair")
    for i, edge in enumerate(edges):
        if str(edge.get("from")) != order[i] or str(edge.get("to")) != order[i + 1]:
            raise RenderError("Plan edges disagree with track order")
    validate_transition_policy(plan, allow_hard_cut=allow_hard_cut,
                               allow_unverified=allow_unverified,
                               tempo_conflict_policy=tempo_conflict_policy)
    directory = Path(work_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    grids = {identity: _grid(by_id[identity], minimum_grid_confidence) for identity in order}
    head: dict[str, _Window | None] = {identity: None for identity in order}
    tail: dict[str, _Window | None] = {identity: None for identity in order}
    reset_head: dict[str, _Window] = {}
    reset_tail: dict[str, _Window] = {}
    transitions: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, edge in enumerate(edges):
        outgoing, incoming = order[index:index + 2]
        ga, gb = grids[outgoing], grids[incoming]
        tempo = edge.get("tempo") or {}
        locks = edge.get("cue_locks") or {}
        if not isinstance(locks, Mapping):
            raise RenderError("cue_locks must be a mapping")
        in_lock, out_lock = locks.get("incoming_start_seconds"), locks.get("outgoing_end_seconds")
        reason = ""
        matched = bool(tempo.get("beatmatch_possible") and not edge.get("tempo_reset"))
        usable_for_search = (ga[0] or _local_validation_possible(by_id[outgoing], ga)) and (gb[0] or _local_validation_possible(by_id[incoming], gb))
        if (in_lock is not None or out_lock is not None) and not usable_for_search:
            raise RenderError(f"{outgoing} → {incoming}: mandatory cues require usable native grids")
        if not usable_for_search:
            reason = "; ".join(f"{identity}: {grid[1]}" for identity, grid in ((outgoing, ga), (incoming, gb)) if not grid[0])
            if strict_grid and not allow_unverified:
                raise RenderError(reason + "; import/review a grid or explicitly allow unverified fades")
            matched = False
        if matched and float(gb[3][0]) > max_intro_trim_seconds:
            reason = "first incoming downbeat would discard too much introductory audio"
            matched = False
        bars = min(transition_bars, 8 if "distant" in str(edge.get("harmonic_relation", "")) else transition_bars)
        target = tempo.get("target_bpm")
        if matched:
            try:
                target = _finite(target, "target_bpm")
                if target <= 0:
                    raise RenderError("Target BPM must be positive")
                if len(ga[3]) < bars * 2 + 2 or len(gb[3]) < bars * 2 + 2:
                    bars = 8
                last_window_error = None
                for window_bars in dict.fromkeys((bars, 8)):
                    try:
                        h = _select_window(by_id[incoming], gb, region="head", count=window_bars,
                                           locked_cue_seconds=in_lock,
                                           target_bpm=target, sample_rate=sample_rate,
                                           max_trim_seconds=max_intro_trim_seconds,
                                           max_local_warp_change=max_local_warp_change,
                                           minimum_grid_confidence=minimum_grid_confidence)
                        t = _select_window(by_id[outgoing], ga, region="tail", count=window_bars,
                                           locked_cue_seconds=out_lock,
                                           target_bpm=target, sample_rate=sample_rate,
                                           max_trim_seconds=max_outro_trim_seconds,
                                           max_local_warp_change=max_local_warp_change,
                                           minimum_grid_confidence=minimum_grid_confidence)
                        bars = window_bars
                        break
                    except RenderError as exc:
                        last_window_error = exc
                else:
                    raise last_window_error or RenderError("No suitable transition window")
                # Check timing correction early, before spending time rendering.
                for window in (h, t):
                    nominal = target / _finite(by_id[incoming if window is h else outgoing]["bpm"], "bpm")
                    if abs(nominal - 1) > max_tempo_change + 1e-6:
                        raise RenderError("Planned target BPM exceeds musical stretch limit")
                    local = np.diff(window.times) / (60 / target)
                    if np.max(np.abs(local - 1)) > max_local_warp_change:
                        raise RenderError("Irregular beat intervals exceed the local timing-correction limit")
                head[incoming], tail[outgoing] = h, t
            except RenderError as exc:
                reason = str(exc)
                matched = False
        if not matched:
            use_bridge = tempo_conflict_policy == "echo_bridge"
            if not allow_hard_cut and not allow_unverified and not use_bridge:
                raise RenderError(f"{outgoing} → {incoming}: {reason or 'no feasible common tempo'}; "
                                  "transition needs review; automatic hard cuts are disabled")
            mode = "unverified_short_crossfade"
            reset_cues: dict[str, Any] = {}
            if usable_for_search and (allow_hard_cut or use_bridge):
                try:
                    hreset = _select_window(by_id[incoming], gb, region="head", count=8,
                                            locked_cue_seconds=in_lock,
                                            target_bpm=_finite(by_id[incoming].get("bpm"), "native incoming BPM"),
                                            sample_rate=sample_rate, max_trim_seconds=max_intro_trim_seconds,
                                            max_local_warp_change=max_local_warp_change,
                                            minimum_grid_confidence=minimum_grid_confidence)
                    treset = _select_window(by_id[outgoing], ga, region="tail", count=8,
                                            locked_cue_seconds=out_lock,
                                            target_bpm=_finite(by_id[outgoing].get("bpm"), "native outgoing BPM"),
                                            sample_rate=sample_rate, max_trim_seconds=max_outro_trim_seconds,
                                            max_local_warp_change=max_local_warp_change,
                                            minimum_grid_confidence=minimum_grid_confidence)
                    reset_head[incoming], reset_tail[outgoing] = hreset, treset
                    mode = "free_time_echo_bridge" if use_bridge else "bar_tempo_reset"
                    reset_cues = {"incoming_cue": hreset.cue_provenance,
                                  "outgoing_cue": treset.cue_provenance,
                                  "bar_alignment": "locally validated predicted/reviewed native cue points; different tempos, no beatmatching"}
                    if use_bridge:
                        overlap_seconds = bridge_overlap_beats * 60 / float(by_id[incoming]["bpm"])
                        if overlap_seconds > float(treset.times[-1] - treset.times[0]):
                            raise RenderError("Echo overlap exceeds the validated outgoing region")
                        reset_cues["echo_parameters"] = {"outgoing_bpm": float(by_id[outgoing]["bpm"]),
                            "incoming_bpm": float(by_id[incoming]["bpm"]), "overlap_beats": bridge_overlap_beats}
                except RenderError as exc:
                    if use_bridge or in_lock is not None or out_lock is not None:
                        raise RenderError(f"{outgoing} → {incoming}: explicit bridge/cue could not be rendered: {exc}") from exc
                    reason = (reason + "; " if reason else "") + f"bar reset unavailable: {exc}"
            if use_bridge and mode != "free_time_echo_bridge":
                raise RenderError(f"{outgoing} → {incoming}: echo bridge requires reliable native grids and cues")
            if mode == "unverified_short_crossfade" and strict_grid and not allow_unverified:
                raise RenderError(f"{outgoing} → {incoming}: {reason or 'no reliable bar-aligned reset'}")
            warnings.append(f"{outgoing} → {incoming}: {reason or 'no feasible common tempo'}; {mode}, not beatmatched")
            overlap = (round(overlap_seconds * sample_rate) if mode == "free_time_echo_bridge"
                       else 0 if mode == "bar_tempo_reset" else round(0.20 * sample_rate))
            transitions.append({"from": outgoing, "to": incoming, "mode": mode,
                                "beatmatched": False, "bars": None, "target_bpm": None,
                                "overlap_frames": overlap, "reason": reason or "no feasible common tempo",
                                **reset_cues})
        else:
            transitions.append({"from": outgoing, "to": incoming, "mode": "beatmapped_bass_swap",
                                "beatmatched": True, "bars": bars, "beats": bars * 4,
                                "target_bpm": target, "overlap_frames": h.frames,
                                "outgoing_source_start": float(t.times[0]), "outgoing_source_end": float(t.times[-1]),
                                "incoming_source_start": float(h.times[0]), "incoming_source_end": float(h.times[-1]),
                                "outgoing_cue": t.cue_provenance, "incoming_cue": h.cue_provenance,
                                "bar_alignment": "predicted/reviewed downbeats, exact sample-frame map",
                                "phrase_alignment": "estimated bar grouping; musical phrase boundaries not guaranteed",
                                "human_reviewed": bool((by_id[outgoing].get("grid") or {}).get("human_reviewed") and
                                                       (by_id[incoming].get("grid") or {}).get("human_reviewed"))})
    clips: list[dict[str, Any]] = []
    prepared_files = prepared_files or {}
    for position, identity in enumerate(order):
        record = by_id[identity]
        source_path = Path(record["path"])
        prepared = prepared_files.get(identity, prepared_files.get(str(source_path), str(source_path)))
        preparation = prepared.get("provenance", {}) if isinstance(prepared, Mapping) else {}
        prepared_path = Path(prepared["path"] if isinstance(prepared, Mapping) else prepared).expanduser().resolve()
        if not prepared_path.is_file():
            raise FileNotFoundError(prepared_path)
        if progress:
            progress(f"Preparing track {position + 1}/{len(order)}")
        decoded = directory / f"track-{position + 1:02d}-decoded.wav"
        clip_input = directory / f"track-{position + 1:02d}-input.wav"
        clip_output = directory / f"track-{position + 1:02d}-warped.wav"
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(prepared_path),
              "-vn", "-ac", "2", "-ar", str(sample_rate), "-c:a", "pcm_f32le", str(decoded)])
        info = sf.info(decoded)
        duration = _finite(record.get("duration", record.get("duration_seconds")), "duration")
        if abs(info.duration - duration) > 0.05:
            raise RenderError("Prepared processing changed track timing; reanalyse that recording before rendering")
        ga = grids[identity]
        trim_start = float(head[identity].times[0]) if head[identity] else (float(reset_head[identity].times[0]) if identity in reset_head else 0.0)
        trim_end = float(tail[identity].times[-1]) if tail[identity] else (float(reset_tail[identity].times[-1]) if identity in reset_tail else info.duration)
        if trim_start > max_intro_trim_seconds:
            if strict_grid and not allow_unverified:
                raise RenderError("Incoming bar-aligned cut would discard too much introductory audio")
            trim_start = 0.0
            warnings.append(f"{identity}: introductory audio preserved; reset is no longer verified bar-aligned")
            transitions[position - 1]["mode"] = "unverified_short_crossfade"
            transitions[position - 1]["overlap_frames"] = round(0.20 * sample_rate)
        first_frame = round(trim_start * sample_rate)
        end_frame = min(info.frames, round(trim_end * sample_rate))
        source_frames = end_frame - first_frame
        if source_frames <= 0:
            raise RenderError("Invalid track trims")
        measurements = _measure(decoded)
        peak = measurements["sample_peak"]
        measured_lufs = (prepared.get("loudness_lufs") if isinstance(prepared, Mapping) else None)
        if measured_lufs is None:
            measured_lufs = record.get("loudness_lufs")
        matching_gain = target_rms_dbfs - measurements["rms_dbfs"]
        gain_method = "constant full-track RMS match, peak cap; no dynamics processing"
        if measured_lufs is not None and math.isfinite(float(measured_lufs)):
            measured_lufs = float(measured_lufs)
            matching_gain = target_lufs - measured_lufs
            measurements["loudness_lufs"] = measured_lufs
            gain_method = "constant measured integrated-LUFS match, peak cap; no dynamics processing"
        gain_db = min(max_gain_db, matching_gain,
                      20 * math.log10(track_peak_ceiling / max(peak, 1e-12)))
        gain = 10 ** (gain_db / 20)
        with sf.SoundFile(decoded) as original, sf.SoundFile(clip_input, "w", samplerate=sample_rate,
                                                           channels=2, subtype="FLOAT", format="WAV") as dest:
            original.seek(first_frame)
            _copy_region(original, dest, source_frames, gain)
        decoded.unlink()
        warp: dict[str, Any] | None = None
        if head[identity] or tail[identity]:
            warp = _warp_anchors(record, trim_start=trim_start, trim_end=trim_end,
                                 source_frames=source_frames, sample_rate=sample_rate,
                                 head=head[identity], tail=tail[identity], max_tempo_change=max_tempo_change,
                                 max_local_warp_change=max_local_warp_change)
            map_path = directory / f"track-{position + 1:02d}-timemap.txt"
            map_path.write_text("".join(f"{item['source_frame']} {item['output_frame']}\n" for item in warp["anchors"]), encoding="ascii")
            ratio = warp["output_frames"] / source_frames
            engine_args = ["--fast", "--detector-perc"] if rubberband_engine == "r2" else ["--fine"]
            _run([rubberband, *engine_args, "--centre-focus", "--ignore-clipping", "--quiet",
                  "--time", f"{ratio:.12f}", "--timemap", map_path.name,
                  clip_input.name, clip_output.name], cwd=directory)
            warped_info = sf.info(clip_output)
            if warped_info.subtype not in {"FLOAT", "DOUBLE"}:
                raise RenderError("Rubber Band did not preserve floating-point PCM")
            difference = warped_info.frames - warp["output_frames"]
            if abs(difference) > 128:
                raise RenderError(f"Rubber Band duration differs from the declared map by {difference} frames")
            if difference:
                fixed = directory / f"track-{position + 1:02d}-exact.wav"
                with sf.SoundFile(clip_output) as original, sf.SoundFile(fixed, "w", samplerate=sample_rate,
                                                                       channels=2, subtype="FLOAT") as dest:
                    _copy_region(original, dest, min(warp["output_frames"], warped_info.frames))
                    if difference < 0:
                        dest.write(np.zeros((-difference, 2), dtype=np.float32))
                fixed.replace(clip_output)
            clip_input.unlink()
        else:
            clip_input.replace(clip_output)
        clips.append({"id": identity, "audio_path": str(clip_output), "source_path": str(source_path),
                      "prepared_path": str(prepared_path), "preparation_provenance": preparation,
                      "source_duration_seconds": info.duration, "trim_start_seconds": first_frame / sample_rate,
                      "trim_end_seconds": end_frame / sample_rate,
                      "discarded_tail_seconds": (info.frames - end_frame) / sample_rate,
                      "level_measurements": measurements, "gain_db": gain_db,
                      "gain_method": gain_method,
                      "warp": warp, "grid_status": record.get("grid_status"),
                      "grid_confidence": record.get("grid_confidence"),
                      "human_reviewed": bool((record.get("grid") or {}).get("human_reviewed")),
                      "output_frames": sf.info(clip_output).frames,
                      "duration_seconds": sf.info(clip_output).frames / sample_rate})
    temporary = output.with_name(output.stem + ".partial.wav")
    if temporary.exists():
        raise FileExistsError(temporary)
    cursor = 0
    previous_tail: np.ndarray | None = None
    try:
        with sf.SoundFile(temporary, "w", samplerate=sample_rate, channels=2, subtype="FLOAT", format="WAV") as dest:
            for position, clip in enumerate(clips):
                incoming = transitions[position - 1]["overlap_frames"] if position else 0
                outgoing = transitions[position]["overlap_frames"] if position < len(transitions) else 0
                if incoming + outgoing >= clip["output_frames"]:
                    raise RenderError("Overlaps consume the entire track")
                clip["set_start_frame"] = cursor - incoming
                clip["set_start_seconds"] = (cursor - incoming) / sample_rate
                clip["start_seconds"] = clip["set_start_seconds"]
                with sf.SoundFile(clip["audio_path"]) as original:
                    if incoming:
                        block = original.read(incoming, dtype="float32", always_2d=True)
                        if previous_tail is None or len(block) != incoming:
                            raise RenderError("Missing overlap audio")
                        mode = transitions[position - 1]["mode"]
                        if mode == "free_time_echo_bridge":
                            from .bridges import echo_bridge
                            mixed, evidence = echo_bridge(previous_tail, block, sample_rate,
                                **transitions[position - 1]["echo_parameters"])
                            transitions[position - 1]["echo_processing"] = evidence
                            dest.write(mixed)
                        else:
                            dest.write(_crossfade(previous_tail, block, sample_rate,
                                                  bass_hz if mode == "beatmapped_bass_swap" else 0.0))
                        transitions[position - 1]["set_start_frame"] = cursor - incoming
                        transitions[position - 1]["set_end_frame"] = cursor
                        transitions[position - 1]["set_start_seconds"] = (cursor - incoming) / sample_rate
                        transitions[position - 1]["set_end_seconds"] = cursor / sample_rate
                        previous_tail = None
                    elif position:
                        transitions[position - 1].update({"set_start_frame": cursor, "set_end_frame": cursor,
                                                         "set_start_seconds": cursor / sample_rate,
                                                         "set_end_seconds": cursor / sample_rate})
                    body = clip["output_frames"] - incoming - outgoing
                    fade_in = not incoming
                    fade_out = not outgoing
                    _copy_region(original, dest, body, fade_in=fade_in, fade_out=fade_out)
                    if outgoing:
                        previous_tail = original.read(outgoing, dtype="float32", always_2d=True)
                        if len(previous_tail) != outgoing:
                            raise RenderError("Missing outgoing transition audio")
                cursor += clip["output_frames"] - incoming
        measurements = _measure(temporary)
        if sf.info(temporary).frames != cursor:
            raise RenderError("Rendered length differs from the segment accounting")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    warped = any(clip["warp"] is not None for clip in clips)
    used_modes = sorted({transition["mode"] for transition in transitions})
    overlap_descriptions = {
        "beatmapped_bass_swap": "equal-power high bands, complementary zero-phase low-band bass swap",
        "free_time_echo_bridge": "filtered finite echo with sequential dry-signal handoff at native tempos",
        "unverified_short_crossfade": "explicit short equal-power fade, no beatmatching",
        "bar_tempo_reset": "explicit zero-overlap cut with anti-click endpoint fades",
    }
    manifest = {"schema_version": 1, "renderer": "AI Super DJ sample-frame timemap renderer",
                "output_path": str(output), "sample_rate": sample_rate, "channels": 2,
                "subtype": "FLOAT", "duration_seconds": cursor / sample_rate, "total_frames": cursor,
                "track_count": len(order), "all_tracks_once": True, "order": order,
                "tracks": clips, "transitions": transitions, "measurements": measurements,
                "limits": {"max_musical_tempo_change": max_tempo_change,
                           "max_local_timing_warp_change": max_local_warp_change,
                           "minimum_grid_confidence": minimum_grid_confidence,
                           "max_intro_trim_seconds": max_intro_trim_seconds,
                           "max_outro_trim_seconds": max_outro_trim_seconds,
                           "strict_grid": strict_grid, "allow_unverified": allow_unverified,
                           "allow_hard_cut": allow_hard_cut, "tempo_conflict_policy": tempo_conflict_policy,
                           "bridge_overlap_beats": bridge_overlap_beats},
                "processing": {"time_stretch": (f"external Rubber Band {rubberband_engine.upper()}, pitch preserved, explicit source→target frame map"
                                                if warped else "none; native sample timing preserved"),
                               "time_stretch_executed": warped,
                               "rubberband_engine": rubberband_engine if warped else None,
                               "transition_modes": used_modes,
                               "overlap": "; ".join(overlap_descriptions[mode] for mode in used_modes) or "none; single track",
                               "low_band_cutoff_hz": bass_hz, "mastering": "none; this is a headroom premix",
                               "intermediate_format": "32-bit floating-point PCM, stereo"},
                "warnings": warnings + ["Predicted downbeats are not human-reviewed musical ground truth.",
                                         "Bar grouping alone does not prove correct musical phrase boundaries."]}
    for transition in transitions:
        transition["start_seconds"] = transition["set_start_seconds"]
        transition["duration_seconds"] = transition["overlap_frames"] / sample_rate
        if transition["beatmatched"]:
            out_clip = next(clip for clip in clips if clip["id"] == transition["from"])
            in_clip = next(clip for clip in clips if clip["id"] == transition["to"])
            out_beats = out_clip["warp"]["input_output_beat_mapping"]["tail"]
            in_beats = in_clip["warp"]["input_output_beat_mapping"]["head"]
            errors = [(out_clip["set_start_seconds"] + left["output_seconds"]) -
                      (in_clip["set_start_seconds"] + right["output_seconds"])
                      for left, right in zip(out_beats, in_beats)]
            transition["mapped_beat_alignment_max_error_seconds"] = max(abs(value) for value in errors)
            transition["alignment_measurement_kind"] = "mathematical map anchors; not a post-render acoustic measurement"
        else:
            out_clip = next(clip for clip in clips if clip["id"] == transition["from"])
            in_clip = next(clip for clip in clips if clip["id"] == transition["to"])
            transition["outgoing_source_cut_seconds"] = out_clip["trim_end_seconds"]
            transition["incoming_source_start_seconds"] = in_clip["trim_start_seconds"]
            if transition["mode"] == "free_time_echo_bridge":
                transition["selected_cues_are_audibility_guarantees"] = False
                transition["audible_handoff_note"] = (
                    "Selected source boundaries stay fixed; effect gain envelopes determine audible dry entry/exit. "
                    "The incoming dry signal is initially muted, then faded in; inspect echo_processing timing."
                )
    manifest_path = output.with_suffix(".render.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest
