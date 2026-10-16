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
                               tempo_conflict_policy: str = "auto") -> None:
    """Reject unresolved boundaries before audio preparation or output creation.

    A tempo reset is a review condition, not permission to stop one deck and
    start the other. Deliberate cuts and unverified fades have separate opt-ins.
    """
    if tempo_conflict_policy not in {"auto", "sync", "adaptive", "shared_tempo", "review", "echo_bridge"}:
        raise RenderError("Tempo conflict policy must be auto, sync, adaptive, shared_tempo, review or echo_bridge")
    for edge in plan.get("edges", []):
        policy = edge.get("transition_policy", tempo_conflict_policy)
        if "transition_policy" in edge and policy not in {"auto", "sync", "review"}:
            raise RenderError("Edge transition_policy must be auto, sync or review")
        if allow_hard_cut or allow_unverified or policy in {"auto", "sync", "adaptive", "shared_tempo", "echo_bridge"}:
            continue
        if edge.get("tempo_reset") or not (edge.get("tempo") or {}).get("beatmatch_possible"):
            raise RenderError(
                f"{edge.get('from')} → {edge.get('to')}: transition needs review; "
                "automatic hard cuts are disabled. Choose a reviewed blend/bridge, "
                "or explicitly allow a deliberate hard cut or unverified fade."
            )


@dataclass
class _Window:
    times: np.ndarray
    bars: float
    meter: int
    target_bpm: float
    frames: int
    cue_provenance: dict[str, Any] | None = None
    clock: dict[str, Any] | None = None


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
            # Microtiming correction is independent of the musical BPM change.
            local = np.diff(window.times) / (60 / _finite(record.get("bpm"), "native BPM"))
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
                  if abs(float(cue.get("time", cue.get("seconds", -999))) - time) <= 0.05]
        cue = max(nearby, key=lambda item: float(item.get("confidence") or 0), default={})
        confidence = max(0.0, min(1.0, float(cue.get("confidence") or 0)))
        probability = max(0.0, min(1.0, float(cue.get("downbeat_model_probability") or 0)))
        novelty = max(0.0, min(1.0, float(cue.get("novelty") or 0)))
        sustained_value = cue.get("sustained_midband_proxy")
        sustained = max(0.0, min(1.0, float(sustained_value if sustained_value is not None else 1)))
        vocal_risk = cue.get("vocal_activity")
        if vocal_risk is not None:
            sustained = max(sustained, float(vocal_risk))
        discarded = time if region == "head" else float(duration - downbeats[begin + count])
        score = (0.25 * discarded / max(max_trim_seconds, 1e-6) - 0.45 * confidence
                 - 0.15 * novelty - 0.15 * (1 - sustained) - 0.20 * probability)
        window.cue_provenance = {"region": region, "bar_index": begin,
                                 "cue_status": cue.get("status", "estimated_bar_grouping"),
                                 "cue_confidence": confidence, "downbeat_model_probability": probability,
                                 "novelty": novelty, "sustained_midband_proxy": sustained,
                                 "method": cue.get("method", "grid-only fallback"),
                                 "cue_detr_evidence": cue.get("cue_detr_evidence", []),
                                 "vocal_activity": vocal_risk,
                                 "local_validation": evidence,
                                 "selection": "stable complete beat/bar window scored for structural cue evidence, within bounded trims",
                                 "musical_phrase_verified": bool(cue.get("reviewed") and cue.get("status") == "reviewed")}
        candidates.append((score, discarded, window))
    if not candidates:
        raise RenderError(f"No stable {count}-bar {region} window within the permitted trim/correction limits")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _select_window(record: Mapping[str, Any], grid: tuple[Any, ...], *,
                   locked_cue_seconds: float | None = None, beat_count: int | None = None,
                   **kwargs: Any) -> _Window:
    """An explicit boundary lock is mandatory, never merely a score bonus."""
    if beat_count is not None:
        if locked_cue_seconds is None:
            raise RenderError("A native beat-count exception requires an explicit boundary")
        cue = _finite(locked_cue_seconds, "locked cue")
        duration = float(record.get("duration", record.get("duration_seconds", grid[2][-1])))
        discarded = cue if kwargs["region"] == "head" else duration - cue
        if discarded < 0 or discarded > kwargs["max_trim_seconds"] + 1e-6:
            raise RenderError("Native beat-count boundary exceeds permitted trim limits")
        beats = grid[2]
        index = int(np.argmin(np.abs(beats - cue)))
        if abs(float(beats[index]) - cue) > 1 / kwargs["sample_rate"] + 1e-6:
            raise RenderError("Locked cue is not on the supplied beat grid; review it instead of moving it")
        first = index if kwargs["region"] == "head" else index - beat_count
        if first < 0 or first + beat_count >= len(beats):
            raise RenderError("Locked cue does not leave a complete transition window")
        times = beats[first:first + beat_count + 1]
        if np.max(np.abs(np.diff(times) / (60 / record["bpm"]) - 1)) > kwargs["max_local_warp_change"]:
            raise RenderError("Native beat-count exception exceeds timing correction limits")
        window = _Window(times, beat_count / grid[4], grid[4], kwargs["target_bpm"],
                         round(beat_count * 60 / kwargs["target_bpm"] * kwargs["sample_rate"]))
        window.cue_provenance = {"region": kwargs["region"], "cue_status": "explicit_beat_boundary",
            "mandatory_boundary": {"seconds": cue, "human_reviewed": bool((record.get("grid") or {}).get("human_reviewed"))},
            "musical_phrase_verified": False, "native_exception_beats": beat_count}
        return window
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
                  tempo_ramp_seconds: float = 15.0) -> dict[str, Any]:
    """Make exact overlap beat anchors and body ramps timed in output seconds.

    Musical rate is the target BPM divided by the measured native BPM. Local
    map slopes additionally correct timing jitter and are bounded separately.
    Outside each overlap a cubic smoothstep changes the musical rate between
    the overlap tempo and native tempo over ``tempo_ramp_seconds`` of rendered
    audio. Its integral determines source consumption, rather than treating
    source seconds or a fixed number of bars as output seconds. Short bodies
    proportionally shorten both ramps and explicitly report their durations.
    """
    native_bpm = _finite(record.get("bpm"), "bpm")
    if native_bpm <= 0:
        raise RenderError("A warped track needs positive measured BPM")
    requested_ramp = _finite(tempo_ramp_seconds, "tempo_ramp_seconds")
    if requested_ramp <= 0:
        raise RenderError("tempo_ramp_seconds must be positive")
    entry_rate = (head.clock["end_bpm"] if head and head.clock else head.target_bpm) / native_bpm if head else 1.0
    exit_rate = (tail.clock["start_bpm"] if tail and tail.clock else tail.target_bpm) / native_bpm if tail else 1.0
    musical_change = max([abs(entry_rate - 1), abs(exit_rate - 1)] +
                         [abs(w.clock[k] / native_bpm - 1) for w in (head, tail)
                          if w and w.clock for k in ("start_bpm", "end_bpm")])
    if musical_change > max_tempo_change + 1e-5:
        raise RenderError("Planned musical tempo change exceeds the per-deck limit")
    duration = source_frames / sample_rate
    points: list[tuple[float, float, str]] = [(0.0, 0.0, "clip_start")]
    beat_maps: dict[str, list[dict[str, Any]]] = {"head": [], "tail": []}

    def add(source_seconds: float, target_seconds: float, role: str) -> None:
        if source_seconds < -max(1e-5, 1 / sample_rate + 1e-6) or source_seconds > duration + 0.02:
            raise RenderError("Warp anchor lies outside the decoded track")
        source_seconds = min(duration, max(0.0, source_seconds))
        if abs(source_seconds - points[-1][0]) < 1 / sample_rate:
            if abs(target_seconds - points[-1][1]) > 2 / sample_rate:
                raise RenderError("Two timing constraints disagree at a shared boundary")
            return
        if source_seconds <= points[-1][0] or target_seconds <= points[-1][1]:
            raise RenderError("Warp map must be strictly monotonic")
        points.append((source_seconds, target_seconds, role))

    def overlap(window: _Window, base: float, role: str) -> None:
        if window.clock:
            from .sync import deck_controls
            sources, outputs = deck_controls(window.times, window.clock, sample_rate)
            for source_time, output_time in zip(sources, outputs):
                add(float(source_time - trim_start), base + float(output_time), role + "_sync_control")
            beat_outputs = np.asarray(window.clock["beat_output_frames"]) / sample_rate
        else:
            beat_outputs = np.linspace(0, window.frames / sample_rate, len(window.times))
            for source_time, output_time in zip(window.times, beat_outputs):
                add(float(source_time - trim_start), base + float(output_time), role + "_beat")
        for index, (source_time, output_time) in enumerate(zip(window.times, beat_outputs)):
            beat_maps[role].append({"beat": index, "source_seconds": float(source_time),
                                    "output_seconds": base + float(output_time)})

    if head:
        prefix = float(head.times[0] - trim_start)
        if prefix > 1 / sample_rate:
            add(prefix, prefix / entry_rate, "intro_before_overlap")
        head_base = points[-1][1]
        overlap(head, head_base, "head")
        body_begin = float(head.times[-1] - trim_start)
    else:
        body_begin = 0.0
    body_end = float(tail.times[0] - trim_start) if tail else duration
    if body_end < body_begin - 1 / sample_rate:
        raise RenderError("Head and tail overlaps intersect; shorter transitions are required")
    downbeats = _array(record.get("downbeats"), "downbeats") - trim_start
    body_span = max(0.0, body_end - body_begin)
    body_output_base = points[-1][1]
    entry_changes = abs(entry_rate - 1.0) > 1e-5
    exit_changes = abs(exit_rate - 1.0) > 1e-5
    if body_span <= 1 / sample_rate and (entry_changes or exit_changes):
        raise RenderError("No body remains for a continuous tempo ramp outside the overlaps")
    # ds/dt is the musical rate. A smoothstep has mean 1/2, hence a ramp
    # consumes T*(r0+r1)/2 source seconds in exactly T output seconds.
    required_source = requested_ramp * (
        (entry_rate + 1) / 2 if entry_changes else 0.0) + requested_ramp * (
        (1 + exit_rate) / 2 if exit_changes else 0.0)
    ramp_scale = min(1.0, body_span / required_source) if required_source else 1.0
    effective_ramp = requested_ramp * ramp_scale
    segments: list[dict[str, Any]] = []
    body_source_cursor, body_output_cursor = body_begin, body_output_base

    def segment(role: str, output_duration: float, start_rate: float, end_rate: float) -> None:
        nonlocal body_source_cursor, body_output_cursor
        if output_duration <= 1e-10:
            return
        source_duration = output_duration * (start_rate + end_rate) / 2
        segments.append({"role": role, "source_start": body_source_cursor,
                         "source_end": body_source_cursor + source_duration,
                         "output_start": body_output_cursor,
                         "output_end": body_output_cursor + output_duration,
                         "start_rate": start_rate, "end_rate": end_rate})
        body_source_cursor += source_duration
        body_output_cursor += output_duration

    if entry_changes:
        segment("body_entry_tempo_ramp", effective_ramp, entry_rate, 1.0)
    native_source = max(0.0, body_span - required_source * ramp_scale)
    segment("body_native_tempo", native_source, 1.0, 1.0)
    if exit_changes:
        segment("body_exit_tempo_ramp", effective_ramp, 1.0, exit_rate)
    # Preserve the selected boundary exactly despite floating-point summation.
    if segments:
        segments[-1]["source_end"] = body_end

    def source_progress(output_progress: float, item: Mapping[str, Any]) -> float:
        length = item["output_end"] - item["output_start"]
        u = min(length, max(0.0, output_progress))
        z = u / length
        return item["start_rate"] * u + (item["end_rate"] - item["start_rate"]) * length * (
            z ** 3 - 0.5 * z ** 4)

    def output_progress(source_offset: float, item: Mapping[str, Any]) -> float:
        length = item["output_end"] - item["output_start"]
        if abs(item["end_rate"] - item["start_rate"]) <= 1e-12:
            return source_offset / item["start_rate"]
        low, high = 0.0, length
        for _ in range(48):
            middle = (low + high) / 2
            if source_progress(middle, item) < source_offset:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    ramp_reports = []
    for item in segments:
        role = item["role"]
        length = item["output_end"] - item["output_start"]
        candidates = [(item["source_start"], item["output_start"], role),
                      (item["source_end"], item["output_end"], role)]
        is_ramp = role != "body_native_tempo"
        if is_ramp:
            # Measured bar anchors alone can be several seconds apart. Dense
            # output-time controls keep the finite CLI map close to the smooth
            # curve even during a large BPM change; the overlap stays exact.
            # Leave 20 ms spacing margin when a nearby measured downbeat
            # replaces a dense control below. Sub-sample control intervals
            # could otherwise turn rounding into large spurious rate jumps.
            steps = max(1, math.ceil(length / 0.23))
            for u in np.linspace(0.0, length, steps + 1)[1:-1]:
                candidates.append((item["source_start"] + source_progress(float(u), item),
                                   item["output_start"] + float(u), role))
        for value in downbeats:
            if item["source_start"] + 1e-5 < value < item["source_end"] - 1e-5:
                u = output_progress(float(value - item["source_start"]), item)
                candidates.append((float(value), item["output_start"] + u, role))
        sorted_candidates = sorted(candidates)
        retained = [sorted_candidates[0]]
        last = sorted_candidates[-1]
        for candidate in sorted_candidates[1:-1]:
            if (candidate[1] - retained[-1][1] >= 0.020 and
                    last[1] - candidate[1] >= 0.020):
                retained.append(candidate)
        retained.append(last)
        for source_seconds, output_seconds, anchor_role in retained:
            add(source_seconds, output_seconds, anchor_role)
        if is_ramp:
            first_output = round(item["output_start"] * sample_rate) / sample_rate
            last_output = round(item["output_end"] * sample_rate) / sample_rate
            ramp_reports.append({
                "role": "entry" if role == "body_entry_tempo_ramp" else "exit",
                "from_bpm": native_bpm * item["start_rate"],
                "to_bpm": native_bpm * item["end_rate"],
                "output_start_seconds": first_output,
                "output_end_seconds": last_output,
                "output_duration_seconds": last_output - first_output,
                "requested_output_duration_seconds": requested_ramp,
                "source_start_seconds": item["source_start"] + trim_start,
                "source_end_seconds": item["source_end"] + trim_start,
                "shortened_for_available_body": ramp_scale < 1 - 1e-12,
                "curve": "cubic smoothstep musical rate in output time; integrated to source time",
                "time_map_approximation": "piecewise-linear anchors at most 0.25 output seconds apart",
            })
    if tail:
        tail_base = points[-1][1]
        overlap(tail, tail_base, "tail")
        suffix = duration - float(tail.times[-1] - trim_start)
        if suffix > 1 / sample_rate:
            add(duration, points[-1][1] + suffix / exit_rate, "outro_after_overlap")
    elif points[-1][0] < duration - 1 / sample_rate:
        add(duration, points[-1][1] + duration - points[-1][0], "clip_end")
    # Sample-frame constraints are the values actually passed to Rubber Band.
    anchors = []
    reference_points = []
    for source_seconds, output_seconds, role in points:
        source_frame = min(source_frames, round(source_seconds * sample_rate))
        output_frame = round(output_seconds * sample_rate)
        if anchors and source_frame == anchors[-1]["source_frame"]:
            continue
        if anchors and output_frame <= anchors[-1]["output_frame"]:
            # Two very close mathematical controls can quantise to one output
            # frame. Retain the final clip boundary rather than a zero-length
            # CLI interval; ordinary controls can safely be coalesced.
            if source_frame == source_frames:
                anchors.pop()
                reference_points.pop()
            else:
                continue
        anchors.append({"source_frame": source_frame, "output_frame": output_frame,
                        "source_seconds": source_frame / sample_rate + trim_start,
                        "output_seconds": output_frame / sample_rate, "role": role})
        reference_points.append((source_seconds, output_seconds))
    if len(anchors) < 2 or anchors[-1]["source_frame"] != source_frames:
        raise RenderError("Warp map does not cover the complete decoded clip")
    rates = np.array([(b["source_frame"] - a["source_frame"]) /
                      (b["output_frame"] - a["output_frame"])
                      for a, b in zip(anchors, anchors[1:])])
    max_local = float(np.max(np.abs(rates - 1)))
    timing_corrections = []
    for index, rate in enumerate(rates):
        middle = (reference_points[index][0] + reference_points[index + 1][0]) / 2
        # Only measured overlap timing is regularised. Body time maps contain
        # deliberate musical automation, not a correction to observed beats.
        if head and middle < float(head.times[-1] - trim_start) - 1e-8:
            if not head.clock:
                timing_corrections.append(abs(float(rate) / entry_rate - 1))
        elif tail and middle > float(tail.times[0] - trim_start) + 1e-8:
            if not tail.clock:
                timing_corrections.append(abs(float(rate) / exit_rate - 1))
    for window in (head, tail):
        if window and window.clock:
            timing_corrections.extend(np.abs(np.diff(window.times) / (60 / native_bpm) - 1).tolist())
    max_timing = max(timing_corrections, default=0.0)
    if max_timing > max_local_warp_change + 1e-5:
        raise RenderError(f"Local timing correction {max_timing:.1%} exceeds {max_local_warp_change:.1%}")
    quantization_error = max(
        max(abs(anchor["source_frame"] / sample_rate - reference[0]),
            abs(anchor["output_seconds"] - reference[1]))
        for anchor, reference in zip(anchors, reference_points))
    result={"anchors": anchors, "output_frames": anchors[-1]["output_frame"],
            "entry_musical_rate": entry_rate, "exit_musical_rate": exit_rate,
            "max_musical_rate_change": musical_change,
            "max_local_warp_rate_change": max_local, "input_output_beat_mapping": beat_maps,
            "max_timing_correction_change": max_timing,
            "timing_correction_measurement": "constant overlap slope divided by requested rate, or source beat spacing versus native period for SYNC; body contains musical automation only",
            "max_anchor_quantization_error_seconds": quantization_error,
            "body_automation": "output-time cubic smoothstep entry-to-native and native-to-exit ramps; exact overlap beat anchors",
            "tempo_ramp_seconds_requested": requested_ramp,
            "tempo_ramps": ramp_reports,
            "tempo_ramps_shortened_for_available_body": bool(ramp_scale < 1 - 1e-12),
            "trim_start_seconds": trim_start, "trim_end_seconds": trim_end}
    from .timing_safety import initialize_rate_reference,audit_warp_rates
    initialize_rate_reference(result,sample_rate,native_bpm=native_bpm,
        windows=[(role,w.clock) for role,w in (('head',head),('tail',tail)) if w],limit=max_local_warp_change)
    audit_warp_rates(result,sample_rate,stage='initial musical map')
    return result


def _validate_playback_budget(clip, section, sample_rate):
    """Enforce the full duration of timed PCM, also for legacy solo plans."""
    maximum = float(section.get('maximum_play_seconds', 300.))
    if not np.isfinite(maximum) or maximum <= 0:
        raise RenderError('Full playback maximum must be finite and positive')
    if maximum > 300 and section.get('allow_long_play') is not True:
        raise RenderError('Full playback above five minutes requires explicit long-play permission in the plan')
    if clip['output_frames']/sample_rate > maximum + 1/sample_rate:
        raise RenderError(f"{clip['id']}: full rendered playback exceeds {maximum:g}s including both overlaps; replan this source")


def _measure(path: Path, block: int = 262144, *, witness=None, regions=()) -> dict[str, Any]:
    square_sum = 0.0
    count = 0
    peak = 0.0
    import hashlib
    digest = hashlib.sha256() if witness else None
    spans = [(edge['set_start_frame'],edge['set_end_frame'],value['pcm_sha256'],hashlib.sha256())
             for edge,value in regions if value]
    offset = 0
    with sf.SoundFile(path) as source:
        for audio in source.blocks(blocksize=block, dtype="float32", always_2d=True):
            if not np.all(np.isfinite(audio)):
                raise RenderError("Audio contains nonfinite samples")
            square_sum += float(np.sum(audio.astype(np.float64) ** 2))
            count += audio.size
            peak = max(peak, float(np.max(np.abs(audio))))
            if digest:
                digest.update(np.asarray(audio,dtype='<f4').tobytes())
            for first,last,expected,observed in spans:
                begin,end=max(0,first-offset),min(len(audio),last-offset)
                if end>begin:
                    observed.update(np.asarray(audio[begin:end],dtype='<f4').tobytes())
            offset += len(audio)
    if not count:
        raise RenderError("Decoded track is empty")
    if witness and (digest.hexdigest()!=witness.digest.hexdigest() or offset!=witness.frames):
        raise RenderError('Assembled PCM differs from the verified write sequence')
    if any(expected!=observed.hexdigest() for first,last,expected,observed in spans):
        raise RenderError('An assembled overlap differs from its verified PCM')
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
    """Compatibility entry point for the default independent-band mixer."""
    if outgoing.shape != incoming.shape or outgoing.ndim != 2 or outgoing.shape[1] != 2:
        raise RenderError("Crossfade needs equally sized stereo blocks")
    from .mixing import blend
    return blend(outgoing, incoming, sample_rate, bass_hz=bass_hz)[0]


def _aligned_vocals(stem, directory, position, *, ffmpeg, rubberband, engine,
                    sample_rate, first_frame, source_frames, output_frames,
                    gain, warp):
    """Render a local estimate with the final deck map and identical gain."""
    decoded = directory / f"voice-{position + 1:02d}-decoded.wav"
    cut = directory / f"voice-{position + 1:02d}-input.wav"
    output = directory / f"voice-{position + 1:02d}.wav"
    _run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(stem["path"]),
          "-ac", "2", "-ar", str(sample_rate), "-c:a", "pcm_f32le", str(decoded)])
    with sf.SoundFile(decoded) as source, sf.SoundFile(cut, "w", samplerate=sample_rate,
                                                     channels=2, subtype="FLOAT") as dest:
        if source.frames < first_frame + source_frames:
            raise RenderError("Vocal estimate is shorter than its source deck")
        source.seek(first_frame)
        _copy_region(source, dest, source_frames, gain)
    decoded.unlink()
    if warp:
        control = directory / f"track-{position + 1:02d}-timemap.txt"
        options = ["--fast", "--detector-perc"] if engine == "r2" else ["--fine"]
        _run([rubberband, *options, "--centre-focus", "--ignore-clipping", "--quiet", "--time",
              f"{output_frames / source_frames:.12f}", "--timemap", control.name, cut.name, output.name], cwd=directory)
        difference = sf.info(output).frames - output_frames
        if abs(difference) > 128:
            raise RenderError("Vocal render differs from deck length")
        if difference:
            fixed = output.with_name(output.stem + "-exact.wav")
            with sf.SoundFile(output) as reader, sf.SoundFile(fixed, "w", samplerate=sample_rate,
                                                            channels=2, subtype="FLOAT") as writer:
                _copy_region(reader, writer, min(output_frames, reader.frames))
                if difference < 0:
                    writer.write(np.zeros((-difference, 2), dtype=np.float32))
            fixed.replace(output)
        cut.unlink()
    else:
        cut.replace(output)
    return str(output)


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
                tempo_conflict_policy: str = "auto", bridge_overlap_beats: float = 8.0,
                tempo_ramp_max_change: float = 0.25, tempo_ramp_seconds: float = 15.0,
                native_exit_seconds: float = 8.0,
                sync_max_change: float = 0.40, sync_transition_bars: int | None = None,
                target_rms_dbfs: float = -20.0, target_lufs: float = -18.0,
                track_peak_ceiling: float = 0.50,
                max_gain_db: float = 4.0, bass_hz: float = 180.0,
                allow_native_attack_preservation: bool = False,
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
    Default `auto` first retains an ordinary valid blend, then tries long SYNC
    only if ordinary selection/rate validation fails. Pair `transition_policy`
    can retain an approved explicit SYNC exception.
    `sync` explicitly shares changing tempo/beat phase during the overlap, keeping the
    main body native. A conservative post-render attack check records ambiguous
    attacks instead of claiming complete acoustic verification.
    `adaptive` preserves incoming native timing on a large downshift using an
    outgoing-only vinyl speed/pitch effect over a short simultaneous overlap.
    `shared_tempo` retains the wider two-deck tempo experiment as an explicit choice.
    """
    if sample_rate < 8000 or transition_bars not in {8, 16, 32}:
        raise RenderError("Use a valid sample rate and 8, 16 or 32 transition bars")
    if not math.isfinite(sync_max_change) or not 0 < sync_max_change < 0.5 or (sync_transition_bars is not None and
            (isinstance(sync_transition_bars, bool) or not isinstance(sync_transition_bars, int) or sync_transition_bars not in {2, 4, 8, 16, 32})):
        raise RenderError("SYNC needs a positive temporary rate cap below 0.5 and 2, 4, 8, 16 or 32 bars")
    if rubberband_engine not in {"r2", "r3"}:
        raise RenderError("Rubber Band engine must be r2 or r3")
    if not 0 < max_tempo_change < 0.5 or not 0 < max_local_warp_change < 0.5:
        raise RenderError("Tempo limits must be positive fractions below 0.5")
    tempo_ramp_max_change = _finite(tempo_ramp_max_change, "tempo ramp limit")
    tempo_ramp_seconds = _finite(tempo_ramp_seconds, "tempo ramp seconds")
    if not max_tempo_change <= tempo_ramp_max_change < 0.5:
        raise RenderError("Tempo ramp limit must be at least the regular limit and below 0.5")
    constraints = plan.get("constraints") or {}
    sections = plan.get("sections") or {}
    if sections:
        max_intro_trim_seconds = max(max_intro_trim_seconds, max(float(r.get("duration", r.get("duration_seconds"))) for r in (records.values() if isinstance(records, Mapping) else records)))
        max_outro_trim_seconds = max_intro_trim_seconds
    if "tempo_ramp_max_change_per_deck" in constraints:
        planned_ramp_limit = constraints["tempo_ramp_max_change_per_deck"]
        planned_ramp_limit = max_tempo_change if planned_ramp_limit is None else _finite(planned_ramp_limit, "planned tempo ramp limit")
        if not max_tempo_change <= planned_ramp_limit < 0.5:
            raise RenderError("Planned tempo ramp limit must be at least the regular limit and below 0.5")
        tempo_ramp_max_change = min(tempo_ramp_max_change, planned_ramp_limit)
    if not 10 <= tempo_ramp_seconds <= 20:
        raise RenderError("Tempo ramp must span 10 to 20 output seconds")
    native_exit_seconds = _finite(native_exit_seconds, "native exit seconds")
    if not 4 <= native_exit_seconds <= 20:
        raise RenderError("Native exit overlap must span 4 to 20 seconds")
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
    musical_caps = {identity: max_tempo_change for identity in order}
    reset_head: dict[str, _Window] = {}
    reset_tail: dict[str, _Window] = {}
    transitions: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, edge in enumerate(edges):
        outgoing, incoming = order[index:index + 2]
        selected_bars = edge.get("overlap_bars")
        selected_beats = edge.get("overlap_beats")
        section_bars = edge.get("section_overlap_bars")
        if section_bars is not None and (plan.get("style") != "zhilin" or section_bars not in {16, 32, 40, 44}):
            raise RenderError("Long section overlaps require the Zhilin style")
        if selected_beats is not None:
            if isinstance(selected_beats, bool) or not isinstance(selected_beats, int) or not 8 <= selected_beats <= 256 or selected_bars is not None:
                raise RenderError("Use one native overlap of 8–256 integer beats")
            if set(edge.get("cue_locks") or {}) != {"incoming_start_seconds", "outgoing_end_seconds"}:
                raise RenderError("Native beat overlap requires both source boundary locks")
        edge_policy = edge.get("transition_policy", tempo_conflict_policy)
        if selected_beats is not None:
            if allow_hard_cut or edge_policy == "review":
                raise RenderError("A mandatory native beat overlap requires SYNC without hard cuts")
            edge_policy = "sync"
        # A pair-specific SYNC length is also an explicit choice of that method.
        if selected_bars is not None and "transition_policy" not in edge and edge_policy == "auto":
            edge_policy = "sync"
        if selected_bars is not None:
            if isinstance(selected_bars, bool) or not isinstance(selected_bars, int) or selected_bars not in {2, 4, 8, 16, 32, 40, 44}:
                raise RenderError("Edge overlap_bars must be 2, 4, 8, 16, 32, 40 or 44")
            if edge_policy != "sync" or allow_hard_cut:
                raise RenderError("Edge overlap_bars requires the SYNC renderer without hard cuts")
        ga, gb = grids[outgoing], grids[incoming]
        tempo = edge.get("tempo") or {}
        locks = edge.get("cue_locks") or {}
        if not isinstance(locks, Mapping):
            raise RenderError("cue_locks must be a mapping")
        in_lock, out_lock = locks.get("incoming_start_seconds"), locks.get("outgoing_end_seconds")
        reason = ""
        matched = bool(tempo.get("beatmatch_possible") and not edge.get("tempo_reset"))
        if edge_policy == "sync":
            matched = False
            reason = "explicit pair/global SYNC selection"
        ramp_selected = tempo.get("technique") == "tempo_ramp"
        if ramp_selected and edge_policy == "adaptive":
            matched = False
        edge_cap = tempo_ramp_max_change if ramp_selected else max_tempo_change
        if not matched and edge_policy == "shared_tempo" and not allow_hard_cut:
            native_a, native_b = by_id[outgoing].get("bpm"), by_id[incoming].get("bpm")
            if native_a is not None and native_b is not None:
                native_a = _finite(native_a, "native outgoing BPM")
                native_b = _finite(native_b, "native incoming BPM")
                if native_a <= 0 or native_b <= 0:
                    raise RenderError("Native BPM must be positive")
                common = 2 * native_a * native_b / (native_a + native_b)
                if max(abs(common / native_a - 1), abs(common / native_b - 1)) <= tempo_ramp_max_change + 1e-6:
                    tempo = {**tempo, "beatmatch_possible": True, "target_bpm": common,
                             "technique": "tempo_ramp"}
                    matched, ramp_selected, edge_cap = True, True, tempo_ramp_max_change
        usable_for_search = (ga[0] or _local_validation_possible(by_id[outgoing], ga)) and (gb[0] or _local_validation_possible(by_id[incoming], gb))
        if selected_bars is not None and not usable_for_search:
            raise RenderError(f"{outgoing} → {incoming}: mandatory overlap_bars requires usable native grids")
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
        bars = section_bars or min(transition_bars, 8 if "distant" in str(edge.get("harmonic_relation", "")) else transition_bars)
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
                    if abs(nominal - 1) > edge_cap + 1e-6:
                        raise RenderError("Planned target BPM exceeds musical stretch limit")
                    native = _finite(by_id[incoming if window is h else outgoing]["bpm"], "native BPM")
                    local = np.diff(window.times) / (60 / native)
                    if np.max(np.abs(local - 1)) > max_local_warp_change:
                        raise RenderError("Irregular beat intervals exceed the local timing-correction limit")
                head[incoming], tail[outgoing] = h, t
                musical_caps[outgoing] = max(musical_caps[outgoing], edge_cap)
                musical_caps[incoming] = max(musical_caps[incoming], edge_cap)
            except RenderError as exc:
                reason = str(exc)
                matched = False
        if not matched and edge_policy in {"auto", "sync"} and usable_for_search and not allow_hard_cut:
            from .sync import shared_clock
            native_a = _finite(by_id[outgoing].get("bpm"), "native outgoing BPM")
            native_b = _finite(by_id[incoming].get("bpm"), "native incoming BPM")
            if min(native_a, native_b) <= 0:
                raise RenderError("SYNC requires positive measured BPM on both decks")
            change = max(abs(native_b / native_a - 1), abs(native_a / native_b - 1))
            if change > sync_max_change + 1e-6:
                raise RenderError(f"{outgoing} → {incoming}: common SYNC curve needs {change:.1%} temporary rate change; cap {sync_max_change:.1%}")
            # Long F is the default fallback. This tempo heuristic chooses a
            # starting length, not a detected phrase or an audible optimum.
            count = (sync_transition_bars if sync_transition_bars is not None else
                     (16 if change > .20 else 32)) if change > max_tempo_change else transition_bars
            if selected_bars is not None:
                count = selected_bars
            elif section_bars and change <= .20:
                count = section_bars
            beat_count = selected_beats or count * 4
            if selected_beats is not None:
                count = selected_beats / 4
            clock = shared_clock(native_a, native_b, beat_count, sample_rate)
            h = _select_window(by_id[incoming], gb, region="head", count=count,
                               beat_count=selected_beats,
                               locked_cue_seconds=in_lock, target_bpm=native_b, sample_rate=sample_rate,
                               max_trim_seconds=max_intro_trim_seconds, max_local_warp_change=max_local_warp_change,
                               minimum_grid_confidence=minimum_grid_confidence)
            t = _select_window(by_id[outgoing], ga, region="tail", count=count,
                               beat_count=selected_beats,
                               locked_cue_seconds=out_lock, target_bpm=native_a, sample_rate=sample_rate,
                               max_trim_seconds=max_outro_trim_seconds, max_local_warp_change=max_local_warp_change,
                               minimum_grid_confidence=minimum_grid_confidence)
            h.clock = t.clock = clock
            h.frames = t.frames = clock["frames"]
            head[incoming], tail[outgoing] = h, t
            musical_caps[incoming] = max(musical_caps[incoming], sync_max_change)
            musical_caps[outgoing] = max(musical_caps[outgoing], sync_max_change)
            transitions.append({"from": outgoing, "to": incoming, "mode": "beatmapped_bass_swap",
                                "tempo_technique": "shared_phase_sync", "shared_clock": clock,
                                "selection_reason": reason or "ordinary blend unavailable; long F replaces a discontinuous reset",
                                "fallback_from_ordinary": edge_policy == "auto",
                                "musical_tempo_limit": sync_max_change, "beatmatched": True,
                                "bars": count, "beats": beat_count, "target_bpm": None,
                                "overlap_frames": clock["frames"],
                                "outgoing_source_start": float(t.times[0]), "outgoing_source_end": float(t.times[-1]),
                                "incoming_source_start": float(h.times[0]), "incoming_source_end": float(h.times[-1]),
                                "outgoing_cue": t.cue_provenance, "incoming_cue": h.cue_provenance,
                                "bar_alignment": "one-to-one shared beat phase; explicit partial-bar exception" if selected_beats else "one-to-one shared beat phase from supplied grids",
                                "phrase_alignment": "bar grouping only; phrase review pending",
                                "human_reviewed": False})
            continue
        if not matched:
            native_a, native_b = by_id[outgoing].get("bpm"), by_id[incoming].get("bpm")
            use_vinyl = bool(edge_policy == "adaptive" and not allow_hard_cut and usable_for_search
                             and native_a and native_b and 0.65 <= float(native_b) / float(native_a) < 1)
            use_bridge = edge_policy == "echo_bridge" and not allow_hard_cut
            if (use_bridge or use_vinyl) and plan.get("constraints", {}).get("allow_tempo_reset") is False:
                raise RenderError(f"{outgoing} → {incoming}: plan forbids native-tempo fallback bridges")
            if not allow_hard_cut and not allow_unverified and not use_bridge and not use_vinyl:
                raise RenderError(f"{outgoing} → {incoming}: {reason or 'no feasible common tempo'}; "
                                  "transition needs review; automatic hard cuts are disabled")
            mode = "unverified_short_crossfade"
            reset_cues: dict[str, Any] = {}
            if usable_for_search and (allow_hard_cut or use_bridge or use_vinyl):
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
                    mode = "outgoing_only_vinyl" if use_vinyl else "free_time_echo_bridge" if use_bridge else "bar_tempo_reset"
                    reset_cues = {"incoming_cue": hreset.cue_provenance,
                                  "outgoing_cue": treset.cue_provenance,
                                  "bar_alignment": "locally validated predicted/reviewed native cue points; different tempos, no beatmatching"}
                    if use_bridge:
                        overlap_seconds = bridge_overlap_beats * 60 / float(by_id[incoming]["bpm"])
                        if overlap_seconds > float(treset.times[-1] - treset.times[0]):
                            raise RenderError("Echo overlap exceeds the validated outgoing region")
                        reset_cues["echo_parameters"] = {"outgoing_bpm": float(by_id[outgoing]["bpm"]),
                            "incoming_bpm": float(by_id[incoming]["bpm"]), "overlap_beats": bridge_overlap_beats}
                    if use_vinyl:
                        from .creative import tape_exit_curve
                        overlap_seconds = native_exit_seconds
                        effect_release = min(2.0, overlap_seconds / 4)
                        effect_ramp = overlap_seconds - effect_release
                        positions, curve = tape_exit_curve(round(overlap_seconds * sample_rate), sample_rate,
                            ramp_seconds=effect_ramp, end_rate=float(native_b) / float(native_a))
                        reset_cues["vinyl_parameters"] = {"ramp_seconds": effect_ramp,
                            "outgoing_release_seconds": effect_release, "end_rate": float(native_b) / float(native_a)}
                        reset_cues["vinyl_curve"] = curve
                        reset_cues["incoming_tempo_preserved"] = True
                except RenderError as exc:
                    if use_bridge or use_vinyl or in_lock is not None or out_lock is not None:
                        raise RenderError(f"{outgoing} → {incoming}: explicit bridge/cue could not be rendered: {exc}") from exc
                    reason = (reason + "; " if reason else "") + f"bar reset unavailable: {exc}"
            if use_bridge and mode != "free_time_echo_bridge":
                raise RenderError(f"{outgoing} → {incoming}: echo bridge requires reliable native grids and cues")
            if mode == "unverified_short_crossfade" and strict_grid and not allow_unverified:
                raise RenderError(f"{outgoing} → {incoming}: {reason or 'no reliable bar-aligned reset'}")
            warnings.append(f"{outgoing} → {incoming}: {reason or 'no feasible common tempo'}; {mode}, not beatmatched")
            overlap = (round(overlap_seconds * sample_rate) if mode in {"free_time_echo_bridge", "outgoing_only_vinyl"}
                       else 0 if mode == "bar_tempo_reset" else round(0.20 * sample_rate))
            transitions.append({"from": outgoing, "to": incoming, "mode": mode,
                                "beatmatched": False, "bars": None, "target_bpm": None,
                                "overlap_frames": overlap, "reason": reason or "no feasible common tempo",
                                **reset_cues})
        else:
            transitions.append({"from": outgoing, "to": incoming, "mode": "beatmapped_bass_swap",
                                "tempo_technique": "tempo_ramp" if ramp_selected else "standard_blend",
                                "selection_reason": "ordinary beatmatched blend succeeded; fallback F bypassed",
                                "musical_tempo_limit": edge_cap,
                                "tempo_ramp_requested_output_seconds": tempo_ramp_seconds,
                                "beatmatched": True, "bars": bars, "beats": bars * 4,
                                "target_bpm": target, "overlap_frames": h.frames,
                                "outgoing_source_start": float(t.times[0]), "outgoing_source_end": float(t.times[-1]),
                                "incoming_source_start": float(h.times[0]), "incoming_source_end": float(h.times[-1]),
                                "outgoing_cue": t.cue_provenance, "incoming_cue": h.cue_provenance,
                                "bar_alignment": "predicted/reviewed downbeats, exact sample-frame map",
                                "phrase_alignment": "estimated bar grouping; musical phrase boundaries not guaranteed",
                                "human_reviewed": bool((by_id[outgoing].get("grid") or {}).get("human_reviewed") and
                                                       (by_id[incoming].get("grid") or {}).get("human_reviewed"))})
    for transition,edge in zip(transitions,edges):
        for key in ('mid_handoff_bars','bass_handoff_beats','handoff_windows'):
            if edge.get(key) is not None:transition[key]=edge[key]
    clips: list[dict[str, Any]] = []
    prepared_files = prepared_files or {}
    # All keyword render controls participate in object identity.
    import inspect
    from .execution import cached
    render_local_values = locals().copy()
    render_controls = {name: locals_value for name, locals_value in render_local_values.items()
        if name in inspect.signature(render_plan).parameters and name not in
        {'plan', 'records', 'prepared_files', 'output_path', 'work_dir', 'progress'}}
    from .audio import ffmpeg_executable
    import shutil
    render_controls['ffmpeg'] = ffmpeg_executable(ffmpeg)
    render_controls['rubberband'] = shutil.which(str(rubberband)) or str(rubberband)
    overlap_files = []
    from copy import deepcopy
    transition_inputs = deepcopy(transitions)
    def prepare_clip(position, identity):
        from copy import deepcopy
        transitions = deepcopy(transition_inputs)
        warnings = []
        warning_start = 0
        clips = []
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
        from .audio import decode_wave_file
        decode_wave_file(prepared_path, decoded, sample_rate=sample_rate, ffmpeg=ffmpeg)
        info = sf.info(decoded)
        duration = _finite(record.get("duration", record.get("duration_seconds")), "duration")
        if abs(info.duration - duration) > 0.05:
            raise RenderError("Prepared processing changed track timing; reanalyse that recording before rendering")
        ga = grids[identity]
        section = sections.get(identity) or {}
        trim_start = float(head[identity].times[0]) if head[identity] else (float(reset_head[identity].times[0]) if identity in reset_head else float(section.get("start", 0.0)))
        trim_end = float(tail[identity].times[-1]) if tail[identity] else (float(reset_tail[identity].times[-1]) if identity in reset_tail else float(section.get("end", info.duration)))
        if section and (abs(trim_start - section["start"]) > 1 / sample_rate or abs(trim_end - section["end"]) > 1 / sample_rate):
            raise RenderError("Selected transition moved a mandatory section boundary")
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
            # The decoded clip starts/ends on actual sample frames. Using two
            # unquantized cue times with separately rounded cuts can lose the
            # final frame at nonzero body cues (a one-frame endpoint mismatch).
            warp = _warp_anchors(record, trim_start=first_frame / sample_rate,
                                 trim_end=end_frame / sample_rate,
                                 source_frames=source_frames, sample_rate=sample_rate,
                                 head=head[identity], tail=tail[identity], max_tempo_change=musical_caps[identity],
                                 max_local_warp_change=max_local_warp_change,
                                 tempo_ramp_seconds=tempo_ramp_seconds)
            if position and transitions[position - 1]["mode"] == "outgoing_only_vinyl":
                preserved_seconds = transitions[position - 1]["overlap_frames"] / sample_rate
                if abs(warp["entry_musical_rate"] - 1) > 1e-12 or any(
                    ramp["output_start_seconds"] < preserved_seconds - 1 / sample_rate
                    for ramp in warp["tempo_ramps"]
                ):
                    raise RenderError("Next tempo automation intersects the native incoming overlap")
            map_path = directory / f"track-{position + 1:02d}-timemap.txt"
            if (head[identity] and head[identity].clock) or (tail[identity] and tail[identity].clock):
                from .kick_grid import refine_source_controls
                kick_evidence=record.get('kick_evidence')
                if kick_evidence is None and (record.get('vocal_stem') or {}).get('drums_path'):
                    stem=record['vocal_stem']
                    drum_path=prepared.get('drum_path') if isinstance(prepared,Mapping) else None
                    if drum_path or not (preparation or {}).get('processors'):
                        kick_evidence={role:{'path':drum_path or stem['drums_path'],'source_start_seconds':stem.get('drum_source_start_seconds',0.)} for role in ('head','tail')}
                refine_source_controls(clip_input,warp,sample_rate,kick_evidence)
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
            if (head[identity] and head[identity].clock) or (tail[identity] and tail[identity].clock):
                from .sync import transient_drift, compensate_backend_drift
                from .kick_grid import rendered_attack_audit
                from copy import deepcopy
                import shutil
                audits = []
                use_kicks=any(r['events'] for r in warp.get('source_kick_alignment',{}).values())
                best=None;best_file=directory/f'track-{position+1:02d}-acoustic-best.wav'
                def rerender():
                    map_path.write_text(''.join(f"{p['source_frame']} {p['output_frame']}\n"
                                               for p in warp['anchors']), encoding='ascii')
                    _run([rubberband, *engine_args, '--centre-focus', '--ignore-clipping', '--quiet',
                          '--time', f'{ratio:.12f}', '--timemap', map_path.name,
                          clip_input.name, clip_output.name], cwd=directory)
                    if sf.info(clip_output).frames != warp['output_frames']:
                        raise RenderError('Compensated SYNC render did not retain its exact declared duration')
                for attempt in range(10):
                    if warp.get('source_attack_controls_unapplied'):
                        if not allow_native_attack_preservation:
                            raise RenderError(f'{identity}: source rhythm needs another compatible window; local attack replacement is disabled')
                        from .kick_grid import preserve_rendered_attacks
                        preserve_rendered_attacks(clip_output,prepared_path,warp,sample_rate,gain=gain)
                        warp['native_attack_preservation']['roles']=['head','tail']
                    audit = rendered_attack_audit(clip_output,warp,sample_rate) if use_kicks else transient_drift(clip_input, clip_output, warp, sample_rate)
                    audits.append(audit)
                    from .execution import current
                    if current() and attempt:
                        current().count("acoustic_rerender_attempts")
                    maximum = audit["maximum_measured_drift_seconds"]
                    if maximum is not None and (best is None or maximum<best[0]):
                        best=(maximum,deepcopy(warp),deepcopy(audit));shutil.copy2(clip_output,best_file)
                    if attempt == 1 and current() and maximum is not None and maximum > .003:
                        from .acoustic_preview import screen_correction
                        auditor = (lambda source, output, mapping, rate: rendered_attack_audit(output, mapping, rate)) if use_kicks else transient_drift
                        try:
                            candidate = screen_correction(clip_input, warp, audit, sample_rate,
                                directory/f'track-{position+1:02d}-preview', rubberband=rubberband,
                                engine_args=engine_args, run=_run, audit=auditor)
                        except (ValueError, RuntimeError):
                            candidate = None
                            current().count('acoustic_preview_fallbacks')
                        if candidate:
                            screen_start = deepcopy(warp)
                            screen_file = directory/f'track-{position+1:02d}-preview-start.wav'
                            shutil.copy2(clip_output,screen_file)
                            warp = candidate
                            rerender()
                            current().count('acoustic_rerender_attempts')
                            full_audit = auditor(clip_input, clip_output, warp, sample_rate)
                            full_maximum = full_audit['maximum_measured_drift_seconds']
                            covered = lambda evidence: {(p['role'], p['beat'], p.get('view')) for p in evidence['measured']}
                            if (full_maximum is not None and full_maximum <= .003
                                    and set().union(*(covered(previous) for previous in audits)) <= covered(full_audit)):
                                audits.append(full_audit)
                                best=(full_maximum,deepcopy(warp),deepcopy(full_audit))
                                shutil.copy2(clip_output,best_file)
                                current().count('acoustic_preview_confirmed')
                                screen_file.unlink()
                                break
                            current().count('acoustic_preview_fallbacks')
                            warp=screen_start;screen_file.replace(clip_output)
                    if maximum is None or maximum <= .003 or attempt == 9:
                        break
                    # The backend can switch a transient's alignment at a
                    # detector threshold. Damped follow-ups prevent oscillation.
                    try:compensate_backend_drift(warp, audit, sample_rate, strength=1.0 if attempt < 2 else .5)
                    except ValueError as error:
                        warp.setdefault('acoustic_correction_rejections',[]).append(str(error))
                        break
                    rerender()
                if best is None:
                    best_file.unlink(missing_ok=True)
                    raise RenderError(f'{identity}: acoustic attack alignment unresolved; select a supported window or provide percussion evidence')
                warp=best[1];shutil.copy2(best_file,clip_output);best_file.unlink()
                if use_kicks:
                    from .kick_grid import fine_align_rendered,preserve_rendered_attacks
                    final_attack=fine_align_rendered(clip_output,warp,sample_rate)
                    protection_roles=[role for role,r in warp.get('source_kick_alignment',{}).items()
                        if len(r['events'])>=8 and (sum(e['role']==role for e in final_attack['measured'])<8
                        or max((abs(e['delay_seconds']) for e in final_attack['measured'] if e['role']==role),default=0)>.003
                        or set((r.get('low_frequency_gap_recovery') or {}).get('recovered_beats',[]))-
                           {e['beat'] for e in final_attack['measured'] if e['role']==role and e.get('view')=='bass_body'})]
                    if protection_roles:
                        if not allow_native_attack_preservation:
                            raise RenderError(f'{identity}: stretched attack recognition needs another compatible window; local attack replacement is disabled')
                        preserve_rendered_attacks(clip_output,prepared_path,warp,sample_rate,gain=gain,roles=protection_roles)
                        warp['native_attack_preservation']['roles']=protection_roles
                        final_attack=rendered_attack_audit(clip_output,warp,sample_rate)
                else:final_attack=best[2]
                if final_attack['maximum_measured_drift_seconds'] is None or final_attack['maximum_measured_drift_seconds']>.003:
                    raise RenderError(f'{identity}: acoustic attack alignment remains unresolved')
                from .timing_safety import audit_warp_rates
                audit_warp_rates(warp,sample_rate,stage='final map after all acoustic refinements')
                map_path.write_text("".join(f"{p['source_frame']} {p['output_frame']}\n" for p in warp['anchors']),encoding='ascii')
                warp["post_render_transient_audit"] = {"iterations": audits,
                    "final": final_attack, "tolerance_seconds": .003,
                    "all_interior_attacks_within_tolerance": bool(final_attack["all_interior_beats_verified"] and
                        final_attack["maximum_measured_drift_seconds"] <= .003),
                    "backend_controls_precompensated": len(audits) > 1}
                if use_kicks:warp['post_render_kick_audit']=warp['post_render_transient_audit']
            clip_input.unlink()
        else:
            clip_input.replace(clip_output)
        creative_warp = None
        if position < len(transitions) and transitions[position]["mode"] == "outgoing_only_vinyl":
            from .creative import tape_exit_curve, tape_speed_warp
            effect = transitions[position]
            positions, curve = tape_exit_curve(effect["overlap_frames"], sample_rate,
                ramp_seconds=effect["vinyl_parameters"]["ramp_seconds"],
                end_rate=effect["vinyl_parameters"]["end_rate"])
            required = curve["source_frames_required"]
            current_frames = sf.info(clip_output).frames
            prefix_frames = current_frames - required
            incoming_frames = transitions[position - 1]["overlap_frames"] if position else 0
            if prefix_frames <= incoming_frames:
                raise RenderError("Not enough track body remains for an outgoing-only exit")
            if warp and warp["tempo_ramps"]:
                last_ramp = warp["tempo_ramps"][-1]
                if last_ramp["output_end_seconds"] > prefix_frames / sample_rate:
                    raise RenderError("Entry automation intersects the outgoing-only exit; choose a later cue")
            changed = directory / f"track-{position + 1:02d}-vinyl.wav"
            with sf.SoundFile(clip_output) as source, sf.SoundFile(changed, "w", samplerate=sample_rate,
                                                                 channels=2, subtype="FLOAT") as dest:
                _copy_region(source, dest, prefix_frames)
                source_tail = source.read(required, dtype="float32", always_2d=True)
                dest.write(tape_speed_warp(source_tail, positions))
            changed.replace(clip_output)
            creative_warp = {**curve, "outgoing_exit_start_in_clip_seconds": prefix_frames / sample_rate,
                             "incoming_tempo_changed": False, "pitch_preserved": False,
                             "effect": "outgoing-only vinyl speed and pitch fall"}
            effect["outgoing_exit_processing"] = creative_warp
        vocal_path = None
        if record.get("vocal_stem"):
            processed_voice = prepared.get("vocal_path") if isinstance(prepared, Mapping) else None
            if creative_warp or ((preparation or {}).get("processors") and not processed_voice):
                warnings.append(f"{identity}: vocal routing bypassed because preparation/creative processing changed the source mixture")
            else:
                vocal_source = {**record["vocal_stem"], "path": processed_voice or record["vocal_stem"]["path"]}
                vocal_path = _aligned_vocals(vocal_source, directory, position, ffmpeg=ffmpeg,
                    rubberband=rubberband, engine=rubberband_engine, sample_rate=sample_rate,
                    first_frame=first_frame, source_frames=source_frames,
                    output_frames=sf.info(clip_output).frames, gain=gain, warp=warp)
                if warp and warp.get('fine_attack_timing'):
                    from .kick_grid import apply_fine_timing
                    fine_voice=Path(vocal_path).with_name(Path(vocal_path).stem+'-fine.wav')
                    apply_fine_timing(vocal_path,fine_voice,warp,sample_rate)
                    fine_voice.replace(vocal_path)
                if warp and warp.get('native_attack_preservation'):
                    from .kick_grid import preserve_rendered_attacks
                    preserve_rendered_attacks(vocal_path,vocal_source['path'],warp,sample_rate,gain=gain,
                        roles=warp['native_attack_preservation']['roles'],record_templates=False)
        clips.append({"id": identity, "audio_path": str(clip_output), "source_path": str(source_path),
                      "aligned_vocal_path": vocal_path,
                      "prepared_path": str(prepared_path), "preparation_provenance": preparation,
                      "source_duration_seconds": info.duration, "trim_start_seconds": first_frame / sample_rate,
                      "trim_end_seconds": end_frame / sample_rate,
                      "discarded_tail_seconds": (info.frames - end_frame) / sample_rate,
                      "level_measurements": measurements, "gain_db": gain_db,
                      "gain_method": gain_method,
                      "warp": warp, "creative_warp": creative_warp, "grid_status": record.get("grid_status"),
                      "grid_confidence": record.get("grid_confidence"),
                      "human_reviewed": bool((record.get("grid") or {}).get("human_reviewed")),
                      "output_frames": sf.info(clip_output).frames,
                      "duration_seconds": sf.info(clip_output).frames / sample_rate})
        if section:
            clips[-1]["section_policy"] = section
            if 'target_solo_seconds' not in section:
                clips[-1]["rendered_selected_span_in_3_4_minutes"] = 180 <= clips[-1]["duration_seconds"] <= 240
            if 'target_solo_seconds' not in section and not clips[-1]["rendered_selected_span_in_3_4_minutes"]:
                warnings.append(f"{identity}: rendered selected span outside 3–4 minutes; source/tempo/manual exception retained")
        def updates(index):
            return {k: v for k, v in transitions[index].items()
                    if k not in transition_inputs[index] or v != transition_inputs[index][k]}
        return {'clip': clips.pop(), 'warnings': warnings[warning_start:],
                'previous_transition': updates(position-1) if position else {},
                'next_transition': updates(position) if position < len(transitions) else {}}

    def prepare_overlap(position):
        clip = clips[position]
        transition = transitions[position-1]
        incoming = transition['overlap_frames']
        if not incoming:
            return None
        def produce():
            nonlocal transition
            with sf.SoundFile(clips[position-1]['audio_path']) as source:
                source.seek(clips[position-1]['output_frames'] - incoming)
                previous_tail = source.read(incoming, dtype='float32', always_2d=True)
            with sf.SoundFile(clip['audio_path']) as source:
                block = source.read(incoming, dtype='float32', always_2d=True)
            previous_vocal = None
            if clips[position-1].get('aligned_vocal_path'):
                with sf.SoundFile(clips[position-1]['aligned_vocal_path']) as source:
                    source.seek(clips[position-1]['output_frames'] - incoming)
                    previous_vocal = source.read(incoming, dtype='float32', always_2d=True)
            if len(previous_tail) != incoming or len(block) != incoming:
                raise RenderError('Missing overlap audio')
            mode = transitions[position - 1]["mode"]
            if mode == "free_time_echo_bridge":
                from .bridges import echo_bridge
                mixed, evidence = echo_bridge(previous_tail, block, sample_rate,
                    **transitions[position - 1]["echo_parameters"])
                transitions[position - 1]["echo_processing"] = evidence
            elif mode == "outgoing_only_vinyl":
                from .creative import mix_tape_exit
                parameters = transitions[position - 1]["vinyl_parameters"]
                mixed, evidence = mix_tape_exit(previous_tail, block, sample_rate,
                    outgoing_release_seconds=parameters["outgoing_release_seconds"], bass_hz=bass_hz)
                transitions[position - 1]["vinyl_gain_processing"] = evidence
            else:
                from .mixing import blend
                from .stems import handoff
                transition = transitions[position - 1]
                incoming_vocal = None
                if clip.get("aligned_vocal_path"):
                    with sf.SoundFile(clip["aligned_vocal_path"]) as voice:
                        incoming_vocal = voice.read(incoming, dtype="float32", always_2d=True)
                previous_tail, block, voice_report = handoff(previous_tail, block, previous_vocal, incoming_vocal)
                transition["vocal_processing"] = voice_report
                risk = any(float((transition.get(role) or {}).get("sustained_midband_proxy") or 0) > .5
                           for role in ("outgoing_cue", "incoming_cue"))
                clock = transition.get('shared_clock')
                beat_phase = (np.interp(np.arange(incoming), clock['output_frames'], clock['phase']) / clock['beats']) if clock else None
                if beat_phase is not None:
                    beat_phase[-1] = 1.
                from .percussion import needs_handoff,final_drum_estimates,ownership_policy,percussion_gains
                percussion_args={}
                if mode=='beatmapped_bass_swap' and needs_handoff(clips[position-1].get('warp'),clip.get('warp'),style=plan.get('style','default')):
                    percussion_args,percussion_report=final_drum_estimates(previous_tail,block,sample_rate,directory/f'percussion-{position:02}',ffmpeg=ffmpeg)
                    transition['percussion_estimates']=percussion_report
                    percussion_args['percussion_policy']=ownership_policy(clips[position-1].get('warp'),clip.get('warp'))
                    transition['percussion_ownership_policy']=percussion_args['percussion_policy']
                mixed, evidence = blend(previous_tail, block, sample_rate,
                    bass_hz=bass_hz if mode == "beatmapped_bass_swap" else 0.0,
                    beats=transition.get("beats", 64), tonal_conflict=risk, beat_phase=beat_phase,
                    style=plan.get('style','default'),
                    mid_handoff_bars=transition.get('mid_handoff_bars'),
                    bass_handoff_beats=transition.get('bass_handoff_beats'),handoff_windows=transition.get('handoff_windows'),**percussion_args)
                transition["frequency_processing"] = evidence
                if (mode == 'beatmapped_bass_swap' and transition.get('tempo_technique') == 'shared_phase_sync'
                        and plan.get('style') == 'zhilin'):
                    from .percussion import verify_blended_fronts,verification_components
                    from .mixing import envelopes
                    gains,_=envelopes(incoming,transition.get('beats',64),beat_phase=beat_phase,style=plan.get('style','default'),mid_handoff_bars=transition.get('mid_handoff_bars'),bass_handoff_beats=transition.get('bass_handoff_beats'),handoff_windows=transition.get('handoff_windows'))
                    gains=percussion_gains(gains,beat_phase,transition.get('beats',64),percussion_args.get('percussion_policy','shared'))
                    components=verification_components(previous_tail,block,sample_rate,gains,bass_hz=bass_hz,**{k:v for k,v in percussion_args.items() if k!='percussion_policy'})
                    try:
                        mixed,front_report=verify_blended_fronts(mixed,sample_rate,clips[position-1].get('warp'),clip.get('warp'),gains,directory/f'percussion-fronts-{position:02}',ffmpeg=ffmpeg,components=components)
                    except ValueError as error:
                        if percussion_args.get('percussion_policy')=='sequential':raise
                        # Both timed decks already passed their independent
                        # clock audit. A conflicting combined figure requires
                        # another orchestration, never a relaxed timing gate.
                        if not percussion_args:
                            percussion_args,percussion_report=final_drum_estimates(previous_tail,block,sample_rate,directory/f'percussion-{position:02}',ffmpeg=ffmpeg)
                            transition['percussion_estimates']=percussion_report
                        percussion_args['percussion_policy']='sequential'
                        transition['percussion_ownership_policy']='sequential'
                        transition['percussion_routing_review']={'reason':str(error),'independent_deck_clock_checks_retained':True,
                            'source_windows_clock_overlap_and_harmonic_curves_unchanged':True,'timing_gate_relaxed':False}
                        mixed,evidence=blend(previous_tail,block,sample_rate,bass_hz=bass_hz,beats=transition.get('beats',64),
                            tonal_conflict=risk,beat_phase=beat_phase,style=plan.get('style','default'),
                            mid_handoff_bars=transition.get('mid_handoff_bars'),bass_handoff_beats=transition.get('bass_handoff_beats'),handoff_windows=transition.get('handoff_windows'),**percussion_args)
                        transition['frequency_processing']=evidence
                        gains,_=envelopes(incoming,transition.get('beats',64),beat_phase=beat_phase,style=plan.get('style','default'),
                            mid_handoff_bars=transition.get('mid_handoff_bars'),bass_handoff_beats=transition.get('bass_handoff_beats'),handoff_windows=transition.get('handoff_windows'))
                        gains=percussion_gains(gains,beat_phase,transition.get('beats',64),'sequential')
                        components=verification_components(previous_tail,block,sample_rate,gains,bass_hz=bass_hz,
                            **{k:v for k,v in percussion_args.items() if k!='percussion_policy'})
                        mixed,front_report=verify_blended_fronts(mixed,sample_rate,clips[position-1].get('warp'),clip.get('warp'),gains,
                            directory/f'percussion-fronts-{position:02}-sequential',ffmpeg=ffmpeg,components=components)
                    transition['blended_front_audit']=front_report
            if transition.get('tempo_technique') == 'shared_phase_sync':
                from .transition_quality import verify_overlap
                transition['quality_gate'] = verify_overlap(previous_tail, block, mixed, sample_rate,
                    clips[position-1:position+1], transition, vocal_out=previous_vocal,
                    vocal_in=incoming_vocal, style=plan.get('style','default'), beat_phase=beat_phase)
            path = directory / f'verified-overlap-{position:02d}.wav'
            sf.write(path, mixed, sample_rate, subtype='FLOAT')
            import hashlib
            return {'path': str(path), 'transition': transitions[position-1],
                    'pcm_sha256':hashlib.sha256(np.asarray(mixed,dtype='<f4').tobytes()).hexdigest()}
        from .zhilin import reference_profile
        from .percussion import needs_handoff
        from .stems import configuration as stem_configuration
        from .execution import current
        # The combined-front audit can request estimates even when source
        # evidence did not request an initial percussion handoff. Its model
        # configuration belongs to the cache key for that fallback as well.
        uses_drums = (transition['mode'] == 'beatmapped_bass_swap' and
                      (needs_handoff(clips[position-1].get('warp'), clip.get('warp'), style=plan.get('style','default'))
                       or (plan.get('style') == 'zhilin' and transition.get('tempo_technique') == 'shared_phase_sync')))
        value = cached('overlap', {'outgoing': clips[position-1], 'incoming': clip,
            'transition': transition, 'style': plan.get('style', 'default'), 'bass_hz': bass_hz,
            'sample_rate': sample_rate, 'decoder': render_controls['ffmpeg'],
            'zhilin_profile': reference_profile() if plan.get('style') == 'zhilin' else None,
            'stem_configuration': stem_configuration() if uses_drums else None,
            'gpu_devices': list(current().devices) if current() and uses_drums else []}, produce, directory)
        transitions[position-1] = value['transition']
        return value

    def render_one(item):
        position, identity = item
        dependencies = {'record': by_id[identity], 'prepared': prepared_files.get(identity, prepared_files.get(str(by_id[identity]['path']), str(by_id[identity]['path']))),
            'head': head[identity], 'tail': tail[identity], 'reset_head': reset_head.get(identity), 'reset_tail': reset_tail.get(identity),
             'section': sections.get(identity),
            'previous': transition_inputs[position-1] if position else None,
            'next': transition_inputs[position] if position < len(transitions) else None, 'controls': render_controls}
        try:
            return position, identity, cached('clip', dependencies, lambda: prepare_clip(position, identity), directory)
        except RenderError as error:
            if not transitions:
                raise
            from .transition_quality import TransitionFailure
            edge = transitions[max(0,position-1)]
            raise TransitionFailure(edge['from'],edge['to'],str(error)) from error

    from contextlib import closing, nullcontext
    from .execution import current
    session = current()
    rendered = session.imap(render_one, enumerate(order)) if session else map(render_one, enumerate(order))
    with closing(rendered) if session else nullcontext(rendered) as results:
        for position, identity, result in results:
            warnings.extend(result['warnings'])
            clip = result['clip']
            clip['id'] = identity
            clip['source_path'] = str(by_id[identity]['path'])
            if clip.get('warp'):
                from .timing_safety import initialize_rate_reference,audit_warp_rates
                initialize_rate_reference(clip['warp'],sample_rate,native_bpm=by_id[identity]['bpm'],
                    windows=[(role,w.clock) for role,w in (('head',head[identity]),('tail',tail[identity])) if w],limit=max_local_warp_change)
                audit_warp_rates(clip['warp'],sample_rate,stage='final timed clip before overlap and delivery')
            _validate_playback_budget(clip, sections.get(identity) or {}, sample_rate)
            clips.append(clip)
            if position:
                transitions[position-1].update(result['previous_transition'])
            if position < len(transitions):
                transitions[position].update(result['next_transition'])
            if position:
                from .kick_grid import validate_overlap_attacks
                try:
                    validate_overlap_attacks(clips[-2:], [transitions[position-1]])
                    overlap_files.append(prepare_overlap(position))
                except (ValueError, RenderError) as error:
                    from .transition_quality import TransitionFailure
                    if isinstance(error,TransitionFailure):
                        raise
                    edge = transitions[position-1]
                    raise TransitionFailure(edge['from'],edge['to'],str(error)) from error
    from .kick_grid import validate_overlap_attacks
    try:
        validate_overlap_attacks(clips,transitions)
    except ValueError as error:
        raise RenderError(str(error)) from error
    # Check actual timed PCM, including both overlaps. A legacy solo-only
    # plan must not bypass the ordinary five-minute full playback maximum.
    for clip in clips:
        _validate_playback_budget(clip, sections.get(clip['id']) or {}, sample_rate)
    temporary = output.with_name(output.stem + ".partial.wav")
    if temporary.exists():
        raise FileExistsError(temporary)
    cursor = 0
    previous_tail: np.ndarray | None = None
    previous_vocal: np.ndarray | None = None
    try:
        predicted_frames = sum(c['output_frames'] for c in clips) - sum(t['overlap_frames'] for t in transitions)
        container = "RF64" if predicted_frames * 8 + 4096 >= 2**32 else "WAV"
        from .integrity import WitnessWriter
        with sf.SoundFile(temporary, "w", samplerate=sample_rate, channels=2, subtype="FLOAT", format=container) as written:
            dest=WitnessWriter(written)
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
                        with sf.SoundFile(overlap_files[position-1]['path']) as verified:
                            mixed = verified.read(dtype='float32', always_2d=True)
                        if len(mixed) != incoming:
                            raise RenderError('Verified overlap length changed')
                        dest.write(mixed)
                        transitions[position - 1]["set_start_frame"] = cursor - incoming
                        transitions[position - 1]["set_end_frame"] = cursor
                        transitions[position - 1]["set_start_seconds"] = (cursor - incoming) / sample_rate
                        transitions[position - 1]["set_end_seconds"] = cursor / sample_rate
                        previous_tail = None
                        previous_vocal = None
                    elif position:
                        transitions[position - 1].update({"set_start_frame": cursor, "set_end_frame": cursor,
                                                         "set_start_seconds": cursor / sample_rate,
                                                         "set_end_seconds": cursor / sample_rate})
                    body = clip["output_frames"] - incoming - outgoing
                    clip['solo_seconds'] = body / sample_rate
                    clip['solo_start_seconds'] = cursor / sample_rate
                    clip['solo_end_seconds'] = (cursor + body) / sample_rate
                    section = sections.get(clip['id']) or {}
                    if section.get('maximum_solo_seconds') is not None and clip['solo_seconds'] > section['maximum_solo_seconds'] + 1 / sample_rate:
                        raise RenderError(f"{clip['id']}: rendered solo exceeds its maximum duration")
                    if section.get('predicted_solo_seconds') is not None and abs(clip['solo_seconds'] - section['predicted_solo_seconds']) > .05:
                        raise RenderError(f"{clip['id']}: rendered solo disagrees with the section budget")
                    fade_in = not incoming
                    fade_out = not outgoing
                    _copy_region(original, dest, body, fade_in=fade_in, fade_out=fade_out)
                    if outgoing:
                        previous_tail = original.read(outgoing, dtype="float32", always_2d=True)
                        if clip.get("aligned_vocal_path"):
                            with sf.SoundFile(clip["aligned_vocal_path"]) as voice:
                                voice.seek(clip["output_frames"] - outgoing)
                                previous_vocal = voice.read(outgoing, dtype="float32", always_2d=True)
                        if len(previous_tail) != outgoing:
                            raise RenderError("Missing outgoing transition audio")
                cursor += clip["output_frames"] - incoming
        measurements = _measure(temporary, witness=dest, regions=zip(transitions,overlap_files))
        if sf.info(temporary).frames != cursor:
            raise RenderError("Rendered length differs from the segment accounting")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    warped = any(clip["warp"] is not None for clip in clips)
    creative_speed = any(clip.get("creative_warp") is not None for clip in clips)
    used_modes = sorted({transition["mode"] for transition in transitions})
    overlap_descriptions = {
        "beatmapped_bass_swap": "independent bass/mid/high curves; gentle mid entry and bounded incoming gain",
        "free_time_echo_bridge": "filtered finite echo with sequential dry-signal handoff at native tempos",
        "outgoing_only_vinyl": "short outgoing-only speed/pitch fall with simultaneous incoming native-tempo mix and bass handoff",
        "unverified_short_crossfade": "explicit short equal-power fade, no beatmatching",
        "bar_tempo_reset": "explicit zero-overlap cut with anti-click endpoint fades",
    }
    manifest = {"schema_version": 1, "renderer": "AI Super DJ sample-frame timemap renderer",
                "output_path": str(output), "sample_rate": sample_rate, "channels": 2,
                "subtype": "FLOAT", "duration_seconds": cursor / sample_rate, "total_frames": cursor,
                "track_count": len(order), "all_tracks_once": True, "order": order,
                "tracks": clips, "transitions": transitions, "measurements": measurements,
                "assembly_verification":{'verified':True,'pcm_sha256':dest.digest.hexdigest(),
                    'overlaps_verified':sum(bool(v) for v in overlap_files),'frames':dest.frames},
                "limits": {"max_musical_tempo_change": max_tempo_change,
                           "tempo_ramp_max_change": tempo_ramp_max_change,
                           "tempo_ramp_requested_output_seconds": tempo_ramp_seconds,
                           "native_exit_seconds": native_exit_seconds,
                           "sync_max_temporary_rate_change": sync_max_change,
                           "sync_transition_bars": sync_transition_bars,
                           "native_exit_max_slowdown": 0.35,
                           "max_local_timing_warp_change": max_local_warp_change,
                           "minimum_grid_confidence": minimum_grid_confidence,
                           "max_intro_trim_seconds": max_intro_trim_seconds,
                           "max_outro_trim_seconds": max_outro_trim_seconds,
                           "strict_grid": strict_grid, "allow_unverified": allow_unverified,
                           "allow_hard_cut": allow_hard_cut, "tempo_conflict_policy": tempo_conflict_policy,
                           "bridge_overlap_beats": bridge_overlap_beats},
                "processing": {"time_stretch": (f"external Rubber Band {rubberband_engine.upper()}, pitch preserved, explicit source→target frame map"
                                                if warped else "none; outgoing playback-speed effects are reported separately" if creative_speed else "none; native sample timing preserved"),
                               "time_stretch_executed": warped,
                               "playback_speed_effect_executed": creative_speed,
                               "playback_speed_effect": "outgoing-only windowed-sinc vinyl slowdown" if creative_speed else None,
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
