"""Full-track measurements and explicitly qualified musical timing estimates.

No excerpt is extrapolated into a verified beat grid. The lightweight backend
estimates beats only; first beats of bars require a model or an imported grid.
Analysis audio, records and model weights live outside the source repository.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = 1
ANALYSIS_VERSION = "superdj-analysis-1.4"
SAMPLE_RATE = 22050
HOP = 512
N_FFT = 2048
MODEL_BASE_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"
MODEL_SOURCE = "https://github.com/CPJKU/beat_this"
MODEL_LICENSE_SOURCE = MODEL_SOURCE + "#license"
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR_CAMELOT = ("8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B")
_MINOR_CAMELOT = ("5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A")


def default_cache_dir() -> Path:
    """User-local private cache, never a default inside the repository."""
    return Path(os.environ.get("SUPERDJ_CACHE", str(Path.home() / ".cache" / "superdj"))).expanduser().resolve()


def _cache_directory(path: str | Path | None) -> Path:
    directory = (Path(path).expanduser() if path is not None else default_cache_dir()).resolve()
    repository = Path(__file__).resolve().parents[1]
    if directory == repository or repository in directory.parents:
        raise ValueError("Analysis caches and audio must be outside the SuperDJ repository")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_beat_model(checkpoint: str = "final0", *, cache_dir: str | Path | None = None,
                       allow_download: bool = False) -> Path:
    """Resolve a local checkpoint; download official weights only with opt-in.

    The upstream README explicitly releases its published weights under MIT.
    No training audio is downloaded. SHA256 is recorded in each analysis record.
    """
    supplied = Path(checkpoint).expanduser()
    if supplied.is_file():
        return supplied.resolve()
    allowed = {f"{prefix}{seed}" for prefix in ("final", "small") for seed in range(3)}
    if checkpoint not in allowed:
        raise ValueError("Use an existing local checkpoint or final0/1/2, small0/1/2")
    target = _cache_directory(cache_dir) / "models" / f"{checkpoint}.ckpt"
    if target.is_file():
        return target
    if not allow_download:
        raise FileNotFoundError(f"Missing Beat This! weights: {target}. Explicitly enable model download.")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=checkpoint + "-", suffix=".partial", dir=target.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with urllib.request.urlopen(f"{MODEL_BASE_URL}/{checkpoint}.ckpt", timeout=90) as response, temporary_path.open("wb") as stream:
            header = response.headers.get("Content-Length")
            while block := response.read(1024 * 1024):
                stream.write(block)
        if temporary_path.stat().st_size < 1024 * 1024:
            raise ValueError("Unexpectedly small model download")
        if header is not None and temporary_path.stat().st_size != int(header):
            raise ValueError("Model download was incomplete")
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def _decode(path: Path, cache: Path, ffmpeg: str) -> tuple[Path, np.memmap]:
    fd, name = tempfile.mkstemp(prefix="analysis-", suffix=".f32", dir=cache)
    os.close(fd)
    temporary = Path(name)
    try:
        result = subprocess.run([str(ffmpeg), "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
                                 "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
                                 "-f", "f32le", "-y", str(temporary)], capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-2000:])
        if temporary.stat().st_size < 4 or temporary.stat().st_size % 4:
            raise ValueError("Decoder returned no usable audio")
        return temporary, np.memmap(temporary, dtype="<f4", mode="r")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _features(signal: np.ndarray) -> dict[str, Any]:
    """Stream the STFT across the entire recording, retaining compact features."""
    import librosa

    if len(signal) < N_FFT:
        signal = np.pad(signal, (0, N_FFT - len(signal)))
    frame_count = 1 + (len(signal) - N_FFT) // HOP
    rms = np.empty(frame_count, dtype=np.float32)
    flux = np.empty(frame_count, dtype=np.float32)
    mid_tonal = np.empty(frame_count, dtype=np.float32)
    centroid = np.empty(frame_count, dtype=np.float32)
    chroma_sum = np.zeros(12, dtype=np.float64)
    frequencies = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    mid = (frequencies >= 300) & (frequencies <= 3500)
    previous: np.ndarray | None = None
    for start_frame in range(0, frame_count, 1024):
        count = min(1024, frame_count - start_frame)
        start = start_frame * HOP
        end = start + (count - 1) * HOP + N_FFT
        block = np.asarray(signal[start:end], dtype=np.float32)
        magnitude = np.abs(librosa.stft(block, n_fft=N_FFT, hop_length=HOP, center=False))
        power = magnitude ** 2
        sl = slice(start_frame, start_frame + count)
        rms[sl] = librosa.feature.rms(S=magnitude, frame_length=N_FFT, hop_length=HOP, center=False)[0]
        logged = np.log1p(magnitude)
        before = np.concatenate([logged[:, :1] if previous is None else previous[:, None], logged[:, :-1]], axis=1)
        flux[sl] = np.mean(np.maximum(logged - before, 0), axis=0)
        previous = logged[:, -1]
        denominator = np.maximum(power.sum(axis=0), 1e-12)
        band = power[mid]
        flatness = np.exp(np.mean(np.log(np.maximum(band, 1e-12)), axis=0)) / np.maximum(np.mean(band, axis=0), 1e-12)
        mid_tonal[sl] = band.sum(axis=0) / denominator * (1 - flatness)
        centroid[sl] = (frequencies[:, None] * power).sum(axis=0) / denominator
        chroma_sum += librosa.feature.chroma_stft(S=power, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, tuning=0, norm=None).sum(axis=1)
    if np.max(flux) > 0:
        flux /= np.max(flux)
    # This proxy is sustained tonal energy in a voice-like frequency band;
    # guitars, synths and strings can also produce it. It is not vocal detection.
    sustained_mid = mid_tonal * (1 - np.minimum(flux, 1))
    return {"rms": rms, "flux": flux, "sustained_mid": sustained_mid,
            "centroid": centroid, "chroma_sum": chroma_sum,
            "times": (np.arange(frame_count) * HOP + N_FFT / 2) / SAMPLE_RATE}


def _key(chroma: np.ndarray) -> dict[str, Any]:
    if np.sum(chroma) < 1e-8 or np.std(chroma) < 1e-10:
        return {"key": None, "key_camelot": None, "key_confidence": 0.0, "key_alternatives": []}
    scores = []
    for mode, template, camelot in (("major", _MAJOR, _MAJOR_CAMELOT), ("minor", _MINOR, _MINOR_CAMELOT)):
        for tonic in range(12):
            correlation = float(np.corrcoef(chroma, np.roll(template, tonic))[0, 1])
            scores.append({"key": f"{_PITCHES[tonic]} {mode}", "key_camelot": camelot[tonic], "correlation": correlation})
    scores.sort(key=lambda row: row["correlation"], reverse=True)
    margin = scores[0]["correlation"] - scores[1]["correlation"]
    confidence = float(np.clip(max(0, scores[0]["correlation"]) * min(1, margin / 0.18), 0, 1))
    return {"key": scores[0]["key"], "key_camelot": scores[0]["key_camelot"],
            "key_confidence": round(confidence, 4), "key_correlation": round(scores[0]["correlation"], 4),
            "key_margin": round(margin, 4), "key_alternatives": scores[:3]}


def _times(values: Any, duration: float, field: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{field} must contain a finite one-dimensional time list")
    if np.any(result < 0) or np.any(result > duration + 0.05) or np.any(np.diff(result) <= 0):
        raise ValueError(f"{field} must be strictly increasing and within the audio duration")
    return result


def _imported_grid(grid: Mapping[str, Any], duration: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    beats = grid.get("beats", [])
    downbeats = grid.get("downbeats", [])
    generated = False
    if len(beats) == 0 and "bpm" in grid and "first_downbeat_s" in grid:
        bpm, anchor = float(grid["bpm"]), float(grid["first_downbeat_s"])
        meter = int(grid.get("meter", 4))
        if not math.isfinite(bpm) or not 20 <= bpm <= 400 or not 0 <= anchor <= duration or meter not in (3, 4):
            raise ValueError("Invalid imported constant-tempo grid")
        period = 60 / bpm
        first_index = -int(math.floor(anchor / period))
        indices = np.arange(first_index, int(math.floor((duration - anchor) / period)) + 1)
        beats = anchor + indices * period
        downbeats = anchor + indices[indices % meter == 0] * period
        generated = True
    beat_times = _times(beats, duration, "beats")
    downbeat_times = _times(downbeats, duration, "downbeats")
    if len(beat_times) < 2:
        raise ValueError("Imported grid requires at least two beats")
    if len(downbeat_times):
        right = np.clip(np.searchsorted(beat_times, downbeat_times), 0, len(beat_times) - 1)
        left = np.maximum(right - 1, 0)
        distances = np.minimum(np.abs(beat_times[left] - downbeat_times), np.abs(beat_times[right] - downbeat_times))
        if np.any(distances > 0.10):
            raise ValueError("Every imported downbeat must align to an imported beat within 100ms")
    reviewed = grid.get("reviewed") is True
    return beat_times, downbeat_times, {"status": "reviewed" if reviewed else "imported_unreviewed",
            "source": str(grid.get("source", "user-supplied grid")), "human_reviewed": reviewed,
            "constant_tempo_generated": generated, "meter_declared": grid.get("meter"),
            "first_downbeat_s": float(downbeat_times[0]) if len(downbeat_times) else None}


def _decode_meter(probabilities: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Original phase decoder constrained by neural downbeat evidence.

    Competes 3/4-beat meters. The starting phase is free and is inferred from
    logits; advancing one phase per detected beat is preferred. Costly phase
    corrections accommodate missing/extra beat events. Weak or ambiguous
    evidence returns no decoded grid. This is not madmom's DBN implementation.
    """
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError("Neural probabilities must be a finite vector")
    if np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("Neural probabilities must lie between 0 and 1")
    probability = np.clip(probability, 0.01, 0.99)
    if len(probability) < 16:
        return None, {"accepted": False, "reason": "too few beat events for meter inference"}
    candidates = []
    for meter in (3, 4):
        emission = np.repeat(np.log1p(-probability)[:, None], meter, axis=1)
        emission[:, 0] = np.log(probability)
        scores = emission[0].copy()
        back = np.zeros((len(probability), meter), dtype=np.int8)
        previous_states = np.arange(meter)
        for index in range(1, len(probability)):
            next_scores = np.empty(meter)
            for phase in range(meter):
                transition = np.where((previous_states + 1) % meter == phase, 0.0, -3.0)
                options = scores + transition
                previous = int(np.argmax(options))
                back[index, phase] = previous
                next_scores[phase] = options[previous] + emission[index, phase]
            scores = next_scores
        phase = int(np.argmax(scores))
        score = float(scores[phase] / len(probability))
        phases = np.empty(len(probability), dtype=np.int8)
        phases[-1] = phase
        for index in range(len(probability) - 1, 0, -1):
            phase = int(back[index, phase])
            phases[index - 1] = phase
        selected = phases == 0
        downbeat_mean = float(np.mean(probability[selected]))
        other_mean = float(np.mean(probability[~selected]))
        corrections = int(np.count_nonzero(phases[1:] != (phases[:-1] + 1) % meter))
        candidates.append({"meter": meter, "score": score, "phases": phases,
                           "downbeat_probability_mean": downbeat_mean,
                           "other_beat_downbeat_probability_mean": other_mean,
                           "phase_contrast": downbeat_mean - other_mean,
                           "phase_corrections": corrections,
                           "phase_correction_fraction": corrections / (len(probability) - 1)})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    winner, runner_up = candidates
    margin = winner["score"] - runner_up["score"]
    accepted = (winner["downbeat_probability_mean"] >= 0.45 and winner["phase_contrast"] >= 0.12
                and margin >= 0.015 and winner["phase_correction_fraction"] <= 0.08)
    details = {key: round(value, 6) if isinstance(value, float) else value
               for key, value in winner.items() if key != "phases"}
    details.update({"accepted": bool(accepted), "meter_score_margin": round(margin, 6),
                    "alternative_meter": runner_up["meter"], "alternative_score": round(runner_up["score"], 6),
                    "method": "SuperDJ Viterbi phase decoder driven by Beat This! downbeat logits; meters 3/4",
                    "confidence_kind": "heuristic evidence separation; not calibrated meter probability"})
    if not accepted:
        details["reason"] = "insufficient neural contrast, ambiguous meter, or too many phase corrections"
    return np.flatnonzero(winner["phases"] == 0) if accepted else None, details


def _beat_this(signal: np.ndarray, checkpoint: Path, device: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import torch
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor
    except ImportError as error:
        raise RuntimeError("Install optional beat-this, torch and torchaudio dependencies to use the model backend") from error
    # A CPU profile should not spawn dozens of workers for a single track.
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    tracker = Audio2Frames(checkpoint_path=str(checkpoint), device=device, float16=False)
    beat_logits, downbeat_logits = tracker(np.asarray(signal), SAMPLE_RATE)
    beats, raw_downbeats = Postprocessor(type="minimal")(beat_logits, downbeat_logits)
    beats = np.asarray(beats, dtype=float)
    frame_indices = np.rint(beats * 50).astype(int)
    probabilities = downbeat_logits.sigmoid().cpu().numpy()
    beat_probabilities = beat_logits.sigmoid().cpu().numpy()
    # The model uses 50Hz frames, while minimal processing moves downbeats to
    # beat times. Retain the strongest nearby neural evidence within 40ms.
    event_probability = np.array([np.max(probabilities[max(0, index - 2):min(len(probabilities), index + 3)])
                                  for index in frame_indices], dtype=float)
    event_beat_probability = np.array([np.max(beat_probabilities[max(0, index - 2):min(len(beat_probabilities), index + 3)])
                                       for index in frame_indices], dtype=float)
    indices, meter_details = _decode_meter(event_probability)
    downbeats = beats[indices] if indices is not None else np.asarray(raw_downbeats, dtype=float)
    return beats, downbeats, {"raw_model_downbeats": [round(float(value), 6) for value in raw_downbeats],
                              "metrical_decode": meter_details,
                              "neural_downbeat_probability_by_beat": [round(float(value), 6) for value in event_probability],
                              "neural_beat_probability_by_beat": [round(float(value), 6) for value in event_beat_probability]}


def _robust_tempo(beats: np.ndarray) -> dict[str, Any]:
    """Reduce frame-quantisation bias without guessing an integer or an octave."""
    intervals = np.diff(np.asarray(beats, dtype=float))
    if len(intervals) == 0:
        return {"bpm": None, "period": None, "tempo_inlier_fraction": 0.0}
    span = min(8, len(beats) - 1)
    seed_period = float(np.median((beats[span:] - beats[:-span]) / span))
    inliers = np.abs(intervals - seed_period) <= seed_period * 0.15
    if not np.any(inliers):
        return {"bpm": None, "period": None, "tempo_inlier_fraction": 0.0}
    period = float(np.mean(intervals[inliers]))
    local = []
    # Local fits expose genuine drift and tempo changes instead of calling a
    # single global BPM a promise of a uniform grid.
    for index in range(0, max(1, len(intervals) - 15), 8):
        selected = intervals[index:index + 16]
        mask = inliers[index:index + 16]
        if len(selected) >= 8 and np.mean(mask) >= 0.90:
            local.append(60 / float(np.mean(selected[mask])))
    p05, p95 = (np.quantile(local, [0.05, 0.95]).tolist() if local else [None, None])
    return {"bpm": round(60 / period, 3), "period": period,
            "bpm_before_quantisation_refinement": round(60 / seed_period, 3),
            "tempo_inlier_fraction": round(float(np.mean(inliers)), 4),
            "local_bpm_p05": round(p05, 3) if p05 is not None else None,
            "local_bpm_p95": round(p95, 3) if p95 is not None else None,
            "tempo_method": "mean detected-beat interval within +/-15% of robust multi-beat period; no integer or octave snapping"}


def refine_tempo_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Cheap upgrade of already measured beat times; no audio/model is rerun."""
    result = dict(record)
    tempo = _robust_tempo(np.asarray(record.get("beats", []), dtype=float))
    result["bpm"] = tempo["bpm"]
    result["tempo_alternatives"] = [tempo["bpm"] / 2, tempo["bpm"] * 2] if tempo["bpm"] else []
    result["bpm_confidence"] = round(float(record.get("grid_confidence", 0)) * tempo["tempo_inlier_fraction"], 4)
    result["grid"] = {**record.get("grid", {}), **{key: value for key, value in tempo.items() if key != "period"}}
    result["provenance"] = {**record.get("provenance", {}),
                            "tempo_refinement_version": ANALYSIS_VERSION,
                            "tempo_refinement_source": "original full-track detected beats; no audio inference repeated"}
    return result


def _grid_diagnostics(beats: np.ndarray, downbeats: np.ndarray, features: dict[str, Any]) -> dict[str, Any]:
    intervals = np.diff(beats)
    if len(intervals) == 0:
        return {"bpm": None, "confidence": 0.0, "regularity": 0.0, "onset_alignment": 0.0,
                "meter": None, "meter_consistency": 0.0, "timing_jitter_seconds": None}
    tempo = _robust_tempo(beats)
    median = tempo["period"]
    if median is None:
        return {"bpm": None, "confidence": 0.0, "regularity": 0.0, "onset_alignment": 0.0,
                "meter": None, "meter_consistency": 0.0, "timing_jitter_seconds": None,
                "tempo_inlier_fraction": 0.0, "warning": "No coherent detected-beat period"}
    jitter = float(np.median(np.abs(intervals - median)))
    regularity = float(np.exp(-6 * max(0, jitter - 0.025) / max(median, 1e-6)))
    outlier_fraction = float(np.mean(np.abs(intervals - median) > max(0.08, median * 0.15)))
    regularity *= 1 - outlier_fraction
    # Windowed STFT flux spreads a transient in time; allow its local 60ms
    # neighbourhood, rather than treating frame-centre quantisation as drift.
    alignment = np.maximum.reduce([np.interp(beats + offset, features["times"], features["flux"], left=0, right=0)
                                   for offset in (-0.06, -0.03, 0, 0.03, 0.06)])
    background = max(float(np.quantile(features["flux"], 0.80)), 0.05)
    onset_alignment = float(np.clip(np.median(alignment) / background, 0, 1))
    meter, consistency = None, 0.0
    downbeat_alignment_max = None
    downbeat_alignment_median = None
    if len(downbeats) >= 3:
        indices = np.searchsorted(beats, downbeats)
        indices = np.clip(indices, 0, len(beats) - 1)
        left = np.maximum(indices - 1, 0)
        indices = np.where(np.abs(beats[left] - downbeats) < np.abs(beats[indices] - downbeats), left, indices)
        distances = np.abs(beats[indices] - downbeats)
        downbeat_alignment_max = float(np.max(distances))
        downbeat_alignment_median = float(np.median(distances))
        counts = np.diff(indices)
        candidates = [(m, float(np.mean(counts == m))) for m in (3, 4)]
        meter, consistency = max(candidates, key=lambda item: item[1])
        if consistency < 0.80 or downbeat_alignment_max > 0.10:
            meter = None
    confidence = regularity * (0.45 + 0.55 * onset_alignment)
    if len(downbeats):
        confidence *= 0.6 + 0.4 * consistency
        if downbeat_alignment_max is not None:
            confidence *= math.exp(-10 * max(0, downbeat_alignment_max - 0.03))
    return {**{key: value for key, value in tempo.items() if key != "period"}, "confidence": round(float(confidence), 4),
            "regularity": round(regularity, 4), "onset_alignment": round(onset_alignment, 4),
            "meter": meter, "meter_consistency": round(consistency, 4),
            "timing_jitter_seconds": round(jitter, 6), "interval_outlier_fraction": round(outlier_fraction, 4),
            "downbeat_alignment_max_seconds": round(downbeat_alignment_max, 6) if downbeat_alignment_max is not None else None,
            "downbeat_alignment_median_seconds": round(downbeat_alignment_median, 6) if downbeat_alignment_median is not None else None}


def _cue_candidates(beats: np.ndarray, downbeats: np.ndarray, features: dict[str, Any],
                    duration: float, grid_status: str, diagnostics: dict[str, Any],
                    imported: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Estimate useful boundaries; a downbeat is not automatically a phrase start."""
    if imported is not None and imported.get("cue_candidates"):
        result = []
        for item in imported["cue_candidates"]:
            time = float(item.get("time", item.get("seconds", -1)))
            if not math.isfinite(time) or not 0 <= time < duration:
                raise ValueError("Imported cue time lies outside the audio")
            reviewed = imported.get("reviewed") is True and item.get("reviewed") is True
            result.append({**item, "time": round(time, 6), "seconds": round(time, 6),
                           "status": "reviewed" if reviewed else "imported_unreviewed",
                           "confidence": 1.0 if reviewed else float(item.get("confidence", 0.5)),
                           "method": "imported cue"})
        return sorted(result, key=lambda row: row["time"])
    anchors = downbeats if len(downbeats) else beats
    if len(anchors) < 2:
        return []
    is_downbeat = bool(len(downbeats))
    is_bar = is_downbeat and diagnostics.get("meter") in (3, 4)
    count = len(anchors)
    times = features["times"]
    rms = np.interp(anchors, times, features["rms"])
    tonal = np.interp(anchors, times, features["sustained_mid"])
    energy = 20 * np.log10(np.maximum(rms, 1e-7))
    novelty = np.zeros(count)
    for index in range(count):
        before = energy[max(0, index - 4):index]
        after = energy[index:min(count, index + 4)]
        if len(before) and len(after):
            novelty[index] = abs(float(np.mean(after) - np.mean(before)))
    if np.max(novelty) > 0:
        novelty /= np.max(novelty)
    region_size = min(120.0, duration * 0.35)
    result = []
    for index, time in enumerate(anchors):
        if time < region_size:
            kind = "intro"
        elif time > duration - region_size:
            kind = "outro"
        else:
            kind = "body"
        score = float(0.45 * novelty[index] + 0.30 * (1 - tonal[index]) + 0.25 * diagnostics["confidence"])
        beat_index = int(np.argmin(np.abs(beats - time))) if len(beats) else None
        neural = diagnostics.get("neural_downbeat_probability_by_beat", [])
        neural_probability = float(neural[beat_index]) if neural and beat_index is not None else None
        if neural_probability is not None:
            score *= 0.65 + 0.35 * neural_probability
        result.append({"time": round(float(time), 6), "seconds": round(float(time), 6), "kind": kind,
                       "bar_index": int(index) if is_bar else None,
                       "beat_index": beat_index,
                       "downbeat_model_probability": round(neural_probability, 4) if neural_probability is not None else None,
                       "bars_available_after": int(count - index - 1) if is_bar else None,
                       "phrase_lengths_bars": [n for n in (8, 16, 32) if is_bar and count - index - 1 >= n],
                       "grid_anchor": "downbeat" if is_downbeat else "beat",
                       "status": "estimated_phrase" if is_bar else "estimated_boundary", "confidence": round(score, 4),
                       "novelty": round(float(novelty[index]), 4),
                       "sustained_midband_proxy": round(float(tonal[index]), 4),
                       "method": "bar-synchronous RMS change and sustained midband proxy" if is_bar else "beat-synchronous RMS change; bar position unknown"})
    # Keep multiple alternatives in each region, rather than only high-energy body changes.
    selected = []
    for kind in ("intro", "body", "outro"):
        ranked = sorted((row for row in result if row["kind"] == kind), key=lambda row: row["confidence"], reverse=True)
        for row in ranked:
            if all(abs(row["time"] - previous["time"]) > 2.0 for previous in selected):
                selected.append(row)
            if sum(item["kind"] == kind for item in selected) >= 16:
                break
    return sorted(selected, key=lambda row: row["time"])


def analyze_track(path: str | Path, *, ffmpeg: str = "ffmpeg", cache_dir: str | Path | None = None,
                  backend: str = "librosa", checkpoint: str = "final0", device: str = "cpu",
                  grid: Mapping[str, Any] | None = None, track_id: str | None = None,
                  allow_model_download: bool = False, use_cache: bool = True) -> dict[str, Any]:
    """Measure a complete track and return a JSON-serializable analysis record.

    `grid` can contain explicit beats/downbeats, or bpm + first_downbeat_s +
    meter for an imported constant-tempo grid. Only `reviewed: true` explicitly
    supplied by a human marks that grid reviewed. Model output stays predicted.
    `grid_confidence` and `key_confidence` are heuristic diagnostics, not
    probabilities of correct timing/key. Phrase candidates are always estimates
    unless separately imported and reviewed.
    """
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("An audio file is required")
    if backend not in ("librosa", "beat_this"):
        raise ValueError("backend must be librosa or beat_this")
    cache = _cache_directory(cache_dir)
    source_hash = _sha256(source)
    model_path = prepare_beat_model(checkpoint, cache_dir=cache, allow_download=allow_model_download) if backend == "beat_this" and grid is None else None
    model_hash = _sha256(model_path) if model_path else None
    options = {"version": ANALYSIS_VERSION, "source_sha256": source_hash, "backend": backend,
               "model_sha256": model_hash, "device": device, "grid": grid, "track_id": track_id}
    cache_key = hashlib.sha256(json.dumps(options, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    record_path = cache / "analysis" / f"{cache_key}.json"
    if use_cache and record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["path"] = str(source)
        return record
    temporary, audio = _decode(source, cache, ffmpeg)
    try:
        duration = len(audio) / SAMPLE_RATE
        features = _features(audio)
        if grid is not None:
            beats, downbeats, grid_info = _imported_grid(grid, duration)
        elif backend == "beat_this":
            beats, downbeats, model_info = _beat_this(audio, model_path, device)
            beats, downbeats = _times(beats, duration, "beats"), _times(downbeats, duration, "downbeats")
            grid_info = {"status": "model_predicted", "source": MODEL_SOURCE, "human_reviewed": False, **model_info,
                         "first_downbeat_s": float(downbeats[0]) if len(downbeats) else None}
        else:
            import librosa

            if np.max(features["flux"]) <= 1e-9:
                beats = np.array([], dtype=float)
            else:
                _, frames = librosa.beat.beat_track(onset_envelope=features["flux"], sr=SAMPLE_RATE,
                                                  hop_length=HOP, trim=False)
                beats = np.asarray(frames, dtype=float) * HOP / SAMPLE_RATE + N_FFT / 2 / SAMPLE_RATE
                beats = beats[beats < duration]
            downbeats = np.array([], dtype=float)
            grid_info = {"status": "estimated_beats_only", "source": "librosa beat_track on complete-track spectral flux",
                         "human_reviewed": False, "first_downbeat_s": None}
        diagnostics = _grid_diagnostics(beats, downbeats, features)
        if grid_info.get("metrical_decode", {}).get("accepted") is False:
            diagnostics["meter"] = None
        if grid_info.get("meter_declared") is not None:
            diagnostics["meter_declared"] = int(grid_info["meter_declared"])
            if grid_info["human_reviewed"] and diagnostics["meter_declared"] in (3, 4):
                diagnostics["meter"] = diagnostics["meter_declared"]
                diagnostics["meter_method"] = "explicit human-reviewed import"
        rms = float(np.sqrt(np.mean(features["rms"].astype(np.float64) ** 2)))
        rms_db = 20 * math.log10(max(rms, 1e-12))
        chroma_descriptor = features["chroma_sum"] / max(float(features["chroma_sum"].sum()), 1e-12)
        descriptor = np.concatenate([chroma_descriptor,
                                     [np.mean(features["centroid"]) / (SAMPLE_RATE / 2),
                                      np.mean(features["sustained_mid"]), np.mean(features["flux"])]]).tolist()
        warnings = ["Key and grid confidence are heuristic diagnostics, not calibrated probabilities.",
                    "Phrase boundaries are estimates; 8/16/32 bar grouping is not musical ground truth.",
                    "Sustained midband proxy is not a vocal detector."]
        if not len(downbeats):
            warnings.append("No downbeats: bar/phrase alignment is unavailable; do not synthesize every fourth beat.")
        if diagnostics["meter"] is None and len(downbeats):
            warnings.append("Stable meter was not established from predicted downbeats.")
        if (diagnostics.get("downbeat_alignment_max_seconds") or 0) > 0.10:
            warnings.append("Downbeat predictions do not align to beat predictions within 100ms.")
        if diagnostics["confidence"] < 0.75:
            warnings.append("Timing needs review before a strict beat/bar-aligned render.")
        record = {"schema_version": SCHEMA_VERSION, "id": track_id or source_hash[:16], "path": str(source),
                  "duration": round(duration, 6), "duration_seconds": round(duration, 6),
                  "bpm": diagnostics["bpm"], "bpm_confidence": round(diagnostics["confidence"] * diagnostics.get("tempo_inlier_fraction", 0), 4),
                  "tempo_alternatives": [diagnostics["bpm"] / 2, diagnostics["bpm"] * 2] if diagnostics["bpm"] else [],
                  **_key(features["chroma_sum"]), "energy": round(float(np.clip((rms_db + 36) / 30, 0, 1)), 4),
                  "energy_method": "RMS level proxy mapped -36..-6dBFS to 0..1; not perceived emotion",
                  "rms_dbfs": round(rms_db, 3), "spectral_centroid_hz": round(float(np.mean(features["centroid"])), 3),
                  "embedding": {"values": [round(float(value), 6) for value in descriptor],
                                "method": "handcrafted complete-track chroma, spectral and rhythm descriptor; not a learned embedding"},
                  "beats": [round(float(value), 6) for value in beats], "downbeats": [round(float(value), 6) for value in downbeats],
                  "cue_candidates": _cue_candidates(beats, downbeats, features, duration, grid_info["status"],
                                                    {**diagnostics, "neural_downbeat_probability_by_beat": grid_info.get("neural_downbeat_probability_by_beat", [])}, grid),
                  "grid_status": grid_info["status"], "grid_confidence": diagnostics["confidence"],
                  "grid": {**grid_info, **diagnostics, "confidence_kind": "heuristic timing regularity and onset alignment; not model posterior"},
                  "warnings": warnings,
                  "provenance": {"analysis_version": ANALYSIS_VERSION, "source_sha256": source_hash,
                                 "source_bytes": source.stat().st_size, "scope": "complete recording",
                                 "analysis_sample_rate": SAMPLE_RATE, "backend": backend if grid is None else "imported_grid",
                                 "model_checkpoint": str(model_path) if model_path else None, "model_sha256": model_hash,
                                 "model_code_and_weights_license": "MIT (upstream README assertion)" if model_path else None,
                                 "model_license_source": MODEL_LICENSE_SOURCE if model_path else None,
                                 "librosa_version": importlib.metadata.version("librosa")}}
        record_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=cache_key + "-", suffix=".partial", dir=record_path.parent)
        os.close(fd)
        atomic = Path(name)
        try:
            atomic.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            atomic.replace(record_path)
        finally:
            atomic.unlink(missing_ok=True)
        return record
    finally:
        audio._mmap.close()
        del audio
        temporary.unlink(missing_ok=True)
