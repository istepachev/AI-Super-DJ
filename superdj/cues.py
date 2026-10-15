"""Local CUE-DETR inference and original structural/grid fusion.

No music leaves the workstation. Relative DETR scores are not probabilities.
The worker can use a separate Python environment (notably NumPy 1.x).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

MODEL_SHA256 = "195b80f5c94e4b424b417a7c04e0aac238082c584ddcb65d01be8a97c69cadd3"
MODEL_ID = "disco-eth/cue-detr"
_MODELS = {}


def download_checkpoint(directory):
    """Fetch only the three audited model/processor files, never music."""
    from urllib.request import urlopen
    from .storage import private_path
    root = private_path(directory)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "model.safetensors": (f"https://huggingface.co/{MODEL_ID}/resolve/main/model.safetensors", MODEL_SHA256),
        "config.json": (f"https://huggingface.co/{MODEL_ID}/resolve/main/config.json", "f83ea58534fb101f52f4cd6ca35a6daa978c8414e629f8483f2665f02f574077"),
        "preprocessor_config.json": ("https://huggingface.co/facebook/detr-resnet-50/resolve/main/preprocessor_config.json", "0673fea2a6d3cf92cdbab3c7426c0ecdf8a4729a2a4d5199033dcd66a2b8759b")}
    for name, (url, expected) in files.items():
        target = root / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        partial = root / (name + ".partial")
        try:
            with urlopen(url, timeout=90) as response, partial.open("wb") as dest:
                while block := response.read(1024 * 1024):
                    dest.write(block)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
                raise ValueError("Downloaded CUE-DETR file differs from audited content")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    return root


def runtime_config() -> dict:
    path = Path.home() / ".cache" / "superdj" / "runtime.local.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def configuration() -> dict | None:
    runtime = runtime_config()
    directory = os.environ.get("SUPERDJ_CUE_MODEL") or runtime.get("cue_model")
    if not directory:
        default = Path.home() / ".cache" / "superdj" / "cue-detr"
        if (default / "model.safetensors").is_file():
            directory = str(default)
    if not directory:
        return None
    model = Path(directory).expanduser().resolve(strict=True)
    from .execution import current, sha256
    digest_file = current().digest if current() else sha256
    digest = digest_file(model / "model.safetensors")
    if digest != MODEL_SHA256:
        raise ValueError("CUE-DETR checkpoint differs from the audited weights")
    for name, expected in {
        "config.json": "f83ea58534fb101f52f4cd6ca35a6daa978c8414e629f8483f2665f02f574077",
        "preprocessor_config.json": "0673fea2a6d3cf92cdbab3c7426c0ecdf8a4729a2a4d5199033dcd66a2b8759b",
    }.items():
        if digest_file(model / name) != expected:
            raise ValueError("CUE-DETR configuration differs from audited content")
    return {"model": str(model), "sha256": digest,
            "python": os.environ.get("SUPERDJ_CUE_PYTHON") or runtime.get("cue_python") or sys.executable,
            "device": runtime.get("cue_device", "auto")}


def infer(source: str | Path, cache: Path, config: dict) -> dict:
    from .audio import ffmpeg_executable, DECODER_VERSION
    config = {**config, 'ffmpeg': ffmpeg_executable(config.get('ffmpeg')), 'audio_decoder': DECODER_VERSION}
    directory = cache / "cue-detr-jobs"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="infer-", dir=directory) as temporary:
        job = Path(temporary) / "job.json"
        result = Path(temporary) / "result.json"
        job.write_text(json.dumps({**config, "source": str(source)}), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONUTF8="1")
        from .execution import current
        if current() is not None:
            return current().model(config["python"], "cue", {**config, "source": str(source)})
        process = subprocess.run([config["python"], "-m", "superdj.cues", str(job), str(result)],
                                 env=env, capture_output=True)
        if process.returncode:
            raise RuntimeError("Local CUE-DETR inference failed: " + process.stderr.decode("utf-8", "replace")[-2000:])
        return json.loads(result.read_text(encoding="utf-8"))


def _worker(job: dict) -> dict:
    import importlib.metadata
    import librosa
    from matplotlib import cm
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor
    from scipy.signal import find_peaks

    from .audio import decode_mono, ffmpeg_executable, DECODER_VERSION
    rate = 22050
    decoder = ffmpeg_executable(job.get('ffmpeg'))
    audio = decode_mono(job['source'], sample_rate=rate, ffmpeg=decoder)
    mel = librosa.feature.melspectrogram(y=audio, sr=rate, n_fft=2048, hop_length=512, n_mels=128)
    db = librosa.power_to_db(mel, ref=np.max)[::-1]
    mapper = cm.ScalarMappable(cmap="viridis")
    mapper.set_clim(db.min(), db.max())
    pixels = mapper.to_rgba(db, bytes=True)[:, :, :3]
    images, borders = [], []
    for index in range(int(np.floor((pixels.shape[1] + 266) / (355 * .25)))):
        left = int(np.floor(index * 355 * .25)) - 266
        right = left + 355
        segment = pixels[:, max(0, left):min(right, pixels.shape[1])]
        segment = np.pad(segment, ((0, 0), (max(0, -left), max(0, right - pixels.shape[1])), (0, 0)), mode="linear_ramp")
        if segment.shape[1] != 355:
            raise ValueError("Invalid spectrogram window")
        images.append(segment)
        borders.append(left)
    device = job.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    signature = (job['model'], job.get('sha256'), device)
    if signature not in _MODELS:
        processor = DetrImageProcessor.from_pretrained(job["model"], local_files_only=True)
        model, loading = DetrForObjectDetection.from_pretrained(job["model"], local_files_only=True,
            use_safetensors=True, use_pretrained_backbone=False, output_loading_info=True)
        if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
            raise ValueError("Incomplete CUE-DETR weight loading")
        model.to(device).eval()
        _MODELS[signature] = (processor, model)
    processor, model = _MODELS[signature]
    positions, raw_scores = [], []
    # Whole-track normalization; bounded inference batches can differ slightly
    # from upstream's single batch. Keep the executed batch size in provenance.
    batch_size = 32
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            encoding = processor.preprocess(batch, do_resize=False, return_tensors="pt")
            output = model(encoding["pixel_values"].to(device))
            predictions = processor.post_process_object_detection(output, 0, [(128, 355)] * len(batch))
            for prediction, left in zip(predictions, borders[start:start + batch_size]):
                raw_scores.extend(prediction["scores"].cpu().tolist())
                positions.extend((((prediction["boxes"][:, 0] + prediction["boxes"][:, 2]) // 2 + left).long()).cpu().tolist())
    scores = np.asarray(raw_scores)
    relative = (scores - scores.min()) / max(float(np.ptp(scores)), 1e-12)
    ordered = sorted(zip(positions, relative.tolist(), scores.tolist()))
    peaks, _ = find_peaks([row[1] for row in ordered], height=.9, distance=16)
    candidates = [{"time": ordered[i][0] * 512 / rate, "relative_score": ordered[i][1],
                   "raw_score": ordered[i][2]} for i in peaks
                  if 0 <= ordered[i][0] * 512 / rate < len(audio) / rate]
    return {"candidates": candidates, "provenance": {"executed": True, "model": MODEL_ID,
            "model_sha256": job["sha256"], "device": device, "batch_size": batch_size,
            "preprocessing": "FFmpeg mono float PCM at 22050Hz; librosa 128 mel bins, 2048 FFT/512 hop, inverted viridis RGB, 355px windows/266 padding",
            "audio_decoder": {"method": DECODER_VERSION, "executable": decoder, "sample_rate": rate, "frames": len(audio), "stream": "0:a:0"},
            "selection": "upstream normalized-score peaks; distance=16 query indices, not musical bars",
            "score_kind": "within-track min/max normalized object scores; not calibrated cue probabilities",
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "librosa", "numpy")}}}


def hybrid_candidates(existing: list[dict], downbeats, features: dict, duration: float,
                      predictions: dict | None, meter: int | None = 4) -> list[dict]:
    """Fuse observed structure with nearby model cues on the supplied bar grid.

    Never move imported decisions. A model cue away from a structural/grid
    boundary remains in the model report, rather than becoming a locked cue.
    """
    if existing and all(row.get("method") == "imported cue" for row in existing):
        return existing
    anchors = np.asarray(downbeats, dtype=float)
    if len(anchors) < 3 or meter not in (3, 4):
        return existing
    # Compare neighbouring four-bar blocks using independent band changes.
    series = []
    for name in ("rms", "bass_rms", "high_rms"):
        values = features.get(name, features["rms"])
        series.append(20 * np.log10(np.maximum(np.interp(anchors, features["times"], values), 1e-7)))
    novelty = np.zeros(len(anchors))
    for i in range(1, len(anchors) - 1):
        changes = [abs(np.mean(x[i:min(i + 4, len(x))]) - np.mean(x[max(0, i - 4):i])) for x in series]
        novelty[i] = float(np.mean(changes))
    novelty /= max(float(novelty.max()), 1e-9)
    model_cues = (predictions or {}).get("candidates", [])
    result = {float(row["time"]): dict(row) for row in existing}
    for index in range(1, len(anchors) - 1):
        time = float(anchors[index])
        local_maximum = novelty[index] >= max(novelty[max(0, index - 1):index + 2])
        nearby = [cue for cue in model_cues if abs(cue["time"] - time) <= min(anchors[index] - anchors[index - 1], anchors[index + 1] - anchors[index])]
        if not local_maximum or (novelty[index] < .30 and not nearby):
            continue
        key = round(time, 6)
        if any((row.get("method") == "imported cue" or row.get("reviewed")) and
               abs(float(row["time"]) - time) <= .05 for row in existing):
            continue
        row = result.get(key, {"time": key, "seconds": key,
            "kind": "intro" if time < min(120, duration * .35) else "outro" if time > duration - min(120, duration * .35) else "body",
            "bar_index": index, "grid_anchor": "downbeat", "status": "estimated_phrase"})
        tonal = float(np.interp(time, features["times"], features["sustained_mid"]))
        row.update({"novelty": round(float(novelty[index]), 4), "sustained_midband_proxy": tonal,
                    "method": "audio band structure + supplied bar grid" + (" + CUE-DETR" if nearby else ""),
                    "confidence": min(.95, .35 + .30 * float(novelty[index]) + .20 * (1 - tonal) + (.10 if nearby else 0)),
                    "cue_detr_evidence": nearby, "musical_phrase_verified": False})
        result[key] = row
    return sorted(result.values(), key=lambda row: row["time"])


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(_worker(job), ensure_ascii=False), encoding="utf-8")
