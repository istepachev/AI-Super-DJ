"""Private local vocal estimates for cue scoring and selective handoffs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .cues import runtime_config
_MODELS = {}


def configuration():
    runtime = runtime_config()
    python = os.environ.get("SUPERDJ_STEM_PYTHON") or runtime.get("stem_python")
    if not python:
        from importlib.util import find_spec
        if find_spec("demucs") is None:
            return None
        python = sys.executable
    from .execution import current, sha256, environment
    digest = current().digest if current() else sha256
    weights = {p.name: digest(p) for p in (Path(os.getenv('TORCH_HOME') or Path(os.getenv('XDG_CACHE_HOME') or Path.home()/'.cache')/'torch')/'hub/checkpoints').glob('955717e8-*.th')}
    from .execution import stage_fingerprint
    return {"python": python, "device": runtime.get("stem_device", "auto"), "model": "htdemucs", "version": "4.0.1", "include_drums": True, "weights": weights, "environment": environment(python),
            "adapter_engine": current().objects.engine_for('stem') if current() else stage_fingerprint('stem')}


def separate(source, source_hash, cache, config, ffmpeg):
    from .execution import object_lock, key_for
    from .audio import ffmpeg_executable
    ffmpeg = ffmpeg_executable(ffmpeg)
    cache = Path(cache)
    key = key_for({'source': source_hash, 'config': config, 'ffmpeg': ffmpeg})
    with object_lock(cache / 'stem-locks' / (key + '.lock')):
        return _separate(source, source_hash, cache, config, ffmpeg)


def _separate(source, source_hash, cache, config, ffmpeg):
    from .storage import write_json
    key = hashlib.sha256(json.dumps({"source": source_hash, "config": config, "ffmpeg": str(ffmpeg), "adapter": 2}, sort_keys=True).encode()).hexdigest()
    directory = cache / "vocals" / key
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory / "result.json"
    if result_file.is_file():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            result = {}
        from .execution import sha256
        hashes = result.get('artifact_sha256', {}) if isinstance(result, dict) else {}
        try:
            valid = hashes and all(sha256(result[name]) == digest for name, digest in hashes.items())
        except (OSError, KeyError):
            valid = False
        if valid and 'path' in hashes and (not config.get('include_drums') or 'drums_path' in hashes):
            return result
    job_file = directory / "job.json"
    write_json(job_file, {**config, "source": str(source), "ffmpeg": ffmpeg,
                         "output": str(directory / "vocals.wav"),
                         **({'drums_output':str(directory/'drums.wav')} if config.get('include_drums') else {})})
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONUTF8="1")
    from .execution import current, sha256
    if current() is not None:
        result = current().model(config['python'], 'stem', json.loads(job_file.read_text(encoding='utf8')))
        result['artifact_sha256'] = {name: sha256(result[name]) for name in ('path', 'drums_path') if result.get(name)}
        write_json(result_file, result)
        return result
    result = subprocess.run([config["python"], "-m", "superdj.stems", str(job_file), str(result_file)], env=env, capture_output=True)
    (directory / "inference.log").write_bytes(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Local vocal separation failed: " + result.stderr.decode("utf-8", "replace")[-2000:])
    result = json.loads(result_file.read_text(encoding="utf-8"))
    result['artifact_sha256'] = {name: sha256(result[name]) for name in ('path', 'drums_path') if result.get(name)}
    write_json(result_file, result)
    return result


def _worker(job):
    import importlib.metadata
    import torch
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from .quality import execute

    if importlib.metadata.version("demucs") != "4.0.1":
        raise ValueError("Vocal adapter requires audited Demucs 4.0.1")
    device = job.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    signature = (device, json.dumps(job.get("weights", {}), sort_keys=True))
    if signature not in _MODELS:
        _MODELS[signature] = get_model("htdemucs").to(device).eval()
    model = _MODELS[signature]
    rate = model.samplerate
    decoded = Path(job["output"]).with_name("decoded.wav")
    try:
        if job.get('decoded_input'):
            from .audio import read_decoded
            audio = read_decoded(job['decoded_input'], sample_rate=rate, channels=2)
        else:
            execute([job["ffmpeg"], "-nostdin", "-v", "error", "-y", "-i", job["source"],
                     "-map", "0:a:0", "-ac", "2", "-ar", str(rate), "-c:a", "pcm_f32le", str(decoded)])
            audio, rate = sf.read(decoded, dtype="float32", always_2d=True)
        mixture = torch.from_numpy(audio.T.copy())
        reference = mixture.mean(0)
        mean, std = reference.mean(), reference.std()
        if float(std) < 1e-8:
            vocals = np.zeros_like(audio)
            drums = np.zeros_like(audio)
        else:
            with torch.inference_mode():
                estimates = apply_model(model, ((mixture - mean) / std)[None],
                    shifts=0, split=True, overlap=.25, device=device, segment=7.8, num_workers=0)
                estimates = estimates[0].cpu() * std + mean
                vocals = estimates[model.sources.index("vocals")].T.numpy()
                drums = estimates[model.sources.index("drums")].T.numpy()
        if vocals.shape != audio.shape or not np.isfinite(vocals).all():
            raise ValueError("Vocal model changed length/channels or returned invalid audio")
        sf.write(job["output"], vocals, rate, subtype="FLOAT")
        if job.get('drums_output'):
            if drums.shape!=audio.shape or not np.isfinite(drums).all():raise ValueError('Invalid estimated drum PCM')
            sf.write(job['drums_output'],drums,rate,subtype='FLOAT')
        profile = []
        for start in range(0, len(audio), rate):
            end = min(len(audio), start + rate)
            full = float(np.mean(audio[start:end].astype(np.float64) ** 2))
            voice = float(np.mean(vocals[start:end].astype(np.float64) ** 2))
            ratio = min(1., voice / max(full, 1e-12)) if full > 1e-8 else 0.
            profile.append({"time": start / rate, "energy_ratio": ratio,
                            "foreground_risk": min(1., ratio / .15)})
        weights = list((Path(torch.hub.get_dir()) / "checkpoints").glob("955717e8-*.th"))
        return {"path": job["output"], "sample_rate": rate, "frames": len(audio), "profile": profile,
                **({'drums_path':job['drums_output'],'drum_source_start_seconds':0.} if job.get('drums_output') else {}),
                "provenance": {"executed": True, "model": "htdemucs", "demucs_version": "4.0.1",
                "device": device, "shifts": 0, "segment_seconds": 7.8, "overlap": .25,
                "weights": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in weights},
                "stem_normalization": "joint mixture standardization, no per-stem peak rescaling",
                "limitations": "Estimated voice contains leakage/artifacts; energy ratio is not a calibrated vocal probability"}}
    finally:
        decoded.unlink(missing_ok=True)


def handoff(outgoing, incoming, vocal_out, vocal_in):
    """Avoid simultaneous foreground estimates with a complementary voice exit/entry."""
    from .mixing import smooth
    ratios = []
    for audio, vocal in ((outgoing, vocal_out), (incoming, vocal_in)):
        if vocal is None:
            return outgoing, incoming, {"executed": False, "reason": "both aligned vocal estimates required"}
        if vocal.shape != audio.shape or not np.isfinite(vocal).all():
            raise ValueError("Vocal estimate must share the exact rendered overlap")
        ratios.append(float(np.mean(vocal.astype(np.float64) ** 2) / max(float(np.mean(audio.astype(np.float64) ** 2)), 1e-12)))
    if min(ratios) < .02:
        return outgoing, incoming, {"executed": False, "reason": "one overlap lacks a prominent estimated voice", "estimated_vocal_energy_ratios": ratios}
    p = np.linspace(0, 1, len(outgoing))[:, None]
    old_keep = 1 - smooth((p - .20) / .25)
    new_keep = smooth((p - .55) / .25)
    return (outgoing - vocal_out * (1 - old_keep)).astype(np.float32), (incoming - vocal_in * (1 - new_keep)).astype(np.float32), {
        "executed": True, "estimated_vocal_energy_ratios": ratios, "outgoing_voice_exit_fraction": [.20, .45],
        "incoming_voice_entry_fraction": [.55, .80], "method": "subtract suppressed part of estimated vocal stems, original accompaniment retained",
        "vocal_overlap_certified_absent": False}


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(_worker(job), ensure_ascii=False), encoding="utf-8")
