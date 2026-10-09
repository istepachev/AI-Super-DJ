"""Pinned, separately installed upstream processors; no vendored source."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

from .quality import execute
from .storage import private_path, read_json, write_json

COMPONENTS = {
    "audamcp": {"url": "https://github.com/Dream-Pixels-Forge/audacity-mcp.git", "revision": "9d812922e46f504f91cca48338ed68511f8d4243", "license": "MIT"},
    "bitwize": {"url": "https://github.com/bitwize-music-studio/claude-ai-music-skills.git", "revision": "58f32c25b9fb2dfe569478459b0a591e0dff22e8", "license": "CC0-1.0"},
}


def checkout_info(path: str | Path, component: str, git="git") -> dict:
    directory = Path(path).resolve(strict=True)
    revision = subprocess.run([git, "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, check=True, text=True).stdout.strip()
    if revision != COMPONENTS[component]["revision"]:
        raise ValueError(f"{component} revision differs from the audited adapter target")
    dirty = subprocess.run([git, "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], capture_output=True, check=True, text=True).stdout
    if dirty.strip():
        raise ValueError(f"{component} has modified tracked source; audit it before use")
    return {"component": component, "path": str(directory), "revision": revision, "license": COMPONENTS[component]["license"]}


def setup_components(directory: str | Path, git="git") -> dict:
    root = private_path(directory)
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, info in COMPONENTS.items():
        target = root / name
        if not target.exists():
            subprocess.run([git, "clone", "--no-checkout", info["url"], str(target)], check=True)
            subprocess.run([git, "-C", str(target), "checkout", "--detach", info["revision"]], check=True)
        result[name] = checkout_info(target, name, git)
    return result


def _worker(job: dict) -> dict:
    # This function executes in a fresh process to isolate upstream `tools` imports.
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, job["checkout"]["path"])
    if job["action"] == "prepare":
        from audamcp.audio_io import load, save
        samples, rate = load(job["decoded"])
        dc = np.mean(samples, axis=1, dtype=np.float64)
        maximum = float(np.max(np.abs(samples)))
        report = {"dc_per_channel": dc.tolist(), "sample_peak": maximum,
                  "restoration": "bypassed: no identified noise/click profile", "processors": []}
        if float(np.max(np.abs(dc))) > 0.001:
            from audamcp.master import apply_highpass
            samples, settings = apply_highpass(samples, rate, cutoff_hz=20)
            report["processors"].append({"function": "audamcp.master.apply_highpass", "reason": "measured DC offset above 0.001", "settings": settings})
        if job.get("noise_profile"):
            from audamcp.denoise import spectral_reduce
            profile, profile_rate = load(job["noise_profile"])
            if profile_rate != rate or profile.shape[0] != samples.shape[0]:
                raise ValueError("Noise-only profile must match prepared sample rate and channels")
            samples, settings = spectral_reduce(samples, rate, noise_profile=profile, strength=0.30)
            report["restoration"] = "explicit noise-only profile, conservative spectral reduction"
            report["processors"].append({"function": "audamcp.denoise.spectral_reduce", "settings": settings})
        save(samples.astype(np.float32), rate, job["output"])
        if sf.info(job["output"]).subtype != "FLOAT":
            raise RuntimeError("Upstream bridge did not retain FLOAT PCM")
        return {"path": job["output"], "provenance": {**job["checkout"], "executed": True, "bridge": "audamcp.audio_io.load/save", **report}}
    if job["action"] == "master":
        from tools.mastering.master_tracks import master_track
        preset = {"target_lufs": job["target_lufs"], "compress_ratio": 1.0, "output_bits": 24,
                  "dither_bits": 24, "processing_oversample": 1, "limiter_lookahead_ms": 5,
                  "limiter_release_ms": 100, "track_gap": 0}
        result = master_track(job["input"], job["output"], ceiling_db=job["ceiling_db"], preset=preset)
        if result.get("skipped") or not Path(job["output"]).is_file():
            raise RuntimeError("Mastering skipped the input")
        return {"path": job["output"], "provenance": {**job["checkout"], "executed": True,
                "function": "tools.mastering.master_tracks.master_track", "preset": preset,
                "ceiling_db": job["ceiling_db"], "upstream_measurements_before_encoded_QA": result}}
    raise ValueError("Unknown adapter action")


def run_worker(job: dict, directory: str | Path, tag: str) -> dict:
    work = private_path(directory)
    work.mkdir(parents=True, exist_ok=True)
    job_file, result_file = work / f"{tag}_job.json", work / f"{tag}_result.json"
    write_json(job_file, job)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-m", "superdj.adapters", str(job_file), str(result_file)], env=env, capture_output=True)
    (work / f"{tag}.log").write_bytes(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-3000:])
    return read_json(result_file)


def prepare_tracks(records: list[dict], work_dir: str | Path, *, audamcp: str | Path, ffmpeg="ffmpeg", git="git", noise_profiles=None, progress=print) -> dict:
    directory = private_path(work_dir) / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    checkout = checkout_info(audamcp, "audamcp", git)
    prepared = {}
    for index, record in enumerate(records, 1):
        track_id = record["id"]
        decoded, output = directory / f"{track_id}_decoded.wav", directory / f"{track_id}.wav"
        progress(f"Preparing {index}/{len(records)}: {Path(record['path']).name}")
        execute([ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", record["path"], "-map", "0:a:0", "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", "-rf64", "auto", str(decoded)])
        result = run_worker({"action": "prepare", "checkout": checkout, "decoded": str(decoded), "output": str(output),
                            "noise_profile": (noise_profiles or {}).get(track_id)}, directory, track_id)
        result["provenance"]["original"] = record["path"]
        prepared[track_id] = result
        decoded.unlink()
    return prepared


def master_set(input_path, output_path, work_dir, *, bitwize, git="git", target_lufs=-14.0, ceiling_db=-1.5):
    return run_worker({"action": "master", "checkout": checkout_info(bitwize, "bitwize", git),
                       "input": str(input_path), "output": str(output_path), "target_lufs": target_lufs,
                       "ceiling_db": ceiling_db}, work_dir, "master")


if __name__ == "__main__":
    write_json(sys.argv[2], _worker(read_json(sys.argv[1])))
