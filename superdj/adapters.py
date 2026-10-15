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
        if job.get("forced_processors") is not None:
            for processor in job["forced_processors"]:
                from audamcp.master import apply_highpass, apply_eq
                if processor["function"] == "audamcp.master.apply_highpass":
                    samples, _ = apply_highpass(samples, rate, cutoff_hz=processor["settings"]["highpass_hz"])
                elif processor["function"] == "audamcp.master.apply_eq":
                    samples, _ = apply_eq(samples, rate, **processor["parameters"])
                else:
                    raise ValueError("Only the mixture's linear preparation can be applied to a vocal estimate")
            report["processors"] = job["forced_processors"]
            save(samples.astype(np.float32), rate, job["output"])
            return {"path": job["output"], "provenance": {**job["checkout"], "executed": True, **report}}
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
        from .tone import settings as tone_settings
        eq, evidence = tone_settings(samples, rate)
        report["tonal_analysis"] = evidence
        if eq:
            from audamcp.master import apply_eq
            samples, settings = apply_eq(samples, rate, **eq)
            report["processors"].append({"function": "audamcp.master.apply_eq", "parameters": eq,
                "reason": "bounded content-dependent tonal balancing", "settings": settings})
        save(samples.astype(np.float32), rate, job["output"])
        if sf.info(job["output"]).subtype != "FLOAT":
            raise RuntimeError("Upstream bridge did not retain FLOAT PCM")
        return {"path": job["output"], "provenance": {**job["checkout"], "executed": True, "bridge": "audamcp.audio_io.load/save", **report}}
    if job["action"] == "master":
        from tools.mastering.master_tracks import master_track
        preset = {"target_lufs": job["target_lufs"], "compress_ratio": 1.0, "output_bits": 24,
                  "dither_bits": 24, "processing_oversample": 1, "limiter_lookahead_ms": job.get("limiter_lookahead_ms", 5),
                  "limiter_release_ms": 100, "track_gap": 0}
        # Long sets can exceed RIFF's four-gigabyte size field. Keep the same
        # PCM mastering path, selecting its large-file container at the write.
        original_write = sf.write
        output_container = []
        def write_large_wav(file, data, samplerate, *args, **kwargs):
            channels = data.shape[1] if data.ndim > 1 else 1
            subtype = kwargs.get("subtype", "PCM_16")
            bytes_per_sample = {"PCM_24": 3, "PCM_32": 4, "FLOAT": 4, "DOUBLE": 8}.get(subtype, 2)
            if Path(file).suffix.lower() == ".wav" and len(data) * channels * bytes_per_sample + 4096 >= 2**32:
                kwargs["format"] = "RF64"
                output_container.append("RF64")
            return original_write(file, data, samplerate, *args, **kwargs)
        sf.write = write_large_wav
        try:
            result = master_track(job["input"], job["output"], ceiling_db=job["ceiling_db"], preset=preset)
        finally:
            sf.write = original_write
        if result.get("skipped") or not Path(job["output"]).is_file():
            raise RuntimeError("Mastering skipped the input")
        return {"path": job["output"], "provenance": {**job["checkout"], "executed": True,
                "function": "tools.mastering.master_tracks.master_track", "preset": preset,
                "ceiling_db": job["ceiling_db"], "large_file_container": output_container or None,
                "upstream_measurements_before_encoded_QA": result}}
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


def prepare_tracks(records, work_dir, *, audamcp, ffmpeg="ffmpeg", git="git", noise_profiles=None, progress=print, sample_rate=44100):
    from .execution import current, cached
    session = current()
    if session is None:
        return _prepare_tracks_uncached(records, work_dir, audamcp=audamcp, ffmpeg=ffmpeg, git=git,
            noise_profiles=noise_profiles, progress=progress, sample_rate=sample_rate)
    checkout = checkout_info(audamcp, 'audamcp', git)
    def one(record):
        from .audio import ffmpeg_executable
        track_id = record['id']
        dependencies = {'source': record['path'], 'stems': record.get('vocal_stem'), 'checkout': checkout,
            'rate': sample_rate, 'decoder': ffmpeg_executable(ffmpeg),
            'noise': (noise_profiles or {}).get(track_id)}
        def produce():
            return _prepare_tracks_uncached([record], work_dir, audamcp=audamcp, ffmpeg=ffmpeg, git=git,
                noise_profiles=noise_profiles, progress=progress, sample_rate=sample_rate)[track_id]
        result = cached('preparation', dependencies, produce, Path(work_dir) / 'prepared')
        result['provenance']['original'] = record['path']
        return track_id, result
    return dict(session.map(one, records))


def _prepare_tracks_uncached(records: list[dict], work_dir: str | Path, *, audamcp: str | Path, ffmpeg="ffmpeg", git="git", noise_profiles=None, progress=print,
                   sample_rate=44100) -> dict:
    directory = private_path(work_dir) / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    checkout = checkout_info(audamcp, "audamcp", git)
    prepared = {}
    for index, record in enumerate(records, 1):
        track_id = record["id"]
        decoded, output = directory / f"{track_id}_decoded.wav", directory / f"{track_id}.wav"
        progress(f"Preparing {index}/{len(records)}: {Path(record['path']).name}")
        from .audio import decode_wave_file
        decode_wave_file(record['path'], decoded, sample_rate=sample_rate, ffmpeg=ffmpeg)
        result = run_worker({"action": "prepare", "checkout": checkout, "decoded": str(decoded), "output": str(output),
                            "noise_profile": (noise_profiles or {}).get(track_id)}, directory, track_id)
        result["provenance"]["original"] = record["path"]
        if record.get("vocal_stem") and not (noise_profiles or {}).get(track_id):
            vocal_decoded, vocal_output = directory / f"{track_id}_voice_decoded.wav", directory / f"{track_id}_voice.wav"
            execute([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", record["vocal_stem"]["path"],
                     "-ac", "2", "-ar", str(sample_rate), "-c:a", "pcm_f32le", str(vocal_decoded)])
            voice = run_worker({"action": "prepare", "checkout": checkout, "decoded": str(vocal_decoded),
                "output": str(vocal_output), "forced_processors": result["provenance"]["processors"]}, directory, track_id + "_voice")
            result["vocal_path"] = voice["path"]
            result["vocal_preparation"] = voice["provenance"]
            vocal_decoded.unlink()
            if record['vocal_stem'].get('drums_path'):
                drum_decoded=directory/f'{track_id}_drums_decoded.wav'
                drum_output=directory/f'{track_id}_drums.wav'
                execute([ffmpeg,'-nostdin','-v','error','-y','-i',record['vocal_stem']['drums_path'],
                         '-ac','2','-ar',str(sample_rate),'-c:a','pcm_f32le',str(drum_decoded)])
                drum=run_worker({'action':'prepare','checkout':checkout,'decoded':str(drum_decoded),
                    'output':str(drum_output),'forced_processors':result['provenance']['processors']},directory,track_id+'_drums')
                result['drum_path']=drum['path'];result['drum_preparation']=drum['provenance']
                drum_decoded.unlink()
        prepared[track_id] = result
        decoded.unlink()
    return prepared


def master_set(input_path, output_path, work_dir, *, bitwize, git="git", target_lufs=-14.0, ceiling_db=-1.5,
               limiter_lookahead_ms=5.0):
    return run_worker({"action": "master", "checkout": checkout_info(bitwize, "bitwize", git),
                       "input": str(input_path), "output": str(output_path), "target_lufs": target_lufs,
                       "ceiling_db": ceiling_db, "limiter_lookahead_ms": limiter_lookahead_ms}, work_dir, "master")


if __name__ == "__main__":
    write_json(sys.argv[2], _worker(read_json(sys.argv[1])))
