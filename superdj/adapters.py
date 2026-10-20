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
    # Preparation workers are scoped to one audited checkout. Mastering retains
    # a fresh interpreter to isolate upstream `tools` imports and stochastic dither.
    import numpy as np
    import soundfile as sf

    if job['checkout']['path'] not in sys.path:
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
        from tools.mastering import master_tracks as mastering_module
        if job.get('memory',{}).get('mode')=='bounded':
            from .stream_master import master
            result,resources=master(job['input'],job['output'],mastering_module,
                target_lufs=job['target_lufs'],ceiling_db=job['ceiling_db'],
                limiter_lookahead_ms=job.get('limiter_lookahead_ms',5.),
                block=job['memory']['block_frames'],scratch=job.get('scratch'),
                # Sequential file windows avoid I/O/thread overhead measured
                # in the bounded path; the in-memory adapter keeps its pool.
                lra_jobs=1)
            return {'path':job['output'],'provenance':{**job['checkout'],'executed':True,
                'function':'superdj.stream_master.master; audited ordinary upstream preset',
                'ceiling_db':job['ceiling_db'],'memory':job['memory'],'bounded_master':resources,
                'upstream_measurements_before_encoded_QA':result}}
        master_track = mastering_module.master_track
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
            from .mastering import accelerated_limiter
            from .lra_parallel import parallel_lra
            with accelerated_limiter(mastering_module, job.get('accelerate', True)
                    and job.get('limiter_lookahead_ms',5)>0) as acceleration, \
                    parallel_lra(mastering_module, enabled=job.get('parallel_lra', True)) as lra:
                result = master_track(job["input"], job["output"], ceiling_db=job["ceiling_db"], preset=preset)
        finally:
            sf.write = original_write
        if result.get("skipped") or not Path(job["output"]).is_file():
            raise RuntimeError("Mastering skipped the input")
        return {"path": job["output"], "provenance": {**job["checkout"], "executed": True,
                "function": "tools.mastering.master_tracks.master_track", "preset": preset,
                "ceiling_db": job["ceiling_db"], "large_file_container": output_container or None,
                "limiter_acceleration": acceleration,
                "lra_parallelism": lra,
                "upstream_measurements_before_encoded_QA": result}}
    raise ValueError("Unknown adapter action")


def run_worker(job: dict, directory: str | Path, tag: str) -> dict:
    work = private_path(directory)
    work.mkdir(parents=True, exist_ok=True)
    job_file, result_file = work / f"{tag}_job.json", work / f"{tag}_result.json"
    write_json(job_file, job)
    from .execution import current
    if job.get('action') == 'prepare' and current():
        result = current().audio(job)
        write_json(result_file, result)
        return result
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    if job.get('action')=='master':
        threads=str(1 if job.get('memory',{}).get('constrained') else min(4,job.get('cpu_jobs',1)))
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
            env[name]=threads
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
    return dict(session.cpu_map(one, records))


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
    from .execution import current
    from .resources import admit_master,_disk,GIB
    import soundfile as sf
    session=current()
    # Models and preparation interpreters are idle after the completed render.
    # Release their actual allocations before choosing the mastering strategy.
    if session: session.release_workers()
    info=sf.info(input_path)
    memory=admit_master(info.frames,info.samplerate,info.channels)
    scratch=session.work if session else Path(work_dir)
    _,scratch_volume,free=_disk(scratch)
    _,output_volume,output_free=_disk(Path(output_path).parent)
    required=memory['temporary_pcm_bytes_estimate']
    output_bytes=info.frames*info.channels*3+GIB
    if scratch_volume==output_volume:
        required+=output_bytes
    elif output_free<output_bytes:
        raise OSError('Insufficient output disk space for the complete master')
    if free<required:
        raise OSError('Insufficient temporary disk space for the bounded complete master')
    if session: session.metrics['master_admission']=memory
    return run_worker({"action": "master", "checkout": checkout_info(bitwize, "bitwize", git),
                       "input": str(input_path), "output": str(output_path), "target_lufs": target_lufs,
                       "ceiling_db": ceiling_db, "limiter_lookahead_ms": limiter_lookahead_ms,
                       'memory':memory,'scratch':str(scratch),
                       'cpu_jobs':session.cpu_jobs if session else 1}, work_dir, "master")


def measure_prepared(records, prepared, work_dir, *, ffmpeg='ffmpeg', ffprobe='ffprobe'):
    """Identical full-recording checks, bounded across independent tracks."""
    from .execution import current, cached
    from .quality import measure_audio
    from .audio import ffmpeg_executable
    import shutil
    session = current()
    def one(record):
        item = dict(prepared[record['id']])
        if item.get('loudness_lufs') is None:
            measured = cached('prepared_measurements', {'audio': item['path'],
                'ffmpeg': ffmpeg_executable(ffmpeg), 'ffprobe': shutil.which(str(ffprobe)) or str(ffprobe)},
                lambda: measure_audio(item['path'], ffmpeg=ffmpeg, ffprobe=ffprobe,
                    log=Path(work_dir)/f"{record['id']}_prepared_qa.log"), Path(work_dir)/'measurement_artifacts')
            item.update(source_measurements=measured, loudness_lufs=measured['integrated_lufs'])
        return record['id'], item
    values = session.cpu_map(one, records) if session else list(map(one, records))
    for identity, item in values:
        prepared[identity] = item
    for record in records:
        record['loudness_lufs'] = prepared[record['id']]['loudness_lufs']
    return prepared


if __name__ == "__main__":
    write_json(sys.argv[2], _worker(read_json(sys.argv[1])))
