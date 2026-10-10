"""Minimal single-file enhancement / multi-file set entry point."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

from .cues import runtime_config
from .storage import MEDIA, private_path, write_json


def collect(inputs):
    files = []
    for value in inputs:
        source = Path(value).expanduser().resolve(strict=True)
        if source.is_dir():
            files.extend(sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA), key=lambda p: str(p).casefold()))
        elif source.suffix.lower() in MEDIA:
            files.append(source)
        else:
            raise ValueError("Input must be an audio file or music folder")
    if not files or len(set(files)) != len(files):
        raise ValueError("Provide nonempty, unique audio inputs")
    return files


def make(args):
    from .adapters import master_set, prepare_tracks
    from .quality import export_audio, measure_audio
    files = collect(args.inputs)
    runtime = runtime_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    output = private_path(args.output or runtime.get("output_root") or files[0].parent / "AI_Super_DJ")
    work = private_path(args.work or Path(runtime.get("cache", Path.home() / ".cache" / "superdj")) / "runs" / stamp)
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    name = args.name or (files[0].stem + "_AI_Master" if len(files) == 1 else "AI_Super_DJ_" + stamp)
    if not name or Path(name).name != name or any(c in name for c in '/\\:'):
        raise ValueError("Use a simple output filename stem")
    if (output / f"{name}.flac").exists():
        raise FileExistsError("Choose a new name: an existing master is never overwritten")
    for key in ("audamcp", "bitwize"):
        if not runtime.get(key):
            raise ValueError("Run setup and configure private runtime.local.json with audited processor paths")
    ffmpeg, ffprobe = runtime.get("ffmpeg", "ffmpeg"), runtime.get("ffprobe", "ffprobe")
    git = runtime.get("git", "git")
    records = [{"id": f"track_{i:02}", "path": str(p)} for i, p in enumerate(files, 1)]
    write_json(work / "playlist.json", {"tracks": records, "selection": "all supplied files, no random omission"})
    if len(files) == 1:
        before = measure_audio(files[0], ffmpeg=ffmpeg, ffprobe=ffprobe, log=work / "source_qa.log")
        prepared = prepare_tracks(records, work, audamcp=runtime["audamcp"], ffmpeg=ffmpeg, git=git,
                                  sample_rate=before["sample_rate"])
        mastered = master_set(prepared[records[0]["id"]]["path"], work / "master.wav", work,
            bitwize=runtime["bitwize"], git=git, target_lufs=-14, ceiling_db=-1.5, limiter_lookahead_ms=0)
        import soundfile as sf
        if sf.info(work / "master.wav").frames != before["decoded_samples"]:
            raise ValueError("Single-file processing changed source timing")
        exports = export_audio(work / "master.wav", output, name, ffmpeg=ffmpeg, ffprobe=ffprobe, mp3=False)
        if exports["flac"]["decoded_samples"] != before["decoded_samples"]:
            raise ValueError("Encoded master changed source timing")
        report = {"source": before, "preparation": prepared, "mastering": mastered, "exports": exports,
                  "scope": "single-file processing, original timing preserved; defect-aware repair, final master, verified lossless encoding",
                  "listening_review": "final master awaiting human audition"}
        write_json(output / f"{name}_manifest.json", report)
        print(f"Completed: {exports['flac']['path']}", flush=True)
        return report
    # Use the same supported stages, including their source, timing and export
    # checks; the simple surface doesn't carry a second renderer implementation.
    from .cli import main
    rates = []
    for source in files:
        probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                                "stream=sample_rate", "-of", "json", str(source)], check=True, capture_output=True)
        rates.append(int(json.loads(probe.stdout)["streams"][0]["sample_rate"]))
    # One common processing clock: preserve the highest supplied rate, never
    # present resampling of other sources as recovered fidelity.
    sample_rate = max(rates)
    argv = ["run", "--playlist", str(work / "playlist.json"), "--work", str(work),
            "--output", str(output), "--name", name, "--audamcp", runtime["audamcp"],
            "--bitwize", runtime["bitwize"], "--rubberband", runtime.get("rubberband", "rubberband"),
            "--git", git, "--ffmpeg", ffmpeg, "--ffprobe", ffprobe,
            "--style", args.style, "--mode", "fixed" if args.keep_order else "auto", "--no-mp3",
            "--sample-rate", str(sample_rate)]
    if runtime.get("cache"):
        argv += ["--cache", runtime["cache"]]
    if runtime.get("beat_checkpoint"):
        argv += ["--checkpoint", runtime["beat_checkpoint"]]
    if runtime.get("beat_device"):
        argv += ["--device", runtime["beat_device"]]
    main(argv)
