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


def _make(args):
    from .adapters import master_set, prepare_tracks
    from .quality import export_audio, measure_audio
    files = collect(args.inputs)
    runtime = runtime_config()
    folder_input=len(args.inputs)==1 and Path(args.inputs[0]).expanduser().is_dir()
    exclusions=[]
    if len(files)>1 and folder_input and not getattr(args,'allow_repeats',False):
        from .history import unseen_files
        files,exclusions=unseen_files(files)
        if not files:raise ValueError('No unused tracks remain in the folder; provide another collection or explicitly allow repeats')
    from .defaults import planning_defaults,playback_defaults
    preferences = planning_defaults(style=getattr(args, 'style', None),
        harmonic_mode=getattr(args, 'harmonic_mode', None),
        energy_arc=getattr(args, 'energy_arc', None), runtime=runtime)
    playback_defaults(long_play=getattr(args,'long_play',False),target_play_seconds=getattr(args,'track_play_seconds',None))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    output = private_path(args.output or runtime.get('private_output_root') or runtime.get("output_root") or files[0].parent / "AI_Super_DJ")
    from .execution import current
    work = private_path(args.work or current().work)
    args._performance_output = output
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    from .naming import dated_name
    title=args.name or (files[0].stem+'_AI_Master' if len(files)==1 else files[0].parent.name)
    name=dated_name(title)
    if not name or Path(name).name != name or any(c in name for c in '/\\:'):
        raise ValueError("Use a simple output filename stem")
    if (output / f"{name}.flac").exists():
        raise FileExistsError("Choose a new name: an existing master is never overwritten")
    for key in ("audamcp", "bitwize"):
        if not runtime.get(key):
            raise ValueError("Run setup and configure private runtime.local.json with audited processor paths")
    ffmpeg, ffprobe = runtime.get("ffmpeg", "ffmpeg"), runtime.get("ffprobe", "ffprobe")
    from .resources import preflight
    resources = preflight(files, output=output, scratch=work,
        cache=current().root if current().persistent else None,
        jobs=getattr(args,'jobs',None) or runtime.get('jobs'),devices=runtime.get('gpu_devices',()),ffprobe=ffprobe,sample_rate=1,
        cpu_jobs=getattr(args,'cpu_jobs',None) or (None if getattr(args,'jobs',None) is not None else runtime.get('cpu_jobs')),
        target_play_seconds=playback_defaults(long_play=getattr(args,'long_play',False),
            target_play_seconds=getattr(args,'track_play_seconds',None))['target_play_range_seconds'][1])
    current().jobs = resources['jobs']
    current().cpu_jobs = resources['cpu_jobs']
    current().preparation_overlap = resources['preparation_overlap']
    current().devices = tuple(resources['devices'])
    current().resource_policy=resources.get('memory_policy',current().resource_policy)
    current().metrics['resources'] = resources
    short_exclusions=[]
    if folder_input:
        files,short_exclusions=_folder_short_sources(files,resources['source_probes'])
        write_json(work/'short_source_exclusions.json',short_exclusions)
        if not files:raise ValueError('No source reaches the one-minute minimum in this automatic folder selection')
        resources['source_probes']=[p for p in resources['source_probes'] if Path(p['path']) in set(files)]
    git = runtime.get("git", "git")
    records = [{"id": f"track_{i:02}", "path": str(p)} for i, p in enumerate(files, 1)]
    write_json(work / "playlist.json", {"tracks": records, "selection": "automatic folder history filtering" if folder_input else "all explicit supplied files retained",
        'excluded_previous_sources':exclusions,'excluded_short_sources':short_exclusions, 'workflow_preferences':preferences})
    if len(files) == 1:
        from .resources import admit_master
        probe = resources['source_probes'][0]
        admit_master(round(probe['duration_seconds']*probe['sample_rate']),probe['sample_rate'])
        before = measure_audio(files[0], ffmpeg=ffmpeg, ffprobe=ffprobe, log=work / "source_qa.log")
        prepared = prepare_tracks(records, work, audamcp=runtime["audamcp"], ffmpeg=ffmpeg, git=git,
                                  sample_rate=before["sample_rate"])
        from .resources import admit_master
        current().metrics['master_admission'] = admit_master(before['decoded_samples'],before['sample_rate'])
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
    rates = [row['sample_rate'] for row in resources['source_probes']]
    # One common processing clock: preserve the highest supplied rate, never
    # present resampling of other sources as recovered fidelity.
    sample_rate = max(rates)
    argv = ["run", "--playlist", str(work / "playlist.json"), "--work", str(work),
            "--output", str(output), "--name", name, "--audamcp", runtime["audamcp"],
            "--bitwize", runtime["bitwize"], "--rubberband", runtime.get("rubberband", "rubberband"),
            "--git", git, "--ffmpeg", ffmpeg, "--ffprobe", ffprobe,
            "--style", preferences['style'], "--harmonic-mode", preferences['harmonic_mode'],
            "--energy-arc", preferences['energy_arc'],
            "--mode", "fixed" if args.keep_order else "auto", "--no-mp3",
            "--sample-rate", str(sample_rate)]
    if getattr(args,'long_play',False):argv.append('--long-play')
    if getattr(args,'track_play_seconds',None) is not None:argv += ['--track-play-seconds',str(args.track_play_seconds)]
    argv += ["--cache", str(current().root)]
    if runtime.get("beat_checkpoint"):
        argv += ["--checkpoint", runtime["beat_checkpoint"]]
    if runtime.get("beat_device"):
        argv += ["--device", runtime["beat_device"]]
    main(argv)


def _folder_short_sources(files,probes):
    """Automatic selection may omit sub-minute sources, always with a ledger.

    Explicitly supplied files never enter this helper. Their selected section
    can instead receive a supported loop; source identity is not discarded.
    """
    by_path={Path(p['path']):p for p in probes};kept=[];excluded=[]
    for path in files:
        duration=float(by_path[Path(path)]['duration_seconds'])
        if duration<60:
            excluded.append({'path':str(path),'duration_seconds':duration,'reason':'automatic folder selection: native source below one minute'})
        else:kept.append(path)
    return kept,excluded


def make(args):
    from .execution import execution
    from .storage import write_json
    runtime = runtime_config()
    explicit_cache = getattr(args, 'cache', None)
    cache = None if getattr(args, 'no_cache', False) else (explicit_cache or
        (runtime.get('cache_root') if runtime.get('cache_mode') == 'persistent' else None))
    jobs = getattr(args,'jobs',None) or runtime.get('jobs',1)
    session = None
    try:
        with execution(cache, jobs=jobs, devices=runtime.get('gpu_devices',()),
                       scratch_root=runtime.get('scratch_root')) as session:
            return _make(args)
    finally:
        report_root = getattr(args, '_performance_output', None) or args.output or runtime.get('private_output_root') or runtime.get('output_root')
        if report_root and session is not None:
            write_json(Path(report_root) / 'performance.json', session.metrics)
