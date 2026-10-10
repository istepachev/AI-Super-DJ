"""Local command-line workflow, with private inputs and explicit backends."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import random

from .storage import private_path, read_json, select_tracks, write_json


def _records(path):
    value = read_json(path)
    return value.get("records", value.get("tracks")) if isinstance(value, dict) else value


def _playlist(args):
    if args.playlist:
        return read_json(args.playlist)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    playlist = select_tracks(args.folder, args.count, seed)
    write_json(Path(args.work) / "playlist.json", playlist)
    return playlist


def analyze(args):
    from .analysis import analyze_track
    playlist = _playlist(args)
    overrides = read_json(args.grid_json) if args.grid_json else {}
    records = []
    for index, track in enumerate(playlist["tracks"], 1):
        print(f"Analyzing {index}/{len(playlist['tracks'])}: {Path(track['path']).name}", flush=True)
        records.append(analyze_track(track["path"], track_id=track["id"], ffmpeg=args.ffmpeg,
            backend=args.backend, cache_dir=args.cache, checkpoint=args.checkpoint, device=args.device,
            grid=overrides.get(track["id"]), allow_model_download=args.download_model))
        write_json(Path(args.work) / "records.partial.json", records)
    write_json(Path(args.work) / "records.json", records)
    return records


def plan(args, records=None):
    from .planner import plan_tracks
    records = records if records is not None else _records(args.records)
    order = read_json(args.fixed_order) if args.fixed_order else None
    result = plan_tracks(records, mode=args.mode, fixed_order=order, first_id=args.first, last_id=args.last,
                         seed=args.seed or 0, energy_arc=args.energy_arc, max_tempo_change=args.max_tempo_change,
                         allow_tempo_reset=not args.no_tempo_reset)
    write_json(Path(args.work) / "plan.json", result)
    print(f"Planned {len(result['order'])} tracks; {result['tempo_reset_count']} tempo conflicts requiring review; {result['method']}", flush=True)
    return result


def render(args, records=None, planned=None):
    from .adapters import master_set, prepare_tracks
    from .quality import export_audio, measure_audio
    from .render import render_plan, validate_transition_policy
    work = private_path(args.work)
    directory = private_path(args.output)
    work.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    if not args.name or Path(args.name).name != args.name or any(c in args.name for c in '/\\:'):
        raise ValueError("Use a simple output filename stem")
    if not args.overwrite and any((directory / f"{args.name}.{ext}").exists() for ext in ("flac", "mp3")):
        raise FileExistsError("Output already exists; use a new name or --overwrite")
    records = records if records is not None else _records(args.records)
    planned = planned if planned is not None else read_json(args.plan)
    validate_transition_policy(planned, allow_hard_cut=args.allow_hard_cut,
                               allow_unverified=args.allow_unverified)
    # Preserve previous intermediates when a user deliberately repeats a run.
    if (work / "premix.wav").exists():
        work = work / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
        work.mkdir(parents=True, exist_ok=False)
    if args.prepared:
        prepared = read_json(args.prepared)
    else:
        profiles = read_json(args.noise_profiles) if args.noise_profiles else None
        prepared = prepare_tracks(records, work, audamcp=args.audamcp, ffmpeg=args.ffmpeg,
                                  git=args.git, noise_profiles=profiles, progress=lambda s: print(s, flush=True))
    for record in records:
        item = prepared[record["id"]]
        if item.get("loudness_lufs") is None:
            measured = measure_audio(item["path"], ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                                     log=work / f"{record['id']}_prepared_qa.log")
            item["source_measurements"] = measured
            item["loudness_lufs"] = measured["integrated_lufs"]
        record["loudness_lufs"] = item["loudness_lufs"]
    write_json(work / "prepared.json", prepared)
    premix = work / "premix.wav"
    print("Rendering tempo maps and transitions", flush=True)
    rendered = render_plan(planned, records, prepared, premix, work / "render",
                           ffmpeg=args.ffmpeg, rubberband=args.rubberband,
                           max_tempo_change=planned.get("constraints", {}).get("max_tempo_change_per_deck", 0.06),
                           max_intro_trim_seconds=args.max_intro_trim_seconds,
                           transition_bars=args.transition_bars, allow_unverified=args.allow_unverified,
                           allow_hard_cut=args.allow_hard_cut,
                           progress=lambda s: print(s, flush=True))
    write_json(work / "render.json", rendered)
    if len(rendered["tracks"]) != len(records) or len({t["id"] for t in rendered["tracks"]}) != len(records):
        raise RuntimeError("Render did not preserve every track exactly once")
    master_path = work / "master.wav"
    print("Mastering the complete set", flush=True)
    mastered = master_set(premix, master_path, work, bitwize=args.bitwize, git=args.git,
                          target_lufs=args.target_lufs, ceiling_db=-1.5)
    print("Exporting and independently checking full FLAC/MP3 files", flush=True)
    exported = export_audio(master_path, directory, args.name, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                            mp3=not args.no_mp3, overwrite=args.overwrite)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "selection_and_analysis": records, "plan": planned, "preparation": prepared,
                "render": rendered, "mastering": mastered, "exports": exported,
                "listening_review": "pending human audition; measurements do not certify musical quality"}
    write_json(directory / f"{args.name}_manifest.json", manifest)
    names = {r["id"]: Path(r["path"]).stem for r in records}
    lines = ["AI Super DJ — timed playlist", ""]
    for index, track in enumerate(rendered["tracks"], 1):
        seconds = float(track.get("start_seconds", track.get("start_s", 0)))
        hours, minutes, seconds = int(seconds // 3600), int(seconds % 3600 // 60), int(seconds % 60)
        lines.append(f"{index:02}. {hours:02}:{minutes:02}:{seconds:02}  {names[track['id']]}")
    (directory / f"{args.name}_tracklist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Completed: {exported['flac']['path']}", flush=True)
    return manifest


def _audio(parser):
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")


def djstudio_audit(args):
    """Inspect native metadata and exported chapters without changing the mix."""
    import json
    import subprocess
    from .djstudio import audit_djstudio
    from .quality import measure_audio
    work = private_path(args.work)
    inputs = [Path(p).expanduser().resolve() for p in (args.project, args.cue, args.audio) if p]
    for name in ("djstudio_audit.json", "djstudio_audit.json.partial", "export_qa.log"):
        target = work / name
        if any(target.resolve() == source or
               (target.exists() and source.exists() and target.samefile(source)) for source in inputs):
            raise ValueError("Audit output must not overwrite a project, CUE or audio input")
    duration = None
    measurement = None
    if args.measure and not args.audio:
        raise ValueError("--measure requires --audio")
    if args.audio:
        if args.measure:
            measurement = measure_audio(args.audio, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                                        log=work / "export_qa.log")
            duration = measurement["duration_seconds"]
        else:
            result = subprocess.run([args.ffprobe, "-v", "error", "-select_streams", "a:0",
                                     "-show_streams", "-show_format", "-of", "json", args.audio],
                                    capture_output=True, check=True)
            data = json.loads(result.stdout)
            if not data.get("streams"):
                raise ValueError("Export has no audio stream")
            stream = data["streams"][0]
            duration = float(stream.get("duration") or data["format"]["duration"])
            measurement = {"path": str(Path(args.audio).resolve()), "codec": stream["codec_name"],
                           "sample_rate": int(stream["sample_rate"]), "channels": int(stream["channels"]),
                           "duration_seconds": duration, "full_decode_ok": None,
                           "measurement": "container probe only; no loudness, peaks or full decode check"}
    report = audit_djstudio(args.project, cue=args.cue, export_duration_s=duration,
                            minimum_track_s=args.min_track_seconds,
                            include_native_state=args.include_native_state)
    report["audio_measurement"] = measurement
    declared_duration = report.get("project_duration_metadata_s")
    if duration is not None and declared_duration is not None and abs(duration - declared_duration) > 2.0:
        report["warnings"].append("Project duration metadata differs from the exported audio; "
                                  "use the export duration for timeline review and investigate the mismatch.")
    report["listening_review"] = "pending; metadata and measurements do not certify audible quality"
    write_json(work / "djstudio_audit.json", report)
    print(f"AI Super DJ: {report['track_count']} tracks, {report['transition_count']} transitions; "
          f"CUE order matches: {report['cue_order_matches']}. Report: {work / 'djstudio_audit.json'}", flush=True)
    return report


def _source(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--playlist", help="Private JSON {tracks:[{id,path}]}")
    group.add_argument("--folder", help="Recursively select random audio files")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("beat_this", "librosa"), default="beat_this")
    parser.add_argument("--cache")
    parser.add_argument("--checkpoint", default="final0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--grid-json", help="Explicit imported grid overrides keyed by track id")


def _planning(parser, seed=True):
    parser.add_argument("--mode", choices=("auto", "fixed"), default="auto")
    parser.add_argument("--fixed-order", help="JSON array of every track id, in user order")
    parser.add_argument("--first")
    parser.add_argument("--last")
    parser.add_argument("--energy-arc", choices=("build", "release", "wave", "flat", "none"), default="build")
    parser.add_argument("--max-tempo-change", type=float, default=0.06)
    parser.add_argument("--no-tempo-reset", action="store_true")
    if seed:
        parser.add_argument("--seed", type=int, default=0)


def _rendering(parser):
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="AI_Super_DJ_001")
    parser.add_argument("--audamcp", required=True, help="Audited external checkout")
    parser.add_argument("--bitwize", required=True, help="Audited external checkout")
    parser.add_argument("--git", default="git")
    parser.add_argument("--rubberband", required=True)
    parser.add_argument("--prepared", help="Private JSON of existing FLOAT preparations")
    parser.add_argument("--noise-profiles", help="Explicit noise-only reference files by track id")
    parser.add_argument("--transition-bars", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--max-intro-trim-seconds", type=float, default=32.0,
                        help="Maximum intro skipped to reach a stable musical transition window")
    parser.add_argument("--target-lufs", type=float, default=-14.0)
    parser.add_argument("--allow-unverified", action="store_true", help="Allow explicitly labelled free-time short transitions when grids are uncertain")
    parser.add_argument("--allow-hard-cut", action="store_true", help="Explicitly permit deliberate zero-overlap tempo resets; disabled by default")
    parser.add_argument("--no-mp3", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ai-super-dj", description="AI Super DJ: offline analysis, ordering, rendering and verified export")
    subs = parser.add_subparsers(dest="command", required=True)
    ap = subs.add_parser("analyze")
    ap.add_argument("--work", required=True)
    _audio(ap); _source(ap)
    pp = subs.add_parser("plan")
    pp.add_argument("--work", required=True); pp.add_argument("--records", required=True)
    _planning(pp)
    rp = subs.add_parser("render")
    rp.add_argument("--work", required=True); rp.add_argument("--records", required=True); rp.add_argument("--plan", required=True)
    _audio(rp); _rendering(rp)
    runp = subs.add_parser("run")
    runp.add_argument("--work", required=True)
    _audio(runp); _source(runp); _planning(runp, seed=False); _rendering(runp)
    djp = subs.add_parser("djstudio-audit", help="Read-only DJ.Studio project/CUE/export audit")
    djp.add_argument("--project", required=True, help="Native .djs archive; never modified")
    djp.add_argument("--cue", help="Exported CUE chapters")
    djp.add_argument("--audio", help="Exported audio for duration and optional full-file QA")
    djp.add_argument("--work", required=True, help="Private report folder outside this repository")
    djp.add_argument("--min-track-seconds", type=float, default=120.0)
    djp.add_argument("--include-native-state", action="store_true", help="Preserve complete native metadata in the private report")
    djp.add_argument("--measure", action="store_true", help="Decode and measure the complete exported audio")
    _audio(djp)
    setup = subs.add_parser("setup", help="Fetch pinned external checkouts and optionally official model weights")
    setup.add_argument("--components", required=True); setup.add_argument("--git", default="git")
    setup.add_argument("--download-model", action="store_true"); setup.add_argument("--cache"); setup.add_argument("--checkpoint", default="final0")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        analyze(args)
    elif args.command == "plan":
        plan(args)
    elif args.command == "render":
        render(args)
    elif args.command == "run":
        records = analyze(args); planned = plan(args, records); render(args, records, planned)
    elif args.command == "djstudio-audit":
        djstudio_audit(args)
    else:
        from .adapters import setup_components
        setup_components(args.components, args.git)
        if args.download_model:
            from .analysis import prepare_beat_model
            prepare_beat_model(args.checkpoint, cache_dir=args.cache, allow_download=True)
