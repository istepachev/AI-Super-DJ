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
    playlist = select_tracks(args.folder, args.count, seed,allow_repeats=getattr(args,'allow_repeats',False))
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
        if track.get("protected_ranges"):
            import math
            ranges = track["protected_ranges"]
            if not isinstance(ranges, list) or any(not isinstance(r, list) or len(r) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in r) or
                not 0 <= r[0] < r[1] <= records[-1]["duration"] for r in ranges):
                raise ValueError("Protected ranges must be valid [start,end] source seconds")
            records[-1]["protected_ranges"] = ranges
        write_json(Path(args.work) / "records.partial.json", records)
    write_json(Path(args.work) / "records.json", records)
    return records


def plan(args, records=None):
    from .planner import plan_tracks
    records = records if records is not None else _records(args.records)
    order = read_json(args.fixed_order) if args.fixed_order else None
    result = plan_tracks(records, mode=args.mode, fixed_order=order, first_id=args.first, last_id=args.last,
                         seed=args.seed or 0, energy_arc=args.energy_arc, max_tempo_change=args.max_tempo_change,
                         tempo_ramp_max_change=args.tempo_ramp_max_change,
                         allow_tempo_reset=not args.no_tempo_reset)
    if args.transition_overrides:
        import math
        overrides = read_json(args.transition_overrides)
        if not isinstance(overrides, list):
            raise ValueError("Transition overrides must be a JSON array")
        edges = {(edge["from"], edge["to"]): edge for edge in result["edges"]}
        seen = set()
        for item in overrides:
            if not isinstance(item, dict) or set(item) - {"from", "to", "cue_locks", "overlap_bars", "overlap_beats", "transition_policy"}:
                raise ValueError("Transition override supports only from, to, cue_locks, overlap_bars, overlap_beats and transition_policy")
            pair = (item.get("from"), item.get("to"))
            if pair not in edges or pair in seen:
                raise ValueError("Transition override must identify one unique existing adjacent pair")
            locks = item.get("cue_locks", {})
            bars = item.get("overlap_bars")
            beats = item.get("overlap_beats")
            policy = item.get("transition_policy")
            if not isinstance(locks, dict) or (not locks and bars is None and beats is None and policy is None) or set(locks) - {"incoming_start_seconds", "outgoing_end_seconds"}:
                raise ValueError("Use explicit incoming-start and/or outgoing-end source seconds")
            for value in locks.values():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("Locked source seconds must be nonnegative finite numbers")
            if locks:
                edges[pair]["cue_locks"] = dict(locks)
            if bars is not None:
                if isinstance(bars, bool) or not isinstance(bars, int) or bars not in {2, 4, 8, 16, 32, 40, 44}:
                    raise ValueError("overlap_bars must be 2, 4, 8, 16, 32, 40 or 44")
                edges[pair]["overlap_bars"] = bars
            if beats is not None:
                if bars is not None or isinstance(beats, bool) or not isinstance(beats, int) or not 8 <= beats <= 256 or len(locks) != 2:
                    raise ValueError("overlap_beats requires 8–256 integer beats, both cue locks and no overlap_bars")
                edges[pair]["overlap_beats"] = beats
            if policy is not None:
                if policy not in {"auto", "sync", "review"}:
                    raise ValueError("transition_policy must be auto, sync or review")
                edges[pair]["transition_policy"] = policy
            seen.add(pair)
    # Automatic sets have one shared-phase mixer and an explicit solo budget.
    # Zhilin prefers 40-bar overlaps using the same core selection/mixer.
    from .compact import arrange_sync
    if all(row.get('downbeats') and row.get('duration', row.get('duration_seconds')) for row in records):
        result = arrange_sync(result, records, zhilin=getattr(args, 'style', 'default') == 'zhilin', ffmpeg=getattr(args, 'ffmpeg', None))
    else:
        result['section_warnings'] = ['Order-only plan: audio analysis/grid required before compact SYNC rendering']
    write_json(Path(args.work) / "plan.json", result)
    sync_count = sum(edge.get('transition_policy') == 'sync' for edge in result['edges'])
    print(f"Planned {len(result['order'])} tracks; {sync_count} shared-phase SYNC transitions; {result['method']}", flush=True)
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
                               allow_unverified=args.allow_unverified,
                               tempo_conflict_policy=args.tempo_conflict_policy)
    # Preserve previous intermediates when a user deliberately repeats a run.
    if (work / "premix.wav").exists():
        work = work / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
        work.mkdir(parents=True, exist_ok=False)
    if args.prepared:
        prepared = read_json(args.prepared)
    else:
        profiles = read_json(args.noise_profiles) if args.noise_profiles else None
        prepared = prepare_tracks(records, work, audamcp=args.audamcp, ffmpeg=args.ffmpeg,
                                  git=args.git, noise_profiles=profiles, progress=lambda s: print(s, flush=True),
                                  sample_rate=args.sample_rate)
    for record in records:
        item = prepared[record["id"]]
        if item.get("loudness_lufs") is None:
            measured = measure_audio(item["path"], ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                                     log=work / f"{record['id']}_prepared_qa.log")
            item["source_measurements"] = measured
            item["loudness_lufs"] = measured["integrated_lufs"]
        record["loudness_lufs"] = item["loudness_lufs"]
    from .minimum import preserve_minimum
    planned, records, prepared = preserve_minimum(planned, records, prepared, work / 'minimum_playback')
    write_json(work / 'effective_plan.json', planned)
    write_json(work / 'effective_records.json', records)
    write_json(work / "prepared.json", prepared)
    premix = work / "premix.wav"
    print("Rendering tempo maps and transitions", flush=True)
    rendered = render_plan(planned, records, prepared, premix, work / "render",
                           ffmpeg=args.ffmpeg, rubberband=args.rubberband,
                           sample_rate=args.sample_rate,
                           max_tempo_change=planned.get("constraints", {}).get("max_tempo_change_per_deck", 0.06),
                           max_intro_trim_seconds=args.max_intro_trim_seconds,
                           transition_bars=args.transition_bars, allow_unverified=args.allow_unverified,
                           allow_hard_cut=args.allow_hard_cut,
                           tempo_conflict_policy=args.tempo_conflict_policy,
                           tempo_ramp_max_change=args.tempo_ramp_max_change if args.tempo_ramp_max_change is not None else 0.25,
                           tempo_ramp_seconds=args.tempo_ramp_seconds,
                           native_exit_seconds=args.native_exit_seconds,
                           sync_max_change=args.sync_max_change,
                           sync_transition_bars=args.sync_transition_bars,
                           bridge_overlap_beats=args.bridge_overlap_beats,
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
    from .delivery import make_cue,publish_pair
    from .cues import runtime_config
    from .history import register_manifest,import_cue_folder,write_text_index
    cue=make_cue(manifest)
    ready=runtime_config().get('delivery_root')
    if ready:
        manifest['delivery']=publish_pair(exported['flac']['path'],cue,ready)
        write_json(directory/f'{args.name}_manifest.json',manifest)
    history_result=register_manifest(manifest,manifest_path=directory/f'{args.name}_manifest.json')
    if ready and history_result['registered']:
        import_cue_folder(ready);write_text_index(ready)
    if history_result['registered']:
        print(f"Completed history updated: {history_result['unique_sources']} unique sources",flush=True)
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
    parser.add_argument('--allow-repeats',action='store_true',help='Explicitly allow previously used sources in automatic folder selection')
    parser.add_argument("--grid-json", help="Explicit imported grid overrides keyed by track id")


def _harmonic_options(parser):
    parser.add_argument("--harmonic-mode", choices=("fuzzy", "mood"), default="fuzzy")
    parser.add_argument("--priority", type=float, default=.5, help="0=BPM, 1=key; our explicit weighting")
    parser.add_argument("--direction", choices=("ascending", "descending", "none"), default="ascending")
    parser.add_argument("--bpm-tolerance", type=float, help="Soft absolute adjacent-BPM tolerance, independent of render stretch caps")


def _harmonic_search_options(parser):
    parser.add_argument("--order-objective", choices=("compatible_first", "weighted"),
                        default="compatible_first", help=argparse.SUPPRESS)
    parser.add_argument("--search-seconds", type=float, default=20., help=argparse.SUPPRESS)


def harmonize(args):
    from .harmonize import harmonize_tracks
    from .harmonize_report import export_order_pack
    records = _records(args.records)
    settings = dict(mode=args.harmonic_mode, priority=args.priority,
        direction=args.direction, bpm_tolerance=args.bpm_tolerance, first_id=args.first,
        last_id=args.last, locked_chains=read_json(args.locked_chains) if args.locked_chains else None,
        seed=args.seed)
    if getattr(args, 'ranking_profile', None):
        from .harmonize_modes_sequence import sequence_mode_tracks
        result = sequence_mode_tracks(records, read_json(args.ranking_profile),
            routing_priority=_routing_priority(args), **settings)
    else:
        if getattr(args, 'routing_priority', None) is not None:
            raise ValueError('--routing-priority requires a mode ranking profile')
        result = harmonize_tracks(records, **settings, beam_width=args.beam_width,
            order_objective=args.order_objective, search_seconds=args.search_seconds)
    exported = export_order_pack(records, result, args.work)
    print(f"Harmonized {len(result['order'])} tracks; report: {exported['report']}", flush=True)
    return result


def harmonize_compare(args):
    from .harmonize_compare import compare_orders, load_order, load_analysis
    from .harmonize import harmonize_tracks, harmonic_relation
    records = _records(args.records)
    our_order = load_order(args.our_order, records)
    other_order = load_order(args.other_order, records)
    other_analysis = load_analysis(args.other_order, records, djs_key_source=args.djs_key_source)
    settings = dict(harmonic_mode=args.harmonic_mode, priority=args.priority,
                    direction=args.direction, bpm_tolerance=args.bpm_tolerance)
    result = compare_orders(records, our_order, other_order, other_analysis=other_analysis, **settings)
    # A second search uses exported DJ metadata only when ALL keys/BPM exist.
    if other_analysis:
        by_id = {str(r.get('id') or r.get('path')): r for r in records}
        dj_rows = [{'id': r['id'], 'path': by_id[r['id']].get('path'), **r} for r in other_analysis]
        if len(dj_rows) == len(records) and all(r.get('bpm') and harmonic_relation(r.get('key_camelot') or r.get('key'), r.get('key_camelot') or r.get('key'))['key_from'] for r in other_analysis):
            rerun = harmonize_tracks(dj_rows, mode=args.harmonic_mode, priority=args.priority,
                direction=args.direction, bpm_tolerance=args.bpm_tolerance, seed=args.seed,
                first_id=args.first, last_id=args.last, order_objective=args.order_objective,
                search_seconds=args.search_seconds)
            result['replanned_on_other_metadata'] = rerun
            result['comparison_on_other_metadata'] = compare_orders(dj_rows, rerun['order'], other_order, **settings)
        else:
            result['replanned_on_other_metadata'] = None
            result['replan_reason'] = 'Complete unambiguous exported BPM and key required; no numeric key encoding is guessed'
    result['experiment_controls'] = {'seed': args.seed, 'first_id': args.first, 'last_id': args.last,
                                    'order_objective': args.order_objective, 'search_seconds': args.search_seconds,
                                    'djs_key_source': args.djs_key_source,
                                    'dj_settings': read_json(args.dj_settings) if args.dj_settings else None}
    if Path(args.our_order).suffix.lower() == '.json':
        original = read_json(args.our_order)
        if isinstance(original, dict) and original.get('settings'):
            expected = dict(mode=args.harmonic_mode, priority=args.priority,
                            direction=args.direction, bpm_tolerance=args.bpm_tolerance)
            result['input_plan_settings'] = original['settings']
            result['comparison_objective_differs_from_input_plan'] = original['settings'] != expected
            result['comparison_search_differs_from_input_plan'] = original.get('order_objective', 'weighted') != args.order_objective
    write_json(Path(args.work) / 'harmonize_comparison.json', result)
    print(f"Comparison written: {Path(args.work) / 'harmonize_comparison.json'}", flush=True)
    return result


def _harmonize_examples(manifest_path, *, priority_aware=False):
    from .harmonize_compare import load_order
    manifest = Path(manifest_path)
    examples = []
    for entry in read_json(manifest):
        if entry.get('training_eligible') is False and not (
                priority_aware and entry.get('priority_aware_training_eligible') is True):
            continue
        source = entry['records']
        if isinstance(source, str):
            source = manifest.parent / source
            records = _records(source)
        else:
            records = source
        order = entry['order']
        if isinstance(order, str):
            order = load_order(manifest.parent / order, records)
        # Keep cohort/mode provenance and other descriptive manifest metadata.
        # Only musical records and the demonstrated order enter fitted features.
        examples.append(dict(entry, records=records, order=order))
    return examples


def harmonize_learn(args):
    examples = _harmonize_examples(args.examples, priority_aware=args.learner == 'priority')
    if args.learner == 'priority':
        from .harmonize_modes import fit_priority_mode_ranker_fixed, rank_mode_candidates, _prepare
        from .harmonize_priority import _example_controls
        profile = fit_priority_mode_ranker_fixed(examples, min_candidates=args.min_candidates)
        write_json(Path(args.work) / 'priority_mode_ranker_profile.json', profile)
        diagnostics = []
        controls = [_example_controls(example) for example in examples]
        for example, context in zip(_prepare(examples, args.min_candidates), controls):
            counts = dict(steps=0, literal_top3=0, unique_top1=0, blocked=0)
            for current, pool in example['queries']:
                ranked = rank_mode_candidates(current, pool, profile, mode=example['mode'], **context)
                identity = str(pool[0].get('id') or pool[0].get('path'))
                target = next(row for row in ranked['candidates'] if row['id'] == identity)
                counts['steps'] += 1
                counts['literal_top3'] += identity in ranked['shortlist_ids']
                counts['unique_top1'] += target['rank'] == 1 and target['tie_count'] == 1
                counts['blocked'] += not target['eligible']
            diagnostics.append(dict(name=example['name'], mode=example['mode'], routing_controls=context, **counts))
        result = dict(scope='Training-set diagnostics after fixed-option fit; not independent validation',
            training_only=True, independent_validation=False, selection=profile['selection'], rows=diagnostics)
        write_json(Path(args.work) / 'priority_mode_ranker_training_diagnostics.json', result)
        print(f"Fixed priority prototype and training-only diagnostics written: {args.work}; defaults unchanged", flush=True)
        return result
    if args.learner == 'modes':
        from .harmonize_modes import FAMILIES, cross_validate_mode_ranker, fit_mode_ranker
        result = cross_validate_mode_ranker(examples, feature_families=FAMILIES,
                                           min_candidates=args.min_candidates)
        write_json(Path(args.work) / 'mode_ranker_validation.json', result)
        profile = fit_mode_ranker(examples, feature_families=FAMILIES,
                                  min_candidates=args.min_candidates)
        write_json(Path(args.work) / 'mode_ranker_profile.json', profile)
        print(f"Mode-aware pool ranker and source-cohort validation written: {args.work}; defaults unchanged", flush=True)
        return result
    if args.learner == 'ranker':
        from .harmonize_ranker import cross_validate_ranker, fit_ranker
        result = cross_validate_ranker(examples, min_candidates=args.min_candidates,
                                       mode=args.harmonic_mode)
        write_json(Path(args.work) / 'successor_ranker_validation.json', result)
        profile = fit_ranker(examples, min_candidates=args.min_candidates)
        write_json(Path(args.work) / 'successor_ranker_profile.json', profile)
        print(f"Experimental local ranker and held-out validation written: {args.work}; defaults unchanged", flush=True)
        return result
    from .harmonize_calibration import cross_validate_successor_controls
    result = cross_validate_successor_controls(examples, min_candidates=args.min_candidates)
    write_json(Path(args.work) / 'successor_calibration.json', result)
    print(f"Held-out successor calibration written: {Path(args.work) / 'successor_calibration.json'}", flush=True)
    return result


def harmonize_rank(args):
    from .harmonize_ranker import rank_remaining_candidates
    records = _records(args.records)
    by_id = {str(r.get('id') or r.get('path')): r for r in records}
    if len(by_id) != len(records):
        raise ValueError('Track IDs must be unique')
    prefix = read_json(args.played_prefix) if args.played_prefix else [args.current]
    if (not isinstance(prefix, list) or not prefix or any(not isinstance(i, str) for i in prefix)
            or len(set(prefix)) != len(prefix) or not set(prefix) <= set(by_id)
            or prefix[-1] != args.current):
        raise ValueError('Played prefix must contain distinct known IDs and end with the current track')
    played = set(prefix)
    remaining = [r for i, r in by_id.items() if i not in played]
    profile = read_json(args.profile)
    if profile.get('model_version') in {'superdj-mode-pool-ranker-2', 'superdj-mode-priority-ranker-3'}:
        from .harmonize_modes import rank_mode_candidates
        result = rank_mode_candidates(by_id[args.current], remaining, profile, mode=args.harmonic_mode,
            priority=_routing_priority(args), bpm_tolerance=getattr(args, 'bpm_tolerance', None))
    else:
        if (getattr(args, 'routing_priority', None) is not None or getattr(args, 'priority', None) is not None
                or getattr(args, 'bpm_tolerance', None) is not None):
            raise ValueError('Routing controls require a mode ranking profile')
        result = rank_remaining_candidates(by_id[args.current], remaining, profile, mode=args.harmonic_mode)
    result.update(current_id=args.current, played_prefix=prefix,
                  remaining_ids=[str(r.get('id') or r.get('path')) for r in remaining],
                  whole_path_optimized=False)
    write_json(Path(args.work) / 'next_track_rankings.json', result)
    print(f"Experimental remaining-track ranks written: {args.work}; whole-path order unchanged", flush=True)
    return result


def _routing_priority(args):
    """Explicit categorical override; numeric endpoints only, no middle guess."""
    from .harmonize import _number
    from .harmonize_priority import routing_controls
    numeric = getattr(args, 'priority', None)
    if numeric is not None:
        numeric = _number(numeric, 'priority')
        if not 0 <= numeric <= 1:
            raise ValueError('priority must be from0(BPM) to1(key)')
    explicit = getattr(args, 'routing_priority', None)
    if explicit is not None:
        return routing_controls(explicit, getattr(args, 'bpm_tolerance', None))['priority']
    return 'bpm' if numeric == 0 else 'key' if numeric == 1 else 'unknown'


def _planning(parser, seed=True):
    parser.add_argument("--style", choices=("default", "zhilin"), default="default")
    parser.add_argument("--mode", choices=("auto", "fixed"), default="auto")
    parser.add_argument("--fixed-order", help="JSON array of every track id, in user order")
    parser.add_argument("--transition-overrides", help="Private JSON array of {from,to,cue_locks?,overlap_bars?} for existing adjacent pairs")
    parser.add_argument("--first")
    parser.add_argument("--last")
    parser.add_argument("--energy-arc", choices=("build", "release", "wave", "flat", "none"), default="build")
    parser.add_argument("--max-tempo-change", type=float, default=0.06)
    parser.add_argument("--no-tempo-reset", action="store_true", help="Require a bounded common tempo; disallow native-tempo fallback bridges")
    if "--tempo-ramp-max-change" not in parser._option_string_actions:
        parser.add_argument("--tempo-ramp-max-change", type=float, default=None,
                            help="Opt in to wider shared-tempo plans; disabled by default")
    if seed:
        parser.add_argument("--seed", type=int, default=0)


def _rendering(parser):
    parser.add_argument("--sample-rate", type=int, default=44100, help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="AI_Super_DJ_001")
    parser.add_argument("--audamcp", required=True, help="Audited external checkout")
    parser.add_argument("--bitwize", required=True, help="Audited external checkout")
    parser.add_argument("--git", default="git")
    parser.add_argument("--rubberband", required=True)
    parser.add_argument("--prepared", help="Private JSON of existing FLOAT preparations")
    parser.add_argument("--noise-profiles", help="Explicit noise-only reference files by track id")
    parser.add_argument("--transition-bars", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--max-intro-trim-seconds", type=float, default=32.0,
                        help="Maximum intro skipped to reach a stable musical transition window")
    parser.add_argument("--target-lufs", type=float, default=-14.0)
    parser.add_argument("--allow-unverified", action="store_true", help="Allow explicitly labelled free-time short transitions when grids are uncertain")
    parser.add_argument("--allow-hard-cut", action="store_true", help="Explicitly permit deliberate zero-overlap tempo resets; disabled by default")
    parser.add_argument("--tempo-conflict-policy", choices=("auto", "sync", "adaptive", "shared_tempo", "review", "echo_bridge"), default="sync",
                        help="Default SYNC uses one integrated phase and tempo curve for both decks")
    parser.add_argument("--sync-max-change", type=float, default=0.40,
                        help="Temporary rate change cap during shared SYNC, not the main track sections")
    parser.add_argument("--sync-transition-bars", type=int, choices=(2, 4, 8, 16, 32), default=None,
                        help="Override conflict overlap length; default long: 32 bars, or 16 above 20%% temporary rate change")
    if "--tempo-ramp-max-change" not in parser._option_string_actions:
        parser.add_argument("--tempo-ramp-max-change", type=float, default=None,
                            help="Explicit shared-tempo cap; a supplied plan can restrict it further")
    parser.add_argument("--tempo-ramp-seconds", type=float, default=15.0,
                        help="Smooth tempo change in output seconds (10 to 20); default 15")
    parser.add_argument("--native-exit-seconds", type=float, default=8.0,
                        help="Short outgoing-only vinyl overlap with native incoming tempo (4 to 20 seconds)")
    parser.add_argument("--bridge-overlap-beats", type=float, default=8.0,
                        help="Echo bridge length in incoming beats (4 to 32), not a beatmatched phrase")
    parser.add_argument("--no-mp3", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ai-super-dj", description="AI Super DJ: offline analysis, ordering, rendering and verified export")
    subs = parser.add_subparsers(dest="command", required=True)
    mp = subs.add_parser("make", help="One file → mastered FLAC; multiple files → mixed FLAC")
    mp.add_argument("inputs", nargs="+", help="Audio files or a folder")
    mp.add_argument("--style", choices=("default", "zhilin"), default=None)
    mp.add_argument('--allow-repeats',action='store_true',help='Explicitly allow previously used sources in automatic folder selection')
    mp.add_argument("--keep-order", action="store_true", help="Keep supplied file order")
    mp.add_argument("--output")
    mp.add_argument("--name")
    mp.add_argument("--work", help=argparse.SUPPRESS)
    ap = subs.add_parser("analyze")
    ap.add_argument("--work", required=True)
    _audio(ap); _source(ap)
    pp = subs.add_parser("plan")
    pp.add_argument("--work", required=True); pp.add_argument("--records", required=True)
    _planning(pp)
    hp = subs.add_parser("harmonize", help="Experimental whole-playlist BPM/key ordering and DJ.Studio test pack")
    hp.add_argument("--records", required=True); hp.add_argument("--work", required=True)
    _harmonic_options(hp)
    _harmonic_search_options(hp)
    hp.add_argument("--first"); hp.add_argument("--last")
    hp.add_argument("--locked-chains", help="Private JSON array of disjoint adjacent id chains")
    hp.add_argument("--seed", type=int, default=0)
    hp.add_argument("--beam-width", type=int, default=512)
    hp.add_argument("--ranking-profile", help=argparse.SUPPRESS)
    hp.add_argument("--routing-priority", choices=("key", "bpm", "unknown"),
                    help="Categorical control for an optional priority-aware ranking profile")
    cp = subs.add_parser("harmonize-compare", help="Compare orders on identical metadata; distinguish detector disagreements")
    _harmonic_search_options(cp)
    cp.add_argument("--records", required=True); cp.add_argument("--work", required=True)
    cp.add_argument("--our-order", required=True); cp.add_argument("--other-order", required=True)
    cp.add_argument("--dj-settings", help="Private JSON recording actual app version, slider and locks")
    cp.add_argument("--djs-key-source", choices=("djstudio", "mixedinkey", "original"), default="djstudio",
                    help="Explicit exported key source; unavailable alternate indices fall back with provenance")
    _harmonic_options(cp)
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("--first"); cp.add_argument("--last")
    lp = subs.add_parser("harmonize-learn", help="Fit local next-track controls; validate on held-out playlists")
    lp.add_argument("--examples", required=True, help="Private JSON list of {name, records, order}; paths relative to this manifest")
    lp.add_argument("--work", required=True)
    lp.add_argument("--min-candidates", type=int, default=4)
    lp.add_argument("--learner", choices=("controls", "ranker", "modes", "priority"), default="controls", help=argparse.SUPPRESS)
    lp.add_argument("--harmonic-mode", choices=("fuzzy", "mood"), default="fuzzy", help=argparse.SUPPRESS)
    nlp = subs.add_parser("harmonize-rank", help="Experimental local next-track ranks; keeps independent harmony checks")
    nlp.add_argument("--records", required=True); nlp.add_argument("--profile", required=True)
    nlp.add_argument("--current", required=True); nlp.add_argument("--work", required=True)
    nlp.add_argument("--played-prefix", help="Private JSON array of played IDs ending with current; omitted means current only")
    nlp.add_argument("--harmonic-mode", choices=("fuzzy", "mood"), default="fuzzy")
    nlp.add_argument("--priority", type=float, help="0=BPM,1=key; intermediate numeric controls remain unknown to the fitted ranker")
    nlp.add_argument("--routing-priority", choices=("key", "bpm", "unknown"))
    nlp.add_argument("--bpm-tolerance", type=float, help="Explicit nullable soft preference for a priority-aware profile")
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
    setup.add_argument("--download-cue-model", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "make":
        from .make import make
        make(args)
    elif args.command == "analyze":
        analyze(args)
    elif args.command == "plan":
        plan(args)
    elif args.command == "harmonize":
        harmonize(args)
    elif args.command == "harmonize-compare":
        harmonize_compare(args)
    elif args.command == "harmonize-learn":
        harmonize_learn(args)
    elif args.command == "harmonize-rank":
        harmonize_rank(args)
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
        if args.download_cue_model:
            from .cues import download_checkpoint
            download_checkpoint(Path.home() / ".cache" / "superdj" / "cue-detr")
