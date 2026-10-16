"""Explicit revisions from verified sources; original sets are never overwritten."""
from copy import deepcopy
from pathlib import Path


def revised_plan(manifest, *, drop=None, transition=None, sample_rate=44100, ffmpeg=None):
    from .planner import plan_tracks
    from .compact import arrange_sync
    from .recovery import alternative
    from .transition_quality import TransitionFailure
    old = manifest['plan']
    records = deepcopy(manifest['selection_and_analysis'])
    order = old['order']
    if len(set(order))!=len(order) or set(order)!={r['id'] for r in records}:
        raise ValueError('Revision manifest must contain every unique analyzed source once')
    if bool(drop)==bool(transition):
        raise ValueError('Choose either removal or one transition repair')
    if transition:
        pair = tuple(transition)
        if pair not in [(a,b) for a,b in zip(order,order[1:])]:
            raise ValueError('Transition repair requires an existing adjacent pair')
        plan, evidence = alternative(old, records, TransitionFailure(*pair,'explicit transition repair'), {}, 1,
            sample_rate=sample_rate, ffmpeg=ffmpeg)
        plan['revision'] = {'kind':'transition','original_order_preserved':True,'selection':evidence}
        return records, plan
    removed = set(drop)
    if len(removed)!=len(drop) or not removed<=set(order):
        raise ValueError('Removal requires unique existing track ids')
    remaining = [identity for identity in order if identity not in removed]
    if len(remaining)<2:
        raise ValueError('A revised mixed set needs at least two retained tracks')
    records = [r for r in records if r['id'] not in removed]
    first = min(order.index(identity) for identity in removed)
    prefix = order[:first]
    suffix = [identity for identity in order[first:] if identity not in removed]
    by_id = {r['id']:r for r in records}
    constraints = old.get('constraints',{})
    options = {'harmonic_mode':old.get('harmonic_mode','classic'),
        'energy_arc':old.get('set_arc',{}).get('shape','build'),
        'max_tempo_change':constraints.get('max_tempo_change_per_deck',.06),
        'tempo_ramp_max_change':constraints.get('tempo_ramp_max_change_per_deck'),
        'allow_tempo_reset':constraints.get('allow_tempo_reset',True),
        'weights':old.get('weights'),'seed':constraints.get('seed',0)}
    if options['energy_arc']=='custom intensity only':
        raise ValueError('A custom position curve needs explicit revision targets')
    candidates = (prefix[-1:] if suffix else [])+suffix
    if suffix:
        tail = plan_tracks([by_id[key] for key in candidates], mode='auto',
            first_id=prefix[-1] if prefix else None,
            last_id=constraints.get('last_id') if constraints.get('last_id') in suffix else None, **options)
        new_order = prefix[:-1]+tail['order'] if prefix else tail['order']
    else:
        new_order = prefix
    plan = plan_tracks(records, mode='fixed', fixed_order=new_order, **options)
    old_pairs = {(e['from'],e['to']):e for e in old['edges']}
    new_pairs = {(e['from'],e['to']) for e in plan['edges']}
    sections = old.get('sections',{})
    for pair, edge in old_pairs.items():
        if removed.intersection(pair):
            continue
        locked = sections.get(pair[0],{}).get('locked_end') or sections.get(pair[1],{}).get('locked_start')
        if locked and pair not in new_pairs:
            raise ValueError('Replanning conflicts with a retained locked transition; provide a revised lock constraint')
    for edge in plan['edges']:
        previous = old_pairs.get((edge['from'],edge['to']),{})
        for key in ('cue_locks','overlap_bars','overlap_beats','transition_policy','mid_handoff_bars','bass_handoff_beats'):
            if key in previous:
                edge[key] = deepcopy(previous[key])
        edge.setdefault('transition_policy','sync')
    for key in ('style','workflow_preferences','mixing_preferences','long_play_explicit'):
        if key in old:
            plan[key] = deepcopy(old[key])
    settings = plan.get('mixing_preferences',{})
    plan = arrange_sync(plan,records,sample_rate=sample_rate,ffmpeg=ffmpeg,zhilin=plan.get('style')=='zhilin',
        target_play_seconds=settings.get('target_play_seconds',210.),
        maximum_play_seconds=settings.get('maximum_play_seconds',300.),
        allow_long_play=bool(plan.get('long_play_explicit')))
    for identity, section in plan['sections'].items():
        for flag in ('locked_start','locked_end'):
            section[flag] = sections.get(identity,{}).get(flag,False)
    assert plan['order'][:len(prefix)]==prefix
    assert len(plan['order'])==len(remaining) and set(plan['order'])==set(remaining)
    plan['revision'] = {'kind':'removal','removed_ids':sorted(removed),'preserved_prefix':prefix,
        'suffix_replanned':bool(suffix),'all_remaining_edges_rescored':True,
        'harmonic_compatibility_is_soft_objective':True}
    return records, plan


def validated_records(manifest, session, *, ffmpeg):
    from . import stems
    records = deepcopy(manifest['selection_and_analysis'])
    def one(record):
        expected = record.get('provenance',{}).get('source_sha256')
        if not expected or session.digest(record['path'])!=expected:
            raise ValueError(f"{record['id']}: original source changed or lacks a verifiable hash; analyze it again")
        estimate = record.get('vocal_stem')
        if estimate:
            hashes = estimate.get('artifact_sha256',{})
            try:
                valid = 'path' in hashes and all(session.digest(estimate[key])==value for key,value in hashes.items())
            except (OSError,KeyError):
                valid = False
            if not valid:
                config = stems.configuration()
                if config is None:
                    raise ValueError('The previous set used separated stems; configure the same model before revision')
                previous = estimate.get('provenance',{})
                if (previous.get('model')!=config['model'] or previous.get('weights')!=config['weights']
                        or previous.get('demucs_version')!=config['version']):
                    raise ValueError('Stem model/weights changed since the previous set; analyze with the new setup explicitly')
                record['vocal_stem'] = stems.separate(record['path'],expected,session.root,config,ffmpeg)
        return record
    return session.map(one,records)


def revise(args):
    from .cues import runtime_config
    from .execution import execution
    from .resources import preflight
    from .storage import read_json, write_json, private_path
    from .cli import main
    manifest = read_json(args.manifest)
    runtime = runtime_config()
    old_export = Path(manifest['exports']['flac']['path']).resolve()
    output = private_path(args.output)
    if not args.name or Path(args.name).name!=args.name or any(c in args.name for c in '/\\:'):
        raise ValueError('Use a simple new output filename stem')
    if (output/f'{args.name}.flac').resolve()==old_export or (output/f'{args.name}.flac').exists():
        raise FileExistsError('Revision must use a new output name; preserve the previous set')
    # Refuse nonportable hidden DSP choices rather than silently replacing them.
    limits = manifest['render'].get('limits',{})
    if limits.get('allow_unverified') or limits.get('allow_hard_cut') or limits.get('tempo_conflict_policy','sync')!='sync':
        raise ValueError('Creative/raw montage requires its original explicit render settings')
    removed = set(args.drop or [])
    source_manifest = deepcopy(manifest)
    source_manifest['selection_and_analysis'] = [r for r in manifest['selection_and_analysis'] if r['id'] not in removed]
    cache = None if args.no_cache else args.cache or (runtime.get('cache_root') if runtime.get('cache_mode')=='persistent' else None)
    resources = preflight([r['path'] for r in source_manifest['selection_and_analysis']], output=output,
        scratch=runtime.get('scratch_root'),cache=cache,jobs=args.jobs or runtime.get('jobs'),
        devices=runtime.get('gpu_devices',()),ffprobe=runtime.get('ffprobe','ffprobe'),sample_rate=manifest['render']['sample_rate'])
    with execution(cache,jobs=resources['jobs'],devices=resources['devices'],scratch_root=runtime.get('scratch_root'),
                   report=output/'performance.json') as session:
        session.metrics['resources'] = resources
        verified = validated_records(source_manifest,session,ffmpeg=runtime.get('ffmpeg','ffmpeg'))
        working = deepcopy(manifest)
        # Keep removed records only for order/id validation; they need no decoding.
        verified_map = {row['id']:row for row in verified}
        working['selection_and_analysis'] = [verified_map.get(row['id'],row) for row in manifest['selection_and_analysis']]
        records, planned = revised_plan(working,drop=args.drop,transition=args.transition,
            sample_rate=manifest['render']['sample_rate'],ffmpeg=runtime.get('ffmpeg','ffmpeg'))
        write_json(session.work/'records.json',records);write_json(session.work/'plan.json',planned)
        argv = ['render','--records',str(session.work/'records.json'),'--plan',str(session.work/'plan.json'),
            '--work',str(session.work),'--output',str(output),'--name',args.name,'--no-mp3',
            '--sample-rate',str(manifest['render']['sample_rate']),
            '--target-lufs',str(manifest.get('mastering',{}).get('provenance',{}).get('preset',{}).get('target_lufs',-14.))]
        for flag in ('audamcp','bitwize','rubberband','ffmpeg','ffprobe','git'):
            argv.extend(['--'+flag,runtime.get(flag,flag)])
        for flag,key in (('sync-max-change','sync_max_temporary_rate_change'),
                         ('tempo-ramp-max-change','tempo_ramp_max_change'),('tempo-ramp-seconds','tempo_ramp_requested_output_seconds')):
            if key in limits:
                argv.extend(['--'+flag,str(limits[key])])
        main(argv)
