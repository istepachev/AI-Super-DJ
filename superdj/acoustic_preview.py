"""Overlap-only correction screening. A preview never certifies a full deck."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import soundfile as sf


def screen_correction(source, warp, initial_audit, sample_rate, directory, *, rubberband,
                      engine_args, run, audit, maximum_attempts=8, completed_attempts=1):
    """Return a promising map; the caller must render and independently audit it.

    Cropping changes a stretcher's context. Keep source context around each
    overlap, and decline previews covering most of the deck. No preview PCM is
    ever spliced into the delivered mix.
    """
    from .sync import compensate_backend_drift
    from .execution import current
    if warp.get('source_attack_controls_unapplied'):
        return None
    anchors = warp['anchors']
    source_frames = sf.info(source).frames
    ranges = []
    for role, mapping in warp['input_output_beat_mapping'].items():
        if not mapping:
            continue
        first = round((mapping[0]['source_seconds'] - warp['trim_start_seconds']) * sample_rate)
        last = round((mapping[-1]['source_seconds'] - warp['trim_start_seconds']) * sample_rate)
        ranges.append((role, max(0, first-2*sample_rate), min(source_frames, last+2*sample_rate)))
    if not ranges or sum(b-a for _, a, b in ranges) >= .60 * source_frames:
        return None
    candidate = deepcopy(warp)
    evidence = initial_audit
    best = None
    seen = set()
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    inputs = {}
    calibration = None
    try:
        for role, first, last in ranges:
            path = directory / (role + '-input.wav')
            with sf.SoundFile(source) as stream, sf.SoundFile(path, 'w', samplerate=sample_rate,
                    channels=stream.channels, subtype='FLOAT') as output:
                stream.seek(first)
                remaining = last-first
                while remaining:
                    block = stream.read(min(65536, remaining), dtype='float32', always_2d=True)
                    if not len(block):
                        raise ValueError('Incomplete acoustic preview source')
                    output.write(block)
                    remaining -= len(block)
            inputs[role] = path
        for attempt in range(maximum_attempts+1):
            if attempt:
                compensate_backend_drift(candidate, evidence, sample_rate,
                                         strength=1.0 if completed_attempts+attempt-1 < 2 else .5)
            signature = tuple((a['source_frame'], a['output_frame']) for a in candidate['anchors'])
            if signature in seen:
                break
            seen.add(signature)
            measured, unverified = [], []
            for role, first, last in ranges:
                x = np.array([a['source_frame'] for a in candidate['anchors']])
                y = np.array([a['output_frame'] for a in candidate['anchors']])
                offset, end = (int(round(v)) for v in np.interp([first, last], x, y))
                frames = end-offset
                controls = [(0, 0), *((int(a-first), int(b-offset)) for a,b in zip(x,y) if first<a<last), (last-first, frames)]
                if any(d<=b for (_,b),(_,d) in zip(controls, controls[1:])):
                    return None
                map_path = directory / (role + '.txt')
                map_path.write_text(''.join(f'{a} {b}\n' for a,b in controls), encoding='ascii')
                output = directory / (role + '-output.wav')
                run([rubberband, *engine_args, '--centre-focus', '--ignore-clipping', '--quiet',
                     '--time', f'{frames/(last-first):.12f}', '--timemap', map_path.name,
                     inputs[role].name, output.name], cwd=directory)
                info = sf.info(output)
                if info.subtype not in {'FLOAT', 'DOUBLE'} or abs(info.frames-frames)>128:
                    return None
                local = deepcopy(candidate)
                local['input_output_beat_mapping'] = {r: [] for r in candidate['input_output_beat_mapping']}
                local['input_output_beat_mapping'][role] = deepcopy(candidate['input_output_beat_mapping'][role])
                for point in local['input_output_beat_mapping'][role]:
                    point['output_seconds'] -= offset/sample_rate
                local['source_kick_alignment'] = {role: deepcopy(candidate.get('source_kick_alignment', {}).get(role, {}))}
                if not local['source_kick_alignment'][role]:
                    local['source_kick_alignment'] = {}
                else:
                    for event in local['source_kick_alignment'][role]['events']:
                        event['output_seconds'] -= offset/sample_rate
                part = audit(source, output, local, sample_rate)
                for item in part['measured']:
                    for key in ('expected_seconds', 'observed_seconds', 'grid_output_seconds'):
                        item[key] += offset/sample_rate
                measured.extend(part['measured'])
                unverified.extend(part['unverified'])
            if current():
                current().count('acoustic_preview_attempts')
            key = lambda point: (point['role'], point['beat'], point.get('view'))
            if calibration is None:
                original = {key(p): p['delay_seconds'] for p in initial_audit['measured']}
                local = {key(p): p['delay_seconds'] for p in measured}
                if not original or not original.keys() <= local.keys():
                    return None
                # A cropped stretcher can have a repeatable context offset.
                # Measure that offset on the unchanged map before predicting
                # whether a correction improves the actual full deck.
                calibration = {k: delay-local[k] for k,delay in original.items()}
                continue
            calibrated = []
            for point in measured:
                if key(point) in calibration:
                    point['delay_seconds'] += calibration[key(point)]
                    point['observed_seconds'] += calibration[key(point)]
                    calibrated.append(point)
            if len(calibrated) != len(calibration):
                return None
            measured = calibrated
            evidence = {'measured': measured, 'unverified': unverified,
                        'maximum_measured_drift_seconds': max((abs(p['delay_seconds']) for p in measured), default=None)}
            maximum = evidence['maximum_measured_drift_seconds']
            if maximum is None:
                return None
            if best is None or maximum < best[0]:
                best = maximum, deepcopy(candidate)
            if maximum <= .003:
                break
        return best[1] if best and best[0] <= .003 else None
    finally:
        # Only this call's private files, never source audio or cached objects.
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
        directory.rmdir()
