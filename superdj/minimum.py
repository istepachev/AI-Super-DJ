"""Short-play policy plus compatibility with older bounded-repeat archives."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import soundfile as sf

from .render import RenderError, _grid, _beat_window


def legacy_minimum(plan, records, prepared, directory):
    """Repeat at most four stable bars inside the solo body for a small deficit.

    Mixture, prepared voice and drums follow the identical edit. No invented beat
    timings, silence padding or repeating an entire section. Locks/protected
    sources are not altered. Record the original-to-edited source timeline.
    Large deficits require another recording/decision rather than fake length.
    """
    result, rows, files = deepcopy(plan), deepcopy(records), deepcopy(prepared)
    edits = []
    directory = Path(directory)
    by_id = {r['id']: r for r in rows}
    for position, identity in enumerate(result['order']):
        section = result.get('sections', {}).get(identity, {})
        minimum = section.get('minimum_play_seconds')
        if minimum is None or section['predicted_play_seconds'] >= minimum:
            continue
        row = by_id[identity]
        deficit = minimum - section['predicted_play_seconds']
        edge = result['edges'][position] if position < len(result['edges']) else None
        if deficit > 10 or row.get('protected_ranges') or section.get('locked_start') or section.get('locked_end'):
            raise RenderError(f"{identity}: cannot safely meet the critical playback minimum")
        grid = _grid(row, .75)
        count = edge['overlap_bars'] if edge is not None else 0
        end_bar = int(np.argmin(abs(grid[3] - section['end'])))
        insertion_index = end_bar - count
        insertion = float(grid[3][insertion_index])
        core_start = section['start']
        if position:
            core_start = float(grid[3][np.argmin(abs(grid[3] - section['start'])) + result['edges'][position - 1]['overlap_bars']])
        choices = []
        voice_profile = (row.get('vocal_stem') or {}).get('profile', [])
        for bar_end in range(insertion_index, 0, -1):
            loop_end = float(grid[3][bar_end])
            if loop_end <= core_start:
                break
            for repeat_bars in (1, 2, 4):
                if bar_end < repeat_bars:
                    continue
                try:
                    window = _beat_window(grid[2], grid[3], bar_end - repeat_bars,
                                          repeat_bars, row['bpm'], 48000)
                except RenderError:
                    continue  # Another interior window can have valid evidence.
                loop_start = float(window.times[0]); length = loop_end - loop_start
                if length >= deficit and length <= 10 and loop_start >= core_start:
                    if np.max(abs(np.diff(window.times) / (60 / row['bpm']) - 1)) <= .15:
                        voices = [float(v['foreground_risk']) for v in voice_profile
                                  if loop_start <= v['time'] < loop_end and 'foreground_risk' in v]
                        voice_risk = float(np.mean(voices)) if voices else 0.
                        choices.append((voice_risk, round(length, 10), -loop_end,
                                        length, loop_start, repeat_bars, loop_end))
        if not choices:
            raise RenderError(f"{identity}: no short stable bar repeat meets playback minimum")
        voice_risk, _, _, length, loop_start, repeat_bars, insertion = min(choices)
        item = files[identity]
        streams=[key for key in ('path','vocal_path','drum_path') if item.get(key)]
        infos={key:sf.info(item[key]) for key in streams}
        native=infos['path']
        if any((value.samplerate,value.frames,value.channels)!=(native.samplerate,native.frames,2)
               or value.subtype not in {'FLOAT','DOUBLE'} for value in infos.values()):
            raise RenderError('A minimum repeat needs all prepared streams on the same stereo sample timeline')
        directory.mkdir(parents=True, exist_ok=True)

        def repeat_file(source, name):
            destination = directory / name
            if destination.exists():
                raise FileExistsError(destination)
            with sf.SoundFile(source) as inp:
                sr, channels = inp.samplerate, inp.channels
                if inp.subtype not in {'FLOAT', 'DOUBLE'} or channels != 2:
                    raise RenderError("A minimum repeat needs prepared stereo floating-point audio")
                first, last = round(loop_start * sr), round(insertion * sr)
                if abs((last - first) / sr - length) > 1 / sr:
                    raise RenderError("Repeat duration does not match the source beat grid")
                inp.seek(first)
                loop = inp.read(last - first, dtype='float32', always_2d=True)
                if len(loop) != last - first or not np.isfinite(loop).all():
                    raise RenderError("Invalid source repeat")
                # Tiny endpoint conditioning suppresses a splice click without
                # moving attack/bar positions or changing the loop duration.
                width = min(round(.002 * sr), len(loop) // 4)
                if width:
                    inp.seek(max(0, last - width))
                    before = inp.read(width, dtype='float32', always_2d=True)
                    inp.seek(last)
                    after = inp.read(width, dtype='float32', always_2d=True)
                    ramp = np.linspace(0., 1., width, dtype=np.float32)[:, None]
                    loop[:width] = before * (1 - ramp) + loop[:width] * ramp
                    loop[-width:] = loop[-width:] * (1 - ramp) + after * ramp
                inp.seek(0)
                with sf.SoundFile(destination, 'w', samplerate=sr, channels=channels, subtype='FLOAT') as out:
                    remaining = last
                    while remaining:
                        block = inp.read(min(262144, remaining), dtype='float32', always_2d=True)
                        if not len(block):
                            raise RenderError("Repeat insertion beyond source")
                        out.write(block); remaining -= len(block)
                    out.write(loop)
                    for block in inp.blocks(blocksize=262144, dtype='float32', always_2d=True):
                        out.write(block)
            return str(destination)

        item['path'] = repeat_file(item['path'], identity + '_minimum.wav')
        for key,label in (('vocal_path','voice'),('drum_path','drums')):
            if item.get(key):item[key]=repeat_file(item[key],identity+'_minimum_'+label+'.wav')
        edit = {'id': identity, 'method': 'one short stable whole-bar repeat inside solo body',
                'loop_start_seconds': loop_start, 'insertion_source_seconds': insertion,
                'added_seconds': length, 'bars_repeated': repeat_bars,
                'reason': 'critical minimum selected playback; model endpoint otherwise too early',
                'repeat_selection': 'stable interior window, least estimated voice activity, then shortest addition',
                'estimated_voice_risk': voice_risk if voice_profile else None,
                'aligned_preparation_streams':streams,
                'original_source_duration_seconds': row['duration'],
                'synthetic_beat_extension': False, 'splices_conditioned_seconds': .002}
        item.setdefault('provenance', {})['timeline_edit'] = edit
        row['timeline_edit'] = edit
        row['grid']['timeline_edit'] = edit
        row['original_duration_seconds'] = row['duration']
        row['duration'] += length
        row['duration_seconds'] = row['duration']
        original_beats = np.asarray(row['beats'])
        repeat_mask = (original_beats >= loop_start - 1e-6) & (original_beats < insertion - 1e-6)
        for key in ('beats', 'downbeats'):
            values = np.asarray(row[key])
            copied = values[(values >= loop_start - 1e-6) & (values < insertion - 1e-6)] + length
            row[key] = np.sort(np.r_[np.where(values >= insertion - 1e-6, values + length, values), copied]).tolist()
        for key in ('neural_downbeat_probability_by_beat', 'neural_beat_probability_by_beat'):
            values = np.asarray(row['grid'].get(key, []))
            if len(values) == len(original_beats):
                ix = int(np.searchsorted(original_beats, insertion - 1e-6))
                row['grid'][key] = np.r_[values[:ix], values[repeat_mask], values[ix:]].tolist()
        for cue in row.get('cue_candidates', []):
            if cue['time'] >= insertion - 1e-6:
                cue['time'] += length
                cue['seconds'] = cue['time']
        original_structure = deepcopy(row.get('structure', []))
        row['structure'] = sorted([
            {**s, 'time': s['time'] + (length if s['time'] >= insertion - 1e-6 else 0)} for s in original_structure
        ] + [{**s, 'time': s['time'] + length} for s in original_structure
             if loop_start - 1e-6 <= s['time'] < insertion - 1e-6], key=lambda s: s['time'])
        section['end'] += length
        for key in ('source_span_seconds', 'predicted_solo_seconds', 'predicted_play_seconds'):
            section[key] += length
        section['minimum_repeat'] = edit
        if edge is not None:edge['cue_locks']['outgoing_end_seconds'] += length
        result['section_warnings'] = [w for w in result.get('section_warnings', [])
                                      if not w.startswith(identity + ': source and stable grid permit only')]
        result['section_warnings'].append(f"{identity}: one {length:.2f}s stable-bar repeat meets minimum playback ({section['predicted_play_seconds']:.2f}s including overlaps)")
        edits.append(edit)
    result['minimum_playback_edits'] = edits
    return result, rows, files


def preserve_minimum(plan,records,prepared,directory):
    """Factory: 60-second floor and optional, verified extension toward 120."""
    from .short_tracks import preserve_short_tracks
    return preserve_short_tracks(plan,records,prepared,directory)
