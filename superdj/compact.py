"""Source-only SYNC selection with a full-deck playback duration budget."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .render import RenderError, _Window, _grid, _beat_window, _local_window_evidence, _local_validation_possible
from .sync import shared_clock
from .rhythm import bass_profile, support, weak_gap


def arrange_sync(plan, records, *, target_solo_seconds=None, maximum_solo_seconds=None,
                 target_play_seconds=210., maximum_play_seconds=300.,
                 minimum_play_seconds=120., sample_rate=48000, zhilin=False, ffmpeg=None,
                 allow_long_play=False, excluded_sections=None, section_ranges=None):
    """Choose energetic cores and stable overlap windows without project timings.

    Full playback includes both adjacent overlaps. Solo excludes them. Short sources
    retain as much usable content as possible. Insufficient evidence/source
    length is recorded, never hidden by silence, repetition or a hard cut.
    Explicit locks/overlap lengths remain mandatory.
    """
    if (not np.isfinite(target_play_seconds) or not np.isfinite(maximum_play_seconds)
            or target_play_seconds < 180 or maximum_play_seconds < target_play_seconds
            or (not allow_long_play and (target_play_seconds > 240 or maximum_play_seconds > 300))):
        raise ValueError("Full playback target must be 180–240 seconds; maximum at most 300")
    play_low,play_high=(max(120.,target_play_seconds-30),target_play_seconds+30) if allow_long_play else (180.,240.)
    if maximum_solo_seconds is None:maximum_solo_seconds=maximum_play_seconds
    if target_solo_seconds is not None and not 180 <= target_solo_seconds <= 240:
        raise ValueError("Legacy solo target must be 180–240 seconds; full playback cap still applies")
    if not 8 <= maximum_solo_seconds <= maximum_play_seconds:
        raise ValueError("Solo maximum must fit within the full playback maximum")
    if minimum_play_seconds < 120 or minimum_play_seconds > target_play_seconds:
        raise ValueError("Minimum selected playback must be 120 seconds or longer")
    result = deepcopy(plan)
    by_id = {str(row['id']): row for row in records}
    order = result['order']
    if set(order) != set(by_id) or len(set(order)) != len(order):
        raise ValueError("SYNC selection must retain every unique input")
    windows = {}
    profiles = {identity: bass_profile(by_id[identity], ffmpeg=ffmpeg) for identity in order}

    def candidates(identity, count, native=False):
        key = identity, count, native
        if key in windows:
            return windows[key]
        row = by_id[identity]
        grid = _grid(row, .75)
        if not grid[0] and not _local_validation_possible(row, grid):
            raise RenderError(f"{identity}: compact SYNC requires a usable grid: {grid[1]}")
        choices = []
        beat_count = round(count * 4)
        indices = range(len(grid[2]) - beat_count) if native else range(len(grid[3]) - count)
        for i in indices:
            try:
                window = (_Window(grid[2][i:i + beat_count + 1].copy(), count, 4, row['bpm'],
                                  round(beat_count * 60 / row['bpm'] * sample_rate)) if native else
                          _beat_window(grid[2], grid[3], i, count, row['bpm'], sample_rate, grid[4]))
                if np.max(np.abs(np.diff(window.times) / (60 / row['bpm']) - 1)) > .15:
                    continue
                evidence = _local_window_evidence(row, grid[2], window)
                if not grid[0] and (not evidence['strong_local_neural_evidence'] or evidence['local_timing_confidence'] < .75):
                    continue
                choices.append((float(window.times[0]), float(window.times[-1]), i, evidence))
            except RenderError:
                continue
        windows[key] = choices
        return choices

    counts = []
    explicit = []
    natives = []
    for edge in result['edges']:
        if edge.get('transition_policy', 'sync') != 'sync':
            raise ValueError('An explicit different technique needs a reviewed raw/preserving plan; it cannot be overwritten by compact SYNC')
        native_beats = edge.get('overlap_beats')
        if native_beats is not None:
            if isinstance(native_beats, bool) or not isinstance(native_beats, int) or not 8 <= native_beats <= 256 or edge.get('overlap_bars') is not None or len(edge.get('cue_locks', {})) != 2:
                raise ValueError("Native beat exception needs 8–256 beats, both locks and no overlap_bars")
        a, b = by_id[edge['from']], by_id[edge['to']]
        change = max(abs(a['bpm'] / b['bpm'] - 1), abs(b['bpm'] / a['bpm'] - 1))
        if 'overlap_bars' in edge:
            supplied = edge['overlap_bars']
            if isinstance(supplied, bool) or not isinstance(supplied, int) or supplied not in {2, 4, 8, 16, 32, 40, 44}:
                raise ValueError('Invalid explicit overlap_bars')
        requested = native_beats / 4 if native_beats else edge.get('overlap_bars')
        preferred = requested or (16 if change > .20 else 32)
        available = [preferred] if requested else [n for n in (40, 32, 16, 8) if n <= preferred]
        chosen = next((n for n in available if candidates(a['id'], n, native_beats is not None) and candidates(b['id'], n, native_beats is not None)), None)
        if chosen is None:
            raise RenderError(f"{a['id']} → {b['id']}: no supported common SYNC window")
        counts.append(chosen)
        explicit.append(requested is not None)
        natives.append(native_beats is not None)

    selection_cache = {}
    def choose_section(position):
        cache_key = (position, counts[position - 1] if position else 0,
                     counts[position] if position < len(counts) else 0)
        if cache_key in selection_cache:
            return selection_cache[cache_key]
        identity = order[position]
        row = by_id[identity]
        duration = float(row.get('duration', row.get('duration_seconds')))
        previous = result['edges'][position - 1] if position else {}
        following = result['edges'][position] if position < len(counts) else {}
        start_lock = (previous.get('cue_locks') or {}).get('incoming_start_seconds')
        end_lock = (following.get('cue_locks') or {}).get('outgoing_end_seconds')
        endpoints = [float(t) for t in row['downbeats']]
        heads = candidates(identity, counts[position - 1], natives[position - 1]) if position else [(t, t, 0, {}) for t in [0.] + endpoints]
        tails = candidates(identity, counts[position], natives[position]) if position < len(counts) else [(t, t, 0, {}) for t in endpoints + [duration]]
        rejected=(result.get('rejected_attack_window_events') or {}).get(identity,{})
        heads=[w for w in heads if not any(w[0]<=t<=w[1] for t in rejected.get('head',[]))]
        tails=[w for w in tails if not any(w[0]<=t<=w[1] for t in rejected.get('tail',[]))]
        if start_lock is not None:
            heads = [w for w in heads if abs(w[0] - start_lock) < 1 / sample_rate]
        elif position:
            heads = [w for w in heads if support(profiles[identity], w[0] + .40 * (w[1] - w[0]), w[1]) >= .80
                     and support(profiles[identity], w[0] + .48 * (w[1] - w[0]), w[0] + .70 * (w[1] - w[0])) >= .80
                     and weak_gap(profiles[identity], w[0] + .48 * (w[1] - w[0]), w[0] + .70 * (w[1] - w[0])) <= .80]
        if end_lock is not None:
            tails = [w for w in tails if abs(w[1] - end_lock) < 1 / sample_rate]
        elif position < len(counts):
            tails = [w for w in tails if support(profiles[identity], w[0], w[0] + .65 * (w[1] - w[0])) >= .80
                     and support(profiles[identity], w[0] + .40 * (w[1] - w[0]), w[0] + .60 * (w[1] - w[0])) >= .80
                     and weak_gap(profiles[identity], w[0] + .40 * (w[1] - w[0]), w[0] + .60 * (w[1] - w[0])) <= .80]
        structure = row.get('structure', [])
        ts = np.array([s['time'] for s in structure])
        energy = np.array([s['energy'] for s in structure])
        activity = np.array([s.get('vocal_activity', s.get('tonal_activity_proxy', 0)) for s in structure])
        cumulative = np.r_[0., np.cumsum(energy)]
        activity_sum = np.r_[0., np.cumsum(activity)]

        def mean_between(values, first, last):
            left, right = np.searchsorted(ts, [first, last])
            return float((values[right] - values[left]) / max(1, right - left))

        cue_scores = {round(c['time'], 4): float(c.get('confidence', 0)) + .5 * float(c.get('novelty', 0))
                      for c in row.get('cue_candidates', [])}
        in_duration = shared_clock(by_id[order[position - 1]]['bpm'], row['bpm'], round(counts[position - 1] * 4), sample_rate)['duration_seconds'] if position else 0.
        out_duration = shared_clock(row['bpm'], by_id[order[position + 1]]['bpm'], round(counts[position] * 4), sample_rate)['duration_seconds'] if position < len(counts) else 0.
        protected = row.get('protected_ranges', [])
        # The late-head bass gap is independent of every possible tail.
        # Cache once per head to avoid repeatedly scanning the same profile
        # in the Cartesian source-window search.
        posthandoff_gaps = ({(w[0], w[1]): weak_gap(profiles[identity],
                             w[0] + .55 * (w[1] - w[0]), w[1]) for w in heads}
                           if position and start_lock is None else {})
        choices = []
        for head in heads:
            for tail in tails:
                start, core_start, core_end, end = head[0], head[1], tail[0], tail[1]
                if any(abs(start-a)<1/sample_rate and abs(end-b)<1/sample_rate
                       for a,b in (excluded_sections or {}).get(identity, [])):
                    continue
                limits = (section_ranges or {}).get(identity, {})
                if any(not low <= value <= high for value, (low,high) in
                       ((start, limits.get('start', (-float('inf'),float('inf')))),
                        (end, limits.get('end', (-float('inf'),float('inf')))))):
                    continue
                if any(all(abs(value-choice)>1/sample_rate for choice in limits.get(boundary+'_choices',[value]))
                       for boundary,value in (('start',start),('end',end))):
                    continue
                solo = core_end - core_start
                if solo < 8 or solo > maximum_solo_seconds + 1e-6:
                    continue
                if any(start > float(r[0]) or end < float(r[1]) for r in protected):
                    continue
                played = solo + in_duration + out_duration
                if played > maximum_play_seconds + 1e-6:
                    continue
                # Prioritise minimum total playback, then the full-deck target. A
                # short radio edit is not extended artificially to hit a goal.
                target_distance = abs(played - target_play_seconds)
                if target_solo_seconds is not None:
                    target_distance = abs(solo - target_solo_seconds)
                score = target_distance / 30 + 20 * max(0, minimum_play_seconds - played)
                score += 5 * max(0, play_low-played, played-play_high) / 30
                score -= 4 * mean_between(cumulative, core_start, core_end)
                score -= .5 * (cue_scores.get(round(start, 4), 0) + cue_scores.get(round(end, 4), 0))
                score += .7 * (mean_between(activity_sum, start, core_start) + mean_between(activity_sum, core_end, end))
                # After bass has transferred, a late intro break is no longer
                # covered by the old deck. Prefer windows that avoid a sustained
                # hole there; brief native fills remain permissible.
                if position and start_lock is None:
                    gap = posthandoff_gaps[(start, core_start)]
                    score += 5 * max(0., gap - .8)
                # Avoid selecting only an intro when an energetic central core
                # scores equally. This is not a claim of a known Cue ordinal.
                score += .15 * abs((core_start + core_end) / 2 - duration / 2) / max(duration, 1)
                choices.append((score, start, end, solo, played, head, tail))
        selection_cache[cache_key] = min(choices, key=lambda c: (c[0], c[1])) if choices else None
        return selection_cache[cache_key]

    # A source can support each overlap separately but not both and a useful
    # body. Prefer long transitions, but reduce them if doing so recovers the
    # minimum playback or the full playback target, including both overlaps.
    def deficiency(position, selection):
        if selection is None:
            return 10000.
        missing = max(0, minimum_play_seconds - selection[4]) * 20
        duration = float(by_id[order[position]].get('duration', by_id[order[position]].get('duration_seconds')))
        play_floor = min(play_low, duration)
        missing += max(0, play_floor - selection[4])
        missing += max(0, selection[4]-play_high)
        return missing

    for _ in range(len(counts) * 3 + 1):
        selections = [choose_section(i) for i in range(len(order))]
        failures = [i for i, section in enumerate(selections) if deficiency(i, section) > 0]
        if not failures:
            break
        trials = []
        for position in failures:
            for edge_index in (position - 1, position):
                if 0 <= edge_index < len(counts) and not explicit[edge_index] and counts[edge_index] > 8:
                    old = counts[edge_index]
                    counts[edge_index] = 32 if old == 40 else 16 if old == 32 else 8
                    new_selections = [choose_section(i) for i in (edge_index, edge_index + 1)]
                    improvement = sum(deficiency(i, selections[i]) - deficiency(i, s)
                                      for i, s in zip((edge_index, edge_index + 1), new_selections))
                    trials.append((improvement, edge_index, counts[edge_index]))
                    counts[edge_index] = old
        if not trials or max(trials)[0] <= 0:
            if any(s is None for s in selections):
                raise RenderError(f"{order[failures[0]]}: locks/source/grid leave no compact section within the {maximum_play_seconds:g}s full playback maximum; review cues/grid instead of extending the track")
            break  # Keep the best evidence-supported short-source exception.
        _, edge_index, count = max(trials)
        counts[edge_index] = count
    else:
        raise RenderError("Compact duration selection did not converge")

    sections, warnings = {}, []
    for position, (identity, selected) in enumerate(zip(order, selections)):
        _, start, end, solo, played, head, tail = selected
        sections[identity] = {'start': start, 'end': end, 'source_span_seconds': end - start,
            'predicted_solo_seconds': solo, 'predicted_play_seconds': played,
            'target_solo_seconds': max(8., target_play_seconds-(played-solo)), 'maximum_solo_seconds': maximum_solo_seconds,
            'target_play_seconds': target_play_seconds, 'maximum_play_seconds': maximum_play_seconds,
            'target_play_range_seconds': [play_low,play_high], 'allow_long_play': bool(allow_long_play),
            'duration_budget_basis': 'full deck including incoming and outgoing overlaps',
            'minimum_play_seconds': minimum_play_seconds,
            'selection': 'energetic source core; CUE-DETR/structure candidates and valid beat/bar windows; no project timing',
            'head_local_evidence': head[3], 'tail_local_evidence': tail[3],
            'incoming_posthandoff_weak_bass_maximum_seconds': weak_gap(profiles[identity], start + .55 * (head[1] - start), head[1]) if position else 0.,
            'locked_start': (result['edges'][position - 1].get('cue_locks') or {}).get('incoming_start_seconds') is not None if position else False,
            'locked_end': (result['edges'][position].get('cue_locks') or {}).get('outgoing_end_seconds') is not None if position < len(result['edges']) else False,
            'duration_exception': not play_low <= played <= play_high}
        if played < minimum_play_seconds:
            warnings.append(f"{identity}: source and stable grid permit only {played:.2f}s playback before minimum-playback preparation")
        elif not play_low <= played <= play_high:
            warnings.append(f"{identity}: source/grid/locks leave {played:.2f}s full playback outside the {play_low:g}–{play_high:g}s target; {solo:.2f}s solo")
    for i, edge in enumerate(result['edges']):
        edge.update(transition_policy='sync')
        if not natives[i]:
            edge['overlap_bars'] = counts[i]
        edge.setdefault('cue_locks', {}).update(outgoing_end_seconds=sections[edge['from']]['end'],
                                               incoming_start_seconds=sections[edge['to']]['start'])
    result.update(sections=sections, section_warnings=warnings, rhythm_profiles=profiles, style='zhilin' if zhilin else 'default',
        section_policy=f'shared-phase SYNC; target {play_low:g}–{play_high:g}s full playback including both overlaps, hard maximum {maximum_play_seconds:g}s; minimum 120s; target exceptions recorded',
        long_play_explicit=bool(allow_long_play))
    return result
