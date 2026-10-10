"""Observable phase, duration and anti-layering checks for automatic sets."""
import os
import numpy as np
import pytest
import soundfile as sf

from superdj.compact import arrange_sync
from superdj.mixing import blend, envelopes
from superdj.planner import plan_tracks
from superdj.render import render_plan
from superdj.sync import shared_clock
from superdj.minimum import preserve_minimum
from test_new_workflow import record


def test_extended_energetic_core_and_solo_budget_are_distinct_from_deck_span():
    records = [record('a', 480), record('b', 450), record('c', 480)]
    for row in records:
        row['structure'] = [{'time': t, 'energy': 1. if 100 <= t < 360 else .1,
                             'vocal_activity': 0.} for t in row['downbeats']]
    original = plan_tracks(records, mode='fixed')
    result = arrange_sync(original, records)
    assert result['order'] == ['a', 'b', 'c']
    assert all(edge['transition_policy'] == 'sync' for edge in result['edges'])
    assert all(edge['overlap_bars'] == 32 for edge in result['edges'])
    middle = result['sections']['b']
    assert 180 <= middle['predicted_solo_seconds'] <= 240
    assert middle['source_span_seconds'] > middle['predicted_solo_seconds'] + 100
    assert middle['start'] > 20
    assert 'sections' not in original


def test_short_sources_reduce_overlaps_instead_of_erasing_the_solo_core():
    rows = [record('a', 420), record('b', 150), record('c', 420)]
    result = arrange_sync(plan_tracks(rows, mode='fixed'), rows)
    assert all(e['overlap_bars'] == 8 for e in result['edges'])
    assert result['sections']['b']['predicted_solo_seconds'] >= 110
    assert result['sections']['b']['predicted_play_seconds'] >= 120
    assert result['sections']['b']['duration_exception']


def test_compact_respects_explicit_boundaries_and_protected_material():
    rows = [record('a', 420), record('b', 420)]
    rows[1]['protected_ranges'] = [[100, 200]]
    plan = plan_tracks(rows, mode='fixed')
    plan['edges'][0].update(overlap_bars=16, cue_locks={'outgoing_end_seconds': 260.25,
                                                     'incoming_start_seconds': 64.25})
    result = arrange_sync(plan, rows)
    assert result['sections']['a']['end'] == 260.25
    assert result['sections']['b']['start'] == 64.25
    assert result['sections']['b']['end'] >= 200
    assert result['edges'][0]['overlap_bars'] == 16


def test_native_partial_phrase_retains_its_exact_beat_locks():
    rows = [record('a', 420), record('b', 420)]
    plan = plan_tracks(rows, mode='fixed')
    locks = {'outgoing_end_seconds': 260.25, 'incoming_start_seconds': 64.75}
    plan['edges'][0].update(overlap_beats=51, cue_locks=locks)
    result = arrange_sync(plan, rows)
    assert result['edges'][0]['overlap_beats'] == 51
    assert 'overlap_bars' not in result['edges'][0]
    assert result['edges'][0]['cue_locks'] == locks
    assert result['sections']['b']['start'] == 64.75


def test_explicit_review_is_not_silently_replaced_with_sync():
    rows = [record('a', 420), record('b', 420)]
    plan = plan_tracks(rows, mode='fixed')
    plan['edges'][0]['transition_policy'] = 'review'
    with pytest.raises(ValueError, match='cannot be overwritten'):
        arrange_sync(plan, rows)


def test_curves_never_boost_layered_bands_and_mids_handoff_faster_than_highs():
    clock = shared_clock(128, 145, 128, 8000)
    n = clock['frames']
    phase = np.interp(np.arange(n), clock['output_frames'], clock['phase']) / 128
    phase[-1] = 1.
    gains, report = envelopes(n, 128, beat_phase=phase)
    for band in ('bass', 'mid', 'high'):
        assert np.max(gains[band + '_out'] + gains[band + '_in']) <= 1 + 1e-12
    assert report['incoming_mid_opening_beats'] == 16
    assert report['bass_handoff_beats'] == 4
    assert report['phase_source'] == 'shared integrated clock'
    assert np.all(gains['mid_out'][phase > .55] == 0)
    assert np.all(gains['mid_in'][phase < .30] == 0)


def test_identical_decks_do_not_gain_three_db_in_highs_and_mids():
    sr = 16000
    t = np.arange(4 * sr) / sr
    signal = .1 * (np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 5000 * t))
    audio = np.column_stack((signal, signal)).astype(np.float32)
    mixed, _ = blend(audio, audio, sr, beats=128)
    assert np.max(abs(mixed)) <= np.max(abs(audio)) + .0002


def test_real_compact_render_matches_solo_budget_and_shared_phase(tmp_path):
    ffmpeg, rubberband = os.getenv('SUPERDJ_FFMPEG'), os.getenv('SUPERDJ_RUBBERBAND')
    if not ffmpeg or not rubberband:
        pytest.skip('Actual audio tools required')
    sr = 8000
    rows = [record('a', 320, 120), record('b', 320, 124)]
    for row in rows:
        t = np.arange(320 * sr) / sr
        audio = .07 * np.sin(2 * np.pi * 440 * t)
        path = tmp_path / (row['id'] + '.wav')
        sf.write(path, np.column_stack((audio, audio)), sr, subtype='FLOAT')
        row['path'] = str(path)
    plan = arrange_sync(plan_tracks(rows, mode='fixed'), rows, sample_rate=sr)
    result = render_plan(plan, rows, {}, tmp_path / 'set.wav', tmp_path / 'render',
                         ffmpeg=ffmpeg, rubberband=rubberband, sample_rate=sr)
    assert result['transitions'][0]['tempo_technique'] == 'shared_phase_sync'
    assert result['transitions'][0]['frequency_processing']['phase_source'] == 'shared integrated clock'
    for track in result['tracks']:
        expected = plan['sections'][track['id']]['predicted_solo_seconds']
        assert abs(track['solo_seconds'] - expected) < .001
        assert 180 <= track['solo_seconds'] <= 240
    assert sf.info(tmp_path / 'set.wav').frames == result['total_frames']


def test_minimum_repeat_preserves_phase_voice_edit_and_original_file(tmp_path):
    sr = 8000
    rows = [record('a', 125, 100), record('b', 320, 126)]
    rows[0]['downbeats'] = [t for t in rows[0]['downbeats'] if t < 119]
    original = .1 * np.sin(2 * np.pi * 300 * np.arange(125 * sr) / sr)
    audio = np.column_stack((original, original)).astype(np.float32)
    source, voice = tmp_path / 'a.wav', tmp_path / 'voice.wav'
    sf.write(source, audio, sr, subtype='FLOAT'); sf.write(voice, audio * .5, sr, subtype='FLOAT')
    plan = arrange_sync(plan_tracks(rows, mode='fixed'), rows)
    assert plan['sections']['a']['predicted_play_seconds'] < 120
    prepared = {'a': {'path': str(source), 'vocal_path': str(voice)}, 'b': {'path': 'unused'}}
    extended, revised, processed = preserve_minimum(plan, rows, prepared, tmp_path / 'repeat')
    assert extended['sections']['a']['predicted_play_seconds'] >= 120
    assert any('meets minimum playback' in w for w in extended['section_warnings'])
    assert not any('permit only' in w for w in extended['section_warnings'])
    assert len(extended['minimum_playback_edits']) == 1
    edit = extended['minimum_playback_edits'][0]
    assert edit['bars_repeated'] == 2 and edit['added_seconds'] < 5
    assert not edit['synthetic_beat_extension']
    mixture, _ = sf.read(processed['a']['path'], dtype='float32')
    foreground, _ = sf.read(processed['a']['vocal_path'], dtype='float32')
    np.testing.assert_array_equal(mixture * .5, foreground)
    np.testing.assert_array_equal(sf.read(source, dtype='float32')[0], audio)
    assert len(mixture) == len(audio) + round(edit['added_seconds'] * sr)
    assert revised[0]['duration'] == rows[0]['duration'] + edit['added_seconds']
    assert plan['sections']['a']['predicted_play_seconds'] < 120  # No mutation.
