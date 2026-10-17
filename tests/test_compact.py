"""Observable phase, duration and anti-layering checks for automatic sets."""
import os
import shutil
import numpy as np
import pytest
import soundfile as sf

from superdj.compact import arrange_sync
from superdj.mixing import blend, envelopes
from superdj.planner import plan_tracks
from superdj.render import render_plan, _grid
from superdj.sync import shared_clock
from superdj.minimum import preserve_minimum
from superdj.rhythm import support
from test_new_workflow import record


def test_full_playback_budget_includes_both_long_overlaps():
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
    assert 180 <= middle['predicted_play_seconds'] <= 240
    assert middle['predicted_solo_seconds'] < 180
    assert all(s['predicted_play_seconds'] <= 300 for s in result['sections'].values())
    assert middle['source_span_seconds'] > middle['predicted_solo_seconds'] + 100
    assert middle['start'] > 20
    assert 'sections' not in original


def test_short_sources_keep_useful_body_and_record_full_duration_exception():
    rows = [record('a', 420), record('b', 150), record('c', 420)]
    result = arrange_sync(plan_tracks(rows, mode='fixed'), rows)
    assert result['sections']['b']['predicted_solo_seconds'] >= 8
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


def test_explicit_long_overlaps_never_expand_full_playback_past_five_minutes():
    rows=[record('a',600),record('b',600),record('c',600)]
    plan=plan_tracks(rows,mode='fixed')
    for edge in plan['edges']:edge['overlap_bars']=32
    result=arrange_sync(plan,rows,target_solo_seconds=240)
    assert all(section['predicted_play_seconds']<=300 for section in result['sections'].values())
    assert result['sections']['b']['predicted_solo_seconds']<180


@pytest.mark.parametrize('section',[{}, {'target_solo_seconds':210}, {'maximum_play_seconds':300}])
def test_rendered_full_duration_cannot_bypass_cap_with_legacy_solo_metadata(section):
    from superdj.render import _validate_playback_budget,RenderError
    # Same failure as the reported seventh track: its solo was under five
    # minutes but its complete timed PCM exceeded seven minutes.
    with pytest.raises(RenderError,match='including both overlaps'):
        _validate_playback_budget({'id':'seven','output_frames':round(426.136*44100)},section,44100)
    _validate_playback_budget({'id':'seven','output_frames':300*44100},section,44100)


@pytest.mark.parametrize('maximum',[float('nan'),float('inf'),0,-1])
def test_invalid_rendered_playback_cap_cannot_disable_the_guard(maximum):
    from superdj.render import _validate_playback_budget,RenderError
    with pytest.raises(RenderError,match='finite and positive'):
        _validate_playback_budget({'id':'track','output_frames':210*44100},
                                  {'maximum_play_seconds':maximum,'allow_long_play':True},44100)


def test_long_playback_is_available_only_after_explicit_opt_in():
    from superdj.render import _validate_playback_budget,RenderError
    rows=[record('a',600),record('b',600),record('c',600)]
    original=plan_tracks(rows,mode='fixed')
    with pytest.raises(ValueError):arrange_sync(original,rows,target_play_seconds=360,maximum_play_seconds=420)
    result=arrange_sync(original,rows,target_play_seconds=360,maximum_play_seconds=420,allow_long_play=True)
    assert result['long_play_explicit']
    assert all(330<=s['predicted_play_seconds']<=390 and s['allow_long_play'] for s in result['sections'].values())
    _validate_playback_budget({'id':'long','output_frames':360*44100},result['sections']['b'],44100)
    with pytest.raises(RenderError,match='explicit long-play'):
        _validate_playback_budget({'id':'long','output_frames':360*44100},{'maximum_play_seconds':420},44100)


def test_bass_support_uses_actual_duration_not_number_of_irregular_bins():
    profile = {'executed': True, 'reference_dbfs': -10, 'threshold_dbfs': -22,
               'rows': [{'time': 0., 'end': .1, 'bass_dbfs': -15},
                        {'time': .1, 'end': 2., 'bass_dbfs': -100}]}
    assert support(profile, 0, 2) == pytest.approx(.05)
    assert support(profile, .2, 1) == 0


def test_auto_entry_has_bass_at_handoff_but_explicit_quiet_cue_is_preserved(tmp_path):
    from superdj.cues import runtime_config
    ffmpeg = runtime_config().get('ffmpeg') or os.getenv('SUPERDJ_FFMPEG') or shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('FFmpeg required for actual bass-profile analysis')
    sr = 8000
    rows = [record('a', 420), record('b', 320)]
    t = np.arange(320 * sr) / sr
    signal = .1 * np.sin(2 * np.pi * 70 * t) * (t > 53) + .05 * np.sin(2 * np.pi * 700 * t)
    source = tmp_path / 'b.wav'
    sf.write(source, np.column_stack((signal, signal)), sr, subtype='FLOAT')
    rows[1]['path'] = str(source)
    plan = plan_tracks(rows, mode='fixed')
    auto = arrange_sync(plan, rows)
    start = auto['sections']['b']['start']
    span = auto['edges'][0]['overlap_bars'] * 4 * 60 / rows[1]['bpm']
    assert support(auto['rhythm_profiles']['b'], start + .48 * span, start + .70 * span) >= .90
    assert start > .25
    plan['edges'][0]['cue_locks'] = {'incoming_start_seconds': .25, 'outgoing_end_seconds': 260.25}
    locked = arrange_sync(plan, rows)
    assert locked['sections']['b']['start'] == .25


def test_strong_cue_does_not_choose_late_intro_bass_hole(tmp_path):
    from superdj.cues import runtime_config
    if not (runtime_config().get('ffmpeg') or os.getenv('SUPERDJ_FFMPEG') or shutil.which('ffmpeg')):
        pytest.skip('FFmpeg required for actual bass-profile analysis')
    sr = 8000
    rows = [record('a', 420), record('b', 320)]
    rows[1]['cue_candidates'] = [{'time': 32.25, 'confidence': 1., 'novelty': 1.}]
    t = np.arange(320 * sr) / sr
    signal = .1 * np.sin(2 * np.pi * 70 * t) * ~((t > 91) & (t < 96))
    source = tmp_path / 'late_break.wav'
    sf.write(source, np.column_stack((signal, signal)), sr, subtype='FLOAT')
    rows[1]['path'] = str(source)
    planned = arrange_sync(plan_tracks(rows, mode='fixed'), rows)
    section = planned['sections']['b']
    assert section['start'] != 32.25
    assert section['incoming_posthandoff_weak_bass_maximum_seconds'] < 1.


def test_curves_never_boost_layered_bands_and_mids_handoff_faster_than_highs():
    clock = shared_clock(128, 145, 128, 8000)
    n = clock['frames']
    phase = np.interp(np.arange(n), clock['output_frames'], clock['phase']) / 128
    phase[-1] = 1.
    gains, report = envelopes(n, 128, beat_phase=phase)
    for band in ('bass', 'mid', 'high'):
        assert np.max(gains[band + '_out'] + gains[band + '_in']) <= 1 + 1e-12
    assert report['incoming_mid_opening_beats'] == 16
    assert report['bass_handoff_beats'] == 16
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


@pytest.mark.parametrize('resonant_kicks', [False, True])
def test_real_compact_render_matches_solo_budget_or_refuses_a_clock_conflict(tmp_path, resonant_kicks):
    ffmpeg, rubberband = os.getenv('SUPERDJ_FFMPEG'), os.getenv('SUPERDJ_RUBBERBAND')
    if not ffmpeg or not rubberband:
        pytest.skip('Actual audio tools required')
    sr = 8000
    rows = [record('a', 320, 120), record('b', 320, 124)]
    for row in rows:
        t = np.arange(320 * sr) / sr
        audio = .07 * np.sin(2 * np.pi * 440 * t)
        if resonant_kicks:
            pulse_t=np.arange(round(.12*sr))/sr
            pulse=.18*np.sin(2*np.pi*80*pulse_t)*np.exp(-pulse_t/.035)+.06*np.random.default_rng(11).normal(size=len(pulse_t))*np.exp(-pulse_t/.003)
            for beat in row['beats']:
                first=round(beat*sr);last=min(first+len(pulse),len(audio));audio[first:last]+=pulse[:last-first]
        else:
            width=64
            pulse=.25*np.exp(-.5*(np.arange(-width,width+1)/(sr*.0015))**2)
            for beat in row['beats']:
                center=round(beat*sr);audio[center-width:center+width+1]+=pulse
        path = tmp_path / (row['id'] + '.wav')
        sf.write(path, np.column_stack((audio, audio)), sr, subtype='FLOAT')
        row['path'] = str(path)
    plan = arrange_sync(plan_tracks(rows, mode='fixed'), rows, sample_rate=sr)
    if resonant_kicks:
        # Retain the old waveform as a refusal regression: its actual blend
        # has a strong delayed bass view. Ordinary SYNC now applies the same
        # unchanged clock gate as Zhilin and must not conceal it by cutting.
        from superdj.transition_quality import TransitionFailure
        with pytest.raises(TransitionFailure,match='Combined percussion attack misses the shared clock'):
            render_plan(plan, rows, {}, tmp_path / 'set.wav', tmp_path / 'render',
                        ffmpeg=ffmpeg, rubberband=rubberband, sample_rate=sr)
        assert not (tmp_path/'set.wav').exists()
        return
    result = render_plan(plan, rows, {}, tmp_path / 'set.wav', tmp_path / 'render',
                         ffmpeg=ffmpeg, rubberband=rubberband, sample_rate=sr)
    assert result['transitions'][0]['tempo_technique'] == 'shared_phase_sync'
    assert result['transitions'][0]['frequency_processing']['phase_source'] == 'shared integrated clock'
    for track in result['tracks']:
        expected = plan['sections'][track['id']]['predicted_solo_seconds']
        assert abs(track['solo_seconds'] - expected) < .001
        assert 180 <= track['output_frames']/sr <= 240
    assert sf.info(tmp_path / 'set.wav').frames == result['total_frames']


def test_minimum_repeat_preserves_phase_voice_edit_and_original_file(tmp_path):
    from superdj.minimum import legacy_minimum as preserve_minimum
    sr = 8000
    rows = [record('a', 125, 100), record('b', 320, 126)]
    rows[0]['downbeats'] = [t for t in rows[0]['downbeats'] if t < 119]
    original = .1 * np.sin(2 * np.pi * 300 * np.arange(125 * sr) / sr)
    audio = np.column_stack((original, original)).astype(np.float32)
    source, voice = tmp_path / 'a.wav', tmp_path / 'voice.wav'
    sf.write(source, audio, sr, subtype='FLOAT'); sf.write(voice, audio * .5, sr, subtype='FLOAT')
    plan = arrange_sync(plan_tracks(rows, mode='fixed'), rows, minimum_play_seconds=120.)
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


def test_minimum_repeat_can_use_earlier_stable_body_when_exit_annotations_disagree(tmp_path):
    from superdj.minimum import legacy_minimum as preserve_minimum
    sr=8000
    rows=[record('a',125,100),record('b',320,126)]
    rows[0]['downbeats']=[t for t in rows[0]['downbeats'] if t<119]
    plan=arrange_sync(plan_tracks(rows,mode='fixed'),rows,minimum_play_seconds=120.)
    grid=_grid(rows[0],.75)
    edge=plan['edges'][0]
    end=int(np.argmin(abs(grid[3]-plan['sections']['a']['end'])))-edge['overlap_bars']
    original_insertion=float(grid[3][end])
    # Three bad predicted downbeat annotations near the exit must not discard
    # independently valid, earlier eight-beat windows inside the solo body.
    for offset in (1,2,4):rows[0]['downbeats'][end-offset]+=.6
    audio=np.column_stack([.1*np.sin(2*np.pi*300*np.arange(125*sr)/sr)]*2).astype(np.float32)
    source=tmp_path/'source.wav';sf.write(source,audio,sr,subtype='FLOAT')
    processed={'a':{'path':str(source)},'b':{'path':'unused'}}
    result,revised,prepared=preserve_minimum(plan,rows,processed,tmp_path/'repeat')
    edit=result['minimum_playback_edits'][0]
    assert edit['insertion_source_seconds']<original_insertion
    assert edit['bars_repeated']==2 and edit['added_seconds']<5
    assert result['sections']['a']['predicted_play_seconds']>=120
    assert sf.info(prepared['a']['path']).frames==len(audio)+round(edit['added_seconds']*sr)
    np.testing.assert_array_equal(sf.read(source,dtype='float32')[0],audio)
