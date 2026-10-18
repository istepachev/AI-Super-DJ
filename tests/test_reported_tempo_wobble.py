"""Anonymous timing regressions measured in two reported musical transitions.

Only numeric frame steps and native offsets are retained; no recordings,
identities, local paths, fitted audio or listening preferences are bundled.
"""
from copy import deepcopy

import numpy as np
import pytest

from superdj.timing_safety import (
    audit_warp_rates,
    classify_native_phase,
    guarded_anchor_edit,
    initialize_rate_reference,
)


@pytest.mark.parametrize(
    'source_step,output_step,stale_change',
    [
        (3992, 5448, .056909057095034354),
        (3923, 5331, .056909057095034354),
        (3969, 5327, .057345607978519286),
        (6617, 5231, .057345607978519286),
        (3976, 5348, .06490872210953347),
        (6602, 5297, .06490872210953347),
    ],
    ids=['first-window-fill', 'first-window-return', 'second-window-slow',
         'second-window-fast', 'late-incoming-slow', 'late-incoming-fast'],
)
def test_reported_short_speed_pulses_cannot_pass_with_stale_statistics(
    source_step, output_step, stale_change,
):
    sr = 44100
    # A correct overall duration can conceal opposite local speed changes.
    warp = {
        'anchors': [{'source_frame': i * output_step, 'output_frame': i * output_step}
                    for i in range(3)],
        'input_output_beat_mapping': {'head': [], 'tail': []},
    }
    initialize_rate_reference(warp, sr)
    warp['anchors'][1]['source_frame'] = source_step
    warp['max_local_warp_rate_change'] = stale_change

    with pytest.raises(ValueError, match='Actual acoustic timing change'):
        audit_warp_rates(warp, sr, stage='reported musical regression')

    measured = warp['actual_timing_rate_audit']
    assert not measured['passed']
    assert measured['maximum_timing_correction_change'] > .20
    assert measured['minimum_actual_playback_rate'] == pytest.approx(
        min(source_step / output_step, 2 - source_step / output_step))
    assert measured['maximum_actual_playback_rate'] == pytest.approx(
        max(source_step / output_step, 2 - source_step / output_step))
    assert warp['anchors'][0] == {'source_frame': 0, 'output_frame': 0}
    assert warp['anchors'][-1]['source_frame'] == warp['anchors'][-1]['output_frame']


@pytest.mark.parametrize('displacement', [-.12, .12])
def test_unsafe_local_correction_restores_the_previously_working_map(displacement):
    sr = 44100
    warp = {
        'anchors': [{'source_frame': i * sr // 2, 'output_frame': i * sr // 2}
                    for i in range(10)],
        'input_output_beat_mapping': {'head': [], 'tail': []},
    }
    initialize_rate_reference(warp, sr)
    original = deepcopy(warp['anchors'])

    def unsafe_edit():
        warp['anchors'][4]['source_frame'] += round(displacement * sr)

    with pytest.raises(ValueError, match='Actual acoustic timing change'):
        guarded_anchor_edit(warp, sr, unsafe_edit, stage='reported fill correction')

    assert warp['anchors'] == original
    assert warp['actual_timing_rate_audit']['passed']
    assert warp['actual_timing_rate_audit']['maximum_timing_correction_change'] == 0
    assert not warp['rejected_timing_rate_audits'][-1]['passed']


@pytest.mark.parametrize(
    'reference_offsets,displaced_offsets',
    [
        ([-.012, .016875, -.002125, .012, -.012125, -.0305, .012, -.012],
         [.11475, -.11825, -.112125, -.11025]),
        ([-.011, -.0075, -.013625, -.014, -.01275, -.01725, -.013875, -.014375],
         [.10975, .10975, .109625, .10975]),
        ([-.023125, -.003625, -.024875, -.0035, -.004, -.023125, -.003625, -.004],
         [.095125, .115125]),
    ],
    ids=['first-outgoing-native-figures', 'second-outgoing-native-figures',
         'late-incoming-native-figures'],
)
def test_reported_native_fills_remain_at_source_time_instead_of_pulling_the_clock(
    reference_offsets, displaced_offsets,
):
    centres = .2 + np.arange(len(reference_offsets) + len(displaced_offsets)) * 60 / 125
    references = [{'index': i, 'time': float(centres[i] + offset)}
                  for i, offset in enumerate(reference_offsets)]
    fills = [{'index': i + len(references),
              'time': float(centres[i + len(references)] + offset),
              'source_template_correlation': .999}
             for i, offset in enumerate(displaced_offsets)]
    observed = deepcopy(references + fills)

    accepted, displaced = classify_native_phase(observed, centres, references)

    assert {row['index'] for row in accepted} == {row['index'] for row in references}
    assert {row['index'] for row in displaced} == {row['index'] for row in fills}
    assert [row['time'] for row in displaced] == [row['time'] for row in fills]
    assert observed == references + fills


def test_a_stable_shared_tempo_and_small_global_phase_offset_stay_unchanged():
    # Retain a successful straight 125 BPM map, including a global phase bias.
    sr = 44100
    step = round(sr * 60 / 125 / 4)
    warp = {
        'anchors': [{'source_frame': i * step, 'output_frame': i * step}
                    for i in range(65)],
        'input_output_beat_mapping': {'head': [], 'tail': []},
    }
    before = deepcopy(warp['anchors'])
    initialize_rate_reference(warp, sr)
    assert audit_warp_rates(warp, sr, stage='working constant-tempo control')['passed']
    assert warp['anchors'] == before
    assert warp['actual_timing_rate_audit']['maximum_timing_correction_change'] == 0
    centres = .2 + np.arange(32) * 60 / 125
    references = [{'index': i, 'time': float(stamp - .032125)}
                  for i, stamp in enumerate(centres)]
    accepted, displaced = classify_native_phase(deepcopy(references), centres, references)
    assert accepted == references and not displaced
