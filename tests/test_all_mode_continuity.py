"""Ordinary public style/order choices cannot opt out of musical continuity."""
import numpy as np
import pytest

from superdj.defaults import planning_defaults
from superdj.mixing import blend, envelopes
from superdj.percussion import needs_handoff, percussion_gains, verification_components


@pytest.mark.parametrize('requested_style', ['zhilin', 'sync', 'default'])
@pytest.mark.parametrize('harmonic_mode', ['mood', 'classic', 'fuzzy'])
def test_every_public_mode_keeps_gradual_actual_rhythm_routing(requested_style, harmonic_mode):
    preferences = planning_defaults(style=requested_style, harmonic_mode=harmonic_mode, runtime={})
    style = preferences['style']
    sr = 8000
    t = np.arange(sr * 6) / sr
    phase = np.linspace(0, 1, len(t)) ** 1.15
    a = np.column_stack([.1 * np.sin(2 * np.pi * 90 * t)] * 2).astype('float32')
    b = np.column_stack([.1 * np.sin(2 * np.pi * 130 * t)] * 2).astype('float32')
    uncertain = {'source_kick_alignment': {'head': {'unverified_native_bass_candidates': [{'source_seconds': 5.}]}}}
    assert needs_handoff({}, uncertain, style=style)
    for beats in (8, 32, 128):
        for estimates in ({}, {'drums_out': a * .7, 'drums_in': b * .6}):
            policy = 'smooth' if estimates else 'shared'
            mixed, report = blend(a, b, sr, beats=beats, beat_phase=phase, style=style,
                                  percussion_policy=policy, **estimates)
            check = report['musical_handoff_continuity']
            assert check['passed'] and not check['abrupt_percussion_explicitly_allowed']
            assert check['effective_10_90_shared_beats']['bass'] > 2
            assert report['bass_handoff_beats'] >= min(beats, 16 if style == 'zhilin' else 8)
            base, _ = envelopes(len(t), beats, beat_phase=phase, style=style)
            gains = percussion_gains(base, phase, beats, policy)
            parts = verification_components(a, b, sr, gains, **estimates)
            np.testing.assert_allclose(parts['tail'] + parts['head'], mixed, atol=3e-8)
            assert report['endpoint_max_error'] < 1e-7
        with pytest.raises(ValueError, match='Percussion handoff is too short'):
            blend(a, b, sr, beats=beats, beat_phase=phase, style=style,
                  percussion_policy='sequential', drums_out=a * .7, drums_in=b * .6)
