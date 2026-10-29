import numpy as np
import pytest
from superdj.mixing import blend, recorded_envelopes
from superdj.blend_recovery import retain_middle_continuity, CONTINUITY_REFUSAL
from superdj.percussion import percussion_gains, verification_components


def independent_parts():
    sr = 8000
    t = np.arange(12 * sr) / sr
    a = np.repeat((.06 * np.sin(2 * np.pi * 430 * t))[:, None], 2, axis=1)
    b = np.repeat((.06 * np.sin(2 * np.pi * 2100 * t))[:, None], 2, axis=1)
    zero = np.zeros_like(a)
    return sr, a, b, dict(style='zhilin', drums_out=zero, drums_in=zero,
        percussion_policy='smooth', vocal_out=zero, vocal_in=zero)


def test_short_selective_transfer_retains_exact_passing_power_pcm_and_replay():
    sr, a, b, options = independent_parts()
    before = a.copy(), b.copy()
    with pytest.raises(ValueError, match='Accompaniment energy handoff is too short'):
        blend.__wrapped__(a, b, sr, beats=32, **options)
    mixed, report = blend(a, b, sr, beats=32, **options)
    expected, _ = blend.__wrapped__(a, b, sr, beats=32, mid_energy_mode='power', **options)
    np.testing.assert_array_equal(mixed, expected)
    assert report['mid_continuity_recovery']['selected_mode'] == 'power'
    assert report['musical_handoff_continuity']['passed']
    assert 'quality_gate' not in report and 'blended_front_audit' not in report
    gains, _ = recorded_envelopes(len(a), 32, {'frequency_processing': report}, style='zhilin', beat_phase=None)
    gains = percussion_gains(gains, None, 32, 'smooth')
    parts = verification_components(a, b, sr, gains, drums_out=options['drums_out'], drums_in=options['drums_in'])
    np.testing.assert_allclose(parts['head'] + parts['tail'], mixed, atol=2e-8)
    np.testing.assert_array_equal(a, before[0]); np.testing.assert_array_equal(b, before[1])


@pytest.mark.parametrize('mode', ['auto', 'power', 'legacy'])
def test_successful_existing_variants_are_bit_identical(mode):
    sr, a, b, options = independent_parts()
    actual = blend(a, b, sr, beats=128, mid_energy_mode=mode, **options)
    old = blend.__wrapped__(a, b, sr, beats=128, mid_energy_mode=mode, **options)
    np.testing.assert_array_equal(actual[0], old[0])
    assert actual[1] == old[1] and 'mid_continuity_recovery' not in actual[1]


def test_all_rejected_curves_remain_rejected_with_unchanged_limits():
    sr, a, b, options = independent_parts()
    with pytest.raises(ValueError, match='Accompaniment energy handoff is too short'):
        blend(a, b, sr, beats=32, mid_handoff_bars=.125, **options)


@pytest.mark.parametrize('options', [{'mid_energy_mode': 'power'}, {'mid_energy_mode': 'legacy'},
    {'whole_spectrum_handoff': {'start_beat': 4, 'end_beat': 36}}])
def test_explicit_middle_or_whole_spectrum_settings_do_not_change(options):
    calls = []
    @retain_middle_continuity
    def fail(*args, **kwargs):
        calls.append(kwargs); raise ValueError(CONTINUITY_REFUSAL + ' rejected')
    with pytest.raises(ValueError):fail('a', 'b', 8000, **options)
    assert calls == [options]


def test_legacy_remains_available_after_early_power_refusal():
    seen = []; a, b = object(), object(); clock = object(); drums = object()
    @retain_middle_continuity
    def route(*args, **kwargs):
        assert args == (a, b, 8000) and kwargs['beat_phase'] is clock and kwargs['drums_out'] is drums
        mode = kwargs.get('mid_energy_mode', 'auto'); seen.append(mode)
        if mode != 'legacy':raise ValueError(CONTINUITY_REFUSAL + mode)
        return a, {'musical_handoff_continuity': {'passed': True}}
    result = route(a, b, 8000, beat_phase=clock, drums_out=drums)
    assert seen == ['auto', 'power', 'legacy'] and result[0] is a
    assert result[1]['mid_continuity_recovery']['selected_mode'] == 'legacy'


def test_unrelated_error_during_fallback_cannot_be_hidden():
    seen = []
    @retain_middle_continuity
    def route(**kwargs):
        mode = kwargs.get('mid_energy_mode', 'auto'); seen.append(mode)
        raise ValueError(CONTINUITY_REFUSAL if mode == 'auto' else 'invalid prepared audio')
    with pytest.raises(ValueError, match='invalid prepared audio'):route()
    assert seen == ['auto', 'power']


def test_quality_recovery_reaches_legacy_after_early_power_continuity_failure(tmp_path, monkeypatch):
    from superdj import orchestration as module
    from superdj.transition_quality import TransitionFailure
    sr, a, b, args = independent_parts(); seen = []; checked = []
    _, frequency = blend(a, b, sr, beats=128, **args)
    assert frequency['selective_mid_energy']['applied']
    original = module.blend
    def routed(*params, **options):
        mode = options['mid_energy_mode']; seen.append(mode)
        if mode == 'power':raise ValueError(CONTINUITY_REFUSAL + ' power')
        return original(*params, **options)
    def fronts(pcm, *args, **options):
        np.testing.assert_allclose(options['components']['head'] + options['components']['tail'], pcm, atol=2e-8)
        return pcm, {'verified': True}
    def quality(pcm, candidate):
        checked.append(candidate['mid_energy_mode']); return {'passed': True}
    monkeypatch.setattr(module, 'blend', routed); monkeypatch.setattr(module, 'verify_blended_fronts', fronts)
    _, result = module.legacy_mid_alternative(a, b, sr, {}, {},
        {'beats': 128, 'frequency_processing': frequency}, tmp_path, style='zhilin', beat_phase=None,
        percussion_args={k: args[k] for k in ('drums_out', 'drums_in', 'percussion_policy')},
        mid_energy_args={k: args[k] for k in ('vocal_out', 'vocal_in')},
        initial_error=TransitionFailure('a', 'b', 'local coverage refusal'), quality_check=quality)
    assert seen == ['power', 'legacy'] and checked == ['legacy']
    assert result['quality_gate']['passed'] and result['blended_front_audit']['verified']
