"""Musical development preferences use measured evidence and actual search."""

import json

import numpy as np
import pytest

from superdj.analysis import SAMPLE_RATE, _features, _set_arc_evidence
from superdj.planner import DEFAULT_WEIGHTS, plan_tracks


def record(identity, bpm=124, intensity=.5, bass=None, **extra):
    return {"id":identity, "bpm":bpm, "key_camelot":"8A", "intensity":intensity,
            "bass_intensity":bass, "key_confidence":1, "bpm_confidence":1, **extra}


def only_position(feature):
    weights = {k:0 for k in DEFAULT_WEIGHTS}
    weights[feature] = 1
    return weights


def test_audio_intensity_and_bass_ignore_unclipped_gain():
    time = np.arange(4*SAMPLE_RATE) / SAMPLE_RATE
    pulse = .35 + .65 * (np.sin(2*np.pi*2*time) > .5)
    audio = (pulse * (np.sin(2*np.pi*70*time) + .3*np.sin(2*np.pi*1100*time))).astype(np.float32)
    original = _set_arc_evidence(_features(audio))
    quieter = _set_arc_evidence(_features(audio*.035))
    for field in ("intensity","energy","bass_intensity"):
        assert original[field] == pytest.approx(quieter[field], abs=2e-6)
    assert original["set_arc_evidence"]["gain_invariant"] is True
    assert original["bass_intensity"] > .85


def test_bass_proxy_measures_band_share_not_file_level():
    time = np.arange(3*SAMPLE_RATE) / SAMPLE_RATE
    low = np.sin(2*np.pi*70*time)
    mid = np.sin(2*np.pi*1100*time)
    heavy = _set_arc_evidence(_features((.1*low + .02*mid).astype(np.float32)))
    thin = _set_arc_evidence(_features((.03*low + .3*mid).astype(np.float32)))
    assert heavy["bass_intensity"] > .9
    assert thin["bass_intensity"] < .05
    assert "not kick detection" in heavy["bass_intensity_method"]


def test_density_proxy_distinguishes_equal_rms_tone_and_broadband_audio():
    time = np.arange(3*SAMPLE_RATE) / SAMPLE_RATE
    tone = np.sin(2*np.pi*440*time)
    noise = np.random.default_rng(42).normal(size=len(time))
    noise *= np.sqrt(np.mean(tone**2)/np.mean(noise**2))
    sparse = _set_arc_evidence(_features(tone.astype(np.float32)))
    dense = _set_arc_evidence(_features(noise.astype(np.float32)))
    assert np.sqrt(np.mean(tone**2)) == pytest.approx(np.sqrt(np.mean(noise**2)))
    assert dense["intensity"] > sparse["intensity"]+.2


def test_silence_provides_no_intensity_or_bass_evidence():
    evidence = _set_arc_evidence(_features(np.zeros(SAMPLE_RATE, dtype=np.float32)))
    assert evidence["intensity"] is None and evidence["bass_intensity"] is None
    assert evidence["set_arc_evidence"]["measured"] is False
    json.dumps(evidence, allow_nan=False)


def test_build_native_tempo_targets_change_the_optimised_order():
    records = [record("fast",132),record("medium",126),record("slow",120),record("rising",129)]
    planned = plan_tracks(records, energy_arc="build", weights=only_position("tempo_arc"))
    assert planned["order"] == ["slow","medium","rising","fast"]
    assert planned["tempo_targets_bpm"] == [120,124,128,132]
    assert planned["order"] != [r["id"] for r in records]
    assert planned["set_arc"]["positions"][0]["components"]["tempo"] == 0
    assert planned["set_arc"]["positions"][-1]["components"]["tempo"] == 0


def test_wave_search_climbs_dips_then_finishes_at_the_highest_tempo():
    records = [record(str(i),110+2*i) for i in range(9)]
    planned = plan_tracks(records[::-1], energy_arc="wave", weights=only_position("tempo_arc"))
    bpms = [p["bpm"] for p in planned["set_arc"]["positions"]]
    targets = planned["tempo_targets_bpm"]
    assert bpms[0] == 110 and bpms[-1] == 126
    assert bpms[3] > bpms[5] < bpms[-1]
    assert targets[3] > targets[5] < targets[-1]
    assert targets[-1] == max(targets)
    assert len(set(planned["order"])) == 9


def test_measured_bass_prefers_lighter_start_and_stronger_finish():
    records = [record("heavy",bass=.9),record("middle",bass=.5),record("light",bass=.1)]
    planned = plan_tracks(records, energy_arc="build", weights=only_position("bass_arc"))
    assert planned["order"] == ["light","middle","heavy"]
    assert planned["bass_targets"] == [.1,.5,.9]


def test_missing_and_legacy_level_evidence_have_zero_position_cost():
    records = [{"id":"missing"}, {"id":"old","bpm":124,"energy":.95,
               "energy_method":"RMS level proxy mapped -36..-6dBFS to 0..1; not perceived emotion"},
               record("measured",bass=.8)]
    planned = plan_tracks(records, mode="fixed")
    for p in planned["set_arc"]["positions"][:2]:
        assert p["intensity"] is None and p["bass_intensity"] is None
        assert p["components"]["intensity"] == p["components"]["bass"] == 0
    assert planned["set_arc"]["positions"][0]["components"]["tempo"] == 0
    assert any("Legacy RMS" in w for w in planned["uncertainty"]["warnings"])
    assert planned["set_arc"]["positions"][0]["intensity_method"] == "unavailable"
    assert planned["set_arc"]["positions"][0]["bass_method"] == "unavailable"


def test_named_arc_never_overrides_fixed_order_or_locked_endpoints():
    records = [record("heavy",132,.9,.9),record("light",120,.1,.1),record("middle",126,.5,.5)]
    fixed = plan_tracks(records, mode="fixed", energy_arc="wave")
    assert fixed["order"] == ["heavy","light","middle"]
    locked = plan_tracks(records, first_id="heavy",last_id="light",energy_arc="build")
    assert locked["order"][0] == "heavy" and locked["order"][-1] == "light"
    assert set(locked["order"]) == {r["id"] for r in records}


def test_custom_intensity_and_none_do_not_add_other_position_preferences():
    records = [record("a",120,.1,.2),record("b",132,.9,.8)]
    for arc in ([.1,.9], "none"):
        result = plan_tracks(records, energy_arc=arc)
        assert result["tempo_targets_bpm"] == result["bass_targets"] == []
        assert all(p["components"]["tempo"] == p["components"]["bass"] == 0
                   for p in result["set_arc"]["positions"])
