"""Check confidence boundaries and imported timing, not one implementation trace."""

import numpy as np
import pytest

from superdj.analysis import _imported_grid, _times, _grid_diagnostics, _cue_candidates, _decode_meter, _robust_tempo, prepare_beat_model


def test_imported_downbeats_must_align_to_beats():
    with pytest.raises(ValueError, match="downbeat"):
        _imported_grid({"beats": [0, 0.5, 1, 1.5, 2], "downbeats": [0.25, 1.25]}, 3)


def test_constant_grid_uses_explicit_anchor_instead_of_first_detected_beat():
    beats, downbeats, info = _imported_grid({"bpm": 120, "first_downbeat_s": 1.25, "meter": 4,
                                            "reviewed": True, "source": "human review"}, 10)
    assert 1.25 in downbeats
    assert 0.25 in beats and 0.25 not in downbeats
    assert np.allclose(np.diff(downbeats), 2.0)
    assert info["status"] == "reviewed"


def test_machine_import_is_not_automatically_reviewed():
    _, _, info = _imported_grid({"beats": [0, 0.5, 1, 1.5, 2], "downbeats": [0, 2]}, 3)
    assert info["status"] == "imported_unreviewed"
    assert info["human_reviewed"] is False


@pytest.mark.parametrize("values", [[1, 1], [2, 1], [float("nan")], [-1], [11]])
def test_invalid_grid_times_rejected(values):
    with pytest.raises(ValueError):
        _times(values, 10, "beats")


def test_model_download_requires_opt_in(tmp_path):
    with pytest.raises(FileNotFoundError, match="Explicitly"):
        prepare_beat_model("small0", cache_dir=tmp_path)


def test_regular_halfbeat_offset_does_not_establish_a_bar_grid():
    times = np.arange(0, 10, 0.025)
    features = {"times": times, "flux": np.ones(len(times))}
    result = _grid_diagnostics(np.arange(0, 10, 0.5), np.arange(0.25, 10, 2), features)
    assert result["meter"] is None
    assert result["downbeat_alignment_max_seconds"] == pytest.approx(0.25)
    assert result["confidence"] < 0.75


def test_ambiguous_meter_cannot_offer_8_16_32_bar_claims():
    times = np.arange(0, 100, 0.025)
    features = {"times": times, "flux": np.ones(len(times)), "rms": np.ones(len(times)),
                "sustained_mid": np.ones(len(times)) * 0.4}
    candidates = _cue_candidates(np.arange(0, 100, 0.5), np.arange(0, 100, 2), features, 100,
                                 "model_predicted", {"confidence": 0.7, "meter": None})
    assert candidates
    assert all(item["phrase_lengths_bars"] == [] for item in candidates)
    assert all(item["status"] == "estimated_boundary" for item in candidates)


@pytest.mark.parametrize("meter,phase", [(3, 1), (4, 2)])
def test_neural_meter_decoder_infers_phase_from_evidence(meter, phase):
    probability = np.full(96, 0.07)
    probability[phase::meter] = 0.95
    indices, info = _decode_meter(probability)
    assert info["accepted"] is True
    assert info["meter"] == meter
    assert np.array_equal(indices, np.arange(phase, 96, meter))
    assert indices[0] != 0


@pytest.mark.parametrize("probability", [0.1, 0.5, 0.9])
def test_uniform_neural_evidence_cannot_establish_meter(probability):
    indices, info = _decode_meter(np.full(96, probability))
    assert indices is None
    assert info["accepted"] is False


def test_tempo_refinement_removes_frame_quantisation_bias():
    beats = np.round(np.arange(300) * 60 / 130 / 0.02) * 0.02
    result = _robust_tempo(beats)
    assert result["bpm"] == pytest.approx(130, abs=0.02)
    assert result["tempo_inlier_fraction"] == 1.0


def test_tempo_refinement_excludes_a_missing_beat_without_octave_guessing():
    beats = np.arange(300) * 60 / 128
    beats = np.delete(beats, [25, 100, 200])
    result = _robust_tempo(beats)
    assert result["bpm"] == pytest.approx(128, abs=0.001)
    assert result["tempo_inlier_fraction"] < 1.0


def test_incoherent_alternating_beats_return_unknown_tempo():
    beats = np.concatenate([[0], np.cumsum(np.tile([0.2, 0.8], 20))])
    times = np.arange(0, 25, 0.025)
    result = _grid_diagnostics(beats, np.array([]), {"times": times, "flux": np.ones(len(times))})
    assert result["bpm"] is None
    assert result["confidence"] == 0
