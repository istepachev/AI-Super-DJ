"""Musical-policy constraints and observable signal/format preservation."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf

from superdj.cues import hybrid_candidates
from superdj.mixing import blend, split_bands
from superdj.project_edit import rewrite_djs
from superdj.render import _select_window, _grid, render_plan
from superdj.sections import arrange
from superdj.sync import shared_clock
from superdj.planner import plan_tracks
from superdj.make import collect
from superdj.tone import settings
from superdj.stems import handoff


def record(identity, duration=360, bpm=120):
    beats = np.arange(.25, duration - .05, 60 / bpm)
    return {"id": identity, "path": identity + ".wav", "duration": duration, "bpm": bpm,
            "beats": beats.tolist(), "downbeats": beats[::4].tolist(), "grid_confidence": 1.,
            "grid_status": "reviewed", "grid": {"meter": 4, "human_reviewed": True},
            "cue_candidates": [{"time": float(t), "kind": "body", "novelty": .8, "confidence": .9,
                                "sustained_midband_proxy": .1} for t in beats[::32]]}


def test_independent_bands_reconstruct_and_soften_entry_without_renormalizing():
    sr = 16000
    t = np.arange(4 * sr) / sr
    signal = .1 * (np.sin(2 * np.pi * 70 * t) + np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 5000 * t))
    audio = np.column_stack((signal, signal)).astype(np.float32)
    bands = split_bands(audio, sr)
    np.testing.assert_allclose(sum(bands), audio, atol=1e-12)
    result, evidence = blend(np.zeros_like(audio), audio, sr, beats=160, tonal_conflict=True)
    np.testing.assert_allclose(result[0], 0, atol=1e-8)
    np.testing.assert_allclose(result[-1], audio[-1], atol=1e-8)
    assert evidence["entry_gain_compensated"] is False
    assert evidence["incoming_initial_attenuation_db"] == 2
    assert np.isfinite(result).all()


def test_estimated_voice_handoff_preserves_accompaniment_and_endpoints():
    sr = 8000
    t = np.arange(sr * 4) / sr
    backing = np.column_stack([.1 * np.sin(2 * np.pi * 70 * t)] * 2)
    voice_a = np.column_stack([.12 * np.sin(2 * np.pi * 600 * t)] * 2)
    voice_b = np.column_stack([.12 * np.sin(2 * np.pi * 1000 * t)] * 2)
    a, b, report = handoff(backing + voice_a, backing + voice_b, voice_a, voice_b)
    assert report["executed"]
    centre = slice(int(len(t) * .46), int(len(t) * .54))
    np.testing.assert_allclose(a[centre], backing[centre], atol=1e-8)
    np.testing.assert_allclose(b[centre], backing[centre], atol=1e-8)
    np.testing.assert_allclose(a[0], (backing + voice_a)[0], atol=1e-8)
    np.testing.assert_allclose(b[-1], (backing + voice_b)[-1], atol=1e-8)
    assert report["vocal_overlap_certified_absent"] is False


def test_tone_control_bypasses_neutral_and_bounds_muddy_correction():
    sr = 16000
    t = np.arange(sr * 3) / sr
    neutral = np.column_stack([.1 * np.sin(2 * np.pi * 1000 * t)] * 2).T
    muddy = np.column_stack([.1 * np.sin(2 * np.pi * 300 * t)] * 2).T
    assert settings(neutral, sr)[0] == {}
    parameters, evidence = settings(muddy, sr)
    assert -.8 <= parameters["peak_db"] < 0
    assert evidence["lowmid_fraction"] > .55


def test_imported_cue_is_preserved_even_when_model_disagrees():
    cue = {"time": 31.25, "method": "imported cue", "reviewed": True}
    assert hybrid_candidates([cue], [0, 2, 4], {}, 100, {"candidates": [{"time": 2}]}) == [cue]


def test_model_evidence_cannot_move_structure_off_bar_grid():
    times = np.arange(0, 100, .1)
    rms = np.where(times < 40, .05, .4)
    features = {"times": times, "rms": rms, "bass_rms": rms, "high_rms": rms,
                "sustained_mid": np.ones(len(times)) * .1}
    cues = hybrid_candidates([], np.arange(0, 100, 2), features, 100,
        {"candidates": [{"time": 40.3, "relative_score": .95, "raw_score": .6}]})
    supported = [cue for cue in cues if cue["cue_detr_evidence"]]
    assert supported and supported[0]["time"] == 40
    assert supported[0]["musical_phrase_verified"] is False
    reviewed = {"time": 40.0, "method": "imported cue", "reviewed": True, "confidence": 1.0}
    mixed = hybrid_candidates([reviewed, {"time": 80.0, "confidence": .5}], np.arange(0, 100, 2), features, 100,
        {"candidates": [{"time": 40.3, "relative_score": .95, "raw_score": .6}]})
    assert next(cue for cue in mixed if cue["time"] == 40) == reviewed


def test_zhilin_long_sections_and_manual_boundaries_survive():
    records = [record("a"), record("b")]
    original = plan_tracks(records, mode="fixed")
    original["edges"][0]["cue_locks"] = {"outgoing_end_seconds": 240.25, "incoming_start_seconds": 32.25}
    result = arrange(original, records)
    assert result["edges"][0]["cue_locks"] == original["edges"][0]["cue_locks"]
    assert result["edges"][0]["section_overlap_bars"] == 40
    assert all(180 <= section["source_span_seconds"] <= 240 for section in result["sections"].values())
    assert "sections" not in original


def test_native_51_beat_exception_has_shared_phase_and_exact_locked_beat():
    item = record("a")
    cue = item["beats"][125]
    window = _select_window(item, _grid(item, .75), region="tail", count=12.75, beat_count=51,
        locked_cue_seconds=cue, target_bpm=120, sample_rate=8000, max_trim_seconds=400,
        max_local_warp_change=.15)
    assert window.times[-1] == cue and len(window.times) == 52
    clock = shared_clock(120, 125, 51, 8000)
    assert clock["beat_output_frames"][-1] == clock["frames"]
    assert len(clock["beat_output_frames"]) == 52


def test_approved_short_section_outranks_duration_preference():
    records = [record("a"), record("b")]
    planned = plan_tracks(records, mode="fixed")
    planned["edges"][0]["cue_locks"] = {"outgoing_end_seconds": 80.25, "incoming_start_seconds": 300.25}
    result = arrange(planned, records)
    assert result["sections"]["a"]["end"] == 80.25
    assert result["sections"]["b"]["start"] == 300.25
    assert len(result["section_warnings"]) == 2


def test_native_partial_bar_lock_cannot_bypass_trim_limit():
    item = record("a")
    with pytest.raises(ValueError, match="trim limits"):
        _select_window(item, _grid(item, .75), region="head", count=12.75, beat_count=51,
            locked_cue_seconds=item["beats"][65], target_bpm=120, sample_rate=8000,
            max_trim_seconds=10, max_local_warp_change=.15)


def test_preserving_project_patch_retains_audio_and_opaque_fields(tmp_path):
    source, output = tmp_path / "source.djs", tmp_path / "new.djs"
    data = {"mix": [{"cueData": {"systemCuePoints": [{"beat": 1}], "opaque": True}, "future": {"x": .5}}],
            "autoEffects": [{"duration": 64, "future": "keep"}], "opaque": [1, 2], "audio": []}
    audio = b"unchanged embedded original" * 1024
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("mixdata.json", json.dumps(data))
        archive.writestr("audio/original.wav", audio)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rewrite_djs(source, output, cue_updates={0: {"systemCuePoints": [{"beat": 2}]}},
                transition_updates={0: {"duration": 51}})
    with zipfile.ZipFile(output) as archive:
        assert archive.read("audio/original.wav") == audio
        changed = json.loads(archive.read("mixdata.json"))
        assert changed["mix"][0]["future"] == data["mix"][0]["future"]
        assert changed["mix"][0]["cueData"]["opaque"] is True
        assert changed["autoEffects"][0]["future"] == "keep"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash


def test_simple_input_collection_never_silently_drops_or_duplicates(tmp_path):
    (tmp_path / "a.flac").touch(); (tmp_path / "b.mp3").touch()
    assert len(collect([tmp_path])) == 2
    with pytest.raises(ValueError, match="unique"):
        collect([tmp_path, tmp_path / "a.flac"])


@pytest.mark.parametrize("native_beats", [None, 51])
def test_real_zhilin_long_render_preserves_three_minute_sections(tmp_path, native_beats):
    ffmpeg, rubberband = os.getenv("SUPERDJ_FFMPEG"), os.getenv("SUPERDJ_RUBBERBAND")
    if not ffmpeg or not rubberband:
        pytest.skip("Actual audio tools required")
    sr = 8000
    records = [record("a", 300, 120), record("b", 300, 122)]
    for row in records:
        samples = .12 * np.sin(2 * np.pi * 440 * np.arange(300 * sr) / sr)
        path = tmp_path / f"{row['id']}.wav"
        sf.write(path, np.column_stack((samples, samples)), sr, subtype="FLOAT")
        row["path"] = str(path)
    planned = arrange(plan_tracks(records, mode="fixed"), records)
    if native_beats:
        for row in records:
            start, end = row["beats"][65], row["beats"][485]
            planned["sections"][row["id"]].update(start=start, end=end)
        edge = planned["edges"][0]
        edge.update(overlap_beats=native_beats, transition_policy="sync",
            cue_locks={"outgoing_end_seconds": records[0]["beats"][485],
                       "incoming_start_seconds": records[1]["beats"][65]})
        edge.pop("section_overlap_bars", None)
    result = render_plan(planned, records, {}, tmp_path / "set.wav", tmp_path / "render",
        ffmpeg=ffmpeg, rubberband=rubberband, sample_rate=sr)
    assert result["transitions"][0]["bars"] == (12.75 if native_beats else 40)
    assert not result["transitions"][0].get("fallback_from_ordinary", False)
    assert all(180 <= track["duration_seconds"] <= 240 for track in result["tracks"])
    assert result["transitions"][0]["frequency_processing"]["entry_gain_compensated"] is False
    assert sf.info(tmp_path / "set.wav").frames == result["total_frames"]
