"""Audit CLI tests preserve inputs and distinguish probing from full decoding."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import soundfile as sf

from superdj.cli import archive_audit, main


def test_cli_plan_attaches_cue_locks_without_reordering(tmp_path):
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"id": "a", "bpm": 174}, {"id": "b", "bpm": 128}]), encoding="utf-8")
    overrides = tmp_path / "locks.json"
    locks = {"incoming_start_seconds": 15.04, "outgoing_end_seconds": 244.18}
    overrides.write_text(json.dumps([{"from": "a", "to": "b", "cue_locks": locks}]), encoding="utf-8")
    work = tmp_path / "private-work"
    main(["plan", "--records", str(records), "--mode", "fixed", "--work", str(work),
          "--transition-overrides", str(overrides)])
    planned = json.loads((work / "plan.json").read_text(encoding="utf-8"))
    assert planned["order"] == ["a", "b"]
    assert planned["edges"][0]["cue_locks"] == locks
    assert planned["edges"][0]["tempo"]["technique"] == "outgoing_only_vinyl"
    assert planned["constraints"]["tempo_ramp_max_change_per_deck"] is None
    overrides.write_text(json.dumps([{"from": "b", "to": "a", "cue_locks": locks}]), encoding="utf-8")
    with pytest.raises(ValueError, match="existing adjacent pair"):
        main(["plan", "--records", str(records), "--mode", "fixed", "--work", str(work),
              "--transition-overrides", str(overrides)])


def test_cli_strict_tempo_cap_refuses_wider_ramp_without_changing_order(tmp_path):
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"id": "a", "bpm": 174}, {"id": "b", "bpm": 128}]), encoding="utf-8")
    with pytest.raises(ValueError, match="strict tempo"):
        main(["plan", "--records", str(records), "--mode", "fixed", "--work", str(tmp_path / "work"),
              "--tempo-ramp-max-change", "0.06", "--no-tempo-reset"])


@pytest.mark.parametrize("bars", [4,32])
def test_cli_plan_accepts_pair_specific_length_without_changing_order(tmp_path, bars):
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"id":"a","bpm":138},{"id":"b","bpm":174}]), encoding="utf-8")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps([{"from":"a","to":"b","overlap_bars":bars}]), encoding="utf-8")
    work = tmp_path / "work"
    main(["plan","--records",str(records),"--mode","fixed","--work",str(work),
          "--transition-overrides",str(overrides)])
    planned = json.loads((work/"plan.json").read_text(encoding="utf-8"))
    assert planned["order"] == ["a","b"]
    assert planned["edges"][0]["overlap_bars"] == bars


def test_cli_plan_keeps_explicit_pair_method(tmp_path):
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"id":"a","bpm":135},{"id":"b","bpm":145}]), encoding="utf-8")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps([{"from":"a","to":"b","transition_policy":"sync","overlap_bars":32}]), encoding="utf-8")
    work = tmp_path / "work"
    main(["plan","--records",str(records),"--mode","fixed","--work",str(work),
          "--transition-overrides",str(overrides)])
    planned = json.loads((work/"plan.json").read_text(encoding="utf-8"))
    assert planned["edges"][0]["transition_policy"] == "sync"
    assert planned["edges"][0]["overlap_bars"] == 32


@pytest.mark.parametrize("bars", [True,16.0,3,64,"32"])
def test_cli_plan_rejects_invalid_pair_length(tmp_path, bars):
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"id":"a","bpm":138},{"id":"b","bpm":174}]), encoding="utf-8")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps([{"from":"a","to":"b","overlap_bars":bars}]), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap_bars"):
        main(["plan","--records",str(records),"--mode","fixed","--work",str(tmp_path/"work"),
              "--transition-overrides",str(overrides)])


def test_audit_refuses_to_replace_its_project_input(tmp_path):
    project = tmp_path / "archive_audit.json"
    project.write_bytes(b"untouched native input")
    args = SimpleNamespace(work=str(tmp_path), project=str(project), cue=None,
                           audio=None, measure=False)
    with pytest.raises(ValueError, match="must not overwrite"):
        archive_audit(args)
    assert project.read_bytes() == b"untouched native input"


def test_measurement_log_cannot_alias_audio_input(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"untouched audio input")
    try:
        os.link(audio, tmp_path / "export_qa.log")
    except OSError:
        pytest.skip("Hard links unavailable on this test filesystem")
    args = SimpleNamespace(work=str(tmp_path), project=str(tmp_path / "project.djs"),
                           cue=None, audio=str(audio), measure=True)
    with pytest.raises(ValueError, match="must not overwrite"):
        archive_audit(args)
    assert audio.read_bytes() == b"untouched audio input"


def test_audit_cli_full_measurement_on_generated_audio(tmp_path):
    ffmpeg = os.environ.get("SUPERDJ_TEST_FFMPEG") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("SUPERDJ_TEST_FFPROBE") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg and FFprobe required for actual audit decode")
    audio = tmp_path / "example.wav"
    sr = 8000
    signal = 0.1 * np.sin(2 * np.pi * 440 * np.arange(3 * sr) / sr)
    sf.write(audio, np.column_stack((signal, signal)), sr, subtype="FLOAT")
    project = tmp_path / "example.djs"
    state = {"mix": [{"libraryKey": "a", "cueData": {"systemCuePoints": [{"beat": 0}, {"beat": 10}]},
                       "mixMap": [{"trackBeatIx": i, "time": i / 2, "duration": 0.5} for i in range(11)]}],
             "audio": [{"key": "a", "name": "Alpha", "bpm": 120, "beatsPerBar": 4}],
             "autoEffects": [], "jingles": [{"opaque_future": {"value": 0.314}}], "duration": 999}
    with zipfile.ZipFile(project, "w") as archive:
        archive.writestr("mixdata.json", json.dumps(state))
    cue = tmp_path / "example.cue"
    cue.write_text('FILE "example.wav" WAVE\nTRACK 01 AUDIO\nTITLE "Alpha"\nINDEX 01 00:00:00\n', encoding="utf-8")
    paths = [audio, project, cue]
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    work = tmp_path / "private-review"
    main(["archive-audit", "--project", str(project), "--cue", str(cue), "--audio", str(audio),
          "--work", str(work), "--ffmpeg", ffmpeg, "--ffprobe", ffprobe,
          "--measure", "--include-native-state"])
    report = json.loads((work / "archive_audit.json").read_text(encoding="utf-8"))
    assert report["cue_order_matches"] is True
    assert report["audio_measurement"]["full_decode_ok"] is True
    assert report["audio_measurement"]["true_peak_dbfs"] < -10
    assert report["export_duration_s"] == pytest.approx(3)
    assert report["native_state"]["mixdata"] == state
    assert report["warnings"]
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == hashes
