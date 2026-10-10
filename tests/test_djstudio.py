"""Generated native fixtures; no user music or project metadata in the repo."""

import copy
import json
import zipfile

import pytest

from superdj.djstudio import (DJStudioError, DJStudioProject, audit_djstudio,
                             cue_timestamp, load_djs, parse_cue_text, read_cue)


def project_data():
    def track(key):
        return {"libraryKey": key,
                "cueData": {"systemCuePoints": [{"beat": 0}, {"beat": 200}],
                            "hotCuePoints": [{"name": "keep this phrase", "beat": 120}]},
                "mixMap": [{"time": i, "duration": 1, "trackBeatIx": i} for i in range(300)],
                "gridBpm": 120, "mixBpm": -1, "phraseData": [],
                "automations": [{"arbitrary": [1, 2, {"future": "native"}]}]}

    return {"name": "Generated example", "mix": [track("a"), track("b")],
            "audio": [{"key": "a", "name": "Alpha", "bpm": 120, "beatsPerBar": 4},
                      {"key": "b", "name": "Beta", "bpm": 180, "beatsPerBar": 4}],
            "trackCount": 2, "duration": 999, "minBpm": 120, "maxBpm": 180,
            "autoEffects": [{"mixTrackKey": "0", "mixTrackAnchor": 1, "duration": 32,
                             "effectOffset": 0, "tempoMultiplier": 1, "bpmLineMode": 0,
                             "isDisabled": False, "isTrackEffect": False,
                             "primaryAutomation": [{"controlName": "eq-low", "points": [
                                 {"time": 0.0, "value": 1.0, "slope": 0.314},
                                 {"time": 32, "value": 0.04, "future_flag": "preserve"}]}],
                             "secondaryAutomation": [{"controlName": "volume", "points": [
                                 {"time": 0, "value": -24}, {"time": 32, "value": 0}]}]}],
            "jingles": [{"key": "sample", "title": "Callback", "mixData": {"custom": [0.33]}}]}


def cue_text(second="Beta", at="01:40:00"):
    return f'''TITLE "Example"
FILE "example.mp3" WAVE
TRACK 01 AUDIO
  TITLE "Alpha"
  INDEX 01 00:00:00
TRACK 02 AUDIO
  TITLE "{second}"
  INDEX 01 {at}
'''


def write_djs(path, data=None, extras=None):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mixdata.json", json.dumps(data or project_data()))
        archive.writestr("settings.json", json.dumps({"future_native_setting": [1, 2]}))
        for name, content in (extras or {"audioV2/example.flac": b"not real audio"}).items():
            archive.writestr(name, content)
    return path


@pytest.mark.parametrize("timestamp,expected", [("00:00:74", 74 / 75),
                                               ("61:02:35", 61 * 60 + 2 + 35 / 75),
                                               ("123:59:00", 123 * 60 + 59)])
def test_cue_uses_75_frames_and_unbounded_minutes(timestamp, expected):
    assert cue_timestamp(timestamp) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["01:60:00", "01:59:75", "-1:00:00", "1:1:1", "01:02"])
def test_bad_cue_timestamp_rejected(value):
    with pytest.raises(DJStudioError):
        cue_timestamp(value)


def test_mp3_declared_wave_is_not_assumed_pcm():
    cue = parse_cue_text(cue_text(at="101:00:35"))
    assert cue["files"] == [{"name": "example.mp3", "declared_type": "WAVE"}]
    assert cue["tracks"][1]["index01_s"] == pytest.approx(6060 + 35 / 75)


def test_metadata_reader_does_not_read_embedded_audio(tmp_path, monkeypatch):
    path = write_djs(tmp_path / "example.djs")
    opened = []
    original = zipfile.ZipFile.open

    def record_open(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", record_open)
    loaded = load_djs(path)
    assert opened == ["mixdata.json", "settings.json"]
    assert loaded.metadata_sha256 and "audioV2/example.flac" in loaded.archive_members


def test_archive_paths_and_metadata_limit_rejected(tmp_path):
    path = write_djs(tmp_path / "unsafe.djs", extras={"../outside.flac": b"x"})
    with pytest.raises(DJStudioError, match="Unsafe"):
        load_djs(path)
    path = write_djs(tmp_path / "large.djs")
    with pytest.raises(DJStudioError, match="oversized"):
        load_djs(path, max_metadata_bytes=10)


def test_duplicate_metadata_cannot_silently_override(tmp_path):
    path = write_djs(tmp_path / "duplicate.djs")
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("mixdata.json", "{}")
    with pytest.raises(DJStudioError, match="exactly one"):
        load_djs(path)


def test_unknown_cue_directives_are_inert_and_preserved():
    cue = parse_cue_text(cue_text() + 'REM EXECUTE "do not execute this text"\n')
    assert cue["unknown_directives"][0]["text"].startswith("REM EXECUTE")


def test_missing_index_nonmonotone_or_duplicate_tracks_rejected():
    for malformed in (cue_text().replace("INDEX 01 00:00:00", "INDEX 00 00:00:00"),
                      cue_text(at="00:00:00"), cue_text().replace("TRACK 02", "TRACK 01")):
        with pytest.raises(DJStudioError):
            parse_cue_text(malformed)


def test_file_local_positions_can_restart_and_do_not_become_one_timeline():
    text = cue_text() + 'FILE "second.mp3" MP3\nTRACK 03 AUDIO\nTITLE "Gamma"\nINDEX 01 00:00:00\n'
    cue = parse_cue_text(text)
    data = project_data()
    data["mix"].append(copy.deepcopy(data["mix"][1]))
    data["mix"][-1]["libraryKey"] = "c"
    data["audio"].append({"key": "c", "name": "Gamma", "bpm": 130})
    report = audit_djstudio(DJStudioProject(data, {}), cue=cue)
    assert report["cue_order_matches"]
    assert not report["cue_timing_usable"]
    assert all(t["playback_duration_bounds_s"] is None for t in report["tracks"])


def test_native_curves_cues_jingles_and_unknown_fields_preserved(tmp_path):
    path = write_djs(tmp_path / "example.djs")
    before = path.read_bytes()
    project = load_djs(path)
    report = audit_djstudio(project, cue=parse_cue_text(cue_text()), include_native_state=True)
    assert report["native_state"]["mixdata"] == project.mixdata
    assert report["transitions"][0]["native_effect"] == project.mixdata["autoEffects"][0]
    assert report["jingles"] == project.mixdata["jingles"]
    report["native_state"]["mixdata"]["mix"][0]["automations"][0]["arbitrary"][0] = 99
    assert project.mixdata["mix"][0]["automations"][0]["arbitrary"][0] == 1
    assert path.read_bytes() == before
    json.dumps(report, allow_nan=False)


def test_source_span_is_not_actual_playback_span_and_cue_intervals_exclude_overlap():
    report = audit_djstudio(DJStudioProject(project_data(), {}), cue=parse_cue_text(cue_text()),
                            export_duration_s=250)
    first = report["tracks"][0]
    assert first["source_selected_span_s"] == 232
    assert first["export_chapter_interval_s"] == 100
    assert first["playback_duration_s"] is None
    assert first["playback_duration_bounds_s"] == pytest.approx([100 + 32 / 3, 116])
    assert first["minimum_duration"]["status"] == "below_minimum_under_timing_assumptions"
    assert first["minimum_duration"]["audibility_verified"] is False
    assert report["transitions"][0]["bars"] == 8
    assert report["transitions"][0]["export_end_bounds_s"] == pytest.approx([100 + 32 / 3, 116])
    assert report["tracks"][1]["playback_duration_s"] == 150
    assert report["project_duration_metadata_s"] == 999
    assert report["export_duration_s"] == 250


def test_phrase_data_or_bar_number_does_not_establish_reviewed_phrasing():
    data = project_data()
    data["mix"][0]["phraseData"] = [{"beat": 32, "kind": "model_guess"}]
    report = audit_djstudio(DJStudioProject(data, {}))
    assert report["tracks"][0]["phrase_metadata_present"]
    assert not report["tracks"][0]["phrase_verified"]
    assert not report["transitions"][0]["phrase_verified"]


def test_native_arbitrary_transition_length_is_retained_without_4_beat_assumption():
    data = project_data()
    data["autoEffects"][0]["duration"] = 176
    report = audit_djstudio(DJStudioProject(data, {}))
    assert report["transitions"][0]["bars"] == 44
    for audio in data["audio"]:
        audio["beatsPerBar"] = 3
    report = audit_djstudio(DJStudioProject(data, {}))
    assert report["transitions"][0]["bars"] == pytest.approx(176 / 3)
    data["audio"][0].pop("beatsPerBar")
    report = audit_djstudio(DJStudioProject(data, {}))
    assert report["transitions"][0]["bars"] is None


def test_source_end_past_file_is_reported_not_counted_as_real_music():
    data = project_data()
    data["audio"][0]["duration"] = 210
    report = audit_djstudio(DJStudioProject(data, {}))
    assert report["tracks"][0]["source_end_beyond_file_s"] == 22
    assert not report["tracks"][0]["minimum_duration"]["audibility_verified"]


def test_zero_overlap_is_visible_as_a_cut_in_imported_project():
    data = project_data()
    data["autoEffects"][0]["duration"] = 0
    report = audit_djstudio(DJStudioProject(data, {}), cue=parse_cue_text(cue_text()))
    assert report["transitions"][0]["kind"] == "cut"
    assert report["tracks"][0]["playback_duration_bounds_s"] == [100, 100]


def test_title_mismatch_never_attaches_export_times_by_track_number():
    report = audit_djstudio(DJStudioProject(project_data(), {}), cue=parse_cue_text(cue_text("Other")))
    assert not report["cue_order_matches"]
    assert not report["cue_timing_usable"]
    assert report["transitions"][0]["export_start_s"] is None
    assert report["tracks"][0]["minimum_duration"]["status"] == "unresolved"


@pytest.mark.parametrize("override", [{"tempoMultiplier": 2}, {"bpmLineMode": 3},
                                       {"effectOffset": 8}, {"isDisabled": True}])
def test_unsupported_tempo_or_effect_controls_do_not_claim_playback_bounds(override):
    data = project_data()
    data["autoEffects"][0].update(override)
    report = audit_djstudio(DJStudioProject(data, {}), cue=parse_cue_text(cue_text()))
    assert report["transitions"][0]["overlap_duration_bounds_s"] is None
    assert report["tracks"][0]["playback_duration_bounds_s"] is None


def test_missing_tempo_envelope_or_transition_leaves_duration_unknown():
    data = project_data()
    data.pop("minBpm")
    report = audit_djstudio(DJStudioProject(data, {}), cue=parse_cue_text(cue_text()))
    assert report["tracks"][0]["playback_duration_bounds_s"] is None
    data["autoEffects"] = []
    report = audit_djstudio(DJStudioProject(data, {}))
    assert report["transitions"][0]["kind"] == "unknown"
    assert report["warnings"]


def test_final_duration_invalid_or_absent_not_fabricated():
    project = DJStudioProject(project_data(), {})
    cue = parse_cue_text(cue_text())
    with pytest.raises(DJStudioError, match="later"):
        audit_djstudio(project, cue=cue, export_duration_s=100)
    report = audit_djstudio(project, cue=cue)
    assert report["tracks"][-1]["playback_duration_bounds_s"] is None


def test_cyrillic_cue_with_frame_positions_read_without_loss(tmp_path):
    path = tmp_path / "export.cue"
    path.write_text(cue_text("Бета", "62:02:35"), encoding="utf-8-sig")
    cue = read_cue(path)
    assert cue["tracks"][1]["title"] == "Бета"
    assert cue["tracks"][1]["index01_s"] == pytest.approx(3722 + 35 / 75)
