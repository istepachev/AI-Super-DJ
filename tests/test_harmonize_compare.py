"""Synthetic exports check identity fidelity and fair order comparisons."""

import json
import zipfile

import pytest

from superdj.harmonize_compare import (OrderMatchError, compare_orders,
                                      load_analysis, load_order)


def records():
    return [{"id": "a", "path": "E:/Music/Alpha.flac", "title": "Alpha", "bpm": 120, "key_camelot": "8A"},
            {"id": "b", "path": "E:/Music/Beta.flac", "title": "Beta", "bpm": 122, "key_camelot": "9A"},
            {"id": "c", "path": "E:/Music/Gamma.flac", "title": "Gamma", "bpm": 124, "key_camelot": "9B"},
            {"id": "d", "path": "E:/Music/Delta.flac", "title": "Delta", "bpm": 126, "key_camelot": "10B"}]


def write_json(tmp_path, value):
    path = tmp_path / "order.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("value", [["b", "a", "c", "d"], {"order": ["b", "a", "c", "d"]},
                                  {"tracks": [{"id": "b"}, {"path": "E:/Music/Alpha.flac"},
                                              {"title": "Gamma"}, {"path": "Delta.flac"}]}])
def test_json_ids_paths_and_unique_titles(tmp_path, value):
    assert load_order(write_json(tmp_path, value), records()) == ["b", "a", "c", "d"]


def test_case_unicode_and_separator_normalization_only(tmp_path):
    refs = [{"id": "a", "path": "E:/Music/Café.flac"}, {"id": "b", "path": "E:/Music/Бета.flac"}]
    value = ["e:\\MUSIC\\Cafe\u0301.flac", "file:///E:/Music/%D0%91%D0%B5%D1%82%D0%B0.flac"]
    assert load_order(write_json(tmp_path, value), refs) == ["a", "b"]


def test_ambiguous_basename_is_not_silently_guessed(tmp_path):
    refs = [{"id": "a", "path": "E:/One/Shared.flac"},
            {"id": "b", "path": "E:/Two/Shared.flac"}]
    with pytest.raises(OrderMatchError, match="Ambiguous basename"):
        load_order(write_json(tmp_path, ["Shared.flac"]), refs)
    assert load_order(write_json(tmp_path, ["E:/Two/Shared.flac"]), refs) == ["b"]


def test_ambiguous_title_and_unknown_title_are_errors(tmp_path):
    refs = records()
    refs[1]["title"] = refs[0]["title"]
    with pytest.raises(OrderMatchError, match="Ambiguous title"):
        load_order(write_json(tmp_path, [{"title": "Alpha"}]), refs)
    with pytest.raises(OrderMatchError, match="Unmatched"):
        load_order(write_json(tmp_path, ["Alfa"]), records())


def test_same_track_twice_by_different_aliases_is_rejected(tmp_path):
    with pytest.raises(OrderMatchError, match="Repeated identities"):
        load_order(write_json(tmp_path, ["a", "Alpha.flac"]), records())


def test_m3u_relative_paths_and_inert_comments(tmp_path):
    path = tmp_path / "order.m3u8"
    path.write_text("\ufeff#EXTM3U\n#EXTINF:123,Beta\nBeta.flac\nAlpha.flac\n", encoding="utf-8")
    assert load_order(path, records()) == ["b", "a"]


def test_single_export_cue_matches_track_titles_not_mix_file(tmp_path):
    path = tmp_path / "export.cue"
    path.write_text('''TITLE "A rendered mix"
FILE "whole_mix.flac" WAVE
TRACK 01 AUDIO
 TITLE "Gamma"
 INDEX 01 00:00:00
TRACK 02 AUDIO
 TITLE "Alpha"
 INDEX 01 03:00:00
''', encoding="utf-8")
    assert load_order(path, records()) == ["c", "a"]


def test_multifile_cue_can_match_original_file_paths(tmp_path):
    path = tmp_path / "source.cue"
    path.write_text('''FILE "Beta.flac" WAVE
TRACK 01 AUDIO
 INDEX 01 00:00:00
FILE "Alpha.flac" WAVE
TRACK 02 AUDIO
 INDEX 01 00:00:00
''', encoding="utf-8")
    assert load_order(path, records()) == ["b", "a"]


def test_native_archive_order_and_analysis_read_only(tmp_path, monkeypatch):
    path = tmp_path / "native.djs"
    data = {"mix": [{"libraryKey": "library_b"}, {"libraryKey": "library_a"}],
            "audio": [{"key": "library_a", "importLocation": "Alpha.flac", "bpm": 119,
                       "camelotKey": 16, "name": "Alpha"},
                      {"key": "library_b", "importLocation": "Beta.flac", "bpm": 123,
                       "camelotKey": "9A", "name": "Beta"}]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mixdata.json", json.dumps(data))
        archive.writestr("audioV2/private.flac", b"not audio")
    original = path.read_bytes()
    opened = []
    actual_open = zipfile.ZipFile.open

    def record_open(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return actual_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", record_open)
    assert load_order(path, records()) == ["b", "a"]
    analysis = load_analysis(path, records())
    assert analysis[0]["id"] == "b" and analysis[0]["bpm"] == 123 and analysis[0]["key_camelot"] == "9A"
    assert analysis[0]["key_mapping_status"] == "explicit_camelot"
    assert analysis[1]["native_key"] == 16 and analysis[1]["key_camelot"] == "12A"
    assert analysis[1]["key_mapping_status"] == "verified_numeric_index"
    assert analysis[1]["key_mapping_provenance"]["verified_app_version"] == "4.2.15"
    assert all(name == "mixdata.json" for name in opened)
    assert path.read_bytes() == original


def test_native_explicit_tag_key_is_fallback_when_native_key_is_unknown(tmp_path):
    path = tmp_path / "native.djs"
    data = {"mix": [{"libraryKey": "library_a"}],
            "audio": [{"key": "library_a", "importLocation": "Alpha.flac", "bpm": 120,
                       "camelotKey": -1, "name": "Alpha", "tag": {"key": "A minor"}}]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mixdata.json", json.dumps(data))
    analysis = load_analysis(path, records())
    assert analysis[0]["native_key"] == -1
    assert analysis[0]["key"] == "A minor"
    assert analysis[0]["key_source"] == "explicit_tag.key"
    assert analysis[0]["key_mapping_status"] == "explicit_tag"
    assert analysis[0]["key_source_fallback"] is True
    assert "key_camelot" not in analysis[0]


def native_analysis(tmp_path, key_fields, *, settings=None, source="djstudio"):
    path = tmp_path / "keys.djs"
    data = {"mix": [{"libraryKey": "library_a"}],
            "audio": [{"key": "library_a", "importLocation": "Alpha.flac", "bpm": 121,
                       **key_fields}]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mixdata.json", json.dumps(data))
        if settings is not None:
            archive.writestr("settings.json", json.dumps(settings))
    return load_analysis(path, records(), djs_key_source=source)[0]


@pytest.mark.parametrize("native, expected", list(enumerate([
    "8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B",
    "8A", "3A", "10A", "5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A",
])))
def test_verified_native_keys_are_chromatic_indices_not_wheel_numbers(tmp_path, native, expected):
    record = native_analysis(tmp_path, {"camelotKey": native})
    assert record["key_camelot"] == expected
    assert record["native_key"] == native and record["native_key_values"]["camelotKey"] == native
    assert record["key_source"] == "djstudio.camelotKey"
    assert record["key_source_fallback"] is False
    assert record["bpm"] == 121


@pytest.mark.parametrize("native", [-2, -1, 24, 100, 10 ** 400, True, False, 12.5, "16", None])
def test_invalid_native_keys_remain_unknown(tmp_path, native):
    record = native_analysis(tmp_path, {"camelotKey": native, "noteKey": 12})
    assert record["native_key"] == native
    assert record["key_mapping_status"] == "unknown_native_key"
    assert "key_camelot" not in record and "key" not in record
    assert "key_mapping_provenance" not in record


def test_integer_json_float_native_key_is_preserved_and_decoded(tmp_path):
    record = native_analysis(tmp_path, {"camelotKey": 8.0})
    assert record["key_camelot"] == "4B" and record["native_key"] == 8.0


def test_native_default_does_not_infer_selected_source_from_analysis_setting(tmp_path):
    fields = {"camelotKey": 8, "mikKey": 15, "originalCamelotKey": 12, "tag": {"key": "A minor"}}
    record = native_analysis(tmp_path, fields, settings={"useMixedInKeyAnalysis": True})
    assert record["key_camelot"] == "4B"
    assert record["key_source_requested"] == "djstudio"
    assert record["native_key_values"]["mikKey"] == 15
    assert "key" not in record  # Explicit but conflicting tag never silently overrides the selection.


@pytest.mark.parametrize("source, field, native, expected", [
    ("mixedinkey", "mikKey", 15, "5A"), ("original", "originalCamelotKey", 12, "8A")])
def test_alternate_native_source_requires_explicit_selection(tmp_path, source, field, native, expected):
    record = native_analysis(tmp_path, {"camelotKey": 8, field: native}, source=source)
    assert record["key_camelot"] == expected and record["key_source"] == f"djstudio.{field}"
    assert record["native_key"] == 8 and record["native_key_values"][field] == native
    assert record["key_source_requested"] == source and record["key_source_fallback"] is False


@pytest.mark.parametrize("source, field", [("mixedinkey", "mikKey"), ("original", "originalCamelotKey")])
@pytest.mark.parametrize("missing", [True, False])
def test_unavailable_selected_native_source_falls_back_with_disclosure(tmp_path, source, field, missing):
    fields = {"camelotKey": 8, **({} if missing else {field: -2})}
    record = native_analysis(tmp_path, fields, source=source)
    assert record["key_camelot"] == "4B" and record["key_source"] == "djstudio.camelotKey"
    assert record["key_source_requested"] == source and record["key_source_fallback"] is True


def test_unknown_default_native_key_never_implicitly_uses_other_detector(tmp_path):
    record = native_analysis(tmp_path, {"camelotKey": -1, "mikKey": 15})
    assert "key_camelot" not in record


def test_invalid_native_source_selection_rejected(tmp_path):
    with pytest.raises(ValueError, match="djs_key_source"):
        native_analysis(tmp_path, {"camelotKey": 8}, source="auto")


def test_numeric_key_in_plain_json_is_not_assumed_to_be_djstudio_index(tmp_path):
    path = write_json(tmp_path, {"tracks": [{"id": "a", "camelotKey": 16, "bpm": 120}]})
    assert load_analysis(path, records()) == [
        {"id": "a", "bpm": 120, "native_key": 16, "key_mapping_status": "uninterpreted"}]


def test_external_native_paths_resolve_ambiguous_filenames_without_opening_sources(tmp_path):
    refs = [{"id": "a", "path": "E:/One/Shared.flac"}, {"id": "b", "path": "E:/Two/Shared.flac"}]
    path = tmp_path / "external.djs"
    data = {"mix": [{"libraryKey": "lib_b"}],
            "audio": [{"key": "lib_b", "importLocation": "Shared.flac", "camelotKey": 8,
                       "externalRec": {"fileLocationPath": "Z:/Old/Shared.flac",
                                       "lastKnownFilePath": "e:\\two\\shared.flac"}}]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mixdata.json", json.dumps(data))
    assert load_order(path, refs) == ["b"]
    assert load_analysis(path, refs)[0]["id"] == "b"
    data["audio"][0]["externalRec"] = {"fileLocationPath": "Z:/Old/Shared.flac"}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mixdata.json", json.dumps(data))
    with pytest.raises(OrderMatchError, match="Ambiguous basename"):
        load_order(path, refs)


def test_position_and_directed_adjacency_are_distinct_metrics():
    report = compare_orders(records(), ["a", "b", "c", "d"], ["c", "d", "a", "b"])
    assert report["membership"]["same_track_set"]
    assert report["positions"]["changed_count"] == 4
    assert report["adjacency"]["common_directed_count"] == 2
    assert report["adjacency"]["shared_pairs"] == [["a", "b"], ["c", "d"]]
    assert report["adjacency"]["jaccard"] == pytest.approx(0.5)
    assert "Different orders alone are not errors" in report["note"]
    json.dumps(report, allow_nan=False)


def test_reverse_edge_is_not_a_directed_match():
    report = compare_orders(records()[:2], ["a", "b"], ["b", "a"])
    assert report["adjacency"]["common_directed_count"] == 0
    assert report["evaluations"]["other"]["direction_reversal_count"] == 1


def test_membership_mismatch_reported_and_totals_not_compared():
    report = compare_orders(records(), ["a", "b", "c", "d"], ["a", "b", "unknown"])
    assert report["membership"]["other"]["missing"] == ["c", "d"]
    assert report["membership"]["other"]["extra"] == ["unknown"]
    assert not report["evaluations"]["costs_comparable"]
    assert report["evaluations"]["other_minus_our_cost"] is None
    assert report["evaluations"]["other"]["total_cost"] is None
    assert report["evaluations"]["other"]["unscored_edge_count"] == 1


def test_duplicates_and_duplicate_reference_ids_rejected():
    with pytest.raises(OrderMatchError, match="Repeated identities"):
        compare_orders(records(), ["a", "a"], ["a", "b"])
    with pytest.raises(OrderMatchError, match="Duplicate reference"):
        compare_orders([records()[0], records()[0]], ["a"], ["a"])


def test_imported_analysis_never_changes_common_input_scores():
    our_order, other_order = ["a", "b", "c", "d"], ["d", "c", "b", "a"]
    base = compare_orders(records(), our_order, other_order)
    supplied = [{"id": "a", "bpm": 180, "key_camelot": "2A"},
                {"id": "b", "bpm": 122, "key_camelot": "9A"}]
    report = compare_orders(records(), our_order, other_order, other_analysis=supplied)
    assert report["evaluations"] == base["evaluations"]
    mismatch = report["input_analysis"]["other_vs_reference"]
    assert mismatch["bpm_mismatch_count"] == mismatch["key_mismatch_count"] == 1
    assert mismatch["entries"][0]["bpm_delta"] == 60
    assert mismatch["unmatched_left_ids"] == ["c", "d"]


def test_enharmonic_analysis_keys_compare_by_camelot_not_spelling():
    refs = records()[:1]
    report = compare_orders(refs, ["a"], ["a"], other_analysis=[{"id": "a", "key": "A minor"}])
    entry = report["input_analysis"]["other_vs_reference"]["entries"][0]
    assert entry["key_mismatch"] is False and entry["bpm_mismatch"] is None


def test_unknown_native_key_does_not_become_verified_key_match():
    refs = records()[:1]
    report = compare_orders(refs, ["a"], ["a"], other_analysis=[{"id": "a", "native_key": 16}])
    entry = report["input_analysis"]["other_vs_reference"]["entries"][0]
    assert entry["key_mismatch"] is None and entry["right_native_key"] == 16


def test_mood_allowed_semitone_is_reported_as_high_risk():
    refs = records()[:2]
    refs[0]["key_camelot"] = "1A"
    refs[1]["key_camelot"] = "8A"
    report = compare_orders(refs, ["a", "b"], ["a", "b"], harmonic_mode="mood")
    evaluation = report["evaluations"]["our"]
    assert evaluation["harmonic_allowed_count"] == evaluation["harmonic_high_risk_count"] == 1
    assert evaluation["harmonic_lower_risk_count"] == 0


def test_input_validation_includes_zero_edge_order():
    with pytest.raises(ValueError, match="priority"):
        compare_orders(records()[:1], ["a"], ["a"], priority=2)
    with pytest.raises(ValueError, match="finite"):
        compare_orders(records()[:1], ["a"], ["a"], other_analysis=[{"id": "a", "bpm": float("nan")}])


def test_plain_json_analysis_retains_named_key(tmp_path):
    path = write_json(tmp_path, {"tracks": [{"id": "a", "key": "A minor", "bpm": 120}]})
    assert load_analysis(path, records()) == [{"id": "a", "key": "A minor", "bpm": 120}]
