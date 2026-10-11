"""Real CLI exports and metadata reruns, using small synthetic playlists."""

import json

from superdj.cli import main
from superdj.harmonize import harmonize_tracks
from superdj.harmonize_compare import load_order


def tracks():
    return [{"id": "a", "path": "E:/Synthetic/Alpha.flac", "bpm": 120, "key_camelot": "8A"},
            {"id": "b", "path": "E:/Synthetic/Beta.flac", "bpm": 122, "key_camelot": "9A"},
            {"id": "c", "path": "E:/Synthetic/Gamma.flac", "bpm": 124, "key_camelot": "9B"},
            {"id": "d", "path": "E:/Synthetic/Delta.flac", "bpm": 126, "key_camelot": "10B"}]


def dump(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_harmonize_cli_pack_compares_roundtrip_without_inventing_metadata(tmp_path):
    refs = tracks()
    source = dump(tmp_path / "records.json", refs)
    pack = tmp_path / "pack"
    main(["harmonize", "--records", source, "--work", str(pack), "--seed", "7",
          "--first", "b", "--last", "d", "--harmonic-mode", "fuzzy", "--priority", ".5"])
    plan = json.loads((pack / "harmonize_plan.json").read_text(encoding="utf-8"))
    assert plan["order"][0] == "b" and plan["order"][-1] == "d"
    assert load_order(pack / "input_tracks.m3u8", refs) == ["a", "b", "c", "d"]
    assert load_order(pack / "harmonized_tracks.m3u8", refs) == plan["order"]
    assert (pack / "harmonize_report.html").is_file()
    compared = tmp_path / "comparison"
    main(["harmonize-compare", "--records", source, "--our-order", str(pack / "harmonize_plan.json"),
          "--other-order", str(pack / "harmonized_tracks.m3u8"), "--work", str(compared),
          "--seed", "7", "--first", "b", "--last", "d"])
    result = json.loads((compared / "harmonize_comparison.json").read_text(encoding="utf-8"))
    assert result["membership"]["same_track_set"]
    assert result["positions"]["changed_count"] == 0
    assert result["adjacency"]["common_directed_count"] == 3
    assert result["evaluations"]["other_minus_our_cost"] == 0
    assert result["replanned_on_other_metadata"] is None
    assert result["experiment_controls"]["first_id"] == "b"


def test_complete_json_metadata_replan_uses_all_exported_values(tmp_path):
    refs = tracks()
    source = dump(tmp_path / "records.json", refs)
    ours = dump(tmp_path / "ours.json", {"order": ["a", "b", "c", "d"]})
    exported = [dict(row, bpm=140 + i * 2, key_camelot=key)
                for i, (row, key) in enumerate(zip(reversed(refs), ["1A", "2A", "3B", "4B"]))]
    other = dump(tmp_path / "other.json", {"tracks": exported})
    work = tmp_path / "comparison"
    main(["harmonize-compare", "--records", source, "--our-order", ours, "--other-order", other,
          "--work", str(work), "--seed", "11", "--direction", "descending", "--priority", ".7"])
    result = json.loads((work / "harmonize_comparison.json").read_text(encoding="utf-8"))
    expected = harmonize_tracks(exported, seed=11, direction="descending", priority=.7)
    assert result["replanned_on_other_metadata"]["order"] == expected["order"]
    assert result["replanned_on_other_metadata"]["total_cost"] == expected["total_cost"]
    assert result["comparison_on_other_metadata"]["membership"]["same_track_set"]
    assert result["input_analysis"]["other_vs_reference"]["bpm_mismatch_count"] == 4
    assert result["input_analysis"]["other_vs_reference"]["key_mismatch_count"] == 4


def test_partial_json_metadata_does_not_borrow_reference_key_for_replan(tmp_path):
    refs = tracks()
    source = dump(tmp_path / "records.json", refs)
    ours = dump(tmp_path / "ours.json", ["a", "b", "c", "d"])
    exported = [{"id": row["id"], "bpm": 140,
                 **({"key_camelot": "1A"} if row["id"] != "d" else {})} for row in refs]
    other = dump(tmp_path / "other.json", {"tracks": exported})
    work = tmp_path / "comparison"
    main(["harmonize-compare", "--records", source, "--our-order", ours, "--other-order", other,
          "--work", str(work)])
    result = json.loads((work / "harmonize_comparison.json").read_text(encoding="utf-8"))
    assert result["replanned_on_other_metadata"] is None
    assert "Complete" in result["replan_reason"]
    entry = next(row for row in result["input_analysis"]["other_vs_reference"]["entries"] if row["id"] == "d")
    assert entry["right_key"] is None and entry["key_mismatch"] is None


def test_complete_named_keys_replace_old_camelot_fields_on_replan(tmp_path):
    refs = tracks()
    source = dump(tmp_path / "records.json", refs)
    ours = dump(tmp_path / "ours.json", ["a", "b", "c", "d"])
    exported = [{"id": row["id"], "bpm": row["bpm"], "key": key}
                for row, key in zip(refs, ["B major", "F# major", "G# minor", "D# minor"])]
    other = dump(tmp_path / "other.json", {"tracks": exported})
    work = tmp_path / "comparison"
    main(["harmonize-compare", "--records", source, "--our-order", ours, "--other-order", other,
          "--work", str(work), "--first", "a", "--last", "d", "--priority", "1"])
    result = json.loads((work / "harmonize_comparison.json").read_text(encoding="utf-8"))
    expected = harmonize_tracks(exported, first_id="a", last_id="d", priority=1)
    rerun = result["replanned_on_other_metadata"]
    assert rerun["order"] == expected["order"]
    assert rerun["total_cost"] == expected["total_cost"]
    assert [edge["harmonic"]["key_from"] for edge in rerun["edges"]] == [
        edge["harmonic"]["key_from"] for edge in expected["edges"]]


def test_unknown_exported_keys_skip_replan_and_objective_difference_is_visible(tmp_path):
    refs = tracks()
    source = dump(tmp_path / "records.json", refs)
    ours = dump(tmp_path / "ours.json", {"order": ["a", "b", "c", "d"],
        "settings": {"mode": "mood", "priority": .7, "direction": "descending", "bpm_tolerance": 4}})
    other = dump(tmp_path / "other.json", {"tracks": [{"id": r['id'], "bpm": 120, "key": "unknown"} for r in refs]})
    work = tmp_path / "comparison"
    main(["harmonize-compare", "--records", source, "--our-order", ours, "--other-order", other, "--work", str(work)])
    result = json.loads((work / "harmonize_comparison.json").read_text("utf-8"))
    assert result['replanned_on_other_metadata'] is None
    assert result['comparison_objective_differs_from_input_plan'] is True


def test_compare_includes_remaining_candidate_ranks(tmp_path):
    refs = tracks()
    source = dump(tmp_path / 'records.json', refs)
    order = dump(tmp_path / 'order.json', ['a','b','c','d'])
    work = tmp_path / 'comparison'
    main(['harmonize-compare','--records',source,'--our-order',order,
          '--other-order',order,'--work',str(work)])
    audit = json.loads((work/'harmonize_comparison.json').read_text(encoding='utf-8'))['successor_audit']
    assert [e['candidate_count'] for e in audit['edges']] == [3,2,1]
    assert audit['edges'][1]['current'] == 'b'
    assert all(row['id'] != 'a' for row in audit['edges'][1]['top_candidates'])


def test_learning_manifest_paths_are_relative_and_defaults_unchanged(tmp_path):
    from superdj.harmonize_calibration import cross_validate_successor_controls
    refs = tracks()
    dump(tmp_path/'records.json', refs)
    dump(tmp_path/'one.json', ['a','b','c','d'])
    dump(tmp_path/'two.json', ['d','c','b','a'])
    manifest = dump(tmp_path/'examples.json', [dict(name='one',records='records.json',order='one.json'),
                                               dict(name='two',records='records.json',order='two.json')])
    main(['harmonize-learn','--examples',manifest,'--work',str(tmp_path/'learn'),'--min-candidates','1'])
    result = json.loads((tmp_path/'learn/successor_calibration.json').read_text(encoding='utf-8'))
    expected = cross_validate_successor_controls([dict(name='one',records=refs,order=['a','b','c','d']),
                                                  dict(name='two',records=refs,order=['d','c','b','a'])],min_candidates=1)
    assert result == expected
    assert result['defaults_changed'] is False
