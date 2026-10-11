"""Read-only identity matching and a fair comparison of two sequence orders.

Order agreement is descriptive, not a musical correctness test. Both paths are
scored against the same reference analysis; imported BPM/key estimates are
reported separately so a detector difference cannot masquerade as a routing bug.
No audio is opened and no file named inside a playlist is executed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import ntpath
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata
from urllib.parse import unquote, urlparse


# Interoperability data, independently checked against the installed key
# formatter and key enumeration in DJ.Studio 4.2.15. These are chromatic note
# indices (major first, then minor), not Camelot wheel positions. No app routing
# or analysis implementation is reproduced here.
DJS_NATIVE_KEY_MAP = (
    "8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B",
    "8A", "3A", "10A", "5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A",
)
DJS_KEY_MAPPING_PROVENANCE = {
    "mapping_id": "djstudio-chromatic-24-v1",
    "verified_app_version": "4.2.15",
    "evidence": "Independent agreement of desktop key formatter and KeyEnumCamelot; "
                "corroborated by exported OpenKey tags 9d and 10m",
    "evidence_bundle_sha256": "e19d8f7c27e0cbea61eb78529c6170b0643b3a57e535d0848e86a75e3cb0dd32",
}
DJS_KEY_SOURCES = {"djstudio": "camelotKey", "mixedinkey": "mikKey", "original": "originalCamelotKey"}


class OrderMatchError(ValueError):
    """An order cannot be mapped to unambiguous reference identities."""


def _text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def _title(value: Any) -> str:
    return " ".join(_text(value).casefold().split())


def _path(value: Any) -> str:
    value = _text(value)
    if value.lower().startswith("file:"):
        parts = urlparse(value)
        value = unquote(parts.path)
        if parts.netloc:
            value = "//" + parts.netloc + value
        elif re.match(r"^/[A-Za-z]:/", value):
            value = value[1:]
    return ntpath.normpath(value.replace("/", "\\")).casefold() if value else ""


def _identity(record: Mapping[str, Any]) -> str:
    value = _text(record.get("id") or record.get("path"))
    if not value:
        raise OrderMatchError("Every reference record needs a nonempty id or path")
    return value


def _records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise OrderMatchError("Reference records must be objects")
        identity = _identity(record)
        if identity in result:
            raise OrderMatchError(f"Duplicate reference identity: {identity}")
        result[identity] = dict(record, id=identity)
    if not result:
        raise OrderMatchError("Reference records must not be empty")
    return result


def _unique(order: Sequence[str], label: str) -> list[str]:
    if isinstance(order, (str, bytes)):
        raise OrderMatchError(f"{label} order must be a list, not a string")
    values = [_text(item) for item in order]
    if any(not item for item in values):
        raise OrderMatchError(f"{label} order contains an empty identity")
    duplicates = [item for item, count in Counter(values).items() if count > 1]
    if duplicates:
        raise OrderMatchError(f"Repeated identities in {label} order: {duplicates}")
    return values


class _Matcher:
    def __init__(self, records: Sequence[Mapping[str, Any]]):
        self.records = _records(records)
        self.paths: dict[str, set[str]] = defaultdict(set)
        self.basenames: dict[str, set[str]] = defaultdict(set)
        self.titles: dict[str, set[str]] = defaultdict(set)
        for identity, record in self.records.items():
            raw_path = record.get("path")
            if raw_path:
                path = _path(raw_path)
                self.paths[path].add(identity)
                basename = ntpath.basename(path)
                self.basenames[basename].add(identity)
                self.titles[_title(ntpath.splitext(basename)[0])].add(identity)
            values = [record.get("title"), record.get("name")]
            for container in ("tag", "tags", "metadata"):
                if isinstance(record.get(container), Mapping):
                    values.append(record[container].get("title"))
            for value in values:
                if _title(value):
                    self.titles[_title(value)].add(identity)

    @staticmethod
    def _pick(matches: set[str], value: Any, kind: str) -> str | None:
        if len(matches) > 1:
            raise OrderMatchError(f"Ambiguous {kind} {value!r}: {sorted(matches)}")
        return next(iter(matches)) if matches else None

    def match(self, value: Any, *, parent: Path | None = None) -> str:
        if isinstance(value, Mapping):
            identity = _text(value.get("id"))
            if identity in self.records:
                return identity
            candidates = [value.get(key) for key in
                          ("path", "source_path", "filePath", "importLocation", "href")]
            external = value.get("externalRec")
            if isinstance(external, Mapping):
                candidates.extend(external.get(key) for key in ("fileLocationPath", "lastKnownFilePath"))
            matched = self._match_paths(candidates, parent)
            if matched:
                return matched
            for field in ("title", "name"):
                candidate = value.get(field)
                if candidate:
                    matched = self._pick(self.titles.get(_title(candidate), set()), candidate, "title")
                    if matched:
                        return matched
            raise OrderMatchError(f"Unmatched order entry: {value!r}")
        candidate = _text(value)
        if candidate in self.records:
            return candidate
        matched = self._match_path(candidate, parent)
        if matched:
            return matched
        matched = self._pick(self.titles.get(_title(candidate), set()), candidate, "title")
        if matched:
            return matched
        raise OrderMatchError(f"Unmatched order entry: {candidate!r}")

    def _match_path(self, candidate: Any, parent: Path | None) -> str | None:
        return self._match_paths([candidate], parent)

    def _match_paths(self, candidates: Sequence[Any], parent: Path | None) -> str | None:
        # Exhaust exact paths before considering filenames. A DJS may export a
        # bare importLocation alongside a precise externalRec path; the bare
        # name must not hide that stronger evidence. Paths are inert identities
        # and are not asserted to exist on the current workstation.
        paths = []
        for candidate in candidates:
            if not candidate:
                continue
            path = _path(candidate)
            full_paths = [path]
            if parent is not None and not ntpath.isabs(path):
                full_paths.insert(0, _path(str(parent / _text(candidate))))
            paths.append((candidate, path, full_paths))
        for candidate, _, full_paths in paths:
            for full_path in full_paths:
                matched = self._pick(self.paths.get(full_path, set()), candidate, "path")
                if matched:
                    return matched
        for candidate, path, _ in paths:
            matched = self._pick(self.basenames.get(ntpath.basename(path), set()), candidate, "basename")
            if matched:
                return matched
        return None


def _json_entries(data: Any) -> list[Any]:
    if isinstance(data, Mapping):
        data = data.get("order", data.get("tracks"))
    if not isinstance(data, list):
        raise OrderMatchError("Order JSON must be a list or an object containing order/tracks")
    return data


def _djs_entries(path: Path) -> list[dict[str, Any]]:
    from .djstudio import load_djs

    project = load_djs(path)
    library = {}
    for audio in project.mixdata["audio"]:
        key = audio.get("key")
        if key in library:
            raise OrderMatchError(f"Duplicate DJ.Studio library key: {key}")
        library[key] = audio
    result = []
    for track in project.mixdata["mix"]:
        key = track.get("libraryKey")
        if key not in library:
            raise OrderMatchError(f"DJ.Studio mix references missing library key: {key}")
        audio = library[key]
        tag = audio.get("tag", {})
        title = tag.get("title") if isinstance(tag, Mapping) else None
        entry = dict(audio)
        entry["title"] = title or audio.get("name")
        # The native library key is a content/library identifier, not our id.
        entry.pop("id", None)
        result.append(entry)
    return result


def _load_entries(path: Path) -> list[Any]:
    suffix = path.suffix.casefold()
    if suffix == ".djs":
        return _djs_entries(path)
    if suffix == ".cue":
        from .djstudio import read_cue

        cue = read_cue(path)
        result = []
        multi_file = len(cue["files"]) == len(cue["tracks"])
        for track in cue["tracks"]:
            entry = {"title": track["title"]}
            if multi_file:
                entry["path"] = cue["files"][track["file_index"]]["name"]
            result.append(entry)
        return result
    if suffix in {".m3u", ".m3u8"}:
        # Local UTF-8 playlists; remote URIs are matched as inert identities.
        return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    if suffix == ".json":
        def invalid_constant(value):
            raise OrderMatchError(f"Non-finite JSON value: {value}")

        return _json_entries(json.loads(path.read_text(encoding="utf-8-sig"),
                                        parse_constant=invalid_constant))
    raise OrderMatchError("Supported order files: JSON, M3U/M3U8, CUE and DJS")


def load_order(path: str | Path, reference_records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Resolve a local export to reference IDs, rejecting ambiguity/duplicates.

    JSON accepts IDs, paths, or track objects. An exported single-file CUE uses
    unique titles because its FILE directive names the complete rendered mix.
    A DJS is read through the existing bounded metadata-only archive adapter.
    A subset is permitted here; :func:`compare_orders` reports missing tracks.
    """
    path = Path(path)
    matcher = _Matcher(reference_records)
    order = [matcher.match(value, parent=path.parent) for value in _load_entries(path)]
    if not order:
        raise OrderMatchError("Order file is empty")
    return _unique(order, "imported")


def _djs_key(value: Any) -> tuple[str | None, bool]:
    """Decode only a valid native index or an explicit Camelot string."""
    if isinstance(value, str) and re.fullmatch(r"(?:[1-9]|1[0-2])[AB]", value.strip().upper()):
        return value.strip().upper(), False
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < len(DJS_NATIVE_KEY_MAP):
            return DJS_NATIVE_KEY_MAP[value], True
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        if 0 <= value < len(DJS_NATIVE_KEY_MAP):
            return DJS_NATIVE_KEY_MAP[int(value)], True
    return None, False


def _import_djs_key(entry: Mapping[str, Any], source: str) -> dict[str, Any]:
    selected_field = DJS_KEY_SOURCES[source]
    fields = [selected_field] if selected_field == "camelotKey" else [selected_field, "camelotKey"]
    record: dict[str, Any] = {"key_source_requested": source,
        "native_key_values": {field: entry[field] for field in
                              ("camelotKey", "mikKey", "originalCamelotKey", "noteKey") if field in entry}}
    if "camelotKey" in entry:
        record["native_key"] = entry["camelotKey"]
    for field in fields:
        key, numeric = _djs_key(entry.get(field))
        if key is not None:
            record.update(key_camelot=key, key_source=f"djstudio.{field}",
                          key_source_fallback=field != selected_field,
                          key_mapping_status="verified_numeric_index" if numeric else "explicit_camelot")
            if numeric:
                record["key_mapping_provenance"] = dict(DJS_KEY_MAPPING_PROVENANCE)
            return record
    record["key_mapping_status"] = "unknown_native_key"
    record["key_source_fallback"] = selected_field != "camelotKey"
    return record


def load_analysis(path: str | Path, reference_records: Sequence[Mapping[str, Any]], *,
                  djs_key_source: str = "djstudio") -> list[dict[str, Any]]:
    """Read available imported analysis without replacing reference estimates.

    DJS native indices use the independently verified 24-key map, with raw
    fields and evidence retained. The default reads ``camelotKey``; selecting
    ``mixedinkey`` or ``original`` explicitly tries that field first, then the
    native DJ.Studio key when unavailable. The active source selector is not
    stored in the archive: ``useMixedInKeyAnalysis`` is never treated as proof
    of its selection. Numeric keys in plain JSON remain uninterpreted.
    CUE/M3U usually carry no BPM/key metadata and therefore return only IDs.
    """
    if djs_key_source not in DJS_KEY_SOURCES:
        raise ValueError(f"djs_key_source must be one of {tuple(DJS_KEY_SOURCES)}")
    path = Path(path)
    is_djs = path.suffix.casefold() == ".djs"
    matcher = _Matcher(reference_records)
    result = []
    for entry in _load_entries(path):
        identity = matcher.match(entry, parent=path.parent)
        record = {"id": identity}
        if isinstance(entry, Mapping):
            bpm = entry.get("bpm", entry.get("source_bpm"))
            if bpm is not None:
                record["bpm"] = bpm
            if is_djs:
                record.update(_import_djs_key(entry, djs_key_source))
            else:
                key = entry.get("key_camelot", entry.get("camelotKey", entry.get("key")))
                if isinstance(key, str) and re.fullmatch(r"(?:[1-9]|1[0-2])[AB]", key.strip().upper()):
                    record["key_camelot"] = key.strip().upper()
                elif isinstance(key, str) and "camelotKey" not in entry:
                    record["key"] = key
                elif key is not None:
                    record["native_key"] = key
                    record["key_mapping_status"] = "uninterpreted"
            tag = entry.get("tag")
            if isinstance(tag, Mapping) and not (record.get("key_camelot") or record.get("key")):
                from .harmonize import harmonic_relation

                explicit_key = tag.get("key")
                if isinstance(explicit_key, str) and harmonic_relation(explicit_key, explicit_key)["key_from"]:
                    record["key"] = explicit_key
                    record["key_source"] = "explicit_tag.key"
                    if is_djs:
                        record["key_mapping_status"] = "explicit_tag"
                        record["key_source_fallback"] = True
        result.append(record)
    _unique([record["id"] for record in result], "imported analysis")
    return result


def _analysis_records(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        if "tracks" in value:
            value = value["tracks"]
        else:
            value = [dict(record, id=identity) for identity, record in value.items()]
    if not isinstance(value, (list, tuple)):
        raise OrderMatchError("Analysis must be a list of records or a mapping by ID")
    return _records(value) if value else {}


def chain_agreement(our_order: Sequence[str], reference_orders: Sequence[Sequence[str]],
                    *, lengths: Sequence[int] = (2, 3, 4)) -> dict[str, Any]:
    """Compare exact directed contiguous windows against same-pool variants.

    Three/four means total tracks, not three/four future steps. References are
    evaluation evidence only. A recurring pair/chain need not have occurred
    with the same remaining pool; recurrence is not a prediction ceiling.
    """
    from collections import Counter

    ours = _unique(our_order, "our chain order")
    if isinstance(reference_orders, (str, bytes)) or not isinstance(reference_orders, (list, tuple)) or not reference_orders:
        raise OrderMatchError("Chain comparison requires at least one reference order")
    references = [_unique(order, "reference chain order") for order in reference_orders]
    if any(set(order) != set(ours) for order in references):
        raise OrderMatchError("Chain comparison requires identical track membership")
    if (isinstance(lengths, (str, bytes)) or not isinstance(lengths, (list, tuple))
            or not lengths or any(type(length) is not int or length < 2 for length in lengths)):
        raise OrderMatchError("Chain lengths must be integers of at least two tracks")

    def windows(order, length):
        return {tuple(order[index:index+length]) for index in range(max(0, len(order)-length+1))}

    rows = []
    for length in dict.fromkeys(lengths):
        our_windows = windows(ours, length)
        reference_windows = [windows(order, length) for order in references]
        counts = Counter(window for batch in reference_windows for window in batch)
        repeated = {window for window, count in counts.items() if count >= 2}
        hits = our_windows & counts.keys()
        denominator = max(0, len(ours)-length+1)
        rows.append(dict(track_count=length, edge_count=length-1,
            our_window_count=denominator, reference_variant_count=len(references),
            matches_per_reference=[len(our_windows & batch) for batch in reference_windows],
            any_reference_count=len(hits),
            any_reference_rate=len(hits)/denominator if denominator else None,
            repeated_reference_window_count=len(repeated),
            repeated_reference_hit_count=len(our_windows & repeated),
            repeated_reference_hit_rate=len(our_windows & repeated)/len(repeated) if repeated else None,
            reference_frequency_weighted_hits=sum(counts[window] for window in hits),
            shared_windows=[list(window) for window in sorted(hits)]))
    return dict(schema_version=1, scope="exact directed contiguous windows on identical source pools",
        definition="A chain of three/four total tracks contains two/three edges; no skipped tracks or reversed order",
        recurrence_scope="same-pool reference recurrence, not identical current/remaining-set evidence or a predictability ceiling",
        track_count=len(ours), references=len(references), lengths=rows)


def compare_orders(reference_records: Sequence[Mapping[str, Any]], our_order: Sequence[str],
                   other_order: Sequence[str], harmonic_mode: str = "fuzzy", priority: float = 0.5,
                   direction: str = "ascending", bpm_tolerance: float | None = None,
                   our_analysis: Any = None, other_analysis: Any = None) -> dict[str, Any]:
    """Compare membership, positions and directed neighbors using common inputs.

    Missing/extra identities are reported explicitly; duplicates are errors.
    Imported orders need not match to have equal or better declared costs.
    Scores express our published heuristic, not a DJ.Studio internal score or
    an assessment of the audible transition. Position numbers are one-based.
    """
    from .harmonize import harmonic_relation, score_pair

    reference = _records(reference_records)
    our_order = _unique(our_order, "our")
    other_order = _unique(other_order, "other")
    our_set, other_set, reference_set = set(our_order), set(other_order), set(reference)
    settings = {"mode": harmonic_mode, "priority": priority,
                "direction": direction, "bpm_tolerance": bpm_tolerance}
    # Validate the objective even when either path contains no edges.
    sample = next(iter(reference.values()))
    score_pair(sample, sample, **settings)

    def evaluate(order):
        edges = []
        for left, right in zip(order, order[1:]):
            if left in reference and right in reference:
                edge = score_pair(reference[left], reference[right], **settings)
            else:
                edge = {"from": left, "to": right, "cost": None,
                        "status": "unknown_reference_identity"}
            edges.append(edge)
        costs = [edge["cost"] for edge in edges if edge["cost"] is not None]
        harmonics = [edge["harmonic"] for edge in edges if "harmonic" in edge]
        tempos = [edge["tempo"] for edge in edges if "tempo" in edge]
        jumps = [tempo["absolute_delta_bpm"] for tempo in tempos
                 if tempo["absolute_delta_bpm"] is not None]
        return {"edges": edges, "total_cost": math.fsum(costs) if len(costs) == len(edges) else None,
                "known_edges_cost": math.fsum(costs), "scored_edge_count": len(costs),
                "unscored_edge_count": len(edges) - len(costs),
                "harmonic_allowed_count": sum(h["accepted"] for h in harmonics),
                "harmonic_lower_risk_count": sum(h["accepted"] and h["risk"] == "lower"
                                                 for h in harmonics),
                "harmonic_review_count": sum(h["risk"] == "review" for h in harmonics),
                "harmonic_high_risk_count": sum(h["risk"] == "high" for h in harmonics),
                "harmonic_unknown_count": sum(h["relation"] == "unknown" for h in harmonics),
                "maximum_bpm_jump": max(jumps, default=None), "total_bpm_travel": math.fsum(jumps),
                "direction_reversal_count": sum((t["direction_reversal_bpm"] or 0) > 0 for t in tempos),
                "tolerance_exceedance_count": sum(t["tolerance_excess_bpm"] > 0 for t in tempos)}

    def membership(order_set):
        return {"missing": sorted(reference_set - order_set), "extra": sorted(order_set - reference_set),
                "matches_reference": order_set == reference_set}

    our_pos = {identity: i + 1 for i, identity in enumerate(our_order)}
    other_pos = {identity: i + 1 for i, identity in enumerate(other_order)}
    positions = [{"id": identity, "our_position": our_pos.get(identity),
                  "other_position": other_pos.get(identity),
                  "delta": other_pos[identity] - our_pos[identity]
                  if identity in our_pos and identity in other_pos else None}
                 for identity in our_order + [item for item in other_order if item not in our_set]]
    common_positions = [entry for entry in positions if entry["delta"] is not None]
    our_edges = set(zip(our_order, our_order[1:]))
    other_edges = set(zip(other_order, other_order[1:]))
    shared_edges, union_edges = our_edges & other_edges, our_edges | other_edges
    our_eval, other_eval = evaluate(our_order), evaluate(other_order)
    complete = our_set == other_set == reference_set
    cost_delta = other_eval["total_cost"] - our_eval["total_cost"] if complete else None

    def analysis_difference(left, right):
        entries = []
        for identity in sorted(set(left) & set(right)):
            a, b = left[identity], right[identity]
            score_pair(a, a, **settings)
            score_pair(b, b, **settings)
            available_bpm = a.get("bpm") is not None and b.get("bpm") is not None
            bpm_delta = float(b["bpm"]) - float(a["bpm"]) if available_bpm else None
            if bpm_delta is not None and not math.isfinite(bpm_delta):
                raise OrderMatchError("Imported BPM must be finite")
            ka = a.get("key_camelot") or a.get("key")
            kb = b.get("key_camelot") or b.get("key")
            canonical_a = harmonic_relation(ka, ka)["key_from"]
            canonical_b = harmonic_relation(kb, kb)["key_from"]
            available_key = canonical_a is not None and canonical_b is not None
            entries.append({"id": identity, "left_bpm": a.get("bpm"), "right_bpm": b.get("bpm"),
                            "bpm_delta": bpm_delta,
                            "bpm_mismatch": abs(bpm_delta) > 0.01 if bpm_delta is not None else None,
                            "left_key": ka, "right_key": kb,
                            "left_camelot": canonical_a, "right_camelot": canonical_b,
                            "key_mismatch": canonical_a != canonical_b if available_key else None,
                            "right_native_key": b.get("native_key")})
        return {"entries": entries, "bpm_mismatch_count": sum(e["bpm_mismatch"] is True for e in entries),
                "key_mismatch_count": sum(e["key_mismatch"] is True for e in entries),
                "unmatched_left_ids": sorted(set(left) - set(right)),
                "unmatched_right_ids": sorted(set(right) - set(left))}

    analysis = {}
    ours, others = _analysis_records(our_analysis), _analysis_records(other_analysis)
    if our_analysis is not None:
        analysis["our_vs_reference"] = analysis_difference(reference, ours)
    if other_analysis is not None:
        analysis["other_vs_reference"] = analysis_difference(reference, others)
    if our_analysis is not None and other_analysis is not None:
        analysis["our_vs_other"] = analysis_difference(ours, others)

    from .harmonize_successors import evaluate_successors
    successors = evaluate_successors(list(reference.values()), other_order, **settings) if other_set == reference_set else None
    return {"schema_version": 1, "objective": settings,
            "note": "Different orders alone are not errors. BPM/key estimates, settings and ties matter; "
                    "costs below use our heuristic on identical reference inputs, not DJ.Studio's score.",
            "counts": {"reference": len(reference), "our": len(our_order), "other": len(other_order)},
            "orders": {"our": our_order, "other": other_order},
            "membership": {"our": membership(our_set), "other": membership(other_set),
                           "same_track_set": complete},
            "positions": {"entries": positions, "shared_count": len(common_positions),
                          "same_count": sum(entry["delta"] == 0 for entry in common_positions),
                          "changed_count": sum(entry["delta"] != 0 for entry in common_positions)},
            "adjacency": {"our_count": len(our_edges), "other_count": len(other_edges),
                          "common_directed_count": len(shared_edges),
                          "shared_pairs": [list(pair) for pair in sorted(shared_edges)],
                          "jaccard": len(shared_edges) / len(union_edges) if union_edges else 1.0},
            "chain_agreement": chain_agreement(our_order, [other_order]) if complete else None,
            "evaluations": {"our": our_eval, "other": other_eval,
                            "other_minus_our_cost": cost_delta,
                            "costs_comparable": complete},
            "input_analysis": analysis,
            "successor_audit": successors,
            "successor_audit_scope": "remaining candidates after the demonstrated prefix; local cost, not conditional global choice"}
