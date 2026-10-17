"""Read-only project archive project and CUE adapter.

``mixMap.time`` is a source-audio position, not an export timeline.  CUE INDEX
01 positions delimit export chapters, not the whole audible span of a track.
This module deliberately preserves native automation without interpreting or
rendering it.  Native tempo interpolation and musical phrasing remain unverified.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
import unicodedata
from typing import Any, Mapping
import zipfile


class ArchiveError(ValueError):
    """Unsupported or inconsistent project/CUE data."""


@dataclass(frozen=True)
class ArchiveProject:
    mixdata: dict[str, Any]
    settings: dict[str, Any]
    archive_members: tuple[str, ...] = ()
    metadata_sha256: str | None = None

    def native_state(self) -> dict[str, Any]:
        """Return an independent, lossless JSON snapshot of the native state."""
        return copy.deepcopy({"mixdata": self.mixdata, "settings": self.settings})


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _bad_constant(value: str) -> None:
    raise ArchiveError(f"Non-finite JSON number: {value}")


def load_djs(path: str | Path, *, max_metadata_bytes: int = 64 * 1024 * 1024) -> ArchiveProject:
    """Read only JSON metadata from a ZIP .djs; never extract embedded audio.

    Every member name is checked, and duplicate metadata entries, encrypted
    metadata and metadata larger than the configured bound are rejected.
    This is not a full archive CRC/audio-integrity verification.
    """
    import hashlib

    if max_metadata_bytes <= 0:
        raise ValueError("max_metadata_bytes must be positive")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            name = member.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or re.match(r"^[A-Za-z]:", name):
                raise ArchiveError("Unsafe archive member name")

        def metadata(name: str, required: bool) -> tuple[dict[str, Any], bytes]:
            candidates = [m for m in members if m.filename == name]
            if not candidates and not required:
                return {}, b""
            if len(candidates) != 1:
                raise ArchiveError(f"Expected exactly one {name}")
            member = candidates[0]
            if member.flag_bits & 1 or member.file_size > max_metadata_bytes:
                raise ArchiveError(f"Encrypted or oversized {name}")
            with archive.open(member) as stream:
                raw = stream.read(max_metadata_bytes + 1)
            if len(raw) > max_metadata_bytes:
                raise ArchiveError(f"Oversized {name}")
            value = json.loads(raw.decode("utf-8-sig"), parse_constant=_bad_constant)
            if not isinstance(value, dict):
                raise ArchiveError(f"{name} must be an object")
            return value, raw

        mixdata, raw = metadata("mixdata.json", True)
        settings, _ = metadata("settings.json", False)
    if not isinstance(mixdata.get("mix"), list) or not isinstance(mixdata.get("audio"), list):
        raise ArchiveError("Project must contain mix and audio lists")
    return ArchiveProject(mixdata, settings, tuple(m.filename for m in members),
                           hashlib.sha256(raw).hexdigest())


def cue_timestamp(value: str) -> float:
    """CUE MM:SS:FF uses 75 frames per second; minutes may exceed 59."""
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})", value.strip())
    if not match:
        raise ArchiveError(f"Invalid CUE timestamp: {value}")
    minutes, seconds, frames = map(int, match.groups())
    if seconds >= 60 or frames >= 75:
        raise ArchiveError(f"Invalid CUE timestamp: {value}")
    return minutes * 60 + seconds + frames / 75


def _cue_string(value: str) -> str:
    value = value.strip()
    if value.startswith('"'):
        if not value.endswith('"'):
            raise ArchiveError("Unclosed CUE quoted string")
        return value[1:-1].replace('\\"', '"')
    return value


def parse_cue_text(text: str) -> dict[str, Any]:
    """Parse chapters without assuming FILE's declared type matches its suffix.

    Unknown directives (including REM) are preserved as inert text.  No named
    file is opened, executed or resolved by this parser.
    """
    result: dict[str, Any] = {"title": "", "performer": "", "files": [], "tracks": [],
                              "unknown_directives": []}
    current_file = None
    current_track = None
    for line_number, line in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        directive, _, value = line.partition(" ")
        if "\t" in directive:
            directive, value = re.split(r"\s+", line, maxsplit=1)
        directive = directive.upper()
        if directive == "FILE":
            match = re.fullmatch(r'(?:"(.*)"|(\S+))\s+(\S+)', value)
            if not match:
                raise ArchiveError(f"Invalid FILE at line {line_number}")
            current_file = {"name": match.group(1) or match.group(2), "declared_type": match.group(3)}
            result["files"].append(current_file)
            current_track = None
        elif directive == "TRACK":
            match = re.fullmatch(r"(\d+)\s+(\S+)", value)
            if current_file is None or not match:
                raise ArchiveError(f"Invalid TRACK at line {line_number}")
            number = int(match.group(1))
            if number <= 0 or any(t["number"] == number for t in result["tracks"]):
                raise ArchiveError("Duplicate or non-positive CUE track number")
            current_track = {"number": number, "type": match.group(2), "title": "", "performer": "",
                             "file_index": len(result["files"]) - 1, "indexes": {}}
            result["tracks"].append(current_track)
        elif directive in {"TITLE", "PERFORMER"}:
            target = current_track if current_track is not None else result
            target[directive.lower()] = _cue_string(value)
        elif directive == "INDEX":
            match = re.fullmatch(r"(\d{2})\s+(\S+)", value)
            if current_track is None or not match:
                raise ArchiveError(f"Invalid INDEX at line {line_number}")
            key = match.group(1)
            if key in current_track["indexes"]:
                raise ArchiveError("Duplicate CUE INDEX")
            current_track["indexes"][key] = cue_timestamp(match.group(2))
        else:
            result["unknown_directives"].append({"line": line_number, "text": line})
    previous_by_file: dict[int, float] = {}
    for track in result["tracks"]:
        if track["type"].upper() != "AUDIO" or "01" not in track["indexes"]:
            raise ArchiveError("Every CUE track must be AUDIO with INDEX 01")
        start = track["indexes"]["01"]
        file_index = track["file_index"]
        if file_index in previous_by_file and start <= previous_by_file[file_index]:
            raise ArchiveError("CUE INDEX 01 must increase within each file")
        if track["indexes"].get("00", start) > start:
            raise ArchiveError("CUE INDEX 00 is later than INDEX 01")
        previous_by_file[file_index] = start
        track["index01_s"] = start
    if not result["tracks"]:
        raise ArchiveError("Empty CUE track list")
    return result


def read_cue(path: str | Path, *, encoding: str | None = None) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if encoding is None:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            encoding = "utf-16"
        else:
            try:
                raw.decode("utf-8-sig")
                encoding = "utf-8-sig"
            except UnicodeDecodeError:
                encoding = "cp1251"
    result = parse_cue_text(raw.decode(encoding))
    result["encoding"] = encoding
    return result


def _title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"\.(mp3|flac|wav|m4a|ogg|aiff?)$", "", value)
    return " ".join(value.split())


def _cue_beat(track: Mapping[str, Any], index: int) -> float | None:
    points = track.get("cueData", {}).get("systemCuePoints", [])
    return _finite_number(points[index].get("beat")) if len(points) > index else None


def _source_time(track: Mapping[str, Any], beat: float | None) -> float | None:
    if beat is None or beat < 0:
        return None
    mapping = track.get("mixMap", [])
    index = math.floor(beat)
    if index >= len(mapping):
        return None
    point = mapping[index]
    time = _finite_number(point.get("time"))
    duration = _finite_number(point.get("duration"))
    if time is None or duration is None or duration < 0:
        return None
    return time + (beat - index) * duration


def _tempo_values(library: Mapping[str, Any], track: Mapping[str, Any]) -> list[float]:
    values = [_finite_number(library.get("bpm")), _finite_number(track.get("gridBpm")),
              _finite_number(track.get("mixBpm"))]
    values.extend(_finite_number(p.get("value")) for p in library.get("bpmLine", []))
    return [value for value in values if value is not None and value > 0]


def _bounds(beats: float, tempos: list[float]) -> list[float] | None:
    if beats < 0 or not tempos:
        return None
    return [beats * 60 / max(tempos), beats * 60 / min(tempos)]


def audit_archive(project: str | Path | ArchiveProject, *, cue: str | Path | Mapping[str, Any] | None = None,
                   export_duration_s: float | None = None, minimum_track_s: float = 60,
                   include_native_state: bool = False) -> dict[str, Any]:
    """Produce a JSON-serializable structural audit, never changing input files.

    Actual playback spans are not inferred from source positions.  With matched
    single-file CUE chapters, playback-span *bounds* use chapter interval plus
    outgoing overlap.  Those bounds are conditional on the renderer staying
    inside declared tempo envelopes; native synthesis is not reproduced here.
    Volume/EQ envelopes can make parts of that span effectively inaudible.
    """
    project = load_djs(project) if not isinstance(project, ArchiveProject) else project
    data = project.mixdata
    mixes = data.get("mix", [])
    library = {a["key"]: a for a in data.get("audio", []) if "key" in a}
    if minimum_track_s <= 0 or not math.isfinite(minimum_track_s):
        raise ValueError("minimum_track_s must be finite and positive")
    if export_duration_s is not None and (_finite_number(export_duration_s) is None or export_duration_s <= 0):
        raise ValueError("export_duration_s must be finite and positive")
    cue_data = read_cue(cue) if isinstance(cue, (str, Path)) else copy.deepcopy(cue)
    warnings = []
    tracks = []
    for i, mix in enumerate(mixes):
        audio = library.get(mix.get("libraryKey"), {})
        name = audio.get("tag", {}).get("title") or audio.get("name", "")
        start, out = _cue_beat(mix, 0), _cue_beat(mix, 1)
        tracks.append({"number": i + 1, "library_key": mix.get("libraryKey"), "title": name,
                       "artist": audio.get("tag", {}).get("artist", ""),
                       "source_bpm": _finite_number(audio.get("bpm")),
                       "source_file_duration_s": _finite_number(audio.get("duration")),
                       "entry_beat": start, "outgoing_transition_beat": out,
                       "source_entry_s": _source_time(mix, start),
                       "source_outgoing_transition_s": _source_time(mix, out),
                       "phrase_verified": False,
                       "phrase_metadata_present": bool(mix.get("phraseData")),
                       "playback_duration_s": None})
        if not audio:
            warnings.append(f"Track {i + 1} references an absent audio record")
    cue_tracks = cue_data.get("tracks", []) if cue_data else []
    cue_order = len(cue_tracks) == len(tracks) and all(
        _title(a["title"]) == _title(b["title"]) and _title(a["title"])
        for a, b in zip(tracks, cue_tracks))
    cue_timing_usable = bool(cue_order and len(cue_data.get("files", [])) == 1)
    if cue_data and not cue_order:
        warnings.append("CUE chapter titles/count do not match native track order; timing not attached")
    if cue_data and len(cue_data.get("files", [])) != 1:
        warnings.append("Multi-file CUE positions are file-local; one export timeline is unavailable")
    if cue_timing_usable and export_duration_s is not None and export_duration_s <= cue_tracks[-1]["index01_s"]:
        raise ArchiveError("Export duration is not later than the final CUE chapter")

    global_tempos = [_finite_number(data.get("minBpm")), _finite_number(data.get("maxBpm"))]
    global_tempos = [v for v in global_tempos if v is not None and v > 0]
    if len(global_tempos) != 2:
        global_tempos = []
    transitions = []
    auto_effects = data.get("autoEffects", [])
    for i in range(max(0, len(mixes) - 1)):
        effects = [f for f in auto_effects if str(f.get("mixTrackKey")) == str(i)
                   and not f.get("isTrackEffect", False)]
        effect = effects[0] if len(effects) == 1 else None
        beats = _finite_number(effect.get("duration")) if effect else None
        supported = bool(effect and beats is not None and beats >= 0
                         and effect.get("mixTrackAnchor", 1) == 1
                         and not effect.get("isDisabled", False)
                         and _finite_number(effect.get("effectOffset", 0)) == 0
                         and _finite_number(effect.get("tempoMultiplier", 1)) == 1
                         and effect.get("bpmLineMode", 0) == 0
                         and not data.get("isCanvasMode", False)
                         and not data.get("mixBpmManualMode", False)
                         and not data.get("mixBpmFixedMode", False))
        meter_out = _finite_number(library.get(mixes[i].get("libraryKey"), {}).get("beatsPerBar"))
        meter_in = _finite_number(library.get(mixes[i + 1].get("libraryKey"), {}).get("beatsPerBar"))
        meter = meter_out if meter_out is not None and meter_out > 0 and meter_out == meter_in else None
        tempos = _tempo_values(library.get(mixes[i].get("libraryKey"), {}), mixes[i])
        tempos += _tempo_values(library.get(mixes[i + 1].get("libraryKey"), {}), mixes[i + 1])
        overlap = _bounds(beats, global_tempos) if supported else None
        local_overlap = _bounds(beats, tempos) if supported else None
        start = cue_tracks[i + 1]["index01_s"] if cue_timing_usable else None
        transition = {"number": i + 1, "from_track": i + 1, "to_track": i + 2,
                      "beats": beats, "declared_meter": meter,
                      "bars": beats / meter if beats is not None and meter else None,
                      "kind": "overlap" if supported and beats > 0 else "cut" if supported else "unknown",
                      "export_start_s": start, "export_start_is_cue_marker": start is not None,
                      "cue_marker_entry_alignment_verified": False,
                      "overlap_duration_s": None, "overlap_duration_bounds_s": overlap,
                      "local_tempo_overlap_bounds_s": local_overlap,
                      "timing_status": "conditional_tempo_envelope_bounds" if overlap else "unresolved",
                      "phrase_verified": False,
                      "native_effect": copy.deepcopy(effect) if effect else None}
        if effect is None:
            warnings.append(f"Transition {i + 1} has missing or ambiguous native effects")
        if start is not None and overlap:
            transition["export_end_bounds_s"] = [start + t for t in overlap]
        outgoing = _cue_beat(mixes[i], 1)
        source_end = _source_time(mixes[i], outgoing + beats) if outgoing is not None and beats is not None else None
        tracks[i]["source_selected_end_s"] = source_end
        if source_end is not None and tracks[i]["source_entry_s"] is not None:
            tracks[i]["source_selected_span_s"] = source_end - tracks[i]["source_entry_s"]
        transitions.append(transition)

    if tracks:
        tracks[-1]["source_selected_end_s"] = tracks[-1]["source_outgoing_transition_s"]
        if tracks[-1]["source_entry_s"] is not None and tracks[-1]["source_selected_end_s"] is not None:
            tracks[-1]["source_selected_span_s"] = tracks[-1]["source_selected_end_s"] - tracks[-1]["source_entry_s"]
    for i, track in enumerate(tracks):
        bound = None
        local_bound = None
        if cue_timing_usable:
            start = cue_tracks[i]["index01_s"]
            track["export_chapter_start_s"] = start
            next_start = cue_tracks[i + 1]["index01_s"] if i + 1 < len(tracks) else export_duration_s
            chapter_interval = next_start - start if next_start is not None else None
            track["export_chapter_interval_s"] = chapter_interval
            if chapter_interval is not None:
                if i + 1 == len(tracks):
                    bound = [chapter_interval, chapter_interval]
                    track["playback_duration_s"] = chapter_interval
                    track["timing_status"] = "final_chapter_to_export_end_not_audibility_verified"
                else:
                    overlap = transitions[i]["overlap_duration_bounds_s"]
                    local_overlap = transitions[i]["local_tempo_overlap_bounds_s"]
                    if overlap:
                        bound = [chapter_interval + s for s in overlap]
                        track["timing_status"] = "conditional_tempo_envelope_bounds"
                    if local_overlap:
                        local_bound = [chapter_interval + s for s in local_overlap]
        track["playback_duration_bounds_s"] = bound
        track["local_tempo_playback_duration_bounds_s"] = local_bound
        source_end = track.get("source_selected_end_s")
        source_duration = track["source_file_duration_s"]
        if source_end is not None and source_duration is not None:
            track["source_end_beyond_file_s"] = max(0.0, source_end - source_duration)
        else:
            track["source_end_beyond_file_s"] = None
        if not bound:
            status = "unresolved"
        elif bound[1] < minimum_track_s:
            status = "below_minimum_under_timing_assumptions"
        elif bound[0] >= minimum_track_s:
            status = "at_least_minimum_under_timing_assumptions"
        else:
            status = "uncertain"
        track["minimum_duration"] = {"required_s": minimum_track_s, "status": status,
                                     "audibility_verified": False}
    report = {"adapter_schema": 1, "adapter_mode": "read_only_audit", "name": data.get("name", ""),
              "track_count": len(tracks), "transition_count": len(transitions),
              "declared_track_count": data.get("trackCount"),
              "project_duration_metadata_s": _finite_number(data.get("duration")),
              "export_duration_s": export_duration_s, "cue_order_matches": bool(cue_order),
              "cue_timing_usable": cue_timing_usable, "tracks": tracks, "transitions": transitions,
              "jingles": copy.deepcopy(data.get("jingles", [])), "warnings": warnings,
              "limitations": [
                  "Source mixMap positions are not playback durations.",
                  "CUE chapters are export markers; chapter intervals exclude outgoing overlap.",
                  "CUE markers are assumed to mark native incoming entries; that correspondence is not independently verified.",
                  "Tempo bounds assume native playback stays within declared envelopes; interpolation is not reproduced.",
                  "An opaque native effect is preserved, not rendered or audibly validated.",
                  "Volume, EQ, stems and fades can make selected spans effectively inaudible.",
                  "Beat/bar metadata does not establish reviewed musical phrase boundaries.",
                  "Metadata reading does not verify every embedded audio member or successful application import.",
              ],
              "metadata_sha256": project.metadata_sha256,
              "native_state_included": include_native_state}
    if include_native_state:
        report["native_state"] = project.native_state()
    return report
