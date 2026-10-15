"""Advanced preserving native-project patching; no application DB writes."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import zipfile

from .project_archive import ArchiveError, load_djs
from .storage import private_path


def rewrite_djs(source, output, *, cue_updates=None, transition_updates=None, name=None):
    """Patch only specified native cue/transition records in a new archive.

    Updates are keyed by zero-based record index. Unknown metadata and all
    embedded members survive. Raw audio bytes are checked by SHA-256, despite
    possible changes to their ZIP compression. No app import/render is claimed.
    """
    source = Path(source).resolve(strict=True)
    output = private_path(output)
    if source == output or output.exists():
        raise FileExistsError("Native project output must be a new file")
    project = load_djs(source)
    data = deepcopy(project.mixdata)
    for index, cue in (cue_updates or {}).items():
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(data["mix"]) or not isinstance(cue, dict):
            raise ArchiveError("Invalid cue record update")
        data["mix"][index].setdefault("cueData", {}).update(deepcopy(cue))
    for index, update in (transition_updates or {}).items():
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(data["autoEffects"]):
            raise ArchiveError("Invalid transition record update")
        allowed = {"duration", "primaryAutomation", "secondaryAutomation", "locked", "manual"}
        if not isinstance(update, dict) or set(update) - allowed:
            raise ArchiveError("Patch supports only explicit duration/curve/lock fields")
        if "duration" in update:
            duration = update["duration"]
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ArchiveError("Native transition duration must be positive and finite")
        if any(key in update and not isinstance(update[key], bool) for key in ("locked", "manual")):
            raise ArchiveError("Native lock/manual fields must be boolean")
        data["autoEffects"][index].update(deepcopy(update))
    if name is not None:
        data["name"] = str(name)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(".partial.djs")
    if staged.exists():
        raise FileExistsError(staged)
    hashes = {}
    try:
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(staged, "w", allowZip64=True) as new:
            if len(old.namelist()) != len(set(old.namelist())):
                raise ArchiveError("Duplicate archive members")
            for info in old.infolist():
                if info.filename == "mixdata.json":
                    new.writestr(deepcopy(info), payload)
                else:
                    digest = hashlib.sha256()
                    with old.open(info) as reader, new.open(deepcopy(info), "w", force_zip64=True) as writer:
                        while block := reader.read(1024 * 1024):
                            digest.update(block)
                            writer.write(block)
                    hashes[info.filename] = digest.hexdigest()
        reread = load_djs(staged)
        if reread.mixdata != data:
            raise ArchiveError("Native metadata roundtrip changed content")
        with zipfile.ZipFile(staged) as archive:
            for member, expected in hashes.items():
                digest = hashlib.sha256()
                with archive.open(member) as reader:
                    while block := reader.read(1024 * 1024):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise ArchiveError("Embedded member changed")
        staged.replace(output)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return {"path": str(output), "preserved_members": hashes,
            "cue_records_changed": sorted(cue_updates or {}),
            "transition_records_changed": sorted(transition_updates or {}),
            "application_import_verified": False}
