"""Small private JSON and reproducible track-selection helpers."""
from __future__ import annotations
import json
from pathlib import Path
import random
from typing import Any

MEDIA = {".flac", ".mp3", ".wav", ".aif", ".aiff", ".m4a", ".ogg", ".opus"}


def private_path(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if target == repository or repository in target.parents:
        raise ValueError("Store music and private run data outside the source repository")
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = private_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def select_tracks(folder: str | Path, count: int, seed: int, *, history=None, allow_repeats=False) -> dict:
    source = Path(folder).expanduser().resolve(strict=True)
    files = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA), key=lambda p: str(p).casefold())
    excluded=[]
    if not allow_repeats:
        from .history import unseen_files
        files,excluded=unseen_files(files,history)
    if not 1 <= count <= len(files):
        raise ValueError(f"Requested {count} tracks from {len(files)} available unseen files; excluded {len(excluded)} previously used files")
    chosen = random.Random(seed).sample(files, count)
    return {"source": str(source), "seed": seed, "tracks": [{"id": f"track_{i:02}", "path": str(p)} for i, p in enumerate(chosen, 1)],
            'history_policy':'explicit repeat allowance' if allow_repeats else 'exclude completed-set sources',
            'excluded_previous_sources':excluded}
