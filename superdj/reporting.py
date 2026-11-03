"""Preserve earlier private performance reports before publishing a new one."""
from pathlib import Path
import hashlib
import os
import tempfile
from uuid import uuid4


def archive_existing_report(path):
    from .storage import private_path
    path=private_path(path)
    if not path.exists():return None
    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    folder=path.parent/'_performance_history'/path.stem
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/(digest+'.json')
    if target.exists():
        if target.read_bytes()==data:return target
        # Keep an edited/damaged previous archive too; never overwrite it.
        target=folder/(digest+'-'+uuid4().hex+'.json')
    fd,name=tempfile.mkstemp(prefix='pending-',dir=folder);os.close(fd)
    temporary=Path(name)
    try:
        temporary.write_bytes(data)
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    return target


def write_performance(path,metrics):
    from .storage import write_json
    archive_existing_report(path)
    write_json(path,metrics)
