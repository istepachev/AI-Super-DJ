"""Scoped runtime field updates; never roll back a whole workstation config."""
from contextlib import contextmanager
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import tempfile
import time

from .resource_limits import LimitSettings, runtime_path, settings


FIELD = 'resource_limits'


@contextmanager
def _lock(path, timeout=10.):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b'0'); handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Another resource-mode update still owns the config lock')
                time.sleep(.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(path):
    return path.read_bytes() if path.exists() else None


def _atomic(path, value, *, expected=None, compare=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.rd-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf8', newline='\n') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        if compare and _read(path) != expected:
            raise RuntimeError('Runtime config changed concurrently; no runtime fields were overwritten')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _field(runtime):
    return {'present': FIELD in runtime, 'value': copy.deepcopy(runtime.get(FIELD))}


def configure(action='status', *, path=None):
    """One-step field-only undo; unrelated new edits always survive.

    This command affects future runs. It neither changes a running build nor
    removes native limits inherited from the OS or a separate resource guard.
    """
    if action not in ('status', 'on', 'off', 'restore'):
        raise ValueError('Use status, on, off or restore')
    path = Path(path) if path is not None else runtime_path()
    backup = path.with_name('resource-mode-rollback.json')
    with _lock(path.with_name('resource-mode.lock')):
        raw = _read(path)
        runtime = json.loads(raw.decode('utf-8-sig')) if raw is not None else {}
        if not isinstance(runtime, dict):
            raise ValueError('Runtime config must be a JSON object')
        before = _field(runtime)
        if action == 'status':
            rollback = False
            if backup.exists():
                try:
                    record = json.loads(backup.read_text(encoding='utf8'))
                    rollback = not record.get('restored') and record.get('after') == before
                except (ValueError, OSError):
                    pass
            return {'resource_limits': asdict(settings(runtime.get(FIELD))),
                    'rollback_available': rollback,
                    'applies_to': 'future runs; existing builds retain their policy'}
        if action == 'restore':
            if not backup.is_file():
                raise ValueError('No resource-mode update is available to restore')
            record = json.loads(backup.read_text(encoding='utf8'))
            if record.get('restored'):
                raise ValueError('The last resource-mode update was already restored')
            if before != record['after']:
                raise RuntimeError('The resource_limits field was edited after this update; refusing to overwrite it')
            previous = record['before']
            if previous['present']:
                runtime[FIELD] = previous['value']
            else:
                runtime.pop(FIELD, None)
            _atomic(path, runtime, expected=raw, compare=True)
            record['restored'] = True
            _atomic(backup, record)
            return {'action': action, 'changed': True, 'future_runs_only': True}
        if action == 'on':
            # Keep any explicitly tuned valid limits; apply defaults on first enable.
            existing = runtime.get(FIELD)
            configured = dict(existing) if isinstance(existing, dict) else asdict(LimitSettings())
            configured['enabled'] = True
            after_value = asdict(settings(configured))
        else:
            existing = runtime.get(FIELD)
            after_value = copy.deepcopy(existing) if isinstance(existing, dict) else {}
            after_value['enabled'] = False
        runtime[FIELD] = after_value
        after = _field(runtime)
        if before == after:
            return {'action': action, 'changed': False, 'future_runs_only': True}
        _atomic(backup, {'before': before, 'after': after, 'restored': False})
        _atomic(path, runtime, expected=raw, compare=True)
        return {'action': action, 'changed': True, 'future_runs_only': True,
                'rollback': 'superdj resource-mode restore'}
