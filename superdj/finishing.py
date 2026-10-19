"""Independent master checks and export, published only after both succeed."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import os
import shutil
import tempfile


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _promote(items, directory, overwrite):
    """Preserve existing outputs and roll back our own partial promotion."""
    backup = directory / 'previous'
    backup.mkdir()
    originals = {}
    written = {}
    try:
        for source, target, audio in items:
            if target.exists():
                if audio and not overwrite:
                    raise FileExistsError('Output already exists; choose a new name or explicitly enable overwrite')
                saved = backup / target.name
                try:
                    os.link(target, saved)
                except OSError:
                    shutil.copy2(target, saved)
                originals[target] = saved
                # Same-volume replacement keeps each verified file atomic.
                os.replace(source, target)
            else:
                try:
                    if os.name == 'nt':
                        # Windows rename is atomic and refuses an existing
                        # destination, without retaining a second live name.
                        os.rename(source, target)
                    else:
                        os.link(source, target)
                except FileExistsError:
                    raise
                except OSError:
                    from .delivery import copy_verified
                    copy_verified(source, target)
            written[target] = _identity(target)
    except BaseException:
        for target, identity in reversed(list(written.items())):
            if _identity(target) != identity:
                raise RuntimeError('Concurrent output edit preserved during export rollback')
            if target in originals:
                os.replace(originals[target], target)
            else:
                target.unlink()
        raise


def finish_verified(reference, master, rendered, output, name, *, ffmpeg='ffmpeg',
                    ffprobe='ffprobe', mp3=True, overwrite=False, parallel=None):
    from .storage import private_path
    from .integrity import verify_master_rhythm
    from .quality import export_audio
    from .execution import current, phase
    from .resources import available_memory
    output = private_path(output)
    output.mkdir(parents=True, exist_ok=True)
    extensions = ('flac', 'mp3') if mp3 else ('flac',)
    if not overwrite and any((output / f'{name}.{ext}').exists() for ext in extensions):
        raise FileExistsError('Output already exists; choose a new name or explicitly enable overwrite')
    session = current()
    memory = available_memory()
    longest = max((edge.get('overlap_frames', 0) for edge in rendered.get('transitions', [])), default=0)
    required = max(512*1024**2, (longest+2*rendered.get('sample_rate', 0))*2*8*6)
    admitted = ((os.cpu_count() or 1) >= 2 and memory is not None and memory >= required
                and (session is None or session.cpu_jobs >= 2))
    parallel = admitted if parallel is None else bool(parallel) and admitted
    with tempfile.TemporaryDirectory(prefix='.superdj-export-', dir=output) as temporary:
        stage = Path(temporary)
        def rhythm():
            with phase('master_rhythm_check_wall'):
                return verify_master_rhythm(reference, master, rendered)
        def encode():
            with phase('export_and_checks_wall'):
                return export_audio(master, stage, name, ffmpeg=ffmpeg, ffprobe=ffprobe, mp3=mp3)
        if parallel:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(copy_context().run, encode)
                try:
                    checked = rhythm()
                    exports = future.result()
                except BaseException:
                    future.cancel()
                    raise
            if session:
                session.count('parallel_master_check_exports')
        else:
            checked = rhythm()
            exports = encode()
        if checked.get('verified') is not True:
            raise ValueError('Master rhythm verification did not pass')
        items = []
        for ext in extensions:
            source = Path(exports[ext]['path'])
            if source.resolve().parent != stage.resolve() or not source.is_file():
                raise ValueError('Exporter did not produce a verified staged file')
            items.append((source, output / f'{name}.{ext}', True))
            log = stage / f'{name}_{ext}_qa.log'
            if log.exists():
                items.append((log, output / log.name, False))
        _promote(items, stage, overwrite)
        for ext in extensions:
            exports[ext]['path'] = str((output / f'{name}.{ext}').resolve())
        return checked, exports
