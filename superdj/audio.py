"""One codec-independent input decoder for the main process and neural workers.

Use the workstation's existing FFmpeg explicitly. Reading an input must not
depend on libsndfile's codec build, audioread, or a worker's PATH.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

AUDIO_EXTENSIONS = frozenset({
    '.flac', '.mp3', '.mp2', '.mp1', '.wav', '.wave', '.bwf', '.rf64', '.w64',
    '.aif', '.aiff', '.aifc', '.m4a', '.m4b', '.aac', '.mp4', '.ogg', '.oga',
    '.opus', '.wma', '.ape', '.wv', '.tta', '.caf', '.au', '.snd', '.amr',
    '.ac3', '.eac3', '.mka', '.webm', '.dsf', '.dff',
})
DECODER_VERSION = 'ffmpeg-f32-first-audio-v1'


def ffmpeg_executable(value: str | Path | None = None) -> str:
    """Explicit caller path, configured runtime, then the system executable."""
    if value is not None and str(value) not in {'ffmpeg', 'ffmpeg.exe'}:
        selected = str(value)
    else:
        selected = os.environ.get('SUPERDJ_FFMPEG')
        if not selected:
            from .cues import runtime_config
            selected = runtime_config().get('ffmpeg')
        selected = str(selected or value or 'ffmpeg')
    candidate = Path(selected).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    executable = shutil.which(selected)
    if executable:
        return executable
    raise FileNotFoundError(f'The configured FFmpeg decoder is unavailable: {selected}')


def _command(source, sample_rate, channels, ffmpeg):
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError('An audio file is required')
    if not isinstance(sample_rate, int) or sample_rate <= 0 or not isinstance(channels, int) or channels <= 0:
        raise ValueError('Decode sample rate and channel count must be positive integers')
    return [ffmpeg_executable(ffmpeg), '-nostdin', '-hide_banner', '-v', 'error',
            '-xerror', '-i', str(source), '-map', '0:a:0', '-vn', '-ac', str(channels),
            '-ar', str(sample_rate), '-c:a', 'pcm_f32le', '-f', 'f32le']


def _run(command, source):
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace')[-2000:].strip()
        raise RuntimeError(f'Cannot decode audio {Path(source).name!r}: {detail}')
    return result


def decode_f32_file(source, destination, *, sample_rate=22050, channels=1, ffmpeg=None) -> Path:
    """Decode the actual container to raw float PCM, without loading it into RAM."""
    destination = Path(destination)
    original = Path(source).expanduser().resolve(strict=True)
    if destination.resolve() == original or (destination.exists() and original.samefile(destination)):
        raise ValueError('A decode destination must not overwrite the original audio')
    if destination.exists() and destination.stat().st_size:
        raise FileExistsError('A decode destination must be new or an empty temporary file')
    try:
        from .execution import current
        if current():
            shared = decoded_artifact(source, sample_rate=sample_rate, channels=channels, ffmpeg=ffmpeg)
            shutil.copyfile(shared['path'], destination)
        else:
            _run(_command(source, sample_rate, channels, ffmpeg) + ['-y', str(destination)], source)
        size = destination.stat().st_size
        if not size or size % (4 * channels):
            raise ValueError(f'Decoder returned no complete audio frames: {Path(source).name}')
        return destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def decode_mono(source, *, sample_rate=22050, ffmpeg=None) -> np.ndarray:
    """Load mono model input through the same decoder used for full-track analysis."""
    from .execution import current
    if current():
        descriptor = decoded_artifact(source, sample_rate=sample_rate, channels=1, ffmpeg=ffmpeg)
        return read_decoded(descriptor, sample_rate=sample_rate, channels=1)
    result = _run(_command(source, sample_rate, 1, ffmpeg) + ['pipe:1'], source)
    if not result.stdout or len(result.stdout) % 4:
        raise ValueError(f'Decoder returned no complete audio frames: {Path(source).name}')
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not np.isfinite(audio).all():
        raise ValueError(f'Decoder returned nonfinite audio: {Path(source).name}')
    return audio


def decoded_artifact(source, *, sample_rate=22050, channels=1, ffmpeg=None):
    """One exact decode per content/format, also shared with model subprocesses."""
    from .execution import current, key_for, cached, sha256
    import threading
    session = current()
    if session is None:
        raise RuntimeError('Shared decoding requires a run-owned execution session')
    command = _command(source, sample_rate, channels, ffmpeg)
    dependencies = {'source': str(source), 'rate': sample_rate, 'channels': channels,
                    'decoder': command[0], 'format': DECODER_VERSION}
    key = key_for(dependencies)
    with session.lock:
        lock = session.decode_locks.setdefault(key, threading.Lock())
    with lock:
        descriptor = session.memo.get(('decode', key))
        if descriptor:
            try:
                valid = sha256(descriptor['path']) == descriptor['sha256']
            except OSError:
                valid = False
            if valid:
                session.count('decode_reuses')
                return dict(descriptor)
        def produce():
            directory = session.work / 'decoded'
            directory.mkdir(exist_ok=True)
            path = directory / (key + '.f32')
            _run(command + ['-y', str(path)], source)
            size = path.stat().st_size
            if not size or size % (4 * channels):
                raise ValueError('Decoder returned no complete audio frames')
            session.count('source_decodes')
            return {'path': str(path), 'sha256': sha256(path), 'sample_rate': sample_rate,
                    'channels': channels, 'frames': size // (4 * channels), 'format': DECODER_VERSION}
        descriptor = cached('decode', dependencies, produce, session.work)
        session.memo[('decode', key)] = descriptor
        return dict(descriptor)


def read_decoded(descriptor, *, sample_rate, channels):
    """Validate shared PCM before a model uses it; never silently resample it."""
    import hashlib
    if (descriptor.get('format') != DECODER_VERSION or descriptor.get('sample_rate') != sample_rate
            or descriptor.get('channels') != channels):
        raise ValueError('Shared PCM does not match the model input format')
    path = Path(descriptor['path'])
    samples = descriptor['frames'] * channels
    # Hash the very bytes the model will consume, rather than reading the same
    # immutable input once for its digest and again for its array. Keep an owned
    # writable array: inference must not retain a mapping or an open input file.
    with path.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size != samples * 4:
            raise ValueError('Shared PCM failed its integrity check')
        audio = np.fromfile(stream, dtype='<f4', count=samples)
        if (audio.size != samples or stream.read(1) or
                hashlib.sha256(memoryview(audio).cast('B')).hexdigest() != descriptor['sha256']):
            raise ValueError('Shared PCM failed its integrity check')
    audio = audio.reshape(-1, channels)
    if not np.isfinite(audio).all():
        raise ValueError('Shared PCM contains nonfinite audio')
    return audio[:, 0].copy() if channels == 1 else audio


def decode_wave_file(source, destination, *, sample_rate, channels=2, ffmpeg=None):
    """Materialize the same float samples in a WAV needed by external engines."""
    import soundfile as sf
    from .execution import current
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if Path(source).resolve() == destination.resolve():
        raise ValueError('A decode destination must not overwrite original audio')
    if current() is None:
        command = _command(source, sample_rate, channels, ffmpeg)[:-2]
        _run(command + ['-rf64', 'auto', '-y', str(destination)], source)
        return destination
    descriptor = decoded_artifact(source, sample_rate=sample_rate, channels=channels, ffmpeg=ffmpeg)
    container = 'RF64' if descriptor['frames'] * channels * 4 + 4096 >= 2**32 else 'WAV'
    try:
        with Path(descriptor['path']).open('rb') as stream, sf.SoundFile(destination, 'w',
                samplerate=sample_rate, channels=channels, subtype='FLOAT', format=container) as output:
            while block := stream.read(65536 * channels * 4):
                audio = np.frombuffer(block, dtype='<f4').reshape(-1, channels)
                if not np.isfinite(audio).all():
                    raise ValueError('Decoder returned nonfinite audio')
                output.write(audio)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination
