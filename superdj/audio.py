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
    result = _run(_command(source, sample_rate, 1, ffmpeg) + ['pipe:1'], source)
    if not result.stdout or len(result.stdout) % 4:
        raise ValueError(f'Decoder returned no complete audio frames: {Path(source).name}')
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not np.isfinite(audio).all():
        raise ValueError(f'Decoder returned nonfinite audio: {Path(source).name}')
    return audio
