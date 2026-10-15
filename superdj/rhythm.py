"""Measured bass availability for automatic transition-window selection."""
from pathlib import Path
import subprocess
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .cues import runtime_config


def bass_profile(record, *, ffmpeg=None):
    from .execution import current, execution, cached
    if current() is None:
        with execution():
            return bass_profile(record, ffmpeg=ffmpeg)
    decoder = ffmpeg or os.getenv('SUPERDJ_FFMPEG') or runtime_config().get('ffmpeg', 'ffmpeg')
    return cached('bass_profile', {'source': record.get('path'), 'beats': record.get('beats'),
        'duration': record.get('duration', record.get('duration_seconds')), 'decoder': decoder},
        lambda: _bass_profile(record, ffmpeg=decoder), current().work)


def _bass_profile(record, *, ffmpeg=None):
    """Beat-level 35–180Hz RMS; not a classifier or a perceived-energy score."""
    source = Path(record.get('path', ''))
    if not source.is_file():
        return {'executed': False, 'reason': 'source unavailable for bass evidence'}
    runtime = runtime_config()
    sr = 8000
    process = subprocess.run([str(ffmpeg or os.getenv('SUPERDJ_FFMPEG') or runtime.get('ffmpeg', 'ffmpeg')), '-nostdin', '-v', 'error',
        '-i', str(source.resolve()), '-vn', '-ac', '1', '-ar', str(sr), '-f', 'f32le', 'pipe:1'],
        capture_output=True, check=True)
    audio = np.frombuffer(process.stdout, dtype='<f4').astype(float)
    if len(audio) < sr or not np.isfinite(audio).all():
        raise ValueError('Invalid audio for bass-profile analysis')
    if abs(len(audio) / sr - float(record.get('duration', record.get('duration_seconds')))) > .05:
        raise ValueError('Source duration differs from the timing analysis')
    filtered = sosfiltfilt(butter(4, (35, 180), btype='bandpass', fs=sr, output='sos'), audio)
    bars = np.asarray(record['beats'])
    rows = []
    for i, time in enumerate(bars):
        end = float(bars[i + 1]) if i + 1 < len(bars) else len(filtered) / sr
        part = filtered[round(time * sr):round(end * sr)]
        db = 10 * np.log10(max(float(np.mean(part ** 2)), 1e-20)) if len(part) else -200.
        rows.append({'time': float(time), 'end': end, 'bass_dbfs': float(db)})
    reference = float(np.quantile([r['bass_dbfs'] for r in rows], .70))
    result = {'executed': True, 'method': 'complete source decoded to mono 8kHz; existing beat grid, 35–180Hz beat RMS',
              'reference_dbfs': reference, 'threshold_dbfs': reference - 16.,
              'source_sha256': (record.get('provenance') or {}).get('source_sha256'),
              'bass_presence_not_kick_classification': True, 'rows': rows}
    return result


def support(profile, start, end):
    """Fraction of source duration with sufficient bass relative to its core."""
    if not profile.get('executed') or profile['reference_dbfs'] < -60:
        return 1.  # A genuinely bassless source/control needs another judgment.
    covered, supported = 0., 0.
    for row in profile['rows']:
        duration = max(0., min(end, row['end']) - max(start, row['time']))
        covered += duration
        if row['bass_dbfs'] >= profile['threshold_dbfs']:
            supported += duration
    return supported / covered if covered else 0.


def weak_gap(profile, start, end):
    """Longest weak-bass run; distinguish a short fill from a sustained hole."""
    if not profile.get('executed') or profile['reference_dbfs'] < -60:
        return 0.
    longest, current = 0., 0.
    for row in profile['rows']:
        duration = max(0., min(end, row['end']) - max(start, row['time']))
        if not duration:
            continue
        if row['bass_dbfs'] < profile['threshold_dbfs']:
            current += duration
            longest = max(longest, current)
        else:
            current = 0.
    return longest
