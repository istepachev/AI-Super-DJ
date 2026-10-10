"""Measured excerpt diagnostics for listening review, never a taste predictor."""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


def excerpt_audit(path, start_seconds: float, end_seconds: float) -> dict:
    """Compare an actual overlap with its preceding/following four seconds.

    One-second RMS limits confusion between an intentional beat gap and a loss
    of sustained energy. These measures cannot rank subjective musical quality,
    identify drum stems or prove rhythm/phrase alignment.
    """
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        if not 4 <= start_seconds < end_seconds < f.frames / sr - 4:
            raise ValueError("Audit needs four seconds of context on both sides")
        first = round((start_seconds - 4) * sr)
        f.seek(first)
        audio = f.read(round((end_seconds - start_seconds + 8) * sr), always_2d=True)
    if not np.all(np.isfinite(audio)):
        raise ValueError("Nonfinite audio")
    bass = sosfiltfilt(butter(4, (35, 180), btype="bandpass", fs=sr, output="sos"), audio, axis=0)
    start, end = 4 * sr, round((end_seconds - start_seconds + 4) * sr)

    def energy(x):
        return 10 * np.log10(max(float(np.mean(x**2)), 1e-20))

    def band(x):
        windows = [(i, energy(x[i:i+sr])) for i in range(start, end-sr+1, round(.1*sr))]
        minimum = min(windows, key=lambda p: p[1])
        before, after = energy(x[:start]), energy(x[end:end+4*sr])
        return {"before_dbfs": before, "overlap_dbfs": energy(x[start:end]),
                "after_dbfs": after, "minimum_one_second_dbfs": minimum[1],
                "minimum_window_start_seconds": (first + minimum[0]) / sr,
                "minimum_relative_to_preceding_db": minimum[1] - before}

    return {"path": str(Path(path).resolve()), "transition_start_seconds": start_seconds,
            "transition_end_seconds": end_seconds, "overlap_seconds": end_seconds-start_seconds,
            "full_band": band(audio), "bass_35_180_hz": band(bass),
            "sample_peak_dbfs": 20*np.log10(max(float(np.max(np.abs(audio))), 1e-10)),
            "taste_predicted": False, "human_auditioned_by_agent": False,
            "limitations": ["Bass-band energy includes other instruments, not isolated kicks",
                            "RMS energy does not certify danceability or subjective preference",
                            "Different source cues confound a technique-only comparison"]}
