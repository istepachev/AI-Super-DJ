"""Conservative content-dependent EQ suggestions, never detail synthesis."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def settings(samples, rate):
    """Measure twelve distributed 2s windows; cap any broad correction at 0.8dB.

    These are balancing heuristics, not diagnoses of bad artistic mastering.
    Clean controls bypass EQ; absent treble is never artificially generated.
    """
    if samples.ndim != 2 or samples.shape[0] not in (1, 2):
        raise ValueError("Tone analysis expects channels × samples")
    n = samples.shape[1]
    if n < 2 or rate <= 0 or not np.isfinite(samples).all():
        raise ValueError("Tone analysis requires finite, nonempty audio and a positive sample rate")
    width = min(n, 2 * rate)
    starts = np.unique(np.linspace(0, max(0, n - width), 12).astype(int))
    power = None
    for start in starts:
        freq, observed = welch(samples[:, start:start + width], fs=rate, nperseg=min(4096, width), axis=1)
        observed = observed.mean(axis=0)
        power = observed if power is None else power + observed
    total = max(float(power.sum()), 1e-20)
    def fraction(low, high):
        return float(power[(freq >= low) & (freq < high)].sum() / total)
    centroid = float(np.sum(freq * power) / total)
    report = {"scope": f"{len(starts)} distributed windows, up to 2 seconds each", "centroid_hz": centroid,
              "lowmid_fraction": fraction(180, 600), "presence_fraction": fraction(3500, 8000),
              "subsonic_fraction": fraction(0, 18), "max_eq_correction_db": .8,
              "kind": "bounded tonal-balancing heuristic; not lost-detail restoration"}
    eq = {}
    if report["subsonic_fraction"] > .05:
        eq["highpass_hz"] = 18.
    if report["presence_fraction"] > .55 and centroid > 3500 and rate >= 16000:
        eq.update(peak_db=-.8, peak_hz=4500., peak_q=.7)
    elif report["lowmid_fraction"] > .55 and centroid < 1200:
        eq.update(peak_db=-.6, peak_hz=300., peak_q=.7)
    report["suggested_eq"] = eq
    return eq, report
