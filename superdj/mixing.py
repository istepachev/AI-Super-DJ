"""Independent frequency envelopes with a gentle, uncompensated mid entry."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def smooth(value):
    x = np.clip(value, 0, 1)
    return x * x * (3 - 2 * x)


def split_bands(audio, sample_rate, bass_hz=180.0, high_hz=3500.0):
    """Complementary zero-phase decomposition; bands sum to the source."""
    low = sosfiltfilt(butter(4, bass_hz, fs=sample_rate, output="sos"), audio, axis=0)
    below_high = sosfiltfilt(butter(4, high_hz, fs=sample_rate, output="sos"), audio, axis=0)
    return low, below_high - low, audio - below_high


def blend(outgoing, incoming, sample_rate, *, bass_hz=180.0,
          beats=64, tonal_conflict=False):
    if outgoing.shape != incoming.shape or outgoing.ndim != 2 or outgoing.shape[1] != 2:
        raise ValueError("Frequency blend needs equally sized stereo blocks")
    if not np.isfinite(outgoing).all() or not np.isfinite(incoming).all():
        raise ValueError("Nonfinite overlap audio")
    n = len(incoming)
    if n < 2:
        return incoming.copy(), {"method": "short endpoint"}
    position = np.linspace(0., 1., n)[:, None]
    out_gain, in_gain = np.cos(position * np.pi / 2), np.sin(position * np.pi / 2)
    if bass_hz <= 0 or n < 128:
        return (outgoing * out_gain + incoming * in_gain).astype(np.float32), {"method": "unverified equal-power fade"}
    low_a, mid_a, high_a = split_bands(outgoing, sample_rate, bass_hz)
    low_b, mid_b, high_b = split_bands(incoming, sample_rate, bass_hz)
    bass_swap = smooth((position - .35) / .30)
    long = beats >= 160
    entry_beats = min(beats, 32 if long and tonal_conflict else 16 if long else 8)
    initial_mid = .30 if long and tonal_conflict else .40 if long else .55
    opening = smooth(position * beats / entry_beats)
    mid_in = in_gain * (initial_mid + (1 - initial_mid) * opening)
    # A risk flag is not a vocal detection result. Reduce competing midrange,
    # and explicitly retain that distinction in the returned manifest.
    mid_out = out_gain * (1 - (.65 * smooth((position - .20) / .40) if tonal_conflict else 0))
    # Relative cap is always attenuation; no per-overlap normalization follows.
    quiet_db = 2 if long and tonal_conflict else 1 if long else 0
    quiet = 10 ** ((-quiet_db * (1 - opening)) / 20)
    mixed = low_a * (1 - bass_swap) + low_b * bass_swap
    mixed += mid_a * mid_out + mid_b * mid_in * quiet
    mixed += high_a * out_gain + high_b * in_gain * quiet
    report = {"method": "three complementary zero-phase bands with independent bass/mid/high envelopes",
              "bass_hz": bass_hz, "high_hz": 3500., "beats": beats,
              "incoming_mid_start_multiplier": initial_mid, "incoming_mid_opening_beats": entry_beats,
              "incoming_initial_attenuation_db": quiet_db,
              "tonal_conflict_risk": bool(tonal_conflict), "vocal_overlap_certified_absent": False,
              "entry_gain_compensated": False, "bass_handoff_fraction": [.35, .65],
              "endpoint_max_error": float(max(np.max(np.abs(mixed[0] - outgoing[0])), np.max(np.abs(mixed[-1] - incoming[-1]))))}
    return mixed.astype(np.float32), report
