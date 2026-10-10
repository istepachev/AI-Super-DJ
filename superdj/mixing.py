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


def envelopes(n, beats, *, tonal_conflict=False, beat_phase=None):
    """Complementary per-band transfer, with a brief soft mid handoff.

    Both bass lines and dense mid parties are never held at full level at once.
    Highs change over the whole transition. Use the renderer's actual shared
    beat phase when supplied, rather than approximating phase by elapsed time.
    """
    if beats <= 0 or n < 2:
        raise ValueError("A transition needs positive beats and at least two samples")
    p = np.linspace(0., 1., n) if beat_phase is None else np.asarray(beat_phase, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or np.any(np.diff(p) < 0) or abs(p[0]) > 1e-9 or abs(p[-1] - 1) > 1e-9:
        raise ValueError("Envelope phase must run monotonically from zero to one")
    p = p[:, None]
    mid_beats = min(beats * .4, 8 if tonal_conflict else 16)
    mid_width = mid_beats / beats
    mid = smooth((p - (.43 - mid_width / 2)) / mid_width)
    bass_beats = min(4, beats * .2)
    bass = smooth((p - (.55 - bass_beats / beats / 2)) / (bass_beats / beats))
    high = smooth(p)
    quiet_db = 2 if tonal_conflict else 1
    quiet = 10 ** (-quiet_db * (1 - mid) / 20)
    gains = {'bass_out': 1 - bass, 'bass_in': bass,
             'mid_out': 1 - mid, 'mid_in': mid * quiet,
             'high_out': 1 - high, 'high_in': high}
    report = {'incoming_mid_opening_beats': mid_beats,
              'incoming_initial_attenuation_db': quiet_db,
              'bass_handoff_beats': bass_beats,
              'mid_handoff_fraction': [.43 - mid_width / 2, .43 + mid_width / 2],
              'bass_handoff_fraction': [.55 - bass_beats / beats / 2, .55 + bass_beats / beats / 2],
              'high_handoff_fraction': [0., 1.],
              'band_pair_gain_sum_maximum': {band: float(np.max(gains[band + '_out'] + gains[band + '_in']))
                                             for band in ('bass', 'mid', 'high')},
              'phase_source': 'shared integrated clock' if beat_phase is not None else 'linear phase'}
    return gains, report


def blend(outgoing, incoming, sample_rate, *, bass_hz=180.0,
          beats=64, tonal_conflict=False, beat_phase=None):
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
    gains, curves = envelopes(n, beats, tonal_conflict=tonal_conflict, beat_phase=beat_phase)
    mixed = low_a * gains['bass_out'] + low_b * gains['bass_in']
    mixed += mid_a * gains['mid_out'] + mid_b * gains['mid_in']
    mixed += high_a * gains['high_out'] + high_b * gains['high_in']
    report = {"method": "three complementary zero-phase bands with independent bass/mid/high envelopes",
              "bass_hz": bass_hz, "high_hz": 3500., "beats": beats,
              **curves, "incoming_mid_start_multiplier": 0.,
              "tonal_conflict_risk": bool(tonal_conflict), "vocal_overlap_certified_absent": False,
              "entry_gain_compensated": False,
              "endpoint_max_error": float(max(np.max(np.abs(mixed[0] - outgoing[0])), np.max(np.abs(mixed[-1] - incoming[-1]))))}
    return mixed.astype(np.float32), report
