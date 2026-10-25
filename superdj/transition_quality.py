"""Internal transition evidence; these measurements are not taste probabilities."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

from .render import RenderError


class TransitionFailure(RenderError):
    def __init__(self, outgoing, incoming, reason, evidence=None):
        super().__init__(f'{outgoing} → {incoming}: {reason}')
        self.outgoing, self.incoming = outgoing, incoming
        self.evidence = evidence or {}


def _power(audio, frames):
    count = len(audio)//frames
    if not count:
        return np.empty(0)
    values = np.asarray(audio[:count*frames], dtype=np.float64).reshape(count, frames, -1)
    return np.mean(values*values, axis=(1, 2))


def _longest(mask, step=.5):
    longest = current = 0
    for value in mask:
        current = current+1 if value else 0
        longest = max(longest, current)
    return longest*step


def audit_energy(outgoing, incoming, mixed, sample_rate):
    if outgoing.shape != incoming.shape or mixed.shape != outgoing.shape:
        raise ValueError('Energy audit requires the actual aligned overlap shapes')
    if not all(np.isfinite(x).all() for x in (outgoing, incoming, mixed)):
        raise ValueError('Nonfinite transition audio')
    step = max(1, round(sample_rate*.5))
    powers = [_power(x, step) for x in (outgoing, incoming, mixed)]
    if not len(powers[0]):
        return {'passed': True, 'status': 'short window; energy coverage unavailable'}
    divisor = max(1, sample_rate//2000)
    rate = sample_rate/divisor
    bass = [_power(sosfiltfilt(butter(3, (35, 180), btype='bandpass', fs=rate, output='sos'),
        resample_poly(x, 1, divisor, axis=0), axis=0), round(rate*.5)) for x in (outgoing, incoming, mixed)]
    count = min(len(x) for x in powers+bass)
    a, b, actual = [x[:count] for x in powers]
    ba, bb, bm = [x[:count] for x in bass]
    reference = np.minimum(a, b)
    both_active = (a > max(np.quantile(a, .75)*.01, 1e-10)) & (b > max(np.quantile(b, .75)*.01, 1e-10))
    full_hole = both_active & (actual < reference*10**(-12/10))
    available_bass = np.maximum(ba, bb)
    bass_active = available_bass > max(float(np.quantile(available_bass, .75))*.05, 1e-10)
    phase = (np.arange(count)+.5)/count
    bass_hole = bass_active & (bm < available_bass*10**(-18/10)) & (phase>.25) & (phase<.9)
    full_seconds, bass_seconds = _longest(full_hole), _longest(bass_hole)
    return {'passed': full_seconds<2 and bass_seconds<2,
        'sustained_full_band_hole_seconds': full_seconds, 'sustained_bass_hole_seconds': bass_seconds,
        'native_quiet_both_seconds': float(np.count_nonzero(~both_active)*.5),
        'scope': 'sustained losses in actual PCM relative to available aligned source energy; not a musicality classifier'}


def audit_vocals(outgoing, incoming, mixed, sample_rate, vocal_out, vocal_in, transition, *, style, beat_phase):
    if vocal_out is None or vocal_in is None:
        return {'passed': True, 'status': 'unknown; both aligned estimates unavailable', 'absence_certified': False}
    from .mixing import envelopes, smooth
    divisor = max(1, sample_rate//4000)
    a, b = [resample_poly(x, 1, divisor, axis=0) for x in (vocal_out, vocal_in)]
    gain, _ = envelopes(len(mixed), transition.get('beats', 64), style=style, beat_phase=beat_phase,
        mid_handoff_bars=transition.get('mid_handoff_bars'), bass_handoff_beats=transition.get('bass_handoff_beats'),
        handoff_windows=transition.get('handoff_windows'),whole_spectrum_handoff=transition.get('whole_spectrum_handoff'))
    p = np.linspace(0, 1, len(mixed))
    keep_a = 1-smooth((p-.20)/.25) if transition.get('vocal_processing', {}).get('executed') else np.ones(len(p))
    keep_b = smooth((p-.55)/.25) if transition.get('vocal_processing', {}).get('executed') else np.ones(len(p))
    indices = np.minimum(np.arange(len(a))*divisor, len(p)-1)
    a *= (keep_a*gain['mid_out'][:, 0])[indices, None]
    b *= (keep_b*gain['mid_in'][:, 0])[indices, None]
    step = round(sample_rate/divisor*.5)
    pa, pb = _power(a, step), _power(b, step)
    pm = _power(resample_poly(mixed, 1, divisor, axis=0), step)
    count = min(len(pa), len(pb), len(pm))
    simultaneous = (pa[:count] > np.maximum(pm[:count]*.10, 1e-9)) & (pb[:count] > np.maximum(pm[:count]*.10, 1e-9))
    seconds = _longest(simultaneous)
    return {'passed': seconds<2, 'sustained_competing_estimates_seconds': seconds,
        'status': 'estimated foreground contribution after the actual handoff gains',
        'absence_certified': False, 'limitations': 'separation leakage and midband gain approximation; not speech recognition'}


def reliability(clips, transition):
    rows = []
    for clip, role in zip(clips, ('tail', 'head')):
        final = (clip.get('warp') or {}).get('post_render_transient_audit', {}).get('final', {})
        recognized = {p['beat'] for p in final.get('measured', []) if p.get('role') == role}
        unknown = {p['beat'] for p in final.get('unverified', []) if p.get('role') == role} - recognized
        total = len(recognized | unknown)
        rows.append({'id': clip['id'], 'role': role, 'recognized_beats': len(recognized),
            'unknown_beats': len(unknown), 'coverage_fraction': len(recognized)/total if total else None,
            'sufficient_for_automatic_window': not total or total<32 or len(recognized)>=max(8, .25*total)})
    return {'passed': all(row['sufficient_for_automatic_window'] for row in rows), 'decks': rows,
        'musical_quality_probability': None, 'human_report_required': False,
        'scope': 'recognition coverage, not certification of unknown figures'}


def coverage_failure_context(report, transition):
    """Attribute only an unambiguous failed coverage view, not other metrics."""
    coverage=report.get('reliability',{})
    failed=[row for row in coverage.get('decks',[]) if row.get('sufficient_for_automatic_window') is False]
    if coverage.get('passed') is not False or len(failed)!=1:
        return {}
    row=failed[0]
    if (row.get('role'),row.get('id')) not in (('tail',transition['from']),('head',transition['to'])):
        return {}
    return {'failed_track':row['id'],'failed_roles':[row['role']],
        'failure_ownership_scope':'single failed rendered rhythm-coverage view; energy and vocal checks remain independent'}


def verify_overlap(outgoing, incoming, mixed, sample_rate, clips, transition, *, vocal_out=None,
                   vocal_in=None, style='default', beat_phase=None):
    energy = audit_energy(outgoing, incoming, mixed, sample_rate)
    voice = audit_vocals(outgoing, incoming, mixed, sample_rate, vocal_out, vocal_in, transition,
        style=style, beat_phase=beat_phase)
    coverage = reliability(clips, transition)
    report = {'passed': energy['passed'] and voice['passed'] and coverage['passed'],
        'energy': energy, 'vocals': voice, 'reliability': coverage}
    if not report['passed']:
        report.update(coverage_failure_context(report,transition))
        reasons = [label for label, value in (('sustained energy loss', energy),
            ('competing estimated foregrounds', voice), ('insufficient recognized rhythm coverage', coverage)) if not value['passed']]
        raise TransitionFailure(transition['from'], transition['to'], '; '.join(reasons), report)
    return report
