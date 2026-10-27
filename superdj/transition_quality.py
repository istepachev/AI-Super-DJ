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


def local_reliability(clips, transition, outgoing, incoming, sample_rate, *, style, beat_phase):
    """Late matches cannot certify an earlier audible part of a shared overlap.

    Use the actual band routing, including residual accompaniment: a zero drum
    fader does not silence every transient. Only physically quiet native input
    or a genuinely inaudible route is excluded. Missing events remain unknown;
    this gate never labels or moves them onto quarter notes.
    """
    if transition.get('tempo_technique') != 'shared_phase_sync':
        return {'applicable':False,'passed':True}
    from .mixing import envelopes
    from .percussion import percussion_gains
    clock=transition.get('shared_clock') or {}
    count=clock.get('beats',transition.get('beats'))
    frames=np.asarray(clock.get('beat_output_frames',[]),dtype=float)
    if (len(clips)!=2 or not isinstance(count,(int,np.integer)) or count<4
            or frames.shape!=(count+1,) or not np.isfinite(frames).all()
            or frames[0]!=0 or np.any(np.diff(frames)<=0)
            or outgoing.shape!=incoming.shape or len(outgoing)<2
            or abs(frames[-1]-len(outgoing))>1 or beat_phase is None):
        raise ValueError('Local rhythm coverage requires the actual shared clock, phase and both timed decks')
    frequency=transition.get('frequency_processing') or {}
    gains,curves=envelopes(len(outgoing),count,beat_phase=beat_phase,style=style,
        tonal_conflict=frequency.get('tonal_conflict_risk',False),
        mid_handoff_bars=transition.get('mid_handoff_bars'),
        bass_handoff_beats=transition.get('bass_handoff_beats'),
        handoff_windows=transition.get('handoff_windows'),
        whole_spectrum_handoff=transition.get('whole_spectrum_handoff'))
    if frequency.get('profile_sha256') and curves.get('profile_sha256')!=frequency['profile_sha256']:
        raise ValueError('Local coverage profile differs from the actual overlap profile')
    if transition.get('whole_spectrum_handoff') is None:
        gains=percussion_gains(gains,beat_phase,count,transition.get('percussion_ownership_policy','shared'))
    indices=np.clip(np.rint(frames).astype(int),0,len(outgoing)-1)
    width=min(16,int(count));starts=sorted(set([*range(0,count-width+1,4),count-width]))
    rows=[]
    for clip,role,audio,side in zip(clips,('tail','head'),(outgoing,incoming),('out','in')):
        final=(clip.get('warp') or {}).get('post_render_transient_audit',{}).get('final',{})
        recognized={int(p['beat']) for p in final.get('measured',[]) if p.get('role')==role
            and isinstance(p.get('beat'),(int,np.integer)) and 0<p['beat']<count
            and isinstance(p.get('delay_seconds'),(int,float,np.integer,np.floating))
            and np.isfinite(p['delay_seconds']) and abs(p['delay_seconds'])<=.003+1e-12}
        audible_gain=np.maximum.reduce([gains[name+'_'+side][indices,0]
            for name in ('bass','mid','high')]+([gains['drums_'+side][indices,0]] if 'drums_'+side in gains else []))
        powers=[]
        for first,last in zip(frames[:-1],frames[1:]):
            block=audio[round(first):min(round(last),len(audio))]
            powers.append(float(np.mean(block*block,dtype=np.float64)) if len(block) else 0.)
        quiet_floor=max(1e-12,max(powers,default=0.)*1e-4)
        audible={i for i in range(1,count) if audible_gain[i]>=.05 and powers[i]>quiet_floor}
        unknown=audible-recognized;longest=run=0
        for i in range(1,count):
            run=run+1 if i in unknown else 0;longest=max(longest,run)
        windows=[]
        for first in starts:
            active=audible.intersection(range(first,first+width))
            if len(active)<4:continue
            known=active & recognized;required=max(2,int(np.ceil(.25*len(active))))
            windows.append({'start_beat':first,'end_beat':first+width,
                'start_seconds':float(frames[first]/sample_rate),'end_seconds':float(frames[first+width]/sample_rate),
                'audible_beats':len(active),'recognized_beats':len(known),'required_recognized_beats':required,
                'passed':len(known)>=required})
        passed=all(w['passed']for w in windows) and longest<=8 and (not audible or bool(audible & recognized))
        rows.append({'id':clip['id'],'role':role,'sufficient_for_automatic_window':passed,
            'audible_beats':len(audible),'recognized_audible_beats':len(audible & recognized),
            'longest_unverified_audible_beats':longest,'maximum_unverified_audible_beats':8,
            'quiet_input_beats':sum(p<=quiet_floor for p in powers),
            'windows':windows,'unverified_audible_beat_indices':sorted(unknown)})
    return {'applicable':True,'passed':all(r['sufficient_for_automatic_window'] for r in rows),'decks':rows,
        'window_beats':width,'stride_beats':4,'minimum_local_fraction':.25,
        'scope':'local recognized coverage of audible routes, including the beginning and handoff; silence is not a verified beat',
        'limitations':'conservative window rejection; not a classifier of every musical figure or a guarantee of musical quality'}


def coverage_failure_context(report, transition):
    """Attribute only an unambiguous failed coverage view, not other metrics."""
    coverage=report.get('reliability',{})
    local=report.get('local_rhythm_coverage',{})
    failed=[row for view in (coverage,local) if view.get('passed') is False
        for row in view.get('decks',[]) if row.get('sufficient_for_automatic_window') is False]
    if not failed:
        return {}
    owners={(row.get('role'),row.get('id')) for row in failed}
    if len(owners)!=1:
        return {}
    role,identity=next(iter(owners))
    if (role,identity) not in (('tail',transition['from']),('head',transition['to'])):
        return {}
    return {'failed_track':identity,'failed_roles':[role],
        'failure_ownership_scope':'single failed rendered rhythm-coverage view; energy and vocal checks remain independent'}


def verify_overlap(outgoing, incoming, mixed, sample_rate, clips, transition, *, vocal_out=None,
                   vocal_in=None, style='default', beat_phase=None):
    energy = audit_energy(outgoing, incoming, mixed, sample_rate)
    voice = audit_vocals(outgoing, incoming, mixed, sample_rate, vocal_out, vocal_in, transition,
        style=style, beat_phase=beat_phase)
    coverage = reliability(clips, transition)
    local=local_reliability(clips,transition,outgoing,incoming,sample_rate,style=style,beat_phase=beat_phase)
    report = {'passed': energy['passed'] and voice['passed'] and coverage['passed'] and local['passed'],
        'energy': energy, 'vocals': voice, 'reliability': coverage,'local_rhythm_coverage':local}
    if not report['passed']:
        report.update(coverage_failure_context(report,transition))
        reasons = [label for label, value in (('sustained energy loss', energy),
            ('competing estimated foregrounds', voice), ('insufficient recognized rhythm coverage', coverage),
            ('insufficient local rhythm coverage',local)) if not value['passed']]
        raise TransitionFailure(transition['from'], transition['to'], '; '.join(reasons), report)
    return report
