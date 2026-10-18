"""Separate native attack identity from metrical assignment and audit real maps."""
from copy import deepcopy
from collections import deque
import numpy as np


def _short_term_rate_span(times, corrections):
    """Rate range in every one-second window, independent of control density."""
    order=np.argsort(times,kind='stable');times=np.asarray(times)[order];values=np.asarray(corrections)[order]
    low=deque();high=deque();worst=0.;span=None
    for i,stamp in enumerate(times):
        while low and times[low[0]]<stamp-1.:low.popleft()
        while high and times[high[0]]<stamp-1.:high.popleft()
        while low and values[low[-1]]>=values[i]:low.pop()
        while high and values[high[-1]]<=values[i]:high.pop()
        low.append(i);high.append(i)
        difference=float(values[high[0]]-values[low[0]])
        if difference>worst:
            worst=difference
            span=sorted([float(times[low[0]]),float(times[high[0]])])
    return {'executed':True,'window_seconds':1.,'maximum_correction_rate_span':worst,
            'maximum_allowed_correction_rate_span':.10,'worst_output_seconds':span,
            'passed':worst<=.10+1e-5,
            'method':'range of actual/intended musical rates in every one-second window; controls cannot hide a pulse by subdivision'}


class SourceWindowFailure(ValueError):
    """A measured source window cannot satisfy the existing timing bounds."""
    def __init__(self, reason, *, role=None, evidence=None):
        super().__init__(reason)
        self.role = role
        self.evidence = evidence or {}


def _fine_rate_views(warp, output, rates, expected, active, sample_rate):
    """Exact extrema of the cubic residual map composed with frame slopes."""
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    views=[];base_times=output/sample_rate
    for controls in warp.get('fine_attack_timing',{}).values():
        times=np.asarray(controls['times'],dtype=float);offsets=np.asarray(controls['offsets'],dtype=float)
        if len(times)<2 or len(times)!=len(offsets) or not np.all(np.isfinite(times)) or not np.all(np.isfinite(offsets)) or np.any(np.diff(times)<=0):
            raise ValueError('Invalid residual timing controls')
        if np.max(abs(offsets))>.035:raise ValueError('Residual timing delay exceeds its bound')
        curve=PchipInterpolator(times,offsets);mapped=times+offsets
        if np.any(np.diff(mapped)<=0):raise ValueError('Nonmonotonic residual timing map')
        nodes=list(times)
        for index in range(len(times)-1):
            a,c=curve.c[0,index],curve.c[1,index]
            if abs(a)>1e-20:
                critical=-c/(3*a)
                if 0<critical<times[index+1]-times[index]:nodes.append(times[index]+critical)
        for target in base_times[(base_times>mapped[0])&(base_times<mapped[-1])]:
            index=int(np.clip(np.searchsorted(mapped,target)-1,0,len(times)-2))
            point=brentq(lambda t:t+float(curve(t))-target,times[index],times[index+1],xtol=1e-12)
            # Check both sides of each piecewise-linear backend-map boundary.
            nodes.extend([max(times[0],point-1e-9),min(times[-1],point+1e-9)])
        t=np.asarray(nodes);positions=t+curve(t);derivative=1+curve.derivative()(t)
        if np.min(derivative)<=0:raise ValueError('Nonmonotonic residual timing map')
        indices=np.clip(np.searchsorted(base_times,positions,side='right')-1,0,len(rates)-1)
        effective=rates[indices]*derivative
        views.append((effective,abs(effective/expected[indices]-1),active[indices],t,effective/expected[indices]))
    return views


def classify_native_phase(rows, centres, references):
    """A familiar waveform near a beat is not necessarily that beat's attack.

    Keep displaced observations for review, without promoting fills/ghosts to
    quarter-note controls. A global source-grid bias remains a valid family.
    """
    if len(references)<8:return rows,[]
    offsets=np.asarray([r['time']-centres[r['index']] for r in references])
    baseline=float(np.median(offsets));period=float(np.median(np.diff(centres)))
    spread=float(np.percentile(abs(offsets-baseline),90))
    tolerance=max(.035,min(.060,.10*period))
    if spread>tolerance:
        raise SourceWindowFailure('Native primary attack phase is ambiguous; review another source window')
    indices=np.asarray([r['index'] for r in references],dtype=float)
    reference_times=np.asarray([r['time'] for r in references],dtype=float)
    consensus=None
    if np.ptp(indices)>=7:
        line=np.polyfit(indices,reference_times,1)
        # Rescue a mislabelled coarse beat only from an independently tight
        # native attack family. A displaced fill must still remain displaced.
        if line[0]>0 and abs(line[0]/period-1)<=.03 and np.max(abs(reference_times-np.polyval(line,indices)))<=.010:
            consensus=line
    accepted=[];displaced=[]
    for row in rows:
        residual=float(row['time']-centres[row['index']]-baseline)
        if abs(residual)<=tolerance:accepted.append(row)
        elif consensus is not None and abs(row['time']-np.polyval(consensus,row['index']))<=.010:
            accepted.append({**row,'coarse_grid_consensus_rescue':{
                'coarse_phase_residual_seconds':residual,
                'native_family_residual_seconds':float(row['time']-np.polyval(consensus,row['index']))}})
        else:displaced.append({**row,'native_phase_residual_seconds':residual,
            'established_native_phase_offset_seconds':baseline,'assignment_tolerance_seconds':tolerance,
            'classification':'recognized native figure; quarter-note assignment unsupported; retain its source timing'})
    return accepted,displaced


def initialize_rate_reference(warp, sample_rate, *, native_bpm=None, windows=(), limit=.15):
    """Freeze the intended musical curve before acoustic controls can mutate it."""
    if warp.get('timing_rate_reference'):return
    anchors=warp['anchors'];source=np.asarray([a['source_frame'] for a in anchors],dtype=float)
    output=np.asarray([a['output_frame'] for a in anchors],dtype=float)
    if np.any(np.diff(source)<=0) or np.any(np.diff(output)<=0):raise ValueError('Nonmonotonic actual timing map')
    rates=np.diff(source)/np.diff(output);expected=rates.copy();active=np.zeros(len(rates),dtype=bool)
    for role,clock in windows:
        mapping=warp['input_output_beat_mapping'][role]
        if not mapping:continue
        first=mapping[0]['output_seconds']*sample_rate;last=mapping[-1]['output_seconds']*sample_rate
        middle=(output[:-1]+output[1:])/2;mask=(middle>=first)&(middle<=last)
        active|=mask
        if clock is not None and native_bpm is not None:
            phase=np.interp(output-first,clock['output_frames'],clock['phase'])
            ideal=np.diff(phase)*60/native_bpm*sample_rate/np.diff(output)
            expected[mask]=ideal[mask]
        elif mask.any():
            expected[mask]=np.median(rates[mask])
    if not windows:
        # Advanced callers without a supplied clock retain their original
        # map as the reference. Standard renderers provide the musical curve.
        active[:]=True
    if np.any(expected<=0):raise ValueError('Invalid intended musical timing reference')
    warp['timing_rate_reference']={'expected_musical_rates':expected.tolist(),'monitored_intervals':active.tolist(),
        'maximum_timing_change':float(limit),'anchor_count':len(anchors),
        'method':'immutable intended musical rates; actual source/output frame differences are remeasured after every edit'}


def audit_warp_rates(warp, sample_rate, *, stage, enforce=True, check_continuity=True):
    reference=warp.get('timing_rate_reference')
    if not reference:return None
    anchors=warp['anchors']
    if len(anchors)!=reference['anchor_count']:raise ValueError('Timing reference no longer matches actual anchors')
    source=np.asarray([a['source_frame'] for a in anchors],dtype=float);output=np.asarray([a['output_frame'] for a in anchors],dtype=float)
    if np.any(np.diff(source)<=0) or np.any(np.diff(output)<=0):raise ValueError('Nonmonotonic actual timing map')
    actual=np.diff(source)/np.diff(output);expected=np.asarray(reference['expected_musical_rates'])
    active=np.asarray(reference['monitored_intervals'],dtype=bool);change=abs(actual/expected-1)
    worst=float(np.max(change[active])) if active.any() else 0.
    minimum=float(actual.min());maximum=float(actual.max())
    # Include both sides of each interval boundary. Checking only adjacent
    # controls or mean rates would miss distributed and compensated pulses.
    times=np.column_stack((output[:-1],output[1:]))[active].ravel()/sample_rate
    corrections=np.repeat((actual/expected)[active],2)
    for effective,additional,monitored,fine_times,fine_corrections in _fine_rate_views(warp,output,actual,expected,active,sample_rate):
        minimum=min(minimum,float(effective.min()));maximum=max(maximum,float(effective.max()))
        if monitored.any():worst=max(worst,float(additional[monitored].max()))
        times=np.r_[times,fine_times[monitored]];corrections=np.r_[corrections,fine_corrections[monitored]]
    continuity=_short_term_rate_span(times,corrections) if check_continuity else {
        'executed':False,'passed':False,'reason':'coarse grid only; native-source refinement must pass before stretching'}
    result={'executed':True,'stage':stage,'minimum_actual_playback_rate':minimum,
        'maximum_actual_playback_rate':maximum,'maximum_timing_correction_change':worst,
        'maximum_allowed_timing_change':reference['maximum_timing_change'],'monitored_intervals':int(active.sum()),
        'all_actual_anchor_intervals_checked':True,'composed_cubic_residual_timing_checked':True,
        'native_attack_islands_are_separate_unpitched_processing':bool(warp.get('native_attack_preservation')),
        'short_term_timing_continuity':continuity,
        'passed':worst<=reference['maximum_timing_change']+1e-5 and (not check_continuity or continuity['passed'])}
    warp['actual_timing_rate_audit']=result
    warp['max_local_warp_rate_change']=max(abs(minimum-1),abs(maximum-1))
    warp['max_timing_correction_change']=worst
    warp['timing_correction_measurement']='actual final source/output frame slopes relative to immutable intended musical rates'
    if enforce and worst>reference['maximum_timing_change']+1e-5:
        raise ValueError(f'Actual acoustic timing change {worst:.1%} exceeds {reference["maximum_timing_change"]:.1%}; review source phase/window')
    if enforce and check_continuity and not continuity['passed']:
        raise ValueError(f'Short-term acoustic timing variation {continuity["maximum_correction_rate_span"]:.1%} exceeds 10.0% within one second; review source phase/window')
    return result


def guarded_anchor_edit(warp, sample_rate, edit, *, stage):
    before=deepcopy(warp['anchors'])
    try:
        edit()
        return audit_warp_rates(warp,sample_rate,stage=stage)
    except ValueError:
        rejected=deepcopy(warp.get('actual_timing_rate_audit'))
        warp['anchors']=before
        audit_warp_rates(warp,sample_rate,stage='restored last safe map')
        if rejected:warp.setdefault('rejected_timing_rate_audits',[]).append(rejected)
        raise
