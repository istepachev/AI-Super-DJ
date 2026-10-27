"""Bounded gradual handoff placement on unchanged, independently timed decks."""
from pathlib import Path
import numpy as np
from .mixing import blend,envelopes,recorded_envelopes
from .percussion import percussion_gains,verification_components,verify_blended_fronts
from .transition_quality import TransitionFailure
from .derived_audio import derived_scope


@derived_scope()
def legacy_mid_alternative(outgoing,incoming,sample_rate,outgoing_warp,incoming_warp,
                           transition,directory,*,style,beat_phase,percussion_args,
                           initial_error,quality_check,ffmpeg='ffmpeg',bass_hz=180.,
                           mid_energy_args=None,allow_pitch_changing_residual=False):
    """Retain a working ordinary route when new mid audibility is uncertain."""
    if not transition.get('frequency_processing',{}).get('adaptive_mid_energy',{}).get('applied'):
        raise initial_error
    if not callable(quality_check):raise initial_error
    controls={k:transition[k]for k in ('mid_handoff_bars','bass_handoff_beats','handoff_windows','whole_spectrum_handoff')if transition.get(k)is not None}
    if controls.get('whole_spectrum_handoff')is not None:raise initial_error
    voice={**(mid_energy_args or{}),'mid_energy_mode':'legacy'}
    mixed,frequency=blend(outgoing,incoming,sample_rate,bass_hz=bass_hz,beats=transition.get('beats',64),
        style=style,beat_phase=beat_phase,tonal_conflict=transition.get('frequency_processing',{}).get('tonal_conflict_risk',False),
        **controls,**percussion_args,**voice)
    candidate={**transition,'frequency_processing':frequency,'mid_energy_mode':'legacy'}
    gains,_=recorded_envelopes(len(mixed),transition.get('beats',64),candidate,style=style,beat_phase=beat_phase)
    gains=percussion_gains(gains,beat_phase,transition.get('beats',64),percussion_args.get('percussion_policy','shared'))
    components=verification_components(outgoing,incoming,sample_rate,gains,bass_hz=bass_hz,
        **{k:v for k,v in percussion_args.items()if k!='percussion_policy'})
    mixed,fronts=verify_blended_fronts(mixed,sample_rate,outgoing_warp,incoming_warp,gains,directory,
        ffmpeg=ffmpeg,components=components,allow_pitch_changing_residual=allow_pitch_changing_residual)
    candidate['blended_front_audit']=fronts
    candidate['quality_gate']=quality_check(mixed,candidate)
    candidate['mid_energy_legacy_recovery']={'executed':True,'initial_reason':str(initial_error),
        'source_clock_overlap_and_band_positions_unchanged':True,'timing_gate_relaxed':False,
        'reason':'new audible mid contribution did not pass; retained ordinary curve passed every actual check'}
    return mixed,candidate


@derived_scope()
def gradual_alternative(outgoing,incoming,sample_rate,outgoing_warp,incoming_warp,
                        transition,directory,*,style,beat_phase,percussion_args,
                        initial_error,ffmpeg='ffmpeg',bass_hz=180.,tonal_conflict=False,
                        quality_check=None,allow_pitch_changing_residual=False,mid_energy_args=None):
    """Move a complete soft curve, never a deck, native attack, or user lock.

    Every candidate keeps the original overlap, clock, bass transfer width,
    middle/high curves and strict combined-front requirements. Failure remains
    a failure; a native-source alternative can then be chosen by the caller.
    """
    if (transition.get('handoff_windows') or {}).get('bass') or transition.get('bass_handoff_beats') is not None:
        raise initial_error
    beats=transition.get('beats',64)
    controls={k:transition[k] for k in ('mid_handoff_bars','handoff_windows') if k in transition}
    _,base=envelopes(len(incoming),beats,style=style,beat_phase=beat_phase,tonal_conflict=tonal_conflict,**controls)
    first,last=base['bass_handoff_fraction'];width=(last-first)*beats
    if width>=beats-1e-9:raise initial_error
    drum_args={k:v for k,v in percussion_args.items() if k!='percussion_policy'}
    if set(drum_args)!={'drums_out','drums_in'}:
        raise ValueError('Gradual orchestration requires both final-deck drum estimates')
    failures=[];last_error=initial_error
    for position,fraction in enumerate((.375,.75,.25)):
        start=max(0.,min(fraction*beats-width/2,beats-width));end=start+width
        if abs(start-first*beats)<1e-7:continue
        windows={**controls.get('handoff_windows',{}),'bass':[start,end]}
        selected={**controls,'handoff_windows':windows}
        mixed,frequency=blend(outgoing,incoming,sample_rate,bass_hz=bass_hz,beats=beats,
            style=style,beat_phase=beat_phase,tonal_conflict=tonal_conflict,
            percussion_policy='smooth',**drum_args,**selected,**(mid_energy_args or{}))
        gains,_=recorded_envelopes(len(mixed),beats,{**transition,**selected,'frequency_processing':frequency},style=style,beat_phase=beat_phase)
        gains=percussion_gains(gains,beat_phase,beats,'smooth')
        components=verification_components(outgoing,incoming,sample_rate,gains,bass_hz=bass_hz,**drum_args)
        candidate={**transition,**selected,'percussion_ownership_policy':'smooth','frequency_processing':frequency}
        try:
            mixed,fronts=verify_blended_fronts(mixed,sample_rate,outgoing_warp,incoming_warp,gains,
                Path(directory)/f'position-{position}',ffmpeg=ffmpeg,components=components,
                allow_pitch_changing_residual=allow_pitch_changing_residual)
            candidate['blended_front_audit']=fronts
            if quality_check is not None:candidate['quality_gate']=quality_check(mixed,candidate)
        except (ValueError,TransitionFailure) as error:
            try:
                if quality_check is None:raise error
                mixed,candidate=legacy_mid_alternative(outgoing,incoming,sample_rate,outgoing_warp,incoming_warp,candidate,
                    Path(directory)/f'position-{position}-legacy-middle',style=style,beat_phase=beat_phase,
                    percussion_args={**drum_args,'percussion_policy':'smooth'},initial_error=error,
                    quality_check=quality_check,ffmpeg=ffmpeg,bass_hz=bass_hz,mid_energy_args=mid_energy_args,
                    allow_pitch_changing_residual=allow_pitch_changing_residual)
            except (ValueError,TransitionFailure):
                failures.append({'bass_shared_beats':[start,end],'reason':str(error)})
                last_error=error
                continue
        candidate['gradual_orchestration_search']={'executed':True,'passed':True,
            'initial_reason':str(initial_error),'failed_alternatives':failures,
            'selected_bass_shared_beats':[start,end],'original_bass_shared_beats':[first*beats,last*beats],
            'maximum_alternatives':3,'source_windows_and_clock_unchanged':True,
            'bass_transfer_width_preserved':True,'middle_and_high_curves_preserved':True,
            'user_bass_position_and_width_locks_preserved':True,'timing_gate_relaxed':False,
            'pitch_or_timing_repair_performed':False}
        return mixed,candidate
    raise last_error


@derived_scope()
def whole_spectrum_alternative(outgoing,incoming,sample_rate,outgoing_warp,incoming_warp,
                              transition,directory,*,style,beat_phase,initial_error,
                              quality_check=None,ffmpeg='ffmpeg',allow_pitch_changing_residual=False):
    """After existing routing fails, retain wide handoffs before the hard case.

    Native cuts, phase, overlap and pitch stay fixed. Approved separate-band
    settings are locks here. This fallback never introduces a short rhythmic
    switch, changes a working normal blend, or waives an acoustic/energy gate.
    """
    if transition.get('whole_spectrum_handoff') is not None or transition.get('handoff_windows') or transition.get('mid_handoff_bars') is not None or transition.get('bass_handoff_beats') is not None:
        raise initial_error
    beats=transition.get('beats',64);failures=[];last_error=initial_error
    # Preserve the previously successful full-window options. The reviewed
    # early whole-spectrum transfer is last, only when both wider routes fail.
    lead=min(4.,max(0.,(float(beats)-32.)/2.))
    recipes=[{'start_beat':0.,'end_beat':float(beats),'bias':bias}for bias in (3.,1.)]
    hard_case={'start_beat':lead,'end_beat':min(float(beats),lead+32.),'bias':1.}
    if hard_case not in recipes:recipes.append(hard_case)
    for number,spec in enumerate(recipes):
        try:
            mixed,frequency=blend(outgoing,incoming,sample_rate,beats=beats,style=style,
                beat_phase=beat_phase,whole_spectrum_handoff=spec)
            gains,_=envelopes(len(mixed),beats,style=style,beat_phase=beat_phase,whole_spectrum_handoff=spec)
            components={'tail':(outgoing*gains['mid_out']).astype('float32'),'head':(incoming*gains['mid_in']).astype('float32')}
            mixed,fronts=verify_blended_fronts(mixed,sample_rate,outgoing_warp,incoming_warp,gains,
                Path(directory)/f'whole-{number}',ffmpeg=ffmpeg,components=components,
                allow_pitch_changing_residual=allow_pitch_changing_residual)
            candidate={**transition,'whole_spectrum_handoff':spec,'frequency_processing':frequency,
                'percussion_ownership_policy':'shared','blended_front_audit':fronts,
                'percussion_estimates_used_in_final_mix':False}
            estimates=candidate.pop('percussion_estimates',None)
            if estimates:candidate['previous_percussion_estimate_attempts']=estimates
            if quality_check is not None:candidate['quality_gate']=quality_check(mixed,candidate)
        except (ValueError,TransitionFailure) as error:
            last_error=error;failures.append({'whole_spectrum_handoff':spec,'reason':str(error)});continue
        candidate['whole_spectrum_recovery']={'executed':True,'initial_reason':str(initial_error),
            'maximum_alternatives':len(recipes),'selected_strategy':'early_whole_spectrum_hard_case' if number==2 else 'wide_whole_spectrum','failed_alternatives':failures,'source_windows_clock_and_overlap_unchanged':True,
            'frequency_filtering_or_percussion_subtraction':False,'timing_or_pitch_repair_performed':False,
            'timing_gate_relaxed':False,'actual_10_90_shared_beats':frequency['actual_10_90_shared_beats']}
        return mixed,candidate
    raise last_error
