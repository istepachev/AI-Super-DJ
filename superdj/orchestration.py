"""Bounded gradual handoff placement on unchanged, independently timed decks."""
from pathlib import Path
from .mixing import blend,envelopes
from .percussion import percussion_gains,verification_components,verify_blended_fronts
from .transition_quality import TransitionFailure
from .derived_audio import derived_scope


@derived_scope()
def gradual_alternative(outgoing,incoming,sample_rate,outgoing_warp,incoming_warp,
                        transition,directory,*,style,beat_phase,percussion_args,
                        initial_error,ffmpeg='ffmpeg',bass_hz=180.,tonal_conflict=False,
                        quality_check=None,allow_pitch_changing_residual=False):
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
            percussion_policy='smooth',**drum_args,**selected)
        gains,_=envelopes(len(mixed),beats,style=style,beat_phase=beat_phase,
            tonal_conflict=tonal_conflict,**selected)
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
