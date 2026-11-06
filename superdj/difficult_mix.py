"""Bounded foreground recovery after an actual ordinary-overlap refusal.

This is orchestration of already timed decks, never a timing repair or a
musical-quality certificate. Ordinary successful PCM does not enter this code.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

from .mixing import smooth, split_bands
from .transition_quality import TransitionFailure, _power, _longest

EPOCH = 1


def settings(value=None):
    value = {} if value is None else value
    if value is False:
        return {"enabled": False, "foreground_priority": "auto"}
    if not isinstance(value, dict) or set(value)-{"enabled", "foreground_priority"}:
        raise ValueError("Difficult-transition settings support enabled and foreground_priority")
    result = {"enabled": True, "foreground_priority": "auto", **value}
    if type(result["enabled"]) is not bool or result["foreground_priority"] not in {"auto", "outgoing", "incoming"}:
        raise ValueError("Choose an enabled flag and auto/outgoing/incoming foreground priority")
    return result


def eligible(error, transition):
    if not settings(transition.get("difficult_transition"))["enabled"]:
        return False
    if any(transition.get(k) is not None for k in
           ("mid_handoff_bars", "bass_handoff_beats", "handoff_windows", "whole_spectrum_handoff")):
        return False
    if transition.get("mid_energy_mode", "auto") != "auto":
        return False
    evidence = getattr(error, "evidence", {})
    if isinstance(error, TransitionFailure):
        return any(evidence.get(k, {}).get("passed") is False
                   for k in ("energy", "vocals", "impulse", "local_rhythm_coverage"))
    return isinstance(error, ValueError) and str(error).startswith((
        "Combined percussion fronts remain unverified", "Known native initial fronts remain unverified",
        "Accompaniment energy handoff is too short", "Percussion handoff is too short"))


def quiet_gaps(voice, sample_rate, *, minimum_seconds=.18):
    """Measured estimate gaps, not word/phrase recognition or ground truth."""
    step = max(1, round(sample_rate*.02))
    power = _power(voice, step)
    if not len(power) or float(np.max(power)) < 1e-12:
        return []
    quiet = power < max(float(np.quantile(power, .9))*.0025, 1e-12)
    changes = np.diff(np.r_[False, quiet, False].astype(int))
    return [(first*step/sample_rate, last*step/sample_rate)
            for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
            if (last-first)*step/sample_rate >= minimum_seconds]


def voice_gains(n, sample_rate, report):
    """Reconstruct exactly the gains recorded by the rescue, also for audits."""
    if report.get("method") != "continuous foreground recovery v1":
        raise ValueError("Unknown foreground recovery envelope")
    t = np.arange(n, dtype=float)/sample_rate
    if report.get("frames") != n or report.get("sample_rate") != sample_rate:
        raise ValueError("Foreground envelope belongs to different timed PCM")
    release = report["outgoing_release_seconds"]
    nodes = np.asarray(report["incoming_db_points"], dtype=float)
    if (len(release) != 2 or not 0 <= release[0] < release[1] <= n/sample_rate
            or nodes.shape != (3, 2) or not np.isfinite(nodes).all()
            or not release[1] <= nodes[0, 0] < nodes[1, 0] < nodes[2, 0] <= (n-1)/sample_rate+1e-9
            or not np.array_equal(nodes[:, 1], [-80., -3., 0.])):
        raise ValueError("Invalid continuous foreground recovery envelope")
    a = 1-smooth((t-release[0])/(release[1]-release[0]))
    db = np.full(n, nodes[0, 1])
    for (first, lo), (last, hi) in zip(nodes[:-1], nodes[1:]):
        use = t >= first
        db[use] = lo+(hi-lo)*smooth((t[use]-first)/(last-first))
    b = 10**(db/20)
    b[t <= nodes[0, 0]] = 0.
    return a[:, None], b[:, None]


def foreground_recipe(outgoing, incoming, vocal_out, vocal_in, sample_rate, priority):
    n = len(outgoing); seconds = (n-1)/sample_rate
    if seconds < 8:
        raise ValueError("Foreground recovery needs supported phrase space, not a forced short cut")
    for audio, voice in ((outgoing, vocal_out), (incoming, vocal_in)):
        if voice is None or voice.shape != audio.shape or not np.isfinite(voice).all():
            raise ValueError("Foreground recovery needs both exact aligned vocal estimates")
    old_gaps = quiet_gaps(vocal_out, sample_rate)
    target = (.58 if priority == "outgoing" else .30)*seconds
    choices = sorted((gap for gap in old_gaps if .12*seconds <= gap[0] <= .72*seconds
                      and gap[1] < seconds-7), key=lambda gap: abs(gap[0]-target))
    for first, last in choices:
        # Fade only inside an actual quiet estimate gap. No activity-following
        # syllable ducking; the incoming verse uses one continuous dB opening.
        release = [first, min(last, first+.3)]
        entries = [end for begin, end in quiet_gaps(vocal_in, sample_rate)
                   if end >= release[1]+2.5 and end <= seconds-5 and begin >= release[1]]
        if not entries:
            continue
        main = min(entries, key=lambda t: abs(t-(.75 if priority == "outgoing" else .55)*seconds))
        opening = max(release[1], main-4.)
        report = {"executed": True, "method": "continuous foreground recovery v1",
                  "frames": n, "sample_rate": sample_rate, "foreground_priority": priority,
                  "outgoing_release_seconds": release,
                  "incoming_db_points": [[opening, -80.], [main, -3.], [seconds, 0.]],
                  "outgoing_measured_quiet_gap_seconds": [first, last],
                  "absence_certified": False,
                  "limitations": "Aligned estimates can contain leakage; gaps are not recognized word boundaries"}
        voice_gains(n, sample_rate, report)
        return report
    raise ValueError("No measured quiet release and gradual next-verse opening; choose another phrase")


def prepare_foregrounds(outgoing, incoming, vocal_out, vocal_in, drums_out, drums_in,
                        sample_rate, recipe):
    """Keep bass and estimated percussion; soften competing residual parties."""
    n = len(outgoing)
    if outgoing.shape != incoming.shape or outgoing.ndim != 2 or outgoing.shape[1] != 2:
        raise ValueError("Recovery requires matching timed stereo decks")
    keep_a, keep_b = voice_gains(n, sample_rate, recipe)
    p = np.linspace(0., 1., n)[:, None]
    result = []
    for audio, voice, drums, keep, side in (
            (outgoing, vocal_out, drums_out, keep_a, "out"),
            (incoming, vocal_in, drums_in, keep_b, "in")):
        if any(x is None or x.shape != audio.shape or not np.isfinite(x).all()
               for x in (audio, voice, drums)):
            raise ValueError("Rescue mixture, voice and percussion must share the exact final timeline")
        low, mid, high = split_bands(audio-voice-drums, sample_rate)
        if side == "in":
            start = .55 if recipe["foreground_priority"] == "outgoing" else .15
            opening = smooth((p-start)/(1-start))
            mid_keep = 10**((-12*(1-opening))/20)
            high_keep = 10**((-15*(1-opening))/20)
        else:
            start = .55 if recipe["foreground_priority"] == "outgoing" else .10
            end = .95 if recipe["foreground_priority"] == "outgoing" else .65
            closing = smooth((p-start)/(end-start))
            mid_keep = 10**((-12*closing)/20)
            high_keep = 10**((-15*closing)/20)
        value = low+mid*mid_keep+high*high_keep+voice*keep+drums
        # The crossfade consumes the native first/last sample at unit gain.
        # Preserve those exact neighbours, including their estimate roundoff.
        if side == "out": value[0] = audio[0]
        else: value[-1] = audio[-1]
        result.append(value.astype("float32"))
    return tuple(result)


def audit_impulse(outgoing, incoming, mixed, sample_rate):
    """Catch a sustained loss masked by a quiet/fading source in a long fade."""
    step = max(1, round(sample_rate*.5))
    pa, pb, pm = [_power(x, step) for x in (outgoing, incoming, mixed)]
    if not len(pm):
        return {"passed": True, "status": "too short for sustained-loss measurement"}
    available = np.maximum(pa, pb)
    active = available > max(float(np.quantile(available, .75))*.05, 1e-10)
    middle = (np.arange(len(pm))+.5)/len(pm)
    loss = active & (middle > .1) & (middle < .95) & (pm < available*10**(-9/10))
    seconds = _longest(loss)
    return {"passed": seconds < 1.5, "sustained_available_energy_loss_seconds": seconds,
            "scope": "actual PCM versus available aligned decks; native quiet breaks are not filled or certified"}


def recover(outgoing, incoming, sample_rate, clips, transition, directory, *,
            vocal_out, vocal_in, initial_error, beat_phase, style, ffmpeg,
            bass_hz=180., allow_pitch_changing_residual=False):
    """At most two cheap foreground recipes; keep every existing audio gate."""
    if not eligible(initial_error, transition):
        raise initial_error
    from .storage_budget import validate_storage_path
    validate_storage_path(directory)
    from .percussion import (final_drum_estimates, percussion_gains,
                             verification_components, verify_blended_fronts)
    from .mixing import blend, recorded_envelopes
    from .transition_quality import verify_overlap, audit_energy
    policy = settings(transition.get("difficult_transition"))
    priorities = ("outgoing", "incoming") if policy["foreground_priority"] == "auto" else (policy["foreground_priority"],)
    # Reuse one final-deck estimate pair for all recipes, not per attempt.
    recipes = []
    failures = []
    for priority in priorities:
        try:
            recipes.append(foreground_recipe(outgoing, incoming, vocal_out, vocal_in, sample_rate, priority))
        except ValueError as error:
            failures.append(str(error))
    if not recipes:
        raise initial_error
    from .resources import memory_policy
    if 80*outgoing.size > memory_policy()["buffer_budget_bytes"]:
        # Keep application headroom; do not launch additional models on a
        # crowded machine just because a foreground recipe exists.
        raise initial_error
    try:
        drums, estimates = final_drum_estimates(outgoing, incoming, sample_rate, Path(directory)/"estimates", ffmpeg=ffmpeg)
    except RuntimeError as error:
        if str(error).startswith("Percussion handoff requires the configured local Demucs runtime"):
            raise initial_error from error
        raise
    for index, recipe in enumerate(recipes):
        try:
            a, b = prepare_foregrounds(outgoing, incoming, vocal_out, vocal_in,
                                      drums["drums_out"], drums["drums_in"], sample_rate, recipe)
            selected = deepcopy(transition)
            selected["vocal_processing"] = recipe
            selected["mid_energy_mode"] = "legacy"
            beats = transition.get("beats", 64)
            controls = {"handoff_windows": {"bass": [.15*beats, .9*beats], "mid": [0., beats]}} if style == "zhilin" else {"whole_spectrum_handoff": {"start_beat": 0., "end_beat": beats}}
            selected.update(controls)
            args = {**drums, "percussion_policy": "smooth"} if style == "zhilin" else {}
            pcm, frequency = blend(a, b, sample_rate, bass_hz=bass_hz, beats=beats,
                                   style=style, beat_phase=beat_phase, mid_energy_mode="legacy",
                                   **controls, **args)
            selected["frequency_processing"] = frequency
            selected["percussion_ownership_policy"] = "smooth" if args else "shared"
            gains, _ = recorded_envelopes(len(pcm), beats, selected, style=style, beat_phase=beat_phase)
            if args: gains = percussion_gains(gains, beat_phase, beats, "smooth")
            components = verification_components(a, b, sample_rate, gains, bass_hz=bass_hz, **(drums if args else {}))
            pcm, fronts = verify_blended_fronts(pcm, sample_rate, clips[0].get("warp"), clips[1].get("warp"),
                gains, Path(directory)/f"recipe-{index}", ffmpeg=ffmpeg, components=components,
                allow_pitch_changing_residual=allow_pitch_changing_residual)
            observed = {row.get("role") for row in fronts.get("checks", []) if len(row.get("recognized", [])) >= 2}
            if not {"tail", "head"} <= observed:
                raise ValueError("Difficult recovery lacks observed final fronts on both decks")
            selected["percussion_estimates"] = estimates
            quality = verify_overlap(a, b, pcm, sample_rate, clips, selected,
                                     vocal_out=vocal_out, vocal_in=vocal_in, style=style, beat_phase=beat_phase)
            raw_energy = audit_energy(outgoing, incoming, pcm, sample_rate)
            impulse = audit_impulse(outgoing, incoming, pcm, sample_rate)
            if not raw_energy["passed"] or not impulse["passed"]:
                raise TransitionFailure(transition["from"], transition["to"], "Recovery lost available source energy",
                                        {"energy": raw_energy, "impulse": impulse})
            selected.update(quality_gate=quality, blended_front_audit=fronts,
                difficult_transition_recovery={"executed": True, "epoch": EPOCH,
                    "initial_reason": str(initial_error), "recipe": recipe["foreground_priority"],
                    "recipes_attempted": index+1, "maximum_recipes": len(priorities), "rejected": failures,
                    "raw_source_energy": raw_energy, "impulse": impulse,
                    "source_clock_overlap_and_pitch_unchanged": True, "quality_gates_relaxed": False,
                    "residual_mid_high_attenuation_db": [12., 15.], "bass_and_estimated_drums_retained": True,
                    "vocal_absence_certified": False})
            return pcm, selected
        except (ValueError, TransitionFailure) as error:
            failures.append(str(error))
    # Keep source ownership and original evidence for the native phrase search.
    if isinstance(initial_error, TransitionFailure):
        initial_error.evidence["difficult_transition_recovery"] = {"executed": True, "passed": False,
            "rejected": failures, "maximum_recipes": len(priorities), "epoch": EPOCH}
    raise initial_error
