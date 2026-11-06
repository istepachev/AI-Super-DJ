"""Conditional tools for difficult transitions and listener-requested repairs.

These produce candidates, not verified deliveries. Re-run the normal source,
clock, blend, duration and export gates after any timeline or pitch change.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import soundfile as sf

from .render import RenderError
from .transition_quality import TransitionFailure
from .mixing import smooth


def require_refusal(failure):
    if not isinstance(failure, TransitionFailure) or not any(
            failure.evidence.get(name, {}).get("passed") is False for name in
            ("energy", "vocals", "impulse", "local_rhythm_coverage", "harmonic_conflict", "listener_review")):
        raise ValueError("A difficult-transition tool requires an actual quality refusal or recorded listener rejection")


def extend_riff(streams, sample_rate, *, first_frame, last_frame, join_frames,
                copies, beat_times, phase_origin_seconds, failure, protected_ranges=()):
    """Repeat a measured native riff identically in mixture/voice/percussion.

    The splice boundary and measured beat-phase origin are separate. Retain
    every observed intra-phrase attack without uniformly rescaling its times.
    Use this bounded in-memory tool on a selected source excerpt, then rebuild
    source timing evidence. It does not manufacture a kick or approve a blend.
    """
    require_refusal(failure)
    from .loops import assess_join, join_tail
    if set(streams) != {"mixture", "voice", "drums"}:
        raise ValueError("Riff extension needs mixture, voice and drums on one timeline")
    shapes = {x.shape for x in streams.values()}
    if (len(shapes) != 1 or any(x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all() for x in streams.values())
            or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000):
        raise ValueError("Riff streams must be finite aligned stereo PCM")
    n = len(streams["mixture"])
    from .resources import memory_policy
    if n > 120*sample_rate or 160*streams["mixture"].size > memory_policy()["buffer_budget_bytes"]:
        raise ValueError("Use a shorter selected riff excerpt within the available memory budget")
    if any(type(x) is not int for x in (first_frame, last_frame, join_frames, copies)):
        raise ValueError("Riff geometry requires integer frames and copy count")
    length = last_frame-first_frame
    if (not 1 <= copies <= 4 or not 128 <= join_frames < length/2
            or not join_frames <= first_frame < last_frame < n
            or copies*length > 60*sample_rate):
        raise ValueError("Riff extension exceeds its supported geometry or sixty-second budget")
    if protected_ranges:
        raise ValueError("Riff extension preserves locked/protected source material")
    beats = np.asarray(beat_times, dtype=float)
    if (beats.ndim != 1 or len(beats) < 5 or not np.isfinite(beats).all()
            or np.any(np.diff(beats) <= 0) or beats[0] < 0 or beats[-1] > n/sample_rate):
        raise ValueError("Riff extension needs measured native attack times")
    period = float(np.median(np.diff(beats)))
    first, last = first_frame/sample_rate, last_frame/sample_rate
    if not np.isfinite(phase_origin_seconds) or abs(first-phase_origin_seconds) > period/2:
        raise ValueError("Splice boundary must retain the measured nearby phase origin")
    measured = beats[(beats >= phase_origin_seconds-1e-6) & (beats < phase_origin_seconds+length/sample_rate-1e-6)]
    count = round(length/sample_rate/period)
    if (count < 4 or count % 4 or len(measured) != count
            or np.max(abs(np.diff(np.r_[measured, measured[0]+length/sample_rate])/period-1)) > .05):
        raise ValueError("Native riff/seam intervals exceed the unchanged five-percent timing bound")
    # Assess the real mixture, and independently its percussion seam. Similar
    # chroma alone cannot establish that a repeated rhythm is continuous.
    proof = {}
    for key in ("mixture", "drums"):
        x = streams[key]
        proof[key], reason = assess_join(x[last_frame-join_frames:last_frame],
                                        x[first_frame-join_frames:first_frame], sample_rate, period)
        if proof[key] is None:
            raise ValueError("Unsupported native riff seam: "+reason)
    voice = streams["voice"][first_frame:last_frame]
    mix = streams["mixture"][first_frame:last_frame]
    ratio = float(np.mean(voice.astype(float)**2)/max(np.mean(mix.astype(float)**2), 1e-12))
    if ratio > .02:
        raise ValueError("Choose an instrumental riff; repeated estimated foreground is too strong")
    result = {}
    for key, x in streams.items():
        loop = x[first_frame:last_frame]
        seam = join_tail(x[last_frame-join_frames:last_frame], x[first_frame-join_frames:first_frame], sample_rate, period)
        pieces = [x[:last_frame-join_frames], seam]
        for copy in range(copies):
            pieces.extend([loop[:-join_frames], seam] if copy < copies-1 else [loop])
        pieces.append(x[last_frame:])
        result[key] = np.concatenate(pieces).astype("float32")
    seconds = length/sample_rate
    # Use the SPLICE interval to copy real events. A measured beat may be in
    # preroll before the splice; shifting an entire grid to the cut is wrong.
    inside = beats[(beats >= first) & (beats < last)]
    mapped = np.sort(np.r_[np.where(beats >= last, beats+copies*seconds, beats),
                           *[inside+j*seconds for j in range(1, copies+1)]])
    if np.any(np.diff(mapped) <= 0):
        raise ValueError("Repeated native events overlap or duplicate at the splice")
    return result, mapped, {"executed": True, "candidate_only": True,
        "first_frame": first_frame, "last_frame": last_frame, "join_frames": join_frames,
        "copies": copies, "added_frames": copies*length, "phase_origin_seconds": phase_origin_seconds,
        "native_attack_offsets_preserved": True, "join_quality": proof,
        "estimated_vocal_energy_ratio": ratio, "vocal_absence_certified": False,
        "requires_fresh_source_clock_blend_and_export_checks": True}


def patch_contribution(mixed, old_contribution, new_contribution, sample_rate, *,
                       first_seconds, last_seconds, margin_seconds=1., failure):
    """Replace one known OWN contribution only, with smooth local margins.

    Never subtract an estimated second deck from an unknown finished mix.
    Caller supplies the original exact contribution from its render recipe.
    Outside the interval the result remains bit-identical.
    """
    require_refusal(failure)
    if (mixed.shape != old_contribution.shape or mixed.shape != new_contribution.shape
            or mixed.dtype != np.float32 or mixed.ndim != 2 or mixed.shape[1] != 2
            or not all(np.isfinite(x).all() for x in (mixed, old_contribution, new_contribution))):
        raise ValueError("Local repair needs exact aligned own contributions and float32 PCM")
    first, last, margin = [round(x*sample_rate) for x in (first_seconds, last_seconds, margin_seconds)]
    if not 0 <= first < last <= len(mixed) or not 2 <= margin <= (last-first)/2:
        raise ValueError("Local repair needs a bounded interval and smooth margins")
    weight = np.ones(last-first)
    weight[:margin] = smooth(np.linspace(0, 1, margin))
    weight[-margin:] = smooth(np.linspace(1, 0, margin))
    result = mixed.copy()
    result[first:last] += ((new_contribution[first:last]-old_contribution[first:last])*weight[:, None]).astype("float32")
    return result, {"candidate_only": True, "patch_frames": [first, last], "margin_frames": margin,
                    "outside_interval_pcm_unchanged": True, "requires_fresh_blend_checks": True}


def pitch_preparation(prepared, destination, *, semitones, rubberband, failure):
    """Pitch-only first stage for an entire prepared source, all streams alike.

    Timing remains the ordinary second stage. Combining pitch and a timemap
    has not established correct pitch on supported executables. Preserve
    formants; do not reconcile neighbouring edges using a hidden pitch glide.
    """
    require_refusal(failure)
    if (isinstance(semitones, bool) or not isinstance(semitones, int)
            or semitones == 0 or abs(semitones) > 2):
        raise ValueError("Difficult-source pitch tool supports a deliberate one/two-semitone change")
    if prepared.get("provenance", {}).get("pitch_adjustment"):
        raise ValueError("Pitch is already assigned to this whole source; reconcile both edges before changing it")
    from .storage_budget import validate_storage_path, shared_workspace, WorkBudget
    validate_storage_path(destination)
    destination = Path(destination).resolve()
    keys = [key for key in ("path", "vocal_path", "drum_path") if prepared.get(key)]
    if set(keys) != {"path", "vocal_path", "drum_path"}:
        raise ValueError("Pitch preparation requires all aligned mixture, voice and drum streams")
    infos = {key: sf.info(prepared[key]) for key in keys}
    geometry = {(v.samplerate, v.frames, v.channels) for v in infos.values()}
    if len(geometry) != 1 or next(iter(geometry))[2] != 2:
        raise ValueError("Pitch preparation streams do not share one stereo timeline")
    estimate = sum(i.frames*i.channels*8+4096 for i in infos.values())
    shared = shared_workspace(destination)
    if shared: WorkBudget(shared[1], [destination]).check(estimate)
    if destination.exists(): raise FileExistsError(destination)
    destination.mkdir(parents=True)
    from .render import _run
    from .execution import sha256
    result = deepcopy(prepared)
    receipts = []
    for key in keys:
        source = Path(prepared[key]).resolve()
        output = destination/(key+"-pitch.wav")
        _run([str(rubberband), "--fine", "--formant", "--centre-focus", "--ignore-clipping",
              "--quiet", "--time", "1", "--pitch", str(semitones), str(source), str(output)], cwd=destination)
        actual = sf.info(output); native = infos[key]
        if ((actual.samplerate, actual.frames, actual.channels) != (native.samplerate, native.frames, native.channels)
                or actual.subtype not in {"FLOAT", "DOUBLE"}):
            raise RenderError("Pitch-only processing changed the source timeline or PCM precision")
        result[key] = str(output)
        receipts.append({"stream": key, "source_sha256": sha256(source), "result_sha256": sha256(output)})
    result.setdefault("provenance", {})["pitch_adjustment"] = {
        "semitones": semitones, "scope": "entire source including both adjacent transitions",
        "method": "separate pitch-only formant-preserving R3 stage before ordinary timemap",
        "streams": receipts, "candidate_only": True,
        "requires_fresh_native_templates_clock_and_both_edge_checks": True}
    return result


def longer_overlap(plan, records, failure, *, bars, sample_rate=48000, ffmpeg=None):
    """One deliberately selected longer alternative, preserving order/locks.

    This advanced fallback is not an unbounded automatic render loop. It uses
    the ordinary arranger's source/solo/chapter constraints and the caller's
    existing retry budget. No shorter transition can be selected here.
    """
    require_refusal(failure)
    from .compact import arrange_sync
    proposal = deepcopy(plan)
    edge = next((e for e in proposal["edges"] if (e["from"], e["to"]) == (failure.outgoing, failure.incoming)), None)
    if edge is None: raise ValueError("Failed pair is absent from the plan")
    old = edge.get("overlap_beats", edge.get("overlap_bars", 32)*4)/4
    if type(bars) is not int or not old < bars <= 64:
        raise ValueError("Choose a longer supported overlap, at most sixty-four bars")
    if (edge.get("overlap_locked") or (edge.get("cue_locks") and not plan.get("sections"))
            or any(edge.get(k) is not None for k in ("handoff_windows", "whole_spectrum_handoff", "mid_handoff_bars", "bass_handoff_beats"))):
        raise ValueError("Longer overlap preserves explicit cues and handoff locks")
    for identity in (failure.outgoing, failure.incoming):
        section = plan.get("sections", {}).get(identity, {})
        row = next(r for r in records if r["id"] == identity)
        if section.get("locked_start") or section.get("locked_end") or row.get("protected_ranges"):
            raise ValueError("Longer overlap preserves locked/protected source material")
    edge.setdefault("cue_locks", {}).pop("incoming_start_seconds", None)
    edge["cue_locks"].pop("outgoing_end_seconds", None)
    edge["overlap_bars"] = bars
    edge.pop("overlap_beats", None)
    proposal.pop("sections", None); proposal.pop("rhythm_profiles", None)
    ranges = {key: {b: [s[b], s[b]] for b in ("start", "end")}
              for key, s in plan.get("sections", {}).items()}
    if failure.outgoing in ranges: ranges[failure.outgoing].pop("end")
    if failure.incoming in ranges: ranges[failure.incoming].pop("start")
    prefs = proposal.get("mixing_preferences", {})
    result = arrange_sync(proposal, records, sample_rate=sample_rate, ffmpeg=ffmpeg,
        zhilin=plan.get("style") == "zhilin", target_play_seconds=prefs.get("target_play_seconds", 210.),
        maximum_play_seconds=prefs.get("maximum_play_seconds", 300.), section_ranges=ranges,
        allow_long_play=bool(plan.get("long_play_explicit")))
    if result["order"] != plan["order"]: raise RenderError("Longer-overlap recovery changed approved order")
    for key, section in result.get("sections", {}).items():
        for flag in ("locked_start", "locked_end"):
            section[flag] = plan.get("sections", {}).get(key, {}).get(flag, False)
    result["difficult_overlap_candidate"] = {"pair": [failure.outgoing, failure.incoming],
        "old_bars": old, "new_bars": bars, "candidate_only": True, "requires_full_render_checks": True}
    return result


def retarget_pitch_record(record, prepared):
    """Point fresh source timing checks at the pitch-adjusted preparations.

    Original source identity is retained. Old template paths cannot describe
    changed guitar harmonics; normal rendering must construct new templates.
    """
    adjustment = prepared.get("provenance", {}).get("pitch_adjustment")
    if not adjustment or not all(prepared.get(k) for k in ("path", "vocal_path", "drum_path")):
        raise ValueError("Retargeting needs a complete whole-source pitch preparation")
    row = deepcopy(record)
    row["pitch_adjustment"] = deepcopy(adjustment)
    row.setdefault("vocal_stem", {}).update(path=prepared["vocal_path"],
        drums_path=prepared["drum_path"], drum_source_start_seconds=0.)
    row["kick_evidence"] = {role: {"path": prepared["drum_path"], "source_start_seconds": 0.}
                            for role in ("head", "tail")}
    return row


def instrumental_support(mixture, voice, sample_rate, *, overlap_frames,
                         restore_seconds=.75, failure):
    """Keep the next verse intact after instrumental backing in the overlap.

    Restore the estimate only inside an actually measured quiet post-overlap
    gap, so the next native word starts at its natural level. No repetition,
    per-word pumping or hidden jump at the end of the fade.
    """
    require_refusal(failure)
    if (mixture.shape != voice.shape or mixture.ndim != 2 or mixture.shape[1] != 2
            or not all(np.isfinite(x).all() for x in (mixture, voice))
            or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000
            or type(overlap_frames) is not int or not np.isfinite(restore_seconds)
            or not .25 <= restore_seconds <= 3.):
        raise ValueError("Instrumental support requires aligned source PCM and a gentle restore interval")
    width = round(restore_seconds*sample_rate)
    if not 0 < overlap_frames < overlap_frames+width <= len(mixture):
        raise ValueError("The native quiet restoration must fit after the overlap")
    from .transition_quality import _power
    power = _power(voice, max(1, round(.02*sample_rate)))
    floor = max(float(np.quantile(power, .9))*.0025, 1e-12) if len(power) else 1e-12
    gap = voice[overlap_frames:overlap_frames+width]
    quiet = _power(gap, max(1, round(.02*sample_rate)))
    if not len(quiet) or np.max(quiet) > floor:
        raise ValueError("Restoring the next voice would cut an active native word; choose another gap")
    from .resources import memory_policy
    if 32*mixture.size > memory_policy()["buffer_budget_bytes"]:
        raise ValueError("Instrumental support exceeds the current buffer budget")
    keep = np.ones((len(mixture), 1));keep[:overlap_frames] = 0.
    keep[overlap_frames:overlap_frames+width, 0] = smooth(np.linspace(0., 1., width))
    processed = (mixture-voice*(1-keep)).astype("float32")
    processed[overlap_frames+width:] = mixture[overlap_frames+width:]
    return processed, (voice*keep).astype("float32"), {
        "candidate_only": True, "overlap_frames": overlap_frames, "restore_frames": width,
        "native_quiet_restore_measured": True, "vocal_absence_certified": False,
        "native_body_after_restore_pcm_unchanged": True, "requires_fresh_both_edge_checks": True}
