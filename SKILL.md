---
name: ai-super-dj
description: Enhance local music files or build DJ sets with structural cues, beat matching, selective repair, gentle frequency transitions, mastering and verified FLAC export. Use for audio improvement, harmonic sequencing or reviewing a DJ mix.
---

# AI Super DJ

Deliver listenable audio, retaining originals and recording actual processing. One input file means enhancement and mastering; several files or a folder mean a mixed set. Use `superdj make` on a prepared workstation. Read [README.md](README.md) for commands, [references/workflow.md](references/workflow.md) for detailed checks, and [THIRD_PARTY.md](THIRD_PARTY.md) before changing a backend.

Optionally read private `~/.cache/superdj/runtime.local.json` for workstation paths. Keep audio, playlists, models, caches and reports outside the source repository. Installing instructions alone does not install engines or weights.

## Preserve musical intent

Keep a supplied order (`fixed`, or `make --keep-order`). Otherwise propose a sequence using harmony, tempo, energy and transition regions. Respect locked positions, approved cues and protected ranges; do not silently omit tracks. For requested random selection retain its seed and selection in the private manifest.

For ordinary automatically ordered sets prefer a slower, lighter start and a faster, denser, bassier finish. Use the existing `build` development; a long set can use `wave` for a climb, breathing valley and another climb to the strongest finish. The planner applies position preferences to native BPM, measured intensity and bass, alongside transition feasibility. Treat these as a musical arc, not a requirement to increase every next track or force a playback-speed change. Use gain-invariant sound descriptors rather than file loudness for intensity/bass; they remain engineering proxies. Missing measurements contribute no guessed preference. Preserve explicit order and endpoints ahead of the arc.

For a requested Harmonize-style ordering experiment, use `harmonize` and `harmonize-compare`. Read [references/harmonize.md](references/harmonize.md). Fuzzy and Mood have different directed key rules; preserve every source and lock. Compare identical analysis before attributing differences to ordering. Freeze the existing profile on new files before fitting; recompute connected source cohorts when combining archives and hold out every linked old/new order/mode variant together. A new order of trained sources is not an independent source test. `harmonize-learn --learner modes` learns from BPM, key and the remaining pool. Compare literal Top3 on steps with at least four remaining tracks, ties and final small pools separately. A private profile assembles a complete route through `harmonize --ranking-profile`: bounded search retains alternative prefixes to the end, recomputes each actual remaining pool and normalizes conditional scores before adding them. A sequential candidate with exact small-tail bounds protects harmonic compatibility; both paths receive bounded repair. Keep original decisions separate from the selected final route. Compare directed neighbors and contiguous three/four-track chains; repeated choices with different remaining pools do not prove the same decision context. Keep local agreement, whole-route similarity and key allowance distinct. Development by intensity and bass belongs to ordinary sets, separately from BPM/key-only comparisons. Search and profiles are ours, not confirmed DJ.Studio internals; actual Solve warnings override theoretical allowance. Feed the selected order into fixed-order planning and SYNC rendering after source/timing checks.

For labelled ordering examples, keep Mood/Fuzzy, Key/BPM priority and tempo tolerance separate. The user's explicit settings override a backup filename; missing settings remain unknown. Priority-aware profiles consume these controls in both local ranking and hypothetical route states. Legacy profiles retain their earlier behavior. A fixed-parameter fit on one connected source cohort is an experimental training result, not held-out validation; save its profile privately for the next forward test.

For blocked demonstrated next tracks, compare native BPM distance with allowed alternatives before attributing the mismatch to ranking. Keep missing wizard priorities unknown; our key table is not DJ.Studio's Solve result. Report improvements in Top3 and complete routes separately, including regressions. When a new profile improves local ranking but worsens the requested whole-route agreement, retain the approved working profile and label the new profile as an experiment for the next test.

Treat beats, downbeats and phrases as separate evidence. Beat This! predicts beats/downbeats; CUE-DETR proposes structural cues from the source spectrogram. Fuse model suggestions with structural changes and the bar grid. Preserve imported reviewed cues even when the model disagrees. Counting sixteen bars alone does not identify a phrase. Record actual model execution, hashes, uncertainty and exceptions. Read [references/engines.md](references/engines.md) for setup and fusion details.

## Default mixing

For newly built sets use F (`sync`) as the standard transition: both decks follow one integrated tempo curve and one beat phase throughout the overlap. Prefer long compatible windows, usually thirty-two bars, sixteen for large tempo changes. Reduce automatic windows when evidence or a short source requires it. Preserve reviewed montage decisions and explicit locks.

Matching BPM alone is insufficient. Main sections retain native tempo. Choose the energetic central core using structural cues and audio features; do not play a whole extended intro/outro by default or assume numbered DJ cues exist. Target 180–240 seconds of solo play between overlaps, with a hard 300-second maximum. Retain at least 120 seconds including overlaps. Short sources need shorter solo play; a small minimum deficit can use a disclosed short, stable whole-bar repeat before the exit. Never invent beats or add silence to meet a duration. Bigger deficits or conflicting locks stop the run. Report exact solo, deck and overlap durations separately. See [references/transition-preferences.md](references/transition-preferences.md).

Use separate bass, middle and high-frequency curves tied to the shared beat phase. Pass bass over four beats near the middle. Softly transfer mids over eight to sixteen beats; initially attenuate the incoming mids and avoid holding two dense mid parties together. Highs use a gentler cross throughout the overlap. Paired band gains never exceed unity. Do not compensate attenuation by raising the entire incoming deck.

Use measured bass availability when selecting automatic windows: the outgoing source should support the handoff and the incoming source should already carry bass there. Avoid bass-empty intro/outro regions while allowing short musical fills. The 35–180 Hz profile is an energy proxy, not a kick detector; reviewed quiet cues remain constraints. Check the actual rendered overlaps for energy holes before delivery.

When local Demucs estimates are available, prefer cues with less competing vocal activity and selectively hand over foreground estimates. Apply the same timing and linear preparation to mixture and voice estimates. Bypass subtraction when estimates cannot be aligned or one voice is insignificant. Stems contain leakage; inspect vocal interference in the actual result.

Check rendered attacks independently of the intended map. The engine can compensate measured isolated-transient drift with up to five re-renders from original float audio. Ambiguous dense attacks remain unverified; mathematical anchors do not certify acoustic perfection.

## Optional DJ Жилин

Use `make --style zhilin` when requested. Prefer forty-bar compatible overlaps with the same core selection, shared SYNC and separate frequency curves. Preserve approved durations/cues ahead of the preference, including forty/forty-four bars and explicit partial-bar beat counts. Source availability and protected material may require shorter overlap. This mode does not guarantee a copy of a proprietary montage. Details are in [references/engines.md](references/engines.md).

## Sound and delivery

Decode originals to float PCM. Repair identified DC/noise defects selectively, use restrained content-dependent EQ, and bypass unnecessary processing. Avoid stacked aggressive per-track masters. Make one complete-set master with headroom; preserve native sample rate for single-file processing and declare set resampling.

Export explicitly configured twenty-four-bit FLAC; MP3 is an optional listening copy. Independently check the entire encoded file: decode, sample rate, channels, bit depth, duration, integrated loudness, sample peak and oversampled true peak. FLAC preserves the processed signal without another lossy encoding stage; it does not reconstruct information removed by MP3.

Deliver audio, timed track list for a set, private processing provenance and a brief verification report. Keep the user's audition/ranking separate from measurements. Compare excerpts at matched loudness. A plan or synthetic test alone is not a completed musical set.

For an existing project or exported CUE/audio, read [references/djstudio.md](references/djstudio.md). The advanced preserving rewrite API changes specified fields in a new archive and checks retained members; application import and reproduction of automation remain separate checks. Keep this capability out of the ordinary one-file/many-file flow.
