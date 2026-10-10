---
name: ai-super-dj
description: Enhance local music files or build DJ sets with structural cues, beat matching, selective repair, gentle frequency transitions, mastering and verified FLAC export. Use for audio improvement, harmonic sequencing or reviewing a DJ mix.
---

# AI Super DJ

Deliver listenable audio, retaining originals and recording actual processing. One input file means enhancement and mastering; several files or a folder mean a mixed set. Use `superdj make` on a prepared workstation. Read [README.md](README.md) for commands, [references/workflow.md](references/workflow.md) for detailed checks, and [THIRD_PARTY.md](THIRD_PARTY.md) before changing a backend.

Optionally read private `~/.cache/superdj/runtime.local.json` for workstation paths. Keep audio, playlists, models, caches and reports outside the source repository. Installing instructions alone does not install engines or weights.

## Preserve musical intent

Keep a supplied order (`fixed`, or `make --keep-order`). Otherwise propose a sequence using harmony, tempo, energy and transition regions. Respect locked positions, approved cues and protected ranges; do not silently omit tracks. For requested random selection retain its seed and selection in the private manifest.

Treat beats, downbeats and phrases as separate evidence. Beat This! predicts beats/downbeats; CUE-DETR proposes structural cues from the source spectrogram. Fuse model suggestions with structural changes and the bar grid. Preserve imported reviewed cues even when the model disagrees. Counting sixteen bars alone does not identify a phrase. Record actual model execution, hashes, uncertainty and exceptions. Read [references/engines.md](references/engines.md) for setup and fusion details.

## Default mixing

First use the existing ordinary beatmatched blend, common tempo and normal windows: usually sixteen bars, shorter when needed. Long F (`sync`) is the fallback when an ordinary blend cannot work and would otherwise leave a discontinuity. Successful ordinary blends and reviewed montages are not replaced wholesale.

F integrates one tempo curve with one beat phase for both decks throughout the overlap; matching BPM alone is insufficient. Main sections retain native tempo. Conflict F defaults to thirty-two bars, or sixteen above 20% temporary rate change. Short F is used when requested or approved for the pair. Preserve pair-specific preferences and cue locks. The F/E/C/cut ranking applies to unresolved transitions; see [references/transition-preferences.md](references/transition-preferences.md). The CLI automatically tries ordinary then F; later creative alternatives remain agent-directed. Unknown tempo or inadequate timing evidence must not be presented as verified alignment.

Use separate bass, middle and high-frequency curves. Hand over bass without two full bass lines; open incoming mids gradually, initially quieter when contrasting timbres would cause abrupt entry. Do not compensate this attenuation by raising the entire incoming deck. These defaults apply to ordinary and F overlaps.

When local Demucs estimates are available, prefer cues with less competing vocal activity and selectively hand over foreground estimates. Apply the same timing and linear preparation to mixture and voice estimates. Bypass subtraction when estimates cannot be aligned or one voice is insignificant. Stems contain leakage; inspect vocal interference in the actual result.

Check rendered attacks independently of the intended map. The engine can compensate measured isolated-transient drift with up to five re-renders from original float audio. Ambiguous dense attacks remain unverified; mathematical anchors do not certify acoustic perfection.

## Optional DJ Жилин

Use `make --style zhilin` when requested. Select energetic structural source spans targeting three to four minutes and prefer long compatible overlaps. Preserve approved durations/cues ahead of that preference, including forty/forty-four bars and explicit partial-bar beat counts. Separate selected source span, rendered deck span and audible foreground duration. Source availability, protected material and harmony may require shorter overlap. This mode uses the same mixer; it does not guarantee a copy of a proprietary montage. Details are in [references/engines.md](references/engines.md).

## Sound and delivery

Decode originals to float PCM. Repair identified DC/noise defects selectively, use restrained content-dependent EQ, and bypass unnecessary processing. Avoid stacked aggressive per-track masters. Make one complete-set master with headroom; preserve native sample rate for single-file processing and declare set resampling.

Export explicitly configured twenty-four-bit FLAC; MP3 is an optional listening copy. Independently check the entire encoded file: decode, sample rate, channels, bit depth, duration, integrated loudness, sample peak and oversampled true peak. FLAC preserves the processed signal without another lossy encoding stage; it does not reconstruct information removed by MP3.

Deliver audio, timed track list for a set, private processing provenance and a brief verification report. Keep the user's audition/ranking separate from measurements. Compare excerpts at matched loudness. A plan or synthetic test alone is not a completed musical set.

For an existing project or exported CUE/audio, read [references/djstudio.md](references/djstudio.md). The advanced preserving rewrite API changes specified fields in a new archive and checks retained members; application import and reproduction of automation remain separate checks. Keep this capability out of the ordinary one-file/many-file flow.
