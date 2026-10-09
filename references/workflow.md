# Set workflow

## Inputs and working directory

Use the playlist the user supplies. Resolve filenames against the intended music directory; ambiguity or a missing file needs a precise request for that item. Read any supplied order, cue times, tempo, key, first/last track, transition length, energy arc, and reference master as constraints. A fixed playlist defaults to `fixed`; optimization requires `auto` or another explicit request to change order.

Store private analysis and render artifacts outside the repository; even an ignored directory inside the repository is rejected by the CLI. The manifest must identify every source locally without placing those identifiers in public documentation. Hash source content when reproducibility matters. The original collection remains the source of truth. For a requested random selection, keep the seed, collection scope, and exact chosen files in this private manifest.

The command stages are `analyze`, `plan`, `render`, and `run`; `setup` provisions pinned external processor checkouts in a private directory. Check the installed CLI help for the exact argument names and available backends. `run` orchestrates the stages; it must preserve unresolved review conditions rather than bypass them.

## Analysis and confidence

Measure at the original rate or a documented analysis-only resampling rate. Analyze more than one region when tempo, key, or structure varies. Distinguish measured source quality from the quality of a detector's estimate.

| Feature | Evidence to retain | Review trigger |
|---|---|---|
| Source | Codec, sample rate, channels, declared bit depth, duration | A reported FLAC could still contain formerly lossy audio |
| Defects | Audible excerpts, clipping runs, impulses, hum/noise evidence | A loud mastered track or narrow bandwidth alone is insufficient for repair |
| Tempo | Candidate BPMs, local intervals, stability and octave alternatives | Half/double time ambiguity, sparse intro, drift |
| Key | Windowed candidates, confidence/margins, disagreement | Percussive regions, modulation, ambiguous relative keys |
| Downbeat | Candidate beat index/time and evidence, manual/model provenance | Generic beat extraction only, unresolved 4/4 phase |
| Phrase | Cue times, bar counts, musical section evidence | Bars counted from an uncertain downbeat, nonstandard phrases |
| Mixability | Intro/outro regions, vocal activity, bass content, energy | Dense vocals, abrupt endings, no usable transition region |

Confidence values from an uncalibrated heuristic are diagnostics, not probabilities. Retain an explicit review status. Do not turn an analysis excerpt's local first beat into a verified full-track bar start without checking the offset.

`beat_this` uses Beat This! 1.1.0 with an explicitly selected local checkpoint, normally `final0`. Its code and published weights have MIT licenses. The optional model download fetches the official checkpoint into the private cache and records its SHA-256; no training recordings are fetched. Install PyTorch/torchaudio for the selected CPU or CUDA environment. `model_predicted` indicates an actual neural downbeat prediction, not a reviewed bar grid. `librosa` supplies beat estimates without a learned bar-start model. A reviewed imported grid is a separate provenance path.

## Sequence planning

In `fixed` mode compute the cost and technique of each existing boundary, retaining the supplied sequence. In `auto` mode search the complete selected collection, subject to locked positions; compare against a reproducible greedy baseline using the same objective and constraints. By default collections of at most 17 tracks use exact Held–Karp dynamic programming with position energy costs. Larger collections use bounded beam search and directional segment improvement. Record the actual method and limits. Exact optimization means optimal for the declared numerical objective, not objectively best audible music; bounded search has no global-optimum guarantee.

Useful edge-cost terms are Camelot/key compatibility with uncertainty, stretch needed for common tempo, energy difference relative to the intended arc, transition-region suitability, and vocals/bass overlap. An uncertain key should weaken its influence. The analyzer's sustained midband proxy is not a learned vocal detector: guitars and synths can resemble vocals. A high-confidence tempo conflict or missing phrase cue should remain visible even when the combined numerical score is good. Give proposed candidates and reasons when the user asks for the next track. The reported top-three candidates use immediate edge plus next-position energy cost, not three alternative globally optimized playlists.

SuperDJ's transparent optimizer is an independent approach. Do not describe it as an implementation or reproduction of DJStudio's proprietary Harmonize AI unless its internals have actually been made available and verified.

## Transition plan and review

For each boundary store outgoing/incoming cue times, tempo ratios, beat phase, intended overlap bars, beat count per bar, effective duration, bass/vocal policy, confidence, and review state. Use the user's verified cues ahead of detector suggestions.

For 4/4 music, 8, 16, and 32 bars correspond to 32, 64, and 128 beats. This arithmetic defines a planned length; it does not detect a phrase. Check the cue's musical content and grid position. In other meters use their actual beat count and phrase structure.

Preserve the beat-to-time mapping after stretching. For a constant BPM change with unchanged pitch, duration ratio is source BPM / target BPM; the inverse is a tempo multiplier. The Rubber Band CLI `-t` argument is a duration ratio. Check output length and transient positions rather than trusting an option label.

Create transition excerpts containing the preceding section, overlap, and following section. Audition them when an audio listening tool is available; otherwise explicitly mark human listening review pending and provide the excerpts for the user. Check kicks and snares, phase drift, phrase changes, vocal collisions, and bass build-up. Use a shorter blend, instrumental cue, controlled bass exchange, echo exit, or free-time cut when the material calls for it. Record the actual technique rather than calling every crossfade beatmatched.

## Restoration and upstream adapters

Choose one documented repair for one detected problem. A clean commercial master can proceed without restoration. Avoid automatic denoising of every quiet passage, synthetic bandwidth expansion, or declipping every loud sample. These can alter intentional musical detail.

The preferred upstream roles are audamcp (`dpf-audio-pro`) for conservative restoration and Bitwize for final mastering. They are external, pinned dependencies. Check their actual call signatures and array layout: audamcp commonly uses channels × samples; soundfile and Bitwize commonly use samples × channels. Preserve the explicit float layout across an adapter boundary.

Test a repair on a known defective signal and an undamaged control before using it. At the audited audamcp revision, `repair.declick` compares a dB residual with `abs(threshold_db)`, so its negative default threshold becomes positive; its documentation also describes interpolation more strongly than its implementation. Do not rely on this stage without a tested fix or disclose it as bypassed. Record any patch with upstream revision, patch description, and behavior test.

For every active adapter, record repository URL, exact revision or package version, executed module/function, parameter values, and modifications. Dependency presence does not establish that a stage ran. If a backend is unavailable, select a supported replacement and describe it, or leave the stage pending; do not retain its name in a claimed execution chain.

## Mastering and export

Keep enough premix headroom for overlaps. Apply a restrained, reference-informed complete-set master; prefer preserving the groove and dynamics over chasing a loudness number. If loudness and true-peak targets conflict, report the tradeoff rather than adding uncontrolled limiting. Use a stated target chosen for the user's listening use.

Retain the float premix. The Bitwize final-master adapter writes a 24-bit PCM WAV with `output_bits=24` and `dither_bits=24`; neither default is relied on. Explicitly set every final codec parameter. A 24-bit FLAC must be verified as 24-bit; extension and command success are insufficient. The initial preparation/render profile uses stereo 44.1 kHz, so higher-rate sources are resampled and that choice must be reported. Avoid presenting resampling as recovered detail. Dither once at the final integer quantization if the chosen export process uses it.

Measure the final encoded file independently using FFmpeg's full-file `ebur128=peak=true` and `astats`; do not reuse the mastering function's pre-export peak report as final QA. Record integrated LUFS, loudness range in LU, sample peak in dBFS, and true peak in dBTP (FFmpeg labels that summary as dBFS). Cover the whole file for decode, loudness, and peaks. The initial export defaults to a −1.0 dBTP maximum after a −1.5 dBTP mastering ceiling; encoded overshoots can trigger documented gain-only re-exports, with at most three attempts per format. Check endpoints and unexpected silence without treating intended breakdowns as errors. Verify the full track count and final order against the user's playlist.

The delivery report separates objective checks, audible transition review, and unresolved judgments. FLAC is lossless relative to the rendered PCM, not a guarantee of original-master fidelity. A verified synthetic render proves mechanics only; the first real musical set remains a separate milestone.
