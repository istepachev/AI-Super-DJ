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

Default hybrid cues combine local CUE-DETR suggestions when configured, measured RMS/bass/high-band structure and the bar grid. Imported cues remain exact. Optional Demucs estimates add vocal-risk scoring and selective handoff. Missing engines are disclosed. Read [engines.md](engines.md) for setup and limitations.

## Sequence planning

In `fixed` mode compute the cost and technique of each existing boundary, retaining the supplied sequence. In `auto` mode search the complete selected collection, subject to locked positions; compare against a reproducible greedy baseline using the same objective and constraints. By default collections of at most 17 tracks use exact Held–Karp dynamic programming with position intensity, native BPM and bass costs. Larger collections use bounded beam search and directional segment improvement. Record the actual method and limits. Exact optimization means optimal for the declared numerical objective, not objectively best audible music; bounded search has no global-optimum guarantee.

The default `build` prefers a slower, lighter opening and stronger, faster finish within the input's measured range. For a long set `wave` climbs, dips and climbs again to its highest final target. Both shapes use the same existing development setting: no separate ordinary-user controls are needed. Their tempo targets express preferred placement of native-tempo tracks, not changes to a deck's playback speed. Harmony, transition feasibility and explicit endpoints can outweigh these soft preferences. Fixed order is never rewritten to satisfy an arc.

Analysis 1.6 measures dimensionless intensity from relative envelope fullness, spectral entropy and normalised spectral variation. Bass uses the fraction of musical power in 35–180 Hz. Both descriptors ignore an unclipped scalar gain; RMS dBFS remains a separate level measurement. These are uncalibrated proxies, not a learned judgement of musical excitement, emotion or kick strength. Missing sound evidence contributes zero position cost; legacy RMS-derived energy is excluded until fresh analysis supplies better evidence. The plan records targets, source methods and separate intensity/tempo/bass contributions for each position.

Useful edge-cost terms are Camelot/key compatibility with uncertainty, stretch needed for common tempo, intensity difference relative to the intended arc, transition-region suitability, and vocals/bass overlap. An uncertain key should weaken its influence. The analyzer's sustained midband proxy is not a learned vocal detector: guitars and synths can resemble vocals. A high-confidence tempo conflict or missing phrase cue should remain visible even when the combined numerical score is good. Give proposed candidates and reasons when the user asks for the next track. The reported seven candidates use immediate edge plus next-position intensity/tempo/bass cost, not seven alternative globally optimized playlists. The old top-three field remains a compatibility subset.

AI Super DJ's transparent optimizer is an independent approach. Do not describe it as an implementation or reproduction of DJStudio's proprietary Harmonize AI unless its internals have actually been made available and verified.

## Transition plan and review

For each boundary store outgoing/incoming cue times, tempo ratios, beat phase, intended overlap bars, beat count per bar, effective duration, bass/vocal policy, confidence, and review state. Use the user's verified cues ahead of detector suggestions.

For 4/4 music, 8, 16, and 32 bars correspond to 32, 64, and 128 beats. This arithmetic defines a planned length; it does not detect a phrase. Check the cue's musical content and grid position. In other meters use their actual beat count and phrase structure.

Preserve the beat-to-time mapping after stretching. For a constant BPM change with unchanged pitch, duration ratio is source BPM / target BPM; the inverse is a tempo multiplier. The Rubber Band CLI `-t` argument is a duration ratio. Check output length and transient positions rather than trusting an option label.

Create excerpts containing preceding music, overlap and following music. Audition when an actual listening tool is available; otherwise mark listening pending. Check kicks, snares, phase drift, phrasing, vocals, bass and energy continuity. New sets use shared-phase F SYNC at every boundary with independent bass/mid/high transfer. Existing reviewed projects and explicit creative choices remain constraints.

The explicit `--tempo-conflict-policy echo_bridge` handles an unresolved edge through a filtered finite echo tail. Both tracks retain native timing: old dry drums fade before the new dry signal enters, while a high-pass echo carries the handoff. `--bridge-overlap-beats` sets the region length in incoming beats (4 to 32, default 8). This arithmetic does not establish phase/phrase alignment. The bridge requires reliable native cue grids; an unavailable cue or oversized region stops the render. Its manifest records the effect settings and non-beatmatched status. Echo and gain coefficients are bounded to preserve the prepared headroom and source endpoints; this does not certify good sound. Automatic selection of a break or atmospheric section is future work.

Mandatory boundaries use `"cue_locks": {"incoming_start_seconds": 15.04, "outgoing_end_seconds": 244.18}` (example source seconds). They must fit the supplied grid and leave a complete locally valid window. New plans target 180–240 seconds of solo play, maximum 300; choose an energetic central core rather than an extended intro/outro. Explicit partial-bar exceptions use beat-grid locks. A minimum of 120 seconds including overlaps is enforced; a small automatic-source deficit can use a recorded stable whole-bar repeat. Protected/locked material is not extended. The preserving legacy source-span API remains separate.

For CLI `plan` or `run`, supply `--transition-overrides` with a private JSON array such as `[{"from":"a","to":"b","cue_locks":{"incoming_start_seconds":15.04,"outgoing_end_seconds":244.18}}]`. Only actual adjacent pairs in the resulting order are accepted; unknown, reversed or duplicate pairs are refused. A standalone `render` consumes locks already stored in its supplied plan. These locks specify selected source boundaries, not guaranteed audible boundaries: an echo bridge initially mutes the incoming dry signal for two incoming beats and then fades it in over one beat. The full timing is recorded in `echo_processing`.

Legacy ordinary constant-tempo APIs retain a 6% change cap. Standard F shared SYNC uses a separate temporary cap (`--sync-max-change`, default 40%) during overlaps; solo music retains native tempo. Above-cap transitions require another decision. Sequence costs still use their declared tempo objective; `--no-tempo-reset` restricts that optimizer, not the independent SYNC cap.

New `plan`/`run`/`make` sets explicitly choose SYNC for all boundaries. Both measured grids map equal beat counts to one integrated smoothstep BPM phase; subbeat controls share exact output frames. Incoming overlap end and outgoing overlap start connect to native-speed bodies. Prefer 32 bars, 16 above 20% temporary correction, with automatic shortening when source/grid/body constraints require it. DJ Zhilin prefers 40 compatible bars. Explicit overlap lengths/locks take priority. Advanced raw plans can still use legacy `auto` ordinary-first behavior or reviewed creative methods; the standard path does not cascade through E/C/cut. Independent complementary frequency gains avoid full doubled bass/mid parties; mids hand off over 8–16 beats, highs throughout the transition. See [transition-preferences.md](transition-preferences.md).

`superdj.creative` exposes `tape_exit_curve`, `tape_speed_warp`, `mix_tape_exit` and the combined `tape_exit_overlap` for custom cue-driven renders. Use the exact required source-frame count; do not feed a longer tail then accidentally discard its end or repeat audio at the join. `superdj.polyrhythm` exposes rational beat anchors and whole-band partial-rhythm mixing. For example, 16 outgoing beats can meet 12 incoming beats in a 4:3 handoff while keeping the incoming clock native. Only the outgoing window is adjusted; off-cycle emphasis is reduced with smooth band gates. This is not isolated drum editing, 1:1 matching or automatic phrase recognition. Select, render and review this technique with an explicit caller; the CLI does not choose it automatically.

For large tempo changes provide a short, level-matched comparison. Record both caps, target tempo, per-deck changes, exact cues, actual ramp durations and local corrections. Compare encoded files at the same listening level and report that mathematical grid alignment is separate from post-render transient and musical listening checks.

## Restoration and upstream adapters

Choose one documented repair for one detected problem. A clean commercial master can proceed without restoration. Avoid automatic denoising of every quiet passage, synthetic bandwidth expansion, or declipping every loud sample. These can alter intentional musical detail.

The preferred upstream roles are audamcp (`dpf-audio-pro`) for conservative restoration and Bitwize for final mastering. They are external, pinned dependencies. Check their actual call signatures and array layout: audamcp commonly uses channels × samples; soundfile and Bitwize commonly use samples × channels. Preserve the explicit float layout across an adapter boundary.

Test a repair on a known defective signal and an undamaged control before using it. At the audited audamcp revision, `repair.declick` compares a dB residual with `abs(threshold_db)`, so its negative default threshold becomes positive; its documentation also describes interpolation more strongly than its implementation. Do not rely on this stage without a tested fix or disclose it as bypassed. Record any patch with upstream revision, patch description, and behavior test.

For every active adapter, record repository URL, exact revision or package version, executed module/function, parameter values, and modifications. Dependency presence does not establish that a stage ran. If a backend is unavailable, select a supported replacement and describe it, or leave the stage pending; do not retain its name in a claimed execution chain.

## Mastering and export

Keep enough premix headroom for overlaps. Apply a restrained, reference-informed complete-set master; prefer preserving the groove and dynamics over chasing a loudness number. If loudness and true-peak targets conflict, report the tradeoff rather than adding uncontrolled limiting. Use a stated target chosen for the user's listening use.

Retain the float premix. The Bitwize final-master adapter writes a 24-bit PCM WAV with `output_bits=24` and `dither_bits=24`; neither default is relied on. Explicitly set every final codec parameter. A 24-bit FLAC must be verified as 24-bit; extension and command success are insufficient. The advanced renderer defaults to stereo 44.1 kHz unless configured; make preserves single-file sample rate and uses the highest supplied rate for a set. Report resampling explicitly. Avoid presenting resampling as recovered detail. Dither once at the final integer quantization if the chosen export process uses it.

Measure the final encoded file independently using FFmpeg's full-file `ebur128=peak=true` and `astats`; do not reuse the mastering function's pre-export peak report as final QA. Record integrated LUFS, loudness range in LU, sample peak in dBFS, and true peak in dBTP (FFmpeg labels that summary as dBFS). Cover the whole file for decode, loudness, and peaks. The initial export defaults to a −1.0 dBTP maximum after a −1.5 dBTP mastering ceiling; encoded overshoots can trigger documented gain-only re-exports, with at most three attempts per format. Check endpoints and unexpected silence without treating intended breakdowns as errors. Verify the full track count and final order against the user's playlist.

The delivery report separates objective checks, audible transition review, and unresolved judgments. FLAC is lossless relative to the rendered PCM, not a guarantee of original-master fidelity. A verified synthetic render proves mechanics only; the first real musical set remains a separate milestone.

## Independent attack and listening diagnostics

A sample-frame timing map is intended phase, not proof of acoustic phase after processing. `superdj.sync.transient_drift` compares unambiguous isolated source/output envelope peaks, accounting for the original source attack offset. Backend drift above 3 ms can trigger up to five compensated re-renders from the original float PCM. Missing/ambiguous attacks receive no new correction; the report retains coverage and residual error. Generated pulse tests cover actual renders in both tempo directions, but this conservative method cannot certify all kicks in dense mixed recordings. Phrase and human listening review remain separate.

`superdj.listening.excerpt_audit` measures full-band and 35–180 Hz energy across an actual overlap, its four-second context and its weakest one-second window. Compare these alongside pitch changes, tempo-alteration duration, bass handoff, vocal interference and cue selection. A bass-band measurement includes instruments besides kicks. Preserve the human ranking separately: these metrics are not a model of human liking, nor does equal whole-file LUFS prove locally equal loudness. Different source cues confound a comparison of algorithms. Use feedback to guide the next comparison, not to claim universal superiority.
