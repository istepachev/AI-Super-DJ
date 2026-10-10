---
name: ai-super-dj
description: Build or review local DJ sets with fixed or optimized order, beat and phrase cues, selective restoration, mastering, verified FLAC export, and DJ.Studio project/export audits. Use for making or improving a DJ mix, including harmonic sequencing.
---

# AI Super DJ

Make a listenable set with an auditable plan: source tracks, actual order, transition cues, processing decisions, and final quality measurements. Read [references/workflow.md](references/workflow.md) for the analysis, transition, restoration, and export procedure. Read [THIRD_PARTY.md](THIRD_PARTY.md) before adding or changing an upstream adapter.

On a prepared workstation, optionally read `~/.cache/superdj/runtime.local.json` for the local Python environment, executable paths, external checkouts and output folder. This private configuration is not part of the repository. Otherwise follow the installation commands in [README.md](README.md).

## Preserve the requested set

- Use `fixed` mode when the user supplies an order. Keep every track in that order; report a difficult transition and adjust its technique instead of silently reordering or dropping tracks.
- Use `auto` mode when the user wants a proposed order. Respect any locked first/last track and other stated positions. Score the whole sequence using harmonic compatibility, tempo changes, energy progression, and available transition regions. Return the actual objective and limitations alongside the proposed sequence.
- If the user will supply the playlist later, build and validate the local workflow while waiting. Do not substitute a random playlist as the requested first set.
- When the user requests a random collection, sample from the specified source, keep the seed and selection in the private run manifest, and apply the requested ordering mode. A subsequent random-selection instruction replaces an earlier plan to wait for a supplied playlist.

## Musical decisions

Treat tempo, key, downbeat, and phrase boundaries as separate estimates with separate evidence. A beat tracker returning regular times does not verify the first beat of a bar. Every fourth beat is a bar hypothesis, and every sixteenth beat is not automatically a musical phrase.

The optional Beat This! neural backend predicts beats and downbeats across the complete track. Its output remains `model_predicted`; model confidence is not human verification or phrase detection. The lightweight librosa backend only supplies estimated beats. Import reviewed cues when available and retain which backend or human supplied each timing decision.

Review uncertain tempo/octave choices, changing-tempo tracks, bar starts, and phrase cues before a strict phrase-aligned render. A listening check or a supplied verified cue can resolve them. Store cue provenance and overrides. If evidence remains uncertain, produce a review plan or choose a clearly labelled free-time transition when the user permits it; do not report verified square alignment.

For ordinary blending, use the default `sync` renderer: both decks follow one integrated tempo curve and the same beat phase during the entire overlap. Equal BPM alone does not establish phase lock. The shared curve moves from outgoing native BPM to incoming native BPM during the overlap; the main sections stay native. Large differences default to four shared bars (`--sync-transition-bars`), with a separate declared temporary rate cap (`--sync-max-change`, default 40%). This can briefly speed up the incoming intro or slow the outgoing exit substantially; audition it rather than calling it optimal. Avoid long stretches at a middle BPM unless requested.

Check rendered attacks independently of the intended map. `superdj.sync` conservatively audits isolated source/output transients and precompensates measured backend drift with up to five re-renders from the original float clip. Ambiguous attacks remain unverified. Synthetic pulse tests do not certify every kick in dense music; mathematical anchors alone must not be described as acoustically perfect SYNC.

For suitable rhythms, consider a rational overlap such as four outgoing beats against three incoming beats, with only a small outgoing adjustment and selective whole-band gating. `superdj.polyrhythm` supplies explicit anchors and a mixer for this technique; its source-window and render orchestration is currently a manual caller workflow, not a CLI auto-choice. A 4:3 shared pulse is not 1:1 beatmatching or proof of aligned 4/4 phrases. Filtering does not separate drum stems.

Unknown tempo or deficient grids still require review; do not invent alignment. Creative exits are explicit choices: `adaptive` uses an outgoing-only vinyl slowdown for downshifts up to 35%, `echo_bridge` uses a sequential dry handoff through an echo. Neither is continuous 1:1 SYNC. `shared_tempo` retains the older constant-middle-BPM experiment with external body ramps. Never substitute a zero-overlap cut: `--allow-hard-cut` is only for a deliberate cut, and `--allow-unverified` permits a labelled short fade. A listener can prefer a well-chosen partial-rhythm transition to SYNC for a particular pair; retain that feedback without generalising the winner to every pair.

Store mandatory incoming-start and outgoing-end boundaries in each plan edge's `cue_locks`, using source seconds. The renderer either respects these locks on the supplied downbeat grid within configured trim limits or refuses the render; it must not relocate them to improve a score. A lock supplied for reproducible comparison is not evidence of human-reviewed phrasing.

Prefer small, pitch-preserving tempo changes and useful intro/outro or instrumental regions. Avoid two prominent vocal passages and two full bass lines playing together. Keep a difficult fixed-order boundary visible in the plan. Harmonic compatibility is a musical heuristic, not proof that a transition sounds good.

## Processing and delivery

- Decode originals into explicit float PCM. Retain source files. Apply restoration only for identified defects, then audition matched-level before/after excerpts; bypass a stage that harms transients, vocals, or stereo balance.
- Use actual available backends. Record their versions, settings, revisions, and any source adaptations. Label an instructions-only implementation as an adaptation, and a subset of a workflow as partial use.
- Keep intermediate audio in 32-bit float WAV. Make level matching and transition headroom part of the mix, then master the complete set once. Do not stack multiple aggressive per-track masters.
- Export the final lossless master as explicitly configured 24-bit FLAC at an appropriate sample rate. MP3 is an optional listening copy. FLAC preserves the processed signal; it cannot recover details removed by an earlier lossy encoder. Raising the bit depth or sample rate does not prove improved fidelity.
- Measure the encoded final file independently of the mastering processor: full decode, duration, channels, sample rate, bit depth, integrated loudness, sample peak, and oversampled true peak. Inspect all transition excerpts for rhythm, phrasing, vocals, and bass balance. Quantitative checks do not replace listening.
- Deliver audio, a timed track list, a transition/processing manifest, and a brief report of what was verified and what remains approximate. Do not claim a completed audio set when only a plan or synthetic test exists.
- For comparative listening, use `superdj.listening.excerpt_audit` for actual overlap and contextual bass/full-band energy. Record the user's ranking separately from measurements; there is no model predicting human preference. Inspect cue differences, pitch effects, tempo-change duration, bass handoff and transient alignment before attributing a preference to the algorithm alone.

## DJ.Studio projects and exported sets

For an existing `.djs` project or exported MP3/CUE, read [references/djstudio.md](references/djstudio.md). Use the project adapter for inspection and preserve the agreed order, cue points, custom curves, and vocal inserts. The adapter does not render DJ.Studio automation or edit its archive. CUE chapter intervals, selected source spans, and actual audible track durations are different measurements; do not substitute one for another or assume a stale project duration matches its export.

Keep any requested minimum track length and protected words/sections visible as constraints. The standalone renderer does not yet enforce minimum track duration, protected cue ranges, custom per-edge transition lengths, or vocal stems. A native project with these decisions needs a preserving workflow rather than a fresh generic automatic render. The proposed “своди как DJ Жилин” mode is deferred and must not be advertised as implemented.

## Local and public artifacts

Keep playlists, absolute source paths, audio, measurements, caches, model weights, secrets, and external checkouts in a private work directory outside the source repository; the CLI enforces this boundary. The public repository contains the skill, original orchestration code, tests using generated signals, and dependency provenance. Do not stage a user's music collection or listening results with the skill.

GitHub publication follows the user's requested sequence and existing authorization. For a request to test the first real set before publishing, finish that set and its review first; preparing the repository locally can proceed meanwhile.
