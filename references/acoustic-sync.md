# Acoustic SYNC verification

## Distributed local rhythm coverage

Keep the existing whole-window recognition gate and add local checks for every ordinary shared-phase overlap, independent of Mood/Classic/Fuzzy ordering and transition style. Examine sixteen shared beats at a time, stepping four beats and including the final window. With at least four audible interior beat slots, require at least two recognized slots and at least 25% local coverage. More than eight consecutive audible slots without a recognized event rejects the window, even if its later part has enough good matches. Duplicate template views count as one beat; unknown events are not invented or moved onto the clock.

Audibility uses the actual bass, middle, high and routed percussion envelopes, not only the bass fader. A zero drum estimate gain does not prove residual accompaniment is free of transients. Physically quiet aligned native input is excluded using its measured beat-window power; its silence is never registered as a verified hit. Short native fills and sufficiently distributed sparse figures retain their uncertainty. This is a conservative source-window eligibility gate, not proof that every unknown figure is a misplaced kick.

Record the failed local intervals and retain the actual failed source/role for bounded cue recovery. Do not repair this failure by accelerating every transient, weakening the 3 ms timing checks, shortening the approved overlap or deleting working orchestration variants. Changing this guard invalidates overlap cache evidence through its transitive code fingerprint. A private excerpt can use full-deck budgeting for its missing outer neighbours; validate the complete candidate set's CUE pacing separately before claiming that it is ready for release.

## Gradual orchestration recovery

Two independently aligned decks can still conflict during their combined bass/percussion crossover. Before selecting different native source windows, ordinary shared-phase styles can try at most three other placements of the same gradual transfer inside the unchanged overlap. Preserve its full width, the shared clock, native source bounds and all middle/high curves. An explicit bass position or width prevents this automatic relocation. The candidates must pass the same native-identity, required-attack, paired-front, 3 ms, actual energy, estimated foreground and recognition-coverage checks. Exhaustion retains a failure; it never authorizes a quick cut, pitch-changing correction or weaker thresholds. Store the chosen beat span and failed alternatives in `gradual_orchestration_search`. All verification components must use the actual tonal-conflict and band settings, including ordinary-style attenuation.

## Native figures and actual map safety

A recovered familiar kick waveform is evidence of identity, not proof that it is the primary hit for the nearest model beat. Against an established native phase family, observations displaced by more than the bounded phase tolerance remain `native_rhythm_observations`; they are retained at source timing, excluded from quarter-note controls, and not counted as certified on-beat primary attacks. A global consistent grid bias is supported; it is the deviation from the established family that matters. Reviewed variable-meter evidence keeps its explicit exception. Do not erase a real fill or invent a missing primary hit.

The immutable `timing_rate_reference` records the intended musical rates separately from acoustic timing corrections. `actual_timing_rate_audit` recomputes every final source-frame/output-frame slope after source refinement, after backend compensation and before a clip reaches mixing or delivery. Deliberate musical tempo changes are normalized out when applying the existing timing limit. Statistics are updated from the actual map, rather than copied from its preliminary version. A rejected compensation restores the last safe anchors; a failed source refinement requires another source window. Cached timed PCM is checked as well. Do not increase the limit to accommodate a recovery or claim a small initial statistic for a later modified map.

When an earlier leading front is ambiguous or displaced native figures would compete, full estimated percussion uses a gradual complementary transfer of at least eight shared beats while retaining the accompaniment curves and approved overlap. A failed combined check can try this routing after both independent deck checks pass, then must pass the same 3 ms gate; failure requires another source window. A separate actual percussion-gain span gate rejects a one-beat core switch even when bass/middle accompaniment and recognized hit times pass. Sequential percussion remains an explicit advanced effect, never a hidden automatic repair. The combined audit includes recognized events while percussion gain is at least 0.05. Separation estimates and unknown transients remain uncertainty.

Beat-model labels and matching BPM establish a preliminary time map, not real attack coincidence. Source kicks can precede or follow a coarse label, and the stretching backend can shift attacks further. Refine supported primary attack positions before rendering; backend correction alone preserves an original grid offset.

Use native prepared audio or time-aligned drum estimates as source evidence. Drum stems are estimates with leakage, not ground truth. Separate a sharp onset from a low-frequency body maximum: different kick timbres can have different body delays. Repeated attacks support a local source grid; isolated peaks require an isolated-impulse test. Continuous bass must not be accepted as a kick sequence. Report measured attacks separately from grid points inferred between them. Missing kicks, fills, intentional swing and ambiguous events remain unverified, rather than being invented or forced onto every subdivision.

For variable native tempo, fit residual attack offsets against the native beat sequence rather than replacing the whole section by a constant BPM. Interpolate missing offsets on that native sequence. An explicitly reviewed native meter can also use a strongly repeated waveform family when live attack timing defeats a straight-line fit: require at least eight shapes with normalized similarity of 0.85 and substantial candidate coverage. This waveform route does not independently certify the meter. If source offsets would make quarter controls nonmonotonic, keep the original monotonic map and require protected native attacks plus new PCM verification; do not accept impossible controls.

For dense percussion, match a short band-limited template of each recognized native primary attack in the rendered PCM. A generic novelty peak may select a different nearby sound even in a drum estimate. Require adequate normalized waveform correlation and separation from alternative peaks; report ambiguous matches as gaps. Preserve template artifacts privately so later checks do not depend on a deleted temporary clip. Fresh separation of rendered drums can provide another PCM view for template verification; an unconstrained peak detector alone cannot establish that it measured the same kick.

Before rendering, search unsupported native beat windows for an already established same-source primary attack shape. A louder fill can hide the true kick from the maximum-novelty detector. Require source waveform correlation of at least 0.90, an alternative-peak margin of 0.05, and adequate signal power. Only an established family with at least eight references can add events; silence, tonal bass and ambiguous duplicate matches stay unsupported. Store the reference-match evidence and refine the recovered attack too. Do not exclude a genuine primary hit merely because its source offset differs from the dominant straight-line family.

After rendering, measure observed attack time against the common output clock. Compare outgoing and incoming observed errors on each jointly recognized beat. Do not compare only backend latency relative to each deck's original offset, and do not average away one large miss. Current detector gate is a maximum of 3 ms on jointly measured primary attacks, with at least eight shared recognized events per overlap. This threshold is an engineering estimate and does not guarantee every percussion event or sample-level waveform phase. Record recognized and unverified beat counts explicitly; no measurements cannot produce a successful alignment claim.

Interlocking rhythms can have few simultaneous recognized attacks while both follow the same clock accurately. With one to seven actual direct coincidences, require at least eight recognized attacks on each deck and a maximum individual clock error of 1.5 ms on every measured event, including those without an opposite-deck event. Record this as sparse direct verification plus individual lattice verification, with the conservative sum of both maximum errors; do not call it eight paired coincidences. Zero direct coincidences still stops this verification path. Do not shift a groove or shorten a window merely to manufacture simultaneous hits.

For backend compensation, regenerate from the original float input rather than repeatedly stretching an already processed file. Damped control updates can reduce oscillation; keep the candidate with the smallest measured maximum, and restore its map as well as its audio. Bound attempts and stop unresolved exports. Re-render voice estimates with the identical final deck map before subtraction or handoff. Preserve approved source windows, overlap lengths, frequency curves and native bodies during a timing repair.

When stretching substantially changes a supported attack shape or prevents reliable timing recovery, the renderer may preserve a short native attack island at its shared-clock position. Current islands span 135 ms, or up to 170 ms for an explicitly recorded variable-meter fallback, with 15 ms cosine joins outside the attack template. Original pitch and waveform are retained inside the island; the surrounding stretched audio remains. Apply the same islands to foreground estimates, refuse overlapping islands or missing source context, and audit the actual result again. Source positions and copying are not verification by themselves. Save the prepared-mixture waveform around each previously identified source percussion attack and locate it in actual output PCM with correlation at least 0.90. This verifies physical preservation of that recognized source region; it does not independently validate percussion classification or unknown fills. Foreground routing must not overwrite the mixture templates. If voices mask the result, fresh drum estimates provide another view, but separation can alter weak attacks or leak melodic material. Preserve conflicting observations; a weak mixture match can be rejected only when the fresh percussion match is substantially stronger, with the decision recorded privately. This fallback can change local texture at the joins and needs musical review.

Accurate recognized primary attacks do not establish compatibility of every drum figure. Inspect unverified events especially around the bass handoff: fills, swing and offbeat attacks can already be displaced in the original recording. Compare original and rendered positions before attributing these to stretching. Do not quantize every fill onto quarter notes. Choose a compatible source window of the same approved overlap length when displaced figures conflict with the other deck. Preserve broad mids where they still support the music. Never describe a small maximum on recognized events as proof that an entire complained-about interval is clean.

Ordinary rendering leaves pitch-preserving backend PCM intact after its final attack audit. A late supported attack requires another compatible source window when bounded backend rerenders cannot align it. Fractional-delay interpolation changes playback speed and pitch; a timing-rate bound of 0.85–1.15 does not make that effect musically acceptable. Deck and combined-waveform residual resampling are both disabled by default. Advanced callers may explicitly select `allow_pitch_changing_residual=True` for a requested effect, with immutable backend PCM, identical voice mapping and fresh audits. Its 35 ms delay/rate limits remain engineering bounds, not proof of pitch preservation.

Tests must include biased coarse grids, dense tonal backgrounds, variable tempo, missing attacks and a single late attack among many accurate events. Synthetic tests validate mechanisms only. Review the actual musical overlaps and compare before/after excerpts at matched loudness; disclose measurement coverage and remaining uncertainty rather than claiming 100% subjective or acoustic perfection.

## Leading attack versus bass body

A 100–1400 Hz novelty search can still follow a dominant delayed 100–180 Hz kick body despite its rising-edge threshold. On supported estimated drums, compare that family with an independently coherent 180–1200 Hz family. The current conservative selector needs at least eight matching source events, a median earlier upper front of 8–85 ms and a 10–90 percentile offset spread below 18 ms. Store the comparison and selected band; use the same selected band for source templates and final PCM matching. This is a conditional recognition method, not a universal kick-band rule. A front too weak to form a coherent family remains unresolved. Test with an earlier upper transient and a stronger delayed low body, and reject claims based solely on the low-body template.

## Full-spectrum percussion ownership

An actual energy or local-rhythm coverage failure uses the same bounded soft
routing search as a combined-front failure, after checking the legacy middle
curve. Preserve explicit bass positions/widths and whole-spectrum choices.
Missing drum estimates come from the exact final timed and foreground-routed
decks. Try existing gradual positions before the existing wider whole-spectrum
routes and their last hard-case option. Every candidate reruns front, energy,
vocal and local-coverage checks. If all fail, retain the refusal and choose a
compatible source window; never certify it using only the accurate later beats.

Every ordinary style validates actual bass, middle and routed percussion gain spans. Mood/Classic/Fuzzy only change ordering, never this smooth-mixing requirement. Factory full bass transfers span at least sixteen shared beats, thirty-two for Zhilin, within the available overlap. Validate the actual 10–90% bass and estimated-percussion spans independently: at least eight shared beats, sixteen for Zhilin, capped by the available half-window. Explicit reviewed widths remain reproducible with the separate reviewed continuity checks. Uncertain source evidence selects a gradual complementary percussion transfer, never an automatic sequential cut. A failed combined-wave check may retry gradual routing; unresolved contradictions require a different source window. Explicitly requested abrupt effects remain separate from ordinary mixing and its continuity claim.

With broad DJ Zhilin mids, a departing kick's upper click may retain nearly half gain after its bass gain reaches zero. When a coherent earlier upper front was selected or a previously missed bass attack was recovered in the relevant source window, the ordinary renderer, in every style, automatically supplies two aligned drum estimates from the actual final deck PCM to `blend`; the same API supports a reviewed collision. Harmonic beds follow the approved bass/mid/high envelopes, while both entire drum estimates follow the bass handoff envelope. The default gradual eight-bar Zhilin handoff can coexist with 32-bar melodic accompaniment; its effective 10–90% span is checked separately. Record final-deck hashes and fresh estimate provenance; reject mismatched shapes, incomplete pairs and unverified fade fallbacks. Foreground processing must precede estimation or be routed identically. Never reuse estimates from a different stretch map.

Check the actually blended interval against recognized leading attacks, not just pre-blend decks. Verify the outgoing estimated percussion gain is zero after its handoff and inspect separation leakage and rhythm continuity. This routing changes percussion ownership; it does not correct a wrong timing map or promise that every residual accompaniment transient is absent. Listener feedback on matched excerpts remains necessary for perceptual judgment.

A later waveform body can dominate full-template correlation even in a selected upper band. Establish attack identity using the complete template, then use an initial-front excerpt of that same reference as an additional view; do not replace identity with an unrelated nearby peak. The matcher accepts templates retaining the same 12 ms pre-context and a shorter post-context. Preserve contradictory measurements. If a strong recognized front still shows a small residual in the actual combined output, a bounded continuous delay can correct the combined waveform locally under the same monotonic-rate limits as deck refinement. Foreground and accompaniment must move together, outside PCM and CUE geometry remain fixed, and a new single master and full encoded check are required. Never claim this optional final correction executed merely because deck timing passed.

For all ordinary shared-phase overlaps, `blended_front_audit` checks both audible decks, including a side without selected upper fronts. It records audible source events, recognized matches and fresh percussion-view use. Ordinary verification rejects a recognized residual above 3 ms instead of resampling the blend. Only an explicit advanced pitch-changing selection permits the older bounded local refinement: a strong residual of at most 6 ms, one correction, context bounded by neighbouring fronts and endpoints, and less than 3% local rate change, followed by a fresh check. Missing coverage, conflicting contexts or an unresolved residual stop delivery. These checks do not establish that every percussion event passed or that prior sets were all audited.

## Bass coverage and final mixed waveform

Validate a newly recovered estimated bass candidate in the native prepared mixture before adding it to primary timing controls. The native region must have a unique reference, or the estimated percussion signature must be independently visible there (correlation at least 0.90, existing alternative-peak margin at least 0.02). A continuous/ambiguous native waveform is not evidence of an exact primary hit. If neither view can identify the candidate, keep its source position and uncertainty privately, leave it out of primary controls and certified recovered-beat requirements, and use gradual estimated percussion ownership with the approved harmonic curves in every ordinary style. This is source-only classification before the first stretch, never removal of a failed rendered observation. A recognized native reference displaced by more than 3 ms remains a contradiction and stops source selection. Every certified source event retains the existing output timing and mandatory bass gates; unknown source candidates are not claimed absent or verified.

Calibrate the supplementary bass-body view against the native input too, including mixture-only operation without stems. A periodic native reference that cannot distinguish its own timestamp cannot authorize backend or combined-waveform correction merely because an output correlation acquires a distinct peak. Preserve those proposed output offsets as unverified evidence. Recognized native-supported contradictions still stop the run. In the actual blend, require up to eight of the natively observable audible bass references; fewer available references remain explicitly partial coverage, and no observable reference means unknown, with a `null` maximum rather than zero. Overall primary and paired-attack coverage requirements remain unchanged. Every certified recovered hit must remain recognized in each natively observable reference view; an unobservable mixture-body view does not waive a required independently visible percussion signature.

On available source drum estimates, also search unsupported windows in 35–180 Hz using established same-source references (correlation at least 0.90, alternative-peak margin at least 0.05). Do not refit away a recovered hit merely because it differs from a dominant timing family. Retain separate primary and bass templates in actual rendered PCM. A correct upper front cannot overwrite a delayed bass observation. Recovered bass events must remain recognized after rendering and, when audible, in the actual blended waveform. Verify both audible decks, regardless of which selected the upper-front path. Missing recovered events or contradictory residuals stop delivery and require another compatible source window. Remaining unknown beats stay explicit coverage gaps; these checks do not certify every fill or offbeat figure.

Two correctly aligned kicks can mask each other's waveform near the bass crossover. Keep the raw mixed-wave measurement. For an unknown or weak match, also measure the actual blend minus the known other deck's contribution, reconstructed with the exact band gains and final drum routing. Require those two known contributions to sum to the actual blend within 2e-6 before using this view. It is source-aware verification, not guessed separation or independent recognition of the entire mixture. Preserve a strong contradictory raw match unless the actual known other contribution independently explains that peak: cosine at least 0.98 in the checked band and at least four times the own contribution's band power. This is source-identity ambiguity, not an aligned-hit observation. Retain its raw delay and explanation; require an eligible native-template match within the unchanged 3 ms limit in the actual own contribution. A fresh percussion view of that exact contribution may supply an otherwise unrecognized native signature, using the existing correlation and alternative-peak thresholds. Never override an eligible late own-contribution match with this exception or a reassuring separation. Retain every mandatory recovered-event and coverage check. Store raw, source-aware and fresh-view evidence separately. Any explicitly requested combined local timing correction must move both known contributions identically before rechecking.

After those source-identity checks, compare all recognized primary/bass errors from the two audible contributions on each coincident shared-clock beat. Two individually small errors of opposite sign must still satisfy the same 3 ms paired limit. Record no paired measurements as unknown, never zero error; retain the earlier minimum-coverage requirements.

Master verification compares differences of second-valued timestamps. Ignore only floating-point subtraction roundoff bounded by eight float64 machine epsilons at the timestamp's scale; do not round measured errors or increase the acoustic tolerance. A value represented as 0.0030000000000006963 seconds at a 3 ms boundary is numerical roundoff; a real detector-sample excess remains a failure. Nonfinite or negative error inputs are invalid.

Use that same bounded comparison for the two rendered decks, including their
individual and sparse-coverage clock limits. Retain the exact measured maximum
and roundoff bound in the pair report. A refused pair carries that report into
window recovery; a formatted `3.00 ms` string is not sufficient diagnostic
evidence and must not trigger repeated backend changes for arithmetic noise.

Pairwise timing uses the largest difference across both primary and bass observations on the same shared beat. Two good primary observations must not hide bass observations displaced in opposite directions. Duplicate target times are grouped only when building correction controls; the independent verification evidence remains separate.

## Coarse-label errors and short speed pulses

A coarse beat label can be wrong even when the native audio has a stable primary pulse. Establish a positive native period within 3% of the coarse period from at least eight references spanning seven beats. For sixteen or more observations, require a deterministic consensus covering at least 60% of observations and 75% of their temporal span, with residuals within 10 ms. Smaller families require agreement of every reference. A rescued primary attack must agree within the same 10 ms and still pass every rendered and combined-waveform check. Recognized off-phase accents remain native figures and cannot bend the primary clock. Record coarse and native-consensus residuals separately.

Before constructing an ordinary shared-SYNC map, calibrate automatic native windows and their sample cuts against the supported source clock. Bound corrections to 120 ms, preserve beat counts, the shared output clock, explicit boundary locks and protected material. Preserve the original labels and calibration evidence. Unsupported or variable pulses retain the existing source path; do not manufacture a constant source rhythm. An automatic section's duration estimates follow its actual calibrated boundaries. This changes analysis positions, never the native audio.

For an unrecognized interior control, infer absolute native time between recognized neighbours no more than four beats apart when a stable clock supports that inference or the coarse label error exceeds 35 ms. Extrapolate the established native clock across unobserved prefixes/suffixes instead of reintroducing noisy coarse intervals. These are inferred controls, never certified attack observations. Sample cuts already at the clip boundary cannot move during refinement. Advanced callers with untrimmed context may calibrate an interior window endpoint. Keep source figures at their native relative times.

The final rate audit rejects a correction-rate range above 0.05 over a complete bar (at least two seconds, extended to four beats at the slowest shared tempo), including composed residual curves. This is a rejection bound, not a target modulation. Divide actual rates by the immutable intended musical curve first, so deliberate smooth BPM changes are preserved. Advanced callers without a declared shared clock use an explicitly inferred slow trend from coarse native observations; they do not acquire an exact musical-clock claim. Measure both sides of interval boundaries: subdividing a bad map must not hide a pulse. The existing per-interval bound remains separate. Coarse initial maps defer this additional check only until native-source refinement, before the first stretch; final and cached clip audits always apply it. Backend compensation may try smaller steps after a rejected edit, always restoring the safe map and retaining the same bounds and fresh full-deck PCM verification. If none passes, select another source window. Retain failed source roles and absolute native positions for recovery. These engineering bounds do not guarantee perceptual quality or retrospectively certify existing exports.


## Native initial-front coverage and routed signal identity

A selected bass band is a clock hint, not permission to skip a coherent earlier
180–1200 Hz source family. Apply the same comparison in native-window calibration
and deck refinement, with and without drum estimates. Require at least eight
shared observations, 60% of the primary family and 75% of its index span before
using an earlier family as the primary clock. Store partial-family uncertainty;
do not quantize every native fill or cymbal onto quarter beats.

Keep a second native-mixture template with the same 12 ms pre-context and only
12 ms post-context, separately from the longer attack and bass-body templates.
Before stretching, require native uniqueness (correlation at least 0.90,
alternative margin at least 0.05, native delay within 3 ms). Fix up to eight
strong, distributed support references using native data alone. Those references
must remain recognized in the timed deck. Every recognized initial front,
including observations outside the support set, enters the unchanged 3 ms maximum.
Unknown output shapes stay explicit and cannot establish zero timing error.
A bass-focused fragment alone does not establish an independent leading click.

For an actual blend, short native templates require compatible signal identity.
Independent bass/middle/high routing or estimated-drum subtraction can change a
24 ms waveform. Do not call a correlation peak against that different waveform
a proven initial-front displacement. Keep that view as an explicit coverage gap;
retain its independent full-deck initial-front check and the existing primary,
bass, recovered-hit, paired and combined-waveform gates. A whole-spectrum scalar
handoff preserves the short reference identity and retains its support checks.
An audible upper reference cannot be declared absent just because the bass fader
is zero. Use the actual routed band and known own contribution; unknown nonquiet
signatures remain uncertainty.

Keep raw additive-interference evidence. When both exact known contributions
reconstruct the actual PCM, independently unique native fronts from both sources
can explain a shifted mixed correlation maximum. Require the original own front
and the other native front within 3 ms, correlation at least 0.98 and margin at
least 0.05. Actual neighbouring native references may bracket a missing other
attack; do not invent an attack there. An eligible late own match can never be
erased by this proof or by a fresh separation. These are identity checks, not
an increased timing tolerance.

## Optional whole-spectrum musical handoff

The advanced per-edge `whole_spectrum_handoff` setting supplies `start_beat`,
`end_beat` and `bias` on the existing shared beat phase. It uses complementary
whole-spectrum gains, without filtering or estimated-percussion subtraction.
For normalized progress p between the selected beats, incoming gain is
`1 - (1 - smooth(p)) ** bias`. Bias 3 and 4 preserve the existing reviewed
whole-window shapes; a bounded earlier handoff can reduce competing native
figures without changing pitch, clock or source cues. No source identity or
private cue position is packaged in the engine.

Do not combine this setting with separate band/percussion overrides. Verify the
actual 10–90% gain span: at least sixteen shared beats for Zhilin, eight for other
ordinary styles, bounded by half the available beats for a short overlap.
Report actual transfer time separately from the unchanged planning window;
a 32-bar shared clock does not mean 32 bars of simultaneous audible music.
The exact same gains feed vocal, continuity and combined-attack checks.

The existing ordinary blend remains first. After its smooth percussion transfer
and three same-width placement candidates fail, unlocked transitions may try
at most two full-window whole-spectrum shapes. User band settings remain locks.
No failed candidate authorizes a shorter window, abrupt cut, relaxed timing gate,
attack-fragment copying or pitch-changing residual resampling.

## Preserve working controls and preparation timelines

Explicit per-source R2 short/crisp6 and R3 selections remain available. Defaults
keep their original backend arguments; mixture and foreground use the same
chosen backend. A supported-native-clock retry is attempted only after a real
source timing failure, retaining actual observations, protected material and
all rate bounds. Successful old timing maps stay on their existing path.

Known phrase loops must rebind explicit kick evidence to the prepared edited
drum timeline, with identical mixture/voice/drum geometry. Shift feature profiles
by the same recorded edit exactly once. Rebinding to a new prepared artifact
must update its path without applying the feature shift twice. Original files
and records remain unchanged.

Regression coverage includes a single 8 ms front displacement with an unchanged
louder bass body, a late upper attack after bass gain reaches zero, native
family coverage, strict additive identity, transparent handoff continuity,
source-backend defaults, loop-timeline rebinding and the real resonant-kick
crossover. Synthetic success does not replace source-music renders or listening.

When a required initial front is missing or a recognized timing view is late,
window recovery rejects its physical native-source timestamp alongside existing
bass failures. Renumbering that event in a nearby window does not make it valid.
Recognized on-time fronts do not exclude otherwise compatible windows.


## Reviewed last-resort early transfer

After both existing full-window transparent curves fail, an unlocked edge can
try the reviewed early whole-spectrum curve. In a 128-beat window it starts at
beat 4 and ends at beat 36, bias 1. Keep this after the existing broad options and preserve successful
ordinary or broad variants. Other windows retain at most32 beats of handoff,
with lead-in bounded by available space; short windows retain their whole span.
All original attack, paired, energy, vocal, duration, pitch and continuity gates
remain mandatory. An explicit accepted edge can use it directly. No track IDs
or native source-cue positions are part of this portable recipe.

### Soft transfer after an uncertain incoming opening

After those existing routes fail, the engine may place the same 32-beat curve
later when only the incoming opening has insufficient local coverage and the
outgoing foundation already passes. The eligible contiguous opening figure is
9–32 unverified shared beats, starts within the first 16 and ends by beat 36;
another long unverified figure makes it ineligible. The original overlap needs
at least 64 beats, and the extra lead-in is bounded to 28. There are at most two
gain placements, the second four beats later, without a change to source CUE, clock, pitch, overlap or curve
width. Explicit band and whole-spectrum settings remain locks. Independent
native timing refusals cannot use this route. Complete actual front, local
coverage, energy, vocal and continuity checks remain mandatory; missing attacks
are not reclassified as verified. A missing complete quality callback retains
the original routes and refusal. Successful earlier variants return unchanged.


## Bounded source recovery without making short play the default

A failed source recognition or paired attack audit must not become an unexplained two-minute chapter. Preserve explicitly supplied track lists, user locks, source history and the successful processing routes. Keep defaults on sources that already pass.

1. For a rendered source attack failure, a supported per-source Rubber Band profile can be tried before abandoning the musical span: `rubberband_engine_override="r2"` with `rubberband_profile_override="short"` or `"crisp6"`, or `rubberband_engine_override="r3"` without an R2 profile. These are existing pitch-preserving backends. Do not override an explicitly user-locked backend. Record each attempted profile and require the same native rate, source identity, 3 ms individual/paired, blended-front and complete-export checks again. A backend change is not permission to copy native attack islands or enable pitch-changing residual interpolation.
2. In an authorized full rebuild, if an automatic 32-bar overlap forces a needlessly short chapter or repeatedly contains an unsupported attack, consider a 16-bar overlap while retaining the musical source span. The transfer must remain gradual and use the shared beat clock. An explicitly user-locked overlap length stays fixed. Recompute the affected native windows and automatic handoff settings; old absolute `handoff_windows`, `mid_handoff_bars` or `bass_handoff_beats` can exceed a shorter overlap. Archive them before replacing automatic values. Do not blindly halve the fade and turn a ten-second transfer into five seconds.
3. Keep exclusions scoped to the actual backend and native window/overlap geometry. A previous failed default profile does not invalidate the exact same section after a different profile has actually passed its full source audit. Retain the old diagnostic history and release only the exclusion justified by that new verification. Changing a profile must not erase independently established native timing or source integrity failures. A changed overlap still requires fresh source and real-blend verification; the old success is not a waiver.
4. If another source section is required, prefer a compatible 3–4 minute chapter and audit the whole set after every candidate. Where the supplied material permits it, the normal share remains at least 80%, preferably 90%; physically insufficient supplied sources retain the existing named constraint handling. A named short exception is allowed when reasonable compatible options fail; keep the one-minute floor and the existing safe loop search for performances below two minutes. It must not become the set-wide default. Preserve a verified short candidate as a checkpoint while testing a longer musical alternative.

For automatic selection from folders, a source that exhausts compatible windows and supported backends can be replaced by another unused source from the same folder. Preserve the requested folder proportions, duration target and normal chapter share. Recheck history immediately before selection; failed candidates do not count as completed use. This does not authorize omitting an explicitly supplied recording.

After bounded profile trials on the same span, available `unsupported_attack_events(evidence, role)` positions can rank the next structural CUE beyond the uncertain figure. Treat rendered recognition misses as context-dependent search hints, not permanent recording bans or invented native timing refusals. Moving only a few beats must not repeatedly put the same uncertain source figure back under a new relative beat number. Preserve independently failed native intervals, explicit locks and complete actual verification. A compatible shorter chapter is a valid exception within the set-wide normal-share budget; do not force an unverified intro merely to recover three minutes on that one track.

Physical-event exclusions belong to the recording hash, not a reused playlist position. Archive the old recording's exclusions when replacing it; do not transfer them to a different recording at the same position.

Supported native-clock recovery keeps the ordinary map first and the existing single-role retry next. If that retry exposes a separate measured rate failure at the other end of the source, it can cumulatively use that end's independently established native clock too. Every attempt starts from the unchanged coarse map, selects each eligible role at most once, and retains all actual attack and displaced-figure observations. Missing clocks, changed references or any remaining timing failure retain the original rejection. This does not relax rendered-attack, rate or blended-waveform checks.

Use unrounded acoustic evidence when diagnosing a threshold failure. A rounded error message is not a measurement report. Do not loosen the 3 ms boundary, hide an unverified required front, introduce rapid tempo excursions or publish a partially assembled set to make a candidate pass. Rebuild the complete affected performance and verify all adjacent transitions and the decoded final export.

Combined individual-front refusals retain the observed output timestamp, signed delay, template correlation, timing view and band. Paired refusals retain the worst participating views of both decks. Both describe measurements in the actual mixture; neither assigns a unique physical source failure or justifies banning a recording. Compare different profiles on the same source span first. If the same mixture refusal persists, review an existing structural source boundary rather than repeatedly applying the same failed candidate. Preserve the original evidence and require every actual gate again.

For an automatically ordered set with unlocked positions, a bounded last-resort review may exchange tracks with exactly the same declared key. Compare the complete ordering objective and prove that the key labels at every position remain identical; equal key alone does not preserve tempo, energy or mixability. Keep the selected recordings, requested folder proportions, playback targets and successful source routes. Rebuild and verify every affected adjacency and the full export. A supplied fixed order or user position lock forbids this exchange; source-integrity and native timing refusals remain in force. Record superseded failed pairs as superseded, never as passed.
