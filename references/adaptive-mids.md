# Adaptive middle handoff

Reviewed listening feedback supersedes the initial experiment's
rule that uncertain chroma should usually shorten mids to sixteen bars.
Different local notes can still support each other. Midband percussion can
maintain a dance rhythm while one source has an empty kick section. Preserve
that contribution rather than treating all middle energy as competing melody.

## Assess the actual overlap

Use the synchronized source excerpts, their intended band gains and aligned
voice estimates, not whole-track keys alone. A local harmonic/percussive
separation and pitch-class profile can screen sustained tonal clashes; neither
is an isolated melody or drum stem. Inspect sustained multi-bar discordance,
vocal foreground competition, note density and actual simultaneous energy.
Matching global key does not certify the local chords; unrelated-looking
chroma does not establish an unpleasant mix.

Evaluate rhythmic support separately: whether outgoing mid/high percussion
fills a sparse incoming section, whether shortening removes a useful groove,
and whether the sources cooperate rather than duplicate conflicting attacks.
Compare the actual long/short candidates at matched levels. Record audible
energy holes and voice competition separately from tonal evidence. RMS matching
alone is not exact perceptual loudness matching.

## Choose and record

- Aim for the full thirty-two-bar handoff where harmonic and rhythmic support
  coexist; retain approved sixteen-bar source windows rather than moving locks.
- Use a smooth sixteen-bar middle handoff when foreground voice/melody
  competition is evidenced; eight bars can resolve stronger persistent conflict.
- If both options sound good, prefer the longer useful musical texture.
- Preserve a long middle where the shorter option leaves a bare kick or loses
  the dance rhythm, even if pitch-class similarity is mediocre.
- If low-frequency attacks appear to compete, inspect bass/drums independently.
  A shorter bass handoff is a separate choice and need not shorten the middle.
  Dense ambiguous attacks do not justify speculative timing offsets.
- User-reviewed choices override scores. Record inferred approvals separately
  from direct comparisons. Keep uncertain algorithmic recommendations labelled.

The DJ Zhilin mixer supports per-edge `mid_handoff_bars` and
`bass_handoff_beats`. A middle override changes only middle envelopes;
a bass override changes only bass envelopes. The whole overlap, shared clock,
high-frequency curves and source chapter starts stay intact. The full-length
middle preserves the approved private reference curve; a shorter middle uses
a complementary smooth cross centered near the bass handoff. Bass defaults to
the slow reference cross; exceptions must be recorded.

## Reviewed examples and validation

The optional private runtime's `adaptive_mid_feedback` points to local
source-cohort feedback. Do not copy another listener's titles, transition
indices or per-edge overrides to a new collection. In reviewed comparisons,
some competing foregrounds benefited from sixteen/eight-bar middles, while
other overlaps retained long middles for rhythmic support. Equally good
candidates kept the longer texture. A disclosed one-bar bass cross mitigated
one uncertain lowband collision; that listening decision did not establish a
certified timing repair. Such exceptions apply to their reviewed source pair,
not every transition in this style.

Keep the original predictions frozen. Save later feedback and reviewed choices
separately. Those comparisons calibrate our policy on this particular set;
they are not held-out validation. On a new set, record decisions before user
audition and check whether the new policy preserves groove while resolving
voices/tones. Do not claim a trained reliable classifier from nineteen examples.

## Preserve energy when the parts do not compete

An amplitude-complementary 50/50 middle has half the energy of two unrelated equal-level parts before fading (about −3 dB). Ordinary frequency routing now checks the actual synchronized accompaniment beds, subtracting the final aligned drum estimates for this assessment where available. Stereo channel powers are retained separately; opposite channel polarity cannot masquerade as silence. Half-second log-frequency occupancy and aligned foreground estimates screen sustained shared content. These are conservative signal proxies, not a certified melody-compatibility classifier.

Prefer an equal-power cross at the same monotonic handoff position: square-root outgoing/incoming energy shares. Shared broad spectral occupancy alone must not veto it: independent music commonly occupies the same frequencies. Estimate actual stereo cross-power on each active half-second. Choose one fixed compensation strength for the entire overlap, as close to full independent power as the local positive-covariance ceiling permits (at most +1 dB above handoff-weighted native powers in every measured active window). A partially coherent pair receives partial support rather than an all-or-nothing spectral refusal. This is an engineering energy bound, not a musical compatibility score.

Retain the previous complementary middle for sustained competing foreground estimates at the newly proposed audibility, coherent shared waveforms, sustained shared narrow tones, insufficient active evidence or unavailable aligned voice estimates. Nearby narrow tones can beat even with little average waveform correlation. Negative correlation never authorizes gains beyond the independent-power law. Every source mid gain remains at or below unity and the pair amplitude sum at or below sqrt(2); bass, highs, drum ownership, source clocks and pitch stay unchanged. The fixed strength avoids a pumping gain follower. Native musical dynamics remain; the rule cannot fill an absent part or certify an inaudible join. `mid_energy_mode: "legacy"` explicitly retains the former sound, and whole-spectrum collision variants remain unchanged.

`adaptive_mid_energy` records the exact law, fixed strength, local energy ceiling and geometry. The v2 reader replays saved v1 gain reports exactly. Combined-front components, vocal/energy checks and local audible-rhythm coverage replay these actual gains. Every newly supported candidate must pass the unchanged actual overlap checks; if it fails, retain a passing legacy route before changing other musical decisions. Older frequency reports retain their recorded effective bass widths rather than acquiring a new factory width during an audit. A changed gain law invalidates cached overlap approval.

Factory bass/percussion transfer is also measured independently of the total overlap: at least eight effective 10–90% shared beats for ordinary SYNC and sixteen for DJ Zhilin, capped by the available half-window. The automatic Zhilin full transfer spans at least 32 beats where available; explicit reviewed band widths remain supported. Full shared-clock, recognized-attack, source-playback, master and complete-export checks still apply.

## Preserve dominant accompaniment regions above the power baseline

In automatic routing, newly supported mid energy also requests exact final-deck drum estimates when that runtime is available. Transfer the estimated whole percussion spectrum together, rather than leaving the outgoing kick's upper attack at a long mid gain after its bass has changed hands. This is a gain-only route on the existing shared clock. Missing foreground/percussion estimates retain the established baseline; no guessed drum subtraction is allowed.

With those exact estimates, split accompaniment into eight overlapping, complementary zero-phase cells from 180 to 3500 Hz. Their sum exactly reconstructs the existing mid band. Gradually retain a dominant source region above the independent-power baseline where the other source contributes little: support starts above 3 dB separation and reaches its maximum at 12 dB separation. Estimate those relationships over 16 shared beats, retaining the real musical dynamics. Every cell gain stays at or below native unity. Conservative monotone projections prevent reopening an outgoing region or withdrawing a previously opened incoming one; they only reduce proposed support. Shared regions keep the established power handoff. These coarse dominance estimates do not certify melody compatibility.

`selective_mid_energy` records cell cutoffs, phase knots and strengths. Actual PCM and independent known-source components replay the same cell gains. Voice and audible-rhythm checks conservatively use the largest actual cell gain of each source. Estimated percussion retains its own whole-spectrum gain. A more complete accompaniment can raise combined mid energy above the one-source independent-power reference; keep headroom with the declared constant master gain and inspect the musical result.

The explicit `mid_energy_mode: "power"` retains the previous v2 broad independent-power model. On a selective-route refusal, check that retained power route before the original `legacy` route; every alternative still passes the unchanged combined-front and actual quality checks. Whole-spectrum variants and saved v1/v2 reports remain reproducible.
