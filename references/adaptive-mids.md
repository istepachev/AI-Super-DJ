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

When the evidence supports separate mid parts, the same monotonic handoff position becomes an equal-power cross: square-root outgoing/incoming energy shares. Each source mid gain remains at or below unity; their amplitude sum can reach sqrt(2), while bass, highs, drum ownership, source clocks and pitch stay unchanged. One fixed decision per overlap avoids a pumping gain follower. Native changes in the available musical material remain possible; the rule does not invent missing mids or certify an inaudible join.

Shared or uncertain spectra, sustained simultaneous foreground estimates, or unavailable aligned voice estimates keep the previous complementary middle. The advanced per-edge `mid_energy_mode: "legacy"` retains it explicitly. Whole-spectrum collision variants retain their existing sound; this frequency-specific compensation does not silently filter them.

`adaptive_mid_energy` records the exact law and geometry. Combined-front components, vocal/energy checks and local audible-rhythm coverage replay these actual gains. Older frequency reports retain their recorded effective bass widths rather than acquiring a new factory width during an audit. A changed gain law invalidates cached overlap approval.

Factory bass/percussion transfer is also measured independently of the total overlap: at least eight effective 10–90% shared beats for ordinary SYNC and sixteen for DJ Zhilin, capped by the available half-window. The automatic Zhilin full transfer spans at least 32 beats where available; explicit reviewed band widths remain supported. Full shared-clock, recognized-attack, source-playback, master and complete-export checks still apply.
