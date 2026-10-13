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
