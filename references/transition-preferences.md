# Transition preference policy

This is the requested listening/workflow preference, not an objective predictor
of quality. Retain subsequent judgments privately for each pair; preserve an
already approved exception ahead of a generic preset.

This ranking applies only when the ordinary beatmatched blend fails. It does
not replace successful ordinary transitions or all boundaries in an existing
reviewed montage. Default `auto` first tries ordinary mixing; long F is the
continuous fallback for a boundary that would otherwise be unresolved or cut.

| Priority | Technique | Use |
|---|---|---|
| 1 | F long: shared-phase SYNC | First fallback for unresolved ordinary blends; usually 16 or 32 bars |
| 2 | F short: same clock, shorter overlap | Only when requested or previously approved for the pair |
| 3 | E: partial-rhythm overlap | When a suitable rational pulse supports continuous musical connection |
| 4 | C: older common-tempo blend with body ramps | Prefer for an upward tempo change; do not apply as a generic downshift fallback |
| 5 | Original on/off cut | Disclosed last resort after suitable continuous methods fail |

Choose within these preferences without repeatedly asking for technique approval.
Do not try a shorter F automatically just because long material is unavailable:
the short version requires a request or an existing approval. Move to the next
eligible continuous technique. Respect locked cues, protected sections and source
integrity; inability to satisfy a mandatory constraint is not permission to
silently discard it. Record the actual technique and any fallback reason.

## Executable coverage

- Default `auto` preserves the original ordinary blend, normally 16 bars with
  its existing 8-bar reductions. It validates actual windows and rate caps before
  deciding whether fallback is needed. F is implemented in the CLI (`sync`).
  F conflicts use 32 bars for at most 20% temporary correction and 16 for larger
  supported conflicts. The threshold is a duration heuristic.
  Specific montage lengths and approved choices take precedence. Keep the
  existing 40% temporary-rate cap unless a different cap is explicitly supplied.
- `--transition-bars` controls ordinary lengths and forced F at close tempos;
  `--sync-transition-bars` controls conflict F. Explicit `sync` forces F globally.
  A plan edge's `transition_policy` permits `auto`, `sync` or `review`; `sync`
  retains an approved F even if ordinary mixing is feasible. Edge `overlap_bars`
  takes precedence in SYNC and permits 2/4/8/16/32. Supplying this SYNC-specific
  length in auto mode also explicitly selects F for that pair. Unknown or
  incompatible values are refused. Bar counts do not identify musical phrases.
- E uses `superdj.polyrhythm` and currently needs explicit source-window/render
  orchestration by the agent. Its rational pulse must fit the supplied grids.
  It is not isolated drum-stem editing or ordinary 1:1 SYNC.
- C corresponds to `shared_tempo` with declared tempo-ramp settings and caps.
  The label is a workflow reference, not a newly implemented effect.
- A cut requires a separate explicit render with `--allow-hard-cut`, retaining
  source boundaries. This workflow permits it as the last resort; never disguise
  it as successful F, E or C. Do not pass the flag on a successful SYNC run.

The CLI now combines ordinary mixing and fallback F per boundary in one run,
including approved F exceptions. There is no full automatic cascade through
E/C/cut. The agent can orchestrate those further fallbacks; do not advertise the
entire cascade as implemented.
Uncertain grids cannot establish beatmatching. Musical listening and dense-kick
alignment remain separate from mathematical clock and isolated-transient checks.
