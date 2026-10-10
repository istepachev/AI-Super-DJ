# Transition preference policy

This is the requested listening/workflow preference, not an objective predictor
of quality. Retain subsequent judgments privately for each pair; preserve an
already approved exception ahead of a generic preset.

| Priority | Technique | Use |
|---|---|---|
| 1 | F long: shared-phase SYNC | Default for suitable source windows; usually 16 or 32 bars |
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

- F is implemented in the CLI (`sync`). Defaults use 32 bars for ordinary tempo
  differences, 32 for conflicts requiring at most 20% temporary correction and
  16 for larger supported conflicts. The threshold is a duration heuristic.
  Specific montage lengths and approved choices take precedence. Keep the
  existing 40% temporary-rate cap unless a different cap is explicitly supplied.
- `--transition-bars` controls ordinary F lengths; `--sync-transition-bars`
  explicitly controls conflicts. A plan edge's `overlap_bars` takes precedence
  for that pair in SYNC and permits 2/4/8/16/32. Unknown or incompatible values
  are refused. These lengths count bars; they do not identify phrases.
- E uses `superdj.polyrhythm` and currently needs explicit source-window/render
  orchestration by the agent. Its rational pulse must fit the supplied grids.
  It is not isolated drum-stem editing or ordinary 1:1 SYNC.
- C corresponds to `shared_tempo` with declared tempo-ramp settings and caps.
  The label is a workflow reference, not a newly implemented effect.
- A cut requires a separate explicit render with `--allow-hard-cut`, retaining
  source boundaries. This workflow permits it as the last resort; never disguise
  it as successful F, E or C. Do not pass the flag on a successful SYNC run.

The current renderer chooses one global technique per run; there is no automatic
CLI cascade combining F/E/C/cut across a full set. The agent can orchestrate
individual comparisons/fallbacks. Do not advertise that cascade as implemented.
Uncertain grids cannot establish beatmatching. Musical listening and dense-kick
alignment remain separate from mathematical clock and isolated-transient checks.
