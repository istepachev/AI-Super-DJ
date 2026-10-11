# Experimental BPM/key ordering

Use `harmonize` when testing the published principles of DJ.Studio Harmonize.
This is our sequencer, not its code or an established equivalent. Keep the
normal multi-feature planner for ordinary sets unless this profile is requested.
Ordering and audio rendering are separate stages; the shared F/SYNC renderer
still validates timing, phrases, vocals, bass availability and source duration.

```text
superdj harmonize --records ../work/records.json --work ../experiment --seed 42
superdj harmonize-compare --records ../experiment/analysis_snapshot.json --our-order ../experiment/harmonize_plan.json --other-order ../dj-export.m3u8 --work ../comparison
```

The first command writes an input M3U8, an ordered M3U8, `order.json`, a plan
with every edge cost, a fixed analysis snapshot, a text track list and an HTML
report. Music is referenced, not copied. A normal set can use that order through
`plan --mode fixed --fixed-order ../experiment/order.json`, followed by `render`.
The ordering report does not certify usable beat grids or audible transitions.

## Explicit controls and objective

- `--priority`: 0 is BPM, 1 is key, default .5, within the final tie-breaking
  score after compatibility tiers. These weights are ours; the
  position of DJ.Studio's slider is not known to have an identical mapping.
- `--direction`: ascending, descending or none. Direction is a soft preference.
- `--bpm-tolerance`: optional positive absolute BPM limit with a soft penalty
  above it. Violations remain in the report. It is not a per-deck stretch cap.
- `--first`, `--last`: mandatory endpoints. `--locked-chains` reads disjoint
  adjacent ID chains; the search moves each chain as one block.
- `--seed`: tie resolution in exact/heuristic searches. Exact Held–Karp is used
  up to 17 blocks. For 18–100 blocks, a directed whole-path MILP searches all
  tracks together, with a verified bounded-search fallback. Its time budget is
  twenty seconds by default. Larger sets retain the bounded search. Only a
  proven optimum is labelled optimal; a time-limited incumbent is not. Equal
  optima can produce different orders between solver versions and platforms.

The default ordering objective is lexicographic: first minimize unaccepted or
unknown key edges, then accepted review/high-risk edges, then the weighted
BPM/key score below. Small tempo convenience cannot buy an incompatible key
edge. A rare-key track is evaluated as part of the complete route, rather than
left as an orphan by repeatedly choosing the cheapest immediate edge. Every
input and locked chain is retained. Incomplete compatible graphs may still
require disclosed unaccepted links; no track is silently discarded.

For reproduction of earlier experiments only, `--order-objective weighted`
restores the version-two scalar objective and beam search. The advanced
`--search-seconds` control changes the MILP budget (positive, at most 120).
`total_cost` remains the old weighted score for comparisons; `search_cost` and
`objective_details` describe the actual objective. A lower weighted score is
not always preferred under the new hierarchy.

For priority `p`, edge cost is:

```text
2*p*harmonic_cost
+ 2*(1-p)*(abs(delta_BPM)/10 + .5*direction_reversal_BPM/10)
+ max(0, abs(delta_BPM)-tolerance)/tolerance  [if tolerance supplied]
```

Energy, vocal activity, embeddings and analysis confidence do not enter this
objective. Uncertain BPM/key analysis is still disclosed. The existing general
planner keeps its original multi-feature objective and confidence handling.
No automatic transposition, octave correction, track removal or insertion.
The direction term is not a monotonic BPM curve: with fixed endpoints its
sum is determined by total BPM travel and the endpoint difference. Do not
describe this setting as enforcing growth at every step.

## Declared relation table

Camelot numbers wrap modulo 12. These are explicit engineering penalties,
not learned probabilities or published DJ.Studio weights.

| Relationship | Cost | Fuzzy profile | Mood profile | Risk flag |
|---|---:|---|---|---|
| Same number and letter | 0 | Allowed | Allowed | Lower |
| Same number, A↔B | .10 | Allowed | Allowed | Lower |
| Same letter, +1 / −1 | .15 | Allowed | Allowed | Lower |
| Number +1 / −1 with letter change | .30 | Allowed | Allowed | Review |
| Same letter, +2 (whole tone up) | .45 | Incompatible | Allowed | Review |
| Parallel tonic: A→B +3, B→A −3 | .55 | Incompatible | Allowed | Review |
| Same letter, +7 (semitone up) | .90 | Incompatible | Allowed | High |
| Other relationship | 1.50 | Retained with penalty | Retained with penalty | Review |
| Unknown key | 1.00 | Unknown | Unknown | Review |

Allowed means permitted by the selected profile; it does not mean safe sounding
audio. Lower-risk and adventurous matches are counted separately, especially
for the dissonant semitone lift. Official Fuzzy/Mood pages contradict one
another, so these names identify our experimental tables rather than a claimed
exact mapping to a proprietary mode.

## Compare the same tracks

1. Import `input_tracks.m3u8` into DJ.Studio and harmonize all tracks together.
   Capture the app version, mode, slider/tolerance, direction and locks. Keep
   different folder-block constraints out of the experiment unless both sides
   have identical blocks.
2. Export M3U/M3U8, CUE or DJS; JSON IDs/paths/track records are also accepted.
   DJS is read as metadata only. CUE without source paths needs unique titles.
   Unknown or ambiguous entries and duplicates are errors, never fuzzy guesses.
3. Compare both orders using the same reference analysis. Report membership,
   directed neighbors, positions, lower-risk/adventurous relations, tempo jumps,
   direction reversals and both costs under our objective.
4. Inspect exported BPM/key differences separately. With complete explicit
   exported musical metadata, replan on those features and compare again.
   DJS native key indices 0–23 now use a verified chromatic-enum mapping,
   corroborated independently in DJ.Studio 4.2.15 and exported OpenKey tags.
   Retain raw indices and mapping provenance. Use `--djs-key-source` to explicitly
   select DJ.Studio, Mixed In Key or original tags; unavailable alternate keys
   fall back to the native DJ.Studio index and disclose that fallback.
   `useMixedInKeyAnalysis=true` does not establish the selected key source.
   Numeric indices in ordinary JSON remain uninterpreted. Missing exported keys
   are never filled from our estimates while
   claiming to use DJ.Studio analysis.

## Learn from actual successors

`harmonize-compare` also reports the demonstrated next track's rank among only
the remaining, unplayed tracks after each actual prefix. Equal candidates have
a rank range, a tie count and a score gap; late steps with fewer than five
candidates are summarized separately. These are local pair ranks. To answer
which track the whole-path engine would actually choose, replan current plus
remaining tracks with the current ID fixed first, then inspect `order[1]`.

For differences, a second run locking `[current, actual_next]` compares valid
remaining routes. Compare full-route `objective_tuple` (unaccepted count,
adventurous count, weighted cost) for the default strategy, or `total_cost` for
the legacy weighted strategy. A better valid full-route objective reveals a
missed search solution; it does not prove that the demonstrated pair was
predicted. Never subtract `search_cost` between different block partitions:
collapsing a chain changes n, scaling and constant internal edges. Keep an
existing valid route as an incumbent when interpreting bounded searches.

Use `harmonize-learn --examples ../examples.json --work ../learning` to fit
explicit controls and validate on separate playlists. The private manifest is
a list of `{name, records, order}`; records and order may be JSON paths relative
to the manifest. Controls are selected on the other playlists, never on the
held-out playlist. The criterion uses the reciprocal midpoint rank to penalize
broad ties. IDs and titles are not fitted features. Learned controls are an
experimental local ranking profile and do not silently replace set defaults.

For a separately trained conditional model use
`harmonize-learn --learner ranker --examples ../examples.json --work ../learning`.
Three playlists are required for nested validation: the outer held-out playlist
never enters fitting or inner parameter selection. Features contain musical
key relationships and BPM differences, not identifiers, titles, paths or the
held-out order. Profiles and validation results stay private.

`harmonize-rank --records ../records.json --profile ../learning/successor_ranker_profile.json --current TRACK_ID --played-prefix ../played.json --work ../ranks`
reports only remaining local choices. The played prefix ends with the current
track. A profile cannot disable the independent harmony gate. Utility is not
confidence, sound quality or a whole-path recommendation. Raw diagnostic
agreement and gated recommendations have separate results with the same
denominator; blocked demonstrations remain in it.
Do not activate a profile merely because it copies more demonstrated choices:
the current examples include many links outside our Fuzzy table, and the learned
model can overprefer semitone lifts. It remains an explicit experiment.

Our relation-table allowance is not DJ.Studio's Solve status. If an imported
test requires frequent Solve, record that as a failed application check and
inspect detector differences before promoting the order. Do not infer a
probability of a successful audible mix from edge cost or agreement alone.
The saved DJS `mix.incompatible` field is not a count of Solve warnings. A
backup retaining the original imported order is useful analysis evidence, not
a demonstrated Harmonize route to learn from. When importing complete native
analysis, retain its provenance and do not describe the resulting correction
as improved independent audio detection.

Different good orders are expected. One match or mismatch does not prove
algorithm equivalence or error; finite beam search can also miss a lower-cost
path. Do not tune rules on one playlist and call that general validation.

Official sources checked 2026-10-11:
[Harmonize](https://help.dj.studio/en/articles/7878402-harmonize-previously-automix),
[harmonic relationships](https://help.dj.studio/en/articles/8118455-harmonic-matches-explained),
[Fuzzy explanation](https://dj.studio/blog/fuzzy-keymixing),
[M3U import](https://help.dj.studio/en/articles/9561495-import-a-playlist-file-m3u-or-m3u8-into-dj-studio).
