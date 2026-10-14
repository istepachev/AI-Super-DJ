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

## Portable ordering presets

For a requested DJ.Studio-style **complete-order comparison**, use
`--preset djstudio-route-v2`. It packages the frozen density/remaining-pool
profile and a frozen Paired safety profile, both using the public route search
and repair. For **local next-track
suggestions**, use `harmonize-rank --preset djstudio-next-v1`; it packages the
frozen priority profile retained from the earlier literal Top3 comparison. Both use schema-three
priority-aware inference. Packaged coefficients and feature layout are exact
copies of the respective frozen models; publication removes private names,
paths, records and orders, retaining anonymous training conditions and metadata
SHA commitments. Parameters were frozen before the latest exports; the
sanitized public files were prepared for release. `djstudio-route-v1` retains
its earlier single-profile policy and coefficients for reproduction.

Route-v2 generates two complete candidates from the same musical inputs,
mode, priority, nullable tolerance, seed and explicit first/last/chain locks.
It selects the safety route only when it has **strictly fewer outside-mode
links across the complete route**. Equal counts retain the primary route.
Reference order agreement and utilities from different fitted models are never
selection criteria. `preset_route_selection` retains both candidate orders,
their outside-mode counts and the choice. If both need a compromise, the
selected result continues to disclose it; this safeguard cannot guarantee a
compatible path through every pool and mandatory lock.

The two-route safeguard was declared after diagnostics of 24 frozen model
routes, before the complete 32-route audit. It was evaluated after that audit.
Its separately labelled evaluation is an engineering check informed by the
model-route diagnostics, not
a pre-frozen or source-independent validation of a newly chosen policy. The
original frozen comparisons remain intact. See the [final daily
benchmark](benchmark-20261013-final.md) and the [earlier
benchmark](benchmark-20261013.md).

The [sixth-export Fuzzy/BPM6 supplement](benchmark-20261013-six.md) checks the
already published 0.17.0 policy on another order of the same 431 sources.
The coefficients and generated routes are unchanged. A repeated complete-prefix
context contains two different DJ choices, one outside our native-key table;
random selection only within our allowed shortlist cannot explain both.
Keep this native-table discrepancy separate from ranking, actual key changes,
audio quality and the still unknown DJ.Studio support or random distribution.

The [three-Key-repeat supplement](benchmark-20261014-key-repeats.md) preserves the published policy and the old-six 63.19% literal Top3. The expanded nine-reference estimate is 61.70%; about 94.8% of its decline is due to condition weights. Equal-control macro changes from 61.86% to 61.74%. It also compares unshifted positions and full-route windows. For requested accuracy-preserving model updates, compare baseline and candidate on the same frozen references, controls and denominators; the report records current baselines. This retrospective regression gate cannot guarantee a nondecreasing estimate under arbitrary future test mixtures. No new model is promoted.

The [eight-order batch4 report](benchmark-20261014-batch4.md) evaluates a distinct120-occurrence/119-source pool with unchanged0.17.0 profiles: Next72.41% on new orders and63.78% over17 references. Actual-width eligible random chance rises to28.75%; absolute excess falls to43.66pp, so raw coverage growth is not an algorithm update.32 new full routes use the inherited budgets. Two unknown-key occurrences are preserved; count their links separately from known incompatible keys. Literal occurrence comparisons remain primary; source-equivalence sensitivity uses separately labelled views and denominators.

The profiles are experimental. Local Top3, whole-order adjacent matches and
contiguous three/four-track chains show different tradeoffs. The completed
comparisons do not establish a universal winner. The evaluated order variants share sources
with training examples, so new export order is not independent source
validation. Utilities are relative scores, not probabilities, confidence or
audible quality. Keep full-route and local results separate, preserve blocked
teachers in comparison denominators and keep ties and short tails explicit.

The comparison protocol used seed **0**, neutral direction and free endpoints
because the user declared no locks; do not infer endpoints from reference
orders. Reuse another user's explicit controls only for the same declared
experiment. Examples of that protocol:

```text
superdj harmonize --records ../work/records.json --preset djstudio-route-v2 --harmonic-mode mood --priority 0 --routing-priority bpm --bpm-tolerance 4 --direction none --seed 0 --work ../mood-order
superdj harmonize --records ../work/records.json --preset djstudio-route-v2 --harmonic-mode fuzzy --priority 0 --routing-priority bpm --bpm-tolerance 6 --direction none --seed 0 --work ../fuzzy-order
superdj harmonize --records ../work/records.json --preset djstudio-route-v2 --harmonic-mode mood --priority 1 --routing-priority key --direction none --seed 0 --work ../key-order
superdj harmonize-rank --records ../work/records.json --preset djstudio-next-v1 --current TRACK_ID --played-prefix ../work/played.json --harmonic-mode mood --priority 0 --routing-priority bpm --bpm-tolerance 4 --work ../next-ranks
```

For Key priority, omit `--bpm-tolerance` to represent known disabled/null
tolerance; Fuzzy Key uses `--harmonic-mode fuzzy` with the same Key controls.
Mood BPM 4 and Fuzzy BPM 6 are user-confirmed conditions for this comparison,
not universal mode defaults. `--routing-priority` communicates the categorical
control; numeric `--priority` sets our route weighting (0 BPM, 1 Key). Intermediate
numeric values do not identify a known fitted categorical control. Tempo
tolerance is a soft preference; it does not forbid a larger BPM jump. The
mode's independent harmonic gate still applies.

`played.json` is the complete actual played-ID array ending at `TRACK_ID`;
omitting it means only that current track is excluded. `harmonize-rank` writes
the literal capped shortlist and candidate diagnostics without optimizing the
complete route. `ordering_preset` in the saved result records the selection.
Explicit presets sort input occurrences by stable ID before seeded route
selection, preserving explicit endpoint/chain constraints. This prevents a
saved teacher file order from becoming the seed's implicit tie permutation;
`preset_input_policy` records that choice. Ordinary planning is unchanged.
`--preset` is mutually exclusive with a private `--ranking-profile` for
`harmonize` and with `--profile` for `harmonize-rank`.

These ordering presets are separate from ordinary `make` development by tempo,
intensity and bass, and from the renderer's `default`/`zhilin` mixing style.
Do not switch the ordinary planner merely because a set uses DJ Zhilin. To
render a requested preset order, feed it to fixed-order planning, then apply
the chosen style and normal source/timing/audio checks. No audio quality is
certified by an ordering match. New learning, examples, private checkpoints
and reports remain outside the public repository; only explicitly sanitized
published parameters are bundled.

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

The default ordering objective first minimizes unaccepted or unknown key
edges. Fuzzy then minimizes accepted review/high-risk edges before the weighted
BPM/key score below. Mood uses the weighted score immediately after allowance:
its Energy, Change and Jaws relations are expressive choices, not a categorical
reason to avoid the route. Small tempo convenience cannot buy an incompatible key
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
| Traditional diagonal: B→A +1, A→B −1 | .30 | Allowed | Allowed | Review |
| Opposite diagonal: B→A −1, A→B +1 | .30 Fuzzy / 1.50 Mood | Allowed | Incompatible | Review |
| Same letter, +2 (whole tone up) | .45 | Incompatible | Allowed | Review |
| Parallel tonic: A→B +3, B→A −3 | .55 | Incompatible | Allowed | Review |
| Same letter, +7 (semitone up) | .90 | Incompatible | Allowed | High |
| Other relationship | 1.50 | Retained with penalty | Retained with penalty | Review |
| Unknown key | 1.00 | Unknown | Unknown | Review |

Allowed means permitted by the selected profile; it does not mean safe sounding
audio. Fuzzy ignores the letter for adjacent numbers; Mood uses the traditional
directional diagonal and three extra expressive moves. Each key has six Fuzzy
destinations and eight Mood destinations, with five shared. Mood is not a
superset of Fuzzy. This table follows the explicit Fuzzy examples, the user's
Mood wheel and [Mixed In Key's directional diagonal](https://mixedinkey.com/book/use-advanced-harmonic-mixing-techniques/);
generic help prose differs. These are source-based tables with our penalties,
not the proprietary search weights. Count adventurous matches separately,
especially the dissonant semitone lift.

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
adventurous count for Fuzzy or zero for Mood, weighted cost) for the default strategy, or `total_cost` for
the legacy weighted strategy. A better valid full-route objective reveals a
missed search solution; it does not prove that the demonstrated pair was
predicted. Never subtract `search_cost` between different block partitions:
collapsing a chain changes n, scaling and constant internal edges. Keep an
existing valid route as an incumbent when interpreting bounded searches.

Use `harmonize-learn --examples ../examples.json --work ../learning` to fit
explicit controls and validate on separate source cohorts. The private manifest is
a list of `{name, records, order, cohort_group, mode}`; records and order may be JSON paths relative
to the manifest. Give every ordering of the same source pool the same cohort,
including both modes. Controls are selected on other cohorts, never on a variant
of the held-out pool. Older controls/ranker manifests fall back to example names
and disclose that grouping limitation. The criterion uses the reciprocal midpoint rank to penalize
broad ties. IDs and titles are not fitted features. Learned controls are an
experimental local ranking profile and do not silently replace set defaults.

For a separately trained conditional model use
`harmonize-learn --learner ranker --examples ../examples.json --work ../learning`.
Three source cohorts are required for nested validation: every outer held-out
variant stays outside fitting and inner parameter selection. Features contain musical
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

## Learn Fuzzy/Mood with the remaining pool

`harmonize-learn --learner modes --examples ../examples.json --work ../learning`
fits a portable JSON model and writes `mode_ranker_validation.json` and
`mode_ranker_profile.json`. Each demonstration must explicitly label `mode`.
Connected shared source hashes/IDs and declared cohort labels keep all repeated
variants together. At least three independent source cohorts are required for
nested validation. No track identifier, title, path or demonstrated position is
a musical feature; identifiers are grouping and matching evidence only.

The model considers directed tonal relations, tonic/scale overlap, BPM movement
and the remaining pool's key connectivity, rare-key exits and viable successors.
The optional `tempo_pool` family additionally lets the cost of absolute/signed
BPM movement depend on the tonal relation, retaining the actual twelve pool
context terms. It has 97 base columns; old 75-column `pool` profiles still load.
Inner training cohorts select feature family, regularization, mode sharing and
training support using gated literal Top3, then unique Top1, reciprocal rank and
supported likelihood. A weaker .01 regularization candidate is included.
`all_candidates` retains the legacy training softmax; `allowed_candidates`
fits only independently allowed candidates and omits unsupported target queries
from fitting. Both options keep every demonstration, including blocked targets,
in validation denominators. Profiles declare support and skipped fitting counts;
legacy profiles without the new field use the all-candidate convention. The
selected profile may retain the simpler family/support: new options are not
automatically active on every input. Repeated orders do not gain extra fitting weight simply by
being saved more often. Unknown keys remain explicit and in the validation
denominator. Profile versions reject stale feature layouts.

`harmonize-rank` accepts the new profile and compares only unplayed candidates.
`harmonize --records ../records.json --ranking-profile ../learning/mode_ranker_profile.json --harmonic-mode mood --work ../order`
assembles a complete sequence by recomputing pool context at every choice.
Endpoint and adjacency locks are preserved. The independent gate rejects
out-of-mode recommendations. If no playable head is allowed, the complete-order
assembler retains every track with an explicitly flagged BPM/key fallback;
it never reports that fallback as a successful learned choice. This sequential
policy protects its last twelve unplayed blocks with exact unrestricted suffix
compatibility bounds. It first minimizes the future unaccepted block boundaries
among currently allowed heads, then preserves the learned utility preference
and seeded ties. Locked chains use their real incoming head and outgoing tail;
a reserved last block stays last. Fixed incompatible edges inside user locks
remain disclosed obligations. The exact result covers that suffix's key
allowance only, not the full route or audio quality. The ranker's displayed
Top3 is not silently replaced by this assembly guard.

A separate bounded directed repair pass then reopens provisional adjacencies
when it can strictly reduce unaccepted final boundaries. It preserves endpoints
and entire user-locked chains. Among improving moves it prefers fewer changed
incumbent adjacencies, then lower declared static BPM/key cost. Transfers and
swaps move blocks without reversing their internal track order. Compatible
routes stay unchanged. A second stage can relocate intact segments of up to
thirty-two blocks when short moves leave bridges. Both stages share finite
scan/evaluation budgets and accept only fewer unaccepted boundaries; they do
not permanently lock provisional same-key groups. The final route and its bridges are reported separately
from the original sequential choices, their actual remaining pools and suffix
bounds. Bounded repair is not a proof that all remaining bridges are necessary
or that a globally best complete route was found.

This policy is separate from the global pair-cost route optimizer, and its local
validation is not whole-playlist agreement or an audio-quality measurement.
With a legacy version-two ranking profile, learned utility determines allowed next choices:
`--priority` and `--bpm-tolerance` affect the diagnostic cost, an explicit
fallback and the static repair tie cost. `--direction` additionally affects an automatic first
choice. These controls do not retune the fitted utilities or create a hard
tempo constraint. Use the global route search when directly adjusting its
declared BPM/key weights; record demonstration settings when learning a profile.
Version-three priority-aware profiles additionally receive the explicitly
declared routing priority and tempo tolerance during ranking and dynamic search.

## Priority-aware fixed training

Use the experimental `harmonize-learn --learner priority` when demonstrations
explicitly distinguish Key and BPM priority. Store `mode`, `priority` (`key`,
`bpm` or `unknown`) and nullable `bpm_tolerance` independently in each example.
Human instructions override a filename such as `key`; never infer a missing
tolerance or priority from the observed tempo jumps. The opt-in manifest field
`priority_aware_training_eligible` permits a controlled example without enabling
it for the legacy, priority-blind learner.

This fixed fit uses the declared feature family, mode interactions, L2 penalty
and training support without choosing parameters on the same examples. It can
fit one connected source cohort, and records that scope as training-only.
Repeated orders, modes and priorities of the same source pool remain one cohort.
The resulting training metrics are not independent validation or a promise of
the same agreement on the next file. Freeze a profile before that next test.
Legacy nested validation and its source-cohort requirements remain separate.

The new profile retains BPM/key and the actual remaining-pool features. It adds
Key/BPM interactions with the relation and tempo terms, plus known-tolerance
descriptors. Candidate features are computed using the entire remaining pool
before unsupported targets or candidates are excluded from fitting. All
demonstrated targets, including blocked ones, remain in evaluation denominators.
Track names, identifiers and the demonstrated future order are not features.

For a version-three profile, numeric CLI priority endpoints `0` and `1` supply
the BPM and Key contexts. Intermediate/default values provide `unknown` unless
the advanced `--routing-priority` explicitly selects a context. A missing
tolerance remains unknown; it is never silently filled with four or six BPM.
The Python ranker accepts `priority="key"`, `"bpm"` or `"unknown"` and
`bpm_tolerance=None` or a positive number. Full route search propagates the same
controls through every hypothetical current/unplayed state. The independent
mode gate, source retention and lock handling still apply.

```text
superdj harmonize-learn --learner priority --examples ../work/controlled_examples.json --work ../priority-fit
superdj harmonize-rank --records ../work/records.json --profile ../work/priority-profile.json --current TRACK_ID --priority 0 --bpm-tolerance 6 --work ../ranks
superdj harmonize --records ../work/records.json --ranking-profile ../work/priority-profile.json --harmonic-mode fuzzy --priority 0 --bpm-tolerance 6 --work ../order
```

Version-two profiles retain their earlier feature layout and behavior. A new
priority-aware profile stays an explicit experiment; training agreement alone
does not replace the working profile or establish better complete routes or
audible transitions.

Use a shortlist of at most three allowed tracks, shrinking when fewer remain.
For the main Top3 comparison require at least four unplayed candidates; report
final pools of at most three separately, since including every remaining track
is trivial coverage. Top7 remains a historical diagnostic on pools of at least
eight. Report unique Top1, literal capped Top3, tied-best, fractional tie credit,
optimistic Top3 and guaranteed Top3 separately. Competition rank at most three
can contain more than three tracks at a tied boundary; the literal shortlist is
capped at three and flags that tie. The Python ranker accepts an explicit
`shortlist_limit` from one to seven for reproducing earlier experiments.
Validation now exposes `literal_top3_informative_rate` as the primary capped
metric. Legacy `top3_*` fields remain optimistic competition-rank counts for
compatibility and may include more than three tied tracks. Literal validation
uses the public ranker's descending utility and lexical-ID tie order, never
the demonstrated successor's first position in a training query. Keep a
separately seeded research tie policy labelled rather than merging its counts
with this metric; missing literal metrics in legacy rows are unavailable.
Repeated DJ.Studio orders can themselves differ; examine
that variation without converting it into a claim that our predictions match.
Tracks with identical BPM/key cannot be distinguished by these features.
Keep source-key/slider/mode provenance and ambiguous filenames visible.

## Route representation and search limits

Represent each source track as a graph vertex and each directed permitted
transition as an edge. Camelot is a circular number from one to twelve plus
mode A/B; BPM is a scalar feature. Twelve numbers are categories on one circle,
not twelve independent spatial axes. An allowance matrix can be precomputed
for the twenty-four keys, and track-level costs for the actual source pool.
Tracks sharing one cell remain separate vertices: their BPM, duration and
available future exits can differ, and repeated keys may bridge different
parts of a route. An existing user-locked chain is the only indivisible block.

The search state includes both the current outgoing tail and **every unplayed
block**. Caching just the current key is invalid: arriving at the same key after
consuming a different bridge track changes the possible continuation. The
route must visit every source once, retain endpoints and keep locks. Ordinary
shortest-path navigation, which can skip or revisit vertices, cannot certify
this requirement. Sparse harmony rules reduce branching without guaranteeing
a cheap exhaustive search over hundreds of tracks.

The pair-cost engine searches complete routes with exact small-pool
optimization and bounded beam/MILP methods described above. With a learned
profile, the default complete-prefix beam retains up to thirty-two prefixes
and up to seven successors per prefix through the final block. Each hypothesis
uses its own actual current tail and unplayed track mask. Equal tail/unplayed
states are merged; endpoints and indivisible locked chains remain constraints.
The branch cap is separate from the displayed Top3: truncating all search to
three immediate recommendations can discard useful route continuations.
Mode allowance is independent of fitted utility. Prefixes first prefer fewer
unaccepted inter-block boundaries, then normalized log weights over allowed
playable heads. When none is allowed, a native-BPM-distance fallback retains
all music. Locked internal links are not discretionary choices.

The older sequential policy with exact bounds for the final twelve blocks
remains a safety candidate. Both complete routes receive the same bounded
directed repair. Use the beam route unless it has more unaccepted final links;
the diagnostic full fitted loss does not select the final route. The report
retains both orders, original local decisions, selected method, prefix progress,
search budgets and separate post-repair scores. The default expansion cap is
100,000; a stopped beam receives a disclosed greedy completion instead of
dropping sources. The Python API `route_search="local"` reproduces the prior
policy for comparisons. Budgets and safety do not prove a global optimum.
Normalize learned utilities within each hypothetical remaining-pool
query before adding them along a path; raw conditional utilities have arbitrary
query offsets. Do not claim a fixed-initial-pool approximation is full dynamic
lookahead. Compare local next-track accuracy and freely generated route
compatibility separately; improvement in one does not establish the other.
If a newly fitted profile improves local Top3 but regresses the user's required
complete-route agreement, retain the approved working profile. Deliver the new
profile and its test playlists as a labelled experiment; do not silently replace
the active profile or claim whole-route superiority from the local metric.

For future bounded routing, preserve multiple candidate continuations, test
remaining connectivity, reopen provisional pairs/triples when stitching fails,
and memoize current-tail/unplayed-set states. A zero connectivity lower bound
does not prove that every vertex can be visited. If a completely allowed route
is not found, retain all music and disclose the bridges rather than label them
compatible or claim infeasibility without a proof.

Our relation-table allowance is not DJ.Studio's Solve status. If an imported
test requires frequent Solve, record that as a failed application check and
inspect detector differences before promoting the order. Do not infer a
probability of a successful audible mix from edge cost or agreement alone.
When a demonstrated next track is blocked by our table, compare its native
BPM distance with the nearest allowed and nearest overall remaining track.
Report the resulting gate-only agreement ceiling separately from measured
accuracy. A tempo association does not establish a saved wizard priority or
justify relabelling a disallowed key relation as compatible. Retain unknown
wizard controls as unknown even when all persisted settings are identical.
The saved DJS `mix.incompatible` field is not a count of Solve warnings. A
backup retaining the original imported order is useful analysis evidence, not
a demonstrated Harmonize route to learn from. When importing complete native
analysis, retain its provenance and do not describe the resulting correction
as improved independent audio detection.

Different good orders are expected. One match or mismatch does not prove
algorithm equivalence or error; finite beam search can also miss a lower-cost
path. Do not tune rules on one playlist and call that general validation.

## Chain and recurrence evidence

`compare_orders` includes exact contiguous windows of two, three and four
tracks when both orders have complete identical membership. The advanced
`chain_agreement` helper compares multiple same-pool reference variants, with
per-reference matches, union coverage and recurrence-weighted hits. Three/four
tracks means two/three edges, not three/four future search steps. Windows must
be adjacent and keep their direction; reversed or interrupted chains do not
match. Recurrence across different remaining pools is not the same as repeated
identical current/unplayed states and does not establish a predictability
ceiling. Group repeated backups by sources and mode, check their metadata and
historical controls, and inspect stable missed musical classes separately from
equivalent alternatives. An inferred expressive preference is not knowledge
of the proprietary algorithm's intention.

Freeze the existing profile and search before ingesting a new source cohort.
Score that independent cohort before fitting, then retain nested whole-cohort
validation and a separately labelled final profile for the next test. A final
profile fitted on current files is not held out on those files. Shared source
hashes are conservative grouping evidence, not proof that different encodings
of the same composition are independent songs. Keep semantic-alias sensitivity
separate from ID/name musical features. Learned weights and user archives stay
outside the public source repository.
Recompute connected source groups when new archives arrive: a new combined
pool can join formerly separate training groups. New orderings of previously
trained sources are order tests, not independent source tests. Exclude the
entire newly connected component, including its older variants, in an outer
holdout; do not reuse an old fold that trained on part of that component.

The fitted all-candidate softmax and a route policy conditioned on playable,
allowed heads have different declared supports. Conditioning is not by itself
a scoring error. A lower all-candidate diagnostic loss does not establish a
better constrained route or closer DJ.Studio agreement. Compare alternative
supports end to end before changing the policy; retain blocked demonstrations
in validation denominators even if an experimental fitter omits unsupported
targets from its training objective.

Additional selectable families `tempo_rank_density` and `tempo_rank_graph`
extend `tempo_pool` with nineteen and twenty-four remaining-pool terms.
They measure signed BPM position, nearby tempo density, position within a
key class, and directed-relation interactions; the graph family also measures
the tempo and residual connectivity of allowed successors. Native differences
within .01 BPM are neutral in percentile terms. These descriptors use the
current track and the unordered remaining metadata, never the teacher's
future route. They are continuation summaries, not a proof of lookahead
optimality. Dynamic rendering of rankings must recompute both mask-only and
current-dependent terms for every hypothetical pool; fixed full-pool values
are a different policy. Keep old feature layouts/profile loading intact.

Official sources checked 2026-10-11:
[Harmonize](https://help.dj.studio/en/articles/7878402-harmonize-previously-automix),
[harmonic relationships](https://help.dj.studio/en/articles/8118455-harmonic-matches-explained),
[Fuzzy explanation](https://dj.studio/blog/fuzzy-keymixing),
[M3U import](https://help.dj.studio/en/articles/9561495-import-a-playlist-file-m3u-or-m3u8-into-dj-studio).
