# Transition preference policy

Version 0.9 uses F shared-phase SYNC as the standard transition for newly built
sets, following the user's later listening feedback. This supersedes the older
ordinary-first/fallback policy. Reviewed montages and explicit cue locks remain
constraints.

Both decks consume one integrated cubic tempo curve and identical beat output
frames. Solo bodies retain native tempo. The intended map is checked separately
from isolated rendered attacks; dense ambiguous attacks remain unverified.

Source-only selection combines CUE-DETR/structure candidates, energy, vocal
estimates and locally valid beat/bar windows. Prefer 32-bar overlaps, 16 above
20% temporary tempo change; DJ Zhilin also prefers 32 at close tempos. Automatically
shorten when evidence or a short source would otherwise erase its core.

Target 180–240 seconds of solo music; hard cap 300. Keep at least 120 seconds
including overlaps. A small deficit can use one short stable whole-bar repeat
inside the body: at most four bars and ten seconds. Apply it identically to
prepared mixture and voice, retain the source and record the timeline edit.
Protected ranges/approved locks do not receive this edit. Larger/unrepairable
deficits stop instead of silently bypassing the minimum.

Complementary bands at 180/3500 Hz reconstruct the source. Bass transfers over
four shared beats near 55% of the transition. Mids transfer over 8–16 beats
around 43%, initially quieter on the incoming deck. Highs cross throughout
the overlap. Each pair's gain sum stays at or below one. These narrow bass/mid
curves belong to the explicit default style. DJ Zhilin instead spreads mids
across almost the whole overlap and bass across one eighth (four of 32 bars),
with gentle full-overlap highs. The private reference profile derives effective
band shares from approved native volume and EQ automation. Incoming
mids precede the bass handoff and outgoing mids remain afterwards. Complementary
gains and separate vocal handoff avoid two equally loud foregrounds. Preserve
native solo audio where feasible; do not infer acoustic alignment from the map.

The previous listening ranking remains useful for explicit creative choices:
F long, F short, E partial rhythms, C for an upward tempo change, deliberate
cut only by explicit request. The standard CLI does not automatically run that
entire creative ladder. Lower-level legacy auto, echo, vinyl and partial-rhythm
APIs remain available for reviewed experiments. New plan/run/make sets use
shared SYNC and a solo duration budget.

Explicit overlap_bars supports 2/4/8/16/32/40/44. Native exceptions of 8–256 beats
require both exact source locks; 51 beats is not a whole 4/4 phrase. Preserve
acoustic verification, phrase confidence and personal audition separately.
