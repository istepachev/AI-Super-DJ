"""Source-only admission for a coherent earlier family as the primary clock."""

def family_coverage(primary,front):
    base={r['index'] for r in primary};upper={r['index'] for r in front};common=sorted(base&upper)
    fraction=len(common)/len(base) if base else 0.
    span=(common[-1]-common[0])/(max(base)-min(base)) if len(common)>1 and max(base)>min(base) else 0.
    return {'primary_events':len(base),'front_events':len(common),'front_primary_fraction':fraction,
        'front_temporal_span_fraction':span,'supported_for_primary_clock':len(common)>=8 and fraction>=.60 and span>=.75,
        'scope':'native family support before rendering; partial figures stay observations, not whole-window timing controls'}
