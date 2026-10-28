"""One bounded placement of the existing soft curve after an uncertain opening."""
from math import ceil
from numbers import Integral


def late_entry_recipe(transition,errors):
    """Retain a verified old foundation until the new opening is supported.

    This only proposes existing gains. No missing attack becomes verified;
    both the source and the complete actual mixture still need every gate.
    """
    count=transition.get('beats',0)
    if isinstance(count,bool)or not isinstance(count,Integral)or count<64:return None
    old,new=transition.get('from'),transition.get('to')
    if old is None or new is None:return None
    for error in errors:
        evidence=error if isinstance(error,dict)else getattr(error,'evidence',{})
        if not isinstance(evidence,dict):continue
        if evidence.get('stage')=='source-only timing preflight':continue
        local=evidence.get('local_rhythm_coverage',{})
        if not isinstance(local,dict):continue
        if local.get('passed')is not False:continue
        decks=local.get('decks',[])
        if not isinstance(decks,(list,tuple)):continue
        outgoing=next((d for d in decks if isinstance(d,dict)and d.get('id')==old and d.get('role')=='tail'),{})
        incoming=next((d for d in decks if isinstance(d,dict)and d.get('id')==new and d.get('role')=='head'),{})
        if outgoing.get('sufficient_for_automatic_window')is not True or incoming.get('sufficient_for_automatic_window')is not False:continue
        missing=incoming.get('unverified_audible_beat_indices',[])
        if not isinstance(missing,(list,tuple)):continue
        if not missing or any(isinstance(i,bool)or not isinstance(i,Integral)or not 0<i<count for i in missing):continue
        groups=[]
        for beat in sorted(set(missing)):
            if not groups or beat!=groups[-1][-1]+1:groups.append([])
            groups[-1].append(int(beat))
        opening=next((g for g in groups if g[0]<=16 and 8<len(g)<=32 and g[-1]<=36),None)
        if opening is None or any(len(g)>8 for g in groups if g is not opening):continue
        first=max(4.,4.*ceil(opening[-1]/4.)-8.)
        if first>28 or first+32>count:continue
        return {'start_beat':first,'end_beat':first+32.,'bias':1.}, {
            'unverified_opening_shared_beats':[opening[0],opening[-1]],
            'outgoing_local_coverage_already_sufficient':True,
            'source_clock_pitch_overlap_and_transfer_width_unchanged':True,
            'missing_attacks_declared_verified':False,
            'actual_source_front_local_energy_and_continuity_gates_required':True}
    return None
