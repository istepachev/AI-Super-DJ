"""Optional source-control retry; actual audio observations and gates stay intact."""
from copy import deepcopy

from .timing_safety import SourceWindowFailure


def refine_with_native_clock_recovery(path, warp, sample_rate, evidence, *, enabled=False):
    from .kick_grid import refine_source_controls
    try:
        return refine_source_controls(path, warp, sample_rate, evidence)
    except SourceWindowFailure as original:
        rejected = original.evidence or {}
        reports = warp.get('source_kick_alignment', {})
        roles = [original.role] if original.role else list(reports)
        eligible = [role for role in roles if
            reports.get(role, {}).get('stable_native_primary_clock') and
            (evidence or {}).get(role, {}).get('path') and
            not evidence[role].get('stable_clock_controls')]
        if not enabled or rejected.get('passed') is not False or not eligible:
            raise
        candidate = deepcopy(warp)
        controls = deepcopy(evidence)
        for role in eligible:
            controls[role]['stable_clock_controls'] = True
        try:
            result = refine_source_controls(path, candidate, sample_rate, controls)
        except SourceWindowFailure:
            raise original
        for role in reports:
            for key in ('events', 'native_rhythm_observations'):
                if reports[role].get(key) != result.get(role, {}).get(key):
                    raise original
        candidate['supported_native_control_recovery'] = {
            'roles': eligible, 'rejected_source_rate_audit': rejected,
            'actual_source_references_preserved': True,
            'timing_gates_unchanged': True,
        }
        warp.clear()
        warp.update(candidate)
        return result
