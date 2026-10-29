"""Retain existing soft mid curves after an automatic continuity refusal."""
from functools import wraps


CONTINUITY_REFUSAL = 'Accompaniment energy handoff is too short;'


def retain_middle_continuity(function):
    @wraps(function)
    def checked(*args, **options):
        try:
            return function(*args, **options)
        except ValueError as initial_error:
            if (options.get('mid_energy_mode', 'auto') != 'auto'
                    or options.get('whole_spectrum_handoff') is not None
                    or not str(initial_error).startswith(CONTINUITY_REFUSAL)):
                raise
            failed = [{'mode': 'auto', 'reason': str(initial_error)}]
            for mode in ('power', 'legacy'):
                try:
                    mixed, report = function(*args, **{**options, 'mid_energy_mode': mode})
                except ValueError as error:
                    if not str(error).startswith(CONTINUITY_REFUSAL):
                        raise
                    failed.append({'mode': mode, 'reason': str(error)})
                    continue
                report = {**report, 'mid_continuity_recovery': {
                    'executed': True, 'selected_mode': mode, 'failed': failed,
                    'source_clock_overlap_and_band_positions_unchanged': True,
                    'continuity_thresholds_unchanged': True,
                    'scope': 'gain continuity only; actual front and quality audits remain required'}}
                return mixed, report
            raise
    return checked
