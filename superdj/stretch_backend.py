"""Explicit per-source pitch-preserving backend, retaining the global default."""


def selected_backend(record,default):
    engine=record.get('rubberband_engine_override',default)
    if engine not in ('r2','r3'):
        raise ValueError('Explicit source Rubber Band engine must be r2 or r3')
    return engine


def backend_arguments(engine,profile=None):
    if engine=='r2':result=['--fast','--detector-perc']
    elif engine=='r3':result=['--fine']
    else:raise ValueError('Rubber Band engine must be r2 or r3')
    if profile is not None:
        if engine!='r2' or profile not in ('short','crisp6'):
            raise ValueError('A source transient profile must be R2 short or crisp6')
        result += ['--window-short'] if profile=='short' else ['--crisp','6']
    return result
