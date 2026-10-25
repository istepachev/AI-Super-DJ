"""Opt-in SuperDJ admission/thread policy; no musical settings or watchdog."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import sys


ENVIRONMENT_KEY = 'SUPERDJ_RESOURCE_LIMITS_JSON'
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                    'BLIS_NUM_THREADS')
_active = ContextVar('superdj_resource_limits', default=None)
_report = ContextVar('superdj_resource_limits_report', default=None)


@dataclass(frozen=True)
class LimitSettings:
    enabled: bool = False
    cpu_cores: int = 4
    model_jobs: int = 1
    cpu_jobs: int = 1
    math_threads: int = 1
    memory_headroom_gib: float = 4.


def settings(value=None):
    if isinstance(value, LimitSettings):
        value = asdict(value)
    if value is None:
        return LimitSettings()
    if not isinstance(value, dict) or not isinstance(value.get('enabled', False), bool):
        raise ValueError('resource_limits must be an object with a boolean enabled field')
    if not value.get('enabled', False):
        return LimitSettings()
    if set(value) - {f.name for f in fields(LimitSettings)}:
        raise ValueError('Unknown SuperDJ resource limit setting')
    result = {**asdict(LimitSettings()), **value}
    for name, maximum in (('cpu_cores', 64), ('model_jobs', 4),
                          ('cpu_jobs', 4), ('math_threads', 16)):
        number = result[name]
        if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= maximum:
            raise ValueError(f'{name} must be an integer from 1 to {maximum}')
    headroom = result['memory_headroom_gib']
    if (isinstance(headroom, bool) or not isinstance(headroom, (int, float))
            or not math.isfinite(headroom) or not 0 <= headroom <= 1024):
        raise ValueError('memory_headroom_gib must be finite and between 0 and 1024')
    return LimitSettings(**result)


def runtime_path():
    return Path.home() / '.cache' / 'superdj' / 'runtime.local.json'


def load_settings(runtime=None):
    if runtime is None:
        path = runtime_path()
        runtime = json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}
    if not isinstance(runtime, dict):
        raise ValueError('SuperDJ runtime config must be a JSON object')
    return settings(runtime.get('resource_limits'))


def current_settings():
    active = _active.get()
    if active is not None:
        return active
    value = os.environ.get(ENVIRONMENT_KEY)
    return settings(json.loads(value)) if value else LimitSettings()


def current_report():
    return _report.get()


def bounded_jobs(jobs, cpu_jobs, limits=None):
    limits = current_settings() if limits is None else settings(limits)
    model = max(1, min(4, int(jobs)))
    cpu = model if cpu_jobs is None else max(1, min(4, int(cpu_jobs)))
    if limits.enabled:
        model = min(model, limits.model_jobs)
        cpu = min(cpu, limits.cpu_jobs)
    return model, cpu


def math_thread_limit(default):
    limits = current_settings()
    if not limits.enabled:
        return default
    inherited = []
    for name in THREAD_VARIABLES:
        try:
            value = int(os.environ.get(name, '0'))
        except ValueError:
            continue
        if value > 0:
            inherited.append(value)
    return min(default, limits.math_threads, *inherited)


def _thread_environment(limits):
    changed = {}
    for name in THREAD_VARIABLES:
        if name == 'NUMBA_NUM_THREADS' and 'numba' in sys.modules:
            # Numba fixes its launch capacity at import. Changing this env
            # after initialization can make later kernels raise on reload.
            # Scope the public thread mask instead; owned new workers still
            # receive their env before importing Numba.
            continue
        previous = os.environ.get(name)
        try:
            existing = int(previous) if previous is not None else limits.math_threads
        except ValueError:
            existing = limits.math_threads
        selected = min(limits.math_threads, existing) if existing > 0 else limits.math_threads
        applied = str(selected)
        if previous != applied:
            changed[name] = (previous, applied)
            os.environ[name] = applied
    return changed


def _owned_cli_process():
    arguments = list(getattr(sys, 'orig_argv', sys.argv))
    return (Path(sys.argv[0]).name.lower() in ('superdj', 'superdj.exe', 'superdj.py', 'ai-super-dj', 'ai-super-dj.exe')
            or any(a == '-m' and b in ('superdj', 'superdj.cli', 'superdj.model_worker')
                   for a, b in zip(arguments, arguments[1:])))


def bootstrap_math_threads():
    """Prepare owned CLI/worker pools before audio libraries are imported.

    An SDK import alone does not impose a private workstation profile on an
    unrelated embedding application. Its execution scope handles pools later.
    """
    if not os.environ.get(ENVIRONMENT_KEY) and not _owned_cli_process():
        return
    try:
        limits = current_settings() if os.environ.get(ENVIRONMENT_KEY) else load_settings()
    except (OSError, ValueError, TypeError):
        # Configuration commands/help remain usable to repair invalid settings.
        return
    if limits.enabled:
        _thread_environment(limits)


@contextmanager
def numba_thread_scope(report=None):
    """Numba's active mask belongs to each calling thread, not the process."""
    module = sys.modules.get('numba')
    previous = applied = None
    limits = current_settings()
    if module is not None and limits.enabled:
        try:
            before = module.get_num_threads()
            selected = min(before, math_thread_limit(limits.math_threads))
            module.set_num_threads(selected)
            previous, applied = before, selected
            if report is not None:
                report['numba_active_threads'] = module.get_num_threads()
        except (AttributeError, ValueError, RuntimeError) as error:
            if report is not None:
                report['numba_thread_control_error'] = type(error).__name__
    try:
        yield
    finally:
        if previous is not None:
            try:
                if module.get_num_threads() == applied:
                    module.set_num_threads(previous)
            except (AttributeError, ValueError, RuntimeError) as error:
                if report is not None:
                    report['numba_thread_restore_error'] = type(error).__name__


@contextmanager
def limit_scope(value=None, *, native=False):
    """Bound this run; never widen inherited native limits on scope exit.

    Native root policy is for an owned CLI process, not an embedding host.
    Known child workers can be limited separately by their Popen handles.
    Config rollback affects future runs and never disables an external guard.
    """
    if _active.get() is not None:
        yield current_report()
        return
    limits = (current_settings() if os.environ.get(ENVIRONMENT_KEY) else load_settings()) if value is None else settings(value)
    if not limits.enabled:
        before = os.environ.get(ENVIRONMENT_KEY)
        encoded = json.dumps(asdict(limits), sort_keys=True)
        os.environ[ENVIRONMENT_KEY] = encoded
        token = _active.set(limits)
        try:
            yield None
        finally:
            _active.reset(token)
            if os.environ.get(ENVIRONMENT_KEY) == encoded:
                if before is None:
                    os.environ.pop(ENVIRONMENT_KEY, None)
                else:
                    os.environ[ENVIRONMENT_KEY] = before
        return
    changes = _thread_environment(limits)
    before = os.environ.get(ENVIRONMENT_KEY)
    encoded = json.dumps(asdict(limits), sort_keys=True)
    os.environ[ENVIRONMENT_KEY] = encoded
    token = _active.set(limits)
    report = {'settings': asdict(limits), 'hard_ram_cap': False,
              'hard_process_count_cap': False, 'hard_io_bandwidth_cap': False,
              'background_monitor_created': False, 'memory_reservation': False,
              'root_native_policy': None, 'math_pool_control': 'environment for new pools'}
    report_token = _report.set(report)
    pool_scope = nullcontext()
    try:
        if native:
            from .native_limits import apply_owned_limits
            report['root_native_policy'] = apply_owned_limits(limits)
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
            counts = [p.get('num_threads', limits.math_threads) for p in threadpool_info()]
            effective = min([limits.math_threads, *[n for n in counts if isinstance(n, int) and n > 0]])
            pool_scope = threadpool_limits(limits=effective)
            report['math_pool_control'] = 'threadpoolctl and inherited environment'
            report['effective_math_threads'] = effective
        except (ImportError, OSError, RuntimeError) as error:
            report['effective_math_threads'] = None
            report['math_pool_control_error'] = type(error).__name__
        with pool_scope, numba_thread_scope(report):
            yield report
    finally:
        _report.reset(report_token)
        _active.reset(token)
        for name, (previous, applied) in changes.items():
            if os.environ.get(name) == applied:
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
        if os.environ.get(ENVIRONMENT_KEY) == encoded:
            if before is None:
                os.environ.pop(ENVIRONMENT_KEY, None)
            else:
                os.environ[ENVIRONMENT_KEY] = before


@contextmanager
def cli_limit_scope(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if (not arguments or arguments[0] == 'resource-mode'
            or any(a in ('--help', '-h', '--version') for a in arguments)):
        yield None
        return
    with limit_scope(native=_owned_cli_process()) as report:
        yield report
