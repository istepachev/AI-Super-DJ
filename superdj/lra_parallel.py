"""Ordered independent loudness windows for the pinned mastering adapter."""
import ast
import hashlib
import inspect
import os
import textwrap
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

_AUDITED_AST = '265f23afdd8607e1170af97b30b62f50831d6537480cae389ad0a94e475a9d10'


def admitted_jobs(jobs=None):
    from .resources import available_memory
    cpu = os.cpu_count() or 1
    selected = min(4, max(1, cpu // 2)) if jobs is None else max(1, min(4, int(jobs)))
    memory = available_memory()
    return selected if memory is not None and memory >= selected*64*1024**2+128*1024**2 else 1


def window_values(data, rate, window, hop, meter_type, jobs, report):
    """Keep each upstream window, filter reset, gating and scalar unchanged."""
    import numpy as np
    starts = range(0, data.shape[0]-window, hop)
    count = len(starts)
    from .resources import available_memory
    memory = available_memory()
    channels = 1 if data.ndim == 1 else data.shape[1]
    required = jobs*window*channels*8*12+128*1024**2
    if count < 2*jobs or memory is None or memory < required:
        jobs = 1
    local = threading.local()
    def one(start):
        if not hasattr(local, 'meter'):
            local.meter = meter_type(rate)
        value = local.meter.integrated_loudness(data[start:start+window])
        return value if np.isfinite(value) else None
    if jobs == 1:
        return [value for start in starts if (value := one(start)) is not None]
    try:
        # Submit bounded batches, rather than retaining every window or copy.
        from collections import deque
        iterator = iter(starts)
        values = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            pending = deque()
            for _ in range(2*jobs):
                start = next(iterator, None)
                if start is not None:
                    pending.append(pool.submit(one, start))
            while pending:
                value = pending.popleft().result()
                if value is not None:
                    values.append(value)
                start = next(iterator, None)
                if start is not None:
                    pending.append(pool.submit(one, start))
        report['parallel_calls'] += 1
        report['windows'] += count
        return values
    except Exception as error:
        # Infrastructure failures never silently discard an LRA result.
        report['fallback'] = type(error).__name__
        meter = meter_type(rate)
        values = []
        for start in starts:
            value = meter.integrated_loudness(data[start:start+window])
            if np.isfinite(value):
                values.append(value)
        return values


@contextmanager
def parallel_lra(module, *, enabled=True, jobs=None):
    """Adapt only an exactly audited loop; always restore the upstream function."""
    original = getattr(module, '_measure_lra', None)
    report = dict(enabled=False, jobs=1, parallel_calls=0, windows=0,
                  unchanged_windows_filter_resets_and_order=True)
    try:
        selected = admitted_jobs(jobs) if enabled else 1
        if original is None:
            raise ValueError('Audited LRA function is unavailable')
        if selected > 1:
            tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
            if hashlib.sha256(ast.dump(tree.body[0]).encode()).hexdigest() != _AUDITED_AST:
                raise ValueError('Audited LRA function signature changed')
            loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
            if len(loops) != 1:
                raise ValueError('Audited LRA window loop changed')
            class Replace(ast.NodeTransformer):
                def visit_For(self, node):
                    return ast.parse('short_term = _superdj_windows(data, rate, window_samples, hop_samples)').body[0]
            def windows(data, rate, window, hop):
                return window_values(data, rate, window, hop, module.pyln.Meter, selected, report)
            namespace = dict(original.__globals__, _superdj_windows=windows)
            exec(compile(ast.fix_missing_locations(Replace().visit(tree)), __file__, 'exec'), namespace)
            module._measure_lra = namespace[original.__name__]
            report.update(enabled=True, jobs=selected)
    except Exception as error:
        report['fallback'] = str(error)
    try:
        yield report
    finally:
        if original is not None:
            module._measure_lra = original
