"""Optional exact-loop compilation for the audited mastering adapter."""
import ast
import inspect
import textwrap
from contextlib import contextmanager


def _release(gain_needed, coefficient):
    import numpy as np
    result = np.ones(len(gain_needed), dtype=np.float64)
    envelope = 1.0
    for i in range(len(gain_needed)):
        target = gain_needed[i]
        if target < envelope:
            envelope = target
        else:
            envelope = coefficient * envelope + (1.0 - coefficient) * target
        result[i] = envelope
    return result


_compiled = None


def exact_release(gain_needed, coefficient):
    global _compiled
    if _compiled is None:
        from numba import njit
        # Imports are outside the compiled function; no fast-math, FMA or disk cache.
        code = textwrap.dedent(inspect.getsource(_release))
        tree = ast.parse(code)
        tree.body[0].body = [n for n in tree.body[0].body if not isinstance(n, ast.Import)]
        import numpy as np
        namespace = {'np': np}
        exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
        _compiled = njit(cache=False, fastmath=False)(namespace['_release'])
    return _compiled(gain_needed, coefficient)


@contextmanager
def accelerated_limiter(module, enabled=True):
    """Replace only the Python envelope loop; retain all upstream DSP operations."""
    original = module.limit_peaks_lookahead
    report = {'enabled': False, 'precision': 'float64; fastmath disabled', 'disk_jit_cache': False}
    try:
        if enabled:
            import numpy as np
            # Compile and verify the available backend before changing the adapter.
            probes = np.array([1., .9, .2, 1., .7, .1, 1., 1.], dtype=np.float64)
            for coefficient in (.0, .5, .999773268338335):
                if not np.array_equal(exact_release(probes, coefficient), _release(probes, coefficient)):
                    raise ValueError('Compiled envelope differs from the reference')
            tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
            loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
            expected = ast.parse('''for i in range(n_samples):
    target = gain_needed[i]
    if target < env:
        env = target
    else:
        env = release_coeff * env + (1.0 - release_coeff) * target
    smoothed[i] = env
''').body[0]
            if len(loops) != 1 or ast.dump(loops[0]) != ast.dump(expected):
                raise ValueError('Audited limiter loop signature changed')
            class Replace(ast.NodeTransformer):
                def visit_For(self, node):
                    return ast.parse('smoothed = _superdj_release(gain_needed, release_coeff)').body[0]
            namespace = dict(original.__globals__, _superdj_release=exact_release)
            exec(compile(ast.fix_missing_locations(Replace().visit(tree)), __file__, 'exec'), namespace)
            module.limit_peaks_lookahead = namespace[original.__name__]
            report['enabled'] = True
    except Exception as error:
        report['fallback'] = str(error)
    try:
        yield report
    finally:
        module.limit_peaks_lookahead = original
