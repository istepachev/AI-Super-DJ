"""Bounded, content-checked DSP intermediates, owned by one acoustic audit."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import threading
import numpy as np

_current = ContextVar('superdj_derived_audio', default=None)


class DerivedAudio:
    def __init__(self, limit_bytes):
        self.limit_bytes = max(0, int(limit_bytes))
        self.bytes = 0
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    def get(self, kind, audio, parameters, calculate):
        if not self.limit_bytes:
            return calculate()
        # Identity alone is unsafe: callers can edit an array in place, and
        # separately padded copies of the same mixture have different IDs.
        pcm = np.ascontiguousarray(audio)
        key = (kind, pcm.dtype.str, pcm.shape, audio.strides, parameters,
               hashlib.sha256(memoryview(pcm).cast('B')).digest())
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                from .execution import current
                if current():
                    current().count(kind + '_memory_hits')
                return self.entries[key][0]
            result = calculate()
            arrays = result if isinstance(result, tuple) else (result,)
            size = sum(a.nbytes for a in arrays)
            if size <= self.limit_bytes:
                while self.entries and self.bytes + size > self.limit_bytes:
                    _, (_, removed) = self.entries.popitem(last=False)
                    self.bytes -= removed
                for array in arrays:
                    array.flags.writeable = False
                self.entries[key] = (result, size)
                self.bytes += size
            return result


@contextmanager
def derived_scope(*, limit_bytes=None):
    """Nested checks share their caller's cache; no audio survives the scope."""
    if _current.get() is not None:
        yield _current.get()
        return
    if limit_bytes is None:
        from .resources import available_memory
        memory = available_memory()
        limit_bytes = min(768 * 1024**2, memory // 16) if memory is not None else 0
    cache = DerivedAudio(limit_bytes)
    token = _current.set(cache)
    try:
        yield cache
    finally:
        _current.reset(token)
        cache.entries.clear()
        cache.bytes = 0


def derived(kind, audio, parameters, calculate):
    cache = _current.get()
    return cache.get(kind, audio, parameters, calculate) if cache is not None else calculate()
