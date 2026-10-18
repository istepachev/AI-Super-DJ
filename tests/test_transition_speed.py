"""Exact DSP reuse and bounded scheduling, including invalidation and failure."""
import threading
import time
from contextvars import copy_context
import numpy as np
import pytest
from superdj.derived_audio import derived_scope, derived
from superdj.execution import execution, current


def test_content_cache_invalidates_edits_format_geometry_and_parameters():
    audio = np.arange(80, dtype='float32').reshape(40, 2)
    calls = []
    def get(a=audio, parameters=(8000, 180)):
        return derived('test', a, parameters, lambda: (calls.append(1), a.copy())[1])
    with derived_scope(limit_bytes=10000) as cache:
        first = get()
        assert not first.flags.writeable
        assert get(audio.copy()) is first
        audio[0, 0] += 1
        assert get() is not first
        get(audio.astype('float64'))
        get(np.asfortranarray(audio))
        get(audio.reshape(20, 4))
        get(parameters=(16000, 180))
        get(parameters=(8000, 200))
        assert len(calls) == 7
        assert cache.bytes <= cache.limit_bytes
    assert not cache.entries and cache.bytes == 0
    get()
    assert len(calls) == 8


def test_cache_eviction_low_memory_and_exception_cleanup(monkeypatch):
    monkeypatch.setattr('superdj.resources.available_memory', lambda: None)
    audio = np.ones((40, 2))
    with derived_scope() as disabled:
        assert disabled.limit_bytes == 0
        assert derived('test', audio, (), lambda: audio.copy()).flags.writeable
        assert not disabled.entries
    with pytest.raises(RuntimeError), derived_scope(limit_bytes=700) as cache:
        for i in range(5):
            derived('test', audio, (i,), lambda: audio.copy())
        assert len(cache.entries) == 1 and cache.bytes == 640
        raise RuntimeError('consumer failed')
    assert not cache.entries and cache.bytes == 0


@pytest.mark.parametrize('style', ['default', 'zhilin'])
@pytest.mark.parametrize('policy', ['shared', 'smooth', 'sequential'])
def test_cached_bands_preserve_every_sample_and_reconstruction(style, policy, monkeypatch):
    from superdj import mixing
    from superdj.percussion import verification_components, percussion_gains
    rng = np.random.default_rng(101)
    a, b, da, db = [rng.normal(0, .03, (24000, 2)).astype('float32') for _ in range(4)]
    settings = dict(style=style, beats=32, drums_out=da, drums_in=db, percussion_policy=policy,
                    allow_abrupt_percussion=policy == 'sequential')
    reference, evidence = mixing.blend(a, b, 8000, **settings)
    gains, _ = mixing.envelopes(len(a), 32, style=style)
    gains = percussion_gains(gains, None, 32, policy)
    parts = verification_components(a, b, 8000, gains, drums_out=da, drums_in=db)
    real = mixing.sosfiltfilt
    calls = []
    def counted(*args, **kw):
        calls.append(1)
        return real(*args, **kw)
    monkeypatch.setattr(mixing, 'sosfiltfilt', counted)
    with derived_scope(limit_bytes=4*24000*2*8*3):
        actual, report = mixing.blend(a, b, 8000, **settings)
        assert np.array_equal(actual, reference) and report == evidence
        actual_parts = verification_components(a, b, 8000, gains, drums_out=da, drums_in=db)
        for role in parts:
            np.testing.assert_array_equal(actual_parts[role], parts[role])
        # Changed gain curves reuse the same PCM bands, without caching gains.
        mixing.blend(a, b, 8000, tonal_conflict=True, **settings)
        assert len(calls) == 8  # two filters for each of the four streams


def test_attack_band_reuse_retains_rows_for_padded_copies_and_changed_audio(monkeypatch):
    from superdj import kick_grid
    rng = np.random.default_rng(202)
    audio = rng.normal(0, .02, (32000, 2)).astype('float32')
    centres = [.7, 1.7, 2.7]
    templates = rng.normal(size=(3, 480))
    reference = kick_grid.template_attack_rows(audio, 8000, centres, templates)
    real = kick_grid.resample_poly
    calls = []
    def counted(*args, **kw):
        calls.append(1)
        return real(*args, **kw)
    monkeypatch.setattr(kick_grid, 'resample_poly', counted)
    with derived_scope(limit_bytes=2000000):
        assert kick_grid.template_attack_rows(audio, 8000, centres, templates) == reference
        assert kick_grid.template_attack_rows(audio.copy(), 8000, centres, templates) == reference
        assert len(calls) == 1
        audio[15000] += .1
        kick_grid.template_attack_rows(audio, 8000, centres, templates)
        kick_grid.template_attack_rows(audio, 8000, centres, templates, attack_band_hz=(35, 180))
        assert len(calls) == 3


def test_completion_pipeline_works_ahead_of_slow_first_item_and_slow_consumer(monkeypatch):
    monkeypatch.setattr('superdj.resources.available_memory', lambda: 8*1024**3)
    release = threading.Event()
    third = threading.Event()
    calls = []
    with execution(jobs=2) as session:
        def one(i):
            assert current() is session
            calls.append(i)
            if i == 0:
                assert release.wait(5)
            if i == 2:
                third.set()
            return i
        iterator = session.imap(one, range(100))
        first = []
        context = copy_context()
        consumer = threading.Thread(target=lambda: context.run(lambda:first.append(next(iterator))))
        consumer.start()
        try:
            assert third.wait(3), 'Free worker stalled behind the first deck'
            assert len(calls) == 3, 'Unbounded rendering ahead of the consumer'
        finally:
            release.set()
            consumer.join(5)
        assert first == [0]
        deadline = time.monotonic()+3
        while len(calls)<4 and time.monotonic()<deadline:
            time.sleep(.001)
        assert len(calls) == 4
        iterator.close()
        assert len(calls) == 4


def test_ordered_failures_and_input_iteration_failure_cancel_cleanly():
    with execution(jobs=2) as session:
        def one(i):
            if i == 2:
                raise ValueError('deck failed')
            return i
        iterator = session.imap(one, range(20))
        assert next(iterator) == 0 and next(iterator) == 1
        with pytest.raises(ValueError, match='deck failed'):
            next(iterator)
        def values():
            yield 5
            raise RuntimeError('input failed')
        iterator = session.imap(lambda x:x, values())
        assert next(iterator) == 5
        with pytest.raises(RuntimeError, match='input failed'):
            next(iterator)


def test_device_pair_runs_on_both_explicit_workers_and_restores_context(monkeypatch):
    from superdj import execution as module
    monkeypatch.setattr('superdj.resources.available_memory', lambda: 8*1024**3)
    barrier = threading.Barrier(2)
    class Worker:
        def __init__(self, python, device, session):
            self.device = device
        def run(self, kind, job):
            barrier.wait(timeout=5)
            return self.device
        def close(self):
            pass
    monkeypatch.setattr(module, 'Worker', Worker)
    with execution(jobs=2, devices=['cuda:0', 'cuda:1']) as session:
        assert session.device_pair(lambda x:session.model('python', 'probe', {'device':'auto'}), [('out',0),('in',0)]) == ['cuda:0','cuda:1']
        assert module._model_device.get() is None
        assert session.metrics['counts']['parallel_final_deck_pairs'] == 1


@pytest.mark.parametrize('jobs,devices,memory', [(1,['cuda:0','cuda:1'],8*1024**3),(2,['cuda','cuda:0'],8*1024**3),(2,['cpu'],8*1024**3),(2,['cuda:0','cuda:1'],10)])
def test_device_pair_fallbacks_retain_serial_execution(monkeypatch,jobs,devices,memory):
    from superdj.execution import _model_device
    monkeypatch.setattr('superdj.resources.available_memory', lambda: memory)
    main = threading.get_ident()
    with execution(jobs=jobs, devices=devices) as session:
        assert session.device_pair(lambda v:(threading.get_ident(),_model_device.get()), [('out',0),('in',0)]) == [(main,None),(main,None)]


def test_device_pair_propagates_failure_and_finishes_other_side(monkeypatch):
    monkeypatch.setattr('superdj.resources.available_memory', lambda: 8*1024**3)
    finished = threading.Event()
    def one(value):
        if value[0] == 'out':
            raise ValueError('separation failed')
        finished.set()
    with execution(jobs=2, devices=['cuda:0','cuda:1']) as session:
        with pytest.raises(ValueError, match='separation failed'):
            session.device_pair(one, [('out',0),('in',0)])
        # The second task either completes or is cancelled before starting.
        assert not any(t.name.startswith('ThreadPoolExecutor') and t.is_alive()
                       for t in threading.enumerate())


def test_final_deck_pair_preserves_roles_geometry_pcm_and_model_settings(tmp_path,monkeypatch):
    import soundfile as sf
    from superdj import stems,percussion
    from superdj.execution import _model_device
    monkeypatch.setattr('superdj.resources.available_memory', lambda: 8*1024**3)
    config={'device':'auto','model':'unchanged-model','precision':'unchanged'}
    monkeypatch.setattr(stems,'configuration',lambda:dict(config))
    barrier=threading.Barrier(2)
    seen=[]
    def separate(source,digest,cache,settings,ffmpeg):
        assert settings==config
        pcm,rate=sf.read(source,dtype='float32',always_2d=True)
        seen.append(_model_device.get())
        barrier.wait(timeout=5)
        return dict(drums_path=str(source),provenance={'model':'unchanged-model'})
    monkeypatch.setattr(stems,'separate',separate)
    a=np.ones((4000,2),dtype='float32')*.1;b=a*.5
    with execution(jobs=2,devices=['cuda:0','cuda:1']):
        result,reports=percussion.final_drum_estimates(a,b,8000,tmp_path)
    np.testing.assert_array_equal(result['drums_out'],a)
    np.testing.assert_array_equal(result['drums_in'],b)
    assert [r['role'] for r in reports]==['out','in']
    assert set(seen)=={'cuda:0','cuda:1'}
