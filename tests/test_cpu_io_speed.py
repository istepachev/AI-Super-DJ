"""Independent queues and exact, single-read model input integrity."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import hashlib
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from superdj.audio import DECODER_VERSION, read_decoded
from superdj.execution import current, execution
from superdj.resources import GIB, preflight


def descriptor(path, samples, channels):
    data = np.asarray(samples, dtype='<f4').tobytes()
    path.write_bytes(data)
    return dict(path=str(path), frames=len(samples)//channels, channels=channels,
                sample_rate=44100, format=DECODER_VERSION, sha256=hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize('channels', [1, 2])
def test_loaded_bytes_are_verified_once_and_owned_after_file_removal(tmp_path, monkeypatch, channels):
    path = tmp_path/'input.f32'
    samples = np.linspace(-.9, .9, 2000, dtype='<f4')
    desc = descriptor(path, samples, channels)
    reads = []
    original = np.fromfile
    def observe(stream, *args, **kwargs):
        reads.append(stream.name)
        return original(stream, *args, **kwargs)
    monkeypatch.setattr(np, 'fromfile', observe)
    actual = read_decoded(desc, sample_rate=44100, channels=channels)
    assert reads == [str(path)]
    path.unlink()
    expected = samples if channels == 1 else samples.reshape(-1, 2)
    np.testing.assert_array_equal(actual, expected)
    actual.flat[0] = .25  # No read-only mapping or retained input file.


@pytest.mark.parametrize('change', ['same_length', 'short', 'long', 'bad_digest'])
def test_input_corruption_is_not_hidden_by_combined_read(tmp_path, change):
    path = tmp_path/'input.f32'
    desc = descriptor(path, np.arange(32, dtype='<f4'), 2)
    data = bytearray(path.read_bytes())
    if change == 'same_length': data[17] ^= 1
    if change == 'short': del data[-1:]
    if change == 'long': data.extend(b'\0')
    if change == 'bad_digest': desc['sha256'] = '0'*64
    path.write_bytes(data)
    with pytest.raises(ValueError, match='integrity'):
        read_decoded(desc, sample_rate=44100, channels=2)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_valid_hash_does_not_authorize_nonfinite_audio(tmp_path, value):
    desc = descriptor(tmp_path/'input.f32', [0., value], 2)
    with pytest.raises(ValueError, match='nonfinite'):
        read_decoded(desc, sample_rate=44100, channels=2)


def test_mutation_during_load_is_checked_against_loaded_bytes(tmp_path, monkeypatch):
    path = tmp_path/'input.f32'
    desc = descriptor(path, np.arange(32, dtype='<f4'), 2)
    original = np.fromfile
    def change_loaded(stream, *args, **kwargs):
        data = original(stream, *args, **kwargs)
        data[2] += .25
        return data
    monkeypatch.setattr(np, 'fromfile', change_loaded)
    with pytest.raises(ValueError, match='integrity'):
        read_decoded(desc, sample_rate=44100, channels=2)


def test_cpu_queue_runs_independently_while_model_queue_is_busy():
    release_model = threading.Event()
    model_started = threading.Event()
    cpu_started = threading.Barrier(4)
    with execution(jobs=1, cpu_jobs=3) as session:
        def model(_):
            model_started.set()
            assert release_model.wait(5)
            return 'model'
        def cpu(value):
            assert current() is session
            cpu_started.wait(timeout=5)
            return value
        with ThreadPoolExecutor(max_workers=2) as driver:
            a = driver.submit(copy_context().run, session.map, model, [0])
            assert model_started.wait(5)
            b = driver.submit(copy_context().run, session.cpu_map, cpu, range(3))
            try:
                cpu_started.wait(timeout=5)
                assert b.result(timeout=5) == [0, 1, 2]
            finally:
                release_model.set()
            assert a.result(timeout=5) == ['model']


@pytest.mark.parametrize('devices', [('cuda:0','cuda:1'), ('cuda:0','cuda:1','cuda:2','cuda:3')])
def test_more_cpu_jobs_do_not_create_extra_workers_per_gpu(monkeypatch, devices):
    from superdj import execution as module
    created = []
    active = {}; maximum = {}
    lock = threading.Lock()
    class Worker:
        def __init__(self, python, device, session, **kwargs):
            self.device = device; self.lock = threading.Lock(); created.append(device)
        def run(self, kind, job):
            with self.lock:
                with lock:
                    active[self.device] = active.get(self.device, 0)+1
                    maximum[self.device] = max(maximum.get(self.device, 0), active[self.device])
                time.sleep(.01)
                with lock: active[self.device] -= 1
                return job['value']
        def close(self): pass
    monkeypatch.setattr(module, 'Worker', Worker)
    with execution(jobs=2, cpu_jobs=4, devices=devices) as session:
        values = session.cpu_map(lambda i: session.model('fixture', 'beat', {'device':'cuda', 'value':i}), range(16))
    assert values == list(range(16))
    assert sorted(created) == ['cuda:0', 'cuda:1']
    assert maximum == {'cuda:0':1, 'cuda:1':1}


def test_cpu_iterator_close_waits_for_started_work_and_bounds_prefetch():
    started = []
    with execution(jobs=1, cpu_jobs=3) as session:
        def one(value):
            started.append(value); time.sleep(.01); return value
        values = session.cpu_imap(one, range(100))
        assert next(values) == 0
        values.close()
        count = len(started)
        time.sleep(.02)
        assert len(started) == count and count <= 6


def test_cpu_audio_uses_the_free_worker_and_releases_it_on_failure(tmp_path, monkeypatch):
    from superdj import execution as module
    slow_started = threading.Event(); release_slow = threading.Event(); created = []
    class Worker:
        def __init__(self, *a, **kw):
            self.lock = threading.Lock(); created.append(self)
        def close(self): pass
        def run(self, kind, job):
            with self.lock:
                if job['value'] == 'slow':
                    slow_started.set(); assert release_slow.wait(5)
                if job['value'] == 'failure': raise RuntimeError('audio failed')
                return job['value']
    monkeypatch.setattr(module, 'Worker', Worker)
    with execution(jobs=1, cpu_jobs=2) as session:
        def one(value): return session.audio(dict(value=value, checkout=dict(path=str(tmp_path), revision='fixed')))
        with ThreadPoolExecutor(max_workers=2) as driver:
            slow = driver.submit(copy_context().run, one, 'slow')
            assert slow_started.wait(5)
            try:
                assert one('fast') == 'fast'
                # The next turn in a round-robin queue would select the busy
                # first worker. A free-worker lease must finish before it does.
                assert driver.submit(copy_context().run, one, 'next').result(timeout=2) == 'next'
                with pytest.raises(RuntimeError, match='audio failed'): one('failure')
                assert one('after failure') == 'after failure'
            finally:
                release_slow.set()
            assert slow.result(timeout=5) == 'slow'
        assert not session.audio_busy
    assert len(created) == 2


def test_model_limit_spans_interpreters_and_releases_slot_on_error(monkeypatch):
    from superdj import execution as module
    active = maximum = 0
    lock = threading.Lock()
    class Worker:
        def __init__(self, *a, **kw): pass
        def close(self): pass
        def run(self, kind, job):
            nonlocal active, maximum
            with lock:
                active += 1; maximum = max(maximum, active)
            try:
                time.sleep(.01)
                if job.get('fail'): raise RuntimeError('inference failed')
                return job['value']
            finally:
                with lock: active -= 1
    monkeypatch.setattr(module, 'Worker', Worker)
    with execution(jobs=2, cpu_jobs=4) as session:
        def one(i):
            try:
                return session.model('interpreter-'+str(i%4), 'beat',
                    {'device':'cpu','value':i,'fail':i in (0, 7)})
            except RuntimeError: return 'rejected'
        assert session.cpu_map(one, range(12)) == ['rejected',1,2,3,4,5,6,'rejected',8,9,10,11]
        assert session.model_running == 0
    assert maximum == 2


def test_cpu_audio_limit_spans_distinct_checkout_workers(tmp_path,monkeypatch):
    from superdj import execution as module
    active=maximum=0;lock=threading.Lock()
    class Worker:
        def __init__(self,*a,**kw):pass
        def close(self):pass
        def run(self,kind,job):
            nonlocal active,maximum
            with lock:active+=1;maximum=max(maximum,active)
            time.sleep(.02)
            with lock:active-=1
            return job['value']
    monkeypatch.setattr(module,'Worker',Worker)
    with execution(jobs=1,cpu_jobs=2)as session:
        def one(i):return session.audio(dict(value=i,checkout=dict(path=str(tmp_path/str(i)),revision='fixed')))
        with ThreadPoolExecutor(max_workers=4)as pool:
            futures=[pool.submit(copy_context().run,one,i)for i in range(4)]
            assert [f.result(timeout=5)for f in futures]==list(range(4))
        assert not session.audio_busy
    assert maximum==2


@pytest.mark.parametrize('memory', [None, 5*GIB, 64*GIB])
def test_resource_plan_separates_gpu_slots_without_multiplying_ram(tmp_path, memory):
    machine = dict(cpu_threads=24, available_memory_bytes=memory, cuda_probe='measured',
                   gpus=[dict(device='cuda:0', free_bytes=10*GIB)])
    result = preflight(['a', 'b'], output=tmp_path, jobs=4, cpu_jobs=4, devices=['cuda:0'],
        machine=machine, probe=lambda p:(480.,44100), disk=lambda p:(tmp_path,'disk',1000*GIB))
    assert result['jobs'] == 1
    if memory == 64*GIB:
        assert result['cpu_jobs'] == 4 and result['preparation_overlap']
    else:
        assert result['cpu_jobs'] == 1 and not result['preparation_overlap']
