"""Run-owned scratch space, verified content objects and bounded model workers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import ast


_STAGE_ROOTS = {
    'decode': ('audio',), 'analysis': ('analysis', 'model_worker'), 'preparation': ('adapters', 'model_worker'),
    'stem': ('stems', 'model_worker'),
    'prepared_measurements': ('quality',), 'bass_profile': ('rhythm',),
    'clip': ('render', 'model_worker'), 'overlap': ('render', 'mixing', 'percussion', 'stems', 'model_worker'),
}


def stage_fingerprint(stage, source=None):
    """Include transitive local imports; unknown stages conservatively use all code."""
    source = Path(source or Path(__file__).parent)
    pending = list(_STAGE_ROOTS.get(stage, (p.stem for p in source.glob('*.py'))))
    pending.append('execution')
    files = {}
    while pending:
        name = pending.pop()
        path = source / (name + '.py')
        if name in files or not path.is_file():
            continue
        data = path.read_bytes()
        files[name] = hashlib.sha256(data).hexdigest()
        for node in ast.walk(ast.parse(data)):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                # execution owns orchestration only: stage-specific imports below
                # are deliberately not transitive dependencies of every stage.
                if name == 'execution':
                    continue
                pending.extend([node.module.split('.')[0]] if node.module else [a.name for a in node.names])
    if 'zhilin' in files:
        for folder in ('assets', 'superdj/profiles'):
            for path in (source.parent / folder).glob('zhilin*.json'):
                files[str(path.relative_to(source.parent))] = sha256(path)
    return hashlib.sha256(json.dumps({'code': files, 'environment': environment()}, sort_keys=True).encode()).hexdigest()

_current = ContextVar('superdj_execution', default=None)
_model_device = ContextVar('superdj_model_device', default=None)


def current():
    return _current.get()


@contextmanager
def phase(name):
    session = current()
    if session is None:
        yield
    else:
        with session.phase(name):
            yield


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    """Hash file contents, not their names or playlist positions."""
    from dataclasses import asdict, is_dataclass
    import numpy as np
    if is_dataclass(value):
        return identity(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: identity(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [identity(v) for v in value]
    if isinstance(value, (str, Path)):
        try:
            path = Path(value)
            if path.is_file():
                return {'sha256': current().digest(path) if current() else sha256(path), 'bytes': path.stat().st_size}
        except (OSError, ValueError):
            pass
        return str(value)
    return value


def key_for(value):
    return hashlib.sha256(json.dumps(identity(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment(python=sys.executable):
    """Version fingerprint without starting another heavy Python process."""
    root = Path(python).expanduser().absolute().parent.parent
    sites = [root/'Lib/site-packages', *root.glob('lib/python*/site-packages')]
    prefixes = ('torch-', 'torchaudio-', 'torchvision-', 'numpy-', 'scipy-', 'librosa-',
                'transformers-', 'timm-', 'demucs-', 'beat_this-', 'soundfile-',
                'matplotlib-', 'pillow-', 'safetensors-', 'pyloudnorm-', 'pedalboard-', 'noisereduce-')
    result = {}
    for site in sites:
        for metadata in site.glob('*.dist-info/METADATA'):
            if metadata.parent.name.lower().startswith(prefixes):
                result[metadata.parent.name] = current().digest(metadata) if current() else sha256(metadata)
    return result


@contextmanager
def object_lock(path):
    """OS-released lock: an interrupted process cannot leave a permanent lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.05)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class Objects:
    """Immutable stage results: manifest committed only after all copies complete."""
    def __init__(self, root):
        self.root = Path(root)
        self.engines = {}

    def location(self, stage, key):
        return self.root / 'objects-v2' / self.engine_for(stage)[:20] / stage / key

    def engine_for(self, stage):
        if stage not in self.engines:
            self.engines[stage] = stage_fingerprint(stage)
        return self.engines[stage]

    def get(self, stage, key):
        directory = self.location(stage, key)
        try:
            manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf8'))
            if hashlib.sha256(json.dumps(manifest['result'], sort_keys=True).encode()).hexdigest() != manifest['result_sha256']:
                return None
            for name, digest in manifest['files'].items():
                if Path(name).name != name or sha256(directory / name) != digest:
                    return None
            def restore(v):
                if isinstance(v, dict):
                    if set(v) == {'$artifact'}:
                        if v['$artifact'] not in manifest['files']:
                            raise ValueError('Unlisted cache artifact')
                        return str(directory / v['$artifact'])
                    return {k: restore(x) for k, x in v.items()}
                if isinstance(v, list):
                    return [restore(x) for x in v]
                return v
            return restore(manifest['result'])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, stage, key, result, owned):
        directory = self.location(stage, key)
        directory.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f'pending-{key}-', dir=directory.parent))
        files, copied = {}, {}
        owned = Path(owned).resolve()
        def pack(v):
            if isinstance(v, dict):
                return {k: pack(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [pack(x) for x in v]
            if isinstance(v, (str, Path)):
                try:
                    p = Path(v).resolve()
                    if p.is_file() and p.is_relative_to(owned):
                        if str(p) not in copied:
                            name = f'{len(files):04d}{p.suffix}'
                            shutil.copyfile(p, scratch / name)
                            files[name] = sha256(scratch / name)
                            copied[str(p)] = name
                        return {'$artifact': copied[str(p)]}
                except (OSError, ValueError):
                    pass
                return str(v)
            return v
        try:
            packed = pack(result)
            checksum = hashlib.sha256(json.dumps(packed, sort_keys=True).encode()).hexdigest()
            (scratch / 'manifest.json').write_text(json.dumps({'files': files, 'result': packed, 'result_sha256': checksum}, allow_nan=False), encoding='utf8')
            # Concurrent identical jobs may win the commit. Never replace valid objects.
            if directory.exists():
                if self.get(stage, key) is not None:
                    return
                shutil.rmtree(directory)
            try:
                scratch.rename(directory)
            except FileExistsError:
                pass
        finally:
            if scratch.exists():
                shutil.rmtree(scratch)


class Worker:
    def __init__(self, python, device, session, family='model'):
        self.lock = threading.Lock()
        self.session = session
        self.family = family
        self.log = tempfile.TemporaryFile()
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONUTF8='1')
        self.process = subprocess.Popen([python, '-u', '-m', 'superdj.model_worker'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, encoding='utf8', env=env)
        self.device = device
        session.count(family + '_process_starts')

    def run(self, kind, job):
        with self.lock:
            self.process.stdin.write(json.dumps({'kind': kind, 'job': {**job, 'device': self.device}}) + '\n')
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Neural worker terminated; inspect the runtime and available device memory')
            reply = json.loads(line)
            if 'error' in reply:
                raise RuntimeError(reply['error'])
            self.session.count(self.family + '_tasks')
            with self.session.lock:
                self.session.metrics.setdefault('worker_models' if self.family == 'model' else 'worker_audio', {})[str(self.process.pid)] = {
                    'device': self.device, 'loaded': reply.get('loaded_models', {})}
            return reply['result']

    def close(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process.stdout.close()
        self.log.close()


class Session:
    def __init__(self, root, *, jobs=1, devices=(), scratch_root=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = max(1, min(int(jobs), 4))
        if isinstance(devices, str) or any(not isinstance(d, str) or not (d == 'cpu' or d == 'cuda' or
                (d.startswith('cuda:') and d[5:].isdigit())) for d in devices):
            raise ValueError('gpu_devices must be a list of available device names, for example cuda:0')
        self.devices = tuple(devices)
        self.objects = Objects(root)
        self.workers = {}
        self.thread_devices = {}
        self.digests = {}
        self.memo = {}
        self.decode_locks = {}
        self.lock = threading.Lock()
        self.audio_next = 0
        from .storage import private_path
        scratch_root = private_path(scratch_root or Path(tempfile.gettempdir())/'superdj-scratch')
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="run-", dir=scratch_root)
        self.work = Path(self.scratch.name)
        self.metrics = {'counts': {}, 'seconds': {}}
        self.started = time.perf_counter()

    def digest(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with self.lock:
            if key not in self.digests:
                self.digests[key] = sha256(path)
            return self.digests[key]

    def count(self, name):
        with self.lock:
            counts = self.metrics['counts']
            counts[name] = counts.get(name, 0) + 1

    def timed(self, name, seconds):
        with self.lock:
            times = self.metrics['seconds']
            times[name] = times.get(name, 0.) + seconds

    @contextmanager
    def phase(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timed(name, time.perf_counter()-started)

    def audio(self, job):
        # Never mix different upstream checkouts in one interpreter.
        checkout = job['checkout']
        with self.lock:
            slot = self.audio_next % self.jobs
            self.audio_next += 1
            key = ('audio', str(Path(checkout['path']).resolve()), checkout['revision'], slot)
            worker = self.workers.get(key)
            if worker is None:
                self.workers[key] = worker = Worker(sys.executable, 'cpu', self, family='audio')
        return worker.run('audio', job)

    def model(self, python, kind, job):
        if kind in {'cue', 'stem'}:
            from .audio import decoded_artifact
            job = {**job, 'decoded_input': decoded_artifact(job['source'],
                sample_rate=22050 if kind == 'cue' else 44100,
                channels=1 if kind == 'cue' else 2, ffmpeg=job.get('ffmpeg'))}
        # One process per interpreter/device, shared by cue and stem tasks.
        device = job.get('device', 'cpu')
        selected_device = _model_device.get()
        if selected_device is not None and device != 'cpu':
            device = selected_device
        elif self.devices and device != 'cpu':
            with self.lock:
                thread = threading.get_ident()
                if thread not in self.thread_devices:
                    self.thread_devices[thread] = self.devices[len(self.thread_devices) % len(self.devices)]
                device = self.thread_devices[thread]
        key = (str(python), device)
        with self.lock:
            worker = self.workers.get(key)
            if worker is None:
                # Worker updates counts; avoid acquiring the same lock recursively.
                self.workers[key] = worker = Worker(python, device, self)
        return worker.run(kind, job)

    def device_pair(self, function, values):
        """Two independent final decks, one worker on each configured GPU."""
        devices = tuple(dict.fromkeys('cuda:0' if d == 'cuda' else d
                                     for d in self.devices if d != 'cpu'))
        values = tuple(values)
        if len(values) != 2:
            raise ValueError('A device pair requires exactly two decks')
        if self.jobs < 2 or len(devices) < 2:
            return list(map(function, values))
        from .resources import available_memory
        memory = available_memory()
        # Extra decoder/model host buffers must fit alongside the caller.
        needed = sum(getattr(v[1], 'nbytes', 0) for v in values)*8 + 1024**3
        if memory is None or memory < needed:
            return list(map(function, values))
        def one(device, value):
            token = _model_device.set(device)
            try:
                return function(value)
            finally:
                _model_device.reset(token)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, one, device, value)
                       for device, value in zip(devices, values)]
            try:
                result = [f.result() for f in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        self.count('parallel_final_deck_pairs')
        return result

    def map(self, function, values):
        return list(self.imap(function, values))

    def imap(self, function, values):
        """Completion-driven bounded pipeline, delivered in input order."""
        if self.jobs == 1:
            yield from map(function, values)
            return
        pool = ThreadPoolExecutor(max_workers=self.jobs)
        values = iter(values)
        context = copy_context()
        condition = threading.Condition(threading.RLock())
        futures = {}
        state = dict(next=0, exhausted=False, stopped=False, failed=False)
        from .resources import available_memory
        memory = available_memory()
        # At most two waves including the item held by the consumer. Running
        # work never exceeds jobs; completed results cannot grow with set size.
        window = 2*self.jobs-1 if memory is not None and memory > (self.jobs+1)*512*1024**2 else self.jobs
        def completed(future):
            with condition:
                if not future.cancelled() and future.exception() is not None:
                    state['failed'] = True
                fill()
                condition.notify_all()
        def fill():
            while (not any(state[k] for k in ('exhausted','stopped','failed'))
                   and len(futures) < window
                   and sum(not f.done() for f in futures.values()) < self.jobs):
                try:
                    value = context.copy().run(next, values)
                except StopIteration:
                    state['exhausted'] = True
                    break
                except BaseException as error:
                    from concurrent.futures import Future
                    failed = Future();failed.set_exception(error)
                    futures[state['next']] = failed
                    state['next'] += 1
                    state['failed'] = True
                    break
                index = state['next'];state['next'] += 1
                future = pool.submit(context.copy().run, function, value)
                futures[index] = future
                future.add_done_callback(completed)
        try:
            with condition:
                fill()
            index = 0
            while True:
                with condition:
                    while index in futures and not futures[index].done():
                        condition.wait()
                    if index not in futures:
                        break
                    future = futures.pop(index);index += 1
                    fill()
                yield future.result()
        finally:
            with condition:
                state['stopped'] = True
                for future in futures.values():
                    future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

    def close(self):
        for worker in self.workers.values():
            worker.close()
        self.scratch.cleanup()
        self.metrics['wall_seconds'] = time.perf_counter() - self.started


@contextmanager
def execution(cache=None, *, jobs=1, devices=(), report=None, scratch_root=None):
    if current() is not None:
        yield current()
        return
    from .storage import private_path, write_json
    temporary = tempfile.TemporaryDirectory(prefix='superdj-') if cache is None else None
    root = private_path(cache if cache is not None else temporary.name)
    session = Session(root, jobs=jobs, devices=devices, scratch_root=scratch_root)
    session.persistent = cache is not None
    session.metrics['cache_policy'] = 'persistent' if session.persistent else 'temporary'
    # Worker creation counts under this lock as well.
    session.lock = threading.RLock()
    token = _current.set(session)
    try:
        yield session
        session.metrics['status'] = 'complete'
    except BaseException:
        session.metrics['status'] = 'failed'
        raise
    finally:
        try:
            session.close()
            if report:
                write_json(report, session.metrics)
        finally:
            _current.reset(token)
            if temporary:
                temporary.cleanup()


def cached(stage, dependencies, producer, owned):
    session = current()
    if session is None:
        return producer()
    if not session.persistent:
        memo_key = (stage, key_for(dependencies)) if stage == 'bass_profile' else None
        if memo_key in session.memo:
            session.count(stage + '_memory_hits')
            return json.loads(json.dumps(session.memo[memo_key]))
        started = time.perf_counter()
        try:
            result = producer()
            if memo_key is not None:
                session.memo[memo_key] = json.loads(json.dumps(result))
            return result
        finally:
            session.timed(stage, time.perf_counter() - started)
    key = key_for(dependencies)
    with object_lock(session.objects.location(stage, key).with_suffix('.lock')):
        parent = session.objects.location(stage, key).parent
        for pending in parent.glob(f'pending-{key}-*'):
            if pending.is_dir() and pending.resolve().is_relative_to(session.root.resolve()):
                shutil.rmtree(pending)
        value = session.objects.get(stage, key)
        if value is not None:
            session.count(stage + '_hits')
            return value
        session.count(stage + '_misses')
        started = time.perf_counter()
        try:
            value = producer()
        finally:
            session.timed(stage, time.perf_counter() - started)
        session.objects.put(stage, key, value, owned)
        return session.objects.get(stage, key) or value
