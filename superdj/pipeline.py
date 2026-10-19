"""Bounded CPU preparation while independent source analyses finish."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
import tempfile
import threading


class PreparationPipeline:
    def __init__(self, args, session):
        self.args, self.session = args, session
        self.pool = ThreadPoolExecutor(max_workers=session.cpu_jobs)
        self.capacity = threading.Semaphore(2*session.cpu_jobs)
        self.lock = threading.Lock()
        self.futures = {}
        self.failure = None
        Path(args.work).mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix='preparation-', dir=args.work))
        from .storage import read_json
        self.noise = read_json(args.noise_profiles) if args.noise_profiles else None

    def _prepare(self, record):
        from .adapters import prepare_tracks, measure_prepared
        # Planner inputs are immutable until it has selected the same order.
        local = dict(record)
        values = prepare_tracks([local], self.work, audamcp=self.args.audamcp,
            ffmpeg=self.args.ffmpeg, git=self.args.git, noise_profiles=self.noise,
            sample_rate=self.args.sample_rate, progress=lambda text: print(text, flush=True))
        measure_prepared([local], values, self.work, ffmpeg=self.args.ffmpeg, ffprobe=self.args.ffprobe)
        return values[record['id']]

    def submit(self, record):
        self.capacity.acquire()
        with self.lock:
            if self.failure:
                self.capacity.release()
                raise RuntimeError('Pipelined preparation failed') from self.failure
            if record['id'] in self.futures:
                self.capacity.release()
                raise ValueError('Duplicate track identity in preparation pipeline')
            try:
                future = self.pool.submit(copy_context().run, self._prepare, record)
            except BaseException:
                self.capacity.release()
                raise
            self.futures[record['id']] = future
        def finished(value):
            if not value.cancelled() and value.exception() is not None:
                with self.lock:
                    self.failure = value.exception()
            self.capacity.release()
        future.add_done_callback(finished)

    def results(self, records):
        return {record['id']: self.futures[record['id']].result() for record in records}

    def close(self):
        with self.lock:
            futures = list(self.futures.values())
        for future in futures:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
