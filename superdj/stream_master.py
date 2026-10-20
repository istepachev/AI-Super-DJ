"""Bounded float64 adaptation of the audited ordinary mastering preset.

No context resets at file blocks: IIR and limiter states cross every boundary.
Temporary DOUBLE PCM is disposable scratch, never an installed audio cache.
"""
from pathlib import Path
from contextlib import contextmanager
import ast
import copy
import hashlib
import inspect
import tempfile
import textwrap
import threading
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d


def _signature(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return hashlib.sha256(ast.dump(tree.body[0]).encode()).hexdigest()


class Readers:
    """One seekable read handle per file/thread, closed before scratch cleanup."""
    def __init__(self):
        self.streams={}
        self.lock=threading.Lock()

    def __enter__(self):return self

    def get(self,path):
        key=(threading.get_ident(),Path(path))
        with self.lock:
            if key not in self.streams:self.streams[key]=sf.SoundFile(path)
            return self.streams[key]

    def close(self,path=None):
        with self.lock:
            keys=[key for key in self.streams if path is None or key[1]==Path(path)]
            streams=[self.streams.pop(key) for key in keys]
        for stream in streams:stream.close()

    def __exit__(self,*args):self.close()


class FileAudio:
    """Read-only processing layout, with the upstream mono-to-stereo rule."""
    def __init__(self, path, block=65536, *, mono_stereo=False, readers=None):
        if type(block) is not int or block<=0:
            raise ValueError('Mastering block size must be a positive integer')
        self.path = Path(path)
        info = sf.info(self.path)
        self.frames, self.rate = info.frames, info.samplerate
        self.duplicate = mono_stereo and info.channels == 1
        self.channels = 2 if self.duplicate else info.channels
        self.shape = (self.frames, self.channels)
        self.ndim = 2
        self.block = block
        self.readers = readers

    def close(self):
        if self.readers:self.readers.close(self.path)

    def read(self, start, count):
        if self.readers:
            stream=self.readers.get(self.path)
            stream.seek(start)
            data = stream.read(count, dtype='float64', always_2d=True)
        else:
            with sf.SoundFile(self.path) as stream:
                stream.seek(start)
                data = stream.read(count, dtype='float64', always_2d=True)
        return np.repeat(data, 2, axis=1) if self.duplicate else data

    def __getitem__(self, item):
        if not isinstance(item, slice) or item.step not in (None, 1):
            raise ValueError('File audio supports contiguous frame slices only')
        start, stop, _ = item.indices(self.frames)
        return self.read(start, stop-start)

    def blocks(self):
        with sf.SoundFile(self.path) as stream:
            for block in stream.blocks(blocksize=self.block, dtype='float64', always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError('Invalid mastering PCM')
                yield np.repeat(block, 2, axis=1) if self.duplicate else block

    def energies(self, meter):
        """The exact per-window sums, after stateful upstream K weighting."""
        tg, step = meter.block_size, 1.-meter.overlap
        count = int(np.round(((self.frames/self.rate-tg)/(tg*step)))+1)
        z = np.zeros((self.channels, count))
        states = [[np.zeros(max(len(stage.a),len(stage.b))-1) for _ in range(self.channels)]
                  for stage in meter._filters.values()]
        pending = np.empty((0,self.channels)); origin = 0; done = 0; j = 0
        for block in self.blocks():
            # Match stage ordering and passband multiplication exactly.
            block = block.copy()
            for stage, state in zip(meter._filters.values(), states):
                for ch in range(self.channels):
                    filtered, state[ch] = signal.lfilter(stage.b,stage.a,block[:,ch],zi=state[ch])
                    block[:,ch] = stage.passband_gain * filtered
            pending = np.concatenate((pending,block)); done += len(block)
            while j < count:
                lo = int(tg*(j*step)*self.rate)
                hi = int(tg*(j*step+1)*self.rate)
                if hi > done and done != self.frames:
                    break
                for ch in range(self.channels):
                    z[ch,j] = (1./(tg*self.rate))*np.sum(np.square(pending[lo-origin:hi-origin,ch]))
                j += 1
            keep = int(tg*(j*step)*self.rate) if j<count else done
            pending = pending[keep-origin:].copy(); origin = keep
        if j != count:
            raise ValueError('Incomplete integrated loudness windows')
        return z


_meter_method = None


def loudness(audio, meter_type):
    """Keep the installed meter's gating/reductions; replace only residency."""
    global _meter_method
    original = meter_type.integrated_loudness
    # Deliberately fail closed on unsupported loudness/filter implementations.
    if _signature(original) != _AUDITED_METER:
        raise ValueError('Unsupported loudness meter for bounded mastering')
    from pyloudnorm.iirfilter import IIRfilter
    if _signature(IIRfilter.apply_filter) != _AUDITED_IIR:
        raise ValueError('Unsupported loudness weighting filter')
    if audio.frames < int(audio.rate*.4):
        raise ValueError('Audio must exceed the loudness measurement block')
    if _meter_method is None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        body = tree.body[0].body
        # Retain all operations from G onwards, except allocating/recalculating z.
        begin = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='G' for t in n.targets))
        kept = []
        for node in body[begin:]:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='z' for t in node.targets):
                continue
            if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='i':
                continue
            kept.append(node)
        tree.body[0].body = ast.parse('numChannels=data.channels\nnumSamples=data.frames\nz=data.energies(self)').body+kept
        namespace = dict(original.__globals__)
        exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
        _meter_method = namespace[original.__name__]
    return _meter_method(meter_type(audio.rate),audio)


def true_peak(audio):
    """Identical 4x polyphase FIR, with real context and global zero ends."""
    peak = 0.
    # The default 4x resample_poly FIR has support of ten input frames.
    halo = 16
    for start in range(0,audio.frames,audio.block):
        count = min(audio.block,audio.frames-start)
        first, last = max(0,start-halo), min(audio.frames,start+count+halo)
        block = audio.read(first,last-first)
        for ch in range(audio.channels):
            up = signal.resample_poly(block[:,ch],up=4,down=1)
            useful = up[(start-first)*4:(start-first+count)*4]
            peak = max(peak,float(np.max(np.abs(useful))))
    return peak


@contextmanager
def double_writer(path,audio,channels=None):
    channels = audio.channels if channels is None else channels
    fmt = 'RF64' if audio.frames*channels*8+4096 >= 2**32 else 'WAV'
    with sf.SoundFile(path,'w',samplerate=audio.rate,channels=channels,subtype='DOUBLE',format=fmt) as stream:
        yield stream


def highpass(audio,path,cutoff=5):
    b,a = signal.butter(2,cutoff/(audio.rate/2),btype='high')
    if not np.all(np.abs(np.roots(a))<1.):
        raise ValueError('Unstable mastering highpass')
    states = [np.zeros(max(len(a),len(b))-1) for _ in range(audio.channels)]
    with double_writer(path,audio) as output:
        for block in audio.blocks():
            result = np.zeros_like(block)
            for ch in range(audio.channels):
                result[:,ch],states[ch] = signal.lfilter(b,a,block[:,ch],zi=states[ch])
            output.write(result)
    return FileAudio(path,audio.block,readers=audio.readers)


def _release_state(values,coefficient,state):
    result = np.ones(len(values),dtype=np.float64)
    for i in range(len(values)):
        target = values[i]
        if target < state:
            state = target
        else:
            state = coefficient*state+(1.-coefficient)*target
        result[i] = state
    return result,state


_release = None


def release_state(values,coefficient,state):
    global _release
    if _release is None:
        try:
            from numba import njit
            _release = njit(cache=False,fastmath=False)(_release_state)
            probe = np.array([1.,.2,1.,.1,1.])
            actual = _release(probe,.999,.7)
            expected = _release_state(probe,.999,.7)
            if not np.array_equal(actual[0],expected[0]) or actual[1]!=expected[1]:
                raise ValueError('Limiter compilation changed float64 arithmetic')
        except Exception:
            _release = _release_state
    return _release(values,coefficient,state)


def lookahead(audio,path,envelope_path,module,gain,ceiling_db,ms,release_ms):
    ceiling = 10**(ceiling_db/20)
    ahead = int(audio.rate*ms/1000.)
    coefficient = np.exp(-1./(audio.rate*release_ms/1000.))
    state = 1.; lowest = 1.
    with double_writer(envelope_path,audio,1) as output:
        for block in audio.blocks():
            peak = np.max(np.abs(block*gain),axis=1)
            needed = np.where(peak>ceiling,ceiling/np.maximum(peak,1e-10),1.)
            smoothed,state = release_state(needed,coefficient,state)
            lowest = min(lowest,float(np.min(smoothed)))
            output.write(smoothed)
    with sf.SoundFile(envelope_path) as envelope, double_writer(path,audio) as output:
        start = 0
        for block in audio.blocks():
            n = len(block)
            if ahead<audio.frames:
                envelope.seek(start)
                values = envelope.read(n+ahead,dtype='float64')
                if len(values)<n+ahead:
                    values = np.concatenate((values,np.ones(n+ahead-len(values))))
                window = ahead+1
                shifted = minimum_filter1d(values,size=window,origin=-(window//2))[:n]
            else:
                shifted = np.full(n,lowest,dtype=np.float64)
            output.write(module.soft_clip((block*gain)*shifted[:,None],ceiling))
            start += n
    Path(envelope_path).unlink()
    return FileAudio(path,audio.block,readers=audio.readers)


def reactive(audio,path,module,ceiling_db,*,gain=1.,mono=False):
    # A first pass measures the gained signal, preserving multiplication order.
    class Gained:
        frames,channels,block = audio.frames,audio.channels,audio.block
        def read(self,start,count): return audio.read(start,count)*gain
    measured = true_peak(Gained())
    ceiling = 10**(ceiling_db/20)
    reduction = ceiling/measured if measured>ceiling else None
    channels = 1 if mono else audio.channels
    with double_writer(path,audio,channels) as output:
        for block in audio.blocks():
            # Upstream converts to mono before its final reactive guard.
            block = block[:,:1] if mono else block
            block = block*gain
            if reduction is not None:
                block = block*reduction
            output.write(module.soft_clip(block,ceiling))
    return FileAudio(path,audio.block,readers=audio.readers)


def write_dithered(audio,path,*,seed=None):
    """Same two whole-array uniform draws in bounded storage, including mono."""
    first = np.random.default_rng(seed)
    second = copy.deepcopy(first)
    remaining = audio.frames*audio.channels
    while remaining:
        n = min(remaining,audio.block*audio.channels)
        second.uniform(-.5,.5,size=n)
        remaining -= n
    fmt = 'RF64' if audio.frames*audio.channels*3+4096>=2**32 else 'WAV'
    with sf.SoundFile(path,'w',samplerate=audio.rate,channels=audio.channels,subtype='PCM_24',format=fmt) as output:
        for block in audio.blocks():
            noise = first.uniform(-.5,.5,size=block.shape)+second.uniform(-.5,.5,size=block.shape)
            noise *= 1./2**23
            output.write(block+noise)


def master(input_path,output_path,module,*,target_lufs=-14.,ceiling_db=-1.5,
           limiter_lookahead_ms=5.,block=65536,scratch=None,seed=None,lra_jobs=1):
    """Ordinary adapter preset only; arbitrary upstream genre presets stay upstream."""
    from .lra_parallel import parallel_lra
    _validate_backend(module)
    source = FileAudio(input_path,block,mono_stereo=True)
    def input_identity():
        value=source.path.stat()
        return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns
    original_identity=input_identity()
    if source.channels!=2 or source.frames<source.rate*.4:
        raise ValueError('Bounded ordinary master requires mono/stereo, at least 400 ms')
    if not np.isfinite([target_lufs,ceiling_db,limiter_lookahead_ms]).all() or not 0<=limiter_lookahead_ms<=1000:
        raise ValueError('Invalid bounded mastering settings')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='superdj-master-',dir=scratch) as temporary, Readers() as readers:
        source.readers=readers
        work = Path(temporary)
        original = loudness(source,module.pyln.Meter)
        filtered = highpass(source,work/'filtered.wav')
        current = loudness(filtered,module.pyln.Meter)
        if not np.isfinite(current):
            raise ValueError('Mastering input is silent or near-silent')
        with parallel_lra(module,jobs=lra_jobs) as lra_report:
            lra = module._measure_lra(filtered,filtered.rate)
        gain_db = target_lufs-current
        gain = 10**(gain_db/20)
        if int(source.rate*limiter_lookahead_ms/1000.)>0:
            limited = lookahead(filtered,work/'limited.wav',work/'envelope.wav',module,
                                gain,ceiling_db,limiter_lookahead_ms,100.)
        else:
            limited = reactive(filtered,work/'limited.wav',module,ceiling_db,gain=gain)
        filtered.close();filtered.path.unlink()
        final_lufs = loudness(limited,module.pyln.Meter)
        guarded = reactive(limited,work/'guarded.wav',module,ceiling_db,
                           mono=sf.info(input_path).channels==1)
        limited.close();limited.path.unlink()
        peak = true_peak(guarded)
        # Stage on the destination volume for atomic replacement after full write.
        with tempfile.TemporaryDirectory(prefix='.superdj-master-',dir=output_path.parent) as stage:
            staged = Path(stage)/'master.wav'
            write_dithered(guarded,staged,seed=seed)
            if (sf.info(staged).frames,sf.info(staged).channels)!=(source.frames,guarded.channels):
                raise ValueError('Incomplete bounded master')
            if input_identity()!=original_identity:
                raise ValueError('Mastering input changed between bounded passes; previous output retained')
            staged.replace(output_path)
    result = dict(original_lufs=original,final_lufs=final_lufs,gain_applied=gain_db,
                  final_peak=20*np.log10(peak) if peak>0 else float('-inf'))
    if lra is not None: result['lra']=lra
    return result,dict(mode='bounded',block_frames=block,precision='float64',
                       filter_and_limiter_state_preserved=True,lra_parallelism=lra_report,
                       temporary_pcm_removed=True)


# Audited upstream mastering revision and installed pyloudnorm 0.2.0.
_AUDITED_METER = 'c5b32d3293579e8344ad2f355322b75c97b67fca50344ab702ee22854ffa3851'
_AUDITED_IIR = '82c8b0c66a96e3b52fbafe2c988a48d161cf8b9eb4d1bb22a4dbc85ad8123e52'
_AUDITED_BACKEND = {'master_track': 'aab63676ab18244a8fee8e7bebbbf34efee8e843d1e16aea1d92ec6d40983cb5', 'apply_highpass': '8a0c3d7f1a024cc4b9e85a6b387bfdcda4366ae39bbc03392d509e33946fa99d', 'soft_clip': '42ced743baa0e58db72bf11f672dff1616e2e6413adb235b253bb3f4af95ba6a', 'measure_true_peak': 'd749af4b2d1f020fe69157ae0cc00e7aab35ae8219637e8a7a8fd601b7519f49', 'limit_peaks': '1e0f2bf7a5167b189822fa4557d4aac7ad4e28410f47efb51be7d918c7510c08', 'limit_peaks_lookahead': '5751a2a5016b1bc6a31fd84078c60fcb370560242f1f02a2671b9751bb999a1b', '_measure_lra': '265f23afdd8607e1170af97b30b62f50831d6537480cae389ad0a94e475a9d10', 'apply_tpdf_dither': '444f51c316b71d8971fe88e1c50ce2760532fbaa6bfe398496ce5f2db0a86f18'}


def _validate_backend(module):
    for name,expected in _AUDITED_BACKEND.items():
        if _signature(getattr(module,name))!=expected:
            raise ValueError('Unsupported bounded mastering backend function: '+name)
    if not _AUDITED_BACKEND:
        raise ValueError('Bounded mastering backend has not been audited')
