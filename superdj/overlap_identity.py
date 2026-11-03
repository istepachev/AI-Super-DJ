"""Exact signal identities for overlap decisions, excluding container dates."""
from pathlib import Path
import hashlib
import zipfile


def dependencies(value,session,memo,field=None):
    """Keep all processing fields; only decode known WAV/NPZ file identities.

    Raw hashes still bind every memo entry and ordinary object integrity checks
    still run. This is not a perceptual hash, rounding or source-name alias.
    """
    if isinstance(value,dict):return {k:dependencies(v,session,memo,k)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [dependencies(v,session,memo,field)for v in value]
    if not isinstance(value,(str,Path)):return value
    try:
        path=Path(value)
        if path.suffix.lower()not in('.wav','.npz')or not path.is_file():return value
        if path.suffix.lower()=='.wav'and field not in('audio_path','aligned_vocal_path'):return value
        before=path.stat();raw=session.digest(path)
        if raw in memo:
            memo.move_to_end(raw);return memo[raw]
        if path.suffix.lower()=='.wav':
            import soundfile as sf
            with sf.SoundFile(path)as stream:
                dtype='float32'if stream.subtype=='FLOAT'else'float64'
                h=hashlib.sha256()
                for block in stream.blocks(blocksize=262144,dtype=dtype,always_2d=True):h.update(block.tobytes(order='C'))
                result={'exact_wave_samples':{'sha256':h.hexdigest(),'sample_rate':stream.samplerate,
                    'channels':stream.channels,'frames':stream.frames,'subtype':stream.subtype,
                    'format':stream.format,'decoded_dtype':dtype}}
        else:
            import numpy as np
            with zipfile.ZipFile(path)as archive:
                entries=archive.infolist()
                if (sum(i.file_size for i in entries)>32*1024**2
                        or len({i.filename for i in entries})!=len(entries)
                        or any(not i.filename.endswith('.npy')for i in entries)):return value
            with np.load(path,allow_pickle=False)as archive:
                arrays={}
                for name in sorted(archive.files):
                    a=archive[name]
                    if a.dtype.hasobject:return value
                    arrays[name]={'dtype':a.dtype.descr if a.dtype.fields else a.dtype.str,'shape':list(a.shape),
                                  'order':'F'if a.flags.f_contiguous and not a.flags.c_contiguous else'C',
                                  'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}
                result={'exact_numpy_arrays':arrays}
        after=path.stat()
        if(before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            return value
        memo[raw]=result
        while len(memo)>128:memo.popitem(last=False)
        return result
    except(OSError,ValueError,RuntimeError,zipfile.BadZipFile):
        # Unsupported files retain their ordinary full-content identity.
        return value
