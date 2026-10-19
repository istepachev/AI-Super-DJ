"""Window arithmetic identity and fail-closed concurrent finalization."""
import ast
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import textwrap
import threading
import numpy as np
import pyloudnorm as pyln
import pytest
from superdj import lra_parallel
from superdj.finishing import finish_verified, _promote


def fixture_lra(data, rate):
    window_samples = int(3.0*rate)
    hop_samples = int(1.0*rate)
    if data.shape[0] <= window_samples:
        return None
    short_term = []
    meter = pyln.Meter(rate)
    for start in range(0, data.shape[0]-window_samples, hop_samples):
        st_lufs = meter.integrated_loudness(data[start:start+window_samples])
        if np.isfinite(st_lufs):
            short_term.append(st_lufs)
    if len(short_term) < 2:
        return None
    return float(np.percentile(short_term, 95)-np.percentile(short_term, 10))


@pytest.fixture
def resources(monkeypatch):
    monkeypatch.setattr('superdj.resources.available_memory', lambda: 8*1024**3)
    monkeypatch.setattr('os.cpu_count', lambda: 8)


@pytest.mark.parametrize('rate,channels', [(8000,1),(16000,2),(44100,2)])
def test_parallel_loudness_windows_and_final_percentiles_are_exact(resources,monkeypatch,rate,channels):
    audio = np.random.default_rng(71).normal(0,.04,(int(rate*15.3),channels))
    original_audio = audio.copy()
    expected = fixture_lra(audio,rate)
    tree=ast.parse(textwrap.dedent(inspect.getsource(fixture_lra)))
    monkeypatch.setattr(lra_parallel,'_AUDITED_AST',hashlib.sha256(ast.dump(tree.body[0]).encode()).hexdigest())
    backend=SimpleNamespace(_measure_lra=fixture_lra,pyln=pyln)
    for jobs in (2,4):
        with lra_parallel.parallel_lra(backend,jobs=jobs) as report:
            assert report['enabled'] and backend._measure_lra(audio,rate)==expected
            assert report['parallel_calls']==1 and report['windows']==13
        assert backend._measure_lra is fixture_lra
    np.testing.assert_array_equal(audio,original_audio)
    a=lra_parallel.window_values(audio,rate,3*rate,rate,pyln.Meter,1,{})
    b=lra_parallel.window_values(audio,rate,3*rate,rate,pyln.Meter,4,dict(parallel_calls=0,windows=0))
    np.testing.assert_array_equal(a,b)


def test_lra_rejects_changed_backend_and_restores_on_failure(resources):
    backend=SimpleNamespace(_measure_lra=fixture_lra,pyln=pyln)
    with lra_parallel.parallel_lra(backend) as report:
        assert not report['enabled'] and backend._measure_lra is fixture_lra
        assert 'signature changed' in report['fallback']
    with pytest.raises(RuntimeError):
        with lra_parallel.parallel_lra(backend):
            raise RuntimeError('master failed')
    assert backend._measure_lra is fixture_lra


def test_parallel_window_failure_retries_the_original_window_arithmetic(resources):
    main=threading.get_ident()
    class Meter:
        def __init__(self,rate):pass
        def integrated_loudness(self,chunk):
            if threading.get_ident()!=main:raise RuntimeError('worker unavailable')
            return float(np.sum(chunk))
    audio=np.arange(1000,dtype=float)[:,None]
    report=dict(parallel_calls=0,windows=0)
    expected=lra_parallel.window_values(audio,10,30,10,Meter,1,{})
    assert lra_parallel.window_values(audio,10,30,10,Meter,4,report)==expected
    assert report['fallback']=='RuntimeError'


def test_lra_low_or_unknown_memory_uses_serial_fallback(monkeypatch):
    for memory in (None,1):
        monkeypatch.setattr('superdj.resources.available_memory',lambda:memory)
        assert lra_parallel.admitted_jobs(4)==1


def test_missing_lra_backend_is_preserved(resources):
    backend=SimpleNamespace()
    with lra_parallel.parallel_lra(backend) as report:
        assert not report['enabled'] and 'unavailable' in report['fallback']
    assert not hasattr(backend,'_measure_lra')


def test_export_remains_private_until_rhythm_check_passes(tmp_path,resources,monkeypatch):
    from superdj import integrity,quality
    out=tmp_path/'out';finished=threading.Event()
    def encode(master,directory,name,**kw):
        path=Path(directory)/f'{name}.flac';path.write_bytes(b'checked audio')
        (Path(directory)/f'{name}_flac_qa.log').write_text('full decode passed')
        finished.set()
        return {'flac':{'path':str(path),'full_decode_ok':True}}
    def rhythm(*args):
        assert finished.wait(5)
        assert not (out/'set.flac').exists()
        return {'verified':True}
    monkeypatch.setattr(quality,'export_audio',encode)
    monkeypatch.setattr(integrity,'verify_master_rhythm',rhythm)
    checked,files=finish_verified('reference','master',{},out,'set',mp3=False)
    assert checked['verified'] and Path(files['flac']['path']).read_bytes()==b'checked audio'
    assert (out/'set_flac_qa.log').read_text()=='full decode passed'
    assert not list(out.glob('.superdj-export-*'))


@pytest.mark.parametrize('failure',['rhythm','encode','unverified'])
def test_either_failure_preserves_existing_output_and_removes_private_stage(tmp_path,resources,monkeypatch,failure):
    from superdj import integrity,quality
    out=tmp_path/'out';out.mkdir();(out/'set.flac').write_bytes(b'previous approved set')
    def rhythm(*args):
        if failure=='rhythm':raise ValueError('rhythm failed')
        if failure=='unverified':return {'verified':False}
        return {'verified':True}
    def encode(master,directory,name,**kw):
        path=Path(directory)/f'{name}.flac';path.write_bytes(b'new audio')
        if failure=='encode':raise ValueError('encoding failed')
        return {'flac':{'path':str(path)}}
    monkeypatch.setattr(quality,'export_audio',encode)
    monkeypatch.setattr(integrity,'verify_master_rhythm',rhythm)
    with pytest.raises(ValueError):finish_verified('a','b',{},out,'set',mp3=False,overwrite=True)
    assert (out/'set.flac').read_bytes()==b'previous approved set'
    assert not list(out.glob('.superdj-export-*'))


def test_promotion_failure_restores_already_replaced_audio(tmp_path,monkeypatch):
    from superdj import finishing
    stage=tmp_path/'stage';stage.mkdir();out=tmp_path/'out';out.mkdir()
    (out/'set.flac').write_bytes(b'old');(stage/'set.flac').write_bytes(b'new')
    real=finishing.os.link
    def fail(source,target):
        if Path(target).suffix=='.mp3':raise FileExistsError('destination conflict')
        return real(source,target)
    monkeypatch.setattr(finishing.os,'link',fail)
    real_rename=finishing.os.rename
    def fail_rename(source,target):
        if Path(target).suffix=='.mp3':raise FileExistsError('destination conflict')
        return real_rename(source,target)
    monkeypatch.setattr(finishing.os,'rename',fail_rename)
    with pytest.raises(FileExistsError):
        _promote([(stage/'set.flac',out/'set.flac',True),(stage/'missing.mp3',out/'set.mp3',True)],stage,True)
    assert (out/'set.flac').read_bytes()==b'old'


def test_unknown_memory_finalization_is_serial(tmp_path,monkeypatch):
    from superdj import integrity,quality
    monkeypatch.setattr('superdj.resources.available_memory',lambda:None)
    calls=[]
    monkeypatch.setattr(integrity,'verify_master_rhythm',lambda *a:(calls.append('rhythm') or {'verified':True}))
    def encode(master,directory,name,**kw):
        calls.append('encode');path=Path(directory)/f'{name}.flac';path.write_bytes(b'pcm')
        return {'flac':{'path':str(path)}}
    monkeypatch.setattr(quality,'export_audio',encode)
    finish_verified('a','b',{},tmp_path/'out','set',mp3=False)
    assert calls==['rhythm','encode']


@pytest.mark.parametrize('model_jobs,cpu_jobs',[(1,2),(2,1)])
def test_finalization_uses_cpu_admission_independently_of_model_limit(tmp_path,resources,monkeypatch,model_jobs,cpu_jobs):
    from superdj import integrity,quality
    from superdj.execution import execution
    encoded=threading.Event()
    def encode(master,directory,name,**kw):
        path=Path(directory)/f'{name}.flac';path.write_bytes(b'verified')
        encoded.set()
        return {'flac':{'path':str(path)}}
    def rhythm(*args):
        if cpu_jobs>=2:assert encoded.wait(5)
        else:assert not encoded.is_set()
        return {'verified':True}
    monkeypatch.setattr(quality,'export_audio',encode)
    monkeypatch.setattr(integrity,'verify_master_rhythm',rhythm)
    with execution(jobs=model_jobs,cpu_jobs=cpu_jobs)as session:
        checked,_=finish_verified('reference','master',{},tmp_path/'out','set',mp3=False)
        assert checked['verified']
        assert session.metrics['counts'].get('parallel_master_check_exports',0)==int(cpu_jobs>=2)


def test_serial_and_parallel_actual_exports_are_byte_identical(tmp_path,resources):
    import soundfile as sf
    from superdj.audio import ffmpeg_executable
    import os,shutil
    ffmpeg=ffmpeg_executable();ffprobe=os.getenv('SUPERDJ_FFPROBE') or shutil.which('ffprobe')
    if not ffprobe:pytest.skip('Actual FFprobe required')
    rate=16000;t=np.arange(rate*2)/rate
    audio=np.column_stack((.1*np.sin(2*np.pi*440*t),.08*np.sin(2*np.pi*700*t)))
    source=tmp_path/'master.wav';sf.write(source,audio,rate,subtype='PCM_24')
    rendered={'tracks':[],'transitions':[]}
    a,files_a=finish_verified(source,source,rendered,tmp_path/'serial','set',ffmpeg=ffmpeg,ffprobe=ffprobe,parallel=False)
    b,files_b=finish_verified(source,source,rendered,tmp_path/'parallel','set',ffmpeg=ffmpeg,ffprobe=ffprobe,parallel=True)
    assert a==b
    for ext in ('flac','mp3'):
        assert Path(files_a[ext]['path']).read_bytes()==Path(files_b[ext]['path']).read_bytes()
    with pytest.raises(FileExistsError):
        finish_verified(source,source,rendered,tmp_path/'parallel','set',ffmpeg=ffmpeg,ffprobe=ffprobe)
