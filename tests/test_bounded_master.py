"""Whole-signal references catch state resets and missing block-edge context."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d
from superdj import stream_master as sm
from superdj.resources import admit_master,memory_policy,preflight,GIB


@pytest.fixture
def audio(tmp_path):
    rng=np.random.default_rng(902)
    x=rng.normal(0,.18,(44100*4+19,2))
    # Transients around an intentionally awkward I/O boundary.
    x[[126,127,128,4092,4093,4094,-2],0]=[1.2,-.9,1.8,-2.,1.6,-1.4,1.9]
    path=tmp_path/'input.wav';sf.write(path,x,44100,subtype='DOUBLE')
    return path,x


@pytest.mark.parametrize('block',[127,4093,65536])
def test_global_loudness_and_highpass_keep_state_and_exact_window_sums(audio,tmp_path,block):
    path,x=audio;file=sm.FileAudio(path,block)
    actual=sm.loudness(file,pyln.Meter)
    expected=pyln.Meter(44100).integrated_loudness(x)
    assert actual==expected
    filtered=sm.highpass(file,tmp_path/'filtered.wav')
    b,a=signal.butter(2,5/(44100/2),btype='high')
    expected=np.column_stack([signal.lfilter(b,a,x[:,ch]) for ch in range(2)])
    assert np.array_equal(sf.read(filtered.path)[0],expected)


@pytest.mark.parametrize('block',[7,127,4093])
def test_true_peak_keeps_fir_context_across_every_block(audio,block):
    path,x=audio
    expected=max(float(np.max(np.abs(signal.resample_poly(x[:,ch],4,1)))) for ch in range(2))
    assert sm.true_peak(sm.FileAudio(path,block))==expected


@pytest.mark.parametrize('frames,ahead',[(47,221),(514,221),(8197,221),(8197,220)])
def test_limiter_release_and_future_minimum_do_not_restart_at_blocks(tmp_path,frames,ahead):
    x=np.random.default_rng(8).normal(0,1.2,(frames,2));path=tmp_path/'input.wav'
    sf.write(path,x,44100,subtype='DOUBLE');coefficient=np.exp(-1./(44100*.1));ceiling=10**(-1.5/20)
    gain=.91;peaks=np.max(abs(x*gain),axis=1)
    needed=np.where(peaks>ceiling,ceiling/np.maximum(peaks,1e-10),1.)
    envelope=[];state=1.
    for target in needed:
        state=target if target<state else coefficient*state+(1.-coefficient)*target
        envelope.append(state)
    shifted=(minimum_filter1d(np.r_[envelope,np.ones(ahead)],size=ahead+1,origin=-((ahead+1)//2))[:frames]
             if ahead<frames else np.full(frames,min(envelope)))
    expected=(x*gain)*shifted[:,None]
    result=sm.lookahead(sm.FileAudio(path,127),tmp_path/'out.wav',tmp_path/'env.wav',
        SimpleNamespace(soft_clip=lambda a,c:a),gain,-1.5,ahead*1000/44100,100.)
    assert np.array_equal(sf.read(result.path)[0],expected)
    assert not (tmp_path/'env.wav').exists()


@pytest.mark.parametrize('channels',[1,2])
def test_dither_draw_order_and_pcm24_quantization_are_exact(audio,tmp_path,channels):
    path,x=audio;x=x[:,:channels]
    sf.write(path,x,44100,subtype='DOUBLE')
    rng=np.random.default_rng(81)
    # Independent whole-array reference, including mono's one-column layout.
    noise=rng.uniform(-.5,.5,size=x.shape)+rng.uniform(-.5,.5,size=x.shape)
    noise*=1./2**23
    sf.write(tmp_path/'reference.wav',x+noise,44100,subtype='PCM_24')
    sm.write_dithered(sm.FileAudio(path,4093),tmp_path/'bounded.wav',seed=81)
    assert (tmp_path/'reference.wav').read_bytes()==(tmp_path/'bounded.wav').read_bytes()


def test_memory_budget_long_master_and_unavailable_gpu_adapt_without_changing_precision(tmp_path,monkeypatch):
    monkeypatch.delenv('SUPERDJ_MEMORY_BUDGET_MB',raising=False)
    policy=memory_policy(memory=2*GIB,total=4*GIB)
    assert policy['constrained'] and policy['buffer_budget_bytes']<=GIB
    report=admit_master(44100*8*3600,44100,memory=2*GIB)
    assert report['mode']=='bounded' and report['required_bytes_estimate']<768*1024**2
    hardware=dict(cpu_threads=4,available_memory_bytes=2*GIB,total_memory_bytes=4*GIB,
                  cuda_probe='measured',gpus=[])
    result=preflight(['one']*100,output=tmp_path,jobs=4,cpu_jobs=4,devices=['cuda:3'],machine=hardware,
                     probe=lambda p:(300.,44100),disk=lambda p:(tmp_path,'disk',1000*GIB))
    assert result['jobs']==result['cpu_jobs']==1
    assert result['devices']==['cpu'] and result['device_fallback']
    assert not result['preparation_overlap'] and not result['audio_precision_changed']


def test_invalid_or_silent_input_retains_existing_output_and_removes_temporary_pcm(tmp_path):
    path=tmp_path/'silent.wav';sf.write(path,np.zeros((44100,2)),44100,subtype='FLOAT')
    output=tmp_path/'existing.wav';output.write_bytes(b'previous master')
    for block in (0,-1,1.5):
        with pytest.raises(ValueError,match='positive integer'):
            sm.FileAudio(path,block)
    # Rejected backend must fail before replacing any saved audio.
    with pytest.raises((AttributeError,ValueError)):
        sm.master(path,output,SimpleNamespace(),scratch=tmp_path)
    assert output.read_bytes()==b'previous master'
    assert not list(tmp_path.glob('superdj-master-*'))


def test_constrained_worker_is_released_even_after_a_model_error(tmp_path,monkeypatch):
    from superdj.execution import execution
    policy=memory_policy(memory=2*GIB,total=4*GIB)
    monkeypatch.setattr('superdj.resources.memory_policy',lambda **kwargs:policy)
    closed=[]
    class Worker:
        def __init__(self,python,device,session,**kwargs):pass
        def run(self,kind,job):
            assert job['cue_batch_size']==1
            raise RuntimeError('model error')
        def close(self):closed.append(True)
    monkeypatch.setattr('superdj.execution.Worker',Worker)
    with execution(jobs=4,cpu_jobs=4,devices=['cpu']) as session:
        assert session.jobs==session.cpu_jobs==1 and not session.preparation_overlap
        with pytest.raises(RuntimeError,match='model error'):
            session.model('python','beat',{'device':'cpu'})
        assert not session.workers and session.model_running==0
    assert closed==[True]


def test_phase_boundary_does_not_close_busy_workers():
    from superdj.execution import execution
    closed=[]
    with execution() as session:
        worker=SimpleNamespace(close=lambda:closed.append(True))
        session.workers['idle']=worker
        session.model_running=1
        with pytest.raises(RuntimeError,match='active'):
            session.release_workers()
        assert not closed and session.workers['idle'] is worker
        session.model_running=0
        session.release_workers()
        assert closed==[True] and not session.workers


def test_file_lra_windows_keep_parallel_scalar_values_and_release_handles(tmp_path,monkeypatch):
    from superdj.lra_parallel import window_values
    monkeypatch.setattr('superdj.resources.available_memory',lambda:16*GIB)
    rate=8000
    x=np.random.default_rng(7).normal(0,.12,(rate*10+17,2))
    path=tmp_path/'lra.wav';sf.write(path,x,rate,subtype='DOUBLE')
    window,hop=rate*3,rate
    expected=[pyln.Meter(rate).integrated_loudness(x[start:start+window])
              for start in range(0,len(x)-window,hop)]
    report=dict(parallel_calls=0,windows=0)
    with sm.Readers() as readers:
        audio=sm.FileAudio(path,readers=readers)
        actual=window_values(audio,rate,window,hop,pyln.Meter,2,report)
        assert actual==expected and report['parallel_calls']==1
        assert 1<=len(readers.streams)<=2
    assert not readers.streams
    path.unlink()  # Windows must not retain a reader across cleanup.
