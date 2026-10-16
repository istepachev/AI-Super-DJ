"""Adversarial delivery, timing and bounded scheduling regressions."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import threading
import time
import sys

import numpy as np
import pytest
import soundfile as sf

from superdj.execution import execution, current
from superdj.integrity import WitnessWriter, verify_exported_pcm, verify_master_rhythm
from superdj.delivery import copy_verified, publish_pair


def test_persistent_cache_uses_separate_temporary_scratch_and_cleans_it(tmp_path):
    with execution(tmp_path/'cache',scratch_root=tmp_path/'local-ssd') as session:
        work=session.work
        assert work.is_relative_to(tmp_path/'local-ssd')
        assert not work.is_relative_to(session.root)
        (work/'private.wav').write_bytes(b'PCM')
    assert not work.exists() and (tmp_path/'cache').exists()


def test_audio_worker_reuses_process_after_a_rejected_job(tmp_path):
    with execution(jobs=1) as session:
        job={'checkout':{'path':str(tmp_path),'revision':'fixed'},'action':'invalid'}
        for _ in range(2):
            with pytest.raises(RuntimeError,match='Unknown adapter action'):
                session.audio(job)
        processes=[w.process for w in session.workers.values()]
        assert session.metrics['counts']['audio_process_starts']==1
        assert not session.metrics['counts'].get('model_process_starts')
    assert all(p.poll() is not None for p in processes)


def test_parallel_loudness_checks_preserve_records_and_bound_concurrency(tmp_path,monkeypatch):
    from superdj import adapters
    active=maximum=0
    lock=threading.Lock()
    records=[{'id':str(i)} for i in range(8)]
    prepared={r['id']:{'path':str(tmp_path/(r['id']+'.wav'))} for r in records}
    def measure(path,**kwargs):
        nonlocal active,maximum
        assert current() is not None
        with lock:
            active+=1;maximum=max(maximum,active)
        time.sleep(.01)
        with lock:active-=1
        return {'integrated_lufs':-18.,'full_decode_ok':True}
    monkeypatch.setattr('superdj.quality.measure_audio',measure)
    monkeypatch.setattr('superdj.audio.ffmpeg_executable',lambda value:'decoder')
    with execution(jobs=2):
        adapters.measure_prepared(records,prepared,tmp_path)
    assert maximum==2
    assert all(r['loudness_lufs']==-18. for r in records)
    assert list(prepared)==[str(i) for i in range(8)]


def test_preparation_pipeline_keeps_planner_inputs_unchanged(tmp_path,monkeypatch):
    from superdj.pipeline import PreparationPipeline
    records=[{'id':str(i),'energy':.5} for i in range(8)]
    args=SimpleNamespace(work=str(tmp_path),noise_profiles=None)
    active=maximum=0;lock=threading.Lock()
    def prepare(self,record):
        nonlocal active,maximum
        assert current() is self.session
        with lock:active+=1;maximum=max(maximum,active)
        time.sleep(.01)
        with lock:active-=1
        return {'loudness_lufs':-18.,'id':record['id']}
    monkeypatch.setattr(PreparationPipeline,'_prepare',prepare)
    with execution(jobs=2) as session:
        pipeline=PreparationPipeline(args,session)
        try:
            for record in records:pipeline.submit(record)
            results=pipeline.results(records)
            assert list(results)==[str(i) for i in range(8)]
            assert records==[{'id':str(i),'energy':.5} for i in range(8)]
        finally:pipeline.close()
    assert maximum==2


def test_pipeline_failure_is_propagated_before_rendering(tmp_path,monkeypatch):
    from superdj.pipeline import PreparationPipeline
    def fail(self,record):raise ValueError('processor failure')
    monkeypatch.setattr(PreparationPipeline,'_prepare',fail)
    with execution(jobs=2) as session:
        pipeline=PreparationPipeline(SimpleNamespace(work=str(tmp_path),noise_profiles=None),session)
        try:
            pipeline.submit({'id':'a'})
            with pytest.raises(ValueError,match='processor failure'):
                pipeline.results([{'id':'a'}])
        finally:pipeline.close()


def test_verified_copy_preserves_source_and_existing_destinations(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'music'*100000)
    destination=tmp_path/'destination'
    digest=copy_verified(source,destination)
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):copy_verified(source,destination)
    assert destination.read_bytes()==source.read_bytes()
    with pytest.raises(ValueError):copy_verified(source,source)
    with pytest.raises(RuntimeError,match='does not match'):
        copy_verified(source,tmp_path/'bad',expected='0'*64)
    assert not (tmp_path/'bad').exists()


def test_delivery_copy_corruption_never_publishes_audio(tmp_path,monkeypatch):
    source=tmp_path/'set.flac';source.write_bytes(b'original music')
    cue=tmp_path/'set.cue';cue.write_text('FILE "set.flac" WAVE',encoding='utf8')
    monkeypatch.setattr('superdj.delivery.content_hash',lambda path:'0'*64)
    with pytest.raises(RuntimeError,match='does not match'):
        publish_pair(source,cue,tmp_path/'ready')
    assert not list((tmp_path/'ready').iterdir())
    assert source.read_bytes()==b'original music'


@pytest.mark.parametrize('damage',['sample','shift','channels','length'])
def test_flac_pcm_guard_rejects_even_one_changed_sample(tmp_path,damage):
    rate=16000
    rng=np.random.default_rng(31)
    audio=rng.normal(0,.08,(rate,2))
    master=tmp_path/'master.wav';sf.write(master,audio,rate,subtype='PCM_24')
    audio=sf.read(master,dtype='float64',always_2d=True)[0]
    encoded=tmp_path/'set.flac';sf.write(encoded,audio,rate,subtype='PCM_24')
    assert verify_exported_pcm(master,encoded)['exact_pcm']
    if damage=='sample':audio[8000,0]+=1/2**23
    elif damage=='shift':audio=np.roll(audio,1,axis=0)
    elif damage=='channels':audio=audio[:,::-1]
    else:audio=audio[:-1]
    sf.write(encoded,audio,rate,subtype='PCM_24')
    with pytest.raises(ValueError,match='changed|differs'):
        verify_exported_pcm(master,encoded)


def test_assembly_check_detects_a_corrupted_overlap_and_wrong_position(tmp_path):
    from superdj.render import _measure,RenderError
    path=tmp_path/'mix.wav'
    audio=np.random.default_rng(19).normal(0,.05,(1000,2)).astype('float32')
    with sf.SoundFile(path,'w',samplerate=16000,channels=2,subtype='FLOAT') as file:
        witness=WitnessWriter(file);witness.write(audio[:500]);witness.write(audio[500:])
    value={'pcm_sha256':hashlib.sha256(audio[100:200].tobytes()).hexdigest()}
    _measure(path,witness=witness,regions=[({'set_start_frame':100,'set_end_frame':200},value)])
    with pytest.raises(RenderError,match='overlap differs'):
        _measure(path,regions=[({'set_start_frame':101,'set_end_frame':201},value)])
    with sf.SoundFile(path,'r+') as file:
        file.seek(150);file.write(np.zeros((1,2),dtype='float32'))
    with pytest.raises(RenderError,match='differs'):_measure(path,witness=witness)


@pytest.mark.parametrize('damage',[False,True])
def test_master_rhythm_check_accepts_gain_and_rejects_five_ms_shift(tmp_path,damage):
    from test_sync import RenderedSyncTests,FFMPEG,RUBBERBAND
    from superdj.render import render_plan
    from superdj.planner import plan_tracks
    if not FFMPEG or not RUBBERBAND:pytest.skip('Actual audio backends required')
    records=RenderedSyncTests().fixtures(tmp_path,[120,122])
    rendered=render_plan(plan_tracks(records,mode='fixed'),records,{},tmp_path/'premix.wav',tmp_path/'work',
        ffmpeg=FFMPEG,rubberband=RUBBERBAND,sample_rate=16000,tempo_conflict_policy='sync',sync_transition_bars=4)
    audio,rate=sf.read(tmp_path/'premix.wav',always_2d=True)
    if damage:audio=np.roll(audio,round(.005*rate),axis=0)
    sf.write(tmp_path/'master.wav',audio*.75,rate,subtype='PCM_24')
    if damage:
        with pytest.raises(ValueError,match='clock tolerance|ambiguous'):
            verify_master_rhythm(tmp_path/'premix.wav',tmp_path/'master.wav',rendered)
    else:
        report=verify_master_rhythm(tmp_path/'premix.wav',tmp_path/'master.wav',rendered)
        assert report['verified'] and report['transitions'][0]['verified_events']>=8
