"""Quality guards for parallel decks, exact shared decode and scoped caches."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from superdj.audio import decoded_artifact, read_decoded, decode_mono, decode_wave_file, ffmpeg_executable
from superdj.execution import execution, stage_fingerprint


def _long_records(directory, bpms):
    rate, duration = 16000, 240
    records = []
    for index, bpm in enumerate(bpms):
        beats = np.arange(.25, duration-.05, 60/bpm)
        audio = np.zeros(rate*duration,dtype=np.float32)
        width = 128
        pulse = .25*np.exp(-.5*(np.arange(-width,width+1)/(rate*.0015))**2)
        for beat in beats:
            frame=round(beat*rate)
            audio[frame-width:frame+width+1]+=pulse
        audio += (.003*np.sin(2*np.pi*(220+11*index)*np.arange(len(audio))/rate)).astype(np.float32)
        path=directory/f'{index}.wav'
        sf.write(path,np.column_stack((audio,audio)),rate,subtype='FLOAT')
        records.append(dict(id=str(index),path=str(path),duration=duration,bpm=bpm,bpm_confidence=1.,
            energy=.5,key_camelot='8A',key_confidence=1.,beats=beats.tolist(),downbeats=beats[::4].tolist(),
            grid_status='reviewed',grid_confidence=1.,grid=dict(meter=4,human_reviewed=True)))
    return records


def test_stage_dependencies_include_transitive_imports_and_unknown_fallback(tmp_path):
    (tmp_path/'audio.py').write_text('from .helper import decode\n', encoding='utf8')
    (tmp_path/'helper.py').write_text('decode = 1\n', encoding='utf8')
    (tmp_path/'planner.py').write_text('order = 1\n', encoding='utf8')
    original = stage_fingerprint('decode', tmp_path)
    unknown = stage_fingerprint('future_stage', tmp_path)
    (tmp_path/'planner.py').write_text('order = 2\n', encoding='utf8')
    assert stage_fingerprint('decode', tmp_path) == original
    assert stage_fingerprint('future_stage', tmp_path) != unknown
    (tmp_path/'helper.py').write_text('decode = 2\n', encoding='utf8')
    assert stage_fingerprint('decode', tmp_path) != original


def test_streaming_pipeline_is_bounded_and_cancels_pending_work(tmp_path):
    started = []
    with execution(jobs=2) as session:
        def one(value):
            started.append(value)
            time.sleep(.02)
            return value
        results = session.imap(one, range(100))
        assert next(results) == 0
        results.close()
        assert len(started) <= 3
        assert not session.workers


@pytest.mark.parametrize('extension,codec', [('wav','pcm_f32le'),('mp3','libmp3lame'),('m4a','aac'),('flac','flac')])
def test_shared_decode_exact_across_formats_processes_and_channel_rates(tmp_path, extension, codec):
    decoder = ffmpeg_executable()
    rate = 48000
    t = np.arange(rate*3)/rate
    source = tmp_path/'source.wav'
    sf.write(source, np.column_stack((.2*np.sin(1000*t), .15*np.sin(1400*t))), rate, subtype='FLOAT')
    encoded = tmp_path/f'input.{extension}'
    subprocess.run([decoder, '-v','error','-i',str(source),'-c:a',codec,str(encoded)], check=True, capture_output=True)
    expected = decode_mono(encoded, ffmpeg=decoder)
    with execution(jobs=2) as session:
        descriptors = session.map(lambda _: decoded_artifact(encoded, ffmpeg=decoder), range(4))
        assert len({item['path'] for item in descriptors}) == 1
        np.testing.assert_array_equal(read_decoded(descriptors[0], sample_rate=22050, channels=1), expected)
        job = tmp_path/'descriptor.json'
        job.write_text(json.dumps(descriptors[0]), encoding='utf8')
        script = "import json,sys,numpy as np;from superdj.audio import read_decoded;np.save(sys.argv[2],read_decoded(json.load(open(sys.argv[1],encoding='utf8')),sample_rate=22050,channels=1))"
        subprocess.run([sys.executable, '-c',script,str(job),str(tmp_path/'worker.npy')],check=True,
                       env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])), capture_output=True)
        np.testing.assert_array_equal(np.load(tmp_path/'worker.npy'), expected)
        stereo = decoded_artifact(encoded, sample_rate=44100, channels=2, ffmpeg=decoder)
        assert stereo['path'] != descriptors[0]['path']
        wav = tmp_path/'shared.wav'
        decode_wave_file(encoded, wav, sample_rate=44100, ffmpeg=decoder)
        reference = tmp_path/'reference.wav'
        subprocess.run([decoder,'-v','error','-i',str(encoded),'-ac','2','-ar','44100','-c:a','pcm_f32le',str(reference)],check=True,capture_output=True)
        np.testing.assert_array_equal(sf.read(wav,dtype='float32')[0],sf.read(reference,dtype='float32')[0])
        assert session.metrics['counts']['source_decodes'] == 2
        assert session.metrics['counts']['decode_reuses'] == 4
        shared = Path(stereo['path'])
    assert not shared.exists()


def test_shared_decode_corruption_recomputed_and_worker_rejects_wrong_format(tmp_path):
    decoder = ffmpeg_executable()
    source = tmp_path/'source.wav'
    sf.write(source, np.zeros(44100), 44100)
    with execution() as session:
        first = decoded_artifact(source, ffmpeg=decoder)
        with pytest.raises(ValueError, match='format'):
            read_decoded(first, sample_rate=44100, channels=2)
        Path(first['path']).write_bytes(b'corrupt')
        with pytest.raises(ValueError, match='integrity'):
            read_decoded(first, sample_rate=22050, channels=1)
        second = decoded_artifact(source, ffmpeg=decoder)
        assert np.isfinite(read_decoded(second, sample_rate=22050, channels=1)).all()
        assert session.metrics['counts']['source_decodes'] == 2


def test_zhilin_curve_change_invalidates_only_overlap_objects(tmp_path, monkeypatch):
    from test_sync import RenderedSyncTests
    from superdj.planner import plan_tracks
    from superdj.render import render_plan
    from superdj.zhilin import reference_profile
    decoder = ffmpeg_executable()
    rubber = os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not rubber:
        pytest.skip('Actual Rubber Band required')
    profile = reference_profile(runtime={})
    path = tmp_path/'curves.json'
    path.write_text(json.dumps(profile),encoding='utf8')
    monkeypatch.setattr('superdj.cues.runtime_config',lambda:{'zhilin_profile':str(path)})
    records = RenderedSyncTests().fixtures(tmp_path,[120,122])
    plan = plan_tracks(records,mode='fixed')
    plan['style'] = 'zhilin'
    with execution(tmp_path/'cache') as session:
        render_plan(plan,records,{},tmp_path/'first.wav',session.work,ffmpeg=decoder,rubberband=rubber,sample_rate=16000,tempo_conflict_policy='sync')
        profile['mid_in'] = [float(v)**1.5 for v in profile['mid_in']]
        path.write_text(json.dumps(profile),encoding='utf8')
        render_plan(plan,records,{},tmp_path/'second.wav',session.work,ffmpeg=decoder,rubberband=rubber,sample_rate=16000,tempo_conflict_policy='sync')
        assert session.metrics['counts']['clip_hits'] == 2
        assert session.metrics['counts']['overlap_misses'] == 2
        assert not session.metrics['counts'].get('overlap_hits')


def test_unconfirmed_preview_restores_full_audio_and_legacy_correction(tmp_path, monkeypatch):
    from copy import deepcopy
    from superdj.planner import plan_tracks
    from superdj.render import render_plan
    decoder = ffmpeg_executable()
    rubber = os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not rubber:
        pytest.skip('Actual Rubber Band required')
    records = _long_records(tmp_path,[174,128,174])
    plan = plan_tracks(records,mode='fixed')
    settings = dict(ffmpeg=decoder,rubberband=rubber,sample_rate=16000,
                    tempo_conflict_policy='sync',transition_bars=16,sync_transition_bars=16)
    baseline=render_plan(plan,records,{},tmp_path/'baseline.wav',tmp_path/'baseline',**settings)
    if not any(len((track.get('warp') or {}).get('post_render_transient_audit',{}).get('iterations',[]))>2
               for track in baseline['tracks']):
        pytest.skip('This backend version aligned the fixture without repeated correction')
    called=[]
    def unconfirmed(source,warp,*args,**kwargs):
        called.append(1)
        return deepcopy(warp)
    monkeypatch.setattr('superdj.acoustic_preview.screen_correction',unconfirmed)
    with execution(jobs=2) as session:
        render_plan(plan,records,{},tmp_path/'screened.wav',session.work,**settings)
        assert called and session.metrics['counts']['acoustic_preview_fallbacks'] >= 1
        assert not session.metrics['counts'].get('acoustic_preview_confirmed')
    np.testing.assert_array_equal(sf.read(tmp_path/'baseline.wav')[0],sf.read(tmp_path/'screened.wav')[0])


@pytest.mark.parametrize('jobs',[1,2,4])
def test_parallel_render_preserves_deck_pcm_transitions_and_final_sync(tmp_path, jobs):
    from test_sync import RenderedSyncTests
    from superdj.planner import plan_tracks
    from superdj.render import render_plan
    decoder = ffmpeg_executable()
    rubber = os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not rubber:
        pytest.skip('Actual Rubber Band required')
    records = RenderedSyncTests().fixtures(tmp_path, [120,122,124])
    plan = plan_tracks(records, mode='fixed')
    settings = dict(ffmpeg=decoder, rubberband=rubber, sample_rate=16000, tempo_conflict_policy='sync')
    baseline = render_plan(plan, records, {}, tmp_path/'baseline.wav', tmp_path/'baseline', **settings)
    with execution(jobs=jobs) as session:
        result = render_plan(plan, records, {}, tmp_path/'parallel.wav', session.work, **settings)
        np.testing.assert_array_equal(sf.read(tmp_path/'baseline.wav')[0], sf.read(tmp_path/'parallel.wav')[0])
        assert result['transitions'] == baseline['transitions']
        assert result['warnings'] == baseline['warnings']
        assert [t['id'] for t in result['tracks']] == [t['id'] for t in baseline['tracks']]
        for before, after in zip(baseline['tracks'], result['tracks']):
            assert before['warp'] == after['warp']
            np.testing.assert_array_equal(sf.read(before['audio_path'])[0], sf.read(after['audio_path'])[0])


@pytest.mark.parametrize('policy',['adaptive','echo_bridge'])
def test_parallel_creative_transitions_retain_both_decks_updates(tmp_path,policy):
    from test_sync import RenderedSyncTests
    from superdj.planner import plan_tracks
    from superdj.render import render_plan
    decoder=ffmpeg_executable()
    rubber=os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not rubber:
        pytest.skip('Actual Rubber Band required')
    records=RenderedSyncTests().fixtures(tmp_path,[174,128,126])
    plan=plan_tracks(records,mode='fixed')
    settings=dict(ffmpeg=decoder,rubberband=rubber,sample_rate=16000,tempo_conflict_policy=policy)
    baseline=render_plan(plan,records,{},tmp_path/'serial.wav',tmp_path/'serial',**settings)
    with execution(jobs=2) as session:
        parallel=render_plan(plan,records,{},tmp_path/'parallel.wav',session.work,**settings)
    assert baseline['transitions']==parallel['transitions']
    assert baseline['warnings']==parallel['warnings']
    np.testing.assert_array_equal(sf.read(tmp_path/'serial.wav')[0],sf.read(tmp_path/'parallel.wav')[0])
