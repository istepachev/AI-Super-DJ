"""Real codec input, explicit worker paths, and preservation of originals."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from superdj.audio import decode_mono, decode_f32_file, ffmpeg_executable
from superdj.history import track_key
from superdj.make import collect


@pytest.fixture
def decoder():
    try:
        return ffmpeg_executable()
    except FileNotFoundError:
        pytest.skip('Actual FFmpeg required for codec integration checks')


@pytest.fixture
def signal_file(tmp_path):
    rate = 48000
    t = np.arange(rate * 2) / rate
    audio = np.column_stack((.15*np.sin(2*np.pi*440*t), .10*np.sin(2*np.pi*880*t)))
    path = tmp_path/'source.wav'
    sf.write(path,audio,rate,subtype='FLOAT')
    return path


@pytest.mark.parametrize('extension,codec', [
    ('wav','pcm_s24le'), ('flac','flac'), ('mp3','libmp3lame'),
    ('m4a','aac'), ('m4a','alac'), ('m4b','aac'), ('aac','aac'),
    ('aiff','pcm_s24be'), ('aif','pcm_s16be'), ('ogg','libvorbis'),
    ('oga','libvorbis'), ('opus','libopus'), ('wma','wmav2'),
    ('wv','wavpack'), ('caf','pcm_s16le'), ('au','pcm_s16be'),
    ('mp4','aac'), ('mka','flac'), ('webm','libopus'),
])
def test_real_codecs_decode_without_path_or_additional_codec_packages(tmp_path,decoder,signal_file,monkeypatch,extension,codec):
    encoded=tmp_path/f'Пример трека λ.{extension}'
    subprocess.run([decoder,'-nostdin','-v','error','-i',str(signal_file),'-c:a',codec,str(encoded)],check=True,capture_output=True)
    digest=hashlib.sha256(encoded.read_bytes()).hexdigest()
    monkeypatch.setenv('PATH','')
    monkeypatch.delenv('SUPERDJ_FFMPEG',raising=False)
    monkeypatch.setattr('superdj.cues.runtime_config',lambda:{'ffmpeg':decoder})
    audio=decode_mono(encoded)
    assert audio.dtype==np.float32 and np.isfinite(audio).all()
    assert 1.9 < len(audio)/22050 < 2.16
    spectrum=np.abs(np.fft.rfft(audio[round(.25*22050):round(1.75*22050)]))
    frequency=np.fft.rfftfreq(round(1.5*22050),1/22050)[np.argmax(spectrum)]
    assert frequency==pytest.approx(440,abs=2)
    assert hashlib.sha256(encoded.read_bytes()).hexdigest()==digest
    assert collect([encoded])==[encoded.resolve()]


def test_container_detection_ignores_misleading_mp3_suffix(tmp_path,decoder,signal_file):
    actual=tmp_path/'AAC.m4a'
    subprocess.run([decoder,'-nostdin','-v','error','-i',str(signal_file),'-c:a','aac',str(actual)],check=True,capture_output=True)
    renamed=tmp_path/'Unknown Artist - пример.m4a.mp3'
    shutil.copyfile(actual,renamed)
    np.testing.assert_array_equal(decode_mono(actual,ffmpeg=decoder),decode_mono(renamed,ffmpeg=decoder))


def test_pipe_and_memory_mapped_analysis_receive_the_same_float_samples(tmp_path,decoder,signal_file):
    raw=decode_f32_file(signal_file,tmp_path/'analysis.f32',ffmpeg=decoder)
    np.testing.assert_array_equal(np.fromfile(raw,dtype='<f4'),decode_mono(signal_file,ffmpeg=decoder))


def test_bad_or_empty_input_has_a_decode_error_and_leaves_no_partial_pcm(tmp_path,decoder):
    source=tmp_path/'broken.m4a'
    source.write_bytes(b'not an audio container')
    output=tmp_path/'partial.f32'
    with pytest.raises(RuntimeError,match='Cannot decode audio'):
        decode_f32_file(source,output,ffmpeg=decoder)
    assert not output.exists() and source.read_bytes()==b'not an audio container'


def test_decode_cannot_overwrite_original_or_existing_output(tmp_path,decoder,signal_file):
    original=signal_file.read_bytes()
    with pytest.raises(ValueError,match='original audio'):
        decode_f32_file(signal_file,signal_file,ffmpeg=decoder)
    assert signal_file.read_bytes()==original
    other=tmp_path/'existing.f32';other.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        decode_f32_file(signal_file,other,ffmpeg=decoder)
    assert other.read_bytes()==b'preserve'


def test_explicit_decoder_is_not_replaced_by_a_different_runtime(decoder,monkeypatch):
    monkeypatch.setenv('SUPERDJ_FFMPEG','missing-env-ffmpeg')
    monkeypatch.setattr('superdj.cues.runtime_config',lambda:{'ffmpeg':'missing-runtime-ffmpeg'})
    assert ffmpeg_executable(decoder)==str(Path(decoder).resolve())
    with pytest.raises(FileNotFoundError):
        ffmpeg_executable('missing-explicit-ffmpeg')


@pytest.mark.parametrize('suffix',['.m4b','.aac','.wma','.ape','.wv','.oga','.caf','.mka','.webm'])
def test_history_matches_new_supported_containers_to_existing_track_names(suffix):
    assert track_key('Artist - Title'+suffix)==track_key('Artist - Title.flac')
