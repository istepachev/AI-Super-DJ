import os
from pathlib import Path
import shutil
import numpy as np
import pytest
import soundfile as sf

from superdj.quality import export_audio, measure_audio, parse_ebu


def test_parse_last_full_summary():
    output = "Summary:\nIntegrated loudness:\n I: -40.0 LUFS\nLoudness range:\n LRA: 0.0 LU\nTrue peak:\n Peak: -22.0 dBFS\nSummary:\nIntegrated loudness:\n I: -14.3 LUFS\nLoudness range:\n LRA: 3.1 LU\nTrue peak:\n Peak: -1.6 dBFS\n"
    assert parse_ebu(output) == {"integrated_lufs": -14.3, "lra_lu": 3.1, "true_peak_dbfs": -1.6}


def test_real_encoded_audio_and_protect_existing(tmp_path):
    ffmpeg = os.environ.get("SUPERDJ_TEST_FFMPEG") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("SUPERDJ_TEST_FFPROBE") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("External FFmpeg/FFprobe required for audio integration")
    rate = 44100
    t = np.arange(rate * 4) / rate
    signal = 0.95 * np.sin(2 * np.pi * 997 * t)
    stereo = np.column_stack((signal, signal * 0.8)).astype(np.float32)
    source = tmp_path / "source.wav"
    sf.write(source, stereo, rate, subtype="FLOAT")
    exports = export_audio(source, tmp_path / "export", "synthetic", ffmpeg=ffmpeg, ffprobe=ffprobe)
    assert exports["flac"]["bits_per_raw_sample"] == 24
    assert exports["flac"]["channels"] == 2
    for result in exports.values():
        assert result["true_peak_dbfs"] <= -1.0
        assert result["full_decode_ok"]
        assert abs(result["duration_seconds"] - 4) < 0.1
    original = Path(exports["flac"]["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        export_audio(source, tmp_path / "export", "synthetic", ffmpeg=ffmpeg, ffprobe=ffprobe)
    assert Path(exports["flac"]["path"]).read_bytes() == original


def test_float_overshoot_is_not_hidden_by_s16(tmp_path):
    ffmpeg = os.environ.get("SUPERDJ_TEST_FFMPEG") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("SUPERDJ_TEST_FFPROBE") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("External FFmpeg/FFprobe required for audio integration")
    rate = 44100
    t = np.arange(rate * 2) / rate
    source = tmp_path / "overshoot.wav"
    sf.write(source, 1.25 * np.sin(2 * np.pi * 500 * t), rate, subtype="FLOAT")
    result = measure_audio(source, ffmpeg=ffmpeg, ffprobe=ffprobe)
    assert result["sample_peak_dbfs"] > 1.8
    assert result["true_peak_dbfs"] > 1.8
