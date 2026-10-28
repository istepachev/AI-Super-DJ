from copy import deepcopy
import json
import struct
import numpy as np
import pytest
import soundfile as sf

from superdj import retained_preparation as tool
from superdj.execution import sha256
from superdj.loop_evidence import render_records
from superdj.loops import repeat_file
from superdj.render import RenderError


@pytest.fixture
def retained(tmp_path):
    sr = 8000
    t = np.arange(12 * sr) / sr
    voice = .025 * np.sin(2 * np.pi * 440 * t)
    drum = .12 * np.sin(2 * np.pi * 65 * t) * np.exp(-((t % .25) / .08))
    signals = {'path': voice + drum, 'vocal_path': voice, 'drum_path': drum}
    edit = {'id': 'track_01', 'method': 'verified phrase repeat with complementary-band join',
        'loop_start_seconds': 4., 'insertion_source_seconds': 6., 'loop_seconds': 2.,
        'repeat_count': 1, 'added_seconds': 2., 'bars_repeated': 2,
        'original_source_duration_seconds': 12., 'join_frames': 4000,
        'beat_seconds': .25, 'sample_rate': sr, 'native_join_contexts': {},
        'aligned_preparation_streams': list(signals)}
    item = {}
    for key, signal in signals.items():
        native = tmp_path / (key + '_native.wav')
        stereo = np.column_stack((signal, signal * .87))
        sf.write(native, stereo, sr, subtype='FLOAT')
        target = tmp_path / (key + '_loop.wav')
        context = tmp_path / (key + '_join.npy')
        edit['native_join_contexts'][key] = repeat_file(native, target, edit, context_path=context)
        item[key] = str(target)
    item['provenance'] = {'timeline_edit': deepcopy(edit)}
    return item


def hashes(item):
    return {k: sha256(item[k]) for k in ('path', 'vocal_path', 'drum_path')}


def test_three_streams_and_feature_timeline_remain_unchanged(retained, tmp_path):
    original = deepcopy(retained)
    before = hashes(retained)
    row = {'id': 'track_01', 'timeline_edit': deepcopy(retained['provenance']['timeline_edit']),
        'vocal_stem': {'profile': [{'time': 2., 'foreground_risk': .1},
            {'time': 4.5, 'foreground_risk': .1}, {'time': 9., 'foreground_risk': .1}]}}
    first, _ = render_records([row], {'track_01': retained})
    result = tool.reprepare_retained_loop('track_01', retained, tmp_path / 'fresh')
    second, _ = render_records(first, {'track_01': result['preparation']})
    assert result['verification']['verified'] is True
    assert len(result['verification']['streams']) == 3
    assert all(s['frames_checked'] == 14 * 8000 for s in result['verification']['streams'])
    assert result['verification']['musical_mix_verified'] is False
    assert result['preparation'] == original == retained
    assert hashes(retained) == before
    assert second[0]['vocal_stem']['profile'] == first[0]['vocal_stem']['profile']


def test_container_metadata_difference_is_not_a_pcm_change(retained, tmp_path):
    path = tmp_path / 'path_loop.wav'
    data = path.read_bytes()
    assert data[:4] == b'RIFF'
    data += b'JUNK' + struct.pack('<I', 4) + b'test'
    data = data[:4] + struct.pack('<I', len(data) - 8) + data[8:]
    path.write_bytes(data)
    result = tool.reprepare_retained_loop('track_01', retained, tmp_path / 'fresh')
    stream = result['verification']['streams'][0]
    assert stream['canonical_sha256'] != stream['fresh_sha256']
    assert stream['all_pcm_samples_identical'] is True
    assert result['verification']['verified'] is True


def test_one_changed_fresh_sample_stops_verification(retained, tmp_path, monkeypatch):
    before = hashes(retained)
    ordinary = tool.repeat_file
    def corrupt(source, destination, edit, **kwargs):
        result = ordinary(source, destination, edit, **kwargs)
        with sf.SoundFile(destination, 'r+') as f:
            f.seek(1000)
            sample = f.read(1, dtype='float64', always_2d=True)
            sample[0, 0] += .001
            f.seek(1000)
            f.write(sample)
        return result
    monkeypatch.setattr(tool, 'repeat_file', corrupt)
    output = tmp_path / 'fresh'
    with pytest.raises(tool.RetainedLoopVerificationError, match='PCM differs'):
        tool.reprepare_retained_loop('track_01', retained, output)
    assert hashes(retained) == before
    assert json.loads((output / 'verification.json').read_text('utf8'))['verified'] is False


def test_float64_difference_is_not_rounded_away(retained, tmp_path):
    audio, sr = sf.read(retained['path'], dtype='float64', always_2d=True)
    audio[1000, 0] += 2e-10
    sf.write(retained['path'], audio, sr, subtype='DOUBLE')
    before = hashes(retained)
    with pytest.raises(tool.RetainedLoopVerificationError, match='PCM differs'):
        tool.reprepare_retained_loop('track_01', retained, tmp_path / 'fresh')
    assert hashes(retained) == before


def test_misaligned_drum_timeline_is_rejected(retained, tmp_path):
    audio, sr = sf.read(retained['drum_path'], dtype='float32', always_2d=True)
    sf.write(retained['drum_path'], audio[:-1], sr, subtype='FLOAT')
    before = hashes(retained)
    with pytest.raises(ValueError, match='timeline'):
        tool.reprepare_retained_loop('track_01', retained, tmp_path / 'fresh')
    assert hashes(retained) == before


def test_missing_native_context_is_not_reported_as_verified(retained, tmp_path):
    from pathlib import Path
    context = retained['provenance']['timeline_edit']['native_join_contexts']['drum_path']['path']
    Path(context).unlink()
    before = hashes(retained)
    output = tmp_path / 'fresh'
    with pytest.raises((FileNotFoundError, RenderError)):
        tool.reprepare_retained_loop('track_01', retained, output)
    assert hashes(retained) == before
    assert json.loads((output / 'verification.json').read_text('utf8'))['verified'] is False


def test_existing_output_is_never_overwritten(retained, tmp_path):
    output = tmp_path / 'fresh'
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('preserve', encoding='utf8')
    before = hashes(retained)
    with pytest.raises(FileExistsError):
        tool.reprepare_retained_loop('track_01', retained, output)
    assert marker.read_text('utf8') == 'preserve'
    assert hashes(retained) == before
