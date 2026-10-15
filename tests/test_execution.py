"""Cache correctness, lifecycle and real rendered PCM invariants."""
import json
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from superdj.execution import cached, current, execution, key_for


@pytest.mark.parametrize('fail', [False, True])
def test_default_scratch_removed_on_success_and_failure(fail):
    root = None
    try:
        with execution() as session:
            root = session.root
            (session.work / 'pcm.wav').write_bytes(b'private')
            if fail:
                raise ValueError('expected')
    except ValueError:
        assert fail
    assert root is not None and not root.exists()
    assert current() is None


def test_content_identity_rename_and_source_change(tmp_path):
    a, b = tmp_path/'a.wav', tmp_path/'renamed.wav'
    a.write_bytes(b'first'); b.write_bytes(b'first')
    with execution(tmp_path/'cache'):
        assert key_for(a) == key_for(b)
        b.write_bytes(b'other')
        assert key_for(a) != key_for(b)


def test_corrupt_missing_objects_recomputed_and_settings_invalidate(tmp_path):
    calls = []
    with execution(tmp_path/'cache') as session:
        def produce():
            calls.append(1)
            path = session.work/'float.wav'
            path.write_bytes(b'original PCM')
            return {'path': str(path)}
        def run(setting=44100):
            return cached('pcm', {'rate': setting}, produce, session.work)
        first = run()
        assert run() == first and len(calls) == 1
        Path(first['path']).write_bytes(b'corrupted PCM')
        run(); assert len(calls) == 2
        Path(first['path']).unlink()
        run(); assert len(calls) == 3
        run(48000); assert len(calls) == 4
    assert Path(first['path']).exists()


def test_bounded_parallel_map_preserves_order_and_context(tmp_path):
    running, maximum = 0, 0
    lock = threading.Lock()
    with execution(tmp_path/'cache', jobs=2) as session:
        def one(i):
            nonlocal running, maximum
            assert current() is session
            with lock:
                running += 1; maximum = max(maximum, running)
            time.sleep(.01)
            with lock:
                running -= 1
            return i
        assert session.map(one, range(8)) == list(range(8))
    assert maximum == 2


@pytest.mark.parametrize('explicit_disable', [False, True])
def test_make_cache_policy_is_opt_in_and_no_cache_wins(tmp_path, monkeypatch, explicit_disable):
    import argparse
    import superdj.make as make_module
    seen = []
    monkeypatch.setattr(make_module, 'runtime_config', lambda: {
        'cache': str(tmp_path/'legacy-models'),
        **({'cache_mode': 'persistent', 'cache_root': str(tmp_path/'persist')} if explicit_disable else {})})
    def fake_make(args):
        seen.append((current().root, current().persistent))
        (current().work/'temporary.wav').write_bytes(b'audio')
    monkeypatch.setattr(make_module, '_make', fake_make)
    make_module.make(argparse.Namespace(cache=None, no_cache=explicit_disable, jobs=None, output=str(tmp_path/'results')))
    assert seen and not seen[0][1] and not seen[0][0].exists()
    assert not (tmp_path/'persist').exists()


def test_parallel_identical_objects_produced_once(tmp_path):
    calls = []
    with execution(tmp_path/'cache', jobs=2) as session:
        def produce():
            time.sleep(.02)
            calls.append(1)
            return {'verified': True}
        result = session.map(lambda _: cached('same', {'x': 1}, produce, session.work), range(4))
    assert result == [{'verified': True}]*4 and len(calls) == 1


def test_preparation_cache_independent_of_playlist_position(tmp_path, monkeypatch):
    from superdj import adapters
    source = tmp_path/'original.wav'; source.write_bytes(b'source')
    alias = tmp_path/'renamed.wav'; alias.write_bytes(source.read_bytes())
    calls = []
    monkeypatch.setattr(adapters, 'checkout_info', lambda *a: {'revision': 'audited'})
    def prepare(records, work, **kwargs):
        calls.append(1)
        output = Path(work)/'prepared'/'pcm.wav'; output.parent.mkdir(exist_ok=True)
        output.write_bytes(b'prepared PCM')
        return {records[0]['id']: {'path': str(output), 'provenance': {'processors': []}}}
    monkeypatch.setattr(adapters, '_prepare_tracks_uncached', prepare)
    with execution(tmp_path/'cache') as session:
        first = adapters.prepare_tracks([{'id':'slot1','path':str(source)}],session.work,audamcp='unused')
        second = adapters.prepare_tracks([{'id':'slot7','path':str(alias)}],session.work,audamcp='unused')
        assert first['slot1']['path'] == second['slot7']['path']
        assert second['slot7']['provenance']['original'] == str(alias)
    assert len(calls) == 1


def test_worker_errors_release_resources_and_reuse_process(tmp_path):
    import sys
    with execution(tmp_path/'cache') as session:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='Unknown neural task'):
                session.model(sys.executable, 'invalid-task', {'device':'cpu'})
        assert session.metrics['counts']['model_process_starts'] == 1
        processes = [w.process for w in session.workers.values()]
    assert all(p.poll() is not None for p in processes)


def test_analysis_cache_rebinds_playlist_id(tmp_path, monkeypatch):
    from superdj import analysis, cues, stems
    from superdj.audio import ffmpeg_executable
    try:
        decoder = ffmpeg_executable()
    except FileNotFoundError:
        pytest.skip('Actual decoder required')
    monkeypatch.setattr(cues, 'configuration', lambda: None)
    monkeypatch.setattr(stems, 'configuration', lambda: None)
    source = tmp_path/'source.wav'
    rate = 22050
    sf.write(source, .1*np.sin(2*np.pi*440*np.arange(8*rate)/rate), rate, subtype='FLOAT')
    grid = {'bpm': 120, 'first_downbeat_s': .25, 'meter': 4, 'reviewed': True}
    with execution(tmp_path/'cache') as session:
        first = analysis.analyze_track(source, track_id='slot_1', ffmpeg=decoder, grid=grid, cue_backend='structure')
        second = analysis.analyze_track(source, track_id='slot_7', ffmpeg=decoder, grid=grid, cue_backend='structure')
        assert first['id'] == 'slot_1' and second['id'] == 'slot_7'
        assert first['beats'] == second['beats']
        assert session.metrics['counts']['analysis_hits'] == 1


def _tools():
    ffmpeg = os.getenv('SUPERDJ_FFMPEG') or shutil.which('ffmpeg')
    rubber = os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not ffmpeg or not rubber:
        pytest.skip('Actual FFmpeg and Rubber Band required')
    return ffmpeg, rubber


def test_cached_render_pcm_identical_and_early_failure(tmp_path, monkeypatch):
    from test_sync import RenderedSyncTests
    from superdj.render import render_plan, RenderError
    from superdj.planner import plan_tracks
    from superdj import kick_grid
    ffmpeg, rubber = _tools()
    records = RenderedSyncTests().fixtures(tmp_path, [120, 122, 124])
    plan = plan_tracks(records, mode='fixed')
    settings = dict(ffmpeg=ffmpeg, rubberband=rubber, sample_rate=16000, tempo_conflict_policy='sync')
    render_plan(plan, records, {}, tmp_path/'uncached.wav', tmp_path/'uncached', **settings)
    for index in range(2):
        with execution(tmp_path/'cache') as session:
            render_plan(plan, records, {}, tmp_path/f'cached{index}.wav', session.work, **settings)
            if index:
                assert session.metrics['counts']['clip_hits'] == 3
                assert session.metrics['counts']['overlap_hits'] == 2
        assert np.array_equal(sf.read(tmp_path/'uncached.wav')[0], sf.read(tmp_path/f'cached{index}.wav')[0])
    progress = []
    def refuse(*args, **kwargs):
        raise ValueError('intentional pair failure')
    monkeypatch.setattr(kick_grid, 'validate_overlap_attacks', refuse)
    with pytest.raises(RenderError, match='intentional pair failure'):
        render_plan(plan, records, {}, tmp_path/'failed.wav', tmp_path/'failed', progress=progress.append, **settings)
    assert not any('3/3' in line for line in progress)
    assert not (tmp_path/'failed.wav').exists()
