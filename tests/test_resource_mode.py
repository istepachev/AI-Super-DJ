"""Resource policy must bound scheduling without changing musical decisions."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import threading

import pytest

from superdj import resource_limits as limits
from superdj import resource_config as config
from superdj import native_limits as native
from superdj.resources import GIB, memory_policy, preflight


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    path = tmp_path/'runtime.local.json'
    monkeypatch.setattr(limits, 'runtime_path', lambda: path)
    monkeypatch.setattr(config, 'runtime_path', lambda: path)
    monkeypatch.delenv(limits.ENVIRONMENT_KEY, raising=False)
    return path


@pytest.mark.parametrize('value', [True, [], {'enabled': 1}, {'enabled': True, 'cpu_cores': True},
    {'enabled': True, 'model_jobs': 5}, {'enabled': True, 'cpu_jobs': 0},
    {'enabled': True, 'math_threads': 17}, {'enabled': True, 'memory_headroom_gib': float('nan')},
    {'enabled': True, 'memory_headroom_gib': -1}, {'enabled': True, 'unknown': 1}])
def test_invalid_profile_rejected(value):
    with pytest.raises(ValueError):
        limits.settings(value)


def test_disabled_profile_leaves_default_queue_policy():
    assert not limits.current_settings().enabled
    assert limits.bounded_jobs(4, 3) == (4, 3)
    assert limits.math_thread_limit(4) == 4


def test_malformed_runtime_has_clear_error():
    with pytest.raises(ValueError, match='JSON object'):
        limits.load_settings([])


def test_enabled_caps_requests_and_lower_requests_survive():
    profile = limits.settings({'enabled': True, 'model_jobs': 2, 'cpu_jobs': 1})
    assert limits.bounded_jobs(4, 4, profile) == (2, 1)
    assert limits.bounded_jobs(1, 1, profile) == (1, 1)


def test_scope_retains_stricter_environment_and_restores_only_own_changes(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '8')
    with limits.limit_scope({'enabled': True, 'math_threads': 2}) as report:
        assert os.environ['OMP_NUM_THREADS'] == '1'
        assert os.environ['MKL_NUM_THREADS'] == '2'
        assert json.loads(os.environ[limits.ENVIRONMENT_KEY])['enabled']
        assert limits.math_thread_limit(4) == 1
        assert report['root_native_policy'] is None
        os.environ['MKL_NUM_THREADS'] = '1'  # another owner's stricter update
    assert os.environ['OMP_NUM_THREADS'] == '1'
    assert os.environ['MKL_NUM_THREADS'] == '1'
    assert limits.ENVIRONMENT_KEY not in os.environ


@pytest.mark.parametrize('enabled', [False, True])
def test_nested_scope_keeps_running_policy(enabled, isolated_runtime):
    with limits.limit_scope({'enabled': enabled}):
        isolated_runtime.write_text(json.dumps({'resource_limits': {'enabled': not enabled}}))
        with limits.limit_scope():
            assert limits.current_settings().enabled == enabled


def test_environment_serializes_only_profile(monkeypatch):
    monkeypatch.setenv(limits.ENVIRONMENT_KEY, json.dumps({'enabled': True, 'cpu_cores': 2}))
    assert limits.current_settings().cpu_cores == 2
    assert set(asdict(limits.current_settings())) == set(asdict(limits.LimitSettings()))


def test_disabled_running_policy_is_also_serialized_for_owned_children(monkeypatch, isolated_runtime):
    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    with limits.limit_scope({'enabled': False}):
        isolated_runtime.write_text(json.dumps({'resource_limits': {'enabled': True}}))
        assert not json.loads(os.environ[limits.ENVIRONMENT_KEY])['enabled']
        limits.bootstrap_math_threads()
        assert os.environ['OMP_NUM_THREADS'] == '4'
    assert limits.ENVIRONMENT_KEY not in os.environ


def test_serialized_owned_run_overrides_a_changed_runtime(monkeypatch, isolated_runtime):
    isolated_runtime.write_text(json.dumps({'resource_limits': {'enabled': False}}))
    monkeypatch.setenv(limits.ENVIRONMENT_KEY, json.dumps({'enabled': True, 'cpu_cores': 2}))
    with limits.limit_scope():
        assert limits.current_settings().enabled
        assert limits.current_settings().cpu_cores == 2


def test_pool_control_unavailable_keeps_queue_and_environment(monkeypatch):
    monkeypatch.setitem(sys.modules, 'threadpoolctl', None)
    with limits.limit_scope({'enabled': True}) as report:
        assert report['math_pool_control_error'] == 'ModuleNotFoundError'
        assert os.environ['OMP_NUM_THREADS'] == '1'
        assert limits.bounded_jobs(4, 4) == (1, 1)


def test_loaded_numba_uses_mask_without_rewriting_fixed_launch_capacity(monkeypatch):
    state = {'threads': 4}
    fake = SimpleNamespace(get_num_threads=lambda: state['threads'],
                           set_num_threads=lambda n: state.update(threads=n))
    monkeypatch.setitem(sys.modules, 'numba', fake)
    monkeypatch.setenv('NUMBA_NUM_THREADS', '4')
    with limits.limit_scope({'enabled': True}) as report:
        assert os.environ['NUMBA_NUM_THREADS'] == '4'
        assert state['threads'] == 1
        assert report['numba_active_threads'] == 1
    assert state['threads'] == 4


def test_numba_mask_applied_in_each_owned_cpu_thread(monkeypatch, tmp_path):
    from superdj.execution import execution
    local = threading.local()
    fake = SimpleNamespace(get_num_threads=lambda: getattr(local, 'threads', 4),
                           set_num_threads=lambda n: setattr(local, 'threads', n))
    monkeypatch.setitem(sys.modules, 'numba', fake)
    with limits.limit_scope({'enabled': True, 'model_jobs': 2, 'cpu_jobs': 2}):
        with execution(tmp_path/'cache', jobs=2, cpu_jobs=2) as session:
            result = session.cpu_map(lambda n: fake.get_num_threads(), range(4))
    assert result == [1, 1, 1, 1]
    assert fake.get_num_threads() == 4


def test_numba_control_failure_keeps_scope_cleanup(monkeypatch):
    def fail(n):
        raise RuntimeError('simulated unsupported runtime')
    monkeypatch.setitem(sys.modules, 'numba', SimpleNamespace(get_num_threads=lambda: 4, set_num_threads=fail))
    with limits.limit_scope({'enabled': True}) as report:
        assert report['numba_thread_control_error'] == 'RuntimeError'
        assert limits.current_settings().enabled
    assert not limits.current_settings().enabled
    assert limits.ENVIRONMENT_KEY not in os.environ


def test_sdk_execution_limits_owned_session_without_pinning_host(tmp_path, monkeypatch):
    from superdj.execution import execution
    monkeypatch.setattr(native, 'apply_owned_limits', lambda *a, **k: pytest.fail('SDK host was pinned'))
    with limits.limit_scope({'enabled': True}):
        with execution(tmp_path/'cache', jobs=4, cpu_jobs=3, scratch_root=tmp_path/'ssd') as session:
            assert (session.jobs, session.cpu_jobs) == (1, 1)
            assert not session.preparation_overlap
            assert session.work.parent == tmp_path/'ssd'
            assert session.limit_settings.enabled
            assert session.metrics['resource_limits']['root_native_policy'] is None


def test_sdk_session_refuses_without_minimum_headroom_before_worker(tmp_path, monkeypatch):
    from superdj.execution import execution
    from superdj import resources
    monkeypatch.setattr(resources, 'available_memory', lambda: 4*GIB)
    monkeypatch.setattr(native, 'apply_owned_limits', lambda *a, **k: pytest.fail('No worker should start'))
    with limits.limit_scope({'enabled': True}):
        with pytest.raises(MemoryError, match='no worker was started'):
            with execution(tmp_path/'cache'):
                pytest.fail('Admission should reject this session')


def test_headroom_is_estimate_and_profile_disables_phase_overlap(tmp_path):
    machine = dict(cpu_threads=24, available_memory_bytes=32*GIB, total_memory_bytes=64*GIB,
                   gpus=[], cuda_probe='unknown')
    controls = dict(output=tmp_path, machine=machine, jobs=4, cpu_jobs=4,
                    probe=lambda p: (240., 44100), disk=lambda p: (tmp_path, 'one', 100*GIB))
    normal = preflight(['a.wav', 'b.wav'], **controls)
    with limits.limit_scope({'enabled': True}):
        policy = memory_policy(memory=32*GIB, total=64*GIB)
        result = preflight(['a.wav', 'b.wav'], **controls)
    assert policy['os_application_headroom_bytes'] >= 4*GIB
    assert policy['reservation'] is False
    assert (result['jobs'], result['cpu_jobs']) == (1, 1)
    assert not result['preparation_overlap']
    assert normal['preparation_overlap']
    assert result['devices'] == normal['devices']


def test_insufficient_headroom_refuses_full_task_before_inference(tmp_path):
    machine = dict(cpu_threads=24, available_memory_bytes=5*GIB, total_memory_bytes=32*GIB,
                   gpus=[], cuda_probe='unknown')
    with limits.limit_scope({'enabled': True}):
        with pytest.raises(MemoryError, match='headroom'):
            preflight(['a.wav'], output=tmp_path, machine=machine,
                      probe=lambda p: (300., 44100), disk=lambda p: (tmp_path, 'one', 100*GIB))


def test_only_currently_allowed_cpus_selected():
    assert native.selected_cpus([8, 10, 12, 14, 16], 4) == [8, 10, 12, 14]
    assert native.selected_cpus([9], 4) == [9]
    with pytest.raises(ValueError):
        native.selected_cpus([], 4)


def test_arbitrary_pid_not_accepted_and_finished_child_not_touched():
    profile = limits.settings({'enabled': True})
    with pytest.raises(TypeError):
        native.apply_owned_limits(profile, 12345)
    result = native.apply_owned_limits(profile, SimpleNamespace(pid=12345, poll=lambda: 0))
    assert not result['applied']


def test_nonwindows_io_limitation_is_explicit_partial_policy(monkeypatch):
    selected = []
    fake = SimpleNamespace(name='posix', sched_getaffinity=lambda pid: {2, 3, 4},
        sched_setaffinity=lambda pid, cpus: selected.extend(cpus),
        setpriority=lambda *a: None, getpriority=lambda *a: 19, PRIO_PROCESS=0)
    monkeypatch.setattr(native, 'os', fake)
    report = native.apply_owned_limits(limits.settings({'enabled': True}))
    assert selected == [2, 3, 4]
    assert report['partial'] and not report['complete_policy_applied']
    assert report['io_priority'] is None


def test_failed_windows_operation_cannot_report_full_success(monkeypatch):
    monkeypatch.setattr(native, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(native, '_windows', lambda *a: dict(cpu_ids=[1], priority_class=64,
        io_priority=None, errors=['I/O priority readback unavailable']))
    report = native.apply_owned_limits(limits.settings({'enabled': True}))
    assert not report['applied'] and report['partial']


def test_config_field_only_undo_preserves_later_foreign_updates(isolated_runtime):
    path = isolated_runtime
    path.write_text(json.dumps({'jobs': 1, 'scratch_root': 'my-ssd', 'other': {'keep': True}}))
    original = json.loads(path.read_text())
    assert config.configure('on')['changed']
    changed = json.loads(path.read_text())
    assert {k:v for k,v in changed.items() if k != config.FIELD} == original
    changed['later'] = 'must survive'; path.write_text(json.dumps(changed))
    config.configure('restore')
    restored = json.loads(path.read_text())
    assert restored == {**original, 'later': 'must survive'}
    with pytest.raises(ValueError, match='already restored'):
        config.configure('restore')


def test_off_then_restore_retains_custom_profile(isolated_runtime):
    original = {'enabled': True, 'cpu_cores': 2, 'memory_headroom_gib': 6.}
    isolated_runtime.write_text(json.dumps({config.FIELD: original, 'jobs': 1}))
    config.configure('off'); assert not config.configure('status')['resource_limits']['enabled']
    config.configure('restore')
    assert json.loads(isolated_runtime.read_text())[config.FIELD] == original


def test_rollback_refuses_foreign_profile_edit(isolated_runtime):
    config.configure('on')
    changed = json.loads(isolated_runtime.read_text()); changed[config.FIELD]['cpu_cores'] = 2
    isolated_runtime.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match='edited after'):
        config.configure('restore')
    assert json.loads(isolated_runtime.read_text()) == changed


def test_compare_before_replace_preserves_concurrent_runtime_change(tmp_path):
    path = tmp_path/'runtime.json'; path.write_bytes(b'new foreign bytes')
    with pytest.raises(RuntimeError, match='concurrently'):
        config._atomic(path, {'wrong': True}, expected=b'old bytes', compare=True)
    assert path.read_bytes() == b'new foreign bytes'
    assert not list(tmp_path.glob('.rd-*'))


def test_config_lock_wait_is_bounded(tmp_path):
    path = tmp_path/'config.lock'
    with config._lock(path):
        with pytest.raises(TimeoutError):
            with config._lock(path, timeout=.01):
                pytest.fail('A second updater acquired an exclusive lock')


def test_repeated_on_preserves_original_rollback(isolated_runtime):
    config.configure('on')
    backup = isolated_runtime.with_name('resource-mode-rollback.json').read_bytes()
    assert not config.configure('on')['changed']
    assert isolated_runtime.with_name('resource-mode-rollback.json').read_bytes() == backup


def test_cli_status_does_not_apply_native_policy(monkeypatch, capsys):
    from superdj.cli import main
    monkeypatch.setattr(native, 'apply_owned_limits', lambda *a, **k: pytest.fail('status changed scheduling'))
    main(['resource-mode', 'status'])
    assert json.loads(capsys.readouterr().out)['resource_limits']['enabled'] is False


@pytest.mark.parametrize('owned', [False, True])
def test_imported_cli_cannot_pin_an_embedding_host(monkeypatch, owned):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['embedding-host.py'])
    monkeypatch.setattr(sys, 'orig_argv', ['python', '-m', 'superdj'] if owned else ['python', 'embedding-host.py'])
    monkeypatch.setattr(limits, 'load_settings', lambda: limits.settings({'enabled': True}))
    monkeypatch.setattr(native, 'apply_owned_limits', lambda value: calls.append(value) or {'owned': True})
    with limits.cli_limit_scope(['plan']) as report:
        assert (report['root_native_policy'] == {'owned': True}) is owned
    assert len(calls) == int(owned)


def test_invalid_config_can_be_disabled_without_full_config_replacement(isolated_runtime):
    isolated_runtime.write_text(json.dumps({config.FIELD: {'enabled': True, 'cpu_cores': 'bad'}, 'keep': 42}))
    config.configure('off')
    assert json.loads(isolated_runtime.read_text())['keep'] == 42
    assert not config.configure('status')['resource_limits']['enabled']


def test_enabled_profile_preserves_actual_sync_pcm_and_gates(tmp_path):
    import numpy as np
    import soundfile as sf
    from test_execution import _tools
    from test_sync import RenderedSyncTests
    from superdj.execution import execution
    from superdj.planner import plan_tracks
    from superdj.render import render_plan
    ffmpeg, rubber = _tools()
    records = RenderedSyncTests().fixtures(tmp_path, [120, 122, 124])
    plan = plan_tracks(records, mode='fixed')
    outputs = []
    for enabled in (False, True):
        destination = tmp_path/f'profile-{enabled}.wav'
        with limits.limit_scope({'enabled': enabled}):
            # Separate empty caches require two real renders, not cache replay.
            with execution(tmp_path/f'cache-{enabled}', jobs=1 if not enabled else 4,
                           cpu_jobs=1 if not enabled else 4) as session:
                result = render_plan(plan, records, {}, destination, session.work,
                    ffmpeg=ffmpeg, rubberband=rubber, sample_rate=16000,
                    tempo_conflict_policy='sync')
                assert session.jobs == session.cpu_jobs == 1
                assert result['assembly_verification']['verified']
                assert len(result['tracks']) == 3
        outputs.append(sf.read(destination)[0])
    assert np.array_equal(*outputs)


def test_actual_worker_reports_own_policy_even_on_request_error(monkeypatch):
    import io
    from superdj import model_worker
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdin', io.StringIO('{"kind":"unsupported-probe","job":{}}\n'))
    monkeypatch.setattr(sys, 'stdout', output)
    # Do not call enabled worker.main in the pytest host: its native scope
    # belongs to a dedicated worker process. The real child smoke covers that.
    with limits.limit_scope({'enabled': True}):
        model_worker._work()
    reply = json.loads(output.getvalue())
    assert 'error' in reply
    assert reply['resource_limits']['pid'] == os.getpid()
    assert reply['resource_limits']['report']['settings']['enabled']
    assert reply['resource_limits']['report']['root_native_policy'] is None


def test_nonwindows_child_native_policy_never_targets_a_recyclable_pid(monkeypatch):
    monkeypatch.setattr(native, 'os', SimpleNamespace(name='posix'))
    report = native.apply_owned_limits(limits.settings({'enabled': True}),
        SimpleNamespace(pid=12345, poll=lambda: None))
    assert not report['applied']
    assert report['worker_self_policy_pending']


def test_low_memory_bootstrap_keeps_an_already_loaded_numba_capacity(monkeypatch):
    from superdj import resources
    monkeypatch.setattr(resources, 'memory_policy', lambda: {'constrained': True})
    monkeypatch.setattr(limits, 'bootstrap_math_threads', lambda: None)
    monkeypatch.setitem(sys.modules, 'numba', SimpleNamespace())
    monkeypatch.setenv('NUMBA_NUM_THREADS', '4')
    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    resources.configure_math_threads()
    assert os.environ['NUMBA_NUM_THREADS'] == '4'
    assert os.environ['OMP_NUM_THREADS'] == '1'
