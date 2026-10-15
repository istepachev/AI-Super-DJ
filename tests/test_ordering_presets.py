"""Fresh-machine preset selection and actual remaining-pool behavior."""
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from superdj.cli import main
from superdj.presets import ORDERING_PRESETS, ordering_profile


@pytest.mark.parametrize('name', ORDERING_PRESETS)
def test_packaged_profile_needs_no_private_runtime(name, monkeypatch):
    import superdj.cues
    def private_access():
        raise AssertionError('A portable ordering preset must not read private runtime')
    monkeypatch.setattr(superdj.cues, 'runtime_config', private_access)
    profile = ordering_profile(name)
    assert profile['identifier_features'] is False
    assert profile['independent_validation'] is False
    assert profile['publication']['inference_parameters_unchanged'] is True
    assert all(set(row) == {'mode', 'routing_controls', 'metadata_sha256', 'cohort_group'}
               for row in profile['training_provenance'])


def test_next_preset_cli_excludes_every_played_track_and_caps_suggestions(tmp_path):
    records = [dict(id=str(i), bpm=118 + i, key='8A') for i in range(12)]
    records.append(dict(id='outside', bpm=120, key='3B'))
    source = tmp_path / 'records.json'
    source.write_text(json.dumps(records), encoding='utf-8')
    prefix = tmp_path / 'played.json'
    prefix.write_text(json.dumps(['0', '1', '2']), encoding='utf-8')
    main(['harmonize-rank', '--records', str(source), '--preset', 'harmonic-next-v1',
          '--current', '2', '--played-prefix', str(prefix), '--harmonic-mode', 'mood',
          '--routing-priority', 'bpm', '--bpm-tolerance', '4', '--work', str(tmp_path / 'rank')])
    result = json.loads((tmp_path / 'rank/next_track_rankings.json').read_text(encoding='utf-8'))
    assert set(result['remaining_ids']) == {str(i) for i in range(3, 12)} | {'outside'}
    assert len(result['shortlist_ids']) == 3
    assert not set(result['shortlist_ids']) & {'0', '1', '2', 'outside'}
    assert result['whole_path_optimized'] is False


def test_route_preset_assembles_twenty_unique_tracks_with_explicit_locks(tmp_path):
    records = [dict(id=str(i), bpm=118 + i / 2, key='8A') for i in range(20)]
    source = tmp_path / 'records.json'
    source.write_text(json.dumps(records), encoding='utf-8')
    chains = tmp_path / 'chains.json'
    chains.write_text(json.dumps([['8', '9']]), encoding='utf-8')
    main(['harmonize', '--records', str(source), '--preset', 'harmonic-route-v1',
          '--harmonic-mode', 'mood', '--routing-priority', 'bpm', '--priority', '0',
          '--bpm-tolerance', '4', '--direction', 'none', '--seed', '0',
          '--first', '0', '--last', '19', '--locked-chains', str(chains),
          '--work', str(tmp_path / 'route')])
    result = json.loads((tmp_path / 'route/harmonize_plan.json').read_text(encoding='utf-8'))
    order = result['order']
    assert len(order) == len(set(order)) == 20
    assert set(order) == {str(i) for i in range(20)}
    assert order[0] == '0' and order[-1] == '19'
    assert order[order.index('8') + 1] == '9'
    assert result['harmonic_matches'] == 19
    assert result['ordering_preset'] == 'harmonic-route-v1'
    source.write_text(json.dumps(list(reversed(records))), encoding='utf-8')
    main(['harmonize', '--records', str(source), '--preset', 'harmonic-route-v1',
          '--harmonic-mode', 'mood', '--routing-priority', 'bpm', '--priority', '0',
          '--bpm-tolerance', '4', '--direction', 'none', '--seed', '0',
          '--first', '0', '--last', '19', '--locked-chains', str(chains),
          '--work', str(tmp_path / 'reversed-route')])
    reversed_result = json.loads((tmp_path / 'reversed-route/harmonize_plan.json').read_text(encoding='utf-8'))
    assert reversed_result['order'] == order
    assert reversed_result['incumbent_order'] == result['incumbent_order']


def test_preset_selector_rejects_unknown_and_conflicting_sources():
    with pytest.raises(ValueError, match='Unknown ordering preset'):
        ordering_profile('../private')
    with pytest.raises(SystemExit):
        main(['harmonize-rank', '--records', 'r.json', '--profile', 'private.json',
              '--preset', 'harmonic-next-v1', '--current', 'a', '--work', 'w'])


@pytest.mark.parametrize('configured,explicit,expected', [
    (None, None, 'zhilin'), ('zhilin', None, 'zhilin'), ('zhilin', 'default', 'default'),
    ('default', None, 'default'), ('zhilin', 'sync', 'default'),
])
def test_make_style_dispatch_and_explicit_twenty_track_selection(tmp_path, monkeypatch, configured, explicit, expected):
    import superdj.make as module
    import superdj.cli
    music = tmp_path / 'music'
    music.mkdir()
    sources = []
    for index in range(21):
        source = music / f'{index:02}.flac'
        source.write_bytes(b'fixture: dispatch only, not decoded audio')
        sources.append(str(source))
    runtime = {'audamcp': 'prepared-processor', 'bitwize': 'prepared-master'}
    if configured is not None:
        runtime['default_style'] = configured
    monkeypatch.setattr(module, 'runtime_config', lambda: runtime)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=b'{"streams":[{"sample_rate":"48000"}]}'))
    calls = []
    monkeypatch.setattr(superdj.cli, 'main', lambda args: calls.append(args))
    module.make(Namespace(inputs=sources[:20], style=explicit, output=str(tmp_path / 'out'),
                          work=str(tmp_path / 'work'), name='friend-set', keep_order=False))
    argv = calls[0]
    assert argv[argv.index('--style') + 1] == expected
    playlist = json.loads((tmp_path / 'work/playlist.json').read_text(encoding='utf-8'))
    assert len(playlist['tracks']) == 20
    assert {row['path'] for row in playlist['tracks']} == set(sources[:20])
    # A larger folder needs a separate selection: collecting it retains all inputs.
    assert len(module.collect([str(music)])) == 21
