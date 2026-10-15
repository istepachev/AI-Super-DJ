"""Observable ordinary-flow defaults, independent of private workstation state."""
import json
import hashlib
from argparse import Namespace
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from superdj.cli import main
from superdj.defaults import planning_defaults
from superdj.planner import plan_tracks, transition_score


def feature(identity, key='8A', bpm=124, intensity=.5):
    return dict(id=identity, key_camelot=key, bpm=bpm, intensity=intensity,
        bass_intensity=intensity, key_confidence=1., bpm_confidence=1.)


@pytest.mark.parametrize('runtime,explicit,expected', [
    ({}, {}, ('zhilin','mood','build')),
    ({}, {'style':'sync'}, ('default','classic','build')),
    ({}, {'style':'zhilin','harmonic_mode':'classic'}, ('zhilin','classic','build')),
    ({'default_style':'zhilin','default_harmonic_mode':'classic'}, {}, ('zhilin','classic','build')),
    ({'default_style':'default'}, {}, ('default','classic','build')),
    ({'default_style':'sync','default_harmonic_mode':'fuzzy','default_energy_arc':'wave'},
        {}, ('default','fuzzy','wave')),
    ({'default_style':'zhilin','default_harmonic_mode':'fuzzy'},
        {'style':'sync','harmonic_mode':'mood','energy_arc':'release'}, ('default','mood','release')),
])
def test_portable_defaults_and_explicit_runtime_precedence(runtime, explicit, expected):
    result=planning_defaults(runtime=runtime,**explicit)
    assert tuple(result[k] for k in ('style','harmonic_mode','energy_arc'))==expected


def test_invalid_runtime_is_not_silently_replaced():
    with pytest.raises(ValueError,match='default_harmonic_mode'):
        planning_defaults(runtime={'default_harmonic_mode':'invented'})


def test_mood_changes_actual_directed_order_and_fuzzy_remains_optional():
    rows=[feature('start','8A'),feature('boost','10A'),feature('diagonal','9B')]
    mood=plan_tracks(rows,first_id='start',energy_arc='none',harmonic_mode='mood')
    fuzzy=plan_tracks(rows,first_id='start',energy_arc='none',harmonic_mode='fuzzy')
    assert mood['harmonic_mode']=='mood'
    assert mood['order']==['start','boost','diagonal']
    assert fuzzy['order']==['start','diagonal','boost']
    assert 'energy_boost' in mood['edges'][0]['harmonic_relation']
    assert transition_score(rows[0],rows[1],harmonic_mode='mood')['cost'] < transition_score(rows[0],rows[1],harmonic_mode='fuzzy')['cost']
    fixed=plan_tracks(rows,mode='fixed',fixed_order=['start','diagonal','boost'],first_id='start',last_id='boost')
    assert fixed['order']==['start','diagonal','boost']


def test_classic_cost_and_unknown_keys_are_explicit():
    edge=transition_score(feature('a','8A'),feature('b','8B'),harmonic_mode='classic')
    assert edge['harmonic_relation']=='relative major/minor'
    unknown=feature('unknown')
    unknown.pop('key_camelot')
    planned=plan_tracks([unknown,feature('known')])
    assert any('Key unknown' in warning for warning in planned['uncertainty']['warnings'])
    assert 'unknown' in planned['edges'][0]['harmonic_relation']


def test_bare_plan_twenty_uses_mood_build_and_zhilin_sections(tmp_path,monkeypatch):
    import superdj.cues
    monkeypatch.setattr(superdj.cues,'runtime_config',lambda:{})
    rows=[]
    for i in reversed(range(20)):
        bpm=120+i*.5
        beats=np.arange(.25,420-.05,60/bpm)
        row=feature(f't{i:02}',bpm=bpm,intensity=.1+.8*i/19)
        row.update(duration=420,beats=beats.tolist(),downbeats=beats[::4].tolist(),
            grid_confidence=1.,grid_status='reviewed',grid={'meter':4,'human_reviewed':True},
            cue_candidates=[{'time':float(t),'kind':'body','novelty':.8,'confidence':.9,
                'sustained_midband_proxy':.1} for t in beats[::32]])
        rows.append(row)
    source=tmp_path/'records.json'
    source.write_text(json.dumps(rows),encoding='utf-8')
    main(['plan','--records',str(source),'--work',str(tmp_path/'plan')])
    result=json.loads((tmp_path/'plan/plan.json').read_text(encoding='utf-8'))
    assert result['style']=='zhilin' and result['harmonic_mode']=='mood'
    assert result['set_arc']['shape']=='build'
    assert result['order'][0]=='t00' and result['order'][-1]=='t19'
    assert len(result['order'])==len(set(result['order']))==20
    assert all(edge['overlap_bars']==32 for edge in result['edges'])
    assert all(180<=section['predicted_solo_seconds']<=240 for section in result['sections'].values())
    assert all(section['predicted_solo_seconds']<=300 for section in result['sections'].values())
    assert 'packaged' in result['mixing_preferences']['curve_profile_status']
    assert result['workflow_preferences']['preference_sources']['style']=='portable default'


def test_order_only_style_propagates_and_explicit_override_keeps_order(tmp_path,monkeypatch):
    import superdj.cues
    monkeypatch.setattr(superdj.cues,'runtime_config',lambda:{'default_style':'zhilin'})
    rows=[feature('a','8A'),feature('c','9B'),feature('b','10A')]
    source=tmp_path/'records.json'
    source.write_text(json.dumps(rows),encoding='utf-8')
    main(['plan','--records',str(source),'--work',str(tmp_path/'fixed'),'--mode','fixed',
        '--style','sync','--harmonic-mode','fuzzy','--energy-arc','none'])
    result=json.loads((tmp_path/'fixed/plan.json').read_text(encoding='utf-8'))
    assert result['style']=='default' and result['harmonic_mode']=='fuzzy'
    assert result['order']==['a','c','b'] and result['set_arc']['shape']=='none'


def test_make_dispatch_carries_resolved_defaults_and_keep_order(tmp_path,monkeypatch):
    import superdj.make as make_module
    import superdj.cli
    files=[]
    for i in range(3):
        path=tmp_path/f'{i}.flac'
        path.write_bytes(b'dispatch fixture, never decoded')
        files.append(str(path))
    monkeypatch.setattr(make_module,'runtime_config',lambda:{'audamcp':'processor','bitwize':'master'})
    monkeypatch.setattr(make_module.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=b'{"streams":[{"sample_rate":"48000"}]}'))
    calls=[]
    monkeypatch.setattr(superdj.cli,'main',lambda argv:calls.append(argv))
    make_module.make(Namespace(inputs=files,style=None,keep_order=True,name='default-flow',
        output=str(tmp_path/'out'),work=str(tmp_path/'work')))
    argv=calls[0]
    assert argv[argv.index('--style')+1]=='zhilin'
    assert argv[argv.index('--harmonic-mode')+1]=='mood'
    assert argv[argv.index('--energy-arc')+1]=='build'
    assert argv[argv.index('--mode')+1]=='fixed'
    playlist=json.loads((tmp_path/'work/playlist.json').read_text(encoding='utf-8'))
    assert [r['path'] for r in playlist['tracks']]==files


def test_missing_configured_reference_has_actionable_error_and_no_download(tmp_path,monkeypatch):
    import superdj.cues
    from superdj.zhilin import reference_profile
    monkeypatch.setattr(superdj.cues,'runtime_config',lambda:{'zhilin_profile':str(tmp_path/'missing-reference.json')})
    with pytest.raises(ValueError,match='remove that setting'):
        reference_profile()


def test_fresh_zhilin_reference_is_exact_approved_packaged_profile(monkeypatch):
    import superdj.cues
    from superdj.zhilin import reference_profile
    from superdj.mixing import envelopes
    monkeypatch.setattr(superdj.cues,'runtime_config',lambda:{})
    approved_path=Path(__file__).resolve().parents[1]/'assets'/'zhilin-reference-v1.json'
    approved_raw=approved_path.read_bytes()
    packaged=files('superdj').joinpath('profiles','zhilin-reference-v1.json')
    assert packaged.read_bytes()==approved_raw
    approved=json.loads(approved_raw)
    fresh=reference_profile()
    assert fresh['profile_source']=='packaged'
    assert fresh['profile_sha256']==hashlib.sha256(approved_raw).hexdigest()
    assert {key:fresh[key] for key in approved}==approved
    gains,report=envelopes(513,128,style='zhilin')
    explicit_gains,explicit_report=envelopes(513,128,style='zhilin',profile=approved_path)
    assert all(np.array_equal(gains[key],explicit_gains[key]) for key in gains)
    assert report==explicit_report


def test_reference_explicit_path_overrides_runtime_and_runtime_overrides_package(tmp_path):
    from superdj.zhilin import reference_profile
    approved=reference_profile(runtime={})
    custom={key:value for key,value in approved.items() if not key.startswith('profile_')}
    custom['name']='Explicit reference test fixture'
    path=tmp_path/'custom-reference.json'
    path.write_text(json.dumps(custom),encoding='utf-8')
    explicit=reference_profile(path,runtime={'zhilin_profile':str(tmp_path/'missing.json')})
    configured=reference_profile(runtime={'zhilin_profile':str(path)})
    assert explicit['name']==configured['name']==custom['name']
    assert explicit['profile_source']=='explicit' and configured['profile_source']=='runtime'
    assert explicit['profile_sha256']==configured['profile_sha256']
    assert explicit['profile_sha256']!=approved['profile_sha256']


def test_default_ordering_retains_original_general_costs_and_build():
    relative=transition_score(feature('a','8A'),feature('b','8B'))
    distant=transition_score(feature('a','8A'),feature('b','10A'))
    assert relative['components']['harmonic']==pytest.approx(.16)
    assert distant['components']['harmonic']==pytest.approx(1.22)
    rows=[feature('high',intensity=.8),feature('low',intensity=.2),feature('middle',intensity=.5)]
    result=plan_tracks(rows)
    assert result['order']==['low','middle','high']
    assert result['harmonic_mode']=='classic' and result['set_arc']['shape']=='build'
    assert result['total_cost']==pytest.approx(.15)


def test_fresh_make_requires_real_audio_engines_without_downloading(tmp_path,monkeypatch):
    import superdj.make as make_module
    sources=[]
    for i in range(2):
        path=tmp_path/f'{i}.flac'
        path.write_bytes(b'entry-point readiness fixture')
        sources.append(str(path))
    monkeypatch.setattr(make_module,'runtime_config',lambda:{})
    def no_external_call(*args,**kwargs):
        raise AssertionError('Missing engines must stop before any external tool or download')
    monkeypatch.setattr(make_module.subprocess,'run',no_external_call)
    with pytest.raises(ValueError,match='Run setup'):
        make_module.make(Namespace(inputs=sources,style=None,keep_order=False,name='unprepared',
            output=str(tmp_path/'out'),work=str(tmp_path/'work')))


def test_single_file_still_uses_mastering_branch_not_set_ordering(tmp_path,monkeypatch):
    import shutil
    import soundfile as sf
    import superdj.make as make_module
    import superdj.adapters
    import superdj.quality
    import superdj.cli
    source=tmp_path/'single.wav'
    sf.write(source,np.zeros((800,2),dtype=np.float32),8000,subtype='FLOAT')
    calls=[]
    monkeypatch.setattr(make_module,'runtime_config',lambda:{'audamcp':'processor','bitwize':'master'})
    monkeypatch.setattr(superdj.quality,'measure_audio',lambda *a,**kw:{'sample_rate':8000,'decoded_samples':800})
    def prepare(records,*args,**kwargs):
        calls.append('prepare')
        return {records[0]['id']:{'path':str(source)}}
    def master(prepared,destination,*args,**kwargs):
        calls.append('master')
        shutil.copyfile(prepared,destination)
        return {'branch':'single-file dispatch fixture'}
    monkeypatch.setattr(superdj.adapters,'prepare_tracks',prepare)
    monkeypatch.setattr(superdj.adapters,'master_set',master)
    monkeypatch.setattr(superdj.quality,'export_audio',lambda *a,**kw:{'flac':{'path':'dispatch-only.flac','decoded_samples':800}})
    def no_set_planning(*args,**kwargs):
        raise AssertionError('Single-file enhancement must not enter set planning')
    monkeypatch.setattr(superdj.cli,'main',no_set_planning)
    report=make_module.make(Namespace(inputs=[str(source)],style=None,keep_order=False,name='single',
        output=str(tmp_path/'out'),work=str(tmp_path/'work')))
    assert calls==['prepare','master']
    assert report['source']['decoded_samples']==800
