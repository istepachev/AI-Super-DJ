"""Recovery and revision must address original audio, not a repeated timeline."""
from copy import deepcopy
from pathlib import Path
import hashlib

import numpy as np
import pytest
import soundfile as sf

from superdj.minimum import preserve_minimum
from superdj.minimum_timeline import (native_records,native_plan,native_prepared,
    native_time,native_failure,render_with_minimum_recovery)
from superdj.render import RenderError
from superdj.transition_quality import TransitionFailure
from test_minimum_stem_alignment import inputs,final_inputs


@pytest.mark.parametrize('final',[False,True])
def test_inverse_recovers_original_grid_plan_and_all_pcm_without_changing_inputs(tmp_path,final):
    _,rows,plan,prepared=(final_inputs if final else inputs)(tmp_path)
    row=next(r for r in rows if r['id']=='a')
    row['grid']['neural_beat_probability_by_beat']=np.linspace(.1,.99,len(row['beats'])).tolist()
    row['grid']['neural_downbeat_probability_by_beat']=np.linspace(.9,.2,len(row['beats'])).tolist()
    row['structure']=[{'time':v,'label':'x'} for v in row['beats'][::9]]
    row['cue_candidates']=[{'time':v,'seconds':v} for v in (2.,80.,120.)]
    row['provenance']={'source_sha256':'source-sha'}
    edited,extended,processed=preserve_minimum(plan,rows,prepared,tmp_path/'repeat')
    snapshot=deepcopy((edited,extended,processed))
    recovered=native_records(extended)
    for original,restored in zip(rows,recovered):
        np.testing.assert_allclose(restored['beats'],original['beats'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(restored['downbeats'],original['downbeats'],rtol=0,atol=1e-12)
        assert restored['duration']==original['duration']
        assert restored.get('provenance')==original.get('provenance')
    restored=next(r for r in recovered if r['id']=='a')
    assert restored['grid']==row['grid']
    for key in ('structure','cue_candidates'):
        for a,b in zip(restored[key],row[key]):assert a['time']==pytest.approx(b['time'],abs=1e-12)
    restored_plan=native_plan(edited)
    for key in ('end','source_span_seconds','predicted_solo_seconds','predicted_play_seconds'):
        assert restored_plan['sections']['a'][key]==pytest.approx(plan['sections']['a'][key],abs=1e-12)
    hashes={k:hashlib.sha256(Path(processed['a'][k]).read_bytes()).hexdigest() for k in ('path','vocal_path','drum_path')}
    restored_files=native_prepared(processed,tmp_path/'native')
    for key in hashes:
        original=sf.read(prepared['a'][key],dtype='float32')[0]
        actual=sf.read(restored_files['a'][key],dtype='float32')[0]
        assert np.array_equal(original,actual)
        assert hashlib.sha256(Path(processed['a'][key]).read_bytes()).hexdigest()==hashes[key]
    assert (edited,extended,processed)==snapshot


def test_legacy_repeat_recovers_mixture_and_voice_but_keeps_native_drums(tmp_path):
    from superdj.minimum import legacy_minimum
    _,rows,plan,prepared=inputs(tmp_path)
    plan['sections']['a']['minimum_play_seconds']=120.
    _,_,processed=legacy_minimum(plan,rows,prepared,tmp_path/'repeat')
    processed['a']['provenance']['timeline_edit'].pop('aligned_preparation_streams')
    processed['a']['drum_path']=prepared['a']['drum_path']
    restored=native_prepared(processed,tmp_path/'native')
    assert restored['a']['drum_path']==prepared['a']['drum_path']
    for key in ('path','vocal_path'):
        assert np.array_equal(sf.read(restored['a'][key])[0],sf.read(prepared['a'][key])[0])


def test_failed_events_map_to_native_positions_without_rewriting_rendered_evidence(tmp_path):
    _,rows,plan,prepared=inputs(tmp_path)
    _,extended,_=preserve_minimum(plan,rows,prepared,tmp_path/'repeat')
    edit=extended[0]['timeline_edit'];ins=edit['insertion_source_seconds'];length=edit['added_seconds']
    assert native_time(ins+length+.25,edit)==pytest.approx(ins+.25)
    assert native_time(ins+.25,edit)==pytest.approx(edit['loop_start_seconds']+.25)
    error=TransitionFailure('a','b','missed native attack',{'failed_track':'a','failed_roles':['tail'],
        'warp':{'source_kick_alignment':{'tail':{'events':[{'beat':1,'source_seconds':ins+length+.25}]}}},
        'final_attack':{'measured':[{'role':'tail','beat':1,'delay_seconds':.004}]}})
    original=deepcopy(error.evidence['warp'])
    native_failure(error,extended)
    assert error.evidence['canonical_attack_rejections']['tail']==pytest.approx([ins+.25])
    assert error.evidence['warp']==original
    native_failure(error,extended)
    assert error.evidence['canonical_attack_rejections']['tail']==pytest.approx([ins+.25])


def test_retry_repeats_afresh_and_searches_only_native_sources(tmp_path,monkeypatch):
    _,rows,plan,prepared=inputs(tmp_path)
    initial=deepcopy((rows,plan,prepared));calls=[]
    def render(candidate,records,files,*args,**kwargs):
        calls.append(files['a']['path'])
        assert records[0]['duration']>rows[0]['duration']
        assert candidate['sections']['a']['predicted_play_seconds']>plan['sections']['a']['predicted_play_seconds']>=60
        assert len({sf.info(files['a'][k]).frames for k in ('path','vocal_path','drum_path')})==1
        if len(calls)==1:raise TransitionFailure('a','b','test failed first window')
        return {'tracks':[]}
    def alternative(candidate,records,failure,*args,**kwargs):
        assert records==rows
        assert 'minimum_repeat' not in candidate['sections']['a']
        assert records[0]['duration']==125
        return deepcopy(candidate),{'test_native_search':True}
    monkeypatch.setattr('superdj.recovery.alternative',alternative)
    rendered,effective,extended,processed=render_with_minimum_recovery(plan,rows,prepared,
        tmp_path/'out.wav',tmp_path/'render',renderer=render)
    assert len(calls)==2 and calls[0]!=calls[1]
    assert rendered['window_recovery']['alternatives_used']==1
    assert processed['a']['path']==calls[1]
    assert extended[0]['duration']==125+effective['minimum_playback_edits'][0]['added_seconds']
    assert (rows,plan,prepared)==initial


def test_revision_returns_native_duration_before_reading_original_audio(tmp_path):
    from types import SimpleNamespace
    from superdj.revision import validated_records
    from superdj.rhythm import bass_profile
    from superdj.cues import runtime_config
    _,rows,plan,prepared=inputs(tmp_path)
    rows[0]['path']=prepared['a']['path']
    rows[0]['provenance']={'source_sha256':hashlib.sha256(Path(rows[0]['path']).read_bytes()).hexdigest()}
    _,extended,_=preserve_minimum(plan,rows,prepared,tmp_path/'repeat')
    session=SimpleNamespace(digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),
        map=lambda f,values:[f(v) for v in values])
    restored=validated_records({'selection_and_analysis':[extended[0]]},session,ffmpeg='ffmpeg')
    assert restored[0]['duration']==125
    # This decode raised "Source duration differs from timing analysis" before
    # restoring the native record; do not mock the source decoder.
    bass_profile(restored[0],ffmpeg=runtime_config().get('ffmpeg','ffmpeg'))


@pytest.mark.parametrize('broken',['method','duration','probabilities'])
def test_ambiguous_old_edits_are_rejected(tmp_path,broken):
    _,rows,plan,prepared=inputs(tmp_path)
    _,extended,_=preserve_minimum(plan,rows,prepared,tmp_path/'repeat')
    if broken=='method':extended[0]['timeline_edit']['method']='unknown splice'
    elif broken=='duration':extended[0]['duration']+=.1
    else:extended[0]['grid']['neural_beat_probability_by_beat']=[.5]*3
    with pytest.raises(RenderError):native_records(extended)
