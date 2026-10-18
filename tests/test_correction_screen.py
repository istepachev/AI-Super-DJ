"""Real backend call elimination without changing correction search or PCM."""
from copy import deepcopy
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from superdj.correction_screen import CorrectionScreen


def warp(offset=0):
    return {'output_frames':1600,'anchors':[
        {'source_frame':0,'output_frame':0},
        {'source_frame':800,'output_frame':800+offset},
        {'source_frame':1600,'output_frame':1600}]}


def test_repeated_map_uses_current_pcm_then_best_snapshot_without_extra_audio(tmp_path):
    output=tmp_path/'current.wav';best=tmp_path/'best.wav'
    output.write_bytes(b'already audited map A');shutil.copy2(output,best)
    screen=CorrectionScreen();screen.rendered(warp(),output);screen.saved_best(warp(),best)
    assert screen.reuse(warp(),output)
    assert screen.reuse(warp(1),output) is None
    output.write_bytes(b'new map B');screen.rendered(warp(1),output)
    assert screen.reuse(warp(),output)
    assert output.read_bytes()==b'already audited map A'
    assert len(list(tmp_path.iterdir()))==2
    assert screen.report()['same_map_backend_calls_skipped']==1
    assert screen.report()['best_map_backend_calls_reused']==1


@pytest.mark.parametrize('flag',['source_attack_controls_unapplied','native_attack_preservation','fine_attack_timing'])
def test_pcm_processing_outside_backend_map_cannot_reuse_an_old_snapshot(tmp_path,flag):
    output=tmp_path/'output.wav';best=tmp_path/'best.wav';output.write_bytes(b'backend PCM');shutil.copy2(output,best)
    screen=CorrectionScreen();screen.rendered(warp(),output);screen.saved_best(warp(),best)
    candidate=warp();candidate[flag]={'head':True}
    assert screen.reuse(candidate,output) is None


def test_modified_or_missing_private_pcm_does_not_count_as_a_rendered_map(tmp_path):
    output=tmp_path/'output.wav';output.write_bytes(b'original')
    screen=CorrectionScreen();screen.rendered(warp(),output)
    output.write_bytes(b'changed PCM')
    assert screen.reuse(warp(),output) is None
    output.unlink()
    assert screen.reuse(warp(),output) is None


def test_changed_best_snapshot_and_changed_output_length_are_not_reused(tmp_path):
    output=tmp_path/'output.wav';best=tmp_path/'best.wav';output.write_bytes(b'original');shutil.copy2(output,best)
    screen=CorrectionScreen();screen.rendered(warp(),output);screen.saved_best(warp(),best)
    changed=warp();changed['output_frames']+=1
    assert screen.reuse(changed,output) is None
    output.write_bytes(b'another map');screen.rendered(warp(2),output)
    best.write_bytes(b'changed best PCM')
    assert screen.reuse(warp(),output) is None


@pytest.mark.parametrize('cycle',[False,True])
def test_repeated_failed_corrections_keep_all_attempts_and_identical_actual_pcm(tmp_path,monkeypatch,cycle):
    from test_sync import RenderedSyncTests
    from superdj import kick_grid,sync,render
    from superdj.audio import ffmpeg_executable
    from superdj.planner import plan_tracks
    rubber=os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not rubber:pytest.skip('Actual Rubber Band required')
    records=RenderedSyncTests().fixtures(tmp_path,[120,122]);plan=plan_tracks(records,mode='fixed')
    real_audit=kick_grid.rendered_attack_audit;real_run=render._run
    observations={};calls=[];mode=['baseline']
    def audit(path,mapping,sample_rate):
        value=real_audit(path,mapping,sample_rate)
        key=(mode[0],str(path));count=observations.get(key,0);observations[key]=count+1
        # Exercise the existing ten-step search, then retain real independent
        # final attack checks. No backend/audio operation is mocked.
        if count<10:value={**value,'maximum_measured_drift_seconds':.006}
        return value
    def compensate(mapping,*args,**kwargs):
        if not cycle:return
        for anchor in mapping['anchors']:
            if anchor['role'].endswith('_sync_control'):
                base=anchor.setdefault('test_original_output',anchor['output_frame'])
                anchor['output_frame']=base+(0 if anchor['output_frame']!=base else 1)
                anchor['output_seconds']=anchor['output_frame']/16000
    def run(command,**kwargs):
        if str(command[0])==str(rubber):calls.append(mode[0])
        return real_run(command,**kwargs)
    monkeypatch.setattr(kick_grid,'rendered_attack_audit',audit)
    monkeypatch.setattr(sync,'compensate_backend_drift',compensate)
    monkeypatch.setattr(render,'_run',run)
    results=[];failures=[]
    for label in ('baseline','candidate'):
        mode[0]=label
        with monkeypatch.context() as patch:
            if label=='baseline':patch.setattr(CorrectionScreen,'reuse',lambda *args:None)
            from superdj.transition_quality import TransitionFailure
            try:
                results.append(render.render_plan(plan,records,{},tmp_path/f'{label}.wav',tmp_path/label,
                    ffmpeg=ffmpeg_executable(),rubberband=rubber,sample_rate=16000,tempo_conflict_policy='sync'))
            except TransitionFailure as error:failures.append(error)
    if failures:
        # An unchanged deliberately uncorrected backend can exceed the new
        # ordinary bound. Caching must retain the same rejected PCM and all
        # ten attempts; it must never turn rejection into a released master.
        assert len(failures)==2 and not results
        assert [e.evidence['failed_track'] for e in failures]==['a','a']
        assert all('attack recognition' in str(e) for e in failures)
        assert not (tmp_path/'baseline.wav').exists() and not (tmp_path/'candidate.wav').exists()
        np.testing.assert_array_equal(sf.read(tmp_path/'baseline/track-01-warped.wav')[0],
                                      sf.read(tmp_path/'candidate/track-01-warped.wav')[0])
        assert calls.count('baseline')==10
        assert calls.count('candidate')<calls.count('baseline')
        return
    np.testing.assert_array_equal(sf.read(tmp_path/'baseline.wav',dtype='float32')[0],sf.read(tmp_path/'candidate.wav',dtype='float32')[0])
    assert calls.count('baseline')==20
    assert calls.count('candidate')<calls.count('baseline')
    for before,after in zip(results[0]['tracks'],results[1]['tracks']):
        b=before['warp']['post_render_transient_audit'];a=after['warp']['post_render_transient_audit']
        assert len(a['iterations'])==len(b['iterations'])==10
        assert a['final']==b['final']
        screen=a['duplicate_map_screen']
        assert screen['same_map_backend_calls_skipped']+screen['best_map_backend_calls_reused']>0


def test_duplicate_screen_changes_invalidate_clip_cache(tmp_path):
    from pathlib import Path
    import superdj.correction_screen as module
    from superdj.execution import stage_fingerprint
    source=tmp_path/'superdj';shutil.copytree(Path(module.__file__).parent,source,ignore=shutil.ignore_patterns('__pycache__'))
    before=stage_fingerprint('clip',source)
    path=source/'correction_screen.py';path.write_text(path.read_text(encoding='utf-8')+'\n# changed screen\n',encoding='utf-8')
    assert stage_fingerprint('clip',source)!=before


def test_source_recovery_moves_only_the_failed_boundary_and_honors_its_lock():
    from test_reliability import source_plan
    from superdj.recovery import alternative
    from superdj.transition_quality import TransitionFailure
    from superdj.render import RenderError
    rows,plan=source_plan(2)
    failure=TransitionFailure('0','1','bad source timing',{
        'stage':'source-only timing preflight','failed_role':'tail'})
    revised,report=alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert report['changed_track']=='0' and report['boundary']=='end'
    assert revised['sections']['1']['start']==plan['sections']['1']['start']
    plan['sections']['0']['locked_end']=True
    with pytest.raises(RenderError,match='No alternative'):
        alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)


@pytest.mark.parametrize('error_type',['source','io'])
def test_source_only_failure_happens_before_that_decks_backend_and_keeps_io_errors(tmp_path,monkeypatch,error_type):
    from test_sync import RenderedSyncTests
    from superdj import render,kick_grid
    from superdj.planner import plan_tracks
    from superdj.audio import ffmpeg_executable
    from superdj.timing_safety import SourceWindowFailure
    from superdj.transition_quality import TransitionFailure
    rubber=os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not rubber:pytest.skip('Actual Rubber Band required')
    records=RenderedSyncTests().fixtures(tmp_path,[120,122]);run=render._run;calls=[]
    original=kick_grid.refine_source_controls
    def refine(path,mapping,*args):
        if mapping['input_output_beat_mapping']['head']:
            if error_type=='io':raise OSError('unreadable source estimate')
            raise SourceWindowFailure('measured source map exceeds its rate bound',role='head')
        return original(path,mapping,*args)
    def execute(command,**kwargs):
        if str(command[0])==str(rubber):calls.append(command)
        return run(command,**kwargs)
    monkeypatch.setattr(kick_grid,'refine_source_controls',refine);monkeypatch.setattr(render,'_run',execute)
    expected=TransitionFailure if error_type=='source' else OSError
    with pytest.raises(expected) as caught:
        render.render_plan(plan_tracks(records,mode='fixed'),records,{},tmp_path/'set.wav',tmp_path/'render',
            ffmpeg=ffmpeg_executable(),rubberband=rubber,sample_rate=16000,tempo_conflict_policy='sync')
    assert not any('track-02-timemap.txt' in command for command in calls)
    if error_type=='source':
        error=caught.value
        assert (error.outgoing,error.incoming)==('a','b')
        assert error.evidence['stretch_not_started'] and not error.evidence['timing_gate_relaxed']


def test_measured_bad_source_window_recovers_automatically_with_real_final_checks(tmp_path,monkeypatch):
    from test_speed_pipeline import _long_records
    from superdj import kick_grid
    from superdj.audio import ffmpeg_executable
    from superdj.planner import plan_tracks
    from superdj.compact import arrange_sync
    from superdj.recovery import render_with_recovery
    from superdj.timing_safety import SourceWindowFailure
    rubber=os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not rubber:pytest.skip('Actual Rubber Band required')
    decoder=ffmpeg_executable();rows=_long_records(tmp_path,[120,122])
    plan=arrange_sync(plan_tracks(rows,mode='fixed'),rows,sample_rate=16000,ffmpeg=decoder)
    original=kick_grid.refine_source_controls;rejected=[]
    def refine(path,mapping,*args):
        points=mapping['input_output_beat_mapping']['head']
        if points and not rejected:
            rejected.append(points[0]['source_seconds'])
            raise SourceWindowFailure('measured primary phase is ambiguous',role='head')
        return original(path,mapping,*args)
    monkeypatch.setattr(kick_grid,'refine_source_controls',refine)
    rendered,effective=render_with_recovery(plan,rows,{},tmp_path/'set.wav',tmp_path/'render',
        ffmpeg=decoder,rubberband=rubber,sample_rate=16000,tempo_conflict_policy='sync',max_intro_trim_seconds=1000)
    assert effective['order']==plan['order']
    assert effective['sections']['1']['start']!=rejected[0]
    assert [e['overlap_bars'] for e in effective['edges']]==[e['overlap_bars'] for e in plan['edges']]
    assert rendered['window_recovery']['alternatives_used']==1
    assert rendered['transitions'][0]['quality_gate']['passed']
    assert all(t['warp']['actual_timing_rate_audit']['passed'] for t in rendered['tracks'])
