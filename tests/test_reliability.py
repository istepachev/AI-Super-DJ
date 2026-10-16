"""Observable recovery, resource admission, revision and exact-DSP regressions."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from superdj.compact import arrange_sync
from superdj.planner import plan_tracks
from superdj.render import RenderError
from superdj.recovery import alternative, render_with_recovery
from superdj.revision import revised_plan, validated_records
from superdj.transition_quality import audit_energy, audit_vocals, reliability, TransitionFailure
from superdj.resources import preflight, GIB
from test_new_workflow import record


def source_plan(count=3):
    rows = [record(str(i),480) for i in range(count)]
    plan = arrange_sync(plan_tracks(rows,mode='fixed',harmonic_mode='mood'),rows,zhilin=True)
    plan['mixing_preferences']={'target_play_seconds':210.,'maximum_play_seconds':300.}
    return rows,plan


def test_alternative_changes_one_unlocked_endpoint_and_preserves_order_lengths_and_protection():
    rows,plan=source_plan()
    before=deepcopy(plan)
    error=TransitionFailure('0','1','test actual blend failed')
    new,evidence=alternative(plan,rows,error,{},1,sample_rate=48000,ffmpeg=None)
    assert plan==before and new['order']==plan['order']
    assert new['sections']['1']['start']!=plan['sections']['1']['start']
    assert new['sections']['1']['end']==plan['sections']['1']['end']
    assert new['sections']['0']['start']==plan['sections']['0']['start']
    assert [e['overlap_bars'] for e in new['edges']]==[e['overlap_bars'] for e in plan['edges']]
    assert all(s['predicted_play_seconds']<=300 for s in new['sections'].values())
    assert evidence['order_preserved'] and not new['sections']['1']['locked_start']


def test_no_window_retry_can_move_user_locks_or_relax_source_budget():
    rows,plan=source_plan(2)
    plan['sections']['0']['locked_end']=True
    plan['sections']['1']['locked_start']=True
    with pytest.raises(RenderError,match='No alternative'):
        alternative(plan,rows,TransitionFailure('0','1','failed'),{},1,sample_rate=48000,ffmpeg=None)


def test_recovery_moves_the_failed_tail_instead_of_an_unrelated_entrance():
    rows,plan=source_plan(2)
    failure=TransitionFailure('0','1','tail attack identity lost',{'failed_track':'0','failed_roles':['tail']})
    new,evidence=alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert evidence['changed_track']=='0' and evidence['boundary']=='end'
    assert new['sections']['1']['start']==plan['sections']['1']['start']
    assert new['sections']['0']['end']!=plan['sections']['0']['end']


@pytest.mark.parametrize('identity,attempt,boundary',[('0',1,'end'),('1',2,'start')])
def test_two_failed_roles_still_recover_the_actual_failed_source(identity,attempt,boundary):
    rows,plan=source_plan(2)
    failure=TransitionFailure('0','1','both roles unresolved',{'failed_track':identity,'failed_roles':['head','tail']})
    updated,evidence=alternative(plan,rows,failure,{},attempt,sample_rate=48000,ffmpeg=None)
    assert evidence['changed_track']==identity and evidence['boundary']==boundary
    other='1' if identity=='0' else '0'
    assert updated['sections'][other]['start']==plan['sections'][other]['start']
    assert updated['sections'][other]['end']==plan['sections'][other]['end']


def test_no_recognized_output_preserves_failed_clip_identity_for_recovery(tmp_path,monkeypatch):
    import os
    from superdj import kick_grid
    from superdj.render import render_plan
    from test_speed_pipeline import _long_records
    ffmpeg,rubberband=os.getenv('SUPERDJ_FFMPEG'),os.getenv('SUPERDJ_RUBBERBAND')
    if not ffmpeg or not rubberband:pytest.skip('Actual audio backends required')
    rows=_long_records(tmp_path,[120,120]);plan=plan_tracks(rows,mode='fixed');plan['style']='zhilin'
    audit={'measured':[],'unverified':[],'maximum_measured_drift_seconds':None,'all_interior_beats_verified':False}
    monkeypatch.setattr(kick_grid,'rendered_attack_audit',lambda *a,**kw:deepcopy(audit))
    with pytest.raises(TransitionFailure) as exc:
        render_plan(plan,rows,{r['id']:r['path'] for r in rows},tmp_path/'out.wav',tmp_path/'work',
            ffmpeg=ffmpeg,rubberband=rubberband,sample_rate=16000)
    assert exc.value.evidence['failed_track']=='0'
    assert exc.value.evidence['failed_roles']==['tail']
    assert exc.value.evidence['final_attack']['maximum_measured_drift_seconds'] is None


def test_recovery_does_not_renumber_the_same_unverified_required_event():
    rows,plan=source_plan(2);start=plan['sections']['1']['start'];stamp=start+10
    evidence={'failed_track':'1','failed_roles':['head'],
        'warp':{'source_kick_alignment':{'head':{'events':[{'beat':21,'source_seconds':stamp}],
            'low_frequency_gap_recovery':{'recovered_beats':[21]}}}},'final_attack':{'measured':[]}}
    new,selection=alternative(plan,rows,TransitionFailure('0','1','mandatory bass identity lost',evidence),{},1,sample_rate=48000,ffmpeg=None)
    new_start=new['sections']['1']['start'];window_end=new_start+plan['edges'][0]['overlap_bars']*4*60/rows[1]['bpm']
    assert not new_start<=stamp<=window_end
    assert new['rejected_attack_window_events']['1']['head']==[stamp]
    assert new['order']==plan['order']
    assert [e['overlap_bars'] for e in new['edges']]==[e['overlap_bars'] for e in plan['edges']]


def test_recovery_does_not_renumber_a_missing_visible_independent_bass_signature():
    rows,plan=source_plan(2);stamp=plan['sections']['1']['start']+10
    evidence={'failed_track':'1','failed_roles':['head'],
        'warp':{'source_kick_alignment':{'head':{'events':[{'beat':21,'source_seconds':stamp}],
            'low_frequency_gap_recovery':{'recovered_beats':[21]},
            'native_mixture_reference':{'bass_visible_beats':[21]}}}},
        'final_attack':{'measured':[{'role':'head','beat':21,'view':'bass_body','delay_seconds':0.}]}}
    new,_=alternative(plan,rows,TransitionFailure('0','1','independent recovered bass identity missing',evidence),{},1,sample_rate=48000,ffmpeg=None)
    assert new['rejected_attack_window_events']['1']['head']==[stamp]
    head=new['sections']['1']['start'];end=head+plan['edges'][0]['overlap_bars']*4*60/rows[1]['bpm']
    assert not head<=stamp<=end


def test_recovery_does_not_keep_a_late_primary_event_in_the_next_window():
    rows,plan=source_plan(2);stamp=plan['sections']['1']['start']+10
    evidence={'failed_track':'1','failed_roles':['head'],
        'warp':{'source_kick_alignment':{'head':{'events':[{'beat':21,'source_seconds':stamp}]}}},
        'final_attack':{'measured':[{'role':'head','beat':21,'view':'primary','delay_seconds':.009}]}}
    new,_=alternative(plan,rows,TransitionFailure('0','1','late source attack',evidence),{},1,sample_rate=48000,ffmpeg=None)
    assert new['rejected_attack_window_events']['1']['head']==[stamp]
    head=new['sections']['1']['start'];end=head+plan['edges'][0]['overlap_bars']*4*60/rows[1]['bpm']
    assert not head<=stamp<=end
    assert new['order']==plan['order']
    assert [e['overlap_bars'] for e in new['edges']]==[e['overlap_bars'] for e in plan['edges']]


def test_recovery_rechecks_the_changed_window_and_is_bounded(tmp_path):
    rows,plan=source_plan(2);calls=[]
    def render(candidate,*args,**kwargs):
        calls.append(deepcopy(candidate))
        if len(calls)<2:raise TransitionFailure('0','1','injected energy hole',{'passed':False})
        return {'verified':True}
    result,effective=render_with_recovery(plan,rows,{},tmp_path/'mix.wav',tmp_path/'work',renderer=render,sample_rate=48000)
    assert len(calls)==2 and calls[0]['sections']!=calls[1]['sections']
    assert result['verified'] and effective['order']==plan['order']
    assert result['window_recovery']['constraints_relaxed'] is False
    calls.clear()
    def fail(candidate,*args,**kwargs):
        calls.append(candidate);raise TransitionFailure('0','1','unresolved')
    with pytest.raises(TransitionFailure):
        render_with_recovery(plan,rows,{},tmp_path/'other.wav',tmp_path/'bad',renderer=fail,maximum_attempts=3,sample_rate=48000)
    assert len(calls)<=3


def test_energy_gate_rejects_sustained_rendered_hole_but_preserves_native_break():
    rate=4000;t=np.arange(rate*12)/rate
    x=np.repeat((.15*np.sin(2*np.pi*55*t)+.1*np.sin(2*np.pi*510*t))[:,None],2,axis=1).astype('float32')
    assert audit_energy(x,x,x,rate)['passed']
    broken=x.copy();broken[rate*5:rate*8]=0
    assert not audit_energy(x,x,broken,rate)['passed']
    assert audit_energy(broken,broken,broken,rate)['passed']
    fill=x.copy();fill[rate*5:round(rate*5.5)]=0
    assert audit_energy(x,x,fill,rate)['passed']


def test_competing_voice_gate_uses_actual_handoff_and_retains_unknown_status():
    rate=4000;t=np.arange(rate*20)/rate
    a=np.repeat((.2*np.sin(2*np.pi*220*t))[:,None],2,axis=1)
    b=np.repeat((.2*np.sin(2*np.pi*330*t))[:,None],2,axis=1)
    edge={'beats':40,'vocal_processing':{'executed':False}}
    report=audit_vocals(a,b,.5*(a+b),rate,a,b,edge,style='zhilin',beat_phase=None)
    assert not report['passed'] and not report['absence_certified']
    edge['vocal_processing']['executed']=True
    assert audit_vocals(a,b,.5*(a+b),rate,a,b,edge,style='zhilin',beat_phase=None)['passed']
    assert audit_vocals(a,b,.5*(a+b),rate,None,b,edge,style='zhilin',beat_phase=None)['status'].startswith('unknown')


def test_coverage_does_not_turn_a_few_good_beats_into_complete_rhythm_evidence():
    final={'measured':[{'role':'tail','beat':i} for i in range(4)],
        'unverified':[{'role':'tail','beat':i} for i in range(4,128)]}
    clips=[{'id':'a','warp':{'post_render_transient_audit':{'final':final}}},{'id':'b'}]
    report=reliability(clips,{})
    assert not report['passed'] and report['musical_quality_probability'] is None
    assert report['decks'][0]['coverage_fraction']==4/128


def machine(memory=16*GIB):
    return {'cpu_threads':16,'available_memory_bytes':memory,'cuda_probe':'measured','gpus':[]}


def test_resource_admission_caps_jobs_aggregates_shared_disk_and_refuses_shortage(tmp_path):
    disk=lambda p:(tmp_path,'one-volume',1000*GIB)
    probe=lambda p:(480.,44100)
    report=preflight(['a','b'],output=tmp_path,scratch=tmp_path,cache=tmp_path,jobs=4,
        machine=machine(5*GIB),probe=probe,disk=disk)
    assert report['jobs']==1 and not report['audio_precision_changed']
    assert len(report['volumes'])==1
    with pytest.raises(OSError,match='disk space'):
        preflight(['a','b'],output=tmp_path,machine=machine(),probe=probe,disk=lambda p:(tmp_path,'one',GIB))
    with pytest.raises(MemoryError,match='RAM'):
        preflight(['a']*30,output=tmp_path,machine=machine(GIB),probe=probe,disk=disk)
    with pytest.raises(ValueError,match='unavailable'):
        preflight(['a'],output=tmp_path,devices=['cuda:1'],machine=machine(),probe=probe,disk=disk)


def test_unknown_memory_and_gpu_probe_do_not_claim_reservation(tmp_path):
    report=preflight(['a'],output=tmp_path,jobs=4,machine={'cpu_threads':2,'available_memory_bytes':None,
        'cuda_probe':'unknown','gpus':[]},probe=lambda p:(60.,44100),disk=lambda p:(tmp_path,'one',100*GIB))
    assert report['jobs']==1 and not report['estimate_is_reservation']


def test_actual_master_admission_refuses_large_set_before_allocation():
    from superdj.resources import admit_master
    with pytest.raises(MemoryError,match='complete master'):
        admit_master(44100*7200,44100,memory=8*GIB)
    assert admit_master(44100*120,44100,memory=8*GIB)['reservation'] is False


def test_removal_replans_suffix_and_rescores_new_neighbours_without_changing_prefix():
    rows,plan=source_plan(5)
    original=deepcopy(plan)
    manifest={'plan':plan,'selection_and_analysis':rows}
    kept,new=revised_plan(manifest,drop=['2'],sample_rate=48000)
    assert plan==original
    assert new['order'][:2]==['0','1'] and set(new['order'])=={'0','1','3','4'}
    assert [e['from'] for e in new['edges']]==new['order'][:-1]
    assert [e['to'] for e in new['edges']]==new['order'][1:]
    assert new['harmonic_mode']=='mood' and new['revision']['suffix_replanned']
    assert {r['id'] for r in kept}==set(new['order'])
    assert all(s['predicted_play_seconds']<=300 for s in new['sections'].values())


def test_transition_revision_keeps_order_and_removal_rejects_unknown_ids():
    rows,plan=source_plan(3);manifest={'plan':plan,'selection_and_analysis':rows}
    _,new=revised_plan(manifest,transition=['0','1'],sample_rate=48000)
    assert new['order']==plan['order'] and new['sections']!=plan['sections']
    with pytest.raises(ValueError,match='existing track'):
        revised_plan(manifest,drop=['missing'])


def test_revision_refuses_changed_original_even_if_old_analysis_exists(tmp_path):
    from superdj.execution import execution
    path=tmp_path/'source.wav';path.write_bytes(b'original')
    manifest={'selection_and_analysis':[{'id':'a','path':str(path),
        'provenance':{'source_sha256':hashlib.sha256(b'original').hexdigest()}}]}
    with execution() as session:
        assert validated_records(manifest,session,ffmpeg='ffmpeg')[0]['id']=='a'
        path.write_bytes(b'changed')
        with pytest.raises(ValueError,match='changed'):
            validated_records(manifest,session,ffmpeg='ffmpeg')


@pytest.mark.parametrize('coefficient',[0.,.5,.99999,1.])
def test_compiled_master_envelope_is_sample_exact(coefficient):
    pytest.importorskip('numba')
    from superdj.mastering import exact_release
    values=np.random.default_rng(451).uniform(.01,1.,50001)
    values[100:200]=1.;expected=np.empty_like(values);env=1.
    for i,target in enumerate(values):
        env=target if target<env else coefficient*env+(1-coefficient)*target
        expected[i]=env
    assert np.array_equal(exact_release(values,coefficient),expected)


def test_master_acceleration_failure_restores_upstream_function(monkeypatch):
    import superdj.mastering as module
    original=lambda x:x
    backend=SimpleNamespace(limit_peaks_lookahead=original)
    def fail(*args):raise ImportError('optional compiler unavailable')
    monkeypatch.setattr(module,'exact_release',fail)
    with module.accelerated_limiter(backend) as report:
        assert not report['enabled'] and backend.limit_peaks_lookahead is original
    assert backend.limit_peaks_lookahead is original
