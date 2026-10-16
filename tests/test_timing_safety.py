from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
import pytest
import soundfile as sf
from superdj.sync import shared_clock,compensate_backend_drift
from superdj.timing_safety import initialize_rate_reference,audit_warp_rates,classify_native_phase
from superdj.percussion import ownership_policy,percussion_gains,verification_components
from superdj.mixing import envelopes,blend


def mapping(rate=1.):
    return {'anchors':[{'source_frame':i*1000,'output_frame':round(i*1000/rate),
        'output_seconds':i/rate,'role':'head_sync_control'} for i in range(5)],
        'input_output_beat_mapping':{'head':[{'output_seconds':0.},{'output_seconds':4/rate}],'tail':[]}}


def test_matching_offbeat_shape_does_not_become_a_primary_grid_control():
    centres=np.arange(20)*.48+.2
    refs=[{'index':i,'time':float(t-.024)} for i,t in enumerate(centres) if i!=11]
    rows=refs+[{'index':11,'time':float(centres[11]+.096),'source_template_correlation':.999}]
    accepted,observed=classify_native_phase(rows,centres,refs)
    assert 11 not in {r['index'] for r in accepted}
    assert len(observed)==1 and abs(observed[0]['native_phase_residual_seconds']-.120)<1e-8
    assert observed[0]['time']==rows[-1]['time']


def test_global_native_grid_offset_and_hidden_same_phase_hit_are_retained():
    centres=np.arange(20)*.5+.3
    refs=[{'index':i,'time':float(t+.075)} for i,t in enumerate(centres) if i!=7]
    rows=refs+[{'index':7,'time':float(centres[7]+.077)}]
    accepted,observed=classify_native_phase(rows,centres,refs)
    assert len(accepted)==20 and not observed


def test_final_frame_slopes_override_stale_reported_statistics():
    warp=mapping();initialize_rate_reference(warp,1000)
    warp['max_local_warp_rate_change']=.05
    warp['anchors'][1]['output_frame']=800
    with pytest.raises(ValueError,match='Actual acoustic timing change'):
        audit_warp_rates(warp,1000,stage='after source correction')
    assert warp['actual_timing_rate_audit']['maximum_actual_playback_rate']==1.25
    assert warp['max_local_warp_rate_change']==.25


def test_backend_correction_cannot_leave_a_bad_map_behind():
    warp=mapping();initialize_rate_reference(warp,1000);before=deepcopy(warp['anchors'])
    drift={'measured':[{'role':'head','grid_output_seconds':1.,'delay_seconds':.25}]}
    with pytest.raises(ValueError,match='Actual acoustic timing change'):
        compensate_backend_drift(warp,drift,1000)
    assert warp['anchors']==before
    assert warp['actual_timing_rate_audit']['passed']
    assert not warp['rejected_timing_rate_audits'][-1]['passed']


def test_separately_bounded_corrections_cannot_hide_unsafe_composed_rate():
    warp=mapping();initialize_rate_reference(warp,1000)
    warp['anchors'][1]['source_frame']=1100
    warp['fine_attack_timing']={'head':{'times':[.2,.3,.4],'offsets':[-.012,0.,.012]}}
    with pytest.raises(ValueError,match='Actual acoustic timing change'):
        audit_warp_rates(warp,1000,stage='complete composed map')
    assert warp['actual_timing_rate_audit']['maximum_actual_playback_rate']>1.23


def test_deliberate_musical_tempo_change_is_not_false_timing_damage():
    sr=16000;clock=shared_clock(130,130,32,sr)
    source=np.rint(np.asarray(clock['phase'])*.6*sr).astype(int)
    warp={'anchors':[{'source_frame':int(a),'output_frame':int(z)} for a,z in zip(source,clock['output_frames'])],
        'input_output_beat_mapping':{'head':[{'output_seconds':0.},{'output_seconds':clock['duration_seconds']}],'tail':[]}}
    initialize_rate_reference(warp,sr,native_bpm=100,windows=[('head',clock)],limit=.15)
    result=audit_warp_rates(warp,sr,stage='musical tempo')
    assert result['passed'] and result['maximum_actual_playback_rate']>1.29
    assert result['maximum_timing_correction_change']<.002


def test_ambiguous_earlier_click_selects_sequential_estimated_percussion():
    warp={'source_kick_alignment':{'tail':{'attack_front_comparison':{'matched_source_events':41,
        'body_minus_upper_median_seconds':.017875,'body_minus_upper_10_90_spread_seconds':.018125,'upper_front_selected':False}}}}
    assert ownership_policy(warp,{})=='sequential'


def test_sequential_percussion_preserves_long_harmonic_curves_and_exact_components():
    sr=8000;n=sr*8;phase=np.linspace(0,1,n);t=np.arange(n)/sr
    a=np.column_stack([.1*np.sin(t*2*np.pi*90)+.05*np.sin(t*2*np.pi*440)]*2).astype('float32')
    b=np.column_stack([.12*np.sin(t*2*np.pi*115)+.04*np.sin(t*2*np.pi*660)]*2).astype('float32')
    da=a*.6;db=b*.7
    output,report=blend(a,b,sr,beats=128,beat_phase=phase,style='zhilin',drums_out=da,drums_in=db,percussion_policy='sequential')
    base,_=envelopes(n,128,beat_phase=phase,style='zhilin');gains=percussion_gains(base,phase,128,'sequential')
    assert all(np.array_equal(base[k],gains[k]) for k in base)
    assert np.max(gains['drums_out']*gains['drums_in'])==0
    parts=verification_components(a,b,sr,gains,drums_out=da,drums_in=db)
    assert np.max(abs(parts['tail']+parts['head']-output))<1e-7
    assert report['percussion_processing']['handoff_beats']<1.01


def test_real_source_recovery_keeps_a_displaced_native_figure_off_the_primary_grid(tmp_path):
    from superdj.kick_grid import refine_source_controls
    sr=16000;times=np.arange(.4,18.8,.5);audio=np.zeros((sr*20,2),dtype='float32')
    time=np.arange(round(.16*sr))/sr;rng=np.random.default_rng(39)
    pulse=.4*np.sin(2*np.pi*80*time)*np.exp(-time/.035)+.10*rng.normal(size=len(time))*np.exp(-time/.003)
    for index,stamp in enumerate(times):
        first=round((stamp+(.105 if index==12 else 0))*sr);audio[first:first+len(pulse)]+=pulse[:,None]
    path=tmp_path/'native.wav';sf.write(path,audio,sr,subtype='FLOAT')
    phase=np.arange((len(times)-1)*4+1)/4;source=np.interp(phase,np.arange(len(times)),times)
    anchors=[{'source_frame':0,'output_frame':0,'source_seconds':0.,'output_seconds':0.,'role':'clip_start'}]
    anchors.extend({'source_frame':round(t*sr),'output_frame':round(t*sr),'source_seconds':float(t),'output_seconds':float(t),'role':'head_sync_control'} for t in source)
    anchors.append({'source_frame':len(audio),'output_frame':len(audio),'source_seconds':20.,'output_seconds':20.,'role':'clip_end'})
    warp={'anchors':anchors,'trim_start_seconds':0.,'input_output_beat_mapping':{'head':[{'beat':i,'source_seconds':float(t),'output_seconds':float(t)} for i,t in enumerate(times)],'tail':[]}}
    report=refine_source_controls(path,warp,sr,evidence={'head':{'path':str(path),'source_start_seconds':0}})['head']
    assert 12 not in {e['beat'] for e in report['events']}
    observation=next(e for e in report['native_rhythm_observations'] if e['beat']==12)
    assert abs(observation['source_seconds']-times[12]-.105)<.005
    assert warp['actual_timing_rate_audit']['passed']
    assert np.array_equal(sf.read(path,dtype='float32',always_2d=True)[0],audio)


def test_timing_guard_changes_invalidate_old_clip_and_overlap_objects(tmp_path):
    import superdj.timing_safety as safety
    from superdj.execution import stage_fingerprint
    source=tmp_path/'superdj';shutil.copytree(Path(safety.__file__).parent,source,ignore=shutil.ignore_patterns('__pycache__'))
    before={stage:stage_fingerprint(stage,source) for stage in ('clip','overlap')}
    path=source/'timing_safety.py';path.write_text(path.read_text(encoding='utf-8')+'\n# changed guard contract\n',encoding='utf-8')
    assert all(stage_fingerprint(stage,source)!=before[stage] for stage in before)
