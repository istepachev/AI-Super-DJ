from copy import deepcopy
import os,subprocess,shutil
import numpy as np
import soundfile as sf
import pytest
from superdj.kick_grid import refine_source_controls,rendered_attack_audit,attack_events,repeated_attacks,validate_overlap_attacks,fine_align_rendered,apply_fine_timing,template_attack_rows,preserve_rendered_attacks,repeated_waveform_family
from superdj.sync import compensate_backend_drift

def fixture(tmp_path,missing=None,variable=False):
    sr=16000;duration=18.;times=np.arange(.3,17.4,.503);nominal=np.round(times/.02)*.02+.014
    if variable:
        times=.3+np.r_[0.,np.cumsum(.503+.045*np.sin(np.arange(len(times)-1)*.22))]
        nominal=np.round(times/.02)*.02+.014
    signal=np.zeros(round(duration*sr));rng=np.random.default_rng(57)
    for index,beat in enumerate(times):
        if index==missing or (isinstance(missing,set) and index in missing):continue
        t=np.arange(round(.16*sr))/sr
        pulse=.35*np.sin(2*np.pi*80*t)*np.exp(-t/.035)+.10*rng.normal(size=len(t))*np.exp(-t/.003)
        first=round(beat*sr);signal[first:first+len(pulse)]+=pulse
    t=np.arange(len(signal))/sr;signal+=.012*np.sin(2*np.pi*220*t)
    path=tmp_path/'source.wav';sf.write(path,np.column_stack([signal]*2),sr,subtype='FLOAT')
    phases=np.arange((len(times)-1)*4+1)/4
    sources=np.interp(phases,np.arange(len(times)),nominal);outputs=.3+phases*.5
    anchors=[{'source_frame':0,'output_frame':0,'source_seconds':0.,'output_seconds':0.,'role':'clip_start'}]
    anchors.extend({'source_frame':round(a*sr),'output_frame':round(b*sr),'source_seconds':a,'output_seconds':b,'role':'head_sync_control'} for a,b in zip(sources,outputs))
    anchors.append({'source_frame':len(signal),'output_frame':len(signal),'source_seconds':duration,'output_seconds':duration,'role':'clip_end'})
    warp={'anchors':anchors,'trim_start_seconds':0.,'output_frames':len(signal),
          'input_output_beat_mapping':{'head':[{'beat':i,'source_seconds':float(a),'output_seconds':float(.3+i*.5)} for i,a in enumerate(nominal)],'tail':[]}}
    return path,warp,sr

def test_continuous_bass_is_not_a_kick_sequence():
    sr=16000;t=np.arange(sr*20)/sr;audio=np.column_stack([.1*np.sin(2*np.pi*150*t)]*2)
    assert not repeated_attacks(attack_events(audio,sr,np.arange(.5,19,.5)))

def test_front_focused_template_retains_timing_and_detects_a_late_attack():
    sr=8000;audio=np.zeros((sr*3,2),dtype='float32');t=np.arange(456)/sr
    pulse=.2*np.random.default_rng(29).normal(size=len(t))*np.exp(-np.maximum(t-.012,0)/.008);pulse[:96]=0
    center=sr;audio[center-96:center+360]=pulse[:,None]
    from scipy.signal import butter,sosfiltfilt
    band=sosfiltfilt(butter(3,(100,1400),btype='bandpass',fs=sr,output='sos'),audio[:,0]);template=band[center-96:center+96].copy()
    row=template_attack_rows(audio,sr,[1.],np.array([template]))[0];assert row['eligible'] and abs(row['time']-1)<.00025
    delayed=np.pad(audio,((72,0),(0,0)))[:len(audio)];row=template_attack_rows(delayed,sr,[1.],np.array([template]))[0];assert row['eligible'] and abs(row['time']-1.009)<.00025

def test_coherent_upper_front_is_selected_ahead_of_delayed_louder_body(tmp_path):
    path,warp,sr=fixture(tmp_path);audio=np.zeros((sf.info(path).frames,2),dtype='float32');times=np.arange(.3,17.4,.503)
    t=np.arange(round(.16*sr))/sr
    front=.3*np.sin(2*np.pi*700*t)*np.exp(-t/.004)
    body=1.2*np.sin(2*np.pi*110*t)*np.exp(-t/.032)
    for stamp in times:
        first=round(stamp*sr);audio[first:first+len(front)]+=front[:,None]
        first=round((stamp+.04)*sr);audio[first:first+len(body)]+=body[:,None]
    sf.write(path,audio,sr,subtype='FLOAT')
    report=refine_source_controls(path,warp,sr,evidence={'head':{'path':str(path),'source_start_seconds':0}})['head']
    assert report['attack_front_comparison']['upper_front_selected']
    assert report['attack_band_hz']==[180,1200]
    assert .025<report['attack_front_comparison']['body_minus_upper_median_seconds']<.055
    assert report['supported']>=20
    assert max(abs(e['source_seconds']-times[e['beat']]) for e in report['events'])<.006


def test_native_attack_islands_repair_delayed_pcm_and_preserve_body_and_voice(tmp_path):
    source,warp,sr=fixture(tmp_path,missing=12);refine_source_controls(source,warp,sr)
    audio,_=sf.read(source,dtype='float32',always_2d=True);delayed=np.pad(audio,((round(.019*sr),0),(0,0)))[:len(audio)]
    mix=tmp_path/'mix.wav';voice_source=tmp_path/'voice-source.wav';voice=tmp_path/'voice.wav'
    sf.write(mix,delayed,sr,subtype='FLOAT');sf.write(voice_source,audio*.3,sr,subtype='FLOAT');sf.write(voice,delayed*.3,sr,subtype='FLOAT')
    events=preserve_rendered_attacks(mix,source,warp,sr);preserve_rendered_attacks(voice,voice_source,warp,sr,record_templates=False)
    repaired,_=sf.read(mix,dtype='float32',always_2d=True);foreground,_=sf.read(voice,dtype='float32',always_2d=True)
    mask=np.ones(len(audio),dtype=bool)
    for e in events:mask[e['output_start_frame']:e['output_start_frame']+e['frames']]=False
    assert np.array_equal(repaired[mask],delayed[mask]);assert np.max(abs(foreground-repaired*.3))<1e-7
    audit=rendered_attack_audit(mix,warp,sr);assert len(audit['measured'])>=20 and audit['maximum_measured_drift_seconds']<=.00025
    assert any(p['beat']==12 for p in audit['unverified'])


def test_native_attack_preservation_refuses_missing_context(tmp_path):
    source,warp,sr=fixture(tmp_path);refine_source_controls(source,warp,sr)
    warp['source_kick_alignment']['head']['events'][0]['output_seconds']=.001
    output=tmp_path/'output.wav';shutil.copy2(source,output)
    with pytest.raises(ValueError,match='context'):preserve_rendered_attacks(output,source,warp,sr)


def test_protected_native_identity_still_detects_one_late_attack(tmp_path):
    source,warp,sr=fixture(tmp_path);refine_source_controls(source,warp,sr);output=tmp_path/'output.wav';shutil.copy2(source,output)
    islands=preserve_rendered_attacks(output,source,warp,sr);a,_=sf.read(output,dtype='float32',always_2d=True);e=islands[10];lo=e['output_start_frame'];hi=lo+e['frames'];block=a[lo:hi].copy();delay=round(.009*sr);a[lo:hi]=0;a[lo+delay:hi+delay]=block;sf.write(output,a,sr,subtype='FLOAT')
    audit=rendered_attack_audit(output,warp,sr);assert audit['maximum_measured_drift_seconds']>=.0085


def test_optional_waveform_family_needs_repeated_shapes_and_rejects_continuous_tone(tmp_path):
    source,warp,sr=fixture(tmp_path);a,_=sf.read(source,dtype='float32',always_2d=True);centres=[p['source_seconds'] for p in warp['input_output_beat_mapping']['head'][1:-1]]
    assert len(repeated_waveform_family(a,sr,attack_events(a,sr,centres)))>=8
    t=np.arange(len(a))/sr;tone=np.column_stack([.1*np.sin(2*np.pi*150*t)]*2)
    assert not repeated_waveform_family(tone,sr,attack_events(tone,sr,centres))


def test_source_primary_shape_is_recovered_behind_one_louder_different_attack(tmp_path):
    source,warp,sr=fixture(tmp_path);a=np.zeros((sf.info(source).frames,2),dtype='float32');times=np.arange(.3,17.4,.503);t=np.arange(round(.16*sr))/sr;rng=np.random.default_rng(9)
    kick=.4*np.sin(2*np.pi*80*t)*np.exp(-t/.035)+.1*rng.normal(size=len(t))*np.exp(-t/.003)
    for stamp in times:
        first=round(stamp*sr);a[first:first+len(kick)]+=kick[:,None]
    a[round((times[10]+.075)*sr)]+=2.;sf.write(source,a,sr,subtype='FLOAT')
    report=refine_source_controls(source,warp,sr)['head'];event=next(e for e in report['events'] if e['beat']==10)
    assert abs(event['source_seconds']-times[10])<.005
    assert event.get('source_template_correlation',0)>=.90

def test_missing_source_hit_is_not_reported_as_verified(tmp_path):
    path,warp,sr=fixture(tmp_path,missing=12)
    report=refine_source_controls(path,warp,sr)['head']
    assert report['supported']>=.8*report['detected']
    assert report['unsupported']>=1
    supported={e['beat'] for e in report['events']}
    assert 12 not in supported
    assert warp['anchors'][0]['source_frame']==0 and warp['anchors'][-1]['source_frame']==sf.info(path).frames

def test_zero_coverage_stays_explicit(tmp_path):
    path,warp,sr=fixture(tmp_path)
    t=np.arange(sf.info(path).frames)/sr;sf.write(path,np.column_stack([.05*np.sin(2*np.pi*150*t)]*2),sr,subtype='FLOAT')
    refine_source_controls(path,warp,sr);audit=rendered_attack_audit(path,warp,sr)
    assert audit['maximum_measured_drift_seconds'] is None
    assert not audit['all_interior_beats_verified'] and audit['unverified']

def test_source_tempo_variation_is_not_discarded_as_bad_attacks(tmp_path):
    path,warp,sr=fixture(tmp_path,variable=True)
    report=refine_source_controls(path,warp,sr)['head']
    assert report['supported']>=.8*report['detected']
    assert all(b['source_frame']>a['source_frame'] for a,b in zip(warp['anchors'],warp['anchors'][1:]))

def test_half_time_kicks_remain_sparse_evidence(tmp_path):
    missing=set(range(1,34,2))
    path,warp,sr=fixture(tmp_path,missing=missing)
    report=refine_source_controls(path,warp,sr)['head']
    supported={e['beat'] for e in report['events']}
    assert len(supported)>=12
    assert not (supported & missing)
    assert report['unsupported']>=12

def test_random_nearby_peaks_do_not_form_a_kick_grid():
    rng=np.random.default_rng(61)
    rows=[{'index':i,'time':i*.5+rng.uniform(-.12,.12),'eligible':True} for i in range(127)]
    assert not repeated_attacks(rows)

def test_source_template_rejects_another_nearby_sharp_attack(tmp_path):
    path,warp,sr=fixture(tmp_path)
    source=refine_source_controls(path,warp,sr)['head']
    audio,_=sf.read(path,dtype='float32',always_2d=True)
    shift=round(.007*sr);mixture=np.pad(audio,((shift,0),(0,0)))[:len(audio)]
    centres=[e['source_seconds'] for e in source['events']]
    for centre in centres:mixture[round((centre+.023)*sr)]+=1.5
    with np.load(source['attack_templates_path']) as f:templates=f['templates']
    rows=template_attack_rows(mixture,sr,centres,templates)
    measured=[r['time']-centre for r,centre in zip(rows,centres) if r['eligible']]
    assert len(measured)>=.8*len(centres)
    assert max(abs(delay-.007) for delay in measured)<.001

def test_fine_timing_keeps_native_body_and_voice_mapping(tmp_path):
    sr=16000;t=np.arange(6*sr)/sr
    audio=np.column_stack([np.sin(2*np.pi*180*t)]*2).astype('float32')
    path=tmp_path/'source.wav';voice=tmp_path/'voice.wav';output=tmp_path/'corrected.wav';vout=tmp_path/'corrected-voice.wav'
    sf.write(path,audio,sr,subtype='FLOAT');sf.write(voice,audio*.2,sr,subtype='FLOAT')
    warp={'fine_attack_timing':{'head':{'times':[0.,.5,1.,1.5,2.],'offsets':[0.,.003,-.002,.001,0.]}}}
    apply_fine_timing(path,output,warp,sr);apply_fine_timing(voice,vout,warp,sr)
    result,_=sf.read(output,dtype='float32',always_2d=True);vresult,_=sf.read(vout,dtype='float32',always_2d=True)
    assert len(result)==len(audio)
    np.testing.assert_array_equal(result[2*sr:],audio[2*sr:])
    np.testing.assert_allclose(vresult,result*.2,atol=4e-8)

def test_one_missed_hit_cannot_hide_in_average_or_source_latency():
    def clip(role,miss):
        points=[{'role':role,'beat':i,'grid_output_seconds':i*.5,'observed_seconds':i*.5+(.02 if i==miss else 0.),'delay_seconds':0.} for i in range(1,128)]
        return {'warp':{'post_render_transient_audit':{'final':{'measured':points}}}}
    edge={'from':'a','to':'b','tempo_technique':'shared_phase_sync','beats':128}
    with pytest.raises(ValueError,match='20.00 ms'):
        validate_overlap_attacks([clip('tail',76),clip('head',None)],[edge])
    report=validate_overlap_attacks([clip('tail',None),clip('head',None)],[edge])[0]
    assert report['verified_common_beats']==127 and report['maximum_pair_error_seconds']==0

def test_zero_common_events_cannot_pass():
    with pytest.raises(ValueError,match='no adequate shared'):
        validate_overlap_attacks([{'warp':{}},{'warp':{}}],[{'from':'a','to':'b','tempo_technique':'shared_phase_sync','beats':128}])

def test_interlocking_rhythms_keep_individual_clock_verification():
    def clip(role,indices,miss=None):
        points=[{'role':role,'beat':i,'grid_output_seconds':i*.5,'observed_seconds':i*.5+(.02 if i==miss else .0002),'delay_seconds':0.} for i in indices]
        return {'warp':{'post_render_transient_audit':{'final':{'measured':points}}}}
    outgoing=list(range(1,128,2))+[30,52,56,86,92,110]
    incoming=list(range(2,128,2));edge={'from':'a','to':'b','tempo_technique':'shared_phase_sync','beats':128}
    report=validate_overlap_attacks([clip('tail',outgoing),clip('head',incoming)],[edge])[0]
    assert report['verified_common_beats']==6 and 'sparse' in report['verification_kind']
    assert report['conservative_pair_error_bound_seconds']<.001
    with pytest.raises(ValueError,match='no adequate shared'):
        validate_overlap_attacks([clip('tail',outgoing,miss=21),clip('head',incoming)],[edge])

def test_actual_rubberband_corrects_biased_quantised_grid(tmp_path):
    rubberband=os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not rubberband:pytest.skip('Actual Rubber Band required')
    path,warp,sr=fixture(tmp_path);original=deepcopy(warp)
    refine_source_controls(path,warp,sr);assert warp['source_kick_alignment']['head']['supported']>=.8*(len(warp['input_output_beat_mapping']['head'])-2)
    def render(mapping,label):
        timing=tmp_path/f'{label}.txt';output=tmp_path/f'{label}.wav'
        timing.write_text(''.join(f"{p['source_frame']} {p['output_frame']}\n" for p in mapping['anchors']),encoding='ascii')
        subprocess.run([rubberband,'--fast','--detector-perc','--centre-focus','--ignore-clipping','--quiet','--time','1','--timemap',str(timing),str(path),str(output)],check=True,capture_output=True)
        assert sf.info(output).frames==warp['output_frames'];return output
    before=render(original,'before');old=rendered_attack_audit(before,warp,sr)
    assert old['maximum_measured_drift_seconds']>.010
    audits=[]
    for attempt in range(10):
        after=render(warp,f'after{attempt}');audit=rendered_attack_audit(after,warp,sr);audits.append(audit)
        if audit['maximum_measured_drift_seconds'] is not None and audit['maximum_measured_drift_seconds']<=.003:break
        compensate_backend_drift(warp,audit,sr,strength=1. if attempt<2 else .5)
    assert audits[-1]['maximum_measured_drift_seconds'] is not None
    assert audits[-1]['maximum_measured_drift_seconds']<=.003
    assert len(audits[-1]['measured'])>=.8*len(warp['source_kick_alignment']['head']['events'])
    # An independent residual stage handles backend detector snapping, with
    # its own audit rather than trusting the intended interpolation map.
    final=fine_align_rendered(after,warp,sr)
    assert final['maximum_measured_drift_seconds']<=.0015
    assert sf.info(after).frames==warp['output_frames']
