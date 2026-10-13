import numpy as np
import pytest
from superdj.mixing import blend,envelopes,split_bands

def test_complete_percussion_transfers_without_shortening_harmonic_beds():
    sr=8000;n=sr*4;t=np.arange(n)/sr;phase=np.linspace(0,1,n)**1.3
    da=np.column_stack([.2*np.sin(2*np.pi*700*t)]*2);db=np.column_stack([.15*np.sin(2*np.pi*900*t)]*2)
    bed_a=np.column_stack([.08*np.sin(2*np.pi*420*t)]*2);bed_b=np.column_stack([.09*np.sin(2*np.pi*520*t)]*2)
    actual,report=blend(bed_a+da,bed_b+db,sr,beats=128,beat_phase=phase,style='zhilin',drums_out=da,drums_in=db)
    beds,_=blend(bed_a,bed_b,sr,beats=128,beat_phase=phase,style='zhilin');gains,_=envelopes(n,128,beat_phase=phase,style='zhilin')
    assert np.max(abs(actual-(beds+da*gains['bass_out']+db*gains['bass_in'])))<1e-7
    after=phase>report['bass_handoff_fraction'][1]
    assert np.max(abs((actual-beds-db)[after]))<1e-7
    assert np.max(gains['mid_out'][after])>.2
    assert report['percussion_processing']['maximum_outgoing_drum_gain_after_handoff']==0
    assert report['endpoint_max_error']<1e-7

def test_percussion_refuses_unpaired_wrong_shape_or_unverified_fallback():
    a=np.zeros((256,2),dtype='float32')
    with pytest.raises(ValueError,match='Both aligned'):blend(a,a,8000,drums_out=a)
    with pytest.raises(ValueError,match='timing and shape'):blend(a,a,8000,drums_out=a[:-1],drums_in=a)
    with pytest.raises(ValueError,match='verified frequency'):blend(a,a,8000,bass_hz=0,drums_out=a,drums_in=a)
    with pytest.raises(ValueError,match='verified frequency'):blend(a[:64],a[:64],8000,drums_out=a[:64],drums_in=a[:64])

def test_automatic_handoff_is_conditional_and_uses_exact_postprocessed_pcm(tmp_path,monkeypatch):
    from superdj.percussion import needs_handoff,final_drum_estimates
    from superdj import stems
    import soundfile as sf
    warp={'source_kick_alignment':{'head':{'attack_front_comparison':{'upper_front_selected':True}}}}
    assert needs_handoff({},warp,style='zhilin')
    assert not needs_handoff({},warp,style='default')
    assert not needs_handoff(warp,{},style='zhilin')
    captured=[]
    monkeypatch.setattr(stems,'configuration',lambda:{'include_drums':True})
    def separate(source,digest,cache,config,ffmpeg):
        pcm,rate=sf.read(source,dtype='float32',always_2d=True);captured.append((digest,pcm.copy()));path=source.with_name(source.stem+'-drums.wav');sf.write(path,pcm*.4,rate,subtype='FLOAT');return {'drums_path':str(path),'provenance':{'executed':True}}
    monkeypatch.setattr(stems,'separate',separate)
    rng=np.random.default_rng(8);a=rng.normal(0,.01,(16000,2)).astype('float32');b=a*.7
    args,reports=final_drum_estimates(a,b,8000,tmp_path)
    assert np.array_equal(captured[0][1],a) and np.array_equal(captured[1][1],b)
    assert captured[0][0]!=captured[1][0]
    assert np.max(abs(args['drums_out']-a*.4))<1e-8
    assert all(x['estimated_not_ground_truth'] for x in reports)

def test_automatic_handoff_stops_when_runtime_is_unavailable(tmp_path,monkeypatch):
    from superdj.percussion import final_drum_estimates
    from superdj import stems
    monkeypatch.setattr(stems,'configuration',lambda:None)
    a=np.zeros((256,2),dtype='float32')
    with pytest.raises(RuntimeError,match='local Demucs runtime'):final_drum_estimates(a,a,8000,tmp_path)

def test_final_blend_corrects_one_supported_front_without_changing_other_pcm(tmp_path,monkeypatch):
    from superdj.percussion import verify_blended_fronts
    from superdj import percussion
    from scipy.signal import butter,sosfiltfilt
    sr=8000;n=sr*20;audio=np.zeros((n,2),dtype='float32');centres=.3+np.arange(39)*.5;t=np.arange(1000)/sr;wave=.15*np.random.default_rng(33).normal(size=len(t))*np.exp(-t/.006)
    for stamp in centres:
        i=round(stamp*sr);audio[i:i+len(wave)]+=wave[:,None]
    band=sosfiltfilt(butter(3,(180,1200),btype='bandpass',fs=sr,output='sos'),audio[:,0]);templates=np.array([band[round(t*sr)-96:round(t*sr)+360] for t in centres]);path=tmp_path/'templates.npz';np.savez(path,templates=templates)
    source={'attack_front_comparison':{'upper_front_selected':True},'attack_templates_path':str(path),'attack_band_hz':[180,1200],'events':[{'output_seconds':float(t),'beat':i} for i,t in enumerate(centres)]};warp={'source_kick_alignment':{'head':source},'input_output_beat_mapping':{'head':[{'output_seconds':0.}]}}
    broken=audio.copy();stamp=centres[26];first=round(stamp*sr);last=first+len(wave);broken[first:last]=0;broken[first+36:last+36]=audio[first:last]
    monkeypatch.setattr(percussion,'final_drum_estimates',lambda a,b,*args,**kw:({'drums_out':a,'drums_in':b},[]))
    gains={'bass_in':np.ones((n,1)),'bass_out':np.zeros((n,1))};repaired,report=verify_blended_fronts(broken,sr,{},warp,gains,tmp_path)
    assert len(report['local_combined_waveform_refinements'])==1
    assert report['checks'][0]['maximum_error_seconds']<=.0005
    fix=report['local_combined_waveform_refinements'][0];lo=fix['start_frame'];hi=fix['end_frame'];assert fix['maximum_local_rate_change']<.02;assert np.array_equal(repaired[:lo],broken[:lo]);assert np.array_equal(repaired[hi:],broken[hi:])
    too_late=audio.copy();too_late[first:last]=0;too_late[first+80:last+80]=audio[first:last]
    with pytest.raises(ValueError,match='misses the shared clock'):verify_blended_fronts(too_late,sr,{},warp,gains,tmp_path)
