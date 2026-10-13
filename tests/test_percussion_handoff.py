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
