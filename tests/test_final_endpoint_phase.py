from copy import deepcopy
import numpy as np
import pytest
import soundfile as sf
from superdj.constant_phase import align_constant_phase,apply_aligned_stream
from superdj.render import _copy_region
from test_constant_phase import fixture,sha,RATE


def final_fixture(tmp_path,delay=27):
    path,data,warp,auditor=fixture(tmp_path,delay)
    warp['source_kick_alignment'].pop('tail')
    def head_audit(*args):
        result=auditor(*args)
        result['measured']=[p for p in result['measured']if p['role']=='head']
        result['maximum_measured_drift_seconds']=max(abs(p['delay_seconds'])for p in result['measured'])
        return result
    return path,data,warp,head_audit


def test_final_endpoint_reaudits_its_one_real_edge_without_inventing_tail(tmp_path):
    path,before,warp,auditor=final_fixture(tmp_path)
    initial=auditor(path,warp,RATE)
    result=align_constant_phase(path,warp,initial,RATE,auditor=auditor,
                               soft_sync_boundaries=True,final_endpoint=True)
    assert result['maximum_measured_drift_seconds']<=.003
    phase=warp['constant_output_phase'];n=phase['advance_frames']
    actual,sr=sf.read(path,dtype='float32',always_2d=True)
    assert np.array_equal(actual[:-n],before[n:])and not actual[-n:].any()
    assert sr==RATE and len(actual)==len(before)and set(warp['source_kick_alignment'])=={'head'}
    assert len(initial['measured'])==len(result['measured'])
    assert phase['playback_rate']==1 and not phase['pitch_changed']and not phase['resampling']
    voice=tmp_path/'voice.wav';sf.write(voice,before*.5,RATE,subtype='FLOAT')
    apply_aligned_stream(voice,phase)
    assert np.array_equal(sf.read(voice,dtype='float32',always_2d=True)[0],actual*.5)


@pytest.mark.parametrize('delay',[0,16,24])
def test_passing_final_endpoint_is_byte_identical(tmp_path,delay):
    path,_,warp,auditor=final_fixture(tmp_path,delay);before=sha(path)
    align_constant_phase(path,warp,auditor(path,warp,RATE),RATE,auditor=auditor,
                         soft_sync_boundaries=True,final_endpoint=True)
    assert sha(path)==before and 'constant_output_phase'not in warp


def test_missing_required_final_view_is_still_refused(tmp_path):
    path,_,warp,auditor=final_fixture(tmp_path);before=sha(path);initial=auditor(path,warp,RATE)
    initial['measured']=[p for p in initial['measured']if not(p['beat']==3 and p['view']=='bass_body')]
    align_constant_phase(path,warp,initial,RATE,auditor=auditor,soft_sync_boundaries=True,final_endpoint=True)
    assert sha(path)==before and 'constant_output_phase'not in warp


def test_final_candidate_cannot_drop_an_old_observation(tmp_path):
    path,_,warp,auditor=final_fixture(tmp_path);before=sha(path)
    def lost(*args):
        result=auditor(*args);result['measured'].pop();return result
    align_constant_phase(path,warp,auditor(path,warp,RATE),RATE,auditor=lost,
                         soft_sync_boundaries=True,final_endpoint=True)
    assert sha(path)==before and 'constant_output_phase'not in warp


def test_final_scope_cannot_hide_a_declared_second_edge(tmp_path):
    path,_,warp,auditor=fixture(tmp_path);before=sha(path)
    align_constant_phase(path,warp,auditor(path,warp,RATE),RATE,auditor=auditor,
                         soft_sync_boundaries=True,final_endpoint=True)
    assert sha(path)==before and 'constant_output_phase'not in warp


def test_native_final_fade_ends_before_the_added_silence(tmp_path):
    source=tmp_path/'source.wav';target=tmp_path/'target.wav';n=1000;padding=12
    data=np.ones((n,2),dtype='float32');data[-padding:]=0;sf.write(source,data,RATE,subtype='FLOAT')
    with sf.SoundFile(source)as reader,sf.SoundFile(target,'w',samplerate=RATE,channels=2,subtype='FLOAT')as writer:
        _copy_region(reader,writer,n,fade_out=True,trailing_silence_frames=padding)
    actual=sf.read(target,dtype='float32',always_2d=True)[0]
    assert len(actual)==n and not actual[-padding-1:].any()
    assert 0<actual[-padding-2,0]<.03 and not(actual<0).any()


def test_old_fade_without_phase_padding_has_same_samples(tmp_path):
    source=tmp_path/'source.wav';target=tmp_path/'target.wav';n=1000
    data=np.column_stack([np.linspace(-.4,.8,n,dtype='float32')]*2);sf.write(source,data,RATE,subtype='FLOAT')
    with sf.SoundFile(source)as reader,sf.SoundFile(target,'w',samplerate=RATE,channels=2,subtype='FLOAT')as writer:
        _copy_region(reader,writer,n,fade_out=True)
    expected=data.copy();expected*=np.minimum(1.,(n-1-np.arange(n))/round(RATE*.005))[:,None]
    assert np.array_equal(sf.read(target,dtype='float32',always_2d=True)[0],expected)
