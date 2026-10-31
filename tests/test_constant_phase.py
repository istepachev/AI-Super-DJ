from copy import deepcopy
import hashlib
import numpy as np
import soundfile as sf
import pytest
from superdj.constant_phase import align_constant_phase,apply_aligned_stream,shift_stream


RATE=8000


def fixture(tmp_path,head_delay=27,tail_delay=None):
    tail_delay=head_delay if tail_delay is None else tail_delay
    data=np.zeros((40000,2),dtype='float32');events=[];warp={'source_kick_alignment':{}}
    for role,first,delay in [('head',4000,head_delay),('tail',22000,tail_delay)]:
        stamps=[first+beat*1600 for beat in range(8)]
        for beat,stamp in enumerate(stamps,1):
            data[stamp+delay]=[1.,.5];events.append((role,beat,stamp))
        warp['source_kick_alignment'][role]={'events':[{'beat':n}for n in range(1,9)],
            'low_frequency_gap_recovery':{'recovered_beats':list(range(1,9))},
            'native_mixture_reference':{'mixture_bass_visible_beats':list(range(1,9)),
                'bass_visible_beats':list(range(1,9)),'initial_front_required_beats':list(range(1,9))}}
    path=tmp_path/'clip.wav';sf.write(path,data,RATE,subtype='FLOAT')
    def audit(path,warp,rate):
        audio,sr=sf.read(path,dtype='float32',always_2d=True);assert sr==rate
        points=[]
        for role,beat,stamp in events:
            lo=stamp-400;actual=lo+int(np.argmax(audio[lo:stamp+400,0]))
            for view in ('primary','bass_body','estimated_bass','initial_front'):
                points.append({'role':role,'beat':beat,'view':view,'expected_seconds':stamp/rate,
                    'observed_seconds':actual/rate,'delay_seconds':(actual-stamp)/rate})
        return {'measured':points,'maximum_measured_drift_seconds':max(abs(p['delay_seconds'])for p in points)}
    return path,data,warp,audit


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_pcm_translation_passes_same_clock_and_preserves_body_exactly(tmp_path):
    path,before,warp,audit=fixture(tmp_path)
    original=audit(path,warp,RATE);assert original['maximum_measured_drift_seconds']>.003
    after=align_constant_phase(path,warp,original,RATE,auditor=audit,soft_sync_boundaries=True)
    phase=warp['constant_output_phase'];n=phase['advance_frames'];actual,rate=sf.read(path,dtype='float32',always_2d=True)
    assert n==4 and after['maximum_measured_drift_seconds']==.002875
    assert np.array_equal(actual[:-n],before[n:])and not actual[-n:].any()
    assert len(actual)==len(before)and sf.info(path).subtype=='FLOAT'and rate==RATE
    assert len(after['measured'])==len(original['measured'])
    assert phase['playback_rate']==1 and not phase['pitch_changed']and not phase['resampling']and not phase['isolated_attacks_copied']
    digest=sha(path);align_constant_phase(path,warp,after,RATE,auditor=audit,soft_sync_boundaries=True);assert sha(path)==digest


@pytest.mark.parametrize('delay',[0,16,24])
def test_passing_old_pcm_is_not_modified(tmp_path,delay):
    path,_,warp,audit=fixture(tmp_path,delay);before=sha(path)
    result=align_constant_phase(path,warp,audit(path,warp,RATE),RATE,auditor=audit,soft_sync_boundaries=True)
    assert sha(path)==before and 'constant_output_phase'not in warp


def test_variable_drift_cannot_be_hidden_by_constant_translation(tmp_path):
    path,_,warp,audit=fixture(tmp_path,32,-32);before=sha(path)
    align_constant_phase(path,warp,audit(path,warp,RATE),RATE,auditor=audit,soft_sync_boundaries=True)
    assert sha(path)==before and 'constant_output_phase'not in warp


def test_missing_native_mandatory_view_is_not_repaired_or_waived(tmp_path):
    path,_,warp,audit=fixture(tmp_path);before=sha(path);initial=audit(path,warp,RATE)
    initial['measured']=[p for p in initial['measured']if not(p['role']=='head'and p['beat']==3 and p['view']=='bass_body')]
    align_constant_phase(path,warp,initial,RATE,auditor=audit,soft_sync_boundaries=True)
    assert sha(path)==before and 'constant_output_phase'not in warp


def test_candidate_losing_one_observation_is_discarded(tmp_path):
    path,_,warp,audit=fixture(tmp_path);before=sha(path);initial=audit(path,warp,RATE)
    def lost(*args):
        out=audit(*args);out['measured']=out['measured'][1:];return out
    align_constant_phase(path,warp,initial,RATE,auditor=lost,soft_sync_boundaries=True)
    assert sha(path)==before and 'constant_output_phase'not in warp


def test_uncovered_edges_and_large_offsets_remain_refusals(tmp_path):
    path,_,warp,audit=fixture(tmp_path,200);before=sha(path)
    align_constant_phase(path,warp,audit(path,warp,RATE),RATE,auditor=audit,soft_sync_boundaries=True)
    assert sha(path)==before
    path,_,warp,audit=fixture(tmp_path);before=sha(path)
    align_constant_phase(path,warp,audit(path,warp,RATE),RATE,auditor=audit,soft_sync_boundaries=False)
    assert sha(path)==before and 'constant_output_phase'not in warp
    warp['source_kick_alignment'].pop('tail')
    align_constant_phase(path,warp,audit(path,warp,RATE),RATE,auditor=audit,soft_sync_boundaries=True)
    assert sha(path)==before and 'constant_output_phase'not in warp


def test_voice_and_both_stereo_channels_receive_identical_offset(tmp_path):
    path,before,warp,audit=fixture(tmp_path);voice=tmp_path/'voice.wav';sf.write(voice,before,RATE,subtype='FLOAT')
    align_constant_phase(path,warp,audit(path,warp,RATE),RATE,auditor=audit,soft_sync_boundaries=True)
    apply_aligned_stream(voice,warp['constant_output_phase'])
    actual,_=sf.read(path,dtype='float32',always_2d=True);aligned,_=sf.read(voice,dtype='float32',always_2d=True)
    assert np.array_equal(actual,aligned)


def test_delay_direction_and_double_precision_stream_are_sample_exact(tmp_path):
    source=tmp_path/'double.wav';out=tmp_path/'out.wav';data=np.arange(200,dtype='float64').reshape(100,2)/200
    sf.write(source,data,RATE,subtype='DOUBLE');shift_stream(source,out,-3)
    actual,_=sf.read(out,dtype='float64',always_2d=True)
    assert not actual[:3].any()and np.array_equal(actual[3:],data[:-3])and sf.info(out).subtype=='DOUBLE'


@pytest.mark.parametrize('key',['fine_attack_timing','native_attack_preservation'])
def test_explicit_legacy_processing_routes_are_unchanged(tmp_path,key):
    path,_,warp,audit=fixture(tmp_path);before=sha(path);warp[key]={'applied':True}
    align_constant_phase(path,warp,audit(path,warp,RATE),RATE,auditor=audit,soft_sync_boundaries=True)
    assert sha(path)==before and 'constant_output_phase'not in warp
