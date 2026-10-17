import numpy as np
import pytest
from superdj.mixing import blend,envelopes
from superdj.percussion import percussion_gains,verification_components
from superdj import orchestration
from superdj.transition_quality import TransitionFailure


@pytest.mark.parametrize('style',['default','zhilin'])
def test_positioned_soft_transfer_keeps_other_bands_and_exact_components(style):
    sr=8000;phase=np.linspace(0,1,sr*4)**1.13;n=len(phase)
    rng=np.random.default_rng(17);a=rng.normal(0,.03,(n,2));b=rng.normal(0,.04,(n,2))
    original,curves=envelopes(n,128,style=style,beat_phase=phase,tonal_conflict=True)
    width=curves['bass_handoff_beats'];window=[48-width/2,48+width/2]
    gains,selected=envelopes(n,128,style=style,beat_phase=phase,tonal_conflict=True,handoff_windows={'bass':window})
    for band in ('mid_out','mid_in','high_out','high_in'):
        assert np.array_equal(gains[band],original[band])
    assert selected['bass_handoff_beats']==pytest.approx(width)
    mixed,report=blend(a,b,sr,beats=128,style=style,beat_phase=phase,tonal_conflict=True,
        handoff_windows={'bass':window},drums_out=a*.4,drums_in=b*.3,percussion_policy='smooth')
    gains=percussion_gains(gains,phase,128,'smooth')
    parts=verification_components(a,b,sr,gains,drums_out=a*.4,drums_in=b*.3)
    np.testing.assert_allclose(parts['head']+parts['tail'],mixed,atol=3e-8)
    assert report['musical_handoff_continuity']['passed']
    assert report['percussion_processing']['handoff_beats']>=width-.01
    assert np.max(abs(mixed[0]-a[0]))<1e-7 and np.max(abs(mixed[-1]-b[-1]))<1e-7


@pytest.mark.parametrize('style',['default','zhilin'])
def test_retry_requires_real_components_and_quality_before_accepting(tmp_path,monkeypatch,style):
    sr=8000;n=sr*4;phase=np.linspace(0,1,n)**1.2
    rng=np.random.default_rng(6);a=rng.normal(0,.02,(n,2)).astype('float32');b=a[::-1].copy()
    before_a=a.copy();before_b=b.copy();audits=[];quality=[]
    def fronts(pcm,rate,wa,wb,gains,directory,**kw):
        assert rate==sr and kw['allow_pitch_changing_residual'] is False
        np.testing.assert_allclose(kw['components']['head']+kw['components']['tail'],pcm,atol=2e-8)
        audits.append(str(directory))
        if len(audits)==1:raise ValueError('Combined percussion attack misses the shared clock')
        return pcm,{'executed':True,'measurement_limit_seconds':.003}
    def check(pcm,transition):
        quality.append(transition['handoff_windows']['bass'])
        if len(quality)==1:raise TransitionFailure('a','b','sustained energy loss')
        return {'passed':True}
    monkeypatch.setattr(orchestration,'verify_blended_fronts',fronts)
    result,selected=orchestration.gradual_alternative(a,b,sr,{}, {},{'from':'a','to':'b','beats':128},tmp_path,
        style=style,beat_phase=phase,percussion_args={'drums_out':a*.3,'drums_in':b*.4},
        initial_error=ValueError('original collision'),quality_check=check)
    assert len(audits)==3 and len(quality)==2 and selected['quality_gate']['passed']
    assert selected['gradual_orchestration_search']['timing_gate_relaxed'] is False
    assert len(selected['gradual_orchestration_search']['failed_alternatives'])==2
    assert np.array_equal(a,before_a) and np.array_equal(b,before_b)


def test_exhausted_positions_preserve_failure_and_never_repair_pcm(tmp_path,monkeypatch):
    a=np.zeros((32000,2),dtype='float32');calls=[]
    def fail(*args,**kwargs):calls.append(1);raise ValueError('still late by 9ms')
    monkeypatch.setattr(orchestration,'verify_blended_fronts',fail)
    with pytest.raises(ValueError,match='still late by 9ms'):
        orchestration.gradual_alternative(a,a,8000,{}, {},{'beats':128},tmp_path,style='zhilin',
            beat_phase=np.linspace(0,1,len(a)),percussion_args={'drums_out':a,'drums_in':a},
            initial_error=ValueError('collision'))
    assert len(calls)==3 and not np.any(a)


@pytest.mark.parametrize('lock',[{'bass_handoff_beats':16},{'handoff_windows':{'bass':[40,56]}}])
def test_approved_bass_position_or_width_is_never_changed(lock,tmp_path,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('User bass lock must stop automatic repositioning')
    monkeypatch.setattr(orchestration,'verify_blended_fronts',forbidden)
    a=np.zeros((800,2),dtype='float32')
    with pytest.raises(ValueError,match='original failure'):
        orchestration.gradual_alternative(a,a,8000,{}, {},{'beats':128,**lock},tmp_path,style='zhilin',
            beat_phase=np.linspace(0,1,len(a)),percussion_args={},initial_error=ValueError('original failure'))


@pytest.mark.parametrize('window',[[-1,16],[16,16],[120,129],[0,float('nan')]])
def test_default_style_rejects_invalid_positioned_bass_window(window):
    with pytest.raises(ValueError,match='increasing shared-beat'):
        envelopes(1000,128,style='default',handoff_windows={'bass':window})
