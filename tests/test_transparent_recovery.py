from copy import deepcopy
import numpy as np
import pytest
from superdj.orchestration import whole_spectrum_alternative
from superdj import orchestration
from superdj.transition_quality import TransitionFailure


def test_transparent_recovery_keeps_clock_sources_length_and_all_quality_checks(monkeypatch,tmp_path):
    rng=np.random.default_rng(500);a=rng.normal(0,.02,(32000,2)).astype('float32');b=rng.normal(0,.02,a.shape).astype('float32');phase=np.linspace(0,1,len(a));edge={'from':'a','to':'b','beats':128,'bars':32,'shared_clock':{'immutable':True},'outgoing_source_start':100.,'incoming_source_start':50.};before=deepcopy(edge);copies=[x.copy()for x in (a,b)];calls=[]
    def verify(y,sr,wa,wb,g,*args,components,**kw):
        np.testing.assert_allclose(components['tail']+components['head'],y,atol=2e-8)
        assert len(y)==len(a) and not kw['allow_pitch_changing_residual']
        calls.append('attacks');return y,{'executed':True}
    def quality(y,selected):
        assert selected['shared_clock']==before['shared_clock']
        assert selected['whole_spectrum_handoff']['end_beat']==128
        calls.append('quality');return {'passed':True}
    monkeypatch.setattr(orchestration,'verify_blended_fronts',verify)
    y,selected=whole_spectrum_alternative(a,b,8000,{}, {},edge,tmp_path,style='zhilin',beat_phase=phase,initial_error=ValueError('existing routing failed'),quality_check=quality)
    assert calls==['attacks','quality'] and edge==before
    assert selected['whole_spectrum_recovery']['source_windows_clock_and_overlap_unchanged']
    assert selected['frequency_processing']['actual_10_90_shared_beats']>=16
    assert np.array_equal(a,copies[0])and np.array_equal(b,copies[1])

@pytest.mark.parametrize('lock',[{'handoff_windows':{'bass':[12,40]}},{'mid_handoff_bars':16},{'bass_handoff_beats':16},{'whole_spectrum_handoff':{}}])
def test_transparent_fallback_never_changes_reviewed_band_settings(monkeypatch,tmp_path,lock):
    x=np.zeros((32000,2));monkeypatch.setattr(orchestration,'blend',lambda *a,**k:pytest.fail('Changed a locked reviewed transfer'))
    failure=ValueError('retained locked failure')
    with pytest.raises(ValueError)as result:whole_spectrum_alternative(x,x,8000,{}, {},{'beats':128,**lock},tmp_path,style='zhilin',beat_phase=np.linspace(0,1,len(x)),initial_error=failure)
    assert result.value is failure


def test_transparent_recovery_does_not_waive_failed_attacks_or_quality(monkeypatch,tmp_path):
    x=np.ones((32000,2),dtype='float32')*.01;calls=[]
    def failure(*args,**kw):calls.append(1);raise ValueError('real late attack')
    monkeypatch.setattr(orchestration,'verify_blended_fronts',failure)
    with pytest.raises(ValueError,match='real late'):whole_spectrum_alternative(x,x,8000,{}, {},{'beats':128},tmp_path,style='zhilin',beat_phase=np.linspace(0,1,len(x)),initial_error=ValueError('old'))
    assert len(calls)==2
