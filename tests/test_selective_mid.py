import copy
import numpy as np
import pytest
from superdj.mixing import blend,split_bands,recorded_envelopes,envelopes
from superdj.selective_mid import cells,decide,apply_report
from superdj.percussion import verification_components,percussion_gains

def music(seconds=12):
    sr=8000;t=np.arange(sr*seconds)/sr
    a=np.repeat((.06*np.sin(2*np.pi*430*t))[:,None],2,axis=1)
    b=np.repeat((.06*np.sin(2*np.pi*2100*t))[:,None],2,axis=1)
    return sr,a,b

def test_cells_are_exactly_complementary_to_the_old_middle():
    sr,a,b=music(4);x=a+b
    np.testing.assert_allclose(sum(cells(x,sr)),split_bands(x,sr)[1],atol=3e-16)

@pytest.mark.parametrize('style',['default','zhilin'])
def test_dominant_parts_are_retained_above_equal_power_without_raising_drums(style):
    sr,a,b=music(32 if style=='default'else 12);zero=np.zeros_like(a);phase=np.linspace(0,1,len(a))
    args=dict(beats=128,style=style,beat_phase=phase,drums_out=zero,drums_in=zero,percussion_policy='smooth',vocal_out=zero,vocal_in=zero)
    old,old_report=blend(a,b,sr,mid_energy_mode='power',**args)
    new,report=blend(a,b,sr,**args)
    assert report['selective_mid_energy']['applied']
    prior,_=recorded_envelopes(len(a),128,{'frequency_processing':old_report},style=style,beat_phase=phase)
    core=(prior['mid_in'][:,0]>.4)&(prior['mid_in'][:,0]<.85)&(prior['mid_out'][:,0]>.4)
    assert np.mean(new[core]**2)>np.mean(old[core]**2)*1.3
    edge={'beats':128,'frequency_processing':report}
    g,_=recorded_envelopes(len(a),128,edge,style=style,beat_phase=phase)
    g=percussion_gains(g,phase,128,'smooth')
    prior,_=recorded_envelopes(len(a),128,{'frequency_processing':old_report},style=style,beat_phase=phase)
    prior=percussion_gains(prior,phase,128,'smooth')
    for key in ('bass_out','bass_in','high_out','high_in','drums_out','drums_in'):assert np.array_equal(g[key],prior[key])
    parts=verification_components(a,b,sr,g,drums_out=zero,drums_in=zero)
    np.testing.assert_allclose(parts['head']+parts['tail'],new,atol=2e-8)
    for i in range(8):
        out,inc=g[f'mid_cell_{i}_out'],g[f'mid_cell_{i}_in']
        assert np.all(np.diff(out[:,0])<=1e-12)and np.all(np.diff(inc[:,0])>=-1e-12)
        assert out.min()>=0 and inc.min()>=0 and out.max()<=1 and inc.max()<=1
        assert out[0,0]==inc[-1,0]==1 and out[-1,0]==inc[0,0]==0

def test_real_percussion_is_subtracted_from_supported_accompaniment():
    sr,a,b=music();t=np.arange(len(a))/sr
    drums=np.repeat((.04*np.sin(2*np.pi*700*t))[:,None],2,axis=1);zero=a*0
    mixed,report=blend(a+drums,b+drums*.7,sr,beats=128,style='zhilin',drums_out=drums,drums_in=drums*.7,
        percussion_policy='smooth',vocal_out=zero,vocal_in=zero)
    assert report['selective_mid_energy']['applied']
    g,_=recorded_envelopes(len(a),128,{'frequency_processing':report},style='zhilin',beat_phase=None)
    g=percussion_gains(g,None,128,'smooth')
    parts=verification_components(a+drums,b+drums*.7,sr,g,drums_out=drums,drums_in=drums*.7)
    np.testing.assert_allclose(parts['head']+parts['tail'],mixed,atol=3e-8)
    with pytest.raises(ValueError,match='drum estimates'):verification_components(a,b,sr,g)

def test_missing_separation_keeps_working_power_route():
    sr,a,b=music();zero=a*0
    new,report=blend(a,b,sr,beats=128,style='zhilin',vocal_out=zero,vocal_in=zero)
    old,_=blend(a,b,sr,beats=128,style='zhilin',vocal_out=zero,vocal_in=zero,mid_energy_mode='power')
    assert not report['selective_mid_energy']['applied']
    assert np.array_equal(new,old)

def test_shared_foreground_stays_on_previous_complementary_route():
    sr,a,b=music();zero=a*0
    new,report=blend(a,b,sr,beats=128,style='zhilin',drums_out=zero,drums_in=zero,vocal_out=a,vocal_in=b)
    old,_=blend(a,b,sr,beats=128,style='zhilin',drums_out=zero,drums_in=zero,vocal_out=a,vocal_in=b,mid_energy_mode='legacy')
    assert not report['selective_mid_energy']['applied']and np.array_equal(new,old)

def test_corrupted_selective_evidence_is_refused():
    sr,a,b=music(6);phase=np.linspace(0,1,len(a));r=decide(a,b,sr,beat_phase=phase,beats=128,enabled=True)
    assert r['applied'];g,_=envelopes(len(a),128,style='zhilin');p=g['mid_in']/(g['mid_in']+g['mid_out'])
    for patch in ({'law':'unknown'},{'frames':len(a)-1},{'maximum_source_cell_gain':2},{'phase_knots':None},
                  {'cutoffs_hz':[180,3500]},{'strengths':[[[2.]]]}):
        with pytest.raises(ValueError,match='actual overlap'):apply_report(g,{'selective_mid_energy':{**r,**patch}},position=p,beat_phase=phase)

def test_coincident_broad_cells_do_not_acquire_dominant_support():
    sr,a,b=music(4);x=a+b
    r=decide(x,x,sr,beat_phase=None,beats=64,enabled=True)
    assert not r['applied']

def test_new_middle_audibility_requests_whole_percussion_in_auto_only():
    from superdj.percussion import supported_mid_needs_drums
    f={'adaptive_mid_energy':{'applied':True}}
    assert supported_mid_needs_drums(f)
    assert not supported_mid_needs_drums(f,has_estimates=True)
    assert not supported_mid_needs_drums(f,mode='power')
    assert not supported_mid_needs_drums(f,mode='legacy')
    assert not supported_mid_needs_drums({'adaptive_mid_energy':{'applied':False}})
    assert not supported_mid_needs_drums({**f,'whole_spectrum_handoff':{'start_beat':0,'end_beat':32}})

def test_selective_refusal_preserves_working_power_before_legacy(tmp_path,monkeypatch):
    from superdj import orchestration
    sr,a,b=music(4);zero=a*0;phase=np.linspace(0,1,len(a))
    _,frequency=blend(a,b,sr,beats=128,style='zhilin',beat_phase=phase,drums_out=zero,drums_in=zero,vocal_out=zero,vocal_in=zero)
    assert frequency['selective_mid_energy']['applied']
    edge={'from':'a','to':'b','beats':128,'frequency_processing':frequency};checked=[]
    def fronts(pcm,*args,**kw):
        np.testing.assert_allclose(kw['components']['head']+kw['components']['tail'],pcm,atol=2e-8)
        return pcm,{'passed':True}
    def quality(pcm,candidate):checked.append(candidate['mid_energy_mode']);return {'passed':True}
    monkeypatch.setattr(orchestration,'verify_blended_fronts',fronts)
    _,candidate=orchestration.legacy_mid_alternative(a,b,sr,{}, {},edge,tmp_path,style='zhilin',beat_phase=phase,
        percussion_args={'drums_out':zero,'drums_in':zero,'percussion_policy':'smooth'},initial_error=ValueError('audible unsupported figure'),
        quality_check=quality,mid_energy_args={'vocal_out':zero,'vocal_in':zero})
    assert checked==['power']and candidate['mid_energy_mode']=='power'
    assert candidate['frequency_processing']['adaptive_mid_energy']['applied']
    assert not candidate['frequency_processing']['selective_mid_energy']['applied']
