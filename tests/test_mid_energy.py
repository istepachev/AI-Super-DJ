import math
import numpy as np
import pytest
from superdj.mid_energy import decide,apply_report,LAW
from superdj.mixing import blend,envelopes,recorded_envelopes
from superdj.percussion import percussion_gains,verification_components

def tones(seconds=8):
    sr=8000;t=np.arange(sr*seconds)/sr
    return sr,np.repeat((.1*np.sin(2*np.pi*420*t))[:,None],2,axis=1),np.repeat((.1*np.sin(2*np.pi*1700*t))[:,None],2,axis=1)

@pytest.mark.parametrize('incoming_scale',[1.,.1,3.])
def test_separate_mid_parts_retain_native_energy_without_center_valley(incoming_scale):
    sr,a,b=tones();b*=incoming_scale;n=len(a);g,_=envelopes(n,128,style='zhilin');zero=np.zeros_like(a)
    report=decide(a,b,sr,g,vocal_out=zero,vocal_in=zero);assert report['applied']
    updated=apply_report(g,{'adaptive_mid_energy':report});p=g['mid_in']/(g['mid_out']+g['mid_in'])
    np.testing.assert_allclose(updated['mid_out']**2+updated['mid_in']**2,1.,atol=1e-14)
    actual=a*updated['mid_out']+b*updated['mid_in'];expected=(1-p)*.1**2/2+p*(.1*incoming_scale)**2/2
    for first in range(sr*2,sr*6,sr//4):
        observed=np.mean(actual[first:first+sr//4]**2);target=np.mean(expected[first:first+sr//4])
        assert abs(10*np.log10(observed/target))<.12
    for key in ('bass_out','bass_in','high_out','high_in'):assert np.array_equal(g[key],updated[key])
    assert max(updated['mid_out'].max(),updated['mid_in'].max())<=1
    assert np.max(updated['mid_out']+updated['mid_in'])<=math.sqrt(2.)+1e-12
    assert np.all(np.diff(updated['mid_in'][:,0])>=-1e-12)
    assert np.all(np.diff(updated['mid_out'][:,0])<=1e-12)

def test_coherent_or_competing_foreground_retains_old_law():
    sr,a,b=tones();g,_=envelopes(len(a),128,style='zhilin');zero=np.zeros_like(a)
    same=decide(a,a,sr,g,vocal_out=zero,vocal_in=zero)
    assert not same['applied']and same['spectral_overlap_90th_percentile']>.99
    voices=decide(a,b,sr,g,vocal_out=a*.8,vocal_in=b*.8)
    assert not voices['applied']and voices['sustained_competing_foreground_seconds']>=2
    unknown=decide(a,b,sr,g);assert not unknown['applied']
    assert apply_report(g,{'adaptive_mid_energy':same})is g

def test_stereo_antiphase_cannot_hide_actual_mid_occupancy():
    sr,a,b=tones();a[:,1]*=-1;b[:,1]*=-1;g,_=envelopes(len(a),128,style='zhilin')
    report=decide(a,b,sr,g,vocal_out=a*0,vocal_in=b*0)
    assert report['applied']and report['simultaneously_active_half_second_windows']>=4

def test_actual_drum_contributions_and_audits_use_identical_mid_law():
    sr,a,b=tones();t=np.arange(len(a))/sr;da=np.repeat((.06*np.sin(2*np.pi*700*t))[:,None],2,axis=1);db=da*.7;zero=np.zeros_like(a)
    mixed,frequency=blend(a+da,b+db,sr,beats=128,style='zhilin',drums_out=da,drums_in=db,percussion_policy='smooth',vocal_out=zero,vocal_in=zero)
    assert frequency['adaptive_mid_energy']['applied']
    transition={'beats':128,'frequency_processing':frequency}
    gains,_=recorded_envelopes(len(a),128,transition,style='zhilin',beat_phase=np.linspace(0,1,len(a)))
    gains=percussion_gains(gains,np.linspace(0,1,len(a)),128,'smooth')
    parts=verification_components(a+da,b+db,sr,gains,drums_out=da,drums_in=db)
    np.testing.assert_allclose(parts['head']+parts['tail'],mixed,atol=3e-8)
    base,_=envelopes(len(a),128,style='zhilin');base=percussion_gains(base,None,128,'smooth')
    assert np.array_equal(gains['drums_in'],base['drums_in'])
    assert frequency['musical_handoff_continuity']['effective_10_90_shared_beats']['percussion']>=16

def test_explicit_legacy_variant_is_retained_without_sample_changes():
    sr,a,b=tones();zero=np.zeros_like(a)
    old,_=blend(a,b,sr,style='zhilin',bass_handoff_beats=16)
    legacy,report=blend(a,b,sr,style='zhilin',bass_handoff_beats=16,mid_energy_mode='legacy',vocal_out=zero,vocal_in=zero)
    assert np.array_equal(old,legacy)
    assert not report['adaptive_mid_energy']['applied']
    expected,_=envelopes(len(a),128,style='zhilin',bass_handoff_beats=16)
    actual,_=recorded_envelopes(len(a),128,{'frequency_processing':{'bass_handoff_beats':16}},style='zhilin',beat_phase=None)
    for key in expected:assert np.array_equal(expected[key],actual[key])

@pytest.mark.parametrize('style,minimum,width',[('default',8,16),('zhilin',16,32)])
def test_factory_drum_handoff_has_a_real_broad_effective_span(style,minimum,width):
    sr,a,b=tones();_,report=blend(a,b,sr,beats=64,style=style,drums_out=a*.4,drums_in=b*.4,percussion_policy='smooth')
    assert report['bass_handoff_beats']>=width
    assert report['musical_handoff_continuity']['effective_10_90_shared_beats']['percussion']>=minimum

def test_report_replay_refuses_wrong_geometry_or_unknown_law():
    sr,a,b=tones();g,_=envelopes(len(a),128,style='zhilin');zero=np.zeros_like(a)
    report=decide(a,b,sr,g,vocal_out=zero,vocal_in=zero)
    for patch in ({'frames':len(a)+1},{'law':'unknown'},{'strength':2},{'maximum_source_gain':2}):
        with pytest.raises(ValueError,match='actual overlap'):apply_report(g,{'adaptive_mid_energy':{**report,**patch}})
    with pytest.raises(ValueError,match='auto or legacy'):blend(a,b,sr,mid_energy_mode='bad',whole_spectrum_handoff={'start_beat':0,'end_beat':64})

def test_local_coverage_sees_earlier_audible_energy_supported_mid():
    from superdj.transition_quality import local_reliability
    count=64;n=6400;phase=np.linspace(0,1,n);frames=np.linspace(0,n,count+1)
    clips=[{'id':identity,'warp':{'post_render_transient_audit':{'final':{'measured':[{'role':role,'beat':i,'delay_seconds':0.}for i in range(start,count)]}}}}for identity,role,start in [('a','tail',1),('b','head',10)]]
    edge={'from':'a','to':'b','beats':count,'tempo_technique':'shared_phase_sync','shared_clock':{'beats':count,'beat_output_frames':frames.tolist()}}
    audio=np.ones((n,2),dtype='float32')*.1
    old=local_reliability(clips,edge,audio,audio,1000,style='zhilin',beat_phase=phase)
    report={'applied':True,'law':LAW,'frames':n,'strength':1.,'maximum_source_gain':1.,'maximum_pair_gain_sum':math.sqrt(2.)}
    edge['frequency_processing']={'adaptive_mid_energy':report}
    new=local_reliability(clips,edge,audio,audio,1000,style='zhilin',beat_phase=phase)
    assert old['passed']and not new['passed']
    assert new['decks'][1]['longest_unverified_audible_beats']>old['decks'][1]['longest_unverified_audible_beats']

def test_original_ordinary_variant_is_checked_before_changing_other_routing(tmp_path,monkeypatch):
    from superdj import orchestration
    from superdj.transition_quality import TransitionFailure
    sr,a,b=tones();zero=np.zeros_like(a);phase=np.linspace(0,1,len(a));before_a=a.copy();before_b=b.copy()
    _,frequency=blend(a,b,sr,style='zhilin',beats=128,vocal_out=zero,vocal_in=zero)
    assert frequency['adaptive_mid_energy']['applied']
    edge={'from':'a','to':'b','beats':128,'frequency_processing':frequency};fronts=[];checks=[]
    def front(pcm,rate,wa,wb,gains,directory,**kw):
        assert np.max(gains['mid_in']+gains['mid_out'])<=1+1e-12
        np.testing.assert_allclose(kw['components']['head']+kw['components']['tail'],pcm,atol=2e-8)
        fronts.append(1);return pcm,{'passed':True}
    def quality(pcm,selected):checks.append(selected);return {'passed':True}
    monkeypatch.setattr(orchestration,'verify_blended_fronts',front)
    _,selected=orchestration.legacy_mid_alternative(a,b,sr,{}, {},edge,tmp_path,style='zhilin',beat_phase=phase,
        percussion_args={},initial_error=ValueError('newly audible uncertain figure'),quality_check=quality,
        mid_energy_args={'vocal_out':zero,'vocal_in':zero})
    assert len(fronts)==len(checks)==1 and selected['quality_gate']['passed']
    assert selected['mid_energy_mode']=='legacy'and not selected['frequency_processing']['adaptive_mid_energy']['applied']
    assert selected['mid_energy_legacy_recovery']['source_clock_overlap_and_band_positions_unchanged']
    assert np.array_equal(a,before_a)and np.array_equal(b,before_b)
    def fail(*args):raise TransitionFailure('a','b','local rhythm coverage still missing')
    with pytest.raises(TransitionFailure,match='still missing'):
        orchestration.legacy_mid_alternative(a,b,sr,{}, {},edge,tmp_path,style='zhilin',beat_phase=phase,
            percussion_args={},initial_error=ValueError('first'),quality_check=fail,
            mid_energy_args={'vocal_out':zero,'vocal_in':zero})
