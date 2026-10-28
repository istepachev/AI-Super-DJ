from copy import deepcopy
import numpy as np
import pytest
from superdj.entry_handoff import late_entry_recipe
from superdj.transition_quality import TransitionFailure
from superdj import orchestration


def refusal(missing=range(4,20),*,outgoing=True):
    return TransitionFailure('a','b','unverified opening',{'local_rhythm_coverage':{'passed':False,'decks':[
        {'id':'a','role':'tail','sufficient_for_automatic_window':outgoing,'unverified_audible_beat_indices':[]},
        {'id':'b','role':'head','sufficient_for_automatic_window':False,'unverified_audible_beat_indices':list(missing)}]}})


@pytest.mark.parametrize('missing,expected',[(range(4,20),(12.,44.)),(range(5,36),(28.,60.))])
def test_bounded_opening_placement_keeps_the_same_soft_width(missing,expected):
    edge={'from':'a','to':'b','beats':128};error=refusal(missing);before=deepcopy(error.evidence)
    spec,evidence=late_entry_recipe(edge,[error])
    assert (spec['start_beat'],spec['end_beat'])==expected and spec['bias']==1
    assert spec['end_beat']-spec['start_beat']==32
    assert not evidence['missing_attacks_declared_verified']and error.evidence==before


@pytest.mark.parametrize('missing',[range(4,12),range(20,40),range(4,38),[True,2,3],list(range(4,20))+list(range(48,64))])
def test_short_middle_long_or_later_gaps_are_not_hidden(missing):
    assert late_entry_recipe({'from':'a','to':'b','beats':128},[refusal(missing)])is None


def test_an_unverified_old_foundation_or_short_overlap_is_not_eligible():
    assert late_entry_recipe({'from':'a','to':'b','beats':128},[refusal(outgoing=False)])is None
    assert late_entry_recipe({'from':'a','to':'b','beats':32},[refusal()])is None
    assert late_entry_recipe({'from':'a','to':'b','beats':128},[ValueError('unknown fronts')])is None
    native=refusal();native.evidence['stage']='source-only timing preflight'
    assert late_entry_recipe({'from':'a','to':'b','beats':128},[native])is None


@pytest.mark.parametrize('data',[None,{'local_rhythm_coverage':None},
    {'local_rhythm_coverage':{'passed':False,'decks':None}}])
def test_missing_diagnostics_do_not_replace_the_original_refusal(data):
    failure=ValueError('original refusal');failure.evidence=data
    assert late_entry_recipe({'from':'a','to':'b','beats':128},[failure])is None


@pytest.mark.parametrize('style',['zhilin','default'])
def test_existing_three_routes_remain_first_and_the_fourth_keeps_every_gate(monkeypatch,tmp_path,style):
    a=np.ones((32000,2),dtype='float32')*.01;b=a*.7;phase=np.linspace(0,1,len(a))
    edge={'from':'a','to':'b','beats':128,'shared_clock':{'identity':'preserve'}};before=deepcopy(edge);error=refusal();seen=[]
    def fronts(audio,*args,**kw):
        assert not kw['allow_pitch_changing_residual'];return audio,{'verified':True}
    def quality(audio,selected):
        spec=selected['whole_spectrum_handoff'];seen.append(spec)
        assert selected['shared_clock']==before['shared_clock']and len(audio)==len(a)
        if spec!={'start_beat':12.,'end_beat':44.,'bias':1.}:raise error
        return {'passed':True}
    monkeypatch.setattr(orchestration,'verify_blended_fronts',fronts)
    import superdj.entry_handoff as entry_module
    original_entry=entry_module.late_entry_recipe
    def detached(transition,errors):
        assert all(isinstance(value,dict)for value in errors)
        return original_entry(transition,errors)
    monkeypatch.setattr(entry_module,'late_entry_recipe',detached)
    _,selected=orchestration.whole_spectrum_alternative(a,b,8000,{}, {},edge,tmp_path,
        style=style,beat_phase=phase,initial_error=error,quality_check=quality)
    assert seen[:3]==[{'start_beat':0.,'end_beat':128.,'bias':3.},{'start_beat':0.,'end_beat':128.,'bias':1.},{'start_beat':4.,'end_beat':36.,'bias':1.}]
    assert len(seen)==4 and edge==before
    assert selected['quality_gate']['passed']and selected['frequency_processing']['actual_10_90_shared_beats']>=16
    assert selected['whole_spectrum_recovery']['selected_strategy']=='late_whole_spectrum_entry'


def test_a_passing_existing_route_wins_even_when_entry_diagnostics_are_eligible(monkeypatch,tmp_path):
    import superdj.entry_handoff as entry_module
    a=np.ones((16000,2),dtype='float32')*.01;b=a*.7;seen=[]
    def unexpected(*args):raise AssertionError('A passing old route must be retained')
    monkeypatch.setattr(entry_module,'late_entry_recipe',unexpected)
    monkeypatch.setattr(orchestration,'verify_blended_fronts',lambda audio,*args,**kw:(audio,{'verified':True}))
    def quality(audio,selected):seen.append(selected['whole_spectrum_handoff']);return {'passed':True}
    _,selected=orchestration.whole_spectrum_alternative(a,b,8000,{}, {},{'from':'a','to':'b','beats':128},tmp_path,
        style='zhilin',beat_phase=np.linspace(0,1,len(a)),initial_error=refusal(),quality_check=quality)
    assert seen==[{'start_beat':0.,'end_beat':128.,'bias':3.}]
    assert selected['whole_spectrum_recovery']['selected_strategy']=='wide_whole_spectrum'


def test_new_placement_cannot_waive_a_real_front_refusal(monkeypatch,tmp_path):
    audio=np.ones((16000,2),dtype='float32')*.01;calls=[];error=refusal()
    def refused(*args,**kw):calls.append(1);raise ValueError('actual attack still late')
    monkeypatch.setattr(orchestration,'verify_blended_fronts',refused)
    with pytest.raises(ValueError,match='still late'):
        orchestration.whole_spectrum_alternative(audio,audio,8000,{}, {},{'from':'a','to':'b','beats':128},tmp_path,
            style='zhilin',beat_phase=np.linspace(0,1,len(audio)),initial_error=error,quality_check=lambda *args:{'passed':True})
    assert len(calls)==5


def test_missing_complete_quality_callback_keeps_the_original_three_routes(monkeypatch,tmp_path):
    audio=np.ones((16000,2),dtype='float32')*.01;calls=[]
    def refused(*args,**kw):calls.append(1);raise ValueError('actual attack still late')
    monkeypatch.setattr(orchestration,'verify_blended_fronts',refused)
    with pytest.raises(ValueError,match='still late'):
        orchestration.whole_spectrum_alternative(audio,audio,8000,{}, {},{'from':'a','to':'b','beats':128},tmp_path,
            style='zhilin',beat_phase=np.linspace(0,1,len(audio)),initial_error=refusal(),quality_check=None)
    assert len(calls)==3


def test_second_position_is_checked_after_the_first_one_is_refused(monkeypatch,tmp_path):
    audio=np.ones((16000,2),dtype='float32')*.01;seen=[];error=refusal(range(3,32))
    monkeypatch.setattr(orchestration,'verify_blended_fronts',lambda pcm,*args,**kw:(pcm,{'verified':True}))
    def quality(pcm,edge):
        spec=edge['whole_spectrum_handoff'];seen.append(spec)
        if spec!={'start_beat':28.,'end_beat':60.,'bias':1.}:raise error
        return {'passed':True}
    _,selected=orchestration.whole_spectrum_alternative(audio,audio,8000,{}, {},{'from':'a','to':'b','beats':64},tmp_path,
        style='zhilin',beat_phase=np.linspace(0,1,len(audio)),initial_error=error,quality_check=quality)
    assert seen[-2:]==[{'start_beat':24.,'end_beat':56.,'bias':1.},{'start_beat':28.,'end_beat':60.,'bias':1.}]
    assert len(seen)==5 and selected['quality_gate']['passed']


def test_later_adjustment_never_extends_the_lead_beyond_28_beats(monkeypatch,tmp_path):
    audio=np.ones((16000,2),dtype='float32')*.01;calls=[]
    def refused(*args,**kw):calls.append(1);raise ValueError('front still late')
    monkeypatch.setattr(orchestration,'verify_blended_fronts',refused)
    with pytest.raises(ValueError,match='still late'):
        orchestration.whole_spectrum_alternative(audio,audio,8000,{}, {},{'from':'a','to':'b','beats':128},tmp_path,
            style='zhilin',beat_phase=np.linspace(0,1,len(audio)),initial_error=refusal(range(5,36)),quality_check=lambda *args:{'passed':True})
    assert len(calls)==4
