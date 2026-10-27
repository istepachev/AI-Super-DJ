"""Distributed evidence must survive at the audible beginning and handoff."""
from copy import deepcopy
import numpy as np
import pytest
from superdj.transition_quality import local_reliability,reliability,coverage_failure_context,verify_overlap,TransitionFailure


def fixture(head=None,tail=None,beats=128):
    sr=2000;n=beats*1000
    pcm=np.repeat((.08*np.sin(2*np.pi*87*np.arange(n)/sr))[:,None],2,axis=1).astype('float32')
    phase=np.linspace(0,1,n)
    edge={'from':'a','to':'b','tempo_technique':'shared_phase_sync','beats':beats,
        'shared_clock':{'beats':beats,'beat_output_frames':list(range(0,n+1,1000))},'frequency_processing':{}}
    clips=[]
    for identity,role,known in [('a','tail',tail),('b','head',head)]:
        known=set(range(1,beats)if known is None else known)
        final={'measured':[{'role':role,'beat':i,'delay_seconds':0.}for i in sorted(known)],
            'unverified':[{'role':role,'beat':i}for i in range(1,beats)if i not in known]}
        clips.append({'id':identity,'warp':{'post_render_transient_audit':{'final':final}}})
    return clips,edge,pcm,sr,phase


def audit(values,style='zhilin',outgoing=None,incoming=None):
    clips,edge,pcm,sr,phase=values
    return local_reliability(clips,edge,pcm if outgoing is None else outgoing,
        pcm if incoming is None else incoming,sr,style=style,beat_phase=phase)


def test_reported_late_35_matches_cannot_certify_first_87_unknown_beats():
    # Same anonymous coverage pattern as the complained-about real overlap.
    known=set(range(88,128))-{103,116,117,118,119}
    values=fixture(head=known);clips,edge,*_=values
    assert len(known)==35 and reliability(clips,edge)['passed']
    report=audit(values)
    assert not report['passed'] and report['decks'][0]['sufficient_for_automatic_window']
    head=report['decks'][1]
    assert head['longest_unverified_audible_beats']>8
    assert any(not w['passed']and w['end_beat']<=88 for w in head['windows'])
    owner=coverage_failure_context({'reliability':reliability(clips,edge),'local_rhythm_coverage':report},edge)
    assert owner['failed_track']=='b' and owner['failed_roles']==['head']


@pytest.mark.parametrize('role,missing',[('head',range(1,25)),('head',range(49,75)),('tail',range(1,25)),('tail',range(95,128))])
def test_good_events_elsewhere_never_hide_an_audible_local_gap(role,missing):
    values=fixture(**{role:set(range(1,128))-set(missing)})
    # Keep both routes audible throughout, to test the outgoing final region too.
    values[1]['whole_spectrum_handoff']={'start_beat':0.,'end_beat':128.,'bias':1.}
    report=audit(values)
    assert not report['passed']
    failed=[r['role']for r in report['decks']if not r['sufficient_for_automatic_window']]
    assert role in failed


@pytest.mark.parametrize('style',['default','zhilin'])
@pytest.mark.parametrize('method',['ordinary','positioned','wide','early'])
def test_existing_gradual_routing_variants_remain_available_and_pass_distributed_evidence(style,method):
    values=fixture();edge=values[1];before=deepcopy(edge)
    if method=='positioned':edge['handoff_windows']={'bass':[64,88]}
    if method=='wide':edge['whole_spectrum_handoff']={'start_beat':0.,'end_beat':128.,'bias':3.}
    if method=='early':edge['whole_spectrum_handoff']={'start_beat':4.,'end_beat':36.,'bias':1.}
    exact=deepcopy(edge)
    assert audit(values,style)['passed'] and edge==exact
    assert edge['beats']==before['beats']


def test_distributed_sparse_native_figures_are_not_forced_into_a_kick_on_every_beat():
    known=range(1,128,4)
    assert audit(fixture(head=known,tail=known))['passed']


def test_short_native_fill_remains_unknown_but_does_not_abort_a_good_overlap():
    values=fixture(head=set(range(1,128))-set(range(60,66)))
    report=audit(values)
    assert report['passed'] and 60 in report['decks'][1]['unverified_audible_beat_indices']


def test_unknown_events_on_a_muted_route_do_not_manufacture_a_failure_or_verified_beats():
    values=fixture(head=range(90,128))
    values[1]['whole_spectrum_handoff']={'start_beat':90.,'end_beat':128.,'bias':1.}
    report=audit(values)
    assert report['passed']
    assert report['decks'][1]['audible_beats']<38


def test_quiet_native_break_is_preserved_without_inventing_recognized_hits():
    values=fixture(head=set(range(1,128))-set(range(45,75)))
    quiet=values[2].copy();quiet[44*1000:75*1000]=0
    report=audit(values,incoming=quiet)
    assert report['passed'] and report['decks'][1]['quiet_input_beats']>=30


def test_zero_local_measurements_cannot_be_called_verified():
    assert not audit(fixture(head=[]))['passed']


def test_duplicate_views_of_the_same_beat_do_not_inflate_local_coverage():
    values=fixture(head=range(88,128));final=values[0][1]['warp']['post_render_transient_audit']['final']
    final['measured']*=10
    assert not audit(values)['passed']


@pytest.mark.parametrize('bad',[None,float('nan'),.03])
def test_unknown_or_late_timestamps_are_never_local_alignment_evidence(bad):
    values=fixture()
    final=values[0][1]['warp']['post_render_transient_audit']['final']
    for event in final['measured']:event['delay_seconds']=bad
    assert not audit(values)['passed']


def test_missing_shared_clock_is_a_format_error_not_a_musical_cue_recovery():
    values=fixture();values[1]['shared_clock'].pop('beat_output_frames')
    with pytest.raises(ValueError,match='actual shared clock'):audit(values)


def test_new_guard_is_in_the_real_overlap_gate_and_preserves_failed_owner(monkeypatch):
    import superdj.transition_quality as quality
    values=fixture(head=range(88,128));clips,edge,pcm,sr,phase=values
    monkeypatch.setattr(quality,'audit_energy',lambda *a:{'passed':True})
    monkeypatch.setattr(quality,'audit_vocals',lambda *a,**k:{'passed':True})
    with pytest.raises(TransitionFailure,match='local rhythm coverage')as error:
        verify_overlap(pcm,pcm,pcm,sr,clips,edge,style='zhilin',beat_phase=phase)
    assert error.value.evidence['reliability']['passed']
    assert error.value.evidence['failed_track']=='b'


def test_local_guard_code_changes_invalidate_cached_overlap_checks(tmp_path):
    from superdj.execution import stage_fingerprint
    # A minimal transitive-import graph, independent of workstation versions.
    (tmp_path/'render.py').write_text('from .transition_quality import verify_overlap\n')
    module=tmp_path/'transition_quality.py';module.write_text('def verify_overlap():return 1\n')
    before=stage_fingerprint('overlap',source=tmp_path)
    module.write_text('def verify_overlap():return 2\n')
    assert stage_fingerprint('overlap',source=tmp_path)!=before
