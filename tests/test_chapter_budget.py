from copy import deepcopy
import pytest
from superdj.compact import arrange_sync
from superdj.planner import plan_tracks
from superdj.source_search import fresh_arrangement_alternative
from superdj.transition_quality import TransitionFailure
from test_new_workflow import record

def plan(rows):
    p = plan_tracks(rows, mode='fixed')
    p['mixing_preferences'] = {'duration_budget_basis':'chapter interval',
        'target_play_seconds':210., 'maximum_play_seconds':300.}
    return p

def test_long_ordinary_sources_get_three_to_four_minute_chapters():
    rows = [record(c,600) for c in 'abc']
    p = plan(rows)
    untouched = deepcopy(p)
    result = arrange_sync(p, rows)
    assert p == untouched
    assert all(e['overlap_bars']==32 for e in result['edges'])
    for s in result['sections'].values():
        assert 180 <= s['predicted_chapter_seconds'] <= 240
        assert s['predicted_play_seconds'] <= 300
        assert not s['duration_exception']
    middle = result['sections']['b']
    assert middle['predicted_play_seconds'] > middle['predicted_chapter_seconds']+50
    assert middle['predicted_solo_seconds'] < middle['predicted_chapter_seconds']-50

def test_existing_full_deck_requests_keep_their_previous_meaning():
    rows=[record(c,600) for c in 'abc']
    p=plan_tracks(rows,mode='fixed')
    result=arrange_sync(p,rows)
    assert all(180<=s['predicted_play_seconds']<=240 for s in result['sections'].values())
    assert result['sections']['b']['predicted_chapter_seconds'] < 180

def test_energy_cannot_buy_an_unnecessary_short_chapter():
    rows=[record(c,600) for c in 'abc']
    for row in rows:
        row['structure']=[{'time':t,'energy':1. if 100<=t<160 else .001,
                           'vocal_activity':0.} for t in row['downbeats']]
        row['cue_candidates']=[{'time':100.25,'confidence':1.,'novelty':1.},
                               {'time':240.25,'confidence':1.,'novelty':1.}]
    result=arrange_sync(plan(rows),rows)
    assert all(180<=s['predicted_chapter_seconds']<=240 for s in result['sections'].values())

def test_true_short_source_exception_retains_available_music():
    rows=[record('a',600),record('b',150),record('c',600)]
    p=plan(rows)
    # Approved overlap length cannot be shortened merely to hide a short source.
    for e in p['edges']:e['overlap_bars']=16
    result=arrange_sync(p,rows)
    s=result['sections']['b']
    assert s['duration_exception']
    assert 'native source' in s['chapter_duration_exception_reason']
    assert s['predicted_play_seconds']>=140
    assert s['predicted_chapter_seconds']<180
    assert all(e['overlap_bars']==16 for e in result['edges'])

def test_recovery_keeps_chapter_budget_and_full_deck_cap():
    rows=[record(c,600) for c in 'abc']
    p=arrange_sync(plan(rows),rows)
    for s in p['sections'].values():s['locked_start']=s['locked_end']=False
    error=TransitionFailure('a','b','another clean source exit required',
        {'failed_track':'a','failed_roles':['tail']})
    result, selection=fresh_arrangement_alternative(p,rows,error,{},0,sample_rate=8000,ffmpeg=None)
    assert selection['changed_track']=='a'
    assert result['mixing_preferences']['duration_budget_basis']=='chapter interval'
    assert all(180<=s['predicted_chapter_seconds']<=240 for s in result['sections'].values())
    assert all(s['predicted_play_seconds']<=300 for s in result['sections'].values())

def test_high_requested_chapter_never_reopens_six_minute_decks():
    rows=[record(c,600) for c in 'abc']
    result=arrange_sync(plan(rows),rows,target_play_seconds=240)
    assert all(180<=s['predicted_chapter_seconds']<=240 for s in result['sections'].values())
    assert all(s['predicted_play_seconds']<=300 for s in result['sections'].values())

def test_native_audio_clock_adjustment_updates_the_chapter_prediction(tmp_path):
    import numpy as np
    from test_kick_grid import fixture
    from superdj.source_clock import calibrate_render_windows
    from superdj.render import _Window
    from superdj.sync import shared_clock
    path,warp,sr=fixture(tmp_path)
    times=np.asarray([x['source_seconds'] for x in warp['input_output_beat_mapping']['head']])+.030
    window=_Window(times.copy(),8,4,120,1,clock=shared_clock(120,120,len(times)-1,sr))
    sections={'b':{'start':float(times[0]),'end':18.,'locked_start':False,
        'locked_end':False,'predicted_solo_seconds':2.,'predicted_play_seconds':17.,
        'predicted_chapter_seconds':10.}}
    records={k:{'id':k,'path':str(path),'duration':18.} for k in ('a','b')}
    result,reports=calibrate_render_windows(records,{},['a','b'],{'a':None,'b':window},
        {'a':None,'b':None},sections,[{'from':'a','to':'b'}],[{}],ffmpeg='unused',directory=tmp_path)
    assert reports['b']['head']
    delta=float(times[-1]-window.times[-1])
    assert abs(delta)>.001
    assert result['b']['predicted_chapter_seconds']==pytest.approx(10.+delta)
    assert result['b']['predicted_play_seconds']==pytest.approx(17.+delta)
    assert sections['b']['predicted_chapter_seconds']==10.

@pytest.mark.parametrize('locked',[False,True])
def test_recovered_late_entrance_searches_a_later_exit_before_allowing_short_play(locked):
    from superdj.chapter_duration import retain_chapter_target
    rows=[record(c,600) for c in 'abc']
    p=arrange_sync(plan(rows),rows)
    old=p['sections']['b']
    p['edges'][0]['cue_locks']['incoming_start_seconds']=old['start']+64
    shortened=arrange_sync(p,rows)
    for s in shortened['sections'].values():s['locked_start']=s['locked_end']=False
    shortened['sections']['b']['locked_end']=locked
    assert shortened['sections']['b']['predicted_chapter_seconds']<180
    changed,proof=retain_chapter_target(shortened,rows,'b','head',sample_rate=8000,ffmpeg=None)
    if locked:
        assert changed is shortened and proof is None
    else:
        assert proof and 180<=changed['sections']['b']['predicted_chapter_seconds']<=240
        assert changed['sections']['b']['start']==shortened['sections']['b']['start']
        assert changed['sections']['b']['end']>shortened['sections']['b']['end']
        assert all(s['predicted_play_seconds']<=300 for s in changed['sections'].values())
        assert all(e['overlap_bars']==32 for e in changed['edges'])
        for k in ('a','c'):
            assert changed['sections'][k]['start']==shortened['sections'][k]['start']
            assert changed['sections'][k]['end']==shortened['sections'][k]['end']
