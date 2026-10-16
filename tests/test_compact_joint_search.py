"""A short supported musical core can require two coordinated overlap changes."""
import pytest

from superdj.compact import arrange_sync
from superdj.planner import plan_tracks
from superdj.render import RenderError
from test_new_workflow import record


def install_core(monkeypatch):
    def profile(row,**_):
        if row['id'] != 'middle':return {'executed':False}
        return {'executed':True,'reference_dbfs':-10.,'threshold_dbfs':-26.,
            'rows':[{'time':i,'end':i+.5,'bass_dbfs':-10. if 80<=i<120 else -80.}
                    for i in [j*.5 for j in range(480)]]}
    monkeypatch.setattr('superdj.compact.bass_profile',profile)


def test_two_automatic_overlaps_are_searched_together_before_rejecting_core(monkeypatch):
    install_core(monkeypatch)
    rows=[record('first',420),record('middle',240),record('last',420)]
    plan=plan_tracks(rows,mode='fixed')
    result=arrange_sync(plan,rows)
    assert result['order']==plan['order']
    section=result['sections']['middle']
    assert section['predicted_solo_seconds']>=8
    assert section['predicted_play_seconds']<=300
    assert all(edge['overlap_bars']<32 for edge in result['edges'])
    assert 'sections' not in plan


def test_joint_search_cannot_shorten_explicit_overlaps_or_report_another_track(monkeypatch):
    install_core(monkeypatch)
    rows=[record('short_first',150),record('middle',240),record('last',420)]
    plan=plan_tracks(rows,mode='fixed')
    for edge in plan['edges']:edge['overlap_bars']=32
    with pytest.raises(RenderError,match='^middle:'):
        arrange_sync(plan,rows)
