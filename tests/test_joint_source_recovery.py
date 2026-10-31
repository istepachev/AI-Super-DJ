from copy import deepcopy
import pytest
from test_reliability import source_plan
from superdj.recovery import alternative
from superdj.render import RenderError
from superdj.transition_quality import TransitionFailure


def test_two_failed_interior_windows_move_together_without_changing_neighbours():
    rows,plan=source_plan(3);before=deepcopy(plan)
    failure=TransitionFailure('0','1','both source windows refused',
        {'failed_track':'1','failed_roles':['head','tail']})
    updated,report=alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert plan==before and updated['order']==plan['order']
    assert report['boundary']=='start,end'
    assert all(updated['sections']['1'][b]!=plan['sections']['1'][b] for b in ('start','end'))
    assert all(updated['sections'][i][b]==plan['sections'][i][b] for i in ('0','2') for b in ('start','end'))
    assert [e['overlap_bars'] for e in updated['edges']]==[e['overlap_bars'] for e in plan['edges']]
    assert all(s['predicted_play_seconds']<=300 for s in updated['sections'].values())
    assert not any(s['locked_start'] or s['locked_end'] for s in updated['sections'].values())


@pytest.mark.parametrize('flag',['locked_start','locked_end'])
def test_joint_recovery_cannot_move_or_ignore_either_failed_user_lock(flag):
    rows,plan=source_plan(3);plan['sections']['1'][flag]=True;before=deepcopy(plan)
    failure=TransitionFailure('0','1','both source windows refused',
        {'failed_track':'1','failed_roles':['head','tail']})
    with pytest.raises(RenderError,match='locked'):
        alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert plan==before


def test_joint_recovery_retains_refused_physical_events_for_both_roles():
    rows,plan=source_plan(3);section=plan['sections']['1']
    rejected={'head':[section['start']+8], 'tail':[section['end']-8]}
    failure=TransitionFailure('0','1','both source windows refused',
        {'failed_track':'1','failed_roles':['head','tail'],'canonical_attack_rejections':rejected})
    updated,_=alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert updated['rejected_attack_window_events']['1']==rejected
