"""Run against the selected baseline first; later retain in the candidate suite."""
from test_reliability import source_plan
from superdj.recovery import alternative
from superdj.transition_quality import TransitionFailure


def test_tail_retry_cannot_retain_the_same_cued_exit_by_moving_the_native_intro():
    rows,plan=source_plan(3)
    old=plan['sections']['0']
    rows[0]['cue_candidates'].append({'time':old['end'],'confidence':1.,'novelty':1.})
    failure=TransitionFailure('0','1','no attacks',{'failed_track':'0','failed_roles':['tail']})
    candidate,report=alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert candidate['sections']['0']['end']!=old['end']
    assert candidate['sections']['0']['start']==old['start']
    assert candidate['sections']['2']['end']==plan['sections']['2']['end']
    assert report['boundary']=='end'


def test_head_retry_keeps_the_native_ending_and_the_other_native_intro():
    rows,plan=source_plan(2)
    old=plan['sections']['1']
    rows[1]['cue_candidates'].append({'time':old['start'],'confidence':1.,'novelty':1.})
    failure=TransitionFailure('0','1','no attacks',{'failed_track':'1','failed_roles':['head']})
    candidate,_=alternative(plan,rows,failure,{},1,sample_rate=48000,ffmpeg=None)
    assert candidate['sections']['1']['start']!=old['start']
    assert candidate['sections']['1']['end']==old['end']
    assert candidate['sections']['0']['start']==plan['sections']['0']['start']
