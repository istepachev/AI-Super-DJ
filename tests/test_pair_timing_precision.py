import numpy as np
import pytest
from superdj.kick_grid import validate_overlap_attacks


def pair(outgoing_error,incoming_error,*,base=262.31349394277373,sparse=False):
    sets=([1,2,3,4,5,6,7,8,9,10,11,12],[1,3,5,7,13,14,15,16,17,18,19,20])if sparse else(range(1,32),range(1,32))
    clips=[]
    for role,offset,error,beats in [('tail',base,outgoing_error,sets[0]),('head',0.,incoming_error,sets[1])]:
        points=[{'role':role,'beat':i,'view':'primary','grid_output_seconds':offset+i*.5,
                 'observed_seconds':offset+i*.5+error}for i in beats]
        clips.append({'warp':{'post_render_transient_audit':{'final':{'measured':points}}}})
    edge={'from':'a','to':'b','tempo_technique':'shared_phase_sync','beats':32}
    return clips,edge


@pytest.mark.parametrize('base',[1.,262.31349394277373,3600.,10000.])
def test_exact_pair_boundary_ignores_only_timestamp_subtraction_roundoff(base):
    clips,edge=pair(.00275,-.00025,base=base)
    report=validate_overlap_attacks(clips,[edge])[0]
    assert report['maximum_pair_error_seconds']>=.003-8*np.finfo(float).eps*max(base+20,1.)
    assert report['tolerance_seconds']==.003 and not report['acoustic_tolerance_increased']
    assert report['timestamp_roundoff_bound_seconds']<1e-8


@pytest.mark.parametrize('sample_rate',[8000,44100,48000])
def test_one_real_detector_or_audio_sample_above_pair_limit_is_still_refused(sample_rate):
    clips,edge=pair(.00275+1/sample_rate,-.00025,base=3600.)
    with pytest.raises(ValueError)as result:validate_overlap_attacks(clips,[edge])
    assert result.value.evidence['paired_attack_audit']['maximum_pair_error_seconds']>.003


def test_individual_and_sparse_boundary_use_the_same_numerical_rule():
    clips,edge=pair(.003,0.)
    validate_overlap_attacks(clips,[edge])
    clips,edge=pair(.0015,-.0015,sparse=True)
    report=validate_overlap_attacks(clips,[edge])[0]
    assert report['verified_common_beats']==4 and 'sparse'in report['verification_kind']
    clips,edge=pair(.0015+1/48000,0.,sparse=True)
    with pytest.raises(ValueError,match='no adequate'):validate_overlap_attacks(clips,[edge])
