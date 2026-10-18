"""Anonymous final-map intervals from a reported rough musical overlap.

No audio, track identities, paths or source positions are distributed. Each
individual correction was below the former 15% interval limit; their rapid
alternation is the regression. A passing primary-hit audit cannot waive it.
"""
from copy import deepcopy

import numpy as np
import pytest

from superdj.timing_safety import audit_warp_rates, initialize_rate_reference


CASES = [
    (
        [5493,5491,5491,5492,4925,4924,4924,4924,5502,5503,5503,5503,4914],
        [5177,5178,5177,5178,5360,5360,5361,5361,5216,5216,5216,5217,5236],
        [.9895029340384989,.9895029340384989,.9895029340384989,
         .9893149942599541,.9893149942599541,.9893149942599541,
         .9891271258599806,.9891271258599806,.9889393287979226,
         .9889393287979226,.9889393287979226,.9887516030331546,.9887516030331546],
        .14371658105858298,
    ),
    (
        [5074,5076,5075,5516,5517,5517,5515,5335,5335,5335,5334,5336,5336],
        [5267,5267,5268,5269,5268,5269,5269,5290,5290,5291,5292,5290,5290],
        [1.012449248915175,1.012449248915175,1.0122570603713414,
         1.0122570603713414,1.0122570603713414,1.0120649447781793,
         1.0120649447781793,1.0120649447781793,1.0118729020941608,
         1.0118729020941608,1.011680932277789,1.011680932277789,1.011680932277789],
        .0830743869992443,
    ),
]


def make_map(source_steps, output_steps, intended, subdivisions):
    output=np.r_[0,np.cumsum(output_steps)]
    source=np.r_[0,np.cumsum(source_steps)]
    if subdivisions>1:
        nodes=np.arange(len(output)-1)[:,None]+np.arange(subdivisions)/subdivisions
        nodes=np.r_[nodes.ravel(),len(output)-1]
        source=np.rint(np.interp(nodes,np.arange(len(source)),source)).astype(int)
        output=np.rint(np.interp(nodes,np.arange(len(output)),output)).astype(int)
    warp={'anchors':[{'source_frame':int(a),'output_frame':int(b)} for a,b in zip(source,output)],
          'input_output_beat_mapping':{'head':[],'tail':[]}}
    initialize_rate_reference(warp,44100,limit=.15)
    warp['timing_rate_reference']['expected_musical_rates']=np.repeat(intended,subdivisions).tolist()
    return warp


@pytest.mark.parametrize('source_steps,output_steps,intended,reported_span',CASES,ids=['outgoing','incoming'])
@pytest.mark.parametrize('subdivisions',[1,4],ids=['original-controls','subdivided-controls'])
def test_reported_final_overlap_pulses_fail_even_after_successful_hit_alignment(
        source_steps,output_steps,intended,reported_span,subdivisions):
    warp=make_map(source_steps,output_steps,intended,subdivisions)
    warp['post_render_kick_audit']={'final':{'maximum_measured_drift_seconds':.002},'tolerance_seconds':.003}
    before=deepcopy(warp['anchors'])
    with pytest.raises(ValueError,match='Short-term acoustic timing variation'):
        audit_warp_rates(warp,44100,stage='reported completed overlap regression')
    result=warp['actual_timing_rate_audit']
    assert result['maximum_timing_correction_change']<.15
    assert result['short_term_timing_continuity']['maximum_correction_rate_span']==pytest.approx(reported_span,abs=.002)
    assert not result['passed']
    assert warp['anchors']==before


@pytest.mark.parametrize('source_steps,output_steps,intended,reported_span',CASES,ids=['outgoing','incoming'])
def test_intended_smooth_curve_from_the_same_overlap_remains_unchanged(
        source_steps,output_steps,intended,reported_span):
    smooth=np.diff(np.rint(np.r_[0,np.cumsum(np.asarray(output_steps)*intended)])).astype(int)
    warp=make_map(smooth,output_steps,intended,1)
    before=deepcopy(warp['anchors'])
    result=audit_warp_rates(warp,44100,stage='corresponding smooth musical curve')
    assert result['passed']
    assert result['short_term_timing_continuity']['maximum_correction_rate_span']<.001
    assert warp['anchors']==before
