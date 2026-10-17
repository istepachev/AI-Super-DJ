import numpy as np
import pytest
from superdj.integrity import _timing_limit_exceeded


def test_exact_three_millisecond_boundary_is_not_a_false_master_failure():
    # Actual C28 observation: subtracting float timestamps added 0.0000000007 ms
    # to a 2.75 ms prior error. A 0.25 ms additional bound still totals 3 ms.
    prior=.0027500000000006963
    assert prior+.00025>.003  # The old strict-float comparison rejected it.
    assert not _timing_limit_exceeded(prior+.00025,.003,reference_seconds=165870043/44100)


@pytest.mark.parametrize('time',[1.,300.,3600.,86400.])
def test_one_actual_detector_sample_above_the_limit_still_stops_delivery(time):
    assert _timing_limit_exceeded(.003+1/8000,.003,reference_seconds=time)
    assert _timing_limit_exceeded(.003+1e-8,.003,reference_seconds=time)
    assert not _timing_limit_exceeded(.002875,.003,reference_seconds=time)
    assert not _timing_limit_exceeded(.003,.003,reference_seconds=time)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.001])
def test_invalid_measurement_cannot_pass_a_timing_comparison(value):
    with pytest.raises(ValueError,match='finite nonnegative'):
        _timing_limit_exceeded(value,.003,reference_seconds=100.)
