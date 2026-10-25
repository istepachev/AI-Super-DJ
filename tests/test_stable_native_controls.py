from copy import deepcopy
import numpy as np
import pytest
from test_kick_grid import fixture
from superdj.kick_grid import refine_source_controls
from superdj.timing_safety import SourceWindowFailure

def evidence(path,**extra):
    return {'head':{'path':str(path),'source_start_seconds':0.,**extra}}

def test_disabled_option_keeps_existing_native_control_maps(tmp_path):
    path,warp,sr=fixture(tmp_path)
    first=deepcopy(warp);second=deepcopy(warp)
    a=refine_source_controls(path,first,sr,evidence(path))
    b=refine_source_controls(path,second,sr,evidence(path,stable_clock_controls=False))
    assert first['anchors']==second['anchors']
    assert a==b

def test_supported_stable_clock_keeps_all_observed_attack_references(tmp_path):
    path,warp,sr=fixture(tmp_path)
    ordinary=deepcopy(warp);stable=deepcopy(warp)
    a=refine_source_controls(path,ordinary,sr,evidence(path))['head']
    b=refine_source_controls(path,stable,sr,evidence(path,stable_clock_controls=True))['head']
    assert b['stable_native_primary_clock']
    assert a['events']==b['events']
    assert a['native_rhythm_observations']==b['native_rhythm_observations']
    assert len(b['native_control_observation_residuals'])==len(b['events'])
    audit=stable['actual_timing_rate_audit']
    assert audit['passed'] and audit['short_term_timing_continuity']['passed']
    controlled=[a for a in stable['anchors'] if a['role']=='head_sync_control']
    for anchor in controlled[4:-4]:
        true_native=.3+(anchor['output_seconds']-.3)/.5*.503
        assert abs(anchor['source_seconds']-true_native)<.004

def test_stable_control_request_cannot_invent_an_unsupported_native_clock(tmp_path):
    path,warp,sr=fixture(tmp_path)
    with pytest.raises(SourceWindowFailure,match='established native audio clock'):
        refine_source_controls(path,warp,sr,evidence(path,
            stable_clock_controls=True,allow_variable_attack_family=True))
