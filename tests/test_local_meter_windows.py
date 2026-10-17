from copy import deepcopy
import pytest

from superdj.compact import arrange_sync
from superdj.planner import plan_tracks
from superdj.render import _grid, _local_validation_possible, RenderError
from test_new_workflow import record


def locally_supported():
    row=record('local',360)
    row.update(grid_status='model_predicted',grid_confidence=.6891)
    row['grid']={'meter':None,'meter_consistency':.7059,'human_reviewed':False,
        'metrical_decode':{'meter':4,'accepted':True,'phase_contrast':.95},
        'neural_downbeat_probability_by_beat':[.99 if n%4==0 else .01 for n in range(len(row['beats']))],
        'neural_beat_probability_by_beat':[.99]*len(row['beats'])}
    return row


def test_accepted_local_meter_does_not_certify_the_rejected_whole_grid():
    row=locally_supported();before=deepcopy(row);grid=_grid(row,.75)
    assert not grid[0] and grid[1]=='downbeat meter consistency is below 0.80'
    assert grid[4]==4 and _local_validation_possible(row,grid)
    assert row==before and row['grid']['meter'] is None
    rows=[record('before'),row,record('after')]
    plan=arrange_sync(plan_tracks(rows,mode='fixed'),rows,zhilin=True)
    for role in ('head','tail'):
        evidence=plan['sections']['local'][role+'_local_evidence']
        assert evidence['strong_local_neural_evidence']
        assert evidence['local_phase_contrast']>=.25 and evidence['local_timing_confidence']>=.75
        assert evidence['bar_beat_count_consistency']==1 and not evidence['human_reviewed']


@pytest.mark.parametrize('change', ['rejected_decode','declared_other_meter','weak_contrast','missing_beat_probabilities'])
def test_unsubstantiated_meter_still_stops_source_selection(change):
    row=locally_supported()
    if change=='rejected_decode':row['grid']['metrical_decode']['accepted']=False
    elif change=='declared_other_meter':row['grid']['meter']=3
    elif change=='weak_contrast':row['grid']['metrical_decode']['phase_contrast']=.1
    else:row['grid'].pop('neural_beat_probability_by_beat')
    grid=_grid(row,.75)
    assert not grid[0] and not _local_validation_possible(row,grid)
    rows=[record('before'),row]
    with pytest.raises(RenderError,match='usable grid'):
        arrange_sync(plan_tracks(rows,mode='fixed'),rows,zhilin=True)


def test_local_windows_still_require_coherent_existing_downbeats():
    row=locally_supported();row['downbeats']=row['beats'][::3]
    rows=[record('before'),row]
    with pytest.raises(RenderError,match='supported common SYNC window'):
        arrange_sync(plan_tracks(rows,mode='fixed'),rows,zhilin=True)
