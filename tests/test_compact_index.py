from copy import deepcopy
import pytest
from superdj import compact
from superdj.execution import execution
from superdj.render_index import enable
from superdj.planner import plan_tracks
from test_new_workflow import record


@pytest.mark.parametrize('persistent',[False,True])
def test_indexed_windows_preserve_plan_and_do_not_repeat_native_lattice(tmp_path,monkeypatch,persistent):
    rows=[record('a',420),record('b',450),record('c',420)]
    plan=plan_tracks(rows,mode='fixed')
    baseline=compact.arrange_sync(plan,rows)
    ranges={i:{b:[s[b],s[b]]for b in('start','end')}for i,s in baseline['sections'].items()}
    expected=compact.arrange_sync(plan,rows,section_ranges=ranges)
    calls=[];original=compact._local_window_evidence
    def counted(*args,**options):calls.append(1);return original(*args,**options)
    monkeypatch.setattr(compact,'_local_window_evidence',counted)
    with execution(tmp_path/'cache'if persistent else None)as session:
        enable(session,'windows',track_count=3)
        first=compact.arrange_sync(plan,rows,section_ranges=ranges)
        count=len(calls);assert count>0
        second=compact.arrange_sync(plan,rows,section_ranges=ranges)
        assert len(calls)==count
        assert first==second==expected
        changed=deepcopy(rows);changed[1]['beat_confidence']=.1
        # Actual record changes must require a fresh lattice/evidence decision.
        compact.arrange_sync(plan,changed,section_ranges=ranges)
        assert len(calls)>count


def test_early_choice_filter_keeps_one_sample_tolerance():
    rows=[record('a',420),record('b',420)]
    plan=plan_tracks(rows,mode='fixed');expected=compact.arrange_sync(plan,rows)
    limits={i:{b+'_choices':[s[b]+.5/48000]for b in('start','end')}for i,s in expected['sections'].items()}
    assert compact.arrange_sync(plan,rows,section_ranges=limits)==expected
