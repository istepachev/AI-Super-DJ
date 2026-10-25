from copy import deepcopy
import pytest
from superdj.stretch_backend import selected_backend,backend_arguments


@pytest.mark.parametrize('default,args',[('r2',['--fast','--detector-perc']),('r3',['--fine'])])
def test_existing_default_commands_remain_unchanged(default,args):
    row={'id':'source'};before=deepcopy(row)
    assert backend_arguments(selected_backend(row,default))==args
    assert row==before


def test_explicit_one_source_override_does_not_change_the_other_source():
    assert selected_backend({'rubberband_engine_override':'r3'},'r2')=='r3'
    assert selected_backend({},'r2')=='r2'
    with pytest.raises(ValueError):selected_backend({'rubberband_engine_override':'guess'},'r2')
