import json
from pathlib import Path
import tempfile
import pytest
from superdj.storage import write_json


def test_atomic_json_does_not_append_destination_name_to_temporary_path(tmp_path,monkeypatch):
    original=tempfile.mkstemp;seen=[]
    def short(*args,**kwargs):
        fd,name=original(*args,**kwargs);seen.append(Path(name));assert len(Path(name).name)<=len('job.json');return fd,name
    monkeypatch.setattr(tempfile,'mkstemp',short);target=tmp_path/'job.json';write_json(target,{'source':'preserved','value':1})
    assert json.loads(target.read_text(encoding='utf8'))['value']==1
    assert all(not p.exists()for p in seen)


def test_atomic_failure_preserves_previous_json_and_removes_only_own_temp(tmp_path,monkeypatch):
    target=tmp_path/'job.json';target.write_text('{"old":1}',encoding='utf8');previous=target.read_bytes();replace=Path.replace
    def refused(self,destination):
        if Path(destination)==target:raise OSError('simulated failed atomic replace')
        return replace(self,destination)
    monkeypatch.setattr(Path,'replace',refused)
    with pytest.raises(OSError):write_json(target,{'new':2})
    assert target.read_bytes()==previous and list(tmp_path.iterdir())==[target]
