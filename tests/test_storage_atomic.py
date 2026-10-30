import json
from pathlib import Path
import pytest
from superdj.storage import write_json


def test_transient_replace_refusal_preserves_old_json_until_commit(tmp_path,monkeypatch):
    target=tmp_path/'state.json';target.write_text('{"old":1}')
    original=Path.replace;calls=[]
    monkeypatch.setattr('superdj.storage.time.sleep',lambda value:None)
    def replace(source,destination):
        calls.append(1)
        if len(calls)<3:
            assert json.loads(target.read_text())=={'old':1}
            raise PermissionError('reader still holds the previous snapshot')
        return original(source,destination)
    monkeypatch.setattr(Path,'replace',replace)
    write_json(target,{'new':'данные'})
    assert len(calls)==3 and json.loads(target.read_text())=={'new':'данные'}
    assert set(tmp_path.iterdir())=={target}


def test_persistent_replace_refusal_raises_and_retains_old_file(tmp_path,monkeypatch):
    target=tmp_path/'state.json';target.write_text('{"old":1}');calls=[]
    monkeypatch.setattr('superdj.storage.time.sleep',lambda value:None)
    def refuse(*args):calls.append(1);raise PermissionError('still locked')
    monkeypatch.setattr(Path,'replace',refuse)
    with pytest.raises(PermissionError,match='still locked'):write_json(target,{'new':2})
    assert len(calls)==5 and json.loads(target.read_text())=={'old':1}
    assert set(tmp_path.iterdir())=={target}


def test_invalid_json_is_never_retried_or_partially_published(tmp_path,monkeypatch):
    target=tmp_path/'state.json';target.write_text('{"old":1}')
    monkeypatch.setattr(Path,'replace',lambda *args:pytest.fail('Invalid JSON must never reach publication'))
    with pytest.raises(ValueError):write_json(target,{'invalid':float('nan')})
    assert json.loads(target.read_text())=={'old':1}and set(tmp_path.iterdir())=={target}
