from pathlib import Path
import hashlib,json
import pytest
from superdj.reporting import archive_existing_report,write_performance


def test_previous_round_survives_byte_exactly_before_next_report(tmp_path):
    path=tmp_path/'performance.json';old=b'{"analysis": 34, "seconds": 931.125}\n';path.write_bytes(old)
    write_performance(path,{'analysis':12,'seconds':139.6})
    saved=tmp_path/'_performance_history/performance'/(hashlib.sha256(old).hexdigest()+'.json')
    assert saved.read_bytes()==old and json.loads(path.read_text())['analysis']==12


def test_unchanged_prior_report_is_archived_once(tmp_path):
    path=tmp_path/'performance.json';path.write_text('{}')
    a=archive_existing_report(path);b=archive_existing_report(path)
    assert a==b and len(list(a.parent.iterdir()))==1


def test_missing_report_creates_no_history_folder(tmp_path):
    assert archive_existing_report(tmp_path/'performance.json')is None
    assert not(tmp_path/'_performance_history').exists()


def test_edited_archive_is_kept_and_prior_report_is_saved_separately(tmp_path):
    path=tmp_path/'performance.json';path.write_bytes(b'old report')
    first=archive_existing_report(path);first.write_bytes(b'edited archive')
    second=archive_existing_report(path)
    assert first.read_bytes()==b'edited archive'and second.read_bytes()==b'old report'and first!=second


def test_unparseable_old_report_is_preserved_without_losing_new_metrics(tmp_path):
    path=tmp_path/'performance.json';old=b'{broken\xff';path.write_bytes(old)
    write_performance(path,{'status':'failed'})
    assert json.loads(path.read_text())=={'status':'failed'}
    assert next((tmp_path/'_performance_history/performance').iterdir()).read_bytes()==old


def test_archive_failure_does_not_overwrite_the_previous_report(tmp_path,monkeypatch):
    path=tmp_path/'performance.json';path.write_bytes(b'old report')
    def refuse(*args):raise PermissionError('archive busy')
    monkeypatch.setattr(Path,'replace',refuse)
    with pytest.raises(PermissionError):write_performance(path,{'status':'complete'})
    assert path.read_bytes()==b'old report'
