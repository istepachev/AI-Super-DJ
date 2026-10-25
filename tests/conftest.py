"""Test policy states explicitly without editing workstation settings.

OS affinity/priority imposed by an external limiter are never changed here.
"""
import pytest


@pytest.fixture(autouse=True)
def isolated_resource_profile(monkeypatch, tmp_path):
    from superdj import resource_config, resource_limits
    path = tmp_path/'test-runtime.local.json'
    monkeypatch.setattr(resource_limits, 'runtime_path', lambda: path)
    monkeypatch.setattr(resource_config, 'runtime_path', lambda: path)
    monkeypatch.delenv(resource_limits.ENVIRONMENT_KEY, raising=False)
