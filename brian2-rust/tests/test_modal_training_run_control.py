"""Local-only safety checks; never contact Modal or allocate a GPU."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

spec=importlib.util.spec_from_file_location('training_cloud_entry',Path(__file__).resolve().parents[1]/'examples/modal_native_training_v4.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class Guard:
    def __init__(self):self.values={}
    def put(self,key,value,*,skip_if_exists=False):
        assert skip_if_exists
        if key in self.values:return False
        self.values[key]=value;return True


def test_replacement_container_cannot_start_tests(monkeypatch):
    guard=Guard();entry.claim_attempt(guard)
    def forbidden(*args,**kwargs):pytest.fail('replayed attempt reached test execution')
    monkeypatch.setattr(entry,'execute_tests',forbidden)
    # The replay also fails before changing to the cloud-only /workspace path.
    with pytest.raises(RuntimeError,match='refusing platform replay'):
        entry.run_tests(['test_training_refractory.py'],guard)


def test_missing_guard_fails_closed():
    with pytest.raises(ValueError,match='attempt guard'):
        entry.run_tests([],None)


def test_live_logs_preserved_for_failed_test_process(capsys):
    result=entry.execute_tests([sys.executable,'-u','-c',
        "import sys;print('case-one PASSED');print('case-two FAILED',file=sys.stderr);sys.exit(3)"],os.environ.copy(),timeout=5)
    assert not result['passed'] and result['exit_code']==3 and not result['timeout']
    assert 'case-one PASSED' in result['stdout'] and 'case-two FAILED' in result['stdout']
    assert result['stdout']==capsys.readouterr().out


def test_timeout_terminates_test_process():
    result=entry.execute_tests([sys.executable,'-u','-c','import time;time.sleep(60)'],os.environ.copy(),timeout=.1)
    assert result['timeout'] and not result['passed'] and result['exit_code']<0
