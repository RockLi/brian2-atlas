import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
spec=importlib.util.spec_from_file_location('resource_guard',Path(__file__).parents[1]/'tools/mpi_resource_guard.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

class Process:
    args=['simulation']
    def __init__(self):self.checks=0;self.done=False
    def poll(self):return 0 if self.done else None
    def wait(self,timeout):
        assert 0<timeout<=5
        self.checks+=1
        if self.checks==1:raise guard.subprocess.TimeoutExpired(self.args,timeout)
        self.done=True


def test_disk_reserve_is_rechecked_while_child_runs(monkeypatch):
    values=iter([1000,99]);p=Process()
    monkeypatch.setattr(guard.os,'statvfs',lambda _:SimpleNamespace(f_bavail=next(values),f_frsize=1))
    with pytest.raises(RuntimeError,match='reserve crossed'):
        guard.wait_with_disk_reserve(p,Path('/data'),100,60)
    assert p.checks==1 and not p.done


def test_bounded_poll_preserves_normal_exit_and_lowest_free_space(monkeypatch):
    values=iter([1000,800]);p=Process()
    monkeypatch.setattr(guard.os,'statvfs',lambda _:SimpleNamespace(f_bavail=next(values),f_frsize=1))
    assert guard.wait_with_disk_reserve(p,Path('/data'),100,60)==800
    assert p.done


def test_deadline_ends_wait_even_with_free_disk(monkeypatch):
    monkeypatch.setattr(guard.os,'statvfs',lambda _:SimpleNamespace(f_bavail=1000,f_frsize=1))
    with pytest.raises(guard.subprocess.TimeoutExpired):
        guard.wait_with_disk_reserve(Process(),Path('/data'),100,0)
