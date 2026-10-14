"""A transfer ablation must keep the other direction's policy fixed."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from brian2_rust import gpu_readback
from gpu_spike_readback_compare import readback_mode
from gpu_spike_upload_compare import upload_mode
from gpu_host_storage_compare import host_copy_mode


@pytest.mark.parametrize('mode',['full','prefix'])
def test_readback_ablation_holds_upload_constant_and_restores_after_failure(mode):
    prefix,upload=gpu_readback.spike_prefix_bindings,gpu_readback.spike_upload_omissions
    arrays=[np.zeros(4,np.int64),np.zeros(1,np.uint32)]
    from types import SimpleNamespace
    plan=SimpleNamespace(buffers=('population/0/4','population/0/5'))
    assert upload(arrays,{0:1})=={0}
    with pytest.raises(RuntimeError,match='injected'):
        with readback_mode(mode):
            assert not gpu_readback.spike_upload_omissions(arrays,{0:1})
            assert gpu_readback.spike_host_copy_omissions(plan,arrays,{0,1},'metal')=={0}
            raise RuntimeError('injected')
    assert gpu_readback.spike_prefix_bindings is prefix
    assert gpu_readback.spike_upload_omissions is upload


@pytest.mark.parametrize('mode',['full','omit'])
def test_upload_ablation_leaves_readback_policy_unchanged(mode):
    prefix,upload=gpu_readback.spike_prefix_bindings,gpu_readback.spike_upload_omissions
    with pytest.raises(RuntimeError,match='injected'):
        with upload_mode(mode):
            assert gpu_readback.spike_prefix_bindings is prefix
            raise RuntimeError('injected')
    assert gpu_readback.spike_prefix_bindings is prefix
    assert gpu_readback.spike_upload_omissions is upload


@pytest.mark.parametrize('wrapper',['modal_spike_upload_compare.py','modal_host_storage_compare.py'])
def test_cloud_child_timeout_preserves_partial_diagnostics(monkeypatch,wrapper):
    import ast
    import subprocess
    source=Path(__file__).resolve().parents[1]/'examples'/wrapper
    tree=ast.parse(source.read_text())
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
    helper=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='bounded')
    def timed_out(command,**kwargs):
        raise subprocess.TimeoutExpired(command,kwargs['timeout'],output=b'one test passed\n',stderr=b'partial diagnostic\n')
    monkeypatch.setattr(subprocess,'run',timed_out)
    namespace=dict(subprocess=subprocess,env={})
    exec(compile(ast.Module(body=[helper],type_ignores=[]),str(source),'exec'),namespace)
    result=namespace['bounded'](['pytest'],900)
    assert result.returncode==124 and result.stdout=='one test passed\n'
    assert result.stderr.startswith('partial diagnostic\n') and '900 seconds' in result.stderr


@pytest.mark.parametrize('mode',['full','omit'])
def test_host_ablation_preserves_gpu_transfer_policies_and_restores(mode):
    from types import SimpleNamespace
    prefix,upload,host=gpu_readback.spike_prefix_bindings,gpu_readback.spike_upload_omissions,gpu_readback.spike_host_copy_omissions
    plan=SimpleNamespace(buffers=('population/0/4','population/0/5'))
    arrays=[np.zeros(4,np.int64),np.zeros(1,np.uint32)]
    with pytest.raises(RuntimeError,match='injected'):
        with host_copy_mode(mode) as calls:
            assert gpu_readback.spike_prefix_bindings is prefix and gpu_readback.spike_upload_omissions is upload
            assert gpu_readback.spike_host_copy_omissions(plan,arrays,{0,1},'metal')==({0} if mode=='omit' else set())
            assert len(calls)==1
            raise RuntimeError('injected')
    assert gpu_readback.spike_host_copy_omissions is host
