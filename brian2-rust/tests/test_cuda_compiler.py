"""Declared CUDA arithmetic survives ambient flags and pre-policy cache entries."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
import brian2 as b
import numpy as np
import pytest
from brian2_rust.cuda import CudaExecutor
from brian2_rust.cuda_codegen import OPTIONS
from brian2_rust.cuda_compiler import compilation_environment,binary_identity,INJECTED_FLAGS,POLICY
from brian2_rust.export import lower_network
from test_metal_delays import device,ROOT
from test_cuda import real_cuda
from test_cuda_graphs import result_exact


def test_compiler_environment_is_a_private_snapshot(monkeypatch):
    for name in INJECTED_FLAGS:monkeypatch.setenv(name,'--use_fast_math --define-macro=PRIVATE_VALUE')
    monkeypatch.setenv('NVCC_CCBIN','/some/toolchain/c++')
    env,report=compilation_environment()
    assert all(name not in env and 'PRIVATE_VALUE' in os.environ[name] for name in INJECTED_FLAGS)
    assert env['NVCC_CCBIN']=='/some/toolchain/c++'
    assert report==dict(policy=POLICY,ignored_environment_variables=list(INJECTED_FLAGS))
    assert 'PRIVATE_VALUE' not in json.dumps(report)
    monkeypatch.setenv('NVCC_CCBIN','/another/compiler')
    assert env['NVCC_CCBIN']=='/some/toolchain/c++'


def test_cache_migration_cannot_reuse_a_legacy_binary():
    source='kernel';arch='sm_89';version='nvcc 12.8'
    old=hashlib.sha256((source+arch+version+repr(OPTIONS)).encode()).hexdigest()
    new=binary_identity(source,arch,version,OPTIONS)
    assert new!=old
    alternatives=[binary_identity(source+'x',arch,version,OPTIONS),binary_identity(source,'sm_80',version,OPTIONS),
                  binary_identity(source,arch,version+'x',OPTIONS),binary_identity(source,arch,version,('--fmad=true',))]
    assert len({new,*alternatives})==5


def witness_model(device,tmp_path,dag):
    b.set_device('rust_standalone',engine='reference',directory=tmp_path/'reference',runner=ROOT/'target/release/b2-runner')
    pop=b.NeuronGroup(4,'a:1 (constant)\nbb:1 (constant)\nc:1 (constant)\ny:1',threshold='y<0',reset='y=y',dtype=np.float32)
    pop.a=1+2**-13;pop.bb=1-2**-13;pop.c=-1
    pop.run_regularly('y=a*bb+c',when='groups')
    objects=[pop,b.SpikeMonitor(pop),b.StateMonitor(pop,'y',record=True)]
    if dag:
        syn=b.Synapses(pop,pop,'w:1 (constant)',on_pre='y_post+=w',delay=b.defaultclock.dt)
        syn.connect(i=[0,1,2,3],j=[1,2,3,0]);syn.w=0;objects.append(syn)
    return lower_network(b.Network(*objects),4*b.defaultclock.dt)


@real_cuda
@pytest.mark.parametrize('dag',[False,True])
def test_real_cuda_fma_witness_ignores_injection_and_old_cache(device,tmp_path,monkeypatch,dag):
    model=witness_model(device,tmp_path,dag);directory=tmp_path/'cuda'
    for name in INJECTED_FLAGS:monkeypatch.delenv(name,raising=False)
    with CudaExecutor(model,directory,numeric_mode='float32',event_delivery='sparse') as ex:
        expected=ex.run();plan=ex.plan
        # Two f32 operations round to zero. A contracted FMA yields -2^-26
        # and changes threshold/spike outcomes. This is an arithmetic witness,
        # independent of the compiler's reported options.
        np.testing.assert_array_equal(expected['populations'][0]['states']['y'],0)
        assert expected['populations'][0]['spike_ticks'].size==0
        result_exact(expected,ex.run(compute='cpu-f32'))
        version=ex.nvcc_version;arch=ex.architecture
    # Old names can contain poisoned or incompatible cubins. Force fresh new
    # compilation while retaining an invalid legacy entry for every kernel.
    for path in directory.glob('*.cubin'):path.unlink()
    for kernel in plan.kernels:
        old=hashlib.sha256((kernel.source+arch+version+repr(plan.compiler_options)).encode()).hexdigest()
        (directory/(kernel.entry+'-'+old+'.cubin')).write_bytes(b'legacy cache must never be loaded')
    monkeypatch.setenv('NVCC_PREPEND_FLAGS','--this-option-must-not-reach-nvcc')
    monkeypatch.setenv('NVCC_APPEND_FLAGS','--fmad=true --ftz=true --use_fast_math')
    original=subprocess.run;calls=[]
    def inspect(command,*args,**kwargs):
        if Path(command[0]).name=='nvcc':
            assert not set(INJECTED_FLAGS)&set(kwargs['env'])
            if '--cubin' in command:calls.append(command)
        return original(command,*args,**kwargs)
    monkeypatch.setattr(subprocess,'run',inspect)
    with CudaExecutor(model,directory,numeric_mode='float32',event_delivery='sparse') as ex:
        actual=ex.run();result_exact(actual,expected)
        assert ex.plan.sha256==plan.sha256
        assert actual['cuda_runtime']['compiler_environment']==dict(policy=POLICY,ignored_environment_variables=list(INJECTED_FLAGS))
        # Returned diagnostics cannot mutate subsequent runtime reports.
        actual['cuda_runtime']['compiler_environment']['ignored_environment_variables'].clear()
        assert len(ex.run()['cuda_runtime']['compiler_environment']['ignored_environment_variables'])==2
    assert len(calls)==len(plan.kernels)
    calls.clear();monkeypatch.setenv('NVCC_APPEND_FLAGS','--invalid-cache-hit-injection')
    with CudaExecutor(model,directory,numeric_mode='float32',event_delivery='sparse') as ex:result_exact(ex.run(),expected)
    assert not calls
    assert os.environ['NVCC_PREPEND_FLAGS']=='--this-option-must-not-reach-nvcc'
    assert os.environ['NVCC_APPEND_FLAGS']=='--invalid-cache-hit-injection'
    (tmp_path/'compiler-isolation.json').write_text(json.dumps(dict(plan_sha256=plan.sha256,dag=dag,kernels=len(plan.kernels),
        policy=POLICY,expected_y=0,contracted_y=-2**-26,spikes=0,fresh_compile_and_cache_hit_exact=True,legacy_cache_ignored=True),indent=2)+'\n')
