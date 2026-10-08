"""Device-relaxed atomic reads: native instruction selection and exact transport."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import numpy as np
import pytest
from brian2_rust import cuda_codegen
from brian2_rust.cuda import build_cuda_plan
from brian2_rust.cuda_compiler import compilation_environment
from brian2_rust.plan import PlanValidationError,verify_execution_plan
from cuda_atomic_load_compare import atomic_reads,LEGACY_ATOMIC_LOAD
from test_cuda import real_cuda,ROOT


def test_atomic_read_source_is_bound_to_plan_and_restored(tmp_path):
    model=json.loads((ROOT/'tests/golden/b2ir-v1/minimal-v1.json').read_text())
    current=build_cuda_plan(model,numeric_mode='float32')
    header=cuda_codegen.CUDA_HEADER
    with atomic_reads(legacy=True):
        old=build_cuda_plan(model,numeric_mode='float32')
        verify_execution_plan(old,model)
    assert cuda_codegen.CUDA_HEADER==header
    assert old.sha256!=current.sha256
    assert old.logical==current.logical and old.dispatches==current.dispatches and old.buffers==current.buffers
    with pytest.raises(PlanValidationError):verify_execution_plan(old,model)
    verify_execution_plan(current,model)
    for before,after in zip(old.kernels,current.kernels,strict=True):
        assert after.source.replace(cuda_codegen.CUDA_ATOMIC_LOAD,LEGACY_ATOMIC_LOAD)==before.source
    (tmp_path/'atomic-plans.json').write_text(json.dumps(dict(legacy=old.to_dict(),load=current.to_dict()))+'\n')


@real_cuda
def test_native_read_is_a_load_with_exact_uint_transport_and_legacy_fallback(tmp_path):
    import cupy as cp
    props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    arch=f"sm_{props['major']}{props['minor']}"
    nvcc=shutil.which('nvcc');assert nvcc is not None
    environment,_=compilation_environment()
    version=subprocess.check_output([nvcc,'--version'],text=True,env=environment)
    match=re.search(r'release (\d+)\.(\d+)',version);assert match
    assert tuple(map(int,match.groups()))>=(12,8) and props['major']>=6
    body=r'''
extern "C" __global__ void read_counts(uint *counts,uint *out,uint n) {
    uint lane=blockIdx.x*blockDim.x+threadIdx.x;
    if(lane<n) out[lane]=atomic_load_explicit(&counts[lane],memory_order_relaxed);
}
'''
    header=cuda_codegen.CUDA_HEADER
    fallback=re.sub(r'^#if .*$', '#if 0',cuda_codegen.CUDA_ATOMIC_LOAD,count=1,flags=re.M)
    variants=dict(legacy=header.replace(cuda_codegen.CUDA_ATOMIC_LOAD,LEGACY_ATOMIC_LOAD),
                  load=header,fallback=header.replace(cuda_codegen.CUDA_ATOMIC_LOAD,fallback))
    values=np.asarray([0,1,3,4,5,2**31-1,2**31,2**32-1]*129,np.uint32)
    outputs={};records={}
    for mode,source in variants.items():
        path=tmp_path/(mode+'.cu');path.write_text(source+body)
        for kind in ('cubin','ptx'):
            command=[nvcc,'--'+kind,f'--gpu-architecture={arch}',*cuda_codegen.OPTIONS,str(path),'-o',str(path.with_suffix('.'+kind))]
            run=subprocess.run(command,env=environment,text=True,capture_output=True)
            assert run.returncode==0,run.stdout+run.stderr
        ptx=path.with_suffix('.ptx').read_text()
        # Inspect executable PTX lines, never source comments or helper names.
        instructions='\n'.join(line for line in ptx.splitlines() if not line.lstrip().startswith('//'))
        has_rmw=bool(re.search(r'\b(?:atom|red)(?:\.[\w]+)+\s',instructions))
        # NVCC 12.8 emits a bitwise b32 load for the uint32 intrinsic.
        has_load=bool(re.search(r'\bld\.relaxed\.gpu(?:\.global)?\.[bu]32\s',instructions))
        assert has_rmw==(mode!='load'),(mode,ptx)
        assert has_load==(mode=='load'),(mode,ptx)
        sass=subprocess.check_output(['cuobjdump','--dump-sass',str(path.with_suffix('.cubin'))],text=True)
        path.with_suffix('.sass').write_text(sass)
        assert bool(re.search(r'\b(?:ATOMG|RED)(?:\.|\s)',sass))==(mode!='load'),(mode,sass)
        module=cp.RawModule(path=str(path.with_suffix('.cubin')))
        counts=cp.asarray(values);out=cp.zeros_like(counts)
        module.get_function('read_counts')(((len(values)+127)//128,),(128,),(counts,out,np.uint32(len(values))))
        cp.cuda.get_current_stream().synchronize()
        outputs[mode]=cp.asnumpy(out);outputs[mode+'_input']=cp.asnumpy(counts)
        np.testing.assert_array_equal(outputs[mode],values)
        np.testing.assert_array_equal(outputs[mode+'_input'],values)
        records[mode]=dict(atomic_rmw=has_rmw,device_relaxed_load=has_load,
            hashes={ext:hashlib.sha256(path.with_suffix('.'+ext).read_bytes()).hexdigest() for ext in ('cu','ptx','cubin','sass')})
    np.savez_compressed(tmp_path/'atomic-transport.npz',reference=values,**outputs)
    (tmp_path/'atomic-instructions.json').write_text(json.dumps(dict(passed=True,architecture=arch,nvcc=version,
        scope='device',order='relaxed',variants=records),indent=2)+'\n')
