"""Compilation reuse preserves validation, inputs, ownership and toolchain scope."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import brian2 as b
import numpy as np
import pytest

from brian2_rust.cuda import CudaExecutor
from brian2_rust.metal import MetalExecutor
from brian2_rust.gpu_compilation import cached_binary,save_binary,dependency_files,compiler_context,request,sha
from brian2_rust.protocol import attach_protocol
from test_metal_delays import device,ROOT
from test_gpu_spike_generator import BACKENDS
from test_gpu_summed_parallel import model_at,control
from test_cuda_graphs import result_exact
from test_gpu_multiclock import network,DT


def test_dependency_parser_hashes_real_inputs_and_preserves_spaces(tmp_path):
    source=tmp_path/'stage.cu';source.write_text('source')
    header=tmp_path/'some header.h';header.write_text('first')
    text='b2: '+str(source)+' \\\n '+str(header).replace(' ','\\ ')+'\n'
    assert dependency_files(text,source)==[header]
    with pytest.raises(RuntimeError):dependency_files('different: '+str(header),source)
    with pytest.raises(RuntimeError):dependency_files('b2: missing-header.h',source)


def test_cache_bounds_corruption_and_context_changes(tmp_path,monkeypatch):
    import brian2_rust.gpu_compilation as c
    monkeypatch.setattr(c,'MAX_CUBIN_BYTES',6)
    ex=SimpleNamespace(_compiled_binaries={});save_binary(ex,'a',b'abc');save_binary(ex,'b',b'defg')
    assert cached_binary(ex,'a')==b'abc' and cached_binary(ex,'b') is None
    ex._compiled_binaries['a']=(sha(b'abc'),b'bad');assert cached_binary(ex,'a') is None
    executable=tmp_path/'compiler';executable.write_bytes(b'first')
    env={'PRIVATE_ENV':'not-published'};one=compiler_context(str(executable),env,[])
    assert 'not-published' not in one
    assert one!=compiler_context(str(executable),env,['new-options'])
    assert one!=compiler_context(str(executable),{'PRIVATE_ENV':'changed'},[])
    executable.write_bytes(b'second');assert one!=compiler_context(str(executable),env,[])
    assert not request({'definition':{'functions':[{'body':None}]}},True,None,SimpleNamespace)


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_exact_sources_reuse_without_reusing_model_or_storage(device,tmp_path,backend):
    first=model_at(tmp_path/'ref',edges=273,steps=4);second=deepcopy(first)
    # Use a different validated instance while preserving emitted source.
    from brian2_rust.spec import bits
    second['instance']['populations'][0]['initial_state']['v']=[bits(.125)]*len(first['instance']['populations'][0]['initial_state']['v'])
    attach_protocol(second);expected=control(second,tmp_path/'control')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(first,tmp_path/'old',numeric_mode='float32',event_delivery='sparse',compile_reuse=True) as old:
        old.run()
        with cls(second,tmp_path/'new',numeric_mode='float32',event_delivery='sparse',compile_reuse=True,reuse_from=old) as new:
            assert new.compilation_report['kernels_reused']==len(new.plan.kernels)
            assert new.compilation_report['kernels_compiled']==0
            if backend=='metal':
                assert new.compilation_report['bridge_reused'] and set(new.handles).isdisjoint(old.handles)
                assert new._resident_dag_bytes==0
            else:assert new._resident_dag is None
            old.close();result_exact(new.run(),expected)
            (tmp_path/'compilation-result.json').write_text(json.dumps(new.compilation_report,indent=2)+'\n')


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_device_compile_only_continuation_restore_and_close(device,tmp_path,backend):
    records=[];reports=[];saved={}
    for reuse in (False,True):
        device.reinit()
        b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
            gpu_compile_reuse=reuse,directory=tmp_path/str(reuse),runner=ROOT/'target/release/b2-runner')
        net,pre,post,syn,monitors,spikes=network(True);states=[];compilation=[]
        for i in range(3):
            if i==1:net.store('saved')
            if i==2:net.restore('saved')
            net.run(6*DT)
            states.append(dict(v=np.asarray(pre.v[:]).copy(),post=np.asarray(post.v[:]).copy(),w=np.asarray(syn.w[:]).copy(),
                traces=[np.asarray(m.v[:]).copy() for m in monitors],spikes=[np.asarray(s.t[:]).copy() for s in spikes]))
            r=json.loads((device.last_run_directory/'rust/summary.json').read_text())[backend+'_runtime']
            compilation.append(r['compilation']);assert not r['activation_buffer_reuse']['adopted']
            if reuse:
                ex=device._gpu_executor
                assert ex._resident_dag_bytes==0 if backend=='metal' else ex._resident_dag is None
        for activation,state in enumerate(states):
            for name,values in state.items():
                for j,value in enumerate(values if isinstance(values,list) else [values]):
                    saved[f'{reuse}/{activation}/{name}/{j}']=value
        records.append(states);reports.append(compilation);device.close_gpu();assert device._gpu_executor is None
    for a,e in zip(*records):
        for name in ('v','post','w'):np.testing.assert_array_equal(a[name],e[name])
        for name in ('traces','spikes'):
            for x,y in zip(a[name],e[name]):np.testing.assert_array_equal(x,y)
    assert reports[1][-1]['kernels_reused']>0
    assert not any(r['kernels_reused'] for r in reports[0])
    np.savez_compressed(tmp_path/'compilation-results.npz',**saved)
    (tmp_path/'compilation-lifecycle.json').write_text(json.dumps(reports,indent=2)+'\n')


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_context_change_and_closed_predecessor_invalidate(device,tmp_path,backend,monkeypatch):
    model=model_at(tmp_path/'ref',edges=273,steps=4)
    expected=control(model,tmp_path/'control')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    with cls(model,tmp_path/'old',numeric_mode='float32',compile_reuse=True) as old:
        monkeypatch.setenv('B2_COMPILATION_CONTEXT_TEST','changed')
        with cls(model,tmp_path/'new',numeric_mode='float32',compile_reuse=True,reuse_from=old) as new:
            assert new.compilation_report['kernels_reused']==0
            assert new.compilation_report['kernels_compiled']==len(new.plan.kernels)
            result_exact(new.run(),expected)
            (tmp_path/'compilation-invalidation.json').write_text(json.dumps(new.compilation_report,indent=2)+'\n')
    with pytest.raises(RuntimeError,match='predecessor is closed'):
        cls(model,tmp_path/'closed',numeric_mode='float32',compile_reuse=True,reuse_from=old)
