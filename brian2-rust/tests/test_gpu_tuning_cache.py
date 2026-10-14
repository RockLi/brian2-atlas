"""Exact-input cache identity, failure recovery, publication and native lifecycles."""
from copy import deepcopy
import importlib
import json

import numpy as np
import pytest

from brian2_rust import gpu_tuning_cache as caching
from brian2_rust import gpu_autotune as tuning
from test_gpu_autotune import Plan, fake_factory, result
from test_metal_delays import device
from test_gpu_spike_generator import BACKENDS


def factories_for(monkeypatch, **kwargs):
    attempts=[]
    make,_=fake_factory(monkeypatch, **kwargs)
    def factories(directory):
        created={};attempts.append(created)
        def factory(*args):
            ex=make(*args);created[args[0]]=ex
            ex.compilation_report={'kernels_reused':0}
            return ex
        return factory, lambda name,*args:Plan(name)
    return factories, attempts


def resolve(factories, path, cache, key='model', **kwargs):
    return caching.resolve(factories,path,cache,key,context_for=lambda ex:'context',**kwargs)


def test_hit_runs_one_fresh_executor_without_reusing_results_or_public_report(monkeypatch,tmp_path):
    factories, attempts=factories_for(monkeypatch)
    cache=caching.TuningCache()
    ex, actual, report, record=resolve(factories,tmp_path/'first',cache)
    assert len(cache)==0 and len(report['calibration']['samples'])==17
    cache.publish('model',record);ex.close()
    actual['synapses'][0]['states']['w'][0]=999
    report['calibration']['reference_observable_sha256']='mutated'
    record['calibration']['selected']='mutated'
    winner, fresh, hit, pending=resolve(factories,tmp_path/'second',cache)
    assert list(attempts[1])==['prefix-bitset'] and winner.calls==1
    assert winner is not ex and hit['cache']['status']=='hit'
    assert tuning.observable_fingerprint(fresh)==tuning.observable_fingerprint(result())
    assert hit['calibration']['selected']=='prefix-bitset'
    assert hit['total_seconds']<report['total_seconds']
    assert json.loads((tmp_path/'second/report.json').read_text())==hit
    winner.close()


@pytest.mark.parametrize('change',['plan','executor-plan','context','result','run','construction'])
def test_invalidated_hit_closes_then_calibrates_full_input(monkeypatch,tmp_path,change):
    factories,attempts=factories_for(monkeypatch)
    cache=caching.TuningCache()
    ex,_,_,record=resolve(factories,tmp_path/'first',cache);ex.close();cache.publish('model',record)
    def altered(directory):
        make,plan=factories(directory)
        if directory.name!='cached':return make,plan
        def build(*args):
            if change=='construction':raise RuntimeError('construction failed')
            ex=make(*args)
            if change=='executor-plan':ex.plan=Plan('changed')
            run=ex.run
            def execute():
                value=run()
                if change=='run':raise RuntimeError('replay failed')
                if change=='result':value['synapses'][0]['events']+=1
                return value
            ex.run=execute
            return ex
        return build, (lambda *args:Plan('changed')) if change=='plan' else plan
    context=(lambda ex:'new-context') if change=='context' else (lambda ex:'context')
    winner,_,report,pending=caching.resolve(altered,tmp_path/'second',cache,'model',context_for=context)
    assert report['cache']['status']=='invalidated' and len(cache)==0
    assert len(report['calibration']['samples'])==17
    assert all(ex.closed for ex in attempts[1].values())
    assert list(attempts[-1])==[p[0] for p in tuning.POLICIES]
    winner.close()


@pytest.mark.parametrize('failure',['baseline','interrupt','save','cleanup'])
def test_cache_failures_never_publish_or_leak(monkeypatch,tmp_path,failure):
    factories,attempts=factories_for(monkeypatch)
    cache=caching.TuningCache()
    ex,_,_,record=resolve(factories,tmp_path/'first',cache);ex.close();cache.publish('model',record)
    def altered(directory):
        make,plan=factories(directory)
        def build(*args):
            ex=make(*args)
            run=ex.run
            def execute():
                if failure=='interrupt':raise KeyboardInterrupt()
                if failure in {'baseline','cleanup'}:raise RuntimeError('injected failure')
                return run()
            ex.run=execute
            if failure=='cleanup':
                close=ex.close
                def failed_close():close();raise RuntimeError('cleanup failed')
                ex.close=failed_close
            return ex
        return build,plan
    if failure=='save':
        from pathlib import Path
        original=Path.write_text
        def failed_save(path,*args,**kwargs):
            if path==tmp_path/'second/report.json':raise OSError('publication failed')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path,'write_text',failed_save)
    with pytest.raises((RuntimeError,KeyboardInterrupt,OSError)):
        resolve(altered,tmp_path/'second',cache)
    assert all(ex.closed for a in attempts for ex in a.values())
    assert len(cache)==(0 if failure in {'baseline','cleanup'} else 1)


def test_cache_is_bounded_and_only_publication_updates_recency():
    cache=caching.TuningCache()
    for i in range(8):cache.publish(str(i),{'selection':i})
    cache.get('0');cache.publish('8',{'selection':8})
    assert len(cache)==8 and cache.get('0') is None
    cache.publish('1',cache.get('1'));cache.publish('9',{'selection':9})
    assert cache.get('1') is not None and cache.get('2') is None
    cache.publish(None,{'ignored':True});assert len(cache)==8
    cache.clear();assert len(cache)==0


def test_key_covers_complete_model_sources_binary_and_execution_options(tmp_path):
    root=tmp_path/'source';root.mkdir();source=root/'runtime.py';source.write_text('one')
    runner=tmp_path/'runner';runner.write_bytes(b'runner-one')
    model={'definition':{'functions':[]},'instance':{'states':[1], 'pending':[2], 'seed':3},
           'run':{'start':0,'duration':6}}
    options={'backend':'metal','dag':'auto'}
    def key(m=model,o=options):return caching.input_key(m,runner,o,source_root=root)
    first=key();assert key(deepcopy(model))==first
    for field in ('states','pending','seed'):
        changed=deepcopy(model);changed['instance'][field]=[999]
        assert key(changed)!=first
    for field in ('start','duration'):
        changed=deepcopy(model);changed['run'][field]+=1
        assert key(changed)!=first
    assert key(o={'backend':'cuda','dag':'auto'})!=first
    assert key(o={'backend':'metal','dag':'direct'})!=first
    source.write_text('two');assert key()!=first
    source.write_text('one');runner.write_bytes(b'runner-two');assert key()!=first
    changed=deepcopy(model);changed['definition']['functions']=[{'body':None}]
    assert key(changed) is None


def test_strict_option_and_lifecycle_validation(device):
    import brian2 as b
    for value in (None,0,1,'yes',np.bool_(True)):
        with pytest.raises(NotImplementedError,match='gpu_autotune_cache must be boolean'):
            b.set_device('rust_standalone',engine='metal',numeric_mode='float32',gpu_autotune_cache=value)
    with pytest.raises(NotImplementedError,match='requires gpu_autotune=True'):
        b.set_device('rust_standalone',engine='metal',numeric_mode='float32',gpu_autotune_cache=True)
    with pytest.raises(NotImplementedError,match='GPU engine'):
        b.set_device('rust_standalone',engine='reference',gpu_autotune_cache=False)
    device._gpu_tuning_cache.publish('test',{})
    device.close_gpu();assert len(device._gpu_tuning_cache)==1
    device.clear_gpu_tuning_cache();assert len(device._gpu_tuning_cache)==0
    device._gpu_tuning_cache.publish('test',{})
    b.set_device('rust_standalone',engine='metal',numeric_mode='float32',gpu_autotune=True,gpu_autotune_cache=True)
    assert len(device._gpu_tuning_cache)==0
    device._gpu_tuning_cache.publish('test',{});device.reinit()
    assert len(device._gpu_tuning_cache)==0


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('reuse',[False,True])
def test_native_restored_pending_state_hits_changed_weights_miss(device,tmp_path,backend,reuse):
    from test_gpu_composed_models import setup,snapshot,DT
    from test_gpu_composed_policies import mixed,full_compare
    from test_gpu_synapse_parallel import control
    from test_gpu_expression_contract import oracle
    from test_gpu_workgroup import save
    from brian2_rust.results import load_results
    setup(tmp_path/'run',backend,gpu_autotune=True,gpu_autotune_cache=True,
          gpu_buffer_reuse=reuse,gpu_compile_reuse=reuse)
    net,pre,post,syn,monitors,spikes,events=mixed(42,traced=True)
    records=[]
    def capture(label,status):
        report=deepcopy(device.last_gpu_tuning);assert report['cache']['status']==status
        path=device.last_run_directory
        assert report==json.loads((path/'gpu-autotune/report.json').read_text())
        assert report==json.loads((path/'rust/summary.json').read_text())[backend+'_runtime']['autotune']
        model=json.loads((path/'model.json').read_text())
        actual=load_results(model,path/'rust');expected=control(model,path/'control',3)
        if status=='hit':
            runtime=actual['metadata'][backend+'_runtime']
            dag=runtime if backend=='metal' else runtime['dag_execution']
            assert runtime['activation_buffer_reuse']['adopted']==dag.get('buffers_adopted',False)
        full_compare(actual,expected)
        full_compare(actual,oracle(model,path/'oracle'),trace_tolerance=True)
        save(tmp_path/(label+'.npz'),actual,expected)
        assert (device._gpu_executor is not None)==reuse
        records.append(dict(label=label,report=report,model=model))
    net.run(6*DT);capture('initial','miss');net.store('pending')
    net.run(6*DT);capture('continued','miss')
    expected=snapshot(pre,post,syn,monitors,spikes,events)
    net.restore('pending',restore_random_state=True);net.run(6*DT);capture('restored','hit')
    for k,v in snapshot(pre,post,syn,monitors,spikes,events).items():
        np.testing.assert_array_equal(v,expected[k],err_msg=k)
    net.restore('pending',restore_random_state=True);syn.w=np.asarray(syn.w[:])+1/4096
    net.run(6*DT);capture('weights','miss')
    assert len(device._gpu_tuning_cache)==3
    device.last_gpu_tuning['calibration']['selected']='mutated-public-report'
    net.restore('pending',restore_random_state=True);net.run(6*DT);capture('restored-again','hit')
    assert len(device._gpu_tuning_cache)==3
    assert records[1]['report']['cache']['input_sha256']==records[2]['report']['cache']['input_sha256']
    assert records[3]['report']['cache']['input_sha256']!=records[2]['report']['cache']['input_sha256']
    net.restore('pending');net.run(6*DT);capture('new-rng','miss')
    assert len(device._gpu_tuning_cache)==4
    (tmp_path/'cache-records.json').write_text(json.dumps(records)+'\n')
    device.close_gpu()


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('failure',['write','load'])
def test_native_failed_publication_does_not_cache_calibration(device,tmp_path,backend,failure,monkeypatch):
    import brian2 as b
    from test_gpu_composed_models import setup,DT
    setup(tmp_path/'run',backend,gpu_autotune=True,gpu_autotune_cache=True,gpu_compile_reuse=True)
    p=b.NeuronGroup(12,'dv/dt=128/second:1',threshold='v>=1',reset='v-=1',dt=DT,method='euler')
    net=b.Network(p)
    def fail(*args,**kwargs):raise RuntimeError('injected publication failure')
    if failure=='write':monkeypatch.setattr(importlib.import_module('brian2_rust.'+backend),'write_'+backend+'_results',fail)
    else:monkeypatch.setattr(device,'_load_results',fail)
    with pytest.raises(RuntimeError,match='injected publication failure'):net.run(8*DT)
    assert len(device._gpu_tuning_cache)==0
    assert net.t==0*b.second
    device.close_gpu()
