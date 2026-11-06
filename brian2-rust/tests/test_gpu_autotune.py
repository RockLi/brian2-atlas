"""Policy selection gates, full-result identity and native Device state ownership."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from brian2_rust import gpu_autotune as tuning
from test_metal_delays import device
from test_gpu_spike_generator import BACKENDS


def result():
    return dict(populations=[dict(states=dict(v=np.array([0.,1.],np.float32)),
        indices=np.array([1],np.int64),spike_ticks=np.array([2],np.int64))],
        synapses=[dict(states=dict(w=np.array([.25],np.float32)),events=2)],
        numeric_profile='test-f32',rng_profile='counter-test')


class Plan:
    def __init__(self,name):self.name=name
    def to_dict(self):return dict(strategy=self.name,chunk_abi='test')
    @property
    def sha256(self):return hashlib.sha256(json.dumps(self.to_dict()).encode()).hexdigest()


def fake_factory(monkeypatch,*,duplicate=False,change=None,fail=None,close_fail=None):
    clock=[0.];created={}
    monkeypatch.setattr(tuning.time,'perf_counter',lambda:clock[0])
    costs=dict(baseline=10.,prefix=8.,bitset=6.,**{'prefix-bitset':4.})
    class Executor:
        device_name='test GPU'
        def __init__(self,name):
            self.name=name;self.plan=Plan('baseline' if duplicate else name)
            self.calls=0;self.closed=False;created[name]=self
        def run(self):
            assert not self.closed
            self.calls+=1;clock[0]+=costs[self.name]
            if fail is not None:fail(self.name,self.calls)
            value=result()
            if change is not None:change(self.name,self.calls,value)
            return value
        def close(self):
            self.closed=True
            if close_fail==self.name:raise RuntimeError('close failed')
    def factory(name,prefix,sparse):
        clock[0]+=.125
        return Executor(name)
    return factory,created


def test_run_with_buffer_budget_preserves_default_and_forwards_override():
    class Executor:
        def __init__(self):self.calls=[]
        def run(self,**kwargs):self.calls.append(kwargs);return result()
    executor=Executor()
    tuning.run_with_buffer_budget(executor,tuning.DEFAULT_MAX_BUFFER_BYTES)
    tuning.run_with_buffer_budget(executor,8*1024**3)
    assert executor.calls==[{},dict(max_buffer_bytes=8*1024**3)]


def test_selection_full_replays_and_sole_winner_ownership(monkeypatch,tmp_path):
    factory,created=fake_factory(monkeypatch)
    winner,actual,report=tuning.tune(factory,tmp_path)
    assert report['selected']=='prefix-bitset' and len(report['samples'])==17
    assert report['total_seconds']>sum(s['seconds'] for s in report['samples'])
    assert report['selected_plan_sha256']==winner.plan.sha256
    assert tuning.observable_fingerprint(actual)==report['reference_observable_sha256']
    assert json.loads((tmp_path/'report.json').read_text())==report
    assert all(ex.closed for name,ex in created.items() if name!=report['selected'])
    assert not winner.closed and winner.calls==5
    winner.close()


@pytest.mark.parametrize('field',['state','spike','event-count','dtype','shape','profile','rng','signed-zero','missing'])
def test_full_observable_fingerprint_detects_changes(field):
    a=result();b=deepcopy(a)
    if field=='state':b['synapses'][0]['states']['w'][0]+=.125
    elif field=='spike':b['populations'][0]['spike_ticks'][0]+=1
    elif field=='event-count':b['synapses'][0]['events']+=1
    elif field=='dtype':b['populations'][0]['indices']=b['populations'][0]['indices'].astype(np.int32)
    elif field=='shape':b['populations'][0]['indices']=b['populations'][0]['indices'].reshape(1,1)
    elif field=='profile':b['numeric_profile']='different'
    elif field=='rng':b['rng_profile']='different'
    elif field=='signed-zero':b['populations'][0]['states']['v'][0]=-0.
    else:del b['synapses'][0]['states']['w']
    assert tuning.observable_fingerprint(a)!=tuning.observable_fingerprint(b)


@pytest.mark.parametrize('bad_round',[1,3])
def test_mismatched_candidate_is_excluded_even_after_good_warmup(monkeypatch,tmp_path,bad_round):
    def change(name,n,value):
        if name=='prefix-bitset' and n==bad_round:value['synapses'][0]['events']+=1
    factory,created=fake_factory(monkeypatch,change=change)
    winner,_,report=tuning.tune(factory,tmp_path)
    assert report['selected']=='bitset'
    assert report['candidates']['prefix-bitset']['status']=='rejected'
    assert 'differs from baseline' in report['candidates']['prefix-bitset']['reason']
    assert created['prefix-bitset'].closed
    winner.close()


@pytest.mark.parametrize('failure',['baseline-replay','selected-final','interrupt','cleanup'])
def test_failure_closes_every_owned_executor_and_never_publishes(monkeypatch,tmp_path,failure):
    def fail(name,n):
        if failure=='baseline-replay' and name=='baseline' and n==2:raise RuntimeError('baseline failed')
        if failure=='selected-final' and name=='prefix-bitset' and n==5:raise RuntimeError('final failed')
        if failure=='interrupt' and name=='bitset':raise KeyboardInterrupt()
    factory,created=fake_factory(monkeypatch,fail=fail,close_fail='prefix' if failure=='cleanup' else None)
    with pytest.raises((RuntimeError,KeyboardInterrupt)):tuning.tune(factory,tmp_path)
    assert all(ex.closed for ex in created.values())
    assert json.loads((tmp_path/'report.json').read_text())['status']=='failed'


def test_duplicate_complete_plans_are_not_replayed(monkeypatch,tmp_path):
    factory,created=fake_factory(monkeypatch,duplicate=True)
    winner,_,report=tuning.tune(factory,tmp_path)
    assert report['selected']=='baseline' and len(report['samples'])==5
    assert all(row['status']=='duplicate' for name,row in report['candidates'].items() if name!='baseline')
    assert all(ex.calls==0 and ex.closed for name,ex in created.items() if name!='baseline')
    winner.close()


def test_preplanned_duplicates_do_not_construct_or_compile(monkeypatch,tmp_path):
    factory,created=fake_factory(monkeypatch,duplicate=True)
    winner,_,report=tuning.tune(factory,tmp_path,plan_for=lambda *args:Plan('baseline'))
    assert list(created)==['baseline'] and len(report['samples'])==5
    assert all(row['compiled'] is False for name,row in report['candidates'].items() if name!='baseline')
    winner.close()


def test_prepared_plan_mismatch_fails_closed(monkeypatch,tmp_path):
    factory,created=fake_factory(monkeypatch)
    with pytest.raises(ValueError,match='prepared tuning plan'):
        tuning.tune(factory,tmp_path,plan_for=lambda *args:Plan('different'))
    assert created['baseline'].closed and created['baseline'].calls==0


def test_unsupported_candidate_construction_falls_back(monkeypatch,tmp_path):
    make,created=fake_factory(monkeypatch)
    def factory(name,prefix,sparse):
        if name!='baseline':raise ValueError('unsupported candidate')
        return make(name,prefix,sparse)
    winner,_,report=tuning.tune(factory,tmp_path)
    assert report['selected']=='baseline' and len(created)==1
    assert all(row['status']=='rejected' for name,row in report['candidates'].items() if name!='baseline')
    winner.close()


@pytest.mark.parametrize('samples,expected',[([8,9,10],'baseline'),([9.6]*3,'baseline'),([9.4]*3,'prefix')])
def test_selection_requires_gain_and_separated_ranges(samples,expected):
    rows={name:dict(status='eligible',seconds=values) for name,values in [('baseline',[10,10,10]),('prefix',samples)]}
    assert tuning.select_candidate(rows)==expected


def test_option_validation_before_any_gpu_setup(device,tmp_path):
    import brian2 as b
    for value in (None,0,1,'yes',np.bool_(True)):
        with pytest.raises(NotImplementedError,match='gpu_autotune must be boolean'):
            b.set_device('rust_standalone',engine='metal',numeric_mode='float32',gpu_autotune=value)
    for engine in ('aot','reference'):
        with pytest.raises(NotImplementedError,match='GPU engine'):
            b.set_device('rust_standalone',engine=engine,gpu_autotune=True)
    for flag in ('gpu_synapse_prefix','gpu_synapse_sparse','gpu_synapse_fusion'):
        with pytest.raises(NotImplementedError,match='explicit synapse policy'):
            b.set_device('rust_standalone',engine='cuda',numeric_mode='float32',gpu_autotune=True,**{flag:False})


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('queued,seed,reuse',[(False,7,False),(False,42,True),(True,7,False),(True,42,True)])
def test_native_device_tunes_without_advancing_state_twice(device,tmp_path,backend,queued,seed,reuse):
    import brian2 as b
    from brian2_rust.results import load_results
    from test_gpu_composed_models import setup,snapshot,DT
    from test_gpu_composed_policies import mixed,full_compare
    from test_gpu_synapse_parallel import control
    from test_gpu_expression_contract import oracle
    from test_gpu_workgroup import save
    records=[];models=[];saved=[]
    for autotune in (False,True):
        device.reinit()
        setup(tmp_path/str(autotune),backend,build_on_run=not queued,gpu_autotune=autotune,
              gpu_buffer_reuse=reuse,gpu_compile_reuse=reuse)
        net,pre,post,syn,monitors,spikes,events=mixed(seed,traced=seed==42)
        checkpoints={}
        def capture(label):
            checkpoints.update({label+'/'+k:v for k,v in snapshot(pre,post,syn,monitors,spikes,events).items()})
            if not autotune:
                assert device.last_gpu_tuning is None
                return
            directory=device.last_run_directory
            report=device.last_gpu_tuning
            assert report['status']=='passed' and report['selected_plan_sha256']==device.last_execution_plan.sha256
            assert report==json.loads((directory/'gpu-autotune/report.json').read_text())
            assert report==json.loads((directory/'rust/summary.json').read_text())[backend+'_runtime']['autotune']
            assert 5<=len(report['samples'])<=17
            for name,row in report['candidates'].items():
                if name!='baseline' and row['status']=='eligible':assert row['compilation']['kernels_reused']>0
            assert all(s['observable_sha256']==report['reference_observable_sha256'] for s in report['samples'])
            assert (device._gpu_executor is not None)==reuse
            if reuse:assert device._gpu_executor.plan.sha256==report['selected_plan_sha256']
            model=json.loads((directory/'model.json').read_text());models.append(model)
            actual=load_results(model,directory/'rust')
            expected=control(model,directory/'cpu-f32',3)
            full_compare(actual,expected)
            independent=oracle(model,directory/'independent-f64')
            full_compare(actual,independent,trace_tolerance=seed==42)
            save(tmp_path/f'autotune-{label}.npz',actual,expected)
            saved.append(dict(label=label,report=report,plan=device.last_execution_plan.to_dict()))
        net.run(6*DT)
        if not queued:capture('first');net.store('pending')
        net.run(6*DT)
        if not queued:
            capture('second');before=snapshot(pre,post,syn,monitors,spikes,events)
            net.restore('pending');net.run(6*DT);capture('restored')
            after=snapshot(pre,post,syn,monitors,spikes,events)
            for key in before:np.testing.assert_array_equal(after[key],before[key],err_msg=key)
            syn.w=np.asarray(syn.w[:])+1/4096
        net.run(12*DT)
        if queued:device.build()
        capture('final');records.append(checkpoints)
        assert net.t==24*DT
        device.close_gpu()
    assert records[0].keys()==records[1].keys()
    for key in records[0]:np.testing.assert_array_equal(records[0][key],records[1][key],err_msg=key)
    assert len(models)==(1 if queued else 4)
    pending=any(p['pending'] for m in models for s in m['instance']['synapses'] for p in s['pathways'])
    assert pending!=queued
    np.savez_compressed(tmp_path/'autotune-device.npz',**{str(i)+'/'+k:v for i,row in enumerate(records) for k,v in row.items()})
    (tmp_path/'autotune-records.json').write_text(json.dumps(saved)+'\n')
    (tmp_path/'autotune-models.json').write_text(json.dumps(models)+'\n')


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_native_seeded_random_replays(device,tmp_path,backend):
    from brian2_rust.metal import MetalExecutor
    from brian2_rust.cuda import CudaExecutor
    from test_gpu_synapse_parallel import model_at,control
    from test_gpu_composed_policies import full_compare
    from test_gpu_workgroup import save
    model=model_at(tmp_path/'ref',flavor='random')
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    winner,actual,report=tuning.tune(lambda name,prefix,sparse:cls(model,tmp_path/name,
        numeric_mode='float32',event_delivery='sparse',synapse_prefix=prefix,synapse_sparse=sparse),tmp_path/'tuning')
    try:
        expected=control(model,tmp_path/'control',3);full_compare(actual,expected)
        full_compare(winner.run(),actual)
        assert report['status']=='passed'
        save(tmp_path/'autotune-random.npz',actual,expected)
        (tmp_path/'autotune-random-model.json').write_text(json.dumps(model)+'\n')
    finally:winner.close()


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('failure',['baseline','publication'])
def test_native_failed_activation_releases_candidates_and_preserves_arrays(device,tmp_path,backend,failure,monkeypatch):
    import importlib
    import brian2 as b
    from test_gpu_buffer_transfer import network,DT,ROOT
    b.set_device('rust_standalone',engine=backend,numeric_mode='float32',event_delivery='sparse',
        gpu_buffer_reuse=True,gpu_compile_reuse=True,gpu_autotune=True,
        directory=tmp_path/'run',runner=ROOT/'target/release/b2-runner')
    net,pre,post,syn,monitors,spikes=network(True)
    net.run(6*DT);previous=device._gpu_executor
    before=[np.asarray(pre.v[:]).copy(),np.asarray(post.v[:]).copy(),np.asarray(syn.w[:]).copy(),
        *(np.asarray(m.v[:]).copy() for m in monitors),*(np.asarray(s.t[:]).copy() for s in spikes)]
    module=importlib.import_module('brian2_rust.'+backend)
    cls=module.MetalExecutor if backend=='metal' else module.CudaExecutor
    closed=[];close=cls.close
    def tracked_close(ex):
        close(ex);closed.append(ex)
    monkeypatch.setattr(cls,'close',tracked_close)
    if failure=='baseline':
        def failed_run(ex,*args,**kwargs):raise RuntimeError('injected baseline failure')
        monkeypatch.setattr(cls,'run',failed_run)
    else:
        def failed_write(*args,**kwargs):raise RuntimeError('injected publication failure')
        monkeypatch.setattr(module,'write_'+backend+'_results',failed_write)
    with pytest.raises(RuntimeError,match='injected'):net.run(6*DT)
    after=[np.asarray(pre.v[:]),np.asarray(post.v[:]),np.asarray(syn.w[:]),
        *(np.asarray(m.v[:]) for m in monitors),*(np.asarray(s.t[:]) for s in spikes)]
    for a,e in zip(after,before,strict=True):np.testing.assert_array_equal(a,e)
    assert device._gpu_executor is None and previous in closed and len(closed)>=2
    assert all(not ex.handles if backend=='metal' else ex.closed for ex in closed)


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_native_temporal_fusion_deduplication_and_compile_only_reuse(device,tmp_path,backend):
    import brian2 as b
    from test_gpu_composed_models import setup,DT
    setup(tmp_path/'run',backend,gpu_autotune=True,gpu_compile_reuse=True,gpu_buffer_reuse=False)
    p=b.NeuronGroup(12,'dv/dt=128/second:1',threshold='v>=1',reset='v-=1',dt=DT,method='euler')
    mon=b.SpikeMonitor(p);net=b.Network(p,mon)
    reports=[]
    for i in range(2):
        net.run(8*DT)
        report=device.last_gpu_tuning;reports.append(deepcopy(report))
        assert report['selected']=='baseline' and len(report['samples'])==5
        assert all(row['status']=='duplicate' for name,row in report['candidates'].items() if name!='baseline')
        assert device.last_execution_plan.strategy=='independent-temporal-fusion'
        assert device._gpu_executor is not None
        assert device._gpu_executor._resident_dag_bytes==0 if backend=='metal' else device._gpu_executor._resident_dag is None
        np.testing.assert_array_equal(mon.count[:],np.full(12,i+1))
        np.testing.assert_array_equal(p.v[:],np.zeros(12))
    (tmp_path/'autotune-fused-reports.json').write_text(json.dumps(reports)+'\n')
    device.close_gpu()
