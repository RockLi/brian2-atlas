"""Activation-local validation preserves identity, isolation and native plans."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from brian2_rust import gpu_validation as validation
from brian2_rust.plan import PlanValidationError
from brian2_rust.protocol import canonical_bytes, PREVIOUS_SCHEMAS
from test_metal_delays import device, ROOT
from test_gpu_spike_generator import BACKENDS


def minimal():
    return json.loads((ROOT/'tests/golden/b2ir-v1/minimal-v1.json').read_text())


def test_one_semantic_validation_independent_plans_and_models(monkeypatch):
    original=validation.validate_model;calls=[]
    def checked(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(validation,'validate_model',checked)
    model=minimal();before=canonical_bytes(model)
    activation=validation.ValidatedActivation(model)
    for backend in ('metal','cuda'):
        module=importlib.import_module('brian2_rust.'+backend)
        public=getattr(module,'build_'+backend+'_plan')
        options=dict(numeric_mode='float32',event_delivery='sparse',synapse_sparse='bitset')
        assert activation.plan(backend,**options).to_dict()==public(model,**options).to_dict()
    assert len(calls)==1
    first=validation.executor_model(model,activation=activation)
    second=validation.executor_model(model,activation=activation)
    assert first==second==model and first is not second
    first['instance']['populations'].clear()
    assert canonical_bytes(second)==before==canonical_bytes(model)
    assert canonical_bytes(activation.snapshot(model))==before
    assert len(calls)==1
    with pytest.raises(FrozenInstanceError):activation._validated_wire=b'{}'
    with pytest.raises(PlanValidationError,match='activation-local'):
        validation.executor_model(model,activation=SimpleNamespace(_validated_wire=before))


@pytest.mark.parametrize('field',['instance','run','definition','protocol'])
def test_any_live_layer_mutation_rejects_snapshot(field):
    model=minimal();activation=validation.ValidatedActivation(model)
    changed=deepcopy(model);changed[field]['undeclared_change']=1
    with pytest.raises(PlanValidationError,match='changed after validation'):
        activation.snapshot(changed)


@pytest.mark.parametrize('during',[False,True])
def test_changed_validator_binary_rejects(tmp_path,monkeypatch,during):
    path=tmp_path/'validator';path.write_bytes(b'before')
    def validate(model,**kwargs):
        if during:path.write_bytes(b'after!')
        return deepcopy(model)
    monkeypatch.setattr(validation,'validate_model',validate)
    if during:
        with pytest.raises(PlanValidationError,match='changed during'):
            validation.ValidatedActivation(minimal(),runner=path)
    else:
        activation=validation.ValidatedActivation(minimal(),runner=path)
        path.write_bytes(b'after!')
        with pytest.raises(PlanValidationError,match='changed after'):
            activation.snapshot(minimal(),runner=path)


def test_invalid_initial_model_still_runs_independent_semantic_check():
    from brian2_rust.protocol import attach_protocol
    model=minimal();model['definition']['populations'][0]['size']=0;attach_protocol(model)
    with pytest.raises(PlanValidationError):validation.ValidatedActivation(model)


def test_changed_current_input_during_validation_is_rejected(monkeypatch):
    original=validation.validate_model
    def changed(model,**kwargs):
        current=original(model,**kwargs)
        current['instance']['rng_seed']+=1
        return current
    monkeypatch.setattr(validation,'validate_model',changed)
    with pytest.raises(PlanValidationError,match='input changed during'):
        validation.ValidatedActivation(minimal())


@pytest.mark.parametrize('schema',PREVIOUS_SCHEMAS)
def test_frozen_legacy_migration_preserves_snapshot_and_plan(schema):
    model=json.loads((ROOT/'tests/golden/b2ir-v1'/('minimal-'+schema.rsplit('-',1)[1]+'.json')).read_text())
    activation=validation.ValidatedActivation(model)
    expected=validation.validate_model(model)
    assert activation.snapshot(model)==expected
    from brian2_rust.metal import build_metal_plan
    assert activation.plan('metal',numeric_mode='float32').to_dict()==build_metal_plan(model,numeric_mode='float32').to_dict()


@pytest.mark.parametrize('kind,distribution',[('total','uniform'),('total','normal'),('indegree','uniform'),('indegree','normal')])
def test_procedural_snapshots_keep_initializers_independent(device,tmp_path,kind,distribution):
    from test_gpu_initialization import make_model,endpoints
    from brian2_rust.gpu_initialization import prepare_model
    model=make_model(device,tmp_path,kind,distribution);before=canonical_bytes(model)
    activation=validation.ValidatedActivation(model)
    first=activation.snapshot(model);second=activation.snapshot(model)
    original=prepare_model(model)
    a=first['instance']['synapses'][0];b=second['instance']['synapses'][0]
    expected=endpoints(kind)
    for key,values in [('source',expected[:,0]),('target',expected[:,1])]:
        np.testing.assert_array_equal(a[key],values)
        np.testing.assert_array_equal(a[key],b[key])
        assert not np.shares_memory(a[key],b[key]) and not a[key].flags.writeable
    np.testing.assert_array_equal(a['parameters']['w'],original['instance']['synapses'][0]['parameters']['w'])
    assert first.initializations==second.initializations==original.initializations
    assert first.initialization_seconds>0 and second.initialization_seconds>0
    first['definition']['populations'].clear()
    assert second['definition']==model['definition'] and canonical_bytes(model)==before


def test_binary_input_is_rechecked_during_host_initialization(device,tmp_path):
    import brian2 as b
    import brian2_rust as rust
    from brian2_rust.binary_topology import HEADER,MAGIC
    from brian2_rust.export import lower_network
    from test_gpu_initialization import DT
    b.set_device('rust_standalone',engine='reference',runner=ROOT/'target/release/b2-runner')
    path=tmp_path/'edges.b2csr'
    path.write_bytes(HEADER.pack(MAGIC,2,2,2,1)+np.array([0,1,2],'<u8').tobytes()+
                     np.array([0,1],'<u4').tobytes()+np.array([1,2],'<f8').tobytes())
    pre=b.SpikeGeneratorGroup(2,[0,1],[0,0]*DT,dt=DT)
    post=b.NeuronGroup(2,'v:1',clock=pre.clock)
    syn=b.Synapses(pre,post,'w:1 (constant)',on_pre='v_post+=w',clock=pre.clock)
    rust.connect_binary_csr(syn,path,parameters={'w':0})
    model=lower_network(b.Network(pre,post,syn),4*DT)
    activation=validation.ValidatedActivation(model);activation.snapshot(model)
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(PlanValidationError):activation.snapshot(model)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('flavor',['explicit','procedural'])
def test_native_snapshot_matches_public_plan_and_full_control(device,tmp_path,backend,flavor):
    from test_gpu_synapse_parallel import model_at
    from test_gpu_spike_generator import execute
    from test_gpu_initialization import make_model
    from test_gpu_composed_policies import full_compare
    from test_gpu_workgroup import save
    model=model_at(tmp_path/'ref') if flavor=='explicit' else make_model(device,tmp_path,'total','uniform')
    before=canonical_bytes(model);activation=validation.ValidatedActivation(model)
    module=importlib.import_module('brian2_rust.'+backend)
    cls=module.MetalExecutor if backend=='metal' else module.CudaExecutor
    options=dict(numeric_mode='float32',event_delivery='sparse',synapse_sparse='bitset')
    plan=activation.plan(backend,**options)
    public=getattr(module,'build_'+backend+'_plan')(model,**options)
    assert plan.to_dict()==public.to_dict()
    expected=execute(model,tmp_path/'control','cpu-f32','sparse')
    with cls(model,tmp_path/'native',plan=plan,_validated_input=activation,**options) as ex:
        actual=ex.run();full_compare(actual,expected)
        full_compare(ex.run(),expected)
    assert canonical_bytes(model)==before
    save(tmp_path/'validation-results.npz',actual,expected)
    (tmp_path/'validation-model.json').write_bytes(before)
    (tmp_path/'validation-plan.json').write_text(plan.to_json())
    poisoned=replace(plan,kernels=(replace(plan.kernels[0],source=plan.kernels[0].source+'\n// changed'),*plan.kernels[1:]))
    with pytest.raises(PlanValidationError,match='plan does not match'):
        cls(model,tmp_path/'poisoned',plan=poisoned,_validated_input=activation,**options)


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('cache',[False,True])
def test_native_each_device_activation_validates_once(device,tmp_path,backend,cache,monkeypatch):
    import brian2 as b
    from test_gpu_composed_models import setup,DT
    setup(tmp_path/'run',backend,gpu_autotune=True,gpu_autotune_cache=cache,gpu_compile_reuse=True)
    p=b.NeuronGroup(12,'dv/dt=128/second:1',threshold='v>=1',reset='v-=1',dt=DT,method='euler')
    spikes=b.SpikeMonitor(p);net=b.Network(p,spikes)
    original=validation.validate_model;calls=[]
    def checked(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(validation,'validate_model',checked)
    for i in range(2):
        net.run(8*DT);assert len(calls)==i+1
        np.testing.assert_array_equal(p.v[:],np.zeros(12))
        np.testing.assert_array_equal(spikes.count[:],np.full(12,i+1))
    device.close_gpu()
