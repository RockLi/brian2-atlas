"""WASM plan validation, resumable execution and native semantic parity.

Build tools/build_wasm.py before running. Node executes the same web package;
wasm/check-browser.html additionally exercises the dedicated browser Worker.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2_rust.export import lower_network
from brian2_rust.plan import build_execution_plan, verify_execution_plan, explain_plan, bind_execution_plan, PlanValidationError
from brian2_rust.protocol import attach_protocol
from brian2_rust.results import load_results
from brian2_rust.wasm import export_wasm_bundle
from test_metal_delays import device, ROOT

PKG = ROOT/'output/wasm/pkg'
DT = b.second/1024


def test_plan_identity_and_export(tmp_path):
    model = json.loads((ROOT/'tests/golden/b2ir-v1/minimal-v1.json').read_text())
    original = copy.deepcopy(model)
    plan = build_execution_plan(model, backend='wasm')
    assert model == original
    invalid=copy.deepcopy(model); invalid['run']['duration']='0000000000000000'; attach_protocol(invalid)
    with pytest.raises(PlanValidationError): build_execution_plan(invalid,backend='wasm')
    assert plan.logical == build_execution_plan(model).logical
    assert 'WASM' in explain_plan(plan)
    verify_execution_plan(plan, model)
    with pytest.raises(PlanValidationError):
        verify_execution_plan(replace(plan, strategy='forged'), model)
    with pytest.raises(PlanValidationError):
        export_wasm_bundle(model, tmp_path/'bad.json', plan=replace(plan, run_sha256='0'*64))
    assert not (tmp_path/'bad.json').exists()
    with pytest.raises(PlanValidationError): build_execution_plan(model, backend='wasm', numeric_mode='float32')
    with pytest.raises(PlanValidationError): build_execution_plan(model, backend='wasm', event_delivery='sparse')
    path=tmp_path/'bundle.json'; export_wasm_bundle(model,path)
    bundle=json.loads(path.read_text())
    assert json.loads(bundle['model_json']) == model
    assert bundle['plan_sha256'] == plan.sha256
    metadata=dict(backend='wasm',plan_sha256=plan.sha256,numeric_profile='reference-f64')
    assert bind_execution_plan(plan,metadata).backend=='wasm'
    with pytest.raises(PlanValidationError): bind_execution_plan(plan,{**metadata,'plan_sha256':'bad'})


def make_model(case, device, directory):
    b.set_device('rust_standalone', engine='reference', directory=directory,
                 runner=ROOT/'target/release/b2-runner')
    if case in ('golden','large_seed'):
        model=json.loads((ROOT/'tests/golden/b2ir-v1/minimal-v1.json').read_text())
        if case=='large_seed': model['instance']['rng_seed']=2**64-1
        attach_protocol(model)
        return model
    if case=='multiclock':
        from test_gpu_multiclock import network
        net,*_=network(True)
        # Second segment carries pending synaptic events and nonzero clocks.
        net.run(6*DT); net.run(6*DT)
        return json.loads((device.last_run_directory/'model.json').read_text())
    if case=='links':
        from test_gpu_links import linked_network
        net,*_=linked_network(feedback=True)
        return lower_network(net,8*DT)
    if case=='procedural':
        from brian2_rust import connect_fixed_indegree, Uniform
        pre=b.SpikeGeneratorGroup(4,[0,1,2,3],[0,1,2,3]*DT,period=4*DT,dt=DT)
        post=b.NeuronGroup(3,'v:1',dt=DT)
        syn=b.Synapses(pre,post,'w:1 (constant)',on_pre='v_post+=w',clock=pre.clock)
        connect_fixed_indegree(syn,2,seed=912,initializers={'w':Uniform(.1,.5)})
        mon=b.StateMonitor(post,'v',record=True)
        return lower_network(b.Network(pre,post,syn,mon),8*DT)
    if case=='owner_clocks':
        pop=b.NeuronGroup(3,'v:1',dt=DT,threshold='False',reset='')
        pop.run_regularly('v+=t/second+dt/second',dt=DT/2)
        syn=b.Synapses(pop,pop,'w:1',clock=pop.clock,on_pre='w+=0')
        syn.connect(i=[0,1,2],j=[1,2,0])
        syn.run_regularly('w+=dt/second+t/second',dt=2*DT,when='end')
        mon=b.StateMonitor(pop,'v',record=True)
        return lower_network(b.Network(pop,syn,mon),6*DT)
    if case=='rng':
        pop=b.NeuronGroup(257,'u:1\nn:1',dt=DT)
        pop.run_regularly('u=rand(); n=randn()')
        mon=b.StateMonitor(pop,['u','n'],record=[0,256])
        return lower_network(b.Network(pop,mon),7*DT,rng_seed=2**64-1)
    if case=='function_timed_array':
        from test_population import PORTABLE_VOLTAGE_FUNCTION
        drive=b.TimedArray(np.array([0,.1,-.2,.3,.2,.1])*b.volt,dt=2*DT)
        pop=b.NeuronGroup(3,'dv/dt=portable_voltage_rhs(v,tau,scale)+drive(t)/tau:volt',
            method='euler',dt=DT,namespace=dict(portable_voltage_rhs=PORTABLE_VOLTAGE_FUNCTION,
                drive=drive,tau=8*DT,scale=5*b.volt))
        pop.v=[-3,2,7]*b.volt
        mon=b.StateMonitor(pop,'v',record=True)
        return lower_network(b.Network(pop,mon),10*DT)
    if case=='custom':
        pop=b.NeuronGroup(3,'v:1',dt=DT,events={'burst':'v>1'})
        pop.run_regularly('v+=0.5');pop.run_on_event('burst','v=0')
        state=b.StateMonitor(pop,'v',record=True)
        event=b.EventMonitor(pop,'burst',variables='v')
        return lower_network(b.Network(pop,state,event),8*DT)
    raise AssertionError(case)


def compare(actual, expected):
    if isinstance(expected,dict):
        for key in expected:
            if key not in {'metadata','directory','_dump','_event_dump'}: compare(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for a,e in zip(actual,expected): compare(a,e)
    elif isinstance(expected,np.ndarray):
        if expected.dtype.kind=='f': np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-15)
        else: np.testing.assert_array_equal(actual,expected)
    else: assert actual==expected


@pytest.mark.parametrize('case',['golden','large_seed','multiclock','links','procedural','rng','custom','function_timed_array','owner_clocks'])
def test_wasm_matches_native_with_tick_slicing(device,tmp_path,case):
    assert (PKG/'b2_runner_bg.wasm').is_file(), 'run tools/build_wasm.py first'
    (tmp_path/'case.txt').write_text(case)
    model=make_model(case,device,tmp_path/'device')
    source=tmp_path/'model.json';source.write_text(json.dumps(model))
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(source),str(tmp_path/'native')],check=True,capture_output=True)
    bundle=tmp_path/'bundle.json';plan=export_wasm_bundle(model,bundle)
    expected=load_results(model,tmp_path/'native')
    for budget in (1,7,10000):
        output=tmp_path/f'wasm-{budget}'
        subprocess.run(['node',str(ROOT/'wasm/check-node.mjs'),str(PKG),str(bundle),str(output),str(budget)],check=True,capture_output=True,text=True)
        actual=load_results(model,output)
        compare(actual,expected)
        assert actual['metadata']['plan_sha256']==plan.sha256
        assert (output/'results.bin').read_bytes()==(tmp_path/'wasm-1/results.bin').read_bytes()


def test_wasm_runtime_error_is_terminal(device,tmp_path):
    b.set_device('rust_standalone',engine='reference',directory=tmp_path/'device',
                 runner=ROOT/'target/release/b2-runner')
    source=b.NeuronGroup(1,'x:1',dt=DT)
    target=b.NeuronGroup(2,'y:1\nexternal:1 (linked)\nk:integer',dt=DT)
    target.k=[0,2];target.external=b.linked_var(source,'x',index='k')
    target.run_regularly('y=external')
    model=lower_network(b.Network(source,target),DT)
    path=tmp_path/'bundle.json';export_wasm_bundle(model,path)
    subprocess.run(['node',str(ROOT/'wasm/check-error.mjs'),str(PKG),str(path)],check=True,capture_output=True,text=True)


def test_browser_authoring_and_experiment_metrics(tmp_path):
    subprocess.run(['node', str(ROOT/'wasm/check-spa.mjs'), str(PKG),
                    str(ROOT/'output/wasm/editor-template.json'), str(tmp_path)],
                   check=True, capture_output=True, text=True)
    for case in tmp_path.iterdir():
        subprocess.run([str(ROOT/'target/release/b2-runner'), str(case/'model.json'),
                        str(case/'native')], check=True, capture_output=True)
        for filename in ('results.bin', 'events.bin'):
            assert (case/filename).read_bytes() == (case/'native'/filename).read_bytes()


@pytest.fixture(scope='module')
def classic_model_results(tmp_path_factory):
    output=tmp_path_factory.mktemp('classic-models')
    result=subprocess.run(['node', str(ROOT/'wasm/check-models.mjs'),
                           str(ROOT/'output/wasm'), str(output)],
                          check=True, capture_output=True, text=True)
    (output/'metrics.json').write_text(result.stdout)
    return output


@pytest.mark.parametrize('case', ['adaptive_lif-asynchronous', 'adaptive_lif-synchronous',
    'adaptive_lif-adapting', 'adaptive_lif-silent', 'izhikevich-regular',
    'izhikevich-bursting', 'izhikevich-fast', 'izhikevich-silent',
    'hodgkin_huxley-tonic', 'hodgkin_huxley-strong',
    'hodgkin_huxley-synchronous', 'hodgkin_huxley-silent'])
def test_classic_models_native_parity(classic_model_results, case):
    directory=classic_model_results/case
    model=json.loads((directory/'model.json').read_text())
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(directory/'model.json'),
                    str(directory/'native')],check=True,capture_output=True)
    # Transcendental HH kinetics may differ slightly between native libm and WASM.
    actual=load_results(model,directory)['populations'][0]
    expected=load_results(model,directory/'native')['populations'][0]
    for field in ('spike_ticks','indices','counts'):
        np.testing.assert_array_equal(actual[field],expected[field])
    for field in ('trace','states'):
        for name in actual[field]:
            np.testing.assert_allclose(actual[field][name],expected[field][name],rtol=1e-10,atol=1e-9)


@pytest.mark.parametrize('case', ['izhikevich-regular', 'izhikevich-bursting',
                                  'hodgkin_huxley-tonic', 'hodgkin_huxley-silent'])
def test_classic_models_brian_numpy(classic_model_results, case):
    import importlib.util
    import struct
    spec=importlib.util.spec_from_file_location('spa_template', ROOT/'tools/spa_template.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    directory=classic_model_results/case
    model=json.loads((directory/'model.json').read_text())
    config=json.loads((directory/'config.json').read_text())
    definition=model['definition']['populations'][0]
    instance=model['instance']['populations'][0]
    previous=b.get_device();target=b.prefs.codegen.target
    try:
        b.set_device('runtime');b.prefs.codegen.target='numpy';b.start_scope()
        group,variables=module.create_group(config['model'],config['neurons'],config['dt_ms'])
        for mapping in ('initial_state','parameters'):
            for name,values in instance[mapping].items():
                decoded=np.array([struct.unpack('>d',bytes.fromhex(v))[0] for v in values])
                if name not in group.variables:
                    assert name=='ms' and np.array_equal(decoded,[.001])
                    continue
                group.variables[name].set_value(decoded)
        states=b.StateMonitor(group,variables,record=definition['monitor']['record'])
        spikes=b.SpikeMonitor(group)
        b.Network(group,states,spikes).run(config['duration_ms']*b.ms)
        actual=load_results(model,directory)['populations'][0]
        np.testing.assert_array_equal(actual['indices'],np.asarray(spikes.i))
        np.testing.assert_array_equal(actual['spike_ticks'],np.rint(spikes.t/(config['dt_ms']*b.ms)).astype(int))
        for name in variables:
            np.testing.assert_allclose(actual['trace'][name],np.asarray(getattr(states,name)).T,rtol=1e-8,atol=1e-8)
    finally:
        b.prefs.codegen.target=target;b.set_device(previous)


def test_equation_state_monitor_outputs():
    """Custom equations replace every monitor output from the base template."""
    subprocess.run(['node', str(ROOT/'wasm/check-equation-monitors.mjs'),
                    str(ROOT/'output/wasm')], check=True, capture_output=True, text=True)
