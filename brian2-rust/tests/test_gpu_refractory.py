"""Expression refractory: sequential gates, integer elapsed ticks and failures."""
import json
import subprocess
import copy

import brian2 as b
import numpy as np
import pytest

from brian2_rust.export import lower_network
from brian2_rust.protocol import attach_protocol
from brian2_rust.spec import bits
from brian2_rust.metal import number
from brian2_rust.results import load_results
from test_gpu_spike_generator import BACKENDS, execute
from test_metal_delays import device, ROOT
from test_metal_plasticity import equivalent


def refresh_code(model,code):
    from brian2_rust.spec import loads
    from brian2_rust.schedule import build_schedule
    statements=code['scalar']+code['vector']
    inputs=set(code['effects']['reads'])|set(code['effects']['writes'])|{'dt'}
    reads=set().union(*(loads(s['value']) for s in statements)) & inputs
    reads|={s['condition'] for s in statements if s['condition'] is not None}
    code['effects']['reads']=sorted(reads)
    model['definition']['schedule']=build_schedule(model['definition'],model['instance'],model['definition']['schedule']['base_slots'])
    attach_protocol(model)


def make_model(device,tmp_path,kind,coupled=False):
    b.set_device('rust_standalone',engine='reference',directory=tmp_path/'reference',runner=ROOT/'target/release/b2-runner')
    refractory={'constant':'3*dt','variable':'tau_ref','dynamic':'tau_ref','boolean':'x < 3',
                'masked':'True'}[kind]
    pop=b.NeuronGroup(3,'dv/dt=512/second:1 (unless refractory)\ndx/dt=1024/second:1\ntau_ref:second',
        threshold='v>=1',reset='v=0; x=0',refractory=refractory,dt=b.second/1024,method='euler',name='population')
    pop.v=.5;pop.tau_ref=[0,2,3]*pop.clock.dt
    if kind=='dynamic':pop.run_regularly('tau_ref = (1+int(x>1))*dt',when='start')
    monitor=b.StateMonitor(pop,['v','x'],record=True);spikes=b.SpikeMonitor(pop)
    objects=[pop,monitor,spikes]
    if coupled:
        source=b.SpikeGeneratorGroup(1,[0],[0]*b.second,period=pop.clock.dt,clock=pop.clock,name='input')
        syn=b.Synapses(source,pop,'w:1',on_pre='v_post+=w',clock=pop.clock)
        syn.connect(i=[0,0,0],j=[0,1,2]);syn.w=.125
        objects.extend([source,syn])
    model=lower_network(b.Network(*objects),10*pop.clock.dt)
    if kind=='masked':
        model['instance']['populations'][0]['refractory']['initial_not_refractory']=[False]*3
        code=next(c for c in model['definition']['populations'][0]['code_objects'] if c['kind']=='state_update')
        stmt=next(s for s in code['vector'] if s['target']=='v')
        assert stmt['condition']=='not_refractory'
        stmt['value']={'op':'tick_to_f64','arg':{'op':'timestep',
                      'time':{'op':'neg','arg':{'op':'load','name':'dt'}},'dt':{'op':'load','name':'dt'}}}
        refresh_code(model,code)
    return model


def reference(model,tmp_path,success=True):
    path=tmp_path/'model.json';path.write_text(json.dumps(model))
    run=subprocess.run([str(ROOT/'target/release/b2-runner'),str(path),str(tmp_path/'oracle')],capture_output=True,text=True)
    if not success:
        assert run.returncode!=0
        assert 'timestep' in run.stderr
        return
    assert run.returncode==0,run.stderr
    return load_results(model,tmp_path/'oracle')


def compare(actual,expected):
    equivalent(actual,expected,exact=True)
    for a,e in zip(actual['populations'],expected['populations'],strict=True):
        if e['refractory'] is not None:
            for key in ('lastspike','not_refractory'):
                np.testing.assert_array_equal(a['refractory'][key],e['refractory'][key])


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('kind',['constant','variable','dynamic','boolean','masked'])
def test_expression_gate_and_frozen_state(device,tmp_path,backend,kind):
    model=make_model(device,tmp_path,kind)
    expected=reference(model,tmp_path)
    actual=execute(model,tmp_path/backend,backend)
    compare(actual,expected)
    if backend!='cpu-f32':compare(load_results(model,tmp_path/backend/'transport'),expected)
    if kind=='constant':
        np.testing.assert_array_equal(actual['populations'][0]['spike_ticks'],np.repeat([0,4,8],3))


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('kind',['variable','boolean'])
def test_expression_coupled_frozen_synaptic_writes(device,tmp_path,backend,kind):
    model=make_model(device,tmp_path,kind,coupled=True)
    expected=reference(model,tmp_path)
    for route in ('scan','sparse'):
        compare(execute(model,tmp_path/(backend+route),backend,route),expected)


@pytest.mark.parametrize('backend',BACKENDS)
def test_duration_large_ticks_and_fractional_initial_lastspike(device,tmp_path,backend):
    model=make_model(device,tmp_path,'variable')
    shift=2**25;dt=number(model['definition']['clocks'][0]['dt'])
    model['run']['start']=bits(shift*dt);model['run']['clocks'][0]['start_tick']=shift
    ref=model['instance']['populations'][0]['refractory']
    ref['initial_lastspike']=[bits((shift-x)*dt) for x in (2.25,1,0)]
    ref['initial_not_refractory']=[False]*3
    attach_protocol(model)
    compare(execute(model,tmp_path/backend,backend),reference(model,tmp_path))


@pytest.mark.parametrize('backend',BACKENDS)
def test_reset_scalar_timestep_is_checked_without_spikes(device,tmp_path,backend):
    model=make_model(device,tmp_path,'constant')
    codes=model['definition']['populations'][0]['code_objects']
    threshold=next(c for c in codes if c['kind']=='threshold')
    next(s for s in threshold['vector'] if s['target']=='_cond')['value']={'op':'boolean','value':False}
    refresh_code(model,threshold)
    reset=next(c for c in codes if c['kind']=='reset')
    dt={'op':'load','name':'dt'}
    reset['scalar'].append({'target':'_checked_time','dtype':'f64','dimensions':[0.]*7,'condition':None,
      'value':{'op':'tick_to_f64','arg':{'op':'timestep','time':{'op':'neg','arg':dt},'dt':dt}}})
    refresh_code(model,reset)
    reference(model,tmp_path,success=False)
    with pytest.raises(FloatingPointError,match='timestep'):
        execute(model,tmp_path/backend,backend)


@pytest.mark.parametrize('backend',BACKENDS)
def test_boolean_operands_preserve_eager_timestep_failure(device,tmp_path,backend):
    model=make_model(device,tmp_path,'boolean')
    code=next(c for c in model['definition']['populations'][0]['code_objects'] if c['kind']=='state_update')
    stmt=next(s for s in code['vector'] if s['target']=='not_refractory')
    dt={'op':'load','name':'dt'}
    stmt['value']={'op':'or','left':{'op':'boolean','value':True},'right':{'op':'eq',
      'left':{'op':'timestep','time':{'op':'neg','arg':dt},'dt':dt},
      'right':{'op':'timestep','time':dt,'dt':dt}}}
    refresh_code(model,code)
    reference(model,tmp_path,success=False)
    with pytest.raises(FloatingPointError,match='timestep'):
        execute(model,tmp_path/backend,backend)


@pytest.mark.parametrize('backend',BACKENDS)
@pytest.mark.parametrize('bad',[-1.,float(2**64)])
def test_invalid_duration_raises_instead_of_publishing(device,tmp_path,backend,bad):
    model=make_model(device,tmp_path,'variable')
    dt=number(model['definition']['clocks'][0]['dt'])
    model['instance']['populations'][0]['initial_state']['tau_ref']=[bits(bad*dt)]*3
    attach_protocol(model)
    reference(model,tmp_path,success=False)
    with pytest.raises(FloatingPointError,match='timestep'):
        execute(model,tmp_path/backend,backend)
    assert not (tmp_path/backend/'transport'/'summary.json').exists()


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('kind',['variable','boolean'])
def test_expression_device_continuation_restore(device,tmp_path,backend,kind):
    snapshots=[]
    for engine in ('reference',backend):
        device.reinit()
        options={'numeric_mode':'float32'} if engine!='reference' else {}
        b.set_device('rust_standalone',engine=engine,directory=tmp_path/engine,runner=ROOT/'target/release/b2-runner',**options)
        pop=b.NeuronGroup(2,'dv/dt=512/second:1 (unless refractory)\ndx/dt=1024/second:1\ntau_ref:second',
          threshold='v>=1',reset='v=0; x=0',refractory='tau_ref' if kind=='variable' else 'x<3',
          dt=b.second/1024,method='euler')
        pop.v=.5;pop.tau_ref=[2,3]*pop.clock.dt
        spikes=b.SpikeMonitor(pop);monitor=b.StateMonitor(pop,['v','x'],record=True)
        net=b.Network(pop,spikes,monitor);net.run(2*pop.clock.dt);net.store('refractory')
        state=copy.deepcopy(np.asarray(pop.not_refractory[:]))
        net.run(8*pop.clock.dt);v=np.asarray(pop.v[:]).copy()
        net.restore('refractory');np.testing.assert_array_equal(pop.not_refractory[:],state)
        net.run(8*pop.clock.dt);np.testing.assert_array_equal(pop.v[:],v)
        snapshots.append([np.asarray(value).copy() for value in (pop.v[:],pop.x[:],pop.lastspike[:],pop.not_refractory[:],spikes.t[:],spikes.i[:],monitor.v,monitor.x)])
    for a,e in zip(*snapshots,strict=True):np.testing.assert_array_equal(a,e)
