"""Event delay storage with clocks, summed state, refractory guards and RK."""
import copy
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_delays import cython_cache
from test_training_integer_ir import engine
from test_training_dynamic_refractory import plastic_model
from test_training_dynamic_clocks import model as clock_model
from test_training_event_delay import scalar_model


@pytest.mark.parametrize('method', ['euler','rk4'])
@pytest.mark.parametrize('warmup', [0,3])
def test_refractory_neuron_guards_do_not_freeze_event_delay_storage(method,warmup):
    net,inp,groups,x,dt,syn = plastic_model(('v','a'),3.5,method,sequential_reference=True)
    b.prefs.codegen.target = 'cython'
    syn.pre.delay = np.array([.24,.64,.04,.44])*b.ms
    syn.post.delay = np.array([.44,.04,.64,.24])*b.ms
    syn.pre.code += '\nw+=.001*delay/ms\ndelay=(.08+.4*int(v_post>.7))*ms'
    syn.post.code += '\nw-=.002*delay/ms\ndelay=(.08+.2*int(t>=.55*ms))*ms'
    if warmup:
        net.run(warmup*dt,namespace={})
    x = x[warmup:]
    bundle = lower_brian_dynamic_training(net,input_group=inp,layers=groups)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    trainer = NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    cursor = 0
    for length in [3,2,len(x)-5]:
        result = trainer.step(x[None,cursor:cursor+length],[0],initial='carry' if cursor else None)
        net.run(length*dt,namespace={});cursor+=length
        for group in groups:
            for name in ('v','a'):
                indices = bundle.provenance['neuron_state_layout'][group.name][name]
                np.testing.assert_allclose(np.asarray(result['final_state'])[0,indices],group.variables[name].get_value(),rtol=5e-13,atol=5e-14)
        indices = bundle.provenance['dynamic_state_layout'][syn.name]['w']
        np.testing.assert_allclose(np.asarray(result['final_state'])[0,indices],syn.w[:],rtol=5e-13,atol=5e-14)
        for path in syn._pathways:
            ids = bundle.provenance['pathway_state_layout'][path.name]['delay']
            np.testing.assert_allclose(np.asarray(result['final_state'])[0,ids],np.asarray(path.delay[:]),rtol=5e-13,atol=1e-15)
            assert path.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('clock_dt', [.3,.99999])
def test_async_time_driven_delay_values_and_summed_updates_match_brian(clock_dt):
    net,inp,groups,syn,x,dt,_ = clock_model(clock_dt,0,event_driven=True,third=True)
    b.prefs.codegen.target = 'cython'
    syn.pre.delay = np.array([.24,.64,.04,.44])*b.ms
    syn.post.delay = np.array([.44,.04,.64,.24])*b.ms
    syn.pre.code += '\nz+=.003*delay/ms\ndelay=(.08+.4*int(t>=.55*ms))*ms'
    syn.post.code += '\ndelay=(.08+.2*int(t>=.55*ms))*ms'
    net.run(.3*b.ms,namespace={})
    bundle = lower_brian_dynamic_training(net,input_group=inp,layers=groups)
    bundle.plan['trainable'] = [False]*len(bundle.weights)
    origin=bundle.plan['clock']['origin'];start=round(origin/float(dt));x=x[start:]
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    cursor=0
    for length in [2,1,len(x)-3]:
        result=trainer.step(x[None,cursor:cursor+length],[0],initial='carry' if cursor else None)
        cursor+=length
        net.run((origin+cursor*float(dt)-float(net.t))*b.second,namespace={})
        for group in groups:
            for name,ids in bundle.provenance['neuron_state_layout'][group.name].items():
                np.testing.assert_allclose(np.asarray(result['final_state'])[0,ids],group.variables[name].get_value(),rtol=7e-13,atol=8e-14)
        for name,ids in bundle.provenance['dynamic_state_layout'][syn.name].items():
            np.testing.assert_allclose(np.asarray(result['final_state'])[0,ids],syn.variables[name].get_value(),rtol=7e-13,atol=8e-14)
        for path in syn._pathways:
            ids=bundle.provenance['pathway_state_layout'][path.name]['delay']
            np.testing.assert_allclose(np.asarray(result['final_state'])[0,ids],np.asarray(path.delay[:]),rtol=7e-13,atol=1e-15)


def test_shared_delay_state_adjoint_is_accumulated_across_edges(engine):
    net,inp,layers,syn,x=scalar_model('v_post+=w+.1*delay/ms')
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    p=copy.deepcopy(bundle.plan);p['backend']='cpu'
    reference=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    for trainer in (actual,reference):
        trainer.step(x[None,:2],[0]);trainer.update_delays({syn.pre.name:.00064})
    result=actual.gradients(x[None,2:],[0],initial='carry')
    expected=reference.gradients(x[None,2:],[0],initial='carry')
    for key in ('final_state','spikes','initial_state_gradients'):
        np.testing.assert_allclose(result[key],expected[key],rtol=4e-4,atol=6e-5)
    for a,c in zip(result['gradients'],expected['gradients']):
        np.testing.assert_allclose(a,c,rtol=4e-4,atol=6e-5)
    [index]=bundle.provenance['pathway_state_layout'][syn.pre.name]['delay']
    assert abs(expected['initial_state_gradients'][0][index])>1e-10
