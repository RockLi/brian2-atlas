"""Undefined false-event Poisson domains fail atomically; detached resets work."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def selected_rate_factory(array):
    def f(x,mean):
        captured=array
        captured*=.8
        draw=np.random.poisson(x+mean,size=len(x))
        return x+.05*draw
    return f


def model(backend,detached):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    storage=b.NeuronGroup(4,'dv/dt=0/second:1\na:1',threshold='v>100',reset='v=0',method='euler',dt=dt);storage.a=[.1,.2,.3,.4]
    g=b.NeuronGroup(2,'dv/dt=0/second:1\nmean:1 (constant,shared)',threshold='v>.5',reset='v=f(v,mean)-.5',method='euler',dt=dt);g.v=[-.01,.7];g.mean=.1
    g.namespace['f']=b.Function(selected_rate_factory(storage.variables['a'].get_value()),arg_units=[1,1],arg_types=['float','float'],return_unit=1,stateless=False)
    net=b.Network(source,storage,g);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[storage,g],backend=backend,mpi_ranks=2,detach_reset=detached,trainable_neuron_parameters={g.name:['mean']})
    return net,storage,g,dt,bundle


def state(trainer):
    return copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))


def test_selected_poisson_candidate_bad_rate_atomic(engine,tmp_path):
    mpi(2);_,_,g,_,bundle=model(engine,False);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    assert any(a.get('poisson_candidate') for a in p['dynamic']['actions'])
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER,request_timeout=300);x=np.zeros((1,1,1));trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER,request_timeout=300);trainer.restore(path)
    cells=bundle.provenance['neuron_state_layout'][g.name]['v'];trainer.neuron_state[0][cells[0]]=-.6;trainer.neuron_state[0][cells[1]]=.7
    before=state(trainer)
    # Actual compact rate is +.8; the false-event alternate compact rate is
    # -.5 at the same ordinal as the real observation. A real-cache hit must
    # not suppress validation of the candidate's own rate.
    for _ in range(2):
        with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
        assert state(trainer)==before
    trainer.neuron_state[0][cells[0]]=-.01
    out=trainer.step(x,[0],initial='carry');assert trainer.elapsed_ticks==before[4]+1
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def test_selected_poisson_detached_reset_ignores_false_rate(engine,monkeypatch):
    mpi(2);net,storage,g,dt,bundle=model(engine,True);cells=bundle.provenance['neuron_state_layout'][g.name]['v']
    bundle.initial_state[cells[0]]=-.6;bundle.plan['dynamic']['initial'][cells[0]]=-.6;g.v[0]=-.6
    assert not any(a.get('poisson_candidate') for a in bundle.plan['dynamic']['actions'])
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)
    count_value=next(entry['count'] for entry in out['poisson_state']['entries'])
    calls=[]
    def sample(rate,*,size):
        np.testing.assert_allclose(rate,[.8]);assert size==1;calls.append(True);return np.array([count_value],dtype=np.int64)
    monkeypatch.setattr(np.random,'poisson',sample);net.run(dt,namespace={});assert calls==[True]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.v[:],rtol=8e-5,atol=8e-6)
    a=bundle.provenance['neuron_state_layout'][storage.name]['a'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,a],storage.a[:],rtol=8e-5,atol=8e-6)
    assert len(out['poisson_state']['entries'])==1
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
