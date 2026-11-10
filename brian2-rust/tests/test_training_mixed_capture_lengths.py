"""Independent differently sized event captures retain NumPy batch/FIFO semantics."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def independent_capture(a,b):
    def f(x):
        first=a;second=b
        first*=.8;second*=.6;x*=.7
        return x
    return f


def singleton_capture(a,b):
    def f(x):
        first=a;second=b
        first*=.8;second+=first;x*=.7
        return x
    return f


def incompatible_capture(a,b):
    def f(x):
        first=a;second=b
        first+=second;x*=.7
        return x
    return f


def readonly_capture(a,b):
    def f(x):
        first=a
        first*=.8;ignored=1./b;x*=.7
        return x
    return f


def model(mode,delay,ranks,backend,kind='independent'):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0],np.array([0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.217,.719];g.u=[.017,.031]
    code='h=f('+('v_post' if mode=='scalar' else 'h')+')'+(';u_post+=gain*h' if mode!='array' else '')
    if mode=='scalar':code='h=f(v_post);v_post+=gain*h'
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=code,dt=dt)
    syn.connect(i=[0,1],j=[0,1] if mode=='array' else [0,0]);syn.h=[.113,.173];syn.gain=[.11,.19];syn.delay=delay*dt
    a=np.array([.23]) if kind=='singleton' else g.variables['v'].get_value()
    c=np.array([.29,.37,.43])
    if kind=='readonly':c.flags.writeable=False
    function={'independent':independent_capture,'singleton':singleton_capture,'readonly':readonly_capture}[kind]
    syn.namespace['f']=b.Function(function(a,c),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    x=np.zeros((1,4,2));x[0,0,:]=1;x[0,2,0]=1
    return net,g,syn,a,c,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('kind',['independent','singleton','readonly'])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_mixed_capture_original_carry_restore(engine,mode,kind,delay,ranks,tmp_path):
    mpi(ranks);net,g,syn,a,c,dt,bundle,x=model(mode,delay,ranks,engine,kind)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=trainer.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        for row in bundle.provenance['mutable_capture_layout']:
            expected=a if row['shape']==list(a.shape) else c
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,row['cells']],expected,rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(trainer.plan,runner=RUNNER);trainer.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('sizes',[(2,3),(1,3)])
def test_incompatible_capture_shape_refused_without_executing(sizes):
    from brian2_rust.training_effects import lower_state_effect_function,mutated_state_effect_captures
    a=np.ones(sizes[0]);c=np.ones(sizes[1]);f=b.Function(incompatible_capture(a,c),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    with pytest.raises(ValueError,match='shapes|different shape'):
        mutated_state_effect_captures(lower_state_effect_function(f))
    np.testing.assert_array_equal(a,1);np.testing.assert_array_equal(c,1)
