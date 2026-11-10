"""Owned floating array mutations preserve aliases, copies and native VJPs."""
import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from brian2_rust.training_functions import lower_pure_function
from test_native_training import RUNNER
from test_training_integer_ir import engine
import test_training_pure_functions as callbacks
import test_training_pure_returns as returns
import test_training_static_control_sde as static_sde
import test_training_dynamic_static_sde as dynamic_sde


@b.check_units(x=1,result=1)
def diffusion(x):
    y = np.array(1.+x,dtype=np.float64)
    saved = y
    y += .2*x*x
    return saved


@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('method,shared',[('heun',False),('heun',True),('heun','mixed'),('milstein',False)])
@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('detach,window',[(False,None),(True,None),(False,3)])
def test_owned_array_sde_all_bank_initial_and_tbptt_vjps(
        engine,dynamic,method,shared,ranks,detach,window,monkeypatch):
    monkeypatch.setattr(static_sde,'diffusion',diffusion)
    verify=(dynamic_sde.test_dynamic_sde_callback_physics_all_banks_and_initial_vjps
            if dynamic else static_sde.test_sde_static_callback_physics_all_bank_initial_and_tbptt_vjps)
    verify(engine,ranks,method,shared,detach,window)


def alias_decay(x):
    y = np.array(x,dtype=float)
    saved = y
    y *= .8
    y += .01*np.sin(x)
    return saved


def copy_and_rebind(x):
    y = np.array(x,dtype=np.float64,copy=True)
    saved = y
    separate = np.array(saved,dtype=float)
    y *= .8
    y = np.array(x,dtype=float)
    y += .1*x
    return .5*saved+.1*separate+.1*y


def scalar_copy(x):
    y = np.array(.6,dtype=float)
    saved = y
    y *= .8
    return saved+.1*x


@pytest.mark.parametrize('function',[alias_decay,copy_and_rebind,scalar_copy])
@pytest.mark.parametrize('position',['pre','post','regular','synaptic_ode'])
def test_owned_array_aliases_original_brian_physics(engine,function,position,monkeypatch):
    monkeypatch.setattr(callbacks,'decay',b.check_units(x=1,result=1)(function))
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine,position)


def noisy_update(voltage,gate,gain,draw):
    y = np.array(voltage,dtype=float)
    saved = y
    y *= .8
    y += gain*gate+.05*gate*draw
    return saved


@pytest.mark.parametrize('ranks',[None,2,8])
def test_owned_array_noisy_all_bank_and_physical_initial_vjps(engine,ranks,monkeypatch):
    monkeypatch.setattr(returns,'noisy_return',noisy_update)
    returns.test_early_return_noisy_all_bank_and_physical_initial_vjps(engine,ranks)


def test_owned_array_scalar_actual_brian_argument(engine):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.25*b.ms
    source=b.SpikeGeneratorGroup(1,[0],np.array([0])*dt,dt=dt)
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
            for _ in range(2)]
    for g in groups:g.v=[.3,.8]
    curve=b.Function(b.check_units(x=1,result=1)(alias_decay))
    groups[1].namespace['curve']=curve
    groups[1].run_regularly('v=curve(.6)+.1*v',dt=dt)
    syn=b.Synapses(source,groups[0],'w:1',on_pre='v_post+=w');syn.connect();syn.w=.1
    net=b.Network(source,*groups,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(
        np.array([[[1.],[0.],[0.]]]),[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(3*dt,namespace={})
    spikes=np.zeros((3,4))
    for layer,monitor in enumerate(monitors):
        ticks=np.rint(np.asarray(monitor.t/b.second)/float(dt/b.second)).astype(int)
        spikes[ticks,2*layer+np.asarray(monitor.i)]=1.
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    for g in groups:
        slots=bundle.provenance['neuron_state_layout'][g.name]['v']
        np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],g.v[:],rtol=2e-5,atol=3e-6)
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


def borrowed_copy(x):
    y = np.array(x,dtype=float,copy=False)
    y *= .8
    return y


def unknown_dtype(x):
    y = np.array(x)
    y *= .8
    return y


def narrow_dtype(x):
    y = np.array(x,dtype=np.float32)
    y *= .8
    return y


def asarray_alias(x):
    y = np.asarray(x,dtype=float)
    y *= .8
    return y


def borrowed_write(x):
    y = np.array(x,dtype=float)
    x *= .8
    return y


def helper_alias(x):
    y = np.array(x,dtype=float)
    saved = helper(y)
    y *= .8
    return saved


def helper(x):return x


@pytest.mark.parametrize('function',[borrowed_copy,unknown_dtype,narrow_dtype,asarray_alias,borrowed_write,helper_alias])
def test_unsafe_or_unspecified_array_ownership_is_refused(function):
    with pytest.raises(ValueError):lower_pure_function(function)


def test_replaced_numpy_array_is_not_executed(monkeypatch):
    def opaque(*args,**kwargs):raise AssertionError('replacement constructor executed')
    monkeypatch.setattr(np,'array',opaque)
    with pytest.raises(ValueError):lower_pure_function(alias_decay)
