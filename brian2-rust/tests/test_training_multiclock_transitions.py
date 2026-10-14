"""Warm dt changes and combined asynchronous synaptic dynamics."""
import copy
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, TrainingConversionError, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_multiclock_frontend import model, check_brian, oracle


def snapshot(net,syn):
    clocks={obj.clock for obj in net.sorted_objects}
    return [(c,c._old_dt,c.variables['t'].get_value().copy(),c.variables['timestep'].get_value().copy()) for c in clocks],copy.deepcopy([p.queue._full_state() for p in syn._pathways])


def unchanged(before,net,syn):
    for c,old,t,i in before[0]:
        assert c._old_dt==old
        np.testing.assert_array_equal(c.variables['t'].get_value(),t)
        np.testing.assert_array_equal(c.variables['timestep'].get_value(),i)
    assert [p.queue._full_state() for p in syn._pathways]==before[1]


@pytest.mark.parametrize('old,new', [((.1,.3),(.2,.6)),((.2,.6),(.1,.3))])
@pytest.mark.parametrize('warm', [.6,.61])
@pytest.mark.parametrize('refractory', [False,True,'(.2+.1*v)*ms'])
def test_warm_source_dt_change_matches_brian(engine,old,new,warm,refractory):
    net,inp,groups,syn,_=model(old,warm,refractory=refractory,event_driven=True,runtime_delay=True)
    for g,dt in zip(groups,new):g.clock.dt=dt*b.ms
    before=snapshot(net,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,backend=engine)
    unchanged(before,net,syn)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,8,1)),[0])
    net.run((bundle.plan['clock']['origin']+8*.0002-float(net.t))*b.second,namespace={})
    check_brian(result,bundle,groups,syn,monitors)


def test_warm_input_dt_change_matches_brian(engine):
    net,inp,groups,syn,_=model((.1,.3),.61,refractory=True)
    inp.clock.dt=.1*b.ms
    before=snapshot(net,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,backend=engine)
    unchanged(before,net,syn)
    assert bundle.plan['clock']['origin']==pytest.approx(.0007)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,8,1)),[0])
    net.run((bundle.plan['clock']['origin']+8*.0001-float(net.t))*b.second,namespace={})
    check_brian(result,bundle,groups,syn,monitors)


def test_invalid_warm_dt_change_rejected_without_mutation():
    net,inp,groups,syn,_=model((.1,.3),.6)
    # The old pending time .6 is also the ceil tick for the new clock, but
    # Brian check_dt compares nearest ticks and rejects this transition.
    groups[0].clock.dt=.4*b.ms
    before=snapshot(net,syn)
    with pytest.raises(TrainingConversionError,match='Cannot set dt'):
        lower_brian_dynamic_training(net,input_group=inp,layers=groups)
    unchanged(before,net,syn)


@pytest.mark.parametrize('dts,third', [((.40003,.4),.2),((.99999,.2),.49998)])
@pytest.mark.parametrize('warm', [0.,.65])
def test_nearby_clock_coalescing_spikes_and_delays(engine,dts,third,warm):
    net,inp,groups,syn,_=model(dts,event_driven=True)
    # A read-only monitor contributes a third clock to the scheduler, which
    # can coalesce a source tick before another neuron's pending threshold.
    net.add(b.StateMonitor(groups[0],'v',record=True,dt=third*b.ms,name='transition_third'))
    if warm:net.run(warm*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,backend=engine)
    clock=bundle.plan['dynamic']['clocks'];order=[]
    for c in {o.clock for o in net.sorted_objects}:
        k=clock['dts'].index(float(c.dt_))
        if k not in order:order.append(k)
    assert clock['order']==order
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,12,1)),[0])
    net.run((bundle.plan['clock']['origin']+12*.0002-float(net.t))*b.second,namespace={})
    check_brian(result,bundle,groups,syn,monitors)


def test_new_monitor_clock_after_warm_snapshot(engine):
    net,inp,groups,syn,_=model((.15,.37),.65)
    net.add(b.StateMonitor(groups[0],'v',record=True,dt=.27*b.ms,name='late_monitor'))
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,backend=engine)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,6,1)),[0])
    net.run((bundle.plan['clock']['origin']+6*.0002-float(net.t))*b.second,namespace={})
    check_brian(result,bundle,groups,syn,monitors)


@pytest.mark.parametrize('old,new', [((.1,.3),(.2,.6)),((.2,.6),(.1,.3))])
@pytest.mark.parametrize('window', [None,2])
def test_retimed_stochastic_paths_independent_full_vjp(engine,old,new,window):
    net,inp,groups,syn,_=model(old,.61,noisy=True,event_noise=True)
    for g,dt in zip(groups,new):g.clock.dt=dt*b.ms
    before=snapshot(net,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,
        trainable_neuron_parameters={g.name:['drive'] for g in groups},backend=engine,tbptt_window=window)
    unchanged(before,net,syn)
    assert any(path['pending'] for path in bundle.plan['dynamic']['delay_layout']['paths'])
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],noise_sequence=17)
    loss,z,counts,events,anchors=oracle(bundle,new);tol=5e-3 if engine!='cpu' else 4e-6
    np.testing.assert_allclose(result['final_state'][0],z,rtol=tol*.1,atol=3e-6)
    np.testing.assert_array_equal(result['spikes'][0],counts);np.testing.assert_array_equal(result['event_visits']['spikes'][0],events)
    assert result['loss']==pytest.approx(loss,rel=tol)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(oracle(bundle,new,w=hi,anchors=anchors)[0]-oracle(bundle,new,w=lo,anchors=anchors)[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=tol,abs=tol*.01)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j]:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(oracle(bundle,new,initial=hi,anchors=anchors)[0]-oracle(bundle,new,initial=lo,anchors=anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=tol,abs=tol*.01)
