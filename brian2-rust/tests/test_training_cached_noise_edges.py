"""Delayed and empty cached synapses, scheduled placement and rejection bounds."""
import copy
from unittest.mock import patch
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training,TrainingConversionError
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_neuron_noise import cython_cache
from test_training_uniform import uniform
from test_training_cached_noise import model


@pytest.mark.parametrize('warm',[0,1])
@pytest.mark.parametrize('empty',[False,True])
def test_cached_delayed_delivery_uses_current_tick(engine,warm,empty,tmp_path):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    pattern=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,0],[0,0]],float);ticks,ids=np.nonzero(pattern)
    source=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='cache_delay_input')
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>10',reset='v=0',method='euler',dt=dt,name=f'cache_delay_{k}') for k in range(2)]
    syn=b.Synapses(source,groups[0],'w:1\nr=rand():1 (constant over dt)',on_pre='v_post+=w*r',dt=dt,name='cache_delay_s')
    if empty:syn.connect(False)
    else:syn.connect(j='i');syn.w=[.3,.4];syn.delay=2*dt
    net=b.Network(source,*groups,syn)
    if warm:b.seed(23);net.run(warm*dt,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine)
    x=pattern[warm:warm+6];bundle.plan['trainable']=[False]*len(bundle.weights)
    domain=bundle.provenance['regular_runner_layout'][syn.subexpression_updater.name]['noise_domain']
    expected=np.asarray(groups[0].v).copy();draws=[];last=[]
    for t in range(6):
        expected*=.8;last=[uniform(bundle.plan['seed'],17,0,domain,j,t,0) for j in range(0 if empty else 2)];draws.extend(last)
        old=t+warm-2
        if not empty and old>=0:expected+=pattern[old]*[.3,.4]*last
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    trainer.step(x[None,:3],[0],**({} if empty else dict(noise_sequence=17)));path=tmp_path/'delay.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER);restored.restore(path);actual=restored.step(x[None,3:],[0],initial='carry')
    slots=bundle.provenance['neuron_state_layout'][groups[0].name]['v'];tol=4e-5 if engine!='cpu' else 4e-12
    np.testing.assert_allclose(np.asarray(actual['final_state'])[0,slots],expected,rtol=tol,atol=tol*.01)
    net.run(0*b.ms,namespace={});device=b.get_device();device.rand_buffer_index[:]=0;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);values=np.zeros(n);values[:len(draws)]=draws;return values
    with patch('numpy.random.rand',refill):net.run(6*dt,namespace={})
    assert len(calls)==int(not empty) and device.rand_buffer_index[0]==len(draws);device.rand_buffer_index[:]=0
    assert syn.subexpression_updater.codeobj.compiled_code['run'] is not None
    np.testing.assert_allclose(groups[0].v[:],expected,rtol=4e-12,atol=4e-14)
    np.testing.assert_allclose(syn.r[:],last,rtol=4e-12,atol=4e-14)


@pytest.mark.parametrize('change,match',[('clock','clock'),('inactive','inactive'),('override','runner'),('static','runner')])
def test_cached_updater_contract(change,match):
    if change=='clock':
        # Distinct updater clocks are supported by the itinerary. Verify actual
        # cached Poisson consumption at thresholds/resets, rather than expecting
        # the obsolete rejection from before multi-clock lowering.
        from test_training_poisson_event_replay import assert_cached_poisson_clock
        assert_cached_poisson_clock('cpu')
        return
    net,groups,syn,_=model(ref=False);source=next(o for o in net.objects if o.name=='cached_input');runner=groups[0].subexpression_updater
    if change=='inactive':runner.active=False
    elif change=='override':runner.update_abstract_code=lambda:None
    with pytest.raises(TrainingConversionError,match=match):
        if change=='static':
            from brian2_rust import lower_brian_training
            # Remove synapses so the static path reaches its neuron runner check.
            net.remove(syn);lower_brian_training(net,input_group=source,layers=groups)
        else:lower_brian_dynamic_training(net,input_group=source,layers=groups)


@pytest.mark.parametrize('when',['before_start','after_groups','end'])
def test_cached_updater_schedule_matches_cython(engine,when):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='schedule_input');groups=[]
    for k in range(2):
        g=b.NeuronGroup(2,'dv/dt=-v/ms:1\nq=rand()+.1*v:1 (constant over dt)',threshold='v>.6+.1*q',reset='v-=.3+.02*q',method='euler',dt=dt,name=f'schedule_{k}')
        g.v=[.9,.8];g.q=[.2,.3];g.subexpression_updater.when=when;groups.append(g)
    net=b.Network(source,*groups);bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,detach_reset=False)
    v=np.array([[.9,.8],[.9,.8]]);q=np.array([[.2,.3],[.2,.3]]);draws=[];spikes=[]
    def refresh(t):
        for k,g in enumerate(groups):
            domain=bundle.provenance['regular_runner_layout'][g.subexpression_updater.name]['noise_domain']
            values=[uniform(bundle.plan['seed'],9,0,domain,j,t,0) for j in range(2)];draws.extend(values);q[k]=values+.1*v[k]
    for t in range(6):
        if when=='before_start':refresh(t)
        v*=.8
        if when=='after_groups':refresh(t)
        event=v>.6+.1*q;spikes.append(event.ravel().copy());v-=event*(.3+.02*q)
        if when=='end':refresh(t)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,6,2)),[0],noise_sequence=9)
    np.testing.assert_array_equal(result['spikes'][0],spikes);tol=4e-5 if engine!='cpu' else 4e-12
    for k,g in enumerate(groups):
        layout=bundle.provenance['neuron_state_layout'][g.name]
        for name,expected in [('v',v[k]),('q',q[k])]:np.testing.assert_allclose(np.asarray(result['final_state'])[0,layout[name]],expected,rtol=tol,atol=tol*.01)
    net.run(0*b.ms,namespace={});device=b.get_device();device.rand_buffer_index[:]=0;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);out=np.zeros(n);out[:len(draws)]=draws;return out
    with patch('numpy.random.rand',refill):net.run(6*dt,namespace={})
    assert len(calls)==1 and device.rand_buffer_index[0]==len(draws);device.rand_buffer_index[:]=0
    for k,g in enumerate(groups):
        assert g.subexpression_updater.codeobj.compiled_code['run'] is not None
        np.testing.assert_allclose(g.v[:],v[k],rtol=4e-12,atol=4e-14);np.testing.assert_allclose(g.q[:],q[k],rtol=4e-12,atol=4e-14)


@pytest.mark.parametrize('kind',['boolean','duration'])
def test_cached_random_refractory_matches_cython(engine,kind):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='cached_ref_input');groups=[]
    for k in range(2):
        g=b.NeuronGroup(2,'dv/dt=(.7-v)/ms:1 (unless refractory)\nq=rand():1 (constant over dt)',threshold='v>.6',reset='v-=.3',method='euler',dt=dt,
            refractory='q>.45' if kind=='boolean' else '(1+int(3*q))*dt',name=f'cached_ref_{k}')
        g.v=[.8,.9];groups.append(g)
    net=b.Network(source,*groups);bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine)
    v=np.array([[.8,.9],[.8,.9]]);active=np.ones((2,2),bool);last=np.full((2,2),-1000000);draws=[];spikes=[]
    for t in range(8):
        q=[]
        for k,g in enumerate(groups):
            domain=bundle.provenance['regular_runner_layout'][g.subexpression_updater.name]['noise_domain']
            u=[uniform(bundle.plan['seed'],9,0,domain,j,t,0) for j in range(2)];q.append(u);draws.extend(u)
        q=np.asarray(q);active=(active | (q<=.45)) if kind=='boolean' else (t-last>=1+(3*q).astype(int))
        v=np.where(active,.8*v+.14,v);event=(v>.6)&active;spikes.append(event.ravel());v-=.3*event;last[event]=t;active[event]=False
    actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,8,2)),[0],noise_sequence=9)
    np.testing.assert_array_equal(actual['spikes'][0],spikes);tol=4e-5 if engine!='cpu' else 4e-12
    for k,g in enumerate(groups):np.testing.assert_allclose(np.asarray(actual['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],v[k],rtol=tol,atol=tol*.01)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(0*b.ms,namespace={});device=b.get_device();device.rand_buffer_index[:]=0;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);out=np.zeros(n);out[:len(draws)]=draws;return out
    with patch('numpy.random.rand',refill):net.run(8*dt,namespace={})
    assert len(calls)==1 and device.rand_buffer_index[0]==len(draws);device.rand_buffer_index[:]=0
    events=np.zeros((8,4))
    for k,(g,m) in enumerate(zip(groups,monitors)):
        assert g.subexpression_updater.codeobj.compiled_code['run'] is not None
        events[np.rint(np.asarray(m.t/b.second)/.0002).astype(int),2*k+np.asarray(m.i)]=1
        np.testing.assert_allclose(g.v[:],v[k],rtol=4e-12,atol=4e-14)
    np.testing.assert_array_equal(events,spikes)


def external_input_bundle(cached,backend='cpu',read_cache=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    if cached:
        source=b.NeuronGroup(2,'dv/dt=(-v+q)/ms:1\nq=rand():1 (constant over dt)',threshold='v>1',reset='v=0',method='euler',dt=dt,name='external_input')
    else:source=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='external_input')
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.4',reset='v-=.3',method='euler',dt=dt,name=f'external_layer_{k}') for k in range(2)]
    synapses=[]
    for k,(a,c) in enumerate(zip([source,*groups],groups)):
        syn=b.Synapses(a,c,'w:1',on_pre='v_post+=q_pre' if read_cache and k==0 else 'v_post+=w',dt=dt,name=f'external_syn_{k}');syn.connect(j='i');syn.w=.4;synapses.append(syn)
    return lower_brian_dynamic_training(b.Network(source,*groups,*synapses),input_group=source,layers=groups,backend=backend,detach_reset=False)


def test_external_input_internal_cache_is_not_lowered(engine):
    actual=external_input_bundle(True,engine);reference=external_input_bundle(False,engine)
    assert actual.plan==reference.plan and actual.weights==reference.weights
    assert not actual.provenance['regular_runner_layout'] and actual.plan.get('noise_streams') is None
    x=np.array([[[1,0],[0,1],[1,1],[0,0],[1,0],[0,1]]],float)
    a=NativeLIFTrainer(actual.plan,weights=actual.weights,runner=RUNNER).gradients(x,[0])
    c=NativeLIFTrainer(reference.plan,weights=reference.weights,runner=RUNNER).gradients(x,[0])
    for key in ('loss','spikes','final_state','initial_state_gradients'):np.testing.assert_array_equal(a[key],c[key])
    for av,cv in zip(a['gradients'],c['gradients']):np.testing.assert_array_equal(av,cv)
    if engine!='cpu':assert a['gpu_dispatches']>0


def test_external_mutable_cache_cannot_be_frozen_as_constant():
    with pytest.raises(TrainingConversionError,match='runtime state input'):
        external_input_bundle(True,read_cache=True)
