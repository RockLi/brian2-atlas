"""Explicit continuous/summed draws against model formulas and compiled Brian."""
import copy
from unittest.mock import patch
import brian2 as b
import numpy as np
import pytest
from brian2_rust import lower_brian_dynamic_training,NativeLIFTrainer
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_uniform import uniform
from test_training_stochastic import normal


def model(warm=0,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='draw_input')
    a=b.NeuronGroup(2,'dv/dt=(.5-v)/ms:1',threshold='v>.6',reset='v-=.3',method='euler',dt=dt,name='draw_a')
    c=b.NeuronGroup(2,'dv/dt=(.5-v+q)/ms:1\nq:1',threshold='v>.6',reset='v-=.3',method='euler',dt=dt,name='draw_c')
    a.v=[.8,.7];c.v=[.65,.6]
    s=b.Synapses(a,c,'dh/dt=(w*rand()+.02*h*randn()-h)/ms:1 (clock-driven)\nq_post=w*h+.03*rand():1 (summed)\nw:1',
        on_pre='v_post+=.01*w',method='euler',dt=dt,name='draw_s')
    s.connect(i=[1,0,1,0],j=[0,1,1,0]);s.w=[.5,.6,.4,.7];s.h=[.1,.2,.3,.4]
    net=b.Network(inp,a,c,s)
    if warm:b.seed(17);net.run(warm*dt,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],detach_reset=False,**options)
    return net,[a,c],s,bundle


def oracle(bundle,weights=None,initial=None,anchors=None,length=8):
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    a=bundle.provenance['neuron_state_layout']['draw_a']['v'];layout=bundle.provenance['neuron_state_layout']['draw_c'];c=layout['v'];q=layout['q']
    h=bundle.provenance['dynamic_state_layout']['draw_s']['h']
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='draw_s');w=np.asarray(weights[bank])
    domain=bundle.provenance['scheduled_noise_domains']['draw_s_summed_variable_q_post']
    pre=[1,0,1,0];post=[0,1,1,0];before=[];margins=[];spikes=[];draws={'rand':[],'randn':[]}
    for tick in range(length):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        summed=np.array([uniform(p['seed'],9,0,domain,e,tick,0) for e in range(4)])
        # Brian's Euler generator emits .02*h*randn() before w*rand().
        n=np.array([normal(p['seed'],9,0,2,e,tick,0) for e in range(4)])
        u=np.array([uniform(p['seed'],9,0,2,e,tick,1) for e in range(4)])
        draws['rand'].extend([*summed,*u]);draws['randn'].extend(n)
        z[q]=0
        for e in range(4):z[q[post[e]]]+=w[e]*z[h[e]]+.03*summed[e]
        z[a]=.8*z[a]+.1;z[c]=.8*z[c]+.1+.2*z[q]
        z[h]=.8*z[h]+.2*w*u+.004*z[h]*n
        margin=z[a+c]-.6;event=(margin>0).astype(float)
        if anchors is not None:
            old=anchors['margins'][tick];event=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());spikes.append(event.copy())
        for e in range(4):z[c[post[e]]]+=.01*w[e]*event[pre[e]]
        z[a+c]-=.3*event
    spikes=np.asarray(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes,dict(before=before,margins=margins,draws=draws)


@pytest.mark.parametrize('warm',[0,2])
def test_synaptic_explicit_draws_compiled_brian(engine,warm):
    net,groups,s,bundle=model(warm,backend=engine);expected=oracle(bundle)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,8,2)),[0],noise_sequence=9)
    np.testing.assert_array_equal(result['spikes'][0],expected[2]);tol=5e-5 if engine!='cpu' else 4e-12
    np.testing.assert_allclose(result['final_state'][0],expected[1],rtol=tol,atol=tol*.01)
    net.run(0*b.ms,namespace={});draws=expected[3]['draws'];calls={k:0 for k in draws};device=b.get_device()
    device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    def sampler(name):
        def refill(n):
            assert n==20000 and calls[name]==0;calls[name]+=1
            values=np.zeros(n);values[:len(draws[name])]=draws[name];return values
        return refill
    with patch('numpy.random.rand',sampler('rand')),patch('numpy.random.randn',sampler('randn')):net.run(8*.2*b.ms,namespace={})
    for name in draws:assert calls[name]==1 and getattr(device,name+'_buffer_index')[0]==len(draws[name])
    device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    assert s.state_updater.codeobj.compiled_code['run'] is not None
    assert all(o.codeobj.compiled_code['run'] is not None for o in s.summed_updaters.values())
    for g in groups:
        for name,slots in bundle.provenance['neuron_state_layout'][g.name].items():np.testing.assert_allclose(expected[1][slots],g.variables[name].get_value(),rtol=4e-12,atol=4e-14)
    np.testing.assert_allclose(expected[1][bundle.provenance['dynamic_state_layout']['draw_s']['h']],s.h[:],rtol=4e-12,atol=4e-14)


@pytest.mark.parametrize('window',[None,3])
def test_synaptic_explicit_draws_derivatives(engine,window):
    _,_,_,bundle=model(backend=engine,tbptt_window=window);x=np.zeros((1,8,2));eps=1e-6
    expected=oracle(bundle);anchors=expected[3];tol=3e-3 if engine!='cpu' else 8e-7
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    np.testing.assert_allclose(result['loss'],expected[0],rtol=tol);np.testing.assert_array_equal(result['spikes'][0],expected[2])
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            a=copy.deepcopy(bundle.weights);c=copy.deepcopy(bundle.weights);a[bank][k]+=eps;c[bank][k]-=eps
            fd=(oracle(bundle,weights=a,anchors=anchors)[0]-oracle(bundle,weights=c,anchors=anchors)[0])/(2*eps)
            np.testing.assert_allclose(result['gradients'][bank][k],fd,rtol=tol,atol=tol*.01)
    for k in range(len(bundle.initial_state)):
        a=np.array(bundle.initial_state);c=a.copy();a[k]+=eps;c[k]-=eps
        fd=(oracle(bundle,initial=a,anchors=anchors)[0]-oracle(bundle,initial=c,anchors=anchors)[0])/(2*eps)
        np.testing.assert_allclose(result['initial_state_gradients'][0][k],fd,rtol=tol,atol=tol*.01)
