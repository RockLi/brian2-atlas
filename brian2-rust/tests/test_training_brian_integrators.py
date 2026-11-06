"""Brian forward oracle and independent RK surrogate finite differences."""
import copy
import os

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_training
from brian2_rust.training_brian import TrainingConversionError
from test_native_training import RUNNER
from test_training_brian import model as scalar_model
from test_training_brian_multistate import model
from test_native_training_gpu_mpi import compare


def bundle_model(method, **options):
    net,source,groups,x,dt=model()
    for group in groups:group.state_updater.method_choice=method
    bundle=lower_brian_training(net,input_group=source,layers=groups,
        trainable_neuron_parameters={g.name:['tau','theta','kick'] for g in groups},
        learning_rate=1e-8,**options)
    return bundle,x


@pytest.mark.parametrize('method',['rk2','rk4'])
@pytest.mark.parametrize('units',[False,True])
@pytest.mark.parametrize('scalar',[False,True])
def test_explicit_forward_matches_brian(method,units,scalar):
    if scalar:
        net,source,groups,_,x,dt=scalar_model(units=units)
    else:net,source,groups,x,dt=model(units=units)
    for g in groups:g.state_updater.method_choice=method
    bundle=lower_brian_training(net,input_group=source,layers=groups)
    before=[g.v[:].copy() for g in groups]
    native=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    result=native.evaluate(x[None],[0],initial=[bundle.initial_state])
    for g,v in zip(groups,before):np.testing.assert_array_equal(g.v[:],v)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*dt,namespace={})
    spikes=np.zeros((len(x),4))
    for l,m in enumerate(monitors):
        ticks=np.rint(np.asarray(m.t/b.second)/float(dt/b.second)).astype(int)
        spikes[ticks,2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    expected=np.concatenate([g.variables[n].get_value() for g,names in zip(groups,bundle.provenance['state_names']) for n in names])
    np.testing.assert_allclose(result['final_state'][0],expected,rtol=2e-12,atol=2e-15)
    assert bundle.provenance['integrators']==[method,method]


@pytest.mark.parametrize('resets',[('v=0','v-=theta'),('v=.4*v\nv+=.1*theta','v=.2*v+.3*theta')])
@pytest.mark.parametrize('methods',[('euler','euler'),('euler','rk4')])
def test_general_scalar_resets_and_mixed_integrators(resets,methods):
    net,source,groups,_,x,dt=scalar_model()
    for g,reset,method in zip(groups,resets,methods):
        g.event_codes['spike']=reset;g.state_updater.method_choice=method
    bundle=lower_brian_training(net,input_group=source,layers=groups,
        trainable_neuron_parameters={g.name:['tau','theta'] for g in groups})
    assert bundle.plan['schema']=='b2-state-training-plan-v4'
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0],initial=[bundle.initial_state])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*dt,namespace={})
    spikes=np.zeros((len(x),4))
    for l,m in enumerate(monitors):
        ticks=np.rint(np.asarray(m.t/b.second)/float(dt/b.second)).astype(int)
        spikes[ticks,2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],np.concatenate([g.v[:] for g in groups]),rtol=2e-13,atol=2e-15)


def oracle(bundle, weights, x, initial, method, anchors=None):
    """Independent textbook RK steps, with locally linearized hard spikes."""
    p=bundle.plan;dt=bundle.provenance['dt_seconds'];live=initial.copy()
    banks=[v['bank'] for v in bundle.provenance['bindings'] if v['kind']=='neuron']
    kick,tau,theta=np.asarray([weights[k] for k in banks]).T
    theta=np.repeat(theta,2);starts=[];pres=[];spikes=[]
    for t in range(len(x)):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:live=anchors[2][t].copy()
        starts.append(live.copy());u=live.copy()
        for l,(offset,count) in enumerate([(0,2),(4,3)]):
            z=live[offset:offset+count*2].reshape(count,2)
            def f(z):
                dz=[(-z[0]+.3*z[1]+(.1*z[2] if count==3 else 0))/tau[l],(.15*z[0]-.4*z[1])/tau[l]]
                if count==3:dz.append((.1*z[1]-.3*z[2])/tau[l])
                return np.array(dz)
            k1=f(z)
            if method=='rk2':next_z=z+dt*f(z+dt*k1/2)
            else:
                k2=f(z+dt*k1/2);k3=f(z+dt*k2/2);k4=f(z+dt*k3)
                next_z=z+dt*(k1+2*k2+2*k3+k4)/6
            u[offset:offset+count*2]=next_z.ravel()
        pre=u[[0,1,4,5]].copy();s=(pre>theta).astype(float);reset_s=s.copy()
        if anchors is not None:
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][t]-anchors[3]))**2
            s=anchors[1][t]+phi*(pre-theta-anchors[0][t]+anchors[3])
            reset_s=anchors[1][t] if p['detach_reset'] else s
        for projection,w in zip(p['projections'][:3],weights[:3]):
            src=projection['source_layer'];offset=[0,4][projection['target_layer']-1]
            inputs=x[t] if src==0 else s[2*(src-1):2*src]
            for i,j,k in zip(projection['sources'],projection['targets'],projection['parameter_ids']):u[offset+j]+=inputs[i]*w[k]
        reset=u.copy()
        for l,offset in enumerate([0,4]):
            reset[offset+2:offset+4]+=kick[l]+.1*u[offset:offset+2]
            reset[offset:offset+2]-=theta[2*l:2*l+2]
        reset[8:10]=.9*u[8:10]+.05*reset[6:8]
        gate=np.concatenate([reset_s[:2]]*2+[reset_s[2:]]*3)
        live=u+gate*(reset-u);pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,2:].mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,spikes,live,(np.array(pres),spikes,np.array(starts),theta)


@pytest.mark.parametrize('method',['rk2','rk4'])
@pytest.mark.parametrize('detach',[True,False])
@pytest.mark.parametrize('window',[None,3])
def test_rk_independent_surrogate_vjp(method,detach,window):
    bundle,x=bundle_model(method,detach_reset=detach,tbptt_window=window)
    w=bundle.weights;initial=np.array(bundle.initial_state)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=w).gradients(x[None],[0],initial=initial[None])
    loss,spikes,live,anchors=oracle(bundle,w,x,initial,method)
    assert result['loss']==pytest.approx(loss,abs=2e-15)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],live,atol=2e-14,rtol=2e-14)
    for bank,row in enumerate(w):
        for i,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3)
            plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[bank][i]+=eps;minus[bank][i]-=eps
            fd=(oracle(bundle,plus,x,initial,method,anchors)[0]-oracle(bundle,minus,x,initial,method,anchors)[0])/(2*eps)
            assert result['gradients'][bank][i]==pytest.approx(fd,abs=3e-6,rel=1e-4)
    for i in range(len(initial)):
        plus=initial.copy();minus=initial.copy();plus[i]+=1e-6;minus[i]-=1e-6
        fd=(oracle(bundle,w,x,plus,method,anchors)[0]-oracle(bundle,w,x,minus,method,anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][i]==pytest.approx(fd,abs=5e-8,rel=1e-4)


@pytest.mark.parametrize('method',['rk2','rk4'])
@pytest.mark.parametrize('backend,ranks',[('cpu',2),('metal',None),('metal',2),('cuda',None),('cuda',2)])
def test_rk_backends_and_checkpoint(method,backend,ranks,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    if backend=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('Metal required')
    if backend=='cuda' and os.environ.get('B2_TEST_CUDA_TRAIN')!='1':pytest.skip('CUDA required')
    bundle,x=bundle_model(method);cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    p=copy.deepcopy(bundle.plan);p['backend']=backend
    if ranks:p['mpi_ranks']=ranks
    target=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    for step in range(2):
        initial=[bundle.initial_state] if step==0 else 'carry'
        a=cpu.step(x[None],[0],initial=initial);c=target.step(x[None],[0],initial=initial)
        compare(a,c)
        for name in ['final_state','initial_state_gradients']:np.testing.assert_allclose(a[name],c[name],atol=6e-6,rtol=4e-4)
    path=tmp_path/'checkpoint';target.store(path)
    restored=NativeLIFTrainer(p,runner=RUNNER);restored.restore(path)
    assert restored.step(x[None],[0],initial='carry')==target.step(x[None],[0],initial='carry')


def test_replaced_integrator_is_rejected(monkeypatch):
    from brian2.stateupdaters.base import StateUpdateMethod
    net,source,groups,x,dt=model()
    for g in groups:g.state_updater.method_choice='rk2'
    monkeypatch.setitem(StateUpdateMethod.stateupdaters,'rk2',lambda *args: 'v = v')
    with pytest.raises(TrainingConversionError,match='built-in'):
        lower_brian_training(net,input_group=source,layers=groups)
