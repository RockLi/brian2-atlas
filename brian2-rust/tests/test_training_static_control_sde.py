"""Nonlinear bounded Python callbacks inside actual Brian SDE integrators."""
import ast
import copy
from unittest.mock import patch

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import model as base_model, lower, normal


@b.check_units(x=1, result=1)
def diffusion(x):
    y = 1.+x
    i = 0
    while i < 2:
        if i == 0:
            y += .1*x*x
        else:
            y = y+.1*x*x
        i += 1
    return y


def model(method, shared):
    # Build a real Brian network with its ordinary original array callback.
    # Restore the constructor before either the converter or Brian executes.
    constructor=b.NeuronGroup
    def make_group(n, equations, **options):
        equations=equations.replace('(v/unit+1)', 'curve(v/unit)').replace('(a/unit+1)', 'curve(a/unit)')
        options['namespace']={'curve':b.Function(diffusion)}
        return constructor(n, equations, **options)
    with patch.object(b, 'NeuronGroup', make_group):
        result=base_model(method, shared)
    # Prepare all objects with Brian itself without evolving state or draws.
    result[0].run(0*result[-1], namespace={})
    return result


def oracle(bundle,w,x,initial,anchors=None,sequence=0,start_tick=0):
    p=bundle.plan;dt=p['clock']['dt'];live=initial.copy();old=[];pres=[];spikes=[];params=[]
    for name in bundle.provenance['layer_names']:
        row={}
        for bind in bundle.provenance['bindings']:
            if bind['object']==name:row.update(zip(bind['variables'],w[bind['bank']]))
        params.append(row)
    theta=np.repeat([q['theta'] for q in params],2);volt=np.array([0,1,4,5])
    for tick in range(len(x)):
        t=p['clock']['origin']+(start_tick+tick)*dt
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:live=anchors[2][tick].copy()
        old.append(live.copy());u=live.copy()
        for l,q in enumerate(params):
            z=live[4*l:4*l+4].reshape(2,2);method=bundle.provenance['integrators'][l];streams=bundle.provenance['noise_names'][l]
            drift=np.array([(-z[0]+.3*z[1]+q['drive']*np.sin(t/q['tau']))/q['tau'],(q['gain']*z[0]-.4*z[1])/q['tau']])
            next_z=z+dt*drift
            for s,name in enumerate(streams):
                dw=np.sqrt(dt)*np.array([normal(p['seed'],sequence,0,l,j,start_tick+tick,s) for j in range(2)])
                mask=np.array([name in ('xi_v','xi_common'),name in ('xi_a','xi_common')])[:,None]
                def g(z):
                    if name=='xi_extra':return q['sigma']/np.sqrt(q['tau'])*np.array([.2*(1+z[1]+.2*z[1]*z[1]),.1*(1+z[0]+.2*z[0]*z[0])])
                    return mask*q['sigma']/np.sqrt(q['tau'])*(1+z+.2*z*z)
                base=g(z)
                if method=='euler':next_z=next_z+base*dw
                elif method=='heun':next_z=next_z+.5*dw*(base+g(z+base*dw))
                else:next_z=next_z+base*dw+(g(z+dt*drift+np.sqrt(dt)*base)-base)*dw**2/(2*np.sqrt(dt))
            u[4*l:4*l+4]=next_z.ravel()
        pre=u[volt].copy();s=(pre>theta).astype(float);gate=s.copy()
        if anchors is not None:
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][tick]-anchors[3]))**2
            s=anchors[1][tick]+phi*(pre-theta-anchors[0][tick]+anchors[3]);gate=anchors[1][tick] if p['detach_reset'] else s
        for proj,row in zip(p['projections'],w):
            src=proj['source_layer'];dst=proj['target_layer']-1;inputs=x[tick] if src==0 else s[2*(src-1):2*src]
            for i,j,k in zip(proj['sources'],proj['targets'],proj['parameter_ids']):u[4*dst+j]+=inputs[i]*row[k]
        for l,q in enumerate(params):
            z=u[4*l:4*l+4].copy();reset=z.copy();reset[2:]+=q['kick']+.1*z[:2];reset[:2]-=q['theta']
            u[4*l:4*l+4]+=np.tile(gate[2*l:2*l+2],2)*(reset-z)
        live=u;pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,2:].mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,spikes,live,(np.array(pres),spikes,np.array(old),theta)


def assert_original_brian_forward(net, groups, bundle, x, dt, result, sequence):
    net.run(0*dt, namespace={})
    order=[]
    for layer,group in enumerate(groups):
        for statement in ast.parse(group.state_updater.abstract_code).body:
            if isinstance(statement,ast.Assign) and any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='randn' for node in ast.walk(statement.value)):
                order.append((layer,bundle.provenance['noise_names'][layer].index(statement.targets[0].id)))
    draws=iter([np.array([normal(7123,sequence,0,layer,j,tick,stream) for j in range(2)])
                for tick in range(len(x)) for layer,stream in order])
    def randn(n):
        values=next(draws);assert len(values)==n;return values
    monitors=[b.SpikeMonitor(group) for group in groups];net.add(*monitors)
    with patch('numpy.random.randn', randn):net.run(len(x)*dt, namespace={})
    with pytest.raises(StopIteration):next(draws)
    expected=[];spikes=np.zeros((len(x),4))
    for layer,(group,monitor) in enumerate(zip(groups,monitors)):
        ticks=np.rint(np.asarray(monitor.t/b.second)/float(dt)).astype(int)
        spikes[ticks,2*layer+np.asarray(monitor.i)]=1
        expected.extend(group.variables['v'].get_value());expected.extend(group.variables['a'].get_value())
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],expected,atol=3e-5,rtol=3e-5)


@pytest.mark.parametrize('method,shared',[('heun',False),('heun',True),('heun','mixed'),('milstein',False)])
@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('detach,window',[(False,None),(True,None),(False,3)])
def test_sde_static_callback_physics_all_bank_initial_and_tbptt_vjps(engine,ranks,method,shared,detach,window):
    mpi(ranks)
    net,source,groups,x,dt=model(method,shared)
    bundle=lower(net,source,groups,backend=engine,mpi_ranks=ranks,detach_reset=detach,tbptt_window=window)
    weights=bundle.weights;initial=np.asarray(bundle.initial_state);sequence=9
    result=NativeLIFTrainer(bundle.plan,weights=weights,runner=RUNNER).gradients(
        x[None],[0],initial=initial[None],noise_sequence=sequence)
    loss,spikes,live,anchors=oracle(bundle,weights,x,initial,sequence=sequence)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],live,atol=3e-5,rtol=3e-5)
    for bank,row in enumerate(weights):
        for index,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3)
            plus=copy.deepcopy(weights);minus=copy.deepcopy(weights)
            plus[bank][index]+=eps;minus[bank][index]-=eps
            fd=(oracle(bundle,plus,x,initial,anchors,sequence=sequence)[0]-oracle(bundle,minus,x,initial,anchors,sequence=sequence)[0])/(2*eps)
            assert result['gradients'][bank][index]==pytest.approx(fd,rel=4e-4,abs=5e-5)
    for index in range(len(initial)):
        plus=initial.copy();minus=initial.copy();plus[index]+=1e-6;minus[index]-=1e-6
        fd=(oracle(bundle,weights,x,plus,anchors,sequence=sequence)[0]-oracle(bundle,weights,x,minus,anchors,sequence=sequence)[0])/2e-6
        assert result['initial_state_gradients'][0][index]==pytest.approx(fd,rel=4e-4,abs=5e-5)
    assert (result['gpu_dispatches']>0)==(engine!='cpu')
    if ranks is None and not detach and window is None:
        assert_original_brian_forward(net,groups,bundle,x,dt,result,sequence)
