"""Heterogeneous neuron parameters: Brian forward and independent VJP oracles."""
import copy
import json
import os
import subprocess
import sys

import brian2 as b
from brian2.core.functions import timestep
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_training
from brian2_rust.training_brian import TrainingConversionError
from test_native_training import RUNNER
from test_native_training_gpu_mpi import compare


def model(method='euler',units=False,refractory=True,train=True,sizes=(3,2)):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.2*b.ms;scale=b.mV if units else 1.;dim='volt' if units else '1'
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[1,1],[0,1],[1,0],[0,0],[1,1],[1,0],[1,1]],float)
    ticks,ids=np.nonzero(x);source=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='het_input')
    groups=[]
    for l,n in enumerate(sizes):
        eq=f'''dv/dt=(-v+.3*a+bias+drive*i/N)/tau : {dim}{' (unless refractory)' if refractory else ''}
        da/dt=(gain*v-.4*a)/tau : {dim}
        tau : second (constant)
        theta : {dim} (constant)
        kick : {dim} (constant)
        bias : {dim} (constant)
        gain : 1 (constant,shared)
        drive : {dim} (constant,shared)'''
        g=b.NeuronGroup(n,eq,threshold='v>theta',reset='a+=kick+.1*v\nv-=theta',
                        method=method,dt=dt,refractory=3*dt if refractory else False,name=f'het_layer_{l}')
        g.tau=np.linspace(1.,1.3,n)*(1+l*.1)*b.ms;g.theta=np.linspace(1.0625,1.3125,n)*scale
        g.kick=np.linspace(.05,.08,n)*scale;g.bias=np.linspace(.01,.03,n)*scale;g.gain=.15;g.drive=.02*scale
        g.v=np.linspace(.2,1.8,n)*scale;g.a=np.linspace(.1,.3,n)*scale;groups.append(g)
    synapses=[]
    for q,(src,dst) in enumerate([(source,groups[0]),(groups[0],groups[1]),(groups[1],groups[0])]):
        syn=b.Synapses(src,dst,'w:'+dim,on_pre='v_post+=w',name=f'het_syn_{q}')
        i,j=np.indices((len(src),len(dst)));i=i.ravel()[::-1];j=j.ravel()[::-1]
        syn.connect(i=i,j=j);syn.w=(.4+.3*((i+2*j+q)%4))*scale;synapses.append(syn)
    net=b.Network(source,*groups,*synapses)
    requested={g.name:['tau','theta','kick','gain'] for g in groups} if train else {}
    return net,source,groups,x,dt,requested


def lower(net,source,groups,requested,**options):
    return lower_brian_training(net,input_group=source,layers=groups,trainable_neuron_parameters=requested,
                                learning_rate=1e-9,**options)


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('units,warmup',[(False,0),(True,3)])
@pytest.mark.parametrize('refractory',[False,True])
@pytest.mark.parametrize('train',[False,True])
def test_heterogeneous_forward_matches_brian(method,units,warmup,refractory,train):
    net,source,groups,x,dt,requested=model(method,units,refractory,train)
    if warmup:net.run(warmup*dt,namespace={})
    bundle=lower(net,source,groups,requested)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None,warmup:],[0],initial=[bundle.initial_state])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run((len(x)-warmup)*dt,namespace={})
    spikes=np.zeros((len(x)-warmup,sum(map(len,groups))));expected=[];offset=0
    for g,m in zip(groups,monitors):
        ticks=np.rint(np.asarray(m.t/b.second)/float(dt/b.second)).astype(int)-warmup
        spikes[ticks,offset+np.asarray(m.i)]=1;offset+=len(g)
        expected.extend(g.variables['v'].get_value());expected.extend(g.variables['a'].get_value())
        if refractory:
            elapsed=timestep(float(g.clock.variables['t'].get_value()[0])-g.variables['lastspike'].get_value(),float(dt))
            expected.extend(np.maximum(3-elapsed,0))
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],expected,atol=3e-15,rtol=3e-12)
    assert bundle.plan['threshold_per_neuron']==[True,True]


def oracle(bundle,w,x,initial,anchors=None):
    p=bundle.plan;dt=bundle.provenance['dt_seconds'];sizes=p['sizes'][1:];starts=np.cumsum([0,*[3*n for n in sizes]])
    neurons=np.cumsum([0,*sizes]);params=[]
    for name in bundle.provenance['layer_names']:
        row={}
        for binding in bundle.provenance['bindings']:
            if binding['object']==name:
                if binding['kind']=='neuron_array':row[binding['variables'][0]]=np.array(w[binding['bank']])
                else:row.update(zip(binding['variables'],w[binding['bank']]))
        params.append(row)
    theta=np.concatenate([q['theta'] for q in params]);live=initial.copy();old=[];pres=[];spikes=[];activities=[]
    volt=np.concatenate([np.arange(starts[l],starts[l]+n) for l,n in enumerate(sizes)])
    for t in range(len(x)):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:live=anchors[2][t].copy()
        old.append(live.copy());active=np.concatenate([live[starts[l]+2*n:starts[l]+3*n]==0 for l,n in enumerate(sizes)])
        if anchors is not None:active=anchors[4][t]
        activities.append(active.copy());u=live.copy()
        for l,n in enumerate(sizes):
            at=starts[l];z=live[at:at+2*n].reshape(2,n);q=params[l]
            def f(z):
                return np.array([(-z[0]+.3*z[1]+q['bias']+.02*np.arange(n)/n)/q['tau']*active[neurons[l]:neurons[l+1]],
                                 (q['gain']*z[0]-.4*z[1])/q['tau']])
            k1=f(z);method=bundle.provenance['integrators'][l]
            if method=='euler':next_z=z+dt*k1
            elif method=='rk2':next_z=z+dt*f(z+dt*k1/2)
            else:
                k2=f(z+dt*k1/2);k3=f(z+dt*k2/2);k4=f(z+dt*k3)
                next_z=z+dt*(k1+2*k2+2*k3+k4)/6
            u[at:at+2*n]=next_z.ravel();u[at+2*n:at+3*n]=np.maximum(live[at+2*n:at+3*n]-1,0)
        pre=u[volt].copy();hard=((pre>theta)&active).astype(float);s=hard.copy();gate=hard.copy()
        if anchors is not None:
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][t]-anchors[3]))**2
            s=anchors[1][t]+active*phi*(pre-theta-anchors[0][t]+anchors[3]);hard=anchors[1][t]
            gate=hard if p['detach_reset'] else s
        for projection,row in zip(p['projections'],w):
            src=projection['source_layer'];dst=projection['target_layer']-1
            inputs=x[t] if src==0 else s[neurons[src-1]:neurons[src]]
            for i,j,k in zip(projection['sources'],projection['targets'],projection['parameter_ids']):
                if active[neurons[dst]+j] and hard[neurons[dst]+j]==0:u[starts[dst]+j]+=inputs[i]*row[k]
        for l,n in enumerate(sizes):
            at=starts[l];ns=slice(neurons[l],neurons[l+1]);reset=u[at:at+2*n].copy()
            reset[n:]+=params[l]['kick']+.1*reset[:n];reset[:n]-=params[l]['theta']
            u[at:at+2*n]+=np.tile(gate[ns],2)*(reset-u[at:at+2*n])
            u[at+2*n:at+3*n]=np.where(hard[ns]!=0,2,u[at+2*n:at+3*n])
        live=u;pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,neurons[-2]:].mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,spikes,live,(np.array(pres),spikes,np.array(old),theta,np.array(activities))


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
def test_heterogeneous_independent_vjp(method,detach,window):
    net,source,groups,x,dt,requested=model(method);bundle=lower(net,source,groups,requested,detach_reset=detach,tbptt_window=window)
    w=bundle.weights;initial=np.array(bundle.initial_state)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=w).gradients(x[None],[0],initial=initial[None])
    loss,spikes,live,anchors=oracle(bundle,w,x,initial)
    assert result['loss']==pytest.approx(loss,abs=2e-14)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],live,atol=3e-14,rtol=3e-14)
    for bank,row in enumerate(w):
        if not bundle.plan['trainable'][bank]:continue
        for i,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);plus=copy.deepcopy(w);minus=copy.deepcopy(w)
            plus[bank][i]+=eps;minus[bank][i]-=eps
            fd=(oracle(bundle,plus,x,initial,anchors)[0]-oracle(bundle,minus,x,initial,anchors)[0])/(2*eps)
            assert result['gradients'][bank][i]==pytest.approx(fd,rel=2e-4,abs=4e-6)
    counters=[6,7,8,13,14]
    for i in range(len(initial)):
        if i in counters:assert result['initial_state_gradients'][0][i]==0;continue
        plus=initial.copy();minus=initial.copy();plus[i]+=1e-6;minus[i]-=1e-6
        fd=(oracle(bundle,w,x,plus,anchors)[0]-oracle(bundle,w,x,minus,anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][i]==pytest.approx(fd,rel=2e-4,abs=8e-8)


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('backend,ranks',[('cpu',2),('cpu',8),('metal',None),('metal',2),('metal',8),('cuda',None),('cuda',2),('cuda',8)])
def test_heterogeneous_backends_and_restore(method,backend,ranks,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    if backend=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('Metal required')
    if backend=='cuda' and os.environ.get('B2_TEST_CUDA_TRAIN')!='1':pytest.skip('CUDA required')
    net,source,groups,x,dt,requested=model(method);bundle=lower(net,source,groups,requested,detach_reset=False)
    p=copy.deepcopy(bundle.plan);p['backend']=backend
    if ranks:p['mpi_ranks']=ranks
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);target=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    for start,end in [(0,3),(3,7)]:
        initial=[bundle.initial_state] if start==0 else 'carry'
        a=cpu.step(x[None,start:end],[0],initial=initial);c=target.step(x[None,start:end],[0],initial=initial);compare(a,c)
        for key in ['final_state','initial_state_gradients']:np.testing.assert_allclose(a[key],c[key],atol=6e-6,rtol=4e-4)
        for bank,train in enumerate(p['trainable']):
            if not train:assert c['state']['weights'][bank]==bundle.weights[bank] and not any(c['state']['first_moment'][bank])
    checkpoint=tmp_path/'checkpoint';target.store(checkpoint)
    request=tmp_path/'request';result=tmp_path/'result';request.write_text(json.dumps(dict(plan=p,x=x[None,7:].tolist())))
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
r=json.load(open(sys.argv[1]));t=NativeLIFTrainer(r['plan'],runner=sys.argv[4]);t.restore(sys.argv[2])
json.dump(t.step(r['x'],[0],initial='carry'),open(sys.argv[3],'w'))
'''
    subprocess.run([sys.executable,'-c',code,str(request),str(checkpoint),str(result),str(RUNNER)],check=True,timeout=120)
    assert json.loads(result.read_text())==target.step(x[None,7:],[0],initial='carry')


@pytest.mark.parametrize('change',['node_range','threshold_range','missing_reference','nonpositive_last','bad_flag_count'])
def test_indexed_parameter_validation_is_atomic(change):
    net,source,groups,x,dt,requested=model();bundle=lower(net,source,groups,requested);p=copy.deepcopy(bundle.plan);w=copy.deepcopy(bundle.weights)
    bank,index=p['threshold_parameters'][0]
    if change=='node_range':p['state_equations'][0][0]=[dict(op='neuron_parameter',bank=bank,index=1)]
    elif change=='threshold_range':p['threshold_parameters'][0][1]=1
    elif change=='missing_reference':p['threshold_parameters'][0]=None
    elif change=='nonpositive_last':w[bank][-1]=0
    else:p['threshold_per_neuron']=[True]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step(x[None],[0],initial=[bundle.initial_state])
    assert trainer.state==before and trainer.neuron_state is None


def test_uniform_nonshared_trainable_parameters_keep_separate_slots():
    net,source,groups,x,dt,requested=model()
    for g in groups:g.tau=1*b.ms
    bundle=lower(net,source,groups,requested)
    for binding in bundle.provenance['bindings']:
        if binding['variables']==['tau']:
            assert binding['kind']=='neuron_array' and len(bundle.weights[binding['bank']])==len(next(g for g in groups if g.name==binding['object']))


@pytest.mark.parametrize('declaration',['integer (constant)','1'])
def test_trainable_parameter_types_reject_discrete_or_mutable(declaration):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    source=b.SpikeGeneratorGroup(2,[],[]*b.ms)
    groups=[b.NeuronGroup(2,'dv/dt=(-v+coefficient)/ms : 1\ncoefficient : '+declaration,
                          threshold='v>1',reset='v=0',method='euler') for _ in range(2)]
    for g in groups:g.coefficient=[1,2]
    with pytest.raises(TrainingConversionError,match='floating-point constants'):
        lower_brian_training(b.Network(source,*groups),input_group=source,layers=groups,
                             trainable_neuron_parameters={g.name:['coefficient'] for g in groups})
