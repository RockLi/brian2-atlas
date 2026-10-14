"""Static TimedArray physical RK dynamics, native VJPs and actual Cython replay."""
import copy
import math
import os
import subprocess
import json
import ast
import importlib
from pathlib import Path

import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training_brian import lower_brian_training
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_poisson_zero_vjp import mpi
from brian2_rust.training_equations import TimedInput, SimulationTime, PoissonNoise, neuron_parameter_bank
from test_training_poisson_static import model as poisson_base, fixture as poisson_fixture, expression, draw, cython_uniforms
from unittest.mock import patch
from test_training_stochastic import normal


@pytest.fixture(params=['cpu','metal','cuda'])
def engine(request):
    name=request.param
    flag={'metal':'B2_TEST_GPU','cuda':'B2_TEST_CUDA_TRAIN'}.get(name)
    if flag and os.environ.get(flag)!='1':pytest.skip('actual '+name+' hardware required')
    return name


def model(dimensions=2, method='euler', warm=0, literal=False, noisy=False, **options):
    b.set_device('runtime'); b.start_scope(); b.prefs.codegen.target='cython'; dt=.2*b.ms
    values=np.array([[.4,.7],[1.2,.6],[.8,1.3],[1.4,.5]])
    if dimensions==1:values=values[:,0].copy()
    drive=b.TimedArray(values,dt=(.4 if noisy and method=='euler' else .3)*b.ms,name='static_drive')
    x=np.array([1.,0.,1.,1.,0.,1.,0.,1.])[:,None]; ticks=np.flatnonzero(x[:,0])
    inp=b.SpikeGeneratorGroup(1,np.zeros(len(ticks),int),ticks*dt,dt=dt,name='static_timed_input')
    groups=[]
    for layer in range(2):
        name='drive' if layer==0 else 'alias'
        def call(time):return name+'('+time+(',i' if dimensions==2 else '')+')'
        left=call('.1*ms' if literal else 't-.1*ms'); right=call('.4*ms' if literal else 't+.1*ms')
        eq=f'dv/dt=(-v+.2*a+gain*{left})/tau:1\nda/dt=(.1*v-.3*a+.04*{right})/tau:1\ngain:1 (shared,constant)\ntau:second (shared,constant)\ntheta:1 (shared,constant)\nkick:1 (shared,constant)'
        if noisy:
            vfactor='1' if method=='euler' else '(v+1)'
            afactor='1' if method=='euler' else '(a+1)'
            eq=eq.replace('/tau:1',f'/tau+sigma*{call("t")}*{vfactor}*xi_v/sqrt(tau):1',1)
            eq=eq.replace('/tau:1',f'/tau+.5*sigma*{call("t")}*{afactor}*xi_a/sqrt(tau):1',1)
            eq+='\nsigma:1 (shared,constant)'
        reset=f'a+=kick*{call(".7*ms" if literal else "t+.09*ms")}+.1*v\nv-=theta+.02*{call("0*ms" if literal else "t-.11*ms")}'
        g=b.NeuronGroup(2,eq,threshold='v>theta',reset=reset,method=method,dt=dt,
                        namespace={name:drive},name=f'static_timed_{layer}')
        g.gain=1.7+.2*layer;g.tau=(1+.1*layer)*b.ms;g.theta=.7+.05*layer;g.kick=.08+.02*layer
        if noisy:g.sigma=.13+.02*layer
        g.v=[.8,1.3] if layer==0 else [1.2,.5];g.a=[.1,.2];groups.append(g)
    synapses=[]
    for index,(source,target) in enumerate(((inp,groups[0]),(groups[0],groups[1]),(groups[1],groups[0]))):
        syn=b.Synapses(source,target,'w:1',on_pre='v_post+=w',name=f'static_timed_syn_{index}');syn.connect()
        syn.w=.06+.04*((np.asarray(syn.i)+2*np.asarray(syn.j)+index)%3);synapses.append(syn)
    net=b.Network(inp,*groups,*synapses)
    if warm:net.run(warm*dt,namespace={})
    net.run(0*dt,namespace={})
    bundle=lower_brian_training(net,input_group=inp,layers=groups,
        trainable_neuron_parameters={g.name:['gain','tau','theta','kick']+(['sigma'] if noisy else []) for g in groups},
        detach_reset=False,**options)
    return net,groups,drive,x[warm:],bundle


def coefficients(bundle,weights):
    result=[]
    for owner in bundle.provenance['layer_names']:
        row={}
        for binding in bundle.provenance['bindings']:
            if binding['object']==owner and binding['kind']=='neuron':
                row.update({name:weights[binding['bank']][index] for index,name in enumerate(binding['variables'])})
        result.append(row)
    return result


def oracle(bundle,weights,inputs,labels,initial,anchors=None,literal=False,change=None,noisy=False,sequence=9):
    p=bundle.plan; batch,length,_=inputs.shape; live=initial.copy(); starts=[]; margins=[]; spikes=[]
    entry=bundle.provenance['timed_inputs'][0]; table=np.array(weights[entry['bank']]).reshape(entry['shape'])
    array_dt=float(f'{entry["dt_seconds"]:.18f}'); dt=p['clock']['dt']; k=max(1,2**math.ceil(math.log2(8*entry['dt_seconds']/dt))); epsilon=array_dt/k
    params=coefficients(bundle,weights); matrices=[]
    for projection,row in zip(p['projections'],weights):
        matrix=np.zeros((p['sizes'][projection['source_layer']],p['sizes'][projection['target_layer']]))
        np.add.at(matrix,(projection['sources'],projection['targets']),np.asarray(row)[projection['parameter_ids']]);matrices.append(matrix)
    def sample(time):
        row=int(np.clip((time/epsilon+.5)/k,0,len(table)-1))
        return np.repeat(table[row],2) if table.ndim==1 else table[row]
    for tick in range(length):
        if change is not None and tick==change[0]:table=np.array(change[1]).reshape(entry['shape'])
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:live=anchors['starts'][tick].copy()
        starts.append(live.copy());u=live.copy();time=p['clock']['origin']+tick*dt
        for layer in range(2):
            c=params[layer];offset=layer*4;z=np.stack((live[:,offset:offset+2],live[:,offset+2:offset+4]),axis=-1)
            def f(z,t):
                v,a=z[...,0],z[...,1]
                return np.stack((-v+.2*a+c['gain']*sample(.0001 if literal else t-.0001),
                                 .1*v-.3*a+.04*sample(.0004 if literal else t+.0001)),axis=-1)*dt/c['tau']
            first=f(z,time);method=bundle.provenance['integrators'][layer]
            if noisy:
                old=z.copy();next_z=old+first
                for stream,name in enumerate(bundle.provenance['noise_names'][layer]):
                    index=0 if name=='xi_v' else 1
                    dw=math.sqrt(dt)*np.array([[normal(p['seed'],sequence,bi,layer,j,tick,stream) for j in range(2)] for bi in range(batch)])
                    def diffusion(value,t):
                        return (1. if index==0 else .5)*c['sigma']/math.sqrt(c['tau'])*sample(t)*(np.ones_like(value) if method=='euler' else value+1)
                    base=diffusion(old[...,index],time)
                    if method=='euler':term=base*dw
                    elif method=='heun':term=.5*dw*(base+diffusion(old[...,index]+base*dw,time+dt))
                    elif method=='milstein':term=base*dw+(diffusion(old[...,index]+first[...,index]+math.sqrt(dt)*base,time)-base)*dw**2/(2*math.sqrt(dt))
                    else:raise AssertionError(method)
                    next_z[...,index]+=term
                z=next_z
            elif method=='euler':z=z+first
            elif method=='rk2':z=z+f(z+.5*first,time+.5*dt)
            elif method=='rk4':
                second=f(z+.5*first,time+.5*dt);third=f(z+.5*second,time+.5*dt);fourth=f(z+third,time+dt)
                z=z+(first+2*second+2*third+fourth)/6
            else:raise AssertionError(method)
            u[:,offset:offset+2]=z[...,0];u[:,offset+2:offset+4]=z[...,1]
        theta=np.repeat([c['theta'] for c in params],2);margin=u[:,[0,1,4,5]]-theta;hard=(margin>0).astype(float);event=hard
        if anchors is not None:
            base=anchors['margins'][tick];event=(base>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margin-base)
        gate=hard if anchors is None or p['detach_reset'] else event
        for projection,matrix in zip(p['projections'],matrices):
            source=projection['source_layer'];target=projection['target_layer'];offset=(target-1)*4
            x=inputs[:,tick] if source==0 else event[:,(source-1)*2:source*2]
            u[:,offset:offset+2]+=x@matrix
        live=u.copy()
        for layer,c in enumerate(params):
            offset=layer*4;g=gate[:,2*layer:2*layer+2]
            live[:,offset+2:offset+4]+=g*(c['kick']*sample(.0007 if literal else time+.00009)+.1*u[:,offset:offset+2])
            live[:,offset:offset+2]-=g*(c['theta']+.02*sample(0. if literal else time-.00011))
        margins.append(margin.copy());spikes.append(event.copy())
    spikes=np.stack(spikes,1);logits=spikes[:,:,2:].mean(1)*p['logit_scale'];maximum=logits.max(1)
    loss=np.mean(maximum+np.log(np.exp(logits-maximum[:,None]).sum(1))-logits[np.arange(batch),labels])
    return loss,live,spikes,dict(starts=starts,margins=margins)


@pytest.mark.parametrize('dimensions',[1,2])
@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('warm',[0,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_static_frontend_actual_cython(engine,dimensions,method,warm,ranks,tmp_path):
    mpi(ranks);net,groups,drive,x,bundle=model(dimensions,method,warm,backend=engine,mpi_ranks=ranks)
    assert bundle.plan['schema']=='b2-state-training-plan-v4' and 'dynamic' not in bundle.plan
    assert len(bundle.provenance['timed_inputs'])==1 and len(bundle.provenance['timed_inputs'][0]['aliases'])==2
    bundle.plan['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);outputs=[];cursor=0
    for part in (x[:2],x[2:]):
        result=t.step(part[None],[0],**({'initial':'carry'} if cursor else {'initial':[bundle.initial_state]}))
        outputs.extend(result['spikes'][0]);net.run(len(part)*.2*b.ms,namespace={});cursor+=len(part)
        expected=np.concatenate([g.variables[name].get_value() for g,names in zip(groups,bundle.provenance['state_names']) for name in names])
        np.testing.assert_allclose(result['final_state'][0],expected,rtol=4e-5 if engine!='cpu' else 3e-12,atol=5e-6 if engine!='cpu' else 3e-14)
        assert (result['gpu_dispatches']>0)==(engine!='cpu') and result['backend']==engine
        path=tmp_path/'state';t.store(path);restored=NativeLIFTrainer(bundle.plan,runner=RUNNER);restored.restore(path);t=restored
    events=np.zeros((len(x),4))
    for layer,monitor in enumerate(monitors):
        ticks=np.rint(np.asarray(monitor.t/b.second)/.0002).astype(int)-warm
        events[ticks,2*layer+np.asarray(monitor.i)]=1.
    np.testing.assert_array_equal(outputs,events)


@pytest.mark.parametrize('method,noisy',[('euler',False),('rk2',False),('rk4',False),('euler',True),('heun',True),('milstein',True)])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_static_all_parameters_and_initial_vjps(engine,method,noisy,window,ranks):
    mpi(ranks);_,_,_,x,bundle=model(method=method,noisy=noisy,tbptt_window=window,backend=engine,mpi_ranks=ranks)
    x=np.stack([x*.75-.125,x*.5+.1]);labels=np.array([0,1]);initial=np.tile(bundle.initial_state,(2,1));initial[1]+=.07
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,labels,initial=initial,**({'noise_sequence':9} if noisy else {}))
    loss,final,events,anchors=oracle(bundle,bundle.weights,x,labels,initial,noisy=noisy)
    np.testing.assert_allclose(result['loss'],loss,atol=3e-6 if engine!='cpu' else 3e-14,rtol=0)
    np.testing.assert_array_equal(result['spikes'],events)
    np.testing.assert_allclose(result['final_state'],final,atol=7e-6 if engine!='cpu' else 3e-14,rtol=4e-5 if engine!='cpu' else 3e-12)
    selectors={entry['bank'] for entry in bundle.provenance['bindings'] if entry['variables']==['i']}
    for bank,row in enumerate(bundle.weights):
        for j,value in enumerate(row):
            if bank in selectors:
                assert result['gradients'][bank][j]==0.;continue
            eps=1e-6*max(abs(value),.001);a=copy.deepcopy(bundle.weights);c=copy.deepcopy(bundle.weights);a[bank][j]+=eps;c[bank][j]-=eps
            fd=(oracle(bundle,a,x,labels,initial,anchors,noisy=noisy)[0]-oracle(bundle,c,x,labels,initial,anchors,noisy=noisy)[0])/(2*eps)
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=5e-3 if engine!='cpu' else 8e-5,abs=5e-3 if engine!='cpu' else 3e-6)
    for sample in range(2):
        for j in range(8):
            a=initial.copy();c=initial.copy();a[sample,j]+=1e-6;c[sample,j]-=1e-6
            fd=(oracle(bundle,bundle.weights,x,labels,a,anchors,noisy=noisy)[0]-oracle(bundle,bundle.weights,x,labels,c,anchors,noisy=noisy)[0])/2e-6
            assert result['initial_state_gradients'][sample][j]==pytest.approx(fd,rel=2e-3 if engine!='cpu' else 8e-5,abs=2e-5 if engine!='cpu' else 3e-7)
    if engine!='cpu':assert result['gpu_dispatches']>0 and result['backend']==engine


@pytest.mark.parametrize('dimensions',[1,2])
def test_literal_calls_choose_vector_clock(engine,dimensions):
    net,groups,drive,x,bundle=model(dimensions,literal=True,backend=engine)
    assert bundle.plan['schema']=='b2-state-training-plan-v4' and bundle.plan['clock']['dt']==.0002
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(x[None],[0],initial=[bundle.initial_state])
    loss,final,events,_=oracle(bundle,bundle.weights,x[None],[0],np.array([bundle.initial_state]),literal=True)
    np.testing.assert_allclose(result['final_state'],final,rtol=4e-5 if engine!='cpu' else 3e-12,atol=7e-6 if engine!='cpu' else 3e-14)
    np.testing.assert_array_equal(result['spikes'],events)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_sequence_boundary_update_and_rollback(engine,ranks,tmp_path):
    mpi(ranks);_,_,_,x,bundle=model(backend=engine,mpi_ranks=ranks)
    bundle.plan['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    t.step(x[None,:3],[0],initial=[bundle.initial_state]);bank=bundle.provenance['timed_inputs'][0]['bank']
    before=(copy.deepcopy(t.state),copy.deepcopy(t.neuron_state),t.clock_tick,copy.deepcopy(t.clock_state),t.noise_sequence)
    for values in ([1.]*3,[float('nan')]*8,[float('inf')]*8):
        with pytest.raises(ValueError):t.update_timed_input(bank,values)
        assert (t.state,t.neuron_state,t.clock_tick,t.clock_state,t.noise_sequence)==before
    values=np.array([[.3,.4],[.6,.2],[.8,.1],[.5,.9]])
    t.update_timed_input(bank,values)
    assert (t.neuron_state,t.clock_tick,t.clock_state,t.noise_sequence)==before[1:]
    assert t.state['weights'][bank]==values.reshape(-1).tolist()
    assert t.state['first_moment']==before[0]['first_moment'] and t.state['second_moment']==before[0]['second_moment']
    path=tmp_path/'updated';t.store(path);q=NativeLIFTrainer(bundle.plan,runner=RUNNER);q.restore(path)
    a=t.step(x[None,3:],[0],initial='carry');assert q.step(x[None,3:],[0],initial='carry')==a
    loss,final,events,_=oracle(bundle,bundle.weights,x[None],[0],np.array([bundle.initial_state]),change=(3,values))
    np.testing.assert_allclose(a['final_state'],final,rtol=5e-5 if engine!='cpu' else 3e-12,atol=8e-6 if engine!='cpu' else 3e-14)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_owner_column_failure_is_atomic(engine,ranks):
    mpi(ranks);_,_,_,x,bundle=model(backend=engine,mpi_ranks=ranks)
    selectors=[binding['bank'] for binding in bundle.provenance['bindings'] if binding['variables']==['i']]
    assert selectors;bundle.weights[selectors[-1]][1]=2.
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='index|nonfinite|invalid|evaluation'):t.step(x[None],[0],initial=[bundle.initial_state])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0


def poisson_model(ranks=None,zero=False,invalid=False,masked=False):
    p,w=poisson_base(ranks=ranks,reset_draw=False)
    p['projections'].append(neuron_parameter_bank(8));w.append([0.]*8 if zero else [.5,1.25,.75,1.5,.4,.6,1.1,.8])
    p['masks'].append([0. if masked else 1.]*8);p['trainable']=[False]*5
    params={'draw':PoissonNoise(0),'scale':(3,0),'amp':(3,1),'t':SimulationTime(),'drive':TimedInput(4,8,1,.00025,8,1)}
    for layer in p['state_equations']:
        layer[0]=expression('1./(1.-draw(scale*drive(t)))' if invalid else '.4*v+amp*draw(scale*drive(t))',params)
    return p,w


def poisson_physical(p,w,x,labels,initial,force=None):
    live=initial.copy();batch,length,_=x.shape;events=[];records=[]
    for tick in range(length):
        time=p['clock']['origin']+tick*p['clock']['dt'];row=int(np.clip((time/.00025+.5)/8,0,7));old=live.copy();counts=np.zeros((batch,3))
        for bindex in range(batch):
            for j in range(3):
                rate=w[3][0]*w[4][row];count=1 if force==(tick,bindex,j) else draw(rate,p['seed'],9,bindex,int(j>0),max(j-1,0),tick)
                counts[bindex,j]=count;records.append((tick,bindex,j,row,rate,count))
        live[:,[0,2,3]]=.4*old[:,[0,2,3]]+w[3][1]*counts;live[:,[1,4,5]]=.8*old[:,[1,4,5]]+.2*w[3][2]
        hard=(live[:,[0,2,3]]>.6).astype(float);events.append(hard.copy())
        live[:,0]+=w[0][0]*x[:,tick,0]+w[2][0]*hard[:,1:].sum(1);live[:,2:4]+=hard[:,:1]*np.array(w[1])
        live[:,[0,2,3]]-=.6*hard;live[:,[1,4,5]]+=hard*(-.1*live[:,[1,4,5]]+.1*w[3][2])
    events=np.stack(events,1);logits=events[:,:,1:].mean(1)*p['logit_scale'];maximum=logits.max(1)
    losses=maximum+np.log(np.exp(logits-maximum[:,None]).sum(1))-logits[np.arange(batch),labels]
    return losses,live,events,records


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('window',[None,2])
def test_timed_rate_positive_and_weak_bank_vjps(ranks,zero,window):
    mpi(ranks);p,w=poisson_model(ranks,zero);p['tbptt_window']=window;x,y,initial=poisson_fixture(3)
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,y,initial=initial,noise_sequence=9)
    losses,final,events,records=poisson_physical(p,w,x,y,initial);expected=np.zeros(8);scale=0.
    for tick,bi,j,row,rate,count in records:
        coef=(poisson_physical(p,w,x,y,initial,(tick,bi,j))[0][bi]-losses[bi])/3 if zero else losses[bi]/3*(count/rate-1)
        expected[row]+=coef*w[3][0];scale+=coef*w[4][row]
    np.testing.assert_array_equal(result['spikes'],events);np.testing.assert_allclose(result['final_state'],final,rtol=3e-13,atol=3e-14)
    np.testing.assert_allclose(result['gradients'][4],expected,rtol=4e-13,atol=4e-14)
    assert result['gradients'][3][0]==pytest.approx(scale,rel=4e-13,abs=4e-14)
    assert np.any(expected!=0.)


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('masked',[False,True])
def test_zero_timed_rate_mask_controls_counterfactual_failure(ranks,masked):
    mpi(ranks);p,w=poisson_model(ranks,zero=True,invalid=True,masked=masked);x,y,initial=poisson_fixture(1)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(t.state)
    if masked:
        result=t.gradients(x[:,:1],y,initial=initial)
        np.testing.assert_array_equal(result['gradients'][4],0.)
    else:
        with pytest.raises(ValueError,match='nonfinite equation'):t.step(x[:,:1],y,initial=initial)
        assert t.state==before and t.poisson_state is None and t.neuron_state is None and t.clock_tick==0


@pytest.mark.parametrize('ranks',[None,2,8])
def test_input_update_preserves_observed_rate_and_checkpoint(ranks,tmp_path):
    mpi(ranks);p,w=poisson_model(ranks);x,y,initial=poisson_fixture(3);t=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    first=t.step(x[:,:2],y,initial=initial,noise_sequence=9);cache=copy.deepcopy(t.poisson_state)
    clock=(t.clock_tick,t.noise_sequence,t.next_noise_sequence);live=copy.deepcopy(t.neuron_state)
    t.update_timed_input(4,[-1.]*8)
    assert t.poisson_state==cache and t.neuron_state==live and (t.clock_tick,t.noise_sequence,t.next_noise_sequence)==clock
    path=tmp_path/'poisson-input';t.store(path);q=NativeLIFTrainer(p,runner=RUNNER);q.restore(path)
    # Rewinding to old identities consumes imported draws, without looking up
    # the new negative rates or assigning their score to the edited table.
    t.clock_tick=0;q.clock_tick=0
    a=t.gradients(x[:,:2],y,initial='carry');assert q.gradients(x[:,:2],y,initial='carry')==a
    assert a['poisson_state']==cache;np.testing.assert_array_equal(a['gradients'][4],0.)
    t.clock_tick=2;before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='Poisson sample'):t.step(x[:,2:],y,initial='carry')
    assert t.state==before and t.poisson_state==cache and t.clock_tick==2


@pytest.mark.parametrize('dimensions',[1,2])
@pytest.mark.parametrize('warm',[0,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_static_timed_poisson_matches_original_cython(dimensions,warm,ranks,tmp_path):
    _static_timed_poisson_cython(dimensions,warm,ranks,tmp_path)


def _static_timed_poisson_cython(dimensions,warm,ranks,tmp_path,backend='cpu'):
    mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.125*b.ms
    table=np.array([[1.25,2.5],[.75,1.5],[2.5,1.25],[1.5,.75]])
    if dimensions==1:table=table[:,0].copy()
    drive=b.TimedArray(table,dt=2*dt,name='poisson_static_drive')
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=dt,name='poisson_timed_input');groups=[]
    for layer in range(2):
        name='drive' if layer==0 else 'alias';call=lambda time:name+'('+time+(',i' if dimensions==2 else '')+')'
        g=b.NeuronGroup(2,f'dv/dt=(-v+.5*poisson(scale*{call("t")}))/ms:1\nda/dt=-a/ms:1\nscale:1 (shared,constant)',
            threshold='v>.3',reset=f'a+=.125*poisson(scale*{call("t+.125*ms")})\nv-=.25+.125*a',
            method='euler',dt=dt,namespace={name:drive},name=f'poisson_static_time_{layer}')
        g.v=[.45,.75];g.a=[.125,.25];g.scale=1.;groups.append(g)
    syn=b.Synapses(groups[0],groups[1],'w:1',on_pre='v_post+=w',name='poisson_static_timed_syn');syn.connect(j='i');syn.w=[.0625,.125]
    net=b.Network(inp,*groups,syn)
    if warm:b.seed(731);net.run(warm*dt,namespace={})
    net.run(0*dt,namespace={});bundle=lower_brian_training(net,input_group=inp,layers=groups,mpi_ranks=ranks,
        trainable_neuron_parameters={g.name:['scale'] for g in groups})
    bundle.plan['backend']=backend
    bundle.plan['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);cursor=0
    for length in (2,3):
        r=t.step(np.zeros((1,length,1)),[0],**({'initial':'carry'} if cursor else {'initial':[bundle.initial_state],'noise_sequence':9}))
        draws=[]
        for tick in range(cursor,cursor+length):
            for layer in range(2):
                for j in range(2):
                    row=min(3,(warm+tick)//2);rate=table[row] if dimensions==1 else table[row,j]
                    draws.extend(cython_uniforms(bundle.plan['seed'],9,layer,j,tick,0,float(rate)))
            for layer in range(2):
                for j in range(2):
                    if r['spikes'][0][tick-cursor][2*layer+j]:
                        row=min(3,(warm+tick+1)//2);rate=table[row] if dimensions==1 else table[row,j]
                        draws.extend(cython_uniforms(bundle.plan['seed'],9,layer,j,tick,1,float(rate)))
        device=b.get_device();device.rand_buffer_index[:]=0;calls=[]
        def refill(n):
            assert n==20000 and not calls;calls.append(n);result=np.full(n,.5);result[:len(draws)]=draws;return result
        with patch('numpy.random.rand',refill):net.run(length*dt,namespace={})
        assert calls==[20000] and device.rand_buffer_index[0]==len(draws);device.rand_buffer_index[:]=0
        expected=np.concatenate([g.variables[name].get_value() for g,names in zip(groups,bundle.provenance['state_names']) for name in names])
        np.testing.assert_allclose(r['final_state'][0],expected,
            rtol=3e-12 if backend=='cpu' else 4e-5,atol=3e-14 if backend=='cpu' else 5e-6)
        assert r['backend']==backend and (r['gpu_dispatches']>0)==(backend!='cpu')
        path=tmp_path/'timed-poisson';t.store(path);q=NativeLIFTrainer(bundle.plan,runner=RUNNER);q.restore(path);t=q;cursor+=length


@pytest.mark.parametrize('method',['euler','heun','milstein'])
@pytest.mark.parametrize('warm',[0,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_static_timed_diffusion_actual_cython(engine,method,warm,ranks,tmp_path):
    mpi(ranks);net,groups,drive,x,bundle=model(method=method,warm=warm,noisy=True,backend=engine,mpi_ranks=ranks)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    order=[]
    for layer,group in enumerate(groups):
        for statement in ast.parse(group.state_updater.abstract_code).body:
            if isinstance(statement,ast.Assign) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='randn' for n in ast.walk(statement.value)):
                order.append((layer,bundle.provenance['noise_names'][layer].index(statement.targets[0].id)))
    assert len(order)==4
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);cursor=0
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);events=[]
    for part in (x[:2],x[2:]):
        result=t.step(part[None],[0],**({'initial':'carry'} if cursor else {'initial':[bundle.initial_state],'noise_sequence':9}));events.extend(result['spikes'][0])
        # Original Cython integration/event code, replacing only external normal
        # samples in its generated per-neuron call order.
        draws=[normal(bundle.plan['seed'],9,0,layer,j,tick,stream) for tick in range(cursor,cursor+len(part)) for layer in range(2) for j in range(2) for owner,stream in order if owner==layer]
        device=b.get_device();device.randn_buffer_index[:]=0;calls=[]
        def refill(n):
            assert n==20000 and not calls;calls.append(n);values=np.zeros(n);values[:len(draws)]=draws;return values
        with patch('numpy.random.randn',refill):net.run(len(part)*.2*b.ms,namespace={})
        assert calls==[20000] and device.randn_buffer_index[0]==len(draws);device.randn_buffer_index[:]=0
        expected=np.concatenate([g.variables[name].get_value() for g,names in zip(groups,bundle.provenance['state_names']) for name in names])
        np.testing.assert_allclose(result['final_state'][0],expected,rtol=5e-5 if engine!='cpu' else 3e-12,atol=8e-6 if engine!='cpu' else 3e-14)
        assert result['backend']==engine and (result['gpu_dispatches']>0)==(engine!='cpu')
        path=tmp_path/'timed-sde';t.store(path);q=NativeLIFTrainer(bundle.plan,runner=RUNNER);q.restore(path);t=q;cursor+=len(part)
    expected=np.zeros((len(x),4))
    for layer,monitor in enumerate(monitors):
        ticks=np.rint(np.asarray(monitor.t/b.second)/.0002).astype(int)-warm;expected[ticks,2*layer+np.asarray(monitor.i)]=1
    np.testing.assert_array_equal(events,expected)


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('version',[None,0,2])
def test_static_timed_input_requires_versioned_gpu_capability(backend,version,tmp_path,monkeypatch):
    *_,x,bundle=model(backend=backend)
    source=tmp_path/'old.c';library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_static_timed_input_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    monkeypatch.setattr(importlib.import_module('brian2_rust.training_'+backend),'build',lambda directory:library)
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='GPU static timed input capability'):t.evaluate(x[None],[0],initial=[bundle.initial_state])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0


@pytest.mark.parametrize('ranks',[None,2,8])
def test_static_update_budget_includes_retained_poisson_cache(ranks):
    mpi(ranks);p,w=poisson_model(ranks);x,y,initial=poisson_fixture(3)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(np.tile(x[:,:2],(1,5,1)),y,initial=initial,noise_sequence=9)
    # Isolate the setter boundary from trajectory and draw allocation budgets.
    request=dict(plan=copy.deepcopy(p),state=copy.deepcopy(t.state),operation='update_timed_input',inputs=[],labels=[],
        initial=t.neuron_state,start_tick=t.clock_tick,noise_sequence=t.noise_sequence,poisson_state=t.poisson_state,
        input_update=dict(bank=4,values=[.5]*8))
    accepted=t._run_request(request);assert accepted['tape_bytes']>=len(t.poisson_state['entries'])*768
    request['plan']['max_tape_bytes']=accepted['tape_bytes']-1
    before=copy.deepcopy((t.state,t.poisson_state,t.neuron_state,t.clock_tick))
    with pytest.raises(ValueError,match='input update exceeds memory budget'):t._run_request(request)
    assert (t.state,t.poisson_state,t.neuron_state,t.clock_tick)==before
