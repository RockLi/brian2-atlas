"""Generated integrator stages keep callback draw identity and NumPy storage."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
import test_training_scalar_event_rng_functions as factories
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_regular_callback_rng import count
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(backend,kind,capture,method,window=None,sde=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=.1*gain*f(v,mean,spread)/ms'+('+gain*xi/ms**.5' if sde else '')+':1\nmean:1 (constant)\nspread:1 (constant)\ngain:1 (constant,shared)',method=method,threshold='v>.5',reset='v-=.5',dt=dt)
    g.v=[.48,.7];g.mean=[1.3,.7] if kind in ('poisson','mixed') else [.04,.07];g.spread=[.13,.11];g.gain=.2
    function=getattr(factories,kind)(g.variables['v'].get_value()) if capture else getattr(factories,'pure_'+kind)
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    return net,g,dt,bundle,np.zeros((1,4,1))


@pytest.mark.parametrize('kind,capture',[('normal',False),('normal',True),('uniform',False),('poisson',False),('mixed',True)])
@pytest.mark.parametrize('method',['euler','rk2','rk4'])
def test_integrator_callback_original_restore(engine,kind,capture,method,monkeypatch,tmp_path):
    mpi(2);net,g,dt,bundle,x=model(engine,kind,capture,method)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind]
    per_tick=dict(euler=1,rk2=2,rk4=4)[method];calls=[]
    layer=1
    for tick in range(4):
        serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;assert which==kinds[serial%len(kinds)];serial+=1
            mean=np.asarray(g.mean[:]);spread=np.asarray(g.spread[:])
            np.testing.assert_allclose(args,(mean,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            calls.append((tick,stream,which))
            if which=='poisson':return np.array([count(p['seed'],9,layer,j,tick,float(mean[j]),stream) for j in range(2)])
            return mean+spread*np.array([(normal if which=='normal' else uniform)(p['seed'],9,0,layer,j,tick,stream) for j in range(2)])
        for which in kinds:monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        cells=bundle.provenance['neuron_state_layout'][g.name]['v']
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.v[:],rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4*per_tick*len(kinds)
    if 'poisson' in kinds:assert len(trainer.poisson_state['entries'])==8*per_tick


def synaptic_model(backend,kind,capture,method,window=None):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    a=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    c=b.NeuronGroup(2,'dv/dt=(q+.01)/ms:1\nq:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    c.v=[.38,.7]
    syn=b.Synapses(a,c,'dh/dt=.1*gain*f(h,mean,spread)/ms:1 (clock-driven)\nq_post=h:1 (summed)\nmean:1 (constant)\nspread:1 (constant)\ngain:1 (constant,shared)',method=method,dt=dt)
    syn.connect(i=[1,0],j=[0,1]);syn.h=[.173,.219];syn.mean=[1.3,.7] if kind=='poisson' else [.04,.07];syn.spread=[.13,.11];syn.gain=.2
    function=getattr(factories,kind)(syn.variables['h'].get_value()) if capture else getattr(factories,'pure_'+kind)
    syn.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    net=b.Network(source,a,c,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[a,c],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_synapse_parameters={syn.name:['gain','mean','spread','h']})
    return net,(a,c),syn,dt,bundle,np.zeros((1,4,1))


@pytest.mark.parametrize('kind,capture',[('normal',False),('normal',True),('poisson',False)])
@pytest.mark.parametrize('method',['euler','rk2','rk4'])
def test_synaptic_integrator_callback_original_restore(engine,kind,capture,method,monkeypatch,tmp_path):
    mpi(2);net,groups,syn,dt,bundle,x=synaptic_model(engine,kind,capture,method)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);per_tick=dict(euler=1,rk2=2,rk4=4)[method];calls=[]
    for tick in range(4):
        serial=0
        def sample(*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1;mean=np.asarray(syn.mean[:]);spread=np.asarray(syn.spread[:]);calls.append((tick,stream))
            np.testing.assert_allclose(args,(mean,) if kind=='poisson' else (mean,spread))
            if kind=='poisson':return np.array([count(p['seed'],9,2,j,tick,float(mean[j]),stream) for j in range(2)])
            return mean+spread*np.array([normal(p['seed'],9,0,2,j,tick,stream) for j in range(2)])
        monkeypatch.setattr(np.random,'poisson' if kind=='poisson' else 'normal',sample)
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for group in groups:
            for name in ('v','q'):
                if name not in bundle.provenance['neuron_state_layout'][group.name]:continue
                cells=bundle.provenance['neuron_state_layout'][group.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],group.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        cells=bundle.provenance['dynamic_state_layout'][syn.name]['h'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],syn.h[:],rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4*per_tick
    if kind=='poisson':assert len(trainer.poisson_state['entries'])==8*per_tick


@pytest.mark.parametrize('method',['euler','heun','milstein'])
def test_integrator_callback_sde_streams_original(engine,method,monkeypatch):
    mpi(2);net,g,dt,bundle,x=model(engine,'normal',False,method,sde=True)
    import ast
    from brian2_rust.training_brian import _INTEGRATORS
    generated=_INTEGRATORS[method](g.equations,variables={**g.variables,'f':g.namespace['f']})
    callback_count=sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='f' for node in ast.walk(ast.parse(generated)))
    assert len(bundle.provenance['noise_names'][1])==1+callback_count
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);calls=[]
    for tick in range(4):
        serial=1
        def standard(n):
            assert n==2;calls.append((tick,0))
            return np.array([normal(p['seed'],9,0,1,j,tick,0) for j in range(2)])
        def callback(*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1;calls.append((tick,stream));mean=np.asarray(g.mean[:]);spread=np.asarray(g.spread[:]);np.testing.assert_allclose(args,(mean,spread))
            return mean+spread*np.array([normal(p['seed'],9,0,1,j,tick,stream) for j in range(2)])
        monkeypatch.setattr(np.random,'randn',standard);monkeypatch.setattr(np.random,'normal',callback)
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        cells=bundle.provenance['neuron_state_layout'][g.name]['v'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.v[:],rtol=8e-5,atol=8e-6)
    assert calls==[(tick,stream) for tick in range(4) for stream in range(1+callback_count)]


def reference(data,kind,capture,method,weights,initial=None,anchors=None,sde=False):
    from unittest.mock import patch
    from brian2_rust.training_brian import _INTEGRATORS
    _,g,_,bundle,_=data;p=bundle.plan
    state=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:state[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v']
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
        return np.asarray(weights[row['bank']]) if row['kind']=='neuron_array' else weights[row['bank']][row['variables'].index(name)]
    stage=compile(_INTEGRATORS[method](g.equations,variables={**g.variables,'f':g.namespace['f']}),'<independent generated Brian integration>','exec')
    before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:state=anchors['before'][tick].copy()
        before.append(state.copy());physical=state[v].copy()
        function=getattr(factories,kind)(physical) if capture else getattr(factories,'pure_'+kind)
        serial=int(sde)
        def distribution(which,*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1
            if which=='poisson':
                values=[]
                for j,rate in enumerate(np.asarray(args[0])):
                    value=count(p['seed'],9,1,j,tick,float(rate),stream)
                    if anchors is not None:value=next(c for t,s,k,r,c in anchors['samples'] if (t,s,k)==(tick,stream,j))
                    samples.append((tick,stream,j,float(rate),value));values.append(value)
                return np.array(values)
            raw=np.array([(normal if which=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(2)])
            return args[0]+(args[1] if which=='normal' else args[1]-args[0])*raw
        env=dict(v=physical.copy(),mean=parameter('mean'),spread=parameter('spread'),gain=parameter('gain'),dt=.0002,ms=.001,
                 f=function,sqrt=np.sqrt,randn=lambda:np.array([normal(p['seed'],9,0,1,j,tick,0) for j in range(2)]))
        with patch.object(np.random,'normal',lambda *args,**kwargs:distribution('normal',*args,**kwargs)),patch.object(np.random,'uniform',lambda *args,**kwargs:distribution('uniform',*args,**kwargs)),patch.object(np.random,'poisson',lambda *args,**kwargs:distribution('poisson',*args,**kwargs)):
            exec(stage,env)
        state[v]=env['v'];margin=state[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());state[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,state,dict(before=before,margins=margins,hard=hard,samples=samples)


@pytest.mark.parametrize('kind,capture,method,sde',[('normal',True,'euler',False),('normal',True,'rk4',False),('poisson',False,'rk2',False),('normal',False,'heun',True),('mixed',True,'rk4',False)])
@pytest.mark.parametrize('window',[None,2])
def test_integrator_callback_all_bank_initial_vjps(engine,kind,capture,method,sde,window):
    mpi(2);data=model(engine,kind,capture,method,window,sde);bundle=data[3]
    loss,state,anchors=reference(data,kind,capture,method,bundle.weights,sde=sde)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[4],[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    cells=bundle.provenance['neuron_state_layout'][data[1].name]['v']
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],state[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],anchors['hard'])
    mean=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==data[1].name and row['variables']==['mean'])
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,kind,capture,method,hi,anchors=anchors,sde=sde)[0]-reference(data,kind,capture,method,lo,anchors=anchors,sde=sde)[0])/2e-6
            score=sum(loss*(c/r-1.) for _,_,j,r,c in anchors['samples'] if j==index) if bank==mean else 0.
            assert out['gradients'][bank][index]==pytest.approx(fd+score,rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float)
    explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[4],[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(data,kind,capture,method,bundle.weights,initial=initial,sde=sde)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,kind,capture,method,bundle.weights,initial=hi,anchors=anchors,sde=sde)[0]-reference(data,kind,capture,method,bundle.weights,initial=lo,anchors=anchors,sde=sde)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),index


@pytest.mark.parametrize('kind,method',[('normal','euler'),('mixed','euler'),('poisson','rk2')])
def test_integrator_callback_invalid_restore_atomic(engine,kind,method,tmp_path):
    mpi(2);data=model(engine,kind,kind!='poisson',method);bundle=data[3];x=data[4]
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x[:,:1],[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind=='poisson' else 'spread';row=next(row for row in bundle.provenance['bindings'] if row['object']==data[1].name and row['variables']==[name])
    trainer.state['weights'][row['bank']][1]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x[:,1:2],[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before


def synaptic_reference(data,kind,capture,method,weights,initial=None,anchors=None):
    from unittest.mock import patch
    from brian2_rust.training_brian import _INTEGRATORS
    _,groups,syn,_,bundle,_=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    ns=bundle.provenance['neuron_state_layout'];ss=bundle.provenance['dynamic_state_layout'][syn.name]
    v=ns[groups[1].name]['v'];q=ns[groups[1].name]['q'];h=ss['h']
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==syn.name and name in row['variables'])
        return weights[row['bank']][0] if syn.variables[name].scalar else np.asarray(weights[row['bank']])
    stage=compile(_INTEGRATORS[method](syn.equations,variables={**syn.variables,'f':syn.namespace['f']}),'<independent generated synaptic integration>','exec')
    before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[q]=z[h];z[v]+=.2*(z[q]+.01)
        physical=z[h].copy();function=getattr(factories,kind)(physical) if capture else getattr(factories,'pure_'+kind);serial=0
        def sample(*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1
            if kind=='poisson':
                values=[]
                for j,rate in enumerate(np.asarray(args[0])):
                    value=count(p['seed'],9,2,j,tick,float(rate),stream)
                    if anchors is not None:value=next(c for t,s,k,r,c in anchors['samples'] if (t,s,k)==(tick,stream,j))
                    samples.append((tick,stream,j,float(rate),value));values.append(value)
                return np.array(values)
            return args[0]+args[1]*np.array([normal(p['seed'],9,0,2,j,tick,stream) for j in range(2)])
        env=dict(h=physical.copy(),mean=parameter('mean'),spread=parameter('spread'),gain=parameter('gain'),dt=.0002,ms=.001,f=function)
        with patch.object(np.random,'poisson' if kind=='poisson' else 'normal',sample):exec(stage,env)
        z[h]=env['h'];margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());z[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples)


@pytest.mark.parametrize('kind,capture,method',[('normal',True,'rk4'),('poisson',False,'rk2')])
@pytest.mark.parametrize('window',[None,2])
def test_synaptic_integrator_callback_all_bank_initial_vjps(engine,kind,capture,method,window):
    mpi(2);data=synaptic_model(engine,kind,capture,method,window);bundle=data[4]
    loss,z,anchors=synaptic_reference(data,kind,capture,method,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[5],[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    ns=bundle.provenance['neuron_state_layout'];cells=sorted(set(ns[data[1][0].name]['v']+ns[data[1][1].name]['v']+ns[data[1][1].name]['q']+bundle.provenance['dynamic_state_layout'][data[2].name]['h']))
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,2:],anchors['hard'])
    mean=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==data[2].name and row['variables']==['mean'])
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(synaptic_reference(data,kind,capture,method,hi,anchors=anchors)[0]-synaptic_reference(data,kind,capture,method,lo,anchors=anchors)[0])/2e-6
            score=sum(loss*(c/r-1.) for _,_,j,r,c in anchors['samples'] if j==index) if bank==mean else 0.
            assert out['gradients'][bank][index]==pytest.approx(fd+score,rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float)
    explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[5],[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=synaptic_reference(data,kind,capture,method,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(synaptic_reference(data,kind,capture,method,bundle.weights,initial=hi,anchors=anchors)[0]-synaptic_reference(data,kind,capture,method,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),index
