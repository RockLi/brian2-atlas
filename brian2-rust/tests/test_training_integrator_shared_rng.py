"""Generated neuron/synapse stages sample scalar callbacks once per block."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
import test_training_scalar_event_rng_functions as factories
from test_training_threshold_shared_rng import scalar_chain
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_regular_callback_rng import count
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def function(kind,capture,array):
    return scalar_chain if kind=='chain' else getattr(factories,kind)(array) if capture else getattr(factories,'pure_'+kind)


def model(backend,kind,capture,method,window=None,synaptic=False,sde=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(2 if synaptic else 1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    if synaptic:
        g=b.NeuronGroup(2,'dv/dt=(q+.01)/ms:1\nq:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
        syn=b.Synapses(hidden,g,'dh/dt=.1*gain*f(h,mean,spread)/ms:1 (clock-driven)\nq_post=h:1 (summed)\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant,shared)',method=method,dt=dt)
        syn.connect(i=[1,0],j=[0,1]);syn.h=[.173,.219];obj=syn;name='h';net=b.Network(source,hidden,g,syn);g.v=[.383,.713]
    else:
        g=b.NeuronGroup(2,'dv/dt=.1*gain*f(v,mean,spread)/ms'+('+gain*xi/ms**.5' if sde else '')+':1\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant,shared)',method=method,threshold='v>.5',reset='v-=.5',dt=dt)
        obj=g;name='v';g.v=[.483,.713];net=b.Network(source,hidden,g)
    obj.mean=1.3 if kind in ('poisson','mixed','chain') else .043;obj.spread=.13;obj.gain=.2
    obj.namespace['f']=b.Function(function(kind,capture,obj.variables[name].get_value()),arg_units=[1,1,1],return_unit=1,stateless=False)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,
        **({'trainable_synapse_parameters':{obj.name:['gain','mean','spread','h']}} if synaptic else {'trainable_neuron_parameters':{g.name:['gain','mean','spread']}}))
    return net,hidden,g,obj,dt,bundle


PROFILES=[('normal',False,'euler',False,False),('normal',True,'rk4',False,False),
    ('uniform',False,'rk2',False,False),('poisson',True,'rk2',False,False),
    ('mixed',True,'rk4',False,False),('chain',False,'rk2',False,False),
    ('normal',True,'rk4',True,False),('poisson',False,'rk2',True,False),('chain',False,'rk2',True,False),
    ('normal',False,'heun',False,True),('normal',False,'milstein',False,True)]


@pytest.mark.parametrize('kind,capture,method,synaptic,sde',PROFILES)
def test_integrator_shared_original_restore(engine,kind,capture,method,synaptic,sde,monkeypatch,tmp_path):
    mpi(2);net,hidden,g,obj,dt,bundle=model(engine,kind,capture,method,synaptic=synaptic,sde=sde)
    monitor=b.SpikeMonitor(g);net.add(monitor)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    records=bundle.provenance['synaptic_integrator_draws'][obj.name] if synaptic else bundle.provenance['neuron_draws'][1]['update']
    records=[row for row in records if row.get('scalar',False)];assert records
    cells=bundle.provenance['synaptic_integrator_shared_draw_layout'][obj.name] if synaptic else bundle.provenance['integrator_shared_draw_layout'][obj.name]
    assert len(cells)==len(records);offset=int(sde);domain=2 if synaptic else 1
    kinds=['normal','uniform','poisson'] if kind=='mixed' else ['normal','poisson'] if kind=='chain' else [kind]
    for tick in range(4):
        serial=offset
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;assert which==kinds[(serial-offset)%len(kinds)];serial+=1
            mean=float(obj.mean[:]);spread=float(obj.spread[:]);assert all(np.ndim(value)==0 for value in args)
            rate=mean+.2*(mean+spread*normal(p['seed'],9,0,domain,0,tick,stream-1)) if kind=='chain' and which=='poisson' else mean
            np.testing.assert_allclose(args,(rate,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            if which=='poisson':return int(count(p['seed'],9,domain,0,tick,rate,stream))
            return float(mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,0,tick,stream))
        for which in kinds:monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        if sde:
            def standard(n):
                assert n==2
                return np.array([normal(p['seed'],9,0,domain,j,tick,0) for j in range(2)])
            monkeypatch.setattr(np.random,'randn',standard)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        assert serial==offset+len(records)
        for group in (hidden,g):
            for name in ('v','q'):
                if name not in bundle.provenance['neuron_state_layout'][group.name]:continue
                indices=bundle.provenance['neuron_state_layout'][group.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,indices],group.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        if synaptic:
            indices=bundle.provenance['dynamic_state_layout'][obj.name]['h']
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,indices],obj.h[:],rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,len(hidden):],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    if 'poisson' in kinds:assert len(trainer.poisson_state['entries'])==4*len(records)//len(kinds)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def parameter(data,weights,name):
    obj=data[3];bundle=data[5]
    row=next(row for row in bundle.provenance['bindings'] if row['object']==obj.name and name in row['variables'])
    return weights[row['bank']][row['variables'].index(name)] if len(row['variables'])>1 or obj.variables[name].scalar else np.asarray(weights[row['bank']])


def reference(data,kind,capture,method,weights,initial=None,anchors=None,override=None,sde=False):
    from unittest.mock import patch
    from brian2_rust.training_brian import _INTEGRATORS
    _,hidden,g,obj,_,bundle=data;p=bundle.plan;synaptic=obj is not g;domain=2 if synaptic else 1
    z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];target=bundle.provenance['dynamic_state_layout'][obj.name]['h'] if synaptic else v
    name='h' if synaptic else 'v';stage=compile(_INTEGRATORS[method](obj.equations,variables={**obj.variables,'f':obj.namespace['f']}),'<independent shared integrator>','exec')
    before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        if synaptic:z[layout['q']]=z[target];z[v]+=.2*(z[layout['q']]+.01)
        physical=z[target].copy();serial=int(sde)
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1
            if which=='poisson':
                rate=float(args[0]);value=count(p['seed'],9,domain,0,tick,rate,stream)
                if anchors is not None:value=next(c for t,s,r,c in anchors['samples'] if (t,s)==(tick,stream))
                if override==(tick,stream):value+=1
                samples.append((tick,stream,rate,value));return int(value)
            return float(args[0]+(args[1] if which=='normal' else args[1]-args[0])*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,0,tick,stream))
        env={name:physical.copy(),'mean':parameter(data,weights,'mean'),'spread':parameter(data,weights,'spread'),'gain':parameter(data,weights,'gain'),
             'dt':.0002,'ms':.001,'f':function(kind,capture,physical),'sqrt':np.sqrt,
             'randn':lambda:np.array([normal(p['seed'],9,0,domain,j,tick,0) for j in range(2)])}
        with patch.object(np.random,'normal',lambda *args,**kwargs:sample('normal',*args,**kwargs)),patch.object(np.random,'uniform',lambda *args,**kwargs:sample('uniform',*args,**kwargs)),patch.object(np.random,'poisson',lambda *args,**kwargs:sample('poisson',*args,**kwargs)):exec(stage,env)
        z[target]=env[name];margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());z[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples,events=events)


VJPS=[('normal',False,'euler',False,False,False),('normal',True,'rk4',False,False,False),
    ('poisson',False,'rk2',False,False,False),('poisson',False,'euler',False,False,True),
    ('chain',False,'rk2',False,False,False),('mixed',True,'rk4',False,False,False),
    ('normal',True,'rk4',True,False,False),('chain',False,'rk2',True,False,False),
    ('normal',False,'heun',False,True,False),('normal',False,'milstein',False,True,False)]


@pytest.mark.parametrize('kind,capture,method,synaptic,sde,zero',VJPS)
@pytest.mark.parametrize('window',[None,2])
def test_integrator_shared_all_bank_initial_vjps(engine,kind,capture,method,synaptic,sde,zero,window):
    mpi(2);data=model(engine,kind,capture,method,window,synaptic,sde);bundle=data[5];x=np.zeros((1,4,1))
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[3].name and 'mean' in row['variables']);mean=(row['bank'],row['variables'].index('mean'))
    if zero:
        bundle.weights[mean[0]][mean[1]]=0.;cell=bundle.provenance['neuron_state_layout'][data[2].name]['v'][0]
        for voltage in np.arange(.46,.5,.0002):
            bundle.initial_state[cell]=float(voltage);bundle.plan['dynamic']['initial'][cell]=float(voltage)
            loss,_,anchors=reference(data,kind,capture,method,bundle.weights,sde=sde)
            weak=sum(reference(data,kind,capture,method,bundle.weights,override=(t,s),sde=sde)[0]-loss for t,s,r,c in anchors['samples'])
            if abs(weak)>1e-5 and min(abs(m[0]) for m in anchors['margins'])>1e-4:break
        else:raise AssertionError('shared integrator zero rate requires a nonzero whole-block counterfactual')
    loss,z,anchors=reference(data,kind,capture,method,bundle.weights,sde=sde)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    layout=bundle.provenance['neuron_state_layout'][data[2].name];cells=layout['v']+(layout['q']+bundle.provenance['dynamic_state_layout'][data[3].name]['h'] if synaptic else [])
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,len(data[1]):],anchors['events'])
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            high=reference(data,kind,capture,method,hi,anchors=anchors,sde=sde);low=reference(data,kind,capture,method,lo,anchors=anchors,sde=sde)
            fd=(high[0]-low[0])/2e-6;score=0.
            for base,up,down in zip(anchors['samples'],high[2]['samples'],low[2]['samples']):
                tick,stream,rate,value=base;derivative=(up[2]-down[2])/2e-6
                signal=loss*(value/rate-1.) if rate else reference(data,kind,capture,method,bundle.weights,override=(tick,stream),sde=sde)[0]-loss
                score+=signal*derivative
            assert out['gradients'][bank][index]==pytest.approx(fd+score,rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(data,kind,capture,method,bundle.weights,initial=initial,sde=sde)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,kind,capture,method,bundle.weights,initial=hi,anchors=anchors,sde=sde)[0]-reference(data,kind,capture,method,bundle.weights,initial=lo,anchors=anchors,sde=sde)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),index


@pytest.mark.parametrize('kind,capture,method,synaptic',[
    ('normal',True,'rk4',False),('mixed',True,'rk4',False),
    ('poisson',False,'rk2',True),('chain',False,'rk2',True),
])
def test_integrator_shared_invalid_restore_atomic(engine,kind,capture,method,synaptic,tmp_path):
    mpi(2);data=model(engine,kind,capture,method,synaptic=synaptic);bundle=data[5];x=np.zeros((1,1,1))
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind in ('poisson','chain') else 'spread';row=next(row for row in bundle.provenance['bindings'] if row['object']==data[3].name and name in row['variables'])
    trainer.state['weights'][row['bank']][row['variables'].index(name)]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
