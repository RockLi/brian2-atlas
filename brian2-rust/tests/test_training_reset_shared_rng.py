"""Reset callbacks draw once for a NumPy block, including empty selections."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integrator_shared_rng import function
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_regular_callback_rng import count
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(backend,kind,capture,selection,window=None,prefix=False,alias=False,refractory=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    reset=('v+=.01*randn()\n' if prefix else '')+'v=.2*f(a,mean,spread)+v-.5'
    g=b.NeuronGroup(2,'dv/dt=gain/ms:1\na:1\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant)',method='euler',threshold='v>.5',reset=reset,dt=dt,refractory=.4*b.ms if refractory else False)
    g.v={'subset':[.103,.713],'all':[.713,.937],'empty':[-4.013,-3.017]}[selection]
    g.a=[.3,.6];g.mean=1.3 if kind in ('poisson','mixed','chain') else .043;g.spread=.13;g.gain=[.217,.319]
    g.namespace['f']=b.Function(function(kind,capture,g.variables['v' if alias else 'a'].get_value()),arg_units=[1,1,1],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    return net,g,dt,bundle


PROFILES=[('normal',False,'empty',False,False,False),('normal',True,'subset',False,False,False),
 ('normal',True,'empty',False,False,False),('uniform',False,'all',False,False,False),
 ('poisson',True,'empty',False,False,False),('poisson',False,'subset',False,False,False),
 ('mixed',True,'all',False,False,False),('chain',False,'subset',False,False,False),
 ('normal',True,'subset',True,False,False),('normal',True,'subset',False,True,False),
 ('poisson',False,'subset',False,False,True)]


@pytest.mark.parametrize('kind,capture,selection,prefix,alias,refractory',PROFILES)
def test_reset_shared_original_restore(engine,kind,capture,selection,prefix,alias,refractory,monkeypatch,tmp_path):
    mpi(2);net,g,dt,bundle=model(engine,kind,capture,selection,prefix=prefix,alias=alias,refractory=refractory)
    monitor=b.SpikeMonitor(g);net.add(monitor)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    kinds=['normal','uniform','poisson'] if kind=='mixed' else ['normal','poisson'] if kind=='chain' else [kind]
    records=[row for row in bundle.provenance['neuron_draws'][1]['reset'] if row.get('scalar',False)]
    assert len(records)==len(kinds);assert len(bundle.provenance['reset_shared_draw_layout'][g.name])==len(kinds)
    for tick in range(4):
        serial=int(prefix)
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1;assert which==kinds[(stream-int(prefix))%len(kinds)]
            assert all(np.ndim(value)==0 for value in args)
            mean=float(g.mean[:]);spread=float(g.spread[:])
            rate=mean+.2*(mean+spread*normal(p['seed'],9,0,1,0,tick,stream-1)) if kind=='chain' and which=='poisson' else mean
            np.testing.assert_allclose(args,(rate,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            if which=='poisson':return int(count(p['seed'],9,1,0,tick,rate,stream))
            return float(mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,1,0,tick,stream))
        for which in kinds:monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        if prefix:
            selected=np.flatnonzero(np.asarray(g.v[:])+.2*np.asarray(g.gain[:])>.5)
            def standard(n):
                assert n==len(selected)
                return np.array([normal(p['seed'],9,0,1,int(j),tick,0) for j in selected])
            monkeypatch.setattr(np.random,'randn',standard)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        assert serial==int(prefix)+len(kinds)
        for name in ('v','a'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,1:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    if 'poisson' in kinds:assert len(trainer.poisson_state['entries'])==4
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def parameter(data,weights,name):
    row=next(row for row in data[3].provenance['bindings'] if row['object']==data[1].name and name in row['variables'])
    return weights[row['bank']][row['variables'].index(name)] if data[1].variables[name].scalar else np.asarray(weights[row['bank']])


def reference(data,kind,capture,weights,initial=None,anchors=None,override=None):
    from unittest.mock import patch
    _,g,_,bundle=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];a=layout['a'];before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*parameter(data,weights,'gain');margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());physical=z[a].copy();serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1
            if which=='poisson':
                rate=float(args[0]);value=count(p['seed'],9,1,0,tick,rate,stream)
                if anchors is not None:value=next(c for t,s,r,c in anchors['samples'] if (t,s)==(tick,stream))
                if override==(tick,stream):value+=1
                samples.append((tick,stream,rate,value));return int(value)
            return float(args[0]+(args[1] if which=='normal' else args[1]-args[0])*(normal if which=='normal' else uniform)(p['seed'],9,0,1,0,tick,stream))
        with patch.object(np.random,'normal',lambda *args,**kwargs:sample('normal',*args,**kwargs)),patch.object(np.random,'uniform',lambda *args,**kwargs:sample('uniform',*args,**kwargs)),patch.object(np.random,'poisson',lambda *args,**kwargs:sample('poisson',*args,**kwargs)):
            candidate=.2*function(kind,capture,physical)(z[a].copy(),parameter(data,weights,'mean'),parameter(data,weights,'spread'))+z[v]-.5
        if capture:z[a]=physical
        z[v]+=event*(candidate-z[v])
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples,events=events)


@pytest.mark.parametrize('kind,capture,selection,zero',[
 ('normal',True,'subset',False),('uniform',False,'all',False),('poisson',False,'subset',False),
 ('mixed',True,'subset',False),('chain',False,'subset',False),('poisson',False,'subset',True),
 ('poisson',True,'empty',False)])
@pytest.mark.parametrize('window',[None,2])
def test_reset_shared_all_bank_initial_vjps(engine,kind,capture,selection,zero,window):
    mpi(2);data=model(engine,kind,capture,selection,window);bundle=data[3];x=np.zeros((1,4,1))
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[1].name and 'mean' in row['variables']);mean=(row['bank'],row['variables'].index('mean'))
    if zero:
        bundle.weights[mean[0]][mean[1]]=0.;cell=bundle.provenance['neuron_state_layout'][data[1].name]['v'][0]
        for voltage in np.arange(.65,.85,.0005):
            bundle.initial_state[cell]=float(voltage);bundle.plan['dynamic']['initial'][cell]=float(voltage)
            loss,_,anchors=reference(data,kind,capture,bundle.weights)
            weak=sum(reference(data,kind,capture,bundle.weights,override=(t,s))[0]-loss for t,s,r,c in anchors['samples'])
            if abs(weak)>1e-5 and min(abs(m[0]) for m in anchors['margins'])>1e-4:break
        else:raise AssertionError('shared reset zero rate must have nonzero whole-block replay')
    loss,z,anchors=reference(data,kind,capture,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    layout=bundle.provenance['neuron_state_layout'][data[1].name];cells=layout['v']+layout['a']
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],anchors['events'])
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            high=reference(data,kind,capture,hi,anchors=anchors);low=reference(data,kind,capture,lo,anchors=anchors)
            fd=(high[0]-low[0])/2e-6;score=0.
            for base,up,down in zip(anchors['samples'],high[2]['samples'],low[2]['samples']):
                tick,stream,rate,value=base;derivative=(up[2]-down[2])/2e-6
                signal=loss*(value/rate-1.) if rate else reference(data,kind,capture,bundle.weights,override=(tick,stream))[0]-loss
                score+=signal*derivative
            assert out['gradients'][bank][index]==pytest.approx(fd+score,rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(data,kind,capture,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,kind,capture,bundle.weights,initial=hi,anchors=anchors)[0]-reference(data,kind,capture,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),index


@pytest.mark.parametrize('kind,selection',[('normal','empty'),('poisson','empty'),('mixed','subset'),('chain','subset')])
def test_reset_shared_invalid_restore_atomic(engine,kind,selection,tmp_path):
    mpi(2);data=model(engine,kind,kind!='chain',selection);bundle=data[3];x=np.zeros((1,1,1));p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind in ('poisson','chain') else 'spread';row=next(row for row in bundle.provenance['bindings'] if row['object']==data[1].name and name in row['variables']);trainer.state['weights'][row['bank']][row['variables'].index(name)]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    for _ in range(2):
        with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
        assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
