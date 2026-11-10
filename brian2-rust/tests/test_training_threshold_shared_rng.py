"""Scalar threshold callback distributions are sampled once per NumPy block."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
import test_training_scalar_event_rng_functions as factories
from test_training_threshold_callback_rng import THRESHOLDS
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_regular_callback_rng import count
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def scalar_chain(x,mean,spread):
    draw=np.random.normal(mean,spread)
    rate=mean+.2*draw
    return x+.1*np.random.poisson(rate)


def hybrid(x,mean,spread):
    draw=np.random.normal(0.,spread)
    return x+.1*draw+.1*np.random.poisson(mean)


def model(backend,kind,capture,pattern,window=None,refractory=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    mean_flags='constant' if kind=='hybrid' else 'constant,shared'
    g=b.NeuronGroup(2,f'dv/dt=gain/ms:1\na:1\nmean:1 ({mean_flags})\nspread:1 (constant,shared)\ngain:1 (constant)',method='euler',threshold=THRESHOLDS[pattern],reset='v-=.5',dt=dt,refractory=.4*b.ms if refractory else False)
    g.v=[.473,.719];g.a=[.317,.611];g.mean=1.3 if kind in ('poisson','mixed','chain') else .043;g.spread=.13;g.gain=[.217,.319]
    if kind=='hybrid':g.mean=[1.3,.7]
    if refractory:g.lastspike=[0.,-1.]*b.second;g.not_refractory=[False,True]
    function=hybrid if kind=='hybrid' else scalar_chain if kind=='chain' else getattr(factories,kind)(g.variables['a'].get_value()) if capture else getattr(factories,'pure_'+kind)
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    return net,g,dt,bundle


@pytest.mark.parametrize('kind,capture,pattern,refractory',[
    ('normal',False,'borrowed',False),('normal',True,'hidden',False),
    ('uniform',False,'right',False),('poisson',True,'borrowed',False),
    ('mixed',True,'predicate',False),('normal',True,'repeated',False),
    ('normal',True,'predicate_alias',False),('poisson',False,'borrowed',True),
    ('chain',False,'borrowed',False),
])
def test_shared_threshold_original_restore(engine,kind,capture,pattern,refractory,monkeypatch,tmp_path):
    mpi(2);net,g,dt,bundle=model(engine,kind,capture,pattern,refractory=refractory)
    monitor=b.SpikeMonitor(g);net.add(monitor)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    kinds=['normal','uniform','poisson'] if kind=='mixed' else ['normal','poisson'] if kind=='chain' else [kind];repetitions=2 if pattern in ('repeated','predicate','predicate_alias') else 1
    records=bundle.provenance['neuron_draws'][1]['threshold'];assert all(r['scalar'] for r in records)
    assert len(bundle.provenance['threshold_shared_draw_layout'][g.name])==len(kinds)*repetitions
    for tick in range(4):
        serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;assert which==kinds[stream%len(kinds)];serial+=1
            mean=float(g.mean[:]);spread=float(g.spread[:]);assert all(np.ndim(value)==0 for value in args)
            rate=mean+.2*(mean+spread*normal(p['seed'],9,0,1,0,tick,stream-1)) if kind=='chain' and which=='poisson' else mean
            np.testing.assert_allclose(args,(rate,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            if which=='poisson':return int(count(p['seed'],9,1,0,tick,rate,stream))
            return float(mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,1,0,tick,stream))
        for which in kinds:monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        assert serial==len(kinds)*repetitions
        for name in ('v','a'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,1:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    if 'poisson' in kinds:assert len(trainer.poisson_state['entries'])==4*repetitions
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def parameter(data,weights,name):
    g=data[1];bundle=data[3]
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
    return weights[row['bank']][row['variables'].index(name)] if row['kind']=='neuron' else np.asarray(weights[row['bank']])


def reference(data,kind,capture,pattern,weights,initial=None,anchors=None,override=None):
    import ast
    from unittest.mock import patch
    _,g,_,bundle=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];a=layout['a']
    comparison=ast.parse(THRESHOLDS[pattern],mode='eval').body;pieces=list(comparison.values) if isinstance(comparison,ast.BoolOp) else [comparison]
    before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*parameter(data,weights,'gain')
        env=dict(v=z[v].copy(),a=z[a].copy(),mean=parameter(data,weights,'mean'),spread=parameter(data,weights,'spread'))
        env['f']=scalar_chain if kind=='chain' else getattr(factories,kind)(env['a']) if capture else getattr(factories,'pure_'+kind);serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1
            if which=='poisson':
                rate=float(args[0]);value=count(p['seed'],9,1,0,tick,rate,stream)
                if anchors is not None:value=next(c for t,s,r,c in anchors['samples'] if (t,s)==(tick,stream))
                if override==(tick,stream):value+=1
                samples.append((tick,stream,rate,value));return int(value)
            raw=(normal if which=='normal' else uniform)(p['seed'],9,0,1,0,tick,stream)
            return float(args[0]+(args[1] if which=='normal' else args[1]-args[0])*raw)
        tick_margins=[];tick_hard=[];gates=[]
        with patch.object(np.random,'normal',lambda *args,**kwargs:sample('normal',*args,**kwargs)),patch.object(np.random,'uniform',lambda *args,**kwargs:sample('uniform',*args,**kwargs)),patch.object(np.random,'poisson',lambda *args,**kwargs:sample('poisson',*args,**kwargs)):
            for index,part in enumerate(pieces):
                left=eval(compile(ast.Expression(part.left),'<independent shared threshold lhs>','eval'),env)
                right=eval(compile(ast.Expression(part.comparators[0]),'<independent shared threshold rhs>','eval'),env)
                margin=right-left if isinstance(part.ops[0],ast.Lt) else left-right
                margin=np.broadcast_to(margin,(2,)).copy();gate=(margin>0).astype(float);tick_margins.append(margin);tick_hard.append(gate.copy())
                if anchors is not None:
                    old=anchors['margins'][tick][index]
                    gate=anchors['hard'][tick][index]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
                gates.append(gate)
        event=gates[0]+gates[1]-gates[0]*gates[1] if isinstance(comparison,ast.BoolOp) and isinstance(comparison.op,ast.Or) else np.prod(gates,axis=0)
        margins.append(tick_margins);hard.append(tick_hard);events.append(event.copy());z[a]=env['a'];z[v]=env['v']-.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples,events=events)


@pytest.mark.parametrize('kind,capture,pattern,zero',[
    ('normal',True,'borrowed',False),('uniform',False,'right',False),
    ('poisson',False,'borrowed',False),('poisson',True,'hidden',True),
    ('normal',True,'repeated',False),('mixed',True,'predicate',False),
    ('chain',False,'borrowed',False),
])
@pytest.mark.parametrize('window',[None,2])
def test_shared_threshold_all_bank_initial_vjps(engine,kind,capture,pattern,zero,window):
    mpi(2);data=model(engine,kind,capture,pattern,window);bundle=data[3];x=np.zeros((1,4,1))
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[1].name and 'mean' in row['variables']);mean=(row['bank'],row['variables'].index('mean'))
    if zero:
        bundle.weights[mean[0]][mean[1]]=0.;v=bundle.provenance['neuron_state_layout'][data[1].name]['v'][0]
        for voltage in np.arange(.35,.8,.005):
            bundle.initial_state[v]=float(voltage);bundle.plan['dynamic']['initial'][v]=float(voltage)
            loss,_,anchors=reference(data,kind,capture,pattern,bundle.weights)
            weak=sum(reference(data,kind,capture,pattern,bundle.weights,override=(t,s))[0]-loss for t,s,r,c in anchors['samples'])
            if abs(weak)>1e-5 and min(abs(m[0]) for tick in anchors['margins'] for m in tick)>1e-4:break
        else:raise AssertionError('shared zero-rate case needs a nonzero whole-block counterfactual')
    loss,z,anchors=reference(data,kind,capture,pattern,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    layout=bundle.provenance['neuron_state_layout'][data[1].name];cells=layout['v']+layout['a']
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],anchors['events'])
    weak=sum(loss*(c/r-1.) if r else reference(data,kind,capture,pattern,bundle.weights,override=(t,s))[0]-loss for t,s,r,c in anchors['samples'])
    if zero:assert abs(weak)>1e-5
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            high=reference(data,kind,capture,pattern,hi,anchors=anchors);low=reference(data,kind,capture,pattern,lo,anchors=anchors)
            fd=(high[0]-low[0])/2e-6
            score=0.
            for sample,high_sample,low_sample in zip(anchors['samples'],high[2]['samples'],low[2]['samples']):
                tick,stream,rate,value=sample
                derivative=(high_sample[2]-low_sample[2])/2e-6
                signal=loss*(value/rate-1.) if rate else reference(data,kind,capture,pattern,bundle.weights,override=(tick,stream))[0]-loss
                score+=signal*derivative
            assert out['gradients'][bank][index]==pytest.approx(fd+score,rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float)
    explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(data,kind,capture,pattern,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,kind,capture,pattern,bundle.weights,initial=hi,anchors=anchors)[0]-reference(data,kind,capture,pattern,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),index


@pytest.mark.parametrize('kind',['normal','mixed','poisson','chain'])
def test_shared_threshold_invalid_restore_atomic(engine,kind,tmp_path):
    mpi(2);data=model(engine,kind,kind in ('normal','mixed'),'borrowed');bundle=data[3];x=np.zeros((1,1,1))
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind in ('poisson','chain') else 'spread';row=next(row for row in bundle.provenance['bindings'] if row['object']==data[1].name and name in row['variables'])
    trainer.state['weights'][row['bank']][row['variables'].index(name)]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before


def test_shared_normal_and_array_poisson_original_restore(engine,monkeypatch,tmp_path):
    mpi(2);net,g,dt,bundle=model(engine,'hybrid',False,'borrowed')
    monitor=b.SpikeMonitor(g);net.add(monitor)
    records=bundle.provenance['neuron_draws'][1]['threshold']
    assert records[0]['scalar'] and not records[1].get('scalar',False)
    assert len(bundle.provenance['threshold_shared_draw_layout'][g.name])==1
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        calls=[]
        def scalar(loc,spread):
            assert np.ndim(loc)==np.ndim(spread)==0 and loc==0.;calls.append('normal')
            return float(spread*normal(p['seed'],9,0,1,0,tick,0))
        def vector(rate):
            np.testing.assert_allclose(rate,g.mean[:]);calls.append('poisson')
            return np.array([count(p['seed'],9,1,j,tick,float(rate[j]),1) for j in range(2)])
        monkeypatch.setattr(np.random,'normal',scalar);monkeypatch.setattr(np.random,'poisson',vector)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        assert calls==['normal','poisson']
        cells=bundle.provenance['neuron_state_layout'][g.name]['v']
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.v[:],rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,1:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(trainer.poisson_state['entries'])==8
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
