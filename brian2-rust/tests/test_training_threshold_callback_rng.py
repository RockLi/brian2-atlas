"""Threshold array RNG preserves borrowed operands and ordered effects."""
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

THRESHOLDS={
    'borrowed':'f(v,mean,spread)>.5',
    'hidden':'v+.2*f(a,mean,spread)>.5',
    'right':'v>f(a,mean,spread)',
    'left_reference':'v>f(v,mean,spread)',
    'left_value':'2*v>f(v,mean,spread)',
    'less_reference':'v<f(v,mean,spread)',
    'repeated':'f(v,mean,spread)>f(v,mean,spread)',
    'predicate':'(f(v,mean,spread)>.5) and (f(a,mean,spread)>.4)',
    'predicate_alias':'(v>f(v,mean,spread)) or (f(a,mean,spread)>.4)',
}


def model(backend,kind,capture,pattern,window=None,refractory=False,phases=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    update='dv/dt=(gain'+('+.01*f(a,mean,spread)' if phases else '')+')'
    g=b.NeuronGroup(2,update+'/ms:1\na:1\nmean:1 (constant)\nspread:1 (constant)\ngain:1 (constant)',method='euler',threshold=THRESHOLDS[pattern],reset='v-=.5'+('+.01*f(a,mean,spread)' if phases else ''),dt=dt,refractory=.4*b.ms if refractory else False)
    g.v=[.473,.719];g.a=[.317,.611];g.mean=[1.3,.7] if kind in ('poisson','mixed') else [.043,.077];g.spread=[.13,.11];g.gain=[.217,.319]
    if refractory:g.lastspike=[0.,-1.]*b.second;g.not_refractory=[False,True]
    function=getattr(factories,kind)(g.variables['a'].get_value()) if capture else getattr(factories,'pure_'+kind)
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    return net,g,dt,bundle


@pytest.mark.parametrize('kind,capture,pattern,refractory',[
    ('normal',False,'borrowed',False),('normal',True,'hidden',False),
    ('uniform',False,'right',False),('poisson',False,'left_value',False),
    ('mixed',True,'borrowed',False),('normal',True,'left_reference',False),
    ('normal',True,'less_reference',False),('normal',True,'repeated',False),
    ('normal',True,'borrowed',True),('mixed',True,'predicate',False),
    ('normal',True,'predicate_alias',False),
])
def test_threshold_callback_original_restore(engine,kind,capture,pattern,refractory,monkeypatch,tmp_path):
    mpi(2);net,g,dt,bundle=model(engine,kind,capture,pattern,refractory=refractory)
    monitor=b.SpikeMonitor(g);net.add(monitor)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind]
    repetitions=2 if pattern in ('repeated','predicate','predicate_alias') else 1;calls=[]
    assert [row['stream'] for row in bundle.provenance['neuron_draws'][1]['threshold']]==list(range(repetitions*len(kinds)))
    for tick in range(4):
        serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;assert which==kinds[serial%len(kinds)];serial+=1
            mean=np.asarray(g.mean[:]);spread=np.asarray(g.spread[:])
            np.testing.assert_allclose(args,(mean,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread));calls.append((tick,stream,which))
            if which=='poisson':return np.array([count(p['seed'],9,1,j,tick,float(mean[j]),stream) for j in range(2)])
            raw=np.array([(normal if which=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(2)])
            return mean+spread*raw
        for which in kinds:monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','a'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,1:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4*len(kinds)*repetitions
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,kind,capture,pattern,weights,initial=None,anchors=None):
    import ast
    from unittest.mock import patch
    _,g,_,bundle=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];a=layout['a']
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and row['variables']==[name])
        return np.asarray(weights[row['bank']])
    comparison=ast.parse(THRESHOLDS[pattern],mode='eval').body
    pieces=list(comparison.values) if isinstance(comparison,ast.BoolOp) else [comparison]
    before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*parameter('gain')
        env=dict(v=z[v].copy(),a=z[a].copy(),mean=parameter('mean'),spread=parameter('spread'))
        env['f']=getattr(factories,kind)(env['a']) if capture else getattr(factories,'pure_'+kind);serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1
            if which=='poisson':
                values=[]
                for j,rate in enumerate(args[0]):
                    value=count(p['seed'],9,1,j,tick,float(rate),stream)
                    if anchors is not None:value=next(c for t,s,k,r,c in anchors['samples'] if (t,s,k)==(tick,stream,j))
                    samples.append((tick,stream,j,float(rate),value));values.append(value)
                return np.array(values)
            raw=np.array([(normal if which=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(2)])
            return args[0]+(args[1] if which=='normal' else args[1]-args[0])*raw
        tick_margins=[];tick_hard=[];gates=[]
        with patch.object(np.random,'normal',lambda *args,**kwargs:sample('normal',*args,**kwargs)),patch.object(np.random,'uniform',lambda *args,**kwargs:sample('uniform',*args,**kwargs)),patch.object(np.random,'poisson',lambda *args,**kwargs:sample('poisson',*args,**kwargs)):
            for index,part in enumerate(pieces):
                left=eval(compile(ast.Expression(part.left),'<independent threshold lhs>','eval'),env)
                right=eval(compile(ast.Expression(part.comparators[0]),'<independent threshold rhs>','eval'),env)
                margin=right-left if isinstance(part.ops[0],ast.Lt) else left-right
                margin=np.broadcast_to(margin,(2,)).copy();gate=(margin>0).astype(float)
                tick_margins.append(margin);tick_hard.append(gate.copy())
                if anchors is not None:
                    old=anchors['margins'][tick][index]
                    gate=anchors['hard'][tick][index]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
                gates.append(gate)
        event=gates[0]+gates[1]-gates[0]*gates[1] if isinstance(comparison,ast.BoolOp) and isinstance(comparison.op,ast.Or) else np.prod(gates,axis=0)
        margins.append(tick_margins);hard.append(tick_hard);events.append(event.copy());z[a]=env['a'];z[v]=env['v']-.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples,events=events)


@pytest.mark.parametrize('kind,capture,pattern',[
    ('normal',True,'borrowed'),('uniform',False,'right'),('poisson',False,'left_value'),
    ('normal',True,'left_reference'),('normal',True,'repeated'),('mixed',True,'predicate'),
    ('normal',True,'predicate_alias'),
])
@pytest.mark.parametrize('window',[None,2])
def test_threshold_callback_all_bank_initial_vjps(engine,kind,capture,pattern,window):
    mpi(2);data=model(engine,kind,capture,pattern,window);bundle=data[3];x=np.zeros((1,4,1))
    loss,z,anchors=reference(data,kind,capture,pattern,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    layout=bundle.provenance['neuron_state_layout'][data[1].name];cells=layout['v']+layout['a']
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],anchors['events'])
    mean=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==data[1].name and row['variables']==['mean'])
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,kind,capture,pattern,hi,anchors=anchors)[0]-reference(data,kind,capture,pattern,lo,anchors=anchors)[0])/2e-6
            score=sum(loss*(c/r-1.) for _,_,j,r,c in anchors['samples'] if j==index) if bank==mean else 0.
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


@pytest.mark.parametrize('kind,capture',[('normal',False),('mixed',True)])
def test_threshold_callback_three_phase_streams(engine,kind,capture,monkeypatch,tmp_path):
    mpi(2);net,g,dt,bundle=model(engine,kind,capture,'borrowed',phases=True)
    monitor=b.SpikeMonitor(g);net.add(monitor)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind];n=len(kinds)
    records=bundle.provenance['neuron_draws'][1]
    assert [r['stream'] for r in records['threshold']]==list(range(n))
    assert [r['stream'] for r in records['update']]==list(range(n,2*n))
    assert [r['stream'] for r in records['reset']]==list(range(2*n,3*n))
    for tick in range(4):
        serial=0;order=[*range(n,2*n),*range(n),*range(2*n,3*n)]
        def sample(which,*args,**kwargs):
            nonlocal serial
            stream=order[serial];serial+=1;assert which==kinds[stream%n]
            mean=np.asarray(args[0]);full_mean=np.asarray(g.mean[:]);spread=np.asarray(g.spread[:])
            indices=np.array([np.flatnonzero(full_mean==value)[0] for value in mean],dtype=int)
            np.testing.assert_allclose(args,(full_mean[indices],) if which=='poisson' else (full_mean[indices],spread[indices] if which=='normal' else full_mean[indices]+spread[indices]))
            if which=='poisson':return np.array([count(p['seed'],9,1,int(j),tick,float(full_mean[j]),stream) for j in indices])
            raw=np.array([(normal if which=='normal' else uniform)(p['seed'],9,0,1,int(j),tick,stream) for j in indices])
            return full_mean[indices]+spread[indices]*raw
        for which in kinds:monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        assert serial==len(order)
        for name in ('v','a'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,1:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('kind',['normal','mixed','poisson'])
def test_threshold_callback_invalid_restore_atomic(engine,kind,tmp_path):
    mpi(2);data=model(engine,kind,kind!='poisson','borrowed');bundle=data[3];x=np.zeros((1,1,1))
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind=='poisson' else 'spread';row=next(row for row in bundle.provenance['bindings'] if row['object']==data[1].name and row['variables']==[name])
    trainer.state['weights'][row['bank']][1]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
