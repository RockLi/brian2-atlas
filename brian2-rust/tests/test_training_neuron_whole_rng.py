"""Complete foreign capture columns in native neuron NumPy blocks."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_whole_size_rng import normal_factory,uniform_factory,poisson_factory,derived_factory
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_regular_callback_rng import count
from test_native_training import RUNNER


def model(backend,phase,kind,method='euler',window=None,threshold_operator='>'):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    storage=b.NeuronGroup(4,'dv/dt=0/second:1\na:1',threshold='v>100',reset='v=0',method='euler',dt=dt);storage.a=[.1,.2,.3,.4]
    update='gain+.1*f(v,k,mean,spread)' if phase=='update' else 'gain'
    threshold='f(v,k,mean,spread)'+threshold_operator+'.5' if phase=='threshold' else 'v>.5'
    reset='v=.2*f(v,k,mean,spread)+v-.5' if phase=='reset' else 'v-=.5'
    g=b.NeuronGroup(2,'dv/dt=('+update+')/ms:1\nk:integer\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant)',threshold=threshold,reset=reset,method=method,dt=dt);g.v=[.473,.719];g.k=[3,0];g.mean=1.3 if kind in ('poisson','derived') else .043;g.spread=.13;g.gain=[.217,.319]
    function=globals()[kind+'_factory'](storage.variables['a'].get_value());g.namespace['f']=b.Function(function,arg_units=[1,1,1,1],arg_types=['float','integer','float','float'],return_unit=1,stateless=False)
    net=b.Network(source,storage,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[storage,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    return net,storage,g,dt,bundle


@pytest.mark.parametrize('phase',['threshold','update','reset'])
@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
def test_neuron_whole_original_restore(engine,phase,kind,monkeypatch,tmp_path):
    mpi(2);net,storage,g,dt,bundle=model(engine,phase,kind)
    monitor=b.SpikeMonitor(g);net.add(monitor);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    calls=[]
    for tick in range(4):
        def sample(*args,**kwargs):
            assert kwargs['size']==4;calls.append(tick);mean=float(g.mean[:]);spread=float(g.spread[:]);rate=mean+np.asarray(storage.a[:]) if kind=='derived' else mean
            np.testing.assert_allclose(args,(rate,) if kind in ('poisson','derived') else (mean,spread if kind=='normal' else mean+spread))
            if kind in ('poisson','derived'):return np.array([count(p['seed'],9,1,j,tick,float(np.broadcast_to(rate,(4,))[j]),0) for j in range(4)])
            return mean+spread*np.array([(normal if kind=='normal' else uniform)(p['seed'],9,0,1,j,tick,0) for j in range(4)])
        monkeypatch.setattr(np.random,'poisson' if kind=='derived' else kind,sample)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for obj,names in [(storage,('v','a')),(g,('v','k'))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)];np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,4:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4
    if kind in ('poisson','derived'):assert len(trainer.poisson_state['entries'])==16


def threshold_reference(data,kind,weights,initial=None,anchors=None,override=None):
    _,storage,g,_,bundle=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
        return weights[row['bank']][row['variables'].index(name)] if g.variables[name].scalar else np.asarray(weights[row['bank']])
    layout=bundle.provenance['neuron_state_layout'];a=layout[storage.name]['a'];v=layout[g.name]['v'];k=layout[g.name]['k'];before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*parameter('gain');z[a]*=.8
        mean=parameter('mean');spread=parameter('spread')
        if kind in ('poisson','derived'):
            rates=np.broadcast_to(mean,(4,))+(z[a] if kind=='derived' else 0.);draws=[]
            for j,rate in enumerate(rates):
                value=anchors['samples'][tick*4+j][3] if anchors is not None else count(p['seed'],9,1,j,tick,float(rate),0)
                if override==(tick,j):value+=1
                samples.append((tick,j,float(rate),value));draws.append(value)
            draws=np.array(draws)
        else:draws=mean+spread*np.array([(normal if kind=='normal' else uniform)(p['seed'],9,0,1,j,tick,0) for j in range(4)])
        margin=z[v]+.1*draws[z[k].astype(int)]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());z[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,events=events,samples=samples)


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('window',[None,2])
def test_neuron_whole_threshold_all_bank_initial_vjps(engine,kind,window):
    mpi(2);data=model(engine,'threshold',kind,window=window);bundle=data[4];x=np.zeros((1,4,1));loss,z,anchors=threshold_reference(data,kind,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9);assert out['loss']==pytest.approx(loss,abs=8e-6)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['a'],*bundle.provenance['neuron_state_layout'][data[2].name]['v']];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,4:],anchors['events'])
    def expected(high,low):
        fd=(high[0]-low[0])/2e-6;score=0.
        for base,up,down in zip(anchors['samples'],high[2]['samples'],low[2]['samples']):
            tick,j,rate,value=base;derivative=(up[2]-down[2])/2e-6
            signal=loss*(value/rate-1.) if rate else threshold_reference(data,kind,bundle.weights,override=(tick,j))[0]-loss;score+=signal*derivative
        return fd+score
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            high=threshold_reference(data,kind,hi,anchors=anchors);low=threshold_reference(data,kind,lo,anchors=anchors);assert out['gradients'][bank][index]==pytest.approx(expected(high,low),rel=3e-3,abs=2e-5),(kind,bank,index)
    initial=np.array(bundle.initial_state,dtype=float);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9);loss,_,anchors=threshold_reference(data,kind,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6;high=threshold_reference(data,kind,bundle.weights,initial=hi,anchors=anchors);low=threshold_reference(data,kind,bundle.weights,initial=lo,anchors=anchors)
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(expected(high,low),rel=3e-3,abs=2e-5),(kind,index)


@pytest.mark.parametrize('kind',['normal','poisson'])
def test_neuron_whole_threshold_invalid_index_atomic(engine,kind,tmp_path):
    mpi(2);data=model(engine,'threshold',kind);bundle=data[4];p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);x=np.zeros((1,1,1));trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path);cell=bundle.provenance['neuron_state_layout'][data[2].name]['k'][1];trainer.neuron_state[0][cell]=4.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    for _ in range(2):
        with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
        assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before


@pytest.mark.parametrize('window',[None,2])
def test_neuron_whole_threshold_zero_state_score(engine,window,monkeypatch):
    mpi(2);data=model(engine,'threshold','derived',window=window);bundle=data[4];g=data[2]
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and 'mean' in row['variables']);bundle.weights[row['bank']][row['variables'].index('mean')]=0.
    rate_cell=bundle.provenance['neuron_state_layout'][data[1].name]['a'][3];bundle.initial_state[rate_cell]=0.;bundle.plan['dynamic']['initial'][rate_cell]=0.
    v=bundle.provenance['neuron_state_layout'][g.name]['v'][0]
    for voltage in np.arange(.1,.6,.001):
        bundle.initial_state[v]=float(voltage);bundle.plan['dynamic']['initial'][v]=float(voltage)
        loss,_,anchors=threshold_reference(data,'derived',bundle.weights)
        weak=sum(threshold_reference(data,'derived',bundle.weights,override=(t,j))[0]-loss for t,j,r,c in anchors['samples'] if r==0.)
        if abs(weak)>1e-5 and min(abs(m[0]) for m in anchors['margins'])>1e-5:break
    else:raise AssertionError('threshold zero-rate count must have nonzero replay')
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:data)
    test_neuron_whole_threshold_all_bank_initial_vjps(engine,'derived',window)


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
def test_neuron_whole_threshold_bad_domain_atomic(engine,kind,tmp_path):
    mpi(2);data=model(engine,'threshold',kind);bundle=data[4];p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);x=np.zeros((1,1,1));trainer.step(x,[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind in ('poisson','derived') else 'spread';row=next(row for row in bundle.provenance['bindings'] if row['object']==data[2].name and name in row['variables']);trainer.state['weights'][row['bank']][row['variables'].index(name)]=float('inf') if kind=='uniform' else -2.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    for _ in range(2):
        with pytest.raises(ValueError):trainer.step(x,[0],initial='carry')
        assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('operator',['>=','<','<='])
def test_neuron_whole_threshold_ordered_original(engine,kind,operator,monkeypatch,tmp_path):
    original=model
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:original(*args,threshold_operator=operator,**kwargs))
    test_neuron_whole_original_restore(engine,'threshold',kind,monkeypatch,tmp_path)
