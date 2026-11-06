"""Complete reset callback with compact empty/partial/full caller arrays."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
import test_training_neuron_whole_rng as base
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_regular_callback_rng import count
from test_native_training import RUNNER


def model(backend,kind,selection='partial',window=None,refractory=False,gain=.02,detach=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    storage=b.NeuronGroup(4,'dv/dt=0/second:1\na:1',threshold='v>100',reset='v=0',method='euler',dt=dt);storage.a=[.1,.2,.3,.4]
    g=b.NeuronGroup(2,'dv/dt=gain/ms:1'+(' (unless refractory)' if refractory else '')+'\nk:integer\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant)',threshold='v>.5',reset='v=.2*f(v,k,mean,spread)+v-.5',refractory=.4*b.ms if refractory else False,method='euler',dt=dt)
    g.v={'empty':[.1,.2],'partial':[.1,.7],'full':[.6,.7]}[selection];g.k=[3,0];g.mean=1.3 if kind in ('poisson','derived') else .043;g.spread=.13;g.gain=gain
    g.namespace['f']=b.Function(getattr(base,kind+'_factory')(storage.variables['a'].get_value()),arg_units=[1,1,1,1],arg_types=['float','integer','float','float'],return_unit=1,stateless=False)
    net=b.Network(source,storage,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[storage,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=detach,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    return net,storage,g,dt,bundle


@pytest.mark.parametrize('selection',['empty','partial','full'])
@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('refractory',[False,True])
def test_whole_reset_original_restore(engine,selection,kind,refractory,monkeypatch,tmp_path):
    data=model(engine,kind,selection,refractory=refractory)
    monkeypatch.setattr(base,'model',lambda *args,**kwargs:data)
    base.test_neuron_whole_original_restore(engine,'reset',kind,monkeypatch,tmp_path)


def reference(data,kind,weights,initial=None,anchors=None,override=None,implicit=False,scalar=False):
    _,storage,g,_,bundle=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'];a=layout[storage.name]['a'];v=layout[g.name]['v'];k=layout[g.name]['k'];counter=layout[g.name].get('__refractory_ticks',[])
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
        return weights[row['bank']][row['variables'].index(name)] if g.variables[name].scalar else np.asarray(weights[row['bank']])
    stream=bundle.provenance['whole_neuron_reset_layout'][g.name]['draws'][0]['stream']
    before=[];margins=[];hard=[];samples=[];events=[]
    for tick in range(6):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());active=z[counter]==0 if counter else np.ones(2,dtype=bool)
        z[v]+=.2*parameter('gain')*active
        if counter:z[counter]=np.maximum(z[counter]-1,0)
        margin=z[v]-.5;event=((margin>0)&active).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+active*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());z[a]*=.8;mean=parameter('mean');spread=parameter('spread')
        if kind in ('poisson','derived'):
            rates=(z[v][hard[-1].astype(bool)]+mean) if implicit else np.broadcast_to(mean,(1 if scalar else 4,))+(z[a] if kind=='derived' else 0.);draws=[]
            for j,rate in enumerate(rates):
                value=next(row[3] for row in anchors['samples'] if row[:2]==(tick,j)) if anchors is not None else count(p['seed'],9,1,j,tick,float(rate),stream)
                if override==(tick,j):value+=1
                samples.append((tick,j,float(rate),value));draws.append(value)
            draws=np.array(draws)
            if implicit:
                selected=hard[-1].astype(bool);expanded=np.zeros(2);expanded[selected]=draws;draws=expanded
        elif implicit:
            selection=hard[-1] if anchors is None else anchors['hard'][tick]
            ordinals=np.array([int(selection[:j].sum()) for j in range(2)])
            draws=z[v]+mean+spread*np.array([normal(p['seed'],9,0,1,int(q),tick,stream) for q in ordinals])
        else:draws=mean+spread*np.array([(normal if kind=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(1 if scalar else 4)])
        # The sampler and closure execute once even for an empty selection;
        # the pointwise reset jump uses the same cached draw for every candidate.
        reset_event=hard[-1] if p['detach_reset'] else event
        z[v]+=reset_event*(.2*z[v]+.02*(draws if implicit else draws[0] if scalar else draws[z[k].astype(int)])-.5)
        if counter:
            control=hard[-1] if anchors is None else anchors['hard'][tick]
            z[counter]=np.where(control.astype(bool),max(p['refractory'][1]['steps']-1,0),z[counter])
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples,events=events)


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('refractory',[False,True])
def test_whole_reset_all_bank_initial_vjps(engine,kind,window,refractory):
    mpi(2);data=model(engine,kind,'partial',window,refractory,gain=.7);bundle=data[4];x=np.zeros((1,6,1))
    loss,z,anchors=reference(data,kind,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['a'],*bundle.provenance['neuron_state_layout'][data[2].name]['v']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    def expected(hi,lo,baseline,base_loss):
        gradient=(hi[0]-lo[0])/2e-6;score=0.
        for sample,up,down in zip(baseline['samples'],hi[2]['samples'],lo[2]['samples']):
            tick,j,rate,value=sample
            signal=base_loss*(value/rate-1.) if rate else reference(data,kind,bundle.weights,override=(tick,j))[0]-base_loss
            score+=signal*(up[2]-down[2])/2e-6
        return gradient+score
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            high=reference(data,kind,hi,anchors=anchors);low=reference(data,kind,lo,anchors=anchors)
            assert out['gradients'][bank][index]==pytest.approx(expected(high,low,anchors,loss),rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,float)
    explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    loss,_,anchors=reference(data,kind,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        high=reference(data,kind,bundle.weights,initial=hi,anchors=anchors);low=reference(data,kind,bundle.weights,initial=lo,anchors=anchors)
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(expected(high,low,anchors,loss),rel=3e-3,abs=2e-5),index


@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('refractory',[False,True])
def test_whole_reset_zero_state_score(engine,window,refractory,monkeypatch):
    gain=.96 if refractory else .65
    data=model(engine,'derived','partial',window,refractory,gain=gain);bundle=data[4]
    row=next(row for row in bundle.provenance['bindings'] if row['object']==data[2].name and 'mean' in row['variables'])
    bundle.weights[row['bank']][row['variables'].index('mean')]=0.
    cell=bundle.provenance['neuron_state_layout'][data[1].name]['a'][3]
    bundle.initial_state[cell]=0.;bundle.plan['dynamic']['initial'][cell]=0.
    voltage=bundle.provenance['neuron_state_layout'][data[2].name]['v'][0]
    value=.504-.6*gain
    bundle.initial_state[voltage]=value;bundle.plan['dynamic']['initial'][voltage]=value
    loss,_,anchors=reference(data,'derived',bundle.weights)
    assert abs(reference(data,'derived',bundle.weights,override=(2,3))[0]-loss)>1e-5
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:data)
    test_whole_reset_all_bank_initial_vjps(engine,'derived',window,refractory)


@pytest.mark.parametrize('kind',['normal','derived'])
@pytest.mark.parametrize('window',[None,2])
def test_whole_reset_detached_vjps(engine,kind,window,monkeypatch):
    original=model
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:original(*args,detach=True,**kwargs))
    test_whole_reset_all_bank_initial_vjps(engine,kind,window,False)


def implicit_normal_factory(array):
    def f(x,k,mean,spread):
        a=array
        a*=.8
        return x+.1*np.random.normal(x+mean,spread)
    return f


@pytest.mark.parametrize('selection',['empty','partial','full'])
def test_whole_reset_implicit_normal_original(engine,selection,monkeypatch,tmp_path):
    mpi(2);monkeypatch.setattr(base,'normal_factory',implicit_normal_factory)
    net,storage,g,dt,bundle=model(engine,'normal',selection)
    monitor=b.SpikeMonitor(g);net.add(monitor)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    calls=[]
    for tick in range(4):
        def sample(mean,spread):
            mean=np.asarray(mean);calls.append(len(mean))
            return mean+spread*np.array([normal(p['seed'],9,0,1,q,tick,0) for q in range(len(mean))])
        monkeypatch.setattr(np.random,'normal',sample)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}))
        net.run(dt,namespace={})
        for obj,names in [(storage,('a',)),(g,('v','k'))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        assert calls[-1]==int(np.asarray(out['spikes'])[0,-1,4:].sum())
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4


@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('refractory',[False,True])
def test_whole_reset_implicit_normal_vjps(engine,window,refractory,monkeypatch):
    monkeypatch.setattr(base,'normal_factory',implicit_normal_factory)
    original=reference
    monkeypatch.setattr(__import__(__name__),'reference',lambda *args,**kwargs:original(*args,implicit=True,**kwargs))
    test_whole_reset_all_bank_initial_vjps(engine,'normal',window,refractory)


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('selection',['empty','partial'])
def test_whole_reset_bad_domain_atomic(engine,kind,selection,tmp_path,monkeypatch):
    data=model(engine,kind,selection,refractory=True)
    monkeypatch.setattr(base,'model',lambda *args,**kwargs:data)
    base.test_neuron_whole_threshold_bad_domain_atomic(engine,kind,tmp_path)


def scalar_normal_factory(array):
    def f(x,k,mean,spread):
        a=array
        a*=.8
        return x+.1*np.random.normal(mean,spread)
    return f


def scalar_uniform_factory(array):
    def f(x,k,mean,spread):
        a=array
        a*=.8
        return x+.1*np.random.uniform(mean,mean+spread)
    return f


def scalar_poisson_factory(array):
    def f(x,k,mean,spread):
        a=array
        a*=.8
        return x+.1*np.random.poisson(mean)
    return f


@pytest.mark.parametrize('kind',['normal','uniform','poisson'])
@pytest.mark.parametrize('window',[None,2])
def test_whole_reset_scalar_vjps(engine,kind,window,monkeypatch):
    monkeypatch.setattr(base,kind+'_factory',globals()['scalar_'+kind+'_factory'])
    original=reference
    monkeypatch.setattr(__import__(__name__),'reference',lambda *args,**kwargs:original(*args,scalar=True,**kwargs))
    test_whole_reset_all_bank_initial_vjps(engine,kind,window,False)


def implicit_poisson_factory(array):
    def f(x,k,mean,spread):
        a=array
        a*=.8
        return x+.1*np.random.poisson(x+mean)
    return f


@pytest.mark.parametrize('selection',['empty','partial','full'])
def test_whole_reset_implicit_poisson_detached_original(engine,selection,monkeypatch,tmp_path):
    mpi(2);monkeypatch.setattr(base,'poisson_factory',implicit_poisson_factory)
    net,storage,g,dt,bundle=model(engine,'poisson',selection,detach=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    stream=bundle.provenance['whole_neuron_reset_layout'][g.name]['draws'][0]['stream'];calls=[];observations=0
    for tick in range(4):
        def sample(rate):
            rates=np.asarray(rate);calls.append(len(rates))
            return np.array([count(p['seed'],9,1,q,tick,float(r),stream) for q,r in enumerate(rates)])
        monkeypatch.setattr(np.random,'poisson',sample)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}))
        net.run(dt,namespace={})
        for obj,names in [(storage,('a',)),(g,('v','k'))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        observations+=calls[-1]
        entries=trainer.poisson_state['entries']
        assert len(entries)==observations
        assert sorted(e['identity']['site']['entity'] for e in entries if e['identity']['instant']==tick)==list(range(calls[-1]))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4


@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('refractory',[False,True])
def test_whole_reset_implicit_poisson_detached_vjps(engine,window,refractory,monkeypatch):
    monkeypatch.setattr(base,'poisson_factory',implicit_poisson_factory)
    original_model=model;original_reference=reference
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:original_model(*args,detach=True,**kwargs))
    monkeypatch.setattr(__import__(__name__),'reference',lambda *args,**kwargs:original_reference(*args,implicit=True,**kwargs))
    test_whole_reset_all_bank_initial_vjps(engine,'poisson',window,refractory)


def test_whole_reset_implicit_poisson_surrogate_cached_candidates(engine,monkeypatch):
    monkeypatch.setattr(base,'poisson_factory',implicit_poisson_factory)
    data=model(engine,'poisson',detach=False);bundle=data[4]
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],noise_sequence=9)
    assert len(out['poisson_state']['entries'])==int(np.asarray(out['spikes'])[0,:,4:].sum())
    assert any(action.get('poisson_candidate') for action in bundle.plan['dynamic']['actions'])


@pytest.mark.parametrize('selection',['empty','partial','full'])
def test_whole_reset_implicit_poisson_invalid_actual_columns(engine,selection,monkeypatch):
    monkeypatch.setattr(base,'poisson_factory',implicit_poisson_factory)
    _,storage,g,_,bundle=model(engine,'poisson',selection,detach=True)
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and 'mean' in row['variables'])
    bundle.weights[row['bank']][row['variables'].index('mean')]=-100.
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    state=copy.deepcopy(trainer.state);clock=trainer.clock_tick;sequence=trainer.next_noise_sequence
    if selection=='empty':
        assert np.random.poisson(np.array([],dtype=float)).shape==(0,)
        out=trainer.step(np.zeros((1,1,1)),[0],noise_sequence=9)
        cells=bundle.provenance['neuron_state_layout'][storage.name]['a']
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],np.array([.1,.2,.3,.4])*.8,rtol=8e-5,atol=8e-6)
        assert trainer.poisson_state['entries']==[]
    else:
        with pytest.raises(ValueError):np.random.poisson(np.array([-99.3]))
        with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0],noise_sequence=9)
        assert trainer.state==state and trainer.clock_tick==clock and trainer.next_noise_sequence==sequence
