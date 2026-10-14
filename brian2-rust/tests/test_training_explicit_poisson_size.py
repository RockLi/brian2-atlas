"""Poisson len(array) output size keeps independent physical-row samples."""
import copy
import numpy as np
import brian2 as b
import pytest
import test_training_mixed_shape_rng as base
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_training_regular_callback_rng import count
from test_native_training import RUNNER
from brian2_rust import NativeLIFTrainer


def array_rate(x,mean,spread):
    return x+.1*np.random.poisson(mean,size=len(x))


def scalar_rate(x,mean,spread):
    return x+.1*np.random.poisson(spread,size=len(x))


def keyword_rate(x,mean,spread):
    return x+.1*np.random.poisson(size=len(x),lam=mean)


def chain_rate(array):
    def f(x,mean,spread):
        captured=array
        captured*=.8
        x*=.7
        draw=np.random.normal(0.,spread)
        return x+.1*draw+.1*np.random.poisson(mean+.2*draw,size=len(x))
    return f


def install(monkeypatch,variant):
    monkeypatch.setattr(base,'callback',lambda chain,capture,array:chain_rate(array) if variant=='chain' else globals()[variant+'_rate'])


PROFILES=[(phase,variant,'euler') for phase in ('threshold','reset','update','synaptic') for variant in ('array','scalar')]+[
 ('threshold','keyword','euler'),('reset','chain','euler'),('update','chain','rk2'),('synaptic','chain','rk2')]


@pytest.mark.parametrize('phase,variant,method',PROFILES)
def test_explicit_size_original_restore(engine,phase,variant,method,monkeypatch,tmp_path):
    install(monkeypatch,variant);mpi(2);net,hidden,g,obj,dt,bundle=base.model(engine,phase,variant=='chain',variant=='chain',method)
    monitor=b.SpikeMonitor(g);net.add(monitor);domain=2 if phase=='synaptic' else 1
    records=bundle.provenance['synaptic_integrator_draws'][obj.name] if phase=='synaptic' else bundle.provenance['neuron_draws'][1][phase]
    kinds=['randn','poisson'] if variant=='chain' else ['poisson'];assert [(r['kind'],r.get('scalar',False)) for r in records]==[(kind,kind=='randn') for kind in kinds]*(2 if method=='rk2' else 1)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);saved_count=0
    for tick in range(4):
        serial=0;previous=0.
        selected=np.flatnonzero(np.asarray(g.v[:])+.2*np.asarray(g.gain[:])>.5) if phase=='reset' else np.arange(2)
        def scalar(loc,spread):
            nonlocal serial,previous
            assert variant=='chain' and kinds[serial%len(kinds)]=='randn' and loc==0.
            previous=float(spread*normal(p['seed'],9,0,domain,0,tick,serial));serial+=1;return previous
        def poisson(*args,**kwargs):
            nonlocal serial,saved_count
            rate=args[0] if args else kwargs['lam'];size=args[1] if len(args)>1 else kwargs['size'];assert size==len(selected)
            expected=float(obj.spread[:]) if variant=='scalar' else np.asarray(obj.mean[:])[selected]+(.2*previous if variant=='chain' else 0.)
            np.testing.assert_allclose(rate,expected);rates=np.broadcast_to(rate,(size,))
            values=np.array([count(p['seed'],9,domain,int(j),tick,float(rates[k]),serial) for k,j in enumerate(selected)]);serial+=1;saved_count+=len(selected);return values
        monkeypatch.setattr(np.random,'normal',scalar);monkeypatch.setattr(np.random,'poisson',poisson)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        assert serial==len(records)
        for group in (hidden,g):
            for name in ('v','a','q'):
                if name not in bundle.provenance['neuron_state_layout'][group.name]:continue
                cells=bundle.provenance['neuron_state_layout'][group.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],group.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        if phase=='synaptic':
            for name,cells in bundle.provenance['dynamic_state_layout'][obj.name].items():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)];np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,len(hidden):],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(trainer.poisson_state['entries'])==saved_count
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,phase,chain,capture,method,weights,initial=None,anchors=None,override=None):
    from unittest.mock import patch
    from brian2_rust.training_brian import _INTEGRATORS
    _,hidden,g,obj,_,bundle=data;p=bundle.plan;domain=2 if phase=='synaptic' else 1
    z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v']
    obj_layout=bundle.provenance['dynamic_state_layout'][obj.name] if phase=='synaptic' else layout
    a=obj_layout.get('a');target=obj_layout['h'] if phase=='synaptic' else v;name='h' if phase=='synaptic' else 'v'
    stage=compile(_INTEGRATORS[method](obj.equations,variables={**obj.variables,'f':obj.namespace['f']}),'<independent mixed integrator>','exec') if phase in ('update','synaptic') else None
    before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        if phase=='synaptic':z[layout['q']]=z[target];z[v]+=.2*(z[layout['q']]+.01)
        elif phase!='update':z[v]+=.2*base.parameter(data,weights,'gain')
        physical=z[a].copy() if a is not None else base.parameter(data,weights,'a').copy();serial=0
        active=(z[v]>.5) if phase=='reset' else np.ones(2,dtype=bool)
        def scalar(loc,spread):
            nonlocal serial
            stream=serial;serial+=1
            return float(loc+spread*normal(p['seed'],9,0,domain,0,tick,stream))
        def vector(rate=None,*args,**kwargs):
            nonlocal serial
            stream=serial;serial+=1;rate=kwargs.get('lam',rate)
            assert (args[0] if args else kwargs['size'])==2
            rate=np.broadcast_to(np.asarray(rate),(2,));values=[]
            for j,r in enumerate(rate):
                value=next(c for t,s,k,rr,c,act in anchors['samples'] if (t,s,k)==(tick,stream,j)) if anchors is not None else count(p['seed'],9,domain,j,tick,float(r),stream)
                if override==(tick,stream,j):value+=1
                samples.append((tick,stream,j,float(r),value,bool(active[j])));values.append(value)
            return np.array(values)
        with patch.object(np.random,'normal',scalar),patch.object(np.random,'poisson',vector):
            f=base.callback(chain,capture,physical)
            if stage is not None:
                env={name:z[target].copy(),'a':physical,'mean':base.parameter(data,weights,'mean'),'spread':base.parameter(data,weights,'spread'),'gain':base.parameter(data,weights,'gain'),'dt':.0002,'ms':.001,'f':f}
                exec(stage,env);z[target]=env[name]
                margin=z[v]-.5
            elif phase=='threshold':
                value=z[v].copy();result=f(value,base.parameter(data,weights,'mean'),base.parameter(data,weights,'spread'));z[v]=value;margin=result-.5
            else:
                margin=z[v]-.5
                candidate=.2*f(physical.copy(),base.parameter(data,weights,'mean'),base.parameter(data,weights,'spread'))+z[v]-.5
        if a is not None:z[a]=physical
        event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy())
        if phase=='reset':z[v]+=event*(candidate-z[v])
        else:z[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,events=events,samples=samples)



@pytest.mark.parametrize('phase,variant,method,zero',[(phase,variant,method,False) for phase,variant,method in PROFILES]+[(phase,'array','euler',True) for phase in ('threshold','reset','update','synaptic')]+[(phase,'scalar','euler','scalar') for phase in ('threshold','reset','update','synaptic')])
@pytest.mark.parametrize('window',[None,2])
def test_explicit_size_all_bank_initial_vjps(engine,phase,variant,method,zero,window,monkeypatch):
    install(monkeypatch,variant);monkeypatch.setattr(base,'reference',reference)
    if zero=='scalar':
        data=base.model(engine,phase,False,False,method,window);bundle=data[5]
        row=next(row for row in bundle.provenance['bindings'] if row['object']==data[3].name and 'spread' in row['variables']);bundle.weights[row['bank']][row['variables'].index('spread')]=0.
        cell=bundle.provenance['neuron_state_layout'][data[2].name]['v'][0]
        voltage_range=np.arange(.65,.85,.0005) if phase=='reset' else np.arange(.25,.7,.0001 if phase=='synaptic' else .0005)
        for voltage in voltage_range:
            bundle.initial_state[cell]=float(voltage);bundle.plan['dynamic']['initial'][cell]=float(voltage)
            loss,_,anchors=reference(data,phase,False,False,method,bundle.weights)
            weak=sum(reference(data,phase,False,False,method,bundle.weights,override=(t,s,j))[0]-loss for t,s,j,r,c,act in anchors['samples'] if act)
            if abs(weak)>1e-5 and min(abs(m[0]) for m in anchors['margins'])>1e-5:break
        else:raise AssertionError('explicit scalar rate zero must have nonzero independent-row replay')
        monkeypatch.setattr(base,'model',lambda *args,**kwargs:data)
    base.test_mixed_shape_all_bank_initial_vjps(engine,phase,variant=='chain',variant=='chain',method,zero is True,window)


@pytest.mark.parametrize('variant,invalid',[('array',False),('array',True),('scalar',False),('scalar',True),('chain',False)])
def test_explicit_size_empty_reset_validation(engine,variant,invalid,monkeypatch):
    install(monkeypatch,variant);mpi(2);net,hidden,g,obj,dt,bundle=base.model(engine,'reset',variant=='chain',variant=='chain')
    cells=bundle.provenance['neuron_state_layout'][g.name]['v'];g.v=[-4.013,-3.017]
    for cell,value in zip(cells,g.v[:]):bundle.initial_state[cell]=float(value);bundle.plan['dynamic']['initial'][cell]=float(value)
    if invalid:
        name='spread' if variant=='scalar' else 'mean';row=next(row for row in bundle.provenance['bindings'] if row['object']==obj.name and name in row['variables'])
        if name=='mean':bundle.weights[row['bank']][:]=[-1.,-2.];g.mean=[-1.,-2.]
        else:bundle.weights[row['bank']][row['variables'].index(name)]=-1.;g.spread=-1.
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    if invalid and variant=='scalar':
        with pytest.raises(b.BrianObjectException) as error:net.run(dt,namespace={})
        assert isinstance(error.value.__cause__,ValueError)
        before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
        with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0],noise_sequence=9)
        assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
        return
    for tick in range(4):
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','a'):
            indices=bundle.provenance['neuron_state_layout'][g.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,indices],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    assert trainer.poisson_state['entries']==[]


@pytest.mark.parametrize('phase',['threshold','reset','update','synaptic'])
def test_explicit_size_invalid_restore_atomic(engine,phase,monkeypatch,tmp_path):
    install(monkeypatch,'array');base.test_mixed_shape_invalid_restore_atomic(engine,phase,'mean',tmp_path)
