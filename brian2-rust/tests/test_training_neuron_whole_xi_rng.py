"""Whole closure columns plus persistent named neuron SDE streams."""
import copy
import ast
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_neuron_whole_rng import normal_factory,uniform_factory,poisson_factory,derived_factory
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_regular_callback_rng import count
from test_native_training import RUNNER


def model(backend,kind,method='euler',window=None,multi=False,multiplicative=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    storage=b.NeuronGroup(4,'dv/dt=0/second:1\na:1',threshold='v>100',reset='v=0',method='euler',dt=dt);storage.a=[.1,.2,.3,.4]
    equation='dv/dt=(gain+.1*f(v,k,mean,spread)'+('+beta*u' if multi else '')+')/ms+noise*'+('v*' if multiplicative else '')+'xi/ms**.5'+('+other_noise*xi_other/ms**.5' if multi else '')+':1\nk:integer\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant)\nnoise:1 (constant,shared)'
    if multi:equation+='\ndu/dt=(.3*v-.15*u)/ms+.5*noise*xi/ms**.5:1\nbeta:1 (constant,shared)\nother_noise:1 (constant,shared)'
    if multi:equation=equation.replace('xi/ms','xi_a/ms').replace('xi_other','xi_b')
    g=b.NeuronGroup(2,equation,threshold='v>.5',reset='v-=.5',method=method,dt=dt);g.v=[.473,.719];g.k=[3,0];g.mean=1.3 if kind in ('poisson','derived') else .043;g.spread=.13;g.gain=[.217,.319];g.noise=.011
    if multi:g.u=[.13,.07];g.beta=.14;g.other_noise=.005
    function=globals()[kind+'_factory'](storage.variables['a'].get_value());g.namespace['f']=b.Function(function,arg_units=[1,1,1,1],arg_types=['float','integer','float','float'],return_unit=1,stateless=False)
    net=b.Network(source,storage,g);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[storage,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread','noise',*(['beta','other_noise'] if multi else [])]})
    bundle.provenance['test_multiplicative']=multiplicative
    return net,storage,g,dt,bundle


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('method',['euler','heun','milstein'])
def test_whole_xi_original_restore(engine,kind,method,monkeypatch,tmp_path):
    mpi(2);net,storage,g,dt,bundle=model(engine,kind,method);monitor=b.SpikeMonitor(g);net.add(monitor);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    calls=[];named=[];stages=2 if method=='milstein' else 1;streams_offset=len(g.equations.stochastic_variables)
    from brian2_rust.training_brian import _INTEGRATORS
    variables={**g.variables,**g.resolve_all(g.equations.identifiers-set(g.variables),run_namespace={})}
    generated=ast.parse(_INTEGRATORS[method](g.equations,variables=variables));noise_names=sorted(g.equations.stochastic_variables)
    named_order=[noise_names.index(statement.targets[0].id) for statement in generated.body if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id in noise_names]
    for tick in range(4):
        streams=iter(range(streams_offset,streams_offset+stages));named_streams=iter(named_order)
        def sample(*args,**kwargs):
            stream=next(streams);calls.append((tick,stream));assert kwargs['size']==4;mean=float(g.mean[:]);spread=float(g.spread[:]);rate=mean+np.asarray(storage.a[:]) if kind=='derived' else mean
            np.testing.assert_allclose(args,(rate,) if kind in ('poisson','derived') else (mean,spread if kind=='normal' else mean+spread))
            if kind in ('poisson','derived'):return np.array([count(p['seed'],9,1,j,tick,float(np.broadcast_to(rate,(4,))[j]),stream) for j in range(4)])
            return mean+spread*np.array([(normal if kind=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(4)])
        def standard(n):
            assert n==2;stream=next(named_streams);named.append((tick,stream));return np.array([normal(p['seed'],9,0,1,j,tick,stream) for j in range(2)])
        monkeypatch.setattr(np.random,'poisson' if kind=='derived' else kind,sample);monkeypatch.setattr(np.random,'randn',standard)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for obj,names in [(storage,('v','a')),(g,('v','k',*(['u'] if 'u' in g.variables else [])))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)];np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,4:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4*stages and len(named)==4*streams_offset
    if kind in ('poisson','derived'):assert len(trainer.poisson_state['entries'])==16*stages


def parameter(data,weights,name):
    g=data[2];row=next(row for row in data[4].provenance['bindings'] if row['object']==g.name and name in row['variables'])
    return weights[row['bank']][row['variables'].index(name)] if g.variables[name].scalar else np.asarray(weights[row['bank']])


def reference(data,kind,method,weights,initial=None,anchors=None,override=None):
    _,storage,g,_,bundle=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'];a=layout[storage.name]['a'];v=layout[g.name]['v'];k=layout[g.name]['k'];before=[];margins=[];hard=[];events=[];samples=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());stream=len(g.equations.stochastic_variables)
        def stage(x):
            nonlocal stream
            z[a]*=.8;mean=parameter(data,weights,'mean');spread=parameter(data,weights,'spread')
            if kind in ('poisson','derived'):
                rates=np.broadcast_to(mean,(4,))+(z[a] if kind=='derived' else 0.);draws=[]
                for j,rate in enumerate(rates):
                    value=next(row[4] for row in anchors['samples'] if row[:3]==(tick,stream,j)) if anchors is not None else count(p['seed'],9,1,j,tick,float(rate),stream)
                    if override==(tick,stream,j):value+=1
                    samples.append((tick,stream,j,float(rate),value));draws.append(value)
                draws=np.array(draws)
            else:draws=mean+spread*np.array([(normal if kind=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(4)])
            stream+=1;return .2*(parameter(data,weights,'gain')+.1*(x+.1*draws[z[k].astype(int)]))
        old=z[v].copy();multiplicative=bundle.provenance.get('test_multiplicative',False)
        support=stage(old) if method=='milstein' else None
        drift=stage(old)
        raw=np.array([normal(p['seed'],9,0,1,j,tick,0) for j in range(2)])
        sigma=parameter(data,weights,'noise');increment=np.sqrt(.2)*sigma*raw
        z[v]=old+drift+increment*(old if multiplicative else 1.)
        if multiplicative and method=='heun':z[v]+=.5*sigma**2*old*.2*raw**2
        if multiplicative and method=='milstein':z[v]+=.5*sigma*np.sqrt(.2)*(sigma*old*np.sqrt(.2)+support)*raw**2
        if 'u' in layout[g.name]:
            u=layout[g.name]['u'];old_u=z[u].copy();other=np.array([normal(p['seed'],9,0,1,j,tick,1) for j in range(2)])
            z[v]+=.2*parameter(data,weights,'beta')*old_u+np.sqrt(.2)*parameter(data,weights,'other_noise')*other
            z[u]=old_u+.2*(.3*old-.15*old_u)+.5*np.sqrt(.2)*parameter(data,weights,'noise')*raw
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());z[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,events=events,samples=samples)


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('method',['euler','heun','milstein'])
@pytest.mark.parametrize('window',[None,2])
def test_whole_xi_all_bank_initial_vjps(engine,kind,method,window):
    mpi(2);data=model(engine,kind,method,window);bundle=data[4];x=np.zeros((1,4,1));loss,z,anchors=reference(data,kind,method,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9);assert out['loss']==pytest.approx(loss,abs=8e-6)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['a'],*bundle.provenance['neuron_state_layout'][data[2].name]['v'],*bundle.provenance['neuron_state_layout'][data[2].name].get('u',[])];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,4:],anchors['events'])
    def expected(high,low):
        fd=(high[0]-low[0])/2e-6;score=0.
        for base,up,down in zip(anchors['samples'],high[2]['samples'],low[2]['samples']):
            tick,stream,j,rate,value=base;derivative=(up[3]-down[3])/2e-6
            signal=loss*(value/rate-1.) if rate else reference(data,kind,method,bundle.weights,override=(tick,stream,j))[0]-loss;score+=signal*derivative
        return fd+score
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            high=reference(data,kind,method,hi,anchors=anchors);low=reference(data,kind,method,lo,anchors=anchors);assert out['gradients'][bank][index]==pytest.approx(expected(high,low),rel=3e-3,abs=2e-5),(kind,method,bank,index)
    initial=np.array(bundle.initial_state,dtype=float);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9);loss,_,anchors=reference(data,kind,method,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6;high=reference(data,kind,method,bundle.weights,initial=hi,anchors=anchors);low=reference(data,kind,method,bundle.weights,initial=lo,anchors=anchors)
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(expected(high,low),rel=3e-3,abs=2e-5),(kind,method,index)



@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
def test_whole_xi_shared_named_multistate_original(engine,kind,monkeypatch,tmp_path):
    original=model
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:original(*args,multi=True,**kwargs))
    test_whole_xi_original_restore(engine,kind,'euler',monkeypatch,tmp_path)


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('window',[None,2])
def test_whole_xi_shared_named_multistate_vjps(engine,kind,window,monkeypatch):
    original=model
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:original(*args,multi=True,**kwargs))
    test_whole_xi_all_bank_initial_vjps(engine,kind,'euler',window)


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('method',['heun','milstein'])
def test_whole_xi_multiplicative_original(engine,kind,method,monkeypatch,tmp_path):
    original=model
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:original(*args,multiplicative=True,**kwargs))
    test_whole_xi_original_restore(engine,kind,method,monkeypatch,tmp_path)


@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
@pytest.mark.parametrize('method',['heun','milstein'])
@pytest.mark.parametrize('window',[None,2])
def test_whole_xi_multiplicative_vjps(engine,kind,method,window,monkeypatch):
    original=model
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:original(*args,multiplicative=True,**kwargs))
    test_whole_xi_all_bank_initial_vjps(engine,kind,method,window)


@pytest.mark.parametrize('method,multiplicative',[('euler',False),('heun',False),('milstein',False),('heun',True),('milstein',True)])
@pytest.mark.parametrize('window',[None,2])
def test_whole_xi_zero_state_score(engine,method,multiplicative,window,monkeypatch):
    mpi(2);data=model(engine,'derived',method,window,multiplicative=multiplicative);bundle=data[4];g=data[2]
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and 'mean' in row['variables']);bundle.weights[row['bank']][row['variables'].index('mean')]=0.
    rate_cell=bundle.provenance['neuron_state_layout'][data[1].name]['a'][3];bundle.initial_state[rate_cell]=0.;bundle.plan['dynamic']['initial'][rate_cell]=0.
    v=bundle.provenance['neuron_state_layout'][g.name]['v'][0];probe=np.array(bundle.initial_state,dtype=float);probe[v]=0.
    low=reference(data,'derived',method,bundle.weights,initial=probe)[1][v];probe[v]=.1;high=reference(data,'derived',method,bundle.weights,initial=probe)[1][v];center=(.5-low)*.1/(high-low)
    for voltage in np.arange(center-.0025,center,.000025):
        bundle.initial_state[v]=float(voltage);bundle.plan['dynamic']['initial'][v]=float(voltage)
        loss,_,anchors=reference(data,'derived',method,bundle.weights)
        weak=sum(reference(data,'derived',method,bundle.weights,override=(t,s,j))[0]-loss for t,s,j,r,c in anchors['samples'] if r==0.)
        if abs(weak)>1e-5 and min(abs(m[0]) for m in anchors['margins'])>1e-5:break
    else:raise AssertionError('whole xi zero-rate count must have nonzero readout replay')
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:data)
    test_whole_xi_all_bank_initial_vjps(engine,'derived',method,window)


@pytest.mark.parametrize('method',['euler','milstein'])
@pytest.mark.parametrize('kind',['normal','poisson'])
def test_whole_xi_invalid_index_atomic(engine,method,kind,monkeypatch,tmp_path):
    import test_training_neuron_whole_rng as base
    data=model(engine,kind,method);monkeypatch.setattr(base,'model',lambda *args,**kwargs:data)
    base.test_neuron_whole_threshold_invalid_index_atomic(engine,kind,tmp_path)


@pytest.mark.parametrize('method',['euler','milstein'])
@pytest.mark.parametrize('kind',['normal','uniform','poisson','derived'])
def test_whole_xi_bad_domain_atomic(engine,method,kind,monkeypatch,tmp_path):
    import test_training_neuron_whole_rng as base
    data=model(engine,kind,method);monkeypatch.setattr(base,'model',lambda *args,**kwargs:data)
    base.test_neuron_whole_threshold_bad_domain_atomic(engine,kind,tmp_path)
