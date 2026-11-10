"""Whole foreign captures in eager NumPy Boolean threshold expressions."""
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


def boolean_normal_factory(array):
    def f(x,k,mean,spread):
        a=array
        a*=.8
        draw=np.random.normal(mean,spread,size=len(a))
        return x+.1*draw[k]>.5
    return f


def model(backend,kind,condition,window=None,refractory=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    storage=b.NeuronGroup(4,'dv/dt=0/second:1\na:1',threshold='v>100',reset='v=0',method='euler',dt=dt);storage.a=[.1,.2,.3,.4]
    call='f(v,k,mean,spread)';threshold={'and':f'({call}>.5) and ({call}<.8)','or':f'({call}>.5) or ({call}<.3)','not':f'not ({call}>.5)','equal':f'{call}=={call}','boolean':call}[condition]
    g=b.NeuronGroup(2,'dv/dt=gain/ms:1'+(' (unless refractory)' if refractory else '')+'\nk:integer\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant)',threshold=threshold,reset='v-=.5',refractory=.4*b.ms if refractory else False,method='euler',dt=dt)
    g.v=[.473,.719];g.k=[3,0];g.mean=1.3 if kind in ('poisson','derived') else .043;g.spread=.13;g.gain=[.217,.319]
    factory=boolean_normal_factory if condition=='boolean' else getattr(base,kind+'_factory')
    g.namespace['f']=b.Function(factory(storage.variables['a'].get_value()),arg_units=[1,1,1,1],arg_types=['float','integer','float','float'],return_unit=bool if condition=='boolean' else 1,return_type='boolean' if condition=='boolean' else 'float',stateless=False)
    net=b.Network(source,storage,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[storage,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean','spread']})
    return net,storage,g,dt,bundle


@pytest.mark.parametrize('kind,condition',[(kind,condition) for kind in ('normal','uniform','poisson','derived') for condition in ('and','or','not','equal')]+[('normal','boolean')])
def test_whole_predicate_original_restore(engine,condition,kind,monkeypatch,tmp_path):
    mpi(2);net,storage,g,dt,bundle=model(engine,kind,condition,refractory=True)
    monitor=b.SpikeMonitor(g);net.add(monitor);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);records=bundle.provenance['whole_neuron_threshold_layout'][g.name]['draws'];calls=[]
    for tick in range(4):
        used=[]
        def sample(*args,**kwargs):
            assert kwargs['size']==4
            stream=records[len(used)]['stream'];used.append(stream);calls.append((tick,stream))
            mean=float(g.mean[:]);spread=float(g.spread[:]);rate=mean+np.asarray(storage.a[:]) if kind=='derived' else mean
            if kind in ('poisson','derived'):return np.array([count(p['seed'],9,1,j,tick,float(np.broadcast_to(rate,(4,))[j]),stream) for j in range(4)])
            return mean+spread*np.array([(normal if kind=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(4)])
        monkeypatch.setattr(np.random,'poisson' if kind=='derived' else kind,sample)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for obj,names in [(storage,('a',)),(g,('v','k'))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,4:],np.isin(np.arange(2),recorded).astype(float))
        assert used==[r['stream'] for r in records]
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==4*len(records)


def predicate_reference(data,kind,condition,weights,initial=None,anchors=None,override=None):
    _,storage,g,_,bundle=data;p=bundle.plan;layout=bundle.provenance['neuron_state_layout']
    z=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
        return weights[row['bank']][row['variables'].index(name)] if g.variables[name].scalar else np.asarray(weights[row['bank']])
    a=layout[storage.name]['a'];v=layout[g.name]['v'];k=layout[g.name]['k']
    counter=layout[g.name].get('__refractory_ticks');flags=bundle.provenance.get('refractory_activity_layout',{}).get(g.name)
    records=bundle.provenance['whole_neuron_threshold_layout'][g.name]['draws'];before=[];margins=[];hard=[];events=[];samples=[];controls=[];gates=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());active=np.ones(2,dtype=bool) if counter is None else z[counter]==0
        if anchors is not None:active=anchors['gates'][tick].copy()
        gates.append(active.copy());z[v]+=.2*parameter('gain')*active
        values=[]
        for call,row in enumerate(records):
            z[a]*=.8;mean=parameter('mean');spread=parameter('spread');stream=row['stream']
            if kind in ('poisson','derived'):
                rates=np.broadcast_to(mean,(4,))+(z[a] if kind=='derived' else 0.);draw=[]
                for j,rate in enumerate(rates):
                    value=count(p['seed'],9,1,j,tick,float(rate),stream) if anchors is None else next(s[4] for s in anchors['samples'] if s[:3]==(tick,call,j))
                    if override==(tick,call,j):value+=1
                    samples.append((tick,call,j,float(rate),value));draw.append(value)
                draw=np.array(draw)
            else:draw=mean+spread*np.array([(normal if kind=='normal' else uniform)(p['seed'],9,0,1,j,tick,stream) for j in range(4)])
            values.append(z[v]+.1*draw[z[k].astype(int)])
        tick_margins=[];tick_hard=[]
        def compare(value):
            index=len(tick_hard);gate=(value>0).astype(float);tick_margins.append(value.copy());tick_hard.append(gate.copy())
            if anchors is not None:
                old=anchors['margins'][tick][index]
                gate=anchors['hard'][tick][index]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(value-old)
            return gate
        if condition=='equal':event=(values[0]==values[1]).astype(float)
        elif condition in ('not','boolean'):
            event=compare(values[0]-.5)
            if condition=='not':event=1-event
        else:
            left=compare(values[0]-.5);right=compare((.8 if condition=='and' else .3)-values[1])
            event=left*right if condition=='and' else left+right-left*right
        event*=active;events.append(event.copy());margins.append(tick_margins);hard.append(tick_hard)
        control=event.astype(bool) if anchors is None else anchors['controls'][tick];controls.append(control.copy())
        z[v]-=.5*event
        if counter is not None:
            z[counter]=np.where(control,max(p['refractory'][1]['steps']-1,0),np.maximum(z[counter]-1,0));z[flags]=active&~control
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,events=np.array(events),samples=samples,controls=controls,gates=gates)


@pytest.mark.parametrize('kind,condition',[(kind,condition) for kind in ('normal','uniform','poisson','derived') for condition in ('and','or','not','equal')]+[('normal','boolean')])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('refractory',[False,True])
def test_whole_predicate_all_bank_initial_vjps(engine,kind,condition,window,refractory):
    mpi(2);data=model(engine,kind,condition,window,refractory);bundle=data[4];x=np.zeros((1,4,1));loss,z,anchors=predicate_reference(data,kind,condition,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['a'],*bundle.provenance['neuron_state_layout'][data[2].name]['v']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,4:],anchors['events'])
    def expected(high,low):
        derivative=(high[0]-low[0])/2e-6;score=0.
        for base,up,down in zip(anchors['samples'],high[2]['samples'],low[2]['samples']):
            tick,call,j,rate,value=base;rate_vjp=(up[3]-down[3])/2e-6
            signal=loss*(value/rate-1.) if rate else predicate_reference(data,kind,condition,bundle.weights,override=(tick,call,j))[0]-loss
            score+=signal*rate_vjp
        return derivative+score
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            high=predicate_reference(data,kind,condition,hi,anchors=anchors);low=predicate_reference(data,kind,condition,lo,anchors=anchors)
            assert out['gradients'][bank][index]==pytest.approx(expected(high,low),rel=3e-3,abs=2e-5),(kind,condition,bank,index)
    initial=np.array(bundle.initial_state,dtype=float);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300).gradients(x,[0],initial=initial[None],noise_sequence=9)
    loss,_,anchors=predicate_reference(data,kind,condition,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        high=predicate_reference(data,kind,condition,bundle.weights,initial=hi,anchors=anchors);low=predicate_reference(data,kind,condition,bundle.weights,initial=lo,anchors=anchors)
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(expected(high,low),rel=3e-3,abs=2e-5),(kind,condition,index)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('corruption',['unproved_input','nonboolean_producer'])
def test_whole_predicate_identity_requires_boolean_proof(engine,corruption):
    mpi(2);bundle=model(engine,'normal','and')[4];p=copy.deepcopy(bundle.plan);spec=p['dynamic']
    threshold=next(a for a in spec['actions'] if a.get('threshold_predicate'))
    cell=threshold['reads'][0]
    if corruption=='unproved_input':spec['binary_states'].remove(cell)
    else:
        producer=next(a for a in spec['actions'] if cell in a['writes'] and len(a['reads'])>1)
        spec['program_sets'][producer['program_set']][producer['writes'].index(cell)]=[dict(op='constant',value=.25)]
    with pytest.raises(ValueError,match='invalid or repeated dynamic threshold'):
        NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER,request_timeout=300).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)


@pytest.mark.parametrize('condition',['and','or'])
@pytest.mark.parametrize('window',[None,2])
def test_whole_predicate_zero_state_score(engine,condition,window,monkeypatch):
    mpi(2);data=model(engine,'derived',condition,window,True);bundle=data[4];g=data[2]
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and 'mean' in row['variables'])
    bundle.weights[row['bank']][row['variables'].index('mean')]=0.
    cell=bundle.provenance['neuron_state_layout'][data[1].name]['a'][3]
    bundle.initial_state[cell]=0.;bundle.plan['dynamic']['initial'][cell]=0.
    voltage_cell=bundle.provenance['neuron_state_layout'][g.name]['v'][0]
    for voltage in np.arange(.1,.8,.003):
        bundle.initial_state[voltage_cell]=float(voltage);bundle.plan['dynamic']['initial'][voltage_cell]=float(voltage)
        loss,_,anchors=predicate_reference(data,'derived',condition,bundle.weights)
        weak=sum(predicate_reference(data,'derived',condition,bundle.weights,override=(t,c,j))[0]-loss for t,c,j,rate,value in anchors['samples'] if rate==0.)
        if abs(weak)>1e-5 and all(np.min(abs(margin))>1e-5 for row in anchors['margins'] for margin in row):break
    else:raise AssertionError('zero-rate whole draw must change complete predicate replay')
    monkeypatch.setattr(__import__(__name__),'model',lambda *args,**kwargs:data)
    test_whole_predicate_all_bank_initial_vjps(engine,'derived',condition,window,True)
