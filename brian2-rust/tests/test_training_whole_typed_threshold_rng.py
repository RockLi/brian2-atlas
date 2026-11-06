"""Exact int32 callback thresholds across the float32 precision boundary."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_regular_callback_rng import count
from test_native_training import RUNNER


def integer_poisson_factory(array):
    def f(k,index,mean):
        captured=array
        captured*=.8
        draw=np.random.poisson(mean,size=len(captured))
        return k+draw[index]
    return f


def model(backend,operator,rate,refractory=False,window=None):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    storage=b.NeuronGroup(4,'dv/dt=0/second:1\na:1',threshold='v>100',reset='v=0',method='euler',dt=dt);storage.a=[.1,.2,.3,.4]
    g=b.NeuronGroup(2,'dv/dt=gain/ms:1'+(' (unless refractory)' if refractory else '')+'\nk:integer\nindex:integer\nmean:1 (constant,shared)\ngain:1 (constant)',threshold='f(k,index,mean)'+operator+'16777216',reset='v-=.5',refractory=.4*b.ms if refractory else False,method='euler',dt=dt)
    g.v=[.473,.719];g.k=[16777217,16777216];g.index=[3,0];g.mean=rate;g.gain=[.217,.319]
    g.namespace['f']=b.Function(integer_poisson_factory(storage.variables['a'].get_value()),arg_units=[1,1,1],arg_types=['integer','integer','float'],return_unit=1,return_type='integer',stateless=False)
    net=b.Network(source,storage,g);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[storage,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain','mean']})
    return net,storage,g,dt,bundle


@pytest.mark.parametrize('operator',['>','>=','<','<=','==','!='])
@pytest.mark.parametrize('rate',[0.,.37])
@pytest.mark.parametrize('refractory',[False,True])
def test_whole_integer_threshold_original_restore(engine,operator,rate,refractory,monkeypatch,tmp_path):
    mpi(2);net,storage,g,dt,bundle=model(engine,operator,rate,refractory);monitor=b.SpikeMonitor(g);net.add(monitor)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    records=bundle.provenance['whole_neuron_threshold_layout'][g.name]['draws'];assert len(records)==1;stream=records[0]['stream'];calls=[]
    for tick in range(4):
        def sample(mean,*,size):
            assert size==4 and mean==rate;calls.append(tick)
            return np.array([count(p['seed'],9,1,j,tick,rate,stream) for j in range(4)],dtype=np.int64)
        monkeypatch.setattr(np.random,'poisson',sample)
        out=trainer.step(np.zeros((1,1,1)),[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for obj,names in [(storage,('a',)),(g,('v','k','index'))]:
            for name in names:
                cells=bundle.provenance['neuron_state_layout'][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=0 if name in ('k','index') else 8e-5,atol=0 if name in ('k','index') else 8e-6)
        recorded=np.asarray(monitor.i)[np.isclose(np.asarray(monitor.t/b.second),tick*float(dt),rtol=0.,atol=1e-12)]
        np.testing.assert_array_equal(np.asarray(out['spikes'])[0,-1,4:],np.isin(np.arange(2),recorded).astype(float))
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER,request_timeout=300);trainer.restore(path)
    assert calls==list(range(4));assert len(trainer.poisson_state['entries'])==16
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,operator,weights,initial=None,anchors=None,override=None):
    _,storage,g,_,bundle=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
        return weights[row['bank']][row['variables'].index(name)] if g.variables[name].scalar else np.asarray(weights[row['bank']])
    layout=bundle.provenance['neuron_state_layout'];a=layout[storage.name]['a'];v=layout[g.name]['v'];k=layout[g.name]['k'];index=layout[g.name]['index'];counter=layout[g.name].get('__refractory_ticks');flags=bundle.provenance.get('refractory_activity_layout',{}).get(g.name)
    compare={'>':np.greater,'>=':np.greater_equal,'<':np.less,'<=':np.less_equal,'==':np.equal,'!=':np.not_equal}[operator];stream=bundle.provenance['whole_neuron_threshold_layout'][g.name]['draws'][0]['stream'];before=[];events=[];samples=[];controls=[];gates=[]
    for tick in range(5):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());active=np.ones(2,dtype=bool) if counter is None else z[counter]==0
        if anchors is not None:active=anchors['gates'][tick].copy()
        gates.append(active.copy());z[v]+=.2*parameter('gain')*active;z[a]*=.8;rate=parameter('mean');draw=[]
        for j in range(4):
            value=count(p['seed'],9,1,j,tick,float(rate),stream) if anchors is None else anchors['samples'][tick*4+j][3]
            if override==(tick,j):value+=1
            samples.append((tick,j,float(rate),value));draw.append(value)
        integer_value=z[k].astype(np.int64)+np.array(draw,dtype=np.int64)[z[index].astype(int)]
        event=compare(integer_value,np.int64(16777216)).astype(float)*active
        if anchors is not None:event=anchors['events'][tick].copy()
        events.append(event.copy());control=event.astype(bool);controls.append(control.copy());z[v]-=.5*event
        if counter is not None:z[counter]=np.where(control,max(p['refractory'][1]['steps']-1,0),np.maximum(z[counter]-1,0));z[flags]=active&~control
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,events=np.array(events),samples=samples,controls=controls,gates=gates)


@pytest.mark.parametrize('operator',['>','>=','<','<=','==','!='])
@pytest.mark.parametrize('rate',[0.,.37])
@pytest.mark.parametrize('refractory',[False,True])
@pytest.mark.parametrize('window',[None,2])
def test_whole_integer_threshold_all_bank_initial_vjps(engine,operator,rate,refractory,window):
    mpi(2);data=model(engine,operator,rate,refractory,window);bundle=data[4];x=np.zeros((1,5,1));loss,z,anchors=reference(data,operator,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300).gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6);np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,4:],anchors['events'])
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['a'],*bundle.provenance['neuron_state_layout'][data[2].name]['v']];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    def expected(high,low):
        score=0.
        for (tick,j,rate,value),up,down in zip(anchors['samples'],high[2]['samples'],low[2]['samples']):
            signal=loss*(value/rate-1.) if rate else reference(data,operator,bundle.weights,override=(tick,j))[0]-loss
            score+=signal*(up[2]-down[2])/2e-6
        return (high[0]-low[0])/2e-6+score
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            assert out['gradients'][bank][index]==pytest.approx(expected(reference(data,operator,hi,anchors=anchors),reference(data,operator,lo,anchors=anchors)),rel=3e-3,abs=2e-5),(operator,rate,bank,index)
    initial=np.array(bundle.initial_state,float);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300).gradients(x,[0],initial=initial[None],noise_sequence=9);loss,_,anchors=reference(data,operator,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(expected(reference(data,operator,bundle.weights,hi,anchors=anchors),reference(data,operator,bundle.weights,lo,anchors=anchors)),rel=3e-3,abs=2e-5),index
    if rate==0. and operator in ('>','<=','==','!='):
        assert abs(sum(reference(data,operator,bundle.weights,override=(t,j))[0]-loss for t,j,r,value in anchors['samples']))>1e-5
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
