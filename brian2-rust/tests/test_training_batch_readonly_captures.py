"""Whole readonly banks are checked on nonempty NumPy event batches."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_batch_event_captures import batch_model
from test_training_event_captures import readonly_event_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(repeated,delay,ranks,backend,window=None,dtype='float'):
    net,g,old,_,dt,_,x=batch_model(repeated,False,delay,ranks,backend)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    net.remove(old)
    field={'float':'gain:1 (constant)','integer':'gain:integer (constant)','boolean':'gain:boolean (constant)'}[dtype]
    kwargs={} if dtype=='float' else {'dtype':{'gain':np.int32 if dtype=='integer' else bool}}
    syn=b.Synapses(source,g,'h:1\n'+field,on_pre='h=f(h)'+(';v_post+=gain*h' if repeated else ''),dt=dt,**kwargs)
    syn.connect(i=[0,1],j=[0,0] if repeated else [0,1]);syn.h=[.113,.173];syn.gain=[.11,.19] if dtype=='float' else [1,2] if dtype=='integer' else [True,True];syn.delay=delay*dt
    syn.namespace['f']=b.Function(readonly_event_capture(syn.variables['gain'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net.add(syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain'] if dtype=='float' else ['h']},learning_rate=.01)
    assert bundle.provenance['mutable_capture_layout']==[] and bundle.provenance['readonly_capture_layout']
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    return net,g,syn,dt,bundle,x,bank


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_readonly_original_carry_restore(engine,repeated,delay,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x,_=model(repeated,delay,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,name in [(g,'neuron_state_layout','v'),(syn,'dynamic_state_layout','h')]:
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('dtype',['float','integer','boolean'])
@pytest.mark.parametrize('ranks',[None,2])
def test_readonly_unselected_column_live_bank_domain_empty_atomic(engine,repeated,dtype,ranks,tmp_path):
    mpi(ranks);_,_,_,_,bundle,_,bank=model(repeated,0,ranks,engine,dtype=dtype)
    weights=copy.deepcopy(bundle.weights);weights[bank][1]=0.
    t=NativeLIFTrainer(bundle.plan,weights=weights,runner=RUNNER)
    t.evaluate(np.zeros((1,1,2)),[0]);before=copy.deepcopy(t.state)
    path=tmp_path/'bank';t.store(path);t=NativeLIFTrainer(bundle.plan,runner=RUNNER);t.restore(path)
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):t.step(np.array([[[1.,0.]]]),[0])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0


def reference(data,weights,initial=None,anchors=None,start=0,count=4):
    _,g,syn,_,bundle,x,bank=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    routes=bundle.provenance['delay_queues'][syn.pre.name+'::numpy-stage:1']['new'];delay=routes[0]['delay']
    targets=np.asarray(syn.j[:]);repeated=targets[0]==targets[1];gain=weights[bank]
    before=[];margins=[];hard=[];spikes=[]
    for local in range(count):
        tick=start+local
        if anchors is not None and p['tbptt_window'] and local and local%p['tbptt_window']==0:z=anchors['before'][local].copy()
        before.append(z.copy());margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][local];event=anchors['hard'][local]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());old=z[h].copy()
        for edge in range(2):
            gate=z[routes[edge]['states'][0]] if delay else x[0,tick,edge]
            new=.7*old[edge];z[h[edge]]+=gate*(new-z[h[edge]])
            if repeated:z[v[targets[edge]]]+=gate*gain[edge]*new
        for path in bundle.provenance['delay_queues'].values():
            for row in path['new']:
                if row['states']:
                    z[row['states'][:-1]]=z[row['states'][1:]];z[row['states'][-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_readonly_all_bank_initial_vjps(engine,repeated,delay,window,ranks):
    mpi(ranks);data=model(repeated,delay,ranks,engine,window);bundle=data[4]
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[5],[0]);loss,z,anchors=reference(data,bundle.weights)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['v'],*bundle.provenance['dynamic_state_layout'][data[2].name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(reference(data,hi,anchors=anchors)[0]-reference(data,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,j)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j] or j in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(data,bundle.weights,hi,anchors=anchors)[0]-reference(data,bundle.weights,lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j


@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_readonly_optimizer_carry_restore(engine,delay,ranks,tmp_path):
    mpi(ranks);data=model(True,delay,ranks,engine);bundle=data[4];x=data[5];bank=data[6]
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:3],[0]);weights=copy.deepcopy(t.state['weights'])
    assert not np.array_equal(weights[bank],bundle.weights[bank])
    path=tmp_path/'optimizer';t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path);tail=t.step(x[:,3:],[0],initial='carry')
    loss,z,_=reference(data,weights,first['final_state'][0],start=3,count=1)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['v'],*bundle.provenance['dynamic_state_layout'][data[2].name]['h']]
    np.testing.assert_allclose(np.asarray(tail['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert tail['loss']==pytest.approx(loss,abs=8e-6)


def mixed_capture(gain,mutable):
    def f(x):
        captured=mutable
        captured*=.8
        ignored=1./gain
        x*=.7
        return x
    return f


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_readonly_and_mutable_columns_share_batch(engine,repeated,ranks):
    mpi(ranks);net,g,syn,dt,_,x,_=model(repeated,0,ranks,engine);array=np.array([.23,.31])
    syn.namespace['f']=b.Function(mixed_capture(syn.variables['gain'].get_value(),array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    assert bundle.provenance['readonly_capture_layout'] and bundle.provenance['mutable_capture_layout']
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,name in [(g,'neuron_state_layout','v'),(syn,'dynamic_state_layout','h')]:
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        cells=bundle.provenance['mutable_capture_layout'][0]['cells']
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],array,rtol=8e-5,atol=8e-6)
