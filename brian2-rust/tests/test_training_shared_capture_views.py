"""Writable and readonly closure views observe one canonical event buffer."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_batch_event_captures import batch_model
from test_training_event_captures import event_capture,readonly_event_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def zero_capture(array):
    def f(x):
        captured=array
        captured*=0.
        x*=.7
        return x
    return f


def model(repeated,readonly_first,delay,ranks,backend,zero=False,owned=False):
    net,g,syn,_,dt,_,x=batch_model(repeated,False,delay,ranks,backend)
    array=syn.variables['h'].get_value() if owned else np.array([.23,.31]);view=array.view();view.flags.writeable=False
    readonly=b.Function(readonly_event_capture(view),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    mutable=b.Function((zero_capture if zero else event_capture)(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    syn.namespace.update(f=readonly if readonly_first else mutable,q=mutable if readonly_first else readonly)
    syn.pre.code='h=f(h);h=q(h)'+(';v_post+=.11*h' if repeated else '')
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h']})
    assert len(bundle.provenance['mutable_capture_layout'])==1
    assert {row['readonly'] for row in bundle.provenance['mutable_capture_layout'][0]['aliases']}=={False,True}
    assert not bundle.provenance['readonly_capture_layout']
    return net,g,syn,array,dt,bundle,x


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_shared_capture_original_carry_restore(engine,repeated,readonly_first,delay,ranks,tmp_path):
    mpi(ranks);net,g,syn,array,dt,bundle,x=model(repeated,readonly_first,delay,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,name in [(g,'neuron_state_layout','v'),(syn,'dynamic_state_layout','h')]:
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        cells=bundle.provenance['mutable_capture_layout'][0]['cells'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],array,rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_readonly_view_checks_value_written_in_same_batch(engine,repeated,ranks):
    mpi(ranks);_,_,_,_,_,bundle,_=model(repeated,False,0,ranks,engine,zero=True)
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);t.evaluate(np.zeros((1,1,2)),[0]);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):t.step(np.ones((1,1,2)),[0])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_readonly_physical_view_observes_numpy_writeback_boundary(engine,repeated,ranks):
    mpi(ranks);net,g,syn,_,dt,bundle,_=model(repeated,False,0,ranks,engine,zero=True,owned=True)
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);x=np.ones((1,1,2))
    if not repeated:
        with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):t.evaluate(x,[0])
        return
    out=t.evaluate(x,[0]);net.run(dt,namespace={})
    for obj,key,name in [(g,'neuron_state_layout','v'),(syn,'dynamic_state_layout','h')]:
        cells=bundle.provenance[key][obj.name][name]
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)


def reference(data,weights,initial=None,anchors=None,window=None):
    _,g,syn,_,_,bundle,x=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h'];cap=bundle.provenance['mutable_capture_layout'][0]['cells']
    stages=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:];delay=bundle.provenance['delay_queues'][stages[0]]['new'][0]['delay'];targets=np.asarray(syn.j[:])
    mutable_stage=next(k for k,name in enumerate(['f','q']) if any(row['function']==name and not row['readonly'] for row in bundle.provenance['mutable_capture_layout'][0]['aliases']))
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());arrival=tick-delay;selected=[0,1] if arrival==0 else [0] if arrival==2 else []
        for k,path in enumerate(stages):
            rows=bundle.provenance['delay_queues'][path]['new'];old=z[h].copy()
            if selected and (len(stages)==1 or k==mutable_stage):z[cap]*=.8
            for edge in range(2):
                gate=z[rows[edge]['states'][0]] if delay else x[0,tick,edge]
                if len(stages)==1 or k<2:z[h[edge]]+=gate*((.49 if len(stages)==1 else .7)*old[edge]-z[h[edge]])
                elif k==2:z[v[targets[edge]]]+=gate*.11*old[edge]
        for path in bundle.provenance['delay_queues'].values():
            for row in path['new']:
                if row['states']:
                    z[row['states'][:-1]]=z[row['states'][1:]];z[row['states'][-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_shared_capture_all_bank_initial_vjps(engine,repeated,readonly_first,delay,window,ranks):
    mpi(ranks);data=model(repeated,readonly_first,delay,ranks,engine);bundle=data[5];p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(data[6],[0]);loss,z,anchors=reference(data,bundle.weights,window=window)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['v'],*bundle.provenance['dynamic_state_layout'][data[2].name]['h'],*bundle.provenance['mutable_capture_layout'][0]['cells']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(reference(data,hi,anchors=anchors,window=window)[0]-reference(data,lo,anchors=anchors,window=window)[0])/2e-6
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,j)
    for j in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][j] or j in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(data,bundle.weights,hi,anchors=anchors,window=window)[0]-reference(data,bundle.weights,lo,anchors=anchors,window=window)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j


@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_shared_capture_scalar_fifo_original(engine,readonly_first,ranks):
    mpi(ranks);net,g,syn,array,dt,_,x=model(True,readonly_first,1,ranks,engine)
    syn.pre.code='h=f(v_post);h=q(h);v_post+=.11*h'
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h']})
    assert bundle.provenance['event_callback_modes'][syn.pre.name]['mode']=='scalar'
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);net.run(4*dt,namespace={})
    for obj,key,name in [(g,'neuron_state_layout','v'),(syn,'dynamic_state_layout','h')]:
        cells=bundle.provenance[key][obj.name][name]
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    cells=bundle.provenance['mutable_capture_layout'][0]['cells'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],array,rtol=8e-5,atol=8e-6)
