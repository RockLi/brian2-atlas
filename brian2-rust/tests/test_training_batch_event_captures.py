"""Actual NumPy batches update whole captures once, then selected rows."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def batch_capture(array):
    def f(x):
        captured=array
        captured*=.8
        x*=.7
        return x
    return f


def batch_model(repeated,owned,delay,ranks,backend,calls=1):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0],np.array([0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.217,.379]
    syn=b.Synapses(source,g,'h:1',on_pre=';'.join(['h=f(h)']*calls)+(';v_post+=.11*h' if repeated else ''),dt=dt)
    syn.connect(i=[0,1],j=[0,0] if repeated else [0,1]);syn.h=[.113,.173];syn.delay=delay*dt
    array=syn.variables['h'].get_value() if owned else np.array([.23,.31])
    syn.namespace['f']=b.Function(batch_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h']})
    assert bundle.provenance['event_callback_modes'][syn.pre.name+'::numpy-stage:1']['mode']==('vectorised' if repeated else 'array')
    x=np.zeros((1,4,2));x[0,0,:]=1;x[0,2,0]=1
    return net,g,syn,array,dt,bundle,x


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('owned',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_capture_original_partial_empty_restore(engine,repeated,owned,delay,ranks,tmp_path):
    mpi(ranks);net,g,syn,array,dt,bundle,x=batch_model(repeated,owned,delay,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    capture=bundle.provenance['mutable_capture_layout'][0]['cells']
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,name in [(g,'neuron_state_layout','v'),(syn,'dynamic_state_layout','h')]:
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,capture],array,rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def batch_reference(data,weights,initial=None,anchors=None,window=None):
    _,g,syn,_,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    captured=bundle.provenance['mutable_capture_layout'][0]['cells'];owned=captured==h
    targets=np.asarray(syn.j[:]);stages=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    routes=bundle.provenance['delay_queues'][stages[0]]['new'];delay=int(routes[0]['delay'])
    calls=str(syn.pre.code).count('f(');vectorised=bundle.provenance['event_callback_modes'][stages[0]]['mode']=='vectorised'
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());arrival=tick-delay
        selected=[0,1] if arrival==0 else [0] if arrival==2 else []
        for stage_number,path in enumerate(stages):
            rows=bundle.provenance['delay_queues'][path]['new'];old=z[h].copy()
            callback_stage=len(stages)==1 or stage_number<calls
            if callback_stage and selected:z[captured]*=.8**(calls if len(stages)==1 else 1)
            for edge in range(2):
                gate=z[rows[edge]['states'][0]] if delay else float(edge in selected)
                new=(.7**(calls if len(stages)==1 else 1))*old[edge]
                if callback_stage:z[h[edge]]+=gate*(new-z[h[edge]])
                if vectorised and (len(stages)==1 or not callback_stage):z[v[targets[edge]]]+=gate*.11*(new if len(stages)==1 else old[edge])
        for path in bundle.provenance['delay_queues'].values():
            for row in path['new']:
                if row['states']:
                    z[row['states'][:-1]]=z[row['states'][1:]];z[row['states'][-1]]=float(row['edge'] in ([0,1] if tick==0 else [0] if tick==2 else []))
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('owned',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('calls',[1,2])
def test_batch_capture_bank_initial_vjps(engine,repeated,owned,delay,window,ranks,calls):
    mpi(ranks);data=batch_model(repeated,owned,delay,ranks,engine,calls);bundle=data[5]
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(data[6],[0])
    loss,z,anchors=batch_reference(data,bundle.weights,window=window)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['v'],*bundle.provenance['dynamic_state_layout'][data[2].name]['h'],*bundle.provenance['mutable_capture_layout'][0]['cells']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(batch_reference(data,hi,anchors=anchors,window=window)[0]-batch_reference(data,lo,anchors=anchors,window=window)[0])/2e-6
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,j)
    for j in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][j] or j in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(batch_reference(data,bundle.weights,hi,anchors=anchors,window=window)[0]-batch_reference(data,bundle.weights,lo,anchors=anchors,window=window)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j


def invalid_batch_capture(array):
    def f(x):
        captured=array
        captured*=.8
        ignored=1./captured
        x*=.7
        return x
    return f


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_whole_domain_partial_and_empty_atomic(engine,repeated,ranks):
    mpi(ranks);net,g,syn,array,dt,_,_=batch_model(repeated,False,0,ranks,engine)
    array[:]=[.2,0.]
    syn.namespace['f']=b.Function(invalid_batch_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,trainable_synapse_parameters={syn.name:['h']})
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    cells=bundle.provenance['mutable_capture_layout'][0]['cells']
    empty=t.evaluate(np.zeros((1,1,2)),[0]);np.testing.assert_allclose(np.asarray(empty['final_state'])[0,cells],[.2,0.],rtol=8e-5,atol=8e-6)
    before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):t.step(np.array([[[1.,0.]]]),[0])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0


@pytest.mark.parametrize('repeated',[False,True])
@pytest.mark.parametrize('owned',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_two_calls_update_each_capture_once(engine,repeated,owned,ranks):
    mpi(ranks);net,g,syn,array,dt,_,_=batch_model(repeated,owned,0,ranks,engine)
    syn.pre.code='h=f(h);h=f(h)'+(';v_post+=.11*h' if repeated else '')
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,trainable_synapse_parameters={syn.name:['h']})
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);x=np.ones((1,1,2))
    out=t.evaluate(x,[0]);net.run(dt,namespace={})
    for obj,key,name in [(g,'neuron_state_layout','v'),(syn,'dynamic_state_layout','h')]:
        cells=bundle.provenance[key][obj.name][name]
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    cells=bundle.provenance['mutable_capture_layout'][0]['cells']
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],array,rtol=8e-5,atol=8e-6)


def interleaved_batch_capture(array):
    def f(x):
        captured=array
        captured-=.0625
        ignored=1./captured
        x*=.7
        return x
    return f


@pytest.mark.parametrize('ranks',[None,2])
def test_vectorised_capture_two_calls_read_intermediate_writeback(engine,ranks):
    mpi(ranks);net,g,syn,array,dt,_,_=batch_model(True,True,0,ranks,engine)
    syn.h=[.125,.125];syn.pre.code='h=f(h);h=f(h);v_post+=.11*h'
    syn.namespace['f']=b.Function(interleaved_batch_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,trainable_synapse_parameters={syn.name:['h']})
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.ones((1,1,2)),[0]);net.run(dt,namespace={})
    for obj,key,name in [(g,'neuron_state_layout','v'),(syn,'dynamic_state_layout','h')]:
        cells=bundle.provenance[key][obj.name][name]
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
