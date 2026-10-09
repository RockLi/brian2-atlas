"""Scalar FIFO event callbacks mutate full closure arrays once per arrival."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def event_capture(array):
    def f(x):
        captured=array
        captured*=.8
        x*=.7
        return x
    return f


def model(kind,ranks,window,delay,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.217,.379]
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre='h=f(v_post);v_post+=gain*h',dt=dt)
    syn.connect(i=[0,0],j=[0,0]);syn.h=[.113,.173];syn.gain=[.11,.19];syn.delay=delay*dt
    array=(syn if kind=='gain' else g).variables['gain' if kind=='gain' else 'v'].get_value()
    syn.namespace['f']=b.Function(event_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    assert bundle.provenance['event_callback_modes'][syn.pre.name]['mode']=='scalar'
    x=np.zeros((1,4,1));x[0,[0,2],0]=1
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('kind',['gain','voltage'])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('delay',[0,1])
def test_scalar_event_capture_actual_original_restore(engine,kind,ranks,delay,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(kind,ranks,None,delay,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v']),(syn,'dynamic_state_layout',['h',*(['gain'] if kind=='gain' else [])])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,kind,delay,weights,initial=None,anchors=None):
    _,g,syn,_,bundle,x=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for i,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[i]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];layout=bundle.provenance['dynamic_state_layout'][syn.name];h=layout['h'];gain=layout.get('gain')
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and 'gain' in row['variables'])
    routes=bundle.provenance['delay_queues'][syn.pre.name]['new'];before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        for edge in range(2):
            gate=z[routes[edge]['states'][0]] if delay else float(tick in (0,2))
            old=z[v].copy();coefficient=z[gain[edge]] if gain is not None else weights[bank][edge]
            new_h=.7*old[0]
            if kind=='gain':z[gain]+=gate*(-.2*z[gain])
            else:z[v]+=gate*(-.2*z[v])
            z[h[edge]]+=gate*(new_h-z[h[edge]])
            z[v[0]]=old[0]+gate*coefficient*new_h
        for row in routes:
            cells=row['states']
            if cells:
                z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick in (0,2))
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('kind',['gain','voltage'])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
def test_scalar_event_capture_all_bank_initial_vjps(engine,kind,ranks,delay,window):
    mpi(ranks);data=model(kind,ranks,window,delay,engine);bundle=data[4]
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[5],[0]);loss,z,anchors=reference(data,kind,delay,bundle.weights)
    np.testing.assert_allclose(np.asarray(out['final_state'])[0],z,rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(reference(data,kind,delay,hi,anchors=anchors)[0]-reference(data,kind,delay,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,j)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j] or j in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(data,kind,delay,bundle.weights,hi,anchors=anchors)[0]-reference(data,kind,delay,bundle.weights,lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j


def invalid_event_capture(array):
    def f(x):
        captured=array
        captured*=.8
        ignored=1./captured
        x*=.7
        return x
    return f


@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_event_capture_overwritten_domain_and_empty_batch(engine,ranks):
    mpi(ranks);net,g,syn,dt,_,_=model('voltage',ranks,None,0,engine)
    g.v=[.2,0.];syn.variables['_synaptic_post'].get_value()[:]=1
    syn.namespace['f']=b.Function(invalid_event_capture(g.variables['v'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    bundle=lower_brian_dynamic_training(net,input_group=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup)),
        layers=[next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g),g],backend=engine,mpi_ranks=ranks,
        trainable_synapse_parameters={syn.name:['h','gain']})
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    empty=t.evaluate(np.zeros((1,1,1)),[0]);cells=bundle.provenance['neuron_state_layout'][g.name]['v']
    expected=np.array([.2,0.],dtype=np.float64 if engine=='cpu' else np.float32)
    np.testing.assert_array_equal(np.asarray(empty['final_state'])[0,cells],expected.astype(float))
    before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):t.step(np.ones((1,1,1)),[0])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0
    with np.errstate(divide='ignore',invalid='ignore'):net.run(dt,namespace={})
    np.testing.assert_allclose(g.v[:],[.2*.8**2,0.],rtol=1e-12,atol=1e-12)


def integer_event_capture(array):
    def f(x):
        captured=array
        captured+=1
        x*=.7
        return x
    return f


def boolean_event_capture(array):
    def f(x):
        captured=array
        captured+=True
        x*=.7
        return x
    return f


@pytest.mark.parametrize('dtype',['integer','boolean'])
@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_event_typed_whole_capture(engine,dtype,ranks):
    mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.217,.379]
    syn=b.Synapses(source,g,'h:1\nk:'+dtype+' (constant)',on_pre='h=f(v_post);v_post+=.11*h',dt=dt)
    syn.connect(i=[0,0],j=[0,0]);syn.h=[.113,.173];syn.k=[2,3] if dtype=='integer' else [False,True]
    array=syn.variables['k'].get_value();syn.namespace['f']=b.Function((integer_event_capture if dtype=='integer' else boolean_event_capture)(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,trainable_synapse_parameters={syn.name:['h']})
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(np.array([[[float(tick in (0,2))]]]),[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v']),(syn,'dynamic_state_layout',['h','k'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        cells=bundle.provenance['dynamic_state_layout'][syn.name]['k']
        np.testing.assert_array_equal(np.asarray(out['initial_state_gradients'])[0,cells],0.)


def readonly_event_capture(array):
    def f(x):
        ignored=1./array
        x*=.7
        return x
    return f


@pytest.mark.parametrize('ranks',[None,2])
def test_scalar_event_readonly_capture_uses_updated_optimizer_bank(engine,ranks):
    mpi(ranks);net,g,syn,dt,_,_=model('gain',ranks,None,0,engine)
    syn.namespace['f']=b.Function(readonly_event_capture(syn.variables['gain'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,trainable_synapse_parameters={syn.name:['h','gain']})
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    weights=copy.deepcopy(bundle.weights);weights[bank][1]=0.
    t=NativeLIFTrainer(bundle.plan,weights=weights,runner=RUNNER);before=copy.deepcopy(t.state)
    t.evaluate(np.zeros((1,1,1)),[0])
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):t.step(np.ones((1,1,1)),[0])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0


def returned_event_capture(array):
    def f(x):
        x*=.7
        return array
    return f


@pytest.mark.parametrize('case',['returned_vector','context_budget'])
def test_scalar_event_capture_shape_and_context_bounds_without_execution(case):
    from brian2_rust.training_effects import lower_state_effect_function,bind_state_effect_captures
    from brian2_rust.training_event_captures import compile_scalar_event_captures
    array=np.ones(2 if case=='returned_vector' else 64)
    callback=returned_event_capture if case=='returned_vector' else event_capture
    f=bind_state_effect_captures(lower_state_effect_function(callback(array)),{'array':'captured'})
    with pytest.raises(ValueError,match='scalar event writeback|64 context slots'):
        compile_scalar_event_captures('h=f(v)',{'v':0,'h':1},[0,1],{'f':f},
            {'captured':(list(range(2,2+len(array))),'float')},state_types={0:'float',1:'float'},typed_parameter=None)
    np.testing.assert_array_equal(array,np.ones(len(array)))
