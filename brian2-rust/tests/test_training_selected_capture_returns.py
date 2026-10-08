"""Whole event return arrays use actual arrival positions, with shape checks."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_returned_captures import return_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(mode,size,partial,ranks,backend,mutate=False,broadcast=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    # Declaration order reverses source order, so returned column 0 goes edge 1.
    source=b.SpikeGeneratorGroup(2,[0,1]+([0] if partial else [0,1]),np.array([0,0]+([2] if partial else [2,2]))*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.217,.719];g.u=[.017,.031]
    array=np.array([.13,.29,.43][:size]);callback=b.Function((broadcast_return if broadcast else return_capture)(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    code='tmp=f(h);'+('tmp*=.8;' if mutate else '')+'h=tmp'+(';u_post+=gain*h' if mode=='vectorised' else '')
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=code,dt=dt,namespace={'f':callback});syn.connect(i=[1,0],j=[0,0] if mode=='vectorised' else [0,1]);syn.h=[.113,.173];syn.gain=[.11,.19];syn.delay=dt
    net=b.Network(source,hidden,g,syn);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    x=np.zeros((1,4,2));x[0,0,:]=1;x[0,2,0]=1
    if not partial:x[0,2,1]=1
    return net,g,syn,array,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('size',[1,2])
@pytest.mark.parametrize('mutate',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_capture_return_original_restore(engine,mode,size,mutate,ranks,tmp_path):
    mpi(ranks);net,g,syn,array,dt,bundle,x=model(mode,size,size==1,ranks,engine,mutate)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        cells=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']=='array' for alias in row['aliases']))
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],array,rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('size',[2,3])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_capture_return_shape_failure_atomic(engine,mode,size,ranks):
    mpi(ranks);net,_,_,_,dt,bundle,x=model(mode,size,True,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    prefix=3 if size==2 else 1;t.step(x[:,:prefix],[0]);net.run(prefix*dt,namespace={})
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks))
    with pytest.raises(ValueError):t.step(x[:,prefix:prefix+1],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


def reference(data,mode,mutate,weights,window,initial=None,anchors=None):
    _,g,syn,_,_,bundle,_=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    a=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']=='array' for alias in row['aliases']))
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    assert len(paths)==(2 if mode=='vectorised' else 1)
    routes=bundle.provenance['delay_queues'][paths[0]]['new']
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        # Selection/order is detached. Each real row action retains its own
        # continuous queue gate, including the declared counterfactual at zero.
        for row in bundle.provenance['event_callback_snapshots'].get(paths[0],[]):z[row['cache']]=z[row['source']]
        old=z.copy();active=tick in (1,3);returned=old[a]*(.8 if mutate else 1.)
        if active and mutate:z[a]=returned
        for ordinal,row in enumerate(routes):
            edge=row['edge'];gate=old[row['states'][0]]
            value=returned[ordinal if active else 0]
            z[h[edge]]=old[h[edge]]+gate*(value-old[h[edge]])
        if mode=='vectorised':
            for row in bundle.provenance['event_callback_snapshots'].get(paths[1],[]):z[row['cache']]=z[row['source']]
            old=z.copy()
            for row in bundle.provenance['delay_queues'][paths[1]]['new']:
                edge=row['edge'];gate=old[row['states'][0]]
                z[u[0]]+=gate*weights[gain][edge]*old[h[edge]]
        for layout in bundle.provenance['delay_queues'].values():
            for row in layout['new']:
                cells=row['states']
                if cells:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick in (0,2))
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('mutate',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_capture_return_all_bank_initial_vjps(engine,mode,mutate,window,ranks):
    mpi(ranks);data=model(mode,2,False,ranks,engine,mutate);net,g,syn,array,dt,bundle,x=data;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,mode,mutate,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,mutate,hi,window,anchors=anchors)[0]-reference(data,mode,mutate,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,mutate,bundle.weights,window,hi,anchors)[0]-reference(data,mode,mutate,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    if window is None and mode=='vectorised':
        a=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']=='array' for alias in row['aliases']))
        assert all(abs(out['initial_state_gradients'][0][cell])>1e-5 for cell in a)
    net.run(4*dt,namespace={});np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells[:2]],g.v[:],rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def broadcast_return(array):
    def f(x):
        x*=.8
        return x+array
    return f


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('mutate',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_singleton_selected_expression_original_restore(engine,mode,mutate,ranks,tmp_path):
    mpi(ranks);net,g,syn,array,dt,bundle,x=model(mode,1,True,ranks,engine,mutate,broadcast=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        np.testing.assert_array_equal(array,[.13])
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_returned_bank_view_reads_current_bank_after_restore(engine,mode,ranks,tmp_path):
    mpi(ranks);net,g,syn,_,dt,_,x=model(mode,2,False,ranks,engine)
    syn.namespace['f']=b.Function(return_capture(syn.variables['gain'].get_value()[::-1]),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    assert any(row['bank']==bank and row['indices']==[1,0] for row in bundle.provenance['readonly_capture_layout'])
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        if tick==2:
            t.state['weights'][bank]=[.31,.47];syn.gain=[.31,.47]
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def delay_return(array,delays):
    def f(x):
        captured=delays
        captured*=.5
        x*=.8
        return array
    return f


def test_mutable_delay_multi_return_requires_arrival_lane_rebuild():
    from brian2_rust import TrainingConversionError
    net,g,syn,array,_,_,_=model('array',2,False,None,'cpu')
    syn.namespace['f']=b.Function(delay_return(array,syn.pre.variables['delay'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    with pytest.raises(TrainingConversionError,match='fixed delay routing'):
        lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g])
