"""Whole capture writes consume the actual compact selected event vector."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def add_selected(array):
    def f(x):
        captured=array
        captured+=x
        x*=.7
        return x
    return f


from test_training_capture_callbacks import add_selected as _add_selected,discard_selected as _discard_selected


def model(mode,ranks,backend,delay=1,coefficient=False,discard=False,short=False,accumulator_input=False,captured_accumulator=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,0,1] if short else [0,1,0],np.array([0,2,2] if short else [0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.217,.719];g.u=[.017,.031]
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=('h=f(v_post)' if accumulator_input else 'h=f(gain)' if coefficient else 'h=f(h)')+(';v_post+=gain*h' if mode=='vectorised' else ''),dt=dt)
    syn.connect(i=[1,0],j=[0,0] if mode=='vectorised' else [0,1]);syn.h=[.113,.173];syn.gain=[.11,.19];syn.delay=delay*dt
    array=np.array([.23]) if short else g.variables['v' if captured_accumulator else 'u'].get_value()
    syn.namespace['f']=b.Function((_discard_selected if discard else _add_selected)(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    x=np.zeros((1,4,2));x[0,0,:]=1;x[0,2,0]=1
    if short:x[0,0,1]=0;x[0,2,1]=1
    return net,g,syn,array,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_selected_original_partial_empty_restore(engine,mode,delay,ranks,tmp_path):
    mpi(ranks);net,g,syn,array,dt,bundle,x=model(mode,ranks,engine,delay)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,mode,delay,weights,window,initial=None,anchors=None,coefficient=False,discard=False,accumulator_input=False,captured_accumulator=False):
    _,g,syn,_,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u']
    h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    stages=bundle.provenance['event_callback_stage_groups'].get(syn.pre.name)
    path=stages[1] if stages else syn.pre.name
    scalar=bundle.provenance['event_callback_modes'][path]['mode']=='scalar'
    paths=stages[1:] if stages else [path]
    assert scalar or len(paths)==(2 if mode=='vectorised' else 1)
    routes=bundle.provenance['delay_queues'][path]['new']
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old_margin=anchors['margins'][tick]
            event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old_margin))**2*(margin-old_margin)
        spikes.append(event.copy())
        for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
        old=z.copy();arrival=tick-delay
        selected=[row['edge'] for row in routes if arrival==0 or arrival==2 and row['edge']==1]
        incoming=old[v][np.asarray(syn.j[:],dtype=int)] if accumulator_input else np.asarray(weights[gain]) if coefficient else old[h]
        if scalar:
            # A dependency on an accumulated postsynaptic variable makes Brian
            # choose its scalar fallback. Each callback observes prior writes.
            for row in routes:
                edge=row['edge'];post=int(syn.j[edge]);before_event=z.copy()
                gate=before_event[row['states'][0]] if delay else float(edge in selected)
                argument=before_event[v[post]] if accumulator_input else weights[gain][edge] if coefficient else before_event[h[edge]]
                value=.7*argument
                if not discard:z[v if captured_accumulator else u]+=gate*argument
                z[h[edge]]=before_event[h[edge]]+gate*(value-before_event[h[edge]])
                if mode=='vectorised':
                    # Scalar writeback of the local voltage wins over a raw
                    # capture update to the same physical destination.
                    z[v[post]]=before_event[v[post]]+gate*weights[gain][edge]*value
        else:
            if selected and not discard:z[v if captured_accumulator else u]+=incoming[selected]
            for ordinal,row in enumerate(routes):
                edge=row['edge'];gate=old[row['states'][0]] if delay else float(edge in selected)
                if selected:
                    chosen=selected[0] if len(selected)==1 else selected[ordinal]
                    value=.7*incoming[chosen]
                else:value=.7*incoming[edge]
                z[h[edge]]=old[h[edge]]+gate*(value-old[h[edge]])
            if mode=='vectorised':
                tail=paths[1]
                for row in bundle.provenance['event_callback_snapshots'].get(tail,[]):z[row['cache']]=z[row['source']]
                scatter_old=z.copy()
                for row in bundle.provenance['delay_queues'][tail]['new']:
                    edge=row['edge'];gate=scatter_old[row['states'][0]] if delay else float(edge in selected)
                    z[v[0]]+=gate*weights[gain][edge]*scatter_old[h[edge]]
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states']
                if cells:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick==0 or tick==2 and row['edge']==1)
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_selected_all_bank_initial_vjps(engine,mode,delay,window,ranks):
    mpi(ranks);data=model(mode,ranks,engine,delay)
    check_gradients(data,mode,delay,window)


def check_gradients(data,mode,delay,window,coefficient=False,discard=False,accumulator_input=False,captured_accumulator=False):
    _,g,syn,_,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0])
    loss,z,anchors=reference(data,mode,delay,bundle.weights,window,coefficient=coefficient,discard=discard,accumulator_input=accumulator_input,captured_accumulator=captured_accumulator)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,delay,hi,window,anchors=anchors,coefficient=coefficient,discard=discard,accumulator_input=accumulator_input,captured_accumulator=captured_accumulator)[0]-reference(data,mode,delay,lo,window,anchors=anchors,coefficient=coefficient,discard=discard,accumulator_input=accumulator_input,captured_accumulator=captured_accumulator)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,delay,bundle.weights,window,hi,anchors,coefficient,discard,accumulator_input,captured_accumulator)[0]-reference(data,mode,delay,bundle.weights,window,lo,anchors,coefficient,discard,accumulator_input,captured_accumulator)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    if window is None:
        assert all(abs(out['initial_state_gradients'][0][cell])>1e-5 for cell in u_cells(bundle,g))


def u_cells(bundle,g):
    return bundle.provenance['neuron_state_layout'][g.name]['u']


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_selected_coefficient_all_bank_initial_vjps(engine,mode,window,ranks):
    mpi(ranks);data=model(mode,ranks,engine,coefficient=True)
    check_gradients(data,mode,1,window,coefficient=True)


def discard_selected(array):
    def f(x):
        temporary=x+array
        temporary*=.8
        x*=.7
        return x
    return f


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_single_selected_broadcast_discarded_original_restore(engine,mode,ranks,tmp_path):
    mpi(ranks);net,g,syn,_,dt,bundle,x=model(mode,ranks,engine,discard=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_single_selected_broadcast_discarded_all_vjps(engine,mode,window,ranks):
    mpi(ranks);data=model(mode,ranks,engine,discard=True)
    check_gradients(data,mode,1,window,discard=True)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_singleton_capture_inplace_runtime_shape_atomic(engine,mode,ranks):
    mpi(ranks);net,_,_,array,dt,bundle,x=model(mode,ranks,engine,short=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    out=t.step(x[:,:3],[0]);net.run(3*dt,namespace={})
    captured=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']=='array' for alias in row['aliases']))
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,captured],array,rtol=8e-5,atol=8e-6)
    np.testing.assert_allclose(array,[.403],rtol=8e-5,atol=8e-6)
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks))
    with pytest.raises(ValueError):t.step(x[:,3:4],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_accumulator_argument_original_partial_restore(engine,mode,ranks,tmp_path):
    mpi(ranks);net,g,syn,_,dt,bundle,x=model(mode,ranks,engine,accumulator_input=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_accumulator_argument_all_bank_initial_vjps(engine,mode,window,ranks):
    mpi(ranks);data=model(mode,ranks,engine,accumulator_input=True)
    check_gradients(data,mode,1,window,accumulator_input=True)


@pytest.mark.parametrize('accumulator_input',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_captured_accumulator_all_bank_initial_vjps(engine,accumulator_input,window,ranks):
    mpi(ranks);data=model('vectorised',ranks,engine,accumulator_input=accumulator_input,captured_accumulator=True)
    check_gradients(data,'vectorised',1,window,accumulator_input=accumulator_input,captured_accumulator=True)
    net,g,syn,_,dt,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);net.run(4*dt,namespace={})
    for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
        for name in names:
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
