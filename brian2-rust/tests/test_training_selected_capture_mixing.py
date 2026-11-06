"""Mixed capture expressions evaluate only the actual selected column."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def subtract_capture(array):
    def f(x):
        x *= .8
        return x-array
    return f


def model(mode, ranks, backend,partial=False):
    b.set_device('runtime'); b.start_scope(); b.prefs.codegen.target='numpy'
    dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0] if partial else [0,1],np.zeros(1 if partial else 2)*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.217,.719]; g.u=[.017,.031]
    array=np.array([.13,.29])
    callback=b.Function(subtract_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre='h=sqrt(f(h))'+(';u_post+=gain*h' if mode=='vectorised' else ''),dt=dt,namespace={'f':callback})
    syn.connect(i=[1,0],j=[0,0] if mode=='vectorised' else [0,1])
    # Arrival order is edge 1, edge 0. The unused pairing .8*.3-.29
    # is negative, whereas both actual selected pairings are positive.
    syn.h=[.7,.3]; syn.gain=[.11,.19]; syn.delay=dt
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    x=np.zeros((1,3,2)); x[0,0,0]=1
    if not partial:x[0,0,1]=1
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_mixed_selected_domain_original_restore(engine,mode,ranks,tmp_path):
    mpi(ranks); net,g,syn,dt,bundle,x=model(mode,ranks,engine)
    p=copy.deepcopy(bundle.plan); p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(3):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None)
        net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick); t.store(path); t=NativeLIFTrainer(t.plan,runner=RUNNER); t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,mode,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data; p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name]; v=layout['v']; u=layout['u']
    h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    a=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']=='array' for alias in row['aliases']))
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    assert len(paths)==(2 if mode=='vectorised' else 1)
    routes=bundle.provenance['delay_queues'][paths[0]]['new']
    before=[]; margins=[]; hard=[]; spikes=[]
    for tick in range(3):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy()); z[v]+=.2*z[u]; margin=z[v]-.5; event=(margin>0).astype(float)
        margins.append(margin.copy()); hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick]
            event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        for row in bundle.provenance['event_callback_snapshots'].get(paths[0],[]):z[row['cache']]=z[row['source']]
        old=z.copy()
        for ordinal,row in enumerate(routes):
            edge=row['edge']; gate=old[row['states'][0]]
            column=ordinal if tick==1 else 0
            value=np.sqrt(.8*old[h[edge]]-old[a[column]])
            z[h[edge]]=old[h[edge]]+gate*(value-old[h[edge]])
        if mode=='vectorised':
            for row in bundle.provenance['event_callback_snapshots'].get(paths[1],[]):z[row['cache']]=z[row['source']]
            old=z.copy()
            for row in bundle.provenance['delay_queues'][paths[1]]['new']:
                edge=row['edge'];gate=old[row['states'][0]]
                z[u[0]]+=gate*weights[gain][edge]*old[h[edge]]
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states']
                if cells:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick==0)
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_mixed_selected_all_bank_initial_vjps(engine,mode,window,ranks):
    mpi(ranks); data=model(mode,ranks,engine); _,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan); p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0])
    loss,z,anchors=reference(data,mode,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights); lo=copy.deepcopy(bundle.weights)
            hi[bank][index]+=1e-6; lo[bank][index]-=1e-6
            fd=(reference(data,mode,hi,window,anchors=anchors)[0]-reference(data,mode,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state); lo=hi.copy(); hi[index]+=1e-6; lo[index]-=1e-6
        fd=(reference(data,mode,bundle.weights,window,hi,anchors)[0]-reference(data,mode,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    if mode=='vectorised' and window is None:
        a=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']=='array' for alias in row['aliases']))
        assert all(abs(out['initial_state_gradients'][0][cell])>1e-5 for cell in a)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_mixed_selected_shape_failure_atomic(engine,mode,ranks):
    mpi(ranks); net,_,_,dt,bundle,x=model(mode,ranks,engine,partial=True)
    p=copy.deepcopy(bundle.plan); p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    t.step(x[:,:1],[0]); net.run(dt,namespace={})
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks))
    with pytest.raises(ValueError):t.step(x[:,1:2],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
