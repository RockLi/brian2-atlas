"""One actual selected row broadcasts independently to differently sized views."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_capture_callbacks import add_two_selected
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(mode,ranks,backend,incompatible=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    ids=[0,0,1] if incompatible else [0,1]
    times=np.array([0,2,2] if incompatible else [0,2])*dt
    source=b.SpikeGeneratorGroup(2,ids,times,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(3,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.217,.719,.379];g.u=[.017,.031,.023]
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre='h=f(h)'+(';v_post+=gain*h' if mode=='vectorised' else ''),dt=dt)
    syn.connect(i=[1,0],j=[0,0] if mode=='vectorised' else [0,1]);syn.h=[.113,.173];syn.gain=[.11,.19];syn.delay=dt
    a=g.variables['u'].get_value()[:2];c=g.variables['v'].get_value()
    syn.namespace['f']=b.Function(add_two_selected(a,c),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    x=np.zeros((1,4,2));x[0,0,0]=1;x[0,2,1]=1
    if incompatible:x[0,2,0]=1
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_multishape_original_restore(engine,mode,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(mode,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        compare(out,bundle,g,syn)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)


def compare(out,bundle,g,syn):
    for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
        for name in names:
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_multishape_runtime_failure_atomic(engine,mode,ranks):
    mpi(ranks);net,g,syn,dt,bundle,x=model(mode,ranks,engine,True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    out=t.step(x[:,:3],[0]);net.run(3*dt,namespace={});compare(out,bundle,g,syn)
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks))
    with pytest.raises(ValueError):t.step(x[:,3:4],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


def reference(data,mode,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u']
    h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    assert len(paths)==(2 if mode=='vectorised' else 1)
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick]
            event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());selected=1 if tick==1 else 0 if tick==3 else None
        for stage,path in enumerate(paths):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            old=z.copy();routes=bundle.provenance['delay_queues'][path]['new']
            if stage==0 and selected is not None:
                incoming=old[h[selected]];z[u[:2]]+=incoming;z[v]+=incoming
            for row in routes:
                edge=row['edge'];gate=old[row['states'][0]]
                if stage==0:
                    value=.7*old[h[edge if selected is None else selected]]
                    z[h[edge]]=old[h[edge]]+gate*(value-old[h[edge]])
                else:z[v[0]]+=gate*weights[gain][edge]*old[h[edge]]
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states']
                if cells:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick==0 and row['edge']==1 or tick==2 and row['edge']==0)
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_multishape_all_bank_initial_vjps(engine,mode,window,ranks):
    mpi(ranks);data=model(mode,ranks,engine);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0])
    loss,z,anchors=reference(data,mode,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,hi,window,anchors=anchors)[0]-reference(data,mode,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,bundle.weights,window,hi,anchors)[0]-reference(data,mode,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
