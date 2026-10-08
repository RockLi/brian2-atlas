"""Captured pathway delay storage follows Brian run-boundary queue latching."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_event_captures import model as event_model
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def capture_delay(array,delta,scale):
    def f(x):
        captured=array
        captured*=scale
        captured+=delta
        x*=.7
        return x
    return f


def model(shared,change,ranks,window,backend):
    net,g,syn,dt,_,x=event_model('gain',ranks,window,1,backend)
    if shared:
        old=syn;syn=b.Synapses(old.source,g,'h:1\ngain:1 (constant)',on_pre='h=f(v_post);v_post+=gain*h',delay=dt,dt=dt)
        syn.connect(i=[0,0],j=[0,0]);syn.h=[.113,.173];syn.gain=[.11,.19];net.remove(old);net.add(syn)
    else:syn.delay=np.array([1,1])*dt
    array=syn.pre.variables['delay'].get_value()
    delta=float(dt) if change=='increase' else 0.;scale=1. if change=='increase' else .5
    syn.namespace['f']=b.Function(capture_delay(array,delta,scale),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    cells=bundle.provenance['pathway_state_layout'][syn.pre.name]['delay']
    assert len(cells)==(1 if shared else 2)
    capture=next(item for item in bundle.provenance['mutable_capture_layout'] if item['cells']==cells)
    assert capture['shape']==[len(array)]
    routes=next(item for item in bundle.plan['dynamic']['delay_layout']['paths'] if item['name']==syn.pre.name)
    assert [item['delay_state'] for item in routes['edges']]==(cells*2 if shared else cells)
    return net,g,syn,dt,bundle,x


def compare(out,data):
    _,g,syn,_,bundle,_=data;z=np.asarray(out['final_state'])[0]
    for obj,key,names in [(g,'neuron_state_layout',['v']),(syn,'dynamic_state_layout',['h'])]:
        for name in names:
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(z[cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    cells=bundle.provenance['pathway_state_layout'][syn.pre.name]['delay']
    np.testing.assert_allclose(z[cells],syn.pre.variables['delay'].get_value(),rtol=8e-5,atol=8e-10)


@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('change',['increase','decrease'])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('segmented',[False,True])
def test_captured_delay_original_run_and_restored_carry(engine,shared,change,ranks,segmented,tmp_path):
    mpi(ranks);data=model(shared,change,ranks,None,engine);net,g,syn,dt,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    if not segmented:
        out=t.evaluate(x,[0]);net.run(4*dt,namespace={});compare(out,data)
    else:
        for tick in range(4):
            out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={});compare(out,data)
            path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,change,weights,initial=None,anchors=None):
    _,g,syn,dt,bundle,x=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for i,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[i]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    delay=bundle.provenance['pathway_state_layout'][syn.pre.name]['delay']
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    routes=bundle.provenance['delay_queues'][syn.pre.name]['new'];before=[];margins=[];hard=[];spikes=[]
    # Runtime-route preparation discards empty imported history cells at this
    # discrete boundary. Keep their expected initial VJP zero, as in the
    # existing event-written-delay oracle, rather than treating them as arrivals.
    for row in routes:z[row['states']]=0.
    # A single native/Brian run latches the emission routes at its boundary.
    # Callback changes are physical state, used when the next run prepares.
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            base=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margin-base)
        spikes.append(event.copy())
        for edge in range(2):
            gate=z[routes[edge]['states'][0]];old=z[v[0]];new_h=.7*old
            candidate=z[delay]+float(dt) if change=='increase' else .5*z[delay]
            z[delay]+=gate*(candidate-z[delay]);z[h[edge]]+=gate*(new_h-z[h[edge]])
            z[v[0]]+=gate*weights[bank][edge]*new_h
        for row in routes:
            cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick in (0,2))
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('change',['increase','decrease'])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_captured_delay_all_bank_initial_vjps(engine,shared,change,ranks,window):
    mpi(ranks);data=model(shared,change,ranks,window,engine);bundle=data[4]
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[5],[0]);loss,z,anchors=reference(data,change,bundle.weights)
    # Queue rebuilding changes private history/latch storage at the run end;
    # the oracle compares stable physical model cells, not allocator bookkeeping.
    physical=[*bundle.provenance['neuron_state_layout'][data[1].name]['v'],
              *bundle.provenance['dynamic_state_layout'][data[2].name]['h'],
              *bundle.provenance['pathway_state_layout'][data[2].pre.name]['delay']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,physical],z[physical],rtol=8e-5,atol=8e-10)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(reference(data,change,hi,anchors=anchors)[0]-reference(data,change,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,j)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j] or j in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(data,change,bundle.weights,hi,anchors)[0]-reference(data,change,bundle.weights,lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j


@pytest.mark.parametrize('issue',['missing_capture_flag','nonshared_flag'])
def test_native_captured_shared_delay_control_validation(issue):
    data=model(issue=='missing_capture_flag','increase',None,None,'cpu');bundle=data[4]
    p=copy.deepcopy(bundle.plan);paths=p['dynamic']['delay_layout']['paths']
    path=next(row for row in paths if row['name']==data[2].pre.name)
    if issue=='missing_capture_flag':path.pop('captured_shared_delay')
    else:path['captured_shared_delay']=True
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='scalar pathway delay is read.only|captured shared delay requires scalar'):
        t.step(data[5],[0])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0


@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('change',['increase','decrease'])
@pytest.mark.parametrize('ranks',[None,2])
def test_captured_delay_imported_arrivals_and_checkpoint(engine,shared,change,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,_,_=model(shared,change,ranks,None,engine)
    net.run(dt,namespace={})
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    pending=bundle.provenance['delay_queues'][syn.pre.name]['pending'];assert len(pending)==2
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    x=np.zeros((1,3,1));x[0,1,0]=1;data=(net,g,syn,dt,bundle,x)
    for tick in range(3):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={});compare(out,data)
        checkpoint=tmp_path/str(tick);t.store(checkpoint);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(checkpoint)
