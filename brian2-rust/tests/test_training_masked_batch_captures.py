"""Filtered callback arguments differ from the nonempty arriving event batch."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_capture_callbacks import add_selected
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(backend,ranks,mode,alias=False,empty=False,continuous=True):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    size=3 if mode=='vectorised' else 2
    ids=[*range(size),0,size-1];times=np.array([0]*size+[1,2])*dt
    source=b.SpikeGeneratorGroup(size,ids,times,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1 (unless refractory)\nu:1',threshold='v>.5',reset='v-=.5',refractory=.4*b.ms,method='euler',dt=dt)
    g.v=[.217,.719];g.u=[.017,.031];g.lastspike=np.array([0.,0.] if empty else [-1.,-1.])*b.second
    syn=b.Synapses(source,g,('dh/dt=0/second:1 (clock-driven)' if continuous else 'h:1')+'\ngain:1 (constant)',on_pre='v_post+=gain*f(h)',method='euler',dt=dt)
    syn.connect(i=list(range(size)),j=[0,0,1] if size==3 else [0,1]);syn.h=[.033,.053,.047][:size];syn.gain=[.11,.19,.17][:size];syn.delay=dt
    if mode=='array':
        assert len(set(np.asarray(syn.j[:],int)))==len(syn)
        syn.variables['_postsynaptic_idx'].unique=True
    array=g.variables['v' if alias else 'u'].get_value()
    syn.namespace['f']=b.Function(add_selected(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    path=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1]
    assert bundle.provenance['event_callback_modes'][path]['mode']==mode
    x=np.zeros((1,5,size));x[0,0,:]=1;x[0,1,0]=1;x[0,2,-1]=1
    return net,g,syn,dt,bundle,x


def compare(out,bundle,g,syn):
    for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
        for name in names:
            if name not in bundle.provenance[key][obj.name]:continue
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_batch_capture_original_restore(engine,mode,alias,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,alias)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(5):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={});compare(out,bundle,g,syn)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_batch_empty_argument_shape_atomic(engine,mode,ranks):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,empty=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    out=t.step(x[:,:1],[0]);net.run(dt,namespace={});compare(out,bundle,g,syn)
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks))
    with pytest.raises(ValueError):t.step(x[:,1:2],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


def reference(data,mode,alias,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];counter=layout['__refractory_ticks']
    activity=bundle.provenance['refractory_activity_layout'][g.name]
    h=bundle.provenance['dynamic_state_layout'][syn.name].get('h')
    h_bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['h'])
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    path=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1]
    rows=bundle.provenance['delay_queues'][path]['new'];posts=np.asarray(syn.j[:],int)
    capture=v if alias else u;before=[];margins=[];hard_rows=[];free_rows=[];raw_rows=[];spikes=[]
    for tick in range(5):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());free=z[counter]<.5;z[counter]=np.maximum(z[counter]-1,0.)
        if anchors is not None:free=anchors['free'][tick]
        margin=z[v]-.5;hard=(margin>0)&free;event=hard.astype(float)
        if anchors is not None:
            hard=anchors['hard'][tick];old=anchors['margins'][tick]
            event=hard.astype(float)+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());hard_rows.append(hard.copy());free_rows.append(free.copy());spikes.append(event.copy());flag=free&~hard;z[activity]=flag
        for copies in bundle.provenance['event_callback_snapshots'].values():
            for row in copies:z[row['cache']]=z[row['source']]
        old=z.copy();raw=[row['edge'] for row in rows if old[row['states'][0]]>.5]
        h_values=np.asarray(weights[h_bank]) if h is None else old[h]
        if anchors is not None:raw=anchors['raw'][tick]
        raw_rows.append(list(raw));active=[edge for edge in raw if flag[posts[edge]]]
        if active:
            incoming=h_values[active]
            z[capture]+=incoming[0] if len(incoming)==1 else incoming
        for index,row in enumerate(rows):
            edge=row['edge'];post=posts[edge];amplitude=old[row['states'][0]];live=z[v[post]]
            ordinal=sum(other['edge'] in active for other in rows[:index])
            selected=active[0] if len(active)==1 else active[ordinal] if ordinal<len(active) else edge
            value=.7*h_values[selected]
            base=old[v[post]] if mode=='array' else live
            new=base+(weights[gain][edge]*value if flag[post] else 0.)
            z[v[post]]=live+amplitude*(new-live)
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];edge=row['edge']
                z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick==0 or tick==1 and edge==0 or tick==2 and edge==len(posts)-1)
        z[v]-=.5*event;z[counter]=np.where(hard,1.,z[counter])
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,free=free_rows,raw=raw_rows)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_batch_all_bank_initial_vjps(engine,mode,alias,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,alias);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,mode,alias,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,alias,hi,window,anchors=anchors)[0]-reference(data,mode,alias,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,alias,bundle.weights,window,hi,anchors)[0]-reference(data,mode,alias,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_batch_parameter_arguments_all_vjps(engine,mode,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,True,continuous=False);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,mode,True,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,True,hi,window,anchors=anchors)[0]-reference(data,mode,True,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,True,bundle.weights,window,hi,anchors)[0]-reference(data,mode,True,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_batch_parameter_banks_carry_restore(engine,mode,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,continuous=False)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    banks={name:next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==[name]) for name in ['h','gain']}
    for tick in range(5):
        if tick==2:
            for name in ['h','gain']:
                values=np.asarray(t.state['weights'][banks[name]])*1.03+.0003;t.state['weights'][banks[name]]=values.tolist();setattr(syn,name,values)
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={});compare(out,bundle,g,syn)
        for name in ['h','gain']:np.testing.assert_allclose(t.state['weights'][banks[name]],syn.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
