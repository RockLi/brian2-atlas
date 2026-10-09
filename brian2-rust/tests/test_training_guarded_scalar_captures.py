"""A scalar refractory if guards every closure effect and checked expression."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from brian2_rust.training import lif_training_plan
from brian2_rust.training_equations import neuron_parameter_bank
from brian2_rust.training_effects import StateEffectFunction
from brian2_rust.training_event_captures import compile_scalar_event_captures
from test_training_capture_callbacks import add_selected,invalid_guarded_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(backend,ranks,mixed=False,alias=False,invalid=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[0,0,0,0],np.arange(4)*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1 (unless refractory)\nu:1',threshold='v>.5',reset='v-=.5',refractory=.4*b.ms,method='euler',dt=dt)
    g.v=[.4,.7];g.u=[.017,.031];g.lastspike=[0.,-1.]*b.second;g.not_refractory=[False,True]
    code=('h=f(h);' if mixed else '')+'v_post+=gain*f(v_post)'
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=code,dt=dt)
    syn.connect(i=[0,0],j=[0,0]);syn.h=[.173,.113];syn.gain=[.19,.11]
    array=g.variables['v' if alias else 'u'].get_value()
    syn.namespace['f']=b.Function((invalid_guarded_capture if invalid else add_selected)(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    assert bundle.provenance['event_callback_modes'][syn.pre.name]['mode']=='scalar'
    return net,g,syn,dt,bundle,np.ones((1,4,1))


def compare(out,bundle,g,syn):
    for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
        for name in names:
            if name not in bundle.provenance[key][obj.name]:continue
            cells=bundle.provenance[key][obj.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_guarded_scalar_capture_original_restore(engine,mixed,alias,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mixed,alias)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={});compare(out,bundle,g,syn)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)


@pytest.mark.parametrize('ranks',[None,2])
def test_guarded_scalar_capture_domain_atomic(engine,ranks):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,invalid=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    out=t.step(x[:,:1],[0]);net.run(dt,namespace={});compare(out,bundle,g,syn)
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks))
    with pytest.raises(ValueError):t.step(x[:,1:],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks)==before
    with pytest.raises(b.BrianObjectException):net.run(3*dt,namespace={})


def core(backend,ranks,mixed,alias,free):
    weights=[[0.],[0.,0.],[.19,.11]];identity=[[[dict(op='state',index=0)]]]*2
    p=lif_training_plan([1,1,2],projections=[neuron_parameter_bank(len(row)) for row in weights],state_equations=identity,state_resets=identity,
        threshold=.5,detach_reset=False,clock=dict(origin=0.,dt=.0002),trainable=[False,False,True],backend=backend,mpi_ranks=ranks)
    initial=[.2,.4,.7,.173,.113,float(free),.017,.031];actions=[];programs=[]
    for edge in range(2):
        reads=[1,3+edge,5]
        f=StateEffectFunction(('x',),'captured=array\ncaptured+=x\nx*=.7\nreturn x',captured_arrays=(('array',np.zeros(2)),),capture_bindings=(('array','capture'),))
        transform,names,capture_slots=compile_scalar_event_captures(('h=f(h)\n' if mixed else '')+'v+=gain*f(v)',
            {'v':0,'h':1,'free':2},reads,{'f':f,'gain':(2,edge)},
            {'capture':([1,2] if alias else [6,7],'float')},state_types={0:'float',1:'float',2:'boolean'},typed_parameter=lambda bank,index,dtype:(bank,index),write_guards={'v':'free'})
        by_slot=dict(zip(transform['writes'],transform['programs']));winners={}
        for slot in [*sorted(capture_slots),names['h'],names['v']]:
            if slot in by_slot:winners[reads[slot]]=slot
        outputs=[(slot,program) for slot,program in by_slot.items() if winners[reads[slot]]==slot]
        programs.append([program for _,program in outputs]);actions.append(dict(owner=1,reads=list(reads),writes=[reads[slot] for slot,_ in outputs],program_set=len(programs)-1,threshold=None,trigger=None))
    for j in range(3):actions.append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None))
    p.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(initial=initial,initial_parameters=[None]*len(initial),detached=[False]*5+[True,False,False],binary_states=[5],voltage=[0,1,2],program_sets=programs,actions=actions))
    return p,weights


def core_reference(p,weights,mixed,alias,initial=None,anchors=None):
    z=np.array(p['dynamic']['initial'] if initial is None else initial,float);capture=[1,2] if alias else [6,7]
    for edge in range(2):
        v=z[1];h=z[3+edge]
        if mixed:z[capture]+=h;h*=.7
        if z[5]:z[capture]+=v;v+=weights[2][edge]*.7*v
        # Brian loads scalar locals once and writes the declared variables at
        # the end even when its statement's if was false. This also overwrites
        # the closure alias at the same physical voltage address.
        z[1]=v
        if mixed:z[3+edge]=h
    margin=z[:3]-.5;spikes=(margin>0).astype(float)
    if anchors is not None:spikes=(anchors>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors))**2*(margin-anchors)
    logits=spikes[1:]*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,margin


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('free',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_guarded_scalar_capture_all_bank_initial_vjps(engine,mixed,alias,free,ranks):
    mpi(ranks);p,weights=core(engine,ranks,mixed,alias,free)
    out=NativeLIFTrainer(p,weights=weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0]);loss,z,anchors=core_reference(p,weights,mixed,alias)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(weights):
        for index in range(len(row)):
            hi=copy.deepcopy(weights);lo=copy.deepcopy(weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(core_reference(p,hi,mixed,alias,anchors=anchors)[0]-core_reference(p,lo,mixed,alias,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index]:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(p['dynamic']['initial']);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(core_reference(p,weights,mixed,alias,hi,anchors)[0]-core_reference(p,weights,mixed,alias,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


def temporal_reference(data,mixed,alias,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];counter=layout['__refractory_ticks']
    activity=bundle.provenance['refractory_activity_layout'][g.name]
    h=bundle.provenance['dynamic_state_layout'][syn.name].get('h')
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    capture=v if alias else u;before=[];margins=[];hard_rows=[];free_rows=[];spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());free=z[counter]<.5;z[counter]=np.maximum(z[counter]-1,0.)
        if anchors is not None:free=anchors['free'][tick]
        margin=z[v]-.5;hard=(margin>0)&free;event=hard.astype(float)
        if anchors is not None:
            hard=anchors['hard'][tick];old=anchors['margins'][tick]
            event=hard.astype(float)+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());hard_rows.append(hard.copy());free_rows.append(free.copy());spikes.append(event.copy());flag=free&~hard;z[activity]=flag
        for edge in range(2):
            local_v=z[v[0]]
            if mixed:
                local_h=z[h[edge]];z[capture]+=local_h;local_h*=.7
            if flag[0]:z[capture]+=local_v;local_v+=weights[gain][edge]*.7*local_v
            z[v[0]]=local_v
            if mixed:z[h[edge]]=local_h
        z[v]-=.5*event;z[counter]=np.where(hard,1.,z[counter])
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,free=free_rows)


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_guarded_scalar_capture_temporal_all_vjps(engine,mixed,alias,window,ranks):
    mpi(ranks);data=model(engine,ranks,mixed,alias);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=temporal_reference(data,mixed,alias,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name].get('h',[])]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(temporal_reference(data,mixed,alias,hi,window,anchors=anchors)[0]-temporal_reference(data,mixed,alias,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(temporal_reference(data,mixed,alias,bundle.weights,window,hi,anchors)[0]-temporal_reference(data,mixed,alias,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
