"""Subgroup endpoint admission, classic projections, and summed view boundaries."""
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_training,lower_brian_dynamic_training,TrainingConversionError
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine


def base():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,1,0,1,1],[1,0,1,0,1],[0,1,1,1,0],[1,1,0,0,1],[0,0,1,1,1]],float)
    inp=b.NeuronGroup(5,'v:1',threshold='external_events(t,i)>0',reset='',dt=dt,
        namespace={'external_events':b.TimedArray(x,dt=dt)},name='edge_input')
    groups=[b.NeuronGroup(6,'dv/dt=-v/ms:1',threshold='v>.6',reset='v-=.6',method='euler',dt=dt,name=f'edge_g{k}') for k in range(2)]
    for g in groups:g.v=[.1,.3,.8,.4,.9,.2]
    return inp,groups,x,dt


@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('layout',['source','target','both'])
def test_subgroup_classic_projection_matches_cython(engine,dynamic,shared,layout):
    inp,groups,x,dt=base();views=[];synapses=[]
    for k,(source,target) in enumerate(((inp,groups[0]),(groups[0],groups[1]))):
        if layout in ('source','both'):source=source[1:4];views.append(source)
        if layout in ('target','both'):target=target[2:5];views.append(target)
        s=b.Synapses(source,target,'w:1'+(' (shared)' if shared else ''),on_pre='v_post+=w',dt=dt,name=f'edge_s{k}')
        s.connect(i=[2,0,1,2],j=[1,2,0,2]);s.w=.21 if shared else [.12,.21,.31,.23];synapses.append(s)
    net=b.Network(inp,*groups,*synapses,*views)
    bundle=lower_brian_training(net,input_group=inp,layers=groups,dynamic=dynamic,backend=engine)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0],**({} if dynamic else dict(initial=[bundle.initial_membrane])))
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(monitors);net.run(len(x)*dt,namespace={})
    assert all(s.pre.codeobj.compiled_code['run'] is not None for s in synapses)
    expected=np.zeros((len(x),12))
    for k,mon in enumerate(monitors):expected[np.rint(np.asarray(mon.t/b.second)/float(dt)).astype(int),np.asarray(mon.i)+6*k]=1
    np.testing.assert_array_equal(result['spikes'][0],expected)
    np.testing.assert_allclose(result['final_membrane'][0],np.r_[groups[0].v[:],groups[1].v[:]],rtol=3e-5 if engine!='cpu' else 5e-13,atol=3e-7 if engine!='cpu' else 1e-14)
    if engine!='cpu':assert result['gpu_dispatches']>0


def summed_model(direction='post',overlap=False,empty=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=dt,name='sumview_input')
    groups=[b.NeuronGroup(6,'dv/dt=(-v+.1*q)/ms:1\nq:1',threshold='v>100',reset='v=0',method='euler',dt=dt,name=f'sumview_g{k}') for k in range(2)]
    g=groups[0];g.q=np.arange(6)+1.;synapses=[]
    for k,(start,stop) in enumerate(((0,2),(1,4) if overlap else (3,5))):
        target=g[start:stop];peer=g[1:5]
        source,dest=(target,peer) if direction=='pre' else (peer,target)
        other='post' if direction=='pre' else 'pre'
        s=b.Synapses(source,dest,f'w:1\nq_{direction}=w+.2*q_{other}+.01*i+.02*j:1 (summed)',dt=dt,name=f'sumview_s{k}')
        if empty:s.connect(condition='False')
        elif direction=='pre':s.connect(i=[1,0,1],j=[3,1,0]);s.w=[.2,.4,.3]
        else:s.connect(i=[3,1,0],j=[1,0,1]);s.w=[.2,.4,.3]
        synapses.append(s)
    return b.Network(inp,*groups,*synapses),inp,groups,synapses,dt


@pytest.mark.parametrize('direction',['pre','post'])
@pytest.mark.parametrize('empty',[False,True])
def test_disjoint_summed_subgroups_clear_only_their_views(engine,direction,empty):
    net,inp,groups,synapses,dt=summed_model(direction,empty=empty)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(np.zeros((1,4,1)),[0]);net.run(4*dt,namespace={})
    z=np.asarray(result['final_state'])[0]
    for g in groups:
        for name,slots in bundle.provenance['neuron_state_layout'][g.name].items():np.testing.assert_allclose(z[slots],g.variables[name].get_value(),rtol=3e-5 if engine!='cpu' else 1e-13,atol=3e-7 if engine!='cpu' else 1e-14)
    slots=bundle.provenance['neuron_state_layout'][groups[0].name]['q'];np.testing.assert_array_equal(z[np.asarray(slots)[[2,5]]],[3.,6.])
    assert all(u.codeobj.compiled_code['run'] is not None for s in synapses for u in s.summed_updaters.values())
    if engine!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('direction',['pre','post'])
def test_overlapping_summed_views_keep_brian_rejection(direction):
    net,inp,groups,_,dt=summed_model(direction,overlap=True)
    with pytest.raises(TrainingConversionError,match='overlapping'):lower_brian_dynamic_training(net,input_group=inp,layers=groups)
    with pytest.raises(NotImplementedError,match='overlapping'):net.run(0*dt,namespace={})


@pytest.mark.parametrize('dynamic',[False,True])
def test_unselected_parent_remains_rejected(dynamic):
    inp,groups,x,dt=base()
    external=b.NeuronGroup(4,'v:1',threshold='v>1',reset='v=0',dt=dt,name='unselected')
    s=b.Synapses(external[1:3],groups[0][2:4],'w:1',on_pre='v_post+=w',dt=dt);s.connect(j='i');s.w=.1
    net=b.Network(inp,*groups,s)
    with pytest.raises(TrainingConversionError,match='endpoints must belong to selected'):
        lower_brian_training(net,input_group=inp,layers=groups,dynamic=dynamic)


def test_cython_queue_locals_cannot_shadow_legal_model_names():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,1,1,1],[0,0,0,0],[1,0,1,0],[0,0,0,0]],float)
    inp=b.NeuronGroup(4,'v:1',threshold='events(t,i)>0',reset='',dt=dt,namespace={'events':b.TimedArray(x,dt=dt)})
    out=b.NeuronGroup(3,'v:1',dt=dt)
    names=['capsule','cpp_queue','spike_vector','num_spikes','spike_data','synapse_id']
    s=b.Synapses(inp[1:4],out,'\n'.join(n+':1 (constant)' for n in names),on_pre='v_post+=i+'+'+'.join(names),dt=dt)
    ii=np.array([2,0,1,2]);jj=np.array([1,0,2,1]);s.connect(i=ii,j=jj)
    for name in names:setattr(s,name,[1,2,3,4])
    delays=np.array([0,1,2,0]);s.delay=delays*dt;net=b.Network(inp,out,s);net.run(6*dt,namespace={})
    assert s.pre.codeobj.compiled_code['run'] is not None
    expected=np.zeros(3)
    for tick,external in enumerate(x):
        for edge,(i,j) in enumerate(zip(ii,jj)):
            if external[i+1] and tick+delays[edge]<6:expected[j]+=i+len(names)*(edge+1)
    np.testing.assert_array_equal(out.v[:],expected)
