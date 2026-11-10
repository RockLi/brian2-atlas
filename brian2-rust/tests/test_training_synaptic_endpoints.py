"""Physical synaptic endpoints: Cython ordering, alias addresses and VJPs."""
import copy
import os

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine


PRE=np.array([0,1,0,1]);POST=np.array([0,1,1,0])
MOD_PRE=np.array([0,1,1,0,0]);MOD_POST=np.array([3,0,3,1,2])
NEST_PRE=np.array([0,1,0]);NEST_POST=np.array([4,0,3])
PATTERN=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1]],float)


def model(before=True,*,summed=False,feedback=False,delayed=False,nested=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    ticks,ids=np.nonzero(PATTERN)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='endpoint_input')
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,
                         name=f'endpoint_layer_{k}') for k in range(2)]
    groups[1].v=[.64,.29]
    conn=b.Synapses(inp,groups[1],'w:1',on_pre='v_post+=w',dt=dt,name='z_endpoint_conn')
    conn.connect(i=PRE,j=POST);conn.w=[.24,.35,.18,.21];conn.pre.order=0
    # Allocated before conn lexically: canonical writes must be discovered first.
    if summed:
        mod=b.Synapses(inp,conn,'w_post=gain*(1+.1*i_post+.2*j_post):1 (summed)\ngain:1',dt=dt,name='a_endpoint_mod')
    else:
        mod=b.Synapses(inp,conn,'gain:1',on_pre='w_post+=gain*(1+.1*i_post+.2*j_post)',dt=dt,name='a_endpoint_mod')
        mod.pre.order=-2 if before else 2
    mod.connect(i=MOD_PRE,j=MOD_POST);mod.gain=[.03,.04,.02,.01,.025]
    if delayed:
        assert not summed
        conn.delay=np.array([1,2,0,1])*dt;mod.delay=np.array([0,1,2,0,1])*dt
    objects=[inp,*groups,conn,mod]
    if nested:
        outer=b.Synapses(inp,mod,'eta:1',on_pre='gain_post+=eta+.01*gain_post',dt=dt,name='b_endpoint_outer')
        outer.connect(i=NEST_PRE,j=NEST_POST);outer.eta=[.008,.006,.007];outer.pre.order=-3;objects.append(outer)
    reader=None
    if feedback:
        reader=b.Synapses(conn,groups[1],on_post='v_post+=.02*w_pre+.01*i_pre+.015*j_pre',dt=dt,name='endpoint_reader')
        reader.connect(i=[3,0,2],j=[0,1,0]);reader.post.order=3;objects.append(reader)
    return b.Network(*objects),inp,groups,conn,mod,reader


def oracle(weights,gains,initial,before,*,summed=False,feedback=False,delayed=False,nested=None,anchors=None):
    """Hand update in physical edge order; smooth only the declared spike VJP."""
    v=np.asarray(initial,float).copy();w=np.asarray(weights,float).copy();gains=np.asarray(gains,float).copy();events=[];membranes=[]
    factors=1+.1*PRE[MOD_POST]+.2*POST[MOD_POST]
    for x in PATTERN:
        u=.8*v;hard=(u>.5).astype(float)
        t=len(events)
        if anchors is None:s=hard
        else:
            base_u,base_s=anchors
            phi=1/(1+5*np.abs(base_u[t]-.5))**2
            s=base_s[t]+phi*(u-base_u[t])
        v=u.copy()
        if nested is not None:
            for k,target in enumerate(NEST_POST):
                gains[target]+=x[NEST_PRE[k]]*(nested[k]+.01*gains[target])
        def modulate():
            if summed:w[:]=0
            for k,target in enumerate(MOD_POST):
                when=t-([0,1,2,0,1][k] if delayed else 0)
                emitted=PATTERN[when,MOD_PRE[k]] if when>=0 else 0
                w[target]+=(1 if summed else emitted)*gains[k]*factors[k]
        if summed or before:modulate()
        for edge in range(len(w)):
            when=t-([1,2,0,1][edge] if delayed else 0)
            emitted=PATTERN[when,PRE[edge]] if when>=0 else 0
            v[POST[edge]]+=emitted*w[edge]
        if not summed and not before:modulate()
        if feedback:
            for edge,target in zip([3,0,2],[0,1,0]):
                v[target]+=s[target]*(.02*w[edge]+.01*PRE[edge]+.015*POST[edge])
        v-=.5*s;events.append(s.copy());membranes.append(u.copy())
    events=np.asarray(events);logits=5*events.mean(axis=0)
    maximum=logits.max();loss=maximum+np.log(np.exp(logits-maximum).sum())-logits[0]
    return loss,events,v,w,(np.asarray(membranes),events)


def bundle_for(net,inp,groups,conn,mod,engine='cpu',ranks=None):
    chosen={conn.name:['w'],mod.name:['gain']}
    for obj in net.objects:
        if isinstance(obj,b.Synapses) and 'eta' in obj.equations:chosen[obj.name]=['eta']
    return lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine,mpi_ranks=ranks,
        detach_reset=False,trainable_synapse_parameters=chosen)


@pytest.mark.parametrize('before,summed,feedback,nested',[(True,False,False,False),(False,False,False,False),(True,False,True,False),(True,True,True,False),(True,False,True,True)])
def test_endpoint_physical_forward_cython_and_surrogate_vjp(engine,before,summed,feedback,nested):
    net,inp,groups,conn,mod,_=model(before,summed=summed,feedback=feedback,nested=nested)
    weights=np.asarray(conn.w[:]).copy();gains=np.asarray(mod.gain[:]).copy();v=np.asarray(groups[1].v[:]).copy()
    bundle=bundle_for(net,inp,groups,conn,mod,engine)
    original=copy.deepcopy(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    eta=np.asarray([.008,.006,.007]) if nested else None
    result=trainer.gradients(PATTERN[None],[0]);expected=oracle(weights,gains,v,before,summed=summed,feedback=feedback,nested=eta)
    tol=5e-5 if engine!='cpu' else 3e-11
    # The finite difference subtracts two rounded losses; its absolute floor
    # differs from the exact forward/Cython comparisons below.
    fd_tol=2e-4 if engine!='cpu' else 3e-8
    np.testing.assert_array_equal(np.asarray(result['spikes'])[0,:,2:],expected[1])
    np.testing.assert_allclose(result['loss'],expected[0],rtol=tol,atol=tol*.01)
    state=np.asarray(result['final_state'])[0]
    layout=bundle.provenance['dynamic_state_layout'][conn.name]
    assert 'w' in layout # foreign writes cannot be lowered as constant parameters
    np.testing.assert_allclose(state[layout['w']],expected[3],rtol=tol,atol=tol*.01)
    np.testing.assert_allclose(state[bundle.provenance['neuron_state_layout'][groups[1].name]['v']],expected[2],rtol=tol,atol=tol*.01)
    anchors=expected[-1];eps=1e-6
    def loss(w,g,z):return oracle(w,g,z,before,summed=summed,feedback=feedback,nested=eta,anchors=anchors)[0]
    def finite_difference(values,which):
        out=[]
        for k in range(len(values)):
            plus=values.copy();minus=values.copy();plus[k]+=eps;minus[k]-=eps
            args=[weights,gains,v];args[which]=plus;a=loss(*args);args[which]=minus;c=loss(*args)
            out.append((a-c)/(2*eps))
        return out
    for owner,values,which in [(conn,weights,0),(mod,gains,1)]:
        bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==owner.name and e['variables']==(['w'] if which==0 else ['gain']))
        np.testing.assert_allclose(result['gradients'][bank],finite_difference(values,which),rtol=fd_tol,atol=fd_tol)
        if which==1:assert np.max(np.abs(result['gradients'][bank]))>1e-5
    slots=bundle.provenance['neuron_state_layout'][groups[1].name]['v']
    np.testing.assert_allclose(np.asarray(result['initial_gradients'])[0,slots],finite_difference(v,2),rtol=fd_tol,atol=fd_tol)
    np.testing.assert_allclose(np.asarray(result['initial_state_gradients'])[0,layout['w']],finite_difference(weights,0),rtol=fd_tol,atol=fd_tol)
    if nested:
        gain_slots=bundle.provenance['dynamic_state_layout'][mod.name]['gain']
        np.testing.assert_allclose(np.asarray(result['initial_state_gradients'])[0,gain_slots],finite_difference(gains,1),rtol=fd_tol,atol=fd_tol)
        bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['variables']==['eta'])
        differences=[]
        for k in range(len(eta)):
            plus=eta.copy();minus=eta.copy();plus[k]+=eps;minus[k]-=eps
            a=oracle(weights,gains,v,before,feedback=feedback,nested=plus,anchors=anchors)[0]
            c=oracle(weights,gains,v,before,feedback=feedback,nested=minus,anchors=anchors)[0]
            differences.append((a-c)/(2*eps))
        np.testing.assert_allclose(result['gradients'][bank],differences,rtol=fd_tol,atol=fd_tol)
    assert trainer.state['weights']==original
    assert (result['gpu_dispatches']>0)==(engine!='cpu')
    np.testing.assert_array_equal(conn.w[:],weights);np.testing.assert_array_equal(mod.gain[:],gains)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    net.run(len(PATTERN)*.2*b.ms,namespace={})
    assert conn.pre.codeobj.compiled_code['run'] is not None
    np.testing.assert_allclose(conn.w[:],expected[3],rtol=3e-12,atol=3e-14)
    np.testing.assert_allclose(groups[1].v[:],expected[2],rtol=3e-12,atol=3e-14)
    if nested:
        np.testing.assert_allclose(mod.gain[:],state[gain_slots],rtol=tol,atol=tol*.01)
    events=np.zeros_like(expected[1]);m=monitors[1]
    events[np.rint(np.asarray(m.t/b.second)/.0002).astype(int),np.asarray(m.i)]=1
    np.testing.assert_array_equal(events,expected[1])


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('summed',[False,True])
def test_endpoint_carry_checkpoint_and_readonly(engine,ranks,summed,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    net,inp,groups,conn,mod,_=model(summed=summed,feedback=True)
    bundle=bundle_for(net,inp,groups,conn,mod,engine,ranks)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    whole=trainer.evaluate(PATTERN[None],[0]);trainer.execute(PATTERN[None,:3],[0]);path=tmp_path/'endpoint.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,request_timeout=300);restored.restore(path)
    before=path.read_bytes();first=restored.gradients(PATTERN[None,3:],[0],initial='carry')
    restored.store(path);assert path.read_bytes()==before
    second=restored.gradients(PATTERN[None,3:],[0],initial='carry');assert first==second
    tail=restored.execute(PATTERN[None,3:],[0],initial='carry');tol=5e-5 if engine!='cpu' else 3e-12
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(np.asarray(tail['spikes']),np.asarray(whole['spikes'])[:,3:])
    assert tail['final_tick']==len(PATTERN)


@pytest.mark.parametrize('before',[False,True])
@pytest.mark.parametrize('warm',[0,1])
def test_endpoint_delayed_weight_reads_and_warm_queue(engine,before,warm,tmp_path):
    net,inp,groups,conn,mod,_=model(before,feedback=True,delayed=True)
    weights=np.asarray(conn.w[:]).copy();gains=np.asarray(mod.gain[:]).copy();v=np.asarray(groups[1].v[:]).copy()
    expected=oracle(weights,gains,v,before,feedback=True,delayed=True)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    if warm:net.run(warm*.2*b.ms,namespace={})
    bundle=bundle_for(net,inp,groups,conn,mod,engine);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    x=PATTERN[warm:];trainer.execute(x[None,:2],[0]);path=tmp_path/'pending-endpoint.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,request_timeout=300);restored.restore(path)
    actual=restored.execute(x[None,2:],[0],initial='carry');tol=5e-5 if engine!='cpu' else 3e-12
    final=np.asarray(actual['final_state'])[0]
    np.testing.assert_allclose(final[bundle.provenance['dynamic_state_layout'][conn.name]['w']],expected[3],rtol=tol,atol=tol*.01)
    np.testing.assert_allclose(final[bundle.provenance['neuron_state_layout'][groups[1].name]['v']],expected[2],rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(np.asarray(actual['spikes'])[0,:,2:],expected[1][warm+2:])
    net.run((len(PATTERN)-warm)*.2*b.ms,namespace={})
    np.testing.assert_allclose(conn.w[:],expected[3],rtol=3e-12,atol=3e-14)
    np.testing.assert_allclose(groups[1].v[:],expected[2],rtol=3e-12,atol=3e-14)
    assert conn.pre.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('constant',[False,True])
@pytest.mark.parametrize('trainable',[False,True])
def test_endpoint_parameter_gathers_snapshot_old_selector(engine,constant,trainable):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='gather_endpoint_input')
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',dt=dt,method='euler',
                         name=f'gather_endpoint_{k}') for k in range(2)]
    groups[1].v=[.72,.87]
    conn=b.Synapses(inp,groups[1],'w:1'+(' (constant)' if constant else ''),on_pre='v_post+=w',dt=dt,name='gather_endpoint_conn')
    conn.connect(j='i');conn.w=[.3,.45]
    reader=b.Synapses(conn,groups[1],'pick:integer',on_post='pick=1-pick\nv_post+=.4*chosen_w',dt=dt,name='gather_endpoint_reader')
    reader.connect(j='i');reader.pick=[0,1]
    reader.variables.add_reference('chosen_w',conn,'w',index='pick')
    net=b.Network(inp,*groups,conn,reader)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine,
        detach_reset=False,trainable_synapse_parameters={conn.name:['w'] if trainable else []})
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    inputs=np.zeros((1,6,2));result=trainer.gradients(inputs,[0]);tol=5e-5 if engine!='cpu' else 3e-12
    def physical(weights,anchors=None):
        v=np.array([.72,.87]);pick=np.array([0,1]);events=[];membranes=[]
        for t in range(6):
            u=.8*v;hard=(u>.5).astype(float)
            if anchors is None:s=hard
            else:
                base_u,base_s=anchors;s=base_s[t]+(u-base_u[t])/(1+5*np.abs(base_u[t]-.5))**2
            # Cython reads chosen_w using the old physical selector before
            # executing pick's assignment, then writes both logical locals.
            v=u+s*(.4*weights[pick]-.5);pick=np.where(hard,1-pick,pick)
            events.append(s.copy());membranes.append(u.copy())
        events=np.asarray(events);logits=5*events.mean(axis=0);maximum=logits.max()
        loss=maximum+np.log(np.exp(logits-maximum).sum())-logits[0]
        return loss,v,pick,events,(np.asarray(membranes),events)
    weights=np.array([.3,.45]);expected=physical(weights)
    state=np.asarray(result['final_state'])[0];vslots=bundle.provenance['neuron_state_layout'][groups[1].name]['v']
    pslots=bundle.provenance['dynamic_state_layout'][reader.name]['pick']
    np.testing.assert_allclose(state[vslots],expected[1],rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(state[pslots],expected[2]);np.testing.assert_array_equal(np.asarray(result['spikes'])[0,:,2:],expected[3])
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==conn.name and e['variables']==['w'])
    assert bundle.plan['trainable'][bank]==trainable
    fd=[]
    for k in range(2):
        plus=weights.copy();minus=weights.copy();plus[k]+=1e-6;minus[k]-=1e-6
        fd.append((physical(plus,expected[4])[0]-physical(minus,expected[4])[0])/2e-6)
    np.testing.assert_allclose(result['gradients'][bank],fd,rtol=2e-4 if engine!='cpu' else 3e-8,atol=2e-4 if engine!='cpu' else 3e-8)
    assert np.max(np.abs(fd))>1e-5
    assert (result['gpu_dispatches']>0)==(engine!='cpu')
    net.run(1.2*b.ms,namespace={});assert reader.post.codeobj.compiled_code['run'] is not None
    np.testing.assert_allclose(groups[1].v[:],expected[1],rtol=3e-12,atol=3e-14);np.testing.assert_array_equal(reader.pick[:],expected[2])
    np.testing.assert_array_equal(conn.w[:],weights)


@pytest.mark.parametrize('dtype',['integer','boolean'])
@pytest.mark.parametrize('shared',[False,True])
def test_endpoint_parameter_gathers_preserve_detached_control(engine,dtype,shared):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='control_endpoint_input')
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',dt=dt,method='euler',
                         name=f'control_endpoint_{k}') for k in range(2)]
    groups[1].v=[.72,.87]
    # 2**24+1 loses its low bit if a GPU host incorrectly uploads it as f32.
    values=([16777217] if shared else [16777217,16777218]) if dtype=='integer' else ([True] if shared else [True,False])
    expression='chosen % 2' if dtype=='integer' else 'int(chosen)'
    conn=b.Synapses(inp,groups[1],'w:'+dtype+(' (shared)' if shared else ''),dt=dt,name='control_endpoint_conn');conn.connect(j='i');conn.w=values[0] if shared else values
    reader=b.Synapses(conn,groups[1],'pick:integer',on_post='pick=1-pick\nv_post+=.04*('+expression+')',dt=dt,name='control_endpoint_reader')
    reader.connect(j='i');reader.pick=[0,1];reader.variables.add_reference('chosen',conn,'w',index='0' if shared else 'pick')
    net=b.Network(inp,*groups,conn,reader)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine,
        trainable_synapse_parameters={conn.name:[]},detach_reset=False)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300).gradients(np.zeros((1,6,2)),[0])
    v=np.array([.72,.87]);pick=np.array([0,1]);events=[]
    for _ in range(6):
        u=.8*v;s=u>.5;selected=np.full(2,values[0]) if shared else np.asarray(values)[pick]
        v=u+s*(.04*(selected%2 if dtype=='integer' else selected.astype(int))-.5);pick=np.where(s,1-pick,pick);events.append(s)
    state=np.asarray(result['final_state'])[0];vslots=bundle.provenance['neuron_state_layout'][groups[1].name]['v']
    pslots=bundle.provenance['dynamic_state_layout'][reader.name]['pick'];tol=5e-5 if engine!='cpu' else 3e-12
    np.testing.assert_allclose(state[vslots],v,rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(state[pslots],pick);np.testing.assert_array_equal(np.asarray(result['spikes'])[0,:,2:],events)
    if not shared:
        bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==conn.name and e['variables']==['w'])
        assert result['gradients'][bank]==[0.,0.];assert not bundle.plan['trainable'][bank]
    assert (result['gpu_dispatches']>0)==(engine!='cpu')
    net.run(1.2*b.ms,namespace={});assert reader.post.codeobj.compiled_code['run'] is not None
    np.testing.assert_allclose(groups[1].v[:],v,rtol=3e-12,atol=3e-14);np.testing.assert_array_equal(reader.pick[:],pick)
