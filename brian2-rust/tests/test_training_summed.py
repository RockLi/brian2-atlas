"""Summed synaptic states and mutable neuron parameters in native BPTT."""
import copy
import os
import tempfile
import ast
from unittest.mock import patch
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training,TrainingConversionError
from test_native_training import RUNNER


@pytest.fixture(scope='module',autouse=True)
def cython_cache():
    old=b.prefs.codegen.runtime.cython.cache_dir
    with tempfile.TemporaryDirectory(prefix='b2-summed-cython-') as directory:
        b.prefs.codegen.runtime.cython.cache_dir=directory
        try:yield
        finally:b.prefs.codegen.runtime.cython.cache_dir=old


def model(method='euler',refractory=False,both=True,empty=False,noise=False,drop=None):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[1,1],[0,1],[1,0],[0,0],[1,1],[1,0],[1,1]],float)
    ticks,ids=np.nonzero(x);inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='sum_input')
    groups=[]
    for layer in range(2):
        noise_term='+.07*(g+1)*xi/sqrt(ms)' if noise else ''
        eq='''dv/dt=(-v+g+drive)/tau%s:1%s
        g:1
        drive:1 (shared,constant)
        tau:second (shared,constant)
        theta:1 (shared,constant)'''%(noise_term,' (unless refractory)' if refractory else '')
        group=b.NeuronGroup(2,eq,threshold='v>theta',reset='v-=theta',method=method,dt=dt,
                            refractory=3*dt if refractory else False,name=f'sum_layer_{layer}')
        group.v=[.2+layer*.3,1.2+layer*.4];group.g=[11+layer*6,13+layer*6];group.drive=.6;group.tau=(1+.1*layer)*b.ms;group.theta=1
        groups.append(group)
    static=b.Synapses(inp,groups[0],'w:1',on_pre='v_post+=w',name='sum_static');static.connect();static.w=[.8,.9,.7,1.]
    pre='\ng_pre=.2*s*(v_post-v_pre):1 (summed)' if both else ''
    noise_term='+.04*xi/sqrt(ms)' if noise else ''
    syn=b.Synapses(*groups,'''w:1
        ds/dt=-s/(2*ms)%s:1 (clock-driven)
        g_post=s*(1.5-v_post):1 (summed)'''%noise_term+pre,
        on_pre='s+=w\nw=clip(w+.015*v_post,0,2)',on_post='w=clip(w-.01*v_pre,0,2)',method='euler',dt=dt,name='sum_plastic')
    if empty:syn.connect(condition='False')
    else:
        keep=[e for e in range(4) if e!=drop]
        syn.connect(i=np.array([1,0,1,0])[keep],j=np.array([0,1,1,0])[keep]);syn.w=np.array([.4,.3,.2,.5])[keep];syn.s=np.array([.2,.3,.1,.4])[keep]
    return b.Network(inp,*groups,static,syn),inp,groups,static,syn,x,dt


def lower(net,inp,groups,**kwargs):
    return lower_brian_dynamic_training(net,input_group=inp,layers=groups,
        trainable_neuron_parameters={g.name:['drive','tau','theta'] for g in groups},learning_rate=1e-7,**kwargs)


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('refractory',[False,True])
@pytest.mark.parametrize('both,empty',[(False,False),(True,False),(True,True)])
def test_summed_forward_matches_brian(method,refractory,both,empty):
    net,inp,groups,static,syn,x,dt=model(method,refractory,both,empty);bundle=lower(net,inp,groups)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*dt,namespace={})
    spikes=np.zeros((len(x),4));offset=0
    for layer,(g,m) in enumerate(zip(groups,monitors)):
        spikes[np.rint(np.asarray(m.t/b.second)/float(dt)).astype(int),layer*2+np.asarray(m.i)]=1
        for k,name in enumerate(bundle.provenance['state_names'][layer]):
            if name=='__refractory_ticks':continue
            np.testing.assert_allclose(actual['final_state'][0][offset+2*k:offset+2*k+2],g.variables[name].get_value(),rtol=5e-13,atol=3e-14)
        if both or layer==1:assert actual['initial_state_gradients'][0][offset+2:offset+4]==[0.,0.]
        offset+=2*len(bundle.provenance['state_names'][layer])
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(np.array(actual['final_state'])[0,indices],syn.variables[name].get_value(),rtol=5e-13,atol=3e-14)
    assert len(bundle.provenance['summed_updates'])==(2 if both else 1)


@pytest.mark.parametrize('direction',['pre','post'])
def test_order_dependent_summed_uses_cython_clear_then_accumulate(direction):
    net,inp,groups,static,old,x,dt=model();net.remove(old)
    target=groups[0 if direction=='pre' else 1];alias='post' if direction=='pre' else 'pre'
    syn=b.Synapses(target,target,f'w:1\ng_{direction}=w+.2*g_{alias}:1 (summed)',name='recursive_sum')
    syn.connect(i=[1,0,1,0],j=[1,0,1,0]);syn.w=[.1,.2,.3,.4];net.add(syn)
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    for updater in syn.summed_updaters.values():updater.codeobj_class=CythonCodeObject
    bundle=lower(net,inp,groups);actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    net.run(len(x)*dt,namespace={})
    np.testing.assert_allclose(actual['final_membrane'][0],np.r_[groups[0].v[:],groups[1].v[:]],rtol=1e-13,atol=1e-14)
    layer=0 if direction=='pre' else 1
    np.testing.assert_allclose(actual['final_state'][0][4*layer+2:4*layer+4],groups[layer].g[:],rtol=1e-13,atol=1e-14)


def test_multiple_summed_writers_rejected():
    net,inp,groups,static,syn,x,dt=model()
    other=b.Synapses(*groups,'w:1\ng_post=w:1 (summed)');other.connect();net.add(other)
    with pytest.raises(TrainingConversionError,match='summed'):lower(net,inp,groups)


@pytest.mark.parametrize('method',['euler','heun'])
@pytest.mark.parametrize('refractory',[False,True])
@pytest.mark.parametrize('delays',[False,True])
def test_stochastic_neuron_and_edge_summed_matches_brian(method,refractory,delays):
    from test_training_stochastic import normal
    net,inp,groups,static,syn,x,dt=model(method,refractory,noise=True)
    if delays:
        static.pre.delay=[0,.2,.4,.6]*b.ms;syn.pre.delay=[.6,.2,.4,0]*b.ms;syn.post.delay=[.4,0,.6,.2]*b.ms
    bundle=lower(net,inp,groups,seed=7123)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0],noise_sequence=11)
    net.run(0*dt,namespace={});order=[]
    for obj in net.sorted_objects:
        if obj not in [g.state_updater for g in groups]+[syn.state_updater]:continue
        domain=groups.index(obj.group) if obj.group in groups else bundle.provenance['synaptic_noise_domains'][syn.name]
        for statement in ast.parse(obj.abstract_code).body:
            if isinstance(statement,ast.Assign) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='randn' for n in ast.walk(statement.value)):
                order.append((domain,len(obj.group)))
    draws=iter([np.array([normal(7123,11,0,domain,j,t,0) for j in range(count)]) for t in range(len(x)) for domain,count in order])
    def randn(count):
        values=next(draws);assert len(values)==count;return values
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    with patch('numpy.random.randn',randn):net.run(len(x)*dt,namespace={})
    with pytest.raises(StopIteration):next(draws)
    np.testing.assert_allclose(result['final_membrane'][0],np.r_[groups[0].v[:],groups[1].v[:]],rtol=1e-12,atol=3e-14)
    expected=np.zeros((len(x),4))
    for layer,m in enumerate(monitors):expected[np.rint(np.asarray(m.t/b.second)/float(dt)).astype(int),2*layer+np.asarray(m.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],expected)
    for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(np.array(result['final_state'])[0,indices],syn.variables[name].get_value(),rtol=1e-12,atol=3e-14)


def oracle(bundle,weights,x,initial=None,anchors=None):
    """Separate continuous summed-current and whole-event surrogate reference."""
    p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for index,binding in enumerate(p['dynamic']['initial_parameters']):
            if binding is not None:z[index]=weights[binding[0]][binding[1]]
    layout=bundle.provenance['dynamic_state_layout']['sum_plastic'];si=layout['s'];wi=layout['w'];vi=p['dynamic']['voltage']
    ref=bool(p.get('refractory'));stride=6 if ref else 4;gi=[2,3,stride+2,stride+3];ci=[4,5,10,11] if ref else [];ai=[12,13,14,15] if ref else []
    parameters=[]
    for name in bundle.provenance['layer_names']:
        binding=next(v for v in bundle.provenance['bindings'] if v['object']==name)
        parameters.append(dict(zip(binding['variables'],weights[binding['bank']])))
    theta=np.repeat([v['theta'] for v in parameters],2)
    static=weights[next(v['bank'] for v in bundle.provenance['bindings'] if v['object']=='sum_static')]
    edges=[(1,0),(0,1),(1,1),(0,0)];dt=bundle.provenance['dt_seconds'];starts=[];pres=[];spikes=[];activities=[]
    for tick,inputs in enumerate(x):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors[2][tick].copy()
        starts.append(z.copy());z[gi[2:]]=0
        for e,(i,j) in enumerate(edges):z[gi[2+j]]+=z[si[e]]*(1.5-z[vi[2+j]])
        if any(u['direction']=='pre' for u in bundle.provenance['summed_updates']):
            z[gi[:2]]=0
            for e,(i,j) in enumerate(edges):z[gi[i]]+=.2*z[si[e]]*(z[vi[2+j]]-z[vi[i]])
        active=z[ci]==0 if ref else np.ones(4,bool)
        if anchors is not None:active=anchors[3][tick]
        activities.append(active.copy())
        for layer in range(2):
            ids=vi[2*layer:2*layer+2];v=z[ids].copy();g=z[gi[2*layer:2*layer+2]]
            def f(v):return (-v+g+parameters[layer]['drive'])/parameters[layer]['tau']*active[2*layer:2*layer+2]
            k1=f(v);method=bundle.provenance['integrators'][layer]
            if method=='euler':new=v+dt*k1
            elif method=='rk2':new=v+dt*f(v+dt*k1/2)
            else:
                k2=f(v+dt*k1/2);k3=f(v+dt*k2/2);k4=f(v+dt*k3);new=v+dt*(k1+2*k2+2*k3+k4)/6
            z[ids]=new
        if ref:z[ci]=np.maximum(z[ci]-1,0)
        z[si]*=.9;pre=z[vi].copy();hard=((pre>theta)&active).astype(float);s=hard.copy()
        # The threshold frontend now stores v-theta in explicit scratch cells.
        # Keep the independent physical-state reference complete without
        # interpreting native SSA or changing its spike/gradient equations.
        for layer,name in enumerate(bundle.provenance['layer_names']):
            margins=bundle.provenance.get('threshold_margin_layout',{}).get(name,[])
            if margins:z[margins]=pre[2*layer:2*layer+2]-theta[2*layer:2*layer+2]
        if anchors is not None:
            hard=anchors[1][tick];phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][tick]-anchors[4]))**2
            s=hard+active*phi*(pre-theta-anchors[0][tick]+anchors[4])
        if ref:z[ai]=active&(hard==0)
        for e in [1,3,0,2]:
            i,j=edges[e];ids=[si[e],wi[e]];old=z[ids].copy();new=np.array([old[0]+old[1],np.clip(old[1]+.015*z[vi[2+j]],0,2)])
            z[ids]=old+s[i]*(new-old)
        for i in range(2):
            for j in range(2):
                if not ref or active[j] and hard[j]==0:z[vi[j]]+=inputs[i]*static[2*i+j]
        for e in [0,3,1,2]:
            i,j=edges[e];old=z[wi[e]];z[wi[e]]=old+s[2+j]*(np.clip(old-.01*z[vi[i]],0,2)-old)
        z[vi]-=theta*(hard if p['detach_reset'] else s)
        if ref:z[ci]=np.where(hard!=0,2,z[ci])
        pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes,(np.array(pres),spikes,np.array(starts),np.array(activities),theta)


@pytest.mark.parametrize('method',['euler','rk4'])
@pytest.mark.parametrize('refractory',[False,True])
@pytest.mark.parametrize('both',[False,True])
@pytest.mark.parametrize('detach,window',[(True,None),(False,3)])
def test_summed_independent_parameter_and_initial_vjp(method,refractory,both,detach,window):
    net,inp,groups,static,syn,x,dt=model(method,refractory,both);bundle=lower(net,inp,groups,detach_reset=detach,tbptt_window=window)
    weights=bundle.weights;trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=weights);actual=trainer.gradients(x[None],[0])
    loss,live,spikes,anchors=oracle(bundle,weights,x)
    np.testing.assert_array_equal(actual['spikes'][0],spikes);np.testing.assert_allclose(actual['final_state'][0],live,rtol=2e-13,atol=2e-14)
    assert actual['loss']==pytest.approx(loss,abs=2e-14)
    for bank,row in enumerate(weights):
        for j,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);a=copy.deepcopy(weights);c=copy.deepcopy(weights);a[bank][j]+=eps;c[bank][j]-=eps
            fd=(oracle(bundle,a,x,anchors=anchors)[0]-oracle(bundle,c,x,anchors=anchors)[0])/(2*eps)
            assert actual['gradients'][bank][j]==pytest.approx(fd,rel=3e-4,abs=2e-6)
    initial=np.array(bundle.initial_state);actual=trainer.gradients(x[None],[0],initial=initial[None]);_,_,_,anchors=oracle(bundle,weights,x,initial)
    for j in range(len(initial)):
        if bundle.plan['dynamic']['detached'][j]:
            assert actual['initial_state_gradients'][0][j]==0;continue
        a=initial.copy();c=initial.copy();a[j]+=1e-6;c[j]-=1e-6
        fd=(oracle(bundle,weights,x,a,anchors)[0]-oracle(bundle,weights,x,c,anchors)[0])/2e-6
        assert actual['initial_state_gradients'][0][j]==pytest.approx(fd,rel=3e-4,abs=2e-7)


@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('noise',[False,True])
def test_summed_mpi_train_and_restore(ranks,noise,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    net,inp,groups,static,syn,x,dt=model('heun' if noise else 'rk4',True,noise=noise)
    bundle=lower(net,inp,groups,detach_reset=False,tbptt_window=3,seed=7123)
    serial=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);mp=copy.deepcopy(bundle.plan);mp['mpi_ranks']=ranks
    parallel=NativeLIFTrainer(mp,runner=RUNNER,weights=bundle.weights)
    for start,stop in [(0,3),(3,7),(7,len(x))]:
        kw={} if start==0 else {'initial':'carry'}
        a=serial.execute(x[None,start:stop],[0],**kw);c=parallel.execute(x[None,start:stop],[0],**kw)
        for key in ['spikes','final_state','initial_state_gradients']:
            np.testing.assert_allclose(a[key],c[key],rtol=2e-13,atol=1e-13)
        for row,other in zip(a['gradients'],c['gradients']):np.testing.assert_allclose(row,other,rtol=2e-13,atol=2e-12)
        path=tmp_path/'sum.json';parallel.store(path);restored=NativeLIFTrainer(mp,runner=RUNNER,weights=bundle.weights);restored.restore(path);parallel=restored
        assert parallel.clock_tick==serial.clock_tick and parallel.noise_sequence==serial.noise_sequence


@pytest.mark.parametrize('noise',[False,True])
def test_summed_frozen_split_replays_exactly(noise):
    net,inp,groups,static,syn,x,dt=model('heun' if noise else 'rk4',True,noise=noise);bundle=lower(net,inp,groups,seed=7123)
    bundle.plan['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    kw={'noise_sequence':7} if noise else {}
    whole=trainer.evaluate(x[None],[0],**kw);trainer.execute(x[None,:5],[0],**kw)
    tail=trainer.execute(x[None,5:],[0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=1e-14,atol=1e-14)
    np.testing.assert_array_equal(tail['spikes'],np.asarray(whole['spikes'])[:,5:])
    assert trainer.next_noise_sequence==(8 if noise else 0)


def test_summed_schedule_after_neuron_updates_preserves_old_current_gradient():
    net,inp,groups,static,syn,x,dt=model()
    for updater in syn.summed_updaters.values():updater.order=1
    bundle=lower(net,inp,groups);actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    assert np.any(np.asarray(actual['initial_state_gradients'])[0,[2,3,6,7]]!=0)
    net.run(len(x)*dt,namespace={})
    np.testing.assert_allclose(actual['final_membrane'][0],np.r_[groups[0].v[:],groups[1].v[:]],rtol=1e-12,atol=1e-13)


def test_mutable_neuron_parameter_event_and_reset_state():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(2,[0,1,0,1],np.arange(4)*dt,dt=dt)
    groups=[b.NeuronGroup(2,'dv/dt=(-v+a)/ms:1\na:1',threshold='v>1',reset='a+=.07*v\nv-=1',method='rk4',dt=dt) for _ in range(2)]
    for g in groups:g.v=[.3,1.4];g.a=[.2,.5]
    synapses=[]
    for source,target in [(inp,groups[0]),(groups[0],groups[1])]:
        syn=b.Synapses(source,target,'w:1',on_pre='a_post+=w\nv_post+=.1*a_post');syn.connect();syn.w=[.4,.5,.6,.7];synapses.append(syn)
    net=b.Network(inp,*groups,*synapses);bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups)
    x=np.zeros((1,12,2));x[0,np.arange(4),[0,1,0,1]]=1
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x,[0])
    # Both pathways have cross-edge read-after-write effects; use the canonical
    # sequential Brian reference as in the autapse/refractory tests.
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    for syn in synapses:syn.pre.codeobj_class=CythonCodeObject
    net.run(12*dt,namespace={})
    np.testing.assert_allclose(actual['final_state'][0],np.r_[groups[0].v[:],groups[0].a[:],groups[1].v[:],groups[1].a[:]],rtol=1e-12,atol=1e-13)


def test_masked_edge_cannot_contribute_old_trace_to_summed_current():
    net,inp,groups,static,syn,x,dt=model();bundle=lower(net,inp,groups)
    bank=next(v['bank'] for v in bundle.provenance['bindings'] if v['object']==syn.name)
    bundle.plan['masks'][bank][1]=0;bundle.weights[bank][1]=0
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    indices=bundle.provenance['dynamic_state_layout'][syn.name]['s']
    assert actual['final_state'][0][indices[1]]==bundle.initial_state[indices[1]]
    assert actual['initial_state_gradients'][0][indices[1]]==0
    reference,_,ref_groups,_,ref_syn,_,_=model(drop=1);reference.run(len(x)*dt,namespace={})
    np.testing.assert_allclose(actual['final_membrane'][0],np.r_[ref_groups[0].v[:],ref_groups[1].v[:]],rtol=1e-12,atol=1e-13)
    for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(np.array(actual['final_state'])[0,np.array(indices)[[0,2,3]]],ref_syn.variables[name].get_value(),rtol=1e-12,atol=1e-13)


@pytest.mark.parametrize('issue',['units','inactive','shared_target'])
def test_summed_target_validation(issue):
    net,inp,groups,static,syn,x,dt=model()
    if issue=='units':
        from brian2.equations.equations import Expression
        next(iter(syn.summed_updaters.values())).expression=Expression('w*ms')
    elif issue=='inactive':next(iter(syn.summed_updaters.values())).active=False
    else:groups[0].variables['g'].scalar=True
    with pytest.raises(TrainingConversionError):lower(net,inp,groups)


@pytest.mark.parametrize('clock_dt',[.1,.3])
def test_summed_and_event_dt_use_synaptic_clock_value(clock_dt):
    net,inp,groups,static,old,x,dt=model();net.remove(old)
    syn=b.Synapses(*groups,'w:1\ng_post=w*dt/ms:1 (summed)',on_pre='v_post+=w*dt/ms',dt=clock_dt*b.ms)
    syn.connect();syn.w=[.1,.2,.3,.4];net.add(syn)
    bundle=lower(net,inp,groups);actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    net.run(len(x)*dt,namespace={})
    np.testing.assert_allclose(actual['final_membrane'][0],np.r_[groups[0].v[:],groups[1].v[:]],rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(actual['final_state'][0][6:8],groups[1].g[:],rtol=1e-13,atol=1e-14)


def test_asynchronous_synaptic_time_matches_brian():
    net,inp,groups,static,old,x,dt=model();net.remove(old)
    syn=b.Synapses(*groups,'w:1\ng_post=w*t/ms:1 (summed)',dt=.3*b.ms);syn.connect();syn.w=.1;net.add(syn)
    bundle=lower(net,inp,groups)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    net.run(len(x)*dt,namespace={})
    np.testing.assert_allclose(actual['final_membrane'][0],np.r_[groups[0].v[:],groups[1].v[:]],rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(actual['final_state'][0][6:8],groups[1].g[:],rtol=1e-13,atol=1e-14)
