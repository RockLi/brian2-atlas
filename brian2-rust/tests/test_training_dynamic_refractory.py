"""Ordered dynamic events with refractory guards and detached discrete state."""
import copy
import os
import tempfile

import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
import test_training_refractory as ref


@pytest.fixture(scope='module',autouse=True)
def isolated_cython_cache():
    old=b.prefs.codegen.runtime.cython.cache_dir
    with tempfile.TemporaryDirectory(prefix='b2-dynamic-ref-cython-') as directory:
        b.prefs.codegen.runtime.cython.cache_dir=directory
        try:yield
        finally:b.prefs.codegen.runtime.cython.cache_dir=old


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('clamp',[(),('v',),('a',),('v','a')])
@pytest.mark.parametrize('detach,window',[(True,None),(False,3)])
def test_dynamic_refractory_matches_validated_static_vjp(method,clamp,detach,window):
    net,inp,groups,x,dt=ref.model(clamp,method=method)
    old=ref.lower(net,inp,groups,detach_reset=detach,tbptt_window=window)
    new=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=detach,tbptt_window=window,
        trainable_neuron_parameters={g.name:['tau','theta','kick'] for g in groups})
    expected=NativeLIFTrainer(old.plan,runner=RUNNER,weights=old.weights).gradients(x[None],[0],initial=[old.initial_state])
    actual=NativeLIFTrainer(new.plan,runner=RUNNER,weights=new.weights).gradients(x[None],[0])
    np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
    np.testing.assert_allclose(actual['final_state'][0][:12],expected['final_state'][0],rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(actual['initial_state_gradients'][0][:12],expected['initial_state_gradients'][0],rtol=2e-12,atol=2e-13)
    # Guards and the threshold margins are recomputed before their first read.
    np.testing.assert_array_equal(actual['initial_state_gradients'][0][12:],
                                  np.zeros(len(new.initial_state)-12))
    for binding in new.provenance['bindings']:
        other=next(v for v in old.provenance['bindings'] if v['object']==binding['object'])
        np.testing.assert_allclose(actual['gradients'][binding['bank']],expected['gradients'][other['bank']],rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('method,shared',[('euler',False),('euler',True),('heun',False),('heun',True),('heun','mixed'),('milstein',False)])
@pytest.mark.parametrize('detach,window',[(True,None),(False,3)])
def test_stochastic_refractory_dynamic_path_matches_v4(method,shared,detach,window):
    import test_training_stochastic as sde
    net,inp,groups,x,dt=sde.model(method,shared,refractory=True)
    old=sde.lower(net,inp,groups,detach_reset=detach,tbptt_window=window)
    new=lower_brian_dynamic_training(net,input_group=inp,layers=groups,seed=7123,detach_reset=detach,tbptt_window=window,
        trainable_neuron_parameters={g.name:['tau','theta','kick','gain','drive','sigma'] for g in groups})
    kw=dict(noise_sequence=17,start_tick=2)
    expected=NativeLIFTrainer(old.plan,runner=RUNNER,weights=old.weights).gradients(x[None],[0],initial=[old.initial_state],**kw)
    actual=NativeLIFTrainer(new.plan,runner=RUNNER,weights=new.weights).gradients(x[None],[0],**kw)
    np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
    for key in ['final_state','initial_state_gradients']:
        np.testing.assert_allclose(actual[key][0][:12],expected[key][0],rtol=5e-12,atol=5e-13)
    for binding in new.provenance['bindings']:
        other=next(v for v in old.provenance['bindings'] if v['object']==binding['object'])
        np.testing.assert_allclose(actual['gradients'][binding['bank']],expected['gradients'][other['bank']],rtol=5e-12,atol=5e-12)


def plastic_model(clamp=('v',),duration=3.,method='euler',warmup=0,sequential_reference=False):
    net,inp,groups,x,dt=ref.model(clamp,duration,method=method)
    original=next(o for o in net.objects if isinstance(o,b.Synapses) and o.name=='ref_syn_1');net.remove(original)
    plastic=b.Synapses(*groups,'dw/dt=-.01*w/ms:1 (clock-driven)',method='euler',dt=dt,
        on_pre='v_post+=w\na_post+=.05\nw=clip(w+.03*a_post,0,2)',on_post='w=clip(w-.02*v_pre,0,2)',name='ref_syn_1')
    plastic.connect(i=[1,0,1,0],j=[0,0,1,1]);plastic.w=[1.2,.4,.3,1.1];net.add(plastic)
    if sequential_reference:
        # Brian warns this cross-edge read-after-write model is order-dependent.
        # Cython executes whole paths per edge, matching the declared v5 order;
        # NumPy's add.at executes each statement over all edges instead.
        from brian2.codegen.runtime.cython_rt import CythonCodeObject
        for path in plastic._pathways:path.codeobj_class=CythonCodeObject
    if warmup:net.run(warmup*dt,namespace={})
    return net,inp,groups,x[warmup:],dt,plastic


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('clamp',[(),('v',),('a',),('v','a')])
@pytest.mark.parametrize('duration,warmup',[(0.,0),(1.,0),(3.,0),(3.5,3)])
def test_plasticity_continues_while_neuron_writes_are_clamped(method,clamp,duration,warmup):
    net,inp,groups,x,dt,plastic=plastic_model(clamp,duration,method,warmup,sequential_reference=True)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*dt,namespace={})
    spikes=np.zeros((len(x),4))
    for layer,m in enumerate(monitors):spikes[np.rint(np.asarray(m.t/b.second)/float(dt)).astype(int)-warmup,2*layer+np.asarray(m.i)]=1
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    for layer,g in enumerate(groups):
        np.testing.assert_allclose(actual['final_state'][0][6*layer:6*layer+4],np.r_[g.v[:],g.a[:]],rtol=5e-13,atol=2e-14)
    indices=bundle.provenance['dynamic_state_layout'][plastic.name]['w']
    np.testing.assert_allclose(np.asarray(actual['final_state'])[0,indices],plastic.w[:],rtol=5e-13,atol=2e-14)


def test_guarded_expression_is_lazy_and_sequential():
    from test_training_dynamic import model
    from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
    p,w,x=model();spec=p['dynamic'];spec['initial'].extend([0.]);spec['initial_parameters'].append(None);spec['detached'].append(True)
    # Held v is zero: evaluating log(v) would fail. The second write reads
    # the held v through its alias, while the unguarded plastic state changes.
    spec['initial'][0]=0.
    transform=compile_dynamic_transform('v=log(v)\nw+=alias+.1',states={'v':0,'alias':0,'w':1,'gate':2},write_guards={0:2})
    ps=len(spec['program_sets']);spec['program_sets'].append(transform['programs'])
    spec['actions'].insert(0,dynamic_action(transform,[0,4,16],owner=0,program_set=ps))
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    out=trainer.gradients(np.zeros((1,1,2)),[0])
    assert np.isfinite(out['loss']) and out['initial_state_gradients'][0][16]==0
    # Making the branch live must reveal the domain error, transactionally.
    p['dynamic']['initial'][16]=1.
    bad=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    before=copy.deepcopy(bad.state)
    with pytest.raises(ValueError,match='domain'):bad.execute(np.zeros((1,1,2)),[0])
    assert bad.state==before and bad.neuron_state is None


@pytest.mark.parametrize('gate',[.5,-1.,2.])
def test_select_rejects_nonbinary_gates(gate):
    from test_training_dynamic import model
    from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
    p,w,x=model();spec=p['dynamic'];index=len(spec['initial']);spec['initial'].append(gate);spec['initial_parameters'].append(None);spec['detached'].append(True)
    transform=compile_dynamic_transform('v+=.1',states={'v':0,'gate':1},write_guards={0:1})
    ps=len(spec['program_sets']);spec['program_sets'].append(transform['programs']);spec['actions'].insert(0,dynamic_action(transform,[0,index],owner=0,program_set=ps))
    with pytest.raises(ValueError,match='binary'):NativeLIFTrainer(p,runner=RUNNER,weights=w).evaluate(x[None],[0])


def test_guard_binding_rejects_mutation_and_invalid_slots():
    from brian2_rust.training_dynamic import compile_dynamic_transform
    for guards in ({0:0},{0:2},{2:1}):
        with pytest.raises(ValueError,match='write guards'):compile_dynamic_transform('v+=1',states={'v':0,'g':1},write_guards=guards)
    with pytest.raises(ValueError,match='read-only'):compile_dynamic_transform('g=0\nv+=1',states={'v':0,'g':1},write_guards={0:1})


def plastic_oracle(bundle,weights,x,initial=None,anchors=None):
    """Independent state equations and locally linearized event gates."""
    p=bundle.plan;state=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for index,binding in enumerate(p['dynamic']['initial_parameters']):
            if binding is not None:state[index]=weights[binding[0]][binding[1]]
    neuron_banks=[next(v for v in bundle.provenance['bindings'] if v['object']==name and v['kind']=='neuron')
                  for name in bundle.provenance['layer_names']]
    params={name:np.array([weights[v['bank']][v['variables'].index(name)] for v in neuron_banks]) for name in ('kick','tau','theta')}
    theta=np.repeat(params['theta'],2);starts=[];pres=[];spikes=[];activities=[]
    fixed={name:weights[next(v['bank'] for v in bundle.provenance['bindings'] if v['object']==name)] for name in ('ref_syn_0','ref_syn_2')}
    edges=[(1,0),(0,0),(1,1),(0,1)];dt=bundle.provenance['dt_seconds']
    for tick,inputs in enumerate(x):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:state=anchors[2][tick].copy()
        starts.append(state.copy());active=state[[4,5,10,11]]==0
        if anchors is not None:active=anchors[4][tick]
        activities.append(active.copy())
        for layer,offset in enumerate([0,6]):
            z=state[offset:offset+4].reshape(2,2).copy()
            def f(z):
                dz=np.array([-z[0]+.3*z[1],.15*z[0]-.4*z[1]])/params['tau'][layer]
                for s in p['refractory'][layer]['clamp']:dz[s]*=active[2*layer:2*layer+2]
                return dz
            method=bundle.provenance['integrators'][layer];k1=f(z)
            if method=='euler':new=z+dt*k1
            elif method=='rk2':new=z+dt*f(z+dt*k1/2)
            else:
                k2=f(z+dt*k1/2);k3=f(z+dt*k2/2);k4=f(z+dt*k3);new=z+dt*(k1+2*k2+2*k3+k4)/6
            state[offset:offset+4]=new.ravel();state[offset+4:offset+6]=np.maximum(state[offset+4:offset+6]-1,0)
        state[16:20]*=1-.01*dt/.001
        pre=state[[0,1,6,7]].copy()
        for layer,name in enumerate(bundle.provenance['layer_names']):
            cells=bundle.provenance['threshold_margin_layout'][name]
            state[cells]=pre[2*layer:2*layer+2]-theta[2*layer:2*layer+2]
        hard=((pre>theta)&active).astype(float);s=hard.copy()
        if anchors is not None:
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][tick]-anchors[3]))**2
            s=anchors[1][tick]+active*phi*(pre-theta-anchors[0][tick]+anchors[3]);hard=anchors[1][tick]
        writable=active&(hard==0);state[12:16]=writable
        def allowed(layer,variable,j):return variable not in p['refractory'][layer]['clamp'] or writable[2*layer+j]
        for e in [1,3,0,2]:
            i,j=edges[e]
            if allowed(0,0,j):state[j]+=inputs[i]*fixed['ref_syn_0'][e]
        for e in [1,3,0,2]:
            i,j=edges[e];ids=[6+j,8+j,16+e];old=state[ids].copy();new=old.copy()
            if allowed(1,0,j):new[0]+=old[2]
            if allowed(1,1,j):new[1]+=.05
            new[2]=np.clip(old[2]+.03*new[1],0,2)
            state[ids]=old+s[i]*(new-old)
        for e in [1,3,0,2]:
            i,j=edges[e]
            if allowed(0,0,j):state[j]+=s[2+i]*fixed['ref_syn_2'][e]
        for e in [0,1,2,3]:
            i,j=edges[e];old=state[16+e];state[16+e]=old+s[2+j]*(np.clip(old-.02*state[i],0,2)-old)
        for layer,offset in enumerate([0,6]):
            old=state[offset:offset+4].copy();new=old.copy();new[2:]+=params['kick'][layer]+.1*old[:2];new[:2]-=params['theta'][layer]
            gate=hard[2*layer:2*layer+2] if p['detach_reset'] else s[2*layer:2*layer+2]
            state[offset:offset+4]=old+np.tile(gate,2)*(new-old)
            state[offset+4:offset+6]=np.where(hard[2*layer:2*layer+2]!=0,max(p['refractory'][layer]['steps']-1,0),state[offset+4:offset+6])
        pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,spikes,state,(np.array(pres),spikes,np.array(starts),theta,np.array(activities))


@pytest.mark.parametrize('clamp',[('v',),('a',),('v','a')])
@pytest.mark.parametrize('method',['euler','rk4'])
@pytest.mark.parametrize('detach,window',[(True,None),(False,3)])
def test_plastic_refractory_independent_finite_difference(clamp,method,detach,window):
    net,inp,groups,x,dt,plastic=plastic_model(clamp,method=method)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=detach,tbptt_window=window,
        trainable_neuron_parameters={g.name:['tau','theta','kick'] for g in groups})
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);out=trainer.gradients(x[None],[0])
    loss,spikes,live,anchors=plastic_oracle(bundle,bundle.weights,x)
    assert out['loss']==pytest.approx(loss,abs=2e-14)
    np.testing.assert_array_equal(out['spikes'][0],spikes);np.testing.assert_allclose(out['final_state'][0],live,rtol=1e-13,atol=1e-14)
    for bank,row in enumerate(bundle.weights):
        for j,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);a=copy.deepcopy(bundle.weights);c=copy.deepcopy(bundle.weights);a[bank][j]+=eps;c[bank][j]-=eps
            fd=(plastic_oracle(bundle,a,x,anchors=anchors)[0]-plastic_oracle(bundle,c,x,anchors=anchors)[0])/(2*eps)
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=3e-4,abs=2e-6)
    initial=np.array(bundle.initial_state);explicit=trainer.gradients(x[None],[0],initial=initial[None])
    _,_,_,anchors=plastic_oracle(bundle,bundle.weights,x,initial)
    for j in range(len(initial)):
        if bundle.plan['dynamic']['detached'][j]:
            assert explicit['initial_state_gradients'][0][j]==0;continue
        a=initial.copy();c=initial.copy();a[j]+=1e-6;c[j]-=1e-6
        fd=(plastic_oracle(bundle,bundle.weights,x,a,anchors)[0]-plastic_oracle(bundle,bundle.weights,x,c,anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][j]==pytest.approx(fd,rel=3e-4,abs=2e-7)


@pytest.mark.parametrize('ranks',[2,8])
def test_dynamic_refractory_mpi_carry_restore(ranks,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    net,inp,groups,x,dt,plastic=plastic_model(('v','a'),3.5,'rk4')
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,tbptt_window=3)
    serial=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    p=copy.deepcopy(bundle.plan);p['mpi_ranks']=ranks
    parallel=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    for start,stop in [(0,3),(3,7),(7,len(x))]:
        kw={} if start==0 else {'initial':'carry'}
        a=serial.execute(x[None,start:stop],[0],**kw);c=parallel.execute(x[None,start:stop],[0],**kw)
        for key in ['final_state','initial_state_gradients','spikes']:
            np.testing.assert_allclose(a[key],c[key],rtol=1e-13,atol=1e-13)
        for row,other in zip(a['gradients'],c['gradients']):np.testing.assert_allclose(row,other,rtol=1e-13,atol=1e-13)
        path=tmp_path/'ref.json';parallel.store(path)
        replacement=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);replacement.restore(path);parallel=replacement


@pytest.mark.parametrize('dual_write',[False,True])
@pytest.mark.parametrize('refractory',[False,True])
@pytest.mark.parametrize('detach,window',[(True,None),(False,3)])
def test_autapse_alias_snapshot_writeback_and_independent_vjp(dual_write,refractory,detach,window):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(2,[],[]*b.second,dt=dt)
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1'+(' (unless refractory)' if refractory else ''),
                         threshold='v>1',reset='v-=.3',method='euler',dt=dt,refractory=3*dt if refractory else False) for _ in range(2)]
    groups[1].v=[1.4,1.8]
    code='v_post+=w\nv_pre+=.2*w\nw=.3*v_post+.05' if dual_write else 'v_post+=w\nw=.3*v_pre+.05'
    syn=b.Synapses(groups[1],groups[1],'w:1',on_pre=code);syn.connect(i=[0,1],j=[0,1]);syn.w=[.7,.9]
    net=b.Network(inp,*groups,syn);bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=detach,tbptt_window=window)
    p=bundle.plan;weights=bundle.weights;vi=p['dynamic']['voltage'];wi=bundle.provenance['dynamic_state_layout'][syn.name]['w']
    bank=next(v['bank'] for v in bundle.provenance['bindings'] if v['object']==syn.name)
    ci=[2,3,6,7] if refractory else [];ai=list(range(8,12)) if refractory else []
    def oracle(w,initial=None,anchors=None):
        z=np.array(bundle.initial_state if initial is None else initial,float)
        if initial is None:z[wi]=w[bank]
        starts=[];pres=[];spikes=[];active_rows=[]
        for tick in range(12):
            if anchors is not None and window and tick and tick%window==0:z=anchors[2][tick].copy()
            starts.append(z.copy());active=z[ci]==0 if refractory else np.ones(4,bool)
            if anchors is not None:active=anchors[3][tick]
            active_rows.append(active.copy());z[vi]*=np.where(active,.8,1.)
            if refractory:z[ci]=np.maximum(z[ci]-1,0)
            pre=z[vi].copy();hard=((pre>1)&active).astype(float);s=hard.copy()
            if anchors is not None:
                hard=anchors[1][tick];phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][tick]-1))**2
                s=hard+active*phi*(pre-anchors[0][tick])
            allowed=active&(hard==0) if refractory else np.ones(4,bool)
            if refractory:z[ai]=allowed
            for j in range(2):
                ids=[vi[2+j],wi[j]];old=z[ids].copy();local_post=old[0]+(old[1] if allowed[2+j] else 0)
                local_pre=old[0]+(.2*old[1] if dual_write and allowed[2+j] else 0)
                final_v=local_pre if dual_write else local_post
                final_w=.3*(local_post if dual_write else local_pre)+.05
                z[ids]=old+s[2+j]*(np.array([final_v,final_w])-old)
            z[vi]-=.3*(hard if detach else s)
            if refractory:z[ci]=np.where(hard!=0,2,z[ci])
            pres.append(pre);spikes.append(s)
        spikes=np.array(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
        return loss,z,spikes,(np.array(pres),spikes,np.array(starts),np.array(active_rows))
    x=np.zeros((1,12,2));trainer=NativeLIFTrainer(p,runner=RUNNER,weights=weights);actual=trainer.gradients(x,[0])
    loss,live,spikes,anchors=oracle(weights)
    np.testing.assert_array_equal(actual['spikes'][0],spikes);np.testing.assert_allclose(actual['final_state'][0],live,rtol=1e-13,atol=1e-14)
    for j in range(2):
        a=copy.deepcopy(weights);c=copy.deepcopy(weights);a[bank][j]+=1e-6;c[bank][j]-=1e-6
        fd=(oracle(a,anchors=anchors)[0]-oracle(c,anchors=anchors)[0])/2e-6
        assert actual['gradients'][bank][j]==pytest.approx(fd,abs=2e-7,rel=3e-4)
    initial=np.array(bundle.initial_state);actual_initial=trainer.gradients(x,[0],initial=initial[None]);_,_,_,anchors=oracle(weights,initial)
    for index in vi+wi:
        a=initial.copy();c=initial.copy();a[index]+=1e-6;c[index]-=1e-6
        fd=(oracle(weights,a,anchors)[0]-oracle(weights,c,anchors)[0])/2e-6
        assert actual_initial['initial_state_gradients'][0][index]==pytest.approx(fd,abs=2e-7,rel=3e-4)
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    syn.pre.codeobj_class=CythonCodeObject;net.run(12*dt,namespace={})
    np.testing.assert_allclose(np.array(actual['final_state'])[0,wi],syn.w[:],rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(actual['final_membrane'][0],np.r_[groups[0].v[:],groups[1].v[:]],rtol=1e-13,atol=1e-14)
