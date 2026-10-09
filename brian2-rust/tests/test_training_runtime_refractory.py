"""Compiled Brian conditional destinations and runtime-selector VJPs."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine
from test_training_delay_update import snapshot


def model(kind='single',*,mutate=True,method='euler',noisy=False,delay=False,post=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,1],[1,0],[0,1],[1,1],[0,1],[1,0]],float);ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='guard_input')
    a=b.NeuronGroup(2,'dv/dt=-v/ms'+('+sigma*xi/sqrt(ms)' if noisy else '')+':1 (unless refractory)'+('\nsigma:1 (constant)' if noisy else ''),
        threshold='v>.5',reset='v-=.2',refractory=3*dt,method=method,dt=dt,name='guard_a');a.v=[.8,.2]
    c=b.NeuronGroup(2,'dv/dt=(-v+.2*signal)/ms:1 (unless refractory)\nsignal:1 (linked)',
        threshold='v>.5',reset='v-=.2',refractory=3*dt,method=method,dt=dt,name='guard_c');c.v=[.1,.8];c.signal=b.linked_var(a,'v')
    if noisy:a.sigma=[.03,.05]
    alias='zlink' if kind in ('late','read_late','flag_late') else 'peer'
    code=('pick=1-pick\n' if mutate else '')
    if kind.startswith('flag_'):code+='not_refractory=False\n'
    if not kind.startswith('read'):code+=f'{alias}+=w\n'
    if kind.startswith('flag_'):code+='not_refractory=True\n'
    if kind!='single':code+=f'v_post+=.1*{alias}\n'
    if kind=='explicit' or kind.startswith('flag_'):code+='w+=.01*int(not_refractory)\n'
    code+='w+=.01'
    kw=dict(on_post=code) if post else dict(on_pre=code)
    syn=b.Synapses(inp,c,f'w:1\n{alias}:1 (linked)\npick:integer',dt=dt,name='guard_syn',**kw)
    syn.connect(i=[0,1],j=[0,1]);syn.pick=[0,1];syn.w=[.03,.07];setattr(syn,alias,b.linked_var(a,'v',index='pick'))
    if delay:getattr(syn,'post' if post else 'pre').delay=[0,2]*dt
    net=b.Network(inp,a,c,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],
        trainable_neuron_parameters={a.name:['sigma']} if noisy else None,**options)
    return net,inp,a,c,syn,x,bundle


@pytest.mark.parametrize('kind',['single','early','late','read_early','read_late','explicit','flag_early','flag_late'])
@pytest.mark.parametrize('mutate',[False,True])
@pytest.mark.parametrize('method',['euler','rk2','rk4'])
def test_conditional_indexing_matches_compiled_brian(kind,mutate,method):
    net,inp,a,c,syn,x,bundle=model(kind,mutate=mutate,method=method)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    sm=[b.SpikeMonitor(g) for g in (a,c)];net.add(*sm);net.run(len(x)*.2*b.ms,namespace={})
    assert syn.pre.codeobj.compiled_code['run'] is not None
    expected=np.zeros((len(x),4))
    for l,m in enumerate(sm):expected[np.rint(np.asarray(m.t/b.second)/.0002).astype(int),2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(out['spikes'][0],expected)
    z=np.asarray(out['final_state'])[0];layout=bundle.provenance['neuron_state_layout']
    for g in (a,c):np.testing.assert_allclose(z[layout[g.name]['v']],np.asarray(g.v[:]),rtol=1e-13,atol=1e-13)
    sl=bundle.provenance['dynamic_state_layout'][syn.name]
    np.testing.assert_allclose(z[sl['w']],np.asarray(syn.w[:]),rtol=1e-13,atol=1e-13)
    np.testing.assert_array_equal(z[sl['pick']],np.asarray(syn.pick[:]))
    assert syn.pre.codeobj.variable_indices['not_refractory']==('pick' if kind in ('single','late','read_late','flag_late') else '_postsynaptic_idx')


def reference(bundle,x,kind,*,initial=None,weights=None,anchors=None,post=False):
    """Independent Euler recurrence, with hard routing and local-surrogate gates."""
    from test_training_stochastic import normal
    p=bundle.plan;spec=p['dynamic'];weights=bundle.weights if weights is None else weights
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for k,ref in enumerate(spec['initial_parameters']):
            if ref is not None:z[k]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'];av=layout['guard_a']['v'];cv=layout['guard_c']['v']
    ac=layout['guard_a']['__refractory_ticks'];cc=layout['guard_c']['__refractory_ticks']
    # The frontend emits activity cells immediately after the rectangular prefix.
    start=sum(len(names)*2 for names in bundle.provenance['state_names']);af=[start,start+1];cf=[start+2,start+3]
    sl=bundle.provenance['dynamic_state_layout']['guard_syn'];wi=sl['w'];pi=sl['pick']
    sigma=next((np.asarray(weights[e['bank']]) for e in bundle.provenance['bindings'] if e['object']=='guard_a' and 'sigma' in e['variables']),None)
    before=[];voltages=[];hard_spikes=[];active=[];spikes=[]
    for t,inp in enumerate(x):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors['before'][t].copy()
        before.append(z.copy());free_a=(z[ac]==0);free_c=(z[cc]==0)
        value=.8*z[av]
        if sigma is not None:value+=sigma*np.sqrt(.2)*np.array([normal(p['seed'],0,0,0,j,t,0) for j in range(2)])
        z[av]=np.where(free_a,value,z[av]);z[cv]=np.where(free_c,.8*z[cv]+.04*z[av],z[cv])
        z[ac]=np.maximum(z[ac]-1,0);z[cc]=np.maximum(z[cc]-1,0)
        free=np.r_[free_a,free_c];v=z[av+cv].copy();hard=((v>.5)&free).astype(float);soft=hard.copy()
        if anchors is not None:
            old=anchors['v'][t];hard=anchors['hard'][t]
            soft=hard+anchors['active'][t]*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old-.5))**2*(v-old)
        voltages.append(v);hard_spikes.append(hard);active.append(free);spikes.append(soft)
        z[af]=free_a*(1-hard[:2]);z[cf]=free_c*(1-hard[2:])
        for j in range(2):
            event=hard[2+j] if post else inp[j];gate=soft[2+j] if post else inp[j]
            old=int(z[pi[j]]);new=1-old;read=z[av[old]];weight=z[wi[j]];c_old=z[cv[j]]
            condition=z[af[old]] if kind in ('single','late','read_late','flag_late') else z[cf[j]]
            peer=read if kind.startswith('read') or kind.startswith('flag_') else read+condition*weight
            if kind.startswith('flag_'):condition=1.
            if not kind.startswith('read'):
                target=av[new];z[target]+=gate*(peer-z[target])
            if kind!='single':z[cv[j]]+=gate*(condition*.1*peer)
            z[wi[j]]+=gate*(.01+(.01*condition if kind=='explicit' or kind.startswith('flag_') else 0))
            if event:
                z[pi[j]]=new
                if kind.startswith('flag_'):z[af[new] if kind=='flag_late' else cf[j]]=1.
        reset=hard if p['detach_reset'] else soft
        z[av]-=.2*reset[:2];z[cv]-=.2*reset[2:]
        for j in range(2):
            if hard[j]:z[ac[j]]=2
            if hard[2+j]:z[cc[j]]=2
    logits=np.asarray(spikes)[:,2:].mean(axis=0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.asarray(spikes),dict(before=before,v=voltages,hard=hard_spikes,active=active)


@pytest.mark.parametrize('kind',['single','early','late','read_early','read_late','explicit','flag_early','flag_late'])
@pytest.mark.parametrize('post',[False,True])
@pytest.mark.parametrize('noisy,detach,window',[(False,True,None),(True,False,None),(True,False,2)])
def test_every_differentiable_state_parameter_vjp(engine,kind,post,noisy,detach,window):
    *_,x,bundle=model(kind,post=post,noisy=noisy,detach_reset=detach,tbptt_window=window,backend=engine)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);before=snapshot(trainer)
    result=trainer.gradients(x[None],[0]);assert snapshot(trainer)[:-1]==before[:-1]
    expected,z,spikes,anchors=reference(bundle,x,kind,post=post)
    tol=4e-6 if engine=='cpu' else 9e-4;absolute=4e-8 if engine=='cpu' else 6e-6
    assert result['loss']==pytest.approx(expected,abs=absolute)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],z,rtol=tol,atol=absolute)
    if engine!='cpu':assert result['gpu_dispatches']>0
    eps=1e-6
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(reference(bundle,x,kind,weights=hi,anchors=anchors,post=post)[0]-reference(bundle,x,kind,weights=lo,anchors=anchors,post=post)[0])/(2*eps)
            assert result['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute)
    for k,detached in enumerate(bundle.plan['dynamic']['detached']):
        if detached:
            assert result['initial_state_gradients'][0][k]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(reference(bundle,x,kind,initial=hi,anchors=anchors,post=post)[0]-reference(bundle,x,kind,initial=lo,anchors=anchors,post=post)[0])/(2*eps)
        assert result['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute)


def neuron_model(*,dynamic=True,duplicate=False,threshold_link=True,method='euler',**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,1],[1,0],[0,1],[1,1],[0,1],[1,0]],float);ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='guard_input')
    a=b.NeuronGroup(2,'dv/dt=-v/ms:1 (unless refractory)',threshold='v>.5',reset='v-=.2',refractory=3*dt,method=method,dt=dt,name='guard_a');a.v=[.8,.2]
    c=b.NeuronGroup(2,'dv/dt=(-v+.2*zsignal)/ms:1 (unless refractory)\nzsignal:1 (linked)'+('\npick:integer' if dynamic else ''),
        threshold='v+.1*zsignal>.5' if threshold_link else 'v>.5',reset='v-=.2'+('\npick=1-pick' if dynamic else ''),refractory=3*dt,method=method,dt=dt,name='guard_c');c.v=[.1,.8]
    mapping=[0,0] if duplicate else [1,0]
    if dynamic:c.pick=mapping
    c.zsignal=b.linked_var(a,'v',index='pick' if dynamic else mapping)
    syn=b.Synapses(inp,c,'w:1',on_pre='v_post+=w',dt=dt,name='guard_syn');syn.connect(j='i');syn.w=[.12,.16]
    net=b.Network(inp,a,c,syn);bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],**options)
    return net,inp,a,c,syn,x,bundle


@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('duplicate',[False,True])
@pytest.mark.parametrize('threshold_link',[False,True])
@pytest.mark.parametrize('method',['euler','rk4'])
def test_neuron_condition_writeback_matches_compiled_brian(dynamic,duplicate,threshold_link,method):
    net,inp,a,c,syn,x,bundle=neuron_model(dynamic=dynamic,duplicate=duplicate,threshold_link=threshold_link,method=method)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);states=[];spikes=[]
    for t in range(len(x)):
        out=trainer.step(x[None,t:t+1],[0],initial='carry' if t else None);states.append(out['final_state'][0]);spikes.append(out['spikes'][0][0])
    monitors=[b.StateMonitor(g,['v','not_refractory'],record=True,when='end') for g in (a,c)];sm=[b.SpikeMonitor(g) for g in (a,c)];net.add(*monitors,*sm)
    net.run(len(x)*.2*b.ms,namespace={});states=np.asarray(states)
    expected=np.zeros((len(x),4))
    for l,(g,m,s) in enumerate(zip((a,c),monitors,sm)):
        np.testing.assert_allclose(states[:,bundle.provenance['neuron_state_layout'][g.name]['v']].T,m.v[:],rtol=1e-13,atol=1e-13)
        np.testing.assert_array_equal(states[:,bundle.provenance['refractory_activity_layout'][g.name]].T,m.not_refractory[:])
        expected[np.rint(np.asarray(s.t/b.second)/.0002).astype(int),2*l+np.asarray(s.i)]=1
    np.testing.assert_array_equal(spikes,expected)
    assert c.state_updater.codeobj.compiled_code['run'] is not None
    assert c.state_updater.codeobj.variable_indices['not_refractory']==c.variables.indices['zsignal']
    assert c.thresholder['spike'].codeobj.variable_indices['not_refractory']==(c.variables.indices['zsignal'] if threshold_link else '_idx')


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('case',['pre','post','neuron','flag_pre','flag_post'])
def test_runtime_guard_carry_checkpoint_mask_and_rollback(engine,ranks,case,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    kw=dict(backend=engine,mpi_ranks=ranks,detach_reset=False,tbptt_window=2)
    if case=='neuron':*_,x,bundle=neuron_model(duplicate=True,threshold_link=True,**kw)
    else:*_,x,bundle=model('flag_late' if case.startswith('flag_') else 'explicit',post=case.endswith('post'),noisy=True,delay=True,**kw)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None);cpu=NativeLIFTrainer(q,runner=RUNNER,weights=bundle.weights)
    initial=np.tile(bundle.initial_state,(2,1));layout=bundle.provenance['neuron_state_layout']
    ix=layout['guard_c']['pick'] if case=='neuron' else bundle.provenance['dynamic_state_layout']['guard_syn']['pick']
    initial[1,ix]=[1,0];xx=np.stack([x,x[:,::-1]])
    for t in (0,2,4):
        kw=dict(initial='carry') if t else dict(initial=initial,start_tick=3,**({} if case=='neuron' else dict(noise_sequence=7)))
        actual=trainer.step(xx[:,t:t+2],[0,1],**kw);expected=cpu.step(xx[:,t:t+2],[0,1],**kw)
        for key in ('final_state','initial_state_gradients','spikes'):
            np.testing.assert_allclose(actual[key],expected[key],rtol=9e-4,atol=6e-6)
        for a,c in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,c,rtol=9e-4,atol=6e-6)
        if engine!='cpu':assert actual['gpu_dispatches']>0
        if ranks:assert 'mpi-dynamic' in actual['numeric_profile']
        if t<4:
            bank,index=p['dynamic']['migration']['controlled_masks'][-1]
            for tr in (trainer,cpu):
                masks=copy.deepcopy(tr.plan['masks']);masks[bank][index]=0 if t==0 else 1;tr.update_mask(masks,growth_weight=.1)
        path=tmp_path/'guard.json';trainer.store(path);restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
        assert trainer.clock_tick==cpu.clock_tick and trainer.noise_sequence==cpu.noise_sequence
    before=snapshot(trainer);bad=initial.copy();bad[1,ix[1]]=-1
    with pytest.raises(ValueError):trainer.step(xx,[0,1],initial=bad)
    assert snapshot(trainer)==before


def neuron_reference(bundle,x,*,initial=None,weights=None,anchors=None):
    """Independent scalar Euler loops, including indexed flag stores/clears."""
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,dtype=float).copy()
    layout=bundle.provenance['neuron_state_layout'];av=layout['guard_a']['v'];cv=layout['guard_c']['v'];ix=layout['guard_c']['pick']
    ac=layout['guard_a']['__refractory_ticks'];cc=layout['guard_c']['__refractory_ticks']
    af=bundle.provenance['refractory_activity_layout']['guard_a'];cf=bundle.provenance['refractory_activity_layout']['guard_c']
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='guard_syn');w=weights[bank]
    before=[];margins=[];hard_spikes=[];active=[];spikes=[]
    for t,inp in enumerate(x):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors['before'][t].copy()
        before.append(z.copy());free_a=(z[ac]==0);free_c=(z[cc]==0)
        z[av]=np.where(free_a,.8*z[av],z[av]);z[af]=free_a
        picks=z[ix].astype(int)
        for j in range(2):
            if free_c[j]:z[cv[j]]=.8*z[cv[j]]+.04*z[av[picks[j]]]
            z[cf[picks[j]]]=free_c[j]
        z[ac]=np.maximum(z[ac]-1,0);z[cc]=np.maximum(z[cc]-1,0)
        hard=[];soft=[];margin=[];flags=[]
        for l in range(2):
            for j in range(2):
                m=z[av[j]]-.5 if l==0 else z[cv[j]]+.1*z[av[picks[j]]]-.5
                flag=z[af[j]] if l==0 else z[cf[picks[j]]]
                k=2*l+j;h=float(m>0 and flag!=0);s=h
                if anchors is not None:
                    h=anchors['hard'][t][k];old=anchors['margin'][t][k]
                    s=h+anchors['active'][t][k]*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(m-old)
                hard.append(h);soft.append(s);margin.append(m);flags.append(flag)
                if h:z[(af if l==0 else cf)[j]]=0
        hard=np.array(hard);soft=np.array(soft);spikes.append(soft);margins.append(margin);hard_spikes.append(hard);active.append(flags)
        for j in range(2):z[cv[j]]+=inp[j]*z[cf[j]]*w[j]
        gate=hard if p['detach_reset'] else soft
        z[av]-=.2*gate[:2];z[cv]-=.2*gate[2:]
        for j in range(2):
            if hard[j]:z[ac[j]]=2
            if hard[2+j]:z[cc[j]]=2;z[ix[j]]=1-z[ix[j]]
    logits=np.asarray(spikes)[:,2:].mean(axis=0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.asarray(spikes),dict(before=before,margin=margins,hard=hard_spikes,active=active)


@pytest.mark.parametrize('duplicate',[False,True])
@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(False,2)])
def test_neuron_indexed_guards_every_float_vjp(engine,duplicate,detach,window):
    *_,x,bundle=neuron_model(duplicate=duplicate,detach_reset=detach,tbptt_window=window,backend=engine)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    loss,z,spikes,anchors=neuron_reference(bundle,x)
    tol=4e-6 if engine=='cpu' else 9e-4;absolute=4e-8 if engine=='cpu' else 6e-6
    assert result['loss']==pytest.approx(loss,abs=absolute);np.testing.assert_array_equal(result['spikes'][0],spikes)
    for name,row in bundle.provenance['neuron_state_layout'].items():
        for var,indices in row.items():
            if var=='zsignal':continue
            np.testing.assert_allclose(np.asarray(result['final_state'])[0,indices],z[indices],rtol=tol,atol=absolute)
    for indices in bundle.provenance['refractory_activity_layout'].values():np.testing.assert_array_equal(np.asarray(result['final_state'])[0,indices],z[indices])
    eps=1e-6
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(neuron_reference(bundle,x,weights=hi,anchors=anchors)[0]-neuron_reference(bundle,x,weights=lo,anchors=anchors)[0])/(2*eps)
            assert result['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute)
    for k,detached in enumerate(bundle.plan['dynamic']['detached']):
        if detached:
            assert result['initial_state_gradients'][0][k]==0;continue
        hi=np.asarray(bundle.initial_state).copy();lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(neuron_reference(bundle,x,initial=hi,anchors=anchors)[0]-neuron_reference(bundle,x,initial=lo,anchors=anchors)[0])/(2*eps)
        assert result['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute)


def test_snapshot_keeps_unwritten_refractory_flags():
    net,inp,a,c,syn,x,_=neuron_model(dynamic=False,duplicate=True,threshold_link=False)
    net.run(2*.2*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c])
    flags=bundle.provenance['refractory_activity_layout']
    for g in (a,c):np.testing.assert_array_equal(np.asarray(bundle.initial_state)[flags[g.name]],g.not_refractory[:])
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None,2:],[0])
    sm=[b.SpikeMonitor(g) for g in (a,c)];net.add(*sm);net.run(4*.2*b.ms,namespace={})
    expected=np.zeros((4,4))
    for l,(g,m) in enumerate(zip((a,c),sm)):
        expected[np.rint(np.asarray(m.t/b.second)/.0002).astype(int)-2,2*l+np.asarray(m.i)]=1
        indices=bundle.provenance['neuron_state_layout'][g.name]['v']
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,indices],g.v[:],rtol=1e-13,atol=1e-13)
        np.testing.assert_array_equal(np.asarray(out['final_state'])[0,flags[g.name]],g.not_refractory[:])
    np.testing.assert_array_equal(out['spikes'][0],expected)


def test_mutable_boolean_guard_is_explicit_and_lazy(engine):
    from brian2_rust.training_dynamic import compile_dynamic_transform
    kw=dict(states={'v':0,'gate':1},write_guards={0:1},state_types={0:'float',1:'boolean'})
    code='gate=False\nv=log(v)\ngate=True\nv+=.1'
    with pytest.raises(ValueError,match='read-only'):compile_dynamic_transform(code,**kw)
    with pytest.raises(ValueError,match='Boolean'):compile_dynamic_transform(code,**dict(kw,state_types={}),mutable_write_guards={1})
    transform=compile_dynamic_transform(code,**kw,mutable_write_guards={1})
    *_,x,bundle=neuron_model(backend=engine);p=copy.deepcopy(bundle.plan);d=p['dynamic'];v=bundle.provenance['neuron_state_layout']['guard_c']['v'][0];gate=bundle.provenance['refractory_activity_layout']['guard_c'][0]
    d['program_sets'].append([[dict(op='constant',value=-.2)]])
    d['actions'].append(dict(owner=2,reads=[v],writes=[v],program_set=len(d['program_sets'])-1,threshold=None,trigger=None))
    d['program_sets'].append(transform['programs'])
    d['actions'].append(dict(owner=2,reads=[v,gate],writes=[v,gate],program_set=len(d['program_sets'])-1,threshold=None,trigger=None))
    # The disabled log(-.2) branch must never be evaluated; the later +.1 is active.
    actual=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).evaluate(x[None,:1],[0])
    q=copy.deepcopy(bundle.plan);qd=q['dynamic']
    qd['program_sets'].append([[dict(op='constant',value=-.1)],[dict(op='constant',value=1.)]])
    qd['actions'].append(dict(owner=2,reads=[v,gate],writes=[v,gate],program_set=len(qd['program_sets'])-1,threshold=None,trigger=None))
    expected=NativeLIFTrainer(q,runner=RUNNER,weights=bundle.weights).evaluate(x[None,:1],[0])
    np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
    np.testing.assert_allclose(actual['final_state'],expected['final_state'],rtol=0,atol=0)
