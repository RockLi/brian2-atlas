"""Dynamic native action BPTT: STDP state, ordered event maps and full VJP."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer
from brian2_rust.training import lif_training_plan
from brian2_rust.training_equations import neuron_parameter_bank
from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
from test_native_training import RUNNER


def model(detach=True,window=None,event_driven=False):
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[1,1],[0,1],[1,0],[0,0],[1,1],[1,0],[1,1]],float)
    weights=[[.85,.95,1.05,.75],[.7,.8,1.,.9],[.001,.0013,.065,-.05]]
    projections=[neuron_parameter_bank(len(w)) for w in weights]
    identity=[[[dict(op='state',index=0)]]]*2
    plan=lif_training_plan([2,2,2],projections=projections,state_equations=identity,state_resets=identity,
                           clock=dict(origin=0.,dt=.0002),beta=.85,threshold=1.,detach_reset=detach,
                           tbptt_window=window,learning_rate=1e-6)
    programs=[];actions=[]
    def add(code,states,parameters,reads,owner,trigger=None,detach_trigger=False):
        compiled=compile_dynamic_transform(code,states=states,parameters=parameters)
        index=len(programs);programs.append(compiled['programs'])
        actions.append(dynamic_action(compiled,reads,owner=owner,program_set=index,trigger=trigger,detach_trigger=detach_trigger))
    for j in range(4):add('v=.85*v',{'v':0},{},[j],j)
    for edge in range(0 if event_driven else 4):add('pre=pre*exp(-dt/taup)\npost=post*exp(-dt/taum)',{'pre':0,'post':1},
                            dict(dt=.0002,taup=(2,0),taum=(2,1)),[8+edge,12+edge],2+edge%2)
    for j in range(4):actions.append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None))
    for i in range(2):
        for j in range(2):add('v+=w',{'v':0},{'w':(0,2*i+j)},[j],j,dict(external=True,index=i))
    for edge in range(4):
        i,j=divmod(edge,2)
        code='v+=w\npre+=Ap\nw=clip(w+post,0,1.2)'
        states=dict(v=0,w=1,pre=2,post=3);reads=[2+j,4+edge,8+edge,12+edge];params={'Ap':(2,2)}
        if event_driven:
            from brian2_rust.training_equations import SimulationTime
            code='pre*=exp(-(t-last)/taup)\npost*=exp(-(t-last)/taum)\n'+code+'\nlast=t'
            states['last']=4;reads.append(16+edge);params.update(t=SimulationTime(),taup=(2,0),taum=(2,1))
        add(code,states,params,reads,2+j,dict(external=False,index=i))
    for edge in [0,2,1,3]:
        i,j=divmod(edge,2)
        code='post+=Am\nw=clip(w+pre,0,1.2)'
        states=dict(w=0,pre=1,post=2);reads=[4+edge,8+edge,12+edge];params={'Am':(2,3)}
        if event_driven:
            from brian2_rust.training_equations import SimulationTime
            code='pre*=exp(-(t-last)/taup)\npost*=exp(-(t-last)/taum)\n'+code+'\nlast=t'
            states['last']=3;reads.append(16+edge);params.update(t=SimulationTime(),taup=(2,0),taum=(2,1))
        add(code,states,params,reads,2+j,dict(external=False,index=2+j))
    for j in range(4):add('v-=1',{'v':0},{},[j],j,dict(external=False,index=j),detach)
    initial=[.2,1.2,.1,1.6,*weights[1],.02,.03,.04,.01,-.02,-.01,-.04,-.03]
    if event_driven:initial.extend([0.]*4)
    bindings=[None]*len(initial)
    for edge in range(4):bindings[4+edge]=[1,edge]
    plan.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(initial=initial,initial_parameters=bindings,detached=[False]*16+([True]*4 if event_driven else []),
        voltage=[0,1,2,3],program_sets=programs,actions=actions))
    return plan,weights,x


def oracle(p,w,x,initial=None,anchors=None,sequence=0,start_tick=0,clock_times=None):
    z=np.array(p['dynamic']['initial'] if initial is None else initial,float)
    if initial is None:z[4:8]=w[1]
    old=[];pres=[];spikes=[];contexts=[]
    for t,inp in enumerate(x):
        timestamp=p['clock']['origin']+(start_tick+t)*.0002 if clock_times is None else clock_times[t]
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors[2][t].copy()
        old.append(z.copy());z[:4]*=.85
        event_driven=len(z)>16
        if len(w)>3:
            from test_training_stochastic import normal
            z[8:12]+=-.0002*z[8:12]/w[2][0]+w[3][0]*np.sqrt(.2)*np.array([normal(p['seed'],sequence,0,2,e,start_tick+t,0) for e in range(4)])
            z[12:16]*=(1-.0002/w[2][1])
        elif not event_driven:z[8:12]*=np.exp(-.0002/w[2][0]);z[12:16]*=np.exp(-.0002/w[2][1])
        pre=z[:4].copy();s=(pre>1.).astype(float);hard=s.copy()
        if anchors is not None:
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][t]-1.))**2
            s=anchors[1][t]+phi*(pre-anchors[0][t]);hard=anchors[1][t]
        for i in range(2):
            for j in range(2):z[j]+=inp[i]*w[0][2*i+j]
        # Each full path is one event transform, not independently gated statements.
        for edge in range(4):
            i,j=divmod(edge,2);ids=[2+j,4+edge,8+edge,12+edge];before=z[ids].copy();after=before.copy()
            if event_driven:
                delta=timestamp-z[16+edge];after[2]*=np.exp(-delta/w[2][0]);after[3]*=np.exp(-delta/w[2][1])
            after[0]+=after[1];after[2]+=w[2][2];after[1]=np.clip(after[1]+after[3],0,1.2)
            z[ids]=before+s[i]*(after-before)
            if event_driven and hard[i]:z[16+edge]=timestamp
        for edge in [0,2,1,3]:
            i,j=divmod(edge,2);ids=[4+edge,8+edge,12+edge];before=z[ids].copy();after=before.copy()
            if event_driven:
                delta=timestamp-z[16+edge];after[1]*=np.exp(-delta/w[2][0]);after[2]*=np.exp(-delta/w[2][1])
            after[2]+=w[2][3];after[0]=np.clip(after[0]+after[1],0,1.2)
            z[ids]=before+s[2+j]*(after-before)
            if event_driven and hard[2+j]:z[16+edge]=timestamp
        z[:4]-=hard if p['detach_reset'] else s
        pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,2:].mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,spikes,z,(np.array(pres),spikes,np.array(old))


@pytest.mark.parametrize('event_driven,noisy',[(False,False),(True,False),(False,True)])
def test_dynamic_forward_matches_brian_stdp(event_driven,noisy):
    p,w,x=noisy_model() if noisy else model(event_driven=event_driven);out=NativeLIFTrainer(p,runner=RUNNER,weights=w).evaluate(x[None],[0])
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    ticks,ids=np.nonzero(x);inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='dyn_input')
    groups=[b.NeuronGroup(2,'dv/dt=-v/tau:1',threshold='v>1',reset='v-=1',method='euler',dt=dt,
                          namespace={'tau':float(dt)/.15*b.second},name=f'dyn_layer_{l}') for l in range(2)]
    groups[0].v=p['dynamic']['initial'][:2];groups[1].v=p['dynamic']['initial'][2:4]
    static=b.Synapses(inp,groups[0],'w:1',on_pre='v_post+=w',name='dyn_static');static.connect();static.w=w[0]
    flag='event-driven' if event_driven else 'clock-driven'
    noise_term='+sigma*xi/sqrt(ms)' if noisy else ''
    syn=b.Synapses(groups[0],groups[1],f'''w:1
        dapre/dt=-apre/taup{noise_term}:1 ({flag})
        dapost/dt=-apost/taum:1 ({flag})''',
        on_pre='v_post+=w\napre+=Ap\nw=clip(w+apost,0,1.2)',on_post='apost+=Am\nw=clip(w+apre,0,1.2)',method='euler' if noisy else 'exact',dt=dt,
        namespace=dict(taup=w[2][0]*b.second,taum=w[2][1]*b.second,Ap=w[2][2],Am=w[2][3],sigma=w[3][0] if noisy else 0.),name='dyn_stdp')
    syn.connect();syn.w=w[1];syn.apre=p['dynamic']['initial'][8:12];syn.apost=p['dynamic']['initial'][12:16]
    monitors=[b.SpikeMonitor(g) for g in groups];net=b.Network(inp,*groups,static,syn,*monitors)
    if noisy:
        from unittest.mock import patch
        from test_training_stochastic import normal
        net.run(0*dt,namespace={})
        draws=iter([np.array([normal(p['seed'],0,0,2,e,t,0) for e in range(4)]) for t in range(len(x))])
        def randn(n):
            assert n==4
            return next(draws)
        with patch('numpy.random.randn',randn):net.run(len(x)*dt,namespace={})
        with pytest.raises(StopIteration):next(draws)
    else:net.run(len(x)*dt,namespace={})
    expected=[*groups[0].v[:],*groups[1].v[:],*syn.w[:],*syn.apre[:],*syn.apost[:]]
    if event_driven:expected.extend(syn.lastupdate[:]/b.second)
    spikes=np.zeros((len(x),4))
    for l,m in enumerate(monitors):spikes[np.rint(np.asarray(m.t/b.second)/float(dt)).astype(int),2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(out['spikes'][0],spikes)
    np.testing.assert_allclose(out['final_state'][0],expected,atol=4e-14,rtol=4e-14)


@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
@pytest.mark.parametrize('explicit_initial',[False,True])
@pytest.mark.parametrize('event_driven',[False,True])
def test_dynamic_independent_stdp_vjp(detach,window,explicit_initial,event_driven):
    p,w,x=model(detach,window,event_driven);initial=np.array(p['dynamic']['initial']) if explicit_initial else None
    out=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None],[0],initial=None if initial is None else initial[None])
    loss,spikes,live,anchors=oracle(p,w,x,initial)
    assert out['loss']==pytest.approx(loss,abs=2e-14)
    np.testing.assert_array_equal(out['spikes'][0],spikes);np.testing.assert_allclose(out['final_state'][0],live,rtol=5e-14,atol=5e-14)
    for bank,row in enumerate(w):
        for j,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);a=copy.deepcopy(w);c=copy.deepcopy(w);a[bank][j]+=eps;c[bank][j]-=eps
            fd=(oracle(p,a,x,initial,anchors)[0]-oracle(p,c,x,initial,anchors)[0])/(2*eps)
            assert out['gradients'][bank][j]==pytest.approx(fd,abs=3e-6,rel=3e-4)
    if explicit_initial:
        for j in range(len(initial)):
            if p['dynamic']['detached'][j]:
                assert out['initial_state_gradients'][0][j]==0
                continue
            a=initial.copy();c=initial.copy();a[j]+=1e-6;c[j]-=1e-6
            fd=(oracle(p,w,x,a,anchors)[0]-oracle(p,w,x,c,anchors)[0])/2e-6
            assert out['initial_state_gradients'][0][j]==pytest.approx(fd,abs=2e-7,rel=3e-4)


@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('event_driven',[False,True])
@pytest.mark.parametrize('detach',[False,True])
def test_dynamic_mpi_carry_and_restore(ranks,event_driven,detach,tmp_path):
    import json,subprocess,sys
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(detach=detach,event_driven=event_driven);mp=copy.deepcopy(p);mp['mpi_ranks']=ranks
    serial=NativeLIFTrainer(p,runner=RUNNER,weights=w);parallel=NativeLIFTrainer(mp,runner=RUNNER,weights=w)
    batch=np.stack([x,x[::-1]]);labels=[0,1]
    for start,end in [(0,3),(3,7)]:
        initial=None if start==0 else 'carry'
        a=serial.step(batch[:,start:end],labels,initial=initial);c=parallel.step(batch[:,start:end],labels,initial=initial)
        for key in ['loss','spikes','final_state','initial_state_gradients','gradients','logits']:
            for av,cv in zip(a[key],c[key]) if key=='gradients' else [(a[key],c[key])]:np.testing.assert_allclose(av,cv,rtol=3e-12,atol=3e-12)
        for key in ['weights','first_moment','second_moment']:
            for av,cv in zip(a['state'][key],c['state'][key]):np.testing.assert_allclose(av,cv,rtol=3e-12,atol=3e-12)
    path=tmp_path/'checkpoint';parallel.store(path);req=tmp_path/'req';out=tmp_path/'out'
    req.write_text(json.dumps(dict(plan=mp,x=batch[:,7:].tolist(),labels=labels)))
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
r=json.load(open(sys.argv[1]));t=NativeLIFTrainer(r['plan'],runner=sys.argv[4]);t.restore(sys.argv[2])
json.dump(t.step(r['x'],r['labels'],initial='carry'),open(sys.argv[3],'w'))
'''
    subprocess.run([sys.executable,'-c',code,str(req),str(path),str(out),str(RUNNER)],check=True,timeout=120)
    assert json.loads(out.read_text())==parallel.step(batch[:,7:],labels,initial='carry')


@pytest.mark.parametrize('event_driven',[False,True])
def test_dynamic_frozen_split_continuity_and_explicit_initial(event_driven):
    p,w,x=model(event_driven=event_driven);p['trainable']=[False]*len(w)
    full=NativeLIFTrainer(p,runner=RUNNER,weights=w).step(x[None],[0])
    t=NativeLIFTrainer(p,runner=RUNNER,weights=w);a=t.step(x[None,:5],[0]);c=t.step(x[None,5:],[0],initial='carry')
    np.testing.assert_array_equal(a['spikes'][0]+c['spikes'][0],full['spikes'][0]);assert c['final_state']==full['final_state']
    assert c['state']['weights']==w and c['final_state'][0][4:8]!=w[1]
    # Explicit state overrides optimizer-linked initial values: no implicit reset.
    new_weights=copy.deepcopy(w);new_weights[1]=[.01]*4
    independent=NativeLIFTrainer(p,runner=RUNNER,weights=new_weights).evaluate(x[None,5:],[0],initial=a['final_state'],start_tick=5)
    assert independent['final_state']==c['final_state']


@pytest.mark.parametrize('change',['missing_threshold','repeat_threshold','invalid_read','invalid_write','duplicate_write',
                                   'program_range','program_context','owner','initial_binding','detached_voltage','initial_width','gate_order','mask_reference'])
def test_dynamic_validation_is_atomic(change):
    p,w,x=model();spec=p['dynamic']
    if change=='missing_threshold':next(a for a in spec['actions'] if a['threshold']==0)['threshold']=None
    elif change=='repeat_threshold':next(a for a in spec['actions'] if a['threshold']==1)['threshold']=0
    elif change=='invalid_read':spec['actions'][0]['reads']=[1000]
    elif change=='invalid_write':spec['actions'][0]['writes']=[1000]
    elif change=='duplicate_write':spec['actions'][4]['writes']=[8,8]
    elif change=='program_range':spec['actions'][0]['program_set']=1000
    elif change=='program_context':spec['program_sets'][0][0]=[dict(op='state',index=63)]
    elif change=='owner':spec['actions'][0]['owner']=4
    elif change=='initial_binding':spec['initial_parameters'][4]=[1,4]
    elif change=='detached_voltage':spec['detached'][0]=True
    elif change=='initial_width':spec['initial']=[]
    elif change=='gate_order':spec['actions'][0]['trigger']=dict(external=False,index=0)
    else:spec['actions'][0]['mask']=[0,4]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step(x[None],[0])
    assert trainer.state==before and trainer.neuron_state is None and trainer.elapsed_ticks==0


def test_dynamic_memory_budget_and_binary_events():
    p,w,x=model();out=NativeLIFTrainer(p,runner=RUNNER,weights=w).evaluate(x[None],[0])
    p['max_tape_bytes']=out['tape_bytes']-1;t=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    with pytest.raises(ValueError,match='budget'):t.step(x[None],[0])
    p['max_tape_bytes']=64*1024**2;t=NativeLIFTrainer(p,runner=RUNNER,weights=w);x[0,0]=.5
    with pytest.raises(ValueError,match='binary'):t.step(x[None],[0])
    assert t.neuron_state is None


def test_dynamic_masked_edge_does_not_grow_through_stdp():
    p,w,x=model();edge=0;p['masks'][1][edge]=0;w[1][edge]=0
    for action in p['dynamic']['actions']:
        if any(k in action['reads'] for k in [4+edge,8+edge,12+edge]):action['mask']=[1,edge]
    out=NativeLIFTrainer(p,runner=RUNNER,weights=w).step(x[None],[0])
    assert out['final_state'][0][4+edge]==0 and out['gradients'][1][edge]==0
    assert out['state']['weights'][1][edge]==0
    assert out['final_state'][0][8+edge]==p['dynamic']['initial'][8+edge]


def test_dynamic_transform_aliases_and_temporaries():
    compiled=compile_dynamic_transform('tmp=pre+2\npost=tmp\npre*=3',states={'pre':0,'post':0})
    assert compiled['writes']==[0]
    p,w,x=model();p['dynamic']['program_sets'][0]=compiled['programs']
    # Only first voltage update changes: aliased names must see each other's writes.
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).evaluate(np.zeros((1,1,2)),[0])
    assert result['final_state'][0][0]==pytest.approx((.2+2)*3-1)


def test_dynamic_gpu_is_not_silently_run_on_cpu(tmp_path):
    import json,subprocess
    p,w,x=model();p['backend']='cuda';t=NativeLIFTrainer(model()[0],runner=RUNNER,weights=w)
    req=tmp_path/'req';out=tmp_path/'out';req.write_text(json.dumps(dict(plan=p,state=t.state,operation='evaluate',inputs=x[None].tolist(),labels=[0],initial=None)))
    run=subprocess.run([str(RUNNER),str(req),str(out)],capture_output=True,text=True,timeout=30)
    assert run.returncode and 'refusing CPU fallback' in run.stderr and not out.exists()


def noisy_model(detach=True,window=None):
    from brian2_rust.training_equations import NormalNoise
    p,w,x=model(detach,window);w.append([.07]);p['projections'].append(neuron_parameter_bank(1));p['masks'].append([1.]);p['trainable'].append(True)
    p['noise_streams']=[0,0]
    compiled=compile_dynamic_transform('pre+=dt*(-pre/taup)+sigma*sqrt(dt/.001)*eta\npost+=dt*(-post/taum)',
                states={'pre':0,'post':1},parameters=dict(dt=.0002,taup=(2,0),taum=(2,1),sigma=(3,0),eta=NormalNoise(0)))
    for edge in range(4):
        p['dynamic']['program_sets'][4+edge]=compiled['programs']
        p['dynamic']['actions'][4+edge].update(noise_domain=2,noise_entity=edge,noise_streams=1)
    return p,w,x


@pytest.mark.parametrize('detach,window',[(False,None),(True,None),(False,3),(True,3)])
def test_stochastic_dynamic_pathwise_vjp(detach,window):
    p,w,x=noisy_model(detach,window);sequence=11;start_tick=2
    out=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None],[0],noise_sequence=sequence,start_tick=start_tick)
    def run(w,anchors=None):return oracle(p,w,x,anchors=anchors,sequence=sequence,start_tick=start_tick)
    loss,spikes,live,anchors=run(w)
    np.testing.assert_allclose(out['final_state'][0],live,rtol=6e-14,atol=6e-14);np.testing.assert_array_equal(out['spikes'][0],spikes)
    assert out['loss']==pytest.approx(loss,abs=2e-14)
    for bank,row in enumerate(w):
        for i,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);a=copy.deepcopy(w);c=copy.deepcopy(w);a[bank][i]+=eps;c[bank][i]-=eps
            fd=(run(a,anchors)[0]-run(c,anchors)[0])/(2*eps)
            assert out['gradients'][bank][i]==pytest.approx(fd,rel=3e-4,abs=4e-6)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_stochastic_dynamic_partition_and_split(ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    p,w,x=noisy_model();p['trainable']=[False]*len(w)
    a=NativeLIFTrainer(p,runner=RUNNER,weights=w).step(x[None],[0],noise_sequence=7)
    if ranks:p['mpi_ranks']=ranks
    t=NativeLIFTrainer(p,runner=RUNNER,weights=w);first=t.step(x[None,:5],[0],noise_sequence=7);last=t.step(x[None,5:],[0],initial='carry')
    assert first['spikes'][0]+last['spikes'][0]==a['spikes'][0]
    np.testing.assert_allclose(last['final_state'],a['final_state'],rtol=3e-12,atol=3e-12)
    assert last['noise_sequence']==7 and t.next_noise_sequence==8
