"""Delayed dynamic event paths: Brian queues, pending snapshots and BPTT."""
import copy
import os
import tempfile
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training,TrainingConversionError
from brian2_rust.training_queue import pending_events
from test_native_training import RUNNER
from test_training_brian_dynamic import network


@pytest.fixture(scope='module',autouse=True)
def cython_cache():
    old=b.prefs.codegen.runtime.cython.cache_dir
    with tempfile.TemporaryDirectory(prefix='b2-delay-cython-') as directory:
        b.prefs.codegen.runtime.cython.cache_dir=directory
        try:yield
        finally:b.prefs.codegen.runtime.cython.cache_dir=old


def model(event_driven=True,warmup=0.,changed=False,new_dt=None,post_first=False,order_sensitive=False):
    net,inp,layers,static,syn,_,_,x=network(event_driven)
    static.pre.delay=[0,.2,.4,.6]*b.ms;syn.pre.delay=[.6,.2,.4,0]*b.ms;syn.post.delay=[.4,0,.6,.2]*b.ms
    if post_first:syn.post.order=-2
    if order_sensitive:
        static.pre.code='v_post=.8*v_post+w'
        syn.pre.code='v_post=.7*v_post+w\napre+=Ap\nw=clip(w+apost,0,1.2)'
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    for obj in (static,syn):
        for path in obj._pathways:path.codeobj_class=CythonCodeObject
    if warmup:net.run(warmup*b.ms,namespace={})
    if changed:
        static.pre.delay=[.2,0,.2,0]*b.ms;syn.pre.delay=[0,.4,0,.2]*b.ms;syn.post.delay=[.2,.2,0,0]*b.ms
    if new_dt is not None:
        assert warmup
        for obj in (inp,*layers,syn):obj.clock.dt=new_dt*b.ms
        inp.set_spikes([],[]*b.second);x=np.zeros((14,2))
    else:x=x[round(float(inp.clock.variables['t'].get_value()[0])/.0002):]
    before=[copy.deepcopy(p.queue._full_state()) for obj in (static,syn) for p in obj._pathways]
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,
        trainable_synapse_parameters={syn.name:['w','taup','taum','Ap','Am']},learning_rate=1e-7)
    after=[p.queue._full_state() for obj in (static,syn) for p in obj._pathways];assert after==before
    return net,inp,layers,static,syn,x,bundle


def compare_brian(net,layers,syn,x,bundle):
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in layers];net.add(*monitors)
    origin=bundle.plan['clock']['origin'];dt=bundle.plan['clock']['dt']
    net.run((origin+len(x)*dt-float(net.t))*b.second,namespace={})
    spikes=np.zeros((len(x),4))
    for layer,monitor in enumerate(monitors):
        ticks=np.rint((np.asarray(monitor.t/b.second)-origin)/dt).astype(int)
        spikes[ticks,2*layer+np.asarray(monitor.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_membrane'][0],np.r_[layers[0].v[:],layers[1].v[:]],rtol=3e-13,atol=5e-14)
    for name,ids in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(np.asarray(result['final_state'])[0,ids],syn.variables[name].get_value(),rtol=3e-13,atol=5e-14)
    return result


@pytest.mark.parametrize('event_driven',[False,True])
@pytest.mark.parametrize('warmup',[0.,.7,.8])
@pytest.mark.parametrize('post_first',[False,True])
@pytest.mark.parametrize('order_sensitive',[False,True])
def test_delay_arrival_and_pending_events_match_brian(event_driven,warmup,post_first,order_sensitive):
    net,inp,layers,static,syn,x,bundle=model(event_driven,warmup,post_first=post_first,order_sensitive=order_sensitive)
    if warmup:assert sum(len(layout['pending']) for layout in bundle.provenance['delay_queues'].values())>0
    compare_brian(net,layers,syn,x,bundle)


@pytest.mark.parametrize('new_dt',[None,.1,.4])
@pytest.mark.parametrize('changed',[False,True])
@pytest.mark.parametrize('post_first',[False,True])
def test_snapshot_preserves_old_arrival_order_after_delay_or_dt_change(new_dt,changed,post_first):
    net,inp,layers,static,syn,x,bundle=model(warmup=.8,changed=changed,new_dt=new_dt,post_first=post_first,order_sensitive=True)
    compare_brian(net,layers,syn,x,bundle)


@pytest.mark.parametrize('ratio',[.5,1,2,3])
def test_queue_snapshot_requantization_and_collision_order(ratio):
    from brian2.synapses.cythonspikequeue import SpikeQueue
    q=SpikeQueue(0,2);sources=np.array([0,1],dtype=np.int32);delays=np.array([.0002,.0008]);dt=.0002
    q.prepare(delays,dt,sources)
    for spikes in ([0,1],[1],[0,1]):q.push(np.array(spikes,dtype=np.int32));q.advance()
    before=q._full_state();actual=pending_events(q,dt*ratio,2,max_bytes=1_000_000);assert q._full_state()==before
    q.prepare(delays,dt*ratio,sources);offset,bins=q._full_state();expected=bins[offset:]+bins[:offset]
    # prepare also extends to the current maximum delay; trailing empties carry no events.
    while actual and not actual[-1]:actual.pop()
    while expected and not expected[-1]:expected.pop()
    assert actual==expected


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('warmup',[0.,.8])
def test_delay_carry_checkpoint_and_mpi(ranks,warmup,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(warmup=warmup,changed=True,order_sensitive=True)
    bundle.plan['trainable']=[False]*len(bundle.weights);serial=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    whole=serial.gradients(x[None],[0]);plan=copy.deepcopy(bundle.plan)
    if ranks:plan['mpi_ranks']=ranks
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights);actual=trainer.gradients(x[None],[0])
    for key in ('spikes','final_state','initial_state_gradients'):
        np.testing.assert_allclose(actual[key],whole[key],rtol=2e-13,atol=2e-13)
    for a,c in zip(actual['gradients'],whole['gradients']):np.testing.assert_allclose(a,c,rtol=2e-13,atol=2e-13)
    trainer.execute(x[None,:3],[0]);filename=tmp_path/'delay.json';trainer.store(filename)
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights);trainer.restore(filename)
    tail=trainer.execute(x[None,3:],[0],initial='carry')
    np.testing.assert_array_equal(tail['spikes'],np.asarray(whole['spikes'])[:,3:])
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('issue',['negative','infinite','huge','binary_initial','binary_update','gate_alias','gate_not_declared','external_and_state'])
def test_delay_validation_is_transactional(issue):
    if issue in ('negative','infinite','huge'):
        net,inp,layers,static,syn,*_=network(True);syn.pre.delay=dict(negative=-.2,infinite=float('inf'),huge=1e9)[issue]*b.ms
        with pytest.raises(TrainingConversionError,match='delay'):lower_brian_dynamic_training(net,input_group=inp,layers=layers)
        return
    *_,x,bundle=model();plan=copy.deepcopy(bundle.plan);state=np.array(bundle.initial_state)[None]
    gate=next(a for a in plan['dynamic']['actions'] if a['trigger'] and a['trigger'].get('state'))
    if issue=='binary_initial':state[0,gate['trigger']['index']]=.3
    elif issue=='gate_alias':gate['reads'].remove(gate['trigger']['index'])
    elif issue=='gate_not_declared':plan['dynamic']['binary_states'].remove(gate['trigger']['index'])
    elif issue=='external_and_state':gate['trigger']['external']=True
    elif issue=='binary_update':
        k=gate['trigger']['index'];ps=len(plan['dynamic']['program_sets'])
        plan['dynamic']['program_sets'].append([[dict(op='constant',value=.3)]])
        plan['dynamic']['actions'].append(dict(owner=0,reads=[k],writes=[k],program_set=ps,threshold=None,trigger=None,mask=gate['mask']))
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='binary|trigger|context|program'):
        trainer.execute(x[None],[0],initial=state)
    assert trainer.state==before and trainer.clock_tick==0


def oracle(bundle,weights,x,*,initial=None,anchors=None,post_first=False,order_sensitive=False):
    """Independent delayed STDP equations; no SSA/program/action interpretation.

    Physical/event-history amplitudes are locally relaxed for finite differences.
    Arrival ordering, timestamps and baseline hard events remain detached.
    """
    p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for k,binding in enumerate(p['dynamic']['initial_parameters']):
            if binding is not None:z[k]=weights[binding[0]][binding[1]]
    layout=bundle.provenance['dynamic_state_layout']['front_stdp'];wi=np.array(layout['w']);pi=np.array(layout['apre']);mi=np.array(layout['apost'])
    li=np.array(layout['lastupdate']) if 'lastupdate' in layout else None
    bindings={(a['object'],a['variables'][0]):a['bank'] for a in bundle.provenance['bindings']}
    static=weights[bindings['front_static','w']]
    static_mask=p['masks'][bindings['front_static','w']];synaptic_mask=np.array(p['masks'][bindings['front_stdp','w']])
    taup,taum,Ap,Am=[weights[bindings['front_stdp',name]][0] for name in ('taup','taum','Ap','Am')]
    queues=bundle.provenance['delay_queues'];dt=p['clock']['dt'];starts=[];pres=[];spikes=[]
    order=['front_stdp_post','front_static_pre','front_stdp_pre'] if post_first else ['front_static_pre','front_stdp_pre','front_stdp_post']
    for t,inp in enumerate(x):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors[2][t].copy()
        starts.append(z.copy());z[:4]*=.85
        if li is None:z[pi]*=np.exp(-dt/taup*synaptic_mask);z[mi]*=np.exp(-dt/taum*synaptic_mask)
        pre=z[:4].copy();s=(pre>1).astype(float);hard=s.copy()
        if anchors is not None:
            hard=anchors[1][t];phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][t]-1))**2
            s=hard+phi*(pre-anchors[0][t])
        stamp=p['clock']['origin']+t*dt
        for path in order:
            queue=queues[path];post=path.endswith('_post');external=path=='front_static_pre'
            for entry in queue['pending']+queue['new']:
                edge=entry['edge']
                if not (static_mask if external else synaptic_mask)[edge]:continue
                i,j=divmod(edge,2);fifo=entry['states'];source=2+j if post else i
                if fifo:
                    gate=z[fifo[0]];hard_gate=gate if anchors is None else anchors[2][t,fifo[0]]
                else:
                    gate=inp[i] if external else s[source];hard_gate=inp[i] if external else hard[source]
                if external:
                    z[j]+=gate*(static[edge]-(.2*z[j] if order_sensitive else 0));continue
                ids=[wi[edge],pi[edge],mi[edge]] if post else [2+j,wi[edge],pi[edge],mi[edge]]
                before=z[ids].copy();after=before.copy()
                if li is not None:
                    delta=stamp-z[li[edge]];after[-2]*=np.exp(-delta/taup);after[-1]*=np.exp(-delta/taum)
                if post:
                    after[2]+=Am;after[0]=np.clip(after[0]+after[1],0,1.2)
                else:
                    after[0]=(0.7 if order_sensitive else 1.)*after[0]+after[1]
                    after[2]+=Ap;after[1]=np.clip(after[1]+after[3],0,1.2)
                z[ids]=before+gate*(after-before)
                if li is not None and hard_gate:z[li[edge]]=stamp
            for entry in queue['pending']+queue['new']:
                fifo=entry['states']
                if not fifo or not (static_mask if external else synaptic_mask)[entry['edge']]:continue
                z[fifo[:-1]]=z[fifo[1:]].copy();z[fifo[-1]]=0
                if 'delay' in entry:
                    i,j=divmod(entry['edge'],2);z[fifo[-1]]=inp[i] if external else s[2+j if post else i]
        z[:4]-=hard if p['detach_reset'] else s
        pres.append(pre);spikes.append(s)
    spikes=np.array(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale'];loss=logits.max()+np.log(np.exp(logits-logits.max()).sum())-logits[0]
    return loss,spikes,z,(np.array(pres),spikes,np.array(starts))


@pytest.mark.parametrize('warmup',[0.,.8])
@pytest.mark.parametrize('event_driven',[False,True])
@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
@pytest.mark.parametrize('explicit_initial',[False,True])
def test_delayed_and_pending_event_independent_vjp(warmup,event_driven,detach,window,explicit_initial):
    *_,x,bundle=model(event_driven,warmup,changed=True,post_first=True,order_sensitive=True)
    p=bundle.plan;p['detach_reset']=detach;p['tbptt_window']=window
    # Reset actions are the final four, after all path queues have advanced.
    for action in p['dynamic']['actions'][-4:]:action['detach_trigger']=detach
    weights=bundle.weights;initial=np.array(bundle.initial_state) if explicit_initial else None
    result=NativeLIFTrainer(p,runner=RUNNER,weights=weights).gradients(x[None],[0],initial=None if initial is None else initial[None])
    kw=dict(post_first=True,order_sensitive=True)
    loss,spikes,state,anchors=oracle(bundle,weights,x,initial=initial,**kw)
    assert result['loss']==pytest.approx(loss,abs=2e-14)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],state,rtol=8e-14,atol=8e-14)
    for bank,row in enumerate(weights):
        for j,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);a=copy.deepcopy(weights);c=copy.deepcopy(weights);a[bank][j]+=eps;c[bank][j]-=eps
            fd=(oracle(bundle,a,x,initial=initial,anchors=anchors,**kw)[0]-oracle(bundle,c,x,initial=initial,anchors=anchors,**kw)[0])/(2*eps)
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=5e-4,abs=3e-6)
    if explicit_initial:
        assert np.max(abs(np.asarray(result['initial_state_gradients'])[0,p['dynamic']['binary_states']]))>1e-7
        for j in range(len(initial)):
            if p['dynamic']['detached'][j]:
                assert result['initial_state_gradients'][0][j]==0;continue
            a=initial.copy();c=initial.copy();a[j]+=1e-6;c[j]-=1e-6
            fd=(oracle(bundle,weights,x,initial=a,anchors=anchors,**kw)[0]-oracle(bundle,weights,x,initial=c,anchors=anchors,**kw)[0])/2e-6
            assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=5e-4,abs=3e-7)


@pytest.mark.parametrize('ticks',[.4999999,.5,1.4999999,1.5,2.5,65,129])
def test_delay_rounding_and_long_history_chunks(ticks):
    net,inp,layers,static,syn,_,_,x=network(True);syn.pre.delay=ticks*.2*b.ms
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
    declared=bundle.provenance['delay_queues'][syn.pre.name]['new']
    expected=int(float(syn.pre.delay[0])/float(inp.clock.dt_)+.5)
    assert all(row['delay']==expected for row in declared)
    if expected>=65:
        # Exercise transfer across both 63-cell chunk boundaries and actual delivery.
        x=np.r_[x,np.zeros((expected+3,2))]
    compare_brian(net,layers,syn,x,bundle)


def test_snapshot_queue_and_history_budgets_are_checked_before_expansion():
    net,inp,layers,static,syn,_,_,x=network(True);syn.pre.delay=100000*.2*b.ms
    with pytest.raises(TrainingConversionError,match='budget'):
        lower_brian_dynamic_training(net,input_group=inp,layers=layers,max_tape_bytes=100000)


def test_old_and_new_event_for_same_edge_in_one_tick_are_both_delivered():
    net,inp,layers,static,syn,_,_,x=network(True);x=np.ones_like(x)
    ticks,ids=np.nonzero(x);inp.set_spikes(ids,ticks*.2*b.ms)
    static.pre.delay=.6*b.ms;static.pre.code='v_post=.8*v_post+w'
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    static.pre.codeobj_class=CythonCodeObject
    net.run(.8*b.ms,namespace={});static.pre.delay=0*b.ms
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
    pending=bundle.provenance['delay_queues'][static.pre.name]['pending']
    assert [row['edge'] for row in pending if row['remaining']==0]==[0,1,2,3]
    assert all(row['delay']==0 for row in bundle.provenance['delay_queues'][static.pre.name]['new'])
    compare_brian(net,layers,syn,x[4:],bundle)


def test_brian_stored_queue_snapshot_replays_after_restore():
    net,inp,layers,static,syn,x,bundle=model(warmup=.8,changed=True,order_sensitive=True)
    net.store('pending');net.run(.4*b.ms,namespace={});net.restore('pending')
    restored=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
    assert restored.provenance['delay_queues']==bundle.provenance['delay_queues']
    compare_brian(net,layers,syn,x,restored)


def test_scalar_delay_storage_is_broadcast_per_edge():
    net,inp,layers,old,syn,_,_,x=network(True);net.remove(old)
    static=b.Synapses(inp,layers[0],'w:1',on_pre='v_post+=w',delay=.5*b.ms);static.connect();static.w=[.85,.95,1.05,.75];net.add(static)
    assert np.asarray(static.pre.delay[:]).size==1
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
    assert [entry['delay'] for entry in bundle.provenance['delay_queues'][static.pre.name]['new']]==[3]*4
    compare_brian(net,layers,syn,x,bundle)


@pytest.mark.parametrize('clock_dt',[.3,.99999])
def test_asynchronous_time_and_pending_delayed_event_decay(clock_dt):
    from test_training_dynamic_clocks import model as clock_model
    net,inp,layers,syn,x,dt,_=clock_model(clock_dt,third=True)
    syn.pre.delay=[.6,.2,.4,0]*b.ms;syn.post.delay=[.4,0,.6,.2]*b.ms
    net.run(.3*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
    compare_brian(net,layers,syn,x[2:],bundle)


def test_delayed_plan_is_plain_serializable_data_without_builder_state():
    *_,bundle=model(warmup=.8)
    assert type(bundle.plan['dynamic']['actions']) is list
    assert copy.deepcopy(bundle.plan)==bundle.plan
