"""Native run-boundary delay changes against actual Brian queues."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_delays import model, cython_cache


def snapshot(trainer):
    return copy.deepcopy((trainer.plan,trainer.state,trainer.neuron_state,trainer.elapsed_ticks,
                          trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence,trainer.last_result))


def change(step,static,syn):
    patterns=[([0,.6,0,.2],[.2,0,.8,0],[0,.2,0,.6]),
              ([.8,0,.2,0],[0,.6,0,.4],[.4,0,.2,0]),
              ([0,0,0,0],[0,0,0,0],[0,0,0,0])]
    paths=[static.pre,syn.pre,syn.post]
    return {p.name:np.asarray(d)*.001 for p,d in zip(paths,patterns[step%3])}


@pytest.mark.parametrize('event_driven',[False,True])
@pytest.mark.parametrize('post_first',[False,True])
@pytest.mark.parametrize('warmup',[0.,.8])
def test_repeated_delay_updates_match_actual_brian(event_driven,post_first,warmup):
    net,inp,layers,static,syn,x,bundle=model(event_driven,warmup,post_first=post_first,order_sensitive=True)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    monitors=[b.SpikeMonitor(g) for g in layers];net.add(*monitors)
    actual=[];cursor=0;paths={p.name:p for obj in (static,syn) for p in obj._pathways}
    for phase,length in enumerate([2,1,3,len(x)-6]):
        if phase:
            values=change(phase-1,static,syn);before=snapshot(trainer)
            trainer.update_delays(values)
            assert trainer.state==before[1]
            assert (trainer.elapsed_ticks,trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence)==before[3:7]
            for name,seconds in values.items():paths[name].delay=seconds*b.second
        result=trainer.step(x[None,cursor:cursor+length],[0],initial='carry' if phase else None)
        actual.extend(result['spikes'][0]);net.run(length*.2*b.ms,namespace={});cursor+=length
        np.testing.assert_allclose(result['final_membrane'][0],np.r_[layers[0].v[:],layers[1].v[:]],rtol=4e-13,atol=6e-14)
        for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
            np.testing.assert_allclose(np.array(result['final_state'])[0,indices],syn.variables[name].get_value(),rtol=4e-13,atol=6e-14)
    expected=np.zeros((len(x),4))
    for l,m in enumerate(monitors):expected[np.rint((m.t-warmup*b.ms)/(.2*b.ms)).astype(int),2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(actual,expected)
    for path in paths.values():assert path.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('window',[None,2])
def test_updated_delay_device_gradient_batch_and_checkpoint(engine,ranks,window,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    net,inp,layers,static,syn,x,bundle=model(order_sensitive=True)
    bundle.plan.update(backend=engine,mpi_ranks=ranks,tbptt_window=window)
    # Preserve runtime plasticity while suppressing optimizer steps during the
    # prefix. Different batch histories must migrate independently.
    bundle.plan['trainable']=[False]*len(bundle.weights)
    xx=np.stack([x,x.copy()]);xx[1,0]=0;xx[1,1]=1
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(xx[:,:2],[0,1]);trainer.update_delays(change(0,static,syn))
    p=copy.deepcopy(trainer.plan);p['backend']='cpu';p['mpi_ranks']=None
    expected=NativeLIFTrainer(p,runner=RUNNER,weights=trainer.state['weights']).gradients(xx[:,2:],[0,1],initial=trainer.neuron_state,start_tick=trainer.clock_tick)
    actual=trainer.gradients(xx[:,2:],[0,1],initial='carry')
    for key in ('final_state','spikes','initial_state_gradients','logits'):
        np.testing.assert_allclose(actual[key],expected[key],rtol=4e-5,atol=6e-6)
    for a,c in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,c,rtol=5e-4,atol=6e-6)
    checkpoint=tmp_path/'delays.json';trainer.store(checkpoint)
    restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(checkpoint)
    tail=restored.evaluate(xx[:,2:],[0,1],initial='carry')
    np.testing.assert_allclose(tail['final_state'],actual['final_state'],rtol=4e-5,atol=6e-6)


@pytest.mark.parametrize('bad',['unknown','negative','nonfinite','shape','huge','budget','units'])
def test_invalid_updates_are_atomic(bad):
    *_,static,syn,x,bundle=model();bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(x[None,:2],[0]);before=snapshot(trainer)
    values={static.pre.name:[.0002]*4}
    if bad=='unknown':values={'missing':[0.]}
    if bad=='negative':values[static.pre.name][0]=-.1
    if bad=='nonfinite':values[static.pre.name][0]=float('nan')
    if bad=='shape':values[static.pre.name]=[0.,0.]
    if bad=='huge':values[static.pre.name]=[1e10]
    if bad=='budget':values[static.pre.name]=[20.]
    if bad=='units':values[static.pre.name]=1*b.volt
    with pytest.raises(ValueError):trainer.update_delays(values)
    assert snapshot(trainer)==before


def test_repeated_empty_updates_reuse_storage_and_preserve_model_indices():
    *_,static,syn,x,bundle=model();bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(np.zeros((1,12,2)),[0])
    # Drain all existing events; zero new delays then longer delays must not
    # leak inaccessible state/actions on every configuration boundary.
    names=[p['name'] for p in trainer.plan['dynamic']['delay_layout']['paths']]
    trainer.update_delays({name:0. for name in names});widths=[]
    for _ in range(5):
        trainer.update_delays({name:.001 for name in names});widths.append(len(trainer.neuron_state[0]))
        trainer.update_delays({name:0. for name in names})
    assert len(set(widths))==1
    trainer.evaluate(np.zeros((1,4,2)),[0],initial='carry')


def oracle_bundle(bundle,trainer):
    reference=copy.deepcopy(bundle);reference.plan=copy.deepcopy(trainer.plan)
    reference.plan['clock']['origin']+=trainer.clock_tick*reference.plan['clock']['dt']
    reference.initial_state=copy.deepcopy(trainer.plan['dynamic']['initial'])
    # Expose only edge IDs, queue histories and their known ordering to the
    # independent STDP oracle; it does not evaluate native actions or SSA.
    reference.provenance['delay_queues']={p['name']:dict(
        pending=[dict(edge=e['edge'],states=e['states']) for e in p['pending']],
        new=[dict(edge=i,states=e['states'],delay=len(e['states'])) for i,e in sorted(enumerate(p['edges']),key=lambda z:z[1]['event'])])
        for p in trainer.plan['dynamic']['delay_layout']['paths']}
    return reference


@pytest.mark.parametrize('event_driven',[False,True])
@pytest.mark.parametrize('window',[None,2])
def test_updated_arrivals_have_independent_weight_and_history_vjps(engine,event_driven,window):
    from test_training_delays import oracle
    *_,static,syn,x,bundle=model(event_driven,post_first=True,order_sensitive=True)
    bundle.plan.update(backend=engine,tbptt_window=window);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(x[None,:2],[0]);trainer.update_delays(change(0,static,syn))
    reference=oracle_bundle(bundle,trainer);weights=trainer.state['weights'];initial=np.asarray(trainer.neuron_state[0])
    tail=x[2:];kw=dict(post_first=True,order_sensitive=True)
    loss,spikes,live,anchors=oracle(reference,weights,tail,initial=initial,**kw)
    actual=trainer.gradients(tail[None],[0],initial='carry')
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    np.testing.assert_allclose(actual['final_state'][0],live,rtol=5e-5,atol=5e-6)
    assert actual['loss']==pytest.approx(loss,abs=5e-6)
    for bank,row in enumerate(weights):
        for j,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);a=copy.deepcopy(weights);c=copy.deepcopy(weights);a[bank][j]+=eps;c[bank][j]-=eps
            fd=(oracle(reference,a,tail,initial=initial,anchors=anchors,**kw)[0]-oracle(reference,c,tail,initial=initial,anchors=anchors,**kw)[0])/(2*eps)
            assert actual['gradients'][bank][j]==pytest.approx(fd,rel=8e-4,abs=8e-6)
    for j in range(len(initial)):
        if reference.plan['dynamic']['detached'][j]:continue
        a=initial.copy();c=initial.copy();a[j]+=1e-6;c[j]-=1e-6
        fd=(oracle(reference,weights,tail,initial=a,anchors=anchors,**kw)[0]-oracle(reference,weights,tail,initial=c,anchors=anchors,**kw)[0])/2e-6
        assert actual['initial_state_gradients'][0][j]==pytest.approx(fd,rel=8e-4,abs=8e-6)


@pytest.mark.parametrize('ranks',[None,2])
def test_noisy_delay_update_prune_regrow_and_cursors(engine,ranks,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    from test_training_brian_dynamic import network
    from brian2_rust import lower_brian_dynamic_training
    net,inp,layers,static,syn,_,_,x=network(False,True)
    static.pre.delay=.4*b.ms;syn.pre.delay=.6*b.ms;syn.post.delay=.2*b.ms
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,backend=engine,mpi_ranks=ranks)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(x[None,:2],[0],noise_sequence=71);before=snapshot(trainer)
    trainer.update_delays(change(0,static,syn))
    assert snapshot(trainer)[3:7]==before[3:7] and trainer.state==before[1]
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==syn.name and e['variables']==['w'])
    masks=copy.deepcopy(trainer.plan['masks']);masks[bank][1]=0;trainer.update_mask(masks)
    for path in trainer.plan['dynamic']['delay_layout']['paths']:
        if path['name'].startswith(syn.name+'_'):
            for states in [path['edges'][1]['states']]+[e['states'] for e in path['pending'] if e['edge']==1]:
                np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[:,states],0)
    masks[bank][1]=1;trainer.update_mask(masks,growth_weight=.43)
    ref=copy.deepcopy(trainer.plan);ref['backend']='cpu';ref['mpi_ranks']=None
    expected=NativeLIFTrainer(ref,runner=RUNNER,weights=trainer.state['weights']).gradients(x[None,2:],[0],initial=trainer.neuron_state,start_tick=trainer.clock_tick,noise_sequence=71)
    checkpoint=tmp_path/'noise-delays.json';trainer.store(checkpoint)
    fresh=NativeLIFTrainer(trainer.plan,runner=RUNNER);fresh.restore(checkpoint)
    actual=fresh.gradients(x[None,2:],[0],initial='carry')
    for key in ('final_state','spikes','initial_state_gradients'):
        np.testing.assert_allclose(actual[key],expected[key],rtol=6e-5,atol=8e-6)
    for a,c in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,c,rtol=8e-4,atol=1e-5)


@pytest.mark.parametrize('issue',['range','order','alias','shift','gate_read','free','foreign'])
def test_malformed_delay_layout_cannot_migrate_model_state(issue):
    *_,x,bundle=model();d=bundle.plan['dynamic'];path=d['delay_layout']['paths'][0]
    edge=next(e for e in path['edges'] if e['states']);cell=edge['states'][0]
    if issue=='range':path['start']+=1
    elif issue=='order':edge['event']=path['start']
    elif issue=='alias':edge['states'][0]=d['voltage'][0]
    elif issue=='shift':d['actions'][path['end']-1]['writes']=[d['voltage'][0]]
    elif issue=='gate_read':
        action=d['actions'][edge['event']];d['program_sets'][action['program_set']][0]=[dict(op='state',index=len(action['reads'])-1)]
    elif issue=='free':d['delay_layout']['free_cells']=[cell]
    elif issue=='foreign':d['actions'][-1]['reads'].append(cell)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);before=snapshot(trainer)
    with pytest.raises(ValueError):trainer.step(x[None,:1],[0])
    assert snapshot(trainer)==before


@pytest.mark.parametrize('ticks',[.5,1.5,65,129])
def test_updated_rounding_and_long_history_chunks(engine,ticks):
    from test_training_delays import oracle
    *_,static,syn,x,bundle=model(post_first=True,order_sensitive=True)
    bundle.plan['backend']=engine;bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(x[None,:2],[0])
    seconds=ticks*.0002;trainer.update_delays({static.pre.name:seconds*b.second})
    path=next(p for p in trainer.plan['dynamic']['delay_layout']['paths'] if p['name']==static.pre.name)
    assert all(len(e['states'])==int(seconds/.0002+.5) for e in path['edges'])
    xx=np.r_[x[2:],np.zeros((int(ticks)+2,2))];reference=oracle_bundle(bundle,trainer)
    _,spikes,state,_=oracle(reference,trainer.state['weights'],xx,initial=trainer.neuron_state[0],post_first=True,order_sensitive=True)
    actual=trainer.evaluate(xx[None],[0],initial='carry')
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    np.testing.assert_allclose(actual['final_state'][0],state,rtol=5e-5,atol=5e-6)


def test_delay_update_preserves_typed_state_and_rejects_stale_checkpoint(engine,tmp_path):
    from test_training_discrete_frontend import network
    *_,bundle,x=network(method='heun',noisy=True,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(x[:,:1],[0],noise_sequence=71)
    indices=trainer.plan['dynamic']['integer_states'];before=np.asarray(trainer.neuron_state)[:,indices].copy()
    checkpoint=tmp_path/'before-update.json';trainer.store(checkpoint)
    name=trainer.plan['dynamic']['delay_layout']['paths'][0]['name'];trainer.update_delays({name:.001})
    np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[:,indices],before)
    committed=snapshot(trainer)
    with pytest.raises(ValueError,match='plan/runtime mismatch'):trainer.restore(checkpoint)
    assert snapshot(trainer)==committed
    trainer.evaluate(x[:,1:],[0],initial='carry')
