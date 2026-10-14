"""One native draw/rate/VJP across owners, actions, visits and checkpoints."""
import copy
import json
import math
import subprocess
import sys

import numpy as np
import pytest
from brian2_rust.training import NativeLIFTrainer
from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
from brian2_rust.training_equations import PoissonNoise
from test_native_training import RUNNER
from test_training_poisson_ssa import model as original_model,draw
from test_training_poisson_zero import hard_loss
from test_training_poisson_zero_vjp import mpi


def model(window=None,ranks=None,first='k=draw(scale*r)\nv=v+amp*k\nr=.8*r+.2*offset',
          second='k=draw(scale*r)\nv=1.-amp*k\nr=.8*r+.2*offset'):
    p,w=original_model(window,ranks);w[0][1]=1.
    p['dynamic']['initial'][0]=0.
    params={'draw':PoissonNoise(0),'scale':(0,0),'amp':(0,1),'offset':(0,2)}
    programs=[]
    for index,code in enumerate((first,second)):
        tr=compile_dynamic_transform(code,states={'v':0,'k':1,'r':2},state_types={1:'integer'},parameters=params)
        programs.append(tr['programs'])
        p['dynamic']['actions'][index]=dynamic_action(tr,[index,2+index,4],owner=1+index,program_set=index,
            noise_domain=71,noise_entity=0,noise_streams=1)
    p['dynamic']['program_sets']=programs
    return p,w


def oracle(p,w,initial,labels,length,anchors=None,sequence=9):
    live=np.array(initial,float).copy();batch=len(live);scale,amp,offset=w[0]
    before=[];counts=[];margins=[];spikes=[];logp=np.zeros(batch)
    for tick in range(length):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:live=anchors['before'][tick].copy()
        before.append(live.copy());rate=scale*live[:,4]
        k=np.array([draw(rate[b],p['seed'],sequence,b,0,tick) for b in range(batch)]) if anchors is None else anchors['counts'][tick]
        counts.append(k.copy());live[:,2]=k;live[:,3]=k
        logp += np.array([c*math.log(r)-r-math.lgamma(c+1) for c,r in zip(k,rate)])
        live[:,0]+=amp*k;live[:,4]=.8*live[:,4]+.2*offset
        live[:,1]=1.-amp*k;live[:,4]=.8*live[:,4]+.2*offset
        margin=live[:,:2]-.8;hard=(margin>0).astype(float)
        if anchors is not None:
            old=anchors['margins'][tick];hard=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());spikes.append(hard)
    spikes=np.stack(spikes,axis=1);logits=spikes.mean(1)*p['logit_scale']
    maximum=logits.max(1);loss=maximum+np.log(np.exp(logits-maximum[:,None]).sum(1))-logits[np.arange(batch),labels]
    objective=loss.mean()
    if anchors is not None:objective+=(anchors['loss']*logp).mean()
    return objective,live,np.concatenate([np.zeros((batch,length,1)),spikes],axis=2),dict(before=before,counts=counts,margins=margins,loss=loss)


@pytest.mark.parametrize('window',[None,1,2])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_shared_draw_has_one_score_and_original_rate_context(window,batch,ranks):
    mpi(ranks);p,w=model(window,ranks);labels=np.arange(batch)%2;initial=np.array([p['dynamic']['initial']]*batch);initial[:,4]+=np.arange(batch)*.1
    r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((batch,4,1)),labels,initial=initial,noise_sequence=9)
    loss,live,spikes,anchors=oracle(p,w,initial,labels,4)
    np.testing.assert_array_equal(r['spikes'],spikes)
    np.testing.assert_allclose(r['final_state'],live,rtol=2e-14,atol=2e-14)
    assert r['loss']==pytest.approx(loss,abs=2e-14)
    assert len(r['poisson_state']['entries'])==batch*4
    for j in range(3):
        plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[0][j]+=1e-6;minus[0][j]-=1e-6
        fd=(oracle(p,plus,initial,labels,4,anchors)[0]-oracle(p,minus,initial,labels,4,anchors)[0])/2e-6
        assert r['gradients'][0][j]==pytest.approx(fd,rel=3e-5,abs=5e-6)
    for b in range(batch):
        for j in (0,1,4):
            plus=initial.copy();minus=initial.copy();plus[b,j]+=1e-6;minus[b,j]-=1e-6
            fd=(oracle(p,w,plus,labels,4,anchors)[0]-oracle(p,w,minus,labels,4,anchors)[0])/2e-6
            assert r['initial_state_gradients'][b][j]==pytest.approx(fd,rel=3e-5,abs=5e-6)


@pytest.mark.parametrize('window',[None,1,2])
@pytest.mark.parametrize('length',[1,4])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_zero_rate_forces_every_alias_and_scores_only_origin(window,length,ranks):
    mpi(ranks);p,w=model(window,ranks);w[0][0]=0.
    r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((1,length,1)),[0])
    rate=p['dynamic']['initial'][4];expected=0.
    for tick in range(length):
        alternate=hard_loss(p['logit_scale']*(1-tick)/length)
        expected+=rate*(alternate-hard_loss(-p['logit_scale']))
        rate=.64*rate+.36*w[0][2]
    assert r['gradients'][0][0]==pytest.approx(expected,rel=1e-13,abs=2e-14)
    assert r['gradients'][0][1]==0.
    np.testing.assert_array_equal(np.asarray(r['final_state'])[:,2:4],0.)
    assert len(r['poisson_state']['entries'])==length


@pytest.mark.parametrize('ranks',[None,2])
def test_cached_consumer_does_not_evaluate_overwritten_invalid_rate(ranks):
    mpi(ranks);p,w=model(ranks=ranks,first='k=draw(scale*r)\nv=amp*k\nr=-1.')
    r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)
    count=draw(w[0][0]*p['dynamic']['initial'][4],p['seed'],9,0,0,0)
    assert r['final_state'][0][2]==r['final_state'][0][3]==count
    assert r['poisson_state']['entries'][0]['rate']==w[0][0]*p['dynamic']['initial'][4]


@pytest.mark.parametrize('ranks',[None,2])
def test_first_actual_visited_branch_owns_the_draw(ranks):
    mpi(ranks);p,w=model(ranks=ranks,first='v=amp*draw(scale*r) if v>1 else v\nr=.8*r+.2*offset')
    r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)
    actual_rate=w[0][0]*(.8*p['dynamic']['initial'][4]+.2*w[0][2])
    assert r['poisson_state']['entries'][0]['rate']==actual_rate
    assert r['final_state'][0][3]==draw(actual_rate,p['seed'],9,0,0,0)
    expected=r['loss']*(r['final_state'][0][3]/actual_rate-1)*(.8*p['dynamic']['initial'][4]+.2*w[0][2])
    assert r['gradients'][0][0]==pytest.approx(expected,abs=2e-14)


@pytest.mark.parametrize('ranks',[None,2])
def test_pending_identity_checkpoint_reuses_old_draw_without_rescoring(ranks,tmp_path):
    mpi(ranks);p,w=model(ranks=ranks);p['trainable']=[False]
    for a in p['dynamic']['actions'][:2]:a.update(event_noise={'delay':0,'pending':73},trigger={'external':True,'index':0})
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);r=trainer.step(np.ones((1,1,1)),[0],noise_sequence=9)
    count=r['final_state'][0][2];assert len(r['poisson_state']['entries'])==1
    checkpoint=tmp_path/'pending';trainer.store(checkpoint)
    restored=NativeLIFTrainer(p,weights=w,runner=RUNNER);restored.restore(checkpoint)
    same=trainer.gradients(np.ones((1,2,1)),[0],initial='carry')
    result=restored.gradients(np.ones((1,2,1)),[0],initial='carry')
    assert same==result and result['final_state'][0][2]==result['final_state'][0][3]==count
    assert result['gradients'][0][0]==0.
    code='''import json,sys,numpy as np
from brian2_rust.training import NativeLIFTrainer
p=json.load(open(sys.argv[1]));t=NativeLIFTrainer(p,runner=sys.argv[4]);t.restore(sys.argv[2]);json.dump(t.gradients(np.ones((1,2,1)),[0],initial='carry'),open(sys.argv[3],'w'))'''
    plan=tmp_path/'plan';plan.write_text(json.dumps(p));out=tmp_path/'out'
    subprocess.run([sys.executable,'-c',code,str(plan),str(checkpoint),str(out),str(RUNNER)],check=True,timeout=90)
    assert json.loads(out.read_text())==result


@pytest.mark.parametrize('ranks',[None,2])
def test_draw_cache_budget_and_failed_alias_are_atomic(ranks):
    mpi(ranks);p,w=model(ranks=ranks);x=np.zeros((1,3,1))
    r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0])
    p['max_tape_bytes']=r['tape_bytes']-1;t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='budget'):t.step(x,[0])
    assert t.state==before and t.poisson_state is None and t.clock_tick==0


@pytest.mark.parametrize('malformation',['duplicate','count','rate','seed','sequence','missing'])
def test_checkpoint_draw_validation_is_atomic(malformation,tmp_path):
    p,w=model();t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(np.zeros((1,1,1)),[0],noise_sequence=9)
    path=tmp_path/'checkpoint';t.store(path);envelope=json.loads(path.read_text());payload=json.loads(envelope['payload'])
    cache=payload['poisson_state']
    if malformation=='duplicate':cache['entries']*=2
    elif malformation=='count':cache['entries'][0]['count']+=1
    elif malformation=='rate':cache['entries'][0]['rate']=-1.
    elif malformation=='missing':payload.pop('poisson_state')
    else:cache[malformation]+=1
    from brian2_rust.training import canonical_bytes
    import hashlib
    raw=canonical_bytes(payload);envelope['payload']=raw.decode();envelope['sha256']=hashlib.sha256(raw).hexdigest();path.write_text(json.dumps(envelope))
    restored=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(restored.state)
    with pytest.raises(ValueError,match='Poisson|poisson'):restored.restore(path)
    assert restored.state==before and restored.poisson_state is None and restored.neuron_state is None


def delayed_oracle(p,w,initial,labels,length,anchors=None,sequence=9):
    live=np.array(initial,float).copy();batch=len(live);scale,amp,offset=w[0]
    before=[];counts=[];margins=[];spikes=[];logp=np.zeros(batch);prior=None
    for tick in range(length):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:live=anchors['before'][tick].copy()
        before.append(live.copy());rate=scale*live[:,4]
        a=np.array([draw(rate[b],p['seed'],sequence,b,0,tick,(1,tick)) for b in range(batch)]) if anchors is None else anchors['counts'][tick][0]
        logp+=np.array([c*math.log(r)-r-math.lgamma(c+1) for c,r in zip(a,rate)])
        live[:,0]+=amp*a;live[:,2]=a;live[:,4]=.8*live[:,4]+.2*offset
        if tick==0:
            rate=scale*live[:,4]
            old=np.array([draw(rate[b],p['seed'],sequence,b,0,0,(1,2**64-1)) for b in range(batch)]) if anchors is None else anchors['counts'][tick][1]
            logp+=np.array([c*math.log(r)-r-math.lgamma(c+1) for c,r in zip(old,rate)])
        else:old=prior
        live[:,1]=1.-amp*old;live[:,3]=old;live[:,4]=.8*live[:,4]+.2*offset
        counts.append((a.copy(),old.copy()));prior=a.copy()
        margin=live[:,:2]-.8;hard=(margin>0).astype(float)
        if anchors is not None:
            old_margin=anchors['margins'][tick];hard=(old_margin>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old_margin))**2*(margin-old_margin)
        margins.append(margin.copy());spikes.append(hard)
    spikes=np.stack(spikes,axis=1);logits=spikes.mean(1)*p['logit_scale'];maximum=logits.max(1)
    loss=maximum+np.log(np.exp(logits-maximum[:,None]).sum(1))-logits[np.arange(batch),labels]
    objective=loss.mean()
    if anchors is not None:objective+=(anchors['loss']*logp).mean()
    return objective,live,np.concatenate([np.zeros((batch,length,1)),spikes],axis=2),dict(before=before,counts=counts,margins=margins,loss=loss)


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('window',[None,2])
def test_delayed_aliases_share_emission_identity_across_visits_and_carry(ranks,window,tmp_path):
    mpi(ranks);p,w=model(window,ranks);p['trainable']=[False]
    for delay,a in enumerate(p['dynamic']['actions'][:2]):a.update(event_noise={'delay':delay},trigger={'external':True,'index':0})
    labels=[0,1];initial=np.array([p['dynamic']['initial']]*2);x=np.ones((2,4,1))
    r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,labels,initial=initial,noise_sequence=9)
    loss,live,spikes,anchors=delayed_oracle(p,w,initial,labels,4)
    np.testing.assert_allclose(r['final_state'],live,atol=2e-14,rtol=2e-14);np.testing.assert_array_equal(r['spikes'],spikes)
    assert r['loss']==pytest.approx(loss,abs=2e-14)
    assert len(r['poisson_state']['entries'])==2*5
    for j in range(3):
        plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[0][j]+=1e-6;minus[0][j]-=1e-6
        fd=(delayed_oracle(p,plus,initial,labels,4,anchors)[0]-delayed_oracle(p,minus,initial,labels,4,anchors)[0])/2e-6
        assert r['gradients'][0][j]==pytest.approx(fd,rel=3e-5,abs=4e-6)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(x[:,:1],labels,noise_sequence=9)
    path=tmp_path/'delay';t.store(path);q=NativeLIFTrainer(p,weights=w,runner=RUNNER);q.restore(path)
    result=q.step(x[:,1:],labels,initial='carry');same=t.step(x[:,1:],labels,initial='carry')
    assert result==same
    np.testing.assert_allclose(result['final_state'],live,atol=2e-14,rtol=2e-14)
    assert len(result['poisson_state']['entries'])==10


@pytest.mark.parametrize('ranks',[None,2])
def test_common_indirect_rate_uses_saved_physical_origin(ranks):
    mpi(ranks);p,w=model(ranks=ranks);d=p['dynamic']
    d['initial'].append(1.);d['initial_parameters'].append(None);d['detached'].append(True);d['integer_states'].append(7)
    for a in d['actions'][:2]:a['indirect']={'reads':{'2':{'index':7,'tables':[[4,5]]}},'writes':{}}
    r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)
    rate=w[0][0]*d['initial'][5];count=draw(rate,p['seed'],9,0,0,0)
    assert r['final_state'][0][2]==r['final_state'][0][3]==count
    score=r['loss']*(count/rate-1)
    assert r['gradients'][0][0]==pytest.approx(score*d['initial'][5],abs=3e-14)
    assert r['initial_state_gradients'][0][5]==pytest.approx(score*w[0][0],abs=3e-14)
    assert r['initial_state_gradients'][0][4]==0.


@pytest.mark.parametrize('ranks',[None,2])
def test_failed_original_sample_discards_partial_cache_and_optimizer(ranks):
    mpi(ranks);p,w=model(ranks=ranks,first='k=draw(scale*r)\nv=log(-1.)\nr=.8*r')
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError):t.step(np.zeros((1,1,1)),[0],noise_sequence=9)
    assert t.state==before and t.poisson_state is None and t.clock_tick==0 and t.next_noise_sequence==0


@pytest.mark.parametrize('change',['rate','physical_source'])
def test_incompatible_declared_common_rate_is_rejected(change):
    p,w=model(second='k=draw((scale+1.)*r)\nv=1.-amp*k\nr=.8*r+.2*offset' if change=='rate' else 'k=draw(scale*r)\nv=1.-amp*k\nr=.8*r+.2*offset')
    if change=='physical_source':p['dynamic']['actions'][1]['reads'][2]=5;p['dynamic']['actions'][1]['writes'][2]=5
    with pytest.raises(ValueError,match='incompatible rate definitions'):
        NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((1,1,1)),[0])


@pytest.mark.parametrize('ranks',[None,2])
def test_async_idle_visits_keep_shared_draws_across_carry(ranks,tmp_path):
    mpi(ranks);p,w=model(ranks=ranks);p['trainable']=[False]
    p['dynamic']['clocks']={'start':0.,'dts':[.001,.0005],'epsilon':1e-4,'order':[0,1]}
    for a in p['dynamic']['actions'][:2]:a['clock']=1
    x=np.zeros((1,3,1));whole=NativeLIFTrainer(p,weights=w,runner=RUNNER).evaluate(x,[0],noise_sequence=9)
    _,expected,_,_=oracle(p,w,[p['dynamic']['initial']],[0],6)
    np.testing.assert_allclose(whole['final_state'],expected,rtol=2e-14,atol=2e-14)
    assert len(whole['poisson_state']['entries'])==6
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(x[:,:1],[0],noise_sequence=9)
    path=tmp_path/'async';t.store(path);q=NativeLIFTrainer(p,weights=w,runner=RUNNER);q.restore(path)
    tail=q.step(x[:,1:],[0],initial='carry')
    assert tail['poisson_state']==whole['poisson_state'] and tail['clock_state']['calls']==whole['clock_state']['calls']
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(tail['spikes'],np.asarray(whole['spikes'])[:,1:])


@pytest.mark.parametrize('ranks',[None,2])
def test_readonly_draw_cache_and_fresh_sequence_do_not_commit_old_entries(ranks):
    mpi(ranks);p,w=model(ranks=ranks);t=NativeLIFTrainer(p,weights=w,runner=RUNNER);x=np.zeros((1,2,1))
    t.step(x,[0],noise_sequence=9);saved=copy.deepcopy(t.poisson_state)
    result=t.gradients(x,[0],initial='carry');assert len(result['poisson_state']['entries'])==4
    assert t.poisson_state==saved and t.clock_tick==2
    t.step(x,[0]);assert t.poisson_state['sequence']==10 and len(t.poisson_state['entries'])==2
    assert {e['identity']['instant'] for e in t.poisson_state['entries']}=={0,1}


@pytest.mark.parametrize('backend',['metal','cuda'])
def test_cpu_draw_checkpoint_is_not_imported_into_gpu(backend):
    p,w=model();t=NativeLIFTrainer(p,weights=w,runner=RUNNER);p['backend']=backend
    before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='invalid Poisson draw checkpoint numerical profile'):
        t._run_request(dict(plan=p,state=t.state,operation='train',inputs=[[[0.]]],labels=[0],initial=None,start_tick=1,poisson_state=dict(schema='b2-poisson-draw-state-v1',seed=p['seed'],sequence=0,batch=1,entries=[])))
    assert t.state==before and t.poisson_state is None and t.clock_tick==0
