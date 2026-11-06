"""Native Poisson SSA and score VJP; independent fixed-count likelihood oracle."""
import copy
import math
import os

import numpy as np
import pytest
from brian2_rust.training import NativeLIFTrainer, lif_training_plan
from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
from brian2_rust.training_equations import PoissonNoise, neuron_parameter_bank
from test_native_training import RUNNER
from test_training_poisson_core import mix, MASK, uniform as counter_uniform


def draw(rate, seed, sequence, batch, entity, tick, event=None):
    kind,instant=event or (0,tick)
    address=mix(seed^0x4232504f49533031)
    for field in (sequence,batch,71,entity,kind,instant,0):address=mix(address^mix((field+0x9e3779b97f4a7c15)&MASK))
    arrival=0.;count=0
    while True:
        arrival-=math.log1p(-counter_uniform(address,count))
        if arrival>=rate:return count
        count+=1


def model(window=None, ranks=None):
    transform=compile_dynamic_transform('k = draw(r * scale)\nv = .4*v + amp*k\nr = .8*r + .2*offset',
        states={'v':0,'k':1,'r':2},state_types={1:'integer'},
        parameters={'draw':PoissonNoise(0),'scale':(0,0),'amp':(0,1),'offset':(0,2)})
    plan=lif_training_plan([1,1,2],projections=[neuron_parameter_bank(3)],
        state_equations=[[[dict(op='state',index=0)]]]*2,
        state_resets=[[[dict(op='state',index=0)]]]*2,threshold=.8,clock={'origin':0.,'dt':.001},noise_streams=[0,1],seed=731,
        tbptt_window=window,mpi_ranks=ranks,logit_scale=1.7)
    actions=[dynamic_action(transform,[j,2+j,4+j],owner=j+1,program_set=0,
        noise_domain=71,noise_entity=j,noise_streams=1) for j in range(2)]
    actions += [dict(owner=j+1,reads=[j],writes=[],program_set=None,threshold=j+1,trigger=None) for j in range(2)]
    actions.append(dict(owner=0,reads=[6],writes=[],program_set=None,threshold=0,trigger=None))
    plan.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(
        initial=[.1,.2,0.,0.,1.1,1.6,0.],initial_parameters=[None]*7,
        detached=[False,False,True,True,False,False,False],integer_states=[2,3],
        voltage=[6,0,1],program_sets=[transform['programs']],actions=actions))
    return plan, [[1.2,.38,1.3]]


def oracle(plan, weights, initial, labels, length=4, sequence=9, anchors=None):
    live=np.asarray(initial,float).copy();batch=len(live);scale,amp,offset=weights[0]
    before=[];counts=[];margins=[];spikes=[];logps=np.zeros(batch)
    for tick in range(length):
        if anchors is not None and plan['tbptt_window'] and tick and tick%plan['tbptt_window']==0:
            live=anchors['before'][tick].copy()
        before.append(live.copy());rate=live[:,4:6]*scale
        k=np.array([[draw(rate[b,j],plan['seed'],sequence,b,j,tick) for j in range(2)] for b in range(batch)]) if anchors is None else anchors['counts'][tick]
        counts.append(k);live[:,2:4]=k
        logps+=np.array([sum(c*math.log(r)-r-math.lgamma(c+1) for c,r in zip(ks,rs)) for ks,rs in zip(k,rate)])
        live[:,:2]=.4*live[:,:2]+amp*k;live[:,4:6]=.8*live[:,4:6]+.2*offset
        margin=live[:,:2]-.8;hard=(margin>0).astype(float)
        if anchors is not None:
            old=anchors['margins'][tick]
            hard=(old>0).astype(float)+plan['surrogate']['scale']/(1+plan['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());spikes.append(hard)
    spikes=np.stack(spikes,axis=1);logits=spikes.mean(axis=1)*plan['logit_scale']
    maximum=logits.max(axis=1);loss=maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=1))-logits[np.arange(batch),labels]
    objective=loss.mean()
    if anchors is not None:objective+=(anchors['loss']*logps).mean()
    spikes=np.concatenate([np.zeros((batch,length,1)),spikes],axis=2)
    return objective,live,spikes,dict(before=before,counts=counts,margins=margins,loss=loss)


@pytest.mark.parametrize('window',[None,1,2])
@pytest.mark.parametrize('batch',[1,3])
def test_all_parameter_and_initial_vjps(window,batch):
    plan,weights=model(window);labels=np.arange(batch)%2
    initial=np.array([plan['dynamic']['initial']]*batch)
    initial[:,4]+=np.arange(batch)*.1
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    result=trainer.gradients(np.zeros((batch,4,1)),labels,initial=initial,noise_sequence=9)
    loss,final,spikes,anchor=oracle(plan,weights,initial,labels)
    np.testing.assert_allclose(result['loss'],loss,rtol=1e-13)
    np.testing.assert_allclose(result['final_state'],final,rtol=1e-13)
    np.testing.assert_array_equal(result['spikes'],spikes)
    eps=1e-6
    for k in range(3):
        a=copy.deepcopy(weights);b=copy.deepcopy(weights);a[0][k]+=eps;b[0][k]-=eps
        expected=(oracle(plan,a,initial,labels,anchors=anchor)[0]-oracle(plan,b,initial,labels,anchors=anchor)[0])/(2*eps)
        np.testing.assert_allclose(result['gradients'][0][k],expected,rtol=2e-7,atol=2e-8)
    for sample in range(batch):
        for k in (0,1,4,5):
            a=initial.copy();b=initial.copy();a[sample,k]+=eps;b[sample,k]-=eps
            expected=(oracle(plan,weights,a,labels,anchors=anchor)[0]-oracle(plan,weights,b,labels,anchors=anchor)[0])/(2*eps)
            np.testing.assert_allclose(result['initial_state_gradients'][sample][k],expected,rtol=2e-7,atol=2e-8)
    np.testing.assert_array_equal(np.asarray(result['initial_state_gradients'])[:,2:4],0.)


@pytest.mark.parametrize('ranks',[2,8])
def test_mpi_and_checkpoint(ranks,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    plan,weights=model();inputs=np.zeros((2,4,1));labels=[0,1]
    reference=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(inputs,labels,noise_sequence=9)
    plan['mpi_ranks']=ranks
    actual=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(inputs,labels,noise_sequence=9)
    for key in ('gradients','initial_state_gradients','final_state','spikes','loss'):
        np.testing.assert_allclose(actual[key],reference[key],rtol=5e-13,atol=5e-14)
    plan['trainable']=[False]
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    trainer.step(inputs[:,:2],labels,noise_sequence=9)
    path=tmp_path/'checkpoint.json';trainer.store(path)
    restored=NativeLIFTrainer(plan,runner=RUNNER);restored.restore(path)
    result=restored.step(inputs[:,2:],labels,initial='carry')
    np.testing.assert_allclose(result['final_state'],reference['final_state'],rtol=1e-13)
    assert result['final_tick']==4 and result['noise_sequence']==9


@pytest.mark.parametrize('backend',['metal','cuda'])
def test_device_poisson_execution(backend):
    flag='B2_TEST_GPU' if backend=='metal' else 'B2_TEST_CUDA_TRAIN'
    if os.environ.get(flag)!='1':pytest.skip('actual '+backend+' hardware required')
    plan,weights=model();plan['backend']=backend
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0])
    assert result['gpu_dispatches']>0 and result['backend']==backend
    assert np.all(np.isfinite(result['gradients']))


def test_conflicting_rate_and_distribution_and_action_identity():
    plan,weights=model();programs=plan['dynamic']['program_sets'][0]
    # Same stream appears in k and v. Alter just one constant-free rate root.
    altered=copy.deepcopy(plan)
    program=altered['dynamic']['program_sets'][0][1]
    node=next(n for n in program if n['op']=='poisson')
    program[node['rate']]=dict(op='constant',value=3.)
    with pytest.raises(ValueError,match='different rate expressions'):
        NativeLIFTrainer(altered,weights=weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
    altered=copy.deepcopy(plan)
    altered['dynamic']['program_sets'][0][2]=[dict(op='noise',stream=0)]
    with pytest.raises(ValueError,match='mixes distributions'):
        NativeLIFTrainer(altered,weights=weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])
    altered=copy.deepcopy(plan);altered['dynamic']['actions'][1]['noise_entity']=0
    with pytest.raises(ValueError,match='site reused across actions'):
        NativeLIFTrainer(altered,weights=weights,runner=RUNNER).evaluate(np.zeros((1,1,1)),[0])


def single_transform(code, *, types=None):
    plan,weights=model()
    transform=compile_dynamic_transform(code,states={'v':0,'k':1,'r':2},
        state_types={1:'integer',**(types or {})},
        parameters={'draw':PoissonNoise(0),'other':PoissonNoise(1),'scale':(0,0),'amp':(0,1)})
    plan['dynamic']['program_sets']=[transform['programs']]
    plan['dynamic']['actions']=[dynamic_action(transform,[0,2,4],owner=1,program_set=0,
        noise_domain=71,noise_entity=0,noise_streams=2),*plan['dynamic']['actions'][2:]]
    return plan,weights


@pytest.mark.parametrize('code',[
    'k = draw(0.)\nv = .1*k',
    'k = draw(-1.) if r < 0 else 0\nv = .1*k',
    'k = draw(0.) if r < 0 else 0\nv = .1*k',
])
def test_constant_zero_and_lazy_invalid_rate(code):
    plan,weights=single_transform(code)
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,2,1)),[0])
    np.testing.assert_array_equal(result['gradients'],np.zeros((1,3)))
    assert result['final_state'][0][2]==0.


@pytest.mark.parametrize('gate',['mask','event'])
def test_inactive_action_does_not_sample_or_score(gate):
    plan,weights=single_transform('k=draw(-scale)\nv=.1*k')
    action=plan['dynamic']['actions'][0]
    if gate=='mask':
        action['mask']=[0,0];plan['masks'][0][0]=0.;weights[0][0]=0.
    else:action['trigger']={'external':True,'index':0}
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,2,1)),[0])
    np.testing.assert_array_equal(result['gradients'],np.zeros((1,3)))


def test_zero_differentiable_boundary_no_loss_change():
    plan,weights=single_transform('k=draw(scale)\nv=.1*k');weights[0][0]=0.
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    result=trainer.gradients(np.zeros((1,2,1)),[0])
    assert result['final_state'][0][2]==0.
    # Neither one-count counterfactual crosses the .8 hard threshold.
    np.testing.assert_array_equal(result['gradients'],np.zeros((1,3)))


def test_exact_expectation_gradient_for_hard_poisson_threshold():
    plan,weights=single_transform('k=draw(scale)\nv=amp*k')
    batch=4096;rate=weights[0][0];amp=weights[0][1]
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((batch,1,1)),[0]*batch,noise_sequence=173)
    counts=np.asarray(result['final_state'])[:,2]
    spikes=(amp*counts>.8).astype(float)
    losses=np.log1p(np.exp(-plan['logit_scale']*spikes))
    samples=losses*(counts-rate)/rate
    np.testing.assert_allclose(result['gradients'][0][0],samples.mean(),rtol=2e-13,atol=1e-14)
    cutoff=math.floor(.8/amp)+1
    derivative=math.exp(-rate)*rate**(cutoff-1)/math.factorial(cutoff-1)
    expected=(math.log1p(math.exp(-plan['logit_scale']))-math.log(2))*derivative
    assert abs(samples.mean()-expected)<7*samples.std(ddof=1)/math.sqrt(batch)


@pytest.mark.parametrize('indirect',[False,True])
def test_detached_integer_output_still_scores(indirect):
    plan,weights=single_transform('k=draw(scale)')
    if indirect:
        # Fixed one-cell table reached through the original integer zero value.
        plan['dynamic']['actions'][0]['indirect']={'reads':{},'writes':{'0':{'index':{'kind':'read','slot':1},'tables':[[2]]}}}
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)
    count=result['final_state'][0][2]
    np.testing.assert_allclose(result['gradients'][0][0],result['loss']*(count-weights[0][0])/weights[0][0],rtol=1e-13)


def test_nested_poisson_distinct_sites():
    plan,weights=single_transform('k=draw(scale)\nv=amp*other(k+scale)')
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)
    first=result['final_state'][0][2];second=round(result['final_state'][0][0]/weights[0][1]);rate=weights[0][0]
    expected=result['loss']*((first-rate)/rate+(second-first-rate)/(first+rate))
    np.testing.assert_allclose(result['gradients'][0][0],expected,rtol=1e-13)


@pytest.mark.parametrize('event',[None,{'delay':3},{'delay':0,'pending':2**64-7}])
def test_full_width_keys_and_event_replay(event):
    plan,weights=single_transform('k=draw(scale)')
    plan['seed']=2**64-1;sequence=2**64-2;start=9
    action=plan['dynamic']['actions'][0]
    if event is not None:action.update(event_noise=event,trigger={'external':True,'index':0})
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.ones((1,1,1)),[0],noise_sequence=sequence,start_tick=start)
    address=None if event is None else (2,event['pending']) if 'pending' in event else (1,start-event['delay'])
    expected=draw(weights[0][0],plan['seed'],sequence,0,0,start,address)
    assert result['final_state'][0][2]==expected
    np.testing.assert_allclose(result['gradients'][0][0],result['loss']*(expected-weights[0][0])/weights[0][0],rtol=1e-13)
    if event is not None:
        if 'pending' not in event:action['event_noise']['delay']+=2
        replay=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.ones((1,1,1)),[0],noise_sequence=sequence,start_tick=start+2)
        assert replay['final_state']==result['final_state'] and replay['gradients']==result['gradients']
