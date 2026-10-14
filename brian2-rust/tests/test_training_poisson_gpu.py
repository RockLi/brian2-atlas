"""Poisson score training on real devices, including owner-compute local MPI."""
import copy
import math
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_ssa import model,single_transform,oracle,draw
from test_training_delay_update import snapshot


@pytest.mark.parametrize('window',[None,1,2])
@pytest.mark.parametrize('batch',[1,3])
def test_device_all_vjps(engine,window,batch):
    plan,weights=model(window);plan['backend']=engine
    initial=np.array([plan['dynamic']['initial']]*batch);initial[:,4]+=np.arange(batch)*.1
    labels=np.arange(batch)%2;inputs=np.zeros((batch,4,1))
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(inputs,labels,initial=initial,noise_sequence=9)
    loss,final,spikes,anchors=oracle(plan,weights,initial,labels)
    tol=3e-5 if engine!='cpu' else 3e-10
    np.testing.assert_allclose(result['final_state'],final,rtol=tol,atol=tol)
    np.testing.assert_array_equal(result['spikes'],spikes)
    np.testing.assert_allclose(result['loss'],loss,rtol=tol)
    eps=1e-6
    for k in range(3):
        a=copy.deepcopy(weights);b=copy.deepcopy(weights);a[0][k]+=eps;b[0][k]-=eps
        expected=(oracle(plan,a,initial,labels,anchors=anchors)[0]-oracle(plan,b,initial,labels,anchors=anchors)[0])/(2*eps)
        np.testing.assert_allclose(result['gradients'][0][k],expected,rtol=max(tol,2e-7),atol=max(tol,2e-8))
    for sample in range(batch):
        for k in (0,1,4,5):
            a=initial.copy();b=initial.copy();a[sample,k]+=eps;b[sample,k]-=eps
            expected=(oracle(plan,weights,a,labels,anchors=anchors)[0]-oracle(plan,weights,b,labels,anchors=anchors)[0])/(2*eps)
            np.testing.assert_allclose(result['initial_state_gradients'][sample][k],expected,rtol=max(tol,2e-7),atol=max(tol,2e-8))
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('indirect',[False,True])
def test_device_mpi_checkpoint_and_indirect_score(engine,ranks,indirect,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    plan,weights=model();plan['backend']=engine
    if indirect:
        # Route rate through a typed zero selector. Both output programs reuse
        # the same sampled count, with a detached integer write.
        d=plan['dynamic'];d['initial'].append(0.);d['initial_parameters'].append(None);d['detached'].append(True);d['integer_states'].append(7)
        for j in range(2):d['actions'][j]['indirect']={'reads':{'2':{'index':7,'tables':[[4+j]]}},'writes':{}}
    x=np.zeros((2,4,1));labels=[0,1]
    expected=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(x,labels,noise_sequence=9)
    plan['mpi_ranks']=ranks
    actual=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(x,labels,noise_sequence=9)
    for key in ('loss','spikes','final_state','gradients','initial_state_gradients'):
        np.testing.assert_allclose(actual[key],expected[key],rtol=3e-5,atol=3e-6)
    plan['trainable']=[False];trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    trainer.step(x[:,:2],labels,noise_sequence=9);path=tmp_path/'poisson-gpu.json';trainer.store(path)
    restored=NativeLIFTrainer(plan,runner=RUNNER);restored.restore(path)
    result=restored.step(x[:,2:],labels,initial='carry')
    np.testing.assert_allclose(result['final_state'],expected['final_state'],rtol=3e-5,atol=3e-6)
    assert result['final_tick']==4 and result['noise_sequence']==9


@pytest.mark.parametrize('event',[None,{'delay':3},{'delay':0,'pending':2**64-7}])
def test_device_full_key_and_events(engine,event):
    plan,weights=single_transform('k=draw(scale)');plan['backend']=engine
    plan['seed']=2**64-1;sequence=2**64-2;start=9
    if event is not None:plan['dynamic']['actions'][0].update(event_noise=event,trigger={'external':True,'index':0})
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.ones((1,1,1)),[0],noise_sequence=sequence,start_tick=start)
    address=None if event is None else (2,event['pending']) if 'pending' in event else (1,start-event['delay'])
    expected=draw(weights[0][0],plan['seed'],sequence,0,0,start,address)
    assert result['final_state'][0][2]==expected
    np.testing.assert_allclose(result['gradients'][0][0],result['loss']*(expected-weights[0][0])/weights[0][0],rtol=3e-6)


@pytest.mark.parametrize('ranks',[None,2])
def test_large_counts_keep_integer_bits_and_scores(engine,ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    plan,weights=single_transform('k=draw(scale)');plan.update(backend=engine,mpi_ranks=ranks);weights[0][0]=1e9
    batch=128;result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((batch,1,1)),[0]*batch,noise_sequence=173)
    counts=np.asarray(result['final_state'])[:,2]
    assert np.all(counts==counts.astype(np.int32)) and np.any(counts%2==1) and np.mean(counts%64!=0)>.9
    expected=result['loss']*np.mean((counts-1e9)/1e9)
    np.testing.assert_allclose(result['gradients'][0][0],expected,rtol=3e-5,atol=1e-11)


@pytest.mark.parametrize('kind',['zero_constant','lazy','mask','inactive','nested'])
def test_device_conditional_and_nested_sites(engine,kind):
    code={'zero_constant':'k=draw(0.)\nv=.1*k',
          'lazy':'k=draw(-scale) if r < 0 else 0\nv=.1*k',
          'mask':'k=draw(-scale)\nv=.1*k','inactive':'k=draw(-scale)\nv=.1*k',
          'nested':'k=draw(scale)\nv=amp*other(k+scale)'}[kind]
    plan,weights=single_transform(code);plan['backend']=engine
    action=plan['dynamic']['actions'][0]
    if kind=='mask':action['mask']=[0,0];plan['masks'][0][0]=0.;weights[0][0]=0.
    if kind=='inactive':action['trigger']={'external':True,'index':0}
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)
    if kind=='nested':
        first=result['final_state'][0][2];second=round(result['final_state'][0][0]/weights[0][1]);rate=weights[0][0]
        expected=result['loss']*((first-rate)/rate+(second-first-rate)/(first+rate))
        np.testing.assert_allclose(result['gradients'][0][0],expected,rtol=3e-5,atol=1e-6)
    else:np.testing.assert_array_equal(result['gradients'],np.zeros((1,3)))


@pytest.mark.parametrize('rate',[0.,-1.,float(2**31)])
@pytest.mark.parametrize('ranks',[None,2])
def test_device_errors_roll_back(engine,rate,ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    plan,weights=single_transform('k=draw(scale)');plan.update(backend=engine,mpi_ranks=ranks);weights[0][0]=rate
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER);before=snapshot(trainer)
    if rate == 0.:
        result=trainer.gradients(np.zeros((1,1,1)),[0])
        # This detached draw has no consumer, so its weak loss difference is zero.
        np.testing.assert_array_equal(result['gradients'],np.zeros((1,3)))
        # A successful gradients call updates last_result, but not training state.
        assert snapshot(trainer)[:-1]==before[:-1] and trainer.last_result==result
    else:
        with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0])
        assert snapshot(trainer)==before


@pytest.mark.parametrize('version',[None,0,2])
def test_old_gpu_library_cannot_ignore_poisson(tmp_path,monkeypatch,version):
    from brian2_rust import training_metal
    source=tmp_path/'old.c'
    capability='' if version is None else f'uint64_t b2_train_poisson_v1(void){{return {version};}}\n'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\nuint64_t b2_train_poisson_shared_v1(void){return 1;}\nuint64_t b2_train_poisson_persistent_v1(void){return 1;}\n'+capability)
    library=tmp_path/'old.dylib'
    subprocess.run(['clang','-dynamiclib',str(source),'-o',str(library)],check=True,capture_output=True,text=True)
    monkeypatch.setattr(training_metal,'build',lambda directory:library)
    plan,weights=model();plan['backend']='metal';trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER);before=snapshot(trainer)
    with pytest.raises(ValueError,match='Poisson ABI capability '+('missing' if version is None else 'mismatch')):
        trainer.gradients(np.zeros((1,1,1)),[0])
    assert snapshot(trainer)==before


@pytest.mark.parametrize('rate',[float(np.nextafter(np.float32(0),np.float32(1))),float(np.finfo(np.float32).tiny),1e-12])
def test_tiny_positive_rates_keep_likelihood_score(engine,rate):
    plan,weights=single_transform('k=draw(scale)');plan['backend']=engine;weights[0][0]=rate
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0],noise_sequence=9)
    assert result['final_state'][0][2]==0
    np.testing.assert_allclose(result['gradients'][0][0],-result['loss'],rtol=2e-6)


def test_gpu_noise_context_is_budgeted_before_allocation(engine):
    plan,weights=model();plan['backend']=engine
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    result=trainer.gradients(np.zeros((2,4,1)),[0,1],noise_sequence=9)
    plan['max_tape_bytes']=result['tape_bytes']-1
    other=NativeLIFTrainer(plan,weights=weights,runner=RUNNER);before=snapshot(other)
    with pytest.raises(ValueError,match='budget'):other.gradients(np.zeros((2,4,1)),[0,1],noise_sequence=9)
    assert snapshot(other)==before


def test_device_score_matches_exact_hard_count_expectation(engine):
    plan,weights=single_transform('k=draw(scale)\nv=amp*k');plan['backend']=engine
    batch=2048;rate=weights[0][0];amp=weights[0][1]
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((batch,1,1)),[0]*batch,noise_sequence=173)
    counts=np.asarray(result['final_state'])[:,2];spikes=(amp*counts>.8).astype(float)
    losses=np.log1p(np.exp(-plan['logit_scale']*spikes));samples=losses*(counts-rate)/rate
    np.testing.assert_allclose(result['gradients'][0][0],samples.mean(),rtol=3e-6,atol=2e-7)
    cutoff=math.floor(.8/amp)+1
    expected=(math.log1p(math.exp(-plan['logit_scale']))-math.log(2))*math.exp(-rate)*rate**(cutoff-1)/math.factorial(cutoff-1)
    assert abs(samples.mean()-expected)<7*samples.std(ddof=1)/math.sqrt(batch)
