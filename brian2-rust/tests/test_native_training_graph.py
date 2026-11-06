"""v2 shared/recurrent graphs: independent smooth VJP and real Metal execution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from brian2_rust import (NativeLIFTrainer, lif_training_plan,
                        dense_training_projection, conv2d_training_projection)
from test_native_training import RUNNER

pytestmark=pytest.mark.skipif(not RUNNER.exists(),reason='build b2-train')


def fixture_graph(reset='subtract',detach=True,window=None):
    p0=dense_training_projection(0,1,2,2)
    recurrent=dict(source_layer=1,target_layer=1,parameter_count=2,
                   sources=[0,1,0,1],targets=[0,1,1,0],parameter_ids=[0,0,1,1])
    p2=dense_training_projection(1,2,2,2)
    feedback=dense_training_projection(2,1,2,2)
    plan=lif_training_plan([2,2,2],projections=[p0,recurrent,p2,feedback],
                          beta=[.5,.75],reset=reset,detach_reset=detach,
                          surrogate_slope=2,tbptt_window=window)
    weights=[[1.2,.4,.2,1.1],[.13,-.03],[.8,.2,.3,.9],[.1,-.02,.04,.1]]
    x=np.array([[[1.,0.],[0.,1.],[1.,1.],[0.,1.],[1.,0.],[1.,1.],[0.,1.],[1.,0.]],
                [[0.,1.],[1.,0.],[1.,1.],[1.,0.],[0.,1.],[1.,0.],[1.,1.],[0.,1.]]])
    # Immediate output spikes make the feedback-bank weight VJP nonzero,
    # including inside the first truncated window.
    initial=np.array([[.3,1.9,1.8,1.7],[1.2,.1,1.6,1.9]])
    return plan,weights,x,[0,1],initial


def oracle(plan,weights,inputs,labels,initial,anchors=None):
    """Expand tied parameters into independent dense operators with np.add.at.

    For derivatives, linearize spikes at a fixed hard forward trajectory and
    detach each TBPTT boundary to the corresponding recorded membrane state.
    """
    x=np.asarray(inputs);batch,time,_=x.shape;sizes=plan['sizes']
    offsets=np.cumsum([0,*sizes[1:]]);v=np.asarray(initial).copy()
    thresholds=np.repeat(plan['threshold'],sizes[1:]);beta=np.repeat(plan['beta'],sizes[1:])
    matrices=[]
    for p,w in zip(plan['projections'],weights):
        matrix=np.zeros((sizes[p['source_layer']],sizes[p['target_layer']]))
        values=np.asarray(w)[p['parameter_ids']]*np.asarray(plan['masks'][len(matrices)])[p['parameter_ids']]
        np.add.at(matrix,(p['sources'],p['targets']),values)
        matrices.append(matrix)
    spikes=[];pres=[];starts=[]
    for t in range(time):
        window=plan['tbptt_window']
        if anchors is not None and window and t>0 and t%window==0:
            v=anchors[2][:,t].copy()
        starts.append(v.copy());u=beta*v;hard=(u>thresholds).astype(float)
        spike=hard;reset_spike=hard
        if anchors is not None:
            base_u,base_s,_=anchors
            phi=plan['surrogate']['scale']/(1+plan['surrogate']['slope']*abs(base_u[:,t]-thresholds))**2
            spike=base_s[:,t]+phi*(u-base_u[:,t])
            reset_spike=base_s[:,t] if plan['detach_reset'] else spike
        v=u.copy()
        for p,matrix in zip(plan['projections'],matrices):
            source=p['source_layer'];target=p['target_layer']
            row=x[:,t] if source==0 else spike[:,offsets[source-1]:offsets[source]]
            v[:,offsets[target-1]:offsets[target]]+=row@matrix
        v=v*(1-reset_spike) if plan['reset']=='zero' else v-thresholds*reset_spike
        pres.append(u.copy());spikes.append(spike.copy())
    spikes=np.stack(spikes,axis=1)
    logits=spikes[:,:,offsets[-2]:].mean(axis=1)*plan['logit_scale']
    maximum=logits.max(axis=1)
    loss=np.mean(maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=1))-logits[np.arange(batch),labels])
    return loss,spikes,v,logits,(np.stack(pres,axis=1),spikes,np.stack(starts,axis=1))


@pytest.mark.parametrize('reset',['subtract','zero'])
@pytest.mark.parametrize('detach',[False,True])
@pytest.mark.parametrize('window',[None,3])
def test_recurrent_shared_vjp_independent(reset,detach,window):
    plan,weights,x,y,initial=fixture_graph(reset,detach,window)
    result=NativeLIFTrainer(plan,runner=RUNNER,weights=weights).gradients(x,y,initial=initial)
    loss,spikes,final,logits,anchors=oracle(plan,weights,x,y,initial)
    assert result['loss']==pytest.approx(loss,abs=2e-15)
    np.testing.assert_array_equal(result['spikes'],spikes)
    np.testing.assert_allclose(result['final_membrane'],final,rtol=0,atol=2e-15)
    assert any(abs(v)>1e-7 for v in result['gradients'][1]),'recurrent VJP must be nontrivial'
    assert any(abs(v)>1e-7 for v in result['gradients'][3]),'feedback VJP must be nontrivial'
    eps=1e-6
    for q,w in enumerate(weights):
        for e in range(len(w)):
            plus=copy.deepcopy(weights);minus=copy.deepcopy(weights)
            plus[q][e]+=eps;minus[q][e]-=eps
            fd=(oracle(plan,plus,x,y,initial,anchors)[0]-oracle(plan,minus,x,y,initial,anchors)[0])/(2*eps)
            assert result['gradients'][q][e]==pytest.approx(fd,rel=3e-5,abs=3e-8)
    for b in range(2):
        for k in range(4):
            plus=initial.copy();minus=initial.copy();plus[b,k]+=eps;minus[b,k]-=eps
            fd=(oracle(plan,weights,x,y,plus,anchors)[0]-oracle(plan,weights,x,y,minus,anchors)[0])/(2*eps)
            assert result['initial_gradients'][b][k]==pytest.approx(fd,rel=3e-5,abs=3e-8)


@pytest.mark.parametrize('stride,padding',[(1,0),(2,1)])
def test_conv2d_geometry_and_tied_optimizer(stride,padding):
    p,shape=conv2d_training_projection(0,1,(2,3,4),2,(2,3),stride=stride,padding=padding)
    hidden=int(np.prod(shape))
    plan=lif_training_plan([24,hidden,2],projections=[p,dense_training_projection(1,2,hidden,2)],
                          optimizer='sgd',learning_rate=.03,beta=.75)
    rng=np.random.default_rng(27)
    x=rng.integers(0,2,(2,9,24)).astype(float)
    kernel=rng.uniform(.2,.5,(2,2,2,3));readout=rng.uniform(.1,.5,(hidden,2))
    weights=[kernel.ravel().tolist(),readout.ravel().tolist()]
    # Independent spatial cross-correlation, without using the generated IDs.
    image=np.pad(x[0,0].reshape(2,3,4),((0,0),(padding,padding),(padding,padding)))
    spatial=np.empty(shape)
    for oc in range(2):
        for oy in range(shape[1]):
            for ox in range(shape[2]):
                spatial[oc,oy,ox]=(image[:,oy*stride:oy*stride+2,ox*stride:ox*stride+3]*kernel[oc]).sum()
    expanded=np.zeros(hidden)
    np.add.at(expanded,p['targets'],x[0,0,p['sources']]*kernel.ravel()[p['parameter_ids']])
    np.testing.assert_allclose(expanded,spatial.ravel(),rtol=1e-15,atol=1e-15)
    shared=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    actual=shared.gradients(x,[0,1])
    untied=copy.deepcopy(plan);expanded_weights=[]
    for q,projection in enumerate(untied['projections']):
        expanded_weights.append([weights[q][i] for i in projection['parameter_ids']])
        count=len(projection['sources']);projection['parameter_ids']=list(range(count));projection['parameter_count']=count
        untied['masks'][q]=[1.]*count
    expected=NativeLIFTrainer(untied,runner=RUNNER,weights=expanded_weights).gradients(x,[0,1])
    np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
    for q,projection in enumerate(plan['projections']):
        summed=np.zeros(projection['parameter_count'])
        np.add.at(summed,projection['parameter_ids'],expected['gradients'][q])
        np.testing.assert_allclose(actual['gradients'][q],summed,rtol=1e-13,atol=1e-13)
    shared.step(x,[0,1])
    for q in range(2):
        np.testing.assert_allclose(shared.state['weights'][q],np.array(weights[q])-.03*np.array(actual['gradients'][q]),rtol=0,atol=1e-15)
    assert len(shared.state['first_moment'][0])==kernel.size


def test_graph_gates_and_resource_preflight():
    with pytest.raises(ValueError,match='before topology allocation'):
        dense_training_projection(0,1,65536,65536)
    with pytest.raises(ValueError,match='before topology allocation'):
        conv2d_training_projection(0,1,(32,4096,4096),64,3)
    plan,weights,x,y,initial=fixture_graph()
    bad=copy.deepcopy(plan);bad['projections'][1]['parameter_ids'][0]=2
    with pytest.raises(ValueError,match='outside declared domain'):
        NativeLIFTrainer(bad,runner=RUNNER,weights=weights).step(x,y)
    bad=copy.deepcopy(plan);bad['schema']='b2-lif-training-plan-v1'
    with pytest.raises(ValueError,match='projection version'):
        NativeLIFTrainer(bad,runner=RUNNER,weights=weights).step(x,y)
    bad=copy.deepcopy(plan);bad['max_tape_bytes']=100
    with pytest.raises(ValueError,match='budget'):
        NativeLIFTrainer(bad,runner=RUNNER,weights=weights).step(x,y)


@pytest.mark.parametrize('backend',['cpu','metal'])
@pytest.mark.parametrize('reset',['zero','subtract'])
def test_v2_dense_matches_v1_exactly(backend,reset):
    if backend=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('real Metal required')
    common=dict(beta=.5,reset=reset,detach_reset=False,backend=backend)
    v1=lif_training_plan([2,3,2],**common)
    v2=lif_training_plan([2,3,2],projections=[dense_training_projection(0,1,2,3),
                                           dense_training_projection(1,2,3,2)],**common)
    x=np.random.default_rng(12).integers(0,2,(2,12,2)).astype(float)
    a=NativeLIFTrainer(v1,runner=RUNNER).step(x,[0,1])
    b=NativeLIFTrainer(v2,runner=RUNNER).step(x,[0,1])
    a.pop('tape_bytes');b.pop('tape_bytes')
    assert a==b


def test_freeze_feedback_parameter_bank():
    plan,weights,x,y,initial=fixture_graph()
    plan['trainable'][3]=False
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    result=trainer.step(x,y,initial=initial)
    assert any(abs(v)>1e-7 for v in result['gradients'][3])
    assert trainer.state['weights'][3]==weights[3]
    assert trainer.state['first_moment'][3]==trainer.state['second_moment'][3]==[0.]*4
    assert trainer.state['weights'][1]!=weights[1]


def test_live_graph_mutation_rejected_before_commit(tmp_path):
    plan,weights,x,y,initial=fixture_graph()
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    trainer.step(x,y,initial=initial);before=copy.deepcopy(trainer.state)
    trainer.plan['projections'][1]['sources'].reverse()
    with pytest.raises(ValueError,match='topology/backend changed'):trainer.step(x,y)
    with pytest.raises(ValueError,match='topology/backend changed'):trainer.store(tmp_path/'invalid')
    assert trainer.state==before and not (tmp_path/'invalid').exists()


@pytest.mark.skipif(os.environ.get('B2_TEST_GPU')!='1',reason='Metal library toolchain required')
def test_old_metal_metadata_abi_is_rejected(tmp_path):
    source=tmp_path/'legacy.c';library=tmp_path/'legacy.dylib'
    source.write_text('#include <stdint.h>\n'
                      'uint64_t b2_train_math_v1(void) { return 1; }\n'
                      'int b2_train_metal(void) { return 0; }\n')
    subprocess.run(['clang','-dynamiclib',str(source),'-o',str(library)],check=True,capture_output=True)
    plan,weights,x,y,initial=fixture_graph();plan['backend']='metal'
    request=tmp_path/'request.json';output=tmp_path/'result.json'
    request.write_text(json.dumps(dict(plan=plan,state=None,operation='gradients',inputs=x.tolist(),labels=y,initial=initial.tolist())))
    run=subprocess.run([str(RUNNER),str(request),str(output)],capture_output=True,text=True,
                       env=dict(os.environ,B2_TRAIN_METAL_LIB=str(library)))
    assert run.returncode!=0 and 'symbol missing' in run.stderr and not output.exists()


@pytest.mark.skipif(os.environ.get('B2_TEST_GPU')!='1',reason='Metal required')
def test_graph_metal_extra_budget_checked_before_dispatch():
    plan,weights,x,y,initial=fixture_graph()
    cpu=NativeLIFTrainer(plan,runner=RUNNER,weights=weights).gradients(x,y,initial=initial)
    plan.update(backend='metal',max_tape_bytes=cpu['tape_bytes'])
    gpu=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    original=copy.deepcopy(gpu.state)
    with pytest.raises(ValueError,match='Metal tape budget exceeded'):gpu.step(x,y,initial=initial)
    assert gpu.state==original


@pytest.mark.parametrize('backend',['cpu','metal'])
def test_graph_checkpoint_mask_and_fresh_process(backend,tmp_path):
    if backend=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('real Metal required')
    plan,weights,x,y,initial=fixture_graph();plan['backend']=backend
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    trainer.step(x,y,initial=initial)
    masks=copy.deepcopy(plan['masks']);masks[1][0]=0
    trainer.update_mask(masks)
    assert trainer.state['weights'][1][0]==trainer.state['first_moment'][1][0]==0
    trainer.step(x,y,initial='carry')
    masks[1][0]=1;trainer.update_mask(masks,growth_weight=.07)
    path=tmp_path/'graph.checkpoint';trainer.store(path)
    expected=trainer.step(x,y,initial='carry');expected_ticks=trainer.elapsed_ticks
    script='''
import json,sys
from brian2_rust import NativeLIFTrainer
t=NativeLIFTrainer(json.loads(sys.argv[2]))
t.restore(sys.argv[1]);result=t.step(json.loads(sys.argv[3]),[0,1],initial='carry')
print(json.dumps([result,t.elapsed_ticks]))
'''
    output=subprocess.run([sys.executable,'-c',script,str(path),json.dumps(trainer.plan),json.dumps(x.tolist())],
                          capture_output=True,text=True,check=True,timeout=60)
    actual,ticks=json.loads(output.stdout)
    assert actual==expected and ticks==expected_ticks
    changed=copy.deepcopy(trainer.plan);changed['projections'][1]['parameter_ids'].reverse()
    other=NativeLIFTrainer(changed,runner=RUNNER,weights=weights)
    with pytest.raises(ValueError,match='plan/runtime mismatch'):other.restore(path)
    assert other.state['step']==0


@pytest.mark.skipif(os.environ.get('B2_TEST_GPU')!='1',reason='real Metal required')
@pytest.mark.parametrize('reset,detach',[('zero',False),('zero',True),('subtract',False),('subtract',True)])
@pytest.mark.parametrize('window',[None,3])
def test_graph_real_metal_vjp(reset,detach,window):
    plan,weights,x,y,initial=fixture_graph(reset,detach,window)
    cpu=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    gpu=NativeLIFTrainer(dict(plan,backend='metal'),runner=RUNNER,weights=weights)
    expected=cpu.gradients(x,y,initial=initial);actual=gpu.gradients(x,y,initial=initial)
    assert actual['gpu_dispatches']==1
    np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
    for field in ('gradients','initial_gradients','final_membrane','logits'):
        for a,b in zip(actual[field],expected[field]):
            np.testing.assert_allclose(a,b,rtol=4e-5,atol=5e-6)
    assert actual['loss']==pytest.approx(expected['loss'],rel=3e-6)
    before=copy.deepcopy(gpu.state)
    gpu.step(x,y,initial=initial)
    assert gpu.state['weights'][1]!=before['weights'][1]
    assert gpu.state['weights'][3]!=before['weights'][3]
