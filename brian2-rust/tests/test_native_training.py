"""Native CPU BPTT; independent local-surrogate VJP and actual learning."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from brian2_rust.training import NativeLIFTrainer, lif_training_plan

RUNNER=Path(os.environ.get('B2_TRAIN_RUNNER',Path(__file__).resolve().parents[1]/'target/release/b2-train'))
pytestmark=pytest.mark.skipif(not RUNNER.exists(), reason='build native b2-train')


def oracle(plan, weights, inputs, labels, initial=None, anchors=None):
    """Independent NumPy forward and smooth local surrogate linearization.

    Perturbing this linearization tests the declared VJP, not the nonexistent
    true derivative of the hard step. Recorded hard spikes and phi are frozen
    at the baseline; reset detach also freezes only its own spike operand.
    """
    inputs=np.asarray(inputs); batch,time,_=inputs.shape
    sizes=plan['sizes']; offsets=np.cumsum([0,*sizes[1:]])
    n=offsets[-1]; state=np.zeros((batch,n)) if initial is None else np.asarray(initial).copy()
    spikes=[]; membranes=[]
    for t in range(time):
        u=np.concatenate([plan['beta'][l]*state[:,offsets[l]:offsets[l+1]] for l in range(len(sizes)-1)],axis=1)
        hard=np.concatenate([(u[:,offsets[l]:offsets[l+1]]>plan['threshold'][l]).astype(float) for l in range(len(sizes)-1)],axis=1)
        s=hard.copy(); reset_spike=hard.copy()
        if anchors is not None:
            base_u,base_s=anchors
            threshold=np.repeat(plan['threshold'],sizes[1:])
            phi=plan['surrogate']['scale']/(1+plan['surrogate']['slope']*np.abs(base_u[:,t]-threshold))**2
            s=base_s[:,t]+phi*(u-base_u[:,t])
            reset_spike=base_s[:,t] if plan['detach_reset'] else s
        state=u.copy()
        for l in range(len(sizes)-1):
            x=inputs[:,t] if l==0 else s[:,offsets[l-1]:offsets[l]]
            w=np.asarray(weights[l]).reshape(sizes[l],sizes[l+1])
            state[:,offsets[l]:offsets[l+1]]+=x@w
        if plan['reset']=='zero': state*=1-reset_spike
        else: state-=np.repeat(plan['threshold'],sizes[1:])*reset_spike
        spikes.append(s.copy());membranes.append(u.copy())
    spikes=np.stack(spikes,axis=1);membranes=np.stack(membranes,axis=1)
    logits=spikes[:,:,offsets[-2]:].mean(axis=1)*plan['logit_scale']
    maximum=logits.max(axis=1)
    loss=np.mean(maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=1))-logits[np.arange(batch),labels])
    return loss,spikes,state,(membranes,spikes)


@pytest.mark.parametrize('reset', ['subtract','zero'])
@pytest.mark.parametrize('detach', [False,True])
def test_independent_surrogate_vjp(reset,detach):
    plan=lif_training_plan([2,2,2],beta=0.5,reset=reset,detach_reset=detach,surrogate_slope=2)
    weights=[[0.8,0.4,0.2,1.1],[0.9,0.3,0.4,0.8]]
    inputs=np.array([[[1.,0.],[0.,1.],[1.,1.],[0.,1.],[1.,0.],[1.,1.]]])
    initial=np.array([[0.3,1.9,0.2,0.1]])
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    result=trainer.gradients(inputs,[0],initial=initial)
    loss,spikes,final,anchors=oracle(plan,weights,inputs,[0],initial)
    np.testing.assert_array_equal(result['spikes'],spikes)
    np.testing.assert_allclose(result['final_membrane'],final,rtol=0,atol=1e-15)
    assert result['loss']==pytest.approx(loss,abs=1e-15)
    epsilon=1e-6
    for l,row in enumerate(weights):
        for e in range(len(row)):
            plus=copy.deepcopy(weights);minus=copy.deepcopy(weights)
            plus[l][e]+=epsilon;minus[l][e]-=epsilon
            derivative=(oracle(plan,plus,inputs,[0],initial,anchors)[0]-oracle(plan,minus,inputs,[0],initial,anchors)[0])/(2*epsilon)
            assert result['gradients'][l][e]==pytest.approx(derivative,rel=2e-5,abs=2e-8)
    for e in range(4):
        plus=initial.copy();minus=initial.copy();plus[0,e]+=epsilon;minus[0,e]-=epsilon
        derivative=(oracle(plan,weights,inputs,[0],plus,anchors)[0]-oracle(plan,weights,inputs,[0],minus,anchors)[0])/(2*epsilon)
        assert result['initial_gradients'][0][e]==pytest.approx(derivative,rel=2e-5,abs=2e-8)


def data():
    inputs=np.zeros((4,24,2));inputs[:2,:,0]=1;inputs[2:,:,1]=1
    return inputs,[0,0,1,1]


def test_hidden_learning_freeze_and_heldout():
    inputs,labels=data()
    plan=lif_training_plan([2,2,2],beta=0.8,learning_rate=0.04,surrogate_slope=3,logit_scale=3)
    weights=[[1.2,0.4,0.4,1.2],[0.2,0.8,0.8,0.2]]
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    before=trainer.evaluate(inputs,labels)['loss']
    for _ in range(60): trainer.step(inputs,labels)
    after=trainer.evaluate(inputs,labels)['loss']
    assert after < before*0.6
    assert trainer.state['weights'][0]!=weights[0]
    assert trainer.state['weights'][1]!=weights[1]
    # Different sequences, never used for an optimizer step.
    heldout=inputs.copy();heldout[:,::3]=0
    state=copy.deepcopy(trainer.state)
    observed=trainer.evaluate(heldout,labels)
    assert np.argmax(observed['logits'],axis=1).tolist()==labels
    assert trainer.state==state
    frozen=NativeLIFTrainer(dict(plan,trainable=[False,True]),runner=RUNNER,weights=weights)
    frozen.step(inputs,labels)
    assert frozen.state['weights'][0]==weights[0]
    assert frozen.state['first_moment'][0]==[0]*4


def test_bptt_time_and_truncation():
    inputs,labels=data();weights=[[1.2,0.4,0.4,1.2],[0.2,0.8,0.8,0.2]]
    plan=lif_training_plan([2,2,2],beta=0.8)
    full=NativeLIFTrainer(plan,runner=RUNNER,weights=weights).gradients(inputs,labels)
    wide=NativeLIFTrainer(dict(plan,tbptt_window=24),runner=RUNNER,weights=weights).gradients(inputs,labels)
    short=NativeLIFTrainer(dict(plan,tbptt_window=2),runner=RUNNER,weights=weights).gradients(inputs,labels)
    assert wide['gradients']==full['gradients']
    assert short['gradients']!=full['gradients']
    assert full['gradient_scope']=='full-bptt'
    assert short['gradient_scope']=='tbptt-detach-boundaries'
    assert any(abs(g)>0 for g in full['initial_gradients'][0])


def test_checkpoint_exact_optimizer_and_failure(tmp_path):
    inputs,labels=data();plan=lif_training_plan([2,3,2])
    trainer=NativeLIFTrainer(plan,runner=RUNNER)
    for _ in range(3): trainer.step(inputs,labels)
    path=tmp_path/'checkpoint.json';trainer.store(path)
    for _ in range(2): trainer.step(inputs,labels)
    expected=copy.deepcopy(trainer.state)
    other=NativeLIFTrainer(plan,runner=RUNNER);other.restore(path)
    for _ in range(2): other.step(inputs,labels)
    assert other.state==expected
    # Cross-process restore includes Adam moments, count and native RNG state.
    script='''import json,sys
from brian2_rust.training import *
t=NativeLIFTrainer(json.load(open(sys.argv[1]))['plan'],runner=sys.argv[2])
t.restore(sys.argv[1]+'.checkpoint')
x=[[[1.,0.]]*24]*2+[[[0.,1.]]*24]*2
for _ in range(2):t.step(x,[0,0,1,1])
json.dump(t.state,open(sys.argv[3],'w'))
'''
    import json
    (tmp_path/'plan').write_text(json.dumps({'plan':plan}))
    (tmp_path/'plan.checkpoint').write_bytes(path.read_bytes())
    subprocess.run([sys.executable,'-c',script,str(tmp_path/'plan'),str(RUNNER),str(tmp_path/'state')],check=True)
    assert json.loads((tmp_path/'state').read_text())==expected
    original=copy.deepcopy(other.state)
    envelope=json.loads(path.read_text());envelope['sha256']='0'*64;path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError,match='integrity'): other.restore(path)
    assert other.state==original


@pytest.mark.parametrize('change', [dict(refractory=2),dict(max_tape_bytes=1),dict(surrogate={'kind':'unknown','slope':5,'scale':1}),dict(tbptt_window=0)])
def test_fail_closed(change):
    plan=lif_training_plan([2,2,2]);plan.update(change)
    trainer=NativeLIFTrainer(plan,runner=RUNNER)
    inputs,labels=data()
    with pytest.raises(ValueError): trainer.step(inputs,labels)
    assert trainer.state is None


def test_parameter_budget_checked_before_allocation():
    with pytest.raises(ValueError,match='budget'):
        lif_training_plan([65536,65536,65536])


def test_mask_boundary_clears_optimizer():
    trainer=NativeLIFTrainer(lif_training_plan([2,2,2]),runner=RUNNER)
    inputs,labels=data();trainer.step(inputs,labels)
    trainer.update_mask([[0,1,1,1],[1,1,1,1]])
    for key in ('weights','first_moment','second_moment'): assert trainer.state[key][0][0]==0
    trainer.step(inputs,labels)
    trainer.update_mask([[1,1,1,1],[1,1,1,1]],growth_weight=0.3)
    assert trainer.state['weights'][0][0]==0.3
    assert trainer.state['first_moment'][0][0]==trainer.state['second_moment'][0][0]==0


@pytest.mark.parametrize('reset',['zero','subtract'])
def test_canonical_brian_forward(reset,tmp_path):
    import brian2 as b
    from brian2.devices.device import all_devices
    from brian2_rust.export import lower_network
    from test_mpi import reference
    native=all_devices['rust_standalone'];native.reinit()
    b.set_device('rust_standalone',runner=os.environ.get('B2_RUNNER'))
    clock=b.Clock(dt=b.second/1024)
    inp=b.SpikeGeneratorGroup(2,[0,1,0,1,0,1],np.arange(6)*clock.dt,clock=clock,name='grad_input')
    groups=[];objects=[inp];previous=inp
    weights=[[0.8,0.4,0.2,1.1],[0.9,0.3,0.4,0.8]]
    for l in range(2):
        g=b.NeuronGroup(2,'dv/dt=-v/(2*dt):1',threshold='v>1',reset='v=0' if reset=='zero' else 'v-=1',clock=clock,method='euler',name=f'grad_layer{l}')
        s=b.Synapses(previous,g,'w:1 (constant)',on_pre='v_post+=w',clock=clock,name=f'grad_projection{l}')
        s.connect(i=[0,0,1,1],j=[0,1,0,1]);s.w=weights[l]
        objects.extend([g,s]);groups.append(g);previous=g
    net=b.Network(*objects)
    model=lower_network(net,6*clock.dt)
    expected=reference(model,tmp_path/'reference')
    x=np.zeros((1,6,2));x[0,np.arange(6),np.arange(6)%2]=1
    trainer=NativeLIFTrainer(lif_training_plan([2,2,2],beta=0.5,reset=reset),runner=RUNNER,weights=weights)
    result=trainer.evaluate(x,[0])
    for l,g in enumerate(groups):
        p=next(p for p,d in enumerate(model['definition']['populations']) if d['name']==g.name)
        observed=expected['populations'][p]
        np.testing.assert_array_equal(result['final_membrane'][0][l*2:l*2+2],observed['states']['v'])
        spikes=np.zeros((6,2))
        spikes[observed['event_streams']['spike']['ticks'],observed['event_streams']['spike']['indices']]=1
        np.testing.assert_array_equal(np.asarray(result['spikes'])[0,:,l*2:l*2+2],spikes)
    native.reinit()


@pytest.mark.skipif(os.environ.get('B2_TEST_GPU')!='1',reason='actual native Metal GPU required')
@pytest.mark.parametrize('reset,detach', [('subtract',True),('subtract',False),('zero',True),('zero',False)])
def test_real_metal_forward_vjp_training_checkpoint(reset,detach,tmp_path):
    inputs,labels=data();weights=[[1.2,0.4,0.4,1.2],[0.2,0.8,0.8,0.2]]
    plan=lif_training_plan([2,2,2],beta=0.8,reset=reset,detach_reset=detach)
    cpu=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    gpu=NativeLIFTrainer(dict(plan,backend='metal'),runner=RUNNER,weights=weights)
    expected=cpu.gradients(inputs,labels)
    actual=gpu.gradients(inputs,labels)
    np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
    for key in ('final_membrane','gradients','initial_gradients','logits'):
        np.testing.assert_allclose(actual[key],expected[key],rtol=3e-5,atol=3e-6)
    assert actual['loss']==pytest.approx(expected['loss'],rel=3e-6)
    gpu.step(inputs,labels)
    assert gpu.state['weights'][0]!=weights[0] and gpu.state['weights'][1]!=weights[1]
    path=tmp_path/'metal-checkpoint';gpu.store(path)
    gpu.step(inputs,labels);expected=copy.deepcopy(gpu.state)
    gpu.restore(path);gpu.step(inputs,labels)
    assert gpu.state==expected
    script='''
import json,sys
import numpy as np
from brian2_rust.training import NativeLIFTrainer
t=NativeLIFTrainer(json.loads(sys.argv[2]))
t.restore(sys.argv[1])
x=np.zeros((4,24,2));x[:2,:,0]=1;x[2:,:,1]=1
t.step(x,[0,0,1,1])
print(json.dumps(t.state))
'''
    fresh=subprocess.run([sys.executable,'-c',script,str(path),json.dumps(gpu.plan)],
                         check=True,capture_output=True,text=True,
                         timeout=float(os.environ.get('B2_TEST_CHILD_TIMEOUT', '60')))
    assert json.loads(fresh.stdout)==expected
    gpu._metal_library.write_bytes(b'broken')
    with pytest.raises(ValueError,match='library integrity'): gpu.step(inputs,labels)


@pytest.mark.skipif(os.environ.get('B2_TEST_GPU')!='1',reason='actual native Metal GPU required')
def test_real_metal_deeper_masked_tbptt():
    inputs,labels=data()
    plan=lif_training_plan([2,2,2,2],beta=[0.8,0.7,0.9],reset='zero',detach_reset=False,
                           tbptt_window=5,masks=[[1,0,1,1],[1,1,1,1],[1,1,0,1]])
    weights=[[1.2,0,0.4,1.2],[1.1,0.2,0.3,1.1],[0.2,0.8,0,0.2]]
    initial=np.array([[.3,.2,.1,.4,.5,.1]]*4)
    cpu=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    gpu=NativeLIFTrainer(dict(plan,backend='metal'),runner=RUNNER,weights=weights)
    expected=cpu.gradients(inputs,labels,initial=initial)
    actual=gpu.gradients(inputs,labels,initial=initial)
    for key in ('gradients','initial_gradients','final_membrane','logits'):
        np.testing.assert_allclose(actual[key],expected[key],rtol=4e-5,atol=3e-6)
    assert actual['gradients'][0][1]==actual['gradients'][2][2]==0
    assert actual['gradient_scope']=='tbptt-detach-boundaries'
