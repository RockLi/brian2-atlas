"""Scalar equation autodiff against an independent smooth finite difference."""
import copy
import os

import numpy as np
import pytest
from brian2_rust import (NativeLIFTrainer, lif_training_plan, compile_training_equation,
                        neuron_parameter_bank)
from test_native_training import RUNNER
from test_native_training_graph import fixture_graph


def fixture_equation(reset='subtract',detach=True,window=None,backend='cpu'):
    old,w,x,y,initial=fixture_graph(reset,detach,window)
    projections=old['projections']+[neuron_parameter_bank(6)]
    expression=('v + dt*(-v/tau + .05*sin(v) + .02*cos(v) + .03*tanh(v)'
                '+ .02*exp(-v*v) + .01*log(v*v+1) + .01*sqrt(v*v+1) + .002*v**2) + bias')
    programs=[compile_training_equation(expression,parameters=dict(dt=.25,tau=(4,2*l),bias=(4,2*l+1)))
              for l in range(2)]
    plan=lif_training_plan([2,2,2],backend=backend,projections=projections,equations=programs,
         reset=reset,detach_reset=detach,tbptt_window=window,surrogate_slope=2,
         threshold=[1.0625,1.03125],threshold_parameters=[[4,4],[4,5]],learning_rate=.003)
    return plan,w+[[1.7,.02,2.1,-.01,1.0625,1.03125]],x,y,initial


def oracle(plan,w,x,y,initial,anchors=None):
    """Independent closed-form dynamics, expanded matrices and smooth spikes."""
    batch,time,_=x.shape;v=initial.copy();threshold=np.repeat(np.asarray(w[4])[4:],2)
    tau=np.repeat(np.asarray(w[4])[:4:2],2);bias=np.repeat(np.asarray(w[4])[1:4:2],2)
    matrices=[]
    for p,weights in zip(plan['projections'][:4],w):
        a=np.zeros((2,2));np.add.at(a,(p['sources'],p['targets']),np.asarray(weights)[p['parameter_ids']]);matrices.append(a)
    pres=[];spikes=[];starts=[]
    for t in range(time):
        window=plan['tbptt_window']
        if anchors is not None and window and t>0 and t%window==0:v=anchors[2][:,t].copy()
        starts.append(v.copy())
        u=v+.25*(-v/tau+.05*np.sin(v)+.02*np.cos(v)+.03*np.tanh(v)+.02*np.exp(-v*v)
                 +.01*np.log(v*v+1)+.01*np.sqrt(v*v+1)+.002*v**2)+bias
        hard=(u>threshold).astype(float);spike=hard;reset_spike=hard
        if anchors is not None:
            base_u,base_s,_,base_threshold=anchors
            phi=plan['surrogate']['scale']/(1+plan['surrogate']['slope']*abs(base_u[:,t]-base_threshold))**2
            spike=base_s[:,t]+phi*(u-threshold-base_u[:,t]+base_threshold)
            reset_spike=base_s[:,t] if plan['detach_reset'] else spike
        v=u.copy()
        for p,m in zip(plan['projections'],matrices):
            src=p['source_layer'];dst=p['target_layer']
            v[:,2*(dst-1):2*dst]+=(x[:,t] if src==0 else spike[:,2*(src-1):2*src])@m
        v=v*(1-reset_spike) if plan['reset']=='zero' else v-threshold*reset_spike
        pres.append(u);spikes.append(spike)
    spikes=np.stack(spikes,axis=1);logits=spikes[:,:,2:].mean(axis=1)*plan['logit_scale']
    maximum=logits.max(axis=1);loss=np.mean(maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=1))-logits[np.arange(batch),y])
    return loss,spikes,v,(np.stack(pres,axis=1),spikes,np.stack(starts,axis=1),threshold.copy())


@pytest.mark.parametrize('reset',['subtract','zero'])
@pytest.mark.parametrize('detach',[True,False])
@pytest.mark.parametrize('window',[None,3])
def test_equation_independent_vjp(reset,detach,window):
    p,w,x,y,initial=fixture_equation(reset,detach,window)
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x,y,initial=initial)
    loss,spikes,v,anchors=oracle(p,w,x,y,initial)
    assert result['loss']==pytest.approx(loss,abs=3e-15)
    np.testing.assert_array_equal(result['spikes'],spikes)
    np.testing.assert_allclose(result['final_membrane'],v,atol=1e-14,rtol=1e-14)
    assert any(abs(g)>1e-7 for g in result['gradients'][4]),'trainable neuron coefficients need nonzero VJP'
    eps=1e-6
    for q,row in enumerate(w):
        for i in range(len(row)):
            plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[q][i]+=eps;minus[q][i]-=eps
            fd=(oracle(p,plus,x,y,initial,anchors)[0]-oracle(p,minus,x,y,initial,anchors)[0])/(2*eps)
            assert result['gradients'][q][i]==pytest.approx(fd,abs=5e-8,rel=5e-5)
    for b in range(2):
        for k in range(4):
            plus=initial.copy();minus=initial.copy();plus[b,k]+=eps;minus[b,k]-=eps
            fd=(oracle(p,w,x,y,plus,anchors)[0]-oracle(p,w,x,y,minus,anchors)[0])/(2*eps)
            assert result['initial_gradients'][b][k]==pytest.approx(fd,abs=5e-8,rel=5e-5)


@pytest.mark.parametrize('backend',['metal','cuda','mpi'])
@pytest.mark.parametrize('reset',['subtract','zero'])
@pytest.mark.parametrize('detach,window',[(False,None),(True,3)])
def test_equation_native_backends(backend,reset,detach,window):
    flag={'metal':'B2_TEST_GPU','cuda':'B2_TEST_CUDA_TRAIN','mpi':'B2_TEST_MPI'}[backend]
    if os.environ.get(flag)!='1':pytest.skip('native hardware/runtime required')
    p,w,x,y,initial=fixture_equation(reset,detach,window)
    cpu=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    if backend=='mpi':p['mpi_ranks']=4
    else:p['backend']=backend
    other=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    a=cpu.step(x,y,initial=initial);b=other.step(x,y,initial=initial)
    assert b['gpu_dispatches']==(0 if backend=='mpi' else 1)
    np.testing.assert_array_equal(a['spikes'],b['spikes'])
    for key in ['final_membrane','initial_gradients','logits']:
        np.testing.assert_allclose(a[key],b[key],rtol=2e-4,atol=4e-6)
    assert a['loss']==pytest.approx(b['loss'],abs=4e-6)
    for key in ['weights','first_moment','second_moment']:
        for av,bv in zip(a['state'][key],b['state'][key]):np.testing.assert_allclose(av,bv,rtol=2e-4,atol=4e-6)
    for av,bv in zip(a['gradients'],b['gradients']):np.testing.assert_allclose(av,bv,rtol=2e-4,atol=4e-6)


def test_equation_validation_domain_and_freeze(tmp_path):
    for expression in ['__import__("os")','v[0]','v if v>0 else 0','v**v']:
        with pytest.raises(ValueError):compile_training_equation(expression)
    p,w,x,y,initial=fixture_equation();p['trainable'][4]=False
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);trainer.step(x,y,initial=initial)
    assert trainer.state['weights'][4]==w[4] and trainer.state['first_moment'][4]==[0.]*6
    trainer.store(tmp_path/'state');other=NativeLIFTrainer(p,runner=RUNNER);other.restore(tmp_path/'state')
    assert other.step(x,y,initial='carry')==trainer.step(x,y,initial='carry')
    bad=copy.deepcopy(p);bad['equations'][0]=[dict(op='add',left=0,right=0)]
    with pytest.raises(ValueError,match='SSA reference'):NativeLIFTrainer(bad,runner=RUNNER,weights=w).step(x,y)
    p['equations'][0]=compile_training_equation('log(v-100)')
    broken=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(broken.state)
    with pytest.raises(ValueError,match='domain error'):broken.step(x,y)
    assert broken.state==before
