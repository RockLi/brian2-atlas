"""Integer forward SDK parity and an independent surrogate-gradient oracle."""
import copy
import json
import os
from pathlib import Path
import subprocess

import brian2 as b
import numpy as np
import pytest
from brian2_atlas import NativeXyloTrainer,xylo_training_plan,export_network
from brian2_atlas.neuromorphic import freeze_native_xylo,export_native_xylo
from brian2_rust.native_xylo_conversion import build_native_xylo_network
from brian2_rust.results import load_results

TRAIN=Path(os.environ.get('B2_TRAIN_RUNNER','/nonexistent'))
RUN=Path(os.environ.get('B2_RUNNER','/nonexistent'))
SDK=os.environ.get('B2_XYLO_PYTHON')

@pytest.fixture(autouse=True)
def brian_settings():
    device=b.get_device();target=b.prefs.codegen.target;optim=b.prefs.codegen.loop_invariant_optimisations
    b.set_device('runtime');b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
    yield
    b.set_device(device);b.prefs.codegen.target=target;b.prefs.codegen.loop_invariant_optimisations=optim

@pytest.fixture
def trainer():
    if not TRAIN.is_file():pytest.skip('set B2_TRAIN_RUNNER')
    p=xylo_training_plan([3,4,2],thresholds=(4,4),weight_scale=8.)
    rng=np.random.default_rng(39)
    return NativeXyloTrainer(p,runner=TRAIN,weights=[rng.uniform(-.5,1.,12),rng.uniform(-.5,1.,8)])


def test_training_checkpoint_and_evaluation_purity(trainer,tmp_path):
    x=np.random.default_rng(70).integers(0,2,(4,16,3));y=[0,1,0,1]
    before=copy.deepcopy(trainer.state)
    gradient=trainer.gradients(x,y)
    assert all(np.linalg.norm(g)>0 for g in gradient['gradients'])
    trainer.evaluate(x,y);assert trainer.state==before
    trainer.step(x,y);assert trainer.state['step']==1
    assert all(np.any(np.asarray(w)!=old) for w,old in zip(trainer.state['weights'],before['weights']))
    trainer.store(tmp_path/'state.json')
    restored=NativeXyloTrainer(trainer.plan,runner=TRAIN);restored.restore(tmp_path/'state.json')
    assert restored.step(x,y)==trainer.step(x,y)
    with pytest.raises(ValueError,match='fresh'):trainer.evaluate(x,y,initial='carry')
    trainer.plan['weight_scale']=16
    with pytest.raises(ValueError,match='changed'):trainer.evaluate(x,y)


@pytest.mark.parametrize('change', ['dimensions','threshold','shift','reset','weight_scale','weights','moments','budget','inputs','steps','labels','extension'])
def test_native_rejects_invalid_request(trainer,change):
    request=dict(plan=copy.deepcopy(trainer.plan),state=copy.deepcopy(trainer.state),operation='train',inputs=[[[1,0,1]]],labels=[0],initial=None)
    if change=='dimensions':request['plan']['sizes']=[17,4,2]
    if change=='threshold':request['plan']['threshold']=[0,4]
    if change=='shift':request['plan']['synaptic_shift']=0
    if change=='reset':request['plan']['detach_reset']=False
    if change=='weight_scale':request['plan']['weight_scale']=3
    if change=='weights':request['state']['weights'][0][0]=1000
    if change=='moments':request['state']['second_moment'][0][0]=-1
    if change=='budget':request['plan']['max_tape_bytes']=1
    if change=='inputs':request['inputs'][0][0][0]=.5
    if change=='steps':request['inputs']=[[[1,0,1]]]*0;request['labels']=[]
    if change=='labels':request['labels']=[2]
    if change=='extension':request['plan']['extra']=True
    original=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer._run_request(request)
    assert trainer.state==original


def _linearized_loss(plan, weights, perturbed, x, labels):
    """Finite differences of the declared local surrogate graph, not hard spikes.

    Each scalar op executes a pair: exact integer nominal, continuous perturbation.
    This forward-only oracle independently checks the native reverse pass.
    """
    scale=plan['weight_scale'];sizes=plan['sizes'];cells=sum(sizes[1:]);batch,steps,_=x.shape
    shapes=list(zip(sizes,sizes[1:]))+([(sizes[1],sizes[1])] if plan.get('recurrent') else [])
    w0=[np.rint(np.asarray(w)*scale).reshape(a,b) for w,(a,b) in zip(weights,shapes)]
    wp=[w+scale*(np.asarray(new)-old).reshape(w.shape) for w,new,old in zip(w0,perturbed,weights)]
    logits=[];rates=[]
    def clip(z,zp):return np.clip(z,-32768,32767),np.clip(z,-32768,32767)+((-32768<=z)&(z<=32767))*(zp-z)
    def decay(z):return np.floor(z/2)+(z==1)
    for row in x:
        i0=np.zeros(cells);ip=i0.copy();v0=i0.copy();vp=i0.copy();q0=i0.copy();qp=i0.copy();counts=np.zeros(sizes[-1])
        for time,tick in enumerate(row):
            new_q0=np.zeros(cells);new_qp=np.zeros(cells)
            for l in range(2):
                offset=0 if l==0 else sizes[1];n=sizes[l+1];cap=31 if l==0 else 1;theta=plan['threshold'][l]
                a0=tick if l==0 else q0[:sizes[1]];ap=tick if l==0 else qp[:sizes[1]]
                for j in range(n):
                    k=offset+j;s0=i0[k];sp=ip[k]
                    for src in range(sizes[l]):s0,sp=clip(s0+w0[l][src,j]*a0[src],sp+wp[l][src,j]*ap[src])
                    if l==0 and plan.get('recurrent'):
                        for src in range(sizes[1]):
                            s0,sp=clip(s0+w0[2][src,j]*q0[src],sp+wp[2][src,j]*qp[src])
                    ni=s0-decay(s0);nip=ni+.5*(sp-s0)
                    delta=-decay(v0[k])+ni;deltap=delta-.5*(vp[k]-v0[k])+nip-ni
                    delta,deltap=clip(delta,deltap)
                    u,up=clip(v0[k]+delta,vp[k]+deltap)
                    q=np.clip(np.floor(u/theta),0,cap)
                    derivative=sum(plan['surrogate']['scale']/(1+plan['surrogate']['slope']*abs(u/theta-nn))**2/theta for nn in range(1,cap+1))
                    qpert=q+derivative*(up-u)
                    i0[k]=ni;ip[k]=nip;v0[k]=u-q*theta;vp[k]=up-q*theta
                    new_q0[k]=q;new_qp[k]=qpert
            q0,qp=new_q0,new_qp
            if time>=plan.get('readout_start',0):
                counts+=(ip[-sizes[-1]:] if plan.get('readout_source') else qp[-sizes[-1]:])
        rates.append(counts/(steps-plan.get('readout_start',0)))
        if 'class_codes' in plan:
            counts = np.asarray(plan['class_codes']) @ counts
        score=counts*plan['logit_scale']/(steps-plan.get('readout_start',0))
        if plan.get('readout_source'):score/=plan['threshold'][1]
        logits.append(score)
    logits=np.asarray(logits);maximum=logits.max(1)
    loss=np.mean(np.log(np.exp(logits-maximum[:,None]).sum(1))+maximum-logits[np.arange(batch),labels])
    if 'code_rate_weight' in plan:
        target=.1+.8*np.asarray(plan['class_codes'])[labels]
        loss+=plan['code_rate_weight']*np.mean((np.asarray(rates)-target)**2)
    return loss


def test_reverse_pass_matches_surrogate_directional_derivative(trainer):
    x=np.random.default_rng(4).integers(0,2,(2,7,3));y=[0,1]
    r=trainer.gradients(x,y);weights=trainer.state['weights']
    np.testing.assert_allclose(_linearized_loss(trainer.plan,weights,weights,x,y),r['loss'],atol=1e-12)
    rng=np.random.default_rng(21)
    for _ in range(3):
        direction=[rng.normal(size=len(w)) for w in weights];eps=1e-5
        plus=[np.asarray(w)+eps*d for w,d in zip(weights,direction)]
        minus=[np.asarray(w)-eps*d for w,d in zip(weights,direction)]
        finite=(_linearized_loss(trainer.plan,weights,plus,x,y)-_linearized_loss(trainer.plan,weights,minus,x,y))/(2*eps)
        reverse=sum(np.dot(g,d) for g,d in zip(r['gradients'],direction))
        np.testing.assert_allclose(reverse,finite,rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize('case', ['learned','saturation','rounding_decay'])
def test_exact_integer_native_atlas_and_sdk(trainer,tmp_path,case):
    if not RUN.is_file() or not SDK:pytest.skip('set B2_RUNNER and B2_XYLO_PYTHON')
    if case=='saturation':
        p=xylo_training_plan([16,64,4],thresholds=(1,1),weight_scale=2.)
        wi=np.full((16,64),127.);wi[:,32:]=-128.
        wo=np.full((64,4),127.);wo[:,1]=-128.;wo[:16,2]=-128.;wo[16:,3]=-128.
        trainer=NativeXyloTrainer(p,runner=TRAIN,weights=[wi.ravel()/2,wo.ravel()/2])
    elif case=='rounding_decay':
        p=xylo_training_plan([3,4,2],thresholds=(1,2),weight_scale=2.)
        trainer=NativeXyloTrainer(p,runner=TRAIN,
            weights=[[.25,.75,1.25,-.75,1.75,-1.75,.5,-.5,-.25,-1.25,0,0], [1,-1]*4])
    sizes=trainer.plan['sizes'];cells=sum(sizes[1:]);hidden=sizes[1]
    x=np.random.default_rng(8).integers(0,2,(4,16,sizes[0]),dtype=np.uint8);x[0]=1;x[1]=0;x[2]=x[0]
    # Silence gives a real zero-gradient training step for boundary fixtures.
    trainer.step(x if case=='learned' else np.zeros_like(x),[0,1,0,1]);f=freeze_native_xylo(trainer)
    if case=='rounding_decay':
        assert f['target_weights'][0]==[0,2,2,-2,4,-4,1,-1,0,-2,0,0]
    native=trainer.evaluate(x,[0,1,0,1])
    b.prefs.codegen.target='numpy';previous=b.prefs.codegen.loop_invariant_optimisations
    b.prefs.codegen.loop_invariant_optimisations=False
    try:
        bundle=export_native_xylo(f,tmp_path/'export',steps=16,runner=RUN)
        assert bundle['native_training_bridge']['dynamics_equivalent'] is True
        for i in range(len(x)):
            m=export_network(build_native_xylo_network(f,x[i]),16*b.ms,tmp_path/f'source-{i}.json')
            subprocess.run([str(RUN),str(tmp_path/f'source-{i}.json'),str(tmp_path/f'replay-{i}')],check=True,capture_output=True)
            recorded=load_results(m,tmp_path/f'replay-{i}')
            for role,sl in (('hidden',slice(0,hidden)),('output',slice(hidden,cells))):
                trace=recorded['populations'][bundle['plan']['roles'][role]]['trace']
                for key,field in (('I','currents'),('v','membranes'),('q','spikes')):
                    np.testing.assert_array_equal(trace[key],np.asarray(native[field])[i,:,sl])
    finally:b.prefs.codegen.loop_invariant_optimisations=previous
    runtime=Path(__file__).resolve().parents[1]/'python/brian2_rust/neuromorphic_runtime.py'
    np.save(tmp_path/'inputs.npy',x)
    script="""import importlib.util,json,numpy as np,sys
s=importlib.util.spec_from_file_location('runtime',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=m.run_simulator(json.load(open(sys.argv[2])),np.load(sys.argv[3]))
assert r['dynamics_equivalent'] and r['configuration_valid']
np.savez(sys.argv[4],**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
"""
    subprocess.run([SDK,'-c',script,str(runtime),str(tmp_path/'export/converted/bundle.json'),str(tmp_path/'inputs.npy'),str(tmp_path/'sdk.npz')],
        cwd=tmp_path,env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},check=True,capture_output=True)
    sdk=np.load(tmp_path/'sdk.npz')
    for key,field in (('spike_raster','spikes'),('voltage','membranes')):np.testing.assert_array_equal(sdk[key],native[field])
    np.testing.assert_array_equal(np.concatenate((sdk['hidden_I'],sdk['output_I']),axis=2),native['currents'])
    np.testing.assert_array_equal(sdk['logits'],native['logits'])
    assert sdk['spike_raster'][0].sum()>0 and sdk['spike_raster'][1].sum()==0
    np.testing.assert_array_equal(sdk['spike_raster'][0],sdk['spike_raster'][2])

    if case=='saturation':
        assert sdk['hidden_q'].max()==31
        assert sdk['output_q'].max()==1
        assert np.abs(sdk['output_I']).max()>=16383
    if case=='rounding_decay':
        # Sum 3 decays to 2; equality at the threshold produces two spikes.
        assert sdk['hidden_I'][0,0,2]==2 and sdk['hidden_q'][0,0,2]==2
        assert sdk['hidden_I'][0,0,3]==-1


def test_freeze_and_export_validation_without_sdk(trainer,tmp_path):
    from brian2_rust.native_xylo_conversion import digest,validate_frozen,validate_inputs
    with pytest.raises(ValueError,match='trained'):freeze_native_xylo(trainer)
    trainer.step(np.ones((1,4,3)),[0])
    before=copy.deepcopy(trainer.state);f=freeze_native_xylo(trainer)
    assert trainer.state==before
    for original,quantized in zip(before['weights'],f['target_weights']):
        np.testing.assert_array_equal(np.rint(np.asarray(original)*8),quantized)
    changed=copy.deepcopy(f);changed['target_weights'][0][0]+=1
    with pytest.raises(ValueError,match='integrity'):validate_frozen(changed)
    for field,value in (('synaptic_shift',0),('sizes',[17,4,2]),('threshold',[0,4]),('backend','metal')):
        changed=copy.deepcopy(f);changed['plan'][field]=value
        changed['sha256']=digest({k:v for k,v in changed.items() if k!='sha256'})
        with pytest.raises(ValueError):validate_frozen(changed)
    with pytest.raises(ValueError,match='binary'):validate_inputs(f,np.full((1,4,3),.5))
    for steps in (0,129,True):
        with pytest.raises(ValueError,match='duration'):export_native_xylo(f,tmp_path/str(steps),steps=steps,runner=RUN)
    assert not list(tmp_path.iterdir())
    if RUN.is_file():
        bundle=export_native_xylo(f,tmp_path/'valid',steps=4,runner=RUN)
        assert bundle['profile']=='xylo-audio2-temporal-v1'
        assert bundle['native_training_bridge']['dynamics_equivalent'] is True
        for name,bank,n,m in zip(('input','readout'),f['target_weights'],[3,4],[4,2]):
            np.testing.assert_array_equal(bundle['plan']['weights'][name],np.asarray(bank).reshape(n,m))
