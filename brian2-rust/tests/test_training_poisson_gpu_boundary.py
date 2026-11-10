"""Full device zero-rate trajectories with independent hard-loss oracles."""
import copy
import math
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from brian2_rust.training import NativeLIFTrainer
from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero import zero_model,hard_loss,stream_draw
from test_training_poisson_zero_vjp import mpi


def run(plan,weights,engine,x,labels,**kw):
    plan['backend']=engine
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    result=trainer.gradients(x,labels,**kw)
    assert result['backend']==engine
    if engine!='cpu':assert result['gpu_dispatches']>=2
    return trainer,result


@pytest.mark.parametrize('length',[1,4])
@pytest.mark.parametrize('window',[None,1,2])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('ranks',[None,2])
def test_full_device_weak_loss_oracle(engine,length,window,batch,ranks):
    mpi(ranks);plan,weights=zero_model(ranks=ranks);plan['tbptt_window']=window
    labels=np.arange(batch)%2
    _,r=run(plan,weights,engine,np.zeros((batch,length,1)),labels)
    expected=sum((hard_loss(plan['logit_scale']*(length-t)/length,labels)-math.log(2)).mean() for t in range(length))
    assert r['gradients'][0][0]==pytest.approx(expected,rel=3e-5,abs=2e-6)
    assert r['gradients'][0][1]==0.
    np.testing.assert_array_equal(np.asarray(r['final_state'])[:,2],0.)
    if engine!='cpu' and ranks is None:assert r['gpu_dispatches']==2+batch*length


@pytest.mark.parametrize('window',[None,1,2])
@pytest.mark.parametrize('binding',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_boundary_state_vjp_and_tbptt_on_baseline(engine,window,binding,ranks):
    mpi(ranks);p,w=zero_model('k=draw(scale*r)\nv=v+amp*k\nr=.8*r',ranks)
    p['tbptt_window']=window;w[0][0]=1.2;w[0][2]=0.;p['dynamic']['initial'][4]=0.
    if binding:p['dynamic']['initial_parameters'][4]=[0,2]
    _,r=run(p,w,engine,np.zeros((1,4,1)),[0])
    expected=sum(1.2*.8**t*(hard_loss(p['logit_scale']*(4-t)/4)-math.log(2)) for t in range(4 if window is None else window))
    assert r['initial_state_gradients'][0][4]==pytest.approx(expected,rel=3e-5,abs=2e-6)
    assert r['gradients'][0][2]==pytest.approx(expected if binding else 0.,rel=3e-5,abs=2e-6)
    assert r['gradients'][0][0]==0.


@pytest.mark.parametrize('ranks',[None,2])
def test_nested_live_rates_preserve_full_batch_counter_keys(engine,ranks):
    mpi(ranks);p,w=zero_model('k=draw(scale)\nv=amp*other(k+scale)',ranks)
    batch=24;sequence=2**64-7;labels=np.arange(batch)%2
    _,r=run(p,w,engine,np.zeros((batch,1,1)),labels,noise_sequence=sequence)
    counts=np.array([stream_draw(1.,p['seed'],sequence,b,1) for b in range(batch)])
    assert np.any(counts==0) and np.any(counts>0)
    expected=(hard_loss(p['logit_scale']*(counts>0),labels)+hard_loss(p['logit_scale'],labels)-2*math.log(2)).mean()
    assert r['gradients'][0][0]==pytest.approx(expected,rel=3e-5,abs=2e-6)


@pytest.mark.parametrize('code',[
    'k=draw(scale)\nv=amp*other(scale) if k > 0 else 0.',
    'k=draw(scale)\nv=amp*other(-1.) if k < 0 else 0.',
])
@pytest.mark.parametrize('ranks',[None,2])
def test_unreached_zero_sites_do_not_create_boundary_terms(engine,code,ranks):
    mpi(ranks);p,w=zero_model(code,ranks)
    _,r=run(p,w,engine,np.zeros((1,1,1)),[0])
    np.testing.assert_array_equal(r['gradients'],[[0.,0.,0.]])


@pytest.mark.parametrize('ranks',[None,2,8])
def test_counterfactual_changes_indirect_destination(engine,ranks):
    mpi(ranks);p,w=zero_model('k=draw(scale)\nv=amp',ranks)
    p['dynamic']['actions'][0]['indirect']={'reads':{},'writes':{'0':{'index':{'kind':'output','slot':1},'tables':[[4,0]]}}}
    _,r=run(p,w,engine,np.zeros((2,1,1)),[0,1])
    expected=np.mean(hard_loss(p['logit_scale'],np.array([0,1]))-math.log(2))
    assert r['gradients'][0][0]==pytest.approx(expected,rel=3e-5,abs=2e-6)
    np.testing.assert_array_equal(np.asarray(r['final_state'])[:,2],0.)


@pytest.mark.parametrize('ranks',[None,2])
def test_event_replay_checkpoint_and_baseline_commit(engine,ranks,tmp_path):
    mpi(ranks);p,w=zero_model('k=draw(scale)\nv=amp*k',ranks);p['backend']=engine;p['trainable']=[False]
    tr=compile_dynamic_transform('v=v+amp',states={'v':0},parameters={'amp':(0,1)})
    p['dynamic']['program_sets'].append(tr['programs'])
    event=dynamic_action(tr,[1],owner=2,program_set=1)
    event.update(trigger={'external':False,'index':1},detach_trigger=True)
    p['dynamic']['actions'].insert(2,event)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    trainer.step(np.zeros((1,2,1)),[0],noise_sequence=2**64-7)
    path=tmp_path/'checkpoint.json';trainer.store(path)
    restored=NativeLIFTrainer(p,runner=RUNNER);restored.restore(path)
    actual=restored.gradients(np.zeros((1,2,1)),[0],initial='carry')
    same=trainer.gradients(np.zeros((1,2,1)),[0],initial='carry')
    for key in ('gradients','initial_state_gradients','final_state','spikes','loss'):
        np.testing.assert_array_equal(actual[key],same[key])
    expected=hard_loss(-p['logit_scale']/2)-math.log(2)
    assert actual['gradients'][0][0]==pytest.approx(expected,rel=3e-5,abs=2e-6)
    assert actual['final_tick']==4 and actual['noise_sequence']==2**64-7
    assert actual['state']['weights']==w


@pytest.mark.parametrize('ranks',[None,2])
def test_async_idle_visits_replayed_on_device(engine,ranks):
    mpi(ranks);p,w=zero_model(ranks=ranks)
    p['dynamic']['clocks']={'start':0.,'dts':[.001,.0005],'epsilon':1e-4,'order':[0,1]}
    p['dynamic']['actions'][0]['clock']=1
    _,r=run(p,w,engine,np.zeros((1,2,1)),[0])
    expected=hard_loss(p['logit_scale'])-math.log(2)+2*(hard_loss(p['logit_scale']/2)-math.log(2))
    assert r['gradients'][0][0]==pytest.approx(expected,rel=3e-5,abs=2e-6)


@pytest.mark.parametrize('code',['k=draw(scale)\nv=1./(1.-k)',
                                'k=draw(scale)\nv=amp*other(-1.) if k > 0 else 0.'])
@pytest.mark.parametrize('ranks',[None,2])
def test_invalid_alternate_trajectory_rolls_back_every_cursor(engine,code,ranks):
    mpi(ranks);p,w=zero_model(code,ranks);p['backend']=engine
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    before=copy.deepcopy(trainer.state);trainer.evaluate(np.zeros((1,1,1)),[0])
    with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0


@pytest.mark.parametrize('ranks',[None,2])
def test_boundary_budget_includes_coefficients_and_replay_masks(engine,ranks):
    mpi(ranks);p,w=zero_model(ranks=ranks)
    _,r=run(p,w,engine,np.zeros((2,3,1)),[0,1])
    p['max_tape_bytes']=r['tape_bytes']-1;p['backend']=engine
    with pytest.raises(ValueError,match='budget'):
        NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((2,3,1)),[0,1])


def test_sgd_uses_boundary_gradient_and_commits_only_baseline(engine):
    p,w=zero_model();p['backend']=engine;p['optimizer'].update(kind='sgd',learning_rate=.05)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    r=trainer.step(np.zeros((1,1,1)),[0],noise_sequence=19)
    expected=hard_loss(p['logit_scale'])-math.log(2)
    assert trainer.state['weights'][0][0]==pytest.approx(-.05*expected,rel=3e-5,abs=2e-6)
    assert r['final_state'][0][2]==0.
    assert trainer.clock_tick==1 and trainer.state['step']==1 and trainer.next_noise_sequence==20


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('version',[None,0,2])
def test_missing_boundary_capability_fails_before_gpu_dispatch(backend,version,tmp_path,monkeypatch):
    p,w=zero_model();p['backend']=backend
    source=tmp_path/'old.c';library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\nuint64_t b2_train_poisson_shared_v1(void){return 1;}\nuint64_t b2_train_poisson_persistent_v1(void){return 1;}\n'
        'uint64_t b2_train_poisson_v1(void){return 1;}\nuint64_t b2_train_poisson_vjp_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_poisson_boundary_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    import importlib
    module=importlib.import_module('brian2_rust.training_'+backend)
    monkeypatch.setattr(module,'build',lambda directory:library)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='GPU Poisson boundary replay capability'):
        trainer.gradients(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0


@pytest.mark.parametrize('ranks',[None,2])
def test_active_boundary_seed_preserves_positive_scores_and_pathwise_vjp(engine,ranks):
    mpi(ranks);p,w=zero_model('k=draw(scale)\nv=amp*(k+other(r))')
    x=np.zeros((4,3,1));labels=[0,1,0,1]
    reference=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,labels,noise_sequence=9)
    p['mpi_ranks']=ranks
    _,r=run(p,w,engine,x,labels,noise_sequence=9)
    for key in ('loss','spikes','final_state','gradients','initial_state_gradients'):
        np.testing.assert_allclose(r[key],reference[key],rtol=4e-5,atol=3e-6)
    assert abs(reference['gradients'][0][0])>1e-4
    assert np.any(np.abs(np.asarray(reference['initial_state_gradients']))>1e-4)


@pytest.mark.parametrize('backend',['metal','cuda'])
def test_needed_boundary_missing_coefficient_fails_and_rolls_back(backend,tmp_path,monkeypatch):
    # Deliberately corrupt only the final reverse-call ready mask in a native
    # ABI wrapper. Collection and alternate trajectories execute on real GPU.
    import importlib,json
    flag='B2_TEST_GPU' if backend=='metal' else 'B2_TEST_CUDA_TRAIN'
    if os.environ.get(flag)!='1':pytest.skip('actual '+backend+' hardware required')
    module=importlib.import_module('brian2_rust.training_'+backend)
    original=module.build(tmp_path)
    wrapper=tmp_path/'wrapped.so';source=tmp_path/'wrapped.c'
    symbol='b2_train_'+backend+'_v5r9'
    arguments='const uint64_t* m,const float* p,const float* x,const float* w,const float* initial,const uint64_t* y,float* tape,float* spikes,float* live,float* g,float* adj,float* logits,float* losses,char* msg,size_t cap'
    source.write_text('#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n#include <dlfcn.h>\n'+
        'uint64_t b2_train_math_v1(void){return 1;}\nuint64_t b2_train_poisson_shared_v1(void){return 1;}\nuint64_t b2_train_poisson_persistent_v1(void){return 1;}\nuint64_t b2_train_poisson_v1(void){return 1;}\n'
        'uint64_t b2_train_poisson_vjp_v1(void){return 1;}\nuint64_t b2_train_poisson_boundary_v1(void){return 1;}\n'
        'uint64_t b2_train_vjp_activity_v1(void){return 1;}\n'+
        'int '+symbol+'('+arguments+'){\n'
        'void* h=dlopen('+json.dumps(str(original))+',RTLD_NOW);if(!h)return 99;\n'
        'typedef int(*Kernel)('+arguments+');Kernel fn=(Kernel)dlsym(h,"'+symbol+'");\n'
        'float* q=malloc(m[12]*4);memcpy(q,p,m[12]*4);\n'
        'if(m[9]==1)for(uint64_t b=0;b<m[0];b++)for(uint64_t t=0;t<m[1];t++)for(uint64_t a=0;a<m[7];a++){\n'
        'uint64_t s=m[13]+16*a;if(m[s+5]&&(m[m[s+5]]&512))q[m[24]+(b*m[1]+t)*m[25]+m[s+12]+96]=0;\n}\n'
        'int code=fn(m,q,x,w,initial,y,tape,spikes,live,g,adj,logits,losses,msg,cap);free(q);dlclose(h);return code;\n}\n')
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(wrapper)],check=True,capture_output=True)
    monkeypatch.setattr(module,'build',lambda directory:wrapper)
    p,w=zero_model();p['backend']=backend
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='nonfinite dynamic GPU result'):
        trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0


@pytest.mark.parametrize('ranks',[None,2,8])
def test_distinct_actions_and_owners_have_separate_replay_coefficients(engine,ranks):
    from brian2_rust.training_equations import PoissonNoise
    mpi(ranks);p,w=zero_model(ranks=ranks);w[0][2]=0.
    tr=compile_dynamic_transform('v=v+amp*draw(scale)',states={'v':0},
        parameters={'draw':PoissonNoise(0),'scale':(0,2),'amp':(0,1)})
    p['dynamic']['program_sets'].append(tr['programs'])
    p['dynamic']['actions'].insert(1,dynamic_action(tr,[1],owner=2,program_set=1,
        noise_domain=72,noise_entity=0,noise_streams=1))
    _,r=run(p,w,engine,np.zeros((1,3,1)),[0])
    first=sum(hard_loss(p['logit_scale']*(3-t)/3)-math.log(2) for t in range(3))
    second=sum(hard_loss(-p['logit_scale']*(3-t)/3)-math.log(2) for t in range(3))
    np.testing.assert_allclose(r['gradients'][0],[first,0.,second],rtol=4e-5,atol=3e-6)
    np.testing.assert_array_equal(np.asarray(r['final_state'])[:,2:4],0.)
    if engine!='cpu' and ranks is None:assert r['gpu_dispatches']==8
