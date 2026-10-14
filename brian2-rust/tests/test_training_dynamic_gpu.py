"""Real v5 GPU execution against CPU and independent dynamic equations."""
import copy
import os
import shutil
import tempfile
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_dynamic import model,noisy_model,oracle


@pytest.fixture(scope='module',params=['metal','cuda'])
def backend(request):
    backend=request.param;flag='B2_TEST_GPU' if backend=='metal' else 'B2_TEST_CUDA_TRAIN'
    if os.environ.get(flag)!='1':pytest.skip('actual dynamic GPU required')
    from brian2_rust import training_metal,training_cuda
    module=training_metal if backend=='metal' else training_cuda
    # Compile the actual library once, then copy it to each trainer's isolated
    # lifetime. Execution, device buffers and shader math are never mocked.
    with tempfile.TemporaryDirectory(prefix='b2-dynamic-gpu-tests-') as directory:
        library=module.build(directory)
        def cached(target):return shutil.copy2(library,os.path.join(target,library.name))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module,'build',lambda target:__import__('pathlib').Path(cached(target)))
            yield backend


@pytest.fixture(scope='module',autouse=True)
def cython_cache():
    old=b.prefs.codegen.runtime.cython.cache_dir
    with tempfile.TemporaryDirectory(prefix='b2-dynamic-gpu-cython-') as directory:
        b.prefs.codegen.runtime.cython.cache_dir=directory
        try:yield
        finally:b.prefs.codegen.runtime.cython.cache_dir=old


def compare(actual,expected,backend):
    assert actual['backend']==backend and actual['gpu_dispatches']>0
    assert 'dynamic-actions-f32' in actual['numeric_profile']
    np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
    assert actual['loss']==pytest.approx(expected['loss'],abs=3e-6)
    for key in ('final_state','initial_state_gradients','final_membrane','initial_gradients','logits'):
        np.testing.assert_allclose(actual[key],expected[key],rtol=3e-4,atol=6e-6,err_msg=key)
    for a,c in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,c,rtol=1e-3,atol=3e-5)


@pytest.mark.parametrize('event',[False,True])
@pytest.mark.parametrize('detach,window',[(True,None),(False,3)])
@pytest.mark.parametrize('batch',[1,2])
@pytest.mark.parametrize('explicit',[False,True])
def test_dynamic_gpu_all_states_vjp(event,detach,window,batch,explicit,backend):
    p,w,x=model(detach,window,event);data=np.stack([x,x[:,::-1]][:batch])
    kw=dict(initial=np.tile(p['dynamic']['initial'],(batch,1))) if explicit else {}
    cpu=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(data,[0]*batch,**kw)
    p['backend']=backend;actual=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(data,[0]*batch,**kw)
    compare(actual,cpu,backend)


@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
def test_gpu_native_noise_and_independent_parameter_derivatives(detach,window,backend):
    p,w,x=noisy_model(detach,window);sequence=11;tick=2
    p['backend']=backend
    actual=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None],[0],noise_sequence=sequence,start_tick=tick)
    def reference(weights,anchors=None):return oracle(p,weights,x,anchors=anchors,sequence=sequence,start_tick=tick)
    loss,spikes,live,anchors=reference(w)
    assert actual['loss']==pytest.approx(loss,abs=3e-6)
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    np.testing.assert_allclose(actual['final_state'][0],live,rtol=3e-4,atol=6e-6)
    for bank,row in enumerate(w):
        for j,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);a=copy.deepcopy(w);c=copy.deepcopy(w);a[bank][j]+=eps;c[bank][j]-=eps
            fd=(reference(a,anchors)[0]-reference(c,anchors)[0])/(2*eps)
            assert actual['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=3e-5)


@pytest.mark.parametrize('event',[False,True])
@pytest.mark.parametrize('window',[None,3])
def test_gpu_delayed_history_independent_initial_state_derivatives(event,window,backend):
    from test_training_delays import model as delayed,oracle as delayed_oracle
    *_,x,bundle=delayed(event,.8,changed=True,post_first=True,order_sensitive=True)
    p=bundle.plan;p['backend']=backend;p['tbptt_window']=window
    initial=np.array(bundle.initial_state);kw=dict(post_first=True,order_sensitive=True)
    result=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],initial=initial[None])
    loss,spikes,live,anchors=delayed_oracle(bundle,bundle.weights,x,initial=initial,**kw)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    assert np.max(abs(np.asarray(result['initial_state_gradients'])[0,p['dynamic']['binary_states']]))>1e-7
    for j,value in enumerate(initial):
        actual=result['initial_state_gradients'][0][j]
        if p['dynamic']['detached'][j]:assert actual==0;continue
        a=initial.copy();c=initial.copy();a[j]+=1e-6;c[j]-=1e-6
        fd=(delayed_oracle(bundle,bundle.weights,x,initial=a,anchors=anchors,**kw)[0]-
            delayed_oracle(bundle,bundle.weights,x,initial=c,anchors=anchors,**kw)[0])/2e-6
        assert actual==pytest.approx(fd,rel=1e-3,abs=2e-6)


def frontend(kind,backend='cpu'):
    if kind in ('delay','pending'):
        from test_training_delays import model as delayed
        *_,x,bundle=delayed(True,.8 if kind=='pending' else 0,changed=True,post_first=True,order_sensitive=True)
    elif kind=='clock':
        from test_training_dynamic_clocks import model as clocked
        *_,x,dt,bundle=clocked(.99999,.3,third=True)
    elif kind=='summed_noise':
        from test_training_summed import model as summed,lower
        net,inp,groups,static,syn,x,dt=summed('heun',True,noise=True)
        syn.pre.delay=[.6,.2,.4,0]*b.ms
        bundle=lower(net,inp,groups,backend=backend)
    else:
        from test_training_dynamic_refractory import plastic_model
        net,inp,groups,x,dt,plastic=plastic_model(('v','a') if kind=='refractory' else ('v',),3.5,'rk4')
        bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=backend,
            trainable_neuron_parameters={g.name:['tau','theta','kick'] for g in groups})
    bundle.plan['backend']=backend
    return bundle,x


@pytest.mark.parametrize('kind',['delay','pending','clock','summed_noise','refractory','selective'])
@pytest.mark.parametrize('window',[None,3])
def test_frontend_dynamic_gpu_combinations(kind,window,backend):
    bundle,x=frontend(kind,backend);p=bundle.plan;p['tbptt_window']=window
    data=np.stack([x,x[:,::-1]]);kw=dict(noise_sequence=71) if p.get('noise_streams') is not None else {}
    gpu=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients(data,[0,1],**kw)
    p=copy.deepcopy(p);p['backend']='cpu'
    cpu=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients(data,[0,1],**kw)
    compare(gpu,cpu,backend)


@pytest.mark.parametrize('kind',['pending','clock','summed_noise'])
def test_gpu_carry_migration_checkpoint_and_optimizer(kind,backend,tmp_path):
    bundle,x=frontend(kind,backend);p=bundle.plan
    gpu=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    cp=copy.deepcopy(p);cp['backend']='cpu';cpu=NativeLIFTrainer(cp,runner=RUNNER,weights=bundle.weights)
    for start,stop in [(0,2),(2,4),(4,len(x))]:
        a=gpu.execute(x[None,start:stop],[0],**(dict(initial='carry') if start else {}))
        c=cpu.execute(x[None,start:stop],[0],**(dict(initial='carry') if start else {}))
        compare(a,c,backend)
        for key in ['weights','first_moment','second_moment']:
            for row,other in zip(gpu.state[key],cpu.state[key]):np.testing.assert_allclose(row,other,rtol=2e-3,atol=4e-5)
        assert gpu.state['step']==cpu.state['step'] and gpu.clock_tick==cpu.clock_tick and gpu.noise_sequence==cpu.noise_sequence
        if stop==len(x):break
        controlled=p['dynamic']['migration']['controlled_masks'][-1];bank,index=controlled
        for trainer in (gpu,cpu):
            masks=copy.deepcopy(trainer.plan['masks']);masks[bank][index]=0 if stop==2 else 1
            before=(trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence,trainer.state['step'])
            trainer.update_mask(masks,growth_weight=.4)
            assert before==(trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence,trainer.state['step'])
        filename=tmp_path/'gpu.json';gpu.store(filename)
        restored=NativeLIFTrainer(gpu.plan,runner=RUNNER);restored.restore(filename);gpu=restored


@pytest.mark.parametrize('active',[False,True])
def test_gpu_lazy_guard_skips_invalid_unselected_branch(active,backend):
    from brian2_rust.training_dynamic import compile_dynamic_transform
    p,w,x=model();tr=compile_dynamic_transform('v=log(v)',states={'v':0,'guard':1},write_guards={0:1})
    d=p['dynamic'];g=len(d['initial']);d['initial'].append(float(active));d['initial_parameters'].append(None);d['detached'].append(True)
    d['initial'][0]=-1;d['program_sets'][0]=tr['programs'];d['actions'][0]['reads']=[0,g]
    p['backend']=backend;trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    if active:
        with pytest.raises(ValueError,match='nonfinite'):trainer.gradients(x[None,:1],[0])
        assert trainer.state==before and trainer.clock_tick==0
    else:
        gpu=trainer.gradients(x[None,:1],[0]);p['backend']='cpu'
        cpu=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None,:1],[0]);compare(gpu,cpu,backend)


@pytest.mark.parametrize('issue',['budget','nonfinite','binary','mpi'])
def test_gpu_failure_preserves_training_state(issue,backend):
    p,w,x=model();p['backend']=backend
    if issue=='budget':
        cp=copy.deepcopy(p);cp['backend']='cpu';minimum=NativeLIFTrainer(cp,runner=RUNNER,weights=w).evaluate(x[None],[0])['tape_bytes']
        p['max_tape_bytes']=minimum
    elif issue=='nonfinite':w[0][0]=1e100
    elif issue=='mpi':p['mpi_ranks']=1
    # MPI admission is checked without starting an MPI launcher or GPU work.
    if issue=='mpi':
        with pytest.raises(ValueError,match='2..256'):NativeLIFTrainer(p,runner=RUNNER,weights=w)
        return
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick))
    if issue=='binary':x=x.copy();x[0,0]=.5
    with pytest.raises(ValueError):trainer.execute(x[None],[0])
    assert (trainer.state,trainer.neuron_state,trainer.clock_tick)==before
