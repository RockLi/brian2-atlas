"""Actual multi-state GPU/MPI execution; no CPU fallback is admissible."""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,compile_training_equation
from test_native_training import RUNNER
from test_native_training_multistate import fixture
from test_native_training_gpu_mpi import compare


@pytest.fixture(params=['metal','cuda'])
def gpu_backend(request):
    flag='B2_TEST_GPU' if request.param=='metal' else 'B2_TEST_CUDA_TRAIN'
    if os.environ.get(flag)!='1':pytest.skip('real GPU required')
    return request.param


def state_compare(a,b):
    compare(a,b)
    for name in ['initial_state_gradients','final_state']:
        np.testing.assert_allclose(a[name],b[name],atol=6e-6,rtol=3e-4)
    assert b['gpu_dispatches']>1 and b['backend'] in ('metal','cuda')


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('reset',['zero','subtract'])
@pytest.mark.parametrize('detach',[True,False])
@pytest.mark.parametrize('window',[None,3])
def test_multistate_gpu_all_vjps(gpu_backend,ranks,reset,detach,window):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    p,w,x,y,initial=fixture(reset,detach,window)
    cpu=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    p['backend']=gpu_backend
    if ranks:p['mpi_ranks']=ranks
    gpu=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    for step in range(2):
        a=cpu.step(x,y,initial=initial if step==0 else 'carry')
        b=gpu.step(x,y,initial=initial if step==0 else 'carry')
        state_compare(a,b)
        assert b['gpu_dispatches']==3+4*x.shape[1]


def test_multistate_gpu_evaluate_empty_owners(gpu_backend):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    p,w,x,y,initial=fixture();cpu=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    p.update(backend=gpu_backend,mpi_ranks=8);gpu=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    a=cpu.evaluate(x,y,initial=initial);b=gpu.evaluate(x,y,initial=initial)
    state_compare(a,b)
    assert b['gpu_dispatches']==3+2*x.shape[1] and not np.any(b['initial_state_gradients'])
    state_compare(cpu.step(x,y,initial=initial),gpu.step(x,y,initial=initial))


def test_multistate_gpu_fresh_process_resume(gpu_backend,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    p,w,x,y,initial=fixture();p.update(backend=gpu_backend,mpi_ranks=2);p['trainable'][1]=False
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);trainer.step(x,y,initial=initial)
    masks=copy.deepcopy(p['masks']);masks[0][0]=0;trainer.update_mask(masks)
    path=tmp_path/'checkpoint';trainer.store(path)
    source=tmp_path/'request';source.write_text(json.dumps(dict(plan=trainer.plan,x=x.tolist(),y=y)))
    output=tmp_path/'result'
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
p=json.load(open(sys.argv[1]));t=NativeLIFTrainer(p['plan'],runner=sys.argv[4]);t.restore(sys.argv[2])
json.dump(t.step(p['x'],p['y'],initial='carry'),open(sys.argv[3],'w'))
'''
    subprocess.run([sys.executable,'-c',code,str(source),str(path),str(output),str(RUNNER)],check=True,timeout=120)
    result=trainer.step(x,y,initial='carry')
    assert json.loads(output.read_text())==result
    assert result['state']['weights'][1]==w[1] and len(trainer.neuron_state[0])==10
    assert result['state']['weights'][0][0]==result['state']['first_moment'][0][0]==0


@pytest.mark.parametrize('ranks',[None,2])
def test_multistate_gpu_intermediate_domain_error(gpu_backend,ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    p,w,x,y,initial=fixture();p['backend']=gpu_backend
    if ranks:p['mpi_ranks']=ranks
    # Only the first tick is invalid; subsequent auxiliary updates heal the
    # state. The GPU must retain the earlier error instead of returning success.
    p['state_equations'][0]=[compile_training_equation('log(a)',states=['v','a']),
                              compile_training_equation('1',states=['v','a'])]
    initial[:,2:4]=-1
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='nonfinite'):trainer.step(x,y,initial=initial)
    assert trainer.state==before and trainer.neuron_state is None and trainer.elapsed_ticks==0


def test_multistate_gpu_sixteen_states_and_nonlinearity(gpu_backend):
    p,w,x,y,initial=fixture();names=['v',*[f'a{i}' for i in range(15)]]
    def compile(expr):return compile_training_equation(expr,states=names)
    p['state_equations'][0]=[compile('v*.85 + .02*tanh(a14) + .001*exp(a0) + .001*log(a1+1) + .001*sqrt(a2+1) + .001*sin(a3) + .001*cos(a4) + .001*a5**2')]
    p['state_equations'][0]+=[compile(f'{name}*.9 + .01*v') for name in names[1:]]
    p['state_resets'][0]=[compile('v-1.0625')]+[compile(name) for name in names[1:]]
    initial=np.concatenate([initial[:,:2],np.full((2,30),.1),initial[:,4:]],axis=1)
    cpu=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    a=cpu.gradients(x,y,initial=initial)
    p['backend']=gpu_backend
    state_compare(a,NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x,y,initial=initial))
    assert np.max(abs(np.array(a['initial_state_gradients'])[:,[30,31]]))>1e-8
    p['max_tape_bytes']=a['tape_bytes']
    with pytest.raises(ValueError,match='tape budget'):
        NativeLIFTrainer(p,runner=RUNNER,weights=w).step(x,y,initial=initial)


@pytest.mark.parametrize('ranks',[None,2])
def test_brian_multistate_gpu_end_to_end(gpu_backend,ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    from test_training_brian_multistate import model
    from brian2_rust import lower_brian_training
    net,source,groups,x,dt=model()
    bundle=lower_brian_training(net,input_group=source,layers=groups,backend=gpu_backend,mpi_ranks=ranks,
        trainable_neuron_parameters={g.name:['tau','theta','kick'] for g in groups},learning_rate=1e-7)
    cpu_plan=copy.deepcopy(bundle.plan);cpu_plan['backend']='cpu';cpu_plan.pop('mpi_ranks',None)
    a=NativeLIFTrainer(cpu_plan,runner=RUNNER,weights=bundle.weights).step(x[None],[0],initial=[bundle.initial_state])
    b=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).step(x[None],[0],initial=[bundle.initial_state])
    state_compare(a,b)
