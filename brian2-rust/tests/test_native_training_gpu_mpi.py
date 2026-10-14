"""Real Metal target-owned MPI BPTT; all ranks share this host's GPU."""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,compile_training_equation
from test_native_training import RUNNER
from test_native_training_equations import fixture_equation
from test_native_training_graph import fixture_graph

@pytest.fixture(params=['metal','cuda'])
def gpu_backend(request):
    flag='B2_TEST_GPU' if request.param=='metal' else 'B2_TEST_CUDA_TRAIN'
    if os.environ.get(flag)!='1' or os.environ.get('B2_TEST_MPI')!='1':pytest.skip('real GPU and MPI required')
    return request.param


def compare(a,b):
    assert a['spikes']==b['spikes']
    assert a['loss']==pytest.approx(b['loss'],abs=4e-6)
    for key in ['final_membrane','initial_gradients','logits']:
        np.testing.assert_allclose(a[key],b[key],atol=5e-6,rtol=3e-4)
    for key in ['weights','first_moment','second_moment']:
        for x,y in zip(a['state'][key],b['state'][key]):np.testing.assert_allclose(x,y,atol=5e-6,rtol=3e-4)
    for x,y in zip(a['gradients'],b['gradients']):np.testing.assert_allclose(x,y,atol=5e-6,rtol=3e-4)


@pytest.mark.parametrize('ranks',[2,4])
@pytest.mark.parametrize('kind',['graph','equation'])
@pytest.mark.parametrize('reset',['zero','subtract'])
@pytest.mark.parametrize('detach,window',[(False,None),(True,3)])
def test_gpu_target_owned_gradients_and_optimizer(ranks,kind,reset,detach,window,gpu_backend):
    p,w,x,y,initial=(fixture_equation if kind=='equation' else fixture_graph)(reset,detach,window)
    p['threshold']=[1.0625,1.03125]
    cpu=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    p['backend']=gpu_backend;serial=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    p['mpi_ranks']=ranks;distributed=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    for step in range(2):
        a=cpu.step(x,y,initial=initial);b=serial.step(x,y,initial=initial);c=distributed.step(x,y,initial=initial)
        compare(a,c);compare(b,c)
        assert 'target-owned' in c['numeric_profile']
        assert c['gpu_dispatches']==3+4*x.shape[1]


def test_gpu_mpi_empty_owners_and_evaluate(gpu_backend):
    p,w,x,y,initial=fixture_equation();p.update(backend=gpu_backend)
    a=NativeLIFTrainer(p,runner=RUNNER,weights=w).evaluate(x,y,initial=initial)
    p['mpi_ranks']=8
    b=NativeLIFTrainer(p,runner=RUNNER,weights=w).evaluate(x,y,initial=initial)
    compare(a,b)
    assert b['state']['step']==0 and b['gpu_dispatches']==3+2*x.shape[1]
    assert not np.any(b['initial_gradients'])
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    c=trainer.step(x,y,initial=initial)
    p.pop('mpi_ranks');compare(NativeLIFTrainer(p,runner=RUNNER,weights=w).step(x,y,initial=initial),c)


def test_gpu_mpi_fresh_process_mask_freeze_checkpoint(tmp_path,gpu_backend):
    p,w,x,y,initial=fixture_equation();p.update(backend=gpu_backend,mpi_ranks=2)
    p['trainable'][1]=False
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);trainer.step(x,y,initial=initial)
    masks=copy.deepcopy(p['masks']);masks[0][0]=0;trainer.update_mask(masks)
    assert trainer.state['weights'][0][0]==trainer.state['first_moment'][0][0]==0
    checkpoint=tmp_path/'checkpoint';trainer.store(checkpoint)
    request=tmp_path/'request';request.write_text(json.dumps(dict(plan=trainer.plan,x=x.tolist(),y=y)))
    output=tmp_path/'result'
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
p=json.load(open(sys.argv[1]));t=NativeLIFTrainer(p['plan'],runner=sys.argv[4]);t.restore(sys.argv[2])
json.dump(t.step(p['x'],p['y'],initial='carry'),open(sys.argv[3],'w'))
'''
    subprocess.run([sys.executable,'-c',code,str(request),str(checkpoint),str(output),str(RUNNER)],check=True,timeout=90)
    result=trainer.step(x,y,initial='carry')
    assert json.loads(output.read_text())==result
    assert result['state']['weights'][1]==w[1]
    assert result['state']['weights'][0][0]==result['state']['first_moment'][0][0]==0


def test_gpu_mpi_domain_failure_is_atomic(gpu_backend):
    p,w,x,y,initial=fixture_equation();p.update(backend=gpu_backend,mpi_ranks=2)
    p['equations'][1]=compile_training_equation('log(v-100)')
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='nonfinite'):trainer.step(x,y,initial=initial)
    assert trainer.state==before and trainer.neuron_state is None and trainer.elapsed_ticks==0
