"""Actual CUDA forward/backward, optimizer and checkpoint acceptance."""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lif_training_plan
from test_native_training import RUNNER
from test_native_training_graph import fixture_graph

@pytest.fixture(params=['cuda','metal'])
def gpu_backend(request):
    flag='B2_TEST_CUDA_TRAIN' if request.param=='cuda' else 'B2_TEST_GPU'
    if os.environ.get(flag)!='1':pytest.skip('actual GPU required')
    return request.param


@pytest.mark.parametrize('graph',[False,True])
@pytest.mark.parametrize('reset',['subtract','zero'])
@pytest.mark.parametrize('detach',[False,True])
@pytest.mark.parametrize('window',[None,3])
def test_cuda_native_vjp(graph,reset,detach,window,gpu_backend):
    plan,weights,x,y,initial=fixture_graph(reset,detach,window)
    if not graph:
        plan=lif_training_plan([2,2,2],beta=[.5,.75],reset=reset,
                               detach_reset=detach,tbptt_window=window,surrogate_slope=2)
        weights=[weights[0],weights[2]]
    # Keep the trajectory away from hard-threshold ties when comparing f64/f32.
    plan['threshold']=[1.0625,1.03125]
    cpu=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    plan['backend']=gpu_backend
    cuda=NativeLIFTrainer(plan,runner=RUNNER,weights=weights)
    for _ in range(3):
        a=cpu.step(x,y,initial=initial);b=cuda.step(x,y,initial=initial)
        assert b['backend']==gpu_backend and b['gpu_dispatches']==1
        assert gpu_backend+'-forward-backward' in b['numeric_profile']
        assert b['loss']==pytest.approx(a['loss'],rel=3e-6,abs=3e-6)
        for key in ['spikes','initial_gradients','final_membrane','logits']:
            np.testing.assert_allclose(a[key],b[key],rtol=5e-5,atol=3e-6)
        for key in ['gradients']:
            for av,bv in zip(a[key],b[key]):np.testing.assert_allclose(av,bv,rtol=5e-5,atol=3e-6)
        for key in ['weights','first_moment','second_moment']:
            for av,bv in zip(a['state'][key],b['state'][key]):
                np.testing.assert_allclose(av,bv,rtol=5e-5,atol=3e-6)


def test_cuda_mask_freeze_fresh_process(tmp_path,gpu_backend):
    plan,w,x,y,initial=fixture_graph();plan['backend']=gpu_backend;plan['trainable'][3]=False
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=w)
    trainer.step(x,y,initial=initial)
    assert trainer.state['weights'][3]==w[3]
    masks=copy.deepcopy(plan['masks']);masks[1][0]=0
    trainer.update_mask(masks);trainer.step(x,y,initial='carry')
    assert trainer.state['weights'][1][0]==trainer.state['first_moment'][1][0]==0
    masks[1][0]=1;trainer.update_mask(masks,growth_weight=.12)
    checkpoint=tmp_path/'state.json';trainer.store(checkpoint)
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(plan=trainer.plan,inputs=x.tolist(),labels=y)))
    expected=trainer.step(x,y,initial='carry')
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
c=json.load(open(sys.argv[1]));t=NativeLIFTrainer(c['plan'])
t.restore(sys.argv[2]);print(json.dumps(t.step(c['inputs'],c['labels'],initial='carry')))
'''
    process=subprocess.run([sys.executable,'-c',code,str(config),str(checkpoint)],
                            capture_output=True,text=True,check=True)
    assert json.loads(process.stdout)==expected


def test_cuda_extra_budget(tmp_path,gpu_backend):
    plan,w,x,y,initial=fixture_graph()
    cpu=NativeLIFTrainer(plan,runner=RUNNER,weights=w).gradients(x,y,initial=initial)
    plan.update(backend=gpu_backend,max_tape_bytes=cpu['tape_bytes'])
    with pytest.raises(ValueError,match='tape budget exceeded'):
        NativeLIFTrainer(plan,runner=RUNNER,weights=w).step(x,y)
