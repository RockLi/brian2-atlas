"""Real target-owned MPI BPTT, shared parameter ownership and restart."""
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

pytestmark=pytest.mark.skipif(os.environ.get('B2_TEST_MPI')!='1',reason='real MPI required')


@pytest.mark.parametrize('ranks',[2,4])
@pytest.mark.parametrize('graph',[False,True])
@pytest.mark.parametrize('reset,detach,window',[
    ('subtract',False,None),('subtract',True,3),('zero',False,3),('zero',True,None)])
def test_target_owned_mpi_against_serial(ranks,graph,reset,detach,window):
    plan,w,x,y,initial=fixture_graph(reset,detach,window)
    if not graph:
        plan=lif_training_plan([2,2,2],reset=reset,detach_reset=detach,
            beta=[.5,.75],surrogate_slope=2,tbptt_window=window)
        w=[w[0],w[2]]
    cpu=NativeLIFTrainer(plan,runner=RUNNER,weights=w)
    plan['mpi_ranks']=ranks;mpi=NativeLIFTrainer(plan,runner=RUNNER,weights=w)
    for _ in range(2):
        a=cpu.step(x,y,initial=initial);b=mpi.step(x,y,initial=initial)
        assert 'mpi-target-owned' in b['numeric_profile']
        for key in ['spikes','final_membrane','logits','initial_gradients']:
            np.testing.assert_allclose(b[key],a[key],rtol=2e-13,atol=2e-14)
        assert b['loss']==pytest.approx(a['loss'],abs=2e-15)
        for av,bv in zip(a['gradients'],b['gradients']):
            np.testing.assert_allclose(av,bv,rtol=2e-13,atol=2e-14)
        for key in ['weights','first_moment','second_moment']:
            for av,bv in zip(a['state'][key],b['state'][key]):
                np.testing.assert_allclose(av,bv,rtol=2e-13,atol=2e-14)


def test_mpi_exact_restart_mask_freeze_and_rank_contract(tmp_path):
    plan,w,x,y,initial=fixture_graph();plan['mpi_ranks']=2;plan['trainable'][3]=False
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=w)
    trainer.step(x,y,initial=initial)
    assert trainer.state['weights'][3]==w[3]
    assert any(v!=0 for v in trainer.state['first_moment'][1])
    masks=copy.deepcopy(plan['masks']);masks[1][0]=0
    trainer.update_mask(masks);trainer.step(x,y,initial='carry')
    assert trainer.state['weights'][1][0]==trainer.state['second_moment'][1][0]==0
    masks[1][0]=1;trainer.update_mask(masks,growth_weight=.15)
    checkpoint=tmp_path/'checkpoint';trainer.store(checkpoint)
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(plan=trainer.plan,x=x.tolist(),y=y)))
    expected=trainer.step(x,y,initial='carry')
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
c=json.load(open(sys.argv[1]));t=NativeLIFTrainer(c['plan'])
t.restore(sys.argv[2]);print(json.dumps(t.step(c['x'],c['y'],initial='carry')))
'''
    process=subprocess.run([sys.executable,'-c',code,str(config),str(checkpoint)],
                            check=True,text=True,capture_output=True,timeout=60)
    assert json.loads(process.stdout)==expected
    other=copy.deepcopy(trainer.plan);other['mpi_ranks']=4
    with pytest.raises(ValueError,match='plan/runtime mismatch'):
        NativeLIFTrainer(other,runner=RUNNER,weights=w).restore(checkpoint)
    trainer.plan['mpi_ranks']=4
    with pytest.raises(ValueError,match='topology/backend'):trainer.step(x,y)


def test_mpi_failure_no_commit_and_empty_owners():
    plan,w,x,y,initial=fixture_graph();plan['mpi_ranks']=4
    # 4 neuron owners but only 2 parameters in the tied bank.
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=w)
    trainer.step(x,y,initial=initial);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='invalid input'):
        trainer.step(x,[99,1])
    assert trainer.state==before
    trainer.plan['max_tape_bytes']=100
    with pytest.raises(ValueError,match='budget'):
        trainer.step(x,y)
    assert trainer.state==before


def test_mpi_empty_neuron_owners():
    plan=lif_training_plan([1,1,1],mpi_ranks=4)
    mpi=NativeLIFTrainer(plan,runner=RUNNER,weights=[[1.2],[.9]])
    del plan['mpi_ranks'];cpu=NativeLIFTrainer(plan,runner=RUNNER,weights=[[1.2],[.9]])
    x=[[[1.]]*6];initial=[[1.3,1.4]]
    a=mpi.step(x,[0],initial=initial);b=cpu.step(x,[0],initial=initial)
    for key in ['state','spikes','initial_gradients','final_membrane','gradients','loss']:
        assert a[key]==b[key]


def test_mpi_rank_request_mismatch_aborts(tmp_path):
    import signal
    plan,w,x,y,initial=fixture_graph();plan['mpi_ranks']=2
    trainer=NativeLIFTrainer(plan,runner=RUNNER,weights=w)
    request=dict(plan=plan,state=trainer.state,operation='gradients',inputs=x.tolist(),
                 labels=y,initial=initial.tolist())
    for rank in [0,1]:
        request['labels']=[rank,1]
        (tmp_path/f'request{rank}.json').write_text(json.dumps(request))
    wrapper=tmp_path/'different.py'
    wrapper.write_text('''import os,sys
rank=int(os.environ.get('PMI_RANK',os.environ.get('OMPI_COMM_WORLD_RANK','0')))
os.execv(sys.argv[1],[sys.argv[1],sys.argv[2]+'/request'+str(rank)+'.json',sys.argv[2]+'/result.json'])
''')
    env=dict(os.environ,B2_TRAIN_MPI_LIB=str(trainer._mpi_library),**trainer._mpi_environment)
    with subprocess.Popen(['mpiexec','-n','2',sys.executable,str(wrapper),str(RUNNER),str(tmp_path)],
                          env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,start_new_session=True) as process:
        try:stdout,stderr=process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.communicate();raise
    assert process.returncode!=0 and 'identity mismatch' in stderr
    assert not (tmp_path/'result.json').exists()
