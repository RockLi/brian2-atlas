"""Actual device-owned dynamic MPI actions, collective VJP and failure abort."""
import copy
import os
import time
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_dynamic import model,noisy_model
from test_training_dynamic_gpu import backend,cython_cache,compare,frontend


@pytest.fixture(autouse=True)
def require_mpi():
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local GPU MPI required')


def pair(plan,weights,ranks,backend):
    serial=copy.deepcopy(plan);serial['backend']=backend
    distributed=copy.deepcopy(serial);distributed['mpi_ranks']=ranks
    return NativeLIFTrainer(serial,runner=RUNNER,weights=weights),NativeLIFTrainer(distributed,runner=RUNNER,weights=weights)


def check_mpi(actual,expected,plan,time,backend,backward=True):
    compare(actual,expected,backend)
    assert 'mpi-dynamic' in actual['numeric_profile'] and 'owner' in actual['numeric_profile']
    actions=plan['dynamic']['actions']
    # Scheduled visits skip masked actions; TBPTT boundaries have their own
    # adjoint-clear dispatch. Count exactly for the synchronous fixtures only.
    if all(a.get('clock',0) in (None,0) for a in actions):
        active=sum(a.get('mask') is None or plan['masks'][a['mask'][0]][a['mask'][1]]!=0 for a in actions)
        cuts=(time-1)//plan['tbptt_window'] if plan.get('tbptt_window') else 0
        assert actual['gpu_dispatches']==3+time*2*active+(time*(1+2*active)+cuts if backward else 0)
    else:
        # A primary frame can contain multiple asynchronous action visits.
        # Check their actual emissions as well as the aggregated frame counts.
        assert actual['gpu_dispatches']>1
        assert actual['event_visits']==expected['event_visits']


@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('detach,window',[(True,None),(False,3)])
@pytest.mark.parametrize('noisy',[False,True])
def test_owner_dynamic_mpi_matches_serial_and_cpu(ranks,detach,window,noisy,backend):
    p,w,x=noisy_model(detach,window) if noisy else model(detach,window,True)
    # Exercise non-default ownership and ranks that own no neuron/action.
    for action in p['dynamic']['actions']:action['owner']=(action['owner']+1)%4
    data=np.stack([x[:6],x[:6,::-1]]);kw=dict(noise_sequence=7,start_tick=2) if noisy else {}
    serial,distributed=pair(p,w,ranks,backend)
    ref=serial.gradients(data,[0,1],**kw);result=distributed.gradients(data,[0,1],**kw)
    check_mpi(result,ref,p,6,backend)
    cpu=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(data,[0,1],**kw)
    compare(result,cpu,backend)
    # Explicit state must not add optimizer initial-value bindings on any rank.
    initial=np.tile(p['dynamic']['initial'],(2,1));initial[1,0]+=.02
    a=distributed.gradients(data[:,:3],[0,1],initial=initial,**kw)
    c=serial.gradients(data[:,:3],[0,1],initial=initial,**kw)
    check_mpi(a,c,p,3,backend)


@pytest.mark.parametrize('kind',['pending','clock','summed_noise','refractory'])
@pytest.mark.parametrize('ranks',[2,8])
def test_frontend_mpi_state_carry_migration_checkpoint(kind,ranks,backend,tmp_path):
    bundle,x=frontend(kind,backend);p=bundle.plan;p['tbptt_window']=2
    serial,parallel=pair(p,bundle.weights,ranks,backend)
    for start,stop in [(0,2),(2,4),(4,6)]:
        kw=dict(initial='carry') if start else {}
        a=parallel.execute(x[None,start:stop],[0],**kw);c=serial.execute(x[None,start:stop],[0],**kw)
        check_mpi(a,c,parallel.plan,stop-start,backend)
        for key in ['weights','first_moment','second_moment']:
            for row,other in zip(parallel.state[key],serial.state[key]):np.testing.assert_allclose(row,other,rtol=2e-3,atol=4e-5)
        assert parallel.clock_tick==serial.clock_tick and parallel.noise_sequence==serial.noise_sequence
        if stop==6:break
        bank,index=p['dynamic']['migration']['controlled_masks'][-1]
        for trainer in (parallel,serial):
            masks=copy.deepcopy(trainer.plan['masks']);masks[bank][index]=0 if stop==2 else 1
            before=(trainer.state['step'],trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence)
            trainer.update_mask(masks,growth_weight=.4)
            assert before==(trainer.state['step'],trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence)
        filename=tmp_path/'mpi-gpu.json';parallel.store(filename)
        new=NativeLIFTrainer(parallel.plan,runner=RUNNER);new.restore(filename);parallel=new


@pytest.mark.parametrize('ranks',[2,8])
def test_evaluate_has_no_vjp_or_optimizer_commit(ranks,backend):
    p,w,x=model();serial,parallel=pair(p,w,ranks,backend)
    before=copy.deepcopy(parallel.state);a=parallel.evaluate(x[None,:3],[0]);c=serial.evaluate(x[None,:3],[0])
    check_mpi(a,c,p,3,backend,False)
    assert parallel.state==before and parallel.neuron_state is None and parallel.clock_tick==0
    assert all(v==0 for row in a['gradients'] for v in row)


@pytest.mark.parametrize('ranks',[2,8])
def test_nonroot_owner_domain_error_aborts_transactionally(ranks,backend):
    from brian2_rust.training_dynamic import compile_dynamic_transform
    p,w,x=model();p['backend']=backend;p['mpi_ranks']=ranks
    p['dynamic']['initial'][0]=-1
    p['dynamic']['program_sets'][0]=compile_dynamic_transform('v=log(v)',states={'v':0})['programs']
    p['dynamic']['actions'][0]['owner']=3
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick))
    started=time.monotonic()
    with pytest.raises(ValueError,match='nonfinite'):trainer.execute(x[None,:2],[0])
    assert time.monotonic()-started<60
    assert before==(trainer.state,trainer.neuron_state,trainer.clock_tick)


def test_gpu_mpi_frontend_admission(backend):
    from test_training_brian_dynamic import network
    net,inp,layers,*_=network()
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,backend=backend,mpi_ranks=2)
    assert bundle.plan['backend']==backend and bundle.plan['mpi_ranks']==2


@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('dual',[False,True])
@pytest.mark.parametrize('refractory',[False,True])
def test_mpi_autapse_aliases_preserve_snapshot_and_writeback(ranks,dual,refractory,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(2,[],[]*b.second,dt=dt)
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1'+(' (unless refractory)' if refractory else ''),
        threshold='v>1',reset='v-=.3',method='euler',dt=dt,refractory=3*dt if refractory else False) for _ in range(2)]
    groups[1].v=[1.4,1.8]
    code='v_post+=w\nv_pre+=.2*w\nw=.3*v_post+.05' if dual else 'v_post+=w\nw=.3*v_pre+.05'
    syn=b.Synapses(groups[1],groups[1],'w:1',on_pre=code);syn.connect(i=[0,1],j=[0,1]);syn.w=[.7,.9]
    bundle=lower_brian_dynamic_training(b.Network(inp,*groups,syn),input_group=inp,layers=groups,
        detach_reset=not dual,tbptt_window=2)
    x=np.zeros((1,6,2));serial,parallel=pair(bundle.plan,bundle.weights,ranks,backend)
    for initial in (None,[bundle.initial_state]):
        result=parallel.gradients(x,[0],initial=initial)
        expected=serial.gradients(x,[0],initial=initial)
        check_mpi(result,expected,bundle.plan,6,backend)
        cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x,[0],initial=initial)
        compare(result,cpu,backend)
