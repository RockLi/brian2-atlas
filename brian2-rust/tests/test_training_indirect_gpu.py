"""Actual GPU runtime addressing, independent VJP and owner-compute MPI."""
import copy
import os
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
from brian2_rust.training_equations import neuron_parameter_bank
from test_native_training import RUNNER
from test_training_indirect import model,oracle
from test_training_indirect_migration import owned
from test_training_dynamic_gpu import backend
from test_training_delay_update import snapshot


def require_mpi(ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')


def compare(actual,reference,backend):
    assert actual['backend']==backend and actual['gpu_dispatches']>0
    np.testing.assert_array_equal(actual['spikes'],reference['spikes'])
    assert actual['loss']==pytest.approx(reference['loss'],abs=4e-6)
    for key in ('final_state','initial_state_gradients','logits'):
        np.testing.assert_allclose(actual[key],reference[key],rtol=8e-4,atol=5e-6,err_msg=key)
    for a,b in zip(actual['gradients'],reference['gradients']):np.testing.assert_allclose(a,b,rtol=8e-4,atol=5e-6)
    assert actual['gradient_scope']==reference['gradient_scope']


@pytest.mark.parametrize('detach',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('nested',[False,True])
@pytest.mark.parametrize('noisy',[False,True])
def test_gpu_index_vjp_independent_difference(backend,detach,window,nested,noisy):
    p,w,x=model(detach,window,nested=nested,noisy=noisy);p['backend']=backend
    actual=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None],[0])
    loss,z,spikes,anchors=oracle(p,w,x)
    assert actual['backend']==backend and actual['gpu_dispatches']>0
    assert actual['loss']==pytest.approx(loss,abs=4e-6)
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    np.testing.assert_allclose(actual['final_state'][0],z,atol=2e-6,rtol=2e-5)
    eps=1e-6
    for bank,row in enumerate(w):
        for k in range(len(row)):
            upper=copy.deepcopy(w);lower=copy.deepcopy(w);upper[bank][k]+=eps;lower[bank][k]-=eps
            fd=(oracle(p,upper,x,anchors=anchors)[0]-oracle(p,lower,x,anchors=anchors)[0])/(2*eps)
            assert actual['gradients'][bank][k]==pytest.approx(fd,rel=8e-4,abs=5e-6)
    for k in range(len(z)):
        if k in p['dynamic']['integer_states']:
            assert actual['initial_state_gradients'][0][k]==0;continue
        upper=np.array(p['dynamic']['initial']);lower=upper.copy();upper[k]+=eps;lower[k]-=eps
        fd=(oracle(p,w,x,upper,anchors)[0]-oracle(p,w,x,lower,anchors)[0])/(2*eps)
        assert actual['initial_state_gradients'][0][k]==pytest.approx(fd,rel=8e-4,abs=5e-6)


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('detach,noisy',[(False,False),(False,True),(True,True)])
def test_gpu_index_batch_carry_rng_checkpoint(backend,ranks,detach,noisy,tmp_path):
    require_mpi(ranks)
    p,w,x=model(detach,2,ranks=ranks,nested=True,noisy=noisy);p['backend']=backend;p['trainable']=[False]*len(w)
    q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);reference=NativeLIFTrainer(q,runner=RUNNER,weights=w)
    xx=np.stack([x,x[:,::-1]]);initial=np.tile(p['dynamic']['initial'],(2,1));initial[1,4:6]=[1,0];initial[1,2:4]=[.8,1.1]
    for start in (0,2,4):
        kw=dict(initial='carry') if start else dict(initial=initial,start_tick=3,**({'noise_sequence':9} if noisy else {}))
        before=snapshot(trainer)
        actual=trainer.gradients(xx[:,start:start+2],[0,1],**kw)
        expected=reference.gradients(xx[:,start:start+2],[0,1],**kw);compare(actual,expected,backend)
        assert snapshot(trainer)[:-1]==before[:-1]
        if ranks:assert 'mpi-dynamic' in actual['numeric_profile'] and 'owner' in actual['numeric_profile']
        trainer.step(xx[:,start:start+2],[0,1],**kw);reference.step(xx[:,start:start+2],[0,1],**kw)
        path=tmp_path/'indices.json';trainer.store(path);restored=NativeLIFTrainer(p,runner=RUNNER);restored.restore(path);trainer=restored
        assert trainer.clock_tick==reference.clock_tick and trainer.noise_sequence==reference.noise_sequence


@pytest.mark.parametrize('ranks',[None,2,8])
def test_gpu_index_collisions_ignore_overwritten_parameter(backend,ranks):
    require_mpi(ranks)
    p,w,x=model(ranks=ranks);p['backend']=backend
    p['projections'].append(neuron_parameter_bank(1));p['masks'].append([1.]);p['trainable'].append(True);w.append([.37])
    transform=compile_dynamic_transform('a=peer+unused\nb=2*peer',states=dict(a=0,b=1,peer=2,pick=3),
        parameters=dict(unused=(2,0)),state_types={0:'float',1:'float',2:'float',3:'integer'})
    ps=len(p['dynamic']['program_sets']);p['dynamic']['program_sets'].append(transform['programs'])
    a=dynamic_action(transform,[6,7,0,4],owner=3,program_set=ps)
    a['indirect']=dict(writes={str(s):dict(index=dict(kind='read',slot=3),tables=[[0,1]]) for s in (0,1)})
    p['dynamic']['actions'].insert(0,a)
    q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    q['dynamic']['actions'][0]=dict(a,indirect=None,writes=[0]);q['dynamic']['program_sets'][-1]=[transform['programs'][1]]
    actual=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None,:1],[0])
    expected=NativeLIFTrainer(q,runner=RUNNER,weights=w).gradients(x[None,:1],[0])
    compare(actual,expected,backend);assert actual['gradients'][2]==[0.]


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('kind',['read','write'])
def test_gpu_nonroot_invalid_address_is_atomic(backend,ranks,kind):
    require_mpi(ranks)
    p,w,x=model(ranks=ranks);p['backend']=backend
    if kind=='read':p['dynamic']['initial'][4]=-1
    else:p['dynamic']['program_sets'][p['dynamic']['actions'][-1]['program_set']][2]=[dict(op='integer_constant',value=7)]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=snapshot(trainer)
    with pytest.raises(ValueError,match='nonfinite|invalid'):trainer.step(x[None],[0])
    assert snapshot(trainer)==before


def inactive(ranks,detach):
    p,w,x=model(detach=detach,ranks=ranks)
    p['dynamic']['actions']=p['dynamic']['actions'][:2]+p['dynamic']['actions'][4:]
    ids=sorted({a['program_set'] for a in p['dynamic']['actions'] if a['program_set'] is not None})
    p['dynamic']['program_sets']=[p['dynamic']['program_sets'][i] for i in ids]
    for a in p['dynamic']['actions']:
        if a['program_set'] is not None:a['program_set']=ids.index(a['program_set'])
    p['dynamic']['initial'][2:6]=[.1,.2,-1,9]
    return p,w,np.zeros((1,3,2))


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('detach',[False,True])
def test_gpu_inactive_invalid_counterfactual_contract(backend,ranks,detach):
    require_mpi(ranks)
    p,w,x=inactive(ranks,detach);q=copy.deepcopy(p);q['mpi_ranks']=None;p['backend']=backend
    actual=NativeLIFTrainer(p,runner=RUNNER,weights=w);ref=NativeLIFTrainer(q,runner=RUNNER,weights=w)
    # Evaluation ignores invalid addresses of unexecuted events.
    compare(actual.evaluate(x,[0]),ref.evaluate(x,[0]),backend)
    if detach:compare(actual.gradients(x,[0]),ref.gradients(x,[0]),backend)
    else:
        for trainer in (actual,ref):
            before=snapshot(trainer)
            with pytest.raises(ValueError,match='counterfactual|nonfinite|invalid'):trainer.gradients(x,[0])
            assert snapshot(trainer)==before


@pytest.mark.parametrize('ranks',[None,2,8])
def test_gpu_indexed_int32_boolean_transport_is_exact(backend,ranks):
    require_mpi(ranks)
    p,w,x=model(ranks=ranks);p['backend']=backend;spec=p['dynamic']
    spec['initial'] += [16777217.,-2147483648.,0.,1.,0.,0.]
    spec['initial_parameters'] += [None]*6;spec['detached'] += [True]*6
    spec['integer_states'] += [8,9,12];spec['binary_states']=[10,11,13]
    transform=compile_dynamic_transform('a+=1\nb=not b',states=dict(a=0,b=1,pick=2),state_types={0:'integer',1:'boolean',2:'integer'})
    ps=len(spec['program_sets']);spec['program_sets'].append(transform['programs'])
    a=dynamic_action(transform,[12,13,4],owner=3,program_set=ps)
    a['indirect']=dict(reads={'0':dict(index=4,tables=[[8,9]]),'1':dict(index=4,tables=[[10,11]])},
        writes={'0':dict(index=dict(kind='read',slot=2),tables=[[8,9]]),'1':dict(index=dict(kind='read',slot=2),tables=[[10,11]])})
    spec['actions'].append(a)
    q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None],[0])
    expected=NativeLIFTrainer(q,runner=RUNNER,weights=w).gradients(x[None],[0]);compare(result,expected,backend)
    np.testing.assert_array_equal(np.array(result['final_state'])[:,8:],np.array(expected['final_state'])[:,8:])
    assert np.all(np.array(result['initial_state_gradients'])[:,8:]==0)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_gpu_index_candidate_migration_and_resume(backend,ranks):
    require_mpi(ranks)
    p,w,x=owned(ranks);p['backend']=backend;q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);ref=NativeLIFTrainer(q,runner=RUNNER,weights=w)
    for t in (trainer,ref):
        t.step(x[None,:1],[0]);cursor=(t.clock_tick,t.state['step'])
        masks=copy.deepcopy(p['masks']);masks[0][0]=0;t.update_mask(masks)
        assert t.neuron_state[0][6:8]==[0.,0.]
        masks[0][0]=1;t.update_mask(masks,growth_weight=.12)
        np.testing.assert_allclose(t.neuron_state[0][6:8],[.3,.6],atol=1e-7)
        assert cursor==(t.clock_tick,t.state['step'])
    compare(trainer.gradients(x[None,1:],[0],initial='carry'),ref.gradients(x[None,1:],[0],initial='carry'),backend)


def test_gpu_index_metadata_and_address_tape_budget(backend):
    p,w,x=model();p['backend']=backend
    p['dynamic']['actions'][2]['indirect']['reads']['1']['tables']=[[0,1]*2048]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);result=trainer.evaluate(x[None],[0])
    p['max_tape_bytes']=result['tape_bytes']-1;limited=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=snapshot(limited)
    with pytest.raises(ValueError,match='GPU tape budget'):limited.evaluate(x[None],[0])
    assert snapshot(limited)==before


def test_metal_index_loader_rejects_previous_abi(tmp_path,monkeypatch):
    import subprocess
    from brian2_rust import training_metal
    if os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal setup required')
    # This library intentionally only exports the previous ABI. It must never
    # be called for an indexed plan, even though it would return success.
    source=tmp_path/'old.c';source.write_text('int b2_train_metal_v5r4(void){return 0;}\n')
    library=tmp_path/'old.dylib'
    subprocess.run(['clang','-dynamiclib',str(source),'-o',str(library)],check=True,capture_output=True,text=True)
    monkeypatch.setattr(training_metal,'build',lambda directory:library)
    p,w,x=model();p['backend']='metal';trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=snapshot(trainer)
    with pytest.raises(ValueError,match='ABI symbol missing'):trainer.gradients(x[None],[0])
    assert snapshot(trainer)==before


@pytest.mark.parametrize('ranks',[None,2,8])
def test_gpu_full_width_address_collective(backend,ranks):
    require_mpi(ranks)
    p,w,x=model(ranks=ranks);p['backend']=backend;spec=p['dynamic']
    spec['initial'] += [.01*(j+1) for j in range(63)]
    spec['initial_parameters'] += [None]*63;spec['detached'] += [False]*63
    p['projections'].append(neuron_parameter_bank(1));p['masks'].append([1.]);p['trainable'].append(True);w.append([.002])
    names={f's{j}':j for j in range(63)};names['pick']=63
    code='\n'.join(f's{j}=1.01*s{j}+gain' for j in range(63))+'\npick=pick'
    transform=compile_dynamic_transform(code,states=names,parameters=dict(gain=(2,0)),state_types={j:'integer' if j==63 else 'float' for j in range(64)})
    ps=len(spec['program_sets']);spec['program_sets'].append(transform['programs'])
    action=dynamic_action(transform,[*range(8,71),4],owner=3,program_set=ps)
    action['indirect']=dict(reads={str(j):dict(index=4,tables=[[8+j,8+(j+1)%63]]) for j in range(63)},
        writes={str(j):dict(index=dict(kind='read',slot=63),tables=[[8+j,8+(j+1)%63]]) for j in range(63)})
    coupling=compile_dynamic_transform('v+=.2*first',states=dict(v=0,first=1));cps=len(spec['program_sets']);spec['program_sets'].append(coupling['programs'])
    spec['actions'][:0]=[action,dynamic_action(coupling,[2,8],owner=3,program_set=cps)]
    q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    initial=np.tile(spec['initial'],(2,1));initial[1,4]=1;xx=np.stack([x,x[::-1]])
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(xx,[0,0],initial=initial)
    expected=NativeLIFTrainer(q,runner=RUNNER,weights=w).gradients(xx,[0,0],initial=initial)
    compare(result,expected,backend)
    assert abs(result['gradients'][2][0])>1e-7
    np.testing.assert_array_equal(np.array(result['final_state'])[:,4:6],np.array(expected['final_state'])[:,4:6])
