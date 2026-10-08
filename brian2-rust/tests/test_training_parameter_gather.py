"""Runtime canonical-bank reads: independent VJPs, exact types and transactions."""
import copy
import os
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training import lif_training_plan
from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
from brian2_rust.training_equations import ParameterBank, NormalNoise, neuron_parameter_bank
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_delay_update import snapshot


def model(engine='cpu',ranks=None,detach=False,window=None,noisy=True):
    weights=[[.13,.21,.34,.55],[1.,0.,1.],[1.,1.,0.],[.03]]
    identity=[[[dict(op='state',index=0)]]]*2
    p=lif_training_plan([1,2,2],projections=[neuron_parameter_bank(len(w)) for w in weights],
        state_equations=identity,state_resets=identity,clock=dict(origin=0.,dt=.0002),
        threshold=.5,detach_reset=detach,tbptt_window=window,backend=engine,mpi_ranks=ranks,
        trainable=[True,False,False,True],noise_streams=[1,1] if noisy else None)
    sets=[];actions=[]
    def add(code,names,reads,owner,params=None,types=None,**kw):
        tr=compile_dynamic_transform(code,states=names,parameters=params,state_types=types)
        n=len(sets);sets.append(tr['programs']);a=dynamic_action(tr,reads,owner=owner,program_set=n,**kw);actions.append(a)
        return a
    for j in range(4):
        params=dict(gain=ParameterBank(0),route=ParameterBank(1,'integer'),enabled=ParameterBank(2,'boolean'))
        if noisy:params.update(sigma=(3,0),eta=NormalNoise(0))
        add('v=.8*v+(.3*gain(route(pick)) if enabled(pick) else 0.)'+('+sigma*eta' if noisy else '')+'\npick=(pick+1)%3',
            dict(v=0,pick=1),[j,4+j],j,params,{0:'float',1:'integer'},noise_domain=3,noise_entity=j,noise_streams=int(noisy))
    for j in range(4):actions.append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None))
    for j in range(4):add('v-=.4',dict(v=0),[j],j,trigger=dict(external=False,index=j),detach_trigger=detach)
    p.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(initial=[.8,.3,.7,.4,0.,1.,2.,0.],
        initial_parameters=[None]*8,detached=[False]*4+[True]*4,integer_states=[4,5,6,7],
        integer_parameters=[[1,j] for j in range(3)],voltage=[0,1,2,3],program_sets=sets,actions=actions))
    return p,weights,np.zeros((1,6,1))


def oracle(p,w,*,initial=None,anchors=None):
    from test_training_stochastic import normal
    z=np.array(p['dynamic']['initial'] if initial is None else initial,float)
    before=[];voltages=[];spikes=[];hard=[]
    for t in range(6):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors['before'][t].copy()
        before.append(z.copy());pick=z[4:].astype(int)
        z[:4]=.8*z[:4]+np.array([.3*w[0][int(w[1][k])] if w[2][k] else 0. for k in pick])
        if p.get('noise_streams'):z[:4]+=w[3][0]*np.array([normal(p['seed'],0,0,3,j,t,0) for j in range(4)])
        z[4:]=(pick+1)%3;v=z[:4].copy();h=(v>.5).astype(float);s=h.copy()
        if anchors is not None:
            h=anchors['hard'][t];s=h+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['v'][t]-.5))**2*(v-anchors['v'][t])
        voltages.append(v);hard.append(h);spikes.append(s);z[:4]-=.4*(h if p['detach_reset'] else s)
    logits=np.asarray(spikes)[:,2:].mean(axis=0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.asarray(spikes),dict(before=before,v=voltages,hard=hard)


@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('detach',[False,True])
@pytest.mark.parametrize('window',[None,2])
def test_gather_all_float_vjps(engine,noisy,detach,window):
    p,w,x=model(engine=engine,noisy=noisy,detach=detach,window=window)
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x,[0]);loss,z,spikes,anchors=oracle(p,w)
    assert result['gradient_scope']==('full-bptt-detached-index-routing' if window is None else 'tbptt-detach-boundaries-and-index-routing')
    tol=4e-6 if engine=='cpu' else 8e-4;absolute=4e-8 if engine=='cpu' else 5e-6
    assert result['loss']==pytest.approx(loss,abs=absolute)
    np.testing.assert_allclose(result['final_state'][0],z,rtol=tol,atol=absolute);np.testing.assert_array_equal(result['spikes'][0],spikes)
    if engine!='cpu':assert result['gpu_dispatches']>0
    eps=1e-6
    for bank in (0,3):
        for k in range(len(w[bank])):
            hi=copy.deepcopy(w);lo=copy.deepcopy(w);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(oracle(p,hi,anchors=anchors)[0]-oracle(p,lo,anchors=anchors)[0])/(2*eps)
            assert result['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute)
    assert result['gradients'][0][2:]==[0.,0.]
    assert result['gradients'][1]==[0.,0.,0.] and result['gradients'][2]==[0.,0.,0.]
    for k in range(4):
        hi=np.asarray(p['dynamic']['initial']).copy();lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(oracle(p,w,initial=hi,anchors=anchors)[0]-oracle(p,w,initial=lo,anchors=anchors)[0])/(2*eps)
        assert result['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute)
    assert result['initial_state_gradients'][0][4:]==[0.,0.,0.,0.]


@pytest.mark.parametrize('ranks',[None,2,8])
def test_gather_optimizer_carry_checkpoint_batch_and_rollback(engine,ranks,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(engine=engine,ranks=ranks);p['masks'][0][0]=0.;w[0][0]=0.
    q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);reference=NativeLIFTrainer(q,runner=RUNNER,weights=w)
    xx=np.repeat(x,2,axis=0);initial=np.tile(p['dynamic']['initial'],(2,1));initial[1,4:]=[2,0,1,2]
    for turn in range(3):
        kw=dict(initial=initial,noise_sequence=7,start_tick=4) if turn==0 else dict(initial='carry')
        actual=trainer.step(xx,[0,1],**kw);expected=reference.step(xx,[0,1],**kw)
        for key in ('final_state','initial_state_gradients','spikes'):np.testing.assert_allclose(actual[key],expected[key],rtol=8e-4,atol=5e-6)
        for a,b in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,b,rtol=8e-4,atol=5e-6)
        for a,b in zip(trainer.state['weights'],reference.state['weights']):np.testing.assert_allclose(a,b,rtol=8e-4,atol=5e-6)
        assert actual['gradients'][0][0]==0.
        if ranks:assert 'mpi-dynamic' in actual['numeric_profile']
        if engine!='cpu':assert actual['gpu_dispatches']>0
        saved=tmp_path/'gather.json';trainer.store(saved);restored=NativeLIFTrainer(p,runner=RUNNER);restored.restore(saved);trainer=restored
    assert trainer.state['weights'][0][1]!=w[0][1]
    # One rank owns this invalid selection, all ranks must roll back together.
    before=snapshot(trainer);bad=initial.copy();bad[1,5]=-1
    with pytest.raises(ValueError):trainer.step(xx,[0,1],initial=bad)
    assert snapshot(trainer)==before


@pytest.mark.parametrize('issue',['negative','large','float_selector','mixed_integer_bank','float_integer_bank','forward_ssa','wrong_bank'])
def test_gather_admission_and_bounds_are_atomic(engine,issue):
    p,w,x=model(engine=engine);spec=p['dynamic'];program=spec['program_sets'][0][0]
    gather=next(n for n in program if n['op']=='integer_parameter_gather')
    if issue=='negative':spec['initial'][4]=-1.
    elif issue=='large':spec['initial'][4]=3.
    elif issue=='float_selector':program[gather['index']]=dict(op='constant',value=0.)
    elif issue=='mixed_integer_bank':spec['integer_parameters'].pop()
    elif issue=='float_integer_bank':gather['op']='parameter_gather'
    elif issue=='forward_ssa':gather['index']=len(program)
    elif issue=='wrong_bank':gather['bank']=len(w)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=snapshot(trainer)
    with pytest.raises(ValueError):trainer.step(x,[0])
    assert snapshot(trainer)==before


@pytest.mark.parametrize('ranks',[None,2,8])
def test_gather_lazy_bounds_and_exact_int32_payload(engine,ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(engine=engine,ranks=ranks,noisy=False);spec=p['dynamic']
    # Preserve values that float32 cannot represent, including a NaN bit pattern.
    w[1]=[2147483647.,-2147483648.,16777217.]
    for j in range(4):
        tr=compile_dynamic_transform('v=v if pick>=0 else gain(pick)\npick=route(pick)',
            states=dict(v=0,pick=1),parameters=dict(gain=ParameterBank(0),route=ParameterBank(1,'integer')),
            state_types={0:'float',1:'integer'})
        spec['program_sets'][j]=tr['programs']
    out=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[:,:1],[0])
    assert out['final_state'][0][4:]==[2147483647.,-2147483648.,16777217.,2147483647.]
    # Out-of-range branch is genuinely lazy, and integer output reaches live state exactly.
    p,w,x=model(engine=engine,ranks=ranks,noisy=False)
    for j in range(4):
        tr=compile_dynamic_transform('v=v if pick>=0 else gain(-1)\npick=pick',states=dict(v=0,pick=1),
            parameters=dict(gain=ParameterBank(0)),state_types={0:'float',1:'integer'})
        p['dynamic']['program_sets'][j]=tr['programs']
    NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x,[0])


def test_parameter_bank_compiler_requires_integer_selector():
    for expr in ('gain(.5)','gain(pick,1)','gain(index=pick)'):
        with pytest.raises(ValueError):compile_dynamic_transform('v='+expr,states=dict(v=0,pick=1),
            parameters=dict(gain=ParameterBank(0)),state_types={0:'float',1:'integer'})


def test_metal_gather_rejects_previous_abi(tmp_path,monkeypatch):
    import subprocess
    from brian2_rust import training_metal
    if os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal setup required')
    source=tmp_path/'old.c';source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\nint b2_train_metal_v5r5(void){return 0;}\n')
    library=tmp_path/'old.dylib';subprocess.run(['clang','-dynamiclib',str(source),'-o',str(library)],check=True,capture_output=True,text=True)
    monkeypatch.setattr(training_metal,'build',lambda directory:library)
    p,w,x=model(engine='metal');trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=snapshot(trainer)
    with pytest.raises(ValueError,match='ABI symbol missing'):trainer.gradients(x,[0])
    assert snapshot(trainer)==before


def test_cuda_gather_translation_contains_typed_bounds_and_reverse_scatter():
    from brian2_rust.training_cuda import kernel_source
    text=kernel_source()
    assert 'case 48:case 49:{int index=atlas_dynamic_int(value[a]);if(index<0||uint64_t(index)>=b)return NAN;' in text
    assert 'gradient[gradient_base+slot+uint64_t(index)]+=g;' in text
    import re
    integer_clause=re.search(r'bool integer=([^;]+);',text).group(1)
    assert 'op==49' in integer_clause.split('||') and '__float_as_int(x)' in text
