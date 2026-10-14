"""Native typed int32 storage, exact device transport, and detached control."""
import copy
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training import lif_training_plan
from brian2_rust.training_equations import neuron_parameter_bank
from test_native_training import RUNNER
from test_training_stochastic import normal


@pytest.fixture(scope='module',params=['cpu','metal','cuda'])
def engine(request):
    name=request.param
    if name=='cpu':yield name;return
    flag='B2_TEST_GPU' if name=='metal' else 'B2_TEST_CUDA_TRAIN'
    if os.environ.get(flag)!='1':pytest.skip('actual '+name+' hardware required')
    from brian2_rust import training_metal,training_cuda
    module=training_metal if name=='metal' else training_cuda
    with tempfile.TemporaryDirectory(prefix='b2-integer-gpu-') as directory:
        library=module.build(directory)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module,'build',lambda target:Path(shutil.copy2(library,Path(target)/library.name)))
            yield name


def model(kind='add',noisy=False,engine='cpu',ranks=None):
    identity=[[[dict(op='state',index=0)]]]*2
    p=lif_training_plan([1,2,2],projections=[neuron_parameter_bank(4)]*2,state_equations=identity,state_resets=identity,
        threshold=.5,detach_reset=False,clock=dict(origin=0.,dt=.0002),trainable=[True,False],backend=engine,mpi_ranks=ranks,
        noise_streams=[1,1] if noisy else None)
    initial=[.6,.1,.7,.3,-1,16777216,2147483647,-2147483648,0.,0.,0.,0.]
    programs=[];actions=[]
    def add(reads,writes,code,owner,**kwargs):
        index=len(programs);programs.append(code)
        actions.append(dict(owner=owner,reads=reads,writes=writes,program_set=index,threshold=None,trigger=None,parameter_index=owner,**kwargs))
    integer=[dict(op='integer_state',index=0),dict(op='integer_neuron_parameter',bank=1,index=0),
             dict(op='integer_binary',left=0,right=1,kind=kind)]
    if kind=='neg':integer[-1]=dict(op='integer_neg',arg=0)
    boolean=integer+[dict(op='integer_constant',value=0),dict(op='integer_compare',left=2,right=3,kind='gt')]
    for j in range(4):
        add([4+j,8+j],[4+j,8+j],[copy.deepcopy(integer),copy.deepcopy(boolean)],j)
        float_code=[dict(op='state',index=0),dict(op='constant',value=.8),dict(op='mul',left=0,right=1),
            dict(op='state',index=1),dict(op='neuron_parameter',bank=0,index=0),dict(op='mul',left=3,right=4),dict(op='add',left=2,right=5)]
        if noisy:float_code += [dict(op='noise',stream=0),dict(op='constant',value=.05),dict(op='mul',left=7,right=8),dict(op='mul',left=9,right=3),dict(op='add',left=6,right=10)]
        add([j,8+j],[j],[float_code],j,noise_domain=5+j//2,noise_entity=j%2,noise_streams=int(noisy))
    for j in range(4):actions.append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None))
    for j in range(4):
        add([j],[j],[[dict(op='state',index=0),dict(op='constant',value=.5),dict(op='sub',left=0,right=1)]],j)
        actions[-1]['trigger']=dict(external=False,index=j);actions[-1]['detach_trigger']=False
    p.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(initial=initial,initial_parameters=[None]*12,
        detached=[False]*4+[True]*8,voltage=list(range(4)),binary_states=list(range(8,12)),integer_states=list(range(4,8)),
        integer_parameters=[[1,j] for j in range(4)],program_sets=programs,actions=actions))
    return p,[[.2,.3,.4,.5],[1.,1.,1.,-1.]],np.zeros((1,4,1))


def wrap(x):return (int(x)+2**31)%2**32-2**31


def oracle(p,weights,kind='add',noisy=False,anchors=None,initial=None):
    z=np.array(p['dynamic']['initial'] if initial is None else initial,float);voltage=z[:4].copy();counter=z[4:8].astype(np.int64)
    gates=[];spikes=[];history=[]
    for tick in range(4):
        for j in range(4):
            a=int(counter[j]);b=int(weights[1][j])
            counter[j]=wrap({'add':a+b,'sub':a-b,'mul':a*b,'min':min(a,b),'max':max(a,b),'neg':-a}[kind])
        flag=(counter>0).astype(float);voltage=.8*voltage+np.array(weights[0])*flag
        if noisy:voltage+=.05*flag*np.array([normal(p['seed'],7,0,5+j//2,j%2,tick,0) for j in range(4)])
        margin=voltage-.5;s=(margin>0).astype(float)
        if anchors is not None:s=(anchors[tick]>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[tick]))**2*(margin-anchors[tick])
        history.append(margin.copy());spikes.append(s.copy());voltage-=.5*s
    logits=np.array(spikes)[:,2:].mean(0)*p['logit_scale']
    return np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0],np.r_[voltage,counter,flag],np.array(spikes),history


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('noisy',[False,True])
def test_exact_int32_counters_and_detached_gradient(engine,ranks,noisy):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(noisy=noisy,engine=engine,ranks=ranks)
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x,[0],**(dict(noise_sequence=7) if noisy else {}))
    loss,live,spikes,anchors=oracle(p,w,noisy=noisy)
    np.testing.assert_array_equal(np.array(result['final_state'])[0,4:],live[4:])
    np.testing.assert_allclose(result['final_state'][0][:4],live[:4],atol=3e-6,rtol=2e-5)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    assert result['loss']==pytest.approx(loss,rel=1e-5,abs=3e-6)
    expected=[]
    for j in range(4):
        hi=copy.deepcopy(w);lo=copy.deepcopy(w);hi[0][j]+=1e-6;lo[0][j]-=1e-6
        expected.append((oracle(p,hi,noisy=noisy,anchors=anchors)[0]-oracle(p,lo,noisy=noisy,anchors=anchors)[0])/2e-6)
    np.testing.assert_allclose(result['gradients'][0],expected,atol=3e-6,rtol=3e-4)
    np.testing.assert_array_equal(result['gradients'][1],0)
    np.testing.assert_array_equal(result['initial_state_gradients'][0][4:],0)


@pytest.mark.parametrize('kind',['sub','mul','min','max','neg'])
def test_integer_operations_wrap_and_preserve_large_values(engine,kind):
    p,w,x=model(kind,engine=engine)
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).evaluate(x,[0])
    _,live,spikes,_=oracle(p,w,kind)
    np.testing.assert_array_equal(np.array(result['final_state'])[0,4:],live[4:])
    np.testing.assert_array_equal(result['spikes'][0],spikes)


@pytest.mark.parametrize('selected',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_integer_nan_bits_are_not_error_status(engine,ranks,selected):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(engine=engine,ranks=ranks)
    # Returning -1 encodes a NaN float bit pattern; a domain failure must remain
    # distinguishable, including in the first integer slot of an MPI action.
    p['dynamic']['program_sets'][0][0]=[
        dict(op='constant',value=float(selected)),dict(op='constant',value=0.),dict(op='div',left=1,right=1),
        dict(op='integer_cast',arg=2),dict(op='integer_constant',value=-1),dict(op='integer_select',condition=0,yes=3,no=4)]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    if selected:
        with pytest.raises((ValueError,RuntimeError)):trainer.step(x[:,:1],[0])
        assert trainer.state==before and trainer.neuron_state is None
    else:
        result=trainer.gradients(x[:,:1],[0]);assert result['final_state'][0][4]==-1


def test_integer_cast_truncates_and_stops_gradient(engine):
    p,w,x=model(engine=engine);p['dynamic']['integer_parameters']=[];p['trainable'][1]=True;w[1]=[1.8,-1.8,.8,-.8]
    for j in range(4):
        code=[dict(op='neuron_parameter',bank=1,index=0),dict(op='integer_cast',arg=0)]
        p['dynamic']['program_sets'][2*j]=[code,code+[dict(op='integer_constant',value=0),dict(op='integer_compare',left=1,right=2,kind='gt')]]
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[:,:1],[0])
    np.testing.assert_array_equal(result['final_state'][0][4:8],[1,-1,0,0])
    np.testing.assert_array_equal(result['gradients'][1],0)


@pytest.mark.parametrize('ranks',[None,2])
def test_integer_checkpoint_carry(engine,ranks,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=model(noisy=True,engine=engine,ranks=ranks);p['trainable']=[False,False]
    whole=NativeLIFTrainer(p,runner=RUNNER,weights=w).evaluate(x,[0],noise_sequence=7)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);trainer.step(x[:,:2],[0],noise_sequence=7)
    path=tmp_path/'integers.json';trainer.store(path)
    restored=NativeLIFTrainer(p,runner=RUNNER,weights=w);restored.restore(path)
    tail=restored.evaluate(x[:,2:],[0],initial='carry')
    np.testing.assert_array_equal(np.array(tail['final_state'])[:,4:],np.array(whole['final_state'])[:,4:])
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=2e-5,atol=3e-6)


@pytest.mark.parametrize('issue',['detached','duplicate','binary','float_read','float_output','float_operand','trainable','initial_fraction','parameter_fraction'])
def test_invalid_integer_storage_is_rejected_atomically(issue):
    p,w,x=model();d=p['dynamic'];code=d['program_sets'][0][0]
    if issue=='detached':d['detached'][4]=False
    elif issue=='duplicate':d['integer_states'].append(4)
    elif issue=='binary':d['binary_states'].append(4)
    elif issue=='float_read':code[0]['op']='state'
    elif issue=='float_output':code[-1]=dict(op='constant',value=1.)
    elif issue=='float_operand':code[-1]=dict(op='add',left=0,right=1)
    elif issue=='trainable':p['trainable'][1]=True
    elif issue=='initial_fraction':d['initial'][4]=.5
    elif issue=='parameter_fraction':w[1][0]=.5
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step(x,[0])
    assert trainer.state==before and trainer.neuron_state is None
