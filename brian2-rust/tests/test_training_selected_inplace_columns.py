"""Selected inplace results preserve columns across repeated target actions."""
import ast
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training import lif_training_plan
from brian2_rust.training_equations import neuron_parameter_bank
from brian2_rust.training_effects import StateEffectFunction,compile_state_effect_transform
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(backend,ranks,reload,alias,accumulator_input=False,inline=False):
    weights=[[0.],[0.,0.],[.19,.11]]
    identity=[[[dict(op='state',index=0)]]]*2
    plan=lif_training_plan([1,1,2],projections=[neuron_parameter_bank(len(row)) for row in weights],
        state_equations=identity,state_resets=identity,clock=dict(origin=0.,dt=.0002),
        threshold=.5,detach_reset=False,trainable=[False,False,True],backend=backend,mpi_ranks=ranks)
    initial=[.2,.217,.719,.113,.173,.173,.113]
    programs=[];actions=[]
    for column,edge in enumerate((1,0)):
        code='h*=.7\n'+('saved=v\nsaved+=gain*h\nv=saved' if alias else 'v+=gain*h')
        if accumulator_input:code='v+=gain*curve(v)' if inline else 'h=curve(v)\nv+=gain*h'
        compiled=compile_state_effect_transform(code,states={'h':0,'v':1,'first':2,'second':3},
            parameters={'gain':(2,column),'curve':StateEffectFunction(('x',),'x*=.7\nreturn x')},array_states={'h','v'},writable_states={'h','v'},
            copied_array_states={'h','v'},reload_arrays_each_statement=reload,
            selected_vectors={('v' if accumulator_input else 'h'):(ast.Name(id='first',ctx=ast.Load()),ast.Name(id='second',ctx=ast.Load()))},
            selected_accumulators={'v'} if accumulator_input else (),
            selected_output=(ast.Constant(column),ast.Constant(2.)))
        reads=[3+edge,1,5,6]
        programs.append(compiled['programs'])
        actions.append(dict(owner=1,reads=reads,writes=[reads[i] for i in compiled['writes']],
            program_set=len(programs)-1,threshold=None,trigger=None))
    for j in range(3):actions.append(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None))
    plan.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(initial=initial,
        initial_parameters=[None]*len(initial),detached=[False]*len(initial),voltage=[0,1,2],
        program_sets=programs,actions=actions))
    return plan,weights


def reference(plan,weights,initial=None,anchors=None,inline=False):
    z=np.array(plan['dynamic']['initial'] if initial is None else initial,float)
    values=.7*z[[5,6]]
    if not inline:z[[4,3]]=values
    for coefficient,value in zip(weights[2],values):z[1]+=coefficient*value
    margins=z[:3]-.5;spikes=(margins>0).astype(float)
    if anchors is not None:
        old=anchors
        spikes=(old>0).astype(float)+plan['surrogate']['scale']/(1+plan['surrogate']['slope']*abs(old))**2*(margins-old)
    logits=spikes[1:]*plan['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,margins


@pytest.mark.parametrize('reload',[False,True])
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_inplace_repeated_target_all_vjps(engine,reload,alias,ranks):
    mpi(ranks);plan,weights=model(engine,ranks,reload,alias)
    check_gradients(plan,weights)


def check_gradients(plan,weights,inline=False):
    out=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0])
    loss,z,anchors=reference(plan,weights,inline=inline)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(weights):
        for index in range(len(row)):
            hi=copy.deepcopy(weights);lo=copy.deepcopy(weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(plan,hi,anchors=anchors,inline=inline)[0]-reference(plan,lo,anchors=anchors,inline=inline)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        hi=np.array(plan['dynamic']['initial']);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(plan,weights,hi,anchors,inline)[0]-reference(plan,weights,lo,anchors,inline)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('inline',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_accumulator_callback_argument_all_vjps(engine,inline,ranks):
    mpi(ranks);plan,weights=model(engine,ranks,True,False,accumulator_input=True,inline=inline)
    check_gradients(plan,weights,inline)
