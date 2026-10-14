"""Do not replay a zero-rate site whose actual baseline VJP vanishes."""
import copy
import os
import numpy as np
import pytest
from brian2_rust.training import NativeLIFTrainer
from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
from brian2_rust.training_equations import PoissonNoise, ParameterBank, _MappedParameter, NeuronParameter
from test_native_training import RUNNER
from test_training_poisson_zero import zero_model, hard_loss
from test_training_poisson_ssa import single_transform


def mpi(ranks):
    if ranks and os.environ.get('B2_TEST_MPI') != '1': pytest.skip('local MPI opt-in')


def assert_zero(plan, weights, length=2):
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    result=trainer.gradients(np.zeros((2,length,1)),[0,1])
    np.testing.assert_array_equal(result['gradients'],np.zeros_like(weights))
    np.testing.assert_array_equal(np.asarray(result['initial_state_gradients'])[:,[0,2,4,5]],np.zeros((2,4)))
    assert result['final_state'][0][2]==0.
    # This branch is finite only for the actual zero count. A spurious replay
    # would evaluate 1/(1-1) and fail, as the original regression did.
    assert result['final_state'][0][0]==1.
    return trainer,result


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('rate',[
    'scale-scale','0.*scale','r-r','scale if r < 0 else 0.',
    'minimum(scale,0.)','sin(scale)-sin(scale)',
])
def test_cancelled_or_selected_constant_rate(rate,ranks):
    mpi(ranks)
    plan,weights=single_transform(f'k=draw({rate})\nv=1./(1.-k)');plan['mpi_ranks']=ranks
    assert_zero(plan,weights)


@pytest.mark.parametrize('kind',['masked_parameter','detached_state'])
@pytest.mark.parametrize('ranks',[None,2])
def test_no_unmasked_rate_gradient(kind,ranks):
    mpi(ranks)
    rate='scale' if kind=='masked_parameter' else 'r'
    plan,weights=zero_model(f'k=draw({rate})\nv=1./(1.-k)',ranks)
    if kind=='masked_parameter':plan['masks'][0][0]=0.
    else:
        plan['dynamic']['initial'][4]=0.
        plan['dynamic']['detached'][4]=True
    assert_zero(plan,weights)


def alias_model(indirect,alias):
    plan,weights=single_transform('k=draw(scale)')
    tr=compile_dynamic_transform('k=draw(r-peer)\nv=1./(1.-k)',
        states={'v':0,'k':1,'r':2,'peer':3},state_types={1:'integer'},parameters={'draw':PoissonNoise(0)})
    plan['dynamic']['program_sets']=[tr['programs']]
    other=4 if alias and not indirect else 5
    action=dynamic_action(tr,[0,2,4,other],owner=1,program_set=0,noise_domain=71,noise_entity=0,noise_streams=1)
    if indirect:
        action['indirect']={'reads':{'3':{'index':2,'tables':[[4 if alias else 5]]}},'writes':{}}
    plan['dynamic']['actions'][0]=action
    plan['dynamic']['initial'][5]=plan['dynamic']['initial'][4]
    return plan,weights


@pytest.mark.parametrize('indirect',[False,True])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_physical_state_alias_cancellation(indirect,ranks):
    mpi(ranks)
    plan,weights=alias_model(indirect,True);plan['mpi_ranks']=ranks
    assert_zero(plan,weights)


@pytest.mark.parametrize('kind',['mapped','gather','neuron'])
@pytest.mark.parametrize('ranks',[None,2])
def test_parameter_aliases_aggregate_before_zero_test(kind,ranks):
    mpi(ranks)
    plan,weights=single_transform('k=draw(scale)');plan['mpi_ranks']=ranks
    alias={'mapped':_MappedParameter(0,0),'gather':ParameterBank(0),'neuron':NeuronParameter(0)}[kind]
    rate='alias(k)-scale' if kind=='gather' else 'alias-scale'
    tr=compile_dynamic_transform(f'k=draw({rate})\nv=1./(1.-k)',states={'v':0,'k':1,'r':2},state_types={1:'integer'},
        parameters={'draw':PoissonNoise(0),'alias':alias,'scale':(0,0)})
    plan['dynamic']['program_sets']=[tr['programs']]
    plan['dynamic']['actions'][0]=dynamic_action(tr,[0,2,4],owner=1,program_set=0,noise_domain=71,noise_entity=0,noise_streams=1)
    if kind=='mapped':plan['dynamic']['parameter_maps']=[[0]]
    assert_zero(plan,weights)


@pytest.mark.parametrize('kind',['plain','tiny','frozen_bank','detached_plus_parameter','distinct','indirect_distinct'])
@pytest.mark.parametrize('ranks',[None,2])
def test_zero_value_with_nonzero_vjp_still_replays_and_rolls_back(kind,ranks):
    mpi(ranks)
    if kind.endswith('distinct'):
        plan,weights=alias_model(kind.startswith('indirect'),False)
    else:
        rate='1e-300*scale' if kind=='tiny' else 'r+scale' if kind=='detached_plus_parameter' else 'scale'
        plan,weights=zero_model(f'k=draw({rate})\nv=1./(1.-k)')
        if kind=='frozen_bank':plan['trainable']=[False] # optimizer freezing does not remove requested gradients
        if kind=='detached_plus_parameter':
            plan['dynamic']['initial'][4]=0.;plan['dynamic']['detached'][4]=True
    plan['mpi_ranks']=ranks
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    before=copy.deepcopy(trainer.state)
    trainer.evaluate(np.zeros((1,1,1)),[0])
    with pytest.raises(ValueError,match='nonfinite equation'):
        trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0


def test_stationary_rate_with_valid_alternate_has_zero_first_order_gradient():
    plan,weights=zero_model('k=draw(scale*scale)\nv=amp*k')
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0])
    np.testing.assert_array_equal(result['gradients'],np.zeros((1,3)))


def test_selected_parameter_branch_retains_weak_derivative():
    plan,weights=zero_model('k=draw(scale if r > 0 else 0.)\nv=amp*k')
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((1,1,1)),[0])
    expected=hard_loss(plan['logit_scale'])-np.log(2.)
    assert result['gradients'][0][0]==pytest.approx(expected,abs=2e-14)


@pytest.mark.parametrize('kind',['masked_parameter','detached_state'])
@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_singular_derivatives_on_inactive_rate_paths_are_not_evaluated(kind,mixed,ranks):
    mpi(ranks)
    rate='sqrt(scale)' if kind=='masked_parameter' else 'sqrt(r)'
    if mixed: rate=('r-r+' if kind=='masked_parameter' else 'scale-scale+')+rate
    plan,weights=zero_model(f'k=draw({rate})\nv=1./(1.-k)',ranks)
    if kind=='masked_parameter':plan['masks'][0][0]=0.
    else:
        plan['dynamic']['initial'][4]=0.;plan['dynamic']['detached'][4]=True
    assert_zero(plan,weights)


def test_singular_active_rate_derivative_remains_an_error():
    plan,weights=zero_model('k=draw(sqrt(scale))\nv=amp*k')
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    with pytest.raises(ValueError,match='nonfinite equation derivative'):
        trainer.gradients(np.zeros((1,1,1)),[0])
