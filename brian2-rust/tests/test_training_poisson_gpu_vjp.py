"""Zero rate VJP classification on actual device interpreters (not CPU fallback)."""
import copy
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from brian2_rust.training import NativeLIFTrainer
from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
from brian2_rust.training_equations import PoissonNoise, ParameterBank, _MappedParameter, NeuronParameter, TimedInput
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_ssa import single_transform
from test_training_poisson_zero import zero_model
from test_training_poisson_zero_vjp import alias_model, assert_zero, mpi


def check(plan,weights,engine):
    plan['backend']=engine
    trainer,result=assert_zero(plan,weights)
    assert result['backend']==engine and (result['gpu_dispatches']>0)==(engine!='cpu')
    return trainer,result


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('rate',['scale-scale','0.*scale','r-r','scale if r < 0 else 0.',
                                 'minimum(scale,0.)','sin(scale)-sin(scale)'])
def test_zero_rate_cancellation_on_device(engine,rate,ranks):
    mpi(ranks)
    plan,weights=single_transform(f'k=draw({rate})\nv=1./(1.-k)');plan['mpi_ranks']=ranks
    check(plan,weights,engine)


@pytest.mark.parametrize('kind',['masked_parameter','detached_state'])
@pytest.mark.parametrize('singular',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_and_detached_device_rate_paths(engine,kind,singular,ranks):
    mpi(ranks)
    rate='scale' if kind=='masked_parameter' else 'r'
    if singular:rate=('r-r+sqrt(scale)' if kind=='masked_parameter' else 'scale-scale+sqrt(r)')
    plan,weights=zero_model(f'k=draw({rate})\nv=1./(1.-k)',ranks)
    if kind=='masked_parameter':plan['masks'][0][0]=0.
    else:plan['dynamic']['initial'][4]=0.;plan['dynamic']['detached'][4]=True
    check(plan,weights,engine)


@pytest.mark.parametrize('indirect',[False,True])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_device_vjp_merges_physical_state_aliases(engine,indirect,ranks):
    mpi(ranks)
    plan,weights=alias_model(indirect,True);plan['mpi_ranks']=ranks
    check(plan,weights,engine)


@pytest.mark.parametrize('kind',['mapped','gather','neuron','timed'])
@pytest.mark.parametrize('ranks',[None,2])
def test_device_vjp_merges_parameter_aliases(engine,kind,ranks):
    mpi(ranks)
    plan,weights=single_transform('k=draw(scale)');plan['mpi_ranks']=ranks
    alias={'mapped':_MappedParameter(0,0),'gather':ParameterBank(0),'neuron':NeuronParameter(0),'timed':TimedInput(0,3,1,.001,1,2)}[kind]
    rate='alias(k)-scale' if kind=='gather' else 'alias(.002,0.)-scale' if kind=='timed' else 'alias-scale'
    tr=compile_dynamic_transform(f'k=draw({rate})\nv=1./(1.-k)',states={'v':0,'k':1,'r':2},state_types={1:'integer'},
        parameters={'draw':PoissonNoise(0),'alias':alias,'scale':(0,2 if kind=='timed' else 0)})
    plan['dynamic']['program_sets']=[tr['programs']]
    plan['dynamic']['actions'][0]=dynamic_action(tr,[0,2,4],owner=1,program_set=0,noise_domain=71,noise_entity=0,noise_streams=1)
    if kind=='mapped':plan['dynamic']['parameter_maps']=[[0]]
    check(plan,weights,engine)


@pytest.mark.parametrize('kind',['plain','tiny','frozen_bank','distinct','indirect_distinct','singular'])
@pytest.mark.parametrize('ranks',[None,2])
def test_nonzero_and_singular_device_vjp_still_fail_invalid_boundary(engine,kind,ranks):
    mpi(ranks)
    if kind.endswith('distinct'):plan,weights=alias_model(kind.startswith('indirect'),False)
    else:
        rate='1e-20*scale' if kind=='tiny' else 'sqrt(scale)' if kind=='singular' else 'scale'
        plan,weights=zero_model(f'k=draw({rate})\nv=1./(1.-k)')
        if kind=='frozen_bank':plan['trainable']=[False]
    plan.update(backend=engine,mpi_ranks=ranks)
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER);before=copy.deepcopy(trainer.state)
    trainer.evaluate(np.zeros((1,1,1)),[0])
    with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('version',[None,0,2])
def test_old_gpu_rate_vjp_capability_rejected_before_dispatch(backend,version,tmp_path,monkeypatch):
    # A host capability probe, independent of NVIDIA availability.
    plan,weights=zero_model();plan['backend']=backend
    source=tmp_path/'old.c';library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\nuint64_t b2_train_poisson_shared_v1(void){return 1;}\nuint64_t b2_train_poisson_persistent_v1(void){return 1;}\n'
        'uint64_t b2_train_poisson_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_poisson_vjp_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    import importlib
    builder=importlib.import_module('brian2_rust.training_'+backend)
    monkeypatch.setattr(builder,'build',lambda directory:library)
    trainer=NativeLIFTrainer(plan,weights=weights,runner=RUNNER)
    before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='Poisson rate VJP capability'):
        trainer.gradients(np.zeros((1,1,1)),[0])
    assert trainer.state==before


@pytest.mark.parametrize('ranks',[None,2])
def test_device_unit_probe_does_not_pollute_positive_score_or_pathwise_gradients(engine,ranks):
    mpi(ranks)
    plan,weights=single_transform('k=draw(scale-scale)\nv=amp*other(scale)')
    reference=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((3,2,1)),[0,1,0],noise_sequence=9)
    plan.update(backend=engine,mpi_ranks=ranks)
    result=NativeLIFTrainer(plan,weights=weights,runner=RUNNER).gradients(np.zeros((3,2,1)),[0,1,0],noise_sequence=9)
    for key in ('loss','spikes','final_state','gradients','initial_state_gradients'):
        np.testing.assert_allclose(result[key],reference[key],rtol=4e-5,atol=3e-6)
    assert np.any(np.asarray(reference['gradients'])!=0.)
    assert result['backend']==engine and (result['gpu_dispatches']>0)==(engine!='cpu')
