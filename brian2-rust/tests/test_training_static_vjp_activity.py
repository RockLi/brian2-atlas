"""Legacy scalar/vector GPU reverse uses canonical requested-leaf masks."""
import copy
import importlib
import subprocess

import numpy as np
import pytest

from brian2_rust.training import NativeLIFTrainer, lif_training_plan
from brian2_rust.training_equations import (
    NeuronParameter, NormalNoise, UniformNoise, SimulationTime, compile_training_equation,
    neuron_parameter_bank,
)
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from brian2_rust.training_dynamic import compile_dynamic_transform


def model(mode, expression='sqrt(frozen)', *, neuron=False, ranks=None,
          window=None, detach=False, distribution=None, phase='update'):
    states=None if mode=='scalar' else ['v']
    identity=compile_training_equation('v', states=states)
    p=lif_training_plan([1,1,2], projections=[neuron_parameter_bank(4 if neuron else 2)],
        threshold=.25, detach_reset=detach, tbptt_window=window, mpi_ranks=ranks,
        clock=dict(origin=.013, dt=.001) if mode=='vector' else None,
        noise_streams=[1,1] if distribution else None,
        **(dict(equations=[identity,identity]) if mode=='scalar' else
           dict(state_equations=[[identity],[identity]],state_resets=[[identity],[identity]])))
    p['masks']=[[1.,1.,0.,0.] if neuron else [1.,0.]]
    w=[[.4,.6,0.,0.] if neuron else [.4,0.]]
    bindings={'scale':NeuronParameter(0,0) if neuron else (0,0),
              'frozen':NeuronParameter(0,2) if neuron else (0,1)}
    if mode=='vector':bindings['t']=SimulationTime()
    if distribution:
        bindings['eta']=NormalNoise(0) if distribution=='normal' else UniformNoise(0)
    def install(branch):
        code='v+scale+'+('.1*t+' if mode=='vector' else '')+branch
        if distribution:code+='*eta'
        tr=compile_training_equation(code,states=states,parameters=bindings)
        if mode=='scalar':p['equations'][1]=tr
        else:p['state_'+('resets' if phase=='reset' else 'equations')][1]=[tr]
    install('0.');reference=copy.deepcopy(p)
    install(expression)
    return p,w,reference


def compare(p,w,reference,backend,*,steps=4):
    x=np.zeros((2,steps,1));labels=[0,0];initial=[[.317,.2,.4],[.16,.6,.1]]
    kwargs=dict(initial=initial)
    if p.get('noise_streams') is not None:kwargs['noise_sequence']=9
    if p.get('clock') is not None:kwargs['start_tick']=7
    expected=NativeLIFTrainer(reference,weights=w,runner=RUNNER).gradients(
        x,labels,**kwargs)
    p['backend']=backend
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(
        x,labels,**kwargs)
    for key in ('loss','spikes','gradients','initial_gradients','final_membrane'):
        np.testing.assert_allclose(result[key],expected[key],rtol=5e-5,atol=4e-6)
    assert np.any(np.abs(np.asarray(expected['gradients']))>1e-5)
    assert np.any(np.abs(np.asarray(expected['initial_gradients']))>1e-5)
    assert (result['gpu_dispatches']>0)==(backend!='cpu')
    return result


@pytest.mark.parametrize('mode',['scalar','vector'])
@pytest.mark.parametrize('expression',['sqrt(frozen)','frozen**.5','arccos(1+frozen)'])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_singular_branch_matches_exact_constant_model(engine,mode,expression,ranks):
    mpi(ranks);p,w,ref=model(mode,expression,ranks=ranks)
    r=compare(p,w,ref,engine)
    assert r['gradients'][0][1]==0.


@pytest.mark.parametrize('expression',['sqrt(frozen)','arccos(1+frozen)'])
@pytest.mark.parametrize('phase',['update','reset'])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_per_neuron_canonical_slices_and_reset_vjp(engine,expression,phase,ranks):
    mpi(ranks);p,w,ref=model('vector',expression,neuron=True,ranks=ranks,phase=phase)
    r=compare(p,w,ref,engine)
    np.testing.assert_array_equal(r['gradients'][0][2:],0.)


@pytest.mark.parametrize('distribution',['normal','uniform'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('detach',[False,True])
def test_mask_suffix_follows_clock_and_noise_staging(engine,distribution,window,detach):
    p,w,ref=model('vector',distribution=distribution,window=window,detach=detach)
    compare(p,w,ref,engine)


@pytest.mark.parametrize('expression,reference',[
    ('minimum(scale,sqrt(frozen))','0.'),
    ('maximum(scale,sqrt(frozen))','scale'),
    ('scale if frozen == 0 else sqrt(frozen)','scale'),
    ('sqrt(frozen) if scale > 0 else scale','0.'),
])
def test_vector_shared_lazy_evaluator_uses_legacy_metadata(engine,expression,reference):
    p,w,ref=model('vector')
    for plan,branch in ((p,expression),(ref,reference)):
        transform=compile_dynamic_transform('v=v+.1*t+('+branch+')',states={'v':0},
            state_types={0:'float'},parameters={'scale':(0,0),'frozen':(0,1),'t':SimulationTime()})
        plan['state_equations'][1]=transform['programs']
    # Selected constant branches legitimately yield zero weight gradients.
    x=np.zeros((1,3,1));initial=[[.317,.2,.4]]
    expected=NativeLIFTrainer(ref,weights=w,runner=RUNNER).gradients(x,[0],initial=initial)
    p['backend']=engine
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0],initial=initial)
    for key in ('loss','gradients','initial_gradients','final_membrane'):
        np.testing.assert_allclose(result[key],expected[key],rtol=5e-5,atol=4e-6)


@pytest.mark.parametrize('mode',['scalar','vector'])
@pytest.mark.parametrize('invalid_forward',[False,True])
def test_activity_does_not_hide_active_singularity_or_forward_domain_error(engine,mode,invalid_forward):
    p,w,_=model(mode,'sqrt(frozen-1)' if invalid_forward else 'sqrt(frozen)');p['backend']=engine
    if not invalid_forward:p['masks'][0][1]=1.
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    if not invalid_forward:trainer.evaluate(np.zeros((1,1,1)),[0],initial=[[.317,.2,.4]])
    with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0],initial=[[.317,.2,.4]])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0


@pytest.mark.parametrize('mode',['scalar','vector'])
def test_frozen_optimizer_preserves_requested_vjp_and_checkpoint(engine,mode,tmp_path):
    p,w,ref=model(mode,distribution='normal' if mode=='vector' else None);p['trainable']=[False];ref['trainable']=[False]
    compare(p,w,ref,engine)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    trainer.step(np.zeros((1,3,1)),[0],initial=[[.317,.2,.4]],
        **(dict(noise_sequence=9) if mode=='vector' else {}))
    assert trainer.state['weights']==w
    checkpoint=tmp_path/'frozen';trainer.store(checkpoint)
    restored=NativeLIFTrainer(p,weights=w,runner=RUNNER);restored.restore(checkpoint)
    r=trainer.step(np.zeros((1,2,1)),[0]);q=restored.step(np.zeros((1,2,1)),[0])
    assert r==q and trainer.state==restored.state


@pytest.mark.parametrize('mode',['scalar','vector'])
def test_static_activity_scratch_budget_is_enforced(engine,mode):
    p,w,ref=model(mode);r=compare(p,w,ref,engine)
    p['max_tape_bytes']=r['tape_bytes']-1
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='budget'):
        trainer.step(np.zeros((2,4,1)),[0,0],initial=[[.317,.2,.4],[.16,.6,.1]])
    assert trainer.state==before


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('mode',['scalar','vector','builtin'])
def test_evaluation_and_builtin_reverse_keep_their_existing_capability_contract(backend,mode,tmp_path,monkeypatch):
    if mode=='builtin':p=lif_training_plan([1,1,2]);w=[[.4],[.3,.2]]
    else:p,w,_=model(mode)
    p['backend']=backend
    source=tmp_path/'old.c';library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\n')
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    monkeypatch.setattr(importlib.import_module('brian2_rust.training_'+backend),'build',lambda directory:library)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    # The deliberately missing execution symbol proves the host passed the
    # activity-capability check; this control never starts a GPU dispatch.
    with pytest.raises(ValueError,match='training symbol missing'):
        trainer.evaluate(np.zeros((1,1,1)),[0])
    if mode=='builtin':
        with pytest.raises(ValueError,match='training symbol missing'):
            trainer.gradients(np.zeros((1,1,1)),[0])


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('mode',['scalar','vector'])
@pytest.mark.parametrize('version',[None,0,2])
def test_old_static_gpu_library_must_declare_its_own_activity_capability(backend,mode,version,tmp_path,monkeypatch):
    p,w,_=model(mode);p['backend']=backend
    source=tmp_path/'old.c';library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\n'
        'uint64_t b2_train_vjp_activity_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_static_vjp_activity_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    monkeypatch.setattr(importlib.import_module('brian2_rust.training_'+backend),'build',lambda directory:library)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='GPU static VJP activity capability'):
        trainer.gradients(np.zeros((1,1,1)),[0])
    assert trainer.state==before
