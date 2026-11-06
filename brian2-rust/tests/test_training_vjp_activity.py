"""Requested-leaf VJPs prune inactive singular paths on native CPU/device."""
import copy
import os
import re
from pathlib import Path
import subprocess
import numpy as np
import pytest
from brian2_rust.training import NativeLIFTrainer,lif_training_plan
from brian2_rust.training_equations import PoissonNoise,ParameterBank,_MappedParameter,NeuronParameter,TimedInput,compile_training_equation,neuron_parameter_bank
from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero import zero_model
from test_training_poisson_zero_vjp import mpi

KINDS=['detached_state','indirect_state','masked_parameter','mapped','gather','neuron','timed','constant']


def model(mode,kind='detached_state',singular='sqrt(r)',active=False,ranks=None):
    p,w=zero_model(ranks=ranks);w[0][0]={'zero':0.,'positive':1.2,'pathwise':.4}[mode]
    w[0][2]=0.;p['dynamic']['initial'][4]=1. if 'arccos' in singular else 0.
    p['dynamic']['initial'][5]=p['dynamic']['initial'][4]
    bindings={'draw':PoissonNoise(0),'scale':(0,0),'amp':(0,1)}
    read=[0,2,4]
    if kind in ('detached_state','indirect_state'):
        p['dynamic']['detached'][4]=not active
    else:
        w[0][2]=p['dynamic']['initial'][4];p['masks'][0][2]=float(active)
        binding={'masked_parameter':(0,2),'mapped':_MappedParameter(0,0),'gather':ParameterBank(0),
                 'neuron':NeuronParameter(0,2),'timed':TimedInput(0,3,1,.001,1,2),'constant':w[0][2]}[kind]
        bindings['frozen']=binding
        argument='frozen(2)' if kind=='gather' else 'frozen(.002,0.)' if kind=='timed' else 'frozen'
        singular=re.sub(r'\br\b',lambda _:argument,singular)
        if kind=='mapped':p['dynamic']['parameter_maps']=[[2]]
    def install(branch):
        code='v=scale+'+branch if mode=='pathwise' else 'k=draw(scale+'+branch+')\nv=amp*k'
        tr=compile_dynamic_transform(code,states={'v':0,'k':1,'r':2},state_types={1:'integer'},parameters=bindings)
        p['dynamic']['program_sets']=[tr['programs']]
        p['dynamic']['actions'][0]=dynamic_action(tr,read,owner=1,program_set=0,noise_domain=71,noise_entity=0,noise_streams=1)
        if kind=='indirect_state':
            p['dynamic']['actions'][0]['reads'][2]=5
            p['dynamic']['actions'][0]['indirect']={'reads':{'2':{'index':3,'tables':[[4]]}},'writes':{}}
    install(singular)
    reference=copy.deepcopy(p)
    install('0.');reference['dynamic']['program_sets']=p['dynamic']['program_sets']
    reference['dynamic']['actions']=p['dynamic']['actions']
    # Return the original singular transform plus an otherwise identical model
    # whose finite forward branch is replaced by its exact constant value.
    install(singular)
    return p,w,reference


def check(p,w,reference,engine):
    x=np.zeros((2,2,1));labels=[0,0]
    expected=NativeLIFTrainer(reference,weights=w,runner=RUNNER).gradients(x,labels,noise_sequence=9)
    p['backend']=engine
    r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,labels,noise_sequence=9)
    for key in ('loss','spikes','final_state','gradients','initial_state_gradients'):
        np.testing.assert_allclose(r[key],expected[key],rtol=4e-5,atol=3e-6)
    assert r['backend']==engine and (r['gpu_dispatches']>0)==(engine!='cpu')
    assert np.any(np.abs(np.asarray(expected['gradients']))>1e-5)
    return r


@pytest.mark.parametrize('mode',['zero','positive','pathwise'])
@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('ranks',[None,2])
def test_mixed_active_and_inactive_singular_vjp(engine,mode,kind,ranks):
    mpi(ranks);p,w,ref=model(mode,kind,ranks=ranks);r=check(p,w,ref,engine)
    if kind in ('detached_state','indirect_state'):
        np.testing.assert_array_equal(np.asarray(r['initial_state_gradients'])[:,4],0.)
    assert r['gradients'][0][2]==0.


@pytest.mark.parametrize('mode',['zero','positive','pathwise'])
@pytest.mark.parametrize('kind',['detached_state','masked_parameter'])
@pytest.mark.parametrize('ranks',[None,2])
def test_real_active_singular_derivatives_still_fail_atomically(engine,mode,kind,ranks):
    mpi(ranks);p,w,_=model(mode,kind,active=True,ranks=ranks);p['backend']=engine
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    trainer.evaluate(np.zeros((1,1,1)),[0])
    with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0


@pytest.mark.parametrize('mode',['zero','positive','pathwise'])
@pytest.mark.parametrize('singular',['r**.5','arccos(r)'])
@pytest.mark.parametrize('ranks',[None,2])
def test_pow_and_standard_math_singular_leaves_are_pruned(engine,mode,singular,ranks):
    mpi(ranks);p,w,ref=model(mode,singular=singular,ranks=ranks)
    check(p,w,ref,engine)


@pytest.mark.parametrize('expression',[
    'minimum(scale,sqrt(r))','maximum(scale,sqrt(r))',
    'scale if r == 0 else sqrt(r)','sqrt(r) if scale > 0 else scale',
])
@pytest.mark.parametrize('ranks',[None,2])
def test_actual_selected_branch_activity_preserves_declared_vjp(engine,expression,ranks):
    mpi(ranks);p,w=zero_model(ranks=ranks);w[0][0]=.4;p['dynamic']['initial'][4]=0.;p['dynamic']['detached'][4]=True
    params={'scale':(0,0)}
    code=compile_dynamic_transform('v='+expression,states={'v':0,'k':1,'r':2},state_types={1:'integer'},parameters=params)
    p['dynamic']['program_sets']=[code['programs']]
    p['dynamic']['actions'][0]=dynamic_action(code,[0,2,4],owner=1,program_set=0)
    ref=copy.deepcopy(p)
    value='scale' if expression.startswith('maximum') or expression.startswith('scale if') else '0.'
    tr=compile_dynamic_transform('v='+value,states={'v':0,'k':1,'r':2},state_types={1:'integer'},parameters=params)
    ref['dynamic']['program_sets']=[tr['programs']]
    ref['dynamic']['actions'][0]=dynamic_action(tr,[0,2,4],owner=1,program_set=0)
    x=np.zeros((1,1,1));labels=[0]
    expected=NativeLIFTrainer(ref,weights=w,runner=RUNNER).gradients(x,labels)
    p['backend']=engine;r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,labels)
    for key in ('gradients','initial_state_gradients','final_state'):
        np.testing.assert_allclose(r[key],expected[key],rtol=4e-5,atol=3e-6)


@pytest.mark.parametrize('mode',['scalar','vector'])
def test_cpu_general_clocked_vjp_prunes_masked_parameter(mode):
    identity=compile_training_equation('v',states=None if mode=='scalar' else ['v'])
    p=lif_training_plan([1,1,2],projections=[neuron_parameter_bank(2)],threshold=.25,detach_reset=False,
        **(dict(equations=[identity,identity]) if mode=='scalar' else dict(state_equations=[[identity],[identity]],state_resets=[[identity],[identity]])))
    p['masks']=[[1.,0.]];w=[[.4,0.]]
    def install(expression):
        tr=compile_training_equation(expression,states=None if mode=='scalar' else ['v'],parameters={'scale':(0,0),'frozen':(0,1)})
        if mode=='scalar':p['equations'][1]=tr
        else:p['state_equations'][1]=[tr]
    install('v+scale');ref=copy.deepcopy(p)
    install('v+scale+sqrt(frozen)')
    x=np.zeros((1,1,1));labels=[0];initial=[[.317,.2,.4]]
    expected=NativeLIFTrainer(ref,weights=w,runner=RUNNER).gradients(x,labels,initial=initial)
    r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,labels,initial=initial)
    for key in ('loss','gradients','initial_gradients','final_membrane'):
        np.testing.assert_allclose(r[key],expected[key],rtol=1e-12,atol=1e-13)


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('version',[None,0,2])
def test_old_dynamic_library_cannot_omit_activity_pruning(backend,version,tmp_path,monkeypatch):
    p,w,_=model('pathwise');p['backend']=backend
    source=tmp_path/'old.c';library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_vjp_activity_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    import importlib
    module=importlib.import_module('brian2_rust.training_'+backend)
    monkeypatch.setattr(module,'build',lambda directory:library)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='GPU VJP activity capability'):
        trainer.gradients(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.clock_tick==0 and trainer.next_noise_sequence==0


@pytest.mark.parametrize('distribution',['normal','uniform'])
@pytest.mark.parametrize('ranks',[None,2])
def test_continuous_noise_amplitude_prunes_detached_singular_path(engine,distribution,ranks):
    from brian2_rust.training_equations import NormalNoise,UniformNoise
    mpi(ranks);p,w=zero_model(ranks=ranks);w[0][0]=.4;p['dynamic']['initial'][4]=0.;p['dynamic']['detached'][4]=True
    params={'scale':(0,0),'eta':NormalNoise(0) if distribution=='normal' else UniformNoise(0)}
    def install(expression):
        tr=compile_dynamic_transform('v='+expression,states={'v':0,'k':1,'r':2},state_types={1:'integer'},parameters=params)
        p['dynamic']['program_sets']=[tr['programs']]
        p['dynamic']['actions'][0]=dynamic_action(tr,[0,2,4],owner=1,program_set=0,noise_domain=71,noise_entity=0,noise_streams=1)
    install('scale');ref=copy.deepcopy(p)
    install('scale+sqrt(r)*eta')
    check(p,w,ref,engine)
