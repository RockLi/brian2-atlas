"""Ordered local SSA in stochastic neurons, dynamic edges, and typed paths."""
import ast
import copy
import importlib
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, PureFunction, compile_training_equation
from brian2_rust.training_equations import NormalNoise, NeuronParameter, PoissonNoise, _compile_training_ast
from brian2_rust.training_functions import lower_pure_function
from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
from test_native_training import RUNNER
from test_training_integer_ir import engine, model, oracle
from test_training_poisson_zero_vjp import mpi
from test_training_poisson_ssa import single_transform
import test_training_pure_functions as callbacks


@b.check_units(v=1, result=1)
def local_smooth(v):
    square = v*v
    local = .05*np.sin(v)+.02*np.cos(v)+.03*np.tanh(v)
    local += .02*np.exp(-square)+.01*np.log(square+1)
    local += .01*np.sqrt(square+1)+.002*v**2
    v = local
    return v


@b.check_units(x=1, result=1)
def local_decay(x):
    original = x
    x = .8*x
    correction = .01*np.sin(original)
    x += correction
    return x


def noisy_update(voltage, gate, gain, draw):
    next_voltage = .8*voltage
    next_voltage += gain*gate
    scaled_draw = .05*gate*draw
    return next_voltage+scaled_draw


@pytest.mark.parametrize('ranks', [None, 2, 8])
@pytest.mark.parametrize('reset', ['zero', 'subtract'])
def test_neuron_local_rebinding_all_bank_and_initial_vjps(engine, ranks, reset, monkeypatch):
    monkeypatch.setattr(callbacks, 'smooth', local_smooth)
    callbacks.test_native_callback_all_parameter_and_initial_vjps(engine, ranks, reset)


@pytest.mark.parametrize('position', ['pre', 'post', 'regular', 'synaptic_ode'])
def test_brian_dynamic_local_functions_match_brian(engine, position, monkeypatch):
    monkeypatch.setattr(callbacks, 'decay', local_decay)
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine, position)


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_noisy_dynamic_local_assignments_vjp(engine, ranks):
    mpi(ranks); p,w,x=model(noisy=True, engine=engine, ranks=ranks)
    function=lower_pure_function(noisy_update)
    for action in p['dynamic']['actions']:
        if action.get('noise_streams')!=1:continue
        p['dynamic']['program_sets'][action['program_set']]=compile_dynamic_transform(
            'v=f(v,flag,gain,draw)', states={'v':0,'flag':1},
            parameters={'f':function,'gain':NeuronParameter(0),'draw':NormalNoise(0)})['programs']
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0],noise_sequence=7)
    loss,final,spikes,anchors=oracle(p,w,noisy=True)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],final,rtol=2e-5,atol=3e-6)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    for i in range(4):
        plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[0][i]+=1e-6;minus[0][i]-=1e-6
        fd=(oracle(p,plus,noisy=True,anchors=anchors)[0]-oracle(p,minus,noisy=True,anchors=anchors)[0])/2e-6
        assert result['gradients'][0][i]==pytest.approx(fd,abs=3e-6,rel=3e-4)
    assert result['gradients'][1]==[0.]*4


@pytest.mark.parametrize('ranks', [None, 8])
@pytest.mark.parametrize('kind', ['integer', 'boolean'])
def test_typed_return_preserves_exact_i32_and_boolean(engine, ranks, kind):
    mpi(ranks); p,w,x=model(engine=engine,ranks=ranks)
    identity=PureFunction(('x',),'saved',statements=(('saved','x'),('ignored','2.')))
    for group in p['dynamic']['program_sets']:
        for i,program in enumerate(group):
            root=program[-1]
            if kind=='integer' and root['op']=='integer_binary':
                root_index=len(program)-1
                program.extend([dict(op='constant',value=2.),dict(op='integer_sequence',left=root_index+1,right=root_index)])
            elif kind=='boolean' and root['op']=='integer_compare':
                root_index=len(program)-1
                program.extend([dict(op='constant',value=2.),dict(op='sequence',left=root_index+1,right=root_index,boolean=True)])
    typed=_compile_training_ast(ast.parse('f(k)',mode='eval').body,states=['v','k'],state_types=['float','integer'],parameters={'f':identity})
    assert typed[-1]['op']=='integer_sequence'
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0])
    loss,final,spikes,_=oracle(p,w)
    np.testing.assert_array_equal(result['final_state'][0][4:],final[4:])
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    assert result['gradients'][1]==[0.]*4


def scoped_rebinding(x):
    old=x
    x=x+1
    x*=old
    return x


def no_arguments():
    value=2.
    return value


def unused_argument(x):
    return 1.


def test_source_snapshot_local_ssa_zero_arity_unused_arguments():
    f=lower_pure_function(scoped_rebinding)
    assert f.statements==(('old','x'),('x','x + 1'),('x','x * old'))
    assert compile_training_equation('f(v)',parameters={'f':f})[-1]['op']=='sequence'
    assert compile_training_equation('f()',parameters={'f':lower_pure_function(no_arguments)})[-1]['op']=='sequence'
    draw=compile_training_equation('f(draw)',parameters={'f':lower_pure_function(unused_argument),'draw':NormalNoise(0)})
    assert sum(n['op']=='noise' for n in draw)==1 and draw[-1]['op']=='sequence'


def read_before_assignment(x):
    y=y+x
    return y


def multiple_return(x):
    return x
    return x+1


def runtime_return(x):
    if x:
        return x
    return x+1


def mutation(x):
    values=[x]
    values[0]+=1
    return values[0]


def borrowed_mutation(x):
    saved=x
    x*=.8
    return saved+.01*np.sin(x)


def aliased_local_mutation(x):
    local=x+1
    saved=local
    local+=2
    return saved


def identity(x):
    return x


def helper_alias_mutation(x):
    local=x+1
    saved=identity(local)
    local+=2
    return saved


def integer_casting_mutation(x):
    local=x+1
    local*=.5
    return local


def test_augmented_integer_array_cannot_silently_change_dtype():
    function=lower_pure_function(integer_casting_mutation)
    assert compile_training_equation('f(v)',parameters={'f':function})
    with pytest.raises(ValueError,match='changes array dtype'):
        _compile_training_ast(ast.parse('f(k)',mode='eval').body,states=['v','k'],
            state_types=['float','integer'],parameters={'f':function})


def local_remainder(x,y):
    remainder=x%y
    remainder+=0
    return remainder


@pytest.mark.parametrize('ranks', [None, 8])
def test_local_integer_remainder_matches_numpy_at_wrap_boundaries(engine,ranks):
    mpi(ranks);p,w,x=model(engine=engine,ranks=ranks)
    p['dynamic']['initial'][4:8]=[-2.,2147483647.,-2147483648.,16777217.]
    w[1]=[-2147483647.,-7.,7.,2147483647.]
    function=lower_pure_function(local_remainder)
    for action in p['dynamic']['actions']:
        group=p['dynamic']['program_sets'][action['program_set']] if action.get('program_set') is not None else []
        if len(group)!=2:continue
        counter=_compile_training_ast(ast.parse('f(v,rhs)',mode='eval').body,states=['v','flag'],
            state_types=['integer','boolean'],parameters={'f':function,'rhs':NeuronParameter(1,dtype='integer')})
        root=len(counter)-1
        predicate=counter+[dict(op='integer_constant',value=0),dict(op='integer_compare',left=root,right=root+1,kind='gt')]
        p['dynamic']['program_sets'][action['program_set']]=[counter,predicate]
    expected=np.remainder(np.asarray(p['dynamic']['initial'][4:8],dtype=np.int32),np.asarray(w[1],dtype=np.int32))
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x[:,:1],[0])
    np.testing.assert_array_equal(result['final_state'][0][4:8],expected)
    np.testing.assert_array_equal(result['final_state'][0][8:],expected>0)
    assert result['gradients'][1]==[0.]*4


@pytest.mark.parametrize('function', [read_before_assignment, runtime_return, mutation, borrowed_mutation, aliased_local_mutation, helper_alias_mutation])
def test_invalid_statements_are_refused_without_running(function):
    with pytest.raises(ValueError):lower_pure_function(function)


@pytest.mark.parametrize('ranks', [None, 2, 8])
@pytest.mark.parametrize('rate', [0., 1.2])
def test_unused_argument_poisson_is_observed_but_zero_weak_change_is_zero(engine,ranks,rate):
    mpi(ranks);p,w=single_transform('k=draw(scale)\nv=v')
    p['backend']=engine;p['mpi_ranks']=ranks;w[0][0]=rate
    transform=compile_dynamic_transform('v=f(v,draw(scale))',states={'v':0,'k':1,'r':2},state_types={1:'integer'},
        parameters={'f':PureFunction(('x','unused'),'x'),'draw':PoissonNoise(0),'scale':(0,0)})
    p['dynamic']['program_sets']=[transform['programs']]
    p['dynamic']['actions'][0]=dynamic_action(transform,[0,2,4],owner=1,program_set=0,noise_domain=71,noise_entity=0,noise_streams=1)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    result=trainer.gradients(np.zeros((1,2,1)),[0])
    assert result['gradients'][0][0]==pytest.approx(0.,abs=3e-6) if rate==0 else np.isfinite(result['gradients'][0][0])
    # Invalid unused arguments must fail, including with a lazy Poisson node.
    w[0][0]=-1.
    bad=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(bad.state)
    with pytest.raises(ValueError):bad.step(np.zeros((1,1,1)),[0])
    assert bad.state==before and bad.clock_tick==0 and bad.neuron_state is None


@pytest.mark.parametrize('ranks', [None, 8])
@pytest.mark.parametrize('selected', [False, True])
def test_discarded_local_primal_checks_and_lazy_branch_atomicity(engine,ranks,selected):
    mpi(ranks);p,w,x=model(engine=engine,ranks=ranks)
    function=PureFunction(('x',),'x',statements=(('discarded','1./0.'),))
    for group in p['dynamic']['program_sets']:
        for i,program in enumerate(group):
            if program[-1]['op']!='add':continue
            group[i]=_compile_training_ast(ast.parse('f(v) if choose else v',mode='eval').body,
                states=['v','flag'],parameters={'f':function,'choose':float(selected)},allow_select=True,typed=True)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    if selected:
        with pytest.raises(ValueError):trainer.step(x,[0])
        assert trainer.state==before and trainer.clock_tick==0 and trainer.neuron_state is None
    else:
        result=trainer.gradients(x,[0]);assert np.isfinite(result['loss'])


@pytest.mark.parametrize('ranks', [None, 8])
def test_discarded_singular_local_has_no_vjp(engine,ranks):
    mpi(ranks);p,w,x=model(engine=engine,ranks=ranks);w[0]=[0.]*4
    function=PureFunction(('voltage','flag','gain'),'next_voltage',
        statements=(('ignored','sqrt(gain)'),('next_voltage','.8*voltage+gain*flag')))
    for action in p['dynamic']['actions']:
        group=p['dynamic']['program_sets'][action['program_set']] if action.get('program_set') is not None else []
        if not group or group[0][-1]['op']!='add':continue
        p['dynamic']['program_sets'][action['program_set']]=compile_dynamic_transform(
            'v=f(v,flag,gain)',states={'v':0,'flag':1},
            parameters={'f':function,'gain':NeuronParameter(0)})['programs']
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,[0])
    loss,final,spikes,anchors=oracle(p,w)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_state'][0],final,rtol=2e-5,atol=3e-6)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    for i in range(4):
        plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[0][i]+=1e-6;minus[0][i]-=1e-6
        fd=(oracle(p,plus,anchors=anchors)[0]-oracle(p,minus,anchors=anchors)[0])/2e-6
        assert result['gradients'][0][i]==pytest.approx(fd,abs=3e-6,rel=3e-4)


@pytest.mark.parametrize('backend', ['metal','cuda'])
@pytest.mark.parametrize('version', [None, 0, 2])
@pytest.mark.parametrize('dynamic', [False, True])
def test_sequence_gpu_capability_is_checked_before_execution(backend,version,dynamic,tmp_path,monkeypatch):
    if dynamic:p,w,x=model(engine=backend)
    else:
        from test_native_training_equations import fixture_equation
        p,w,x,_,_=fixture_equation('subtract',False,None,backend)
        f=PureFunction(('x',),'x')
        p['equations']=[compile_training_equation('f(v)',parameters={'f':f}) for _ in range(2)]
    if dynamic:
        p['dynamic']['program_sets'][0][0]+=[dict(op='integer_sequence',left=0,right=2)]
    source=tmp_path/'old.c';library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_sequence_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    monkeypatch.setattr(importlib.import_module('brian2_rust.training_'+backend),'build',lambda directory:library)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='GPU sequence capability'):trainer.gradients(x,(np.arange(len(x))%2).tolist())
    assert trainer.state==before and trainer.neuron_state is None
