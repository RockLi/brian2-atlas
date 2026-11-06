"""Piecewise stochastic callbacks: independent trajectory/VJP and eager/lazy law."""
import ast
import copy
import importlib
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, PureFunction, compile_training_equation, lower_brian_dynamic_training
from brian2_rust.training_equations import NormalNoise, NeuronParameter, PoissonNoise, _compile_training_ast
from brian2_rust.training_functions import lower_pure_function
from brian2_rust.training_dynamic import compile_dynamic_transform, dynamic_action
from test_native_training import RUNNER
from test_native_training_equations import fixture_equation
from test_training_integer_ir import engine, model
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_training_poisson_ssa import single_transform
import test_training_pure_functions as callbacks
from test_training_boolean_thresholds import stationary
from test_training_dynamic_thresholds import bank


@b.check_units(v=1, draw=1, result=1)
def piecewise(v,draw):
    active=np.logical_and(v>=.2,v<.7)
    positive=.04*np.sin(v)+.01*draw
    other=-.03*np.cos(v)-.02*draw
    return np.where(active,positive,other)


def trajectory(p,w,x,labels,initial,anchors=None,sequence=9):
    batch,length,_=x.shape;v=initial.copy();before=[];margins=[];spikes=[]
    theta=np.repeat(w[4][4:6],2);tau=np.repeat(w[4][0:4:2],2);bias=np.repeat(w[4][1:4:2],2)
    for tick in range(length):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:v=anchors['before'][tick].copy()
        before.append(v.copy())
        noise=np.asarray([[normal(p['seed'],sequence,sample,j//2,j%2,tick,0) for j in range(4)] for sample in range(batch)])
        term=np.where((v>=.2)&(v<.7),.04*np.sin(v)+.01*noise,-.03*np.cos(v)-.02*noise)
        u=v+.25*(-v/tau+term)+bias;margin=u-theta;event=(margin>0).astype(float)
        if anchors is not None:
            base=anchors['margins'][tick]
            event=(base>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margin-base)
        gate=(margin>0).astype(float) if anchors is None or p['detach_reset'] else event
        v=u.copy() if p['reset']=='zero' else u-theta*gate
        for projection,bank in zip(p['projections'][:4],w):
            source=projection['source_layer'];target=projection['target_layer']
            for a,z,slot in zip(projection['sources'],projection['targets'],projection['parameter_ids']):
                src=x[:,tick,a] if source==0 else event[:,2*(source-1)+a]
                v[:,2*(target-1)+z]+=src*bank[slot]
        if p['reset']=='zero':v*=1-gate
        margins.append(margin.copy());spikes.append(event.copy())
    spikes=np.stack(spikes,axis=1);logits=spikes[:,:,2:].mean(axis=1)*p['logit_scale']
    maximum=logits.max(axis=1);loss=np.mean(maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=1))-logits[np.arange(batch),labels])
    return loss,spikes,v,dict(before=before,margins=margins)


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('reset',['zero','subtract'])
@pytest.mark.parametrize('window',[None,2])
def test_piecewise_stochastic_all_bank_and_initial_vjps(engine,ranks,reset,window):
    mpi(ranks);p,w,x,labels,initial=fixture_equation(reset,window is not None,window,engine)
    p['mpi_ranks']=ranks;p['clock']=dict(origin=0.,dt=.001);p['noise_streams']=[1,1];p['seed']=731
    function=lower_pure_function(b.Function(piecewise))
    p['equations']=[compile_training_equation('v+dt*(-v/tau+curve(v,draw))+bias',
        parameters={'dt':.25,'tau':(4,2*l),'bias':(4,2*l+1),'curve':function,'draw':NormalNoise(0)}) for l in range(2)]
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,labels,initial=initial,noise_sequence=9)
    loss,spikes,final,anchors=trajectory(p,w,x,labels,initial)
    np.testing.assert_array_equal(result['spikes'],spikes)
    np.testing.assert_allclose(result['final_membrane'],final,rtol=2e-5,atol=3e-6)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    for bank,row in enumerate(w):
        for j in range(len(row)):
            plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[bank][j]+=1e-6;minus[bank][j]-=1e-6
            fd=(trajectory(p,plus,x,labels,initial,anchors)[0]-trajectory(p,minus,x,labels,initial,anchors)[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(fd,abs=3e-6,rel=4e-4)
    for sample in range(len(initial)):
        for j in range(initial.shape[1]):
            plus=initial.copy();minus=initial.copy();plus[sample,j]+=1e-6;minus[sample,j]-=1e-6
            fd=(trajectory(p,w,x,labels,plus,anchors)[0]-trajectory(p,w,x,labels,minus,anchors)[0])/2e-6
            assert result['initial_gradients'][sample][j]==pytest.approx(fd,abs=3e-6,rel=4e-4)


@b.check_units(x=1,result=1)
def vector_decay(x):
    return np.where(x>.45,.65*x+.02*np.sin(x),.9*x-.01*np.cos(x))


@pytest.mark.parametrize('position',['pre','post','regular','synaptic_ode'])
def test_piecewise_dynamic_positions_match_brian(engine,position,monkeypatch):
    monkeypatch.setattr(callbacks,'decay',vector_decay)
    callbacks.test_dynamic_brian_callback_positions_and_forward(engine,position)


@pytest.mark.parametrize('ranks',[None,8])
@pytest.mark.parametrize('kind',['scalar_if','scalar_and','scalar_or','numpy_where','numpy_and','numpy_or'])
def test_scalar_lazy_and_numpy_eager_errors_are_atomic(engine,ranks,kind):
    mpi(ranks);p,w,x=model(engine=engine,ranks=ranks)
    expression={'scalar_if':'1./0. if x else .8*v',
        'scalar_and':'x and 1./0.','scalar_or':'x or 1./0.',
        'numpy_where':'_b2_where(x,1./0.,.8*v)',
        'numpy_and':'_b2_logical_and(x,1./0.)','numpy_or':'_b2_logical_or(x,1./0.)'}[kind]
    choice=1. if kind=='scalar_or' else 0.
    f=PureFunction(('v','x'),expression)
    for action in p['dynamic']['actions']:
        group=p['dynamic']['program_sets'][action['program_set']] if action.get('program_set') is not None else []
        if not group or group[0][-1]['op']!='add':continue
        p['dynamic']['program_sets'][action['program_set']]=compile_dynamic_transform('v=f(v,choice)',
            states={'v':0,'flag':1},parameters={'f':f,'choice':choice})['programs']
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    if kind.startswith('numpy'):
        with pytest.raises(ValueError):trainer.step(x,[0])
        assert trainer.state==before and trainer.clock_tick==0 and trainer.neuron_state is None
    else:assert np.isfinite(trainer.gradients(x,[0])['loss'])


@pytest.mark.parametrize('expression', ['x and y','x or y','x if c else y'])
def test_scalar_operand_return_and_integer_kind(expression):
    f=PureFunction(('x','y','c'),expression)
    code=_compile_training_ast(ast.parse('f(k,other,choice)',mode='eval').body,states=['v','k','other'],
        state_types=['float','integer','integer'],parameters={'f':f,'choice':0.})
    assert code[-1]['op']=='integer_sequence'


FLAG=np.bool_(False)
def captured_boolean(x):
    return np.where(FLAG,x,x+1)


def python_array_condition(x):
    return x if x>0 else -x


def python_array_chain(x):
    return 0<x<1


def test_boolean_closure_snapshot_and_ambiguous_array_refusal():
    descriptor=lower_pure_function(captured_boolean)
    assert dict(descriptor.parameters)=={'FLAG':False}
    assert compile_training_equation('f(v)',parameters={'f':descriptor})
    for function in (python_array_condition,python_array_chain):
        with pytest.raises(ValueError,match='array callback conditions'):lower_pure_function(function)


def test_numpy_where_identity_guard(monkeypatch):
    monkeypatch.setattr(np,'where',lambda condition,yes,no:no)
    with pytest.raises(ValueError,match='unsupported pure function math attribute'):lower_pure_function(piecewise)


def eager_and(v,theta,a,limit):
    return np.logical_and(v>theta,a>=limit)


def eager_or(v,theta,a,limit):
    return np.logical_or(v>theta,a>=limit)


def boolean_where(v,theta,a,limit):
    return np.where(v>=0,np.logical_and(v>theta,a>=limit),False)


@pytest.mark.parametrize('kind',['and','or','where'])
@pytest.mark.parametrize('values',[([.5,.7],[.3,.1]),([.3,.7],[.1,.5])])
@pytest.mark.parametrize('ranks',[None,8])
def test_eager_boolean_threshold_all_parameter_initial_vjps(engine,kind,values,ranks):
    mpi(ranks);net,g,_=stationary('v>theta and a>=limit',values);b.prefs.codegen.target='numpy'
    function=b.Function({'and':eager_and,'or':eager_or,'where':boolean_where}[kind],
        arg_units=[1]*4,return_unit=bool,arg_types=['float']*4,return_type='boolean')
    g.events['spike']='predicate(v,theta,a,limit)';g.namespace['predicate']=function
    inp=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    groups=[obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g]+[g]
    bundle=lower_brian_dynamic_training(
        net,input_group=inp,layers=groups,trainable_neuron_parameters={g.name:['theta','limit']},
        backend=engine,mpi_ranks=ranks,surrogate_slope=3.,surrogate_scale=.8)
    p=bundle.plan
    def reference(weights,initial,anchors=None):
        v=np.asarray(initial)[bundle.provenance['neuron_state_layout'][g.name]['v']]
        a=np.asarray(initial)[bundle.provenance['neuron_state_layout'][g.name]['a']]
        theta=np.asarray(weights[bank(bundle,g,'theta')[0]]);limit=np.asarray(weights[bank(bundle,g,'limit')[0]])
        margins=np.array([v-theta,a-limit]);base=margins if anchors is None else anchors
        gates=np.array([base[0]>0,base[1]>=0],float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margins-base)
        event=gates[0]+gates[1]-gates[0]*gates[1] if kind=='or' else gates[0]*gates[1]
        logits=event*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
        return loss,event,base.copy()
    result=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients([[[0.]]],[0])
    loss,event,anchors=reference(bundle.weights,bundle.initial_state)
    np.testing.assert_array_equal(result['spikes'][0][0][1:],event);assert result['loss']==pytest.approx(loss,abs=3e-6)
    for i,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[i][j]+=1e-6;lo[i][j]-=1e-6
            fd=(reference(hi,bundle.initial_state,anchors)[0]-reference(lo,bundle.initial_state,anchors)[0])/2e-6
            assert result['gradients'][i][j]==pytest.approx(fd,abs=3e-6,rel=3e-4)
    for j in range(len(bundle.initial_state)):
        hi=np.asarray(bundle.initial_state).copy();lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(bundle.weights,hi,anchors)[0]-reference(bundle.weights,lo,anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][j]==pytest.approx(fd,abs=3e-6,rel=3e-4)
    monitor=b.SpikeMonitor(g);net.add(monitor);net.run(.2*b.ms,namespace={})
    np.testing.assert_array_equal(monitor.count[:],event)


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('version',[None,0,2])
@pytest.mark.parametrize('dynamic',[False,True])
def test_eager_boolean_capability_rejection(backend,version,dynamic,tmp_path,monkeypatch):
    if dynamic:
        p,w,x=model(engine=backend)
        for group in p['dynamic']['program_sets']:
            if group[0][-1]['op']=='add':group[0]=[dict(op='constant',value=1.),dict(op='eager_boolean_and',left=0,right=0)]
    else:
        p,w,x,_,_=fixture_equation(backend=backend)
        p['equations']=[compile_training_equation('f(v)',parameters={'f':PureFunction(('x',),'_b2_logical_and(x>0,x<2)')})]*2
    source=tmp_path/'old.c';library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\nuint64_t b2_train_sequence_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_boolean_eager_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    monkeypatch.setattr(importlib.import_module('brian2_rust.training_'+backend),'build',lambda directory:library)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='GPU eager Boolean capability'):trainer.gradients(x,(np.arange(len(x))%2).tolist())
    assert trainer.state==before and trainer.neuron_state is None


@pytest.mark.parametrize('ranks',[None,8])
@pytest.mark.parametrize('kind',['lazy','eager'])
def test_conditional_poisson_observation_contract(engine,ranks,kind):
    mpi(ranks);p,w=single_transform('k=draw(scale)\nv=v');p['backend']=engine;p['mpi_ranks']=ranks;w[0][0]=-1.
    # Lazy helper calls are skipped as a whole; eager where evaluates its
    # argument call, including all call-site actual arguments, before selection.
    helper=PureFunction(('v','unused'),'v')
    expression='f(v,draw(scale)) if False else v' if kind=='lazy' else '_b2_where(False,f(v,draw(scale)),v)'
    # Noise/rate are caller values: the pure body cannot capture stateful draws.
    tree=ast.parse(expression,mode='eval').body
    transform=compile_dynamic_transform('v=v',states={'v':0,'k':1,'r':2},state_types={1:'integer'})
    code=_compile_training_ast(tree,states=['v','k','r'],state_types=['float','integer','float'],
        parameters={'f':helper,'draw':PoissonNoise(0),'scale':(0,0)},allow_select=True,typed=True)
    transform['programs']=[code]
    p['dynamic']['program_sets']=[transform['programs']]
    p['dynamic']['actions'][0]=dynamic_action(transform,[0,2,4],owner=1,program_set=0,noise_domain=71,noise_entity=0,noise_streams=1)
    trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(trainer.state)
    if kind=='eager':
        with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0])
        assert trainer.state==before and trainer.clock_tick==0
    else:assert np.isfinite(trainer.gradients(np.zeros((1,1,1)),[0])['loss'])
