"""Standard scalar math against high precision values and analytic loss VJPs."""
import copy
import math
import subprocess
import sys
import mpmath as mp
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lif_training_plan, compile_training_equation, neuron_parameter_bank
from brian2_rust.training_equations import NeuronParameter
from brian2_rust.training_dynamic import compile_dynamic_transform
from test_native_training import RUNNER
from test_training_integer_ir import engine

CASES={
    'tan':[-1.1,-.3,0.,.2,1.3],
    'cosh':[-10.,-1e-8,0.,1e-8,2.,20.],
    'sinh':[-10.,-1e-8,0.,1e-8,2.,20.],
    'log10':[1e-5,.1,1.,2.,100.],
    'expm1':[-20.,-1e-12,0.,1e-12,.5,20.],
    'log1p':[-.999,-1e-12,0.,1e-12,.5,100.],
    'exprel':[-100.,-1.,-.1001,-.0999,-.01001,-.00999,-1e-12,0.,1e-12,.00999,.01001,.0999,.1001,1.,20.,80.],
    'arccos':[-.999,-.4,0.,.4,.999],
    'arcsin':[-.999,-.4,0.,.4,.999],
    'arctan':[-1e10,-2.,-.4,0.,.4,2.,1e10],
    'floor':[-2.,-.3,0.,.3,2.],
    'ceil':[-2.,-.3,0.,.3,2.],
    'abs':[-2.,-.3,-0.,.3,2.],
    'sign':[-2.,-.3,-0.,.3,2.],
}


def reference(kind,x):
    with mp.workdps(90):
        if kind=='exprel':f=lambda x:mp.expm1(x)/x if x else mp.mpf(1)
        elif kind=='abs':f=abs
        else:f=getattr(mp,{'arccos':'acos','arcsin':'asin','arctan':'atan'}.get(kind,kind))
        value=f(mp.mpf(x))
        if kind in ('ceil','sign','floor'):derivative=0.
        elif kind=='abs' and not x:derivative=0.
        elif kind=='exprel' and not x:derivative=.5
        else:derivative=mp.diff(f,mp.mpf(x))
        return float(value),float(derivative)


def plan(kind,values,scale,mode,backend):
    parameters=dict(gain=(0,0) if mode=='scalar' else NeuronParameter(0),scale=scale)
    code=compile_training_equation(f'scale*{kind}(gain*v)',parameters=parameters,
        states=None if mode=='scalar' else ['v'])
    identity=[dict(op='state',index=0)]
    p=lif_training_plan([1,1,2],projections=[neuron_parameter_bank(1 if mode=='scalar' else 2)],
        backend=backend,threshold=[100.,.25],reset='subtract',detach_reset=False,
        **(dict(equations=[compile_training_equation('v'),code]) if mode=='scalar' else dict(state_equations=[[identity],[code]],state_resets=[[identity],[identity]])))
    if mode=='dynamic':
        p['clock']=dict(origin=0.,dt=.001)
        actions=[dict(owner=j+1,reads=[j+1],writes=[j+1],program_set=0,parameter_index=j,threshold=None,trigger=None) for j in range(2)]
        actions.extend(dict(owner=j,reads=[j],writes=[],program_set=None,threshold=j,trigger=None) for j in range(3))
        p.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(initial=[.317,*values],initial_parameters=[None]*3,
            detached=[False]*3,voltage=[0,1,2],program_sets=[[code]],actions=actions))
    return p,[[1.]*(1 if mode=='scalar' else 2)]


@pytest.mark.parametrize('kind',list(CASES))
@pytest.mark.parametrize('mode',['scalar','vector','dynamic'])
def test_standard_math_high_precision_values_and_vjp(engine,kind,mode):
    for point in CASES[kind]:
        values=[point,.317] if mode=='scalar' else [point,point]
        if engine!='cpu':values=np.array(values,dtype=np.float32).astype(float).tolist()
        refs=[reference(kind,x) for x in values];scale=1/max(1.,*(abs(r[0]) for r in refs))
        if engine!='cpu':scale=float(np.float32(scale))
        p,w=plan(kind,values,scale,mode,engine)
        result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((1,1,1)),[0],initial=[[.317,*values]])
        pre=np.array([r[0]*scale for r in refs]);spikes=(pre>.25).astype(float)
        expected=pre-.25*spikes if mode=='scalar' else pre
        tol=3e-5 if engine!='cpu' else 8e-13
        np.testing.assert_allclose(result['final_membrane'][0][1:],expected,rtol=tol,atol=1e-36)
        np.testing.assert_array_equal(result['spikes'][0][0][1:],spikes)
        logits=spikes*p['logit_scale'];prob=np.exp(logits-logits.max());prob/=prob.sum();prob[0]-=1
        phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(pre-.25))**2
        gradient=prob*p['logit_scale']*phi*np.array([r[1]*scale for r in refs])
        np.testing.assert_allclose(result['initial_gradients'][0][1:],gradient,rtol=tol*3,atol=1e-36)
        expected_weight=gradient*np.array(values)
        if mode=='scalar':expected_weight=[expected_weight.sum()]
        np.testing.assert_allclose(result['gradients'][0],expected_weight,rtol=tol*4,atol=1e-36)
        if engine!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('kind,x',[('log1p',-1.1),('log10',-.1),('arccos',1.1),('arcsin',-1.1)])
def test_standard_math_domain_failure_is_atomic(engine,kind,x):
    p,w=plan(kind,[x,x],1.,'dynamic',engine);trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step(np.zeros((1,1,1)),[0])
    assert trainer.state==before and trainer.neuron_state is None


def test_lazy_math_skips_invalid_domain(engine):
    values=[-2.,.2];p,w=plan('exprel',values,1.,'dynamic',engine)
    transform=compile_dynamic_transform('v=log1p(v) if v>0 else exprel(v)',states={'v':0},state_types={0:'float'})
    p['dynamic']['program_sets']=[transform['programs']]
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((1,1,1)),[0])
    refs=[reference('exprel',-2.),reference('log1p',float(np.float32(.2)) if engine!='cpu' else .2)]
    pre=np.array([r[0] for r in refs]);spikes=(pre>.25).astype(float)
    logits=spikes*p['logit_scale'];prob=np.exp(logits-logits.max());prob/=prob.sum();prob[0]-=1
    phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(pre-.25))**2
    tol=4e-5 if engine!='cpu' else 3e-13
    np.testing.assert_allclose(result['final_state'][0][1:],pre,rtol=tol,atol=tol*.01)
    np.testing.assert_allclose(result['initial_state_gradients'][0][1:],prob*p['logit_scale']*phi*np.array([r[1] for r in refs]),rtol=tol,atol=tol*.01)


@pytest.mark.parametrize('mode',['scalar','vector','dynamic'])
@pytest.mark.parametrize('capability',[None,2])
def test_stale_math_library_rejected_before_execution(mode,capability,tmp_path,monkeypatch):
    from brian2_rust import training_metal
    c=tmp_path/'stub.c';library=tmp_path/'stub.dylib'
    c.write_text('#include <stdint.h>\n'+('int unrelated(void){return 0;}' if capability is None else f'uint64_t b2_train_math_v1(void){{return {capability};}}'))
    subprocess.run(['cc','-dynamiclib' if sys.platform=='darwin' else '-shared','-fPIC',str(c),'-o',str(library)],check=True,capture_output=True)
    monkeypatch.setattr(training_metal,'build',lambda _:library)
    p,w=plan('exprel',[.2,.2],1.,mode,'metal');trainer=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='GPU math ABI capability'):
        trainer.step(np.zeros((1,1,1)),[0],initial=[[.317,.2,.2]])
    assert trainer.state==before and trainer.neuron_state is None


@pytest.mark.parametrize('kind',['abs','sign'])
@pytest.mark.parametrize('values',[[-2147483648,2147483647],[-7,5],[0,-1]])
def test_integer_abs_sign_preserve_int32_control(engine,kind,values):
    p,w=plan(kind,[.2,.2],1.,'dynamic',engine);d=p['dynamic']
    d['initial'].extend(values);d['initial_parameters'].extend([None,None]);d['detached'].extend([True,True]);d['integer_states']=[3,4]
    code=compile_training_equation(kind+'(q)',states=['v','q'],state_types=['float','integer'])
    d['program_sets']=[[code]]
    for j in range(2):d['actions'][j].update(reads=[j+1,3+j],writes=[3+j])
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(np.zeros((1,1,1)),[0])
    expected=[(abs(x)+2**31)%2**32-2**31 if kind=='abs' else int(x>0)-int(x<0) for x in values]
    np.testing.assert_array_equal(result['final_state'][0][3:],expected)
    np.testing.assert_array_equal(result['initial_state_gradients'][0][3:],[0,0])
