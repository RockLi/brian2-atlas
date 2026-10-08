"""RK stages with detached refractory gates: independent forward/VJP oracles."""
import copy

import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
import test_training_refractory as ref


@pytest.mark.parametrize('method',['rk2','rk4'])
@pytest.mark.parametrize('clamp',[(),('v',),('a',),('v','a')])
@pytest.mark.parametrize('duration',[0.,1.,3.,3.5])
@pytest.mark.parametrize('warmup,units',[(0,False),(3,True)])
def test_rk_refractory_forward(method,clamp,duration,warmup,units):
    ref.test_refractory_forward_matches_brian(clamp,duration,warmup,units,method)


@pytest.mark.parametrize('method',['rk2','rk4'])
@pytest.mark.parametrize('clamp',[(),('v',),('a',),('v','a')])
@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
def test_rk_refractory_independent_vjp(method,clamp,detach,window):
    ref.test_refractory_independent_vjp(clamp,detach,window,method)


@pytest.mark.parametrize('methods',[('euler','rk4'),('rk2','rk4')])
@pytest.mark.parametrize('clamp',[('v',),('a',)])
def test_mixed_integrators_with_refractory(methods,clamp):
    ref.test_refractory_forward_matches_brian(clamp,3.5,3,True,methods)
    ref.test_refractory_independent_vjp(clamp,False,3,methods)


@pytest.mark.parametrize('method',['rk2','rk4'])
@pytest.mark.parametrize('clamp',[('v',),('a',)])
@pytest.mark.parametrize('backend,ranks',[('cpu',2),('cpu',8),('metal',None),('metal',2),('metal',8),
                                         ('cuda',None),('cuda',2),('cuda',8)])
def test_rk_refractory_backends_restore(method,clamp,backend,ranks,tmp_path):
    ref.test_refractory_backends_carry_and_fresh_restore(backend,ranks,clamp,tmp_path,method)


@pytest.mark.parametrize('change',['missing_spec','wrong_counter','reset_gate','raw_counter'])
def test_refractory_activity_rejected_outside_update_contract(change):
    net,source,groups,x,dt=ref.model(method='rk4');bundle=ref.lower(net,source,groups)
    p=copy.deepcopy(bundle.plan)
    if change=='missing_spec':p.pop('refractory')
    elif change=='wrong_counter':p['state_equations'][0][1]=[dict(op='refractory_active',index=0)]
    elif change=='reset_gate':p['state_resets'][0][0]=[dict(op='refractory_active',index=2)]
    else:p['state_equations'][0][1]=[dict(op='state',index=2)]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='refractory'):
        trainer.step(x[None],[0],initial=[bundle.initial_state])
    assert trainer.state==before and trainer.neuron_state is None


def test_rk_stage_gate_affects_unclamped_state():
    # Partial clamp is essential: replacing only the final held state permits
    # its invalid intermediate RK values to leak into the other state's update.
    net,source,groups,x,dt=ref.model(('v',),method='rk4');bundle=ref.lower(net,source,groups)
    initial=np.array(bundle.initial_state);initial[[4,5,10,11]]=2
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    result=trainer.gradients(x[None,:1],[0],initial=initial[None])
    _,_,expected,_=ref.oracle(bundle,bundle.weights,x[:1],initial)
    np.testing.assert_allclose(result['final_state'][0],expected,rtol=2e-14,atol=2e-14)
    broken=copy.deepcopy(bundle.plan)
    for layer in broken['state_equations']:
        for program in layer:
            for node in program:
                if node['op']=='refractory_active':node.clear();node.update(op='constant',value=1.)
    ungated=NativeLIFTrainer(broken,runner=RUNNER,weights=bundle.weights).evaluate(x[None,:1],[0],initial=initial[None])
    assert np.max(abs(np.array(ungated['final_state'])[0,[2,3,8,9]]-expected[[2,3,8,9]]))>1e-5
    assert np.all(np.array(result['initial_state_gradients'])[0,[4,5,10,11]]==0)
