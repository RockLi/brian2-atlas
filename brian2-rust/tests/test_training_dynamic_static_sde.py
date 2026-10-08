"""The same nonlinear SDE callback through the ordered dynamic action plan."""
import copy
from types import SimpleNamespace
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_static_control_sde import model,lower,oracle,assert_original_brian_forward


def physical_reference(bundle,network,source,groups):
    # v5 projection records are parameter-bank placeholders. Obtain the
    # independent physical edge order from Brian, using public bank bindings.
    projections=copy.deepcopy(bundle.plan['projections'])
    layer={source.name:0,**{group.name:i+1 for i,group in enumerate(groups)}}
    for synapse in sorted((o for o in network.objects if isinstance(o,b.Synapses)),key=lambda o:o.name):
        bank=next(item['bank'] for item in bundle.provenance['bindings']
                  if item['object']==synapse.name and item['variables']==['w'])
        projections[bank]=dict(source_layer=layer[synapse.source.name],target_layer=layer[synapse.target.name],
            parameter_count=len(synapse),sources=np.asarray(synapse.i[:]).tolist(),
            targets=np.asarray(synapse.j[:]).tolist(),parameter_ids=list(range(len(synapse))))
    return SimpleNamespace(plan={**bundle.plan,'projections':projections},provenance=bundle.provenance)


@pytest.mark.parametrize('method,shared',[('heun',False),('heun',True),('heun','mixed'),('milstein',False)])
@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('detach,window',[(False,None),(True,None),(False,3)])
def test_dynamic_sde_callback_physics_all_banks_and_initial_vjps(engine,ranks,method,shared,detach,window):
    mpi(ranks);network,source,groups,x,dt=model(method,shared)
    bundle=lower(network,source,groups,dynamic=True,backend=engine,mpi_ranks=ranks,detach_reset=detach,tbptt_window=window)
    reference=physical_reference(bundle,network,source,groups)
    initial=np.asarray(bundle.initial_state);weights=bundle.weights;sequence=9
    def expected(w,z,anchors=None):return oracle(reference,w,x,z[:8],anchors,sequence=sequence)
    result=NativeLIFTrainer(bundle.plan,weights=weights,runner=RUNNER).gradients(x[None],[0],initial=initial[None],noise_sequence=sequence)
    loss,spikes,live,anchors=expected(weights,initial)
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,:8],live,rtol=3e-5,atol=3e-5)
    for bank,row in enumerate(weights):
        for index,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);plus=copy.deepcopy(weights);minus=copy.deepcopy(weights)
            plus[bank][index]+=eps;minus[bank][index]-=eps
            fd=(expected(plus,initial,anchors)[0]-expected(minus,initial,anchors)[0])/(2*eps)
            assert result['gradients'][bank][index]==pytest.approx(fd,rel=4e-4,abs=5e-5)
    for index in range(len(initial)):
        plus=initial.copy();minus=initial.copy();plus[index]+=1e-6;minus[index]-=1e-6
        fd=(expected(weights,plus,anchors)[0]-expected(weights,minus,anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][index]==pytest.approx(fd,rel=4e-4,abs=5e-5)
    assert (result['gpu_dispatches']>0)==(engine!='cpu')
    if ranks is None and not detach and window is None:
        # The original Brian reference only exposes neuronal physical values.
        visible={**result,'final_state':[np.asarray(result['final_state'])[0,:8].tolist()]}
        assert_original_brian_forward(network,groups,bundle,x,dt,visible,sequence)
