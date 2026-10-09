"""Batch queue migration must agree with independent single-sample runs."""
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_delays import model,cython_cache
from test_training_delay_update import change


@pytest.mark.parametrize('event_driven',[False,True])
def test_batch_migration_matches_separate_histories_and_gradients(event_driven):
    *_,static,syn,x,bundle=model(event_driven,post_first=True,order_sensitive=True)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    batch=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    singles=[NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights) for _ in range(2)]
    inputs=np.stack([x,x.copy()]);inputs[1,:2]=0
    physical=sorted({i for group in bundle.provenance['neuron_state_layout'].values() for cells in group.values() for i in cells}|
                    {i for synapse in bundle.provenance['dynamic_state_layout'].values() for cells in synapse.values() for i in cells})
    for phase,(start,stop) in enumerate([(0,2),(2,3),(3,len(x))]):
        if phase:
            values=change(phase-1,static,syn)
            for trainer in [batch,*singles]:trainer.update_delays(values)
            np.testing.assert_array_equal(np.asarray(batch.neuron_state)[:,physical],
                [np.asarray(t.neuron_state)[0,physical] for t in singles])
        kwargs=dict(initial='carry') if phase else {}
        result=batch.step(inputs[:,start:stop],[0,1],**kwargs)
        references=[t.step(inputs[i:i+1,start:stop],[i],**kwargs) for i,t in enumerate(singles)]
        np.testing.assert_array_equal(result['spikes'],[r['spikes'][0] for r in references])
        np.testing.assert_allclose(np.asarray(result['final_state'])[:,physical],
            [np.asarray(r['final_state'])[0,physical] for r in references],rtol=2e-13,atol=2e-13)
        for bank,gradient in enumerate(result['gradients']):
            np.testing.assert_allclose(gradient,np.mean([r['gradients'][bank] for r in references],axis=0),rtol=3e-12,atol=2e-13)
        np.testing.assert_allclose(np.asarray(result['initial_state_gradients'])[:,physical],
            np.asarray([np.asarray(r['initial_state_gradients'])[0,physical] for r in references])/2,rtol=3e-12,atol=2e-13)


def test_empty_pathways_accept_boundary_updates_and_reject_invalid_values():
    from test_training_summed import model as summed,lower
    net,inp,layers,static,syn,x,_=summed(empty=True)
    bundle=lower(net,inp,layers);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(x[None,:2],[0])
    empty=[p['name'] for p in trainer.plan['dynamic']['delay_layout']['paths'] if not p['edges']]
    assert len(empty)==2
    before=copy.deepcopy((trainer.state,trainer.neuron_state))
    trainer.update_delays({name:.001 for name in empty})
    assert (trainer.state,trainer.neuron_state)==before
    with pytest.raises(ValueError):trainer.update_delays({empty[0]:-.001})
    assert (trainer.state,trainer.neuron_state)==before
    trainer.evaluate(x[None,2:],[0],initial='carry')
