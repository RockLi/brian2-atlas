"""Candidate ownership must include every possible runtime-index destination."""
import copy
import os
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
from test_native_training import RUNNER
from test_training_indirect import model
from test_training_delay_update import snapshot


def owned(ranks):
    p,w,x=model(ranks=ranks);spec=p['dynamic'];spec['initial'][6:8]=[.3,.6]
    transform=compile_dynamic_transform('a+=.1',states=dict(a=0,b=1,pick=2),
        state_types={0:'float',1:'float',2:'integer'})
    ps=len(spec['program_sets']);spec['program_sets'].append(transform['programs'])
    action=dynamic_action(transform,[6,7,4],owner=3,program_set=ps,mask=[0,0])
    action['indirect']=dict(reads={'0':dict(index=4,tables=[[6,7]])},
        writes={'0':dict(index=dict(kind='read',slot=2),tables=[[6,7]])})
    spec['actions'].append(action)
    spec['migration']=dict(controlled_masks=[[0,0]],cells=[dict(index=k,owners=[[0,0]],restart=dict(kind='initial')) for k in (6,7)])
    return p,w,x


@pytest.mark.parametrize('ranks',[None,2,8])
def test_mask_migration_resets_all_runtime_index_candidates(ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=owned(ranks);trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    trainer.step(x[None,:1],[0])
    np.testing.assert_allclose(trainer.neuron_state[0][6:8],[.3,.7],atol=1e-15)
    cursor=(trainer.clock_tick,trainer.noise_sequence,trainer.state['step'])
    masks=copy.deepcopy(p['masks']);masks[0][0]=0;trainer.update_mask(masks)
    assert trainer.neuron_state[0][6:8]==[0.,0.]
    assert cursor==(trainer.clock_tick,trainer.noise_sequence,trainer.state['step'])
    masks[0][0]=1;trainer.update_mask(masks,growth_weight=.12)
    np.testing.assert_array_equal(trainer.neuron_state[0][6:8],[.3,.6])
    assert cursor==(trainer.clock_tick,trainer.noise_sequence,trainer.state['step'])


@pytest.mark.parametrize('ranks',[None,2,8])
def test_ownership_cannot_omit_a_runtime_candidate(ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=owned(ranks);p['dynamic']['migration']['cells'].pop()
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);before=snapshot(trainer)
    with pytest.raises(ValueError,match='ownership must cover exactly'):
        trainer.step(x[None],[0])
    assert snapshot(trainer)==before
