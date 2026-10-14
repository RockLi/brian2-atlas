"""Captured bank subviews read the current optimizer row, including after restore."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_mixed_capture_lengths import model
from test_training_event_captures import readonly_event_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('ranks',[None,2])
def test_partial_bank_current_offset_after_carry_restore(engine,mode,ranks,tmp_path):
    mpi(ranks);net,g,syn,_,_,_,_,x=model(mode,0,ranks,engine)
    view=syn.variables['gain'].get_value()[1:].view();view.flags.writeable=False
    syn.namespace['f']=b.Function(readonly_event_capture(view),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    alias=next(row for row in bundle.provenance['readonly_capture_layout'] if row.get('capture')=='array')
    assert alias['indices']==[1]
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and 'gain' in row['variables'])
    assert alias['bank']==bank and not bundle.provenance['mutable_capture_layout']
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);trainer.step(x[:,:3],[0])
    trainer.state['weights'][bank][0]=0.;trainer.evaluate(np.ones((1,1,2)),[0],initial='carry')
    path=tmp_path/'partial-bank';trainer.store(path);trainer=NativeLIFTrainer(trainer.plan,runner=RUNNER);trainer.restore(path)
    trainer.state['weights'][bank][1]=0.;before=copy.deepcopy(dict(state=trainer.state,neuron=trainer.neuron_state,clock=trainer.clock_state,tick=trainer.clock_tick,elapsed=trainer.elapsed_ticks))
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):
        trainer.step(np.ones((1,1,2)),[0],initial='carry')
    assert dict(state=trainer.state,neuron=trainer.neuron_state,clock=trainer.clock_state,tick=trainer.clock_tick,elapsed=trainer.elapsed_ticks)==before
