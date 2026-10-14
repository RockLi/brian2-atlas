"""Reverse and repeated readonly bank indices remain distinct and live."""
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
def test_reverse_and_repeated_bank_views_current_after_restore(engine,mode,ranks,tmp_path):
    mpi(ranks);net,g,syn,_,_,_,_,x=model(mode,0,ranks,engine)
    parent=syn.variables['gain'].get_value();reverse=parent[::-1].view();reverse.flags.writeable=False
    repeated=np.broadcast_to(parent[1:],(2,))
    syn.namespace.update(f=b.Function(readonly_event_capture(reverse),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False),
                         q=b.Function(readonly_event_capture(repeated),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False))
    syn.pre.code='h=f('+('v_post' if mode=='scalar' else 'h')+');h=q(h)'+(';v_post+=gain*h' if mode=='scalar' else ';u_post+=gain*h' if mode=='vectorised' else '')
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    aliases=[row for row in bundle.provenance['readonly_capture_layout'] if row.get('capture')=='array']
    assert {tuple(row['indices']) for row in aliases}=={(1,0),(1,1)}
    assert not bundle.provenance['mutable_capture_layout']
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and 'gain' in row['variables'])
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);trainer.step(x[:,:3],[0])
    path=tmp_path/'strided-bank';trainer.store(path);trainer=NativeLIFTrainer(trainer.plan,runner=RUNNER);trainer.restore(path)
    trainer.state['weights'][bank][0]=0.
    before=copy.deepcopy(dict(state=trainer.state,neuron=trainer.neuron_state,clock=trainer.clock_state,tick=trainer.clock_tick,elapsed=trainer.elapsed_ticks))
    # The repeated view only reads index 1. The reversed view must still read
    # index 0, so a field-name collision cannot silently replace it.
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):
        trainer.step(np.ones((1,1,2)),[0],initial='carry')
    assert dict(state=trainer.state,neuron=trainer.neuron_state,clock=trainer.clock_state,tick=trainer.clock_tick,elapsed=trainer.elapsed_ticks)==before


@pytest.mark.parametrize('selection',[slice(None,None,2),slice(None,None,-1),slice(3,None,-2)])
def test_capture_view_indices_match_numpy(selection):
    from brian2_rust.training_effects import capture_view_indices
    parent=np.arange(5,dtype=float);view=parent[selection]
    assert capture_view_indices(parent,view)==list(np.arange(5)[selection])
    assert capture_view_indices(parent,np.array(view)) is None
