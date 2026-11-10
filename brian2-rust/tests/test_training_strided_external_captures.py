"""External strided/reversed aliases share addresses across FIFO/batch calls."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_mixed_capture_lengths import model
from test_training_event_captures import event_capture,readonly_event_capture
from test_training_shared_capture_views import zero_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def data_model(mode,readonly_first,ranks,backend,zero=False):
    net,g,syn,_,_,dt,_,x=model(mode,0,ranks,backend)
    parent=np.array([.23,.31,.41,.47,.53]);a=parent[::2];c=parent[::-1].view();c.flags.writeable=False
    mutable=b.Function((zero_capture if zero else event_capture)(a),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    readonly=b.Function(readonly_event_capture(c),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    syn.namespace.update(f=readonly if readonly_first else mutable,q=mutable if readonly_first else readonly)
    syn.pre.code='h=f('+('v_post' if mode=='scalar' else 'h')+');h=q(h)'+(';v_post+=gain*h' if mode=='scalar' else ';u_post+=gain*h' if mode=='vectorised' else '')
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    mutable=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(not alias['readonly'] for alias in row['aliases']))
    readonly=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if all(alias['readonly'] for alias in row['aliases']))
    assert mutable==readonly[::-1][::2]
    assert len(set(mutable+readonly))==5
    return net,g,syn,parent,dt,bundle,x,readonly[::-1]


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_external_strided_views_original_restore(engine,mode,readonly_first,ranks,tmp_path):
    mpi(ranks);net,g,syn,parent,dt,bundle,x,cells=data_model(mode,readonly_first,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=trainer.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],parent,rtol=8e-5,atol=8e-6)
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                ids=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,ids],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(trainer.plan,runner=RUNNER);trainer.restore(path)


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_external_reversed_view_observes_strided_zero_write(engine,mode,readonly_first,ranks):
    mpi(ranks);_,_,_,_,_,bundle,_,_=data_model(mode,readonly_first,ranks,engine,zero=True)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);trainer.evaluate(np.zeros((1,1,2)),[0])
    if readonly_first:trainer.step(np.array([[[1.,0.]]]),[0])
    before=copy.deepcopy(dict(state=trainer.state,neuron=trainer.neuron_state,clock=trainer.clock_state,tick=trainer.clock_tick,elapsed=trainer.elapsed_ticks))
    with pytest.raises(ValueError,match='[Nn]on.?finite|domain'):
        trainer.step(np.ones((1,1,2)),[0],**(dict(initial='carry') if readonly_first else {}))
    assert dict(state=trainer.state,neuron=trainer.neuron_state,clock=trainer.clock_state,tick=trainer.clock_tick,elapsed=trainer.elapsed_ticks)==before
