"""Contiguous partial captures share actual Brian runtime cells."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_mixed_capture_lengths import model,independent_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def data_model(mode,kind,offset,delay,ranks,backend):
    data=list(model(mode,delay,ranks,backend))
    net,g,syn,_,c,_,_,_=data
    array=(syn if kind=='h' else g).variables['h' if kind=='h' else 'v'].get_value()
    view=array[offset:offset+1];data[3]=view
    syn.namespace['f']=b.Function(independent_capture(view,c),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    data[6]=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    expected=data[6].provenance['dynamic_state_layout' if kind=='h' else 'neuron_state_layout'][syn.name if kind=='h' else g.name]['h' if kind=='h' else 'v'][offset:offset+1]
    row=next(row for row in data[6].provenance['mutable_capture_layout'] if any(alias['capture']=='a' for alias in row['aliases']))
    assert row['cells']==expected
    return data


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('kind',['v','h'])
@pytest.mark.parametrize('offset',[0,1])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_partial_runtime_view_original_carry_restore(engine,mode,kind,offset,delay,ranks,tmp_path):
    mpi(ranks);net,g,syn,view,c,dt,bundle,x=data_model(mode,kind,offset,delay,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=trainer.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        row=next(row for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']=='b' for alias in row['aliases']))
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,row['cells']],c,rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(trainer.plan,runner=RUNNER);trainer.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
