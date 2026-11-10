"""Do not split a writable buffer and its readonly view into disconnected banks."""
import brian2 as b
import numpy as np
import pytest
from brian2_rust import lower_brian_dynamic_training
from test_training_batch_event_captures import batch_model
from test_training_event_captures import event_capture,readonly_event_capture


@pytest.mark.parametrize('readonly_first',[False,True])
def test_shared_mutability_views_use_canonical_binding(readonly_first):
    net,g,syn,_,_,_,_=batch_model(False,False,0,None,'cpu');array=np.ones(2);view=array.view();view.flags.writeable=False
    readonly=b.Function(readonly_event_capture(view),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    mutable=b.Function(event_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    syn.namespace.update(f=readonly if readonly_first else mutable,q=mutable if readonly_first else readonly);syn.pre.code='h=f(h);h=q(h)'
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup));hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],trainable_synapse_parameters={syn.name:['h']})
    assert len(bundle.provenance['mutable_capture_layout'])==1
    assert {row['readonly'] for row in bundle.provenance['mutable_capture_layout'][0]['aliases']}=={False,True}
    assert not bundle.provenance['readonly_capture_layout']
    np.testing.assert_array_equal(array,[1.,1.])
