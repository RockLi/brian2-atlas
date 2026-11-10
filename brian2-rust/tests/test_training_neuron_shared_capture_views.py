"""Neuron integration/threshold/reset retain readonly and writable view identity."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_event_captures import event_capture,readonly_event_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(where,readonly_first,method,ranks,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    array=np.array([.23,.31]);view=array.view();view.flags.writeable=False
    readonly=b.Function(readonly_event_capture(view),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    mutable=b.Function(event_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    namespace=dict(f=readonly if readonly_first else mutable,q=mutable if readonly_first else readonly)
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt='+('q(f(v))/ms' if where=='integrator' else '0/second')+':1',
        threshold='v>'+('q(f(v))' if where=='threshold' else '.5'),reset='v=q(f(v))' if where=='reset' else 'v-=.5',method=method,dt=dt,namespace=namespace);g.v=[.173,.719]
    net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False)
    return net,g,array,dt,bundle


@pytest.mark.parametrize('where,method',[('integrator','euler'),('integrator','rk2'),('integrator','rk4'),('threshold','euler'),('reset','euler')])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_neuron_shared_views_original_restore(engine,where,method,readonly_first,ranks,tmp_path):
    mpi(ranks);net,g,array,dt,bundle=model(where,readonly_first,method,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    assert len(bundle.provenance['mutable_capture_layout'])==1
    for tick in range(4):
        out=t.step(np.zeros((1,1,1)),[0],initial='carry' if tick else None);net.run(dt,namespace={})
        cells=bundle.provenance['neuron_state_layout'][g.name]['v'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.v[:],rtol=8e-5,atol=8e-6)
        cells=bundle.provenance['mutable_capture_layout'][0]['cells'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],array,rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
