"""Partial physical voltage/h views use all-bank and all-initial surrogate VJPs."""
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_partial_capture_views import data_model
from test_training_mixed_capture_vjps import reference
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('kind',['v','h'])
@pytest.mark.parametrize('offset',[0,1])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_partial_capture_all_bank_initial_vjps(engine,mode,kind,offset,delay,window,ranks):
    mpi(ranks);data=data_model(mode,kind,offset,delay,ranks,engine);net,g,syn,_,_,dt,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,mode,False,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    cells+=list({cell for row in bundle.provenance['mutable_capture_layout'] for cell in row['cells']})
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,False,hi,window,anchors=anchors)[0]-reference(data,mode,False,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,False,bundle.weights,window,hi,anchors)[0]-reference(data,mode,False,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    net.run(4*dt,namespace={})
    for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
        for name in names:
            cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
