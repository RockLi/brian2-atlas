"""Surrogate VJPs when a whole reset capture aliases selected voltage storage."""
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_reset_captures import model


def reference(data,weights,initial=None,anchors=None):
    _,g,_,_,bundle=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for i,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[i]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v']
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and 'gain' in row['variables'])
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());old=z[v]+.2*np.array(weights[bank]);margin=old-.5
        event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            base=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margin-base)
        spikes.append(event.copy())
        # The closure mutates every row to .8*old. NumPy scatters selected
        # private locals (.7*old) afterwards; the surrogate jump is -.1*old.
        z[v]=.8*old+event*(-.1*old)
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('selection',['empty','partial','full'])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_reset_voltage_alias_all_bank_initial_vjps(engine,selection,ranks,window):
    mpi(ranks);data=model(selection,False,ranks,window,engine,capture_voltage=True);bundle=data[4]
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
    loss,z,anchors=reference(data,bundle.weights);cells=bundle.provenance['neuron_state_layout'][data[1].name]['v']
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(reference(data,hi,anchors=anchors)[0]-reference(data,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,j)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j]:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(data,bundle.weights,hi,anchors)[0]-reference(data,bundle.weights,lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j
