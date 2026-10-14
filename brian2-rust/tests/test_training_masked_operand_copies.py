"""Every Boolean-indexed RHS operand is a separate NumPy advanced-index copy."""
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER
import test_training_masked_multistatement_captures as base

CODES={'reread':'v_post+=gain*(f(v_post)+v_post)',
       'twice':'v_post+=gain*(f(v_post)+f(v_post))'}


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_rhs_operand_independent_copies(engine,kind,ranks,tmp_path):
    mpi(ranks);base.CODES['operand_copy']=CODES[kind]
    try:net,g,syn,dt,bundle,x=base.model(engine,ranks,'array','operand_copy',True,self_argument=True)
    finally:del base.CODES['operand_copy']
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(2):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        cells=bundle.provenance['neuron_state_layout'][g.name]['v']
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.v[:],rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)


def reference(data,kind,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];counter=bundle.provenance['neuron_state_layout'][g.name]['__refractory_ticks']
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    path=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1];rows=bundle.provenance['delay_queues'][path]['new']
    before=[];margins=[];hard_rows=[];raw_rows=[];spikes=[]
    for tick in range(3):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());margin=z[v]-.5;hard=margin>0;event=hard.astype(float)
        if anchors is not None:
            hard=anchors['hard'][tick];old=anchors['margins'][tick]
            event=hard.astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        assert not hard.any();margins.append(margin.copy());hard_rows.append(hard.copy());spikes.append(event.copy())
        for copies in bundle.provenance['event_callback_snapshots'].values():
            for row in copies:z[row['cache']]=z[row['source']]
        old=z.copy();raw=[row['edge'] for row in rows if old[row['states'][0]]>.5]
        if anchors is not None:raw=anchors['raw'][tick]
        raw_rows.append(raw)
        if raw:z[v]*=.9 if kind=='reread' else .81
        # Each indexed read in the RHS starts from the caller's old array.
        # The detached empty-batch ordinal is zero. Each row's selected
        # input falls back to its own old value, while the fixed capture
        # vector is projected at column zero for a counterfactual row gate.
        capture_values=old[v] if raw else np.full(len(v),old[v[0]])
        rhs=.9*capture_values*old[v]+old[v] if kind=='reread' else 1.71*capture_values*old[v]
        for row in rows:
            edge=row['edge'];amplitude=old[row['states'][0]]
            new=old[v[edge]]+weights[gain][edge]*rhs[edge]
            z[v[edge]]+=amplitude*(new-z[v[edge]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=1.
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,raw=raw_rows)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_masked_rhs_operand_all_vjps(engine,kind,ranks,window):
    mpi(ranks);base.CODES['operand_copy']=CODES[kind]
    try:data=base.model(engine,ranks,'array','operand_copy',True,self_argument=True)
    finally:del base.CODES['operand_copy']
    _,g,syn,_,bundle,_=data;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window;x=np.ones((1,3,2))
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,kind,bundle.weights,window)
    cells=bundle.provenance['neuron_state_layout'][g.name]['v']
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,kind,hi,window,anchors=anchors)[0]-reference(data,kind,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,kind,bundle.weights,window,hi,anchors)[0]-reference(data,kind,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
