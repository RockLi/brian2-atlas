"""Per-sample external tables: independent physical models and full bank VJPs."""
import copy
import os
from types import SimpleNamespace
from unittest.mock import patch

import brian2 as b
import numpy as np
import pytest

from brian2_rust import (BatchTimedArray, NativeLIFTrainer, external_state_input_vjp,
                        lower_brian_dynamic_training)
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_external_selector_roots import (model as root_model, oracle as root_oracle,
    physical_cells, DRIVE, ROUTES, PATTERN)
from test_training_discrete_external_inputs import (model as discrete_model, oracle as discrete_oracle,
    CODES, FLAGS)

DRIVES=np.stack([DRIVE,DRIVE[:,::-1]*.65+.25])
ROUTE_TABLES=np.stack([ROUTES,1-ROUTES])
INPUTS=np.repeat(PATTERN[None],2,axis=0);LABELS=[0,1]


def model(*,engine='cpu',ranks=None,window=None,noisy=False,event_driven=False,mixed=False):
    net,inp,hidden,out,syn,drive,route=root_model(noisy=noisy,event_driven=event_driven,convert=False)
    fields={'drive':drive if mixed else BatchTimedArray(DRIVES,dt=.2*b.ms,name='batch_drive'),
            'route':BatchTimedArray(ROUTE_TABLES,dt=.2*b.ms,name='batch_route')}
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,out],external_state_inputs=fields,
        trainable_neuron_parameters={out.name:['gain']},trainable_synapse_parameters={syn.name:['w','sgain']},
        backend=engine,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,seed=3527,learning_rate=1e-9)
    return bundle


def sample_bundle(bundle,weights,sample):
    p=copy.deepcopy(bundle.provenance);w=copy.deepcopy(weights)
    source=p['external_state_inputs']['sources']['drive']
    if 'samples' in source:
        table=np.array(w[source['bank']]).reshape(6,2,2).transpose(1,0,2)[sample]
        w[source['bank']]=table.reshape(-1).tolist();source['shape']=[6,2]
    initial=list(bundle.initial_state);initial[bundle.plan['dynamic']['sample_index']['slot']]=sample
    return SimpleNamespace(plan=bundle.plan,weights=w,initial_state=initial,provenance=p)


def oracle(bundle,*,weights=None,initial=None,anchors=None,noisy=False,event_driven=False,
           routes=ROUTE_TABLES,change=None):
    weights=bundle.weights if weights is None else weights;outputs=[];losses=[]
    for sample in range(2):
        child=sample_bundle(bundle,weights,sample)
        replacement=None if change is None else (change[0],change[1][sample])
        result=root_oracle(child,weights=child.weights,initial=None if initial is None else initial[sample],
            anchors=None if anchors is None else anchors[sample],noisy=noisy,event_driven=event_driven,sample=sample,
            routes=routes[sample],change=replacement)
        logits=5*result[2].mean(0);m=logits.max()
        losses.append(m+np.log(np.exp(logits-m).sum())-logits[LABELS[sample]]);outputs.append(result)
    return np.mean(losses),outputs


@pytest.mark.parametrize('noisy,event_driven',[(False,False),(True,False),(False,True)])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('mixed',[False,True])
def test_batch_tables_all_physics_and_vjps(engine,noisy,event_driven,window,mixed):
    bundle=model(engine=engine,noisy=noisy,event_driven=event_driven,window=window,mixed=mixed)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    result=trainer.gradients(INPUTS,LABELS,**(dict(noise_sequence=7) if noisy else {}))
    expected=oracle(bundle,noisy=noisy,event_driven=event_driven);cells=physical_cells(bundle);tol=5e-5 if engine!='cpu' else 4e-12
    for sample in range(2):
        np.testing.assert_allclose(np.array(result['final_state'][sample])[cells],expected[1][sample][1][cells],rtol=tol,atol=tol*.01)
        np.testing.assert_array_equal(np.array(result['spikes'])[sample,:,2:],expected[1][sample][2])
    assert result['loss']==pytest.approx(expected[0],abs=tol)
    anchors=[value[3] for value in expected[1]]
    sources=bundle.provenance['external_state_inputs']['sources'];integer_bank=sources['route']['bank']
    integers={tuple(x) for x in bundle.plan['dynamic']['integer_parameters']}
    for bank,values in enumerate(bundle.weights):
        for j in range(len(values)):
            if bank==integer_bank or (bank,j) in integers:assert result['gradients'][bank][j]==0;continue
            lo=copy.deepcopy(bundle.weights);hi=copy.deepcopy(lo);lo[bank][j]-=1e-6;hi[bank][j]+=1e-6
            fd=(oracle(bundle,weights=hi,anchors=anchors,noisy=noisy,event_driven=event_driven)[0]-
                oracle(bundle,weights=lo,anchors=anchors,noisy=noisy,event_driven=event_driven)[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=4e-4,abs=3e-6),(bank,j)
    initial=np.array([sample_bundle(bundle,bundle.weights,s).initial_state for s in range(2)])
    for sample in range(2):
        for cell in cells:
            if bundle.plan['dynamic']['detached'][cell]:assert result['initial_state_gradients'][sample][cell]==0;continue
            lo=initial.copy();hi=initial.copy();lo[sample,cell]-=1e-6;hi[sample,cell]+=1e-6
            fd=(oracle(bundle,initial=hi,anchors=anchors,noisy=noisy,event_driven=event_driven)[0]-
                oracle(bundle,initial=lo,anchors=anchors,noisy=noisy,event_driven=event_driven)[0])/2e-6
            assert result['initial_state_gradients'][sample][cell]==pytest.approx(fd,rel=4e-4,abs=3e-6)
    sample_slot=bundle.plan['dynamic']['sample_index']['slot']
    assert [row[sample_slot] for row in result['final_state']]==[0,1]
    assert all(row[sample_slot]==0 for row in result['initial_state_gradients'])
    for source in sources.values():
        vjp=external_state_input_vjp(source,result['gradients'][source['bank']]);assert list(vjp.shape)==source['shape']
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('noisy,event_driven',[(False,False),(True,False),(False,True)])
def test_batch_tables_independent_original_cython(noisy,event_driven):
    bundle=model(noisy=noisy,event_driven=event_driven);expected=oracle(bundle,noisy=noisy,event_driven=event_driven)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(INPUTS,LABELS,**(dict(noise_sequence=7) if noisy else {}))
    for sample in range(2):
        net,inp,hidden,out,syn,drive,route=root_model(noisy=noisy,event_driven=event_driven,convert=False)
        inp.namespace.update(drive_trace=b.TimedArray(DRIVES[sample],dt=.2*b.ms),route_trace=b.TimedArray(ROUTE_TABLES[sample],dt=.2*b.ms))
        draws=expected[1][sample][3]['draws'];net.run(0*b.ms,namespace={});device=b.get_device();device.randn_buffer_index[:]=0;calls=[]
        def refill(n):
            assert n==20000 and not calls;calls.append(n);values=np.zeros(n);values[:len(draws)]=draws;return values
        try:
            with patch('numpy.random.randn',refill):net.run(1.2*b.ms,namespace={})
            if noisy:assert device.randn_buffer_index[0]==len(draws)
        finally:device.randn_buffer_index[:]=0
        np.testing.assert_allclose(out.v[:],actual['final_membrane'][sample][2:],atol=2e-13)
        for obj,name in ((hidden,'z'),(syn,'h')):
            cells=bundle.provenance['neuron_state_layout' if obj is hidden else 'dynamic_state_layout'][obj.name][name]
            np.testing.assert_allclose(getattr(obj,name)[:],np.array(actual['final_state'][sample])[cells],atol=2e-13)
        assert syn.pre.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('noisy',[False,True])
def test_batch_tables_update_carry_restore(engine,ranks,noisy,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    bundle=model(engine=engine,ranks=ranks,noisy=noisy);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(INPUTS[:,:3],LABELS,**(dict(noise_sequence=7) if noisy else {}))
    source=bundle.provenance['external_state_inputs']['sources'];drives=DRIVES*.7+.2;routes=1-ROUTE_TABLES
    trainer.update_external_state_input(source['drive'],drives);trainer.update_external_state_input(source['route'],routes)
    trainer.store(tmp_path/'sample-tables.json');restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    restored.restore(tmp_path/'sample-tables.json');actual=restored.step(INPUTS[:,3:],LABELS,initial='carry')
    route=ROUTE_TABLES.copy();route[:,3:]=routes[:,3:]
    expected=oracle(bundle,noisy=noisy,routes=route,change=(3,drives));cells=physical_cells(bundle)
    for sample in range(2):np.testing.assert_allclose(np.array(actual['final_state'][sample])[cells],expected[1][sample][1][cells],rtol=5e-5,atol=3e-6)


@pytest.mark.parametrize('shared',[False,True])
def test_batch_tables_boolean_and_exact_int32(engine,shared):
    net,inp,hidden,out,syn,_,_=discrete_model(convert=False,shared=shared)
    codes=np.stack([CODES,CODES[:,::-1]]);flags=np.stack([FLAGS,~FLAGS])
    if shared:codes=codes[:,:,0];flags=flags[:,:,0]
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,out],
        external_state_inputs={'code':BatchTimedArray(codes,dt=.2*b.ms),'flag':BatchTimedArray(flags,dt=.2*b.ms)},
        trainable_neuron_parameters={out.name:['gain']},trainable_synapse_parameters={syn.name:['w']},backend=engine,detach_reset=False)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(INPUTS,LABELS)
    for sample in range(2):
        child=SimpleNamespace(plan=bundle.plan,weights=bundle.weights,initial_state=bundle.initial_state,provenance=bundle.provenance)
        c=codes[sample,:,None] if shared else codes[sample];f=flags[sample,:,None] if shared else flags[sample]
        expected=discrete_oracle(child,codes=c,flags=f,shared=shared)
        layout=bundle.provenance['neuron_state_layout'][out.name];edge=bundle.provenance['dynamic_state_layout'][syn.name]
        cells=layout['seen']+layout['pick']+edge['edge_seen']
        np.testing.assert_array_equal(np.array(result['final_state'][sample])[cells],expected[1][cells])
        np.testing.assert_allclose(result['final_membrane'][sample][1:],expected[1][layout['v']],rtol=5e-5,atol=3e-6)
    for source in bundle.provenance['external_state_inputs']['sources'].values():
        assert not np.any(external_state_input_vjp(source,result['gradients'][source['bank']]))


@pytest.mark.parametrize('issue',['batch','identity','direct_write','indirect_write','migration','table_shape','nonfinite'])
def test_batch_tables_invalid_operations_are_atomic(issue):
    bundle=model();bundle.plan['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(INPUTS[:,:2],LABELS);before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.clock_state))
    source=bundle.provenance['external_state_inputs']['sources']['drive'];slot=bundle.plan['dynamic']['sample_index']['slot']
    if issue=='batch':
        with pytest.raises(ValueError,match='batch|sample'):trainer.step(INPUTS[:1,2:],[0],initial='carry')
    elif issue=='identity':
        changed=np.array(trainer.neuron_state);changed[1,slot]=0
        with pytest.raises(ValueError,match='sample'):trainer.step(INPUTS[:,2:],LABELS,initial=changed)
    elif issue in ('direct_write','indirect_write','migration'):
        plan=copy.deepcopy(bundle.plan)
        if issue=='migration':plan['dynamic']['migration']['cells'].append(dict(index=slot,owners=[[0,0]],restart=dict(kind='initial')))
        else:
            action=next(a for a in plan['dynamic']['actions'] if a.get('writes'))
            if issue=='direct_write':action['writes'][0]=slot
            else:action['indirect']=dict(writes={'0':dict(index=dict(kind='read',slot=0),tables=[[slot]])})
        with pytest.raises(ValueError,match='sample'):NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights).evaluate(INPUTS,LABELS)
    else:
        invalid=DRIVES[:1] if issue=='table_shape' else DRIVES*np.nan
        with pytest.raises(ValueError):trainer.update_external_state_input(source,invalid)
    assert (trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.clock_state)==before


def test_batch_tables_converter_rejects_sample_disagreement_and_units():
    net,inp,hidden,out,syn,_,_=root_model(convert=False)
    for drive,route in ((BatchTimedArray(DRIVES,dt=.2*b.ms),BatchTimedArray(ROUTE_TABLES[:1],dt=.2*b.ms)),
                        (BatchTimedArray(DRIVES*b.mV,dt=.2*b.ms),BatchTimedArray(ROUTE_TABLES,dt=.2*b.ms))):
        with pytest.raises(ValueError,match='sample|matching'):
            lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,out],external_state_inputs={'drive':drive,'route':route})


@pytest.mark.parametrize('ranks',[None,2])
def test_batch_tables_poisson_score_and_zero_replay_keep_sample_id(engine,ranks):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    b.set_device('runtime');b.start_scope();dt=.2*b.ms
    inp=b.NeuronGroup(1,'rate:1 (shared)',threshold='False',reset='',dt=dt,name='batch_poisson_input')
    hidden=b.NeuronGroup(1,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',dt=dt,method='euler',name='batch_poisson_hidden')
    out=b.NeuronGroup(2,'dv/dt=-v/ms:1\nrate:1 (linked)\nk:integer',threshold='v>.5',reset='v-=.5',
        dt=dt,method='euler',name='batch_poisson_output')
    out.rate=b.linked_var(inp,'rate');out.run_regularly('k=poisson(rate); v+=k',when='groups',order=-1)
    bundle=lower_brian_dynamic_training(b.Network(inp,hidden,out),input_group=inp,layers=[hidden,out],
        external_state_inputs={'rate':BatchTimedArray(np.array([[1.3],[0.]]),dt=dt)},
        backend=engine,mpi_ranks=ranks,seed=8327,detach_reset=False)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(np.zeros((2,1,1)),[0,1],noise_sequence=9)
    source=bundle.provenance['external_state_inputs']['sources']['rate'];bank=source['bank']
    cells=bundle.provenance['neuron_state_layout'][out.name]['k'];counts=np.array(result['final_state'])[:,cells]
    np.testing.assert_array_equal(counts[1],[0,0])
    hard=(counts[0]>=1).astype(float);logits=5*hard;m=logits.max();positive_loss=m+np.log(np.exp(logits-m).sum())-logits[0]
    expected_positive=positive_loss*np.sum(counts[0]/1.3-1)/2
    base=np.log(2.);alternate=[np.logaddexp(0.,5.)-5.,np.logaddexp(0.,5.)]
    expected_zero=sum(x-base for x in alternate)/2
    np.testing.assert_allclose(external_state_input_vjp(source,result['gradients'][bank])[:,0],
        [expected_positive,expected_zero],rtol=5e-5,atol=4e-6)
    slot=bundle.plan['dynamic']['sample_index']['slot'];assert [row[slot] for row in result['final_state']]==[0,1]
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


def test_batch_table_units_preserve_si_values():
    b.set_device('runtime');b.start_scope();dt=.2*b.ms
    inp=b.NeuronGroup(2,'drive:volt',threshold='False',reset='',dt=dt,name='batch_unit_input')
    hidden=b.NeuronGroup(1,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',dt=dt,method='euler',name='batch_unit_hidden')
    out=b.NeuronGroup(2,'dv/dt=(-v+drive/mV)/ms:1\ndrive:volt (linked)',threshold='v>.5',reset='v-=.5',dt=dt,
        method='euler',name='batch_unit_output');out.drive=b.linked_var(inp,'drive')
    bundle=lower_brian_dynamic_training(b.Network(inp,hidden,out),input_group=inp,layers=[hidden,out],
        external_state_inputs={'drive':BatchTimedArray(DRIVES*b.mV,dt=dt)})
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);actual=trainer.evaluate(INPUTS,LABELS)
    for sample in range(2):
        v=np.zeros(2)
        for field in DRIVES[sample]:
            u=.8*v+.2*field;v=u-.5*(u>.5)
        np.testing.assert_allclose(actual['final_membrane'][sample][1:],v,atol=2e-13)
    source=bundle.provenance['external_state_inputs']['sources']['drive']
    np.testing.assert_allclose(np.array(bundle.weights[source['bank']]).reshape(6,2,2).transpose(1,0,2),DRIVES*.001)
