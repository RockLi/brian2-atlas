"""Explicit physical input-field traces, native VJPs and source isolation."""
import copy
import os

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training, lower_brian_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_stochastic import normal

PRE=np.array([1,0,1,0]);POST=np.array([0,1,1,0])
PATTERN=np.array([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]],float)
DELAY=np.array([1,2,0,1])


def model(*,noisy=False,engine='cpu',ranks=None,window=None,convert=True,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    trace=b.TimedArray(np.array([[.4,.7],[1.2,.6],[.8,1.3],[1.4,.5],[.3,.9],[1.1,.4]])*b.mV,dt=dt,name='external_drive_trace')
    bias=b.TimedArray(np.array([.1,.2,.15,.3,.05,.25])*b.mV,dt=dt,name='external_bias_trace')
    inp=b.NeuronGroup(2,'drive:volt\nbias:volt (shared)',threshold='timestep(t,dt)%2==i',reset='',
                      dt=dt,name='external_input',namespace={'trace':trace,'bias_trace':bias})
    inp.drive=999*b.mV;inp.bias=888*b.mV
    source=inp.run_regularly('bias=bias_trace(t); drive=trace(t,i)',when='before_start',order=-3)
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',dt=dt,
                         method='euler',name=f'external_layer_{k}') for k in range(2)]
    groups[1].v=[.64,.29]
    code='dh/dt=(-h+.02*drive_pre/mV+.01*bias_pre/mV)/ms'
    if noisy:code+=' + .04*drive_pre/mV*(1+h)*xi/sqrt(ms)'
    syn=b.Synapses(inp,groups[1],code+':1 (clock-driven)\nw:1',
        on_pre='v_post+=w*(1+h)+.03*drive_pre/mV',on_post='h+=.01*drive_pre/mV',
        dt=dt,method='heun' if noisy else 'euler',name='z_external_synapses')
    syn.connect(i=PRE,j=POST);syn.w=[.24,.35,.18,.21];syn.h=[.1,.2,.15,.13]
    syn.delay=DELAY*dt
    syn.run_regularly('h+=.004*drive_pre/mV',when='groups',order=-2)
    net=b.Network(inp,*groups,syn)
    if not convert:return net,inp,groups,syn,trace,bias,source
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,
        external_state_inputs={'drive':trace,'bias':bias},trainable_synapse_parameters={syn.name:['w']},
        detach_reset=False,seed=1709,backend=engine,mpi_ranks=ranks,tbptt_window=window,
        learning_rate=1e-9,**options)
    return net,inp,groups,syn,trace,bias,bundle


def oracle(bundle,*,weights=None,initial=None,anchors=None,noisy=False,change=None):
    weights=bundle.weights if weights is None else weights;p=bundle.plan
    state=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,parameter in enumerate(p['dynamic']['initial_parameters']):
            if parameter is not None:state[cell]=weights[parameter[0]][parameter[1]]
    entries={entry['name']:entry for entry in bundle.provenance['timed_inputs']}
    field=entries['external_drive_trace'];drive=np.array(weights[field['bank']]).reshape(field['shape'])/.001
    bias=np.array(weights[entries['external_bias_trace']['bank']])/.001
    layout=bundle.provenance['dynamic_state_layout']['z_external_synapses'];voltage=bundle.provenance['neuron_state_layout']['external_layer_1']['v']
    weight_bank=next(x['bank'] for x in bundle.provenance['bindings'] if x['object']=='z_external_synapses' and x['variables']==['w'])
    w=weights[weight_bank]
    domain=bundle.provenance['synaptic_noise_domains']['z_external_synapses']
    history=[];spikes=[];before=[]
    for tick in range(len(PATTERN)):
        if change is not None and tick==change[0]:drive=np.array(change[1]).reshape(field['shape'])/.001
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:state=anchors['before'][tick].copy()
        before.append(state.copy());u=.8*state[voltage];state[voltage]=u
        for edge,source in enumerate(PRE):
            h=state[layout['h'][edge]]+.004*drive[tick,source]
            drift=-h+.02*drive[tick,source]+.01*bias[tick]
            value=h+.2*drift
            if noisy:
                dw=np.sqrt(.2)*normal(p['seed'],7,0,domain,edge,tick,0)
                g=.04*drive[tick,source]*(1+h);support=h+g*dw
                value+=.5*dw*(g+.04*drive[tick,source]*(1+support))
            state[layout['h'][edge]]=value
        event=(u>.5).astype(float)
        if anchors is not None:
            old=anchors['voltage'][tick];event=anchors['hard'][tick]+(u-old)/(1+5*abs(old-.5))**2
        for edge,target in enumerate(POST):
            emitted=tick-DELAY[edge]
            if emitted>=0:state[voltage[target]]+=PATTERN[emitted,PRE[edge]]*(w[edge]*(1+state[layout['h'][edge]])+.03*drive[tick,PRE[edge]])
        for edge,target in enumerate(POST):state[layout['h'][edge]]+=.01*drive[tick,PRE[edge]]*event[target]
        state[voltage]-=.5*event;history.append(u.copy());spikes.append(event.copy())
    spikes=np.array(spikes);logits=5*spikes.mean(0);maximum=logits.max()
    loss=maximum+np.log(np.exp(logits-maximum).sum())-logits[0]
    return loss,state,spikes,dict(before=before,voltage=np.array(history),hard=spikes.copy())


def physical_cells(bundle):
    return sorted({cell for layout in bundle.provenance['neuron_state_layout'].values() for name,cells in layout.items()
                   if not name.startswith('__') for cell in cells}
                  |{cell for layout in bundle.provenance['dynamic_state_layout'].values() for cells in layout.values() for cell in cells})


@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('window',[None,2])
def test_external_state_physics_and_all_table_vjps(engine,noisy,window):
    *_,bundle=model(noisy=noisy,engine=engine,window=window)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(PATTERN[None],[0],noise_sequence=7 if noisy else None)
    expected=oracle(bundle,noisy=noisy);tol=5e-5 if engine!='cpu' else 4e-12
    np.testing.assert_array_equal(np.array(out['spikes'])[0,:,2:],expected[2])
    cells=physical_cells(bundle)
    np.testing.assert_allclose(np.array(out['final_state'][0])[cells],expected[1][cells],rtol=tol,atol=tol*.01)
    assert out['loss']==pytest.approx(expected[0],abs=tol)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights)
            epsilon=1e-9 if bank in {x['bank'] for x in bundle.provenance['timed_inputs']} else 1e-6
            hi[bank][index]+=epsilon;lo[bank][index]-=epsilon
            fd=(oracle(bundle,weights=hi,anchors=expected[3],noisy=noisy)[0]-oracle(bundle,weights=lo,anchors=expected[3],noisy=noisy)[0])/(2*epsilon)
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=4e-4,abs=3e-5 if engine!='cpu' else 3e-7),(bank,index)
    for cell in physical_cells(bundle):
        if bundle.plan['dynamic']['detached'][cell]:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[cell]+=1e-6;lo[cell]-=1e-6
        fd=(oracle(bundle,initial=hi,anchors=expected[3],noisy=noisy)[0]-oracle(bundle,initial=lo,anchors=expected[3],noisy=noisy)[0])/2e-6
        assert out['initial_state_gradients'][0][cell]==pytest.approx(fd,rel=4e-4,abs=3e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def test_external_state_matches_original_cython_source_generator():
    net,inp,groups,syn,*_,bundle=model()
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(PATTERN[None],[0])
    monitor=b.SpikeMonitor(inp);net.add(monitor);net.run(1.2*b.ms,namespace={})
    spikes=np.zeros_like(PATTERN);spikes[np.rint(monitor.t/(.2*b.ms)).astype(int),monitor.i[:]]=1
    np.testing.assert_array_equal(spikes,PATTERN)
    np.testing.assert_allclose(actual['final_membrane'][0][2:],groups[1].v[:],atol=2e-13)
    layout=bundle.provenance['dynamic_state_layout'][syn.name]
    np.testing.assert_allclose(np.array(actual['final_state'][0])[layout['h']],syn.h[:],atol=2e-13)


def test_external_state_public_entry_snapshot_and_atomic_read_only():
    net,inp,groups,syn,trace,bias,_=model(convert=False)
    bundle=lower_brian_training(net,input_group=inp,layers=groups,dynamic=True,
        external_state_inputs={'drive':trace,'bias':bias})
    provenance=bundle.provenance['external_state_inputs']
    assert provenance['sampling']=='consumer-clock-timed-array' and provenance['batch_semantics']=='shared'
    assert {read['source_variable'] for read in provenance['reads']}=={'drive','bias'}
    original=copy.deepcopy(bundle.weights)
    trace.values[:]=999;bias.values[:]=888
    assert bundle.weights==original
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(PATTERN[None,:2],[0])
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.clock_state,
                          trainer.noise_sequence,trainer.next_noise_sequence))
    trainer.evaluate(PATTERN[None,2:],[0],initial='carry')
    trainer.gradients(PATTERN[None,2:],[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.clock_state,
            trainer.noise_sequence,trainer.next_noise_sequence)==before
    bank=bundle.provenance['timed_inputs'][0]['bank']
    with pytest.raises(ValueError):trainer.update_timed_input(bank,[float('nan')]*len(bundle.weights[bank]))
    assert (trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.clock_state,
            trainer.noise_sequence,trainer.next_noise_sequence)==before


def test_external_state_async_consumer_clock_matches_explicit_brian_table():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    trace=b.TimedArray([.1,.7,.2,1.1,.4,.9,.5],dt=.2*b.ms)
    inp=b.NeuronGroup(1,'drive:1',threshold='True',reset='',dt=.2*b.ms,name='external_async_input')
    inp.drive=999
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',method='euler',
                         dt=.2*b.ms,name=f'external_async_layer_{k}') for k in range(2)]
    syn=b.Synapses(inp,groups[1],'w:1',on_pre='v_post+=w*drive_pre',dt=.3*b.ms,
                   namespace={'trace':trace},name='external_async_synapses')
    syn.connect();syn.w=[.4,.25];syn.delay=[.2,.4]*b.ms
    net=b.Network(inp,*groups,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,external_state_inputs={'drive':trace})
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(np.ones((1,6,1)),[0])
    # The explicit contract samples the consumer clock, which may be ahead of
    # the source clock. This reference uses Brian's own table implementation.
    syn.pre.code='v_post+=w*trace(t)'
    monitor=b.SpikeMonitor(groups[1]);net.add(monitor);net.run(1.2*b.ms,namespace={})
    expected=np.zeros((6,2));expected[np.rint(monitor.t/(.2*b.ms)).astype(int),monitor.i[:]]=1
    np.testing.assert_array_equal(np.array(actual['spikes'])[0,:,2:],expected)
    np.testing.assert_allclose(actual['final_membrane'][0][2:],groups[1].v[:],atol=2e-13)


def test_external_state_cannot_override_selected_graph_storage_through_input_alias():
    net,inp,groups,syn,trace,bias,_=model(convert=False)
    inp.variables.add_reference('foreign',groups[0],'v')
    syn.variables.add_reference('foreign_pre',inp,'foreign',index='_synaptic_pre')
    syn.pre.code+='; v_post+=foreign_pre'
    foreign=b.TimedArray(np.ones((6,2)),dt=.2*b.ms)
    with pytest.raises(ValueError,match='external.*own|own.*input'):
        lower_brian_dynamic_training(net,input_group=inp,layers=groups,
            external_state_inputs={'drive':trace,'bias':bias,'foreign':foreign})


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('noisy',[False,True])
def test_external_state_carry_replace_and_new_trainer_restore(engine,ranks,noisy,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,bundle=model(noisy=noisy,engine=engine,ranks=ranks)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(PATTERN[None,:3],[0],**(dict(noise_sequence=7) if noisy else {}))
    bank=next(x['bank'] for x in bundle.provenance['timed_inputs'] if x['name']=='external_drive_trace')
    replacement=np.array(bundle.weights[bank])*.7+.0002
    before=copy.deepcopy((trainer.neuron_state,trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence,trainer.clock_state))
    trainer.update_timed_input(bank,replacement)
    assert (trainer.neuron_state,trainer.clock_tick,trainer.noise_sequence,trainer.next_noise_sequence,trainer.clock_state)==before
    trainer.store(tmp_path/'external.json')
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(tmp_path/'external.json')
    actual=restored.step(PATTERN[None,3:],[0],initial='carry')
    expected=oracle(bundle,noisy=noisy,change=(3,replacement))
    cells=physical_cells(bundle)
    np.testing.assert_allclose(np.array(actual['final_state'][0])[cells],expected[1][cells],rtol=5e-5,atol=3e-6)


@pytest.mark.parametrize('issue',['shape','units','integer','nonfinite','missing','write','regular_write','unmapped_runner','static','absent_contract'])
def test_external_state_invalid_contracts_are_rejected(issue):
    net,inp,groups,syn,trace,bias,source=model(convert=False)
    fields={'drive':trace,'bias':bias}
    if issue=='shape':fields['drive']=b.TimedArray([1,2]*b.mV,dt=.2*b.ms)
    if issue=='units':fields['drive']=b.TimedArray(np.ones((2,2)),dt=.2*b.ms)
    if issue=='integer':fields['i']=trace
    if issue=='nonfinite':fields['drive']=b.TimedArray(np.full((2,2),np.nan)*b.mV,dt=.2*b.ms)
    if issue=='missing':fields['absent']=trace
    if issue=='write':syn.pre.code='drive_pre+=.1*mV'
    if issue=='regular_write':syn.run_regularly('drive_pre+=.1*mV',when='end')
    if issue=='unmapped_runner':fields.pop('bias')
    if issue=='absent_contract':fields=None
    fn=lower_brian_training if issue=='static' else lower_brian_dynamic_training
    with pytest.raises(ValueError,match='external|input|matching|read-only|supplied|selected'):
        fn(net,input_group=inp,layers=groups,external_state_inputs=fields)
