"""Physical external fields selected by mutable int32 Brian indices."""
import copy
import os
from unittest.mock import patch

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_stochastic import normal

PRE=np.array([1,0,1,0]);POST=np.array([0,1,1,0])
PATTERN=np.array([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]],float)
DELAYS=np.array([1,2,0,1])


def model(*,noisy=False,engine='cpu',ranks=None,window=None,event_driven=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    trace=b.TimedArray(np.array([[.4,.7],[1.2,.6],[.8,1.3],[1.4,.5],[.3,.9],[1.1,.4]]),dt=dt,name='indexed_external_trace')
    inp=b.NeuronGroup(2,'drive:1',threshold='timestep(t,dt)%2==i',reset='',dt=dt,
                      namespace={'trace':trace},name='indexed_external_input')
    inp.drive=999;inp.run_regularly('drive=trace(t,i)',when='before_start',order=-3)
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,
                         name=f'indexed_external_layer_{k}') for k in range(2)]
    groups[1].v=[.64,.29]
    code='dh/dt=(-h+gain*picked)/ms'
    if noisy:code+=' + .04*picked*(1+h)*xi/sqrt(ms)'
    syn=b.Synapses(inp,groups[1],code+(':1 (event-driven)' if event_driven else ':1 (clock-driven)')+'\nw:1\ngain:1 (constant)\npick:integer',
        on_pre='pick=1-pick; v_post+=w*(1+h)+.03*picked',
        on_post='pick=1-pick; h+=.01*picked',dt=dt,method='heun' if noisy else 'euler',name='indexed_external_synapses')
    syn.connect(i=PRE,j=POST);syn.w=[.24,.35,.18,.21];syn.h=[.1,.2,.15,.13]
    syn.gain=[.11,.08,.1,.07];syn.pick=[0,1,1,0];syn.delay=DELAYS*dt
    syn.variables.add_reference('picked',inp,'drive',index='pick')
    syn.run_regularly('h+=.003*picked; pick=1-pick',when='groups',order=-2)
    net=b.Network(inp,*groups,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,external_state_inputs={'drive':trace},
        trainable_synapse_parameters={syn.name:['w','gain']},detach_reset=False,seed=2087,
        backend=engine,mpi_ranks=ranks,tbptt_window=window,learning_rate=1e-9)
    return net,groups,syn,trace,bundle


def oracle(bundle,*,weights=None,initial=None,anchors=None,noisy=False,sample=0,change=None,event_driven=False):
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    state=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,parameter in enumerate(p['dynamic']['initial_parameters']):
            if parameter is not None:state[cell]=weights[parameter[0]][parameter[1]]
    layout=bundle.provenance['dynamic_state_layout']['indexed_external_synapses']
    voltage=bundle.provenance['neuron_state_layout']['indexed_external_layer_1']['v']
    banks={x['variables'][0]:x['bank'] for x in bundle.provenance['bindings']
           if x['object']=='indexed_external_synapses' and x['kind']!='parameter_index_table'}
    source=bundle.provenance['timed_inputs'][0];table=np.array(weights[source['bank']]).reshape(source['shape'])
    gain=np.array(weights[banks['gain']]);w=np.array(weights[banks['w']])
    domain=bundle.provenance['synaptic_noise_domains']['indexed_external_synapses']
    before=[];history=[];spikes=[];draws=[]
    for tick in range(len(PATTERN)):
        if change is not None and tick==change[0]:table=np.array(change[1]).reshape(source['shape'])
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:state=anchors['before'][tick].copy()
        before.append(state.copy());pick=state[layout['pick']].astype(int)
        # A complete regular vector block snapshots picked before changing pick.
        state[layout['h']]+=.003*table[tick,pick];pick=1-pick
        state[layout['pick']]=pick
        u=.8*state[voltage];state[voltage]=u
        for edge in ([] if event_driven else range(len(PRE))):
            h=state[layout['h'][edge]];value=table[tick,pick[edge]]
            update=h+.2*(-h+gain[edge]*value)
            if noisy:
                draw=normal(p['seed'],7,sample,domain,edge,tick,0);draws.append(draw)
                dw=np.sqrt(.2)*draw;g=.04*value*(1+h);support=h+g*dw
                update+=.5*dw*(g+.04*value*(1+support))
            state[layout['h'][edge]]=update
        hard=(u>.5).astype(float);event=hard
        if anchors is not None:
            old=anchors['voltage'][tick];hard=anchors['hard'][tick]
            event=hard+(u-old)/(1+5*abs(old-.5))**2
        for edge,target in enumerate(POST):
            emitted=tick-DELAYS[edge]
            if emitted>=0 and PATTERN[emitted,PRE[edge]]:
                selected=table[tick,pick[edge]];pick[edge]=1-pick[edge]
                if event_driven:
                    decay=np.exp(-(.0002*tick-state[layout['lastupdate'][edge]])/.001)
                    state[layout['h'][edge]]=decay*state[layout['h'][edge]]+(1-decay)*gain[edge]*selected
                    state[layout['lastupdate'][edge]]=.0002*tick
                state[voltage[target]]+=w[edge]*(1+state[layout['h'][edge]])+.03*selected
        for edge,target in enumerate(POST):
            selected=table[tick,pick[edge]]
            if hard[target]:pick[edge]=1-pick[edge]
            if event_driven:
                h=state[layout['h'][edge]];decay=np.exp(-(.0002*tick-state[layout['lastupdate'][edge]])/.001)
                changed=decay*h+(1-decay)*gain[edge]*selected+.01*selected
                state[layout['h'][edge]]=h+event[target]*(changed-h)
                if hard[target]:state[layout['lastupdate'][edge]]=.0002*tick
            else:state[layout['h'][edge]]+=.01*selected*event[target]
        state[layout['pick']]=pick;state[voltage]-=.5*event
        history.append(u.copy());spikes.append(event.copy())
    spikes=np.array(spikes);logits=5*spikes.mean(0);maximum=logits.max()
    loss=maximum+np.log(np.exp(logits-maximum).sum())-logits[0]
    return loss,state,spikes,dict(before=before,voltage=np.array(history),hard=spikes.copy(),draws=draws)


def physical_cells(bundle):
    return sorted({cell for layout in bundle.provenance['neuron_state_layout'].values() for name,cells in layout.items()
                   if not name.startswith('__') for cell in cells}
                  |{cell for layout in bundle.provenance['dynamic_state_layout'].values() for cells in layout.values() for cell in cells})


@pytest.mark.parametrize('noisy,event_driven',[(False,False),(True,False),(False,True)])
@pytest.mark.parametrize('window',[None,2])
def test_indexed_external_physics_and_all_vjps(engine,noisy,event_driven,window):
    *_,bundle=model(noisy=noisy,engine=engine,window=window,event_driven=event_driven)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(PATTERN[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    expected=oracle(bundle,noisy=noisy,event_driven=event_driven);cells=physical_cells(bundle);tol=5e-5 if engine!='cpu' else 4e-12
    np.testing.assert_array_equal(np.array(out['spikes'])[0,:,2:],expected[2])
    np.testing.assert_allclose(np.array(out['final_state'][0])[cells],expected[1][cells],rtol=tol,atol=tol*.01)
    assert out['loss']==pytest.approx(expected[0],abs=tol)
    integers={tuple(x) for x in bundle.plan['dynamic']['integer_parameters']}
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            if (bank,index) in integers:
                assert out['gradients'][bank][index]==0;continue
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(bundle,weights=hi,anchors=expected[3],noisy=noisy,event_driven=event_driven)[0]-oracle(bundle,weights=lo,anchors=expected[3],noisy=noisy,event_driven=event_driven)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=4e-4,abs=3e-6),(bank,index)
    for cell in cells:
        if bundle.plan['dynamic']['detached'][cell]:
            assert out['initial_state_gradients'][0][cell]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[cell]+=1e-6;lo[cell]-=1e-6
        fd=(oracle(bundle,initial=hi,anchors=expected[3],noisy=noisy,event_driven=event_driven)[0]-oracle(bundle,initial=lo,anchors=expected[3],noisy=noisy,event_driven=event_driven)[0])/2e-6
        assert out['initial_state_gradients'][0][cell]==pytest.approx(fd,rel=4e-4,abs=3e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
    assert any(isinstance(read['column'],dict) for read in bundle.provenance['external_state_inputs']['reads'])
    assert not any(node['op'].startswith('_deferred_') for programs in bundle.plan['dynamic']['program_sets'] for program in programs for node in program)


@pytest.mark.parametrize('noisy,event_driven',[(False,False),(True,False),(False,True)])
def test_indexed_external_original_cython_snapshot_semantics(noisy,event_driven):
    net,groups,syn,trace,bundle=model(noisy=noisy,event_driven=event_driven);expected=oracle(bundle,noisy=noisy,event_driven=event_driven)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(PATTERN[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    net.run(0*b.ms,namespace={});device=b.get_device();device.randn_buffer_index[:]=0;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);values=np.zeros(n);values[:len(expected[3]['draws'])]=expected[3]['draws'];return values
    try:
        with patch('numpy.random.randn',refill):net.run(1.2*b.ms,namespace={})
        assert device.randn_buffer_index[0]==len(expected[3]['draws'])
    finally:device.randn_buffer_index[:]=0
    layout=bundle.provenance['dynamic_state_layout'][syn.name]
    np.testing.assert_allclose(syn.h[:],expected[1][layout['h']],atol=2e-13)
    np.testing.assert_array_equal(syn.pick[:],expected[1][layout['pick']])
    np.testing.assert_allclose(groups[1].v[:],actual['final_membrane'][0][2:],atol=2e-13)
    assert syn.pre.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('noisy',[False,True])
def test_indexed_external_carry_replace_restore(engine,ranks,noisy,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,bundle=model(noisy=noisy,engine=engine,ranks=ranks);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(PATTERN[None,:3],[0],**(dict(noise_sequence=7) if noisy else {}))
    bank=bundle.provenance['timed_inputs'][0]['bank'];values=np.array(bundle.weights[bank])*.7+.2
    trainer.update_timed_input(bank,values);trainer.store(tmp_path/'indexed-external.json')
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(tmp_path/'indexed-external.json')
    actual=restored.step(PATTERN[None,3:],[0],initial='carry');expected=oracle(bundle,noisy=noisy,change=(3,values));cells=physical_cells(bundle)
    np.testing.assert_allclose(np.array(actual['final_state'][0])[cells],expected[1][cells],rtol=5e-5,atol=3e-6)


@pytest.mark.parametrize('noisy',[False,True])
def test_indexed_external_shared_table_batch_mean_gradient(engine,noisy):
    *_,bundle=model(noisy=noisy,engine=engine)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(np.repeat(PATTERN[None],2,axis=0),[0,0],**(dict(noise_sequence=7) if noisy else {}))
    expected=[oracle(bundle,noisy=noisy,sample=sample) for sample in range(2)];cells=physical_cells(bundle)
    for sample in range(2):np.testing.assert_allclose(np.array(out['final_state'][sample])[cells],expected[sample][1][cells],rtol=5e-5,atol=3e-6)
    integers={tuple(x) for x in bundle.plan['dynamic']['integer_parameters']}
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            if (bank,index) in integers:assert out['gradients'][bank][index]==0;continue
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=sum((oracle(bundle,weights=hi,anchors=expected[s][3],noisy=noisy,sample=s)[0]-oracle(bundle,weights=lo,anchors=expected[s][3],noisy=noisy,sample=s)[0])/2e-6 for s in range(2))/2
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=4e-4,abs=4e-6)


@pytest.mark.parametrize('bad',[-1,2])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_indexed_external_invalid_selector_fails_atomically(engine,ranks,bad):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,bundle=model(noisy=True,engine=engine,ranks=ranks);trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights,request_timeout=60)
    trainer.step(PATTERN[None,:2],[0],noise_sequence=7)
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.clock_state,trainer.noise_sequence,trainer.next_noise_sequence,trainer.last_result))
    invalid=copy.deepcopy(trainer.neuron_state);pick=bundle.provenance['dynamic_state_layout']['indexed_external_synapses']['pick']
    invalid[0][pick[0]]=bad
    with pytest.raises(ValueError):trainer.step(PATTERN[None,2:],[0],initial=invalid,start_tick=2,noise_sequence=7,clock_state=trainer.clock_state)
    assert (trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.clock_state,trainer.noise_sequence,trainer.next_noise_sequence,trainer.last_result)==before
