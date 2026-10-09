"""Explicit external linked fields in neuron integration, gates and resets."""
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

PRE=np.array([1,0,1,0]);POST=np.array([0,1,1,0]);DELAYS=np.array([1,2,0,1])
PATTERN=np.array([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]],float)


def model(*,runtime=True,noisy=False,engine='cpu',ranks=None,window=None,convert=True,refractory=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    trace=b.TimedArray(np.array([[.4,.7],[1.2,.6],[.8,1.3],[1.4,.5],[.3,.9],[1.1,.4]]),dt=dt,name='neuron_link_trace')
    inp=b.NeuronGroup(2,'drive:1',threshold='timestep(t,dt)%2==i',reset='',dt=dt,
                      namespace={'trace':trace},name='neuron_link_input')
    inp.drive=999;inp.run_regularly('drive=trace(t,i)',when='before_start',order=-3)
    hidden=b.NeuronGroup(1,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,name='neuron_link_hidden')
    equation='dv/dt=(-v+gain*drive)/ms'
    if noisy:equation+=' + .03*drive*(1+v)*xi/sqrt(ms)'
    flag=' (unless refractory)' if refractory else ''
    ref=3*dt if refractory=='fixed' else 'drive>1.05' if refractory=='field' else False
    out=b.NeuronGroup(2,equation+':1'+flag+'\ndrive:1 (linked)\ngain:1 (constant)\npick:integer',
        threshold='v>.5+.025*drive',reset='pick=1-pick; v-=.5+.025*drive',method='heun' if noisy else 'euler',
        dt=dt,refractory=ref,name='neuron_link_output')
    out.v=[.64,.29];out.gain=[.6,.9];out.pick=[0,1]
    if refractory:out.lastspike=-1*b.ms;out.not_refractory=True
    out.drive=b.linked_var(inp,'drive',index='pick' if runtime else np.array([1,0]))
    out.run_regularly('pick=1-pick; v+=.01*drive',when='groups',order=-1)
    syn=b.Synapses(inp,out,'w:1',on_pre='v_post+=w+.01*drive_pre',dt=dt,name='neuron_link_synapses')
    syn.connect(i=PRE,j=POST);syn.w=[.24,.35,.18,.21];syn.delay=DELAYS*dt
    net=b.Network(inp,hidden,out,syn)
    if not convert:return net,inp,hidden,out,syn,trace
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,out],external_state_inputs={'drive':trace},
        trainable_neuron_parameters={out.name:['gain']},trainable_synapse_parameters={syn.name:['w']},
        backend=engine,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,seed=2851,learning_rate=1e-9)
    return net,hidden,out,syn,trace,bundle


def oracle(bundle,*,weights=None,initial=None,anchors=None,runtime=True,noisy=False,change=None,refractory=False):
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    state=np.array(bundle.initial_state if initial is None else initial,float)
    for cell,param in enumerate(p['dynamic']['initial_parameters']):
        if initial is None and param is not None:state[cell]=weights[param[0]][param[1]]
    layout=bundle.provenance['neuron_state_layout']['neuron_link_output'];voltage=layout['v'];picker=layout['pick']
    source=bundle.provenance['timed_inputs'][0];table=np.array(weights[source['bank']]).reshape(source['shape'])
    gain_bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='neuron_link_output' and 'gain' in e['variables'])
    weight_bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='neuron_link_synapses' and e['variables']==['w'])
    gain=np.array(weights[gain_bank]);w=weights[weight_bank];before=[];margins=[];spikes=[];draws=[]
    latch=np.ones(2,bool);counter=layout.get('__refractory_ticks',[])
    for tick in range(len(PATTERN)):
        if change is not None and tick==change[0]:table=np.array(change[1]).reshape(source['shape'])
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:state=anchors['before'][tick].copy()
        before.append(state.copy());pick=state[picker].astype(int)
        old=table[tick,pick if runtime else [1,0]]
        pick=1-pick;state[voltage]+=.01*old*(latch if refractory else 1)
        field=table[tick,pick if runtime else [1,0]];v=state[voltage].copy();u=.8*v+.2*gain*field
        active=(state[counter]==0) if refractory=='fixed' else (latch | ~(field>1.05)) if refractory=='field' else np.ones(2,bool)
        if refractory=='fixed':state[counter]=np.maximum(state[counter]-1,0)
        if noisy:
            draw=np.array([normal(p['seed'],7,0,1,j,tick,0) for j in range(2)]);draws.extend(draw)
            dw=np.sqrt(.2)*draw;g=.03*field*(1+v);support=v+g*dw
            u+=.5*dw*(g+.03*field*(1+support))
        u=np.where(active,u,v);margin=u-(.5+.025*field);hard=((margin>0)&active).astype(float);event=hard
        if anchors is not None:
            old_margin=anchors['margins'][tick];hard=anchors['hard'][tick]
            event=hard+active*(margin-old_margin)/(1+5*abs(old_margin))**2
        state[voltage]=u
        latch=active & ~hard.astype(bool)
        for edge,target in enumerate(POST):
            emitted=tick-DELAYS[edge]
            if emitted>=0:state[voltage[target]]+=PATTERN[emitted,PRE[edge]]*(w[edge]+.01*table[tick,PRE[edge]])*(latch[target] if refractory else 1)
        # The reset reads drive before its integer selector write.
        state[picker]=np.where(hard,1-pick,pick)
        state[voltage]-=event*(.5+.025*field);margins.append(margin.copy());spikes.append(event.copy())
        if refractory=='fixed':state[counter]=np.where(hard,2,state[counter])
    spikes=np.array(spikes);logits=5*spikes.mean(0);maximum=logits.max()
    loss=maximum+np.log(np.exp(logits-maximum).sum())-logits[0]
    return loss,state,spikes,dict(before=before,margins=np.array(margins),hard=spikes.copy(),draws=draws)


@pytest.mark.parametrize('runtime',[False,True])
@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('refractory',[False,'fixed','field'])
def test_external_neuron_link_all_physics_and_vjps(engine,runtime,noisy,window,refractory):
    *_,bundle=model(runtime=runtime,noisy=noisy,engine=engine,window=window,refractory=refractory)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(PATTERN[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    expected=oracle(bundle,runtime=runtime,noisy=noisy,refractory=refractory);layout=bundle.provenance['neuron_state_layout']['neuron_link_output']
    cells=layout['v']+layout['pick'];tol=5e-5 if engine!='cpu' else 4e-12
    np.testing.assert_allclose(np.array(result['final_state'][0])[cells],expected[1][cells],rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(np.array(result['spikes'])[0,:,1:],expected[2])
    assert result['loss']==pytest.approx(expected[0],abs=tol)
    integers={tuple(x) for x in bundle.plan['dynamic']['integer_parameters']}
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            if (bank,index) in integers:assert result['gradients'][bank][index]==0;continue
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(bundle,weights=hi,anchors=expected[3],runtime=runtime,noisy=noisy,refractory=refractory)[0]-oracle(bundle,weights=lo,anchors=expected[3],runtime=runtime,noisy=noisy,refractory=refractory)[0])/2e-6
            assert result['gradients'][bank][index]==pytest.approx(fd,rel=4e-4,abs=4e-6),(bank,index)
    for cell in layout['v']:
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[cell]+=1e-6;lo[cell]-=1e-6
        fd=(oracle(bundle,initial=hi,anchors=expected[3],runtime=runtime,noisy=noisy,refractory=refractory)[0]-oracle(bundle,initial=lo,anchors=expected[3],runtime=runtime,noisy=noisy,refractory=refractory)[0])/2e-6
        assert result['initial_state_gradients'][0][cell]==pytest.approx(fd,rel=4e-4,abs=4e-6)
    assert all(result['initial_state_gradients'][0][cell]==0 for cell in layout['drive']+layout['pick'])
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('runtime',[False,True])
@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('refractory',[False,'fixed','field'])
def test_external_neuron_link_original_cython(runtime,noisy,refractory):
    net,hidden,out,syn,trace,bundle=model(runtime=runtime,noisy=noisy,refractory=refractory);expected=oracle(bundle,runtime=runtime,noisy=noisy,refractory=refractory)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(PATTERN[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    net.run(0*b.ms,namespace={});device=b.get_device();device.randn_buffer_index[:]=0;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);values=np.zeros(n);values[:len(expected[3]['draws'])]=expected[3]['draws'];return values
    try:
        with patch('numpy.random.randn',refill):net.run(1.2*b.ms,namespace={})
        assert device.randn_buffer_index[0]==len(expected[3]['draws'])
    finally:device.randn_buffer_index[:]=0
    np.testing.assert_allclose(out.v[:],actual['final_membrane'][0][1:],atol=2e-13)
    np.testing.assert_array_equal(out.pick[:],expected[1][bundle.provenance['neuron_state_layout'][out.name]['pick']])
    assert out.state_updater.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('noisy',[False,True])
def test_external_neuron_link_carry_update_restore(engine,ranks,noisy,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,bundle=model(noisy=noisy,engine=engine,ranks=ranks);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(PATTERN[None,:3],[0],**(dict(noise_sequence=7) if noisy else {}))
    bank=bundle.provenance['timed_inputs'][0]['bank'];values=np.array(bundle.weights[bank])*.7+.2
    trainer.update_timed_input(bank,values);trainer.store(tmp_path/'neuron-field.json')
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(tmp_path/'neuron-field.json')
    actual=restored.step(PATTERN[None,3:],[0],initial='carry');expected=oracle(bundle,noisy=noisy,change=(3,values))
    layout=bundle.provenance['neuron_state_layout']['neuron_link_output'];cells=layout['v']+layout['pick']
    np.testing.assert_allclose(np.array(actual['final_state'][0])[cells],expected[1][cells],rtol=5e-5,atol=3e-6)


@pytest.mark.parametrize('writer',['reset','regular','pathway'])
def test_external_neuron_link_remains_read_only(writer):
    net,inp,hidden,out,syn,trace=model(convert=False)
    if writer=='reset':out.event_codes['spike']='drive=1'
    if writer=='regular':out.run_regularly('drive=1',when='end')
    if writer=='pathway':syn.pre.code='drive_pre=1'
    with pytest.raises(ValueError,match='external.*read-only'):
        lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,out],external_state_inputs={'drive':trace})
