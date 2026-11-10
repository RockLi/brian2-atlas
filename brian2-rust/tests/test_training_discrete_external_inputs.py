"""Exact discrete external fields: physical recurrence, Cython and native VJP."""
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

CODES=np.array([[16777217,-2147483648],[2147483647,16777216],[-1,16777217],
                [16777217,2147483647],[0,-2147483648],[16777216,16777217]],np.int64)
FLAGS=np.array([[1,0],[0,1],[1,1],[0,1],[1,0],[1,1]],bool)
PATTERN=np.array([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]],float)


def model(*,engine='cpu',ranks=None,window=None,convert=True,noisy=False,shared=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    codes=b.TimedArray(CODES[:,0] if shared else CODES,dt=dt,name='external_codes')
    flags=b.TimedArray(FLAGS[:,0] if shared else FLAGS,dt=dt,name='external_flags')
    declaration='code:integer (shared)\nflag:boolean (shared)' if shared else 'code:integer\nflag:boolean'
    inp=b.NeuronGroup(2,declaration,threshold='timestep(t,dt)%2==i',reset='',dt=dt,
                      namespace={'codes':codes,'flags':flags},name='discrete_input')
    inp.run_regularly('code=int(codes(t)); flag=flags(t)>0' if shared else 'code=int(codes(t,i)); flag=flags(t,i)>0',when='before_start',order=-3)
    hidden=b.NeuronGroup(1,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,name='discrete_hidden')
    equation='dv/dt=(-v+gain*int(code==16777217))/ms'
    if noisy:equation+=' + .03*int(code==16777217)*(1+v)*xi/sqrt(ms)'
    out=b.NeuronGroup(2,equation+':1\ncode:integer (linked)\nflag:boolean (linked)\n'
        'pick:integer\nseen:integer\ngain:1 (constant)',threshold='v>.5 and flag',reset='seen=code; v-=.5',
        method='heun' if noisy else 'euler',dt=dt,name='discrete_output')
    out.v=[.64,.29];out.gain=[.6,.9];out.pick=[0,1]
    out.code=b.linked_var(inp,'code',**({} if shared else dict(index='pick')))
    out.flag=b.linked_var(inp,'flag',**({} if shared else dict(index=np.array([1,0]))))
    out.run_regularly('seen=code; pick=1-pick; v+=.01*int(code==16777217)',when='groups',order=-1)
    syn=b.Synapses(inp,out,'w:1\nedge_seen:integer',on_pre='edge_seen=code_pre; v_post+=w*(1+int(edge_seen==16777217))',
                   dt=dt,name='discrete_synapses')
    syn.connect(i=[0,1],j=[0,1]);syn.w=[.24,.35]
    net=b.Network(inp,hidden,out,syn)
    if not convert:return net,inp,hidden,out,syn,codes,flags
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,out],
        external_state_inputs={'code':codes,'flag':flags},trainable_neuron_parameters={out.name:['gain']},
        trainable_synapse_parameters={syn.name:['w']},backend=engine,mpi_ranks=ranks,tbptt_window=window,
        detach_reset=False,learning_rate=1e-9)
    return net,out,syn,bundle


def oracle(bundle,*,weights=None,initial=None,anchors=None,codes=CODES,flags=FLAGS,noisy=False,shared=False,steps=6):
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    state=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,param in enumerate(p['dynamic']['initial_parameters']):
            if param is not None:state[cell]=weights[param[0]][param[1]]
    layout=bundle.provenance['neuron_state_layout']['discrete_output']
    syn=bundle.provenance['dynamic_state_layout']['discrete_synapses']
    bank=lambda obj,name:next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==obj and name in e['variables'])
    gains=np.array(weights[bank('discrete_output','gain')]);w=np.array(weights[bank('discrete_synapses','w')])
    if shared:
        codes=np.repeat(np.asarray(codes)[:,0,None],2,axis=1);flags=np.repeat(np.asarray(flags)[:,0,None],2,axis=1)
    before=[];margins=[];spikes=[];draws=[]
    for tick,external in enumerate(PATTERN[:steps]):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
            state=anchors['before'][tick].copy()
        before.append(state.copy());pick=state[layout['pick']].astype(int)
        old=codes[tick,pick];state[layout['seen']]=old;state[layout['pick']]=1-pick
        v=state[layout['v']]+.01*(old==16777217);field=codes[tick,1-pick]
        u=.8*v+.2*gains*(field==16777217);margin=u-.5;active=flags[tick,[1,0]]
        if noisy:
            draw=np.array([normal(p['seed'],7,0,1,j,tick,0) for j in range(2)]);draws.extend(draw)
            dw=np.sqrt(.2)*draw;g=.03*(field==16777217)*(1+v);support=v+g*dw
            u+=.5*dw*(g+.03*(field==16777217)*(1+support));margin=u-.5
        hard=((margin>0)&active).astype(float);event=hard
        if anchors is not None:
            # The declared lazy Boolean VJP differentiates only visited
            # operands: when the first operand is false, flag is not read.
            slope=np.where(anchors['margins'][tick]>0,active,True)
            event=anchors['hard'][tick]+slope*(margin-anchors['margins'][tick])/(1+5*abs(anchors['margins'][tick]))**2
        state[bundle.provenance['threshold_margin_layout']['discrete_output']]=hard
        state[layout['v']]=u+external*w*(1+(codes[tick]==16777217))-.5*event
        for edge in range(2):
            if external[edge]:state[syn['edge_seen'][edge]]=codes[tick,edge]
        state[layout['seen']]=np.where(hard,field,state[layout['seen']])
        margins.append(margin);spikes.append(event)
    spikes=np.array(spikes);logits=5*spikes.mean(0);m=logits.max()
    return m+np.log(np.exp(logits-m).sum())-logits[0],state,spikes,dict(before=before,margins=np.array(margins),hard=spikes.copy(),draws=draws)


@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('noisy',[False,True])
def test_discrete_external_exact_physics_and_all_vjps(engine,window,noisy):
    *_,bundle=model(engine=engine,window=window,noisy=noisy)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(PATTERN[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    expected=oracle(bundle,noisy=noisy);tol=5e-5 if engine!='cpu' else 3e-12
    layout=bundle.provenance['neuron_state_layout']['discrete_output'];syn=bundle.provenance['dynamic_state_layout']['discrete_synapses']
    np.testing.assert_allclose(result['final_state'][0],expected[1],rtol=tol,atol=tol*.01)
    for cells in (layout['seen'],layout['pick'],syn['edge_seen']):
        np.testing.assert_array_equal(np.array(result['final_state'][0])[cells],expected[1][cells])
    np.testing.assert_array_equal(np.array(result['spikes'])[0,:,1:],expected[2]);assert result['loss']==pytest.approx(expected[0],abs=tol)
    h=2e-6
    for bank,values in enumerate(bundle.weights):
        if not bundle.plan['trainable'][bank]:
            # Timed discrete controls and index banks are detached.
            assert not np.any(result['gradients'][bank]);continue
        for j in range(len(values)):
            losses=[]
            for sign in (-1,1):
                changed=copy.deepcopy(bundle.weights);changed[bank][j]+=sign*h
                losses.append(oracle(bundle,weights=changed,anchors=expected[3],noisy=noisy)[0])
            assert result['gradients'][bank][j]==pytest.approx((losses[1]-losses[0])/(2*h),rel=max(3e-7,tol*10),abs=max(2e-9,tol))
    for cell in layout['v']:
        losses=[]
        for sign in (-1,1):
            changed=np.array(bundle.initial_state);changed[cell]+=sign*h
            losses.append(oracle(bundle,initial=changed,anchors=expected[3],noisy=noisy)[0])
        assert result['initial_state_gradients'][0][cell]==pytest.approx((losses[1]-losses[0])/(2*h),rel=max(3e-7,tol*10),abs=max(2e-9,tol))
    assert all(result['initial_state_gradients'][0][k]==0 for name in ('code','flag','pick','seen') for k in layout[name])
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('noisy',[False,True])
def test_discrete_external_original_cython(noisy):
    net,out,syn,bundle=model(noisy=noisy);actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(PATTERN[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    expected=oracle(bundle,noisy=noisy);net.run(0*b.ms,namespace={});device=b.get_device();device.randn_buffer_index[:]=0
    calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);values=np.zeros(n);values[:len(expected[3]['draws'])]=expected[3]['draws'];return values
    try:
        with patch('numpy.random.randn',refill):net.run(1.2*b.ms,namespace={})
        if noisy:assert device.randn_buffer_index[0]==len(expected[3]['draws'])
    finally:device.randn_buffer_index[:]=0
    layout=bundle.provenance['neuron_state_layout'][out.name];edge=bundle.provenance['dynamic_state_layout'][syn.name]
    np.testing.assert_allclose(out.v[:],actual['final_membrane'][0][1:],atol=2e-13)
    for name in ('seen','pick'):np.testing.assert_array_equal(getattr(out,name)[:],np.array(actual['final_state'][0])[layout[name]])
    np.testing.assert_array_equal(syn.edge_seen[:],np.array(actual['final_state'][0])[edge['edge_seen']])
    assert out.state_updater.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('noisy',[False,True])
def test_discrete_external_atomic_update_carry_restore(engine,ranks,noisy,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,bundle=model(engine=engine,ranks=ranks,noisy=noisy);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(PATTERN[None,:3],[0],**(dict(noise_sequence=7) if noisy else {}))
    sources=bundle.provenance['external_state_inputs']['sources'];replacement=CODES[:,::-1].copy()
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.clock_state,trainer.noise_sequence))
    for invalid in (CODES+.5,CODES.astype(float)*100,CODES[:2],CODES.astype(float)*np.nan):
        with pytest.raises(ValueError):trainer.update_external_state_input(sources['code'],invalid)
        assert (trainer.state,trainer.neuron_state,trainer.clock_tick,trainer.clock_state,trainer.noise_sequence)==before
    trainer.update_external_state_input(sources['code'],replacement)
    with pytest.raises(ValueError):trainer.update_external_state_input(sources['flag'],FLAGS.astype(float)+.1)
    replacement_flags=FLAGS[:,::-1].copy();trainer.update_external_state_input(sources['flag'],replacement_flags)
    assert (trainer.neuron_state,trainer.clock_tick,trainer.clock_state,trainer.noise_sequence)==before[1:]
    trainer.store(tmp_path/'discrete.json');restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    restored.restore(tmp_path/'discrete.json');actual=restored.step(PATTERN[None,3:],[0],initial='carry')
    changed=CODES.copy();changed[3:]=replacement[3:];flags=FLAGS.copy();flags[3:]=replacement_flags[3:]
    expected=oracle(bundle,codes=changed,flags=flags,noisy=noisy)
    np.testing.assert_allclose(actual['final_state'][0],expected[1],rtol=5e-5,atol=3e-6)


@pytest.mark.parametrize('shared',[False,True])
def test_discrete_external_every_prefix_exact(engine,shared):
    *_,bundle=model(engine=engine,shared=shared)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    layout=bundle.provenance['neuron_state_layout']['discrete_output'];syn=bundle.provenance['dynamic_state_layout']['discrete_synapses']
    cells=layout['seen']+layout['pick']+syn['edge_seen']
    for steps in range(1,7):
        actual=trainer.evaluate(PATTERN[None,:steps],[0]);expected=oracle(bundle,shared=shared,steps=steps)
        np.testing.assert_array_equal(np.array(actual['final_state'][0])[cells],expected[1][cells])


@pytest.mark.parametrize('writer',['reset','regular','pathway'])
def test_discrete_external_fields_read_only(writer):
    net,inp,hidden,out,syn,codes,flags=model(convert=False)
    if writer=='reset':out.event_codes['spike']='code=1'
    if writer=='regular':out.run_regularly('flag=True',when='end')
    if writer=='pathway':syn.pre.code='code_pre=1'
    with pytest.raises(ValueError,match='external.*read-only'):
        lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,out],external_state_inputs={'code':codes,'flag':flags})


@pytest.mark.parametrize('field,bad',[('code',.5),('code',2**31),('code',-2**31-1),('flag',2),('flag',-.1)])
def test_discrete_external_rejects_inexact_values(field,bad):
    net,inp,hidden,out,syn,codes,flags=model(convert=False)
    table=b.TimedArray(np.full(CODES.shape,bad),dt=.2*b.ms)
    with pytest.raises(ValueError,match='external (integer|boolean) values'):
        lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,out],
            external_state_inputs={'code':table if field=='code' else codes,'flag':table if field=='flag' else flags})
