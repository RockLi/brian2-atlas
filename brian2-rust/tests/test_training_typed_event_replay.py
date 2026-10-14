"""Typed constant array replay retains private aliases and exact integer bits."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_typed_callback_effects import increment,fill


def model(kind,discard,ranks,window,delay,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';b.prefs.codegen.runtime.numpy.discard_units=discard
    dt=.2*b.ms;source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.317,.731]
    boolean=kind=='boolean';shared=kind=='shared_integer'
    coefficient='flagc:boolean (constant)' if boolean else 'kc:integer ('+('shared, ' if shared else '')+'constant)'
    factor='0.1' if boolean or shared else '1e-10' if kind=='wrap' else '0.01'
    operand='int(temp%2)' if shared else 'temp'
    alias='int(alias%2)' if shared else 'alias'
    code='temp=change('+('flagc' if boolean else 'kc')+');alias=temp;v_post+='+factor+'*gain*'+operand+';temp*='+('False' if boolean else '2')+';h='+factor+'*'+alias+'+v_post'
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)\n'+coefficient,on_pre=code,dt=dt,
                   namespace={'change':b.Function(fill if boolean else increment,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)})
    syn.connect(i=[0,0],j=[0,0]);syn.h=[.213,.327];syn.gain=[.113,.217];syn.delay=delay*dt
    if boolean:syn.flagc=[True,False]
    else:syn.kc=16777217 if shared else [2147483647,-2147483648] if kind=='wrap' else [2,3]
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,
                                       trainable_synapse_parameters={syn.name:['h','gain']})
    assert len(bundle.provenance['event_callback_stage_groups'][syn.pre.name])==2
    x=np.zeros((1,4,1));x[0,[0,2],0]=1
    return net,g,syn,dt,bundle,x


def oracle(bundle,g,syn,kind,delay,weights,initial=None,anchors=None):
    p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for slot,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[slot]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    gain_bank=next(r['bank'] for r in bundle.provenance['bindings'] if r['object']==syn.name and r['variables']==['gain'])
    gain=np.array(weights[gain_bank]);stages=bundle.provenance['event_callback_stage_groups'][syn.pre.name]
    if kind=='boolean':temporary=np.ones(2,bool);factor=.1;scaled=np.zeros(2,bool)
    elif kind=='shared_integer':temporary=np.ones(2,np.int32);scaled=temporary.copy();factor=.1
    else:
        temporary=np.array([16777217,16777217] if kind=='shared_integer' else [2147483647,-2147483648] if kind=='wrap' else [2,3],np.int32)
        temporary+=1;scaled=temporary.copy();scaled*=2;factor=1e-10 if kind=='wrap' else .01
    before=[];margins=[];hard=[];spikes=[];events=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        for stage,path in enumerate(stages):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            queues=bundle.provenance['delay_queues'][path]['new']
            arrivals=np.array([z[row['states'][0]] if row['states'] else float(tick in (0,2)) for row in queues])
            for edge in range(2):
                if stage==0:z[v[0]]+=arrivals[edge]*factor*gain[edge]*temporary[edge]
                else:z[h[edge]]+=arrivals[edge]*(factor*scaled[edge]+z[v[0]]-z[h[edge]])
            for row in queues:
                slots=row['states']
                if slots:z[slots[:-1]]=z[slots[1:]];z[slots[-1]]=float(tick in (0,2))
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('kind',['integer','wrap','boolean','shared_integer'])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_typed_replay_original_and_all_vjps(engine,kind,delay,ranks,window):
    mpi(ranks);net,g,syn,dt,bundle,x=model(kind,True,ranks,window,delay,engine)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0])
    def ref(weights,initial=None,anchors=None):return oracle(bundle,g,syn,kind,delay,weights,initial,anchors)
    loss,z,spikes,anchors=ref(bundle.weights)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(ref(hi,anchors=anchors)[0]-ref(lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p_detached(bundle,index):
            assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(ref(bundle.weights,hi,anchors)[0]-ref(bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    net.run(4*dt,namespace={})
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=8e-5,atol=8e-6)
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['dynamic_state_layout'][syn.name]['h']],syn.h[:],rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def p_detached(bundle,index):
    return bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']


@pytest.mark.parametrize('kind',['integer','wrap','boolean','shared_integer'])
@pytest.mark.parametrize('discard',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_typed_replay_carry_and_restore(engine,kind,discard,ranks,tmp_path):
    mpi(ranks);_,_,_,_,bundle,x=model(kind,discard,ranks,None,1,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable'])
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0])
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);first=trainer.step(x[:,:2],[0]);path=tmp_path/'typed-replay';trainer.store(path)
    trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path);last=trainer.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
    assert trainer.clock_tick==4 and (last['gpu_dispatches']>0)==(engine!='cpu')
