"""Borrowed declared constants become persistent state with initial bank VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import curve
from test_training_typed_callback_effects import increment,fill
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(kind,ranks,window,backend,trainable=True):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    dt=.2*b.ms;inp=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=('+('.1*gain' if kind in ('consumer','noise') else '0')+')/ms'+('+.011*xi/sqrt(ms)' if kind=='noise' else '')+':1\ngain:1 (constant)\nk:integer (constant)\nflag:boolean (constant)',
                    threshold='v>'+('gain' if kind=='threshold' else '.5'),reset='v-=.5',method='euler',dt=dt)
    g.v=[.173,.719];g.gain=[.113,.217];g.k=[2147483647,-2147483648];g.flag=[True,False]
    for name,cb in [('curve',curve),('change',increment),('fill',fill)]:g.namespace[name]=b.Function(cb,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    code='tmp=curve(gain);v+=.1*tmp'
    if kind=='integer':code='tmp=change(k);v+=1e-10*tmp+.1*gain'
    elif kind=='boolean':code='tmp=fill(flag);v+=.1*tmp*gain'
    elif kind=='private':code='tmp=curve(gain+.1);v+=.1*tmp'
    g.run_regularly(code,when='start');net=b.Network(inp,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,
                                       detach_reset=False,trainable_neuron_parameters={g.name:['gain']} if trainable else {})
    promoted=bundle.provenance['mutable_constant_layout'][g.name]
    assert ('gain' in promoted)==(kind in ['gain','consumer','threshold','noise'])
    assert ('k' in promoted)==(kind=='integer') and ('flag' in promoted)==(kind=='boolean')
    return net,g,dt,bundle


def oracle(bundle,g,kind,weights,initial=None,anchors=None):
    p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for i,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[i]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];promoted=bundle.provenance['mutable_constant_layout'][g.name]
    bank=next((r['bank'] for r in bundle.provenance['bindings'] if r['object']==g.name and r['variables']==['gain']),None)
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());gain=z[promoted['gain']].copy() if 'gain' in promoted else np.array(weights[bank]) if bank is not None else np.asarray(g.gain[:])
        if kind=='integer':
            value=z[promoted['k']].astype(np.int32);value+=1;z[promoted['k']]=value;z[layout['v']]+=1e-10*value+.1*gain
        elif kind=='boolean':z[promoted['flag']]=1.;z[layout['v']]+=.1*gain
        elif kind=='private':z[layout['v']]+=.08*(gain+.1)
        else:gain*=.8;z[promoted['gain']]=gain;z[layout['v']]+=.1*gain
        if kind in ('consumer','noise'):z[layout['v']]+=.02*gain
        if kind=='noise':
            from test_training_stochastic import normal
            z[layout['v']]+=.011*np.sqrt(.2)*np.array([normal(p['seed'],0,0,1,j,tick,0) for j in range(2)])
        margin=z[layout['v']]-(gain if kind=='threshold' else .5);event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());z[layout['v']]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('kind',['gain','consumer','threshold','integer','boolean','private','noise'])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_mutable_constant_original_and_all_vjps(engine,kind,ranks,window,monkeypatch):
    mpi(ranks);net,g,dt,bundle=model(kind,ranks,window,engine)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
    loss,z,spikes,anchors=oracle(bundle,g,kind,bundle.weights)
    layout=bundle.provenance['neuron_state_layout'][g.name]
    for field,cells in layout.items():
        if np.dtype(g.variables[field].dtype).kind in 'ib':np.testing.assert_array_equal(np.asarray(out['final_state'])[0,cells],z[cells])
        else:np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        if not bundle.plan['trainable'][bank]:continue
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(bundle,g,kind,hi,anchors=anchors)[0]-oracle(bundle,g,kind,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert out['initial_state_gradients'][0][index]==0.;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(oracle(bundle,g,kind,bundle.weights,hi,anchors)[0]-oracle(bundle,g,kind,bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    if kind=='noise':
        from test_training_stochastic import normal
        draws=iter(np.array([normal(bundle.plan['seed'],0,0,1,j,tick,0) for j in range(2)]) for tick in range(4))
        def replay(*shape):
            assert shape==(2,);return next(draws).copy()
        monkeypatch.setattr(np.random,'randn',replay)
    net.run(4*dt,namespace={})
    for field,cells in layout.items():
        if np.dtype(g.variables[field].dtype).kind in 'ib':np.testing.assert_array_equal(np.asarray(out['final_state'])[0,cells],g.variables[field].get_value())
        else:np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    for field,cells in bundle.provenance['mutable_constant_layout'][g.name].items():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('kind',['gain','consumer','threshold','integer','boolean','private','noise'])
@pytest.mark.parametrize('trainable',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_mutable_constant_carry_and_restore(engine,kind,trainable,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,dt,bundle=model(kind,ranks,None,engine,trainable)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable']);x=np.zeros((1,4,1))
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0])
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:2],[0]);path=tmp_path/'constant';t.store(path)
    t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
    if kind=='noise':
        from test_training_stochastic import normal
        draws=iter(np.array([normal(bundle.plan['seed'],0,0,1,j,tick,0) for j in range(2)]) for tick in range(4))
        def replay(*shape):
            assert shape==(2,);return next(draws).copy()
        monkeypatch.setattr(np.random,'randn',replay)
    net.run(4*dt,namespace={})
    for field,cells in bundle.provenance['mutable_constant_layout'][g.name].items():np.testing.assert_allclose(np.asarray(last['final_state'])[0,cells],g.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert t.clock_tick==4 and (last['gpu_dispatches']>0)==(engine!='cpu')


def synapse_model(ranks,window,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.173,.719]
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre='v_post+=h*gain',dt=dt,
                   namespace={'curve':b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)})
    syn.connect(i=[0,0],j=[0,1]);syn.h=[.137,.223];syn.gain=[.113,.217]
    syn.run_regularly('tmp=curve(gain);h+=.1*tmp',when='start');net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,
                                       detach_reset=False,trainable_synapse_parameters={syn.name:['gain','h']})
    assert 'gain' in bundle.provenance['mutable_constant_layout'][syn.name]
    x=np.zeros((1,4,1));x[0,[0,2],0]=1
    return net,g,syn,dt,bundle,x


def synapse_oracle(bundle,g,syn,weights,initial=None,anchors=None):
    p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for i,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[i]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];layout=bundle.provenance['dynamic_state_layout'][syn.name]
    h=layout['h'];gain=layout['gain'];before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[gain]*=.8;z[h]+=.1*z[gain]
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        if tick in (0,2):z[v]+=z[h]*z[gain]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_synaptic_mutable_constant_all_vjps(engine,ranks,window):
    mpi(ranks);net,g,syn,dt,bundle,x=synapse_model(ranks,window,engine)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,spikes,anchors=synapse_oracle(bundle,g,syn,bundle.weights)
    layouts=[bundle.provenance['neuron_state_layout'][g.name],bundle.provenance['dynamic_state_layout'][syn.name]]
    for layout in layouts:
        for cells in layout.values():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(synapse_oracle(bundle,g,syn,hi,anchors=anchors)[0]-synapse_oracle(bundle,g,syn,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(synapse_oracle(bundle,g,syn,bundle.weights,hi,anchors)[0]-synapse_oracle(bundle,g,syn,bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    net.run(4*dt,namespace={})
    for obj,layout in zip([g,syn],layouts):
        for field,cells in layout.items():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('ranks',[None,2])
def test_synaptic_mutable_constant_carry_and_restore(engine,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=synapse_model(ranks,None,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable'])
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    first=t.step(x[:,:2],[0]);path=tmp_path/'synconst';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    last=t.step(x[:,2:],[0],initial='carry');np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
    net.run(4*dt,namespace={})
    for field,cells in bundle.provenance['dynamic_state_layout'][syn.name].items():np.testing.assert_allclose(np.asarray(last['final_state'])[0,cells],syn.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert t.clock_tick==4 and (last['gpu_dispatches']>0)==(engine!='cpu')
