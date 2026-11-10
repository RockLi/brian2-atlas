"""Brian dynamic conversion is checked against independent native/Brian models."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import lower_brian_dynamic_training, NativeLIFTrainer, TrainingConversionError
from test_native_training import RUNNER
from test_training_dynamic import model, noisy_model


def network(event_driven=False,noisy=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    p,w,x=noisy_model() if noisy else model(event_driven=event_driven)
    dt=.2*b.ms;ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='front_input')
    layers=[b.NeuronGroup(2,'dv/dt=-v/tau:1',threshold='v>1',reset='v-=1',method='euler',dt=dt,
                         namespace={'tau':float(dt)/.15*b.second},name=f'front_layer_{l}') for l in range(2)]
    layers[0].v=p['dynamic']['initial'][:2];layers[1].v=p['dynamic']['initial'][2:4]
    static=b.Synapses(inp,layers[0],'w:1',on_pre='v_post+=w',name='front_static');static.connect();static.w=w[0]
    flag='event-driven' if event_driven else 'clock-driven';noise='+sigma*xi/sqrt(ms)' if noisy else ''
    syn=b.Synapses(*layers,f'''w:1
        dapre/dt=-apre/taup{noise}:1 ({flag})
        dapost/dt=-apost/taum:1 ({flag})
        taup:second (shared,constant)
        taum:second (shared,constant)
        Ap:1 (shared,constant)
        Am:1 (shared,constant)
        sigma:1 (shared,constant)''',
        on_pre='v_post+=w\napre+=Ap\nw=clip(w+apost,0,1.2)',on_post='apost+=Am\nw=clip(w+apre,0,1.2)',
        dt=dt,method='euler' if noisy else 'exact',name='front_stdp')
    syn.connect();syn.w=w[1];syn.apre=p['dynamic']['initial'][8:12];syn.apost=p['dynamic']['initial'][12:16]
    syn.taup=w[2][0]*b.second;syn.taum=w[2][1]*b.second;syn.Ap=w[2][2];syn.Am=w[2][3];syn.sigma=w[3][0] if noisy else 0
    return b.Network(inp,*layers,static,syn),inp,layers,static,syn,p,w,x


@pytest.mark.parametrize('event_driven,noisy',[(False,False),(True,False),(False,True)])
@pytest.mark.parametrize('detach,window',[(True,None),(False,None),(True,3),(False,3)])
def test_frontend_dynamic_forward_and_vjp(event_driven,noisy,detach,window):
    net,inp,layers,static,syn,p,w,x=network(event_driven,noisy)
    p['detach_reset']=detach;p['tbptt_window']=window
    for action in p['dynamic']['actions'][-4:]:action['detach_trigger']=detach
    if noisy:
        # The frontend reserves one domain per Synapses object in name order.
        for action in p['dynamic']['actions']:
            if action.get('noise_streams'):action['noise_domain']=3
    before={name:np.asarray(syn.variables[name].get_value()).copy() for name in ('w','apre','apost')}
    requested={syn.name:['w','taup','taum','Ap','Am']+(['sigma'] if noisy else [])}
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,trainable_synapse_parameters=requested,
                                       detach_reset=detach,tbptt_window=window,learning_rate=1e-6)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    expected=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x[None],[0])
    np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
    np.testing.assert_allclose(actual['final_membrane'],expected['final_membrane'],atol=5e-14,rtol=5e-14)
    np.testing.assert_allclose(actual['initial_gradients'],expected['initial_gradients'],atol=1e-12,rtol=1e-12)
    for binding in bundle.provenance['bindings']:
        bank=binding['bank'];name=binding['variables'][0]
        if binding['object']==static.name:expected_grad=expected['gradients'][0]
        elif name=='w':expected_grad=expected['gradients'][1]
        elif name=='sigma':expected_grad=expected['gradients'][3]
        else:expected_grad=[expected['gradients'][2][['taup','taum','Ap','Am'].index(name)]]
        np.testing.assert_allclose(actual['gradients'][bank],expected_grad,atol=2e-11,rtol=2e-11)
    layout=bundle.provenance['dynamic_state_layout'][syn.name]
    for name,start in [('w',4),('apre',8),('apost',12)]+([('lastupdate',16)] if event_driven else []):
        np.testing.assert_allclose(np.array(actual['final_state'])[0,layout[name]],expected['final_state'][0][start:start+4],atol=5e-14,rtol=5e-14)
    for name,value in before.items():np.testing.assert_array_equal(syn.variables[name].get_value(),value)
    assert float(inp.clock.variables['t'].get_value()[0])==0


@pytest.mark.parametrize('ranks',[None,2,8])
def test_frontend_dynamic_frozen_split_restore(ranks,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    net,inp,layers,static,syn,_,_,x=network(True)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,mpi_ranks=ranks,
                                       trainable_synapse_parameters={static.name:[],syn.name:[]})
    assert not any(bundle.plan['trainable'])
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    whole=trainer.evaluate(x[None],[0]);trainer.execute(x[None,:5],[0])
    filename=tmp_path/'dynamic.json';trainer.store(filename)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(filename)
    tail=restored.execute(x[None,5:],[0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],atol=1e-14,rtol=1e-14)
    np.testing.assert_array_equal(tail['spikes'],np.array(whole['spikes'])[:,5:])
    assert tail['final_tick']==len(x)


@pytest.mark.parametrize('post_first',[False,True])
def test_frontend_preserves_brian_path_schedule(post_first):
    net,inp,layers,static,syn,_,_,x=network(True)
    if post_first:syn.post.order=-2
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in layers];net.add(*monitors)
    net.run(len(x)*.2*b.ms,namespace={})
    spikes=np.zeros((len(x),4))
    for layer,monitor in enumerate(monitors):
        spikes[np.rint(np.asarray(monitor.t/b.second)/.0002).astype(int),2*layer+np.asarray(monitor.i)]=1
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    for name,cells in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(np.asarray(actual['final_state'])[0,cells],syn.variables[name].get_value(),atol=5e-14,rtol=5e-14)


@pytest.mark.parametrize('issue',['delay','inactive','reserved','unknown_parameter','mask_missing_layout'])
def test_frontend_rejects_unimplemented_behavior_transactionally(issue):
    net,inp,layers,static,syn,_,_,x=network()
    options={}
    if issue=='delay':syn.pre.delay=-.2*b.ms
    elif issue=='inactive':syn.pre.active=False
    elif issue=='reserved':options['_defer_synapses']=True
    elif issue=='unknown_parameter':options['trainable_synapse_parameters']={syn.name:['missing']}
    if issue!='mask_missing_layout':
        with pytest.raises(TrainingConversionError):lower_brian_dynamic_training(net,input_group=inp,layers=layers,**options)
        return
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
    bundle.plan['dynamic'].pop('migration')
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.execute(x[None],[0])
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.plan));masks=copy.deepcopy(trainer.plan['masks']);masks[-1][0]=0
    with pytest.raises(ValueError,match='migration'):trainer.update_mask(masks)
    assert before==(trainer.state,trainer.neuron_state,trainer.plan)


@pytest.mark.parametrize('method',['euler','rk2','rk4','exact'])
def test_short_term_plasticity_subexpressions_time_and_neuron_reads(method):
    net,inp,layers,static,old,_,_,x=network()
    net.remove(old)
    syn=b.Synapses(*layers,'''w:1
        du/dt=(U-u)/tauf:1 (clock-driven)
        dr/dt=(1-r)/taur:1 (clock-driven)
        released=w*u*r:1
        U:1 (shared,constant)''',
        on_pre='u+=U*(1-u)\nv_post+=released*(1+.1*v_pre)+.02*sin(t/ms)\nr*=1-u',
        dt=.2*b.ms,method=method,namespace={'tauf':2*b.ms,'taur':3*b.ms},name='stp')
    syn.connect(i=[1,0,1,0],j=[0,1,1,0]);syn.w=[.7,.8,.9,1.];syn.u=[.1,.2,.3,.4];syn.r=.8;syn.U=.25
    net.add(syn)
    from brian2_rust import lower_brian_training
    bundle=lower_brian_training(net,input_group=inp,layers=layers,dynamic=True,
                               trainable_synapse_parameters={syn.name:['w','U']})
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in layers];net.add(*monitors);net.run(len(x)*.2*b.ms,namespace={})
    spikes=np.zeros((len(x),4))
    for layer,monitor in enumerate(monitors):spikes[np.rint(np.asarray(monitor.t/b.second)/.0002).astype(int),2*layer+np.asarray(monitor.i)]=1
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    np.testing.assert_allclose(actual['final_membrane'][0],np.r_[layers[0].v[:],layers[1].v[:]],rtol=5e-14,atol=5e-14)
    for name,cells in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(np.asarray(actual['final_state'])[0,cells],syn.variables[name].get_value(),rtol=5e-14,atol=5e-14)


def test_frontend_mask_disables_all_plastic_edge_actions():
    net,inp,layers,static,syn,_,_,x=network(True)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
    bank=next(binding['bank'] for binding in bundle.provenance['bindings'] if binding['object']==syn.name and binding['variables']==['w'])
    bundle.plan['masks'][bank][1]=0
    bundle.weights[bank][1]=0
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    out=trainer.gradients(x[None],[0])
    for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
        index=indices[1]
        assert out['final_state'][0][index]==(0 if name=='w' else bundle.initial_state[index])
    assert out['gradients'][bank][1]==0


def test_neuron_parameter_aliases_keep_indexing_and_optimizer_bindings():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(2,[0,1,0,1],np.arange(4)*dt,dt=dt)
    layers=[b.NeuronGroup(2,'dv/dt=-v/(2*ms)+gain/ms:1\ngain:1 (constant)',threshold='v>1',reset='v-=1',method='euler',dt=dt) for _ in range(2)]
    layers[0].gain=[.3,.6];layers[1].gain=[.8,.4];layers[0].v=[.9,1.3]
    a=b.Synapses(inp,layers[0],'w:1',on_pre='v_post+=w');a.connect();a.w=.5
    s=b.Synapses(*layers,'w:1',on_pre='v_post+=w*gain+.02*gain_pre');s.connect(i=[1,0,1,0],j=[0,1,1,0]);s.w=[.7,.8,.9,1.]
    net=b.Network(inp,*layers,a,s);chosen={g.name:['gain'] for g in layers}
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,trainable_neuron_parameters=chosen)
    s.pre.code='v_post+=w*gain_post+.02*gain_pre'
    explicit=lower_brian_dynamic_training(net,input_group=inp,layers=layers,trainable_neuron_parameters=chosen)
    assert bundle.plan==explicit.plan
    x=np.zeros((1,12,2));x[0,np.arange(4),[0,1,0,1]]=1
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x,[0])
    net.run(12*dt,namespace={})
    np.testing.assert_allclose(actual['final_membrane'][0],np.r_[layers[0].v[:],layers[1].v[:]],rtol=5e-14,atol=5e-14)
    for binding in bundle.provenance['bindings']:
        if binding['kind']=='neuron_array':assert all(v!=0 for v in actual['gradients'][binding['bank']])
