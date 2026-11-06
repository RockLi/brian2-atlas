"""Discrete Brian storage: real Cython parity and native device continuations."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer
from brian2_rust.training_brian_dynamic import lower_brian_dynamic_training
from brian2_rust.training_dynamic import compile_dynamic_transform
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache


def network(method='euler',noisy=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.5*b.ms
    inp=b.SpikeGeneratorGroup(1,[0,0,0],[0,1,3]*dt,dt=dt)
    h=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>1',reset='v=0',dt=dt,method=method)
    g=b.NeuronGroup(2,'dv/dt=int(enabled)/ms'+('+.1*int(enabled)*xi/sqrt(ms)' if noisy else '')+':1\nenabled:boolean\ncount:integer',
        threshold='v>.5 and enabled',reset='v-=.5\ncount+=1\nenabled=not enabled',dt=dt,method=method)
    g.v=[.1,.3];g.enabled=True
    s=b.Synapses(inp,g,'w:1\nquota:integer\nready:boolean',
        on_pre='v_post+=w*int(ready)\nquota-=1\nready=quota>0\nenabled_post=not enabled_post',dt=dt)
    s.connect();s.w=[.2,.4];s.quota=[2,1];s.ready=True
    trace=b.StateMonitor(g,['v','enabled','count'],record=True,when='end')
    edges=b.StateMonitor(s,['quota','ready'],record=True,when='end');spikes=b.SpikeMonitor(g)
    net=b.Network(inp,h,g,s,trace,edges,spikes)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[h,g],**options)
    return net,g,s,trace,edges,spikes,bundle,np.array([[[1],[1],[0],[1]]],float)


@pytest.mark.parametrize('method',['euler','rk4'])
def test_integer_boolean_frontend_matches_real_cython(method):
    net,g,s,trace,edges,spikes,bundle,x=network(method)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    live=[];actual=[]
    for tick in range(4):
        result=trainer.execute(x[:,tick:tick+1],[0],operation='train',initial='carry' if tick else None)
        live.append(result['final_state'][0]);actual.append(result['spikes'][0][0][1:])
    net.run(4*.5*b.ms);live=np.array(live)
    for obj,monitor,layout,names in [(g,trace,'neuron_state_layout',['v','enabled','count']),(s,edges,'dynamic_state_layout',['quota','ready'])]:
        for name in names:
            cells=bundle.provenance[layout][obj.name][name]
            np.testing.assert_allclose(live[:,cells].T,np.asarray(getattr(monitor,name)),atol=2e-14,rtol=0)
    expected=np.zeros((4,2));expected[np.rint(spikes.t/(.5*b.ms)).astype(int),spikes.i[:]]=1
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('noisy',[False,True])
def test_typed_events_device_gradient_and_carry(engine,ranks,window,noisy,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,g,s,trace,edges,spikes,bundle,x=network(method='heun' if noisy else 'euler',noisy=noisy,backend=engine,mpi_ranks=ranks,tbptt_window=window)
    # Compare the emitted plan to CPU, including derivatives through event
    # triggers and float weights; integer and Boolean gradients remain zero.
    p=copy.deepcopy(bundle.plan);p['backend']='cpu';p['mpi_ranks']=None
    reference=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients(x,[0])
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x,[0])
    for key in ['final_state','spikes','initial_state_gradients','logits']:
        np.testing.assert_allclose(actual[key],reference[key],rtol=3e-5,atol=3e-6)
    for a,c in zip(actual['gradients'],reference['gradients']):np.testing.assert_allclose(a,c,rtol=4e-4,atol=5e-6)
    discrete=bundle.plan['dynamic']['integer_states']+[k for k in bundle.plan['dynamic']['binary_states'] if bundle.plan['dynamic']['detached'][k]]
    np.testing.assert_array_equal(np.array(actual['initial_state_gradients'])[:,discrete],0)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(x[:,:2],[0])
    path=tmp_path/'typed.json';trainer.store(path)
    other=NativeLIFTrainer(bundle.plan,runner=RUNNER);other.restore(path)
    tail=other.evaluate(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],reference['final_state'],rtol=3e-5,atol=3e-6)


def test_discrete_synaptic_state_migration_restarts_declared_values():
    *_,bundle,x=network();bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(x[:,:1],[0])
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['variables']==['w'])
    masks=copy.deepcopy(trainer.plan['masks']);masks[bank][0]=0;trainer.update_mask(masks)
    layout=next(iter(bundle.provenance['dynamic_state_layout'].values()))
    for name in ('quota','ready'):assert trainer.neuron_state[0][layout[name][0]]==0
    masks[bank][0]=1;trainer.update_mask(masks,growth_weight=.2)
    assert trainer.neuron_state[0][layout['quota'][0]]==2
    assert trainer.neuron_state[0][layout['ready'][0]]==1


def test_assignment_cast_is_visible_to_next_statement(engine):
    *_,bundle,x=network(backend=engine)
    # Replace one action with a sequential transform; count=int(1.8) must
    # already be 1 when the following float output reads it.
    transform=compile_dynamic_transform('count=value\nv=count+0.25\nenabled=not enabled',
        states={'v':0,'count':1,'enabled':2},parameters={'value':1.8},state_types={0:'float',1:'integer',2:'boolean'})
    d=bundle.plan['dynamic'];layout=list(bundle.provenance['neuron_state_layout'].values())[1]
    program_set=len(d['program_sets']);d['program_sets'].append(transform['programs'])
    d['actions'].append(dict(owner=1,reads=[layout[n][0] for n in ('v','count','enabled')],writes=[layout[n][0] for n in ('v','count','enabled')],program_set=program_set,threshold=None,trigger=None))
    r=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[:,:1],[0])
    for name,value in [('v',1.25),('count',1),('enabled',0)]:assert r['final_state'][0][layout[name][0]]==value


@pytest.mark.parametrize('constant',[False,True])
def test_integer_and_boolean_links_preserve_physical_storage(engine,constant):
    b.set_device('runtime');b.start_scope();dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1,[0],[0]*dt,dt=dt)
    flag='constant' if constant else ''
    a=b.NeuronGroup(2,'dv/dt=0/second:1\ncount:integer'+(' (constant)' if constant else '')+'\nenabled:boolean'+(' (constant)' if constant else ''),
        threshold='v>1',reset='v=0',dt=dt,method='euler');a.count=[2147483647,-2147483648];a.enabled=[True,False]
    c=b.NeuronGroup(2,'dv/dt=0/second:1\npeer:integer (linked)\nactive:boolean (linked)',
        threshold='peer>0 and active',reset='v=0',dt=dt,method='euler')
    c.peer=b.linked_var(a,'count',index=[1,0]);c.active=b.linked_var(a,'enabled',index=[1,0])
    s=b.Synapses(inp,c,'w:1\npeer_syn:integer (linked)\nactive_syn:boolean (linked)\npick:integer (constant)',
        on_pre='v_post+=w*int(active_syn)'+('' if constant else '\npeer_syn+=1\nactive_syn=not active_syn'),dt=dt)
    s.connect();s.pick=[0,1];s.peer_syn=b.linked_var(a,'count',index='pick');s.active_syn=b.linked_var(a,'enabled',index='pick');s.w=.2
    bundle=lower_brian_dynamic_training(b.Network(inp,a,c,s),input_group=inp,layers=[a,c],backend=engine)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients([[[1]]],[0])
    np.testing.assert_array_equal(result['spikes'][0][0],[0,0,0,1])
    np.testing.assert_allclose(result['final_membrane'][0],[0,0,.2,0],atol=1e-7)
    if not constant:
        layout=bundle.provenance['neuron_state_layout'][a.name]
        np.testing.assert_array_equal(np.array(result['final_state'])[0,layout['count']],[-2147483648,-2147483647])
        np.testing.assert_array_equal(np.array(result['final_state'])[0,layout['enabled']],[0,1])
    assert all(all(v==0 for v in result['gradients'][bank]) for bank,_ in bundle.plan['dynamic']['integer_parameters'])


def test_cython_int32_overflow_and_sequential_float_assignment():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1,[],[]*dt,dt=dt)
    h=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>1',reset='v=0',dt=dt,method='euler')
    g=b.NeuronGroup(4,'dv/dt=0/second:1\ncount:integer\ncopy:integer',threshold='v>-100',reset='count+=1\ncopy=int(v)\nv=copy+.25',dt=dt,method='euler')
    g.count=[2147483647,-2147483648,16777216,-1];g.v=[1.8,-1.8,.8,-.8]
    net=b.Network(inp,h,g);bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[h,g])
    r=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate([[[0]]],[0]);net.run(dt)
    assert g.resetter['spike'].codeobj.compiled_code['run'] is not None
    for name,indices in bundle.provenance['neuron_state_layout'][g.name].items():
        np.testing.assert_array_equal(np.array(r['final_state'])[0,indices],np.asarray(getattr(g,name)[:]))
