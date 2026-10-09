"""Persistent captures in neuron integrators and full-array thresholds."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_mutable_captures import callback
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER
from brian2.codegen.translation import make_statements
from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2_rust.training_brian import _INTEGRATORS


def model(method,noisy,where,ranks,window,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    array=np.array([.113,.217]);f=b.Function(callback(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt='+('f(v)/ms' if where=='integrator' else '0/second')+('+.011*xi/sqrt(ms)' if noisy else '')+':1\ngain:1 (constant)',
        threshold='v>'+('f(gain)' if where=='threshold' else '.5'),reset='v-=.5',method=method,dt=dt,namespace={'f':f});g.v=[.173,.719];g.gain=[.137,.223]
    net=b.Network(source,hidden,g);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain']})
    variables={**b.core.functions.DEFAULT_FUNCTIONS,**g.variables,**g.resolve_all(g.equations.identifiers-set(g.variables),run_namespace={}),**g.namespace};variables={key:variables[key] for key in sorted(variables)}
    scalar,vector=make_statements(_INTEGRATORS[method](g.equations,variables=variables),variables,np.float64,optimise=True)
    gen=NumpyCodeGenerator(variables,g.variables.indices,g,{'_idx'},NumpyCodeObject,g.name,'stateupdate',allows_scalar_write=True)
    code=compile('\n'.join([*gen.translate_one_statement_sequence(scalar,scalar=True),*gen.translate_one_statement_sequence(vector)]),'<original neuron capture integrator>','exec')
    return net,g,array,dt,bundle,variables,gen,code,noisy,where


def oracle(data,weights,initial=None,anchors=None):
    net,g,array,dt,bundle,variables,gen,code,noisy,where=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    capture=bundle.provenance['mutable_capture_layout'][0]['cells'];v=bundle.provenance['neuron_state_layout'][g.name]['v']
    bank=next(entry['bank'] for entry in bundle.provenance['bindings'] if entry['object']==g.name and 'gain' in entry['variables']);gain=np.array(weights[bank]);before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());captured=z[capture].copy();function=callback(captured);env=dict(f=function,sqrt=np.sqrt,_numpy=np,_vectorisation_idx=np.arange(2))
        def randn(index):return np.array([normal(p['seed'],0,0,1,j,tick,0) for j in range(2)])
        env['randn']=randn
        for name,var in variables.items():
            if not isinstance(var,b.core.variables.ArrayVariable):
                if hasattr(var,'get_value') and not isinstance(var,b.core.variables.AuxiliaryVariable):env[name]=np.asarray(var.get_value()).reshape(-1)[0]
                continue
            key=gen.get_array_name(var)
            if key in env:continue
            value=z[v] if var is g.variables['v'] else gain if var is g.variables['gain'] else np.array([tick*float(dt)]) if name=='t' else var.get_value()
            env[key]=np.asarray(value).copy()
        exec(code,env);z[v]=env[gen.get_array_name(g.variables['v'])]
        threshold=function(gain.copy()) if where=='threshold' else .5;z[capture]=captured
        margin=z[v]-threshold
        if g.name in bundle.provenance.get('threshold_margin_layout',{}):z[bundle.provenance['threshold_margin_layout'][g.name]]=margin
        event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


CASES=[('euler',False,'integrator'),('rk2',False,'integrator'),('rk4',False,'integrator'),('euler',True,'integrator'),('heun',True,'integrator'),('milstein',True,'integrator'),('euler',False,'threshold')]
@pytest.mark.parametrize('method,noisy,where',CASES)
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_neuron_capture_original_all_vjps(engine,method,noisy,where,ranks,window,monkeypatch):
    mpi(ranks);data=model(method,noisy,where,ranks,window,engine);net,g,array,dt,bundle,*_=data;x=np.zeros((1,4,1))
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,spikes,anchors=oracle(data,bundle.weights)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(data,hi,anchors=anchors)[0]-oracle(data,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(oracle(data,bundle.weights,hi,anchors)[0]-oracle(data,bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    if noisy:
        draws=iter(np.array([normal(bundle.plan['seed'],0,0,1,j,tick,0) for j in range(2)]) for tick in range(4))
        def replay(*shape):assert shape==(2,);return next(draws).copy()
        monkeypatch.setattr(np.random,'randn',replay)
    net.run(4*dt,namespace={});capture=bundle.provenance['mutable_capture_layout'][0]['cells'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,capture],array,rtol=8e-5,atol=8e-6)
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,v],g.v[:],rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('method,noisy,where',CASES)
@pytest.mark.parametrize('ranks',[None,2])
def test_neuron_capture_restore(engine,method,noisy,where,ranks,tmp_path):
    mpi(ranks);data=model(method,noisy,where,ranks,None,engine);bundle=data[4];x=np.zeros((1,4,1));p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable'])
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:2],[0]);path=tmp_path/'neuron-capture';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])


@pytest.mark.parametrize('ranks',[None,2])
def test_capture_identity_shared_by_neuron_synapse_and_regular(engine,ranks,tmp_path):
    mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    array=np.array([.113,.217]);f=b.Function(callback(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=f(v)/ms:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,namespace={'f':f});g.v=[.173,.719]
    g.run_regularly('v=.9*f(v)',when='start')
    syn=b.Synapses(source,g,'dh/dt=f(h)/ms:1 (clock-driven)',on_pre='v_post+=h',namespace={'f':f},method='euler',dt=dt)
    syn.connect(i=[0,0],j=[0,1]);syn.h=[.137,.223];net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False)
    assert len(bundle.provenance['mutable_capture_layout'])==1;capture=bundle.provenance['mutable_capture_layout'][0]['cells']
    x=np.zeros((1,4,1));x[0,[0,2],0]=1;out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:2],[0]);path=tmp_path/'mixed-capture';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],out['final_state'],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),out['spikes'])
    np.testing.assert_array_equal(array,[.113,.217]);monitor=b.SpikeMonitor(g);net.add(monitor);net.run(4*dt,namespace={})
    expected=np.zeros((4,2));expected[np.rint(monitor.t/dt).astype(int),np.asarray(monitor.i)]=1;np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],expected)
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,capture],array,rtol=8e-5,atol=8e-6)
    for obj,key in [(g,'neuron_state_layout'),(syn,'dynamic_state_layout')]:
        for name,cells in bundle.provenance[key][obj.name].items():
            if name.startswith('__'):continue
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)


@pytest.mark.parametrize('ranks',[None,2])
def test_capture_identity_shared_by_two_neuron_layers(engine,ranks):
    mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    array=np.array([.113,.217]);f=b.Function(callback(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt);groups=[]
    for j in range(2):
        g=b.NeuronGroup(2,'dv/dt=f(v)/ms:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,namespace={'f':f},name='captured_layer_'+str(j));g.v=[.173+.1*j,.719-.1*j];groups.append(g)
    net=b.Network(source,*groups);bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,backend=engine,mpi_ranks=ranks,detach_reset=False)
    assert len(bundle.provenance['mutable_capture_layout'])==1;capture=bundle.provenance['mutable_capture_layout'][0]['cells'];out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,4,1)),[0]);net.run(4*dt,namespace={})
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,capture],array,rtol=8e-5,atol=8e-6)
    for g in groups:
        v=bundle.provenance['neuron_state_layout'][g.name]['v'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,v],g.v[:],rtol=8e-5,atol=8e-6)
    unused=bundle.provenance['inactive_alias_slots'];assert unused;np.testing.assert_array_equal(np.asarray(out['final_state'])[0,unused],0.)


@pytest.mark.parametrize('where',['integrator','threshold'])
@pytest.mark.parametrize('ranks',[None,2])
def test_capture_aliases_actual_neuron_storage(engine,where,ranks):
    mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt);hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt='+('f(v)/ms' if where=='integrator' else '0/second')+':1',threshold='v>'+('f(v)' if where=='threshold' else '.5'),reset='v-=.5',method='euler',dt=dt)
    g.v=[.173,.719];array=g.variables['v'].get_value();g.namespace['f']=b.Function(callback(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,detach_reset=False)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,4,1)),[0]);monitor=b.SpikeMonitor(g);net.add(monitor);net.run(4*dt,namespace={})
    expected=np.zeros((4,2));expected[np.rint(monitor.t/dt).astype(int),np.asarray(monitor.i)]=1;np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],expected)
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,v],g.v[:],rtol=8e-5,atol=8e-6)
    capture=bundle.provenance['mutable_capture_layout'][0]['cells'];assert capture==v
