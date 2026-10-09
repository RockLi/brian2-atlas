"""Continuous Synapses integrators with persistent mutable Python captures."""
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


def model(method,noisy,ranks,window,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    array=np.array([.113,.217]);f=b.Function(callback(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.173,.719]
    syn=b.Synapses(source,g,'dh/dt=f(h)/ms'+('+.011*xi/sqrt(ms)' if noisy else '')+':1 (clock-driven)\ngain:1 (constant)',on_pre='v_post+=h*gain',namespace={'f':f},method=method,dt=dt)
    syn.connect(i=[0,0],j=[0,1]);syn.h=[.137,.223];syn.gain=[.31,.41];net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,trainable_synapse_parameters={syn.name:['gain','h']})
    variables={**b.core.functions.DEFAULT_FUNCTIONS,**syn.variables,**syn.resolve_all(syn.equations.identifiers-set(syn.variables),run_namespace={}),**syn.namespace};variables={key:variables[key] for key in sorted(variables)}
    generated=_INTEGRATORS[method](syn.equations,variables=variables);scalar,vector=make_statements(generated,variables,np.float64,optimise=True)
    gen=NumpyCodeGenerator(variables,syn.variables.indices,syn,{'_idx'},NumpyCodeObject,syn.name,'stateupdate',allows_scalar_write=True)
    code=compile('\n'.join([*gen.translate_one_statement_sequence(scalar,scalar=True),*gen.translate_one_statement_sequence(vector)]),'<original captured Synapses integrator>','exec')
    x=np.zeros((1,4,1));x[0,[0,2],0]=1
    return net,g,syn,array,dt,bundle,x,variables,gen,code,noisy


def oracle(data,weights,initial=None,anchors=None):
    net,g,syn,array,dt,bundle,x,variables,gen,code,noisy=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for index,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[index]=weights[ref[0]][ref[1]]
    capture=bundle.provenance['mutable_capture_layout'][0]['cells'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h'];v=bundle.provenance['neuron_state_layout'][g.name]['v']
    gain_bank=next(entry['bank'] for entry in bundle.provenance['bindings'] if entry['object']==syn.name and 'gain' in entry['variables']);gain=np.array(weights[gain_bank]);before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());captured=z[capture].copy();env=dict(f=callback(captured),sqrt=np.sqrt,_numpy=np,_vectorisation_idx=np.arange(2))
        def randn(index):return np.array([normal(p['seed'],0,0,2,j,tick,0) for j in range(2)])
        env['randn']=randn
        for name,var in variables.items():
            if not isinstance(var,b.core.variables.ArrayVariable):
                if hasattr(var,'get_value') and not isinstance(var,b.core.variables.AuxiliaryVariable):env[name]=np.asarray(var.get_value()).reshape(-1)[0]
                continue
            key=gen.get_array_name(var)
            if key in env:continue
            value=z[h] if var is syn.variables['h'] else gain if var is syn.variables['gain'] else np.array([tick*float(dt)]) if name=='t' else var.get_value()
            env[key]=np.asarray(value).copy()
        exec(code,env);z[h]=env[gen.get_array_name(syn.variables['h'])];z[capture]=captured
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        if tick in (0,2):z[v]+=z[h]*gain
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('method,noisy',[('euler',False),('rk2',False),('rk4',False),('euler',True),('heun',True),('milstein',True)])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_synaptic_capture_integrator_all_vjps(engine,method,noisy,ranks,window,monkeypatch):
    mpi(ranks);data=model(method,noisy,ranks,window,engine);net,g,syn,array,dt,bundle,x,*_=data
    np.testing.assert_array_equal(array,[.113,.217]);out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,spikes,anchors=oracle(data,bundle.weights)
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
        draws=iter(np.array([normal(bundle.plan['seed'],0,0,2,j,tick,0) for j in range(2)]) for tick in range(4))
        def replay(*shape):assert shape==(2,);return next(draws).copy()
        monkeypatch.setattr(np.random,'randn',replay)
    net.run(4*dt,namespace={})
    capture=bundle.provenance['mutable_capture_layout'][0]['cells'];np.testing.assert_allclose(np.asarray(out['final_state'])[0,capture],array,rtol=8e-5,atol=8e-6)
    for obj,key in [(g,'neuron_state_layout'),(syn,'dynamic_state_layout')]:
        for name,cells in bundle.provenance[key][obj.name].items():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('method,noisy',[('euler',False),('rk2',False),('rk4',False),('euler',True),('heun',True),('milstein',True)])
@pytest.mark.parametrize('ranks',[None,2])
def test_synaptic_capture_integrator_restore(engine,method,noisy,ranks,tmp_path):
    mpi(ranks);data=model(method,noisy,ranks,None,engine);bundle=data[5];x=data[6];p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable'])
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:2],[0]);path=tmp_path/'capture-integrator';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
