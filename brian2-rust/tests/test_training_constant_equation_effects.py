"""Borrowed constant mutation in generated integrators and threshold arrays."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(method,where,ranks,window,backend):
    from brian2.codegen.translation import make_statements
    from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
    from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
    from brian2_rust.training_brian import _INTEGRATORS
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    code='dv/dt='+('curve(gain)/ms' if where=='integrator' else '0/second')+':1\ngain:1 (constant)'
    g=b.NeuronGroup(2,code,threshold='v>'+('curve(gain)' if where=='threshold' else '.5'),reset='v-=.5',method=method,dt=dt,
                    namespace={'curve':b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)})
    g.v=[.173,.719];g.gain=[.113,.217];net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,
                                       trainable_neuron_parameters={g.name:['gain']})
    assert 'gain' in bundle.provenance['mutable_constant_layout'][g.name]
    variables={**b.core.functions.DEFAULT_FUNCTIONS,**g.variables,**g.resolve_all(g.equations.identifiers-set(g.variables),run_namespace={}),**g.namespace}
    variables={name:variables[name] for name in sorted(variables)}
    generated=_INTEGRATORS[method](g.equations,variables=variables)
    scalar,vector=make_statements(generated,variables,np.float64,optimise=True)
    gen=NumpyCodeGenerator(variables,g.variables.indices,g,{'_idx'},NumpyCodeObject,g.name,'stateupdate',allows_scalar_write=True)
    code='\n'.join([*gen.translate_one_statement_sequence(scalar,scalar=True),*gen.translate_one_statement_sequence(vector)])
    return net,g,dt,bundle,variables,gen,compile(code,'<actual original NumPy integrator>','exec')


def oracle(bundle,g,where,variables,gen,code,weights,initial=None,anchors=None):
    p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for index,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[index]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];gain=layout['gain'];before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());env=dict(curve=curve,_numpy=np,_vectorisation_idx=np.arange(2))
        for name,var in variables.items():
            if not isinstance(var,b.core.variables.ArrayVariable):
                if hasattr(var,'get_value'):env[name]=np.asarray(var.get_value()).reshape(-1)[0]
                continue
            key=gen.get_array_name(var)
            if key not in env:env[key]=z[v].copy() if var is g.variables['v'] else z[gain].copy() if var is g.variables['gain'] else np.asarray(var.get_value()).copy()
        exec(code,env);z[v]=env[gen.get_array_name(g.variables['v'])];z[gain]=env[gen.get_array_name(g.variables['gain'])]
        if where=='threshold':z[gain]*=.8;threshold=z[gain].copy()
        else:threshold=.5
        margin=z[v]-threshold;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('where',['integrator','threshold'])
@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_constant_equation_original_and_all_vjps(engine,where,method,ranks,window):
    mpi(ranks);net,g,dt,bundle,variables,gen,code=model(method,where,ranks,window,engine)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
    def ref(weights,initial=None,anchors=None):return oracle(bundle,g,where,variables,gen,code,weights,initial,anchors)
    loss,z,spikes,anchors=ref(bundle.weights);layout=bundle.provenance['neuron_state_layout'][g.name]
    for cells in layout.values():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(ref(hi,anchors=anchors)[0]-ref(lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(ref(bundle.weights,hi,anchors)[0]-ref(bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    net.run(4*dt,namespace={})
    for field,cells in layout.items():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('where',['integrator','threshold'])
@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('ranks',[None,2])
def test_constant_equation_carry_and_restore(engine,where,method,ranks,tmp_path):
    mpi(ranks);net,g,dt,bundle,_,_,_=model(method,where,ranks,None,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable']);x=np.zeros((1,4,1))
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    first=t.step(x[:,:2],[0]);path=tmp_path/'equation';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
    net.run(4*dt,namespace={})
    for field,cells in bundle.provenance['neuron_state_layout'][g.name].items():np.testing.assert_allclose(np.asarray(last['final_state'])[0,cells],g.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert t.clock_tick==4 and (last['gpu_dispatches']>0)==(engine!='cpu')


def synapse_model(method,ranks,window,backend):
    from brian2.codegen.translation import make_statements
    from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
    from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
    from brian2_rust.training_brian import _INTEGRATORS
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.173,.719]
    syn=b.Synapses(source,g,'dh/dt=curve(gain)/ms:1 (clock-driven)\ngain:1 (constant)',on_pre='v_post+=h*gain',dt=dt,method=method,
                   namespace={'curve':b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)})
    syn.connect(i=[0,0],j=[0,1]);syn.h=[.137,.223];syn.gain=[.113,.217];net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,
                                       detach_reset=False,trainable_synapse_parameters={syn.name:['gain','h']})
    assert 'gain' in bundle.provenance['mutable_constant_layout'][syn.name]
    variables={**b.core.functions.DEFAULT_FUNCTIONS,**syn.variables,**syn.resolve_all(syn.equations.identifiers-set(syn.variables),run_namespace={}),**syn.namespace}
    variables={name:variables[name] for name in sorted(variables)};generated=_INTEGRATORS[method](syn.equations,variables=variables)
    scalar,vector=make_statements(generated,variables,np.float64,optimise=True)
    gen=NumpyCodeGenerator(variables,syn.variables.indices,syn,{'_idx'},NumpyCodeObject,syn.name,'stateupdate',allows_scalar_write=True)
    code='\n'.join([*gen.translate_one_statement_sequence(scalar,scalar=True),*gen.translate_one_statement_sequence(vector)])
    x=np.zeros((1,4,1));x[0,[0,2],0]=1
    return net,g,syn,dt,bundle,x,variables,gen,compile(code,'<original Synapses NumPy integrator>','exec')


def synapse_oracle(bundle,g,syn,variables,gen,code,weights,initial=None,anchors=None):
    p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for index,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[index]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];layout=bundle.provenance['dynamic_state_layout'][syn.name];h=layout['h'];gain=layout['gain']
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());env=dict(curve=curve,_numpy=np,_vectorisation_idx=np.arange(2))
        for name,var in variables.items():
            if not isinstance(var,b.core.variables.ArrayVariable):
                if hasattr(var,'get_value'):env[name]=np.asarray(var.get_value()).reshape(-1)[0]
                continue
            key=gen.get_array_name(var)
            if key not in env:env[key]=z[h].copy() if var is syn.variables['h'] else z[gain].copy() if var is syn.variables['gain'] else np.asarray(var.get_value()).copy()
        exec(code,env);z[h]=env[gen.get_array_name(syn.variables['h'])];z[gain]=env[gen.get_array_name(syn.variables['gain'])]
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        if tick in (0,2):z[v]+=z[h]*z[gain]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_synaptic_constant_integrator_all_vjps(engine,method,ranks,window):
    mpi(ranks);net,g,syn,dt,bundle,x,variables,gen,code=synapse_model(method,ranks,window,engine)
    def ref(weights,initial=None,anchors=None):return synapse_oracle(bundle,g,syn,variables,gen,code,weights,initial,anchors)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,spikes,anchors=ref(bundle.weights)
    layouts=[bundle.provenance['neuron_state_layout'][g.name],bundle.provenance['dynamic_state_layout'][syn.name]]
    for layout in layouts:
        for cells in layout.values():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(ref(hi,anchors=anchors)[0]-ref(lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(ref(bundle.weights,hi,anchors)[0]-ref(bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    net.run(4*dt,namespace={})
    for obj,layout in zip([g,syn],layouts):
        for field,cells in layout.items():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('method',['euler','rk2','rk4','heun','milstein'])
@pytest.mark.parametrize('ranks',[None,2])
def test_synaptic_constant_integrator_carry_and_restore(engine,method,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x,_,_,_=synapse_model(method,ranks,None,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable']);whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0])
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:2],[0]);path=tmp_path/'synconst';t.store(path)
    t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])
    net.run(4*dt,namespace={})
    for field,cells in bundle.provenance['dynamic_state_layout'][syn.name].items():np.testing.assert_allclose(np.asarray(last['final_state'])[0,cells],syn.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert t.clock_tick==4 and (last['gpu_dispatches']>0)==(engine!='cpu')
