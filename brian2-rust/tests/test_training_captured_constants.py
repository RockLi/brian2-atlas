"""Captured declared constants promote canonical state and retain bank VJPs."""
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


def model(where,method,noisy,ranks,window,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms;synaptic=where=='synapse'
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    drift='(f(v)+gain)/ms' if where=='integrator' else '0/second'
    g=b.NeuronGroup(2,'dv/dt='+drift+('+.011*xi/sqrt(ms)' if noisy and not synaptic else '')+':1'+('' if synaptic else '\ngain:1 (constant)'),
        threshold='v>'+('f(gain)' if where=='threshold' else '.5'),reset='v-=.5',method=method,dt=dt);g.v=[.173,.719];objects=[source,hidden,g]
    if synaptic:
        owner=b.Synapses(source,g,'dh/dt=(f(h)+gain)/ms'+('+.011*xi/sqrt(ms)' if noisy else '')+':1 (clock-driven)\ngain:1 (constant)',on_pre='v_post+=h*gain',method=method,dt=dt)
        owner.connect(i=[0,0],j=[0,1]);owner.h=[.137,.223];objects.append(owner)
    else:owner=g
    owner.gain=[.113,.217];array=owner.variables['gain'].get_value();f=b.Function(callback(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False);owner.namespace['f']=f
    if where=='regular':owner.run_regularly('v=f(v)+gain',when='groups',order=-1)
    net=b.Network(*objects);options=dict(trainable_synapse_parameters={owner.name:['gain','h']}) if synaptic else dict(trainable_neuron_parameters={g.name:['gain']})
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,**options)
    assert 'gain' in bundle.provenance['mutable_constant_layout'][owner.name]
    layout=bundle.provenance['dynamic_state_layout' if synaptic else 'neuron_state_layout'][owner.name]
    assert bundle.provenance['mutable_capture_layout'][0]['cells']==layout['gain']
    variables={**b.core.functions.DEFAULT_FUNCTIONS,**owner.variables,**owner.resolve_all(owner.equations.identifiers-set(owner.variables),run_namespace={}),**owner.namespace};variables={key:variables[key] for key in sorted(variables)}
    generated=_INTEGRATORS[method](owner.equations,variables=variables);scalar,vector=make_statements(generated,variables,np.float64,optimise=True)
    gen=NumpyCodeGenerator(variables,owner.variables.indices,owner,{'_idx'},NumpyCodeObject,owner.name,'stateupdate',allows_scalar_write=True)
    code=compile('\n'.join([*gen.translate_one_statement_sequence(scalar,scalar=True),*gen.translate_one_statement_sequence(vector)]),'<original captured constant integration>','exec')
    regular=None
    if where=='regular':
        scalar,vector=make_statements('v=f(v)+gain',variables,np.float64,optimise=True)
        regular=compile('\n'.join([*gen.translate_one_statement_sequence(scalar,scalar=True),*gen.translate_one_statement_sequence(vector)]),'<original captured constant regular>','exec')
    x=np.zeros((1,4,1));x[0,[0,2],0]=1
    return net,g,owner,array,dt,bundle,x,variables,gen,code,regular,noisy,where


def oracle(data,weights,initial=None,anchors=None):
    net,g,owner,array,dt,bundle,x,variables,gen,code,regular,noisy,where=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for index,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[index]=weights[ref[0]][ref[1]]
    synaptic=where=='synapse';layout=bundle.provenance['dynamic_state_layout' if synaptic else 'neuron_state_layout'][owner.name];gain=layout['gain'];field='h' if synaptic else 'v';live=layout[field];v=bundle.provenance['neuron_state_layout'][g.name]['v']
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());env=dict(sqrt=np.sqrt,_numpy=np,_vectorisation_idx=np.arange(2))
        def randn(index):return np.array([normal(p['seed'],0,0,2 if synaptic else 1,j,tick,0) for j in range(2)])
        env['randn']=randn
        for name,var in variables.items():
            if not isinstance(var,b.core.variables.ArrayVariable):
                if hasattr(var,'get_value') and not isinstance(var,b.core.variables.AuxiliaryVariable):env[name]=np.asarray(var.get_value()).reshape(-1)[0]
                continue
            key=gen.get_array_name(var)
            if key in env:continue
            value=z[live] if var is owner.variables[field] else z[gain] if var is owner.variables['gain'] else np.array([tick*float(dt)]) if name=='t' else var.get_value()
            env[key]=np.asarray(value).copy()
        env['f']=callback(env[gen.get_array_name(owner.variables['gain'])])
        if regular is not None:exec(regular,env)
        exec(code,env);z[live]=env[gen.get_array_name(owner.variables[field])];z[gain]=env[gen.get_array_name(owner.variables['gain'])]
        threshold=env['f'](env[gen.get_array_name(owner.variables['gain'])]) if where=='threshold' else .5
        z[gain]=env[gen.get_array_name(owner.variables['gain'])];margin=z[v]-threshold
        if g.name in bundle.provenance.get('threshold_margin_layout',{}):z[bundle.provenance['threshold_margin_layout'][g.name]]=margin
        event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        if synaptic and tick in (0,2):z[v]+=z[live]*z[gain]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


CASES=[('integrator','euler',False),('integrator','rk2',False),('integrator','rk4',False),('integrator','euler',True),('integrator','heun',True),('integrator','milstein',True),('threshold','euler',False),('regular','euler',False),('synapse','euler',False),('synapse','rk2',False),('synapse','rk4',False),('synapse','euler',True),('synapse','heun',True),('synapse','milstein',True)]
@pytest.mark.parametrize('where,method,noisy',CASES)
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_captured_constant_original_all_vjps(engine,where,method,noisy,ranks,window,monkeypatch):
    mpi(ranks);data=model(where,method,noisy,ranks,window,engine);net,g,owner,array,dt,bundle,x,*_=data
    np.testing.assert_array_equal(array,[.113,.217]);out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,spikes,anchors=oracle(data,bundle.weights)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(data,hi,anchors=anchors)[0]-oracle(data,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    layouts=[bundle.provenance['neuron_state_layout'][g.name]]+([bundle.provenance['dynamic_state_layout'][owner.name]] if where=='synapse' else [])
    indices=set(index for layout in layouts for cells in layout.values() for index in cells)
    for index in sorted(indices):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(oracle(data,bundle.weights,hi,anchors)[0]-oracle(data,bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    if noisy:
        draws=iter(np.array([normal(bundle.plan['seed'],0,0,2 if where=='synapse' else 1,j,tick,0) for j in range(2)]) for tick in range(4))
        def replay(*shape):assert shape==(2,);return next(draws).copy()
        monkeypatch.setattr(np.random,'randn',replay)
    net.run(4*dt,namespace={})
    for obj,layout in zip([g,owner] if where=='synapse' else [g],layouts):
        for field,cells in layout.items():
            if field.startswith('__'):continue
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[field].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('where,method,noisy',CASES)
@pytest.mark.parametrize('ranks',[None,2])
def test_captured_constant_restore(engine,where,method,noisy,ranks,tmp_path):
    mpi(ranks);data=model(where,method,noisy,ranks,None,engine);bundle=data[5];x=data[6];p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable'])
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:2],[0]);path=tmp_path/'captured-constant';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes'])


@pytest.mark.parametrize('where',['integrator','synapse','regular'])
@pytest.mark.parametrize('dtype',['int32','bool'])
@pytest.mark.parametrize('ranks',[None,2])
def test_captured_typed_constant_exact_and_detached(engine,where,dtype,ranks):
    from test_training_mutable_captures import integer_callback,boolean_callback
    mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt);hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>1e12',reset='v=0',method='euler',dt=dt)
    decl='gain:'+('integer' if dtype=='int32' else 'boolean')+' (constant)'
    g=b.NeuronGroup(2,'dv/dt='+('f(v)/ms' if where=='integrator' else '0/second')+':1'+('' if where=='synapse' else '\n'+decl),threshold='v>1e12',reset='v=0',method='euler',dt=dt);g.v=[.173,.719];objects=[source,hidden,g]
    if where=='synapse':
        owner=b.Synapses(source,g,'dh/dt=f(h)/ms:1 (clock-driven)\n'+decl,on_pre='v_post+=h',method='euler',dt=dt);owner.connect(i=[0,0],j=[0,1]);owner.h=[.137,.223];objects.append(owner)
    else:owner=g
    owner.gain=[16777217,2147483647] if dtype=='int32' else [True,False];array=owner.variables['gain'].get_value();function=integer_callback(array) if dtype=='int32' else boolean_callback(array);owner.namespace['f']=b.Function(function,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    if where=='regular':g.run_regularly('v=f(v)',when='groups')
    net=b.Network(*objects);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks)
    cells=bundle.provenance['mutable_constant_layout'][owner.name]['gain'];assert bundle.provenance['mutable_capture_layout'][0]['cells']==cells
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,3,1)),[0]);net.run(3*dt,namespace={})
    np.testing.assert_array_equal(np.asarray(out['final_state'])[0,cells],array);np.testing.assert_array_equal(np.asarray(out['initial_state_gradients'])[0,cells],0.)
    field='h' if where=='synapse' else 'v';key='dynamic_state_layout' if where=='synapse' else 'neuron_state_layout';values=bundle.provenance[key][owner.name][field]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,values],owner.variables[field].get_value(),rtol=8e-5,atol=8e-6)


def read_capture_callback(array):
    def read_capture(x):
        x*=.8
        return x+array
    return read_capture


def test_capture_write_probe_and_dynamic_view_identity():
    from brian2_rust.training_effects import lower_state_effect_function,mutated_state_effect_captures,capture_array_key,same_capture_storage
    array=np.array([.113,.217]);view=array.view()
    assert array is not view and capture_array_key(array)==capture_array_key(view) and same_capture_storage(array,view)
    assert not same_capture_storage(np.array(array),array)
    assert mutated_state_effect_captures(lower_state_effect_function(callback(array)))=={'array'}
    assert mutated_state_effect_captures(lower_state_effect_function(read_capture_callback(array)))==set()
    np.testing.assert_array_equal(array,[.113,.217])


@pytest.mark.parametrize('where',['integrator','synapse','regular'])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_read_capture_preserves_borrowed_constant_argument_vjps(engine,where,ranks,window):
    mpi(ranks);b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    external=np.array([.03,.07]);f=b.Function(read_capture_callback(external),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt);hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt='+('f(gain)/ms' if where=='integrator' else '0/second')+':1'+('' if where=='synapse' else '\ngain:1 (constant)'),threshold='v>.5',reset='v-=.5',method='euler',dt=dt,namespace={'f':f});g.v=[.173,.719];objects=[source,hidden,g]
    if where=='synapse':
        owner=b.Synapses(source,g,'dh/dt=f(gain)/ms:1 (clock-driven)\ngain:1 (constant)',on_pre='v_post+=h*gain',method='euler',dt=dt,namespace={'f':f});owner.connect(i=[0,0],j=[0,1]);owner.h=[.137,.223];objects.append(owner)
    else:owner=g
    owner.gain=[.113,.217]
    if where=='regular':g.run_regularly('v+=f(gain)',when='groups',order=-1)
    net=b.Network(*objects);options=dict(trainable_synapse_parameters={owner.name:['gain','h']}) if where=='synapse' else dict(trainable_neuron_parameters={g.name:['gain']})
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=engine,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,**options)
    assert 'gain' in bundle.provenance['mutable_constant_layout'][owner.name]
    layout=bundle.provenance['dynamic_state_layout' if where=='synapse' else 'neuron_state_layout'][owner.name];gain=layout['gain'];v=bundle.provenance['neuron_state_layout'][g.name]['v'];capture=bundle.provenance['mutable_capture_layout'][0]['cells'];h=layout.get('h')
    def reference(weights,initial=None,anchors=None):
        p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float);old=[];margins=[];hard=[];spikes=[]
        if initial is None:
            for j,ref in enumerate(p['dynamic']['initial_parameters']):
                if ref is not None:z[j]=weights[ref[0]][ref[1]]
        for tick in range(4):
            if anchors is not None and window and tick and tick%window==0:z=anchors['old'][tick].copy()
            old.append(z.copy());z[gain]*=.8;z[h if where=='synapse' else v]+=(1. if where=='regular' else .2)*(z[gain]+z[capture])
            margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
            if anchors is not None:
                base=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margin-base)
            spikes.append(event.copy())
            if where=='synapse' and tick in (0,2):z[v]+=z[h]*z[gain]
            z[v]-=.5*event
        logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
        return loss,z,dict(old=old,margins=margins,hard=hard)
    x=np.zeros((1,4,1));x[0,[0,2],0]=1;out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(bundle.weights)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(reference(hi,anchors=anchors)[0]-reference(lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=1e-5)
    for j in set([*gain,*v,*capture,*([*h] if h is not None else [])]):
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(bundle.weights,hi,anchors)[0]-reference(bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5)
    net.run(4*dt,namespace={});np.testing.assert_allclose(owner.gain[:],z[gain]);np.testing.assert_array_equal(external,[.03,.07])
