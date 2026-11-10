"""Timed sampling, borrowed/private callback mutation, SDE and synaptic state VJPs."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2.codegen.translation import make_statements
from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from brian2_rust.training_brian import _INTEGRATORS
from test_training_callback_effects import curve
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_native_training import RUNNER


def model(dimensions,private,noisy,synaptic,ranks,window,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    values=np.array([[.13,.21],[.31,.41],[.23,.17],[.19,.37]])
    drive=b.TimedArray(values if dimensions==2 else values[:,0],dt=.3*b.ms)
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    call='drive(t'+(', j)' if synaptic else ', i)') if dimensions==2 else 'drive(t)'
    rhs='('+('curve(gain+'+call+')' if private else 'curve(gain)+'+call)+')/ms'+('+.011*xi/sqrt(ms)' if noisy else '')
    namespace=dict(drive=drive,curve=b.Function(curve,arg_units=[1],arg_names=['x'],return_unit=1,stateless=False))
    g=b.NeuronGroup(2,('dv/dt=0/second:1' if synaptic else 'dv/dt='+rhs+':1\ngain:1 (constant)'),threshold='v>.5',reset='v-=.5',method='euler',dt=dt,namespace=namespace)
    g.v=[.173,.719];objects=[source,hidden,g]
    if synaptic:
        owner=b.Synapses(source,g,'dh/dt='+rhs+':1 (clock-driven)\ngain:1 (constant)',on_pre='v_post+=h*gain',method='euler',dt=dt,namespace=namespace)
        owner.connect(i=[0,0],j=[0,1]);owner.h=[.137,.223];objects.append(owner)
    else:owner=g
    owner.gain=[.113,.217];net=b.Network(*objects)
    options=dict(trainable_synapse_parameters={owner.name:['gain','h']}) if synaptic else dict(trainable_neuron_parameters={g.name:['gain']})
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,**options)
    promoted=bundle.provenance['mutable_constant_layout'].get(owner.name,{})
    assert ('gain' in promoted)==(not private)
    variables={**b.core.functions.DEFAULT_FUNCTIONS,**owner.variables,**owner.resolve_all(owner.equations.identifiers-set(owner.variables),run_namespace={}),**namespace}
    variables={key:variables[key] for key in sorted(variables)}
    scalar,vector=make_statements(_INTEGRATORS['euler'](owner.equations,variables=variables),variables,np.float64,optimise=True)
    gen=NumpyCodeGenerator(variables,owner.variables.indices,owner,{'_idx'},NumpyCodeObject,owner.name,'stateupdate',allows_scalar_write=True)
    code=compile('\n'.join([*gen.translate_one_statement_sequence(scalar,scalar=True),*gen.translate_one_statement_sequence(vector)]),'<original NumPy timed SDE updater>','exec')
    x=np.zeros((1,4,1));x[0,[0,2],0]=1
    return net,g,owner,dt,bundle,x,variables,gen,code,drive


def oracle(data,weights,initial=None,anchors=None):
    net,g,owner,dt,bundle,x,variables,gen,code,drive=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for index,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[index]=weights[ref[0]][ref[1]]
    synaptic=owner is not g
    layout=bundle.provenance['dynamic_state_layout' if synaptic else 'neuron_state_layout'][owner.name]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];field='h' if synaptic else 'v'
    gain_bank=next(entry['bank'] for entry in bundle.provenance['bindings'] if entry['object']==owner.name and 'gain' in entry['variables'])
    gain=np.array(weights[gain_bank]) if 'gain' not in layout else None
    timed=bundle.provenance['timed_inputs'][0]
    table=b.TimedArray(np.array(weights[timed['bank']]).reshape(timed['shape']),dt=drive.dt*b.second)
    sample=table.implementations['numpy'].get_code(owner)
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());env=dict(curve=curve,drive=sample,sqrt=np.sqrt,_numpy=np,_vectorisation_idx=np.arange(2))
        def randn(index):
            assert len(index)==2
            return np.array([normal(p['seed'],0,0,2 if synaptic else 1,j,tick,0) for j in range(2)])
        env['randn']=randn
        for name,var in variables.items():
            if not isinstance(var,b.core.variables.ArrayVariable):
                if hasattr(var,'get_value') and not isinstance(var,b.core.variables.AuxiliaryVariable):env[name]=np.asarray(var.get_value()).reshape(-1)[0]
                continue
            key=gen.get_array_name(var)
            if key in env:continue
            if var is owner.variables[field]:value=z[layout[field]]
            elif var is owner.variables['gain']:value=gain if gain is not None else z[layout['gain']]
            elif name=='t':value=np.array([tick*float(dt)])
            else:value=var.get_value()
            env[key]=np.asarray(value).copy()
        exec(code,env);z[layout[field]]=env[gen.get_array_name(owner.variables[field])]
        if 'gain' in layout:z[layout['gain']]=env[gen.get_array_name(owner.variables['gain'])]
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        if synaptic and tick in (0,2):z[v]+=z[layout['h']]*(gain if gain is not None else z[layout['gain']])
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('dimensions',[1,2])
@pytest.mark.parametrize('private',[False,True])
@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('synaptic',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_timed_mutation_original_and_all_vjps(engine,dimensions,private,noisy,synaptic,ranks,window,monkeypatch):
    mpi(ranks);data=model(dimensions,private,noisy,synaptic,ranks,window,engine)
    net,g,owner,dt,bundle,x,*_=data
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,spikes,anchors=oracle(data,bundle.weights)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes);assert out['loss']==pytest.approx(loss,abs=8e-6)
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
        draws=iter(np.array([normal(bundle.plan['seed'],0,0,2 if synaptic else 1,j,tick,0) for j in range(2)]) for tick in range(4))
        def replay(*shape):assert shape==(2,);return next(draws).copy()
        monkeypatch.setattr(np.random,'randn',replay)
    net.run(4*dt,namespace={})
    for obj,key in [(g,'neuron_state_layout')]+([(owner,'dynamic_state_layout')] if synaptic else []):
        for name,cells in bundle.provenance[key][obj.name].items():np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    np.testing.assert_allclose(owner.gain[:],[.113,.217] if private else np.array([.113,.217])*.8**4)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('dimensions',[1,2])
@pytest.mark.parametrize('private',[False,True])
@pytest.mark.parametrize('synaptic',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_timed_noise_mutation_restore(engine,dimensions,private,synaptic,ranks,tmp_path):
    mpi(ranks);data=model(dimensions,private,True,synaptic,ranks,None,engine);bundle=data[4];x=data[5]
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(p['trainable'])
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    first=trainer.step(x[:,:2],[0]);path=tmp_path/'timed-noise';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    last=trainer.step(x[:,2:],[0],initial='carry');np.testing.assert_allclose(last['final_state'],whole['final_state'],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),whole['spikes']);assert trainer.clock_tick==4
