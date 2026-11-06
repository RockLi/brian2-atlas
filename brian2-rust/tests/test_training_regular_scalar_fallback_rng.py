"""Actual Brian scalar NumPy fallback samples per row, retaining captures."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
import test_training_scalar_event_rng_functions as factories
from test_training_regular_callback_rng import count
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(backend,kind,capture,window=None):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=(gain*u+.01)/ms:1\nu:1\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant,shared)\npick:integer\npeer:1 (linked)',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.59,.38];g.u=[.173,.219];g.mean=1.3 if kind in ('poisson','mixed') else .04;g.spread=.13;g.gain=.2;g.pick=[0,0];g.peer=b.linked_var(g,'v',index='pick')
    function=getattr(factories,kind)(g.variables['v'].get_value()) if capture else getattr(factories,'pure_'+kind)
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    runner=g.run_regularly('peer=.05*f(u,mean,spread)+.9*peer;u+=.01*randn()',when='start')
    net=b.Network(inp,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],backend=backend,mpi_ranks=2,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['mean','spread','gain']})
    assert bundle.provenance['regular_runner_layout'][runner.name]['numpy_mode']=='scalar'
    return net,g,runner,dt,bundle,np.zeros((1,6,1))


@pytest.mark.parametrize('kind,capture',[('normal',False),('normal',True),('uniform',False),('poisson',True),('mixed',True)])
def test_regular_scalar_fallback_original_restore(engine,kind,capture,monkeypatch,tmp_path):
    mpi(2);net,g,runner,dt,bundle,x=model(engine,kind,capture)
    domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain']
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    order=(['normal','uniform','poisson'] if kind=='mixed' else [kind])+['randn'];calls=[]
    for tick in range(6):
        serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            entity,stream=divmod(serial,len(order));assert which==order[stream] and entity<2;serial+=1
            calls.append((tick,entity,stream,which))
            if which=='randn':return normal(p['seed'],9,0,domain,entity,tick,stream)
            mean=float(g.mean[:]);spread=float(g.spread[:])
            assert all(np.ndim(a)==0 for a in args)
            np.testing.assert_allclose(args,(mean,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            if which=='poisson':return int(count(p['seed'],9,domain,entity,tick,mean,stream))
            return mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,entity,tick,stream)
        for which in set(order):monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6,err_msg=name)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==12*len(order)
    if kind in ('poisson','mixed'):assert len(trainer.poisson_state['entries'])==12


def parameter(data,weights,name):
    row=next(row for row in data[4].provenance['bindings'] if name in row['variables'])
    return weights[row['bank']][row['variables'].index(name)]


def reference(data,kind,capture,weights,initial=None,anchors=None,override=None):
    _,g,runner,_,bundle,_=data;p=bundle.plan
    state=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:state[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u']
    domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain']
    before=[];margins=[];hard=[];events=[];samples=[]
    kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind]
    mean=parameter(data,weights,'mean');spread=parameter(data,weights,'spread')
    for tick in range(6):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
            state=anchors['before'][tick].copy()
        before.append(state.copy())
        for entity in range(2):
            # Actual scalar fallback loads all row locals before callbacks.
            peer=state[v[0]];operand=state[u[entity]]
            if capture:state[v]*=.8;operand*=.7
            total=0.
            for stream,which in enumerate(kinds):
                if which=='poisson':
                    noise=count(p['seed'],9,domain,entity,tick,float(mean),stream)
                    if anchors is not None:noise=next(c for t,j,s,r,c in anchors['samples'] if (t,j,s)==(tick,entity,stream))
                    if override==(tick,entity,stream):noise+=1
                    samples.append((tick,entity,stream,float(mean),noise))
                else:noise=mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,entity,tick,stream)
                total+=noise
            state[v[0]]=.05*(operand+.1*total)+.9*peer
            state[u[entity]]+=.01*normal(p['seed'],9,0,domain,entity,tick,len(kinds))
        state[v]+=.2*(parameter(data,weights,'gain')*state[u]+.01)
        margin=state[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());state[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,state,dict(before=before,margins=margins,hard=hard,samples=samples)


@pytest.mark.parametrize('kind,capture,zero',[('normal',True,False),('mixed',True,False),('poisson',True,True)])
@pytest.mark.parametrize('window',[None,2])
def test_regular_scalar_fallback_all_bank_initial_vjps(engine,kind,capture,zero,window):
    mpi(2);data=model(engine,kind,capture,window);bundle=data[4]
    row=next(row for row in bundle.provenance['bindings'] if 'mean' in row['variables']);mean=(row['bank'],row['variables'].index('mean'))
    if zero:
        bundle.weights[mean[0]][mean[1]]=0.;cell=bundle.provenance['neuron_state_layout'][data[1].name]['v'][0]
        for voltage in np.arange(.35,1.0,.001):
            bundle.initial_state[cell]=float(voltage);bundle.plan['dynamic']['initial'][cell]=float(voltage)
            loss,_,a=reference(data,kind,capture,bundle.weights)
            weak=sum(reference(data,kind,capture,bundle.weights,override=(t,j,s))[0]-loss for t,j,s,r,c in a['samples'])
            if abs(weak)>1e-5 and min(abs(m[0]) for m in a['margins'])>1e-5:break
        else:raise AssertionError('zero boundary requires a nonzero counterfactual')
    loss,state,anchors=reference(data,kind,capture,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[5],[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    layout=bundle.provenance['neuron_state_layout'][data[1].name];cells=sorted(set(layout['v']+layout['u']))
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],state[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],anchors['hard'])
    weak=sum(loss*(c/r-1.) if r else reference(data,kind,capture,bundle.weights,override=(t,j,s))[0]-loss for t,j,s,r,c in anchors['samples'])
    if zero:assert abs(weak)>1e-5
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            # At zero the scalar Poisson has no continuous count dependence.
            fd=0. if zero and (bank,index)==mean else (reference(data,kind,capture,hi,anchors=anchors)[0]-reference(data,kind,capture,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd+(weak if (bank,index)==mean else 0.),rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float)
    explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[5],[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(data,kind,capture,bundle.weights,initial=initial)
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0;continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,kind,capture,bundle.weights,initial=hi,anchors=anchors)[0]-reference(data,kind,capture,bundle.weights,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=2e-5),index


@pytest.mark.parametrize('kind',['normal','poisson'])
def test_regular_scalar_fallback_invalid_restore_is_atomic(engine,kind,tmp_path):
    mpi(2);data=model(engine,kind,True);bundle=data[4];x=data[5]
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x[:,:1],[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind=='poisson' else 'spread'
    row=next(row for row in bundle.provenance['bindings'] if name in row['variables'])
    trainer.state['weights'][row['bank']][row['variables'].index(name)]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x[:,1:2],[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before
