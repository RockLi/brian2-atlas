"""Shared scheduled callbacks sample once, before neuron vector execution."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
import test_training_scalar_event_rng_functions as factories
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal
from test_training_uniform import uniform
from test_training_poisson_brian import uniforms
from test_native_training import RUNNER


def count(seed,sequence,domain,entity,tick,rate,stream):
    return len(uniforms(seed,sequence,domain,entity,tick,stream,rate))-1


def model(backend,ranks,kind,capture=True,period=1,window=None):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=(gain*z+.2*u)/ms:1\nu:1\nz:1 (shared)\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant,shared)',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.41,.31];g.u=[.017,.031];g.z=.2;g.mean=1.3 if kind in ('poisson','mixed') else .04;g.spread=.13;g.gain=.2
    function=getattr(factories,kind)(g.variables['u'].get_value()) if capture else getattr(factories,'pure_'+kind)
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    runner=g.run_regularly('z=f(z,mean,spread)',dt=period*dt,when='start')
    net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['mean','spread','gain']})
    assert bundle.provenance['regular_runner_layout'][runner.name]['callback_draws']['scalar']
    return net,g,runner,dt,bundle,np.zeros((1,6,1))


@pytest.mark.parametrize('kind,capture',[(kind,capture) for kind in ('normal','uniform','poisson','mixed') for capture in (False,True)])
@pytest.mark.parametrize('period',[1,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_regular_scalar_rng_original_restore(engine,kind,capture,period,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,runner,dt,bundle,x=model(engine,ranks,kind,capture,period)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain'];calls=[];kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind]
    for tick in range(6):
        serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            assert tick%period==0 and serial<len(kinds) and which==kinds[serial]
            mean=float(g.mean[:]);spread=float(g.spread[:]);assert all(np.ndim(value)==0 for value in args)
            np.testing.assert_allclose(args,(mean,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            assert kwargs==({'size':None} if which=='uniform' and kind!='mixed' else {})
            calls.append((tick,serial,which));stream=serial;serial+=1
            if which=='poisson':return int(count(p['seed'],9,domain,0,tick//period,mean,stream))
            return float(mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,0,tick//period,stream))
        for which in kinds:monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u','z'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],np.broadcast_to(g.variables[name].get_value(),(len(cells),)),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==6//period*len(kinds)
    if 'poisson' in kinds:assert len(trainer.poisson_state['entries'])==6//period


def reference(data,kind,capture,period,weights,initial=None,anchors=None,override=None):
    """Physical Brian recurrence, independent of generated native programs."""
    _,g,runner,_,bundle,_=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];shared=sorted(set(layout['z']))
    banks={name:(row['bank'],index) for row in bundle.provenance['bindings'] if row['object']==g.name for index,name in enumerate(row['variables'])}
    def parameter(name):bank,index=banks[name];return weights[bank][index]
    domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain'];before=[];margins=[];hard=[];spikes=[];samples=[]
    kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind]
    for tick in range(6):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        if tick%period==0:
            mean=parameter('mean');spread=parameter('spread');noise=0.
            for stream,which in enumerate(kinds):
                if which=='poisson':
                    value=count(p['seed'],9,domain,0,tick//period,float(mean),stream)
                    if anchors is not None:value=next(c for t,s,r,c in anchors['samples'] if (t,s)==(tick,stream))
                    if override==(tick,stream):value+=1
                    samples.append((tick,stream,float(mean),value))
                else:value=mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,0,tick//period,stream)
                noise+=value
            if capture:z[u]*=.8
            z[shared]=( .7 if capture else 1.)*z[shared]+.1*noise
        z[v]+=.2*(parameter('gain')*z[shared[0]]+.2*z[u])
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        spikes.append(event.copy());z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples)


@pytest.mark.parametrize('kind,zero',[('normal',False),('uniform',False),('poisson',False),('poisson',True),('mixed',False)])
@pytest.mark.parametrize('capture',[False,True])
@pytest.mark.parametrize('period',[1,2])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_regular_scalar_rng_all_vjps(engine,kind,zero,capture,period,window,ranks):
    mpi(ranks);data=model(engine,ranks,kind,capture,period,window);_,g,_,_,bundle,x=data
    if zero:
        bank,index=parameter_slot(bundle,'mean');bundle.weights[bank][index]=0.
        layout=bundle.provenance['neuron_state_layout'][g.name]
        for cell,value in [(cell,0.) for cell in set(layout['z'])]+[(layout['v'][0],.49)]:
            bundle.initial_state[cell]=value;bundle.plan['dynamic']['initial'][cell]=value
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    args=(data,kind,capture,period,bundle.weights);loss,z,anchors=reference(*args)
    cells=sorted(set(cell for obj in bundle.provenance['neuron_state_layout'].values() for name,slots in obj.items() if not name.startswith('__') for cell in slots))
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    mean=parameter_slot(bundle,'mean')
    weak=sum(loss*(c/rate-1.) if rate else reference(*args,override=(tick,stream))[0]-loss for tick,stream,rate,c in anchors['samples'])
    if zero:assert abs(weak)>1e-6
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,kind,capture,period,hi,anchors=anchors)[0]-reference(data,kind,capture,period,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd+(weak if (bank,index)==mean else 0.),rel=2e-3,abs=1e-5),(bank,index)
    initial=np.array(bundle.initial_state);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(*args,initial=initial)
    for index in range(len(z)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:assert explicit['initial_state_gradients'][0][index]==0;continue
        hi=initial.copy();lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(*args,initial=hi,anchors=anchors)[0]-reference(*args,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=2e-3,abs=1e-5),index


@pytest.mark.parametrize('kind',['normal','poisson'])
def test_regular_scalar_rng_invalid_atomic(engine,kind,tmp_path):
    mpi(2);data=model(engine,2,kind,True);_,_,_,_,bundle,x=data;p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x[:,:1],[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind=='poisson' else 'spread';bank,index=parameter_slot(bundle,name);trainer.state['weights'][bank][index]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x[:,1:2],[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before


def parameter_slot(bundle,name):
    return next((row['bank'],row['variables'].index(name)) for row in bundle.provenance['bindings'] if name in row['variables'])
