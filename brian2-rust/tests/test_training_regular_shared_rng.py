"""A vector callback draws scalar distributions once per NumPy clock block."""
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


def model(backend,ranks,mode,kind,capture,leading,period=1,window=None):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1,[],np.array([])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=(gain*u+.01)/ms:1\nu:1\nz:1 (shared)\nmean:1 (constant,shared)\nspread:1 (constant,shared)\ngain:1 (constant,shared)\npick:integer\npeer:1 (linked)',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.59,.38];g.u=[.173,.219];g.z=.2;g.mean=1.3 if kind in ('poisson','mixed','hybrid') else .04;g.spread=.13;g.gain=.2;g.pick=[0,0];g.peer=b.linked_var(g,'v',index='pick')
    function=hybrid(g.variables['v'].get_value()) if kind=='hybrid' else getattr(factories,kind)(g.variables['v'].get_value()) if capture else getattr(factories,'pure_'+kind)
    g.namespace['f']=b.Function(function,arg_units=[1,1,1],return_unit=1,stateless=False)
    g.namespace['p']=b.Function(factories.pure_normal,arg_units=[1,1,1],return_unit=1,stateless=False)
    vector='u=f(u,mean,spread);v+=.01*randn()' if mode=='array' else 'peer+=.05*f(u,mean,spread);peer+=.03*f(u,mean,spread);v+=.01*randn()'
    runner=g.run_regularly(('z=p(z,mean,spread);' if leading else '')+vector,when='start',dt=period*dt)
    net=b.Network(inp,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['mean','spread','gain']})
    assert bundle.provenance['regular_runner_layout'][runner.name]['whole_callback_blocks']
    return net,g,runner,dt,bundle,np.zeros((1,6,1))


@pytest.mark.parametrize('kind,capture',[(kind,capture) for kind in ('normal','uniform','poisson','mixed') for capture in (False,True) if capture or kind!='mixed'])
@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('leading',[False,True])
@pytest.mark.parametrize('period',[1,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_regular_shared_rng_original_restore(engine,kind,capture,mode,leading,period,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,runner,dt,bundle,x=model(engine,ranks,mode,kind,capture,leading,period)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain'];calls=[];kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind]
    order=(['normal'] if leading else [])+kinds*(1 if mode=='array' else 2)+['randn']
    for tick in range(6):
        serial=0
        def sample(which,*args,**kwargs):
            nonlocal serial
            assert tick%period==0 and serial<len(order) and which==order[serial]
            stream=serial;serial+=1;mean=float(g.mean[:]);spread=float(g.spread[:]);calls.append((tick,stream,which))
            if which=='randn':
                assert args==(2,) and not kwargs
                return np.array([normal(p['seed'],9,0,domain,j,tick//period,stream) for j in range(2)])
            assert all(np.ndim(value)==0 for value in args)
            np.testing.assert_allclose(args,(mean,) if which=='poisson' else (mean,spread if which=='normal' else mean+spread))
            assert kwargs==({'size':None} if which=='uniform' and kind!='mixed' else {})
            if which=='poisson':return int(count(p['seed'],9,domain,0,tick//period,mean,stream))
            return float(mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,0,tick//period,stream))
        for which in set(order):monkeypatch.setattr(np.random,which,lambda *args,_which=which,**kwargs:sample(_which,*args,**kwargs))
        out=trainer.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u','z'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],np.broadcast_to(g.variables[name].get_value(),(len(cells),)),rtol=8e-5,atol=8e-6,err_msg=name)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    assert len(calls)==6//period*len(order)
    if 'poisson' in kinds:assert len(trainer.poisson_state['entries'])==6//period*(1 if mode=='array' else 2)


def parameter(data,weights,name):
    row=next(row for row in data[4].provenance['bindings'] if name in row['variables'])
    return weights[row['bank']][row['variables'].index(name)]


def reference(data,kind,capture,mode,leading,period,weights,initial=None,anchors=None,override=None):
    _,g,runner,_,bundle,_=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];shared=sorted(set(layout['z']));domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain']
    before=[];margins=[];hard=[];spikes=[];samples=[];kinds=['normal','uniform','poisson'] if kind=='mixed' else [kind]
    mean=parameter(data,weights,'mean');spread=parameter(data,weights,'spread')
    for tick in range(6):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        if tick%period==0:
            stream=0
            if leading:z[shared]+=.1*(mean+spread*normal(p['seed'],9,0,domain,0,tick//period,stream));stream+=1
            for coefficient in ([None] if mode=='array' else [.05,.03]):
                if capture:
                    if mode!='array':z[v]*=.8
                    z[u]*=.7
                value=0.
                for which in kinds:
                    if which=='poisson':
                        noise=count(p['seed'],9,domain,0,tick//period,float(mean),stream)
                        if anchors is not None:noise=next(c for t,s,r,c in anchors['samples'] if (t,s)==(tick,stream))
                        if override==(tick,stream):noise+=1
                        samples.append((tick,stream,float(mean),noise))
                    else:noise=mean+spread*(normal if which=='normal' else uniform)(p['seed'],9,0,domain,0,tick//period,stream)
                    value+=noise;stream+=1
                result=z[u]+.1*value
                if mode=='array':z[u]=result
                else:
                    for j in range(2):z[v[0]]+=coefficient*result[j]
            z[v]+=.01*np.array([normal(p['seed'],9,0,domain,j,tick//period,stream) for j in range(2)])
        z[v]+=.2*(parameter(data,weights,'gain')*z[u]+.01)
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        spikes.append(event.copy());z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples)


PROFILES=[('normal',True,'array',False,1,False),('uniform',False,'vectorised',True,2,False),
          ('poisson',True,'vectorised',False,1,False),('poisson',False,'array',False,1,True),
          ('poisson',True,'vectorised',True,2,True),('mixed',True,'array',True,2,False),
          ('mixed',True,'vectorised',False,1,False)]


@pytest.mark.parametrize('kind,capture,mode,leading,period,zero',PROFILES)
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_regular_shared_rng_all_vjps(engine,kind,capture,mode,leading,period,zero,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind,capture,leading,period,window);_,g,_,_,bundle,x=data
    row=next(row for row in bundle.provenance['bindings'] if 'mean' in row['variables']);mean=(row['bank'],row['variables'].index('mean'))
    args=(data,kind,capture,mode,leading,period,bundle.weights)
    if zero:
        bundle.weights[mean[0]][mean[1]]=0.;cell=bundle.provenance['neuron_state_layout'][g.name]['v'][0]
        for voltage in np.arange(.35,1.0,.005):
            bundle.initial_state[cell]=float(voltage);bundle.plan['dynamic']['initial'][cell]=float(voltage)
            loss,_,a=reference(*args);weak=sum(reference(*args,override=(t,s))[0]-loss for t,s,r,c in a['samples'])
            if abs(weak)>1e-5 and min(abs(m[0]) for m in a['margins'])>1e-4:break
        else:raise AssertionError('zero-rate fixture needs nonzero counterfactual signal')
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    loss,z,anchors=reference(*args);layout=bundle.provenance['neuron_state_layout'][g.name];cells=sorted(set(layout['v']+layout['u']+layout['z']))
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],anchors['hard'])
    weak=sum(loss*(c/rate-1.) if rate else reference(*args,override=(tick,stream))[0]-loss for tick,stream,rate,c in anchors['samples'])
    if zero:assert abs(weak)>1e-5
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,kind,capture,mode,leading,period,hi,anchors=anchors)[0]-reference(data,kind,capture,mode,leading,period,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd+(weak if (bank,index)==mean else 0.),rel=2e-3,abs=1e-5),(bank,index)
    initial=np.array(bundle.initial_state);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],initial=initial[None],noise_sequence=9)
    _,_,anchors=reference(*args,initial=initial)
    for index in range(len(z)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:assert explicit['initial_state_gradients'][0][index]==0;continue
        hi=initial.copy();lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(*args,initial=hi,anchors=anchors)[0]-reference(*args,initial=lo,anchors=anchors)[0])/2e-6
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=2e-3,abs=1e-5),index


@pytest.mark.parametrize('kind,mode',[('normal','vectorised'),('poisson','array')])
def test_regular_shared_rng_invalid_atomic(engine,kind,mode,tmp_path):
    mpi(2);data=model(engine,2,mode,kind,True,True);bundle=data[4];x=data[5];p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);trainer.step(x[:,:1],[0],noise_sequence=9)
    path=tmp_path/'prefix';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER);trainer.restore(path)
    name='mean' if kind=='poisson' else 'spread';row=next(row for row in bundle.provenance['bindings'] if name in row['variables']);trainer.state['weights'][row['bank']][row['variables'].index(name)]=-1.
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence))
    with pytest.raises(ValueError):trainer.step(x[:,1:2],[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.poisson_state,trainer.elapsed_ticks,trainer.next_noise_sequence)==before


def hybrid(array):
    def f(x,mean,spread):
        captured=array
        captured*=.8
        x*=.7
        scalar=np.random.poisson(mean)
        vector=np.random.poisson(x+.1)
        return x+.1*scalar+.03*vector
    return f


def test_regular_hybrid_poisson_original(engine,monkeypatch):
    mpi(2);net,g,runner,dt,bundle,x=model(engine,2,'array','hybrid',True,False)
    domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain'];p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    def poisson(rate):
        if np.ndim(rate)==0:return int(count(p['seed'],9,domain,0,0,float(rate),0))
        np.testing.assert_allclose(rate,.7*np.asarray(g.u[:])+.1)
        return np.array([count(p['seed'],9,domain,j,0,float(rate[j]),1) for j in range(2)])
    monkeypatch.setattr(np.random,'poisson',poisson)
    monkeypatch.setattr(np.random,'randn',lambda n:np.array([normal(p['seed'],9,0,domain,j,0,2) for j in range(n)]))
    trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);out=trainer.step(x[:,:1],[0],noise_sequence=9);net.run(dt,namespace={})
    for name in ('v','u'):
        cells=bundle.provenance['neuron_state_layout'][g.name][name]
        np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    assert len(trainer.poisson_state['entries'])==3
