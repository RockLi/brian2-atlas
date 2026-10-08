"""Condition storage introduced by Brian code generation, absent from Synapses."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine
from test_training_delay_update import snapshot


def model(*,flag_on=False,method='euler',post=False,noisy=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,1],[1,0],[0,1],[1,1],[0,1],[1,0]],float);ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='implicit_input')
    a=b.NeuronGroup(2,'dv/dt=-v/ms'+('+sigma*xi/sqrt(ms)' if noisy else '')+':1 (unless refractory)'+('\nsigma:1 (constant)' if noisy else ''),threshold='v>.5',reset='v-=.2',refractory=3*dt,method=method,dt=dt,name='implicit_a');a.v=[.8,.2]
    if noisy:a.sigma=[.03,.05]
    c=b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.2',method=method,dt=dt,name='implicit_c');c.v=[.1,.8]
    code='pick=1-pick\n'+('not_refractory=pick<0\n' if flag_on else 'not_refractory=False\n')+'peer+=w\n'+('not_refractory=True\npeer+=.5*w\n' if flag_on else '')+'v_post+=.1*peer\nw+=.01*int(not_refractory)'
    syn=b.Synapses(inp,c,'w:1\npeer:1 (linked)\npick:integer',dt=dt,name='implicit_syn',**({'on_post':code} if post else {'on_pre':code}));syn.connect(j='i');syn.w=[.03,.07];syn.pick=[0,1];syn.peer=b.linked_var(a,'v',index='pick')
    assert 'not_refractory' not in syn.variables
    net=b.Network(inp,a,c,syn);bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],trainable_neuron_parameters={a.name:['sigma']} if noisy else None,**options)
    assert 'not_refractory' not in syn.variables
    return net,inp,a,c,syn,x,bundle


@pytest.mark.parametrize('flag_on',[False,True])
@pytest.mark.parametrize('method',['euler','rk4'])
def test_implicit_condition_matches_compiled_brian(engine,flag_on,method):
    net,inp,a,c,syn,x,bundle=model(flag_on=flag_on,method=method,backend=engine)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    sm=[b.SpikeMonitor(g) for g in (a,c)];net.add(*sm);net.run(len(x)*.2*b.ms,namespace={})
    assert syn.pre.codeobj.compiled_code['run'] is not None
    assert syn.pre.codeobj.variables['not_refractory'] is a.variables['not_refractory']
    assert syn.pre.codeobj.variable_indices['not_refractory']=='pick'
    z=np.asarray(out['final_state'])[0];layout=bundle.provenance['neuron_state_layout'];expected=np.zeros((len(x),4));tol=1e-13 if engine=='cpu' else 5e-6
    for l,(g,m) in enumerate(zip((a,c),sm)):
        np.testing.assert_allclose(z[layout[g.name]['v']],g.v[:],rtol=tol,atol=tol)
        expected[np.rint(np.asarray(m.t/b.second)/.0002).astype(int),2*l+np.asarray(m.i)]=1
    sl=bundle.provenance['dynamic_state_layout'][syn.name]
    np.testing.assert_allclose(z[sl['w']],syn.w[:],rtol=tol,atol=tol);np.testing.assert_array_equal(z[sl['pick']],syn.pick[:])
    np.testing.assert_array_equal(z[bundle.provenance['refractory_activity_layout'][a.name]],a.not_refractory[:]);np.testing.assert_array_equal(out['spikes'][0],expected)
    if engine!='cpu':assert out['gpu_dispatches']>0


def reference(bundle,x,flag_on,post,*,initial=None,weights=None,anchors=None):
    from test_training_stochastic import normal
    p=bundle.plan;spec=p['dynamic'];weights=bundle.weights if weights is None else weights;z=np.asarray(bundle.initial_state if initial is None else initial,dtype=float).copy()
    if initial is None:
        for k,ref in enumerate(spec['initial_parameters']):
            if ref is not None:z[k]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'];av=layout['implicit_a']['v'];cv=layout['implicit_c']['v'];ac=layout['implicit_a']['__refractory_ticks'];af=bundle.provenance['refractory_activity_layout']['implicit_a']
    sl=bundle.provenance['dynamic_state_layout']['implicit_syn'];wi=sl['w'];pi=sl['pick'];sigma=next((np.asarray(weights[e['bank']]) for e in bundle.provenance['bindings'] if e['object']=='implicit_a' and 'sigma' in e['variables']),None)
    before=[];voltages=[];hard_spikes=[];active=[];spikes=[]
    for t,inp in enumerate(x):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors['before'][t].copy()
        before.append(z.copy());free=(z[ac]==0);value=.8*z[av]
        if sigma is not None:value+=sigma*np.sqrt(.2)*np.array([normal(p['seed'],0,0,0,j,t,0) for j in range(2)])
        z[av]=np.where(free,value,z[av]);z[cv]*=.8;z[ac]=np.maximum(z[ac]-1,0)
        enabled=np.r_[free,[True,True]];v=z[av+cv].copy();hard=((v>.5)&enabled).astype(float);soft=hard.copy()
        if anchors is not None:
            hard=anchors['hard'][t];old=anchors['v'][t];soft=hard+anchors['active'][t]*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old-.5))**2*(v-old)
        voltages.append(v);hard_spikes.append(hard);active.append(enabled);spikes.append(soft);z[af]=free*(1-hard[:2])
        for j in range(2):
            event=hard[2+j] if post else inp[j];gate=soft[2+j] if post else inp[j]
            old=int(z[pi[j]]);new=1-old;peer=z[av[old]]+(.5*z[wi[j]] if flag_on else 0)
            target=av[new];z[target]+=gate*(peer-z[target]);z[cv[j]]+=gate*.1*peer;z[wi[j]]+=gate*.01*flag_on
            if event:z[pi[j]]=new;z[af[new]]=float(flag_on)
        reset=hard if p['detach_reset'] else soft;z[av]-=.2*reset[:2];z[cv]-=.2*reset[2:]
        for j in range(2):
            if hard[j]:z[ac[j]]=2
    logits=np.asarray(spikes)[:,2:].mean(axis=0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.asarray(spikes),dict(before=before,v=voltages,hard=hard_spikes,active=active)


@pytest.mark.parametrize('flag_on',[False,True])
@pytest.mark.parametrize('post',[False,True])
@pytest.mark.parametrize('detach,window',[(True,None),(False,2)])
def test_implicit_condition_every_float_vjp(engine,flag_on,post,detach,window):
    *_,x,bundle=model(flag_on=flag_on,post=post,noisy=True,backend=engine,detach_reset=detach,tbptt_window=window)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0]);loss,z,spikes,anchors=reference(bundle,x,flag_on,post)
    tol=4e-6 if engine=='cpu' else 9e-4;absolute=4e-8 if engine=='cpu' else 6e-6
    assert out['loss']==pytest.approx(loss,abs=absolute);np.testing.assert_array_equal(out['spikes'][0],spikes);np.testing.assert_allclose(out['final_state'][0],z,rtol=tol,atol=absolute)
    if engine!='cpu':assert out['gpu_dispatches']>0
    eps=1e-6
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(reference(bundle,x,flag_on,post,weights=hi,anchors=anchors)[0]-reference(bundle,x,flag_on,post,weights=lo,anchors=anchors)[0])/(2*eps)
            assert out['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute)
    for k,detached in enumerate(bundle.plan['dynamic']['detached']):
        if detached:
            assert out['initial_state_gradients'][0][k]==0;continue
        hi=np.asarray(bundle.initial_state).copy();lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(reference(bundle,x,flag_on,post,initial=hi,anchors=anchors)[0]-reference(bundle,x,flag_on,post,initial=lo,anchors=anchors)[0])/(2*eps)
        assert out['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute)


@pytest.mark.parametrize('flag_on',[False,True])
@pytest.mark.parametrize('post',[False,True])
@pytest.mark.parametrize('ranks',[2,8])
def test_implicit_condition_mpi_carry_checkpoint_rollback(engine,flag_on,post,ranks,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(flag_on=flag_on,post=post,noisy=True,backend=engine,mpi_ranks=ranks,detach_reset=False,tbptt_window=2)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights)
    q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None);cpu=NativeLIFTrainer(q,runner=RUNNER,weights=bundle.weights);xx=np.stack([x,x[:,::-1]])
    for t in (0,2,4):
        kw=dict(initial='carry') if t else dict(noise_sequence=7,start_tick=3)
        actual=trainer.step(xx[:,t:t+2],[0,1],**kw);expected=cpu.step(xx[:,t:t+2],[0,1],**kw)
        for key in ('final_state','initial_state_gradients','spikes'):np.testing.assert_allclose(actual[key],expected[key],rtol=9e-4,atol=6e-6)
        for a,c in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,c,rtol=9e-4,atol=6e-6)
        assert 'mpi-dynamic' in actual['numeric_profile']
        if engine!='cpu':assert actual['gpu_dispatches']>0
        path=tmp_path/'implicit.json';trainer.store(path);restored=NativeLIFTrainer(p,runner=RUNNER);restored.restore(path);trainer=restored
        assert trainer.clock_tick==cpu.clock_tick and trainer.noise_sequence==cpu.noise_sequence
    before=snapshot(trainer);bad=np.tile(bundle.initial_state,(2,1));bad[1,bundle.provenance['dynamic_state_layout']['implicit_syn']['pick'][1]]=-1
    with pytest.raises(ValueError):trainer.step(xx,[0,1],initial=bad)
    assert snapshot(trainer)==before


def test_scalar_temporary_reassignment_keeps_brian_rejection():
    from brian2_rust import TrainingConversionError
    from brian2.core.base import BrianObjectException
    net,inp,a,c,syn,x,_=model(flag_on=False)
    syn.pre.code='pick=1-pick\nnot_refractory=False\npeer+=w\nnot_refractory=True\npeer+=.5*w'
    with pytest.raises(TrainingConversionError,match='scalar variables'):
        lower_brian_dynamic_training(net,input_group=inp,layers=[a,c])
    with pytest.raises(BrianObjectException) as error:net.run(0*b.ms,namespace={})
    assert isinstance(error.value.__cause__,SyntaxError)
    assert 'scalar variables' in str(error.value.__cause__)
