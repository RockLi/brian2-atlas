"""State-dependent refractory expressions with detached gates and live durations."""
import copy
import ast
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training, TrainingConversionError
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine
from test_training_delay_update import snapshot


def model(kind='boolean',method='euler',warm=False,noisy=False,linked=False,refexpr=None,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,0],[0,1],[1,1],[1,0],[0,1],[1,1],[0,1],[1,0]],float)
    ticks,ids=np.nonzero(x);inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='ref_input')
    g=b.NeuronGroup(2,'dv/dt=(drive-v+.15*u)/ms'+('+sigma*xi/sqrt(ms)' if noisy else '')+':1 (unless refractory)\n'
        'du/dt=-u/ms:1\ndr/dt=-r/(2*ms):second\ndrive:1 (constant)\nlimit:1 (constant)\nshift:second (constant)'+('\nsigma:1 (constant)' if noisy else '')+('\npeer:1 (linked)\npick:integer' if linked else ''),
        threshold='v>.6',reset='v-=.4\nu+=.8\nr+=.4*ms'+('\npick=1-pick' if linked else ''),
        refractory=refexpr if refexpr is not None else ('peer > limit' if linked else 'u > limit') if kind=='boolean' else ('r+shift+.1*peer*ms' if linked else 'r+shift'),method=method,dt=dt,name='ref_group')
    g.v=[.7,.3];g.u=[.1,.2];g.r=[.75,1.2]*b.ms;g.drive=[1.1,.9];g.limit=[.35,.45];g.shift=[.1,.15]*b.ms
    if noisy:g.sigma=[.04,.06]
    syn=b.Synapses(inp,g,'w:1',on_pre='v_post+=w\nr_post+=.1*ms\nw+=.01',on_post='w-=.002*u_post',dt=dt,name='ref_syn');syn.connect(j='i');syn.w=[.12,.16]
    hidden=b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>100',reset='v=0',method=method,dt=dt,name='ref_hidden')
    if linked:
        hidden.v=[1.2,.2];g.pick=[0,1];g.peer=b.linked_var(hidden,'v',index='pick')
    net=b.Network(inp,hidden,g,syn)
    if warm:net.run(2*dt,namespace={});x=x[2:]
    control='limit' if kind=='boolean' else 'shift'
    used=None if refexpr is None else {n.id for n in ast.walk(ast.parse(refexpr,mode='eval')) if isinstance(n,ast.Name)}
    chosen=['drive']+([control] if used is None or control in used else [])+(['sigma'] if noisy else [])
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],trainable_neuron_parameters={g.name:chosen},**options)
    return net,inp,g,syn,x,bundle


def assert_matches_brian(engine,kind,method='euler',warm=False,**options):
    net,inp,g,syn,x,bundle=model(kind,method,warm,backend=engine,**options)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    monitor=b.SpikeMonitor(g);net.add(monitor);net.run(len(x)*.2*b.ms,namespace={})
    assert all(r.codeobj.compiled_code['run'] is not None for r in (g.state_updater,g.thresholder['spike'],g.resetter['spike'],syn.pre,syn.post))
    z=np.asarray(out['final_state'])[0];layout=bundle.provenance['neuron_state_layout'][g.name];tol=1e-12 if engine=='cpu' else 8e-6
    for name in ('v','u','r'):np.testing.assert_allclose(z[layout[name]],np.asarray(getattr(g,name)[:]),rtol=tol,atol=tol*1e-3)
    np.testing.assert_allclose(z[bundle.provenance['dynamic_state_layout'][syn.name]['w']],syn.w[:],rtol=tol,atol=tol)
    np.testing.assert_array_equal(z[bundle.provenance['refractory_activity_layout'][g.name]],g.not_refractory[:])
    np.testing.assert_allclose(z[bundle.provenance['refractory_lastspike_layout'][g.name]],np.asarray(g.lastspike[:]),rtol=tol,atol=tol*1e-3)
    expected=np.zeros((len(x),2));origin=bundle.plan['clock']['origin'];expected[np.rint((np.asarray(monitor.t/b.second)-origin)/.0002).astype(int),np.asarray(monitor.i)]=1
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,2:],expected)
    if engine!='cpu':assert out['gpu_dispatches']>0


@pytest.mark.parametrize('kind',['boolean','duration'])
@pytest.mark.parametrize('method',['euler','rk4'])
@pytest.mark.parametrize('warm',[False,True])
def test_state_refractory_matches_compiled_brian(engine,kind,method,warm):
    assert_matches_brian(engine,kind,method,warm)


@pytest.mark.parametrize('kind',['boolean','duration'])
def test_refractory_runtime_link_matches_brian(engine,kind):
    assert_matches_brian(engine,kind,linked=True)


@pytest.mark.parametrize('expression',[
    'not_refractory or u>limit',
    '(t-lastspike < .6*ms) and (u>limit)',
    '(.6*ms > t-lastspike) and (u>limit)',
    '((t-lastspike == .6*ms) or (t-lastspike < .6*ms)) and (u>limit)',
])
def test_refractory_explicit_flag_and_timestamp(engine,expression):
    assert_matches_brian(engine,'boolean',warm=True,refexpr=expression)


def reference(bundle,x,kind,*,weights=None,initial=None,anchors=None,start_tick=0,noise_sequence=0,batch=0):
    """Independent Euler recurrence; freeze only discontinuous control decisions."""
    from test_training_stochastic import normal
    p=bundle.plan;d=p['dynamic'];weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for k,ref in enumerate(d['initial_parameters']):
            if ref is not None:z[k]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'];v=layout['ref_group']['v'];u=layout['ref_group']['u'];r=layout['ref_group']['r'];hv=layout['ref_hidden']['v']
    flag=bundle.provenance['refractory_activity_layout']['ref_group'];age=bundle.provenance['refractory_age_layout']['ref_group'];last=bundle.provenance['refractory_lastspike_layout']['ref_group']
    wi=bundle.provenance['dynamic_state_layout']['ref_syn']['w']
    words=bundle.provenance['refractory_timestamp_words']['ref_group']
    def param(name):
        entry=next(e for e in bundle.provenance['bindings'] if e['object']=='ref_group' and name in e['variables'])
        return np.asarray(weights[entry['bank']])
    drive=param('drive');control=param('limit' if kind=='boolean' else 'shift')
    sigma=param('sigma') if p.get('noise_streams') else None
    before=[];gates=[];voltages=[];hard_spikes=[];spikes=[]
    for t,inp in enumerate(x):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors['before'][t].copy()
        before.append(z.copy());free=(z[flag].astype(bool)|(z[u]<=control)) if kind=='boolean' else z[age]>=np.trunc((z[r]+control+.0000002)/.0002)
        if anchors is not None:free=anchors['gates'][t]
        gates.append(free.copy());z[hv]*=.8
        value=.8*z[v]+.2*drive+.03*z[u]
        if sigma is not None:value+=sigma*np.sqrt(.2)*np.array([normal(p['seed'],noise_sequence,batch,1,j,t+start_tick,0) for j in range(2)])
        z[v]=np.where(free,value,z[v]);z[u]*=.8;z[r]*=.9
        z[age]=np.minimum(z[age],2**31-2)+1
        voltage=z[v].copy();hard=((voltage>.6)&free).astype(float);soft=hard.copy()
        if anchors is not None:
            hard=anchors['hard'][t];old=anchors['voltage'][t]
            soft=hard+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old-.6))**2*(voltage-old)
        voltages.append(voltage);hard_spikes.append(hard.copy());spikes.append(soft)
        z[flag]=free*(1-hard)
        for j in range(2):
            if hard[j]:
                timestamp=(round(p['clock']['origin']/.0002)+t+start_tick)*.0002
                z[age[j]]=1;z[last[j]]=timestamp;bits=np.float64(timestamp).view(np.uint64).item()
                for word in (0,1):z[words[word][j]]=((bits>>(32*word))&0xffffffff)-((1<<32) if (bits>>(32*word))&(1<<31) else 0)
        for j in range(2):
            if inp[j]:
                if z[flag[j]]:z[v[j]]+=z[wi[j]]
                z[r[j]]+=.0001;z[wi[j]]+=.01
        z[wi]-=.002*z[u]*soft
        reset=hard if p['detach_reset'] else soft
        z[v]-=.4*reset;z[u]+=.8*reset;z[r]+=.0004*reset
    logits=np.asarray(spikes).mean(axis=0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.asarray(spikes),dict(before=before,gates=gates,voltage=voltages,hard=hard_spikes)


@pytest.mark.parametrize('kind',['boolean','duration'])
@pytest.mark.parametrize('detach,window',[(True,None),(False,3)])
def test_state_refractory_all_float_derivatives(engine,kind,detach,window):
    *_,x,bundle=model(kind,noisy=True,backend=engine,detach_reset=detach,tbptt_window=window)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    loss,z,spikes,anchors=reference(bundle,x,kind)
    tol=3e-5 if engine=='cpu' else 2e-3;absolute=2e-7 if engine=='cpu' else 9e-6
    assert out['loss']==pytest.approx(loss,abs=absolute)
    np.testing.assert_allclose(out['final_state'][0],z,rtol=tol,atol=absolute)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,2:],spikes)
    eps=1e-6
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(reference(bundle,x,kind,weights=hi,anchors=anchors)[0]-reference(bundle,x,kind,weights=lo,anchors=anchors)[0])/(2*eps)
            assert out['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=absolute)
    for k,detached in enumerate(bundle.plan['dynamic']['detached']):
        if detached:assert out['initial_state_gradients'][0][k]==0.;continue
        hi=np.asarray(bundle.initial_state).copy();lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(reference(bundle,x,kind,initial=hi,anchors=anchors)[0]-reference(bundle,x,kind,initial=lo,anchors=anchors)[0])/(2*eps)
        assert out['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=absolute)


@pytest.mark.parametrize('kind',['boolean','duration'])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_state_refractory_optimizer_carry_checkpoint(engine,kind,ranks,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(kind,noisy=True,backend=engine,mpi_ranks=ranks,detach_reset=False,tbptt_window=2)
    p=bundle.plan;q=copy.deepcopy(p);q.update(backend='cpu',mpi_ranks=None)
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);cpu=NativeLIFTrainer(q,runner=RUNNER,weights=bundle.weights)
    xx=np.stack([x,x[:,::-1]]);carried=[None,None]
    for t in (0,4):
        chunk=xx[:,t:t+4];kw=dict(noise_sequence=7) if t==0 else dict(initial='carry')
        expected=[reference(bundle,chunk[i],kind,weights=trainer.state['weights'],initial=carried[i],start_tick=t,noise_sequence=7,batch=i)[1] for i in range(2)]
        actual=trainer.step(chunk,[0,1],**kw);baseline=cpu.step(chunk,[0,1],**kw)
        np.testing.assert_allclose(actual['final_state'],expected,rtol=2e-3,atol=9e-6);carried=expected
        for key in ('final_state','spikes','initial_state_gradients'):np.testing.assert_allclose(actual[key],baseline[key],rtol=2e-3,atol=9e-6)
        for a,c in zip(actual['gradients'],baseline['gradients']):np.testing.assert_allclose(a,c,rtol=2e-3,atol=9e-6)
        if ranks is not None:assert 'mpi-dynamic' in actual['numeric_profile']
        if engine!='cpu':assert actual['gpu_dispatches']>0
        path=tmp_path/'refractory.json';trainer.store(path);restored=NativeLIFTrainer(p,runner=RUNNER);restored.restore(path);trainer=restored
        assert trainer.clock_tick==cpu.clock_tick and trainer.noise_sequence==cpu.noise_sequence
    assert any(a!=c for a,c in zip(trainer.state['weights'],bundle.weights))
    before=snapshot(trainer);bad=np.asarray(trainer.neuron_state).copy()
    if kind=='duration':bad[1,bundle.provenance['neuron_state_layout']['ref_group']['r'][1]]=1e20
    else:bad[1,bundle.provenance['refractory_activity_layout']['ref_group'][1]]=2
    with pytest.raises(ValueError):trainer.step(xx,[0,1],initial=bad)
    assert snapshot(trainer)==before


@pytest.mark.parametrize('kind',['boolean','duration'])
@pytest.mark.parametrize('ranks',[2,8])
def test_refractory_link_nonroot_runtime_failure_is_atomic(engine,kind,ranks):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(kind,linked=True,backend=engine,mpi_ranks=ranks)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);xx=np.stack([x,x])
    trainer.step(xx,[0,1]);before=snapshot(trainer);bad=np.asarray(trainer.neuron_state).copy()
    bad[1,bundle.provenance['neuron_state_layout']['ref_group']['pick'][1]]=2
    bad[1,bundle.provenance['refractory_activity_layout']['ref_group'][1]]=0
    with pytest.raises(ValueError):trainer.step(xx,[0,1],initial=bad)
    assert snapshot(trainer)==before


@pytest.mark.parametrize('expression',['u','v+ms','1','sin(v)*ms + v'])
def test_refractory_rejects_nonboolean_and_bad_dimensions(expression):
    with pytest.raises(TrainingConversionError,match='refractory'):
        model('boolean',refexpr=expression)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_refractory_elapsed_counter_saturates_exactly(engine,ranks):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model('duration',backend=engine,mpi_ranks=ranks)
    p=bundle.plan;layout=bundle.provenance['neuron_state_layout']['ref_group']
    ages=bundle.provenance['refractory_age_layout']['ref_group']
    for slot,value in zip(ages,[16777217,2147483647]):p['dynamic']['initial'][slot]=float(value)
    # Keep this tick below threshold so a spike cannot reset the tested ages.
    for slot in layout['v']:p['dynamic']['initial'][slot]=-10.
    out=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients(np.zeros((1,1,2)),[0])
    np.testing.assert_array_equal(np.asarray(out['final_state'])[0,ages],[16777218,2147483647])
    np.testing.assert_array_equal(np.asarray(out['initial_state_gradients'])[0,ages],0.)
