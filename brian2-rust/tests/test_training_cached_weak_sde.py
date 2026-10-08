"""Independent cached Poisson weak derivatives with SDE and delayed plasticity."""
import copy
import gc
import math
import os
from unittest.mock import patch

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_neuron_noise import cython_cache
from test_training_poisson_event_replay import observation, identity
from test_training_poisson_zero_vjp import mpi
from test_training_stochastic import normal


def model(backend='cpu',ranks=None,warm=0,fast=False,window=None):
    gc.collect();b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='weak_cache_input')
    groups=[]
    for k in range(2):
        group=b.NeuronGroup(2,'''dv/dt=(-v+.1*h+gain*q)/ms+sigma*xi_shared/sqrt(ms):1
dh/dt=(-h+.2*v+.1*q)/ms+.5*sigma*xi_shared/sqrt(ms):1
q=poisson(lam):1 (constant over dt)
lam:1 (constant)
gain:1 (constant)
sigma:1 (constant)
theta:1 (constant)''',threshold='v>theta+.03*q',
            reset='v-=.3+.02*q;h+=.01*q',method='euler',dt=dt,name=f'weak_cache_{k}')
        group.v=[.9,.8];group.h=[.1,.2];group.q=[.2,.3]
        group.lam=[0.,1.25];group.gain=[1.5,.45];group.sigma=[.035,.045];group.theta=[.45,.55]
        group.subexpression_updater.when='before_start'
        group.subexpression_updater._clock=b.Clock(dt=(.4 if k==0 else .1 if fast else .2)*b.ms)
        groups.append(group)
    syn=b.Synapses(*groups,'w:1',on_pre='w=.95*w+.015*q_pre;v_post+=w;h_post+=.01*q_pre',
                  on_post='w+=.002*q_post',dt=dt,name='weak_cache_s')
    syn.connect(j='i');syn.w=[.14,.19];syn.pre.delay=.4*b.ms
    net=b.Network(source,*groups,syn)
    if warm:
        b.seed(997);net.run(warm*dt,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,
        trainable_neuron_parameters={g.name:['lam','gain','sigma','theta'] for g in groups},
        backend=backend,mpi_ranks=ranks,detach_reset=False,tbptt_window=window,
        seed=731,learning_rate=1e-9)
    return net,groups,syn,bundle


def parameter_banks(bundle):
    return {(row['object'],row['variables'][0]):row['bank']
            for row in bundle.provenance['bindings'] if len(row['variables'])==1}


def cached_domain(bundle,k):
    matches=[row['noise_domain'] for name,row in bundle.provenance['regular_runner_layout'].items()
             if name.startswith(f'weak_cache_{k}_subexpression_update')]
    assert len(matches)==1
    return matches[0]


def oracle(bundle,*,weights=None,initial=None,length=6,warm=0,fast=False,
           anchors=None,forced=None,counts=None,sequence=9):
    """Physical equations and FIFO; no native IR/action/tape interpretation."""
    plan=bundle.plan;weights=bundle.weights if weights is None else weights
    state=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for cell,binding in enumerate(plan['dynamic']['initial_parameters']):
            if binding is not None:state[cell]=weights[binding[0]][binding[1]]
    banks=parameter_banks(bundle)
    layouts=[bundle.provenance['neuron_state_layout'][f'weak_cache_{k}'] for k in range(2)]
    edge=bundle.provenance['dynamic_state_layout']['weak_cache_s']['w']
    periods=[4,1 if fast else 2]
    domains=[cached_domain(bundle,k) for k in range(2)]
    bins={};pending=bundle.provenance['delay_queues']['weak_cache_s_pre']['pending']
    for row in pending:bins.setdefault(row['remaining'],[]).append((row['edge'],1.))
    records={};uniforms=[];normals=[];spikes=[];margins=[];before=[];queue_before=[];calls=[0,0]
    for half in range(2*length):
        tick=half//2
        if half%2==0:
            if anchors is not None and plan['tbptt_window'] and tick and tick%plan['tbptt_window']==0:
                state=anchors['before'][tick].copy();bins=copy.deepcopy(anchors['queue_before'][tick])
            before.append(state.copy());queue_before.append(copy.deepcopy(bins))
        for k,layout in enumerate(layouts):
            absolute=half+2*warm
            if absolute%periods[k]:continue
            # A fresh conversion keeps physical pending times but starts a
            # new random execution sequence at call zero on each clock.
            instant=calls[k];calls[k]+=1
            for j in range(2):
                lam=weights[banks[(f'weak_cache_{k}','lam')]][j]
                entry,draws=observation(plan['seed'],sequence,domains[k],j,instant,None,lam,clock=True)
                key=identity(entry)
                if counts is not None and key in counts:entry['count']=counts[key]
                if forced==key:entry['count']=1
                records[key]=entry
                # Brian's original zero-rate Poisson implementation returns
                # without consuming its sequential uniform buffer.
                if lam!=0.:uniforms.extend(draws)
                state[layout['q'][j]]=entry['count']
        if half%2:continue
        for k,layout in enumerate(layouts):
            gain=np.asarray(weights[banks[(f'weak_cache_{k}','gain')]])
            sigma=np.asarray(weights[banks[(f'weak_cache_{k}','sigma')]])
            noise=np.array([normal(plan['seed'],sequence,0,k,j,tick,0) for j in range(2)])
            normals.extend(noise)
            q=state[layout['q']].copy()
            v=state[layout['v']].copy();h=state[layout['h']].copy()
            state[layout['v']]=.8*v+.02*h+.2*gain*q+math.sqrt(.2)*sigma*noise
            state[layout['h']]=.8*h+.04*v+.02*q+.5*math.sqrt(.2)*sigma*noise
        margin=np.concatenate([state[layout['v']]-weights[banks[(f'weak_cache_{k}','theta')]]-.03*state[layout['q']]
                               for k,layout in enumerate(layouts)])
        event=(margin>0).astype(float)
        if anchors is not None:
            base=anchors['margins'][tick]
            event=(base>0).astype(float)+plan['surrogate']['scale']/(1+plan['surrogate']['slope']*abs(base))**2*(margin-base)
        margins.append(margin.copy());spikes.append(event.copy())
        # Keep zero-valued counterfactual gates, so finite differences include
        # delayed contributions at the original emission rather than arrival.
        for j in range(2):bins.setdefault(tick+2,[]).append((j,event[j]))
        for j,gate in bins.pop(tick,[]):
            old=state[edge[j]];q=state[layouts[0]['q'][j]];new=.95*old+.015*q
            state[edge[j]]=old+gate*(new-old)
            state[layouts[1]['v'][j]]+=gate*new
            state[layouts[1]['h'][j]]+=gate*.01*q
        state[edge]+=.002*state[layouts[1]['q']]*event[2:]
        for k,layout in enumerate(layouts):
            gate=event[2*k:2*k+2];q=state[layout['q']]
            state[layout['v']]-=gate*(.3+.02*q);state[layout['h']]+=gate*.01*q
    spikes=np.asarray(spikes);logits=spikes[:,2:].mean(0)*plan['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return float(loss),state,spikes,dict(before=before,queue_before=queue_before,margins=margins,
        records=records,uniforms=uniforms,normals=normals)


@pytest.mark.parametrize('warm,fast',[(0,False),(1,True)])
def test_cached_trainable_poisson_sde_delayed_forward(engine,warm,fast):
    _,_,_,bundle=model(engine,warm=warm,fast=fast)
    loss,state,spikes,_=oracle(bundle,warm=warm,fast=fast)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,6,2)),[0],noise_sequence=9)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    tolerance=5e-5 if engine!='cpu' else 4e-12
    assert result['loss']==pytest.approx(loss,abs=tolerance)
    for layout in bundle.provenance['neuron_state_layout'].values():
        for name in ('v','h','q'):
            np.testing.assert_allclose(np.asarray(result['final_state'])[0,layout[name]],state[layout[name]],rtol=tolerance,atol=tolerance*.02)
    slots=bundle.provenance['dynamic_state_layout']['weak_cache_s']['w']
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],state[slots],rtol=tolerance,atol=tolerance*.02)


@pytest.mark.parametrize('warm,fast',[(0,False),(1,True)])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_cached_poisson_weak_and_all_pathwise_derivatives(engine,warm,fast,window,ranks):
    mpi(ranks);_,_,_,bundle=model(engine,ranks,warm,fast,window)
    expected,state,spikes,anchors=oracle(bundle,warm=warm,fast=fast)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,
                             request_timeout=300)
    result=trainer.gradients(np.zeros((1,6,2)),[0],noise_sequence=9)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    tolerance=6e-4 if engine!='cpu' else 6e-6
    assert result['loss']==pytest.approx(expected,abs=tolerance*.1)
    saved_counts={key:entry['count'] for key,entry in anchors['records'].items()}
    banks=parameter_banks(bundle)
    rate_banks={banks[(f'weak_cache_{k}','lam')]:cached_domain(bundle,k) for k in range(2)}
    zero_derivatives=[]
    for bank,row in enumerate(bundle.weights):
        for j,value in enumerate(row):
            if bank in rate_banks:
                records=[(key,entry) for key,entry in anchors['records'].items()
                         if entry['identity']['site']['domain']==rate_banks[bank]
                         and entry['identity']['site']['entity']==j]
                assert records
                if value==0.:
                    derivative=sum(oracle(bundle,warm=warm,fast=fast,forced=key)[0]-expected
                                   for key,_ in records)
                    zero_derivatives.append(derivative)
                else:
                    derivative=expected*sum(entry['count']/value-1 for _,entry in records)
            else:
                plus=copy.deepcopy(bundle.weights);minus=copy.deepcopy(bundle.weights)
                plus[bank][j]+=1e-6;minus[bank][j]-=1e-6
                derivative=(oracle(bundle,warm=warm,fast=fast,weights=plus,anchors=anchors,counts=saved_counts)[0]
                           -oracle(bundle,warm=warm,fast=fast,weights=minus,anchors=anchors,counts=saved_counts)[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(derivative,rel=4e-4,abs=tolerance),(bank,j)
    assert any(abs(value)>1e-8 for value in zero_derivatives), 'boundary fixture must change actual loss'
    physical=sorted(set(sum([row[name] for row in bundle.provenance['neuron_state_layout'].values()
                             for name in ('v','h','q')],[])+bundle.provenance['dynamic_state_layout']['weak_cache_s']['w']))
    for cell in physical:
        plus=np.asarray(bundle.initial_state).copy();minus=plus.copy()
        plus[cell]+=1e-6;minus[cell]-=1e-6
        derivative=(oracle(bundle,warm=warm,fast=fast,initial=plus,anchors=anchors,counts=saved_counts)[0]
                   -oracle(bundle,warm=warm,fast=fast,initial=minus,anchors=anchors,counts=saved_counts)[0])/2e-6
        assert result['initial_state_gradients'][0][cell]==pytest.approx(derivative,rel=4e-4,abs=tolerance),cell
    # Each reached clock observation is stored once, including zero-rate and
    # idle fast-clock calls. Counterfactual replays leave the baseline intact.
    actual=result['poisson_state']['entries']
    assert {identity(entry):entry['count'] for entry in actual}==saved_counts
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('warm,fast',[(0,False),(1,True)])
def test_cached_poisson_sde_original_cython_replay(warm,fast):
    net,groups,syn,bundle=model(warm=warm,fast=fast)
    _,state,spikes,anchors=oracle(bundle,warm=warm,fast=fast)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(0*b.ms,namespace={})
    device=b.get_device();device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    used={'rand':0,'randn':0}
    def refill(kind,n):
        assert n==20000 and used[kind]==0
        used[kind]+=1;values=anchors['uniforms' if kind=='rand' else 'normals']
        array=np.full(n,.5);array[:len(values)]=values;return array
    try:
        with patch('numpy.random.rand',lambda n:refill('rand',n)),patch('numpy.random.randn',lambda n:refill('randn',n)):
            net.run(1.2*b.ms,namespace={})
        assert used=={'rand':1,'randn':1}
        assert device.rand_buffer_index[0]==len(anchors['uniforms'])
        assert device.randn_buffer_index[0]==len(anchors['normals'])
    finally:
        device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    actual=np.zeros((6,4))
    for k,(group,monitor) in enumerate(zip(groups,monitors)):
        ticks=np.rint(np.asarray(monitor.t/b.ms)/.2).astype(int)-warm
        actual[ticks,2*k+np.asarray(monitor.i)]=1.
        assert group.state_updater.codeobj.compiled_code['run'] is not None
        assert group.subexpression_updater.codeobj.compiled_code['run'] is not None
        for name in ('v','h','q'):
            layout=bundle.provenance['neuron_state_layout'][group.name][name]
            np.testing.assert_allclose(np.asarray(getattr(group,name)),state[layout],rtol=5e-12,atol=5e-14)
    layout=bundle.provenance['dynamic_state_layout'][syn.name]['w']
    np.testing.assert_allclose(np.asarray(syn.w),state[layout],rtol=5e-12,atol=5e-14)
    np.testing.assert_array_equal(actual,spikes)


def hard_loss(spikes,scale):
    logits=np.asarray(spikes)[:,2:].mean(0)*scale
    return float(np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0])


@pytest.mark.parametrize('warm,fast',[(0,False),(1,True)])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_cached_weak_sde_restore_readonly_invalid_new_rate_atomicity(engine,warm,fast,ranks,tmp_path):
    mpi(ranks);_,_,_,bundle=model(engine,ranks,warm,fast)
    plan=copy.deepcopy(bundle.plan);plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    inputs=np.zeros((1,6,2));_,_,spikes,anchors=oracle(bundle,warm=warm,fast=fast)
    full=trainer.evaluate(inputs,[0],noise_sequence=9)
    head=trainer.step(inputs[:,:3],[0],noise_sequence=9)
    path=tmp_path/'head.json';trainer.store(path)
    saved=path.read_bytes()
    restored=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER,request_timeout=300);restored.restore(path)
    tail=restored.gradients(inputs[:,3:],[0],initial='carry')
    again=restored.gradients(inputs[:,3:],[0],initial='carry')
    assert again==tail
    same=tmp_path/'readonly.json';restored.store(same);assert same.read_bytes()==saved
    assert tail['poisson_state']==full['poisson_state']
    for field in ('next_tick','ticks','calls'):
        assert tail['clock_state'][field]==full['clock_state'][field]
    assert tail['clock_state']['initial_calls']==head['clock_state']['calls']
    assert tail['clock_state']['start']==pytest.approx(plan['clock']['origin']+3*plan['clock']['dt'])
    np.testing.assert_array_equal(head['spikes'][0]+tail['spikes'][0],spikes)
    tolerance=7e-4 if engine!='cpu' else 7e-12
    np.testing.assert_allclose(tail['final_state'],full['final_state'],rtol=tolerance,atol=tolerance*.02)
    banks=parameter_banks(bundle)
    incoming={identity(entry) for entry in head['poisson_state']['entries']}
    baseline=hard_loss(spikes[3:],plan['logit_scale'])
    for k in range(2):
        bank=banks[(f'weak_cache_{k}','lam')];domain=cached_domain(bundle,k)
        for j,rate in enumerate(bundle.weights[bank]):
            new=[(key,entry) for key,entry in anchors['records'].items() if key not in incoming
                 and entry['identity']['site']['domain']==domain and entry['identity']['site']['entity']==j]
            if rate==0.:
                expected=sum(hard_loss(oracle(bundle,warm=warm,fast=fast,forced=key)[2][3:],plan['logit_scale'])-baseline
                             for key,_ in new)
            else:expected=baseline*sum(entry['count']/rate-1 for _,entry in new)
            assert tail['gradients'][bank][j]==pytest.approx(expected,abs=max(tolerance,6e-6),rel=5e-4)
    # The slower cached rate is not visited at the .6ms head boundary. A saved
    # cache may be consumed with an edited invalid live rate until refresh.
    if not warm:
        restored.state['weights'][banks[('weak_cache_0','lam')]][1]=-1.
        restored.evaluate(inputs[:,:1],[0],initial='carry')
        error_snapshot=tmp_path/'before-error.json';restored.store(error_snapshot)
        before=error_snapshot.read_bytes()
        with pytest.raises(ValueError,match='Poisson|poisson|rate|GPU'):
            restored.step(inputs[:,:2],[0],initial='carry')
        restored.store(error_snapshot);assert error_snapshot.read_bytes()==before
