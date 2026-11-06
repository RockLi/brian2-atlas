"""Ordered differentiable threshold margins in native dynamic training."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_dynamic_gpu import backend,compare
from test_training_stochastic import normal


def bank(bundle,obj,name):
    entry=next(e for e in bundle.provenance['bindings'] if e['object']==obj.name and name in e['variables'])
    return entry['bank'],entry['variables'].index(name)


def model(method='euler',comparison='>',noisy=False,refractory=False,subexpression=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    drive=b.TimedArray([[.2,.5],[.7,.3],[.4,.8]],dt=.4*b.ms,name='threshold_drive')
    x=np.array([[1],[0],[1],[1],[0],[1],[0],[1]],float);ts=np.nonzero(x[:,0])[0]
    inp=b.SpikeGeneratorGroup(1,np.zeros(len(ts),int),ts*dt,dt=dt,name='threshold_input')
    groups=[]
    rhs='theta+.2*a+.1*drive(t,i)+.05*sin(t/ms)'
    for layer in range(2):
        right='cutoff' if subexpression else rhs
        threshold='v+.1*a '+comparison+' '+right if comparison in ('>','>=') else right+' '+comparison+' v+.1*a'
        equations='dv/dt=(-v+.12*a)/ms'+('+.08*xi/ms**.5' if noisy else '')+':1'+(' (unless refractory)' if refractory else '')+'\nda/dt=-a/(2*ms):1\ntheta:1 (constant)'
        if subexpression:equations+='\ncutoff='+rhs+':1'
        g=b.NeuronGroup(2,equations,threshold=threshold,reset='v-=.6\na+=.15',method=method,dt=dt,
            refractory=.4*b.ms if refractory else False,namespace={'drive':drive},name=f'threshold_g{layer}')
        g.v=[[1.1,.7],[.9,1.15]][layer];g.a=[[.2,.35],[.15,.3]][layer];g.theta=[.6,.9];groups.append(g)
    first=b.Synapses(inp,groups[0],'w:1',on_pre='v_post+=w\na_post+=.03*w',dt=dt,name='threshold_first');first.connect();first.w=[.3,.4]
    last=b.Synapses(groups[0],groups[1],'w:1',on_pre='v_post+=w\na_post+=.03*w',dt=dt,name='threshold_last');last.connect();last.w=[.2,.25,.3,.22]
    net=b.Network(inp,*groups,first,last)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,
        trainable_neuron_parameters={g.name:['theta'] for g in groups},**options)
    return net,groups,[first,last],drive,x,bundle


@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('comparison',['>','>=','<','<='])
def test_thresholds_match_real_brian(method,comparison):
    net,groups,synapses,drive,x,bundle=model(method,comparison,refractory=method=='rk2',subexpression=method=='rk4')
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*.2*b.ms,namespace={})
    expected=np.zeros((len(x),4))
    for layer,(g,m) in enumerate(zip(groups,monitors)):
        expected[np.rint(m.t/(.2*b.ms)).astype(int),np.asarray(m.i[:])+layer*2]=1
        for name,slots in bundle.provenance['neuron_state_layout'][g.name].items():
            if name.startswith('__'):continue
            np.testing.assert_allclose(np.array(result['final_state'][0])[slots],np.asarray(getattr(g,name)[:]),atol=3e-13,rtol=3e-13)
    np.testing.assert_array_equal(result['spikes'][0],expected)


def oracle(bundle,groups,synapses,weights,x,anchors=None,noisy=False,sequence=0,initial=None):
    p=bundle.plan;table=np.array(weights[bundle.provenance['timed_inputs'][0]['bank']]).reshape(3,2)
    theta=[np.array(weights[bank(bundle,g,'theta')[0]]) for g in groups]
    w=[np.array(weights[bank(bundle,s,'w')[0]]) for s in synapses]
    v=np.array([[1.1,.7],[.9,1.15]]);a=np.array([[.2,.35],[.15,.3]]);history=[];spikes=[]
    if initial is not None:
        v=np.array([initial[:2],initial[4:6]]);a=np.array([initial[2:4],initial[6:8]])
    for tick,external in enumerate(x[:,0]):
        v=.8*v+.024*a;a=.9*a
        if noisy:v+=.08*np.sqrt(.2)*np.array([[normal(p['seed'],sequence,0,l,j,tick,0) for j in range(2)] for l in range(2)])
        u=table[min(tick//2,2)];margin=v-np.array(theta)-.1*a-.1*u-.05*np.sin(tick*.2)
        hard=(margin>0).astype(float);s=hard
        if anchors is not None:
            s=(anchors[tick]>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[tick]))**2*(margin-anchors[tick])
        history.append(margin.copy());spikes.append(s.reshape(-1))
        v[0]+=external*w[0];a[0]+=.03*external*w[0]
        for edge in range(4):
            i,j=divmod(edge,2);v[1,j]+=s[0,i]*w[1][edge];a[1,j]+=.03*s[0,i]*w[1][edge]
        gate=s if not p['detach_reset'] else hard if anchors is None else (anchors[tick]>0).astype(float)
        v-=.6*gate;a+=.15*gate
    logits=np.array(spikes)[:,2:].mean(axis=0)*p['logit_scale']
    return np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0],history,v,a


@pytest.mark.parametrize('noisy',[False,True])
@pytest.mark.parametrize('detach',[False,True])
def test_dynamic_threshold_vjp_independent(noisy,detach):
    _,groups,synapses,_,x,bundle=model(noisy=noisy,detach_reset=detach)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    loss,anchors,v,a=oracle(bundle,groups,synapses,bundle.weights,x,noisy=noisy,sequence=7)
    assert result['loss']==pytest.approx(loss,abs=2e-13)
    np.testing.assert_allclose(result['final_membrane'][0],v.reshape(-1),atol=2e-13)
    for index,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[index][j]+=1e-6;lo[index][j]-=1e-6
            fd=(oracle(bundle,groups,synapses,hi,x,anchors,noisy,7)[0]-oracle(bundle,groups,synapses,lo,x,anchors,noisy,7)[0])/2e-6
            assert result['gradients'][index][j]==pytest.approx(fd,rel=4e-4,abs=4e-7)
    for j in range(8):
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(oracle(bundle,groups,synapses,bundle.weights,x,anchors,noisy,7,hi)[0]-oracle(bundle,groups,synapses,bundle.weights,x,anchors,noisy,7,lo)[0])/2e-6
        assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=4e-4,abs=4e-7)
    for slots in bundle.provenance['threshold_margin_layout'].values():
        np.testing.assert_array_equal(np.array(result['initial_state_gradients'][0])[slots],0)


@pytest.mark.parametrize('noisy,refractory',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_threshold_gpu_mpi(noisy,refractory,ranks,backend):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(noisy=noisy,refractory=refractory,detach_reset=False,tbptt_window=3)
    kwargs=dict(noise_sequence=9) if noisy else {}
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],**kwargs)
    bundle.plan.update(backend=backend,mpi_ranks=ranks)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],**kwargs)
    compare(result,cpu,backend)



def boundary_model(comparison='>',theta=.5,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>1',reset='v=0',dt=.2*b.ms,method='euler')
    g=b.NeuronGroup(2,'dv/dt=0/second:1\ntheta:1 (constant)',threshold='v '+comparison+' theta',reset='v=v',dt=.2*b.ms,method='euler')
    g.v=[theta,theta+.25];g.theta=theta
    bundle=lower_brian_dynamic_training(b.Network(inp,hidden,g),input_group=inp,layers=[hidden,g],
        trainable_neuron_parameters={g.name:['theta']},**options)
    return g,bundle


@pytest.mark.parametrize('engine',['cpu','metal'])
@pytest.mark.parametrize('comparison',['>','>=','<','<='])
def test_exact_boundary_and_comparison_sign(comparison,engine):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    g,bundle=boundary_model(comparison,backend=engine);p=bundle.plan
    result=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients([[[0]]],[0])
    spikes={'>':[0,1],'>=':[1,1],'<':[0,0],'<=':[1,0]}[comparison]
    np.testing.assert_array_equal(result['spikes'][0][0],[0,*spikes])
    logits=np.array(spikes)*p['logit_scale'];prob=np.exp(logits-logits.max());prob/=prob.sum()
    expected=(prob-[1,0])*p['logit_scale']*p['surrogate']['scale']/(1+p['surrogate']['slope']*np.array([0,.25]))**2
    if comparison in ('<','<='):expected=-expected
    np.testing.assert_allclose(result['initial_gradients'][0][1:],expected,rtol=1e-5,atol=2e-6)
    np.testing.assert_allclose(result['gradients'][bank(bundle,g,'theta')[0]],-expected,rtol=1e-5,atol=2e-6)


@pytest.mark.parametrize('engine',['cpu','metal'])
def test_optimizer_can_cross_zero_threshold(engine,tmp_path):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    g,bundle=boundary_model('>',theta=.001,backend=engine,optimizer='sgd',learning_rate=.1)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step([[[0]]],[0]);source=bank(bundle,g,'theta')[0]
    assert trainer.state['weights'][source][0]<0
    path=tmp_path/'threshold.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(path)
    continued=restored.evaluate([[[0]]],[0],initial='carry')
    np.testing.assert_array_equal(continued['spikes'][0][0][1:],np.array(g.v[:])>np.array(trainer.state['weights'][source]))


@pytest.mark.parametrize('issue',['normal_flag','inclusive_without_margin','voltage_as_margin','unprepared','wrong_owner','detached','initial_binding'])
def test_margin_validation_rejects_invalid_plans_atomically(issue):
    g,bundle=boundary_model();spec=bundle.plan['dynamic']
    action=next(a for a in spec['actions'] if a.get('threshold_margin'))
    cell=action['reads'][0];writer=next(a for a in spec['actions'] if cell in a['writes'])
    if issue=='normal_flag':writer['threshold_margin']=True
    elif issue=='inclusive_without_margin':action.update(threshold_margin=False,threshold_inclusive=True)
    elif issue=='voltage_as_margin':action['reads'][0]=spec['voltage'][action['threshold']]
    elif issue=='unprepared':spec['actions'].remove(writer)
    elif issue=='wrong_owner':writer['owner']=0
    elif issue=='detached':spec['detached'][cell]=True
    elif issue=='initial_binding':spec['initial_parameters'][cell]=[0,0]
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step([[[0]]],[0])
    assert trainer.state==before and trainer.neuron_state is None


@pytest.mark.parametrize('engine,ranks',[('cpu',None),('cpu',2),('metal',None),('metal',2)])
def test_margin_carry_checkpoint_and_threshold_input_update(engine,ranks,tmp_path):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(noisy=True,refractory=True,backend=engine,mpi_ranks=ranks)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    whole=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0],noise_sequence=7)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(x[None,:3],[0],noise_sequence=7)
    path=tmp_path/'margin.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(path)
    tail=restored.evaluate(x[None,3:],[0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=2e-5,atol=3e-6)
    source=bundle.provenance['timed_inputs'][0]['bank'];before=copy.deepcopy(restored.neuron_state)
    restored.update_timed_input(source,[-100.]*len(bundle.weights[source]))
    assert restored.neuron_state==before and restored.clock_tick==3 and restored.noise_sequence==7
    changed=restored.evaluate(x[None,3:],[0],initial='carry')
    assert np.count_nonzero(changed['spikes'])>0 and not np.array_equal(changed['spikes'],tail['spikes'])


@pytest.mark.parametrize('theta',[0.,-.5])
@pytest.mark.parametrize('engine',['cpu','metal'])
def test_zero_and_negative_literal_threshold(theta,engine):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    b.set_device('runtime');b.start_scope()
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms)
    groups=[b.NeuronGroup(2,'dv/dt=0/second:1',threshold=f'v>{theta}',reset='v=v',dt=.2*b.ms,method='euler') for _ in range(2)]
    for g in groups:g.v=[theta,theta+.25]
    bundle=lower_brian_dynamic_training(b.Network(inp,*groups),input_group=inp,layers=groups,backend=engine)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate([[[0]]],[0])
    np.testing.assert_array_equal(result['spikes'][0][0],[0,1,0,1])


def test_time_only_threshold_with_time_units_matches_brian():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>1',reset='v=0',dt=.2*b.ms,method='euler')
    g=b.NeuronGroup(2,'dv/dt=0*volt/second:volt\nonset:second (constant)',threshold='t>=onset',reset='v-=.1*mV',dt=.2*b.ms,method='euler')
    g.onset=[.2,.4]*b.ms;net=b.Network(inp,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],trainable_neuron_parameters={g.name:['onset']})
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(np.zeros((1,5,1)),[0])
    monitor=b.SpikeMonitor(g);net.add(monitor);net.run(1*b.ms,namespace={})
    expected=np.zeros((5,2));expected[np.rint(monitor.t/(.2*b.ms)).astype(int),monitor.i[:]]=1
    np.testing.assert_array_equal(np.array(result['spikes'][0])[:,1:],expected)
    np.testing.assert_allclose(result['final_membrane'][0][1:],np.asarray(g.v[:]),atol=2e-15)
    np.testing.assert_array_equal(result['initial_gradients'],0)
    assert np.linalg.norm(result['gradients'][bank(bundle,g,'onset')[0]])>0


@pytest.mark.parametrize('engine',['cpu','metal'])
def test_threshold_through_shared_synaptic_link(engine):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    from test_training_cross_linked import model as cross_model,bank as cross_bank
    net,inp,groups,source,link,x,_=cross_model('neuron')
    groups[1].events['spike']='v>.5+peer'
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine,detach_reset=False,
        trainable_synapse_parameters={source.name:['w','u'],link.name:['w']})
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    source_bank=cross_bank(bundle,source,'u');u=bundle.weights[source_bank][0]
    def reference(value,anchors=None):
        a=np.array([1.7,1.3]);c=np.array([1.2,.9]);w=np.array(bundle.weights[cross_bank(bundle,source,'w')]);lw=bundle.weights[cross_bank(bundle,link,'w')]
        saved=[];spikes=[]
        for t in range(len(x)):
            a*=.8;c=.8*c+.02*value;margin=np.r_[a-1,c-(.5+value)]
            s=(margin>0).astype(float)
            if anchors is not None:
                old=anchors[t];p=bundle.plan;s=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
            saved.append(margin);spikes.append(s)
            for edge in range(4):
                gate=x[t,edge//2];a[edge%2]+=gate*(w[edge]+.1*value);w[edge]+=gate*.02*value
            for edge in range(4):c[edge%2]+=s[edge//2]*(lw[edge]+.2*value)
            a-=.3*s[:2];c-=.3*s[2:]
        logits=np.array(spikes)[:,2:].mean(axis=0)*bundle.plan['logit_scale']
        return np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0],saved
    loss,anchors=reference(u);fd=(reference(u+1e-6,anchors)[0]-reference(u-1e-6,anchors)[0])/2e-6
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    assert result['gradients'][source_bank][0]==pytest.approx(fd,rel=1e-3,abs=3e-5)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*.2*b.ms,namespace={})
    spikes=np.zeros((len(x),4))
    for l,m in enumerate(monitors):spikes[np.rint(m.t/(.2*b.ms)).astype(int),l*2+np.asarray(m.i[:])]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
