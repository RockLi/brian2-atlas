"""One optimizer bank per linked constant, including indexed thresholds."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_dynamic_gpu import backend,compare


def binding(bundle,obj,name):
    row=next(v for v in bundle.provenance['bindings'] if v['object']==obj.name and name in v['variables'])
    return row['bank']


def model(kind='array',method='euler',**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1]],float);ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='constant_input')
    shared=kind=='shared';syn_source=kind=='synapse'
    a=b.NeuronGroup(2,'dv/dt=(-v'+('' if kind=='unused' else '+.1*k')+')/ms:1\nk:1 ('+('shared,' if shared else '')+'constant)\ntheta:1 (constant)',
        threshold='v>theta',reset='v-=.3',dt=dt,method=method,name='constant_a')
    a.k=.7 if shared else [.7,1.1];a.theta=[.95,1.3];a.v=[1.7,1.4]
    c=b.NeuronGroup(3,'dv/dt=(-v+.3*p+.1*r)/ms:1\np:1 (linked)\nr:1 (linked)\nh:1 (linked)',
        threshold='v>h',reset='v-=.2*h',dt=dt,method=method,name='constant_c');c.v=[1.6,1.1,1.4]
    d=b.NeuronGroup(2,'dv/dt=(-v+.2*q)/ms:1\nq:1 (linked)\nh:1 (linked)',
        threshold='v>h',reset='v-=.25*h',dt=dt,method=method,name='constant_d');d.v=[1.1,1.4]
    pool=b.Synapses(inp,a,'w:1\ndrive:1 (shared,constant)',on_pre='v_post+=w',dt=dt,name='constant_pool');pool.connect();pool.w=[.2,.15,.18,.24];pool.drive=.4
    src,name=(pool,'drive') if syn_source else (a,'k')
    c.p=b.linked_var(src,name,**({} if shared or syn_source else {'index':[1,0,1]}))
    c.r=b.linked_var(src,name,**({} if shared or syn_source else {'index':[0,1,0]}))
    c.h=b.linked_var(a,'theta',index=[1,0,0])
    d.q=b.linked_var(src,name,**({} if shared or syn_source else {'index':[1,0]}));d.h=b.linked_var(a,'theta',index=[0,1])
    syn=b.Synapses(a,c,'w:1\ngain:1 (linked)'+('' if shared or syn_source else '\npick:integer (constant)'),
        on_pre='v_post+=w+.05*k_pre+.02*gain',dt=dt,name='constant_link');syn.connect();syn.w=.18
    if not shared and not syn_source:syn.pick=[1,0,1,0,1,0]
    syn.gain=b.linked_var(src,name,**({} if shared or syn_source else {'index':'pick'}))
    last=b.Synapses(c,d,'w:1',on_pre='v_post+=w',dt=dt,name='constant_last');last.connect();last.w=.15
    net=b.Network(inp,a,c,d,pool,syn,last)
    frozen=kind=='frozen'
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c,d],
        trainable_neuron_parameters={a.name:[] if frozen else ['k','theta']},
        trainable_synapse_parameters={pool.name:['w','drive'] if syn_source else ['w'],syn.name:['w'],last.name:['w']},**options)
    return net,inp,[a,c,d],[pool,syn,last],x,bundle


@pytest.mark.parametrize('kind',['array','shared','synapse','unused','frozen'])
@pytest.mark.parametrize('method',['euler','rk2'])
def test_linked_constants_match_real_brian(kind,method):
    net,inp,groups,synapses,x,bundle=model(kind,method)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*.2*b.ms,namespace={})
    spikes=np.zeros((len(x),7));offset=0
    for g,m in zip(groups,monitors):
        spikes[np.rint(np.asarray(m.t/b.second)/.0002).astype(int),offset+np.asarray(m.i)]=1;offset+=len(g)
    np.testing.assert_array_equal(out['spikes'][0],spikes)
    np.testing.assert_allclose(out['final_membrane'][0],np.r_[*[g.v[:] for g in groups]],rtol=2e-13,atol=2e-13)
    owners={(v['object'],v['variable']) for v in bundle.provenance['constant_sources']}
    for obj,name in owners:
        assert sum(v['object']==obj and name in v['variables'] for v in bundle.provenance['bindings'])==1
    assert all(v['object'] not in [groups[1].name,groups[2].name] for v in bundle.provenance['bindings'])
    assert all(name=='v' for group in groups for name in bundle.provenance['neuron_state_layout'][group.name])


def oracle(bundle,groups,synapses,x,weights,anchors=None):
    p=bundle.plan;k=np.array(weights[binding(bundle,groups[0],'k')]);theta=np.array(weights[binding(bundle,groups[0],'theta')])
    w0=np.array(weights[binding(bundle,synapses[0],'w')]);w1=np.array(weights[binding(bundle,synapses[1],'w')]);w2=np.array(weights[binding(bundle,synapses[2],'w')])
    a=np.array([1.7,1.4]);c=np.array([1.6,1.1,1.4]);d=np.array([1.1,1.4]);saved=[];spikes=[]
    tc=theta[[1,0,0]];td=theta[[0,1]];th=np.r_[theta,tc,td]
    for t in range(len(x)):
        a=.8*a+.02*k;c=.8*c+.06*k[[1,0,1]]+.02*k[[0,1,0]];d=.8*d+.04*k[[1,0]];v=np.r_[a,c,d]
        if anchors is None:s=(v>th).astype(float)
        else:
            old,oldth=anchors[t];phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old-oldth))**2
            s=(old>oldth).astype(float)+phi*((v-th)-(old-oldth))
        saved.append((v,th));spikes.append(s)
        # Brian's same-order pre paths sort by object name: last, link, pool.
        for edge in range(6):d[edge%2]+=s[2+edge//2]*w2[edge]
        for edge in range(6):c[edge%3]+=s[edge//3]*(w1[edge]+.05*k[edge//3]+.02*k[[1,0,1,0,1,0][edge]])
        for edge in range(4):a[edge%2]+=x[t,edge//2]*w0[edge]
        gate=(v>th).astype(float) if p['detach_reset'] else s
        a-=.3*gate[:2];c-=.2*tc*gate[2:5];d-=.25*td*gate[5:]
    logits=np.array(spikes)[:,5:].mean(axis=0)*p['logit_scale']
    return np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0],saved


@pytest.mark.parametrize('detach',[False,True])
def test_all_tied_parameter_and_threshold_derivatives_independent(detach):
    _,_,groups,synapses,x,bundle=model(detach_reset=detach)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0]);loss,anchors=oracle(bundle,groups,synapses,x,bundle.weights)
    assert out['loss']==pytest.approx(loss,abs=2e-13)
    for bank,row in enumerate(bundle.weights):
        for i,value in enumerate(row):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][i]+=1e-6;lo[bank][i]-=1e-6
            fd=(oracle(bundle,groups,synapses,x,hi,anchors)[0]-oracle(bundle,groups,synapses,x,lo,anchors)[0])/2e-6
            assert out['gradients'][bank][i]==pytest.approx(fd,abs=4e-7,rel=4e-4)


@pytest.mark.parametrize('kind',['array','shared','synapse'])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_constant_links_gpu_and_mpi(kind,ranks,backend):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(kind,detach_reset=False,tbptt_window=3)
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    bundle.plan.update(backend=backend,mpi_ranks=ranks)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0]);compare(out,cpu,backend)


def test_large_permutation_uses_bounded_program_count():
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';n=4100;dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=dt)
    a=b.NeuronGroup(n,'dv/dt=-v/ms:1\nk:1 (constant)',threshold='v>1',reset='v=0',method='euler',dt=dt)
    c=b.NeuronGroup(n,'dv/dt=p/ms:1\np:1 (linked)',threshold='v>1',reset='v=0',method='euler',dt=dt)
    a.k=np.linspace(.2,.8,n);c.p=b.linked_var(a,'k',index=np.arange(n-1,-1,-1))
    bundle=lower_brian_dynamic_training(b.Network(inp,a,c),input_group=inp,layers=[a,c],trainable_neuron_parameters={a.name:['k']},max_tape_bytes=128*1024**2)
    assert len(bundle.plan['dynamic']['program_sets'])<10
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate([[[0]]],[0])
    np.testing.assert_allclose(out['final_membrane'][0][n:],np.array(a.k[:])[::-1]*.2,rtol=2e-14,atol=2e-14)


@pytest.mark.parametrize('issue',['missing_map','bad_index','short_map','missing_threshold','bad_threshold_ref'])
def test_invalid_mappings_rejected_without_state_commit(issue):
    *_,x,bundle=model();p=bundle.plan
    if issue=='missing_map':p['dynamic']['parameter_maps']=[]
    elif issue=='bad_index':p['dynamic']['parameter_maps'][0][0]=999999
    elif issue=='short_map':p['dynamic']['parameter_maps'][0].pop()
    elif issue=='missing_threshold':p['dynamic']['threshold_references'].pop()
    elif issue=='bad_threshold_ref':p['dynamic']['threshold_references'][-1]=[999,0]
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_tick))
    with pytest.raises(ValueError):trainer.gradients(x[None],[0])
    assert before==(trainer.state,trainer.neuron_state,trainer.clock_tick)


@pytest.mark.parametrize('kind',['array','frozen','synapse'])
@pytest.mark.parametrize('engine',['cpu','metal'])
def test_optimizer_aliases_read_updated_bank_after_checkpoint(kind,engine,tmp_path):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    _,_,groups,synapses,x,bundle=model(kind,learning_rate=.03)
    bundle.plan['backend']=engine
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    before=copy.deepcopy(trainer.state['weights']);out=trainer.step(x[None],[0])
    assert trainer.state['step']==1
    opt=bundle.plan['optimizer']
    for bank,(old,grad) in enumerate(zip(before,out['gradients'])):
        old=np.array(old);grad=np.array(grad)
        expected=old-opt['learning_rate']*grad/(abs(grad)+opt['epsilon']) if bundle.plan['trainable'][bank] else old
        np.testing.assert_allclose(trainer.state['weights'][bank],expected,rtol=1e-5,atol=3e-6)
    kbank=binding(bundle,groups[0],'k');tbank=binding(bundle,groups[0],'theta')
    if kind=='frozen':
        assert trainer.state['weights'][kbank]==before[kbank]
        assert trainer.state['weights'][tbank]==before[tbank]
    else:assert not np.allclose(trainer.state['weights'][kbank],before[kbank])
    path=tmp_path/'constants.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(path)
    assert restored.state==trainer.state
    weights=restored.state['weights'];k=np.array(weights[kbank]);theta=np.array(weights[tbank])
    v=np.array(out['final_membrane'][0]);a,c,d=v[:2],v[2:5],v[5:]
    if kind=='synapse':
        drive=weights[binding(bundle,synapses[0],'drive')][0]
        c=.8*c+.08*drive;d=.8*d+.04*drive;gain=np.full(6,drive)
    else:
        c=.8*c+.06*k[[1,0,1]]+.02*k[[0,1,0]];d=.8*d+.04*k[[1,0]];gain=k[[1,0,1,0,1,0]]
    a=.8*a+.02*k;th=np.r_[theta,theta[[1,0,0]],theta[[0,1]]];spikes=(np.r_[a,c,d]>th).astype(float)
    w1=weights[binding(bundle,synapses[1],'w')];w2=weights[binding(bundle,synapses[2],'w')]
    for edge in range(6):d[edge%2]+=spikes[2+edge//2]*w2[edge]
    for edge in range(6):c[edge%3]+=spikes[edge//3]*(w1[edge]+.05*k[edge//3]+.02*gain[edge])
    a-=.3*spikes[:2];c-=.2*theta[[1,0,0]]*spikes[2:5];d-=.25*theta[[0,1]]*spikes[5:]
    tail=restored.evaluate([[[0,0]]],[0],initial='carry')
    np.testing.assert_array_equal(tail['spikes'][0][0],spikes)
    np.testing.assert_allclose(tail['final_membrane'][0],np.r_[a,c,d],rtol=1e-5,atol=3e-6)



def test_negative_linked_thresholds_are_valid_dynamic_comparisons():
    _,_,groups,synapses,x,bundle=model()
    index=binding(bundle,groups[0],'theta');bundle.weights[index]=[-.2,0.]
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    loss,anchors=oracle(bundle,groups,synapses,x,bundle.weights)
    assert actual['loss']==pytest.approx(loss,abs=2e-13)
    for j in range(2):
        hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[index][j]+=1e-6;lo[index][j]-=1e-6
        fd=(oracle(bundle,groups,synapses,x,hi,anchors)[0]-oracle(bundle,groups,synapses,x,lo,anchors)[0])/2e-6
        assert actual['gradients'][index][j]==pytest.approx(fd,abs=4e-7,rel=4e-4)
