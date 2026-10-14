"""Legal shared Synapses links, forwarded neuron links and reference lifetime."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_dynamic_gpu import backend,compare


def bank(bundle,obj,name):
    return next(v['bank'] for v in bundle.provenance['bindings'] if v['object']==obj.name and v['variables']==[name])


def model(kind='shared',reverse=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1]],float);ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='cross_input')
    proxy=kind=='proxy';neuron=kind=='neuron'
    a=b.NeuronGroup(2,'dv/dt=-v/ms:1\ng:1',threshold='v>1',reset='v-=.3',dt=dt,method='euler',name='cross_a');a.v=[1.7,1.3];a.g=[.3,.5]
    c=b.NeuronGroup(2,'dv/dt=(-v'+('+.1*peer' if neuron else '')+')/ms:1'+('\npeer:1 (linked)' if neuron else ''),
                    threshold='v>1',reset='v-=.3',dt=dt,method='euler',name='cross_c');c.v=[1.2,.9]
    source_name='a_source' if reverse else 'z_source';link_name='z_link' if reverse else 'a_link'
    source=b.Synapses(inp,a,'w:1\nu:1 ('+('linked' if proxy else 'shared')+')',
        on_pre='v_post+=w+.1*u\nw+=.02*u'+('\nu+=.03' if proxy else ''),dt=dt,name=source_name)
    if proxy:source.connect(j='i')
    else:source.connect()
    source.w=[.16,.22] if proxy else [.16,.22,.18,.14];source.pre.order=-2
    if proxy:source.u=b.linked_var(a,'g')
    else:source.u=.4
    link=b.Synapses(a,c,'w:1\npeer:1 (linked)'+('\npick:integer (constant)' if proxy else ''),
        on_pre='v_post+=w+.2*peer'+('\npeer+=.05' if proxy else ''),dt=dt,name=link_name)
    link.connect();link.w=[.14,.18,.16,.2];link.pre.order=-1
    if proxy:link.pick=[1,0,0,1]
    link.peer=b.linked_var(source,'u',**({'index':'pick'} if proxy else {}))
    if neuron:c.peer=b.linked_var(source,'u')
    net=b.Network(inp,a,c,source,link)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],
        trainable_synapse_parameters={source.name:['w']+([] if proxy else ['u']),link.name:['w']},**options)
    return net,inp,[a,c],source,link,x,bundle


@pytest.mark.parametrize('kind',['shared','neuron','proxy'])
@pytest.mark.parametrize('reverse',[False,True])
def test_cross_links_match_real_cython(kind,reverse):
    net,inp,groups,source,link,x,bundle=model(kind,reverse)
    out=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*.2*b.ms,namespace={})
    spikes=np.zeros((len(x),4))
    for l,m in enumerate(monitors):spikes[np.rint(np.asarray(m.t/b.second)/.0002).astype(int),2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(out['spikes'][0],spikes)
    objects=[(g,bundle.provenance['neuron_state_layout'][g.name]) for g in groups]+[(s,bundle.provenance['dynamic_state_layout'][s.name]) for s in (source,link)]
    for obj,layout in objects:
        for name,indices in layout.items():
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,indices],np.asarray(getattr(obj,name)[:]),atol=2e-13,rtol=2e-13)


def oracle(bundle,source,link,x,weights,initial,anchors=None):
    p=bundle.plan;layout=bundle.provenance['dynamic_state_layout'];nl=bundle.provenance['neuron_state_layout']
    a=initial[nl['cross_a']['v']].copy();c=initial[nl['cross_c']['v']].copy()
    w=initial[layout[source.name]['w']].copy();u=initial[layout[source.name]['u'][0]];lw=weights[bank(bundle,link,'w')]
    saved=[];spikes=[]
    for t in range(len(x)):
        a*=.8;c*=.8;v=np.r_[a,c]
        if anchors is None:s=(v>1).astype(float)
        else:
            old=anchors[t];phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old-1))**2
            s=(old>1).astype(float)+phi*(v-old)
        saved.append(v);spikes.append(s)
        for edge in range(4):
            gate=x[t,edge//2];a[edge%2]+=gate*(w[edge]+.1*u);w[edge]+=gate*.02*u
        for edge in range(4):c[edge%2]+=s[edge//2]*(lw[edge]+.2*u)
        reset=(v>1).astype(float) if p['detach_reset'] else s
        a-=.3*reset[:2];c-=.3*reset[2:]
    logits=np.array(spikes)[:,2:].mean(axis=0)*p['logit_scale']
    return np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0],saved


@pytest.mark.parametrize('detach',[False,True])
def test_shared_cross_link_independent_initial_and_parameter_vjp(detach):
    *_,source,link,x,bundle=model(detach_reset=detach)
    weights=[np.array(w) for w in bundle.weights];initial=np.array(bundle.initial_state)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],initial=initial[None])
    loss,anchors=oracle(bundle,source,link,x,weights,initial)
    assert actual['loss']==pytest.approx(loss,abs=2e-13)
    for i in range(len(initial)):
        hi=initial.copy();lo=initial.copy();hi[i]+=1e-6;lo[i]-=1e-6
        fd=(oracle(bundle,source,link,x,weights,hi,anchors)[0]-oracle(bundle,source,link,x,weights,lo,anchors)[0])/2e-6
        assert actual['initial_state_gradients'][0][i]==pytest.approx(fd,abs=3e-7,rel=3e-4)
    default=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    for entry in bundle.provenance['bindings']:
        idx=entry['bank']
        for j in range(len(weights[idx])):
            hi=copy.deepcopy(weights);lo=copy.deepcopy(weights);hi[idx][j]+=1e-6;lo[idx][j]-=1e-6
            ai=initial.copy();ci=initial.copy()
            if entry['kind']=='synapse_initial':
                cell=bundle.provenance['dynamic_state_layout'][entry['object']][entry['variables'][0]][j];ai[cell]+=1e-6;ci[cell]-=1e-6
            fd=(oracle(bundle,source,link,x,hi,ai,anchors)[0]-oracle(bundle,source,link,x,lo,ci,anchors)[0])/2e-6
            assert default['gradients'][idx][j]==pytest.approx(fd,abs=3e-7,rel=3e-4)


@pytest.mark.parametrize('kind',['shared','neuron','proxy'])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_cross_links_gpu_mpi(kind,ranks,backend):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(kind,detach_reset=False,tbptt_window=3)
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    bundle.plan.update(backend=backend,mpi_ranks=ranks)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0]);compare(actual,cpu,backend)


@pytest.mark.parametrize('kind',['shared','neuron'])
@pytest.mark.parametrize('ranks',[None,2])
def test_shared_reference_lifetime_and_checkpoint(kind,ranks,tmp_path):
    check_lifetime(kind,ranks,tmp_path)


def check_lifetime(kind,ranks,tmp_path,execution_backend='cpu'):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,source,link,x,bundle=model(kind,mpi_ranks=ranks,backend=execution_backend);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);whole=trainer.evaluate(x[None],[0])
    trainer.execute(x[None,:3],[0]);path=tmp_path/'cross.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(path)
    tail=restored.execute(x[None,3:],[0],initial='carry');np.testing.assert_allclose(tail['final_state'],whole['final_state'],atol=1e-13)
    cell=bundle.provenance['dynamic_state_layout'][source.name]['u'][0];old=restored.neuron_state[0][cell]
    masks=copy.deepcopy(restored.plan['masks']);masks[bank(bundle,source,'w')]=[0.]*4;restored.update_mask(masks)
    assert restored.neuron_state[0][cell]==old  # link edges still refer to the storage
    masks[bank(bundle,link,'w')]=[0.]*4;restored.update_mask(masks)
    assert restored.neuron_state[0][cell]==(old if kind=='neuron' else 0.)
    masks[bank(bundle,source,'w')]=[1.]*4;restored.update_mask(masks,growth_weight=.4)
    expected=old if kind=='neuron' else bundle.weights[bank(bundle,source,'u')][0]
    assert restored.neuron_state[0][cell]==expected


@pytest.mark.parametrize('kind',['shared','neuron'])
def test_gpu_mpi_reference_lifetime_carry_checkpoint(kind,tmp_path,backend):
    check_lifetime(kind,2,tmp_path,backend)


def test_pinned_shared_layout_rejects_mask_ownership():
    *_,source,link,x,bundle=model('neuron');cell=bundle.provenance['dynamic_state_layout'][source.name]['u'][0]
    bundle.plan['dynamic']['migration']['cells'].append(dict(index=cell,owners=[[bank(bundle,source,'w'),0]],restart=dict(kind='initial')))
    with pytest.raises(ValueError,match='unmasked writer'):NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])


def test_brian_itself_rejects_links_to_dynamic_edge_arrays():
    _,_,_,source,_,_,_=model()
    with pytest.raises(NotImplementedError,match='fixed size'):b.linked_var(source,'w')
