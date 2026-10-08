"""Canonical Brian storage: real Cython writeback and independent alias VJPs."""
import copy
import os
import tempfile

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, TrainingConversionError, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_dynamic_gpu import backend, compare


@pytest.fixture(scope='module',autouse=True)
def cython_cache():
    old=b.prefs.codegen.runtime.cython.cache_dir
    with tempfile.TemporaryDirectory(prefix='b2-linked-cython-') as directory:
        b.prefs.codegen.runtime.cython.cache_dir=directory
        try:yield
        finally:b.prefs.codegen.runtime.cython.cache_dir=old


def model(kind='permuted',method='euler',**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1]],float)
    ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='alias_input')
    a=b.NeuronGroup(2,'dv/dt=-v/ms:1'+(' (unless refractory)' if kind=='refractory' else '')+'\nz:1 (shared)'+('\nq:1' if kind=='summed' else ''),
                    threshold='v>1',reset='v-=.3',refractory=.4*b.ms if kind=='refractory' else False,
                    method=method,dt=dt,name='alias_a')
    a.v=[1.4,1.8];a.z=.7
    reset='v-=.3'+('\npeer+=.1' if kind in ('reset','syn_write','refractory') or kind.startswith('self') else '')
    c=b.NeuronGroup(2,'dv/dt=(-v+.2*peer+.1*z)/ms:1\npeer:1 (linked)\nz:1 (linked)',
                    threshold='v>1',reset=reset,method=method,dt=dt,name='alias_c')
    source=c if kind.startswith('self') else b.NeuronGroup(2,'v:1',name='outside_alias') if kind=='external' else a
    c.peer=b.linked_var(source,'z' if kind=='broadcast' else 'q' if kind=='summed' else 'v',
                       **({} if kind=='broadcast' else {'index':[0,1] if kind in ('identity','self_identity') else [0,0] if kind=='self_repeated' else [1,0]}))
    c.z=b.linked_var(a,'z');c.v=[1.3,1.7]
    layers=[a,c]
    if kind=='chain':
        d=b.NeuronGroup(2,'dv/dt=(-v+.1*peer)/ms:1\npeer:1 (linked)',threshold='v>1',
                        reset='v-=.3\npeer+=.05',method=method,dt=dt,name='alias_d')
        d.peer=b.linked_var(c,'peer',index=[1,0]);d.v=[1.6,1.2];layers.append(d)
    syn=b.Synapses(inp,a,'w:1',on_pre='v_post+=w',dt=dt,name='alias_drive');syn.connect(j='i');syn.w=[.12,.16]
    extras=[]
    if kind=='summed':
        s=b.Synapses(inp,a,'q_post=w:1 (summed)\nw:1',dt=dt,name='alias_sum')
        s.connect();s.w=[.1,.2,.3,.4];extras.append(s)
    if kind.startswith('syn_'):
        s=b.Synapses(a,c,'w:1\nk:1 (linked)'+ ('\npick:integer (constant)' if kind=='syn_write' else ''),
                     on_pre='v_post+=w+.1*k'+('\nk+=.03' if kind=='syn_write' else ''),dt=dt,name='alias_syn')
        s.connect();s.w=[.08,.07,.05,.06]
        if kind=='syn_write':s.pick=[1,0,0,1]
        s.k=b.linked_var(a,'v' if kind=='syn_write' else 'z',**({'index':'pick'} if kind=='syn_write' else {}))
        extras.append(s)
    net=b.Network(inp,*layers,syn,*extras)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,**options)
    return net,inp,layers,[syn,*extras],x,bundle


@pytest.mark.parametrize('kind',['identity','permuted','broadcast','reset','chain','syn_read','syn_write','self','self_identity','self_repeated','refractory','summed'])
@pytest.mark.parametrize('method',['euler','rk2','rk4'])
def test_linked_matches_real_brian(kind,method):
    net,inp,layers,synapses,x,bundle=model(kind,method)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in layers];net.add(*monitors);net.run(len(x)*.2*b.ms,namespace={})
    spikes=np.zeros((len(x),2*len(layers)))
    for l,m in enumerate(monitors):spikes[np.rint(np.asarray(m.t/b.second)/.0002).astype(int),2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(actual['spikes'][0],spikes)
    for group in layers:
        for name,indices in bundle.provenance['neuron_state_layout'][group.name].items():
            if name.startswith('__'):continue
            np.testing.assert_allclose(np.array(actual['final_state'])[0,indices],np.asarray(getattr(group,name)[:]),atol=2e-13,rtol=2e-13)
    for syn in synapses:
        for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
            np.testing.assert_allclose(np.array(actual['final_state'])[0,indices],np.asarray(getattr(syn,name)[:]),atol=2e-13,rtol=2e-13)
    unused=bundle.provenance['inactive_alias_slots']
    assert unused
    assert np.all(np.asarray(actual['initial_state_gradients'])[0,unused]==0)


def oracle(bundle,x,initial,weights,anchors=None):
    """Independent Euler recurrence; soft perturbations around recorded spikes."""
    p=bundle.plan;layout=bundle.provenance['neuron_state_layout'];a=initial[layout['alias_a']['v']].copy()
    c=initial[layout['alias_c']['v']].copy();z=initial[layout['alias_a']['z'][0]]
    bank=next(v['bank'] for v in bundle.provenance['bindings'] if v['object']=='alias_drive')
    saved=[];spikes=[]
    for t in range(len(x)):
        a*=.8;c=.8*c+.04*a[::-1]+.02*z
        v=np.r_[a,c]
        if anchors is None:s=(v>1).astype(float)
        else:
            old=anchors[t];phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old-1))**2
            s=(old>1).astype(float)+phi*(v-old)
        saved.append(v);spikes.append(s)
        a+=weights[bank]*x[t]
        gate=(v>1).astype(float) if p['detach_reset'] else s
        a-=.3*gate[:2];c-=.3*gate[2:]
        # c reset updates its linked peer (reversed a) after a reset.
        a+=.1*gate[2:][::-1]
    logits=np.asarray(spikes)[:,2:].mean(axis=0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,saved


@pytest.mark.parametrize('detach',[False,True])
def test_canonical_initial_and_weight_gradients_independent_finite_difference(detach):
    *_,x,bundle=model('reset',detach_reset=detach)
    init=np.array(bundle.initial_state);weights=[np.array(row) for row in bundle.weights]
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],initial=init[None])
    loss,anchors=oracle(bundle,x,init,weights)
    assert actual['loss']==pytest.approx(loss,abs=1e-13)
    canonical=sorted(set(sum((v for row in bundle.provenance['neuron_state_layout'].values() for v in row.values()),[])))
    assert abs(actual['initial_state_gradients'][0][bundle.provenance['neuron_state_layout']['alias_a']['z'][0]])>1e-7
    for i in canonical:
        hi=init.copy();lo=init.copy();hi[i]+=1e-6;lo[i]-=1e-6
        fd=(oracle(bundle,x,hi,weights,anchors)[0]-oracle(bundle,x,lo,weights,anchors)[0])/2e-6
        assert actual['initial_state_gradients'][0][i]==pytest.approx(fd,abs=3e-7,rel=3e-4)
    for bank,row in enumerate(weights):
        for i in range(len(row)):
            hi=copy.deepcopy(weights);lo=copy.deepcopy(weights);hi[bank][i]+=1e-6;lo[bank][i]-=1e-6
            fd=(oracle(bundle,x,init,hi,anchors)[0]-oracle(bundle,x,init,lo,anchors)[0])/2e-6
            assert actual['gradients'][bank][i]==pytest.approx(fd,abs=3e-7,rel=3e-4)


@pytest.mark.parametrize('kind',['reset','chain','syn_write','self_identity','summed'])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_linked_metal_and_mpi(kind,ranks,backend):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(kind,detach_reset=False,tbptt_window=3)
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    bundle.plan.update(backend=backend,mpi_ranks=ranks)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    compare(actual,cpu,backend)


@pytest.mark.parametrize('kind',['reset','syn_write'])
@pytest.mark.parametrize('ranks',[None,2])
def test_linked_carry_checkpoint_and_prune(kind,ranks,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(kind,mpi_ranks=ranks)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    whole=trainer.evaluate(x[None],[0]);trainer.execute(x[None,:3],[0]);path=tmp_path/'alias.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(path)
    tail=restored.execute(x[None,3:],[0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],atol=1e-13)
    before=np.array(restored.neuron_state);mask=copy.deepcopy(restored.plan['masks']);mask[-1]=[0.]*len(mask[-1]);restored.update_mask(mask)
    np.testing.assert_array_equal(restored.neuron_state,before)  # linked neuron state is never edge-private


@pytest.mark.parametrize('issue',['mutable_index','written_index','scalar_reset','scalar_path','constant_alias_parameter','external_link'])
def test_invalid_storage_rejected_before_execution(issue):
    if issue=='external_link':
        with pytest.raises(TrainingConversionError,match='selected neuron layer'):model('external')
        return
    net,inp,layers,synapses,x,bundle=model('syn_write')
    a,c=layers;s=synapses[-1]
    if issue=='mutable_index':
        s.variables['pick'].constant=False
        dynamic=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
        assert any(action.get('indirect') for action in dynamic.plan['dynamic']['actions'])
        actual=NativeLIFTrainer(dynamic.plan,runner=RUNNER,weights=dynamic.weights).evaluate(x[None],[0])
        expected=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
        np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
        assert float(net.t)==0
        return
    elif issue=='written_index':s.pre.code='pick=0\nv_post+=w+.1*k'
    elif issue=='scalar_reset':c.event_codes['spike']='v-=.3\nz+=.1'
    elif issue=='scalar_path':synapses[0].pre.code='z_post+=w'
    elif issue=='constant_alias_parameter':a.variables['z'].constant=True
    snapshot=[np.asarray(g.v[:]).copy() for g in layers]
    options=dict(trainable_neuron_parameters={c.name:['z']}) if issue=='constant_alias_parameter' else {}
    with pytest.raises(TrainingConversionError):lower_brian_dynamic_training(net,input_group=inp,layers=layers,**options)
    for g,v in zip(layers,snapshot):np.testing.assert_array_equal(g.v[:],v)
    assert float(net.t)==0
