"""Standard scheduled assignments against compiled Brian and native VJPs."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training, TrainingConversionError
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine


def model(kind='shared',when='start',warm=0,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1]],float)
    ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='regular_input')
    refractory=kind=='refractory'
    a=b.NeuronGroup(3,'dv/dt=(.4-v)/ms:1'+(' (unless refractory)' if refractory else '')+'\ng:1 (shared)\ngain:1 (shared,constant)',
        threshold='v>.6',reset='v-=.3',method='euler',refractory=.4*b.ms if refractory else False,dt=dt,name='regular_a')
    a.v=[.2,.5,.7];a.g=.1;a.gain=.2
    c=b.NeuronGroup(3,'dv/dt=(.4-v)/ms:1\np:1 (linked)\nq:1 (linked)\npeer:1 (linked)\npick:integer',
        threshold='v>.6',reset='v-=.3',method='euler',dt=dt,name='regular_c')
    c.v=[.7,.3,.8];c.pick=[2,0,1];c.p=b.linked_var(a,'g');c.q=b.linked_var(a,'g');c.peer=b.linked_var(a,'v',index='pick')
    s=b.Synapses(a,c,'w:1\nh:1 (shared)',on_pre='v_post+=w',dt=dt,name='regular_s')
    s.connect(i=[2,0,1,2],j=[1,2,0,2]);s.w=[.1,.12,.15,.17];s.h=.2
    drive=b.Synapses(inp,a,'w:1',on_pre='v_post+=w',dt=dt,name='regular_drive');drive.connect(i=[0,1],j=[0,1]);drive.w=.03
    target=a[1:3] if kind=='subgroup' else a
    target.run_regularly('g+=gain*.02\ntmp=g+.05\nv+=tmp*.2+.01*i',when=when,order=1,name='regular_neuron')
    c.run_regularly('p+=.03\nq+=.07\nsaved=p+2*q\nv+=.03*saved',when=when,order=2,name='regular_alias')
    if kind=='indexed':c.run_regularly('pick=(pick+1)%3\npeer+=.02\nv+=.1*peer',when='end',name='regular_index')
    s.run_regularly('h+=.01\nlocal=2*h\nw=.9*w+.01*local+.001*i\nv_post+=.02*w',when=when,order=3,name='regular_synapse')
    net=b.Network(inp,a,c,s,drive)
    if warm:net.run(warm*dt,namespace={});x=x[warm:]
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],
        trainable_neuron_parameters={a.name:['gain']},**options)
    return net,inp,[a,c],[s,drive],x,bundle


@pytest.mark.parametrize('kind',['shared','subgroup','indexed','refractory'])
@pytest.mark.parametrize('when',['start','after_synapses','end'])
def test_regular_matches_compiled_brian(engine,kind,when,tmp_path):
    net,inp,groups,synapses,x,bundle=model(kind,when,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(monitors);all_spikes=[]
    for part in (x[:3],x[3:]):
        result=trainer.step(part[None],[0],**({'initial':'carry'} if all_spikes else {}))
        all_spikes.extend(result['spikes'][0]);net.run(len(part)*.2*b.ms,namespace={})
        for obj in net.sorted_objects:
            if hasattr(obj,'codeobj') and obj.codeobj is not None:assert obj.codeobj.compiled_code['run'] is not None
        z=np.asarray(result['final_state'])[0];tol=4e-5 if engine!='cpu' else 3e-12
        for group in groups:
            for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():
                if name.startswith('__'):continue
                expected=group.variables[name].get_value()
                actual=z[slots]
                if name=='peer':
                    expected=np.asarray(groups[0].v)[np.asarray(group.pick)]
                    descriptors=bundle.provenance['runtime_index_layout'][group.name][name]
                    values=[]
                    for descriptor in descriptors:
                        index=int(z[descriptor['index']])
                        for table in descriptor['tables'][:-1]:index=int(z[table[index]])
                        values.append(z[descriptor['tables'][-1][index]])
                    actual=values
                np.testing.assert_allclose(actual,np.broadcast_to(expected,(len(group),)),rtol=tol,atol=tol*1e-2,err_msg=group.name+'.'+name)
        for syn in synapses:
            for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
                np.testing.assert_allclose(z[slots],syn.variables[name].get_value(),rtol=tol,atol=tol*1e-2,err_msg=syn.name+'.'+name)
        path=tmp_path/'regular.json';trainer.store(path);other=NativeLIFTrainer(trainer.plan,runner=RUNNER);other.restore(path);trainer=other
    expected=np.zeros((len(x),6))
    for k,monitor in enumerate(monitors):expected[np.rint(np.asarray(monitor.t/b.second)/.0002).astype(int),np.asarray(monitor.i)+3*k]=1
    np.testing.assert_array_equal(all_spikes,expected)
    if engine!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('ranks',[2,8])
def test_regular_mpi(ranks):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    _,_,_,_,x,bundle=model('indexed','end')
    single=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    bundle.plan['mpi_ranks']=ranks
    distributed=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    for key in ('loss','spikes','final_state','initial_state_gradients'):
        np.testing.assert_allclose(distributed[key],single[key],rtol=2e-12,atol=2e-12)
    for a,c in zip(distributed['gradients'],single['gradients']):np.testing.assert_allclose(a,c,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('change,match',[
    ('constant','read-only|constant|mutable'),
    ('template','standard run_regularly'),('inactive','inactive'),('input','selected neuron'),
])
def test_regular_rejects_unsupported(change,match):
    net,inp,groups,synapses,x,bundle=model()
    runner=next(o for o in net.sorted_objects if o.name=='regular_neuron')
    if change=='constant':runner.abstract_code='gain+=.1'
    elif change=='template':runner.template='reset'
    elif change=='inactive':runner.active=False
    elif change=='input':inp.run_regularly('i=0')
    with pytest.raises(TrainingConversionError,match=match):lower_brian_dynamic_training(net,input_group=inp,layers=groups)


def oracle(bundle,x,weights=None,initial=None,anchors=None):
    """Independent end-slot recurrence; no generated actions are interpreted."""
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for k,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[k]=weights[ref[0]][ref[1]]
    nl=bundle.provenance['neuron_state_layout'];sl=bundle.provenance['dynamic_state_layout']
    a=nl['regular_a']['v'];c=nl['regular_c']['v'];g=nl['regular_a']['g'][0]
    w=sl['regular_s']['w'];h=sl['regular_s']['h'][0]
    gain=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='regular_a')
    drive=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='regular_drive')
    sources=[2,0,1,2];targets=[1,2,0,2];spikes=[];margins=[];before=[]
    for tick,external in enumerate(x):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[a]=.8*z[a]+.08;z[c]=.8*z[c]+.08
        margin=z[a+c]-.6;event=(margin>0).astype(float)
        if anchors is not None:
            old=anchors['margins'][tick]
            event=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());margins.append(margin.copy())
        z[a[:2]]+=external*np.asarray(weights[drive])
        for k,j in enumerate(targets):z[c[j]]+=event[sources[k]]*z[w[k]]
        z[a+c]-=.3*event
        z[g]+=.02*weights[gain][0];z[a]+=.2*(z[g]+.05)+.01*np.arange(3)
        left=z[g]+.03;right=z[g]+.07;z[g]=right;z[c]+=.03*(left+2*right)
        z[h]+=.01
        for k,j in enumerate(targets):
            z[w[k]]=.9*z[w[k]]+.02*z[h]+.001*sources[k];z[c[j]]+=.02*z[w[k]]
    spikes=np.asarray(spikes);logits=spikes[:,3:].mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes,dict(before=before,margins=margins)


@pytest.mark.parametrize('window',[None,2])
def test_regular_independent_derivatives(engine,window):
    _,_,_,_,x,bundle=model('shared','end',backend=engine,tbptt_window=window,detach_reset=False)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    loss,z,spikes,anchors=oracle(bundle,x)
    np.testing.assert_allclose(result['loss'],loss,rtol=5e-5 if engine!='cpu' else 1e-12)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    eps=1e-6;tol=2e-3 if engine!='cpu' else 3e-7
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            plus=copy.deepcopy(bundle.weights);minus=copy.deepcopy(bundle.weights)
            plus[bank][index]+=eps;minus[bank][index]-=eps
            expected=(oracle(bundle,x,plus,anchors=anchors)[0]-oracle(bundle,x,minus,anchors=anchors)[0])/(2*eps)
            np.testing.assert_allclose(result['gradients'][bank][index],expected,rtol=tol,atol=tol*1e-2,err_msg=f'weight {bank}/{index}')
    # Explicit initial state disables optimizer initialization, as in native.
    initial=np.asarray(bundle.initial_state,float)
    explicit=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],initial=initial[None])
    _,_,_,anchors=oracle(bundle,x,initial=initial)
    for index,detached in enumerate(bundle.plan['dynamic']['detached']):
        if detached:continue
        plus=initial.copy();minus=initial.copy();plus[index]+=eps;minus[index]-=eps
        expected=(oracle(bundle,x,initial=plus,anchors=anchors)[0]-oracle(bundle,x,initial=minus,anchors=anchors)[0])/(2*eps)
        np.testing.assert_allclose(explicit['initial_state_gradients'][0][index],expected,rtol=tol,atol=tol*1e-2,err_msg=f'initial {index}')


@pytest.mark.parametrize('warm',[0,2])
def test_regular_normal_time_and_carry(engine,warm):
    from unittest.mock import patch
    from test_training_stochastic import normal
    net,inp,groups,synapses,x,_=model('shared','end',warm=warm)
    runners={obj.name:obj for obj in net.sorted_objects if obj.name in ('regular_neuron','regular_synapse')}
    runners['regular_neuron'].abstract_code='g+=.01*randn()\ndraw=randn()\nv+=.02*draw+.01*draw+.001*t/ms'
    runners['regular_synapse'].abstract_code='h+=.01*randn()\nw+=.002*randn()+.001*h'
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    net.run(0*b.ms,namespace={})
    ordered=[(obj,3 if obj.name=='regular_neuron' else 4) for obj in net.sorted_objects if obj.name in runners]
    for obj,count in ordered:assert obj.codeobj.compiled_code['run'] is not None
    cursor=0
    for part in (x[:2],x[2:]):
        actual=trainer.step(part[None],[0],**({'initial':'carry'} if cursor else {'noise_sequence':9}))
        draws=[]
        for tick in range(cursor,cursor+len(part)):
            for obj,count in ordered:
                domain=bundle.provenance['regular_runner_layout'][obj.name]['noise_domain']
                draws.append(normal(bundle.plan['seed'],9,0,domain,0,tick,0))
                draws.extend(normal(bundle.plan['seed'],9,0,domain,j,tick,1) for j in range(count))
        device=b.get_device();device.randn_buffer_index[:]=0;calls=[]
        def refill(n):
            assert n==20000 and not calls;calls.append(n)
            values=np.zeros(n);values[:len(draws)]=draws;return values
        with patch('numpy.random.randn',refill):net.run(len(part)*.2*b.ms,namespace={})
        assert calls==[20000] and device.randn_buffer_index[0]==len(draws);device.randn_buffer_index[:]=0
        state=np.asarray(actual['final_state'])[0];tol=4e-5 if engine!='cpu' else 3e-12
        for group in groups:
            for name in ('v','g') if group is groups[0] else ('v',):
                slots=bundle.provenance['neuron_state_layout'][group.name][name]
                np.testing.assert_allclose(state[slots],np.broadcast_to(group.variables[name].get_value(),(len(group),)),rtol=tol,atol=tol*1e-2)
        for name,slots in bundle.provenance['dynamic_state_layout']['regular_s'].items():
            np.testing.assert_allclose(state[slots],synapses[0].variables[name].get_value(),rtol=tol,atol=tol*1e-2)
        cursor+=len(part)
