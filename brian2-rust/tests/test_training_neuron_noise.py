"""Neuron explicit draws: independent integrators, compiled Brian and VJPs."""
import copy
import gc
import os
import tempfile
from unittest.mock import patch
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_uniform import uniform
from test_training_stochastic import normal


@pytest.fixture(scope='module',autouse=True)
def cython_cache():
    old=b.prefs.codegen.runtime.cython.cache_dir
    with tempfile.TemporaryDirectory(prefix='b2-neuron-noise-cython-') as directory:
        b.prefs.codegen.runtime.cython.cache_dir=directory
        try:yield
        finally:b.prefs.codegen.runtime.cython.cache_dir=old


def model(method='euler',refractory=False,warm=0,sde=False,typed=False,**options):
    gc.collect();b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='neuron_input');groups=[]
    for k in range(2):
        eq='dv/dt=(.7-v+.15*h+gain*rand()+.01*v*randn())/ms'+('+.02*xi/sqrt(ms)' if sde else '')+':1'+(' (unless refractory)' if refractory else '')+'\ndh/dt=-h/ms:1\ngain:1 (constant)'
        reset=('k=int(4*rand())+2147483647; flag=k>0; ' if typed else '')+'u=rand(); n=randn(); v-=.3+.02*u+.001*n'+('+.003*(k%3)' if typed else '')+'; h+=.1*u'+('*int(flag)' if typed else '')
        g=b.NeuronGroup(2,eq,threshold='v>.6+.05*rand()',reset=reset,
            method=method,dt=dt,refractory=3*dt if refractory else False,name=f'neuron_{k}')
        g.v=[.8-.04*k,.9+.03*k];g.h=[.1,.2];g.gain=[.1+.02*k,.2];groups.append(g)
    syn=b.Synapses(*groups,'w:1',on_pre='v_post+=w',dt=dt,name='neuron_syn');syn.connect(j='i');syn.w=[.12,.16]
    net=b.Network(inp,*groups,syn)
    if warm:b.seed(491);net.run(warm*dt,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,
        trainable_neuron_parameters={g.name:['gain'] for g in groups},**options)
    return net,groups,bundle


def oracle(bundle,weights=None,initial=None,anchors=None,length=8,sequence=9,batch=0):
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    layouts=[bundle.provenance['neuron_state_layout'][f'neuron_{k}'] for k in range(2)]
    banks=[next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==f'neuron_{k}' and e['variables']==['gain']) for k in range(2)]
    syn_bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='neuron_syn')
    refs=p.get('refractory',[None,None]);methods=bundle.provenance['integrators'];calls=bundle.provenance['neuron_draws'];named=[int('xi' in n) for n in bundle.provenance['noise_names']]
    def sample(kind,l,j,t,stream):return (uniform if kind=='rand' else normal)(p['seed'],sequence,batch,l,j,t,stream)
    def phase(name,l,j,t):return {kind:[sample(kind,l,j,t,c['stream']) for c in calls[l][name] if c['kind']==kind] for kind in ('rand','randn')}
    def record(name,l,j,t,draws):
        if name=='update' and named[l]:draws['randn'].append(sample('randn',l,j,t,0))
        for c in calls[l][name]:draws[c['kind']].append(sample(c['kind'],l,j,t,c['stream']))
    before=[];margins=[];spikes=[];draws={'rand':[],'randn':[]}
    for tick in range(length):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());active=np.ones(4,dtype=bool);event=np.zeros(4);hard=np.zeros(4);margin=np.zeros(4)
        for l,layout in enumerate(layouts):
            if refs[l] is not None:
                counter=layout['__refractory_ticks'];active[l*2:l*2+2]=z[counter]<=0;z[counter]=np.maximum(z[counter]-1,0)
            for j in range(2):
                record('update',l,j,tick,draws);d=phase('update',l,j,tick);v=z[layout['v'][j]];h=z[layout['h'][j]];gain=weights[banks[l]][j]
                def f(v,h,stage):return np.array([(.7-v+.15*h+gain*d['rand'][stage]+.01*v*d['randn'][stage])*active[2*l+j],-h])*.2
                state=np.array([v,h]);k1=f(v,h,0)
                if methods[l]=='euler':out=state+k1
                elif methods[l]=='rk2':out=state+f(*(state+.5*k1),1)
                elif methods[l]=='rk4':
                    k2=f(*(state+.5*k1),1);k3=f(*(state+.5*k2),2);k4=f(*(state+k3),3);out=state+(k1+2*k2+2*k3+k4)/6
                else:raise AssertionError(methods[l])
                if named[l]:out[0]+=.02*np.sqrt(.2)*sample('randn',l,j,tick,0)*active[2*l+j]
                z[layout['v'][j]],z[layout['h'][j]]=out
        for l,layout in enumerate(layouts):
            for j in range(2):
                record('threshold',l,j,tick,draws);margin[2*l+j]=z[layout['v'][j]]-.6-(.05*phase('threshold',l,j,tick)['rand'][0] if calls[l]['threshold'] else 0.)
        hard=(margin>0)*active;event=hard.astype(float)
        if anchors is not None:
            old=anchors['margins'][tick];event=hard+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)*active
            # Control decisions, including reset membership and refractory
            # latches, are frozen for the pathwise finite difference.
            hard=anchors['hard'][tick]
        margins.append(margin.copy());spikes.append(event.copy())
        guard=active[2:] & ~np.asarray(hard[2:],bool) if refs[1] is not None else np.ones(2,dtype=bool)
        z[layouts[1]['v']]+=np.asarray(weights[syn_bank])*event[:2]*guard
        for l,layout in enumerate(layouts):
            for j in range(2):
                if hard[2*l+j]:record('reset',l,j,tick,draws)
                d=phase('reset',l,j,tick);u=d['rand'][-1];n=d['randn'][0]
                typed=len(d['rand'])>1;k=(int(4*d['rand'][0])+2147483647+2**31)%2**32-2**31 if typed else 0
                z[layout['v'][j]]-=event[2*l+j]*(.3+.02*u+.001*n+.003*(k%3));z[layout['h'][j]]+=event[2*l+j]*.1*u*(int(k>0) if typed else 1)
                if refs[l] is not None and hard[2*l+j]:z[layout['__refractory_ticks'][j]]=2
    spikes=np.asarray(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes,dict(before=before,margins=margins,hard=spikes.astype(bool),draws=draws)


CASES=[('euler',False),('euler',True),('rk2',False),('rk4',False)]
@pytest.mark.parametrize('method,sde',CASES)
@pytest.mark.parametrize('refractory,warm',[(False,0),(True,2)])
def test_neuron_draws_compiled_brian(engine,method,sde,refractory,warm,tmp_path):
    _check_compiled_brian(engine,method,sde,refractory,warm,tmp_path)


def _check_compiled_brian(engine,method,sde,refractory,warm,tmp_path,typed=False):
    net,groups,bundle=model(method,refractory,warm,sde,typed=typed,backend=engine);bundle.plan['trainable']=[False]*len(bundle.weights)
    expected=oracle(bundle);trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);spikes=[]
    for start,stop in ((0,3),(3,8)):
        result=trainer.step(np.zeros((1,stop-start,2)),[0],**({'initial':'carry'} if start else {'noise_sequence':9}));spikes.extend(result['spikes'][0])
        path=tmp_path/'neuron.json';trainer.store(path);restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
    np.testing.assert_array_equal(spikes,expected[2]);tol=5e-5 if engine!='cpu' else 5e-12
    slots=sum((layout['v']+layout['h'] for layout in bundle.provenance['neuron_state_layout'].values()),[])
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],expected[1][slots],rtol=tol,atol=tol*.01)
    net.run(0*b.ms,namespace={});monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    draws=expected[3]['draws'];device=b.get_device();device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0;calls={k:0 for k in draws}
    def sampler(name):
        def refill(n):
            assert n==20000 and calls[name]==0;calls[name]+=1;values=np.zeros(n);values[:len(draws[name])]=draws[name];return values
        return refill
    with patch('numpy.random.rand',sampler('rand')),patch('numpy.random.randn',sampler('randn')):net.run(8*.2*b.ms,namespace={})
    for name in draws:assert calls[name]==1 and getattr(device,name+'_buffer_index')[0]==len(draws[name])
    device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    actual=np.zeros((8,4))
    for l,(g,m) in enumerate(zip(groups,monitors)):
        assert all(r.codeobj.compiled_code['run'] is not None for r in [g.state_updater,g.thresholder['spike'],g.resetter['spike']])
        ticks=np.rint(np.asarray(m.t/b.second)/.0002).astype(int)-warm;actual[ticks,2*l+np.asarray(m.i)]=1
        for name in ('v','h'):np.testing.assert_allclose(expected[1][bundle.provenance['neuron_state_layout'][g.name][name]],g.variables[name].get_value(),rtol=5e-12,atol=5e-14)
    np.testing.assert_array_equal(actual,expected[2])


@pytest.mark.parametrize('method,sde',CASES)
@pytest.mark.parametrize('window',[None,3])
def test_neuron_draws_independent_derivatives(engine,method,sde,window):
    _,_,bundle=model(method,True,2,sde,backend=engine,tbptt_window=window)
    x=np.zeros((1,8,2));result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    loss,z,spikes,anchors=oracle(bundle);tol=3e-3 if engine!='cpu' else 1e-6;eps=1e-6
    np.testing.assert_array_equal(result['spikes'][0],spikes);np.testing.assert_allclose(result['loss'],loss,rtol=tol)
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            a=copy.deepcopy(bundle.weights);c=copy.deepcopy(bundle.weights);a[bank][k]+=eps;c[bank][k]-=eps
            fd=(oracle(bundle,weights=a,anchors=anchors)[0]-oracle(bundle,weights=c,anchors=anchors)[0])/(2*eps)
            np.testing.assert_allclose(result['gradients'][bank][k],fd,rtol=tol,atol=tol*.01,err_msg=f'weight {bank}/{k}')
    slots=sum((layout['v']+layout['h'] for layout in bundle.provenance['neuron_state_layout'].values()),[])
    for k in slots:
        a=np.array(bundle.initial_state);c=a.copy();a[k]+=eps;c[k]-=eps
        fd=(oracle(bundle,initial=a,anchors=anchors)[0]-oracle(bundle,initial=c,anchors=anchors)[0])/(2*eps)
        np.testing.assert_allclose(result['initial_state_gradients'][0][k],fd,rtol=tol,atol=tol*.01,err_msg=f'initial {k}')


@pytest.mark.parametrize('ranks',[2,8])
def test_neuron_draws_mpi_batch_restore(engine,ranks,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    _,_,bundle=model('rk2',True,2,backend=engine,mpi_ranks=ranks)
    x=np.zeros((2,8,2));cpu=copy.deepcopy(bundle.plan);cpu.update(backend='cpu',mpi_ranks=None)
    actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0,0],noise_sequence=9)
    expected=NativeLIFTrainer(cpu,weights=bundle.weights,runner=RUNNER).gradients(x,[0,0],noise_sequence=9)
    for key in ('loss','spikes','final_state','initial_state_gradients'):
        np.testing.assert_allclose(actual[key],expected[key],rtol=3e-3 if engine!='cpu' else 4e-12,atol=3e-6 if engine!='cpu' else 4e-13)
    for a,c in zip(actual['gradients'],expected['gradients']):np.testing.assert_allclose(a,c,rtol=3e-3 if engine!='cpu' else 4e-12,atol=3e-6 if engine!='cpu' else 4e-13)
    for batch in range(2):np.testing.assert_array_equal(actual['spikes'][batch],oracle(bundle,batch=batch)[2])
    bundle.plan['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    trainer.step(x[:,:3],[0,0],noise_sequence=9);path=tmp_path/'mpi-neuron.json';trainer.store(path)
    restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);result=restored.step(x[:,3:],[0,0],initial='carry')
    np.testing.assert_allclose(result['final_state'],expected['final_state'],rtol=5e-5 if engine!='cpu' else 4e-12,atol=5e-7 if engine!='cpu' else 4e-14)
    assert result['noise_sequence']==9 and result['final_tick']==8
    if engine!='cpu':assert result['gpu_dispatches']>0 and actual['gpu_dispatches']>0


@pytest.mark.parametrize('damage',['stream_budget','conditional_margin','masked_margin'])
def test_neuron_draw_validation(damage):
    net,groups,bundle=model()
    if damage=='stream_budget':
        groups[0].events['spike']='v>.6+.001*('+ '+'.join(['rand()']*13)+')'
        source=next(o for o in net.objects if o.name=='neuron_input')
        with pytest.raises(ValueError,match='16 random streams'):
            lower_brian_dynamic_training(net,input_group=source,layers=groups)
        return
    scratch=bundle.provenance['threshold_margin_layout']['neuron_0'][0]
    action=next(a for a in bundle.plan['dynamic']['actions'] if a.get('writes')==[scratch])
    if damage=='conditional_margin':action['trigger']=dict(external=True,index=0)
    else:action['mask']=[0,0]
    with pytest.raises(ValueError,match='invalid or repeated dynamic threshold'):
        NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,2,2)),[0],noise_sequence=9)


@pytest.mark.parametrize('method',['euler','rk2'])
@pytest.mark.parametrize('typed',[False,True])
def test_neuron_draws_static_frontend(engine,method,typed):
    from brian2_rust import lower_brian_training
    net,groups,_=model(method,typed=typed)
    source=next(o for o in net.objects if o.name=='neuron_input')
    for g in groups:g.events['spike']='v>.6'
    options=dict(input_group=source,layers=groups,trainable_neuron_parameters={g.name:['gain'] for g in groups},detach_reset=False)
    static=lower_brian_training(net,backend=engine,**options);dynamic=lower_brian_dynamic_training(net,**options)
    assert static.plan['schema']=='b2-state-training-plan-v4'
    x=np.zeros((1,8,2));result=NativeLIFTrainer(static.plan,weights=static.weights,runner=RUNNER).gradients(x,[0],initial=[static.initial_state],noise_sequence=9)
    expected=oracle(dynamic);np.testing.assert_array_equal(result['spikes'][0],expected[2]);tol=3e-3 if engine!='cpu' else 1e-6
    np.testing.assert_allclose(result['loss'],expected[0],rtol=tol)
    np.testing.assert_allclose(result['final_state'][0],expected[1],rtol=tol,atol=tol*.01)
    reference=NativeLIFTrainer(dynamic.plan,weights=dynamic.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    for entry in static.provenance['bindings']:
        peer=next(e for e in dynamic.provenance['bindings'] if e['object']==entry['object'] and e['variables']==entry['variables'])
        np.testing.assert_allclose(result['gradients'][entry['bank']],reference['gradients'][peer['bank']],rtol=tol,atol=tol*.01)
    np.testing.assert_allclose(result['initial_state_gradients'],reference['initial_state_gradients'],rtol=tol,atol=tol*.01)


def test_neuron_typed_reset_compiled_brian(engine,tmp_path):
    _check_compiled_brian(engine,'euler',False,False,0,tmp_path,typed=True)


@pytest.mark.parametrize('equations,match',[
    ('dv/dt=(rand()-rand())/ms:1','more than one call'),
    ('dv/dt=(-v+noise)/ms:1\nnoise=rand():1','constant.*over dt'),
])
def test_neuron_random_restrictions_before_symbolic_expansion(equations,match):
    from brian2_rust import TrainingConversionError
    b.set_device('runtime');b.start_scope();dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=dt)
    groups=[b.NeuronGroup(1,equations,threshold='v>.6',reset='v-=.3',method='euler',dt=dt) for _ in range(2)]
    # Brian caches SymPy substitutions. A pre-filled cache must not hide a
    # repeated stateful call or bypass the original subexpression restriction.
    for group in groups:group.equations.get_substituted_expressions()
    with pytest.raises(TrainingConversionError,match=match):
        lower_brian_dynamic_training(b.Network(source,*groups),input_group=source,layers=groups)
