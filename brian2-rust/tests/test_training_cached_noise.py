"""Cached stochastic subexpressions: independent schedule, Cython and VJPs."""
import copy
import gc
import os
from unittest.mock import patch
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training,TrainingConversionError
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_neuron_noise import cython_cache
from test_training_uniform import uniform
from test_training_stochastic import normal


def model(method='rk2',ref=True,warm=0,**options):
    gc.collect();b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='cached_input');groups=[]
    for k in range(2):
        eq='dv/dt=(.7-v+q)/ms:1'+(' (unless refractory)' if ref else '')+'''\ndh/dt=(-h+.1*q)/ms:1
sh=rand():1 (shared,constant over dt)
u=rand():1 (constant over dt)
n=randn():1 (constant over dt)
flag=u>.5:boolean (constant over dt)
q=gain*u+.01*n+.02*sh+.05*v+.01*int(flag)+.001*t/ms:1 (constant over dt)
gain:1 (constant)'''
        g=b.NeuronGroup(2,eq,threshold='v>.6+.03*q',reset='v-=.3+.02*q;h+=.01*q',method=method,dt=dt,
            refractory=3*dt if ref else False,name=f'cached_{k}')
        g.v=[.8-.04*k,.9+.03*k];g.h=[.1,.2];g.gain=[.1+.02*k,.2];groups.append(g)
    syn=b.Synapses(*groups,'''w:1
r=rand()+.02*h_pre:1 (constant over dt)''',
        on_pre='v_post+=.99*w*r+.01*q_post;w*=.99',on_post='w+=.002*r',dt=dt,name='cached_s')
    syn.connect(j='i');syn.w=[.12,.16];net=b.Network(source,*groups,syn)
    if warm:b.seed(739);net.run(warm*dt,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,detach_reset=False,
        trainable_neuron_parameters={g.name:['gain'] for g in groups},**options)
    return net,groups,syn,bundle


def oracle(bundle,weights=None,initial=None,anchors=None,length=8,sequence=9,batch=0):
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    layouts=[bundle.provenance['neuron_state_layout'][f'cached_{k}'] for k in range(2)]
    syn=bundle.provenance['dynamic_state_layout']['cached_s'];regular=bundle.provenance['regular_runner_layout']
    gains=[next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==f'cached_{k}' and e['variables']==['gain']) for k in range(2)]
    # Native initial parameters overwrite carry only for an explicit fresh start.
    if initial is None:
        for index,binding in enumerate(p['dynamic']['initial_parameters']):
            if binding is not None:z[index]=weights[binding[0]][binding[1]]
    refs=p.get('refractory',[None,None]);methods=bundle.provenance['integrators']
    draws={'rand':[],'randn':[]};before=[];margins=[];spikes=[]
    def sample(obj,phase,entity,tick):
        meta=regular[obj+'_subexpression_update'];out={'rand':[],'randn':[]}
        for call in meta['draws'][phase]:
            kind=call['kind'];value=(uniform if kind=='rand' else normal)(p['seed'],sequence,batch,meta['noise_domain'],entity,tick,call['stream'])
            out[kind].append(value);draws[kind].append(value)
        return out
    for tick in range(length):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        for l,layout in enumerate(layouts):
            sh=sample(f'cached_{l}','scalar',0,tick)['rand'][0];z[layout['sh']]=sh
            for j in range(2):
                d=sample(f'cached_{l}','vector',j,tick);u=d['rand'][0];n=d['randn'][0]
                z[layout['u'][j]]=u;z[layout['n'][j]]=n;z[layout['flag'][j]]=float(u>.5)
                z[layout['q'][j]]=weights[gains[l]][j]*u+.01*n+.02*sh+.05*z[layout['v'][j]]+.01*float(u>.5)+p['clock']['origin']+.0002*tick
        for j in range(2):z[syn['r'][j]]=sample('cached_s','vector',j,tick)['rand'][0]+.02*z[layouts[0]['h'][j]]
        active=np.ones(4,dtype=bool)
        for l,layout in enumerate(layouts):
            if refs[l] is not None:
                active[2*l:2*l+2]=z[layout['__refractory_ticks']]<=0;z[layout['__refractory_ticks']]=np.maximum(z[layout['__refractory_ticks']]-1,0)
            for j in range(2):
                q=z[layout['q'][j]];state=z[[layout['v'][j],layout['h'][j]]]
                def f(x):return .2*np.array([(.7-x[0]+q)*active[2*l+j],-x[1]+.1*q])
                a=f(state)
                if methods[l]=='euler':out=state+a
                elif methods[l]=='rk2':out=state+f(state+.5*a)
                elif methods[l]=='rk4':
                    c=f(state+.5*a);d=f(state+.5*c);e=f(state+d);out=state+(a+2*c+2*d+e)/6
                else:raise AssertionError(methods[l])
                z[[layout['v'][j],layout['h'][j]]]=out
        margin=np.concatenate([z[l['v']]-.6-.03*z[l['q']] for l in layouts]);hard=(margin>0)*active;event=hard.astype(float)
        if anchors is not None:
            old=anchors['margins'][tick];hard=anchors['hard'][tick]
            event=hard+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)*active
        margins.append(margin.copy());spikes.append(event.copy())
        w=z[syn['w']].copy();r=z[syn['r']];q=z[layouts[1]['q']]
        guard=active[2:] & ~np.asarray(hard[2:],bool) if refs[1] is not None else np.ones(2,dtype=bool)
        z[layouts[1]['v']]+=event[:2]*(.99*w*r+.01*q)*guard
        z[syn['w']]=w*(1-.01*event[:2])+.002*r*event[2:]
        for l,layout in enumerate(layouts):
            s=event[2*l:2*l+2];q=z[layout['q']];z[layout['v']]-=s*(.3+.02*q);z[layout['h']]+=.01*q*s
            if refs[l] is not None:
                slots=np.array(layout['__refractory_ticks']);z[slots[np.asarray(hard[2*l:2*l+2],bool)]]=2
    spikes=np.asarray(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes,dict(before=before,margins=margins,hard=spikes.astype(bool),draws=draws)


def slots(bundle):
    return sorted({i for layout in [*bundle.provenance['neuron_state_layout'].values(),*bundle.provenance['dynamic_state_layout'].values()] for name,row in layout.items() if not name.startswith('__') for i in row})


@pytest.mark.parametrize('method,ref',[('euler',False),('rk2',True),('rk4',True)])
@pytest.mark.parametrize('warm',[0,2])
def test_cached_random_compiled_brian(engine,method,ref,warm,tmp_path):
    net,groups,syn,bundle=model(method,ref,warm,backend=engine);expected=oracle(bundle)
    bundle.plan['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);spikes=[]
    for start,stop in ((0,3),(3,8)):
        result=trainer.step(np.zeros((1,stop-start,2)),[0],**(dict(noise_sequence=9) if not start else dict(initial='carry')));spikes.extend(result['spikes'][0])
        path=tmp_path/'cached.json';trainer.store(path);trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER);trainer.restore(path)
    np.testing.assert_array_equal(spikes,expected[2]);tol=5e-5 if engine!='cpu' else 5e-12
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots(bundle)],expected[1][slots(bundle)],rtol=tol,atol=tol*.01)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(0*b.ms,namespace={});device=b.get_device();draws=expected[3]['draws'];calls={k:0 for k in draws}
    device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    def sampler(name):
        def refill(n):
            assert n==20000 and calls[name]==0;calls[name]+=1;values=np.zeros(n);values[:len(draws[name])]=draws[name];return values
        return refill
    with patch('numpy.random.rand',sampler('rand')),patch('numpy.random.randn',sampler('randn')):net.run(8*.2*b.ms,namespace={})
    for name in draws:assert calls[name]==1 and getattr(device,name+'_buffer_index')[0]==len(draws[name])
    device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    actual=np.zeros((8,4))
    for l,(g,m) in enumerate(zip(groups,monitors)):
        ticks=np.rint(np.asarray(m.t/b.second)/.0002).astype(int)-warm;actual[ticks,2*l+np.asarray(m.i)]=1
        assert g.subexpression_updater.codeobj.compiled_code['run'] is not None
        for name,row in bundle.provenance['neuron_state_layout'][g.name].items():
            if not name.startswith('__'):np.testing.assert_allclose(expected[1][row],np.broadcast_to(g.variables[name].get_value(),(len(row),)),rtol=5e-12,atol=5e-14)
    assert syn.subexpression_updater.codeobj.compiled_code['run'] is not None
    for name,row in bundle.provenance['dynamic_state_layout'][syn.name].items():np.testing.assert_allclose(expected[1][row],syn.variables[name].get_value(),rtol=5e-12,atol=5e-14)
    np.testing.assert_array_equal(actual,expected[2])
    if engine!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('method',['euler','rk2'])
@pytest.mark.parametrize('window',[None,3])
def test_cached_random_independent_derivatives(engine,method,window):
    _,_,_,bundle=model(method,True,2,backend=engine,tbptt_window=window);expected=oracle(bundle);anchors=expected[3]
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,8,2)),[0],noise_sequence=9)
    tol=3e-3 if engine!='cpu' else 2e-6;eps=1e-6
    np.testing.assert_allclose(result['loss'],expected[0],rtol=tol);np.testing.assert_array_equal(result['spikes'][0],expected[2])
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            a=copy.deepcopy(bundle.weights);c=copy.deepcopy(bundle.weights);a[bank][k]+=eps;c[bank][k]-=eps
            fd=(oracle(bundle,weights=a,anchors=anchors)[0]-oracle(bundle,weights=c,anchors=anchors)[0])/(2*eps)
            np.testing.assert_allclose(result['gradients'][bank][k],fd,rtol=tol,atol=tol*.01,err_msg=f'weight {bank}/{k}')
    initial=np.asarray(bundle.initial_state,float)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,8,2)),[0],initial=initial[None],noise_sequence=9)
    anchors=oracle(bundle,initial=initial)[3]
    discrete=set(bundle.plan['dynamic'].get('binary_states',[]))|set(bundle.plan['dynamic'].get('integer_states',[]))
    for k in slots(bundle):
        if k in discrete:
            assert result['initial_state_gradients'][0][k]==0
            continue
        a=np.array(bundle.initial_state);c=a.copy();a[k]+=eps;c[k]-=eps
        # Explicit initial suppresses trainable initial bank substitution.
        fd=(oracle(bundle,initial=a,anchors=anchors)[0]-oracle(bundle,initial=c,anchors=anchors)[0])/(2*eps)
        np.testing.assert_allclose(result['initial_state_gradients'][0][k],fd,rtol=tol,atol=tol*.01,err_msg=f'initial {k}')


@pytest.mark.parametrize('ranks',[2,8])
def test_cached_random_mpi_restore(engine,ranks,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    _,_,_,bundle=model(warm=2,backend=engine,mpi_ranks=ranks);p=copy.deepcopy(bundle.plan);p.update(backend='cpu',mpi_ranks=None);x=np.zeros((2,8,2))
    a=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0,0],noise_sequence=9)
    c=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0,0],noise_sequence=9)
    for key in ('loss','spikes','final_state','initial_state_gradients'):np.testing.assert_allclose(a[key],c[key],rtol=3e-3 if engine!='cpu' else 4e-12,atol=3e-6 if engine!='cpu' else 4e-13)
    for av,cv in zip(a['gradients'],c['gradients']):np.testing.assert_allclose(av,cv,rtol=3e-3 if engine!='cpu' else 4e-12,atol=3e-6 if engine!='cpu' else 4e-13)
    for batch in range(2):np.testing.assert_array_equal(a['spikes'][batch],oracle(bundle,batch=batch)[2])
    bundle.plan['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    t.step(x[:,:3],[0,0],noise_sequence=9);path=tmp_path/'mpi.json';t.store(path);r=NativeLIFTrainer(bundle.plan,runner=RUNNER);r.restore(path)
    result=r.step(x[:,3:],[0,0],initial='carry');np.testing.assert_allclose(result['final_state'],c['final_state'],rtol=5e-5 if engine!='cpu' else 4e-12,atol=5e-7 if engine!='cpu' else 4e-14)
    assert result['noise_sequence']==9 and result['final_tick']==8
    if engine!='cpu':assert result['gpu_dispatches']>0
