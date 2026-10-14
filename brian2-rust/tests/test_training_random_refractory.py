"""Direct random refractory: conditional Brian draws and fixed-control VJPs."""
import copy
import gc
import os
from unittest.mock import patch
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_neuron_noise import cython_cache
from test_training_uniform import uniform
from test_training_stochastic import normal


def model(kind='boolean',method='euler',warm=0,**options):
    gc.collect();b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[],[]*b.ms,dt=dt,name='raw_ref_input');groups=[]
    for k in range(2):
        g=b.NeuronGroup(2,'''dv/dt=(.7-v+.15*h+.03*gain*randn())/ms:1 (unless refractory)
dh/dt=-h/ms:1
gain:1 (constant)''',threshold='v>.6+.02*rand()',reset='u=rand();v-=.3+.01*u;h+=.02*u',
            refractory='randn()>0' if kind=='boolean' else '(1+int(3*rand()))*dt',method=method,dt=dt,name=f'raw_ref_{k}')
        g.v=[.8-.04*k,.9+.03*k];g.h=[.1,.2];g.gain=[.1+.02*k,.2];groups.append(g)
    syn=b.Synapses(*groups,'w:1',on_pre='v_post+=w',dt=dt,name='raw_ref_s');syn.connect(j='i');syn.w=[.12,.16];net=b.Network(source,*groups,syn)
    if warm:b.seed(619);net.run(warm*dt,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=groups,detach_reset=False,
        trainable_neuron_parameters={g.name:['gain'] for g in groups},**options)
    return net,groups,bundle


def oracle(bundle,kind,weights=None,initial=None,anchors=None,length=8,sequence=9,batch=0):
    p=bundle.plan;weights=bundle.weights if weights is None else weights;z=np.array(bundle.initial_state if initial is None else initial,float)
    layouts=[bundle.provenance['neuron_state_layout'][f'raw_ref_{k}'] for k in range(2)]
    gains=[next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==f'raw_ref_{k}' and e['variables']==['gain']) for k in range(2)]
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='raw_ref_s')
    oldactive=np.array([z[bundle.provenance['refractory_activity_layout'][f'raw_ref_{k}']] for k in range(2)],bool)
    last=np.array([z[bundle.provenance['refractory_lastspike_layout'][f'raw_ref_{k}']] for k in range(2)])
    calls=bundle.provenance['neuron_draws'];methods=bundle.provenance['integrators'];draws={'rand':[],'randn':[]};before=[];margins=[];spikes=[]
    def phase(name,l,j,t,record=True):
        values={'rand':[],'randn':[]}
        for call in calls[l][name]:
            k=call['kind'];value=(uniform if k=='rand' else normal)(p['seed'],sequence,batch,l,j,t,call['stream']);values[k].append(value)
            if record:draws[k].append(value)
        return values
    for t in range(length):
        if anchors is not None and p['tbptt_window'] and t and t%p['tbptt_window']==0:z=anchors['before'][t].copy()
        before.append(z.copy());active=np.ones((2,2),bool);time=p['clock']['origin']+t*.0002
        for l,layout in enumerate(layouts):
            for j in range(2):
                ref=phase('refractory',l,j,t,record=kind!='boolean' or not oldactive[l,j])
                if kind=='boolean':active[l,j]=oldactive[l,j] or ref['randn'][0]<=0
                else:active[l,j]=int((time-last[l,j]+.0002*.001)/.0002)>=1+int(3*ref['rand'][0])
                ns=phase('update',l,j,t)['randn'];state=z[[layout['v'][j],layout['h'][j]]];gain=weights[gains[l]][j]
                def f(x,stage):return .2*np.array([(.7-x[0]+.15*x[1]+.03*gain*ns[stage])*active[l,j],-x[1]])
                a=f(state,0)
                if methods[l]=='euler':out=state+a
                elif methods[l]=='rk2':out=state+f(state+.5*a,1)
                elif methods[l]=='rk4':
                    c=f(state+.5*a,1);d=f(state+.5*c,2);e=f(state+d,3);out=state+(a+2*c+2*d+e)/6
                else:raise AssertionError(methods[l])
                z[[layout['v'][j],layout['h'][j]]]=out
        margin=np.zeros(4)
        for l,layout in enumerate(layouts):
            for j in range(2):margin[2*l+j]=z[layout['v'][j]]-.6-.02*phase('threshold',l,j,t)['rand'][0]
        hard=(margin>0)&active.ravel();event=hard.astype(float)
        if anchors is not None:
            old=anchors['margins'][t];hard=anchors['hard'][t];event=hard+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)*active.ravel()
        margins.append(margin.copy());spikes.append(event.copy())
        z[layouts[1]['v']]+=np.array(weights[bank])*event[:2]*(active[1]&~np.asarray(hard[2:],bool))
        for l,layout in enumerate(layouts):
            for j in range(2):
                u=phase('reset',l,j,t,record=bool(hard[2*l+j]))['rand'][0]
                z[layout['v'][j]]-=event[2*l+j]*(.3+.01*u);z[layout['h'][j]]+=event[2*l+j]*.02*u
                oldactive[l,j]=active[l,j] and not hard[2*l+j]
                if hard[2*l+j]:last[l,j]=time
    spikes=np.array(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes,dict(before=before,margins=margins,hard=spikes.astype(bool),draws=draws)


def physical(bundle):return sum((l['v']+l['h'] for l in bundle.provenance['neuron_state_layout'].values()),[])


@pytest.mark.parametrize('kind',['boolean','duration'])
@pytest.mark.parametrize('method',['euler','rk2','rk4'])
@pytest.mark.parametrize('warm',[0,2])
def test_random_refractory_compiled_brian(engine,kind,method,warm,tmp_path):
    net,groups,bundle=model(kind,method,warm,backend=engine);expected=oracle(bundle,kind);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);spikes=[]
    for start,stop in ((0,3),(3,8)):
        result=trainer.step(np.zeros((1,stop-start,2)),[0],**(dict(noise_sequence=9) if not start else dict(initial='carry')));spikes.extend(result['spikes'][0])
        path=tmp_path/'ref.json';trainer.store(path);trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER);trainer.restore(path)
    np.testing.assert_array_equal(spikes,expected[2]);tol=5e-5 if engine!='cpu' else 5e-12
    np.testing.assert_allclose(np.array(result['final_state'])[0,physical(bundle)],expected[1][physical(bundle)],rtol=tol,atol=tol*.01)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(0*b.ms,namespace={});device=b.get_device();device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0;draws=expected[3]['draws'];calls={k:0 for k in draws}
    def sampler(name):
        def refill(n):
            assert n==20000 and calls[name]==0;calls[name]+=1;values=np.zeros(n);values[:len(draws[name])]=draws[name];return values
        return refill
    with patch('numpy.random.rand',sampler('rand')),patch('numpy.random.randn',sampler('randn')):net.run(8*.2*b.ms,namespace={})
    for name in draws:assert calls[name]==1 and getattr(device,name+'_buffer_index')[0]==len(draws[name])
    device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0;actual=np.zeros((8,4))
    for l,(g,m) in enumerate(zip(groups,monitors)):
        assert g.state_updater.codeobj.compiled_code['run'] is not None
        actual[np.rint(np.array(m.t/b.second)/.0002).astype(int)-warm,2*l+np.array(m.i)]=1
        for name in ('v','h'):np.testing.assert_allclose(g.variables[name].get_value(),expected[1][bundle.provenance['neuron_state_layout'][g.name][name]],rtol=5e-12,atol=5e-14)
    np.testing.assert_array_equal(actual,expected[2])
    if engine!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('kind',['boolean','duration'])
@pytest.mark.parametrize('window',[None,3])
def test_random_refractory_independent_derivatives(engine,kind,window):
    _,_,bundle=model(kind,'rk2',2,backend=engine,tbptt_window=window);expected=oracle(bundle,kind);tol=3e-3 if engine!='cpu' else 2e-6;eps=1e-6
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,8,2)),[0],noise_sequence=9)
    np.testing.assert_array_equal(result['spikes'][0],expected[2]);np.testing.assert_allclose(result['loss'],expected[0],rtol=tol)
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            a=copy.deepcopy(bundle.weights);c=copy.deepcopy(bundle.weights);a[bank][k]+=eps;c[bank][k]-=eps
            fd=(oracle(bundle,kind,weights=a,anchors=expected[3])[0]-oracle(bundle,kind,weights=c,anchors=expected[3])[0])/(2*eps)
            np.testing.assert_allclose(result['gradients'][bank][k],fd,rtol=tol,atol=tol*.01)
    for k in physical(bundle):
        a=np.array(bundle.initial_state);c=a.copy();a[k]+=eps;c[k]-=eps
        fd=(oracle(bundle,kind,initial=a,anchors=expected[3])[0]-oracle(bundle,kind,initial=c,anchors=expected[3])[0])/(2*eps)
        np.testing.assert_allclose(result['initial_state_gradients'][0][k],fd,rtol=tol,atol=tol*.01)


@pytest.mark.parametrize('kind',['boolean','duration'])
@pytest.mark.parametrize('ranks',[2,8])
def test_random_refractory_mpi_restore(engine,kind,ranks,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    _,_,bundle=model(kind,'rk2',2,backend=engine,mpi_ranks=ranks);p=copy.deepcopy(bundle.plan);p.update(backend='cpu',mpi_ranks=None);x=np.zeros((2,8,2))
    a=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0,0],noise_sequence=9)
    c=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0,0],noise_sequence=9)
    for key in ('loss','spikes','final_state','initial_state_gradients'):np.testing.assert_allclose(a[key],c[key],rtol=3e-3 if engine!='cpu' else 4e-12,atol=3e-6 if engine!='cpu' else 4e-13)
    for av,cv in zip(a['gradients'],c['gradients']):np.testing.assert_allclose(av,cv,rtol=3e-3 if engine!='cpu' else 4e-12,atol=3e-6 if engine!='cpu' else 4e-13)
    for batch in range(2):np.testing.assert_array_equal(a['spikes'][batch],oracle(bundle,kind,batch=batch)[2])
    bundle.plan['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    t.step(x[:,:3],[0,0],noise_sequence=9);path=tmp_path/'mpi.json';t.store(path);r=NativeLIFTrainer(bundle.plan,runner=RUNNER);r.restore(path);tail=r.step(x[:,3:],[0,0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],c['final_state'],rtol=5e-5 if engine!='cpu' else 4e-12,atol=5e-7 if engine!='cpu' else 4e-14)
    if engine!='cpu':assert tail['gpu_dispatches']>0 and a['gpu_dispatches']>0
