"""True asynchronous action/VJP execution against Brian and fixed recurrences."""
import copy
import os
from unittest.mock import patch

import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_stochastic import normal
from test_training_uniform import uniform


def model(syn_dt=.15,method='euler',warm=0,noisy=False,cached=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms,name='async_input')
    groups=[b.NeuronGroup(2,'dv/dt=(-v+g)/ms:1\ng:1',threshold='v>.55',reset='v-=.3',method='euler',dt=.2*b.ms,name=f'async_n{k}') for k in range(2)]
    groups[0].v=[.7,.3];groups[1].v=[.4,.8];groups[0].g=[.9,1.1]
    drive='drive' if cached else 'w+.1*v_pre+.01*t_pre/ms+.02*t_post/ms+.03*t/ms'
    eq=f'dz/dt=(-z+{drive})/ms'+('+.04*xi/sqrt(ms)' if noisy else '')+':1 (clock-driven)\nw:1\ng_post=z+.03*t/ms:1 (summed)'
    if cached:eq+='\ndrive=w+.1*v_pre+.03*t/ms:1 (constant over dt)'
    syn=b.Synapses(*groups,eq,on_pre='v_post+=.1*w',on_post='z+=.02',method=method,dt=syn_dt*b.ms,name='async_syn')
    syn.connect(j='i');syn.w=[.7,.5];syn.z=[.15,.25]
    net=b.Network(inp,*groups,syn)
    if warm:net.run(warm*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,**options)
    return net,groups,syn,bundle


def oracle(bundle,syn_dt=.15,weights=None,initial=None,anchors=None,steps=6,sequence=17,batch=0):
    """Integer-microsecond Euler visit union; no native clock/SSA interpreter."""
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    z=np.array(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for i,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[i]=weights[ref[0]][ref[1]]
    nl=bundle.provenance['neuron_state_layout'];sl=bundle.provenance['dynamic_state_layout']['async_syn']
    vi=nl['async_n0']['v']+nl['async_n1']['v'];gi=nl['async_n0']['g']+nl['async_n1']['g'];zi=sl['z'];bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='async_syn' and e['variables']==['w']);w=np.asarray(weights[bank])
    start=round(p['dynamic']['clocks']['start']*1e6);main=((start+199)//200)*200;dt=round(syn_dt*1000)
    syn=((start+dt-1)//dt)*dt;end=main+steps*200;calls=0;frame=0;spikes=[];states=[];margins=[];draws=[]
    window=p.get('tbptt_window');noisy=p.get('noise_streams') is not None
    while min(main,syn)<end:
        at=min(main,syn);active_main=main==at;active_syn=syn==at
        if active_main and anchors is not None and window and frame and frame%window==0:z=anchors['before'][frame].copy()
        if active_main:
            states.append(z.copy());z[gi[2:]]=z[zi]+.03*syn/1000
            z[vi]=.8*z[vi]+.2*z[gi]
        if active_syn:
            h=syn_dt
            z[zi]=(1-h)*z[zi]+h*(w+.1*z[vi[:2]]+.03*main/1000+.03*syn/1000)
            if noisy:
                noise=np.array([normal(p['seed'],sequence,batch,2,j,calls,0) for j in range(2)])
                draws.extend(noise);z[zi]+=.04*np.sqrt(h)*noise
            calls+=1
        if active_main:
            margin=z[vi]-.55;hard=(margin>0).astype(float);s=hard.copy()
            if anchors is not None:
                old=anchors['margins'][frame];s=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
            margins.append(margin.copy());spikes.append(s.copy())
            z[vi[2:]]+=.1*w*s[:2];z[zi]+=.02*s[2:];z[vi]-=.3*s
            main+=200;frame+=1
        if active_syn:syn+=dt
    logits=np.array(spikes)[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=states,margins=margins,draws=draws)


CASES=[(.1,'euler',True,False),(.3,'euler',True,False),(.15,'rk2',False,False),(.37,'rk4',False,False),(.15,'euler',False,True)]
@pytest.mark.parametrize('syn_dt,method,noisy,cached',CASES)
@pytest.mark.parametrize('warm',[0.,.65])
def test_async_continuous_and_cache_match_compiled_brian(engine,syn_dt,method,noisy,cached,warm):
    net,groups,syn,bundle=model(syn_dt,method,warm,noisy,cached,backend=engine)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],**(dict(noise_sequence=17) if noisy else {}))
    if noisy:
        expected=oracle(bundle,syn_dt)
        np.testing.assert_allclose(result['final_state'][0],expected[1],rtol=8e-5 if engine!='cpu' else 3e-12,atol=2e-7 if engine!='cpu' else 5e-14)
    net.run(0*b.ms,namespace={});monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    end=bundle.plan['clock']['origin']+6*.0002
    device=b.get_device();device.randn_buffer_index[:]=0;count=[0]
    def refill(n):
        assert n==20000 and count[0]==0;count[0]+=1;values=np.zeros(n);values[:len(expected[3]['draws'])]=expected[3]['draws'];return values
    with patch('numpy.random.randn',refill) if noisy else __import__('contextlib').nullcontext():net.run((end-float(net.t))*b.second,namespace={})
    if noisy:assert count[0]==1 and device.randn_buffer_index[0]==len(expected[3]['draws'])
    device.randn_buffer_index[:]=0
    tolerance=8e-5 if engine!='cpu' else 3e-12;actual=np.asarray(result['final_state'])[0]
    for g in groups:
        for name,indices in bundle.provenance['neuron_state_layout'][g.name].items():np.testing.assert_allclose(actual[indices],g.variables[name].get_value(),rtol=tolerance,atol=tolerance*.01)
    for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():np.testing.assert_allclose(actual[indices],syn.variables[name].get_value(),rtol=tolerance,atol=tolerance*.01)
    spikes=np.zeros((6,4))
    for l,m in enumerate(monitors):spikes[np.rint((np.asarray(m.t/b.second)-bundle.plan['clock']['origin'])/.0002).astype(int),2*l+np.asarray(m.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    assert any(a.get('clock',0)!=0 for a in bundle.plan['dynamic']['actions'])
    if engine!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('syn_dt',[.1,.15,.3])
@pytest.mark.parametrize('window',[None,2])
def test_async_weight_and_initial_vjp_independent(engine,syn_dt,window):
    _,_,_,bundle=model(syn_dt,noisy=True,backend=engine,tbptt_window=window)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],noise_sequence=17)
    loss,z,spikes,anchors=oracle(bundle,syn_dt);tol=5e-3 if engine!='cpu' else 2e-6;eps=1e-6
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    assert result['loss']==pytest.approx(loss,rel=tol)
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][k]+=eps;lo[bank][k]-=eps
            fd=(oracle(bundle,syn_dt,weights=hi,anchors=anchors)[0]-oracle(bundle,syn_dt,weights=lo,anchors=anchors)[0])/(2*eps)
            assert result['gradients'][bank][k]==pytest.approx(fd,rel=tol,abs=tol*.01)
    for k in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][k]:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[k]+=eps;lo[k]-=eps
        fd=(oracle(bundle,syn_dt,initial=hi,anchors=anchors)[0]-oracle(bundle,syn_dt,initial=lo,anchors=anchors)[0])/(2*eps)
        assert result['initial_state_gradients'][0][k]==pytest.approx(fd,rel=tol,abs=tol*.01)


@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('window',[None,2])
def test_async_mpi_batch_carry_restore(engine,ranks,window,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    _,_,_,bundle=model(.15,noisy=True,backend=engine,mpi_ranks=ranks,tbptt_window=window)
    bundle.plan['trainable']=[False]*len(bundle.weights);x=np.zeros((2,6,1))
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);whole=t.gradients(x,[0,0],noise_sequence=17)
    cpu=copy.deepcopy(bundle.plan);cpu.update(backend='cpu',mpi_ranks=None)
    ref=NativeLIFTrainer(cpu,weights=bundle.weights,runner=RUNNER).gradients(x,[0,0],noise_sequence=17)
    for key in ('final_state','spikes','initial_state_gradients'):np.testing.assert_allclose(whole[key],ref[key],rtol=5e-3,atol=3e-6)
    for a,c in zip(whole['gradients'],ref['gradients']):np.testing.assert_allclose(a,c,rtol=5e-3,atol=3e-6)
    t.step(x[:,:2],[0,0],noise_sequence=17);path=tmp_path/'async.json';t.store(path)
    new=NativeLIFTrainer(bundle.plan,runner=RUNNER);new.restore(path)
    tail=new.gradients(x[:,2:],[0,0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=5e-5,atol=3e-7)
    np.testing.assert_array_equal(tail['spikes'],np.asarray(whole['spikes'])[:,2:])
    if engine!='cpu':assert tail['gpu_dispatches']>0


@pytest.mark.parametrize('dt',[.1,.15,.3])
def test_async_regular_shared_indexed_and_own_time(engine,dt):
    from test_training_regular import model as regular_model
    net,inp,groups,synapses,x,_=regular_model('indexed','end')
    for obj in net.sorted_objects:
        if obj.name in ('regular_neuron','regular_synapse','regular_index'):obj._clock=b.Clock(dt=dt*b.ms)
        if obj.name=='regular_synapse':obj.abstract_code+='\nw+=.001*t/ms+.001*dt/ms'
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(x[None],[0])
    net.run(len(x)*.2*b.ms,namespace={});z=np.array(result['final_state'])[0];tol=6e-5 if engine!='cpu' else 3e-12
    for obj in [*groups,*synapses]:
        layout=bundle.provenance['neuron_state_layout' if obj in groups else 'dynamic_state_layout'][obj.name]
        for name,ids in layout.items():
            if name in ('peer','p','q') or name.startswith('__'):continue
            np.testing.assert_allclose(z[ids],np.broadcast_to(obj.variables[name].get_value(),len(ids)),rtol=tol,atol=tol*.01)


@pytest.mark.parametrize('kind',['randn','rand'])
@pytest.mark.parametrize('dt',[.15,.3])
def test_async_regular_random_identity_matches_fixed_brian(engine,kind,dt):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms,name='draw_input')
    groups=[b.NeuronGroup(2,'dv/dt=(-v+g)/ms:1\ng:1\ngain:1 (constant)',threshold='v>.55',reset='v-=.3',method='euler',dt=.2*b.ms,name=f'draw_n{k}') for k in range(2)]
    groups[0].v=[.7,.3];groups[1].v=[.4,.8];groups[0].g=[.9,1.1];groups[1].g=[.35,.65];groups[0].gain=[.02,.03]
    groups[0].run_regularly(f'v+=gain*{kind}()+.01*t/ms+.02*dt/ms',dt=dt*b.ms,when='end',name='draw_regular')
    syn=b.Synapses(*groups,'w:1',on_pre='v_post+=.1*w',dt=.2*b.ms,name='draw_syn');syn.connect(j='i');syn.w=[.7,.5]
    net=b.Network(inp,*groups,syn);bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],noise_sequence=7)
    v=np.array([.7,.3,.4,.8]);g=np.array([.9,1.1,.35,.65]);spikes=[];draws=[];main=regular=calls=0;step=round(dt*1000)
    sample=normal if kind=='randn' else uniform
    while min(main,regular)<1200:
        at=min(main,regular);on_main=main==at;on_regular=regular==at
        if on_main:
            v=.8*v+.2*g;s=(v>.55).astype(float);spikes.append(s.copy());v[2:]+=.1*np.array([.7,.5])*s[:2];v-=.3*s
        if on_regular:
            values=np.array([sample(bundle.plan['seed'],7,0,3,j,calls,0) for j in range(2)]);draws.extend(values)
            # CodeRunner t/dt refer to the owner group, whose pending main
            # clock can be ahead of the runner's current visit.
            v[:2]+=np.array([.02,.03])*values+.01*main/1000+.02*.2;calls+=1
        if on_main:main+=200
        if on_regular:regular+=step
    tol=6e-5 if engine!='cpu' else 3e-12
    np.testing.assert_allclose(result['final_membrane'][0],v,rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    net.run(0*b.ms,namespace={});device=b.get_device();getattr(device,kind+'_buffer_index')[:]=0;refills=[0]
    def refill(n):
        assert n==20000 and refills[0]==0;refills[0]+=1;values=np.zeros(n);values[:len(draws)]=draws;return values
    with patch('numpy.random.'+kind,refill):net.run(1.2*b.ms,namespace={})
    assert refills[0]==1 and getattr(device,kind+'_buffer_index')[0]==len(draws)
    getattr(device,kind+'_buffer_index')[:]=0
    np.testing.assert_allclose(v,np.r_[groups[0].v[:],groups[1].v[:]],rtol=3e-12,atol=3e-14)
    if engine!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('damage',['clock','threshold','spike_trigger','budget'])
def test_async_admission_failure_is_atomic(damage):
    _,_,_,bundle=model();p=bundle.plan
    if damage=='clock':next(a for a in p['dynamic']['actions'] if a.get('clock'))['clock']=256
    elif damage=='threshold':next(a for a in p['dynamic']['actions'] if a['threshold'] is not None)['clock']=1
    elif damage=='spike_trigger':next(a for a in p['dynamic']['actions'] if a['trigger'] is not None and not a['trigger'].get('state',False))['clock']=1
    else:
        # Baseline admits a short main-only tape; the fast async clock must
        # account for its many more action contexts before allocating them.
        p['dynamic']['clocks']['dts'][1]=1e-8;p['max_tape_bytes']=200_000
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);old=copy.deepcopy(t.state)
    with pytest.raises(ValueError):t.step(np.zeros((1,6,1)),[0])
    assert t.state==old and t.neuron_state is None and t.clock_state is None
