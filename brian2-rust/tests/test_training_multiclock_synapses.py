"""Combined neuron clocks, continuous SDE synapses, summed and delayed VJPs."""
import copy
import os
from unittest.mock import patch
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_stochastic import normal


def model(warm=0.,syn_dt=.15,cached=False,method='euler',**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms,name='mix_input')
    groups=[b.NeuronGroup(2,'dv/dt=(-v+g)/ms:1\ng:1',threshold='v>.55',reset='v-=.3',
            method='euler',dt=dt*b.ms,name=f'mix_n{k}') for k,dt in enumerate([.1,.3])]
    groups[0].v=[.7,.3];groups[1].v=[.4,.8];groups[0].g=[.9,1.1]
    drive='w+.1*v_pre+.01*t_pre/ms+.02*t_post/ms+.03*t/ms'
    eq='dz/dt=(-z+'+('drive' if cached else drive)+')/ms+.04*xi/sqrt(ms):1 (clock-driven)\nw:1\ng_post=z+.03*t/ms:1 (summed)'
    if cached:eq+='\ndrive='+drive+'+.02*randn():1 (constant over dt)'
    syn=b.Synapses(*groups,eq,on_pre='v_post+=.1*w',on_post='z+=.02',method=method,dt=syn_dt*b.ms,name='mix_syn')
    syn.connect(j='i');syn.w=[.7,.5];syn.z=[.15,.25]
    syn.pre.delay=[.2,.3]*b.ms;syn.post.delay=[.3,0]*b.ms
    net=b.Network(inp,*groups,syn)
    if warm:b.seed(717);net.run(warm*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,**options)
    return net,inp,groups,syn,bundle


def oracle(bundle,syn_dt=.15,weights=None,initial=None,anchors=None,steps=6,sequence=17,batch=0):
    """Explicit scalar recurrence and FIFO events on integer microsecond clocks.

    Uses physical state/parameter layout, not the action programs or native
    clock tape. Summed assignment runs on the target neuron clock before all
    group integrators; synaptic time expressions read each clock's pending t.
    """
    p=bundle.plan;d=p['dynamic'];weights=bundle.weights if weights is None else weights
    z=np.array(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for i,ref in enumerate(d['initial_parameters']):
            if ref is not None:z[i]=weights[ref[0]][ref[1]]
    nl=bundle.provenance['neuron_state_layout'];sl=bundle.provenance['dynamic_state_layout']['mix_syn']
    vi=[nl[f'mix_n{k}']['v'] for k in range(2)];gi=[nl[f'mix_n{k}']['g'] for k in range(2)];zi=sl['z']
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='mix_syn' and e['variables']==['w']);w=np.array(weights[bank])
    buf=[bundle.provenance['spike_buffer_layout'][f'mix_n{k}'] for k in range(2)]
    start=round(d['clocks']['start']*1e6);delta=np.array([100,300,round(syn_dt*1000),200]);ticks=((start+delta-1)//delta)*delta
    end=ticks[3]+steps*200;main=0;frame=0;calls=0;before=[];margins=[];events=[];frames=[];draws=[]
    paths={a['name']:a for a in d['delay_layout']['paths']};cached='drive' in sl
    while ticks.min()<end:
        active=ticks==ticks.min()
        if active[3]:frame=main;main+=1
        if anchors is not None and active[3] and p.get('tbptt_window') and frame and frame%p['tbptt_window']==0:z=anchors['before'][len(before)].copy()
        before.append(z.copy())
        if active[2] and cached:
            regular=bundle.provenance['regular_runner_layout'];assert len(regular)==1
            meta=next(iter(regular.values()))
            cache_noise=[normal(p['seed'],sequence,batch,meta['noise_domain'],j,calls,0) for j in range(2)]
            draws.extend(cache_noise)
            z[sl['drive']]=w+.1*z[vi[0]]+.01*ticks[0]/1000+.02*ticks[1]/1000+.03*ticks[2]/1000+.02*np.array(cache_noise)
            scalar=meta['scalar']
            assert set(scalar)=={'t','t_pre','t_post'}
            for name,k in [('t',2),('t_pre',0),('t_post',1)]:z[scalar[name]]=ticks[k]/1e6
        if active[1]:z[gi[1]]=z[zi]+.03*ticks[2]/1000
        for k in range(2):
            if active[k]:z[vi[k]]+=delta[k]/1000*(z[gi[k]]-z[vi[k]])
        if active[2]:
            drive=z[sl['drive']] if cached else w+.1*z[vi[0]]+.01*ticks[0]/1000+.02*ticks[1]/1000+.03*ticks[2]/1000
            z[zi]+=syn_dt*(drive-z[zi]);noise=[normal(p['seed'],sequence,batch,2,j,calls,0) for j in range(2)]
            draws.extend(noise);z[zi]+=.04*np.sqrt(syn_dt)*np.array(noise);calls+=1
        s=np.zeros(4);margin=np.zeros(4)
        for k in range(2):
            if not active[k]:continue
            for j in range(2):
                n=k*2+j;margin[n]=z[vi[k][j]]-.55;s[n]=float(margin[n]>0)
                if anchors is not None:
                    a=anchors['margins'][len(before)-1][n];s[n]=float(a>0)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(a))**2*(margin[n]-a)
                z[buf[k][j]]=s[n]
        for k,name in enumerate(['mix_syn_pre','mix_syn_post']):
            if not active[k]:continue
            path=paths[name];hist=[]
            def deliver(edge,gate):
                if k==0:z[vi[1][edge]]+=.1*w[edge]*gate
                else:z[zi[edge]]+=.02*gate
            for event in path['pending']:
                cells=event['states'];deliver(event['edge'],z[cells[0]]);hist.append((cells,0.))
            for edge in sorted(range(2),key=lambda e:(-len(path['edges'][e]['states']),e)):
                cells=path['edges'][edge]['states'];source=z[buf[k][edge]]
                deliver(edge,z[cells[0]] if cells else source)
                if cells:hist.append((cells,source))
            for cells,source in hist:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=source
        for k in range(2):
            if active[k]:z[vi[k]]-=.3*z[buf[k]]
        frames.append(frame);events.append(s);margins.append(margin);ticks[active]+=delta[active]
    counts=np.zeros((steps,4))
    for frame,s in zip(frames,events):counts[frame]+=s
    logits=counts[:,2:].sum(0)*p['logit_scale']/steps;peak=logits.max();loss=peak+np.log(np.exp(logits-peak).sum())-logits[0]
    z[d['spike_buffers']]=0
    return loss,z,counts,np.array(events),dict(before=before,margins=margins,draws=draws)


def check_brian(result,bundle,groups,syn,monitors):
    tol=7e-5 if bundle.plan['backend']!='cpu' else 4e-12;z=np.array(result['final_state'])[0]
    for obj in [*groups,syn]:
        for name,indices in bundle.provenance['neuron_state_layout' if obj in groups else 'dynamic_state_layout'][obj.name].items():
            if not name.startswith('__'):np.testing.assert_allclose(z[indices],obj.variables[name].get_value(),rtol=tol,atol=tol*.01)
    ev=result['event_visits'];s=np.array(ev['spikes'][0]);t=np.array(ev['clock_times'])
    for k,m in enumerate(monitors):
        for j in range(2):
            n=2*k+j;np.testing.assert_allclose(t[s[:,n]>0,ev['neuron_clocks'][n]],np.asarray(m.t/b.second)[np.asarray(m.i)==j],rtol=0,atol=2e-15)
    np.testing.assert_array_equal(z[bundle.plan['dynamic']['spike_buffers']],0)
    if bundle.plan['backend']!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('warm',[0.,.61])
@pytest.mark.parametrize('syn_dt,cached,method',[(.15,False,'euler'),(.37,False,'heun'),(.15,True,'milstein')])
def test_multiclock_continuous_summed_fixed_noise_brian(engine,warm,syn_dt,cached,method):
    net,_,groups,syn,bundle=model(warm,syn_dt,cached,method,backend=engine)
    expected=oracle(bundle,syn_dt);result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],noise_sequence=17)
    np.testing.assert_allclose(result['final_state'][0],expected[1],rtol=7e-5 if engine!='cpu' else 4e-12,atol=3e-7 if engine!='cpu' else 5e-14)
    np.testing.assert_array_equal(result['event_visits']['spikes'][0],expected[3])
    net.run(0*b.ms,namespace={});monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    device=b.get_device();device.randn_buffer_index[:]=0;calls=[0];draws=expected[4]['draws']
    def refill(n):
        assert n==20000 and calls[0]==0;calls[0]+=1;v=np.zeros(n);v[:len(draws)]=draws;return v
    with patch('numpy.random.randn',refill):net.run((bundle.plan['clock']['origin']+6*.0002-float(net.t))*b.second,namespace={})
    assert calls[0]==1 and device.randn_buffer_index[0]==len(draws);device.randn_buffer_index[:]=0
    check_brian(result,bundle,groups,syn,monitors)


@pytest.mark.parametrize('warm',[0.,.61])
@pytest.mark.parametrize('cached',[False,True])
@pytest.mark.parametrize('window',[None,2])
def test_multiclock_continuous_summed_full_vjp(engine,warm,cached,window):
    *_,bundle=model(warm,cached=cached,backend=engine,tbptt_window=window)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],noise_sequence=17)
    loss,z,counts,events,anchors=oracle(bundle);tol=5e-3 if engine!='cpu' else 4e-6
    np.testing.assert_array_equal(result['spikes'][0],counts);np.testing.assert_array_equal(result['event_visits']['spikes'][0],events)
    assert result['loss']==pytest.approx(loss,rel=tol)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(oracle(bundle,weights=hi,anchors=anchors)[0]-oracle(bundle,weights=lo,anchors=anchors)[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=tol,abs=tol*.01)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j]:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(oracle(bundle,initial=hi,anchors=anchors)[0]-oracle(bundle,initial=lo,anchors=anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=tol,abs=tol*.01)


@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('window',[None,2])
def test_multiclock_continuous_summed_mpi_checkpoint(engine,ranks,window,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,bundle=model(.61,cached=True,backend=engine,mpi_ranks=ranks,tbptt_window=window)
    bundle.plan['trainable']=[False]*len(bundle.weights);x=np.zeros((2,6,1))
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    whole=trainer.gradients(x,[0,1],noise_sequence=17)
    cpu=copy.deepcopy(bundle.plan);cpu.update(backend='cpu',mpi_ranks=None)
    reference=NativeLIFTrainer(cpu,weights=bundle.weights,runner=RUNNER).gradients(x,[0,1],noise_sequence=17)
    for key in ('final_state','spikes','initial_state_gradients'):
        np.testing.assert_allclose(whole[key],reference[key],rtol=5e-3,atol=3e-6)
    for a,c in zip(whole['gradients'],reference['gradients']):np.testing.assert_allclose(a,c,rtol=5e-3,atol=3e-6)
    trainer.step(x[:,:2],[0,1],noise_sequence=17);path=tmp_path/'combined.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER);restored.restore(path)
    tail=restored.gradients(x[:,2:],[0,1],initial='carry')
    # Brian clears threshold buffers between runs, while queue history and
    # continuous noise positions remain carried across the boundary.
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=7e-5,atol=4e-7)
    np.testing.assert_array_equal(tail['spikes'],np.asarray(whole['spikes'])[:,2:])


@pytest.mark.parametrize('warm',[0.,.65])
@pytest.mark.parametrize('when',['after_synapses','end'])
def test_multiclock_shared_linked_indexed_delayed_brian(engine,warm,when):
    from test_training_regular import model as regular_model
    net,inp,groups,synapses,x,_=regular_model('indexed',when)
    for group,dt in zip(groups,[.1,.3]):group.clock.dt=dt*b.ms
    synapses[0].clock.dt=.15*b.ms;synapses[0].pre.delay=[.2,.1,.3,.4]*b.ms
    for obj in net.sorted_objects:
        if obj.name=='regular_index':obj._clock=b.Clock(dt=.17*b.ms)
    if warm:net.run(warm*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine,detach_reset=False,
        trainable_neuron_parameters={groups[0].name:['gain']})
    start=round(bundle.plan['clock']['origin']/.0002);x=x[start:]
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    net.run((bundle.plan['clock']['origin']+len(x)*.0002-float(net.t))*b.second,namespace={})
    z=np.array(result['final_state'])[0];tol=7e-5 if engine!='cpu' else 4e-12
    for group in groups:
        for name,indices in bundle.provenance['neuron_state_layout'][group.name].items():
            if name.startswith('__'):continue
            actual=z[indices];expected=group.variables[name].get_value()
            if name=='peer':
                expected=np.array(groups[0].v)[np.array(group.pick)];actual=[]
                for descriptor in bundle.provenance['runtime_index_layout'][group.name][name]:
                    index=int(z[descriptor['index']])
                    for table in descriptor['tables'][:-1]:index=int(z[table[index]])
                    actual.append(z[descriptor['tables'][-1][index]])
            np.testing.assert_allclose(actual,np.broadcast_to(expected,len(group)),rtol=tol,atol=tol*.01)
    for syn in synapses:
        for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
            np.testing.assert_allclose(z[indices],syn.variables[name].get_value(),rtol=tol,atol=tol*.01)
    ev=result['event_visits'];s=np.array(ev['spikes'][0]);t=np.array(ev['clock_times'])
    for k,m in enumerate(monitors):
        for j in range(3):
            n=3*k+j;np.testing.assert_allclose(t[s[:,n]>0,ev['neuron_clocks'][n]],np.asarray(m.t/b.second)[np.asarray(m.i)==j],rtol=0,atol=2e-15)
    if engine!='cpu':assert result['gpu_dispatches']>0
