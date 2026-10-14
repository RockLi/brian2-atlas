"""Brian multi-clock neuron lowering and pathway-local delay histories."""
import copy
import os
from unittest.mock import patch
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_stochastic import normal
from test_training_event_noise import draw as event_draw


def model(dts=(.1,.3),warm=0.,noisy=False,refractory=False,method='euler',runtime_delay=False,event_driven=False,event_noise=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms,name='mc_input');groups=[]
    for l,dt in enumerate(dts):
        eq='dv/dt=(drive-v)/ms'+('+.02*xi/sqrt(ms)' if noisy else '')+':1'+(' (unless refractory)' if refractory else '')+'\ndrive:1 (constant)'
        g=b.NeuronGroup(2,eq,threshold='v>.6',reset='v-=.4',method=method,dt=dt*b.ms,
            refractory=refractory if isinstance(refractory,str) else .25*b.ms if refractory else False,name=f'mc_n{l}')
        g.v=[.7-.1*l,.35+.1*l];g.drive=[1.6-.3*l,1.8-.2*l];groups.append(g)
    pre='v_post+=w; w+=.02'+('; v_post+=.01*randn()' if event_noise else '')+('; delay=clip(delay+.05*ms,0*ms,.5*ms)' if runtime_delay else '')
    syn=b.Synapses(*groups,'w:1'+('\nda/dt=-a/(2*ms):1 (event-driven)' if event_driven else ''),
        on_pre=pre+('; a+=.03; v_post+=a' if event_driven else ''),
        on_post='w-=.01'+('; w+=.005*randn()' if event_noise else '')+('; w+=.01*a' if event_driven else ''),dt=.17*b.ms,name='mc_syn')
    syn.connect(j='i');syn.w=[.14,.18];
    if event_driven:syn.a=[.2,.1]
    syn.pre.delay=[.15,.35]*b.ms;syn.post.delay=[.25,0]*b.ms
    net=b.Network(inp,*groups,syn)
    if warm:b.seed(511);net.run(warm*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,
        trainable_neuron_parameters={g.name:['drive'] for g in groups},**options)
    return net,inp,groups,syn,bundle


def check_brian(result,bundle,groups,syn,monitors,monitor_start=None):
    assert result['backend']==bundle.plan['backend']
    if result['backend']!='cpu':assert result['gpu_dispatches']>0
    z=np.array(result['final_state'])[0];tol=6e-5 if result['backend']!='cpu' else 5e-12
    for g in groups:
        for name,indices in bundle.provenance['neuron_state_layout'][g.name].items():
            if name.startswith('__'):continue
            np.testing.assert_allclose(z[indices],g.variables[name].get_value(),rtol=tol,atol=tol*.01)
    for name,indices in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(z[indices],syn.variables[name].get_value(),rtol=tol,atol=tol*.01)
    events=result['event_visits'];spikes=np.array(events['spikes'][0]);times=np.array(events['clock_times'])
    for l,m in enumerate(monitors):
        for j in range(len(groups[l])):
            n=2*l+j;old=0 if monitor_start is None else monitor_start[l]
            np.testing.assert_allclose(times[spikes[:,n]>0,events['neuron_clocks'][n]],np.asarray(m.t/b.second)[old:][np.asarray(m.i)[old:]==j],rtol=0,atol=2e-15)
        buf=bundle.provenance['spike_buffer_layout'][groups[l].name];space=groups[l].variables['_spikespace'].get_value();last=np.zeros(2);last[space[:space[-1]]]=1
        np.testing.assert_array_equal(z[buf],last)


@pytest.mark.parametrize('dts',[(.1,.3),(.15,.37)])
@pytest.mark.parametrize('warm',[0.,.65])
@pytest.mark.parametrize('method,refractory',[('euler',False),('rk2',True),('rk4',False)])
def test_multiclock_frontend_actual_brian(engine,dts,warm,method,refractory):
    net,inp,groups,syn,bundle=model(dts,warm,refractory=refractory,method=method,backend=engine)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0])
    net.run((bundle.plan['clock']['origin']+6*.0002-float(net.t))*b.second,namespace={})
    check_brian(result,bundle,groups,syn,monitors)
    for path in bundle.plan['dynamic']['delay_layout']['paths']:
        expected=dts[0 if path['name'].endswith('pre') else 1]/1000
        assert bundle.plan['dynamic']['clocks']['dts'][path.get('clock',0)]==expected


def oracle(bundle,dts,w=None,initial=None,anchors=None,steps=6,sequence=17,batch=0):
    """Integer-time scalar recurrence and FIFO shifts, independent of action IR."""
    p=bundle.plan;d=p['dynamic'];w=bundle.weights if w is None else w
    z=np.array(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for k,ref in enumerate(d['initial_parameters']):
            if ref is not None:z[k]=w[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'];vi=[layout[f'mc_n{l}']['v'] for l in range(2)]
    wi=bundle.provenance['dynamic_state_layout']['mc_syn']['w'];buf=bundle.provenance['spike_buffer_layout']
    banks=[next(a['bank'] for a in bundle.provenance['bindings'] if a['object']==f'mc_n{l}' and a['variables']==['drive']) for l in range(2)]
    paths={a['name']:a for a in d['delay_layout']['paths']}
    start=round(d['clocks']['start']*1e6);delta=np.array([round(x*1000) for x in dts]+[200]);ticks=((start+delta-1)//delta)*delta
    end=ticks[2]+steps*200;calls=[0,0];main=0;frame=0;before=[];margins=[];spikes=[];frames=[];draws=[]
    noisy=any(p.get('noise_streams') or [])
    while ticks.min()<end:
        active=ticks==ticks.min()
        if active[2]:frame=main;main+=1
        if anchors is not None and active[2] and p.get('tbptt_window') and frame and frame%p['tbptt_window']==0:z=anchors['before'][len(before)].copy()
        before.append(z.copy());s=np.zeros(4);margin=np.zeros(4)
        for l in range(2):
            if not active[l]:continue
            h=dts[l];z[vi[l]]+=h*(np.array(w[banks[l]])-z[vi[l]])
            if noisy:
                sample=[normal(p['seed'],sequence,batch,l,j,calls[l],0) for j in range(2)];draws.extend(sample);z[vi[l]]+=.02*np.sqrt(h)*np.array(sample)
        for l in range(2):
            if not active[l]:continue
            for j in range(2):
                n=2*l+j;margin[n]=z[vi[l][j]]-.6;s[n]=float(margin[n]>0)
                if anchors is not None:
                    a=anchors['margins'][len(before)-1][n];s[n]=float(a>0)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(a))**2*(margin[n]-a)
                z[buf[f'mc_n{l}'][j]]=s[n]
        for l,name in enumerate(['mc_syn_pre','mc_syn_post']):
            if not active[l]:continue
            path=paths[name];hist=[]
            def deliver(edge,gate,event,cells):
                noise=0.;action=d['actions'][event]
                if action.get('event_noise') is not None:
                    address=action['event_noise'];noise=event_draw('randn',p['seed'],sequence,batch,action['noise_domain'],edge,calls[l]-len(cells),address.get('pending'),0)
                    if gate:draws.append(noise)
                if l==0:z[vi[1][edge]]+=(z[wi[edge]]+.01*noise)*gate;z[wi[edge]]+=.02*gate
                else:z[wi[edge]]+=(-.01+.005*noise)*gate
            for event in path['pending']:
                cells=event['states'];deliver(event['edge'],z[cells[0]],event['event'],cells);hist.append((cells,0.))
            for edge in sorted(range(2),key=lambda e:(-len(path['edges'][e]['states']),e)):
                cells=path['edges'][edge]['states'];source=z[buf[f'mc_n{l}'][edge]]
                deliver(edge,z[cells[0]] if cells else source,path['edges'][edge]['event'],cells)
                if cells:hist.append((cells,source))
            for cells,source in hist:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=source
        for l in range(2):
            if active[l]:z[vi[l]]-=.4*z[buf[f'mc_n{l}']];calls[l]+=1
        frames.append(frame);spikes.append(s);margins.append(margin);ticks[active]+=delta[active]
    events=np.array(spikes);counts=np.zeros((steps,4))
    for f,s in zip(frames,events):counts[f]+=s
    logits=counts[:,2:].sum(0)*p['logit_scale']/steps;maximum=logits.max();loss=maximum+np.log(np.exp(logits-maximum).sum())-logits[0]
    z[d['spike_buffers']]=0
    return loss,z,counts,events,dict(before=before,margins=margins,draws=draws)


@pytest.mark.parametrize('dts',[(.1,.3),(.15,.37)])
@pytest.mark.parametrize('warm',[0.,.65])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('event_noise',[False,True])
def test_multiclock_stochastic_delayed_independent_vjp(engine,dts,warm,window,event_noise):
    *_,bundle=model(dts,warm,noisy=True,event_noise=event_noise,backend=engine,tbptt_window=window)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],noise_sequence=17)
    loss,z,counts,events,anchors=oracle(bundle,dts);tol=4e-3 if engine!='cpu' else 4e-6
    np.testing.assert_array_equal(result['spikes'][0],counts);np.testing.assert_array_equal(result['event_visits']['spikes'][0],events)
    np.testing.assert_allclose(result['final_state'][0],z,rtol=tol*.1,atol=3e-6)
    assert result['loss']==pytest.approx(loss,rel=tol)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(oracle(bundle,dts,w=hi,anchors=anchors)[0]-oracle(bundle,dts,w=lo,anchors=anchors)[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=tol,abs=tol*.01)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j]:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(oracle(bundle,dts,initial=hi,anchors=anchors)[0]-oracle(bundle,dts,initial=lo,anchors=anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=tol,abs=tol*.01)


@pytest.mark.parametrize('dts',[(.1,.3),(.15,.37)])
@pytest.mark.parametrize('warm',[0.,.65])
@pytest.mark.parametrize('method,event_noise',[('euler',False),('heun',False),('milstein',False),('euler',True)])
def test_multiclock_fixed_noise_actual_brian(engine,dts,warm,method,event_noise):
    net,inp,groups,syn,bundle=model(dts,warm,noisy=True,method=method,event_noise=event_noise,backend=engine)
    expected=oracle(bundle,dts);result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,6,1)),[0],noise_sequence=17)
    net.run(0*b.ms,namespace={});monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    device=b.get_device();device.randn_buffer_index[:]=0;calls=[0];draws=expected[4]['draws']
    def refill(n):
        assert n==20000 and calls[0]==0;calls[0]+=1;values=np.zeros(n);values[:len(draws)]=draws;return values
    with patch('numpy.random.randn',refill):net.run((bundle.plan['clock']['origin']+6*.0002-float(net.t))*b.second,namespace={})
    assert calls[0]==1 and device.randn_buffer_index[0]==len(draws);device.randn_buffer_index[:]=0
    check_brian(result,bundle,groups,syn,monitors)


@pytest.mark.parametrize('dts',[(.1,.3),(.15,.37)])
@pytest.mark.parametrize('warm',[0.,.65])
@pytest.mark.parametrize('runtime',[False,True])
def test_multiclock_delay_rebuild_matches_brian(engine,dts,warm,runtime):
    net,inp,groups,syn,bundle=model(dts,warm,runtime_delay=runtime,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);cursor=0
    for phase,length in enumerate([2,1,3]):
        if phase:
            changes={syn.pre.name:[.00005,.00045],syn.post.name:[.00055,.00015]} if phase==1 else {syn.pre.name:[.0003,0.],syn.post.name:[0.,.0004]}
            before=copy.deepcopy(trainer.clock_state);trainer.update_delays(changes);assert trainer.clock_state==before
            for path in (syn.pre,syn.post):path.delay=np.array(changes[path.name])*b.second
            for path in trainer.plan['dynamic']['delay_layout']['paths']:
                dt=trainer.plan['dynamic']['clocks']['dts'][path.get('clock',0)]
                expected=np.floor(np.array(changes[path['name']])/dt+.5).astype(int)
                assert [len(e['states']) for e in path['edges']]==expected.tolist()
        starts=[len(m.t) for m in monitors]
        result=trainer.step(np.zeros((1,length,1)),[0],initial='carry' if phase else None)
        cursor+=length;net.run((bundle.plan['clock']['origin']+cursor*.0002-float(net.t))*b.second,namespace={})
        check_brian(result,bundle,groups,syn,monitors,starts)
        for name,layout in bundle.provenance.get('pathway_state_layout',{}).items():
            actual=next(p for p in (syn.pre,syn.post) if p.name==name)
            np.testing.assert_allclose(np.asarray(result['final_state'])[0,layout['delay']],np.asarray(actual.delay/b.second),rtol=2e-5,atol=2e-9)


@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('window',[None,2])
def test_multiclock_mpi_batch_checkpoint_and_migration(engine,ranks,window,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,syn,bundle=model((.15,.37),.65,noisy=True,runtime_delay=True,backend=engine,mpi_ranks=ranks,tbptt_window=window)
    bundle.plan['trainable']=[False]*len(bundle.weights);x=np.zeros((2,6,1));trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    trainer.step(x[:,:2],[0,1],noise_sequence=17)
    changes={syn.pre.name:[.00005,.00045],syn.post.name:[.00055,.00015]};trainer.update_delays(changes)
    # Mask migration must leave group-owned buffers and clock identity intact.
    masks=copy.deepcopy(trainer.plan['masks']);bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='mc_syn')
    buffers=trainer.plan['dynamic']['spike_buffers'];saved_clock=copy.deepcopy(trainer.clock_state)
    masks[bank][0]=0;trainer.update_mask(masks);masks[bank][0]=1;trainer.update_mask(masks)
    assert trainer.clock_state==saved_clock;assert np.asarray(trainer.neuron_state)[:,buffers].sum()==0
    p=copy.deepcopy(trainer.plan);p.update(backend='cpu',mpi_ranks=None)
    expected=NativeLIFTrainer(p,weights=trainer.state['weights'],runner=RUNNER).gradients(x[:,2:],[0,1],initial=trainer.neuron_state,start_tick=trainer.clock_tick,clock_state=trainer.clock_state,noise_sequence=17)
    checkpoint=tmp_path/'mc.json';trainer.store(checkpoint);restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(checkpoint)
    result=restored.gradients(x[:,2:],[0,1],initial='carry')
    for key in ('spikes','final_state','initial_state_gradients','logits'):
        np.testing.assert_allclose(result[key],expected[key],rtol=5e-3,atol=5e-6)
    for actual,reference in zip(result['gradients'],expected['gradients']):np.testing.assert_allclose(actual,reference,rtol=5e-3,atol=5e-6)
    np.testing.assert_array_equal(result['event_visits']['spikes'],expected['event_visits']['spikes'])


@pytest.mark.parametrize('bad',['clock','action','source','clear_without_buffers'])
def test_multiclock_layout_rejects_inconsistent_clocks(bad):
    *_,bundle=model();p=copy.deepcopy(bundle.plan);d=p['dynamic'];path=d['delay_layout']['paths'][0]
    if bad=='clock':path['clock']=999
    elif bad=='action':d['actions'][path['start']]['clock']=0
    elif bad=='source':path['edges'][0]['source']['index']=d['spike_buffers'][-1]
    else:d['spike_buffers']=[]
    with pytest.raises((RuntimeError,ValueError)) as error:NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).evaluate(np.zeros((1,2,1)),[0])
    assert 'panicked' not in str(error.value)


@pytest.mark.parametrize('warm',[0.,.65])
def test_multiclock_event_driven_synapse(engine,warm):
    net,inp,groups,syn,bundle=model((.15,.37),warm,event_driven=True,backend=engine)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0])
    net.run((bundle.plan['clock']['origin']+6*.0002-float(net.t))*b.second,namespace={})
    check_brian(result,bundle,groups,syn,monitors)


def test_multiclock_external_input_delayed_pathway(engine):
    net,inp,groups,syn,_=model((.15,.37))
    inp.set_spikes([0,0,0],np.array([0.,.4,.8])*b.ms)
    external=b.Synapses(inp,groups[0],'w:1',on_pre='v_post+=w',name='mc_external');external.connect();external.w=[.15,.21];external.pre.delay=.2*b.ms;net.add(external)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine,detach_reset=False)
    x=np.zeros((1,6,1));x[0,[0,2,4],0]=1
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0])
    net.run(1.2*b.ms,namespace={});check_brian(result,bundle,groups,syn,monitors)


@pytest.mark.parametrize('warm',[0.,.65])
@pytest.mark.parametrize('expression',['v>.4','(.2+.1*v)*ms'])
def test_multiclock_state_refractory_actual_brian(engine,warm,expression):
    net,inp,groups,syn,bundle=model((.15,.37),warm,refractory=expression,backend=engine)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0])
    net.run((bundle.plan['clock']['origin']+6*.0002-float(net.t))*b.second,namespace={});check_brian(result,bundle,groups,syn,monitors)
