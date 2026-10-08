"""Independent emission-addressed event bins, actual Cython, and pathwise VJPs."""
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


def draw(kind,seed,sequence,batch,domain,edge,emitted,pending,stream):
    mask=(1<<64)-1
    def mix(x):
        x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask;x=((x^(x>>27))*0x94d049bb133111eb)&mask
        return x^(x>>31)
    key=mix(seed^(0x4232455655303031 if kind=='rand' else 0x423245564e303031))
    for value in [sequence,batch,domain,edge,0 if pending else emitted&mask,int(bool(pending)),pending or 0,stream]:
        key=mix(key^mix((value+0x9e3779b97f4a7c15)&mask))
    bits=mix(key^0xa0761d6478bd642f)>>11
    if kind=='rand':return bits/2**53
    u=(bits+.5)/2**53;v=((mix(key^0xe7037ed1a0b428db)>>11)+.5)/2**53
    return np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)


def model(warm=0,variable=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.ones((12,2));ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='event_input')
    a=b.NeuronGroup(2,'dv/dt=(.7-v)/ms:1',threshold='v>.6',reset='v-=.4',method='euler',dt=dt,name='event_a')
    c=b.NeuronGroup(2,'dv/dt=(.5-v)/ms:1',threshold='v>.6',reset='v-=.4',method='euler',dt=dt,name='event_c')
    a.v=[1.2,.95];c.v=[.8,.4]
    drive=b.Synapses(inp,a,'w:1',on_pre='v_post+=w',dt=dt,name='event_drive');drive.connect(j='i');drive.w=.18
    pre='u=rand()\nn=randn()\nw=.95*w+.015*u\nv_post+=w*(.6+.4*u)+.01*a+.002*n\na+=.03*u\ncount+=1'
    if variable:pre+='\ndelay=(count%3)*dt'
    s=b.Synapses(a,c,'w:1\nda/dt=-a/(2*ms):1 (event-driven)\ncount:integer',on_pre=pre,
        on_post='u=rand()\nw+=.003*u\na*=.9',dt=dt,name='event_s')
    s.connect(i=[0,1,0,1],j=[0,1,1,0]);s.w=[.15,.18,.21,.12];s.a=[.1,.2,.3,.4];s.count=0
    s.pre.delay=.6*b.ms;s.post.delay=.2*b.ms
    net=b.Network(inp,a,c,drive,s)
    if warm:b.seed(991);net.run(warm*dt,namespace={});x=x[warm:]
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],detach_reset=False,**options)
    return net,[a,c],s,x,bundle


def oracle(bundle,x,weights=None,initial=None,anchors=None,change=True,variable=False,sequence=9,latch_each=True,batch=0):
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    z=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for k,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[k]=weights[ref[0]][ref[1]]
    nl=bundle.provenance['neuron_state_layout'];a=nl['event_a']['v'];c=nl['event_c']['v']
    sl=bundle.provenance['dynamic_state_layout']['event_s'];w=sl['w'];trace=sl['a'];last=sl['lastupdate'];count=sl['count']
    drive=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='event_drive')
    pre=np.array([0,1,0,1]);post=np.array([0,1,1,0]);names=['event_s_pre','event_s_post']
    domains=[bundle.provenance['scheduled_noise_domains'][name] for name in names]
    bins=[{},{}];delays=[np.full(4,3),np.ones(4,dtype=int)];draws={'rand':[],'randn':[]}
    layouts=[next(path for path in p['dynamic']['delay_layout']['paths'] if path['name']==name) for name in names]
    for path,layout in enumerate(layouts):
        for pending in layout['pending']:
            address=p['dynamic']['actions'][pending['event']]['event_noise']
            for offset,slot in enumerate(pending['states']):
                if z[slot]:bins[path].setdefault(offset,[]).append((pending['edge'],z[slot],True,0,address['pending']))
    delay_state=bundle.provenance['pathway_state_layout'].get('event_s_pre',{}).get('delay')
    margins=[];before=[];spikes=[];collisions=[]
    for tick,external in enumerate(x):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
            z,bins=copy.deepcopy(anchors['before'][tick])
        before.append(copy.deepcopy((z,bins)))
        if variable and (tick==0 or latch_each):delays[0]=np.floor(z[delay_state]/.0002+.5).astype(int) if anchors is None else anchors['delays'][tick].copy()
        elif not variable and change and tick==3:delays=[np.zeros(4,dtype=int),np.zeros(4,dtype=int)]
        if anchors is None:
            if tick==0:delay_history=[]
            delay_history.append(delays[0].copy())
        z[a]=.8*z[a]+.14;z[c]=.8*z[c]+.1
        margin=z[a+c]-.6;hard=(margin>0).astype(float);event=hard.copy()
        if anchors is not None:
            old=anchors['margins'][tick];hard=(old>0).astype(float)
            event=hard+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());spikes.append(event.copy());z[a]+=np.asarray(weights[drive])*external
        time=p['clock']['origin']+tick*.0002
        for path in range(2):
            sources=pre if path==0 else post;current=event[:2] if path==0 else event[2:];real=hard[:2] if path==0 else hard[2:]
            # Potential zero-gate deliveries retain their counterfactual VJP.
            for edge in sorted(range(4),key=lambda e:(sources[e],e)):
                bins[path].setdefault(tick+delays[path][edge],[]).append((edge,current[sources[edge]],bool(real[sources[edge]]),tick,None))
            arrivals=bins[path].pop(tick,[])
            active=[e for e,amplitude,yes,emitted,pending in arrivals if yes]
            if len(active)!=len(set(active)):collisions.append((tick,path,active))
            for edge,amplitude,yes,emitted,pending in arrivals:
                u=draw('rand',p['seed'],sequence,batch,domains[path],edge,emitted,pending,0)
                n=draw('randn',p['seed'],sequence,batch,domains[path],edge,emitted,pending,1) if path==0 else 0.
                if yes:
                    draws['rand'].append(u)
                    if path==0:draws['randn'].append(n)
                old_w=z[w[edge]];old_a=z[trace[edge]];new_a=old_a*np.exp(-(time-z[last[edge]])/.002)
                if path==0:
                    new_w=.95*old_w+.015*u
                    z[c[post[edge]]]+=amplitude*(new_w*(.6+.4*u)+.01*new_a+.002*n)
                    new_a+=.03*u
                    if yes:
                        z[count[edge]]+=1
                        if variable:z[delay_state[edge]]=(int(z[count[edge]])%3)*.0002
                else:new_w=old_w+.003*u;new_a*=.9
                z[w[edge]]+=amplitude*(new_w-old_w);z[trace[edge]]+=amplitude*(new_a-old_a)
                if yes:z[last[edge]]=time
        z[a+c]-=.4*event
    spikes=np.asarray(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes,dict(before=before,margins=margins,delays=delay_history if anchors is None else anchors['delays'],draws=draws,collisions=collisions)


@pytest.mark.parametrize('warm',[0,2])
@pytest.mark.parametrize('variable',[False,True])
def test_event_random_matches_compiled_brian(engine,warm,variable,tmp_path):
    net,groups,s,x,bundle=model(warm,variable,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    expected=oracle(bundle,x,variable=variable);draws=expected[3]['draws']
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    native=[];cursor=0
    lengths=[1]*len(x) if variable else [3,2,len(x)-5]
    for length in lengths:
        if not variable and cursor==3:trainer.update_delays({'event_s_pre':0.,'event_s_post':0.})
        result=trainer.step(x[None,cursor:cursor+length],[0],**({'initial':'carry'} if cursor else {'noise_sequence':9}))
        native.extend(result['spikes'][0]);cursor+=length
        path=tmp_path/'events.json';trainer.store(path);restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
    np.testing.assert_array_equal(native,expected[2])
    layout=bundle.provenance['neuron_state_layout'];slots=layout['event_a']['v']+layout['event_c']['v']
    slots+=sum(bundle.provenance['dynamic_state_layout']['event_s'].values(),[])
    tol=5e-5 if engine!='cpu' else 4e-12
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],expected[1][slots],rtol=tol,atol=tol*1e-2)
    net.run(0*b.ms,namespace={});calls={name:0 for name in draws};device=b.get_device()
    device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    def sampler(name):
        def refill(n):
            assert n==20000 and calls[name]==0;calls[name]+=1
            values=np.zeros(n);values[:len(draws[name])]=draws[name];return values
        return refill
    with patch('numpy.random.rand',sampler('rand')),patch('numpy.random.randn',sampler('randn')):
        if variable:
            for _ in x:net.run(.2*b.ms,namespace={})
        else:
            net.run(.6*b.ms,namespace={});s.pre.delay=0*b.ms;s.post.delay=0*b.ms;net.run((len(x)-3)*.2*b.ms,namespace={})
    for name in draws:assert calls[name]==1 and getattr(device,name+'_buffer_index')[0]==len(draws[name])
    device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    assert s.pre.codeobj.compiled_code['run'] is not None and s.post.codeobj.compiled_code['run'] is not None
    for group in groups:np.testing.assert_allclose(expected[1][layout[group.name]['v']],group.v[:],rtol=4e-12,atol=4e-14)
    for name,indices in bundle.provenance['dynamic_state_layout']['event_s'].items():
        np.testing.assert_allclose(expected[1][indices],s.variables[name].get_value(),rtol=4e-12,atol=4e-14)
    if not variable and not warm:assert expected[3]['collisions']


@pytest.mark.parametrize('warm',[0,2])
@pytest.mark.parametrize('window',[None,3])
def test_event_random_independent_derivatives(engine,warm,window):
    _,_,_,x,bundle=model(warm,True,backend=engine,tbptt_window=window)
    actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x[None],[0],noise_sequence=9)
    loss,z,spikes,anchors=oracle(bundle,x,variable=True,latch_each=False)
    tol=3e-3 if engine!='cpu' else 8e-7;eps=1e-6
    np.testing.assert_allclose(actual['loss'],loss,rtol=tol);np.testing.assert_array_equal(actual['spikes'][0],spikes)
    for bank,row in enumerate(bundle.weights):
        for k in range(len(row)):
            a=copy.deepcopy(bundle.weights);c=copy.deepcopy(bundle.weights);a[bank][k]+=eps;c[bank][k]-=eps
            fd=(oracle(bundle,x,weights=a,anchors=anchors,variable=True,latch_each=False)[0]-oracle(bundle,x,weights=c,anchors=anchors,variable=True,latch_each=False)[0])/(2*eps)
            np.testing.assert_allclose(actual['gradients'][bank][k],fd,rtol=tol,atol=tol*1e-2,err_msg=f'weight {bank}/{k}')
    initial=np.asarray(bundle.initial_state,float);actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x[None],[0],initial=initial[None],noise_sequence=9)
    anchors=oracle(bundle,x,initial=initial,variable=True,latch_each=False)[3]
    slots=sum((g['v'] for g in bundle.provenance['neuron_state_layout'].values()),[])
    states=bundle.provenance['dynamic_state_layout']['event_s'];slots+=states['w']+states['a']
    for k in slots:
        a=initial.copy();c=initial.copy();a[k]+=eps;c[k]-=eps
        fd=(oracle(bundle,x,initial=a,anchors=anchors,variable=True,latch_each=False)[0]-oracle(bundle,x,initial=c,anchors=anchors,variable=True,latch_each=False)[0])/(2*eps)
        np.testing.assert_allclose(actual['initial_state_gradients'][0][k],fd,rtol=tol,atol=tol*1e-2,err_msg=f'initial {k}')


@pytest.mark.parametrize('ranks',[2,8])
def test_event_random_mpi_rebuild_restore(engine,ranks,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI opt-in')
    _,_,_,x,bundle=model(2,backend=engine,mpi_ranks=ranks)
    actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x[None],[0],noise_sequence=9)
    cpu=copy.deepcopy(bundle.plan);cpu.update(backend='cpu',mpi_ranks=None)
    reference=NativeLIFTrainer(cpu,weights=bundle.weights,runner=RUNNER).gradients(x[None],[0],noise_sequence=9)
    for key in ('loss','spikes','initial_state_gradients','final_state'):
        np.testing.assert_allclose(actual[key],reference[key],rtol=2e-3 if engine!='cpu' else 3e-12,atol=2e-6 if engine!='cpu' else 3e-13)
    for row,expected in zip(actual['gradients'],reference['gradients']):
        np.testing.assert_allclose(row,expected,rtol=2e-3 if engine!='cpu' else 3e-12,atol=2e-6 if engine!='cpu' else 3e-13)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);spikes=[]
    for start,stop in ((0,3),(3,5),(5,len(x))):
        if start==3:trainer.update_delays({'event_s_pre':0.,'event_s_post':0.})
        if start==5:
            # Even a redundant update rebuilds pending generations; their draws
            # must retain the old emission identity.
            trainer.update_delays({'event_s_pre':0.,'event_s_post':0.})
        result=trainer.step(x[None,start:stop],[0],**({'initial':'carry'} if start else {'noise_sequence':9}))
        spikes.extend(result['spikes'][0]);path=tmp_path/'events.json';trainer.store(path)
        restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
    expected=oracle(bundle,x);np.testing.assert_array_equal(spikes,expected[2])
    slots=sum((g['v'] for g in bundle.provenance['neuron_state_layout'].values()),[])
    slots+=sum(bundle.provenance['dynamic_state_layout']['event_s'].values(),[])
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],expected[1][slots],rtol=5e-5 if engine!='cpu' else 4e-12,atol=5e-7 if engine!='cpu' else 4e-14)
    assert result['noise_sequence']==9 and result['final_tick']==len(x)
    if engine!='cpu':assert actual['gpu_dispatches']>0 and result['gpu_dispatches']>0


@pytest.mark.parametrize('damage',['route_delay','pending_zero','pending_delay','duplicate_id','initial_multiple','live_multiple','different_program'])
def test_event_random_address_validation(damage):
    _,_,_,x,bundle=model(2)
    spec=bundle.plan['dynamic'];path=next(p for p in spec['delay_layout']['paths'] if p['name']=='event_s_pre')
    pending=path['pending'][0];action=spec['actions'][pending['event']];initial=None
    message='event'
    if damage=='route_delay':
        spec['actions'][path['edges'][0]['event']]['event_noise']['delay']+=1;message='emission noise delay'
    elif damage=='pending_zero':action['event_noise']['pending']=0
    elif damage=='pending_delay':action['event_noise']['delay']=1
    elif damage=='duplicate_id':
        same=next(p for p in path['pending'][1:] if p['edge']==pending['edge'])
        spec['actions'][same['event']]['event_noise']['pending']=action['event_noise']['pending'];message='identity'
    elif damage in ('initial_multiple','live_multiple'):
        pending=next(p for p in path['pending'] if len(p['states'])>1)
        row=spec['initial'] if damage=='initial_multiple' else list(spec['initial'])
        row[pending['states'][0]]=row[pending['states'][-1]]=1.
        if damage=='live_multiple':initial=[row]
        message='identity'
    else:action['noise_entity']+=1;message='differs from its pathway'
    with pytest.raises(ValueError,match=message):
        NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(x[None],[0],initial=initial)


def test_event_random_mask_migration(engine,tmp_path):
    _,_,_,x,bundle=model(2,backend=engine)
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='event_s' and e['variables']==['w'])
    cpu=copy.deepcopy(bundle.plan);cpu['backend']='cpu'
    trainers=[NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER) for p in (bundle.plan,cpu)]
    for start,stop in ((0,2),(2,5),(5,len(x))):
        results=[t.step(x[None,start:stop],[0],**({'initial':'carry'} if start else {'noise_sequence':9})) for t in trainers]
        for key in ('loss','spikes','initial_state_gradients','final_state'):
            np.testing.assert_allclose(results[0][key],results[1][key],rtol=2e-3,atol=2e-6)
        for a,c in zip(results[0]['gradients'],results[1]['gradients']):np.testing.assert_allclose(a,c,rtol=2e-3,atol=2e-6)
        if stop==len(x):break
        for index,t in enumerate(trainers):
            masks=copy.deepcopy(t.plan['masks']);masks[bank]=[0. if stop==2 else 1.]*4
            clock=(t.clock_tick,t.noise_sequence,t.next_noise_sequence,t.state['step'])
            t.update_mask(masks,growth_weight=.12)
            assert clock==(t.clock_tick,t.noise_sequence,t.next_noise_sequence,t.state['step'])
            for path in t.plan['dynamic']['delay_layout']['paths']:
                if not path['name'].startswith('event_s_'):continue
                for entry in path['pending']+path['edges']:
                    assert all(t.neuron_state[0][k]==0 for k in entry['states'])
            # Rebuild the graph after mask migration; the stochastic event
            # metadata must survive both operations and checkpoint restoration.
            t.update_delays({'event_s_pre':.0004,'event_s_post':0.})
            path=tmp_path/f'mask-{index}.json';t.store(path)
            restored=NativeLIFTrainer(t.plan,runner=RUNNER);restored.restore(path);trainers[index]=restored


def test_event_random_batch_addresses(engine):
    _,_,_,x,bundle=model(2,backend=engine)
    bundle.plan['seed']=2**64-1;sequence=2**64-2
    actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(np.stack([x,x]),[0,0],noise_sequence=sequence)
    for batch in range(2):
        expected=oracle(bundle,x,change=False,sequence=sequence,batch=batch)
        np.testing.assert_array_equal(actual['spikes'][batch],expected[2])
        slots=sum((g['v'] for g in bundle.provenance['neuron_state_layout'].values()),[])
        slots+=sum(bundle.provenance['dynamic_state_layout']['event_s'].values(),[])
        np.testing.assert_allclose(np.asarray(actual['final_state'])[batch,slots],expected[1][slots],rtol=5e-5 if engine!='cpu' else 4e-12,atol=5e-7 if engine!='cpu' else 4e-14)
    assert not np.array_equal(actual['final_state'][0],actual['final_state'][1])
