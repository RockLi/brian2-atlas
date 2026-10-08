"""Persistent device observations, detached carried scores and restore admission."""
import copy
import hashlib
import json
import math
import subprocess
import numpy as np
import pytest
from brian2_rust.training import NativeLIFTrainer,canonical_bytes
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_shared import model
from test_training_poisson_zero_vjp import mpi


def snapshot(t):
    return copy.deepcopy((t.plan,t.state,t.neuron_state,t.poisson_state,t.elapsed_ticks,t.clock_tick,t.clock_state,t.noise_sequence,t.next_noise_sequence))


def pending(engine,ranks=None,window=None,zero=False):
    mpi(ranks);p,w=model(window,ranks);p.update(backend=engine,trainable=[False])
    for a in p['dynamic']['actions'][:2]:a.update(event_noise={'delay':0,'pending':73},trigger={'external':True,'index':0})
    if zero:w[0][0]=0.
    return p,w


def fixed(p,w,initial,counts,labels,length,anchors=None):
    z=np.array(initial,float).copy();before=[];margins=[];spikes=[]
    for tick in range(length):
        if anchors and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[:,2]=z[:,3]=counts
        z[:,0]+=w[0][1]*counts;z[:,4]=.8*z[:,4]+.2*w[0][2]
        z[:,1]=1.-w[0][1]*counts;z[:,4]=.8*z[:,4]+.2*w[0][2]
        margin=z[:,:2]-.8;hard=(margin>0).astype(float)
        if anchors:
            old=anchors['margins'][tick];hard=(old>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());spikes.append(hard)
    spikes=np.stack(spikes,axis=1);logits=spikes.mean(1)*p['logit_scale'];maximum=logits.max(1)
    loss=(maximum+np.log(np.exp(logits-maximum[:,None]).sum(1))-logits[np.arange(len(z)),labels]).mean()
    return loss,z,dict(before=before,margins=margins)


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('window',[None,1,2])
@pytest.mark.parametrize('zero',[False,True])
def test_pending_carry_detaches_saved_rate_and_keeps_surrogate_vjp(engine,ranks,window,zero):
    p,w=pending(engine,ranks,window,zero);t=NativeLIFTrainer(p,weights=w,runner=RUNNER);x=np.ones((2,3,1));labels=[0,1]
    t.step(x[:,:1],labels,noise_sequence=9);saved=copy.deepcopy(t.poisson_state);counts=np.array([e['count'] for e in saved['entries']])
    # An invalid current rate is irrelevant to a previously observed draw.
    t.state['weights'][0][0]=-3.;t.neuron_state[0][4]=-1.
    initial=np.array(t.neuron_state);before=snapshot(t);r=t.gradients(x,labels,initial='carry')
    assert snapshot(t)==before and t.last_result==r and r['poisson_state']==saved
    assert r['gradients'][0][0]==r['gradients'][0][2]==0.
    assert r['initial_state_gradients'][0][4]==r['initial_state_gradients'][1][4]==0.
    loss,live,anchors=fixed(p,t.state['weights'],initial,counts,labels,3)
    assert r['loss']==pytest.approx(loss,abs=3e-6);np.testing.assert_allclose(r['final_state'],live,rtol=5e-6,atol=3e-6)
    for k in (0,1,2):
        plus=copy.deepcopy(t.state['weights']);minus=copy.deepcopy(plus);plus[0][k]+=1e-6;minus[0][k]-=1e-6
        fd=(fixed(p,plus,initial,counts,labels,3,anchors)[0]-fixed(p,minus,initial,counts,labels,3,anchors)[0])/2e-6
        assert r['gradients'][0][k]==pytest.approx(fd,rel=5e-4,abs=8e-6)


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_delayed_emission_split_restore_matches_whole_device(engine,ranks,window,tmp_path):
    mpi(ranks);p,w=model(window,ranks);p.update(backend=engine,trainable=[False])
    for delay,a in enumerate(p['dynamic']['actions'][:2]):a.update(event_noise={'delay':delay},trigger={'external':True,'index':0})
    x=np.ones((2,5,1));labels=[0,1]
    whole=NativeLIFTrainer(p,weights=w,runner=RUNNER).step(x,labels,noise_sequence=9)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(x[:,:2],labels,noise_sequence=9);path=tmp_path/'split';t.store(path)
    q=NativeLIFTrainer(p,weights=w,runner=RUNNER);q.restore(path);tail=q.step(x[:,2:],labels,initial='carry')
    assert tail['poisson_state']==whole['poisson_state']
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=4e-6,atol=3e-6)
    np.testing.assert_array_equal(tail['spikes'],np.asarray(whole['spikes'])[:,2:])


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('bad',['count','duplicate','rate','missing','schema','profile','batch','sequence','kind','pending'])
def test_restore_invalid_cache_is_atomic_then_valid_retry(engine,ranks,bad,tmp_path):
    p,w=pending(engine,ranks);t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(np.ones((1,1,1)),[0],noise_sequence=9)
    good=tmp_path/'good';bad_path=tmp_path/'bad';t.store(good);envelope=json.loads(good.read_text());payload=json.loads(envelope['payload']);cache=payload['poisson_state']
    if bad=='count':cache['entries'][0]['count']+=1
    elif bad=='duplicate':cache['entries']*=2
    elif bad=='rate':cache['entries'][0]['rate']=-1.
    elif bad=='missing':payload.pop('poisson_state')
    elif bad=='schema':cache['schema']='unknown'
    elif bad=='profile':cache['numeric_profile']='unknown'
    elif bad=='kind':cache['entries'][0]['identity']['site']['kind']=3
    elif bad=='pending':cache['entries'][0]['identity']['instant']+=1
    else:cache[bad]+=1
    raw=canonical_bytes(payload);envelope.update(payload=raw.decode(),sha256=hashlib.sha256(raw).hexdigest());bad_path.write_text(json.dumps(envelope))
    q=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=snapshot(q)
    with pytest.raises(ValueError,match='Poisson|poisson'):q.restore(bad_path)
    assert snapshot(q)==before
    q.restore(good);assert q.poisson_state==t.poisson_state
    r=q.evaluate(np.ones((1,2,1)),[0],initial='carry');assert r['poisson_state']==t.poisson_state


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('rate',[1e9,2.146e9])
def test_device_large_count_cache_survives_restore(engine,ranks,rate,tmp_path):
    p,w=pending(engine,ranks);w[0]=[rate,1e-9,1.3];p['dynamic']['initial'][4]=1.
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(np.ones((1,1,1)),[0],noise_sequence=9)
    saved=copy.deepcopy(t.poisson_state);assert saved['entries'][0]['count']>2**24
    path=tmp_path/'large';t.store(path);q=NativeLIFTrainer(p,weights=w,runner=RUNNER);q.restore(path)
    r=q.gradients(np.ones((1,3,1)),[0],initial='carry');assert r['poisson_state']==saved
    assert r['final_state'][0][2]==r['final_state'][0][3]==saved['entries'][0]['count'] and r['gradients'][0][0]==0.


@pytest.mark.parametrize('ranks',[None,2])
def test_async_idle_boundary_cache_and_fresh_sequence(engine,ranks,tmp_path):
    p,w=pending(engine,ranks);p['dynamic']['clocks']={'start':0.,'dts':[.001,.0005],'epsilon':1e-4,'order':[0,1]}
    for a in p['dynamic']['actions'][:2]:a.update(clock=1,trigger={'state':True,'external':False,'index':10},detach_trigger=True);a['reads'].append(10)
    d=p['dynamic'];d['spike_buffers']=[7,8,9];d['binary_states']=[7,8,9,10]
    d['initial']+=[0.,0.,0.,1.];d['initial_parameters']+=[None]*4;d['detached']+=[False]*3+[True]
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);x=np.ones((2,3,1));t.step(x,[0,1],noise_sequence=9);saved=copy.deepcopy(t.poisson_state)
    path=tmp_path/'idle';t.store(path);q=NativeLIFTrainer(p,weights=w,runner=RUNNER);q.restore(path)
    q.step(x,[0,1],initial='carry');assert q.poisson_state==saved
    q.step(x,[0,1]);assert q.poisson_state['sequence']==10 and len(q.poisson_state['entries'])==2


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('mode',['missing','version','symbol','work','error'])
def test_checkpoint_capability_and_bounded_work_native_controls(backend,mode,tmp_path,monkeypatch):
    source=tmp_path/'stub.c';library=tmp_path/'stub.dylib';calls=tmp_path/'calls'
    code='#include <stdint.h>\n#include <stdio.h>\n'
    if mode!='missing':code+='uint64_t b2_train_poisson_checkpoint_v1(void){return '+('0' if mode=='version' else '1')+';}\n'
    symbol='b2_train_'+backend+'_poisson_validate_v1'
    if mode in ('work','error'):
        code+='int '+symbol+'(uint64_t n,const uint64_t* k,const float* r,const int32_t* c,uint32_t* w,uint32_t* e,char* msg,uint64_t cap){FILE* f=fopen('+json.dumps(str(calls))+',"a");fprintf(f,"%llu\\n",(unsigned long long)n);fclose(f);for(uint64_t j=0;j<n;j++){w[j]='+('100000' if mode=='work' else '0')+';e[j]='+('0' if mode=='work' else '5')+';}return 0;}\n'
    source.write_text(code);subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    monkeypatch.setenv('B2_TRAIN_METAL_LIB' if backend=='metal' else 'B2_TRAIN_CUDA_LIB',str(library))
    p,w=model();t=NativeLIFTrainer(p,weights=w,runner=RUNNER);p['backend']=backend
    entries=[dict(identity=dict(site=dict(domain=71,entity=0,stream=0,kind=0,pending=0),batch=0,instant=j),count=0,rate=0.) for j in range(101 if mode=='work' else 1)]
    cache=dict(schema='b2-poisson-draw-state-v2',numeric_profile='native-'+backend+'-poisson-f32-v1',seed=p['seed'],sequence=9,batch=1,entries=entries)
    with pytest.raises(ValueError,match='GPU Poisson checkpoint'):
        t._run_request(dict(plan=p,state=t.state,initial=[p['dynamic']['initial']],operation='validate_poisson_state',inputs=[],labels=[],noise_sequence=9,poisson_state=cache))
    if mode in ('work','error'):assert [int(s) for s in calls.read_text().splitlines()]==([100,1] if mode=='work' else [1])
    assert t.poisson_state is None and t.clock_tick==0


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('bad',['cpu-profile','other-backend','missing-profile','non-f32','nan','range'])
def test_device_profile_and_float32_rate_reject_before_library(backend,bad):
    p,w=model();t=NativeLIFTrainer(p,weights=w,runner=RUNNER);p['backend']=backend
    cache=dict(schema='b2-poisson-draw-state-v2',numeric_profile='native-'+backend+'-poisson-f32-v1',seed=p['seed'],sequence=9,batch=1,
        entries=[dict(identity=dict(site=dict(domain=71,entity=0,stream=0,kind=0,pending=0),batch=0,instant=0),count=0,rate=0.)])
    if bad=='cpu-profile':cache.update(schema='b2-poisson-draw-state-v1');cache.pop('numeric_profile')
    elif bad=='other-backend':cache['numeric_profile']='native-'+('cuda' if backend=='metal' else 'metal')+'-poisson-f32-v1'
    elif bad=='missing-profile':cache.pop('numeric_profile')
    elif bad=='non-f32':cache['entries'][0]['rate']=.1
    elif bad=='nan':cache['entries'][0]['rate']=float('nan')
    else:cache['entries'][0]['rate']=2147483648.
    with pytest.raises(ValueError,match='Poisson|poisson|invalid JSON'):
        t._run_request(dict(plan=p,state=t.state,initial=[p['dynamic']['initial']],operation='validate_poisson_state',inputs=[],labels=[],noise_sequence=9,poisson_state=cache))


@pytest.mark.parametrize('ranks',[None,2])
def test_uneven_orphan_cache_is_retained_with_exact_lane_identity(engine,ranks):
    p,w=pending(engine,ranks);t=NativeLIFTrainer(p,weights=w,runner=RUNNER);x=np.ones((2,1,1));x[1]=0.
    t.step(x,[0,1],noise_sequence=9);cache=copy.deepcopy(t.poisson_state);assert len(cache['entries'])==1
    for j in range(103):
        cache['entries'].append(dict(identity=dict(site=dict(domain=91,entity=j,stream=0,kind=0,pending=0),batch=0,instant=0),count=0,rate=0.))
    r=t.gradients(x,[0,1],initial=t.neuron_state,start_tick=t.clock_tick,noise_sequence=9,poisson_state=cache)
    assert r['poisson_state']['entries']==sorted(cache['entries'],key=lambda e:(e['identity']['site']['domain'],e['identity']['site']['entity'],e['identity']['site']['stream'],e['identity']['site']['kind'],e['identity']['site']['pending'],e['identity']['batch'],e['identity']['instant']))
    assert {e['identity']['batch'] for e in r['poisson_state']['entries']}=={0}


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('rate',[0.,1.3])
def test_generated_brian_delay_migration_keeps_draw_records(engine,ranks,rate,tmp_path):
    import brian2 as b
    from brian2_rust import lower_brian_dynamic_training
    from test_training_delays import model as delay_model
    from test_training_delay_update import change
    mpi(ranks);net,inp,layers,static,syn,x,_=delay_model(order_sensitive=True)
    # Explicit floating literal preserves Brian's supported local type.
    syn.pre.code+='\nw+=.01*poisson('+str(rate)+')'
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers)
    bundle.plan.update(backend=engine,mpi_ranks=ranks,trainable=[False]*len(bundle.weights))
    p=copy.deepcopy(bundle.plan);p.update(backend='cpu',mpi_ranks=None)
    reference=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    cursor=0
    for phase,length in enumerate([3,2,3]):
        if phase:
            updates=change(phase-1,static,syn);before=copy.deepcopy(actual.poisson_state)
            actual.update_delays(updates);reference.update_delays(updates)
            assert actual.poisson_state==before
            for obj in (static,syn):
                for path in obj._pathways:path.delay=np.asarray(updates[path.name])*b.second
            path=tmp_path/('phase-'+str(phase));actual.store(path)
            restored=NativeLIFTrainer(actual.plan,weights=bundle.weights,runner=RUNNER);restored.restore(path)
            assert restored.poisson_state==actual.poisson_state;actual=restored
        part=x[None,cursor:cursor+length]
        kw={'initial':'carry'} if phase else {'noise_sequence':9}
        r=actual.step(part,[0],**kw);expected=reference.step(part,[0],**kw)
        np.testing.assert_allclose(r['final_state'],expected['final_state'],rtol=4e-5,atol=5e-6)
        np.testing.assert_array_equal(r['spikes'],expected['spikes'])
        assert r['poisson_state']['entries']
        if rate==0.:
            # Actual Cython runs the same lowered model and real queues; its
            # zero-rate Poisson counts are deterministic without RNG injection.
            net.run(length*.2*b.ms,namespace={})
            state=np.asarray(r['final_state'])[0]
            for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
                np.testing.assert_allclose(state[slots],syn.variables[name].get_value(),rtol=4e-5,atol=5e-6)
            np.testing.assert_allclose(r['final_membrane'][0],np.r_[layers[0].v[:],layers[1].v[:]],rtol=4e-5,atol=5e-6)
            assert syn.pre.codeobj.compiled_code['run'] is not None
        cursor+=length


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('version',[None,0])
def test_persistent_layout_requires_capability_before_device(backend,version,tmp_path,monkeypatch):
    source=tmp_path/'old.c';library=tmp_path/'old.dylib'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\nuint64_t b2_train_poisson_shared_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_poisson_persistent_v1(void){return 0;}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    monkeypatch.setenv('B2_TRAIN_METAL_LIB' if backend=='metal' else 'B2_TRAIN_CUDA_LIB',str(library))
    p,w=model();t=NativeLIFTrainer(p,weights=w,runner=RUNNER);p['backend']=backend
    with pytest.raises(ValueError,match='GPU persistent Poisson cache capability'):
        t._run_request(dict(plan=p,state=t.state,operation='evaluate',inputs=[[[0.]]],labels=[0],initial=None))
