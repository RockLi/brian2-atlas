"""Device invocation cache: one first-origin score and all-alias replay."""
import copy
import math
import numpy as np
import pytest
from brian2_rust.training import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_poisson_shared import model,oracle,delayed_oracle
from test_training_poisson_ssa import draw
from test_training_poisson_zero import hard_loss
from test_training_poisson_zero_vjp import mpi


def run(p,w,engine,x,labels,**kw):
    p['backend']=engine;t=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    r=t.gradients(x,labels,**kw)
    if engine!='cpu':assert r['gpu_dispatches']>0 and r['backend']==engine
    return t,r


@pytest.mark.parametrize('window',[None,1,2])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_positive_shared_device_draw_has_one_origin_score(engine,window,batch,ranks):
    mpi(ranks);p,w=model(window,ranks);labels=np.arange(batch)%2;initial=np.array([p['dynamic']['initial']]*batch);initial[:,4]+=np.arange(batch)*.1
    _,r=run(p,w,engine,np.zeros((batch,4,1)),labels,initial=initial,noise_sequence=9)
    loss,live,spikes,anchors=oracle(p,w,initial,labels,4)
    np.testing.assert_array_equal(r['spikes'],spikes);np.testing.assert_allclose(r['final_state'],live,rtol=3e-6,atol=2e-6)
    assert r['loss']==pytest.approx(loss,rel=3e-6,abs=2e-6)
    for j in range(3):
        plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[0][j]+=1e-6;minus[0][j]-=1e-6
        fd=(oracle(p,plus,initial,labels,4,anchors)[0]-oracle(p,minus,initial,labels,4,anchors)[0])/2e-6
        assert r['gradients'][0][j]==pytest.approx(fd,rel=5e-4,abs=5e-6)
    for b in range(batch):
        for j in (0,1,4):
            plus=initial.copy();minus=initial.copy();plus[b,j]+=1e-6;minus[b,j]-=1e-6
            fd=(oracle(p,w,plus,labels,4,anchors)[0]-oracle(p,w,minus,labels,4,anchors)[0])/2e-6
            assert r['initial_state_gradients'][b][j]==pytest.approx(fd,rel=5e-4,abs=5e-6)


@pytest.mark.parametrize('window',[None,1,2])
@pytest.mark.parametrize('length',[1,4])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_zero_device_draw_forces_all_aliases_once(engine,window,length,ranks):
    mpi(ranks);p,w=model(window,ranks);w[0][0]=0.
    _,r=run(p,w,engine,np.zeros((1,length,1)),[0])
    rate=p['dynamic']['initial'][4];expected=0.
    for tick in range(length):
        expected+=rate*(hard_loss(p['logit_scale']*(1-tick)/length)-hard_loss(-p['logit_scale']))
        rate=.64*rate+.36*w[0][2]
    assert r['gradients'][0][0]==pytest.approx(expected,rel=5e-5,abs=5e-6)
    assert r['gradients'][0][1]==0.
    np.testing.assert_array_equal(np.asarray(r['final_state'])[:,2:4],0.)


@pytest.mark.parametrize('ranks',[None,2])
def test_device_consumer_skips_overwritten_invalid_rate(engine,ranks):
    mpi(ranks);p,w=model(ranks=ranks,first='k=draw(scale*r)\nv=amp*k\nr=-1.')
    _,r=run(p,w,engine,np.zeros((1,1,1)),[0],noise_sequence=9)
    count=draw(w[0][0]*p['dynamic']['initial'][4],p['seed'],9,0,0,0)
    assert r['final_state'][0][2]==r['final_state'][0][3]==count


@pytest.mark.parametrize('ranks',[None,2])
def test_device_first_lazy_visited_call_owns_score(engine,ranks):
    mpi(ranks);p,w=model(ranks=ranks,first='v=amp*draw(scale*r) if v>1 else v\nr=.8*r+.2*offset')
    _,r=run(p,w,engine,np.zeros((1,1,1)),[0],noise_sequence=9)
    rate=w[0][0]*(.8*p['dynamic']['initial'][4]+.2*w[0][2]);count=draw(rate,p['seed'],9,0,0,0)
    assert r['final_state'][0][3]==count
    expected=r['loss']*(count/rate-1)*(.8*p['dynamic']['initial'][4]+.2*w[0][2])
    assert r['gradients'][0][0]==pytest.approx(expected,rel=5e-5,abs=5e-6)


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('zero',[False,True])
def test_pending_identity_reuses_count_across_device_visits(engine,ranks,zero):
    mpi(ranks);p,w=model(ranks=ranks)
    for a in p['dynamic']['actions'][:2]:a.update(event_noise={'delay':0,'pending':73},trigger={'external':True,'index':0})
    if zero:w[0][0]=0.
    _,r=run(p,w,engine,np.ones((1,4,1)),[0],noise_sequence=9)
    rate=p['dynamic']['initial'][4]
    if zero:expected=rate*(hard_loss(p['logit_scale'])-hard_loss(-p['logit_scale']))
    else:
        count=draw(w[0][0]*rate,p['seed'],9,0,0,0,(2,73))
        assert r['final_state'][0][2]==r['final_state'][0][3]==count
        expected=r['loss']*(count/(w[0][0]*rate)-1)*rate
    assert r['gradients'][0][0]==pytest.approx(expected,rel=5e-5,abs=5e-6)


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('window',[None,2])
def test_delayed_device_aliases_share_emission_rate_origin(engine,ranks,window):
    mpi(ranks);p,w=model(window,ranks)
    for delay,a in enumerate(p['dynamic']['actions'][:2]):a.update(event_noise={'delay':delay},trigger={'external':True,'index':0})
    initial=np.array([p['dynamic']['initial']]*2);labels=[0,1];x=np.ones((2,4,1))
    _,r=run(p,w,engine,x,labels,initial=initial,noise_sequence=9)
    loss,live,spikes,anchors=delayed_oracle(p,w,initial,labels,4)
    np.testing.assert_allclose(r['final_state'],live,atol=2e-6,rtol=3e-6);np.testing.assert_array_equal(r['spikes'],spikes)
    assert r['loss']==pytest.approx(loss,rel=3e-6,abs=2e-6)
    for j in range(3):
        plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[0][j]+=1e-6;minus[0][j]-=1e-6
        fd=(delayed_oracle(p,plus,initial,labels,4,anchors)[0]-delayed_oracle(p,minus,initial,labels,4,anchors)[0])/2e-6
        assert r['gradients'][0][j]==pytest.approx(fd,rel=5e-4,abs=5e-6)


@pytest.mark.parametrize('ranks',[None,2])
def test_async_device_idle_visits_reuse_shared_origin(engine,ranks):
    mpi(ranks);p,w=model(ranks=ranks)
    p['dynamic']['clocks']={'start':0.,'dts':[.001,.0005],'epsilon':1e-4,'order':[0,1]}
    for a in p['dynamic']['actions'][:2]:a['clock']=1
    _,r=run(p,w,engine,np.zeros((1,3,1)),[0],noise_sequence=9)
    _,expected,_,_=oracle(p,w,[p['dynamic']['initial']],[0],6)
    np.testing.assert_allclose(r['final_state'],expected,rtol=3e-6,atol=2e-6)


@pytest.mark.parametrize('ranks',[None,2])
def test_shared_device_cache_budget_and_failure_do_not_commit(engine,ranks):
    mpi(ranks);p,w=model(ranks=ranks)
    _,r=run(p,w,engine,np.zeros((1,3,1)),[0])
    p['max_tape_bytes']=r['tape_bytes']-1;p['backend']=engine;t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='budget'):t.step(np.zeros((1,3,1)),[0])
    assert t.state==before and t.neuron_state is None and t.clock_tick==0 and t.next_noise_sequence==0


@pytest.mark.parametrize('ranks',[None,2])
def test_device_cache_preserves_large_integer_count_payload(engine,ranks):
    mpi(ranks);p,w=model(ranks=ranks,first='k=draw(scale*r)\nv=amp*k\nr=-1.')
    w[0]=[1e9,1e-9,1.3]
    _,r=run(p,w,engine,np.zeros((1,1,1)),[0],noise_sequence=9)
    a,b=r['final_state'][0][2:4];assert a==b and a>2**24 and a==math.floor(a)


@pytest.mark.parametrize('ranks',[None,2])
def test_shared_indirect_device_rate_scores_original_physical_cell(engine,ranks):
    mpi(ranks);p,w=model(ranks=ranks);d=p['dynamic']
    d['initial'].append(1.);d['initial_parameters'].append(None);d['detached'].append(True);d['integer_states'].append(7)
    for a in d['actions'][:2]:a['indirect']={'reads':{'2':{'index':7,'tables':[[4,5]]}},'writes':{}}
    _,r=run(p,w,engine,np.zeros((1,1,1)),[0],noise_sequence=9)
    rate=w[0][0]*d['initial'][5];count=draw(rate,p['seed'],9,0,0,0)
    assert r['final_state'][0][2]==r['final_state'][0][3]==count
    score=r['loss']*(count/rate-1)
    assert r['gradients'][0][0]==pytest.approx(score*d['initial'][5],rel=5e-5,abs=5e-6)
    assert r['initial_state_gradients'][0][5]==pytest.approx(score*w[0][0],rel=5e-5,abs=5e-6)
    assert r['initial_state_gradients'][0][4]==0.


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('detached',[True,False])
def test_shared_singular_original_rate_vjp_keeps_activity_contract(engine,ranks,detached):
    mpi(ranks);p,w=model(ranks=ranks,first='k=draw(scale+sqrt(r))\nv=v+amp*k\nr=-1.',second='k=draw(scale+sqrt(r))\nv=1.-amp*k\nr=-1.')
    w[0][0]=0.;p['dynamic']['initial'][4]=0.;p['dynamic']['detached'][4]=detached
    if detached:
        _,r=run(p,w,engine,np.zeros((1,1,1)),[0])
        expected=hard_loss(p['logit_scale'])-hard_loss(-p['logit_scale'])
        assert r['gradients'][0][0]==pytest.approx(expected,rel=5e-5,abs=5e-6)
    else:
        p['backend']=engine;t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(t.state)
        with pytest.raises(ValueError):t.step(np.zeros((1,1,1)),[0])
        assert t.state==before and t.neuron_state is None and t.clock_tick==0 and t.next_noise_sequence==0


@pytest.mark.parametrize('ranks',[None,2])
def test_nested_shared_poisson_replay_preserves_both_stream_identities(engine,ranks):
    from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
    from brian2_rust.training_equations import PoissonNoise
    mpi(ranks);p,w=model(ranks=ranks);w[0][0]=0.
    for index,voltage in enumerate(['v=v+amp*k','v=1.-amp*k']):
        tr=compile_dynamic_transform('k=draw(.5*scale+1.*inner(scale))\n'+voltage+'\nr=.8*r+.2*offset',
            states={'v':0,'k':1,'r':2},state_types={1:'integer'},parameters={'draw':PoissonNoise(0),'inner':PoissonNoise(1),'scale':(0,0),'amp':(0,1),'offset':(0,2)})
        p['dynamic']['program_sets'][index]=tr['programs'];p['dynamic']['actions'][index]=dynamic_action(tr,[index,2+index,4],owner=1+index,program_set=index,noise_domain=71,noise_entity=0,noise_streams=2)
    _,r=run(p,w,engine,np.zeros((1,1,1)),[0],noise_sequence=9)
    baseline=hard_loss(-p['logit_scale']);inner_alternate=draw(1.,p['seed'],9,0,0,0)
    inner_loss=hard_loss(p['logit_scale'] if inner_alternate>=1 else -p['logit_scale'])
    expected=(inner_loss-baseline)+.5*(hard_loss(p['logit_scale'])-baseline)
    assert r['gradients'][0][0]==pytest.approx(expected,rel=5e-5,abs=5e-6)


@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('version',[None,0])
def test_shared_cache_capability_is_required_before_device_entry(backend,version,tmp_path,monkeypatch):
    import subprocess
    source=tmp_path/'old.c';library=tmp_path/'old.dylib'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_poisson_shared_v1(void){return 0;}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    monkeypatch.setenv('B2_TRAIN_METAL_LIB' if backend=='metal' else 'B2_TRAIN_CUDA_LIB',str(library))
    p,w=model();t=NativeLIFTrainer(p,weights=w,runner=RUNNER);p['backend']=backend
    with pytest.raises(ValueError,match='GPU shared Poisson cache capability'):
        t._run_request(dict(plan=p,state=t.state,operation='gradients',inputs=[[[0.]]],labels=[0],initial=None))


@pytest.mark.parametrize('ranks',[None,2])
def test_device_cache_carry_succeeds_without_readonly_commit(engine,ranks):
    mpi(ranks);p,w=model(ranks=ranks);p['trainable']=[False];p['backend']=engine
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(np.zeros((1,1,1)),[0],noise_sequence=9)
    before=copy.deepcopy((t.state,t.neuron_state,t.poisson_state,t.clock_tick,t.noise_sequence,t.next_noise_sequence))
    r=t.gradients(np.zeros((1,1,1)),[0],initial='carry');assert r['final_tick']==2
    assert len(r['poisson_state']['entries'])==2
    assert (t.state,t.neuron_state,t.poisson_state,t.clock_tick,t.noise_sequence,t.next_noise_sequence)==before
    assert t.last_result==r
    if engine!='cpu':assert r['poisson_state']['numeric_profile']=='native-'+engine+'-poisson-f32-v1' and r['gpu_dispatches']>0
