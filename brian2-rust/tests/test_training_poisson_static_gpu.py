"""Actual static vector device draws, all VJPs, weak alternates and cache replay."""
import copy,importlib,json,math,os,subprocess,hashlib
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_poisson_zero_vjp import mpi
from test_training_poisson_static import model,fixture,oracle,expression,draw,_static_frontend_cython
from test_training_static_timed_inputs import poisson_model,poisson_physical,_static_timed_poisson_cython
from brian2_rust.training_equations import PoissonNoise
from brian2_rust.training import canonical_bytes
from test_training_poisson_checkpoint_gpu import snapshot

@pytest.fixture(params=['metal','cuda'])
def engine(request):
    flag='B2_TEST_GPU' if request.param=='metal' else 'B2_TEST_CUDA_TRAIN'
    if os.environ.get(flag)!='1':pytest.skip('actual '+request.param+' hardware required')
    return request.param

def compare(result,expected):
    assert result['gpu_dispatches']>0 and result['backend'] in ('metal','cuda')
    np.testing.assert_array_equal(result['spikes'],expected['spikes'])
    np.testing.assert_allclose(result['final_state'],expected['final_state'],rtol=6e-6,atol=3e-6)
    assert result['loss']==pytest.approx(expected['loss'],rel=6e-6,abs=3e-6)
    for a,b in zip(result['gradients'],expected['gradients']):np.testing.assert_allclose(a,b,rtol=1e-3,atol=5e-6)
    np.testing.assert_allclose(result['initial_state_gradients'],expected['initial_state_gradients'],rtol=1e-3,atol=5e-6)

@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('detach',[False,True])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_positive_physical_all_parameter_and_state_vjps(engine,window,detach,batch,ranks):
    mpi(ranks);p,w=model(window,detach,ranks);p['backend']=engine;x,y,initial=fixture(batch)
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,y,initial=initial,noise_sequence=9)
    loss,final,spikes,anchors=oracle(p,w,x,y,initial)
    assert result['backend']==engine and result['gpu_dispatches']>0
    assert result['loss']==pytest.approx(loss,abs=3e-6)
    np.testing.assert_array_equal(result['spikes'],spikes);np.testing.assert_allclose(result['final_state'],final,rtol=6e-6,atol=3e-6)
    assert len(result['poisson_state']['entries'])==batch*4*3
    for bank,row in enumerate(w):
        for j in range(len(row)):
            hi=copy.deepcopy(w);lo=copy.deepcopy(w);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(oracle(p,hi,x,y,initial,anchors)[0]-oracle(p,lo,x,y,initial,anchors)[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=5e-6)
    for bi in range(batch):
        for j in range(6):
            hi=initial.copy();lo=initial.copy();hi[bi,j]+=1e-6;lo[bi,j]-=1e-6
            fd=(oracle(p,w,x,y,hi,anchors)[0]-oracle(p,w,x,y,lo,anchors)[0])/2e-6
            assert result['initial_state_gradients'][bi][j]==pytest.approx(fd,rel=1e-3,abs=5e-6)

@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_zero_rate_whole_sample_weak_vjp(engine,window,batch,ranks):
    mpi(ranks);p,w=model(window,ranks=ranks);w[3][0]=0.;x,y,initial=fixture(batch)
    baseline,final,spikes,anchors=oracle(p,w,x,y,initial);expected=0.
    for tick in range(4):
        for bi in range(batch):
            for j in range(3):expected+=anchors['before'][tick][bi,[1,4,5][j]]*(oracle(p,w,x,y,initial,forced=(tick,bi,j))[0]-baseline)
    cpu=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,y,initial=initial,noise_sequence=9)
    p['backend']=engine;result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,y,initial=initial,noise_sequence=9)
    compare(result,cpu);assert result['gradients'][3][0]==pytest.approx(expected,rel=1e-3,abs=5e-6)
    assert all(e['count']==0 and e['rate']==0 for e in result['poisson_state']['entries'])

@pytest.mark.parametrize('rate',['scale-scale','0.*scale','r-r','scale if r<0 else 0.','sin(scale)-sin(scale)'])
@pytest.mark.parametrize('ranks',[None,2,8])
def test_cancelled_rate_never_replays_singular_alternate(engine,rate,ranks):
    mpi(ranks);p,w=model(ranks=ranks,rate=rate,reset_draw=False);p['backend']=engine
    for layer in p['state_equations']:layer[0]=expression(f'1./(1.-draw({rate}))',{'draw':PoissonNoise(0),'scale':(3,0)})
    x,y,initial=fixture(1);r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x[:,:2],y,initial=initial)
    assert r['gradients'][3][0]==0. and all(e['count']==0 for e in r['poisson_state']['entries'])

@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('masked',[False,True])
def test_active_weak_failure_and_mask_atomicity(engine,ranks,masked):
    mpi(ranks);p,w=model(ranks=ranks,reset_draw=False,rate='scale');p['backend']=engine;w[3][0]=0.
    for layer in p['state_equations']:layer[0]=expression('1./(1.-draw(scale))',{'draw':PoissonNoise(0),'scale':(3,0)})
    if masked:p['masks'][3][0]=0.
    x,y,initial=fixture(1);t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(t.state)
    if masked:
        r=t.gradients(x[:,:1],y,initial=initial);assert r['gradients'][3][0]==0.
    else:
        with pytest.raises(ValueError,match='nonfinite|GPU'):t.step(x[:,:1],y,initial=initial)
        assert t.state==before and t.neuron_state is None and t.poisson_state is None and t.clock_tick==0

@pytest.mark.parametrize('ranks',[None,2,8])
def test_carry_checkpoint_rewind_and_tail_records(engine,ranks,tmp_path):
    mpi(ranks);p,w=model(ranks=ranks);p['backend']=engine;p['trainable']=[False]*len(w);x,y,initial=fixture(3)
    whole=NativeLIFTrainer(p,weights=w,runner=RUNNER).evaluate(x,y,initial=initial,noise_sequence=9)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);first=t.step(x[:,:2],y,initial=initial,noise_sequence=9)
    path=tmp_path/'static-gpu';t.store(path);q=NativeLIFTrainer(p,runner=RUNNER);q.restore(path)
    a=t.step(x[:,2:],y,initial='carry');assert q.step(x[:,2:],y,initial='carry')==a
    np.testing.assert_allclose(a['final_state'],whole['final_state'],rtol=1e-6,atol=1e-7);assert a['poisson_state']==whole['poisson_state']
    saved=copy.deepcopy(q.poisson_state);q.state['weights'][3][0]=-1.;q.clock_tick=0
    r=q.gradients(x[:,:1],y,initial='carry')
    assert r['poisson_state']==saved and r['gradients'][3][0]==0.
    assert len(saved['entries'])>len(x)*3 # More imported identities than current trajectory slots.
    q.clock_tick=4;before=copy.deepcopy((q.state,q.neuron_state,q.poisson_state))
    with pytest.raises(ValueError,match='nonfinite|GPU'):q.step(x[:,:1],y,initial='carry')
    assert (q.state,q.neuron_state,q.poisson_state)==before and q.clock_tick==4

@pytest.mark.parametrize('backend',['metal','cuda'])
@pytest.mark.parametrize('version',[None,0,2])
def test_static_poisson_versioned_capability(backend,version,tmp_path,monkeypatch):
    p,w=model();p['backend']=backend;x,y,initial=fixture(1)
    source=tmp_path/'old.c';library=tmp_path/'old.so'
    source.write_text('#include <stdint.h>\nuint64_t b2_train_math_v1(void){return 1;}\n'+
        ('' if version is None else 'uint64_t b2_train_static_poisson_v1(void){return '+str(version)+';}\n'))
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    monkeypatch.setattr(importlib.import_module('brian2_rust.training_'+backend),'build',lambda directory:library)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='GPU static Poisson capability'):t.gradients(x,y,initial=initial)
    assert t.state==before and t.neuron_state is None


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_reset_first_observation_uses_post_update_rate_context(engine, ranks):
    mpi(ranks); p, w = model(ranks=ranks); p['backend']=engine; x, y, initial = fixture(3)
    initial[:, [0, 2, 3]] = .7
    params = {'draw': PoissonNoise(0), 'scale': (3, 0)}
    for layer in p['state_equations']:
        layer[0] = expression('v+draw(scale*r) if v>2. else v', params)
    for layer in p['state_resets']:
        layer[0] = expression('v-.6+.05*draw(scale*r)', params)
        layer[1] = expression('r', params)
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x[:, :1], y, initial=initial, noise_sequence=9)
    entries = result['poisson_state']['entries']; assert len(entries) == 9
    # All output neurons spike; sample CE is log(2), and scale only enters a
    # detached count, so its VJP consists solely of the first-observation score.
    expected = sum(math.log(2.)/3*(e['count']/e['rate']-1)*e['rate']/w[3][0] for e in entries)
    assert result['gradients'][3][0] == pytest.approx(expected, abs=5e-6)
    for e in entries:
        id = e['identity']; j = id['site']['entity']; l = id['site']['domain']; sample = id['batch']
        source = [1, 4, 5][0 if l == 0 else 1+j]
        assert e['rate'] == pytest.approx(w[3][0]*(.8*initial[sample, source]+.2*w[3][2]))
        assert result['initial_state_gradients'][sample][source] == pytest.approx(
            math.log(2.)/3*(e['count']/e['rate']-1)*w[3][0]*.8, abs=5e-6)
    assert result['backend']==engine and result['gpu_dispatches']>0


@pytest.mark.parametrize('ranks', [None, 2, 8])
def test_refractory_clamps_draw_only_reached_outputs(engine, ranks):
    mpi(ranks); p, w = model(ranks=ranks, reset_draw=False); p['backend']=engine
    params = {'draw':PoissonNoise(1), 'offset':(3, 2)}
    for layer in p['state_equations']:
        layer[1] = expression('.8*r+.1*draw(offset)', params)
        layer.append([{'op':'state', 'index':2}])
    for layer in p['state_resets']: layer.append([{'op':'state', 'index':2}])
    p['noise_streams'] = [2, 2]; p['refractory'] = [{'steps':3, 'clamp':[0]}]*2
    initial = np.array([[.7, 1.1, 2., .2, .9, 1.2, 1.4, 0., 1.]])
    x = np.array([.25, -.5, 1.3, .125, .4])[None, :, None]
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, [0], initial=initial, noise_sequence=9)
    live = initial[0].copy(); entries = []; spikes = []
    vslots = [0, 3, 4]; rslots = [1, 5, 6]; cslots = [2, 7, 8]
    for tick in range(5):
        old = live.copy(); active = old[cslots] == 0.
        live[cslots] = np.maximum(old[cslots]-1, 0.)
        for j in range(3):
            l, neuron = int(j > 0), max(j-1, 0)
            if active[j]:
                rate = w[3][0]*old[rslots[j]]; count = draw(rate, p['seed'], 9, 0, l, neuron, tick)
                entries.append((l, neuron, 0, tick, count, rate)); live[vslots[j]] = .4*old[vslots[j]]+w[3][1]*count
            count = draw(w[3][2], p['seed'], 9, 0, l, neuron, tick, 1)
            entries.append((l, neuron, 1, tick, count, w[3][2])); live[rslots[j]] = .8*old[rslots[j]]+.1*count
        hard = (live[vslots] > .6) & active; spikes.append(hard.copy())
        if active[0] and not hard[0]: live[0] += w[0][0]*x[0, tick, 0]+w[2][0]*hard[1:].sum()
        for j in (1, 2):
            if active[j] and not hard[j]: live[vslots[j]] += w[1][j-1]*hard[0]
        for j in range(3):
            if hard[j]:
                live[vslots[j]] -= .6; live[rslots[j]] = .9*live[rslots[j]]+.1*w[3][2]; live[cslots[j]] = 2.
    np.testing.assert_array_equal(result['spikes'][0], spikes)
    np.testing.assert_allclose(result['final_state'][0], live, rtol=6e-6, atol=3e-6)
    actual = [(e['identity']['site']['domain'], e['identity']['site']['entity'], e['identity']['site']['stream'],
               e['identity']['instant'], e['count'], e['rate']) for e in result['poisson_state']['entries']]
    assert [e[:5] for e in sorted(actual)] == [e[:5] for e in sorted(entries)]
    np.testing.assert_allclose([e[5] for e in sorted(actual)],[e[5] for e in sorted(entries)],rtol=6e-6,atol=3e-6)
    np.testing.assert_array_equal(np.asarray(result['initial_state_gradients'])[:, cslots], 0.)
    assert result['backend']==engine and result['gpu_dispatches']>0


@pytest.mark.parametrize('ranks', [None, 2, 8])
@pytest.mark.parametrize('scale', [0., 1.2])
def test_nested_sites_and_duplicate_inner_consumer(engine, ranks, scale):
    mpi(ranks); p, w = model(ranks=ranks, reset_draw=False); p['backend']=engine; w[3][0] = scale
    params = {'inner':PoissonNoise(0), 'outer':PoissonNoise(1), 'scale':(3, 0)}
    for layer in p['state_equations']:
        layer[0] = expression('.4*v+.8*outer(.4+2.*inner(scale*r))+.05*inner(scale*r)', params)
    p['noise_streams'] = [2, 2]; x, y, initial = fixture(1); x = x[:, :3]
    def physical(force=None):
        live = initial[0].copy(); events = []; records = []
        for tick in range(3):
            old = live.copy(); inner = []; outer = []
            for j in range(3):
                l, entity = int(j>0), max(j-1, 0); rate = scale*old[[1, 4, 5][j]]
                a = 1 if force == (tick, j) else draw(rate, p['seed'], 9, 0, l, entity, tick)
                bcount = draw(.4+2*a, p['seed'], 9, 0, l, entity, tick, 1)
                inner.append(a); outer.append(bcount); records.append((tick, j, rate, a, bcount, old[[1, 4, 5][j]]))
            live[[0, 2, 3]] = .4*old[[0, 2, 3]]+.8*np.array(outer)+.05*np.array(inner)
            live[[1, 4, 5]] = .8*old[[1, 4, 5]]+.2*w[3][2]
            event = live[[0, 2, 3]] > .6; events.append(event.copy())
            live[0] += w[0][0]*x[0, tick, 0]+w[2][0]*event[1:].sum()
            live[2:4] += np.array(w[1])*event[0]
            live[[0, 2, 3]] -= .6*event
            live[[1, 4, 5]] += event*(-.1*live[[1, 4, 5]]+.1*w[3][2])
        logits = np.array(events)[:, 1:].mean(0)*p['logit_scale']; maximum = logits.max()
        loss = maximum+np.log(np.exp(logits-maximum).sum())-logits[0]
        return loss, live, events, records
    loss, final, events, records = physical()
    result = NativeLIFTrainer(p, weights=w, runner=RUNNER).gradients(x, y, initial=initial, noise_sequence=9)
    assert result['loss'] == pytest.approx(loss, abs=3e-6)
    np.testing.assert_allclose(result['final_state'][0], final, rtol=6e-6, atol=3e-6)
    np.testing.assert_array_equal(result['spikes'][0], events)
    assert len(result['poisson_state']['entries']) == 18
    if scale:
        expected = sum(loss*(a/rate-1)*source for tick, j, rate, a, bcount, source in records)
    else:
        expected = sum(source*(physical((tick, j))[0]-loss) for tick, j, rate, a, bcount, source in records)
    assert result['gradients'][3][0] == pytest.approx(expected, rel=1e-3, abs=5e-6)
    assert result['backend']==engine and result['gpu_dispatches']>0


@pytest.mark.parametrize('warm',[0,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_static_frontend_original_cython(engine,warm,ranks,tmp_path):
    _static_frontend_cython(warm,ranks,tmp_path,engine)

@pytest.mark.parametrize('dimensions',[1,2])
@pytest.mark.parametrize('warm',[0,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_static_timed_poisson_original_cython(engine,dimensions,warm,ranks,tmp_path):
    _static_timed_poisson_cython(dimensions,warm,ranks,tmp_path,engine)

@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('window',[None,2])
def test_timed_rate_physical_positive_and_weak_bank_vjps(engine,ranks,zero,window):
    mpi(ranks);p,w=poisson_model(ranks,zero);p.update(backend=engine,tbptt_window=window);x,y,initial=fixture(3)
    result=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,y,initial=initial,noise_sequence=9)
    losses,final,events,records=poisson_physical(p,w,x,y,initial);expected=np.zeros(8);scale=0.
    for tick,bi,j,row,rate,count in records:
        coef=(poisson_physical(p,w,x,y,initial,(tick,bi,j))[0][bi]-losses[bi])/3 if zero else losses[bi]/3*(count/rate-1)
        expected[row]+=coef*w[3][0];scale+=coef*w[4][row]
    np.testing.assert_array_equal(result['spikes'],events);np.testing.assert_allclose(result['final_state'],final,rtol=6e-6,atol=3e-6)
    np.testing.assert_allclose(result['gradients'][4],expected,rtol=1e-3,atol=5e-6)
    assert result['gradients'][3][0]==pytest.approx(scale,rel=1e-3,abs=5e-6)
    assert np.any(expected!=0.) and result['backend']==engine and result['gpu_dispatches']>0

@pytest.mark.parametrize('ranks',[None,2,8])
def test_timed_input_edit_preserves_cache_and_detaches_old_rates(engine,ranks,tmp_path):
    mpi(ranks);p,w=poisson_model(ranks);p['backend']=engine;x,y,initial=fixture(3)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(x[:,:2],y,initial=initial,noise_sequence=9)
    cache=copy.deepcopy(t.poisson_state);live=copy.deepcopy(t.neuron_state);clock=(t.clock_tick,t.noise_sequence,t.next_noise_sequence)
    t.update_timed_input(4,[-1.]*8)
    assert t.poisson_state==cache and t.neuron_state==live and (t.clock_tick,t.noise_sequence,t.next_noise_sequence)==clock
    path=tmp_path/'timed-cache';t.store(path);q=NativeLIFTrainer(p,runner=RUNNER);q.restore(path)
    q.clock_tick=0;t.clock_tick=0;r=t.gradients(x[:,:2],y,initial='carry');assert q.gradients(x[:,:2],y,initial='carry')==r
    assert r['poisson_state']==cache;np.testing.assert_array_equal(r['gradients'][4],0.)
    t.clock_tick=2;before=copy.deepcopy(t.state)
    with pytest.raises(ValueError,match='GPU|nonfinite'):t.step(x[:,2:],y,initial='carry')
    assert t.state==before and t.poisson_state==cache and t.clock_tick==2


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('masked',[False,True])
def test_zero_timed_rate_mask_controls_counterfactual_failure(engine,ranks,masked):
    mpi(ranks);p,w=poisson_model(ranks,zero=True,invalid=True,masked=masked);p['backend']=engine;x,y,initial=fixture(1)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=snapshot(t)
    if masked:
        result=t.gradients(x[:,:1],y,initial=initial);np.testing.assert_array_equal(result['gradients'][4],0.)
    else:
        with pytest.raises(ValueError,match='GPU|nonfinite'):t.step(x[:,:1],y,initial=initial)
        assert snapshot(t)==before


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('bad',['count','duplicate','rate','missing','schema','profile','batch','sequence','kind','pending'])
def test_static_cache_restore_validation_and_atomic_retry(engine,ranks,bad,tmp_path):
    mpi(ranks);p,w=model(ranks=ranks);p['backend']=engine;x,y,initial=fixture(1)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(x[:,:1],y,initial=initial,noise_sequence=9)
    good=tmp_path/'good';broken=tmp_path/'broken';t.store(good);envelope=json.loads(good.read_text());payload=json.loads(envelope['payload']);cache=payload['poisson_state']
    if bad=='count':cache['entries'][0]['count']+=1
    elif bad=='duplicate':cache['entries']*=2
    elif bad=='rate':cache['entries'][0]['rate']=-1.
    elif bad=='missing':payload.pop('poisson_state')
    elif bad=='schema':cache['schema']='unknown'
    elif bad=='profile':cache['numeric_profile']='unknown'
    elif bad=='kind':cache['entries'][0]['identity']['site']['kind']=3
    elif bad=='pending':cache['entries'][0]['identity']['site']['pending']=1
    else:cache[bad]+=1
    raw=canonical_bytes(payload);envelope.update(payload=raw.decode(),sha256=hashlib.sha256(raw).hexdigest());broken.write_text(json.dumps(envelope))
    q=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=snapshot(q)
    with pytest.raises(ValueError,match='Poisson|poisson'):q.restore(broken)
    assert snapshot(q)==before;q.restore(good);assert q.poisson_state==t.poisson_state
    assert q.step(x[:,1:2],y,initial='carry')==t.step(x[:,1:2],y,initial='carry')


@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('rate',[1e9,2.146e9])
def test_large_device_counts_survive_bit_exact_cache_restore(engine,ranks,rate,tmp_path):
    mpi(ranks);p,w=model(ranks=ranks,rate='scale',reset_draw=False);p.update(backend=engine,trainable=[False]*len(w));w[3][0]=rate;w[3][1]=1e-9
    x,y,initial=fixture(1);t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(x[:,:1],y,initial=initial,noise_sequence=9)
    saved=copy.deepcopy(t.poisson_state);assert all(e['count']>2**24 for e in saved['entries'])
    assert any(e['count']!=int(np.float32(e['count'])) for e in saved['entries'])
    path=tmp_path/'large';t.store(path);q=NativeLIFTrainer(p,runner=RUNNER);q.restore(path);q.clock_tick=0
    q.state['weights'][3][0]=-1.;result=q.gradients(x[:,:1],y,initial='carry')
    assert result['poisson_state']==saved and result['gradients'][3][0]==0.


@pytest.mark.parametrize('rate',[float(np.nextafter(np.float32(0),np.float32(1))),float(np.finfo(np.float32).tiny),1e-12])
def test_tiny_positive_rate_has_score_and_no_weak_alternate(engine,rate):
    p,w=model(rate='scale',reset_draw=False);p['backend']=engine;w[3][0]=rate
    x,y,initial=fixture(1);r=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x[:,:1],y,initial=initial,noise_sequence=9)
    assert all(e['count']==0 and e['rate']==float(np.float32(rate)) for e in r['poisson_state']['entries'])
    assert r['gradients'][3][0]==pytest.approx(-3*r['loss'],rel=1e-5,abs=3e-6)
    assert r['gpu_dispatches']==14 # zero-site probe + baseline, no alternates


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('problem',['budget','invalid_rate','different_rate','mixed_distribution'])
def test_admission_and_last_owner_failure_are_atomic(engine,ranks,problem):
    mpi(ranks);p,w=model(ranks=ranks);p['backend']=engine;x,y,initial=fixture(1);match='rate expressions'
    if problem=='budget':
        good=NativeLIFTrainer(p,weights=w,runner=RUNNER).gradients(x,y,initial=initial)
        p['max_tape_bytes']=good['tape_bytes']-1;match='budget'
    elif problem=='invalid_rate':initial[0,5]=-1.;match='GPU|nonfinite'
    elif problem=='different_rate':p['state_resets'][1][1]=expression('r+.05*draw(scale*r+1.)',{'draw':PoissonNoise(0),'scale':(3,0)})
    else:p['state_resets'][1][1]=[{'op':'noise','stream':0}];match='mixes distributions'
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);before=snapshot(t)
    with pytest.raises(ValueError,match=match):t.step(x,y,initial=initial)
    assert snapshot(t)==before


@pytest.mark.parametrize('ranks',[None,2,8])
def test_negative_zero_rate_bits_survive_mpi_export_restore_and_rewind(engine,ranks,tmp_path):
    mpi(ranks);p,w=model(ranks=ranks,rate='-scale');p.update(backend=engine,trainable=[False]*len(w));w[3][0]=0.
    x,y,initial=fixture(1);t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(x[:,:1],y,initial=initial,noise_sequence=9)
    saved=copy.deepcopy(t.poisson_state);assert all(np.signbit(e['rate']) and e['rate']==0. and e['count']==0 for e in saved['entries'])
    path=tmp_path/'negative-zero';t.store(path);q=NativeLIFTrainer(p,runner=RUNNER);q.restore(path)
    assert all(np.signbit(e['rate']) for e in q.poisson_state['entries'])
    q.clock_tick=0;q.state['weights'][3][0]=1. # An invalid fresh rate is irrelevant to imported observations.
    result=q.gradients(x[:,:1],y,initial='carry');assert result['poisson_state']==saved
    assert all(np.signbit(e['rate']) for e in result['poisson_state']['entries'])
    assert result['gradients'][3][0]==0. and result['gpu_dispatches']>0 and result['backend']==engine

from test_training_linked import cython_cache
