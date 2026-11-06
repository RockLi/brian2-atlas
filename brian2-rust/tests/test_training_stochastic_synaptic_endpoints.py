"""SDE, cached random modulation and delayed physical Synapses endpoints."""
import copy
import os
from unittest.mock import patch

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_stochastic import normal
from test_training_uniform import uniform

PRE=np.array([0,1,0,1]);POST=np.array([0,1,1,0])
MOD_PRE=np.array([0,1,1,0,0]);MOD_POST=np.array([3,0,3,1,2])
PATTERN=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1]],float)
DELAYS=[1,2,0,1];MOD_DELAYS=[0,1,2,0,1]


def model(*,engine='cpu',ranks=None,window=None,delayed=True,method='euler'):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    ticks,ids=np.nonzero(PATTERN)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='random_endpoint_input')
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,
                         name=f'random_endpoint_layer_{k}') for k in range(2)]
    groups[1].v=[.64,.29]
    coefficient='sigma' if method=='euler' else 'sigma*(1+h)'
    conn=b.Synapses(inp,groups[1],f'''w:1
dh/dt=-h/ms+{coefficient}*xi_shared/sqrt(ms):1 (clock-driven)
sigma:1 (constant)''',on_pre='v_post+=w+.05*h',method=method,dt=dt,name='z_random_endpoint_conn')
    conn.connect(i=PRE,j=POST);conn.w=[.24,.35,.18,.21];conn.h=[.1,.2,.15,.13];conn.sigma=[.03,.02,.025,.035]
    conn.pre.order=0
    mod=b.Synapses(inp,conn,'''gain:1
u=rand():1 (constant over dt)
r=randn():1 (constant over dt)''',on_pre='w_post+=gain*(1+.1*u)+.01*r',dt=dt,name='a_random_endpoint_mod')
    mod.connect(i=MOD_PRE,j=MOD_POST);mod.gain=[.03,.04,.02,.01,.025];mod.pre.order=-2
    mod.subexpression_updater.when='before_start'
    if delayed:conn.delay=np.array(DELAYS)*dt;mod.delay=np.array(MOD_DELAYS)*dt
    net=b.Network(inp,*groups,conn,mod)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine,mpi_ranks=ranks,
        trainable_synapse_parameters={conn.name:['w','sigma'],mod.name:['gain']},detach_reset=False,
        tbptt_window=window,seed=917,learning_rate=1e-9)
    return net,groups,conn,mod,bundle


def oracle(bundle,*,weights=None,initial=None,anchors=None,delayed=True,sequence=7,migration=False,method='euler',sample_index=0):
    """Physical recurrences; no executable IR/tape interpretation."""
    plan=bundle.plan;weights=bundle.weights if weights is None else weights
    state=np.asarray(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for cell,parameter in enumerate(plan['dynamic']['initial_parameters']):
            if parameter is not None:state[cell]=weights[parameter[0]][parameter[1]]
    layouts=bundle.provenance['dynamic_state_layout'];conn=layouts['z_random_endpoint_conn'];mod=layouts['a_random_endpoint_mod']
    voltage=bundle.provenance['neuron_state_layout']['random_endpoint_layer_1']['v']
    banks={(e['object'],e['variables'][0]):e['bank'] for e in bundle.provenance['bindings'] if len(e['variables'])==1}
    sigma=np.asarray(weights[banks['z_random_endpoint_conn','sigma']]);gain=np.asarray(weights[banks['a_random_endpoint_mod','gain']])
    cache=next(row for name,row in bundle.provenance['regular_runner_layout'].items() if name.startswith('a_random_endpoint_mod_subexpression_update'))
    sde_domain=bundle.provenance['synaptic_noise_domains']['z_random_endpoint_conn']
    draws={'rand':[],'randn':[]};events=[];membranes=[];before=[]
    active=np.ones(len(PRE),bool);generation=np.zeros(len(PRE),int)
    for tick in range(len(PATTERN)):
        if migration and tick in (2,4):
            active[0]=tick==4;generation[0]=tick
            # Foreign unmasked modulation still owns w[0]. Only this edge's
            # private continuous trace and delayed emissions restart.
            state[conn['h'][0]]=bundle.initial_state[conn['h'][0]] if tick==4 else 0.
        if anchors is not None and plan['tbptt_window'] and tick and tick%plan['tbptt_window']==0:
            state=anchors['before'][tick].copy()
        before.append(state.copy())
        # Cython reads the cached vector code in edge order before groups.
        for edge in range(len(MOD_POST)):
            values={}
            for call in cache['draws']['vector']:
                kind=call['kind'];value=(uniform if kind=='rand' else normal)(plan['seed'],sequence,sample_index,cache['noise_domain'],edge,tick,call['stream'])
                draws[kind].append(value);values[kind]=value
            state[mod['u'][edge]]=values['rand'];state[mod['r'][edge]]=values['randn']
        u=.8*state[voltage];state[voltage]=u
        for edge in range(len(PRE)):
            draw=normal(plan['seed'],sequence,sample_index,sde_domain,edge,tick,0);draws['randn'].append(draw)
            if not active[edge]:continue
            h=state[conn['h'][edge]];dw=np.sqrt(.2)*draw;drift=-h
            if method=='euler':value=h+.2*drift+sigma[edge]*dw
            else:
                base=sigma[edge]*(1+h)
                if method=='heun':
                    support=h+base*dw
                    value=h+.2*drift+.5*dw*(base+sigma[edge]*(1+support))
                elif method=='milstein':
                    # Brian's published derivative-free formula includes the
                    # drift in its support and dW**2, with no Ito subtraction.
                    support=h+.2*drift+np.sqrt(.2)*base
                    value=h+.2*drift+base*dw+(sigma[edge]*(1+support)-base)*dw**2/(2*np.sqrt(.2))
                else:raise AssertionError(method)
            state[conn['h'][edge]]=value
        hard=(u>.5).astype(float);event=hard
        if anchors is not None:
            old=anchors['membranes'][tick];event=anchors['hard'][tick]+(u-old)/(1+5*np.abs(old-.5))**2
        for edge,target in enumerate(MOD_POST):
            emitted=tick-(MOD_DELAYS[edge] if delayed else 0)
            if emitted>=0:
                state[conn['w'][target]]+=PATTERN[emitted,MOD_PRE[edge]]*(gain[edge]*(1+.1*state[mod['u'][edge]])+.01*state[mod['r'][edge]])
        for edge,target in enumerate(POST):
            emitted=tick-(DELAYS[edge] if delayed else 0)
            if active[edge] and emitted>=generation[edge]:state[voltage[target]]+=PATTERN[emitted,PRE[edge]]*(state[conn['w'][edge]]+.05*state[conn['h'][edge]])
        state[voltage]-=.5*event;events.append(event.copy());membranes.append(u.copy())
    events=np.asarray(events);logits=5*events.mean(0);maximum=logits.max()
    loss=maximum+np.log(np.exp(logits-maximum).sum())-logits[0]
    return loss,state,events,dict(before=before,membranes=np.asarray(membranes),hard=events.copy(),draws=draws)


@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('delayed',[False,True])
@pytest.mark.parametrize('method',['euler','heun','milstein'])
def test_stochastic_endpoint_forward_and_all_vjps(engine,window,delayed,method):
    net,groups,conn,mod,bundle=model(engine=engine,window=window,delayed=delayed,method=method)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    result=trainer.gradients(PATTERN[None],[0],noise_sequence=7);expected=oracle(bundle,delayed=delayed,method=method);tol=5e-5 if engine!='cpu' else 4e-12
    np.testing.assert_array_equal(np.asarray(result['spikes'])[0,:,2:],expected[2])
    np.testing.assert_allclose(result['loss'],expected[0],rtol=tol,atol=tol*.01)
    slots=sorted({cell for row in bundle.provenance['dynamic_state_layout'].values() for cells in row.values() for cell in cells}
                 |{cell for row in bundle.provenance['neuron_state_layout'].values() for name,cells in row.items()
                        if not name.startswith('__') for cell in cells})
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],expected[1][slots],rtol=tol,atol=tol*.01)
    fd_tol=3e-4 if engine!='cpu' else 4e-8
    for bank,row in enumerate(bundle.weights):
        for edge in range(len(row)):
            plus=copy.deepcopy(bundle.weights);minus=copy.deepcopy(bundle.weights);plus[bank][edge]+=1e-6;minus[bank][edge]-=1e-6
            a=oracle(bundle,weights=plus,anchors=expected[3],delayed=delayed,method=method)[0];c=oracle(bundle,weights=minus,anchors=expected[3],delayed=delayed,method=method)[0]
            assert result['gradients'][bank][edge]==pytest.approx((a-c)/2e-6,rel=fd_tol,abs=fd_tol),(bank,edge)
    for cell in slots:
        plus=np.asarray(bundle.initial_state).copy();minus=plus.copy();plus[cell]+=1e-6;minus[cell]-=1e-6
        a=oracle(bundle,initial=plus,anchors=expected[3],delayed=delayed,method=method)[0];c=oracle(bundle,initial=minus,anchors=expected[3],delayed=delayed,method=method)[0]
        assert result['initial_state_gradients'][0][cell]==pytest.approx((a-c)/2e-6,rel=fd_tol,abs=fd_tol),cell
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('delayed',[False,True])
@pytest.mark.parametrize('method',['euler','heun','milstein'])
def test_stochastic_endpoint_original_cython_draw_replay(delayed,method):
    net,groups,conn,mod,bundle=model(delayed=delayed,method=method);expected=oracle(bundle,delayed=delayed,method=method)
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(0*b.ms,namespace={})
    device=b.get_device();draws=expected[3]['draws'];calls={kind:0 for kind in draws}
    device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    def refill(kind,n):
        assert n==20000 and calls[kind]==0;calls[kind]+=1;values=np.zeros(n);values[:len(draws[kind])]=draws[kind];return values
    try:
        with patch('numpy.random.rand',lambda n:refill('rand',n)),patch('numpy.random.randn',lambda n:refill('randn',n)):
            net.run(1.2*b.ms,namespace={})
        assert calls=={'rand':1,'randn':1}
        for kind in draws:assert getattr(device,kind+'_buffer_index')[0]==len(draws[kind])
    finally:device.rand_buffer_index[:]=0;device.randn_buffer_index[:]=0
    assert conn.state_updater.codeobj.compiled_code['run'] is not None
    assert mod.subexpression_updater.codeobj.compiled_code['run'] is not None
    for obj in (conn,mod):
        for name,cells in bundle.provenance['dynamic_state_layout'][obj.name].items():
            np.testing.assert_allclose(obj.variables[name].get_value(),expected[1][cells],rtol=4e-12,atol=4e-14)
    np.testing.assert_allclose(groups[1].v[:],expected[1][bundle.provenance['neuron_state_layout'][groups[1].name]['v']],rtol=4e-12,atol=4e-14)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_stochastic_endpoint_carry_restore_and_readonly(engine,ranks,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    _,_,_,_,bundle=model(engine=engine,ranks=ranks);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    whole=trainer.evaluate(PATTERN[None],[0],noise_sequence=7);trainer.step(PATTERN[None,:3],[0],noise_sequence=7)
    path=tmp_path/'stochastic-endpoint.json';trainer.store(path);before=path.read_bytes()
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,request_timeout=300);restored.restore(path)
    first=restored.gradients(PATTERN[None,3:],[0],initial='carry');restored.store(path);assert path.read_bytes()==before
    second=restored.gradients(PATTERN[None,3:],[0],initial='carry');assert first==second
    tail=restored.step(PATTERN[None,3:],[0],initial='carry');tol=5e-5 if engine!='cpu' else 4e-12
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(np.asarray(tail['spikes']),np.asarray(whole['spikes'])[:,3:]);assert tail['final_tick']==6


@pytest.mark.parametrize('method',['heun','milstein'])
@pytest.mark.parametrize('ranks',[2,8])
def test_multiplicative_endpoint_mpi_partition_invariant_vjp(engine,method,ranks):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    _,_,_,_,bundle=model(engine=engine,method=method,window=2)
    serial=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    expected=serial.gradients(PATTERN[None],[0],noise_sequence=7)
    plan=copy.deepcopy(bundle.plan);plan['mpi_ranks']=ranks
    parallel=NativeLIFTrainer(plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    result=parallel.gradients(PATTERN[None],[0],noise_sequence=7)
    tol=3e-4 if engine!='cpu' else 4e-11
    for name in ('final_state','initial_state_gradients','gradients','logits'):
        # Different bank lengths require comparing parameter rows separately.
        if name=='gradients':
            for actual,reference in zip(result[name],expected[name]):np.testing.assert_allclose(actual,reference,rtol=tol,atol=tol*.01)
        else:np.testing.assert_allclose(result[name],expected[name],rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(result['spikes'],expected['spikes'])
    assert result['loss']==pytest.approx(expected['loss'],rel=tol,abs=tol*.01)
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('method',['euler','heun','milstein'])
def test_stochastic_endpoint_batch_draw_addresses_and_mean_vjp(engine,method):
    _,groups,conn,mod,bundle=model(engine=engine,method=method)
    inputs=np.stack([PATTERN,PATTERN]);trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    result=trainer.gradients(inputs,[0,0],noise_sequence=7)
    expected=[oracle(bundle,method=method,sample_index=sample) for sample in range(2)]
    tol=5e-5 if engine!='cpu' else 4e-12;fd_tol=3e-4 if engine!='cpu' else 4e-8
    slots=sorted({cell for row in bundle.provenance['dynamic_state_layout'].values() for cells in row.values() for cell in cells}
                 |{cell for row in bundle.provenance['neuron_state_layout'].values() for name,cells in row.items() if not name.startswith('__') for cell in cells})
    np.testing.assert_allclose(np.asarray(result['final_state'])[:,slots],np.array([e[1][slots] for e in expected]),rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(np.asarray(result['spikes'])[:,:,2:],np.array([e[2] for e in expected]))
    assert result['loss']==pytest.approx(np.mean([e[0] for e in expected]),rel=tol,abs=tol*.01)
    assert not np.allclose(expected[0][3]['draws']['rand'],expected[1][3]['draws']['rand'])
    assert not np.allclose(expected[0][3]['draws']['randn'],expected[1][3]['draws']['randn'])
    def averaged(weights):
        return np.mean([oracle(bundle,weights=weights,anchors=expected[sample][3],method=method,sample_index=sample)[0] for sample in range(2)])
    for bank,row in enumerate(bundle.weights):
        for edge in range(len(row)):
            plus=copy.deepcopy(bundle.weights);minus=copy.deepcopy(bundle.weights);plus[bank][edge]+=1e-6;minus[bank][edge]-=1e-6
            assert result['gradients'][bank][edge]==pytest.approx((averaged(plus)-averaged(minus))/2e-6,rel=fd_tol,abs=fd_tol),(bank,edge)
    for sample in range(2):
        for cell in slots:
            plus=np.asarray(bundle.initial_state).copy();minus=plus.copy();plus[cell]+=1e-6;minus[cell]-=1e-6
            a=oracle(bundle,initial=plus,anchors=expected[sample][3],method=method,sample_index=sample)[0]
            c=oracle(bundle,initial=minus,anchors=expected[sample][3],method=method,sample_index=sample)[0]
            # Each sample's physical-state adjoint includes mean-loss scaling.
            assert result['initial_state_gradients'][sample][cell]==pytest.approx((a-c)/4e-6,rel=fd_tol,abs=fd_tol),(sample,cell)


@pytest.mark.parametrize('ranks',[None,2,8])
def test_stochastic_endpoint_migration_preserves_foreign_writer(engine,ranks,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    _,groups,conn,_,bundle=model(engine=engine,ranks=ranks);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300)
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==conn.name and e['variables']==['w'])
    layout=bundle.provenance['dynamic_state_layout'][conn.name];all_spikes=[]
    def cursors():return copy.deepcopy((trainer.clock_tick,trainer.clock_state,trainer.noise_sequence,trainer.next_noise_sequence,trainer.poisson_state,trainer.state['step'],trainer.state['rng']))
    for start,stop in [(0,2),(2,4),(4,6)]:
        result=trainer.step(PATTERN[None,start:stop],[0],**(dict(noise_sequence=7) if start==0 else dict(initial='carry')))
        all_spikes.extend(result['spikes'][0])
        if stop==6:break
        before=cursors();old_w=trainer.neuron_state[0][layout['w'][0]];masks=copy.deepcopy(trainer.plan['masks']);masks[bank][0]=0 if stop==2 else 1
        trainer.update_mask(masks,growth_weight=.41);assert cursors()==before
        assert trainer.neuron_state[0][layout['w'][0]]==old_w
        assert trainer.neuron_state[0][layout['h'][0]]==(0. if stop==2 else bundle.initial_state[layout['h'][0]])
        if stop==4:
            path=tmp_path/'migrated-stochastic-endpoint.json';trainer.store(path)
            restored=NativeLIFTrainer(trainer.plan,runner=RUNNER,request_timeout=300);restored.restore(path);trainer=restored
    expected=oracle(bundle,migration=True);tol=5e-5 if engine!='cpu' else 4e-12
    np.testing.assert_array_equal(np.asarray(all_spikes)[:,2:],expected[2])
    voltage=bundle.provenance['neuron_state_layout'][groups[1].name]['v']
    slots=[*voltage,*layout['w'],*layout['h']]
    np.testing.assert_allclose(np.asarray(trainer.neuron_state)[0,slots],expected[1][slots],rtol=tol,atol=tol*.01)
