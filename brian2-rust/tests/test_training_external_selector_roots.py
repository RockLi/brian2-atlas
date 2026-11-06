"""External integer roots address mutable graph state and timed physical fields."""
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

PRE=np.array([1,0,1,0]);POST=np.array([0,1,1,0]);DELAYS=np.array([1,2,0,1])
PATTERN=np.array([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]],float)
ROUTES=np.array([[0,1],[1,1],[1,0],[0,0],[0,1],[1,0]])
DRIVE=np.array([[.4,.7],[1.2,.6],[.8,1.3],[1.4,.5],[.3,.9],[1.1,.4]])


def model(*,engine='cpu',ranks=None,window=None,noisy=False,event_driven=False,convert=True):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    drive=b.TimedArray(DRIVE,dt=dt,name='root_drive_trace');route=b.TimedArray(ROUTES,dt=dt,name='root_route_trace')
    inp=b.NeuronGroup(2,'drive:1\nroute:integer',threshold='timestep(t,dt)%2==i',reset='',dt=dt,
        namespace={'drive_trace':drive,'route_trace':route},name='root_input')
    inp.run_regularly('drive=drive_trace(t,i); route=int(route_trace(t,i))',when='before_start',order=-3)
    hidden=b.NeuronGroup(2,'dv/dt=-v/ms:1\nz:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,name='root_hidden')
    hidden.z=[.1,.2]
    equation='dv/dt=(-v+gain*drive)/ms'
    if noisy:equation+=' + .03*drive*(1+v)*xi/sqrt(ms)'
    out=b.NeuronGroup(2,equation+':1\ndrive:1 (linked)\npick:integer (linked)\nwatched:1 (linked)\ngain:1 (constant)',
        threshold='v>.5+.025*drive',reset='v-=.5+.025*drive',method='heun' if noisy else 'euler',dt=dt,name='root_output')
    out.v=[.64,.29];out.gain=[.6,.9]
    out.pick=b.linked_var(inp,'route')
    out.drive=b.linked_var(inp,'drive',index='pick');out.watched=b.linked_var(hidden,'z',index='pick')
    out.run_regularly('watched+=.01*drive; v+=.02*watched',when='groups',order=-1)
    equation='dh/dt=(-h+sgain*picked)/ms'
    if noisy:equation+=' + .04*picked*(1+h)*xi/sqrt(ms)'
    syn=b.Synapses(inp,out,equation+(':1 (event-driven)' if event_driven else ':1 (clock-driven)')+'\nw:1\nsgain:1 (constant)',
        on_pre='z_routed+=.02*picked; v_post+=w*(1+h)+.03*z_routed',on_post='h+=.01*picked',
        method='heun' if noisy else 'euler',dt=dt,name='root_synapses')
    syn.connect(i=PRE,j=POST);syn.w=[.24,.35,.18,.21];syn.sgain=[.11,.08,.1,.07];syn.h=[.1,.2,.15,.13]
    syn.delay=DELAYS*dt
    syn.variables.add_reference('route_index',inp,'route',index='_presynaptic_idx')
    syn.variables.add_reference('picked',inp,'drive',index='route_index')
    syn.variables.add_reference('z_routed',hidden,'z',index='route_index')
    syn.run_regularly('h+=.003*picked',when='groups',order=-2)
    net=b.Network(inp,hidden,out,syn)
    # Conversion must never use the source's initial index snapshot.
    inp.route=[16777217,-2147483648]
    if not convert:return net,inp,hidden,out,syn,drive,route
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,out],external_state_inputs={'drive':drive,'route':route},
        trainable_neuron_parameters={out.name:['gain']},trainable_synapse_parameters={syn.name:['w','sgain']},
        backend=engine,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,seed=3527,learning_rate=1e-9)
    return net,hidden,out,syn,bundle


def oracle(bundle,*,weights=None,initial=None,anchors=None,noisy=False,event_driven=False,sample=0,
           routes=ROUTES,change=None,delays=DELAYS):
    p=bundle.plan;weights=bundle.weights if weights is None else weights
    state=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,param in enumerate(p['dynamic']['initial_parameters']):
            if param is not None:state[cell]=weights[param[0]][param[1]]
    hidden=bundle.provenance['neuron_state_layout']['root_hidden'];out=bundle.provenance['neuron_state_layout']['root_output']
    syn=bundle.provenance['dynamic_state_layout']['root_synapses']
    def bank(obj,name):return next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==obj and name in e['variables'])
    gain=np.array(weights[bank('root_output','gain')]);sgain=np.array(weights[bank('root_synapses','sgain')]);w=np.array(weights[bank('root_synapses','w')])
    source=bundle.provenance['external_state_inputs']['sources']['drive'];table=np.array(weights[source['bank']]).reshape(source['shape'])
    before=[];margins=[];spikes=[];draws=[];domain=bundle.provenance['synaptic_noise_domains']['root_synapses']
    for tick in range(6):
        if change is not None and tick==change[0]:table=np.array(change[1]).reshape(table.shape)
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:state=anchors['before'][tick].copy()
        before.append(state.copy());route=routes[tick];picked=table[tick,route[PRE]]
        state[syn['h']]+=.003*picked
        for j in range(2):
            column=route[j];cell=hidden['z'][column]
            state[cell]+=.01*table[tick,column];state[out['v'][j]]+=.02*state[cell]
        v=state[out['v']].copy();field=table[tick,route];u=.8*v+.2*gain*field
        if noisy:
            draw=np.array([normal(p['seed'],7,sample,1,j,tick,0) for j in range(2)]);draws.extend(draw)
            dw=np.sqrt(.2)*draw;g=.03*field*(1+v);support=v+g*dw
            u+=.5*dw*(g+.03*field*(1+support))
        state[out['v']]=u
        for edge in ([] if event_driven else range(4)):
            h=state[syn['h'][edge]];value=picked[edge];updated=.8*h+.2*sgain[edge]*value
            if noisy:
                draw=normal(p['seed'],7,sample,domain,edge,tick,0);draws.append(draw)
                dw=np.sqrt(.2)*draw;g=.04*value*(1+h);support=h+g*dw
                updated+=.5*dw*(g+.04*value*(1+support))
            state[syn['h'][edge]]=updated
        margin=u-(.5+.025*field);hard=(margin>0).astype(float);event=hard
        if anchors is not None:
            event=anchors['hard'][tick]+(margin-anchors['margins'][tick])/(1+5*abs(anchors['margins'][tick]))**2
        for edge in sorted(range(4),key=lambda k:(-int(delays[k]),int(PRE[k]),k)):
            emission=tick-int(delays[edge])
            if emission<0 or not PATTERN[emission,PRE[edge]]:continue
            if event_driven:
                decay=np.exp(-(.0002*tick-state[syn['lastupdate'][edge]])/.001)
                state[syn['h'][edge]]=decay*state[syn['h'][edge]]+(1-decay)*sgain[edge]*picked[edge]
                state[syn['lastupdate'][edge]]=.0002*tick
            z=hidden['z'][route[PRE[edge]]];state[z]+=.02*picked[edge]
            state[out['v'][POST[edge]]]+=w[edge]*(1+state[syn['h'][edge]])+.03*state[z]
        for edge,target in enumerate(POST):
            if event_driven:
                old=state[syn['h'][edge]];decay=np.exp(-(.0002*tick-state[syn['lastupdate'][edge]])/.001)
                changed=decay*old+(1-decay)*sgain[edge]*picked[edge]+.01*picked[edge]
                state[syn['h'][edge]]=old+event[target]*(changed-old)
                if hard[target]:state[syn['lastupdate'][edge]]=.0002*tick
            else:state[syn['h'][edge]]+=.01*picked[edge]*event[target]
        state[out['v']]-=event*(.5+.025*field);margins.append(margin.copy());spikes.append(event.copy())
    spikes=np.array(spikes);logits=5*spikes.mean(0);m=logits.max()
    return m+np.log(np.exp(logits-m).sum())-logits[0],state,spikes,dict(before=before,margins=np.array(margins),hard=spikes.copy(),draws=draws)


def physical_cells(bundle):
    return sorted({cell for group in bundle.provenance['neuron_state_layout'].values() for cells in group.values() for cell in cells}
                  |{cell for group in bundle.provenance['dynamic_state_layout'].values() for cells in group.values() for cell in cells})


@pytest.mark.parametrize('noisy,event_driven',[(False,False),(True,False),(False,True)])
@pytest.mark.parametrize('window',[None,2])
def test_external_root_physics_all_vjps(engine,noisy,event_driven,window):
    *_,bundle=model(engine=engine,noisy=noisy,event_driven=event_driven,window=window)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(PATTERN[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    expected=oracle(bundle,noisy=noisy,event_driven=event_driven);cells=physical_cells(bundle);tol=5e-5 if engine!='cpu' else 4e-12
    np.testing.assert_allclose(np.array(result['final_state'][0])[cells],expected[1][cells],rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(np.array(result['spikes'])[0,:,2:],expected[2]);assert result['loss']==pytest.approx(expected[0],abs=tol)
    integer_bank=bundle.provenance['external_state_inputs']['sources']['route']['bank']
    integers={tuple(x) for x in bundle.plan['dynamic']['integer_parameters']}
    for bank,values in enumerate(bundle.weights):
        for j in range(len(values)):
            if bank==integer_bank or (bank,j) in integers:assert result['gradients'][bank][j]==0;continue
            lo=copy.deepcopy(bundle.weights);hi=copy.deepcopy(lo);lo[bank][j]-=1e-6;hi[bank][j]+=1e-6
            fd=(oracle(bundle,weights=hi,anchors=expected[3],noisy=noisy,event_driven=event_driven)[0]-
                oracle(bundle,weights=lo,anchors=expected[3],noisy=noisy,event_driven=event_driven)[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=4e-4,abs=3e-6),(bank,j)
    for cell in cells:
        if bundle.plan['dynamic']['detached'][cell]:assert result['initial_state_gradients'][0][cell]==0;continue
        lo=np.array(bundle.initial_state);hi=lo.copy();lo[cell]-=1e-6;hi[cell]+=1e-6
        fd=(oracle(bundle,initial=hi,anchors=expected[3],noisy=noisy,event_driven=event_driven)[0]-
            oracle(bundle,initial=lo,anchors=expected[3],noisy=noisy,event_driven=event_driven)[0])/2e-6
        assert result['initial_state_gradients'][0][cell]==pytest.approx(fd,rel=4e-4,abs=3e-6),cell
    for entry in bundle.provenance['external_selector_caches']:
        assert all(result['initial_state_gradients'][0][k]==0 for k in entry['cells'])
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('noisy,event_driven',[(False,False),(True,False),(False,True)])
def test_external_root_original_cython(noisy,event_driven):
    net,hidden,out,syn,bundle=model(noisy=noisy,event_driven=event_driven);expected=oracle(bundle,noisy=noisy,event_driven=event_driven)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(PATTERN[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    net.run(0*b.ms,namespace={});device=b.get_device();device.randn_buffer_index[:]=0;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);values=np.zeros(n);values[:len(expected[3]['draws'])]=expected[3]['draws'];return values
    try:
        with patch('numpy.random.randn',refill):net.run(1.2*b.ms,namespace={})
        if noisy:assert device.randn_buffer_index[0]==len(expected[3]['draws'])
    finally:device.randn_buffer_index[:]=0
    np.testing.assert_allclose(hidden.z[:],expected[1][bundle.provenance['neuron_state_layout'][hidden.name]['z']],atol=2e-13)
    np.testing.assert_allclose(syn.h[:],expected[1][bundle.provenance['dynamic_state_layout'][syn.name]['h']],atol=2e-13)
    np.testing.assert_allclose(out.v[:],actual['final_membrane'][0][2:],atol=2e-13)
    assert syn.pre.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('noisy',[False,True])
def test_external_root_table_update_carry_restore(engine,ranks,noisy,tmp_path):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,bundle=model(engine=engine,ranks=ranks,noisy=noisy);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(PATTERN[None,:3],[0],**(dict(noise_sequence=7) if noisy else {}))
    source=bundle.provenance['external_state_inputs']['sources'];replacement=1-ROUTES
    trainer.update_external_state_input(source['route'],replacement);trainer.update_external_state_input(source['drive'],DRIVE*.7+.2)
    trainer.store(tmp_path/'external-root.json');restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    restored.restore(tmp_path/'external-root.json');actual=restored.step(PATTERN[None,3:],[0],initial='carry')
    routes=ROUTES.copy();routes[3:]=replacement[3:];expected=oracle(bundle,noisy=noisy,routes=routes,change=(3,DRIVE*.7+.2));cells=physical_cells(bundle)
    np.testing.assert_allclose(np.array(actual['final_state'][0])[cells],expected[1][cells],rtol=5e-5,atol=3e-6)


@pytest.mark.parametrize('bad',[-1,2,16777217,-2147483648])
def test_external_root_oob_read_is_atomic(engine,bad):
    *_,bundle=model(engine=engine);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(PATTERN[None,:2],[0])
    source=bundle.provenance['external_state_inputs']['sources']['route'];routes=ROUTES.copy();routes[2,0]=bad
    trainer.update_external_state_input(source,routes)
    before=copy.deepcopy((trainer.state,trainer.neuron_state,trainer.clock_state,trainer.clock_tick,trainer.poisson_state))
    with pytest.raises(ValueError,match='index|indirect|gather|nonfinite dynamic GPU result'):trainer.step(PATTERN[None,2:],[0],initial='carry')
    assert (trainer.state,trainer.neuron_state,trainer.clock_state,trainer.clock_tick,trainer.poisson_state)==before
    trainer.update_external_state_input(source,ROUTES);trainer.step(PATTERN[None,2:],[0],initial='carry')
    expected=oracle(bundle);cells=physical_cells(bundle)
    np.testing.assert_allclose(np.array(trainer.neuron_state[0])[cells],expected[1][cells],rtol=5e-5,atol=3e-6)


def asynchronous_model(engine='cpu',ranks=None):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    drive=b.TimedArray(DRIVE,dt=dt,name='async_root_drive');route=b.TimedArray(ROUTES,dt=dt,name='async_root_route')
    inp=b.NeuronGroup(2,'drive:1\nroute:integer',threshold='timestep(t,dt)%2==i',reset='',dt=dt,
        namespace={'drive_trace':drive,'route_trace':route},name='async_root_input')
    # This reference generator supplies the declared consumer-clock contract:
    # Synapses' pending .4 ms clock can lead the source's .2 ms clock.
    sample='dt*(((timestep(t,dt)+1)//2)*2)'
    inp.run_regularly('drive=drive_trace('+sample+',i); route=int(route_trace('+sample+',i))',when='before_start',order=-3)
    groups=[b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>.5',reset='v-=.5',dt=dt,method='euler',name=f'async_root_layer_{k}') for k in range(2)]
    groups[1].v=[.64,.29]
    syn=b.Synapses(inp,groups[1],'dh/dt=(-h+.1*picked)/ms:1 (clock-driven)\nw:1',
        on_pre='v_post+=w*(1+h)+.05*picked+.01*route_index',dt=2*dt,method='euler',name='async_root_synapses')
    syn.connect(i=[0,1],j=[0,1]);syn.w=[.24,.35];syn.h=[.1,.2]
    syn.variables.add_reference('route_index',inp,'route',index='_presynaptic_idx')
    syn.variables.add_reference('picked',inp,'drive',index='route_index')
    net=b.Network(inp,*groups,syn)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,external_state_inputs={'drive':drive,'route':route},
        trainable_synapse_parameters={syn.name:['w']},backend=engine,mpi_ranks=ranks,detach_reset=False)
    return net,groups,syn,bundle


def asynchronous_oracle(bundle,weights=None,anchors=None):
    weights=bundle.weights if weights is None else weights
    source=bundle.provenance['external_state_inputs']['sources']['drive'];table=np.array(weights[source['bank']]).reshape(source['shape'])
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='async_root_synapses' and 'w' in e['variables'])
    w=np.array(weights[bank]);v=np.array([.64,.29]);h=np.array([.1,.2]);margins=[];spikes=[]
    for tick in range(6):
        row=min(((tick+1)//2)*2,5);route=ROUTES[row];field=table[row,route]
        u=.8*v
        if tick%2==0:h=.6*h+.04*field
        hard=(u>.5).astype(float);event=hard
        if anchors is not None:event=anchors[1][tick]+(u-anchors[0][tick])/(1+5*abs(anchors[0][tick]-.5))**2
        v=u+PATTERN[tick]*(w*(1+h)+.05*field+.01*route)-.5*event
        margins.append(u.copy());spikes.append(event.copy())
    logits=5*np.array(spikes).mean(0);m=logits.max()
    return m+np.log(np.exp(logits-m).sum())-logits[0],v,h,(np.array(margins),np.array(spikes))


@pytest.mark.parametrize('ranks',[None,2])
def test_external_root_asynchronous_pending_clock_and_vjps(engine,ranks):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,bundle=asynchronous_model(engine,ranks);result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(PATTERN[None],[0])
    expected=asynchronous_oracle(bundle);tol=5e-5 if engine!='cpu' else 3e-12
    np.testing.assert_allclose(result['final_membrane'][0][2:],expected[1],rtol=tol,atol=tol)
    h=bundle.provenance['dynamic_state_layout']['async_root_synapses']['h']
    np.testing.assert_allclose(np.array(result['final_state'][0])[h],expected[2],rtol=tol,atol=tol)
    integer_bank=bundle.provenance['external_state_inputs']['sources']['route']['bank']
    integers={tuple(x) for x in bundle.plan['dynamic']['integer_parameters']}
    for bank,values in enumerate(bundle.weights):
        for j in range(len(values)):
            if bank==integer_bank or (bank,j) in integers:assert result['gradients'][bank][j]==0;continue
            lo=copy.deepcopy(bundle.weights);hi=copy.deepcopy(lo);lo[bank][j]-=1e-6;hi[bank][j]+=1e-6
            fd=(asynchronous_oracle(bundle,hi,expected[3])[0]-asynchronous_oracle(bundle,lo,expected[3])[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=4e-4,abs=3e-6)
    assert any(len(entry['execution_clocks'])==2 for entry in bundle.provenance['external_selector_caches'])
    assert (result['gpu_dispatches']>0)==(engine!='cpu')


def test_external_root_asynchronous_original_cython():
    net,groups,syn,bundle=asynchronous_model();expected=asynchronous_oracle(bundle)
    net.run(1.2*b.ms,namespace={})
    np.testing.assert_allclose(groups[1].v[:],expected[1],atol=2e-13)
    np.testing.assert_allclose(syn.h[:],expected[2],atol=2e-13)
    assert syn.pre.codeobj.compiled_code['run'] is not None


@pytest.mark.parametrize('noisy',[False,True])
def test_external_root_delay_update_preserves_pending_and_cache(engine,noisy,tmp_path):
    net,hidden,out,syn,bundle=model(engine=engine,noisy=noisy);bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.step(PATTERN[None,:3],[0],**(dict(noise_sequence=7) if noisy else {}))
    old_tick=trainer.clock_tick;new_delays=np.array([0,1,2,0])
    trainer.update_delays({syn.pre.name:new_delays*.2*b.ms});assert trainer.clock_tick==old_tick
    trainer.store(tmp_path/'root-delays.json');restored=NativeLIFTrainer(trainer.plan,runner=RUNNER,weights=bundle.weights)
    restored.restore(tmp_path/'root-delays.json');actual=restored.step(PATTERN[None,3:],[0],initial='carry')
    draws=oracle(bundle,noisy=noisy)[3]['draws'];net.run(0*b.ms,namespace={});device=b.get_device();device.randn_buffer_index[:]=0;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n);values=np.zeros(n);values[:len(draws)]=draws;return values
    try:
        with patch('numpy.random.randn',refill):
            net.run(.6*b.ms,namespace={});syn.delay=new_delays*.2*b.ms;net.run(.6*b.ms,namespace={})
        if noisy:assert device.randn_buffer_index[0]==len(draws)
    finally:device.randn_buffer_index[:]=0
    tol=5e-5 if engine!='cpu' else 3e-12
    np.testing.assert_allclose(out.v[:],actual['final_membrane'][0][2:],rtol=tol,atol=tol)
    np.testing.assert_allclose(hidden.z[:],np.array(actual['final_state'][0])[bundle.provenance['neuron_state_layout'][hidden.name]['z']],rtol=tol,atol=tol)
    np.testing.assert_allclose(syn.h[:],np.array(actual['final_state'][0])[bundle.provenance['dynamic_state_layout'][syn.name]['h']],rtol=tol,atol=tol)
    assert (actual['gpu_dispatches']>0)==(engine!='cpu')
