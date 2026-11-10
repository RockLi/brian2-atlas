"""Structural-boundary migration: physical state, delayed events and shared owners."""
import copy
import os
import tempfile
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_delays import model,oracle


@pytest.fixture(scope='module',autouse=True)
def cython_cache():
    old=b.prefs.codegen.runtime.cython.cache_dir
    with tempfile.TemporaryDirectory(prefix='b2-migration-cython-') as directory:
        b.prefs.codegen.runtime.cython.cache_dir=directory
        try:yield
        finally:b.prefs.codegen.runtime.cython.cache_dir=old


def snapshot(trainer):
    return copy.deepcopy((trainer.state,trainer.neuron_state,trainer.plan,trainer.clock_tick,trainer.elapsed_ticks,
                          trainer.noise_sequence,trainer.next_noise_sequence,trainer.last_result,trainer.clock_state,trainer.poisson_state))


def cursors(trainer):
    return (trainer.state['step'],trainer.state['rng'],trainer.clock_tick,trainer.elapsed_ticks,
            trainer.noise_sequence,trainer.next_noise_sequence)


def bank(bundle,obj,name='w'):
    return next(a['bank'] for a in bundle.provenance['bindings'] if a['object']==obj and a['variables']==[name])


def manual_restart(bundle,live,obj,edge,*,active,tick,growth):
    """Model-specific reference, independent of native ownership metadata."""
    out=np.asarray(live).copy()
    for name,indices in bundle.provenance['dynamic_state_layout'][obj].items():
        k=indices[edge]
        value=0
        if active:
            if name=='w':value=growth
            elif name=='lastupdate':value=bundle.plan['clock']['origin']+tick*bundle.plan['clock']['dt']
            else:value=bundle.initial_state[k]
        out[:,k]=value
    for path,layout in bundle.provenance['delay_queues'].items():
        if path.startswith(obj+'_'):
            for entry in layout['pending']+layout['new']:
                if entry['edge']==edge:out[:,entry['states']]=0
    return out


def compare_phase(bundle,trainer,x,initial,tick):
    reference=copy.deepcopy(bundle);reference.plan=copy.deepcopy(trainer.plan)
    reference.plan['clock']['origin']+=tick*reference.plan['clock']['dt']
    expectations=[oracle(reference,trainer.state['weights'],row,initial=state,post_first=True,order_sensitive=True)
                  for row,state in zip(x,initial)]
    result=trainer.execute(x,[0]*len(x),initial='carry' if tick else initial)
    np.testing.assert_array_equal(result['spikes'],[e[1] for e in expectations])
    np.testing.assert_allclose(result['final_state'],[e[2] for e in expectations],rtol=2e-13,atol=2e-13)
    assert result['loss']==pytest.approx(np.mean([e[0] for e in expectations]),abs=3e-14)
    return np.array([e[2] for e in expectations])


@pytest.mark.parametrize('event_driven',[False,True])
@pytest.mark.parametrize('warmup',[0.,.8])
@pytest.mark.parametrize('batch',[1,2])
@pytest.mark.parametrize('window',[None,2])
def test_prune_carry_regrow_restore_matches_independent_reference(event_driven,warmup,batch,window,tmp_path):
    *_,x,bundle=model(event_driven,warmup,post_first=True,order_sensitive=True)
    bundle.plan['tbptt_window']=window;bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    inputs=np.stack([x,x[:,::-1]][:batch]);initial=np.tile(bundle.initial_state,(batch,1))
    expected=compare_phase(bundle,trainer,inputs[:,:2],initial,0)
    changes=[('front_static',2),('front_stdp',1)]
    masks=copy.deepcopy(trainer.plan['masks']);old=cursors(trainer)
    for obj,edge in changes:
        bnk=bank(bundle,obj);masks[bnk][edge]=0
        trainer.state['first_moment'][bnk][edge]=.31;trainer.state['second_moment'][bnk][edge]=.27
        expected=manual_restart(bundle,expected,obj,edge,active=False,tick=2,growth=0)
    neuron_width=sum(len(programs)*size for programs,size in zip(trainer.plan['state_equations'],trainer.plan['sizes'][1:]))
    neuron_before=np.asarray(trainer.neuron_state)[:,:neuron_width].copy()
    trainer.update_mask(masks)
    np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[:,:neuron_width],neuron_before)
    assert cursors(trainer)==old
    np.testing.assert_allclose(trainer.neuron_state,expected,rtol=2e-13,atol=2e-13)
    for obj,edge in changes:
        bnk=bank(bundle,obj)
        assert trainer.state['weights'][bnk][edge]==trainer.state['first_moment'][bnk][edge]==trainer.state['second_moment'][bnk][edge]==0
    expected=compare_phase(bundle,trainer,inputs[:,2:4],expected,2)
    old=cursors(trainer);growth=.43
    for obj,edge in changes:
        masks[bank(bundle,obj)][edge]=1
        expected=manual_restart(bundle,expected,obj,edge,active=True,tick=4,growth=growth)
    trainer.update_mask(masks,growth_weight=growth)
    assert cursors(trainer)==old
    np.testing.assert_allclose(trainer.neuron_state,expected,rtol=2e-13,atol=2e-13)
    filename=tmp_path/'migration.json';trainer.store(filename)
    restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(filename)
    assert cursors(restored)==cursors(trainer)
    compare_phase(bundle,restored,inputs[:,4:],expected,4)


@pytest.mark.parametrize('ranks',[2,8])
def test_migration_mpi_matches_serial_and_preserves_cursor(ranks,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=model(warmup=.8,post_first=True,order_sensitive=True)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    serial=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    plan=copy.deepcopy(bundle.plan);plan['mpi_ranks']=ranks
    parallel=NativeLIFTrainer(plan,runner=RUNNER,weights=bundle.weights)
    for start,stop in [(0,2),(2,4),(4,len(x))]:
        for trainer in (serial,parallel):
            trainer.execute(x[None,start:stop],[0],**({} if start==0 else dict(initial='carry')))
        np.testing.assert_allclose(parallel.neuron_state,serial.neuron_state,rtol=2e-13,atol=2e-13)
        if stop==len(x):break
        masks=copy.deepcopy(serial.plan['masks']);masks[bank(bundle,'front_stdp')][1]=0 if stop==2 else 1
        for trainer in (serial,parallel):
            before=cursors(trainer);trainer.update_mask(masks,growth_weight=.43);assert cursors(trainer)==before
        np.testing.assert_allclose(parallel.neuron_state,serial.neuron_state,rtol=2e-13,atol=2e-13)
        filename=tmp_path/'mpi.json';parallel.store(filename)
        other=NativeLIFTrainer(parallel.plan,runner=RUNNER);other.restore(filename);parallel=other
    for a,c in zip(parallel.last_result['gradients'],serial.last_result['gradients']):np.testing.assert_allclose(a,c,rtol=3e-13,atol=3e-13)


@pytest.mark.parametrize('issue',['growth_nan','growth_inf','growth_bad','mask_shape','mask_value','auxiliary_mask','bad_live','bad_binary','bad_moments','no_committed_state'])
def test_migration_failures_leave_all_state_unchanged(issue):
    *_,x,bundle=model();trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    if issue!='no_committed_state':trainer.execute(x[None,:3],[0])
    masks=copy.deepcopy(trainer.plan['masks']);masks[bank(bundle,'front_stdp')][1]=0;growth=.2
    if issue=='growth_nan':growth=float('nan')
    elif issue=='growth_inf':growth=float('inf')
    elif issue=='growth_bad':growth='not-a-number'
    elif issue=='mask_shape':masks[0].append(1)
    elif issue=='mask_value':masks[0][0]=.2
    elif issue=='auxiliary_mask':masks[bank(bundle,'front_stdp','taup')][0]=0
    elif issue=='bad_live':trainer.neuron_state[0].pop()
    elif issue=='bad_binary':trainer.neuron_state[0][bundle.plan['dynamic']['binary_states'][0]]=.2
    elif issue=='bad_moments':trainer.state['second_moment'][0][0]=-.1
    before=snapshot(trainer)
    with pytest.raises(ValueError):trainer.update_mask(masks,growth_weight=growth)
    assert snapshot(trainer)==before


@pytest.mark.parametrize('issue',['missing_cell','extra_owner','neuron_cell','duplicate_cell','duplicate_owner','drop_control','unknown_control','restart_queue','restart_time','unmasked_writer'])
def test_native_ownership_validation_rejects_incomplete_or_unsafe_layout(issue):
    *_,x,bundle=model();p=copy.deepcopy(bundle.plan);layout=p['dynamic']['migration'];cell=layout['cells'][0]
    if issue=='missing_cell':layout['cells'].pop()
    elif issue=='extra_owner':cell['owners'].append(next(m for m in layout['controlled_masks'] if m not in cell['owners']))
    elif issue=='neuron_cell':cell['index']=0
    elif issue=='duplicate_cell':layout['cells'].append(copy.deepcopy(cell))
    elif issue=='duplicate_owner':cell['owners'].append(cell['owners'][0])
    elif issue=='drop_control':layout['controlled_masks'].pop()
    elif issue=='unknown_control':layout['controlled_masks'].append([999,0])
    elif issue=='restart_queue':cell['restart']=dict(kind='queue')
    elif issue=='restart_time':cell['restart']=dict(kind='time',clock=999)
    elif issue=='unmasked_writer':
        ps=len(p['dynamic']['program_sets']);p['dynamic']['program_sets'].append([[dict(op='constant',value=0.)]])
        p['dynamic']['actions'].append(dict(owner=0,reads=[cell['index']],writes=[cell['index']],threshold=None,trigger=None,program_set=ps))
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights);before=snapshot(trainer)
    with pytest.raises(ValueError,match='migration'):trainer.execute(x[None],[0])
    assert snapshot(trainer)==before


def test_noop_masks_preserve_last_result_and_all_cursors():
    *_,x,bundle=model();trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.execute(x[None,:2],[0])
    before=snapshot(trainer);trainer.update_mask(copy.deepcopy(trainer.plan['masks']),growth_weight=.99)
    assert snapshot(trainer)==before


@pytest.mark.parametrize('cursor',['checkpoint','prefix'])
def test_late_native_clock_failure_rolls_back_and_allows_retry(cursor):
    *_,x,bundle=model();trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    trainer.execute(x[None,:3],[0])
    masks=copy.deepcopy(trainer.plan['masks']);bnk=bank(bundle,'front_stdp');masks[bnk][1]=0
    trainer.update_mask(masks)
    old_tick=trainer.clock_tick;old_cursor=copy.deepcopy(trainer.clock_state)
    if cursor=='prefix':trainer.clock_state=None
    trainer.clock_tick=10_000_001
    masks[bnk][1]=1;before=snapshot(trainer)
    # A mismatched saved cursor rejects before staging. Without that cursor,
    # replay fails after parameter/physical-cell changes have been staged.
    # Both paths must preserve every trainer field and permit a valid retry.
    reason='checkpoint tick/width mismatch' if cursor=='checkpoint' else 'work budget'
    with pytest.raises(ValueError,match=reason):trainer.update_mask(masks,growth_weight=.43)
    assert snapshot(trainer)==before
    trainer.clock_tick=old_tick;trainer.clock_state=old_cursor;trainer.update_mask(masks,growth_weight=.43)
    assert trainer.plan['masks'][bnk][1]==1
    assert trainer.state['weights'][bnk][1]==.43


@pytest.mark.parametrize('growth',[float('nan'),float('inf'),'bad'])
def test_static_mask_invalid_growth_is_transactional(growth):
    from brian2_rust.training import lif_training_plan
    p=lif_training_plan([2,2,2]);trainer=NativeLIFTrainer(p,runner=RUNNER,weights=[[.1]*4,[.2]*4]);before=snapshot(trainer)
    masks=copy.deepcopy(p['masks']);masks[0][0]=0
    with pytest.raises(ValueError):trainer.update_mask(masks,growth_weight=growth)
    assert snapshot(trainer)==before


def shared_plan():
    from test_training_dynamic import model as raw_model
    from brian2_rust.training_dynamic import compile_dynamic_transform,dynamic_action
    p,w,x=raw_model();d=p['dynamic'];q=len(d['initial'])
    d['initial'].append(.37);d['initial_parameters'].append(None);d['detached'].append(False)
    for action in d['actions']:
        if action['threshold'] is not None:continue
        writes=action['writes'];trigger=action['trigger']
        if trigger and trigger['external']:action['mask']=[0,2*trigger['index']+action['owner']]
        elif any(4<=k<16 for k in writes):action['mask']=[1,(next(k for k in writes if 4<=k<16)-4)%4]
    for edge in (0,1):
        tr=compile_dynamic_transform('q=.8*q+.3*w\nv+=.05*q',states={'q':0,'v':1},parameters={'w':(1,edge)})
        ps=len(d['program_sets']);d['program_sets'].append(tr['programs'])
        d['actions'].append(dynamic_action(tr,[q,2+edge],owner=2+edge,program_set=ps,mask=[1,edge]))
    d['migration']=dict(controlled_masks=[[bank,edge] for bank in (0,1) for edge in range(4)],
        cells=[dict(index=k,owners=[[1,(k-4)%4]],restart=dict(kind='initial')) for k in range(4,16)]+
              [dict(index=q,owners=[[1,0],[1,1]],restart=dict(kind='initial'))])
    p['trainable']=[False]*len(w)
    return p,w,x,q


@pytest.mark.parametrize('ranks',[None,2])
def test_shared_state_survives_existing_owner_and_restarts_for_new_generation(ranks):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x,q=shared_plan()
    if ranks:p['mpi_ranks']=ranks
    trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w);initial=np.tile(p['dynamic']['initial'],(2,1));initial[1,q]=.61
    trainer.execute(np.tile(x[None,:2],(2,1,1)),[0,0],initial=initial)
    masks=copy.deepcopy(p['masks']);masks[1][0]=0;before=np.array(trainer.neuron_state)
    trainer.update_mask(masks)
    np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[:,q],before[:,q])
    masks[1][1]=0;trainer.update_mask(masks)
    np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[:,q],[0.,0.])
    masks[1][0]=1;trainer.update_mask(masks,growth_weight=.41)
    np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[:,q],[.37,.37])
    trainer.execute(np.tile(x[None,2:3],(2,1,1)),[0,0],initial='carry')
    assert np.all(np.asarray(trainer.neuron_state)[:,q]!=.37)
    # No old owner survives: switch 0 -> 1 in one atomic mask change.
    masks[1][0]=0;masks[1][1]=1;trainer.update_mask(masks,growth_weight=.52)
    np.testing.assert_array_equal(np.asarray(trainer.neuron_state)[:,q],[.37,.37])


@pytest.mark.parametrize('clock_dt',[.3,.99999])
def test_regrowth_timestamps_use_actual_asynchronous_boundary_clock(clock_dt):
    from test_training_dynamic_clocks import model as clock_model
    net,inp,layers,syn,x,dt,bundle=clock_model(clock_dt,third=True)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.execute(x[None,:5],[0])
    masks=copy.deepcopy(bundle.plan['masks']);bnk=bank(bundle,syn.name);masks[bnk][0]=0;trainer.update_mask(masks)
    before=cursors(trainer);masks[bnk][0]=1;trainer.update_mask(masks,growth_weight=.43)
    assert cursors(trainer)==before
    net.run(5*dt,namespace={});expected=float(syn.clock.variables['t'].get_value()[0])
    last=bundle.provenance['dynamic_state_layout'][syn.name]['lastupdate'][0]
    assert trainer.neuron_state[0][last]==expected
    assert expected!=trainer.clock_tick*float(dt)


def test_migration_preserves_stochastic_sequence_and_noise_cursor():
    from test_training_brian_dynamic import network
    from brian2_rust import lower_brian_dynamic_training
    net,inp,layers,static,syn,_,_,x=network(False,True)
    syn.pre.delay=[.6,.2,.4,0]*b.ms
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,seed=371)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.execute(x[None,:3],[0],noise_sequence=11)
    before=cursors(trainer);masks=copy.deepcopy(bundle.plan['masks']);bnk=bank(bundle,syn.name);masks[bnk][1]=0
    trainer.update_mask(masks);assert cursors(trainer)==before
    masks[bnk][1]=1;trainer.update_mask(masks,growth_weight=.4);assert cursors(trainer)==before
    trainer.execute(x[None,3:6],[0],initial='carry');assert trainer.noise_sequence==11 and trainer.next_noise_sequence==12
    trainer.execute(x[None,:3],[0]);assert trainer.noise_sequence==12 and trainer.next_noise_sequence==13


def test_regrowth_restores_learned_initial_values_for_other_edge_states():
    from test_training_brian_dynamic import network
    from brian2_rust import lower_brian_dynamic_training
    net,inp,layers,static,syn,_,_,x=network(True)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=layers,trainable_synapse_parameters={syn.name:['w','apre']})
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.execute(x[None,:3],[0])
    bnk=bank(bundle,syn.name);init_bank=bank(bundle,syn.name,'apre');learned=.12345
    trainer.state['weights'][init_bank][1]=learned
    masks=copy.deepcopy(bundle.plan['masks']);masks[bnk][1]=0;trainer.update_mask(masks)
    masks[bnk][1]=1;trainer.update_mask(masks,growth_weight=.42)
    layout=bundle.provenance['dynamic_state_layout'][syn.name]
    assert trainer.neuron_state[0][layout['w'][1]]==.42
    assert trainer.neuron_state[0][layout['apre'][1]]==learned
