"""Training clock cursors against real Brian runs, restore and transactions."""
import copy
import hashlib
import json
import os
import subprocess
import sys

import brian2 as b
import numpy as np
import pytest

from brian2_rust import NativeLIFTrainer
from test_native_training import RUNNER
from test_training_clock_itinerary import training_model
from test_training_integer_ir import engine, model as integer_model
from test_training_dynamic_clocks import cython_cache


def trainer_state(t):
    return copy.deepcopy([t.state,t.neuron_state,t.clock_tick,t.clock_state,
                          t.noise_sequence,t.next_noise_sequence,t.elapsed_ticks,t.last_result])


@pytest.mark.parametrize('dts',[(.0002,.0003,.0001),(.0002,.00015,.00033),(.00040003,.0004,.0002)])
@pytest.mark.parametrize('warm',[0.,.00065,.00080006001])
def test_committed_cursors_match_real_brian_runs(engine,dts,warm):
    net,groups,bundle=training_model(warm,engine,dts)
    schedule=bundle.plan['dynamic']['clocks']
    clocks=[next(c for c in {o.clock for o in net.sorted_objects} if float(c.dt_)==dt) for dt in schedule['dts']]
    calls=[0]*len(clocks)
    def callback(i):
        def record():calls[i]+=1
        return record
    for i,c in enumerate(clocks):net.add(b.NetworkOperation(callback(i),clock=c,when='start',name=f'checkpoint_count_{i}'))
    schedule['order']=[clocks.index(c) for c in {o.clock for o in net.sorted_objects}]
    bundle.plan['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    for k in range(3):
        previous=list(calls)
        result=t.step(np.zeros((1,2,1)),[0],initial='carry' if k else None)
        end=bundle.plan['clock']['origin']+(k+1)*2*dts[0]
        net.run((end-float(net.t))*b.second,namespace={})
        saved=result['clock_state']
        assert saved['calls']==calls and saved['initial_calls']==previous
        assert saved['ticks']==[int(c.variables['timestep'].get_value()[0]) for c in clocks]
        assert saved['next_tick']==t.clock_tick==(k+1)*2
        assert t.clock_state==saved
        for g in groups:
            for name,indices in bundle.provenance['neuron_state_layout'][g.name].items():
                np.testing.assert_allclose(np.array(result['final_state'])[0,indices],g.variables[name].get_value(),rtol=5e-5,atol=3e-7)
        if engine!='cpu':assert result['gpu_dispatches']>0
    result['clock_state']['calls'][0]=999
    assert t.clock_state['calls'][0]==6


@pytest.mark.parametrize('ranks',[None,2,8])
def test_noise_carry_readonly_and_checkpoint(engine,ranks,tmp_path):
    if ranks and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    p,w,x=integer_model(noisy=True,engine=engine,ranks=ranks)
    p['dynamic']['clocks']=dict(start=0.,dts=[.0002,.00015,.00033],epsilon=1e-4,order=[0,1,2])
    p['trainable']=[False]*len(w)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER)
    whole=t.gradients(x,[0],noise_sequence=17)
    head=t.step(x[:,:2],[0],noise_sequence=17)
    before=trainer_state(t)
    tail=t.gradients(x[:,2:],[0],initial='carry')
    assert trainer_state(t)[:-1]==before[:-1]
    manual=t.gradients(x[:,2:],[0],initial=head['final_state'],start_tick=2,noise_sequence=17,clock_state=head['clock_state'])
    assert manual==tail
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=2e-6,atol=2e-7)
    np.testing.assert_array_equal(tail['spikes'],np.asarray(whole['spikes'])[:,2:])
    path=tmp_path/'checkpoint.json';t.store(path)
    fresh=NativeLIFTrainer(p,weights=w,runner=RUNNER);fresh.restore(path)
    restored=fresh.gradients(x[:,2:],[0],initial='carry')
    assert restored==tail and fresh.clock_state==head['clock_state']
    if engine!='cpu':assert restored['gpu_dispatches']>0


def bad_cursor(saved,issue):
    saved=copy.deepcopy(saved)
    if issue=='tick':saved['next_tick']+=1
    elif issue=='calls':saved['calls'][0]+=1
    elif issue=='initial_calls':saved['initial_calls'][0]+=1
    elif issue=='ticks':saved['ticks'][1]+=1
    elif issue=='visits':saved['visits']+=1
    elif issue=='start':saved['start']=-1
    elif issue=='width':saved['calls'].pop()
    elif issue=='overflow':saved['visits']=10_000_001
    elif issue=='extra':saved['unrecognized']=1
    elif issue=='type':saved['ticks'][0]=True
    return saved


@pytest.mark.parametrize('issue',['tick','calls','initial_calls','ticks','visits','start','width','overflow','extra','type'])
def test_invalid_cursor_and_resigned_checkpoint_do_not_commit(issue,tmp_path):
    _,_,bundle=training_model(dts=(.0002,.0003,.0001))
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    t.step(np.zeros((1,2,1)),[0]);before=trainer_state(t)
    corrupted=bad_cursor(t.clock_state,issue)
    with pytest.raises(ValueError):t.step(np.zeros((1,2,1)),[0],initial=t.neuron_state,start_tick=2,clock_state=corrupted)
    assert trainer_state(t)==before
    path=tmp_path/'bad.json';t.store(path)
    envelope=json.loads(path.read_text());payload=json.loads(envelope['payload']);payload['clock_state']=corrupted
    envelope['payload']=json.dumps(payload);envelope['sha256']=hashlib.sha256(envelope['payload'].encode()).hexdigest();path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError):t.restore(path)
    assert trainer_state(t)==before


def test_new_python_process_restores_run_clock_and_rng(tmp_path):
    p,w,x=integer_model(noisy=True)
    p['dynamic']['clocks']=dict(start=0.,dts=[.0002,.00015,.00033],epsilon=1e-4,order=[0,1,2])
    p['trainable']=[False]*len(w)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(x[:,:2],[0],noise_sequence=23)
    path=tmp_path/'saved.json';t.store(path)
    expected=t.gradients(x[:,2:],[0],initial='carry')
    script="""import json,sys
from brian2_rust import NativeLIFTrainer
path,runner=sys.argv[1:];p=json.loads(json.loads(open(path).read())['payload'])
t=NativeLIFTrainer(p['plan'],runner=runner);t.restore(path)
print(json.dumps(t.gradients([[[0.],[0.]]],[0],initial='carry')))
"""
    result=subprocess.run([sys.executable,'-c',script,str(path),str(RUNNER)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==expected


def test_legacy_checkpoint_without_cursor_and_unsupported_plan(tmp_path):
    p,w,x=integer_model(noisy=True)
    t=NativeLIFTrainer(p,weights=w,runner=RUNNER);t.step(x,[0],noise_sequence=11)
    assert t.clock_state is None
    path=tmp_path/'legacy.json';t.store(path);t.restore(path)
    with pytest.raises(ValueError,match='clock_state requires'):
        t.step(x,[0],clock_state=dict(next_tick=0,start=0.,ticks=[0],calls=[0],initial_calls=[0],visits=0))


def test_failed_carry_does_not_commit_any_cursor():
    _,_,bundle=training_model(dts=(.0002,.00099995,.00049996))
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    t.step(np.zeros((1,2,1)),[0]);before=trainer_state(t)
    with pytest.raises(ValueError,match='interval end'):t.step(np.zeros((1,3,1)),[0],initial='carry')
    assert trainer_state(t)==before


def test_checkpoint_cannot_silently_drop_dynamic_clock_history(tmp_path):
    _,_,bundle=training_model(dts=(.0002,.0003,.0001))
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    t.step(np.zeros((1,2,1)),[0]);before=trainer_state(t)
    path=tmp_path/'missing.json';t.store(path)
    envelope=json.loads(path.read_text());payload=json.loads(envelope['payload']);payload.pop('clock_state')
    envelope['payload']=json.dumps(payload);envelope['sha256']=hashlib.sha256(envelope['payload'].encode()).hexdigest();path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError,match='missing.*clock state'):t.restore(path)
    assert trainer_state(t)==before


def test_boundary_updates_preserve_cursor_and_use_pending_clock_times():
    from test_training_dynamic_clocks import model
    _,_,_,syn,x,_,bundle=model(.3,.3)
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);t.step(x[None,:2],[0])
    saved=copy.deepcopy(t.clock_state)
    bank=next(v['bank'] for v in bundle.provenance['bindings'] if v['object']==syn.name and v['variables']==['w'])
    masks=copy.deepcopy(t.plan['masks']);masks[bank]=[0.]*len(masks[bank]);t.update_mask(masks)
    masks[bank]=[1.]*len(masks[bank]);t.update_mask(masks,growth_weight=.25)
    assert t.clock_state==saved
    for cell in t.plan['dynamic']['migration']['cells']:
        restart=cell['restart']
        if restart['kind']=='time' and any(owner[0]==bank for owner in cell['owners']):
            k=restart['clock'];assert t.neuron_state[0][cell['index']]==saved['ticks'][k]*t.plan['dynamic']['clocks']['dts'][k]
    paths=t.plan['dynamic']['delay_layout']['paths']
    t.update_delays({paths[0]['name']:.0004})
    assert t.clock_state==saved
    from test_training_timed_inputs import model as timed_model
    *_,inputs,bundle=timed_model(syn_dt=.4)
    t=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER);t.step(inputs[None,:2],[0])
    saved=copy.deepcopy(t.clock_state);bank=bundle.provenance['timed_inputs'][0]['bank']
    t.update_timed_input(bank,[.2]*len(t.state['weights'][bank]))
    assert t.clock_state==saved
