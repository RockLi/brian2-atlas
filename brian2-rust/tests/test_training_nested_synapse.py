"""Nested synaptic selectors, mapped coefficient banks and mutable shared routes."""
import copy
import os
import ast
import re
from unittest.mock import patch
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_integer_ir import engine
from test_training_delay_update import snapshot
from test_training_stochastic import normal


SOURCES=np.array([1,0,1,0]);TARGETS=np.array([0,1,1,0]);BANK_MAP=np.array([3,1,4],np.int32)


def model(mutation='both',noisy=False,delayed=False,warm=0,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,1],[1,0],[0,1],[1,1],[1,0],[0,1],[1,1],[0,1]],float);ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='nested_input')
    pool=b.NeuronGroup(5,'dv/dt=-v/ms:1\nroute:integer\ngain:1 (constant)\nh:1 (shared,constant)',
        threshold='v>100',reset='v=0',method='euler',dt=dt,name='nested_pool')
    pool.route=[2,0,1,2,0];pool.gain=[.11,.23,.37,.41,.59];pool.h=.3;pool.v=np.arange(5)*.1
    out=b.NeuronGroup(2,'dv/dt=(.7-v+.2*q)/ms'+('+.03*(q+1)*xi/sqrt(ms)' if noisy else '')+':1\nq:1',
        threshold='v>.65',reset='v-=.4',method='euler',dt=dt,name='nested_output');out.v=[.7,.4]
    change=('pick=(pick+1)%5\n' if mutation in ('pick','both') else '')+('route=(route+1)%3\n' if mutation in ('route','both') else '')
    syn=b.Synapses(inp,out,'dw/dt=(-w+.05*coef)/ms'+('+.02*(coef+1)*xi/sqrt(ms)' if noisy else '')+':1 (clock-driven)\n'
        'pick:integer\nroute:integer (linked)\nh:1 (linked)\nq_post=w+.1*coef:1 (summed)',
        on_pre=change+'v_post+=.12*coef+.1*h+w\nw+=.01*coef'+('\ndelay=.6*delay+.1*ms*coef' if delayed else ''),
        on_post='w-=.005*coef',method='euler',dt=dt,name='nested_syn')
    syn.connect(i=SOURCES,j=TARGETS);syn.w=[.12,.17,.21,.15];syn.pick=[0,1,4,2]
    syn.route=b.linked_var(pool,'route',index='pick');syn.h=b.linked_var(pool,'h')
    # Explicit Brian Variables references: pick -> mutable route -> fixed bank map -> gain.
    # This preserves the high-level linked_var restriction on composing source indices.
    syn.variables.add_array('mapped',size=3,dtype=np.int32,constant=True,read_only=True,index='route',values=BANK_MAP)
    syn.variables.add_reference('coef',pool,'gain',index='mapped')
    if delayed:syn.pre.delay=np.array([.04,.44,.24,.04])*b.ms
    net=b.Network(inp,pool,out,syn)
    if warm:
        b.seed(7123);net.run(warm*dt,namespace={});x=x[warm:]
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[pool,out],
        trainable_neuron_parameters={pool.name:['gain','h']},trainable_synapse_parameters={syn.name:['w']},**options)
    return net,pool,out,syn,x,bundle


def run_compiled_common_noise(net,out,syn,bundle,length,start_tick):
    """Feed identical draws to genuine Cython neuron and synapse updaters."""
    net.run(0*b.ms,namespace={});schedules=[]
    for updater in net.sorted_objects:
        if updater not in (out.state_updater,syn.state_updater):continue
        group=out if updater is out.state_updater else syn
        names=sorted(group.equations.stochastic_variables)
        domain=1 if group is out else bundle.provenance['synaptic_noise_domains'][syn.name]
        order=[]
        for line in updater.codeobj.code.run.splitlines():
            match=re.match(r'\s*(\w+)\s*=.*\b_randn\(',line)
            if match:order.append(names.index(match[1]))
        abstract=[]
        for statement in ast.parse(updater.abstract_code).body:
            if isinstance(statement,ast.Assign) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='randn' for n in ast.walk(statement.value)):
                abstract.append(names.index(statement.targets[0].id))
        assert order and sorted(order)==sorted(abstract)
        schedules.append((domain,len(group),order))
    assert len(schedules)==2 and len({domain for domain,_,_ in schedules})==2
    draws=[normal(bundle.plan['seed'],9,0,domain,j,t,stream) for t in range(start_tick,start_tick+length) for domain,count,order in schedules for j in range(count) for stream in order]
    assert len(draws)==length*6 and len(draws)<20000;calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n)
        values=np.zeros(n);values[:len(draws)]=draws;return values
    device=b.get_device();device.randn_buffer_index[:]=0
    with patch('numpy.random.randn',refill):net.run(length*.2*b.ms,namespace={})
    assert calls==[20000] and device.randn_buffer_index[0]==len(draws)
    device.randn_buffer_index[:]=0


@pytest.mark.parametrize('mutation',['pick','route','both'])
@pytest.mark.parametrize('delayed',[False,True])
@pytest.mark.parametrize('warm',[0,2])
@pytest.mark.parametrize('noisy',[False,True])
def test_nested_synaptic_mapping_matches_compiled_brian(engine,mutation,delayed,warm,noisy,tmp_path):
    net,pool,out,syn,x,bundle=model(mutation,noisy=noisy,delayed=delayed,warm=warm,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);mon=b.SpikeMonitor(out);net.add(mon);spikes=[];cursor=0
    for length in (3,len(x)-3):
        result=trainer.step(x[None,cursor:cursor+length],[0],**(dict(initial='carry') if cursor else dict(noise_sequence=9) if noisy else {}))
        if noisy:run_compiled_common_noise(net,out,syn,bundle,length,cursor)
        else:net.run(length*.2*b.ms,namespace={})
        cursor+=length;z=np.asarray(result['final_state'])[0];tol=3e-12 if engine=='cpu' else 4e-5
        for group in (pool,out):
            for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():
                if name.startswith('__'):continue
                actual=np.asarray(group.variables[name].get_value());np.testing.assert_allclose(z[slots],actual,rtol=tol,atol=tol*1e-3,err_msg=name)
        for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
            if name in ('route','h'):continue
            np.testing.assert_allclose(z[slots],np.asarray(syn.variables[name].get_value()),rtol=tol,atol=tol*1e-3,err_msg=name)
        if delayed:
            slots=bundle.provenance['pathway_state_layout'][syn.pre.name]['delay'];np.testing.assert_allclose(z[slots],np.asarray(syn.pre.delay[:]),rtol=tol,atol=tol*1e-3)
        assert all(r.codeobj.compiled_code['run'] is not None for r in (syn.state_updater,syn.pre,syn.post,*syn.summed_updaters.values()))
        if engine!='cpu':assert result['gpu_dispatches']>0
        spikes.extend(np.asarray(result['spikes'])[0,:,5:]);path=tmp_path/'nested-synapse.json';trainer.store(path)
        restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(path);trainer=restored
    expected=np.zeros((len(x),2));expected[np.rint((np.asarray(mon.t/b.second)-bundle.plan['clock']['origin'])/.0002).astype(int),np.asarray(mon.i)]=1
    np.testing.assert_array_equal(spikes,expected)
