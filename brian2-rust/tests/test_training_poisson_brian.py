"""Brian Poisson conversion and actual Cython replay with matched uniforms."""
import math
from unittest.mock import patch
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_regular import model as regular_model
from test_training_poisson_core import mix,MASK,uniform


def uniforms(seed,sequence,domain,entity,tick,stream,rate):
    key=mix(seed^0x4232504f49533031)
    for field in (sequence,0,domain,entity,0,tick,stream):key=mix(key^mix((field+0x9e3779b97f4a7c15)&MASK))
    total=0.;out=[]
    for draw in range(10000):
        u=uniform(key,draw);out.append(1-u);total-=math.log1p(-u)
        if total>=rate:return out
    raise AssertionError('reference draw budget exceeded')


@pytest.mark.parametrize('warm',[0,2])
def test_regular_and_synaptic_poisson_match_actual_cython(engine,warm):
    tolerance=4e-5 if engine!='cpu' else 3e-12
    net,inp,groups,synapses,x,_=regular_model('shared','end',warm=warm)
    runners={obj.name:obj for obj in net.sorted_objects if obj.name in ('regular_neuron','regular_synapse')}
    runners['regular_neuron'].abstract_code='g+=.01*poisson(.7)\nk=1.0*poisson(1.3)\nv+=.03*k'
    runners['regular_synapse'].abstract_code='k=1.0*poisson(2.1)\nw+=.02*k\nv_post+=.01*w'
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    net.run(0*b.ms,namespace={});ordered=[o for o in net.sorted_objects if o.name in runners]
    for obj in ordered:assert obj.codeobj.compiled_code['run'] is not None
    cursor=0
    for part in (x[:2],x[2:]):
        result=trainer.step(part[None],[0],**({'initial':'carry'} if cursor else {'noise_sequence':9}))
        draws=[]
        for tick in range(cursor,cursor+len(part)):
            for obj in ordered:
                domain=bundle.provenance['regular_runner_layout'][obj.name]['noise_domain']
                if obj.name=='regular_neuron':
                    draws+=uniforms(bundle.plan['seed'],9,domain,0,tick,0,.7)
                    for j in range(3):draws+=uniforms(bundle.plan['seed'],9,domain,j,tick,1,1.3)
                else:
                    for j in range(4):draws+=uniforms(bundle.plan['seed'],9,domain,j,tick,0,2.1)
        calls=[];device=b.get_device();device.rand_buffer_index[:]=0
        def refill(n):
            assert n==20000 and not calls;calls.append(n)
            out=np.full(n,.5);out[:len(draws)]=draws;return out
        with patch('numpy.random.rand',refill):net.run(len(part)*.2*b.ms,namespace={})
        assert calls==[20000] and device.rand_buffer_index[0]==len(draws)
        device.rand_buffer_index[:]=0
        state=np.asarray(result['final_state'])[0]
        for group in groups:
            for name,slots in bundle.provenance['neuron_state_layout'][group.name].items():
                if name.startswith('__'):continue
                actual=state[slots];expected=np.broadcast_to(group.variables[name].get_value(),(len(group),))
                if name=='peer':
                    expected=np.asarray(groups[0].v)[np.asarray(group.pick)]
                    actual=[]
                    for descriptor in bundle.provenance['runtime_index_layout'][group.name][name]:
                        index=int(state[descriptor['index']])
                        for table in descriptor['tables'][:-1]:index=int(state[table[index]])
                        actual.append(state[descriptor['tables'][-1][index]])
                np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance*.1,err_msg=group.name+'.'+name)
        for syn in synapses:
            for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
                np.testing.assert_allclose(state[slots],syn.variables[name].get_value(),rtol=tolerance,atol=tolerance*.1)
        cursor+=len(part)


@pytest.mark.parametrize('where',['ode','threshold','reset','cached','pre','post','synaptic_ode'])
def test_brian_code_positions_accept_and_execute(engine,where):
    b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1,[0,0],[0.,.4]*b.ms,dt=dt,name='poisson_input')
    hidden=b.NeuronGroup(1,'dv/dt=0*Hz:1',threshold='v>1',reset='v=0',method='euler',dt=dt,name='poisson_hidden')
    equations='dv/dt=(-v+.2)/ms:1\nrate:1 (constant)'
    if where=='ode':equations='dv/dt=(-v+.2*poisson(rate))/ms:1\nrate:1 (constant)'
    if where=='cached':equations='dv/dt=(-v+.2*sample)/ms:1\nsample=poisson(rate):1 (constant over dt)\nrate:1 (constant)'
    threshold='v>.1*poisson(rate)' if where=='threshold' else 'v>-.5'
    reset='v=.1*poisson(rate)' if where=='reset' else 'v-=.1'
    output=b.NeuronGroup(2,equations,threshold=threshold,reset=reset,method='euler',dt=dt,name='poisson_output');output.v=[.3,.5];output.rate=[1.2,1.8]
    syn_eq='w:1\nsyn_rate:1 (constant)'
    if where=='synaptic_ode':syn_eq='dw/dt=(-w+.1*poisson(syn_rate))/ms:1 (clock-driven)\nsyn_rate:1 (constant)'
    pre='w+=.1*poisson(syn_rate)\nv_post+=w' if where=='pre' else 'v_post+=w'
    post='w+=.1*poisson(syn_rate)' if where=='post' else None
    syn=b.Synapses(inp,output,syn_eq,on_pre=pre,on_post=post,method='euler',dt=dt,name='poisson_synapse');syn.connect();syn.w=.1;syn.syn_rate=1.4
    net=b.Network(inp,hidden,output,syn)
    net.run(0*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,output],backend=engine,
        trainable_neuron_parameters={output.name:['rate']},trainable_synapse_parameters={syn.name:['syn_rate']})
    assert any(node['op']=='poisson' for programs in bundle.plan['dynamic']['program_sets'] for program in programs for node in program)
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    result=trainer.gradients(np.array([[[1.],[0.],[1.]]]),[0],noise_sequence=9)
    assert all(np.all(np.isfinite(row)) for row in result['gradients'])
    assert np.all(np.isfinite(result['final_state']))
    assert (result['gpu_dispatches']>0)==(engine!='cpu')
