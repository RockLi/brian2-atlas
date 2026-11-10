"""Multiplicative SDEs, live refractory, event traces and mutable delay together."""
import ast
import copy
import os
import re
from unittest.mock import patch

import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_stochastic import normal
from test_training_delay_update import snapshot


VARIANTS=['heun_shared','heun_mixed','milstein']


def model(variant='heun_shared',kind='boolean',delayed=True,warm=0,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    x=np.array([[1,0],[0,1],[1,1],[1,0],[0,1],[1,1],[0,1],[1,0],[1,1],[0,1]],float)
    ticks,ids=np.nonzero(x)
    inp=b.SpikeGeneratorGroup(2,ids,ticks*dt,dt=dt,name='combo_input')
    nv='xi_v' if variant=='milstein' else 'xi_common';nu='xi_u' if variant=='milstein' else 'xi_common'
    ev='+.2*sigma*(1+u)*xi_extra/sqrt(ms)' if variant=='heun_mixed' else ''
    eu='+.1*sigma*(1+v)*xi_extra/sqrt(ms)' if variant=='heun_mixed' else ''
    g=b.NeuronGroup(2,f'''dv/dt=(drive-v+.15*u+.1*q)/ms+sigma*(1+v)*{nv}/sqrt(ms){ev}:1 (unless refractory)
        du/dt=-u/ms+.5*sigma*(1+u)*{nu}/sqrt(ms){eu}:1
        dr/dt=-r/(2*ms):second
        q:1
        drive:1 (constant)
        sigma:1 (constant)
        limit:1 (constant)
        shift:second (constant)''',threshold='v>.6',reset='v-=.4\nu+=.8\nr+=.4*ms',
        refractory='u>limit' if kind=='boolean' else 'r+shift',method=variant.split('_')[0],dt=dt,name='combo_group')
    g.v=[.7,.3];g.u=[.1,.2];g.r=[.75,1.2]*b.ms;g.drive=[1.1,.9];g.sigma=[.05,.07];g.limit=[.35,.45];g.shift=[.1,.15]*b.ms
    hidden=b.NeuronGroup(2,'dv/dt=-v/ms:1',threshold='v>100',reset='v=0',method='euler',dt=dt,name='combo_hidden')
    syn=b.Synapses(inp,g,'''w:1
        dap/dt=-ap/(2*ms):1 (event-driven)
        dz/dt=(-z+.2*w)/ms+rho*(1+z)*xi_syn/sqrt(ms):1 (clock-driven)
        rho:1 (constant)
        q_post=w+.2*w*w+.1*z:1 (summed)''',
        on_pre='v_post+=w+.03*ap+.02*z\nr_post+=.1*ms\nap+=.2\nw+=.01*ap+.002*z'+('\ndelay=.6*delay+.12*ms*w' if delayed else ''),
        on_post='ap+=.1\nw-=.002*u_post+.001*z'+('\ndelay=.7*delay+.06*ms*w' if delayed else ''),method=variant.split('_')[0],dt=dt,name='combo_syn')
    syn.connect(j='i');syn.w=[.12,.16];syn.ap=[.1,.2];syn.z=[.15,.25];syn.rho=[.03,.04]
    if delayed:syn.pre.delay=[.04,.44]*b.ms;syn.post.delay=[.24,.04]*b.ms
    net=b.Network(inp,hidden,g,syn)
    if warm:
        b.seed(1984);net.run(warm*dt,namespace={});x=x[warm:]
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],seed=7123,
        trainable_neuron_parameters={g.name:['drive','sigma','limit' if kind=='boolean' else 'shift']},
        trainable_synapse_parameters={syn.name:['w','rho']},**options)
    return net,g,syn,x,bundle


def run_compiled_common_noise(net,g,syn,bundle,length,start_tick=0,sequence=17):
    """Replace only randn's refill; execute the real generated Cython updater."""
    net.run(0*b.ms,namespace={})
    schedules=[]
    for updater in net.sorted_objects:
        if updater not in (g.state_updater,syn.state_updater):continue
        names=bundle.provenance['noise_names'][1] if updater is g.state_updater else sorted(syn.equations.stochastic_variables)
        domain=1 if updater is g.state_updater else bundle.provenance['synaptic_noise_domains'][syn.name]
        order=[]
        for line in updater.codeobj.code.run.splitlines():
            match=re.match(r'\s*(\w+)\s*=.*\b_randn\(',line)
            if match:order.append(names.index(match[1]))
        abstract=[]
        for statement in ast.parse(updater.abstract_code).body:
            if isinstance(statement,ast.Assign) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='randn' for n in ast.walk(statement.value)):
                abstract.append(names.index(statement.targets[0].id))
        assert order and sorted(order)==sorted(abstract)
        schedules.append((domain,order))
    assert len(schedules)==2 and len({domain for domain,_ in schedules})==2
    draws=[normal(bundle.plan['seed'],sequence,0,domain,j,t,stream) for t in range(start_tick,start_tick+length) for domain,order in schedules for j in range(2) for stream in order]
    assert len(draws)<20000
    calls=[]
    def refill(n):
        assert n==20000 and not calls;calls.append(n)
        values=np.zeros(n);values[:len(draws)]=draws;return values
    device=b.get_device();device.randn_buffer_index[:]=0
    with patch('numpy.random.randn',refill):net.run(length*.2*b.ms,namespace={})
    assert calls==[20000] and device.randn_buffer_index[0]==len(draws)
    for runner in (g.state_updater,g.thresholder['spike'],g.resetter['spike'],syn.state_updater,syn.pre,syn.post):
        assert runner.codeobj.compiled_code['run'] is not None
    device.randn_buffer_index[:]=0


def compare_brian(out,bundle,g,syn,engine):
    z=np.asarray(out['final_state'])[0];tol=3e-12 if engine=='cpu' else 4e-5
    for name in ('v','u','r','q'):
        np.testing.assert_allclose(z[bundle.provenance['neuron_state_layout'][g.name][name]],np.asarray(getattr(g,name)[:]),rtol=tol,atol=tol*1e-3,err_msg=name)
    for name,slots in bundle.provenance['dynamic_state_layout'][syn.name].items():
        np.testing.assert_allclose(z[slots],np.asarray(syn.variables[name].get_value()),rtol=tol,atol=tol*1e-3,err_msg=name)
    for path in syn._pathways:
        for name,slots in bundle.provenance['pathway_state_layout'].get(path.name,{}).items():
            np.testing.assert_allclose(z[slots],np.asarray(getattr(path,name)[:]),rtol=tol,atol=tol*1e-3)
    np.testing.assert_array_equal(z[bundle.provenance['refractory_activity_layout'][g.name]],g.not_refractory[:])
    if engine!='cpu':assert out['gpu_dispatches']>0


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('kind',['boolean','duration'])
@pytest.mark.parametrize('delayed',[False,True])
@pytest.mark.parametrize('warm',[0,2])
def test_combined_common_noise_compiled_brian(engine,variant,kind,delayed,warm,tmp_path):
    net,g,syn,x,bundle=model(variant,kind,delayed,warm=warm,backend=engine)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    mon=b.SpikeMonitor(g);net.add(mon);spikes=[];cursor=0
    for length in (3,2,len(x)-5):
        out=trainer.step(x[None,cursor:cursor+length],[0],**(dict(noise_sequence=17) if cursor==0 else dict(initial='carry')))
        run_compiled_common_noise(net,g,syn,bundle,length,cursor);cursor+=length
        compare_brian(out,bundle,g,syn,engine);spikes.extend(np.asarray(out['spikes'])[0,:,2:])
        saved=tmp_path/'combined.json';trainer.store(saved)
        restored=NativeLIFTrainer(trainer.plan,runner=RUNNER);restored.restore(saved);trainer=restored
    expected=np.zeros((len(x),2));expected[np.rint((np.asarray(mon.t/b.second)-bundle.plan['clock']['origin'])/.0002).astype(int),np.asarray(mon.i)]=1
    np.testing.assert_array_equal(spikes,expected)
