"""Regular scalar-only, typed temporary, input table and optimizer boundaries."""
import copy
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_regular import model,oracle


@pytest.mark.parametrize('kind',['empty','scalar_only','zero_synapses','typed','subexpression','timed'])
def test_regular_edge_semantics(engine,kind):
    net,inp,groups,synapses,x,_=model('shared','end')
    group=groups[0]
    runner=next(o for o in net.sorted_objects if o.name=='regular_neuron')
    runner.abstract_code={'empty':'','scalar_only':'g+=.1','zero_synapses':'v+=.01',
        'typed':'v+=.01','subexpression':'v+=.01','timed':'v+=drive_table(t,i)'}[kind]
    extras=[]
    if kind=='timed':group.namespace['drive_table']=b.TimedArray(np.arange(18).reshape(6,3)*.001,dt=.2*b.ms)
    if kind in ('typed','subexpression'):
        s=b.Synapses(groups[0],groups[1],
            'h:1 (shared)\ncounter:integer\nflag:boolean\ntwice=2*h:1 (shared)\nlocal=counter+1:integer',dt=.2*b.ms,name='regular_extra')
        s.connect(i=[0,1],j=[1,2]);s.h=.2;s.counter=[2147483647,-2];s.flag=False
        code=('h+=.01\ncounter+=1\ntmp=counter*2.0\nflag=tmp>0\nv_post+=.01*int(flag)' if kind=='typed' else
              'h+=.01\ncached=twice\nh+=.03\nv_post+=.01*(cached+twice)')
        s.run_regularly(code,when='end');extras.append(s)
    if kind=='zero_synapses':
        s=b.Synapses(*groups,'h:1 (shared)',dt=.2*b.ms,name='regular_empty');s.connect(False);s.h=.1
        s.run_regularly('h+=.07',when='end');extras.append(s)
    net.add(extras)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,backend=engine)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x[None],[0])
    net.run(len(x)*.2*b.ms,namespace={})
    for obj in net.sorted_objects:
        if hasattr(obj,'codeobj') and obj.codeobj is not None:assert obj.codeobj.compiled_code['run'] is not None
    z=np.asarray(result['final_state'])[0];tol=4e-5 if engine!='cpu' else 3e-12
    for g in groups:np.testing.assert_allclose(z[bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=tol,atol=tol*1e-2)
    for s in extras:
        for name,slots in bundle.provenance['dynamic_state_layout'][s.name].items():
            np.testing.assert_allclose(z[slots],s.variables[name].get_value(),rtol=tol,atol=tol*1e-2)


def test_regular_adam_preserves_carry():
    _,_,_,_,x,bundle=model('shared','end',detach_reset=False)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights)
    state=None
    for part in (x[:3],x[3:]):
        weights=copy.deepcopy(trainer.state['weights'])
        loss,z,spikes,_=oracle(bundle,part,weights=weights,initial=state)
        result=trainer.step(part[None],[0],initial='carry' if state is not None else None)
        np.testing.assert_allclose(result['loss'],loss,rtol=1e-12)
        np.testing.assert_array_equal(result['spikes'][0],spikes)
        slots=bundle.provenance['neuron_state_layout']['regular_a']['v']+bundle.provenance['neuron_state_layout']['regular_c']['v']
        np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots],z[slots],rtol=1e-12)
        assert any(not np.array_equal(a,c) for a,c in zip(weights,trainer.state['weights']))
        state=np.asarray(result['final_state'])[0]


@pytest.mark.parametrize('linked',[False,True])
def test_regular_array_parameter_binding(engine,linked):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=dt,name='array_input')
    a=b.NeuronGroup(3,'dv/dt=-v/ms:1\ngain:1 (constant)',threshold='v>.5',reset='v-=.3',method='euler',dt=dt,name='array_a')
    c=b.NeuronGroup(3,'dv/dt=-v/ms:1'+('\npeer:1 (linked)\npick:integer' if linked else ''),threshold='v>.5',reset='v-=.3',method='euler',dt=dt,name='array_c')
    a.gain=[.2,.3,.4];a.v=[.3,.6,.7];c.v=[.2,.4,.6]
    a.run_regularly('v+=.1*gain',when='end')
    if linked:
        c.pick=[2,0,1];c.peer=b.linked_var(a,'gain',index='pick');c.run_regularly('pick=(pick+1)%3\nv+=.1*peer',when='start')
    s=b.Synapses(a,c,'w:1',on_pre='v_post+=w',dt=dt);s.connect(j='i');s.w=.1
    net=b.Network(inp,a,c,s)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[a,c],trainable_neuron_parameters={a.name:['gain']},backend=engine)
    actual=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
    net.run(4*dt,namespace={});tol=4e-5 if engine!='cpu' else 3e-12
    for g in (a,c):np.testing.assert_allclose(np.asarray(actual['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=tol,atol=tol*1e-2)
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==a.name and e['variables']==['gain'])
    assert np.any(np.asarray(actual['gradients'][bank])!=0)


def test_regular_masked_vectors_keep_shared_clock_update():
    net,inp,groups,synapses,x,bundle=model('shared','end')
    bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']=='regular_s' and e['variables']==['w'])
    bundle.plan['masks'][bank]=[0.]*4;bundle.weights[bank]=[0.]*4
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0])
    synapses[0].pre.active=False
    next(o for o in net.sorted_objects if o.name=='regular_synapse').abstract_code='h+=.01'
    net.run(len(x)*.2*b.ms,namespace={})
    np.testing.assert_allclose(result['final_membrane'][0],np.r_[groups[0].v[:],groups[1].v[:]],rtol=1e-12)
    slots=bundle.provenance['dynamic_state_layout']['regular_s']
    np.testing.assert_allclose(np.asarray(result['final_state'])[0,slots['h']],synapses[0].h[:],rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(result['final_state'])[0,slots['w']],0)
    np.testing.assert_array_equal(result['gradients'][bank],0)
