"""Positive/negative stride captures retain physical cell order and adjoints."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_event_captures import event_capture,readonly_event_capture
from test_training_shared_capture_views import zero_capture
from test_training_partial_overlap_vjps import oracle
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(mode,layout,readonly_first,delay,ranks,backend,zero=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0],np.array([0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(4,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.217,.319,.419,.719];g.u=[.017,.021,.023,.031]
    parent=g.variables['v'].get_value();a=parent[::2] if layout=='stride' else parent[::-1];c=(parent[::-1] if layout=='stride' else parent[::2]).view();c.flags.writeable=False
    mutable=b.Function((zero_capture if zero else event_capture)(a),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    readonly=b.Function(readonly_event_capture(c),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    code='h=f('+('v_post' if mode=='scalar' else 'h')+');h=q(h)'+(';v_post+=gain*h' if mode=='scalar' else ';u_post+=gain*h' if mode=='vectorised' else '')
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=code,dt=dt,namespace=dict(f=readonly if readonly_first else mutable,q=mutable if readonly_first else readonly))
    syn.connect(i=[0,1],j=[0,1] if mode=='array' else [0,0]);syn.h=[.113,.173];syn.gain=[.11,.19];syn.delay=delay*dt
    net=b.Network(source,hidden,g,syn);bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    cells=bundle.provenance['neuron_state_layout'][g.name]['v']
    for row in bundle.provenance['mutable_capture_layout']:
        writable=any(not alias['readonly'] for alias in row['aliases'])
        expected=cells[::2] if (layout=='stride')==writable else cells[::-1]
        assert row['cells']==expected
    x=np.zeros((1,4,2));x[0,0,:]=1;x[0,2,0]=1
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('layout',['stride','reverse'])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('ranks',[None,2])
def test_strided_views_original_carry_restore(engine,mode,layout,readonly_first,delay,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(mode,layout,readonly_first,delay,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=trainer.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);trainer.store(path);trainer=NativeLIFTrainer(trainer.plan,runner=RUNNER);trainer.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('layout',['stride','reverse'])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_strided_views_all_bank_initial_vjps(engine,mode,layout,readonly_first,delay,window,ranks):
    mpi(ranks);data=model(mode,layout,readonly_first,delay,ranks,engine);net,g,syn,dt,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=oracle(data,mode,readonly_first,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(data,mode,readonly_first,hi,window,anchors=anchors)[0]-oracle(data,mode,readonly_first,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(oracle(data,mode,readonly_first,bundle.weights,window,hi,anchors)[0]-oracle(data,mode,readonly_first,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    net.run(4*dt,namespace={})
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
