"""Shared capture permissions checked against generated NumPy and finite differences."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2.codegen.translation import make_statements
from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from brian2_rust.training_brian import _INTEGRATORS
from test_training_event_captures import event_capture, readonly_event_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(where, method, readonly_first, ranks, window, backend):
    b.set_device('runtime'); b.start_scope(); b.prefs.codegen.target='numpy'
    dt=.2*b.ms; array=np.array([.23,.31]); view=array.view(); view.flags.writeable=False
    mutable=b.Function(event_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    readonly=b.Function(readonly_event_capture(view),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    namespace=dict(f=readonly if readonly_first else mutable,q=mutable if readonly_first else readonly)
    source=b.SpikeGeneratorGroup(1,[],np.array([])*b.second,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt='+('gain*q(f(v))/ms' if where=='integrator' else '0/second')+':1\ngain:1 (constant)',
        threshold='v>'+('q(f(v+.1))' if where=='threshold' else '.5'),
        reset='v=q(f(v))' if where=='reset' else 'v-=.5',method=method,dt=dt,namespace=namespace)
    g.v=[.173,.719];g.gain=[.137,.223];net=b.Network(source,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,
        mpi_ranks=ranks,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['gain']})
    variables={**b.core.functions.DEFAULT_FUNCTIONS,**g.variables,**g.resolve_all(g.equations.identifiers-set(g.variables),run_namespace={}),**g.namespace}
    variables={key:variables[key] for key in sorted(variables)}
    scalar,vector=make_statements(_INTEGRATORS[method](g.equations,variables=variables),variables,np.float64,optimise=True)
    gen=NumpyCodeGenerator(variables,g.variables.indices,g,{'_idx'},NumpyCodeObject,g.name,'stateupdate',allows_scalar_write=True)
    code=compile('\n'.join([*gen.translate_one_statement_sequence(scalar,scalar=True),*gen.translate_one_statement_sequence(vector)]),'<NumPy shared-view integrator>','exec')
    return net,g,array,dt,bundle,variables,gen,code,where,readonly_first


def oracle(data, weights, initial=None, anchors=None):
    _,g,_,dt,bundle,variables,gen,code,where,readonly_first=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];capture=bundle.provenance['mutable_capture_layout'][0]['cells']
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==g.name and 'gain' in row['variables'])
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
            z=anchors['before'][tick].copy()
        before.append(z.copy());captured=z[capture].copy()
        def mutable(x):
            captured[:]*=.8;x*=.7;return x
        def readonly(x):
            assert np.all(captured!=0);x*=.7;return x
        f,q=(readonly,mutable) if readonly_first else (mutable,readonly)
        env=dict(f=f,q=q,sqrt=np.sqrt,_numpy=np,_vectorisation_idx=np.arange(2))
        for name,var in variables.items():
            if not isinstance(var,b.core.variables.ArrayVariable):
                if hasattr(var,'get_value') and not isinstance(var,b.core.variables.AuxiliaryVariable):env[name]=np.asarray(var.get_value()).reshape(-1)[0]
                continue
            key=gen.get_array_name(var)
            if key in env:continue
            value=z[v] if var is g.variables['v'] else weights[bank] if var is g.variables['gain'] else [tick*float(dt)] if name=='t' else var.get_value()
            env[key]=np.asarray(value).copy()
        exec(code,env);z[v]=env[gen.get_array_name(g.variables['v'])]
        threshold=q(f(z[v]+.1)) if where=='threshold' else .5
        margin=z[v]-threshold;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if g.name in bundle.provenance.get('threshold_margin_layout',{}):z[bundle.provenance['threshold_margin_layout'][g.name]]=margin
        if anchors is not None:
            old=anchors['margins'][tick]
            event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        if where=='reset':
            # NumPy calls even on an empty advanced-index selection. The raw
            # closure update is unconditional; selected voltage writeback is gated.
            reset=q(f(z[v].copy()));z[v]+=event*(reset-z[v])
        else:z[v]-=.5*event
        z[capture]=captured
    logits=np.array(spikes).mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,np.array(spikes),dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('where,method',[('integrator','euler'),('integrator','rk2'),('integrator','rk4'),('threshold','euler'),('reset','euler')])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_neuron_shared_views_all_bank_initial_vjps(engine,where,method,readonly_first,window,ranks):
    mpi(ranks);data=model(where,method,readonly_first,ranks,window,engine);net,g,array,dt,bundle,*_=data
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,4,1)),[0])
    loss,z,spikes,anchors=oracle(data,bundle.weights)
    # Native reset caches are implementation scratch, overwritten before use.
    # Compare actual Brian storage; all continuous scratch initial VJPs are
    # still checked below, independently of their final cached values.
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['mutable_capture_layout'][0]['cells']]
    cells+=bundle.provenance.get('threshold_margin_layout',{}).get(g.name,[])
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],spikes)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(data,hi,anchors=anchors)[0]-oracle(data,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(oracle(data,bundle.weights,hi,anchors)[0]-oracle(data,bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    net.run(4*dt,namespace={})
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['neuron_state_layout'][g.name]['v']],g.v[:],rtol=8e-5,atol=8e-6)
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,bundle.provenance['mutable_capture_layout'][0]['cells']],array,rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
