"""Returned capture aliases retain physical ownership through caller writes."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def return_capture(array):
    def f(x):
        x*=.8
        return array
    return f


CODES={'alias':'tmp=f(v)\ntmp*=.8\nv+=tmp',
       'copy':'tmp=f(v)+.1\ntmp*=.8\nv+=tmp',
       'rebind':'tmp=f(v)\nsaved=tmp\ntmp=tmp+.1\nsaved*=.8\nv+=tmp',
       'two_calls':'tmp=f(v)\ntmp*=.8\nother=f(v)\nother*=.8\nv+=tmp+other'}


def model(kind,synaptic,cross,ranks,window,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(1,[0,0],np.array([0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(2,'dv/dt=0/second:1\ngain:1 (constant)',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=0/second:1'+('' if synaptic else '\ngain:1 (constant)'),threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.173,.719];objects=[source,hidden,g]
    if synaptic:
        owner=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre='v_post+=h',dt=dt);owner.connect(i=[0,0],j=[0,1]);owner.h=[.137,.223];objects.append(owner)
    else:owner=g
    physical=hidden if cross else owner;physical.gain=[.113,.217];array=physical.variables['gain'].get_value();f=b.Function(return_capture(array),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    owner.namespace['f']=f;owner.run_regularly(CODES[kind].replace('v','h') if synaptic else CODES[kind],when='groups',order=-1)
    net=b.Network(*objects);options=dict(trainable_neuron_parameters={physical.name:['gain']} if cross or not synaptic else {},trainable_synapse_parameters={owner.name:['h',*(['gain'] if not cross else [])]} if synaptic else {})
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,**options)
    promoted=bundle.provenance['mutable_constant_layout'][physical.name];assert ('gain' in promoted)==(kind!='copy')
    x=np.zeros((1,4,1));x[0,[0,2],0]=1
    return net,hidden,g,owner,physical,array,dt,bundle,x


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('synaptic',[False,True])
@pytest.mark.parametrize('cross',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_returned_capture_original_restore(engine,kind,synaptic,cross,ranks,tmp_path):
    mpi(ranks);net,hidden,g,owner,physical,array,dt,bundle,x=model(kind,synaptic,cross,ranks,None,engine)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).evaluate(x,[0]);p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);first=t.step(x[:,:2],[0]);path=tmp_path/'returned-capture';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path);last=t.step(x[:,2:],[0],initial='carry')
    np.testing.assert_allclose(last['final_state'],out['final_state'],rtol=8e-5,atol=8e-6);np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),out['spikes'])
    monitor=b.SpikeMonitor(g);net.add(monitor);np.testing.assert_array_equal(array,[.113,.217]);net.run(4*dt,namespace={})
    expected=np.zeros((4,2));expected[np.rint(monitor.t/dt).astype(int),np.asarray(monitor.i)]=1;np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,2:],expected)
    for obj,key in [(hidden,'neuron_state_layout'),(g,'neuron_state_layout')]+([(owner,'dynamic_state_layout')] if synaptic else []):
        for name,cells in bundle.provenance[key][obj.name].items():
            if name.startswith('__'):continue
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)


def reference(data,kind,synaptic,cross,weights,initial=None,anchors=None):
    net,hidden,g,owner,physical,array,dt,bundle,x=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for index,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[index]=weights[ref[0]][ref[1]]
    field='h' if synaptic else 'v';key='dynamic_state_layout' if synaptic else 'neuron_state_layout';live=bundle.provenance[key][owner.name][field];v=bundle.provenance['neuron_state_layout'][g.name]['v']
    gain=bundle.provenance['mutable_constant_layout'][physical.name].get('gain');bank=next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==physical.name and 'gain' in e['variables']);before=[];margins=[];hard=[];spikes=[]
    from brian2.codegen.translation import make_statements
    from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
    from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
    variables={**b.core.functions.DEFAULT_FUNCTIONS,**owner.variables,**owner.namespace};variables={name:variables[name] for name in sorted(variables)}
    scalar,vector=make_statements(CODES[kind].replace('v','h') if synaptic else CODES[kind],variables,np.float64,optimise=True)
    gen=NumpyCodeGenerator(variables,owner.variables.indices,owner,{'_idx'},NumpyCodeObject,owner.name,'stateupdate',allows_scalar_write=True)
    code=compile('\n'.join([*gen.translate_one_statement_sequence(scalar,scalar=True),*gen.translate_one_statement_sequence(vector)]),'<actual original returned-capture NumPy block>','exec')

    for tick in range(4):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());captured=np.array(weights[bank]) if gain is None else z[gain].copy();value=z[live].copy();f=return_capture(captured)
        env=dict(f=f,_numpy=np,_vectorisation_idx=np.arange(2))
        for name,var in variables.items():
            if not isinstance(var,b.core.variables.ArrayVariable):
                if hasattr(var,'get_value') and not isinstance(var,b.core.variables.AuxiliaryVariable):env[name]=np.asarray(var.get_value()).reshape(-1)[0]
                continue
            key=gen.get_array_name(var)
            if key not in env:env[key]=value.copy() if var is owner.variables[field] else np.asarray(var.get_value()).copy()
        exec(code,env);value=env[gen.get_array_name(owner.variables[field])]

        z[live]=value
        if gain is not None:z[gain]=captured
        margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            base=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(base))**2*(margin-base)
        spikes.append(event.copy())
        if synaptic and tick in (0,2):z[v]+=z[live]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('synaptic',[False,True])
@pytest.mark.parametrize('cross',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
@pytest.mark.parametrize('window',[None,2])
def test_returned_capture_all_bank_and_initial_vjps(engine,kind,synaptic,cross,ranks,window):
    mpi(ranks);data=model(kind,synaptic,cross,ranks,window,engine);bundle=data[7];x=data[8]
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,kind,synaptic,cross,bundle.weights)
    for layout in [bundle.provenance['neuron_state_layout'][data[1].name],bundle.provenance['neuron_state_layout'][data[2].name]]+([bundle.provenance['dynamic_state_layout'][data[3].name]] if synaptic else []):
        for name,cells in layout.items():
            if not name.startswith('__'):np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(reference(data,kind,synaptic,cross,hi,anchors=anchors)[0]-reference(data,kind,synaptic,cross,lo,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,j)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j] or j in bundle.plan['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(data,kind,synaptic,cross,bundle.weights,hi,anchors)[0]-reference(data,kind,synaptic,cross,bundle.weights,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][j]==pytest.approx(fd,rel=1e-3,abs=1e-5),j
