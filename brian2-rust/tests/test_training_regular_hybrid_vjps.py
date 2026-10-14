"""Independent likelihood and pathwise derivatives for mixed sampler shapes."""
import copy
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_regular_shared_rng import model,parameter,count,normal,RUNNER
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi


def reference(data,weights,initial=None,anchors=None,*,chained=False,override=None):
    _,g,runner,_,bundle,_=data;p=bundle.plan
    state=np.array(bundle.initial_state if initial is None else initial,dtype=float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:state[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u']
    domain=bundle.provenance['regular_runner_layout'][runner.name]['noise_domain']
    before=[];margins=[];hard=[];events=[];samples=[];log_probability=0.
    for tick in range(6):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
            state=anchors['before'][tick].copy()
        before.append(state.copy())
        # The array caller copies v and u at block entry. The captured v*=.8
        # is superseded by explicit v writeback; x*=.7 survives in the u result.
        rate_scalar=float(parameter(data,weights,'mean'));counts=[]
        def sample(stream,entity,rate):
            nonlocal log_probability
            assert rate>=0
            value=count(p['seed'],9,domain,entity,tick,rate,stream)
            if anchors is not None:
                value=next(c for t,s,j,r,c in anchors['samples'] if (t,s,j)==(tick,stream,entity))
            if override==(tick,stream,entity):value+=1
            samples.append((tick,stream,entity,rate,value))
            if rate:log_probability+=value*np.log(rate)-rate
            return value
        counts.append(sample(0,0,rate_scalar))
        rate_array=.7*state[u]+.1+(.5*counts[0] if chained else 0.)
        counts.extend(sample(1,j,float(rate_array[j])) for j in range(2))
        state[u]=.7*state[u]+.1*counts[0]+.03*np.array(counts[1:])
        state[v]+=.01*np.array([normal(p['seed'],9,0,domain,j,tick,2) for j in range(2)])
        state[v]+=.2*(parameter(data,weights,'gain')*state[u]+.01)
        margin=state[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());state[v]-=.5*event
    logits=np.array(events).mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,state,log_probability,dict(before=before,margins=margins,hard=hard,samples=samples)


def finite_difference(hi,lo,base_loss):
    return (hi[0]-lo[0]+base_loss*(hi[2]-lo[2]))/2e-6


@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_regular_hybrid_all_bank_initial_likelihood_vjps(engine,window,ranks):
    mpi(ranks);data=model(engine,ranks,'array','hybrid',True,False,window=window)
    bundle=data[4];x=data[5];base=reference(data,bundle.weights);loss,state,logp,anchors=base
    trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    out=trainer.gradients(x,[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    layout=bundle.provenance['neuron_state_layout'][data[1].name]
    cells=sorted(set(layout['v']+layout['u']+layout['z']))
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],state[cells],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.asarray(out['spikes'])[0,:,1:],anchors['hard'])
    assert len(anchors['samples'])==18
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights)
            hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            expected=finite_difference(reference(data,hi,anchors=anchors),reference(data,lo,anchors=anchors),loss)
            assert out['gradients'][bank][index]==pytest.approx(expected,rel=3e-3,abs=2e-5),(bank,index)
    initial=np.array(bundle.initial_state,dtype=float)
    explicit=trainer.gradients(x,[0],initial=initial[None],noise_sequence=9)
    loss,_,_,anchors=reference(data,bundle.weights,initial=initial)
    nonzero_score=[]
    for index in range(len(initial)):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic']['integer_states']:
            assert explicit['initial_state_gradients'][0][index]==0
            continue
        hi=initial.copy();lo=initial.copy();hi[index]+=1e-6;lo[index]-=1e-6
        upper=reference(data,bundle.weights,initial=hi,anchors=anchors)
        lower=reference(data,bundle.weights,initial=lo,anchors=anchors)
        expected=finite_difference(upper,lower,loss)
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(expected,rel=3e-3,abs=2e-5),index
        nonzero_score.append(abs(loss*(upper[2]-lower[2])/2e-6))
    assert max(nonzero_score)>1e-4,'state-dependent likelihood gradient must be exercised'


def chained_callback(array):
    def f(x,mean,spread):
        captured=array
        captured*=.8
        x*=.7
        scalar=np.random.poisson(mean)
        vector=np.random.poisson(x+.1+.5*scalar)
        return x+.1*scalar+.03*vector
    return f


def chained_model(backend,ranks,window):
    import brian2 as b
    from brian2_rust import lower_brian_dynamic_training
    data=list(model(backend,ranks,'array','hybrid',True,False,window=window))
    net,g,runner,dt,old,x=data
    g.namespace['f']=b.Function(chained_callback(g.variables['v'].get_value()),arg_units=[1,1,1],return_unit=1,stateless=False)
    inp=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
    data[4]=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],backend=backend,mpi_ranks=ranks,tbptt_window=window,detach_reset=False,trainable_neuron_parameters={g.name:['mean','spread','gain']})
    return tuple(data)


@pytest.mark.parametrize('window',[None,2])
def test_regular_chained_zero_boundary_resamples_downstream(engine,window):
    mpi(2);data=chained_model(engine,2,window);bundle=data[4]
    row=next(row for row in bundle.provenance['bindings'] if 'mean' in row['variables'])
    mean=(row['bank'],row['variables'].index('mean'));bundle.weights[mean[0]][mean[1]]=0.
    cell=bundle.provenance['neuron_state_layout'][data[1].name]['v'][0]
    for voltage in np.arange(.35,1.0,.005):
        bundle.initial_state[cell]=float(voltage);bundle.plan['dynamic']['initial'][cell]=float(voltage)
        loss,state,_,anchors=reference(data,bundle.weights,chained=True)
        alternates=[reference(data,bundle.weights,chained=True,override=(tick,0,0)) for tick in range(6)]
        weak=sum(alt[0]-loss for alt in alternates)
        if abs(weak)>1e-5 and min(abs(m[0]) for m in anchors['margins'])>1e-4:break
    else:raise AssertionError('zero boundary must affect loss')
    assert any(c!=bc for alt in alternates for (_,s,_,_,c),(_,_,_,_,bc) in zip(alt[3]['samples'],anchors['samples']) if s==1),'downstream draws must change in this fixture'
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(data[5],[0],noise_sequence=9)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    layout=bundle.provenance['neuron_state_layout'][data[1].name];cells=sorted(set(layout['v']+layout['u']+layout['z']))
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],state[cells],rtol=8e-5,atol=8e-6)
    assert out['gradients'][mean[0]][mean[1]]==pytest.approx(weak,rel=3e-3,abs=2e-5)
    for bank,values in enumerate(bundle.weights):
        for index in range(len(values)):
            if (bank,index)==mean:continue
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            expected=finite_difference(reference(data,hi,anchors=anchors,chained=True),reference(data,lo,anchors=anchors,chained=True),loss)
            assert out['gradients'][bank][index]==pytest.approx(expected,rel=3e-3,abs=2e-5),(bank,index)
