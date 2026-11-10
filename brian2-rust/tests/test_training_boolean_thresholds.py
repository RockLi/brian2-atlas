"""Exact Boolean threshold control flow and independent surrogate references."""
import copy
import os
import numpy as np
import pytest
import brian2 as b
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training,TrainingConversionError
from test_native_training import RUNNER
from test_training_linked import cython_cache
from test_training_dynamic_gpu import backend,compare
from test_training_dynamic_thresholds import model as base_model,bank
from test_training_stochastic import normal


EXPRESSIONS={
    'and':'v>theta and a>=limit',
    'or':'v>theta or a>=limit',
    'not':'not (v>theta and a>=limit)',
    'nested':'(v>theta and not (a>=limit)) or (v==theta)',
    'unequal':'v!=theta and a>=limit',
}
TREES={
    'and':('and',('gt',0),('ge',1)),
    'or':('or',('gt',0),('ge',1)),
    'not':('not',('and',('gt',0),('ge',1))),
    'nested':('or',('and',('gt',0),('not',('ge',1))),('eq',0)),
    'unequal':('and',('ne',0),('ge',1)),
}


def gate(tree,margins,surrogate,record,anchors=None,path=()):
    """Independent scalar decision tree; freeze decisions for finite differences."""
    op=tree[0]
    if op in ('gt','ge','eq','ne'):
        m=margins[tree[1]];base=m if anchors is None else anchors[path]
        record[path]=base
        hard=float({'gt':base>0,'ge':base>=0,'eq':base==0,'ne':base!=0}[op])
        if op in ('eq','ne'):return hard
        return hard+surrogate['scale']/(1+surrogate['slope']*abs(base))**2*(m-base)
    a=gate(tree[1],margins,surrogate,record,anchors,path+(0,))
    if op=='not':return 1-a
    left=a if anchors is None else anchors[path]
    record[path]=left
    if op=='and' and left==0 or op=='or' and left==1:return a
    c=gate(tree[2],margins,surrogate,record,anchors,path+(1,))
    return a*c if op=='and' else a+c-a*c


def stationary(expression,values=None,subexpression=False,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython';dt=.2*b.ms
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>1',reset='v=0',dt=dt,method='euler')
    equations='dv/dt=0/second:1\nda/dt=0/second:1\ntheta:1 (constant)\nlimit:1 (constant)'
    if subexpression:equations+='\ncondition='+expression+':boolean'
    g=b.NeuronGroup(2,equations,threshold='condition' if subexpression else expression,reset='v=v\na=a',dt=dt,method='euler')
    g.v,g.a=([.5,.7],[.3,.1]) if values is None else values;g.theta=.5;g.limit=.3
    names=[n for n in ('theta','limit') if n in expression]
    net=b.Network(inp,hidden,g)
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=[hidden,g],
        trainable_neuron_parameters={g.name:names},surrogate_slope=3.,surrogate_scale=.8,**options)
    return net,g,bundle


@pytest.mark.parametrize('kind',list(EXPRESSIONS))
@pytest.mark.parametrize('values',[([.5,.7],[.3,.1]),([.3,.7],[.1,.5])])
def test_boolean_vjp_independent(kind,values):
    _,g,bundle=stationary(EXPRESSIONS[kind],values);p=bundle.plan
    result=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients([[[0]]],[0])
    def reference(weights,initial,anchors=None):
        v=np.array(initial)[bundle.provenance['neuron_state_layout'][g.name]['v']]
        a=np.array(initial)[bundle.provenance['neuron_state_layout'][g.name]['a']]
        theta=np.array(weights[bank(bundle,g,'theta')[0]])
        limit=np.array(weights[bank(bundle,g,'limit')[0]])
        records=[{},{}];s=np.array([gate(TREES[kind],[v[j]-theta[j],a[j]-limit[j]],p['surrogate'],records[j],None if anchors is None else anchors[j]) for j in range(2)])
        z=s*p['logit_scale'];return np.log(np.exp(z-z.max()).sum())+z.max()-z[0],s,records
    loss,spikes,anchors=reference(bundle.weights,bundle.initial_state)
    assert result['loss']==pytest.approx(loss,abs=2e-14)
    np.testing.assert_array_equal(result['spikes'][0][0][1:],spikes)
    for i,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[i][j]+=1e-6;lo[i][j]-=1e-6
            fd=(reference(hi,bundle.initial_state,anchors)[0]-reference(lo,bundle.initial_state,anchors)[0])/2e-6
            assert result['gradients'][i][j]==pytest.approx(fd,rel=2e-5,abs=2e-8)
    for j in range(len(bundle.initial_state)):
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(reference(bundle.weights,hi,anchors)[0]-reference(bundle.weights,lo,anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=2e-5,abs=2e-8)


@pytest.mark.parametrize('expression',list(EXPRESSIONS.values())+['True','False','v>theta and t>=.2*ms'])
def test_boolean_matches_real_brian(expression):
    net,g,bundle=stationary(expression)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(np.zeros((1,3,1)),[0])
    monitor=b.SpikeMonitor(g);net.add(monitor);net.run(.6*b.ms,namespace={})
    expected=np.zeros((3,2));expected[np.rint(monitor.t/(.2*b.ms)).astype(int),monitor.i[:]]=1
    np.testing.assert_array_equal(np.asarray(result['spikes'][0])[:,1:],expected)


@pytest.mark.parametrize('expression',['v<theta and log(a-limit)>0','v>=theta or 1/(a-limit)>0'])
@pytest.mark.parametrize('engine',['cpu','metal'])
def test_short_circuit_skips_invalid_domains(expression,engine):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    net,g,bundle=stationary(expression,([.7,.8],[.3,.3]),backend=engine)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients([[[0]]],[0])
    expected=[0,0] if ' and ' in expression else [1,1]
    np.testing.assert_array_equal(result['spikes'][0][0][1:],expected)
    assert np.linalg.norm(result['gradients'][bank(bundle,g,'theta')[0]])>0
    np.testing.assert_array_equal(result['gradients'][bank(bundle,g,'limit')[0]],0)
    monitor=b.SpikeMonitor(g);net.add(monitor);net.run(.2*b.ms,namespace={})
    np.testing.assert_array_equal(monitor.count[:],expected)


COMPOUND='(v>theta+.1*a+.1*drive(t,i)+.05*sin(t/ms) and a>=.18) or not (t<.9*ms)'
COMPOUND_TREE=('or',('and',('gt',0),('ge',1)),('not',('gt',2)))


def evolving(noisy=False,plastic=False,refractory=False,method='euler',**options):
    net,groups,synapses,drive,x,_=base_model(method=method,noisy=noisy,refractory=refractory)
    for g in groups:g.events['spike']=COMPOUND
    if plastic:synapses[1].pre.code+='\nw+=.01*a_post'
    inp=next(g for g in net.objects if isinstance(g,b.SpikeGeneratorGroup))
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,
        trainable_neuron_parameters={g.name:['theta'] for g in groups},**options)
    return net,groups,synapses,drive,x,bundle


def evolving_oracle(bundle,groups,synapses,weights,x,noisy,plastic,anchors=None,initial=None):
    p=bundle.plan;table=np.array(weights[bundle.provenance['timed_inputs'][0]['bank']]).reshape(3,2)
    theta=np.array([weights[bank(bundle,g,'theta')[0]] for g in groups])
    w=[np.array(weights[bank(bundle,s,'w')[0]]) for s in synapses]
    v=np.array([[1.1,.7],[.9,1.15]]);a=np.array([[.2,.35],[.15,.3]]);history=[];spikes=[]
    if initial is not None:
        v=np.array([np.array(initial)[bundle.provenance['neuron_state_layout'][g.name]['v']] for g in groups])
        a=np.array([np.array(initial)[bundle.provenance['neuron_state_layout'][g.name]['a']] for g in groups])
        if plastic:w[1]=np.array(initial)[bundle.provenance['dynamic_state_layout'][synapses[1].name]['w']].copy()
    for tick,external in enumerate(x[:,0]):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:
            v,a,wlast=copy.deepcopy(anchors[tick]['start']);w[1]=wlast
        start=(v.copy(),a.copy(),w[1].copy());v=.8*v+.024*a;a=.9*a
        if noisy:v+=.08*np.sqrt(.2)*np.array([[normal(p['seed'],7,0,l,j,tick,0) for j in range(2)] for l in range(2)])
        margin=v-theta-.1*a-.1*table[min(tick//2,2)]-.05*np.sin(tick*.2)
        records=[[{},{}],[{},{}]];s=np.zeros((2,2));hard=np.zeros((2,2))
        for l in range(2):
            for j in range(2):
                base=None if anchors is None else anchors[tick]['gates'][l][j]
                values=[margin[l,j],a[l,j]-.18,.0009-tick*.0002]
                s[l,j]=gate(COMPOUND_TREE,values,p['surrogate'],records[l][j],base)
                hard[l,j]=s[l,j] if anchors is None else anchors[tick]['hard'][l,j]
        history.append(dict(start=start,gates=records,hard=hard.copy()));spikes.append(s.reshape(-1))
        v[0]+=external*w[0];a[0]+=.03*external*w[0]
        for e in range(4):
            i,j=divmod(e,2);weight=w[1][e];new_a=a[1,j]+.03*weight
            v[1,j]+=s[0,i]*weight;a[1,j]+=.03*s[0,i]*weight
            if plastic:w[1][e]+=.01*s[0,i]*new_a
        reset=hard if p['detach_reset'] else s;v-=.6*reset;a+=.15*reset
    z=np.array(spikes)[:,2:].mean(0)*p['logit_scale']
    return np.log(np.exp(z-z.max()).sum())+z.max()-z[0],np.array(spikes),v,a,w,history


@pytest.mark.parametrize('noisy,plastic,detach,window',[(False,False,False,None),(True,True,False,None),(True,True,True,3)])
def test_boolean_stochastic_plastic_vjp(noisy,plastic,detach,window):
    _,groups,synapses,_,x,bundle=evolving(noisy,plastic,detach_reset=detach,tbptt_window=window)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],**(dict(noise_sequence=7) if noisy else {}))
    loss,spikes,v,a,w,anchors=evolving_oracle(bundle,groups,synapses,bundle.weights,x,noisy,plastic)
    assert result['loss']==pytest.approx(loss,abs=2e-13)
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    np.testing.assert_allclose(result['final_membrane'][0],v.reshape(-1),atol=2e-13)
    if plastic:
        cells=bundle.provenance['dynamic_state_layout'][synapses[1].name]['w']
        np.testing.assert_allclose(np.array(result['final_state'][0])[cells],w[1],atol=2e-13)
    for i,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[i][j]+=1e-6;lo[i][j]-=1e-6
            fd=(evolving_oracle(bundle,groups,synapses,hi,x,noisy,plastic,anchors)[0]-evolving_oracle(bundle,groups,synapses,lo,x,noisy,plastic,anchors)[0])/2e-6
            assert result['gradients'][i][j]==pytest.approx(fd,rel=4e-4,abs=5e-7)
    explicit=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],initial=[bundle.initial_state],**(dict(noise_sequence=7) if noisy else {}))
    for j in range(len(bundle.initial_state)):
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(evolving_oracle(bundle,groups,synapses,bundle.weights,x,noisy,plastic,anchors,hi)[0]-evolving_oracle(bundle,groups,synapses,bundle.weights,x,noisy,plastic,anchors,lo)[0])/2e-6
        assert explicit['initial_state_gradients'][0][j]==pytest.approx(fd,rel=4e-4,abs=5e-7)


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('refractory',[False,True])
def test_boolean_actual_gpu_mpi(ranks,refractory,backend):
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=evolving(True,True,refractory,detach_reset=False,tbptt_window=3)
    cpu=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],noise_sequence=7)
    bundle.plan.update(backend=backend,mpi_ranks=ranks)
    actual=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients(x[None],[0],noise_sequence=7)
    compare(actual,cpu,backend)


@pytest.mark.parametrize('kind',list(EXPRESSIONS))
def test_boolean_gpu_boundaries_and_equality(kind,backend):
    _,_,bundle=stationary(EXPRESSIONS[kind]);p=bundle.plan
    expected=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients([[[0]]],[0])
    p['backend']=backend
    actual=NativeLIFTrainer(p,runner=RUNNER,weights=bundle.weights).gradients([[[0]]],[0])
    compare(actual,expected,backend)


@pytest.mark.parametrize('method',['euler','rk4'])
def test_boolean_plasticity_matches_real_brian(method):
    net,groups,synapses,_,x,bundle=evolving(plastic=True,refractory=method=='rk4',method=method)
    result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0])
    monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors);net.run(len(x)*.2*b.ms,namespace={})
    expected=np.zeros((len(x),4))
    for layer,(g,monitor) in enumerate(zip(groups,monitors)):
        expected[np.rint(monitor.t/(.2*b.ms)).astype(int),np.asarray(monitor.i[:])+2*layer]=1
        for name,slots in bundle.provenance['neuron_state_layout'][g.name].items():
            if name.startswith('__'):continue
            np.testing.assert_allclose(np.array(result['final_state'][0])[slots],np.asarray(getattr(g,name)[:]),rtol=4e-13,atol=4e-13)
    np.testing.assert_array_equal(result['spikes'][0],expected)
    slots=bundle.provenance['dynamic_state_layout'][synapses[1].name]['w']
    np.testing.assert_allclose(np.array(result['final_state'][0])[slots],synapses[1].w[:],rtol=4e-13,atol=4e-13)


def test_boolean_subexpression_and_chained_comparison():
    results=[]
    for expression,subexpression in [('theta-.2 < v <= theta+.2',False),('theta-.2 < v and v<=theta+.2',True)]:
        net,g,bundle=stationary(expression,subexpression=subexpression)
        result=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).gradients([[[0]]],[0]);results.append(result)
        # Brian's renderer rejects Python chains; use its equivalent explicit
        # conjunction for the real Cython reference to the chain extension.
        if not subexpression:g.events['spike']='theta-.2 < v and v<=theta+.2'
        monitor=b.SpikeMonitor(g);net.add(monitor);net.run(.2*b.ms,namespace={})
        np.testing.assert_array_equal(result['spikes'][0][0][1:],monitor.count[:])
    for key in ('spikes','final_state','gradients','initial_state_gradients'):
        for actual,expected in zip(results[0][key],results[1][key]):np.testing.assert_allclose(actual,expected,atol=2e-14)


@pytest.mark.parametrize('issue',['normal_flag','no_margin','inclusive','numeric_root','forward_reference','numeric_operand','wrong_slope','wrong_scale'])
def test_invalid_boolean_plans_fail_atomically(issue):
    _,_,bundle=stationary(EXPRESSIONS['and']);spec=bundle.plan['dynamic']
    action=next(a for a in spec['actions'] if a.get('threshold_predicate'))
    cell=action['reads'][0];writer=next(a for a in spec['actions'] if cell in a['writes'])
    program=spec['program_sets'][writer['program_set']][0]
    step=next(n for n in program if n['op']=='surrogate_step')
    if issue=='normal_flag':writer['threshold_predicate']=True
    elif issue=='no_margin':action['threshold_margin']=False
    elif issue=='inclusive':action['threshold_inclusive']=True
    elif issue=='numeric_root':program[-1]=dict(op='add',left=0,right=0)
    elif issue=='forward_reference':step['arg']=len(program)
    elif issue=='numeric_operand':program[-1]['right']=0
    elif issue=='wrong_slope':program[step['slope']]['value']=7.
    elif issue=='wrong_scale':program[step['scale']]['value']=2.
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError):trainer.step([[[0]]],[0])
    assert trainer.state==before and trainer.neuron_state is None


@pytest.mark.parametrize('expression',['v>theta and a>1*ms','not (v<theta or a>1*ms)','v>theta and 3'])
def test_invalid_boolean_units_and_types_rejected(expression):
    with pytest.raises(TrainingConversionError):stationary(expression)


@pytest.mark.parametrize('engine',['cpu','metal'])
def test_executed_invalid_branch_fails_without_state_commit(engine):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    _,_,bundle=stationary('v>theta and log(a-limit)>0',([.7,.8],[.3,.3]),backend=engine)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);before=copy.deepcopy(trainer.state)
    with pytest.raises((ValueError,RuntimeError)):trainer.step([[[0]]],[0])
    assert trainer.state==before and trainer.neuron_state is None


@pytest.mark.parametrize('engine,ranks',[('cpu',None),('cpu',2),('metal',None),('metal',2)])
def test_boolean_carry_checkpoint_and_input_update(engine,ranks,tmp_path):
    if engine=='metal' and os.environ.get('B2_TEST_GPU')!='1':pytest.skip('actual Metal required')
    if ranks is not None and os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,x,bundle=evolving(True,True,True,backend=engine,mpi_ranks=ranks)
    bundle.plan['trainable']=[False]*len(bundle.weights)
    whole=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights).evaluate(x[None],[0],noise_sequence=7)
    trainer=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);trainer.step(x[None,:2],[0],noise_sequence=7)
    path=tmp_path/'boolean.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER,weights=bundle.weights);restored.restore(path)
    tail=restored.evaluate(x[None,2:],[0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=2e-5,atol=3e-6)
    source=bundle.provenance['timed_inputs'][0]['bank'];before=copy.deepcopy(restored.neuron_state)
    restored.update_timed_input(source,[-100.]*len(bundle.weights[source]))
    assert restored.neuron_state==before and restored.clock_tick==2 and restored.noise_sequence==7
    changed=restored.evaluate(x[None,2:],[0],initial='carry')
    assert not np.array_equal(changed['spikes'],tail['spikes'])
