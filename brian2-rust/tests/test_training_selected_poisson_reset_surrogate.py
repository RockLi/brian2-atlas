"""Implicit compact Poisson reset: real observations vs cached candidates."""
import numpy as np
import pytest
import test_training_neuron_whole_reset_rng as reset
from test_training_integer_ir import engine


def implicit_poisson_factory(array):
    def f(x,k,mean,spread):
        a=array
        a*=.8
        return x+.1*np.random.poisson(x+mean)
    return f


def constant_array_poisson_factory(array):
    def f(x,k,mean,spread):
        a=array
        a*=.8
        return x+.1*np.random.poisson(0.*x+mean)
    return f


def reference(data,kind,weights,initial=None,anchors=None,override=None,constant_rate=False,batch=0,label=0):
    _,storage,g,_,bundle=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'];a=layout[storage.name]['a'];v=layout[g.name]['v'];counter=layout[g.name].get('__refractory_ticks',[])
    def parameter(name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and name in row['variables'])
        return weights[row['bank']][row['variables'].index(name)] if g.variables[name].scalar else np.asarray(weights[row['bank']])
    stream=bundle.provenance['whole_neuron_reset_layout'][g.name]['draws'][0]['stream'];before=[];margins=[];hard=[];samples=[];events=[];candidate_counts=[]
    for tick in range(6):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());active=z[counter]==0 if counter else np.ones(2,dtype=bool);z[v]+=.2*parameter('gain')*active
        if counter:z[counter]=np.maximum(z[counter]-1,0)
        margin=z[v]-.5;event=((margin>0)&active).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:event=anchors['hard'][tick]+active*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        events.append(event.copy());z[a]*=.8;selection=hard[-1] if anchors is None else anchors['hard'][tick];mean=parameter('mean');counts=[]
        for j in range(2):
            q=int(selection[:j].sum());rate=mean if constant_rate else z[v][j]+mean
            value=int(anchors['candidate_counts'][tick][j]) if anchors is not None else batch_count(p['seed'],9,batch,1,q,tick,float(rate),stream)
            if selection[j]:
                if override==(tick,q):value+=1
                samples.append((tick,q,float(rate),value))
            counts.append(value)
        candidate_counts.append(np.array(counts));reset_event=hard[-1] if p['detach_reset'] else event
        z[v]+=reset_event*(.2*z[v]+.02*np.array(counts)-.5)
        if counter:
            control=hard[-1] if anchors is None else anchors['hard'][tick]
            z[counter]=np.where(control.astype(bool),max(p['refractory'][1]['steps']-1,0),z[counter])
    logits=np.array(events).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[label]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples,events=events,candidate_counts=candidate_counts)


@pytest.mark.parametrize('selection',['empty','partial','full'])
def test_selected_poisson_surrogate_original_restore(engine,selection,monkeypatch,tmp_path):
    original=reset.model
    def model(*args,**kwargs):
        kwargs['detach']=False
        return original(*args,**kwargs)
    monkeypatch.setattr(reset,'model',model)
    reset.test_whole_reset_implicit_poisson_detached_original(engine,selection,monkeypatch,tmp_path)


@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('refractory',[False,True])
def test_selected_poisson_surrogate_all_bank_initial_vjps(engine,window,refractory,monkeypatch):
    monkeypatch.setattr(reset.base,'poisson_factory',implicit_poisson_factory)
    monkeypatch.setattr(reset,'reference',reference)
    reset.test_whole_reset_all_bank_initial_vjps(engine,'poisson',window,refractory)


@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('refractory',[False,True])
def test_selected_poisson_surrogate_zero_score(engine,window,refractory,monkeypatch):
    monkeypatch.setattr(reset.base,'poisson_factory',constant_array_poisson_factory)
    gain=.96 if refractory else .65
    data=reset.model(engine,'poisson','partial',window,refractory,gain=gain);bundle=data[4];g=data[2]
    row=next(row for row in bundle.provenance['bindings'] if row['object']==g.name and 'mean' in row['variables'])
    bundle.weights[row['bank']][row['variables'].index('mean')]=0.
    voltage=bundle.provenance['neuron_state_layout'][g.name]['v'][0];value=.504-.6*gain
    bundle.initial_state[voltage]=value;bundle.plan['dynamic']['initial'][voltage]=value
    loss,_,anchors=reference(data,'poisson',bundle.weights,constant_rate=True)
    assert abs(reference(data,'poisson',bundle.weights,override=(2,0),constant_rate=True)[0]-loss)>1e-5
    monkeypatch.setattr(reset,'model',lambda *args,**kwargs:data)
    monkeypatch.setattr(reset,'reference',lambda *args,**kwargs:reference(*args,constant_rate=True,**kwargs))
    reset.test_whole_reset_all_bank_initial_vjps(engine,'poisson',window,refractory)


def batch_count(seed,sequence,batch,domain,entity,tick,rate,stream):
    import math
    from test_training_poisson_core import mix,MASK,uniform
    key=mix(seed^0x4232504f49533031)
    for field in (sequence,batch,domain,entity,0,tick,stream):key=mix(key^mix((field+0x9e3779b97f4a7c15)&MASK))
    total=0.
    for draw in range(10000):
        total-=math.log1p(-uniform(key,draw))
        if total>=rate:return draw
    raise AssertionError('independent count work budget exceeded')


@pytest.mark.parametrize('ranks',[None,2,8])
@pytest.mark.parametrize('window',[None,2])
def test_selected_poisson_batched_rank_restore(engine,ranks,window,monkeypatch,tmp_path):
    import copy
    reset.mpi(ranks);monkeypatch.setattr(reset.base,'poisson_factory',implicit_poisson_factory)
    data=reset.model(engine,'poisson','partial',window,True,gain=.7);bundle=data[4]
    p=copy.deepcopy(bundle.plan);p['mpi_ranks']=ranks;p['trainable']=[False]*len(bundle.weights)
    full=reset.NativeLIFTrainer(p,weights=bundle.weights,runner=reset.RUNNER,request_timeout=300).step(np.zeros((2,6,1)),[0,1],noise_sequence=9)
    references=[reference(data,'poisson',bundle.weights,batch=j,label=j) for j in range(2)]
    assert full['loss']==pytest.approx(sum(r[0] for r in references)/2,abs=8e-6)
    cells=[*bundle.provenance['neuron_state_layout'][data[1].name]['a'],*bundle.provenance['neuron_state_layout'][data[2].name]['v']]
    for j,(loss,z,anchor) in enumerate(references):
        np.testing.assert_allclose(np.asarray(full['final_state'])[j,cells],z[cells],rtol=8e-5,atol=8e-6)
        np.testing.assert_array_equal(np.asarray(full['spikes'])[j,:,4:],anchor['hard'])
    assert not np.array_equal(references[0][2]['candidate_counts'],references[1][2]['candidate_counts'])
    entries=full['poisson_state']['entries']
    for j,(_,_,anchor) in enumerate(references):assert sum(e['identity']['batch']==j for e in entries)==len(anchor['samples'])
    trainer=reset.NativeLIFTrainer(p,weights=bundle.weights,runner=reset.RUNNER,request_timeout=300)
    first=trainer.step(np.zeros((2,3,1)),[0,1],noise_sequence=9);path=tmp_path/'candidate-batch';trainer.store(path)
    trainer=reset.NativeLIFTrainer(p,runner=reset.RUNNER,request_timeout=300);trainer.restore(path)
    last=trainer.step(np.zeros((2,3,1)),[0,1],initial='carry')
    np.testing.assert_allclose(last['final_state'],full['final_state'],rtol=8e-5,atol=8e-6)
    np.testing.assert_array_equal(np.concatenate([first['spikes'],last['spikes']],axis=1),full['spikes'])
    assert last['poisson_state']==full['poisson_state']
