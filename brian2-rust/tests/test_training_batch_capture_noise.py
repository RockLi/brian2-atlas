"""Emission-addressed random arrays mixed with mutable captured callbacks."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_native_training import RUNNER
from test_training_masked_multistatement_captures import model as base_model,CODES


def model(backend,ranks,mode,alias):
    key='batch_noise'
    CODES[key]='v_post+=gain*f(h+.003*randn()+.004*rand())'
    try:return base_model(backend,ranks,mode,key,alias)
    finally:del CODES[key]


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_capture_random_original_restore(engine,mode,alias,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,alias)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    posts=np.asarray(syn.j[:],int);calls=[]
    for tick in range(5):
        def sampler(kind,stream):
            def sample(*shape):
                raw=np.flatnonzero(x[0,tick-1]) if tick else np.array([],int)
                flags=np.asarray(g.not_refractory[:],bool)
                rows=[int(edge) for edge in raw if flags[posts[edge]]]
                assert shape==(len(rows),),(tick,kind,shape,rows)
                calls.append((tick,kind,len(rows)))
                return np.array([draw(kind,p['seed'],9,0,domain,edge,tick-1,None,stream) for edge in rows])
            return sample
        monkeypatch.setattr(np.random,'randn',sampler('randn',0));monkeypatch.setattr(np.random,'rand',sampler('rand',1))
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert len(calls)==6
    assert t.next_noise_sequence==10


def reference(data,mode,alias,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];counter=layout['__refractory_ticks']
    activity=bundle.provenance['refractory_activity_layout'][g.name]
    h=bundle.provenance['dynamic_state_layout'][syn.name].get('h')
    h_bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['h'])
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    path=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1]
    rows=bundle.provenance['delay_queues'][path]['new'];posts=np.asarray(syn.j[:],int)
    capture=v if alias else u;before=[];margins=[];hard_rows=[];free_rows=[];raw_rows=[];spikes=[]
    for tick in range(5):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());free=z[counter]<.5;z[counter]=np.maximum(z[counter]-1,0.)
        if anchors is not None:free=anchors['free'][tick]
        margin=z[v]-.5;hard=(margin>0)&free;event=hard.astype(float)
        if anchors is not None:
            hard=anchors['hard'][tick];old=anchors['margins'][tick]
            event=hard.astype(float)+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        margins.append(margin.copy());hard_rows.append(hard.copy());free_rows.append(free.copy());spikes.append(event.copy());flag=free&~hard;z[activity]=flag
        raw=[row['edge'] for row in rows if z[row['states'][0]]>.5]
        if anchors is not None:raw=anchors['raw'][tick]
        random=bundle.provenance['event_callback_random_fields'][syn.pre.name]
        domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
        noise_rows={row['key'][1]:row['fields'] for row in random['rows'] if row['key'][0]=='new'}
        for edge in raw:
            for sample in random['draws']:
                z[noise_rows[edge][sample['name']]]=draw(sample['kind'],p['seed'],9,0,domain,edge,tick-1,None,sample['stream'])
        for copies in bundle.provenance['event_callback_snapshots'].values():
            for row in copies:z[row['cache']]=z[row['source']]
        old=z.copy();raw=[row['edge'] for row in rows if old[row['states'][0]]>.5]
        h_values=(np.asarray(weights[h_bank]) if h is None else old[h])+np.array([.003*old[noise_rows[edge]['_b2_batch_random_0']]+.004*old[noise_rows[edge]['_b2_batch_random_1']] for edge in range(len(posts))])
        if anchors is not None:raw=anchors['raw'][tick]
        raw_rows.append(list(raw));active=[edge for edge in raw if flag[posts[edge]]]
        if active:
            incoming=h_values[active]
            z[capture]+=incoming[0] if len(incoming)==1 else incoming
        for index,row in enumerate(rows):
            edge=row['edge'];post=posts[edge];amplitude=old[row['states'][0]];live=z[v[post]]
            ordinal=sum(other['edge'] in active for other in rows[:index])
            selected=active[0] if len(active)==1 else active[ordinal] if ordinal<len(active) else edge
            value=.7*h_values[selected]
            base=old[v[post]] if mode=='array' else live
            new=base+(weights[gain][edge]*value if flag[post] else 0.)
            z[v[post]]=live+amplitude*(new-live)
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];edge=row['edge']
                z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick==0 or tick==1 and edge==0 or tick==2 and edge==len(posts)-1)
        z[v]-=.5*event;z[counter]=np.where(hard,1.,z[counter])
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,free=free_rows,raw=raw_rows)

@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_capture_random_all_vjps(engine,mode,alias,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,alias);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9);loss,z,anchors=reference(data,mode,alias,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,alias,hi,window,anchors=anchors)[0]-reference(data,mode,alias,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,alias,bundle.weights,window,hi,anchors)[0]-reference(data,mode,alias,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_capture_empty_random_atomic(engine,mode,ranks,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    for cell in bundle.provenance['neuron_state_layout'][g.name]['__refractory_ticks']:p['dynamic']['initial'][cell]=2.
    g.lastspike=0*b.second
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    t.step(x[:,:1],[0],noise_sequence=9);net.run(dt,namespace={})
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,1:2],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks,t.next_noise_sequence)==before
    calls=[]
    def empty(*shape):
        assert shape==(0,);calls.append(shape);return np.empty(0)
    monkeypatch.setattr(np.random,'randn',empty);monkeypatch.setattr(np.random,'rand',empty)
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
    assert len(calls)==2
