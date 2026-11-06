"""Poisson array temporaries preserve native likelihood sites across callbacks."""
import copy
import math
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_capture_callbacks import add_selected
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_poisson_event_replay import observation
from test_native_training import RUNNER
from test_training_event_noise import draw
from test_training_poisson_core import mix,uniform,MASK


def observation_stream(seed,sequence,domain,edge,emitted,rate,stream):
    value=mix(seed^0x4232504F49533031)
    for field in (sequence,0,domain,edge,1,emitted&MASK,stream):
        value=mix(value^mix((field+0x9E3779B97F4A7C15)&MASK))
    total=0.
    for count in range(10000):
        total-=math.log1p(-uniform(value,count))
        if total>=rate:return count
    raise AssertionError('independent small-rate draw budget exceeded')


def model(backend,ranks,mode,zero=False,code=None,masked=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0],np.array([0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    options={'refractory':.4*b.ms} if masked else {}
    g=b.NeuronGroup(2,'dv/dt=u/ms:1'+(' (unless refractory)' if masked else '')+'\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt,**options)
    g.v=[.217,.719] if masked else [.41,.31];g.u=[.017,.031]
    if masked:g.lastspike=np.array([-1.,-1.])*b.second
    if code is None:code='v_post+=gain*f(.2*poisson(h))'
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=code,dt=dt)
    syn.connect(i=[0,1],j=[0,0] if mode=='vectorised' and not masked else [0,1]);syn.h=[0,0] if zero else [1.1,-1.] if masked else [1.1,1.9];syn.gain=[.11,.19];syn.delay=dt
    if mode=='array':syn.variables['_postsynaptic_idx'].unique=True
    syn.namespace['f']=b.Function(add_selected(g.variables['u'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn);x=np.zeros((1,5,2));x[0,0,:]=1.;x[0,2,0]=1.
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    assert all(bundle.provenance['event_callback_modes'][path]['mode']==mode for path in bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:])
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_poisson_original_restore(engine,mode,ranks,tmp_path,monkeypatch):
    mpi(ranks)
    original_restore(model(engine,ranks,mode),tmp_path,monkeypatch)


def original_restore(data,tmp_path,monkeypatch,rate_factor=1.,poisson_streams=(0,),normal_rate_offset=0.):
    net,g,syn,dt,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        site=[0]
        def sampler(rate,*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
            if 'not_refractory' in g.variables:rows=[edge for edge in rows if g.not_refractory[int(syn.j[edge])]]
            noise=np.array([draw('randn',p['seed'],9,0,domain,edge,tick-1,None,0) for edge in rows])
            np.testing.assert_allclose(rate,rate_factor*np.asarray(syn.h[:])[rows]+normal_rate_offset*noise);assert not args and kwargs=={'size':len(rows)}
            calls.append((tick,len(rows)))
            stream=poisson_streams[site[0]];site[0]+=1
            return np.array([observation_stream(p['seed'],9,domain,edge,tick-1,float(value),stream) for edge,value in zip(rows,rate)],dtype=np.int32)
        if normal_rate_offset:
            def normal(*shape):
                rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
                assert shape==(len(rows),)
                return np.array([draw('randn',p['seed'],9,0,domain,edge,tick-1,None,0) for edge in rows])
            monkeypatch.setattr(np.random,'randn',normal)
        monkeypatch.setattr(np.random,'poisson',sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key].get(obj.name,{}).get(name)
                if cells is not None:np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        if 'not_refractory' in g.variables:
            bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['h'])
            assert out['gradients'][bank][1]==0
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    expected=[(1,1),(3,1)] if 'not_refractory' in g.variables else [(1,2),(3,1)]
    assert calls==[row for row in expected for _ in poisson_streams] and t.next_noise_sequence==10
    return t


def reference(data,weights,window=None,initial=None,anchors=None,override=None):
    _,g,syn,_,bundle,x=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    v=bundle.provenance['neuron_state_layout'][g.name]['v'];u=bundle.provenance['neuron_state_layout'][g.name]['u']
    banks={tuple(row['variables']):row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name}
    rates=np.array(weights[banks[('h',)]]);gains=np.array(weights[banks[('gain',)]])
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    vectorised=len(set(np.asarray(syn.j[:],int)))==1
    assert len(paths)==(4 if vectorised else 5)
    locals_=bundle.provenance['event_callback_caller_locals'][syn.pre.name]
    fields={row['key'][1]:row['fields'] for row in locals_['rows'] if row['key'][0]=='new'}
    tokens=locals_['trace']['tokens'];source={name:index for name,index in locals_['trace']['initial'].items()}
    output=[next(iter(stage['outputs'].values())) for stage in locals_['trace']['stages']]
    assert len(output)==4
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    before=[];margins=[];hard=[];spikes=[];samples=[];selections=[]
    posts=np.asarray(syn.j[:],int)
    for tick in range(5):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        for stage,path in enumerate(paths):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            rows=bundle.provenance['delay_queues'][path]['new']
            gates=[z[row['states'][0]] for row in rows]
            selected=[row['edge'] for row,gate in zip(rows,gates) if gate>.5]
            if anchors is not None:selected=anchors['selections'][tick][stage]
            if stage==0:
                # Array mode loads its caller locals before the first statement.
                for edge,bank in fields.items():
                    if 'v_post' in source:z[bank[source['v_post']]]=z[v[posts[edge]]]
                    if 'gain' in source:z[bank[source['gain']]]=gains[edge]
            if stage==2:
                for edge,bank in fields.items():z[bank[source['h']]]=rates[edge]
            if stage==3 and selected:
                incoming=np.array([z[fields[e][output[1]]]*z[fields[e][output[2]]] for e in selected])
                z[u]+=incoming[0] if len(incoming)==1 else incoming
            for row,gate in zip(rows,gates):
                edge=row['edge'];bank=fields[edge]
                if stage==0:z[bank[output[0]]]+=gate*(gains[edge]-z[bank[output[0]]])
                elif stage==1:z[bank[output[1]]]+=gate*(.2-z[bank[output[1]]])
                elif stage==2:
                    if edge in selected:
                        sample=observation(p['seed'],9,domain,edge,tick-1,None,rates[edge])[0]['count']
                        samples.append((tick,edge,sample))
                        if override==(tick,edge):sample+=1
                        z[bank[output[2]]]=sample
                elif stage==3:
                    value=.7*z[bank[output[0]]]*z[bank[output[1]]]*z[bank[output[2]]]
                    if vectorised:z[v[posts[edge]]]+=gate*value
                    else:z[bank[output[3]]]+=gate*value
                elif stage==4:z[v[posts[edge]]]+=gate*(z[bank[output[3]]]-z[v[posts[edge]]])
            if stage==0:selections.append([])
            selections[-1].append(selected)
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard,samples=samples,selections=selections)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_poisson_all_vjps(engine,mode,zero,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,zero);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    loss,z,anchors=reference(data,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    hbank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['h'])
    rate_derivative=np.zeros(2)
    for tick,edge,count in anchors['samples']:
        rate=bundle.weights[hbank][edge]
        rate_derivative[edge]+=reference(data,bundle.weights,override=(tick,edge))[0]-loss if not rate else loss*(count/rate-1.)
    assert np.any(abs(rate_derivative)>1e-5)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            # Counts are fixed when differentiating the deterministic equations;
            # rates instead use the native likelihood/boundary derivative.
            fd=rate_derivative[index] if bank==hbank else (reference(data,hi,window,anchors=anchors)[0]-reference(data,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:
            assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,bundle.weights,window,hi,anchors)[0]-reference(data,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['prior_statement','prior_operand','rate_callback'])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_poisson_callback_order(engine,mode,kind,ranks,tmp_path,monkeypatch):
    mpi(ranks)
    codes={'prior_statement':'h=f(h);v_post+=gain*f(.2*poisson(h))',
           'prior_operand':'v_post+=gain*(f(h)+f(.2*poisson(h)))',
           'rate_callback':'v_post+=gain*f(.2*poisson(f(h)))'}
    data=model(engine,ranks,mode,code=codes[kind])
    original_restore(data,tmp_path,monkeypatch,rate_factor=.7 if kind!='prior_statement' or mode=='array' else 1.)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_poisson_masked_rate_not_evaluated(engine,mode,ranks,tmp_path,monkeypatch):
    mpi(ranks)
    data=model(engine,ranks,mode,masked=True)
    t=original_restore(data,tmp_path,monkeypatch)
    assert len(t.poisson_state['entries'])==2
    assert {row['identity']['site']['entity'] for row in t.poisson_state['entries']}=={0}


@pytest.mark.parametrize('kind',['negative_rate','zero_boundary_domain'])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_poisson_restore_atomic_failure(engine,kind,ranks,tmp_path):
    mpi(ranks)
    code='v_post+=gain*f(.2/(1.-poisson(h)))' if kind=='zero_boundary_domain' else None
    _,_,syn,_,bundle,x=model(engine,ranks,'vectorised',zero=kind=='zero_boundary_domain',code=code)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);t.step(x[:,:1],[0],noise_sequence=9)
    path=tmp_path/'before-poisson-error';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    if kind=='negative_rate':
        bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['h'])
        t.state['weights'][bank][0]=-1.
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,1:2],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence)==before


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',['multiple','mixed'])
@pytest.mark.parametrize('ranks',[None,2])
def test_batch_poisson_stream_identity(engine,mode,kind,ranks,tmp_path,monkeypatch):
    mpi(ranks)
    code='v_post+=gain*f(.2*(poisson(h)+poisson(h)))' if kind=='multiple' else 'v_post+=gain*f(.2*poisson(h+.003*randn()))'
    data=model(engine,ranks,mode,code=code)
    t=original_restore(data,tmp_path,monkeypatch,poisson_streams=(0,1) if kind=='multiple' else (1,),normal_rate_offset=.003 if kind=='mixed' else 0.)
    assert {row['identity']['site']['stream'] for row in t.poisson_state['entries']}==({0,1} if kind=='multiple' else {1})
    assert len(t.poisson_state['entries'])==(6 if kind=='multiple' else 3)
