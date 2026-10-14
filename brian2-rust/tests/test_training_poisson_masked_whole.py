"""Whole callback rates retain NumPy refractory-selected caller copies."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_capture_callbacks import add_selected
from test_training_returned_captures import return_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_batch_poisson_captures import observation_stream
from test_native_training import RUNNER


def model(backend,ranks,mode,derived=False,empty=False,lower=True,full_rate=False,zero=False,full_active=False):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0],np.array([0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1 (unless refractory)\nu:1',threshold='v>.5',reset='v-=.5',refractory=.4*b.ms,method='euler',dt=dt)
    g.v=[.38 if zero else .117,.217 if full_active else .719];g.u=[.017,.031];g.lastspike=np.array([0.,0.] if empty else [-1.,-1.])*b.second
    rate='q(h)+.1' if derived else 'q(h)'
    syn=b.Synapses(source,g,'h:1\nrate:1 (constant)',on_pre='v_post+=.1*f(.2*poisson('+rate+'))+.01*h',dt=dt)
    syn.connect(i=[0,1],j=[0,1]);syn.h=[.113,.173];syn.rate=[(-.1 if derived else 0.) if zero else 1.1,1.9 if full_rate else -1.];syn.delay=dt
    if mode=='array':syn.variables['_postsynaptic_idx'].unique=True
    syn.namespace['q']=b.Function(return_capture(syn.variables['rate'].get_value() if full_rate else syn.variables['rate'].get_value()[:1]),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    syn.namespace['f']=b.Function(add_selected(g.variables['u'].get_value()),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn);x=np.zeros((1,5,2));x[0,0,:]=1;x[0,2,0]=1
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','rate']}) if lower else None
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('derived',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_whole_rate_original_restore(engine,mode,derived,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,derived)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(5):
        def sampler(rate,*args,**kwargs):
            raw=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
            rows=[edge for edge in raw if g.not_refractory[int(syn.j[edge])]]
            expected=float(syn.rate[0])+(.1 if derived else 0.);np.testing.assert_allclose(rate,[expected])
            assert not args and kwargs=={'size':len(rows)};calls.append((tick,len(rows)))
            return np.array([observation_stream(p['seed'],9,domain,edge,tick-1,expected,0) for edge in rows],dtype=np.int32)
        monkeypatch.setattr(np.random,'poisson',sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        np.testing.assert_array_equal(syn.h[:],[.113,.173])
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,1),(3,1)] and len(t.poisson_state['entries'])==2
    assert {row['identity']['site']['entity'] for row in t.poisson_state['entries']}=={0}


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('derived',[False,True])
@pytest.mark.parametrize('kind',['empty','shape','negative'])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_whole_rate_atomic_restore(engine,mode,derived,kind,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,derived,empty=kind=='empty',full_rate=kind=='shape')
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER);t.step(x[:,:1],[0],noise_sequence=9);net.run(dt,namespace={})
    path=tmp_path/'before';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    if kind=='negative':
        bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['variables']==['rate'])
        t.state['weights'][bank][0]=-1.;syn.rate[0]=-1.
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,1:2],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})


def reference(data,weights,mode,derived,window=None,initial=None,anchors=None,override=None):
    _,g,syn,_,bundle,x=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];counter=layout['__refractory_ticks']
    activity=bundle.provenance['refractory_activity_layout'][g.name]
    banks={row['variables'][0]:row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name}
    rate=float(weights[banks['rate']][0])+(.1 if derived else 0.)
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    locals_=bundle.provenance['event_callback_caller_locals'][syn.pre.name];trace=locals_['trace']
    fields={row['key'][1]:row['fields'] for row in locals_['rows'] if row['key'][0]=='new'}
    targets=[stage['outputs'][next(iter(stage['outputs']))] for stage in trace['stages']]
    assert len(targets)==5 and len(paths)==(5 if mode=='vectorised' else 6)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];before=[];margins=[];hard_rows=[];free_rows=[];spikes=[];selections=[];samples=[]
    for tick in range(5):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());free=z[counter]<.5;z[counter]=np.maximum(z[counter]-1,0.)
        if anchors is not None:free=anchors['free'][tick]
        z[v]+=.2*z[u]*free;margin=z[v]-.5;hard=(margin>0)&free;event=hard.astype(float)
        margins.append(margin.copy());hard_rows.append(hard.copy());free_rows.append(free.copy())
        if anchors is not None:event=anchors['hard'][tick].astype(float)+free*p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors['margins'][tick]))**2*(margin-anchors['margins'][tick])
        spikes.append(event.copy());flag=free&~hard;z[activity]=flag;selections.append([])
        for stage,path in enumerate(paths):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            rows=bundle.provenance['delay_queues'][path]['new'];gates=[z[row['states'][0]] for row in rows]
            selected=[row['edge'] for row,gate in zip(rows,gates) if gate>.5 and flag[row['edge']]]
            if anchors is not None:selected=anchors['selections'][tick][stage]
            selections[-1].append(selected)
            if stage==0:
                for edge,field in fields.items():
                    z[field[trace['stages'][0]['inputs']['v_post']]]=z[v[edge]]
                    z[field[trace['stages'][0]['inputs']['h']]]=weights[banks['h']][edge]
            if stage==2 and any(gate>.5 for gate in gates):
                for entry in bundle.provenance['event_callback_whole_locals'].get(syn.pre.name,{}).values():z[entry['cells']]=rate
            if stage==4 and selected:
                incoming=np.array([z[fields[edge][targets[1]]]*z[fields[edge][targets[3]]] for edge in selected]);z[u]+=incoming[0] if len(incoming)==1 else incoming
            for row,gate in zip(rows,gates):
                edge=row['edge'];field=fields[edge];active=float(flag[edge]);amplitude=gate*active
                if stage==0:z[field[targets[0]]]+=amplitude*(.1-z[field[targets[0]]])
                elif stage==1:z[field[targets[1]]]+=amplitude*(.2-z[field[targets[1]]])
                elif stage==3 and edge in selected:
                    count=observation_stream(p['seed'],9,domain,edge,tick-1,rate,0);samples.append((tick,edge,count))
                    if override==(tick,edge):count+=1
                    z[field[targets[3]]]=count
                elif stage==4:
                    value=.7*z[field[targets[0]]]*z[field[targets[1]]]*z[field[targets[3]]]+.01*z[field[trace['stages'][4]['inputs']['h']]]
                    if mode=='vectorised':z[v[edge]]+=amplitude*value
                    else:z[field[targets[4]]]+=amplitude*value
                elif stage==5:z[v[edge]]+=gate*(z[field[targets[4]]]-z[v[edge]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=x[0,tick,row['edge']]
        z[v]-=.5*event;z[counter]=np.where(hard,1.,z[counter])
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard_rows,free=free_rows,selections=selections,samples=samples)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('derived',[False,True])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_whole_rate_all_vjps(engine,mode,derived,zero,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,derived,zero=zero);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    args=(data,bundle.weights,mode,derived);loss,z,anchors=reference(*args,window=window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['rate'])
    rate=float(bundle.weights[bank][0])+(.1 if derived else 0.);score=0.
    for tick,edge,count in anchors['samples']:score+=loss*(count/rate-1.) if rate else reference(*args,override=(tick,edge))[0]-loss
    assert abs(score)>1e-5
    for bindex,row in enumerate(bundle.weights):
        for index in range(len(row)):
            if bindex==bank:expected=score if index==0 else 0.
            else:
                hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bindex][index]+=1e-6;lo[bindex][index]-=1e-6
                expected=(reference(data,hi,mode,derived,window=window,anchors=anchors)[0]-reference(data,lo,mode,derived,window=window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bindex][index]==pytest.approx(expected,rel=1e-3,abs=1e-5),(bindex,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(*args,window=window,initial=hi,anchors=anchors)[0]-reference(*args,window=window,initial=lo,anchors=anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('derived',[False,True])
@pytest.mark.parametrize('ranks',[None,2])
def test_masked_whole_full_rate_prefix_restore(engine,mode,derived,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,derived,full_rate=True,full_active=True)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    for tick in range(3):
        def sampler(rate,*args,**kwargs):
            rows=[edge for edge in range(2) if x[0,tick-1,edge]>0] if tick else []
            rows=[edge for edge in rows if g.not_refractory[int(syn.j[edge])]]
            expected=np.array(syn.rate[:])+(.1 if derived else 0.);np.testing.assert_allclose(rate,expected)
            assert not args and kwargs=={'size':len(rows)};calls.append((tick,len(rows)))
            return np.array([observation_stream(p['seed'],9,domain,edge,tick-1,float(expected[i]),0) for i,edge in enumerate(rows)],dtype=np.int32)
        monkeypatch.setattr(np.random,'poisson',sampler)
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for name in ('v','u'):
            cells=bundle.provenance['neuron_state_layout'][g.name][name]
            np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],g.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    assert calls==[(1,2)] and len(t.poisson_state['entries'])==2
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,3:4],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.poisson_state,t.elapsed_ticks,t.next_noise_sequence)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
