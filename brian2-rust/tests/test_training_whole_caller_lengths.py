"""Selected-length caller arrays retain shape across indexed write stages."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_training_returned_captures import return_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_native_training import RUNNER


def model(backend, ranks, mode, code=None):
    b.set_device('runtime'); b.start_scope(); b.prefs.codegen.target='numpy'
    dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0],np.array([0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.217,.719]; g.u=[.017,.031]
    if code is None:code='tmp=f(h)+.003*randn()+.004*rand();saved=tmp;h=tmp;tmp*=.8;h=saved'
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=code+(';u_post+=gain*h' if mode=='vectorised' else ''),dt=dt)
    syn.connect(i=[1,0],j=[0,0] if mode=='vectorised' else [0,1])
    syn.h=[.113,.173]; syn.gain=[.11,.19]; syn.delay=dt
    syn.namespace['f']=b.Function(return_capture(syn.variables['gain'].get_value()[:1]),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    if 'q(' in code:
        syn.namespace['q']=b.Function(return_capture(syn.variables['gain'].get_value()[::-1]),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn); x=np.zeros((1,4,2)); x[0,0,:]=1.; x[0,2,0]=1.
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_length_original_restore(engine,mode,ranks,tmp_path,monkeypatch):
    mpi(ranks)
    original_restore(model(engine,ranks,mode),mode,tmp_path,monkeypatch)


def original_restore(data,mode,tmp_path,monkeypatch):
    net,g,syn,dt,bundle,x=data
    p=copy.deepcopy(bundle.plan); p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    path=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1]
    order=[row['edge'] for row in bundle.provenance['delay_queues'][path]['new']]
    calls=[]
    for tick in range(4):
        rows=[edge for edge in order if x[0,tick-1,int(syn.i[edge])]>0] if tick in (1,3) else []
        def sampler(kind,stream):
            def sample(*shape):
                assert shape==(len(rows),); calls.append((tick,kind,len(rows)))
                return np.array([draw(kind,p['seed'],9,0,domain,edge,tick-1,None,stream) for edge in rows])
            return sample
        monkeypatch.setattr(np.random,'randn',sampler('randn',0)); monkeypatch.setattr(np.random,'rand',sampler('rand',1))
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}))
        net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        stored=tmp_path/str(tick); t.store(stored); t=NativeLIFTrainer(p,runner=RUNNER); t.restore(stored)
    assert calls==[(tick,kind,int(x[0,tick-1].sum())) for tick in (1,3) for kind in ('randn','rand')]
    assert t.next_noise_sequence==10
    if mode=='vectorised':
        storage=list(bundle.provenance['event_callback_whole_locals'][syn.pre.name].values())
        assert storage
        for row in storage:
            assert row['length_cell'] in p['dynamic']['integer_states']
            assert p['dynamic']['detached'][row['length_cell']]
            assert np.asarray(out['final_state'])[0,row['length_cell']]==1
    return out


def reference(data,mode,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data; p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name]; v=layout['v']; u=layout['u']
    h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    storage=list(bundle.provenance['event_callback_whole_locals'].get(syn.pre.name,{}).values())
    buffers=storage[0]['cells'] if storage else None
    length_cell=storage[0]['length_cell'] if storage else None
    random=bundle.provenance['event_callback_random_fields'][syn.pre.name]
    noise_fields={row['key'][1]:row['fields'] for row in random['rows'] if row['key'][0]=='new'}
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    order=[row['edge'] for row in bundle.provenance['delay_queues'][paths[0]]['new']]
    before=[]; margins=[]; hard=[]; spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy()); z[v]+=.2*z[u]; margin=z[v]-.5; event=(margin>0).astype(float)
        margins.append(margin.copy()); hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick]
            event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy())
        selected=order if tick==1 else [edge for edge in order if syn.i[edge]==0] if tick==3 else []
        if selected:
            for edge in selected:
                for sample in random['draws']:
                    z[noise_fields[edge][sample['name']]]=draw(sample['kind'],p['seed'],9,0,domain,edge,tick-1,None,sample['stream'])
        def returned(edge):
            return weights[bank][0]+.003*z[noise_fields[edge]['_b2_batch_random_0']]+.004*z[noise_fields[edge]['_b2_batch_random_1']]
        compact=np.array([returned(edge) for edge in selected])
        for stage,path in enumerate(paths):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            old=z.copy()
            if buffers is not None and selected:
                if stage==0:z[buffers[:len(selected)]]=compact; z[length_cell]=len(selected)
                if stage==1:z[buffers[:len(selected)]]=.8*old[buffers[:len(selected)]]
            for row in bundle.provenance['delay_queues'][path]['new']:
                edge=row['edge']; gate=old[row['states'][0]]
                if stage==2:z[u[0]]+=gate*weights[bank][edge]*old[h[edge]]
                else:
                    ordinal=sum(prior in selected for prior in order[:order.index(edge)])
                    if stage==0:
                        value=compact[min(ordinal,len(compact)-1)] if len(compact) else returned(edge)
                        if mode=='array':value*=.8
                    else:
                        # Only the actual prefix is mutated; padded cells remain
                        # available to the declared inactive-row counterfactual.
                        size=int(old[length_cell]); column=0 if size==1 else ordinal
                        value=.8*old[buffers[column]] if column<size else old[buffers[column]]
                    z[h[edge]]+=gate*(value-old[h[edge]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states']
                if cells:
                    z[cells[:-1]]=z[cells[1:]]
                    z[cells[-1]]=float(tick==0 or tick==2 and syn.i[row['edge']]==0)
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale']
    loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_length_all_vjps(engine,mode,window,ranks):
    mpi(ranks); data=model(engine,ranks,mode); _,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan); p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    loss,z,anchors=reference(data,mode,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights); lo=copy.deepcopy(bundle.weights)
            hi[bank][index]+=1e-6; lo[bank][index]-=1e-6
            fd=(reference(data,mode,hi,window,anchors=anchors)[0]-reference(data,mode,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:
            assert out['initial_state_gradients'][0][index]==0; continue
        hi=np.array(bundle.initial_state); lo=hi.copy(); hi[index]+=1e-6; lo[index]-=1e-6
        fd=(reference(data,mode,bundle.weights,window,hi,anchors)[0]-reference(data,mode,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('ranks',[None,2])
def test_selected_length_versions_restore(engine,mode,ranks,tmp_path,monkeypatch):
    mpi(ranks)
    code='tmp=f(h)+.003*randn()+.004*rand();saved=tmp;h=tmp;tmp=2.*tmp+.03;h=tmp;saved*=.8;h=saved;h=tmp'
    data=model(engine,ranks,mode,code)
    original_restore(data,mode,tmp_path,monkeypatch)
    if mode=='vectorised':
        storage=list(data[4].provenance['event_callback_whole_locals'][data[2].pre.name].values())
        assert len(storage)==2
        assert set(storage[0]['cells']).isdisjoint(storage[1]['cells'])
        assert storage[0]['length_cell']!=storage[1]['length_cell']


@pytest.mark.parametrize('ranks',[None,2])
def test_selected_length_padding_domain(engine,ranks,tmp_path,monkeypatch):
    mpi(ranks)
    code='tmp=f(h)+.003*randn()+.004*rand();h=tmp;tmp=1./tmp;h=tmp'
    data=model(engine,ranks,'vectorised',code)
    source=next(obj for obj in data[0].objects if isinstance(obj,b.SpikeGeneratorGroup))
    source.set_spikes([0,0],np.array([0,2])*data[3]);data[5][0,0,1]=0
    # The second column remains exactly zero. Reciprocal must evaluate only
    # its actual prefix; empty batches never touch either private buffer.
    out=original_restore(data,'vectorised',tmp_path,monkeypatch)
    for row in data[4].provenance['event_callback_whole_locals'][data[2].pre.name].values():
        assert np.asarray(out['final_state'])[0,row['cells'][1]]==0


@pytest.mark.parametrize('ranks',[None,2])
def test_selected_length_shape_rollback(engine,ranks,tmp_path,monkeypatch):
    mpi(ranks)
    code='tmp=f(h)+.003*randn()+.004*rand();h=tmp;tmp*=q(h);h=tmp'
    net,_,syn,dt,bundle,x=model(engine,ranks,'vectorised',code)
    p=copy.deepcopy(bundle.plan); p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    t.step(x[:,:3],[0],noise_sequence=9)
    path=tmp_path/'before-shape-error'; t.store(path); t=NativeLIFTrainer(p,runner=RUNNER); t.restore(path)
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,3:],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks,t.next_noise_sequence)==before
    net.run(3*dt,namespace={}); calls=[]
    def singleton(*shape):
        assert shape==(1,);calls.append(shape);return np.zeros(1)
    monkeypatch.setattr(np.random,'randn',singleton);monkeypatch.setattr(np.random,'rand',singleton)
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
    assert len(calls)==2
