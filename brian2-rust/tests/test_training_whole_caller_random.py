"""Emission-addressed draws and time in private whole-vector caller storage."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_training_event_noise import draw
from test_native_training import RUNNER
import test_training_event_returned_constants as base

CODES={kind:'tmp=f(h)+.003*randn()+.004*rand()'+('+.002*t/dt' if kind=='time' else '')+';saved=tmp;h=tmp;tmp*=.8;h=saved' for kind in ('noise','time')}

def model(backend,ranks,mode,kind):
    old=base.CODES['copy'];base.CODES['copy']=CODES[kind]
    try:return base.model(mode,'copy',ranks,backend)
    finally:base.CODES['copy']=old

@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_random_original_restore(engine,mode,kind,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name];calls=[]
    order=[row['edge'] for row in bundle.provenance['delay_queues'][bundle.provenance['event_callback_stage_groups'][syn.pre.name][1]]['new']]
    for tick in range(4):
        def sampler(draw_kind,stream):
            def sample(*shape):
                rows=order if tick in (1,3) else []
                assert shape==(len(rows),)
                calls.append((tick,draw_kind,len(rows)))
                return np.array([draw(draw_kind,p['seed'],9,0,domain,edge,tick-1,None,stream) for edge in rows])
            return sample
        monkeypatch.setattr(np.random,'randn',sampler('randn',0));monkeypatch.setattr(np.random,'rand',sampler('rand',1))
        out=t.step(x[:,tick:tick+1],[0],**({'initial':'carry'} if tick else {'noise_sequence':9}));net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        np.testing.assert_array_equal(syn.gain[:],[.11,.19])
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert len(calls)==4 and t.next_noise_sequence==10


def reference(data,mode,kind,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan
    z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u']
    h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    paths=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    assert len(paths)==(3 if mode=='vectorised' else 1)
    storage=bundle.provenance['event_callback_whole_locals'].get(syn.pre.name,{})
    buffers=[row['cells'] for row in storage.values()];assert len(buffers)==(1 if mode=='vectorised' else 0)
    random=bundle.provenance['event_callback_random_fields'][syn.pre.name]
    noise_fields={row['key'][1]:row['fields'] for row in random['rows'] if row['key'][0]=='new'}
    domain=bundle.provenance['scheduled_noise_domains'][syn.pre.name]
    first=bundle.provenance['delay_queues'][paths[0]]['new'];order=[row['edge'] for row in first]
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());active=tick in (1,3)
        if active:
            for edge in order:
                for sample in random['draws']:
                    z[noise_fields[edge][sample['name']]]=draw(sample['kind'],p['seed'],9,0,domain,edge,tick-1,None,sample['stream'])
        offsets={edge:.003*z[noise_fields[edge]['_b2_batch_random_0']]+.004*z[noise_fields[edge]['_b2_batch_random_1']]+(.002*tick if kind=='time' else 0.) for edge in order}
        for stage,path in enumerate(paths):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            old=z.copy();returned=np.array(weights[bank])[::-1]+np.array([offsets[edge] for edge in order])
            if stage==0 and mode=='vectorised' and active:z[buffers[0]]=returned
            if stage==1 and active:z[buffers[0]]=.8*old[buffers[0]]
            for ordinal,row in enumerate(bundle.provenance['delay_queues'][path]['new']):
                edge=row['edge'];gate=old[row['states'][0]]
                if stage==2:z[u[0]]+=gate*weights[bank][edge]*old[h[edge]]
                else:
                    if stage==0:
                        value=returned[ordinal] if active else weights[bank][-1]+offsets[edge]
                        if mode=='array':value*=.8
                    else:value=.8*old[buffers[0]][ordinal if active else 0]
                    z[h[edge]]+=gate*(value-old[h[edge]])
        for queue in bundle.provenance['delay_queues'].values():
            for row in queue['new']:
                cells=row['states']
                if cells:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick in (0,2))
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_random_all_vjps(engine,mode,kind,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0],noise_sequence=9)
    loss,z,anchors=reference(data,mode,kind,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,kind,hi,window,anchors=anchors)[0]-reference(data,mode,kind,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(z)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:
            assert out['initial_state_gradients'][0][index]==0;continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,kind,bundle.weights,window,hi,anchors)[0]-reference(data,mode,kind,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index


@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_random_shape_failure_after_restore(engine,kind,ranks,tmp_path,monkeypatch):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,'vectorised',kind)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    source.set_spikes([0,1,0],np.array([0,0,2])*dt);x[0,2,1]=0
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    t.step(x[:,:3],[0],noise_sequence=9)
    path=tmp_path/'before-shape-error';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,3:],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks,t.next_noise_sequence)==before
    net.run(3*dt,namespace={});calls=[]
    def singleton(*shape):
        assert shape==(1,);calls.append(shape);return np.zeros(1)
    monkeypatch.setattr(np.random,'randn',singleton);monkeypatch.setattr(np.random,'rand',singleton)
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
    assert len(calls)==2
