"""Private whole arrays survive indexed publication without borrowing captures."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER
import test_training_event_returned_constants as base

CODES={'derived':'tmp=f(h)+.1;h=tmp;tmp*=.8;h=tmp',
       'borrow':'tmp=f(h)+.1;saved=tmp;h=tmp;tmp*=.8;h=saved',
       'rebind':'tmp=f(h)+.1;saved=tmp;h=tmp;tmp=2.*tmp+.03;h=tmp;h=saved',
       'versions':'tmp=f(h)+.1;saved=tmp;h=tmp;tmp=2.*tmp+.03;h=tmp;h=saved;h=tmp'}

def model(backend,ranks,mode,kind):
    old=base.CODES['copy'];base.CODES['copy']=CODES[kind]
    try:return base.model(mode,'copy',ranks,backend)
    finally:base.CODES['copy']=old

@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_derived_original_restore(engine,mode,kind,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,mode,kind)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    assert 'gain' not in bundle.provenance['dynamic_state_layout'][syn.name]
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name]
                np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        np.testing.assert_array_equal(syn.gain[:],[.11,.19])
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)


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
    count=5 if kind=='versions' else 4 if kind=='rebind' else 3
    assert len(paths)==(count if mode=='vectorised' else 1)
    storage=bundle.provenance['event_callback_whole_locals'].get(syn.pre.name,{})
    buffers=[row['cells'] for _,row in sorted(storage.items(),key=lambda row:int(row[0]))]
    assert len(buffers)==(0 if mode=='array' else 2 if kind=='versions' else 1)
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());active=tick in (1,3)
        for stage,path in enumerate(paths):
            for row in bundle.provenance['event_callback_snapshots'].get(path,[]):z[row['cache']]=z[row['source']]
            old=z.copy();returned=np.array(weights[bank])[::-1]+.1
            tail=mode=='vectorised' and stage==len(paths)-1
            if mode=='array':values=returned*.8 if kind in ('derived','borrow') else 2*returned+.03 if kind=='versions' else returned
            elif stage==0:
                values=returned
                if active:z[buffers[0]]=values
            elif stage==1:
                values=.8*old[buffers[0]] if kind in ('derived','borrow') else 2*old[buffers[0]]+.03
                if active and kind in ('derived','borrow'):z[buffers[0]]=values
                if active and kind=='versions':z[buffers[1]]=values
            elif not tail:values=old[buffers[0 if stage==2 else 1]]
            for ordinal,row in enumerate(bundle.provenance['delay_queues'][path]['new']):
                edge=row['edge'];gate=old[row['states'][0]]
                if tail:z[u[0]]+=gate*weights[bank][edge]*old[h[edge]]
                else:z[h[edge]]+=gate*(values[ordinal if active else 0]-old[h[edge]])
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
def test_whole_derived_all_vjps(engine,mode,kind,window,ranks):
    mpi(ranks);data=model(engine,ranks,mode,kind);_,g,syn,_,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,mode,kind,bundle.weights,window)
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


@pytest.mark.parametrize('kind',['derived','versions'])
@pytest.mark.parametrize('ranks',[None,2])
def test_whole_derived_shape_failure_after_restore(engine,kind,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(engine,ranks,'vectorised',kind)
    source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
    source.set_spikes([0,1,0],np.array([0,0,2])*dt);x[0,2,1]=0
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    t.step(x[:,:3],[0]);net.run(3*dt,namespace={})
    path=tmp_path/'before-shape-error';t.store(path);t=NativeLIFTrainer(p,runner=RUNNER);t.restore(path)
    before=copy.deepcopy((t.state,t.neuron_state,t.clock_state,t.elapsed_ticks,t.next_noise_sequence))
    with pytest.raises(ValueError):t.step(x[:,3:],[0],initial='carry')
    assert (t.state,t.neuron_state,t.clock_state,t.elapsed_ticks,t.next_noise_sequence)==before
    with pytest.raises(b.BrianObjectException):net.run(dt,namespace={})
