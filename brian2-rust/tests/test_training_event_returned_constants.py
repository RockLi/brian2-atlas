"""Caller writes through returned views promote physical constant storage."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_returned_captures import return_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER

CODES={'alias':'tmp=f(h);tmp*=.8;h=tmp',
       'copy':'tmp=f(h)+.1;tmp*=.8;h=tmp',
       'identity':'tmp=f(h)+0.;tmp*=.8;h=tmp',
       'rebind':'tmp=f(h);saved=tmp;tmp=tmp+.1;saved*=.8;h=tmp',
       'written_copy':'h=f(h);h*=.8'}


def model(mode,kind,ranks,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0,1],np.array([0,0,2,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(2,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt);g.v=[.217,.719];g.u=[.017,.031]
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=CODES[kind]+(';u_post+=gain*h' if mode=='vectorised' else ''),dt=dt)
    syn.connect(i=[1,0],j=[0,0] if mode=='vectorised' else [0,1]);syn.h=[.113,.173];syn.gain=[.11,.19];syn.delay=dt
    syn.namespace['f']=b.Function(return_capture(syn.variables['gain'].get_value()[::-1]),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    net=b.Network(source,hidden,g,syn);x=np.zeros((1,4,2));x[0,[0,2],:]=1.
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,trainable_synapse_parameters={syn.name:['h','gain']})
    promoted=kind not in ('copy',) and not (kind=='written_copy' and mode=='vectorised')
    assert ('gain' in bundle.provenance['mutable_constant_layout'][syn.name])==promoted
    return net,g,syn,dt,bundle,x


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('ranks',[None,2])
def test_returned_constant_caller_original_restore(engine,mode,kind,ranks,tmp_path):
    mpi(ranks);net,g,syn,dt,bundle,x=model(mode,kind,ranks,engine)
    p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights);t=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER)
    for tick in range(4):
        out=t.step(x[:,tick:tick+1],[0],initial='carry' if tick else None);net.run(dt,namespace={})
        for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h',*(['gain'] if 'gain' in bundle.provenance['dynamic_state_layout'][syn.name] else [])])]:
            for name in names:
                cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
        path=tmp_path/str(tick);t.store(path);t=NativeLIFTrainer(t.plan,runner=RUNNER);t.restore(path)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


def reference(data,mode,kind,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];fields=bundle.provenance['dynamic_state_layout'][syn.name];h=fields['h'];gain=fields.get('gain')
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    path=bundle.provenance['event_callback_stage_groups'][syn.pre.name][1];routes=bundle.provenance['delay_queues'][path]['new']
    before=[];margins=[];hard=[];spikes=[]
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());old=z.copy();coefficient=old[gain] if gain is not None else np.array(weights[bank]);active=tick in (1,3)
        # Brian rewrites tmp=tmp+.1 as tmp+=.1 in this generated block.
        # Both saved and tmp consequently retain the physical capture alias.
        changed=(coefficient+.1)*.8 if kind=='rebind' else coefficient*.8 if gain is not None else coefficient.copy()
        returned=(.8*(coefficient[::-1]+.1) if kind=='copy' else changed[::-1] if kind=='rebind' else .8*coefficient[::-1])
        if active and gain is not None:z[gain]=changed
        for ordinal,row in enumerate(routes):
            edge=row['edge'];gate=old[row['states'][0]];value=returned[ordinal if active else 0]
            z[h[edge]]=old[h[edge]]+gate*(value-old[h[edge]])
            if mode=='vectorised':z[u[0]]+=gate*changed[edge]*value
        for layout in bundle.provenance['delay_queues'].values():
            for row in layout['new']:
                cells=row['states']
                if cells:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(tick in (0,2))
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('mode',['array','vectorised'])
@pytest.mark.parametrize('kind',list(CODES))
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_returned_constant_caller_all_bank_initial_vjps(engine,mode,kind,window,ranks):
    mpi(ranks);data=model(mode,kind,ranks,engine);net,g,syn,dt,bundle,x=data;p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,mode,kind,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h'],*bundle.provenance['dynamic_state_layout'][syn.name].get('gain',[])]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6);assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,kind,hi,window,anchors=anchors)[0]-reference(data,mode,kind,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,kind,bundle.weights,window,hi,anchors)[0]-reference(data,mode,kind,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    bank=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and row['variables']==['gain'])
    if mode=='vectorised' and window is None:assert all(abs(value)>1e-5 for value in out['gradients'][bank])
    net.run(4*dt,namespace={});np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells[:2]],g.v[:],rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
