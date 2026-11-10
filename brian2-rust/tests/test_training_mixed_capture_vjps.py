"""Independent FIFO/batch references, including the adjoint of singleton broadcast."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_mixed_capture_lengths import model,singleton_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def data_model(mode,singleton,delay,ranks,backend):
    data=list(model(mode,delay,ranks,backend,'singleton' if singleton else 'independent'))
    if singleton:
        net,g,syn,a,_,_,_,_=data;data[4]=g.variables['v'].get_value()
        syn.namespace['f']=b.Function(singleton_capture(a,data[4]),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
        source=next(obj for obj in net.objects if isinstance(obj,b.SpikeGeneratorGroup))
        hidden=next(obj for obj in net.objects if isinstance(obj,b.NeuronGroup) and obj is not g)
        data[6]=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,
            trainable_synapse_parameters={syn.name:['h','gain']})
    return data


def reference(data,mode,singleton,weights,window,initial=None,anchors=None):
    _,g,syn,_,_,_,bundle,_=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and 'gain' in row['variables'])
    def capture(name):return next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']==name for alias in row['aliases']))
    first=capture('a');second=capture('b');before=[];margins=[];hard=[];spikes=[]
    paths=[syn.pre.name] if mode=='scalar' else bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    delay=bundle.provenance['delay_queues'][paths[0]]['new'][0]['delay']
    targets=np.asarray(syn.j[:])
    for tick in range(4):
        if anchors is not None and window and tick and tick%window==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v]+=.2*z[u];margin=z[v]-.5;event=(margin>0).astype(float)
        margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        spikes.append(event.copy());arrival=tick-delay;selected=[0,1] if arrival==0 else [0] if arrival==2 else []
        if mode=='scalar':
            for edge in range(2):
                rows=bundle.provenance['delay_queues'][paths[0]]['new'];gate=z[rows[edge]['states'][0]] if delay else float(edge in selected)
                old=z.copy();post=targets[edge];new_h=.7*old[v[post]]
                if singleton:
                    z[first]+=gate*(-.2*old[first]);z[second]+=gate*(.8*old[first[0]])
                else:
                    z[first]+=gate*(-.2*old[first]);z[second]+=gate*(-.4*old[second])
                z[h[edge]]=old[h[edge]]+gate*(new_h-old[h[edge]])
                # Explicit scalar voltage writeback wins over the raw capture.
                z[v[post]]=old[v[post]]+gate*weights[gain][edge]*new_h
        else:
            for stage,path in enumerate(paths):
                rows=bundle.provenance['delay_queues'][path]['new'];old_h=z[h].copy()
                if stage==0:
                    if selected:
                        z[first]*=.8
                        if singleton:z[second]+=z[first[0]]
                        else:z[second]*=.6
                    for edge in range(2):
                        gate=z[rows[edge]['states'][0]] if delay else float(edge in selected)
                        z[h[edge]]+=gate*(.7*old_h[edge]-z[h[edge]])
                if mode=='vectorised' and (len(paths)==1 or stage>0):
                    for edge in range(2):
                        gate=z[rows[edge]['states'][0]] if delay else float(edge in selected)
                        z[u[targets[edge]]]+=gate*weights[gain][edge]*(.7*old_h[edge] if len(paths)==1 else old_h[edge])
        for path in bundle.provenance['delay_queues'].values():
            for row in path['new']:
                if row['states']:
                    cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(row['edge'] in ([0,1] if tick==0 else [0] if tick==2 else []))
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('singleton',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_mixed_capture_all_bank_initial_vjps(engine,mode,singleton,delay,window,ranks):
    mpi(ranks);data=data_model(mode,singleton,delay,ranks,engine);net,g,syn,_,_,dt,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=reference(data,mode,singleton,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    cells+=list({cell for row in bundle.provenance['mutable_capture_layout'] for cell in row['cells']})
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(reference(data,mode,singleton,hi,window,anchors=anchors)[0]-reference(data,mode,singleton,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(reference(data,mode,singleton,bundle.weights,window,hi,anchors)[0]-reference(data,mode,singleton,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    if singleton and window is None:
        # With delayed arrivals, a TBPTT cut can sever every path from the
        # original singleton to a later logit. Its zero VJP is checked by FD;
        # full BPTT must exercise a nonzero broadcast adjoint.
        slot=next(row['cells'][0] for row in bundle.provenance['mutable_capture_layout'] if any(alias['capture']=='a' for alias in row['aliases']))
        assert abs(out['initial_state_gradients'][0][slot])>1e-5
    net.run(4*dt,namespace={})
    for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
        for name in names:
            cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
