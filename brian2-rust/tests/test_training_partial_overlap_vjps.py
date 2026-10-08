"""Overlapping offset views of live voltage retain one nonzero initial adjoint."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_event_captures import event_capture,readonly_event_capture
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(mode,readonly_first,delay,ranks,backend):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy';dt=.2*b.ms
    source=b.SpikeGeneratorGroup(2,[0,1,0],np.array([0,0,2])*dt,dt=dt)
    hidden=b.NeuronGroup(1,'dv/dt=0/second:1',threshold='v>100',reset='v=0',method='euler',dt=dt)
    g=b.NeuronGroup(3,'dv/dt=u/ms:1\nu:1',threshold='v>.5',reset='v-=.5',method='euler',dt=dt)
    g.v=[.217,.419,.719];g.u=[.017,.023,.031]
    parent=g.variables['v'].get_value();a=parent[:2];c=parent[1:].view();c.flags.writeable=False
    mutable=b.Function(event_capture(a),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    readonly=b.Function(readonly_event_capture(c),arg_units=[1],arg_names=['x'],return_unit=1,stateless=False)
    code='h=f('+('v_post' if mode=='scalar' else 'h')+');h=q(h)'+(';v_post+=gain*h' if mode=='scalar' else ';u_post+=gain*h' if mode=='vectorised' else '')
    syn=b.Synapses(source,g,'h:1\ngain:1 (constant)',on_pre=code,dt=dt,
        namespace=dict(f=readonly if readonly_first else mutable,q=mutable if readonly_first else readonly))
    syn.connect(i=[0,1],j=[0,1] if mode=='array' else [0,0]);syn.h=[.113,.173];syn.gain=[.11,.19];syn.delay=delay*dt
    net=b.Network(source,hidden,g,syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=[hidden,g],backend=backend,mpi_ranks=ranks,detach_reset=False,
        trainable_synapse_parameters={syn.name:['h','gain']})
    x=np.zeros((1,4,2));x[0,0,:]=1;x[0,2,0]=1
    return net,g,syn,dt,bundle,x


def oracle(data,mode,readonly_first,weights,window,initial=None,anchors=None):
    _,g,syn,_,bundle,_=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    layout=bundle.provenance['neuron_state_layout'][g.name];v=layout['v'];u=layout['u'];h=bundle.provenance['dynamic_state_layout'][syn.name]['h']
    gain=next(row['bank'] for row in bundle.provenance['bindings'] if row['object']==syn.name and 'gain' in row['variables'])
    mutable=next(row['cells'] for row in bundle.provenance['mutable_capture_layout'] if any(not alias['readonly'] for alias in row['aliases']))
    paths=[syn.pre.name] if mode=='scalar' else bundle.provenance['event_callback_stage_groups'][syn.pre.name][1:]
    delay=bundle.provenance['delay_queues'][paths[0]]['new'][0]['delay'];before=[];margins=[];hard=[];spikes=[]
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
                old=z.copy();new_h=.49*old[v[0]];z[mutable]+=gate*(-.2*old[mutable])
                z[h[edge]]=old[h[edge]]+gate*(new_h-old[h[edge]])
                z[v[0]]=old[v[0]]+gate*weights[gain][edge]*new_h
        else:
            for stage,path in enumerate(paths):
                rows=bundle.provenance['delay_queues'][path]['new'];old_h=z[h].copy()
                if stage<2:
                    if selected and (len(paths)==1 or stage==int(readonly_first)):z[mutable]*=.8
                    scale=.49 if len(paths)==1 else .7
                    for edge in range(2):
                        gate=z[rows[edge]['states'][0]] if delay else float(edge in selected)
                        z[h[edge]]+=gate*(scale*old_h[edge]-z[h[edge]])
                if mode=='vectorised' and (len(paths)==1 or stage==2):
                    for edge in range(2):
                        gate=z[rows[edge]['states'][0]] if delay else float(edge in selected)
                        z[u[0]]+=gate*weights[gain][edge]*(.49*old_h[edge] if len(paths)==1 else old_h[edge])
        for path in bundle.provenance['delay_queues'].values():
            for row in path['new']:
                if row['states']:
                    cells=row['states'];z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=float(row['edge'] in ([0,1] if tick==0 else [0] if tick==2 else []))
        z[v]-=.5*event
    logits=np.array(spikes).mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('mode',['array','vectorised','scalar'])
@pytest.mark.parametrize('readonly_first',[False,True])
@pytest.mark.parametrize('delay',[0,1])
@pytest.mark.parametrize('window',[None,2])
@pytest.mark.parametrize('ranks',[None,2])
def test_overlap_voltage_all_bank_initial_vjps(engine,mode,readonly_first,delay,window,ranks):
    mpi(ranks);data=model(mode,readonly_first,delay,ranks,engine);net,g,syn,dt,bundle,x=data
    p=copy.deepcopy(bundle.plan);p['tbptt_window']=window
    out=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER).gradients(x,[0]);loss,z,anchors=oracle(data,mode,readonly_first,bundle.weights,window)
    cells=[*bundle.provenance['neuron_state_layout'][g.name]['v'],*bundle.provenance['neuron_state_layout'][g.name]['u'],*bundle.provenance['dynamic_state_layout'][syn.name]['h']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],z[cells],rtol=8e-5,atol=8e-6)
    assert out['loss']==pytest.approx(loss,abs=8e-6)
    for bank,row in enumerate(bundle.weights):
        for index in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=1e-6;lo[bank][index]-=1e-6
            fd=(oracle(data,mode,readonly_first,hi,window,anchors=anchors)[0]-oracle(data,mode,readonly_first,lo,window,anchors=anchors)[0])/2e-6
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),(bank,index)
    for index in range(len(bundle.initial_state)):
        if p['dynamic']['detached'][index] or index in p['dynamic']['integer_states']:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[index]+=1e-6;lo[index]-=1e-6
        fd=(oracle(data,mode,readonly_first,bundle.weights,window,hi,anchors)[0]-oracle(data,mode,readonly_first,bundle.weights,window,lo,anchors)[0])/2e-6
        assert out['initial_state_gradients'][0][index]==pytest.approx(fd,rel=1e-3,abs=1e-5),index
    slot=bundle.provenance['neuron_state_layout'][g.name]['v'][1];assert abs(out['initial_state_gradients'][0][slot])>1e-5
    net.run(4*dt,namespace={})
    for obj,key,names in [(g,'neuron_state_layout',['v','u']),(syn,'dynamic_state_layout',['h'])]:
        for name in names:
            cells=bundle.provenance[key][obj.name][name];np.testing.assert_allclose(np.asarray(out['final_state'])[0,cells],obj.variables[name].get_value(),rtol=8e-5,atol=8e-6)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')
