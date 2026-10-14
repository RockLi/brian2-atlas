"""Independent STP all-state BPTT, including updated-u transmission and time."""
import copy
import brian2 as b
import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer,lower_brian_dynamic_training
from test_training_brian_dynamic import network
from test_training_integer_ir import engine
from test_training_poisson_zero_vjp import mpi
from test_native_training import RUNNER


def model(backend,method,detach,window,ranks):
    net,source,layers,static,old,_,_,x=network();net.remove(old);dt=.2*b.ms
    syn=b.Synapses(*layers,'''w:1
        du/dt=(U-u)/tauf:1 (clock-driven)
        dr/dt=(1-r)/taur:1 (clock-driven)
        released=w*u*r:1
        U:1 (shared,constant)
        tauf:second (shared,constant)
        taur:second (shared,constant)''',on_pre='u+=U*(1-u)\nv_post+=released*(1+.1*v_pre)+.02*sin(t/ms)\nr*=1-u',dt=dt,method=method,name='stp_vjp')
    syn.connect(i=[1,0,1,0],j=[0,1,1,0]);syn.w=[.7,.8,.9,1.];syn.u=[.1,.2,.3,.4];syn.r=[.8,.7,.9,.6];syn.U=.25;syn.tauf=2*b.ms;syn.taur=3*b.ms;net.add(syn)
    bundle=lower_brian_dynamic_training(net,input_group=source,layers=layers,backend=backend,mpi_ranks=ranks,detach_reset=detach,tbptt_window=window,trainable_synapse_parameters={syn.name:['w','U','tauf','taur']})
    return net,source,layers,static,syn,dt,bundle,x


def reference(data,method,weights,initial=None,anchors=None):
    _,source,layers,static,syn,dt,bundle,x=data;p=bundle.plan;z=np.array(bundle.initial_state if initial is None else initial,float)
    if initial is None:
        for cell,ref in enumerate(p['dynamic']['initial_parameters']):
            if ref is not None:z[cell]=weights[ref[0]][ref[1]]
    def parameter(owner,name):
        row=next(row for row in bundle.provenance['bindings'] if row['object']==owner.name and name in row['variables'])
        return np.asarray(weights[row['bank']]) if row['kind'] not in ('synapse_scalar','neuron_scalar') and len(row['variables'])==1 and len(weights[row['bank']])>1 else weights[row['bank']][row['variables'].index(name)]
    neurons=bundle.provenance['neuron_state_layout'];v=[neurons[g.name]['v'] for g in layers];states=bundle.provenance['dynamic_state_layout'][syn.name];u=states['u'];r=states['r'];sources=np.asarray(syn.i[:]);targets=np.asarray(syn.j[:]);wi=np.asarray(static.i[:]);wj=np.asarray(static.j[:]);before=[];margins=[];hard=[];events=[];seconds=float(dt)
    def relax(value,target,tau):
        h=seconds/tau
        # Independent closed form/stability polynomial for a linear relaxer.
        factor=np.exp(-h) if method=='exact' else 1-h if method=='euler' else 1-h+h*h/2 if method=='rk2' else 1-h+h*h/2-h**3/6+h**4/24
        return target+(value-target)*factor
    for tick,inp in enumerate(x):
        if anchors is not None and p['tbptt_window'] and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy());z[v[0]]*=.85;z[v[1]]*=.85
        U=parameter(syn,'U');z[u]=relax(z[u],U,parameter(syn,'tauf'));z[r]=relax(z[r],1.,parameter(syn,'taur'))
        values=np.r_[z[v[0]],z[v[1]]];margin=values-1.;event=(margin>0).astype(float);margins.append(margin.copy());hard.append(event.copy())
        if anchors is not None:
            old=anchors['margins'][tick];event=anchors['hard'][tick]+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(old))**2*(margin-old)
        events.append(event.copy())
        static_w=parameter(static,'w')
        for edge,(i,j) in enumerate(zip(wi,wj)):z[v[0][j]]+=inp[i]*static_w[edge]
        w=parameter(syn,'w')
        for edge,(i,j) in enumerate(zip(sources,targets)):
            cells=[u[edge],v[1][j],r[edge]];old=z[cells].copy();new=old.copy()
            new[0]+=U*(1-new[0])
            new[1]+=w[edge]*new[0]*new[2]*(1+.1*z[v[0][i]])+.02*np.sin(tick*seconds/.001)
            new[2]*=1-new[0]
            z[cells]=old+event[i]*(new-old)
        reset=hard[-1] if p['detach_reset'] else event
        z[v[0]]-=reset[:2];z[v[1]]-=reset[2:]
    spikes=np.array(events);logits=spikes[:,2:].mean(0)*p['logit_scale'];loss=np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0]
    return loss,z,spikes,dict(before=before,margins=margins,hard=hard)


@pytest.mark.parametrize('method',['euler','rk2','rk4','exact'])
@pytest.mark.parametrize('detach',[False,True])
@pytest.mark.parametrize('window',[None,3])
@pytest.mark.parametrize('ranks',[None,2])
def test_stp_all_bank_all_initial_vjps(engine,method,detach,window,ranks):
    mpi(ranks);data=model(engine,method,detach,window,ranks);bundle=data[6];x=data[7];loss,z,spikes,anchors=reference(data,method,bundle.weights)
    out=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300).gradients(x[None],[0]);assert out['loss']==pytest.approx(loss,abs=8e-6);np.testing.assert_array_equal(out['spikes'][0],spikes)
    visible=[*bundle.provenance['neuron_state_layout'][data[2][0].name]['v'],*bundle.provenance['neuron_state_layout'][data[2][1].name]['v'],*bundle.provenance['dynamic_state_layout'][data[4].name]['u'],*bundle.provenance['dynamic_state_layout'][data[4].name]['r']]
    np.testing.assert_allclose(np.asarray(out['final_state'])[0,visible],z[visible],rtol=7e-5,atol=7e-6)
    for bank,row in enumerate(bundle.weights):
        for index,value in enumerate(row):
            eps=1e-6*max(abs(value),1e-3);hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][index]+=eps;lo[bank][index]-=eps
            fd=(reference(data,method,hi,anchors=anchors)[0]-reference(data,method,lo,anchors=anchors)[0])/(2*eps)
            assert out['gradients'][bank][index]==pytest.approx(fd,rel=3e-3,abs=3e-5),(method,bank,index)
    initial=np.array(bundle.initial_state,float);explicit=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER,request_timeout=300).gradients(x[None],[0],initial=initial[None]);_,_,_,anchors=reference(data,method,bundle.weights,initial)
    for index,value in enumerate(initial):
        if bundle.plan['dynamic']['detached'][index] or index in bundle.plan['dynamic'].get('integer_states',[]):
            assert explicit['initial_state_gradients'][0][index]==0.;continue
        eps=1e-6*max(abs(value),1e-3);hi=initial.copy();lo=initial.copy();hi[index]+=eps;lo[index]-=eps
        fd=(reference(data,method,bundle.weights,hi,anchors)[0]-reference(data,method,bundle.weights,lo,anchors)[0])/(2*eps)
        assert explicit['initial_state_gradients'][0][index]==pytest.approx(fd,rel=3e-3,abs=3e-5),(method,index)
    assert (out['gpu_dispatches']>0)==(engine!='cpu')


@pytest.mark.parametrize('method',['euler','rk2','rk4','exact'])
@pytest.mark.parametrize('ranks',[None,2])
def test_stp_original_split_restore(engine,method,ranks,tmp_path):
    mpi(ranks);data=model(engine,method,False,None,ranks);net,source,layers,static,syn,dt,bundle,x=data;p=copy.deepcopy(bundle.plan);p['trainable']=[False]*len(bundle.weights)
    whole=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER,request_timeout=300).evaluate(x[None],[0]);trainer=NativeLIFTrainer(p,weights=bundle.weights,runner=RUNNER,request_timeout=300);prefix=trainer.step(x[None,:5],[0]);path=tmp_path/'stp';trainer.store(path);trainer=NativeLIFTrainer(p,runner=RUNNER,request_timeout=300);trainer.restore(path);tail=trainer.step(x[None,5:],[0],initial='carry')
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=7e-5,atol=7e-6);np.testing.assert_array_equal(np.concatenate([prefix['spikes'],tail['spikes']],axis=1),whole['spikes'])
    monitors=[b.SpikeMonitor(g) for g in layers];net.add(*monitors);net.run(len(x)*dt,namespace={});observed=np.zeros((len(x),4))
    for layer,monitor in enumerate(monitors):observed[np.rint(np.asarray(monitor.t/b.second)/float(dt)).astype(int),2*layer+np.asarray(monitor.i)]=1.
    np.testing.assert_array_equal(whole['spikes'][0],observed)
    for group in layers:
        cells=bundle.provenance['neuron_state_layout'][group.name]['v'];np.testing.assert_allclose(np.asarray(whole['final_state'])[0,cells],group.v[:],rtol=7e-5,atol=7e-6)
    for name in ('u','r'):
        cells=bundle.provenance['dynamic_state_layout'][syn.name][name];np.testing.assert_allclose(np.asarray(whole['final_state'])[0,cells],syn.variables[name].get_value(),rtol=7e-5,atol=7e-6)
    assert (whole['gpu_dispatches']>0)==(engine!='cpu')
