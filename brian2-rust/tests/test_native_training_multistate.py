"""Coupled-state VJP oracle uses closed-form dynamics and local smooth spikes."""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from brian2_rust import NativeLIFTrainer, lif_training_plan, compile_training_equation, neuron_parameter_bank
from test_native_training import RUNNER
from test_native_training_graph import fixture_graph


def fixture(reset='subtract',detach=False,window=None):
    old,w,x,y,v=fixture_graph(reset,detach,window)
    names=[['v','a'],['v','a','c']]
    updates=[];resets=[]
    for l,states in enumerate(names):
        parameters=dict(tau=(4,l),kick=(4,2),theta=(4,3+l))
        def compile(expr):return compile_training_equation(expr,parameters=parameters,states=states)
        updates.append([compile('v+.2*(-v/tau+.3*a'+('+.1*c' if l else '')+')'),compile('a+.2*(.15*v-.4*a)')])
        resets.append([compile('0' if reset=='zero' else 'v-theta'),compile('a+kick+.1*v')])
        if l:
            updates[-1].append(compile('c+.2*(.1*a-.3*c)'))
            # Brian a+=kick+.1*v; v-=theta; c=.9*c+.05*a
            resets[-1].append(compile('.9*c+.05*(a+kick+.1*v)'))
    plan=lif_training_plan([2,2,2],projections=old['projections']+[neuron_parameter_bank(5)],
        state_equations=updates,state_resets=resets,threshold=[1.0625,1.03125],
        threshold_parameters=[[4,3],[4,4]],reset=reset,detach_reset=detach,tbptt_window=window,
        surrogate_slope=2,learning_rate=.001)
    initial=np.concatenate([v[:,:2],np.array([[.1,.2],[.3,.4]]),v[:,2:],
                            np.array([[.2,.1],[.4,.3]]),np.array([[.05,.1],[.15,.2]])],axis=1)
    return plan,w+[[1.7,2.1,.07,1.0625,1.03125]],x,y,initial


def oracle(p,w,x,y,initial,anchors=None):
    live=initial.copy();batch,time,_=x.shape
    offsets=[0,4];counts=[2,3];theta=np.repeat(w[4][3:],2)
    matrices=[]
    for projection,row in zip(p['projections'][:4],w):
        mat=np.zeros((2,2));np.add.at(mat,(projection['sources'],projection['targets']),np.array(row)[projection['parameter_ids']]);matrices.append(mat)
    pre=[];spikes=[];starts=[]
    for t in range(time):
        if anchors is not None and p['tbptt_window'] and t>0 and t%p['tbptt_window']==0:
            live=anchors[2][:,t].copy()
        starts.append(live.copy());u=live.copy()
        for l,offset in enumerate(offsets):
            v=live[:,offset:offset+2];a=live[:,offset+2:offset+4]
            u[:,offset:offset+2]=v+.2*(-v/w[4][l]+.3*a+(.1*live[:,8:10] if l else 0))
            u[:,offset+2:offset+4]=a+.2*(.15*v-.4*a)
            if l:u[:,8:10]=live[:,8:10]+.2*(.1*a-.3*live[:,8:10])
        uv=np.concatenate([u[:,:2],u[:,4:6]],axis=1);spike=(uv>theta).astype(float);reset_spike=spike
        if anchors is not None:
            phi=p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(anchors[0][:,t]-anchors[3]))**2
            spike=anchors[1][:,t]+phi*(uv-theta-anchors[0][:,t]+anchors[3])
            reset_spike=anchors[1][:,t] if p['detach_reset'] else spike
        for projection,mat in zip(p['projections'],matrices):
            src=projection['source_layer'];dst=projection['target_layer'];offset=offsets[dst-1]
            u[:,offset:offset+2]+=(x[:,t] if src==0 else spike[:,2*(src-1):2*src])@mat
        reset_values=u.copy()
        for l,offset in enumerate(offsets):
            reset_values[:,offset+2:offset+4]+=w[4][2]+.1*u[:,offset:offset+2]
            reset_values[:,offset:offset+2]=0 if p['reset']=='zero' else u[:,offset:offset+2]-w[4][3+l]
            if l:reset_values[:,8:10]=.9*u[:,8:10]+.05*reset_values[:,6:8]
        gate=np.concatenate([reset_spike[:,:2]]*2+[reset_spike[:,2:]]*3,axis=1)
        live=u+gate*(reset_values-u)
        pre.append(uv);spikes.append(spike)
    spikes=np.stack(spikes,axis=1);logits=spikes[:,:,2:].mean(axis=1)*p['logit_scale']
    maximum=logits.max(axis=1);loss=np.mean(maximum+np.log(np.exp(logits-maximum[:,None]).sum(axis=1))-logits[np.arange(batch),y])
    return loss,spikes,live,(np.stack(pre,axis=1),spikes,np.stack(starts,axis=1),theta.copy())


@pytest.mark.parametrize('reset',['zero','subtract'])
@pytest.mark.parametrize('detach',[True,False])
@pytest.mark.parametrize('window',[None,3])
def test_multistate_independent_vjp(reset,detach,window):
    p,w,x,y,initial=fixture(reset,detach,window)
    result=NativeLIFTrainer(p,runner=RUNNER,weights=w).gradients(x,y,initial=initial)
    loss,spikes,live,anchors=oracle(p,w,x,y,initial)
    assert result['loss']==pytest.approx(loss,abs=3e-15)
    np.testing.assert_array_equal(result['spikes'],spikes)
    np.testing.assert_allclose(result['final_state'],live,rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(result['final_membrane'],np.array(result['final_state'])[:,[0,1,4,5]])
    assert np.max(abs(np.array(result['initial_state_gradients'])[:,[2,3,6,7,8,9]]))>1e-7
    eps=1e-6
    for bank,row in enumerate(w):
        for i in range(len(row)):
            plus=copy.deepcopy(w);minus=copy.deepcopy(w);plus[bank][i]+=eps;minus[bank][i]-=eps
            fd=(oracle(p,plus,x,y,initial,anchors)[0]-oracle(p,minus,x,y,initial,anchors)[0])/(2*eps)
            assert result['gradients'][bank][i]==pytest.approx(fd,abs=6e-8,rel=5e-5)
    for b in range(2):
        for i in range(initial.shape[1]):
            plus=initial.copy();minus=initial.copy();plus[b,i]+=eps;minus[b,i]-=eps
            fd=(oracle(p,w,x,y,plus,anchors)[0]-oracle(p,w,x,y,minus,anchors)[0])/(2*eps)
            assert result['initial_state_gradients'][b][i]==pytest.approx(fd,abs=6e-8,rel=5e-5)


@pytest.mark.parametrize('ranks',[2,4,8])
@pytest.mark.parametrize('detach,window',[(False,None),(True,3)])
def test_multistate_target_owned_mpi(ranks,detach,window):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('MPI required')
    p,w,x,y,initial=fixture(detach=detach,window=window)
    serial=NativeLIFTrainer(p,runner=RUNNER,weights=w);p['mpi_ranks']=ranks
    distributed=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    for step in range(2):
        a=serial.step(x,y,initial=initial if step==0 else 'carry')
        b=distributed.step(x,y,initial=initial if step==0 else 'carry')
        np.testing.assert_array_equal(a['spikes'],b['spikes'])
        for key in ['final_state','initial_state_gradients','logits']:
            np.testing.assert_allclose(a[key],b[key],atol=3e-14,rtol=3e-14)
        for key in ['weights','first_moment','second_moment']:
            for u,v in zip(a['state'][key],b['state'][key]):np.testing.assert_allclose(u,v,atol=3e-14,rtol=3e-14)
        for u,v in zip(a['gradients'],b['gradients']):np.testing.assert_allclose(u,v,atol=3e-14,rtol=3e-14)


def test_multistate_fresh_process_checkpoint(tmp_path):
    p,w,x,y,initial=fixture();trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    trainer.step(x,y,initial=initial);checkpoint=tmp_path/'checkpoint';trainer.store(checkpoint)
    source=tmp_path/'request';source.write_text(json.dumps(dict(plan=p,x=x.tolist(),y=y)))
    dest=tmp_path/'result'
    code='''import json,sys
from brian2_rust import NativeLIFTrainer
p=json.load(open(sys.argv[1]));t=NativeLIFTrainer(p['plan'],runner=sys.argv[4]);t.restore(sys.argv[2])
json.dump(t.step(p['x'],p['y'],initial='carry'),open(sys.argv[3],'w'))
'''
    subprocess.run([sys.executable,'-c',code,str(source),str(checkpoint),str(dest),str(RUNNER)],check=True,timeout=30)
    assert json.loads(dest.read_text())==trainer.step(x,y,initial='carry')
    assert len(trainer.neuron_state[0])==10


def test_multistate_validation_and_failure_atomicity():
    p,w,x,y,initial=fixture();trainer=NativeLIFTrainer(p,runner=RUNNER,weights=w)
    before=copy.deepcopy(trainer.state)
    with pytest.raises(ValueError,match='initial membrane'):trainer.step(x,y,initial=initial[:,:4])
    assert trainer.state==before
    for mutate,match in [
        (lambda p:p['state_equations'][0][0].__setitem__(0,dict(op='state',index=2)),'SSA reference'),
        (lambda p:p['state_resets'][0].pop(),'state updates and resets'),
        (lambda p:p.update(max_tape_bytes=10000),'tape budget'),
        (lambda p:p.update(backend='invalid'),'unsupported native training backend'),
    ]:
        bad=copy.deepcopy(p);mutate(bad)
        # Direct CLI avoids trying to compile an unsupported GPU frontend.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            request=Path(d)/'request';output=Path(d)/'output'
            request.write_text(json.dumps(dict(plan=bad,state=before,operation='train',inputs=x.tolist(),labels=y,initial=initial.tolist())))
            run=subprocess.run([str(RUNNER),str(request),str(output)],capture_output=True,text=True,timeout=10)
            assert run.returncode!=0 and match in run.stderr
            assert not output.exists()
    bad=copy.deepcopy(p);bad['state_resets'][0][1]=compile_training_equation('log(-1)',states=['v','a'])
    broken=NativeLIFTrainer(bad,runner=RUNNER,weights=w)
    with pytest.raises(ValueError,match='domain error'):broken.step(x,y,initial=initial)
    assert broken.state==before
