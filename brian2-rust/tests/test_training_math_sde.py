"""Standard math in real Brian SDEs, plastic delayed paths and full BPTT."""
import copy
import os
from unittest.mock import patch
import numpy as np
import pytest
import brian2 as b
from scipy.special import exprel
from brian2_rust import NativeLIFTrainer, lower_brian_dynamic_training
from test_native_training import RUNNER
from test_training_integer_ir import engine
from test_training_linked import cython_cache
from test_training_stochastic import normal
from test_training_standard_math import CASES


def function(kind):
    return exprel if kind=='exprel' else getattr(np,kind)


def model(kind='exprel',method='heun',warm=0.,**options):
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='cython'
    inp=b.SpikeGeneratorGroup(1,[],[]*b.ms,dt=.2*b.ms,name='math_input');groups=[]
    for k in range(2):
        fv=f'{kind}(.2+.1*v)'
        g=b.NeuronGroup(2,f'dv/dt=(gain-v+.02*{fv})/ms+(.01+.001*{fv})*xi/sqrt(ms):1\ngain:1 (constant)',
            threshold=f'v>.6+.01*{fv}',reset=f'v-=.3+.01*{fv}',method=method,dt=.2*b.ms,name=f'math_n{k}')
        g.v=[.73-.03*k,.35+.03*k];g.gain=[.9+.05*k,1.1+.03*k];groups.append(g)
    fz=f'{kind}(.2+.1*z)';fw=f'{kind}(.2+.1*w)';fp=f'{kind}(.2+.1*v_post)'
    syn=b.Synapses(*groups,f'dz/dt=(-z+.1*{fz})/ms+(.005+.001*{fz})*xi/sqrt(ms):1 (clock-driven)\nw:1',
        on_pre=f'v_post+=w*{fw}+.1*z\nz+=.01*{fw}\nw*=.99',on_post=f'w+=.002*{fp}',
        method=method,dt=.2*b.ms,name='math_syn')
    syn.connect(j='i');syn.w=[.1,.12];syn.z=[.15,.2];syn.pre.delay=.2*b.ms
    net=b.Network(inp,*groups,syn)
    if warm:b.seed(109);net.run(warm*b.ms,namespace={})
    bundle=lower_brian_dynamic_training(net,input_group=inp,layers=groups,detach_reset=False,
        trainable_neuron_parameters={g.name:['gain'] for g in groups},
        trainable_synapse_parameters={syn.name:['w','z']},**options)
    return net,groups,syn,bundle


def oracle(bundle,kind,method,weights=None,initial=None,anchors=None,steps=6,sequence=17,batch=0):
    """Independent fixed-normal scalar SDE stages and explicit FIFO recurrence.

    Brian's derivative-free Milstein uses dW**2 (not dW**2-dt). Thresholds
    are relaxed only around the unperturbed trajectory for surrogate VJPs.
    """
    f=function(kind);p=bundle.plan;d=p['dynamic'];weights=bundle.weights if weights is None else weights
    z=np.array(bundle.initial_state if initial is None else initial,float).copy()
    if initial is None:
        for k,ref in enumerate(d['initial_parameters']):
            if ref is not None:z[k]=weights[ref[0]][ref[1]]
    vi=[bundle.provenance['neuron_state_layout'][f'math_n{k}']['v'] for k in range(2)]
    sl=bundle.provenance['dynamic_state_layout']['math_syn'];wi=sl['w'];zi=sl['z']
    banks=[next(e['bank'] for e in bundle.provenance['bindings'] if e['object']==f'math_n{k}' and e['variables']==['gain']) for k in range(2)]
    paths={path['name']:path for path in d['delay_layout']['paths']};before=[];margins=[];spikes=[];draws=[]
    def integrate(x,drive,sigma,noise):
        drift=drive(x);g=sigma(x);dw=np.sqrt(.2)*noise
        if method=='heun':return x+.2*drift+.5*dw*(g+sigma(x+g*dw))
        assert method=='milstein'
        support=x+.2*drift+np.sqrt(.2)*g
        return x+.2*drift+g*dw+.5*np.sqrt(.2)*(sigma(support)-g)*noise**2
    for tick in range(steps):
        if anchors is not None and p.get('tbptt_window') and tick and tick%p['tbptt_window']==0:z=anchors['before'][tick].copy()
        before.append(z.copy())
        for k in range(2):
            noise=np.array([normal(p['seed'],sequence,batch,k,j,tick,0) for j in range(2)]);draws.extend(noise)
            z[vi[k]]=integrate(z[vi[k]],lambda v:np.array(weights[banks[k]])-v+.02*f(.2+.1*v),lambda v:.01+.001*f(.2+.1*v),noise)
        noise=np.array([normal(p['seed'],sequence,batch,2,j,tick,0) for j in range(2)]);draws.extend(noise)
        z[zi]=integrate(z[zi],lambda a:-a+.1*f(.2+.1*a),lambda a:.005+.001*f(.2+.1*a),noise)
        v=np.concatenate([z[row] for row in vi]);margin=v-.6-.01*f(.2+.1*v);s=(margin>0).astype(float)
        for k in range(2):z[bundle.provenance['threshold_margin_layout'][f'math_n{k}']]=margin[2*k:2*k+2]
        if anchors is not None:
            a=anchors['margins'][tick];s=(a>0).astype(float)+p['surrogate']['scale']/(1+p['surrogate']['slope']*abs(a))**2*(margin-a)
        margins.append(margin);spikes.append(s)
        for k,name in enumerate(['math_syn_pre','math_syn_post']):
            path=paths[name];hist=[]
            def deliver(edge,gate):
                if k==0:
                    q=f(.2+.1*z[wi[edge]])
                    z[vi[1][edge]]+=(z[wi[edge]]*q+.1*z[zi[edge]])*gate
                    z[zi[edge]]+=.01*q*gate;z[wi[edge]]*=1-.01*gate
                else:z[wi[edge]]+=.002*f(.2+.1*z[vi[1][edge]])*gate
            for event in path['pending']:
                cells=event['states'];deliver(event['edge'],z[cells[0]]);hist.append((cells,0.))
            for edge in sorted(range(2),key=lambda e:(-len(path['edges'][e]['states']),e)):
                cells=path['edges'][edge]['states'];source=s[2*k+edge]
                deliver(edge,z[cells[0]] if cells else source)
                if cells:hist.append((cells,source))
            for cells,source in hist:z[cells[:-1]]=z[cells[1:]];z[cells[-1]]=source
        for k in range(2):z[vi[k]]-=s[2*k:2*k+2]*(.3+.01*f(.2+.1*z[vi[k]]))
    spikes=np.array(spikes);logits=spikes[:,2:].mean(0)*p['logit_scale'];peak=logits.max()
    loss=peak+np.log(np.exp(logits-peak).sum())-logits[0]
    return loss,z,spikes,dict(before=before,margins=margins,draws=draws)


@pytest.mark.parametrize('kind',list(CASES))
@pytest.mark.parametrize('method',['heun','milstein'])
@pytest.mark.parametrize('warm',[0.,.65])
def test_math_sde_paths_match_fixed_noise_cython(engine,kind,method,warm):
    net,groups,syn,bundle=model(kind,method,warm,backend=engine)
    expected=oracle(bundle,kind,method)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],noise_sequence=17)
    tol=1e-4 if engine!='cpu' else 5e-12
    np.testing.assert_allclose(result['final_state'][0],expected[1],rtol=tol,atol=tol*.01)
    np.testing.assert_array_equal(result['spikes'][0],expected[2]);assert result['loss']==pytest.approx(expected[0],rel=tol)
    net.run(0*b.ms,namespace={});monitors=[b.SpikeMonitor(g) for g in groups];net.add(*monitors)
    device=b.get_device();device.randn_buffer_index[:]=0;calls=[0];draws=expected[3]['draws']
    def refill(n):
        assert n==20000 and calls[0]==0;calls[0]+=1;values=np.zeros(n);values[:len(draws)]=draws;return values
    with patch('numpy.random.randn',refill):net.run((bundle.plan['clock']['origin']+6*.0002-float(net.t))*b.second,namespace={})
    assert calls[0]==1 and device.randn_buffer_index[0]==len(draws);device.randn_buffer_index[:]=0
    actual=np.array(result['final_state'])[0]
    for obj in [*groups,syn]:
        layout=bundle.provenance['neuron_state_layout' if obj in groups else 'dynamic_state_layout'][obj.name]
        for name,indices in layout.items():np.testing.assert_allclose(actual[indices],obj.variables[name].get_value(),rtol=tol,atol=tol*.01)
    spikes=np.zeros((6,4))
    for k,m in enumerate(monitors):
        ticks=np.rint((np.asarray(m.t/b.second)-bundle.plan['clock']['origin'])/.0002).astype(int);spikes[ticks,2*k+np.asarray(m.i)]=1
    np.testing.assert_array_equal(result['spikes'][0],spikes)
    if engine!='cpu':assert result['gpu_dispatches']>0


@pytest.mark.parametrize('kind',list(CASES))
@pytest.mark.parametrize('method',['heun','milstein'])
@pytest.mark.parametrize('window',[None,2])
def test_math_sde_full_weight_initial_vjp(engine,kind,method,window):
    *_,bundle=model(kind,method,.65,backend=engine,tbptt_window=window)
    result=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER).gradients(np.zeros((1,6,1)),[0],noise_sequence=17)
    loss,z,spikes,anchors=oracle(bundle,kind,method);tol=6e-3 if engine!='cpu' else 6e-6
    np.testing.assert_array_equal(result['spikes'][0],spikes);assert result['loss']==pytest.approx(loss,rel=tol)
    for bank,row in enumerate(bundle.weights):
        for j in range(len(row)):
            hi=copy.deepcopy(bundle.weights);lo=copy.deepcopy(bundle.weights);hi[bank][j]+=1e-6;lo[bank][j]-=1e-6
            fd=(oracle(bundle,kind,method,weights=hi,anchors=anchors)[0]-oracle(bundle,kind,method,weights=lo,anchors=anchors)[0])/2e-6
            assert result['gradients'][bank][j]==pytest.approx(fd,rel=tol,abs=tol*.01)
    for j in range(len(bundle.initial_state)):
        if bundle.plan['dynamic']['detached'][j]:continue
        hi=np.array(bundle.initial_state);lo=hi.copy();hi[j]+=1e-6;lo[j]-=1e-6
        fd=(oracle(bundle,kind,method,initial=hi,anchors=anchors)[0]-oracle(bundle,kind,method,initial=lo,anchors=anchors)[0])/2e-6
        assert result['initial_state_gradients'][0][j]==pytest.approx(fd,rel=tol,abs=tol*.01)
    assert any(abs(g)>1e-8 for row in result['gradients'] for g in row)


@pytest.mark.parametrize('method',['heun','milstein'])
@pytest.mark.parametrize('ranks',[2,8])
@pytest.mark.parametrize('window',[None,2])
def test_math_sde_mpi_batch_checkpoint(engine,method,ranks,window,tmp_path):
    if os.environ.get('B2_TEST_MPI')!='1':pytest.skip('local MPI required')
    *_,bundle=model('exprel',method,.65,backend=engine,mpi_ranks=ranks,tbptt_window=window)
    bundle.plan['trainable']=[False]*len(bundle.weights);x=np.zeros((2,6,1));trainer=NativeLIFTrainer(bundle.plan,weights=bundle.weights,runner=RUNNER)
    whole=trainer.gradients(x,[0,1],noise_sequence=17)
    cpu=copy.deepcopy(bundle.plan);cpu.update(backend='cpu',mpi_ranks=None)
    ref=NativeLIFTrainer(cpu,weights=bundle.weights,runner=RUNNER).gradients(x,[0,1],noise_sequence=17)
    tol=6e-3 if engine!='cpu' else 1e-11
    assert 'mpi-dynamic' in whole['numeric_profile']
    assert whole['loss']==pytest.approx(ref['loss'],rel=tol,abs=tol*.001)
    np.testing.assert_array_equal(whole['spikes'],ref['spikes'])
    for key in ('final_state','logits','initial_state_gradients'):
        np.testing.assert_allclose(whole[key],ref[key],rtol=tol,atol=tol*.001)
    for a,c in zip(whole['gradients'],ref['gradients']):np.testing.assert_allclose(a,c,rtol=tol,atol=tol*.001)
    if engine!='cpu':assert whole['gpu_dispatches']>0
    trainer.step(x[:,:2],[0,1],noise_sequence=17);path=tmp_path/'math-sde.json';trainer.store(path)
    restored=NativeLIFTrainer(bundle.plan,runner=RUNNER);restored.restore(path)
    tail=restored.gradients(x[:,2:],[0,1],initial='carry')
    carry_tol=1e-4 if engine!='cpu' else 1e-12
    np.testing.assert_allclose(tail['final_state'],whole['final_state'],rtol=carry_tol,atol=carry_tol*.004)
    np.testing.assert_array_equal(tail['spikes'],np.asarray(whole['spikes'])[:,2:])
