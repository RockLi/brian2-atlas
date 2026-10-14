"""Recurrent native BPTT, explicit continuation and stock SDK streaming parity."""
import copy
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import brian2 as b
from brian2_atlas import NativeXyloTrainer,xylo_recurrent_training_plan,export_network
from brian2_atlas.neuromorphic import freeze_native_xylo,export_native_xylo
from brian2_rust.native_xylo_conversion import build_native_xylo_network
from brian2_rust.results import load_results
from test_native_xylo_training import _linearized_loss,brian_settings

TRAIN=Path(os.environ.get('B2_TRAIN_RUNNER','/nonexistent'))
RUN=Path(os.environ.get('B2_RUNNER','/nonexistent'))
SDK=os.environ.get('B2_XYLO_PYTHON')

@pytest.fixture
def recurrent():
    if not TRAIN.is_file():pytest.skip('set B2_TRAIN_RUNNER')
    p=xylo_recurrent_training_plan([3,4,2],readout_start=5,thresholds=(4,4),weight_scale=8.)
    rng=np.random.default_rng(81)
    return NativeXyloTrainer(p,runner=TRAIN,weights=[rng.uniform(-.5,1.,12),rng.uniform(-.5,1.,8),
        (np.eye(4)*.5+rng.uniform(-.2,.2,(4,4))).ravel()])


def test_recurrent_surrogate_gradient_and_checkpoint(recurrent,tmp_path):
    x=np.random.default_rng(14).integers(0,2,(2,16,3));y=[0,1]
    before=copy.deepcopy(recurrent.state);r=recurrent.gradients(x,y)
    assert all(np.linalg.norm(g)>0 for g in r['gradients'])
    rng=np.random.default_rng(5)
    for bank in range(3):
        direction=[np.zeros(len(w)) for w in before['weights']];direction[bank]=rng.normal(size=len(direction[bank]))
        eps=1e-6
        plus=[np.asarray(w)+eps*d for w,d in zip(before['weights'],direction)]
        minus=[np.asarray(w)-eps*d for w,d in zip(before['weights'],direction)]
        finite=(_linearized_loss(recurrent.plan,before['weights'],plus,x,y)-
                _linearized_loss(recurrent.plan,before['weights'],minus,x,y))/(2*eps)
        np.testing.assert_allclose(sum(np.dot(g,d) for g,d in zip(r['gradients'],direction)),finite,rtol=1e-5,atol=1e-7)
    assert recurrent.state==before
    recurrent.step(x,y)
    assert all(np.any(np.asarray(w)!=old) for w,old in zip(recurrent.state['weights'],before['weights']))
    recurrent.store(tmp_path/'checkpoint.json')
    restored=NativeXyloTrainer(recurrent.plan,runner=TRAIN);restored.restore(tmp_path/'checkpoint.json')
    assert restored.step(x,y)==recurrent.step(x,y)


def test_native_irregular_chunks_and_invalid_continuation(recurrent):
    x=np.random.default_rng(3).integers(0,2,(3,23,3));y=[0,1,0]
    state=copy.deepcopy(recurrent.state);full=recurrent.evaluate(x,y)
    for width in (1,2,5,7,23):
        chunks=recurrent.evaluate_chunks(x,y,chunk_steps=width)
        for key in ('spikes','currents','membranes','logits','continuation','final_membrane','loss'):
            assert chunks[key]==full[key]
    assert recurrent.state==state
    prefix=recurrent.evaluate(x[:,:4],y);assert not np.any(prefix['logits'])
    saved=json.loads(json.dumps(prefix['continuation']))
    # A new Python trainer/native process resumes the explicit forward state.
    fresh=NativeXyloTrainer(recurrent.plan,runner=TRAIN,weights=state['weights'])
    tail=fresh.evaluate(x[:,4:],y,initial=saved)
    assert tail['continuation']==full['continuation']
    with pytest.raises(ValueError,match='evaluation-only'):recurrent.step(x[:,4:],y,initial=saved)
    with pytest.raises(ValueError,match='evaluation-only'):recurrent.gradients(x[:,4:],y,initial=saved)
    for key,value in [('elapsed_ticks',129),('model_sha256','wrong'),('spikes',[[32]*6]*3),
                      ('currents',[[32768]*6]*3),('membranes',[[0]*5]*3),('counts',[[1,0]]*3)]:
        bad=copy.deepcopy(saved);bad[key]=value
        with pytest.raises(ValueError):recurrent.evaluate(x[:,4:],y,initial=bad)
    with pytest.raises(ValueError):recurrent.evaluate(x[:,4:],[0,1],initial=saved)
    changed=copy.deepcopy(state['weights']);changed[0][0]+=1/recurrent.plan['weight_scale']
    other=NativeXyloTrainer(recurrent.plan,runner=TRAIN,weights=changed)
    with pytest.raises(ValueError,match='identity'):other.evaluate(x[:,4:],y,initial=saved)
    assert recurrent.state==state


@pytest.mark.parametrize('case',['learned','mixed_saturation'])
def test_recurrent_atlas_sdk_and_sessions(recurrent,tmp_path,case):
    if not RUN.is_file():pytest.skip('set B2_RUNNER')
    if case=='mixed_saturation':
        p=xylo_recurrent_training_plan([16,63,2],readout_start=3,thresholds=(1,1),weight_scale=2.)
        wi=np.full((16,63),127.);wo=np.full((63,2),127.);wo[:,1]=-128.
        rec=np.full((63,63),127.);rec[:16,::2]=-128.;rec[32:,1::2]=-128.
        recurrent=NativeXyloTrainer(p,runner=TRAIN,weights=[wi.ravel()/2,wo.ravel()/2,rec.ravel()/2])
    sizes=recurrent.plan['sizes'];steps=23
    x=np.random.default_rng(37).integers(0,2,(3,steps,sizes[0]),dtype=np.uint8);x[0]=1;x[1]=0;x[2]=x[0]
    recurrent.step(x if case=='learned' else np.zeros_like(x),[0,1,0])
    frozen=freeze_native_xylo(recurrent);full=recurrent.evaluate(x,[0,1,0])
    bundle=export_native_xylo(frozen,tmp_path/'export',steps=steps,runner=RUN)
    assert frozen['schema']=='atlas.native-xylo-frozen.v2' and bundle['native_training_bridge']['stateful_chunks']
    np.testing.assert_array_equal(bundle['plan']['weights']['recurrent'],np.asarray(frozen['target_weights'][2]).reshape(sizes[1],sizes[1]))
    for index in range(len(x)):
        model=export_network(build_native_xylo_network(frozen,x[index]),steps*b.ms,tmp_path/f'{index}.json')
        subprocess.run([str(RUN),str(tmp_path/f'{index}.json'),str(tmp_path/f'replay-{index}')],check=True,capture_output=True)
        results=load_results(model,tmp_path/f'replay-{index}')
        for role,sl in [('hidden',slice(0,sizes[1])),('output',slice(sizes[1],sum(sizes[1:])) )]:
            trace=results['populations'][bundle['plan']['roles'][role]]['trace']
            for key,field in [('I','currents'),('v','membranes'),('q','spikes')]:
                np.testing.assert_array_equal(trace[key],np.asarray(full[field])[index,:,sl])
    if not SDK:pytest.skip('AtlasIR checked; set B2_XYLO_PYTHON for SDK/session checks')
    runtime=Path(__file__).resolve().parents[1]/'python/brian2_rust/neuromorphic_runtime.py'
    np.save(tmp_path/'inputs.npy',x)
    script="""import importlib.util,json,numpy as np,sys
spec=importlib.util.spec_from_file_location('runtime',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
b=json.load(open(sys.argv[2]));x=np.load(sys.argv[3]);full=m.run_simulator(b,x)
for width in (1,5,7,23):
 r=m.run_simulator(b,x,chunk_steps=width)
 for k,v in full.items():
  if isinstance(v,np.ndarray):np.testing.assert_array_equal(v,r[k])
s=m._module('xylo_runtime').XyloInferenceSession(b)
for index in (0,1,2):
 s.reset();chunks=[];start=0
 for invalid in (x[index,:0],np.full_like(x[index,:1],2)):
  try:s.push(invalid)
  except ValueError:pass
  else:raise AssertionError('invalid chunk accepted')
  assert s.elapsed_ticks==0
 for end in (1,4,5,13,23):
  chunk=s.push(x[index,start:end]);chunks.append(chunk)
  read=b['native_training_bridge']['readout_start'];ticks=max(0,end-read)
  np.testing.assert_array_equal(chunk['counts'],full['output_q'][index,read:end].sum(0))
  np.testing.assert_array_equal(chunk['logits'],chunk['counts']*b['native_training_bridge']['logit_scale']/max(1,ticks))
  start=end
 for k in ('hidden_I','hidden_v','hidden_q','output_I','output_v','output_q'):
  np.testing.assert_array_equal(np.concatenate([c[k] for c in chunks]),full[k][index])
 try:s.push(x[index,:1])
 except ValueError:pass
 else:raise AssertionError('horizon not enforced')
np.savez(sys.argv[4],**{k:v for k,v in full.items() if isinstance(v,np.ndarray)})
"""
    sdk_run=subprocess.run([SDK,'-c',script,str(runtime),str(tmp_path/'export/converted/bundle.json'),str(tmp_path/'inputs.npy'),str(tmp_path/'sdk.npz')],
        cwd=tmp_path,env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},check=False,capture_output=True,text=True)
    assert sdk_run.returncode==0,sdk_run.stderr
    sdk=np.load(tmp_path/'sdk.npz')
    for key,field in [('spike_raster','spikes'),('voltage','membranes'),('logits','logits')]:np.testing.assert_array_equal(sdk[key],full[field])
    np.testing.assert_array_equal(np.concatenate((sdk['hidden_I'],sdk['output_I']),axis=2),full['currents'])


def test_recurrent_fanout_rejected_before_sdk(tmp_path):
    if not TRAIN.is_file():pytest.skip('set B2_TRAIN_RUNNER')
    p=xylo_recurrent_training_plan([1,64,1])
    t=NativeXyloTrainer(p,runner=TRAIN,weights=[np.ones(64)/32,np.ones(64)/32,np.ones(4096)/32])
    t.step(np.zeros((1,1,1)),[0])
    with pytest.raises(ValueError,match='fanout'):freeze_native_xylo(t)
    from brian2_rust.xylo_network import build_xylo_network
    from brian2_rust.export import lower_network
    from brian2_rust.xylo_b2ir import _lower_validated
    net=build_xylo_network([1,64,1],[np.ones(64),np.ones(64),np.ones(4096)],[32,32],np.zeros((1,1)))
    with pytest.raises(ValueError,match='fanout'):_lower_validated(lower_network(net,b.ms))


@pytest.mark.parametrize('change',['missing_recurrence','missing_window','v1_extra_fields','readout_outside_horizon','empty_readout','bank_shape'])
def test_recurrent_plan_protocol_rejections(recurrent,change):
    request=dict(plan=copy.deepcopy(recurrent.plan),state=copy.deepcopy(recurrent.state),operation='train',
                 inputs=np.zeros((1,8,3)).tolist(),labels=[0],initial=None)
    if change=='missing_recurrence':del request['plan']['recurrent']
    if change=='missing_window':del request['plan']['readout_start']
    if change=='v1_extra_fields':request['plan']['schema']='atlas-xylo-training-plan-v1'
    if change=='readout_outside_horizon':request['plan']['readout_start']=128
    if change=='empty_readout':request['plan']['readout_start']=8
    if change=='bank_shape':request['state']['weights'][2].pop()
    original=copy.deepcopy(recurrent.state)
    with pytest.raises(ValueError):recurrent._run_request(request)
    assert recurrent.state==original
