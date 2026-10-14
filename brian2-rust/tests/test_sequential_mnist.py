"""Class-code training derivatives, input order, and exact SDK conversion."""
import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from brian2_atlas import NativeXyloTrainer, xylo_recurrent_training_plan
from brian2_atlas.neuromorphic import freeze_native_xylo, export_native_xylo
from brian2_rust.native_xylo_conversion import validate_frozen, digest
from brian2_rust.xylo_readout import decode_counts, validate_codes
from test_native_xylo_training import _linearized_loss, brian_settings

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path(os.environ.get('B2_TRAIN_RUNNER', '/nonexistent'))
RUN = Path(os.environ.get('B2_RUNNER', '/nonexistent'))
CODES = [[int(j in pair) for j in range(5)] for pair in itertools.combinations(range(5), 2)]


def test_codebook_and_encoding():
    np.testing.assert_array_equal(np.argmax(decode_counts(CODES, {'class_codes':CODES, 'logit_scale':3.}, 2), 1), np.arange(10))
    for invalid in ([[0,0],[1,0]], [[1,0],[1,0]], [[1,0],[1,1]], [[1,.5],[.5,1]]):
        with pytest.raises(ValueError): validate_codes(invalid, 2)
    spec = importlib.util.spec_from_file_location('smnist_train', ROOT/'examples/neuromorphic/sequential_mnist/train.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    h=m.class_codes('hadamard8')
    assert h.shape==(10,8) and (h.sum(1)==4).all()
    distances=np.count_nonzero(h[:,None,:]!=h[None,:,:],axis=2)
    assert distances[np.triu_indices(10,1)].min()==4
    np.testing.assert_array_equal(np.argmax(decode_counts(h,{'class_codes':h,'logit_scale':1.},1),1),np.arange(10))
    images = np.zeros((1,28,28), np.uint8); images[0,8,13] = 255
    raster = m.encode(images, 'blocks')
    assert raster.shape == (1,57,16) and raster.sum() == 1
    assert raster[0,2*7+3,1] == 1
    rows = m.encode(np.full((1,28,28),255,np.uint8), 'rows')
    assert rows.shape == (1,64,14) and rows[:,:56].all() and not rows[:,56:].any()
    spec=importlib.util.spec_from_file_location('smnist_worker', ROOT/'examples/neuromorphic/sequential_mnist/sdk_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    np.testing.assert_array_equal(worker.integer_trace([-32768.,0.,32767.]),[-32768,0,32767])
    for invalid in ([.5], [np.nan], [32768.]):
        with pytest.raises(AssertionError):worker.integer_trace(invalid)
    count_rows=m.encode(np.full((1,28,28),255,np.uint8),'rows-counts')
    assert count_rows.shape==(1,22,16) and (count_rows[:,:14,:14]==15).all()
    assert not count_rows[:,14:,:14].any() and (count_rows[:,:,14]==1).all()
    assert m.encode(images, 'rows', 2).shape == (1,120,14)
    with pytest.raises(ValueError):m.encode(images, 'blocks', 3)


@pytest.fixture(params=[1,15])
def coded(request):
    if not TRAIN.is_file():pytest.skip('set B2_TRAIN_RUNNER')
    plan = xylo_recurrent_training_plan([3,4,5], thresholds=(4,4), weight_scale=8.,
        readout_start=2, class_codes=CODES, input_max=request.param)
    rng = np.random.default_rng(41)
    return NativeXyloTrainer(plan, runner=TRAIN, weights=[rng.uniform(-.3,1.,12),
        rng.uniform(-.3,.7,20), (np.eye(4)*.5 + rng.uniform(-.1,.1,(4,4))).ravel()])


def test_coded_gradient_chunks_and_export_contract(coded):
    x = np.random.default_rng(6).integers(0,coded.plan["input_max"]+1,(2,8,3)); y = [8,9]
    result = coded.gradients(x,y); weights = coded.state['weights']
    np.testing.assert_allclose(_linearized_loss(coded.plan,weights,weights,x,y), result['loss'], atol=1e-12)
    rng = np.random.default_rng(82)
    for bank in range(3):
        direction = [np.zeros(len(w)) for w in weights]; direction[bank] = rng.normal(size=len(weights[bank]))
        eps=1e-6
        plus=[np.asarray(w)+eps*d for w,d in zip(weights,direction)]
        minus=[np.asarray(w)-eps*d for w,d in zip(weights,direction)]
        finite=(_linearized_loss(coded.plan,weights,plus,x,y)-_linearized_loss(coded.plan,weights,minus,x,y))/(2*eps)
        np.testing.assert_allclose(sum(np.dot(g,d) for g,d in zip(result['gradients'],direction)),finite,rtol=1e-5,atol=1e-7)
    coded.step(x,y); full=coded.evaluate(x,y); chunk=coded.evaluate_chunks(x,y,chunk_steps=3)
    assert np.asarray(full['logits']).shape == (2,10)
    for key in ('logits','spikes','currents','membranes','continuation'):assert full[key] == chunk[key]
    counts=np.asarray(full['spikes'])[:,2:,-5:].sum(1)
    np.testing.assert_array_equal(full['logits'],decode_counts(counts,coded.plan,6))
    f=freeze_native_xylo(coded); assert validate_frozen(f)==[3,4,5]
    bad=copy.deepcopy(f);bad['plan']['class_codes'][1]=bad['plan']['class_codes'][0]
    bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError):validate_frozen(bad)
    request=dict(plan=copy.deepcopy(coded.plan),state=coded.state,operation='evaluate',inputs=x.tolist(),labels=y,initial=None)
    request['plan']['class_codes'][1]=request['plan']['class_codes'][0]
    with pytest.raises(ValueError):coded._run_request(request)
    for invalid in (0,16):
        request['plan']=copy.deepcopy(coded.plan);request['plan']['input_max']=invalid
        with pytest.raises(ValueError):coded._run_request(request)
    request['plan']=copy.deepcopy(coded.plan);request['inputs'][0][0][0]=coded.plan['input_max']+1
    with pytest.raises(ValueError):coded._run_request(request)


@pytest.mark.parametrize('target,environment',[('xylosim','B2_XYLO_PYTHON'),('spinnaker2','B2_SPINNAKER2_PYTHON')])
def test_coded_sdk_parity(coded,tmp_path,target,environment):
    sdk=os.environ.get(environment)
    if not RUN.is_file() or not sdk:pytest.skip('set runners and SDK python')
    x=np.random.default_rng(42).integers(0,coded.plan["input_max"]+1,(3,19,3),dtype=np.uint8);y=[7,8,9]
    coded.step(x,y);f=freeze_native_xylo(coded);reference=coded.evaluate(x,y)
    bundle=export_native_xylo(f,tmp_path/'export',steps=19,runner=RUN,target=target)
    assert bundle['native_training_bridge']['class_codes']==CODES
    np.save(tmp_path/'inputs.npy',x)
    script="""import importlib.util,json,numpy as np,sys
s=importlib.util.spec_from_file_location('runtime',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
b=json.load(open(sys.argv[2]));x=np.load(sys.argv[3]);full=m.run_simulator(b,x)
r=m.run_simulator(b,x,chunk_steps=7)
for k in ('spike_raster','voltage','hidden_I','output_I','logits'):np.testing.assert_array_equal(full[k],r[k])
if b['target']=='xylosim':
 session=m._module('xylo_runtime').XyloInferenceSession(b)
 for i,row in enumerate(x):
  session.reset();session.push(row[:6]);part=session.push(row[6:])
  np.testing.assert_array_equal(part['logits'],full['logits'][i])
np.savez(sys.argv[4],**{k:v for k,v in full.items() if isinstance(v,np.ndarray)})
"""
    subprocess.run([sdk,'-c',script,str(ROOT/'python/brian2_rust/neuromorphic_runtime.py'),
        str(tmp_path/'export/converted/bundle.json'),str(tmp_path/'inputs.npy'),str(tmp_path/'results.npz')],
        cwd=tmp_path,env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},check=True,capture_output=True)
    r=np.load(tmp_path/'results.npz')
    for field,key in [('spikes','spike_raster'),('membranes','voltage'),('logits','logits')]:
        np.testing.assert_array_equal(reference[field],r[key])
    np.testing.assert_array_equal(reference['currents'],np.concatenate([r['hidden_I'],r['output_I']],2))
