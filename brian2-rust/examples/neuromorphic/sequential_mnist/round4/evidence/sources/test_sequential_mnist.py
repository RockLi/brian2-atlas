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


def test_refined_encodings_and_preprocessing():
    spec = importlib.util.spec_from_file_location('smnist_refined', ROOT/'examples/neuromorphic/sequential_mnist/train.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    image = np.zeros((1,28,28), dtype=np.uint8); image[0,8,12] = 255
    cross = m.encode(image, 'rows-columns-counts')
    assert cross.shape == (1,36,16)
    assert cross[0,4,6] == 4 and cross[0,14+6,4] == 4
    assert cross[:,:,:14].sum() == 8
    interleaved=m.encode(image,'rows-interleaved-counts')
    assert interleaved.shape==(1,64,16) and interleaved[0,16,6]==15
    assert interleaved[:,:,:14].sum()==15
    pixels=np.random.default_rng(4).integers(0,256,(3,28,28),dtype=np.uint8)
    scan=m.encode(pixels,'rows-interleaved-counts')[:,:56,:14]
    restored=scan.reshape(3,28,2,14).transpose(0,1,3,2).reshape(3,28,28)
    np.testing.assert_array_equal(restored,np.rint(pixels.astype(float)*15/255).astype(np.uint8))
    assert m.encode(pixels,'rows-interleaved-counts',repeats=2).shape==(3,120,16)
    with pytest.raises(ValueError):m.encode(pixels,'rows-interleaved-counts',repeats=3)
    full = m.encode(image, 'rows-full-counts')
    assert full.shape == (1,36,16) and full[0,8,6] == 8
    assert full[:,:,:14].sum() == 8
    for encoding in ('rows-counts','rows-columns-counts','rows-full-counts'):
        x = m.encode(image, encoding, repeats=2)
        scan = (x.shape[1]-8)//2
        np.testing.assert_array_equal(x[:,:scan,:14],x[:,scan:2*scan,:14])
        assert (x[:,:,14]==1).all()
        assert not x[:,-8:,:14].any() and (x[:,-8:,15]==1).all()
        assert not x[:,:-8,15].any()
        with pytest.raises(ValueError):m.encode(image, encoding, repeats=10)
    position = m.encode(image,'rows-position-counts',repeats=2,query_ticks=16)
    assert position.shape == (1,44,16)
    np.testing.assert_array_equal(position[0,:28,15],np.tile(np.arange(1,15),2))
    assert (position[0,28:,15]==15).all() and not position[0,28:,:14].any()
    with pytest.raises(ValueError):m.encode(image,'rows-counts',query_ticks=0)
    blank = np.zeros((2,28,28),dtype=np.uint8)
    np.testing.assert_array_equal(m.deskew_images(blank),blank)
    # A slanted line becomes vertical without losing its intensity or centroid.
    diagonal = blank[:1].copy(); diagonal[0,np.arange(8,20),np.arange(8,20)] = 200
    corrected = m.deskew_images(diagonal)
    np.testing.assert_allclose(corrected.sum(),diagonal.sum())
    assert not corrected[:,:,:13].any() and not corrected[:,:,15:].any()
    np.testing.assert_allclose(corrected[0,8:20,13:15],100)
    np.testing.assert_array_equal(m.encode(diagonal,'rows-counts',deskew=True),m.encode(corrected,'rows-counts'))
    # Translation is zero-filled, never a wrap-around, and leaves its source intact.
    edge = np.zeros((1,28,28),dtype=np.uint8);edge[0,0,0]=255;edge[0,12,12]=100
    class FixedOffsets:
        def integers(self,*args,**kwargs):return np.array([[-2,1]])
    shifted = m.translate_images(edge,FixedOffsets(),2)
    assert shifted.sum()==100 and shifted[0,10,13]==100 and edge.sum()==355
    np.testing.assert_array_equal(m.translate_images(edge,np.random.default_rng(4),0),edge)
    np.testing.assert_array_equal(m.translate_images(edge,np.random.default_rng(4),2),m.translate_images(edge,np.random.default_rng(4),2))
    with pytest.raises(ValueError):m.translate_images(edge,np.random.default_rng(4),4)
    # Bounded-memory preprocessing must not change any quantized input count.
    samples=np.random.default_rng(8).integers(0,256,(17,28,28),dtype=np.uint8)
    for encoding in ('rows-counts','rows-position-counts','rows-interleaved-counts','gradient-counts'):
        whole=m.encode(samples,encoding,deskew=True)
        chunks=np.concatenate([m.encode(samples[i:i+3],encoding,deskew=True) for i in range(0,len(samples),3)])
        np.testing.assert_array_equal(whole,chunks)



def test_training_affine_augmentation():
    spec=importlib.util.spec_from_file_location('smnist_affine',ROOT/'examples/neuromorphic/sequential_mnist/train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    yy,xx=np.mgrid[:28,:28]
    ramp=(xx+2*yy)[None].astype(float)
    original=ramp.copy()
    class FixedTransform:
        def uniform(self,low,high,shape):
            return np.array([10.]) if isinstance(shape,int) else np.array([[1.05,.95]])
    transformed=m.affine_images(ramp,FixedTransform(),10)
    angle=np.deg2rad(10.)
    sx=(np.cos(angle)*(xx-13.5)+np.sin(angle)*(yy-13.5))/1.05+13.5
    sy=(-np.sin(angle)*(xx-13.5)+np.cos(angle)*(yy-13.5))/.95+13.5
    interior=(sx>=0)&(sx<27)&(sy>=0)&(sy<27)
    np.testing.assert_allclose(transformed[0][interior],(sx+2*sy)[interior],atol=1e-12)
    outside=(sx<=-1)|(sx>=28)|(sy<=-1)|(sy>=28)
    assert outside.any() and not transformed[0][outside].any()
    np.testing.assert_array_equal(ramp,original)
    identity=m.affine_images(ramp,np.random.default_rng(1),0)
    np.testing.assert_array_equal(identity,ramp)
    assert not np.shares_memory(identity,ramp)
    np.testing.assert_array_equal(m.affine_images(ramp,np.random.default_rng(9),8),m.affine_images(ramp,np.random.default_rng(9),8))
    for degrees in (-1,16,float('nan')):
        with pytest.raises(ValueError):m.affine_images(ramp,np.random.default_rng(1),degrees)


def test_training_elastic_augmentation():
    spec=importlib.util.spec_from_file_location('smnist_elastic',ROOT/'examples/neuromorphic/sequential_mnist/train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    images=np.random.default_rng(77).integers(0,256,(7,28,28),dtype=np.uint8)
    original=images.copy()
    identity=m.elastic_images(images,np.random.default_rng(1),0)
    np.testing.assert_array_equal(identity,images)
    assert not np.shares_memory(identity,images)
    class ConstantField:
        def standard_normal(self,shape):return np.ones(shape)
    np.testing.assert_array_equal(m.elastic_images(images,ConstantField(),1),images)
    whole=m.elastic_images(images,np.random.default_rng(41),.7)
    rng=np.random.default_rng(41)
    chunks=np.concatenate([m.elastic_images(images[i:i+2],rng,.7) for i in range(0,len(images),2)])
    np.testing.assert_array_equal(whole,chunks)
    np.testing.assert_array_equal(images,original)
    assert np.isfinite(whole).all() and whole.min()>=0 and whole.max()<=255
    assert np.count_nonzero(whole!=images)>images.size//2
    # A constant image exposes zero padding: deformed borders cannot wrap.
    border=m.elastic_images(np.full((3,28,28),255),np.random.default_rng(8),2.)
    assert border.min()==0 and np.allclose(border[:,10:18,10:18],255)
    for amplitude,sigma in [(-1,4),(2.1,4),(float('nan'),4),(1,0),(1,9)]:
        with pytest.raises(ValueError):m.elastic_images(images,np.random.default_rng(1),amplitude,sigma)


def test_gradient_cell_encoding():
    spec=importlib.util.spec_from_file_location('smnist_gradient',ROOT/'examples/neuromorphic/sequential_mnist/train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ramp=np.broadcast_to(np.arange(28,dtype=float)*9,(1,28,28)).copy()
    features=m.gradient_cells(ramp)
    assert features.shape==(1,16,14)
    assert (features[:,:,0]==15).all() and not features[:,:,1:8].any()
    assert (features[:,:,9]==15).all() and not features[:,:,10:12].any()
    np.testing.assert_array_equal(features[0,:,12],np.repeat([3,6,9,12],4))
    np.testing.assert_array_equal(features[0,:,13],np.tile([3,6,9,12],4))
    blank=m.gradient_cells(np.zeros((1,28,28)))
    assert not blank[:,:,:12].any()
    vertical=m.gradient_cells(ramp.transpose(0,2,1))
    assert (vertical[:,:,4]==15).all() and not vertical[:,:,:4].any() and not vertical[:,:,5:8].any()
    raster=m.encode(ramp,'gradient-counts',query_ticks=12)
    assert raster.shape==(1,28,16) and raster.max()<=15
    np.testing.assert_array_equal(raster[:,:16,:14],features)
    assert not raster[:,16:,:14].any() and (raster[:,16:,15]==1).all()


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
@pytest.mark.parametrize('readout',['spikes','rate','current','hadamard'])
def test_coded_sdk_parity(coded,tmp_path,target,environment,readout):
    plan=copy.deepcopy(coded.plan);weights=copy.deepcopy(coded.state['weights'])
    if readout=='rate':plan['code_rate_weight']=1.
    if readout=='current':plan['readout_source']='synaptic_current'
    if readout=='hadamard':
        spec=importlib.util.spec_from_file_location('smnist_codes',ROOT/'examples/neuromorphic/sequential_mnist/train.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        plan['sizes'][-1]=8;plan['class_codes']=m.class_codes('hadamard8').tolist()
        weights[1]=np.random.default_rng(12).uniform(-.3,.7,32)
    coded=NativeXyloTrainer(plan,runner=TRAIN,weights=weights)
    sdk=os.environ.get(environment)
    if not RUN.is_file() or not sdk:pytest.skip('set runners and SDK python')
    x=np.random.default_rng(42).integers(0,coded.plan["input_max"]+1,(3,19,3),dtype=np.uint8);y=[7,8,9]
    coded.step(x,y);f=freeze_native_xylo(coded);reference=coded.evaluate(x,y)
    bundle=export_native_xylo(f,tmp_path/'export',steps=19,runner=RUN,target=target)
    assert bundle['native_training_bridge']['class_codes']==coded.plan['class_codes']
    assert bundle['native_training_bridge']['schema']==('atlas.native-xylo-current-bridge.v1' if readout=='current' else 'atlas.native-xylo-frozen.v2')
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


def test_native_warm_weight_averaging(coded,tmp_path):
    spec=importlib.util.spec_from_file_location('smnist_warm',ROOT/'examples/neuromorphic/sequential_mnist/train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    x=np.ones((2,6,3),dtype=int);coded.step(x,[0,1]);paths=[];weights=[]
    for index in range(2):
        if index:coded.step(x,[2,3])
        path=tmp_path/str(index);path.mkdir();coded.store(path/'checkpoint.json')
        frozen=freeze_native_xylo(coded);(path/'frozen.json').write_text(json.dumps(frozen))
        report=dict(frozen_file_sha256=m.sha(path/'frozen.json'),checkpoint_sha256=m.sha(path/'checkpoint.json'),selected=dict(validation_accuracy=.5))
        (path/'training.json').write_text(json.dumps(report));paths.append(path)
        weights.append(copy.deepcopy(coded.state['weights']))
    before=copy.deepcopy(coded.state)
    averaged,receipt=m.warm_weights(paths,coded.plan,TRAIN)
    for a,w1,w2 in zip(averaged,*weights):np.testing.assert_array_equal(a,(np.asarray(w1)+np.asarray(w2))/2)
    assert receipt['kind']=='averaged-warm-start' and not receipt['optimizer_moments_preserved']
    assert coded.state==before
    different=copy.deepcopy(coded.plan);different['sizes']=[3,5,5]
    with pytest.raises(ValueError,match='incompatible warm-start sizes'):m.warm_weights(paths,different,TRAIN)
    with pytest.raises(ValueError,match='int8 range'):m.warm_weights(paths,coded.plan,TRAIN,1000)
    (paths[0]/'checkpoint.json').write_text('{}')
    with pytest.raises(AssertionError):m.warm_weights(paths,coded.plan,TRAIN)


def test_grouped_widening_and_projected_optimizer(coded):
    spec=importlib.util.spec_from_file_location('smnist_groups',ROOT/'examples/neuromorphic/sequential_mnist/train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    coded.step(np.ones((2,6,3),dtype=int),[0,1]);f=freeze_native_xylo(coded)
    plan=copy.deepcopy(coded.plan);plan['sizes']=[3,8,5]
    weights=m.widen_weights(f,plan,2,0.,np.random.default_rng(4))
    source=[np.asarray(w)/plan['weight_scale'] for w in f['target_weights']]
    np.testing.assert_array_equal(weights[0].reshape(3,8),np.tile(source[0].reshape(3,4),(1,2)))
    np.testing.assert_array_equal(weights[1].reshape(8,5),np.tile(source[1].reshape(4,5)/2,(2,1)))
    scaled=copy.deepcopy(plan);scaled['threshold'][1]*=2
    precise=m.widen_weights(f,scaled,2,0.,np.random.default_rng(4))
    np.testing.assert_array_equal(precise[1].reshape(8,5),np.tile(source[1].reshape(4,5),(2,1)))
    mask=m.recurrent_group_mask(8,2)
    np.testing.assert_array_equal(weights[2].reshape(8,8)[:4,:4],source[2].reshape(4,4))
    assert not weights[2].reshape(8,8)[~mask].any()
    trainer=NativeXyloTrainer(plan,runner=TRAIN,weights=weights)
    for _ in range(3):
        trainer.step(np.ones((2,7,3),dtype=int),[2,4]);m.project_recurrence(trainer,mask)
        for key in ('weights','first_moment','second_moment'):
            assert not np.asarray(trainer.state[key][2]).reshape(8,8)[~mask].any()
    frozen=freeze_native_xylo(trainer)
    assert not np.asarray(frozen['target_weights'][2]).reshape(8,8)[~mask].any()
    assert m.recurrent_group_mask(126,2).sum(1).max()==63
    assert m.recurrent_group_mask(128,4).sum(1).max()==32
    for hidden,groups in ((64,1),(128,2),(129,3),(127,2),(0,1),(8,0)):
        with pytest.raises(ValueError):m.recurrent_group_mask(hidden,groups)


@pytest.mark.parametrize('target,environment',[('xylosim','B2_XYLO_PYTHON'),('spinnaker2','B2_SPINNAKER2_PYTHON')])
def test_wide_native_sdk_boundary(tmp_path,target,environment):
    sdk=os.environ.get(environment)
    if not TRAIN.is_file() or not RUN.is_file() or not sdk:pytest.skip('set native runners and SDK environments')
    spec=importlib.util.spec_from_file_location('smnist_wide',ROOT/'examples/neuromorphic/sequential_mnist/train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    plan=xylo_recurrent_training_plan([2,128,5],thresholds=(4,9),weight_scale=8.,readout_start=3,class_codes=CODES,input_max=15)
    rng=np.random.default_rng(211);mask=m.recurrent_group_mask(128,4)
    weights=[rng.integers(-3,5,(2,128)).ravel()/8,rng.integers(-3,5,(128,5)).ravel()/8,(rng.integers(-2,3,(128,128))*mask).ravel()/8]
    trainer=NativeXyloTrainer(plan,runner=TRAIN,weights=weights)
    x=rng.integers(0,16,(2,17,2));y=[1,4]
    trainer.step(x,y);m.project_recurrence(trainer,mask)
    frozen=freeze_native_xylo(trainer);ref=trainer.evaluate(x,y)
    export_native_xylo(frozen,tmp_path/'export',steps=17,runner=RUN,target=target)
    np.savez_compressed(tmp_path/'inputs.npz',spikes=x)
    subprocess.run([sdk,str(ROOT/'examples/neuromorphic/sequential_mnist/sdk_worker.py'),
        '--runtime',str(ROOT/'python/brian2_rust/neuromorphic_runtime.py'),
        '--bundle',str(tmp_path/'export/converted/bundle.json'),
        '--inputs',str(tmp_path/'inputs.npz'),'--output',str(tmp_path/'sdk')],
        cwd=tmp_path,env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},check=True,capture_output=True)
    with np.load(tmp_path/'sdk/00000.npz') as actual:
        for key in ('spikes','currents','membranes','logits'):np.testing.assert_array_equal(ref[key],actual[key])
    assert all(check['differences']==0 for check in json.loads((tmp_path/'sdk/execution.json').read_text())['chunk_checks'])
    for schema,hidden in (('atlas-xylo-training-plan-v1',65),('atlas-xylo-training-plan-v2',129)):
        bad=copy.deepcopy(plan);bad['schema']=schema;bad['sizes']=[2,hidden,5]
        if schema.endswith('v1'):
            for key in ('recurrent','readout_start','class_codes'):bad.pop(key)
        with pytest.raises(ValueError):NativeXyloTrainer(bad,runner=TRAIN).evaluate(np.zeros((1,4,2)),[0])


@pytest.mark.parametrize('weight',[0.,2.5])
def test_rate_objective_surrogate_and_forward_contract(coded,tmp_path,weight):
    x=np.random.default_rng(73).integers(0,coded.plan["input_max"]+1,(2,9,3));y=[2,9]
    before=copy.deepcopy(coded.state);weights=before['weights']
    plan=copy.deepcopy(coded.plan);plan['code_rate_weight']=weight
    regularized=NativeXyloTrainer(plan,runner=TRAIN,weights=weights)
    original=coded.gradients(x,y);result=regularized.gradients(x,y)
    assert regularized.state==before
    for key in ('spikes','logits','final_membrane'):
        assert result[key]==original[key]
    np.testing.assert_allclose(result['loss'],_linearized_loss(plan,weights,weights,x,y),atol=1e-12)
    rng=np.random.default_rng(21)
    for _ in range(3):
        direction=[rng.normal(size=len(w)) for w in weights];eps=1e-5
        plus=[np.asarray(w)+eps*d for w,d in zip(weights,direction)]
        minus=[np.asarray(w)-eps*d for w,d in zip(weights,direction)]
        finite=(_linearized_loss(plan,weights,plus,x,y)-_linearized_loss(plan,weights,minus,x,y))/(2*eps)
        reverse=sum(np.dot(g,d) for g,d in zip(result['gradients'],direction))
        np.testing.assert_allclose(reverse,finite,rtol=1e-5,atol=1e-7)
    if weight==0:
        assert result['loss']==original['loss'] and result['gradients']==original['gradients']
    else:
        assert result['loss']>original['loss'] and result['gradients']!=original['gradients']
    regularized.step(x,y);f=freeze_native_xylo(regularized)
    assert validate_frozen(f)==plan['sizes']
    full=regularized.evaluate(x,y);chunk=regularized.evaluate_chunks(x,y,chunk_steps=2)
    for key in ('logits','spikes','currents','membranes','continuation'):assert full[key]==chunk[key]
    assert full['loss']==chunk['loss']
    regularized.store(tmp_path/'state.json')
    restored=NativeXyloTrainer(plan,runner=TRAIN);restored.restore(tmp_path/'state.json')
    assert restored.step(x,y)==regularized.step(x,y)
    for invalid in (-1,101):
        request=dict(plan={**plan,'code_rate_weight':invalid},state=before,
            operation='train',inputs=x.tolist(),labels=y,initial=None)
        with pytest.raises(ValueError):regularized._run_request(request)
    request['plan']={**plan};del request['plan']['class_codes']
    with pytest.raises(ValueError):regularized._run_request(request)
    for invalid in (-1,101,float('nan'),True):
        with pytest.raises(ValueError):xylo_recurrent_training_plan([3,4,5],class_codes=CODES,code_rate_weight=invalid)
    with pytest.raises(ValueError):xylo_recurrent_training_plan([3,4,5],code_rate_weight=1)
    bad=copy.deepcopy(f);bad['plan']['code_rate_weight']=-1
    bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError):validate_frozen(bad)


def test_readout_transfer_and_cosine_schedule(coded):
    spec=importlib.util.spec_from_file_location('smnist_round4',ROOT/'examples/neuromorphic/sequential_mnist/train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    x=np.random.default_rng(21).integers(0,coded.plan["input_max"]+1,(2,9,3));y=[2,9]
    coded.step(x,y);f=freeze_native_xylo(coded)
    plan=copy.deepcopy(coded.plan);plan['sizes'][-1]=8;plan['class_codes']=m.class_codes('hadamard8').tolist()
    initial=m.recode_weights(f,plan)
    assert [len(w) for w in initial]==[12,32,16]
    for k in (0,2):np.testing.assert_array_equal(initial[k],np.asarray(f['target_weights'][k])/plan['weight_scale'])
    # Changing class labels changes the initialization, so class order is explicit.
    permuted=copy.deepcopy(plan);permuted['class_codes']=plan['class_codes'][::-1]
    assert not np.array_equal(initial[1],m.recode_weights(f,permuted)[1])
    bad=copy.deepcopy(plan);bad['threshold'][0]+=1
    with pytest.raises(ValueError):m.recode_weights(f,bad)
    baseline=copy.deepcopy(coded.state)
    changed=m.scheduled_trainer(coded,.0002)
    assert changed.state==baseline and coded.state==baseline
    assert changed.plan['optimizer']['learning_rate']==.0002
    assert coded.plan['optimizer']['learning_rate']!=.0002
    changed.step(x,y)
    assert changed.state['step']==baseline['step']+1 and coded.state==baseline
    rates=[m.cosine_rate(k,101,.001,.1) for k in range(1,102)]
    np.testing.assert_allclose([rates[0],rates[-1]],[.001,.0001])
    assert np.all(np.diff(rates)<0)
    for args in ((0,10,.1,.1),(1,10,.1,0),(11,10,.1,.1)):
        with pytest.raises(ValueError):m.cosine_rate(*args)


def test_current_readout_gradients_and_history(coded,tmp_path):
    from brian2_rust.xylo_readout import decode_readout
    plan=copy.deepcopy(coded.plan);plan['readout_source']='synaptic_current'
    weights=copy.deepcopy(coded.state['weights'])
    model=NativeXyloTrainer(plan,runner=TRAIN,weights=weights)
    x=np.random.default_rng(43).integers(0,plan['input_max']+1,(2,11,3));y=[1,8]
    result=model.gradients(x,y)
    np.testing.assert_allclose(result['loss'],_linearized_loss(plan,weights,weights,x,y),rtol=1e-12,atol=1e-12)
    rng=np.random.default_rng(32)
    for bank in range(3):
        direction=[np.zeros(len(w)) for w in weights];direction[bank]=rng.normal(size=len(weights[bank]));eps=1e-6
        plus=[np.asarray(w)+eps*d for w,d in zip(weights,direction)]
        minus=[np.asarray(w)-eps*d for w,d in zip(weights,direction)]
        finite=(_linearized_loss(plan,weights,plus,x,y)-_linearized_loss(plan,weights,minus,x,y))/(2*eps)
        np.testing.assert_allclose(sum(np.dot(g,d) for g,d in zip(result['gradients'],direction)),finite,rtol=1e-5,atol=1e-7)
    before=coded.evaluate(x,y);after=model.evaluate(x,y)
    for key in ('spikes','currents','membranes'):assert before[key]==after[key]
    assert before['logits']!=after['logits']
    sums=np.asarray(after['currents'])[:,plan['readout_start']:,-5:].sum(1).astype(np.int64)
    bridge={**plan,'readout_divisor':plan['threshold'][1]}
    np.testing.assert_array_equal(after['continuation']['current_sums'],sums)
    np.testing.assert_array_equal(after['logits'],decode_readout(after['continuation']['counts'],sums,bridge,9))
    for width in (1,3,7):
        chunk=model.evaluate_chunks(x,y,chunk_steps=width)
        for key in ('logits','spikes','currents','membranes','continuation','loss'):assert after[key]==chunk[key]
    prefix=model.evaluate(x[:,:3],y)['continuation']
    for invalid in (None,[[32768*129]*5]*2,[[0]*4]*2):
        state=copy.deepcopy(prefix);state['current_sums']=invalid
        with pytest.raises(ValueError):model.evaluate(x[:,3:],y,initial=state)
    model.step(x,y);f=freeze_native_xylo(model);assert validate_frozen(f)==plan['sizes']
    model.store(tmp_path/'current.json')
    restored=NativeXyloTrainer(plan,runner=TRAIN);restored.restore(tmp_path/'current.json')
    assert restored.step(x,y)==model.step(x,y)
    for source in ('membrane','spikes',''):
        with pytest.raises(ValueError):xylo_recurrent_training_plan([3,4,5],readout_source=source)
    with pytest.raises(ValueError):xylo_recurrent_training_plan([3,4,5],readout_source='synaptic_current',class_codes=CODES,code_rate_weight=0)
    for invalid in (0,32768,True):
        with pytest.raises(ValueError):decode_readout([[0]*5],[[0]*5],{**bridge,'readout_divisor':invalid},9)
    # Signed current sums must retain sign; casting to unsigned would invert the ranking.
    signed=np.array([[-100,-50,0,50,100]])
    np.testing.assert_array_equal(decode_readout([[0]*5],signed,bridge,9),
        (signed @ np.asarray(CODES).T)*plan['logit_scale']/9/plan['threshold'][1])


def test_current_readout_bridge_threshold_contract():
    from brian2_rust.xylo_readout import validate_readout_contract
    bridge=dict(schema='atlas.native-xylo-current-bridge.v1',source_schema='atlas.native-xylo-frozen.v2',dynamics_equivalent=True,
        readout_source='synaptic_current',readout_divisor=64)
    validate_readout_contract(bridge,[64]*5)
    validate_readout_contract({},[64]*5)
    for key,value in [('schema','atlas.native-xylo-frozen.v1'),('schema','atlas.native-xylo-frozen.v2'),('source_schema','wrong'),('dynamics_equivalent',False),
            ('readout_source','voltage'),('readout_divisor',32),('readout_divisor',True)]:
        with pytest.raises(ValueError):validate_readout_contract({**bridge,key:value},[64]*5)
    with pytest.raises(ValueError):validate_readout_contract(bridge,[64,32])
    with pytest.raises(ValueError):validate_readout_contract(bridge,[])
    with pytest.raises(ValueError):validate_readout_contract({'readout_divisor':64},[64]*5)

    missing={**bridge};del missing['readout_source']
    with pytest.raises(ValueError):validate_readout_contract(missing,[64]*5)
