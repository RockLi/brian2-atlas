"""Native checkpoint admission, schedule transformation and real SDK parity."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import brian2 as b
import numpy as np
import pytest

from brian2_atlas import NativeLIFTrainer, lif_training_plan, export_network
from brian2_atlas.neuromorphic import freeze_native_lif, export_native_lif
from brian2_rust.native_lif_conversion import build_native_lif_network, validate_frozen, validate_spike_inputs
from brian2_rust.protocol import canonical_bytes
from brian2_rust.results import load_results

TRAIN=Path(os.environ.get('B2_TRAIN_RUNNER','/nonexistent'))
RUN=Path(os.environ.get('B2_RUNNER','/nonexistent'))
SDK=os.environ.get('B2_SPINNAKER2_PYTHON')


@pytest.fixture
def trained():
    if not TRAIN.is_file():pytest.skip('set B2_TRAIN_RUNNER')
    plan=lif_training_plan([3,4,2],beta=.5,learning_rate=.02,logit_scale=4)
    rng=np.random.default_rng(62)
    trainer=NativeLIFTrainer(plan,runner=TRAIN,weights=[rng.uniform(-.5,1.5,12),rng.uniform(-.5,1.5,8)])
    trainer.step(rng.integers(0,2,(8,16,3)),np.arange(8)%2)
    return trainer


@pytest.fixture(autouse=True)
def brian_settings():
    device=b.get_device();target=b.prefs.codegen.target;optim=b.prefs.codegen.loop_invariant_optimisations
    b.set_device('runtime');b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
    yield
    b.set_device(device);b.prefs.codegen.target=target;b.prefs.codegen.loop_invariant_optimisations=optim


def test_freeze_is_explicit_and_does_not_mutate_training(trained):
    before=copy.deepcopy(trained.state)
    with pytest.raises(ValueError,match='opt into quantization'):
        freeze_native_lif(trained,scales=[8,8])
    frozen=freeze_native_lif(trained,scales=[8,8],quantize=True)
    assert before==trained.state
    assert frozen['quantization']['clipping'] is False
    assert frozen['provenance']['optimizer_step']==1
    for w,q in zip(frozen['weights'],frozen['target_weights']):
        np.testing.assert_array_equal(np.asarray(w)*4,q)
    frozen['weights'][0][0]+=1
    with pytest.raises(ValueError,match='integrity'):validate_frozen(frozen)


@pytest.mark.parametrize('change,message',[
    ('schema','dense v1'),('reset','subtract'),('decay','beta=0.5'),('mask','unmasked'),
    ('extension','without extensions'),('overflow','exceed'),('scale','power of two'),
    ('untrained','trained checkpoint'),('population','budget'),
])
def test_rejects_unsupported_training_contract(trained,change,message):
    scales=[8,8]
    if change=='schema':trained.plan['schema']='b2-lif-training-plan-v2'
    if change=='reset':trained.plan['reset']='zero'
    if change=='decay':trained.plan['beta'][0]=.875
    if change=='mask':trained.plan['masks'][0][0]=0
    if change=='extension':trained.plan['dynamic']={}
    if change=='overflow':scales=[128,128]
    if change=='scale':scales=[7,7]
    if change=='untrained':trained.state['step']=0
    if change=='population':trained.plan['sizes'][0]=257
    with pytest.raises(ValueError,match=message):freeze_native_lif(trained,scales=scales,quantize=True)


def test_spike_input_domain(trained):
    frozen=freeze_native_lif(trained,scales=[8,8],quantize=True)
    for x in (np.zeros((16,3)),np.zeros((1,16,4)),np.full((1,16,3),.5),np.full((1,16,3),np.nan),np.zeros((1,17,3))):
        with pytest.raises(ValueError):validate_spike_inputs(frozen,x)


def test_quantized_native_to_atlas_to_sdk(trained,tmp_path):
    if not RUN.is_file():pytest.skip('set B2_RUNNER')
    f=freeze_native_lif(trained,scales=[8,8],quantize=True)
    x=np.random.default_rng(12).integers(0,2,(3,16,3),dtype=np.uint8)
    x[0]=0;x[1]=1
    native=NativeLIFTrainer(f['plan'],runner=TRAIN,weights=f['weights']).evaluate(x,[0,0,1])
    model=export_network(build_native_lif_network(f,x),16*b.ms,tmp_path/'source.json')
    subprocess.run([str(RUN),str(tmp_path/'source.json'),str(tmp_path/'atlas')],check=True,capture_output=True)
    atlas=load_results(model,tmp_path/'atlas')['populations'][1]
    raster=np.zeros((3,16,6),dtype=np.uint8)
    indices=np.asarray(atlas['indices'],dtype=int);ticks=np.asarray(atlas['spike_ticks'],dtype=int)
    raster[indices//6,ticks,indices%6]=1
    np.testing.assert_array_equal(native['spikes'],raster)
    bundle=export_native_lif(f,tmp_path/'export',steps=16,runner=RUN)
    assert bundle['native_training_bridge']['source_is_zero_input_template']
    assert sum(len(p['connections']) for p in bundle['plan']['projections'])==sum(np.count_nonzero(w) for w in f['target_weights'])
    if not SDK:return
    np.save(tmp_path/'inputs.npy',x)
    runtime=Path(__file__).resolve().parents[1]/'python/brian2_rust/neuromorphic_runtime.py'
    script='''import importlib.util,json,numpy as np,sys
spec=importlib.util.spec_from_file_location('runtime',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
bundle=json.load(open(sys.argv[2]));x=np.load(sys.argv[3]);r=m.run_simulator(bundle,x,batch_size=2)
np.savez(sys.argv[4],spikes=r['spike_raster'],voltage=r['voltage_f64'],final=r['final_v'])
'''
    subprocess.run([SDK,'-c',script,str(runtime),str(tmp_path/'export/converted/bundle.json'),str(tmp_path/'inputs.npy'),str(tmp_path/'sdk.npz')],cwd=tmp_path,
                   env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},check=True,capture_output=True)
    sdk=np.load(tmp_path/'sdk.npz')
    np.testing.assert_array_equal(sdk['spikes'],native['spikes'])
    np.testing.assert_allclose(sdk['voltage'],atlas['trace']['v'].reshape(16,3,6).transpose(1,0,2),atol=1e-12,rtol=0)
