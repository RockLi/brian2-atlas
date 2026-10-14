"""Explicit adaptation, unchanged weights, native replay and stock Xylo tests."""
import copy
import os
from pathlib import Path
import subprocess

import brian2 as b
import numpy as np
import pytest
from brian2_atlas import NativeLIFTrainer, lif_training_plan, export_network
from brian2_atlas.neuromorphic import freeze_native_lif, export_native_lif, run_simulator
from brian2_rust.native_lif_xylo import build_network
from brian2_rust.protocol import canonical_bytes
from brian2_rust.results import load_results

TRAIN=Path(os.environ.get('B2_TRAIN_RUNNER','/nonexistent'))
RUN=Path(os.environ.get('B2_RUNNER','/nonexistent'))
SDK=os.environ.get('B2_XYLO_PYTHON')

@pytest.fixture
def frozen():
    if not TRAIN.is_file() or not RUN.is_file():
        pytest.skip('set B2_TRAIN_RUNNER and B2_RUNNER')
    rng=np.random.default_rng(13)
    plan=lif_training_plan([3,4,2],beta=.5,learning_rate=.01)
    trainer=NativeLIFTrainer(plan,runner=TRAIN,
        weights=[rng.uniform(-.5,1.5,12),rng.uniform(-.5,1.5,8)])
    trainer.step(rng.integers(0,2,(4,16,3)),[0,1,0,1])
    return freeze_native_lif(trainer,scales=[8,8],quantize=True)

@pytest.fixture(autouse=True)
def settings():
    device=b.get_device();target=b.prefs.codegen.target;optim=b.prefs.codegen.loop_invariant_optimisations
    b.set_device('runtime');b.prefs.codegen.target='numpy';b.prefs.codegen.loop_invariant_optimisations=False
    yield
    b.set_device(device);b.prefs.codegen.target=target;b.prefs.codegen.loop_invariant_optimisations=optim


def test_xylo_requires_explicit_semantic_opt_in(frozen,tmp_path):
    before=copy.deepcopy(frozen)
    with pytest.raises(ValueError,match='changes neuron dynamics'):
        export_native_lif(frozen,tmp_path/'no',steps=16,runner=RUN,target='xylosim')
    assert not (tmp_path/'no').exists()
    for value in (1, 'yes', None):
        with pytest.raises(ValueError,match='must be boolean'):
            export_native_lif(frozen,tmp_path/'bad',steps=16,runner=RUN,target='xylosim',allow_dynamics_approximation=value)
    bundle=export_native_lif(frozen,tmp_path/'yes',steps=16,runner=RUN,target='xylosim',allow_dynamics_approximation=True)
    assert frozen==before
    assert bundle['native_training_bridge']['dynamics_equivalent'] is False
    assert bundle['native_training_bridge']['frozen_sha256']==frozen['sha256']
    assert bundle['plan']['sizes']==dict(input=3,hidden=4,output=2)
    for key,bank,shape in zip(('input','readout'),frozen['target_weights'],((3,4),(4,2))):
        np.testing.assert_array_equal(bundle['plan']['weights'][key],np.asarray(bank).reshape(shape))
    assert (tmp_path/'yes/frozen.json').read_bytes()==canonical_bytes(frozen)


@pytest.mark.parametrize('sizes,threshold,message',[
    ([17,4,2],1.,'input<=16'),([3,65,2],1.,'input<=16'),
    ([3,4,9],1.,'input<=16'),([3,4,4,2],1.,'input<=16'),
    ([3,4,2],.0625,'exact integers'),([3,4,2],8192.,'exact integers'),
])
def test_xylo_rejects_unrepresentable_models(sizes,threshold,message,tmp_path):
    if not TRAIN.is_file():pytest.skip('set B2_TRAIN_RUNNER')
    plan=lif_training_plan(sizes,beta=.5,threshold=threshold)
    trainer=NativeLIFTrainer(plan,runner=TRAIN,weights=[np.ones(a*z) for a,z in zip(sizes,sizes[1:])])
    trainer.step(np.ones((1,16,sizes[0])),[0])
    f=freeze_native_lif(trainer,scales=[8]*(len(sizes)-1),quantize=True)
    with pytest.raises(ValueError,match=message):
        export_native_lif(f,tmp_path/'export',steps=16,runner=RUN,target='xylosim',allow_dynamics_approximation=True)
    assert not (tmp_path/'export').exists()


def test_native_xylo_stock_sdk_and_reset(frozen,tmp_path):
    if not SDK:pytest.skip('set B2_XYLO_PYTHON for actual SDK test')
    bundle=export_native_lif(frozen,tmp_path/'export',steps=16,runner=RUN,target='xylosim',allow_dynamics_approximation=True)
    x=np.random.default_rng(87).integers(0,2,(4,16,3),dtype=np.uint8)
    x[0]=1;x[1]=0;x[2]=x[0];x[3]=0;x[3,-1]=1
    arrays={key:[] for key in ('hidden_I','hidden_v','hidden_q','output_I','output_v','output_q')}
    for i,raster in enumerate(x):
        model=export_network(build_network(frozen,raster),16*b.ms,tmp_path/f'source-{i}.json')
        subprocess.run([str(RUN),str(tmp_path/f'source-{i}.json'),str(tmp_path/f'result-{i}')],check=True,capture_output=True)
        result=load_results(model,tmp_path/f'result-{i}')
        for role in ('hidden','output'):
            trace=result['populations'][bundle['plan']['roles'][role]]['trace']
            for key in ('I','v','q'):arrays[role+'_'+key].append(trace[key])
    np.save(tmp_path/'inputs.npy',x)
    runtime=Path(__file__).resolve().parents[1]/'python/brian2_rust/neuromorphic_runtime.py'
    script="""import importlib.util,json,numpy as np,sys
spec=importlib.util.spec_from_file_location('runtime',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
bundle=json.load(open(sys.argv[2]));x=np.load(sys.argv[3]);r=m.run_simulator(bundle,x)
assert r['configuration_valid']
separate=m.run_simulator(bundle,x[2:])
for key in ('spike_raster','voltage','counts'):
 np.testing.assert_array_equal(r[key][2:],separate[key])
np.savez(sys.argv[4],**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
"""
    subprocess.run([SDK,'-c',script,str(runtime),str(tmp_path/'export/converted/bundle.json'),str(tmp_path/'inputs.npy'),str(tmp_path/'sdk.npz')],
        cwd=tmp_path,env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},check=True,capture_output=True)
    sdk=np.load(tmp_path/'sdk.npz')
    for key,values in arrays.items():np.testing.assert_array_equal(sdk[key],np.stack(values))
    assert sdk['spike_raster'][0].sum()>0
    assert sdk['spike_raster'][1].sum()==0 and np.all(sdk['voltage'][1]==0)
    np.testing.assert_array_equal(sdk['spike_raster'][0],sdk['spike_raster'][2])


def test_native_xylo_runtime_rejects_bad_input_before_sdk(frozen,tmp_path):
    bundle=export_native_lif(frozen,tmp_path/'export',steps=16,runner=RUN,target='xylosim',allow_dynamics_approximation=True)
    for x in (np.zeros((16,3)),np.zeros((0,16,3)),np.ones((1,17,3)),np.full((1,16,3),2),np.full((1,16,3),np.nan)):
        with pytest.raises(ValueError,match='binary batch'):
            run_simulator(bundle,x)
    bundle['native_training_bridge']['dynamics_equivalent']=True
    with pytest.raises(ValueError,match='contract'):run_simulator(bundle,np.zeros((1,16,3)))
