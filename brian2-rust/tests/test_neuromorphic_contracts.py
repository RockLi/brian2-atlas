"""Public conversion routing and rejection before SDK allocation or execution."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from brian2_atlas.neuromorphic import TARGETS, PROFILES, convert_ir, run_simulator
from brian2_rust import xylo_runtime, spinnaker2_integer_runtime


SUPPORTED = {
    'dense-int8-v1': {'akida', 'lava', 'spinnaker2'},
    'binary-count-v1': {'akida', 'lava', 'specksim', 'spinnaker2', 'xylosim'},
    'spinnaker2-lif-v1': {'spinnaker2'},
    'xylo-audio2-temporal-v1': {'xylosim'},
    'dynapse-dpi-v1': {'dynapse'},
    'spinnaker2-xylo-integer-v1': {'spinnaker2'},
}


@pytest.mark.parametrize('profile,target', [
    (p,t) for p in PROFILES for t in TARGETS if t not in SUPPORTED[p]
])
def test_rejected_pair_does_not_read_source_or_start_runner(tmp_path, profile, target):
    with pytest.raises(ValueError, match='only supported by|requires binary-count-v1|requires dynapse-dpi-v1'):
        convert_ir(tmp_path/'absent.json', target=target, profile=profile, runner=tmp_path/'absent-runner')


def test_public_conversion_import_does_not_load_vendor_sdks(tmp_path):
    code = '''import sys
from brian2_atlas.neuromorphic import convert_ir, export_ir, run_simulator, freeze_native_lif, export_native_lif, freeze_native_xylo, export_native_xylo, XyloInferenceSession
assert all(callable(x) for x in (convert_ir,export_ir,run_simulator,freeze_native_lif,export_native_lif,freeze_native_xylo,export_native_xylo,XyloInferenceSession))
assert not set(sys.modules).intersection({'akida','lava','sinabs','samna','rockpool','xylosim','spinnaker2','torch','tensorflow'})
'''
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'PYTHONPATH':os.pathsep.join([str(root/'brian2-rust/python'), str(root)])}
    result = subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.parametrize('profile,target', [
    ('xylo-audio2-temporal-v1','xylosim'),
    ('spinnaker2-xylo-integer-v1','spinnaker2'),
    ('spinnaker2-lif-v1','spinnaker2'),
])
def test_native_bridge_requires_samples_before_loading_runtime(monkeypatch,profile,target):
    import brian2_rust.neuromorphic_runtime as runtime
    monkeypatch.setattr(runtime,'_module',lambda name:pytest.fail('runtime loaded for missing inputs'))
    bundle=dict(schema='atlas.neuromorphic.simulator-bundle.v1',profile=profile,target=target,
                native_training_bridge={})
    with pytest.raises(ValueError,match='explicit input batch'):
        run_simulator(bundle)


@pytest.mark.parametrize('entry',['xylo_batch','xylo_session','spinnaker2'])
@pytest.mark.parametrize('field,value',[
    ('sizes',[1,2,1]), ('logit_scale',float('nan')), ('logit_scale',0),
    ('logit_scale',True), ('logit_scale',101), ('logit_scale',None),
    ('logit_scale','invalid'), ('readout_start',4),
    ('readout_start',True), ('input_max',16), ('class_codes',[[1,0],[1,0]]),
    ('class_codes',[[1],[0]]), ('readout_divisor',64),
])
def test_malformed_decoder_rejected_before_sdk_allocation(monkeypatch,entry,field,value):
    plan=dict(schema=spinnaker2_integer_runtime.SCHEMA,execution_kind='custom-c-host-kernel',
        sizes=dict(input=1,hidden=1,output=2),steps=4,input_raster=[[0]]*4,
        parameters=dict(hidden=dict(dash_syn=[1],dash_mem=[1],bias=[0],threshold=[64]),
                        output=dict(dash_syn=[1,1],dash_mem=[1,1],bias=[0,0],threshold=[64,64])))
    bridge=dict(schema='atlas.native-xylo-frozen.v2',sizes=[1,1,2],logit_scale=1.,
                readout_start=1,stateful_chunks=True,fresh_state_per_sample=True,
                source_is_zero_input_template=True,dynamics_equivalent=True)
    bridge[field]=value
    original=copy.deepcopy(bridge)
    monkeypatch.setattr(xylo_runtime,'_model',lambda *a:pytest.fail('Xylo SDK allocated before validation'))
    monkeypatch.setattr(spinnaker2_integer_runtime,'build_network',lambda *a:pytest.fail('SpiNNaker2 SDK allocated before validation'))
    x=np.zeros((1,4,1),dtype=np.uint8)
    with pytest.raises(ValueError):
        if entry=='xylo_batch':xylo_runtime.run_native_batch(plan,bridge,x)
        elif entry=='xylo_session':
            xylo_runtime.XyloInferenceSession(dict(schema='atlas.neuromorphic.simulator-bundle.v1',
                target='xylosim',profile='xylo-audio2-temporal-v1',plan=plan,native_training_bridge=bridge))
        else:spinnaker2_integer_runtime.run(plan,x,bridge=bridge)
    # Compare serializable metadata, including the deliberately invalid NaN.
    import json
    assert json.dumps(bridge,sort_keys=True)==json.dumps(original,sort_keys=True)


@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf'),True,None,'128'])
def test_lif_readout_rejects_invalid_scale_before_sdk(monkeypatch,scale):
    plan=dict(sizes=dict(input=1,hidden=1,output=2),steps=4,input_raster=[[0]]*4)
    bridge=dict(schema='atlas.native-lif-frozen.v1',sizes=[1,1,2],logit_scale=scale,
        dynamics_equivalent=False,fresh_state_per_sample=True,source_is_zero_input_template=True,
        adaptation=dict(schema='atlas.native-lif.xylo-adaptation.v1'))
    monkeypatch.setattr(xylo_runtime,'_model',lambda *a:pytest.fail('SDK allocated for an invalid scale'))
    with pytest.raises(ValueError,match='logit_scale'):
        xylo_runtime.run_native_batch(plan,bridge,np.zeros((1,4,1),dtype=np.uint8))
