"""Admission and SDK-boundary regressions for the static binary-decision encoding."""
import copy
import os
from pathlib import Path
import numpy as np
import pytest
from brian2_rust.spinnaker2_dense import graph_plan


def fixture():
    return dict(schema='atlas.akida.dense-int8-plan.v1',inputs=['x'],
        hidden_states=['h'],outputs=['h','y'],weights1=[[127]],weights2=[[1,-127]],
        bias1=[-1],bias_shift1=[0],bias2=[0,-127],bias_shift2=[0,0],shift1=[7],
        verification_inputs=[[0]],verification_outputs=[[0,-127]])


def test_compiled_resources_and_weight_preservation():
    source = fixture()
    before = copy.deepcopy(source)
    plan = graph_plan(source)
    assert source == before
    assert 'verification_inputs' not in plan and 'verification_outputs' not in plan
    assert plan['weights1'] == source['weights1']
    assert plan['weights2'] == source['weights2']
    assert plan['pipeline_delay_ticks'] == 8
    assert plan['resources'] == dict(input_neurons=8,comparator_neurons=7,readout_neurons=2,
        first_layer_connections=49,bias_connections=7,decision_connections=21,readout_connections=14)


@pytest.mark.parametrize('shift',[1,7,13])
def test_admits_20bit_weight_boundary(shift):
    source = fixture()
    source['shift1'] = [shift]
    assert graph_plan(source)['maximum_encoded_weight'] <= 1048575


@pytest.mark.parametrize('change',['shift','bias'])
def test_rejects_encoded_weight_overflow(change):
    source = fixture()
    if change == 'shift':
        source['shift1'] = [14]
    else:
        source['bias1'], source['bias_shift1'] = [127],[15]
    with pytest.raises(ValueError,match='20-bit'):
        graph_plan(source)


def test_public_dense_route_uses_native_validation(tmp_path):
    import brian2 as b
    from brian2_atlas.neuromorphic import export_ir
    from brian2_rust.export import lower_network
    from brian2_rust.protocol import canonical_bytes
    runner = Path(os.environ.get('B2_RUNNER','missing'))
    if not runner.is_file():
        pytest.skip('set B2_RUNNER')
    b.start_scope()
    b.set_device('runtime')
    b.prefs.codegen.loop_invariant_optimisations = False
    g = b.NeuronGroup(2,'x:1 (constant)\nh:1\ny:1',dt=b.ms)
    g.x = [0,127]
    g.run_regularly('h=clip(floor((127*x-1+64)/128),0,127)\ny=-127*h-5',when='start')
    source = tmp_path/'source.json'
    source.write_bytes(canonical_bytes(lower_network(b.Network(g),b.ms)))
    bundle = export_ir(source,tmp_path/'converted',target='spinnaker2',runner=runner)
    assert bundle['plan']['schema'] == 'atlas.spinnaker2.dense-int8-plan.v1'
    assert bundle['atlas_validator_sha256']
    assert (tmp_path/'converted/source.json').read_bytes() == source.read_bytes()
