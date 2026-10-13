"""B2IR semantic admission and lowering; SDK end-to-end checks live in the experiment."""
import importlib.util
import copy
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from brian2_rust import akida_b2ir as api
from brian2_rust.protocol import attach_protocol

FIXTURE = ROOT / 'tests/golden/akida-b2ir/static-affine.json'
RUNNER = Path(os.environ.get('B2_RUNNER', ROOT / 'target/release/b2-runner'))


@pytest.fixture
def model():
    return json.loads(FIXTURE.read_text())


def block(model):
    return model['definition']['populations'][0]['code_objects'][1]['vector']


def literal(value):
    return dict(op='literal', bits=struct.pack('>d', value).hex())


def test_extracts_matrices_and_preserves_all_states(model):
    plan = api._lower_validated(model)
    assert plan['inputs'] == ['a', 'bb']
    assert plan['outputs'] == ['h0', 'h1', 'y']
    assert plan['weights1'] == [[1, 1], [1, 1]]
    assert plan['bias1'] == [0, -1]
    assert plan['weights2'] == [[1, 0, 1], [0, 1, -2]]
    assert plan['exhaustive_outputs'] == [[0, 0, 0], [1, 0, 1], [1, 0, 1], [2, 1, 0]]


def test_modified_coefficient_is_compiled_not_hardcoded(model):
    block(model)[2]['value']['right']['left'] = literal(3)
    plan = api._lower_validated(model)
    assert plan['weights2'][-1][-1] == -3
    assert plan['exhaustive_outputs'][-1] == [2, 1, -1]


@pytest.mark.parametrize('kind,reason', [
    ('ticks', 'one tick'), ('feedback', 'read before assignment'),
    ('clip', 'upper bound'), ('clip_min', 'lower bound'),
    ('fraction', 'exact integers'), ('nonlinear', 'nonlinear'),
    ('conditional', 'conditional'), ('unwritten', 'all states'),
    ('duplicate', 'exactly once'), ('input', 'binary values'),
    ('events', 'events'), ('units', 'dimensionless'),
    ('scalar', 'scalar statements'), ('output_bias', 'output bias'),
    ('negative_zero', 'negative zero'),
])
def test_rejects_semantics_it_cannot_preserve(model, kind, reason):
    p = model['definition']['populations'][0]
    statements = block(model)
    if kind == 'ticks':
        model['run']['clocks'][0]['steps'] = 2
    elif kind == 'feedback':
        statements[0]['value']['value']['left']['name'] = 'h0'
    elif kind == 'clip':
        statements[0]['value']['max'] = literal(1)
    elif kind == 'clip_min':
        statements[0]['value']['min'] = literal(1)
    elif kind == 'fraction':
        statements[2]['value']['right']['left'] = literal(1.5)
    elif kind == 'nonlinear':
        statements[0]['value']['value']['op'] = 'mul'
    elif kind == 'conditional':
        statements[0]['condition'] = 'gate'
    elif kind == 'unwritten':
        statements.pop()
    elif kind == 'duplicate':
        statements[-1]['target'] = 'h0'
    elif kind == 'input':
        model['instance']['populations'][0]['parameters']['a'][0] = struct.pack('>d', 2.).hex()
    elif kind == 'events':
        p['events'] = [dict(name='spike')]
    elif kind == 'units':
        p['parameters'][0]['dimensions'][2] = 1
    elif kind == 'scalar':
        p['code_objects'][1]['scalar'] = [dict(target='temp')]
    elif kind == 'output_bias':
        statements[-1]['value'] = dict(op='add', left=dict(op='load', name='h0'), right=literal(1))
    elif kind == 'negative_zero':
        statements[-1]['value'] = dict(op='neg', arg=dict(op='load', name='h0'))
    with pytest.raises(api.B2IRAkidaError, match=reason):
        api._lower_validated(model)


def test_inference_rejects_invalid_values_before_cast(model):
    class MustNotRun:
        def predict(self, values):
            raise AssertionError('SDK must not run for invalid inputs')
    plan = api._lower_validated(model)
    for value in [-1, 2, .5, np.nan, np.inf, 256]:
        with pytest.raises(api.B2IRAkidaError, match='binary domain'):
            api.predict_binary(MustNotRun(), plan, [[0, value]])
    with pytest.raises(api.B2IRAkidaError, match='dimensions'):
        api.predict_binary(MustNotRun(), plan, [[0]])


def test_classifier_profile_adds_bias_and_wider_accumulation_explicitly(model):
    block(model)[2]['value'] = dict(op='add', left=dict(op='mul', left=literal(127),
                                      right=dict(op='load', name='h0')), right=literal(3))
    with pytest.raises(api.B2IRAkidaError, match='output bias'):
        api._lower_validated(model)
    plan = api._lower_validated(model, profile='binary-classifier-v1')
    assert plan['bias2'] == [0, 0, 3]
    assert plan['output_absolute_bounds'][-1] == 257
    assert plan['verification_outputs'][-1] == [2, 1, 257]
    assert 'exhaustive_inputs' not in plan


def test_wide_profile_uses_bounded_checks_not_exponential_enumeration(model):
    p = model['definition']['populations'][0]
    for index in range(62):
        name = f'extra{index}'
        p['parameters'].append(dict(name=name, dtype='f64', index_domain='neuron', dimensions=[0.]*7))
        model['instance']['populations'][0]['parameters'][name] = ['0000000000000000'] * 4
    with pytest.raises(api.B2IRAkidaError, match='count exceeds'):
        api._lower_validated(model)
    plan = api._lower_validated(model, profile='binary-classifier-v1')
    assert len(plan['inputs']) == 64
    assert len(plan['verification_inputs']) <= 1158
    assert 'not exhaustive' in plan['proof']
    assert max(plan['output_absolute_bounds']) <= 32767


def test_classifier_v2_explicitly_extends_input_count_and_preserves_v1_limit(model):
    p = model['definition']['populations'][0]
    for index in range(126):
        name = f'extra{index}'
        p['parameters'].append(dict(name=name, dtype='f64', index_domain='neuron', dimensions=[0.]*7))
        model['instance']['populations'][0]['parameters'][name] = ['0000000000000000'] * 4
    with pytest.raises(api.B2IRAkidaError, match='count exceeds'):
        api._lower_validated(model, profile='binary-classifier-v1')
    plan = api._lower_validated(model, profile='binary-classifier-v2')
    assert plan['schema'] == 'atlas.akida.binary-classifier-plan.v2'
    assert len(plan['inputs']) == 128
    p['parameters'].append(dict(name='overflow', dtype='f64', index_domain='neuron', dimensions=[0.]*7))
    with pytest.raises(api.B2IRAkidaError, match='count exceeds'):
        api._lower_validated(model, profile='binary-classifier-v2')


def test_classifier_v3_state_budget_preserves_older_profile_limits(model):
    p = model['definition']['populations'][0]
    for index in range(125):
        name = f'z{index:03d}'
        state = dict(p['states'][-1], name=name)
        statement = copy.deepcopy(block(model)[-1])
        statement['target'] = name
        p['states'].append(state)
        block(model).append(statement)
        model['instance']['populations'][0]['initial_state'][name] = ['0000000000000000'] * 4
    for profile in ['binary-classifier-v1', 'binary-classifier-v2']:
        with pytest.raises(api.B2IRAkidaError, match='count exceeds'):
            api._lower_validated(model, profile=profile)
    plan = api._lower_validated(model, profile='binary-classifier-v3')
    assert len(plan['outputs']) == 128
    assert plan['schema'] == 'atlas.akida.binary-classifier-plan.v3'
    p['states'].append(dict(p['states'][-1], name='overflow'))
    with pytest.raises(api.B2IRAkidaError, match='count exceeds'):
        api._lower_validated(model, profile='binary-classifier-v3')


@pytest.mark.skipif(not RUNNER.is_file(), reason='requires trusted Atlas runner')
def test_public_export_uses_rust_validation_and_checks_integrity(model, tmp_path):
    valid = api.export_b2ir_plan(FIXTURE, tmp_path / 'valid', runner=RUNNER)
    assert valid['atlas_validator_sha256']
    assert (tmp_path / 'valid/source-b2ir.json').is_file()
    # Re-sealed invalid schedule passes hash checking, but must fail the actual
    # Atlas Rust validator, before any Akida export directory is created.
    model['definition']['schedule']['nodes'][0]['effects']['writes'] = []
    attach_protocol(model)
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(model))
    with pytest.raises(api.B2IRAkidaError, match='Atlas B2IR validation failed'):
        api.export_b2ir_plan(bad, tmp_path / 'bad-export', runner=RUNNER)
    assert not (tmp_path / 'bad-export').exists()
    model['instance']['rng_seed'] += 1
    bad.write_text(json.dumps(model))
    with pytest.raises(ValueError, match='hash mismatch'):
        api.lower_b2ir(bad, runner=RUNNER)


def quantized_fixture(model):
    for statement in block(model)[:2]:
        affine = statement['value']['value']
        statement['value'] = dict(op='clip', min=literal(0), max=literal(127),
            value=dict(op='floor', arg=dict(op='div',
                left=dict(op='add',left=affine,right=literal(1)),right=literal(2))))
    return model


def test_quantized_profile_preserves_rounding_and_saturation(model):
    quantized_fixture(model)
    plan=api._lower_validated(model,profile='dense-int8-v1')
    assert plan['shift1']==[1,1]
    assert plan['bias1']==[0,-1]
    assert plan['schema']=='atlas.akida.dense-int8-plan.v1'
    assert plan['verification_outputs'][0]==[0,0,0]
    with pytest.raises(api.B2IRAkidaError):
        api._lower_validated(model,profile='binary-classifier-v3')


@pytest.mark.parametrize('kind,reason', [('divisor','power of two'),('clip','saturate'),('bias','bias cannot'),
                                        ('domain','integer inputs'),('fraction','exact integers')])
def test_quantized_profile_rejects_inexact_semantics(model,kind,reason):
    quantized_fixture(model)
    clip=block(model)[0]['value']
    if kind=='divisor':clip['value']['arg']['right']=literal(3)
    elif kind=='clip':clip['max']=literal(126)
    elif kind=='bias':clip['value']['arg']['left']['right']=literal(130)  # bias 129 cannot fit int8<<shift
    elif kind=='domain':model['instance']['populations'][0]['parameters']['a'][0]=struct.pack('>d',128.).hex()
    else:clip['value']['arg']['left']['right']=literal(1.5)
    with pytest.raises(api.B2IRAkidaError,match=reason):
        api._lower_validated(model,profile='dense-int8-v1')


def test_quantized_inference_rejects_values_before_cast(model):
    class MustNotRun:
        def predict(self,values):raise AssertionError('must reject before SDK')
    plan=api._lower_validated(quantized_fixture(model),profile='dense-int8-v1')
    for value in [-1,128,256,1.5,np.nan,np.inf]:
        with pytest.raises(api.B2IRAkidaError,match='integer'):
            api.predict_quantized(MustNotRun(),plan,[[0,value]])
