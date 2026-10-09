"""Public AtlasIR naming must preserve old model and native-function identities."""
import io
import json
from pathlib import Path

import brian2 as b
import pytest
from brian2_atlas import IR_NAME, ir
from brian2_rust.spec import portable_function_contract

GOLDEN = Path(__file__).parent / 'golden/b2ir-v1/minimal-v1.json'


def curve(x):
    return x * x + 0.5


def function(target, source='double curve(double x) { return x*x+0.5; }'):
    value = b.Function(curve, arg_units=[1], return_unit=1, stateless=True)
    value.implementations.add_implementation(target, code=source, name='curve')
    return value


def test_public_ir_roundtrip_preserves_frozen_v1_bytes_and_hashes():
    assert IR_NAME == ir.IR_NAME == 'AtlasIR'
    model = json.loads(GOLDEN.read_text())
    original = ir.canonical_bytes(model)
    migrated = ir.migrate_model(model)
    stream = io.BytesIO()
    ir.write_canonical(migrated, stream)
    assert stream.getvalue() == original
    assert json.loads(stream.getvalue()) == model
    ir.verify_protocol(migrated)
    assert ir.layer_hashes(migrated) == model['protocol']['layers']
    assert ir.CURRENT_SCHEMA == model['schema'] == 'b2ir-v1'


@pytest.mark.parametrize('suffix,backend', [
    ('c-abi-v1','cpu'), ('cuda-device-v1','cuda'),
    ('metal-v1','metal'), ('wgsl-v1','wgsl'),
])
def test_function_registration_alias_preserves_serialized_contract(suffix, backend):
    old = portable_function_contract('curve', function('b2ir-' + suffix))
    new = portable_function_contract('curve', function('atlasir-' + suffix))
    assert ir.canonical_bytes(new) == ir.canonical_bytes(old)
    assert new['backend_implementations'][backend]['abi'] == 'b2ir-' + suffix


@pytest.mark.parametrize('suffix', ['c-abi-v1','cuda-device-v1','metal-v1','wgsl-v1'])
def test_duplicate_aliases_must_agree(suffix):
    value = function('b2ir-' + suffix)
    original = portable_function_contract('curve', value)
    value.implementations.add_implementation('atlasir-' + suffix,
        code='double curve(double x) { return x*x+0.5; }', name='curve')
    assert portable_function_contract('curve', value) == original
    value.implementations.add_implementation('atlasir-' + suffix,
        code='double curve(double x) { return x*x+1.0; }', name='curve')
    with pytest.raises(NotImplementedError, match='conflicting AtlasIR'):
        portable_function_contract('curve', value)
