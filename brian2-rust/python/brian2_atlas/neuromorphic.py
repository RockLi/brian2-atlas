"""Experimental AtlasIR conversion to optional CPU software simulators.

No SDK is imported by conversion. This API is deliberately separate from the
Atlas simulation Device: it emits backend-private plans, not a new IR schema.
"""
import hashlib
import json
from pathlib import Path
import struct

from brian2_rust.neuromorphic_runtime import run_simulator, validate_inputs

from brian2_rust.akida_b2ir import lower_b2ir
from brian2_rust.protocol import canonical_bytes

TARGETS = ('akida', 'lava', 'specksim', 'spinnaker2', 'xylosim', 'dynapse')
PROFILES = ('dense-int8-v1', 'binary-count-v1', 'spinnaker2-lif-v1', 'xylo-audio2-temporal-v1', 'dynapse-dpi-v1')


def convert_ir(model_path, *, target, runner, profile='dense-int8-v1'):
    """Validate AtlasIR and compile a simulator plan, rejecting semantic loss.

    ``binary-count-v1`` restricts its five targets to binary inputs and to the same
    bias-free count classifier. ``dense-int8-v1`` also handles signed biased
    MLPs on Akida/Lava/SpiNNaker2; Specksim explicitly rejects that general profile.
    ``spinnaker2-lif-v1`` admits a separate temporal LIF subset with embedded
    spike input; use target ``spinnaker2``. See the dedicated example.
    ``xylo-audio2-temporal-v1`` admits bounded integer current/membrane
    dynamics, recurrent spike counts and hardware-sized Xylo Audio 2 graphs.
    ``dynapse-dpi-v1`` admits a narrow feed-forward DynapSim circuit model
    with adaptation and refractory state; it does not admit static classifiers.
    The runner is a trusted native Atlas executable supplied by the caller.
    """
    if target not in TARGETS:
        raise ValueError(f'unknown simulator target: {target!r}; expected {TARGETS}')
    if profile not in PROFILES:
        raise ValueError(f'unknown conversion profile: {profile!r}')
    if profile == 'dynapse-dpi-v1':
        if target != 'dynapse':
            raise ValueError('dynapse-dpi-v1 is only supported by dynapse')
        from brian2_rust.dynapse_b2ir import lower_b2ir as lower_dynapse
        plan = lower_dynapse(model_path, runner=runner)
        return dict(schema='atlas.neuromorphic.simulator-bundle.v1', target=target, profile=profile,
                    input_domain='embedded binary spike raster', plan=plan,
                    source_model_sha256=plan['source_model_sha256'], source_hashes=plan['source_hashes'],
                    atlas_validator_sha256=plan['atlas_validator_sha256'],
                    physical_hardware=False, hardware_mapping_tested=False)
    if target == 'dynapse':
        raise ValueError('dynapse requires dynapse-dpi-v1; other profiles are unsupported')
    if profile == 'xylo-audio2-temporal-v1':
        if target != 'xylosim':
            raise ValueError('xylo-audio2-temporal-v1 is only supported by xylosim')
        from brian2_rust.xylo_b2ir import lower_b2ir as lower_xylo
        plan = lower_xylo(model_path, runner=runner)
        return dict(schema='atlas.neuromorphic.simulator-bundle.v1', target=target, profile=profile,
                    input_domain='embedded integer spike-count raster', plan=plan,
                    source_model_sha256=plan['source_model_sha256'], source_hashes=plan['source_hashes'],
                    atlas_validator_sha256=plan['atlas_validator_sha256'],
                    physical_hardware=False, hardware_mapping_tested=False)
    if profile == 'spinnaker2-lif-v1':
        if target != 'spinnaker2' or profile != 'spinnaker2-lif-v1':
            raise ValueError('spinnaker2-lif-v1 is only supported by target spinnaker2')
        from brian2_rust.spinnaker2_b2ir import lower_b2ir as lower_lif
        plan = lower_lif(model_path, runner=runner)
        return dict(schema='atlas.neuromorphic.simulator-bundle.v1', target=target, profile=profile,
                    input_domain='embedded spike schedule', plan=plan,
                    source_model_sha256=plan['source_model_sha256'], source_hashes=plan['source_hashes'],
                    atlas_validator_sha256=plan['atlas_validator_sha256'],
                    physical_hardware=False, hardware_mapping_tested=False)
    if target in ('specksim', 'xylosim') and profile != 'binary-count-v1':
        raise ValueError(f'{target} requires binary-count-v1; arbitrary biased/signed TF MLPs are unsupported')
    source = lower_b2ir(model_path, runner=runner, profile='dense-int8-v1')
    # Recheck bytes after native validation; do not admit another source if it
    # changed between reads. Save the exact canonical source with the bundle.
    model = json.loads(Path(model_path).read_text())
    encoded = canonical_bytes(model)
    if hashlib.sha256(encoded).hexdigest() != source['source_model_sha256']:
        raise ValueError('AtlasIR source changed during conversion')
    count_plan = None
    if profile == 'binary-count-v1':
        from brian2_rust.specksim_dense_b2ir import graph_plan
        count_plan = graph_plan(source)
        for values in model['instance']['populations'][0]['parameters'].values():
            if any(struct.unpack('>d', bytes.fromhex(v))[0] not in (0, 1) for v in values):
                raise ValueError('source instance is outside binary input domain')
    if target == 'akida':
        plan = source
    elif target == 'lava':
        from brian2_rust.lava_dense_b2ir import graph_plan
        plan = graph_plan(source)
    elif target == 'spinnaker2' and profile == 'dense-int8-v1':
        from brian2_rust.spinnaker2_dense import graph_plan
        plan = graph_plan(source)
    elif target in ('spinnaker2', 'xylosim'):
        from brian2_rust.count_targets import graph_plan
        plan = graph_plan(count_plan, target)
    else:
        plan = count_plan
    return dict(schema='atlas.neuromorphic.simulator-bundle.v1', target=target, profile=profile,
                input_domain='binary' if profile == 'binary-count-v1' else 'integer [0,127]',
                source_model_sha256=source['source_model_sha256'], source_hashes=source['source_hashes'],
                atlas_validator_sha256=source['atlas_validator_sha256'], plan=plan,
                physical_hardware=False, hardware_mapping_tested=False)


def export_ir(model_path, directory, **kwargs):
    """Export canonical source + simulator bundle; refuse existing destinations."""
    bundle = convert_ir(model_path, **kwargs)
    encoded = canonical_bytes(json.loads(Path(model_path).read_text()))
    if hashlib.sha256(encoded).hexdigest() != bundle['source_model_sha256']:
        raise ValueError('AtlasIR source changed during export')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'source.json').write_bytes(encoded)
    (directory/'bundle.json').write_text(json.dumps(bundle, indent=2)+'\n')
    return bundle


# Native training is frozen explicitly before target conversion.
from brian2_rust.native_lif_conversion import freeze_native_lif, export_native_lif

from brian2_rust.native_xylo_conversion import freeze_native_xylo, export_native_xylo
