"""AtlasIR admission for the custom SpiNNaker2 SDK host integer kernel."""
import numpy as np
from .xylo_b2ir import lower_b2ir as lower_xylo

PROFILE = 'spinnaker2-xylo-integer-v1'
SCHEMA = 'atlas.spinnaker2.xylo-integer-plan.v1'


def graph_plan(plan):
    if plan['steps'] > 128:
        raise ValueError('SpiNNaker2 integer kernel: at most 128 ticks')
    for params in plan['parameters'].values():
        if (np.any(np.asarray(params['dash_syn']) != 1)
                or np.any(np.asarray(params['dash_mem']) != 1)
                or np.any(np.asarray(params['bias']) != 0)):
            raise ValueError('SpiNNaker2 integer kernel: shifts must be one and bias zero')
    plan = dict(plan, schema=SCHEMA, execution_kind='custom-c-host-kernel',
                stock_sdk_neuron_model=False, firmware_adapter_implemented=False)
    # Xylo resource admission remains conservative, but is not S2 placement proof.
    plan.pop('hardware_dimension_limits_passed', None)
    plan.pop('hardware_config_validation_required', None)
    return plan


def lower_b2ir(model_path, *, runner):
    return graph_plan(lower_xylo(model_path, runner=runner))
