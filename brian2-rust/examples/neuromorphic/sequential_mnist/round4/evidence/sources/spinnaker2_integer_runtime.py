"""Custom SpiNNaker2 SDK network execution using Atlas's independent C kernel.

This is a host extension, not stock Brian2Backend execution or S2 PE firmware.
Only the shipped kernel source is compiled; bundles cannot supply code or paths.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

if __package__:
    from .xylo_readout import decode_readout, validate_readout_contract
else:
    import importlib.util
    _spec = importlib.util.spec_from_file_location('atlas_xylo_readout', Path(__file__).with_name('xylo_readout.py'))
    _readout = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_readout)
    decode_readout = _readout.decode_readout
    validate_readout_contract = _readout.validate_readout_contract

MODEL = 'atlas_xylo_integer_host_v1'
PROFILE = 'spinnaker2-xylo-integer-v1'
SCHEMA = 'atlas.spinnaker2.xylo-integer-plan.v1'
KERNEL_DIR = Path(__file__).with_name('spinnaker2_integer_kernel')


def require(condition, message):
    if not condition:
        raise ValueError('SpiNNaker2 integer kernel: ' + message)


def integer_array(value, shape, lower, upper, name, dtype=np.int32):
    array = np.asarray(value)
    require(array.shape == shape and np.issubdtype(array.dtype, np.number)
            and np.isfinite(array).all() and (array == np.floor(array)).all()
            and ((array >= lower) & (array <= upper)).all(), 'invalid ' + name)
    return np.ascontiguousarray(array, dtype=dtype)


class IntegerKernel:
    """Compile the shipped C99 implementation and retain explicit batch state."""
    def __init__(self):
        compiler = shutil.which('cc')
        if compiler is None:
            raise RuntimeError('custom SpiNNaker2 host kernel requires a C99 compiler (cc)')
        self.directory = tempfile.TemporaryDirectory(prefix='atlas-s2-integer-')
        library = Path(self.directory.name) / 'kernel.so'
        subprocess.run([compiler, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
            '-shared', '-fPIC', str(KERNEL_DIR / 'kernel.c'), '-o', str(library)],
            capture_output=True, check=True, timeout=60)
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.atlas_s2_integer_run
        pointer = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        self.function.argtypes = [ctypes.c_int] * 5 + [pointer] * 4 + [
            np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'), pointer, pointer]
        self.function.restype = ctypes.c_int
        self.receipt = dict(source_hashes={name: hashlib.sha256((KERNEL_DIR / name).read_bytes()).hexdigest()
            for name in ('kernel.c', 'kernel.h', 'CMakeLists.txt')},
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            compiler=subprocess.run([compiler, '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0])

    def run(self, weights, thresholds, inputs, state):
        ni, nh = weights[0].shape
        no = weights[1].shape[1]
        x = np.asarray(inputs)
        require(x.ndim == 3, 'input must be batch x ticks x channels')
        batch, steps, channels = x.shape
        require(1 <= ni <= 16 and 1 <= nh <= 128 and 1 <= no <= 8
                and 1 <= batch <= 128 and 1 <= steps <= 128 and channels == ni, 'dimension bounds')
        shapes = ((ni, nh), (nh, no), (nh, nh))
        require(len(weights) == 3, 'requires three weight banks')
        banks = [integer_array(w, shape, -128, 127, 'weights') for w, shape in zip(weights, shapes)]
        cells = nh + no
        threshold = integer_array(thresholds, (cells,), 1, 32767, 'threshold')
        x = integer_array(x, (batch, steps, ni), 0, 15, 'input counts', np.uint8)
        s = integer_array(state, (batch, 3, cells), -32768, 32767, 'state').copy()
        require(((s[:, 2] >= 0) & (s[:, 2] <= np.array([31]*nh + [1]*no))).all(), 'spike state bounds')
        trace = np.empty((batch, steps, 3, cells), dtype=np.int32)
        status = self.function(ni, nh, no, batch, steps, *banks, threshold, x, s, trace)
        if status:
            raise RuntimeError(f'integer C kernel rejected request: {status}')
        return trace, s

    def close(self):
        self.directory.cleanup()


def register_model():
    """Use SDK registration, but explicitly refuse every firmware mapping hook."""
    from spinnaker2 import neuron_models
    from spinnaker2.neuron_models.application import BaseApplication
    from spinnaker2.synapses import SynapseWordSize, SynapseWordSpec

    class AtlasIntegerHostApplication(BaseApplication):
        app_name = MODEL
        max_atoms_per_core = 128
        splittable = False
        default_params = dict(threshold=8, spike_cap=31)
        sw_spec = SynapseWordSpec(word_size=SynapseWordSize.SIZE_32, weight=20, delay=3, synapse_type=1, target=8)

        @staticmethod
        def recordables():
            return ['spikes', 'I', 'v', 'q']

        def get_app_path(self):
            return KERNEL_DIR

        @staticmethod
        def unavailable(*args, **kwargs):
            raise NotImplementedError('Atlas integer model is host-only; SpiNNaker2 firmware transport and placement are not implemented')

        pe_config = unavailable
        get_compiler_flags = unavailable
        create_vertex = unavailable
        get_vertex_splitter = unavailable

    existing = neuron_models.NEURON_MODEL_TO_APPLICATION.get(MODEL)
    if existing is not None:
        require(Path(existing().get_app_path()).resolve() == KERNEL_DIR.resolve(), 'custom model name collision')
    else:
        neuron_models.register_custom_model(AtlasIntegerHostApplication)
    return neuron_models.get_application(MODEL)


def build_network(plan):
    """Build SDK containers from the admitted plan; do not change source weights."""
    from spinnaker2 import snn
    register_model()
    sizes = plan['sizes']
    net = snn.Network('atlas_integer_host')
    source = snn.Population(sizes['input'], 'spike_list', {}, name='input')
    hidden = snn.Population(sizes['hidden'], MODEL,
        dict(threshold=plan['parameters']['hidden']['threshold'], spike_cap=31),
        name='hidden', record=['spikes', 'I', 'v', 'q'])
    output = snn.Population(sizes['output'], MODEL,
        dict(threshold=plan['parameters']['output']['threshold'], spike_cap=1),
        name='output', record=['spikes', 'I', 'v', 'q'])
    net.add(source, hidden, output)
    for role, pre, post in [('input', source, hidden), ('readout', hidden, output), ('recurrent', hidden, hidden)]:
        w = np.asarray(plan['weights'][role])
        ii, jj = np.nonzero(w)
        if len(ii):
            net.add(snn.Projection(pre, post, [[int(i), int(j), int(w[i, j]), 0] for i, j in zip(ii, jj)], name=role+'_projection'))
    return net


class IntegerHostBackend:
    """Explicit SDK extension. C, not Brian2 or the Atlas Rust runner, updates state.

    Inputs are independent raster replicas of a single SDK graph. Each call
    starts from zero; only chunks inside that call retain C kernel state.
    SDK Population accessors expose the last replica's records.
    """
    def __init__(self):
        self.kernel = IntegerKernel()

    def run(self, network, inputs, *, chunk_steps=None):
        pops = {p.name: p for p in network.populations}
        require(len(network.populations) == 3 and set(pops) == {'input', 'hidden', 'output'}, 'requires input/hidden/output SDK populations')
        source, hidden, output = [pops[name] for name in ('input', 'hidden', 'output')]
        require(source.neuron_model == 'spike_list' and not source.params
                and hidden.neuron_model == output.neuron_model == MODEL, 'incorrect SDK neuron model/input transport')
        require(hidden.params['spike_cap'] == 31 and output.params['spike_cap'] == 1, 'spike caps must be 31/1')
        ni, nh, no = source.size, hidden.size, output.size
        weights = [np.zeros(shape, dtype=np.int32) for shape in ((ni, nh), (nh, no), (nh, nh))]
        endpoints = {(source, hidden): 0, (hidden, output): 1, (hidden, hidden): 2}
        seen = set()
        for projection in network.projections:
            pair = (projection.pre, projection.post)
            require(pair in endpoints and pair not in seen, 'unsupported or duplicate SDK projection')
            seen.add(pair)
            require(not np.any(projection.sparse_delays.data), 'explicit delays are unsupported')
            index = endpoints[pair]
            weights[index] = integer_array(projection.sparse_weights.toarray(), weights[index].shape, -128, 127, 'SDK weights')
        thresholds = np.concatenate((np.asarray(hidden.params['threshold']), np.asarray(output.params['threshold'])))
        x = np.asarray(inputs)
        require(x.ndim == 3 and len(x) > 0 and 1 <= x.shape[1] <= 128 and x.shape[2] == ni, 'input dimensions')
        x = integer_array(x, x.shape, 0, 15, 'input counts', np.uint8)
        width = x.shape[1] if chunk_steps is None else chunk_steps
        require(type(width) is int and 1 <= width <= 128, 'chunk_steps must be 1..128')
        state = np.zeros((len(x), 3, nh + no), dtype=np.int32)
        parts = []
        for start in range(0, x.shape[1], width):
            part, state = self.kernel.run(weights, thresholds, x[:, start:start+width], state)
            parts.append(part)
        trace = np.concatenate(parts, axis=1)
        # Keep SDK accessors useful and test count-to-event multiplicity explicitly.
        for pop, begin, end in ((hidden, 0, nh), (output, nh, nh+no)):
            pop.clear_records()
            for axis, key in enumerate(('I', 'v', 'q')):
                pop.add_variable(key, {j: trace[-1, :, axis, begin+j].copy() for j in range(end-begin)})
            pop.add_spikes({j: np.repeat(np.arange(x.shape[1]), trace[-1, :, 2, begin+j]).tolist()
                            for j in range(end-begin)})
        return dict(currents=trace[:, :, 0], membranes=trace[:, :, 1], spikes=trace[:, :, 2],
                    final_state=state, weight_banks=[w.tolist() for w in weights])

    def close(self):
        self.kernel.close()


def run(plan, samples=None, *, bridge=None, batch_size=50, chunk_steps=None):
    require(plan.get('schema') == SCHEMA and plan.get('execution_kind') == 'custom-c-host-kernel', 'unknown plan')
    require(type(batch_size) is int and 1 <= batch_size <= 128, 'batch_size must be 1..128')
    require(type(plan['steps']) is int and 1 <= plan['steps'] <= 128, 'duration must be 1..128')
    for params in plan['parameters'].values():
        require(np.all(np.asarray(params['dash_syn']) == 1) and np.all(np.asarray(params['dash_mem']) == 1)
                and not np.any(params['bias']), 'requires shifts one and bias zero')
    if samples is None:
        require(bridge is None, 'native bridge requires an input batch')
        samples = np.asarray(plan['input_raster'])[None]
    else:
        require(bridge is not None and bridge.get('schema') in ('atlas.native-xylo-frozen.v1', 'atlas.native-xylo-frozen.v2', 'atlas.native-xylo-current-bridge.v1')
                and bridge.get('fresh_state_per_sample') and bridge.get('source_is_zero_input_template')
                and bridge.get('dynamics_equivalent') is True, 'unknown native bridge')
        require(not np.any(plan['input_raster']), 'native bridge requires zero-input template')
    validate_readout_contract(bridge or {},plan['parameters']['output']['threshold'])
    x = np.asarray(samples)
    require(x.ndim == 3 and len(x) > 0 and x.shape[1:] == (plan['steps'], plan['sizes']['input']), 'input shape')
    upper = bridge.get('input_max',1) if bridge else 15
    require(type(upper) is int and 1 <= upper <= 15, 'invalid input_max')
    x = integer_array(x, x.shape, 0, upper, 'input counts', np.uint8)
    net = build_network(plan)
    backend = IntegerHostBackend()
    arrays = {key: [] for key in ('currents', 'membranes', 'spikes')}
    try:
        for i in range(0, len(x), batch_size):
            result = backend.run(net, x[i:i+batch_size], chunk_steps=chunk_steps)
            for bank, role in zip(result['weight_banks'], ('input', 'readout', 'recurrent')):
                np.testing.assert_array_equal(bank, plan['weights'][role])
            for key in arrays:
                arrays[key].append(result[key])
        receipt = dict(backend.kernel.receipt, sdk_weight_banks_unchanged=True,
            sdk_weights_sha256=hashlib.sha256(json.dumps(result['weight_banks'],separators=(',',':')).encode()).hexdigest())
    finally:
        backend.close()
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    nh, no = plan['sizes']['hidden'], plan['sizes']['output']
    for role, sl in [('hidden', slice(0, nh)), ('output', slice(nh, nh+no))]:
        for key, field in [('I', 'currents'), ('v', 'membranes'), ('q', 'spikes')]:
            arrays[role+'_'+key] = arrays[field][:, :, sl]
    arrays['spike_raster'] = arrays['spikes']
    arrays['voltage'] = arrays['membranes']
    start = bridge.get('readout_start', 0) if bridge else 0
    require(type(start) is int and 0 <= start < plan['steps'], 'readout window outside horizon')
    arrays['counts'] = arrays['output_q'][:, start:].sum(1)
    arrays['current_sums'] = arrays['output_I'][:, start:].sum(1,dtype=np.int64)
    arrays['logits'] = decode_readout(arrays['counts'], arrays['current_sums'], bridge if bridge else {'logit_scale': 1.}, plan['steps'] - start)
    return dict(**arrays, execution_kind='custom-c-host-kernel', dynamics_equivalent=True,
                stock_sdk_neuron_model=False, physical_hardware=False, hardware_mapping_tested=False,
                firmware_adapter_implemented=False, kernel_receipt=receipt)
