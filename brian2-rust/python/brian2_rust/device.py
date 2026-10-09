"""Brian2 Device for the Atlas execution backends."""

import ast
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
from brian2 import (EventMonitor, NeuronGroup, PoissonGroup, PoissonInput,
                    PopulationRateMonitor, SpikeGeneratorGroup, SpikeMonitor, StateMonitor, Synapses,
                    second)
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2.core.magic import MagicNetwork
from brian2.core.namespace import get_local_namespace
from brian2.core.network import Network, TextReport, _get_all_objects
from brian2.core.operations import NetworkOperation
from brian2.core.preferences import prefs
from brian2.core.variables import ArrayVariable, DynamicArrayVariable, VariableView
from brian2.devices.device import Device
from brian2.groups.group import CodeRunner
from brian2.groups.subgroup import Subgroup
from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D
from brian2.spatialneuron import SpatialNeuron
from brian2.synapses.parse_synaptic_generator_syntax import parse_synapse_generator
from brian2.units.fundamentalunits import (DIMENSIONLESS, Quantity,
                                           fail_for_dimension_mismatch)


from .capabilities import CapabilityError, collect_network_issues
from .monitor_observables import (reconstruct as reconstruct_monitor_observable,
                                  reconstruct_synapse)
from .export import _array_bits, lower_network, model_objects, require
from .results import load_results
from .native import _write_project_verified
from .protocol import attach_protocol, write_canonical
from .encoded_array import packed_export
from .topology import ClippedNormal, Uniform
from .resource_limits import candidate_pair_budget, explicit_synapse_budget

from ._runtime import (runtime_root, source_root, executable_path, cargo_target,
                       cache_root, create_run_directory)

ROOT = runtime_root()


def _read_checkpoint(path):
    """Read a checksummed checkpoint or an earlier plain state dictionary."""
    with Path(path).open("rb") as stream:
        stored = pickle.load(stream)
        require(not stream.read(1), "Rust checkpoint has trailing data")
    if isinstance(stored, dict) and "_rust_checkpoint_format" in stored:
        require(stored["_rust_checkpoint_format"] == 1 and
                set(stored) == {"_rust_checkpoint_format", "payload", "sha256"},
                "unsupported Rust checkpoint format")
        payload = stored["payload"]
        require(isinstance(payload, bytes) and
                hashlib.sha256(payload).hexdigest() == stored["sha256"],
                "Rust checkpoint checksum mismatch")
        stored = pickle.loads(payload)
    require(type(stored) is dict, "checkpoint file must contain a state dictionary")
    return stored


def _write_checkpoint(path, stored):
    """Commit one complete checkpoint atomically; preserve the old file on failure."""
    path = Path(path)
    payload = pickle.dumps(stored, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {"_rust_checkpoint_format": 1, "payload": payload,
                "sha256": hashlib.sha256(payload).hexdigest()}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=path.name+".", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(envelope, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)



class _NumpyDynamicArray:
    """Frontend-only fallback for dtypes omitted by Brian's Cython buffer."""

    def __init__(self, size, dtype):
        self.data = np.zeros(size, dtype=dtype)

    def resize(self, new_size):
        shape = ((new_size,) if np.isscalar(new_size) else tuple(new_size))
        resized = np.zeros(shape, dtype=self.data.dtype)
        overlap = tuple(slice(0, min(old, new))
                        for old, new in zip(self.data.shape, resized.shape))
        resized[overlap] = self.data[overlap]
        self.data = resized

    def resize_along_first(self, new_size):
        self.resize((new_size, *self.data.shape[1:]) if np.isscalar(new_size) else new_size)


class _RestoreOnlySpikeQueue:
    """Absorb Brian's runtime queue state while native queues are restored."""

    def _restore_from_full_state(self, state):
        require(state is None,
                "Rust Device checkpoints must not contain a runtime SpikeQueue")


class AtlasDevice(Device):
    """Own frontend arrays and dispatch a supported model to the selected Atlas engine.

    Python arrays hold initial values and completed results. There is no Python
    simulation loop, runtime-device delegation, or target-language templating.
    Consecutive runs preserve absolute time, monitor history, refractory state,
    and delayed events that cross a run boundary.
    """

    def __init__(self):
        super().__init__()
        self.arrays = WeakKeyDictionary()
        self.has_been_run = False
        self.last_run_directory = None
        self.last_build_timings = {}
        self.last_execution_plan = None
        self.last_gpu_tuning = None
        from .gpu_tuning_cache import TuningCache
        self._gpu_tuning_cache = TuningCache()
        self.last_runtime_binding = None
        self.native_artifact = None
        self._aot_binary_cache = {}
        self._run_count = 0
        self._network_id = None
        self._pending_events = {}
        self._pending_layouts = {}
        self._procedural_synapses = WeakKeyDictionary()
        self._binary_continuation_contracts = {}
        self._mpi_continuation_contract = None
        self._queued_network = None
        self._queued_model = None
        self._queued_segments = []
        self._queued_parameter_segments = False
        self._queued_objects = None
        self._queued_roots = None
        self._explicit_build_directory = None
        self._explicit_run_count = 0
        self._rng_seed = int.from_bytes(os.urandom(8), "little")
        self.last_capability_report = None
        self._gpu_executor = None
        self.last_monitor_stream = None
        self._monitor_stream_chunks = 0

    def __getstate__(self):
        """Serialize a CPU standalone device for multiprocessing workers."""
        require(self._gpu_executor is None,
                "an active GPU executor cannot be serialized")
        state = self.__dict__.copy()
        # WeakKeyDictionary stores weakref objects in its pickle state. The
        # variables and Synapses keys are already part of the queued Network
        # object graph, so temporarily retaining them strongly is both bounded
        # and necessary for a self-contained worker snapshot.
        state["arrays"] = [
            (variable, np.asarray(
                value.data if isinstance(variable, DynamicArrayVariable)
                else value).copy())
            for variable, value in self.arrays.items()
        ]
        state["_procedural_synapses"] = list(
            self._procedural_synapses.items())
        return state

    def __setstate__(self, state):
        arrays = state.pop("arrays")
        procedural = state.pop("_procedural_synapses")
        self.__dict__.update(state)
        self.arrays = WeakKeyDictionary()
        for variable, values in arrays:
            if isinstance(variable, DynamicArrayVariable):
                storage = _NumpyDynamicArray(values.shape, values.dtype)
                storage.data[...] = values
                self.arrays[variable] = storage
            else:
                self.arrays[variable] = values
        self._procedural_synapses = WeakKeyDictionary(procedural)

    def explain_plan(self, *, format="text"):
        """Explain the latest CPU/Metal plan and observations from its successful run.

        This does not run the model. Worker affinity and queue ownership are
        bound by the executable at runtime and are not predicted here.
        """
        from .plan import explain_plan
        if self.last_execution_plan is None:
            raise RuntimeError("no validated execution plan; build the model first")
        return explain_plan(self.last_execution_plan, format=format, binding=self.last_runtime_binding)

    def connect_fixed_total(self, synapses, edge_count, seed=None,
                            initializers=None, delay_initializer=None):
        """Attach a deferred fixed-total topology to a Brian2 Synapses object.

        This is a Device extension because Brian2's ``connect(p=...)`` is a
        Bernoulli rule and therefore does not have fixed-total semantics.
        """
        require(type(synapses) is Synapses,
                "connect_fixed_total expects a Brian2 Synapses object")
        require(not synapses._connect_called and len(synapses) == 0,
                "connect_fixed_total requires a fresh Synapses object")
        require(type(edge_count) is int and 1 <= edge_count <= 2**32 - 1,
                "edge_count must be an integer within 1..2^32-1")
        require(type(synapses.source) in
                (NeuronGroup, PoissonGroup, SpikeGeneratorGroup, Subgroup) and
                type(synapses.target) in (NeuronGroup, Subgroup),
                "unsupported fixed-total Synapses endpoint")
        if seed is None:
            seed = self._rng_seed ^ (
                (len(self._procedural_synapses) + 1) * 0x9E3779B97F4A7C15)
            seed &= 2**64 - 1
        require(type(seed) is int and 0 <= seed <= 2**64 - 1,
                "fixed-total seed must be an unsigned 64-bit integer")
        initializers = {} if initializers is None else dict(initializers)
        require(all(type(name) is str and name and
                    isinstance(initializer, (ClippedNormal, Uniform))
                    for name, initializer in initializers.items()),
                "initializers must map variable names to initializer descriptors")
        require(delay_initializer is None or
                isinstance(delay_initializer, (ClippedNormal, Uniform)),
                "delay_initializer must be an initializer descriptor or None")
        synapses._connect_called = True
        # Preserve the useful public len(S) without allocating any per-edge
        # Python arrays. Export rejects per-edge state until a procedural
        # initializer has been declared for it.
        synapses.variables["N"].set_value(edge_count)
        self._procedural_synapses[synapses] = {
            "kind": "fixed_total",
            "edge_count": edge_count,
            "seed": seed,
            "initializers": initializers,
            "delay_initializer": delay_initializer,
        }

    def connect_binary_csr(self, synapses, path, parameters):
        """Attach an immutable empirical CSR without allocating Python edges."""
        from .binary_topology import inspect_csr, file_hash
        require(type(synapses) is Synapses and not synapses._connect_called,
                "connect_binary_csr requires a fresh Synapses object")
        info = inspect_csr(path)
        require(info["source_count"] == len(synapses.source) and
                info["target_count"] == len(synapses.target),
                "binary CSR endpoint shape mismatch")
        parameters = dict(parameters)
        require(all(type(name) is str and type(column) is int and
                    0 <= column < info["column_count"]
                    for name, column in parameters.items()),
                "binary CSR parameters must map names to valid column indices")
        require(set(parameters.values()) == set(range(info["column_count"])),
                "every binary CSR column must be mapped")
        synapses._connect_called = True
        synapses.variables["N"].set_value(info["edge_count"])
        self._procedural_synapses[synapses] = {
            "kind": "binary_csr", "edge_count": info["edge_count"],
            "path": info["path"], "sha256": file_hash(info["path"]),
            "column_count": info["column_count"],
            "initializers": parameters, "delay_initializer": None,
        }

    def connect_fixed_indegree(self, synapses, indegree, seed=None,
                               initializers=None, delay_initializer=None):
        """Attach a deferred exact-indegree topology without multapses."""
        require(type(synapses) is Synapses,
                "connect_fixed_indegree expects a Brian2 Synapses object")
        require(not synapses._connect_called and len(synapses) == 0,
                "connect_fixed_indegree requires a fresh Synapses object")
        require(type(synapses.source) in
                (NeuronGroup, PoissonGroup, SpikeGeneratorGroup, Subgroup) and
                type(synapses.target) in (NeuronGroup, Subgroup),
                "unsupported fixed-indegree Synapses endpoint")
        source_count, target_count = len(synapses.source), len(synapses.target)
        require(type(indegree) is int and 1 <= indegree <= source_count,
                "indegree must be an integer within 1..source_count")
        edge_count = indegree * target_count
        require(edge_count <= 2**32 - 1,
                "fixed-indegree edge count must fit u32")
        if seed is None:
            seed = self._rng_seed ^ (
                (len(self._procedural_synapses) + 1) * 0x9E3779B97F4A7C15)
            seed &= 2**64 - 1
        require(type(seed) is int and 0 <= seed <= 2**64 - 1,
                "fixed-indegree seed must be an unsigned 64-bit integer")
        initializers = {} if initializers is None else dict(initializers)
        require(all(type(name) is str and name and
                    isinstance(initializer, (ClippedNormal, Uniform))
                    for name, initializer in initializers.items()),
                "initializers must map variable names to initializer descriptors")
        require(delay_initializer is None or
                isinstance(delay_initializer, (ClippedNormal, Uniform)),
                "delay_initializer must be an initializer descriptor or None")
        synapses._connect_called = True
        synapses.variables["N"].set_value(edge_count)
        self._procedural_synapses[synapses] = {
            "kind": "fixed_indegree",
            "edge_count": edge_count,
            "indegree": indegree,
            "seed": seed,
            "initializers": initializers,
            "delay_initializer": delay_initializer,
        }

    def procedural_synapse_topology(self, synapses):
        topology = self._procedural_synapses.get(synapses)
        return None if topology is None else topology.copy()

    def activate(self, build_on_run=True, **kwargs):
        require(type(build_on_run) is bool, "build_on_run must be boolean")
        require(not (set(kwargs) - {"directory", "runner", "engine", "profile",
                                    "threads", "thread_affinity",
                                    "retain_run_artifacts",
                                    "recording_window_steps", "monitor_streaming_steps",
                                    "numeric_mode", "event_delivery", "cuda_dag_execution", "metal_dag_execution", "gpu_buffer_reuse", "gpu_compile_reuse", "gpu_max_buffer_bytes", "gpu_synapse_prefix", "gpu_synapse_fusion", "gpu_synapse_sparse", "gpu_autotune", "gpu_autotune_cache", "ranks", "rank_backends"}),
                "Device options: directory, runner, engine, profile, threads and "
                "thread_affinity, retain_run_artifacts, recording_window_steps, "
                "monitor_streaming_steps, "
                "numeric_mode, event_delivery, cuda_dag_execution, metal_dag_execution, gpu_buffer_reuse, gpu_compile_reuse, gpu_max_buffer_bytes, gpu_synapse_prefix, gpu_synapse_fusion, gpu_synapse_sparse, gpu_autotune, gpu_autotune_cache, ranks and rank_backends only")
        require(type(kwargs.get('gpu_autotune_cache',False)) is bool,'gpu_autotune_cache must be boolean')
        require(not kwargs.get('gpu_autotune_cache',False) or kwargs.get('gpu_autotune',False) is True,
                'gpu_autotune_cache requires gpu_autotune=True')
        require('gpu_autotune_cache' not in kwargs or kwargs.get('engine') in {'metal','cuda'},
                'gpu_autotune_cache requires a GPU engine')
        require(type(kwargs.get('gpu_autotune',False)) is bool,'gpu_autotune must be boolean')
        require('gpu_autotune' not in kwargs or kwargs.get('engine') in {'metal','cuda'},
                'gpu_autotune requires a GPU engine')
        require(not kwargs.get('gpu_autotune',False) or not
                ({'gpu_synapse_prefix','gpu_synapse_fusion','gpu_synapse_sparse'} & kwargs.keys()),
                'gpu_autotune cannot be combined with explicit synapse policy options')
        sparse=kwargs.get('gpu_synapse_sparse',False)
        require(type(sparse) is bool or type(sparse) is str and sparse=='bitset',
                "gpu_synapse_sparse must be boolean or 'bitset'")
        require('gpu_synapse_sparse' not in kwargs or kwargs.get('engine') in {'metal','cuda'},
                'gpu_synapse_sparse requires a GPU engine')
        require(type(kwargs.get('gpu_synapse_fusion',False)) is bool,'gpu_synapse_fusion must be boolean')
        require('gpu_synapse_fusion' not in kwargs or kwargs.get('engine') in {'metal','cuda'},
                'gpu_synapse_fusion requires a GPU engine')
        require(type(kwargs.get('gpu_synapse_prefix',False)) is bool,'gpu_synapse_prefix must be boolean')
        require('gpu_synapse_prefix' not in kwargs or kwargs.get('engine') in {'metal','cuda'},
                'gpu_synapse_prefix requires a GPU engine')
        require(type(kwargs.get('gpu_compile_reuse',False)) is bool,'gpu_compile_reuse must be boolean')
        require('gpu_compile_reuse' not in kwargs or kwargs.get('engine') in {'metal','cuda'},
                'gpu_compile_reuse requires a GPU engine')
        require(type(kwargs.get('gpu_max_buffer_bytes',512*1024**2)) is int and
                kwargs.get('gpu_max_buffer_bytes',512*1024**2) > 0,
                'gpu_max_buffer_bytes must be a positive integer')
        require('gpu_max_buffer_bytes' not in kwargs or kwargs.get('engine') in {'metal','cuda'},
                'gpu_max_buffer_bytes requires a GPU engine')
        require(type(kwargs.get('gpu_buffer_reuse',False)) is bool,
                'gpu_buffer_reuse must be boolean')
        require('gpu_buffer_reuse' not in kwargs or kwargs.get('engine') in {'metal','cuda'},
                'gpu_buffer_reuse requires a GPU engine')
        require(kwargs.get("profile", False) in (True, False),
                "profile Device option must be boolean")
        require(type(kwargs.get("retain_run_artifacts", True)) is bool,
                "retain_run_artifacts Device option must be boolean")
        require(kwargs.get("engine", "reference") in {"reference", "aot", "metal", "cuda", "mpi"},
                "engine must be reference, aot, metal, cuda or mpi")
        require("ranks" not in kwargs or kwargs.get("engine") == "mpi",
                "ranks requires engine='mpi'")
        require(type(kwargs.get("ranks", 2)) is int and 1 <= kwargs.get("ranks", 2) <= 256,
                "ranks must be an integer in 1..256")
        require("rank_backends" not in kwargs or kwargs.get("engine") == "mpi",
                "rank_backends requires engine='mpi'")
        require(kwargs.get("numeric_mode", "reference-f64") in {"reference-f64", "float32", "mixed-f32"},
                "numeric_mode must be reference-f64, float32 or mixed-f32")
        require(kwargs.get("numeric_mode") != "mixed-f32" or kwargs.get("engine") == "mpi",
                "mixed-f32 requires engine='mpi'")
        if kwargs.get("engine") == "mpi":
            from .mpi_gpu import backend_policy
            backend_policy(kwargs.get("ranks", 2), kwargs.get("rank_backends"),
                           kwargs.get("numeric_mode", "reference-f64"))
        require((kwargs.get("engine") in {"metal","cuda"}) == (kwargs.get("numeric_mode") == "float32"),
                "Metal/CUDA require explicit numeric_mode='float32'; CPU retains reference-f64")
        require(kwargs.get("event_delivery", "scan") in {"scan", "sparse"},
                "event_delivery must be scan or sparse")
        require("event_delivery" not in kwargs or kwargs.get("engine") in {"metal","cuda"},
                "event_delivery requires engine='metal' or engine='cuda'")
        require(kwargs.get("cuda_dag_execution","auto") in {"auto","direct","resident","graph","chunked","workgroup","cooperative"},
                "cuda_dag_execution must be auto, direct, resident, graph, chunked, workgroup or cooperative")
        require("cuda_dag_execution" not in kwargs or kwargs.get("engine")=="cuda",
                "cuda_dag_execution requires engine='cuda'")
        require(kwargs.get("metal_dag_execution","auto") in {"auto","direct","resident","workgroup","indirect"},
                "metal_dag_execution must be auto, direct, resident, workgroup or indirect")
        require("metal_dag_execution" not in kwargs or kwargs.get("engine")=="metal",
                "metal_dag_execution requires engine='metal'")
        threads = kwargs.get("threads", 1)
        require(type(threads) is int and 1 <= threads <= 256,
                "threads must be an integer within 1..256")
        require(threads == 1 or kwargs.get("engine", "reference") == "aot",
                "threads > 1 requires engine='aot'")
        require(kwargs.get("thread_affinity", "auto") in
                {"auto", "off", "required"},
                "thread_affinity must be auto, off, or required")
        recording_window = kwargs.get("recording_window_steps")
        require(recording_window is None or
                (type(recording_window) is int and 1 <= recording_window <= 10_000_000),
                "recording_window_steps must be None or an integer within 1..10,000,000")
        streaming_steps = kwargs.get("monitor_streaming_steps")
        require(streaming_steps is None or
                (type(streaming_steps) is int and
                 1 <= streaming_steps <= 10_000_000),
                "monitor_streaming_steps must be None or an integer within "
                "1..10,000,000")
        require(streaming_steps is None or recording_window is None,
                "monitor_streaming_steps cannot be combined with "
                "recording_window_steps")
        require(streaming_steps is None or build_on_run,
                "monitor streaming requires build_on_run=True")
        require(streaming_steps is None or kwargs.get("directory") is not None,
                "monitor streaming requires an explicit Device directory")
        require(streaming_steps is None or
                kwargs.get("engine", "reference") in {"reference", "aot"},
                "monitor streaming currently requires reference or AOT CPU")
        self.close_gpu()
        self.clear_gpu_tuning_cache()
        self.last_gpu_tuning = None
        super().activate(build_on_run=build_on_run, **kwargs)

    def close_gpu(self):
        """Release retained GPU allocations; host results and clocks stay valid."""
        gpu=getattr(self,'_gpu_executor',None)
        self._gpu_executor=None
        if gpu is not None:gpu.close()

    def clear_gpu_tuning_cache(self):
        """Forget tuning decisions without releasing GPU resources or host state."""
        self._gpu_tuning_cache.clear()

    def reinit(self):
        """Forget old model ownership; call activate() before constructing a new model."""
        self.close_gpu()
        self.__init__()

    def add_array(self, var):
        if isinstance(var, DynamicArrayVariable):
            cls = DynamicArray1D if var.ndim == 1 else DynamicArray
            try:
                self.arrays[var] = cls(var.size, dtype=var.dtype)
            except KeyError:
                self.arrays[var] = _NumpyDynamicArray(var.size, var.dtype)
        else:
            self.arrays[var] = np.zeros(var.size, dtype=var.dtype)

    def get_value(self, var, access_data=True):
        if var not in self.arrays:
            raise RuntimeError("Array belongs to a previous Rust Device initialization; recreate the model")
        value = self.arrays[var]
        return value.data if isinstance(var, DynamicArrayVariable) and access_data else value

    def get_len(self, var):
        return var.size

    def get_array_name(self, var, access_data=True):
        prefix = "array" if access_data else "dynamic_array"
        owner_name = getattr(var.owner, "name", "temporary")
        return f"_{prefix}_{owner_name}_{var.name}"

    def fill_with_array(self, var, value):
        self.get_value(var)[...] = value

    set_value = fill_with_array

    def init_with_zeros(self, var, dtype):
        self.fill_with_array(var, 0)

    def init_with_arange(self, var, start, dtype):
        self.fill_with_array(var, np.arange(start, start + var.size, dtype=dtype))

    def resize(self, var, new_size):
        self.get_value(var, access_data=False).resize(new_size)

    def resize_along_first(self, var, new_size):
        self.get_value(var, access_data=False).resize_along_first(new_size)

    def spike_queue(self, source_start, source_end):
        # Queue ownership lives in the standalone runner. Returning None keeps
        # Brian from attaching its runtime Cython queue/capsule to the pathway.
        return None

    def seed(self, seed=None):
        """Seed topology generation and one-time frontend expressions."""
        np.random.seed(seed)
        self._rng_seed = (int.from_bytes(os.urandom(8), "little")
                          if seed is None else int(seed))

    def get_random_state(self):
        """Return frontend NumPy and counter-RNG state for Network.store."""
        return {"numpy_state": np.random.get_state(),
                "counter_seed": self._rng_seed}

    def set_random_state(self, state):
        """Restore a state produced by :meth:`get_random_state`."""
        require(set(state) == {"numpy_state", "counter_seed"},
                "invalid Rust Device random state")
        np.random.set_state(state["numpy_state"])
        seed = state["counter_seed"]
        require(type(seed) is int and 0 <= seed <= 2**64 - 1,
                "invalid Rust Device counter seed")
        self._rng_seed = seed

    def network_store(self, network, name="default", filename=None):
        """Store Brian state plus delayed-event continuation metadata."""
        require(type(name) is str and name, "checkpoint name must be non-empty")
        mpi_contract = self._mpi_checkpoint_contract(network)
        if self._mpi_continuation_contract is not None and self._network_id == network.id:
            from .mpi_checkpoint import verify
            verify(self._mpi_continuation_contract, mpi_contract or {})
        Network.store.original_function(network, name=name, filename=None)
        state = network._stored_state[name]
        state["_rust_device_state"] = {
            "pending_events": copy.deepcopy(self._pending_events),
            "pending_layouts": copy.deepcopy(self._pending_layouts),
            "binary_contracts": copy.deepcopy(self._binary_continuation_contracts),
            "mpi_contract": mpi_contract,
        }
        if filename is not None:
            path = Path(filename)
            if path.exists():
                stored = _read_checkpoint(path)
            else:
                stored = {}
            stored[name] = state
            _write_checkpoint(path, stored)
            del network._stored_state[name]

    def network_restore(self, network, name="default", filename=None,
                        restore_random_state=False):
        """Restore Brian arrays, time, RNG and native pending delay queues."""
        require(type(name) is str and name, "checkpoint name must be non-empty")
        if filename is None:
            state = network._stored_state[name]
        else:
            state = _read_checkpoint(filename)[name]
        require("_rust_device_state" in state,
                "checkpoint does not contain Rust Device continuation state")
        device_state = copy.deepcopy(state["_rust_device_state"])
        current_contract = self._mpi_checkpoint_contract(network)
        if current_contract is not None or device_state.get('mpi_contract') is not None:
            from .mpi_checkpoint import verify
            verify(device_state.get('mpi_contract'), current_contract or {})
        restored = {key: value for key, value in state.items()
                    if key != "_rust_device_state"}
        temporary = f"__rust_restore_{id(restored)}"
        require(temporary not in network._stored_state,
                "temporary checkpoint name collision")
        network._stored_state[temporary] = restored
        queue_owners = [obj for obj in _get_all_objects(network.objects)
                        if hasattr(obj, "queue") and obj.queue is None]
        for owner in queue_owners:
            owner.queue = _RestoreOnlySpikeQueue()
        try:
            Network.restore.original_function(
                network, name=temporary, filename=None,
                restore_random_state=restore_random_state)
        finally:
            for owner in queue_owners:
                owner.queue = None
            del network._stored_state[temporary]
        self._pending_events = device_state["pending_events"]
        self._pending_layouts = device_state["pending_layouts"]
        self._binary_continuation_contracts = device_state.get("binary_contracts", {})
        self._mpi_continuation_contract = device_state.get('mpi_contract')
        if not restore_random_state:
            # Brian's contract requires a fresh stream when RNG restoration is
            # not requested. Counter draws remain worker-count independent.
            self._rng_seed = int.from_bytes(os.urandom(8), "little")

    def _mpi_checkpoint_contract(self, network):
        if self.build_options.get('engine') != 'mpi':
            return None
        from .mpi_checkpoint import contract
        with packed_export():
            model = lower_network(network, 0 * second, rng_seed=self._rng_seed)
        return contract(model, self.build_options, self._runner())

    @staticmethod
    def _connection_expression(expression, synapses, sources, targets,
                               source_count, target_count, namespace,
                               extra_values=None):
        """Evaluate a side-effect-free frontend connection expression."""
        values = {
            "i": sources,
            "j": targets,
            "N_pre": source_count,
            "N_post": target_count,
        }
        if extra_values:
            values.update(extra_values)
        namespace = {} if namespace is None else namespace
        source_offset = int(synapses.variables["_source_offset"].get_value())
        target_offset = int(synapses.variables["_target_offset"].get_value())

        def endpoint_value(name, endpoint, indices, offset):
            variable = endpoint.variables[name]
            raw = variable.get_value()
            if variable.scalar:
                value = np.asarray(raw).reshape(-1)[0]
            else:
                value = np.asarray(raw)[np.asarray(indices) + offset]
            return Quantity(value, dim=variable.dim)

        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in (bool, int, float):
                return node.value
            if isinstance(node, ast.Name):
                if node.id in values:
                    return values[node.id]
                if (node.id.endswith("_pre") and
                        node.id[:-4] in synapses.source.variables):
                    return endpoint_value(
                        node.id[:-4], synapses.source, sources, source_offset)
                if (node.id.endswith("_post") and
                        node.id[:-5] in synapses.target.variables):
                    return endpoint_value(
                        node.id[:-5], synapses.target, targets, target_offset)
                require(node.id in namespace,
                        f"connect expression name {node.id!r} is unsupported")
                raw = namespace[node.id]
                value = np.asanyarray(raw)
                require(value.size == 1 and value.dtype.kind in "fiub",
                        "connect expression constants must be numeric or boolean scalars")
                return raw if isinstance(raw, Quantity) else value.reshape(-1)[0]
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                functions = {
                    "abs": np.abs, "ceil": np.ceil, "cos": np.cos,
                    "exp": np.exp, "floor": np.floor, "log": np.log,
                    "int": lambda value: np.asarray(value, dtype=np.int64),
                    "sin": np.sin, "sqrt": np.sqrt,
                }
                if node.func.id == "rand":
                    require(not node.args and not node.keywords,
                            "connect rand() takes no arguments")
                    shape = np.broadcast_arrays(
                        np.asarray(sources), np.asarray(targets))[0].shape
                    return np.random.random(shape)
                require(node.func.id in functions and len(node.args) == 1 and
                        not node.keywords,
                        "unsupported connect expression function")
                return functions[node.func.id](evaluate(node.args[0]))
            if isinstance(node, ast.UnaryOp):
                value = evaluate(node.operand)
                if isinstance(node.op, ast.Not):
                    return np.logical_not(value)
                if isinstance(node.op, ast.USub):
                    return -value
                if isinstance(node.op, ast.UAdd):
                    return value
            binary = {
                ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
                ast.Div: np.divide, ast.FloorDiv: np.floor_divide,
                ast.Mod: np.mod, ast.Pow: np.power,
            }
            if isinstance(node, ast.BinOp) and type(node.op) in binary:
                return binary[type(node.op)](evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                operation = (np.logical_and if isinstance(node.op, ast.And)
                             else np.logical_or)
                result = evaluate(node.values[0])
                for value in node.values[1:]:
                    result = operation(result, evaluate(value))
                return result
            comparisons = {
                ast.Eq: np.equal, ast.NotEq: np.not_equal,
                ast.Gt: np.greater, ast.GtE: np.greater_equal,
                ast.Lt: np.less, ast.LtE: np.less_equal,
            }
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and \
                    type(node.ops[0]) in comparisons:
                return comparisons[type(node.ops[0])](
                    evaluate(node.left), evaluate(node.comparators[0]))
            raise NotImplementedError(
                f"Atlas connect expression is unsupported: {ast.dump(node)}")

        return evaluate(ast.parse(expression, mode="eval").body)

    @classmethod
    def _connection_condition(cls, condition, synapses, sources, targets,
                              source_count, target_count, namespace,
                              extra_values=None):
        result = np.asarray(cls._connection_expression(
            condition, synapses, sources, targets, source_count, target_count,
            namespace, extra_values))
        require(result.dtype.kind == "b", "connect condition must be boolean")
        try:
            return np.broadcast_to(result, targets.shape)
        except ValueError as error:
            raise ValueError("connect condition has an invalid shape") from error

    @classmethod
    def _connection_probability(cls, expression, synapses, sources, targets,
                                source_count, target_count, namespace):
        result = cls._connection_expression(
            expression, synapses, sources, targets, source_count, target_count,
            namespace)
        require(not isinstance(result, Quantity) or result.dim is DIMENSIONLESS,
                "connect probability must be dimensionless")
        result = np.asarray(result, dtype=np.float64)
        try:
            result = np.broadcast_to(result, targets.shape)
        except ValueError as error:
            raise ValueError("connect probability has an invalid shape") from error
        # Brian evaluates string probabilities as ``rand() < p`` without
        # clipping: values above one always connect and values below zero do
        # not.  Preserve that expression semantics while rejecting NaN/Inf.
        require(np.all(np.isfinite(result)),
                "connect probability expression must be finite")
        return result

    def _connection_generator_targets(self, synapses, expression, namespace):
        """Resolve an optional deterministic target per presynaptic neuron."""
        source_count = len(synapses.source)
        source_offset = int(synapses.variables["_source_offset"].get_value())
        indices = np.arange(source_count, dtype=np.int64)

        def evaluate(node):
            if isinstance(node, ast.Name):
                if node.id == "i":
                    return indices
                if node.id.endswith("_pre"):
                    name = node.id[:-4]
                    require(name in synapses.source.variables,
                            f"unknown source variable in Synapses.connect: {name}")
                    values = np.asarray(
                        synapses.source.variables[name].get_value())
                    require(values.ndim == 1 and
                            len(values) >= source_offset + source_count,
                            f"invalid source variable for Synapses.connect: {name}")
                    return values[source_offset:source_offset + source_count]
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return node.value
            if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
                value = evaluate(node.operand)
                return value if type(node.op) is ast.UAdd else -value
            if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub):
                left, right = evaluate(node.left), evaluate(node.right)
                return left + right if type(node.op) is ast.Add else left - right
            raise NotImplementedError(
                f"Synapses.connect(j=...) generator is unsupported: {ast.dump(node)}")

        condition = None
        try:
            parsed = ast.parse(expression, mode="eval").body
        except SyntaxError as original_error:
            # Brian uses ``element if condition`` as a generator shorthand:
            # a false condition skips the source instead of needing an else
            # value. Parse it as a temporary Python conditional expression so
            # nested comparisons and parentheses retain their normal grammar.
            try:
                conditional = ast.parse(
                    f"({expression} else -1)", mode="eval").body
            except SyntaxError as error:
                raise NotImplementedError(
                    "Synapses.connect(j=...) generator comprehensions are unsupported"
                ) from error
            if not isinstance(conditional, ast.IfExp):
                raise NotImplementedError(
                    "Synapses.connect(j=...) generator comprehensions are unsupported"
                ) from original_error
            parsed = conditional.body
            condition = ast.unparse(conditional.test)
        generated = np.asarray(evaluate(parsed))
        require(generated.dtype.kind in "iu",
                "Synapses.connect(j=...) must generate integer target indices")
        try:
            generated = np.broadcast_to(generated, (source_count,)).astype(
                np.int64, copy=False)
        except ValueError as error:
            raise ValueError("Synapses.connect(j=...) has invalid shape") from error
        if condition is None:
            return indices, generated
        selected = self._connection_condition(
            condition, synapses, indices, generated, source_count,
            len(synapses.target), namespace)
        return indices[selected], generated[selected]

    @staticmethod
    def _structured_equality_connections(synapses, condition):
        """Recognize index-to-endpoint-label equalities without an N*M scan."""
        if not isinstance(condition, str):
            return None
        try:
            parsed = ast.parse(condition, mode="eval").body
        except SyntaxError:
            return None
        if (not isinstance(parsed, ast.Compare) or len(parsed.ops) != 1 or
                not isinstance(parsed.ops[0], ast.Eq) or
                len(parsed.comparators) != 1 or
                not isinstance(parsed.left, ast.Name) or
                not isinstance(parsed.comparators[0], ast.Name)):
            return None
        left, right = parsed.left.id, parsed.comparators[0].id
        pairs = {(left, right), (right, left)}
        source_count, target_count = len(synapses.source), len(synapses.target)

        def endpoint_values(endpoint, name, offset, count):
            if name not in endpoint.variables:
                return None
            values = np.asarray(endpoint.variables[name].get_value()).reshape(-1)
            if values.size < offset + count or values.dtype.kind not in "iu":
                return None
            return values[offset:offset + count].astype(np.int64, copy=False)

        if any(index == "i" and label.endswith("_post")
               for index, label in pairs):
            label = next(label for index, label in pairs
                         if index == "i" and label.endswith("_post"))
            offset = int(synapses.variables["_target_offset"].get_value())
            sources = endpoint_values(
                synapses.target, label[:-5], offset, target_count)
            if sources is None:
                return None
            targets = np.arange(target_count, dtype=np.int64)
            valid = (sources >= 0) & (sources < source_count)
            sources, targets = sources[valid], targets[valid]
        elif any(label.endswith("_pre") and index == "j"
                 for label, index in pairs):
            label = next(label for label, index in pairs
                         if label.endswith("_pre") and index == "j")
            offset = int(synapses.variables["_source_offset"].get_value())
            targets = endpoint_values(
                synapses.source, label[:-4], offset, source_count)
            if targets is None:
                return None
            sources = np.arange(source_count, dtype=np.int64)
            valid = (targets >= 0) & (targets < target_count)
            sources, targets = sources[valid], targets[valid]
        else:
            return None
        order = np.lexsort((targets, sources))
        return sources[order], targets[order]

    @staticmethod
    def _sample_connection_range(low, high, step, probability=None, size=None,
                                 skip_if_invalid=False):
        values = np.arange(low, high, step, dtype=np.int64)
        if size is not None:
            if size < 0:
                if skip_if_invalid:
                    return values[:0]
                raise IndexError("Requested sample size is negative")
            if size > values.size:
                if not skip_if_invalid:
                    raise IndexError("Requested sample size exceeds iterator size")
                size = values.size
            if size == values.size:
                return values
            return np.sort(np.random.choice(values, size=size, replace=False))
        if probability == 0 or values.size == 0:
            return values[:0]
        if probability == 1:
            return values
        require(probability is not None and np.isfinite(probability) and
                0 <= probability <= 1,
                "sample probability must be finite and between zero and one")
        # Match Brian's direct path for ordinary iterators. For sparse, large
        # iterators use geometric jumps so official connectivity examples do
        # not accidentally become O(N^2).
        if values.size <= 1000 or probability > 0.25:
            return values[np.random.random(values.size) < probability]
        selected = []
        position = -1
        scale = 1.0 / np.log1p(-probability)
        while True:
            jump = int(np.ceil(np.log(np.random.random()) * scale))
            position += jump
            if position >= values.size:
                break
            selected.append(position)
        return values[np.asarray(selected, dtype=np.int64)]

    def _connection_generator_pairs(self, synapses, expression,
                                    skip_if_invalid, namespace):
        """Materialize Brian's bounded postsynaptic generator subset."""
        try:
            parsed = parse_synapse_generator(expression)
        except SyntaxError as error:
            raise NotImplementedError(
                "Synapses.connect(j=...) generator syntax is unsupported") from error
        inner = parsed["inner_variable"]
        require(parsed["iterator_func"] in {"range", "sample"},
                "only range/sample connection iterators are supported")
        source_count, target_count = len(synapses.source), len(synapses.target)
        if namespace is None:
            namespace = {}

        def scalar(source, code, label, integer=False):
            value = self._connection_expression(
                code, synapses, source, 0, source_count, target_count,
                namespace)
            require(not isinstance(value, Quantity) or value.dim is DIMENSIONLESS,
                    f"generator {label} must be dimensionless")
            array = np.asarray(value)
            require(array.size == 1 and np.isfinite(float(array.reshape(-1)[0])),
                    f"generator {label} must be a finite scalar")
            number = float(array.reshape(-1)[0])
            if integer:
                require(number.is_integer(), f"generator {label} must be integer")
                return int(number)
            return number

        source_chunks, target_chunks = [], []
        iterator = parsed["iterator_kwds"]
        for source in range(source_count):
            low = scalar(source, iterator["low"], "lower bound", integer=True)
            high = scalar(source, iterator["high"], "upper bound", integer=True)
            step = scalar(source, iterator["step"], "step", integer=True)
            require(step != 0, "generator step must be non-zero")
            if parsed["iterator_func"] == "range":
                generated = np.arange(low, high, step, dtype=np.int64)
            elif iterator["sample_size"] == "random":
                probability = scalar(source, iterator["p"], "sample probability")
                generated = self._sample_connection_range(
                    low, high, step, probability=probability,
                    skip_if_invalid=skip_if_invalid)
            else:
                size = scalar(source, iterator["size"], "sample size", integer=True)
                generated = self._sample_connection_range(
                    low, high, step, size=size,
                    skip_if_invalid=skip_if_invalid)
            if not generated.size:
                continue
            extra = {inner: generated}
            targets = np.asarray(self._connection_expression(
                parsed["element"], synapses, source, generated,
                source_count, target_count, namespace, extra))
            try:
                targets = np.broadcast_to(targets, generated.shape)
            except ValueError as error:
                raise ValueError("generator element has invalid shape") from error
            require(targets.dtype.kind in "iu",
                    "generator element must produce integer indices")
            targets = targets.astype(np.int64, copy=False)
            condition = parsed["if_expression"]
            if condition != "True":
                selected = self._connection_condition(
                    condition, synapses, source, targets,
                    source_count, target_count, namespace,
                    {inner: generated})
                generated, targets = generated[selected], targets[selected]
                if not targets.size:
                    continue
            valid = (targets >= 0) & (targets < target_count)
            if not np.all(valid) and not skip_if_invalid:
                raise IndexError("Postsynaptic generator index outside target group")
            generated, targets = generated[valid], targets[valid]
            if not targets.size:
                continue
            if targets.size:
                source_chunks.append(np.full(targets.size, source, dtype=np.int64))
                target_chunks.append(targets)
        return ((np.concatenate(source_chunks) if source_chunks else
                 np.empty(0, dtype=np.int64)),
                (np.concatenate(target_chunks) if target_chunks else
                 np.empty(0, dtype=np.int64)))

    def synapses_connect(self, synapses, condition=None, i=None, j=None, p=1.0,
                         n=1, skip_if_invalid=False, namespace=None, level=0):
        """Materialize deterministic or seeded probabilistic static edges."""
        synapses._verify_connect_argument_types(condition, i, j, n, p)
        require(not isinstance(i, str),
                "Synapses.connect(i=...) generator is unsupported")
        require(not skip_if_invalid or isinstance(j, str),
                "skip_if_invalid requires a postsynaptic generator")
        require((type(synapses.source) in
                 (NeuronGroup, PoissonGroup, SpikeGeneratorGroup, SpatialNeuron,
                  Synapses) or
                 isinstance(synapses.source, Subgroup)) and
                (type(synapses.target) in
                 (NeuronGroup, PoissonGroup, SpikeGeneratorGroup, SpatialNeuron,
                  Synapses) or
                 isinstance(synapses.target, Subgroup)),
                "Synapses must connect a NeuronGroup/PoissonGroup/"
                "SpikeGeneratorGroup source to a "
                "population target, optionally through contiguous subgroups")
        probability = None if isinstance(p, str) else float(p)
        if probability is not None:
            require(np.isfinite(probability) and 0 <= probability <= 1,
                    "Synapses.connect probability p must be between 0 and 1")
        old = len(synapses)
        if condition is None and i is None and j is None:
            condition = True
        if condition is not None:
            require(i is None and j is None,
                    "cannot combine a connect condition with i or j")
            require(type(n) is int and n >= 0,
                    "condition-based Synapses.connect multiplicity n must be "
                    "a non-negative integer")
            synapses._connect_called = True
            if condition is False or condition == "False":
                return
            source_count, target_count = len(synapses.source), len(synapses.target)
            if namespace is None:
                namespace = get_local_namespace(level=level + 2)
            structured = (None if isinstance(p, str) else
                          self._structured_equality_connections(
                              synapses, condition))
            if structured is not None:
                sources, targets = structured
                if probability == 0:
                    sources, targets = sources[:0], targets[:0]
                elif probability != 1:
                    selected = np.random.random(sources.size) < probability
                    sources, targets = sources[selected], targets[selected]
                sources, targets = np.repeat(sources, n), np.repeat(targets, n)
            else:
                require(source_count * target_count <= candidate_pair_budget(),
                        f"at most {candidate_pair_budget():,} candidate connection pairs")
                target_indices = np.arange(target_count, dtype=np.int64)
                source_chunks, target_chunks = [], []
                selected_count = 0
                for source in range(source_count):
                    selected = (np.ones(target_count, dtype=np.bool_)
                                if condition is True else
                                self._connection_condition(
                                    condition, synapses, source, target_indices,
                                    source_count, target_count, namespace))
                    if isinstance(p, str):
                        probabilities = self._connection_probability(
                            p, synapses, source, target_indices, source_count,
                            target_count, namespace)
                        selected = selected & (
                            np.random.random(target_count) < probabilities)
                    elif probability == 0:
                        selected = np.zeros(target_count, dtype=np.bool_)
                    elif probability != 1:
                        selected = selected & (
                            np.random.random(target_count) < probability)
                    selected_targets = target_indices[selected]
                    if selected_targets.size:
                        selected_targets = np.repeat(selected_targets, n)
                        selected_count += selected_targets.size
                        require(old + selected_count <= explicit_synapse_budget(),
                                f"at most {explicit_synapse_budget():,} explicit synapses")
                        source_chunks.append(np.full(
                            selected_targets.size, source, dtype=np.int64))
                        target_chunks.append(selected_targets)
                sources = (np.concatenate(source_chunks) if source_chunks
                           else np.empty(0, dtype=np.int64))
                targets = (np.concatenate(target_chunks) if target_chunks
                           else np.empty(0, dtype=np.int64))
        elif isinstance(j, str):
            require(not isinstance(p, str),
                    "string probability with j generator is unsupported")
            require(i is None and condition is None,
                    "Synapses.connect(j=...) cannot combine with i or condition")
            require(type(n) is int and n >= 0,
                    "Synapses.connect(j=...) requires non-negative integer n")
            if namespace is None:
                namespace = get_local_namespace(level=level + 2)
            try:
                parse_synapse_generator(j)
            except SyntaxError:
                is_generator = False
            else:
                is_generator = True
            if is_generator:
                sources, targets = self._connection_generator_pairs(
                    synapses, j, skip_if_invalid, namespace)
            else:
                require(not skip_if_invalid,
                        "skip_if_invalid requires generator-comprehension syntax")
                sources, targets = self._connection_generator_targets(
                    synapses, j, namespace)
            sources, targets = np.repeat(sources, n), np.repeat(targets, n)
            if probability != 1:
                selected = np.random.random(sources.size) < probability
                sources, targets = sources[selected], targets[selected]
        else:
            require(not isinstance(p, str),
                    "string probability with explicit i/j is unsupported")
            require(i is not None and j is not None,
                    "Synapses.connect requires both i and j")
            i, j, n = synapses._verify_connect_array_arguments(i, j, n)
            require(np.all(n >= 0),
                    "Synapses.connect multiplicity n must be non-negative")
            sources = np.repeat(np.atleast_1d(i), n)
            targets = np.repeat(np.atleast_1d(j), n)
            if probability != 1:
                selected = np.random.random(sources.size) < probability
                sources, targets = sources[selected], targets[selected]
        require(sources.ndim == targets.ndim == 1 and sources.size == targets.size,
                "Synapses i/j arrays must be one-dimensional and equally sized")
        if sources.size:
            if np.any(sources < 0) or np.any(sources >= len(synapses.source)):
                raise IndexError("Presynaptic index outside the source group")
            if np.any(targets < 0) or np.any(targets >= len(synapses.target)):
                raise IndexError("Postsynaptic index outside the target group")
        source_offset = int(synapses.variables["_source_offset"].get_value())
        target_offset = int(synapses.variables["_target_offset"].get_value())
        sources = sources + source_offset
        targets = targets + target_offset
        require(old + sources.size <= explicit_synapse_budget(),
                f"at most {explicit_synapse_budget():,} explicit synapses")
        synapses._connect_called = True
        synapses._resize(old + sources.size)
        synapses.variables["_synaptic_pre"].get_value()[old:] = sources
        synapses.variables["_synaptic_post"].get_value()[old:] = targets
        synapses._update_synapse_numbers(old)
        for var in synapses._registered_variables:
            var.size = len(var.get_value())
        synapses.variables["N_incoming"].size = len(
            synapses.variables["N_incoming"].get_value())
        synapses.variables["N_outgoing"].size = len(
            synapses.variables["N_outgoing"].get_value())

    def code_object_class(self, codeobj_class=None, fallback_pref="codegen.target"):
        if codeobj_class is NumpyCodeObject or (
                codeobj_class is None and
                fallback_pref == "codegen.string_expression_target"):
            # String initialization is a one-time frontend operation. It writes
            # this Device's owned arrays and does not participate in simulation.
            return NumpyCodeObject
        raise NotImplementedError(
            "Atlas Device executes whole networks; custom CodeObjects and "
            "runtime-generated model code are unsupported"
        )

    def insert_code(self, *args, **kwargs):
        raise NotImplementedError("Atlas Device does not support inserted native code")

    def build(self, directory=None, run=True, **kwargs):
        require(not kwargs,
                "Rust Device build supports only directory and run options")
        require(type(run) is bool, "build run option must be boolean")
        require(not self.build_on_run,
                "explicit build requires set_device(..., build_on_run=False)")
        require(self._queued_network is not None and self._queued_model is not None,
                "no queued Network.run calls to build")
        require(not self.has_been_run and self._explicit_build_directory is None,
                "queued Rust network has already been built")
        if directory is not None:
            require(self.build_options.get("directory") is None,
                    "directory was specified both at activation and build")
            self.build_options["directory"] = directory
        require(run or self.build_options.get("engine") in {"reference", "aot"},
                "build(run=False) currently requires reference or AOT engine")
        if self._queued_parameter_segments:
            require(run,
                    "queued parameter changes require build(run=True)")
            require(self.build_options.get("engine", "reference") in
                    {"reference", "aot"},
                    "queued parameter changes require reference or AOT engine")
            previous_results = None
            for segment in self._queued_segments:
                segment = copy.deepcopy(segment)
                if previous_results is not None:
                    self._continue_queued_segment(segment, previous_results)
                self._inject_pending_events(segment)
                previous_results = self._execute_model(
                    self._queued_network, segment, self._queued_objects,
                    advance_clock=False)
            return
        self._execute_model(self._queued_network, self._queued_model,
                            self._queued_objects, advance_clock=False,
                            compile_only=not run)

    @staticmethod
    def _run_arg_storage(model, key):
        require(isinstance(key, VariableView),
                "run_args keys must be VariableView objects")
        collections = (
            (model["definition"]["populations"],
             model["instance"]["populations"]),
            (model["definition"]["synapses"],
             model["instance"]["synapses"]),
        )
        for definitions, instances in collections:
            for definition, instance in zip(definitions, instances, strict=True):
                if definition["name"] != key.group_name:
                    continue
                for layer in ("states", "parameters"):
                    symbol = next((item for item in definition[layer]
                                   if item["name"] == key.name), None)
                    if symbol is None:
                        continue
                    storage = (instance["initial_state"] if layer == "states"
                               else instance["parameters"])
                    require(key.name in storage,
                            f"{key.group_name}.{key.name} is not mutable at replay")
                    return storage, symbol
                require(False,
                        f"{key.group_name}.{key.name} is not stored in the model")
        require(False, f"run_args group {key.group_name!r} is not in the model")

    def _apply_run_args(self, model, run_args):
        if run_args is None:
            return
        require(isinstance(run_args, Mapping),
                "run_args must be a mapping from VariableView to values")
        for key, value in run_args.items():
            fail_for_dimension_mismatch(key.dim, value)
            storage, symbol = self._run_arg_storage(model, key)
            values = np.asarray(value, dtype=key.variable.dtype).reshape(-1)
            expected = len(storage[key.name])
            if values.size == 1 and expected != 1:
                values = np.repeat(values, expected)
            if values.size != expected:
                raise TypeError(
                    f"Incorrect size for variable '{key.group_name}.{key.name}': "
                    f"expected {expected}, got {values.size}")
            storage[key.name] = _array_bits(values, symbol["dtype"])

    def _clear_queued_monitors(self):
        for obj in self._queued_roots or ():
            if type(obj) not in (StateMonitor, EventMonitor, SpikeMonitor,
                                 PopulationRateMonitor):
                continue
            obj.resize(0)
            if "N" in obj.variables:
                obj.variables["N"].set_value(0)
            if "count" in obj.variables:
                obj.variables["count"].set_value(
                    np.zeros(len(obj.source), dtype=obj.variables["count"].dtype))

    def run(self, directory=None, results_directory=None, with_output=True,
            run_args=None):
        """Replay one queued standalone model with optional initial overrides."""
        require(not self.build_on_run,
                "explicit run requires set_device(..., build_on_run=False)")
        require(self._explicit_build_directory is not None,
                "call build(run=False) before run")
        require(type(with_output) is bool, "with_output must be boolean")
        build_directory = self._explicit_build_directory
        if directory is not None:
            require(Path(directory).expanduser().resolve() == build_directory,
                    "run directory must match the explicit build directory")
        next_run = self._explicit_run_count + 1
        if results_directory is None:
            output = build_directory / f"run-{next_run:04d}"
        else:
            relative = Path(results_directory)
            require(not relative.is_absolute() and ".." not in relative.parts,
                    "results_directory must be a relative child path")
            output = build_directory / relative
        model = copy.deepcopy(self._queued_model)
        self._apply_run_args(model, run_args)
        self._explicit_run_count = next_run
        self._pending_events = {}
        self._pending_layouts = {}
        self._binary_continuation_contracts = {}
        configured = self.build_options.get("directory")
        run_count = self._run_count
        try:
            self.build_options["directory"] = output
            self._run_count = 0
            self._execute_model(
                self._queued_network, model, self._queued_objects,
                advance_clock=False, replace_monitor_results=True)
        finally:
            self.build_options["directory"] = configured
            self._run_count = run_count

    def _check_ownership(self, network):
        for obj in network.sorted_objects:
            variables = list(getattr(obj, "variables", {}).values())
            variables += list(obj.clock.variables.values())
            for var in variables:
                if isinstance(var, ArrayVariable) and (var.device is not self or var not in self.arrays):
                    raise RuntimeError(
                        f"{obj.name}.{var.name} belongs to another Device or an old initialization. "
                        "Call set_device('atlas') before constructing clocks, groups and monitors."
                    )

    def _prepare_network_identity(self, network):
        """Reset physical state when the next Network contains only new objects."""
        if self._network_id in (None, network.id):
            return
        if (type(network) is not MagicNetwork and
                any(obj._network is not None for obj in network.sorted_objects)):
            raise RuntimeError(
                "A Rust Device can switch Networks only when every object in "
                "the new Network is fresh")
        # MagicNetwork assigns itself a new id only after proving that all
        # visible objects are new. Explicit Network objects are checked above.
        # Treat either case as a fresh physical model while retaining
        # Device-owned arrays so preceding results remain readable.
        self.close_gpu()
        self.has_been_run = False
        self.native_artifact = None
        self.last_execution_plan = None
        self.last_runtime_binding = None
        self._network_id = None
        self._pending_events = {}
        self._pending_layouts = {}
        self._binary_continuation_contracts = {}
        self._mpi_continuation_contract = None
        for clock in {obj.clock for obj in network.sorted_objects}:
            clock.set_interval(network.t, network.t)

    def _runner(self):
        configured = self.build_options.get("runner")
        if configured is None:
            configured = os.environ.get("B2_RUNNER")
        if configured is not None:
            runner = Path(configured).expanduser().resolve()
            if not runner.is_file():
                raise FileNotFoundError(f"Rust runner does not exist: {runner}")
            return runner
        if source_root() is None:
            runner = executable_path("b2-runner")
            if not runner.is_file():
                raise FileNotFoundError(f"Atlas installation is missing its native runner: {runner}")
            return runner
        # Source installations rebuild with Cargo; wheels use their bundled binary.
        env = os.environ.copy()
        env.setdefault("CARGO_HOME", str(cache_root() / "cargo"))
        self._pinned_rustc()
        cargo_version = self._invoke(["cargo", "--version"], cwd=ROOT,
                                     env=env).stdout.splitlines()[0]
        require(cargo_version.startswith("cargo 1.98.1 "),
                f"Rust runner build requires cargo 1.98.1; got {cargo_version}")
        command = ["cargo", "build", "--release", "--locked", "--manifest-path", str(ROOT / "Cargo.toml"), "--target-dir", str(cargo_target())]
        # rustup selects this checkout's pinned rust-toolchain.toml from cwd,
        # including when the caller runs a Brian2 script outside the checkout.
        self._invoke(command, env=env, cwd=ROOT)
        return cargo_target() / "release" / ("b2-runner.exe" if os.name == "nt" else "b2-runner")

    @staticmethod
    def _configured_run_directory(base, run_count):
        if run_count:
            return base / f"run-{run_count + 1:04d}"
        if not base.exists():
            return base
        owned = ((base / "model.json").is_file() and
                 ((base / "native/manifest.json").is_file() or
                  (base / "rust/summary.json").is_file()))
        if not owned:
            # Preserve the longstanding fail-closed behavior for user-owned
            # paths and incomplete artifacts.
            return base
        index = 2
        while (base / f"run-{index:04d}").exists():
            index += 1
        return base / f"run-{index:04d}"

    @staticmethod
    def _invoke(command, **kwargs):
        try:
            result = subprocess.run(command, capture_output=True, text=True, **kwargs)
        except FileNotFoundError as error:
            raise RuntimeError(f"Required executable is missing: {command[0]}") from error
        if result.returncode:
            raise RuntimeError(f"Rust backend failed ({command[0]}):\n{result.stderr or result.stdout}")
        return result

    @staticmethod
    def _pinned_rustc(rustc="rustc"):
        verbose = AtlasDevice._invoke(
            [str(rustc), "--version", "--verbose"], cwd=ROOT).stdout
        release = next((line.removeprefix("release: ").strip()
                        for line in verbose.splitlines()
                        if line.startswith("release: ")), None)
        host = next((line.removeprefix("host: ").strip()
                     for line in verbose.splitlines()
                     if line.startswith("host: ")), None)
        native_arch = {"arm64": "aarch64", "amd64": "x86_64"}.get(
            platform.machine().lower(), platform.machine().lower())
        require(release == "1.98.1",
                f"Rust build requires rustc 1.98.1; got {release or 'unknown'}")
        require(host is not None and host.startswith(native_arch + "-"),
                f"Rust build requires native {native_arch} rustc; got {host or 'unknown'}")
        return verbose, host

    def _build_aot(self, model, runner, model_path, directory, *, validate_model=True):
        # The same Rust validator gates both execution paths. Compilation is
        # explicit and failures never fall back to the reference executor.
        started = time.perf_counter()
        if validate_model:
            self._invoke([str(runner), "--validate", str(model_path)])
            self.last_build_timings["validation_seconds"] = time.perf_counter() - started
        else:
            self.last_build_timings["validation_seconds"] = 0.0
        started = time.perf_counter()
        from .plan import _derive_execution_plan
        self.last_execution_plan = _derive_execution_plan(model)
        source, instance, manifest = _write_project_verified(
            model, directory / "native", plan=self.last_execution_plan)
        native_sources = sorted(source.parent.glob("function-*.c"))
        self.last_build_timings["generate_seconds"] = time.perf_counter() - started
        binary = source.parent / ("b2-native.exe" if os.name == "nt" else "b2-native")
        # A generated standalone executable has no recoverable panic boundary.
        # Aborting also lets LLVM remove unwind edges from bounds checks; those
        # edges otherwise inhibit vectorization of mixed-clock threshold loops.
        flags = ["--edition=2021", "-C", "opt-level=3", "-C", "codegen-units=1",
                 "-C", "panic=abort"]
        started = time.perf_counter()
        native_objects = []
        rustc = os.environ.get("B2_RUSTC", "rustc")
        rustc_verbose, rustc_host = self._pinned_rustc(rustc)
        cache_key = None
        cached_binary = None
        if not native_sources:
            digest = hashlib.sha256(source.read_bytes())
            digest.update(rustc.encode())
            digest.update(rustc_verbose.encode())
            digest.update(platform.machine().encode())
            for flag in flags:
                digest.update(b"\0" + flag.encode())
            cache_key = digest.hexdigest()
            cached_binary = self._aot_binary_cache.get(cache_key)
        compile_reused = cached_binary is not None
        if compile_reused:
            payload, mode = cached_binary
            binary.write_bytes(payload)
            binary.chmod(mode)
        elif native_sources:
            compiler = os.environ.get("CC", "cc")
            c_target_flags = []
            if sys.platform == "darwin":
                if rustc_host.startswith("aarch64-apple-darwin"):
                    c_target_flags = ["-arch", "arm64"]
                elif rustc_host.startswith("x86_64-apple-darwin"):
                    c_target_flags = ["-arch", "x86_64"]
                else:
                    raise RuntimeError(
                        f"Unsupported macOS rustc host for C ABI: {rustc_host}")
            for native_source in native_sources:
                native_object = native_source.with_suffix(".o")
                self._invoke([
                    compiler, *c_target_flags, "-O3", "-std=c11", "-fPIC", "-c",
                    str(native_source), "-o", str(native_object)])
                native_objects.append(native_object)
            for native_object in native_objects:
                flags.extend(["-C", f"link-arg={native_object}"])
            if sys.platform.startswith("linux"):
                flags.extend(["-C", "link-arg=-lm"])
        if not compile_reused:
            self._invoke([rustc, *flags, str(source), "-o", str(binary)], cwd=ROOT)
            if cache_key is not None:
                self._aot_binary_cache[cache_key] = (
                    binary.read_bytes(), stat.S_IMODE(binary.stat().st_mode))
        self.last_build_timings["compile_seconds"] = time.perf_counter() - started
        manifest.update(
            rustc=rustc_verbose.splitlines()[0], rustc_host=rustc_host,
            rustc_executable=rustc,
            target=platform.machine(), flags=flags,
            compile_reused=compile_reused,
            validation_reused=not validate_model,
            timings=self.last_build_timings)
        (source.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.native_artifact = {"binary": binary, "instance": instance}
        return binary, instance

    def network_run(self, network, duration, report=None, report_period=10 * second,
                    namespace=None, profile=None, level=0):
        operations = [obj for obj in network.sorted_objects
                      if type(obj) is NetworkOperation and obj.active]
        streaming_steps = self.build_options.get("monitor_streaming_steps")
        if streaming_steps is not None and not getattr(
                self, "_running_monitor_stream_segments", False):
            require(not operations,
                    "monitor streaming and NetworkOperation cannot yet be combined")
            if namespace is None:
                namespace = get_local_namespace(level=level + 2)
            return self._run_with_monitor_streaming(
                network, duration, streaming_steps, report, report_period,
                namespace, profile, level)
        if operations and not getattr(
                self, "_running_network_operation_segments", False):
            if namespace is None:
                namespace = get_local_namespace(level=level + 2)
            return self._run_with_network_operation(
                network, duration, operations, report, report_period,
                namespace, profile, level)
        if not self.build_on_run and self.has_been_run:
            raise RuntimeError("Queued Rust network has already been built; reinit before another run")
        if profile is None:
            profile = self.build_options.get("profile", False)
        require(type(profile) is bool, "profile must be boolean")
        require(report is None or report in ("text", "stdout", "stderr") or callable(report),
                "report must be None, 'text', 'stdout', 'stderr', or a callback")
        report_seconds = float(report_period / second)
        require(np.isfinite(report_seconds) and report_seconds > 0,
                "report_period must be finite and positive")
        require(self.build_on_run or (report is None and not profile),
                "report/profile with queued explicit build is unsupported")
        require(self._maximum_run_time is None, "maximum run time is unsupported")
        self._check_ownership(network)
        self._prepare_network_identity(network)
        prefs.check_all_validated()
        objects = network.sorted_objects
        names = Counter(obj.name for obj in objects)
        if any(count > 1 for count in names.values()):
            raise ValueError("All objects in a network need unique names")
        if any(obj._network not in (None, network.id) for obj in objects):
            raise RuntimeError("An object has already run in another Network")
        network.check_dependencies()
        if namespace is None:
            namespace = get_local_namespace(level=level + 2)
        # This replaces CodeRunner.before_run only for the validated subset;
        # it resolves units/symbols and lowers statements without a templater.
        try:
            with packed_export():
                model = lower_network(
                    network, duration, namespace, self._rng_seed,
                    self.build_options.get("recording_window_steps"),
                    _network_operations_prevalidated=getattr(
                        self, "_running_network_operation_segments", False))
        except CapabilityError as error:
            self.last_capability_report = error.report
            raise
        # lower_network has already performed the complete lowering pass.  Do
        # not lower a second time merely to expose the successful preflight.
        self.last_capability_report = collect_network_issues(network, duration)
        if not self.build_on_run:
            self._queue_model(network, duration, model, objects)
            return
        report_callback = None
        report_started = None
        if report is not None:
            if report in ("text", "stdout"):
                report_callback = TextReport(sys.stdout)
            elif report == "stderr":
                report_callback = TextReport(sys.stderr)
            else:
                report_callback = report
            report_started = time.perf_counter()
            report_callback(0 * second, 0.0, network.t, duration)
        procedural = any(
            instance.get("topology", {}).get("kind") not in ("explicit", "binary_csr")
            for instance in model["instance"]["synapses"])
        require(not (self.has_been_run and procedural),
                "procedural fixed-total topology currently supports one run per activation")
        binary_contracts = {}
        for definition, instance in zip(model["definition"]["synapses"],
                                        model["instance"]["synapses"], strict=True):
            if instance.get("topology", {}).get("kind") != "binary_csr":
                continue
            for pathway in instance["pathways"]:
                require(pathway["kind"] == "pre" and len(pathway["delay_ticks"]) == 1,
                        "binary CSR continuation requires uniform pre pathways")
            binary_contracts[definition["name"]] = json.dumps({
                "layouts": [self._synapse_layout(definition, instance, p)
                            for p in instance["pathways"]],
                "parameters": instance["parameters"],
            }, sort_keys=True)
        if self._binary_continuation_contracts:
            require(binary_contracts == self._binary_continuation_contracts,
                    "binary CSR topology, delay or parameters changed across continuation")
        self._inject_pending_events(model)
        self._execute_model(network, model, objects, advance_clock=True, profile=profile)
        self._binary_continuation_contracts = binary_contracts
        if report_callback is not None:
            report_callback((time.perf_counter() - report_started) * second,
                            1.0, network.t - duration, duration)
        if not (getattr(self, "_running_network_operation_segments", False) or
                getattr(self, "_running_monitor_stream_segments", False)):
            network.after_run()

    @staticmethod
    def _monitor_objects(network):
        return [obj for obj in model_objects(network)
                if type(obj) in (StateMonitor, EventMonitor, SpikeMonitor,
                                 PopulationRateMonitor) and obj.active]

    @staticmethod
    def _clear_monitor_objects(monitors):
        for monitor in monitors:
            monitor.resize(0)
            if "N" in monitor.variables:
                monitor.variables["N"].set_value(0)
            if "count" in monitor.variables:
                monitor.variables["count"].set_value(np.zeros(
                    len(monitor.source), dtype=monitor.variables["count"].dtype))

    @staticmethod
    def _monitor_stream_schema(monitors):
        schema = {}
        for monitor in monitors:
            if type(monitor) is StateMonitor:
                variables = list(monitor.record_variables)
                schema[monitor.name] = {
                    "kind": "state", "variables": variables,
                    "record": np.asarray(monitor.record, dtype=np.int64).tolist(),
                    "layout": "time_record",
                }
            elif type(monitor) in (EventMonitor, SpikeMonitor):
                variables = sorted(set(monitor.record_variables) - {"i", "t"})
                schema[monitor.name] = {
                    "kind": "spike" if type(monitor) is SpikeMonitor else "event",
                    "variables": variables,
                }
            else:
                schema[monitor.name] = {
                    "kind": "rate", "variables": ["rate"]}
        return schema

    @staticmethod
    def _write_monitor_chunk(root, index, monitors, start, end):
        root.mkdir(parents=True, exist_ok=True)
        chunk = root / f"chunk-{index:08d}"
        chunk.mkdir(parents=True, exist_ok=False)
        files = {}
        for monitor in monitors:
            if type(monitor) is StateMonitor:
                names = ["t", *monitor.record_variables]
            elif type(monitor) in (EventMonitor, SpikeMonitor):
                names = ["i", "t", *(sorted(
                    set(monitor.record_variables) - {"i", "t"}))]
            else:
                names = ["t", "rate"]
            arrays = {
                name: np.asarray(monitor.variables[name].get_value()).copy()
                for name in names
            }
            if "N" in monitor.variables:
                arrays["N"] = np.asarray(
                    monitor.variables["N"].get_value()).copy()
            if "count" in monitor.variables:
                arrays["count"] = np.asarray(
                    monitor.variables["count"].get_value()).copy()
            target = chunk / f"{monitor.name}.npz"
            temporary = chunk / f".{monitor.name}.npz.tmp"
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            temporary.replace(target)
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[monitor.name] = {
                "file": target.name,
                "bytes": target.stat().st_size,
                "sha256": digest,
            }
        return {
            "index": index,
            "start_seconds": start,
            "end_seconds": end,
            "directory": chunk.name,
            "monitors": files,
        }

    @staticmethod
    def _publish_monitor_manifest(root, manifest):
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(root / "manifest.json")

    def _run_with_monitor_streaming(self, network, duration, streaming_steps,
                                    report, report_period, namespace, profile,
                                    level):
        """Execute bounded continuations and persist complete Monitor chunks."""
        if profile is None:
            profile = self.build_options.get("profile", False)
        require(profile is False,
                "profile is not yet available with monitor streaming")
        require(report is None or report in ("text", "stdout", "stderr") or
                callable(report),
                "report must be None, 'text', 'stdout', 'stderr', or a callback")
        total = float(duration / second)
        require(np.isfinite(total) and total >= 0,
                "duration must be finite and non-negative")
        report_seconds = float(report_period / second)
        require(np.isfinite(report_seconds) and report_seconds > 0,
                "report_period must be finite and positive")
        require(self._maximum_run_time is None,
                "maximum run time is unsupported")
        monitors = self._monitor_objects(network)
        require(monitors, "monitor streaming requires at least one active Monitor")
        native_clocks = {obj.clock for obj in network.sorted_objects if obj.active}
        quantum = min(float(clock.dt / second) for clock in native_clocks)
        chunk_seconds = streaming_steps * quantum
        base = Path(self.build_options["directory"]).expanduser().resolve()
        stream_root = base / "monitor-stream"
        schema = self._monitor_stream_schema(monitors)
        manifest_path = stream_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            require(manifest.get("schema") == "b2-monitor-stream-v1" and
                    manifest.get("chunk_steps") == streaming_steps and
                    manifest.get("quantum_seconds") == quantum and
                    manifest.get("monitors") == schema,
                    "existing monitor stream has an incompatible schema")
        else:
            manifest = {
                "schema": "b2-monitor-stream-v1",
                "chunk_steps": streaming_steps,
                "quantum_seconds": quantum,
                "monitors": schema,
                "chunks": [],
                "complete": False,
            }
        self.last_monitor_stream = stream_root
        self._monitor_stream_chunks = len(manifest["chunks"])
        if self._monitor_stream_chunks:
            self._clear_monitor_objects(monitors)
        start = float(network.t / second)
        elapsed = 0.0
        wall_start = time.perf_counter()
        report_callback = None
        if report is not None:
            report_callback = (
                TextReport(sys.stdout) if report in ("text", "stdout") else
                TextReport(sys.stderr) if report == "stderr" else report)
            report_callback(0 * second, 0.0, network.t, duration)
        manifest["complete"] = False
        if stream_root.exists():
            self._publish_monitor_manifest(stream_root, manifest)
        self._running_monitor_stream_segments = True
        try:
            if total == 0:
                self.network_run(
                    network, 0 * second, report=None,
                    report_period=report_period, namespace=namespace,
                    profile=False, level=level + 1)
            while elapsed < total:
                segment = min(chunk_seconds, total - elapsed)
                self.network_run(
                    network, segment * second, report=None,
                    report_period=report_period, namespace=namespace,
                    profile=False, level=level + 1)
                self._monitor_stream_chunks += 1
                entry = self._write_monitor_chunk(
                    stream_root, self._monitor_stream_chunks, monitors,
                    start + elapsed, start + elapsed + segment)
                manifest["chunks"].append(entry)
                self._publish_monitor_manifest(stream_root, manifest)
                elapsed += segment
                if elapsed < total:
                    self._clear_monitor_objects(monitors)
        finally:
            self._running_monitor_stream_segments = False
        network.after_run()
        manifest["complete"] = True
        self._publish_monitor_manifest(stream_root, manifest)
        self._last_run_time = time.perf_counter() - wall_start
        self._last_run_completed_fraction = 1.0
        if report_callback is not None:
            report_callback(self._last_run_time * second, 1.0,
                            start * second, duration)

    def _run_with_network_operation(self, network, duration, operations, report,
                                    report_period, namespace, profile, level):
        """Run tick-boundary Python callbacks between native continuations."""
        require(self.build_on_run,
                "NetworkOperation requires build_on_run=True")
        if profile is None:
            profile = self.build_options.get("profile", False)
        require(type(profile) is bool, "profile must be boolean")
        require(report is None or report in ("text", "stdout", "stderr") or
                callable(report),
                "report must be None, 'text', 'stdout', 'stderr', or a callback")
        report_seconds = float(report_period / second)
        require(np.isfinite(report_seconds) and report_seconds > 0,
                "report_period must be finite and positive")
        require(self._maximum_run_time is None, "maximum run time is unsupported")
        require(1 <= len(operations) <= 16,
                "at most 16 active NetworkOperations are supported")
        require(all(operation.when in {"start", "end"} and
                    not operation.contained_objects
                    for operation in operations),
                "NetworkOperations must be childless and scheduled at a start "
                "or end tick boundary")
        for slot in ("start", "end"):
            slot_operations = [operation for operation in operations
                               if operation.when == slot]
            slot_effects = [obj for obj in network.sorted_objects
                            if obj.active and obj.when == slot and
                            (isinstance(obj, CodeRunner) or
                             type(obj) is NetworkOperation)]
            expected = (slot_effects[:len(slot_operations)] if slot == "start"
                        else slot_effects[len(slot_effects)-len(slot_operations):]
                        if slot_operations else [])
            require(expected == slot_operations,
                    "start NetworkOperations must precede native start effects "
                    "and end NetworkOperations must follow native end effects")
        intervals = [float(operation.clock.dt / second)
                     for operation in operations]
        start = float(network.t / second)
        total = float(duration / second)
        require(all(np.isfinite(interval) and interval > 0
                    for interval in intervals) and
                np.isfinite(total) and total >= 0,
                "NetworkOperation dt must be finite and positive and duration "
                "must be finite and non-negative")
        require(all(math.isclose(
                        start / interval, round(start / interval),
                        rel_tol=0, abs_tol=1e-9)
                    for interval in intervals),
                "run start must align to NetworkOperation dt")
        native_clocks = {obj.clock for obj in network.sorted_objects
                         if obj.active and type(obj) is not NetworkOperation}
        require(all(math.isclose(
                        interval / float(clock.dt / second),
                        round(interval / float(clock.dt / second)),
                        rel_tol=0, abs_tol=1e-9)
                    for interval in intervals for clock in native_clocks),
                "NetworkOperation dt must be an integer multiple of every native clock dt")
        quantum = min(float(clock.dt / second) for clock in native_clocks)
        require(math.isclose(total / quantum, round(total / quantum),
                             rel_tol=0, abs_tol=1e-9),
                "run duration must align to the native clock quantum")
        interval_ticks = [round(interval / quantum) for interval in intervals]
        total_ticks = round(total / quantum)
        operation_ticks = dict(zip(operations, interval_ticks, strict=True))
        start_operations = [operation for operation in operations
                            if operation.when == "start"]
        end_operations = [operation for operation in operations
                          if operation.when == "end"]

        # Fail closed before invoking user Python.  The recursive segment run
        # still performs full lowering, but all statically inspectable model,
        # ownership and lifecycle errors must be reported before a callback
        # can have observable side effects.
        self._check_ownership(network)
        self._prepare_network_identity(network)
        prefs.check_all_validated()
        objects = network.sorted_objects
        names = Counter(obj.name for obj in objects)
        if any(count > 1 for count in names.values()):
            raise ValueError("All objects in a network need unique names")
        if any(obj._network not in (None, network.id) for obj in objects):
            raise RuntimeError("An object has already run in another Network")
        network.check_dependencies()
        self.last_capability_report = collect_network_issues(network, duration)
        if not self.last_capability_report.supported:
            raise CapabilityError(self.last_capability_report)

        network._stopped = False
        Network._globally_stopped = False
        started_at = network.t
        elapsed_ticks = 0
        wall_start = time.perf_counter()
        report_callback = None
        if report is not None:
            report_callback = (
                TextReport(sys.stdout) if report in ("text", "stdout") else
                TextReport(sys.stderr) if report == "stderr" else report)
            report_callback(0 * second, 0.0, started_at, duration)
        self._running_network_operation_segments = True
        self._network_operation_aot_validated_definition = None
        self._network_operation_last_model = None
        self._network_operation_last_model_path = None
        previous_segment_directory = None
        try:
            while elapsed_ticks < total_ticks:
                for operation in start_operations:
                    if elapsed_ticks % operation_ticks[operation] == 0:
                        operation.clock._set_t_update_dt(
                            (start + elapsed_ticks * quantum) * second)
                        operation.run()
                if network._stopped or Network._globally_stopped:
                    break
                boundaries = [total_ticks - elapsed_ticks]
                boundaries.extend(
                    ticks - elapsed_ticks % ticks
                    for operation, ticks in operation_ticks.items()
                    if operation.when == "start")
                boundaries.extend(
                    (-elapsed_ticks) % ticks + 1
                    for operation, ticks in operation_ticks.items()
                    if operation.when == "end")
                segment_ticks = min(boundaries)
                segment = segment_ticks * quantum
                self.network_run(
                    network, segment * second, report=None,
                    report_period=report_period, namespace=namespace,
                    profile=profile, level=level + 1)
                current_segment_directory = self.last_run_directory
                if previous_segment_directory is not None:
                    self._discard_superseded_run(
                        previous_segment_directory,
                        current_segment_directory)
                previous_segment_directory = current_segment_directory
                elapsed_ticks += segment_ticks
                callback_tick = elapsed_ticks - 1
                for operation in end_operations:
                    if callback_tick % operation_ticks[operation] == 0:
                        operation.clock._set_t_update_dt(
                            (start + callback_tick * quantum) * second)
                        operation.run()
                        operation.clock._set_t_update_dt(
                            (start + elapsed_ticks * quantum) * second)
                if network._stopped or Network._globally_stopped:
                    break
            model_path = self._network_operation_last_model_path
            if model_path is not None and not model_path.is_file():
                # Intermediate callback continuations use the native binary
                # instance directly. Publish and independently validate the
                # final input snapshot once, so the retained public artifact
                # has the same contract as an ordinary AOT run.
                with model_path.open("wb") as stream:
                    write_canonical(self._network_operation_last_model, stream)
                    stream.write(b"\n")
                self._invoke([str(self._runner()), "--validate", str(model_path)])
        finally:
            self._running_network_operation_segments = False
            self._network_operation_aot_validated_definition = None
            self._network_operation_last_model = None
            self._network_operation_last_model_path = None
        network.after_run()
        self._last_run_time = time.perf_counter() - wall_start
        self._last_run_completed_fraction = (
            1.0 if total_ticks == 0 else elapsed_ticks / total_ticks)
        if report_callback is not None:
            report_callback(self._last_run_time * second,
                            self._last_run_completed_fraction,
                            started_at, duration)

    @staticmethod
    def _discard_superseded_run(previous, current):
        """Discard a completed run while retaining its successful successor."""
        previous, current = Path(previous), Path(current)
        require(previous != current,
                "superseded runs must use distinct run directories")
        try:
            relative_current = current.relative_to(previous)
        except ValueError:
            shutil.rmtree(previous)
            return
        # The first configured run owns the root directory; later segments are
        # nested below it.  Remove the first run's artifacts without deleting
        # the subtree that contains the newly completed segment.
        retained = previous / relative_current.parts[0]
        for child in previous.iterdir():
            if child == retained:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    @staticmethod
    def _definition_signature(model):
        definition = copy.deepcopy(model["definition"])
        for population in definition["populations"]:
            population["steps"] = 0
            population["monitor"]["window_steps"] = 0
        return definition

    @staticmethod
    def _queued_instance_signature(model):
        """Return instance data that must stay fixed across queued segments."""
        instance = copy.deepcopy(model["instance"])
        for population in instance["populations"]:
            population["parameters"] = {}
            population["spike_generator"] = None
        for synapse in instance["synapses"]:
            synapse["parameters"] = {}
        return instance

    @staticmethod
    def _continue_queued_segment(model, previous_results):
        """Seed a queued segment from the preceding native result."""
        for definition, instance, result in zip(
                model["definition"]["populations"],
                model["instance"]["populations"],
                previous_results["populations"], strict=True):
            instance["initial_state"] = {
                symbol["name"]: _array_bits(
                    result["states"][symbol["name"]], symbol["dtype"])
                for symbol in definition["states"]
            }
            if instance["refractory"] is not None:
                refractory = result["refractory"]
                require(refractory is not None,
                        "queued refractory result is missing")
                instance["refractory"]["initial_lastspike"] = _array_bits(
                    refractory["lastspike"], "f64")
                instance["refractory"]["initial_not_refractory"] = \
                    refractory["not_refractory"].tolist()
        for definition, instance, result in zip(
                model["definition"]["synapses"],
                model["instance"]["synapses"],
                previous_results["synapses"], strict=True):
            instance["initial_state"] = {
                symbol["name"]: _array_bits(
                    result["states"][symbol["name"]], symbol["dtype"])
                for symbol in definition["states"]
            }

    def _queue_model(self, network, duration, candidate, objects):
        if self._queued_network is None:
            self._queued_network = network
            self._queued_model = candidate
            self._queued_segments = [copy.deepcopy(candidate)]
            # MagicNetwork can drop its ephemeral object set after run().
            # Retain the original Brian objects for later explicit build and
            # public monitor/state backfill.
            self._queued_objects = tuple(objects)
            self._queued_roots = tuple(model_objects(network))
        else:
            procedural = any(
                instance.get("topology", {}).get("kind") != "explicit"
                for instance in candidate["instance"]["synapses"])
            require(not procedural,
                    "procedural fixed-total topology currently supports one "
                    "run per activation")
            require(self._queued_network is network,
                    "a Rust Device initialization can queue only one Network")
            previous_segment = self._queued_segments[-1]
            compatible = (
                self._definition_signature(candidate) ==
                self._definition_signature(previous_segment) and
                self._queued_instance_signature(candidate) ==
                self._queued_instance_signature(previous_segment))
            require(compatible,
                    "model mutation between queued runs is unsupported")
            for previous, addition in zip(
                    previous_segment["run"]["clocks"],
                    candidate["run"]["clocks"], strict=True):
                require(
                    addition["start_tick"] ==
                    previous["start_tick"] + previous["steps"],
                    "queued Clock intervals must be contiguous")
            current_instances = copy.deepcopy(
                self._queued_model["instance"]["populations"])
            candidate_instances = copy.deepcopy(
                candidate["instance"]["populations"])
            for population in current_instances:
                population["spike_generator"] = None
            for population in candidate_instances:
                population["spike_generator"] = None
            same_parameters = (
                candidate_instances == current_instances and
                candidate["instance"]["synapses"] ==
                self._queued_model["instance"]["synapses"])
            self._queued_segments.append(copy.deepcopy(candidate))
            self._queued_parameter_segments |= not same_parameters
            if not self._queued_parameter_segments:
                for current, addition in zip(
                        self._queued_model["definition"]["populations"],
                        candidate["definition"]["populations"], strict=True):
                    current["steps"] += addition["steps"]
                    configured_window = self.build_options.get(
                        "recording_window_steps")
                    current["monitor"]["window_steps"] = (
                        current["steps"] if configured_window is None else
                        min(current["steps"], configured_window))
                for current, addition in zip(
                        self._queued_model["instance"]["populations"],
                        candidate["instance"]["populations"], strict=True):
                    require((current["spike_generator"] is None) ==
                            (addition["spike_generator"] is None),
                            "population kind cannot change between runs")
                    if current["spike_generator"] is not None:
                        current["spike_generator"]["spike_ticks"].extend(
                            addition["spike_generator"]["spike_ticks"])
                        current["spike_generator"]["spike_indices"].extend(
                            addition["spike_generator"]["spike_indices"])
                for current, addition in zip(
                        self._queued_model["run"]["clocks"],
                        candidate["run"]["clocks"], strict=True):
                    require(
                        addition["start_tick"] ==
                        current["start_tick"] + current["steps"],
                        "queued Clock intervals must be contiguous")
                    current["steps"] += addition["steps"]
                total = struct.unpack(
                    ">d", bytes.fromhex(
                        self._queued_model["run"]["duration"]))[0]
                total += float(duration / second)
                self._queued_model["run"]["duration"] = struct.pack(
                    ">d", total).hex()
        network._clocks = {obj.clock for obj in objects}
        start, end = network.t, network.t + duration
        for clock in network._clocks:
            clock.set_interval(start, end)
            clock._set_t_update_dt(end)
        network.t_ = float(end / second)
        network._profiling_info = None
        for obj in objects:
            obj._network = network.id
        self._network_id = network.id
        network.after_run()

    def _execute_model(self, network, model, objects, advance_clock, profile=False,
                       compile_only=False, replace_monitor_results=False):
        # Device-owned lifecycle operations (pending delayed events and queued
        # run aggregation) deliberately mutate instance/run layers. Re-seal
        # those trusted changes before validation; external artifacts remain
        # tamper-evident at their public load boundary.
        attach_protocol(model)
        from .binary_topology import file_hash
        for synapse in model["instance"]["synapses"]:
            topology = synapse.get("topology", {})
            if topology.get("kind") == "binary_csr":
                require(file_hash(topology["path"]) == topology["sha256"],
                        "binary CSR changed after attachment")
        runner = self._runner()
        engine = self.build_options.get("engine", "reference")
        require(self._mpi_continuation_contract is None or engine == 'mpi',
                'MPI checkpoint configuration mismatch: engine')
        has_synapse_endpoints = any(
            synapse.get("source_synapse") is not None or
            synapse.get("target_synapse") is not None
            for synapse in model["definition"]["synapses"])
        require(not has_synapse_endpoints or engine == "reference",
                "Synapses edge endpoints currently require engine='reference'")
        distributed_plan = None
        mpi_contract = None
        if engine == "mpi":
            from .distributed import build_distributed_plan
            distributed_plan = build_distributed_plan(
                model, ranks=self.build_options.get("ranks", 2), runner=runner,
                rank_backends=self.build_options.get("rank_backends"),
                numeric_mode=self.build_options.get("numeric_mode", "reference-f64"))
            from .mpi_checkpoint import contract, verify
            mpi_contract = contract(model, self.build_options, runner)
            if self._mpi_continuation_contract is not None:
                verify(self._mpi_continuation_contract, mpi_contract)
        has_spatial = any(
            population.get("spatial") is not None
            for population in model["definition"]["populations"])
        require(not has_spatial or engine == "reference",
                "SpatialNeuron currently requires engine='reference'")
        previous_run_directory = self.last_run_directory
        configured = self.build_options.get("directory")
        if configured is None:
            directory = create_run_directory()
        else:
            base = Path(configured).expanduser().resolve()
            directory = self._configured_run_directory(base, self._run_count)
            directory.mkdir(parents=True, exist_ok=False)
        callback_aot = (engine == "aot" and getattr(
            self, "_running_network_operation_segments", False))
        definition_hash = model["protocol"]["layers"]["definition"]
        validation_reused = (
            callback_aot and
            self._network_operation_aot_validated_definition == definition_hash)
        self.last_runtime_binding = None
        model_path = directory / "model.json"
        if not validation_reused:
            with model_path.open("wb") as stream:
                write_canonical(model, stream)
                stream.write(b"\n")
        executable, input_path = runner, model_path
        has_synapse_monitors = any(
            synapse.get("state_monitors")
            for synapse in model["definition"]["synapses"])
        require(not has_synapse_monitors or engine == "reference" or
                (engine == "aot" and distributed_plan is None),
                "Synapses StateMonitor requires reference or non-MPI AOT engine")
        has_synapse_links = any(
            synapse.get("linked_variables")
            for synapse in model["definition"]["synapses"])
        require(not has_synapse_links or engine == "reference" or
                (engine == "aot" and distributed_plan is None),
                "Synapses linked variables require reference or non-MPI AOT engine")
        if engine == "aot":
            executable, input_path = self._build_aot(
                model, runner, model_path, directory,
                validate_model=not validation_reused)
            if callback_aot:
                self._network_operation_aot_validated_definition = definition_hash
        elif compile_only:
            self._invoke([str(runner), "--validate", str(model_path)])
        if compile_only:
            self._explicit_build_directory = directory
            self.last_run_directory = directory
            return
        started = time.perf_counter()
        run_environment = os.environ.copy()
        run_environment["B2_NUM_THREADS"] = str(self.build_options.get("threads", 1))
        run_environment["B2_THREAD_AFFINITY"] = str(
            self.build_options.get("thread_affinity", "auto"))
        tuning_cache_entry = None
        if distributed_plan is not None:
            from .distributed import write_mpi_project, compile_mpi_project, run_mpi_project
            project = directory / "mpi"
            self.last_execution_plan = write_mpi_project(
                model, project, ranks=distributed_plan.ranks, runner=runner, plan=distributed_plan)
            compile_mpi_project(project)
            run_mpi_project(project, directory / "rust")
        elif self.build_options.get("engine") in {"metal","cuda"}:
            from .gpu_buffer_transfer import execute_device
            tuning_cache_entry = execute_device(self,model,directory,runner)
        else:
            self._invoke([str(executable), str(input_path), str(directory / "rust")],
                         env=run_environment)
        results = load_results(model, directory / "rust")
        if self.last_execution_plan is not None:
            from .plan import bind_execution_plan
            self.last_runtime_binding = bind_execution_plan(
                self.last_execution_plan, results["metadata"],
                requested_threads=self.build_options.get("threads", 1),
                requested_affinity=self.build_options.get("thread_affinity", "auto"))
            (directory / "runtime-binding.json").write_text(self.last_runtime_binding.to_json())
        next_pending, next_layouts = self._next_pending_events(model, results)
        if replace_monitor_results:
            self._clear_queued_monitors()
        self._load_results(network, model, directory / "rust", results)
        if advance_clock:
            network._clocks = {obj.clock for obj in objects}
            start = network.t
            duration_seconds = struct.unpack(
                ">d", bytes.fromhex(model["run"]["duration"]))[0]
            end = start + duration_seconds * second
            for clock in network._clocks:
                clock.set_interval(start, end)
                clock._set_t_update_dt(end)
            network.t_ = float(end / second)
        if profile:
            timings = results["metadata"]["timings"]
            network._profiling_info = [
                ("rust_simulation_and_recording",
                 timings["simulation_and_recording_seconds"] * second),
                ("rust_result_dump", timings["dump_write_seconds"] * second),
            ]
        else:
            network._profiling_info = None
        for obj in objects:
            obj._network = network.id
        self.has_been_run = True
        self._network_id = network.id
        self._pending_events = next_pending
        self._pending_layouts = next_layouts
        self._mpi_continuation_contract = mpi_contract
        self._run_count += 1
        self.last_run_directory = directory
        if (not self.build_options.get("retain_run_artifacts", True) and
                previous_run_directory is not None and
                Path(previous_run_directory).exists() and
                Path(previous_run_directory) != directory and
                not getattr(self, "_running_network_operation_segments", False)):
            # Preserve the last successful run until its successor has run and
            # all results have been loaded.  This opt-in bounded mode is useful
            # for workloads that repeatedly restore and replay a network.
            self._discard_superseded_run(previous_run_directory, directory)
        if callback_aot:
            self._network_operation_last_model = model
            self._network_operation_last_model_path = model_path
        self._last_run_time = time.perf_counter() - started
        self._last_run_completed_fraction = 1.0
        if tuning_cache_entry is not None:
            self._gpu_tuning_cache.publish(*tuning_cache_entry)
        return results

    def _inject_pending_events(self, model):
        start = struct.unpack(">d", bytes.fromhex(model["run"]["start"]))[0]
        populations = model["definition"]["populations"]
        for definition, instance in zip(
                model["definition"]["synapses"], model["instance"]["synapses"],
                strict=True):
            for pathway in instance["pathways"]:
                population_index = (definition["source_population"]
                                    if pathway["kind"] == "pre" else
                                    definition["target_population"])
                endpoint = populations[population_index]
                dt = struct.unpack(">d", bytes.fromhex(endpoint["dt"]))[0]
                start_tick = int(round(start / dt))
                end_tick = start_tick + endpoint["steps"]
                key = self._pathway_key(definition, pathway)
                pending = self._pending_events.get(key, [])
                if pending:
                    require(self._pending_layouts.get(key) ==
                            self._synapse_layout(definition, instance, pathway),
                            "synaptic topology/delay cannot change while events are pending")
                require(all(event["delivery_tick"] >= start_tick
                            for event in pending),
                        "pending synaptic event precedes the next run")
                pathway["pending"] = [event.copy() for event in pending
                                      if event["delivery_tick"] < end_tick]

    @staticmethod
    def _pathway_key(definition, pathway):
        return definition["name"], pathway["name"]

    @staticmethod
    def _synapse_layout(definition, instance, pathway):
        digest = hashlib.sha256()
        topology = instance.get("topology", {"kind": "explicit"})
        digest.update(json.dumps(topology, sort_keys=True).encode())
        if topology["kind"] == "explicit":
            digest.update(np.asarray(instance["source"], dtype="<u8").tobytes())
            digest.update(np.asarray(instance["target"], dtype="<u8").tobytes())
        endpoint = ("source" if pathway["kind"] == "pre" else "target")
        return (pathway["kind"], pathway["event"],
                definition[f"{endpoint}_population"],
                definition[f"{endpoint}_start"],
                definition[f"{endpoint}_count"],
                tuple(pathway["delay_ticks"]), digest.hexdigest())

    def _next_pending_events(self, model, results):
        from .schedule import pathway_sample_lags
        sample_lags = pathway_sample_lags(model['definition'])
        populations = model["definition"]["populations"]
        next_pending, next_layouts = {}, {}
        for synapse_index, (definition, instance) in enumerate(zip(
                model["definition"]["synapses"], model["instance"]["synapses"],
                strict=True)):
            if instance.get("topology", {"kind": "explicit"})["kind"] not in ("explicit", "binary_csr"):
                # The first procedural revision deliberately has no continuation
                # contract. Events after this run boundary are discarded and a
                # second run is rejected above.
                continue
            for pathway in instance["pathways"]:
                endpoint = ("source" if pathway["kind"] == "pre" else "target")
                population_index = definition[f"{endpoint}_population"]
                endpoint_def = populations[population_index]
                start = struct.unpack(
                    ">d", bytes.fromhex(model["run"]["start"]))[0]
                dt = struct.unpack(">d", bytes.fromhex(endpoint_def["dt"]))[0]
                end_tick = int(round(start / dt)) + endpoint_def["steps"]
                key = self._pathway_key(definition, pathway)
                retained = [event.copy() for event in
                            self._pending_events.get(key, [])
                            if event["delivery_tick"] >= end_tick]
                delays = pathway["delay_ticks"]
                uniform = (bool(delays) and
                           all(delay == delays[0] for delay in delays))
                new_events = []
                population_results = results["populations"][population_index]
                event_stream = population_results["event_streams"][pathway["event"]]
                # Pre-threshold pathways sample each emission one endpoint tick
                # later. The final emission is deferred to the next run; encode
                # its arrivals as pending because frozen B2IR starts flags empty.
                spike_ticks = event_stream["ticks"] + sample_lags.get((synapse_index, pathway['name']), 0)
                spike_indices = event_stream["indices"]
                endpoint_start = definition[f"{endpoint}_start"]
                endpoint_count = definition[f"{endpoint}_count"]
                if uniform:
                    local = spike_indices - endpoint_start
                    selected = ((local >= 0) & (local < endpoint_count) &
                                (spike_ticks + delays[0] >= end_tick))
                    new_events = [
                        {"delivery_tick": int(tick) + delays[0],
                         "item": int(item)}
                        for tick, item in zip(
                            spike_ticks[selected], local[selected], strict=True)]
                else:
                    require(instance.get("topology", {}).get("kind") == "explicit",
                            "nonuniform continuation requires explicit topology")
                    endpoint_indices = np.asarray(instance[endpoint], dtype=np.int64)
                    maximum_delay = max(delays, default=0)
                    tail = spike_ticks + maximum_delay >= end_tick
                    spike_ticks = spike_ticks[tail]
                    spike_indices = spike_indices[tail]
                    for tick, neuron in zip(
                            spike_ticks, spike_indices, strict=True):
                        local = int(neuron) - endpoint_start
                        if not 0 <= local < endpoint_count:
                            continue
                        for edge in np.flatnonzero(endpoint_indices == local):
                            delivery = int(tick) + delays[int(edge)]
                            if delivery >= end_tick:
                                new_events.append({"delivery_tick": delivery,
                                                   "item": int(edge)})
                                require(len(retained) + len(new_events) <= 10_000_000,
                                        "at most 10,000,000 pending events per pathway")
                pending = retained + new_events
                require(len(pending) <= 10_000_000,
                        "at most 10,000,000 pending events per pathway")
                pending.sort(key=lambda event: event["delivery_tick"])
                next_pending[key] = pending
                if pending:
                    next_layouts[key] = self._synapse_layout(
                        definition, instance, pathway)
        return next_pending, next_layouts

    def _load_results(self, network, model, directory, results=None):
        if results is None:
            results = load_results(model, directory)
        populations = model["definition"]["populations"]
        population_by_group = {item["name"]: (item, results["populations"][index])
                               for index, item in enumerate(populations)}
        population_by_state_monitor = {
            monitor["name"]: (item, monitor, results["populations"][index], index)
            for index, item in enumerate(populations)
            for monitor in item["state_monitors"]}
        synapse_by_state_monitor = {
            monitor["name"]: (item, model["instance"]["synapses"][index],
                              monitor, results["synapses"][index])
            for index, item in enumerate(model["definition"]["synapses"])
            for monitor in item.get("state_monitors", [])}
        population_by_name = {
            item["name"]: (item, results["populations"][index])
            for index, item in enumerate(populations)}
        population_by_rate_monitor = {
            monitor["name"]: (item, results["populations"][index])
            for index, item in enumerate(populations)
            for monitor in item.get("rate_monitors", [])
        }
        population_by_event_monitor = {
            monitor["name"]: (item, monitor,
                              results["populations"][index]["event_monitors"][monitor["name"]])
            for index, item in enumerate(populations)
            for monitor in item.get("event_monitors", [])
        }
        synapse_results = {item["name"]: item for item in results["synapses"]}
        roots = (self._queued_roots if network is self._queued_network
                 and self._queued_roots is not None else model_objects(network))
        population_by_spike_monitor = {}
        for obj in roots:
            if type(obj) is not SpikeMonitor:
                continue
            source = (obj.source.source if isinstance(obj.source, Subgroup) else
                      obj.source)
            if source.name in population_by_name:
                population_by_spike_monitor[obj.name] = population_by_name[source.name]
        for obj in roots:
            if type(obj) in (NeuronGroup, SpatialNeuron):
                population, result = population_by_group[obj.name]
                for name, values in result["states"].items():
                    variable = obj.variables[name]
                    variable.set_value(values[0] if variable.scalar else values)
                if result["refractory"] is not None:
                    for name, values in result["refractory"].items():
                        obj.variables[name].set_value(values)
                if "_spikespace" in obj.variables:
                    space = obj.variables["_spikespace"].get_value()
                    space[:] = 0
                    last = result["last_spikes"]
                    space[:len(last)] = last
                    space[-1] = len(last)
            elif type(obj) in (PoissonGroup, SpikeGeneratorGroup):
                population, result = population_by_group[obj.name]
                for name, values in result["states"].items():
                    obj.variables[name].set_value(values)
                if "_spikespace" in obj.variables:
                    space = obj.variables["_spikespace"].get_value()
                    space[:] = 0
                    last = result["last_spikes"]
                    space[:len(last)] = last
                    space[-1] = len(last)
            elif type(obj) is PoissonInput:
                # Stateless: its effects have already been written into the
                # target NeuronGroup state arrays above.
                continue
            elif type(obj) is StateMonitor:
                previous = len(obj.t)
                old_times = np.asarray(obj.variables["t"].get_value()).copy()
                if obj.name in synapse_by_state_monitor:
                    synapse, synapse_instance, monitor, result = \
                        synapse_by_state_monitor[obj.name]
                    monitor_result = result["state_monitors"][obj.name]
                    old_values = {
                        name: np.asarray(obj.variables[name].get_value()).copy()
                        for name in monitor["output_variables"]}
                    times = np.concatenate((old_times, monitor_result["times"]))
                    window = self.build_options.get("recording_window_steps")
                    if window is not None:
                        times = times[-window:]
                    obj.resize(len(times))
                    obj.variables["t"].set_value(times)
                    for name in monitor["output_variables"]:
                        values = reconstruct_synapse(
                            synapse, synapse_instance, monitor,
                            monitor_result, name, float(obj.clock.dt / second))
                        merged = np.concatenate((old_values[name], values), axis=0)
                        obj.variables[name].set_value(
                            merged if window is None else merged[-window:])
                    continue
                population, monitor, result, population_index = \
                    population_by_state_monitor[obj.name]
                outputs = monitor.get("output_variables") or monitor["variables"]
                old_values = {
                    name: np.asarray(obj.variables[name].get_value()).copy()
                    for name in outputs}
                window = self.build_options.get("recording_window_steps")
                times = np.concatenate((old_times, result["times"]))
                if window is not None:
                    times = times[-window:]
                obj.resize(len(times))
                obj.variables["t"].set_value(times)
                positions = {neuron: index for index, neuron in enumerate(
                    population["monitor"]["record"])}
                columns = [positions[neuron] for neuron in monitor["record"]]
                start = struct.unpack(">d", bytes.fromhex(model["run"]["start"]))[0]
                for name in outputs:
                    values = reconstruct_monitor_observable(
                        population, model["instance"]["populations"][population_index],
                        monitor, result, name, columns, obj, start)
                    obj.variables[name].set_value(
                        (np.concatenate((old_values[name], values), axis=0)
                         if window is None else
                         np.concatenate((old_values[name], values), axis=0)[-window:]))
            elif type(obj) is EventMonitor:
                population, monitor, result = population_by_event_monitor[obj.name]
                old_indices = np.asarray(obj.variables["i"].get_value()).copy()
                old_times = np.asarray(obj.variables["t"].get_value()).copy()
                indices = np.concatenate((old_indices, result["indices"]))
                times = np.concatenate((old_times, result["times"]))
                values = {
                    name: np.concatenate((
                        np.asarray(obj.variables[name].get_value()).copy(),
                        result["values"][name]))
                    for name in monitor["variables"]}
                window = self.build_options.get("recording_window_steps")
                if window is not None:
                    start = struct.unpack(">d", bytes.fromhex(model["run"]["start"]))[0]
                    duration = struct.unpack(">d", bytes.fromhex(
                        model["run"]["duration"]))[0]
                    cutoff = start + duration - window * float(
                        obj.source.clock.dt / second)
                    retained = times >= cutoff - np.finfo(float).eps * max(
                        1.0, abs(cutoff))
                    indices, times = indices[retained], times[retained]
                    values = {name: array[retained]
                              for name, array in values.items()}
                obj.resize(len(indices))
                obj.variables["N"].set_value(len(indices))
                obj.variables["count"].set_value(
                    np.bincount(indices.astype(np.intp), minlength=len(obj.source)))
                obj.variables["i"].set_value(indices)
                obj.variables["t"].set_value(times)
                for name, array in values.items():
                    obj.variables[name].set_value(array)
            elif type(obj) is Synapses:
                for name, values in synapse_results[obj.name]["states"].items():
                    obj.variables[name].set_value(values)
            elif type(obj) is PopulationRateMonitor:
                population, result = population_by_rate_monitor[obj.name]
                dt = float(obj.source.clock.dt / second)
                start = struct.unpack(">d", bytes.fromhex(
                    model["run"]["start"]))[0]
                steps = population["steps"]
                times = np.asarray(result["spike_times"])
                indices = np.asarray(result["indices"])
                if isinstance(obj.source, Subgroup):
                    selected = ((indices >= obj.source.start) &
                                (indices < obj.source.stop))
                    times = times[selected]
                ticks = np.rint((times - start) / dt).astype(np.intp)
                require(np.all((ticks >= 0) & (ticks < steps)),
                        "PopulationRateMonitor spike outside run window")
                rates = 1.0 * np.bincount(ticks, minlength=steps) / dt / len(obj.source)
                old_times = np.asarray(obj.variables["t"].get_value()).copy()
                old_rates = np.asarray(obj.variables["rate"].get_value()).copy()
                new_times = start + np.arange(steps) * dt
                window = self.build_options.get("recording_window_steps")
                merged_times = np.concatenate((old_times, new_times))
                merged_rates = np.concatenate((old_rates, rates))
                if window is not None:
                    merged_times = merged_times[-window:]
                    merged_rates = merged_rates[-window:]
                obj.resize(len(merged_times))
                obj.variables["N"].set_value(len(merged_times))
                obj.variables["t"].set_value(merged_times)
                obj.variables["rate"].set_value(merged_rates.astype(
                    obj.variables["rate"].dtype, copy=False))
            elif type(obj) is SpikeMonitor:
                population, result = population_by_spike_monitor[obj.name]
                indices = result["indices"]
                times = result["spike_times"]
                extra = result["event_monitors"].get(obj.name)
                extra_names = sorted(set(obj.record_variables) - {"i", "t"})
                require((extra is not None) == bool(extra_names),
                        "SpikeMonitor auxiliary result mismatch")
                if extra is not None:
                    require(np.array_equal(indices, extra["indices"]) and
                            np.array_equal(times, extra["times"]),
                            "SpikeMonitor auxiliary event order mismatch")
                if isinstance(obj.source, Subgroup):
                    selected = ((indices >= obj.source.start) &
                                (indices < obj.source.stop))
                    indices = indices[selected] - obj.source.start
                    times = times[selected]
                    if extra is not None:
                        extra = dict(extra)
                        extra["values"] = {
                            name: values[selected]
                            for name, values in extra["values"].items()}
                if not obj.record:
                    # Brian's ``record=False`` monitor deliberately has no i/t
                    # arrays, but its per-neuron and total counts remain
                    # cumulative across runs.
                    old_count = np.asarray(
                        obj.variables["count"].get_value()).copy()
                    run_count = np.bincount(
                        indices.astype(np.intp), minlength=len(obj.source))
                    obj.variables["count"].set_value(old_count + run_count)
                    obj.variables["N"].set_value(
                        int(np.asarray(
                            obj.variables["N"].get_value()).item()) + len(indices))
                    continue
                old_values = {
                    name: np.asarray(obj.variables[name].get_value()).copy()
                    for name in extra_names}
                values = {
                    name: np.concatenate((old_values[name], extra["values"][name]))
                    for name in extra_names}
                old_indices = np.asarray(obj.variables["i"].get_value()).copy()
                old_times = np.asarray(obj.variables["t"].get_value()).copy()
                indices = np.concatenate((old_indices, indices))
                times = np.concatenate((old_times, times))
                window = self.build_options.get("recording_window_steps")
                if window is not None:
                    start = struct.unpack(">d", bytes.fromhex(model["run"]["start"]))[0]
                    duration = struct.unpack(">d", bytes.fromhex(
                        model["run"]["duration"]))[0]
                    cutoff = start + duration - window * float(
                        obj.source.clock.dt / second)
                    retained = times >= cutoff - np.finfo(float).eps * max(
                        1.0, abs(cutoff))
                    indices, times = indices[retained], times[retained]
                    values = {name: array[retained]
                              for name, array in values.items()}
                obj.resize(len(indices))
                obj.variables["N"].set_value(len(indices))
                obj.variables["count"].set_value(
                    np.bincount(indices.astype(np.intp), minlength=len(obj.source)))
                obj.variables["i"].set_value(indices)
                obj.variables["t"].set_value(times)
                for name, array in values.items():
                    obj.variables[name].set_value(array)


# Compatibility with existing imports and serialized class references.
RustStandaloneDevice = AtlasDevice
