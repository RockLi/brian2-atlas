"""Import to register the experimental ``rust_standalone`` Brian2 Device."""

from brian2.devices.device import all_devices, get_device

from .artifact import (ArtifactCompatibilityError, compatible_source_sha256,
                       run_compatible_instance, write_compatible_instance)
from .capabilities import CapabilityError, CapabilityIssue, CapabilityReport
from .device import RustStandaloneDevice
from .export import capability_report, export_network
from .spec import CodeObjectSpec
from .topology import ClippedNormal, Uniform
from .streaming import MonitorStream, iter_monitor_chunks, open_monitor_stream
from .plan import (ExecutionPlan, PlanValidationError, build_execution_plan,
                   explain_plan, verify_execution_plan, RuntimeBinding, bind_execution_plan)
from .distributed import (DistributedPlan, build_distributed_plan,
                          write_mpi_project, compile_mpi_project, run_mpi_project)

if "rust_standalone" not in all_devices:
    all_devices["rust_standalone"] = RustStandaloneDevice()


def connect_fixed_total(synapses, edge_count, *, seed=None, initializers=None,
                        delay_initializer=None):
    """Defer a global fixed-total topology to the Rust standalone backend."""
    device = get_device()
    if not isinstance(device, RustStandaloneDevice):
        raise RuntimeError(
            "connect_fixed_total requires set_device('rust_standalone')")
    device.connect_fixed_total(
        synapses, edge_count, seed=seed, initializers=initializers,
        delay_initializer=delay_initializer)

def connect_binary_csr(synapses, path, *, parameters):
    """Import an immutable CSR file; map Brian constant parameters to columns."""
    device = get_device()
    if not isinstance(device, RustStandaloneDevice):
        raise RuntimeError("connect_binary_csr requires rust_standalone")
    device.connect_binary_csr(synapses, path, parameters)


def connect_fixed_indegree(synapses, indegree, *, seed=None, initializers=None,
                           delay_initializer=None):
    """Defer a no-multapse exact-indegree topology to the Rust backend."""
    device = get_device()
    if not isinstance(device, RustStandaloneDevice):
        raise RuntimeError(
            "connect_fixed_indegree requires set_device('rust_standalone')")
    device.connect_fixed_indegree(
        synapses, indegree, seed=seed, initializers=initializers,
        delay_initializer=delay_initializer)

__all__ = [
    "RustStandaloneDevice", "CodeObjectSpec", "CapabilityError",
    "CapabilityIssue", "CapabilityReport", "capability_report", "export_network",
    "connect_fixed_total", "connect_binary_csr", "connect_fixed_indegree", "ClippedNormal", "Uniform",
    "MonitorStream", "open_monitor_stream", "iter_monitor_chunks",
    "ArtifactCompatibilityError", "compatible_source_sha256",
    "write_compatible_instance", "run_compatible_instance",
    "ExecutionPlan", "PlanValidationError", "build_execution_plan",
    "explain_plan", "verify_execution_plan", "RuntimeBinding", "bind_execution_plan",
    "DistributedPlan", "build_distributed_plan", "write_mpi_project",
    "compile_mpi_project", "run_mpi_project",
]

from .wasm import WasmPlan, export_wasm_bundle
__all__ += ["WasmPlan", "export_wasm_bundle"]

from .training import NativeLIFTrainer, lif_training_plan
__all__ += ["NativeLIFTrainer", "lif_training_plan"]
from .training_graph import dense_training_projection, conv2d_training_projection
__all__ += ["dense_training_projection", "conv2d_training_projection"]

from .training_equations import compile_training_equation, neuron_parameter_bank, PureFunction
__all__ += ["compile_training_equation", "neuron_parameter_bank", "PureFunction"]

from .training_brian import BrianTrainingBundle, TrainingConversionError, lower_brian_training
__all__ += ["BrianTrainingBundle", "TrainingConversionError", "lower_brian_training"]

from .training_brian_dynamic import lower_brian_dynamic_training
__all__ += ["lower_brian_dynamic_training"]
from .training_inputs import BatchTimedArray, external_state_input_vjp
__all__ += ["BatchTimedArray", "external_state_input_vjp"]
