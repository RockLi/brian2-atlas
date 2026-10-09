"""Lower the checked Brian2 population/synapse subset into stable AtlasIR v1."""

import json
import math
import struct
from pathlib import Path

import numpy as np
from brian2 import (EventMonitor, Network, NeuronGroup, PoissonGroup, PoissonInput,
                    PopulationRateMonitor, SpikeGeneratorGroup, SpikeMonitor, StateMonitor, Synapses,
                    second)
from brian2.codegen.translation import analyse_identifiers
from brian2.core.functions import DEFAULT_CONSTANTS, DEFAULT_FUNCTIONS, Function
from brian2.core.magic import MagicNetwork
from brian2.core.preferences import prefs
from brian2.core.variables import AuxiliaryVariable, Constant, Subexpression
from brian2.devices.device import get_device
from brian2.equations.unitcheck import check_units_statements
from brian2.groups.subgroup import Subgroup
from brian2.groups.group import CodeRunner
from brian2.core.operations import NetworkOperation
from brian2.input.binomial import BinomialFunction
from brian2.input.timedarray import TimedArray
from brian2.parsing.statements import parse_statement
from brian2.spatialneuron import SpatialNeuron
from brian2.units.fundamentalunits import Quantity, fail_for_dimension_mismatch
from brian2.utils.stringtools import get_identifiers, word_substitute

from .capabilities import (CapabilityError, CapabilityIssue, CapabilityReport,
                           collect_network_issues, supported_method_choice)
from .spec import (CodeObjectSpec, SUPPORTED_FUNCTIONS, bits, dtype_name, loads,
                   portable_function_contract, require, timed_array_config)
from .schedule import build_schedule, pathway_sample_lags
from .protocol import CURRENT_SCHEMA, attach_protocol
from .encoded_array import EncodedArray, index_array, should_pack
from .topology import ClippedNormal, Uniform
from .resource_limits import (explicit_synapse_budget, initial_value_budget,
                              neuron_budget, population_step_budget)

SCHEMA = CURRENT_SCHEMA
MAX_SYNAPSE_TICKS = 50_000_000_000_000
MAX_NEURON_TICKS = 250_000_000_000

GSL_METHODS = {
    "gsl": "rkf45",
    "gsl_rk2": "rk2",
    "gsl_rk4": "rk4",
    "gsl_rkf45": "rkf45",
    "gsl_rkck": "rkck",
    "gsl_rk8pd": "rk8pd",
}


def _gsl_configuration(group):
    """Validate Brian's default adaptive GSL profile for portable lowering."""
    method = group.state_updater.method_choice
    if method not in GSL_METHODS:
        return None
    from brian2.stateupdaters.GSL import default_method_options
    from brian2.stateupdaters.base import extract_method_options

    require(not group.equations.is_stochastic,
            "GSL state updaters do not support stochastic equations")
    options = extract_method_options(
        group.state_updater.method_options, default_method_options)
    require(type(options["adaptable_timestep"]) is bool and
            type(options["use_last_timestep"]) is bool and
            type(options["save_failed_steps"]) is bool and
            type(options["save_step_count"]) is bool,
            "GSL boolean method options must be bool")
    require(type(options["max_steps"]) is int and
            1 <= options["max_steps"] <= 100_000,
            "GSL max_steps must be an integer in 1..100000")
    require(isinstance(options["absolute_error"], float) and
            math.isfinite(options["absolute_error"]) and
            options["absolute_error"] > 0,
            "GSL absolute_error must be a finite positive float")
    per_variable = options["absolute_error_per_variable"]
    require(per_variable is None or isinstance(per_variable, dict),
            "GSL absolute_error_per_variable must be None or a dict")
    differential = {
        equation.varname: equation
        for equation in group.equations.values()
        if equation.type == "differential equation"
    }
    require(all(np.dtype(group.variables[name].dtype) == np.dtype(np.float64)
                for name in differential),
            "GSL integration currently requires float64 ODE states")
    require(not any(equation.flags for equation in differential.values()) or
            group._refractory is not False,
            "GSL ODE flags require refractory semantics")
    errors = {}
    for name, value in ({} if per_variable is None else per_variable).items():
        require(name in differential,
                f"GSL absolute_error_per_variable names non-ODE variable {name}")
        fail_for_dimension_mismatch(
            value, group.variables[name],
            f"GSL absolute error for {name} has incompatible dimensions")
        numeric = float(value)
        require(math.isfinite(numeric) and numeric > 0,
                f"GSL absolute error for {name} must be finite and positive")
        errors[name] = numeric
    for name in differential:
        errors.setdefault(name, float(options["absolute_error"]))
    return {
        "integrator": GSL_METHODS[method],
        "adaptable_timestep": options["adaptable_timestep"],
        "max_steps": options["max_steps"],
        "use_last_timestep": options["use_last_timestep"],
        "save_failed_steps": options["save_failed_steps"],
        "save_step_count": options["save_step_count"],
        "absolute_errors": errors,
    }


def _ensure_gsl_meta_variables(group, config):
    """Expose the same per-neuron diagnostic arrays as Brian's GSL generator."""
    arrays = []
    if config["use_last_timestep"]:
        arrays.append(("_last_timestep", np.float64,
                       np.full(len(group), group.clock.dt_, dtype=np.float64)))
    if config["save_failed_steps"]:
        arrays.append(("_failed_steps", np.int32,
                       np.zeros(len(group), dtype=np.int32)))
    if config["save_step_count"]:
        arrays.append(("_step_count", np.int32,
                       np.zeros(len(group), dtype=np.int32)))
    for name, dtype, values in arrays:
        if name not in group.variables:
            group.variables.add_array(name, size=len(group), dtype=dtype,
                                      values=values)
    return [name for name, _dtype, _values in arrays]


def _procedural_initializer(initializer, dimensions, stream, label):
    require(isinstance(initializer, (ClippedNormal, Uniform)),
            f"{label}: expected a procedural initializer")
    names = (("mean", "std", "minimum", "maximum")
             if isinstance(initializer, ClippedNormal)
             else ("minimum", "maximum"))
    for name in names:
        value = getattr(initializer, name)
        if value is not None:
            fail_for_dimension_mismatch(
                value, dimensions,
                f"{label}.{name} has incompatible dimensions")
    if isinstance(initializer, Uniform):
        minimum, maximum = float(initializer.minimum), float(initializer.maximum)
        require(math.isfinite(minimum) and math.isfinite(maximum) and
                minimum <= maximum,
                f"{label}: invalid uniform parameters")
        return {
            "kind": "uniform", "minimum": bits(minimum),
            "maximum": bits(maximum), "stream": stream,
        }
    mean, std = float(initializer.mean), float(initializer.std)
    minimum = (None if initializer.minimum is None
               else float(initializer.minimum))
    maximum = (None if initializer.maximum is None
               else float(initializer.maximum))
    require(math.isfinite(mean) and math.isfinite(std) and std >= 0 and
            (minimum is None or math.isfinite(minimum)) and
            (maximum is None or math.isfinite(maximum)) and
            (minimum is None or maximum is None or minimum <= maximum),
            f"{label}: invalid clipped-normal parameters")
    return {
        "kind": "clipped_normal", "mean": bits(mean), "std": bits(std),
        "minimum": None if minimum is None else bits(minimum),
        "maximum": None if maximum is None else bits(maximum),
        "stream": stream,
    }


def _supported_function(name, var, function_contracts):
    if (isinstance(var, TimedArray) or
            (name in SUPPORTED_FUNCTIONS and isinstance(var, Function)
             and var is DEFAULT_FUNCTIONS[name])):
        return True
    if not isinstance(var, Function) or isinstance(var, BinomialFunction):
        return False
    contract = portable_function_contract(name, var)
    previous = function_contracts.setdefault(name, contract)
    require(previous == contract,
            f"{name}: conflicting Function contracts across code objects")
    return True


def _random_function_configs(variables):
    configs = {}
    for name, variable in variables.items():
        if not isinstance(variable, BinomialFunction):
            continue
        require(1 <= variable.n <= 2**31 - 1 and
                math.isfinite(variable.p) and 0 <= variable.p <= 1,
                f"{name}: invalid BinomialFunction parameters")
        configs[name] = {"n": variable.n, "p": variable.p,
                         "approximate": variable.approximate}
    return configs


class RandomStreams:
    """Assign deterministic, model-wide identifiers to runtime draw sites."""

    def __init__(self):
        self.next_stream = 0

    def allocate(self):
        stream = self.next_stream
        self.next_stream += 1
        return stream


def symbol(name, var, index_domain=None):
    return {
        "name": name,
        "dtype": dtype_name(var.dtype),
        "dimensions": [float(dim) for dim in var.dim._dims],
        "index_domain": index_domain or ("scalar" if var.scalar else "neuron"),
    }


def _demote_mutable_shared_scalar_reads(spec, mutable_shared):
    """Evaluate scalar hoists that read synchronized shared lanes per neuron."""
    vector_backed = set(mutable_shared)
    scalar, vector_prefix = [], []
    for statement in spec["scalar"]:
        target = statement["target"]
        dependencies = loads(statement["value"]) & vector_backed
        if dependencies and not (
                target in mutable_shared and target in dependencies):
            vector_prefix.append(statement)
            vector_backed.add(target)
        else:
            scalar.append(statement)
            # A scalar assignment in this code object shadows the synchronized
            # input for all following scalar statements. This preserves
            # sequential run_regularly assignments such as ``x=...; y=x``.
            vector_backed.discard(target)
    spec["scalar"] = scalar
    spec["vector"] = vector_prefix + spec["vector"]


def _value_bits(value, dtype):
    """Encode a finite public value in its declared IEEE storage width."""
    if dtype == "f32":
        value = np.float32(value)
        if not np.isfinite(value):
            raise ValueError("AtlasIR values must be finite")
        return struct.pack(">f", value).hex()
    if dtype == "bool":
        return "01" if bool(value) else "00"
    if dtype in {"i32", "i64", "u32", "u64"}:
        numpy_dtype = np.dtype({
            "i32": np.int32,
            "i64": np.int64,
            "u32": np.uint32,
            "u64": np.uint64,
        }[dtype])
        integer = int(np.asarray(value, dtype=numpy_dtype).item())
        return integer.to_bytes(
            numpy_dtype.itemsize, "big", signed=numpy_dtype.kind == "i").hex()
    require(dtype == "f64", f"unsupported encoded value dtype {dtype}")
    return bits(value)


def _array_bits(values, dtype):
    if should_pack(values):
        return EncodedArray.from_values(values, dtype)
    # Plasticity traces and initial weights are commonly uniform across millions
    # of edges. Reuse immutable encoded strings without changing wire bytes.
    # Byte comparison is intentional: +0.0 and -0.0 must remain distinct.
    array = np.asarray(values)
    if array.ndim == 1 and array.size and array.flags.c_contiguous:
        raw = array.view(f"V{array.dtype.itemsize}")
        if np.all(raw == raw[0]):
            return [_value_bits(array[0], dtype)] * len(array)
    return [_value_bits(value, dtype) for value in values]


def model_objects(network):
    """MagicNetwork also lists the contained code runners; return its roots."""
    objects = set(network.objects)
    pending = list(objects)
    while pending:
        obj = pending.pop()
        for child in obj.contained_objects:
            if type(child) is Synapses:
                if child not in objects:
                    objects.add(child)
                    pending.append(child)
            else:
                objects.discard(child)
    return sorted(objects, key=lambda obj: obj.name)


def _endpoint(endpoint, groups):
    """Return parent population and the endpoint's contiguous local window."""
    if type(endpoint) in (NeuronGroup, PoissonGroup, SpikeGeneratorGroup,
                          SpatialNeuron):
        return endpoint, 0, len(endpoint)
    require(isinstance(endpoint, Subgroup),
            "Synapses endpoints must be NeuronGroups or contiguous subgroups")
    parents = [group for group in groups if group.name == endpoint.source.name]
    require(len(parents) == 1, "Synapses subgroup parent must be in the network")
    require(0 <= endpoint.start < endpoint.stop <= len(parents[0]),
            "invalid Synapses subgroup bounds")
    return parents[0], int(endpoint.start), int(endpoint.stop)


def _population(group, groups, group_indices, monitors, event_monitors, spike_monitors,
                rate_monitors,
                poisson_inputs, start, duration, namespace, offset, random_streams,
                clock_ids, run_clocks, function_contracts,
                recording_window_steps=None):
    """Validate and lower one NeuronGroup without consulting another group."""
    has_refractory = group._refractory is not False
    fixed_refractory = has_refractory and isinstance(group._refractory, Quantity)
    if has_refractory:
        require((fixed_refractory and group._refractory.size == 1) or
                isinstance(group._refractory, str),
                "refractory must be a fixed scalar duration or expression")
        if fixed_refractory:
            fail_for_dimension_mismatch(group._refractory, second)
        require(not prefs.legacy.refractory_timing, "legacy refractory timing is unsupported")
    method = group.state_updater.method_choice
    gsl_config = _gsl_configuration(group)
    require(gsl_config is None or type(group) is NeuronGroup,
            "GSL integration currently supports NeuronGroup populations only")
    require(supported_method_choice(method, group.equations),
            "method must use Brian2's default deterministic selection or be "
            "explicit 'exact', 'linear', 'independent', 'euler', 'rk2', "
            "'rk4', 'exponential_euler', 'heun', 'milstein', 'gsl' or "
            "'gsl_rk2', 'gsl_rk4', 'gsl_rkf45', 'gsl_rkck' or "
            "'gsl_rk8pd'; independent "
            "requires ODEs without cross-state dependencies")
    require(not group.state_updater.method_options or gsl_config is not None,
            "custom method options are supported only for GSL methods")

    gsl_meta_states = (_ensure_gsl_meta_variables(group, gsl_config)
                       if gsl_config is not None else [])

    states, declared_parameters, frozen_states, linked_names = [], [], [], []
    mutable_shared = []
    stochastic_names = set(group.equations.stochastic_variables)
    for equation in sorted(group.equations.values(), key=lambda eq: eq.varname):
        if has_refractory and equation.varname in {"lastspike", "not_refractory"}:
            continue
        if equation.type == "differential equation":
            require(not equation.flags or
                    (has_refractory and set(equation.flags) == {"unless refractory"}),
                    "ODE flags require fixed refractory and support only (unless refractory)")
            states.append(equation.varname)
            if equation.flags:
                frozen_states.append(equation.varname)
        elif equation.type == "subexpression":
            # Brian's state updater expands deterministic subexpressions into
            # its abstract code. They therefore need no runtime storage in
            # AtlasIR, but accepting them is essential for HH alpha/beta rates.
            require(not equation.flags or set(equation.flags) == {"shared"},
                    "subexpressions support only Brian's (shared) flag")
        else:
            require(equation.type == "parameter",
                    "only ODEs, deterministic subexpressions, and parameters")
            flags = set(equation.flags)
            if flags == {"linked"}:
                linked_names.append(equation.varname)
            elif flags == {"shared"}:
                # A Brian mutable shared parameter is assigned in the scalar
                # part of a run_regularly CodeRunner and then read by its
                # vector part.  AtlasIR stores a synchronized copy per neuron so
                # existing state persistence and monitor machinery can remain
                # unchanged; the restrictions below prove every copy receives
                # the same scalar value.
                require(group.variables[equation.varname].scalar,
                        "mutable (shared) parameters must use scalar storage")
                states.append(equation.varname)
                mutable_shared.append(equation.varname)
            elif not flags:
                # A plain Brian parameter (``x : 1``) is mutable per-neuron
                # model state. Store it beside ODE state so reset/on_pre,
                # monitors, result persistence and subsequent runs all share
                # one mutable array domain.
                require(not group.variables[equation.varname].scalar,
                        "mutable shared parameters are unsupported")
                states.append(equation.varname)
            else:
                require(flags in ({"constant"}, {"constant", "shared"}),
                        "parameters support mutable per-neuron state or "
                        "constant/constant-shared values")
                declared_parameters.append(equation.varname)
        dtype = np.dtype(group.variables[equation.varname].dtype)
        if equation.type == "differential equation":
            require(dtype in {np.dtype(np.float32), np.dtype(np.float64)},
                    "ODE states must use float32 or float64")
        else:
            require(dtype in {np.dtype(np.float32), np.dtype(np.float64),
                              np.dtype(np.int32), np.dtype(np.int64),
                              np.dtype(np.uint32), np.dtype(np.uint64),
                              np.dtype(np.bool_)},
                    "model variables must use f32/f64/i32/i64/u32/u64/bool")
    states.extend(gsl_meta_states)
    require(gsl_config is None or not linked_names,
            "GSL integration currently does not support linked variables")
    require(len(states) <= 32,
            "at most 32 mutable states per population")

    linked_variables = []
    for name in linked_names:
        variable = group.variables[name]
        source_owner = variable.owner
        source_groups = [candidate for candidate in groups
                         if candidate.id == source_owner.id]
        require(len(source_groups) == 1 and
                type(source_groups[0]) in (NeuronGroup, SpatialNeuron),
                f"{name}: linked source must be a NeuronGroup in the network")
        source_group = source_groups[0]
        source_name = variable.name
        require(source_name in source_group.equations.names,
                f"{name}: linked source must be declared in its NeuronGroup equations")
        source_equation = source_group.equations[source_name]
        require(source_equation.type == "differential equation" or
                (source_equation.type == "parameter" and
                 not source_equation.flags),
                f"{name}: linked source must be mutable per-neuron state")
        index_name = group.variables.indices[name]
        if index_name == "_idx":
            require(len(group) == len(source_group),
                    f"{name}: direct linked mapping requires equal group sizes")
            index = {"kind": "identity"}
        elif (isinstance(index_name, (int, np.integer)) or
              (isinstance(index_name, str) and
               index_name.removeprefix("-").isdigit())):
            # Brian represents a scalar linked-variable broadcast by storing
            # the fixed source index directly in ``variables.indices``.  Keep
            # the frozen AtlasIR schema and materialize that mapping as the
            # already-supported constant index form.
            source_index = int(index_name)
            require(0 <= source_index < len(source_group),
                    f"{name}: linked scalar broadcast is out of source bounds")
            index = {"kind": "constant",
                     "values": [source_index] * len(group)}
        elif index_name in states:
            index_variable = group.variables[index_name]
            require(np.dtype(index_variable.dtype).kind in "iu",
                    f"{name}: linked index state must be integer")
            index = {"kind": "state", "name": index_name}
        elif index_name in declared_parameters:
            index_variable = group.variables[index_name]
            require(np.dtype(index_variable.dtype).kind in "iu",
                    f"{name}: linked index parameter must be integer")
            values = np.asarray(index_variable.get_value()).reshape(-1)
            require(values.size == len(group),
                    f"{name}: linked index parameter must be per-neuron")
            integer_values = [int(value) for value in values]
            require(all(0 <= value < len(source_group) for value in integer_values),
                    f"{name}: linked index parameter is out of source bounds")
            index = {"kind": "constant", "values": integer_values}
        else:
            require(index_name in group.variables,
                    f"{name}: linked index mapping is unavailable")
            values = np.asarray(group.variables[index_name].get_value()).reshape(-1)
            require(values.size == len(group) and values.dtype.kind in "iu",
                    f"{name}: linked index array must contain one integer per neuron")
            integer_values = [int(value) for value in values]
            require(all(0 <= value < len(source_group) for value in integer_values),
                    f"{name}: linked index array is out of source bounds")
            index = {"kind": "constant", "values": integer_values}
        require(source_group is not group or index["kind"] == "identity",
                f"{name}: self-linked variables require identity mapping")
        linked_variables.append({
            "name": name,
            "dtype": dtype_name(variable.dtype),
            "dimensions": [float(dim) for dim in variable.dim._dims],
            "source_population": group_indices[source_group],
            "source_state": source_name,
            "index": index,
        })

    event_names = sorted(group.events)
    require(all(isinstance(name, str) and name.isidentifier()
                for name in event_names),
            "event names must be valid identifiers")
    has_spikes = "spike" in group.events
    reset_events = sorted(group.event_codes)
    require(set(reset_events) <= set(event_names),
            "run_on_event requires a matching event condition")
    require(not has_refractory or has_spikes,
            "refractory requires a spike threshold")
    runners = []
    if group.subexpression_updater is not None:
        runners.append((group.subexpression_updater,
                        "subexpression_update", "before_start", None))
    runners.append((group.state_updater, "state_update", "groups", None))
    if type(group) is SpatialNeuron:
        runners.append((group.diffusion_state_updater,
                        "spatial_state_update", "groups", None))
    for event_name in event_names:
        runner = group.thresholder[event_name]
        runners.append((runner, "threshold", runner.when, event_name))
    if has_spikes:
        for spike_monitor in spike_monitors:
            source_group, _, _ = _endpoint(spike_monitor.source, groups)
            expected_order = (spike_monitor.source.order + 1
                              if isinstance(spike_monitor.source, Subgroup) else 1)
            require(source_group is group,
                    "SpikeMonitor source must match its population")
            require(spike_monitor.when == "thresholds" and
                    spike_monitor.order == expected_order and
                    spike_monitor.clock is group.clock,
                    "SpikeMonitor must use its default schedule and group clock")
    else:
        require(not spike_monitors,
                "non-spiking population cannot have a SpikeMonitor")
    for rate_monitor in rate_monitors:
        source_group, _, _ = _endpoint(rate_monitor.source, groups)
        require(has_spikes and source_group is group and
                rate_monitor.when == "end" and rate_monitor.order == 0 and
                rate_monitor.clock is group.clock and
                not rate_monitor.contained_objects and
                np.dtype(rate_monitor.variables["rate"].dtype) in
                {np.dtype(np.float32), np.dtype(np.float64)},
                "PopulationRateMonitor needs a spiking source, default schedule, and f32/f64 rate")
    group_runners = list(runners)
    for event_name in reset_events:
        runner = group.resetter[event_name]
        group_runners.append((runner, "reset", runner.when, event_name))
    expected_runners = {runner for runner, _, _, _ in group_runners}
    expected_runners.update(child for child in group.contained_objects
                            if type(child) is Synapses and child.source is group)
    subgroups = sorted(
        (child for child in group.contained_objects if isinstance(child, Subgroup)),
        key=lambda subgroup: subgroup.name)
    expected_runners.update(subgroups)
    direct_regular_runners = list(
        runner for runner in group.contained_objects
        if runner not in expected_runners)
    require(all(type(runner) is CodeRunner and runner.group.id == group.id and
                not runner.contained_objects and runner.codeobj_class is None
                for runner in direct_regular_runners),
            "run_regularly must use the standard stateless CodeRunner")
    require(set(group.contained_objects) == expected_runners |
            set(direct_regular_runners),
            "unsupported population contained code")
    subgroup_runner_masks = {}
    for ordinal, subgroup in enumerate(subgroups):
        require(subgroup.source.id == group.id and
                0 <= subgroup.start < subgroup.stop <= len(group),
                "invalid contiguous subgroup")
        mask_name = f"b2_subgroup_mask_{ordinal}"
        index_name = f"b2_subgroup_index_{ordinal}"
        require(all(name not in group.variables and name not in namespace
                    for name in (mask_name, index_name)),
                "reserved subgroup parameter name is already in use")
        for runner in subgroup.contained_objects:
            require(type(runner) is CodeRunner and runner.group.id == subgroup.id and
                    not runner.contained_objects and runner.codeobj_class is None,
                    "subgroup run_regularly must use the standard stateless CodeRunner")
            subgroup_runner_masks[runner] = (mask_name, index_name, subgroup)
    regular_runners = sorted(
        direct_regular_runners + list(subgroup_runner_masks),
        key=lambda runner: (runner.when, runner.order, runner.name))
    for runner, kind, slot, _ in group_runners:
        require(runner.when == slot and runner.clock is group.clock and
                (kind in {"threshold", "reset"} or
                 (kind == "spatial_state_update" and runner.order == 1) or
                 runner.order == 0),
                "state updates require their default schedule; event threshold/reset "
                "runners must share the population clock")

    require(len(poisson_inputs) <= 32, "at most 32 PoissonInput objects per population")
    poisson_probabilities = {}
    poisson_counts = {}
    poisson_input_masks = {}
    for ordinal, poisson_input in enumerate(
            sorted(poisson_inputs, key=lambda item: item.name)):
        target_group, target_start, target_stop = _endpoint(
            poisson_input._group, groups)
        require(target_group is group and poisson_input.clock is group.clock,
                "PoissonInput must target its NeuronGroup or a contiguous subgroup "
                "and share its clock")
        require(poisson_input.when == "synapses" and poisson_input.order == 0 and
                not poisson_input.contained_objects,
                "PoissonInput must use the default synapses schedule and order")
        require(poisson_input.target_var in states,
                "PoissonInput target must be an ODE state")
        probability = float(poisson_input.rate * group.clock.dt)
        require(math.isfinite(probability) and 0 <= probability <= 1,
                "PoissonInput rate*dt must be a finite probability in [0, 1]")
        probability_name = f"b2_poisson_probability_{ordinal}"
        mask_name = f"b2_subgroup_mask_poisson_{ordinal}"
        reserved_names = ({probability_name, mask_name}
                          if type(poisson_input._group) is Subgroup else
                          {probability_name})
        require(all(name not in group.variables and name not in namespace
                    for name in reserved_names),
                f"reserved PoissonInput parameter name: {probability_name}")
        poisson_probabilities[poisson_input.name] = (probability_name, probability)
        if type(poisson_input._group) is Subgroup:
            poisson_input_masks[poisson_input] = (
                mask_name, target_start, target_stop)
        count = float(poisson_input.N)
        require(math.isfinite(count) and count.is_integer() and
                1 <= count <= 2**53,
                "PoissonInput N must be a positive exact integer")
        poisson_counts[poisson_input.name] = int(count)
    runners.extend((item, "poisson_input", "synapses", None)
                   for item in sorted(poisson_inputs, key=lambda item: item.name))
    runners.extend((group.resetter[event_name], "reset",
                    group.resetter[event_name].when, event_name)
                   for event_name in reset_events)
    runners.extend((runner, "run_regularly", runner.when, None)
                   for runner in regular_runners)

    state_monitors = []
    physical_variables, physical_record = [], []
    direct_monitorable = (set(states) | set(declared_parameters) |
                          set(linked_names) | set(gsl_meta_states))
    substituted_population_expressions = dict(
        group.equations.get_substituted_expressions(
            variables=dict(group.variables), include_subexpressions=True))
    deterministic_subexpressions = {
        name for name, expression in substituted_population_expressions.items()
        if (name in group.equations and
            group.equations[name].type == "subexpression" and
            get_identifiers(expression.code).isdisjoint(stochastic_names))
    }
    # A stochastic subexpression is expanded into the state updater, where its
    # named Wiener increment is lowered once and reused by that update.  It
    # cannot be reconstructed later for a monitor or Synapses endpoint without
    # drawing a second, semantically different random value.
    monitor_expressions = {
        equation.varname: equation.expr.code
        for equation in group.equations.values()
        if equation.varname in deterministic_subexpressions
    }
    endpoint_subexpressions = {
        name: expression.code
        for name, expression in substituted_population_expressions.items()
        if name in deterministic_subexpressions
    }
    monitorable = direct_monitorable | set(monitor_expressions)
    if has_refractory:
        monitorable |= {"lastspike", "not_refractory"}
    for monitor in monitors:
        source_group, source_start, source_stop = _endpoint(
            monitor.source, groups)
        require(source_group is group and
                set(monitor.record_variables) <= monitorable,
                "StateMonitor variables must be model states, parameters, "
                "linked variables, deterministic subexpressions, or refractory fields")
        require(len(monitor.record_variables) > 0 and
                len(set(monitor.record_variables)) == len(monitor.record_variables),
                "StateMonitor needs unique recorded variable names")
        local_record = [int(i) for i in monitor.record]
        require(all(0 <= i < source_stop - source_start for i in local_record),
                "recorded index out of range")
        record = [source_start + index for index in local_record]
        outputs = list(monitor.record_variables)
        physical = [name for name in outputs if name in direct_monitorable]
        if set(outputs) & set(monitor_expressions):
            for name in states + declared_parameters + linked_names:
                if name not in physical:
                    physical.append(name)
        require(physical, "StateMonitor needs at least one physical dependency")
        monitor_definition = {
            "name": monitor.name,
            "variables": physical,
            "output_variables": outputs,
            "record": record,
            "clock": clock_ids[monitor.clock],
        }
        if monitor.when != "start":
            monitor_definition["when"] = monitor.when
        if monitor.order != 0:
            monitor_definition["order"] = monitor.order
        state_monitors.append(monitor_definition)
        if record:
            for name in physical:
                if name not in physical_variables:
                    physical_variables.append(name)
            for neuron in record:
                if neuron not in physical_record:
                    physical_record.append(neuron)
    require(len({(monitor.when, monitor.order, monitor.clock)
                 for monitor in monitors}) <= 1,
            "StateMonitors sharing a source must use the same when/order/clock")
    require(recording_window_steps is None or not monitors or
            monitors[0].clock is group.clock,
            "bounded recording with an independent StateMonitor clock is unsupported")

    lowered_event_monitors = []
    for monitor in sorted(event_monitors, key=lambda item: item.name):
        variables = sorted(set(monitor.record_variables) - {"i", "t"})
        require(monitor.source is group and monitor.event in set(event_names) and
                monitor.record and set(variables) <= monitorable,
                "EventMonitor must record a declared event and floating-point model "
                "states/parameters")
        require(monitor.clock is group.clock and not monitor.contained_objects,
                "EventMonitor must share its source clock and have no children")
        lowered_event_monitors.append({
            "name": monitor.name, "event": monitor.event,
            "variables": variables, "clock": clock_ids[monitor.clock],
            "when": monitor.when, "order": monitor.order,
        })
    for spike_monitor in spike_monitors:
        variables = sorted(set(spike_monitor.record_variables) - {"i", "t"})
        require(set(variables) <= direct_monitorable,
                "SpikeMonitor variables must be model states, parameters, or "
                "linked variables")
        if variables:
            # The ordinary spike history remains in the compact population
            # result.  Additional values use the typed EventMonitor sidecar so
            # they are sampled at the same scheduled spike event, before reset.
            lowered_event_monitors.append({
                "name": spike_monitor.name, "event": "spike",
                "variables": variables, "clock": clock_ids[spike_monitor.clock],
                "when": spike_monitor.when, "order": spike_monitor.order,
            })

    dt = float(group.clock.dt / second)
    require(math.isfinite(dt) and dt > 0, "dt must be finite and positive")
    steps = run_clocks[clock_ids[group.clock]]["steps"]
    monitor_steps = (steps if recording_window_steps is None else
                     min(steps, recording_window_steps))
    require(steps <= population_step_budget() and
            steps * len(group) <= MAX_NEURON_TICKS,
            "probe population duration budget exceeded")
    if monitors:
        state_monitor_steps = run_clocks[clock_ids[monitors[0].clock]]["steps"]
    else:
        state_monitor_steps = monitor_steps
    require(state_monitor_steps * len(physical_record) *
            len(physical_variables) <= 100_000_000,
            "probe output budget: <=100,000,000 recorded values per population")

    refractory_definition = refractory_instance = None
    if has_refractory:
        period = float(group._refractory / second) if fixed_refractory else 0.0
        period_steps = (period + 1e-3 * dt) / dt if fixed_refractory else 0.0
        require(math.isfinite(period) and period >= 0 and
                math.isfinite(period_steps) and period_steps < 1_000_001,
                "refractory duration must be finite, non-negative, and at most "
                "1,000,000 ticks")
        lastspike = np.asarray(group.variables["lastspike"].get_value())
        available = np.asarray(group.variables["not_refractory"].get_value())
        require(lastspike.dtype == np.float64 and available.dtype == np.bool_ and
                lastspike.shape == available.shape == (len(group),),
                "invalid refractory state arrays")
        start_seconds = float(start / second)
        require(np.isfinite(lastspike).all() and np.all(lastspike <= start_seconds),
                "initial lastspike must be finite and at or before the run start")
        elapsed = ((start_seconds + steps * dt - lastspike) + 1e-3 * dt) / dt
        require(np.isfinite(elapsed).all() and np.all(elapsed < 2**63),
                "refractory timestep exceeds Brian2 int64 range")
        refractory_definition = {
            "frozen_states": frozen_states,
            "mode": "fixed" if fixed_refractory else "expression",
        }
        refractory_instance = {
            "period": bits(period), "period_ticks": int(period_steps),
            "initial_lastspike": _array_bits(lastspike, "f64"),
            "initial_not_refractory": available.tolist(),
        }

    group.equations.check_units(group, namespace)
    resolved = dict(group.variables)
    resolved["_cond"] = AuxiliaryVariable("_cond", dtype=bool)
    for mask_name, index_name, _ in subgroup_runner_masks.values():
        resolved[mask_name] = AuxiliaryVariable(mask_name, dtype=bool)
        resolved[index_name] = AuxiliaryVariable(index_name, dtype=np.int32)
    for probability_name, probability in poisson_probabilities.values():
        resolved[probability_name] = Constant(
            probability_name, probability, owner=group)
    for mask_name, _, _ in poisson_input_masks.values():
        resolved[mask_name] = AuxiliaryVariable(mask_name, dtype=bool)
    external_names = set(declared_parameters) | {
        name for name, _ in poisson_probabilities.values()}
    codes = []
    for runner, kind, _, event_name in runners:
        adaptive = None
        if kind == "state_update" and gsl_config is not None:
            derivatives = group.equations.get_substituted_expressions(
                variables=dict(group.variables))
            derivative_names = [
                name for name, _expression in derivatives
                if group.equations[name].type == "differential equation"
            ]
            require(derivative_names,
                    "GSL integration requires at least one differential equation")
            temporary_names = {
                name: f"_b2_gsl_derivative_{position}"
                for position, name in enumerate(derivative_names)
            }
            expression_by_name = dict(derivatives)
            code = "\n".join(
                f"{temporary_names[name]} = {expression_by_name[name]}"
                for name in derivative_names)
            adaptive = {
                "integrator": gsl_config["integrator"],
                "states": derivative_names,
                "derivatives": [temporary_names[name]
                                for name in derivative_names],
                "absolute_errors": [bits(gsl_config["absolute_errors"][name])
                                    for name in derivative_names],
                "adaptable_timestep": gsl_config["adaptable_timestep"],
                "max_steps": gsl_config["max_steps"],
                "use_last_timestep": gsl_config["use_last_timestep"],
                "last_timestep": ("_last_timestep"
                                  if gsl_config["use_last_timestep"] else None),
                "failed_steps": ("_failed_steps"
                                 if gsl_config["save_failed_steps"] else None),
                "step_count": ("_step_count"
                               if gsl_config["save_step_count"] else None),
                "frozen_states": list(frozen_states),
            }
        else:
            runner.update_abstract_code(namespace)
            code = runner.abstract_code
        if (kind == "state_update" and not code.strip() and
                not states and not has_refractory):
            # A completely inert NeuronGroup is useful as a Synapses endpoint.
            # It has a Brian StateUpdater object but no runtime operation.
            continue
        if runner in subgroup_runner_masks:
            _, index_name, subgroup = subgroup_runner_masks[runner]
            code = word_substitute(
                code, {"i": index_name, "N": str(len(subgroup))})
        if (has_refractory and kind == "state_update" and
                fixed_refractory and adaptive is None):
            prefix = runner._get_refractory_code(namespace)
            require(code.startswith(prefix), "unexpected refractory state updater")
            code = code[len(prefix):]
        elif has_refractory and kind == "threshold":
            code = runner.user_code
        additional = {"_cond": resolved["_cond"]}
        adaptive_temporaries = set()
        if adaptive is not None:
            require(not has_refractory or fixed_refractory,
                    "GSL currently supports only fixed-duration refractory periods")
            for state, temporary in zip(adaptive["states"],
                                        adaptive["derivatives"], strict=True):
                additional[temporary] = AuxiliaryVariable(
                    temporary,
                    dimensions=group.variables[state].dim / second.dim,
                    dtype=group.variables[state].dtype)
                adaptive_temporaries.add(temporary)
        for mask_name, index_name, _ in subgroup_runner_masks.values():
            additional[mask_name] = resolved[mask_name]
            additional[index_name] = resolved[index_name]
        random_functions = {}
        if kind == "poisson_input":
            additional.update(runner.variables)
            function_names = [name for name, var in runner.variables.items()
                              if isinstance(var, Function)]
            require(len(function_names) == 1,
                    "PoissonInput must expose exactly one binomial function")
            probability_name, _ = poisson_probabilities[runner.name]
            random_functions[function_names[0]] = {
                "n": poisson_counts[runner.name], "p_name": probability_name,
                "approximate": True}
        analysis_variables = dict(resolved)
        analysis_variables.update(additional)
        _, used, unknown = analyse_identifiers(code, analysis_variables, recursive=True)
        variables = group.resolve_all(used | unknown, namespace,
                                      additional_variables=additional)
        variables.update(additional)
        written = {
            parse_statement(line)[0]
            for line in code.splitlines() if line.strip()
        }
        # In Brian's generated state-updater code, ``xi`` no longer denotes
        # white noise with dimensions time**-0.5.  It is a private Wiener
        # increment assigned as ``sqrt(dt) * randn()`` and therefore has
        # dimensions time**0.5.  Use that generated-code meaning for both
        # Brian's statement unit check and AtlasIR dimension inference.
        for name in stochastic_names & set(variables):
            variables[name] = AuxiliaryVariable(
                name, dimensions=(second**0.5).dim,
                dtype=group.variables[name].dtype)
        for name, config in _random_function_configs(variables).items():
            random_functions.setdefault(name, config)
        code_variables = dict(resolved)
        code_variables.update(variables)
        for name in (set(mutable_shared) & set(code_variables)) - written:
            variable = code_variables[name]
            # AtlasIR represents mutable shared storage as synchronized neuron
            # lanes. Code objects that only read it therefore lower the read
            # as a vector value instead of Brian's scalar-loop hoist. A scalar
            # run_regularly writer retains Brian's original scalar metadata.
            code_variables[name] = AuxiliaryVariable(
                name, dimensions=variable.dim, dtype=variable.dtype,
                scalar=False)
        # Brian has already unit-checked differential equations before it
        # expands them into a numerical state updater. That generated code can
        # replace a dimensionful zero (for example ``0*mV`` in ``clip``) with
        # the polymorphic literal ``0``, which Brian's source-level checker no
        # longer accepts. AtlasIR's own dimension inference below deliberately
        # preserves this zero-literal rule and validates every assignment.
        if kind not in {"state_update"}:
            check_units_statements(code, code_variables)
        intrinsic = {"lastspike", "not_refractory", "int"} if has_refractory else set()
        if has_refractory and "int" in variables:
            require(variables["int"] is DEFAULT_FUNCTIONS["int"],
                    "custom refractory int function is unsupported")
        hidden_subgroup_parameters = {
            name for mask_name, index_name, _ in subgroup_runner_masks.values()
            for name in (mask_name, index_name)}
        hidden_subgroup_parameters.update(
            mask_name for mask_name, _, _ in poisson_input_masks.values())
        external_names |= (set(variables) - set(states) - set(linked_names) -
                           set(monitor_expressions) -
                           stochastic_names -
                           set(random_functions) -
                           adaptive_temporaries -
                           hidden_subgroup_parameters -
                           {"dt", "t", "i", "N", "_cond"} - intrinsic)
        resolved.update(variables)
        codes.append((runner, kind, code, code_variables,
                      random_functions, event_name, adaptive))

    # Monitor-only subexpressions can use scalar namespace values not used by
    # state updates. Freeze them in the same model instance.
    monitor_timed_arrays = {}
    for expression in monitor_expressions.values():
        identifiers = get_identifiers(expression)
        variables = group.resolve_all(identifiers, namespace,
                                      user_identifiers=identifiers)
        resolved.update(variables)
        for name, variable in variables.items():
            if isinstance(variable, TimedArray):
                monitor_timed_arrays[name] = timed_array_config(
                    variable, group.clock.dt_)
        external_names |= (set(variables) - set(states) - set(linked_names) -
                           set(monitor_expressions) - {"dt", "t", "i", "N",
                                                      "lastspike", "not_refractory"})

    parameters, parameter_values = [], {}
    for name in sorted(external_names):
        var = resolved[name]
        if _supported_function(name, var, function_contracts):
            continue
        require(not isinstance(var, Function) and var.constant,
                f"{name}: only built-in deterministic math functions or constants")
        require(var.scalar or name in declared_parameters,
                f"{name}: declare per-neuron parameters in the equations")
        value = np.asarray(var.get_value()).reshape(-1)
        if name == "inf" and var is DEFAULT_CONSTANTS["inf"]:
            # Public AtlasIR storage is finite-only. Generated state-updater code
            # nevertheless keeps Brian's ``inf*unit`` clip bound as a named
            # constant. Saturating that built-in constant to the largest f64
            # preserves the clip result for all representable model states.
            value = np.asarray([np.finfo(np.float64).max])
        require(value.size == (1 if var.scalar else len(group)) and
                value.dtype.kind in "fibu", f"{name}: invalid numeric parameter shape")
        parameter = symbol(name, var)
        parameters.append(parameter)
        parameter_values[name] = _array_bits(value, parameter["dtype"])
    for mask_name, index_name, subgroup in sorted(
            set(subgroup_runner_masks.values()), key=lambda item: item[0]):
        mask = np.zeros(len(group), dtype=np.bool_)
        mask[subgroup.start:subgroup.stop] = True
        local_index = np.zeros(len(group), dtype=np.int32)
        local_index[subgroup.start:subgroup.stop] = np.arange(
            len(subgroup), dtype=np.int32)
        parameters.append(symbol(mask_name, resolved[mask_name], "neuron"))
        parameter_values[mask_name] = _array_bits(mask, "bool")
        parameters.append(symbol(index_name, resolved[index_name], "neuron"))
        parameter_values[index_name] = _array_bits(local_index, "i32")
    for mask_name, target_start, target_stop in sorted(
            poisson_input_masks.values(), key=lambda item: item[0]):
        mask = np.zeros(len(group), dtype=np.bool_)
        mask[target_start:target_stop] = True
        parameters.append(symbol(mask_name, resolved[mask_name], "neuron"))
        parameter_values[mask_name] = _array_bits(mask, "bool")
    require(len(parameters) <= 128, "at most 128 parameters per population")
    total_values = len(states) * len(group) + sum(len(v) for v in parameter_values.values())
    if has_refractory:
        total_values += 2 * len(group)
    require(total_values <= initial_value_budget(),
            f"probe array budget: <={initial_value_budget():,} values per population")
    inputs = (set(states) | set(parameter_values) | set(linked_names) |
              {"dt", "t", "i", "N"})
    if has_refractory:
        inputs |= {"lastspike", "not_refractory"}
    specs = []
    for (runner, kind, code, code_variables, random_functions, event_name,
         adaptive) in codes:
        writable = set() if adaptive is not None else set(states)
        if (kind == "state_update" and has_refractory and
                not fixed_refractory):
            writable.add("not_refractory")
        # Brian's ``(unless refractory)`` flag is a conditional-write rule on
        # the state variable, not only a state-updater concern.  In
        # particular, PoissonInput and run_regularly writes have to be
        # suppressed during the refractory interval as well.  Reset code is
        # deliberately exempt: it runs after the threshold has marked the
        # neuron refractory and must still assign the reset value.
        conditionally_frozen = kind not in {"threshold", "reset"}
        conditional_writes = ({s: "not_refractory" for s in frozen_states}
                              if conditionally_frozen else {})
        if runner in subgroup_runner_masks:
            require(not frozen_states,
                    "subgroup run_regularly cannot write refractory-frozen states")
            mask_name, _, _ = subgroup_runner_masks[runner]
            conditional_writes.update({state: mask_name for state in states})
        if runner in poisson_input_masks:
            mask_name, _, _ = poisson_input_masks[runner]
            conditional_writes[runner.target_var] = mask_name
        spec = CodeObjectSpec.from_abstract(
            runner, kind, code, code_variables, inputs, writable,
            conditional_writes=conditional_writes,
            allocate_random_stream=random_streams.allocate,
            random_functions=random_functions,
            functions=function_contracts).to_dict()
        if adaptive is not None:
            spec["adaptive"] = adaptive
            adaptive_writes = set(adaptive["states"])
            adaptive_writes |= {
                name for name in (adaptive["last_timestep"],
                                  adaptive["failed_steps"],
                                  adaptive["step_count"])
                if name is not None
            }
            spec["effects"]["writes"] = sorted(adaptive_writes)
        _demote_mutable_shared_scalar_reads(spec, mutable_shared)
        spec["clock"] = clock_ids[runner.clock]
        if event_name is not None:
            spec["event_name"] = event_name
        specs.append(spec)
    for name in mutable_shared:
        writers = [spec for spec in specs if name in spec["effects"]["writes"]]
        require(len(writers) <= 1,
                f"{name}: mutable (shared) parameter must have at most one writer")
        if not writers:
            # The scalar can be changed by user Python between continuation
            # runs (including a NetworkOperation).  Each exported segment
            # broadcasts the current scalar into the IR's per-neuron storage.
            continue
        require(writers[0]["kind"] == "run_regularly",
                f"{name}: mutable (shared) parameter writer must be a "
                "run_regularly CodeRunner")
        writer = writers[0]
        scalar_assignments = [statement for statement in writer["scalar"]
                              if statement["target"] == name]
        vector_assignments = [statement for statement in writer["vector"]
                              if statement["target"] == name]
        require(len(scalar_assignments) == 1 and not vector_assignments and
                scalar_assignments[0]["condition"] is None,
                f"{name}: mutable (shared) assignment must be one unconditional "
                "scalar statement")
        require(name not in loads(scalar_assignments[0]["value"]),
                f"{name}: mutable (shared) assignment cannot read its previous value")
    definition = {
        "name": group.name, "offset": offset, "count": len(group),
        "states": [symbol(name, group.variables[name], "neuron")
                   for name in states],
        "parameters": parameters, "linked_variables": linked_variables,
        "refractory": refractory_definition,
        "events": event_names,
        "code_objects": specs,
        "state_monitors": state_monitors,
        "monitor_expressions": monitor_expressions,
        "event_monitors": lowered_event_monitors,
        "spike_monitor": (spike_monitors[0].name if spike_monitors else
                          rate_monitors[0].name if rate_monitors else
                          next((item.name for item in monitors
                                if set(item.record_variables) &
                                {"lastspike", "not_refractory"}), None)),
        "rate_monitors": [{"name": item.name,
                           "dtype": dtype_name(item.variables["rate"].dtype)}
                          for item in rate_monitors],
        "monitor": {"variables": physical_variables, "record": physical_record,
                    "window_steps": monitor_steps},
        "clock": clock_ids[group.clock], "dt": bits(dt), "steps": steps,
    }
    if type(group) is SpatialNeuron:
        morphology = group.flat_morphology
        definition["spatial"] = {
            "voltage": "v", "membrane_current": "Ic",
            "capacitance": "Cm", "resistivity": "Ri",
            "area": "area", "r_length_1": "r_length_1",
            "r_length_2": "r_length_2",
            "starts": np.asarray(morphology.starts, dtype=np.int64).tolist(),
            "ends": np.asarray(morphology.ends, dtype=np.int64).tolist(),
            "parents": np.asarray(
                morphology.morph_parent_i, dtype=np.int64).tolist(),
            "child_slots": np.asarray(
                morphology.morph_idxchild, dtype=np.int64).tolist(),
            "children_count": np.asarray(
                morphology.morph_children_num, dtype=np.int64).tolist(),
            "children": np.asarray(
                morphology.morph_children, dtype=np.int64).tolist(),
        }
    if monitor_timed_arrays:
        definition["monitor_timed_arrays"] = monitor_timed_arrays
    instance = {
        "initial_state": {
            name: _array_bits(
                (np.repeat(np.asarray(group.variables[name].get_value()).reshape(-1)[0],
                           len(group))
                 if name in mutable_shared else
                 group.variables[name].get_value()),
                dtype_name(group.variables[name].dtype),
            )
            for name in states
        },
        "parameters": parameter_values,
        "refractory": refractory_instance,
        "spike_generator": None,
    }
    return definition, instance, {"states": states, "frozen": frozen_states,
                                  "parameters": [item["name"]
                                                 for item in parameters],
                                  "mutable_shared": mutable_shared,
                                  "linked": linked_names,
                                  "subexpressions": endpoint_subexpressions,
                                  "resolved": resolved, "dt": dt,
                                  "has_spikes": has_spikes,
                                  "events": set(event_names),
                                  "value_count": total_values}


def _poisson_population(group, monitors, spike_monitor, duration, namespace, offset,
                        random_streams, clock_ids, run_clocks, function_contracts,
                        recording_window_steps=None):
    """Lower a PoissonGroup, including mutable rates and their monitors."""
    require(group._refractory is False and set(group.events) == {"spike"},
            "PoissonGroup must use its default spike event")
    threshold = group.thresholder["spike"]
    children = set(group.contained_objects)
    regular_runners = sorted(
        (child for child in children
         if child is not threshold and type(child) is CodeRunner),
        key=lambda runner: (runner.when, runner.order, runner.name))
    require(threshold in children and
            all(child is threshold or
                (type(child) is Synapses and child.source is group) or
                child in regular_runners
                for child in children) and
            threshold.when == "thresholds" and threshold.order == 0 and
            threshold.clock is group.clock,
            "PoissonGroup must use its default threshold schedule")
    require(all(runner.group.id == group.id and
                not runner.contained_objects and runner.codeobj_class is None
                for runner in regular_runners),
            "PoissonGroup run_regularly must use the standard stateless CodeRunner")
    if spike_monitor is not None:
        require(spike_monitor.source is group and
                not spike_monitor.record_variables - {"i", "t"} and
                spike_monitor.when == "thresholds" and spike_monitor.order == 1 and
                spike_monitor.clock is group.clock,
                "PoissonGroup SpikeMonitor must record only i/t on the default schedule")

    rates = group.variables["rates"]
    dynamic_rates = isinstance(rates, Subexpression)
    require(not (dynamic_rates and regular_runners),
            "expression-rate PoissonGroup cannot also assign rates with run_regularly")
    require(not (dynamic_rates and monitors),
            "expression-rate PoissonGroup StateMonitor is unsupported")
    dt = float(group.clock.dt / second)
    require(math.isfinite(dt) and dt > 0, "dt must be finite and positive")
    steps = run_clocks[clock_ids[group.clock]]["steps"]
    monitor_steps = (steps if recording_window_steps is None else
                     min(steps, recording_window_steps))
    require(steps <= population_step_budget() and
            steps * len(group) <= MAX_NEURON_TICKS,
            "probe population budget exceeded")

    threshold.update_abstract_code(namespace)
    code = threshold.abstract_code
    resolved = dict(group.variables)
    resolved["_cond"] = AuxiliaryVariable("_cond", dtype=bool)
    if dynamic_rates:
        # Brian's thresholder keeps ``rates`` as a read-only Subexpression in
        # its abstract code.  Materialise that expression as an IR-local
        # temporary so t/i/TimedArray dependencies remain runtime values.
        resolved["_rates"] = AuxiliaryVariable(
            "_rates", dimensions=rates.dim, dtype=np.float64)
        code = f"_rates = {rates.expr}\n_cond = rand() < _rates * dt"
    _, used, unknown = analyse_identifiers(code, resolved, recursive=True)
    additional = {name: resolved[name] for name in ("_cond", "_rates")
                  if name in resolved}
    variables = group.resolve_all(used | unknown, namespace,
                                  additional_variables=additional)
    check_units_statements(code, variables)
    require(variables.get("rand") is DEFAULT_FUNCTIONS["rand"],
            "PoissonGroup requires Brian2's built-in rand function")
    resolved.update(variables)
    parameter_symbols, parameter_values = [], {}
    initial_state = {}
    if dynamic_rates:
        external_names = (set(variables) - {"dt", "t", "i", "N", "_cond",
                                            "_rates", "rand"})
        for name in sorted(external_names):
            var = variables[name]
            if _supported_function(name, var, function_contracts):
                continue
            require(not isinstance(var, Function) and var.constant and var.scalar,
                    f"{name}: dynamic PoissonGroup rates use scalar constants or "
                    "TimedArray inputs only")
            value = np.asarray(var.get_value()).reshape(-1)
            require(value.size == 1 and value.dtype.kind in "fiu" and
                    np.isfinite(value).all(),
                    f"{name}: invalid PoissonGroup rate parameter")
            parameter_symbols.append(symbol(name, var))
            parameter_values[name] = [bits(value[0])]
    else:
        values = np.asarray(rates.get_value()).reshape(-1)
        require(values.dtype == np.float64 and values.size == len(group) and
                np.isfinite(values).all() and np.all(values >= 0),
                "PoissonGroup rates must be finite and non-negative")
        if regular_runners:
            initial_state = {"rates": [bits(v) for v in values]}
        else:
            parameter_symbols = [symbol("rates", rates)]
            parameter_values = {"rates": [bits(v) for v in values]}
    inputs = {"dt", "t", "i", "N"} | set(parameter_values) | set(initial_state)
    if not dynamic_rates:
        inputs.add("rates")
    spec = CodeObjectSpec.from_abstract(
        threshold, "threshold", code, resolved,
        inputs, set(),
        allocate_random_stream=random_streams.allocate,
        functions=function_contracts).to_dict()
    spec["clock"] = clock_ids[threshold.clock]
    spec["event_name"] = "spike"
    specs = [spec]
    for runner in regular_runners:
        runner.update_abstract_code(namespace)
        runner_variables = dict(group.variables)
        _, used, unknown = analyse_identifiers(
            runner.abstract_code, runner_variables, recursive=True)
        runner_variables.update(group.resolve_all(used | unknown, namespace))
        random_functions = _random_function_configs(runner_variables)
        check_units_statements(runner.abstract_code, runner_variables)
        allowed = {"dt", "t", "i", "N", "rates"}
        for name in sorted((used | unknown) - allowed):
            if name in parameter_values:
                allowed.add(name)
                continue
            variable = runner_variables[name]
            if name in random_functions or _supported_function(
                    name, variable, function_contracts):
                continue
            require(not isinstance(variable, Function) and
                    variable.constant and variable.scalar,
                    f"{name}: PoissonGroup run_regularly supports scalar constants only")
            values = np.asarray(variable.get_value()).reshape(-1)
            require(values.size == 1 and values.dtype.kind in "fiu" and
                    np.isfinite(values).all(),
                    f"{name}: invalid PoissonGroup run_regularly constant")
            parameter = symbol(name, variable, "scalar")
            parameter_symbols.append(parameter)
            parameter_values[name] = _array_bits(values, parameter["dtype"])
            allowed.add(name)
        runner_spec = CodeObjectSpec.from_abstract(
            runner, "run_regularly", runner.abstract_code, runner_variables,
            allowed, {"rates"}, allocate_random_stream=random_streams.allocate,
            random_functions=random_functions,
            functions=function_contracts).to_dict()
        runner_spec["clock"] = clock_ids[runner.clock]
        specs.append(runner_spec)

    state_monitors = []
    physical_record = []
    for monitor in monitors:
        require(monitor.source is group and
                list(monitor.record_variables) == ["rates"],
                "PoissonGroup StateMonitor must record rates")
        record = [int(index) for index in monitor.record]
        require(all(0 <= index < len(group) for index in record),
                "PoissonGroup StateMonitor record index out of range")
        state_monitors.append({
            "name": monitor.name, "variables": ["rates"],
            "output_variables": ["rates"], "record": record,
            "clock": clock_ids[monitor.clock], "when": monitor.when,
            "order": monitor.order,
        })
        for neuron in record:
            if neuron not in physical_record:
                physical_record.append(neuron)
    require(len({(monitor.when, monitor.order, monitor.clock)
                 for monitor in monitors}) <= 1,
            "StateMonitors sharing a PoissonGroup must use one schedule")
    require(recording_window_steps is None or not monitors or
            monitors[0].clock is group.clock,
            "bounded recording with an independent StateMonitor clock is unsupported")
    if monitors:
        monitor_ticks = run_clocks[clock_ids[monitors[0].clock]]["steps"]
        require(monitor_ticks * len(physical_record) <= 100_000_000,
                "PoissonGroup StateMonitor output budget exceeded")
    definition = {
        "name": group.name, "offset": offset, "count": len(group),
        "states": ([symbol("rates", rates)] if regular_runners else []),
        "parameters": parameter_symbols, "linked_variables": [],
        "refractory": None, "events": ["spike"],
        "code_objects": specs, "state_monitors": state_monitors,
        "event_monitors": [],
        "monitor_expressions": {},
        "spike_monitor": spike_monitor.name if spike_monitor is not None else None,
        "rate_monitors": [],
        "monitor": {"variables": (["rates"] if monitors else []),
                    "record": physical_record,
                    "window_steps": monitor_steps},
        "clock": clock_ids[group.clock], "dt": bits(dt), "steps": steps,
    }
    instance = {
        "initial_state": initial_state, "parameters": parameter_values,
        "refractory": None, "spike_generator": None,
    }
    return definition, instance, {
        "states": (["rates"] if regular_runners else []),
        "frozen": [],
        "parameters": [item["name"] for item in parameter_symbols],
        "resolved": resolved, "dt": dt,
        "has_spikes": True, "events": {"spike"},
        "value_count": (sum(len(values) for values in parameter_values.values()) +
                        sum(len(values) for values in initial_state.values())),
    }


def _spike_generator_population(group, spike_monitor, start, duration, offset,
                                clock_ids, run_clocks,
                                recording_window_steps=None):
    """Lower Brian's static/repeating SpikeGeneratorGroup schedule."""
    require(group.when == "thresholds" and group.order == 0 and
            not group.contained_objects and group.codeobj_class is None,
            "SpikeGeneratorGroup must use its default schedule and code object")
    if spike_monitor is not None:
        require(spike_monitor.source is group and
                not spike_monitor.record_variables - {"i", "t"} and
                spike_monitor.when == "thresholds" and spike_monitor.order == 1 and
                spike_monitor.clock is group.clock,
                "SpikeGeneratorGroup SpikeMonitor must record only i/t on the default schedule")
    dt = float(group.clock.dt / second)
    require(math.isfinite(dt) and dt > 0, "dt must be finite and positive")
    clock_run = run_clocks[clock_ids[group.clock]]
    steps = clock_run["steps"]
    monitor_steps = (steps if recording_window_steps is None else
                     min(steps, recording_window_steps))
    require(steps <= population_step_budget() and
            steps * len(group) <= MAX_NEURON_TICKS,
            "probe population budget exceeded")

    indices = np.asarray(group._neuron_index, dtype=np.int64)
    times = np.asarray(group._spike_time, dtype=np.float64)
    require(indices.shape == times.shape and np.isfinite(times).all() and
            np.all(times >= 0) and np.all((indices >= 0) & (indices < len(group))),
            "invalid SpikeGeneratorGroup indices/times")
    timebins = np.asarray((times + 1e-3 * dt) / dt, dtype=np.int64)
    if len(timebins) > 1:
        duplicate = ((np.diff(timebins) == 0) & (np.diff(indices) == 0)).any()
        require(not duplicate,
                "SpikeGeneratorGroup cannot emit twice from one neuron in one timestep")
    period = float(group.period / second)
    require(math.isfinite(period) and period >= 0,
            "SpikeGeneratorGroup period must be finite and non-negative")
    period_ticks = int(round(period / dt)) if period else 0
    if period:
        require(period_ticks >= 1 and
                abs(period_ticks * dt - period) <= period * np.finfo(float).eps,
                "SpikeGeneratorGroup period must be an integer multiple of dt")
        require(not len(timebins) or int(timebins.max()) < period_ticks,
                "SpikeGeneratorGroup period must exceed its latest spike")

    start_tick = clock_run["start_tick"]
    end_tick = start_tick + steps
    schedule = []
    for spike_tick, neuron in zip(timebins, indices, strict=True):
        if period_ticks:
            repeat = max(0, (start_tick - int(spike_tick) + period_ticks - 1) //
                         period_ticks)
            tick = int(spike_tick) + repeat * period_ticks
            while tick < end_tick:
                schedule.append((tick, int(neuron)))
                tick += period_ticks
        elif start_tick <= spike_tick < end_tick:
            schedule.append((int(spike_tick), int(neuron)))
    schedule.sort()
    require(len(schedule) <= 10_000_000,
            "at most 10,000,000 generated spikes per run")
    definition = {
        "name": group.name, "offset": offset, "count": len(group),
        "states": [], "parameters": [], "linked_variables": [], "refractory": None,
        "events": ["spike"],
        "code_objects": [], "state_monitors": [], "event_monitors": [],
        "monitor_expressions": {},
        "spike_monitor": spike_monitor.name if spike_monitor is not None else None,
        "rate_monitors": [],
        "monitor": {"variables": [], "record": [],
                    "window_steps": monitor_steps},
        "clock": clock_ids[group.clock], "dt": bits(dt), "steps": steps,
    }
    instance = {
        "initial_state": {}, "parameters": {}, "refractory": None,
        "spike_generator": {
            "spike_ticks": [tick for tick, _ in schedule],
            "spike_indices": [neuron for _, neuron in schedule],
        },
    }
    return definition, instance, {
        "states": [], "frozen": [], "parameters": [],
        "resolved": dict(group.variables),
        "dt": dt, "has_spikes": True, "events": {"spike"},
        "value_count": 2 * len(schedule),
    }


def _lower_network(network, duration, namespace=None, rng_seed=0,
                   recording_window_steps=None):
    """Lower independently shaped populations and their Synapses."""
    initial_value_budget()  # Reject invalid process configuration before lowering.
    namespace = {} if namespace is None else namespace
    require(type(rng_seed) is int and 0 <= rng_seed < 2**64,
            "rng_seed must be a u64 integer")
    require(type(network) in (Network, MagicNetwork), "expected Network or MagicNetwork")
    start = network.t
    start_seconds = float(start / second)
    require(math.isfinite(start_seconds) and start_seconds >= 0,
            "network start time must be finite and non-negative")
    fail_for_dimension_mismatch(duration, second)
    duration_seconds = float(duration / second)
    require(math.isfinite(duration_seconds) and duration_seconds >= 0 and
            math.isfinite(start_seconds + duration_seconds),
            "duration must be finite and non-negative")
    objects = model_objects(network)
    clock_objects = sorted(
        {obj.clock for obj in network.sorted_objects
         if type(obj) is not NetworkOperation},
        key=lambda clock: clock.name)
    require(clock_objects, "at least one active clock is required")
    clock_ids = {clock: index for index, clock in enumerate(clock_objects)}
    clock_definitions = []
    run_clocks = []
    for clock in clock_objects:
        dt = float(clock.dt / second)
        require(math.isfinite(dt) and dt > 0,
                "Clock dt must be finite and positive")
        start_tick = int(clock._calc_timestep(start_seconds))
        end_tick = int(clock._calc_timestep(start_seconds + duration_seconds))
        require(0 <= start_tick <= end_tick,
                "Clock interval must have ordered non-negative ticks")
        clock_definitions.append({"name": clock.name, "dt": bits(dt)})
        run_clocks.append({
            "start_tick": start_tick,
            "steps": end_tick - start_tick,
        })
    require(all(type(obj) in (NeuronGroup, SpatialNeuron,
                              PoissonGroup, SpikeGeneratorGroup,
                              PoissonInput,
                              StateMonitor, EventMonitor, SpikeMonitor,
                              PopulationRateMonitor, Synapses, NetworkOperation)
                for obj in objects),
            "only NeuronGroup/PoissonGroup/SpikeGeneratorGroup/PoissonInput/StateMonitor/"
            "EventMonitor/SpikeMonitor/PopulationRateMonitor/Synapses/NetworkOperation")
    groups = [obj for obj in objects
              if type(obj) in (NeuronGroup, SpatialNeuron,
                               PoissonGroup, SpikeGeneratorGroup)]
    group_indices = {group: index for index, group in enumerate(groups)}
    neuron_groups = [obj for obj in groups
                     if type(obj) in (NeuronGroup, SpatialNeuron)]
    monitors = [obj for obj in objects if type(obj) is StateMonitor]
    event_monitors = [obj for obj in objects if type(obj) is EventMonitor]
    spikes = [obj for obj in objects if type(obj) is SpikeMonitor]
    rates = [obj for obj in objects if type(obj) is PopulationRateMonitor]
    synapses = [obj for obj in objects if type(obj) is Synapses]
    poisson_inputs = [obj for obj in objects if type(obj) is PoissonInput]
    require(len(groups) >= 1, "at least one population is required")
    require(sum(len(group) for group in groups) <= neuron_budget(), "neuron preparation budget exceeded")
    require(len(poisson_inputs) <= 64, "at most 64 PoissonInput objects are supported")
    require(all(not m.contained_objects for m in monitors) and
            all(not s.contained_objects for s in spikes),
            "custom monitor children are unsupported")

    monitor_by_group = {group: [] for group in groups
                        if type(group) in (NeuronGroup, SpatialNeuron,
                                           PoissonGroup)}
    monitor_by_synapse = {synapse: [] for synapse in synapses}
    rate_by_group = {group: [] for group in neuron_groups}
    event_monitor_by_group = {group: [] for group in neuron_groups}
    spike_by_group = {group: [] for group in groups}
    for monitor in monitors:
        if (type(monitor.source) in (NeuronGroup, SpatialNeuron, PoissonGroup) or
                isinstance(monitor.source, Subgroup)):
            source, _, _ = _endpoint(monitor.source, groups)
            require(source in monitor_by_group and
                    (not isinstance(monitor.source, Subgroup) or
                     type(source) in (NeuronGroup, SpatialNeuron)),
                    "StateMonitor source must be a NeuronGroup, PoissonGroup, "
                    "or NeuronGroup subgroup in the network")
            monitor_by_group[source].append(monitor)
        else:
            require(type(monitor.source) is Synapses and
                    monitor.source in monitor_by_synapse,
                    "StateMonitor source must be a NeuronGroup or Synapses in the network")
            monitor_by_synapse[monitor.source].append(monitor)
    for monitor in rates:
        source, _, _ = _endpoint(monitor.source, groups)
        require(type(source) in (NeuronGroup, SpatialNeuron) and
                source in rate_by_group,
                "PopulationRateMonitor source must be a NeuronGroup or subgroup")
        rate_by_group[source].append(monitor)
    for monitor in event_monitors:
        require(type(monitor.source) in (NeuronGroup, SpatialNeuron) and
                monitor.source in event_monitor_by_group,
                "EventMonitor source must be a NeuronGroup in the network")
        event_monitor_by_group[monitor.source].append(monitor)
    for spike in spikes:
        source, _, _ = _endpoint(spike.source, groups)
        require(source in spike_by_group,
                "SpikeMonitor source must be a population or subgroup")
        spike_by_group[source].append(spike)
    inputs_by_group = {group: [] for group in neuron_groups}
    for poisson_input in poisson_inputs:
        target, _, _ = _endpoint(poisson_input._group, groups)
        require(target in inputs_by_group,
                "PoissonInput must target a NeuronGroup or subgroup in the network")
        inputs_by_group[target].append(poisson_input)

    offsets, neuron_count = {}, 0
    random_streams = RandomStreams()
    function_contracts = {}
    population_defs, population_instances, contexts = [], [], {}
    for group in groups:
        offsets[group] = neuron_count
        if type(group) in (NeuronGroup, SpatialNeuron):
            definition, instance, context = _population(
                group, groups, group_indices, monitor_by_group[group],
                event_monitor_by_group[group],
                spike_by_group[group],
                rate_by_group[group],
                inputs_by_group[group], start, duration, namespace, neuron_count,
                random_streams, clock_ids, run_clocks, function_contracts,
                recording_window_steps)
        elif type(group) is PoissonGroup:
            definition, instance, context = _poisson_population(
                group, monitor_by_group[group],
                spike_by_group[group][0] if spike_by_group[group] else None,
                duration, namespace,
                neuron_count, random_streams, clock_ids, run_clocks,
                function_contracts,
                recording_window_steps)
        else:
            definition, instance, context = _spike_generator_population(
                group, spike_by_group[group][0] if spike_by_group[group] else None,
                start, duration, neuron_count,
                clock_ids, run_clocks, recording_window_steps)
        population_defs.append(definition)
        population_instances.append(instance)
        contexts[group] = context
        neuron_count += len(group)
    require(1 <= neuron_count <= neuron_budget(), "neuron preparation budget exceeded")
    require({group for group, monitors_for_group in spike_by_group.items()
             if monitors_for_group} <= {group for group in groups
                                        if contexts[group]["has_spikes"]},
            "SpikeMonitor requires a spiking population")

    synapse_definitions, synapse_instances = [], []
    synapse_ticks = 0
    active_device = get_device()
    synapse_indices = {item: index for index, item in enumerate(synapses)}

    def mutable_synapse_states(endpoint):
        """Return the stored edge-domain states exposed by a Synapses endpoint."""
        endpoint_equations = {
            equation.varname: equation
            for equation in endpoint.equations.values()
        }
        if endpoint.event_driven is not None:
            endpoint_equations.update({
                equation.varname: equation
                for equation in endpoint.event_driven.values()
            })
        names = []
        for equation in sorted(endpoint_equations.values(),
                               key=lambda item: item.varname):
            variable = endpoint.variables[equation.varname]
            if (equation.type == "differential equation" or
                    (equation.type == "parameter" and
                     not equation.flags and not variable.scalar)):
                names.append(equation.varname)
        if endpoint.event_driven is not None:
            names.append("lastupdate")
        return names

    for synapse in synapses:
        synapse_monitors = sorted(
            monitor_by_synapse[synapse], key=lambda monitor: monitor.name)
        procedural_topology = getattr(
            active_device, "procedural_synapse_topology", lambda _synapse: None
        )(synapse)
        procedural_initializers = (
            {} if procedural_topology is None else
            procedural_topology.pop("initializers"))
        procedural_delay = (
            None if procedural_topology is None else
            procedural_topology.pop("delay_initializer"))
        edge_count = (procedural_topology["edge_count"]
                      if procedural_topology is not None else len(synapse))
        source_owner = synapse.source
        source_synapse = None
        if type(source_owner) is Synapses:
            require(source_owner in synapse_indices and
                    type(source_owner.source) is not Synapses and
                    type(source_owner.target) is not Synapses,
                    "Synapses source must be a first-level edge domain in the network")
            source, _, _ = _endpoint(source_owner.source, groups)
            source_start, source_stop = 0, len(source_owner)
            source_synapse = synapse_indices[source_owner]
            source_context = {
                "states": mutable_synapse_states(source_owner),
                "frozen": [],
                "subexpressions": {},
                "dt": float(source_owner.clock.dt / second),
                "events": set(),
            }
        else:
            source, source_start, source_stop = _endpoint(source_owner, groups)
            source_owner = source
            source_context = contexts[source]
        target_owner = synapse.target
        target_synapse = None
        if type(target_owner) is Synapses:
            require(target_owner in synapse_indices and
                    type(target_owner.source) is not Synapses and
                    type(target_owner.target) is not Synapses,
                    "Synapses target must be a first-level edge domain in the network")
            target, _, _ = _endpoint(target_owner.source, groups)
            target_start, target_stop = 0, len(target_owner)
            target_synapse = synapse_indices[target_owner]
            target_context = {
                "states": mutable_synapse_states(target_owner),
                "frozen": [],
                "subexpressions": {},
                "dt": float(target_owner.clock.dt / second),
                "events": set(),
            }
        else:
            target, target_start, target_stop = _endpoint(target_owner, groups)
            target_owner = target
            target_context = contexts[target]
        require(type(target) in (NeuronGroup, SpatialNeuron,
                                 PoissonGroup, SpikeGeneratorGroup),
                "Synapses target must resolve to a population in the network")
        source_states = source_context["states"]
        target_states = target_context["states"]
        source_endpoint_parameters = [
            name for name in source_context.get("parameters", ())
            if f"{name}_pre" in synapse.variables]
        target_endpoint_parameters = [
            name for name in target_context.get("parameters", ())
            if f"{name}_post" in synapse.variables]
        pre_aliases = {f"{name}_pre": name for name in source_states}
        post_aliases = {name: name for name in target_states}
        post_aliases.update({f"{name}_post": name for name in target_states})
        require(len(set(pre_aliases) | set(post_aliases)) ==
                len(pre_aliases) + len(post_aliases),
                "ODE names create colliding pre/post aliases")

        endpoint_subexpression_replacements = {}
        source_replacements = {
            **{name: f"{name}_pre" for name in source_states},
            **{name: f"{name}_pre" for name in source_endpoint_parameters},
            "i": "i", "N": "N_pre",
        }
        target_replacements = {
            **{name: f"{name}_post" for name in target_states},
            **{name: f"{name}_post" for name in target_endpoint_parameters},
            "i": "j", "N": "N_post",
        }
        synapse_equation_names = set(synapse.equations.names)
        for name, expression in source_context.get("subexpressions", {}).items():
            endpoint_subexpression_replacements[f"{name}_pre"] = (
                f"({word_substitute(expression, source_replacements)})")
        for name, expression in target_context.get("subexpressions", {}).items():
            lowered = f"({word_substitute(expression, target_replacements)})"
            endpoint_subexpression_replacements[f"{name}_post"] = lowered
            if name not in synapse_equation_names:
                endpoint_subexpression_replacements[name] = lowered

        def inline_endpoint_subexpressions(code):
            return word_substitute(code, endpoint_subexpression_replacements)

        require(synapse._connect_called and
                (procedural_topology is not None or
                 edge_count <= explicit_synapse_budget()),
                "Synapses.connect must be called; explicit topology supports "
                f"at most {explicit_synapse_budget():,} connections")
        if synapse.subexpression_updater is not None:
            require(synapse.subexpression_updater.when == "before_start" and
                    synapse.subexpression_updater.order == synapse.order and
                    synapse.subexpression_updater.clock is source.clock,
                    "synaptic (constant over dt) subexpressions require Brian2's "
                    "default before_start schedule")
        if synapse.state_updater is not None:
            require(supported_method_choice(
                        synapse.state_updater.method_choice, synapse.equations) and
                    not synapse.state_updater.method_options,
                    "synaptic differential equations require Brian's default method "
                    "selection or explicit 'exact', 'linear', 'independent', "
                    "'euler', 'rk2', 'rk4', 'exponential_euler', 'heun' or "
                    "'milstein'; independent requires ODEs without cross-state "
                    "dependencies")
            require(synapse.state_updater.when == "groups" and
                    synapse.state_updater.order == 0 and synapse.state_updater.clock is source.clock,
                    "synaptic state updater must use the default schedule")
        require(len(synapse._pathways) == len(synapse._synaptic_updaters),
                "invalid Synapses pathway registration")
        pathways = sorted(synapse._pathways,
                          key=lambda pathway: (pathway.order, pathway.name))
        pre_pathways = [pathway for pathway in pathways
                        if pathway.prepost == "pre"]
        post_pathways = [pathway for pathway in pathways
                         if pathway.prepost == "post"]
        require(pre_pathways or post_pathways or synapse.summed_updaters or
                synapse.state_updater is not None,
                "Synapses requires an event pathway, summed variable or state update")
        if pre_pathways:
            require(source_context["events"], "on_pre requires a source event")
        require(all(pathway.event in source_context["events"] and
                    pathway.when == "synapses" and pathway.order == -1 and
                    pathway.clock is source.clock for pathway in pre_pathways),
                "on_pre pathways must reference a source event and use Brian2's "
                "default schedule")
        if post_pathways:
            require(target_context["events"],
                    "on_post requires a target event")
        require(all(pathway.event in target_context["events"] and
                    pathway.when == "synapses" and pathway.order == 1 and
                    pathway.clock is target.clock for pathway in post_pathways),
                "on_post pathways must reference a target event and use Brian2's "
                "default schedule")
        expected_children = set(synapse._pathways)
        if synapse.subexpression_updater is not None:
            expected_children.add(synapse.subexpression_updater)
        if synapse.state_updater is not None:
            expected_children.add(synapse.state_updater)
        expected_children.update(synapse.summed_updaters.values())
        regular_runners = sorted(
            set(synapse.contained_objects) - expected_children,
            key=lambda runner: (runner.when, runner.order, runner.name))
        require(all(type(runner) is CodeRunner and
                    runner.group.id == synapse.id and
                    not runner.contained_objects and runner.codeobj_class is None
                    for runner in regular_runners),
                "custom Synapses contained code must be standard run_regularly")

        synapse_states, clock_driven_states, synapse_initial_state = [], [], {}
        synapse_parameters, synapse_parameter_values = [], {}
        declared_synapse_parameters = []
        synapse_linked_names = []

        def materialize_endpoint_parameters(owner, names, suffix, endpoint_indices):
            if not names:
                return
            require(procedural_topology is None,
                    "endpoint parameter aliases require explicit topology")
            for name in names:
                alias = f"{name}_{suffix}"
                values = np.asarray(owner.variables[name].get_value()).reshape(-1)
                if values.size == 1:
                    selected = np.repeat(values, edge_count)
                else:
                    require(np.all((endpoint_indices >= 0) &
                                   (endpoint_indices < values.size)),
                            f"{alias}: endpoint parameter index out of range")
                    selected = values[endpoint_indices]
                parameter = symbol(alias, synapse.variables[alias], "synapse")
                synapse_parameters.append(parameter)
                synapse_parameter_values[alias] = _array_bits(
                    selected, parameter["dtype"])
                declared_synapse_parameters.append(alias)

        if procedural_topology is None:
            raw_sources = np.asarray(
                synapse.variables["_synaptic_pre"].get_value(), dtype=np.int64)
            raw_targets = np.asarray(
                synapse.variables["_synaptic_post"].get_value(), dtype=np.int64)
        else:
            raw_sources = raw_targets = np.empty(0, dtype=np.int64)
        materialize_endpoint_parameters(
            source_owner, source_endpoint_parameters, "pre", raw_sources)
        materialize_endpoint_parameters(
            target_owner, target_endpoint_parameters, "post", raw_targets)

        def materialize_degree_parameter(name):
            if name in synapse_parameter_values:
                return
            require(procedural_topology is None,
                    f"{name} requires explicit topology")
            indices = raw_targets if name == "N_incoming" else raw_sources
            values = np.asarray(
                synapse.variables[name].get_value()).reshape(-1)
            require(np.all((indices >= 0) & (indices < values.size)),
                    f"{name}: connection index out of range")
            parameter = symbol(name, synapse.variables[name], "synapse")
            synapse_parameters.append(parameter)
            synapse_parameter_values[name] = _array_bits(
                values[indices], parameter["dtype"])
            declared_synapse_parameters.append(name)

        equations = {equation.varname: equation
                     for equation in synapse.equations.values()}
        if synapse.event_driven is not None:
            equations.update({equation.varname: equation
                              for equation in synapse.event_driven.values()})
        for equation in sorted(equations.values(), key=lambda eq: eq.varname):
            var = synapse.variables[equation.varname]
            if procedural_topology is not None:
                per_edge_storage = (
                    equation.type == "differential equation" or
                    (equation.type == "parameter" and
                     not equation.flags and not var.scalar))
                procedural_parameter = (
                    equation.type == "parameter" and not var.scalar and
                    set(equation.flags) == {"constant"} and
                    equation.varname in procedural_initializers)
                require(not per_edge_storage and
                        (equation.varname not in procedural_initializers or
                         procedural_parameter),
                        "procedural fixed-total topology currently supports no "
                        "mutable per-edge state; initialized per-edge variables "
                        "must use the (constant) flag")
            if equation.varname == synapse.multisynaptic_index:
                require(equation.type == "parameter" and not equation.flags and
                        var.dtype == np.int32 and not var.scalar,
                        "multisynaptic_index must use Brian2's canonical int32 layout")
                values = np.asarray(var.get_value()).reshape(-1)
                require(procedural_topology is None and
                        values.shape == (edge_count,) and
                        np.all((values >= 0) & (values <= 2**53)),
                        "multisynaptic_index requires explicit topology")
                declared_synapse_parameters.append(equation.varname)
                synapse_parameters.append(symbol(
                    equation.varname, var, "synapse"))
                synapse_parameter_values[equation.varname] = _array_bits(values, "i32")
                continue
            dtype = np.dtype(var.dtype)
            if equation.type == "differential equation":
                require(dtype in {np.dtype(np.float32), np.dtype(np.float64)},
                        "synaptic ODE states must use float32 or float64")
            else:
                require(dtype in {np.dtype(np.float32), np.dtype(np.float64),
                                  np.dtype(np.int32), np.dtype(np.int64),
                                  np.dtype(np.uint32), np.dtype(np.uint64),
                                  np.dtype(np.bool_)},
                        "synaptic variables must use f32/f64/i32/i64/u32/u64/bool")
            if set(equation.flags) == {"linked"}:
                require(equation.type == "parameter" and
                        isinstance(var, Subexpression),
                        "synaptic linked variables must resolve to a deterministic "
                        "linked expression")
                synapse_linked_names.append(equation.varname)
                continue
            if equation.type == "subexpression":
                flags = set(equation.flags)
                require(flags in (set(), {"shared"}) and
                        (not flags or var.scalar),
                        "synaptic subexpressions support only a scalar (shared) flag")
                continue
            values = np.asarray(var.get_value()).reshape(-1)
            if equation.type == "differential equation":
                require(set(equation.flags) in ({"clock-driven"}, {"event-driven"}),
                        "synaptic ODEs require a clock-driven or event-driven flag")
                require(not var.scalar, "synaptic ODE state must be per-synapse")
                synapse_states.append(equation.varname)
                if set(equation.flags) == {"clock-driven"}:
                    clock_driven_states.append(equation.varname)
                expected, destination = edge_count, synapse_initial_state
            else:
                require(equation.type == "parameter" and
                        set(equation.flags) <= {"constant", "shared"},
                        "synaptic parameters support only constant/shared flags")
                if not equation.flags and not var.scalar:
                    synapse_states.append(equation.varname)
                    expected, destination = len(synapse), synapse_initial_state
                else:
                    declared_synapse_parameters.append(equation.varname)
                    synapse_parameters.append(symbol(
                        equation.varname, var, "scalar" if var.scalar else "synapse"))
                    expected = 1 if var.scalar else edge_count
                    destination = synapse_parameter_values
                    if (procedural_topology is not None and
                            equation.varname in procedural_initializers):
                        destination[equation.varname] = []
                        continue
            require(values.size == expected and values.dtype.kind in "fibu",
                    f"{equation.varname}: invalid synaptic variable shape")
            destination[equation.varname] = _array_bits(
                values, dtype_name(var.dtype))
        if synapse.event_driven is not None:
            lastupdate = synapse.variables["lastupdate"]
            values = np.asarray(lastupdate.get_value()).reshape(-1)
            require(procedural_topology is None and
                    values.shape == (edge_count,) and values.dtype == np.float64,
                    "invalid event-driven lastupdate state")
            synapse_states.append("lastupdate")
            synapse_initial_state["lastupdate"] = _array_bits(values, "f64")

        linked_expressions = {
            name: str(synapse.variables[name].expr)
            for name in synapse_linked_names
        }
        synapse_linked_variables = []
        linked_input_names = set()
        linked_resolved_variables = {}
        for linked_name, expression in linked_expressions.items():
            linked_variable = synapse.variables[linked_name]
            remote_count = 0
            expression_source = None
            for input_name in sorted(get_identifiers(expression)):
                if input_name not in synapse.variables:
                    continue
                input_variable = synapse.variables[input_name]
                source_group = next(
                    (candidate for candidate in groups
                     if candidate.id == getattr(input_variable.owner, "id", None)),
                    None)
                if source_group is None:
                    continue
                source_name = input_variable.name
                require(type(source_group) in (NeuronGroup, SpatialNeuron) and
                        source_name in contexts[source_group]["states"],
                        f"{linked_name}: linked expression sources must be mutable "
                        "NeuronGroup state")
                index_name = synapse.variables.indices[input_name]
                require(isinstance(index_name, (int, np.integer)) or
                        (isinstance(index_name, str) and
                         index_name.removeprefix("-").isdigit()),
                        f"{linked_name}: Synapses linked mapping must use a fixed "
                        "source index")
                source_index = int(index_name)
                require(0 <= source_index < len(source_group),
                        f"{linked_name}: linked source index is out of bounds")
                definition = {
                    "name": input_name,
                    "dtype": dtype_name(input_variable.dtype),
                    "dimensions": [float(dim) for dim in input_variable.dim._dims],
                    "source_population": groups.index(source_group),
                    "source_state": source_name,
                    "index": {"kind": "constant",
                              "values": [source_index] * edge_count},
                }
                previous = next(
                    (item for item in synapse_linked_variables
                     if item["name"] == input_name), None)
                require(previous is None or previous == definition,
                        f"{input_name}: inconsistent linked source")
                if previous is None:
                    synapse_linked_variables.append(definition)
                linked_input_names.add(input_name)
                require(expression_source is None or expression_source is source_group,
                        f"{linked_name}: linked expression must use one source group")
                expression_source = source_group
                remote_count += 1
            require(remote_count > 0,
                    f"{linked_name}: linked expression needs a population state")
            external = get_identifiers(expression) - set(synapse.variables)
            linked_resolved_variables.update(
                expression_source.resolve_all(external, namespace,
                                              user_identifiers=external))

        state_monitor_defs = []
        for monitor in synapse_monitors:
            outputs = list(monitor.record_variables)
            variables = []
            for name in outputs:
                if name in linked_expressions:
                    for dependency in sorted(
                            get_identifiers(linked_expressions[name]) &
                            linked_input_names):
                        if dependency not in variables:
                            variables.append(dependency)
                elif name not in variables:
                    variables.append(name)
            sources = []
            for name in variables:
                variable = synapse.variables[name]
                if name in synapse_states:
                    kind, state_name = "synapse_state", name
                elif name in linked_input_names:
                    kind, state_name = "linked", name
                else:
                    owner_id = getattr(variable.owner, "id", None)
                    if (owner_id == getattr(source_owner, "id", None) and
                            variable.name in source_context["states"]):
                        kind, state_name = "pre_state", variable.name
                    elif (owner_id == getattr(target_owner, "id", None) and
                          variable.name in target_context["states"]):
                        kind, state_name = "post_state", variable.name
                    else:
                        require(False,
                                f"{name}: Synapses StateMonitor variable must be "
                                "a mutable synaptic state or pre/post neuron state")
                sources.append({
                    "kind": kind, "name": state_name,
                    "dtype": dtype_name(variable.dtype),
                })
            require(variables and len(set(variables)) == len(variables),
                    "Synapses StateMonitor needs unique recorded variables")
            require(len(outputs) == len(set(outputs)),
                    "Synapses StateMonitor needs unique output variables")
            record = [int(index) for index in monitor.record]
            require(all(0 <= index < edge_count for index in record),
                    "Synapses StateMonitor record index out of range")
            state_monitor_defs.append({
                "name": monitor.name, "variables": variables,
                "output_variables": outputs, "record": record,
                "sources": sources,
                "clock": clock_ids[monitor.clock], "when": monitor.when,
                "order": monitor.order,
            })
        require(len(synapse_states) <= 32,
                "at most 32 mutable synaptic states including lastupdate")
        if procedural_topology is not None:
            parameter_names = {
                parameter["name"] for parameter in synapse_parameters
                if parameter["index_domain"] == "synapse"}
            if procedural_topology["kind"] == "binary_csr":
                require(all(p["dtype"] == "f64" for p in synapse_parameters
                            if p["index_domain"] == "synapse"),
                        "binary CSR columns require float64 constant parameters")
            require(not synapse_states and
                    parameter_names == set(procedural_initializers),
                    "procedural fixed-total initializers must cover exactly all "
                    "per-edge constant parameters")
            require(synapse.subexpression_updater is None and
                    synapse.state_updater is None and
                    not post_pathways and not synapse.summed_updaters and
                    (procedural_delay is None or len(pathways) == 1),
                    "procedural fixed-total topology currently supports "
                    "on_pre pathways and one optional delay initializer")

        substituted = dict(synapse.equations.get_substituted_expressions(
            variables=dict(synapse.variables), include_subexpressions=True))
        subexpressions = {
            name: expr.code for name, expr in substituted.items()
            if name in equations and equations[name].type == "subexpression"
        }
        subexpressions.update(linked_expressions)

        def materialize_subexpressions(code, variables):
            """Precompute only subexpressions directly used by this CodeRunner."""
            used = get_identifiers(code) & set(subexpressions)
            if not used:
                return code, used
            for name in used:
                original = synapse.variables[name]
                variables[name] = AuxiliaryVariable(
                    name, dimensions=original.dim, dtype=original.dtype)
            assignments = [f"{name} = {subexpressions[name]}"
                           for name in sorted(used)]
            return "\n".join([*assignments, code]), used

        def restore_subexpression_temporaries(variables, used):
            for name in used:
                original = synapse.variables[name]
                variables[name] = AuxiliaryVariable(
                    name, dimensions=original.dim, dtype=original.dtype)

        def add_synapse_constant(name, var, context):
            if name in synapse_parameter_values:
                return
            if _supported_function(name, var, function_contracts):
                return
            require(not isinstance(var, Function) and var.constant and var.scalar,
                    f"{name}: {context} supports built-in deterministic math functions, "
                    "synaptic variables and scalar constants only")
            values = np.asarray(var.get_value()).reshape(-1)
            require(values.size == 1 and values.dtype.kind in "fibu",
                    f"{name}: invalid synaptic scalar constant")
            parameter = symbol(name, var, "scalar")
            synapse_parameters.append(parameter)
            synapse_parameter_values[name] = _array_bits(
                values, parameter["dtype"])

        for name, variable in sorted(linked_resolved_variables.items()):
            add_synapse_constant(name, variable, "synaptic linked expression")

        synapse_base_inputs = (set(synapse_states) | set(declared_synapse_parameters) |
                               linked_input_names |
                               set(linked_resolved_variables) |
                               set(pre_aliases) | set(post_aliases) |
                               {"dt", "t", "i", "j", "N", "N_pre", "N_post"})
        synapse_specs = []

        def normalize_local_indices(variables):
            """Keep AtlasIR i/j local while discarding Brian subgroup offset helpers."""
            dependencies = set()
            for index_name in ("i", "j"):
                variable = variables.get(index_name)
                if getattr(variable, "expr", None) is not None:
                    dependencies |= set(variable.identifiers)
                variables[index_name] = AuxiliaryVariable(index_name, dtype=np.float64)
            return dependencies

        if synapse.subexpression_updater is not None:
            runner = synapse.subexpression_updater
            runner.update_abstract_code(namespace)
            variables = dict(synapse.variables)
            variables.update(linked_resolved_variables)
            _, used, unknown = analyse_identifiers(
                runner.abstract_code, variables, recursive=True)
            resolved = synapse.resolve_all(
                (used | unknown) - set(linked_resolved_variables) -
                linked_input_names, namespace)
            variables.update(resolved)
            random_functions = _random_function_configs(variables)
            check_units_statements(runner.abstract_code, variables)
            index_dependencies = normalize_local_indices(variables)
            for name in sorted(set(resolved) - synapse_base_inputs -
                               index_dependencies):
                if name in random_functions:
                    continue
                if _supported_function(name, variables[name], function_contracts):
                    continue
                add_synapse_constant(
                    name, variables[name], "synaptic constant-over-dt update")
                synapse_base_inputs.add(name)
            spec = CodeObjectSpec.from_abstract(
                runner, "synapse_subexpression_update", runner.abstract_code,
                variables, synapse_base_inputs, set(synapse_states),
                iteration_domain="all_synapses",
                allocate_random_stream=random_streams.allocate,
                random_functions=random_functions,
                functions=function_contracts).to_dict()
            spec["clock"] = clock_ids[runner.clock]
            synapse_specs.append(spec)

        if synapse.state_updater is not None:
            runner = synapse.state_updater
            runner.update_abstract_code(namespace)
            stochastic_names = set(synapse.equations.stochastic_variables)
            variables = dict(synapse.variables)
            variables.update(linked_resolved_variables)
            code, used_subexpressions = materialize_subexpressions(
                inline_endpoint_subexpressions(runner.abstract_code), variables)
            _, used, unknown = analyse_identifiers(code, variables, recursive=True)
            resolved = synapse.resolve_all(
                (used | unknown) - used_subexpressions -
                set(linked_resolved_variables) - linked_input_names, namespace)
            variables.update(resolved)
            for name in stochastic_names & set(variables):
                variables[name] = AuxiliaryVariable(
                    name, dimensions=(second**0.5).dim,
                    dtype=synapse.variables[name].dtype)
            random_functions = _random_function_configs(variables)
            restore_subexpression_temporaries(variables, used_subexpressions)
            check_units_statements(code, variables)
            index_dependencies = normalize_local_indices(variables)
            for name in sorted(set(resolved) - synapse_base_inputs -
                               index_dependencies - used_subexpressions -
                               stochastic_names):
                if name in random_functions:
                    continue
                if _supported_function(name, variables[name], function_contracts):
                    continue
                add_synapse_constant(name, variables[name], "synaptic state update")
                synapse_base_inputs.add(name)
            spec = CodeObjectSpec.from_abstract(
                runner, "synapse_state_update", code, variables,
                synapse_base_inputs, set(clock_driven_states),
                iteration_domain="all_synapses",
                allocate_random_stream=random_streams.allocate,
                random_functions=random_functions,
                functions=function_contracts).to_dict()
            spec["clock"] = clock_ids[runner.clock]
            synapse_specs.append(spec)

        require(not regular_runners or synapse.clock is source.clock,
                "synaptic run_regularly must share its owner clock with the source")
        for runner in regular_runners:
            runner.update_abstract_code(namespace)
            variables = dict(synapse.variables)
            variables.update(linked_resolved_variables)
            code, used_subexpressions = materialize_subexpressions(
                inline_endpoint_subexpressions(runner.abstract_code), variables)
            _, used, unknown = analyse_identifiers(code, variables, recursive=True)
            variables.update(synapse.resolve_all(
                (used | unknown) - used_subexpressions -
                set(linked_resolved_variables) - linked_input_names, namespace))
            random_functions = _random_function_configs(variables)
            restore_subexpression_temporaries(variables, used_subexpressions)
            check_units_statements(code, variables)
            index_dependencies = normalize_local_indices(variables)
            allowed = synapse_base_inputs | set(pre_aliases) | set(post_aliases)
            for name in sorted((used | unknown) - allowed -
                               index_dependencies - used_subexpressions):
                if name in random_functions:
                    continue
                if _supported_function(name, variables[name], function_contracts):
                    continue
                add_synapse_constant(name, variables[name], "synaptic run_regularly")
                allowed.add(name)
            spec = CodeObjectSpec.from_abstract(
                runner, "synapse_run_regularly", code, variables, allowed,
                set(synapse_states), iteration_domain="all_synapses",
                allocate_random_stream=random_streams.allocate,
                random_functions=random_functions,
                functions=function_contracts).to_dict()
            spec["clock"] = clock_ids[runner.clock]
            synapse_specs.append(spec)

        def lower_pathway(pathway, kind):
            pathway.update_abstract_code(namespace)
            path_variables = dict(pathway.variables)
            path_variables.update({name: synapse.variables[name]
                                   for name in linked_input_names})
            path_variables.update(linked_resolved_variables)
            pathway_code = inline_endpoint_subexpressions(pathway.abstract_code)
            pathway_identifiers = get_identifiers(pathway_code)
            # Brian exposes an unqualified postsynaptic state name as well as
            # its explicit ``_post`` alias.  Canonicalise code that happens to
            # use both spellings so AtlasIR has one storage alias per state.
            duplicate_post_aliases = {
                state: f"{state}_post" for state in target_states
                if state in pathway_identifiers and
                f"{state}_post" in pathway_identifiers}
            if duplicate_post_aliases:
                pathway_code = word_substitute(
                    pathway_code, duplicate_post_aliases)
            pathway_code, used_subexpressions = materialize_subexpressions(
                pathway_code, path_variables)
            _, used, unknown = analyse_identifiers(
                pathway_code, path_variables, recursive=True)
            resolved_path = pathway.resolve_all(
                (used | unknown) - used_subexpressions -
                set(linked_resolved_variables) - linked_input_names, namespace)
            path_variables.update(resolved_path)
            random_functions = _random_function_configs(path_variables)
            restore_subexpression_temporaries(
                path_variables, used_subexpressions)
            check_units_statements(pathway_code, path_variables)
            index_dependencies = normalize_local_indices(path_variables)
            allowed = (set(declared_synapse_parameters) | set(synapse_states) |
                       linked_input_names |
                       set(linked_resolved_variables) |
                       {"dt", "t", "i", "j", "N", "N_pre", "N_post"} |
                       set(pre_aliases) | set(post_aliases))
            if (target_synapse is None and
                    population_defs[groups.index(target)]["refractory"] is not None):
                allowed.add("not_refractory_post")
            for name in sorted(set(resolved_path) - allowed - index_dependencies -
                               used_subexpressions):
                if name in random_functions:
                    continue
                if _supported_function(name, path_variables[name], function_contracts):
                    continue
                add_synapse_constant(name, path_variables[name], kind)
                allowed.add(name)
            writable = set(synapse_states)
            conditional_writes = {}
            if kind in {"synapses", "synapses_post"}:
                writable |= set(post_aliases)
                if kind == "synapses":
                    writable |= {alias for alias, state in pre_aliases.items()
                                 if state not in source_context["frozen"]}
                conditional_writes = {
                    alias: "not_refractory_post"
                    for alias, state in post_aliases.items()
                    if state in target_context["frozen"]}
            spec = CodeObjectSpec.from_abstract(
                pathway, kind, pathway_code, path_variables, allowed, writable,
                iteration_domain="active_synapses",
                conditional_writes=conditional_writes,
                allocate_random_stream=random_streams.allocate,
                random_functions=random_functions,
                functions=function_contracts).to_dict()
            spec["pathway_name"] = pathway.name
            spec["event_name"] = pathway.event
            spec["clock"] = clock_ids[pathway.clock]
            synapse_specs.append(spec)

        for pathway in pre_pathways:
            lower_pathway(pathway, "synapses")
        for pathway in post_pathways:
            lower_pathway(pathway, "synapses_post")

        summed_specs = []
        for target_name, updater in sorted(synapse.summed_updaters.items()):
            if target_name.endswith("_pre"):
                endpoint, target_state = "pre", target_name[:-4]
                summed_context, summed_group = source_context, source_owner
                summed_endpoint = synapse.source
            else:
                require(target_name.endswith("_post"),
                        "summed variable target must end in _pre or _post")
                endpoint, target_state = "post", target_name[:-5]
                summed_context, summed_group = target_context, target_owner
                summed_endpoint = synapse.target
            require(target_state in summed_context["states"],
                    "summed variable must target mutable f64 population state")
            require(updater.when in {"groups", "after_groups"} and
                    updater.order == summed_endpoint.order - 1,
                    "summed variables require groups/after_groups and Brian2's "
                    "default endpoint-relative order")
            updater.update_abstract_code(namespace)
            summed_variables = dict(synapse.variables)
            summed_variables.update(linked_resolved_variables)
            target_variable = summed_group.variables[target_state]
            summed_variables["_synaptic_var"] = AuxiliaryVariable(
                "_synaptic_var",
                dimensions=target_variable.dim,
                dtype=target_variable.dtype,
            )
            summed_code, used_subexpressions = materialize_subexpressions(
                inline_endpoint_subexpressions(updater.abstract_code),
                summed_variables)
            _, used, unknown = analyse_identifiers(
                summed_code, summed_variables, recursive=True)
            summed_resolved = synapse.resolve_all(
                (used | unknown) - used_subexpressions -
                set(linked_resolved_variables) - linked_input_names,
                namespace,
                additional_variables={
                    "_synaptic_var": summed_variables["_synaptic_var"]
                },
            )
            summed_variables.update(summed_resolved)
            for degree_name in ({"N_incoming", "N_outgoing"} &
                                set(summed_resolved)):
                materialize_degree_parameter(degree_name)
            random_functions = _random_function_configs(summed_variables)
            restore_subexpression_temporaries(
                summed_variables, used_subexpressions)
            check_units_statements(summed_code, summed_variables)
            index_dependencies = normalize_local_indices(summed_variables)
            summed_allowed = (
                set(declared_synapse_parameters) | set(synapse_states) |
                linked_input_names |
                set(linked_resolved_variables) |
                {"dt", "t", "i", "j", "N", "N_pre", "N_post"} |
                set(pre_aliases) | set(post_aliases))
            for name in sorted(set(summed_resolved) - summed_allowed -
                               index_dependencies - used_subexpressions -
                               {"_synaptic_var"}):
                if name in random_functions:
                    continue
                if _supported_function(name, summed_variables[name], function_contracts):
                    continue
                add_synapse_constant(name, summed_variables[name], "summed variable")
                summed_allowed.add(name)
            spec = CodeObjectSpec.from_abstract(
                updater, "summed_variable", summed_code, summed_variables,
                summed_allowed, set(), iteration_domain="all_synapses",
                allocate_random_stream=random_streams.allocate,
                random_functions=random_functions,
                functions=function_contracts).to_dict()
            spec["summed_target"] = endpoint
            spec["summed_state"] = target_state
            spec["clock"] = clock_ids[updater.clock]
            summed_specs.append(spec)
        # Full endpoints use groups/order=-1; subgroup endpoints inherit
        # subgroup.order-1 (normally 0). Preserve their stable order metadata
        # so both executors can place the update on the correct side of state.
        constant_over_dt_specs = [
            code for code in synapse_specs
            if code["kind"] == "synapse_subexpression_update"]
        remaining_specs = [
            code for code in synapse_specs
            if code["kind"] != "synapse_subexpression_update"]
        synapse_specs = (constant_over_dt_specs +
                         sorted(summed_specs,
                                key=lambda code: (code["order"], code["name"])) +
                         remaining_specs)
        require(len(synapse_parameters) <= 128, "at most 128 synaptic parameters")
        if procedural_topology is None:
            sources = (np.asarray(
                synapse.variables["_synaptic_pre"].get_value(), dtype=np.int64)
                - source_start)
            targets = (np.asarray(
                synapse.variables["_synaptic_post"].get_value(), dtype=np.int64)
                - target_start)
            require(sources.shape == targets.shape == (edge_count,),
                    "invalid Synapses topology")
            require(np.all((sources >= 0) &
                           (sources < source_stop - source_start)) and
                    np.all((targets >= 0) &
                           (targets < target_stop - target_start)),
                    "Synapses topology index out of range")
            topology = {"kind": "explicit"}
        else:
            sources = targets = np.empty(0, dtype=np.int64)
            topology = procedural_topology
            topology["initializers"] = (procedural_initializers if
                topology["kind"] == "binary_csr" else {
                name: _procedural_initializer(
                    initializer, synapse.variables[name].dim,
                    2 + 2 * position, f"{synapse.name}.{name}")
                for position, (name, initializer) in enumerate(
                    sorted(procedural_initializers.items()))
            })
        pathway_instances = []
        for pathway_position, pathway in enumerate(pathways):
            delay_descriptor = None
            if procedural_delay is not None:
                delay_descriptor = _procedural_initializer(
                    procedural_delay, second,
                    2 + 2 * len(procedural_initializers) +
                    2 * pathway_position,
                    f"{synapse.name}.{pathway.name}.delay")
                delay_values = np.empty(0, dtype=np.float64)
            else:
                delay_values = np.asarray(
                    pathway._delays.get_value(), dtype=np.float64).reshape(-1)
            if (procedural_topology is not None and
                    delay_descriptor is None and delay_values.size == 0):
                # Brian2 represents the implicit default zero delay as an
                # unallocated per-edge DynamicArray. Preserve its semantics
                # without materializing edge_count zeros in Python.
                delay_values = np.zeros(1, dtype=np.float64)
            require(delay_descriptor is not None or
                    (delay_values.size in (1, edge_count) and
                     np.isfinite(delay_values).all() and
                     np.all(delay_values >= 0)),
                    "pathway delays must be finite and non-negative")
            if (delay_values.size == edge_count and delay_values.size and
                    np.all(delay_values == delay_values[0])):
                delay_values = delay_values[:1]
            pathway_dt = (source_context["dt"] if pathway.prepost == "pre"
                          else target_context["dt"])
            delay_ratios = delay_values / pathway_dt + 0.5
            require(delay_descriptor is not None or
                    (np.isfinite(delay_ratios).all() and
                     np.all(delay_ratios < 1_000_001)),
                    "pathway delay exceeds 1,000,000 clock ticks")
            pathway_instances.append({
                "name": pathway.name,
                "kind": pathway.prepost,
                "event": pathway.event,
                "delay": _array_bits(delay_values, "f64"),
                "delay_ticks": index_array(np.floor(delay_ratios).astype(np.int64)),
                "delay_initializer": delay_descriptor,
                "pending": [],
            })
        if any(code["kind"] in {
                "synapse_subexpression_update", "synapse_state_update",
                "summed_variable", "synapse_run_regularly"} for code in synapse_specs):
            synapse_ticks += (
                population_defs[groups.index(source)]["steps"] * edge_count)
        require(synapse_ticks <= MAX_SYNAPSE_TICKS,
                f"probe budget: <={MAX_SYNAPSE_TICKS:,} total synapse-ticks")
        synapse_definition = {
            "name": synapse.name,
            "states": [symbol(name, synapse.variables[name], "synapse")
                       for name in synapse_states],
            "clock_driven_states": clock_driven_states,
            "parameters": synapse_parameters,
            "linked_variables": synapse_linked_variables,
            "pre_state_aliases": pre_aliases, "post_state_aliases": post_aliases,
            "source_population": groups.index(source),
            "target_population": groups.index(target),
            "source_synapse": source_synapse,
            "target_synapse": target_synapse,
            "source_start": source_start, "target_start": target_start,
            "source_count": source_stop - source_start,
            "target_count": target_stop - target_start,
            "code_objects": synapse_specs,
            "state_monitors": state_monitor_defs,
            "monitor_expressions": linked_expressions,
        }
        synapse_instance = {
            "source": index_array(sources), "target": index_array(targets),
            "topology": topology,
            "initial_state": synapse_initial_state,
            "parameters": synapse_parameter_values,
            "pathways": pathway_instances,
        }
        synapse_definitions.append(synapse_definition)
        synapse_instances.append(synapse_instance)

    checked = [obj for obj in objects if type(obj) is not NetworkOperation]
    checked += [runner for group in groups
                         for runner in group.contained_objects]
    for synapse in synapses:
        checked.extend(synapse.contained_objects)
    for obj in checked:
        object_time = float(obj.clock.t / second)
        # A Clock points at its first tick at or after the Network's current
        # time. For a partial interval this can be later than network.t (e.g.
        # a 50 ms Clock after run(10*ms) points at 50 ms). Compare exact tick
        # identity and retain a check that t/timestep agree.
        object_tick = int(obj.clock.variables["timestep"].get_value().item())
        expected_tick = run_clocks[clock_ids[obj.clock]]["start_tick"]
        canonical_time = object_tick * float(obj.clock.dt / second)
        require(obj.active and object_tick == expected_tick and
                object_time == canonical_time,
                "all objects must be active at the next scheduled Clock tick")
        require(getattr(obj, "codeobj_class", None) is None,
                "custom codeobj_class is unsupported")

    total_values = sum(context["value_count"] for context in contexts.values())
    for synapse, synapse_instance in zip(synapses, synapse_instances, strict=True):
        pathway_delay_values = sum(
            2 * len(pathway["delay"])
            for pathway in synapse_instance["pathways"])
        explicit_topology_values = (2 * len(synapse_instance["source"]))
        total_values += (explicit_topology_values + pathway_delay_values +
                         sum(len(v) for v in synapse_instance["initial_state"].values()) +
                         sum(len(v) for v in synapse_instance["parameters"].values()))
    require(total_values <= initial_value_budget(),
            f"probe array budget: <={initial_value_budget():,} values including topology")
    definition = {
        "populations": population_defs, "synapses": synapse_definitions,
        "clocks": clock_definitions,
        "functions": [function_contracts[name]
                      for name in sorted(function_contracts)],
        "numeric_profile": "reference-f64",
        "rng_algorithm": "splitmix64-counter-v1",
    }
    instance = {
        "neuron_count": neuron_count, "populations": population_instances,
        "synapses": synapse_instances, "rng_seed": int(rng_seed),
    }
    definition["schedule"] = build_schedule(
        definition, instance, network.schedule)
    if recording_window_steps is not None:
        lags = pathway_sample_lags(definition)
        for q, (syn, inst) in enumerate(zip(synapse_definitions, synapse_instances, strict=True)):
            if inst.get("topology", {"kind": "explicit"})["kind"] != "explicit":
                continue
            for path in inst["pathways"]:
                side = "source" if path["kind"] == "pre" else "target"
                endpoint = population_defs[syn[side+"_population"]]
                window = endpoint["monitor"]["window_steps"]
                if window < endpoint["steps"]:
                    required = max(path["delay_ticks"], default=0) + lags.get((q,path["name"]),0)
                    require(required <= window,
                            "recording_window_steps must cover the maximum delay plus pathway sample lag "
                            "so cross-run pending events remain reconstructable")
    return attach_protocol({
        "schema": SCHEMA, "definition": definition, "instance": instance,
        "run": {"start": bits(start_seconds), "duration": bits(duration_seconds),
                "clocks": run_clocks},
    })


def lower_network(network, duration, namespace=None, rng_seed=0,
                  recording_window_steps=None,
                  _network_operations_prevalidated=False):
    """Return AtlasIR or raise one structured model-wide capability error."""
    report = collect_network_issues(
        network, duration,
        network_operations_prevalidated=_network_operations_prevalidated)
    if not report.supported:
        raise CapabilityError(report)
    try:
        return _lower_network(network, duration, namespace, rng_seed,
                              recording_window_steps)
    except CapabilityError:
        raise
    except NotImplementedError as error:
        issue = CapabilityIssue("lowering.unsupported", str(error))
        raise CapabilityError(CapabilityReport(
            False, (issue,), report.summary)) from error


def capability_report(network, duration, namespace=None, rng_seed=0):
    """Inspect a model without building or executing a Rust artifact."""
    report = collect_network_issues(network, duration)
    if not report.supported:
        return report
    try:
        _lower_network(network, duration, namespace, rng_seed)
    except NotImplementedError as error:
        return CapabilityReport(
            False, (CapabilityIssue("lowering.unsupported", str(error)),),
            report.summary)
    return report


def export_network(network, duration, path, namespace=None, rng_seed=0):
    """Export initial state for standalone replay without running the model."""
    model = lower_network(network, duration, namespace, rng_seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return model
