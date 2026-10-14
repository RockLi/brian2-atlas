"""Structured, model-wide capability diagnostics for the Rust backend."""

from dataclasses import asdict, dataclass
import math

import numpy as np
from brian2 import (EventMonitor, Network, NeuronGroup, PoissonGroup, PoissonInput,
                    PopulationRateMonitor, SpikeGeneratorGroup, SpikeMonitor, StateMonitor, Synapses,
                    second)
from brian2.core.magic import MagicNetwork
from brian2.core.preferences import prefs
from brian2.groups.subgroup import Subgroup
from brian2.groups.group import CodeRunner
from brian2.core.operations import NetworkOperation
from brian2.core.variables import Subexpression
from brian2.spatialneuron import SpatialNeuron
from brian2.units.fundamentalunits import Quantity


REPORT_SCHEMA = "b2-capability-report-v1"
SUPPORTED_STATE_UPDATERS = frozenset({
    "exact", "linear", "independent", "euler", "rk2", "rk4",
    "exponential_euler", "heun", "milstein",
    "gsl", "gsl_rk2", "gsl_rk4", "gsl_rkf45", "gsl_rkck", "gsl_rk8pd",
})
DEFAULT_METHOD_CHOICE = ("exact", "euler", "heun")
SPATIAL_DEFAULT_METHOD_CHOICE = (
    "exact", "exponential_euler", "rk2", "heun")
SUPPORTED_ROOT_TYPES = (
    NeuronGroup, SpatialNeuron, PoissonGroup, SpikeGeneratorGroup, PoissonInput,
    StateMonitor, EventMonitor, SpikeMonitor, PopulationRateMonitor, Synapses,
    NetworkOperation)


@dataclass(frozen=True)
class CapabilityIssue:
    code: str
    message: str
    object_name: str | None = None
    object_type: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CapabilityReport:
    supported: bool
    issues: tuple[CapabilityIssue, ...]
    summary: dict
    schema: str = REPORT_SCHEMA

    def to_dict(self):
        return {
            "schema": self.schema,
            "supported": self.supported,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def format_text(self):
        if self.supported:
            return "Rust standalone capability report: supported"
        lines = [
            f"Rust standalone capability report: {len(self.issues)} issue(s)"]
        for issue in self.issues:
            owner = ""
            if issue.object_name is not None:
                owner = f" [{issue.object_type} {issue.object_name}]"
            lines.append(f"- {issue.code}{owner}: {issue.message}")
        return "\n".join(lines)


class CapabilityError(NotImplementedError):
    """Raised before build when a model exceeds the supported semantic set."""

    def __init__(self, report):
        self.report = report
        super().__init__(report.format_text())


def supported_method_choice(choice, equations):
    """Whether Brian can safely resolve this method choice before IR lowering."""
    if choice == "independent":
        if equations.is_stochastic:
            return False
        differential = {
            equation.varname for equation in equations.values()
            if equation.type == "differential equation"
        }
        return all(
            not (expression.identifiers & (differential - {name}))
            for name, expression in equations.get_substituted_expressions()
        )
    if choice in {"exact", "linear"}:
        return not equations.is_stochastic
    if choice in SUPPORTED_STATE_UPDATERS:
        return True
    return (isinstance(choice, (tuple, list)) and
            tuple(choice) in {DEFAULT_METHOD_CHOICE,
                              SPATIAL_DEFAULT_METHOD_CHOICE})


def _root_objects(network):
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


def _population_endpoint(endpoint, groups):
    """Resolve a full population or a contiguous subgroup to its parent."""
    if type(endpoint) in (
            NeuronGroup, PoissonGroup, SpikeGeneratorGroup, SpatialNeuron):
        return endpoint if endpoint in groups else None
    if not isinstance(endpoint, Subgroup):
        return None
    parent = next((group for group in groups
                   if group.id == endpoint.source.id), None)
    if (parent is None or
            not 0 <= endpoint.start < endpoint.stop <= len(parent)):
        return None
    return parent


def collect_network_issues(
        network, duration, *, network_operations_prevalidated=False):
    """Collect independent, safely inspectable incompatibilities in one pass."""
    issues = []

    def add(code, message, obj=None):
        issues.append(CapabilityIssue(
            code, message, getattr(obj, "name", None),
            type(obj).__name__ if obj is not None else None))

    if type(network) not in (Network, MagicNetwork):
        add("network.type", "expected Network or MagicNetwork")
        return CapabilityReport(False, tuple(issues), {"objects": 0})

    try:
        duration_seconds = float(duration / second)
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            add("run.duration", "duration must be finite and non-negative")
    except Exception:
        # Invalid dimensions are a Brian2 API error, not a backend capability
        # mismatch.  Preserve the native DimensionMismatchError raised during
        # lowering instead of relabelling it as unsupported functionality.
        pass

    objects = _root_objects(network)
    for obj in objects:
        if type(obj) not in SUPPORTED_ROOT_TYPES:
            add("object.type", "root object type is not supported", obj)

    groups = [obj for obj in objects
              if type(obj) in (NeuronGroup, SpatialNeuron,
                               PoissonGroup, SpikeGeneratorGroup)]
    spatial_groups = [obj for obj in objects if type(obj) is SpatialNeuron]
    endpoint_groups = groups
    neuron_groups = [obj for obj in groups
                     if type(obj) in (NeuronGroup, SpatialNeuron)]
    state_monitors = [obj for obj in objects if type(obj) is StateMonitor]
    event_monitors = [obj for obj in objects if type(obj) is EventMonitor]
    spike_monitors = [obj for obj in objects if type(obj) is SpikeMonitor]
    rate_monitors = [obj for obj in objects if type(obj) is PopulationRateMonitor]
    synapses = [obj for obj in objects if type(obj) is Synapses]
    poisson_inputs = [obj for obj in objects if type(obj) is PoissonInput]
    network_operations = [obj for obj in network.sorted_objects
                          if type(obj) is NetworkOperation]
    validated_operations = ([] if network_operations_prevalidated
                            else network_operations)
    if len(validated_operations) > 16:
        add("network_operation.count",
            "at most 16 Python NetworkOperations are supported")
    active_operations = [obj for obj in validated_operations if obj.active]
    for slot in ("start", "end"):
        slot_operations = [obj for obj in active_operations if obj.when == slot]
        slot_effects = [obj for obj in network.sorted_objects
                        if obj.active and obj.when == slot and
                        (isinstance(obj, CodeRunner) or
                         type(obj) is NetworkOperation)]
        expected = (slot_effects[:len(slot_operations)] if slot == "start"
                    else slot_effects[len(slot_effects)-len(slot_operations):]
                    if slot_operations else [])
        if expected != slot_operations:
            add("network_operation.order",
                "start NetworkOperations must precede native start effects and "
                "end NetworkOperations must follow native end effects")
    for operation in validated_operations:
        if (operation.when not in {"start", "end"} or
                operation.contained_objects or
                not operation.active):
            add("network_operation.configuration",
                "NetworkOperation must be active, childless, and scheduled at "
                "the start or end tick boundary",
                operation)
            continue
        interval = float(operation.clock.dt / second)
        start = float(network.t / second)
        total = float(duration / second)
        native_clocks = {obj.clock for obj in network.sorted_objects
                         if obj.active and type(obj) is not NetworkOperation}
        quantum = min(
            (float(clock.dt / second) for clock in native_clocks),
            default=math.nan)
        if (not math.isfinite(interval) or interval <= 0 or
                not math.isclose(start / interval, round(start / interval),
                                 rel_tol=0, abs_tol=1e-9) or
                not math.isclose(total / quantum, round(total / quantum),
                                 rel_tol=0, abs_tol=1e-9) or
                not all(math.isclose(
                    interval / float(clock.dt / second),
                    round(interval / float(clock.dt / second)),
                    rel_tol=0, abs_tol=1e-9)
                        for clock in native_clocks)):
            add("network_operation.clock",
                "run start must align to NetworkOperation dt, run duration must align "
                "to the native clock quantum, and NetworkOperation dt must be an "
                "integer multiple of every active native clock dt",
                operation)
    if not endpoint_groups:
        add("population.missing", "at least one population is required")

    state_by_source = {}
    for monitor in state_monitors:
        source = monitor.source
        parent = _population_endpoint(source, endpoint_groups)
        state_by_source.setdefault(source, []).append(monitor)
        if (type(source) is not Synapses and
                type(parent) not in (NeuronGroup, PoissonGroup, SpatialNeuron)) or (
                type(source) is Synapses and source not in synapses):
            add("monitor.state.source",
                "StateMonitor source must be a NeuronGroup, PoissonGroup, "
                "contiguous subgroup, or Synapses in this network", monitor)
        if monitor.contained_objects:
            add("monitor.state.children", "custom monitor children are unsupported",
                monitor)
        if type(parent) in (NeuronGroup, SpatialNeuron):
            monitorable = {
                equation.varname for equation in parent.equations.values()
                if equation.type == "differential equation" or
                equation.type == "parameter" or
                equation.type == "subexpression"
            }
            if parent.state_updater.method_choice in {
                    "gsl", "gsl_rk2", "gsl_rk4", "gsl_rkf45",
                    "gsl_rkck", "gsl_rk8pd"}:
                monitorable.update(
                    name for name in {
                        "_last_timestep", "_failed_steps", "_step_count"}
                    if name in parent.variables)
            if parent._refractory is not False:
                monitorable.update({"lastspike", "not_refractory"})
            if (not monitor.record_variables or
                    not set(monitor.record_variables) <= monitorable):
                add("monitor.state.variables",
                    "StateMonitor variables must be supported model states, "
                    "parameters, or linked variables",
                    monitor)
        elif type(parent) is PoissonGroup:
            if (source is not parent or not monitor.record_variables or
                    set(monitor.record_variables) != {"rates"} or
                    isinstance(parent.variables["rates"], Subexpression)):
                add("monitor.state.variables",
                    "PoissonGroup StateMonitor can record stored rates from the full group",
                    monitor)
        elif type(source) is Synapses:
            equations = {equation.varname: equation
                         for equation in source.equations.values()}
            if source.event_driven is not None:
                equations.update({equation.varname: equation
                                  for equation in source.event_driven.values()})
            direct_monitorable = {
                equation.varname for equation in equations.values()
                if (equation.type == "differential equation" or
                    (equation.type == "parameter" and not equation.flags))
                and equation.varname in source.variables
                and not source.variables[equation.varname].scalar
                and not source.variables[equation.varname].constant
            }
            endpoint_monitorable = set()
            for name, variable in source.variables.items():
                owner = getattr(variable, "owner", None)
                owner_id = getattr(owner, "id", None)
                if owner_id not in {
                        getattr(source.source, "id", None),
                        getattr(source.target, "id", None)}:
                    continue
                equations = getattr(owner, "equations", {})
                equation = equations.get(variable.name)
                if equation is not None and equation.type in {
                        "differential equation", "parameter"}:
                    endpoint_monitorable.add(name)
            monitorable = (direct_monitorable | endpoint_monitorable |
                           set(source._linked_variables))
            if (not monitor.record_variables or
                    not set(monitor.record_variables) <= monitorable):
                add("monitor.state.variables",
                    "Synapses StateMonitor variables must be mutable synaptic states "
                    "or pre/post neuron states",
                    monitor)
    for source, source_monitors in state_by_source.items():
        schedules = {(monitor.when, monitor.order, monitor.clock)
                     for monitor in source_monitors}
        # Synapse monitors have independent typed result streams, so each can
        # keep its own Brian schedule. Population monitors currently share one
        # compact physical trace and therefore still need one schedule.
        if type(source) is not Synapses and len(schedules) > 1:
            add("monitor.state.schedule",
                "StateMonitors sharing a source must use the same when/order/clock",
                source_monitors[0])

    spike_by_source = {}
    for monitor in spike_monitors:
        source = monitor.source
        parent = _population_endpoint(source, endpoint_groups)
        expected_order = source.order + 1 if isinstance(source, Subgroup) else 1
        spike_by_source.setdefault(source, []).append(monitor)
        if parent is None:
            add("monitor.spike.source",
                "SpikeMonitor source must be a population or contiguous subgroup "
                "in this network", monitor)
        variables = set(monitor.record_variables) - {"i", "t"}
        if (monitor.when != "thresholds" or
                monitor.order != expected_order or
                parent is None or monitor.clock is not parent.clock):
            add("monitor.spike.configuration",
                "SpikeMonitor must use its default schedule",
                monitor)
        if type(parent) in (NeuronGroup, SpatialNeuron):
            monitorable = {
                equation.varname for equation in parent.equations.values()
                if equation.type in {"differential equation", "parameter"}
            }
            if not variables <= monitorable:
                add("monitor.spike.variables",
                    "SpikeMonitor variables must be model states, parameters, "
                    "or linked variables", monitor)
        elif variables:
            add("monitor.spike.variables",
                "stateless event sources cannot record additional SpikeMonitor variables",
                monitor)
        if monitor.contained_objects:
            add("monitor.spike.children", "custom monitor children are unsupported",
                monitor)
    for source, monitors in spike_by_source.items():
        if len(monitors) > 1:
            add("monitor.spike.multiple",
                "at most one SpikeMonitor per population is supported", source)

    for monitor in rate_monitors:
        source = monitor.source
        parent = _population_endpoint(source, endpoint_groups)
        if type(parent) not in (NeuronGroup, SpatialNeuron):
            add("monitor.rate.source",
                "PopulationRateMonitor source must be a NeuronGroup or contiguous "
                "subgroup", monitor)
        if (monitor.when != "end" or monitor.order != 0 or
                parent is None or monitor.clock is not parent.clock or
                monitor.contained_objects or
                np.dtype(monitor.variables["rate"].dtype) not in
                {np.dtype(np.float32), np.dtype(np.float64)}):
            add("monitor.rate.configuration",
                "PopulationRateMonitor needs its default schedule, source clock, and f32/f64 rate", monitor)

    for monitor in event_monitors:
        source = monitor.source
        variables = set(monitor.record_variables) - {"i", "t"}
        if type(source) not in (NeuronGroup, SpatialNeuron) or source not in groups:
            add("monitor.event.source",
                "EventMonitor source must be a NeuronGroup in this network", monitor)
        elif monitor.event not in source.events:
            add("monitor.event.name",
                "EventMonitor must reference a declared source event", monitor)
        elif (not monitor.record or monitor.clock is not source.clock or
              monitor.contained_objects):
            add("monitor.event.configuration",
                "EventMonitor must record indices on its source clock", monitor)
        else:
            monitorable = {
                equation.varname for equation in source.equations.values()
                if equation.type in {"differential equation", "parameter"}
            }
            if not variables <= monitorable:
                add("monitor.event.variables",
                    "EventMonitor variables must be supported model states, "
                    "parameters, or linked variables",
                    monitor)

    inputs_by_group = {group: [] for group in neuron_groups}
    for item in poisson_inputs:
        parent = _population_endpoint(item._group, endpoint_groups)
        if parent not in inputs_by_group:
            add("poisson_input.target",
                "PoissonInput must target a NeuronGroup or contiguous subgroup in "
                "this network", item)
        else:
            inputs_by_group[parent].append(item)
        if (parent is None or item.clock is not parent.clock or
                item.when != "synapses" or
                item.order != 0 or item.contained_objects):
            add("poisson_input.schedule",
                "PoissonInput must use its target clock and default schedule", item)

    for group in neuron_groups:
        method = group.state_updater.method_choice
        gsl_method = method in {
            "gsl", "gsl_rk2", "gsl_rk4", "gsl_rkf45", "gsl_rkck",
            "gsl_rk8pd"}
        gsl_compatible = (not gsl_method or
                          (type(group) is NeuronGroup and
                           not group.equations.is_stochastic and
                           all(np.dtype(group.variables[equation.varname].dtype) ==
                               np.dtype(np.float64)
                               for equation in group.equations.values()
                               if equation.type == "differential equation") and
                           not any(set(equation.flags) == {"linked"}
                                   for equation in group.equations.values()) and
                           (group._refractory is False or
                            isinstance(group._refractory, Quantity))))
        if (not supported_method_choice(method, group.equations) or
                (group.state_updater.method_options and not gsl_method) or
                not gsl_compatible):
            add("population.method",
                "use Brian2's default deterministic method selection or explicit "
                "exact, linear, independent, euler, rk2, rk4, "
                "exponential_euler, heun, milstein, gsl, gsl_rk2, gsl_rk4, "
                "gsl_rkf45, gsl_rkck or gsl_rk8pd; only GSL "
                "methods accept method options; "
                "independent requires ODEs without cross-state dependencies",
                group)
        if group._refractory is not False:
            if prefs.legacy.refractory_timing:
                add("population.refractory",
                    "legacy refractory timing is unsupported", group)
            elif (not isinstance(group._refractory, Quantity) or
                  group._refractory.size != 1) and not isinstance(
                      group._refractory, str):
                add("population.refractory",
                    "refractory must be a fixed scalar duration or expression", group)
        if (not all(isinstance(name, str) and name.isidentifier()
                    for name in group.events) or
                not set(group.event_codes) <= set(group.events)):
            add("population.events",
                "events need identifier names and run_on_event code must reference "
                "a declared event", group)
        for equation in group.equations.values():
            if group._refractory is not False and equation.varname in {
                    "lastspike", "not_refractory"}:
                continue
            variable = group.variables[equation.varname]
            supported = {np.dtype(np.float32), np.dtype(np.float64)}
            if equation.type != "differential equation":
                supported.update({
                    np.dtype(np.int32), np.dtype(np.int64),
                    np.dtype(np.uint32), np.dtype(np.uint64),
                    np.dtype(np.bool_),
                })
            if np.dtype(variable.dtype) not in supported:
                add("population.dtype",
                    f"{equation.varname} must use float32/float64"
                    + ("" if equation.type == "differential equation" else
                       "/int32/int64/uint32/uint64/bool"),
                    group)
        expected_children = {group.state_updater}
        if type(group) is SpatialNeuron:
            expected_children.add(group.diffusion_state_updater)
            if (group.diffusion_state_updater.when != "groups" or
                    group.diffusion_state_updater.order != 1 or
                    group.diffusion_state_updater.clock is not group.clock):
                add("population.spatial_schedule",
                    "SpatialNeuron cable updates require the default groups/order=1 schedule",
                    group)
        if group.subexpression_updater is not None:
            expected_children.add(group.subexpression_updater)
        expected_children.update(group.thresholder.values())
        expected_children.update(group.resetter.values())
        expected_children.update(child for child in group.contained_objects
                                 if type(child) is Synapses and child.source is group)
        subgroups = [child for child in group.contained_objects
                     if isinstance(child, Subgroup)]
        expected_children.update(subgroups)
        extra_children = set(group.contained_objects) - expected_children
        invalid_children = [child for child in extra_children
                            if type(child) is not CodeRunner]
        if invalid_children:
            add("population.children",
                "contained objects must be standard runners or run_regularly",
                group)
        for runner in extra_children:
            if type(runner) is CodeRunner and (
                    runner.group.id != group.id or
                    runner.contained_objects or runner.codeobj_class is not None):
                add("population.run_regularly",
                    "run_regularly must use the standard stateless CodeRunner",
                    runner)
        for subgroup in subgroups:
            if (subgroup.source.id != group.id or
                    not 0 <= subgroup.start < subgroup.stop <= len(group)):
                add("population.subgroup", "invalid contiguous subgroup", subgroup)
                continue
            for runner in subgroup.contained_objects:
                if (type(runner) is not CodeRunner or
                        runner.group.id != subgroup.id or
                        runner.contained_objects or runner.codeobj_class is not None):
                    add("population.run_regularly",
                        "subgroup run_regularly must use the standard stateless CodeRunner",
                        runner)
        if len(inputs_by_group.get(group, ())) > 32:
            add("poisson_input.count", "at most 32 PoissonInput objects per population",
                group)

    for synapse in synapses:
        source_is_synapses = type(synapse.source) is Synapses
        target_is_synapses = type(synapse.target) is Synapses
        edge_endpoints_are_flat = all(
            type(endpoint.source) is not Synapses and
            type(endpoint.target) is not Synapses
            for endpoint in (synapse.source, synapse.target)
            if type(endpoint) is Synapses)
        # A presynaptic Synapses endpoint would have to generate events from
        # an edge domain, which Brian does not currently define.  A
        # postsynaptic edge domain is useful, however: a normal spiking group
        # can modulate mutable state on another Synapses object (e.g. the
        # dopamine/reward projection in Izhikevich 2007).
        edge_endpoint = source_is_synapses or target_is_synapses
        non_pathway_children = [
            child for child in set(synapse.contained_objects) -
            set(synapse._pathways)
            if type(child) is CodeRunner
        ]
        edge_summed_only = (
            edge_endpoint and edge_endpoints_are_flat and
            bool(synapse.summed_updaters) and
            not synapse._pathways and synapse.state_updater is None and
            synapse.subexpression_updater is None and
            not non_pathway_children and
            all(name.endswith("_post")
                for name in synapse.summed_updaters))
        edge_target_event_only = (
            target_is_synapses and not source_is_synapses and
            edge_endpoints_are_flat and
            bool(synapse._pathways) and not synapse.summed_updaters and
            synapse.state_updater is None and
            synapse.subexpression_updater is None and
            not non_pathway_children and
            all(pathway.prepost == "pre" for pathway in synapse._pathways))
        if edge_endpoint and not (edge_summed_only or edge_target_event_only):
            add(
                "synapse.endpoint.synapses",
                "this use of Synapses as an endpoint requires additional "
                "synaptic-edge population-domain semantics",
                synapse)
        source_supported = (
            type(synapse.source) in (
                NeuronGroup, PoissonGroup, SpikeGeneratorGroup, SpatialNeuron,
                Synapses)
            or isinstance(synapse.source, Subgroup))
        target_supported = (
            type(synapse.target) in (
                NeuronGroup, PoissonGroup, SpikeGeneratorGroup, SpatialNeuron,
                Synapses)
            or isinstance(synapse.target, Subgroup))
        if not source_supported or not target_supported:
            add("synapse.endpoint", "unsupported source or target endpoint", synapse)
        if not synapse._connect_called:
            add("synapse.connect", "Synapses.connect must be called", synapse)
        for name in synapse._linked_variables:
            variable = synapse.variables[name]
            remote = [
                dependency for dependency in getattr(variable, "identifiers", ())
                if dependency in synapse.variables and
                any(candidate.id == getattr(
                    synapse.variables[dependency].owner, "id", None)
                    for candidate in groups)
            ]
            fixed = all(
                isinstance(synapse.variables.indices[dependency],
                           (int, np.integer)) or
                (isinstance(synapse.variables.indices[dependency], str) and
                 synapse.variables.indices[dependency].removeprefix("-").isdigit())
                for dependency in remote
            )
            if not isinstance(variable, Subexpression) or not remote or not fixed:
                add("synapse.linked",
                    "Synapses linked variables require a deterministic linked "
                    "expression with fixed population-state source indices",
                    synapse)
        if synapse.subexpression_updater is not None and (
                synapse.subexpression_updater.when != "before_start" or
                synapse.subexpression_updater.order != synapse.order or
                synapse.subexpression_updater.clock is not synapse.clock):
            add("synapse.subexpression.schedule",
                "synaptic (constant over dt) subexpressions require the default schedule",
                synapse)
        if synapse.state_updater is not None and (
                not supported_method_choice(
                    synapse.state_updater.method_choice, synapse.equations) or
                synapse.state_updater.method_options or
                synapse.state_updater.when != "groups" or
                synapse.state_updater.order != 0):
            add("synapse.state_update",
                "synaptic ODEs require Brian's default method selection or explicit "
                "'exact', 'linear', 'independent', 'euler', 'rk2', 'rk4', "
                "'exponential_euler', 'heun' or 'milstein' and the default "
                "schedule; independent requires ODEs without cross-state "
                "dependencies",
                synapse)

    summary = {
        "objects": len(objects),
        "populations": len(endpoint_groups),
        "neuron_groups": len(neuron_groups),
        "spatial_neurons": len(spatial_groups),
        "synapses": len(synapses),
        "state_monitors": len(state_monitors),
        "event_monitors": len(event_monitors),
        "spike_monitors": len(spike_monitors),
        "rate_monitors": len(rate_monitors),
        "poisson_inputs": len(poisson_inputs),
        "network_operations": len(network_operations),
    }
    return CapabilityReport(not issues, tuple(issues), summary)
