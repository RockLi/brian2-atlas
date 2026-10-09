"""Checked Brian-to-training lowering for explicit spiking networks.

This module snapshots a supported Network without running it or modifying its
objects. Unsupported behavior is rejected before a trainer can be launched.
"""
import ast
import copy
from dataclasses import dataclass
import hashlib

import numpy as np
from brian2 import (Network, NeuronGroup, Synapses, SpikeMonitor, StateMonitor,
                    PopulationRateMonitor, SpikeGeneratorGroup, PoissonGroup, prefs, second)
from brian2.core.functions import DEFAULT_FUNCTIONS, timestep, Function
from brian2.core.clocks import Clock, check_dt
from brian2.codegen.translation import make_statements, get_identifiers_recursively
from brian2.core.variables import Subexpression
from brian2.groups.subgroup import Subgroup
from brian2.equations.unitcheck import check_units_statements
from brian2.equations.equations import check_subexpressions, extract_constant_subexpressions
from brian2.parsing.sympytools import check_expression_for_multiple_stateful_functions
from brian2.parsing.expressions import parse_expression_dimensions, is_boolean_expression
from brian2.units.fundamentalunits import have_same_dimensions, Quantity
from brian2.stateupdaters.base import StateUpdateMethod, UnsupportedEquationsException
from brian2.stateupdaters.explicit import euler, rk2, rk4, heun, milstein

from .training_inputs import TimedInputRegistry, TimedArray
from .training_functions import lower_pure_function
from .training_effects import StateEffectFunction, lower_state_effect_function, bind_state_effect_captures, capture_array_key, same_capture_storage, compile_state_effect_transform
from .protocol import canonical_bytes
from .training import lif_training_plan
from .training_brian_regular import is_regular, regular_children
from .training_equations import compile_training_equation, compile_training_predicate, neuron_parameter_bank, _compile_training_ast, NeuronParameter, SimulationTime, NormalNoise, UniformNoise, PoissonNoise, _MappedParameter, _DeferredParameter, coerce_state_expression, StateSlot

_INTEGRATORS={'euler':euler,'rk2':rk2,'rk4':rk4,'heun':heun,'milstein':milstein}


def _snapshot_clock_time(network, clock):
    """Read the pending tick that Brian will use, without preparing the clock.

    Assigning dt leaves the physical t/timestep arrays on the old grid until
    Network.run. Preserve that deferred transition (including its validation)
    while compiling warm refractory state and the native itinerary.
    """
    if type(clock) is not Clock:
        raise TrainingConversionError('clock',clock,'dynamic conversion requires standard Brian clocks')
    dt=float(clock.dt_)
    if not np.isfinite(dt) or dt<=0:
        raise TrainingConversionError('clock',clock,'dynamic conversion requires a finite positive clock step')
    target=float(network.t)
    if clock._old_dt is not None and clock._old_dt!=dt:
        try:check_dt(dt,clock._old_dt,target)
        except ValueError as error:raise TrainingConversionError('clock',clock,str(error)) from error
        return float(clock._calc_timestep(target))*dt
    actual=float(clock.variables['t'].get_value()[0])
    if float(clock._calc_timestep(target))*dt!=actual:
        raise TrainingConversionError('clock',clock,'snapshot clock differs from the next Network.run tick')
    return actual


def _endpoint_parent(group, selected):
    """Return the selected physical population, never a Subgroup weak proxy."""
    parent=group.source if isinstance(group,Subgroup) else group
    return next((candidate for candidate in selected if candidate==parent),None)


def _endpoint_start(group):
    return group.start if isinstance(group,Subgroup) else 0


def _precise_refractory_clock(program, last_index, low_slot, high_slot):
    """Keep elapsed-clock predicates exact without changing model-state dtype."""
    constants={};nodes=[];remap={};words={}
    def elapsed(index):
        n=program[index]
        return (n['op']=='sub' and program[n['left']]['op']=='time'
                and program[n['right']]==dict(op='state',index=last_index))
    def word(slot):
        if slot not in words:
            words[slot]=len(nodes);nodes.append(dict(op='integer_state',index=slot))
        return words[slot]
    for old,node in enumerate(program):
        op=node['op'];value=None
        if op in ('constant','integer_constant'):value=float(node['value'])
        elif op in ('neg','integer_float') and node['arg'] in constants:
            value=constants[node['arg']]*(-1 if op=='neg' else 1)
        elif op in ('add','sub','mul','div') and node['left'] in constants and node['right'] in constants:
            a=constants[node['left']];b=constants[node['right']]
            if op=='add':value=a+b
            elif op=='sub':value=a-b
            elif op=='mul':value=a*b
            elif b!=0:value=a/b
        if value is not None and np.isfinite(value):constants[old]=value
        if op=='discrete_compare' and (elapsed(node['left']) != elapsed(node['right'])):
            reverse=not elapsed(node['left']);right=node['left'] if reverse else node['right']
            kind=node['kind']
            if reverse:kind=dict(gt='lt',ge='le',lt='gt',le='ge',eq='eq',ne='ne')[kind]
            low=word(low_slot);high=word(high_slot)
            nodes.append(dict(op='elapsed_compare',low=low,high=high,right=remap[right],kind=kind,constant=constants.get(right)))
        else:
            n=dict(node);fields=['rate','arg','left','right','condition','yes','no','slope','scale']
            if op in ('timed_parameter','parameter_gather','integer_parameter_gather'):
                fields+=['index']
                if op=='timed_parameter':fields+=['time']
            for key in fields:
                if key in n:n[key]=remap[n[key]]
            nodes.append(n)
        remap[old]=len(nodes)-1
    if len(nodes)>128:raise ValueError('precise refractory expression exceeds 128 nodes')
    return nodes


class TrainingConversionError(ValueError):
    def __init__(self, code, owner, message):
        self.code=code;self.owner=getattr(owner,'name',str(owner))
        super().__init__(f'{code} [{self.owner}]: {message}')


def _state_dtype(variable):
    dtype=np.dtype(variable.dtype)
    if dtype.kind=='f':return 'float'
    if dtype==np.dtype('int32'):return 'integer'
    if dtype.kind=='b':return 'boolean'
    raise ValueError('dynamic storage supports floating point, int32 and Boolean values')


def _runtime_index(group, name, seen=()):
    if name in seen or len(seen)>=16:
        raise TrainingConversionError('linked',group,'cyclic or excessive linked index depth')
    index=group.variables.indices[name]
    if index in ('0','_idx'):return False
    return not group.variables[index].constant or _runtime_index(group,index,(*seen,name))


def _storage_indices(group, name):
    """Resolve Brian's physical indices without running code or copying aliases."""
    variable=group.variables[name];index=group.variables.indices[name]
    if index=='0':indices=np.zeros(len(group),dtype=np.int64)
    elif index=='_idx':indices=np.arange(len(group),dtype=np.int64)
    else:
        mapping=group.variables[index]
        if not mapping.constant or not np.issubdtype(mapping.dtype,np.integer):
            raise TrainingConversionError('linked',group,'linked indices must be fixed integer storage')
        indices=np.asarray(mapping.get_value()).reshape(-1)
    size=np.asarray(variable.get_value()).size
    if len(indices)!=len(group) or np.any(indices<0) or np.any(indices>=size):
        raise TrainingConversionError('linked',group,f'invalid physical indices for {name}')
    return indices


@dataclass
class BrianTrainingBundle:
    plan: dict
    weights: list
    initial_membrane: list
    provenance: dict
    initial_state: list | None = None


def lower_brian_training(network, *, input_group, layers, trainable_neuron_parameters=None,
                         run_namespace=None, backend='cpu', mpi_ranks=None,
                         detach_reset=True, dynamic=False, trainable_synapse_parameters=None,
                         external_state_inputs=None,
                         _defer_synapses=False, _input_registry=None, _mutable_constant_ids=(), _mutable_capture_inputs=None, _readonly_capture_inputs=None, _capture_constant_ids=(), **training_options):
    """Snapshot an explicit deterministic or stochastic Network to a v3/v4 plan.

    ``layers`` orders all non-input NeuronGroups; the last supplies class logits.
    Input spikes remain explicit batch/time/neuron arrays at training execution.
    Requested trainable neuron names must be constant parameters. Shared names
    have one slot; non-shared names have one slot per neuron. Values use SI units.
    The returned initial membrane is one sample; replicate it explicitly for a
    batch. Learned values are not written back to Brian objects automatically.
    Set ``dynamic=True`` to lower ordered stateful synapses through the v5 path.
    """
    if type(dynamic) is not bool:raise ValueError('dynamic must be a boolean')
    if dynamic:
        if _defer_synapses:raise ValueError('reserved conversion option')
        from .training_brian_dynamic import lower_brian_dynamic_training
        return lower_brian_dynamic_training(network,input_group=input_group,layers=layers,
            trainable_neuron_parameters=trainable_neuron_parameters,trainable_synapse_parameters=trainable_synapse_parameters,
            external_state_inputs=external_state_inputs,
            run_namespace=run_namespace,backend=backend,mpi_ranks=mpi_ranks,detach_reset=detach_reset,**training_options)
    if trainable_synapse_parameters is not None:raise ValueError('synaptic parameter selection requires dynamic=True')
    if external_state_inputs is not None and not _defer_synapses:raise ValueError('external state inputs require dynamic=True')
    def require(condition,code,owner,message):
        if not condition:raise TrainingConversionError(code,owner,message)
    require(isinstance(network,Network),'network',network,'an explicit Brian Network is required')
    layers=list(layers);groups=[input_group,*layers]
    require(isinstance(input_group,(NeuronGroup,SpikeGeneratorGroup,PoissonGroup)),
            'input',input_group,'input must be a spiking population; its supplied spikes are external to training')
    require(all(isinstance(g,NeuronGroup) and 'v' in g.variables for g in layers),
            'neuron',network,'each trained layer must be a NeuronGroup with voltage v')
    require(len(layers)>=2 and len({id(g) for g in groups})==len(groups),
            'layers',network,'distinct input, hidden and output groups are required')
    require(len(groups)<=17 and all(0<len(g)<=65536 for g in groups),
            'size',network,'training requires 3..17 layers with 1..65536 neurons each')
    require(network.schedule==['start','groups','thresholds','synapses','resets','end'],
            'schedule',network,'only the default synchronous schedule is supported')
    object_names=[o.name for o in network.sorted_objects]
    require(len(object_names)==len(set(object_names)),'names',network,'Brian object names must be unique')
    roots=set(network.objects)
    require(all(g in roots for g in groups),'membership',network,'all selected groups must belong to the Network')
    synapses=[o for o in (network.sorted_objects if _defer_synapses else roots) if isinstance(o,Synapses)]
    regular_owners=[*([input_group] if external_state_inputs else []),*layers,*synapses,*[o for o in network.sorted_objects
                    if isinstance(o,Subgroup) and _endpoint_parent(o,layers) is not None]]
    monitors=(SpikeMonitor,StateMonitor,PopulationRateMonitor)
    for obj in roots:
        require(obj in groups or obj in synapses or isinstance(obj,monitors)
                or isinstance(obj,Subgroup) and _endpoint_parent(obj,groups) is not None
                or _defer_synapses and is_regular(obj,regular_owners),
                'object',obj,'object behavior has no training lowering')
        require(obj.active,'inactive',obj,'inactive objects must be removed explicitly')
    dt=float(input_group.clock.dt_)
    require(dt>0,'clock',input_group,'dt must be positive')
    namespace={} if run_namespace is None else dict(run_namespace)
    requested={} if trainable_neuron_parameters is None else dict(trainable_neuron_parameters)
    require(set(requested)<=set(g.name for g in layers),'parameter',network,'unknown trainable group name')
    forbidden={'sizes','beta','threshold','reset','projections','equations','threshold_parameters','threshold_per_neuron','masks','trainable','state_equations','state_resets','refractory','clock','noise_streams'}
    require(not (forbidden & training_options.keys()),'options',network,'derived model fields cannot be overridden')

    # Respect Brian's pathway scheduling; within a pathway spikes are ordered
    # by source index and each source traverses its original creation order.
    pathways=[o for o in network.sorted_objects if any(o is p for s in synapses for p in s._pathways)]
    synapses=sorted(synapses,key=lambda s:next((i for i,p in enumerate(pathways) if any(p is q for q in s._pathways)),len(pathways)))
    budget=training_options.get('max_tape_bytes',64*1024**2)
    require(type(budget) is int and 0<budget<=1024**3,'budget',network,'invalid training memory budget')
    projections=[];weights=[];bindings=[];admitted=0;frozen_banks=set()
    for syn in ([] if _defer_synapses else synapses):
        source_group=_endpoint_parent(syn.source,groups);target_group=_endpoint_parent(syn.target,layers)
        require(source_group is not None and target_group is not None,'endpoint',syn,'endpoints must belong to selected populations')
        require(len(syn._pathways)==1,'pathway',syn,'exactly one pre-spike pathway is required')
        path=syn._pathways[0]
        require(path.prepost=='pre' and path.event=='spike' and path.when=='synapses' and path.active,
                'pathway',syn,'only active pre-spike delivery in the synapses slot is supported')
        require(set(syn.contained_objects)=={path},'synaptic-dynamics',syn,'continuous, summed, or additional synaptic runners are unsupported')
        require(all(e.type=='parameter' for e in syn.equations.values()),'synaptic-dynamics',syn,'dynamic or subexpression synapse equations are unsupported')
        code=ast.parse(path.code,mode='exec').body
        require(len(code)==1 and isinstance(code[0],ast.AugAssign) and isinstance(code[0].op,ast.Add)
                and isinstance(code[0].target,ast.Name) and code[0].target.id=='v_post'
                and isinstance(code[0].value,ast.Name),'pathway-code',syn,'expected only v_post += weight')
        weight_name=code[0].value.id
        require(weight_name in syn.variables and weight_name in syn.equations,
                'weight',syn,'the additive weight must be a synaptic parameter')
        variable=syn.variables[weight_name]
        admitted+=len(syn)*24+(1 if variable.scalar else len(syn))*48+128
        require(admitted<=budget,'budget',syn,'parameter/topology budget exceeded before index-list allocation')
        require(have_same_dimensions(variable.dim,syn.target.variables['v'].dim),
                'units',syn,'weight units must match target voltage')
        require(np.issubdtype(variable.dtype,np.floating),'weight',syn,'trainable weights must be floating point')
        try:
            source=np.asarray(syn.variables['_synaptic_pre'].get_value(),dtype=int)
            target=np.asarray(syn.variables['_synaptic_post'].get_value(),dtype=int)
            values=np.asarray(variable.get_value(),dtype=float).reshape(-1)
            delay=np.asarray(path.delay[:],dtype=float)
        except Exception as error:raise TrainingConversionError('materialization',syn,'connections and values must be readable before conversion') from error
        require(source.size>0 and np.all(delay==0),'delay',syn,'nonempty, zero-delay connections are required')
        require(np.all(np.isfinite(values)),'weight',syn,'weight values must be finite')
        ids=np.zeros(source.size,dtype=int) if variable.scalar else np.arange(source.size)
        require(values.size==(1 if variable.scalar else source.size),'weight',syn,'weight shape mismatch')
        order=np.argsort(source,kind='stable')
        projections.append(dict(source_layer=groups.index(source_group),target_layer=groups.index(target_group),
            parameter_count=values.size,sources=source[order].tolist(),targets=target[order].tolist(),parameter_ids=ids[order].tolist()))
        weights.append(values.tolist());bindings.append(dict(bank=len(weights)-1,object=syn.name,variables=[weight_name],kind='synapse'))

    # Preserve the legacy v3 representation for canonical scalar Euler models.
    # General or mixed scalar resets use the already-supported one-state v4 ABI.
    def scalar_reset_kind(group):
        code=ast.parse(group.event_codes.get('spike',''),mode='exec').body
        if len(code)!=1:return None
        statement=code[0]
        if (isinstance(statement,ast.Assign) and len(statement.targets)==1
                and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id=='v'):
            value=statement.value
            if isinstance(value,ast.Constant) and value.value==0:return 'zero'
            if (isinstance(value,ast.BinOp) and isinstance(value.op,ast.Mult)
                    and isinstance(value.left,ast.Constant) and value.left.value==0
                    and isinstance(value.right,ast.Name)):return 'zero'
        threshold=ast.parse(group.events.get('spike','False'),mode='eval').body
        if (isinstance(statement,ast.AugAssign) and isinstance(statement.op,ast.Sub)
                and isinstance(statement.target,ast.Name) and statement.target.id=='v'
                and isinstance(threshold,ast.Compare) and len(threshold.comparators)==1
                and ast.dump(statement.value)==ast.dump(threshold.comparators[0])):return 'subtract'
        return None
    kinds=[scalar_reset_kind(g) for g in layers]
    def has_timed_input(group):
        codes=[eq.expr.code for eq in group.equations.values() if eq.expr is not None]
        codes += [group.events.get('spike','False'),group.event_codes.get('spike','')]
        calls={node.func.id for code in codes for node in ast.walk(ast.parse(code))
               if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        try:resolved=group.resolve_all(calls,run_namespace=namespace)
        except Exception:return False # Preserve the later coefficient/unit diagnostics.
        return any(isinstance(value,TimedArray) for value in resolved.values())
    temporal=any(has_timed_input(g) or g.equations.is_stochastic or {'rand','randn','poisson'} & g.equations.identifiers or
                 any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('rand','randn','poisson')
                     for code in [g.events.get('spike','False'),g.event_codes.get('spike','')] for n in ast.walk(ast.parse(code))) or 't' in g.equations.identifiers or
                 any(isinstance(n,ast.Name) and n.id=='t' for n in ast.walk(ast.parse(g.event_codes.get('spike',''))))
                 for g in layers)
    origin=_snapshot_clock_time(network,input_group.clock) if _defer_synapses else float(input_group.clock.variables['t'].get_value()[0])
    if temporal:
        require(np.isfinite(origin) and origin>=0 and (_defer_synapses or all(float(g.clock.variables['t'].get_value()[0])==origin for g in layers)),
                'clock',network,'time-dependent layers must share the input snapshot time')
    vector=(_defer_synapses or temporal or any(len(g.equations.diff_eq_names)>1 or g.state_updater.method_choice!='euler' or g._refractory is not False
                or any(e.type=='parameter' and g.variables[e.varname].constant and not g.variables[e.varname].scalar
                       for e in g.equations.values())
                or 'i' in g.equations.identifiers
                or any(isinstance(n,ast.Name) and n.id=='i' for n in ast.walk(ast.parse(g.event_codes.get('spike',''))))
                for g in layers)
            or None in kinds or len(set(kinds))>1)
    # A linked constant is one physical parameter bank, regardless of which
    # group names or fixed indices expose it. Do not turn it into carried state.
    constant_banks={};constant_sources=[];parameter_mappings=[];mapping_ids={};threshold_mappings=[]
    deferred_parameters=[];deferred_ids={}
    if _defer_synapses:
        owners={};targets=set(_mutable_constant_ids)|set(_capture_constant_ids)
        for obj in [*layers,*sorted(synapses,key=lambda s:s.name)]:
            eqs=obj.user_equations if obj in layers else obj.equations
            for eq in eqs.values():
                if eq.type!='parameter':continue
                variable=obj.variables[eq.varname]
                if 'linked' in eq.flags:
                    if variable.constant:targets.add(id(variable))
                elif variable.constant:owners[id(variable)]=(obj,eq.varname)
        # Explicit Variables.add_reference aliases can expose selected canonical
        # storage without an equation carrying the "linked" flag.
        for obj in [*layers,*synapses]:
            for name,variable in obj.variables.items():
                if id(variable) in owners and _runtime_index(obj,name):targets.add(id(variable))
        require(targets<=owners.keys(),'linked',network,'constant links require a selected canonical source')
        for identity in sorted(targets,key=lambda key:(owners[key][0].name,owners[key][1])):
            obj,name=owners[identity];variable=obj.variables[name]
            values=np.asarray(variable.get_value()).reshape(-1)
            require(values.size>0 and values.dtype.kind in 'fib' and np.all(np.isfinite(values)),
                    'parameter',obj,'linked constant sources require finite floating-point storage')
            admitted+=values.size*48+128
            require(admitted<=budget and len(weights)<256,'budget',obj,'canonical parameter budget exceeded')
            bank=len(weights);constant_banks[identity]=bank
            trainable=obj in layers and name in requested.get(obj.name,())
            kind=('neuron' if variable.scalar else 'neuron_array') if obj in layers else 'synapse_constant'
            bindings.append(dict(bank=bank,object=obj.name,variables=[name],kind=kind,trainable=trainable))
            constant_sources.append(dict(bank=bank,object=obj.name,variable=name))
            projections.append(neuron_parameter_bank(len(values),target_layer=layers.index(obj)+1 if obj in layers else 1))
            weights.append(values.astype(float).tolist())
            if not trainable:frozen_banks.add(bank)
    def mapped_parameter(group,name,variable):
        nonlocal admitted
        key=(group.name,name)
        if _defer_synapses and _runtime_index(group,name):
            if key not in deferred_ids:
                admitted+=128;require(admitted<=budget,'budget',group,'runtime parameter reference exceeds budget')
                deferred_ids[key]=len(deferred_parameters)
                deferred_parameters.append(dict(object=group.name,variable=name,bank=constant_banks[id(variable)]))
            return _DeferredParameter(constant_banks[id(variable)],deferred_ids[key],_state_dtype(variable))
        if key not in mapping_ids:
            admitted+=len(group)*16+128
            require(admitted<=budget,'budget',group,'constant index mapping exceeds budget')
            indices=_storage_indices(group,name).tolist();mapping_ids[key]=len(parameter_mappings)
            parameter_mappings.append(dict(object=group.name,variable=name,indices=indices))
        return _MappedParameter(constant_banks[id(variable)],mapping_ids[key],_state_dtype(variable))

    input_registry=_input_registry if _input_registry is not None else TimedInputRegistry()
    def input_bank(name,values):
        nonlocal admitted
        admitted+=values.size*48+128
        require(admitted<=budget and len(weights)<256,'budget',network,'timed input parameter budget exceeded')
        bank=len(weights);weights.append(values.astype(float).tolist());frozen_banks.add(bank)
        projections.append(neuron_parameter_bank(len(values)))
        bindings.append(dict(bank=bank,object=name,variables=['values'],kind='timed_input',trainable=False))
        return bank

    state_types=[];integer_parameters=[]
    for entry in constant_sources:
        owner=next(obj for obj in [*layers,*synapses] if obj.name==entry['object'])
        if _state_dtype(owner.variables[entry['variable']])=='integer':
            integer_parameters.extend([entry['bank'],i] for i in range(len(weights[entry['bank']])))
    programs=[];thresholds=[];threshold_refs=[];initial=[];reset_kind=None
    threshold_margins=[];vector_programs=[];vector_resets=[];initial_state=[];state_names=[];refractory=[];refractory_expressions=[];threshold_indexed=[];noise_names=[];neuron_draws=[]
    integrator_effect_writes=[None]*len(layers)
    reset_effect_writes=[None]*len(layers)
    reset_scalar_programs=[[] for _ in layers]
    reset_capture_programs=[{} for _ in layers]
    for layer,group in enumerate(layers,1):
        require(isinstance(group,NeuronGroup),'neuron',group,'only scalar NeuronGroups are supported')
        group_dt=float(group.clock.dt_)
        require(np.isfinite(group_dt) and group_dt>0 and (_defer_synapses or group_dt==dt),'clock',group,'static conversion requires equal finite clock steps')
        group_time=_snapshot_clock_time(network,group.clock) if _defer_synapses else float(group.clock.variables['t'].get_value()[0])
        # Extract cached subexpressions without Brian's synthetic refractory states.
        user_equations,_=extract_constant_subexpressions(group.user_equations)
        runtime_parameters=sorted(eq.varname for eq in user_equations.values()
            if _defer_synapses and eq.type=='parameter' and (not group.variables[eq.varname].constant or id(group.variables[eq.varname]) in _mutable_constant_ids))
        require(all(set(group.equations[name].flags)<=({'shared','linked','constant'} if id(group.variables[name]) in _mutable_constant_ids else {'shared','linked'}) for name in runtime_parameters),
                'state',group,'unsupported mutable neuron parameter flags')
        require(_defer_synapses or all('linked' not in eq.flags or not group.variables[eq.varname].constant for eq in group.user_equations.values()),
                'linked',group,'constant parameter links require dynamic=True')
        capture_inputs=(_mutable_capture_inputs or {}).get(group.name,{})
        readonly_inputs=(_readonly_capture_inputs or {}).get(group.name,{})
        names_of_states=['v',*sorted(group.equations.diff_eq_names-{'v'}),*runtime_parameters,*capture_inputs]
        identities={};state_aliases={}
        for slot,name in enumerate(names_of_states):
            array=capture_inputs[name] if name in capture_inputs else group.variables[name].get_value()
            state_aliases[slot]=identities.setdefault(capture_array_key(array),slot)
        require('v' in group.equations.diff_eq_names and len(names_of_states)<=16,
                'equations',group,'1..16 physical states including differential v are required')
        require(all(np.issubdtype(group.variables[name].dtype,np.floating) for name in group.equations.diff_eq_names),
                'equations',group,'differential states must be floating point')
        try:types=[('integer' if capture_inputs[name].dtype==np.dtype('int32') else 'boolean' if capture_inputs[name].dtype==np.dtype('bool') else 'float') if name in capture_inputs else _state_dtype(group.variables[name]) for name in names_of_states]
        except ValueError as error:raise TrainingConversionError('state',group,str(error)) from error
        state_types.append(types)
        state_names.append(names_of_states)
        method=group.state_updater.method_choice
        require(isinstance(method,str) and method in _INTEGRATORS
                and StateUpdateMethod.stateupdaters.get(method) is _INTEGRATORS[method],
                'integrator',group,'select a built-in euler, rk2, rk4, heun or milstein method explicitly')
        require(not group.state_updater.method_options,'integrator',group,'custom integration options are unsupported')
        ref=group._refractory
        equations=group.equations
        spec=None;counter=None;variable_ref=None
        if ref is not False:
            variable_ref=str(ref) if isinstance(ref,str) else None
            require(variable_ref is None or _defer_synapses,'refractory',group,'refractory expressions require dynamic=True')
            require(variable_ref is None or 0<len(variable_ref)<=8192,'refractory',group,'refractory expression exceeds source budget')
            require(variable_ref is not None or isinstance(ref,Quantity) and ref.ndim==0 and have_same_dimensions(ref,second)
                    and np.isfinite(float(ref)) and float(ref)>=0,
                    'refractory',group,'refractory requires a time Quantity or a Boolean/time expression')
            require(not prefs.legacy.refractory_timing
                    and 'lastspike' in group.variables and 'not_refractory' in group.variables,
                    'refractory',group,'refractory requires modern timing and a refractory-enabled group')
            steps=0 if variable_ref is not None else int(timestep(float(ref),group_dt))
            require(steps<=2**24 and len(names_of_states)<16,'refractory',group,'refractory exceeds tick or state budget')
            equations=user_equations
            spec=dict(steps=steps,clamp=[i for i,name in enumerate(names_of_states)
                                      if name not in capture_inputs and 'unless refractory' in equations[name].flags])
            elapsed=np.asarray(timestep(group_time-group.variables['lastspike'].get_value(),group_dt))
            require(np.all(elapsed>=0),'refractory',group,'lastspike must not be in the future')
            counter=np.maximum(steps-elapsed,0).astype(float)
        require(all(name in capture_inputs or set(equations[name].flags)<=({'shared','linked','constant'} if name in runtime_parameters and id(group.variables[name]) in _mutable_constant_ids else {'shared','linked'} if name in runtime_parameters else
                    {'unless refractory'} if spec is not None else set()) for name in names_of_states),
                'refractory',group,'unsupported differential equation flags')
        refractory.append(spec)
        refractory_expressions.append(None)
        require(set(group.events)=={'spike'} and set(group.event_codes)=={'spike'},
                'events',group,'one spike threshold and reset are required')
        runners=[group.state_updater,group.thresholder['spike'],group.resetter['spike']]
        extras=regular_children(group,synapses) if _defer_synapses else set()
        require(set(group.contained_objects)==set(runners)|extras,'runner',group,'additional runners require standard run_regularly or constant-over-dt with dynamic=True')
        require(all(r.active and r.when==slot and r.order==0 for r,slot in zip(runners,['groups','thresholds','resets'])),
                'schedule',group,'neuron runners require default slots and order')
        try:group.equations.check_units(group,run_namespace=namespace)
        except Exception as error:raise TrainingConversionError('units',group,str(error)) from error
        try:
            check_subexpressions(group,group.equations,namespace)
            # Validate before SymPy's cached expansion can merge repeated
            # stateful calls, even if an earlier consumer filled that cache.
            for eq in equations.values():
                if eq.expr is not None:
                    functions=group.resolve_all(eq.identifiers,run_namespace=namespace)
                    check_expression_for_multiple_stateful_functions(eq.expr.code,functions)
        except Exception as error:raise TrainingConversionError('noise',group,str(error)) from error
        expressions={name:expr.code for name,expr in equations.get_substituted_expressions()}
        expression=expressions['v']
        threshold=ast.parse(group.events['spike'],mode='eval').body
        ordered=isinstance(threshold,ast.Compare) and len(threshold.ops)==1 and isinstance(threshold.ops[0],(ast.Gt,ast.GtE,ast.Lt,ast.LtE))
        require(_defer_synapses or ordered and isinstance(threshold.left,ast.Name)
                and threshold.left.id=='v' and isinstance(threshold.ops[0],ast.Gt),
                'threshold',group,'static mode requires v > positive_scalar_threshold')
        substitutions={name:ast.parse(expr.code,mode='eval').body for name,expr in equations.get_substituted_expressions(include_subexpressions=True)
                       if name in equations.subexpr_names}
        class ExpandThreshold(ast.NodeTransformer):
            def visit_Name(self,node):
                return copy.deepcopy(substitutions[node.id]) if node.id in substitutions else node
        if _defer_synapses:threshold=ExpandThreshold().visit(threshold)
        if variable_ref is not None:
            try:variable_ref=ast.unparse(ExpandThreshold().visit(ast.parse(variable_ref,mode='eval').body))
            except (SyntaxError,ValueError) as error:raise TrainingConversionError('refractory',group,str(error)) from error
        threshold_node=threshold.comparators[0] if ordered else None
        threshold_text=ast.unparse(threshold_node) if ordered else ast.unparse(threshold)
        threshold_left=ast.unparse(threshold.left) if ordered else threshold_text
        reset=ast.parse(group.event_codes['spike'],mode='exec').body
        for node in ast.walk(ast.parse(group.event_codes['spike'],mode='exec')):
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store) and node.id in group.variables:
                require(not group.variables[node.id].scalar and group.variables.indices[node.id]!='0',
                        'reset',group,'Brian resets cannot write scalar/shared storage')
        zero=(len(reset)==1 and isinstance(reset[0],ast.Assign) and len(reset[0].targets)==1
              and isinstance(reset[0].targets[0],ast.Name) and reset[0].targets[0].id=='v'
              and ((isinstance(reset[0].value,ast.Constant) and reset[0].value.value==0) or
                   (isinstance(reset[0].value,ast.BinOp) and isinstance(reset[0].value.op,ast.Mult)
                    and isinstance(reset[0].value.left,ast.Constant) and reset[0].value.left.value==0
                    and isinstance(reset[0].value.right,ast.Name))))
        subtract=(len(reset)==1 and isinstance(reset[0],ast.AugAssign) and isinstance(reset[0].op,ast.Sub)
              and isinstance(reset[0].target,ast.Name) and reset[0].target.id=='v'
              and threshold_node is not None and ast.dump(reset[0].value)==ast.dump(threshold_node))
        require(vector or zero or subtract,'reset',group,'expected v=0 or v-=the_threshold_expression')
        kind='zero' if zero else 'subtract'
        require(vector or reset_kind in (None,kind),'reset',group,'mixed reset types across layers are unsupported')
        reset_kind=kind
        names={n.id for code in [*expressions.values(),threshold_text,threshold_left] for n in ast.walk(ast.parse(code,mode='eval')) if isinstance(n,ast.Name)}
        reset_tree=ast.parse(group.event_codes['spike'],mode='exec')
        reset_temporaries={n.id for n in ast.walk(reset_tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id not in group.variables}
        reset_names=get_identifiers_recursively([group.event_codes['spike']],group.variables)-reset_temporaries
        names |= reset_names-set(names_of_states)
        if variable_ref is not None:names.update(n.id for n in ast.walk(ast.parse(variable_ref,mode='eval')) if isinstance(n,ast.Name))
        names.update(entry['variable'] for entry in constant_sources if entry['object']==group.name)
        if _defer_synapses:
            # A selected constant may only be read by a synaptic path or summed
            # expression. Keep its canonical bank even when neuron code omits it.
            names.update(name for name in requested.get(group.name,())
                         if name in group.equations and group.equations[name].type=='parameter')
        try:resolved=group.resolve_all(names|reset_names|{'dt'},run_namespace=namespace)
        except Exception as error:raise TrainingConversionError('coefficient',group,str(error)) from error
        try:check_units_statements(group.event_codes['spike'],resolved)
        except Exception as error:raise TrainingConversionError('units',group,str(error)) from error
        try:
            require(is_boolean_expression(threshold,resolved),'threshold',group,'threshold must be Boolean')
            parse_expression_dimensions(threshold,resolved)
        except TrainingConversionError:raise
        except Exception as error:raise TrainingConversionError('units',group,str(error)) from error
        require(have_same_dimensions(parse_expression_dimensions(threshold_text,resolved),parse_expression_dimensions(threshold_left,resolved))
                and (_defer_synapses or have_same_dimensions(parse_expression_dimensions(threshold_left,resolved),group.variables['v'].dim)),
                'units',group,'threshold units must match voltage')
        chosen=list(requested.get(group.name,()))
        require(len(chosen)==len(set(chosen)) and set(chosen)<=names|set(runtime_parameters),'parameter',group,'trainable names must occur in equations or threshold')
        require(all(name in group.equations and group.equations[name].type=='parameter'
                    and group.variables[name].constant
                    and np.issubdtype(group.variables[name].dtype,np.floating) for name in chosen),
                'parameter',group,'trainable neuron parameters must be declared floating-point constants')
        require(all('linked' not in group.equations[name].flags for name in chosen),
                'parameter',group,'train the canonical source instead of a linked alias')
        streams=sorted(equations.stochastic_variables);noise_names.append(list(streams))
        calls={phase:[] for phase in ('refractory','update','threshold','reset')};neuron_draws.append(calls)
        names-=equations.subexpr_names
        require(len(streams)<=16,'noise',group,'at most 16 named noise streams per layer')
        require(not set(streams)&reset_names,'noise',group,'xi is defined only in differential equations')
        parameters={'dt':group_dt};parameter_values=[];shared_names=[]
        for name in sorted(names-set(names_of_states)-{'dt'}):
            if name in streams:continue
            var=resolved[name]
            if variable_ref is not None and name in ('not_refractory','lastspike'):
                parameters[name]=StateSlot(len(names_of_states)+(1 if name=='not_refractory' else 3),
                                           dtype='boolean' if name=='not_refractory' else 'float')
                continue
            if name=='t':
                parameters[name]=SimulationTime()
                continue
            if isinstance(var,TimedArray):
                try:parameters[name]=input_registry.resolve(var,group,name,input_bank,max_values=budget//48)
                except ValueError as error:raise TrainingConversionError('input',group,str(error)) from error
                continue
            if name in DEFAULT_FUNCTIONS:
                require(var is DEFAULT_FUNCTIONS[name],'function',group,'custom function replacements are unsupported')
                continue
            if isinstance(var,Function):
                try:parameters[name]=lower_pure_function(var)
                except ValueError as error:
                    try:
                        descriptor=lower_state_effect_function(var)
                        if descriptor.captured_arrays:
                            capture_bindings={key:next((field for field,array in capture_inputs.items() if same_capture_storage(array,value) and array.flags.writeable==value.flags.writeable),
                                next((field for field,var in readonly_inputs.items() if same_capture_storage(var.get_value(),value)),None)) for key,value in descriptor.captured_arrays}
                            require(all(capture_bindings.values()),'function',group,'mutable neuron captures require canonical state binding')
                            descriptor=bind_state_effect_captures(descriptor,capture_bindings,readonly=tuple(key for key,source in capture_bindings.items() if source in readonly_inputs))
                        parameters[name]=descriptor
                    except (ValueError,TypeError,OSError,SyntaxError,RecursionError) as effect_error:
                        raise TrainingConversionError('function',group,str(effect_error)) from error
                continue
            require(name!='xi' and hasattr(var,'get_value') and var.constant,
                    'coefficient',group,f'{name} must be a time-independent constant')
            if id(var) in constant_banks:
                parameters[name]=mapped_parameter(group,name,var)
                continue
            values=np.asarray(var.get_value()).reshape(-1)
            require(values.size>0 and values.dtype.kind in 'fiub' and np.all(np.isfinite(values)),
                    'coefficient',group,f'{name} must contain finite real values')
            indexed=not var.scalar and (name in chosen or not np.all(values==values[0]))
            if indexed:
                require(vector and values.size==len(group) and
                        (name=='i' or name in group.equations and group.equations[name].type=='parameter'),
                        'coefficient',group,f'{name} must be a declared per-neuron constant or i')
                admitted+=values.size*48+128
                require(admitted<=budget,'budget',group,'neuron parameter budget exceeded before list allocation')
                bank=len(weights);dtype=_state_dtype(var) if _defer_synapses else 'float';parameters[name]=NeuronParameter(bank,dtype=dtype)
                if dtype=='integer':integer_parameters.extend([bank,i] for i in range(len(values)))
                projections.append(neuron_parameter_bank(len(group),target_layer=layer));weights.append(values.astype(float).tolist())
                if name not in chosen:frozen_banks.add(bank)
                bindings.append(dict(bank=bank,object=group.name,variables=[name],kind='neuron_array',
                                     layout='neuron',trainable=name in chosen))
            elif name in chosen:
                shared_names.append(name);parameter_values.append(float(values[0]))
            else:parameters[name]=values[0].item() if _defer_synapses else float(values[0])
        for index,name in enumerate(shared_names):parameters[name]=(len(weights),index)
        for source,var in readonly_inputs.items():
            parameters[source]=NeuronParameter(constant_banks[id(var)],dtype=_state_dtype(var))
            resolved[source]=var
        def bind_draws(tree,phase):
            class Draw(ast.NodeTransformer):
                def visit_Assign(self,node):
                    # Named SDE increments retain their established streams.
                    if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in streams:return node
                    return self.generic_visit(node)
                def visit_Call(self,node):
                    if isinstance(node.func,ast.Name) and node.func.id in ('rand','randn','poisson'):
                        is_poisson=node.func.id=='poisson'
                        require(len(node.args)==(1 if is_poisson else 0) and not node.keywords,'noise',group,'invalid random function arguments')
                        require(resolved.get(node.func.id) is DEFAULT_FUNCTIONS[node.func.id],
                                'function',group,'custom random functions are unsupported')
                        index=len(noise_names[-1]);require(index<16,'noise',group,'neuron code exceeds 16 random streams')
                        name=f'_b2_explicit_draw_{index}'
                        parameters[name]=(PoissonNoise if is_poisson else UniformNoise if node.func.id=='rand' else NormalNoise)(index)
                        noise_names[-1].append(name);calls[phase].append(dict(kind=node.func.id,stream=index))
                        if is_poisson:return ast.copy_location(ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=[self.visit(node.args[0])],keywords=[]),node)
                        return ast.copy_location(ast.Name(id=name,ctx=ast.Load()),node)
                    return self.generic_visit(node)
            return Draw().visit(tree)
        typed_types=types if _defer_synapses else None
        if variable_ref is not None:
            try:
                dimensions=parse_expression_dimensions(variable_ref,resolved)
                boolean=is_boolean_expression(variable_ref,resolved)
                require(boolean or have_same_dimensions(dimensions,second),
                        'refractory',group,'refractory expression must be Boolean or have time units')
                bound_ref=ast.unparse(bind_draws(ast.parse(variable_ref,mode='eval'),'refractory'))
                context=dict(parameters,_b2_old_active=StateSlot(len(names_of_states)+1,'boolean'),
                             _b2_elapsed=StateSlot(len(names_of_states)+2,'integer'))
                expression=f'_b2_old_active or not ({bound_ref})' if boolean else f'_b2_elapsed >= int((({bound_ref}) + 0.001*dt)/dt)'
                program=compile_training_equation(expression,parameters=context,states=names_of_states,state_types=typed_types)
                program=_precise_refractory_clock(program,len(names_of_states)+3,len(names_of_states)+4,len(names_of_states)+5)
                refractory_expressions[-1]=dict(expression=variable_ref,kind='boolean' if boolean else 'duration',program=program,
                                                gradient='detached-hard-gate',duration_ticks='checked-int32')
            except TrainingConversionError:raise
            except Exception as error:raise TrainingConversionError('refractory',group,str(error)) from error
        threshold_dimensions=str(parse_expression_dimensions(threshold_left,resolved))
        threshold=bind_draws(threshold,'threshold')
        threshold_text=ast.unparse(threshold.comparators[0]) if ordered else ast.unparse(threshold)
        threshold_left=ast.unparse(threshold.left) if ordered else threshold_text
        threshold_effect=_uses_state_effect(ast.unparse(threshold),parameters)
        # Lower the threshold through the same bounded SSA compiler; the
        # current threshold binding supports one constant or parameter node.
        try:threshold_program=compile_training_equation(threshold_text,parameters=parameters,states=names_of_states if vector else None,state_types=typed_types) if ordered and not threshold_effect else []
        except ValueError as error:raise TrainingConversionError('expression',group,str(error)) from error
        if _defer_synapses and len(threshold_program)==1 and threshold_program[0]['op']=='integer_constant':
            threshold_program=[dict(op='constant',value=float(threshold_program[0]['value']))]
        # Keep the legacy path only for an immutable positive literal v > c.
        # Every dynamic parameter/expression threshold uses an explicit margin,
        # so learned thresholds may cross zero without an artificial constraint.
        literal=(ordered and len(threshold_program)==1 and threshold_program[0]['op']=='constant'
                 and threshold_program[0]['value']>0 and isinstance(threshold.left,ast.Name)
                 and threshold.left.id=='v' and isinstance(threshold.ops[0],ast.Gt))
        margin=None
        if _defer_synapses and not literal:
            if ordered and any(types[names_of_states.index(n.id)]=='integer' for n in ast.walk(threshold) if isinstance(n,ast.Name) and n.id in names_of_states):
                ordered=False;threshold_text=ast.unparse(threshold)
            if ordered:
                left,right=(threshold_left,threshold_text) if isinstance(threshold.ops[0],(ast.Gt,ast.GtE)) else (threshold_text,threshold_left)
                expression=f'({left})-({right})'
            else:expression=threshold_text
            try:
                if threshold_effect:
                    program,effect_outputs=_compile_effect_threshold(threshold,group,names_of_states,parameters,
                        {**group.variables,**resolved},types,predicate=not ordered,
                        slope=training_options.get('surrogate_slope',5.),scale=training_options.get('surrogate_scale',1.),physical_slots=state_aliases)
                else:
                    program=compile_training_equation(expression,parameters=parameters,states=names_of_states,state_types=typed_types) if ordered else compile_training_predicate(
                        expression,parameters=parameters,states=names_of_states,
                        slope=training_options.get('surrogate_slope',5.),scale=training_options.get('surrogate_scale',1.),state_types=typed_types)
            except ValueError as error:raise TrainingConversionError('threshold',group,str(error)) from error
            if ordered and program[-1]['op'].startswith('integer_'):
                ordered=False;expression=ast.unparse(threshold)
                program=compile_training_predicate(expression,parameters=parameters,states=names_of_states,state_types=typed_types,
                    slope=training_options.get('surrogate_slope',5.),scale=training_options.get('surrogate_scale',1.))
            margin=dict(program=program,expression=expression,dimensions=threshold_dimensions,
                        inclusive=ordered and isinstance(threshold.ops[0],(ast.GtE,ast.LtE)),predicate=not ordered)
            if threshold_effect:margin['effect_outputs']=effect_outputs
            if not ordered:margin['gradient']=('numpy-eager-product-equality-and-integer-detached' if threshold_effect
                                              else 'short-circuit-product-neutral-skipped-equality-detached')
            thresholds.append(1.);threshold_refs.append(None);threshold_indexed.append(False);threshold_mappings.append(None)
        else:
            require(len(threshold_program)==1 and threshold_program[0]['op'] in ('constant','parameter','neuron_parameter','mapped_parameter'),
                    'threshold',group,'threshold must be a scalar name or numeric literal')
            node=threshold_program[0]
            threshold_mappings.append(node.get('mapping'))
            threshold_indexed.append(node['op']=='neuron_parameter')
            if 'mapping' in node:
                indices=parameter_mappings[node['mapping']]['indices'];values=[weights[node['bank']][i] for i in indices]
                require(all(v>0 for v in values),'threshold',group,'all linked thresholds must be positive')
                value=values[0];threshold_refs.append([node['bank'],indices[0]])
            elif node['op']=='neuron_parameter':
                values=weights[node['bank']][node['index']:node['index']+len(group)]
                require(all(v>0 for v in values),'threshold',group,'all neuron thresholds must be positive')
                value=values[0];threshold_refs.append([node['bank'],node['index']])
            elif node['op']=='parameter':
                value=parameter_values[node['index']];threshold_refs.append([node['bank'],node['index']])
            else:value=node['value'];threshold_refs.append(None)
            require(value>0,'threshold',group,'threshold must be positive in SI voltage coordinates')
            thresholds.append(value)
        threshold_margins.append(margin)
        try:
            if vector:
                if _uses_state_effect('\n'.join(expressions.values()),parameters):
                    require(_defer_synapses and not isinstance(group,Subgroup)
                            and all(name in capture_inputs or not group.variables[name].scalar and group.variables.indices[name]=='_idx'
                                    for name in names_of_states),
                            'function',group,'integrator callback effects require canonical neuron arrays')
                    integration=bind_draws(ast.parse(_INTEGRATORS[method](group.equations,variables={**group.variables,**resolved})),'update')
                    effect_programs,effect_writes=_compile_effect_integrator(ast.unparse(ast.fix_missing_locations(integration)),
                        group,names_of_states,parameters,{**group.variables,**resolved},noise_names=streams,state_types=types,
                        refractory_index=len(names_of_states) if spec is not None else None,physical_slots=state_aliases)
                    vector_programs.append(effect_programs);integrator_effect_writes[layer-1]=effect_writes
                elif method=='euler' and not streams:
                    vector_programs.append([_compile_training_ast(
                        coerce_state_expression(bind_draws(ast.parse(name+'+dt*('+expressions[name]+')' if name in expressions else name,mode='eval').body,'update'),types[i]) if _defer_synapses else bind_draws(ast.parse(name+'+dt*('+expressions[name]+')' if name in expressions else name,mode='eval').body,'update'),
                        parameters=parameters,states=names_of_states,state_types=typed_types) for i,name in enumerate(names_of_states)])
                else:
                    integration=bind_draws(ast.parse(_INTEGRATORS[method](group.equations,variables={**group.variables,**resolved})),'update')
                    vector_programs.append(_compile_integrator(ast.unparse(ast.fix_missing_locations(integration)),
                        names_of_states,parameters,refractory_index=len(names_of_states) if spec is not None else None,noise_names=streams,state_types=typed_types))
                scalar_statements,vector_statements=make_statements(group.event_codes['spike'],{**group.variables,**resolved},np.float64,optimise=False)
                reset_statements=[*scalar_statements,*vector_statements]
                temporaries={stmt.var:('integer' if np.dtype(stmt.dtype).kind=='i' else 'boolean' if np.dtype(stmt.dtype).kind=='b' else 'float')
                    for stmt in reset_statements if stmt.var not in group.variables or isinstance(group.variables[stmt.var],Subexpression)}
                reset_code='\n'.join(f'{stmt.var} {"=" if stmt.op==":=" else stmt.op} {stmt.expr}' for stmt in reset_statements)
                reset=bind_draws(ast.parse(reset_code),'reset').body
                if _uses_state_effect(reset_code,parameters):
                    require(_defer_synapses,'function',group,'reset callback effects require dynamic training')
                    effect_programs,effect_writes,scalar_programs,capture_programs=_compile_effect_reset(
                        ast.unparse(ast.Module(body=reset,type_ignores=[])),group,names_of_states,
                        parameters,{**group.variables,**resolved},types,
                        gate_index=len(names_of_states)+(1 if spec is not None else 0)+(5 if variable_ref is not None else 0),
                        capture_states=set(capture_inputs),physical_slots=state_aliases)
                    vector_resets.append(effect_programs);reset_effect_writes[layer-1]=effect_writes
                    reset_scalar_programs[layer-1]=scalar_programs
                    reset_capture_programs[layer-1]=capture_programs
                else:
                    vector_resets.append(_compile_resets(reset,names_of_states,parameters,state_types=typed_types,temporary_types=temporaries))
            else:
                programs.append(compile_training_equation('v+dt*('+expression+')',parameters=parameters))
        except TrainingConversionError:raise
        except (ValueError,RecursionError,UnsupportedEquationsException) as error:raise TrainingConversionError('expression',group,str(error)) from error
        if parameter_values:
            projections.append(neuron_parameter_bank(len(parameter_values),target_layer=layer));weights.append(parameter_values)
            binding_names=shared_names
            bindings.append(dict(bank=len(weights)-1,object=group.name,variables=binding_names,kind='neuron'))
        value=np.asarray(group.variables['v'].get_value(),dtype=float)
        require(np.all(np.isfinite(value)),'initial',group,'initial membrane must be finite')
        initial.extend(value.tolist())
        for name in names_of_states:
            # Dynamic lowering binds linked names to canonical source storage
            # after every source exists. Alias slots are inert, including when
            # the index itself is mutable; never freeze its snapshot here.
            if _defer_synapses and name not in capture_inputs and 'linked' in group.user_equations[name].flags:
                initial_state.extend([0.]*len(group));continue
            values=np.asarray(capture_inputs[name] if name in capture_inputs else group.variables[name].get_value(),dtype=float).reshape(-1)
            if _defer_synapses and name not in capture_inputs:values=values[_storage_indices(group,name)]
            require(np.all(np.isfinite(values)),'initial',group,'all initial states must be finite')
            initial_state.extend(values.tolist())
        if spec is not None:
            identity=[dict(op='state',index=len(names_of_states))]
            vector_programs[-1].append(copy.deepcopy(identity));vector_resets[-1].append(identity)
            names_of_states.append('__refractory_ticks');types.append('float')
            initial_state.extend(counter.tolist())
    if _defer_synapses and not weights:
        projections.append(neuron_parameter_bank(1));weights.append([0.0]);frozen_banks.add(0)
    plan=lif_training_plan([len(g) for g in groups],backend=backend,mpi_ranks=mpi_ranks,
         projections=projections,equations=None if vector else programs,threshold=thresholds,threshold_parameters=threshold_refs,
         state_equations=vector_programs if vector else None,state_resets=vector_resets if vector else None,
         refractory=refractory if any(r is not None for r in refractory) else None,
         threshold_per_neuron=threshold_indexed if any(threshold_indexed) else None,
         clock=dict(origin=origin,dt=dt) if temporal else None,
         noise_streams=[len(names) for names in noise_names] if any(noise_names) else None,
         trainable=[bank not in frozen_banks for bank in range(len(weights))],
         reset=reset_kind,detach_reset=detach_reset,**training_options)
    provenance=dict(schema='b2-brian-training-lowering-v1',dt_seconds=dt,input_group=input_group.name,
        layer_names=[g.name for g in layers],bindings=bindings,voltage_dimensions=[str(g.variables['v'].dim) for g in layers],
        scope='multistate-explicit-zero-delay-static-additive-synapses' if vector else 'scalar-euler-zero-delay-static-additive-synapses',
        integrators=[g.state_updater.method_choice for g in layers],
        integrator_effect_writes=integrator_effect_writes,
        reset_effect_writes=reset_effect_writes,
        reset_scalar_programs=reset_scalar_programs,
        reset_capture_programs=reset_capture_programs,
        refractory_gradient=('expression-discrete-gate-stop-gradient' if any(r is not None for r in refractory_expressions) else 'fixed-discrete-gate-stop-gradient') if any(r is not None for r in refractory) else None,
        state_names=state_names,state_types=state_types,integer_parameters=integer_parameters,state_layout='layer-state-neuron',noise_names=noise_names,
        noise_gradient=('likelihood-score-poisson-and-pathwise-fixed-draws'
                        if any(c['kind']=='poisson' for phases in neuron_draws for calls in phases.values() for c in calls)
                        else 'pathwise-fixed-draws' if any(any(v.values()) for v in neuron_draws) else 'pathwise-fixed-normal-draws') if any(noise_names) else None,
        neuron_draws=neuron_draws)
    provenance['threshold_margins']=threshold_margins
    provenance['refractory_expressions']=refractory_expressions
    provenance['deferred_parameter_references']=deferred_parameters
    provenance['timed_inputs']=input_registry.entries
    if constant_sources:provenance.update(constant_sources=constant_sources,parameter_mappings=parameter_mappings,
                                         threshold_mappings=threshold_mappings)
    provenance['snapshot_sha256']=hashlib.sha256(canonical_bytes(dict(plan=plan,weights=weights,initial=initial,initial_state=initial_state,provenance=provenance))).hexdigest()
    return BrianTrainingBundle(plan,weights,initial,provenance,initial_state)


def _uses_state_effect(code,parameters):
    return any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
               and type(parameters.get(node.func.id)) is StateEffectFunction
               for node in ast.walk(ast.parse(code)))


def _compile_effect_threshold(threshold,group,states,parameters,variables,state_types,predicate=False,slope=5.,scale=1.,physical_slots=None):
    """Threshold loads borrow full arrays; comparison operands retain order."""
    if any(name in group.variables and (group.variables[name].scalar or group.variables.indices[name]!='_idx') for name in states):
        raise ValueError('threshold callback effects require canonical neuron arrays')
    arrays={name for name in parameters if name in variables and hasattr(variables[name],'scalar')
            and not variables[name].scalar}
    if any(group.variables.indices[name]!='_idx' for name in arrays):
        raise ValueError('threshold callback coefficients require canonical indices')
    draws={name for name,value in parameters.items() if type(value) in (NormalNoise,UniformNoise)}
    # Never reverse evaluation of lhs/rhs for <. A returned array alias on
    # the left still observes mutation while evaluating the right operand.
    result='_b2_effect_threshold_margin'
    if result in states or result in parameters:raise ValueError('reserved threshold effect result name')
    if predicate:
        class NumpyBoolean(ast.NodeTransformer):
            def visit_BoolOp(self,node):
                values=[self.visit(value) for value in node.values];value=values[0]
                for right in values[1:]:
                    value=ast.Call(func=ast.Name(id='_b2_logical_and' if isinstance(node.op,ast.And) else '_b2_logical_or',ctx=ast.Load()),args=[value,right],keywords=[])
                return value
            def visit_UnaryOp(self,node):
                if isinstance(node.op,ast.Not):return ast.Call(func=ast.Name(id='_b2_logical_not',ctx=ast.Load()),args=[self.visit(node.operand)],keywords=[])
                return self.generic_visit(node)
        code=f'{result}='+ast.unparse(ast.fix_missing_locations(NumpyBoolean().visit(copy.deepcopy(threshold))))
    else:
        left,right=ast.unparse(threshold.left),ast.unparse(threshold.comparators[0])
        difference='_b2_threshold_left-_b2_threshold_right' if isinstance(threshold.ops[0],(ast.Gt,ast.GtE)) else '_b2_threshold_right-_b2_threshold_left'
        code=f'_b2_threshold_left={left}\n_b2_threshold_right={right}\n{result}={difference}'
    names={name:i for i,name in enumerate(states)};names[result]=len(states)
    transform=compile_state_effect_transform(code,states=names,parameters=parameters,
        state_types={**dict(enumerate(state_types)),len(states):'float'},
        array_states=set(states),writable_states=set(states),
        array_parameters=arrays|draws,temporary_parameters=draws,
        parameter_types={name:_state_dtype(variables[name]) for name in arrays},
        array_callables={name for name,value in parameters.items() if type(value) is PoissonNoise},
        predicate_outputs={len(states):(slope,scale)} if predicate else None,
        physical_slots={**physical_slots,len(states):len(states)} if physical_slots is not None else None)
    compiled=dict(zip(transform['writes'],transform['programs']))
    return compiled.pop(len(states)),[dict(state=slot,program=program) for slot,program in compiled.items()]


def _compile_effect_reset(code,group,states,parameters,variables,state_types,gate_index=None,capture_states=(),physical_slots=None):
    """Reset reads are advanced-index copies, including read-only coefficients."""
    from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
    from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
    if any(group.variables[name].scalar or group.variables.indices[name]!='_idx' for name in set(states)-set(capture_states)):
        raise ValueError('reset callback effects require canonical neuron arrays')
    variables={**DEFAULT_FUNCTIONS,**variables}
    parameters=dict(parameters)
    guard='_b2_reset_executed'
    if guard in parameters or guard in states:raise ValueError('reserved reset execution guard')
    parameters[guard]=StateSlot(len(states) if gate_index is None else gate_index,'boolean')
    scalar,vector=make_statements(code,variables,np.float64,optimise=False)
    generator=NumpyCodeGenerator(variables,group.variables.indices,group,set(),NumpyCodeObject,
                                 group.resetter['spike'].name,'reset',allows_scalar_write=True)
    _,writes,_,_=generator.arrays_helper(vector)
    arrays={name for name in parameters if name in variables and hasattr(variables[name],'scalar')
            and not variables[name].scalar}
    if any(group.variables.indices[name]!='_idx' for name in arrays):
        raise ValueError('reset callback coefficients require canonical indices')
    draws={name for name,value in parameters.items() if type(value) in (NormalNoise,UniformNoise)}
    code='\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {s.expr}' for s in [*scalar,*vector])
    transform=compile_state_effect_transform(code,states={name:i for i,name in enumerate(states)},
        parameters=parameters,state_types=dict(enumerate(state_types)),array_states=set(states),
        writable_states=set(states),copied_array_states=set(states)-set(capture_states),
        array_parameters=arrays|draws,temporary_parameters=arrays|draws,
        parameter_types={name:_state_dtype(variables[name]) for name in arrays},
        array_callables={name for name,value in parameters.items() if type(value) is PoissonNoise},scalar_eager=True,eager_guard=guard,
        unconditional_states=capture_states,array_aliases={name:next(other for other in states if other in capture_states
            and physical_slots[states.index(other)]==physical_slots[states.index(name)]) for name in capture_states} if physical_slots is not None else None)
    compiled=dict(zip(transform['writes'],transform['programs']))
    if not set(transform['writes'])<={states.index(name) for name in writes if name in states}:
        raise ValueError('reset callback attempted a write outside explicit NumPy writeback')
    return ([compiled.get(i,[dict(op='integer_state' if dtype=='integer' else 'state',index=i)])
             for i,dtype in enumerate(state_types)], [states[i] for i in transform['writes']],transform['scalar_programs'],
             {states[i]:program for i,program in transform['unconditional_programs'].items()})


def _compile_effect_integrator(code, group, states, parameters, variables, noise_names=(), state_types=None,refractory_index=None,physical_slots=None):
    """Keep the generated NumPy integration block's copies and live references."""
    from brian2.codegen.generators.cython_generator import CythonCodeGenerator
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    parameters=dict(parameters)
    # The generated stages introduce builtins such as int(not_refractory)
    # which may be absent from the user's resolved equation namespace. Brian's
    # actual code object resolves these before constant/Boolean optimisation.
    variables={**DEFAULT_FUNCTIONS,**variables}
    scalar,vector=make_statements(code,variables,np.float64,optimise=True)
    generator=CythonCodeGenerator(variables,group.variables.indices,group,{'_idx'},CythonCodeObject,
                                  group.state_updater.name,'stateupdate',allows_scalar_write=True)
    _,writes,_,conditions=generator.arrays_helper(vector)
    guards={name:flag for name,flag in conditions.items() if name in writes}
    if guards:
        from .training_equations import RefractoryActive
        if refractory_index is None:raise ValueError('guarded integrator requires a refractory context')
        parameters.update({flag:RefractoryActive(refractory_index) for flag in guards.values()})
    def numpy_expression(statement):
        # NumPy evaluates both where arguments. In particular, a callback in
        # both simplified branches mutates its borrowed array twice, even if
        # every neuron is refractory. Match its statement translation before
        # interpreting effects, rather than selecting a branch in Python.
        if (statement.used_boolean_variables is not None
                and len(statement.used_boolean_variables)==1
                and np.dtype(statement.dtype).kind=='f'
                and statement.complexity_std>sum(statement.complexities.values())):
            branches={assignments[0][1]:expression for assignments,expression
                      in statement.boolean_simplified_expressions.items()}
            return f'_b2_where({statement.used_boolean_variables[0]}, {branches[True]}, {branches[False]})'
        return statement.expr
    tree=ast.parse('\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {numpy_expression(s)}' for s in [*scalar,*vector]))
    draws=[]
    class Draw(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in noise_names:
                stream=noise_names.index(node.targets[0].id);name='_b2_effect_normal_'+str(stream)
                parameters[name]=NormalNoise(stream);draws.append(name)
                class Normal(ast.NodeTransformer):
                    def visit_Call(self,call):
                        if not (isinstance(call.func,ast.Name) and call.func.id=='randn' and not call.args and not call.keywords):
                            raise ValueError('unexpected effectful stochastic integrator draw')
                        return ast.Name(id=name,ctx=ast.Load())
                node.value=Normal().visit(node.value)
            return node
    tree=Draw().visit(tree)
    draws=list(dict.fromkeys([*draws,*[name for name,value in parameters.items()
                                     if type(value) in (NormalNoise,UniformNoise)]]))
    arrays={name for name in parameters if name in variables and hasattr(variables[name],'scalar') and not variables[name].scalar and name not in guards.values()}
    if any(group.variables.indices[name]!='_idx' for name in arrays):
        raise ValueError('integrator callback effects require canonical array coefficients')
    transform=compile_state_effect_transform(ast.unparse(tree),states={name:i for i,name in enumerate(states)},
        parameters=parameters,array_states=set(states),writable_states=set(states),state_types=dict(enumerate(state_types)),
        array_parameters=arrays|set(draws),temporary_parameters=set(draws),
        parameter_types={name:_state_dtype(variables[name]) for name in arrays},
        array_callables={name for name,value in parameters.items() if type(value) is PoissonNoise},
        physical_slots=physical_slots if physical_slots is not None else {i:i for i in range(len(states))},writeback_order=tuple(writes),eager_limit=128,
        write_guards=guards,indexed_guard_reads=(set(states)|arrays|{s.var for s in vector if s.op==':='}) if guards else ())
    compiled=dict(zip(transform['writes'],transform['programs']))
    return ([compiled.get(i,[dict(op='integer_state' if state_types[i]=='integer' else 'state',index=i)]) for i in range(len(states))],
            [states[i] for i in transform['writes']])


def _compile_integrator(code, states, parameters, refractory_index=None, noise_names=(), state_types=None):
    """Compose Brian's built-in stage assignments as an immutable expression DAG.

    State writes use snapshots, so the final sequential assignments still read
    simultaneous old states. Reusing stage nodes bounds RK4 code growth without
    changing arithmetic ordering or introducing a new native opcode.
    """
    statements=ast.parse(code,mode='exec').body
    if len(statements)>256:raise ValueError('integration stage budget exceeded')
    parameters=dict(parameters)
    for i,name in enumerate(noise_names):parameters['_b2_normal_'+name]=NormalNoise(i)
    values={name:ast.Name(id=name,ctx=ast.Load()) for name in states}
    class Substitute(ast.NodeTransformer):
        def visit_Name(self,node):return values.get(node.id,node)
        def visit_Call(self,node):
            # Brian inserts this factor into every stage of flagged equations.
            # Use the old tick counter, with no derivative through the predicate.
            if (refractory_index is not None and isinstance(node.func,ast.Name)
                    and node.func.id=='int' and len(node.args)==1 and not node.keywords
                    and isinstance(node.args[0],ast.Name) and node.args[0].id=='not_refractory'):
                return ast.Name(id='_b2_refractory_active',ctx=ast.Load())
            return self.generic_visit(node)
    for statement in statements:
        if (not isinstance(statement,ast.Assign) or len(statement.targets)!=1
                or not isinstance(statement.targets[0],ast.Name)):
            raise ValueError('integration stages require simple assignments')
        name=statement.targets[0].id
        if name in noise_names:
            class Draw(ast.NodeTransformer):
                def visit_Call(self,node):
                    if not (isinstance(node.func,ast.Name) and node.func.id=='randn' and not node.args and not node.keywords):
                        raise ValueError('unexpected stochastic integrator draw')
                    return ast.Name(id='_b2_normal_'+name,ctx=ast.Load())
            statement.value=Draw().visit(statement.value)
        if name in parameters:raise ValueError('integration cannot overwrite a coefficient')
        values[name]=Substitute().visit(statement.value)
    return [_compile_training_ast(values[name],parameters=parameters,states=states,deduplicate=True,
                                 refractory_index=refractory_index,state_types=state_types) for name in states]


def _compile_resets(statements, states, parameters, state_types=None, temporary_types=None):
    """Compose Brian's sequential assignments into simultaneous final SSA.

    Each replacement is a snapshot of the expression before that statement;
    visiting replacements again would incorrectly make a=v; v=a recursive.
    """
    values={name:ast.Name(id=name,ctx=ast.Load()) for name in states}
    temporary_types={} if temporary_types is None else temporary_types
    # v4 physical states are floating point, but its reset locals can still
    # require Brian's integer/Boolean assignment conversions.
    if state_types is None and temporary_types:state_types=['float']*len(states)
    class Substitute(ast.NodeTransformer):
        def visit_Name(self,node):
            return copy.deepcopy(values[node.id]) if node.id in values else node
    for statement in statements:
        if isinstance(statement,ast.Assign) and len(statement.targets)==1:
            target=statement.targets[0];expression=statement.value
        elif isinstance(statement,ast.AugAssign) and isinstance(statement.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod)):
            target=statement.target
            expression=ast.BinOp(left=ast.Name(id=getattr(target,'id',''),ctx=ast.Load()),
                                 op=statement.op,right=statement.value)
            if isinstance(statement.op,ast.Mod):
                expression=ast.Call(func=ast.Name(id='_b2_mod',ctx=ast.Load()),args=[expression.left,expression.right],keywords=[])
        else:raise ValueError('resets require simple state assignments or +=, -=, *=, /=, //=, %=')
        if not isinstance(target,ast.Name) or target.id not in values and target.id not in temporary_types:
            raise ValueError('reset assignment target must be a differential state')
        expression=Substitute().visit(copy.deepcopy(expression))
        # Bound growth after each substitution, before another can expand it.
        if sum(1 for _ in ast.walk(expression))>1024:
            raise ValueError('composed reset expression exceeds its budget')
        if target.id in temporary_types:expression=coerce_state_expression(expression,temporary_types[target.id])
        elif state_types is not None:expression=coerce_state_expression(expression,state_types[states.index(target.id)])
        values[target.id]=expression
    return [compile_training_equation(ast.unparse(values[name]),parameters=parameters,states=states,state_types=state_types) for name in states]
