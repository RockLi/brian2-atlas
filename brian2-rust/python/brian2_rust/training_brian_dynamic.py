"""Brian scheduling to native ordered dynamic actions (v5).

Conversion snapshots objects and emits SSA only; no Python simulation or VJP.
"""
import ast
import copy
import hashlib
from collections.abc import Mapping

import numpy as np
from brian2 import Synapses, second
from brian2.groups.group import CodeRunner
from brian2.groups.neurongroup import SubexpressionUpdater
from brian2.groups.subgroup import Subgroup
from brian2.core.variables import ArrayVariable
from brian2.units.fundamentalunits import get_unit
from brian2.core.functions import DEFAULT_FUNCTIONS, timestep, Function
from brian2.core.clocks import Clock
from brian2.core.network import _check_multiple_summed_updaters
from brian2.codegen.translation import get_identifiers_recursively, make_statements
from brian2.equations.unitcheck import check_units_statements
from brian2.stateupdaters.base import StateUpdateMethod, UnsupportedEquationsException
from brian2.stateupdaters.exact import linear

from .training_inputs import (TimedInputRegistry, TimedArray, BatchTimedArray,
                              encode_external_state_values, sample_major_to_timed_columns)
from .training_functions import lower_pure_function
from .training_effects import (Effects, StateEffectFunction, lower_state_effect_function, bind_state_effect_captures, mutated_state_effect_captures, mutated_state_effect_caller_captures, capture_array_key, same_capture_storage, capture_view_indices,
                               compile_state_effect_transform)
from .protocol import canonical_bytes
from .training_brian import (TrainingConversionError, lower_brian_training,
                             _INTEGRATORS, _compile_effect_integrator, _storage_indices, _state_dtype, _runtime_index, _endpoint_parent, _endpoint_start,
                             _snapshot_clock_time)
from .training_dynamic import compile_dynamic_transform, dynamic_action
from .training_queue import pending_events
from .training_brian_events import prepare_effect_event
from .training_brian_regular import prepare_regular, is_regular, mutated_regular_constants, mutated_equation_constants
from .training_equations import (SimulationTime, ClockTime, NormalNoise, UniformNoise, PoissonNoise,
                                 neuron_parameter_bank, typed_parameter, _DeferredParameter, _TimedState)


def _with_refractory_gate(program, guard):
    """Replace fixed-counter stage factors with a detached expression gate."""
    if not any(n['op']=='refractory_active' for n in program):return program
    nodes=copy.deepcopy(guard);gate=len(nodes)-1;remap={}
    for old,node in enumerate(program):
        if node['op']=='refractory_active':remap[old]=gate;continue
        value=dict(node);fields=['rate','arg','left','right','low','high','condition','yes','no','slope','scale']
        if node['op'] in ('timed_parameter','parameter_gather','integer_parameter_gather'):
            fields+=['index']
            if node['op']=='timed_parameter':fields+=['time']
        for field in fields:
            if field in value:value[field]=remap[value[field]]
        nodes.append(value);remap[old]=len(nodes)-1
    return nodes[:remap[len(program)-1]+1]


def lower_brian_dynamic_training(network, *, input_group, layers,
                                 trainable_synapse_parameters=None, external_state_inputs=None,
                                 run_namespace=None, **options):
    """Snapshot clocked/event-driven synapses, pre/post paths and runtime weights.

    Synaptic ``w`` is trainable by default. Override with a mapping from Synapses
    names to lists of variables (an empty list freezes that object's parameters).
    An event-written variable is runtime state; training its initial value never
    overwrites carried state. Other selected constants remain optimizer slots.
    Values and equations use SI units. CPU supports local MPI; Metal/CUDA use
    the native v5 action ABI. Delayed paths and fixed/expression refractory are included.
    Standard clocked run_regularly and constant-over-dt updaters keep Brian's
    scalar/vector separation, chronological schedule and typed cached state,
    including uniform/normal draws reused throughout a timestep.
    Materialized selected Synapses objects can be source or target endpoints;
    pathway, summed and scheduled writes share their canonical physical state.
    ``external_state_inputs`` maps input-group physical field names to explicit
    Brian TimedArrays with matching units and columns. Reads sample the table
    at the consumer's clock; spikes remain supplied separately. Tables are
    shared across a batch and frozen optimizer banks with input VJPs. Source
    generators are outside this graph; imported fields cannot be written by
    its selected neurons or synapses. No initial-value fallback is used.
    """
    def require(ok, code, owner, message):
        if not ok: raise TrainingConversionError(code, owner, message)
    layers=list(layers);groups=[input_group,*layers]
    require(not {'_defer_synapses','_input_registry','_mutable_constant_ids','_mutable_capture_inputs','_readonly_capture_inputs','_capture_constant_ids','dynamic'}&options.keys(),'options',network,'reserved conversion option')
    external_sources={};sample_count=None
    require(external_state_inputs is None or isinstance(external_state_inputs,Mapping),
            'input',input_group,'external_state_inputs must map input-group field names to TimedArrays')
    for name,table in (external_state_inputs or {}).items():
        require(isinstance(name,str) and name in input_group.variables and not name.startswith('_'),
                'input',input_group,'external state requires a named physical input-group field')
        variable=input_group.variables[name]
        require(variable.owner==input_group,'input',input_group,
                'external state must be owned by the input group, not selected graph storage')
        require(isinstance(variable,ArrayVariable) and not variable.constant and not variable.read_only
                and (np.dtype(variable.dtype).kind=='f' or np.dtype(variable.dtype) in (np.dtype('int32'),np.dtype('bool'))),
                'input',input_group,'external state fields require mutable float, int32 or boolean physical storage')
        require(type(table) in (TimedArray,BatchTimedArray) and table.dim==variable.dim,
                'input',input_group,f'{name} requires an unmodified TimedArray with matching units')
        values=np.asarray(table.values);size=np.asarray(variable.get_value()).size
        per_sample=type(table) is BatchTimedArray
        shape=list(values.shape)
        require((values.ndim==2 and size==1 or values.ndim==3 and values.shape[2]==size) if per_sample else
                (values.ndim==1 and size==1 or values.ndim==2 and values.shape[1]==size),
                'input',input_group,f'{name} table columns must match its canonical physical storage')
        require(values.size>0 and np.all(np.isfinite(values)),
                'input',input_group,f'{name} table must contain finite nonempty values')
        require(id(variable) not in external_sources,'input',input_group,'external fields cannot duplicate physical storage')
        dtype=_state_dtype(variable)
        if per_sample:
            require(sample_count is None or sample_count==values.shape[0],
                    'input',input_group,'per-sample fields must have the same sample count')
            sample_count=values.shape[0]
            columns=sample_count*size*(2 if dtype=='integer' else 1)
            require(columns<=2**24,'input',input_group,'per-sample columns exceed exact GPU index precision')
            budget=options.get('max_tape_bytes',64*1024**2)
            require(type(budget) is int and 0<budget<=1024**3 and values.size*(2 if dtype=='integer' else 1)<=budget//48,
                    'budget',input_group,'per-sample table exceeds input memory budget')
            values=sample_major_to_timed_columns(values)
        if dtype!='float':
            try:encoded=encode_external_state_values(values,dtype)
            except ValueError as error:raise TrainingConversionError('input',input_group,str(error)) from error
            # Keep the caller's TimedArray unchanged, including uses as an
            # ordinary floating expression elsewhere in the same graph.
            encoded_table=TimedArray(encoded,dt=table.dt*second,name=table.name+'_external_discrete')
        elif per_sample:encoded_table=TimedArray(values*get_unit(table.dim),dt=table.dt*second,name=table.name+'_per_sample')
        else:encoded_table=table
        external_sources[id(variable)]=dict(name=name,table=encoded_table,size=size,dtype=dtype,shape=shape,
                                           samples=sample_count if per_sample else None)
    for group in layers:
        written={node.id for code in group.event_codes.values() for node in ast.walk(ast.parse(code))
                 if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}
        require(not any(name in group.variables and id(group.variables[name]) in external_sources for name in written),
                'input',group,'explicit external state fields are read-only in neuron resets')
    input_registry=TimedInputRegistry()
    pre_regular={};mutable_constant_ids=set()
    pre_owners=[*layers,*[o for o in network.sorted_objects if isinstance(o,Synapses)],
                *[o for o in network.sorted_objects if isinstance(o,Subgroup) and _endpoint_parent(o,layers) is not None]]
    for obj in network.sorted_objects:
        if is_regular(obj,pre_owners):
            try:
                spec=prepare_regular(obj,{} if run_namespace is None else dict(run_namespace),float(input_group.clock.dt_),require)
                spec['capture_owners']=pre_owners
                changed=mutated_regular_constants(spec)
            except TrainingConversionError:raise
            except Exception as error:raise TrainingConversionError('runner',obj,str(error)) from error
            pre_regular[obj.name]=spec;mutable_constant_ids.update(changed)
    probe_namespace={} if run_namespace is None else dict(run_namespace)
    for group in [*layers,*[o for o in network.sorted_objects if isinstance(o,Synapses)]]:
        method=getattr(group.state_updater,'method_choice',None)
        if isinstance(method,str) and method in _INTEGRATORS and group.equations.diff_eq_names:
            variables={**group.variables,**group.resolve_all(group.equations.identifiers-set(group.variables),run_namespace=probe_namespace)}
            try:
                generated=_INTEGRATORS[method](group.equations,variables=variables)
                mutable_constant_ids.update(mutated_equation_constants(group,generated,probe_namespace))
            except UnsupportedEquationsException:pass
        if group in layers:
            mutable_constant_ids.update(mutated_equation_constants(group,'_cond = '+group.events['spike'],probe_namespace,threshold=True))
    capture_constant_ids=set()
    capture_models=[*layers,*[obj for obj in network.sorted_objects if isinstance(obj,Synapses)]]
    capture_paths=[path for syn in capture_models if isinstance(syn,Synapses) for path in syn._pathways]
    delay_variables={id(path.variables['delay']):path.variables['delay'] for path in capture_paths}
    captured_delay_ids=set();writable_capture_keys=set();writable_capture_arrays={}
    capture_owners=[*capture_models,*capture_paths]
    for group in capture_models:
        codes=[expression.code for _,expression in group.equations.get_substituted_expressions()]
        if group in layers:codes.extend([group.events['spike'],group.event_codes['spike']])
        else:codes.extend(path.code for path in group._pathways)
        codes.extend(spec['runner'].abstract_code for spec in pre_regular.values() if spec['group']==group)
        called={node.func.id for code in codes for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        caller_modified=set()
        if isinstance(group,Synapses):
            from brian2.codegen.translation import analyse_identifiers
            for path in group._pathways:
                _,known,unknown=analyse_identifiers(path.code,group.variables,recursive=True)
                variables={**group.variables,**group.resolve_all(sorted(known|unknown),run_namespace=probe_namespace)}
                functions={}
                for name,value in variables.items():
                    if type(value) is not Function or name in DEFAULT_FUNCTIONS:continue
                    try:functions[name]=lower_state_effect_function(value)
                    except (ValueError,TypeError,OSError,SyntaxError,RecursionError):
                        try:functions[name]=lower_pure_function(value)
                        except (ValueError,TypeError,OSError,SyntaxError,RecursionError):pass
                if not any(type(value) is StateEffectFunction and value.captured_arrays for value in functions.values()):continue
                try:
                    spec=prepare_effect_event(group,path,path.code,variables)
                    caller_modified.update(mutated_state_effect_caller_captures(spec['code'],functions,variables,
                        scalar=spec['mode']=='scalar',reload=spec['mode']=='vectorised'))
                except (ValueError,TypeError,OSError,SyntaxError,RecursionError):pass
        for function_name in sorted(called-set(DEFAULT_FUNCTIONS)):
            variable=group.resolve_all({function_name},run_namespace=probe_namespace)[function_name]
            if type(variable) is not Function:continue
            try:
                descriptor=lower_state_effect_function(variable)
                modified=mutated_state_effect_captures(descriptor)
            except (ValueError,TypeError,OSError,SyntaxError,RecursionError):continue
            for capture_name,array in descriptor.captured_arrays:
                if array.flags.writeable:
                    writable_capture_keys.add(capture_array_key(array));writable_capture_arrays[capture_array_key(array)]=array
                for owner in capture_owners:
                    for field in owner.variables.values():
                        if isinstance(field,ArrayVariable) and capture_view_indices(field.get_value(),array) is not None:
                            if id(field) in delay_variables:
                                captured_delay_ids.add(id(field))
                                continue
                            if capture_name in modified or capture_array_key(array) in caller_modified:
                                require(not field.read_only,'function',group,'captured writes require writable physical Brian storage')
                                if field.constant:mutable_constant_ids.add(id(field))
                            elif field.constant and not field.read_only:capture_constant_ids.add(id(field))
    # Pathway delays are queue controls, not equation parameter banks. Regular
    # capture discovery can also find a Synapses alias of the same storage.
    captured_delay_ids.update(mutable_constant_ids & delay_variables.keys())
    mutable_constant_ids.difference_update(delay_variables)
    capture_constant_ids.difference_update(delay_variables)
    neuron_capture_inputs={};readonly_capture_inputs={};capture_identities={}
    for group in layers:
        codes=[expression.code for _,expression in group.equations.get_substituted_expressions()]+[group.events['spike'],group.event_codes['spike']]
        called={node.func.id for code in codes for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        fields={};readonly={}
        for function_name in sorted(called-set(DEFAULT_FUNCTIONS)):
            variable=group.resolve_all({function_name},run_namespace=probe_namespace)[function_name]
            if type(variable) is not Function:continue
            try:descriptor=lower_state_effect_function(variable)
            except (ValueError,TypeError,OSError,SyntaxError,RecursionError):continue
            for name,array in descriptor.captured_arrays:
                require(array.size==len(group),'function',group,'mutable capture columns must match whole-array neuron population')
                require(not array.flags.writeable or array.size==1 or array.strides[0]!=0,
                        'function',group,'writable repeated captures require whole event binding')
                capture_identities.setdefault(capture_array_key(array),'__mutable_neuron_capture_'+str(len(capture_identities)))
                field=next((field for owner in capture_owners for field in owner.variables.values()
                    if isinstance(field,ArrayVariable) and id(field) in capture_constant_ids and id(field) not in mutable_constant_ids
                    and same_capture_storage(field.get_value(),array)),None)
                source=capture_identities[capture_array_key(array)]
                if not array.flags.writeable:source=source.replace('__mutable_neuron_capture_','__readonly_neuron_capture_')
                if field is None:fields[source]=array
                else:readonly[source]=field
        neuron_capture_inputs[group.name]=fields;readonly_capture_inputs[group.name]=readonly
    bundle=lower_brian_training(network,input_group=input_group,layers=layers,
                                run_namespace=run_namespace,external_state_inputs=external_state_inputs,
                                _defer_synapses=True,_input_registry=input_registry,_mutable_constant_ids=mutable_constant_ids,_mutable_capture_inputs=neuron_capture_inputs,_readonly_capture_inputs=readonly_capture_inputs,_capture_constant_ids=capture_constant_ids,**options)
    plan=bundle.plan;weights=bundle.weights;provenance=bundle.provenance
    objects={o.name:o for o in [*layers,*network.sorted_objects]}
    constant_banks={id(objects[e['object']].variables[e['variable']]):e['bank'] for e in provenance.get('constant_sources',[])}
    namespace={} if run_namespace is None else dict(run_namespace)
    dt=float(input_group.clock.dt_);origin=_snapshot_clock_time(network,input_group.clock)
    require(np.isfinite(origin) and origin>=0,'clock',network,'invalid input snapshot time')
    buffered=any(float(g.clock.dt_)!=dt or float(g.thresholder['spike'].clock.dt_)!=dt for g in layers)
    plan['clock']=dict(origin=origin,dt=dt)
    # Include even monitor/inactive-object clocks: Brian's Network._clocks does.
    # Equal dt clocks share a schedule because before_run aligns all to net.t.
    clock_dts=[dt]
    for obj in network.sorted_objects:
        clock=obj.clock
        require(type(clock) is Clock,'clock',obj,'dynamic conversion requires standard Brian clocks')
        value=float(clock.dt_)
        require(clock.epsilon_dt==Clock.epsilon_dt and np.isfinite(value) and value>0,
                'clock',obj,'dynamic conversion requires standard finite Brian clocks')
        if clock._old_dt is not None and clock._old_dt!=value:
            _snapshot_clock_time(network,clock)
        if value not in clock_dts:clock_dts.append(value)
    require(len(clock_dts)<=256,'budget',network,'dynamic clock count exceeds 256')
    # Mirror Network.run's clock-set iteration order for exact minimum-time
    # ties. Different dt values make tolerance coalescing non-transitive.
    clock_order=[]
    for clock in {obj.clock for obj in network.sorted_objects}:
        index=clock_dts.index(float(clock.dt_))
        if index not in clock_order:clock_order.append(index)
    clocks=dict(start=float(network.t),dts=clock_dts,epsilon=float(Clock.epsilon_dt),order=clock_order)
    synapses=sorted((o for o in network.sorted_objects if isinstance(o,Synapses)),key=lambda s:s.name)
    regular_owners=[*layers,*synapses,*[o for o in network.sorted_objects
                    if isinstance(o,Subgroup) and _endpoint_parent(o,layers) is not None]]
    regular={}
    for obj in network.sorted_objects:
        if type(obj) not in (CodeRunner,SubexpressionUpdater):continue
        # This population supplies external spikes; its standard internal
        # updater is outside the trainable state graph, like its integrator.
        if (type(obj) is SubexpressionUpdater and obj.group==input_group
                and getattr(input_group,'subexpression_updater',None) is obj):continue
        if external_sources and is_regular(obj,[input_group]):
            try:spec=prepare_regular(obj,namespace,dt,require)
            except TrainingConversionError:raise
            except Exception as error:raise TrainingConversionError('input',obj,str(error)) from error
            require(all(id(spec['variables'][name]) in external_sources
                        for name in spec['scalar_writes']|spec['vector_writes']),
                    'input',obj,'external source runners may only write explicitly supplied fields')
            continue
        require(is_regular(obj,regular_owners),'runner',obj,'regular runner requires a selected neuron, subgroup or Synapses owner')
        try:regular[obj.name]=pre_regular.get(obj.name) or prepare_regular(obj,namespace,dt,require)
        except TrainingConversionError:raise
        except Exception as error:raise TrainingConversionError('runner',obj,str(error)) from error
    regular_written={id(spec['variables'][name]) for spec in regular.values()
                     for name in spec['scalar_writes']|spec['vector_writes']}
    # A declaration can be written through another object's endpoint alias.
    # Discover physical writes before allocating any Synapses state, so the
    # declaration order cannot turn a modulated weight into a constant bank.
    endpoint_written={id(updater.target_var) for syn in synapses for updater in syn.summed_updaters.values()}
    indexed_parameters={id(variable) for group in [*layers,*synapses]
                        for name,variable in group.variables.items() if _runtime_index(group,name)}
    for syn in synapses:
        for path in syn._pathways:
            endpoint_written.update(id(syn.variables[node.id]) for node in ast.walk(ast.parse(path.code))
                if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store) and node.id in syn.variables)
    require(not external_sources.keys() & (regular_written|endpoint_written),
            'input',network,'explicit external state fields are read-only in the training graph')
    scheduled_noise_domains={obj.name:len(layers)+len(synapses)+len(regular)+k
        for k,obj in enumerate(sorted([obj for syn in synapses
                    for obj in [*syn._pathways,*syn.summed_updaters.values()]],key=lambda o:o.name))}
    requested={} if trainable_synapse_parameters is None else dict(trainable_synapse_parameters)
    require(set(requested)<=set(s.name for s in synapses),'parameter',network,'unknown trainable Synapses name')
    initial=list(bundle.initial_state);initial_parameters=[None]*len(initial);detached=[False]*len(initial)
    cells={};neuron_offsets={};voltage=[];offset=0;neuron=0
    for group,names in zip(layers,provenance['state_names']):
        neuron_offsets[group.name]=neuron
        for k,name in enumerate(names):cells[group.name,name]=list(range(offset+k*len(group),offset+(k+1)*len(group)))
        voltage.extend(cells[group.name,'v']);offset+=len(names)*len(group);neuron+=len(group)
    # The rectangular v4 prefix remains ABI-compatible, but each logical name
    # resolves to one canonical physical variable. Unused alias slots are inert.
    storage={};unused=[];linked_targets=set()
    for group in [*layers,*synapses]:
        equations=group.user_equations if group in layers else group.equations
        linked_targets.update(id(group.variables[e.varname]) for e in equations.values() if 'linked' in e.flags)
    for group,names in zip(layers,provenance['state_names']):
        for name in names:
            if name.startswith('__'):continue
            variable=group.variables[name]
            if 'linked' in group.user_equations[name].flags:continue
            own=cells[group.name,name]
            if variable.scalar:
                unused.extend(own[1:]);own=own[:1]
                cells[group.name,name]=own*len(group)
            storage[id(variable)]=own
            if id(variable) in mutable_constant_ids:
                bank=constant_banks[id(variable)]
                if _state_dtype(variable)=='float':
                    for k,cell in enumerate(own):initial_parameters[cell]=[bank,k]
    indirect_cells={};constant_index_tables={};external_selector_caches={};external_selector_states=set()
    def address(group,name,j,table=None,seen=()):
        nonlocal mapping_bytes
        require(name not in seen and len(seen)<16,'linked',group,'cyclic or excessive linked index depth')
        variable=group.variables[name]
        if table is None:
            if id(variable) in external_sources:
                source=external_sources[id(variable)]
                require(source['dtype']=='integer','linked',group,'external address roots require int32 storage')
                clock=clock_dts.index(float(group.clock.dt_));key=(id(variable),clock)
                if key not in external_selector_caches:
                    size=source['size'];check_budget(size*64+128)
                    require(len(initial)+size<=1_000_000,'budget',group,'external selector storage exceeds state budget')
                    slots=list(range(len(initial),len(initial)+size))
                    initial.extend([0.]*size);initial_parameters.extend([None]*size);detached.extend([True]*size)
                    integer_states.update(slots)
                    external_selector_states.update(slots)
                    external_selector_caches[key]=dict(source=source,group=group,clock=clock,cells=slots)
                table=external_selector_caches[key]['cells']
            else:
                require(id(variable) in storage,'linked',group,
                        f'{name} must link to mutable storage in a selected neuron layer or Synapses object')
                table=storage[id(variable)]
        index=group.variables.indices[name]
        if index in ('0','_idx'):
            k=0 if index=='0' else j
            require(k<len(table),'linked',group,f'invalid physical index for {name}')
            return table[k]
        mapping=group.variables[index]
        require(np.issubdtype(mapping.dtype,np.integer),'linked',group,'linked indices require integer storage')
        if mapping.constant:
            key=(id(mapping),id(table))
            if key not in constant_index_tables:
                values=np.asarray(mapping.get_value()).reshape(-1)
                require(np.all(values>=0) and np.all(values<len(table)),'linked',group,f'invalid physical indices for {name}')
                cost=len(values)*16+128;check_budget(cost);mapping_bytes+=cost
                constant_index_tables[key]=[table[int(k)] for k in values]
            return address(group,index,j,constant_index_tables[key],(*seen,name))
        require(_state_dtype(mapping)=='integer','linked',group,'runtime indices require int32 storage')
        source=address(group,index,j,seen=(*seen,name))
        if isinstance(source,int):return dict(index=source,tables=[table],root_name=index)
        return dict(source,tables=[*source['tables'],table])

    def linked_cells(group,name):
        nonlocal mapping_bytes
        out=[];row=[];indirect_cells[group.name,name]=row
        for j in range(len(group)):
            r=address(group,name,j)
            cost=16 if isinstance(r,int) else 128+sum(24+8*len(t) for t in r['tables'])
            check_budget(cost);mapping_bytes+=cost
            row.append(r if isinstance(r,dict) else None)
            if not isinstance(r,dict):out.append(r);continue
            if (group.name,name) in cells:index=cells[group.name,name][j]
            else:
                check_budget(64);index=len(initial);initial.append(0.)
                initial_parameters.append(None);detached.append(True);unused.append(index)
            dtype=_state_dtype(group.variables[name])
            if dtype=='integer':integer_states.add(index)
            elif dtype=='boolean':binary_states.append(index)
            out.append(index)
        return out
    for group,names in zip(layers,provenance['state_names']):
        for name in names:
            if name.startswith('__') or 'linked' not in group.user_equations[name].flags:continue
            unused.extend(cells[group.name,name])
    for index in unused:initial[index]=0.;detached[index]=True
    activity={};condition_storage={};neuron_condition_indices={};refractory_age={};refractory_lastspike={};refractory_words={}
    refractory_expressions=provenance.get('refractory_expressions',[None]*len(layers))
    for layer,group in enumerate(layers):
        spec=plan.get('refractory',[None]*len(layers))[layer]
        if spec is None:continue
        counters=cells[group.name,'__refractory_ticks']
        extra=len(group)*(5 if refractory_expressions[layer] is not None else 1)
        require(len(initial)+extra<=1_000_000 and (len(initial)+extra)*64+sum(len(w)*48 for w in weights)<=plan['max_tape_bytes'],
                'budget',group,'refractory control storage exceeds state budget')
        activity[group.name]=list(range(len(initial),len(initial)+len(group)))
        condition_storage[id(group.variables['not_refractory'])]=activity[group.name]
        initial.extend(np.asarray(group.variables['not_refractory'].get_value(),dtype=float).tolist());initial_parameters.extend([None]*len(group));detached.extend([True]*len(group))
        for index in counters:detached[index]=True
        if refractory_expressions[layer] is not None:
            last=np.asarray(group.variables['lastspike'].get_value(),dtype=float)
            require(np.all(np.isfinite(last)),'refractory',group,'lastspike must be finite')
            elapsed=np.asarray(timestep(_snapshot_clock_time(network,group.clock)-last,float(group.clock.dt_)))
            require(np.all(elapsed>=0),'refractory',group,'lastspike must not be in the future')
            for table,values in ((refractory_age,np.minimum(elapsed,2**31-1)),(refractory_lastspike,last)):
                table[group.name]=list(range(len(initial),len(initial)+len(group)))
                initial.extend(values.astype(float).tolist());initial_parameters.extend([None]*len(group));detached.extend([True]*len(group))
            bits=last.astype(np.float64).view(np.uint64);refractory_words[group.name]=[]
            for word in (0,1):
                values=(bits>>(32*word)).astype(np.uint32).view(np.int32)
                indices=list(range(len(initial),len(initial)+len(group)));refractory_words[group.name].append(indices)
                initial.extend(values.astype(float).tolist());initial_parameters.extend([None]*len(group));detached.extend([True]*len(group))
        variables=group.resolve_all(group.equations.names|group.equations.identifiers|{'dt'},
                                    run_namespace=namespace,user_identifiers=set())
        update=_INTEGRATORS[group.state_updater.method_choice](group.equations,variables=variables)
        if refractory_expressions[layer] is not None:update=group.state_updater._get_refractory_code(namespace)+update
        indices={}
        for phase,code in [('update',update),('threshold',group.events['spike']),('reset',group.event_codes['spike'])]:
            index_name='not_refractory'
            for name in sorted(get_identifiers_recursively([code],group.variables)):
                condition=getattr(group.variables.get(name),'conditional_write',None)
                if condition is not None and condition.name=='not_refractory':index_name=name
            indices[phase]=index_name
        neuron_condition_indices[group.name]=indices
    programs=[];program_ids={};bindings=provenance['bindings'];syn_info={};summations=[]
    integer_states={k for row in refractory_age.values() for k in row}|{k for rows in refractory_words.values() for row in rows for k in row};binary_states=[k for row in activity.values() for k in row]
    integer_parameters=list(provenance.get('integer_parameters',[]))
    for identity,bank in constant_banks.items():
        if identity in mutable_constant_ids:
            var=next(v for g in [*layers,*synapses] for v in g.variables.values() if id(v)==identity)
            if _state_dtype(var)=='integer':
                integer_parameters.extend([bank,k] for k in range(len(weights[bank])) if [bank,k] not in integer_parameters)
    for group,names in zip(layers,provenance['state_names']):
        for name,dtype in zip(names,provenance['state_types'][layers.index(group)]):
            if dtype=='float' or name.startswith('__') and name not in neuron_capture_inputs[group.name] or name in group.user_equations and 'linked' in group.user_equations[name].flags:continue
            indices=set(cells[group.name,name])
            if dtype=='integer':integer_states.update(indices)
            else:binary_states.extend(indices)
            for index in indices:detached[index]=True
    queue_layout={};delay_paths=[];action_bytes=0;program_bytes=0;restart_rules={};constant_indices={}
    mapping_bytes=sum(len(m['indices'])*16+128 for m in provenance.get('parameter_mappings',[]))
    def check_budget(extra=0):
        require(len(initial)<=1_000_000 and len(initial)*64+sum(len(w)*48 for w in weights)+action_bytes+program_bytes+mapping_bytes+extra<=plan['max_tape_bytes'],
                'budget',network,'dynamic state/actions exceed memory budget')
    sample_slot=None
    if sample_count is not None:
        check_budget(64);sample_slot=len(initial)
        initial.append(0.);initial_parameters.append(None);detached.append(True);integer_states.add(sample_slot)
    spike_buffers=[];spike_buffer_layout={}
    if buffered:
        for group in layers:
            check_budget(len(group)*64)
            space=np.asarray(group.variables['_spikespace'].get_value());count=int(space[-1]);fired=space[:count]
            require(0<=count<=len(group) and len(set(fired.tolist()))==count and np.all(fired>=0) and np.all(fired<len(group)),
                    'spikes',group,'invalid warm spike buffer')
            values=np.zeros(len(group));values[fired]=1
            indices=list(range(len(initial),len(initial)+len(group)));spike_buffers.extend(indices);spike_buffer_layout[group.name]=indices
            initial.extend(values.tolist());initial_parameters.extend([None]*len(group));detached.extend([False]*len(group));binary_states.extend(indices)
    class Actions(list):
        clock=0
        def append(self,action):
            nonlocal action_bytes
            require(len(self)<1_000_000,'budget',network,'dynamic action count exceeds budget')
            trigger=action.get('trigger')
            if buffered and trigger is not None and not trigger.get('external') and not trigger.get('state'):
                cell=spike_buffers[trigger['index']];action['trigger']=dict(external=False,state=True,index=cell)
                require(len(action['reads'])<64,'budget',network,'buffered event context exceeds 64 slots')
                # Some control actions share the reads/writes list. Adding a
                # trigger operand must not turn it into another write target.
                action['reads']=[*action['reads'],cell]
            access=action.get('indirect',{})
            cost=256+len(action['reads'])*40+sum(128+sum(24+8*len(t) for t in r['tables'])
                for field in ('reads','writes') for r in access.get(field,{}).values())
            check_budget(cost)
            if self.clock:action["clock"]=self.clock
            super().append(action);action_bytes+=cost
    actions=Actions()
    def context_slots(programs):
        return {node.get('index',0) for program in programs for node in program
                if node['op'] in ('voltage','state','integer_state','refractory_active')}

    def attach_indices(action,descriptors,output_slots,names,write_descriptors=None):
        """Bind logical locals before writes; keep their Brian writeback order."""
        if not descriptors:return action
        used=context_slots(programs[action['program_set']])
        writes={}
        for output,slot in enumerate(output_slots):
            if slot not in descriptors:continue
            r=(write_descriptors or descriptors)[slot];root=names.get(r['root_name'])
            # External linked aliases keep inert rectangular prefix cells.
            # Writeback must use the sampled physical root, never that alias.
            if root is not None and r['index'] in external_selector_states and action['reads'][root]!=r['index']:
                root=None
            if root is None:
                root=len(action['reads']);action['reads'].append(r['index'])
            if root in output_slots and not r.get('snapshot_index'):index=dict(kind='output',slot=output_slots.index(root))
            else:index=dict(kind='read',slot=root);used.add(root)
            writes[str(output)]=dict(index=index,tables=r['tables'])
        reads={str(slot):dict(index=r['index'],tables=r['tables']) for slot,r in descriptors.items() if slot in used}
        require(len(action['reads'])<=64,'budget',network,'indexed context exceeds 64 slots')
        if reads or writes:action['indirect']=dict(reads=reads,writes=writes)
        return action

    def descriptors(group,names,j):
        return {slot:indirect_cells[group.name,name][j] for slot,name in enumerate(names)
                if (group.name,name) in indirect_cells and indirect_cells[group.name,name][j] is not None}

    def index_descriptor(group,name,j):
        row=indirect_cells.get((group.name,name))
        return None if row is None else row[j]

    def possible_writes(action):
        access=action.get('indirect',{}).get('writes',{})
        return {index for slot,fixed in enumerate(action['writes'])
                for index in (access[str(slot)]['tables'][-1] if str(slot) in access else [fixed])}

    def ordered_outputs(names,written,reads,indexed):
        ordered=[names[name] for name in sorted(written&names.keys())]
        if indexed:return ordered
        winners={reads[slot]:slot for slot in ordered if slot not in indexed}
        return [slot for slot in ordered if slot in indexed or winners[reads[slot]]==slot]
    try:_check_multiple_summed_updaters(network.sorted_objects)
    except Exception as error:raise TrainingConversionError('summed',network,str(error)) from error

    def program_set(value):
        nonlocal program_bytes
        key=canonical_bytes(value)
        if key not in program_ids:
            require(len(programs)<4096,'budget',network,'dynamic program set budget exceeded')
            cost=sum(len(program) for program in value)*128;check_budget(cost)
            program_ids[key]=len(programs);programs.append(value);program_bytes+=cost
        return program_ids[key]

    def parameter_bank(owner,name,values,trainable,kind):
        bank=len(weights);weights.append(values.tolist())
        plan['projections'].append(neuron_parameter_bank(len(values)))
        plan['masks'].append([1.]*len(values));plan['trainable'].append(trainable)
        bindings.append(dict(bank=bank,object=owner.name,variables=[name],kind=kind,trainable=trainable))
        return bank

    def parameter_literal(variable,value):
        dtype=_state_dtype(variable)
        return int(value) if dtype=='integer' else bool(value) if dtype=='boolean' else float(value)

    def input_bank(name,values):
        check_budget(values.size*48+128)
        require(len(weights)<256,'budget',network,'timed input bank count exceeded')
        # parameter_bank accepts Brian objects only for their stable names.
        from types import SimpleNamespace
        return parameter_bank(SimpleNamespace(name=name),'values',values.astype(float),False,'timed_input')

    external_reads=[];external_read_ids=set()
    def external_table(source,group,name):
        try:table=input_registry.resolve(source['table'],group,name,input_bank,max_values=plan['max_tape_bytes']//48)
        except ValueError as error:raise TrainingConversionError('input',group,str(error)) from error
        source['bank']=table.bank
        return table

    def external_parameter(group,name,j):
        nonlocal mapping_bytes
        source=external_sources[id(group.variables[name])]
        if 'indices' not in source:
            cost=source['size']*8+128;check_budget(cost);mapping_bytes+=cost
            source['indices']=list(range(source['size']))
        column=address(group,name,j,table=source['indices'])
        table=external_table(source,group,name)
        # Runtime address tables already live in runtime_parameter_layout.
        # Do not expand the same N-column table again for every read in JSON.
        reported_column=column if isinstance(column,int) else dict(state=column['index'],depth=len(column['tables']),runtime=True)
        read=dict(object=group.name,variable=name,source_variable=source['name'],bank=table.bank,
                  column=reported_column,clock=clock_dts.index(float(group.clock.dt_)))
        identity=(group.name,name,j if isinstance(column,dict) else column,read['clock'])
        if identity not in external_read_ids:
            check_budget(256);mapping_bytes+=256
            external_read_ids.add(identity);external_reads.append(read)
        if isinstance(column,dict):
            if 'column_bank' not in source:
                check_budget(source['size']*48+128)
                require(len(weights)<256,'budget',group,'external column routing bank count exceeded')
                source['column_bank']=parameter_bank(input_group,source['name'],np.arange(source['size']),False,'external_input_columns')
                integer_parameters.extend([source['column_bank'],k] for k in range(source['size']))
            key=(group.name,name)
            if key not in parameter_reference_ids:
                check_budget(128);mapping_bytes+=128
                parameter_reference_ids[key]=len(parameter_references)
                parameter_references.append(dict(object=group.name,variable=name,bank=source['column_bank'],kind='external_column'))
            column=_DeferredParameter(source['column_bank'],parameter_reference_ids[key],'integer')
        return _TimedState(table,read['clock'],column,source['dtype'],
                           sample_slot if source['samples'] is not None else None,
                           source['size'] if source['samples'] is not None else None)

    parameter_references=provenance.pop('deferred_parameter_references',[])
    check_budget(128*len(parameter_references));mapping_bytes+=128*len(parameter_references)
    parameter_reference_ids={(r['object'],r['variable']):i for i,r in enumerate(parameter_references)}
    parameter_addresses={};bank_tables={};gather_tables={}

    def deferred_parameter(group,name):
        nonlocal mapping_bytes
        key=(group.name,name)
        if key not in parameter_reference_ids:
            check_budget(128);mapping_bytes+=128
            parameter_reference_ids[key]=len(parameter_references)
            parameter_references.append(dict(object=group.name,variable=name,bank=constant_banks[id(group.variables[name])]))
        return _DeferredParameter(constant_banks[id(group.variables[name])],parameter_reference_ids[key],_state_dtype(group.variables[name]))

    def resolve_parameters(source,j,reads,indexed):
        """Replace private references with reads of pre-statement selectors.

        Extra integer context slots always snapshot physical state. They never
        refer to an SSA local mutated earlier in the user's statement sequence.
        """
        nonlocal mapping_bytes
        slots={};out=[]
        for program in source:
            nodes=[];remap={}
            for old,node in enumerate(program):
                if node['op']=='_sample_index':
                    key=('sample',node['index'])
                    if key not in slots:
                        slots[key]=len(reads);reads.append(node['index'])
                    nodes.append(dict(op='integer_state',index=slots[key]))
                elif node['op'] in ('_deferred_parameter','_deferred_integer_parameter'):
                    ref=node['reference'];entry=parameter_references[ref];bank=entry['bank']
                    key=(ref,j)
                    if bank not in bank_tables:
                        cost=len(weights[bank])*8+128;check_budget(cost);mapping_bytes+=cost
                        bank_tables[bank]=list(range(len(weights[bank])))
                    if key not in parameter_addresses:
                        group=objects[entry['object']]
                        resolved=address(group,entry['variable'],j,table=bank_tables[bank])
                        cost=16 if isinstance(resolved,int) else 128+sum(24+8*len(t) for t in resolved['tables'])
                        check_budget(cost);mapping_bytes+=cost;parameter_addresses[key]=resolved
                    r=parameter_addresses[key]
                    dtype='integer' if node['op']=='_deferred_integer_parameter' else 'float'
                    if isinstance(r,int):nodes.append(dict(op='integer_parameter' if dtype=='integer' else 'parameter',bank=bank,index=r))
                    else:
                        if ref not in slots:
                            slot=len(reads);reads.append(r['index']);slots[ref]=slot
                            if len(r['tables'])>1:indexed[slot]=dict(r,tables=r['tables'][:-1])
                        nodes.append(dict(op='integer_state',index=slots[ref]));selector=len(nodes)-1
                        table=r['tables'][-1]
                        if len(table)!=len(weights[bank]) or not all(k==i for i,k in enumerate(table)):
                            table_key=id(table)
                            if table_key not in gather_tables:
                                cost=len(table)*48+128;check_budget(cost)
                                require(len(weights)<256,'budget',network,'parameter routing bank count exceeded')
                                from types import SimpleNamespace
                                route=parameter_bank(SimpleNamespace(name=entry['object']),entry['variable'],np.asarray(table),False,'parameter_index_table')
                                integer_parameters.extend([route,k] for k in range(len(table)))
                                gather_tables[table_key]=route
                            nodes.append(dict(op='integer_parameter_gather',bank=gather_tables[table_key],index=selector));selector=len(nodes)-1
                        nodes.append(dict(op='integer_parameter_gather' if dtype=='integer' else 'parameter_gather',bank=bank,index=selector))
                else:
                    value=dict(node)
                    fields=['rate','arg','left','right','low','high','condition','yes','no','slope','scale']
                    if node['op'] in ('timed_parameter','parameter_gather','integer_parameter_gather'):
                        fields+=['index']
                        if node['op']=='timed_parameter':fields+=['time']
                    for field in fields:
                        if field in value:value[field]=remap[value[field]]
                    nodes.append(value)
                remap[old]=len(nodes)-1
            require(len(nodes)<=128,'budget',network,'resolved parameter program exceeds 128 nodes')
            out.append(nodes)
        require(len(reads)<=64,'budget',network,'parameter gather context exceeds 64 slots')
        return out

    def constant_mapping(group,name):
        nonlocal mapping_bytes
        key=(group.name,name)
        if key not in constant_indices:
            cost=len(group)*16+128;check_budget(cost);mapping_bytes+=cost
            constant_indices[key]=dict(object=group.name,variable=name,bank=constant_banks[id(group.variables[name])])
            if _runtime_index(group,name):constant_indices[key]['runtime']=True
            else:constant_indices[key]['indices']=_storage_indices(group,name).tolist()
        return constant_indices[key]

    def constant_parameter(group,name,j):
        entry=constant_mapping(group,name)
        return deferred_parameter(group,name) if entry.get('runtime') else typed_parameter(entry['bank'],entry['indices'][j],_state_dtype(group.variables[name]))

    def endpoint_owner(group,j,seen=()):
        if group in layers:return neuron_offsets[group.name]+j
        require(group in synapses and group.name not in seen and len(seen)<16,'endpoint',group,
                'synaptic state ownership must reach a selected neuron layer')
        target=_endpoint_parent(group.target,[*layers,*synapses])
        require(target is not None,'endpoint',group,'target must belong to a selected neuron or Synapses object')
        index=int(group.variables['_synaptic_post'].get_value()[j]) if len(group) else 0
        return endpoint_owner(target,index,(*seen,group.name))

    for domain,syn in enumerate(synapses,len(layers)):
        source_group=_endpoint_parent(syn.source,[*groups,*synapses]);target_group=_endpoint_parent(syn.target,[*layers,*synapses])
        require(source_group is not None and target_group is not None,'endpoint',syn,'endpoints must belong to selected populations or Synapses objects')
        require(len(syn)>0 or bool(syn.summed_updaters) or any(spec['group']==syn for spec in regular.values()),
                'topology',syn,'materialized connections, summed or regular updates are required')
        allowed=set(syn._pathways)|set(syn.summed_updaters.values())
        if syn.state_updater is not None:allowed.add(syn.state_updater)
        allowed.update(obj for obj in syn.contained_objects if obj.name in regular)
        require(set(syn.contained_objects)==allowed,'runner',syn,'additional synaptic runners are unsupported')
        eqs=syn.equations
        substitutions={name:ast.parse(expr.code,mode='eval').body for name,expr in eqs.get_substituted_expressions(include_subexpressions=True)
                       if name in eqs.subexpr_names}
        class Expand(ast.NodeTransformer):
            def visit_Name(self,node):
                return copy.deepcopy(substitutions[node.id]) if isinstance(node.ctx,ast.Load) and node.id in substitutions else node
        summed_codes={}
        for updater in syn.summed_updaters.values():
            summed_group=_endpoint_parent(updater.target,[*layers,*synapses])
            require(updater.active and updater.when=='groups' and summed_group is not None,'summed',updater,
                    'summed updates require an active groups runner targeting a selected neuron or Synapses object')
            suffix='_pre' if updater.target_varname.endswith('_pre') else '_post'
            name=updater.target_varname[:-len(suffix)]
            require(not updater.target_var.constant and not updater.target_var.scalar,
                    'summed',updater,'summed targets must be mutable per-element physical states')
            code=f'{updater.target_varname} += ({updater.expression.code})'
            try:
                resolved=dict(syn.variables);resolved.update(syn.resolve_all(updater.expression.identifiers,run_namespace=namespace))
                check_units_statements(code,resolved)
            except Exception as error:raise TrainingConversionError('units',updater,str(error)) from error
            summed_codes[updater.name]=ast.unparse(Expand().visit(ast.parse(code)))
            summations.append(dict(object=updater.name,synapses=syn.name,target=updater.target.name,variable=name,
                                   direction=suffix[1:],order=updater.order,expression=updater.expression.code))
        event_names=set() if syn.event_driven is None else set(syn.event_driven.diff_eq_names)
        mutable=set(eqs.diff_eq_names)|event_names|{name for name in eqs
                    if id(syn.variables[name]) in regular_written|endpoint_written}
        # An ordinary a:1 declaration becomes runtime storage if a drift
        # callback mutates its borrowed array. Inspect argument ownership with
        # the bounded symbolic interpreter; never run the Python callback.
        effect_arguments={}
        from .training_effects import ArrayCell
        effect_codes=[expression.code for _,expression in eqs.get_substituted_expressions()]
        syn_regular=[spec for spec in regular.values() if spec['group']==syn and spec['state_effects']]
        effect_codes.extend(statement.expr for spec in syn_regular for statement in [*spec['scalar'],*spec['vector']])
        ownership_used={node.id for code in effect_codes for node in ast.walk(ast.parse(code,mode='eval')) if isinstance(node,ast.Name)}
        ownership_used.update(statement.var for spec in syn_regular for statement in [*spec['scalar'],*spec['vector']])
        ownership_names=[name for name in sorted(set(eqs)&ownership_used) if name in syn.variables
                          and not syn.variables[name].scalar
                          and syn.variables.indices[name]=='_idx']
        ownership_fields={name:k for k,name in enumerate(ownership_names)}
        ownership=Effects(ownership_fields,{},array_states=set(ownership_fields),writable_states=set(ownership_fields),eager_limit=128,
                          state_types={slot:_state_dtype(syn.variables[name]) for name,slot in ownership_fields.items()})
        def bind_probe_captures(descriptor,engine):
            bindings={}
            for k,(name,array) in enumerate(descriptor.captured_arrays):
                source='__probe_capture_'+str(id(array))
                dtype='integer' if array.dtype==np.dtype('int32') else 'boolean' if array.dtype==np.dtype('bool') else 'float'
                if source not in engine.environment:
                    engine.environment[source]=ArrayCell(ast.Constant(1),True,-1-k,dtype=dtype)
                bindings[name]=source
            return bind_state_effect_captures(descriptor,bindings) if bindings else descriptor

        def borrowed_expression(expression):
            if isinstance(expression,ast.Name) and expression.id in ownership_fields:
                return ownership.environment[expression.id]
            if isinstance(expression,ast.Call) and isinstance(expression.func,ast.Name) and expression.func.id not in DEFAULT_FUNCTIONS:
                function=syn.resolve_all({expression.func.id},run_namespace=namespace)[expression.func.id]
                if type(function) is Function:
                    try:descriptor=lower_pure_function(function)
                    except ValueError:descriptor=lower_state_effect_function(function)
                    return ownership.call(bind_probe_captures(descriptor,ownership),[borrowed_expression(arg) for arg in expression.args])
            # Arithmetic and native input/draw calls allocate independent
            # values. Preserve their shape without evaluating numeric data.
            used={node.id for node in ast.walk(expression) if isinstance(node,ast.Name)}
            array=any(name in syn.variables and hasattr(syn.variables[name],'scalar') and not syn.variables[name].scalar for name in used)
            return ArrayCell(ast.Constant(1.)) if array else ast.Constant(1.)
        for code in effect_codes:
            for call in ast.walk(ast.parse(code,mode='eval')):
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name):continue
                function_name=call.func.id
                if function_name in DEFAULT_FUNCTIONS:continue
                if function_name not in effect_arguments:
                    try:function=syn.resolve_all({function_name},run_namespace=namespace)[function_name]
                    except Exception as error:raise TrainingConversionError('function',syn,str(error)) from error
                    modified=()
                    if type(function) is Function:
                        try:lower_pure_function(function)
                        except ValueError:
                            try:
                                descriptor=lower_state_effect_function(function)
                                slots={name:i for i,name in enumerate(descriptor.arguments)}
                                symbolic=Effects(slots,{},array_states=set(slots),writable_states=set(slots),eager_limit=128)
                                symbolic.call(bind_probe_captures(descriptor,symbolic),[symbolic.environment[name] for name in descriptor.arguments])
                                modified=tuple(sorted(slot for slot in symbolic.effect_writes if slot>=0))
                            except (ValueError,TypeError,OSError,SyntaxError,RecursionError) as error:
                                raise TrainingConversionError('function',syn,str(error)) from error
                    effect_arguments[function_name]=modified
                for position in effect_arguments[function_name]:
                    if position>=len(call.args):continue  # Exact arity is checked by the compiler.
                    argument=call.args[position]
                    ownership=Effects(ownership_fields,{},array_states=set(ownership_fields),writable_states=set(ownership_fields),eager_limit=128,
                                      state_types={slot:_state_dtype(syn.variables[name]) for name,slot in ownership_fields.items()})
                    try:value=borrowed_expression(argument)
                    except (ValueError,TypeError,SyntaxError,RecursionError) as error:raise TrainingConversionError('function',syn,str(error)) from error
                    borrowed=next((name for name,index in ownership_fields.items() if isinstance(value,ArrayCell) and value.origin==index),None)
                    if borrowed is not None:
                        variable=syn.variables[borrowed]
                        if (not variable.scalar and not variable.constant and not variable.read_only
                                and syn.variables.indices[borrowed]=='_idx'):
                            mutable.add(borrowed)
        # A scheduled temporary can retain a borrowed alias across statements.
        # Trace complete blocks using only authenticated descriptor bodies and
        # symbolic placeholders; this does not invoke callback Python or draw.
        for spec in syn_regular:
            functions={};parameters={};arrays=set();parameter_types={}
            for name,variable in spec['variables'].items():
                if name in ownership_fields or name in DEFAULT_FUNCTIONS:continue
                if type(variable) is Function:
                    try:parameters[name]=lower_pure_function(variable)
                    except ValueError:functions[name]=lower_state_effect_function(variable)
                elif hasattr(variable,'dtype'):
                    dtype=_state_dtype(variable);parameter_types[name]=dtype
                    parameters[name]=True if dtype=='boolean' else 1 if dtype=='integer' else 1.
                    if not variable.scalar:arrays.add(name)
            engine=Effects(ownership_fields,functions,parameters,array_states=set(ownership_fields),
                writable_states=set(ownership_fields),array_parameters=arrays,eager_limit=128,
                state_types={slot:_state_dtype(syn.variables[name]) for name,slot in ownership_fields.items()},
                parameter_types=parameter_types)
            class DrawPlaceholder(ast.NodeTransformer):
                def visit_Call(self,node):
                    if (isinstance(node.func,ast.Name) and
                            (node.func.id in ('rand','randn','poisson') or isinstance(spec['variables'].get(node.func.id),TimedArray))):
                        return ast.Call(func=ast.Name(id='_b2_owned_array',ctx=ast.Load()),
                                        args=[ast.Constant(1 if node.func.id=='poisson' else 1.)],keywords=[])
                    return self.generic_visit(node)
            block='\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {s.expr}' for s in [*spec['scalar'],*spec['vector']])
            engine.functions={name:bind_probe_captures(function,engine) for name,function in functions.items()}
            try:engine.transform(ast.unparse(DrawPlaceholder().visit(ast.parse(block))))
            except (ValueError,TypeError,SyntaxError,RecursionError) as error:raise TrainingConversionError('function',syn,str(error)) from error
            for slot in engine.effect_writes:
                if slot<0:continue
                name=ownership_names[slot];variable=syn.variables[name]
                if not variable.constant and not variable.read_only:mutable.add(name)
        path_codes={};path_delays={};path_pending={};path_runtime={};path_variables={}
        for path in syn._pathways:
            require(path.active and path.prepost in ('pre','post') and path.event=='spike' and path.when=='synapses',
                    'pathway',path,'only active spike pre/post pathways in the synapses slot are supported')
            path_dt=float(path.clock.dt_)
            require(path_dt==float(path.source.clock.dt_),'clock',path,'pathway clock must match its source clock')
            delays=np.asarray(path.delay[:],dtype=float).reshape(-1)
            require(len(delays) in (1,len(syn)) and np.all(np.isfinite(delays)) and np.all(delays>=0)
                    and np.all(delays/path_dt+.5<=1_000_000),'delay',path,'delays must be finite nonnegative values within the history budget')
            path_delays[path.name]=np.floor(np.broadcast_to(delays,(len(syn),))/path_dt+.5).astype(np.int64)
            try:path_pending[path.name]=pending_events(path.queue,path_dt,len(syn),max_bytes=plan['max_tape_bytes'])
            except Exception as error:raise TrainingConversionError('queue',path,str(error)) from error
            tree=ast.parse(path.code,mode='exec')
            for node in ast.walk(tree):
                if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):
                    require(node.id=='delay' or node.id not in path.variables or node.id in syn.variables,
                            'pathway',path,f'event writes to pathway-owned storage {node.id} require runtime lowering')
                    if node.id=='delay':
                        require(not path.variables['delay'].scalar,'pathway',path,
                                'Brian event pathways cannot write scalar/shared delay storage')
                if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store) and node.id in syn.variables:
                    require(not syn.variables[node.id].scalar and syn.variables.indices[node.id]!='0',
                            'pathway',path,'Brian event pathways cannot write scalar/shared storage')
                    require(not syn.variables[node.id].constant,'pathway',path,
                            'event writes to constant parameters or fixed index storage require runtime lowering')
            mutable|={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id in eqs}
            try:
                ids={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
                # Brian checks local temporaries with the same statement unit checker.
                resolved=dict(syn.variables);resolved['delay']=path.variables['delay']
                external=ids-set(resolved)-{n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
                resolved.update(syn.resolve_all(external|set(eqs.identifiers),run_namespace=namespace))
                check_units_statements(path.code,resolved)
            except Exception as error:raise TrainingConversionError('units',path,str(error)) from error
            if 'delay' in ids or id(path.variables['delay']) in captured_delay_ids:
                # Pathway storage is distinct for pre/post paths, even though
                # both use the same logical name in their source statements.
                require(len(initial)+len(syn)<=1_000_000,'budget',path,'pathway delay storage exceeds state budget')
                check_budget(len(syn)*64)
                values=(delays[:1] if path.variables['delay'].scalar and len(syn) else np.broadcast_to(delays,(len(syn),))).tolist()
                indices=list(range(len(initial),len(initial)+len(values)))
                initial.extend(values);initial_parameters.extend([None]*len(values));detached.extend([False]*len(values))
                restart_rules.update((index,dict(kind='initial')) for index in indices)
                path_runtime[path.name]={'delay':indices}
                storage[id(path.variables['delay'])]=indices
            path_variables[path.name]=dict(resolved)
            # Materialize Brian's subexpression caches before replacing random
            # calls. Expanding a stochastic subexpression at each use would
            # accidentally turn one cached draw into several independent draws.
            try:
                scalar,vector=make_statements(path.code,resolved,np.float64,optimise=False)
                path_codes[path.name]='\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {s.expr}'
                                               for s in [*scalar,*vector])
            except Exception as error:raise TrainingConversionError('pathway',path,str(error)) from error
        if event_names:mutable.add('lastupdate')
        chosen=list(requested.get(syn.name,['w'] if 'w' in eqs and 'linked' not in eqs['w'].flags else []))
        declarations=(set(eqs)-eqs.subexpr_names)|event_names
        require(len(chosen)==len(set(chosen)) and set(chosen)<=declarations,'parameter',syn,'trainable names must be synaptic declarations')
        source=np.asarray(syn.variables['_synaptic_pre'].get_value(),dtype=int)
        target=np.asarray(syn.variables['_synaptic_post'].get_value(),dtype=int)
        require(len(syn)*256+len(initial)*64+sum(len(w)*48 for w in weights)<=plan['max_tape_bytes'],
                'budget',syn,'dynamic topology exceeds memory budget')
        runtime={};params={};aliases=set();constant_links=set()
        link_indices={syn.variables.indices[eq.varname] for eq in eqs.values() if 'linked' in eq.flags}
        for name in sorted(declarations|mutable):
            variable=syn.variables[name]
            if name in link_indices and variable.constant and np.issubdtype(variable.dtype,np.integer):
                require(name not in chosen,'parameter',syn,'linked index storage is not differentiable')
                continue
            if name in link_indices and not variable.constant:mutable.add(name)
            if id(variable) in mutable_constant_ids:mutable.add(name)
            try:dtype=_state_dtype(variable)
            except ValueError as error:raise TrainingConversionError('state',syn,str(error)) from error
            require(name not in chosen or dtype=='float','parameter',syn,'trainable synaptic variables must be floating point')
            if name in eqs and 'linked' in eqs[name].flags:
                require(name not in chosen,'parameter',syn,'train the canonical source instead of a linked alias')
                if variable.constant:
                    constant_mapping(syn,name);constant_links.add(name)
                    continue
                aliases.add(name)
                continue
            if id(variable) in linked_targets and not variable.constant:mutable.add(name)
            values=np.asarray(variable.get_value(),dtype=float).reshape(-1)
            require(values.size==(1 if variable.scalar else len(syn)) and np.all(np.isfinite(values)),
                    'state',syn,f'invalid initial values for {name}')
            if id(variable) in constant_banks:
                bank=constant_banks[id(variable)];plan['trainable'][bank]=name in chosen
                next(binding for binding in bindings if binding['bank']==bank)['trainable']=name in chosen
            else:
                bank=None
                if len(values) and (name in chosen or name not in mutable and id(variable) in indexed_parameters):
                    require(len(weights)<256,'budget',syn,'canonical parameter bank count exceeded')
                    check_budget(values.size*48+128)
                    bank=parameter_bank(syn,name,values,name in chosen,'synapse_initial' if name in mutable else 'synapse_constant')
            if name in mutable:
                require(name=='lastupdate' or not variable.constant or id(variable) in mutable_constant_ids,'state',syn,'constant synaptic variables cannot be written')
                indices=list(range(len(initial),len(initial)+len(values)));runtime[name]=indices
                storage[id(variable)]=indices
                restart=dict(kind='time',clock=clock_dts.index(float(syn.clock.dt_))) if name=='lastupdate' else dict(kind='initial')
                restart_rules.update((index,restart) for index in indices)
                initial.extend(values.tolist());detached.extend([name=='lastupdate' or dtype!='float']*len(values))
                if dtype=='integer':integer_states.update(indices)
                elif dtype=='boolean':binary_states.extend(indices)
                initial_parameters.extend([None if bank is None or dtype!='float' else [bank,i] for i in range(len(values))])
            else:
                params[name]=(values,bank)
                if bank is not None:
                    constant_banks[id(variable)]=bank
                    if dtype=='integer':
                        known={tuple(item) for item in integer_parameters}
                        integer_parameters.extend([bank,j] for j in range(len(values)) if (bank,j) not in known)
        syn_info[syn.name]=dict(runtime=runtime,params=params,aliases=aliases,constant_links=constant_links,source=source,target=target,
            source_group=source_group,target_group=target_group,domain=domain)
        try:
            eqs.check_units(syn,run_namespace=namespace)
            if syn.event_driven is not None:syn.event_driven.check_units(syn,run_namespace=namespace)
        except Exception as error:raise TrainingConversionError('units',syn,str(error)) from error
        integrator_variables=dict(syn.variables)
        identifiers=set(eqs.identifiers)-set(eqs.stochastic_variables)
        if syn.event_driven is not None:identifiers.update(syn.event_driven.identifiers)
        try:integrator_variables.update(syn.resolve_all(identifiers,run_namespace=namespace))
        except Exception as error:raise TrainingConversionError('coefficient',syn,str(error)) from error
        continuous=''
        if eqs.diff_eq_names:
            method=syn.state_updater.method_choice
            methods=[method] if isinstance(method,str) else method
            require(isinstance(methods,(tuple,list)) and not syn.state_updater.method_options,
                    'integrator',syn,'select builtin synaptic integration methods without options')
            registry={**_INTEGRATORS,'exact':linear,'linear':linear}
            for method in methods:
                require(isinstance(method,str) and method in registry and StateUpdateMethod.stateupdaters.get(method) is registry[method],
                        'integrator',syn,'custom synaptic integrators are unsupported')
                try:continuous=registry[method](eqs,variables=integrator_variables);break
                except UnsupportedEquationsException:continue
            require(bool(continuous),'integrator',syn,'no selected integration method supports these equations')
        event_code=''
        if event_names:
            try:event_code=linear(syn.event_driven,variables=syn.variables)
            except Exception as error:raise TrainingConversionError('event-driven',syn,str(error)) from error
            class Elapsed(ast.NodeTransformer):
                def visit_Name(self,node):
                    return ast.parse('(t-lastupdate)',mode='eval').body if node.id=='dt' else node
            event_code=ast.unparse(Elapsed().visit(ast.parse(event_code)))
        syn_info[syn.name].update(continuous=continuous,event_code=event_code,path_codes=path_codes,summed_codes=summed_codes,
                                 path_delays=path_delays,path_pending=path_pending,path_runtime=path_runtime,
                                 integrator_variables=integrator_variables,path_variables=path_variables)

    # Resolve after every canonical source has been allocated: names and
    # network declaration order must not decide whether a cross-object link works.
    external_neuron_states={}
    for group,names in zip(layers,provenance['state_names']):
        for name in names:
            if not name.startswith('__') and 'linked' in group.user_equations[name].flags:
                if id(group.variables[name]) in external_sources:
                    # The rectangular ABI slot is already an inert detached
                    # alias. Actual reads come from the explicit table, never
                    # from this placeholder's initial/carried value.
                    external_neuron_states.setdefault(group.name,{})[names.index(name)]=name
                    continue
                cells[group.name,name]=linked_cells(group,name)
    for syn in synapses:
        info=syn_info[syn.name]
        for name in sorted(info['aliases']):info['runtime'][name]=linked_cells(syn,name)

    def external_neuron_programs(source,group,j):
        aliases=external_neuron_states.get(group.name,{})
        if not aliases:return source
        replacements={};out=[]
        for program in source:
            nodes=[];remap={}
            for old,node in enumerate(program):
                if node['op'] in ('state','integer_state') and node['index'] in aliases:
                    slot=node['index']
                    if slot not in replacements:
                        descriptor=external_parameter(group,aliases[slot],j)
                        replacements[slot]=compile_dynamic_transform('result=field',states={'result':0},
                            parameters={'field':descriptor},state_types={0:_state_dtype(group.variables[aliases[slot]])})['programs'][0]
                    part=replacements[slot];local={}
                    for k,value in enumerate(part):
                        copied=dict(value)
                        for field in ('arg','time','index','left','right'):
                            if field in copied and (field!='index' or copied['op']=='timed_parameter'):
                                copied[field]=local[copied[field]]
                        nodes.append(copied);local[k]=len(nodes)-1
                else:
                    value=dict(node);fields=['rate','arg','left','right','low','high','condition','yes','no','slope','scale']
                    if node['op'] in ('timed_parameter','parameter_gather','integer_parameter_gather'):
                        fields+=['index']
                        if node['op']=='timed_parameter':fields+=['time']
                    for field in fields:
                        if field in value:value[field]=remap[value[field]]
                    nodes.append(value)
                remap[old]=len(nodes)-1
            require(len(nodes)<=128,'budget',group,'external linked neuron program exceeds 128 nodes')
            out.append(nodes)
        return out

    def edge_mask(info,edge):
        if 'w' in info['aliases'] or 'w' in info['constant_links']:return None
        if 'w' in info['runtime']:
            ids=info['runtime']['w'];return initial_parameters[ids[0 if len(ids)==1 else edge]]
        if 'w' in info['params']:
            values,bank=info['params']['w']
            if bank is not None:return [bank,0 if len(values)==1 else edge]
        return None

    def physical_capture_cells(array):
        for selected in capture_owners:
            for var in selected.variables.values():
                if not isinstance(var,ArrayVariable) or id(var) not in storage:continue
                indices=capture_view_indices(var.get_value(),array)
                if indices is not None:return [storage[id(var)][index] for index in indices]
        return None

    def capture_addresses(array):
        return [(array.dtype.str,array.ctypes.data+j*array.strides[0]) for j in range(array.size)]

    mutable_capture_storage={};mutable_capture_layout=[];readonly_capture_layout=[];external_readonly_banks={};capture_address_cells={}
    def persistent_capture_cells(array):
        canonical=physical_capture_cells(array)
        addresses=capture_addresses(array)
        if canonical is not None:
            capture_address_cells.update(zip(addresses,canonical));return canonical
        # Repeated logical columns share one physical cell. Allocate each
        # missing address once; buffered vector writes choose the final column.
        seen=set(capture_address_cells)
        missing=[]
        for j,address in enumerate(addresses):
            if address not in seen:missing.append(j);seen.add(address)
        if missing:
            check_budget(len(missing)*64+128)
            require(len(missing)<=1_000_000-len(initial),'budget',network,'mutable capture state budget exceeded')
            dtype='integer' if array.dtype==np.dtype('int32') else 'boolean' if array.dtype==np.dtype('bool') else 'float'
            for j in missing:
                slot=len(initial);initial.append(float(array[j]));initial_parameters.append(None);detached.append(dtype!='float')
                if dtype=='integer':integer_states.add(slot)
                elif dtype=='boolean':binary_states.append(slot)
                capture_address_cells[addresses[j]]=slot
        return [capture_address_cells[address] for address in addresses]
    for group in layers:
        for source,var in readonly_capture_inputs[group.name].items():
            readonly_capture_layout.append(dict(runner=group.name,capture=source,bank=constant_banks[id(var)],dtype=_state_dtype(var),shape=list(var.get_value().shape)))
    for group in layers:
        for name,array in neuron_capture_inputs[group.name].items():
            key=capture_array_key(array);dtype='integer' if array.dtype==np.dtype('int32') else 'boolean' if array.dtype==np.dtype('bool') else 'float'
            slots=cells[group.name,name]
            canonical=physical_capture_cells(array)
            addresses=capture_addresses(array)
            if canonical is None:canonical=[capture_address_cells.get(address,slot) for address,slot in zip(addresses,slots)]
            for slot,target in zip(slots,canonical):
                if slot!=target:unused.append(slot);initial[slot]=0.;detached[slot]=True
            cells[group.name,name]=canonical;capture_address_cells.update(zip(addresses,canonical))
            if key not in mutable_capture_storage:
                mutable_capture_storage[key]=(canonical,dtype)
                mutable_capture_layout.append(dict(cells=canonical,dtype=dtype,shape=list(array.shape),aliases=[]))

    # Discover shared external storage before registering a readonly view.
    # Per-address cells also cover overlapping slices with no enclosing view.
    for array in writable_capture_arrays.values():
        if any(isinstance(var,ArrayVariable) and id(var) in constant_banks and id(var) not in storage
               and capture_view_indices(var.get_value(),array) is not None
               for selected in capture_owners for var in selected.variables.values()):continue
        persistent_capture_cells(array)


    def register_mutable_captures(descriptor,group,owner,function_name,whole_event=False):
        bindings={};fields={};readonly=[]
        for capture_name,array in descriptor.captured_arrays:
            require(whole_event or array.size==len(group),'function',owner,'mutable capture columns must match whole-array population')
            require(whole_event or not array.flags.writeable or array.size==1 or array.strides[0]!=0,
                    'function',owner,'writable repeated captures require whole event binding')
            key=capture_array_key(array);dtype='integer' if array.dtype==np.dtype('int32') else 'boolean' if array.dtype==np.dtype('bool') else 'float'
            variable=next((var for selected in capture_owners for var in selected.variables.values()
                if isinstance(var,ArrayVariable) and id(var) in constant_banks and id(var) not in storage
                and capture_view_indices(var.get_value(),array) is not None),None)
            if variable is not None:
                bank=constant_banks[id(variable)];source='__readonly_capture_bank_'+str(bank)
                indices=capture_view_indices(variable.get_value(),array)
                if indices!=list(range(variable.get_value().size)):
                    source+='_view_'+str(indices[0])+'_'+str(len(indices))
                    if len(indices)>1 and indices[1]-indices[0]!=1:
                        step=indices[1]-indices[0];source+='_step_'+('m' if step<0 else 'p')+str(abs(step))
                    from types import SimpleNamespace
                    variable=SimpleNamespace(dtype=array.dtype,scalar=False,constant=True,get_value=lambda value=array:value)
                    fields[source]=(None,dtype,bank,variable,indices)
                else:fields[source]=(None,dtype,bank,variable)
                bindings[capture_name]=source;readonly.append(capture_name)
                alias=dict(runner=owner.name,function=function_name,capture=capture_name,bank=bank,dtype=dtype,shape=list(array.shape),indices=indices)
                if alias not in readonly_capture_layout:readonly_capture_layout.append(alias)
                continue
            canonical=physical_capture_cells(array)
            if not array.flags.writeable and canonical is None and not any(address in capture_address_cells for address in capture_addresses(array)) and key not in mutable_capture_storage:
                if key not in external_readonly_banks:
                    check_budget(array.size*48+128)
                    require(len(weights)<256,'budget',owner,'readonly capture bank count exceeded')
                    bank=parameter_bank(owner,'__readonly_external_'+str(len(external_readonly_banks)),array.astype(float),False,'readonly_external_capture')
                    if dtype=='integer':integer_parameters.extend([bank,j] for j in range(array.size))
                    external_readonly_banks[key]=bank
                bank=external_readonly_banks[key];source='__readonly_capture_bank_'+str(bank)
                from types import SimpleNamespace
                variable=SimpleNamespace(dtype=array.dtype,scalar=False,constant=True,get_value=lambda value=array:value)
                fields[source]=(None,dtype,bank,variable);bindings[capture_name]=source;readonly.append(capture_name)
                alias=dict(runner=owner.name,function=function_name,capture=capture_name,bank=bank,dtype=dtype,shape=list(array.shape),storage='readonly-external-bank')
                if alias not in readonly_capture_layout:readonly_capture_layout.append(alias)
                continue
            if key not in mutable_capture_storage:
                canonical=persistent_capture_cells(array)
                require(len(canonical)==array.size,'function',owner,'captured Brian storage has incompatible physical shape')
                mutable_capture_storage[key]=(canonical,dtype)
                mutable_capture_layout.append(dict(cells=canonical,dtype=dtype,shape=list(array.shape),aliases=[]))
            slots,known_dtype=mutable_capture_storage[key]
            require(len(slots)==(array.size if whole_event else len(group)) and known_dtype==dtype,'function',owner,'shared capture storage has incompatible shape')
            name=('__mutable_capture_' if array.flags.writeable else '__readonly_state_capture_')+str(list(mutable_capture_storage).index(key))
            fields[name]=(slots,dtype);bindings[capture_name]=name
            entry=next(entry for entry in mutable_capture_layout if entry['cells']==slots)
            alias=dict(runner=owner.name,function=function_name,capture=capture_name,readonly=not array.flags.writeable)
            if alias not in entry['aliases']:entry['aliases'].append(alias)
        return bind_state_effect_captures(descriptor,bindings,readonly=tuple(readonly)),fields

    event_effect_actions={}
    event_effect_snapshots={}
    event_snapshot_all_reads={}
    event_effect_modes={}
    event_path_origins={}
    event_stage_groups={}
    batch_presence_stages={}
    batch_capture_gates={}
    batch_capture_programs={}
    batch_capture_checks={}
    batch_arrival_rows={}
    batch_filtered_rows={}
    batch_array_locals={}
    batch_random_fields={}
    batch_compacted_random={}
    batch_caller_locals={}
    batch_whole_locals={}

    def build_synaptic(syn,edge,code,*,trigger=None,streams=(),owner=None,pathway=None,
                       noise_domain=None,event_noise=None):
        info=syn_info[syn.name];pre=int(info['source'][edge]);post=int(info['target'][edge])
        original_pathway=event_path_origins.get(pathway,pathway)
        caller_batch=batch_caller_locals.get(original_pathway)
        caller_spec=event_effect_modes.get(pathway,{})
        caller_row=None
        def caller_destination(name):
            if caller_row is None or name not in caller_spec.get('local_outputs',{}):return None
            if name in caller_spec.get('local_physical_outputs',()):return address(syn,name,edge)
            return caller_row[caller_spec['local_outputs'][name]]
        if caller_batch is not None and pathway not in batch_presence_stages:
            key=('pending',event_noise['pending']) if event_noise and 'pending' in event_noise else ('new',edge)
            caller_row=caller_batch['rows'][key]
            if caller_spec.get('publish_caller_locals'):
                winners={address(syn,name,edge):caller_row[caller_batch['trace']['final'][name]] for name in caller_batch['writes']}
                destinations=list(winners);sources=list(winners.values())
                action=dict(owner=endpoint_owner(info['target_group'],post),reads=[*sources,*destinations],writes=destinations,
                            program_set=program_set([[dict(op='integer_state' if source in integer_states else 'state',index=k)] for k,source in enumerate(sources)]),threshold=None,trigger=trigger,mask=edge_mask(info,edge))
                if trigger is not None and trigger.get('state'):action['reads'].append(trigger['index'])
                actions.append(action);return action
        local_batch=batch_array_locals.get(original_pathway)
        local_row=None
        if local_batch is not None:
            key=('pending',event_noise['pending']) if event_noise and 'pending' in event_noise else ('new',edge)
            local_row=local_batch['rows'][key]
            if event_effect_modes.get(pathway,{}).get('publish_array_locals'):
                winners={}
                for name in local_batch['writes']:
                    origin,cache=local_row[name];winners[origin]=cache
                destinations=list(winners);sources=list(winners.values())
                reads=[*sources,*destinations]
                programs=[[dict(op='integer_state' if source in integer_states else 'state',index=j)] for j,source in enumerate(sources)]
                action=dict(owner=endpoint_owner(info['target_group'],post),reads=reads,writes=destinations,
                            program_set=program_set(programs),threshold=None,trigger=trigger,mask=edge_mask(info,edge))
                if trigger is not None and trigger.get('state'):reads.append(trigger['index'])
                actions.append(action);return action
        if pathway in batch_presence_stages:
            cell=batch_presence_stages[pathway]
            rows=batch_arrival_rows[pathway]
            key=('pending',event_noise['pending']) if event_noise and 'pending' in event_noise else ('new',edge)
            row=next(item['cell'] for item in rows if item['key']==key)
            random_batch=batch_random_fields.get(pathway)
            random_cells=[] if random_batch is None else list(random_batch['rows'][key].values())
            random_programs=[] if random_batch is None else [[dict(op='uniform_noise' if draw['kind']=='rand' else 'noise',stream=draw['stream'])] for draw in random_batch['draws']]
            action=dict(owner=endpoint_owner(info['target_group'],post),reads=[cell,row,*random_cells],writes=[cell,row,*random_cells],
                        program_set=program_set([[dict(op='constant',value=1.)],[dict(op='constant',value=1.)],*random_programs]),threshold=None,
                        trigger=trigger,detach_trigger=True,mask=edge_mask(info,edge))
            if random_batch is not None:
                action.update(noise_domain=scheduled_noise_domains[pathway],noise_entity=edge,
                              noise_streams=len(random_batch['draws']),event_noise=dict(event_noise))
            if trigger is not None and trigger.get('state'):action['reads'].append(trigger['index'])
            actions.append(action);return actions[-1]
        tree=ast.parse(code,mode='exec');used_names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        written_names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        synaptic_dt=float(syn.clock.dt_)
        reads=[];names={};indexed={};parameters={'dt':synaptic_dt,'t':ClockTime(clock_dts.index(synaptic_dt)),
            't_pre':ClockTime(clock_dts.index(float(syn.source.clock.dt_))),'t_post':ClockTime(clock_dts.index(float(syn.target.clock.dt_))),
            'dt_pre':float(syn.source.clock.dt_),'dt_post':float(syn.target.clock.dt_)}
        def bind(name,index,descriptor=None):
            if name not in used_names and not name.startswith('_b2_guard_'):return
            # Brian Cython reads each logical name into a distinct local before
            # running the path. Autapse aliases can share physical storage but
            # must not observe each other's local assignments immediately.
            if name not in names:
                names[name]=len(reads);reads.append(index)
                if local_row is not None and name in local_row:
                    require(descriptor is None,'function',syn,'batch local arrays require fixed addresses')
                    reads[-1]=local_row[name][1]
                if descriptor is not None:indexed[names[name]]=descriptor
        if caller_row is not None:
            for name,token in caller_spec.get('local_inputs',{}).items():
                bind(name,address(syn,name,edge) if name in caller_spec.get('accum',()) else caller_row[token])
        random_batch=batch_random_fields.get(original_pathway)
        if random_batch is not None:
            key=('pending',event_noise['pending']) if event_noise and 'pending' in event_noise else ('new',edge)
            for name,cell in random_batch['rows'][key].items():bind(name,cell)
        for name,indices in info['runtime'].items():
            descriptor=index_descriptor(syn,name,edge)
            bind(name,indices[0 if len(indices)==1 else edge],descriptor)
        for name,indices in info['path_runtime'].get(original_pathway,{}).items():bind(name,indices[0 if len(indices)==1 else edge])
        for group,j,suffix in [(info['source_group'],pre,'_pre'),(info['target_group'],post,'_post')]:
            if group is input_group or group in synapses:continue
            for name in provenance['state_names'][layers.index(group)]:
                if name in group.variables and id(group.variables[name]) in external_sources:continue
                descriptor=index_descriptor(group,name,j)
                if descriptor is not None:descriptor=dict(descriptor,root_name=descriptor['root_name']+suffix)
                if name+suffix in syn.variables:bind(name+suffix,cells[group.name,name][j],descriptor)
                if suffix=='_post' and name in syn.variables and name not in info['runtime'] and name not in info['params']:
                    bare=None if descriptor is None else dict(descriptor,root_name=descriptor['root_name'][:-len(suffix)])
                    bind(name,cells[group.name,name][j],bare)
        # Brian's alias index chain includes endpoint edge indices and any
        # nested aliases. Resolve the canonical variable rather than assuming
        # that every endpoint is a rectangular neuron layer.
        for name in sorted(used_names-names.keys()):
            variable=syn.variables.get(name)
            if variable is None or id(variable) not in storage:continue
            resolved=address(syn,name,edge)
            bind(name,resolved['tables'][-1][0] if isinstance(resolved,dict) else resolved,
                 resolved if isinstance(resolved,dict) else None)
        def bind_condition(name,variable,index_name):
            require(id(variable) in condition_storage,'refractory',syn,
                    'conditional writes require a selected standard refractory source')
            table=condition_storage[id(variable)]
            resolved=address(syn,index_name,edge,table=table)
            bind(name,resolved['tables'][-1][0] if isinstance(resolved,dict) else resolved,
                 resolved if isinstance(resolved,dict) else None)

        # Match create_runner_codeobj: resolve sorted identifiers, add condition
        # variables by name, then assign each condition's index from the last
        # referencing variable. Cython snapshots these flags before statements,
        # while writeback may use an integer local modified by those statements.
        referenced={name:syn.variables[name] for name in sorted(used_names) if name in syn.variables}
        conditions={};condition_indices={}
        for name,variable in referenced.items():
            condition=getattr(variable,'conditional_write',None)
            if condition is None:continue
            condition_indices[condition.name]=name
            if condition.name not in referenced or referenced[condition.name] is condition:
                conditions[condition.name]=condition
        resolved_conditions={**referenced,**conditions}
        # Include flags introduced by code generation: a user assignment to
        # that name becomes a physical condition write, not a local temporary.
        for name,variable in resolved_conditions.items():
            if name in used_names and id(variable) in condition_storage:
                bind_condition(name,resolved_conditions[name],condition_indices.get(name,name))
        guards={};mutable_guards=set()
        for name in sorted(written_names&names.keys()):
            if pathway in event_effect_modes and name not in event_effect_modes[pathway]['guarded_writes']:continue
            variable=syn.variables.get(name)
            condition=getattr(variable,'conditional_write',None)
            if condition is None:continue
            guard=condition.name if condition.name in written_names and condition.name in names else f'_b2_guard_{condition.name}'
            if guard not in names:
                bind_condition(guard,resolved_conditions[condition.name],condition_indices[condition.name])
            guards[names[name]]=names[guard]
            if guard==condition.name:mutable_guards.add(names[guard])
        for name,(values,bank) in info['params'].items():
            index=0 if len(values)==1 else edge
            parameters[name]=(int(values[index]) if _state_dtype(syn.variables[name])=='integer' else bool(values[index]) if _state_dtype(syn.variables[name])=='boolean' else float(values[index])) if bank is None else typed_parameter(bank,index,_state_dtype(syn.variables[name]))
        for name in sorted(info['constant_links']):
            parameters[name]=constant_parameter(syn,name,edge)
        if caller_row is not None:
            for name in names:parameters.pop(name,None)
        replay_noise=event_effect_modes.get(pathway,{}).get('replay_noise',{})
        for alias,draw in replay_noise.items():
            parameters[alias]=(PoissonNoise if draw['kind']=='poisson' else NormalNoise if draw['kind']=='randn' else UniformNoise)(draw['stream'])
        replay_parameter_arrays=set();replay_parameter_types={}
        used_names.update(event_effect_modes.get(pathway,{}).get('replay_guards',{}).values())
        for alias,entry in event_effect_modes.get(pathway,{}).get('replay_fields',{}).items():
            original=entry['variable'];variable=info['path_variables'][original_pathway].get(original)
            if id(variable) in condition_storage or original in syn.variables and id(syn.variables[original]) in storage:
                origin=(address(syn,entry.get('index_variable',original),edge,table=condition_storage[id(variable)])
                        if id(variable) in condition_storage else address(syn,original,edge))
                earlier=original_pathway if entry['stage']==0 else original_pathway+'::numpy-stage:'+str(entry['stage'])
                table={row['source']:row['cache'] for row in event_effect_snapshots[earlier]}
                require(type(origin) is int and origin in table,'function',syn,'staged replay requires a captured fixed input')
                bind(alias,table[origin])
            else:
                if original in parameters:parameters[alias]=parameters[original]
                else:
                    try:value=syn.resolve_all({original},run_namespace=namespace)[original]
                    except Exception as error:raise TrainingConversionError('function',syn,str(error)) from error
                    require(hasattr(value,'get_value') and value.scalar,'function',syn,'replayed coefficients require a resolved scalar or bank')
                    dtype=_state_dtype(value)
                    raw=np.asarray(value.get_value()).reshape(-1)[0]
                    parameters[alias]=int(raw) if dtype=='integer' else bool(raw) if dtype=='boolean' else float(raw)
                if variable is not None and not variable.scalar:
                    require(np.dtype(variable.dtype).kind in 'fb' or np.dtype(variable.dtype)==np.dtype('int32'),
                            'function',syn,'replayed coefficient arrays require float, int32 or boolean storage')
                    replay_parameter_arrays.add(alias)
                    replay_parameter_types[alias]=_state_dtype(variable)
        for target,flag in event_effect_modes.get(pathway,{}).get('replay_guards',{}).items():
            require(target in names and flag in names,'function',syn,'guarded replay requires captured target and mask')
            guards[names[target]]=names[flag]
        for k,name in enumerate(streams):parameters['__native_normal_'+name]=NormalNoise(k)
        class Draw(ast.NodeTransformer):
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in streams:
                    name=node.targets[0].id
                    class Replace(ast.NodeTransformer):
                        def visit_Call(self,call):
                            require(isinstance(call.func,ast.Name) and call.func.id=='randn' and not call.args and not call.keywords,
                                    'noise',syn,'unexpected stochastic integrator draw')
                            return ast.Name(id='__native_normal_'+name,ctx=ast.Load())
                    node.value=Replace().visit(node.value)
                return node
        tree=Draw().visit(tree)
        explicit_streams=[]
        class ExplicitDraw(ast.NodeTransformer):
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name) and node.func.id in ('rand','randn','poisson'):
                    is_poisson=node.func.id=='poisson'
                    require(len(node.args)==(1 if is_poisson else 0) and not node.keywords,'noise',syn,'invalid random function arguments')
                    resolved=syn.resolve_all({node.func.id},run_namespace=namespace)[node.func.id]
                    require(resolved is DEFAULT_FUNCTIONS[node.func.id],'function',syn,'custom random functions are unsupported')
                    index=event_effect_modes.get(pathway,{}).get('stream_offset',0)+len(streams)+len(explicit_streams)
                    require(index<16,'noise',syn,'synaptic code exceeds 16 random streams')
                    name='__native_explicit_random_'+str(index)
                    parameters[name]=(PoissonNoise if is_poisson else NormalNoise if node.func.id=='randn' else UniformNoise)(index)
                    explicit_streams.append(node.func.id)
                    if is_poisson:return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=[self.visit(node.args[0])],keywords=[])
                    return ast.Name(id=name,ctx=ast.Load())
                return self.generic_visit(node)
        tree=ExplicitDraw().visit(tree)
        locals_={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        whole_local_inputs=caller_spec.get('whole_local_inputs',{})
        whole_local_outputs=caller_spec.get('whole_local_outputs',{})
        carry_functions={}
        for local,entry in whole_local_inputs.items():
            if isinstance(entry,dict):parameters[local]=0.;continue
            function,capture=entry
            hidden='_b2_carry_function_'+function
            require(hidden not in used_names,'function',syn,'reserved physical caller function name')
            parameters[hidden]=lower_state_effect_function(info['path_variables'][original_pathway][function])
            parameters[local]=0.
            carry_functions[local]=(hidden,capture)
        needed={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}-names.keys()-parameters.keys()-locals_
        # Non-scalar neuron constants retain any optimizer bindings from the base lowerer.
        for name in sorted(needed):
            try:variable=syn.resolve_all({name},run_namespace=namespace)[name]
            except Exception as error:raise TrainingConversionError('coefficient',syn,str(error)) from error
            if id(variable) in external_sources:
                parameters[name]=external_parameter(syn,name,edge);continue
            if isinstance(variable,TimedArray):
                try:parameters[name]=input_registry.resolve(variable,syn,name,input_bank,max_values=plan['max_tape_bytes']//48)
                except ValueError as error:raise TrainingConversionError('input',syn,str(error)) from error
                continue
            if name in DEFAULT_FUNCTIONS:
                require(variable is DEFAULT_FUNCTIONS[name],'function',syn,'custom function replacements are unsupported');continue
            if isinstance(variable,Function):
                try:parameters[name]=lower_pure_function(variable)
                except ValueError as error:
                    try:parameters[name]=lower_state_effect_function(variable)
                    except (ValueError,TypeError,OSError,SyntaxError,RecursionError) as effect_error:
                        raise TrainingConversionError('function',syn,str(effect_error)) from error
                continue
            if name in ('i','j','N','N_pre','N_post'):
                parameters[name]=dict(i=pre-_endpoint_start(syn.source),j=post-_endpoint_start(syn.target),
                    N=len(syn),N_pre=len(syn.source),N_post=len(syn.target))[name];continue
            if id(variable) in constant_banks:
                parameters[name]=constant_parameter(syn,name,edge);continue
            canonical=next(((values,bank) for other in synapses
                for key,(values,bank) in syn_info[other.name]['params'].items() if variable is other.variables[key]),None)
            if canonical is not None:
                values,bank=canonical
                resolved=address(syn,name,edge,table=list(range(len(values))))
                require(isinstance(resolved,int),'coefficient',syn,'runtime-indexed endpoint parameters require a canonical parameter bank')
                parameters[name]=parameter_literal(variable,values[resolved]) if bank is None else typed_parameter(bank,resolved,_state_dtype(variable))
                continue
            require(hasattr(variable,'get_value') and variable.constant,'coefficient',syn,f'{name} requires a runtime state input that is not available')
            values=np.asarray(variable.get_value()).reshape(-1)
            require(values.size>0 and values.dtype.kind in 'fiub' and np.all(np.isfinite(values)),
                    'coefficient',syn,f'{name} must be a finite real constant')
            mapped=None;index=0;indexed_owner=False
            for group,j,suffix in [(info['source_group'],pre,'_pre'),(info['target_group'],post,'_post')]:
                original=name[:-len(suffix)] if name.endswith(suffix) else name
                if original in group.variables and variable is group.variables[original] and (name.endswith(suffix) or suffix=='_post'):
                    index=j;indexed_owner=True
                    for binding in bindings:
                        if binding['object']==group.name and original in binding['variables']:
                            mapped=(binding['bank'],j if binding['kind']=='neuron_array' else binding['variables'].index(original))
            if not variable.scalar and not indexed_owner:
                require(name in syn.variables,'coefficient',syn,f'{name} has an unsupported indexed shape')
                resolved=address(syn,name,edge,table=list(range(len(values))))
                require(isinstance(resolved,int),'coefficient',syn,'runtime-indexed endpoint constants require a canonical parameter bank')
                index=resolved;indexed_owner=True
            require(variable.scalar or indexed_owner and len(values)>index,'coefficient',syn,f'{name} has an unsupported indexed shape')
            parameters[name]=typed_parameter(*mapped,_state_dtype(variable)) if mapped is not None else values[0 if variable.scalar else index].item()
        capture_fields={};event_capture_slots=set();whole_local_fields={}
        for function_name,descriptor in list(parameters.items()):
            if type(descriptor) is not StateEffectFunction or not descriptor.captured_arrays:continue
            scalar_event=pathway in event_effect_modes and event_effect_modes[pathway]['mode']=='scalar'
            batch_event=pathway in batch_capture_gates
            require(scalar_event or batch_event or pathway is None and trigger is None and code==info['continuous'],
                    'function',syn,'event captures require selection-aware array binding')
            parameters[function_name],fields=register_mutable_captures(descriptor,syn,objects[original_pathway] if scalar_event or batch_event else syn.state_updater,function_name,whole_event=scalar_event or batch_event)
            capture_fields.update(fields)
            if scalar_event or batch_event:continue
            for name,field in fields.items():
                if field[0] is None:
                    parameters[name]=typed_parameter(field[2],field[4][edge] if len(field)>4 else edge,field[1])
                    info['integrator_variables'][name]=field[3]
                else:
                    slots,dtype=field;used_names.add(name);bind(name,slots[edge])
        for local,(function,capture) in carry_functions.items():
            whole_local_fields[local]=dict(parameters[function].capture_bindings)[capture]
        whole_output_fields={}
        buffers=batch_whole_locals.get(original_pathway,{})
        for local,entry in whole_local_inputs.items():
            if isinstance(entry,dict):
                source,field=buffers[entry['token']];capture_fields[source]=field;whole_local_fields[local]=source
        for local,token in whole_local_outputs.items():
            source,field=buffers[token];capture_fields[source]=field;whole_output_fields[local]=source
        try:
            if any(type(value) is StateEffectFunction for value in parameters.values()) or whole_local_fields or whole_output_fields:
                if pathway is not None:
                    spec=event_effect_modes[pathway]
                    guard_slots=set(guards.values())
                    require(not mutable_guards and not indexed,
                            'function',syn,'event callback arrays require fixed addresses and detached refractory guards')
                    array_names={name for name,slot in names.items() if slot not in guard_slots} if spec['mode']!='scalar' else set()
                    parameter_arrays={name for name in parameters if spec['mode']!='scalar' and name in syn.variables
                                      and hasattr(syn.variables[name],'scalar') and not syn.variables[name].scalar}
                    require(all(np.dtype(syn.variables[name].dtype).kind in 'fb' or np.dtype(syn.variables[name].dtype)==np.dtype('int32') for name in parameter_arrays),
                            'function',syn,'event callback copied coefficients require float, int32 or boolean arrays')
                    draws={name for name,value in parameters.items() if spec['mode']!='scalar' and type(value) in (NormalNoise,UniformNoise)}
                    guard_names={next(name for name,slot in names.items() if slot==target):next(name for name,slot in names.items() if slot==gate)
                                 for target,gate in guards.items()}
                    created=({node.id for node in ast.walk(tree) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}-set(names))&set(spec['indexed_locals'])
                    if capture_fields and spec['mode']!='scalar':
                        require(pathway in batch_capture_gates and not spec.get('replay_fields')
                                and not draws and not replay_noise and not any(type(value) is PoissonNoise for value in parameters.values()),
                                'function',syn,'batch captures require independent non-replayed statements')
                        require(not guard_names or len(tree.body)==1 and len(set(guard_names.values()))==1,
                                'function',syn,'masked batch captures require one statement selection')
                        # Whole capture writes precede the selected actions.
                        # Accumulators read their live physical destination;
                        # callback operands and capture expressions use the
                        # independent batch snapshot, even at the same address.
                        from .training_event_captures import compile_batch_event_captures
                        rows=batch_arrival_rows[original_pathway]
                        key=('pending',event_noise['pending']) if event_noise and 'pending' in event_noise else ('new',edge)
                        selected_row=next(i for i,row in enumerate(rows) if row['key']==key)
                        row_names=[]
                        for i,row in enumerate(rows):
                            name='_b2_event_arrival_'+str(i)
                            require(name not in names and name not in parameters,'function',syn,'reserved selected event row name')
                            names[name]=len(reads);reads.append(row['cell']);row_names.append(name)
                        def row_sum(selected):
                            expression=ast.Constant(0.)
                            for name in selected:
                                value=ast.Call(func=ast.Name(id='_b2_float',ctx=ast.Load()),args=[copy.deepcopy(name)],keywords=[]) if isinstance(name,ast.expr) else ast.Name(id=name,ctx=ast.Load())
                                expression=ast.BinOp(left=expression,op=ast.Add(),right=value)
                            return expression
                        selected_rows=row_names
                        selected_ordinals=None;selected_count=None
                        call_presence=None
                        if guard_names:
                            target=next(iter(guard_names));condition=syn.variables[target].conditional_write
                            if pathway not in batch_filtered_rows:
                                controls=[]
                                check_budget((2*len(rows)+1)*64)
                                require(len(initial)+2*len(rows)+1<=1_000_000,'budget',syn,'filtered callback state budget exceeded')
                                for row in rows:
                                    cell=address(syn,condition_indices[condition.name],row['edge'],table=condition_storage[id(resolved_conditions[condition.name])])
                                    require(isinstance(cell,int),'function',syn,'masked callback operands require fixed refractory addresses')
                                    effective=len(initial);initial.append(0.);initial_parameters.append(None);detached.append(True);binary_states.append(effective)
                                    prefix=len(initial);initial.append(0.);initial_parameters.append(None);detached.append(True)
                                    controls.append(dict(presence=row['cell'],mask=cell,effective=effective,prefix=prefix))
                                count=len(initial);initial.append(0.);initial_parameters.append(None);detached.append(True)
                                batch_filtered_rows[pathway]=dict(rows=controls,count=count)
                            filtered=batch_filtered_rows[pathway]
                            selected_rows=[]
                            selected_ordinals=[]
                            for i,row in enumerate(filtered['rows']):
                                name='_b2_event_mask_'+str(i)
                                require(name not in names and name not in parameters,'function',syn,'reserved selected event mask name')
                                names[name]=len(reads);reads.append(row['effective']);selected_rows.append(ast.Name(id=name,ctx=ast.Load()))
                                name='_b2_event_filtered_prefix_'+str(i)
                                require(name not in names and name not in parameters,'function',syn,'reserved event prefix name')
                                names[name]=len(reads);reads.append(row['prefix']);selected_ordinals.append(ast.Name(id=name,ctx=ast.Load()))
                            name='_b2_event_filtered_count'
                            require(name not in names and name not in parameters,'function',syn,'reserved event count name')
                            names[name]=len(reads);reads.append(filtered['count']);selected_count=ast.Name(id=name,ctx=ast.Load())
                            name='_b2_event_callback_presence'
                            require(name not in names and name not in parameters,'function',syn,'reserved callback presence name')
                            names[name]=len(reads);reads.append(batch_capture_gates[pathway])
                            call_presence=ast.Name(id=name,ctx=ast.Load())
                        row_operands=set()
                        def selected_vector_factory(vector_names,vector_reads,vector_parameters,vector_types,size):
                            require(original_pathway not in info['path_runtime'],'function',syn,
                                    'whole selected operands require fixed delay routing')
                            result={}
                            size=max(size,len(rows))
                            callback_operands={item.id for call in ast.walk(tree)
                                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name)
                                and type(parameters.get(call.func.id)) is StateEffectFunction
                                for argument in call.args for item in ast.walk(argument) if isinstance(item,ast.Name)}
                            actual_callback_operands=set(callback_operands)
                            if random_batch is not None:
                                # A whole returned expression can consume a
                                # cached draw outside the callback arguments.
                                callback_operands.update(item.id for item in ast.walk(tree) if isinstance(item,ast.Name) and isinstance(item.ctx,ast.Load))
                            if guard_names:
                                callback_operands.update(item.id for item in ast.walk(tree) if isinstance(item,ast.Name) and isinstance(item.ctx,ast.Load))
                            for operand in sorted((array_names|parameter_arrays)&callback_operands):
                                if operand not in syn.variables and (random_batch is None or operand not in random_batch['rows'][key]) and operand not in caller_spec.get('local_inputs',{}):continue
                                lane_values=[]
                                for lane,row in enumerate(rows):
                                    source='_b2_event_operand_'+operand+'_'+str(lane)
                                    require(source not in vector_names and source not in vector_parameters,'function',syn,'reserved batch operand name')
                                    if operand in parameter_arrays:
                                        variable=syn.variables[operand]
                                        if id(variable) in constant_banks:
                                            require(not constant_mapping(syn,operand).get('runtime'),'function',syn,'whole selected coefficients require fixed indices')
                                            vector_parameters[source]=constant_parameter(syn,operand,row['edge'])
                                        else:
                                            values=np.asarray(variable.get_value()).reshape(-1)
                                            index=address(syn,operand,row['edge'],table=list(range(len(values))))
                                            require(isinstance(index,int),'function',syn,'whole selected coefficients require fixed indices')
                                            entry=info['params'].get(operand)
                                            vector_parameters[source]=typed_parameter(entry[1],index,_state_dtype(variable)) if entry is not None and entry[1] is not None else values[index].item()
                                    else:
                                        cell=(caller_batch['rows'][row['key']][caller_spec['local_inputs'][operand]] if caller_batch is not None and operand in caller_spec.get('local_inputs',{})
                                              else random_batch['rows'][row['key']][operand] if random_batch is not None and operand in random_batch['rows'][row['key']]
                                              else address(syn,operand,row['edge']))
                                        require(isinstance(cell,int),'function',syn,'whole selected operands require fixed physical addresses')
                                        if local_batch is not None and operand in local_batch['rows'][row['key']]:
                                            cell=local_batch['rows'][row['key']][operand][1]
                                        vector_names[source]=len(vector_reads);vector_reads.append(cell)
                                        vector_types[vector_names[source]]='integer' if cell in integer_states else 'boolean' if cell in binary_states else 'float'
                                    lane_values.append(ast.Name(id=source,ctx=ast.Load()))
                                columns=[];count=copy.deepcopy(selected_count) if selected_count is not None else row_sum(selected_rows)
                                for column in range(size):
                                    # A single selected row broadcasts; otherwise
                                    # compact the actual arrival lanes by ordinal.
                                    wanted=ast.IfExp(test=ast.Compare(left=copy.deepcopy(count),ops=[ast.Eq()],comparators=[ast.Constant(1)]),body=ast.Constant(0),orelse=ast.Constant(column))
                                    value=ast.Name(id=operand,ctx=ast.Load())
                                    for lane in reversed(range(len(rows))):
                                        condition=ast.BoolOp(op=ast.And(),values=[
                                            copy.deepcopy(selected_rows[lane]) if isinstance(selected_rows[lane],ast.expr) else ast.Name(id=selected_rows[lane],ctx=ast.Load()),
                                            ast.Compare(left=copy.deepcopy(selected_ordinals[lane]) if selected_ordinals is not None else row_sum(selected_rows[:lane]),ops=[ast.Eq()],comparators=[copy.deepcopy(wanted)])])
                                        value=ast.IfExp(test=condition,body=copy.deepcopy(lane_values[lane]),orelse=value)
                                    columns.append(ast.Call(func=ast.Name(id='_b2_where',ctx=ast.Load()),args=[ast.Constant(1.),value,copy.deepcopy(value)],keywords=[]))
                                if random_batch is not None:
                                    from .training_equations import StateSlot,_compile_training_ast
                                    compact=[]
                                    for expression in columns:
                                        inputs=sorted({node.id for node in ast.walk(expression) if isinstance(node,ast.Name)}&vector_names.keys())
                                        context=[vector_reads[vector_names[name]] for name in inputs]
                                        bindings={**vector_parameters,**{name:StateSlot(slot,vector_types.get(vector_names[name],'float')) for slot,name in enumerate(inputs)}}
                                        program=_compile_training_ast(expression,parameters=bindings,states=['v'],typed=True,allow_select=True,deduplicate=True)
                                        check_budget(64);target=len(initial)
                                        require(target<1_000_000,'budget',syn,'compacted random state budget exceeded')
                                        dtype=_state_dtype(syn.variables[operand]) if operand in syn.variables else 'float'
                                        initial.append(0.);initial_parameters.append(None);detached.append(dtype!='float')
                                        if dtype=='integer':integer_states.add(target)
                                        if dtype=='boolean':binary_states.append(target)
                                        batch_compacted_random.setdefault(pathway,[]).append((target,context,program))
                                        source='_b2_compacted_random_'+str(target)
                                        vector_names[source]=len(vector_reads);vector_reads.append(target);vector_types[vector_names[source]]=dtype
                                        compact.append(ast.Name(id=source,ctx=ast.Load()))
                                    columns=compact
                                result[operand]=tuple(columns)
                                if guard_names and operand not in actual_callback_operands:row_operands.add(operand)
                            require(len(vector_reads)<=64,'budget',syn,'whole selected operands exceed 64 context slots')
                            return result
                        local_aliases={};canonical_locals={}
                        if caller_row is not None:
                            for name in sorted(array_names):
                                cell=reads[names[name]];source=canonical_locals.setdefault(cell,name)
                                if source!=name:local_aliases[name]=source
                        transform,names,event_capture_slots=compile_batch_event_captures(ast.unparse(tree),names,reads,parameters,capture_fields,
                            state_types={slot:'integer' if index in integer_states else 'boolean' if index in binary_states else 'float' for slot,index in enumerate(reads)},
                            array_names=array_names,parameter_arrays=parameter_arrays,
                            parameter_types={name:_state_dtype(syn.variables[name]) for name in parameter_arrays},reload=spec['mode']=='vectorised',typed_parameter=typed_parameter,
                            selected_output=(copy.deepcopy(selected_ordinals[selected_row]) if selected_ordinals is not None else row_sum(selected_rows[:selected_row]),copy.deepcopy(selected_count) if selected_count is not None else row_sum(selected_rows)),selected_vector_factory=selected_vector_factory,
                            selected_accumulators=set(spec['accum'])&array_names if spec['mode']=='vectorised' else (),
                            selected_guards=guard_names,selected_call_presence=call_presence,selected_row_operands=row_operands,
                            copied_array_aliases=local_aliases,retained_copied_outputs=caller_spec.get('retained_local_outputs',()),
                            whole_local_inputs=whole_local_fields,whole_local_outputs=whole_output_fields)
                        require(not any(length>1 for length in transform['selected_output_lengths']) or original_pathway not in info['path_runtime'],
                                'function',syn,'multi-column event returns require fixed delay routing')
                        if transform['unconditional_checks'] and pathway not in batch_capture_checks:
                            batch_capture_checks[pathway]=[(list(reads),program) for program in transform['unconditional_checks']]
                        for slot,program in transform['unconditional_programs'].items():
                            batch_capture_programs.setdefault(pathway,{})[reads[slot]]=(list(reads),program)
                    elif capture_fields:
                        require(spec['mode']=='scalar' and not spec.get('replay_fields'),
                                'function',syn,'scalar event captures require non-replayed FIFO statements')
                        from .training_event_captures import compile_scalar_event_captures
                        transform,names,event_capture_slots=compile_scalar_event_captures(ast.unparse(tree),names,reads,parameters,capture_fields,
                            state_types={slot:'integer' if index in integer_states else 'boolean' if index in binary_states else 'float' for slot,index in enumerate(reads)},
                            typed_parameter=typed_parameter,write_guards=guard_names)
                    else:transform=compile_state_effect_transform(ast.unparse(tree),states=names,parameters=parameters,
                        array_states=array_names,writable_states=array_names,
                        state_types={slot:'integer' if index in integer_states else 'boolean' if index in binary_states else 'float' for slot,index in enumerate(reads)},
                        array_parameters=parameter_arrays|draws|replay_parameter_arrays,temporary_parameters=parameter_arrays|draws|replay_parameter_arrays,
                        array_callables={name for name,value in parameters.items() if type(value) is PoissonNoise and spec['mode']!='scalar'},
                        parameter_types={**{name:_state_dtype(syn.variables[name]) for name in parameter_arrays},**replay_parameter_types},
                        copied_array_states=array_names,reload_arrays_each_statement=spec['mode']=='vectorised',eager_limit=128,
                        write_guards=guard_names,scalar_write_guards=spec['mode']=='scalar',
                        indexed_guard_reads=(array_names|parameter_arrays|draws|replay_parameter_arrays|created) if spec['mode']!='scalar' else ())
                else:
                    require(pathway is None and trigger is None and bool(info['continuous'])
                            and code==info['continuous'] and not guards and not indexed
                            and all(name in capture_fields or name in info['runtime'] and not syn.variables[name].scalar
                                    and syn.variables.indices[name]=='_idx'
                                    for name in names),
                            'function',syn,'synaptic integrator callback effects require unguarded canonical own arrays')
                    canonical={};physical_slots={slot:canonical.setdefault(cell,slot) for slot,cell in enumerate(reads)}
                    programs,effect_writes=_compile_effect_integrator(ast.unparse(tree),syn,list(names),parameters,
                        info['integrator_variables'],noise_names=streams,state_types=[capture_fields[name][1] if name in capture_fields else _state_dtype(syn.variables[name]) for name in names],physical_slots=physical_slots)
                    locals_.update(effect_writes)
                    output_slots=[names[name] for name in effect_writes]
                    transform=dict(writes=output_slots,programs=[programs[slot] for slot in output_slots],context_size=len(reads))
            else:
                transform=compile_dynamic_transform(ast.unparse(tree),states=names,parameters=parameters,write_guards=guards,mutable_write_guards=mutable_guards,state_types={slot:'integer' if index in integer_states else 'boolean' if index in binary_states else 'float' for slot,index in enumerate(reads)})
        except (ValueError,RecursionError) as error:raise TrainingConversionError('expression',syn,str(error)) from error
        if pathway in event_effect_modes:
            replay_names=set(event_effect_modes[pathway].get('replay_fields',{}))
            keep=[(slot,program) for slot,program in zip(transform['writes'],transform['programs'])
                  if not any(name in replay_names and value==slot for name,value in names.items())]
            transform['writes']=[slot for slot,_ in keep];transform['programs']=[program for _,program in keep]
        # Brian's Cython writeback is sorted by logical variable name. If two
        # names write one cell, retain the final write but preserve both locals
        # in every expression and accumulate all aliased read adjoints natively.
        if indexed:
            by_slot=dict(zip(transform['writes'],transform['programs']))
            outputs=[(slot,by_slot[slot]) for slot in ordered_outputs(names,locals_,reads,indexed)]
        else:
            winners={}
            capture_names=[name for name,slot in names.items() if slot in event_capture_slots]
            locals_.update(capture_names)
            locals_.update(caller_spec.get('retained_local_outputs',()))
            order=([*capture_names,*event_effect_modes[pathway]['writes']] if pathway in event_effect_modes else sorted(locals_&names.keys()))
            available=set(transform['writes'])
            for name in order:
                if name in names and name in locals_ and names[name] in available:
                    destination=caller_destination(name);target=reads[names[name]] if destination is None else destination
                    winners[target]=names[name]
            output_targets={slot:caller_destination(name) for name,slot in names.items() if caller_destination(name) is not None}
            outputs=[(slot,program) for slot,program in zip(transform['writes'],transform['programs']) if winners[output_targets.get(slot,reads[slot])]==slot]
        transform['writes']=[slot for slot,_ in outputs];transform['programs']=[program for _,program in outputs]
        transform['programs']=resolve_parameters(transform['programs'],edge,reads,indexed)
        transform['context_size']=len(reads)
        mask=edge_mask(info,edge)
        action=dynamic_action(transform,reads,owner=endpoint_owner(info['target_group'],post) if owner is None else owner,
                                      program_set=program_set(transform['programs']),trigger=trigger,
                                      parameter_index=edge,
                                      noise_domain=(noise_domain if noise_domain is not None else scheduled_noise_domains.get(original_pathway,info['domain'])) if explicit_streams or replay_noise else info['domain'],
                                      noise_entity=edge,noise_streams=max(event_effect_modes.get(pathway,{}).get('stream_offset',0)+len(streams)+len(explicit_streams) if explicit_streams or streams else 0,
                                          max((draw['stream']+1 for draw in replay_noise.values()),default=0)),mask=mask)
        if caller_row is not None:
            for k,slot in enumerate(transform['writes']):
                target=output_targets.get(slot,action['writes'][k]);action['writes'][k]=target
                if target not in action['reads']:action['reads'].append(target)
            require(len(action['reads'])<=64,'budget',syn,'caller local destinations exceed 64 context slots')
        if pathway is not None and action['noise_streams']:
            require(event_noise is not None,'noise',syn,'event draws require a persistent emission identity')
            action['event_noise']=dict(event_noise)
        action=attach_indices(action,indexed,transform['writes'],names)
        if trigger is not None and trigger.get('state'):
            # Delay graph rebuilding removes this private trailing gate. Resolve
            # parameter gathers and indirect write roots before appending it so
            # no model context can accidentally be removed in its place.
            require(len(action['reads'])<64,'budget',syn,'delayed context exceeds 64 slots')
            action['reads'].append(trigger['index'])
        actions.append(action)
        if pathway in event_effect_modes and event_effect_modes[pathway]['mode']!='scalar':
            spec=event_effect_modes[pathway]
            event_effect_actions[id(action)]=[slot for name,slot in names.items() if name not in spec['accum']]
            if original_pathway in event_stage_groups:event_snapshot_all_reads[id(action)]=list(names.values())
        return actions[-1]

    def snapshot_event_batch(start,end,pathway):
        selected=[action for action in actions[start:end] if id(action) in event_effect_actions]
        if not selected:return 0
        compacted=batch_compacted_random.get(pathway,[])
        computed={target for target,_,_ in compacted}
        origins=sorted(({action['reads'][slot] for action in selected for slot in event_snapshot_all_reads.get(id(action),event_effect_actions[id(action)])}|{cell for reads,_ in [*batch_capture_programs.get(pathway,{}).values(),*batch_capture_checks.get(pathway,[]),*[(reads,program) for _,reads,program in compacted]] for cell in reads})-computed)
        filtered=batch_filtered_rows.get(pathway)
        if filtered is not None:
            controls={filtered['count']}|{row[key] for row in filtered['rows'] for key in ('effective','prefix')}
            origins=sorted((set(origins)|{row[key] for row in filtered['rows'] for key in ('presence','mask')})-controls)
        check_budget(len(origins)*64)
        require(len(initial)+len(origins)<=1_000_000,'budget',network,'event snapshot state budget exceeded')
        cached={origin:len(initial)+j for j,origin in enumerate(origins)}
        initial.extend([0.]*len(origins));initial_parameters.extend([None]*len(origins));detached.extend([False]*len(origins))
        # Mask snapshots remain discrete, detached state rather than becoming
        # continuous cache inputs whose perturbation could change execution.
        for origin in origins:
            if origin in binary_states:
                binary_states.append(cached[origin]);detached[cached[origin]]=True
            elif origin in integer_states:
                integer_states.add(cached[origin]);detached[cached[origin]]=True
        event_effect_snapshots[pathway]=[dict(source=origin,cache=cached[origin]) for origin in origins]
        copies=[];identity=program_set([[dict(op='state',index=1)]])
        for origin in origins:
            copied=program_set([[dict(op='integer_state',index=1)]]) if origin in integer_states else identity
            copy_action=dict(owner=0,reads=[cached[origin],origin],writes=[cached[origin]],program_set=copied,threshold=None,trigger=None)
            if actions.clock:copy_action['clock']=actions.clock
            copies.append(copy_action)
        for action in selected:
            reads=list(action['reads'])
            for slot in event_effect_actions[id(action)]:reads[slot]=cached.get(reads[slot],reads[slot])
            gate=action['trigger'];trailing=reads.pop() if gate is not None and gate.get('state') else None
            for target in action['writes']:
                if target not in reads:reads.append(target)
            if trailing is not None:reads.append(trailing)
            require(len(reads)<=64,'budget',network,'event copied context exceeds 64 slots')
            action['reads']=reads
        if filtered is not None:
            for row in filtered['rows']:
                program=[dict(op='state',index=0),dict(op='boolean_cast',arg=0),
                         dict(op='state',index=1),dict(op='boolean_cast',arg=2),dict(op='eager_boolean_and',left=1,right=3)]
                action=dict(owner=0,reads=[cached[row['presence']],cached[row['mask']],row['effective']],writes=[row['effective']],program_set=program_set([program]),threshold=None,trigger=None)
                if actions.clock:action['clock']=actions.clock
                copies.append(action)
            programs=[]
            for count in [*range(len(filtered['rows'])),len(filtered['rows'])]:
                program=[dict(op='constant',value=0.)]
                for j in range(count):
                    program.append(dict(op='state',index=j));program.append(dict(op='add',left=len(program)-2,right=len(program)-1))
                programs.append(program)
            targets=[*[row['prefix'] for row in filtered['rows']],filtered['count']]
            action=dict(owner=0,reads=[*[row['effective'] for row in filtered['rows']],*targets],writes=targets,
                        program_set=program_set(programs),threshold=None,trigger=None)
            if actions.clock:action['clock']=actions.clock
            copies.append(action)
        for target,reads,program in compacted:
            context=[cached.get(cell,cell) for cell in reads]
            resolved=resolve_parameters([program],0,context,{})
            action=dict(owner=0,reads=[*context,target],writes=[target],program_set=program_set(resolved),threshold=None,trigger=None)
            if actions.clock:action['clock']=actions.clock
            copies.append(action)
        # Keep preparatory copies outside the delay path's rebuild range. Delay
        # updates preserve them as ordinary clocked model actions.
        # Check every eager root against the common input snapshot before any
        # physical capture write. Each root is an independent bounded program.
        whole_actions=[]
        for reads,program in batch_capture_checks.get(pathway,[]):
            check_budget(64);target=len(initial)
            require(target<1_000_000,'budget',network,'batch domain scratch budget exceeded')
            initial.append(0.);initial_parameters.append(None);detached.append(True)
            whole_actions.append((target,(reads,program)))
        whole_actions.extend(batch_capture_programs.get(pathway,{}).items())
        for target,(reads,program) in whole_actions:
            capture_reads=[cached.get(cell,cell) for cell in reads]
            resolved=resolve_parameters([program],0,capture_reads,{})
            gate=batch_capture_gates[pathway]
            if target not in capture_reads:capture_reads.append(target)
            capture_reads.append(gate)
            require(len(capture_reads)<=64,'budget',network,'batch capture context exceeds 64 slots')
            action=dict(owner=0,reads=capture_reads,writes=[target],program_set=program_set(resolved),threshold=None,
                        trigger=dict(external=False,state=True,index=gate),detach_trigger=True)
            if actions.clock:action['clock']=actions.clock
            copies.append(action)
        actions[start:start]=copies
        return len(copies)

    regular_layout={}

    def build_regular(spec,domain):
        runner=spec['runner'];group=spec['group'];variables=spec['variables']
        parent=_endpoint_parent(group,layers)
        info=syn_info.get(group.name)
        owner=(neuron_offsets[parent.name]+_endpoint_start(group) if parent is not None
               else endpoint_owner(info['target_group'],int(info['target'][0]) if len(group) else 0))

        def allocate(dtype):
            check_budget(64);index=len(initial)
            initial.append(0.);initial_parameters.append(None);detached.append(dtype!='float')
            if dtype=='integer':integer_states.add(index)
            if dtype=='boolean':binary_states.append(index)
            return index

        effect_temp_types={}
        def statement_type(statement):
            dtype=np.dtype(statement.dtype)
            if spec['state_effects'] and dtype.kind=='i' and dtype.itemsize>4:
                expression=ast.parse(statement.expr,mode='eval').body
                if (isinstance(expression,ast.Name) and
                        (effect_temp_types.get(expression.id)=='integer' or expression.id in variables
                         and hasattr(variables[expression.id],'dtype') and np.dtype(variables[expression.id].dtype)==np.dtype('int32'))):
                    # NumPy assignment binds the existing int32 array. Brian's
                    # inferred C scalar type must not promote this borrowed alias.
                    effect_temp_types[statement.var]='integer';return 'integer'
            require(dtype.kind in 'fb' or dtype==np.dtype('int32'),'state',runner,
                    'regular temporaries require float, Boolean or int32 storage')
            result='float' if dtype.kind=='f' else 'boolean' if dtype.kind=='b' else 'integer'
            effect_temp_types[statement.var]=result;return result

        scalar_names=spec['scalar_arrays']|{s.var for s in spec['scalar']}
        types={name:_state_dtype(variables[name]) for name in spec['scalar_arrays']}
        types.update({s.var:statement_type(s) for s in spec['scalar']})
        scratch={name:allocate(types[name]) for name in sorted(scalar_names)}
        # Vector temporaries also receive typed cells, so integer overflow and
        # Boolean assignments follow the same coercions as physical storage.
        temporary={}
        for statement in spec['vector']:
            name=statement.var
            if not isinstance(variables.get(name),ArrayVariable) and name not in scratch and name not in temporary:
                temporary[name]=allocate(statement_type(statement))
                # Masked synaptic actions own their vector-local scratch too.
                # Every declaration overwrites it, but structural migration
                # still needs an explicit restart rule for all masked writes.
                restart_rules[temporary[name]]=dict(kind='initial')
        regular_layout[runner.name]=dict(scalar=scratch,temporary=temporary,noise_domain=domain,
                                         when=runner.when,order=runner.order)
        scalar_vector=spec.get('numpy_mode')=='scalar'
        regular_layout[runner.name]['numpy_mode']=spec.get('numpy_mode','array')
        if spec['state_effects']:
            regular_layout[runner.name]['numpy_write_order']={key:list(value) for key,value in spec['numpy_write_order'].items()}

        capture_functions={};capture_fields={}
        for function_name,variable in list(variables.items()):
            if type(variable) is not Function or function_name in DEFAULT_FUNCTIONS:continue
            try:descriptor=lower_state_effect_function(variable)
            except (ValueError,TypeError,OSError,SyntaxError,RecursionError):continue
            if not descriptor.captured_arrays:continue
            capture_functions[function_name],fields=register_mutable_captures(descriptor,group,runner,function_name)
            capture_fields.update(fields)
            for name,field in fields.items():
                if field[0] is None:variables[name]=field[3]

        def resolve(name,j):
            if name in capture_functions:return None,capture_functions[name]
            if name in capture_fields:
                field=capture_fields[name]
                return (None,typed_parameter(field[2],field[4][j] if len(field)>4 else j,field[1])) if field[0] is None else (field[0][j],None)
            variable=variables[name]
            if id(variable) in external_sources:return None,external_parameter(group,name,j)
            if name in DEFAULT_FUNCTIONS:
                require(variable is DEFAULT_FUNCTIONS[name],'function',runner,'custom function replacements are unsupported')
                return None,None
            if isinstance(variable,TimedArray):
                return None,input_registry.resolve(variable,group,name,input_bank,max_values=plan['max_tape_bytes']//48)
            if isinstance(variable,Function):
                try:return None,lower_pure_function(variable)
                except ValueError as error:
                    try:return None,lower_state_effect_function(variable)
                    except (ValueError,TypeError,OSError,SyntaxError,RecursionError) as effect_error:
                        raise TrainingConversionError('function',runner,str(effect_error)) from error
            if id(variable) in condition_storage:
                index=spec['indices'][name]
                reference=next((key for key in variables if key in group.variables and
                                group.variables.indices[key]==index),None)
                require(reference is not None,'refractory',runner,'unresolved regular condition index')
                return address(group,reference,j,table=condition_storage[id(variable)]),None
            if id(variable) in storage:return address(group,name,j),None
            if name in ('t','dt'):
                value=float(group.clock.dt_)
                return None,ClockTime(clock_dts.index(value)) if name=='t' else value
            if name in ('t_pre','t_post','dt_pre','dt_post') and info is not None:
                endpoint=group.source if name.endswith('_pre') else group.target
                value=float(endpoint.clock.dt_)
                return None,ClockTime(clock_dts.index(value)) if name.startswith('t_') else value
            if id(variable) in constant_banks:return None,constant_parameter(group,name,j)
            for binding in bindings:
                if binding['kind'] not in ('neuron','neuron_array'):continue
                source=objects[binding['object']]
                for k,key in enumerate(binding['variables']):
                    if source.variables.get(key) is not variable:continue
                    if binding['kind']=='neuron':return None,typed_parameter(binding['bank'],k,_state_dtype(variable))
                    constant_banks[id(variable)]=binding['bank']
                    return None,constant_parameter(group,name,j)
            for syn in synapses:
                for key,(values,bank) in syn_info[syn.name]['params'].items():
                    if variable is syn.variables[key]:
                        table=list(range(len(values)))
                        r=address(group,name,j,table=table)
                        require(isinstance(r,int),'linked',runner,'runtime-indexed regular constants require a canonical parameter bank')
                        return None,parameter_literal(variable,values[r]) if bank is None else typed_parameter(bank,r,_state_dtype(variable))
            require(hasattr(variable,'get_value') and variable.constant,'coefficient',runner,
                    f'{name} requires runtime storage in a selected object')
            values=np.asarray(variable.get_value()).reshape(-1)
            require(values.size>0 and values.dtype.kind in 'fiub' and np.all(np.isfinite(values)),
                    'coefficient',runner,'regular constants must be finite real values')
            if variable.scalar:return None,values[0].item()
            r=address(group,name,j,table=list(range(len(values))))
            require(isinstance(r,int),'linked',runner,'runtime-indexed constants require a canonical parameter bank')
            return None,values[r].item()

        def final_selector(name):
            index_name=spec['indices'][name];frozen=False
            # address() folds constant maps into its final physical table.
            # Capture the first runtime index before that folded map; a later
            # local reassignment must not change a previously loaded map copy.
            while index_name not in ('_idx','0') and variables[index_name].constant:
                frozen=True;index_name=spec['indices'][index_name]
            return index_name,frozen

        def emit(code,j,local,*,destinations=None,writeback=False,stream_offset=0,vector=False,empty_vector=False,staged=None):
            tree=ast.parse(code);used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
            written={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
            guards={name:flag for name,flag in spec['guards'].items() if name in written and not writeback}
            if staged is not None:guards.update(staged.get('guard_targets',{}))
            used.update(guards.values())
            for name in tuple(used):
                if name in capture_functions:
                    require(vector and not scalar_vector and not empty_vector,'function',runner,'mutable captures require whole-array vector execution')
                    used.update(source for _,source in capture_functions[name].capture_bindings)
            if scalar_vector and vector:
                # Scalar NumPy fallback loads the complete index chain once
                # per row; writeback uses its final local, possibly reassigned.
                for name in list(used):
                    index_name=spec['indices'].get(name,'_idx')
                    while index_name not in ('_idx','0') and index_name not in used:
                        if not variables[index_name].constant:used.add(index_name)
                        index_name=spec['indices'][index_name]
            reads=[];names={};indexed={};parameters={};dtypes={}
            for name in sorted(used):
                if (empty_vector and name in variables and isinstance(variables[name],ArrayVariable)
                        and not variables[name].scalar):
                    # No physical element exists. Keep typed symbolic array
                    # inputs solely to identify scalar eager roots; never read
                    # an endpoint, parameter bank or element at index zero.
                    r=allocate(_state_dtype(variables[name]));parameter=None
                elif name in local:r=local[name];parameter=None
                elif name in variables or name in capture_fields:r,parameter=resolve(name,j)
                else:continue
                if r is None:
                    if parameter is not None:parameters[name]=parameter
                    continue
                slot=len(reads);names[name]=slot
                index=r['tables'][-1][0] if isinstance(r,dict) else r
                reads.append(index)
                if isinstance(r,dict):indexed[slot]=r
                dtypes[slot]='integer' if index in integer_states else 'boolean' if index in binary_states else 'float'
            # Calls are replaced before SSA composition, making repeated uses
            # of a local share its draw and distinct calls use distinct streams.
            streams=[];draws=[]
            class Draw(ast.NodeTransformer):
                def visit_Call(self,node):
                    if isinstance(node.func,ast.Name) and node.func.id in ('randn','rand','poisson'):
                        is_poisson=node.func.id=='poisson'
                        require(len(node.args)==(1 if is_poisson else 0) and not node.keywords,'noise',runner,'invalid random function arguments')
                        draws.append(dict(kind=node.func.id,stream=stream_offset+len(streams)))
                        name='__regular_normal_'+str(len(streams));parameters[name]=(PoissonNoise if is_poisson else NormalNoise if node.func.id=='randn' else UniformNoise)(stream_offset+len(streams));streams.append(name)
                        if is_poisson:return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=[self.visit(node.args[0])],keywords=[])
                        return ast.Name(id=name,ctx=ast.Load())
                    return self.generic_visit(node)
            tree=Draw().visit(tree)
            if type(runner) is SubexpressionUpdater and j==0 and draws:
                regular_layout[runner.name].setdefault('draws',{})['vector' if vector else 'scalar']=draws
            require(stream_offset+len(streams)<=16,'noise',runner,'regular runner exceeds 16 random call sites')
            guard_slots={names[name]:names[flag] for name,flag in guards.items()}
            mutable={names[flag] for flag in guards.values() if flag in written}
            try:
                if any(isinstance(value,StateEffectFunction) for value in parameters.values()) or staged is not None and staged.get('force_effect'):
                    require((parent is not None or info is not None)
                            and (not indexed or scalar_vector and vector) and destinations is None and not writeback,
                            'function',runner,'callback state effects require a selected group with fixed physical addresses')
                    local_arrays=set() if staged is None else staged.get('array_locals',set())
                    array_names={name for name in names if vector and not scalar_vector and
                                 (name in variables and not variables[name].scalar or name in local_arrays or name in capture_fields) and name not in guards.values()}
                    writable={name for name in array_names if name in group.variables
                              and not group.variables[name].read_only and (not group.variables[name].constant or id(group.variables[name]) in mutable_constant_ids)
                              }
                    writable.update(local_arrays&array_names)
                    writable.update(set(capture_fields)&array_names)
                    array_parameters={name for name in parameters if vector and not scalar_vector and name in variables
                                      and hasattr(variables[name],'scalar') and not variables[name].scalar}
                    require(all(np.dtype(variables[name].dtype).kind in 'fb' or np.dtype(variables[name].dtype)==np.dtype('int32')
                                for name in array_parameters),
                            'function',runner,'callback arrays require float, int32 or boolean storage')
                    subgroup_arrays=isinstance(group,Subgroup) and vector
                    require(all(staged is not None or spec['indices'].get(name,'_idx')=='_idx' or subgroup_arrays and spec['indices'][name]=='_sub_idx'
                                for name in array_names|array_parameters),
                            'function',runner,'callback state effects require canonical whole-array inputs; indexed NumPy reads are copies')
                    copied_arrays={name for name in array_names if spec['indices'].get(name,'_idx')!='_idx' and (subgroup_arrays or staged is not None)}
                    copied_parameters={name for name in array_parameters if spec['indices'][name]!='_idx' and (subgroup_arrays or staged is not None)}
                    # Subgroup NumPy reads use advanced-index copies. Even a
                    # constant's copied local can be mutated by a callback;
                    # only explicit array assignments write physical storage.
                    writable.update(copied_arrays)
                    temporary_parameters={name for name in streams if vector and not scalar_vector}
                    physical=None;write_order=()
                    if vector and not scalar_vector:
                        canonical={}
                        physical={}
                        for name,slot in names.items():
                            origin=(staged['origins'][name] if staged is not None and name in staged['origins'] and name not in copied_arrays else reads[slot])
                            physical[slot]=canonical.setdefault(origin,slot) if type(origin) is int else slot
                        order=spec['numpy_write_order']['vector'] if staged is None else staged.get('write_order',spec['numpy_write_order']['vector'])
                        write_order=tuple(name for name in order if name in array_names and name in written)
                    effect_args=dict(states=names,parameters=parameters,
                        array_states=array_names,writable_states=writable,state_types=dtypes,
                        array_parameters=array_parameters|temporary_parameters,
                        temporary_parameters=temporary_parameters|copied_parameters,physical_slots=physical,writeback_order=write_order,
                        copied_array_states=copied_arrays,
                        parameter_types={name:_state_dtype(variables[name]) for name in array_parameters},
                        array_callables={name for name in streams if empty_vector and type(parameters[name]) is PoissonNoise},
                        write_guards=guards,indexed_guard_reads=set() if scalar_vector and vector else array_names|array_parameters|(set(temporary)&set(names)),
                        scalar_write_guards=scalar_vector and vector,
                        scalar_eager=empty_vector,empty_vector=empty_vector)
                    transform=compile_state_effect_transform(ast.unparse(tree),**effect_args)
                    if staged is not None:
                        changed=False
                        for name in staged.get('fresh_targets',[]):
                            dtype=transform['output_types'].get(name)
                            if dtype is None or dtypes[names[name]]==dtype:continue
                            slot=names[name];cell=reads[slot];dtypes[slot]=dtype;changed=True
                            integer_states.discard(cell)
                            if cell in binary_states:binary_states.remove(cell)
                            if dtype=='integer':integer_states.add(cell)
                            elif dtype=='boolean':binary_states.append(cell)
                            detached[cell]=dtype!='float'
                        if changed:transform=compile_state_effect_transform(ast.unparse(tree),**effect_args)
                    written.update(name for name,slot in names.items() if slot in transform['effect_writes'])
                else:
                    transform=compile_dynamic_transform(ast.unparse(tree),states=names,parameters=parameters,
                        write_guards=guard_slots,mutable_write_guards=mutable,state_types=dtypes)
            except (ValueError,RecursionError) as error:raise TrainingConversionError('expression',runner,str(error)) from error
            if empty_vector:
                for program in transform.get('scalar_programs',[]):
                    eager_reads=list(reads);eager_indexed={}
                    resolved=resolve_parameters([program],j,eager_reads,eager_indexed)
                    require(not eager_indexed,'linked',runner,'empty vector scalar work requires fixed scalar addresses')
                    sink=allocate('float');detached[sink]=True
                    eager_reads.append(sink)
                    require(len(eager_reads)<=64,'budget',runner,'empty vector eager context exceeds budget')
                    actions.append(dict(owner=owner,reads=eager_reads,writes=[sink],
                        program_set=program_set(resolved),threshold=None,trigger=None,
                        parameter_index=0,noise_domain=domain,noise_entity=0,
                        noise_streams=stream_offset+len(streams) if streams else 0))
                return
            by_slot=dict(zip(transform['writes'],transform['programs']))
            output=ordered_outputs(names,written,reads,indexed)
            if scalar_vector and vector:
                order=[name for name in spec['numpy_write_order']['vector'] if name in written and name in names]
                order+=sorted((written&names.keys())-set(order))
                ordered=[names[name] for name in order if names[name] in by_slot]
                winners={reads[slot]:slot for slot in ordered if slot not in indexed}
                output=[slot for slot in ordered if slot in indexed or winners[reads[slot]]==slot]
            transform['writes']=output;transform['programs']=[by_slot[slot] for slot in output]
            transform['programs']=resolve_parameters(transform['programs'],j,reads,indexed)
            transform['context_size']=len(reads)
            entity_owner=owner+j if parent is not None else endpoint_owner(info['target_group'],int(info['target'][j]) if len(group) else 0)
            action=dynamic_action(transform,reads,owner=owner if destinations is not None or writeback else entity_owner,
                program_set=program_set(transform['programs']),parameter_index=j,
                noise_domain=domain,noise_entity=j,noise_streams=stream_offset+len(streams) if streams else 0,
                mask=edge_mask(info,j) if vector and info is not None else None)
            if staged is not None:
                staged['aliases']={name:(next((key for key,value in names.items() if value==slot),None) if slot is not None else None)
                                   for name,slot in transform.get('output_aliases',{}).items()}
                staged['arrays']=list(transform.get('output_arrays',[]))
                staged['types']=dict(transform.get('output_types',{}))
                staged['direct_outputs']={}
                # Calculation writes private result cells. Scatter happens
                # only after every row has observed the block-entry snapshot.
                for k,slot in enumerate(output):
                    name=next(name for name,value in names.items() if value==slot)
                    if name not in staged['origins']:
                        staged['direct_outputs'][name]=action['writes'][k];continue
                    cell=allocate(dtypes[slot]);staged['outputs'][name]=cell
                    action['writes'][k]=cell;action['reads'].append(cell)
                require(len(action['reads'])<=64,'budget',runner,'staged callback context exceeds budget')
                # Parameter resolution can add gathers; those are reads only.
                action=attach_indices(action,indexed,[],names)
            else:
                write_descriptors=None
                if scalar_vector and vector:
                    write_descriptors={slot:dict(r,index=reads[names[selector]],root_name=selector,
                                                snapshot_index=frozen,tables=r['tables'][-1:])
                                       for name,slot in names.items() if slot in indexed
                                       for r in [indexed[slot]] for selector,frozen in [final_selector(name)]}
                action=attach_indices(action,indexed,output,names,write_descriptors)
            if destinations is not None:
                action['writes']=[destinations[name] for name in sorted(written&names.keys())
                                  if names[name] in output]
                require(len(action['writes'])==len(output),'runner',runner,'invalid regular scalar snapshot')
                action['reads'].extend(index for index in action['writes'] if index not in action['reads'])
                require(len(action['reads'])<=64,'budget',runner,'regular writeback exceeds context budget')
                if 'indirect' in action:action['indirect'].pop('writes',None)
            actions.append(action)

        # All scalar array reads happen before scalar assignments. Each alias
        # has its own scratch cell, including aliases of the same shared array.
        if spec['scalar_arrays']:
            copies={f'__regular_snapshot_{k}':scratch[name]
                    for k,name in enumerate(sorted(spec['scalar_arrays']))}
            emit('\n'.join(f'__regular_snapshot_{k} = {name}'
                           for k,name in enumerate(sorted(spec['scalar_arrays']))),0,copies,writeback=True)
        def source(statements):
            return '\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {s.expr}' for s in statements)

        def scatter(name,row,j,*,operation=None):
            value=row['outputs'][name];origin=row['origins'][name]
            require(origin is not None,'coefficient',runner,'callback cannot write a constant parameter bank')
            target=origin['tables'][-1][0] if isinstance(origin,dict) else origin
            reads=[value,target];integer=value in integer_states
            program=[dict(op='integer_state' if integer else 'state',index=0)]
            if operation is not None:
                require(operation in ('+=','-=','*=','/='),'runner',runner,'unsupported NumPy indexed ufunc')
                target_type='integer' if target in integer_states else 'boolean' if target in binary_states else 'float'
                value_type='integer' if integer else 'boolean' if value in binary_states else 'float'
                expression='target = target'+operation[0]+'operand'
                # ufunc.at computes in the promoted dtype, then casts each
                # updated element into the destination array. It differs from
                # both an augmented ndarray assignment and casting the RHS.
                try:
                    compiled=compile_state_effect_transform(expression,states={'target':1,'operand':0},parameters={},
                        array_states={'target','operand'},writable_states={'target'},state_types={1:target_type,0:value_type})
                except (ValueError,RecursionError) as error:
                    raise TrainingConversionError('expression',runner,str(error)) from error
                program=compiled['programs'][0]
            owner_j=owner+j if parent is not None else endpoint_owner(info['target_group'],int(info['target'][j]))
            action=dict(owner=owner_j,reads=reads,writes=[target],program_set=program_set([program]),threshold=None,trigger=None,
                        mask=edge_mask(info,j) if info is not None else None)
            if isinstance(origin,dict):
                index_name,frozen=final_selector(name)
                selector=row['captures'].get(index_name) if frozen else row['outputs'].get(index_name,row['captures'].get(index_name))
                require(selector is not None,'linked',runner,'staged callback writeback requires a captured final index')
                # NumPy loads transitive index arrays at block/statement entry.
                # The final local selector survives later mutations of its
                # root or intermediate arrays until an explicit re-read.
                tables=origin['tables'][-1:]
                reads.append(selector)
                action['indirect']=dict(writes={'0':dict(index=dict(kind='read',slot=2),tables=tables)})
                if operation is not None:action['indirect']['reads']={'1':dict(index=selector,tables=tables)}
            if operation is not None and row.get('scatter_guard') is not None:
                guard=row['captures'][row['scatter_guard']]
                reads.append(guard);action['trigger']=dict(external=False,state=True,index=guard);action['detach_trigger']=True
            actions.append(action)

        def build_vectorised(offset):
            # NumPy vectorise_code loads and writes each statement separately.
            # Locals retain references to either live physical arrays or their
            # own allocated buffers across these statement boundaries.
            bindings=[{} for _ in range(len(group))];stages=[]
            regular_layout[runner.name]['vectorised_stages']=stages
            for k,statement in enumerate(spec['vector']):
                original=statement.var
                physical=isinstance(variables.get(original),ArrayVariable)
                accumulate=physical and statement.inplace and spec['indices'][original]!='_idx'
                fresh=not physical and not statement.inplace or accumulate
                result='__regular_stage_value_'+str(k) if fresh else original
                code=result+' = '+statement.expr if fresh else source([statement])
                used={node.id for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Name)}
                used.update(flag for target,flag in spec['guards'].items() if target==original)
                array_fields={name for name in used if isinstance(variables.get(name),ArrayVariable) and not variables[name].scalar}
                array_fields.update(final_selector(name)[0] for name in [*array_fields,*([original] if physical else [])]
                                    if spec['indices'][name] not in ('_idx','0') and
                                    isinstance(resolve(name,0)[0],dict))
                stage=dict(statement=k,target=original,operation=statement.op,accumulate=accumulate,result=result,code=code,rows=[])
                stages.append(stage)
                for j in range(len(group)):
                    row=dict(captures={},origins={},outputs={},force_effect=True,array_locals=set(),write_order=[])
                    stage['rows'].append(row)
                    for name in sorted(array_fields):
                        origin,_=resolve(name,j);row['origins'][name]=origin
                        cache=allocate(_state_dtype(variables[name]));row['captures'][name]=cache
                        alias='__regular_capture_'+name;emit(alias+' = '+name,j,{alias:cache},writeback=True,vector=True)
                    for name,(cell,array) in bindings[j].items():
                        if name not in used:continue
                        cache=allocate('integer' if cell in integer_states else 'boolean' if cell in binary_states else 'float')
                        row['captures'][name]=cache;row['origins'][name]=cell
                        alias='__regular_local_capture_'+name
                        emit(alias+' = '+name,j,{name:cell,alias:cache},writeback=True,vector=True)
                        if array:row['array_locals'].add(name)
                    if fresh:
                        row['result_cell']=allocate(statement_type(statement));row['fresh_targets']=[result]
                    row['write_order']=[original] if not fresh else []
                    if accumulate and original in spec['guards']:
                        flag=spec['guards'][original];row['scatter_guard']=flag
                        row['guard_targets']={result:flag};row['array_locals'].add(result)
                        row['origins'][result]=row['result_cell'];row['write_order']=[result]
                for j,row in enumerate(stage['rows']):
                    local={**scratch,**row['captures']}
                    if fresh:local[result]=row['result_cell']
                    elif not physical and original in bindings[j]:local[original]=row['captures'][original]
                    emit(code,j,local,stream_offset=offset,vector=True,staged=row)
                    row['array_locals']=sorted(row['array_locals'])
                    if accumulate:
                        # The ufunc operand is evaluated independently of the
                        # target, then each repeated index applies to its latest
                        # physical value in source-row order.
                        row['origins'][original]=resolve(original,j)[0]
                        row['outputs'][original]=row['direct_outputs'].get(result,row['outputs'].get(result))
                    elif not physical:
                        value=row['direct_outputs'].get(result,row['outputs'].get(result))
                        origin_name=row['aliases'].get(result)
                        origin=row['origins'].get(origin_name)
                        # Copied/indexed expressions have no borrowed origin.
                        # Rebinding allocates a new buffer; other aliases retain
                        # their old buffer even when the original name changes.
                        require(origin_name is None or type(origin) is int,'function',runner,'temporary array alias requires fixed storage')
                        bindings[j][original]=(origin if origin_name is not None else value,result in row['arrays'])
                        row['binding']=dict(cell=bindings[j][original][0],array=bindings[j][original][1])
                hidden=sorted({name for row in stage['rows'] for name in row['outputs']} - ({original} if physical else set()))
                for name in hidden:
                    for j,row in enumerate(stage['rows']):
                        if name in row['outputs']:scatter(name,row,j)
                if physical:
                    for j,row in enumerate(stage['rows']):
                        if original in row['outputs']:scatter(original,row,j,operation=statement.op if accumulate else None)
                offset+=sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('randn','rand','poisson')
                            for node in ast.walk(ast.parse(statement.expr)))
            regular_layout[runner.name]['vectorised_locals']=bindings
        if spec['scalar']:emit(source(spec['scalar']),0,scratch)
        # Sorted Cython array writeback, leaving all scalar locals intact for
        # the vector loop even when several names alias one physical cell.
        if spec['scalar_writes']:
            scalar_order=spec['numpy_write_order']['scalar'] if spec['state_effects'] else sorted(spec['scalar_writes'])
            destinations={name:resolve(name,0)[0] for name in scalar_order}
            require(all(isinstance(index,int) for index in destinations.values()),'linked',runner,
                    'scalar regular writes require fixed physical addresses')
            winners={index:name for name,index in destinations.items()}
            selected={name:index for name,index in destinations.items() if winners[index]==name}
            emit('\n'.join(name+' = '+name for name in selected),0,scratch,destinations=selected,writeback=True)
        if spec['vector']:
            if spec['state_effects'] and spec['vector_scalar_reads']:
                # NumPy loads shared arrays again at vector-block entry. Scalar
                # AuxiliaryVariable temporaries (e.g. _lio_1) remain untouched.
                reloads={f'__regular_vector_reload_{k}':scratch[name]
                         for k,name in enumerate(sorted(spec['vector_scalar_reads']))}
                emit('\n'.join(f'__regular_vector_reload_{k} = {name}'
                               for k,name in enumerate(sorted(spec['vector_scalar_reads']))),0,reloads,writeback=True)
            offset=sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('randn','rand','poisson')
                       for node in ast.walk(ast.parse(source(spec['scalar']))))
            if not len(group) and spec['state_effects']:
                emit(source(spec['vector']),0,{**scratch,**temporary},stream_offset=offset,vector=True,empty_vector=True)
            staged_names={name for statement in spec['vector'] for name in
                          ({statement.var}|{node.id for node in ast.walk(ast.parse(statement.expr)) if isinstance(node,ast.Name)})
                          if isinstance(variables.get(name),ArrayVariable) and not variables[name].scalar}
            staged_names.update(name for name in spec['guards'].values()
                                if isinstance(variables.get(name),ArrayVariable) and not variables[name].scalar)
            if spec.get('numpy_mode')=='vectorised':
                build_vectorised(offset);return
            use_staging=spec['state_effects'] and not scalar_vector and not isinstance(group,Subgroup) and any(spec['indices'][name]!='_idx' for name in staged_names)
            if use_staging:
                staged_names.update(final_selector(name)[0] for name in list(staged_names)
                                    if spec['indices'][name] not in ('_idx','0') and
                                    isinstance(resolve(name,0)[0],dict))
                rows=[]
                for j in range(len(group)):
                    row=dict(captures={},origins={},outputs={});rows.append(row)
                    for name in sorted(staged_names):
                        origin,_=resolve(name,j);row['origins'][name]=origin
                        cache=allocate(_state_dtype(variables[name]));row['captures'][name]=cache
                        alias='__regular_capture_'+name
                        emit(alias+' = '+name,j,{alias:cache},writeback=True,vector=True)
                regular_layout[runner.name]['staged']=rows
                for j,row in enumerate(rows):
                    emit(source(spec['vector']),j,{**scratch,**temporary,**row['captures']},stream_offset=offset,vector=True,staged=row)
                explicit=list(spec['numpy_write_order']['vector'])
                hidden=sorted({name for row in rows for name in row['outputs']} - set(explicit))
                for name in [*hidden,*explicit]:
                    for j,row in enumerate(rows):
                        if name not in row['outputs']:continue
                        scatter(name,row,j)
            else:
                for j in range(len(group)):emit(source(spec['vector']),j,{**scratch,**temporary},stream_offset=offset,vector=True)

    def history(length, pending=False):
        require(length<=1_000_000-len(initial),'budget',network,'delay history exceeds state budget')
        check_budget(length*64)
        indices=list(range(len(initial),len(initial)+length))
        initial.extend([0.]*(length-1)+[float(pending)])
        initial_parameters.extend([None]*length);detached.extend([False]*length);binary_states.extend(indices)
        restart_rules.update((index,dict(kind='queue')) for index in indices)
        return indices

    def advance_history(indices,owner,mask,trigger):
        # Shift in ascending chunks: each chunk's final source has not yet been
        # overwritten. 63 writes plus its next source fit the 64-slot context.
        for start in range(0,len(indices)-1,63):
            reads=indices[start:min(start+64,len(indices))]
            ps=program_set([[dict(op='state',index=k)] for k in range(1,len(reads))])
            actions.append(dict(owner=owner,mask=mask,reads=reads,writes=reads[:-1],program_set=ps,threshold=None,trigger=None))
        last=indices[-1]
        actions.append(dict(owner=owner,mask=mask,reads=[last],writes=[last],program_set=program_set([[dict(op='constant',value=0.)]]),threshold=None,trigger=None))
        if trigger is not None:
            reads=[last]+([trigger['index']] if trigger.get('state') else [])
            actions.append(dict(owner=owner,mask=mask,reads=reads,writes=[last],program_set=program_set([[dict(op='constant',value=1.)]]),threshold=None,trigger=trigger))

    # Linked shared storage retains every referencing object's lifetime even
    # when it is only read. A one-cell identity action makes that explicit in
    # the same action graph audited by native migration (and has identity VJP).
    for syn in synapses:
        info=syn_info[syn.name]
        for name,indices in info['runtime'].items():
            candidates=set(indices)
            for r in indirect_cells.get((syn.name,name),[]):
                if r is not None:candidates.update(r['tables'][-1])
            if not any(index in restart_rules for index in candidates) or id(syn.variables[name]) not in linked_targets:continue
            for edge in range(len(syn)):
                r=index_descriptor(syn,name,edge)
                targets=r['tables'][-1] if r is not None else [indices[0 if len(indices)==1 else edge]]
                for index in sorted(set(targets)):
                    ps=program_set([[dict(op='integer_state' if index in integer_states else 'state',index=0)]])
                    actions.append(dict(owner=endpoint_owner(info['target_group'],int(info['target'][edge])),
                        reads=[index],writes=[index],program_set=ps,threshold=None,trigger=None,mask=edge_mask(info,edge)))
    for group,names in zip(layers,provenance['state_names']):
        for name in names:
            for j,index in enumerate(cells[group.name,name]):
                r=index_descriptor(group,name,j)
                for target in sorted(set(r['tables'][-1] if r is not None else [index])):
                    if target not in restart_rules:continue
                    actions.append(dict(owner=neuron_offsets[group.name]+j,reads=[target],writes=[target],
                        program_set=program_set([[dict(op='integer_state' if target in integer_states else 'state',index=0)]]),threshold=None,trigger=None))

    def append_refractory_context(group,j,phase,reads,indexed):
        r=address(group,neuron_condition_indices[group.name][phase],j,table=activity[group.name])
        slot=len(reads);reads.append(r['tables'][-1][0] if isinstance(r,dict) else r)
        if isinstance(r,dict):indexed[slot]=r
        reads.extend([refractory_age[group.name][j],refractory_lastspike[group.name][j],*[row[j] for row in refractory_words[group.name]]])

    for obj in network.sorted_objects:
        actions.clock=clock_dts.index(float(obj.clock.dt_))
        if obj.name in regular:
            build_regular(regular[obj.name],len(layers)+len(synapses)+list(regular).index(obj.name))
            continue
        handled=False
        for layer,group in enumerate(layers):
            if obj is group.state_updater or obj is group.resetter['spike']:
                reset=obj is group.resetter['spike'];key='state_resets' if reset else 'state_equations'
                state_names=provenance['state_names'][layer];code=copy.deepcopy(plan[key][layer])
                capture_programs=provenance.get('reset_capture_programs',[{} for _ in layers])[layer] if reset else {}
                capture_snapshots={}
                if capture_programs:
                    # Reset locals are advanced-index copies; closure arrays
                    # retain full physical storage and run even on no events.
                    # Snapshot before either kind of write to preserve aliases.
                    for name in capture_programs:
                        dtype=provenance['state_types'][layer][state_names.index(name)]
                        slots=[]
                        for j,cell in enumerate(cells[group.name,name]):
                            check_budget(64);slot=len(initial);slots.append(slot)
                            initial.append(0.);initial_parameters.append(None);detached.append(dtype!='float')
                            if dtype=='integer':integer_states.add(slot)
                            elif dtype=='boolean':binary_states.append(slot)
                            actions.append(dict(owner=neuron_offsets[group.name]+j,reads=[cell,slot],writes=[slot],
                                program_set=program_set([[dict(op='integer_state' if dtype=='integer' else 'state',index=0)]]),threshold=None,trigger=None))
                        capture_snapshots[name]=slots
                    for name in state_names:
                        alias=next((capture for capture in capture_programs
                                    if cells[group.name,name]==cells[group.name,capture]),None)
                        if alias is not None:capture_snapshots[name]=capture_snapshots[alias]
                    for j in range(len(group)):
                        reads=[capture_snapshots.get(name,cells[group.name,name])[j] for name in state_names];indexed={}
                        expanded=external_neuron_programs(list(capture_programs.values()),group,j)
                        ps=program_set(resolve_parameters(expanded,j,reads,indexed))
                        writes=[cells[group.name,name][j] for name in capture_programs]
                        reads.extend(cell for cell in writes if cell not in reads)
                        require(len(reads)<=64,'budget',group,'reset capture context exceeds 64 slots')
                        actions.append(dict(owner=neuron_offsets[group.name]+j,reads=reads,
                            writes=writes,program_set=ps,
                            threshold=None,trigger=None,parameter_index=j))
                reset_markers=[]
                if reset and provenance.get('reset_effect_writes',[None]*len(layers))[layer] is not None:
                    check_budget(len(group)*64)
                    for j in range(len(group)):
                        marker=len(initial);reset_markers.append(marker)
                        initial.append(0.);initial_parameters.append(None);detached.append(True);binary_states.append(marker)
                        owner=neuron_offsets[group.name]+j
                        for value,trigger in [(0.,None),(1.,dict(external=False,index=owner))]:
                            actions.append(dict(owner=owner,reads=[marker],writes=[marker],
                                program_set=program_set([[dict(op='constant',value=value)]]),threshold=None,
                                trigger=trigger,detach_trigger=True))
                scalar_programs=provenance.get('reset_scalar_programs',[[] for _ in layers])[layer] if reset else []
                if scalar_programs:
                    # NumPy executes scalar callback operations even for an
                    # empty selection. With events present they retain their
                    # original order inside the reset candidate programs.
                    check_budget(128);empty=len(initial);scratch=empty+1
                    initial.extend([1.,0.]);initial_parameters.extend([None,None]);detached.extend([True,True]);binary_states.append(empty)
                    owner=neuron_offsets[group.name]
                    actions.append(dict(owner=owner,reads=[empty],writes=[empty],
                        program_set=program_set([[dict(op='constant',value=1.)]]),threshold=None,trigger=None))
                    for j in range(len(group)):
                        actions.append(dict(owner=owner,reads=[empty],writes=[empty],
                            program_set=program_set([[dict(op='constant',value=0.)]]),threshold=None,
                            trigger=dict(external=False,index=neuron_offsets[group.name]+j),detach_trigger=True))
                    for program in scalar_programs:
                        reads=[scratch,empty];indexed={}
                        expanded=external_neuron_programs([program],group,0)
                        resolved=resolve_parameters(expanded,0,reads,indexed)
                        actions.append(dict(owner=owner,reads=reads,writes=[scratch],program_set=program_set(resolved),
                            threshold=None,trigger=dict(external=False,state=True,index=empty),detach_trigger=True,
                            parameter_index=0,noise_domain=layer,noise_entity=0,noise_streams=0))
                spec=plan.get('refractory',[None]*len(layers))[layer]
                expression_ref=refractory_expressions[layer]
                if spec is not None:
                    counter=len(state_names)-1
                    if reset:code[counter]=[dict(op='constant',value=float(max(spec['steps']-1,0)))]
                    else:
                        effect_integrator=provenance.get('integrator_effect_writes',[None]*len(layers))[layer] is not None
                        for s in ([] if effect_integrator else spec['clamp']):
                            nodes=code[s];result=len(nodes)-1;gate=len(nodes);old=gate+1
                            nodes.extend([dict(op='refractory_active',index=counter),dict(op='state',index=s),dict(op='select',condition=gate,yes=result,no=old)])
                        code[counter]=[dict(op='state',index=counter),dict(op='constant',value=1.),dict(op='sub',left=0,right=1),
                                       dict(op='constant',value=0.),dict(op='max',left=2,right=3)]
                        code.append([dict(op='refractory_active',index=counter)])
                        if expression_ref is not None:
                            code=[_with_refractory_gate(p,expression_ref['program']) for p in code]
                            # Elapsed ticks are for the next tick, unless this
                            # tick's threshold overwrites them with one.
                            code.append([dict(op='integer_state',index=len(state_names)+1),
                                         dict(op='integer_constant',value=2**31-2),
                                         dict(op='integer_binary',kind='min',left=0,right=1),
                                         dict(op='integer_constant',value=1),
                                         dict(op='integer_binary',kind='add',left=2,right=3)])
                written=({n.id for n in ast.walk(ast.parse(group.event_codes['spike']))
                          if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)} if reset else set(group.equations.diff_eq_names))
                if not reset:
                    written.update(provenance.get('integrator_effect_writes',[None]*len(layers))[layer] or ())
                else:
                    written.update(provenance.get('reset_effect_writes',[None]*len(layers))[layer] or ())
                if spec is not None:written.add('__refractory_ticks')
                for j in range(len(group)):
                    reads=[capture_snapshots.get(name,cells[group.name,name])[j] for name in state_names];owner=neuron_offsets[group.name]+j
                    indexed=descriptors(group,state_names,j);names={name:s for s,name in enumerate(state_names)}
                    needed={names[name] for name in written&names.keys()}
                    needed.update(context_slots([code[s] for s in needed]))
                    if expression_ref is not None and not reset:needed.update(context_slots([expression_ref['program']]))
                    indexed={s:r for s,r in indexed.items() if s in needed}
                    if expression_ref is not None:
                        append_refractory_context(group,j,'reset' if reset else 'update',reads,indexed)
                    elif spec is not None and not reset:
                        # Brian assigns the freshly computed flag through the
                        # final conditional variable's index, even for a read.
                        r=address(group,neuron_condition_indices[group.name]['update'],j,table=activity[group.name])
                        reads.append(r['tables'][-1][0] if isinstance(r,dict) else r)
                        if isinstance(r,dict):indexed[len(state_names)]=r
                    if reset_markers:reads.append(reset_markers[j])
                    outputs=ordered_outputs(names,written,reads,indexed)
                    if not indexed:outputs.sort()
                    if spec is not None and not reset:outputs.append(len(state_names))
                    if expression_ref is not None and not reset:outputs.append(len(state_names)+1)
                    if not outputs:continue
                    expanded=external_neuron_programs([code[s] for s in outputs],group,j)
                    ps=program_set(resolve_parameters(expanded,j,reads,indexed))
                    action=dict(owner=owner,reads=reads,writes=[reads[s] for s in outputs],program_set=ps,threshold=None,
                        trigger=dict(external=False,index=owner) if reset else None,detach_trigger=plan['detach_reset'] if reset else False,
                        parameter_index=j,noise_domain=layer,noise_entity=j,noise_streams=len(provenance['noise_names'][layer]) if not reset or provenance['neuron_draws'][layer]['reset'] else 0)
                    if capture_snapshots:
                        action['writes']=[cells[group.name,state_names[s]][j] if s<len(state_names) else reads[s] for s in outputs]
                        reads.extend(cell for cell in action['writes'] if cell not in reads)
                        require(len(reads)<=64,'budget',group,'reset capture writeback context exceeds 64 slots')
                    actions.append(attach_indices(action,indexed,outputs,names))
                handled=True;break
            if obj is group.thresholder['spike']:
                for j in range(len(group)):
                    owner=neuron_offsets[group.name]+j
                    margin=provenance['threshold_margins'][layer]
                    if margin is not None:
                        check_budget(64);index=len(initial);initial.append(0.);initial_parameters.append(None);detached.append(False)
                        state_names=provenance['state_names'][layer];indexed=descriptors(group,state_names,j)
                        context=[cells[group.name,name][j] for name in state_names]
                        if refractory_expressions[layer] is not None:append_refractory_context(group,j,'threshold',context,indexed)
                        context.append(index);output_slot=len(context)-1
                        effects=margin.get('effect_outputs',[])
                        winners={context[row['state']]:k for k,row in enumerate(effects)}
                        effects=[row for k,row in enumerate(effects) if winners[context[row['state']]]==k]
                        outputs=[row['state'] for row in effects]+[output_slot]
                        expanded=external_neuron_programs([row['program'] for row in effects]+[margin['program']],group,j)
                        resolved=resolve_parameters(expanded,j,context,indexed)
                        action=dict(owner=owner,reads=context,writes=[context[s] for s in outputs],program_set=program_set(resolved),
                            threshold=None,trigger=None,parameter_index=j,noise_domain=layer,noise_entity=j,
                            noise_streams=len(provenance['noise_names'][layer]) if provenance['neuron_draws'][layer]['threshold'] else 0)
                        actions.append(attach_indices(action,indexed,outputs,
                                                      {name:s for s,name in enumerate(state_names)}))
                        provenance.setdefault('threshold_margin_layout',{}).setdefault(group.name,[]).append(index)
                        reads=[index]
                    else:reads=[cells[group.name,'v'][j]]
                    if group.name in activity:
                        r=address(group,neuron_condition_indices[group.name]['threshold'],j,table=activity[group.name])
                        if isinstance(r,dict):
                            check_budget(64);gate=len(initial);initial.append(0.);initial_parameters.append(None);detached.append(True);binary_states.append(gate)
                            actions.append(dict(owner=owner,reads=[r['tables'][-1][0],gate],writes=[gate],
                                program_set=program_set([[dict(op='state',index=0)]]),threshold=None,trigger=None,
                                indirect=dict(reads={'0':dict(index=r['index'],tables=r['tables'])},writes={})))
                        else:gate=r
                        reads.append(gate)
                    actions.append(dict(owner=owner,reads=reads,writes=[],program_set=None,threshold=owner,trigger=None,
                        threshold_margin=margin is not None,threshold_inclusive=margin is not None and margin['inclusive'],
                        threshold_predicate=margin is not None and margin.get('predicate',False)))
                    if group.name in activity:
                        gate=activity[group.name][j]
                        actions.append(dict(owner=owner,reads=[gate],writes=[gate],program_set=program_set([[dict(op='constant',value=0.)]]),
                                            threshold=None,trigger=dict(external=False,index=owner),detach_trigger=True))
                        if refractory_expressions[layer] is not None:
                            controls=[refractory_age[group.name][j],refractory_lastspike[group.name][j],*[row[j] for row in refractory_words[group.name]]]
                            actions.append(dict(owner=owner,reads=controls,writes=controls,
                                program_set=program_set([[dict(op='integer_constant',value=1)],[dict(op='time')],
                                    *[[dict(op='time_word',word=word,float32=np.dtype(group.variables['lastspike'].dtype)==np.dtype('float32'))] for word in (0,1)]]),
                                threshold=None,trigger=dict(external=False,index=owner),detach_trigger=True))
                handled=True;break
        if handled:continue
        for syn in synapses:
            info=syn_info[syn.name]
            if obj.name in info['summed_codes']:
                pre=obj.target_varname.endswith('_pre');name=obj.target_varname[:-4 if pre else -5]
                # Cython clears exactly the target view, then adds in edge order.
                group=_endpoint_parent(obj.target,[*layers,*synapses]);start=_endpoint_start(obj.target)
                ps=program_set([[dict(op='integer_constant',value=0)]]) if _state_dtype(obj.target_var)=='integer' else program_set([[dict(op='constant',value=0.)]])
                for j in range(start,start+len(obj.target)):
                    index=address(group,name,j)
                    require(isinstance(index,int),'summed',obj,'summed targets require fixed physical addresses')
                    actions.append(dict(owner=endpoint_owner(group,j),reads=[index],writes=[index],
                                        program_set=ps,threshold=None,trigger=None))
                indices=info['source'] if pre else info['target']
                for edge in range(len(syn)):
                    build_synaptic(syn,edge,info['summed_codes'][obj.name],owner=endpoint_owner(group,int(indices[edge])),
                                   noise_domain=scheduled_noise_domains[obj.name])
            if obj is syn.state_updater and info['continuous']:
                require(obj.active and obj.when=='groups','schedule',obj,'continuous synaptic updates must run in groups')
                for edge in range(len(syn)):build_synaptic(syn,edge,info['continuous'],streams=sorted(syn.equations.stochastic_variables))
            if any(obj is path for path in syn._pathways):
                pre=obj.prepost=='pre';source=info['source_group'] if pre else info['target_group']
                require(source in groups,'endpoint',obj,'spike pathways require a selected neuron or input event source')
                indices=info['source'] if pre else info['target']
                code=info['event_code']+'\n'+info['path_codes'][obj.name]
                if info['event_code']:code+='\nlastupdate=t'
                try:effect_spec=prepare_effect_event(syn,obj,code,info['path_variables'][obj.name])
                except (ValueError,TypeError,SyntaxError,RecursionError) as error:raise TrainingConversionError('function',obj,str(error)) from error
                stage_specs=effect_spec['stages'] if effect_spec is not None else [None]
                called={node.func.id for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
                captured_batch=False;captured_functions=set()
                if effect_spec is not None and effect_spec['mode']!='scalar':
                    for name in called:
                        variable=info['path_variables'][obj.name].get(name)
                        if isinstance(variable,Function):
                            try:descriptor=lower_state_effect_function(variable)
                            except (ValueError,TypeError,OSError,SyntaxError,RecursionError):continue
                            if descriptor.captured_arrays:captured_batch=True;captured_functions.add(name)
                if captured_batch:
                    caller_trace=None
                    grouped_caller=False
                    materialise_locals=any(part.get('indexed_locals') for part in stage_specs)
                    plan_locals=materialise_locals or effect_spec['mode']=='array'
                    if plan_locals:
                        template=dict(stage_specs[0],code=effect_spec['code'],writes=effect_spec['writes'],accum=effect_spec['accum'],
                                      replay_fields={},replay_noise={},replay_guards={},stream_offset=0,
                                      guarded_writes=tuple(dict.fromkeys(name for part in stage_specs for name in part['guarded_writes'])))
                        stage_specs=[template]
                    require(len(stage_specs)==1,'function',obj,'batch capture replay stages require persistent temporaries')
                    from .training_batch_random import cached_batch_draws
                    try:rewritten,random_draws=cached_batch_draws(stage_specs[0]['code'],info['path_variables'][obj.name])
                    except ValueError as error:raise TrainingConversionError('noise',obj,str(error)) from error
                    if random_draws:stage_specs=[dict(stage_specs[0],code=rewritten)]
                    if plan_locals:
                        from .training_batch_locals import batch_local_lifetimes,whole_capture_carry_groups
                        caller_functions={name:lower_state_effect_function(info['path_variables'][obj.name][name]) for name in captured_functions}
                        try:caller_trace=batch_local_lifetimes(stage_specs[0]['code'],info['path_variables'][obj.name],caller_functions,
                                mode=effect_spec['mode'],synthetic_types={draw['name']:'float' for draw in random_draws})
                        except (ValueError,TypeError,RecursionError) as error:raise TrainingConversionError('function',obj,str(error)) from error
                        if any(token['whole_capture'] for token in caller_trace['tokens']):
                            grouped=([dict(code=stage_specs[0]['code'],whole_local_inputs={})] if effect_spec['mode']=='array'
                                     else whole_capture_carry_groups(caller_trace,syn.variables))
                            require(grouped is not None,'function',obj,
                                    'whole-vector caller temporaries spanning indexed writes require persistent whole-vector storage')
                            if grouped is not None:
                                template=stage_specs[0];parts=[]
                                tokens={token for part in grouped for token in part.get('whole_local_outputs',{}).values()}
                                buffers={}
                                for token in sorted(tokens):
                                    descriptor=caller_trace['tokens'][token];length=descriptor['length']
                                    require(1<=length<=64,'budget',obj,'whole caller vector exceeds 64 columns')
                                    require(len(initial)+length<=1_000_000,'budget',obj,'whole caller state budget exceeded')
                                    check_budget(length*64);slots=[]
                                    for column in range(length):
                                        cell=len(initial);initial.append(0.);initial_parameters.append(None);detached.append(descriptor['dtype']!='float');slots.append(cell)
                                        if descriptor['dtype']=='integer':integer_states.add(cell)
                                        if descriptor['dtype']=='boolean':binary_states.append(cell)
                                    buffers[token]=('_b2_whole_local_'+str(len(batch_whole_locals))+'_'+str(token),(slots,descriptor['dtype']))
                                if buffers:batch_whole_locals[obj.name]=buffers
                                for group in grouped:
                                    part_code=group['code']
                                    target=ast.parse(part_code).body[-1]
                                    target=target.targets[0].id if isinstance(target,ast.Assign) else target.target.id
                                    parts.append(dict(template,code=part_code,writes=(target,),
                                        whole_local_inputs=group['whole_local_inputs'],
                                        whole_local_outputs=group.get('whole_local_outputs',{}),
                                        accum=tuple(name for name in template['accum'] if name==target),
                                        guarded_writes=tuple(name for name in template['guarded_writes'] if name==target)))
                                stage_specs=parts;caller_trace=None;grouped_caller=True
                                if effect_spec['mode']=='array' and len(parts)>1:
                                    stage_specs.append(dict(template,code='',writes=(),accum=(),guarded_writes=(),publish_array_locals=True))
                            if effect_spec['mode']=='array':stage_specs=[dict(template,code=grouped[0]['code'])]
                        elif not materialise_locals:caller_trace=None
                    if caller_trace is not None:
                        require(obj.name not in info['path_runtime'],'function',obj,'caller local arrays require fixed delay routing')
                        template=stage_specs[0];parts=[]
                        for stage in caller_trace['stages']:
                            writes=tuple(stage['outputs'])
                            parts.append(dict(template,code=stage['code'],writes=writes,local_inputs=stage['inputs'],local_outputs=stage['outputs'],
                                local_physical_outputs=tuple(name for name in writes if name in syn.variables) if effect_spec['mode']=='vectorised' else (),
                                retained_local_outputs=tuple(name for name in writes if name!=stage['target']),
                                accum=tuple(name for name in template['accum'] if name==stage['target']),
                                guarded_writes=tuple(name for name in template['guarded_writes'] if name==stage['target'])))
                        stage_specs=parts
                        if effect_spec['mode']=='array':
                            stage_specs.append(dict(template,code='',writes=(),accum=(),guarded_writes=(),publish_caller_locals=True))
                    if caller_trace is None and not grouped_caller and effect_spec['mode'] in ('vectorised','array'):
                        body=ast.parse(stage_specs[0]['code']).body
                        call_rows=[i for i,statement in enumerate(body) if any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                                  and node.func.id in captured_functions for node in ast.walk(statement))]
                        if call_rows and len(body)>1:
                            # NumPy's vectorised generator reloads and writes each
                            # persistent statement before the next Python call.
                            # Raw capture roots must observe those physical writes.
                            split=[]
                            for statement in body:
                                target=(statement.targets[0].id if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name)
                                        else statement.target.id if isinstance(statement,ast.AugAssign) and isinstance(statement.target,ast.Name) else None)
                                require(target in syn.variables and target in stage_specs[0]['writes'],
                                        'function',obj,'interleaved batch captures require persistent statement targets')
                                part=dict(stage_specs[0],code=ast.unparse(statement),writes=(target,),
                                          accum=tuple(name for name in stage_specs[0]['accum'] if name==target),
                                          guarded_writes=tuple(name for name in stage_specs[0]['guarded_writes'] if name==target))
                                split.append(part)
                            stage_specs=split
                            if effect_spec['mode']=='array':
                                stage_specs.append(dict(stage_specs[0],code='',writes=(),accum=(),guarded_writes=(),publish_array_locals=True))
                    gate=len(initial);initial.append(0.);initial_parameters.append(None);detached.append(True);binary_states.append(gate)
                    actions.append(dict(owner=0,reads=[gate],writes=[gate],program_set=program_set([[dict(op='constant',value=0.)]]),threshold=None,trigger=None))
                    batch_presence_stages[obj.name]=gate
                    pending_edges=[edge for row in info['path_pending'][obj.name] for edge in row]
                    arrival_keys=[(('pending',k+1),edge) for k,edge in enumerate(pending_edges)]
                    arrival_keys.extend((('new',edge),edge) for edge in sorted(range(len(syn)),key=lambda e:(-int(info['path_delays'][obj.name][e]),int(indices[e]),e)))
                    rows=[]
                    for key,edge in arrival_keys:
                        cell=len(initial);initial.append(0.);initial_parameters.append(None);detached.append(True);binary_states.append(cell)
                        rows.append(dict(key=key,cell=cell,edge=edge))
                        actions.append(dict(owner=0,reads=[cell],writes=[cell],program_set=program_set([[dict(op='constant',value=0.)]]),threshold=None,trigger=None))
                    batch_arrival_rows[obj.name]=rows
                    if random_draws:
                        random_rows={}
                        for row in rows:
                            random_rows[row['key']]={}
                            for draw in random_draws:
                                check_budget(64);cell=len(initial)
                                require(cell<1_000_000,'budget',obj,'batch random state budget exceeded')
                                initial.append(0.);initial_parameters.append(None);detached.append(True)
                                random_rows[row['key']][draw['name']]=cell
                        batch_random_fields[obj.name]=dict(draws=random_draws,rows=random_rows)
                    if caller_trace is not None:
                        from .training_equations import _compile_training_ast
                        local_rows={};copies_by_stage={};seeded=set()
                        for row in rows:
                            local_rows[row['key']]=[]
                            for token in caller_trace['tokens']:
                                check_budget(64);cell=len(initial)
                                require(cell<1_000_000,'budget',obj,'caller local state budget exceeded')
                                initial.append(0.);initial_parameters.append(None);detached.append(token['dtype']!='float')
                                if token['dtype']=='integer':integer_states.add(cell)
                                if token['dtype']=='boolean':binary_states.append(cell)
                                local_rows[row['key']].append(cell)
                        for number,stage in enumerate(caller_trace['stages'],1):
                            seed_tokens=set(stage['inputs'].values())
                            if number==1 and effect_spec['mode']=='array':seed_tokens.update(caller_trace['initial'].values())
                            for token in sorted(seed_tokens):
                                caller_source=caller_trace['tokens'][token]['source']
                                if caller_source is None or token in seeded:continue
                                seeded.add(token)
                                for row in rows:
                                    cell=local_rows[row['key']][token];variable=syn.variables.get(caller_source)
                                    if caller_source in {draw['name'] for draw in random_draws}:origin=random_rows[row['key']][caller_source]
                                    elif variable is not None and id(variable) in storage:origin=address(syn,caller_source,row['edge'])
                                    else:origin=None
                                    if origin is not None:
                                        require(isinstance(origin,int),'function',obj,'caller local reads require fixed addresses')
                                        context=[origin,cell];program=[dict(op='integer_state' if origin in integer_states else 'state',index=0)]
                                    else:
                                        require(variable is not None,'function',obj,f'caller local source {caller_source} is unavailable')
                                        if caller_source in ('t','t_pre','t_post'):
                                            owner_clock=syn.clock if caller_source=='t' else syn.source.clock if caller_source=='t_pre' else syn.target.clock
                                            value=ClockTime(clock_dts.index(float(owner_clock.dt_)))
                                        elif id(variable) in constant_banks:value=constant_parameter(syn,caller_source,row['edge'])
                                        elif caller_source in info['params']:
                                            values,bank=info['params'][caller_source];index=0 if len(values)==1 else row['edge']
                                            value=typed_parameter(bank,index,_state_dtype(variable)) if bank is not None else values[index].item()
                                        else:value=np.asarray(variable.get_value()).reshape(-1)[0].item()
                                        context=[cell];program=_compile_training_ast(ast.Name(id='caller_source',ctx=ast.Load()),parameters={'caller_source':value},states=['v'],typed=True,allow_select=True)
                                    resolved=resolve_parameters([program],row['edge'],context,{})
                                    action=dict(owner=0,reads=context,writes=[cell],program_set=program_set(resolved),threshold=None,trigger=None)
                                    if actions.clock:action['clock']=actions.clock
                                    copies_by_stage.setdefault(number,[]).append(action)
                        batch_caller_locals[obj.name]=dict(trace=caller_trace,rows=local_rows,copies=copies_by_stage,writes=effect_spec['writes'])
                    if caller_trace is None and effect_spec['mode']=='array' and len(stage_specs)>1:
                        require(obj.name not in info['path_runtime'],'function',obj,'batch local arrays require fixed delay routing')
                        identifiers={node.id for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Name)}
                        local_names=sorted(name for name in identifiers if name in syn.variables and id(syn.variables[name]) in storage
                                           and not syn.variables[name].scalar)
                        require(set(effect_spec['writes'])<=set(local_names),'function',obj,
                                'interleaved batch capture writeback requires non-scalar persistent arrays')
                        local_rows={};copies=[]
                        for row in rows:
                            local_rows[row['key']]={}
                            for name in local_names:
                                origin=address(syn,name,row['edge'])
                                require(isinstance(origin,int),'function',obj,'batch local arrays require fixed addresses')
                                check_budget(64);cache=len(initial)
                                require(cache<1_000_000,'budget',obj,'batch local array state budget exceeded')
                                initial.append(0.);initial_parameters.append(None);detached.append(detached[origin])
                                if origin in integer_states:integer_states.add(cache)
                                if origin in binary_states:binary_states.append(cache)
                                local_rows[row['key']][name]=(origin,cache)
                                action=dict(owner=0,reads=[origin,cache],writes=[cache],program_set=program_set([[dict(op='integer_state' if origin in integer_states else 'state',index=0)]]),threshold=None,trigger=None)
                                if actions.clock:action['clock']=actions.clock
                                copies.append(action)
                        batch_array_locals[obj.name]=dict(rows=local_rows,writes=effect_spec['writes'],copies=copies)
                    for k in range(1,len(stage_specs)+1):batch_capture_gates[obj.name+'::numpy-stage:'+str(k)]=gate
                    stage_specs=[None,*stage_specs]
                if len(stage_specs)>1:
                    event_stage_groups[obj.name]=[obj.name,*[obj.name+'::numpy-stage:'+str(k) for k in range(1,len(stage_specs))]]
                for stage_number,stage_spec in enumerate(stage_specs):
                    if obj.name in batch_caller_locals:actions.extend(batch_caller_locals[obj.name]['copies'].get(stage_number,()))
                    if stage_number==1 and obj.name in batch_array_locals:
                        actions.extend(batch_array_locals[obj.name]['copies'])
                    pathway=obj.name if stage_number==0 else obj.name+'::numpy-stage:'+str(stage_number)
                    event_path_origins[pathway]=obj.name
                    stage_code=code if stage_spec is None else stage_spec['code']
                    if stage_spec is not None:event_effect_modes[pathway]=stage_spec
                    histories=[];layout=dict(pending=[],new=[]);queue_layout[pathway]=layout
                    delay_path=dict(name=pathway,start=len(actions),end=0,pending=[],edges=[None]*len(syn))
                    if actions.clock:delay_path['clock']=actions.clock
                    if obj.name in info['path_runtime'] and obj.variables['delay'].scalar:
                        delay_path['shared_delay']=True
                        if id(obj.variables['delay']) in captured_delay_ids:delay_path['captured_shared_delay']=True
                    # Events already queued at conversion precede every newly
                    # emitted event in an arrival bin. Preserve all duplicates and
                    # original entry order, including after a dt/delay change.
                    for remaining,row in enumerate(info['path_pending'][obj.name]):
                        for edge in row:
                            fifo=history(remaining+1,pending=True)
                            action=build_synaptic(syn,edge,stage_code,trigger=dict(external=False,state=True,index=fifo[0]),pathway=pathway,
                                                  event_noise=dict(delay=0,pending=len(delay_path['pending'])+1))
                            delay_path['pending'].append(dict(edge=edge,event=len(actions)-1,states=fifo))
                            histories.append((fifo,action['owner'],action['mask'],None))
                            layout['pending'].append(dict(edge=edge,remaining=remaining,states=fifo))
                    delays=info['path_delays'][obj.name]
                    for edge in sorted(range(len(syn)),key=lambda e:(-int(delays[e]),int(indices[e]),e)):
                        trigger=dict(external=source is input_group,index=int(indices[edge])+(0 if source is input_group else neuron_offsets[source.name]))
                        if buffered and not trigger['external']:trigger=dict(external=False,state=True,index=spike_buffers[trigger['index']])
                        fifo=history(int(delays[edge])) if delays[edge] else []
                        action=build_synaptic(syn,edge,stage_code,trigger=dict(external=False,state=True,index=fifo[0]) if fifo else trigger,pathway=pathway,
                                              event_noise=dict(delay=int(delays[edge])))
                        delay_path['edges'][edge]=dict(event=len(actions)-1,source=trigger,states=fifo)
                        if obj.name in info['path_runtime']:
                            delay_cells=info['path_runtime'][obj.name]['delay']
                            delay_path['edges'][edge]['delay_state']=delay_cells[0 if len(delay_cells)==1 else edge]
                        if fifo:histories.append((fifo,action['owner'],action['mask'],trigger))
                        layout['new'].append(dict(edge=edge,delay=int(delays[edge]),states=fifo))
                    added=snapshot_event_batch(delay_path['start'],len(actions),pathway)
                    if added:
                        delay_path['start']+=added
                        for event in [*delay_path['pending'],*delay_path['edges']]:event['event']+=added
                    for item in histories:advance_history(*item)
                    delay_path['end']=len(actions);delay_paths.append(delay_path)
    # Indirection resolves before executing each statement program. External
    # roots therefore need typed physical cells at the consumer's clock.
    # Sample once before scheduled actions; these fields are read-only and
    # their table contract is independent of the selected graph's writes.
    if external_selector_caches:
        consumers={}
        for action in actions:
            used=set(action['reads'])
            for descriptor in action.get('indirect',{}).get('reads',{}).values():
                used.add(descriptor['index'])
                used.update(cell for row in descriptor['tables'] for cell in row)
            for descriptor in action.get('indirect',{}).get('writes',{}).values():
                used.update(cell for row in descriptor['tables'] for cell in row)
            for cell in used & external_selector_states:
                consumers.setdefault(cell,set()).add(action.get('clock',0))
        prefix=Actions();cache_provenance=[]
        for entry in external_selector_caches.values():
            source=entry['source'];group=entry['group']
            table=external_table(source,group,source['name'])
            columns=[k for k,cell in enumerate(entry['cells']) if cell in consumers]
            execution_clocks=set()
            for k in columns:
                cell=entry['cells'][k]
                descriptor=_TimedState(table,entry['clock'],k,'integer',
                    sample_slot if source['samples'] is not None else None,
                    source['size'] if source['samples'] is not None else None)
                programs_for_read=compile_dynamic_transform('result=field',states={'result':0},
                    parameters={'field':descriptor},state_types={0:'integer'})['programs']
                for clock in sorted(consumers[cell]):
                    # A path can execute on its source clock while reading
                    # its owning Synapses clock's pending time. Refresh on
                    # every reader clock, preserving that pending time.
                    prefix.clock=clock;execution_clocks.add(clock)
                    reads=[cell];programs_for_sample=resolve_parameters(programs_for_read,k,reads,{})
                    prefix.append(dict(owner=0,reads=reads,writes=[cell],program_set=program_set(programs_for_sample),
                                       threshold=None,trigger=None))
            cache_provenance.append(dict(source_variable=source['name'],clock=entry['clock'],bank=table.bank,
                cells=entry['cells'],sampled_columns=columns,execution_clocks=sorted(execution_clocks),
                storage='read-only-consumer-clock-int32-cache',initial_adjoint='zero'))
        shift=len(prefix)
        actions[:0]=prefix
        for path in delay_paths:
            path['start']+=shift;path['end']+=shift
            for event in [*path['pending'],*path['edges']]:event['event']+=shift
        provenance['external_selector_caches']=cache_provenance
    controlled=sorted({tuple(a['mask']) for a in actions if a.get('mask') is not None})
    owned={};migration_bytes=len(controlled)*32
    check_budget(migration_bytes)
    def register(index,mask,restart):
        nonlocal migration_bytes
        if mask is None:return
        mask=tuple(mask);entry=owned.get(index)
        if entry is not None and mask in entry['owners']:return
        cost=16+(64 if entry is None else 0);check_budget(migration_bytes+cost);migration_bytes+=cost
        if entry is None:entry=owned[index]=dict(owners=set(),restart=restart)
        require(entry['restart']==restart,'migration',network,'inconsistent shared-state restart rules')
        entry['owners'].add(mask)
    pinned={index for action in actions if action.get('mask') is None for index in possible_writes(action)}
    for action in actions:
        for index in possible_writes(action):
            if index in restart_rules and index not in pinned:register(index,action.get('mask'),restart_rules[index])
    migration=dict(controlled_masks=[list(mask) for mask in controlled],cells=[dict(index=index,
        owners=[list(mask) for mask in sorted(entry['owners'])],restart=entry['restart']) for index,entry in sorted(owned.items())])
    # v5 executes the ordered action programs. Its retained v4 prefix declares
    # rectangular state shape; action-specific extra selector slots cannot be
    # represented there. Never serialize private unresolved nodes as executable IR.
    placeholders=[]
    for key in ('state_equations','state_resets'):
        for layer,row in enumerate(plan[key]):
            for slot,program in enumerate(row):
                if refractory_expressions[layer] is not None or any(n['op'].startswith('_deferred_') or
                        n['op'] in ('state','integer_state','refractory_active') and n['index']>=len(row) for n in program):
                    row[slot]=[dict(op='integer_state' if provenance['state_types'][layer][slot]=='integer' else 'state',index=slot)]
                    placeholders.append(dict(field=key,layer=layer,state=slot))
    if placeholders:provenance['v4_shape_placeholders']=placeholders
    if external_sources:
        provenance['external_state_inputs']=dict(input_group=input_group.name,
            fields=sorted(source['name'] for source in external_sources.values()),reads=external_reads,
            sources={source['name']:dict(bank=source['bank'],dtype=source['dtype'],shape=source['shape'],
                encoding='signed-high16-low16' if source['dtype']=='integer' else 'plain',
                **({'samples':source['samples']} if source['samples'] is not None else {}))
                for source in external_sources.values() if 'bank' in source},
            sampling='consumer-clock-timed-array',batch_semantics='per-sample-and-shared' if sample_count is not None else 'shared',writes='read-only')
        provenance['timed_inputs']=input_registry.entries
    if external_neuron_states:
        provenance['external_neuron_aliases']={group:{name:dict(slot=slot,storage='inert-prefix-timed-read',
            initial_adjoint='zero',writes='read-only') for slot,name in aliases.items()}
            for group,aliases in external_neuron_states.items()}
    plan.update(schema='b2-dynamic-training-plan-v5',dynamic=dict(migration=migration,clocks=clocks,binary_states=sorted(set(binary_states)),integer_states=sorted(integer_states),integer_parameters=integer_parameters,initial=initial,initial_parameters=initial_parameters,
        detached=detached,voltage=voltage,program_sets=programs,actions=list(actions)))
    if sample_slot is not None:
        plan['dynamic']['sample_index']=dict(slot=sample_slot,samples=sample_count)
        provenance['native_sample_index']=dict(slot=sample_slot,samples=sample_count,writes='read-only',initial_adjoint='zero')
    if buffered:
        plan['dynamic']['spike_buffers']=spike_buffers;plan['dynamic']['clear_spike_buffers']=True;provenance['spike_buffer_layout']=spike_buffer_layout
    plan['dynamic']['delay_layout']=dict(paths=delay_paths,free_cells=[])
    check_budget(sum(256+len(p['name'])+sum(96+len(e['states'])*16 for e in p['edges'])+
                     sum(80+len(e['states'])*16 for e in p['pending']) for p in delay_paths))
    if provenance.get('constant_sources'):
        maps=provenance.get('parameter_mappings',[])
        plan['dynamic']['parameter_maps']=[m['indices'] for m in maps]
        refs=[]
        for l,group in enumerate(layers):
            mapping=provenance['threshold_mappings'][l];ref=plan['threshold_parameters'][l]
            for j in range(len(group)):
                refs.append(None if ref is None else [ref[0],maps[mapping]['indices'][j] if mapping is not None else
                    ref[1]+(j if plan.get('threshold_per_neuron',[False]*len(layers))[l] else 0)])
        plan['dynamic']['threshold_references']=refs
        provenance['synaptic_constant_mappings']=list(constant_indices.values())
    if any(a.get('noise_streams',0) for a in actions):plan['noise_streams']=[len(n) for n in provenance['noise_names']]
    provenance['runtime_parameter_layout']=[dict(**entry,addresses={str(j):r for (ref,j),r in parameter_addresses.items() if ref==i})
                                            for i,entry in enumerate(parameter_references)]
    provenance.update(scope='ordered-dynamic-delayed-synapses',dynamic_state_layout={name:info['runtime'] for name,info in syn_info.items()},
        neuron_state_layout={group.name:{name:cells[group.name,name] for name in names}
                             for group,names in zip(layers,provenance['state_names'])},
        inactive_alias_slots=sorted(unused),storage_semantics='canonical-physical-variable-runtime-index-aliases' if any(any(r is not None for r in row) for row in indirect_cells.values()) else 'canonical-physical-variable-fixed-index-aliases',
        runtime_index_layout={group:{name:row for (owner,name),row in indirect_cells.items() if owner==group and any(r is not None for r in row)}
                              for group in sorted({owner for (owner,_),row in indirect_cells.items() if any(r is not None for r in row)})},
        state_layout='canonical-physical-with-rectangular-neuron-prefix',
        delay_queues=queue_layout,delay_semantics='brian-half-up-ticks-pending-prefix-then-emission-order',
        pathway_state_layout={path:states for info in syn_info.values() for path,states in info['path_runtime'].items()},
        synaptic_noise_domains={name:info['domain'] for name,info in syn_info.items()},
        scheduled_noise_domains=scheduled_noise_domains,
        pathway_alias_semantics='brian-cython-distinct-locals-sorted-writeback',
        conditional_write_semantics='brian-cython-shared-condition-name-last-referenced-index',
        refractory_activity_layout=activity,refractory_condition_indices=neuron_condition_indices,
        refractory_age_layout=refractory_age,refractory_lastspike_layout=refractory_lastspike,
        refractory_timestamp_words=refractory_words,
        refractory_expression_semantics='hard-detached-gate-current-duration-int32-next-tick-age' if refractory_age else None,
        regular_runner_layout=regular_layout,mutable_capture_layout=mutable_capture_layout,readonly_capture_layout=readonly_capture_layout,
        mutable_constant_layout={g.name:{n:storage[id(v)] for n,v in g.variables.items() if id(v) in mutable_constant_ids and id(v) in storage} for g in [*layers,*synapses]},
        regular_semantics='scalar-snapshot-and-sorted-writeback-then-vector-loop',
        summed_updates=summations,summed_semantics='clear-target-then-original-edge-order-cython',
        clock_semantics='brian-network-min-clock-coalescing-f64-detached',
        event_gradient='one-hard-event-gate-per-composed-path-with-surrogate-vjp',timestamp_gradient='stop-gradient',
        event_callback_modes={name:dict(mode=spec['mode'],write_order=list(spec['writes']),accum=list(spec['accum'])) for name,spec in event_effect_modes.items()},
        event_callback_snapshots=event_effect_snapshots,event_callback_stage_groups=event_stage_groups,
        event_callback_array_locals={path:dict(rows=[dict(key=list(key),fields={name:dict(source=source,local=local) for name,(source,local) in fields.items()})
                                                      for key,fields in spec['rows'].items()],write_order=list(spec['writes']))
                                    for path,spec in batch_array_locals.items()},
        event_callback_random_fields={path:dict(draws=spec['draws'],rows=[dict(key=list(key),fields=fields) for key,fields in spec['rows'].items()])
                                     for path,spec in batch_random_fields.items()},
        event_callback_caller_locals={path:dict(trace=spec['trace'],rows=[dict(key=list(key),fields=fields) for key,fields in spec['rows'].items()])
                                     for path,spec in batch_caller_locals.items()},
        event_callback_whole_locals={path:{str(token):dict(name=source,cells=field[0],dtype=field[1]) for token,(source,field) in fields.items()}
                                    for path,fields in batch_whole_locals.items()},
        phased_event_gradient='surrogate-through-operational-NumPy-stage-order-and-row-gates' if event_stage_groups else None)
    provenance.pop('snapshot_sha256',None)
    bundle.initial_state=list(initial)
    provenance['snapshot_sha256']=hashlib.sha256(canonical_bytes(dict(plan=plan,weights=weights,initial=bundle.initial_membrane,
        initial_state=bundle.initial_state,provenance=provenance))).hexdigest()
    return bundle
