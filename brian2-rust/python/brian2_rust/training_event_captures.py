"""Whole closure arrays in NumPy's scalar event fallback."""
import ast
from .training_effects import (StateEffectFunction,bind_state_effect_captures,
                              compile_state_effect_transform)


def _columns(fields,names,reads,parameters,state_types,typed_parameter,kind):
    """Allocate real columns once; repeat short independent views only symbolically.

    The effect interpreter checks original capture lengths before combining
    vectors. Thus unequal independent buffers and singleton broadcasting work,
    without silently accepting an incompatible elementwise NumPy operation.
    """
    lengths={name:len(field[0]) if field[0] is not None else len(field[3].get_value())
             for name,field in fields.items()}
    count=max(lengths.values());arrays=set();readonly=set();sources={};types={}
    if count>64 or len(reads)+sum(lengths[name] for name,field in fields.items() if field[0] is not None)>64:
        raise ValueError('whole event captures exceed 64 context slots')
    for name,field in fields.items():
        sources[name]=[]
        for column in range(lengths[name]):
            source=name+'_'+kind+'_column_'+str(column);sources[name].append(source)
            if source in names or source in parameters:raise ValueError('reserved event capture name collision')
            if field[0] is None:
                parameters[source]=typed_parameter(field[2],field[4][column] if len(field)>4 else column,field[1]);readonly.add(source);types[source]=field[1]
            else:
                slot=len(reads);names[source]=slot;reads.append(field[0][column]);state_types[slot]=field[1];arrays.add(source)
    bindings=[{name:columns[min(column,len(columns)-1)] for name,columns in sources.items()} for column in range(count)]
    vectors={source:tuple(columns) for columns in sources.values() for source in columns}
    return bindings,lengths,arrays,readonly,types,vectors


def compile_scalar_event_captures(code,names,reads,parameters,fields,*,
                                  state_types,typed_parameter,write_guards=None):
    """One FIFO event calls Python once, updating every captured column.

    Ordinary pathway locals remain scalar copies. Closure arrays are physical
    vectors; expand their elementwise expressions into one atomic event action.
    The existing event trigger consequently gates all closure outputs together.
    """
    names=dict(names);parameters=dict(parameters);state_types=dict(state_types)
    bindings,lengths,arrays,parameter_arrays,parameter_types,vectors=_columns(fields,names,reads,parameters,state_types,typed_parameter,'event')
    capture_slots={names[name] for name in arrays}
    canonical={};aliases={}
    for name,slot in names.items():
        if name in arrays:aliases[name]=canonical.setdefault(reads[slot],name)
    outputs={};eager=[]
    for column,mapping in enumerate(bindings):
        bound=dict(parameters)
        for name,descriptor in parameters.items():
            if type(descriptor) is StateEffectFunction and descriptor.captured_arrays:
                bound[name]=bind_state_effect_captures(descriptor,
                    {capture:mapping[source] for capture,source in descriptor.capture_bindings},
                    readonly=descriptor.parameter_captures)
        transform=compile_state_effect_transform(code,states=names,parameters=bound,
            array_states=arrays,writable_states=arrays,state_types=state_types,
            array_parameters=parameter_arrays,
            parameter_types=parameter_types,eager_limit=128,array_aliases=aliases,capture_vectors=vectors,
            write_guards=write_guards,scalar_write_guards=bool(write_guards))
        if any(name not in arrays for name in transform['output_arrays']):
            raise ValueError('scalar event writeback cannot store a whole captured array')
        eager.extend(transform['programs'])
        own={names[source] for name,source in mapping.items() if source in names and column<lengths[name]}
        for slot,program in zip(transform['writes'],transform['programs']):
            if slot in own or column==0 and slot not in capture_slots:outputs[slot]=program
    # A scalar writeback may overwrite a closure output at the same address.
    # Keep the whole callback's checked work even when that output is dropped.
    from .protocol import canonical_bytes
    def checked(program):
        nodes=[];interned={};ends=[]
        for group in [*eager,program]:
            remap={}
            for i,original in enumerate(group):
                node=dict(original);references=['arg','left','right','low','high','condition','yes','no','rate','slope','scale']
                if node['op'] in ('timed_parameter','parameter_gather','integer_parameter_gather'):
                    references+=['index']
                    if node['op']=='timed_parameter':references+=['time']
                for key in references:
                    if key in node:node[key]=remap[node[key]]
                key=canonical_bytes(node)
                if key not in interned:interned[key]=len(nodes);nodes.append(node)
                remap[i]=interned[key]
            ends.append(remap[len(group)-1])
        result=ends[-1]
        integer_ops={'poisson','integer_constant','integer_state','integer_parameter','integer_neuron_parameter',
                     'integer_mapped_parameter','integer_parameter_gather','integer_cast','integer_binary',
                     'integer_select','integer_neg','integer_sequence','_deferred_integer_parameter'}
        for left in reversed(ends[:-1]):
            if nodes[result]['op'] in integer_ops:nodes.append(dict(op='integer_sequence',left=left,right=result))
            else:nodes.append(dict(op='sequence',left=left,right=result,boolean=nodes[result]['op'] in
                {'boolean_cast','boolean_and','boolean_or','boolean_not','eager_boolean_and','eager_boolean_or','integer_compare'}))
            result=len(nodes)-1
        if len(nodes)>128:raise ValueError('whole scalar event callback exceeds 128 native nodes')
        return nodes
    return dict(writes=list(outputs),programs=[checked(p) for p in outputs.values()],context_size=len(reads)),names,capture_slots


def compile_batch_event_captures(code,names,reads,parameters,fields,*,state_types,
                                 array_names,parameter_arrays,parameter_types,reload,typed_parameter,selected_output=None,selected_vector_factory=None,selected_accumulators=(),selected_guards=None,selected_call_presence=None,selected_row_operands=(),copied_array_aliases=None,retained_copied_outputs=(),whole_local_inputs=None,whole_local_outputs=None):
    """Separate independent whole-column writes from selected-row copies."""
    names=dict(names);parameters=dict(parameters);state_types=dict(state_types)
    whole_local_inputs={} if whole_local_inputs is None else dict(whole_local_inputs)
    whole_local_outputs={} if whole_local_outputs is None else dict(whole_local_outputs)
    if any(name in names or name in array_names or name in parameter_arrays or source not in fields for name,source in whole_local_inputs.items()):
        raise ValueError('invalid physical caller array binding')
    if any(source not in fields or fields[source][0] is None for source in whole_local_outputs.values()):
        raise ValueError('whole caller outputs require private physical vectors')
    bindings,lengths,arrays,readonly,capture_types,vectors=_columns(fields,names,reads,parameters,state_types,typed_parameter,'batch')
    parameter_types={**parameter_types,**capture_types}
    canonical={};aliases={}
    for name,slot in names.items():
        if name in arrays:aliases[name]=canonical.setdefault(reads[slot],name)
    explicit={node.id for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}
    copied_sources={name:canonical[reads[slot]] for name,slot in names.items()
                    if name in array_names and name not in explicit and reads[slot] in canonical}
    normal=None;whole={};checks=[];selected_vectors=None
    if selected_guards:
        if selected_vector_factory is None or len(set(selected_guards.values()))!=1:
            raise ValueError('masked batch captures require one compact callback selection')
        selected_vectors=selected_vector_factory(names,reads,parameters,state_types,max(lengths.values()))
    for column,mapping in enumerate(bindings):
        bound=dict(parameters);column_names=dict(names);column_aliases=dict(aliases);column_vectors=dict(vectors);column_types=dict(parameter_types);local_arrays=set();local_parameters=set()
        for local,field in whole_local_inputs.items():
            source=mapping[field];bound.pop(local,None)
            column_vectors[local]=tuple(local if name==source else name for name in vectors[source])
            if source in names:
                column_names[local]=names[source];local_arrays.add(local);column_aliases[local]=aliases.get(source,source)
            else:
                bound[local]=bound[source];local_parameters.add(local)
                column_types[local]=parameter_types[source]
        for name,descriptor in parameters.items():
            if type(descriptor) is StateEffectFunction and descriptor.captured_arrays:
                bound[name]=bind_state_effect_captures(descriptor,
                    {capture:mapping[source] for capture,source in descriptor.capture_bindings},
                    readonly=descriptor.parameter_captures)
        def compile_column(whole_eager=False):
            # Compaction can allocate scalar inputs after the first compile
            # requests selected vectors. Keep the column's local aliases while
            # including those newly allocated context slots on the retry.
            column_names.update({name:slot for name,slot in names.items() if name not in column_names})
            column_code=code if not whole_local_outputs else code+'\n'+'\n'.join(mapping[field]+' = '+local for local,field in whole_local_outputs.items())
            return compile_state_effect_transform(column_code,states=column_names,parameters=bound,
                array_states=set(array_names)|arrays|local_arrays,writable_states=set(array_names)|arrays|local_arrays,
                copied_array_states=array_names,array_parameters=set(parameter_arrays)|readonly|local_parameters,temporary_parameters=parameter_arrays,
                parameter_types=column_types,state_types=state_types,eager_limit=128,
                reload_arrays_each_statement=reload,unconditional_states=arrays|local_arrays,retained_array_states=arrays|local_arrays,
                unconditional_parameters=readonly|local_parameters,whole_eager=whole_eager,array_aliases=column_aliases,capture_vectors=column_vectors,selected_output=selected_output,copied_array_sources=copied_sources,selected_vectors=selected_vectors,selected_accumulators=selected_accumulators,separate_whole_eager=True,
                selected_guards=selected_guards,selected_call_presence=selected_call_presence,
                selected_row_operands=selected_row_operands,
                copied_array_aliases=copied_array_aliases,retained_copied_outputs=retained_copied_outputs,
                whole_array_locals=whole_local_inputs,
                eager_guard=next(iter(selected_guards.values())) if selected_guards else None)
        try:
            transform=compile_column()
            if transform.get('selected_whole_mixed',False) and selected_vector_factory is not None and selected_vectors is None:
                selected_vectors=selected_vector_factory(names,reads,parameters,state_types,max(lengths.values()))
                bound.update({name:value for name,value in parameters.items() if name not in bound})
                transform=compile_column()
        except ValueError as error:
            if selected_vector_factory is None or selected_vectors is not None or 'whole reset captures cannot depend on event-selected arrays' not in str(error):raise
            selected_vectors=selected_vector_factory(names,reads,parameters,state_types,max(lengths.values()))
            # Retry with an actual compact batch, never a single owner's value.
            bound.update({name:value for name,value in parameters.items() if name not in bound})
            transform=compile_column()
        own={names[source] for name,source in mapping.items() if source in names and column<lengths[name]}
        # A read-only external buffer has persistent state but no write output.
        # Keep its discarded eager work for every column as a separate check.
        if not readonly and not own<=set(transform['unconditional_programs']):transform=compile_column(whole_eager=True)
        if normal is None:normal=transform
        if transform['unconditional_eager_program'] is not None:checks.append(transform['unconditional_eager_program'])
        checks.extend(transform['unconditional_eager_programs'])
        whole.update({slot:program for slot,program in transform['unconditional_programs'].items() if slot in own})
    normal['unconditional_programs']=whole
    # Columns may share exactly the same eager root. All checks read the same
    # snapshot, so retain its first occurrence without allocating duplicate
    # scratch actions. Distinct columns and source-order roots stay ordered.
    from .protocol import canonical_bytes
    unique={}
    for program in checks:unique.setdefault(canonical_bytes(program),program)
    normal['unconditional_checks']=list(unique.values())
    return normal,names,{names[name] for name in arrays}
