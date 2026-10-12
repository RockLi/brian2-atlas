"""Fixed physical columns for once-per-block scheduled random callbacks."""
import ast
from .training_effects import compile_state_effect_transform


def compile_whole_regular(code,bindings,parameters,*,column=0,write_order=(),compact_eager_roots=False,separate_whole_eager=False,predicate_outputs=None):
    """Compile one output column while retaining all real array operands.

    bindings describe already loaded NumPy locals or physical closure storage;
    choosing separate cells for block copies preserves the original writeback.
    """
    states={};params=dict(parameters);reads=[];slots={};types={};arrays=set();writable=set();param_arrays=set();param_types={};vectors={};aliases={};canonical={};lengths={};guarded_outputs={}
    used={node.id for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Name)}
    used.update(source for value in parameters.values() for _,source in getattr(value,'capture_bindings',()))
    for number,(name,binding) in enumerate(bindings.items()):
        if name not in used:continue
        values=binding['values'];dtype=binding['dtype'];array=binding.get('array',False)
        if not values:raise ValueError('whole scheduled arrays require actual columns')
        chosen=min(column,len(values)-1) if array else 0
        if array and 'length' in binding:
            length_name='__regular_whole_length_'+str(number);cell=binding['length']
            if cell not in slots:slots[cell]=len(reads);reads.append(cell)
            states[length_name]=slots[cell];types[slots[cell]]='integer'
        symbols=[name if j==chosen else '__regular_whole_column_'+str(number)+'_'+str(j) for j in range(len(values))]
        for symbol,value in zip(symbols,values):
            if type(value) is int:
                if value not in slots:slots[value]=len(reads);reads.append(value)
                states[symbol]=slots[value];types[slots[value]]=dtype
                if array:
                    arrays.add(symbol)
                    if binding.get('writable',False):writable.add(symbol)
                    aliases[symbol]=canonical.setdefault(value,symbol)
            else:
                params[symbol]=value[1] if type(value) is tuple and value[0]=='parameter' else value
                param_types[symbol]=dtype
                if array:param_arrays.add(symbol)
            if array:
                vectors[symbol]=tuple(symbols)
                if 'length' in binding:
                    lengths[symbol]=length_name
                    if symbol in states:guarded_outputs[symbol]=symbols.index(symbol)

    if len(reads)>64:raise ValueError('whole scheduled callback exceeds64 physical context slots')
    predicate_slots={}
    for name,coefficients in (predicate_outputs or {}).items():
        for symbol in vectors.get(name,(name,)):
            if symbol in states:predicate_slots[states[symbol]]=coefficients
    transform=compile_state_effect_transform(code,states=states,parameters=params,array_states=arrays,writable_states=writable,
        state_types=types,array_parameters=param_arrays,parameter_types=param_types,predicate_outputs=predicate_slots,
        array_aliases=aliases,capture_vectors=vectors,whole_array_locals=set(vectors),whole_array_lengths=lengths,whole_array_outputs=guarded_outputs,unconditional_states=arrays,
        unconditional_parameters=param_arrays,writeback_order=write_order,eager_limit=128,scheduled_scalar_noise=True,
        compact_eager_roots=compact_eager_roots,separate_whole_eager=separate_whole_eager and bool(vectors),whole_writeback={node.id for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}&set(vectors)&set(bindings))
    programs=dict(transform['unconditional_programs'])
    programs.update(zip(transform['writes'],transform['programs']))
    return dict(reads=reads,programs=programs,eager_programs=transform['unconditional_eager_programs'],names=states,output_arrays=transform['output_arrays'],effect_writes=transform['effect_writes'])


def cached_poisson_code(code,records):
    """Expose rate-only prefixes, then replace sampled counts by real caches."""
    tree=ast.parse(code);sites={row['name']:row for row in records if row['kind']=='poisson'};prefixes=[]
    class Cache(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id in sites:
                row=sites[node.func.id]
                return ast.Call(func=ast.Name(id='_b2_scheduled_poisson',ctx=ast.Load()),args=[self.visit(node.args[0]),ast.Name(id=row['cache_name'],ctx=ast.Load())],keywords=[])
            return self.generic_visit(node)
    done=[]
    for statement in tree.body:
        call=statement.value
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id in sites:
            row=sites[call.func.id]
            prefix=ast.unparse(ast.fix_missing_locations(ast.Module(body=[*done,ast.Assign(targets=[ast.Name(id='__regular_whole_rate',ctx=ast.Store())],value=call.args[0])],type_ignores=[])))
            prefixes.append((row,prefix))
        done.append(Cache().visit(statement))
    if len(prefixes)!=len(sites):raise ValueError('scheduled Poisson sampler must have one exposed rate stage')
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=done,type_ignores=[]))),prefixes
