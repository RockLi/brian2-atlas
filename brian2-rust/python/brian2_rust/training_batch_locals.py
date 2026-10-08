"""Symbolic NumPy caller local lifetimes between captured callback statements."""
import ast
import copy
import numpy as np
from brian2.core.variables import ArrayVariable,AuxiliaryVariable
from .training_effects import (Effects,ArrayCell,StateEffectFunction,
                               bind_state_effect_captures)


class ShapeEffects(Effects):
    """Infer a fixed broadcast result without choosing an arrival count.

    A selected vector can broadcast with a fixed N>1 vector only at counts
    one or N. The effect compiler retains the actual runtime shape check.
    Singleton captures leave the result's selected length dynamic.
    """
    def broadcast_length(self,values):
        result=super().broadcast_length(values)
        arrays=[value for value in values if isinstance(value,ArrayCell) and not value.zero_dim]
        fixed={value.length for value in arrays if value.length not in (None,1) and not value.selection_shape}
        unknown=[value for value in arrays if value.length is None]
        if result is None and len(fixed)==1 and unknown and all(value.selection_shape for value in unknown):return next(iter(fixed))
        return result

    def result(self,node,*operands):
        value=super().result(node,*operands)
        arrays=[operand for operand in operands if isinstance(operand,ArrayCell) and not operand.zero_dim]
        if isinstance(value,ArrayCell) and value.length is None and arrays and all(operand.selection_shape or operand.length==1 for operand in arrays):
            value.selection_shape=True
        return value


def batch_local_lifetimes(code,variables,functions,*,mode,synthetic_types=None):
    """Trace object identity; never execute callbacks or numeric model code.

    Array augmented assignments retain the object, ordinary assignment may
    rebind it, and vectorised persistent reads reload advanced-index copies.
    Stage inputs use the old object while output destinations use the new one.
    Returned raw closure vectors require a separate whole-vector binding.
    """
    if mode not in ('array','vectorised'):raise ValueError('batch locals require a NumPy array mode')
    tree=ast.parse(code)
    if len(code)>16384 or not 1<=len(tree.body)<=256:raise ValueError('batch local statement budget exceeded')
    used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    normal={name:var for name,var in variables.items() if name in used and hasattr(var,'dtype') and name not in functions and not isinstance(var,AuxiliaryVariable)}
    types={name:('boolean' if np.dtype(var.dtype).kind=='b' else 'integer' if np.dtype(var.dtype).kind=='i' else 'float') for name,var in normal.items()}
    types.update({} if synthetic_types is None else synthetic_types)
    states={name:k for k,name in enumerate(types)}
    arrays={name for name,var in normal.items() if isinstance(var,ArrayVariable) and not var.scalar}|set(synthetic_types or {})
    captures=set();protected=set();bound={};capture_bindings={}
    for name,descriptor in functions.items():
        if type(descriptor) is not StateEffectFunction:continue
        bindings={}
        for capture,value in descriptor.captured_arrays:
            key='_b2_local_capture_'+str(len(states));slot=len(states);states[key]=slot;captures.add(key)
            capture_bindings[slot]=[name,capture]
            types[key]='boolean' if value.dtype.kind=='b' else 'integer' if value.dtype.kind=='i' else 'float'
            if capture in descriptor.parameter_captures or not value.flags.writeable:protected.add(key)
            bindings[capture]=key
        bound[name]=bind_state_effect_captures(descriptor,bindings,readonly=descriptor.parameter_captures)
    engine=ShapeEffects(states,bound,array_states=arrays|captures,writable_states=(arrays|captures)-protected,
                   state_types={slot:types[name] for name,slot in states.items()},eager_limit=128)
    for name in arrays:engine.environment[name].origin=None;engine.environment[name].selection_shape=True
    objects=[];by_identity={};tokens=[]
    def token(value,source=None):
        key=id(value)
        if key not in by_identity:
            by_identity[key]=len(tokens);objects.append(value)
            whole=isinstance(value,ArrayCell) and any(isinstance(node,ast.Name) and node.id in captures for node in ast.walk(engine.materialize(value)))
            tokens.append(dict(dtype=engine.dtype(value),array=isinstance(value,ArrayCell),zero_dim=isinstance(value,ArrayCell) and value.zero_dim,source=source,whole_capture=whole,
                               length=value.length if isinstance(value,ArrayCell) else None,
                               capture_binding=capture_bindings.get(value.origin) if isinstance(value,ArrayCell) else None))
        return by_identity[key]
    initial={name:token(engine.environment[name],name) for name in types if name not in captures}
    stages=[]
    for statement in tree.body:
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):target=statement.targets[0].id
        elif isinstance(statement,ast.AugAssign) and isinstance(statement.target,ast.Name):target=statement.target.id
        else:raise ValueError('batch caller locals require simple assignments')
        reads={n.id for n in ast.walk(statement) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        if isinstance(statement,ast.AugAssign):reads.add(target)
        if mode=='vectorised':
            for name in arrays&reads:
                engine.environment[name]=ArrayCell(ast.Name(id=name,ctx=ast.Load()),dtype=types[name],selection_shape=True);token(engine.environment[name],name)
        before={name:token(value) for name,value in engine.environment.items() if name not in captures}
        original={name:ast.dump(engine.materialize(value),include_attributes=False) for name,value in engine.environment.items() if isinstance(value,ArrayCell) and name not in captures}
        condition=getattr(normal.get(target),'conditional_write',None)
        if condition is not None:
            scope=dict(engine.environment)
            for name,value in list(scope.items()):
                if isinstance(value,ArrayCell) and name not in captures:scope[name]=copy.deepcopy(value)
            engine.statement(statement,scope,physical=True)
            left=engine.environment[target];left.value=engine.materialize(scope[target])
        else:engine.statement(statement,engine.environment,physical=True)
        after={name:token(value) for name,value in engine.environment.items() if name not in captures}
        changed=[name for name,old in original.items() if name in engine.environment and ast.dump(engine.materialize(engine.environment[name]),include_attributes=False)!=old]
        destinations={target:after[target]}
        if mode=='array':
            for name in changed:
                if after[name] not in destinations.values():destinations[name]=after[name]
        inputs={name:before[name] for name in reads if name in before}
        if target not in inputs:inputs[target]=before.get(target,after[target])
        stages.append(dict(code=ast.unparse(statement),target=target,inputs=inputs,outputs=destinations))
        # Bound expression growth while keeping array object identity. Scalar
        # values are immutable; aliases keep the same new symbolic value.
        scalar={}
        for name,value in list(engine.environment.items()):
            if name in captures:
                value.value=ast.Name(id=name,ctx=ast.Load());continue
            index=after[name];symbol='_b2_local_token_'+str(index);engine.input_dtypes[symbol]=tokens[index]['dtype']
            if isinstance(value,ArrayCell):value.value=ast.Name(id=symbol,ctx=ast.Load())
            else:
                replacement=scalar.setdefault(index,ast.Name(id=symbol,ctx=ast.Load()))
                engine.environment[name]=replacement;by_identity[id(replacement)]=index;objects.append(replacement)
        engine.executed.clear();engine.executed_keys.clear();engine.operation_count=0
    return dict(tokens=tokens,initial=initial,stages=stages,final={name:token(value) for name,value in engine.environment.items() if name not in captures})


def whole_capture_carry_groups(trace,persistent):
    """Carry borrowed physical roots without projecting them into event lanes."""
    persistent=set(persistent);groups=[];current=[]
    for stage in trace['stages']:
        current.append(stage)
        if stage['target'] in persistent:groups.append(current);current=[]
    if current:return None
    previous=set();result=[];buffers=set();live={}
    for group in groups:
        defined=set();bindings={}
        for stage in group:
            statement=ast.parse(stage['code']).body[0]
            reads={n.id for n in ast.walk(statement) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
            if isinstance(statement,ast.AugAssign):reads.add(stage['target'])
            for name in reads&previous-defined:
                token=trace['tokens'][stage['inputs'][name]]
                if token['capture_binding'] is not None:bindings[name]=token['capture_binding']
                elif token['source'] is None and token['array'] and not token['zero_dim'] and type(token['length']) is int and token['length']>0:
                    index=stage['inputs'][name];bindings[name]=dict(token=index);buffers.add(index)
                else:return None
            if stage['target'] not in persistent:defined.add(stage['target'])
            live.update({name:index for name,index in stage['outputs'].items() if name not in persistent})
        result.append(dict(code='\n'.join(stage['code'] for stage in group),whole_local_inputs=bindings,live=dict(live)))
        previous.update(defined)
    seeded=set()
    for group in result:
        outputs={}
        inputs={entry['token'] for entry in group['whole_local_inputs'].values() if isinstance(entry,dict)}
        for name,index in group.pop('live').items():
            if index in buffers and index not in seeded and index not in inputs:
                outputs.setdefault(name,index);seeded.add(index)
        group['whole_local_outputs']=outputs
    return result
