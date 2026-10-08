"""Symbolic NumPy caller local lifetimes between captured callback statements."""
import ast
import copy
import numpy as np
from brian2.core.variables import ArrayVariable,AuxiliaryVariable
from .training_effects import (Effects,ArrayCell,StateEffectFunction,
                               bind_state_effect_captures)


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
    captures=set();protected=set();bound={}
    for name,descriptor in functions.items():
        if type(descriptor) is not StateEffectFunction:continue
        bindings={}
        for capture,value in descriptor.captured_arrays:
            key='_b2_local_capture_'+str(len(states));slot=len(states);states[key]=slot;captures.add(key)
            types[key]='boolean' if value.dtype.kind=='b' else 'integer' if value.dtype.kind=='i' else 'float'
            if capture in descriptor.parameter_captures or not value.flags.writeable:protected.add(key)
            bindings[capture]=key
        bound[name]=bind_state_effect_captures(descriptor,bindings,readonly=descriptor.parameter_captures)
    engine=Effects(states,bound,array_states=arrays|captures,writable_states=(arrays|captures)-protected,
                   state_types={slot:types[name] for name,slot in states.items()},eager_limit=128)
    for name in arrays:engine.environment[name].origin=None
    objects=[];by_identity={};tokens=[]
    def token(value,source=None):
        key=id(value)
        if key not in by_identity:
            by_identity[key]=len(tokens);objects.append(value)
            whole=isinstance(value,ArrayCell) and any(isinstance(node,ast.Name) and node.id in captures for node in ast.walk(engine.materialize(value)))
            tokens.append(dict(dtype=engine.dtype(value),array=isinstance(value,ArrayCell),zero_dim=isinstance(value,ArrayCell) and value.zero_dim,source=source,whole_capture=whole))
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
                engine.environment[name]=ArrayCell(ast.Name(id=name,ctx=ast.Load()),dtype=types[name]);token(engine.environment[name],name)
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


def whole_capture_local_groups(trace,persistent):
    """Keep temporary whole vectors inside their actual statement scope.

    Projection is only valid at a persistent indexed write, never at a local
    assignment. A temporary that survives that boundary needs full-vector
    persistent storage and cannot use this grouping.
    """
    persistent=set(persistent);groups=[];current=[]
    for stage in trace['stages']:
        current.append(stage)
        if stage['target'] in persistent:
            groups.append(current);current=[]
    if current:return None
    created=set()
    for group in groups:
        for stage in group:
            reads={n.id for n in ast.walk(ast.parse(stage['code'])) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
            if isinstance(ast.parse(stage['code']).body[0],ast.AugAssign):reads.add(stage['target'])
            if reads&created:return None
        created.update(stage['target'] for stage in group if stage['target'] not in persistent)
    return ['\n'.join(stage['code'] for stage in group) for group in groups]
