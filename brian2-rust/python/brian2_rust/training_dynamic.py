"""v5 dynamic action construction; execution and reverse mode remain native.

This is the action IR layer. Brian object/schedule lowering is implemented
separately; these helpers never execute a user's model in Python.
"""
import ast
import copy
from .training_equations import StateSlot, _compile_training_ast, coerce_state_expression


def compile_dynamic_transform(statements, *, states, parameters=None, write_guards=None, state_types=None,
                              mutable_write_guards=()):
    """Compose one event's sequential assignments into simultaneous outputs.

    states maps source names to context slots. Multiple names may alias one slot;
    writes through either alias are immediately visible through all its aliases.
    Local temporaries are accepted and do not create persistent state cells.
    Guard writes require explicit Boolean slots in mutable_write_guards; later
    statements then use the updated local guard. Other guards remain read-only.
    """
    if not isinstance(statements,str) or len(statements)>16384:
        raise ValueError('dynamic statements exceed source budget')
    states=dict(states);parameters={} if parameters is None else dict(parameters)
    write_guards={} if write_guards is None else dict(write_guards)
    if not states or any(not isinstance(name,str) or not name.isidentifier() or type(slot) is not int or not 0<=slot<64 for name,slot in states.items()):
        raise ValueError('dynamic context requires named slots 0..63')
    if states.keys()&parameters.keys():raise ValueError('dynamic state and parameter names overlap')
    try:code=ast.parse(statements,mode='exec').body
    except (SyntaxError,RecursionError):raise ValueError('invalid dynamic statements') from None
    if not 1<=len(code)<=256:raise ValueError('dynamic statement budget exceeded')
    state_types={} if state_types is None else dict(state_types)
    if any(slot not in states.values() or dtype not in ('float','integer','boolean') for slot,dtype in state_types.items()):
        raise ValueError('invalid dynamic state types')
    width=max(states.values())+1
    slots={slot:ast.Name(id=f'_b2_context_{slot}',ctx=ast.Load()) for slot in states.values()}
    if any(type(slot) is not int or type(gate) is not int or slot not in slots or gate not in slots or slot==gate for slot,gate in write_guards.items()):
        raise ValueError('write guards require distinct state/context slots')
    mutable_write_guards=set(mutable_write_guards)
    if not mutable_write_guards<=set(write_guards.values()) or any(state_types.get(slot)!='boolean' for slot in mutable_write_guards):
        raise ValueError('mutable write guards require Boolean guard slots')
    bindings={f'_b2_context_{slot}':StateSlot(slot,state_types.get(slot,"float")) for slot in slots}
    if bindings.keys()&parameters.keys():raise ValueError('reserved dynamic context name')
    values={name:slots[slot] for name,slot in states.items()};written=set()
    class Substitute(ast.NodeTransformer):
        def visit_Name(self,node):return values.get(node.id,node)
    for statement in code:
        if isinstance(statement,ast.Assign) and len(statement.targets)==1:
            target=statement.targets[0];expression=statement.value
        elif isinstance(statement,ast.AugAssign) and isinstance(statement.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.BitAnd,ast.BitOr,ast.BitXor,ast.LShift,ast.RShift)):
            target=statement.target
            if not isinstance(target,ast.Name) or target.id not in values:raise ValueError('augmented assignment needs a known state or temporary')
            expression=ast.BinOp(left=ast.Name(id=target.id,ctx=ast.Load()),op=statement.op,right=statement.value)
            if isinstance(statement.op,ast.Mod):
                # Cython emits %= directly, without the renderer's ((a%b)+b)%b.
                expression=ast.Call(func=ast.Name(id='_b2_mod',ctx=ast.Load()),args=[expression.left,expression.right],keywords=[])
        else:raise ValueError('dynamic paths require simple assignments')
        if not isinstance(target,ast.Name) or target.id in parameters or target.id.startswith('_b2_'):
            raise ValueError('invalid dynamic assignment target')
        expression=Substitute().visit(copy.deepcopy(expression))
        if sum(1 for _ in ast.walk(expression))>4096:raise ValueError('composed dynamic expression exceeds budget')
        if target.id in states:
            slot=states[target.id]
            if state_types:expression=coerce_state_expression(expression,state_types.get(slot,"float"))
            if slot in write_guards.values() and slot not in mutable_write_guards:raise ValueError('write guard cells are read-only within a transform')
            if slot in write_guards:
                expression=ast.IfExp(test=copy.deepcopy(slots[write_guards[slot]]),body=expression,orelse=slots[slot])
            written.add(slot);slots[slot]=expression
            for name,index in states.items():
                if index==slot:values[name]=expression
        else:values[target.id]=expression
    if not written:raise ValueError('dynamic path must write persistent state')
    targets=sorted(written)
    programs=[_compile_training_ast(slots[slot],parameters={**parameters,**bindings},states=['v'],deduplicate=True,allow_select=True,typed=bool(state_types)) for slot in targets]
    return dict(writes=targets,programs=programs,context_size=width)


def dynamic_action(transform, reads, *, owner, program_set, trigger=None, detach_trigger=False,
                   parameter_index=0, noise_domain=0, noise_entity=0, noise_streams=0, mask=None):
    """Bind a composed transform to global runtime state cells."""
    reads=list(reads)
    if len(reads)!=transform['context_size']:raise ValueError('dynamic context shape mismatch')
    return dict(mask=copy.deepcopy(mask),owner=owner,reads=reads,writes=[reads[i] for i in transform['writes']],program_set=program_set,
                threshold=None,trigger=copy.deepcopy(trigger),detach_trigger=detach_trigger,parameter_index=parameter_index,
                noise_domain=noise_domain,noise_entity=noise_entity,noise_streams=noise_streams)
