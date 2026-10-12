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
    callback_random_planning=True

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

    def expression(self,node,environment):
        value=super().expression(node,environment)
        from .training_equations import PoissonNoise
        if (isinstance(value,ArrayCell) and isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                and type(environment.get(node.func.id)) is PoissonNoise):value.selection_shape=True
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                and node.func.id in getattr(self,'callback_poisson_names',())):
            rate=super().expression(node.args[0],environment)
            scalar=not isinstance(rate,ArrayCell) or rate.zero_dim
            if scalar:self.scalar_poisson_sites.add(node.func.id)
            self.poisson_shapes[node.func.id]=dict(scalar=scalar,length=None if scalar else rate.length)
            if getattr(self,'scheduled_shapes',False):
                if scalar:return self.materialize(value)
                value.selection_shape=rate.selection_shape
        return value


def batch_local_lifetimes(code,variables,functions,*,mode,synthetic_types=None,statement_groups=None,masked_statement_groups=(),normal_domains=None,callback_poisson_names=(),whole_normal_domains=False):
    """Trace object identity; never execute callbacks or numeric model code.

    Array augmented assignments retain the object, ordinary assignment may
    rebind it, and vectorised persistent reads reload advanced-index copies.
    Stage inputs use the old object while output destinations use the new one.
    Returned raw closure vectors require a separate whole-vector binding.
    """
    if mode not in ('array','vectorised'):raise ValueError('batch locals require a NumPy array mode')
    tree=ast.parse(code)
    if statement_groups is not None and (len(statement_groups)!=len(tree.body) or any(type(g) is not int or g<0 for g in statement_groups) or list(statement_groups)!=sorted(statement_groups)):
        raise ValueError('invalid caller statement scopes')
    if len(code)>16384 or not 1<=len(tree.body)<=256:raise ValueError('batch local statement budget exceeded')
    normal_domains={} if normal_domains is None else normal_domains
    scalar_normal_sites=set();normal_shapes={}
    masked_statement_groups=set(masked_statement_groups)
    if masked_statement_groups and statement_groups is None:raise ValueError('masked caller scopes require statement groups')
    used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    normal={name:var for name,var in variables.items() if name in used and hasattr(var,'dtype') and name not in functions and not isinstance(var,AuxiliaryVariable)}
    types={name:('boolean' if np.dtype(var.dtype).kind=='b' else 'integer' if np.dtype(var.dtype).kind=='i' else 'float') for name,var in normal.items()}
    types.update({} if synthetic_types is None else synthetic_types)
    states={name:k for k,name in enumerate(types)}
    arrays={name for name,var in normal.items() if isinstance(var,ArrayVariable) and not var.scalar}|set(synthetic_types or {})
    captures=set();protected=set();bound={};capture_bindings={}
    from .training_equations import PureFunction
    for name,descriptor in functions.items():
        if type(descriptor) is PureFunction:
            bound[name]=descriptor;continue
        if type(descriptor) is not StateEffectFunction:continue
        bindings={}
        for capture,value in descriptor.captured_arrays:
            key='_b2_local_capture_'+str(len(states));slot=len(states);states[key]=slot;captures.add(key)
            capture_bindings[slot]=[name,capture]
            types[key]='boolean' if value.dtype.kind=='b' else 'integer' if value.dtype.kind=='i' else 'float'
            if capture in descriptor.parameter_captures or not value.flags.writeable:protected.add(key)
            bindings[capture]=key
        bound[name]=bind_state_effect_captures(descriptor,bindings,readonly=descriptor.parameter_captures)
    from .training_equations import PoissonNoise
    poisson={name:value for name,value in functions.items() if type(value) is PoissonNoise}
    engine=ShapeEffects(states,bound,poisson,array_callables=poisson,array_states=arrays|captures,writable_states=(arrays|captures)-protected,
                   state_types={slot:types[name] for name,slot in states.items()},eager_limit=128)
    engine.callback_poisson_names=set(callback_poisson_names);engine.scalar_poisson_sites=set();engine.poisson_shapes={};engine.scheduled_shapes=whole_normal_domains
    for name in arrays:engine.environment[name].origin=None;engine.environment[name].selection_shape=True
    objects=[];by_identity={};tokens=[]
    def token(value,source=None):
        key=id(value)
        if key not in by_identity:
            by_identity[key]=len(tokens);objects.append(value)
            whole=isinstance(value,ArrayCell) and any(isinstance(node,ast.Name) and node.id in captures for node in ast.walk(engine.materialize(value)))
            tokens.append(dict(dtype=engine.dtype(value),array=isinstance(value,ArrayCell),zero_dim=isinstance(value,ArrayCell) and value.zero_dim,source=source,whole_capture=whole,
                               length=value.length if isinstance(value,ArrayCell) else None,
                               selection_shape=isinstance(value,ArrayCell) and value.selection_shape,
                               capture_binding=capture_bindings.get(value.origin) if isinstance(value,ArrayCell) else None))
        return by_identity[key]
    initial={name:token(engine.environment[name],name) for name in types if name not in captures and name not in poisson}
    stages=[];scope_reads={}
    if statement_groups is not None:
        for group,statement in zip(statement_groups,tree.body):
            scope_reads.setdefault(group,set()).update(n.id for n in ast.walk(statement) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load))
            if isinstance(statement,ast.AugAssign) and isinstance(statement.target,ast.Name):scope_reads[group].add(statement.target.id)
    for number,statement in enumerate(tree.body):
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):target=statement.targets[0].id
        elif isinstance(statement,ast.AugAssign) and isinstance(statement.target,ast.Name):target=statement.target.id
        else:raise ValueError('batch caller locals require simple assignments')
        reads={n.id for n in ast.walk(statement) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        if isinstance(statement,ast.AugAssign):reads.add(target)
        for draw_name in reads&normal_domains.keys():
            domain_names=normal_domains[draw_name]
            if isinstance(domain_names,tuple):
                # size=None uses the broadcast shape of both parameters.
                # Whole vectors must not silently become selected-size draws.
                operands=[engine.environment.get(name) for name in domain_names]
                domain_arrays=[value for value in operands if isinstance(value,ArrayCell) and not value.zero_dim]
                if not domain_arrays:scalar_normal_sites.add(draw_name)
                normal_shapes[draw_name]=dict(scalar=not domain_arrays,length=engine.broadcast_length(operands) if domain_arrays else None)
                if domain_arrays and not whole_normal_domains and (not any(value.selection_shape for value in domain_arrays) or any(not value.selection_shape and value.length!=1 for value in domain_arrays)):
                    raise ValueError('callback Normal/Uniform inferred size currently requires selected array broadcast domain')
            else:
                domain=engine.environment.get(domain_names)
                if not isinstance(domain,ArrayCell) or not whole_normal_domains and not domain.selection_shape:raise ValueError('callback Normal/Uniform currently requires selected array size')
                normal_shapes[draw_name]=dict(scalar=False,length=domain.length)
        new_scope=statement_groups is None or number==0 or statement_groups[number]!=statement_groups[number-1]
        if mode=='vectorised' and new_scope:
            for name in arrays&(reads if statement_groups is None else scope_reads[statement_groups[number]]):
                engine.environment[name]=ArrayCell(ast.Name(id=name,ctx=ast.Load()),dtype=types[name],selection_shape=True);token(engine.environment[name],name)
        masked_reads=(arrays&normal.keys()&scope_reads[statement_groups[number]] if statement_groups is not None and statement_groups[number] in masked_statement_groups else set())
        engine.selected_operand_copies=set(masked_reads)
        before={name:token(value) for name,value in engine.environment.items() if name not in captures and name not in poisson}
        original={name:ast.dump(engine.materialize(value),include_attributes=False) for name,value in engine.environment.items() if isinstance(value,ArrayCell) and name not in captures}
        condition=getattr(normal.get(target),'conditional_write',None)
        if condition is not None:
            scope=dict(engine.environment)
            for name,value in list(scope.items()):
                if isinstance(value,ArrayCell) and name not in captures:scope[name]=copy.deepcopy(value)
            engine.statement(statement,scope,physical=True)
            left=engine.environment[target];left.value=engine.materialize(scope[target])
        else:engine.statement(statement,engine.environment,physical=True)
        after={name:token(value) for name,value in engine.environment.items() if name not in captures and name not in poisson}
        changed=[name for name,old in original.items() if name in engine.environment and ast.dump(engine.materialize(engine.environment[name]),include_attributes=False)!=old]
        destinations={target:after[target]}
        if mode=='array' or statement_groups is not None:
            for name in changed:
                if after[name] not in destinations.values():destinations[name]=after[name]
        inputs={name:before[name] for name in reads if name in before}
        if statement_groups is not None and new_scope:
            inputs.update({name:before[name] for name in arrays&scope_reads[statement_groups[number]]})
        if target not in inputs:inputs[target]=before.get(target,after[target])
        stages.append(dict(code=ast.unparse(statement),target=target,inputs=inputs,outputs=destinations,masked_reads=sorted(masked_reads)))
        # Bound expression growth while keeping array object identity. Scalar
        # values are immutable; aliases keep the same new symbolic value.
        scalar={}
        for name,value in list(engine.environment.items()):
            if name in poisson:continue
            if name in captures:
                value.value=ast.Name(id=name,ctx=ast.Load());continue
            index=after[name];symbol='_b2_local_token_'+str(index);engine.input_dtypes[symbol]=tokens[index]['dtype']
            if isinstance(value,ArrayCell):value.value=ast.Name(id=symbol,ctx=ast.Load())
            else:
                replacement=scalar.setdefault(index,ast.Name(id=symbol,ctx=ast.Load()))
                engine.environment[name]=replacement;by_identity[id(replacement)]=index;objects.append(replacement)
        engine.executed.clear();engine.executed_keys.clear();engine.operation_count=0
    return dict(tokens=tokens,initial=initial,stages=stages,scalar_normal_sites=sorted(scalar_normal_sites),scalar_poisson_sites=sorted(engine.scalar_poisson_sites),normal_shapes=normal_shapes,poisson_shapes=engine.poisson_shapes,final={name:token(value) for name,value in engine.environment.items() if name not in captures and name not in poisson})


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
                elif token['source'] is None and token['array'] and not token['zero_dim'] and (type(token['length']) is int and token['length']>0 or token['selection_shape']):
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


def poisson_whole_local_stages(trace):
    """Keep whole objects and callback scalars separate from arrival lanes."""
    whole={i for i,t in enumerate(trace['tokens']) if t['array'] and not t['zero_dim']
           and (t['capture_binding'] is not None or t['source'] is None and type(t['length']) is int and not t['selection_shape'])}
    scalar_whole={i for i,t in enumerate(trace['tokens']) if not t['array'] and t['source'] is None} if trace.get('scalar_normal_sites') else set()
    whole.update(scalar_whole)
    buffers={i for i in whole if trace['tokens'][i]['capture_binding'] is None}
    stages=[]
    for stage in trace['stages']:
        inputs={};normal_inputs={};outputs={};normal_outputs={}
        for name,index in stage['inputs'].items():
            if index in whole:
                inputs[name]=trace['tokens'][index]['capture_binding'] or dict(token=index)
            else:normal_inputs[name]=index
        for name,index in stage['outputs'].items():
            if index in buffers:outputs[name]=index
            elif index not in whole:normal_outputs[name]=index
        code=stage['code']
        scalar_inputs={name for name,index in stage['inputs'].items() if index in scalar_whole}
        if scalar_inputs:
            class ReadScalar(ast.NodeTransformer):
                def visit_AugAssign(self,node):
                    if isinstance(node.target,ast.Name) and node.target.id in scalar_inputs:
                        read=self.visit(ast.Name(id=node.target.id,ctx=ast.Load()))
                        return ast.Assign(targets=[node.target],value=ast.BinOp(left=read,op=node.op,right=self.visit(node.value)))
                    return self.generic_visit(node)
                def visit_Name(self,node):
                    if node.id in scalar_inputs and isinstance(node.ctx,ast.Load):
                        return ast.Call(func=ast.Name(id='_b2_scalar_value',ctx=ast.Load()),args=[node],keywords=[])
                    return node
            code=ast.unparse(ast.fix_missing_locations(ReadScalar().visit(ast.parse(code))))
        if stage['target'] in stage['outputs'] and stage['outputs'][stage['target']] in whole:
            # Borrowing a readonly root may only mutate its argument or execute
            # eager checks. A detached dummy output keeps that native action
            # explicit without projecting or copying the returned rate object.
            anchor='_b2poisson_anchor';index=stage['outputs'][stage['target']]
            normal_inputs[anchor]=index;normal_outputs[anchor]=index
            code=anchor+' = '+anchor+'\n'+code
        stages.append(dict(stage,code=code,inputs=normal_inputs,outputs=normal_outputs,
                           whole_local_inputs=inputs,whole_local_outputs=outputs))
    return stages,buffers,whole
