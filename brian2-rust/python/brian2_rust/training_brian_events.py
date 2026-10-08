"""Retain NumPy event code generation's array versus scalar execution mode."""
import ast
import copy
import numpy as np
from brian2.codegen.translation import make_statements
from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2.core.functions import Function,DEFAULT_FUNCTIONS
from .training_functions import lower_pure_function
from .training_effects import lower_state_effect_function


def prepare_effect_event(syn,path,code,variables):
    effects={}
    called={node.func.id for node in ast.walk(ast.parse(code)) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    for name,variable in variables.items():
        if name in called and type(variable) is Function and name not in DEFAULT_FUNCTIONS:
            try:lower_pure_function(variable)
            except ValueError:
                try:effects[name]=lower_state_effect_function(variable)
                except (ValueError,TypeError,OSError,SyntaxError,RecursionError):pass
    if not effects:return None
    scalar,vector=make_statements(code,variables,np.float64,optimise=True)
    indices=copy.copy(syn.variables.indices)
    for name,variable in variables.items():
        flag=getattr(variable,'conditional_write',None)
        if flag is not None:indices[flag.name]=indices[name]
    generator=NumpyCodeGenerator(variables,indices,path,set(),NumpyCodeObject,path.name,'synapses')
    repeated=generator.has_repeated_indices(vector)
    lines=generator.translate_one_statement_sequence(vector)
    mode=('scalar' if any(line.startswith('for _idx in _full_idx:') for line in lines)
          else 'vectorised' if repeated else 'array')
    _,writes,_,_=generator.arrays_helper(vector)
    accum=[];aliases=[];prior=[]
    if mode=='vectorised':
        for statement in vector:
            used={n.id for n in ast.walk(ast.parse(statement.expr,mode='eval')) if isinstance(n,ast.Name)}
            for previous in prior:
                for name in used:
                    if name in variables and variables[name] is variables[previous]:aliases.append((previous,name))
            if statement.var in variables and statement.inplace and indices[statement.var]!='_idx':
                accum.append(statement.var);prior.append(statement.var)
    def source(statements):
        return '\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {s.expr}' for s in statements)
    # A read after an aliased scatter must execute after every event row's
    # preceding writes. Retain vectorisation rather than substituting a scalar
    # loop: its callback argument and return ownership would be different.
    stages=[];current=[];previous=[]
    if mode=='vectorised':
        for statement in vector:
            used={n.id for n in ast.walk(ast.parse(statement.expr,mode='eval')) if isinstance(n,ast.Name)}
            dependency=any(name in variables and variables[name] is variables[written]
                           for written in previous for name in used)
            if dependency and current:
                stages.append(current);current=[];previous=[]
            current.append(statement)
            if statement.var in variables and statement.inplace and indices[statement.var]!='_idx':previous.append(statement.var)
    else:current=list(vector)
    if current:stages.append(current)
    parts=[];stream_offset=0;created=set();prior_parts=[];prior_offsets=[];prior_guards=[]
    for statements in stages:
        used={n.id for statement in statements for n in ast.walk(ast.parse(statement.expr,mode='eval')) if isinstance(n,ast.Name)}
        replay=bool(created & used)
        _,part_writes,_,part_conditions=generator.arrays_helper(statements)
        guarded_writes=tuple(statement.var for statement in statements
                            if mode!='vectorised' or statement.inplace and indices[statement.var]!='_idx')
        part_guards={name:flag for name,flag in part_conditions.items() if name in guarded_writes}
        part_accum=tuple(dict.fromkeys(statement.var for statement in statements
                         if statement.var in variables and statement.inplace and indices[statement.var]!='_idx')) if mode=='vectorised' else ()
        code_part=source([*scalar,*statements])
        replay_fields={};replay_code=[];replay_noise={};replay_guards={}
        if replay:
            for earlier,earlier_part in enumerate(prior_parts):
                tree=ast.parse(earlier_part)
                renamed={name:'__event_replay_'+str(earlier)+'_'+name for name in
                         {node.id for node in ast.walk(tree) if isinstance(node,ast.Name)}
                         if name in variables and not isinstance(variables[name],Function) and hasattr(variables[name],'dtype')}
                # Implicit conditional flags are absent from the abstract RHS,
                # but must be captured alongside a replayed guarded statement.
                renamed.update({flag:'__event_replay_'+str(earlier)+'_'+flag for flag in prior_guards[earlier].values()})
                replay_fields.update({alias:dict(stage=earlier,variable=name) for name,alias in renamed.items()})
                for target,flag in prior_guards[earlier].items():
                    replay_fields[renamed[flag]]['index_variable']=target
                    replay_guards[renamed[target]]=renamed[flag]
                class Rename(ast.NodeTransformer):
                    def visit_Name(self,node):
                        return ast.copy_location(ast.Name(id=renamed.get(node.id,node.id),ctx=node.ctx),node)
                tree=Rename().visit(tree)
                class ReplayDraw(ast.NodeTransformer):
                    counter=prior_offsets[earlier]
                    def visit_Call(self,node):
                        if isinstance(node.func,ast.Name) and node.func.id in ('rand','randn','poisson'):
                            kind=node.func.id;stream=self.counter;self.counter+=1;alias='__event_replay_random_'+str(stream)
                            replay_noise[alias]=dict(kind=kind,stream=stream)
                            if kind=='poisson':return ast.copy_location(ast.Call(func=ast.Name(id=alias,ctx=ast.Load()),args=[self.visit(arg) for arg in node.args],keywords=node.keywords),node)
                            return ast.copy_location(ast.Name(id=alias,ctx=ast.Load()),node)
                        return self.generic_visit(node)
                replay_code.append(ast.unparse(ReplayDraw().visit(tree)))
            code_part='\n'.join([*replay_code,code_part])
        parts.append(dict(mode=mode,code=code_part,writes=tuple(part_writes),accum=part_accum,alias_pairs=(),stream_offset=stream_offset,replay_fields=replay_fields,replay_noise=replay_noise,
                          indexed_locals=tuple(statement.var for statement in vector if statement.op==':='),
                          # The vectorised NumPy generator applies conditions
                          # only to endpoint in-place ufunc.at operations. Its
                          # plain indexed assignments do not use the mask.
                          guarded_writes=guarded_writes,replay_guards=replay_guards))
        prior_parts.append(source([*scalar,*statements]));prior_offsets.append(stream_offset)
        prior_guards.append(part_guards)
        stream_offset+=sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('rand','randn','poisson')
                           for node in ast.walk(ast.parse(code_part)))
        created.update(statement.var for statement in statements if statement.op==':=')
    return dict(mode=mode,code=source([*scalar,*vector]),writes=tuple(writes),accum=tuple(dict.fromkeys(accum)),alias_pairs=tuple(dict.fromkeys(aliases)),stages=parts)
