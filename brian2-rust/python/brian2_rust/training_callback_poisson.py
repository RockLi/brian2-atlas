"""Separate scalar callback counts from their actual arrival-lane sampler."""
import ast
import copy
from .training_effects import StateEffectFunction


def split_scalar_callback_poisson(code,scalar_sites,groups):
    tree=ast.parse(code)
    if len(tree.body)!=len(groups):raise ValueError('invalid scalar Poisson scopes')
    occupied={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    if any(n.startswith('_b2_scalar_poisson_') for n in occupied):raise ValueError('reserved scalar Poisson local collision')
    output=[];scopes=[];helpers={};domains={};caches={}
    def emit(statement,group):output.append(statement);scopes.append(group)
    for statement,group in zip(tree.body,groups):
        call=statement.value
        if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id in scalar_sites):
            emit(statement,group);continue
        if not (isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name)
                and len(call.args)==1 and not call.keywords):raise ValueError('scalar callback Poisson requires an exposed rate stage')
        number=len(caches);rate=copy.deepcopy(call.args[0])
        if not isinstance(rate,ast.Name):
            name='_b2_scalar_poisson_rate_'+str(number)
            emit(ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],value=rate),group)
            rate=ast.Name(id=name,ctx=ast.Load())
        cache='_b2_scalar_poisson_cache_'+str(number);helper='_b2_scalar_poisson_shape_'+str(number)
        helpers[helper]=StateEffectFunction(('rate','noise'),'return _b2_shape_noise(rate,0.,noise)')
        domains[cache]=(rate.id,);caches[cache]=call.func.id
        emit(ast.Assign(targets=[ast.Name(id=cache,ctx=ast.Store())],value=ast.Call(func=call.func,args=[copy.deepcopy(rate)],keywords=[])),group)
        emit(ast.Assign(targets=statement.targets,value=ast.Call(func=ast.Name(id=helper,ctx=ast.Load()),args=[rate,ast.Name(id=cache,ctx=ast.Load())],keywords=[])),group)
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=output,type_ignores=[]))),scopes,helpers,domains,caches


def guard_first_arrival(program,prior_slots):
    """Lazy native select: inactive lanes neither sample nor register a score."""
    result=[dict(node) for node in program];value=len(result)-1
    integer=result[value]['op'] in {'poisson','integer_constant','integer_state','integer_parameter','integer_cast','integer_binary','integer_select','integer_sequence','integer_neg'}
    result.append(dict(op='constant',value=0.));total=len(result)-1
    for slot in prior_slots:
        result.append(dict(op='state',index=slot));right=len(result)-1
        result.append(dict(op='add',left=total,right=right));total=len(result)-1
    result.append(dict(op='constant',value=0.));zero=len(result)-1
    result.append(dict(op='equality',left=total,right=zero,unequal=False));condition=len(result)-1
    result.append(dict(op='integer_constant' if integer else 'constant',value=0 if integer else 0.));inactive=len(result)-1
    result.append(dict(op='integer_select' if integer else 'select',condition=condition,yes=value,no=inactive))
    if len(result)>128:raise ValueError('scalar Poisson first-arrival guard exceeds native node budget')
    return result
