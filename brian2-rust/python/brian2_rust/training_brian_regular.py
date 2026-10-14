"""Validate scheduled Brian assignments and retain their scalar/vector blocks."""
import copy

import numpy as np
from brian2.codegen.translation import analyse_identifiers, make_statements
from brian2.groups.group import CodeRunner
from brian2.groups.neurongroup import SubexpressionUpdater
from brian2.groups.subgroup import Subgroup
from brian2.equations.unitcheck import check_units_statements
from brian2.core.functions import Function, DEFAULT_FUNCTIONS
from .training_functions import lower_pure_function
from .training_effects import lower_state_effect_function, StateEffectFunction, bind_state_effect_captures, capture_array_key, same_capture_storage


def is_regular(obj, owners):
    return (type(obj) is CodeRunner or type(obj) is SubexpressionUpdater
            and getattr(obj.group, 'subexpression_updater', None) is obj) and any(obj.group == owner for owner in owners)


def regular_children(group, synapses):
    """Containers inserted by Group.run_on_clock are not extra execution."""
    return {obj for obj in group.contained_objects
            if is_regular(obj, [group]) or obj in synapses or
            isinstance(obj, Subgroup) and obj.source == group and
            all(is_regular(child, [obj]) for child in obj.contained_objects)}


def prepare_regular(runner, namespace, dt, require):
    from brian2.codegen.generators.cython_generator import CythonCodeGenerator
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    require(type(runner) in (CodeRunner, SubexpressionUpdater) and runner.template == 'stateupdate'
            and not runner.template_kwds and not runner.needed_variables
            and not runner.override_conditional_write
            and not hasattr(runner, 'variables')
            and 'update_abstract_code' not in runner.__dict__,
            'runner', runner, 'only standard run_regularly or constant-over-dt stateupdate runners are supported')
    require(runner.active, 'inactive', runner, 'inactive runners must be removed explicitly')
    require(isinstance(runner.abstract_code, str) and len(runner.abstract_code) <= 16384,
            'budget', runner, 'regular source exceeds statement budget')
    group = runner.group
    _, known, unknown = analyse_identifiers(runner.abstract_code, group.variables, recursive=True)
    variables = group.resolve_all(sorted(known | unknown), run_namespace=namespace)
    conditions = {}
    for variable in variables.values():
        flag = getattr(variable, 'conditional_write', None)
        if flag is not None and (flag.name not in variables or variables[flag.name] is flag):
            conditions[flag.name] = flag
    variables.update(conditions)
    check_units_statements(runner.abstract_code, variables)
    indices = copy.copy(group.variables.indices)
    for name, variable in variables.items():
        flag = getattr(variable, 'conditional_write', None)
        if flag is not None:
            indices[flag.name] = indices[name]
    # Index dependencies are needed by the array helpers but not resolve_all.
    for name in list(variables):
        index = indices[name]
        seen = set()
        while index not in ('0', '_idx') and index not in seen:
            seen.add(index)
            variables[index] = group.variables[index]
            index = indices[index]
    effects = {}
    for name, variable in variables.items():
        if type(variable) is Function and name not in DEFAULT_FUNCTIONS:
            try:
                lower_pure_function(variable)
            except ValueError:
                try:
                    effects[name] = lower_state_effect_function(variable)
                except (ValueError, TypeError, OSError, SyntaxError, RecursionError):
                    # The public resolver supplies the detailed conversion error.
                    pass
    # Identity arithmetic changes whether NumPy receives borrowed storage.
    # Use Brian's own optimizer for actions containing authenticated effects.
    scalar, vector = make_statements(runner.abstract_code, variables, np.float64,
                                     optimise=bool(effects))
    generator = CythonCodeGenerator(variables, indices, runner, {'_idx'}, CythonCodeObject,
                                    runner.name, 'stateupdate', allows_scalar_write=True)
    sr, sw, si, guards = generator.arrays_helper(scalar)
    vr, vw, vi, _ = generator.arrays_helper(vector)
    # NumPy write_arrays iterates this exact set. ArrayVariable names are
    # resolved in the same sorted order as create_runner_codeobj, so preserve
    # its iteration order rather than choosing a different alias winner.
    numpy_write_order = dict(scalar=tuple(sw), vector=tuple(vw))
    numpy_mode='array'
    if effects:
        from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
        from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
        numpy=NumpyCodeGenerator(variables,indices,runner,{'_idx'},NumpyCodeObject,
                                 runner.name,'stateupdate',allows_scalar_write=True)
        if numpy.has_repeated_indices(vector):
            lines=numpy.translate_one_statement_sequence(vector)
            numpy_mode='scalar' if any(line.startswith('for _idx in _full_idx:') for line in lines) else 'vectorised'
    vector_scalar_reads={name for name in vr if variables[name].scalar and name not in vw}
    for name in list(vr):
        if variables[name].scalar and name not in vw:
            sr.add(name)
            vr.remove(name)
    require(all(not variables[name].read_only and not variables[name].constant
                for name in sw | vw), 'runner', runner, 'regular writes require mutable variables')
    return dict(runner=runner, group=group, variables=variables, indices=indices,
                scalar=scalar, vector=vector, scalar_arrays=sr | sw,
                scalar_writes=sw, vector_writes=vw, guards=guards,
                state_effects=bool(effects), numpy_write_order=numpy_write_order,numpy_mode=numpy_mode,
                vector_scalar_reads=vector_scalar_reads)


def mutated_regular_constants(spec):
    """Discover physical constant-array effects with the authenticated compiler."""
    from .training_effects import compile_state_effect_transform
    from .training_equations import NormalNoise,UniformNoise,PoissonNoise
    from brian2.core.variables import ArrayVariable
    import ast
    if not spec['state_effects'] or spec.get('numpy_mode')=='scalar' or not len(spec['group']):return set()
    variables=dict(spec['variables']);states={};types={};arrays=set();copied=set();parameters={};physical={};canonical={}
    from .training_inputs import TimedInputRegistry,TimedArray
    timed=TimedInputRegistry()
    for name,var in variables.items():
        if type(var) is TimedArray:
            # Probe shape/ownership with the production resolver; no bank or
            # callback data is executed or retained by this local analysis.
            parameters[name]=timed.resolve(var,spec['group'],name,lambda name,values:0,max_values=var.values.size)
        elif isinstance(var,Function):
            if name in DEFAULT_FUNCTIONS:continue
            try:parameters[name]=lower_pure_function(var)
            except ValueError:
                try:parameters[name]=lower_state_effect_function(var)
                except (ValueError,TypeError,OSError,SyntaxError,RecursionError):return set()
        elif hasattr(var,'dtype'):
            slot=len(states);states[name]=slot
            dtype=np.dtype(var.dtype);types[slot]='integer' if dtype.kind in 'iu' else 'boolean' if dtype.kind=='b' else 'float'
            if isinstance(var,ArrayVariable) and not var.scalar:
                arrays.add(name)
                if spec['indices'][name]!='_idx':copied.add(name)
            physical[slot]=canonical.setdefault(id(var),slot)
    external_captures={}
    for function_name,descriptor in list(parameters.items()):
        if type(descriptor) is not StateEffectFunction or not descriptor.captured_arrays:continue
        bindings={}
        for capture_name,array in descriptor.captured_arrays:
            source=next((name for name,var in variables.items() if name in arrays-copied and isinstance(var,ArrayVariable)
                         and same_capture_storage(var.get_value(),array)),None)
            if source is None:
                variable=next((var for owner in spec.get('capture_owners',[spec['group']])
                    for var in owner.variables.values() if isinstance(var,ArrayVariable)
                    and same_capture_storage(var.get_value(),array)),None)
                if variable is not None:
                    source='__physical_discovery_capture_'+str(len(states));slot=len(states)
                    variables[source]=variable;states[source]=slot;arrays.add(source)
                    physical[slot]=canonical.setdefault(id(variable),slot)
                    dtype=np.dtype(variable.dtype);types[slot]='integer' if dtype.kind in 'iu' else 'boolean' if dtype.kind=='b' else 'float'
            if source is None:
                key=capture_array_key(array)
                if key not in external_captures:
                    source='__discovery_capture_'+str(len(external_captures));slot=len(states)
                    states[source]=slot;arrays.add(source);physical[slot]=slot
                    types[slot]='integer' if array.dtype==np.dtype('int32') else 'boolean' if array.dtype==np.dtype('bool') else 'float'
                    external_captures[key]=source
                source=external_captures[key]
            bindings[capture_name]=source
        parameters[function_name]=bind_state_effect_captures(descriptor,bindings)
    arrays-=set(spec['guards'].values())
    tree=ast.parse('\n'.join(f'{s.var} {"=" if s.op==":=" else s.op} {s.expr}' for s in [*spec.get('scalar',()),*spec['vector']]))
    temporary=set()
    class Draw(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id in ('rand','randn','poisson'):
                kind=node.func.id;name='__constant_effect_draw_'+str(len(temporary));temporary.add(name)
                parameters[name]=(PoissonNoise if kind=='poisson' else NormalNoise if kind=='randn' else UniformNoise)(0)
                return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=[self.visit(a) for a in node.args],keywords=[]) if kind=='poisson' else ast.Name(id=name,ctx=ast.Load())
            return self.generic_visit(node)
    tree=Draw().visit(tree)
    try:
        transform=compile_state_effect_transform(ast.unparse(tree),states=states,parameters=parameters,array_states=arrays,
                    writable_states={n for n in arrays if n not in variables or not variables[n].read_only},state_types=types,
                    physical_slots=physical,copied_array_states=copied,array_parameters=temporary,temporary_parameters=temporary,
                    writeback_order=spec['numpy_write_order']['vector'],write_guards=spec['guards'],
                    indexed_guard_reads=arrays|temporary)
    except (ValueError,RecursionError):return set()
    writes=set(transform['effect_writes'])
    return {id(variables[name]) for name,slot in states.items() if slot in writes and name in arrays-copied and name in variables
            and variables[name].constant and not variables[name].read_only}


def mutated_equation_constants(group,code,namespace,*,threshold=False):
    """Inspect a generated whole-array integration/threshold callback block."""
    import ast
    from brian2.core.variables import AuxiliaryVariable
    from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
    from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
    if not len(group):return set()
    variables={**DEFAULT_FUNCTIONS,**group.variables}
    if threshold:variables['_cond']=AuxiliaryVariable(name='_cond',dtype=bool,scalar=False)
    _,_,unknown=analyse_identifiers(code,variables,recursive=True)
    variables.update(group.resolve_all(unknown,run_namespace=namespace))
    effects=False
    for node in ast.walk(ast.parse(code)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            var=variables.get(node.func.id)
            if isinstance(var,Function) and node.func.id not in DEFAULT_FUNCTIONS:
                try:lower_pure_function(var)
                except ValueError:
                    try:lower_state_effect_function(var);effects=True
                    except (ValueError,TypeError,OSError,SyntaxError,RecursionError):pass
    if not effects:return set()
    scalar,vector=make_statements(code,variables,np.float64,optimise=not threshold)
    indices=copy.copy(group.variables.indices)
    generator=NumpyCodeGenerator(variables,indices,group,{'_idx'},NumpyCodeObject,group.name,'stateupdate')
    _,writes,_,conditions=generator.arrays_helper(vector)
    return mutated_regular_constants(dict(group=group,state_effects=True,numpy_mode='array',variables=variables,
            indices=indices,scalar=scalar,vector=vector,guards={name:flag for name,flag in conditions.items() if name in writes},
            numpy_write_order={'vector':tuple(writes)}))
