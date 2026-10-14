"""Read bounded pure Brian callbacks without executing their Python code."""
import ast
import bisect
import builtins
import copy
import __future__
import dis
import inspect
import math
import textwrap
from types import CodeType, FunctionType

import numpy as np
from brian2.core.functions import (Function, FunctionImplementation,
                                   FunctionImplementationContainer, DEFAULT_FUNCTIONS)
from brian2.units.fundamentalunits import Quantity, Unit, Dimension, DIMENSIONLESS, get_dimensions, check_units

from .training_equations import PureFunction


_MATH = ('exp', 'log', 'tanh', 'sqrt', 'sin', 'cos', 'clip', 'minimum', 'maximum',
         'tan', 'cosh', 'sinh', 'log10', 'expm1', 'log1p', 'exprel', 'arccos',
         'arcsin', 'arctan', 'ceil', 'abs', 'sign', 'floor', 'int', 'bool')
_FUNCTION_IDENTITIES=tuple((candidate,name) for name in _MATH for candidate in (
    getattr(np,name,None),getattr(math,name,None),getattr(builtins,name,None),
    DEFAULT_FUNCTIONS[name].pyfunc if name in DEFAULT_FUNCTIONS else None) if candidate is not None)
_FUNCTION_IDENTITIES+=((np.where,'_b2_where'),(np.logical_and,'_b2_logical_and'),
                       (np.logical_or,'_b2_logical_or'),(np.logical_not,'_b2_logical_not'))
_NUMPY_SCALAR_TYPES={np.dtype(name).type for name in (
    'int8','int16','int32','int64','uint8','uint16','uint32','uint64',
    'float16','float32','float64','longdouble','bool')}
_PRIVATE_ARRAY_CONSTRUCTOR = np.array
_PRIVATE_ARRAY_DTYPES = (float, np.float64)
_UNIT_CHECK_WRAPPER_CODE = check_units()(lambda x: x).__code__


def _keyword_default_bindings(function, signature):
    """Snapshot omitted keyword-only scalars from the loaded function object."""
    if not signature.kwonlyargs:return []
    defaults=function.__kwdefaults__
    if type(defaults) is not dict:
        raise ValueError('pure function keyword-only arguments need immutable numeric defaults')
    bindings=[]
    for argument in signature.kwonlyargs:
        if argument.arg not in defaults:
            raise ValueError('pure function keyword-only arguments need immutable numeric defaults')
        value=defaults[argument.arg]
        # Read actual defaults rather than evaluate source expressions. Mutable
        # objects and NumPy scalar arithmetic need a separate storage contract.
        if (type(value) not in (bool,int,float) or
                type(value) is float and not math.isfinite(value)):
            raise ValueError('pure function keyword-only defaults must be finite immutable Python scalars')
        if type(value) is int and not -2**31<=value<2**31:
            raise ValueError('pure function keyword-only integer default requires int32')
        bindings.append(ast.Assign(targets=[ast.Name(id=argument.arg,ctx=ast.Store())],
                                   value=ast.Constant(value)))
    return bindings


def _bind_wrapper_packets(definition, scope, count):
    """Specialize positional-only call packets without invoking the wrapper."""
    signature=definition.args
    positional=signature.posonlyargs+signature.args
    if (signature.defaults
            or not 0<=count<=16 or count<len(positional)):
        raise ValueError('pure wrapper needs a bounded fixed positional call signature')
    packet=signature.vararg.arg if signature.vararg else None
    keywords=signature.kwarg.arg if signature.kwarg else None
    if not packet and count!=len(positional):
        raise ValueError('pure wrapper positional arity differs from its declared signature')
    locals={n.id for n in ast.walk(definition) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    if locals & {packet,keywords}:
        raise ValueError('pure wrapper cannot rebind its call packets')
    names=[arg.arg for arg in positional]
    occupied=({n.id for n in ast.walk(definition) if isinstance(n,ast.Name)}|set(names)
              |{arg.arg for arg in signature.kwonlyargs})
    for index in range(count-len(names)):
        name='_b2_wrapper_arg_'+str(index)
        while name in occupied:name+='_'
        occupied.add(name);names.append(name)
    values={**scope.builtins,**scope.globals,**scope.nonlocals}

    def container(node):
        if isinstance(node,ast.Tuple) and getattr(node,'_b2_packet',False):return len(node.elts)
        if isinstance(node,ast.Dict) and not node.keys:return 0

    class Bind(ast.NodeTransformer):
        def visit_Name(self,node):
            if packet and node.id==packet:
                result=ast.Tuple(elts=[ast.Name(id=name,ctx=ast.Load())
                    for name in names[len(positional):]],ctx=ast.Load())
                result._b2_packet=True
                return ast.copy_location(result,node)
            if keywords and node.id==keywords:return ast.copy_location(ast.Dict(keys=[],values=[]),node)
            return node

        def visit_Subscript(self,node):
            node=self.generic_visit(node)
            if not isinstance(node.value,ast.Tuple) or not getattr(node.value,'_b2_packet',False):return node
            try:index=ast.literal_eval(node.slice)
            except (ValueError,TypeError):
                if not isinstance(node.slice,ast.Slice):
                    raise ValueError('pure wrapper packet indices must be static') from None
                try:index=slice(*(None if value is None else ast.literal_eval(value)
                    for value in (node.slice.lower,node.slice.upper,node.slice.step)))
                except (ValueError,TypeError):raise ValueError('pure wrapper packet slices must be static') from None
            if type(index) not in (int,bool,slice):raise ValueError('invalid pure wrapper packet index')
            try:result=node.value.elts[index]
            except (IndexError,ValueError,TypeError):raise ValueError('invalid pure wrapper packet index') from None
            if isinstance(index,slice):
                result=ast.Tuple(elts=result,ctx=ast.Load());result._b2_packet=getattr(node.value,'_b2_packet',False)
            return ast.copy_location(result,node)

        def visit_Call(self,node):
            # Only the genuine builtin len is folded; no user callable runs.
            original=node.func
            node=self.generic_visit(node)
            if (isinstance(original,ast.Name) and original.id not in locals|set(names)
                    and values.get(original.id) is builtins.len and len(node.args)==1 and not node.keywords
                    and container(node.args[0]) is not None):
                return ast.copy_location(ast.Constant(container(node.args[0])),node)
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Dict)
                    and not node.func.value.keys and node.func.attr=='get'
                    and 1<=len(node.args)<=2 and not node.keywords):
                if not isinstance(node.args[0],ast.Constant):raise ValueError('pure wrapper keyword key must be static')
                return ast.copy_location(node.args[1] if len(node.args)==2 else ast.Constant(None),node)
            args=[]
            for arg in node.args:
                if isinstance(arg,ast.Starred):
                    if not isinstance(arg.value,ast.Tuple):raise ValueError('pure wrapper needs a fixed positional packet')
                    args.extend(arg.value.elts)
                else:args.append(arg)
            node.args=args
            node.keywords=[keyword for keyword in node.keywords
                if not (keyword.arg is None and isinstance(keyword.value,ast.Dict) and not keyword.value.keys)]
            return node

        def visit_If(self,node):
            node=self.generic_visit(node)
            if container(node.test) is not None:node.test=ast.Constant(bool(container(node.test)))
            return node

        def visit_While(self,node):return self.visit_If(node)

        def visit_UnaryOp(self,node):
            node=self.generic_visit(node)
            if isinstance(node.op,ast.Not) and container(node.operand) is not None:
                return ast.copy_location(ast.Constant(not container(node.operand)),node)
            return node

    body=[Bind().visit(copy.deepcopy(statement)) for statement in definition.body]
    return body,tuple(names)


def _verify_loaded_body(definition, function):
    """Reject stale source rather than translate a different loaded function.

    Compile only, never execute. Reconstruct the lexical cells of nested
    functions so code comparison also works for captured constants. Line/file
    metadata is intentionally excluded; executable instructions are retained.
    """
    code=function.__code__;body=copy.deepcopy(definition);body.decorator_list=[]
    if code.co_flags & inspect.CO_NESTED:
        assignments=[ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],value=ast.Constant(None))
                     for name in code.co_freevars]
        wrapper=ast.FunctionDef(name='_b2_source_scope',args=ast.arguments(
            posonlyargs=[],args=[],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),
            body=assignments+[body],decorator_list=[],type_params=[])
        tree=ast.Module(body=[wrapper],type_ignores=[])
    else:tree=ast.Module(body=[body],type_ignores=[])
    flags=0
    for name in __future__.all_feature_names:flags|=getattr(__future__,name).compiler_flag
    try:compiled=compile(ast.fix_missing_locations(tree),code.co_filename,'exec',
                         flags=code.co_flags&flags,dont_inherit=True)
    except (SyntaxError,ValueError,TypeError) as error:
        raise ValueError('pure function source cannot reproduce loaded code') from error
    def find(parent):
        for child in parent.co_consts:
            if isinstance(child,CodeType):
                if child.co_name==definition.name:return child
                found=find(child)
                if found is not None:return found
    def fingerprint(value):
        if isinstance(value,CodeType):
            # CPython may encode a module attribute call as LOAD_ATTR followed
            # by PUSH_NULL or as LOAD_ATTR with its method-call bit. Both have
            # the same semantics for the only admitted attribute owners:
            # numpy/math modules. Compare instructions by resolved operands.
            retained=[i for i in dis.get_instructions(value) if i.opname!='PUSH_NULL']
            offsets=[i.offset for i in retained]
            jumps=getattr(dis,'hasjump',dis.hasjabs+dis.hasjrel)
            # Equivalent call setup can shift byte offsets inside a loop.
            # Preserve its control-flow destination by logical instruction
            # position rather than comparing the incidental encoded offset.
            instructions=tuple((i.opname,('target',bisect.bisect_left(offsets,i.argval))
                                if i.opcode in jumps else fingerprint(i.argval)) for i in retained)
            return (instructions,tuple(fingerprint(c) for c in value.co_consts),value.co_names,
                    value.co_varnames,value.co_freevars,value.co_cellvars,value.co_argcount,
                    value.co_posonlyargcount,value.co_kwonlyargcount,value.co_flags)
        return value
    candidate=find(compiled)
    if candidate is None or fingerprint(candidate)!=fingerprint(code):
        raise ValueError('pure function source differs from loaded Python code')


def _expand_static_loops(body, scope, arguments):
    """Unroll bounded builtin ranges without evaluating callback code.

    Bounds may use immutable integer captures and already known scalar locals,
    including an enclosing loop index. Unknown locals shadow captures. Every
    iteration binds its index in source order; the normal alias/SSA admission
    still checks every expanded assignment.
    """
    unknown=object();values={**scope.builtins,**scope.globals,**scope.nonlocals}
    # Python determines local bindings for the entire function before its
    # first instruction. A later assignment must already shadow a capture at
    # an earlier loop, including the index of an empty range.
    local_names={node.id for statement in body for node in ast.walk(statement)
                 if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}
    environment={name:unknown for name in set(arguments)|local_names};expanded=[]

    def bounded(value):
        if not -2**31<=value<2**31:raise ValueError('pure function static loop bound requires int32')
        return value

    def integer(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,bool):return bounded(int(node.value))
        if isinstance(node,ast.Name):
            value=environment.get(node.id,values.get(node.id,unknown))
            if type(value) in _NUMPY_SCALAR_TYPES and np.asarray(value).dtype.kind in 'iub':value=value.item()
            if type(value) in (int,bool):return bounded(int(value))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=integer(node.operand);return bounded(value if isinstance(node.op,ast.UAdd) else -value)
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.FloorDiv,ast.Mod)):
            left=integer(node.left);right=integer(node.right)
            if isinstance(node.op,ast.Add):return bounded(left+right)
            if isinstance(node.op,ast.Sub):return bounded(left-right)
            if isinstance(node.op,ast.Mult):return bounded(left*right)
            if not right:raise ValueError('static loop integer bound divides by zero')
            return bounded(left//right if isinstance(node.op,ast.FloorDiv) else left%right)
        raise ValueError('pure function loop bounds require statically known integers')

    def scalar(node):
        """Inspect exact immutable scalars, never callbacks or array truth hooks."""
        if isinstance(node,ast.Constant):value=node.value
        elif isinstance(node,ast.Name):value=environment.get(node.id,values.get(node.id,unknown))
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=scalar(node.operand);value=value if isinstance(node.op,ast.UAdd) else -value
        elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod)):
            left=scalar(node.left);right=scalar(node.right)
            if isinstance(node.op,ast.Add):value=left+right
            elif isinstance(node.op,ast.Sub):value=left-right
            elif isinstance(node.op,ast.Mult):value=left*right
            else:
                if not right:raise ValueError('static callback scalar divides by zero')
                if isinstance(node.op,ast.Div):value=left/right
                elif isinstance(node.op,ast.FloorDiv):value=left//right
                else:value=left%right
        else:raise ValueError('static callback conditions require immutable real scalars')
        if type(value) in _NUMPY_SCALAR_TYPES:
            # Keep NumPy real scalar precision during static evaluation.
            # Converting float32(.1)/float32(.3) to Python doubles changes
            # repeated-addition loop counts and can change physical spikes.
            if value.dtype.kind == 'f':
                if math.isfinite(value):return value
                raise ValueError('static callback conditions require finite immutable real scalars')
            value=value.item()
        if type(value) in (int,bool):return bounded(int(value))
        if type(value) is float and math.isfinite(value):return value
        raise ValueError('static callback conditions require finite immutable real scalars')

    def known(node):
        try:return scalar(node)
        except ValueError:return unknown

    def truth(node):
        # Only exact immutable real scalars are inspected. Python's lazy predicates
        # can skip an unknown array/call without evaluating it, just as the
        # original callback does; a reached unknown predicate is refused.
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):
            return not truth(node.operand)
        if isinstance(node,ast.BoolOp):
            if isinstance(node.op,ast.And):
                return all(truth(value) for value in node.values)
            if isinstance(node.op,ast.Or):
                return any(truth(value) for value in node.values)
        if isinstance(node,ast.Compare):
            left=scalar(node.left)
            for operator,expression in zip(node.ops,node.comparators):
                right=scalar(expression)
                if isinstance(operator,ast.Lt):accepted=left<right
                elif isinstance(operator,ast.LtE):accepted=left<=right
                elif isinstance(operator,ast.Gt):accepted=left>right
                elif isinstance(operator,ast.GtE):accepted=left>=right
                elif isinstance(operator,ast.Eq):accepted=left==right
                elif isinstance(operator,ast.NotEq):accepted=left!=right
                else:raise ValueError('static callback conditions require scalar comparisons')
                if not accepted:return False
                left=right
            return True
        return bool(scalar(node))

    def emit(statement):
        if len(expanded)>=65:raise ValueError('pure function expanded loop exceeds 64 local assignments')
        expanded.append(copy.deepcopy(statement))
        if isinstance(statement,ast.Assign):
            for target in statement.targets:
                if isinstance(target,ast.Name):environment[target.id]=known(statement.value)
        elif isinstance(statement,ast.AugAssign) and isinstance(statement.target,ast.Name):
            environment[statement.target.id]=known(ast.BinOp(left=ast.Name(id=statement.target.id,ctx=ast.Load()),
                op=statement.op,right=statement.value))

    def walk(statements,depth=0):
        if depth>8:raise ValueError('pure function loop nesting exceeds eight levels')
        for statement in statements:
            if isinstance(statement,ast.Return):emit(statement);return 'return'
            if isinstance(statement,ast.Break):return 'break'
            if isinstance(statement,ast.Continue):return 'continue'
            if isinstance(statement,ast.If):
                flow=walk(statement.body if truth(statement.test) else statement.orelse,depth+1)
                if flow:return flow
                continue
            if isinstance(statement,ast.While):
                interrupted=False;iterations=0
                while truth(statement.test):
                    if iterations>=64:raise ValueError('pure function static while exceeds 64 iterations')
                    iterations+=1
                    flow=walk(statement.body,depth+1)
                    if flow=='return':return flow
                    if flow=='break':interrupted=True;break
                if not interrupted:
                    flow=walk(statement.orelse,depth+1)
                    if flow:return flow
                continue
            if not isinstance(statement,ast.For):emit(statement);continue
            call=statement.iter
            if (not isinstance(statement.target,ast.Name) or not isinstance(call,ast.Call)
                or not isinstance(call.func,ast.Name) or call.keywords or not 1<=len(call.args)<=3
                or environment.get(call.func.id,values.get(call.func.id,unknown)) is not builtins.range):
                raise ValueError('pure function loops require an unmodified builtin range and a local index')
            bounds=[integer(arg) for arg in call.args]
            try:iterations=range(*bounds);length=len(iterations)
            except (ValueError,OverflowError):raise ValueError('invalid or oversized pure function loop range') from None
            if length>64:raise ValueError('pure function expanded loop exceeds 64 local assignments')
            interrupted=False
            for index in iterations:
                if not -2**31<=index<2**31:raise ValueError('pure function loop index requires int32')
                assignment=ast.Assign(targets=[copy.deepcopy(statement.target)],value=ast.Constant(index))
                emit(assignment)
                flow=walk(statement.body,depth+1)
                if flow=='return':return flow
                if flow=='break':interrupted=True;break
                # continue skips this iteration's remaining body, then the
                # builtin range binds its next index normally.
            if not interrupted:
                # A control statement in an inner loop's else suite belongs
                # to the enclosing loop; propagate it instead of consuming it
                # as an exit from the already completed inner loop.
                flow=walk(statement.orelse,depth+1)
                if flow:return flow
    if walk(body) in ('break','continue'):raise ValueError('pure function loop control needs an enclosing loop')
    return expanded


def _automatic_numpy_function(function):
    """Inspect a canonical default binding without calling implementation hooks."""
    container=function.implementations
    if type(container) is not FunctionImplementationContainer or container._function is not function:
        raise ValueError('pure function requires a standard implementation container')
    entries=container._implementations
    if type(entries) is not dict or len(entries)!=1:
        raise ValueError('pure function requires stateless non-vectorised Python semantics without target overrides')
    target,impl=next(iter(entries.items()))
    if type(target) is not str or target!='numpy' or type(impl) is not FunctionImplementation:
        raise ValueError('pure function requires stateless non-vectorised Python semantics without target overrides')
    binding=impl._automatic_numpy_binding
    if (type(binding) is not tuple or len(binding)!=7 or binding[0] is not function.pyfunc
        or type(binding[1]) is not FunctionType or type(impl._code) is not FunctionType
        or impl._code is not binding[2] or impl._code.__code__ is not binding[3]
        or type(binding[4]) is not bool or type(binding[5]) is not tuple
        or type(binding[6]) is not tuple
        or impl.dynamic is not False or impl._namespace is not None
        or type(impl.dependencies) is not dict or len(impl.dependencies)
        or impl.availability_check is not None
        or type(impl.compiler_kwds) is not dict or len(impl.compiler_kwds) or impl.name is not None):
        raise ValueError('pure function requires stateless non-vectorised Python semantics without target overrides')
    try:closure=tuple(cell.cell_contents for cell in (impl._code.__closure__ or ()))
    except ValueError:raise ValueError('pure function automatic NumPy closure is empty') from None
    if len(closure)!=len(binding[5]) or any(a is not b for a,b in zip(closure,binding[5])):
        raise ValueError('pure function automatic NumPy binding was modified')
    if binding[4]:
        # Brian's unitless implementation has its own copied global namespace.
        # Use that actual namespace, including its converted scalar quantities,
        # instead of rereading globals from the original callback.
        if any(type(entry) is not tuple or len(entry)!=2 or type(entry[0]) is not str
               or type(entry[1]) is not np.ndarray for entry in binding[6]):
            raise ValueError('pure function automatic NumPy namespace was modified')
        captures={name for name,value in binding[6]
                  if impl._code.__globals__.get(name) is value}
        return impl._code,captures
    return binding[1],set()


def _numpy_array_contract(function):
    """Snapshot Brian's authentic wrapper's zero-dimensional conversions."""
    if type(function) is not Function:return (),False
    from brian2.core.preferences import prefs
    if (type(function.implementations) is not FunctionImplementationContainer
            or type(function.implementations._implementations) is not dict):
        raise ValueError('NumPy callback requires ordinary implementation metadata')
    if function.implementations._implementations:
        _automatic_numpy_function(function)
        discard=next(iter(function.implementations._implementations.values()))._automatic_numpy_binding[4]
    else:discard=prefs.codegen.runtime.numpy.discard_units
    if discard:return (),False
    units=function._arg_units
    if type(units) not in (list,tuple) or len(units)>16:raise ValueError('invalid NumPy callback units')
    flags=[]
    for unit in units:
        if unit is None or unit is bool or type(unit) is str:flags.append(False)
        elif type(unit) in (Unit,Quantity,Dimension,int,float) or type(unit) in _NUMPY_SCALAR_TYPES:
            flags.append(get_dimensions(unit) is DIMENSIONLESS)
        else:raise ValueError('NumPy callback units need ordinary dimensions')
    return tuple(flags),True


def lower_pure_function(function, _stack=()):
    """Snapshot a stateless callback with bounded sequential local assignments.

    Registered target implementations can change its semantics, so they require
    a separate contract and are refused here. Captured finite scalar constants
    are copied; state, mutation, randomness and opaque callable objects are not
    admitted. Native SSA performs both forward execution and differentiation.
    """
    scalar_array_captures=set();captured_arrays=set();declared_arity=None
    scalarize_arguments,array_return=_numpy_array_contract(function)
    if type(function) is Function:
        if type(function._arg_units) in (list,tuple):declared_arity=len(function._arg_units)
        if (type(function.stateless) not in (bool,np.bool_) or not bool(function.stateless)
            or type(function.auto_vectorise) not in (bool,np.bool_) or bool(function.auto_vectorise)):
            raise ValueError('pure function requires stateless non-vectorised Python semantics without target overrides')
        container=function.implementations
        if type(container) is not FunctionImplementationContainer or type(container._implementations) is not dict:
            raise ValueError('pure function requires a standard implementation container')
        if len(container._implementations):
            function,scalar_array_captures=_automatic_numpy_function(function)
        else:function = function.pyfunc
    elif issubclass(type(function),Function):
        raise ValueError('pure function requires a standard Brian Function object')
    # Inspect only ordinary function dictionaries. inspect.unwrap on an opaque
    # callable can invoke a user-defined __wrapped__ property before refusal.
    wrappers=set();chain=[];cursor=function
    while True:
        if type(cursor) is not FunctionType or id(cursor) in wrappers or len(wrappers)>=8:
            raise ValueError('opaque, cyclic or excessively wrapped pure function')
        wrappers.add(id(cursor));chain.append(cursor)
        wrapped=vars(cursor).get('__wrapped__')
        if wrapped is None:break
        cursor=wrapped
    # Unit checks leave the numeric result unchanged. Other decorators are
    # executable numeric code: __wrapped__ is metadata, not a substitute body.
    while function.__code__ is _UNIT_CHECK_WRAPPER_CODE:
        wrapped=vars(function).get('__wrapped__')
        try:original=inspect.getclosurevars(function).nonlocals.get('f')
        except ValueError:raise ValueError('pure unit-check wrapper has an empty closure') from None
        if original is not wrapped or vars(function).get('_orig_func') is not wrapped:
            raise ValueError('pure unit-check wrapper binding was modified')
        function=wrapped
    if id(function) in _stack or len(_stack)>=8:
        raise ValueError('recursive, opaque or excessively nested pure function')
    try:
        source = textwrap.dedent(inspect.getsource(function.__code__))
        if len(source)>8192:raise ValueError('pure function source exceeds 8192 characters')
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, RecursionError) as error:
        raise ValueError('pure function needs readable bounded source') from error
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('pure function requires one Python function definition')
    definition=tree.body[0];signature=definition.args
    _verify_loaded_body(definition,function)
    keyword_bindings=_keyword_default_bindings(function,signature)
    scope=inspect.getclosurevars(function)
    if signature.vararg or signature.kwarg:
        if declared_arity is None:
            target=chain[-1]
            if (len(chain)<2 or target.__code__.co_flags & (inspect.CO_VARARGS|inspect.CO_VARKEYWORDS)
                    or target.__defaults__):
                raise ValueError('pure wrapper requires a known fixed positional arity')
            declared_arity=target.__code__.co_argcount
        body,arguments=_bind_wrapper_packets(definition,scope,declared_arity)
    elif (signature.defaults or not 0<=len(signature.posonlyargs+signature.args)<=16):
        raise ValueError('pure function requires 0..16 positional arguments without defaults')
    else:body=list(definition.body);arguments=tuple(arg.arg for arg in signature.posonlyargs+signature.args)
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
        body=body[1:]
    body=keyword_bindings+body
    if (any(isinstance(node,(ast.For,ast.While,ast.If)) for statement in body for node in ast.walk(statement))
        or any(isinstance(node,ast.Return) for statement in body[:-1] for node in ast.walk(statement))):
        body=_expand_static_loops(body,scope,arguments)
    if not body or not isinstance(body[-1],ast.Return) or body[-1].value is None:
        raise ValueError('pure function requires local assignments followed by one return')
    if len(body)>65:raise ValueError('pure function requires at most 64 local assignments')
    statements=[];locals=set();augmented=[]
    # Brian NumPy callbacks receive arrays. An augmented write to an argument
    # or an aliased array mutates visible storage, rather than rebinding a
    # scalar. Admit only private, unaliased intermediate storage here.
    aliases={name:('borrowed',name) for name in arguments}
    values={**scope.globals,**scope.nonlocals}
    bindings=set(arguments)|{node.id for node in ast.walk(definition)
                            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}

    def resolved(node):
        if isinstance(node,ast.Name) and node.id not in bindings:
            return values.get(node.id,scope.builtins.get(node.id))
        if (isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name)
                and node.value.id not in bindings and values.get(node.value.id) is np):
            if node.attr == 'array':return np.array
            if node.attr == 'float64':return np.float64

    def private_array_copy(node):
        # Explicit floating, owned storage has identical alias semantics for
        # scalar and array arguments. Plain arithmetic intermediates do not:
        # NumPy arrays mutate in place, Python scalars rebind independently.
        if (not isinstance(node,ast.Call) or resolved(node.func) is not _PRIVATE_ARRAY_CONSTRUCTOR
                or len(node.args)!=1):return False
        options={keyword.arg:keyword.value for keyword in node.keywords}
        if len(options)!=len(node.keywords) or not set(options)<= {'dtype','copy'} or 'dtype' not in options:
            return False
        if not any(resolved(options['dtype']) is dtype for dtype in _PRIVATE_ARRAY_DTYPES):return False
        if 'copy' in options and not (isinstance(options['copy'],ast.Constant) and options['copy'].value is True):
            return False
        return True

    def storage(expression,serial):
        if isinstance(expression,ast.Name) and expression.id in aliases:
            return aliases[expression.id]
        if private_array_copy(expression):return ('array',serial)
        if isinstance(expression,ast.Call):return ('borrowed',serial)
        return ('private',serial)
    def expose_calls(expression):
        # A pure helper may return its argument unchanged. Any local passed
        # through a call can therefore have an alias retained by the result.
        for call in (n for n in ast.walk(expression) if isinstance(n,ast.Call)):
            if private_array_copy(call):continue
            exposed={aliases[n.id] for arg in call.args for n in ast.walk(arg)
                     if isinstance(n,ast.Name) and n.id in aliases}
            for name,identity in list(aliases.items()):
                if identity in exposed:aliases[name]=('borrowed',identity)
    for statement in body[:-1]:
        if (isinstance(statement,ast.Assign) and len(statement.targets)==1
                and isinstance(statement.targets[0],ast.Name)):
            name=statement.targets[0].id;expression=statement.value
            expose_calls(expression)
            aliases[name]=storage(expression,len(statements))
        elif isinstance(statement,ast.AugAssign) and isinstance(statement.target,ast.Name):
            name=statement.target.id
            expose_calls(statement.value)
            identity=aliases.get(name)
            if (identity is None or identity[0]=='borrowed'
                    or identity[0]!='array' and list(aliases.values()).count(identity)>1):
                raise ValueError('pure function augmented assignment cannot mutate arguments or aliased arrays')
            augmented.append(len(statements))
            expression=ast.BinOp(left=ast.Name(id=name,ctx=ast.Load()),op=statement.op,right=statement.value)
        else:raise ValueError('pure function requires scalar local assignments without mutation or control flow')
        statements.append((name,expression));locals.add(name)
        if isinstance(statement,ast.AugAssign) and identity[0]=='array':
            # The old SSA values stay available to earlier derived arrays.
            # Refresh only live aliases of this particular owned allocation.
            for alias,allocation in aliases.items():
                if alias!=name and allocation==identity:
                    statements.append((alias,ast.Name(id=name,ctx=ast.Load())))
                    locals.add(alias)
    closure={}

    def canonical(value):
        for candidate,name in _FUNCTION_IDENTITIES:
            if value is candidate:return name
        return None

    def capture(name):
        if name in closure:return
        if name in _MATH:
            raise ValueError('pure function closure shadows standard math: '+name)
        if name not in values:raise ValueError('unknown pure function closure name: '+name)
        value=values[name]
        if name in scalar_array_captures and type(value) is np.ndarray:
            if value.shape or value.dtype.kind not in 'fiub':
                raise ValueError('pure function automatic unit closure must be a real scalar')
            captured_arrays.add(name);value=value.item()
        if type(value) in (Quantity,Unit):
            array=np.asarray(value)
            if array.shape or array.dtype.kind not in 'fiub':
                raise ValueError('pure function unit closure must be a real scalar')
            # Brian values and native state/parameters use SI. Signature and
            # expression unit checks are performed by the Brian lowerer.
            captured_arrays.add(name);value=array.item()
        if type(value) in (bool,int,float) or type(value) in _NUMPY_SCALAR_TYPES:
            value=value.item() if type(value) in _NUMPY_SCALAR_TYPES else value
            if not math.isfinite(value):raise ValueError('pure function closure must be finite')
            closure[name]=value
        elif type(value) is FunctionType or issubclass(type(value),Function):
            closure[name]=lower_pure_function(value,_stack+(id(function),))
        else:raise ValueError('pure function closure must be a scalar constant or pure function')

    class Normalize(ast.NodeTransformer):
        def visit_Call(self,node):
            if private_array_copy(node):
                # Identity VJP plus explicit float coercion; the argument is
                # evaluated once. Ownership exists only in local SSA aliases.
                return ast.copy_location(ast.Call(func=ast.Name(id='_b2_owned_array',ctx=ast.Load()),
                    args=[self.visit(node.args[0])],keywords=[]),node)
            if node.keywords:raise ValueError('pure function calls require positional arguments')
            if isinstance(node.func,ast.Name):
                name=node.func.id
                if name in set(arguments)|locals:raise ValueError('pure function locals cannot be callable')
                value=values.get(name,scope.builtins.get(name))
            elif isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
                if node.func.value.id in set(arguments)|locals:
                    raise ValueError('pure function local attributes are not scalar operations')
                owner=values.get(node.func.value.id)
                if owner is not np and owner is not math:
                    raise ValueError('pure function attributes require standard numpy/math functions')
                value=getattr(owner,node.func.attr,None);name=None
            else:raise ValueError('pure function call requires a named scalar function')
            builtin=canonical(value)
            if builtin is not None:target=builtin
            elif name is not None:capture(name);target=name
            else:raise ValueError('unsupported pure function math attribute')
            return ast.copy_location(ast.Call(func=ast.Name(id=target,ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],keywords=[]),node)

        def visit_Name(self,node):
            if node.id not in set(arguments)|locals:capture(node.id)
            return node

    normalize=Normalize()
    # Python truth tests on Brian's array arguments are not elementwise. Keep
    # explicit NumPy selection/logical calls distinct from scalar descriptor
    # control flow, so lowering cannot hide an ambiguous-array exception.
    trees=[expression for _,expression in statements]+[body[-1].value]
    if any(isinstance(n,(ast.IfExp,ast.BoolOp)) or isinstance(n,ast.Compare) and len(n.ops)>1
           for tree in trees for n in ast.walk(tree)):
        raise ValueError('array callback conditions require explicit numpy where/logical operations')
    assignments=tuple((name,ast.unparse(normalize.visit(expression))) for name,expression in statements)
    expression=normalize.visit(body[-1].value)
    result=PureFunction(arguments,ast.unparse(expression),tuple(sorted(closure.items())),assignments,tuple(augmented),scalarize_arguments,array_return,tuple(sorted(captured_arrays)))
    # Validate the same lexical/body bounds as native compilation now, before
    # the lowerer allocates a trainer or changes any source object.
    from .training_equations import compile_training_equation
    compile_training_equation('callback('+','.join('0.' for _ in arguments)+')',parameters={'callback':result})
    return result
