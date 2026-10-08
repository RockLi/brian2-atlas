"""Bounded NumPy callback effects compiled into native state transformations."""
import ast
import copy
import inspect
import textwrap
from dataclasses import dataclass
from types import FunctionType
import math
import numpy as np
from brian2.core.functions import Function, FunctionImplementationContainer
from .training_functions import (_verify_loaded_body, _automatic_numpy_function,
                                 _UNIT_CHECK_WRAPPER_CODE, _numpy_array_contract)


@dataclass(frozen=True)
class StateEffectFunction:
    arguments: tuple
    body: str
    constants: tuple = ()
    scalarize_arguments: tuple = ()
    array_return: bool = False
    captured_arrays: tuple = ()
    capture_bindings: tuple = ()
    parameter_captures: tuple = ()


def _bounded_tree(tree):
    for count, _ in enumerate(ast.walk(tree), 1):
        if count > 4096:
            raise ValueError('state effect expression exceeds 4096 AST nodes')
    return tree


def lower_state_effect_function(function):
    """Inspect a deterministic local body; never invoke it or metadata hooks."""
    scalarize_arguments,array_return=_numpy_array_contract(function)
    if type(function) is Function:
        if (type(function.stateless) not in (bool,np.bool_) or
                type(function.auto_vectorise) not in (bool,np.bool_) or bool(function.auto_vectorise)):
            raise ValueError('state effects require ordinary non-vectorised function metadata')
        container=function.implementations
        if type(container) is not FunctionImplementationContainer or type(container._implementations) is not dict:
            raise ValueError('state effects require ordinary implementation metadata')
        if len(container._implementations):
            function,_=_automatic_numpy_function(function)
        else:function=function.pyfunc
    if type(function) is not FunctionType:raise ValueError('state effect callback needs ordinary readable Python source')
    seen=set()
    while function.__code__ is _UNIT_CHECK_WRAPPER_CODE:
        if id(function) in seen or len(seen)>=8:raise ValueError('cyclic state effect unit wrapper')
        seen.add(id(function));wrapped=vars(function).get('__wrapped__')
        if (type(wrapped) is not FunctionType or vars(function).get('_orig_func') is not wrapped
                or inspect.getclosurevars(function).nonlocals.get('f') is not wrapped):
            raise ValueError('state effect unit wrapper binding was modified')
        function=wrapped
    source=textwrap.dedent(inspect.getsource(function.__code__))
    if len(source)>8192:raise ValueError('state effect source exceeds budget')
    tree=ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('state effect requires one definition')
    definition=tree.body[0];_verify_loaded_body(definition,function);signature=definition.args
    if signature.vararg or signature.kwarg or signature.kwonlyargs or signature.defaults:
        raise ValueError('state effects currently require fixed positional signatures')
    arguments=tuple(a.arg for a in signature.posonlyargs+signature.args)
    if not 0<=len(arguments)<=16 or not 1<=len(definition.body)<=65:raise ValueError('state effect body exceeds argument/statement budget')
    local=set(arguments)|{n.id for n in ast.walk(definition) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    scope=inspect.getclosurevars(function);values={**scope.globals,**scope.nonlocals};constants={};captured_arrays={}
    for name in {n.id for statement in definition.body for n in ast.walk(statement) if isinstance(n,ast.Name)}-local:
        value=values.get(name)
        if type(value) is np.ndarray:
            if (value.ndim!=1 or not 0<value.size<=1_000_000 or
                    not (value.dtype.kind=='f' or value.dtype in (np.dtype('int32'),np.dtype('bool'))) or
                    value.strides[0]%value.dtype.itemsize or
                    not np.all(np.isfinite(value))):
                raise ValueError('state effect captures require finite aligned float/int32/Boolean vectors')
            captured_arrays[name]=value
            continue
        if (type(value) not in (bool,int,float) or type(value) is int and not -2**31<=value<2**31
                or type(value) is float and not math.isfinite(value)):
            raise ValueError('state effect captures must be finite immutable builtin scalars')
        constants[name]=value
    if not any(isinstance(n,ast.AugAssign) for n in ast.walk(definition)):
        raise ValueError('state effect body does not contain an explicit local augmented write')
    return StateEffectFunction(arguments,'\n'.join(ast.unparse(n) for n in definition.body),tuple(sorted(constants.items())),scalarize_arguments,array_return,tuple(sorted(captured_arrays.items())))


def bind_state_effect_captures(function, bindings, *, readonly=()):
    """Bind authenticated capture identities to persistent symbolic array names.

    This does not snapshot arrays as constants or execute callback code. The
    caller allocates canonical state and owns shape/alias validation.
    """
    from dataclasses import replace
    if type(function) is not StateEffectFunction or type(bindings) is not dict:
        raise ValueError('capture binding requires an authenticated effect descriptor and names')
    captures=dict(function.captured_arrays)
    if set(bindings)!=set(captures) or any(type(value) is not str or not value.isidentifier() for value in bindings.values()):
        raise ValueError('every mutable capture requires a persistent array binding')
    if type(readonly) is not tuple or not set(readonly)<=set(captures):
        raise ValueError('invalid readonly capture binding')
    readonly=tuple(sorted(set(readonly)|{name for name,array in captures.items() if not array.flags.writeable}))
    return replace(function,capture_bindings=tuple(sorted(bindings.items())),parameter_captures=readonly)


def capture_array_key(array):
    """Exact NumPy storage identity, including Brian dynamic-array views."""
    if type(array) is not np.ndarray:raise ValueError('capture storage requires ordinary NumPy arrays')
    return (array.ctypes.data,array.dtype.str,array.shape,array.strides)


def same_capture_storage(left,right):
    return type(left) is np.ndarray and type(right) is np.ndarray and capture_array_key(left)==capture_array_key(right)


def capture_view_indices(base,view):
    """Locate aligned, possibly reversed/strided cells in a live buffer."""
    if (type(base) is not np.ndarray or type(view) is not np.ndarray or
            base.ndim!=1 or view.ndim!=1 or base.dtype!=view.dtype or
            not base.flags.c_contiguous):return None
    offset=view.ctypes.data-base.ctypes.data;size=base.dtype.itemsize
    step=view.strides[0]
    if offset%size or step%size:return None
    indices=[offset//size+j*(step//size) for j in range(view.size)]
    if any(index<0 or index>=base.size for index in indices):return None
    return indices


def mutated_state_effect_captures(function):
    """Trace canonical capture writes without invoking callback Python."""
    if type(function) is not StateEffectFunction:raise ValueError('expected authenticated effect descriptor')
    states={name:k for k,name in enumerate(function.arguments)}
    types={slot:'float' for slot in states.values()};bindings={}
    for name,array in function.captured_arrays:
        source='__capture_probe_'+str(len(states));slot=len(states);states[source]=slot
        types[slot]='integer' if array.dtype==np.dtype('int32') else 'boolean' if array.dtype==np.dtype('bool') else 'float'
        bindings[name]=source
    bound=bind_state_effect_captures(function,bindings)
    protected={bindings[name] for name,array in function.captured_arrays if not array.flags.writeable}
    engine=Effects(states,{},array_states=set(states),writable_states=set(states)-protected,state_types=types,eager_limit=128)
    engine.call(bound,[engine.environment[name] for name in function.arguments])
    return {name for name,source in bindings.items() if states[source] in engine.effect_writes}


def mutated_state_effect_caller_captures(statements,functions,variables,*,scalar=False,reload=False):
    """Trace returned capture references through generated caller statements.

    Event arguments are private copies, while closures keep physical identity.
    Vectorised NumPy reloads persistent operands at each statement; temporaries
    retain their references. No callback or numerical model code is executed.
    """
    from brian2.core.variables import ArrayVariable
    tree=_bounded_tree(ast.parse(statements));needed={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    states={};types={};arrays=set();persistent=set();bound=dict(functions)
    for name,var in variables.items():
        if name not in needed or name in functions or not hasattr(var,'dtype'):continue
        slot=len(states);states[name]=slot;dtype=np.dtype(var.dtype)
        types[slot]='integer' if dtype==np.dtype('int32') else 'boolean' if dtype.kind=='b' else 'float'
        if isinstance(var,ArrayVariable) and not var.scalar and not scalar:arrays.add(name);persistent.add(name)
    captures={};protected=set()
    for name,function in functions.items():
        if type(function) is not StateEffectFunction:continue
        bindings={}
        for capture,array in function.captured_arrays:
            key=(capture_array_key(array),bool(array.flags.writeable))
            if key not in captures:
                source='__caller_capture_'+str(len(captures));slot=len(states)
                if source in needed:raise ValueError('reserved caller capture name')
                states[source]=slot;arrays.add(source)
                types[slot]='integer' if array.dtype==np.dtype('int32') else 'boolean' if array.dtype==np.dtype('bool') else 'float'
                captures[key]=source
                if not array.flags.writeable:protected.add(source)
            bindings[capture]=captures[key]
        bound[name]=bind_state_effect_captures(function,bindings)
    engine=Effects(states,bound,array_states=arrays,writable_states=arrays-protected,state_types=types,eager_limit=128)
    for name in persistent:engine.environment[name].origin=None
    for statement in tree.body:
        if reload:
            used={n.id for n in ast.walk(statement) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
            if isinstance(statement,ast.AugAssign) and isinstance(statement.target,ast.Name):used.add(statement.target.id)
            for name in used&persistent:
                engine.environment[name]=ArrayCell(ast.Name(id=name,ctx=ast.Load()),dtype=types[states[name]])
        engine.statement(statement,engine.environment,physical=True)
    return {key[0] for key,source in captures.items() if states[source] in engine.effect_writes}


@dataclass
class ArrayCell:
    value: ast.expr
    writable: bool = True
    origin: int | None = None
    dtype: str = 'float'
    zero_dim: bool = False
    source: 'ArrayCell | None' = None
    length: int | None = None
    columns: tuple | None = None
    column: int = 0
    selection_shape: bool = False


class Effects:
    def __init__(self, states, functions, scalar_parameters=(), *, array_states=None, writable_states=None,
                 array_parameters=(), temporary_parameters=(), eager_limit=64, array_callables=(),state_types=None,parameter_types=None):
        from .training_equations import (PureFunction,PoissonNoise,TimedInput,IntegerParameter,BooleanParameter,
                                         NeuronParameter,_MappedParameter,_DeferredParameter,StateSlot,_TimedState)
        self.callable_types=(PureFunction,PoissonNoise,TimedInput)
        array_states=set(states) if array_states is None else set(array_states)
        writable_states=set(states) if writable_states is None else set(writable_states)
        state_types={} if state_types is None else dict(state_types)
        self.input_dtypes={name:state_types.get(slot,'float') for name,slot in states.items()}
        self.input_dtypes.update({name:value.dtype if type(value) in (NeuronParameter,_MappedParameter,_DeferredParameter,StateSlot,_TimedState) else
                                 'integer' if type(value) is IntegerParameter or type(value) is int else
                                 'boolean' if type(value) is BooleanParameter or type(value) is bool else 'float'
                                 for name,value in dict(scalar_parameters).items()})
        self.input_dtypes.update({} if parameter_types is None else dict(parameter_types))
        self.roots={name:ArrayCell(ast.Name(id=name,ctx=ast.Load()),name in writable_states,slot,dtype=self.input_dtypes[name])
                    if name in array_states else ast.Name(id=name,ctx=ast.Load()) for slot,name in enumerate(states)}
        self.environment=dict(self.roots)
        self.environment.update({name:value if type(value) in self.callable_types else ast.Name(id=name,ctx=ast.Load())
                                 for name,value in dict(scalar_parameters).items()})
        forced=set(array_callables)
        for name in array_parameters:
            if type(self.environment.get(name)) in self.callable_types:forced.add(name)
            else:self.environment[name]=ArrayCell(ast.Name(id=name,ctx=ast.Load()),name in temporary_parameters,dtype=self.input_dtypes.get(name,'float'))
        self.functions=functions
        self.array_callables=forced
        self.pure_stack=[]
        self.eager_limit=eager_limit
        self.executed=[]
        self.executed_keys=set()
        self.effect_writes=set()
        self.indexed_reads=set()
        self.validate_empty_arrays=False
        self.record_scope=None;self.operation_count=0
        self.capture_vectors={}
        self.selected_vector_inputs=set()
        self.selected_vector_output=None
        self.selected_vector_keys=set()
        self.vector_record_guard=None
        self.whole_selection_keys=set()
        self.selected_call_presence=None
        self.selected_write_scope=False
        self.selected_operand_copies=set()

    def materialize(self,value):
        if isinstance(value,ArrayCell) and value.columns is not None:value=value.columns[value.column]
        while isinstance(value,ArrayCell) and value.source is not None:value=value.source
        return copy.deepcopy(_bounded_tree(value.value if isinstance(value,ArrayCell) else value))

    def element(self,value,column):
        if not isinstance(value,ArrayCell) or value.columns is None:return value
        part=value.columns[0 if len(value.columns)==1 else column]
        return ArrayCell(self.materialize(part),dtype=value.dtype,zero_dim=True,selection_shape=value.selection_shape)

    def vector_expression(self,node,values,environment):
        """Compose an elementwise expression for every actual capture cell."""
        self.broadcast_length(values)
        vectors=[v for v in values if isinstance(v,ArrayCell) and v.columns is not None]
        whole_vectors=[v for v in vectors if not v.selection_shape]
        fixed_vectors=[v for v in whole_vectors if len(v.columns)>1]
        count=max(len(v.columns) for v in (fixed_vectors or vectors));column=next(v.column for v in (fixed_vectors or vectors) if len(v.columns)==count)
        whole_selection=any(isinstance(value,ArrayCell) and value.selection_shape or any(
            ast.dump(item,include_attributes=False) in self.whole_selection_keys for item in ast.walk(self.materialize(value))) for value in values)
        expressions=[]
        for j in range(count):
            scope=dict(environment);names=[]
            for i,value in enumerate(values):
                name='_b2_whole_vector_operand_'+str(i);names.append(ast.Name(id=name,ctx=ast.Load()));scope[name]=self.element(value,j)
            if isinstance(node,ast.BinOp):part=ast.BinOp(left=names[0],op=copy.deepcopy(node.op),right=names[1])
            elif isinstance(node,ast.UnaryOp):part=ast.UnaryOp(op=copy.deepcopy(node.op),operand=names[0])
            elif isinstance(node,ast.Compare):part=ast.Compare(left=names[0],ops=copy.deepcopy(node.ops),comparators=[names[1]])
            else:part=ast.Call(func=copy.deepcopy(node.func),args=names,keywords=[])
            previous=self.vector_record_guard
            if self.selected_vector_output is not None and whole_selection:
                ordinal,selected_count=self.selected_vector_output
                nonempty=ast.Compare(left=copy.deepcopy(selected_count),ops=[ast.Gt()],comparators=[ast.Constant(0)])
                if fixed_vectors:
                    if self.selected_call_presence is not None:nonempty=copy.deepcopy(self.selected_call_presence)
                    valid=ast.BoolOp(op=ast.Or(),values=[ast.Compare(left=copy.deepcopy(selected_count),ops=[ast.Eq()],comparators=[ast.Constant(1)]),
                        ast.Compare(left=copy.deepcopy(selected_count),ops=[ast.Eq()],comparators=[ast.Constant(count)])])
                    check=ast.BinOp(left=ast.Constant(1.),op=ast.Div(),right=ast.IfExp(test=valid,body=ast.Constant(1.),orelse=ast.Constant(0.)))
                else:
                    nonempty=ast.BoolOp(op=ast.And(),values=[nonempty,ast.Compare(left=ast.Constant(j),ops=[ast.Lt()],comparators=[copy.deepcopy(selected_count)])])
                    check=ast.Constant(1.)
                self.vector_record_guard=(nonempty,check,True)
            if self.selected_vector_output is not None and not whole_selection and any(
                    isinstance(item,ast.Name) and item.id in self.selected_vector_inputs
                    for value in values for item in ast.walk(self.materialize(value))):
                ordinal,selected_count=self.selected_vector_output
                # A singleton capture broadcasts over selected rows; larger
                # mixed captures require equal lengths. A single event row
                # broadcast over a larger capture needs a whole-vector action.
                valid=ast.Constant(True) if count==1 else ast.Compare(
                    left=copy.deepcopy(selected_count),ops=[ast.Eq()],comparators=[ast.Constant(count)])
                check=ast.BinOp(left=ast.Constant(1.),op=ast.Div(),right=ast.IfExp(test=valid,body=ast.Constant(1.),orelse=ast.Constant(0.)))
                self.vector_record_guard=(ast.BoolOp(op=ast.And(),values=[
                    ast.Compare(left=copy.deepcopy(selected_count),ops=[ast.Gt()],comparators=[ast.Constant(0)]),
                    ast.Compare(left=copy.deepcopy(ordinal),ops=[ast.Eq()],comparators=[ast.Constant(j)])]),check)
            if self.selected_write_scope and self.selected_vector_output is not None:
                ordinal,selected_count=self.selected_vector_output
                self.vector_record_guard=(ast.BoolOp(op=ast.And(),values=[
                    ast.Compare(left=copy.deepcopy(selected_count),ops=[ast.Gt()],comparators=[ast.Constant(0)]),
                    ast.Compare(left=copy.deepcopy(ordinal),ops=[ast.Eq()],comparators=[ast.Constant(j)])]),ast.Constant(1.))
            try:expressions.append(self.expression(part,scope))
            finally:self.vector_record_guard=previous
        dtype=self.dtype(expressions[column]);parts=tuple(ArrayCell(self.materialize(v),dtype=dtype,zero_dim=True) for v in expressions)
        selection_shape=any(v.selection_shape for v in vectors) and all(v.selection_shape or len(v.columns)==1 for v in vectors)
        return ArrayCell(self.materialize(parts[column]),dtype=dtype,length=count,columns=parts,column=column,selection_shape=selection_shape)

    def broadcast_length(self,values):
        arrays=[value for value in values if isinstance(value,ArrayCell) and not value.zero_dim]
        lengths={value.length for value in arrays if value.length not in (None,1) and not value.selection_shape}
        if len(lengths)>1:raise ValueError('capture array shapes cannot be broadcast together')
        if any(value.length is None for value in arrays):return None
        return next(iter(lengths),max((value.length or 1 for value in arrays),default=1)) if arrays else None

    def physical_cast(self,value,dtype):
        from .training_equations import coerce_state_expression
        expression=self.materialize(value)
        if dtype=='integer' and self.dtype(value)!='integer':
            # NumPy's finite overflow cast differs by host implementation.
            # Snapshot its builtin array assignment, without invoking callback
            # Python. Serialized plans retain the producer's NumPy behavior.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                probe=np.zeros(1,dtype=np.int32);probe[:]=np.array([2147483648.],dtype=np.float64)
            positive=int(probe[0])
            test=ast.BoolOp(op=ast.And(),values=[
                ast.Compare(left=copy.deepcopy(expression),ops=[ast.GtE()],comparators=[ast.Constant(-2147483648.)]),
                ast.Compare(left=copy.deepcopy(expression),ops=[ast.Lt()],comparators=[ast.Constant(2147483648.)])])
            overflow=ast.IfExp(test=ast.Compare(left=copy.deepcopy(expression),ops=[ast.GtE()],comparators=[ast.Constant(2147483648.)]),
                               body=ast.Constant(positive),orelse=ast.Constant(-2147483648))
            return ast.IfExp(test=test,body=coerce_state_expression(expression,dtype),orelse=overflow)
        return coerce_state_expression(expression,dtype)

    def record(self, expression):
        # Expressions contain materialized immutable input snapshots, including
        # counter-addressed draws. An identical expression already evaluated
        # earlier cannot add an error or observe a later array mutation.
        if self.vector_record_guard is not None:
            condition,check=self.vector_record_guard[:2]
            expression=ast.IfExp(test=copy.deepcopy(condition),body=ast.Call(
                func=ast.Name(id='_b2_where',ctx=ast.Load()),args=[copy.deepcopy(check),expression,copy.deepcopy(expression)],keywords=[]),orelse=ast.Constant(0.))
            keys=self.whole_selection_keys if len(self.vector_record_guard)==3 and self.vector_record_guard[2] else self.selected_vector_keys
            keys.add(ast.dump(expression,include_attributes=False))
        expression=_bounded_tree(expression)
        key=(ast.dump(expression,include_attributes=False),self.record_scope)
        if key in self.executed_keys:return
        if self.operation_count >= self.eager_limit:
            raise ValueError(f'state effect action exceeds {self.eager_limit} eager operations')
        self.executed.append(copy.deepcopy(expression))
        self.executed_keys.add(key)
        self.operation_count+=1

    def result(self,node,*operands):
        # Materialize only after every operand has been evaluated. A left
        # array reference observes mutations made while evaluating the right.
        length=self.broadcast_length(operands)
        if isinstance(node,ast.BinOp):
            left,right=operands;dt=self.promote(operands)
            if (any(isinstance(value,ArrayCell) and value.dtype=='boolean' for value in operands)
                    and any(isinstance(value,ast.Constant) and type(value.value) is int for value in operands)):
                raise ValueError('Boolean array arithmetic with Python integers requires unsupported int64 promotion')
            if self.dtype(left)==self.dtype(right)=='boolean' and isinstance(node.op,(ast.Add,ast.Mult)):
                expression=ast.Call(func=ast.Name(id='_b2_logical_or' if isinstance(node.op,ast.Add) else '_b2_logical_and',ctx=ast.Load()),
                                    args=[self.materialize(left),self.materialize(right)],keywords=[])
            elif self.dtype(left)==self.dtype(right)=='boolean' and isinstance(node.op,ast.Sub):
                raise ValueError('NumPy boolean subtraction is unsupported')
            elif isinstance(node.op,ast.Pow) and dt=='integer':
                exponent=self.materialize(right)
                if not isinstance(exponent,ast.Constant) or type(exponent.value) is not int or not 0<=exponent.value<=32:
                    raise ValueError('integer array power requires a constant exponent 0..32')
                expression=ast.Constant(1)
                for _ in range(exponent.value):expression=ast.BinOp(left=expression,op=ast.Mult(),right=self.materialize(left))
            elif isinstance(node.op,ast.Pow) and dt=='boolean':
                raise ValueError('NumPy Boolean power requires unsupported int8 array promotion')
            else:
                args=[self.materialize(value) for value in operands]
                if dt=='integer':
                    args=[ast.Call(func=ast.Name(id='int',ctx=ast.Load()),args=[arg],keywords=[]) if self.dtype(value)=='boolean' else arg
                          for value,arg in zip(operands,args)]
                expression=ast.BinOp(left=args[0],op=copy.deepcopy(node.op),right=args[1])
        else:raise ValueError('unsupported effect operation')
        self.record(expression)
        dtype='float' if isinstance(node.op,ast.Div) else dt
        return ArrayCell(expression,dtype=dtype,length=length) if any(isinstance(v,ArrayCell) and not v.zero_dim for v in operands) else expression

    def dtype(self,value):
        if isinstance(value,ArrayCell):return value.dtype
        if isinstance(value,ast.Constant):return 'float' if type(value.value) is float else 'boolean' if type(value.value) is bool else 'integer'
        if isinstance(value,ast.Name):return self.input_dtypes.get(value.id,'float')
        if isinstance(value,ast.Compare):return 'boolean'
        if isinstance(value,ast.UnaryOp):return 'boolean' if isinstance(value.op,ast.Not) else self.dtype(value.operand)
        if isinstance(value,ast.BinOp):return 'float' if isinstance(value.op,ast.Div) else self.promote((value.left,value.right))
        if isinstance(value,ast.IfExp):return self.promote((value.body,value.orelse))
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Name):
            if value.func.id in ('bool','_b2_logical_and','_b2_logical_or','_b2_logical_not'):return 'boolean'
            if value.func.id=='int':return 'integer'
        return 'float'

    def promote(self,values):
        return 'float' if any(self.dtype(v)=='float' for v in values) else 'integer' if any(self.dtype(v)=='integer' for v in values) else 'boolean'

    def expression(self,node,environment):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                and node.func.id in environment and type(environment[node.func.id]) not in self.callable_types):
            raise ValueError('state effect call is shadowed by a local value: '+node.func.id)
        if isinstance(node,ast.Constant) and type(node.value) in (bool,int,float):return copy.deepcopy(node)
        if isinstance(node,ast.Name):
            if node.id not in environment:raise ValueError('unknown effect name: '+node.id)
            if type(environment[node.id]) in self.callable_types:raise ValueError('effect callable used as a value')
            value=environment[node.id]
            if environment is self.environment and node.id in self.selected_operand_copies:
                return copy.deepcopy(value)
            if node.id in self.indexed_reads:
                if not isinstance(value,ArrayCell) or value.zero_dim:
                    raise ValueError('NumPy conditional indexing requires a vector value: '+node.id)
                return ArrayCell(self.materialize(value),dtype=value.dtype,length=value.length)
            return value
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.FloorDiv,ast.Mod)):
            left=self.expression(node.left,environment);right=self.expression(node.right,environment)
            if any(isinstance(v,ArrayCell) and v.columns is not None for v in (left,right)):
                return self.vector_expression(node,(left,right),environment)
            return self.result(node,left,right)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=self.expression(node.operand,environment)
            if isinstance(value,ArrayCell) and value.columns is not None:return self.vector_expression(node,(value,),environment)
            if isinstance(value,ArrayCell) and value.dtype=='boolean':raise ValueError('NumPy boolean arrays do not support unary arithmetic')
            expression=ast.UnaryOp(op=copy.deepcopy(node.op),operand=self.materialize(value))
            self.record(expression)
            return ArrayCell(expression,dtype=value.dtype,length=value.length) if isinstance(value,ArrayCell) and not value.zero_dim else expression
        if isinstance(node,ast.Compare) and len(node.ops)==1:
            left=self.expression(node.left,environment);right=self.expression(node.comparators[0],environment)
            if any(isinstance(v,ArrayCell) and v.columns is not None for v in (left,right)):
                return self.vector_expression(node,(left,right),environment)
            length=self.broadcast_length((left,right))
            expression=ast.Compare(left=self.materialize(left),ops=copy.deepcopy(node.ops),comparators=[self.materialize(right)])
            self.record(expression)
            return ArrayCell(expression,dtype='boolean',length=length) if any(isinstance(v,ArrayCell) and not v.zero_dim for v in (left,right)) else expression
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and type(environment.get(node.func.id)) in self.callable_types:
            if node.keywords:raise ValueError('effect descriptor calls require positional arguments')
            function=environment[node.func.id]
            arguments=[self.expression(arg,environment) for arg in node.args]
            from .training_equations import PureFunction,PoissonNoise
            if type(function) is PureFunction:return self.call(function,arguments)
            if (type(function) is PoissonNoise and (self.indexed_reads or self.validate_empty_arrays)
                    and arguments and (not isinstance(arguments[0],ArrayCell) or arguments[0].zero_dim)):
                # NumPy validates a scalar lambda even for size=0. Validate
                # it without drawing/counting or adding a likelihood site.
                self.record(ast.Call(func=ast.Name(id='sqrt',ctx=ast.Load()),args=[self.materialize(arguments[0])],keywords=[]))
                maximum=float(np.iinfo(np.int64).max)
                maximum-=10.*np.sqrt(maximum)
                self.record(ast.Call(func=ast.Name(id='sqrt',ctx=ast.Load()),
                    args=[ast.BinOp(left=ast.Constant(float(maximum)),op=ast.Sub(),right=self.materialize(arguments[0]))],keywords=[]))
            expression=ast.Call(func=copy.deepcopy(node.func),args=[self.materialize(v) for v in arguments],keywords=[])
            self.record(expression)
            array=node.func.id in self.array_callables or any(isinstance(v,ArrayCell) and not v.zero_dim for v in arguments)
            return ArrayCell(expression,dtype='integer' if type(function) is PoissonNoise else 'float',length=self.broadcast_length(arguments)) if array else expression
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in self.functions:
            if node.keywords:raise ValueError('state effects require positional calls')
            arguments=[self.expression(arg,environment) for arg in node.args]
            return self.call(self.functions[node.func.id],arguments)
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                and node.func.id in ('exp','log','tanh','sqrt','sin','cos','clip','minimum','maximum',
                    'tan','cosh','sinh','log10','expm1','log1p','exprel','arccos','arcsin','arctan','ceil','abs','sign','floor',
                    'int','bool','_b2_float','_b2_mod','_b2_where','_b2_logical_and','_b2_logical_or','_b2_logical_not','_b2_owned_array')
                and not node.keywords):
            values=[self.expression(arg,environment) for arg in node.args]
            if any(isinstance(v,ArrayCell) and v.columns is not None for v in values):return self.vector_expression(node,values,environment)
            expression=ast.Call(func=copy.deepcopy(node.func),args=[self.materialize(v) for v in values],keywords=[])
            self.record(expression)
            if node.func.id=='_b2_float' and len(values)==1 and isinstance(values[0],ArrayCell) and values[0].dtype=='float':return values[0]
            dtype=('boolean' if node.func.id in ('bool','_b2_logical_and','_b2_logical_or','_b2_logical_not')
                   else 'integer' if node.func.id=='int'
                   else self.promote(values[1:] if node.func.id=='_b2_where' else values) if node.func.id in ('clip','minimum','maximum','_b2_mod','_b2_where','abs','sign')
                   else 'float')
            return ArrayCell(expression,dtype=dtype,zero_dim=all(not isinstance(v,ArrayCell) or v.zero_dim for v in values),length=self.broadcast_length(values)) if node.func.id in ('_b2_owned_array','_b2_where') or any(isinstance(v,ArrayCell) and not v.zero_dim for v in values) else expression
        raise ValueError('unsupported effect expression: '+ast.dump(node))

    def call(self,function,arguments):
        from .training_equations import PureFunction,compile_training_equation
        if type(function) not in (PureFunction,StateEffectFunction):
            raise ValueError('state effect callback requires an authenticated descriptor')
        flags=function.scalarize_arguments
        if (type(flags) is not tuple or flags and len(flags)!=len(arguments)
                or any(type(flag) is not bool for flag in flags) or type(function.array_return) is not bool):
            raise ValueError('invalid effect callback array contract')
        arguments=[self.materialize(value) if flags and flags[position] and isinstance(value,ArrayCell) and value.zero_dim else value
                   for position,value in enumerate(arguments)]
        def returned(value):
            return ArrayCell(self.materialize(value),dtype=self.dtype(value),zero_dim=True) if function.array_return and not isinstance(value,ArrayCell) else value
        if type(function) is PureFunction:
            if type(function.arguments) is not tuple or len(function.arguments)>16:
                raise ValueError('invalid pure effect callback arguments')
            compile_training_equation('__descriptor('+','.join('0.' for _ in function.arguments)+')',parameters={'__descriptor':function})
            if len(arguments)!=len(function.arguments):raise ValueError('pure effect callback arity mismatch')
            if id(function) in self.pure_stack or len(self.pure_stack)>=8:raise ValueError('recursive or excessively nested pure effect callback')
            environment={name:value if type(value) in self.callable_types else ast.Constant(value=value) for name,value in function.parameters}
            for name in function.captured_arrays:
                value=environment[name];environment[name]=ArrayCell(value,writable=False,dtype=self.dtype(value),zero_dim=True)
            environment.update(zip(function.arguments,arguments));self.pure_stack.append(id(function))
            try:
                for position,(name,expression) in enumerate(function.statements):
                    value=self.expression(_bounded_tree(ast.parse(expression,mode='eval')).body,environment)
                    previous=environment.get(name)
                    if position in function.augmented and isinstance(previous,ArrayCell) and previous.dtype!=self.dtype(value):
                        raise ValueError('pure effect augmented assignment changes array dtype')
                    if position in function.augmented and isinstance(previous,ArrayCell) and not isinstance(value,ArrayCell):
                        value=ArrayCell(self.materialize(value),dtype=previous.dtype,zero_dim=previous.zero_dim)
                    environment[name]=value
                return returned(self.expression(_bounded_tree(ast.parse(function.expression,mode='eval')).body,environment))
            finally:self.pure_stack.pop()
        if type(function) is StateEffectFunction:
            if (type(function.arguments) is not tuple or type(function.body) is not str
                    or type(function.constants) is not tuple or len(function.body)>8192
                    or len(function.arguments)>16
                    or any(type(name) is not str or not name.isidentifier() for name in function.arguments)
                    or len(set(function.arguments))!=len(function.arguments)):
                raise ValueError('invalid state effect descriptor')
            if len(function.arguments)!=len(arguments):raise ValueError('state effect callback arity mismatch')
            for item in function.constants:
                if (type(item) is not tuple or len(item)!=2 or type(item[0]) is not str
                        or type(item[1]) not in (bool,int,float)
                        or type(item[1]) is int and not -2**31<=item[1]<2**31
                        or type(item[1]) is float and not math.isfinite(item[1])):
                    raise ValueError('invalid state effect scalar capture')
            environment={name:ast.Constant(value=value) for name,value in function.constants}
            if type(function.captured_arrays) is not tuple or type(function.capture_bindings) is not tuple or type(function.parameter_captures) is not tuple:
                raise ValueError('invalid mutable capture descriptor')
            captures=dict(function.captured_arrays);bindings=dict(function.capture_bindings)
            if not set(function.parameter_captures)<=set(captures):raise ValueError('invalid readonly capture descriptor')
            if set(captures)!=set(bindings):
                raise ValueError('mutable captures require persistent state bindings')
            for name,source in bindings.items():
                value=self.environment.get(source)
                array=captures[name]
                if type(array) is not np.ndarray or not isinstance(value,ArrayCell) or value.origin is None and name not in function.parameter_captures:
                    raise ValueError('mutable capture binding must name canonical persistent array storage')
                dtype='integer' if array.dtype==np.dtype('int32') else 'boolean' if array.dtype==np.dtype('bool') else 'float'
                if value.dtype!=dtype or (value.writable if name in function.parameter_captures else not value.writable):
                    raise ValueError('mutable capture binding has incompatible dtype or ownership')
                if value.length is not None and value.length!=array.size:
                    raise ValueError('mutable capture binding has incompatible vector length')
                value.length=int(array.size)
                if source in self.capture_vectors:
                    columns,column=self.capture_vectors[source]
                    value=ArrayCell(self.materialize(value),value.writable,value.origin,dtype=value.dtype,
                                    length=value.length,columns=columns,column=column)
                environment[name]=value
            environment.update(zip(function.arguments,arguments))
            tree=_bounded_tree(ast.parse(function.body))
            if not 1<=len(tree.body)<=65:raise ValueError('state effect body exceeds statement budget')
            locals_={node.id for node in ast.walk(tree) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)}|set(function.arguments)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in locals_:
                    raise ValueError('state effect call is shadowed by a local value: '+node.func.id)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in self.functions:
                    raise ValueError('nested state effect callbacks are unsupported')
            for statement in tree.body:
                if isinstance(statement,ast.Return):
                    if statement.value is None:raise ValueError('state effect return requires value')
                    return returned(self.expression(statement.value,environment))
                self.statement(statement,environment,physical=False)
            raise ValueError('state effect callback missing return')
        raise ValueError('state effect callback requires an authenticated descriptor')

    def statement(self,node,environment,physical):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id;value=self.expression(node.value,environment)
            # NumPy code binds local array references, then writes physical
            # results at the end of the generated action. Returned aliases
            # remain live through later augmented writes in this action.
            environment[name]=value
            return
        if isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name):
            if not isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod)):
                raise ValueError('unsupported state effect augmented operation')
            name=node.target.id
            if name not in environment:raise ValueError('unbound effect local: '+name)
            left=environment[name];right=self.expression(node.value,environment)
            operation=ast.BinOp(left=ast.Constant(0.),op=node.op,right=ast.Constant(0.))
            value=self.vector_expression(operation,(left,right),environment) if any(isinstance(v,ArrayCell) and v.columns is not None for v in (left,right)) else self.result(operation,left,right)
            if isinstance(left,ArrayCell):
                if not left.writable:raise ValueError('effect cannot mutate readonly storage')
                dynamic_shape=None
                if self.selected_vector_output is not None and isinstance(value,ArrayCell):
                    count=self.selected_vector_output[1]
                    if left.selection_shape and not value.selection_shape and value.columns is not None:
                        dynamic_shape=ast.Compare(left=copy.deepcopy(count),ops=[ast.Eq()],comparators=[ast.Constant(len(value.columns))])
                    elif not left.selection_shape and value.selection_shape and left.length is not None:
                        dynamic_shape=ast.BoolOp(op=ast.Or(),values=[
                            ast.Compare(left=copy.deepcopy(count),ops=[ast.Eq()],comparators=[ast.Constant(1)]),
                            ast.Compare(left=copy.deepcopy(count),ops=[ast.Eq()],comparators=[ast.Constant(left.length)])])
                    if dynamic_shape is not None:
                        check=ast.BinOp(left=ast.Constant(1.),op=ast.Div(),right=ast.IfExp(test=dynamic_shape,body=ast.Constant(1.),orelse=ast.Constant(0.)))
                        previous=self.vector_record_guard
                        self.vector_record_guard=(copy.deepcopy(self.selected_call_presence) if self.selected_call_presence is not None else ast.Compare(left=copy.deepcopy(count),ops=[ast.Gt()],comparators=[ast.Constant(0)]),ast.Constant(1.),True)
                        try:self.record(check)
                        finally:self.vector_record_guard=previous
                if left.length is not None and isinstance(value,ArrayCell) and value.length is not None and left.length!=value.length:
                    if dynamic_shape is None:raise ValueError('capture augmented assignment cannot broadcast into a different shape')
                if left.dtype!='float' and left.dtype!=self.dtype(value):raise ValueError('effect augmented assignment changes array dtype')
                if left.columns is not None:
                    for j,target in enumerate(left.columns):
                        if value.columns is not None and j>=len(value.columns):continue
                        while target.source is not None:target=target.source
                        target.value=self.materialize(self.element(value,j))
                        if target.origin is not None:self.effect_writes.add(target.origin)
                else:
                    left.value=self.materialize(value)
                    if isinstance(value,ArrayCell) and value.columns is not None:
                        # An independently loaded selected target still owns a
                        # vector result. Keep all columns through inplace writes
                        # so the eventual row writeback selects its own column.
                        left.columns=value.columns
                        left.column=value.column
                        left.length=value.length
                        left.selection_shape=value.selection_shape
                    if left.origin is not None:self.effect_writes.add(left.origin)
            else:
                # Scalar += array creates a new array. Its subsequent aliases
                # must retain that cell rather than becoming immutable ASTs.
                environment[name]=value if isinstance(value,ArrayCell) else self.materialize(value)
            return
        raise ValueError('unsupported effect statement: '+ast.dump(node))

    def transform(self,code):
        if type(code) is not str or len(code)>16384:raise ValueError('state effect statements exceed budget')
        tree=_bounded_tree(ast.parse(code))
        if len(tree.body)>256:raise ValueError('state effect action exceeds 256 statements')
        for statement in tree.body:self.statement(statement,self.environment,physical=True)
        return {name:ast.unparse(self.materialize(self.environment[name])) for name in self.roots}

def compile_state_effect_transform(statements, *, states, parameters, array_states,
                                   writable_states, state_types=None,array_parameters=(),temporary_parameters=(),
                                   physical_slots=None, writeback_order=(), eager_limit=64,
                                   copied_array_states=(), reload_arrays_each_statement=False,array_callables=(),
                                   write_guards=None,indexed_guard_reads=(),scalar_write_guards=False,parameter_types=None,
                                   predicate_outputs=None,scalar_eager=False,eager_guard=None,empty_vector=False,
                                   unconditional_states=(),retained_array_states=(),unconditional_parameters=(),whole_eager=False,array_aliases=None,capture_vectors=None,selected_output=None,copied_array_sources=None,selected_vectors=None,selected_accumulators=(),separate_whole_eager=False,selected_call_presence=None,selected_guards=None,selected_row_operands=(),copied_array_aliases=None,retained_copied_outputs=(),whole_array_locals=()):
    """Compose state-effect callbacks into the existing native action ABI."""
    from .training_equations import StateSlot, RefractoryActive, _compile_training_ast
    states=dict(states);parameters=dict(parameters);types={} if state_types is None else dict(state_types)
    copied_array_sources={} if copied_array_sources is None else dict(copied_array_sources)
    selected_accumulators=set(selected_accumulators)
    selected_guards={} if selected_guards is None else dict(selected_guards)
    copied_array_aliases={} if copied_array_aliases is None else dict(copied_array_aliases)
    if not set(retained_copied_outputs)<=set(copied_array_states) or any(name not in copied_array_states or source not in copied_array_states for name,source in copied_array_aliases.items()):
        raise ValueError('caller aliases and retained outputs require copied arrays')
    if (selected_guards and selected_output is None or
            any(name not in copied_array_states or gate not in states or types.get(states[gate])!='boolean' for name,gate in selected_guards.items())):
        raise ValueError('selected guards require detached Boolean batch masks')
    if selected_call_presence is not None and (selected_output is None or not isinstance(selected_call_presence,ast.expr)):
        raise ValueError('selected call presence requires a batch condition expression')
    if selected_accumulators and (not reload_arrays_each_statement or not selected_accumulators<=set(copied_array_states)):
        raise ValueError('live accumulation requires copied per-statement state operands')
    if (not set(copied_array_sources)<=set(copied_array_states) or
            not set(copied_array_sources.values())<=set(array_states)-set(copied_array_states) or
            any(types.get(states[name],'float')!=types.get(states[source],'float') for name,source in copied_array_sources.items())):
        raise ValueError('copied array reload sources require matching physical capture types')
    if selected_output is not None and (type(selected_output) is not tuple or len(selected_output)!=2
                                       or not all(isinstance(value,ast.expr) for value in selected_output)):
        raise ValueError('selected output requires ordinal and length expressions')
    if not set(unconditional_states)<=set(array_states)-set(copied_array_states):
        raise ValueError('unconditional effects require physical array states')
    if not set(retained_array_states)<=set(unconditional_states):
        raise ValueError('retained reload arrays require whole physical states')
    if not set(unconditional_parameters)<=set(array_parameters)-set(temporary_parameters):
        raise ValueError('whole parameter arrays require readonly bindings')
    if type(whole_eager) is not bool or whole_eager and not (unconditional_states or unconditional_parameters):
        raise ValueError('whole eager validation requires whole arrays')
    if type(separate_whole_eager) is not bool or separate_whole_eager and selected_output is None:
        raise ValueError('separate whole checks require a selected batch output')
    if type(scalar_eager) is not bool:raise ValueError('invalid scalar eager contract')
    if type(empty_vector) is not bool or empty_vector and not scalar_eager:
        raise ValueError('empty vector requires scalar eager validation')
    predicate_outputs={} if predicate_outputs is None else dict(predicate_outputs)
    if any(slot not in states.values() or type(coefficients) is not tuple or len(coefficients)!=2
           or any(type(value) not in (int,float) or not np.isfinite(value) or value<=0 for value in coefficients)
           for slot,coefficients in predicate_outputs.items()):
        raise ValueError('invalid effect predicate output contract')
    write_guards={} if write_guards is None else dict(write_guards)
    if type(scalar_write_guards) is not bool or scalar_write_guards and (indexed_guard_reads or copied_array_states or set(write_guards or ())&set(array_states)):
        raise ValueError('scalar conditional writes require scalar operands')
    locals_={node.id for node in ast.walk(_bounded_tree(ast.parse(statements))) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store)} if write_guards or indexed_guard_reads else set()
    if (any(name not in states or not (gate in states and types.get(states[gate])=='boolean' or type(parameters.get(gate)) is RefractoryActive)
            for name,gate in write_guards.items()) or not set(indexed_guard_reads)<=set(states)|set(parameters)|locals_):
        raise ValueError('invalid state effect conditional write contract')
    if type(eager_limit) is not int or eager_limit not in (64,128):
        raise ValueError("state effect eager limit must be 64 or 128")
    if (not states or set(states)&set(parameters) or
            any(type(slot) is not int or not 0<=slot<64 for slot in states.values())):
        raise ValueError('state effects require named context slots 0..63')
    if not set(copied_array_states)<=set(array_states) or type(reload_arrays_each_statement) is not bool:
        raise ValueError('invalid indexed-copy ownership')
    if reload_arrays_each_statement and physical_slots is not None:
        raise ValueError('statement reloads require independent indexed reads')
    if not set(array_states)<=set(states) or not set(writable_states)<=set(states):
        raise ValueError('invalid state effect ownership names')
    if any(types.get(states[name],'float') not in ('float','integer','boolean') for name in array_states):
        raise ValueError('state effects require float, int32 or boolean array inputs')
    functions={name:value for name,value in parameters.items() if type(value) is StateEffectFunction}
    protected={source for function in functions.values() for capture,source in function.capture_bindings
               if capture in function.parameter_captures and source in states}
    writable_states=set(writable_states)-protected
    aliases={} if array_aliases is None else dict(array_aliases)
    raw=set(array_states)-set(copied_array_states)
    if (not set(aliases)<=raw or not set(aliases.values())<=raw or
            any(aliases.get(target,target)!=target or types.get(states[name],'float')!=types.get(states[target],'float')
                for name,target in aliases.items())):
        raise ValueError('raw array aliases require canonical matching storage types')
    if aliases and physical_slots is not None:raise ValueError('raw aliases and physical slot tables are separate contracts')
    scalar_parameters={name:value for name,value in parameters.items() if name not in functions}
    engine=Effects(states,functions,scalar_parameters,array_states=array_states,writable_states=writable_states,
                   array_parameters=array_parameters,temporary_parameters=temporary_parameters,eager_limit=eager_limit,array_callables=array_callables,
                   state_types=types,parameter_types=parameter_types)
    engine.validate_empty_arrays=scalar_eager
    engine.selected_vector_inputs=(set(array_states)|set(array_parameters))-(set(unconditional_states)|set(unconditional_parameters))
    engine.selected_vector_output=selected_output
    engine.selected_call_presence=selected_call_presence
    physical=None
    if physical_slots is not None:
        physical=dict(physical_slots)
        if (set(physical)!=set(states.values()) or
                any(type(slot) is not int or slot not in physical or physical[slot]!=slot
                    for slot in physical.values())):
            raise ValueError('state effect physical aliases require canonical context slots')
        if (type(writeback_order) is not tuple or len(writeback_order)!=len(set(writeback_order))
                or not set(writeback_order)<=set(array_states)):
            raise ValueError('invalid NumPy state effect writeback order')
    cells={}
    def origin_of(name,slot):
        return states[aliases.get(name,name)] if physical is None else physical[slot]
    for name,slot in states.items():
        value=engine.environment[name]
        if isinstance(value,ArrayCell):
            origin=origin_of(name,slot)
            if origin not in cells:
                canonical=aliases.get(name,name)
                cells[origin]=ArrayCell(engine.materialize(engine.environment[canonical]),False,origin,dtype=value.dtype)
            if cells[origin].dtype!=value.dtype:raise ValueError('aliased state types differ')
            cells[origin].writable|=value.writable
    for name,slot in states.items():
        value=engine.environment[name]
        if isinstance(value,ArrayCell):
            origin=origin_of(name,slot);cell=cells[origin]
            if name in copied_array_states or physical is not None and name in writeback_order:
                value.value=engine.materialize(cell);value.origin=None
            elif value.writable==cell.writable:
                engine.roots[name]=engine.environment[name]=cell
            else:
                engine.roots[name]=engine.environment[name]=ArrayCell(engine.materialize(cell),False,origin,dtype=cell.dtype,source=cell)
    vectors={} if capture_vectors is None else dict(capture_vectors)
    for name,group in vectors.items():
        if (type(group) is not tuple or not group or name not in group or len(set(group))!=len(group)
                or not set(group)<=set(array_states)-set(copied_array_states)|set(array_parameters)-set(temporary_parameters)):
            raise ValueError('capture vectors require whole physical or readonly parameter columns')
        engine.capture_vectors[name]=(tuple(engine.environment[source] for source in group),group.index(name))
    whole_array_locals=set(whole_array_locals)
    if not whole_array_locals<=set(vectors) or not whole_array_locals<=set(unconditional_states)|set(unconditional_parameters):
        raise ValueError('whole caller locals require physical or readonly vectors')
    for name in whole_array_locals:
        value=engine.environment[name];columns,column=engine.capture_vectors[name]
        engine.environment[name]=ArrayCell(engine.materialize(value),value.writable,value.origin,dtype=value.dtype,
                                          length=len(columns),columns=columns,column=column)
    selected_vectors={} if selected_vectors is None else dict(selected_vectors)
    if not set(selected_row_operands)<=set(selected_vectors) or selected_row_operands and not selected_guards:
        raise ValueError('row operands require masked compact batch vectors')
    if selected_vectors and selected_output is None:raise ValueError('whole selected operands require arrival shape metadata')
    for name,expressions in selected_vectors.items():
        if name not in set(copied_array_states)|set(temporary_parameters) or type(expressions) is not tuple or not expressions or not all(isinstance(v,ast.expr) for v in expressions):
            raise ValueError('whole selected operands require copied state columns')
        root=engine.environment[name]
        parts=tuple(ArrayCell(copy.deepcopy(value),dtype=root.dtype,zero_dim=True) for value in expressions)
        engine.environment[name]=ArrayCell(engine.materialize(parts[0]),dtype=root.dtype,length=len(parts),columns=parts,selection_shape=True)
        engine.whole_selection_keys.update(ast.dump(value,include_attributes=False) for value in expressions)
    for name,source in copied_array_aliases.items():engine.environment[name]=engine.environment[source]
    guarded_prefix=[];assignment_types={};projected_keys=set();selected_output_lengths=set()
    def output_value(name,value):
        if selected_output is None or name not in copied_array_states or not isinstance(value,ArrayCell) or value.columns is None:
            return engine.materialize(value)
        ordinal,count=selected_output;size=len(value.columns)
        selected_output_lengths.add(size)
        result=engine.materialize(value.columns[-1])
        for column in reversed(range(size-1)):
            result=ast.IfExp(test=ast.Compare(left=copy.deepcopy(ordinal),ops=[ast.Eq()],comparators=[ast.Constant(column)]),
                             body=engine.materialize(value.columns[column]),orelse=result)
        check=ast.Constant(1.)
        if size>1:
            # Empty pathways skip writeback; otherwise NumPy indexed assignment
            # requires matching rows. Keep the checked expression in the typed
            # result, including integer/Boolean outputs, without float coercion.
            valid=ast.IfExp(test=ast.Compare(left=copy.deepcopy(count),ops=[ast.Eq()],comparators=[ast.Constant(size)]),body=ast.Constant(1.),
                orelse=ast.IfExp(test=ast.Compare(left=copy.deepcopy(count),ops=[ast.Eq()],comparators=[ast.Constant(0)]),body=ast.Constant(1.),orelse=ast.Constant(0.)))
            check=ast.BinOp(left=ast.Constant(1.),op=ast.Div(),right=valid)
            if value.selection_shape:
                check=ast.BinOp(left=ast.Constant(1.),op=ast.Div(),right=ast.IfExp(
                    test=ast.Compare(left=copy.deepcopy(count),ops=[ast.LtE()],comparators=[ast.Constant(size)]),body=ast.Constant(1.),orelse=ast.Constant(0.)))
            elif selected_call_presence is not None:
                check=ast.BinOp(left=ast.Constant(1.),op=ast.Div(),right=ast.IfExp(
                    test=ast.Compare(left=copy.deepcopy(count),ops=[ast.Eq()],comparators=[ast.Constant(size)]),body=ast.Constant(1.),orelse=ast.Constant(0.)))
                previous=engine.vector_record_guard
                engine.vector_record_guard=(copy.deepcopy(selected_call_presence),ast.Constant(1.),True)
                try:engine.record(check)
                finally:engine.vector_record_guard=previous
        result=ast.Call(func=ast.Name(id='_b2_where',ctx=ast.Load()),args=[check,result,copy.deepcopy(result)],keywords=[])
        projected_keys.add(ast.dump(result,include_attributes=False))
        return result
    def statement_effect(statement):
        target=statement.targets[0].id if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name) else statement.target.id if isinstance(statement,ast.AugAssign) and isinstance(statement.target,ast.Name) else None
        gate=write_guards.get(target)
        first=len(engine.executed)
        if gate is None:
            if selected_guards and isinstance(statement,ast.AugAssign) and target in copied_array_states:
                # Evaluate all RHS callbacks once against the compact masked
                # batch. The subsequent local/LHS operation belongs to this
                # selected row, including a live repeated-target accumulator.
                temporary='_b2_selected_write_rhs'
                while temporary in engine.environment:temporary+='_'
                # Boolean advanced indexing makes each callback argument an
                # independent array. Its in-place mutations cannot change the
                # caller's local array, or the LHS slice already read by +=.
                # Keep the returned mutated argument and physical captures.
                selected_locals=copy.deepcopy({name:engine.environment[name] for name in copied_array_states})
                previous_copies=engine.selected_operand_copies
                engine.selected_operand_copies=set(copied_array_states)|set(temporary_parameters)
                try:right=engine.expression(statement.value,engine.environment)
                finally:engine.selected_operand_copies=previous_copies
                engine.environment.update(selected_locals)
                engine.environment[temporary]=right
                staged=copy.deepcopy(statement);staged.value=ast.Name(id=temporary,ctx=ast.Load())
                previous=engine.selected_write_scope;engine.selected_write_scope=True
                try:engine.statement(staged,engine.environment,physical=True)
                finally:engine.selected_write_scope=previous;del engine.environment[temporary]
            else:engine.statement(statement,engine.environment,physical=True)
            value=engine.environment.get(target)
            if selected_output is None and target in copied_array_states and isinstance(value,ArrayCell) and value.columns is not None and len(value.columns)>1:
                raise ValueError('whole capture return requires selection-aware array writeback')
            if write_guards:
                guarded_prefix.extend(engine.executed[first:]);del engine.executed[first:]
            return
        left=engine.environment[target]
        if scalar_write_guards:
            # NumPy's repeated-index fallback emits an ordinary Python if.
            # It skips the whole callback on a false gate, including physical
            # closure arrays. Ordinary pathway arguments are scalar locals.
            old=engine.materialize(left)
            old_cells={origin:engine.materialize(cell) for origin,cell in cells.items()}
            gate_expr=engine.materialize(engine.environment[gate])
            engine.record_scope=ast.dump(gate_expr,include_attributes=False)
            try:engine.statement(statement,engine.environment,physical=True)
            finally:engine.record_scope=None
            guarded_prefix.extend(ast.IfExp(test=copy.deepcopy(gate_expr),body=expression,orelse=ast.Constant(0.))
                                  for expression in engine.executed[first:])
            del engine.executed[first:]
            for origin,cell in cells.items():
                previous=old_cells[origin];current=engine.materialize(cell)
                if ast.dump(previous,include_attributes=False)!=ast.dump(current,include_attributes=False):
                    cell.value=ast.IfExp(test=copy.deepcopy(gate_expr),body=current,orelse=previous)
            if isinstance(engine.environment[target],ArrayCell) and not engine.environment[target].zero_dim:
                raise ValueError('scalar event writeback cannot store a whole captured array')
            engine.environment[target]=ast.IfExp(test=gate_expr,body=engine.materialize(engine.environment[target]),orelse=old)
            return
        if not isinstance(left,ArrayCell) or left.zero_dim:
            raise ValueError('NumPy conditional target requires a vector array')
        old=engine.materialize(left)
        gate_expr=engine.materialize(engine.environment[gate])
        engine.record_scope=ast.dump(gate_expr,include_attributes=False)
        engine.indexed_reads=set(indexed_guard_reads)
        try:
            if isinstance(statement,ast.AugAssign):
                right=engine.expression(statement.value,engine.environment)
                value=engine.result(ast.BinOp(left=ast.Constant(0.),op=statement.op,right=ast.Constant(0.)),
                    ArrayCell(old,dtype=left.dtype),right)
                if left.dtype!='float' and left.dtype!=engine.dtype(value):
                    raise ValueError('effect augmented assignment changes array dtype')
            else:value=engine.expression(statement.value,engine.environment)
        finally:engine.indexed_reads=set();engine.record_scope=None
        assignment_types[target]=engine.dtype(value)
        vector_names=set(array_states)|set(array_parameters)|set(array_callables)
        for expression in engine.executed[first:]:
            used={node.id for node in ast.walk(expression) if isinstance(node,ast.Name)}
            guarded_prefix.append(ast.IfExp(test=copy.deepcopy(gate_expr),body=expression,orelse=ast.Constant(0.)) if used&vector_names else expression)
        del engine.executed[first:]
        # A masked assignment mutates the existing local array. Earlier
        # borrowed aliases continue to observe its selected rows.
        left.value=ast.IfExp(test=gate_expr,body=engine.physical_cast(value,left.dtype),orelse=old)
        if left.origin is not None:engine.effect_writes.add(left.origin)

    if reload_arrays_each_statement:
        tree=_bounded_tree(ast.parse(statements))
        if type(statements) is not str or len(statements)>16384 or len(tree.body)>256:
            raise ValueError('state effect statements exceed budget')
        physical_values={name:engine.materialize(engine.environment[name]) for name in array_states}
        for name in selected_accumulators:physical_values[name]=ast.Name(id=name,ctx=ast.Load())
        physical_vectors={name:copy.deepcopy(engine.environment[name]) for name in selected_vectors}
        parameter_arrays=set(array_parameters);temporary=set(temporary_parameters)
        for statement in tree.body:
            callback_reads={item.id for call in ast.walk(statement)
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id in engine.functions
                for argument in call.args for item in ast.walk(argument) if isinstance(item,ast.Name)}
            used={n.id for n in ast.walk(statement) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
            if isinstance(statement,ast.AugAssign):used.add(statement.target.id)
            for name in used & (set(array_states)-set(retained_array_states)):
                value=copy.deepcopy(physical_values[name])
                if name in copied_array_sources:
                    # A new advanced-index read observes preceding whole writes,
                    # but is an independent selected copy rather than a borrow.
                    value=ast.Call(func=ast.Name(id='_b2_owned_array',ctx=ast.Load()),
                                   args=[engine.materialize(engine.environment[copied_array_sources[name]])],keywords=[])
                    projected_keys.add(ast.dump(value,include_attributes=False))
                engine.environment[name]=ArrayCell(value,name in writable_states,None,dtype=types.get(states[name],'float'))
                if name in physical_vectors and (name not in selected_accumulators or name in callback_reads):engine.environment[name]=copy.deepcopy(physical_vectors[name])
            for name in used & parameter_arrays-whole_array_locals:
                engine.environment[name]=ArrayCell(ast.Name(id=name,ctx=ast.Load()),name in temporary,None,dtype=engine.input_dtypes.get(name,'float'))
                if name in physical_vectors:engine.environment[name]=copy.deepcopy(physical_vectors[name])
            if isinstance(statement,ast.AugAssign) and statement.target.id in selected_accumulators:
                target=statement.target.id
                # NumPy add.at reads the live destination after computing the
                # RHS from an independent selected copy, including callbacks
                # that mutate that copy. Never reuse it as the physical LHS.
                right=engine.expression(statement.value,engine.environment)
                temporary='_b2_live_accumulation_rhs'
                while temporary in engine.environment:temporary+='_'
                engine.environment[temporary]=right
                engine.environment[target]=ArrayCell(copy.deepcopy(physical_values[target]),dtype=types.get(states[target],'float'))
                staged=copy.deepcopy(statement);staged.value=ast.Name(id=temporary,ctx=ast.Load())
                try:statement_effect(staged)
                finally:del engine.environment[temporary]
            else:statement_effect(statement)
            assigned={n.id for n in ast.walk(statement) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
            for name in assigned & set(array_states):
                if name in physical_vectors and name not in selected_accumulators:physical_vectors[name]=copy.deepcopy(engine.environment[name])
                value=output_value(name,engine.environment[name])
                physical_values[name]=engine.physical_cast(value,types.get(states[name],'float')) if types else value
        outputs={name:ast.unparse(output_value(name,engine.environment[name]) if name in retained_array_states or name not in array_states else physical_values[name]) for name in states}
    elif write_guards:
        tree=_bounded_tree(ast.parse(statements))
        if type(statements) is not str or len(statements)>16384 or len(tree.body)>256:
            raise ValueError('state effect statements exceed budget')
        for statement in tree.body:statement_effect(statement)
        outputs={name:ast.unparse(output_value(name,engine.environment[name])) for name in states}
    else:
        tree=_bounded_tree(ast.parse(statements))
        if type(statements) is not str or len(statements)>16384 or len(tree.body)>256:
            raise ValueError('state effect statements exceed budget')
        for statement in tree.body:statement_effect(statement)
        outputs={name:ast.unparse(output_value(name,engine.environment[name])) for name in states}
    # Whole caller bindings are local references. Assignment rebinds that
    # reference; only an in-place mutation recorded in effect_writes updates
    # its physical vector. It must not overwrite a previously borrowed array.
    explicit=({n.id for n in ast.walk(ast.parse(statements)) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}&set(states))-whole_array_locals
    if physical is not None:
        if set(writeback_order)!=explicit&set(array_states):
            raise ValueError('NumPy writeback order must cover all explicit array writes')
        # A writeback changes physical storage and every live borrowed alias,
        # but it never changes previously computed arrays or copied locals.
        for name in writeback_order:
            origin=physical[states[name]]
            from .training_equations import coerce_state_expression
            value=engine.materialize(engine.environment[name]);dtype=types.get(states[name],'float')
            cells[origin].value=engine.physical_cast(engine.environment[name],dtype) if types else value
            engine.effect_writes.add(origin)
        effect_slots={states[name] for name in array_states if physical[states[name]] in engine.effect_writes}
        written=effect_slots|{states[name] for name in explicit-set(array_states)}
    else:
        effect_slots=engine.effect_writes
        written={states[name] for name in explicit}|effect_slots
    if not written:raise ValueError('state effect action must write persistent state')
    written|={states[name] for name in retained_copied_outputs}
    bindings={**scalar_parameters,**{'_b2_effect_context_'+str(slot):StateSlot(slot,types.get(slot,'float')) for slot in states.values()}}
    class Inputs(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id in states:return ast.copy_location(ast.Name(id='_b2_effect_context_'+str(states[node.id]),ctx=ast.Load()),node)
            return node
    def compile_value(expression,predicate=None):
        expression=Inputs().visit(copy.deepcopy(expression))
        return _compile_training_ast(expression,parameters=bindings,states=['v'],allow_select=True,typed=bool(types) or selected_output is not None,deduplicate=True,
                                     predicate_surrogate=predicate)
    vector_names=set(array_states)|set(array_parameters)|set(array_callables)
    row_parts={ast.dump(part,include_attributes=False):name for name in selected_row_operands for part in selected_vectors[name]}
    row_part_types={type(part) for name in selected_row_operands for part in selected_vectors[name]}
    def row_expression(expression):
        if not row_parts:return expression
        class RowOperands(ast.NodeTransformer):
            def visit(self,node):
                name=row_parts.get(ast.dump(node,include_attributes=False)) if type(node) in row_part_types else None
                return ast.copy_location(ast.Name(id=name,ctx=ast.Load()),node) if name is not None else super().visit(node)
        return RowOperands().visit(copy.deepcopy(expression))
    eager_expressions=guarded_prefix if write_guards else engine.executed
    prefix=[compile_value(ast.IfExp(test=ast.Name(id=eager_guard,ctx=ast.Load()),body=expression,orelse=ast.Constant(0.))
                          if eager_guard is not None and any(isinstance(node,ast.Name) and node.id in vector_names for node in ast.walk(expression))
                          else expression) for expression in eager_expressions]
    def joined(expression,predicate=None,roots=None):
        from .protocol import canonical_bytes
        groups=(prefix if roots is None else roots)+[compile_value(expression,predicate)];nodes=[];ends=[];interned={}
        for group in groups:
            remap={}
            for local,original in enumerate(group):
                node=dict(original)
                fields=['arg','left','right','low','high','condition','yes','no','rate','slope','scale']
                if node['op'] in ('timed_parameter','parameter_gather','integer_parameter_gather'):
                    fields+=['index']
                    if node['op']=='timed_parameter':fields+=['time']
                for field in fields:
                    if field in node:node[field]=remap[node[field]]
                # All supported nodes are deterministic within this action,
                # including bound counter-addressed draws. Reuse only nodes
                # already evaluated in source order; keep every eager root.
                key=canonical_bytes(node)
                if key not in interned:interned[key]=len(nodes);nodes.append(node)
                remap[local]=interned[key]
            ends.append(remap[len(group)-1])
        result=ends[-1]
        integer_ops={'poisson','integer_constant','integer_state','integer_parameter','integer_neuron_parameter','integer_mapped_parameter',
                     'integer_parameter_gather','integer_cast','integer_binary','integer_select','integer_neg','integer_sequence','_deferred_integer_parameter','_sample_index'}
        def sequence(left,right):
            if nodes[right]['op'] in integer_ops:nodes.append(dict(op='integer_sequence',left=left,right=right))
            else:nodes.append(dict(op='sequence',left=left,right=right,boolean=nodes[right]['op'] in
                                   {'boolean_cast','boolean_and','boolean_or','boolean_not','eager_boolean_and','eager_boolean_or','integer_compare'}))
            return len(nodes)-1
        lazy={'poisson','select','integer_select','boolean_and','boolean_or'}
        if any(node['op'] in lazy for node in nodes):
            for left in reversed(ends[:-1]):
                result=sequence(left,result)
        elif result!=len(nodes)-1:
            # CPU, Metal and CUDA evaluate every node of an entirely eager
            # program in insertion order. Keep discarded nodes in the program;
            # only relocate its returned value to the last node. No eager root
            # is erased or moved across another potentially failing operation.
            sequence(len(nodes)-1,result)
        if len(nodes)>128:raise ValueError('state effect program exceeds128 native nodes')
        return nodes
    winners={states[name]:name for name in sorted(explicit)}
    targets=sorted(written);programs=[];unconditional_programs={}
    full_names=set(unconditional_states)|set(unconditional_parameters)
    singleton_names={name for name,group in vectors.items() if len(group)==1}
    selected_names=((set(array_states)|set(array_parameters))-full_names)|{'__selected_capture_output'}
    def dependencies(expression):
        names=set()
        class Dependencies(ast.NodeVisitor):
            def visit(self,node):
                if ast.dump(node,include_attributes=False) in engine.whole_selection_keys:
                    names.add('__whole_event_input');return
                if ast.dump(node,include_attributes=False) in projected_keys|engine.selected_vector_keys:
                    names.add('__selected_capture_output');return
                super().visit(node)
            def visit_Name(self,node):names.add(node.id)
        Dependencies().visit(expression)
        return names
    if full_names:
        for expression in eager_expressions:
            names=dependencies(expression)
            if names&full_names and names&selected_names and not (selected_output is not None and names&full_names<=singleton_names):
                raise ValueError('reset captures cannot mix whole arrays with event-selected arrays')
        full_prefix=[compile_value(expression) for expression in eager_expressions
                     if not dependencies(expression)&selected_names]
        if selected_output is not None:
            # Batch lowering executes whole-array checks or writes separately,
            # using the same input snapshot and presence gate. Selected actions
            # need only their selected eager work and returned row expression.
            # Duplicating whole roots here can overflow the fixed program ABI.
            prefix=[compile_value(ast.IfExp(test=ast.Name(id=eager_guard,ctx=ast.Load()),body=row_expression(expression),orelse=ast.Constant(0.))
                if eager_guard is not None and any(isinstance(node,ast.Name) and node.id in vector_names for node in ast.walk(expression))
                else row_expression(expression)) for expression in eager_expressions if dependencies(expression)&selected_names]
    for slot in ([] if empty_vector else targets):
        if physical is not None and slot in effect_slots:
            value=engine.materialize(cells[physical[slot]])
        elif physical is not None:
            value=engine.materialize(engine.environment[winners[slot]])
        else:
            value=ast.parse(outputs[winners[slot]],mode='eval').body if slot in winners else engine.materialize(cells[slot])
        from .training_equations import coerce_state_expression
        expression=engine.physical_cast(value,types.get(slot,'float')) if types else value
        if slot in {states[name] for name in unconditional_states}:
            if dependencies(value)&selected_names:
                raise ValueError('whole reset captures cannot depend on event-selected arrays')
            unconditional_programs[slot]=joined(expression,roots=[] if separate_whole_eager else full_prefix)
        else:
            if dependencies(value)&full_names and not (selected_output is not None and dependencies(value)&full_names<=singleton_names):
                raise ValueError('reset output cannot broadcast a whole capture into an event selection')
            name=next((name for name,gate in selected_guards.items() if states[name]==slot),None)
            expression=row_expression(expression)
            if name is not None:
                expression=ast.IfExp(test=ast.Name(id=selected_guards[name],ctx=ast.Load()),body=expression,
                                     orelse=ast.Name(id=name,ctx=ast.Load()))
            programs.append(joined(expression,predicate_outputs.get(slot)))
    scalar_programs=[]
    if scalar_eager:
        for expression in engine.executed:
            if not any(isinstance(node,ast.Name) and node.id in vector_names for node in ast.walk(expression)):
                # A detached floating scratch cell runs the operation solely
                # for its checked domain/eager effects, including int roots.
                scalar_programs.append(compile_value(ast.Call(func=ast.Name(id='_b2_float',ctx=ast.Load()),
                                                               args=[expression],keywords=[])))
    aliases={name:(engine.environment[name].origin if isinstance(engine.environment[name],ArrayCell) else None)
             for name in explicit}
    arrays={name for name in explicit if isinstance(engine.environment[name],ArrayCell) and not engine.environment[name].zero_dim}
    return dict(writes=[] if empty_vector else [slot for slot in targets if slot not in unconditional_programs],programs=programs,context_size=max(states.values())+1,
                unconditional_programs=unconditional_programs,
                unconditional_eager_program=joined(ast.Constant(0.),roots=full_prefix) if not separate_whole_eager and (unconditional_parameters or whole_eager) else None,
                unconditional_eager_programs=[joined(ast.Constant(0.),roots=[root]) for root in full_prefix] if separate_whole_eager and full_names else [],
                effect_writes=sorted(effect_slots),scalar_programs=scalar_programs,
                selected_output_lengths=sorted(selected_output_lengths),
                selected_whole_mixed=bool(engine.selected_vector_keys),
                output_aliases=aliases,output_arrays=sorted(arrays),
                output_types={name:assignment_types.get(name,engine.dtype(engine.environment[name])) for name in explicit})
