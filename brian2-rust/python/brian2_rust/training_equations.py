"""Bounded scalar update expressions compiled to native differentiable SSA.

An expression is the discrete pre-threshold voltage update. Unit conversion and
integration step choice belong to the caller; arbitrary Brian statements and
Python execution are deliberately not accepted by this compiler.
"""
import ast
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimedInput:
    bank: int
    rows: int
    columns: int
    epsilon: float
    k: int
    dimensions: int



@dataclass(frozen=True)
class NeuronParameter:
    """Contiguous parameter slice indexed by the layer-local neuron number."""
    bank: int
    index: int = 0
    dtype: str = "float"


@dataclass(frozen=True)
class _MappedParameter:
    """Canonical parameter read through a fixed neuron-index mapping."""
    bank: int
    mapping: int
    dtype: str = "float"


@dataclass(frozen=True)
class ParameterBank:
    """Callable canonical bank read using a detached int32 SSA selector.

    Bind a name to this descriptor and use ``name(index)`` in a typed dynamic
    expression. Integer banks are frozen; float bank values keep their VJP.
    """
    bank: int
    dtype: str = "float"


@dataclass(frozen=True)
class _DeferredParameter:
    """Private Brian reference resolved after canonical state allocation."""
    bank: int
    reference: int
    dtype: str = "float"


@dataclass(frozen=True)
class SimulationTime:
    """Read the current tick's SI time from a scalar or vector equation clock."""


@dataclass(frozen=True)
class ClockTime:
    """Read a v5 clock at the current neuron tick; clock scheduling is detached."""
    index: int


@dataclass(frozen=True)
class _TimedState:
    """Explicit external physical-field read at the consumer's Brian clock."""
    source: TimedInput
    clock: int
    column: int | _DeferredParameter
    dtype: str = 'float'
    sample_slot: int | None = None
    physical_columns: int | None = None


@dataclass(frozen=True)
class IntegerParameter:
    """A frozen int32 parameter slot in the dynamic ABI."""
    bank: int
    index: int


@dataclass(frozen=True)
class BooleanParameter:
    bank: int
    index: int


@dataclass(frozen=True)
class StateSlot:
    """A v5 action-context slot (including synaptic or referenced neuron state)."""
    index: int
    dtype: str = "float"


@dataclass(frozen=True)
class RefractoryActive:
    """Detached activity computed from a bounded refractory counter slot."""
    index: int


@dataclass(frozen=True)
class NormalNoise:
    """A standard normal draw shared by uses of a declared per-neuron stream."""
    stream: int


@dataclass(frozen=True)
class UniformNoise:
    """A native uniform [0, 1) draw at a declared per-entity call site."""
    stream: int


@dataclass(frozen=True)
class PoissonNoise:
    """Callable native Poisson stream for scalar v3, vector v4 and dynamic v5.

    Bind a name and call name(rate). Repeated uses of one stream reuse its draw
    and likelihood score; distinct call sites require distinct stream numbers.
    CPU update/reset phases or actions sharing one address cache the first actual count and rate across
    consumers, carry and checkpoint; their likelihood score is applied once.
    Dynamic GPU actions preserve draw records across carry and checkpoint through a
    backend-specific f32 profile and device count validation.
    Integer sample paths are
    detached. Positive continuous rates receive the
    stopped per-sample loss times d log P/d rate in the native reverse pass.
    CPU and dynamic GPU zero-rate derivatives use full one-count counterfactual replay.
    Scalar/vector Metal execution uses the same native persistent-draw and weak
    replay contracts; final phase validation is recorded separately.
    Actual Metal acceptance is recorded; CUDA still needs NVIDIA runtime tests.
    """
    stream: int


@dataclass(frozen=True)
class PureFunction:
    """A bounded pure scalar function inlined into native SSA and its VJP.

    Arguments are evaluated once at the call site. The body has its own lexical
    namespace; closure constants/functions must be supplied explicitly in
    ``parameters``. Bodies accept eager arithmetic and standard scalar math.
    ``statements`` binds ordered (local_name, expression) pairs before the
    return expression. Discarded values still execute without receiving VJPs.
    Stateful draws, conditional expressions and arbitrary Python are refused.
    """
    arguments: tuple
    expression: str
    parameters: tuple = ()
    statements: tuple = ()
    augmented: tuple = ()
    scalarize_arguments: tuple = ()
    array_return: bool = False
    captured_arrays: tuple = ()


def compile_training_equation(expression, *, parameters=None, states=None, state_types=None):
    """Compile v_next=f(v, parameters); parameter values are constants or (bank,id).

    Supports arithmetic, literal powers and standard Brian scalar math. A (bank,id)
    reference shares that native optimizer slot with every other use.
    """
    if not isinstance(expression,str) or len(expression)>8192:
        raise ValueError('training equation must be a bounded expression')
    try:root=ast.parse(expression,mode='eval').body
    except (SyntaxError,RecursionError):raise ValueError('invalid or excessively nested equation') from None
    return _compile_training_ast(root,parameters=parameters,states=states,state_types=state_types)


def compile_training_predicate(expression, *, parameters, states, slope=5., scale=1., state_types=None):
    """Boolean threshold with ordered, short-circuit surrogate gates.

    Ordered comparisons differentiate their SI margin. Equality is detached.
    A skipped Boolean operand acts as its operator's neutral element in VJP.
    """
    if not isinstance(expression,str) or len(expression)>8192:
        raise ValueError('threshold must be a bounded expression')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (slope,scale)):
        raise ValueError('predicate surrogate coefficients must be finite and positive')
    try:root=ast.parse(expression,mode='eval').body
    except (SyntaxError,RecursionError):raise ValueError('invalid threshold expression') from None
    return _compile_training_ast(root,parameters=parameters,states=states,predicate_surrogate=(slope,scale),state_types=state_types)


def _compile_training_ast(root, *, parameters=None, states=None, deduplicate=False, refractory_index=None, allow_select=False, predicate_surrogate=None, state_types=None, typed=False):
    """Compile an immutable expression DAG, retaining shared RK stage nodes."""
    parameters={} if parameters is None else dict(parameters)
    states=[] if states is None else list(states)
    if states and (states[0]!='v' or len(states)>16 or len(set(states))!=len(states)
                   or any(not isinstance(n,str) or not n.isidentifier() for n in states)):
        raise ValueError('states require 1..16 unique names with v first')
    if set(parameters)&(set(states)|{'v'}):raise ValueError('state names are reserved')
    types={} if state_types is None else dict(zip(states,state_types))
    typed=typed or state_types is not None
    if any(t not in ('float','integer','boolean') for t in types.values()):raise ValueError('invalid state type')
    nodes=[];memo={};emitted={};inline_arguments=None;inline_stack=[]
    def emit(op,**fields):
        key=(op,*((name,value.hex() if isinstance(value,float) else value) for name,value in fields.items()))
        if deduplicate and key in emitted:return emitted[key]
        if len(nodes)>=128:raise ValueError('training equation exceeds 128 nodes')
        nodes.append(dict(op=op,**fields));emitted[key]=len(nodes)-1;return len(nodes)-1
    integer_ops={'poisson','integer_constant','integer_state','integer_parameter','integer_neuron_parameter','integer_mapped_parameter','integer_parameter_gather','_deferred_integer_parameter','integer_cast','integer_binary','integer_select','integer_neg'}
    integer_ops.update(('integer_sequence','_sample_index'))
    def integer(index):return nodes[index]['op'] in integer_ops
    def boolean(index):
        node=nodes[index]
        return (node['op'] in {'eager_boolean_and','eager_boolean_or','surrogate_step','boolean_and','boolean_or','boolean_not',
            'equality','integer_compare','discrete_compare','elapsed_compare','boolean_cast'}
            or node['op']=='select' and node.get('boolean',False)
            or node['op']=='sequence' and node['boolean']
            or node['op']=='constant' and node['value'] in (0.,1.))
    def sequence(left,right):
        if integer(right):return emit('integer_sequence',left=left,right=right)
        return emit('sequence',left=left,right=right,boolean=boolean(right))
    def truth(index):return index if boolean(index) else emit('boolean_cast',arg=floating(index))
    def select(condition,yes,no,boolean_output=False):
        if integer(yes) and integer(no):return emit('integer_select',condition=condition,yes=yes,no=no)
        fields={'boolean':True} if boolean_output and boolean(yes) and boolean(no) else {}
        return emit('select',condition=condition,yes=floating(yes),no=floating(no),**fields)
    def floating(index):return emit('integer_float',arg=index) if integer(index) else index
    def binary(op,left,right):
        if op in ('bit_and','bit_or','bit_xor','left_shift','right_shift'):
            if not typed or not integer(left) or not integer(right):
                raise ValueError('bitwise operands must be typed int32')
            return emit('integer_binary',left=left,right=right,kind=op)
        if typed and integer(left) and integer(right) and op in ('add','sub','mul','min','max','floor_div','mod'):
            return emit('integer_binary',left=left,right=right,kind=op)
        return emit('modulo' if op=='mod' else op,left=floating(left),right=floating(right))
    def constant(value):
        if typed and type(value) is bool:return emit('constant',value=float(value))
        if typed and type(value) is int:
            if not -2**31<=value<2**31:raise ValueError('integer literal is outside int32')
            return emit('integer_constant',value=value)
        if type(value) not in (bool,int,float) or not math.isfinite(value):
            raise ValueError('equation constants must be finite real numbers')
        return emit('constant',value=float(value))
    def visit(node):
        key=id(node)
        if key not in memo:memo[key]=lower(node)
        return memo[key]
    def lower(node):
        if predicate_surrogate is not None or typed or inline_arguments is not None:
            if isinstance(node,ast.Constant) and type(node.value) is bool:
                return emit("constant",value=float(node.value))
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):
                arg=visit(node.operand)
                return emit('boolean_not',arg=truth(arg) if inline_arguments is not None else arg)
            if isinstance(node,ast.BoolOp):
                if inline_arguments is not None:
                    result=visit(node.values[0])
                    for child in node.values[1:]:
                        right=visit(child);condition=truth(result)
                        if boolean(result) and boolean(right):
                            result=emit('boolean_and' if isinstance(node.op,ast.And) else 'boolean_or',left=result,right=right)
                        else:
                            result=select(condition,right,result,True) if isinstance(node.op,ast.And) else select(condition,result,right,True)
                    return result
                op='boolean_and' if isinstance(node.op,ast.And) else 'boolean_or'
                result=visit(node.values[0])
                for child in node.values[1:]:result=emit(op,left=result,right=visit(child))
                return result
            if isinstance(node,ast.Compare):
                left=visit(node.left);result=None
                for op,child in zip(node.ops,node.comparators):
                    right=visit(child)
                    kind={ast.Gt:'gt',ast.GtE:'ge',ast.Lt:'lt',ast.LtE:'le',ast.Eq:'eq',ast.NotEq:'ne'}.get(type(op))
                    if kind is None:raise ValueError('unsupported comparison')
                    if typed and integer(left) and integer(right):
                        current=emit('integer_compare',left=left,right=right,kind=kind)
                    elif predicate_surrogate is None:
                        current=emit('discrete_compare',left=floating(left),right=floating(right),kind=kind)
                    elif isinstance(op,(ast.Eq,ast.NotEq)):
                        current=emit('equality',left=floating(left),right=floating(right),unequal=isinstance(op,ast.NotEq))
                    elif isinstance(op,(ast.Gt,ast.GtE,ast.Lt,ast.LtE)):
                        a,b=(left,right) if isinstance(op,(ast.Gt,ast.GtE)) else (right,left)
                        margin=binary('sub',a,b)
                        current=emit('surrogate_step',arg=margin,slope=constant(float(predicate_surrogate[0])),
                                     scale=constant(float(predicate_surrogate[1])),inclusive=isinstance(op,(ast.GtE,ast.LtE)))
                    else:raise ValueError('unsupported threshold comparison')
                    result=current if result is None else emit('boolean_and',left=result,right=current)
                    left=right
                return result
        if (allow_select or inline_arguments is not None) and isinstance(node,ast.IfExp):
            condition=visit(node.test);yes=visit(node.body);no=visit(node.orelse)
            return select(truth(condition) if inline_arguments is not None else condition,yes,no,inline_arguments is not None)
        if isinstance(node,ast.Constant):return constant(node.value)
        if isinstance(node,ast.Name):
            if inline_arguments is not None:
                if node.id in inline_arguments:
                    result=inline_arguments[node.id]
                    if result is None:raise ValueError('pure function local read before assignment: '+node.id)
                    return result
                if node.id in parameters and type(parameters[node.id]) in (bool,int,float):
                    return constant(parameters[node.id])
                raise ValueError('unknown pure function body name')
            if node.id=='_b2_refractory_active' and refractory_index is not None:
                return emit('refractory_active',index=refractory_index)
            if node.id in states:
                index=emit('integer_state' if types.get(node.id)=='integer' else 'state',index=states.index(node.id))
                return emit('boolean_cast',arg=index) if types.get(node.id)=='boolean' else index
            if node.id=='v':return emit('voltage')
            if node.id not in parameters:raise ValueError('unknown equation name: '+node.id)
            value=parameters[node.id]
            if type(value) is RefractoryActive:
                if type(value.index) is not int or not 0<=value.index<64:
                    raise ValueError('refractory activity requires a bounded context slot')
                return emit('refractory_active',index=value.index)
            if isinstance(value,StateSlot):
                if not states or type(value.index) is not int or not 0<=value.index<64:
                    raise ValueError('state slot requires a bounded v5 action context')
                index=emit('integer_state' if value.dtype=='integer' else 'state',index=value.index)
                return emit('boolean_cast',arg=index) if value.dtype=='boolean' else index
            if isinstance(value,(NormalNoise,UniformNoise)):
                if type(value.stream) is not int or not 0<=value.stream<16:
                    raise ValueError('noise requires stream 0..15')
                return emit('uniform_noise' if isinstance(value,UniformNoise) else 'noise',stream=value.stream)
            if isinstance(value,SimulationTime):
                return emit('time')
            if isinstance(value,ClockTime):
                if not allow_select or type(value.index) is not int or not 0<=value.index<256:
                    raise ValueError('clock time requires a bounded v5 clock index')
                return emit('clock_time',index=value.index)
            if isinstance(value,_TimedState):
                source=value.source
                deferred=isinstance(value.column,_DeferredParameter)
                if (not allow_select or not isinstance(source,TimedInput) or value.dtype not in ('float','integer','boolean')
                        or type(value.clock) is not int or not 0<=value.clock<256
                        or not (deferred and typed and states and value.column.dtype=='integer'
                                and all(type(i) is int and i>=0 for i in (value.column.bank,value.column.reference))
                                or type(value.column) is int and 0<=value.column<source.columns)):
                    raise ValueError('external state requires a valid timed source, clock and column')
                time=emit('clock_time',index=value.clock)
                if deferred:
                    index=emit('_deferred_integer_parameter',bank=value.column.bank,reference=value.column.reference)
                    index=emit('integer_float',arg=index)
                else:index=emit('constant',value=float(value.column))
                if value.sample_slot is not None:
                    if (not typed or not states or type(value.sample_slot) is not int or value.sample_slot<0
                            or type(value.physical_columns) is not int or value.physical_columns<=0):
                        raise ValueError('per-sample source requires its native sample identity and physical columns')
                    sample=emit('_sample_index',index=value.sample_slot)
                    sample=emit('integer_float',arg=sample)
                    offset=emit('mul',left=sample,right=emit('constant',value=float(value.physical_columns)))
                    index=emit('add',left=index,right=offset)
                def timed(column):
                    return emit('timed_parameter',bank=source.bank,rows=source.rows,columns=source.columns,
                                epsilon=source.epsilon,k=source.k,time=time,index=column)
                if value.dtype=='integer':
                    high_column=emit('mul',left=index,right=emit('constant',value=2.))
                    low_column=emit('add',left=high_column,right=emit('constant',value=1.))
                    high=emit('integer_cast',arg=timed(high_column))
                    low=emit('integer_cast',arg=timed(low_column))
                    high=emit('integer_binary',left=high,right=emit('integer_constant',value=16),kind='left_shift')
                    return emit('integer_binary',left=high,right=low,kind='bit_or')
                result=timed(index)
                return emit('boolean_cast',arg=result) if value.dtype=='boolean' else result
            if isinstance(value,NeuronParameter):
                if (not states and value.dtype!='float') or any(type(i) is not int or i<0 for i in (value.bank,value.index)):
                    raise ValueError('neuron parameter requires valid bank/index and a compatible dtype')
                index=emit('integer_neuron_parameter' if value.dtype=='integer' else 'neuron_parameter',bank=value.bank,index=value.index)
                return emit('boolean_cast',arg=index) if value.dtype=='boolean' else index
            if isinstance(value,_MappedParameter):
                if not states:raise ValueError('mapped parameters require state conversion')
                index=emit('integer_mapped_parameter' if value.dtype=='integer' else 'mapped_parameter',bank=value.bank,mapping=value.mapping)
                return emit('boolean_cast',arg=index) if value.dtype=='boolean' else index
            if isinstance(value,_DeferredParameter):
                if not typed or not states:raise ValueError('deferred Brian parameters require typed state conversion')
                index=emit('_deferred_integer_parameter' if value.dtype=='integer' else '_deferred_parameter',bank=value.bank,reference=value.reference)
                return emit('boolean_cast',arg=index) if value.dtype=='boolean' else index
            if isinstance(value,BooleanParameter):return emit('boolean_cast',arg=emit('parameter',bank=value.bank,index=value.index))
            if isinstance(value,IntegerParameter):return emit('integer_parameter',bank=value.bank,index=value.index)
            if isinstance(value,(tuple,list)) and len(value)==2:
                if any(type(i) is not int or i<0 for i in value):raise ValueError('invalid parameter reference')
                return emit('parameter',bank=value[0],index=value[1])
            return constant(value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Invert):
            return binary('bit_xor',visit(node.operand),constant(-1))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            if typed and isinstance(node.op,ast.USub) and isinstance(node.operand,ast.Constant) and type(node.operand.value) is int:
                return constant(-node.operand.value)
            arg=visit(node.operand)
            return emit('integer_neg' if integer(arg) else 'neg',arg=arg) if isinstance(node.op,ast.USub) else arg
        if isinstance(node,ast.BinOp):
            if isinstance(node.op,ast.Pow):
                try:value=ast.literal_eval(node.right)
                except (ValueError,TypeError):raise ValueError('equation powers require a literal exponent') from None
                if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('invalid exponent')
                return emit('pow',arg=floating(visit(node.left)),value=float(value))
            op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'div',ast.FloorDiv:'floor_div',ast.Mod:'mod',
                ast.BitAnd:'bit_and',ast.BitOr:'bit_or',ast.BitXor:'bit_xor',ast.LShift:'left_shift',ast.RShift:'right_shift'}.get(type(node.op))
            if op:
                left=visit(node.left);right=visit(node.right)
                if op=='mod' and inline_arguments is None:
                    # Match Brian's Cython renderer, including its intermediate
                    # int32 wrap and floating cancellation at boundaries.
                    return binary('mod',binary('add',binary('mod',left,right),right),right)
                return binary(op,left,right)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and isinstance(parameters.get(node.func.id),PureFunction):
            return inline(parameters[node.func.id],node)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and isinstance(parameters.get(node.func.id),PoissonNoise):
            source=parameters[node.func.id]
            if node.keywords or len(node.args)!=1 or type(source.stream) is not int or not 0<=source.stream<16:
                raise ValueError('Poisson stream requires one rate and stream 0..15')
            return emit('poisson',rate=floating(visit(node.args[0])),stream=source.stream)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and isinstance(parameters.get(node.func.id),ParameterBank):
            source=parameters[node.func.id]
            if not typed or not states or node.keywords or len(node.args)!=1 or type(source.bank) is not int or source.bank<0 or source.dtype not in ('float','integer','boolean'):
                raise ValueError('parameter bank gather requires a typed dynamic expression and one index')
            index=visit(node.args[0])
            if not integer(index):raise ValueError('parameter bank selector must be int32')
            value=emit('integer_parameter_gather' if source.dtype=='integer' else 'parameter_gather',bank=source.bank,index=index)
            return emit('boolean_cast',arg=value) if source.dtype=='boolean' else value
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and isinstance(parameters.get(node.func.id),TimedInput):
            source=parameters[node.func.id]
            if node.keywords or len(node.args)!=source.dimensions:
                raise ValueError('TimedArray call has invalid arguments')
            time=floating(visit(node.args[0]));index=floating(visit(node.args[1])) if source.dimensions==2 else emit('constant',value=0.)
            return emit('timed_parameter',bank=source.bank,rows=source.rows,columns=source.columns,
                        epsilon=source.epsilon,k=source.k,time=time,index=index)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            if node.func.id=='_b2_owned_array' and len(node.args)==1:
                return binary('mul',visit(node.args[0]),emit('constant',value=1.))
            if node.func.id=='_b2_where' and len(node.args)==3:
                condition=visit(node.args[0]);yes=visit(node.args[1]);no=visit(node.args[2])
                result=select(truth(condition),yes,no,True)
                # NumPy evaluates both value arguments before selecting. Preserve
                # their eager execution even though Select is otherwise lazy.
                for value in (no,yes,condition):result=sequence(value,result)
                return result
            if node.func.id in ('_b2_logical_and','_b2_logical_or') and len(node.args)==2:
                left=visit(node.args[0]);right=visit(node.args[1])
                result=emit('eager_boolean_and' if node.func.id=='_b2_logical_and' else 'eager_boolean_or',left=truth(left),right=truth(right))
                return sequence(left,sequence(right,result))
            if node.func.id=='_b2_logical_not' and len(node.args)==1:
                return emit('boolean_not',arg=truth(visit(node.args[0])))
            if node.func.id=='_b2_mod' and len(node.args)==2:
                return binary('mod',visit(node.args[0]),visit(node.args[1]))
            if typed and node.func.id in ('int','bool','_b2_float') and len(node.args)==1:
                arg=visit(node.args[0])
                if node.func.id=='int':return arg if integer(arg) else emit('integer_cast',arg=arg)
                if node.func.id=='_b2_float':return floating(arg)
                return emit('boolean_cast',arg=floating(arg))
            if typed and node.func.id=='floor' and len(node.args)==1:
                return binary('floor_div',floating(visit(node.args[0])),emit('constant',value=1.))
            if node.func.id=='clip' and len(node.args)==3:
                value,lo,hi=map(visit,node.args)
                return binary('min',binary('max',value,lo),hi)
            if node.func.id in ('minimum','maximum') and len(node.args)==2:
                return binary('min' if node.func.id=='minimum' else 'max',visit(node.args[0]),visit(node.args[1]))
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and
                node.func.id in ('exp','log','tanh','sqrt','sin','cos') and
                len(node.args)==1 and not node.keywords):
            return emit(node.func.id,arg=floating(visit(node.args[0])))
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and
                node.func.id in ('tan','cosh','sinh','log10','expm1','log1p','exprel','arccos','arcsin','arctan','ceil','abs','sign','floor') and
                len(node.args)==1 and not node.keywords):
            arg=visit(node.args[0]);kind=node.func.id
            if integer(arg) and kind=='abs':
                return binary('max',arg,emit('integer_neg',arg=arg))
            if integer(arg) and kind=='sign':
                zero=emit('integer_constant',value=0)
                positive=emit('integer_compare',left=arg,right=zero,kind='gt')
                negative=emit('integer_compare',left=arg,right=zero,kind='lt')
                return emit('integer_cast',arg=emit('sub',left=positive,right=negative))
            return emit('math',kind=kind,arg=floating(arg))
        raise ValueError('unsupported training equation expression')
    def inline(function, call):
        nonlocal parameters, states, types, memo, inline_arguments
        arguments=function.arguments
        if (type(arguments) is not tuple or not 0<=len(arguments)<=16 or
                any(type(name) is not str or not name.isidentifier() for name in arguments) or
                len(set(arguments))!=len(arguments) or
                call.keywords or len(call.args)!=len(arguments)):
            raise ValueError('pure function requires 0..16 distinct positional arguments and exact arity')
        if (type(function.scalarize_arguments) is not tuple
                or function.scalarize_arguments and len(function.scalarize_arguments)!=len(arguments)
                or any(type(flag) is not bool for flag in function.scalarize_arguments)
                or type(function.array_return) is not bool):
            raise ValueError('invalid pure function array contract')
        # Compile actual arguments in the caller scope before entering the body:
        # f(f(v)) is nesting, whereas a body calling itself is recursion.
        actual=[visit(arg) for arg in call.args]
        if id(function) in inline_stack or len(inline_stack)>=8:
            raise ValueError('recursive or excessively nested pure function')
        if type(function.expression) is not str or len(function.expression)>8192:
            raise ValueError('pure function requires a bounded expression')
        try:body=ast.parse(function.expression,mode='eval').body
        except (SyntaxError,RecursionError):raise ValueError('invalid pure function expression') from None
        if type(function.parameters) is not tuple or len(function.parameters)>64:
            raise ValueError('pure function closure requires a bounded tuple')
        try:closure=dict(function.parameters)
        except (TypeError,ValueError):raise ValueError('invalid pure function closure') from None
        if (len(closure)!=len(function.parameters) or set(closure)&set(arguments) or
                any(type(name) is not str or not name.isidentifier() for name in closure) or
                any(not isinstance(value,PureFunction) and
                    (type(value) not in (bool,int,float) or not math.isfinite(value)) for value in closure.values())):
            raise ValueError('pure function closure requires finite constants or pure functions')
        if (type(function.captured_arrays) is not tuple or len(function.captured_arrays)>64
                or any(type(name) is not str or name not in closure or isinstance(closure[name],PureFunction) for name in function.captured_arrays)
                or len(set(function.captured_arrays))!=len(function.captured_arrays)):
            raise ValueError('invalid pure function captured array metadata')
        if type(function.statements) is not tuple or len(function.statements)>64:
            raise ValueError('pure function requires at most 64 local assignments')
        if (type(function.augmented) is not tuple
                or any(type(i) is not int or not 0<=i<len(function.statements) for i in function.augmented)
                or len(set(function.augmented))!=len(function.augmented)):
            raise ValueError('invalid pure function augmented assignment metadata')
        statements=[];locals=set()
        for statement in function.statements:
            if (type(statement) is not tuple or len(statement)!=2 or
                    type(statement[0]) is not str or not statement[0].isidentifier() or
                    type(statement[1]) is not str or len(statement[1])>8192):
                raise ValueError('invalid pure function local assignment')
            try:tree=ast.parse(statement[1],mode='eval').body
            except (SyntaxError,RecursionError):raise ValueError('invalid pure function local expression') from None
            statements.append((statement[0],tree));locals.add(statement[0])
        trees=[tree for _,tree in statements]+[body]
        used={n.id for tree in trees for n in ast.walk(tree) if isinstance(n,ast.Name)}
        if any(isinstance(n,(ast.NamedExpr,ast.Attribute,
                             ast.Lambda,ast.ListComp,ast.GeneratorExp)) for tree in trees for n in ast.walk(tree)):
            raise ValueError('pure function body requires bounded scalar expressions')
        if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in set(arguments)|locals
               for tree in trees for n in ast.walk(tree)):
            raise ValueError('pure function arguments cannot be callable')
        old=(parameters,states,types,memo,inline_arguments)
        parameters=closure;states=[];types={};memo={};inline_arguments={name:None for name in locals}
        inline_arguments.update(zip(arguments,actual))
        inline_stack.append(id(function))
        try:
            # No implicit v/t/namespace capture is allowed inside a body.
            builtins={'exp','log','tanh','sqrt','sin','cos','clip','minimum','maximum',
                      'tan','cosh','sinh','log10','expm1','log1p','exprel','arccos',
                      'arcsin','arctan','ceil','abs','sign','floor','int','bool','_b2_float','_b2_mod',
                      '_b2_where','_b2_logical_and','_b2_logical_or','_b2_logical_not','_b2_owned_array'}
            if used-set(arguments)-locals-set(closure)-builtins:
                raise ValueError('unknown pure function body name')
            eager=list(actual)
            for position,(name,tree) in enumerate(statements):
                previous=inline_arguments[name]
                value=visit(tree)
                if position in function.augmented and (previous is None or integer(previous)!=integer(value)):
                    raise ValueError('pure function augmented assignment changes array dtype')
                inline_arguments[name]=value;eager.append(value)
            result=visit(body)
            # Sequencing retains primal execution without a derivative through
            # discarded values. In lazy graphs it also preserves source order.
            for value in reversed(eager):result=sequence(value,result)
            return result
        finally:
            inline_stack.pop();parameters,states,types,memo,inline_arguments=old
    try:root=visit(root)
    except RecursionError:raise ValueError('invalid or excessively nested equation') from None
    # Unary plus may return an existing root, which is already the last node.
    return nodes[:root+1]


def neuron_parameter_bank(count, *, target_layer=1):
    """A v3 parameter bank with no synaptic edges, for equation coefficients."""
    if type(count) is not int or count<1:raise ValueError('parameter count must be positive')
    return dict(source_layer=0,target_layer=target_layer,parameter_count=count,
                sources=[],targets=[],parameter_ids=[])


def coerce_state_expression(expression, dtype):
    """Materialize Brian assignment conversion before the next statement."""
    function={'integer':'int','boolean':'bool','float':'_b2_float'}[dtype]
    return ast.Call(func=ast.Name(id=function,ctx=ast.Load()),args=[expression],keywords=[])


def typed_parameter(bank, index, dtype):
    return IntegerParameter(bank,index) if dtype=='integer' else BooleanParameter(bank,index) if dtype=='boolean' else (bank,index)
