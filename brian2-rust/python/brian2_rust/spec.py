"""Backend-neutral code-object subset, before target-language rendering."""

import ast
import copy
import hashlib
import inspect
import json
import math
import struct
import textwrap
from dataclasses import asdict, dataclass

import numpy as np
from brian2.codegen.translation import make_statements
from brian2.core.variables import AuxiliaryVariable
from brian2.input.timedarray import TimedArray, _find_K
from brian2.parsing.bast import brian_ast
from brian2.units.fundamentalunits import get_dimensions

from .resource_limits import timed_array_value_budget


UNARY_FUNCTIONS = frozenset({
    "abs", "arccos", "arcsin", "arctan", "ceil", "cos", "cosh", "exp",
    "expm1", "exprel", "floor", "log", "log10", "log1p", "sign", "sin",
    "sinh", "sqrt", "tan", "tanh",
})
SUPPORTED_FUNCTIONS = UNARY_FUNCTIONS | {
    "clip", "int", "poisson", "rand", "randn", "timestep",
}
DIMENSIONLESS = (0.0,) * 7
TIME_DIMENSIONS = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
INTEGER_DTYPES = {"i32", "i64", "u32", "u64"}


def dtype_name(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        return "f64"
    if dtype == np.dtype(np.float32):
        return "f32"
    if dtype == np.dtype(np.bool_):
        return "bool"
    if dtype == np.dtype(np.int32):
        return "i32"
    if dtype == np.dtype(np.int64):
        return "i64"
    if dtype == np.dtype(np.uint32):
        return "u32"
    if dtype == np.dtype(np.uint64):
        return "u64"
    raise NotImplementedError(f"Atlas capability: unsupported dtype {dtype}")


def _decode_bits(value):
    return struct.unpack(">d", bytes.fromhex(value))[0]


def _same_dimensions(left, right):
    return all(abs(a - b) <= 1e-12 for a, b in zip(left, right, strict=True))


def _zero_literal(expr):
    return ((expr["op"] == "literal" and _decode_bits(expr["bits"]) == 0) or
            (expr["op"] == "integer" and int(expr["value"]) == 0) or
            expr["op"] in {"neg", "cast", "f32_to_f64", "f64_to_f32"} and
            _zero_literal(expr["arg"]))


def _constant_literal(expr):
    if expr["op"] == "literal":
        return _decode_bits(expr["bits"])
    if expr["op"] == "integer":
        return int(expr["value"])
    if expr["op"] == "neg":
        value = _constant_literal(expr["arg"])
        return None if value is None else -value
    if expr["op"] == "cast":
        return _constant_literal(expr["arg"])
    return None


def infer_dimensions(expr, symbols, functions=None):
    """Infer SI dimensions from lowered IR, including Brian's polymorphic zero."""
    functions = {} if functions is None else functions
    op = expr["op"]
    if op in {"literal", "integer", "boolean", "rand", "randn", "binomial"}:
        return DIMENSIONLESS
    if op == "load":
        return symbols[expr["name"]]
    if op == "call":
        function = functions[expr["function"]]
        require(len(expr["arguments"]) == len(function["arguments"]),
                f"{expr['function']}: argument count does not match contract")
        for value, argument in zip(expr["arguments"], function["arguments"],
                                   strict=True):
            require(_same_dimensions(infer_dimensions(value, symbols, functions),
                                     tuple(argument["dimensions"])),
                    f"{expr['function']}: argument dimensions do not match contract")
        return tuple(function["return_dimensions"])
    if op in {"index_to_f64", "tick_to_f64"}:
        require(_same_dimensions(infer_dimensions(expr["arg"], symbols, functions),
                                 DIMENSIONLESS),
                f"{op} requires a dimensionless logical value")
        return DIMENSIONLESS
    if op in {"f32_to_f64", "f64_to_f32", "cast"}:
        return infer_dimensions(expr["arg"], symbols, functions)
    if op == "timestep":
        require(all(_same_dimensions(infer_dimensions(expr[name], symbols, functions),
                                     TIME_DIMENSIONS)
                    for name in ("time", "dt")),
                "timestep requires time-valued arguments")
        return DIMENSIONLESS
    if op == "tick_offset":
        require(_same_dimensions(infer_dimensions(expr["tick"], symbols, functions),
                                 DIMENSIONLESS),
                "tick_offset requires a dimensionless tick")
        return DIMENSIONLESS
    if op == "bool_to_f64":
        require(_same_dimensions(infer_dimensions(expr["arg"], symbols, functions),
                                 DIMENSIONLESS),
                "bool_to_f64 requires a dimensionless bool")
        return DIMENSIONLESS
    if op == "poisson":
        require(_same_dimensions(infer_dimensions(expr["lambda"], symbols, functions),
                                 DIMENSIONLESS),
                "poisson lambda must be dimensionless")
        return DIMENSIONLESS
    if op == "timed_array":
        require(_same_dimensions(infer_dimensions(expr["time"], symbols, functions),
                                 TIME_DIMENSIONS),
                "TimedArray time argument must have time dimensions")
        if "index" in expr:
            require(_same_dimensions(infer_dimensions(expr["index"], symbols, functions),
                                     DIMENSIONLESS),
                    "TimedArray index must be dimensionless")
        return tuple(expr["dimensions"])
    if op in {"neg", "abs", "ceil", "floor", "trunc"}:
        return infer_dimensions(expr["arg"], symbols, functions)
    if op in {"not", "sign"}:
        infer_dimensions(expr["arg"], symbols, functions)
        return DIMENSIONLESS
    if op in {"arccos", "arcsin", "arctan", "cos", "cosh", "exp",
              "expm1", "exprel", "log", "log10", "log1p", "sin", "sinh",
              "tan", "tanh"}:
        require(_same_dimensions(infer_dimensions(expr["arg"], symbols, functions),
                                 DIMENSIONLESS),
                f"{op} requires a dimensionless operand")
        return DIMENSIONLESS
    if op == "sqrt":
        return tuple(value * .5 for value in
                     infer_dimensions(expr["arg"], symbols, functions))
    if op in {"add", "sub", "mod"}:
        left = infer_dimensions(expr["left"], symbols, functions)
        right = infer_dimensions(expr["right"], symbols, functions)
        if _zero_literal(expr["left"]):
            return right
        if _zero_literal(expr["right"]):
            return left
        require(_same_dimensions(left, right),
                "add/sub/mod operands must have identical dimensions")
        return left
    if op in {"mul", "div"}:
        left = infer_dimensions(expr["left"], symbols, functions)
        right = infer_dimensions(expr["right"], symbols, functions)
        sign = 1 if op == "mul" else -1
        return tuple(a + sign*b for a, b in zip(left, right, strict=True))
    if op == "pow":
        base = infer_dimensions(expr["left"], symbols, functions)
        require(_same_dimensions(infer_dimensions(expr["right"], symbols, functions),
                                 DIMENSIONLESS),
                "power exponent must be dimensionless")
        if _same_dimensions(base, DIMENSIONLESS):
            return DIMENSIONLESS
        exponent = _constant_literal(expr["right"])
        require(exponent is not None,
                "dimensionful base requires a literal exponent")
        return tuple(value*exponent for value in base)
    if op in {"floor_div", "eq", "ne", "gt", "ge", "lt", "le"}:
        left = infer_dimensions(expr["left"], symbols, functions)
        right = infer_dimensions(expr["right"], symbols, functions)
        require(_same_dimensions(left, right) or _zero_literal(expr["left"])
                or _zero_literal(expr["right"]),
                "comparison/floor_div operands must have identical dimensions")
        return DIMENSIONLESS
    if op in {"and", "or"}:
        require(all(_same_dimensions(infer_dimensions(expr[side], symbols, functions),
                                     DIMENSIONLESS)
                    for side in ("left", "right")),
                "boolean operands must be dimensionless")
        return DIMENSIONLESS
    if op == "clip":
        value = infer_dimensions(expr["value"], symbols, functions)
        for bound in (expr["min"], expr["max"]):
            require(_zero_literal(bound) or
                    _same_dimensions(value, infer_dimensions(bound, symbols, functions)),
                    "clip operands must have identical dimensions")
        return value
    raise AssertionError(f"missing dimension inference for {op}")


def infer_dtype(expr, symbols, functions=None):
    """Infer the exact B2IR dtype used to insert explicit storage casts."""
    functions = {} if functions is None else functions
    op = expr["op"]
    if op in {"literal", "rand", "randn", "binomial", "poisson",
              "bool_to_f64", "index_to_f64", "tick_to_f64", "f32_to_f64"}:
        return "f64"
    if op == "boolean":
        return "bool"
    if op == "integer":
        return expr["dtype"]
    if op == "load":
        return symbols[expr["name"]]
    if op == "call":
        return functions[expr["function"]]["return_dtype"]
    if op == "f64_to_f32":
        return "f32"
    if op == "cast":
        return expr["dtype"]
    if op in {"timestep", "tick_offset"}:
        return "tick"
    if op in {"not", "eq", "ne", "gt", "ge", "lt", "le", "and", "or"}:
        return "bool"
    if op in {"add", "sub", "mul", "mod", "floor_div", "neg", "abs"}:
        if op in {"neg", "abs"}:
            dtype = infer_dtype(expr["arg"], symbols, functions)
            return dtype if dtype in INTEGER_DTYPES else "f64"
        left = infer_dtype(expr["left"], symbols, functions)
        right = infer_dtype(expr["right"], symbols, functions)
        return left if left == right and left in INTEGER_DTYPES else "f64"
    return "f64"


def require(condition, message):
    if not condition:
        raise NotImplementedError(f"Atlas capability: {message}")


def bits(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("B2IR values must be finite")
    return struct.pack(">d", value).hex()


def timed_array_config(variable, clock_dt):
    """Return the canonical, self-contained table payload for a TimedArray."""
    values = np.asarray(variable.values, dtype=np.float64)
    maximum = timed_array_value_budget()
    require(values.ndim in (1, 2) and values.size > 0 and
            values.size <= maximum and np.isfinite(values).all(),
            f"TimedArray needs 1..{maximum:,} finite f64 values")
    factor = _find_K(float(clock_dt), variable.dt)
    return {
        "values": [bits(value) for value in values.ravel(order="C")],
        "rows": int(values.shape[0]),
        "columns": int(values.shape[1]) if values.ndim == 2 else None,
        "epsilon": bits(variable.dt / factor),
        "upsampling": factor,
        "dimensions": [float(dim) for dim in variable.dim._dims],
    }


def expression(source, allocate_random_stream=None, random_functions=None,
               timed_arrays=None, functions=None, logical_indices=None,
               logical_ticks=None, symbol_dtypes=None, integer_dtype=None,
               private_constants=None):
    random_functions = {} if random_functions is None else random_functions
    timed_arrays = {} if timed_arrays is None else timed_arrays
    functions = {} if functions is None else functions
    logical_indices = set() if logical_indices is None else logical_indices
    logical_ticks = set() if logical_ticks is None else logical_ticks
    symbol_dtypes = {} if symbol_dtypes is None else symbol_dtypes
    private_constants = {} if private_constants is None else private_constants
    require(integer_dtype is None or integer_dtype in INTEGER_DTYPES,
            "invalid integer expression dtype")

    def integer(value, dtype=None):
        dtype = dtype or integer_dtype or ("i64" if value < 2**63 else "u64")
        numpy_dtype = {
            "i32": np.int32,
            "i64": np.int64,
            "u32": np.uint32,
            "u64": np.uint64,
        }[dtype]
        info = np.iinfo(numpy_dtype)
        require(info.min <= value <= info.max,
                f"integer literal {value} does not fit {dtype}")
        return {"op": "integer", "dtype": dtype, "value": str(value)}

    def logical_type(value):
        if value["op"] in {"timestep", "tick_offset"}:
            return "tick"
        if value["op"] == "load":
            if value["name"] in logical_indices:
                return "index"
            if value["name"] in logical_ticks:
                return "tick"
            return symbol_dtypes.get(value["name"])
        if value["op"] == "f64_to_f32":
            return "f32"
        if value["op"] == "integer":
            return value["dtype"]
        if value["op"] == "cast":
            return value["dtype"]
        if value["op"] in {"f32_to_f64", "index_to_f64", "tick_to_f64"}:
            return "f64"
        if value["op"] in {"literal", "rand", "randn", "binomial", "poisson",
                           "bool_to_f64"}:
            return "f64"
        if value["op"] == "boolean":
            return "bool"
        try:
            return infer_dtype(value, symbol_dtypes, functions)
        except (KeyError, NotImplementedError):
            return None

    def as_f64(value):
        dtype = logical_type(value)
        if dtype in INTEGER_DTYPES:
            return {"op": "cast", "dtype": "f64", "arg": value}
        return ({"op": f"{dtype}_to_f64", "arg": value}
                if dtype in {"f32", "index", "tick"} else value)

    def cast(value, dtype):
        return value if logical_type(value) == dtype else {
            "op": "cast", "dtype": dtype, "arg": value}

    def integer_pair(left, right):
        left_dtype, right_dtype = logical_type(left), logical_type(right)
        if left.get("op") == "integer" and right_dtype in INTEGER_DTYPES:
            left = integer(int(left["value"]), right_dtype)
            left_dtype = right_dtype
        if right.get("op") == "integer" and left_dtype in INTEGER_DTYPES:
            right = integer(int(right["value"]), left_dtype)
            right_dtype = left_dtype
        if left_dtype in INTEGER_DTYPES and right_dtype in INTEGER_DTYPES:
            if left_dtype != right_dtype:
                require(integer_dtype is not None,
                        "mixed-width integer expression needs a typed assignment target")
                left, right = cast(left, integer_dtype), cast(right, integer_dtype)
            return left, right, logical_type(left)
        return left, right, None

    def integer_literal(node):
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            node = node.operand
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sign * node.value
        return None

    def lower(node):
        if isinstance(node, ast.Constant):
            if type(node.value) is bool:
                return {"op": "boolean", "value": node.value}
            if type(node.value) in (int, float):
                if type(node.value) is int:
                    return integer(node.value)
                return {"op": "literal", "bits": bits(node.value)}
        if isinstance(node, ast.Name):
            if node.id in private_constants:
                return integer(private_constants[node.id],
                               symbol_dtypes.get(node.id))
            return {"op": "load", "name": node.id}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            require(not node.keywords, f"{name}: keyword arguments are unsupported")
            if name in timed_arrays:
                config = timed_arrays[name]
                expected = 1 if config["columns"] is None else 2
                require(len(node.args) == expected,
                        f"{name}: expected {expected} arguments")
                result = {"op": "timed_array", **config,
                          "time": lower(node.args[0])}
                if expected == 2:
                    result["index"] = lower(node.args[1])
                return result
            if name in {"rand", "randn"}:
                require(not node.args, f"{name} requires no arguments")
                require(allocate_random_stream is not None,
                        f"runtime {name} is not enabled for this code object")
                return {"op": name, "stream": allocate_random_stream()}
            if name == "poisson":
                require(len(node.args) == 1, "poisson requires one argument")
                require(allocate_random_stream is not None,
                        "runtime poisson is not enabled for this code object")
                return {"op": "poisson", "stream": allocate_random_stream(),
                        "lambda": lower(node.args[0])}
            if name in random_functions:
                require(not node.args, f"{name} requires no arguments")
                require(allocate_random_stream is not None,
                        "runtime binomial is not enabled for this code object")
                config = random_functions[name]
                probability = (
                    {"op": "load", "name": config["p_name"]}
                    if "p_name" in config else
                    {"op": "literal", "bits": bits(config["p"])}
                )
                return {"op": "binomial", "stream": allocate_random_stream(),
                        "n": config["n"],
                        "p": probability,
                        "approximate": config.get("approximate", True)}
            if name in functions:
                contract = functions[name]
                require(len(node.args) == len(contract["arguments"]),
                        f"{name}: expected {len(contract['arguments'])} arguments")
                return {"op": "call", "function": name,
                        "arguments": [
                            cast(lower(arg), argument["dtype"])
                            for arg, argument in zip(
                                node.args, contract["arguments"], strict=True)
                        ]}
            if name in UNARY_FUNCTIONS:
                require(len(node.args) == 1, f"{name} requires one argument")
                return {"op": name, "arg": as_f64(lower(node.args[0]))}
            if name == "int":
                require(len(node.args) == 1, "int requires one argument")
                arg = lower(node.args[0])
                target = integer_dtype or "i32"
                return cast(arg, target)
            if name == "timestep":
                require(len(node.args) == 2, "timestep requires two arguments")
                return {"op": "timestep", "time": lower(node.args[0]),
                        "dt": lower(node.args[1])}
            if name == "clip":
                require(len(node.args) == 3, "clip requires three arguments")
                return {"op": "clip", "value": as_f64(lower(node.args[0])),
                        "min": as_f64(lower(node.args[1])),
                        "max": as_f64(lower(node.args[2]))}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return {"op": "not", "arg": lower(node.operand)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            arg = lower(node.operand)
            require(not (isinstance(node.op, ast.USub) and
                         logical_type(arg) in {"u32", "u64"}),
                    "unary minus requires a signed integer")
            if isinstance(node.op, ast.USub):
                if logical_type(arg) in {"f32", "index", "tick"}:
                    arg = as_f64(arg)
                return {"op": "neg", "arg": arg}
            return arg
        operators = {
            ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div",
            ast.Pow: "pow", ast.Mod: "mod", ast.FloorDiv: "floor_div",
        }
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left, right = lower(node.left), lower(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                right_literal = integer_literal(node.right)
                left_literal = integer_literal(node.left)
                if logical_type(left) == "tick" and right_literal is not None:
                    offset = right_literal * (-1 if isinstance(node.op, ast.Sub) else 1)
                    require(-(2**53) <= offset <= 2**53,
                            "tick offset exceeds exact execution range")
                    return {"op": "tick_offset", "tick": left,
                            "offset": offset}
                if (isinstance(node.op, ast.Add) and
                        logical_type(right) == "tick" and left_literal is not None):
                    require(-(2**53) <= left_literal <= 2**53,
                            "tick offset exceeds exact execution range")
                    return {"op": "tick_offset", "tick": right,
                            "offset": left_literal}
            left, right, pair_dtype = integer_pair(left, right)
            if pair_dtype is not None and not isinstance(node.op, (ast.Div, ast.Pow)):
                return {"op": operators[type(node.op)],
                        "left": left, "right": right}
            return {"op": operators[type(node.op)],
                    "left": as_f64(left), "right": as_f64(right)}
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            op = "and" if isinstance(node.op, ast.And) else "or"
            values = [lower(value) for value in node.values]
            result = values[0]
            for value in values[1:]:
                result = {"op": op, "left": result, "right": value}
            return result
        comparisons = {
            ast.Eq: "eq", ast.NotEq: "ne", ast.Gt: "gt", ast.GtE: "ge",
            ast.Lt: "lt", ast.LtE: "le",
        }
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and type(node.ops[0]) in comparisons):
            left, right = lower(node.left), lower(node.comparators[0])
            left, right, _ = integer_pair(left, right)
            if logical_type(left) != logical_type(right):
                left, right = as_f64(left), as_f64(right)
            return {"op": comparisons[type(node.ops[0])], "left": left,
                    "right": right}
        raise NotImplementedError(f"Atlas expression: {ast.dump(node)}")
    return lower(ast.parse(source, mode="eval").body)


def portable_function_contract(name, function):
    """Build a deterministic B2IR contract for a Brian ``Function``.

    A restricted Python expression is the portable implementation.  An
    explicitly registered ``b2ir-c-abi-v1`` implementation can additionally
    provide a content-addressed CPU implementation for AOT.  The latter is
    never inferred from Python and is compiled as a separate C translation
    unit, so its language boundary is a versioned C ABI rather than Rust's
    unstable native ABI.
    """
    # Brian conservatively wraps ordinary @check_units callables with
    # stateless=False. Do not trust that hint in either direction: the
    # restricted source grammar below proves purity structurally.
    require(not function.auto_vectorise,
            f"{name}: auto-vectorised functions are not portable")
    pyfunc = function.pyfunc
    require(pyfunc is not None, f"{name}: portable function needs a Python definition")
    signature = inspect.signature(pyfunc)
    parameters = list(signature.parameters.values())
    require(all(parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                   inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and parameter.default is inspect.Parameter.empty
                for parameter in parameters),
            f"{name}: only required positional arguments are portable")
    argument_names = [parameter.name for parameter in parameters]
    require(len(function._arg_units) == len(argument_names) and
            (function._return_unit is bool or
             not callable(function._return_unit)) and
            all(not callable(unit) for unit in function._arg_units),
            f"{name}: dynamic unit mappings are not portable")
    arg_types = list(function._arg_types)
    effective_return_type = (
        "boolean" if function._return_unit is bool else function._return_type)
    require(len(arg_types) == len(argument_names) and
            all(dtype in {"any", "float", "integer", "boolean"}
                for dtype in arg_types) and
            effective_return_type in {"float", "integer", "boolean"},
            f"{name}: portable profile supports f64/i64/bool arguments and return")
    def dimensions(unit):
        if unit is bool:
            return list(DIMENSIONLESS)
        dim = get_dimensions(unit)
        return [float(value) for value in dim._dims]

    dtype_for_type = {
        "any": "f64", "float": "f64", "integer": "i64",
        "boolean": "bool",
    }
    arguments = [
        {"name": argument_name, "dtype": dtype_for_type[arg_type],
         "dimensions": dimensions(unit)}
        for argument_name, arg_type, unit in zip(
            argument_names, arg_types, function._arg_units, strict=True)
    ]
    return_dtype = dtype_for_type[effective_return_type]
    return_dimensions = dimensions(function._return_unit)
    lowered = None
    portable_error = None
    try:
        module = ast.parse(textwrap.dedent(inspect.getsource(pyfunc)))
        definitions = [
            node for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        require(len(definitions) == 1 and
                isinstance(definitions[0], ast.FunctionDef),
                f"{name}: expected one synchronous function definition")
        definition = definitions[0]
        require(not definition.args.vararg and not definition.args.kwarg and
                not definition.args.kwonlyargs and
                not definition.args.defaults and
                not definition.args.kw_defaults,
                f"{name}: variadic/default arguments are not portable")
        require([argument.arg for argument in definition.args.args] ==
                argument_names,
                f"{name}: inspected signature does not match source")
        body = list(definition.body)
        if (body and isinstance(body[0], ast.Expr) and
                isinstance(body[0].value, ast.Constant) and
                isinstance(body[0].value.value, str)):
            body.pop(0)
        require(len(body) == 1 and isinstance(body[0], ast.Return) and
                body[0].value is not None,
                f"{name}: portable function body must be a single return expression")
        deterministic_calls = UNARY_FUNCTIONS | {"clip", "int", "timestep"}
        for node in ast.walk(body[0].value):
            if isinstance(node, ast.Call):
                require(isinstance(node.func, ast.Name) and
                        node.func.id in deterministic_calls,
                        f"{name}: body calls a non-portable function")
            require(not isinstance(
                node, (ast.Attribute, ast.Subscript, ast.Lambda,
                       ast.IfExp, ast.NamedExpr)),
                f"{name}: body contains unsupported Python semantics")
        lowered = expression(
            ast.unparse(body[0].value),
            symbol_dtypes={argument["name"]: argument["dtype"]
                           for argument in arguments},
            integer_dtype=("i64" if return_dtype == "i64" else None))
        require(loads(lowered) <= set(argument_names),
                f"{name}: body captures Python/global state")
        require(_same_dimensions(
            infer_dimensions(lowered, {
                arg["name"]: tuple(arg["dimensions"])
                for arg in arguments}), tuple(return_dimensions)),
            f"{name}: declared return units do not match portable body")
    except (OSError, TypeError, IndentationError, SyntaxError,
            NotImplementedError, ValueError) as error:
        # A native-only contract is allowed below, but an unportable Python
        # body is never executed by either Rust backend.
        lowered = None
        portable_error = error

    backend_implementations = {}
    backend_targets = {
        "b2ir-c-abi-v1": ("cpu", "b2ir-c-abi-v1"),
        "b2ir-cuda-device-v1": ("cuda", "b2ir-cuda-device-v1"),
        "b2ir-metal-v1": ("metal", "b2ir-metal-v1"),
        "b2ir-wgsl-v1": ("wgsl", "b2ir-wgsl-v1"),
    }
    for target, (backend, abi) in backend_targets.items():
        if target not in function.implementations:
            continue
        implementation = function.implementations[target]
        require(not implementation.dynamic and not implementation.dependencies and
                not implementation.compiler_kwds and
                not implementation.get_namespace(None),
                f"{name}: {backend} implementation must be static and self-contained")
        source = implementation.get_code(None)
        symbol_name = implementation.name or name
        require(type(source) is str and source.strip() and
                type(symbol_name) is str and symbol_name.isidentifier() and
                symbol_name.isascii(),
                f"{name}: invalid {backend} source or entry point")
        backend_implementations[backend] = {
            "abi": abi,
            "symbol": symbol_name,
            "source": source,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    if lowered is None and not backend_implementations:
        if isinstance(portable_error, NotImplementedError):
            raise portable_error
        raise NotImplementedError(
            f"Atlas capability: {name}: provide a portable expression or "
            "b2ir-c-abi-v1 implementation") from portable_error
    implementations = {}
    if lowered is not None:
        canonical_body = json.dumps(
            lowered, sort_keys=True, separators=(",", ":"))
        implementations["b2ir-expression-v1"] = hashlib.sha256(
            canonical_body.encode()).hexdigest()
    return {
        "name": name,
        "semantic_version": "1.0.0",
        "abi": "b2ir-function-v1",
        "arguments": arguments,
        "return_dtype": return_dtype,
        "return_dimensions": return_dimensions,
        "effects": {"stateful": False, "deterministic": True,
                    "thread_safe": True, "rng": False},
        "body": lowered,
        "implementations": implementations,
        "backend_implementations": backend_implementations,
    }


def private_integer_constant(source, constants):
    """Evaluate a compiler-private integer expression, or return ``None``.

    Brian's symbolic state updaters occasionally emit integer-typed affine
    coefficients (e.g. ``_BA_v = -1; _lio_1 = -_BA_v``). Only expressions
    rooted exclusively in such private constants are accepted here; model
    indices and other integer-valued inputs therefore remain unsupported.
    """
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            value = node.value
        elif isinstance(node, ast.Name) and node.id in constants:
            value = constants[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = evaluate(node.operand)
            if operand is None:
                return None
            value = -operand if isinstance(node.op, ast.USub) else operand
        else:
            return None
        return value if abs(value) <= 2**53 else None

    return evaluate(ast.parse(source, mode="eval").body)


def private_runtime_integer(source, constants):
    """Whether Brian emitted a bounded, non-model integer temporary.

    ``timestep(t, dt)`` and boolean ``int(...)`` are often hoisted into i64
    scalar temporaries even when the user-visible destination is f64. B2IR
    represents timestep results as logical ticks and makes any later f64
    conversion explicit; boolean integers remain exact 0/1 conversions.
    """
    def literal_integer(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if (isinstance(node, ast.UnaryOp) and
                isinstance(node.op, (ast.USub, ast.UAdd)) and
                isinstance(node.operand, ast.Constant) and
                type(node.operand.value) is int):
            return (-node.operand.value if isinstance(node.op, ast.USub)
                    else node.operand.value)
        return None

    def safe(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            return True
        if isinstance(node, ast.Name):
            return node.id in constants
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return safe(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return ((safe(node.left) and literal_integer(node.right) is not None) or
                    (isinstance(node.op, ast.Add) and safe(node.right)
                     and literal_integer(node.left) is not None))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "timestep":
                # Brian hoists both ``timestep(t-lastspike, dt)`` and
                # ``timestep(duration_expression, dt)`` into private i64
                # temporaries. The public operands remain unit-checked f64
                # quantities; the result is a typed logical tick in B2IR.
                return len(node.args) == 2 and not node.keywords
            if node.func.id == "int" and len(node.args) == 1 and not node.keywords:
                return isinstance(node.args[0], (ast.Compare, ast.BoolOp))
        return False

    return safe(ast.parse(source, mode="eval").body)


def loads(expr):
    if expr["op"] == "load":
        return {expr["name"]}
    result = set()
    for value in expr.values():
        if isinstance(value, dict):
            result |= loads(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    result |= loads(item)
    return result


def load_count(expr, name):
    """Count syntactic reads of ``name`` in an expression tree."""
    if expr["op"] == "load":
        return int(expr["name"] == name)
    result = 0
    for value in expr.values():
        if isinstance(value, dict):
            result += load_count(value, name)
        elif isinstance(value, list):
            result += sum(load_count(item, name) for item in value
                          if isinstance(item, dict))
    return result


def guard_private_temporaries(statements, model_inputs):
    """Move an exclusive downstream guard onto compiler temporaries.

    Brian's C++ generator avoids calculating state-updater temporaries while an
    ``(unless refractory)`` state is frozen. Express the same data-flow fact
    explicitly in B2IR so every executor gets identical work and semantics.
    """
    changed = True
    while changed:
        changed = False
        for index in range(len(statements) - 1, -1, -1):
            statement = statements[index]
            if statement["condition"] is not None or statement["target"] in model_inputs:
                continue
            consumers = [other for other in statements[index + 1:]
                         if statement["target"] in loads(other["value"])]
            conditions = {other["condition"] for other in consumers}
            if consumers and len(conditions) == 1 and None not in conditions:
                statement["condition"] = conditions.pop()
                changed = True


def fuse_private_temporaries(statements, model_inputs):
    """Inline pure single-use vector temporaries under the same guard."""
    changed = True
    while changed:
        changed = False
        for index, statement in enumerate(statements):
            target = statement["target"]
            if target in model_inputs:
                continue
            consumers = [(other_index, other) for other_index, other in
                         enumerate(statements[index + 1:], index + 1)
                         if target in loads(other["value"])]
            condition_users = [other for other in statements[index + 1:]
                               if other["condition"] == target]
            if len(consumers) != 1 or condition_users:
                continue
            consumer_index, consumer = consumers[0]
            # A single downstream statement can read the temporary more than
            # once. Inlining would duplicate its evaluation. That is observably
            # wrong for stochastic temporaries (for example Brian's one Wiener
            # increment reused by every ``xi`` occurrence in Heun), and it also
            # destroys the IR's one-random-stream-per-draw-site invariant.
            if load_count(consumer["value"], target) != 1:
                continue
            if consumer["condition"] != statement["condition"]:
                continue
            # The temporary was evaluated at its original position. Do not
            # move an input read past a write to that same model symbol; this
            # preserves each integrator stage's synchronous reads for coupled ODEs.
            intervening_writes = {other["target"] for other in
                                  statements[index + 1:consumer_index]}
            if loads(statement["value"]) & model_inputs & intervening_writes:
                continue

            def substitute(expr):
                if expr.get("op") == "load" and expr.get("name") == target:
                    return copy.deepcopy(statement["value"])
                return {key: substitute(value) if isinstance(value, dict) else value
                        for key, value in expr.items()}

            consumer["value"] = substitute(consumer["value"])
            statements.pop(index)
            changed = True
            break


@dataclass(frozen=True)
class CodeObjectSpec:
    name: str
    kind: str
    when: str
    order: int
    iteration_domain: str
    scalar: list
    vector: list
    effects: dict

    @classmethod
    def from_abstract(cls, runner, kind, code, variables, inputs, writable,
                      iteration_domain=None, conditional_writes=None,
                      allocate_random_stream=None, random_functions=None,
                      functions=None):
        """Preserve Brian's scalar/vector and state-updater statement order."""
        scalar, vector = make_statements(code, variables, np.float64, optimise=True)
        timed_arrays = {}
        for name, variable in variables.items():
            if not isinstance(variable, TimedArray):
                continue
            try:
                timed_arrays[name] = timed_array_config(variable, runner.clock.dt_)
            except NotImplementedError as error:
                raise NotImplementedError(f"{name}: {error}") from error
        type_variables = dict(variables)
        dimension_variables = {
            name: tuple(float(dim) for dim in variable.dim._dims)
            for name, variable in variables.items()
            if hasattr(variable, "dim")
        }
        conditional_writes = {} if conditional_writes is None else conditional_writes
        integer_constants = {}
        logical_indices = {name for name in inputs
                           if name in {"i", "j", "N", "N_pre", "N_post"}}
        logical_ticks = set()
        symbol_dtypes = {
            name: dtype_name(variable.dtype)
            for name, variable in type_variables.items()
            if hasattr(variable, "dtype") and
            np.dtype(variable.dtype) in {np.dtype(np.float32), np.dtype(np.float64),
                                         np.dtype(np.int32), np.dtype(np.int64),
                                         np.dtype(np.uint32), np.dtype(np.uint64),
                                         np.dtype(np.bool_)}
        }
        symbol_dtypes.update({name: "index" for name in logical_indices})
        def lower(statement):
            # exponential_euler can emit a private affine coefficient such as
            # ``_BA_v = -1`` with an integer dtype. Such a private literal is
            # exactly representable as f64 and never observable as model
            # state. Runtime timestep temporaries instead retain Tick dtype;
            # all other integer values and operations remain fail-closed.
            integer_value = (private_integer_constant(statement.expr, integer_constants)
                             if np.issubdtype(statement.dtype, np.integer)
                             and statement.var not in inputs else None)
            private_integer_literal = integer_value is not None
            private_integer_runtime = (
                np.issubdtype(statement.dtype, np.integer) and
                statement.var not in inputs and
                private_runtime_integer(statement.expr, set(integer_constants)))
            require(np.dtype(statement.dtype) in {
                        np.dtype(np.float32), np.dtype(np.float64),
                        np.dtype(np.int32), np.dtype(np.int64),
                        np.dtype(np.uint32), np.dtype(np.uint64),
                        np.dtype(np.bool_)}
                    or private_integer_literal or private_integer_runtime,
                    f"unsupported statement dtype; got {statement.var}: "
                    f"{statement.dtype} = {statement.expr}")
            if private_integer_literal:
                integer_constants[statement.var] = integer_value
            declared_dtype = (symbol_dtypes.get(statement.var) or
                              dtype_name(statement.dtype))
            expression_integer_dtype = (
                declared_dtype if declared_dtype in INTEGER_DTYPES else None)
            value = expression(
                statement.expr, allocate_random_stream, random_functions,
                timed_arrays, functions, logical_indices, logical_ticks,
                symbol_dtypes, expression_integer_dtype,
                private_constants=integer_constants)
            inplace = {"+=": "add", "-=": "sub", "*=": "mul", "/=": "div",
                       "%=": "mod"}
            if statement.op in inplace:
                left = {"op": "load", "name": statement.var}
                storage_dtype = symbol_dtypes.get(statement.var)
                if storage_dtype in INTEGER_DTYPES:
                    rhs_dtype = infer_dtype(value, symbol_dtypes, functions)
                    require(rhs_dtype in INTEGER_DTYPES,
                            f"{statement.var}: integer in-place assignment "
                            "requires an integer RHS")
                    if rhs_dtype != storage_dtype:
                        value = {"op": "cast", "dtype": storage_dtype,
                                 "arg": value}
                elif storage_dtype == "f32":
                    left = {"op": "f32_to_f64", "arg": left}
                if storage_dtype not in INTEGER_DTYPES:
                    rhs_dtype = infer_dtype(value, symbol_dtypes, functions)
                    if rhs_dtype == "f32":
                        value = {"op": "f32_to_f64", "arg": value}
                    elif rhs_dtype in INTEGER_DTYPES:
                        value = {"op": "cast", "dtype": "f64", "arg": value}
                    elif rhs_dtype == "bool":
                        value = {"op": "bool_to_f64", "arg": value}
                    elif rhs_dtype == "tick":
                        value = {"op": "tick_to_f64", "arg": value}
                value = {"op": inplace[statement.op], "left": left,
                         "right": value}
            else:
                require(statement.op in ("=", ":="), f"statement operator {statement.op!r}")
            require(statement.var not in inputs or statement.var in writable,
                    f"cannot write read-only parameter {statement.var}")
            expression_dimension = infer_dimensions(value, dimension_variables,
                                                     functions)
            logical_dtype = ("tick" if value["op"] in {"timestep", "tick_offset"}
                             else None)
            if private_integer_runtime and logical_dtype is not None:
                target_dtype = logical_dtype
            elif private_integer_literal or private_integer_runtime:
                target_dtype = declared_dtype
            elif statement.var in symbol_dtypes:
                target_dtype = symbol_dtypes[statement.var]
            else:
                target_dtype = dtype_name(statement.dtype)
            if (logical_dtype is not None and
                    statement.dtype in (np.float32, np.float64)):
                value = {"op": "tick_to_f64", "arg": value}
                logical_dtype = None
                target_dtype = dtype_name(statement.dtype)
            value_dtype = infer_dtype(value, symbol_dtypes, functions)
            if (statement.var not in inputs and
                    value_dtype in {"f32", "bool"}):
                # Brian can type a private alias conservatively as float64
                # even when its RHS is an unchanged float32 model value.
                # Preserve the proven RHS type; observable writes still use
                # their declared storage dtype below.
                target_dtype = value_dtype
            if target_dtype == "f32" and value_dtype in INTEGER_DTYPES:
                value = {"op": "cast", "dtype": "f64", "arg": value}
                value_dtype = "f64"
            if target_dtype == "f32" and value_dtype == "f64":
                value = {"op": "f64_to_f32", "arg": value}
                value_dtype = "f32"
            if target_dtype == "f64" and value_dtype in INTEGER_DTYPES:
                value = {"op": "cast", "dtype": "f64", "arg": value}
                value_dtype = "f64"
            if target_dtype in INTEGER_DTYPES and value_dtype != target_dtype:
                value = {"op": "cast", "dtype": target_dtype, "arg": value}
                value_dtype = target_dtype
            require(value_dtype == target_dtype,
                    f"{statement.var}: cannot assign {value_dtype} to "
                    f"{target_dtype} from {statement.expr}")
            if statement.var not in type_variables:
                type_variables[statement.var] = AuxiliaryVariable(
                    statement.var,
                    dtype=statement.dtype,
                    scalar=statement.scalar)
                dimension_variables[statement.var] = expression_dimension
            if logical_dtype == "tick":
                logical_ticks.add(statement.var)
            symbol_dtypes[statement.var] = target_dtype
            return {"target": statement.var,
                    "dtype": target_dtype,
                    "dimensions": list(dimension_variables[statement.var]),
                    "value": value, "condition": conditional_writes.get(statement.var)}
        scalar, vector = [lower(s) for s in scalar], [lower(s) for s in vector]
        guard_private_temporaries(vector, inputs)
        fuse_private_temporaries(vector, inputs)
        statements = scalar + vector
        require(len(statements) <= 128, "at most 128 statements per code object")
        reads = set().union(*(loads(s["value"]) for s in statements)) & inputs
        reads |= {s["condition"] for s in statements if s["condition"] is not None}
        writes = {s["target"] for s in statements} & inputs
        return cls(
            name=runner.name, kind=kind, when=runner.when, order=runner.order,
            iteration_domain=(iteration_domain or
                              ("spiking_neurons" if kind == "reset" else "all_neurons")),
            scalar=scalar, vector=vector,
            effects={"reads": sorted(reads), "writes": sorted(writes)},
        )

    def to_dict(self):
        return asdict(self)
