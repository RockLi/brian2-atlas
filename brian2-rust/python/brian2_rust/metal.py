"""Opt-in float32 Metal execution with validated backend-private plans.

Independent populations fuse all ticks per lane. Coupled models use the bounded
canonical DAG in metal_dag, with explicit cross-lane synchronization. AtlasIR stays
reference-f64; float32 is an explicitly approximate execution contract.
"""
from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import time

import numpy as np

from .plan import (LogicalPlan, PlanValidationError, _logical_plan, validate_model)
from .protocol import canonical_bytes
from . import gpu_types as gt
from .spec import infer_dtype
from .metal_random import SOURCE as _RANDOM_SOURCE, RNG_PROFILE, has_random
from .metal_poisson import SOURCE as _POISSON_SOURCE
from .metal_timed_array import SOURCE as _TIMED_SOURCE, TimedTables
from .gpu_initialization import prepare_model, initialization_records, initialization_seconds

METAL_PROFILE = "b2-metal-f32-v0"


def number(bits):
    if isinstance(bits, np.floating):
        return float(bits)
    return struct.unpack(">f" if len(bits) == 8 else ">d", bytes.fromhex(bits))[0]


def _first_available_ticks(last, dt, period):
    """Use the reference-f64 boundary test for checkpoint refractory state."""
    raw = np.maximum(0, np.ceil(period + last/dt - 1e-3))
    if np.any(raw >= float(2**63 - 1)):
        raise PlanValidationError("Metal refractory tick exceeds signed 64-bit range")
    ticks = raw.astype(np.int64)
    def available(candidate):
        return ((candidate*dt-last)+1e-3*dt)/dt >= period
    while True:
        earlier = (ticks > 0) & available(ticks-1)
        if not earlier.any():
            break
        ticks[earlier] -= 1
    while True:
        later = ~available(ticks)
        if not later.any():
            break
        ticks[later] += 1
    return ticks


def _literal(value):
    value = np.float32(value)
    if not np.isfinite(value):
        raise PlanValidationError("Metal float32 constant is outside the finite range")
    return f"as_type<float>(0x{int(value.view(np.uint32)):08x}u)"


@dataclass(frozen=True)
class MetalKernel:
    population: int
    entry: str
    nodes: tuple[str, ...]
    neurons: int
    start_tick: int
    steps: int
    monitor_steps: int
    spike_capacity: int
    source: str


@dataclass(frozen=True)
class MetalPlan:
    schema: str
    numeric_profile: str
    definition_sha256: str
    instance_sha256: str
    run_sha256: str
    logical: LogicalPlan
    kernels: tuple[MetalKernel, ...]
    dispatches: tuple = ()
    buffers: tuple = ()
    strategy: str = "independent-temporal-fusion"
    event_delivery: str = "none"
    elided_nodes: tuple[str, ...] = ()
    rng_profile: str | None = None
    initializations: tuple = ()

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @property
    def sha256(self):
        return hashlib.sha256(canonical_bytes(self.to_dict())).hexdigest()


class _Expressions:
    def __init__(self, symbols, functions, snapshot=False, *, math_error=None, refractory_elapsed=None, rng=None, timed_tables=None, dtypes=None):
        self.symbols = dict(symbols)
        self.dtypes = {name: "f64" for name in symbols}
        self.dtypes.update(dtypes or {})
        self.snapshot = dict(symbols) if snapshot else {}
        self.functions = functions
        self.lines = []
        self.counter = 0
        self.math_error = math_error
        self.refractory_elapsed = refractory_elapsed
        self.guard = None
        self.rng = rng
        self.timed_tables = timed_tables

    def checked(self, value, dtype):
        local = f"x{self.counter}"
        self.counter += 1
        if self.guard is None:
            self.lines.append(f"{dtype} {local} = {value};")
        else:
            self.lines.append(f"{dtype} {local} = 0; if ({self.guard}) {local} = {value};")
        return local

    def numeric(self, value):
        """Sequence every floating AST result before later expressions can hide it.

        Statement masks suppress the complete RHS; checked() preserves that
        guard. Sequenced locals also avoid unsequenced writes to math_error in
        nested C++ expressions, including eager logical/function arguments.
        """
        if self.math_error is None:
            raise PlanValidationError('GPU floating arithmetic requires a checked execution domain')
        return self.checked(f"b2_finite(float({value}), &{self.math_error})", 'float')

    def expr(self, node):
        op = node["op"]
        if op == "timed_array":
            if self.timed_tables is None or self.math_error is None:
                raise PlanValidationError("GPU TimedArray requires a supported buffer domain")
            time = self.expr(node["time"])
            column = self.expr(node["index"]) if node.get("index") is not None else "0.0f"
            value = (f"b2_timed_array({self.timed_tables.address(node)}, {node['rows']}u, "
                     f"{node['columns'] or 1}u, {_literal(number(node['epsilon']))}, "
                     f"{_literal(node['upsampling'])}, float({time}), float({column}), &{self.math_error})")
            return self.checked(value, "float")
        if op in {"rand", "randn", "binomial", "poisson"}:
            if self.rng is None or self.math_error is None:
                raise PlanValidationError("GPU RNG requires a supported counter domain")
            args = f"{self.rng['seed']}ul, {node['stream']}ul, ulong(tick), {self.rng['index']}"
            if op == "rand":
                return f"b2_uniform({args}, 0ul)"
            if op == "randn":
                return self.checked(f"b2_normal({args}, &{self.math_error})", "float")
            if op == "poisson":
                rate=self.expr(node['lambda'])
                return self.checked(f"b2_poisson({args},float({rate}),&{self.math_error})",'float')
            probability = self.expr(node["p"])
            return self.checked(f"b2_binomial({args}, {node['n']}ul, float({probability}), "
                                f"{'true' if node['approximate'] else 'false'}, &{self.math_error})", "float")
        if op == "load":
            value = self.snapshot.get(node["name"], self.symbols.get(node["name"])) or self._unsupported(op)
            return self.numeric(value) if node["name"] == "t" else value
        if op == "literal":
            return _literal(number(node["bits"]))
        if op == "boolean":
            return "true" if node["value"] else "false"
        if op == "integer":
            return gt.integer_literal(node["value"], node["dtype"])
        if op == "call":
            function = self.functions[node["function"]]
            args = {arg["name"]: self.expr(value) for arg, value in zip(
                function["arguments"], node["arguments"], strict=True)}
            if function['body'] is None:
                from .gpu_functions import namespace
                values = ', '.join(f"{gt.CTYPES[a['dtype']]}({args[a['name']]})" for a in function['arguments'])
                value = f"{namespace(function)}::b2_invoke({values})"
                dtype = function['return_dtype']
                return self.numeric(value) if dtype in {'f32','f64'} else self.checked(value, gt.CTYPES[dtype])
            nested = _Expressions(args, self.functions, math_error=self.math_error, rng=self.rng,
                                  timed_tables=self.timed_tables,
                                  dtypes={arg["name"]:arg["dtype"] for arg in function["arguments"]})
            nested.counter, nested.guard = self.counter, self.guard
            value = nested.expr(function["body"])
            self.counter = nested.counter
            self.lines.extend(nested.lines)
            return value
        if op == "timestep":
            if self.math_error is None:
                raise PlanValidationError("GPU timestep requires a checked execution domain")
            if (self.refractory_elapsed is not None and node["dt"] == {"op":"load","name":"dt"}
                    and node["time"] == {"op":"sub","left":{"op":"load","name":"t"},
                                         "right":{"op":"load","name":"lastspike"}}):
                return self.refractory_elapsed
            value = f"b2_timestep(float({self.expr(node['time'])}), float({self.expr(node['dt'])}), &{self.math_error})"
            # AtlasIR eagerly evaluates logical operands, but a statement's mask
            # suppresses its entire RHS. Sequence checked calls accordingly.
            return self.checked(value, "long")
        if op == "tick_offset":
            if self.math_error is None:
                raise PlanValidationError("GPU tick_offset requires a checked execution domain")
            tick = self.expr(node["tick"])
            return self.checked(f"b2_tick_offset({tick}, {node['offset']}l, &{self.math_error})", "long")
        if op in {"cast", "f32_to_f64", "f64_to_f32", "bool_to_f64", "index_to_f64", "tick_to_f64"}:
            dtype = node.get("dtype", "f32")
            value = self.expr(node['arg'])
            source = infer_dtype(node['arg'], self.dtypes, self.functions)
            if dtype in gt.INTEGERS:
                if source in gt.INTEGERS: return gt.wrap(value, dtype)
                return f"b2_cast_{dtype}(float({value}))"
            return f"{gt.CTYPES.get(dtype, 'float')}({value})"
        if op in {"neg", "abs"} and infer_dtype(node['arg'],self.dtypes,self.functions) in gt.INTEGERS:
            dtype=infer_dtype(node['arg'],self.dtypes,self.functions)
            value=self.expr(node['arg']); unsigned='uint' if dtype.endswith('32') else 'ulong'
            neg=gt.wrap(f'{unsigned}(0)-{unsigned}({value})',dtype)
            return neg if op=='neg' else f'(({value}) < 0 ? {neg} : ({value}))'
        if op in {"neg", "not"}:
            return f"({'-' if op == 'neg' else '!'}({self.expr(node['arg'])}))"
        if op in {'add','sub','mul','mod','floor_div'}:
            dtype=infer_dtype(node,self.dtypes,self.functions)
            if dtype in gt.INTEGERS:
                left,right=self.expr(node['left']),self.expr(node['right'])
                if op in {'mod','floor_div'}:
                    if self.math_error is None:
                        raise PlanValidationError('GPU integer division requires a checked execution domain')
                    return self.checked(f"b2_div_{dtype}({left}, {right}, {'true' if op=='mod' else 'false'}, &{self.math_error})",gt.CTYPES[dtype])
                unsigned='uint' if dtype.endswith('32') else 'ulong'
                operator={'add':'+','sub':'-','mul':'*'}[op]
                return gt.wrap(f'{unsigned}({left}) {operator} {unsigned}({right})',dtype)
            if op in {'mod','floor_div'}:
                if self.math_error is None:
                    raise PlanValidationError('GPU floor division requires a checked execution domain')
                left,right=self.expr(node['left']),self.expr(node['right'])
                return self.checked(f"b2_float_div(float({left}),float({right}),{'true' if op=='mod' else 'false'},&{self.math_error})",'float')
        unary = {name: name for name in ("exp", "log", "log10", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "floor", "ceil", "trunc")}
        unary.update(exp="b2_exp", abs="abs", arccos="acos", arcsin="asin", arctan="atan",
                     expm1="b2_expm1", exprel="b2_exprel", log1p="b2_log1p", sign="sign")
        if op in unary:
            return self.numeric(f"{unary[op]}(float({self.expr(node['arg'])}))")
        if op == "clip":
            return self.numeric(f"b2_clip(float({self.expr(node['value'])}), float({self.expr(node['min'])}), float({self.expr(node['max'])}))")
        # Stabilize exponential Euler's exact exported pattern:
        # -q + (q+x)*exp(a) = x + (q+x)*expm1(a).
        # This avoids catastrophic cancellation when a is close to zero.
        if op == "add" and node["left"].get("op") == "neg":
            q = node["left"]["arg"]
            product = node["right"]
            if product.get("op") == "mul":
                total, exponential = product["left"], product["right"]
                if (total.get("op") == "add" and total["left"] == q and
                        exponential.get("op") == "exp"):
                    x = self.expr(total["right"])
                    return self.numeric(f"(({x}) + (({self.expr(q)}) + ({x}))*b2_expm1(float({self.expr(exponential['arg'])})))")
        operators = {"add": "+", "sub": "-", "mul": "*", "div": "/", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=", "and": "&&", "or": "||"}
        if op in operators:
            value = f"(({self.expr(node['left'])}) {operators[op]} ({self.expr(node['right'])}))"
            return self.numeric(value) if op in {"add","sub","mul","div"} else value
        if op == "pow":
            # Exported HH equations can contain exp(1000*v)**fraction.
            # Evaluating the inner exp in f32 underflows on Apple GPUs before
            # the fractional power rescales it. Lower this positive-base form
            # directly; the CPU f32 control uses the same stabilized operation.
            if node["left"]["op"] == "exp":
                return self.numeric(f"b2_exp(float({self.expr(node['left']['arg'])}) * float({self.expr(node['right'])}))")
            from .native import _constant_number
            exponent = _constant_number(node["right"])
            if exponent is not None and float(exponent).is_integer() and abs(exponent) <= 32:
                return self.numeric(f"b2_powi(float({self.expr(node['left'])}), {int(exponent)})")
            return self.numeric(f"pow(float({self.expr(node['left'])}), float({self.expr(node['right'])}))")
        self._unsupported(op)

    @staticmethod
    def _unsupported(op):
        raise PlanValidationError(f"Metal float32 does not support expression {op!r}")

    def statements(self, statements):
        for stmt in statements:
            self.guard = self.symbols[stmt["condition"]] if stmt.get("condition") is not None else None
            value = self.expr(stmt["value"])
            if stmt.get("condition") is not None:
                guard = self.symbols[stmt["condition"]]
                old = self.symbols.get(stmt["target"], "NAN")
                value = f"({guard} ? ({value}) : ({old}))"
            dtype = gt.CTYPES[stmt["dtype"]]
            local = f"x{self.counter}"
            self.counter += 1
            self.lines.append(f"{dtype} {local} = {value};")
            self.symbols[stmt["target"]] = local
            self.dtypes[stmt["target"]] = stmt["dtype"]


_PRELUDE = '''#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline long b2_timestep(float value, float dt, thread bool *fault) {
    float steps = (value + 1e-3f*dt)/dt;
    if (!(value >= 0.0f && isfinite(dt) && dt > 0.0f && isfinite(steps) && steps <= 0x1p53f)) {
        *fault = true;
        return 0;
    }
    return long(steps);
}
inline long b2_tick_offset(long tick, long offset, thread bool *fault) {
    // Frozen AtlasIR evaluates logical ticks as f64. The boundary ties
    // +(2^53+1) and -(2^53+1) round to +/-2^53 and are accepted there.
    // Check before addition so even invalid inputs cannot overflow int64.
    const long limit = 9007199254740992l;
    const long allowed = limit + 1l;
    if (tick > allowed-offset || tick < -allowed-offset) {
        *fault = true;
        return 0;
    }
    long result = tick + offset;
    return result > limit ? limit : (result < -limit ? -limit : result);
}
inline float b2_finite(float value, thread bool *error) {
    if (!isfinite(value)) { *error = true; return 0.0f; }
    return value;
}
inline float b2_clip(float value, float lo, float hi) {
    // AtlasIR defines ordered max/min even when bounds are reversed. std::clamp
    // has a precondition lo <= hi and is not this operation.
    return fmin(fmax(value,lo),hi);
}
inline float b2_powi(float x, int power) {
    uint n = uint(power < 0 ? -power : power);
    float result = 1.0f;
    while (n) { if (n & 1u) result *= x; x *= x; n >>= 1; }
    return power < 0 ? 1.0f/result : result;
}
inline float b2_exp(float x) {
    if (x > 88.722839f) return INFINITY;
    if (x < -103.97208f) return 0.0f;
    float nf = floor(x*1.4426950408889634f + 0.5f);
    int n = int(nf);
    float r = (x - nf*0.693359375f) - nf*(-0.00021219444005469058f);
    float p = 1.0f/5040.0f;
    p = 1.0f/720.0f + r*p;
    p = 1.0f/120.0f + r*p;
    p = 1.0f/24.0f + r*p;
    p = 1.0f/6.0f + r*p;
    p = 0.5f + r*p;
    float value = 1.0f + (r + r*r*p);
    float result = ldexp(value, n);
    // Metal arithmetic flushes subnormals; make the shared exp contract explicit.
    return fabs(result) < 0x1p-126f ? 0.0f : result;
}
inline float b2_expm1(float x) {
    if (abs(x) < 0.001f) return x * (1.0f + x * (0.5f + x * (1.0f/6.0f + x/24.0f)));
    return b2_exp(x) - 1.0f;
}
inline float b2_exprel(float x) {
    // Avoid cancellation in exp(x)-1 near zero. The first omitted term
    // is x^6/7!, below 8e-10 throughout this interval.
    if (abs(x) < 0.125f) return 1.0f + x * (0.5f + x * (1.0f/6.0f +
        x * (1.0f/24.0f + x * (1.0f/120.0f + x/720.0f))));
    // exp(x) overflows before exprel(x) does. Split only that range;
    // divide before multiplying so every intermediate of a finite result fits.
    // The subtracted 1/x is far below an f32 ulp in this range.
    if (x >= 88.722839f) {
        float half_exp = b2_exp(0.5f*x);
        return (half_exp/x)*half_exp;
    }
    return b2_expm1(x)/x;
}
inline float b2_log1p(float x) {
    if (abs(x) < 0.001f) return x * (1.0f + x * (-0.5f + x * (1.0f/3.0f - x/4.0f)));
    return log(1.0f + x);
}
'''


_PRELUDE += _RANDOM_SOURCE
_PRELUDE += _POISSON_SOURCE
_PRELUDE += _TIMED_SOURCE
_PRELUDE += gt.SOURCE


def population_types(model, p, *, single_tick=False):
    """Slot 1 holds scalar parameters or an integer event-source input table."""
    generator = model["instance"]["populations"][p].get("spike_generator") is not None
    return ("float", "const long" if generator else "const float", "float", "const int",
            "long", "uint", "uchar", "long", "long", "uchar",
            "float" if single_tick else "const float") + (("uchar",) if single_tick else ()) + (("const float",) if model["definition"]["populations"][p].get("linked_variables") else ())


def _kernel(model, logical, p, *, entry=None, single_tick=False, dispatch_clock=None):
    from .metal_event_layout import event_lanes, event_offset
    pop = model["definition"]["populations"][p]
    instance = model["instance"]["populations"][p]
    clock_index = pop["clock"] if dispatch_clock is None else dispatch_clock
    if not single_tick and clock_index != pop["clock"]:
        raise PlanValidationError("Temporal fusion requires the population clock")
    clock = logical.clocks[clock_index]
    n, start, steps = pop["count"], clock.start_tick, clock.steps
    monitor = pop["monitor"]
    window = monitor["window_steps"]
    recording = pop.get("spike_monitor") is not None
    ref = pop["refractory"]
    fixed_ref = ref is not None and ref["mode"] == "fixed"
    period = instance["refractory"]["period_ticks"] if ref is not None else 0
    capacity = (window + max(1, period) - 1) // max(1, period) if recording else 0
    nodes = [node for node in logical.nodes if node.owner_kind == "population" and node.owner_index == p]
    if any(node.clock != clock_index for node in nodes):
        raise PlanValidationError("Metal independent-lane kernel requires one clock per population")
    states = {s["name"]: f"s{i}" for i, s in enumerate(pop["states"])}
    owner_clock = logical.clocks[pop["clock"]]
    from .gpu_schedule import owner_tick_expression
    needs_owner_time = any(
        node.operation != "code_object"
        or pop["code_objects"][node.item_index]["kind"] != "run_regularly"
        or "t" in pop["code_objects"][node.item_index]["effects"]["reads"]
        for node in nodes)
    owner_tick = (owner_tick_expression(logical, clock_index, pop["clock"])
                  if needs_owner_time else "tick")
    symbols = {**states, "dt": _literal(number(owner_clock.dt)), "t": "time", "i": "i", "N": f"{n}u"}
    state_layout, _ = gt.layout(pop['states'],n)
    parameter_layout, offset = gt.layout(pop['parameters'],n)
    dtypes = {s['name']:s['dtype'] for s in pop['states']+pop['parameters']+pop.get('linked_variables',[])}
    dtypes.update(i='index',N='index',t='f64',dt='f64',lastspike='f64',not_refractory='bool')
    for parameter in pop['parameters']:
        scalar=parameter['index_domain']=='scalar'
        symbols[parameter['name']]=gt.read('parameters',parameter_layout[parameter['name']],'0' if scalar else 'i')
    linked_layout,_=gt.layout(pop.get('linked_variables',[]),n)
    symbols.update({name:gt.read('linked_values',field) for name,field in linked_layout.items()})
    trace_symbols=[dict(name=name,dtype=dtypes[name]) for name in monitor['variables']]
    trace_layout,_=gt.layout(trace_symbols,window*len(monitor['record']))
    if ref is not None:
        symbols.update(lastspike="last_time", not_refractory="available")
    functions = {f["name"]: f for f in model["definition"]["functions"]}
    timed_tables = TimedTables(pop["code_objects"], functions, buffer="parameters", offset=offset)
    generator = instance.get("spike_generator") is not None
    input_type = "long" if generator else "float"
    entry = entry or f"population_{p}"
    extra_arguments=[];next_binding=11
    if single_tick:
        extra_arguments.append('device uchar *event_history [[buffer(11)]]');next_binding+=1
    if linked_layout:
        extra_arguments.append(f'device const float *linked_values [[buffer({next_binding})]]');next_binding+=1
    if single_tick:extra_arguments.append(f'constant long &tick [[buffer({next_binding})]]')
    tick_arguments=', '.join(extra_arguments)+', ' if extra_arguments else ''
    initial_time_qualifier = "" if single_tick else "const "
    fired_initial = "last_fired[i] != 0" if single_tick else "false"
    emitted_initial = "spike_counts[i]" if single_tick else "0"
    lines = [_PRELUDE, f'''kernel void {entry}(
    device float *state [[buffer(0)]], device const {input_type} *parameters [[buffer(1)]],
    device float *trace [[buffer(2)]], device const int *record_slot [[buffer(3)]],
    device long *spike_ticks [[buffer(4)]], device uint *spike_counts [[buffer(5)]],
    device uchar *last_fired [[buffer(6)]], device long *last_tick [[buffer(7)]],
    device long *until_tick [[buffer(8)]], device uchar *not_refractory [[buffer(9)]],
    device {initial_time_qualifier}float *initial_last_time [[buffer(10)]], {tick_arguments}uint i [[thread_position_in_grid]]) {{
    if (i >= {n}u) return;
    long last = last_tick[i];
    long until = until_tick[i];
    bool math_error = until < 0;
    bool available = not_refractory[i] != 0;
    float last_time = initial_last_time[i];
    bool fired = {fired_initial};
    uint emitted = {emitted_initial};
    int record = record_slot[i];''']
    lines += [f"    {gt.CTYPES[dtypes[name]]} {local} = {gt.read('state',state_layout[name])};" for name,local in states.items()]
    event_flags = {'spike':'fired'}
    for e,event in enumerate(event_lanes(pop)[1:],1):
        event_flags[event]=f'fired_event_{e}'
        initial=f'last_fired[{e*n}+i] != 0' if single_tick else 'false'
        lines.append(f'    bool fired_event_{e} = {initial};')
    lines += ["    {" if single_tick else f"    for (long tick={start}; tick<{start+steps}; ++tick) {{", f"        float time = float({owner_tick}) * {_literal(number(owner_clock.dt))};"]
    recorded_state = False
    for node in nodes:
        if node.operation == "event_monitor":
            # Dedicated DAG stages capture variable values at their exact slot.
            continue
        if node.operation == "event_source":
            if not generator:
                raise PlanValidationError("Metal event source requires a validated spike schedule")
            # CSR offsets and absolute emission ticks are int64, including for
            # long-running Device continuations. No float time comparison.
            lines += ["        {",
                      "            ulong lo = ulong(parameters[i]);",
                      "            ulong end = ulong(parameters[i+1]);",
                      "            ulong hi = end;",
                      "            while (lo < hi) {",
                      "                ulong mid = lo + (hi-lo)/2;",
                      "                if (parameters[mid] < tick) lo = mid+1; else hi = mid;",
                      "            }",
                      "            fired = lo < end && parameters[lo] == tick;",
                      "        }"]
            if single_tick:
                lines.append(f"        if (tick >= {start+steps-window}) event_history[(tick-{start+steps-window})*{n}+i] = uchar(fired);")
            continue
        if node.operation == "state_monitor":
            # Frozen v1 records the shared variable/index union at the first
            # scheduled StateMonitor for this population, once per tick.
            if recorded_state:
                continue
            recorded_state = True
            lines.append(f"        if (record >= 0 && tick >= {start+steps-window}) {{")
            for index, variable in enumerate(monitor["variables"]):
                if variable not in symbols:
                    raise PlanValidationError(f"unsupported Metal monitor variable {variable}")
                at=f"(tick-{start+steps-window})*{len(monitor['record'])}+record"
                lines.append("            "+gt.write('trace',trace_layout[variable],at,symbols[variable]))
            lines.append("        }")
            continue
        if node.operation == "spike_monitor":
            lines.append(f"        if (fired && tick >= {start+steps-window}) spike_ticks[ulong(i)*{capacity}+emitted++] = tick;")
            continue
        if node.operation != "code_object":
            raise PlanValidationError(f"Metal does not support operation {node.operation}")
        code = pop["code_objects"][node.item_index]
        kind = code["kind"]
        if kind not in {"state_update", "threshold", "reset", "run_regularly", "subexpression_update", "poisson_input"}:
            raise PlanValidationError(f"Metal does not support code object {kind}")
        event=code.get('event_name')
        event_flag=event_flags.get(event,'fired')
        if kind == "state_update" and fixed_ref:
            lines.append("        available = tick >= until;")
        elapsed = f"(last >= 0 ? tick-last : until+(tick-{start}))" if ref is not None and not fixed_ref and clock_index == pop["clock"] else None
        block = _Expressions(symbols, functions, math_error="math_error", refractory_elapsed=elapsed,
                             rng={"seed":model["instance"]["rng_seed"], "index":"ulong(i)"}, timed_tables=timed_tables, dtypes=dtypes)
        block.statements(code["scalar"])
        scalar_lines = block.lines
        block.lines = []
        if kind == "state_update":
            # Model states have simultaneous-update semantics. The refractory
            # gate is updated sequentially inside this same code object.
            block.snapshot = {name: symbols[name] for name in states}
        block.statements(code["vector"])
        lines.append("        {")
        lines += ["            " + line for line in scalar_lines]
        if kind == "reset":
            lines.append(f"            if ({event_flag}) {{")
        lines += ["            " + line for line in block.lines]
        if kind == "threshold":
            condition = block.symbols["_cond"]
            lines.append(f"            {event_flag} = {condition}" + (" && available;" if ref is not None and event=='spike' else ";"))
            if single_tick:
                history_offset=event_offset(pop,event)*window
                lines.append(f"            if (tick >= {start+steps-window}) event_history[{history_offset}+(tick-{start+steps-window})*{n}+i] = uchar({event_flag});")
            if ref is not None and event=='spike':
                expiry = f"until = tick + {period};" if fixed_ref else ""
                lines.append(f"            if (fired) {{ last = tick; last_time = time; available = false; {expiry} }}")
        else:
            for name in code["effects"]["writes"]:
                if name in states:
                    lines.append(f"            {states[name]} = {block.symbols[name]};")
                elif name == "not_refractory":
                    lines.append(f"            available = {block.symbols[name]};")
                else:
                    raise PlanValidationError(f"Metal cannot write resource {name}")
        if kind == "reset":
            lines.append("            }")
        lines.append("        }")
    lines += ["    }"]
    lines += ['    '+gt.write('state',state_layout[name],'i',local) for name,local in states.items()]
    if single_tick:
        lines.append("    initial_last_time[i] = last_time;")
    for e,event in enumerate(event_lanes(pop)[1:],1):
        lines.append(f'    last_fired[{e*n}+i] = uchar({event_flags[event]});')
    lines += ["    spike_counts[i] = emitted; last_fired[i] = uchar(fired);",
              "    last_tick[i] = last; until_tick[i] = math_error ? -1 : until; not_refractory[i] = uchar(available);", "}"]
    return MetalKernel(p, entry, tuple(node.id for node in nodes), n,
                       start, steps, window, capacity, "\n".join(lines))


def _derive_metal_plan(model, *, numeric_mode, event_delivery="scan", _native_backend="metal", synapse_prefix=False, synapse_fusion=False, synapse_sparse=False):
    if not (type(synapse_sparse) is bool or type(synapse_sparse) is str and synapse_sparse=='bitset'):
        raise PlanValidationError("synapse_sparse must be boolean or 'bitset'")
    if type(synapse_fusion) is not bool:raise PlanValidationError("synapse_fusion must be boolean")
    if type(synapse_prefix) is not bool:raise PlanValidationError("synapse_prefix must be boolean")
    if event_delivery not in {"sparse", "scan"}:
        raise PlanValidationError("Metal event_delivery must be sparse or scan")
    if numeric_mode != "float32":
        raise PlanValidationError("Metal requires explicit numeric_mode='float32'; reference-f64 is unchanged")
    d = model["definition"]
    from .gpu_functions import validate_functions, inject_sources
    validate_functions(model, _native_backend)
    logical = _logical_plan(model)
    separate_population_clock = any(node.owner_kind == "population" and
        node.clock != d["populations"][node.owner_index]["clock"] for node in logical.nodes)
    if separate_population_clock or d["synapses"] or any(pop.get('linked_variables') or pop.get('event_monitors') or any(e!='spike' for e in pop['events']) for pop in d['populations']):
        from .metal_dag import derive_dag
        plan = derive_dag(model, logical, event_delivery=event_delivery, synapse_prefix=synapse_prefix,synapse_fusion=synapse_fusion,synapse_sparse=synapse_sparse)
        return replace(plan, kernels=tuple(replace(k, source=inject_sources(k.source, model, 'metal'))
                                          for k in plan.kernels)) if _native_backend == 'metal' else plan
    hashes = model["protocol"]["layers"]
    kernels = tuple(_kernel(model, logical, p) for p in range(len(d["populations"])))
    if _native_backend == 'metal':
        kernels = tuple(replace(k, source=inject_sources(k.source, model, 'metal')) for k in kernels)
    return MetalPlan("b2-metal-plan-v0", METAL_PROFILE, hashes["definition"],
                     hashes["instance"], hashes["run"], logical, kernels,
                     rng_profile=RNG_PROFILE if has_random(d) else None,
                     initializations=initialization_records(model))


def build_metal_plan(model, *, numeric_mode, runner=None, event_delivery="scan", synapse_prefix=False, synapse_fusion=False, synapse_sparse=False):
    return _derive_metal_plan(prepare_model(validate_model(model, runner=runner), runner=runner),
                             numeric_mode=numeric_mode, event_delivery=event_delivery, synapse_prefix=synapse_prefix,synapse_fusion=synapse_fusion,synapse_sparse=synapse_sparse)


_CPU_PRELUDE = """#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
using uint = uint32_t; using ulong = uint64_t; using uchar = uint8_t;
template<class T, class U> T as_type(U value) { static_assert(sizeof(T)==sizeof(U)); T result; std::memcpy(&result, &value, sizeof(result)); return result; }
using std::abs; using std::exp; using std::log; using std::log10; using std::pow;
using std::sin; using std::cos; using std::tan; using std::sinh; using std::cosh;
using std::tanh; using std::sqrt; using std::floor; using std::ceil; using std::trunc;
using std::ldexp; using std::acos; using std::asin; using std::atan; using std::clamp;
using std::isfinite; using std::isnan; using std::fmin; using std::fmax;
inline float sign(float x) { return (x>0.0f)-(x<0.0f); }
"""

def population_arrays(model, p, kernel, max_buffer_bytes):
    from .metal_event_layout import event_lanes
    pop = model["definition"]["populations"][p]
    inst = model["instance"]["populations"][p]
    n, steps, window = kernel.neurons, kernel.steps, kernel.monitor_steps
    monitor = pop["monitor"]
    state_parts=[gt.pack(inst['initial_state'][symbol['name']],symbol['dtype']) for symbol in pop['states']]
    state=np.concatenate(state_parts) if state_parts else np.zeros(1,np.float32)
    tables = TimedTables(pop["code_objects"], {f["name"]:f for f in model["definition"]["functions"]}, buffer="parameters", offset=0)
    parameter_parts = [gt.pack(inst["parameters"][s["name"]],s["dtype"]) for s in pop["parameters"]]
    parameters = np.concatenate([*parameter_parts, tables.values])
    if not parameters.size:
        parameters = np.zeros(1, np.float32)
    generator = inst.get("spike_generator")
    if generator is not None:
        size = (n+1+len(generator["spike_ticks"]))*8
        if size > max_buffer_bytes:
            raise MemoryError(f"Metal spike schedule requires {size} bytes")
        indices = np.asarray(generator["spike_indices"], dtype=np.int64)
        ticks = np.asarray(generator["spike_ticks"], dtype=np.int64)
        offsets = np.concatenate(([n+1], n+1+np.cumsum(np.bincount(indices, minlength=n))))
        parameters = np.concatenate((offsets, ticks[np.argsort(indices, kind="stable")])).astype(np.int64)
    dtypes={s["name"]:s["dtype"] for s in pop["states"]+pop["parameters"]+pop.get("linked_variables",[])}
    trace_count = sum(gt.width(dtypes[name]) for name in monitor["variables"]) * len(monitor["record"]) * window
    required = max(trace_count*4, n*kernel.spike_capacity*8)
    if required > max_buffer_bytes:
        raise MemoryError(f"Metal recording buffer requires {required} bytes; use a bounded recording window")
    trace = np.zeros(max(1, trace_count), np.float32)
    record_slot = np.full(n, -1, np.int32)
    record_slot[monitor["record"]] = np.arange(len(monitor["record"]), dtype=np.int32)
    ticks = np.zeros(max(1, n*kernel.spike_capacity), np.int64)
    counts = np.zeros(n, np.uint32)
    flag_bytes=n*len(event_lanes(pop))
    if flag_bytes>max_buffer_bytes:
        raise MemoryError('GPU named-event flags exceed configured buffer limit')
    last_fired = np.zeros(flag_bytes, np.uint8)
    last_tick = np.full(n, -1, np.int64)
    until = np.zeros(n, np.int64)
    available = np.ones(n, np.uint8)
    last_time = np.zeros(n, np.float32)
    initial_last = None
    if inst.get("refractory") is not None:
        ref = inst["refractory"]
        dt = number(pop["dt"])
        initial_last = np.asarray([number(v) for v in ref["initial_lastspike"]])
        last_time = initial_last.astype(np.float32)
        if pop["refractory"]["mode"] == "fixed":
            until = _first_available_ticks(initial_last, dt, ref["period_ticks"])
        else:
            # Preserve integer elapsed ticks across checkpoints. Subsequent
            # emissions use last_tick directly, never float32 t-lastspike.
            elapsed = ((kernel.start_tick*dt-initial_last)+1e-3*dt)/dt
            if not np.isfinite(elapsed).all() or np.any(elapsed < 0) or np.any(elapsed >= float(2**63)):
                raise PlanValidationError("GPU refractory elapsed ticks exceed signed 64-bit range")
            until = elapsed.astype(np.int64)
        available = np.asarray(ref["initial_not_refractory"], dtype=np.uint8)
    arrays = [state, parameters, trace, record_slot, ticks, counts, last_fired, last_tick, until, available, last_time]
    if any(a.nbytes > max_buffer_bytes for a in arrays):
        raise MemoryError("Metal buffer exceeds configured memory limit")
    return arrays, initial_last


def population_result(model, p, kernel, arrays, initial_last):
    pop = model["definition"]["populations"][p]
    n, window = pop["count"], kernel.monitor_steps
    monitor = pop["monitor"]
    state_layout,_=gt.layout(pop['states'],n)
    dtypes={s['name']:s['dtype'] for s in pop['states']+pop['parameters']+pop.get('linked_variables',[])}
    trace_layout,_=gt.layout([dict(name=name,dtype=dtypes[name]) for name in monitor['variables']],window*len(monitor['record']))
    state, parameters, trace, record_slot, ticks, counts, last_fired, last_tick, until, available, last_time = arrays
    if np.any(until < 0):
        raise FloatingPointError("GPU numeric evaluation failed (floating arithmetic, linked index, division, timestep/tick_offset, random sampler or TimedArray)")
    decoded_states={name:gt.unpack(state,field) for name,field in state_layout.items()}
    decoded_traces={name:gt.unpack(trace,field).reshape(window,len(monitor['record'])) for name,field in trace_layout.items()}
    if not gt.finite([*decoded_states.values(),*decoded_traces.values()]):
        raise FloatingPointError("Metal float32 produced non-finite state or trace")
    from .metal_event_layout import spike_coordinates
    event_ticks, indices = spike_coordinates(ticks, counts, kernel.spike_capacity)
    ref_result = None
    if initial_last is not None:
        changed = last_tick >= 0
        initial_last[changed] = last_tick[changed] * number(pop["dt"])
        ref_result = {"lastspike": initial_last, "not_refractory": available.astype(bool)}
    return {
        "states": decoded_states,
        "trace": decoded_traces,
        "spike_ticks": event_ticks, "indices": indices,
        "counts": counts.astype(np.int64), "last_spikes": np.flatnonzero(last_fired[:n]),
        "refractory": ref_result,
    }



class MetalExecutor:
    """Compile a planned kernel once; execute/replay with explicit float32 data."""
    def __init__(self, model, directory, *, numeric_mode, runner=None, plan=None, event_delivery="scan", dag_execution="direct", dag_synchronization="explicit", compile_reuse=False, reuse_from=None, synapse_prefix=False, synapse_fusion=False, synapse_sparse=False, _validated_input=None):
        if platform.system() != "Darwin":
            raise RuntimeError("Apple Metal requires macOS")
        from .metal_buffers import validate_mode, validate_synchronization
        self.dag_execution = validate_mode(dag_execution)
        self.dag_synchronization = validate_synchronization(dag_synchronization)
        self._resident_dag_bytes = 0
        self._metal_dag_metadata = None
        self._metal_dag_report = None
        from .gpu_validation import executor_model
        self.model = executor_model(model,runner=runner,activation=_validated_input)
        expected = _derive_metal_plan(self.model, numeric_mode=numeric_mode, event_delivery=event_delivery, synapse_prefix=synapse_prefix,synapse_fusion=synapse_fusion,synapse_sparse=synapse_sparse)
        if plan is not None and canonical_bytes(plan.to_dict()) != canonical_bytes(expected.to_dict()):
            raise PlanValidationError("Metal plan does not match model or numeric mode")
        self.plan = expected
        from .gpu_compilation import request,compiler_context
        portable=request(self.model,compile_reuse,reuse_from,type(self))
        self.compilation_report=dict(requested=compile_reuse,portable=portable,bridge_reused=False,kernels_reused=0,kernels_compiled=0)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        bridge = Path(__file__).parent / "metal_runtime/bridge.m"
        bridge_options = ("-O2", "-ffp-contract=off", "-fobjc-arc", "-dynamiclib", "-framework", "Foundation", "-framework", "Metal")
        digest = hashlib.sha256(bridge.read_bytes()+bridge.with_name('clocks.h').read_bytes()+repr(bridge_options).encode()).hexdigest()[:16]
        library = directory / f"metal-bridge-{digest}.dylib"
        self._bridge_identity=(digest,compiler_context('clang',dict(os.environ),list(bridge_options))) if compile_reuse else None
        reuse_bridge=compile_reuse and reuse_from is not None and self._bridge_identity==getattr(reuse_from,'_bridge_identity',None)
        if not reuse_bridge and not library.exists():
            subprocess.run(["clang", *bridge_options, str(bridge), "-o", str(library)], check=True,
                           capture_output=True, text=True)
        self.bridge = reuse_from.bridge if reuse_bridge else ctypes.CDLL(str(library))
        self.compilation_report['bridge_reused']=reuse_bridge
        pointer, u64, u32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32
        self.bridge.b2_metal_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, pointer, ctypes.c_size_t]
        self.bridge.b2_metal_create.restype = pointer
        self.bridge.b2_metal_destroy.argtypes = [pointer]
        self.bridge.b2_metal_clone_pipeline.argtypes=[pointer]
        self.bridge.b2_metal_clone_pipeline.restype=pointer
        self.bridge.b2_metal_device_name.argtypes = [pointer, pointer, ctypes.c_size_t]
        self.bridge.b2_metal_run.argtypes = [pointer, ctypes.POINTER(pointer), ctypes.POINTER(u64), u32, u32, u32,
                                           ctypes.POINTER(ctypes.c_double), pointer, ctypes.c_size_t]
        self.handles = []
        self.device_name = ""
        error = ctypes.create_string_buffer(8192)
        started = time.perf_counter()
        old_kernels={(k.entry,k.source):h for k,h in zip(reuse_from.plan.kernels,reuse_from.handles,strict=True)} if portable and reuse_bridge else {}
        try:
            for kernel in self.plan.kernels:
                (directory / f"{kernel.entry}.metal").write_text(kernel.source)
                old=old_kernels.get((kernel.entry,kernel.source))
                handle=self.bridge.b2_metal_clone_pipeline(old) if old is not None else None
                if handle:self.compilation_report['kernels_reused']+=1
                else:
                    handle = self.bridge.b2_metal_create(kernel.source.encode(), kernel.entry.encode(), error, len(error))
                    self.compilation_report['kernels_compiled']+=1
                if not handle:
                    raise RuntimeError(error.value.decode())
                self.handles.append(handle)
                self.bridge.b2_metal_device_name(handle, error, len(error))
                self.device_name = error.value.decode()
        except BaseException:
            self.close()
            raise
        self.compile_seconds = time.perf_counter() - started
        (directory / "metal-plan.json").write_text(self.plan.to_json())
        (directory / 'compilation.json').write_text(json.dumps(self.compilation_report,indent=2)+'\n')

    def _release_resident_dag(self):
        self._activation_upload_indices=[];self._activation_buffers_adopted=False
        if getattr(self, 'handles', []):
            clear = self.bridge.b2_metal_clear_dag
            clear.argtypes = [ctypes.c_void_p]
            clear.restype = None
            clear(self.handles[0])
        self._resident_dag_bytes = 0

    def _execute_dag(self, arrays, *, max_buffer_bytes):
        if self._requested_dag_execution=='workgroup':
            from .gpu_workgroup import execute
            self._release_resident_dag()
            return execute(self,arrays,max_buffer_bytes=max_buffer_bytes,backend='metal')
        from .metal_buffers import execute_dag
        return execute_dag(self, arrays, max_buffer_bytes=max_buffer_bytes)

    def close(self):
        from .gpu_readback import release_host_spike_cache
        release_host_spike_cache(self)
        from .gpu_workgroup import close
        close(self)
        for handle in getattr(self, "handles", []):
            self.bridge.b2_metal_destroy(handle)
        self.handles = []
        self._resident_dag_bytes = 0
        self._metal_dag_metadata = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _cpu_mirror(self):
        """Benchmark control with identical f32 operations and temporal fusion."""
        from .gpu_functions import require_cpu_mirror
        require_cpu_mirror(self.model)
        if hasattr(self, "cpu_bridge"):
            return self.cpu_bridge
        head = _CPU_PRELUDE
        parts = [head]
        for kernel in self.plan.kernels:
            types = population_types(self.model, kernel.population)
            source = kernel.source.replace("#include <metal_stdlib>", "").replace("using namespace metal;", "")
            source = source.replace("kernel void", "void").replace("device ", "").replace("thread ", "")
            source = re.sub(r"\[\[(?:buffer\(\d+\)|thread_position_in_grid)\]\]", "", source)
            parts += [f"namespace p{kernel.population} {{", source, "}"]
            arguments = ", ".join(f"static_cast<{dtype} *>(data[{i}])" for i, dtype in enumerate(types))
            parts.append(f"""extern "C" void cpu_{kernel.entry}(void **data, uint workers) {{
    auto run = [&](uint rank) {{
        uint begin = uint(uint64_t({kernel.neurons})*rank/workers);
        uint end = uint(uint64_t({kernel.neurons})*(rank+1)/workers);
        for (uint i=begin; i<end; ++i) p{kernel.population}::{kernel.entry}({arguments}, i);
    }};
    std::vector<std::thread> threads;
    for (uint rank=1; rank<workers; ++rank) threads.emplace_back(run, rank);
    run(0);
    for (auto &thread: threads) thread.join();
}}""")
        source = "\n".join(parts)
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        path = self.directory / f"cpu-f32-mirror-{digest}.cpp"
        library = path.with_suffix(".dylib")
        path.write_text(source)
        if not library.exists():
            subprocess.run(["clang++", "-std=c++17", "-O3", "-ffp-contract=off", "-fno-fast-math",
                            *(["-dynamiclib"] if platform.system()=="Darwin" else ["-shared","-fPIC"]),
                            "-pthread", str(path), "-o", str(library)],
                           check=True, capture_output=True, text=True)
        self.cpu_bridge = ctypes.CDLL(str(library))
        for kernel in self.plan.kernels:
            function = getattr(self.cpu_bridge, f"cpu_{kernel.entry}")
            function.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
            function.restype = None
        return self.cpu_bridge

    def run(self, *, max_buffer_bytes=512*1024*1024, compute="metal", workers=1, dag_execution=None, dag_synchronization=None):
        if len(self.handles) != len(self.plan.kernels):
            raise RuntimeError("Metal executor is closed")
        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be a positive integer")
        from .metal_buffers import validate_mode, validate_synchronization
        self._requested_dag_execution = validate_mode(self.dag_execution if dag_execution is None else dag_execution)
        if compute=='metal' and self._requested_dag_execution in {'workgroup','indirect'} and not self.plan.dispatches:
            raise PlanValidationError(f'{self._requested_dag_execution.capitalize()} execution requires an explicit DAG')
        self._requested_dag_synchronization = validate_synchronization(
            self.dag_synchronization if dag_synchronization is None else dag_synchronization)
        self._metal_dag_report = None
        if self._resident_dag_bytes > max_buffer_bytes:
            self._release_resident_dag()
        if compute not in {"metal", "cpu-f32"}:
            raise ValueError("compute must be metal or cpu-f32")
        if type(workers) is not int or not 1 <= workers <= 256:
            raise ValueError("workers must be within 1..256")
        if self.plan.dispatches:
            from .metal_dag import run_dag
            result = run_dag(self, max_buffer_bytes=max_buffer_bytes, compute=compute, workers=workers)
            if compute == 'metal': result['metal_runtime'] = dict(self._metal_dag_report)
            return result
        cpu = self._cpu_mirror() if compute == "cpu-f32" else None
        results, metrics = [], []
        started = time.perf_counter()
        for handle, kernel in zip(self.handles, self.plan.kernels, strict=True):
            arrays, initial_last = population_arrays(self.model, kernel.population, kernel, max_buffer_bytes)
            n = kernel.neurons
            pointers = (ctypes.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
            sizes = (ctypes.c_uint64 * len(arrays))(*(a.nbytes for a in arrays))
            timing = (ctypes.c_double * 4)()
            error = ctypes.create_string_buffer(8192)
            if cpu is None:
                code = self.bridge.b2_metal_run(handle, pointers, sizes, len(arrays), n,
                                               sum(1 << i for i in (0, 2, 4, 5, 6, 7, 8, 9)),
                                               timing, error, len(error))
                if code:
                    raise RuntimeError(error.value.decode())
            else:
                cpu_started = time.perf_counter()
                getattr(cpu, f"cpu_{kernel.entry}")(pointers, workers)
                timing[1] = time.perf_counter() - cpu_started
            results.append(population_result(self.model, kernel.population, kernel, arrays, initial_last))
            metrics.append(dict(zip(("input_seconds", "command_seconds", "gpu_seconds", "readback_seconds"), timing, strict=True)))
        return {"populations": results, "synapses": [], "numeric_profile": METAL_PROFILE if cpu is None else "b2-cpu-f32-mirror-v0",
                "device": self.device_name if cpu is None else f"CPU f32 mirror ({workers} workers)",
                "timings": metrics, "run_seconds": time.perf_counter()-started,
                "compile_seconds": self.compile_seconds, "plan_sha256": self.plan.sha256,
                "rng_profile": self.plan.rng_profile}


def write_metal_results(model, result, directory):
    """Publish the existing typed result transport with an explicit f32 profile.

    Storage may be f64 to match Brian's public arrays; the metadata always
    identifies the actual float32 arithmetic. This is not reference-f64 output.
    """
    return _write_gpu_results(model,result,directory,engine="metal",numeric_profile=METAL_PROFILE)


def _write_gpu_results(model,result,directory,*,engine,numeric_profile):
    if result.get("numeric_profile") != numeric_profile:
        raise PlanValidationError(f"{engine} transport requires its own float32 result, not another backend or CPU control")
    if result.get("rng_profile") != (RNG_PROFILE if has_random(model["definition"]) else None):
        raise PlanValidationError("GPU result RNG profile does not match model")
    from .results import MAGIC, END, VERSION, ENDIAN_MARKER, NUMPY_DTYPES
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    final_time = number(model["run"]["start"]) + number(model["run"]["duration"])
    started = time.perf_counter()
    with (directory / "results.bin").open("wb") as output:
        output.write(struct.pack("<8sIIQQQ", MAGIC, VERSION, ENDIAN_MARKER,
                                 len(result["populations"]), model["instance"]["neuron_count"], 0))
        for pop, values in zip(model["definition"]["populations"], result["populations"], strict=True):
            monitor = pop["monitor"]
            output.write(struct.pack("<8Q", pop["count"], pop["steps"], len(monitor["record"]),
                                     len(monitor["variables"]), len(pop["states"]),
                                     len(values["indices"]), len(values["last_spikes"]),
                                     int(values["refractory"] is not None)))
            symbols = {s["name"]: s for s in pop["states"] + pop["parameters"] + pop.get("linked_variables",[])}
            for name in monitor["variables"]:
                output.write(np.asarray(values["trace"][name], dtype=NUMPY_DTYPES[symbols[name]["dtype"]]).tobytes())
            events = np.column_stack((values["spike_ticks"], values["indices"])).astype("<i8")
            output.write(events.tobytes())
            output.write(np.asarray(values["counts"], dtype="<i8").tobytes())
            output.write(np.asarray(values["last_spikes"], dtype="<i8").tobytes())
            for state in pop["states"]:
                output.write(np.asarray(values["states"][state["name"]], dtype=NUMPY_DTYPES[state["dtype"]]).tobytes())
            if values["refractory"] is not None:
                output.write(np.asarray(values["refractory"]["lastspike"], dtype="<f8").tobytes())
                output.write(np.asarray(values["refractory"]["not_refractory"], dtype="u1").tobytes())
        synapses = model["definition"]["synapses"]
        output.write(struct.pack("<Q", len(synapses)))
        for q, synapse in enumerate(synapses):
            values = result["synapses"][q]
            inst = model["instance"]["synapses"][q]
            edges = inst.get("topology", {}).get("edge_count", len(inst["source"]))
            output.write(struct.pack("<2Q", len(synapse["states"]), edges))
            for state in synapse["states"]:
                output.write(np.asarray(values["states"][state["name"]], dtype=NUMPY_DTYPES[state["dtype"]]).tobytes())
            output.write(struct.pack("<Q", values["events"]))
        output.write(struct.pack("<d8s", final_time, END))
        length = output.tell()
        output.seek(32)
        output.write(struct.pack("<Q", length))
    event_bytes = 0
    if any(pop.get('event_streams') for pop in result['populations']):
        from .results import EVENT_MAGIC, EVENT_END
        streams = [(p, event) for p, pop in enumerate(model["definition"]["populations"]) for event in pop["events"]]
        with (directory / "events.bin").open("wb") as output:
            output.write(EVENT_MAGIC + struct.pack("<Q", len(streams)))
            for p, event in streams:
                values = result["populations"][p]["event_streams"][event]
                output.write(struct.pack("<Q", len(values["ticks"])))
                output.write(np.column_stack((values["ticks"], values["indices"])).astype("<i8").tobytes())
            monitor_defs = [(p, monitor) for p,pop in enumerate(model['definition']['populations'])
                            for monitor in pop.get('event_monitors', [])]
            output.write(struct.pack('<Q',len(monitor_defs)))
            for p,monitor in monitor_defs:
                values=result['populations'][p]['event_monitors'][monitor['name']]
                output.write(struct.pack('<2Q',len(values['ticks']),len(monitor['variables'])))
                output.write(np.column_stack((values['ticks'],values['indices'])).astype('<i8').tobytes())
                symbols={s['name']:s for s in model['definition']['populations'][p]['states']+
                         model['definition']['populations'][p]['parameters']+model['definition']['populations'][p].get('linked_variables',[])}
                for name in monitor['variables']:
                    output.write(np.asarray(values['values'][name],dtype=NUMPY_DTYPES[symbols[name]['dtype']]).tobytes())
            output.write(EVENT_END)
            event_bytes = output.tell()
    summary = {"schema": "b2-result-dump-v3", "engine": engine,
               "initializations": result.get("initializations", []),
               "initialization_seconds": result.get("initialization_seconds", 0.0),
               "rng_profile": result.get("rng_profile"),
               "numeric_profile": numeric_profile, "device": result["device"], "event_dump_bytes": event_bytes,
               "plan_sha256": result["plan_sha256"], "dump_bytes": length,
               "population_count": len(result["populations"]),
               "neuron_count": model["instance"]["neuron_count"],
               "spike_count": sum(len(p["indices"]) for p in result["populations"]),
               "synaptic_events": sum(s["events"] for s in result.get("synapses", [])), "final_time_seconds": final_time,
               "timings": {"initialization_seconds": sum(t["input_seconds"] for t in result["timings"]),
                           "simulation_and_recording_seconds": sum(t["command_seconds"] for t in result["timings"]),
                           "dump_write_seconds": time.perf_counter()-started},
               f"{engine}_timings": result["timings"], "run_seconds": result["run_seconds"],
               "compile_seconds": result["compile_seconds"]}
    if "cuda_runtime" in result:
        summary["cuda_runtime"] = result["cuda_runtime"]
    if "metal_runtime" in result:
        summary["metal_runtime"] = result["metal_runtime"]
    (directory / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary
