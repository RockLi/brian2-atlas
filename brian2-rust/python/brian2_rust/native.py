"""Generate a model-specialized, std-only Rust executable from validated AtlasIR."""

from __future__ import annotations

import array
import hashlib
import json
import struct
from pathlib import Path

from .protocol import CURRENT_SCHEMA, migrate_model
from .codegen_sums import usize_sum
from .encoded_array import EncodedArray, IndexArray
from .plan import _derive_execution_plan, verify_execution_plan
from .planner import (
    PARALLEL_EVENT_MIN_EDGES,
    PARALLEL_MIN_TASK_ITEMS,
    PARALLEL_MIN_TOTAL_WORK,
    PARALLEL_TARGET_TASK_WORK,
    SOURCE_BATCH_MIN_EDGES,
    TargetParallelPlan,
    _explicit_topology,
    _has_runtime_random,
    _number,
    _plastic_pathway,
    _schedule_can_contract,
    _schedule_can_fuse_bundles,
    _schedule_code_node,
    _schedule_effects_conflict,
    _synapse_edge_count,
    _target_parallel_route,
    _uses_v6,
    _v7_fused_summed_groups,
    _v7_summed_final_only,
    code_work,
    post_synapse_work,
    _v7_regular_parallel_capable,
    fixed_schedule_is_semantically_equivalent,
    parallel_phase_capable,
    parallel_task_limit,
    target_parallel_pathway,
    target_parallel_plan,
)

TIMED_ARRAY_INLINE_VALUES = 4096


def _timed_array_blob(node):
    payload = b"".join(
        struct.pack("<d", _number(value)) for value in node["values"])
    digest = hashlib.sha256(payload).hexdigest()
    return f"B2_TIMED_ARRAY_{digest.upper()}", f"timed-array-{digest}.bin", payload


def timed_array_blobs(value):
    """Return unique large TimedArray payloads referenced by a model."""
    blobs = {}

    def visit(item):
        if isinstance(item, dict):
            if (item.get("op") == "timed_array" and
                    len(item["values"]) > TIMED_ARRAY_INLINE_VALUES):
                symbol, filename, payload = _timed_array_blob(item)
                blobs[filename] = (symbol, payload)
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return [(symbol, filename, payload)
            for filename, (symbol, payload) in sorted(blobs.items())]


def write_timed_array_blobs(model, directory):
    blobs = timed_array_blobs(model)
    for _symbol, filename, payload in blobs:
        (directory / filename).write_bytes(payload)
    return blobs

# Keep state updates free of spike-vector pushes so LLVM can vectorize the
# arithmetic loop; threshold, recording, synapses, and reset are distinct
# scheduled phases for every model size. Delayed queues store source IDs once;
# delivery expands small serial batches into a flat edge scratch buffer. Large
# safe batches use source CSR partitioned by target owner.


PRE_SYNAPSE_INLINE_MIN_EDGES = 500_000


def _join_source(lines):
    body = "\n".join(lines[1:]) + "\n"
    runtime = lines[0]
    # Keep unrelated models' generated source (and artifact identity) unchanged.
    # The cached binomial helper also calls the cached normal helper.
    if ("counter_normal_cached(" in body or
            "counter_binomial_cached(" in body or
            "counter_binomial_weighted(" in body):
        for function in ("counter_normal_cached", "counter_binomial_cached"):
            runtime = runtime.replace(
                f"fn {function}(", f"#[inline(always)]\nfn {function}(")
    if "counter_binomial_weighted(" in body:
        runtime += WEIGHTED_BINOMIAL_RUNTIME
    return runtime + "\n" + body


def number(bits):
    return f"f64::from_bits(0x{bits}u64)"




def _constant_number(node):
    if node["op"] == "literal":
        return _number(node["bits"])
    if node["op"] == "integer":
        return int(node["value"])
    if node["op"] == "cast":
        # Preserve the common frontend promotion of a small integer literal.
        # It is exact in both the f64 and GPU f32 profiles. Other casts may
        # saturate, wrap or round; they must go through normal typed lowering.
        arg = node["arg"]
        if node["dtype"] == "f64" and arg["op"] in {"literal", "integer"}:
            value = _constant_number(arg)
            if arg["op"] == "literal" or abs(value) <= 2**24:
                return value
        return None
    if node["op"] == "neg":
        value = _constant_number(node["arg"])
        return None if value is None else -value
    return None


def _rust_dtype(symbol):
    dtype = symbol["dtype"]
    if dtype not in {"bool", "f32", "f64", "i32", "i64", "u32", "u64"}:
        raise ValueError(f"unsupported AOT storage dtype: {dtype}")
    return "u8" if dtype == "bool" else dtype


def _dump_dtype(symbol):
    """Name the result writer without colliding with structural dump_u64."""
    return "u64_values" if symbol["dtype"] == "u64" else _rust_dtype(symbol)


def _storage_load(symbol, expression):
    """Convert a compact SoA cell into its typed expression value."""
    return f"({expression} != 0)" if symbol["dtype"] == "bool" else expression


def _storage_store(symbol, expression):
    """Convert a typed expression value into its compact SoA cell."""
    return f"u8::from({expression})" if symbol["dtype"] == "bool" else expression


def _dtype_size(symbol):
    return {
        "bool": 1, "f32": 4, "f64": 8,
        "i32": 4, "i64": 8, "u32": 4, "u64": 8,
    }[symbol["dtype"]]






def _initializer_call(initializer, *, ticks=False):
    suffix = "_ticks" if ticks else ""
    if initializer["kind"] == "clipped_normal":
        option = lambda value: ("None" if value is None else
                                f"Some({number(value)})")
        arguments = (
            f"{initializer['stream']}u64, {number(initializer['mean'])}, "
            f"{number(initializer['std'])}, {option(initializer['minimum'])}, "
            f"{option(initializer['maximum'])}")
        return f"materialize_clipped_normal{suffix}", arguments
    if initializer["kind"] == "uniform":
        arguments = (
            f"{initializer['stream']}u64, {number(initializer['minimum'])}, "
            f"{number(initializer['maximum'])}")
        return f"materialize_uniform{suffix}", arguments
    raise ValueError("unsupported procedural initializer")


class Block:
    def __init__(self, inputs, dtypes=None, rng=None, functions=None,
                 temp_prefix="r"):
        self.inputs = dict(inputs)
        self.symbols = dict(inputs)
        self.dtypes = {} if dtypes is None else dict(dtypes)
        self.input_snapshot = None
        self.assumptions = {}
        self.common = {}
        self.common_reads = {}
        self.counter = 0
        self.lines = []
        self.rng = rng
        self.functions = {} if functions is None else functions
        self.temp_prefix = temp_prefix
        self.expression_overrides = {}

    def dtype(self, node):
        op = node["op"]
        if op in {"literal", "rand", "randn", "binomial", "poisson",
                  "bool_to_f64", "index_to_f64", "tick_to_f64", "f32_to_f64"}:
            return "f64"
        if op == "boolean":
            return "bool"
        if op == "integer":
            return node["dtype"]
        if op in {"cast", "f64_to_f32"}:
            return node.get("dtype", "f32")
        if op == "load":
            # The compact v6 generator predates explicit symbol dtypes and is
            # selected only for all-f64 models. The general v7 path supplies
            # every public dtype here.
            return self.dtypes.get(node["name"], "f64")
        if op == "call":
            return self.functions[node["function"]]["return_dtype"]
        if op in {"timestep", "tick_offset"}:
            return "tick"
        if op in {"not", "eq", "ne", "gt", "ge", "lt", "le", "and", "or"}:
            return "bool"
        if op in {"neg", "abs", "add", "sub", "mul", "mod", "floor_div"}:
            child = node["arg"] if "arg" in node else node["left"]
            dtype = self.dtype(child)
            if dtype in {"i32", "i64", "u32", "u64"}:
                return dtype
        return "f64"

    def temp(self):
        self.counter += 1
        return f"{self.temp_prefix}{self.counter}"

    def expr(self, node, indent="        "):
        if self.expression_overrides:
            from .summed_cache import expression_key
            cached = self.expression_overrides.get(expression_key(node))
            if cached is not None:
                return cached
        op = node["op"]
        if op == "load":
            if node["name"] in self.assumptions:
                return self.assumptions[node["name"]]
            if self.input_snapshot is not None and node["name"] in self.input_snapshot:
                return self.input_snapshot[node["name"]]
            return self.symbols[node["name"]]
        if op == "literal":
            return number(node["bits"])
        if op == "boolean":
            return "true" if node["value"] else "false"
        if op == "integer":
            return f"{node['value']}{node['dtype']}"
        if op == "call":
            contract = self.functions[node["function"]]
            values = [self.expr(argument, indent)
                      for argument in node["arguments"]]
            native = contract.get("backend_implementations", {}).get("cpu")
            if native is not None:
                arguments = []
                for argument, value in zip(
                        contract["arguments"], values, strict=True):
                    arguments.append(
                        f"u8::from({value})" if argument["dtype"] == "bool"
                        else value)
                call = (f"unsafe {{ {native['symbol']}({', '.join(arguments)}) }}")
                if contract["return_dtype"] == "bool":
                    call = f"({call} != 0)"
                return call
            previous_symbols = {}
            previous_dtypes = {}
            previous_assumptions = {}
            previous_snapshot = {}
            for argument, value in zip(contract["arguments"], values,
                                       strict=True):
                name = argument["name"]
                previous_symbols[name] = self.symbols.get(name)
                previous_dtypes[name] = self.dtypes.get(name)
                previous_assumptions[name] = self.assumptions.pop(name, None)
                if self.input_snapshot is not None:
                    previous_snapshot[name] = self.input_snapshot.pop(name, None)
                self.symbols[name] = value
                self.dtypes[name] = argument["dtype"]
            # Function-local CSE keys mention formal argument names and cannot
            # be shared by calls with different actual arguments.
            outer_common, outer_common_reads = self.common, self.common_reads
            self.common, self.common_reads = {}, {}
            if contract["body"] is None:
                raise ValueError(
                    f"Function {node['function']} has no AOT implementation")
            result = self.expr(contract["body"], indent)
            self.common, self.common_reads = outer_common, outer_common_reads
            for name, value in previous_symbols.items():
                if value is None:
                    del self.symbols[name]
                else:
                    self.symbols[name] = value
                if previous_dtypes[name] is None:
                    del self.dtypes[name]
                else:
                    self.dtypes[name] = previous_dtypes[name]
                if previous_assumptions[name] is not None:
                    self.assumptions[name] = previous_assumptions[name]
                if (self.input_snapshot is not None and
                        previous_snapshot[name] is not None):
                    self.input_snapshot[name] = previous_snapshot[name]
            return result
        if op in {"rand", "randn", "binomial", "poisson"}:
            if self.rng is None:
                raise ValueError("random expression outside a vector execution domain")
            arguments = (f"{self.rng['seed']}, {node['stream']}u64, "
                         f"{self.rng['tick']}, {self.rng['index']}")
            if op == "rand":
                return f"counter_uniform({arguments})"
            if op == "poisson":
                return (f"counter_poisson({arguments}, "
                        f"{self.expr(node['lambda'], indent)})")
            cache = self.rng.get("normal_cache")
            if op == "randn":
                function = "counter_normal_cached" if cache else "counter_normal"
                suffix = f", &mut {cache}" if cache else ""
                return f"{function}({arguments}{suffix})"
            function = "counter_binomial_cached" if cache else "counter_binomial"
            suffix = f", &mut {cache}" if cache else ""
            return (f"{function}({arguments}, {node['n']}u64, "
                    f"{self.expr(node['p'], indent)}, "
                    f"{str(node.get('approximate', True)).lower()}{suffix})")
        key = json.dumps(node, sort_keys=True, separators=(",", ":"))
        if key in self.common:
            return self.common[key]
        if op == "bool_to_f64":
            return f"f64::from({self.expr(node['arg'], indent)})"
        if op == "f32_to_f64":
            return f"({self.expr(node['arg'], indent)} as f64)"
        if op == "f64_to_f32":
            return f"checked_f32({self.expr(node['arg'], indent)})"
        if op == "cast":
            source = self.dtype(node["arg"])
            value = self.expr(node["arg"], indent)
            target = node["dtype"]
            if source == "bool":
                value = f"u8::from({value})"
            return f"({value} as {target})"
        if op in {"index_to_f64", "tick_to_f64"}:
            # Logical values use an exact f64 representation only after the
            # independently validated conversion boundary.
            return self.expr(node["arg"], indent)
        if op == "timestep":
            return (f"checked_timestep({self.expr(node['time'], indent)}, "
                    f"{self.expr(node['dt'], indent)})")
        if op == "tick_offset":
            return (f"checked_tick_offset({self.expr(node['tick'], indent)}, "
                    f"{node['offset']}i64)")
        if op == "trunc":
            return f"checked_trunc({self.expr(node['arg'], indent)})"
        unary = {
            "abs": "abs", "arccos": "acos", "arcsin": "asin",
            "arctan": "atan", "ceil": "ceil", "cos": "cos", "cosh": "cosh",
            "exp": "exp", "expm1": "exp_m1", "floor": "floor", "log": "ln",
            "log10": "log10", "log1p": "ln_1p", "sin": "sin", "sinh": "sinh",
            "sqrt": "sqrt", "tan": "tan", "tanh": "tanh",
        }
        if op == "abs" and self.dtype(node) in {"i32", "i64"}:
            expression = f"{self.expr(node['arg'], indent)}.wrapping_abs()"
        elif op in unary:
            expression = f"{self.expr(node['arg'], indent)}.{unary[op]}()"
        elif op == "exprel":
            expression = f"exprel({self.expr(node['arg'], indent)})"
        elif op == "sign":
            arg = self.expr(node["arg"], indent)
            expression = f"if {arg} > 0.0 {{ 1.0 }} else if {arg} < 0.0 {{ -1.0 }} else {{ 0.0 }}"
        elif op == "neg":
            arg = self.expr(node["arg"], indent)
            expression = (f"{arg}.wrapping_neg()"
                          if self.dtype(node) in {"i32", "i64"} else f"-{arg}")
        elif op == "not":
            expression = f"!{self.expr(node['arg'], indent)}"
        elif op == "clip":
            value = self.expr(node["value"], indent)
            minimum = self.expr(node["min"], indent)
            maximum = self.expr(node["max"], indent)
            expression = f"{value}.max({minimum}).min({maximum})"
        elif op == "timed_array":
            time = self.expr(node["time"], indent)
            epsilon = number(node["epsilon"])
            row = (f"((({time} / {epsilon} + 0.5) / "
                   f"{node['upsampling']}f64) as usize).min({node['rows'] - 1})")
            large = len(node["values"]) > TIMED_ARRAY_INLINE_VALUES
            if large:
                symbol, _filename, _payload = _timed_array_blob(node)
            else:
                # Round-trippable decimal literals avoid one const-evaluation
                # call to f64::from_bits per sample. Finite f64 values,
                # including signed zero, retain their exact bits.
                values = ",".join(
                    repr(_number(value)) for value in node["values"])
            if node["columns"] is None:
                expression = (f"timed_array_f64({symbol}, {row})" if large
                              else f"[{values}][{row}]")
            else:
                index = self.expr(node["index"], indent)
                offset = f"{row} * {node['columns']} + ({index} as usize)"
                expression = (f"timed_array_f64({symbol}, {offset})" if large
                              else f"[{values}].get({offset}).copied()"
                                   ".unwrap_or(f64::NAN)")
        else:
            left, right = self.expr(node["left"], indent), self.expr(node["right"], indent)
            operators = {
                "add": "+", "sub": "-", "mul": "*", "div": "/",
                "eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<",
                "le": "<=", "and": "&&", "or": "||",
            }
            dtype = self.dtype(node)
            if op in {"add", "sub", "mul"} and dtype in {
                    "i32", "i64", "u32", "u64"}:
                expression = f"{left}.wrapping_{op}({right})"
            elif (op == "mul" and dtype == "f64" and self.rng is not None
                  and self.rng.get("normal_cache")
                  and node["left"]["op"] == "binomial"
                  and node["right"]["op"] == "timed_array"
                  and any(value in {"0000000000000000", "8000000000000000"}
                          for value in node["right"]["values"])):
                # A sparse time-varying input often multiplies exact binomial
                # samples by zero. The helper preserves the signed product and
                # leaves Gaussian samples unchanged. Dense/unweighted samplers
                # keep their existing source and avoid an extra runtime branch.
                expression = (left.replace("counter_binomial_cached(",
                                           "counter_binomial_weighted(", 1)[:-1]
                              + f", {right})")
            elif op == "pow":
                exponent = _constant_number(node["right"])
                if (exponent is not None and float(exponent).is_integer() and
                        -(2**31) <= exponent < 2**31):
                    expression = f"{left}.powi({int(exponent)})"
                else:
                    expression = f"{left}.powf({right})"
            elif op == "mod" and dtype in {"i32", "i64", "u32", "u64"}:
                expression = f"checked_{dtype}_mod({left}, {right})"
            elif op == "mod":
                expression = f"{left} - {right} * ({left} / {right}).floor()"
            elif op == "floor_div" and dtype in {"i32", "i64", "u32", "u64"}:
                expression = f"checked_{dtype}_floor_div({left}, {right})"
            elif op == "floor_div":
                expression = f"({left} / {right}).floor()"
            else:
                expression = f"{left} {operators[op]} {right}"
        name = self.temp()
        self.lines.append(f"{indent}let {name} = {expression};")
        self.common[key] = name
        self.common_reads[key] = expression_loads(node)
        return name

    def statements(self, statements, indent="        "):
        for statement in statements:
            condition = statement.get("condition")
            if condition is None:
                value = self.expr(statement["value"], indent)
            else:
                old = self.symbols.get(statement["target"])
                guard, result = self.symbols[condition], self.temp()
                self.lines.append(f"{indent}let {result} = if {guard} {{")
                # The guarded branch proves the condition true. Brian's state updater
                # form can still contain ``int(not_refractory)``; substitute
                # that fact so the native loop does not multiply by a mask it
                # has already branched on.
                # Temporaries declared inside the branch cannot escape its
                # lexical scope. Assumptions also change expression meaning,
                # so neither import nor export CSE entries across this boundary.
                outer_common, outer_reads = self.common, self.common_reads
                self.common, self.common_reads = {}, {}
                self.assumptions[condition] = "true"
                value = self.expr(statement["value"], indent + "    ")
                del self.assumptions[condition]
                self.common, self.common_reads = outer_common, outer_reads
                self.lines.append(f"{indent}    {value}")
                default = old if old is not None else ("false" if statement["dtype"] == "bool" else "f64::NAN")
                self.lines.append(f"{indent}}} else {{ {default} }};")
                value = result
            self.symbols[statement["target"]] = value
            self.dtypes[statement["target"]] = statement["dtype"]
            # Retain pure subexpressions across consecutive assignments when
            # their inputs did not change. This matters for event-driven STDP,
            # where independent traces often share the same decay factor.
            stale = [key for key, reads in self.common_reads.items()
                     if statement["target"] in reads]
            for key in stale:
                del self.common[key]
                del self.common_reads[key]


def expression_loads(node):
    if node["op"] == "load":
        return {node["name"]}
    result = set()
    for child in node.values():
        if isinstance(child, dict):
            result |= expression_loads(child)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, dict):
                    result |= expression_loads(item)
    return result












def _v6_target_event_kernel(model, code):
    """Generate an isolated target-owner on_pre kernel and call template."""
    d = model["definition"]
    synapse = d["synapses"]
    arguments = ["parallel: &Parallel", "owner_csr: &TargetOwnerCsr",
                 "active: &[usize]", "target_index: &[u32]",
                 "time: f64", "dt: f64", "edge_count: usize"]
    calls = ["&parallel", "target_owner.as_ref().unwrap()", "{active}",
             "&target_index", "time", "dt", "edge_count"]
    declarations, overrides = [], {}
    state_positions = {symbol["name"]: position for position, symbol in
                       enumerate(d["states"])}
    synapse_positions = {symbol["name"]: position for position, symbol in
                         enumerate(synapse["states"])}
    for position, symbol in enumerate(d["states"]):
        arguments.append(f"state_{position}: &mut [f64]")
        calls.append(f"&mut state_{position}")
        pointer = f"state_{position}_event_ptr"
        declarations.append(
            f"let {pointer} = state_{position}.as_mut_ptr() as usize;")
        for alias, state in synapse["pre_state_aliases"].items():
            if state == symbol["name"]:
                overrides[alias] = (
                    f"unsafe {{ *(({pointer} as *const f64).add(source)) }}")
        for alias, state in synapse["post_state_aliases"].items():
            if state == symbol["name"]:
                overrides[alias] = (
                    f"unsafe {{ *(({pointer} as *const f64).add(target)) }}")
    for position, symbol in enumerate(synapse["states"]):
        arguments.append(f"syn_state_{position}: &mut [f64]")
        calls.append(f"&mut syn_state_{position}")
        pointer = f"syn_state_{position}_event_ptr"
        declarations.append(
            f"let {pointer} = syn_state_{position}.as_mut_ptr() as usize;")
        overrides[symbol["name"]] = (
            f"unsafe {{ *(({pointer} as *const f64).add(edge)) }}")
    for position, symbol in enumerate(synapse["parameters"]):
        if symbol["index_domain"] == "scalar":
            arguments.append(f"syn_parameter_{position}: f64")
            calls.append(f"syn_parameter_{position}")
        else:
            arguments.append(f"syn_parameter_{position}: &[f64]")
            calls.append(f"&syn_parameter_{position}")
    if d["refractory"] is not None:
        arguments.append("not_refractory: &[u8]")
        calls.append("&not_refractory")
        declarations.append(
            "let not_refractory_event_ptr = not_refractory.as_ptr() as usize;")
        overrides["not_refractory_post"] = (
            "unsafe { *((not_refractory_event_ptr as *const u8).add(target)) } != 0")
    scalar, vector, symbols = code_block(
        model, code, "synapse", "edge", overrides=overrides)
    writes = []
    for name in code["effects"]["writes"]:
        if name in synapse["post_state_aliases"]:
            state = synapse["post_state_aliases"][name]
            position = state_positions[state]
            writes.append(
                f"unsafe {{ *((state_{position}_event_ptr as *mut f64).add(target)) = "
                f"{symbols[name]}; }}")
        else:
            position = synapse_positions[name]
            writes.append(
                f"unsafe {{ *((syn_state_{position}_event_ptr as *mut f64).add(edge)) = "
                f"{symbols[name]}; }}")
    lines = ["#[inline(never)]",
             f"fn target_on_pre({', '.join(arguments)}) {{"]
    lines += ["    " + line for line in declarations]
    lines += ["    " + line for line in scalar]
    lines += ["    parallel.for_lanes(move |owner| {",
              "        let owner_offsets = &owner_csr.offsets[owner];",
              "        let owner_edges = &owner_csr.edges[owner];",
              "        for &source in active {",
              "            for &edge_index in &owner_edges[owner_offsets[source]..owner_offsets[source+1]] {",
              "                let edge = edge_index as usize;",
              "                let target = target_index[edge] as usize;"]
    lines += ["        " + line for line in vector]
    lines += ["                " + line for line in writes]
    lines += ["            }", "        }", "    });", "}"]
    return "\n".join(lines), f"target_on_pre({', '.join(calls)});"


def input_expressions(model, domain, index, overrides=None):
    definition = model["definition"]
    values = {"dt": "dt", "t": "time"}
    n = model["instance"]["neuron_count"]
    if domain == "neuron":
        values.update(i=f"{index} as f64", N=f"{n}.0")
        for pos, symbol in enumerate(definition["states"]):
            values[symbol["name"]] = f"state_{pos}[{index}]"
        for pos, symbol in enumerate(definition["parameters"]):
            suffix = "" if symbol["index_domain"] == "scalar" else f"[{index}]"
            values[symbol["name"]] = f"parameter_{pos}{suffix}"
        if definition["refractory"] is not None:
            values.update(lastspike=f"lastspike[{index}]",
                          not_refractory=f"not_refractory[{index}] != 0")
    else:
        syn = definition["synapses"]
        values.update(i="source as f64", j="target as f64", N="edge_count as f64",
                      N_pre=f"{syn['source_count']}.0",
                      N_post=f"{syn['target_count']}.0")
        states = {symbol["name"]: pos for pos, symbol in enumerate(definition["states"])}
        for alias, state in syn["pre_state_aliases"].items():
            values[alias] = f"state_{states[state]}[source]"
        for alias, state in syn["post_state_aliases"].items():
            values[alias] = f"state_{states[state]}[target]"
        for pos, symbol in enumerate(syn["states"]):
            values[symbol["name"]] = f"syn_state_{pos}[edge]"
        for pos, symbol in enumerate(syn["parameters"]):
            suffix = "" if symbol["index_domain"] == "scalar" else "[edge]"
            values[symbol["name"]] = f"syn_parameter_{pos}{suffix}"
        if definition["refractory"] is not None:
            values["not_refractory_post"] = "not_refractory[target] != 0"
    values.update(overrides or {})
    return values


def vector_inputs(code, inputs):
    """Return model inputs whose pre-phase value the vector block can read."""
    used = set()

    def visit(node):
        if node["op"] == "load":
            if node["name"] in inputs:
                used.add(node["name"])
            return
        for child in node.values():
            if isinstance(child, dict):
                visit(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, dict):
                        visit(item)

    for statement in code["vector"]:
        visit(statement["value"])
        condition = statement.get("condition")
        if condition in inputs:
            used.add(condition)
        # A guarded assignment keeps the old target value on inactive lanes.
        if condition is not None and statement["target"] in inputs:
            used.add(statement["target"])
    return used


def code_block(model, code, domain, index="i", overrides=None):
    inputs = input_expressions(model, domain, index, overrides)
    functions = {function["name"]: function
                 for function in model["definition"].get("functions", [])}
    block = Block(inputs, functions=functions)
    block.statements(code["scalar"], "        ")
    scalar = block.lines
    block.lines = []
    # Snapshot only model inputs that the vector block reads before any state
    # is committed. This shares repeated loads and preserves the pre-stage
    # state semantics without relying on LLVM to discard unrelated arrays.
    used = vector_inputs(code, inputs) - {
        statement["target"] for statement in code["scalar"]}
    for pos, (name, expression) in enumerate(inputs.items()):
        if name not in used:
            continue
        local = f"input_{pos}"
        block.lines.append(f"        let {local} = {expression};")
        block.symbols[name] = local
    if code["kind"] in {"state_update", "synapse_state_update"}:
        # Integrator stages share the pre-phase model inputs even after private
        # temporaries are fused into their final assignments; generated stage
        # temporaries continue to resolve to their latest local values.
        block.input_snapshot = {name: block.symbols[name] for name in used}
    block.statements(code["vector"])
    return scalar, block.lines, block.symbols


def update_kernel(model, parallel_capable, fuse_threshold=False):
    d = model["definition"]
    args, call = ["parallel: &Parallel"], ["&parallel"]
    for pos, _ in enumerate(d["states"]):
        args.append(f"state_{pos}: &mut [f64]"); call.append(f"&mut state_{pos}")
    for pos, symbol in enumerate(d["parameters"]):
        scalar = symbol["index_domain"] == "scalar"
        args.append(f"parameter_{pos}: " + ("f64" if scalar else "&[f64]"))
        call.append(f"parameter_{pos}" if scalar else f"&parameter_{pos}")
    if d["refractory"] is not None:
        args += ["lastspike: &mut [f64]", "not_refractory: &mut [u8]",
                 ("refractory_until: &mut [usize]" if fuse_threshold else
                  "refractory_until: &[usize]")]
        call += ["&mut lastspike", "&mut not_refractory",
                 ("&mut refractory_until" if fuse_threshold else
                  "&refractory_until")]
        if fuse_threshold:
            args.append("period_ticks: usize")
            call.append("period_ticks")
    if fuse_threshold:
        args.append("fired_lanes: &mut [Vec<usize>]")
        call.append("&mut fired_lanes")
    args += ["tick: usize", "time: f64", "dt: f64"]; call += ["tick", "time", "dt"]
    lines = ["#[inline(always)]", "fn state_update(" + ", ".join(args) + ") -> bool {"]
    for pos, _ in enumerate(d["states"]): lines.append(f"assert_eq!(state_{pos}.len(), N);")
    for pos, symbol in enumerate(d["parameters"]):
        if symbol["index_domain"] != "scalar": lines.append(f"assert_eq!(parameter_{pos}.len(), N);")
    if d["refractory"] is not None:
        lines += ["assert_eq!(lastspike.len(), N);", "assert_eq!(not_refractory.len(), N);",
                  "assert_eq!(refractory_until.len(), N);"]
    code = d["code_objects"][0]
    scalar, vector, symbols = code_block(model, code, "neuron")
    work = code_work(code)
    indices = {symbol["name"]: pos for pos, symbol in enumerate(d["states"])}
    if not parallel_capable:
        lines += scalar + ["for i in 0..N {"]
        if d["refractory"] is not None:
            lines += ["    not_refractory[i] = u8::from(tick >= refractory_until[i]);"]
        lines += vector
        for name in code["effects"]["writes"]:
            lines.append(f"    state_{indices[name]}[i] = {symbols[name]};")
        lines += ["}", "false", "}"]
        return "\n".join(lines), "state_update(" + ", ".join(call) + ")"
    lines += [f"if !parallel.is_parallel(N, {work}) {{"] + ["    " + line for line in scalar]
    lines += ["    for i in 0..N {"]
    if d["refractory"] is not None:
        lines += ["        not_refractory[i] = u8::from(tick >= refractory_until[i]);"]
    lines += ["    " + line for line in vector]
    for name in code["effects"]["writes"]:
        lines.append(f"        state_{indices[name]}[i] = {symbols[name]};")
    lines += ["    }", "    return false;", "}"]

    overrides = {}
    for pos, symbol in enumerate(d["states"]):
        lines.append(f"let state_{pos}_ptr = state_{pos}.as_mut_ptr() as usize;")
        overrides[symbol["name"]] = (
            f"unsafe {{ *((state_{pos}_ptr as *const f64).add(i)) }}")
    for pos, symbol in enumerate(d["parameters"]):
        if symbol["index_domain"] != "scalar":
            lines.append(f"let parameter_{pos}_ptr = parameter_{pos}.as_ptr() as usize;")
            overrides[symbol["name"]] = (
                f"unsafe {{ *((parameter_{pos}_ptr as *const f64).add(i)) }}")
    if d["refractory"] is not None:
        lines += ["let lastspike_ptr = lastspike.as_mut_ptr() as usize;",
                  "let not_refractory_ptr = not_refractory.as_mut_ptr() as usize;",
                  ("let refractory_until_ptr = refractory_until.as_mut_ptr() as usize;"
                   if fuse_threshold else
                   "let refractory_until_ptr = refractory_until.as_ptr() as usize;")]
        overrides.update(
            lastspike="unsafe { *((lastspike_ptr as *const f64).add(i)) }",
            not_refractory=(
                "unsafe { *((not_refractory_ptr as *const u8).add(i)) } != 0"))
    parallel_scalar, parallel_vector, parallel_symbols = code_block(
        model, code, "neuron", overrides=overrides)
    if parallel_scalar != scalar:
        raise ValueError("parallel neuron scalar lowering differs from serial lowering")
    if fuse_threshold:
        threshold = next(code for code in d["code_objects"]
                         if code["kind"] == "threshold")
        threshold_scalar, threshold_vector, threshold_symbols = code_block(
            model, threshold, "neuron", overrides=overrides)
        lines += ["let fired_lanes_ptr = fired_lanes.as_mut_ptr() as usize;"]
        worker_call = f"parallel.for_each_ranked(N, {work}, move |rank, start, end| {{"
    else:
        threshold_scalar, threshold_vector, threshold_symbols = [], [], {}
        worker_call = f"parallel.for_each(N, {work}, move |start, end| {{"
    lines += scalar + threshold_scalar + [worker_call]
    if fuse_threshold:
        lines += ["    let lane_fired = unsafe { &mut *((fired_lanes_ptr as *mut Vec<usize>).add(rank)) };",
                  "    lane_fired.clear();"]
    lines += [
                       "    for i in start..end {"]
    if d["refractory"] is not None:
        lines.append(
            "        unsafe { *((not_refractory_ptr as *mut u8).add(i)) = "
            "u8::from(tick >= *((refractory_until_ptr as *const usize).add(i))); }")
    lines += ["    " + line for line in parallel_vector]
    for name in code["effects"]["writes"]:
        lines.append(
            f"        unsafe {{ *((state_{indices[name]}_ptr as *mut f64).add(i)) = "
            f"{parallel_symbols[name]}; }}")
    if fuse_threshold:
        lines += ["    " + line for line in threshold_vector]
        condition = threshold_symbols["_cond"]
        gate = (f"{condition} && unsafe {{ *((not_refractory_ptr as *const u8).add(i)) }} != 0"
                if d["refractory"] is not None else condition)
        lines += [f"        if {gate} {{", "            lane_fired.push(i);"]
        if d["refractory"] is not None:
            lines += ["            unsafe { *((lastspike_ptr as *mut f64).add(i)) = time; }",
                      "            unsafe { *((not_refractory_ptr as *mut u8).add(i)) = 0; }",
                      "            unsafe { *((refractory_until_ptr as *mut usize).add(i)) = tick + period_ticks; }"]
        lines.append("        }")
    lines += ["    }", "});", str(fuse_threshold).lower(), "}"]
    return "\n".join(lines), "state_update(" + ", ".join(call) + ")"


def generate_source_v6(model, *, choices=None):
    d, inst, run = model["definition"], model["instance"], model["run"]
    n, steps = inst["neuron_count"], run["steps"]
    states, parameters = d["states"], d["parameters"]
    has_ref, has_syn = d["refractory"] is not None, d["synapses"] is not None
    edge_count = len(inst["synapses"]["source"]) if has_syn else 0
    delay_ticks = inst["synapses"]["delay_ticks"] if has_syn else []
    uniform_delay = (delay_ticks[0] if has_syn and
                     all(value == delay_ticks[0] for value in delay_ticks) else None)
    delay_groups = sorted(set(delay_ticks), reverse=True) if has_syn else []
    monitor_states = {s["name"]: i for i, s in enumerate(states)}
    code_by_kind = {code["kind"]: code for code in d["code_objects"]}
    synapse_state_positions = ({symbol["name"]: pos for pos, symbol in
                                enumerate(d["synapses"]["states"])} if has_syn else {})
    if choices is None:
        raise ValueError("compact emitter requires CPU plan choices")
    source_batch_min_edges = choices["source_batch_min_edges"]
    pack_uniform_edges = choices["pack_uniform_edges"]
    state_work = choices["state_work"]
    parallel_state_capable = choices["parallel_state_capable"]
    parallel_event_capable = choices["parallel_event_capable"]
    parallel_capable = choices["parallel_capable"]
    kernel, kernel_call = update_kernel(
        model, parallel_capable, fuse_threshold=parallel_event_capable)
    if parallel_event_capable:
        event_kernel, event_kernel_call = _v6_target_event_kernel(
            model, code_by_kind["synapses"])
    else:
        event_kernel = event_kernel_call = ""
    timed_array_constants = [
        f'static {symbol}: &[u8] = include_bytes!("{filename}");'
        for symbol, filename, _payload in timed_array_blobs(model)]
    lines = [RUNTIME, kernel, event_kernel, f"const N: usize = {n};", f"const STEPS: usize = {steps};",
             *timed_array_constants,
             f"const DT_BITS: u64 = 0x{run['dt']};", "", "fn execute(mut data: Reader, output: &Path, initialization_started: Instant, input: Option<SpikeInput>) -> Result<()> {",
             "    check(input.is_none(), \"spike input requires a multi-population artifact\")?;",
             "    let dt = data.f64()?;", "    check(dt.to_bits() == DT_BITS, \"instance dt mismatch\")?;",
             "    check(data.usize()? == STEPS && data.usize()? == N, \"instance shape mismatch\")?;"]
    for pos, _ in enumerate(states):
        lines.append(f"    let mut state_{pos} = data.f64_vec(N)?;")
    for pos, symbol in enumerate(parameters):
        length = 1 if symbol["index_domain"] == "scalar" else n
        lines.append(f"    let parameter_{pos} = data.f64()?;" if length == 1 and symbol["index_domain"] == "scalar" else f"    let parameter_{pos} = data.f64_vec({length})?;")
    if has_ref:
        lines += ["    let period_ticks = data.usize()?;", "    check(period_ticks <= 1_000_000, \"invalid refractory period\")?;", "    let mut lastspike = data.f64_vec(N)?;",
                  "    let mut not_refractory = data.bool_vec(N)?;",
                  '    for &last in &lastspike { let elapsed = ((STEPS as f64*dt - last) + 1e-3*dt)/dt; check(last <= 0.0 && elapsed.is_finite() && elapsed < 9_223_372_036_854_775_808.0, "invalid refractory initial state")?; }',
                  "    let mut refractory_until: Vec<usize> = lastspike.iter().map(|&last| first_available_tick(last, dt, period_ticks)).collect();"]
    else:
        lines += ["    let period_ticks = 0usize;", "    let mut lastspike: Vec<f64> = Vec::new();",
                  "    let mut not_refractory: Vec<u8> = Vec::new();",
                  "    let mut refractory_until: Vec<usize> = Vec::new();"]
    if has_syn:
        lines += [f"    let edge_count = {edge_count}usize;", "    let source = data.u32_vec(edge_count)?;",
                  "    let target_index = data.u32_vec(edge_count)?;"]
        for pos, _ in enumerate(d["synapses"]["states"]):
            lines.append(f"    let mut syn_state_{pos} = data.f64_vec(edge_count)?;")
        for pos, symbol in enumerate(d["synapses"]["parameters"]):
            length = 1 if symbol["index_domain"] == "scalar" else edge_count
            lines.append(f"    let syn_parameter_{pos} = data.f64()?;" if symbol["index_domain"] == "scalar" else f"    let syn_parameter_{pos} = data.f64_vec({length})?;")
        lines += [f"    let delay_values = data.usize_vec({len(delay_ticks)})?;",
                  '    check(delay_values.iter().all(|&delay| delay <= 1_000_000), "invalid delay")?;',
                  '    check(source.iter().chain(&target_index).all(|&i| (i as usize) < N), "invalid topology index")?;']
        if uniform_delay is not None:
            lines += ["    let (offsets, edges) = source_csr(&source, N);",
                      f'    check(delay_values.iter().all(|&delay| delay == {uniform_delay}), "instance delay specialization mismatch")?;',
                      f"    let delay_ticks = {uniform_delay}usize;",
                      "    let mut edge_scratch: Vec<usize> = Vec::new();"]
            if pack_uniform_edges:
                # Event delivery walks source CSR order. Packing immutable edge
                # inputs once removes one indirection and turns vector parameter
                # reads into sequential accesses without changing per-source
                # Brian edge order. There are no synaptic states to map back.
                lines += [
                    "    let target_index: Vec<u32> = edges.iter().map(|&edge| target_index[edge]).collect();",
                    "    let source: Vec<u32> = edges.iter().map(|&edge| source[edge]).collect();",
                ]
                for pos, symbol in enumerate(d["synapses"]["parameters"]):
                    if symbol["index_domain"] != "scalar":
                        lines.append(
                            f"    let syn_parameter_{pos}: Vec<f64> = edges.iter()"
                            f".map(|&edge| syn_parameter_{pos}[edge]).collect();")
                lines.append("    let edges: Vec<usize> = (0..edge_count).collect();")
            if uniform_delay > 0:
                lines += ["    let mut queue: Vec<EventBatch> = (0..(delay_ticks + 1).min(STEPS)).map(|_| EventBatch::default()).collect();"]
        else:
            lines += ["    let delay_ticks = delay_values;",
                      f"    let delay_groups = [{', '.join(f'{value}usize' for value in delay_groups)}];",
                      "    let mut delay_offsets: Vec<Vec<usize>> = Vec::new();",
                      "    let mut delay_edges: Vec<Vec<usize>> = Vec::new();",
                      "    for &delay in &delay_groups { let (group_offsets, group_edges) = source_csr_delay(&source, &delay_ticks, delay, N); delay_offsets.push(group_offsets); delay_edges.push(group_edges); }",
                      "    let queue_size = (delay_groups[0] + 1).min(STEPS);",
                      "    let mut queues: Vec<Vec<EventBatch>> = (0..delay_groups.len()).map(|_| (0..queue_size).map(|_| EventBatch::default()).collect()).collect();"]
    else:
        lines += ["    let edge_count = 0usize;", "    let target_index: Vec<u32> = Vec::new();",
                  "    let delay_ticks = 0usize;", "    let offsets = vec![0usize; N + 1];",
                  "    let edges: Vec<usize> = Vec::new();"]
    for pos in range(len(states)):
        lines.append(f"    assert_eq!(state_{pos}.len(), N);")
    for pos, symbol in enumerate(parameters):
        if symbol["index_domain"] != "scalar": lines.append(f"    assert_eq!(parameter_{pos}.len(), N);")
    if has_ref: lines += ["    assert_eq!(lastspike.len(), N);", "    assert_eq!(not_refractory.len(), N);",
                         "    assert_eq!(refractory_until.len(), N);"]
    lines += ["    data.end()?;", "    drop(data);",
              f"    let parallel = Parallel::from_env({str(parallel_capable).lower()})?;"]
    if parallel_event_capable:
        lines += ["    let target_owner = if parallel.threads() > 1 {",
                  "        let mut target_degrees = vec![0usize; N];",
                  "        for &target in &target_index { target_degrees[target as usize] += 1; }",
                  "        let target_owners = degree_balanced_target_owners(&target_degrees, parallel.threads());",
                  "        Some(target_owner_csr_usize(&offsets, &edges, &target_index, N, &target_owners, parallel.threads()))",
                  "    } else { None };"]
    lines += ["    let mut parallel_on_pre = false;",
              "    let mut fired_lanes: Vec<Vec<usize>> = (0..parallel.threads()).map(|_| Vec::new()).collect();",
              "    let mut samples: Vec<f64> = Vec::new();", f"    samples.try_reserve_exact({d['monitor']['window_steps'] * len(d['monitor']['record']) * len(d['monitor']['variables'])})?;",
              "    let mut spikes: Vec<(u32, u32)> = Vec::new();", "    let mut counts = vec![0usize; N];",
              "    let mut fired: Vec<usize> = Vec::new();", "    let mut delivered = 0usize;",
              "    let initialization_seconds = initialization_started.elapsed().as_secs_f64();", "    let started = Instant::now();", "    for tick in 0..STEPS {", "        let time = tick as f64 * dt;"]
    lines.append(f"        if tick >= STEPS - {d['monitor']['window_steps']} {{")
    for i in d["monitor"]["record"]:
        for name in d["monitor"]["variables"]:
            lines.append(f"            samples.push(state_{monitor_states[name]}[{i}]);")
    lines.append("        }")
    lines += ["        fired.clear();",
              "        let threshold_fused = " + kernel_call + ";",
              "        if threshold_fused { for lane in &mut fired_lanes { fired.append(lane); } }"]
    if "synapse_state_update" in code_by_kind:
        synapse_update = code_by_kind["synapse_state_update"]
        scalar, vector, symbols = code_block(model, synapse_update, "synapse", "edge")
        lines += scalar + ["        for edge in 0..edge_count {",
                           "            let source = source[edge] as usize;",
                           "            let target = target_index[edge] as usize;"]
        lines += ["    " + line for line in vector]
        for name in synapse_update["effects"]["writes"]:
            lines.append(f"            syn_state_{synapse_state_positions[name]}[edge] = {symbols[name]};")
        lines += ["        }"]
    if "threshold" in code_by_kind:
        threshold = code_by_kind["threshold"]
        scalar, vector, symbols = code_block(model, threshold, "neuron")
        lines += ["        if !threshold_fused {"] + ["    " + line for line in scalar] + ["            for i in 0..N {"] + ["    " + line for line in vector]
        condition = symbols["_cond"]
        gate = f"{condition} && not_refractory[i] != 0" if has_ref else condition
        lines += [f"                if {gate} {{", "                    fired.push(i);"]
        if has_ref:
            lines += ["                    lastspike[i] = time;", "                    not_refractory[i] = 0;",
                      "                    refractory_until[i] = tick + period_ticks;"]
        lines += ["                }", "            }", "        }"]
    if "threshold" in code_by_kind:
        lines += [f"        if tick >= STEPS - {d['monitor']['window_steps']} {{",
                  "            for &i in &fired { counts[i] += 1; record_spike(&mut spikes, tick, i)?; }",
                  "        }"]
    if "threshold" in code_by_kind and has_syn:
        syn = code_by_kind["synapses"]
        scalar, vector, symbols = code_block(model, syn, "synapse", "edge")
        aliases = d["synapses"]["post_state_aliases"]
        if uniform_delay is None:
            lines += ["        for group in 0..delay_groups.len() {",
                      "            let delivery = tick + delay_groups[group];",
                      "            if delivery < STEPS {",
                      "                let group_offsets = &delay_offsets[group];",
                      "                let group_edges = &delay_edges[group];",
                      "                let pending = &mut queues[group][delivery % queue_size];",
                      "                pending.event_count = fired.iter().map(|&source| group_offsets[source + 1] - group_offsets[source]).sum();",
                      f"                pending.source_mode = pending.event_count >= {source_batch_min_edges};",
                      "                if pending.source_mode { pending.sources.extend_from_slice(&fired); }",
                      "                else { for &source in &fired { pending.edges.extend_from_slice(&group_edges[group_offsets[source]..group_offsets[source + 1]]); } }",
                      "            }",
                      "        }"] + scalar
            lines += ["        for group in 0..delay_groups.len() {",
                      "            let group_offsets = &delay_offsets[group];",
                      "            let group_edges = &delay_edges[group];",
                      "            let active = &queues[group][tick % queue_size];",
                      "            delivered += active.event_count;",
                      "            if active.source_mode {",
                      "                for &source in &active.sources {",
                      "                    for &edge in &group_edges[group_offsets[source]..group_offsets[source + 1]] {",
                      "                        let target = target_index[edge] as usize;"]
            lines += ["            " + line for line in vector]
            for alias in syn["effects"]["writes"]:
                if alias in aliases:
                    lines.append(f"                        state_{monitor_states[aliases[alias]]}[target] = {symbols[alias]};")
                else:
                    lines.append(f"                        syn_state_{synapse_state_positions[alias]}[edge] = {symbols[alias]};")
            lines += ["                    }", "                }", "            } else {",
                      "                for &edge in &active.edges {",
                      "                    let source = source[edge] as usize;",
                      "                    let target = target_index[edge] as usize;"]
            lines += ["        " + line for line in vector]
            for alias in syn["effects"]["writes"]:
                if alias in aliases:
                    lines.append(f"                    state_{monitor_states[aliases[alias]]}[target] = {symbols[alias]};")
                else:
                    lines.append(f"                    syn_state_{synapse_state_positions[alias]}[edge] = {symbols[alias]};")
            lines += ["                }", "            }", "        }",
                      "        for group in 0..delay_groups.len() { queues[group][tick % queue_size].clear(); }"]
        elif uniform_delay == 0:
            lines += ["        let active_event_count: usize = fired.iter().map(|&source| offsets[source + 1] - offsets[source]).sum();",
                      "        delivered += active_event_count;"]
        else:
            lines += ["        let delivery = tick + delay_ticks;", "        let queue_size = queue.len();",
                      "        if delivery < STEPS {",
                      "            let pending = &mut queue[delivery % queue_size];",
                      "            pending.event_count = fired.iter().map(|&source| offsets[source + 1] - offsets[source]).sum();",
                      f"            pending.source_mode = parallel.events_parallel(pending.event_count) || pending.event_count >= {source_batch_min_edges};",
                      "            if pending.source_mode { pending.sources.extend_from_slice(&fired); }",
                      "            else { for &source in &fired { pending.edges.extend_from_slice(&edges[offsets[source]..offsets[source + 1]]); } }",
                      "        }",
                      "        let mut active = std::mem::take(&mut queue[tick % queue_size]);",
                      "        delivered += active.event_count;"]
        if uniform_delay is not None:
            condition = (f"active_event_count >= {source_batch_min_edges}"
                         if uniform_delay == 0 else "active.source_mode")
            source_values = "fired.iter()" if uniform_delay == 0 else "active.sources.iter()"
            event_count = ("active_event_count" if uniform_delay == 0 else
                           "active.event_count")
            if parallel_event_capable:
                active_slice = "&fired" if uniform_delay == 0 else "&active.sources"
                lines += [f"        if parallel.events_parallel({event_count}) {{",
                                   "            parallel_on_pre = true;",
                                   "            " + event_kernel_call.format(
                                       active=active_slice),
                                   f"        }} else if {condition} {{"] + [
                                       "    " + line for line in scalar]
            else:
                lines += scalar + [f"        if {condition} {{"]
            lines += [f"            for &source in {source_values} {{",
                               "                for &edge in &edges[offsets[source]..offsets[source + 1]] {",
                               "                    let target = target_index[edge] as usize;"]
            lines += ["        " + line for line in vector]
            for alias in syn["effects"]["writes"]:
                if alias in aliases:
                    lines.append(f"                    state_{monitor_states[aliases[alias]]}[target] = {symbols[alias]};")
                else:
                    lines.append(f"                    syn_state_{synapse_state_positions[alias]}[edge] = {symbols[alias]};")
            lines += ["                }", "            }", "        } else {"]
            edge_values = "&edge_scratch" if uniform_delay == 0 else "&active.edges"
            if uniform_delay == 0:
                lines += ["            edge_scratch.clear();",
                          "            for &source in &fired { edge_scratch.extend_from_slice(&edges[offsets[source]..offsets[source + 1]]); }"]
            lines += [f"            for &edge in {edge_values} {{", "                let source = source[edge] as usize;",
                      "                let target = target_index[edge] as usize;"]
            lines += ["    " + line for line in vector]
            for alias in syn["effects"]["writes"]:
                if alias in aliases:
                    lines.append(f"                state_{monitor_states[aliases[alias]]}[target] = {symbols[alias]};")
                else:
                    lines.append(f"                syn_state_{synapse_state_positions[alias]}[edge] = {symbols[alias]};")
            lines += ["            }", "        }"]
            if uniform_delay > 0:
                lines += ["        active.clear();", "        queue[tick % queue_size] = active;"]
    if "reset" in code_by_kind:
        reset = code_by_kind["reset"]
        scalar, vector, symbols = code_block(model, reset, "neuron")
        lines += scalar + ["        for &i in &fired {"] + vector
        for name in reset["effects"]["writes"]:
            lines.append(f"            state_{monitor_states[name]}[i] = {symbols[name]};")
        lines += ["        }"]
    lines += ["    }"]
    for pos in range(len(states)):
        lines.append(f'    check(state_{pos}.iter().all(|value| value.is_finite()), "non-finite final state")?;')
    for pos in range(len(synapse_state_positions)):
        lines.append(f'    check(syn_state_{pos}.iter().all(|value| value.is_finite()), "non-finite final synaptic state")?;')
    parallel_work = code_work(code_by_kind["state_update"])
    lines += ["    let simulation_and_recording_seconds = started.elapsed().as_secs_f64();",
              "    let simulation_threads = parallel.threads();",
              "    let thread_affinity = parallel.affinity_enabled();",
              "    let thread_cpus = parallel.affinity_json();",
              f"    let parallel_state_update = parallel.is_parallel(N, {parallel_work});",
              "    drop(parallel);",
              "    let output_started = Instant::now();", "    fs::create_dir_all(output)?;"]
    lines += emit_dump(model)
    lines += ["    let dump_write_seconds = output_started.elapsed().as_secs_f64();"]
    lines += emit_metadata(model)
    lines += ["    println!(\"completed {} ticks for {} neurons (native AOT)\", STEPS, N);", "    Ok(())", "}"]
    return _join_source(lines)


def emit_dump(model):
    d, inst, run = model["definition"], model["instance"], model["run"]
    refractory_bytes = " + lastspike.len()*8 + not_refractory.len()" if d["refractory"] is not None else ""
    synapse_states = len((d["synapses"] or {}).get("states", []))
    edge_count = len((inst["synapses"] or {}).get("source", []))
    synapse_bytes = (f" + 24 + {synapse_states}*{edge_count}*8"
                      if d["synapses"] is not None else "")
    size = ("40usize + 64 + samples.len()*8 + spikes.len()*16 + counts.len()*8 + "
            "fired.len()*8 + " + str(len(d["states"])) + "*N*8" + refractory_bytes +
            " + 24" + synapse_bytes)
    lines = [f"    let dump_bytes = {size};",
             '    let mut dump = dump_start(output, dump_bytes, 1, N)?;',
             "    dump_u64(&mut dump, N)?;", "    dump_u64(&mut dump, STEPS)?;",
             f"    dump_u64(&mut dump, {len(d['monitor']['record'])})?;",
             f"    dump_u64(&mut dump, {len(d['monitor']['variables'])})?;",
             f"    dump_u64(&mut dump, {len(d['states'])})?;",
             "    dump_u64(&mut dump, spikes.len())?;",
             "    dump_u64(&mut dump, fired.len())?;",
             f"    dump_u64(&mut dump, {int(d['refractory'] is not None)})?;"]
    for column in range(len(d["monitor"]["variables"])):
        lines += [
            f"    let sample_column_{column}: Vec<f64> = samples.iter()"
            f".skip({column}).step_by({len(d['monitor']['variables'])}).copied().collect();",
            f"    dump_f64(&mut dump, &sample_column_{column})?;",
        ]
    lines += ["    dump_spikes_u32(&mut dump, &spikes)?;",
             "    dump_indices(&mut dump, &counts)?;", "    dump_indices(&mut dump, &fired)?;"]
    for position in range(len(d["states"])):
        lines.append(f"    dump_f64(&mut dump, &state_{position})?;")
    if d["refractory"] is not None:
        lines += ["    dump_f64(&mut dump, &lastspike)?;", "    dump.write_all(&not_refractory)?;"]
    lines += [f"    dump_u64(&mut dump, {int(d['synapses'] is not None)})?;"]
    if d["synapses"] is not None:
        lines += [f"    dump_u64(&mut dump, {synapse_states})?;",
                  f"    dump_u64(&mut dump, {edge_count})?;"]
        for position in range(synapse_states):
            lines.append(f"    dump_f64(&mut dump, &syn_state_{position})?;")
        lines += ["    dump_u64(&mut dump, delivered)?;"]
    lines += ["    dump.write_all(&(STEPS as f64*dt).to_le_bytes())?;",
              "    dump_finish(dump, output, dump_bytes)?;"]
    return lines


def emit_metadata(model):
    d, inst, run = model["definition"], model["instance"], model["run"]
    lines = ['    let mut summary = BufWriter::new(File::create(output.join("summary.json"))?);']
    def text(value):
        lines.append(f"    summary.write_all({json.dumps(value)}.as_bytes())?;")
    def value(expression):
        lines.append(f'    write!(summary, "{{}}", {expression})?;')
    text('{"schema":"b2-result-dump-v3","dump_bytes":'); value("dump_bytes")
    text(',"threads":'); value('simulation_threads')
    text(',"thread_affinity":'); value('thread_affinity')
    text(',"thread_cpus":'); value('&thread_cpus')
    text(',"parallel_state_update":'); value('parallel_state_update')
    text(',"parallel_on_pre":'); value('parallel_on_pre')
    text(',"population_count":1,"neuron_count":' + str(inst["neuron_count"]) + ',"spike_count":')
    value('spikes.len()')
    text(',"final_time_seconds":'); value('STEPS as f64 * dt')
    text(',"synaptic_events":'); value('delivered')
    text(',"timings":{"initialization_seconds":'); value('initialization_seconds')
    text(',"simulation_and_recording_seconds":'); value('simulation_and_recording_seconds')
    text(',"dump_write_seconds":'); value('dump_write_seconds')
    text('}}\n'); lines.append('    summary.flush()?;')
    return lines


def write_instance_v6(model, path):
    d, inst, run = model["definition"], model["instance"], model["run"]
    out = bytearray(b"B2AOT001")
    out += struct.pack("<dQQ", struct.unpack(">d", bytes.fromhex(run["dt"]))[0], run["steps"], inst["neuron_count"])
    values = lambda items: b"".join(struct.pack("<d", struct.unpack(">d", bytes.fromhex(v))[0]) for v in items)
    for symbol in d["states"]: out += values(inst["initial_state"][symbol["name"]])
    for symbol in d["parameters"]: out += values(inst["parameters"][symbol["name"]])
    if inst["refractory"] is not None:
        ref = inst["refractory"]; out += struct.pack("<Q", ref["period_ticks"])
        out += values(ref["initial_lastspike"]) + bytes(ref["initial_not_refractory"])
    if inst["synapses"] is not None:
        syn = inst["synapses"]
        out += struct.pack(f"<{len(syn['source'])}I", *syn["source"])
        out += struct.pack(f"<{len(syn['target'])}I", *syn["target"])
        for symbol in d["synapses"]["states"]: out += values(syn["initial_state"][symbol["name"]])
        for symbol in d["synapses"]["parameters"]: out += values(syn["parameters"][symbol["name"]])
        out += struct.pack(f"<{len(syn['delay_ticks'])}Q", *syn["delay_ticks"])
    path.write_bytes(out)
    return hashlib.sha256(out).hexdigest()


def _v7_inputs(model, population, domain, index="i", synapse=None,
               dt_expression=None):
    d = model["definition"]
    pop = d["populations"][population]
    prefix = f"p{population}_"
    values = {"dt": (prefix + "dt" if dt_expression is None else
                     dt_expression), "t": "time"}
    if domain == "neuron":
        values.update(i=f"{index} as f64", N=f"{pop['count']}.0")
        for position, symbol in enumerate(pop["states"]):
            values[symbol["name"]] = _storage_load(
                symbol,
                f"unsafe {{ *{prefix}state_{position}.get_unchecked({index}) }}")
        for position, symbol in enumerate(pop["parameters"]):
            if symbol["index_domain"] == "scalar":
                values[symbol["name"]] = _storage_load(
                    symbol, f"{prefix}parameter_{position}")
            else:
                values[symbol["name"]] = _storage_load(
                    symbol,
                    f"unsafe {{ *{prefix}parameter_{position}.get_unchecked({index}) }}")
        for position, linked in enumerate(pop.get("linked_variables", [])):
            values[linked["name"]] = _v7_link_value(
                model, population, position, index)
        if pop["refractory"] is not None:
            values.update(
                lastspike=f"unsafe {{ *{prefix}lastspike.get_unchecked({index}) }}",
                not_refractory=(
                    f"unsafe {{ *{prefix}not_refractory.get_unchecked({index}) }} != 0"))
    else:
        syn = d["synapses"][synapse]
        syn_prefix = f"s{synapse}_"
        source, target = syn["source_population"], syn["target_population"]
        source_states = {s["name"]: i for i, s in enumerate(d["populations"][source]["states"])}
        target_states = {s["name"]: i for i, s in enumerate(d["populations"][target]["states"])}
        values.update(i="source as f64", j="target as f64",
                      N=f"{syn_prefix}edge_count as f64",
                      N_pre=f"{syn['source_count']}.0", N_post=f"{syn['target_count']}.0")
        for alias, state in syn["pre_state_aliases"].items():
            symbol = d["populations"][source]["states"][source_states[state]]
            values[alias] = _storage_load(
                symbol, f"p{source}_state_{source_states[state]}[source_state]")
        for alias, state in syn["post_state_aliases"].items():
            symbol = d["populations"][target]["states"][target_states[state]]
            values[alias] = _storage_load(
                symbol, f"p{target}_state_{target_states[state]}[target_state]")
        for position, symbol in enumerate(syn["states"]):
            values[symbol["name"]] = _storage_load(
                symbol,
                f"unsafe {{ *{syn_prefix}state_{position}.get_unchecked(edge) }}")
        for position, symbol in enumerate(syn["parameters"]):
            if symbol["index_domain"] == "scalar":
                values[symbol["name"]] = _storage_load(
                    symbol, f"{syn_prefix}parameter_{position}")
            else:
                values[symbol["name"]] = _storage_load(
                    symbol,
                    f"unsafe {{ *{syn_prefix}parameter_{position}.get_unchecked(edge) }}")
        for position, linked in enumerate(syn.get("linked_variables", [])):
            values[linked["name"]] = _v7_synapse_link_value(
                model, synapse, position)
        if d["populations"][target]["refractory"] is not None:
            values["not_refractory_post"] = f"p{target}_not_refractory[target_state] != 0"
    return values


def _v7_dtypes(model, population, domain, synapse=None):
    definition = model["definition"]
    population_def = definition["populations"][population]
    dtypes = {"dt": "f64", "t": "f64", "i": "index", "N": "index"}
    if domain == "neuron":
        for symbol in (population_def["states"] + population_def["parameters"] +
                       population_def.get("linked_variables", [])):
            dtypes[symbol["name"]] = symbol["dtype"]
        if population_def["refractory"] is not None:
            dtypes.update(lastspike="f64", not_refractory="bool")
        return dtypes
    synapse_def = definition["synapses"][synapse]
    dtypes.update(j="index", N_pre="index", N_post="index")
    source = definition["populations"][synapse_def["source_population"]]
    target = definition["populations"][synapse_def["target_population"]]
    source_states = {symbol["name"]: symbol for symbol in source["states"]}
    target_states = {symbol["name"]: symbol for symbol in target["states"]}
    for alias, state in synapse_def["pre_state_aliases"].items():
        dtypes[alias] = source_states[state]["dtype"]
    for alias, state in synapse_def["post_state_aliases"].items():
        dtypes[alias] = target_states[state]["dtype"]
    for symbol in (synapse_def["states"] + synapse_def["parameters"] +
                   synapse_def.get("linked_variables", [])):
        dtypes[symbol["name"]] = symbol["dtype"]
    if target["refractory"] is not None:
        dtypes["not_refractory_post"] = "bool"
    return dtypes


def _v7_link_value(model, population, position, index):
    populations = model["definition"]["populations"]
    linked = populations[population]["linked_variables"][position]
    source_population = linked["source_population"]
    source = populations[source_population]
    source_position = next(
        position for position, symbol in enumerate(source["states"])
        if symbol["name"] == linked["source_state"])
    source_symbol = source["states"][source_position]
    mapping = linked["index"]
    if mapping["kind"] == "identity":
        source_index = index
        storage = (
            f"unsafe {{ *p{source_population}_state_{source_position}."
            f"get_unchecked({source_index}) }}")
    elif mapping["kind"] == "constant":
        source_index = f"P{population}_LINK_{position}_INDICES[{index}]"
        storage = (
            f"unsafe {{ *p{source_population}_state_{source_position}."
            f"get_unchecked({source_index}) }}")
    else:
        name = mapping["name"]
        local = populations[population]
        if mapping["kind"] == "state":
            local_position = next(
                position for position, symbol in enumerate(local["states"])
                if symbol["name"] == name)
            raw = _storage_load(
                local["states"][local_position],
                f"p{population}_state_{local_position}[{index}]")
        else:
            local_position = next(
                position for position, symbol in enumerate(local["parameters"])
                if symbol["name"] == name)
            raw = _storage_load(
                local["parameters"][local_position],
                f"p{population}_parameter_{local_position}[{index}]")
        source_index = f"({raw}) as usize"
        storage = (
            f"*p{source_population}_state_{source_position}.get({source_index})"
            f".ok_or(\"linked variable index out of bounds\")?")
    return _storage_load(source_symbol, storage)


def _v7_synapse_link_value(model, synapse, position):
    """Read the fixed population-state input accepted for a Synapses link."""
    definition = model["definition"]
    linked = definition["synapses"][synapse]["linked_variables"][position]
    mapping = linked["index"]
    if mapping["kind"] != "constant" or not mapping["values"] or any(
            value != mapping["values"][0] for value in mapping["values"]):
        raise ValueError("AOT Synapses linked variables require one fixed source index")
    source_population = linked["source_population"]
    source = definition["populations"][source_population]
    source_position, source_symbol = next(
        (index, symbol) for index, symbol in enumerate(source["states"])
        if symbol["name"] == linked["source_state"])
    storage = (
        f"unsafe {{ *p{source_population}_state_{source_position}."
        f"get_unchecked({mapping['values'][0]}usize) }}")
    return _storage_load(source_symbol, storage)


def _v7_event_vector(population, population_definition, event):
    """Return the generated Vec<usize> name for one named EventStream."""
    if event == "spike":
        return f"p{population}_fired"
    position = population_definition["events"].index(event)
    return f"p{population}_event_{position}"


def _single_population_v6(model):
    """Feed the established optimized kernel generator for the one-population case."""
    pop = model["definition"]["populations"][0]
    pop_instance = model["instance"]["populations"][0]
    synapses = model["definition"]["synapses"]
    synapse_instances = model["instance"]["synapses"]
    synapse = synapses[0] if synapses else None
    synapse_instance = synapse_instances[0] if synapse_instances else None
    if synapse_instance is not None:
        primary = next(pathway for pathway in synapse_instance["pathways"]
                       if pathway["kind"] == "pre")
        synapse_instance = {
            **{key: value for key, value in synapse_instance.items()
               if key != "pathways"},
            "delay": primary["delay"],
            "delay_ticks": primary["delay_ticks"],
            "pending": primary["pending"],
        }
    code_objects = [next(code for code in pop["code_objects"]
                         if code["kind"] == "state_update")]
    if synapse is not None:
        state_update = next((code for code in synapse["code_objects"]
                             if code["kind"] == "synapse_state_update"), None)
        if state_update is not None:
            code_objects.append(state_update)
    threshold = next((code for code in pop["code_objects"]
                      if code["kind"] == "threshold"), None)
    if threshold is not None:
        code_objects.append(threshold)
        if synapse is not None:
            code_objects.append(next(code for code in synapse["code_objects"]
                                     if code["kind"] == "synapses"))
        reset = next((code for code in pop["code_objects"]
                      if code["kind"] == "reset"), None)
        if reset is not None:
            code_objects.append(reset)
    legacy_synapse = None if synapse is None else {
        **{key: value for key, value in synapse.items()
           if key not in {"source_population", "target_population", "code_objects"}},
        "source_offset": 0, "target_offset": 0,
    }
    return {
        "schema": "b2ir-gate0-probe-v6",
        "definition": {
            "states": pop["states"], "parameters": pop["parameters"],
            "functions": model["definition"].get("functions", []),
            "synapses": legacy_synapse, "refractory": pop["refractory"],
            "populations": [{"name": pop["name"], "offset": 0,
                             "count": pop["count"],
                             "state_monitor": (pop["state_monitors"][0]["name"]
                                               if pop["state_monitors"] else None),
                             "monitor_start": 0,
                             "monitor_count": len(pop["monitor"]["record"]),
                             "spike_monitor": pop["spike_monitor"]}],
            "code_objects": code_objects, "monitor": pop["monitor"],
            "schedule": [], "numeric_profile": "reference-f64",
        },
        "instance": {
            "neuron_count": pop["count"], "initial_state": pop_instance["initial_state"],
            "parameters": pop_instance["parameters"], "synapses": synapse_instance,
            "refractory": pop_instance["refractory"],
        },
        "run": {"dt": pop["dt"], "steps": pop["steps"]},
    }




def _poisson_zero_gate(pop, code, symbols, population):
    """Prove a zero scalar gate can replace one exact, nonnegative input.

    Preserve the original multiplication order in the fallback. In particular,
    a Gaussian approximation may be negative, and adding positive zero to a
    negative-zero state is observable; neither operation can just be dropped.
    """
    if pop["count"] < 512 or len(code["vector"]) != 1:
        return None
    statement = code["vector"][0]
    target = statement["target"]
    positions = {s["name"]: i for i, s in enumerate(pop["states"])}
    if (target not in positions or statement.get("condition") is not None
            or statement["dtype"] != "f64"
            or pop["states"][positions[target]]["dtype"] != "f64"
            or code["effects"]["writes"] != [target]):
        return None
    value = statement["value"]
    if (value["op"] != "add"
            or value["left"] != {"op": "load", "name": target}):
        return None
    product = value["right"]
    if product["op"] != "mul" or product["left"]["op"] != "load":
        return None
    scalar_names = {s["target"] for s in code["scalar"]
                    if s["dtype"] == "f64" and s.get("condition") is None}
    # Only temporaries already evaluated outside the vector loop qualify.
    gate = product["left"]["name"]
    if gate not in scalar_names or scalar_names & set(positions):
        return None
    weighted = product["right"]
    if (weighted["op"] != "mul" or weighted["left"]["op"] != "binomial"
            or weighted["right"]["op"] != "timed_array"):
        return None
    draw, table = weighted["left"], weighted["right"]
    if (table["columns"] != pop["count"]
            or table["index"] != {"op": "load", "name": "i"}
            or table["time"]["op"] != "load"
            or table["time"]["name"] not in scalar_names | {"t"}
            or not set(table["values"]) <= {
                "0000000000000000", "3ff0000000000000"}):
        return None
    # A 0/1 table with a clamped row and in-range i cannot produce NaN,
    # negative zero, or overflow, even for the largest supported draw count.
    probability = draw["p"]
    parameters = {s["name"]: i for i, s in enumerate(pop["parameters"])
                  if s["index_domain"] == "scalar" and s["dtype"] == "f64"}
    if probability["op"] != "load" or probability["name"] not in parameters:
        return None
    if set(parameters) & {s["target"] for s in code["scalar"]}:
        return None
    p = f"p{population}_parameter_{parameters[probability['name']]}"
    n = f"{draw['n']}.0"
    exact = (f"!({n}*{p}>5.0 && {n}*(1.0-{p})>5.0)"
             if draw["approximate"] else "true")
    return symbols[gate], p, exact, positions[target]


def _v7_block(model, code, population, domain, index="i", synapse=None,
              overrides=None, normal_cache=None, clock=None, summed_caches=()):
    clock_prefix = None if clock is None else f"c{clock}_"
    inputs = _v7_inputs(
        model, population, domain, index, synapse,
        dt_expression=None if clock_prefix is None else clock_prefix + "dt")
    dtypes = _v7_dtypes(model, population, domain, synapse)
    inputs.update(overrides or {})
    rng_index = f"{index} as u64" if domain == "neuron" else "edge as u64"
    functions = {function["name"]: function
                 for function in model["definition"].get("functions", [])}
    block = Block(inputs, dtypes, {"seed": "rng_seed",
                           "tick": (f"p{population}_tick as u64"
                                    if clock_prefix is None else
                                    clock_prefix + "tick as u64"),
                           "index": rng_index,
                           "normal_cache": normal_cache},
                  functions)
    # Scalar statements run once per code-object activation.  Brian places a
    # mutable ``(shared)`` assignment such as ``k = int(rand()*N)`` here, so
    # give counter RNG a canonical scalar index instead of the loop variable
    # that is only introduced by the following vector block.
    vector_rng_index = block.rng["index"]
    block.rng["index"] = "0u64"
    block.statements(code["scalar"], "        ")
    block.rng["index"] = vector_rng_index
    scalar = block.lines
    block.lines = []
    # These expressions are stable only within this exact node activation.
    # Rebuild each cache here, after scalar statements and before its edge
    # loop. Separate temporary names prevent shadowing scalar CSE values.
    for position, cache in enumerate(summed_caches):
        from .summed_cache import expression_key
        endpoint = "source" if cache["endpoint"] == "pre" else "target"
        name = f"sum_cache_{synapse}_{position}"
        cache_inputs = dict(block.symbols)
        for alias in cache["aliases"]:
            cache_inputs[alias] = inputs[alias].replace(
                f"[{endpoint}_state]", "[sum_cache_state]")
        cached_block = Block(cache_inputs, dtypes, functions=functions,
                             temp_prefix=name + "_r")
        result = cached_block.expr(cache["expression"], "            ")
        begin, count = cache["begin"], cache["count"]
        scalar += [
            f"        let {name}: Vec<f64> = ({begin}..{begin + count}).map(|sum_cache_state| {{",
            *cached_block.lines, f"            {result}", "        }).collect();"]
        block.expression_overrides[expression_key(cache["expression"])] = (
            f"{name}[{endpoint}]")
    used = vector_inputs(code, inputs) - {
        statement["target"] for statement in code["scalar"]}
    for position, (name, expression) in enumerate(inputs.items()):
        if name in used:
            local = f"input_{population}_{synapse if synapse is not None else 'p'}_{position}"
            block.lines.append(f"        let {local} = {expression};")
            block.symbols[name] = local
    if code["kind"] in {"state_update", "synapse_state_update"}:
        block.input_snapshot = {name: block.symbols[name] for name in used}
    block.statements(code["vector"])
    return scalar, block.lines, block.symbols


def _v7_parallel_neuron(model, population, code, positions,
                        fuse_threshold=False, normal_cache=None):
    """Generate pointer setup and a worker-safe neuron block."""
    pop = model["definition"]["populations"][population]
    prefix = f"p{population}_"
    declarations, overrides = [], {}
    for position, symbol in enumerate(pop["states"]):
        dtype = _rust_dtype(symbol)
        pointer = f"{prefix}state_{position}_ptr"
        declarations.append(
            f"let {pointer} = {prefix}state_{position}.as_mut_ptr() as usize;")
        overrides[symbol["name"]] = _storage_load(
            symbol, f"unsafe {{ *(({pointer} as *const {dtype}).add(i)) }}")
    for position, symbol in enumerate(pop["parameters"]):
        if symbol["index_domain"] != "scalar":
            dtype = _rust_dtype(symbol)
            pointer = f"{prefix}parameter_{position}_ptr"
            declarations.append(
                f"let {pointer} = {prefix}parameter_{position}.as_ptr() as usize;")
            overrides[symbol["name"]] = _storage_load(
                symbol, f"unsafe {{ *(({pointer} as *const {dtype}).add(i)) }}")
    if pop["refractory"] is not None:
        declarations += [
            f"let {prefix}lastspike_ptr = {prefix}lastspike.as_mut_ptr() as usize;",
            f"let {prefix}not_refractory_ptr = "
            f"{prefix}not_refractory.as_mut_ptr() as usize;",
            f"let {prefix}refractory_until_ptr = " +
            (f"{prefix}refractory_until.as_mut_ptr() as usize;" if fuse_threshold
             else f"{prefix}refractory_until.as_ptr() as usize;"),
        ]
        overrides.update(
            lastspike=(
                f"unsafe {{ *(({prefix}lastspike_ptr as *const f64).add(i)) }}"),
            not_refractory=(
                f"unsafe {{ *(({prefix}not_refractory_ptr as *const u8).add(i)) }} != 0"),
        )
    scalar, vector, symbols = _v7_block(
        model, code, population, "neuron", overrides=overrides,
        normal_cache=normal_cache)
    writes = []
    for name in code["effects"]["writes"]:
        if name == "not_refractory":
            writes.append(
                f"unsafe {{ *(({prefix}not_refractory_ptr as *mut u8).add(i)) = "
                f"u8::from({symbols[name]}); }}")
        else:
            dtype = _rust_dtype(pop["states"][positions[name]])
            value = _storage_store(pop["states"][positions[name]], symbols[name])
            writes.append(
                f"unsafe {{ *(({prefix}state_{positions[name]}_ptr as *mut {dtype}).add(i)) = "
                f"{value}; }}")
    if not fuse_threshold:
        return declarations, scalar, vector, writes, None
    threshold = next(item for item in pop["code_objects"]
                     if item["kind"] == "threshold")
    threshold_scalar, threshold_vector, threshold_symbols = _v7_block(
        model, threshold, population, "neuron", overrides=overrides)
    gate = threshold_symbols["_cond"]
    if pop["refractory"] is not None:
        gate += (f" && unsafe {{ *(({prefix}not_refractory_ptr as *const u8).add(i)) }} != 0")
    threshold_writes = [f"if {gate} {{", "    lane_fired.push(i);"]
    if pop["refractory"] is not None:
        threshold_writes += [
            f"    unsafe {{ *(({prefix}lastspike_ptr as *mut f64).add(i)) = time; }}",
            f"    unsafe {{ *(({prefix}not_refractory_ptr as *mut u8).add(i)) = 0; }}",
        ]
        if pop["refractory"]["mode"] == "fixed":
            threshold_writes.append(
                f"    unsafe {{ *(({prefix}refractory_until_ptr as *mut usize).add(i)) = p{population}_tick + p{population}_period_ticks; }}")
    threshold_writes.append("}")
    return (declarations, scalar, vector, writes,
            (threshold_scalar, threshold_vector, threshold_writes))






def _v7_target_route_kernel(model, route, members, heterogeneous=False):
    """Generate one target-owner dispatch for a fused pre-event route."""
    d = model["definition"]
    populations, synapses = d["populations"], d["synapses"]
    source_populations = sorted(
        {synapses[q]["source_population"] for q in members})
    population_ids = sorted(
        set(source_populations) |
        {synapses[q]["target_population"] for q in members})
    arguments = ["parallel: &Parallel", "time: f64", "rng_seed: u64"]
    calls = ["&parallel", "time", "rng_seed"]
    for source_population in source_populations:
        arguments += [f"p{source_population}_tick: usize",
                      f"p{source_population}_dt: f64"]
        calls += [f"p{source_population}_tick", f"p{source_population}_dt"]
    if not heterogeneous:
        arguments.insert(1, "active: &[usize]")
        calls.insert(1, "{active}")
    declarations = []
    state_pointers = {}
    for population in population_ids:
        for position, symbol in enumerate(populations[population]["states"]):
            name = f"p{population}_state_{position}"
            pointer = name + "_event_ptr"
            arguments.append(f"{name}: &mut [{_rust_dtype(symbol)}]")
            calls.append(f"&mut {name}")
            declarations.append(f"let {pointer} = {name}.as_mut_ptr() as usize;")
            state_pointers[population, position] = pointer
        if populations[population]["refractory"] is not None:
            name = f"p{population}_not_refractory"
            pointer = name + "_event_ptr"
            arguments.append(f"{name}: &[u8]")
            calls.append(f"&{name}")
            declarations.append(f"let {pointer} = {name}.as_ptr() as usize;")

    synapse_pointers = {}
    for q in members:
        synapse = synapses[q]
        prefix = f"s{q}_"
        if heterogeneous:
            arguments += [f"{prefix}active: &[Vec<usize>]",
                          f"{prefix}source_index: &[u32]",
                          f"{prefix}target_index: &[u32]",
                          f"{prefix}edge_count: usize"]
            calls += [f"{{active_{q}}}", f"&{prefix}source_index",
                      f"&{prefix}target_index", f"{prefix}edge_count"]
        else:
            arguments += [f"{prefix}owner_csr: &TargetOwnerCsr",
                          f"{prefix}target_index: &[u32]",
                          f"{prefix}edge_count: usize"]
            calls += [f"{prefix}target_owner.as_ref().unwrap()",
                      f"&{prefix}target_index", f"{prefix}edge_count"]
        for position, symbol in enumerate(synapse["states"]):
            name = f"{prefix}state_{position}"
            pointer = name + "_event_ptr"
            arguments.append(f"{name}: &mut [{_rust_dtype(symbol)}]")
            calls.append(f"&mut {name}")
            declarations.append(f"let {pointer} = {name}.as_mut_ptr() as usize;")
            synapse_pointers[q, position] = pointer
        for position, symbol in enumerate(synapse["parameters"]):
            name = f"{prefix}parameter_{position}"
            dtype = _rust_dtype(symbol)
            arguments.append(
                f"{name}: " + (dtype if symbol["index_domain"] == "scalar"
                                else f"&[{dtype}]"))
            calls.append(name if symbol["index_domain"] == "scalar" else f"&{name}")

    suffix = "heterogeneous" if heterogeneous else "uniform"
    lines = ["#[inline(never)]",
             f"fn route_{route}_{suffix}_target_on_pre({', '.join(arguments)}) {{"]
    lines += ["    " + line for line in declarations]
    lines.append("    parallel.for_lanes(move |owner| {")
    for q in members:
        synapse = synapses[q]
        prefix = f"s{q}_"
        source = synapse["source_population"]
        target = synapse["target_population"]
        source_positions = {symbol["name"]: position for position, symbol in
                            enumerate(populations[source]["states"])}
        target_positions = {symbol["name"]: position for position, symbol in
                            enumerate(populations[target]["states"])}
        synapse_positions = {symbol["name"]: position for position, symbol in
                             enumerate(synapse["states"])}
        overrides = {}
        for alias, state in synapse["pre_state_aliases"].items():
            position = source_positions[state]
            pointer = state_pointers[source, position]
            symbol = populations[source]["states"][position]
            dtype = _rust_dtype(symbol)
            overrides[alias] = _storage_load(
                symbol,
                f"unsafe {{ *(({pointer} as *const {dtype}).add(source_state)) }}")
        for alias, state in synapse["post_state_aliases"].items():
            position = target_positions[state]
            pointer = state_pointers[target, position]
            symbol = populations[target]["states"][position]
            dtype = _rust_dtype(symbol)
            overrides[alias] = _storage_load(
                symbol,
                f"unsafe {{ *(({pointer} as *const {dtype}).add(target_state)) }}")
        for name, position in synapse_positions.items():
            pointer = synapse_pointers[q, position]
            symbol = synapse["states"][position]
            dtype = _rust_dtype(symbol)
            overrides[name] = _storage_load(
                symbol, f"unsafe {{ *(({pointer} as *const {dtype}).add(edge)) }}")
        if populations[target]["refractory"] is not None:
            overrides["not_refractory_post"] = (
                f"unsafe {{ *((p{target}_not_refractory_event_ptr as *const u8).add(target_state)) }} != 0")
        pathway = next(code for code in synapse["code_objects"]
                       if code["kind"] == "synapses")
        scalar, vector, symbols = _v7_block(
            model, pathway, source, "synapse", "edge", q,
            overrides=overrides)
        lines += ["        {"] + ["        " + line for line in scalar]
        if heterogeneous:
            lines += [f"            for &queued in &{prefix}active[owner] {{"]
            if _explicit_topology(model["instance"]["synapses"][q]):
                lines += ["                let edge = queued;",
                          f"                let source = unsafe {{ *{prefix}source_index.get_unchecked(edge) }} as usize;"]
            else:
                lines += ["                let source = queued >> 32;",
                          "                let edge = queued & (u32::MAX as usize);"]
            lines += [f"                let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                      f"                let source_state = source + {synapse['source_start']};",
                      f"                let target_state = target + {synapse['target_start']};"]
            vector_indent = "            "
            close = ["            }"]
        else:
            lines += [f"            let owner_offsets = &{prefix}owner_csr.offsets[owner];",
                      f"            let owner_edges = &{prefix}owner_csr.edges[owner];",
                      "            for &source in active {",
                      "                for &edge_index in &owner_edges[owner_offsets[source]..owner_offsets[source+1]] {",
                      "                    let edge = edge_index as usize;",
                      f"                    let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                      f"                    let source_state = source + {synapse['source_start']};",
                      f"                    let target_state = target + {synapse['target_start']};"]
            vector_indent = "            "
            close = ["                }", "            }"]
        lines += [vector_indent + line for line in vector]
        for name in pathway["effects"]["writes"]:
            if name in synapse["post_state_aliases"]:
                state = synapse["post_state_aliases"][name]
                position = target_positions[state]
                pointer = state_pointers[target, position]
                symbol = populations[target]["states"][position]
                dtype = _rust_dtype(symbol)
                value = _storage_store(symbol, symbols[name])
                lines.append(
                    f"{vector_indent}unsafe {{ *(({pointer} as *mut {dtype}).add(target_state)) = {value}; }}")
            else:
                position = synapse_positions[name]
                pointer = synapse_pointers[q, position]
                symbol = synapse["states"][position]
                dtype = _rust_dtype(symbol)
                value = _storage_store(symbol, symbols[name])
                lines.append(
                    f"{vector_indent}unsafe {{ *(({pointer} as *mut {dtype}).add(edge)) = {value}; }}")
        lines += [*close, "        }"]
    lines += ["    });", "}"]
    return lines, f"route_{route}_{suffix}_target_on_pre({', '.join(calls)});"


def _v7_plastic_kernel(model, synapse_index, kind):
    """Generate a parallel noalias kernel for a plasticity-only pathway."""
    synapse = model["definition"]["synapses"][synapse_index]
    code = next(item for item in synapse["code_objects"] if item["kind"] == kind)
    population = (synapse["source_population"] if kind == "synapses" else
                  synapse["target_population"])
    prefix = f"s{synapse_index}_"
    active_type = "&[u32]" if kind == "synapses_post" else "&[usize]"
    arguments = ["parallel: &Parallel", f"active: {active_type}"]
    calls = ["&parallel", "{active}"]
    if kind != "synapses_post":
        arguments += ["offsets: &[usize]", "edges: &[u32]"]
        calls += ["{offsets}", "{edges}"]
    arguments += ["time: f64", f"p{population}_dt: f64",
                 f"{prefix}edge_count: usize"]
    calls += ["time", f"p{population}_dt", f"{prefix}edge_count"]
    for position, symbol in enumerate(synapse["states"]):
        arguments.append(
            f"{prefix}state_{position}: &mut [{_rust_dtype(symbol)}]")
        calls.append(f"&mut {prefix}state_{position}")
    for position, symbol in enumerate(synapse["parameters"]):
        if symbol["index_domain"] == "scalar":
            arguments.append(
                f"{prefix}parameter_{position}: {_rust_dtype(symbol)}")
        else:
            arguments.append(
                f"{prefix}parameter_{position}: &[{_rust_dtype(symbol)}]")
        calls.append((f"{prefix}parameter_{position}"
                      if symbol["index_domain"] == "scalar" else
                      f"&{prefix}parameter_{position}"))
    state_positions = {symbol["name"]: position for position, symbol in
                       enumerate(synapse["states"])}
    declarations, overrides = [], {}
    for position, symbol in enumerate(synapse["states"]):
        dtype = _rust_dtype(symbol)
        pointer = f"{prefix}state_{position}_ptr"
        declarations.append(
            f"let {pointer} = {prefix}state_{position}.as_mut_ptr() as usize;")
        overrides[symbol["name"]] = _storage_load(
            symbol, f"unsafe {{ *(({pointer} as *const {dtype}).add(edge)) }}")
    for position, symbol in enumerate(synapse["parameters"]):
        if symbol["index_domain"] != "scalar":
            dtype = _rust_dtype(symbol)
            pointer = f"{prefix}parameter_{position}_ptr"
            declarations.append(
                f"let {pointer} = {prefix}parameter_{position}.as_ptr() as usize;")
            overrides[symbol["name"]] = _storage_load(
                symbol, f"unsafe {{ *(({pointer} as *const {dtype}).add(edge)) }}")
    scalar, vector, symbols = _v7_block(
        model, code, population, "synapse", "edge", synapse_index,
        overrides=overrides)
    name = f"{prefix}{'pre' if kind == 'synapses' else 'post'}_plastic"
    work = code_work(code)
    if kind == "synapses_post":
        parallel_work = post_synapse_work(synapse, code)
    else:
        average_degree = max(
            1, (_synapse_edge_count(model["instance"]["synapses"][synapse_index]) +
                synapse["source_count"] - 1) // synapse["source_count"])
        parallel_work = average_degree * work
    # Large graphs benefit from exposing the pre-pathway's setup to the caller.
    # Smaller graphs retain the original outlining: paired measurements found
    # no consistent benefit there. Post traversals also remain outlined.
    inline_pre = (kind == "synapses" and
                  _synapse_edge_count(model["instance"]["synapses"][synapse_index])
                  >= PRE_SYNAPSE_INLINE_MIN_EDGES)
    lines = ["#[inline(always)]" if inline_pre else "#[inline(never)]",
             f"fn {name}({', '.join(arguments)}) -> bool {{"]
    lines += ["    " + line for line in declarations]
    lines += ["    " + line for line in scalar]
    lines += [f"    let parallel_work = {parallel_work}usize;",
              "    let used_parallel = parallel.is_parallel(active.len(), parallel_work);",
              "    parallel.for_each(active.len(), parallel_work, move |start, end| {",
              "        for active_index in start..end {"]
    if kind == "synapses_post":
        lines += ["            let edge = unsafe { *active.get_unchecked(active_index) } as usize;"]
    else:
        endpoint_start = synapse["source_start"]
        endpoint_count = synapse["source_count"]
        lines += ["            let endpoint_state = unsafe { *active.get_unchecked(active_index) };",
                  f"            if endpoint_state < {endpoint_start} || endpoint_state >= {endpoint_start + endpoint_count} {{ continue; }}",
                  f"            let endpoint = endpoint_state - {endpoint_start};",
                  "            for &edge_index in &edges[offsets[endpoint]..offsets[endpoint+1]] {",
                  "                let edge = edge_index as usize;"]
    lines += ["            " + line for line in vector]
    for state in code["effects"]["writes"]:
        position = state_positions[state]
        symbol = synapse["states"][position]
        dtype = _rust_dtype(symbol)
        value = _storage_store(symbol, symbols[state])
        lines.append(
            f"                unsafe {{ *(({prefix}state_{position}_ptr as *mut {dtype}).add(edge)) = {value}; }}")
    if kind != "synapses_post":
        lines.append("            }")
    lines += ["        }", "    });", "    used_parallel", "}"]
    return lines, f"parallel_plasticity |= {name}({', '.join(calls)});"
















def _v7_summed_parts(model, q, code):
    """Lower serial and raw-pointer forms of one post-summed projection."""
    populations = model["definition"]["populations"]
    synapse = model["definition"]["synapses"][q]
    prefix = f"s{q}_"
    source_pop = synapse["source_population"]
    target_pop = synapse["target_population"]
    state_positions = {
        state["name"]: position for position, state in
        enumerate(populations[target_pop]["states"])}
    state_position = state_positions[code["summed_state"]]
    scalar, vector, symbols = _v7_block(
        model, code, target_pop, "synapse", "edge", q)
    declarations, overrides = [], {}
    for position, symbol in enumerate(synapse["states"]):
        dtype = _rust_dtype(symbol)
        pointer = f"{prefix}summed_state_{position}_ptr"
        declarations.append(
            f"let {pointer} = {prefix}state_{position}.as_ptr() as usize;")
        overrides[symbol["name"]] = _storage_load(
            symbol, f"unsafe {{ *(({pointer} as *const {dtype}).add(edge)) }}")
    for position, symbol in enumerate(synapse["parameters"]):
        if symbol["index_domain"] != "scalar":
            dtype = _rust_dtype(symbol)
            pointer = f"{prefix}summed_parameter_{position}_ptr"
            declarations.append(
                f"let {pointer} = {prefix}parameter_{position}.as_ptr() as usize;")
            overrides[symbol["name"]] = _storage_load(
                symbol, f"unsafe {{ *(({pointer} as *const {dtype}).add(edge)) }}")
    for alias, state in synapse["pre_state_aliases"].items():
        position = next(i for i, item in enumerate(
            populations[source_pop]["states"]) if item["name"] == state)
        symbol = populations[source_pop]["states"][position]
        dtype = _rust_dtype(symbol)
        pointer = f"{prefix}summed_pre_{alias}_{position}_ptr"
        declarations.append(
            f"let {pointer} = p{source_pop}_state_{position}.as_ptr() as usize;")
        overrides[alias] = _storage_load(
            symbol,
            f"unsafe {{ *(({pointer} as *const {dtype}).add(source_state)) }}")
    for alias, state in synapse["post_state_aliases"].items():
        position = next(i for i, item in enumerate(
            populations[target_pop]["states"]) if item["name"] == state)
        symbol = populations[target_pop]["states"][position]
        dtype = _rust_dtype(symbol)
        pointer = f"{prefix}summed_post_{alias}_{position}_ptr"
        declarations.append(
            f"let {pointer} = p{target_pop}_state_{position}.as_ptr() as usize;")
        overrides[alias] = _storage_load(
            symbol,
            f"unsafe {{ *(({pointer} as *const {dtype}).add(target_state)) }}")
    parallel_scalar, parallel_vector, parallel_symbols = _v7_block(
        model, code, target_pop, "synapse", "edge", q,
        overrides=overrides)
    if parallel_scalar != scalar:
        raise ValueError(
            "parallel summed-variable scalar lowering differs from serial lowering")
    return {
        "q": q, "code": code, "synapse": synapse, "prefix": prefix,
        "source_pop": source_pop, "target_pop": target_pop,
        "state_position": state_position,
        "summed_start": synapse["target_start"],
        "summed_count": synapse["target_count"],
        "scalar": scalar, "vector": vector, "symbols": symbols,
        "declarations": declarations, "parallel_vector": parallel_vector,
        "parallel_symbols": parallel_symbols,
    }




def generate_source(model, *, plan=None, distributed_plan=None):
    return _generate_source_verified(migrate_model(model), plan=plan,
                                     distributed_plan=distributed_plan)


def _generate_source_verified(model, *, plan=None, distributed_plan=None):
    """Generate from the private, verified snapshot owned by this build."""
    if plan is None:
        plan = _derive_execution_plan(model)
    else:
        verify_execution_plan(plan, model)
    choices = plan.cpu.choices()
    canonical_slots = plan.cpu.emitter == "slot-v1"
    if distributed_plan is not None:
        from .distributed import verify_distributed_plan
        from .planner import slot_cpu_choices
        verify_distributed_plan(distributed_plan, model)
        choices = slot_cpu_choices(model)
        canonical_slots = True
    if plan.cpu.emitter == "compact-v6" and distributed_plan is None:
        return generate_source_v6(_single_population_v6(model), choices=choices)
    d, inst = model["definition"], model["instance"]
    populations = d["populations"]
    needs_event_dump = choices["needs_event_dump"]
    total_n = inst["neuron_count"]
    max_steps = max(pop["steps"] for pop in populations)
    syn_defs, syn_insts = d["synapses"], inst["synapses"]
    path_prefix = lambda q, r: f"s{q}_" if r == 0 else f"s{q}p{r}_"
    uniform_delays = choices["uniform_delays"]
    route_members = choices["route_members"]
    route_keys = [tuple(key) for key in choices["route_keys"]]
    route_parallel = choices["route_parallel"]
    fused_heterogeneous_groups = choices["fused_heterogeneous_groups"]
    fused_population_groups = choices["fused_population_groups"]
    parallel_state_capable = choices["parallel_state_capable"]
    parallel_poisson_capable = choices["parallel_poisson_capable"]
    parallel_summed_capable = choices["parallel_summed_capable"]
    parallel_plastic_capable = choices["parallel_plastic_capable"]
    parallel_capable = choices["parallel_capable"]
    fused_route_members = {int(k): v for k, v in choices["fused_route_members"].items()}
    fused_route_lookup = {int(k): v for k, v in choices["fused_route_lookup"].items()}
    synapse_routes = {int(k): v for k, v in choices["synapse_routes"].items()}
    population_parallel_work = {int(k): v for k, v in choices["population_parallel_work"].items()}
    population_poisson_work = {int(k): v for k, v in choices["population_poisson_work"].items()}
    population_threshold_fusion = {int(k): v for k, v in choices["population_threshold_fusion"].items()}
    fused_population_lookup = {int(k): v for k, v in choices["fused_population_lookup"].items()}
    fused_summed_groups = {
        int(target): [(q, syn_defs[q]["code_objects"][index]) for q, index in entries]
        for target, entries in choices["fused_summed_groups"].items()
    }
    final_only_summed = {tuple(ref) for ref in choices["final_only_summed"]}

    def summed_final_only(q, code):
        return (q, syn_defs[q]["code_objects"].index(code)) in final_only_summed

    kernels, plastic_calls = [], {}
    for q, kind in choices["plastic_calls"]:
        kernel, call = _v7_plastic_kernel(model, q, kind)
        kernels += kernel
        plastic_calls[q, kind] = call
    route_parallel_calls = {}
    for route, (key, members) in enumerate(zip(route_keys, route_members, strict=True)):
        if route_parallel[route]:
            kernel, call = _v7_target_route_kernel(
                model, route, members, heterogeneous=key[4] is None)
            kernels += kernel
            route_parallel_calls[route] = call
    fused_route_calls = {}
    for group, routes in enumerate(fused_heterogeneous_groups):
        kernel, call = _v7_target_route_kernel(
            model, f"f{group}", fused_route_members[group], heterogeneous=True)
        kernels += kernel
        fused_route_calls[group] = call
    linked_constants = [
        f"const P{p}_LINK_{position}_INDICES: [usize; {len(linked['index']['values'])}] = "
        f"{linked['index']['values']!r};"
        for p, population in enumerate(populations)
        for position, linked in enumerate(population.get("linked_variables", []))
        if linked["index"]["kind"] == "constant"
    ]
    timed_array_constants = [
        f'static {symbol}: &[u8] = include_bytes!("{filename}");'
        for symbol, filename, _payload in timed_array_blobs(model)]
    ffi_types = {
        "f64": "f64", "i64": "i64", "bool": "u8",
    }
    native_declarations = []
    native_symbols = set()
    for function in d.get("functions", []):
        native = function.get("backend_implementations", {}).get("cpu")
        if native is None:
            continue
        if native["symbol"] in native_symbols:
            raise ValueError("native Function C symbols must be globally unique")
        native_symbols.add(native["symbol"])
        try:
            arguments = ", ".join(
                f"arg{position}: {ffi_types[argument['dtype']]}"
                for position, argument in enumerate(function["arguments"]))
            result = ffi_types[function["return_dtype"]]
        except KeyError as error:
            raise ValueError(
                "b2ir-c-abi-v1 supports only f64/i64/bool signatures") from error
        native_declarations.append(
            f'extern "C" {{ fn {native["symbol"]}({arguments}) -> {result}; }}')
    external_bytes = sum(
        8 * (definition["source_count"] + 1) +
        _synapse_edge_count(instance) * (4 + 8 * len(instance["topology"]["initializers"]))
        for definition, instance in zip(syn_defs, syn_insts, strict=True)
        if instance.get("topology", {}).get("kind") == "binary_csr")
    explicit_bytes = 1_048_576 + sum(
        16 * len(instance["source"]) +
        8 * sum(len(values) for values in instance["initial_state"].values()) +
        8 * sum(len(values) for values in instance["parameters"].values()) +
        16 * sum(len(pathway["delay"]) + len(pathway["pending"])
                 for pathway in instance["pathways"])
        for instance in syn_insts)
    explicit_bytes += 8 * sum(
        sum(len(values) for values in population["initial_state"].values()) +
        sum(len(values) for values in population["parameters"].values())
        for population in inst["populations"])
    external_bytes += max(0, explicit_bytes - 128*1_048_576)
    runtime = RUNTIME.replace("128*1_048_576", str(128*1_048_576 + external_bytes))
    if distributed_plan is not None:
        from .mpi_codegen import runtime_source
        runtime = runtime_source(runtime, distributed_plan, streamed=bool(external_bytes))
    if external_bytes and distributed_plan is None:
        # Empirical arrays can be large: do not retain a second byte-for-byte
        # copy of instance.bin after loading the runtime's typed vectors.
        runtime = runtime.replace(
            "struct Reader { bytes: Vec<u8>, at: usize }",
            "struct Reader { input: std::io::BufReader<File>, length: usize, at: usize }")
        reader_start = runtime.index(" fn open(path: &Path) -> Result<Self>")
        reader_end = runtime.index(" fn u64(&mut self)", reader_start)
        limit = 128*1_048_576 + external_bytes
        runtime = runtime[:reader_start] + f"""
 fn open(path:&Path)->Result<Self>{{let file=File::open(path)?;let length=usize::try_from(file.metadata()?.len())?;check(length<={limit}, "native instance exceeds input budget")?;let mut input=std::io::BufReader::with_capacity(65536,file);let mut magic=[0u8;8];input.read_exact(&mut magic)?;check(&magic==b"B2AOT001","invalid native instance")?;Ok(Self{{input,length,at:8}})}}
 fn take<const S:usize>(&mut self)->Result<[u8;S]>{{let end=self.at.checked_add(S).ok_or("instance overflow")?;check(end<=self.length,"truncated instance")?;let mut value=[0u8;S];self.input.read_exact(&mut value)?;self.at=end;Ok(value)}}
""" + runtime[reader_end:]
        runtime = runtime.replace("self.at==self.bytes.len()", "self.at==self.length")
        if distributed_plan is not None:
            # Check the same bytes that become typed arrays, including finite
            # parameter edits. No full input Vec or separate validation/read race.
            runtime = runtime.replace(
                "self.input.read_exact(&mut value)?;self.at=end;",
                'self.input.read_exact(&mut value)?;check(MPI_INSTANCE_BYTES.get(self.at..end) '
                '== Some(value.as_slice()), "MPI instance differs from compiled plan; rebuild required")?;self.at=end;')
    lines = [runtime, *native_declarations, f"const N: usize = {total_n};",
             f"const STEPS: usize = {max_steps};", *linked_constants,
             *timed_array_constants, *kernels,
             "fn execute(mut data: Reader, output: &Path, initialization_started: Instant, input: Option<SpikeInput>) -> Result<()> {"]
    if distributed_plan is not None:
        lines[-1] = "fn execute(mut data: Reader, output: &Path, initialization_started: Instant, mpi: &MpiWorld) -> Result<()> {"
    else:
        input_populations = [str(p) for p, state in enumerate(inst["populations"])
                             if state.get("spike_generator") is not None]
        lines.append("    if let Some(ref supplied) = input { "
                     f"let allowed: &[usize] = &[{','.join(input_populations)}]; "
                     "check(allowed.contains(&supplied.population), \"spike input population is not a generator\")?; }")
    lines += [f"    check(data.usize()? == {len(populations)}, \"instance population count\")?;",
              "    let rng_seed = data.u64()?;",
              f"    let parallel = Parallel::from_env({str(parallel_capable).lower()})?;",
              "    let parallel_threads = parallel.threads();"]
    for p, (pop, pop_inst) in enumerate(zip(populations, inst["populations"], strict=True)):
        prefix = f"p{p}_"
        if distributed_plan is not None:
            from .mpi_partition import population_ranges
            lines += population_ranges(pop, p)
        local_count = f"p{p}_stop-p{p}_start" if distributed_plan is not None else str(pop["count"])
        lines += [f"    let {prefix}dt = data.f64()?;",
                  f"    check({prefix}dt.to_bits() == 0x{pop['dt']}u64, \"population dt mismatch\")?;",
                  f"    check(data.usize()? == {pop['steps']} && data.usize()? == {pop['count']}, \"population shape mismatch\")?;"]
        for position, symbol in enumerate(pop["states"]):
            dtype = _rust_dtype(symbol)
            count = (str(pop["count"]) if distributed_plan is not None and
                     (p, symbol["name"]) in distributed_plan.readonly_pre_states else local_count)
            lines.append(f"    let mut {prefix}state_{position} = data.{dtype}_vec({count})?;")
        for position, symbol in enumerate(pop["parameters"]):
            dtype = _rust_dtype(symbol)
            if symbol["index_domain"] == "scalar":
                lines.append(f"    let {prefix}parameter_{position} = data.{dtype}()?;")
            else:
                lines.append(
                    f"    let {prefix}parameter_{position} = "
                    f"data.{dtype}_vec({local_count})?;")
        if pop_inst["refractory"] is not None:
            lines += [f"    let {prefix}period_ticks = data.usize()?;",
                      f"    let mut {prefix}lastspike = data.f64_vec({local_count})?;",
                      f"    let mut {prefix}not_refractory = data.bool_vec({local_count})?;",
                      f"    let mut {prefix}refractory_until: Vec<usize> = {prefix}lastspike.iter().map(|&last| first_available_tick(last, {prefix}dt, {prefix}period_ticks)).collect();"]
        else:
            lines += [f"    let {prefix}period_ticks = 0usize;",
                      f"    let mut {prefix}lastspike: Vec<f64> = Vec::new();",
                      f"    let mut {prefix}not_refractory: Vec<u8> = Vec::new();",
                      f"    let mut {prefix}refractory_until: Vec<usize> = Vec::new();"]
        population_run_clock = model["run"]["clocks"][pop["clock"]]
        lines += [f"    let mut {prefix}tick = 0usize;",
                  f"    let mut {prefix}end_tick = {population_run_clock['steps']}usize;",
                  f"    let mut {prefix}fired: Vec<usize> = Vec::new();",
                  f"    let mut {prefix}last_fired: Vec<usize> = Vec::new();",
                  f"    let mut {prefix}counts = vec![0usize; " + (f"if mpi.rank == 0 {{ {pop['count']} }} else {{ 0 }}" if distributed_plan is not None else str(pop["count"])) + "];",
                  f"    let mut {prefix}spikes: Vec<(usize,usize)> = Vec::new();"]
        for event_position, _event in enumerate(pop.get("events", [])):
            if _event != "spike":
                lines.append(
                    f"    let mut {prefix}event_{event_position}: Vec<usize> = Vec::new();")
            if needs_event_dump:
                lines.append(
                    f"    let mut {prefix}event_history_{event_position}: "
                    "Vec<(usize,usize)> = Vec::new();")
        public_symbols = {
            symbol["name"]: symbol
            for symbol in (pop["states"] + pop["parameters"] +
                           pop.get("linked_variables", []))
        }
        for column, name in enumerate(pop["monitor"]["variables"]):
            lines.append(
                f"    let mut {prefix}samples_{column}: "
                f"Vec<{_rust_dtype(public_symbols[name])}> = Vec::new();")
        for monitor_position, monitor in enumerate(pop.get("event_monitors", [])):
            lines.append(
                f"    let mut {prefix}event_monitor_{monitor_position}_events: "
                "Vec<(usize,usize)> = Vec::new();")
            for column, name in enumerate(monitor["variables"]):
                lines.append(
                    f"    let mut {prefix}event_monitor_{monitor_position}_samples_{column}: "
                    f"Vec<{_rust_dtype(public_symbols[name])}> = Vec::new();")
        generator = pop_inst.get("spike_generator")
        if generator is not None:
            # Frozen event schedules belong to Instance, including their length.
            # Loading the count permits validated instance replacement without
            # recompiling a model for every input image's event count.
            lines += [f"    let {prefix}generated_count = data.usize()?;"]
            if distributed_plan is not None:
                count = len(generator["spike_ticks"])
                lines += [f'    check({prefix}generated_count <= {count}, "invalid rank generator count")?;']
            lines += [f"    let mut {prefix}generated_ticks = data.usize_vec({prefix}generated_count)?;",
                      f"    let mut {prefix}generated_indices = data.usize_vec({prefix}generated_count)?;",
                      f"    let mut {prefix}generated_cursor = 0usize;"]
    # Every CodeObject clock is a first-class scheduler input. Population
    # ticks remain as hot-loop aliases because existing optimized kernels use
    # them, while c*_tick also covers independent run_regularly clocks.
    for clock, (definition, run_clock) in enumerate(zip(
            d["clocks"], model["run"]["clocks"], strict=True)):
        clock_dt = _number(definition["dt"])
        lines += [f"    let c{clock}_dt = {clock_dt!r}f64;",
                  f"    let mut c{clock}_tick = 0usize;",
                  f"    let mut c{clock}_end_tick = {run_clock['steps']}usize;"]
    for q, (syn_def, syn_inst) in enumerate(zip(syn_defs, syn_insts, strict=True)):
        prefix = f"s{q}_"
        if distributed_plan is not None:
            from .mpi_partition import synapse_loading
            lines += synapse_loading(model, q, distributed_plan)
            continue
        edge_count = _synapse_edge_count(syn_inst)
        topology = syn_inst.get("topology", {"kind": "explicit"})
        lines += [f"    let {prefix}edge_count = {edge_count}usize;"]
        if topology["kind"] == "explicit":
            lines += [f"    let {prefix}source_index = data.u32_vec({prefix}edge_count)?;",
                      f"    let {prefix}target_index = data.u32_vec({prefix}edge_count)?;",
                      f"    let ({prefix}offsets, {prefix}edges) = source_csr_u32(&{prefix}source_index, {syn_def['source_count']});"]
        elif topology["kind"] == "binary_csr":
            lines += [f"    let {prefix}offsets = data.usize_vec({syn_def['source_count'] + 1})?;",
                      f"    check({prefix}offsets[0] == 0 && {prefix}offsets[{syn_def['source_count']}] == {prefix}edge_count && {prefix}offsets.windows(2).all(|w| w[0] <= w[1]), \"invalid CSR offsets\")?;",
                      f"    let {prefix}target_index = data.u32_vec({prefix}edge_count)?;",
                      f"    let {prefix}source_index: Vec<u32> = Vec::new();",
                      f"    let {prefix}edges: Vec<u32> = Vec::new();"]
        elif topology["kind"] == "fixed_total":
            lines += [f"    let ({prefix}offsets, {prefix}target_index) = fixed_total_topology(&parallel, {syn_def['source_count']}, {syn_def['target_count']}, {prefix}edge_count, {topology['seed']}u64)?;",
                      f"    let {prefix}source_index: Vec<u32> = Vec::new();",
                      f"    let {prefix}edges: Vec<u32> = Vec::new();"]
        elif topology["kind"] == "fixed_indegree":
            lines += [f"    let ({prefix}offsets, {prefix}target_index) = fixed_indegree_topology(&parallel, {syn_def['source_count']}, {syn_def['target_count']}, {topology['indegree']}usize, {prefix}edge_count, {topology['seed']}u64)?;",
                      f"    let {prefix}source_index: Vec<u32> = Vec::new();",
                      f"    let {prefix}edges: Vec<u32> = Vec::new();"]
        else:
            raise ValueError("unsupported procedural topology")
        lines += [
                  f'    check({prefix}source_index.iter().all(|&value| value < {syn_def["source_count"]}), "source index out of range")?;',
                  f'    check({prefix}target_index.iter().all(|&value| value < {syn_def["target_count"]}), "target index out of range")?;']
        for position, symbol in enumerate(syn_def["states"]):
            dtype = _rust_dtype(symbol)
            lines.append(
                f"    let mut {prefix}state_{position} = "
                f"data.{dtype}_vec({prefix}edge_count)?;")
        for position, symbol in enumerate(syn_def["parameters"]):
            dtype = _rust_dtype(symbol)
            if symbol["index_domain"] == "scalar":
                lines.append(f"    let {prefix}parameter_{position} = data.{dtype}()?;")
            elif topology["kind"] in {"fixed_total", "fixed_indegree"}:
                initializer = topology["initializers"][symbol["name"]]
                initializer_function, initializer_arguments = _initializer_call(
                    initializer)
                suffix = (".into_iter().map(checked_f32).collect::<Vec<_>>()"
                          if dtype == "f32" else "")
                lines.append(
                    f"    let {prefix}parameter_{position} = {initializer_function}(&parallel, "
                    f"{prefix}edge_count, {topology['seed']}u64, "
                    f"{initializer_arguments})?{suffix};")
            else:
                lines.append(
                    f"    let {prefix}parameter_{position} = "
                    f"data.{dtype}_vec({prefix}edge_count)?;")
        for monitor_position, monitor in enumerate(
                syn_def.get("state_monitors", [])):
            for column, source in enumerate(monitor["sources"]):
                lines.append(
                    f"    let mut {prefix}monitor_{monitor_position}_samples_{column}: "
                    f"Vec<{_rust_dtype(source)}> = Vec::new();")
        if any(pathway["kind"] == "post" for pathway in syn_inst["pathways"]):
            lines.append(
                f"    let ({prefix}target_offsets, {prefix}target_edges) = "
                f"source_csr_u32(&{prefix}target_index, {syn_def['target_count']});")
        for r, pathway in enumerate(syn_inst["pathways"]):
            pathway_prefix = path_prefix(q, r)
            delay_values = pathway["delay_ticks"]
            delay_initializer = pathway.get("delay_initializer")
            uniform_delay = (delay_values[0] if delay_values and
                             all(value == delay_values[0] for value in delay_values)
                             else None)
            endpoint_count = (syn_def["source_count"] if pathway["kind"] == "pre"
                              else syn_def["target_count"])
            endpoint_steps = populations[(syn_def["source_population"]
                                          if pathway["kind"] == "pre" else
                                          syn_def["target_population"])]["steps"]
            if delay_initializer is None:
                lines += [f"    let {pathway_prefix}delay_values = data.usize_vec({len(delay_values)})?;"]
            else:
                dt = _number(populations[(syn_def["source_population"]
                                          if pathway["kind"] == "pre" else
                                          syn_def["target_population"])]["dt"])
                initializer_function, initializer_arguments = _initializer_call(
                    delay_initializer, ticks=True)
                lines += [
                    f"    let {pathway_prefix}delay_values = {initializer_function}("
                    f"&parallel, {prefix}edge_count, {topology['seed']}u64, "
                    f"{initializer_arguments}, {dt!r}f64)?;"]
            lines += [
                      f"    let {pathway_prefix}pending_count = data.usize()?;",
                      f"    let {pathway_prefix}pending_ticks = data.usize_vec({pathway_prefix}pending_count)?;",
                      f"    let {pathway_prefix}pending_items = data.usize_vec({pathway_prefix}pending_count)?;"]
            pending_limit = endpoint_count if uniform_delay is not None else edge_count
            lines.append(
                f'    check({pathway_prefix}pending_items.iter().all(|&item| item < {pending_limit}), "pending pathway item out of range")?;')
            if uniform_delay is not None:
                lines += [f'    check({pathway_prefix}delay_values.iter().all(|&delay| delay == {uniform_delay}), "instance delay specialization mismatch")?;',
                          f"    let {pathway_prefix}delay_ticks = {uniform_delay}usize;"]
                if (pathway["kind"] == "post" and
                        (q, "synapses_post") in plastic_calls):
                    lines.append(
                        f"    let mut {pathway_prefix}edge_scratch: Vec<u32> = Vec::new();")
                needs_private_queue = not canonical_slots and not (r == 0 and pathway["kind"] == "pre")
                if needs_private_queue and (uniform_delay > 0 or pathway["pending"]):
                    lines += [f"    let mut {pathway_prefix}queue: Vec<Vec<usize>> = vec![Vec::new(); ({uniform_delay}+1).min({endpoint_steps})];",
                              f"    for (&delivery, &item) in {pathway_prefix}pending_ticks.iter().zip(&{pathway_prefix}pending_items) {{ let slot=delivery%{pathway_prefix}queue.len(); {pathway_prefix}queue[slot].push(item); }}"]
            else:
                heterogeneous_parallel = (
                    r == 0 and pathway["kind"] == "pre" and
                    route_parallel[synapse_routes[q]])
                lines += [f"    let {pathway_prefix}delay_ticks = {pathway_prefix}delay_values;",
                          f"    let {pathway_prefix}max_delay = {pathway_prefix}delay_ticks.iter().copied().max().unwrap_or(0);"]
                if heterogeneous_parallel:
                    lines += [
                        f"    let {pathway_prefix}queue_slots = "
                        f"({pathway_prefix}max_delay+1).min({endpoint_steps});",
                        f"    let mut {pathway_prefix}queue: Vec<Vec<usize>> = "
                        f"(0..{pathway_prefix}queue_slots*parallel.threads())"
                        ".map(|_|Vec::new()).collect();"]
                elif not canonical_slots:
                    lines += [f"    let mut {pathway_prefix}queue: Vec<Vec<usize>> = vec![Vec::new(); "
                              f"({pathway_prefix}max_delay+1).min({endpoint_steps})];",
                              f"    for (&delivery, &item) in {pathway_prefix}pending_ticks.iter().zip(&{pathway_prefix}pending_items) {{ let slot=delivery%{pathway_prefix}queue.len(); {pathway_prefix}queue[slot].push(item); }}"]
        lines += [f"    let mut {prefix}delivered = 0usize;",
                  f"    let mut {prefix}post_delivered = 0usize;"]
    for route, route_key in enumerate(route_keys):
        source_pop, _start, _count, _event, delay = route_key
        if not canonical_slots and delay is not None and delay > 0:
            first = route_members[route][0]
            if any(syn_insts[q]["pathways"][0]["pending"] !=
                   syn_insts[first]["pathways"][0]["pending"]
                   for q in route_members[route][1:]):
                raise ValueError("fused projections have inconsistent pending events")
            lines.append(
                f"    let mut r{route}_queue: Vec<Vec<usize>> = vec![Vec::new(); "
                f"({delay}+1).min({populations[source_pop]['steps']})];")
            lines.append(
                f"    for (&delivery, &item) in s{first}_pending_ticks.iter().zip(&s{first}_pending_items) {{ let slot=delivery%r{route}_queue.len(); r{route}_queue[slot].push(item); }}")
    clock_epsilon = min(_number(clock["dt"]) for clock in d["clocks"]) * 1e-12
    lines += [
        f"    let run_clock_starts = data.usize_vec({len(d['clocks'])})?;",
        "    let final_time = data.f64()?;",
        "    check(final_time.is_finite() && final_time >= 0.0, \"invalid final time\")?;",
    ]
    for p, pop in enumerate(populations):
        clock = pop["clock"]
        lines += [f"    p{p}_tick = run_clock_starts[{clock}];",
                  f"    p{p}_end_tick = p{p}_tick.checked_add(p{p}_end_tick).ok_or(\"population clock overflow\")?;"]
    for clock in range(len(d["clocks"])):
        lines += [f"    c{clock}_tick = run_clock_starts[{clock}];",
                  f"    c{clock}_end_tick = c{clock}_tick.checked_add(c{clock}_end_tick).ok_or(\"clock overflow\")?;"]
    if distributed_plan is None:
        for p, (pop, pop_inst) in enumerate(zip(
                populations, inst["populations"], strict=True)):
            if pop_inst.get("spike_generator") is None:
                continue
            prefix = f"p{p}_"
            lines += [
                f"    if let Some(ref supplied) = input {{ if supplied.population == {p} {{",
                f"        supplied.validate({pop['count']}, run_clock_starts[{pop['clock']}], {pop['steps']})?;",
                f"        {prefix}generated_ticks = supplied.ticks.clone();",
                f"        {prefix}generated_indices = supplied.indices.clone();",
                "    } }",
            ]
    lines += ["    data.end()?;", "    drop(data);"]
    synapse_owner_maps = {}
    target_owner_synapses = {}
    for route, members in enumerate(route_members):
        if not route_parallel[route]:
            continue
        for q in members:
            target_owner_synapses.setdefault(
                syn_defs[q]["target_population"], []).append(q)
    for q in choices["summed_owner_synapses"]:
        target_owner_synapses.setdefault(syn_defs[q]["target_population"], []).append(q)
    for target_population, entries in fused_summed_groups.items():
        for q, _code in entries:
            target_owner_synapses.setdefault(target_population, []).append(q)
    for target_population, members in sorted(target_owner_synapses.items()):
        members = list(dict.fromkeys(members))
        count = populations[target_population]["count"]
        owner_map = f"p{target_population}_target_owners"
        lines += [f"    let {owner_map} = if parallel.threads() > 1 {{",
                  f"        let mut target_degrees = vec![0usize; {count}];"]
        for q in members:
            synapse = syn_defs[q]
            lines.append(
                f"        for &target in &s{q}_target_index {{ "
                f"target_degrees[{synapse['target_start']} + target as usize] += 1; }}")
            synapse_owner_maps[q] = owner_map
        lines += ["        degree_balanced_target_owners(&target_degrees, parallel.threads())",
                  f"    }} else {{ vec![0usize; {count}] }};"]
    for q in sorted(synapse_owner_maps):
        synapse = syn_defs[q]
        prefix = f"s{q}_"
        owner_map = synapse_owner_maps[q]
        owner_builder = (
            f"target_owner_csr_contiguous(&{prefix}offsets, &{prefix}target_index, "
            if not _explicit_topology(syn_insts[q]) else
            f"target_owner_csr_u32(&{prefix}offsets, &{prefix}edges, &{prefix}target_index, "
        )
        lines.append(
            f"    let {prefix}target_owner = Some({owner_builder}"
            f"{synapse['source_count']}, {synapse['target_start']}, "
            f"&{owner_map}, parallel.threads()));")
    for p in sorted(p for p, fused in population_threshold_fusion.items() if fused):
        lines.append(
            f"    let mut p{p}_fired_lanes: Vec<Vec<usize>> = "
            "(0..parallel.threads()).map(|_| Vec::new()).collect();")
    if canonical_slots:
        from .slot_codegen import initialize_queues
        lines += initialize_queues(model, distributed=distributed_plan is not None)
    phase_names = ("scheduler", "groups", "neuron_state", "synapse_state",
                   "threshold", "primary_events", "primary_enqueue",
                   "primary_apply", "additional_pre",
                   "poisson_input", "post", "reset")
    lines += ["    let mut parallel_on_pre = false;",
              "    let mut parallel_event_enqueue = false;",
              "    let mut parallel_plasticity = false;",
              "    let mut parallel_summed_variable = false;",
              "    let mut parallel_synapse_regular = false;",
              "    let mut parallel_synapse_state = false;",
              "    let mut parallel_poisson_input = false;",
              '    let phase_profile_enabled = env_flag("B2_AOT_PROFILE_PHASES")?;']
    lines += [f"    let mut phase_{name}_seconds = 0.0f64;"
              for name in phase_names]
    lines += ["    let mut phase_profile_ticks = 0usize;",
              f"    let clock_epsilon = {clock_epsilon!r}f64;"]
    if distributed_plan is not None and distributed_plan.rank_backends:
        lines.append("    let mut mpi_gpu = MpiGpu::new(mpi.rank)?;")
    if distributed_plan is not None:
        from .mpi_codegen import initialization_timing
        lines += initialization_timing()
    lines += ["    let initialization_seconds = initialization_started.elapsed().as_secs_f64();",
              "    let started = Instant::now();", "    loop {",
              "        let phase_scheduler_started = phase_start(phase_profile_enabled);",
              "        let mut time = f64::INFINITY;"]
    for clock in range(len(d["clocks"])):
        lines.append(
            f"        if c{clock}_tick < c{clock}_end_tick {{ "
            f"time = time.min(c{clock}_tick as f64 * c{clock}_dt); }}")
    lines += ["        if !time.is_finite() { break; }"]
    for clock in range(len(d["clocks"])):
        lines.append(
            f"        let c{clock}_active = c{clock}_tick < c{clock}_end_tick && "
            f"((c{clock}_tick as f64*c{clock}_dt-time).abs() <= clock_epsilon);")
    for p, pop in enumerate(populations):
        lines.append(
            f"        let p{p}_active = c{pop['clock']}_active;")
        if not canonical_slots:
            lines += [f"        let mut p{p}_threshold_fused = false;",
                      f"        if p{p}_active {{ p{p}_fired.clear(); }}"]
            for event_position, _event in enumerate(pop.get("events", [])):
                if _event != "spike":
                    lines.append(
                        f"        if p{p}_active {{ p{p}_event_{event_position}.clear(); }}")
    lines += ["        phase_finish(&mut phase_scheduler_seconds, phase_scheduler_started);",
              "        phase_profile_ticks += usize::from(phase_profile_enabled);",
              "        let phase_groups_started = phase_start(phase_profile_enabled);"]
    if canonical_slots:
        from .slot_codegen import emit_schedule
        schedule = emit_schedule(model, plan, needs_event_dump=needs_event_dump,
                                 distributed=distributed_plan is not None,
                                 gpu_offload=bool(distributed_plan and distributed_plan.rank_backends))
        if distributed_plan is not None:
            from .mpi_partition import localize_accesses
            schedule = localize_accesses(schedule, model, distributed_plan)
        lines += schedule
    else:
        for p, pop in enumerate(populations):
            code = next((item for item in pop["code_objects"]
                         if item["kind"] == "subexpression_update"), None)
            if code is None:
                continue
            positions = {state["name"]: position for position, state in
                         enumerate(pop["states"])}
            scalar, vector, symbols = _v7_block(model, code, p, "neuron")
            lines += [f"        if p{p}_active {{", _v7_local_time(model, p)] + ["    " + line for line in scalar]
            lines += [f"            for i in 0..{pop['count']} {{"]
            lines += ["    " + line for line in vector]
            for name in code["effects"]["writes"]:
                value = _storage_store(pop["states"][positions[name]], symbols[name])
                lines.append(
                    f"                p{p}_state_{positions[name]}[i] = {value};")
            lines += ["            }", "        }"]
        # Synaptic ``constant over dt`` values are stored state updated in
        # before_start, before StateMonitor and every regular schedule slot.
        for q, syn_def in enumerate(syn_defs):
            prefix = f"s{q}_"
            source_pop = syn_def["source_population"]
            code = next((item for item in syn_def["code_objects"]
                         if item["kind"] == "synapse_subexpression_update"), None)
            if code is None:
                continue
            state_positions = {state["name"]: position for position, state in
                               enumerate(syn_def["states"])}
            scalar, vector, symbols = _v7_block(
                model, code, source_pop, "synapse", "edge", q)
            lines += [f"        if p{source_pop}_active {{", _v7_local_time(model, source_pop)] + [
                "    " + line for line in scalar]
            lines += [f"            for edge in 0..{prefix}edge_count {{",
                      f"                let source = unsafe {{ *{prefix}source_index.get_unchecked(edge) }} as usize;",
                      f"                let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                      f"                let source_state = source + {syn_def['source_start']};",
                      f"                let target_state = target + {syn_def['target_start']};"]
            lines += ["    " + line for line in vector]
            for name in code["effects"]["writes"]:
                value = _storage_store(
                    syn_def["states"][state_positions[name]], symbols[name])
                lines.append(
                    f"                {prefix}state_{state_positions[name]}[edge] = {value};")
            lines += ["            }", "        }"]
        # start monitors
        for p, pop in enumerate(populations):
            state_positions = {s["name"]: i for i, s in enumerate(pop["states"])}
            parameter_positions = {s["name"]: i for i, s in enumerate(pop["parameters"])}
            lines.append(
                f"        if p{p}_active && p{p}_tick >= p{p}_end_tick - "
                f"{pop['monitor']['window_steps']} {{")
            for neuron in pop["monitor"]["record"]:
                for column, name in enumerate(pop["monitor"]["variables"]):
                    if name in state_positions:
                        value = f"p{p}_state_{state_positions[name]}[{neuron}]"
                    elif name in {linked["name"] for linked in
                                  pop.get("linked_variables", [])}:
                        position = next(
                            position for position, linked in enumerate(
                                pop["linked_variables"])
                            if linked["name"] == name)
                        value = _v7_link_value(model, p, position, str(neuron))
                    else:
                        position = parameter_positions[name]
                        scalar = pop["parameters"][position]["index_domain"] == "scalar"
                        value = (f"p{p}_parameter_{position}" if scalar else
                                 f"p{p}_parameter_{position}[{neuron}]")
                    lines.append(f"            p{p}_samples_{column}.push({value});")
            lines.append("        }")
        # Brian2 summed variables run in groups/order=-1: clear the exact endpoint
        # window, then accumulate each expression in connection-creation order.
        fused_summed_keys = {
            (q, code["name"]) for entries in fused_summed_groups.values()
            for q, code in entries}
        fused_summed_first = {
            (entries[0][0], entries[0][1]["name"]): entries
            for entries in fused_summed_groups.values()}
        for q, syn_def in enumerate(syn_defs):
            prefix = f"s{q}_"
            source_pop = syn_def["source_population"]
            target_pop = syn_def["target_population"]
            for code in (item for item in syn_def["code_objects"]
                         if item["kind"] == "summed_variable" and item["order"] < 0):
                fused_key = (q, code["name"])
                if fused_key in fused_summed_keys:
                    if fused_key not in fused_summed_first:
                        continue
                    fused_entries = fused_summed_first[fused_key]
                    fused_parts = [
                        _v7_summed_parts(model, entry_q, entry_code)
                        for entry_q, entry_code in fused_entries]
                    summed_pop = fused_parts[0]["target_pop"]
                    total_edges = sum(
                        _synapse_edge_count(syn_insts[item["q"]])
                        for item in fused_parts)
                    total_work = sum(
                        _synapse_edge_count(syn_insts[item["q"]]) *
                        code_work(item["code"]) for item in fused_parts)
                    average_work = max(
                        1, (total_work + total_edges - 1) // total_edges)
                    final_only = all(
                        summed_final_only(item["q"], item["code"])
                        for item in fused_parts)
                    final_guard = (
                        f" && c{code['clock']}_tick + 1 == c{code['clock']}_end_tick"
                        if final_only else "")
                    lines.append(
                        f"        if p{summed_pop}_active{final_guard} {{")
                    lines.append(_v7_local_time(model, summed_pop))
                    for item in fused_parts:
                        lines.append(
                            f"            p{summed_pop}_state_{item['state_position']}"
                            f"[{item['summed_start']}.."
                            f"{item['summed_start'] + item['summed_count']}].fill(0.0);")
                        lines += ["    " + line for line in item["scalar"]]
                    lines += [
                        f"            if parallel.is_parallel({total_edges}, {average_work}) {{",
                        "                parallel_summed_variable = true;"]
                    for item in fused_parts:
                        prefix = item["prefix"]
                        lines += ["                " + line
                                  for line in item["declarations"]]
                        lines += [
                            f"                let {prefix}summed_destination_ptr = p{summed_pop}_state_{item['state_position']}.as_mut_ptr() as usize;",
                            f"                let {prefix}summed_owner_ptr = {prefix}target_owner.as_ref().unwrap() as *const TargetOwnerCsr as usize;",
                            f"                let {prefix}summed_target_index_ptr = {prefix}target_index.as_ptr() as usize;"]
                    lines.append("                parallel.for_lanes(move |owner| {")
                    for item in fused_parts:
                        prefix = item["prefix"]
                        entry_synapse = item["synapse"]
                        lines += [
                            f"                    let {prefix}owner_csr = unsafe {{ &*({prefix}summed_owner_ptr as *const TargetOwnerCsr) }};",
                            f"                    let {prefix}owner_offsets = &{prefix}owner_csr.offsets[owner];",
                            f"                    let {prefix}owner_edges = &{prefix}owner_csr.edges[owner];",
                            f"                    for source in 0..{entry_synapse['source_count']} {{",
                            f"                        for &edge_index in &{prefix}owner_edges[{prefix}owner_offsets[source]..{prefix}owner_offsets[source+1]] {{",
                            "                            let edge = edge_index as usize;",
                            f"                            let target = unsafe {{ *(({prefix}summed_target_index_ptr as *const u32).add(edge)) }} as usize;",
                            f"                            let source_state = source + {entry_synapse['source_start']};",
                            f"                            let target_state = target + {entry_synapse['target_start']};"]
                        lines += ["                    " + line
                                  for line in item["parallel_vector"]]
                        lines += [
                            f"                            unsafe {{ *(({prefix}summed_destination_ptr as *mut f64).add(target_state)) += {item['parallel_symbols']['_synaptic_var']}; }}",
                            "                        }", "                    }"]
                    lines += ["                });", "            } else {"]
                    for item in fused_parts:
                        prefix = item["prefix"]
                        entry_synapse = item["synapse"]
                        if _explicit_topology(syn_insts[item["q"]]):
                            lines += [
                                f"                for edge in 0..{prefix}edge_count {{",
                                f"                    let source = unsafe {{ *{prefix}source_index.get_unchecked(edge) }} as usize;"]
                            serial_close = ["                }"]
                        else:
                            lines += [
                                f"                for source in 0..{entry_synapse['source_count']} {{",
                                f"                    for edge in {prefix}offsets[source]..{prefix}offsets[source+1] {{"]
                            serial_close = ["                    }", "                }"]
                        lines += [
                            f"                    let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                            f"                    let source_state = source + {entry_synapse['source_start']};",
                            f"                    let target_state = target + {entry_synapse['target_start']};"]
                        lines += ["    " + line for line in item["vector"]]
                        lines += [
                            f"                    p{summed_pop}_state_{item['state_position']}[target_state] += {item['symbols']['_synaptic_var']};",
                            *serial_close]
                    lines += ["            }", "        }"]
                    continue
                target_is_pre = code["summed_target"] == "pre"
                summed_pop = source_pop if target_is_pre else target_pop
                summed_start = (syn_def["source_start"] if target_is_pre else
                                syn_def["target_start"])
                summed_count = (syn_def["source_count"] if target_is_pre else
                                syn_def["target_count"])
                state_positions = {
                    state["name"]: position for position, state in
                    enumerate(populations[summed_pop]["states"])}
                state_position = state_positions[code["summed_state"]]
                summed_dtype = _rust_dtype(
                    populations[summed_pop]["states"][state_position])
                destination = "source_state" if target_is_pre else "target_state"
                scalar, vector, symbols = _v7_block(
                    model, code, summed_pop, "synapse", "edge", q)
                parallel_summed = (
                    summed_dtype == "f64" and not target_is_pre and
                    q in synapse_owner_maps and
                    parallel_phase_capable(
                        _synapse_edge_count(syn_insts[q]), code_work(code)))
                final_guard = (
                    f" && c{code['clock']}_tick + 1 == c{code['clock']}_end_tick"
                    if summed_final_only(q, code) else "")
                lines += [f"        if c{code['clock']}_active{final_guard} {{",
                          _v7_local_time(model, source_pop),
                          f"            p{summed_pop}_state_{state_position}[{summed_start}..{summed_start + summed_count}].fill(0.0);"]
                if parallel_summed:
                    declarations, overrides = [], {}
                    for position, symbol in enumerate(syn_def["states"]):
                        dtype = _rust_dtype(symbol)
                        pointer = f"{prefix}summed_state_{position}_ptr"
                        declarations.append(
                            f"let {pointer} = {prefix}state_{position}.as_ptr() as usize;")
                        overrides[symbol["name"]] = _storage_load(
                            symbol,
                            f"unsafe {{ *(({pointer} as *const {dtype}).add(edge)) }}")
                    for position, symbol in enumerate(syn_def["parameters"]):
                        if symbol["index_domain"] != "scalar":
                            dtype = _rust_dtype(symbol)
                            pointer = f"{prefix}summed_parameter_{position}_ptr"
                            declarations.append(
                                f"let {pointer} = {prefix}parameter_{position}.as_ptr() as usize;")
                            overrides[symbol["name"]] = _storage_load(
                                symbol,
                                f"unsafe {{ *(({pointer} as *const {dtype}).add(edge)) }}")
                    for alias, state in syn_def["pre_state_aliases"].items():
                        position = next(i for i, item in enumerate(
                            populations[source_pop]["states"]) if item["name"] == state)
                        symbol = populations[source_pop]["states"][position]
                        dtype = _rust_dtype(symbol)
                        pointer = f"{prefix}summed_pre_state_{position}_ptr"
                        declarations.append(
                            f"let {pointer} = p{source_pop}_state_{position}.as_ptr() as usize;")
                        overrides[alias] = _storage_load(
                            symbol,
                            f"unsafe {{ *(({pointer} as *const {dtype}).add(source_state)) }}")
                    for alias, state in syn_def["post_state_aliases"].items():
                        position = next(i for i, item in enumerate(
                            populations[target_pop]["states"]) if item["name"] == state)
                        symbol = populations[target_pop]["states"][position]
                        dtype = _rust_dtype(symbol)
                        pointer = f"{prefix}summed_post_state_{position}_ptr"
                        declarations.append(
                            f"let {pointer} = p{target_pop}_state_{position}.as_ptr() as usize;")
                        overrides[alias] = _storage_load(
                            symbol,
                            f"unsafe {{ *(({pointer} as *const {dtype}).add(target_state)) }}")
                    parallel_scalar, parallel_vector, parallel_symbols = _v7_block(
                        model, code, summed_pop, "synapse", "edge", q,
                        overrides=overrides)
                    if parallel_scalar != scalar:
                        raise ValueError(
                            "parallel summed-variable scalar lowering differs from serial lowering")
                    work = code_work(code)
                    lines += ["    " + line for line in parallel_scalar]
                    lines += ["            if parallel.is_parallel("
                              f"{prefix}edge_count, {work}) {{",
                              "                parallel_summed_variable = true;"]
                    lines += ["                " + line for line in declarations]
                    lines += [
                        f"                let {prefix}summed_destination_ptr = p{summed_pop}_state_{state_position}.as_mut_ptr() as usize;",
                        f"                let {prefix}summed_owner_ptr = {prefix}target_owner.as_ref().unwrap() as *const TargetOwnerCsr as usize;",
                        f"                let {prefix}summed_target_index_ptr = {prefix}target_index.as_ptr() as usize;",
                        "                parallel.for_lanes(move |owner| {",
                        f"                    let owner_csr = unsafe {{ &*({prefix}summed_owner_ptr as *const TargetOwnerCsr) }};",
                        "                    let owner_offsets = &owner_csr.offsets[owner];",
                        "                    let owner_edges = &owner_csr.edges[owner];",
                        f"                    for source in 0..{syn_def['source_count']} {{",
                        "                        for &edge_index in &owner_edges[owner_offsets[source]..owner_offsets[source+1]] {",
                        "                            let edge = edge_index as usize;",
                        f"                            let target = unsafe {{ *(({prefix}summed_target_index_ptr as *const u32).add(edge)) }} as usize;",
                        f"                            let source_state = source + {syn_def['source_start']};",
                        f"                            let target_state = target + {syn_def['target_start']};",
                    ]
                    lines += ["                    " + line for line in parallel_vector]
                    lines += [
                        f"                            unsafe {{ *(({prefix}summed_destination_ptr as *mut f64).add(target_state)) += {parallel_symbols['_synaptic_var']}; }}",
                        "                        }", "                    }",
                        "                });", "            } else {",
                    ]
                else:
                    lines += ["    " + line for line in scalar]
                if _explicit_topology(syn_insts[q]):
                    lines += [f"                for edge in 0..{prefix}edge_count {{",
                              f"                    let source = unsafe {{ *{prefix}source_index.get_unchecked(edge) }} as usize;"]
                    serial_close = ["                }"]
                else:
                    lines += [f"                for source in 0..{syn_def['source_count']} {{",
                              f"                    for edge in {prefix}offsets[source]..{prefix}offsets[source+1] {{"]
                    serial_close = ["                    }", "                }"]
                lines += [
                          f"                let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                          f"                let source_state = source + {syn_def['source_start']};",
                          f"                let target_state = target + {syn_def['target_start']};"]
                lines += ["    " + line for line in vector]
                destination_value = (
                    f"p{summed_pop}_state_{state_position}[{destination}] + "
                    f"{symbols['_synaptic_var']}" if summed_dtype == "f64" else
                    f"checked_f32(p{summed_pop}_state_{state_position}[{destination}] "
                    f"as f64 + {symbols['_synaptic_var']})")
                lines += [
                    f"                p{summed_pop}_state_{state_position}"
                    f"[{destination}] = {destination_value};",
                    *serial_close]
                if parallel_summed:
                    lines.append("            }")
                lines.append("        }")

        # population state updates
        lines += ["        phase_finish(&mut phase_groups_seconds, phase_groups_started);",
                  "        let phase_neuron_state_started = phase_start(phase_profile_enabled);"]
        for p, pop in enumerate(populations):
            fused_group = fused_population_lookup.get(p)
            if fused_group is not None:
                if p != fused_group[0]:
                    continue
                group_data = []
                for group_population in fused_group:
                    group_pop = populations[group_population]
                    group_code = next(
                        code for code in group_pop["code_objects"]
                        if code["kind"] == "state_update")
                    group_positions = {
                        state["name"]: position for position, state in
                        enumerate(group_pop["states"])}
                    group_scalar, _group_vector, _group_symbols = _v7_block(
                        model, group_code, group_population, "neuron")
                    fuse_threshold = population_threshold_fusion[group_population]
                    (declarations, parallel_scalar, parallel_vector,
                     parallel_writes, parallel_threshold) = _v7_parallel_neuron(
                        model, group_population, group_code, group_positions,
                        fuse_threshold=fuse_threshold)
                    if parallel_scalar != group_scalar:
                        raise ValueError(
                            "parallel neuron scalar lowering differs from serial lowering")
                    group_data.append((
                        group_population, group_pop, declarations,
                        parallel_scalar, parallel_vector, parallel_writes,
                        parallel_threshold, fuse_threshold))
                first_population = fused_group[0]
                lines += [f"        if p{first_population}_active {{", _v7_local_time(model, first_population)]
                for (group_population, _group_pop, declarations, _scalar,
                     _vector, _writes, _threshold, fuse_threshold) in group_data:
                    lines += ["            " + line for line in declarations]
                    if fuse_threshold:
                        lines.append(
                            f"            let p{group_population}_fired_lanes_ptr = "
                            f"p{group_population}_fired_lanes.as_mut_ptr() as usize;")
                lines.append("            // fused population clock group")
                lines.append("            parallel.for_lanes(move |rank| {")
                for (group_population, group_pop, _declarations, parallel_scalar,
                     parallel_vector, parallel_writes, parallel_threshold,
                     fuse_threshold) in group_data:
                    lines.append("                {")
                    if fuse_threshold:
                        lines += [
                            f"                    let lane_fired = unsafe {{ &mut *((p{group_population}_fired_lanes_ptr as *mut Vec<usize>).add(rank)) }};",
                            "                    lane_fired.clear();",
                        ]
                    lines += ["                    " + line
                              for line in parallel_scalar]
                    lines += [
                        f"                    let start = {group_pop['count']}*rank/parallel_threads;",
                        f"                    let end = {group_pop['count']}*(rank+1)/parallel_threads;",
                        "                    for i in start..end {",
                    ]
                    if (group_pop["refractory"] is not None and
                            group_pop["refractory"]["mode"] == "fixed"):
                        lines.append(
                            f"                        unsafe {{ *((p{group_population}_not_refractory_ptr as *mut u8).add(i)) = "
                            f"u8::from(p{group_population}_tick >= *((p{group_population}_refractory_until_ptr as *const usize).add(i))); }}")
                    lines += ["            " + line for line in parallel_vector]
                    lines += ["                        " + line
                              for line in parallel_writes]
                    lines.append("                    }")
                    if fuse_threshold:
                        threshold_scalar, threshold_vector, threshold_writes = (
                            parallel_threshold)
                        lines += ["                    " + line
                                  for line in threshold_scalar]
                        lines.append("                    for i in start..end {")
                        lines += ["                " + line
                                  for line in threshold_vector]
                        lines += ["                        " + line
                                  for line in threshold_writes]
                        lines.append("                    }")
                    lines.append("                }")
                lines.append("            });")
                for (group_population, _group_pop, _declarations, _scalar,
                     _vector, _writes, _threshold, fuse_threshold) in group_data:
                    if fuse_threshold:
                        lines += [
                            f"            p{group_population}_threshold_fused = true;",
                            f"            for lane in &mut p{group_population}_fired_lanes {{ p{group_population}_fired.append(lane); }}",
                        ]
                lines.append("        }")
                continue
            code = next((code for code in pop["code_objects"]
                         if code["kind"] == "state_update"), None)
            if code is None:
                continue
            scalar, vector, symbols = _v7_block(model, code, p, "neuron")
            positions = {s["name"]: i for i, s in enumerate(pop["states"])}
            lines += [f"        if p{p}_active {{", _v7_local_time(model, p)] + ["    " + line for line in scalar]
            work = population_parallel_work[p]
            fuse_threshold = population_threshold_fusion[p]
            pop_parallel_capable = parallel_phase_capable(pop["count"], work)
            if not pop_parallel_capable:
                lines += [f"            for i in 0..{pop['count']} {{"]
                if (pop["refractory"] is not None and
                        pop["refractory"]["mode"] == "fixed"):
                    lines.append(
                        f"                p{p}_not_refractory[i] = "
                        f"u8::from(p{p}_tick >= p{p}_refractory_until[i]);")
                lines += ["    " + line for line in vector]
                for name in code["effects"]["writes"]:
                    if name == "not_refractory":
                        lines.append(
                            f"                p{p}_not_refractory[i] = "
                            f"u8::from({symbols[name]});")
                    else:
                        value = _storage_store(
                            pop["states"][positions[name]], symbols[name])
                        lines.append(
                            f"                unsafe {{ *p{p}_state_{positions[name]}."
                            f"get_unchecked_mut(i) = {value}; }}")
                lines += ["            }", "        }"]
                continue
            (declarations, parallel_scalar, parallel_vector, parallel_writes,
             parallel_threshold) = _v7_parallel_neuron(
                 model, p, code, positions, fuse_threshold=fuse_threshold)
            if parallel_scalar != scalar:
                raise ValueError("parallel neuron scalar lowering differs from serial lowering")
            lines += [f"            if parallel.is_parallel({pop['count']}, {work}) {{"]
            lines += ["                " + line for line in declarations]
            if fuse_threshold:
                lines += [f"                let p{p}_fired_lanes_ptr = p{p}_fired_lanes.as_mut_ptr() as usize;",
                          f"                parallel.for_each_ranked({pop['count']}, {work}, move |rank, start, end| {{",
                          f"                    let lane_fired = unsafe {{ &mut *((p{p}_fired_lanes_ptr as *mut Vec<usize>).add(rank)) }};",
                          "                    lane_fired.clear();"]
            else:
                lines += [f"                parallel.for_each({pop['count']}, {work}, move |start, end| {{"]
            lines += ["                    for i in start..end {"]
            if (pop["refractory"] is not None and
                    pop["refractory"]["mode"] == "fixed"):
                lines.append(
                    f"                        unsafe {{ *((p{p}_not_refractory_ptr as *mut u8).add(i)) = "
                    f"u8::from(p{p}_tick >= *((p{p}_refractory_until_ptr as *const usize).add(i))); }}")
            lines += ["            " + line for line in parallel_vector]
            lines += ["                        " + line for line in parallel_writes]
            lines.append("                    }")
            if fuse_threshold:
                threshold_scalar, threshold_vector, threshold_writes = parallel_threshold
                lines += ["                    {"]
                lines += ["            " + line for line in threshold_scalar]
                lines += ["                        for i in start..end {"]
                lines += ["                " + line for line in threshold_vector]
                lines += ["                            " + line for line in threshold_writes]
                lines += ["                        }", "                    }"]
            lines += ["                });"]
            if fuse_threshold:
                lines += [f"                p{p}_threshold_fused = true;",
                          f"                for lane in &mut p{p}_fired_lanes {{ p{p}_fired.append(lane); }}"]
            lines += ["            } else {",
                      f"                for i in 0..{pop['count']} {{"]
            if (pop["refractory"] is not None and
                    pop["refractory"]["mode"] == "fixed"):
                lines.append(f"                    p{p}_not_refractory[i] = u8::from(p{p}_tick >= p{p}_refractory_until[i]);")
            lines += ["        " + line for line in vector]
            for name in code["effects"]["writes"]:
                if name == "not_refractory":
                    lines.append(
                        f"                    p{p}_not_refractory[i] = "
                        f"u8::from({symbols[name]});")
                else:
                    value = _storage_store(
                        pop["states"][positions[name]], symbols[name])
                    lines.append(
                        f"                    unsafe {{ *p{p}_state_{positions[name]}.get_unchecked_mut(i) = {value}; }}")
            lines += ["                }", "            }", "        }"]
        lines += ["        phase_finish(&mut phase_neuron_state_seconds, phase_neuron_state_started);",
                  "        let phase_synapse_state_started = phase_start(phase_profile_enabled);"]
        # clock-driven synapse update
        for q, syn_def in enumerate(syn_defs):
            state_code = next((c for c in syn_def["code_objects"] if c["kind"] == "synapse_state_update"), None)
            if state_code is not None:
                lines += _v7_emit_synapse_state_update(model, q, state_code)
        # Synaptic CodeRunners use their own clocks and write disjoint edge state.
        for q, syn_def in enumerate(syn_defs):
            for regular in (c for c in syn_def["code_objects"]
                            if c["kind"] == "synapse_run_regularly" and c["when"] == "groups"):
                lines += _v7_emit_synapse_regular(model, q, regular)
        # Default summed updaters attached to a Subgroup have order 0 and sort
        # after the parent/synapse state updaters in Brian's groups slot.
        for q, syn_def in enumerate(syn_defs):
            prefix = f"s{q}_"
            source_pop = syn_def["source_population"]
            target_pop = syn_def["target_population"]
            for code in (item for item in syn_def["code_objects"]
                         if item["kind"] == "summed_variable" and item["order"] >= 0):
                target_is_pre = code["summed_target"] == "pre"
                summed_pop = source_pop if target_is_pre else target_pop
                summed_start = (syn_def["source_start"] if target_is_pre else
                                syn_def["target_start"])
                summed_count = (syn_def["source_count"] if target_is_pre else
                                syn_def["target_count"])
                state_positions = {
                    state["name"]: position for position, state in
                    enumerate(populations[summed_pop]["states"])}
                state_position = state_positions[code["summed_state"]]
                destination = "source_state" if target_is_pre else "target_state"
                scalar, vector, symbols = _v7_block(
                    model, code, summed_pop, "synapse", "edge", q)
                lines += [f"        if c{code['clock']}_active {{",
                          _v7_local_time(model, source_pop),
                          f"            p{summed_pop}_state_{state_position}[{summed_start}..{summed_start + summed_count}].fill(0.0);"]
                lines += ["    " + line for line in scalar]
                lines += [f"            for edge in 0..{prefix}edge_count {{",
                          f"                let source = unsafe {{ *{prefix}source_index.get_unchecked(edge) }} as usize;",
                          f"                let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                          f"                let source_state = source + {syn_def['source_start']};",
                          f"                let target_state = target + {syn_def['target_start']};"]
                lines += ["    " + line for line in vector]
                lines += [f"                p{summed_pop}_state_{state_position}[{destination}] += {symbols['_synaptic_var']};",
                          "            }", "        }"]
        # thresholds
        lines += ["        phase_finish(&mut phase_synapse_state_seconds, phase_synapse_state_started);",
                  "        let phase_threshold_started = phase_start(phase_profile_enabled);"]
        for p, pop in enumerate(populations):
            lines += [f"        if p{p}_active {{", _v7_local_time(model, p)]
            thresholds = [code for code in pop["code_objects"]
                          if code["kind"] == "threshold"]
            for threshold in thresholds:
                event = threshold["event_name"]
                event_position = pop["events"].index(event)
                event_vector = _v7_event_vector(p, pop, event)
                scalar, vector, symbols = _v7_block(model, threshold, p, "neuron")
                fused_guard = (f"!p{p}_threshold_fused"
                               if event == "spike" else "true")
                lines += [f"            if {fused_guard} {{"]
                lines += ["        " + line for line in scalar] + [f"                for i in 0..{pop['count']} {{"]
                lines += ["    " + line for line in vector]
                gate = symbols["_cond"]
                if event == "spike" and pop["refractory"] is not None:
                    gate += f" && p{p}_not_refractory[i] != 0"
                lines += [f"                    if {gate} {{",
                          f"                        {event_vector}.push(i);"]
                if event == "spike" and pop["refractory"] is not None:
                    lines += [f"                        p{p}_lastspike[i] = time;",
                              f"                        p{p}_not_refractory[i] = 0;"]
                    if pop["refractory"]["mode"] == "fixed":
                        lines.append(
                            f"                        p{p}_refractory_until[i] = p{p}_tick + p{p}_period_ticks;")
                lines += ["                    }", "                }", "            }"]
            spike_position = (pop.get("events", []).index("spike")
                              if "spike" in pop.get("events", []) else None)
            if thresholds and spike_position is not None:
                lines += [f"            if p{p}_tick >= p{p}_end_tick - {pop['monitor']['window_steps']} {{",
                          f"            for &i in &p{p}_fired {{",
                          f"                p{p}_counts[i] += 1;",
                          f"                p{p}_spikes.push((p{p}_tick,i));",
                          "            }", "            }"]
            elif inst["populations"][p].get("spike_generator") is not None:
                lines += [f"            while p{p}_generated_cursor < p{p}_generated_ticks.len() && p{p}_generated_ticks[p{p}_generated_cursor] < p{p}_tick {{ p{p}_generated_cursor += 1; }}",
                          f"            while p{p}_generated_cursor < p{p}_generated_ticks.len() && p{p}_generated_ticks[p{p}_generated_cursor] == p{p}_tick {{",
                          f"                let i = p{p}_generated_indices[p{p}_generated_cursor];",
                          f"                p{p}_fired.push(i);",
                          f"                if p{p}_tick >= p{p}_end_tick - {pop['monitor']['window_steps']} {{",
                          f"                    p{p}_counts[i] += 1;",
                          f"                    p{p}_spikes.push((p{p}_tick,i));",
                          "                }",
                          f"                p{p}_generated_cursor += 1;",
                          "            }"]
            if needs_event_dump:
                for event_position, _event in enumerate(pop.get("events", [])):
                    event_vector = _v7_event_vector(p, pop, _event)
                    lines.append(
                        f"            p{p}_event_history_{event_position}.extend("
                        f"{event_vector}.iter().map(|&i| (p{p}_tick,i)));")
            state_positions = {symbol["name"]: position
                               for position, symbol in enumerate(pop["states"])}
            parameter_positions = {symbol["name"]: position
                                   for position, symbol in enumerate(pop["parameters"])}
            linked_positions = {symbol["name"]: position
                                for position, symbol in enumerate(
                                    pop.get("linked_variables", []))}
            for monitor_position, monitor in enumerate(pop.get("event_monitors", [])):
                event_vector = _v7_event_vector(p, pop, monitor["event"])
                lines += [f"            for &i in &{event_vector} {{",
                          f"                p{p}_event_monitor_{monitor_position}_events.push((p{p}_tick,i));"]
                for column, name in enumerate(monitor["variables"]):
                    if name in state_positions:
                        value = f"p{p}_state_{state_positions[name]}[i]"
                    elif name in linked_positions:
                        value = _v7_link_value(model, p, linked_positions[name], "i")
                    else:
                        position = parameter_positions[name]
                        value = (f"p{p}_parameter_{position}"
                                 if pop["parameters"][position]["index_domain"] == "scalar"
                                 else f"p{p}_parameter_{position}[i]")
                    lines.append(
                        f"                p{p}_event_monitor_{monitor_position}_samples_{column}.push({value});")
                lines.append("            }")
            lines += [f"            p{p}_last_fired.clone_from(&p{p}_fired);", "        }"]
        lines += ["        phase_finish(&mut phase_threshold_seconds, phase_threshold_started);",
                  "        let phase_primary_events_started = phase_start(phase_profile_enabled);"]
        def emit_heterogeneous_pathway(q):
            syn_def = syn_defs[q]
            syn_inst = syn_insts[q]
            prefix = f"s{q}_"
            source_pop = syn_def["source_population"]
            target_pop = syn_def["target_population"]
            pathway = next(c for c in syn_def["code_objects"]
                           if c["kind"] == "synapses")
            scalar, vector, symbols = _v7_block(
                model, pathway, source_pop, "synapse", "edge", q)
            target_positions = {s["name"]: i for i, s in
                                enumerate(populations[target_pop]["states"])}
            source_positions = {s["name"]: i for i, s in
                                enumerate(populations[source_pop]["states"])}
            syn_positions = {s["name"]: i for i, s in enumerate(syn_def["states"])}
            full_source = (syn_def["source_start"] == 0 and
                           syn_def["source_count"] == populations[source_pop]["count"])
            source_events = _v7_event_vector(
                source_pop, populations[source_pop],
                syn_inst["pathways"][0]["event"])
            fired_name = source_events if full_source else f"{prefix}fired"
            fired_setup = [] if full_source else [
                f"            let {fired_name}: Vec<usize> = {source_events}.iter()"
                f".copied().filter(|&i| i >= {syn_def['source_start']} && "
                f"i < {syn_def['source_start'] + syn_def['source_count']})"
                f".map(|i| i - {syn_def['source_start']}).collect();"]
            procedural = not _explicit_topology(syn_inst)
            edge_loop = (
                f"                for edge in {prefix}offsets[source]..{prefix}offsets[source+1] {{"
                if procedural else
                f"                for &edge_index in &{prefix}edges[{prefix}offsets[source]..{prefix}offsets[source+1]] {{"
            )
            lines.extend([f"        if p{source_pop}_active {{", _v7_local_time(model, source_pop), *fired_setup,
                          "            let primary_enqueue_started = phase_start(phase_profile_enabled);",
                          f"            for &source in &{fired_name} {{",
                          edge_loop])
            if not procedural:
                lines.append("                    let edge = edge_index as usize;")
            queued_item = ("(((source as u64)<<32)|(edge as u64)) as usize"
                           if procedural else "edge")
            lines.extend([
                          f"                    let delay = unsafe {{ *{prefix}delay_ticks.get_unchecked(edge) }};",
                          f"                    let delivery = p{source_pop}_tick + delay;",
                          f"                    if delivery < p{source_pop}_end_tick {{ let slot=delivery%{prefix}queue.len(); {prefix}queue[slot].push({queued_item}); }}",
                          "                }", "            }",
                          f"            let slot = p{source_pop}_tick % {prefix}queue.len();",
                          f"            let mut active_edges = std::mem::take(&mut {prefix}queue[slot]);"])
            lines.extend(["            phase_finish(&mut phase_primary_enqueue_seconds, primary_enqueue_started);",
                          "            let primary_apply_started = phase_start(phase_profile_enabled);"])
            lines.extend("    " + line for line in scalar)
            lines.append("            for &queued in &active_edges {")
            if procedural:
                lines.extend(["                let source = queued >> 32;",
                              "                let edge = queued & (u32::MAX as usize);"])
            else:
                lines.extend(["                let edge = queued;",
                              f"                let source = unsafe {{ *{prefix}source_index.get_unchecked(edge) }} as usize;"])
            lines.extend([
                          f"                let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                          f"                let source_state = source + {syn_def['source_start']};",
                          f"                let target_state = target + {syn_def['target_start']};"])
            lines.extend("    " + line for line in vector)
            for name in pathway["effects"]["writes"]:
                if name in syn_def["post_state_aliases"]:
                    state = syn_def["post_state_aliases"][name]
                    value = _storage_store(
                        populations[target_pop]["states"][target_positions[state]],
                        symbols[name])
                    lines.append(
                        f"                unsafe {{ *p{target_pop}_state_{target_positions[state]}.get_unchecked_mut(target_state) = {value}; }}")
                elif name in syn_def["pre_state_aliases"]:
                    state = syn_def["pre_state_aliases"][name]
                    value = _storage_store(
                        populations[source_pop]["states"][source_positions[state]],
                        symbols[name])
                    lines.append(
                        f"                unsafe {{ *p{source_pop}_state_{source_positions[state]}.get_unchecked_mut(source_state) = {value}; }}")
                else:
                    value = _storage_store(
                        syn_def["states"][syn_positions[name]], symbols[name])
                    lines.append(
                        f"                unsafe {{ *{prefix}state_{syn_positions[name]}.get_unchecked_mut(edge) = {value}; }}")
            lines.extend([f"                {prefix}delivered += 1;", "            }",
                          "            phase_finish(&mut phase_primary_apply_seconds, primary_apply_started);",
                          "            active_edges.clear();",
                          f"            {prefix}queue[slot] = active_edges;", "        }"])

        def emit_parallel_heterogeneous_enqueues(batches):
            """Broadcast spikes once; each target owner fills only local queues."""
            lines.append(
                "        let primary_enqueue_started = phase_start(phase_profile_enabled);")
            batch_details = []
            for batch, (source_key, members) in enumerate(batches):
                source_pop, source_start, source_count, event = source_key
                full_source = (source_start == 0 and
                               source_count == populations[source_pop]["count"])
                source_events = _v7_event_vector(
                    source_pop, populations[source_pop], event)
                fired_name = (source_events if full_source else
                              f"h{batch}_fired")
                if not full_source:
                    lines.append(
                        f"        let {fired_name}: Vec<usize> = if p{source_pop}_active {{ "
                        f"{source_events}.iter().copied().filter(|&i| "
                        f"i >= {source_start} && i < {source_start + source_count})"
                        f".map(|i| i - {source_start}).collect() }} else {{ Vec::new() }};")
                enqueue_work = max(1, sum(
                    _synapse_edge_count(syn_insts[q]) for q in members) //
                    source_count)
                lines.extend([
                    f"        let h{batch}_fired_ptr = {fired_name}.as_ptr() as usize;",
                    f"        let h{batch}_fired_len = if p{source_pop}_active {{ {fired_name}.len() }} else {{ 0 }};",
                ])
                batch_details.append((batch, source_pop, members, enqueue_work))
                for q in members:
                    prefix = f"s{q}_"
                    lines.extend([
                        f"        let {prefix}enqueue_owner_csr_ptr = {prefix}target_owner.as_ref().unwrap() as *const TargetOwnerCsr as usize;",
                        f"        let {prefix}enqueue_delays_ptr = {prefix}delay_ticks.as_ptr() as usize;",
                        f"        let {prefix}enqueue_queue_ptr = {prefix}queue.as_mut_ptr() as usize;",
                    ])
            work_terms = [
                f"h{batch}_fired_len.saturating_mul({enqueue_work})"
                for batch, _source_pop, _members, enqueue_work in batch_details]
            lines.extend([
                "        let fused_enqueue_work = " +
                ".saturating_add(".join(work_terms) + ")" * (len(work_terms) - 1) + ";",
                "        parallel_event_enqueue |= fused_enqueue_work > 0 && parallel_threads > 1;",
                "        if fused_enqueue_work > 0 {",
                "            parallel.for_lanes(move |owner| {",
            ])
            for batch, source_pop, members, _enqueue_work in batch_details:
                for q in members:
                    syn_inst = syn_insts[q]
                    prefix = f"s{q}_"
                    procedural = not _explicit_topology(syn_inst)
                    lines.extend([
                        f"                let {prefix}owner_csr = unsafe {{ &*({prefix}enqueue_owner_csr_ptr as *const TargetOwnerCsr) }};",
                        f"                let {prefix}owner_offsets = &{prefix}owner_csr.offsets[owner];",
                        f"                let {prefix}owner_edges = &{prefix}owner_csr.edges[owner];",
                        f"                for fired_at in 0..h{batch}_fired_len {{",
                        f"                    let source = unsafe {{ *((h{batch}_fired_ptr as *const usize).add(fired_at)) }};",
                        f"                    for edge_at in {prefix}owner_offsets[source]..{prefix}owner_offsets[source+1] {{",
                    ])
                    lines.append(
                        f"                        let edge = unsafe {{ *{prefix}owner_edges.get_unchecked(edge_at) }} as usize;")
                    queued = ("(((source as u64)<<32)|(edge as u64)) as usize"
                              if procedural else "edge")
                    lines.extend([
                        f"                        let delivery = p{source_pop}_tick + unsafe {{ *(({prefix}enqueue_delays_ptr as *const usize).add(edge)) }};",
                        f"                        if delivery < p{source_pop}_end_tick {{",
                        f"                            let slot = delivery % {prefix}queue_slots;",
                        f"                            let queue_at = slot*parallel_threads+owner;",
                        f"                            unsafe {{ (&mut *(({prefix}enqueue_queue_ptr as *mut Vec<usize>).add(queue_at))).push({queued}); }}",
                        "                        }", "                    }",
                    ])
                    lines.append("                }")
            lines.extend([
                "            });", "        }",
                "        phase_finish(&mut phase_primary_enqueue_seconds, primary_enqueue_started);",
            ])

        def emit_parallel_heterogeneous_apply(label, source_pop, members, call):
            lines.extend([f"        if p{source_pop}_active {{", _v7_local_time(model, source_pop)])
            event_terms = []
            active_names = {}
            for q in members:
                prefix = f"s{q}_"
                lines.extend([
                    f"            let {prefix}slot = p{source_pop}_tick % {prefix}queue_slots;",
                    f"            let {prefix}active_start = {prefix}slot*parallel_threads;",
                    f"            let {prefix}active_end = {prefix}active_start+parallel_threads;",
                    f"            let {prefix}active_by_owner = &{prefix}queue[{prefix}active_start..{prefix}active_end];",
                    f"            let {prefix}active_events: usize = {prefix}active_by_owner.iter().map(Vec::len).sum();",
                    f"            {prefix}delivered += {prefix}active_events;"])
                event_terms.append(f"{prefix}active_events")
                active_names[q] = f"{prefix}active_by_owner"
            lines.extend(["            let primary_apply_started = phase_start(phase_profile_enabled);",
                          f"            let {label}_event_count = " +
                          usize_sum(event_terms) + ";",
                          f"            if {label}_event_count > 0 {{",
                          "                parallel_on_pre |= parallel.threads() > 1;",
                          "                " + call.format(
                              **{f"active_{q}": name
                                 for q, name in active_names.items()}),
                          "            }",
                          "            phase_finish(&mut phase_primary_apply_seconds, primary_apply_started);"])
            for q in members:
                prefix = f"s{q}_"
                lines.append(
                    f"            for active in &mut {prefix}queue[{prefix}active_start..{prefix}active_end] {{ active.clear(); }}")
            lines.append("        }")

        def emit_parallel_heterogeneous_route(route, members):
            emit_parallel_heterogeneous_apply(
                f"r{route}", route_keys[route][0], members,
                route_parallel_calls[route])

        # Consecutive projections with an identical source window/clock and
        # uniform delay share one routed spike queue. Projection execution order
        # stays unchanged, while source filtering and queue traffic happen once.
        heterogeneous_source_batches = {}
        for route, (route_key, members) in enumerate(
                zip(route_keys, route_members, strict=True)):
            if route_key[4] is None and route_parallel[route]:
                heterogeneous_source_batches.setdefault(route_key[:4], []).extend(members)
        if heterogeneous_source_batches:
            emit_parallel_heterogeneous_enqueues(
                list(heterogeneous_source_batches.items()))
        for route, (route_key, members) in enumerate(
                zip(route_keys, route_members, strict=True)):
            if route in fused_route_lookup:
                group = fused_route_lookup[route]
                if route != fused_heterogeneous_groups[group][0]:
                    continue
                emit_parallel_heterogeneous_apply(
                    f"g{group}", route_key[0], fused_route_members[group],
                    fused_route_calls[group])
                continue
            if route_key[4] is None:
                if route_parallel[route]:
                    emit_parallel_heterogeneous_route(route, members)
                else:
                    for q in members:
                        emit_heterogeneous_pathway(q)
                continue
            source_pop, source_start, source_count, event, delay = route_key
            full_source = (source_start == 0 and
                           source_count == populations[source_pop]["count"])
            source_events = _v7_event_vector(
                source_pop, populations[source_pop], event)
            fired_name = source_events if full_source else f"r{route}_fired"
            lines.extend([f"        if p{source_pop}_active {{", _v7_local_time(model, source_pop),
                          "            let primary_enqueue_started = phase_start(phase_profile_enabled);"])
            if not full_source:
                lines.append(
                    f"            let {fired_name}: Vec<usize> = {source_events}.iter()"
                    f".copied().filter(|&i| i >= {source_start} && "
                    f"i < {source_start + source_count})"
                    f".map(|i| i - {source_start}).collect();")
            if delay > 0:
                lines += [f"            let r{route}_delivery = p{source_pop}_tick + {delay};",
                          f"            let r{route}_queue_size = r{route}_queue.len();",
                          f"            if r{route}_delivery < p{source_pop}_end_tick {{",
                          f"                r{route}_queue[r{route}_delivery % r{route}_queue_size].extend_from_slice(&{fired_name});",
                          "            }",
                          f"            let r{route}_slot = p{source_pop}_tick % r{route}_queue_size;",
                          f"            let mut r{route}_active = std::mem::take(&mut r{route}_queue[r{route}_slot]);"]
                active_sources = f"r{route}_active.iter()"
                active_slice = f"&r{route}_active"
            else:
                active_sources = f"{fired_name}.iter()"
                active_slice = f"&{fired_name}"
            event_terms = []
            for q in members:
                prefix = f"s{q}_"
                count_name = f"r{route}_s{q}_events"
                lines += [f"            let {count_name}: usize = {active_sources}.map(|&source| "
                          f"{prefix}offsets[source+1]-{prefix}offsets[source]).sum();",
                          f"            {prefix}delivered += {count_name};"]
                event_terms.append(count_name)
            lines += ["            phase_finish(&mut phase_primary_enqueue_seconds, primary_enqueue_started);",
                      "            let primary_apply_started = phase_start(phase_profile_enabled);"]
            if route_parallel[route]:
                lines += [f"            let r{route}_event_count = " + usize_sum(event_terms) + ";",
                          f"            if parallel.events_parallel(r{route}_event_count) {{",
                          "                parallel_on_pre = true;",
                          "                " + route_parallel_calls[route].format(
                              active=active_slice),
                          "            } else {"]
            for q in members:
                syn_def = syn_defs[q]
                syn_inst = syn_insts[q]
                prefix = f"s{q}_"
                target_pop = syn_def["target_population"]
                pathway = next(c for c in syn_def["code_objects"]
                               if c["kind"] == "synapses")
                if (q, "synapses") in plastic_calls:
                    indent = "                " if route_parallel[route] else "            "
                    lines.append(indent + plastic_calls[q, "synapses"].format(
                        active=active_slice, offsets=f"&{prefix}offsets",
                        edges=f"&{prefix}edges"))
                    continue
                scalar, vector, symbols = _v7_block(
                    model, pathway, source_pop, "synapse", "edge", q)
                target_positions = {s["name"]: i for i, s in
                                    enumerate(populations[target_pop]["states"])}
                source_positions = {s["name"]: i for i, s in
                                    enumerate(populations[source_pop]["states"])}
                syn_positions = {s["name"]: i for i, s in enumerate(syn_def["states"])}
                extra = "    " if route_parallel[route] else ""
                lines += [extra + "    " + line for line in scalar]
                edge_loop = (
                    f"                for edge in {prefix}offsets[source]..{prefix}offsets[source+1] {{"
                    if not _explicit_topology(syn_inst) else
                    f"                for &edge_index in &{prefix}edges[{prefix}offsets[source]..{prefix}offsets[source+1]] {{"
                )
                lines += [extra + f"            for &source in {active_sources} {{",
                          extra + edge_loop]
                if _explicit_topology(syn_inst):
                    lines.append(extra + "                    let edge = edge_index as usize;")
                lines += [
                          extra + f"                    let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                          extra + f"                    let source_state = source + {syn_def['source_start']};",
                          extra + f"                    let target_state = target + {syn_def['target_start']};"]
                lines += [extra + "        " + line for line in vector]
                for name in pathway["effects"]["writes"]:
                    if name in syn_def["post_state_aliases"]:
                        state = syn_def["post_state_aliases"][name]
                        value = _storage_store(
                            populations[target_pop]["states"][target_positions[state]],
                            symbols[name])
                        lines.append(
                            extra + f"                    unsafe {{ *p{target_pop}_state_{target_positions[state]}.get_unchecked_mut(target_state) = {value}; }}")
                    elif name in syn_def["pre_state_aliases"]:
                        state = syn_def["pre_state_aliases"][name]
                        value = _storage_store(
                            populations[source_pop]["states"][source_positions[state]],
                            symbols[name])
                        lines.append(
                            extra + f"                    unsafe {{ *p{source_pop}_state_{source_positions[state]}.get_unchecked_mut(source_state) = {value}; }}")
                    else:
                        value = _storage_store(
                            syn_def["states"][syn_positions[name]], symbols[name])
                        lines.append(
                            extra + f"                    unsafe {{ *{prefix}state_{syn_positions[name]}.get_unchecked_mut(edge) = {value}; }}")
                lines += [extra + "                }", extra + "            }"]
            if route_parallel[route]:
                lines.append("            }")
            lines += ["            phase_finish(&mut phase_primary_apply_seconds, primary_apply_started);"]
            if delay > 0:
                lines += [f"            r{route}_active.clear();",
                          f"            r{route}_queue[r{route}_slot] = r{route}_active;"]
            lines.append("        }")

        # Additional pre pathways share the Synapses topology/state arrays but
        # keep their own clock-quantized delay queue and Brian name order.
        lines += ["        phase_finish(&mut phase_primary_events_seconds, phase_primary_events_started);",
                  "        let phase_additional_pre_started = phase_start(phase_profile_enabled);"]
        for q, (syn_def, syn_inst) in enumerate(zip(syn_defs, syn_insts, strict=True)):
            prefix = f"s{q}_"
            source_pop = syn_def["source_population"]
            target_pop = syn_def["target_population"]
            full_source = (syn_def["source_start"] == 0 and
                           syn_def["source_count"] == populations[source_pop]["count"])
            for r, pathway_instance in enumerate(syn_inst["pathways"]):
                if r == 0 or pathway_instance["kind"] != "pre":
                    continue
                pathway_prefix = path_prefix(q, r)
                code = next(item for item in syn_def["code_objects"]
                            if item.get("pathway_name") == pathway_instance["name"])
                scalar, vector, symbols = _v7_block(
                    model, code, source_pop, "synapse", "edge", q)
                target_positions = {state["name"]: position for position, state in
                                    enumerate(populations[target_pop]["states"])}
                syn_positions = {state["name"]: position for position, state in
                                 enumerate(syn_def["states"])}
                source_events = _v7_event_vector(
                    source_pop, populations[source_pop], pathway_instance["event"])
                fired_name = source_events if full_source else f"{pathway_prefix}fired"
                lines += [f"        if p{source_pop}_active {{", _v7_local_time(model, source_pop)]
                if not full_source:
                    lines.append(
                        f"            let {fired_name}: Vec<usize> = {source_events}.iter()"
                        f".copied().filter(|&i| i >= {syn_def['source_start']} && "
                        f"i < {syn_def['source_start'] + syn_def['source_count']})"
                        f".map(|i| i - {syn_def['source_start']}).collect();")
                delays = pathway_instance["delay_ticks"]
                uniform = (delays[0] if delays and
                           all(value == delays[0] for value in delays) else None)
                if uniform is not None:
                    if uniform > 0 or pathway_instance["pending"]:
                        lines += [f"            let {pathway_prefix}delivery = p{source_pop}_tick + {uniform};",
                                  f"            if {pathway_prefix}delivery < p{source_pop}_end_tick {{ let size={pathway_prefix}queue.len(); {pathway_prefix}queue[{pathway_prefix}delivery%size].extend_from_slice(&{fired_name}); }}",
                                  f"            let {pathway_prefix}slot = p{source_pop}_tick % {pathway_prefix}queue.len();",
                                  f"            let mut {pathway_prefix}active = std::mem::take(&mut {pathway_prefix}queue[{pathway_prefix}slot]);"]
                        active_sources = f"{pathway_prefix}active.iter()"
                    else:
                        active_sources = f"{fired_name}.iter()"
                    lines += ["    " + line for line in scalar]
                    edge_loop = (
                        f"                for edge in {prefix}offsets[source]..{prefix}offsets[source+1] {{"
                        if not _explicit_topology(syn_inst) else
                        f"                for &edge_index in &{prefix}edges[{prefix}offsets[source]..{prefix}offsets[source+1]] {{"
                    )
                    lines += [f"            for &source in {active_sources} {{",
                              edge_loop]
                    if _explicit_topology(syn_inst):
                        lines.append("                    let edge = edge_index as usize;")
                    lines += [
                              f"                    let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                              f"                    let source_state = source + {syn_def['source_start']};",
                              f"                    let target_state = target + {syn_def['target_start']};"]
                else:
                    lines += [f"            for &source in &{fired_name} {{",
                              f"                for &edge_index in &{prefix}edges[{prefix}offsets[source]..{prefix}offsets[source+1]] {{",
                              "                    let edge = edge_index as usize;",
                              f"                    let delivery = p{source_pop}_tick + unsafe {{ *{pathway_prefix}delay_ticks.get_unchecked(edge) }};",
                              f"                    if delivery < p{source_pop}_end_tick {{ let slot=delivery%{pathway_prefix}queue.len(); {pathway_prefix}queue[slot].push(edge); }}",
                              "                }", "            }",
                              f"            let {pathway_prefix}slot = p{source_pop}_tick % {pathway_prefix}queue.len();",
                              f"            let mut {pathway_prefix}active = std::mem::take(&mut {pathway_prefix}queue[{pathway_prefix}slot]);"]
                    lines += ["    " + line for line in scalar]
                    lines += [f"            for &edge in &{pathway_prefix}active {{",
                              f"                let source = unsafe {{ *{prefix}source_index.get_unchecked(edge) }} as usize;",
                              f"                let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                              f"                let source_state = source + {syn_def['source_start']};",
                              f"                let target_state = target + {syn_def['target_start']};"]
                lines += ["    " + line for line in vector]
                for name in code["effects"]["writes"]:
                    if name in syn_def["post_state_aliases"]:
                        state = syn_def["post_state_aliases"][name]
                        value = _storage_store(
                            populations[target_pop]["states"][target_positions[state]],
                            symbols[name])
                        lines.append(
                            f"                    p{target_pop}_state_{target_positions[state]}[target_state] = {value};")
                    else:
                        value = _storage_store(
                            syn_def["states"][syn_positions[name]], symbols[name])
                        lines.append(
                            f"                    {prefix}state_{syn_positions[name]}[edge] = {value};")
                if uniform is not None:
                    lines += [f"                    {prefix}delivered += 1;",
                              "                }", "            }"]
                else:
                    lines += [f"                {prefix}delivered += 1;", "            }",
                              f"            {pathway_prefix}active.clear();",
                              f"            {pathway_prefix}queue[{pathway_prefix}slot] = {pathway_prefix}active;"]
                if uniform is not None and (uniform > 0 or pathway_instance["pending"]):
                    lines += [f"            {pathway_prefix}active.clear();",
                              f"            {pathway_prefix}queue[{pathway_prefix}slot] = {pathway_prefix}active;"]
                lines.append("        }")

        # PoissonInput uses Brian's default synapses/order=0 schedule, after the
        # on_pre pathways above (order=-1) and before resets.
        lines += ["        phase_finish(&mut phase_additional_pre_seconds, phase_additional_pre_started);",
                  "        let phase_poisson_input_started = phase_start(phase_profile_enabled);"]
        for p, pop in enumerate(populations):
            positions = {s["name"]: i for i, s in enumerate(pop["states"])}
            for code in (item for item in pop["code_objects"]
                         if item["kind"] == "poisson_input"):
                work = code_work(code)
                capable = parallel_phase_capable(pop["count"], work)
                scalar, vector, symbols = _v7_block(
                    model, code, p, "neuron", normal_cache="normal_cache")
                lines += [f"        if p{p}_active {{", _v7_local_time(model, p),
                          "            let mut normal_cache = NormalCache::default();"]
                lines += ["    " + line for line in scalar]
                zero_gate = _poisson_zero_gate(pop, code, symbols, p)
                if zero_gate is not None:
                    gate, probability, exact, position = zero_gate
                    lines += [
                        f"            if {gate} == 0.0 && {exact} {{",
                        f'                assert!({probability}.is_finite() && (0.0..=1.0).contains(&{probability}), "invalid binomial probability");',
                        f"                for value in p{p}_state_{position}.iter_mut() {{ *value += {gate}; }}",
                        "            } else {",
                    ]
                if capable:
                    (declarations, parallel_scalar, parallel_vector,
                     parallel_writes, _threshold) = _v7_parallel_neuron(
                         model, p, code, positions, normal_cache="normal_cache")
                    if parallel_scalar != scalar:
                        raise ValueError(
                            "parallel PoissonInput scalar lowering differs from serial lowering")
                    lines += [f"            if parallel.is_parallel({pop['count']}, {work}) {{",
                              "                parallel_poisson_input = true;"]
                    lines += ["                " + line for line in declarations]
                    lines += [f"                parallel.for_each({pop['count']}, {work}, move |start, end| {{",
                              "                    let mut normal_cache = NormalCache::default();",
                              "                    for i in start..end {"]
                    lines += ["            " + line for line in parallel_vector]
                    lines += ["                        " + line for line in parallel_writes]
                    lines += ["                    }", "                });",
                              "            } else {"]
                lines += [f"                for i in 0..{pop['count']} {{"]
                lines += ["        " + line for line in vector]
                for name in code["effects"]["writes"]:
                    value = _storage_store(
                        pop["states"][positions[name]], symbols[name])
                    lines.append(
                        f"                    unsafe {{ *p{p}_state_{positions[name]}.get_unchecked_mut(i) = {value}; }}")
                lines.append("                }")
                if capable:
                    lines.append("            }")
                if zero_gate is not None:
                    lines.append("            }")
                lines.append("        }")

        # Post pathways route target spikes through target CSR. Each named pathway
        # owns its delay queue, matching Brian's independent pathway clocks.
        lines += ["        phase_finish(&mut phase_poisson_input_seconds, phase_poisson_input_started);",
                  "        let phase_post_started = phase_start(phase_profile_enabled);"]
        for q, (syn_def, syn_inst) in enumerate(
                zip(syn_defs, syn_insts, strict=True)):
            prefix = f"s{q}_"
            source_pop = syn_def["source_population"]
            target_pop = syn_def["target_population"]
            syn_positions = {s["name"]: i for i, s in enumerate(syn_def["states"])}
            target_positions = {
                state["name"]: position
                for position, state in enumerate(populations[target_pop]["states"])}

            def post_pathway_write(name, indent):
                if name in syn_def["post_state_aliases"]:
                    state_name = syn_def["post_state_aliases"][name]
                    position = target_positions[state_name]
                    symbol = populations[target_pop]["states"][position]
                    value = _storage_store(symbol, symbols[name])
                    return (f"{indent}unsafe {{ *p{target_pop}_state_{position}."
                            f"get_unchecked_mut(target_state) = {value}; }}")
                position = syn_positions[name]
                symbol = syn_def["states"][position]
                value = _storage_store(symbol, symbols[name])
                return (f"{indent}unsafe {{ *{prefix}state_{position}."
                        f"get_unchecked_mut(edge) = {value}; }}")
            for r, pathway_instance in enumerate(syn_inst["pathways"]):
                if pathway_instance["kind"] != "post":
                    continue
                pathway_prefix = path_prefix(q, r)
                code = next(item for item in syn_def["code_objects"]
                            if item.get("pathway_name") == pathway_instance["name"])
                scalar, vector, symbols = _v7_block(
                    model, code, target_pop, "synapse", "edge", q)
                fired_name = f"{pathway_prefix}fired"
                target_events = _v7_event_vector(
                    target_pop, populations[target_pop], pathway_instance["event"])
                lines += [f"        if p{target_pop}_active {{", _v7_local_time(model, target_pop),
                          f"            let {fired_name}: Vec<usize> = {target_events}.iter()",
                          f"                .copied().filter(|&i| i >= {syn_def['target_start']} && i < {syn_def['target_start'] + syn_def['target_count']})",
                          f"                .map(|i| i - {syn_def['target_start']}).collect();"]
                delays = pathway_instance["delay_ticks"]
                uniform = (delays[0] if delays and
                           all(value == delays[0] for value in delays) else None)
                if uniform is not None:
                    queued = uniform > 0 or bool(pathway_instance["pending"])
                    if queued:
                        lines += [f"            let {pathway_prefix}delivery = p{target_pop}_tick + {uniform};",
                                  f"            if {pathway_prefix}delivery < p{target_pop}_end_tick {{ let size={pathway_prefix}queue.len(); {pathway_prefix}queue[{pathway_prefix}delivery%size].extend_from_slice(&{fired_name}); }}",
                                  f"            let {pathway_prefix}slot = p{target_pop}_tick % {pathway_prefix}queue.len();",
                                  f"            let mut {pathway_prefix}active = std::mem::take(&mut {pathway_prefix}queue[{pathway_prefix}slot]);"]
                        active_targets = f"&{pathway_prefix}active"
                    else:
                        active_targets = f"&{fired_name}"
                    if (q, "synapses_post") in plastic_calls:
                        lines += [f"            {pathway_prefix}edge_scratch.clear();",
                                  f"            for &target in {active_targets} {{",
                                  f"                {pathway_prefix}edge_scratch.extend_from_slice(&{prefix}target_edges[{prefix}target_offsets[target]..{prefix}target_offsets[target+1]]);",
                                  "            }",
                                  f"            {prefix}post_delivered += {pathway_prefix}edge_scratch.len();",
                                  "            " + plastic_calls[q, "synapses_post"].format(
                                      active=f"&{pathway_prefix}edge_scratch",
                                      offsets=f"&{prefix}target_offsets",
                                      edges=f"&{prefix}target_edges")]
                        if queued:
                            lines += [f"            {pathway_prefix}active.clear();",
                                      f"            {pathway_prefix}queue[{pathway_prefix}slot] = {pathway_prefix}active;"]
                        lines.append("        }")
                        continue
                    lines += ["    " + line for line in scalar]
                    lines += [f"            for &target in {active_targets} {{",
                              f"                let target_state = target + {syn_def['target_start']};",
                              f"                for &edge_index in &{prefix}target_edges[{prefix}target_offsets[target]..{prefix}target_offsets[target+1]] {{",
                              "                    let edge = edge_index as usize;",
                              f"                    let source = unsafe {{ *{prefix}source_index.get_unchecked(edge) }} as usize;",
                              f"                    let source_state = source + {syn_def['source_start']};"]
                    lines += ["        " + line for line in vector]
                    for name in code["effects"]["writes"]:
                        lines.append(post_pathway_write(name, "                    "))
                    lines += [f"                    {prefix}post_delivered += 1;",
                              "                }", "            }"]
                    if queued:
                        lines += [f"            {pathway_prefix}active.clear();",
                                  f"            {pathway_prefix}queue[{pathway_prefix}slot] = {pathway_prefix}active;"]
                else:
                    lines += [f"            for &target in &{fired_name} {{",
                              f"                for &edge_index in &{prefix}target_edges[{prefix}target_offsets[target]..{prefix}target_offsets[target+1]] {{",
                              "                    let edge = edge_index as usize;",
                              f"                    let delivery = p{target_pop}_tick + unsafe {{ *{pathway_prefix}delay_ticks.get_unchecked(edge) }};",
                              f"                    if delivery < p{target_pop}_end_tick {{ let slot=delivery%{pathway_prefix}queue.len(); {pathway_prefix}queue[slot].push(edge); }}",
                              "                }", "            }",
                              f"            let {pathway_prefix}slot = p{target_pop}_tick % {pathway_prefix}queue.len();",
                              f"            let mut {pathway_prefix}active = std::mem::take(&mut {pathway_prefix}queue[{pathway_prefix}slot]);"]
                    lines += ["    " + line for line in scalar]
                    lines += [f"            for &edge in &{pathway_prefix}active {{",
                              f"                let source = unsafe {{ *{prefix}source_index.get_unchecked(edge) }} as usize;",
                              f"                let target = unsafe {{ *{prefix}target_index.get_unchecked(edge) }} as usize;",
                              f"                let source_state = source + {syn_def['source_start']};",
                              f"                let target_state = target + {syn_def['target_start']};"]
                    lines += ["    " + line for line in vector]
                    for name in code["effects"]["writes"]:
                        lines.append(post_pathway_write(name, "                "))
                    lines += [f"                {prefix}post_delivered += 1;", "            }",
                              f"            {pathway_prefix}active.clear();",
                              f"            {pathway_prefix}queue[{pathway_prefix}slot] = {pathway_prefix}active;"]
                lines.append("        }")

        # resets and clock increments
        lines += ["        phase_finish(&mut phase_post_seconds, phase_post_started);",
                  "        let phase_reset_started = phase_start(phase_profile_enabled);"]
        for p, pop in enumerate(populations):
            lines += [f"        if p{p}_active {{", _v7_local_time(model, p)]
            for reset in (code for code in pop["code_objects"]
                          if code["kind"] == "reset"):
                scalar, vector, symbols = _v7_block(model, reset, p, "neuron")
                positions = {s["name"]: i for i, s in enumerate(pop["states"])}
                event_vector = _v7_event_vector(p, pop, reset["event_name"])
                lines += ["    " + line for line in scalar] + [
                    f"            for &i in &{event_vector} {{"]
                lines += ["    " + line for line in vector]
                for name in reset["effects"]["writes"]:
                    value = _storage_store(
                        pop["states"][positions[name]], symbols[name])
                    lines.append(
                        f"                unsafe {{ *p{p}_state_{positions[name]}.get_unchecked_mut(i) = {value}; }}")
                lines.append("            }")
            lines.append("        }")
            for regular in sorted(
                    (code for code in pop["code_objects"]
                     if code["kind"] == "run_regularly"),
                    key=lambda code: (code["order"], code["name"])):
                if regular["when"] != "end":
                    raise ValueError(
                        "AOT run_regularly requires when='end'")
                scalar, vector, symbols = _v7_block(
                    model, regular, p, "neuron", clock=regular["clock"],
                    overrides={"dt": f"p{p}_dt"})
                positions = {s["name"]: i for i, s in enumerate(pop["states"])}
                lines.append(f"        if c{regular['clock']}_active {{")
                lines.append(_v7_local_time(model, p))
                lines += ["    " + line for line in scalar]
                lines += [f"            for i in 0..{pop['count']} {{"]
                lines += ["    " + line for line in vector]
                for name in regular["effects"]["writes"]:
                    value = _storage_store(
                        pop["states"][positions[name]], symbols[name])
                    lines.append(
                        f"                unsafe {{ *p{p}_state_{positions[name]}.get_unchecked_mut(i) = {value}; }}")
                lines += ["            }", "        }"]
            lines.append(f"        if p{p}_active {{ p{p}_tick += 1; }}")
        for q, syn_def in enumerate(syn_defs):
            for regular in (c for c in syn_def["code_objects"]
                            if c["kind"] == "synapse_run_regularly" and c["when"] == "end"):
                lines += _v7_emit_synapse_regular(model, q, regular)
        for clock in range(len(d["clocks"])):
            lines.append(
                f"        if c{clock}_active {{ c{clock}_tick += 1; }}")
        lines += ["        phase_finish(&mut phase_reset_seconds, phase_reset_started);"]
    parallel_checks = [
        f"parallel.is_parallel({populations[p]['count']}, {work})"
        for p, work in population_parallel_work.items()
    ]
    lines += ["    }"]
    if distributed_plan is not None:
        from .mpi_codegen import collect_results, simulation_timing
        lines += simulation_timing()
        lines += collect_results(model, distributed_plan)
    lines += ["    let simulation_and_recording_seconds = started.elapsed().as_secs_f64();",
              "    let simulation_threads = parallel.threads();",
              "    let thread_affinity = parallel.affinity_enabled();",
              "    let thread_cpus = parallel.affinity_json();",
              "    let parallel_state_update = " + (" || ".join(parallel_checks) or "false") + ";",
              "    drop(parallel);",
              "    let output_started = Instant::now();", "    fs::create_dir_all(output)?;"]
    lines += _v7_emit_dump(model)
    lines += _v7_emit_event_dump(model, needs_event_dump=needs_event_dump)
    lines += ["    let dump_write_seconds = output_started.elapsed().as_secs_f64();"]
    lines += _v7_emit_metadata(
        model, route_count=len(route_members),
        parallel_route_count=sum(route_parallel),
        fused_route_count=len(fused_heterogeneous_groups),
        final_only_summed_count=len(final_only_summed),
        distributed=distributed_plan is not None)
    lines += [f"    println!(\"completed {len(populations)} populations for {total_n} neurons (native AOT)\");",
              "    Ok(())", "}"]
    return _join_source(lines)


def _v7_local_time(model, population):
    """Brian evaluates t using the owner's clock, not the scheduler's minimum."""
    clock = model["definition"]["populations"][population]["clock"]
    return f"            let time = c{clock}_tick as f64 * c{clock}_dt;"


def _v7_parallel_synapse_regular(model, q, code, *, state_update=False):
    synapse = model["definition"]["synapses"][q]
    populations = model["definition"]["populations"]
    declarations, overrides, pointers = [], {}, {}
    for position, symbol in enumerate(synapse["states"]):
        pointer = f"s{q}_regular_state_{position}_ptr"
        pointers[symbol["name"]] = pointer
        declarations.append(f"let {pointer} = s{q}_state_{position}.as_mut_ptr() as usize;")
        overrides[symbol["name"]] = _storage_load(symbol,
            f"unsafe {{ *(({pointer} as *const {_rust_dtype(symbol)}).add(edge)) }}")
    for position, symbol in enumerate(synapse["parameters"]):
        if symbol["index_domain"] != "scalar":
            pointer = f"s{q}_regular_parameter_{position}_ptr"
            declarations.append(f"let {pointer} = s{q}_parameter_{position}.as_ptr() as usize;")
            overrides[symbol["name"]] = _storage_load(symbol,
                f"unsafe {{ *(({pointer} as *const {_rust_dtype(symbol)}).add(edge)) }}")
    for side, population, index in [
            ("pre", synapse["source_population"], "source_state"),
            ("post", synapse["target_population"], "target_state")]:
        states = populations[population]["states"]
        for alias, name in synapse[f"{side}_state_aliases"].items():
            position = next(i for i, symbol in enumerate(states) if symbol["name"] == name)
            symbol = states[position]
            pointer = f"s{q}_regular_{side}_{position}_ptr"
            declarations.append(f"let {pointer} = p{population}_state_{position}.as_ptr() as usize;")
            overrides[alias] = _storage_load(symbol,
                f"unsafe {{ *(({pointer} as *const {_rust_dtype(symbol)}).add({index})) }}")
    target = synapse["target_population"]
    if populations[target]["refractory"] is not None:
        pointer = f"s{q}_regular_refractory_ptr"
        declarations.append(f"let {pointer} = p{target}_not_refractory.as_ptr() as usize;")
        overrides["not_refractory_post"] = (
            f"unsafe {{ *(({pointer} as *const u8).add(target_state)) }} != 0")
    if not state_update:
        overrides["dt"] = f"p{synapse['source_population']}_dt"
    scalar, vector, symbols = _v7_block(model, code,
        synapse["source_population"], "synapse", "edge", q,
        clock=None if state_update else code["clock"], overrides=overrides)
    writes = []
    for symbol in synapse["states"]:
        if symbol["name"] in code["effects"]["writes"]:
            value = _storage_store(symbol, symbols[symbol["name"]])
            writes.append(f"unsafe {{ *(({pointers[symbol['name']]} as *mut "
                          f"{_rust_dtype(symbol)}).add(edge)) = {value}; }}")
    return declarations, scalar, vector, writes


def _v7_emit_synapse_state_update(model, q, code):
    """Update independent clock-driven edges; preserve the serial lowering."""
    synapse = model["definition"]["synapses"][q]
    positions = {symbol["name"]: i for i, symbol in enumerate(synapse["states"])}
    source_pop = synapse["source_population"]
    scalar, vector, symbols = _v7_block(
        model, code, source_pop, "synapse", "edge", q)
    lines = [f"        if p{source_pop}_active {{",
             _v7_local_time(model, source_pop),
             *["    " + line for line in scalar],
             f"            for edge in 0..s{q}_edge_count {{",
             f"                let source = unsafe {{ *s{q}_source_index.get_unchecked(edge) }} as usize;",
             f"                let target = unsafe {{ *s{q}_target_index.get_unchecked(edge) }} as usize;",
             f"                let source_state = source + {synapse['source_start']};",
             f"                let target_state = target + {synapse['target_start']};",
             *["    " + line for line in vector]]
    for name in code["effects"]["writes"]:
        value = _storage_store(synapse["states"][positions[name]], symbols[name])
        lines.append(f"                s{q}_state_{positions[name]}[edge] = {value};")
    lines += ["            }", "        }"]
    if not _v7_regular_parallel_capable(model, q, code):
        return lines
    declarations, parallel_scalar, parallel_vector, writes = (
        _v7_parallel_synapse_regular(model, q, code, state_update=True))
    if parallel_scalar != scalar:
        raise ValueError("parallel synapse state scalar lowering differs from serial lowering")
    work = code_work(code)
    parallel_lines = [f"        if p{source_pop}_active {{",
        _v7_local_time(model, source_pop),
        *["    " + line for line in scalar],
        f"            if parallel.is_parallel(s{q}_edge_count, {work}) {{",
        "                parallel_synapse_state = true;",
        *["                " + line for line in declarations],
        f"                let source_indices = &s{q}_source_index;",
        f"                let target_indices = &s{q}_target_index;",
        f"                parallel.for_each(s{q}_edge_count, {work}, move |start, end| {{",
        "                    for edge in start..end {",
        "                        let source = unsafe { *source_indices.get_unchecked(edge) } as usize;",
        "                        let target = unsafe { *target_indices.get_unchecked(edge) } as usize;",
        f"                        let source_state = source + {synapse['source_start']};",
        f"                        let target_state = target + {synapse['target_start']};",
        *["                " + line for line in parallel_vector],
        *["                        " + line for line in writes],
        "                    }", "                });", "            } else {"]
    parallel_lines += ["    " + line for line in lines[2 + len(scalar):-1]]
    return parallel_lines + ["            }", "        }"]


def _v7_emit_synapse_regular(model, q, code):
    synapse = model["definition"]["synapses"][q]
    instance = model["instance"]["synapses"][q]
    positions = {s["name"]: i for i, s in enumerate(synapse["states"])}
    scalar, vector, symbols = _v7_block(
        model, code, synapse["source_population"], "synapse", "edge", q,
        clock=code["clock"], overrides={"dt": f"p{synapse['source_population']}_dt"})
    lines = [f"        if c{code['clock']}_active {{",
             _v7_local_time(model, synapse["source_population"])]
    lines += ["    " + line for line in scalar]
    if _explicit_topology(instance):
        lines += [f"            for edge in 0..s{q}_edge_count {{",
                  f"                let source = s{q}_source_index[edge] as usize;"]
        closing = ["            }"]
    else:
        lines += [f"            for source in 0..{synapse['source_count']} {{",
                  f"                for edge in s{q}_offsets[source]..s{q}_offsets[source+1] {{"]
        closing = ["                }", "            }"]
    lines += [f"                let target = s{q}_target_index[edge] as usize;",
              f"                let source_state = source + {synapse['source_start']};",
              f"                let target_state = target + {synapse['target_start']};"]
    lines += ["    " + line for line in vector]
    for name in code["effects"]["writes"]:
        value = _storage_store(synapse["states"][positions[name]], symbols[name])
        lines.append(f"                s{q}_state_{positions[name]}[edge] = {value};")
    lines += closing + ["        }"]
    if not _v7_regular_parallel_capable(model, q, code):
        return lines
    declarations, parallel_scalar, parallel_vector, writes = (
        _v7_parallel_synapse_regular(model, q, code))
    if parallel_scalar != scalar:
        raise ValueError("parallel synapse regular scalar lowering differs from serial lowering")
    work = code_work(code)
    parallel_lines = [f"        if c{code['clock']}_active {{",
        _v7_local_time(model, synapse["source_population"]),
        *["    " + line for line in scalar],
        f"            if parallel.is_parallel(s{q}_edge_count, {work}) {{",
        "                parallel_synapse_regular = true;",
        *["                " + line for line in declarations],
        f"                let source_indices = &s{q}_source_index;",
        f"                let target_indices = &s{q}_target_index;",
        f"                parallel.for_each(s{q}_edge_count, {work}, move |start, end| {{",
        "                    for edge in start..end {",
        "                        let source = unsafe { *source_indices.get_unchecked(edge) } as usize;",
        "                        let target = unsafe { *target_indices.get_unchecked(edge) } as usize;",
        f"                        let source_state = source + {synapse['source_start']};",
        f"                        let target_state = target + {synapse['target_start']};",
        *["                " + line for line in parallel_vector],
        *["                        " + line for line in writes],
        "                    }", "                });", "            } else {"]
    # Reuse the serial loop without duplicating clock/scalar evaluation.
    parallel_lines += ["    " + line for line in lines[2 + len(scalar):-1]]
    return parallel_lines + ["            }", "        }"]


def _v7_emit_dump(model):
    d, inst = model["definition"], model["instance"]
    pops = d["populations"]
    terms = ["40usize", "24"]
    for q, (syn_def, syn_inst) in enumerate(
            zip(d["synapses"], inst["synapses"], strict=True)):
        state_bytes = sum(_dtype_size(symbol) for symbol in syn_def["states"])
        terms += ["24", f"{state_bytes}*{_synapse_edge_count(syn_inst)}"]
        for monitor_position, monitor in enumerate(
                syn_def.get("state_monitors", [])):
            terms.extend(
                f"s{q}_monitor_{monitor_position}_samples_{column}.len()*"
                f"{_dtype_size(source)}"
                for column, source in enumerate(monitor["sources"]))
    for p, pop in enumerate(pops):
        sample_terms = [
            f"p{p}_samples_{column}.len()*{_dtype_size(symbol)}"
            for column, name in enumerate(pop["monitor"]["variables"])
            for symbol in (pop["states"] + pop["parameters"] +
                           pop.get("linked_variables", []))
            if symbol["name"] == name
        ]
        terms += ["64", *sample_terms, f"p{p}_spikes.len()*16",
                  f"p{p}_counts.len()*8", f"p{p}_last_fired.len()*8",
                  f"{sum(_dtype_size(symbol) for symbol in pop['states'])}*{pop['count']}"]
        if pop["refractory"] is not None:
            terms += [f"p{p}_lastspike.len()*8", f"p{p}_not_refractory.len()"]
    lines = ["    let dump_bytes = " + usize_sum(terms) + ";",
             f"    let mut dump = dump_start(output, dump_bytes, {len(pops)}, {inst['neuron_count']})?;"]
    for p, pop in enumerate(pops):
        lines += [f"    dump_u64(&mut dump, {pop['count']})?;",
                  f"    dump_u64(&mut dump, {pop['steps']})?;",
                  f"    dump_u64(&mut dump, {len(pop['monitor']['record'])})?;",
                  f"    dump_u64(&mut dump, {len(pop['monitor']['variables'])})?;",
                  f"    dump_u64(&mut dump, {len(pop['states'])})?;",
                  f"    dump_u64(&mut dump, p{p}_spikes.len())?;",
                  f"    dump_u64(&mut dump, p{p}_last_fired.len())?;",
                  f"    dump_u64(&mut dump, {int(pop['refractory'] is not None)})?;",
                  f"    dump_spikes_usize(&mut dump, &p{p}_spikes)?;",
                  f"    dump_indices(&mut dump, &p{p}_counts)?;",
                  f"    dump_indices(&mut dump, &p{p}_last_fired)?;"]
        public_symbols = {
            symbol["name"]: symbol
            for symbol in (pop["states"] + pop["parameters"] +
                           pop.get("linked_variables", []))
        }
        sample_dumps = [
            f"    dump_{_dump_dtype(public_symbols[name])}(&mut dump, "
            f"&p{p}_samples_{column})?;"
            for column, name in enumerate(pop["monitor"]["variables"])
        ]
        # Samples precede spike/event arrays in the v3 population payload.
        lines[-3:-3] = sample_dumps
        for position, symbol in enumerate(pop["states"]):
            lines.append(
                f"    dump_{_dump_dtype(symbol)}(&mut dump, "
                f"&p{p}_state_{position})?;")
        if pop["refractory"] is not None:
            lines += [f"    dump_f64(&mut dump, &p{p}_lastspike)?;",
                      f"    dump.write_all(&p{p}_not_refractory)?;"]
    lines += [f"    dump_u64(&mut dump, {len(d['synapses'])})?;"]
    for q, (syn_def, syn_inst) in enumerate(
            zip(d["synapses"], inst["synapses"], strict=True)):
        lines += [f"    dump_u64(&mut dump, {len(syn_def['states'])})?;",
                  f"    dump_u64(&mut dump, {_synapse_edge_count(syn_inst)})?;"]
        for position, symbol in enumerate(syn_def["states"]):
            lines.append(
                f"    dump_{_dump_dtype(symbol)}(&mut dump, "
                f"&s{q}_state_{position})?;")
        for monitor_position, monitor in enumerate(
                syn_def.get("state_monitors", [])):
            for column, source in enumerate(monitor["sources"]):
                lines.append(
                    f"    dump_{_dump_dtype(source)}(&mut dump, "
                    f"&s{q}_monitor_{monitor_position}_samples_{column})?;")
        lines += [f"    dump_u64(&mut dump, s{q}_delivered + s{q}_post_delivered)?;"]
    lines += ["    dump.write_all(&final_time.to_le_bytes())?;",
              "    dump_finish(dump, output, dump_bytes)?;"]
    return lines


def _v7_emit_event_dump(model, *, needs_event_dump):
    """Emit named EventStreams and EventMonitor samples as binary sidecar."""
    populations = model["definition"]["populations"]
    stream_count = sum(len(pop.get("events", [])) for pop in populations)
    monitor_count = sum(len(pop.get("event_monitors", []))
                        for pop in populations)
    if not needs_event_dump:
        return ["    let event_dump_bytes = 0usize;"]
    terms = ["32usize"]
    for p, pop in enumerate(populations):
        for event_position, _event in enumerate(pop.get("events", [])):
            terms += ["8", f"p{p}_event_history_{event_position}.len()*16"]
        symbols = {symbol["name"]: symbol for symbol in (
            pop["states"] + pop["parameters"] +
            pop.get("linked_variables", []))}
        for monitor_position, monitor in enumerate(pop.get("event_monitors", [])):
            terms += ["16", f"p{p}_event_monitor_{monitor_position}_events.len()*16"]
            terms.extend(
                f"p{p}_event_monitor_{monitor_position}_samples_{column}.len()*"
                f"{_dtype_size(symbols[name])}"
                for column, name in enumerate(monitor["variables"]))
    lines = ["    let event_dump_bytes = " + usize_sum(terms) + ";",
             "    let mut events_dump = BufWriter::with_capacity(65536, "
             "File::create(output.join(\"events.bin\"))?);",
             "    events_dump.write_all(b\"B2EVT001\")?;",
             f"    dump_u64(&mut events_dump, {stream_count})?;"]
    for p, pop in enumerate(populations):
        for event_position, _event in enumerate(pop.get("events", [])):
            lines += [
                f"    dump_u64(&mut events_dump, p{p}_event_history_{event_position}.len())?;",
                f"    dump_spikes_usize(&mut events_dump, "
                f"&p{p}_event_history_{event_position})?;",
            ]
    lines.append(f"    dump_u64(&mut events_dump, {monitor_count})?;")
    for p, pop in enumerate(populations):
        symbols = {symbol["name"]: symbol for symbol in (
            pop["states"] + pop["parameters"] +
            pop.get("linked_variables", []))}
        for monitor_position, monitor in enumerate(pop.get("event_monitors", [])):
            lines += [
                f"    dump_u64(&mut events_dump, "
                f"p{p}_event_monitor_{monitor_position}_events.len())?;",
                f"    dump_u64(&mut events_dump, {len(monitor['variables'])})?;",
                f"    dump_spikes_usize(&mut events_dump, "
                f"&p{p}_event_monitor_{monitor_position}_events)?;",
            ]
            for column, name in enumerate(monitor["variables"]):
                lines.append(
                    f"    dump_{_dump_dtype(symbols[name])}(&mut events_dump, "
                    f"&p{p}_event_monitor_{monitor_position}_samples_{column})?;")
    lines += ["    events_dump.write_all(b\"B2EEND01\")?;",
              "    events_dump.flush()?;",
              "    check(File::open(output.join(\"events.bin\"))?.metadata()?.len() "
              "== event_dump_bytes as u64, \"event dump size mismatch\")?;"]
    return lines


def _v7_emit_metadata(model, *, route_count, parallel_route_count,
                      fused_route_count, final_only_summed_count, distributed=False):
    d, inst = model["definition"], model["instance"]
    pops = d["populations"]
    lines = ['    let mut summary = BufWriter::new(File::create(output.join("summary.json"))?);']
    def text(value): lines.append(f"    summary.write_all({json.dumps(value)}.as_bytes())?;")
    def value(expr): lines.append(f'    write!(summary, "{{}}", {expr})?;')
    text('{"schema":"b2-result-dump-v3","dump_bytes":'); value("dump_bytes")
    text(',"event_dump_bytes":'); value("event_dump_bytes")
    text(',"threads":'); value('simulation_threads')
    text(',"thread_affinity":'); value('thread_affinity')
    text(',"thread_cpus":'); value('&thread_cpus')
    text(',"parallel_state_update":'); value('parallel_state_update')
    text(',"parallel_poisson_input":'); value('parallel_poisson_input')
    text(',"parallel_on_pre":'); value('parallel_on_pre')
    text(',"parallel_event_enqueue":'); value('parallel_event_enqueue')
    text(',"parallel_plasticity":'); value('parallel_plasticity')
    text(',"parallel_summed_variable":'); value('parallel_summed_variable')
    text(',"parallel_synapse_regular":'); value('parallel_synapse_regular')
    text(',"parallel_synapse_state":'); value('parallel_synapse_state')
    text(',"final_only_summed_variable_count":' +
         str(final_only_summed_count))
    text(',"phase_profile":{"enabled":'); value('phase_profile_enabled')
    text(',"ticks":'); value('phase_profile_ticks')
    for name in ("scheduler", "groups", "neuron_state", "synapse_state",
                 "threshold", "primary_events", "primary_enqueue",
                 "primary_apply", "additional_pre",
                 "poisson_input", "post", "reset"):
        text(f',"{name}_seconds":'); value(f'phase_{name}_seconds')
    text('}')
    text(',"event_route_count":' + str(route_count))
    text(',"parallel_event_route_count":' + str(parallel_route_count))
    text(',"fused_event_dispatch_count":' + str(fused_route_count))
    text(',"population_count":' + str(len(pops)) + ',"neuron_count":' + str(inst["neuron_count"]) + ',"spike_count":')
    value(usize_sum(f"p{p}_spikes.len()" for p in range(len(pops))))
    text(',"final_time_seconds":'); value('final_time')
    text(',"synaptic_events":'); value(
        usize_sum(f"s{q}_delivered + s{q}_post_delivered"
                   for q in range(len(d["synapses"]))) or "0")
    text(',"timings":{"initialization_seconds":'); value('initialization_seconds')
    text(',"simulation_and_recording_seconds":'); value('simulation_and_recording_seconds')
    text(',"dump_write_seconds":'); value('dump_write_seconds')
    if distributed:
        text('},"mpi":'); value('mpi_report.trim()'); text('}\n')
    else:
        text('}}\n')
    lines.append('    summary.flush()?;')
    return lines


def write_instance(model, path):
    return _write_instance_verified(migrate_model(model), path)


def _write_instance_verified(model, path):
    if model.get("schema") != CURRENT_SCHEMA:
        return write_instance_v6(model, path)
    if _uses_v6(model):
        return write_instance_v6(_single_population_v6(model), path)
    return write_streamed_instance(model, path)


def write_streamed_instance(model, path, *, schedule_layout=None):
    """Write large external arrays without a whole-instance Python bytearray."""
    from .binary_topology import inspect_csr, copy_region, file_hash, HEADER
    d, inst = model["definition"], model["instance"]
    def encode_values(items, dtype="f64"):
        if dtype == "bool":
            if any(value not in {"00", "01"} for value in items):
                raise ValueError("invalid bool instance encoding")
            return bytes(value == "01" for value in items)
        widths = {"f32": (4, "I"), "i32": (4, "I"), "u32": (4, "I"),
                  "f64": (8, "Q"), "i64": (8, "Q"), "u64": (8, "Q")}
        if dtype not in widths:
            raise ValueError(f"unsupported AOT instance dtype: {dtype}")
        width, code = widths[dtype]
        if any(len(value) != 2 * width for value in items):
            raise ValueError("invalid instance value width")
        # Decode a bounded block of big-endian bit patterns in C. Integer
        # storage preserves floating signed zero and subnormals exactly.
        packed = array.array(code)
        if packed.itemsize != width:
            raise ValueError("unsupported host integer storage width")
        raw = bytes.fromhex("".join(items))
        if len(raw) != width * len(items):
            raise ValueError("invalid instance value width")
        packed.frombytes(raw)
        packed.byteswap()
        return packed.tobytes()

    def values(stream, items, dtype="f64"):
        if isinstance(items, EncodedArray):
            if items.dtype != dtype:
                raise ValueError("packed instance dtype does not match definition")
            for chunk in items.little_endian_chunks():
                stream.write(chunk)
            return
        for start in range(0, len(items), 65536):
            stream.write(encode_values(items[start:start+65536], dtype))
    def integers(stream, items, code="Q"):
        if isinstance(items, IndexArray):
            import numpy as np

            for start in range(0, len(items), 65536):
                raw = items._data[start * items._width:(start + 65536) * items._width]
                width = {"Q": 8, "I": 4}.get(code)
                if width is None:
                    raise ValueError("unsupported packed index width")
                if width == items._width:
                    stream.write(raw)
                else:
                    block = np.frombuffer(raw, dtype=f"<u{items._width}")
                    if width == 4 and np.any(block > 0xffffffff):
                        raise ValueError("instance index exceeds u32")
                    stream.write(block.astype(f"<u{width}").tobytes())
            return
        for start in range(0, len(items), 65536):
            chunk = items[start:start+65536]
            stream.write(struct.pack(f"<{len(chunk)}{code}", *chunk))
    with path.open("wb") as out:
        out.write(b"B2AOT001")
        out.write(struct.pack("<QQ", len(d["populations"]), inst["rng_seed"]))
        for position, (pop, state) in enumerate(zip(d["populations"], inst["populations"], strict=True)):
            out.write(struct.pack("<dQQ", _number(pop["dt"]), pop["steps"], pop["count"]))
            for symbol in pop["states"]: values(out, state["initial_state"][symbol["name"]], symbol["dtype"])
            for symbol in pop["parameters"]: values(out, state["parameters"][symbol["name"]], symbol["dtype"])
            if state["refractory"] is not None:
                ref = state["refractory"]
                out.write(struct.pack("<Q", ref["period_ticks"]))
                values(out, ref["initial_lastspike"])
                out.write(bytes(ref["initial_not_refractory"]))
            generator = state.get("spike_generator")
            if generator is not None:
                schedule_start = out.tell()
                out.write(struct.pack("<Q", len(generator["spike_ticks"])))
                integers(out, generator["spike_ticks"])
                integers(out, generator["spike_indices"])
                if schedule_layout is not None:
                    schedule_layout[position] = (schedule_start, out.tell())
        for definition, syn in zip(d["synapses"], inst["synapses"], strict=True):
            topology = syn["topology"]
            info = None
            if topology["kind"] == "binary_csr":
                info = inspect_csr(topology["path"])
                if file_hash(info["path"]) != topology["sha256"]:
                    raise ValueError("binary CSR changed after export")
                copy_region(info["path"], out, HEADER.size,
                            (info["source_count"]+1)*8 + info["edge_count"]*4)
            elif _explicit_topology(syn):
                integers(out, syn["source"], "I")
                integers(out, syn["target"], "I")
            for symbol in definition["states"]: values(out, syn["initial_state"][symbol["name"]], symbol["dtype"])
            for symbol in definition["parameters"]:
                if info is not None and symbol["index_domain"] == "synapse":
                    column = topology["initializers"][symbol["name"]]
                    copy_region(info["path"], out,
                                info["parameter_offset"] + column*info["edge_count"]*8,
                                info["edge_count"]*8)
                else:
                    values(out, syn["parameters"][symbol["name"]], symbol["dtype"])
            for pathway in syn["pathways"]:
                integers(out, pathway["delay_ticks"])
                out.write(struct.pack("<Q", len(pathway["pending"])))
                integers(out, [e["delivery_tick"] for e in pathway["pending"]])
                integers(out, [e["item"] for e in pathway["pending"]])
        run_clocks = model.get("run", {}).get("clocks")
        if run_clocks is not None:
            integers(out, [clock["start_tick"] for clock in run_clocks])
            final_time = (_number(model["run"]["start"]) +
                          _number(model["run"]["duration"]))
            out.write(struct.pack("<d", final_time))
    return file_hash(path)


def write_project(model, directory, *, plan=None):
    return _write_project_verified(migrate_model(model), directory, plan=plan)


def _write_project_verified(model, directory, *, plan=None):
    """Build using a device-owned verified model and its explicit execution plan."""
    if plan is None:
        plan = _derive_execution_plan(model)
    source = _generate_source_verified(model, plan=plan)
    directory.mkdir(parents=True, exist_ok=False)
    source_path, data_path = directory / "main.rs", directory / "instance.bin"
    source_path.write_text(source)
    timed_blobs = write_timed_array_blobs(model, directory)
    (directory / "execution-plan.json").write_text(plan.to_json())
    native_sources = []
    native_manifest = []
    c_abi_types = {"f64": "double", "i64": "int64_t", "bool": "uint8_t"}
    for position, function in enumerate(model["definition"].get("functions", [])):
        native = function.get("backend_implementations", {}).get("cpu")
        if native is None:
            continue
        native_path = directory / f"function-{position}-{native['source_sha256'][:16]}.c"
        try:
            c_arguments = ", ".join(
                f"{c_abi_types[argument['dtype']]} arg{argument_position}"
                for argument_position, argument in enumerate(
                    function["arguments"])) or "void"
            c_return = c_abi_types[function["return_dtype"]]
        except KeyError as error:
            raise ValueError(
                "b2ir-c-abi-v1 supports only f64/i64/bool signatures") from error
        signature_check = (
            "\n#include <stdint.h>\n"
            f"extern {c_return} {native['symbol']}({c_arguments});\n"
            f"static {c_return} (*const b2ir_signature_check_{position})"
            f"({c_arguments}) = &{native['symbol']};\n")
        translation_unit = native["source"] + signature_check
        native_path.write_text(translation_unit)
        native_sources.append(native_path)
        native_manifest.append({
            "function": function["name"], "abi": native["abi"],
            "symbol": native["symbol"], "source_sha256": native["source_sha256"],
            "translation_unit_sha256": hashlib.sha256(
                translation_unit.encode("utf-8")).hexdigest(),
        })
    data_hash = _write_instance_verified(model, data_path)
    manifest = {"schema": "b2-native-probe-v0", "engine": "rustc-aot",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "execution_plan_schema": plan.schema,
                "execution_plan_sha256": plan.sha256,
                "plan_policy_sha256": plan.policy_sha256,
                "definition_sha256": model["protocol"]["layers"]["definition"],
                "run_sha256": model["protocol"]["layers"]["run"],
                "instance_sha256": data_hash, "neuron_count": model["instance"]["neuron_count"],
                "steps": (model["run"].get("steps") or
                          max(pop["steps"] for pop in model["definition"]["populations"])),
                "synapse_count": sum(_synapse_edge_count(syn)
                                      for syn in model["instance"].get("synapses", [])),
                "timed_arrays": [
                    {"file": filename, "sha256": hashlib.sha256(payload).hexdigest(),
                     "bytes": len(payload)}
                    for _symbol, filename, payload in timed_blobs],
                "native_functions": native_manifest}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return source_path, data_path, manifest


WEIGHTED_BINOMIAL_RUNTIME = r'''
#[inline(always)]
fn counter_binomial_weighted(seed:u64,stream:u64,tick:u64,index:u64,n:u64,p:f64,approximate:bool,cache:&mut NormalCache,weight:f64)->f64 {
    assert!(p.is_finite()&&(0.0..=1.0).contains(&p),"invalid binomial probability");
    if weight==0.0 && (!approximate || n as f64*p<=5.0 || n as f64*(1.0-p)<=5.0) {
        // Exact binomial samples are finite and nonnegative, so their product
        // with zero has exactly the weight's sign. Counter streams have no
        // global draw position to advance. Gaussian samples can be negative
        // and must still be evaluated, including for a zero weight.
        cache.valid=false;
        return weight;
    }
    counter_binomial_cached(seed,stream,tick,index,n,p,approximate,cache)*weight
}
'''


RUNTIME = r'''#![allow(unused_variables, unused_mut, dead_code)]
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};use std::path::Path;
use std::time::Instant;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn env_flag(name:&str)->Result<bool>{match std::env::var(name){Err(std::env::VarError::NotPresent)=>Ok(false),Err(error)=>Err(error.into()),Ok(value)=>match value.as_str(){"1"|"true"|"yes"|"on"=>Ok(true),"0"|"false"|"no"|"off"=>Ok(false),_=>Err(format!("{name} must be a boolean flag").into())}}}
#[inline(always)] fn timed_array_f64(values:&[u8],index:usize)->f64{let Some(start)=index.checked_mul(8)else{return f64::NAN};let Some(end)=start.checked_add(8)else{return f64::NAN};let Some(bytes)=values.get(start..end)else{return f64::NAN};f64::from_le_bytes(bytes.try_into().unwrap())}
#[inline(always)] fn phase_start(enabled:bool)->Option<Instant>{enabled.then(Instant::now)}
#[inline(always)] fn phase_finish(total:&mut f64,started:Option<Instant>){if let Some(started)=started{*total+=started.elapsed().as_secs_f64();}}
const PARALLEL_MIN_TOTAL_WORK:usize=__PARALLEL_MIN_TOTAL_WORK__;
const PARALLEL_TARGET_TASK_WORK:usize=__PARALLEL_TARGET_TASK_WORK__;
const PARALLEL_MIN_TASK_ITEMS:usize=__PARALLEL_MIN_TASK_ITEMS__;
const PARALLEL_EVENT_MIN_EDGES:usize=__PARALLEL_EVENT_MIN_EDGES__;
#[repr(align(64))]
struct WorkerSignal { wake: std::sync::atomic::AtomicUsize, done: std::sync::atomic::AtomicUsize }
struct ParallelShared { generation: std::sync::atomic::AtomicUsize, broadcast: std::sync::atomic::AtomicUsize, active: std::sync::atomic::AtomicUsize, done: std::sync::atomic::AtomicUsize, signals: Vec<WorkerSignal>, data: std::sync::atomic::AtomicUsize, call: std::sync::atomic::AtomicUsize, items: std::sync::atomic::AtomicUsize, tasks: std::sync::atomic::AtomicUsize, shutdown: std::sync::atomic::AtomicBool, ready: std::sync::atomic::AtomicUsize, pin_failed: std::sync::atomic::AtomicBool }
struct Parallel { shared: std::sync::Arc<ParallelShared>, workers: Vec<std::thread::JoinHandle<()>>, threads: usize, affinity_cpus: Vec<usize> }
struct TargetOwnerCsr { offsets: Vec<Vec<usize>>, edges: Vec<Vec<u32>> }
#[derive(Default)]
struct NormalCache { valid: bool, seed: u64, stream: u64, tick: u64, pair: u64, second: f64, binomial: BinomialParameters }
#[derive(Default)]
struct BinomialParameters { valid: bool, n: u64, p_bits: u64, loc: f64, complement: f64, scale: f64, reverse: bool, probability: f64, q: f64, qn: f64, bound: f64 }
fn parse_cpu_list(value:&str)->Option<Vec<usize>>{let mut cpus=Vec::new();for item in value.trim().split(',').filter(|item|!item.is_empty()){let mut range=item.split('-');let start:usize=range.next()?.parse().ok()?;let end:usize=range.next().map_or(Some(start),|value|value.parse().ok())?;if range.next().is_some()||start>end{return None}cpus.extend(start..=end);}cpus.sort_unstable();cpus.dedup();Some(cpus)}
#[cfg(target_os="linux")]
fn cpu_topology_value(cpu:usize,name:&str)->usize{std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/{name}")).ok().and_then(|value|value.trim().parse().ok()).unwrap_or(cpu)}
#[cfg(target_os="linux")]
fn cpu_numa_node(cpu:usize)->usize{let path=format!("/sys/devices/system/cpu/cpu{cpu}");std::fs::read_dir(path).ok().and_then(|entries|entries.filter_map(std::result::Result::ok).filter_map(|entry|entry.file_name().to_str().and_then(|name|name.strip_prefix("node")).and_then(|node|node.parse().ok())).min()).unwrap_or(0)}
#[cfg(target_os="linux")]
fn automatic_cpu_order()->Vec<usize>{let allowed=std::fs::read_to_string("/proc/self/status").ok().and_then(|status|status.lines().find_map(|line|line.strip_prefix("Cpus_allowed_list:").and_then(parse_cpu_list))).unwrap_or_default();let mut nodes:std::collections::BTreeMap<usize,std::collections::BTreeMap<(usize,usize),Vec<usize>>>=std::collections::BTreeMap::new();for cpu in allowed{let node=cpu_numa_node(cpu);let package=cpu_topology_value(cpu,"physical_package_id");let core=cpu_topology_value(cpu,"core_id");nodes.entry(node).or_default().entry((package,core)).or_default().push(cpu);}let mut ordered=Vec::new();for cores in nodes.values_mut(){for siblings in cores.values_mut(){siblings.sort_unstable();}let layers=cores.values().map(Vec::len).max().unwrap_or(0);for layer in 0..layers{for siblings in cores.values(){if let Some(&cpu)=siblings.get(layer){ordered.push(cpu);}}}}ordered}
#[cfg(not(target_os="linux"))]
fn automatic_cpu_order()->Vec<usize>{Vec::new()}
#[cfg(target_os="linux")]
fn pin_current_cpu(cpu:usize)->bool{unsafe extern "C"{fn sched_setaffinity(pid:i32,cpusetsize:usize,mask:*const u8)->i32;}let mut mask=vec![0u8;(cpu/8+1).max(128)];mask[cpu/8]|=1u8<<(cpu%8);unsafe{sched_setaffinity(0,mask.len(),mask.as_ptr())==0}}
#[cfg(not(target_os="linux"))]
fn pin_current_cpu(_cpu:usize)->bool{false}
// Safety contract: lane dispatchers publish a borrowed closure with Release,
// assign disjoint neuron or target-owner ranges, and do not return (or drop
// the closure) until every participating worker has acknowledged the epoch.
// Per-worker cache-line signals avoid a shared completion counter bouncing
// between cores. Inactive lanes park immediately; active lanes also park after
// a bounded wait for a new epoch. Serial generated phases can bypass dispatch,
// so an old active set alone cannot identify workers that should stay runnable.
// Every dispatch unparks its participating lanes after publishing the epoch.
// The park/unpark token prevents a lost wake if dispatch races with parking.
fn parallel_wait(spins:&mut usize){if *spins<64{std::hint::spin_loop();*spins+=1}else{std::thread::yield_now();*spins=0;}}
fn parallel_worker(shared:std::sync::Arc<ParallelShared>,rank:usize,cpu:Option<usize>){use std::sync::atomic::Ordering;if cpu.is_some_and(|cpu|!pin_current_cpu(cpu)){shared.pin_failed.store(true,Ordering::Relaxed);}shared.ready.fetch_add(1,Ordering::Release);let signal=&shared.signals[rank-1];let pool=shared.signals.len()+1;let mut seen=0;loop{let mut spins=0;let mut idle_rounds=0;let epoch=loop{let active=shared.active.load(Ordering::Acquire);if rank>=active{std::thread::park();spins=0;idle_rounds=0;continue}let value=if active==pool{shared.broadcast.load(Ordering::Acquire)}else{signal.wake.load(Ordering::Acquire)};if value>seen{break value}idle_rounds+=1;if idle_rounds>=4096{std::thread::park();spins=0;idle_rounds=0;}else{parallel_wait(&mut spins)}};if shared.shutdown.load(Ordering::Acquire){break}let tasks=shared.tasks.load(Ordering::Relaxed);let call:unsafe fn(usize,usize,usize,usize)=unsafe{std::mem::transmute(shared.call.load(Ordering::Relaxed))};unsafe{call(shared.data.load(Ordering::Relaxed),rank,tasks,shared.items.load(Ordering::Relaxed));}signal.done.store(epoch,Ordering::Release);seen=epoch;}}
impl Parallel {
 fn from_env(enabled:bool)->Result<Self>{use std::sync::atomic::Ordering;let raw=std::env::var("B2_NUM_THREADS").unwrap_or_else(|_|"1".into());let requested:usize=raw.parse().map_err(|_|"B2_NUM_THREADS must be an integer")?;check((1..=256).contains(&requested),"B2_NUM_THREADS must be within 1..256")?;let threads=if enabled{requested}else{1};let mode=std::env::var("B2_THREAD_AFFINITY").unwrap_or_else(|_|"auto".into());check(matches!(mode.as_str(),"auto"|"off"|"required"),"B2_THREAD_AFFINITY must be auto, off, or required")?;let mut affinity_cpus=if mode=="off"{Vec::new()}else{automatic_cpu_order()};if affinity_cpus.len()<threads{check(mode!="required","not enough allowed CPUs for required affinity")?;affinity_cpus.clear();}else{affinity_cpus.truncate(threads);}if let Some(&cpu)=affinity_cpus.first(){let pinned=pin_current_cpu(cpu);check(mode!="required"||pinned,"failed to pin main worker")?;if !pinned{affinity_cpus.clear();}}let signals=(1..threads).map(|_|WorkerSignal{wake:0.into(),done:0.into()}).collect();let shared=std::sync::Arc::new(ParallelShared{generation:0.into(),broadcast:0.into(),active:1.into(),done:0.into(),signals,data:0.into(),call:0.into(),items:0.into(),tasks:0.into(),shutdown:false.into(),ready:0.into(),pin_failed:false.into()});let mut workers=Vec::new();for rank in 1..threads{let shared=shared.clone();let cpu=affinity_cpus.get(rank).copied();workers.push(std::thread::spawn(move||parallel_worker(shared,rank,cpu)));}while shared.ready.load(Ordering::Acquire)<threads-1{std::thread::yield_now();}if shared.pin_failed.load(Ordering::Relaxed){check(mode!="required","failed to pin worker")?;affinity_cpus.clear();}Ok(Self{shared,workers,threads,affinity_cpus})}
 fn threads(&self)->usize{self.threads}
 fn affinity_enabled(&self)->bool{self.affinity_cpus.len()==self.threads}
 fn affinity_json(&self)->String{format!("{:?}",self.affinity_cpus)}
 fn task_count(&self,n:usize,work:usize)->usize{let by_items=n.saturating_add(PARALLEL_MIN_TASK_ITEMS-1)/PARALLEL_MIN_TASK_ITEMS;let total=n.saturating_mul(work);let by_work=total.saturating_add(PARALLEL_TARGET_TASK_WORK-1)/PARALLEL_TARGET_TASK_WORK;self.threads.min(by_items).min(by_work).max(1)}
 fn is_parallel(&self,n:usize,work:usize)->bool{self.threads>1&&n.saturating_mul(work)>=PARALLEL_MIN_TOTAL_WORK&&self.task_count(n,work)>1}
 fn sparse_task_count(&self,n:usize,work:usize)->usize{let total=n.saturating_mul(work);let by_work=total.saturating_add(PARALLEL_EVENT_MIN_EDGES-1)/PARALLEL_EVENT_MIN_EDGES;self.threads.min(n).min(by_work).max(1)}
 fn is_sparse_parallel(&self,n:usize,work:usize)->bool{self.threads>1&&n.saturating_mul(work)>=PARALLEL_EVENT_MIN_EDGES&&self.sparse_task_count(n,work)>1}
 fn events_parallel(&self,events:usize)->bool{self.threads>1&&events>=2048}
 fn dispatch(&self,data:usize,call:usize,items:usize,tasks:usize)->usize{use std::sync::atomic::Ordering;self.shared.data.store(data,Ordering::Relaxed);self.shared.call.store(call,Ordering::Relaxed);self.shared.items.store(items,Ordering::Relaxed);self.shared.tasks.store(tasks,Ordering::Relaxed);self.shared.active.store(tasks,Ordering::Relaxed);let epoch=self.shared.generation.fetch_add(1,Ordering::Relaxed)+1;if tasks==self.threads{self.shared.done.store(0,Ordering::Relaxed);self.shared.broadcast.store(epoch,Ordering::Release);}else{for signal in self.shared.signals.iter().take(tasks-1){signal.wake.store(epoch,Ordering::Release);}}for worker in &self.workers[..tasks-1]{worker.thread().unpark();}epoch}
 fn finish(&self,epoch:usize,tasks:usize){use std::sync::atomic::Ordering;let mut spins=0;for signal in self.shared.signals.iter().take(tasks-1){while signal.done.load(Ordering::Acquire)!=epoch{parallel_wait(&mut spins);}spins=0;}}
 fn for_each<F>(&self,n:usize,work:usize,function:F)where F:Fn(usize,usize)+Send+Sync{unsafe fn invoke<F:Fn(usize,usize)>(data:usize,rank:usize,tasks:usize,n:usize){let function=unsafe{&*(data as *const F)};let chunk=(n+tasks-1)/tasks;let start=rank*chunk;function(start,(start+chunk).min(n));}if !self.is_parallel(n,work){function(0,n);return}let tasks=self.task_count(n,work);let epoch=self.dispatch(&function as *const F as usize,invoke::<F> as usize,n,tasks);let chunk=(n+tasks-1)/tasks;function(0,chunk.min(n));self.finish(epoch,tasks);}
 fn for_each_ranked<F>(&self,n:usize,work:usize,function:F)where F:Fn(usize,usize,usize)+Send+Sync{unsafe fn invoke<F:Fn(usize,usize,usize)>(data:usize,rank:usize,tasks:usize,n:usize){let function=unsafe{&*(data as *const F)};let chunk=(n+tasks-1)/tasks;let start=rank*chunk;function(rank,start,(start+chunk).min(n));}if !self.is_parallel(n,work){function(0,0,n);return}let tasks=self.task_count(n,work);let epoch=self.dispatch(&function as *const F as usize,invoke::<F> as usize,n,tasks);let chunk=(n+tasks-1)/tasks;function(0,0,chunk.min(n));self.finish(epoch,tasks);}
 fn for_each_ranked_sparse<F>(&self,n:usize,work:usize,function:F)where F:Fn(usize,usize,usize)+Send+Sync{unsafe fn invoke<F:Fn(usize,usize,usize)>(data:usize,rank:usize,tasks:usize,n:usize){let function=unsafe{&*(data as *const F)};let chunk=(n+tasks-1)/tasks;let start=rank*chunk;function(rank,start,(start+chunk).min(n));}if !self.is_sparse_parallel(n,work){function(0,0,n);return}let tasks=self.sparse_task_count(n,work);let epoch=self.dispatch(&function as *const F as usize,invoke::<F> as usize,n,tasks);let chunk=(n+tasks-1)/tasks;function(0,0,chunk.min(n));self.finish(epoch,tasks);}
 fn for_lanes<F>(&self,function:F)where F:Fn(usize)+Send+Sync{unsafe fn invoke<F:Fn(usize)>(data:usize,rank:usize,_tasks:usize,_items:usize){let function=unsafe{&*(data as *const F)};function(rank);}if self.threads==1{function(0);return}let epoch=self.dispatch(&function as *const F as usize,invoke::<F> as usize,self.threads,self.threads);function(0);self.finish(epoch,self.threads);}
}
impl Drop for Parallel{fn drop(&mut self){use std::sync::atomic::Ordering;self.shared.shutdown.store(true,Ordering::Relaxed);self.shared.active.store(self.threads,Ordering::Relaxed);let epoch=self.shared.generation.fetch_add(1,Ordering::Relaxed)+1;self.shared.broadcast.store(epoch,Ordering::Release);for worker in &self.workers{worker.thread().unpark();}while let Some(worker)=self.workers.pop(){let _=worker.join();}}}
fn check(ok: bool, message: &str) -> Result<()> { if ok { Ok(()) } else { Err(message.into()) } }
fn exprel(x: f64) -> f64 { if x.abs() < 1e-16 { 1.0 } else if x > 717.0 { f64::INFINITY } else { x.exp_m1()/x } }
struct Reader { bytes: Vec<u8>, at: usize }
#[derive(Default)]
struct EventBatch { sources: Vec<usize>, edges: Vec<usize>, source_mode: bool, event_count: usize }
impl EventBatch { fn clear(&mut self) { self.sources.clear(); self.edges.clear(); self.source_mode=false; self.event_count=0; } }
impl Reader {
 fn open(path: &Path) -> Result<Self> { let mut bytes=Vec::new(); File::open(path)?.take(128*1_048_576+1).read_to_end(&mut bytes)?; check(bytes.len() <= 128*1_048_576, "native instance exceeds input budget")?; check(bytes.starts_with(b"B2AOT001"), "invalid native instance")?; Ok(Self{bytes,at:8}) }
 fn take<const S:usize>(&mut self)->Result<[u8;S]>{let end=self.at.checked_add(S).ok_or("instance overflow")?;let value=self.bytes.get(self.at..end).ok_or("truncated instance")?.try_into()?;self.at=end;Ok(value)}
 fn u64(&mut self)->Result<u64>{Ok(u64::from_le_bytes(self.take()?))}
 fn usize(&mut self)->Result<usize>{usize::try_from(u64::from_le_bytes(self.take()?)).map_err(Into::into)}
 fn usize_vec(&mut self,n:usize)->Result<Vec<usize>>{(0..n).map(|_|self.usize()).collect()}
 fn u8(&mut self)->Result<u8>{let v=self.take::<1>()?[0];check(v<=1,"invalid bool")?;Ok(v)}
 fn u8_vec(&mut self,n:usize)->Result<Vec<u8>>{(0..n).map(|_|self.u8()).collect()}
 fn i32(&mut self)->Result<i32>{Ok(i32::from_le_bytes(self.take()?))}
 fn i32_vec(&mut self,n:usize)->Result<Vec<i32>>{(0..n).map(|_|self.i32()).collect()}
 fn i64(&mut self)->Result<i64>{Ok(i64::from_le_bytes(self.take()?))}
 fn i64_vec(&mut self,n:usize)->Result<Vec<i64>>{(0..n).map(|_|self.i64()).collect()}
 fn u32(&mut self)->Result<u32>{Ok(u32::from_le_bytes(self.take()?))}
 fn f32(&mut self)->Result<f32>{let v=f32::from_le_bytes(self.take()?);check(v.is_finite(),"non-finite instance value")?;Ok(v)}
 fn f32_vec(&mut self,n:usize)->Result<Vec<f32>>{(0..n).map(|_|self.f32()).collect()}
 fn f64(&mut self)->Result<f64>{let v=f64::from_le_bytes(self.take()?);check(v.is_finite(),"non-finite instance value")?;Ok(v)}
 fn f64_vec(&mut self,n:usize)->Result<Vec<f64>>{(0..n).map(|_|self.f64()).collect()}
 fn u32_vec(&mut self,n:usize)->Result<Vec<u32>>{(0..n).map(|_|self.u32()).collect()}
 fn u64_vec(&mut self,n:usize)->Result<Vec<u64>>{(0..n).map(|_|self.u64()).collect()}
 fn bool_vec(&mut self,n:usize)->Result<Vec<u8>>{(0..n).map(|_|{let v=self.take::<1>()?[0];check(v<=1,"invalid bool")?;Ok(v)}).collect()}
 fn end(&self)->Result<()>{check(self.at==self.bytes.len(),"trailing instance data")}
}
fn dump_u64<W:Write>(w:&mut W,value:usize)->Result<()>{w.write_all(&(value as u64).to_le_bytes())?;Ok(())}
fn dump_u8<W:Write>(w:&mut W,values:&[u8])->Result<()>{check(values.iter().all(|&v|v<=1),"invalid bool output")?;w.write_all(values)?;Ok(())}
fn dump_i32<W:Write>(w:&mut W,values:&[i32])->Result<()>{let mut bytes=[0u8;8192];for chunk in values.chunks(2048){for(slot,value)in bytes.chunks_exact_mut(4).zip(chunk){slot.copy_from_slice(&value.to_le_bytes());}w.write_all(&bytes[..chunk.len()*4])?;}Ok(())}
fn dump_i64<W:Write>(w:&mut W,values:&[i64])->Result<()>{let mut bytes=[0u8;8192];for chunk in values.chunks(1024){for(slot,value)in bytes.chunks_exact_mut(8).zip(chunk){slot.copy_from_slice(&value.to_le_bytes());}w.write_all(&bytes[..chunk.len()*8])?;}Ok(())}
fn dump_u32<W:Write>(w:&mut W,values:&[u32])->Result<()>{let mut bytes=[0u8;8192];for chunk in values.chunks(2048){for(slot,value)in bytes.chunks_exact_mut(4).zip(chunk){slot.copy_from_slice(&value.to_le_bytes());}w.write_all(&bytes[..chunk.len()*4])?;}Ok(())}
fn dump_u64_values<W:Write>(w:&mut W,values:&[u64])->Result<()>{let mut bytes=[0u8;8192];for chunk in values.chunks(1024){for(slot,value)in bytes.chunks_exact_mut(8).zip(chunk){slot.copy_from_slice(&value.to_le_bytes());}w.write_all(&bytes[..chunk.len()*8])?;}Ok(())}
fn dump_f32<W:Write>(w:&mut W,values:&[f32])->Result<()>{let mut bytes=[0u8;8192];for chunk in values.chunks(2048){for(slot,value)in bytes.chunks_exact_mut(4).zip(chunk){slot.copy_from_slice(&value.to_le_bytes());}w.write_all(&bytes[..chunk.len()*4])?;}Ok(())}
fn dump_f64<W:Write>(w:&mut W,values:&[f64])->Result<()>{let mut bytes=[0u8;8192];for chunk in values.chunks(1024){for(slot,value)in bytes.chunks_exact_mut(8).zip(chunk){slot.copy_from_slice(&value.to_le_bytes());}w.write_all(&bytes[..chunk.len()*8])?;}Ok(())}
fn dump_indices<W:Write>(w:&mut W,values:&[usize])->Result<()>{let mut bytes=[0u8;8192];for chunk in values.chunks(1024){for(slot,&value)in bytes.chunks_exact_mut(8).zip(chunk){slot.copy_from_slice(&(value as i64).to_le_bytes());}w.write_all(&bytes[..chunk.len()*8])?;}Ok(())}
fn dump_spikes_u32<W:Write>(w:&mut W,values:&[(u32,u32)])->Result<()>{let mut bytes=[0u8;8192];for chunk in values.chunks(512){for(slot,&(tick,index))in bytes.chunks_exact_mut(16).zip(chunk){slot[..8].copy_from_slice(&(tick as i64).to_le_bytes());slot[8..].copy_from_slice(&(index as i64).to_le_bytes());}w.write_all(&bytes[..chunk.len()*16])?;}Ok(())}
fn dump_spikes_usize<W:Write>(w:&mut W,values:&[(usize,usize)])->Result<()>{let mut bytes=[0u8;8192];for chunk in values.chunks(512){for(slot,&(tick,index))in bytes.chunks_exact_mut(16).zip(chunk){slot[..8].copy_from_slice(&(tick as i64).to_le_bytes());slot[8..].copy_from_slice(&(index as i64).to_le_bytes());}w.write_all(&bytes[..chunk.len()*16])?;}Ok(())}
fn dump_start(output:&Path,size:usize,populations:usize,neurons:usize)->Result<BufWriter<File>>{let mut w=BufWriter::with_capacity(65536,File::create(output.join("results.bin"))?);w.write_all(b"B2DMP001")?;w.write_all(&3u32.to_le_bytes())?;w.write_all(&0x01020304u32.to_le_bytes())?;dump_u64(&mut w,populations)?;dump_u64(&mut w,neurons)?;dump_u64(&mut w,size)?;Ok(w)}
fn dump_finish(mut w:BufWriter<File>,output:&Path,size:usize)->Result<()>{w.write_all(b"B2END001")?;w.flush()?;drop(w);check(File::open(output.join("results.bin"))?.metadata()?.len()==size as u64,"result dump size mismatch")}
fn record_spike(spikes: &mut Vec<(u32,u32)>, tick:usize, i:usize)->Result<()> { if spikes.len()==spikes.capacity(){let limit=(N*STEPS).min(10_000_000);check(spikes.len()<limit,"spike recording budget exceeded")?;let capacity=(spikes.len().max(512)*2).min(limit);spikes.try_reserve_exact(capacity-spikes.len())?;}spikes.push((tick as u32,i as u32));Ok(())}
fn refractory_available(tick:usize,last:f64,dt:f64,period:usize)->bool{(((tick as f64*dt-last)+1e-3*dt)/dt)>=period as f64}
fn first_available_tick(last:f64,dt:f64,period:usize)->usize{let raw=(period as f64+last/dt-1e-3).ceil();let mut tick=if raw<=0.0{0}else{raw as usize};while tick>0&&refractory_available(tick-1,last,dt,period){tick-=1;}while !refractory_available(tick,last,dt,period){tick+=1;}tick}
fn mix64(mut value:u64)->u64{value=(value^(value>>30)).wrapping_mul(0xbf58476d1ce4e5b9);value=(value^(value>>27)).wrapping_mul(0x94d049bb133111eb);value^(value>>31)}
fn counter_uniform_draw(seed:u64,stream:u64,tick:u64,index:u64,draw:u64)->f64{let counter=seed^stream.wrapping_mul(0x9e3779b97f4a7c15)^tick.wrapping_mul(0xd1b54a32d192ed03)^index.wrapping_mul(0x94d049bb133111eb)^draw.wrapping_mul(0x369dea0f31a53f85);((mix64(counter)>>11)as f64)*(1.0/9007199254740992.0)}
fn counter_uniform(seed:u64,stream:u64,tick:u64,index:u64)->f64{counter_uniform_draw(seed,stream,tick,index,0)}
fn counter_normal_pair(seed:u64,stream:u64,tick:u64,pair:u64)->(f64,f64){let mut draw=0u64;loop{let x1=2.0*counter_uniform_draw(seed,stream,tick,pair,draw)-1.0;let x2=2.0*counter_uniform_draw(seed,stream,tick,pair,draw+1)-1.0;let radius=x1*x1+x2*x2;if radius<1.0&&radius!=0.0{let factor=(-2.0*radius.ln()/radius).sqrt();return(factor*x1,factor*x2)}draw=draw.wrapping_add(2);}}
fn counter_normal(seed:u64,stream:u64,tick:u64,index:u64)->f64{let pair=counter_normal_pair(seed,stream,tick,index/2);if index&1==0{pair.0}else{pair.1}}
fn counter_normal_cached(seed:u64,stream:u64,tick:u64,index:u64,cache:&mut NormalCache)->f64{let pair=index/2;if index&1==1&&cache.valid&&cache.seed==seed&&cache.stream==stream&&cache.tick==tick&&cache.pair==pair{cache.valid=false;return cache.second}let values=counter_normal_pair(seed,stream,tick,pair);if index&1==0{cache.valid=true;cache.seed=seed;cache.stream=stream;cache.tick=tick;cache.pair=pair;cache.second=values.1;values.0}else{cache.valid=false;values.1}}
fn counter_binomial(seed:u64,stream:u64,tick:u64,index:u64,n:u64,p:f64,approximate:bool)->f64{assert!(p.is_finite()&&(0.0..=1.0).contains(&p),"invalid binomial probability");if p==0.0{return 0.0}if p==1.0{return n as f64}let loc=n as f64*p;let complement=n as f64*(1.0-p);if approximate&&loc>5.0&&complement>5.0{return counter_normal(seed,stream,tick,index)*(loc*(1.0-p)).sqrt()+loc}let reverse=p>0.5;let probability=if reverse{1.0-p}else{p};let q=1.0-probability;let qn=(n as f64*q.ln()).exp();let bound=(n as f64).min(n as f64*probability+10.0*(n as f64*probability*q+1.0).sqrt());let mut draw=0u64;'sample:loop{let mut x=0u64;let mut px=qn;let mut u=counter_uniform_draw(seed,stream,tick,index,draw);loop{if u<=px{return(if reverse{n-x}else{x})as f64}x+=1;if x as f64>bound{draw=draw.wrapping_add(1);continue 'sample}u-=px;px=((n-x+1)as f64*probability*px)/(x as f64*q);}}}
fn counter_binomial_cached(seed:u64,stream:u64,tick:u64,index:u64,n:u64,p:f64,approximate:bool,cache:&mut NormalCache)->f64 {
    assert!(p.is_finite()&&(0.0..=1.0).contains(&p),"invalid binomial probability");
    if p==0.0 { cache.valid=false; return 0.0; }
    if p==1.0 { cache.valid=false; return n as f64; }
    // Cache distribution setup only: arithmetic order and counter draws stay
    // identical to the scalar sampler. The exact key also supports several
    // distributions sharing one vector-local cache.
    if !cache.binomial.valid || cache.binomial.n!=n || cache.binomial.p_bits!=p.to_bits() {
        let loc=n as f64*p;
        let complement=n as f64*(1.0-p);
        let reverse=p>0.5;
        let probability=if reverse{1.0-p}else{p};
        let q=1.0-probability;
        let qn=(n as f64*q.ln()).exp();
        let bound=(n as f64).min(n as f64*probability+10.0*(n as f64*probability*q+1.0).sqrt());
        cache.binomial=BinomialParameters{valid:true,n,p_bits:p.to_bits(),loc,complement,scale:(loc*(1.0-p)).sqrt(),reverse,probability,q,qn,bound};
    }
    if approximate && cache.binomial.loc>5.0 && cache.binomial.complement>5.0 {
        return counter_normal_cached(seed,stream,tick,index,cache)*cache.binomial.scale+cache.binomial.loc;
    }
    cache.valid=false;
    let parameters=&cache.binomial;
    let mut draw=0u64;
    'sample:loop {
        let mut x=0u64;
        let mut px=parameters.qn;
        let mut u=counter_uniform_draw(seed,stream,tick,index,draw);
        loop {
            if u<=px {return (if parameters.reverse{n-x}else{x}) as f64;}
            x+=1;
            if x as f64>parameters.bound {draw=draw.wrapping_add(1);continue 'sample;}
            u-=px;
            px=((n-x+1) as f64*parameters.probability*px)/(x as f64*parameters.q);
        }
    }
}
fn log_gamma_positive(value:f64)->f64{const C:[f64;8]=[676.5203681218851,-1259.1392167224028,771.3234287776531,-176.6150291621406,12.507343278686905,-0.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];let z=value-1.0;let mut x=0.9999999999998099;for(index,coefficient)in C.iter().enumerate(){x+=coefficient/(z+index as f64+1.0)}let t=z+7.5;0.9189385332046727+(z+0.5)*t.ln()-t+x.ln()}
fn counter_poisson(seed:u64,stream:u64,tick:u64,index:u64,lambda:f64)->f64{if lambda==0.0{return 0.0}assert!(lambda.is_finite()&&(0.0..=1.0e12).contains(&lambda),"invalid poisson lambda");if lambda<10.0{let limit=(-lambda).exp();let mut product=1.0;let mut sample=0u64;loop{product*=counter_uniform_draw(seed,stream,tick,index,sample);if product<=limit{return sample as f64}sample+=1}}let root=lambda.sqrt();let b=0.931+2.53*root;let a=-0.059+0.02483*b;let inverse_alpha=1.1239+1.1328/(b-3.4);let squeeze=0.9277-3.6224/(b-2.0);let mut draw=0u64;loop{let u=counter_uniform_draw(seed,stream,tick,index,draw)-0.5;let v=counter_uniform_draw(seed,stream,tick,index,draw+1);draw=draw.wrapping_add(2);let us=0.5-u.abs();let candidate=((2.0*a/us+b)*u+lambda+0.43).floor();if us>=0.07&&v<=squeeze{return candidate}if candidate<0.0||(us<0.013&&v>us){continue}let left=(v*inverse_alpha/(a/(us*us)+b)).ln();let right=-lambda+candidate*lambda.ln()-log_gamma_positive(candidate+1.0);if left<=right{return candidate}}}
fn checked_trunc(value:f64)->f64{assert!(value.abs()<=9007199254740992.0,"int conversion outside exact f64 range");value.trunc()}
fn checked_i32_floor_div(a:i32,b:i32)->i32{assert!(b!=0,"integer division by zero");let q=a.wrapping_div(b);let r=a.wrapping_rem(b);if r!=0&&(r<0)!=(b<0){q.wrapping_sub(1)}else{q}}
fn checked_i64_floor_div(a:i64,b:i64)->i64{assert!(b!=0,"integer division by zero");let q=a.wrapping_div(b);let r=a.wrapping_rem(b);if r!=0&&(r<0)!=(b<0){q.wrapping_sub(1)}else{q}}
fn checked_u32_floor_div(a:u32,b:u32)->u32{a.checked_div(b).expect("integer division by zero")}
fn checked_u64_floor_div(a:u64,b:u64)->u64{a.checked_div(b).expect("integer division by zero")}
fn checked_i32_mod(a:i32,b:i32)->i32{a.wrapping_sub(checked_i32_floor_div(a,b).wrapping_mul(b))}
fn checked_i64_mod(a:i64,b:i64)->i64{a.wrapping_sub(checked_i64_floor_div(a,b).wrapping_mul(b))}
fn checked_u32_mod(a:u32,b:u32)->u32{a.checked_rem(b).expect("integer modulo by zero")}
fn checked_u64_mod(a:u64,b:u64)->u64{a.checked_rem(b).expect("integer modulo by zero")}
fn checked_f32(value:f64)->f32{let result=value as f32;assert!(result.is_finite(),"f32 conversion produced a non-finite value");result}
fn checked_timestep(time:f64,dt:f64)->f64{let value=((time+1e-3*dt)/dt).trunc();assert!(time>=0.0&&dt.is_finite()&&dt>0.0&&value.is_finite()&&value<=9007199254740992.0,"timestep outside exact f64 range");value}
fn checked_tick_offset(tick:f64,offset:i64)->f64{let value=tick+offset as f64;assert!(value.abs()<=9007199254740992.0,"tick offset outside exact f64 range");value}
fn topology_mix64(mut value:u64)->u64{value=value.wrapping_add(0x9e3779b97f4a7c15);value=(value^(value>>30)).wrapping_mul(0xbf58476d1ce4e5b9);value=(value^(value>>27)).wrapping_mul(0x94d049bb133111eb);value^(value>>31)}
fn topology_draw(seed:u64,stream:u64,edge:usize,attempt:u64)->u64{topology_mix64(seed^stream.wrapping_mul(0xd2b74407b1ce6e93)^(edge as u64).wrapping_mul(0x9e3779b97f4a7c15)^attempt.wrapping_mul(0xca5a826395121157))}
fn topology_bounded(seed:u64,stream:u64,edge:usize,upper:usize)->usize{((topology_draw(seed,stream,edge,0)as u128*upper as u128)>>64)as usize}
fn topology_normal(seed:u64,stream:u64,edge:usize,attempt:u64)->f64{let u1=((topology_draw(seed,stream,edge,attempt*2)>>11)as f64+0.5)*(1.0/9007199254740992.0);let u2=((topology_draw(seed,stream+1,edge,attempt*2+1)>>11)as f64+0.5)*(1.0/9007199254740992.0);(-2.0*u1.ln()).sqrt()*(std::f64::consts::TAU*u2).cos()}
fn valid_clipped_normal(mean:f64,std:f64,minimum:Option<f64>,maximum:Option<f64>)->bool{mean.is_finite()&&std.is_finite()&&std>=0.0&&minimum.is_none_or(f64::is_finite)&&maximum.is_none_or(f64::is_finite)&&minimum.zip(maximum).is_none_or(|(low,high)|low<=high)}
fn clipped_normal_value(seed:u64,stream:u64,edge:usize,mean:f64,std:f64,minimum:Option<f64>,maximum:Option<f64>)->Option<f64>{for attempt in 0..1000{let value=mean+std*topology_normal(seed,stream,edge,attempt);if minimum.is_none_or(|low|value>=low)&&maximum.is_none_or(|high|value<=high){return Some(value)}}None}
fn materialize_clipped_normal(parallel:&Parallel,edges:usize,seed:u64,stream:u64,mean:f64,std:f64,minimum:Option<f64>,maximum:Option<f64>)->Result<Vec<f64>>{use std::sync::atomic::{AtomicBool,Ordering};check(valid_clipped_normal(mean,std,minimum,maximum),"invalid clipped-normal initializer")?;let mut values=vec![0.0;edges];let pointer=values.as_mut_ptr() as usize;let failed=AtomicBool::new(false);parallel.for_each(edges,32,|start,end|{for edge in start..end{if let Some(value)=clipped_normal_value(seed,stream,edge,mean,std,minimum,maximum){unsafe{*((pointer as *mut f64).add(edge))=value;}}else{failed.store(true,Ordering::Relaxed);}}});check(!failed.load(Ordering::Relaxed),"clipped-normal rejection limit exceeded")?;Ok(values)}
fn materialize_clipped_normal_ticks(parallel:&Parallel,edges:usize,seed:u64,stream:u64,mean:f64,std:f64,minimum:Option<f64>,maximum:Option<f64>,dt:f64)->Result<Vec<usize>>{use std::sync::atomic::{AtomicBool,Ordering};check(valid_clipped_normal(mean,std,minimum,maximum)&&dt.is_finite()&&dt>0.0,"invalid clipped-normal delay initializer")?;let mut values=vec![0usize;edges];let pointer=values.as_mut_ptr() as usize;let failed=AtomicBool::new(false);parallel.for_each(edges,32,|start,end|{for edge in start..end{if let Some(value)=clipped_normal_value(seed,stream,edge,mean,std,minimum,maximum){unsafe{*((pointer as *mut usize).add(edge))=(value/dt+0.5).floor()as usize;}}else{failed.store(true,Ordering::Relaxed);}}});check(!failed.load(Ordering::Relaxed),"clipped-normal rejection limit exceeded")?;Ok(values)}
fn materialize_uniform(parallel:&Parallel,edges:usize,seed:u64,stream:u64,minimum:f64,maximum:f64)->Result<Vec<f64>>{check(minimum.is_finite()&&maximum.is_finite()&&minimum<=maximum,"invalid uniform initializer")?;let mut values=vec![0.0;edges];let pointer=values.as_mut_ptr()as usize;parallel.for_each(edges,32,|start,end|{for edge in start..end{let unit=((topology_draw(seed,stream,edge,0)>>11)as f64+0.5)*(1.0/9007199254740992.0);unsafe{*((pointer as*mut f64).add(edge))=minimum+(maximum-minimum)*unit;}}});Ok(values)}
fn materialize_uniform_ticks(parallel:&Parallel,edges:usize,seed:u64,stream:u64,minimum:f64,maximum:f64,dt:f64)->Result<Vec<usize>>{check(minimum.is_finite()&&maximum.is_finite()&&minimum>=0.0&&minimum<=maximum&&dt.is_finite()&&dt>0.0&&maximum/dt<1000001.0,"invalid uniform delay initializer")?;Ok(materialize_uniform(parallel,edges,seed,stream,minimum,maximum)?.into_iter().map(|value|(value/dt+0.5).floor()as usize).collect())}
fn fixed_total_topology(parallel:&Parallel,sources:usize,targets:usize,edges:usize,seed:u64)->Result<(Vec<usize>,Vec<u32>)>{check(sources>0&&targets>0&&edges>0&&edges<=u32::MAX as usize,"invalid fixed-total topology")?;const WORK:usize=8;if !parallel.is_parallel(edges,WORK){let mut counts=vec![0usize;sources];for edge in 0..edges{counts[topology_bounded(seed,0,edge,sources)]+=1;}let mut offsets=Vec::with_capacity(sources+1);offsets.push(0usize);for count in counts{offsets.push(offsets.last().copied().unwrap().checked_add(count).ok_or("fixed-total edge count exceeds usize")?);}let mut cursor=offsets[..sources].to_vec();let mut target_index=vec![0u32;edges];for edge in 0..edges{let source=topology_bounded(seed,0,edge,sources);let position=cursor[source];cursor[source]+=1;target_index[position]=topology_bounded(seed,1,edge,targets)as u32;}return Ok((offsets,target_index))}let tasks=parallel.task_count(edges,WORK);let mut lane_counts:Vec<Vec<usize>>=(0..tasks).map(|_|vec![0usize;sources]).collect();let counts_pointer=lane_counts.as_mut_ptr()as usize;parallel.for_each_ranked(edges,WORK,|rank,start,end|{let counts=unsafe{&mut *((counts_pointer as *mut Vec<usize>).add(rank))};for edge in start..end{counts[topology_bounded(seed,0,edge,sources)]+=1;}});let mut offsets=Vec::with_capacity(sources+1);offsets.push(0usize);for source in 0..sources{let count=lane_counts.iter().map(|lane|lane[source]).sum::<usize>();offsets.push(offsets.last().copied().unwrap().checked_add(count).unwrap());}check(offsets[sources]==edges,"fixed-total edge count mismatch")?;let mut lane_cursors=vec![vec![0usize;sources];tasks];for source in 0..sources{let mut position=offsets[source];for rank in 0..tasks{lane_cursors[rank][source]=position;position+=lane_counts[rank][source];}}let cursors_pointer=lane_cursors.as_mut_ptr()as usize;let mut target_index=vec![0u32;edges];let target_pointer=target_index.as_mut_ptr()as usize;parallel.for_each_ranked(edges,WORK,|rank,start,end|{let cursor=unsafe{&mut *((cursors_pointer as *mut Vec<usize>).add(rank))};for edge in start..end{let source=topology_bounded(seed,0,edge,sources);let position=cursor[source];cursor[source]+=1;unsafe{*((target_pointer as *mut u32).add(position))=topology_bounded(seed,1,edge,targets)as u32;}}});Ok((offsets,target_index))}
fn fixed_indegree_each(sources:usize,indegree:usize,target:usize,seed:u64,mut visit:impl FnMut(usize)){let mut selected=std::collections::HashSet::with_capacity(indegree);for candidate in sources-indegree..sources{let logical=target*sources+candidate;let draw=topology_bounded(seed,0,logical,candidate+1);let source=if selected.contains(&draw){candidate}else{draw};selected.insert(source);visit(source);}}
fn fixed_indegree_topology(parallel:&Parallel,sources:usize,targets:usize,indegree:usize,edges:usize,seed:u64)->Result<(Vec<usize>,Vec<u32>)>{check(sources>0&&targets>0&&indegree>0&&indegree<=sources&&targets.checked_mul(indegree)==Some(edges)&&targets.checked_mul(sources).is_some()&&edges<=u32::MAX as usize,"invalid fixed-indegree topology")?;let work=indegree.max(1);let tasks=parallel.task_count(targets,work);let mut lane_counts:Vec<Vec<usize>>=(0..tasks).map(|_|vec![0usize;sources]).collect();let counts_pointer=lane_counts.as_mut_ptr()as usize;parallel.for_each_ranked(targets,work,|rank,start,end|{let counts=unsafe{&mut *((counts_pointer as *mut Vec<usize>).add(rank))};for target in start..end{fixed_indegree_each(sources,indegree,target,seed,|source|counts[source]+=1);}});let mut offsets=Vec::with_capacity(sources+1);offsets.push(0usize);for source in 0..sources{let count=lane_counts.iter().map(|lane|lane[source]).sum::<usize>();offsets.push(offsets.last().copied().unwrap().checked_add(count).unwrap());}check(offsets[sources]==edges,"fixed-indegree edge count mismatch")?;let mut lane_cursors=vec![vec![0usize;sources];tasks];for source in 0..sources{let mut position=offsets[source];for rank in 0..tasks{lane_cursors[rank][source]=position;position+=lane_counts[rank][source];}}let cursors_pointer=lane_cursors.as_mut_ptr()as usize;let mut target_index=vec![0u32;edges];let target_pointer=target_index.as_mut_ptr()as usize;parallel.for_each_ranked(targets,work,|rank,start,end|{let cursor=unsafe{&mut *((cursors_pointer as *mut Vec<usize>).add(rank))};for target in start..end{fixed_indegree_each(sources,indegree,target,seed,|source|{let position=cursor[source];cursor[source]+=1;unsafe{*((target_pointer as *mut u32).add(position))=target as u32;}});}});Ok((offsets,target_index))}
fn source_csr(source:&[u32],n:usize)->(Vec<usize>,Vec<usize>){let mut offsets=vec![0;n+1];for &s in source{offsets[s as usize+1]+=1;}for i in 0..n{offsets[i+1]+=offsets[i];}let mut next=offsets[..n].to_vec();let mut edges=vec![0;source.len()];for(e,&s)in source.iter().enumerate(){let s=s as usize;edges[next[s]]=e;next[s]+=1;}(offsets,edges)}
fn source_csr_u32(source:&[u32],n:usize)->(Vec<usize>,Vec<u32>){let mut offsets=vec![0;n+1];for &s in source{offsets[s as usize+1]+=1;}for i in 0..n{offsets[i+1]+=offsets[i];}let mut next=offsets[..n].to_vec();let mut edges=vec![0;source.len()];for(e,&s)in source.iter().enumerate(){let s=s as usize;edges[next[s]]=e as u32;next[s]+=1;}(offsets,edges)}
fn source_csr_delay(source:&[u32],delays:&[usize],wanted:usize,n:usize)->(Vec<usize>,Vec<usize>){let mut offsets=vec![0;n+1];for(&s,&d)in source.iter().zip(delays){if d==wanted{offsets[s as usize+1]+=1;}}for i in 0..n{offsets[i+1]+=offsets[i];}let mut next=offsets[..n].to_vec();let mut edges=vec![0;offsets[n]];for(e,(&s,&d))in source.iter().zip(delays).enumerate(){if d==wanted{let s=s as usize;edges[next[s]]=e;next[s]+=1;}}(offsets,edges)}
fn degree_balanced_target_owners(degrees:&[usize],owners:usize)->Vec<usize>{let total:usize=degrees.iter().sum();if total==0{return(0..degrees.len()).map(|target|target.saturating_mul(owners)/degrees.len().max(1)).collect()}let mut boundaries=vec![0usize;owners+1];boundaries[owners]=degrees.len();let(mut target,mut prefix)=(0usize,0usize);for owner in 1..owners{let desired=((total as u128*owner as u128)/owners as u128)as usize;while target<degrees.len()&&prefix.saturating_add(degrees[target])<=desired{prefix+=degrees[target];target+=1;}if target<degrees.len(){let after=prefix.saturating_add(degrees[target]);if desired.saturating_sub(prefix)>after.saturating_sub(desired){prefix=after;target+=1;}}boundaries[owner]=target.max(boundaries[owner-1]);}let mut result=vec![owners-1;degrees.len()];for owner in 0..owners{for value in &mut result[boundaries[owner]..boundaries[owner+1]]{*value=owner;}}result}
fn target_owner_csr_usize(source_offsets:&[usize],source_edges:&[usize],target:&[u32],sources:usize,target_owners:&[usize],owners:usize)->TargetOwnerCsr{let mut offsets=vec![vec![0usize;sources+1];owners];for source in 0..sources{for &edge in &source_edges[source_offsets[source]..source_offsets[source+1]]{let owner=target_owners[target[edge]as usize];offsets[owner][source+1]+=1;}}for owner_offsets in &mut offsets{for source in 0..sources{owner_offsets[source+1]+=owner_offsets[source];}}let mut next:Vec<Vec<usize>>=offsets.iter().map(|owner|owner[..sources].to_vec()).collect();let mut edges:Vec<Vec<u32>>=offsets.iter().map(|owner|vec![0u32;owner[sources]]).collect();for source in 0..sources{for &edge in &source_edges[source_offsets[source]..source_offsets[source+1]]{let owner=target_owners[target[edge]as usize];edges[owner][next[owner][source]]=edge as u32;next[owner][source]+=1;}}TargetOwnerCsr{offsets,edges}}
fn target_owner_csr_u32(source_offsets:&[usize],source_edges:&[u32],target:&[u32],sources:usize,target_start:usize,target_owners:&[usize],owners:usize)->TargetOwnerCsr{let mut offsets=vec![vec![0usize;sources+1];owners];for source in 0..sources{for &edge_index in &source_edges[source_offsets[source]..source_offsets[source+1]]{let edge=edge_index as usize;let owner=target_owners[target_start+target[edge]as usize];offsets[owner][source+1]+=1;}}for owner_offsets in &mut offsets{for source in 0..sources{owner_offsets[source+1]+=owner_offsets[source];}}let mut next:Vec<Vec<usize>>=offsets.iter().map(|owner|owner[..sources].to_vec()).collect();let mut edges:Vec<Vec<u32>>=offsets.iter().map(|owner|vec![0u32;owner[sources]]).collect();for source in 0..sources{for &edge_index in &source_edges[source_offsets[source]..source_offsets[source+1]]{let edge=edge_index as usize;let owner=target_owners[target_start+target[edge]as usize];edges[owner][next[owner][source]]=edge_index;next[owner][source]+=1;}}TargetOwnerCsr{offsets,edges}}
fn target_owner_csr_contiguous(source_offsets:&[usize],target:&[u32],sources:usize,target_start:usize,target_owners:&[usize],owners:usize)->TargetOwnerCsr{let mut offsets=vec![vec![0usize;sources+1];owners];for source in 0..sources{for edge in source_offsets[source]..source_offsets[source+1]{let owner=target_owners[target_start+target[edge]as usize];offsets[owner][source+1]+=1;}}for owner_offsets in &mut offsets{for source in 0..sources{owner_offsets[source+1]+=owner_offsets[source];}}let mut next:Vec<Vec<usize>>=offsets.iter().map(|owner|owner[..sources].to_vec()).collect();let mut edges:Vec<Vec<u32>>=offsets.iter().map(|owner|vec![0u32;owner[sources]]).collect();for source in 0..sources{for edge in source_offsets[source]..source_offsets[source+1]{let owner=target_owners[target_start+target[edge]as usize];edges[owner][next[owner][source]]=edge as u32;next[owner][source]+=1;}}TargetOwnerCsr{offsets,edges}}
fn main(){if let Err(e)=run(){eprintln!("b2-native: {e}");std::process::exit(1)}}
struct SpikeInput { population: usize, ticks: Vec<usize>, indices: Vec<usize> }
impl SpikeInput {
 fn load(population:usize,path:&Path)->Result<Self>{
    let mut bytes=Vec::new();File::open(path)?.take(128*1_048_576+1).read_to_end(&mut bytes)?;
    check(bytes.len()<=128*1_048_576 && bytes.len()>=16,"invalid spike input size")?;
    check(&bytes[..8]==b"B2SPIK01","invalid spike input header")?;
    let n=usize::try_from(u64::from_le_bytes(bytes[8..16].try_into()?))?;
    check(n.checked_mul(16).and_then(|v|v.checked_add(16))==Some(bytes.len()),"spike input length mismatch")?;
    let values=bytes[16..].chunks_exact(8).map(|v|usize::try_from(u64::from_le_bytes(v.try_into().unwrap()))).collect::<std::result::Result<Vec<_>,_>>()?;
    Ok(Self{population,ticks:values[..n].to_vec(),indices:values[n..].to_vec()})
 }
 fn validate(&self,count:usize,start:usize,steps:usize)->Result<()>{
    let stop=start.checked_add(steps).ok_or("spike input tick overflow")?;
    for i in 0..self.ticks.len(){
        check(self.indices[i]<count && self.ticks[i]>=start && self.ticks[i]<stop,"spike input outside population/run")?;
        if i>0 { check((self.ticks[i-1],self.indices[i-1])<(self.ticks[i],self.indices[i]),"spike input must be sorted and unique per tick")?; }
    }
    Ok(())
 }
}
fn run()->Result<()>{
 let initialization_started=Instant::now();let args:Vec<_>=std::env::args_os().collect();
 check(args.len()==3 || args.len()==6,"usage: b2-native INSTANCE.bin OUTPUT [--spike-input POPULATION SCHEDULE.bin]")?;
 let input=if args.len()==6 {check(args[3]=="--spike-input","unknown native option")?;Some(SpikeInput::load(args[4].to_str().ok_or("invalid population")?.parse()?,Path::new(&args[5]))?)} else {None};
 let output=Path::new(&args[2]);check(!output.exists(),"output exists")?;
 execute(Reader::open(Path::new(&args[1]))?,output,initialization_started,input)
}
'''
RUNTIME = (RUNTIME
           .replace("__PARALLEL_MIN_TOTAL_WORK__", str(PARALLEL_MIN_TOTAL_WORK))
           .replace("__PARALLEL_TARGET_TASK_WORK__", str(PARALLEL_TARGET_TASK_WORK))
           .replace("__PARALLEL_MIN_TASK_ITEMS__", str(PARALLEL_MIN_TASK_ITEMS))
           .replace("__PARALLEL_EVENT_MIN_EDGES__", str(PARALLEL_EVENT_MIN_EDGES)))
