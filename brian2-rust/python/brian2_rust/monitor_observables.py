"""Reconstruct Brian2 read-only StateMonitor observables from AtlasIR samples.

The runtime records physical state dependencies at the monitor's start slot.
This adapter evaluates deterministic subexpressions in SI units and derives
refractory fields from the exact event history. It never changes state updates.
"""

import ast

import numpy as np


_DTYPES = {"f64": ">f8", "f32": ">f4", "i64": ">i8", "u64": ">u8",
           "i32": ">i4", "u32": ">u4", "bool": "?"}
_FUNCTIONS = {
    "abs": np.abs, "exp": np.exp, "sqrt": np.sqrt,
    "log": np.log, "log10": np.log10, "sin": np.sin,
    "cos": np.cos, "tan": np.tan, "sinh": np.sinh,
    "cosh": np.cosh, "tanh": np.tanh, "clip": np.clip,
    "int": lambda value: np.asarray(value).astype(np.int64),
}
_ALLOWED = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
    ast.Call, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
}


def _decode(values, dtype):
    storage = np.dtype(_DTYPES[dtype])
    return np.frombuffer(b"".join(bytes.fromhex(value) for value in values),
                         dtype=storage)


def _safe_expression(code, namespace, functions=None):
    functions = {} if functions is None else functions
    allowed_functions = {**_FUNCTIONS, **functions}
    tree = ast.parse(code, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED:
            raise NotImplementedError(
                f"StateMonitor subexpression contains unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name)
                or node.func.id not in allowed_functions):
            raise NotImplementedError(
                "StateMonitor subexpression needs a supported pure math function")
    return eval(compile(tree, "<b2ir-monitor-expression>", "eval"),
                {"__builtins__": {}}, {**allowed_functions, **namespace})


def _timed_array(config):
    values = _decode(config["values"], "f64")
    rows, columns = config["rows"], config["columns"]
    values = values.reshape(rows, columns or 1)
    epsilon = _decode([config["epsilon"]], "f64")[0]
    upsampling = config["upsampling"]

    def evaluate(time, index=None):
        scaled = (np.asarray(time) / epsilon + 0.5) / upsampling
        row = np.clip(scaled.astype(np.int64), 0, rows - 1)
        if columns is None:
            return values[:, 0][row]
        if index is None:
            raise RuntimeError("two-dimensional TimedArray needs an index")
        column = np.asarray(index, dtype=np.int64)
        if np.any(column < 0) or np.any(column >= columns):
            raise IndexError("TimedArray index outside its column range")
        return values[row, column]

    return evaluate


def _refractory_fields(population, instance, result, record, steps, dt, start):
    refractory = instance["refractory"]
    if refractory is None:
        raise RuntimeError("refractory monitor field without refractory state")
    ticks = np.rint((np.asarray(result["spike_times"]) - start) / dt).astype(np.int64)
    indices = np.asarray(result["indices"])
    if len(ticks) != len(indices):
        raise RuntimeError("spike index/time length mismatch")
    sampled_ticks = np.arange(steps, dtype=np.int64)
    last = np.empty((steps, len(record)), dtype=np.float64)
    available = np.empty((steps, len(record)), dtype=np.bool_)
    initial_last = _decode(refractory["initial_lastspike"], "f64")
    initial_available = np.asarray(refractory["initial_not_refractory"], dtype=np.bool_)
    period_ticks = int(refractory["period_ticks"])
    for column, neuron in enumerate(record):
        previous = np.sort(ticks[indices == neuron])
        locations = np.searchsorted(previous, sampled_ticks, side="left") - 1
        had_spike = locations >= 0
        prior_tick = previous[np.maximum(locations, 0)] if len(previous) else np.zeros(steps)
        last[:, column] = np.where(
            had_spike, start + prior_tick * dt, initial_last[neuron])
        available[:, column] = np.where(
            had_spike, sampled_ticks - prior_tick > period_ticks,
            initial_available[neuron])
    return {"lastspike": last, "not_refractory": available}


def reconstruct(population, instance, monitor, result, output_name, columns,
                state_monitor, start):
    """Return time × recorded-neuron values for one public monitor variable."""
    physical = {
        name: result["trace"][name][:, columns]
        for name in population["monitor"]["variables"]
    }
    if output_name in physical:
        return physical[output_name]
    steps = len(result["times"])
    dt = float(state_monitor.clock.dt)
    record = monitor["record"]
    refractory = None

    symbols = {symbol["name"]: symbol["dtype"]
               for symbol in population["parameters"]}
    for name, dtype in symbols.items():
        if name in physical:
            continue
        values = _decode(instance["parameters"][name], dtype)
        if len(values) == 1:
            physical[name] = values[0]
        elif len(values) == population["count"]:
            physical[name] = values[np.asarray(record)][None, :]
        else:
            raise RuntimeError(f"monitor parameter {name} has invalid shape")

    expressions = population.get("monitor_expressions", {})
    cache = dict(physical)
    cache.update({
        "t": start + np.arange(steps, dtype=np.float64)[:, None] * dt,
        "dt": dt,
        "i": np.asarray(record, dtype=np.int64)[None, :],
        "N": population["count"],
    })
    functions = {
        name: _timed_array(config)
        for name, config in population.get("monitor_timed_arrays", {}).items()
    }

    def value(name):
        nonlocal refractory
        if name in cache:
            return cache[name]
        if name in {"lastspike", "not_refractory"}:
            if refractory is None:
                refractory = _refractory_fields(
                    population, instance, result, record, steps, dt, start)
            cache.update(refractory)
            return cache[name]
        code = expressions.get(name)
        if code is None:
            raise RuntimeError(f"monitor expression symbol {name} missing")
        identifiers = {node.id for node in ast.walk(ast.parse(code, mode="eval"))
                       if isinstance(node, ast.Name)} - set(_FUNCTIONS) - set(functions)
        namespace = {identifier: value(identifier) for identifier in identifiers}
        result_value = _safe_expression(code, namespace, functions)
        cache[name] = np.broadcast_to(result_value, (steps, len(record)))
        return cache[name]

    return value(output_name)


def reconstruct_synapse(synapse, instance, monitor, result, output_name, dt):
    """Return time × recorded-edge values for a synaptic monitor output."""
    if output_name in result["trace"]:
        return result["trace"][output_name]
    steps = len(result["times"])
    record = np.asarray(monitor["record"], dtype=np.int64)
    cache = dict(result["trace"])
    for symbol in synapse["parameters"]:
        values = _decode(instance["parameters"][symbol["name"]], symbol["dtype"])
        if len(values) == 1:
            cache[symbol["name"]] = values[0]
        elif len(values):
            cache[symbol["name"]] = values[record][None, :]
    cache.update({
        "t": result["times"][:, None],
        "dt": dt,
        "N": instance.get("topology", {}).get(
            "edge_count", len(instance.get("source", []))),
    })
    expressions = synapse.get("monitor_expressions", {})

    def value(name):
        if name in cache:
            return cache[name]
        code = expressions.get(name)
        if code is None:
            raise RuntimeError(f"synapse monitor expression symbol {name} missing")
        identifiers = {node.id for node in ast.walk(ast.parse(code, mode="eval"))
                       if isinstance(node, ast.Name)} - set(_FUNCTIONS)
        result_value = _safe_expression(
            code, {identifier: value(identifier) for identifier in identifiers})
        cache[name] = np.broadcast_to(result_value, (steps, len(record)))
        return cache[name]

    return value(output_name)
