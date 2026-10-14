"""Validate and expose the versioned binary result dump as NumPy views."""

import json
import struct
import mmap
import os

import numpy as np


MAGIC = b"B2DMP001"
END = b"B2END001"
VERSION = 3
ENDIAN_MARKER = 0x01020304
EVENT_MAGIC = b"B2EVT001"
EVENT_END = b"B2EEND01"
COMPACT_VERSION = 4
COMPACT_EVENT_MAGIC = b"B2EVT002"

NUMPY_DTYPES = {
    "bool": "?",
    "f32": "<f4",
    "f64": "<f8",
    "i32": "<i4",
    "i64": "<i8",
    "u32": "<u4",
    "u64": "<u8",
}


def _number(encoded):
    return struct.unpack(">d", bytes.fromhex(encoded))[0]


def _numbers(encoded):
    """Decode a JSON array of f64 bit patterns without a Python scalar loop."""
    if not encoded:
        return np.empty(0, dtype=np.float64)
    return np.frombuffer(bytes.fromhex("".join(encoded)), dtype=">f8").astype("<f8")


VALIDATION_CHUNK = 131072


class _DumpReader:
    def __init__(self, path, release_file_cache=False):
        self.data = np.memmap(path, mode="r", dtype=np.uint8)
        self.at = 0
        self.cache_fd = os.open(path, os.O_RDONLY) if release_file_cache else None

    def close_cache_fd(self):
        if self.cache_fd is not None:
            os.close(self.cache_fd)
            self.cache_fd = None

    def release_range(self, start, end):
        if self.cache_fd is not None:
            page = mmap.PAGESIZE
            start = start // page * page
            end = end // page * page
            if end > start:
                self.data._mmap.madvise(mmap.MADV_DONTNEED, start, end - start)
                os.posix_fadvise(self.cache_fd, start, end - start, os.POSIX_FADV_DONTNEED)

    def release(self, values):
        if self.cache_fd is not None and len(values):
            start = values.ctypes.data - self.data.ctypes.data
            end = start + (len(values) - 1) * values.strides[0] + values.dtype.itemsize
            self.release_range(start, end)

    def raw(self, size):
        if size < 0 or self.at + size > self.data.size:
            raise ValueError("truncated result dump")
        result = self.data[self.at:self.at + size]
        self.at += size
        return result

    def scalar(self, dtype):
        return self.array(dtype, 1)[0]

    def array(self, dtype, count):
        dtype = np.dtype(dtype)
        raw = self.raw(int(count) * dtype.itemsize)
        return np.ndarray((int(count),), dtype=dtype, buffer=raw, offset=0)

    def u64(self):
        return int(self.scalar("<u8"))

    def expect(self, value):
        if bytes(self.raw(len(value))) != value:
            raise ValueError("invalid result dump marker")


def _all_chunks(values, predicate, release):
    for start in range(0, len(values), VALIDATION_CHUNK):
        chunk = values[start:start + VALIDATION_CHUNK]
        valid = bool(np.all(predicate(chunk)))
        release(chunk)
        if not valid:
            return False
    return True


def _event_range(ticks, indices, lower, upper, count, release):
    for start in range(0, len(ticks), VALIDATION_CHUNK):
        t, i = ticks[start:start + VALIDATION_CHUNK], indices[start:start + VALIDATION_CHUNK]
        valid = np.all(t >= lower) and np.all(t < upper) and np.all(i >= 0) and np.all(i < count)
        release(t); release(i)
        if not valid:
            return False
    return True


def _event_counts(indices, count, release):
    result = np.zeros(count, dtype=np.int64)
    for start in range(0, len(indices), VALIDATION_CHUNK):
        chunk = indices[start:start + VALIDATION_CHUNK]
        result += np.bincount(chunk, minlength=count)
        release(chunk)
    return result


def _last_events(ticks, indices, final_tick):
    # Strict time order has already been checked. Only the final equal-tick
    # suffix is inspected; it has at most one event per neuron.
    end = len(ticks)
    start = end
    while start:
        left = max(0, start - VALIDATION_CHUNK)
        different = np.flatnonzero(ticks[left:start] != final_tick)
        if len(different):
            start = left + int(different[-1]) + 1
            break
        start = left
    return indices[start:end]


def _refractory_events(ticks, indices, initial, dt, period_ticks, full_window, release):
    """Validate consecutive events with O(neurons + fixed chunk) scratch space.

    Stable grouping happens inside each chunk only. The last tick per neuron
    carries exact integer intervals across chunk boundaries; first events keep
    the original floating-point initial-lastspike rule.
    """
    seen = np.zeros(len(initial), dtype=np.bool_)
    last_tick = np.zeros(len(initial), dtype=np.int64)
    for start in range(0, len(ticks), VALIDATION_CHUNK):
        i = indices[start:start + VALIDATION_CHUNK]
        t = ticks[start:start + VALIDATION_CHUNK]
        order = np.argsort(i, kind="stable")
        neurons, times = i[order], t[order]
        first = np.empty(len(i), dtype=np.bool_)
        first[0] = True
        first[1:] = neurons[1:] != neurons[:-1]
        first_neurons, first_ticks = neurons[first], times[first]
        previously_seen = seen[first_neurons]
        if period_ticks is not None:
            same = neurons[1:] == neurons[:-1]
            valid = np.all(np.diff(times)[same] >= period_ticks)
            valid = valid and np.all(first_ticks[previously_seen] -
                last_tick[first_neurons[previously_seen]] >= period_ticks)
            if full_window:
                new_neurons = first_neurons[~previously_seen]
                elapsed = (first_ticks[~previously_seen] * dt - initial[new_neurons] + 1e-3 * dt) / dt
                valid = valid and np.all(np.trunc(elapsed) >= period_ticks)
            if not valid:
                raise ValueError("spike during refractory period")
        final = np.empty(len(i), dtype=np.bool_)
        final[-1] = True
        final[:-1] = neurons[:-1] != neurons[1:]
        last_tick[neurons[final]] = times[final]
        seen[first_neurons] = True
        release(t); release(i)
    neurons = np.flatnonzero(seen)
    return neurons, last_tick[neurons] * dt


def _strict_event_order(ticks, indices, release=lambda _: None):
    """Lexicographic ordering without packed-key overflow or unsigned subtraction."""
    for start in range(1, len(ticks), VALIDATION_CHUNK):
        end = min(start + VALIDATION_CHUNK, len(ticks))
        left_t, right_t = ticks[start-1:end-1], ticks[start:end]
        left_i, right_i = indices[start-1:end-1], indices[start:end]
        valid = np.all((right_t > left_t) | ((right_t == left_t) & (right_i > left_i)))
        release(right_t); release(right_i)
        if not valid:
            return False
    return True


def load_results(model, directory, *, include_times=True, release_file_cache=False):
    """Return validated, memmap-backed arrays from ``results.bin``.

    Set ``include_times=False`` for tick-based analysis to omit derived
    ``times`` and ``spike_times`` arrays, including event streams/monitors.
    Integer event ticks, traces, state and all validation remain unchanged.
    The default preserves the device-facing result API.
    Validation scratch arrays are bounded by neuron count plus fixed chunks.
    On supported POSIX systems, ``release_file_cache=True`` releases checked
    mmap/file pages; returned views remain valid and can fault pages back in.
    It does not bound caller allocations (including default derived times).
    Version 4 returns zero-copy u32 spike tick/index views; version 3 retains
    its i64 views. Other state/sample dtypes are unchanged.
    """
    if type(include_times) is not bool:
        raise TypeError('include_times must be a bool')
    if type(release_file_cache) is not bool:
        raise TypeError('release_file_cache must be a bool')
    if release_file_cache and not (hasattr(mmap.mmap, 'madvise') and
            hasattr(mmap, 'MADV_DONTNEED') and hasattr(os, 'posix_fadvise') and
            hasattr(os, 'POSIX_FADV_DONTNEED')):
        raise RuntimeError('file-cache release is unsupported on this platform')
    reader = event_reader = None
    definition, instance = model["definition"], model["instance"]
    run_start = _number(model["run"].get("start", "0000000000000000"))

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        summary = json.loads((directory / "summary.json").read_text())
        reader = _DumpReader(directory / "results.bin", release_file_cache)
        reader.expect(MAGIC)
        version = int(reader.scalar("<u4"))
        check(version in (VERSION, COMPACT_VERSION), "result dump version")
        compact = version == COMPACT_VERSION
        event_dtype = np.dtype([("tick", "<u4" if compact else "<i8"),
                                ("index", "<u4" if compact else "<i8")])
        check(int(reader.scalar("<u4")) == ENDIAN_MARKER, "result dump byte order")
        population_count = reader.u64()
        neuron_count = reader.u64()
        declared_size = reader.u64()
        population_defs = definition["populations"]
        check(population_count == len(population_defs), "population count")
        check(neuron_count == instance["neuron_count"], "neuron count")
        check(declared_size == reader.data.size, "result dump byte length")
        check(isinstance(summary, dict), "metadata object")
        check(summary.get("schema") == f"b2-result-dump-v{version}", "metadata schema")
        check(summary.get("dump_bytes") == declared_size, "metadata dump length")
        check(summary.get("population_count") == population_count,
              "metadata population count")
        check(summary.get("neuron_count") == neuron_count, "metadata neuron count")

        populations = []
        total_spikes = 0
        for pop_def, pop_inst in zip(
                population_defs, instance["populations"], strict=True):
            population_start = reader.at
            count, steps = pop_def["count"], pop_def["steps"]
            dt = _number(pop_def["dt"])
            start_tick = int(round(run_start / dt))
            end_tick = start_tick + steps
            if compact:
                check(0 <= start_tick < 2**32 and end_tick <= 2**32 and
                      0 <= count <= 2**32 and
                      pop_def.get('events', []) in ([], ['spike']) and
                      not pop_def.get('event_monitors'), 'compact spike domain')
            monitor = pop_def["monitor"]
            variables, record = monitor["variables"], monitor["record"]
            event_window_steps = monitor["window_steps"]
            event_start_tick = end_tick - event_window_steps
            if pop_def["state_monitors"]:
                state_clock = pop_def["state_monitors"][0]["clock"]
                state_run = model["run"]["clocks"][state_clock]
                state_steps = state_run["steps"]
                state_start_tick = state_run["start_tick"]
                # Bounded recording is currently accepted only when the
                # StateMonitor shares the population clock.  The writer
                # therefore emits the final ``window_steps`` samples, not
                # every activation of that clock.  Keep the reader's byte
                # count and derived absolute times on the same suffix.
                if event_window_steps < steps:
                    state_steps = event_window_steps
                    state_start_tick += state_run["steps"] - state_steps
                state_dt = _number(definition["clocks"][state_clock]["dt"])
            else:
                state_steps = event_window_steps
                state_start_tick = end_tick - state_steps
                state_dt = dt
            state_symbols = pop_def["states"]
            state_names = [state["name"] for state in state_symbols]
            fields = [reader.u64() for _ in range(8)]
            (dump_count, dump_steps, record_count, variable_count, state_count,
             spike_count, last_count, flags) = fields
            check((dump_count, dump_steps, record_count, variable_count, state_count) ==
                  (count, steps, len(record), len(variables), len(state_names)),
                  f"{pop_def['name']} dimensions")
            check(flags in (0, 1) and
                  bool(flags) == (pop_inst.get("refractory") is not None),
                  f"{pop_def['name']} refractory flag")

            symbols = {
                symbol["name"]: symbol
                for symbol in (pop_def["states"] + pop_def["parameters"] +
                               pop_def.get("linked_variables", []))
            }
            trace = {}
            for name in variables:
                dtype = NUMPY_DTYPES[symbols[name]["dtype"]]
                values = reader.array(dtype, state_steps * len(record))
                check(_all_chunks(values, np.isfinite, reader.release),
                      f"{pop_def['name']} trace values")
                trace[name] = values.reshape(state_steps, len(record))
            events = reader.array(event_dtype, spike_count)
            event_ticks, indices_i64 = events["tick"], events["index"]
            check(_event_range(event_ticks, indices_i64, event_start_tick, end_tick, count, reader.release),
                  f"{pop_def['name']} spike range")
            check(_strict_event_order(event_ticks, indices_i64, reader.release),
                  f"{pop_def['name']} spike ordering")
            counts_i64 = reader.array("<i8", count)
            expected_counts = _event_counts(indices_i64, count, reader.release)
            check(np.array_equal(counts_i64, expected_counts),
                  f"{pop_def['name']} spike counts")
            last_i64 = reader.array("<i8", last_count)
            check(np.all(last_i64 >= 0) and np.all(last_i64 < count) and
                  (last_count < 2 or np.all(np.diff(last_i64) > 0)),
                  f"{pop_def['name']} last spike range")
            if pop_def.get("spike_monitor") is not None:
                expected_last = _last_events(event_ticks, indices_i64, end_tick - 1)
                check(np.array_equal(last_i64, expected_last),
                      f"{pop_def['name']} last spikes")
            else:
                # The final event stream is runtime state used to restore
                # ``_spikespace`` across segmented runs.  It exists even when
                # no SpikeMonitor requested a persisted event history.
                expected_last = last_i64

            states = {}
            for state in state_symbols:
                name = state["name"]
                dtype = NUMPY_DTYPES[state["dtype"]]
                values = reader.array(dtype, count)
                check(_all_chunks(values, np.isfinite, reader.release), f"{pop_def['name']} final state")
                states[name] = values

            refractory = None
            ref_instance = pop_inst.get("refractory")
            if ref_instance is not None:
                last = reader.array("<f8", count)
                available_u8 = reader.array("u1", count)
                check(np.isfinite(last).all() and np.all(available_u8 <= 1),
                      f"{pop_def['name']} refractory values")
                fixed_refractory = pop_def["refractory"].get("mode") == "fixed"
                expected_lastspike = _numbers(ref_instance["initial_lastspike"])
                if spike_count:
                    try:
                        last_neurons, last_times = _refractory_events(
                            event_ticks, indices_i64, expected_lastspike, dt,
                            ref_instance["period_ticks"] if fixed_refractory else None,
                            event_window_steps == steps, reader.release)
                    except ValueError as error:
                        raise ValueError(f"{pop_def['name']} {error}") from error
                    if event_window_steps == steps:
                        expected_lastspike[last_neurons] = last_times
                    else:
                        check(np.array_equal(last[last_neurons], last_times),
                              f"{pop_def['name']} bounded lastspike mismatch")
                if (not fixed_refractory and event_window_steps == steps and
                        pop_def.get("spike_monitor") is not None):
                    check(np.array_equal(last, expected_lastspike),
                          f"{pop_def['name']} expression refractory lastspike")
                elif fixed_refractory and event_window_steps == steps and \
                        pop_def.get("spike_monitor") is not None:
                    expected_available = np.trunc(
                        ((end_tick - 1) * dt - expected_lastspike + 1e-3 * dt) / dt
                    ) >= ref_instance["period_ticks"]
                    expected_available[expected_last] = False
                    check(np.array_equal(last, expected_lastspike) and
                          np.array_equal(available_u8, expected_available),
                          f"{pop_def['name']} refractory state")
                elif fixed_refractory:
                    expected_available = np.trunc(
                        ((end_tick - 1) * dt - last + 1e-3 * dt) / dt
                    ) >= ref_instance["period_ticks"]
                    expected_available[expected_last] = False
                    check((not len(expected_last) or np.all(
                              last[expected_last] == (end_tick - 1) * dt)) and
                          np.all(last <= (end_tick - 1) * dt + 1e-3 * dt) and
                          np.array_equal(available_u8, expected_available),
                          f"{pop_def['name']} bounded refractory state")
                refractory = {"lastspike": last,
                              "not_refractory": available_u8.view(np.bool_)}

            populations.append({
                "states": states,
                **({"times": (state_start_tick + np.arange(state_steps)) * state_dt}
                   if include_times else {}),
                "trace": trace,
                **({"spike_times": event_ticks * dt} if include_times else {}),
                "spike_ticks": event_ticks,
                "indices": indices_i64,
                "counts": counts_i64,
                "last_spikes": last_i64,
                "event_monitors": {},
                "event_streams": ({"spike": {
                    "ticks": event_ticks, "indices": indices_i64,
                    **({"times": event_ticks * dt} if include_times else {}),
                }} if "spike" in pop_def.get("events", []) else {}),
                "refractory": refractory,
            })
            total_spikes += spike_count
            reader.release_range(population_start, reader.at)

        synapse_defs = definition.get("synapses", [])
        synapse_instances = instance.get("synapses", [])
        check(reader.u64() == len(synapse_defs), "synapse object count")
        synapse_results = []
        for synapse_def, synapse_instance in zip(
                synapse_defs, synapse_instances, strict=True):
            state_symbols = synapse_def["states"]
            state_names = [state["name"] for state in state_symbols]
            topology = synapse_instance.get("topology", {"kind": "explicit"})
            edge_count = (len(synapse_instance["source"])
                          if topology["kind"] == "explicit"
                          else topology["edge_count"])
            check(reader.u64() == len(state_names),
                  f"{synapse_def['name']} state count")
            check(reader.u64() == edge_count,
                  f"{synapse_def['name']} edge count")
            states = {}
            for state in state_symbols:
                name = state["name"]
                dtype = NUMPY_DTYPES[state["dtype"]]
                values = reader.array(dtype, edge_count)
                check(_all_chunks(values, np.isfinite, reader.release),
                      f"{synapse_def['name']} final state")
                states[name] = values
            monitor_results = {}
            for monitor in synapse_def.get("state_monitors", []):
                steps = model["run"]["clocks"][monitor["clock"]]["steps"]
                record_count = len(monitor["record"])
                trace = {}
                for name, source in zip(
                        monitor["variables"], monitor["sources"], strict=True):
                    values = reader.array(
                        NUMPY_DTYPES[source["dtype"]], steps * record_count)
                    check(_all_chunks(values, np.isfinite, reader.release),
                          f"{monitor['name']} synapse trace")
                    trace[name] = values.reshape(steps, record_count)
                dt = _number(definition["clocks"][monitor["clock"]]["dt"])
                start_tick = model["run"]["clocks"][monitor["clock"]]["start_tick"]
                monitor_results[monitor["name"]] = {
                    "times": (start_tick + np.arange(steps)) * dt,
                    "trace": trace,
                }
            synapse_results.append({
                "name": synapse_def["name"], "states": states,
                "state_monitors": monitor_results, "events": reader.u64()})
        synaptic_events = sum(result["events"] for result in synapse_results)
        final_time = float(reader.scalar("<f8"))
        reader.expect(END)
        check(reader.at == reader.data.size, "trailing result dump data")
        expected_final_time = run_start + _number(model["run"]["duration"])
        check(np.isfinite(final_time) and final_time == expected_final_time,
              "final time")
        check(summary.get("spike_count") == total_spikes, "metadata spike count")
        check(summary.get("synaptic_events") == synaptic_events,
              "metadata synaptic events")
        check(summary.get("final_time_seconds") == final_time,
              "metadata final time")
        event_monitor_defs = [
            (population_index, monitor)
            for population_index, population in enumerate(population_defs)
            for monitor in population.get("event_monitors", [])]
        event_stream_defs = [
            (population_index, event)
            for population_index, population in enumerate(population_defs)
            for event in population.get("events", [])]
        event_dump = None
        event_dump_bytes = summary.get("event_dump_bytes", 0)
        if event_dump_bytes:
            event_reader = _DumpReader(directory / "events.bin", release_file_cache)
            event_dump = event_reader.data
            event_reader.expect(COMPACT_EVENT_MAGIC if compact else EVENT_MAGIC)
            check(event_reader.u64() == len(event_stream_defs),
                  "custom event stream count")
            for population_index, event_name in event_stream_defs:
                event_count = event_reader.u64()
                events = event_reader.array(event_dtype, event_count)
                ticks, indices = events["tick"], events["index"]
                population = population_defs[population_index]
                clock_index = population["clock"]
                dt = _number(definition["clocks"][clock_index]["dt"])
                run_clock = model["run"]["clocks"][clock_index]
                first_tick = run_clock["start_tick"]
                end_tick = first_tick + run_clock["steps"]
                check(_event_range(ticks, indices, first_tick, end_tick,
                                   population["count"], event_reader.release),
                      f"{population['name']}.{event_name} event range")
                populations[population_index]["event_streams"][event_name] = {
                    "ticks": ticks, "indices": indices,
                    **({"times": ticks * dt} if include_times else {}),
                }
            check(event_reader.u64() == len(event_monitor_defs),
                  "event monitor count")
            for population_index, monitor in event_monitor_defs:
                event_count = event_reader.u64()
                variable_count = event_reader.u64()
                check(variable_count == len(monitor["variables"]),
                      f"{monitor['name']} variable count")
                events = event_reader.array(event_dtype, event_count)
                ticks, indices = events["tick"], events["index"]
                population = population_defs[population_index]
                clock = definition["clocks"][monitor["clock"]]
                dt = _number(clock["dt"])
                run_clock = model["run"]["clocks"][monitor["clock"]]
                first_tick = run_clock["start_tick"]
                end_tick = first_tick + run_clock["steps"]
                check(_event_range(ticks, indices, first_tick, end_tick,
                                   population["count"], event_reader.release),
                      f"{monitor['name']} event range")
                symbols = {
                    symbol["name"]: symbol
                    for symbol in (population["states"] + population["parameters"] +
                                   population.get("linked_variables", []))
                }
                values = {}
                for name in monitor["variables"]:
                    dtype = NUMPY_DTYPES[symbols[name]["dtype"]]
                    variable = event_reader.array(dtype, event_count)
                    check(_all_chunks(variable, np.isfinite, event_reader.release),
                          f"{monitor['name']} event values")
                    values[name] = variable
                populations[population_index]["event_monitors"][monitor["name"]] = {
                    "indices": indices, "ticks": ticks,
                    **({"times": ticks * dt} if include_times else {}),
                    "values": values,
                }
            event_reader.expect(EVENT_END)
            check(event_reader.at == event_reader.data.size,
                  "trailing event monitor data")
            check(event_dump_bytes == event_reader.data.size,
                  "metadata event dump length")
        else:
            check(not event_monitor_defs,
                  "missing event monitor dump")
        reader.release_range(0, reader.at)
        if event_reader is not None:
            event_reader.release_range(0, event_reader.at)
        return {
            "populations": populations,
            "synapses": synapse_results,
            "synaptic_states": (synapse_results[0]["states"]
                                 if len(synapse_results) == 1 else {}),
            "synaptic_events": synaptic_events,
            "metadata": summary,
            "_dump": reader.data,
            "_event_dump": event_dump,
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Rust runner returned inconsistent results: {error}") from error
    finally:
        for dump_reader in (reader, event_reader):
            if dump_reader is not None:
                dump_reader.close_cache_fd()
