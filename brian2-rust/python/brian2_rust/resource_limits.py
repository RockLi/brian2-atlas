"""Explicit process-local preparation limits; these do not reserve host memory."""
import os


def _environment(name, default, maximum):
    text = os.environ.get(name, str(default))
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError(f"{name} requires decimal digits in 1..{maximum}")
    value = int(text)
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} requires decimal digits in 1..{maximum}")
    return value


def initial_value_budget():
    # Larger external workloads may opt in; the Rust validator uses the same cap.
    return _environment("B2_MAX_INITIAL_VALUES", 100_000_000, 8_000_000_000)


def explicit_synapse_budget():
    return _environment("B2_MAX_EXPLICIT_SYNAPSES", 50_000_000, 1_000_000_000)


def candidate_pair_budget():
    return _environment("B2_MAX_CANDIDATE_PAIRS", 500_000_000, 1_000_000_000)


def timed_array_value_budget():
    # Per TimedArray; the IR byte budget still gates aggregate serialized size.
    return _environment("B2_MAX_TIMED_ARRAY_VALUES", 10_000_000, 1_000_000_000)


def ir_byte_budget():
    return _environment("B2_MAX_IR_BYTES", 2048 * 2**20, 64 * 2**30)


def neuron_budget():
    return _environment("B2_MAX_NEURONS", 1_000_000, 16_000_000)


def population_step_budget():
    return _environment("B2_MAX_POPULATION_STEPS", 10_000_000, 10_000_000)
