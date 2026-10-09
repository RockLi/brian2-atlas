"""Bit-exact AOT wire encoding across chunk boundaries and storage widths."""
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust.native import write_streamed_instance  # noqa: E402


@pytest.mark.parametrize("dtype,code,samples", [
    ("f32", "f", [0.0, -0.0, 2**-149, -1.25, 3.4028234663852886e38]),
    ("f64", "d", [0.0, -0.0, 5e-324, -1.25, 1.7976931348623157e308]),
    ("i32", "i", [0, -1, -2**31, 2**31-1]),
    ("u32", "I", [0, 1, 2**32-1]),
    ("i64", "q", [0, -1, -2**63, 2**63-1]),
    ("u64", "Q", [0, 1, 2**64-1]),
    ("bool", "?", [False, True]),
])
def test_chunked_instance_values_preserve_exact_storage_bits(tmp_path, dtype, code, samples):
    values = [samples[index % len(samples)] for index in range(65539)]
    encoded = [struct.pack(">" + code, value).hex() for value in values]
    model = {
        "definition": {"populations": [{
            "count": len(values), "steps": 1, "dt": struct.pack(">d", .001).hex(),
            "states": [{"name": "v", "dtype": dtype}], "parameters": [],
        }], "synapses": []},
        "instance": {"rng_seed": 0, "populations": [{
            "initial_state": {"v": encoded}, "parameters": {}, "refractory": None,
        }], "synapses": []},
    }
    path = tmp_path / "instance.bin"
    write_streamed_instance(model, path)
    expected = (b"B2AOT001" + struct.pack("<QQdQQ", 1, 0, .001, 1, len(values))
                + b"".join(struct.pack("<" + code, value) for value in values))
    assert path.read_bytes() == expected


def test_instance_appends_runtime_clock_starts_and_final_time(tmp_path):
    bits = lambda value: struct.pack(">d", value).hex()
    model = {
        "definition": {"populations": [{
            "count": 1, "steps": 3, "dt": bits(.001),
            "states": [], "parameters": [],
        }], "synapses": []},
        "instance": {"rng_seed": 4, "populations": [{
            "initial_state": {}, "parameters": {}, "refractory": None,
        }], "synapses": []},
        "run": {
            "start": bits(.125), "duration": bits(.375),
            "clocks": [
                {"start_tick": 7, "steps": 3},
                {"start_tick": 11, "steps": 5},
            ],
        },
    }
    path = tmp_path / "instance.bin"
    write_streamed_instance(model, path)

    fixed = b"B2AOT001" + struct.pack("<QQdQQ", 1, 4, .001, 3, 1)
    assert path.read_bytes() == fixed + struct.pack("<QQd", 7, 11, .5)
