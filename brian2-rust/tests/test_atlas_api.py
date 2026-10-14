"""Public Atlas naming must preserve Device state and legacy behavior."""
import os
import pickle
import subprocess
import sys
from pathlib import Path

import brian2 as b
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import brian2_atlas as atlas
import brian2_rust as legacy
from brian2.devices.device import all_devices


@pytest.mark.parametrize('first,second', [('brian2_atlas', 'brian2_rust'),
                                          ('brian2_rust', 'brian2_atlas')])
def test_fresh_import_order_and_serialized_legacy_class(first, second):
    code = f'''
import importlib,pickle
importlib.import_module({first!r})
importlib.import_module({second!r})
import brian2_atlas as atlas
import brian2_rust as legacy
from brian2_atlas.device import AtlasDevice
from brian2.devices.device import all_devices
assert AtlasDevice is atlas.AtlasDevice is legacy.RustStandaloneDevice
assert all_devices['atlas'] is all_devices['rust_standalone']
assert pickle.loads(b'cbrian2_rust.device\\nRustStandaloneDevice\\n.') is AtlasDevice
for name in legacy.__all__:
    assert getattr(atlas,name) is getattr(legacy,name),name
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True,
                   text=True, timeout=90)


@pytest.mark.parametrize('name', ['atlas', 'rust_standalone'])
@pytest.mark.parametrize('engine', ['reference', 'aot'])
def test_both_names_run_the_same_native_model(name, engine, tmp_path):
    previous = b.get_device()
    device = all_devices['atlas']
    try:
        device.reinit()
        b.start_scope()
        b.set_device(name, engine=engine, directory=tmp_path / 'project',
                     runner=Path(os.environ['B2_RUNNER']))
        assert b.get_device() is device is all_devices['rust_standalone']
        assert type(device) is atlas.AtlasDevice is legacy.RustStandaloneDevice
        group = b.NeuronGroup(1, 'dv/dt=(1.5-v)/(10*ms) : 1', threshold='v>1',
                              reset='v=0', method='euler', dt=0.1*b.ms)
        spikes = b.SpikeMonitor(group)
        network = b.Network(group, spikes)
        network.run(100*b.ms)
        np.testing.assert_allclose(spikes.t[:]/b.ms, np.arange(10.9, 100, 11),
                                   rtol=0, atol=1e-12)
        assert np.isclose(group.v[0], 0.14342688748679328)
        assert float(network.t/b.ms) == 100
    finally:
        b.set_device(previous)
        device.reinit()
        b.start_scope()
