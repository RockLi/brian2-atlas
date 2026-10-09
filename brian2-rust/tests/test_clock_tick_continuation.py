"""Long fractional-second boundaries must match integer Clock ticks exactly."""
from pathlib import Path
import sys
import numpy as np
import pytest
import brian2 as b
from brian2.devices.device import all_devices

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import brian2_rust  # noqa: F401
from brian2_rust.export import lower_network


def execute(directory,backend,split):
    old=b.get_device();old_target=b.prefs.codegen.target
    all_devices["rust_standalone"].reinit()
    b.device.reinit();b.start_scope()
    try:
        if backend=='numpy':b.set_device('runtime');b.prefs.codegen.target='numpy'
        else:b.set_device('rust_standalone',runner=ROOT/'target/release/b2-runner',engine=backend,directory=directory)
        g=b.NeuronGroup(1,'dv/dt=1*Hz : 1',threshold='v>1',reset='v=0',
                        dt=.1*b.ms,method='euler',name='time_neurons')
        trace=b.StateMonitor(g,'v',record=True,name='time_trace')
        spikes=b.SpikeMonitor(g,name='time_spikes')
        net=b.Network(g,trace,spikes)
        differences=[]
        for end in ([2100,8100,10600,16600,19100] if split else [19100]):
            differences.append(float(net.t/b.second-g.clock.t/b.second))
            # Intentionally reproduce the public unit conversion that caused
            # the original 1-ULP Network/Clock mismatch at 10.6 seconds.
            net.run((end-float(net.t/b.ms))*b.ms,namespace={})
        return dict(v=np.array(g.v[:]),trace=np.array(trace.v),times=np.array(trace.t[:]),
                    spike_i=np.array(spikes.i[:]),spike_t=np.array(spikes.t[:])),differences
    finally:
        b.device.reinit();b.set_device(old);b.prefs.codegen.target=old_target;b.start_scope()


def test_fractional_second_segments_match_uninterrupted_and_numpy(tmp_path):
    expected,_=execute(tmp_path/'numpy','numpy',False)
    full,_=execute(tmp_path/'full','aot',False)
    split,differences=execute(tmp_path/'split','aot',True)
    assert any(abs(d)>1e-15 for d in differences)
    for key in expected:
        np.testing.assert_array_equal(full[key],expected[key],err_msg=key)
        np.testing.assert_array_equal(split[key],full[key],err_msg=key)


@pytest.mark.parametrize('corruption',['tick','time'])
def test_actual_clock_mismatch_is_rejected(tmp_path,corruption):
    old=b.get_device();all_devices["rust_standalone"].reinit();b.device.reinit();b.start_scope()
    try:
        b.set_device('rust_standalone',runner=ROOT/'target/release/b2-runner',engine='aot')
        g=b.NeuronGroup(1,'v : 1',dt=.1*b.ms,name='misaligned_neurons')
        if corruption=='tick':g.clock._set_t_update_dt(.1*b.ms)
        else:g.clock.variables['t'].set_value(.1e-3)
        with pytest.raises(NotImplementedError,match='all objects must be active'):
            lower_network(b.Network(g),1*b.ms)
    finally:b.device.reinit();b.set_device(old);b.start_scope()
