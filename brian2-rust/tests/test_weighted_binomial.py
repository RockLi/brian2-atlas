"""Sparse weighted counter draws preserve the scalar sampler's exact bits."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust.native import RUNTIME, WEIGHTED_BINOMIAL_RUNTIME  # noqa: E402


def test_weighted_binomial_preserves_draw_bits_and_probability_checks(tmp_path):
    structures = RUNTIME[RUNTIME.index('#[derive(Default)]\nstruct NormalCache'):
                         RUNTIME.index('fn parse_cpu_list')]
    functions = RUNTIME[RUNTIME.index('fn mix64('):
                        RUNTIME.index('fn log_gamma_positive(')]
    source = structures + functions + WEIGHTED_BINOMIAL_RUNTIME + r'''
fn main() {
    let mut expected_cache=NormalCache::default();
    let mut actual_cache=NormalCache::default();
    let cases=[(0,0.0),(0,0.3),(1,0.0),(1,1.0),(1000,0.00045),
               (1000,0.0008),(1000,0.99955),(100,0.1),(100,0.9),
               (100,0.05),(100,0.050000001),(2,0.5),(11,0.5)];
    let weights=[0.0,-0.0,1.0,-1.0,f64::from_bits(1),f64::MAX];
    for tick in 0..8u64 {
        for &(n,p) in &cases {
            for approximate in [false,true,false] {
                for index in 0..512u64 {
                    let weight=weights[index as usize%weights.len()];
                    let scalar=counter_binomial(19,3,tick,index,n,p,approximate);
                    let expected=counter_binomial_cached(19,3,tick,index,n,p,approximate,&mut expected_cache)*weight;
                    let actual=counter_binomial_weighted(19,3,tick,index,n,p,approximate,&mut actual_cache,weight);
                    assert_eq!(expected.to_bits(),(scalar*weight).to_bits());
                    assert_eq!(actual.to_bits(),expected.to_bits(),"n={n},p={p},i={index},w={weight}");
                    if index%7==0 {
                        assert_eq!(counter_normal_cached(5,8,tick,index,&mut actual_cache).to_bits(),
                                   counter_normal_cached(5,8,tick,index,&mut expected_cache).to_bits());
                    }
                }
            }
        }
    }
    // Explicitly locate a negative Gaussian draw instead of depending on
    // the interleaved weight schedule to cover this rare tail case.
    let index=(0..100_000u64).find(|&i| counter_binomial(19,3,0,i,11,0.5,true)<0.0)
        .expect("deterministic negative Gaussian sample");
    for weight in [0.0,-0.0] {
        let mut cache=NormalCache::default();
        let expected=counter_binomial(19,3,0,index,11,0.5,true)*weight;
        let actual=counter_binomial_weighted(19,3,0,index,11,0.5,true,&mut cache,weight);
        assert_eq!(actual.to_bits(),expected.to_bits());
        assert_ne!(actual.to_bits(),weight.to_bits());
    }
    for p in [f64::NAN,f64::INFINITY,-0.1,1.1] {
        for weight in [0.0,-0.0,1.0] {
            let rejected=std::panic::catch_unwind(|| {
                let mut cache=NormalCache::default();
                counter_binomial_weighted(0,0,0,0,1,p,true,&mut cache,weight)
            });
            assert!(rejected.is_err());
        }
    }
}
'''
    path = tmp_path / 'weighted.rs'
    path.write_text(source)
    binary = tmp_path / 'weighted'
    subprocess.run(['rustc', '--edition=2021', '-O', str(path), '-o', str(binary)],
                   check=True, capture_output=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('rate_hz,sparse', [(8, True), (1000, True), (8, False)])
def test_weighted_timed_input_matches_reference_and_threads(tmp_path, rate_hz, sparse, dtype):
    import brian2 as b
    import brian2_rust  # noqa: F401
    from brian2.devices.device import all_devices

    previous = b.get_device()
    device = all_devices['rust_standalone']
    snapshots, dumps = [], []
    try:
        for engine, threads in [('reference', 1), ('aot', 1), ('aot', 4)]:
            device.reinit()
            b.start_scope()
            b.set_device('rust_standalone', runner=ROOT/'target/release/b2-runner',
                         directory=tmp_path/f'{engine}-{threads}', engine=engine,
                         threads=threads)
            b.seed(312907)
            size = 4096
            values = np.resize([0.0, -0.0, .5, -1.0] if sparse else [.5, 1.0],
                               (3, size))
            weights = b.TimedArray(values, dt=.2*b.ms)
            group = b.NeuronGroup(size, 'dv/dt=0/second : 1', method='euler',
                                 dt=.1*b.ms, name='weighted_target', dtype=dtype)
            group.v = np.resize([-0.0, 0.0, 1.0], size)
            drive = b.PoissonInput(group, 'v', N=1000, rate=rate_hz*b.Hz,
                                   weight='weights(t, i)')
            monitor = b.StateMonitor(group, 'v', record=[0, 1, size-1])
            b.Network(group, drive, monitor).run(.6*b.ms)
            bit_dtype = np.uint32 if dtype == np.float32 else np.uint64
            snapshots.append((np.asarray(group.v[:]).view(bit_dtype).copy(),
                              np.asarray(monitor.v).view(bit_dtype).copy()))
            if engine == 'aot':
                directory = device.last_run_directory
                source = (directory/'native/main.rs').read_text()
                assert ('fn counter_binomial_weighted(' in source) == sparse
                summary = json.loads((directory/'rust/summary.json').read_text())
                assert summary['parallel_poisson_input'] == (threads > 1)
                dumps.append((directory/'rust/results.bin').read_bytes())
        for snapshot in snapshots[1:]:
            for actual, expected in zip(snapshot, snapshots[0], strict=True):
                np.testing.assert_array_equal(actual, expected)
        assert dumps[0] == dumps[1]
    finally:
        device.reinit()
        b.set_device(previous)
        b.start_scope()


@pytest.mark.parametrize('rate_hz,pattern', [(8, 'positive'), (1000, 'positive'),
                                           (8, 'signed'), (8, 'reversed')])
def test_scalar_gated_input_matches_reference(tmp_path, rate_hz, pattern):
    import brian2 as b
    import brian2_rust  # noqa: F401
    from brian2.devices.device import all_devices

    previous = b.get_device()
    device = all_devices['rust_standalone']
    snapshots, dumps = [], []
    try:
        for engine, threads in [('reference', 1), ('aot', 1), ('aot', 4)]:
            device.reinit()
            b.start_scope()
            b.set_device('rust_standalone', runner=ROOT/'target/release/b2-runner',
                         directory=tmp_path/f'{engine}-{threads}', engine=engine,
                         threads=threads)
            b.seed(312907)
            size = 4096
            values = np.resize([0.0, 1.0] if pattern != 'signed' else [-0.0, -1.0],
                               (3, size))
            weights = b.TimedArray(values, dt=.2*b.ms)
            gate = b.TimedArray([0.0, -0.0, 1.0, -1.0], dt=.2*b.ms)
            group = b.NeuronGroup(size, 'dv/dt=0/second : 1', method='euler',
                                 dt=.1*b.ms, name='gated_target')
            group.v = np.resize([-0.0, 0.0, 1.0], size)
            index = '4095-i' if pattern == 'reversed' else 'i'
            drive = b.PoissonInput(group, 'v', N=1000, rate=rate_hz*b.Hz,
                                   weight=f'gate(t)*weights(t, {index})')
            monitor = b.StateMonitor(group, 'v', record=[0, 1, size-1])
            b.Network(group, drive, monitor).run(.8*b.ms)
            snapshots.append((np.asarray(group.v[:]).view(np.uint64).copy(),
                              np.asarray(monitor.v).view(np.uint64).copy()))
            if engine == 'aot':
                directory = device.last_run_directory
                source = (directory/'native/main.rs').read_text()
                assert ('for value in p0_state_0.iter_mut()' in source) == (pattern == 'positive')
                dumps.append((directory/'rust/results.bin').read_bytes())
        for snapshot in snapshots[1:]:
            for actual, expected in zip(snapshot, snapshots[0], strict=True):
                np.testing.assert_array_equal(actual, expected)
        assert dumps[0] == dumps[1]
    finally:
        device.reinit()
        b.set_device(previous)
        b.start_scope()
