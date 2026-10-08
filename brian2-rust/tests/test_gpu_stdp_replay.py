"""Precompiled STDP replays must reset learned weights, queues and recordings."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize('backend',['rust-f64','cpu-f32','cpp-f64-t1','cpp-f64-t4'])
def test_compiled_replay_resets_full_plastic_network(backend,tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    from gpu_stdp_precompiled import Replay,artifact_hashes
    from gpu_stdp_compare import oracle,checks
    replay=Replay(backend,17,48,7,tmp_path)
    try:
        for _ in range(2):
            actual,timing=replay.run()
            assert checks(actual,oracle(17,7,48))['passed']
            if backend.startswith('cpp-f64-'):assert timing['details']['observed_openmp_threads']==int(backend[-1])
            assert all(np.array_equal(actual[k],replay.bootstrap[k]) for k in actual)
            assert artifact_hashes(tmp_path)==replay.artifacts
            assert timing['details']['result_array_bytes']==sum(a.nbytes for a in actual.values())
    finally:replay.close()


def test_rust_thread_request_reports_serial_slot_fallback(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    from gpu_stdp_precompiled import Replay
    replay=Replay('rust-f64',17,48,7,tmp_path)
    try:
        actual,timing=replay.run(rust_threads=4)
        runtime=timing['details']['rust_runtime']
        assert timing['details']['requested_rust_threads']==4 and runtime['threads']==1
        assert replay.adapter_evidence['execution_plan']['cpu']['emitter']=='slot-v1'
        assert not any(runtime[k] for k in runtime if k.startswith('parallel_') and isinstance(runtime[k],bool))
        assert all(np.array_equal(actual[k],replay.bootstrap[k]) for k in actual)
    finally:replay.close()
