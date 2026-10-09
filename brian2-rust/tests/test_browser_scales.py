"""Large browser populations retain full statistics and native numerical parity."""
import hashlib
import json
import subprocess
import pytest
import numpy as np
from test_wasm import ROOT,compare
from brian2_rust.results import load_results


@pytest.fixture(scope='module')
def scale_results(tmp_path_factory):
    directory=tmp_path_factory.mktemp('browser-scales')
    subprocess.run(['node',str(ROOT/'wasm/check-scales.mjs'),str(ROOT/'output/wasm'),str(directory)],check=True,capture_output=True,text=True)
    return directory


@pytest.mark.parametrize('case',['adaptive_lif-32768','izhikevich-8192','hodgkin_huxley-4096','flywire-1024','flywire-4096'])
def test_large_native_parity(scale_results,case):
    directory=scale_results/case;model=json.loads((directory/'model.json').read_text())
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(directory/'model.json'),str(directory/'native')],check=True,capture_output=True)
    actual=load_results(model,directory);expected=load_results(model,directory/'native')
    if case.startswith('hodgkin_huxley'):
        # Same libm tolerance as the existing classic-model parity checks.
        a=actual['populations'][0];e=expected['populations'][0]
        for field in ('spike_ticks','indices','counts'):
            np.testing.assert_array_equal(a[field],e[field])
        for field in ('trace','states'):
            for name in a[field]:np.testing.assert_allclose(a[field][name],e[field][name],rtol=1e-10,atol=1e-9)
    else:compare(actual,expected)


def test_nested_real_circuits():
    previous=None
    for size,edges,contacts in [(240,6660,66815),(1024,54609,458573),(4096,284105,901832)]:
        suffix='' if size==240 else f'-{size}'
        circuit=json.loads((ROOT/f'wasm/flywire-circuit{suffix}.json').read_text())
        nodes=circuit['nodes'];roots={n['root_id'] for n in nodes}
        assert len(nodes)==len(roots)==size
        assert len(circuit['edges'])==edges
        assert sum(e['contacts'] for e in circuit['edges'])==contacts
        assert sum(g['count'] for g in circuit['groups'])==size
        assert all(isinstance(n['root_id'],str) and 2**53<int(n['root_id'])<2**64 for n in nodes)
        assert all(e['signed_contacts']==e['contacts']*nodes[e['source']]['sign'] for e in circuit['edges'])
        canonical=json.dumps({key:circuit[key] for key in ('nodes','edges')},sort_keys=True,separators=(',',':')).encode()
        assert hashlib.sha256(canonical).hexdigest()==circuit['circuit_sha256']
        connections={(nodes[e['source']]['root_id'],nodes[e['target']]['root_id']):(e['contacts'],e['signed_contacts']) for e in circuit['edges']}
        if previous:
            old_roots,old_connections,source_hash=previous
            assert old_roots<=roots
            assert {key:value for key,value in connections.items() if set(key)<=old_roots}==old_connections
            assert circuit['source_csr_sha256']==source_hash
        previous=roots,connections,circuit['source_csr_sha256']
