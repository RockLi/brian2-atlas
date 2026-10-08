"""Real subset identity, intervention controls, stochastic WASM/native parity."""
import hashlib
import importlib.util
import json
import struct
import subprocess
import numpy as np
import brian2 as b
import pytest
from test_wasm import ROOT, compare
from brian2_rust.results import load_results


@pytest.fixture(scope='module')
def flywire_results(tmp_path_factory):
    directory=tmp_path_factory.mktemp('flywire-browser')
    result=subprocess.run(['node',str(ROOT/'wasm/check-flywire.mjs'),str(ROOT/'output/wasm'),str(directory)],check=True,capture_output=True,text=True)
    (directory/'metrics.json').write_text(result.stdout)
    return directory


def test_real_circuit_identity():
    circuit=json.loads((ROOT/'wasm/flywire-circuit.json').read_text())
    assert circuit['source_csr_sha256']=='b84a19b5c6d899181e6ade3a914eba89d53cb3a4cc36789702785b4cfa35ec38'
    assert circuit['original_csr_sha256']=='4f0a4a31332ba489d796fefc7228c7471d0d0ca0939805ffe7369bfca1e12682'
    canonical=json.dumps({key:circuit[key] for key in ('nodes','edges')},sort_keys=True,separators=(',',':')).encode()
    assert hashlib.sha256(canonical).hexdigest()==circuit['circuit_sha256']
    assert len(circuit['nodes'])==len({n['root_id'] for n in circuit['nodes']})==240
    assert len(circuit['edges'])==6660
    assert sum(e['contacts'] for e in circuit['edges'])==66815
    assert all(isinstance(n['root_id'],str) and 2**53<int(n['root_id'])<2**64 for n in circuit['nodes'])
    assert all(e['signed_contacts']==e['contacts']*circuit['nodes'][e['source']]['sign'] for e in circuit['edges'])
    assert {n['root_id'] for n in circuit['nodes'] if n['group']=='projection'}=={'720575940619071005','720575940630770042'}


@pytest.mark.parametrize('case',['odor','rest','cut','cut_rest','pulse'])
def test_flywire_native_parity(flywire_results,case):
    directory=flywire_results/case;model=json.loads((directory/'model.json').read_text())
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(directory/'model.json'),str(directory/'native')],check=True,capture_output=True)
    compare(load_results(model,directory),load_results(model,directory/'native'))


def test_flywire_deterministic_brian_numpy(flywire_results):
    directory=flywire_results/'pulse';model=json.loads((directory/'model.json').read_text())
    config=json.loads((directory/'config.json').read_text())
    circuit=json.loads((ROOT/'wasm/flywire-circuit.json').read_text())
    spec=importlib.util.spec_from_file_location('spa_template',ROOT/'tools/spa_template.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    previous=b.get_device();target=b.prefs.codegen.target
    decode=lambda values:np.array([struct.unpack('>d',bytes.fromhex(v))[0] for v in values])
    try:
        b.set_device('runtime');b.prefs.codegen.target='numpy';b.start_scope()
        net,group,syn,states,spikes=module.create_flywire_network(circuit)
        pop=model['instance']['populations'][0]
        for kind in ('initial_state','parameters'):
            for name,values in pop[kind].items():
                if name=='ms':assert np.array_equal(decode(values),[.001]);continue
                group.variables[name].set_value(decode(values))
        parameters=model['instance']['synapses'][0]['parameters']
        syn.signed_contacts=decode(parameters['signed_contacts'])
        syn.namespace.update({key:decode(parameters[key])[0] for key in ['recurrent_weight','inhibitory_gain']})
        net.run(config['duration_ms']*b.ms)
        actual=load_results(model,directory)['populations'][0]
        np.testing.assert_array_equal(actual['indices'],np.asarray(spikes.i))
        np.testing.assert_array_equal(actual['spike_ticks'],np.rint(spikes.t/(.1*b.ms)).astype(int))
        for name in ('v','ge','gi'):
            np.testing.assert_allclose(actual['trace'][name],np.asarray(getattr(states,name)).T,rtol=1e-10,atol=1e-10)
    finally:
        b.prefs.codegen.target=target;b.set_device(previous)
