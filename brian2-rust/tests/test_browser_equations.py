"""Compare the frontend equation compiler with independent Brian2 NumPy runs."""
import json
import struct
import subprocess
import numpy as np
import pytest
import brian2 as b
from test_wasm import ROOT

CASES=['adex-tonic','adex-adapting','adex-silent','quadratic_if-tonic','quadratic_if-strong','quadratic_if-silent','custom-lif','custom-adex','custom-qif','custom-izh','simultaneous']

@pytest.fixture(scope='module')
def equation_results(tmp_path_factory):
    path=tmp_path_factory.mktemp('equations')
    subprocess.run(['node',str(ROOT/'wasm/check-equations.mjs'),str(ROOT/'output/wasm'),str(path)],check=True,capture_output=True,text=True)
    return path

@pytest.mark.parametrize('case',CASES)
def test_browser_equations_match_brian_numpy(equation_results,case):
    path=equation_results/case
    c=json.loads((path/'config.json').read_text());spec=c['custom']
    model=json.loads((path/'model.json').read_text());actual=json.loads((path/'observed.json').read_text())
    b.set_device('runtime');b.start_scope();b.prefs.codegen.target='numpy'
    group=b.NeuronGroup(c['neurons'],spec['equations']+'\ndrive : 1 (constant)',threshold=spec['threshold'],reset=spec['reset'],refractory=spec['refractory_ms']*b.ms,dt=c['dt_ms']*b.ms,method='euler',namespace=spec['parameters'])
    for key,value in spec['initial'].items():setattr(group,key,value)
    group.drive=[struct.unpack('>d',bytes.fromhex(value))[0] for value in model['instance']['populations'][0]['parameters']['drive']]
    trace=b.StateMonitor(group,list(spec['initial']),record=True)
    spikes=b.SpikeMonitor(group)
    b.Network(group,trace,spikes).run(c['duration_ms']*b.ms)
    np.testing.assert_array_equal(actual['ticks'],np.rint(spikes.t[:]/(c['dt_ms']*b.ms)).astype(int))
    np.testing.assert_array_equal(actual['indices'],spikes.i[:])
    for state,values in actual['trace'].items():
        np.testing.assert_allclose(np.asarray(values).reshape(-1,c['neurons']),getattr(trace,state).T,rtol=1e-9,atol=1e-8)
