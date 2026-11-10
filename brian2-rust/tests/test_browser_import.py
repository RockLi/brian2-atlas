"""Brian2 exports replay unchanged in WASM and decode without assuming one population."""
import json
import runpy
import struct
import subprocess
import numpy as np
import brian2 as b
import pytest
from brian2_rust.export import lower_network
from brian2_rust import export_wasm_bundle
from brian2_rust.results import load_results
from test_metal_delays import ROOT, device

@pytest.mark.parametrize('case',['connected','multiclock_window','large_seed','typed'])
def test_import_bundle(case,device,tmp_path):
    if case=='connected':
        path=runpy.run_path(str(ROOT/'wasm/export-browser-model.py'))['export_example'](tmp_path)
        model=json.loads((tmp_path/'model.json').read_text())
    else:
        b.set_device('rust_standalone',build_on_run=False)
        if case=='typed':
            p=b.NeuronGroup(2,'x:1\nwide:integer',dtype={'x':np.float32,'wide':np.int64},dt=.1*b.ms,name='typed')
            p.x=[.1,.2];p.wide=[2**53+1,2**53+3]
            mon=b.StateMonitor(p,['x','wide'],record=True)
            network=b.Network(p,mon);duration=1*b.ms;window=None
        else:
            p=b.NeuronGroup(3,'dx/dt=1000/second:1',threshold='x>0.5',reset='x=0',dt=.1*b.ms,method='euler',name='fast')
            q=b.NeuronGroup(2,'dy/dt=500/second:1',threshold='y>0.5',reset='y=0',dt=.2*b.ms,method='euler',name='slow')
            network=b.Network(p,q,b.SpikeMonitor(p),b.SpikeMonitor(q),b.StateMonitor(p,'x',record=True),b.StateMonitor(q,'y',record=True));duration=4*b.ms;window=10 if case=='multiclock_window' else None
        model=lower_network(network,duration,rng_seed=2**64-1,recording_window_steps=window)
        path=tmp_path/'model.browser.json';export_wasm_bundle(model,path)
        (tmp_path/'model.json').write_text(json.dumps(model))
    output=tmp_path/'wasm';output.mkdir()
    subprocess.run(['node',str(ROOT/'wasm/check-import.mjs'),str(ROOT/'output/wasm'),str(path),str(output)],check=True,capture_output=True,text=True)
    subprocess.run([str(ROOT/'target/release/b2-runner'),str(tmp_path/'model.json'),str(tmp_path/'native')],check=True,capture_output=True)
    native=load_results(model,tmp_path/'native');actual=load_results(model,output)
    decoded=json.loads((output/'decoded.json').read_text())
    assert len(decoded['populations'])==len(native['populations'])
    for k,(n,a,d) in enumerate(zip(native['populations'],actual['populations'],decoded['populations'])):
        for field in ('spike_ticks','indices','counts'):np.testing.assert_array_equal(a[field],n[field])
        np.testing.assert_array_equal(d['indices'],n['indices']);np.testing.assert_array_equal(d['counts'],n['counts'])
        dt=struct.unpack('>d',bytes.fromhex(model['definition']['populations'][k]['dt']))[0]
        np.testing.assert_allclose(np.array(d['ticks'])*dt+d['startMs']/1000,n['spike_times'],rtol=0,atol=1e-15)
        for key,values in n['trace'].items():
            np.testing.assert_array_equal(a['trace'][key],values)
            if key=='wide':assert key in d['omitted'] and key not in d['trace']
            else:np.testing.assert_array_equal(np.asarray(d['trace'][key]).reshape(values.shape),values)
    if case=='connected':assert decoded['synapticEvents']>0 and all(p['spikes']>0 for p in decoded['populations'])
    if case=='multiclock_window':assert all(p['startMs']>0 for p in decoded['populations'])
