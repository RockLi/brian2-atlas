"""Empirical CSR import, exact root IDs, and native/runtime conformance."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import brian2 as b
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
sys.path.insert(0, str(ROOT/'examples'))
import brian2_rust as rust
from brian2_rust.binary_topology import HEADER, MAGIC, inspect_csr, csr_arrays
from flywire_import import convert


def fixture(path, n=4):
    with path.open('wb') as f:
        f.write(HEADER.pack(MAGIC, n, n, 5, 1))
        f.write(np.array([0,2,3,4,5]+[5]*(n-4),dtype='<u8').tobytes())
        f.write(np.array([0,2,3,1,0],dtype='<u4').tobytes())
        f.write(np.array([1,3,2,4,5],dtype='<f8').tobytes())
    return path


def test_stream_import_aggregates_neuropils_and_keeps_exact_ids_and_isolates(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.feather as feather
    base = 720575940624078480
    # int64 Arrow IDs and uint64 NPY IDs must not promote to float64.
    ids = np.array([base+4,base+5,base+9,base+13],dtype=np.uint64)
    np.save(tmp_path/'ids.npy',ids)
    table=pa.table({'pre_pt_root_id':np.array([base+4,base+4,base+5],dtype=np.int64),
                    'post_pt_root_id':np.array([base+5,base+5,base+4],dtype=np.int64),
                    'syn_count':[2,3,7]})
    feather.write_feather(table,tmp_path/'data.feather',chunksize=1)
    result=convert(tmp_path/'data.feather',tmp_path/'ids.npy',tmp_path/'out',False)
    assert (result['neurons'],result['directed_pair_edges'],result['biological_contacts'])==(4,2,12)
    offsets,targets,values=csr_arrays(inspect_csr(tmp_path/'out/connectome.b2csr'))
    np.testing.assert_array_equal(offsets,[0,1,2,2,2])
    np.testing.assert_array_equal(targets,[1,0])
    np.testing.assert_array_equal(values,[[5,7]])


@pytest.mark.parametrize('corruption',['offsets','target','nan','truncated'])
def test_malformed_csr_rejected_by_python(tmp_path,corruption):
    path=fixture(tmp_path/'graph')
    offsets={'offsets':(HEADER.size+8,struct.pack('<Q',9)),
             'target':(HEADER.size+5*8,struct.pack('<I',4)),
             'nan':(HEADER.size+5*8+5*4,struct.pack('<d',float('nan')))}
    if corruption=='truncated':path.write_bytes(path.read_bytes()[:-1])
    else:
        at,data=offsets[corruption]
        with path.open('r+b') as f:f.seek(at);f.write(data)
    with pytest.raises(ValueError):inspect_csr(path)


def run_model(backend,path,tmp_path,n=4):
    old=b.get_device();b.device.reinit();b.start_scope()
    try:
        if backend=='numpy':
            b.set_device('runtime');b.prefs.codegen.target='numpy'
        else:
            b.set_device('rust_standalone',engine=backend,runner=ROOT/'target/release/b2-runner',
                         directory=tmp_path/backend)
        g=b.NeuronGroup(n,'dv/dt=(1.2-v)/(10*ms) : 1',threshold='v>1',reset='v=0',
                        dt=.1*b.ms,method='euler',name='neurons')
        g.v=np.linspace(0,1.1,n)
        s=b.Synapses(g,g,'multiplicity : 1 (constant)',on_pre='v_post += .02*multiplicity',
                     delay=.2*b.ms,clock=g.clock,name='projection')
        if backend=='numpy':
            offsets,targets,values=csr_arrays(inspect_csr(path))
            s.connect(i=np.repeat(np.arange(n),np.diff(offsets).astype(int)),j=targets.astype(np.int32))
            s.multiplicity=values[0]
        else:rust.connect_binary_csr(s,path,parameters={'multiplicity':0})
        m=b.StateMonitor(g,'v',record=list(range(min(4,n))),name='trace')
        spikes=b.SpikeMonitor(g,name='spikes')
        net=b.Network(g,s,m,spikes);net.run(2*b.ms)
        result=(g.v[:].copy(),m.v[:].copy(),spikes.i[:].copy(),np.asarray(spikes.t[:]).copy())
        if backend!='numpy':
            model_path=b.get_device().last_run_directory/'model.json'
            model=json.loads(model_path.read_text())
            assert model['instance']['synapses'][0]['source']==[]
            assert model['instance']['synapses'][0]['parameters']['multiplicity']==[]
            if n==4 and backend=='aot':
                # The compiled artifact must replay after its source CSR is gone.
                moved=path.with_suffix('.moved')
                path.rename(moved)
                try:
                    folder=b.get_device().last_run_directory
                    output=tmp_path/'independent-replay'
                    subprocess.run([str(folder/'native/b2-native'),str(folder/'native/instance.bin'),str(output)],check=True,capture_output=True)
                    assert (output/'results.bin').read_bytes()==(folder/'rust/results.bin').read_bytes()
                finally:moved.rename(path)
            if n==4:
                # Independent runner must reject a malformed external file too.
                original=path.read_bytes()
                try:
                    with path.open('r+b') as f:f.seek(HEADER.size+8);f.write(struct.pack('<Q',999))
                    check=subprocess.run([str(ROOT/'target/release/b2-runner'),'--validate',str(model_path)],capture_output=True)
                    assert check.returncode!=0
                finally:path.write_bytes(original)
                # A finite, structurally valid edit must also fail its declared digest.
                try:
                    info=inspect_csr(path)
                    with path.open('r+b') as f:
                        f.seek(info['parameter_offset']);f.write(struct.pack('<d', 9.0))
                    check=subprocess.run([str(ROOT/'target/release/b2-runner'),'--validate',str(model_path)],capture_output=True)
                    assert check.returncode != 0
                    assert b'binary CSR checksum mismatch' in check.stderr
                finally:path.write_bytes(original)
        return result
    finally:
        b.device.reinit();b.set_device(old);b.start_scope()


def test_binary_csr_matches_numpy_reference_and_aot(tmp_path):
    path=fixture(tmp_path/'graph')
    expected=run_model('numpy',path,tmp_path)
    for backend in ['reference','aot']:
        actual=run_model(backend,path,tmp_path)
        for left,right in zip(actual,expected,strict=True):
            np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-14)


def test_full_flywire_neuron_count_passes_native_validation(tmp_path):
    path=fixture(tmp_path/'graph',139255)
    actual=run_model('aot',path,tmp_path,139255)
    assert actual[0].shape==(139255,)


def test_binary_csr_rejects_non_float64_column_parameter(tmp_path):
    from brian2_rust.export import lower_network
    old=b.get_device();b.device.reinit();b.start_scope()
    try:
        b.set_device('rust_standalone',engine='aot',runner=ROOT/'target/release/b2-runner',
                     directory=tmp_path/'aot')
        g=b.NeuronGroup(4,'v : 1',threshold='v>1',reset='v=0',name='typed_neurons')
        s=b.Synapses(g,g,'w : 1 (constant)',on_pre='v_post += w',
                     dtype={'w':np.float32},clock=g.clock,name='typed_projection')
        rust.connect_binary_csr(s,fixture(tmp_path/'graph'),parameters={'w':0})
        with pytest.raises(NotImplementedError,match='binary CSR columns require float64'):
            lower_network(b.Network(g,s),1*b.ms)
    finally:
        b.device.reinit();b.set_device(old);b.start_scope()
