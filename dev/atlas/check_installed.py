"""Exercise installed Atlas without source paths or native-binary overrides."""
import argparse,copy,hashlib,importlib.metadata,json,os,pathlib,subprocess,sys,tempfile,time
p=argparse.ArgumentParser();p.add_argument('--report',type=pathlib.Path,required=True);p.add_argument('--metal',action='store_true');a=p.parse_args()
for key in ['PYTHONPATH','B2_RUNNER','B2_TRAIN_RUNNER']:
    assert not os.environ.get(key),f'Installation gate forbids {key}'
started=time.time()
import brian2 as b
import brian2_atlas as atlas
import brian2_rust as implementation
import numpy as np
from brian2_rust._runtime import executable_path,source_root
prefix=pathlib.Path(sys.prefix).resolve();package=pathlib.Path(implementation.__file__).resolve().parent
assert pathlib.Path(atlas.__file__).resolve().is_relative_to(prefix)
assert atlas.AtlasDevice is implementation.RustStandaloneDevice
assert pathlib.Path(b.__file__).resolve().is_relative_to(prefix)
assert package.is_relative_to(prefix) and source_root() is None
assert importlib.metadata.version('brian2-atlas')=='0.1.0'
assert b.__version__=='2.10.1.post241'
def inventory():
    return {str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
before=inventory();build=json.loads((package/'_bin/build.json').read_text())
for name in ['b2-runner','b2-train']:
    binary=executable_path(name);assert binary.is_relative_to(package/'_bin')
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binaries'][binary.name]
    run=subprocess.run([str(pathlib.Path(sys.executable).parent/name),'--help'],capture_output=True,text=True,timeout=900)
    assert 'usage:' in (run.stdout+run.stderr).lower(),(name,run.returncode,run.stdout,run.stderr)
for resource in ['browser_host.rs','training_metal.m','training_dynamic.metal','training_cuda.cu',
                 'training_mpi.c','metal_runtime/bridge.m','mpi_runtime/gpu.rs','rust-toolchain.toml']:
    assert (package/resource).is_file(),resource
b.prefs.codegen.target='numpy'
b.prefs.codegen.runtime.cython.cache_dir=str(pathlib.Path(os.environ['XDG_CACHE_HOME'])/'installed-atlas-cython')
def simulate(engine):
    if engine=='numpy':b.set_device('runtime')
    else:
        from brian2.devices.device import all_devices
        assert all_devices['atlas'] is all_devices['rust_standalone']
        all_devices['atlas'].reinit();b.set_device('atlas',engine=engine)
    b.start_scope()
    neurons=b.NeuronGroup(1,'dv/dt=(1.5-v)/(10*ms) : 1',threshold='v>1',reset='v=0',method='euler',dt=0.1*b.ms)
    state=b.StateMonitor(neurons,'v',record=True);spikes=b.SpikeMonitor(neurons)
    network=b.Network(neurons,state,spikes);network.run(100*b.ms)
    result={'state':np.asarray(state.v).copy(),'final':np.asarray(neurons.v[:]).copy(),
            'spikes':np.asarray(spikes.t[:]/b.ms).copy()}
    assert spikes.num_spikes==9
    if engine!='numpy':
        directory=pathlib.Path(b.get_device().last_run_directory).resolve()
        assert directory.is_relative_to(pathlib.Path(os.environ['TMPDIR']).resolve()),directory
        assert not directory.is_relative_to(prefix)
        result['artifact_directory']=str(directory)
    return result
reference=simulate('numpy');simulations={}
for engine in ['reference','aot']:
    result=simulate(engine)
    for key in ['state','final']:np.testing.assert_allclose(result[key],reference[key],rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(result['spikes'],reference['spikes'])
    simulations[engine]={'spikes':result['spikes'].tolist(),'artifact_directory':result['artifact_directory']}
b.set_device('runtime')
inputs=np.zeros((4,24,2));inputs[:2,:,0]=1;inputs[2:,:,1]=1;labels=[0,0,1,1]
plan=atlas.lif_training_plan([2,2,2],beta=0.8)
weights=[[1.2,0.4,0.4,1.2],[0.2,0.8,0.8,0.2]]
trainer=atlas.NativeLIFTrainer(plan,weights=weights)
expected=trainer.gradients(inputs,labels)
assert np.isfinite(expected['loss']) and any(np.any(np.asarray(g)!=0) for g in expected['gradients'])
trainer.step(inputs,labels)
with tempfile.TemporaryDirectory(prefix='atlas-installed-checkpoint-') as directory:
    checkpoint=pathlib.Path(directory)/'checkpoint.json';trainer.store(checkpoint)
    trainer.step(inputs,labels);expected_state=copy.deepcopy(trainer.state)
    code='''import json,sys
import numpy as np
from brian2_atlas import NativeLIFTrainer
trainer=NativeLIFTrainer(json.loads(sys.argv[2]));trainer.restore(sys.argv[1])
x=np.zeros((4,24,2));x[:2,:,0]=1;x[2:,:,1]=1
trainer.step(x,[0,0,1,1]);print(json.dumps(trainer.state))
'''
    child=subprocess.run([sys.executable,'-c',code,str(checkpoint),json.dumps(plan)],capture_output=True,text=True,check=True,timeout=900)
    assert json.loads(child.stdout)==expected_state
if a.metal:
    gpu=atlas.NativeLIFTrainer(dict(plan,backend='metal'),weights=weights)
    actual=gpu.gradients(inputs,labels)
    np.testing.assert_array_equal(actual['spikes'],expected['spikes'])
    for key in ['gradients','initial_gradients','final_membrane','logits']:
        np.testing.assert_allclose(actual[key],expected[key],rtol=3e-5,atol=3e-6)
assert inventory()==before,'Installed package contents changed during execution'
report={'schema':'atlas-installed-flow-v1','started_unix':started,'ended_unix':time.time(),
        'python':sys.version,'prefix':str(prefix),'brian2':str(b.__file__),'backend':str(atlas.__file__),
        'distribution':importlib.metadata.version('brian2-atlas'),'compatibility_version':b.__version__,
        'native_build':build,'simulations':simulations,'training_checkpoint_fresh_process':'passed',
        'metal_training':'passed' if a.metal else 'not requested',
        'installed_package_unchanged':True,'status':'passed'}
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'passed','report':str(a.report)}))
