from pathlib import Path
import hashlib,importlib.util,json,subprocess,tempfile,numpy as np
from brian2_atlas.neuromorphic import export_native_xylo
repo=Path('/Users/rock/workspace/bettiai/brian2-atlas');root=Path('/Volumes/T7/atlas-sequential-mnist-round4-20261014');base='ef772a28abe88a3e26abb33199c87485e49b0e94'
f=json.loads((root/'current5/frozen.json').read_text());assert f['plan']['readout_source']=='synaptic_current'
receipts=[]
with tempfile.TemporaryDirectory(prefix='atlas-legacy-readout-') as name:
 temp=Path(name)
 for file in ('xylo_readout.py','xylo_runtime.py','spinnaker2_integer_runtime.py'):
  data=subprocess.check_output(['git','show',base+':brian2-rust/python/brian2_rust/'+file],cwd=repo);(temp/file).write_bytes(data)
 for target,file in [('xylosim','xylo_runtime.py'),('spinnaker2','spinnaker2_integer_runtime.py')]:
  bundle=export_native_xylo(f,temp/target,steps=36,runner='/Volumes/T7/atlas-sequential-mnist-round3-20261014/b2-runner-wide',target=target)
  assert bundle['native_training_bridge']['schema']=='atlas.native-xylo-current-bridge.v1'
  spec=importlib.util.spec_from_file_location('legacy_'+target,temp/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  try:
   if target=='xylosim':m.run_native_batch(bundle['plan'],bundle['native_training_bridge'],np.zeros((1,36,16),dtype=np.uint8))
   else:m.run(bundle['plan'],np.zeros((1,36,16),dtype=np.uint8),bridge=bundle['native_training_bridge'])
  except ValueError as error:
   assert 'unknown native' in str(error)
   receipts.append(dict(target=target,rejected=True,error=str(error),old_runtime_sha256=hashlib.sha256((temp/file).read_bytes()).hexdigest()))
  else:raise AssertionError('old runtime silently accepted current readout')
(root/'build-evidence/legacy-rejection.json').write_text(json.dumps(dict(base_commit=base,test_arrays_read=False,checks=receipts),indent=2)+'\n')
print(json.dumps(receipts,indent=2))
