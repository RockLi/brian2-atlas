"""Run the actual frozen TF SavedModel, without labels or target outputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import tensorflow as tf
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
with np.load(a.inputs,allow_pickle=False) as data:x=data['inputs'].astype(np.int64)
model=tf.saved_model.load(str(a.model))
counts=model(tf.constant(x))['counts'].numpy()
np.savez_compressed(a.output/'results.npz',counts=counts)
report=dict(tensorflow_version=tf.__version__,samples=len(x),
            inputs_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
            model_files={str(p.relative_to(a.model)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.model.rglob('*')) if p.is_file()},
            results_sha256=hashlib.sha256((a.output/'results.npz').read_bytes()).hexdigest())
(a.output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
