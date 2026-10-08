"""Copy native-validated pytest WASM results into the browser verification corpus."""
import argparse
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=ROOT/'output/wasm-tests',
                    help='pytest WASM corpus directory from a validated run')
parser.add_argument('--output',type=Path,default=ROOT/'output/wasm/checks',
                    help='checks directory inside the served WASM example')
args=parser.parse_args()
source=args.source.resolve()
destination=args.output.resolve()
names=[]
for case in sorted(source.glob('test_wasm_matches_native_with_[0-9]*')):
    if case.is_symlink(): continue
    name=(case/'case.txt').read_text()
    target=destination/name; target.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(case/'bundle.json',target/'bundle.json')
    for filename in ('results.bin','events.bin'):
        path=case/'wasm-1'/filename
        (target/filename).write_bytes(path.read_bytes() if path.exists() else b'')
    names.append(name)
if not names: raise SystemExit(f'No validated WASM cases found under {source}; run test_wasm.py and pass its --basetemp directory as --source')
(destination/'index.json').write_text(json.dumps(names))
print(f'Prepared {len(names)} cases; open /check-browser.html on the WASM example server')
