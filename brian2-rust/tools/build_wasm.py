"""Build the shared WASM engine and a locally served browser example."""
import argparse
import json
import hashlib
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


def copy_web_assets(output):
    """Version local module URLs together so browser caches cannot mix builds."""
    sources=sorted(p for p in (ROOT/'wasm').iterdir() if p.is_file())
    digest=hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode());digest.update(source.read_bytes())
    wasm=output/'pkg/b2_runner_bg.wasm'
    if wasm.exists():digest.update(wasm.read_bytes())
    version=digest.hexdigest()[:16]
    for source in sources:
        target=output/source.name
        if source.suffix in ('.js','.html'):
            content=source.read_text()
            content=re.sub(r"(['\"])(\./[\w/.-]+\.js)\1",lambda m:f'{m[1]}{m[2]}?v={version}{m[1]}',content)
            content=re.sub(r'(src|href)="([^":?]+\.(?:js|css))"',lambda m:f'{m[1]}="{m[2]}?v={version}"',content)
            target.write_text(content)
        else:shutil.copyfile(source,target)
    package=output/'pkg/b2_runner.js'
    if package.exists():
        package.write_text(re.sub(r"b2_runner_bg\.wasm(?:\?v=[a-f0-9]+)?",f'b2_runner_bg.wasm?v={version}',package.read_text()))
    (output/'asset-version.json').write_text(json.dumps({'version':version})+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/wasm')
    parser.add_argument('--wasm-bindgen', default='wasm-bindgen')
    args = parser.parse_args()
    version = subprocess.check_output([args.wasm_bindgen, '--version'], text=True).strip()
    if version != 'wasm-bindgen 0.2.100':
        parser.error('install the matching tool: cargo install wasm-bindgen-cli --version 0.2.100 --locked')
    subprocess.run(['cargo', 'build', '--manifest-path', str(ROOT/'Cargo.toml'), '--locked',
                    '--release', '--lib', '--target', 'wasm32-unknown-unknown',
                    '--target-dir', str(ROOT/'target')], check=True)
    subprocess.run(['cargo', 'build', '--manifest-path', str(ROOT/'Cargo.toml'), '--locked',
                    '--release', '--bin', 'b2-runner', '--target-dir', str(ROOT/'target')], check=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.wasm_bindgen, str(ROOT/'target/wasm32-unknown-unknown/release/b2_runner.wasm'),
                    '--target', 'web', '--out-dir', str(output/'pkg')], check=True)
    copy_web_assets(output)
    from brian2_rust.wasm import export_wasm_bundle
    model = json.loads((ROOT/'tests/golden/b2ir-v1/minimal-v1.json').read_text())
    export_wasm_bundle(model, output/'sample.json')
    from spa_template import write_templates
    write_templates(output)
    import runpy
    runpy.run_path(str(ROOT/'wasm/export-browser-model.py'))['export_example'](output)
    print(f'Built {output}\nServe: python -m http.server 8765 --bind 127.0.0.1 --directory {output}')


if __name__ == '__main__': main()
