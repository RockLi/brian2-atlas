"""Compile a frozen generated AOT kernel for a single browser Worker.

Only host imports/environment access change; equations, scheduling, RNG and
binary serialization are preserved from the supplied generated source.
"""
import argparse, hashlib, json, pathlib, shutil, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build(native,output,spike_population,wasm_bindgen):
    native=native.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    source=(native/'main.rs').read_text()
    for old,new in [('use std::fs::{self, File};','mod browser_host;\nuse browser_host::{fs,File};'),
                    ('use std::time::Instant;','use browser_host::Instant;')]:
        if source.count(old)!=1:raise ValueError('unsupported AOT host layout: '+old)
        source=source.replace(old,new)
    source=source.replace('std::env::var(', 'browser_host::env_var(')
    identity=sha(native/'instance.bin')
    source+='''
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub fn load_snapshot(bytes:Vec<u8>)->std::result::Result<(),JsValue>{
 use sha2::{Digest,Sha256};
 let actual=format!("{:x}",Sha256::digest(&bytes));
 if actual!="SNAPSHOT_HASH"{return Err(JsValue::from_str("snapshot checksum mismatch"));}
 browser_host::install_snapshot(bytes);Ok(())
}
#[wasm_bindgen]
pub fn infer_spikes(indices:Vec<u32>,ticks:Vec<u32>)->std::result::Result<Vec<u8>,JsValue>{
 browser_host::clear_outputs();
 let operation=||->Result<Vec<u8>>{
  check(indices.len()==ticks.len(),"spike schedule length mismatch")?;
  let input=SpikeInput{population:SPIKE_POPULATION,indices:indices.into_iter().map(|x|x as usize).collect(),ticks:ticks.into_iter().map(|x|x as usize).collect()};
  execute(Reader::open(Path::new("base.bin"))?,Path::new("output"),Instant::now(),Some(input))?;
  browser_host::take("output/results.bin").map_err(Into::into)
 };
 match operation(){Ok(value)=>Ok(value),Err(error)=>{browser_host::clear_outputs();Err(JsValue::from_str(&error.to_string()))}}
}
#[wasm_bindgen]
pub fn take_summary()->std::result::Result<String,JsValue>{
 let result=browser_host::take("output/summary.json").and_then(|bytes|String::from_utf8(bytes).map_err(|_|std::io::Error::other("invalid summary")));
 browser_host::clear_outputs();result.map_err(|e|JsValue::from_str(&e.to_string()))
}
'''.replace('SNAPSHOT_HASH',identity).replace('SPIKE_POPULATION',str(spike_population))
    crate=output/'crate';(crate/'src').mkdir(parents=True,exist_ok=True)
    (crate/'Cargo.toml').write_text('[package]\nname="flywire-browser-aot"\nversion="0.1.0"\nedition="2021"\n[lib]\ncrate-type=["cdylib"]\n[dependencies]\nwasm-bindgen="=0.2.100"\nsha2="0.10"\n[profile.release]\nopt-level=3\nlto=true\ncodegen-units=1\npanic="abort"\n')
    (crate/'src/lib.rs').write_text(source)
    shutil.copyfile(ROOT/'python/brian2_rust/browser_host.rs',crate/'src/browser_host.rs')
    subprocess.run(['cargo','build','--offline','--release','--target','wasm32-unknown-unknown','--manifest-path',str(crate/'Cargo.toml')],check=True)
    subprocess.run([wasm_bindgen,str(crate/'target/wasm32-unknown-unknown/release/flywire_browser_aot.wasm'),'--target','web','--out-dir',str(output/'pkg')],check=True)
    (output/'build.json').write_text(json.dumps(dict(schema='b2-wasm-aot-build-v1',numeric_profile='wasm-aot-f64-serial',source_sha256=sha(native/'main.rs'),adapted_source_sha256=sha(crate/'src/lib.rs'),host_sha256=sha(crate/'src/browser_host.rs'),snapshot_sha256=identity,wasm_sha256=sha(output/'pkg/flywire_browser_aot_bg.wasm'),spike_population=spike_population),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--spike-population',type=int,required=True);p.add_argument('--wasm-bindgen',default='wasm-bindgen');a=p.parse_args()
    if a.spike_population<0:p.error('spike population must be nonnegative')
    build(a.native,a.output,a.spike_population,a.wasm_bindgen)
