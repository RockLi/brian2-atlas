//! Single-worker, memory-only host for unchanged generated AOT kernels.
//! The snapshot is immutable; each invocation reconstructs all simulation state.
use std::{cell::RefCell, collections::BTreeMap, io::{self, Read, Write}, path::Path, rc::Rc};
use wasm_bindgen::prelude::*;
thread_local! { static FILES: RefCell<BTreeMap<String,Rc<RefCell<Vec<u8>>>>> = RefCell::new(BTreeMap::new()); }
fn invalid(message:&str)->io::Error{io::Error::new(io::ErrorKind::InvalidInput,message)}
pub fn install_snapshot(bytes:Vec<u8>){FILES.with(|files|{let mut files=files.borrow_mut();files.clear();files.insert("base.bin".into(),Rc::new(RefCell::new(bytes)));});}
pub fn clear_outputs(){FILES.with(|files|files.borrow_mut().retain(|key,_|key=="base.bin"));}
pub fn take(path:&str)->io::Result<Vec<u8>>{FILES.with(|files|{
    let data=files.borrow_mut().remove(path).ok_or_else(||invalid("missing output"))?;
    Rc::try_unwrap(data).map(|value|value.into_inner()).map_err(|_|invalid("output still open"))
})}
pub struct File{data:Rc<RefCell<Vec<u8>>>,at:usize,writable:bool}
pub struct Metadata{length:u64}
impl Metadata{pub fn len(&self)->u64{self.length}}
impl File{
    pub fn open(path:impl AsRef<Path>)->io::Result<Self>{FILES.with(|files|{
        let key=path.as_ref().to_str().ok_or_else(||invalid("invalid path"))?;
        let data=files.borrow().get(key).cloned().ok_or_else(||invalid("unknown memory resource"))?;
        Ok(Self{data,at:0,writable:false})
    })}
    pub fn create(path:impl AsRef<Path>)->io::Result<Self>{FILES.with(|files|{
        let key=path.as_ref().to_str().ok_or_else(||invalid("invalid path"))?;
        if !["output/results.bin","output/events.bin","output/summary.json"].contains(&key){return Err(invalid("output path not allowed"));}
        let data=Rc::new(RefCell::new(Vec::new()));files.borrow_mut().insert(key.into(),data.clone());
        Ok(Self{data,at:0,writable:true})
    })}
    pub fn metadata(&self)->io::Result<Metadata>{Ok(Metadata{length:self.data.borrow().len()as u64})}
}
impl Read for File{fn read(&mut self,out:&mut[u8])->io::Result<usize>{let data=self.data.borrow();let count=out.len().min(data.len().saturating_sub(self.at));out[..count].copy_from_slice(&data[self.at..self.at+count]);self.at+=count;Ok(count)}}
impl Write for File{
    fn write(&mut self,bytes:&[u8])->io::Result<usize>{if !self.writable{return Err(invalid("snapshot is read only"));}self.data.borrow_mut().extend_from_slice(bytes);self.at+=bytes.len();Ok(bytes.len())}
    fn flush(&mut self)->io::Result<()>{Ok(())}
}
pub mod fs{pub fn create_dir_all(path:impl AsRef<std::path::Path>)->std::io::Result<()>{if path.as_ref()==std::path::Path::new("output"){Ok(())}else{Err(super::invalid("directory not allowed"))}}}
pub fn env_var(_name:&str)->Result<String,std::env::VarError>{Err(std::env::VarError::NotPresent)}
#[wasm_bindgen] extern "C" {#[wasm_bindgen(js_namespace=performance,js_name=now)] fn now()->f64;}
#[derive(Clone,Copy)] pub struct Instant(f64);
impl Instant{pub fn now()->Self{Self(now())}pub fn elapsed(&self)->std::time::Duration{std::time::Duration::from_secs_f64(((now()-self.0)/1000.).max(0.))}}
