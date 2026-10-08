//! Authenticate GPU checkpoint counts with the selected device sampler.
//! Host work is structural/key staging only. Each dispatch has at most 100
//! records, hence at most MAX_WORK uniforms even before work is inspected.
use super::*;
pub(super) fn profile(plan:&Plan)->&'static str {
    if plan.backend=="cuda" {"native-cuda-poisson-f32-v1"} else {"native-metal-poisson-f32-v1"}
}
#[cfg(not(unix))]
pub(super) fn validate(_: &Plan,_:&poisson_cache::Checkpoint)->Result<usize>{
    Err("GPU Poisson checkpoint validation requires a Unix host".into())
}
#[cfg(unix)]
pub(super) fn validate(plan:&Plan,state:&poisson_cache::Checkpoint)->Result<usize>{
    use std::ffi::{c_char,c_void,CStr,CString};
    ensure(state.entries.len().checked_mul(768).and_then(|n|n.checked_add(4096))
        .is_some_and(|n|n<=plan.max_tape_bytes),"GPU Poisson checkpoint validation budget exceeded")?;
    #[cfg_attr(target_os="macos",link(name="System"))]
    #[cfg_attr(not(target_os="macos"),link(name="dl"))]
    extern "C" {fn dlopen(p:*const c_char,f:i32)->*mut c_void;fn dlsym(h:*mut c_void,s:*const c_char)->*mut c_void;fn dlclose(h:*mut c_void)->i32;}
    struct Library(*mut c_void);impl Drop for Library{fn drop(&mut self){unsafe{dlclose(self.0);}}}
    let path=CString::new(std::env::var(if plan.backend=="cuda" {"B2_TRAIN_CUDA_LIB"}else{"B2_TRAIN_METAL_LIB"})
        .map_err(|_|"GPU Poisson checkpoint requires the selected native library")?)?;
    let handle=unsafe{dlopen(path.as_ptr(),2)};ensure(!handle.is_null(),"cannot load GPU Poisson checkpoint library")?;
    let _library=Library(handle);
    let cap=CString::new("b2_train_poisson_checkpoint_v1")?;let function=unsafe{dlsym(handle,cap.as_ptr())};
    ensure(!function.is_null(),"GPU Poisson checkpoint capability missing")?;
    let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
    ensure(unsafe{version()}==1,"GPU Poisson checkpoint capability mismatch")?;
    let symbol=CString::new(if plan.backend=="cuda" {"b2_train_cuda_poisson_validate_v1"}else{"b2_train_metal_poisson_validate_v1"})?;
    let function=unsafe{dlsym(handle,symbol.as_ptr())};ensure(!function.is_null(),"GPU Poisson checkpoint validation symbol missing")?;
    type Kernel=unsafe extern "C" fn(u64,*const u64,*const f32,*const i32,*mut u32,*mut u32,*mut c_char,usize)->i32;
    let kernel:Kernel=unsafe{std::mem::transmute(function)};
    let mut work=0u64;let mut dispatches=0usize;
    for chunk in state.entries.chunks(100){
        let keys=chunk.iter().map(|e|e.identity.site.key(state.seed,state.sequence,e.identity.batch,e.identity.instant)).collect::<Vec<_>>();
        let rates=chunk.iter().map(|e|e.rate as f32).collect::<Vec<_>>();
        let expected=chunk.iter().map(|e|e.count).collect::<Vec<_>>();
        let mut draws=vec![0u32;chunk.len()];let mut errors=vec![0u32;chunk.len()];let mut message=vec![0 as c_char;2048];
        let code=unsafe{kernel(chunk.len() as u64,keys.as_ptr(),rates.as_ptr(),expected.as_ptr(),draws.as_mut_ptr(),errors.as_mut_ptr(),message.as_mut_ptr(),message.len())};
        ensure(code==0,&format!("GPU Poisson checkpoint device validation failed: {}",unsafe{CStr::from_ptr(message.as_ptr())}.to_string_lossy()))?;
        dispatches+=1;
        for (&draws,&error) in draws.iter().zip(&errors){
            ensure(error==0 && draws<=100000,"GPU Poisson checkpoint count mismatch or invalid device sample")?;
            work=work.checked_add(draws as u64).ok_or("GPU Poisson checkpoint work overflow")?;
            ensure(work<=clock::MAX_WORK,"GPU Poisson checkpoint work budget exceeded")?;
        }
    }
    Ok(dispatches)
}
