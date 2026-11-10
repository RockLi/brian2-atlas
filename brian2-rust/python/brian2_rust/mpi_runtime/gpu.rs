// Backend libraries are loaded only on GPU ranks. Their ABI is model-private.
type GpuCreate = unsafe extern "C" fn(u32, i32, *mut u8, usize) -> *mut std::ffi::c_void;
type GpuDestroy = unsafe extern "C" fn(*mut std::ffi::c_void);
type GpuRun = unsafe extern "C" fn(*mut std::ffi::c_void, *mut f32, u64, *mut u64, *mut u32, *mut u8, usize) -> i32;
extern "C" {
    fn dlopen(path: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
    fn dlsym(handle: *mut std::ffi::c_void, name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn dlclose(handle: *mut std::ffi::c_void) -> i32;
    fn dlerror() -> *const std::ffi::c_char;
}
struct MpiGpu {
    library: *mut std::ffi::c_void,
    create: Option<GpuCreate>, destroy: Option<GpuDestroy>, run_fn: Option<GpuRun>,
    handles: std::collections::BTreeMap<u32, *mut std::ffi::c_void>,
    device: i32, values: Vec<f32>, faults: Vec<u32>, dispatches: u64,
}
impl MpiGpu {
    fn new(rank: usize) -> Result<Self> {
        let backends = RANK_BACKENDS;
        let backend = backends[rank];
        let mut gpu = Self { library: std::ptr::null_mut(), create: None, destroy: None, run_fn: None,
            handles: std::collections::BTreeMap::new(), device: 0, values: Vec::new(), faults: Vec::new(), dispatches: 0 };
        if backend == "cpu" { return Ok(gpu); }
        let filename = if backend == "metal" { "libmpi-metal.dylib" } else { "libmpi-cuda.so" };
        gpu.device = backend.split_once(':').map(|(_, v)| v.parse::<i32>()).transpose()?.unwrap_or(0);
        let executable = std::env::current_exe()?;
        let path = executable.parent().ok_or("missing MPI executable directory")?.join(filename);
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(path.as_os_str().as_bytes())?;
        gpu.library = unsafe { dlopen(path.as_ptr(), 2) };
        if gpu.library.is_null() {
            let message = unsafe { dlerror() };
            return Err(if message.is_null() { "MPI GPU library unavailable".into() }
                else { unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy().into_owned().into() });
        }
        unsafe {
            let create = dlsym(gpu.library, b"b2gpu_create\0".as_ptr().cast());
            let destroy = dlsym(gpu.library, b"b2gpu_destroy\0".as_ptr().cast());
            let run = dlsym(gpu.library, b"b2gpu_run\0".as_ptr().cast());
            check(!create.is_null() && !destroy.is_null() && !run.is_null(), "MPI GPU library ABI mismatch")?;
            gpu.create = Some(std::mem::transmute::<*mut std::ffi::c_void, GpuCreate>(create));
            gpu.destroy = Some(std::mem::transmute::<*mut std::ffi::c_void, GpuDestroy>(destroy));
            gpu.run_fn = Some(std::mem::transmute::<*mut std::ffi::c_void, GpuRun>(run));
        }
        Ok(gpu)
    }
    fn enabled(&self) -> bool { !self.library.is_null() }
    fn prepare(&mut self, count: usize, fields: usize) -> Result<()> {
        let length = count.checked_mul(fields).ok_or("MPI GPU storage overflow")?;
        check(count <= u32::MAX as usize && length <= 128*1024*1024, "MPI GPU staging exceeds 512 MiB values limit")?;
        self.values.clear();
        self.values.try_reserve(length)?;
        self.faults.resize(count, 0);
        Ok(())
    }
    fn push(&mut self, value: f64) -> Result<()> {
        let rounded = value as f32;
        check(!value.is_finite() || rounded.is_finite(), "MPI GPU input exceeds float32 range")?;
        self.values.push(rounded);
        Ok(())
    }
    fn run(&mut self, kernel: u32, mut meta: [u64; 5]) -> Result<()> {
        let mut error = [0u8; 8192];
        if !self.handles.contains_key(&kernel) {
            let handle = unsafe { self.create.ok_or("missing GPU create")?(kernel, self.device, error.as_mut_ptr(), error.len()) };
            if handle.is_null() { return Err(format!("MPI GPU initialization failed: {}", gpu_error(&error)).into()); }
            self.handles.insert(kernel, handle);
        }
        self.faults.fill(0);
        let rc = unsafe { self.run_fn.ok_or("missing GPU dispatch")?(self.handles[&kernel], self.values.as_mut_ptr(),
            self.values.len() as u64, meta.as_mut_ptr(), self.faults.as_mut_ptr(), error.as_mut_ptr(), error.len()) };
        if rc != 0 { return Err(format!("MPI GPU execution failed: {}", gpu_error(&error)).into()); }
        check(self.faults.iter().all(|&v| v == 0), "MPI GPU arithmetic fault")?;
        self.dispatches += 1;
        Ok(())
    }
}
fn gpu_error(error: &[u8]) -> String {
    String::from_utf8_lossy(&error[..error.iter().position(|&c| c == 0).unwrap_or(error.len())]).into_owned()
}
impl Drop for MpiGpu {
    fn drop(&mut self) {
        if let Some(destroy) = self.destroy { for &handle in self.handles.values() { unsafe { destroy(handle); } } }
        if !self.library.is_null() { unsafe { dlclose(self.library); } }
    }
}
