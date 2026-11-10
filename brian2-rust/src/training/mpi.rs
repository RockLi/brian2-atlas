//! Optional native MPI lifecycle. Any rank failure aborts the whole request;
//! the Python coordinator only commits after successful MPI finalization.
use super::{ensure, Result};
use std::ffi::{c_char, c_void, CString};

#[cfg_attr(target_os = "macos", link(name = "System"))]
#[cfg_attr(not(target_os = "macos"), link(name = "dl"))]
extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}
pub struct Context {
    handle: *mut c_void,
    pub rank: usize,
    pub size: usize,
    sum: unsafe extern "C" fn(*mut f64, u64) -> i32,
    agree: unsafe extern "C" fn(*const u8) -> i32,
    finish: unsafe extern "C" fn() -> i32,
    abort: unsafe extern "C" fn(),
    done: bool,
}
impl Context {
    pub fn from_environment() -> Result<Option<Self>> {
        let path = match std::env::var("B2_TRAIN_MPI_LIB") {
            Ok(path) => path,
            Err(_) => {
                for name in ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"] {
                    ensure(
                        std::env::var(name)
                            .ok()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(1)
                            <= 1,
                        "MPI launch requires B2_TRAIN_MPI_LIB",
                    )?;
                }
                return Ok(None);
            }
        };
        let path = CString::new(path)?;
        let handle = unsafe { dlopen(path.as_ptr(), 2) };
        ensure(!handle.is_null(), "cannot load native MPI training library")?;
        let symbols = [
            "b2_train_mpi_init",
            "b2_train_mpi_sum",
            "b2_train_mpi_agree",
            "b2_train_mpi_finish",
            "b2_train_mpi_abort",
        ];
        let mut pointers = Vec::new();
        for name in symbols {
            let name = CString::new(name)?;
            let pointer = unsafe { dlsym(handle, name.as_ptr()) };
            if pointer.is_null() {
                unsafe {
                    dlclose(handle);
                }
                return Err("native MPI training ABI symbol missing".into());
            }
            pointers.push(pointer);
        }
        let init: unsafe extern "C" fn(*mut i32, *mut i32) -> i32 =
            unsafe { std::mem::transmute(pointers[0]) };
        let (mut rank, mut size) = (0, 0);
        ensure(
            unsafe { init(&mut rank, &mut size) } == 0,
            "MPI initialization failed",
        )?;
        Ok(Some(Self {
            handle,
            rank: rank as usize,
            size: size as usize,
            sum: unsafe { std::mem::transmute(pointers[1]) },
            agree: unsafe { std::mem::transmute(pointers[2]) },
            finish: unsafe { std::mem::transmute(pointers[3]) },
            abort: unsafe { std::mem::transmute(pointers[4]) },
            done: false,
        }))
    }
    pub fn owns(&self, index: usize, count: usize) -> bool {
        index * self.size / count == self.rank
    }
    pub fn sum(&self, values: &mut [f64]) -> Result<()> {
        ensure(
            unsafe { (self.sum)(values.as_mut_ptr(), values.len() as u64) } == 0,
            "MPI ordered reduction failed",
        )
    }
    pub fn agree(&self, digest: &[u8; 32]) -> Result<()> {
        ensure(
            unsafe { (self.agree)(digest.as_ptr()) } == 0,
            "MPI request/runtime identity mismatch",
        )
    }
    pub fn finish(&mut self) -> Result<()> {
        ensure(unsafe { (self.finish)() } == 0, "MPI finalization failed")?;
        self.done = true;
        Ok(())
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            if !self.done {
                (self.abort)();
            }
            dlclose(self.handle);
        }
    }
}
