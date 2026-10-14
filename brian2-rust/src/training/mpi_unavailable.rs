//! Keep CPU training and other library targets independent of a Unix MPI ABI.
use super::Result;
pub struct Context {
    pub rank: usize,
    pub size: usize,
}
impl Context {
    pub fn from_environment() -> Result<Option<Self>> {
        if std::env::var_os("B2_TRAIN_MPI_LIB").is_some() {
            Err("native MPI training currently requires Unix".into())
        } else {
            Ok(None)
        }
    }
    pub fn owns(&self, index: usize, count: usize) -> bool {
        index * self.size / count == self.rank
    }
    pub fn sum(&self, _: &mut [f64]) -> Result<()> {
        Err("native MPI training unavailable".into())
    }
    pub fn agree(&self, _: &[u8; 32]) -> Result<()> {
        Err("native MPI training unavailable".into())
    }
    pub fn finish(&mut self) -> Result<()> {
        Err("native MPI training unavailable".into())
    }
}
