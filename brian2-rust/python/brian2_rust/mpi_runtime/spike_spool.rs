// Fixed-size population buffers and a shared cumulative disk-byte budget.
// Kept after successful copies until explicit cleanup; failed runs retain evidence.
const MPI_SPOOL_BUFFER: usize = 65536;
const MPI_SPOOL_DIRTY: u64 = 1048576;

#[cfg(target_os="linux")]
fn mpi_spool_advise(file: &std::fs::File, offset: u64, length: u64, advice: i32) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    unsafe extern "C" { fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32; }
    let offset = i64::try_from(offset).map_err(std::io::Error::other)?;
    let length = i64::try_from(length).map_err(std::io::Error::other)?;
    let code = unsafe { posix_fadvise(file.as_raw_fd(), offset, length, advice) };
    if code != 0 { return Err(std::io::Error::from_raw_os_error(code)); }
    Ok(())
}
#[cfg(not(target_os="linux"))]
fn mpi_spool_advise(_: &std::fs::File, _: u64, _: u64, _: i32) -> std::io::Result<()> {
    Err(std::io::Error::other("bounded MPI spool requires Linux cache release"))
}

fn mpi_spool_release(file: &std::fs::File, offset: u64, length: u64) -> std::io::Result<()> {
    mpi_spool_advise(file,offset,length,4)
}

struct MpiSpoolBudget { maximum: u64, maximum_population: u64, used: std::cell::Cell<u64> }
impl MpiSpoolBudget {
    fn new(maximum: u64) -> std::io::Result<std::rc::Rc<Self>> {
        Self::with_population_limit(maximum,maximum)
    }
    fn with_population_limit(maximum: u64, maximum_population: u64) -> std::io::Result<std::rc::Rc<Self>> {
        if maximum_population==0 || maximum_population>maximum {
            return Err(std::io::Error::other("MPI spool population byte budget must be in 1..total budget"));
        }
        if !cfg!(target_os="linux") || usize::BITS != 64 || maximum == 0 || maximum > 512*1024*1024*1024 {
            return Err(std::io::Error::other("MPI spool requires Linux64 and a byte budget in 1..512 GiB"));
        }
        Ok(std::rc::Rc::new(Self { maximum, maximum_population, used: std::cell::Cell::new(0) }))
    }
    fn reserve(&self) -> std::io::Result<()> {
        let next = self.used.get().checked_add(8).ok_or_else(|| std::io::Error::other("MPI spool count overflow"))?;
        if next > self.maximum { return Err(std::io::Error::other("MPI spool byte budget exceeded before write")); }
        self.used.set(next);
        Ok(())
    }
}

struct MpiSpikeSpool {
    path: std::path::PathBuf,
    budget: std::rc::Rc<MpiSpoolBudget>,
    writer: std::cell::RefCell<Option<std::io::BufWriter<std::fs::File>>>,
    records: usize,
    synced: std::cell::Cell<u64>,
    poisoned: bool,
    removed: bool,
}
impl MpiSpikeSpool {
    fn new(path: std::path::PathBuf, budget: std::rc::Rc<MpiSpoolBudget>) -> Self {
        Self { path, budget, writer: std::cell::RefCell::new(None), records: 0,
            synced: std::cell::Cell::new(0), poisoned: false, removed: false }
    }
    fn len(&self) -> usize { self.records }
    fn capacity(&self) -> usize { if self.writer.borrow().is_some() { MPI_SPOOL_BUFFER/8 } else { 0 } }
    fn push(&mut self, value: (u32,u32)) -> std::io::Result<()> {
        if self.poisoned || self.removed { return Err(std::io::Error::other("MPI spool is poisoned or removed")); }
        if self.budget.maximum_population < self.budget.maximum
                && (self.records as u64+1)*8 > self.budget.maximum_population {
            return Err(std::io::Error::other("MPI spool population byte budget exceeded before write"));
        }
        self.budget.reserve()?;
        // A partial write must never be replayed or counted as a complete record.
        self.poisoned = true;
        let writer = self.writer.get_mut();
        if writer.is_none() {
            let file = std::fs::OpenOptions::new().write(true).create_new(true).open(&self.path)?;
            *writer = Some(std::io::BufWriter::with_capacity(MPI_SPOOL_BUFFER,file));
        }
        let writer = writer.as_mut().unwrap();
        let mut bytes = [0u8;8];
        bytes[..4].copy_from_slice(&value.0.to_le_bytes());
        bytes[4..].copy_from_slice(&value.1.to_le_bytes());
        writer.write_all(&bytes)?;
        self.records += 1;
        if self.records as u64*8-self.synced.get() >= MPI_SPOOL_DIRTY {
            writer.flush()?;
            writer.get_ref().sync_data()?;
            mpi_spool_release(writer.get_ref(),self.synced.get(),self.records as u64*8-self.synced.get())?;
            self.synced.set(self.records as u64*8);
        }
        self.poisoned = false;
        Ok(())
    }
    fn extend<I: IntoIterator<Item=(u32,u32)>>(&mut self, values: I) -> std::io::Result<()> {
        for value in values { self.push(value)?; }
        Ok(())
    }
    fn copy_to<W:std::io::Write>(&self, target: &mut W) -> std::io::Result<()> {
        use std::io::Read;
        if self.poisoned || self.removed { return Err(std::io::Error::other("MPI spool is poisoned or removed")); }
        if let Some(writer) = self.writer.borrow_mut().as_mut() {
            writer.flush()?;
            writer.get_ref().sync_data()?;
            mpi_spool_release(writer.get_ref(),0,0)?;
        } else if self.records == 0 { return Ok(()); }
        let mut file = std::fs::File::open(&self.path)?;
        // Asynchronous read-ahead can repopulate pages after per-block release.
        // Disable it for this explicit buffered copy and evict the completed file.
        mpi_spool_advise(&file,0,0,1)?;
        let expected = self.records as u64*8;
        if file.metadata()?.len() != expected { return Err(std::io::Error::other("MPI spool size mismatch")); }
        let mut buffer = [0u8;65536];
        let mut copied = 0u64;
        while copied < expected {
            let count = (expected-copied).min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..count])?;
            target.write_all(&buffer[..count])?;
            copied += count as u64;
            mpi_spool_release(&file,copied-count as u64,count as u64)?;
        }
        mpi_spool_release(&file,0,0)?;
        Ok(())
    }
    fn copy_final(&mut self, target: &mut MpiBoundedOutput, directory: &std::path::Path) -> std::io::Result<()> {
        self.copy_to(target)?;
        // The complete prefix and its directory entry must survive before the
        // sole temporary copy is removed. Failures retain the current spool.
        target.persist_prefix(directory)?;
        self.remove()
    }
    fn remove(&mut self) -> std::io::Result<()> {
        if self.poisoned || self.removed { return Err(std::io::Error::other("MPI spool is poisoned or removed")); }
        if let Some(mut writer) = self.writer.get_mut().take() {
            writer.flush()?;
            drop(writer);
            std::fs::remove_file(&self.path)?;
        }
        self.removed = true;
        Ok(())
    }
}

// Final output writes also need a cache bound: bounded source reads alone do
// not prevent Linux charging many GiB of dirty destination pages to the job.
struct MpiBoundedOutput {
    writer: std::io::BufWriter<std::fs::File>,
    maximum: u64,
    written: u64,
    released: u64,
}
impl MpiBoundedOutput {
    fn new(file: std::fs::File, maximum: usize) -> Self {
        Self { writer: std::io::BufWriter::with_capacity(MPI_SPOOL_BUFFER,file),
            maximum: maximum as u64, written: 0, released: 0 }
    }
    fn persist_prefix(&mut self, directory: &std::path::Path) -> std::io::Result<()> {
        self.flush()?;
        if self.writer.get_ref().metadata()?.len()!=self.written {
            return Err(std::io::Error::other("MPI final output prefix size mismatch"));
        }
        std::fs::File::open(directory)?.sync_all()?;
        Ok(())
    }
}
impl std::io::Write for MpiBoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.maximum-self.written {
            return Err(std::io::Error::other("MPI final output byte budget exceeded before write"));
        }
        let size=bytes.len().min(MPI_SPOOL_BUFFER);
        self.writer.write_all(&bytes[..size])?;
        self.written+=size as u64;
        if self.written-self.released>=64*1024*1024 { self.flush()?; }
        Ok(size)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        mpi_spool_release(self.writer.get_ref(),self.released,self.written-self.released)?;
        // Keep the last partial page in the next release interval.
        self.released=self.written/4096*4096;
        Ok(())
    }
}
