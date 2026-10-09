//! Bounded-memory validation and loading of empirical source-major topology.
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn u64_value(reader: &mut impl Read) -> Result<u64> {
    let mut data = [0u8; 8];
    reader.read_exact(&mut data)?;
    Ok(u64::from_le_bytes(data))
}
fn u32_value(reader: &mut impl Read) -> Result<u32> {
    let mut data = [0u8; 4];
    reader.read_exact(&mut data)?;
    Ok(u32::from_le_bytes(data))
}
pub struct BinaryCsr {
    pub source: Vec<usize>,
    pub target: Vec<usize>,
    pub columns: Vec<Vec<f64>>,
}
// Validation scans values without allocating edge-sized arrays. Execution may
// materialize them in the reference engine; AOT copies them into its instance.
pub fn read_binary_csr(
    path: &Path,
    sources: usize,
    targets: usize,
    edges: usize,
    columns: usize,
    sha256: &str,
    materialize: bool,
) -> Result<BinaryCsr> {
    if sources == 0
        || targets == 0
        || sources > 1_000_000
        || targets > 1_000_000
        || edges == 0
        || edges > 100_000_000
        || columns > 128
    {
        return Err("invalid binary CSR shape".into());
    }
    let mut file = File::open(path)?;
    let expected = 40u64 + (sources as u64 + 1) * 8 + edges as u64 * (4 + 8 * columns as u64);
    if file.metadata()?.len() != expected {
        return Err("binary CSR file length mismatch".into());
    }
    let mut digest = Sha256::new();
    let mut chunk = [0u8; 65536];
    loop {
        let count = file.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        digest.update(&chunk[..count]);
    }
    if format!("{:x}", digest.finalize()) != sha256.to_ascii_lowercase() {
        return Err("binary CSR checksum mismatch".into());
    }
    file.seek(SeekFrom::Start(0))?;
    let mut reader = BufReader::with_capacity(65536, file);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"B2CSR001"
        || u64_value(&mut reader)? != sources as u64
        || u64_value(&mut reader)? != targets as u64
        || u64_value(&mut reader)? != edges as u64
        || u64_value(&mut reader)? != columns as u64
    {
        return Err("binary CSR header mismatch".into());
    }
    let mut source = Vec::new();
    if materialize {
        source.try_reserve_exact(edges)?;
    }
    let mut previous = u64_value(&mut reader)?;
    if previous != 0 {
        return Err("binary CSR offsets must start at zero".into());
    }
    for neuron in 0..sources {
        let next = u64_value(&mut reader)?;
        if next < previous || next > edges as u64 {
            return Err("invalid binary CSR offsets".into());
        }
        if materialize {
            source.resize(next as usize, neuron);
        }
        previous = next;
    }
    if previous != edges as u64 {
        return Err("binary CSR final offset mismatch".into());
    }
    let mut target = Vec::new();
    if materialize {
        target.try_reserve_exact(edges)?;
    }
    for _ in 0..edges {
        let value = u32_value(&mut reader)? as usize;
        if value >= targets {
            return Err("binary CSR target outside population".into());
        }
        if materialize {
            target.push(value);
        }
    }
    let mut values = Vec::new();
    for _ in 0..columns {
        let mut column = Vec::new();
        if materialize {
            column.try_reserve_exact(edges)?;
        }
        for _ in 0..edges {
            let value = f64::from_bits(u64_value(&mut reader)?);
            if !value.is_finite() {
                return Err("nonfinite binary CSR parameter".into());
            }
            if materialize {
                column.push(value);
            }
        }
        if materialize {
            values.push(column);
        }
    }
    Ok(BinaryCsr {
        source,
        target,
        columns: values,
    })
}
