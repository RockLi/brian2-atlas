"""Bounded batches for post-simulation projection counts and topology reports."""

PROJECTION_REPORT_BATCH_SIZE = 128


def projection_report_source():
    # All fields are integer counts or unchanged f64 bit patterns. The only
    # reduction is the existing checked usize delivery sum, in rank order.
    return r'''    let mut topology_report = String::from("[");
    let mut mpi_projection_report_collectives = 0usize;
    let mut mpi_projection_report_packet_values = 0usize;
    for (batch_index, batch) in mpi_projections.chunks_mut(BATCH_PROJECTIONS).enumerate() {
        let mut records = Vec::<u64>::with_capacity(batch.len()*5);
        for (index, projection) in batch.iter().enumerate() {
            let q = batch_index*BATCH_PROJECTIONS + index;
            let owner = MPI_POPULATION_OWNERS[MPI_PROJECTION_TARGETS[q]];
            let draws = if owner < 0 { projection.edge_count*(mpi.rank+1)/mpi.size-projection.edge_count*mpi.rank/mpi.size } else if mpi.rank == owner as usize { projection.edge_count } else { 0usize };
            let csr_bytes = projection.offsets.capacity().checked_mul(std::mem::size_of::<usize>()).ok_or("MPI CSR byte count overflow")?;
            records.extend_from_slice(&[u64::try_from(projection.delivered)?, u64::try_from(projection.local_edge_count)?, u64::try_from(draws)?, projection.build_seconds.to_bits(), u64::try_from(csr_bytes)?]);
        }
        let gathered = mpi.gather(&records)?;
        mpi_projection_report_collectives += 1;
        mpi_projection_report_packet_values = mpi_projection_report_packet_values.max(records.len());
        for (index, projection) in batch.iter_mut().enumerate() {
            let q = batch_index*BATCH_PROJECTIONS + index;
            let base = index*5;
            projection.delivered = gathered.chunks_exact(records.len()).try_fold(0usize, |total, row| {
                total.checked_add(usize::try_from(row[base])?).ok_or_else(|| -> Box<dyn std::error::Error> { "MPI count overflow".into() })
            })?;
            let topology_stats: Vec<u64> = gathered.chunks_exact(records.len()).flat_map(|row| [row[base+1], row[base+2]]).collect();
            let topology_seconds: Vec<f64> = gathered.chunks_exact(records.len()).map(|row| f64::from_bits(row[base+3])).collect();
            let topology_csr_bytes: Vec<u64> = gathered.chunks_exact(records.len()).map(|row| row[base+4]).collect();
            let owner = MPI_POPULATION_OWNERS[MPI_PROJECTION_TARGETS[q]];
            if q > 0 { topology_report.push(','); }
            topology_report.push_str(&format!("{{\"projection\":{},\"global_edges\":{},\"construction\":\"{}\",\"rank_stats\":{:?},\"build_seconds\":{:?},\"rank_csr_offset_bytes\":{:?}}}", q, projection.edge_count, if owner < 0 { "distributed-draw-ranges" } else { "target-owner-local" }, topology_stats, topology_seconds, topology_csr_bytes));
        }
    }
    topology_report.push_str("]");'''.replace('BATCH_PROJECTIONS', str(PROJECTION_REPORT_BATCH_SIZE))
