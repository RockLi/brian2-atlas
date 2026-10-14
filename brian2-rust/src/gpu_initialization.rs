//! Export canonical host initialization without running any simulation tick.
use super::*;

fn indices(output: &mut Vec<u8>, values: &[usize]) -> Result<serde_json::Value> {
    let offset = output.len();
    for &value in values {
        output.extend_from_slice(&u32::try_from(value)?.to_le_bytes());
    }
    Ok(serde_json::json!({"offset":offset,"length":values.len(),"dtype":"<u4"}))
}

fn floats(output: &mut Vec<u8>, values: &[f64]) -> Result<serde_json::Value> {
    let offset = output.len();
    for &value in values {
        check(value.is_finite(), "non-finite GPU initializer result")?;
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(serde_json::json!({"offset":offset,"length":values.len(),"dtype":"<f8"}))
}

pub(super) fn export(model: &Model, directory: &Path, budget: usize) -> Result<()> {
    // Check the total binary payload before allocating any projection. Rust
    // construction and Python/GPU working copies need additional host memory.
    let mut bytes = 0usize;
    for (def, inst) in model
        .definition
        .synapses
        .iter()
        .zip(&model.instance.synapses)
    {
        if matches!(inst.topology, TopologyInstance::Explicit) {
            continue;
        }
        let edges = inst.edge_count();
        let parameters = def
            .parameters
            .iter()
            .filter(|p| p.index_domain == IndexDomain::Synapse)
            .count();
        let width = 8usize
            .checked_add(
                parameters
                    .checked_mul(8)
                    .ok_or("initializer size overflow")?,
            )
            .ok_or("initializer size overflow")?;
        bytes = bytes
            .checked_add(
                edges
                    .checked_mul(width)
                    .ok_or("initializer size overflow")?,
            )
            .ok_or("initializer size overflow")?;
        for path in &inst.pathways {
            let count = if path.delay_initializer.is_some() {
                edges
            } else {
                path.delay_ticks.len()
            };
            bytes = bytes
                .checked_add(count.checked_mul(4).ok_or("initializer size overflow")?)
                .ok_or("initializer size overflow")?;
        }
    }
    check(
        bytes <= budget,
        "GPU initialization payload exceeds configured byte budget",
    )?;
    fs::create_dir(directory)?;
    let mut projections = Vec::new();
    for (q, (def, inst)) in model
        .definition
        .synapses
        .iter()
        .zip(&model.instance.synapses)
        .enumerate()
    {
        if matches!(inst.topology, TopologyInstance::Explicit) {
            continue;
        }
        let (source, target, parameters, delays) =
            executor::materialize_projection(&model.definition, def, inst)?;
        let mut data = Vec::new();
        let source_info = indices(&mut data, &source)?;
        let target_info = indices(&mut data, &target)?;
        let mut parameter_info = BTreeMap::new();
        for (name, values) in parameters {
            parameter_info.insert(name, floats(&mut data, &values)?);
        }
        let mut delay_info = Vec::new();
        for values in delays {
            // Preserve the validated initializer's actual rounded ticks. A
            // continuous bound below 1_000_001 can round to 1_000_001.
            delay_info.push(indices(&mut data, &values)?);
        }
        let name = format!("projection-{q}.bin");
        fs::write(directory.join(&name), &data)?;
        projections.push(serde_json::json!({"projection":q,"file":name,"bytes":data.len(),
            "sha256":format!("{:x}",Sha256::digest(&data)),"source":source_info,"target":target_info,
            "parameters":parameter_info,"delays":delay_info,"edge_count":source.len()}));
    }
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"b2-gpu-initialization-v0","projections":projections
        }))?,
    )?;
    Ok(())
}
