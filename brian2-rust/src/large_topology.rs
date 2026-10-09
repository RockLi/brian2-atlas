//! Memory-bounded procedural construction of large static projection blocks.
//!
//! A fixed-total projection is sampled with replacement, matching the usual
//! multapse/autapse-enabled rule.  Edges are stored source-major, so the source
//! index and a second CSR edge-id array are both implicit.

use std::collections::HashSet;
use std::error::Error;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug)]
pub struct FixedTotalSpec {
    pub source_count: usize,
    pub target_count: usize,
    pub target_offset: usize,
    pub edge_count: usize,
    pub seed: u64,
    pub weight_mean: f64,
    pub weight_std: f64,
    pub excitatory: bool,
    pub delay_mean_ticks: f64,
    pub delay_std_ticks: f64,
    pub minimum_delay_ticks: u16,
}

pub struct CompactProjection {
    pub offsets: Vec<u32>,
    pub targets: Vec<u32>,
    pub weights: Vec<f32>,
    pub delay_ticks: Vec<u16>,
}

impl CompactProjection {
    pub fn resident_bytes(&self) -> usize {
        self.offsets.len() * size_of::<u32>()
            + self.targets.len() * size_of::<u32>()
            + self.weights.len() * size_of::<f32>()
            + self.delay_ticks.len() * size_of::<u16>()
    }
}

pub fn estimated_resident_bytes(source_count: usize, edge_count: usize) -> Result<usize> {
    (source_count + 1)
        .checked_mul(size_of::<u32>())
        .and_then(|offsets| {
            edge_count
                .checked_mul(size_of::<u32>() + size_of::<f32>() + size_of::<u16>())
                .and_then(|edges| offsets.checked_add(edges))
        })
        .ok_or_else(|| "projection memory estimate overflow".into())
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn draw(seed: u64, stream: u64, edge: usize, attempt: u64) -> u64 {
    mix64(
        seed ^ stream.wrapping_mul(0xd2b7_4407_b1ce_6e93)
            ^ (edge as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ attempt.wrapping_mul(0xca5a_8263_9512_1157),
    )
}

fn bounded(seed: u64, stream: u64, edge: usize, upper: usize) -> usize {
    ((draw(seed, stream, edge, 0) as u128 * upper as u128) >> 64) as usize
}

fn uniform_open(seed: u64, stream: u64, edge: usize, attempt: u64) -> f64 {
    let mantissa = draw(seed, stream, edge, attempt) >> 11;
    (mantissa as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
}

fn normal(seed: u64, stream: u64, edge: usize, attempt: u64) -> f64 {
    let u1 = uniform_open(seed, stream, edge, attempt * 2);
    let u2 = uniform_open(seed, stream + 1, edge, attempt * 2 + 1);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

pub fn deterministic_normal(seed: u64, stream: u64, index: usize) -> f64 {
    normal(seed, stream, index, 0)
}

pub fn deterministic_clipped_normal(
    seed: u64,
    stream: u64,
    index: usize,
    mean: f64,
    std: f64,
    minimum: Option<f64>,
    maximum: Option<f64>,
) -> Result<f64> {
    if !mean.is_finite()
        || !std.is_finite()
        || std < 0.0
        || minimum.is_some_and(|value| !value.is_finite())
        || maximum.is_some_and(|value| !value.is_finite())
        || minimum.zip(maximum).is_some_and(|(low, high)| low > high)
    {
        return Err("invalid clipped-normal initializer".into());
    }
    for attempt in 0..1_000 {
        let value = mean + std * normal(seed, stream, index, attempt);
        if minimum.is_none_or(|low| value >= low) && maximum.is_none_or(|high| value <= high) {
            return Ok(value);
        }
    }
    Err("clipped-normal rejection limit exceeded".into())
}

pub fn materialize_clipped_normal(
    edge_count: usize,
    seed: u64,
    stream: u64,
    mean: f64,
    std: f64,
    minimum: Option<f64>,
    maximum: Option<f64>,
) -> Result<Vec<f64>> {
    (0..edge_count)
        .map(|edge| deterministic_clipped_normal(seed, stream, edge, mean, std, minimum, maximum))
        .collect()
}

pub fn materialize_uniform(
    edge_count: usize,
    seed: u64,
    stream: u64,
    minimum: f64,
    maximum: f64,
) -> Result<Vec<f64>> {
    if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
        return Err("invalid uniform initializer".into());
    }
    Ok((0..edge_count)
        .map(|edge| minimum + (maximum - minimum) * uniform_open(seed, stream, edge, 0))
        .collect())
}

/// Build source-major endpoint arrays for a fixed-total, with-replacement rule.
///
/// The returned arrays use local population indices. Their order is stable for
/// a fixed seed and is shared by the reference and generated AOT executors.
pub fn build_fixed_total_indices(
    source_count: usize,
    target_count: usize,
    edge_count: usize,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>)> {
    if source_count == 0 || target_count == 0 || edge_count == 0 || edge_count > u32::MAX as usize {
        return Err("invalid fixed-total topology specification".into());
    }
    let mut counts = vec![0u32; source_count];
    for edge in 0..edge_count {
        let source = bounded(seed, 0, edge, source_count);
        counts[source] = counts[source]
            .checked_add(1)
            .ok_or("per-source edge count exceeds u32")?;
    }
    let mut offsets = Vec::with_capacity(source_count + 1);
    offsets.push(0u32);
    for count in counts {
        offsets.push(
            offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(count)
                .ok_or("fixed-total edge count exceeds u32")?,
        );
    }
    let mut cursor = offsets[..source_count].to_vec();
    let mut sources = vec![0usize; edge_count];
    let mut targets = vec![0usize; edge_count];
    for edge in 0..edge_count {
        let source = bounded(seed, 0, edge, source_count);
        let position = cursor[source] as usize;
        cursor[source] += 1;
        sources[position] = source;
        targets[position] = bounded(seed, 1, edge, target_count);
    }
    Ok((sources, targets))
}

fn fixed_indegree_sources(
    source_count: usize,
    indegree: usize,
    target: usize,
    seed: u64,
    mut visit: impl FnMut(usize),
) {
    // Floyd's algorithm samples a uniform subset in O(indegree) memory and
    // time. Only membership tests are used, so HashSet iteration order cannot
    // affect the generated topology.
    let mut selected = HashSet::with_capacity(indegree);
    for candidate in source_count - indegree..source_count {
        let logical = target * source_count + candidate;
        let draw = bounded(seed, 0, logical, candidate + 1);
        let source = if selected.contains(&draw) {
            candidate
        } else {
            draw
        };
        selected.insert(source);
        visit(source);
    }
}

/// Build source-major endpoints with exactly `indegree` distinct sources per target.
pub fn build_fixed_indegree_indices(
    source_count: usize,
    target_count: usize,
    indegree: usize,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let edge_count = target_count
        .checked_mul(indegree)
        .ok_or("fixed-indegree edge count overflow")?;
    if source_count == 0
        || target_count == 0
        || indegree == 0
        || indegree > source_count
        || edge_count > u32::MAX as usize
        || target_count.checked_mul(source_count).is_none()
    {
        return Err("invalid fixed-indegree topology specification".into());
    }
    let mut counts = vec![0u32; source_count];
    for target in 0..target_count {
        fixed_indegree_sources(source_count, indegree, target, seed, |source| {
            counts[source] += 1;
        });
    }
    let mut offsets = Vec::with_capacity(source_count + 1);
    offsets.push(0u32);
    for count in counts {
        offsets.push(
            offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(count)
                .expect("validated fixed-indegree edge count fits u32"),
        );
    }
    let mut cursor = offsets[..source_count].to_vec();
    let mut sources = vec![0usize; edge_count];
    let mut targets = vec![0usize; edge_count];
    for target in 0..target_count {
        fixed_indegree_sources(source_count, indegree, target, seed, |source| {
            let position = cursor[source] as usize;
            cursor[source] += 1;
            sources[position] = source;
            targets[position] = target;
        });
    }
    Ok((sources, targets))
}

fn clipped_weight(spec: &FixedTotalSpec, edge: usize) -> f32 {
    for attempt in 0..1_000 {
        let value = spec.weight_mean + spec.weight_std * normal(spec.seed, 2, edge, attempt);
        if (spec.excitatory && value >= 0.0) || (!spec.excitatory && value <= 0.0) {
            return value as f32;
        }
    }
    if spec.excitatory {
        0.0
    } else {
        -0.0
    }
}

fn clipped_delay(spec: &FixedTotalSpec, edge: usize) -> u16 {
    for attempt in 0..1_000 {
        let value =
            spec.delay_mean_ticks + spec.delay_std_ticks * normal(spec.seed, 4, edge, attempt);
        let rounded = (value + 0.5).floor();
        if rounded >= f64::from(spec.minimum_delay_ticks) && rounded <= f64::from(u16::MAX) {
            return rounded as u16;
        }
    }
    spec.minimum_delay_ticks
}

pub fn build_fixed_total(spec: FixedTotalSpec) -> Result<CompactProjection> {
    if spec.source_count == 0
        || spec.target_count == 0
        || spec.edge_count > u32::MAX as usize
        || spec.target_offset.checked_add(spec.target_count).is_none()
        || spec.target_offset + spec.target_count > u32::MAX as usize
        || !spec.weight_mean.is_finite()
        || !spec.weight_std.is_finite()
        || spec.weight_std < 0.0
        || !spec.delay_mean_ticks.is_finite()
        || !spec.delay_std_ticks.is_finite()
        || spec.delay_std_ticks < 0.0
        || spec.minimum_delay_ticks == 0
    {
        return Err("invalid fixed-total projection specification".into());
    }
    let _ = estimated_resident_bytes(spec.source_count, spec.edge_count)?;
    let mut counts = vec![0u32; spec.source_count];
    for edge in 0..spec.edge_count {
        let source = bounded(spec.seed, 0, edge, spec.source_count);
        counts[source] = counts[source]
            .checked_add(1)
            .ok_or("per-source edge count exceeds u32")?;
    }
    let mut offsets = Vec::with_capacity(spec.source_count + 1);
    offsets.push(0u32);
    for count in counts {
        offsets.push(
            offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(count)
                .ok_or("projection edge count exceeds u32")?,
        );
    }
    let mut cursor = offsets[..spec.source_count].to_vec();
    let mut targets = vec![0u32; spec.edge_count];
    let mut weights = vec![0f32; spec.edge_count];
    let mut delay_ticks = vec![0u16; spec.edge_count];
    for edge in 0..spec.edge_count {
        let source = bounded(spec.seed, 0, edge, spec.source_count);
        let position = cursor[source] as usize;
        cursor[source] += 1;
        targets[position] =
            (spec.target_offset + bounded(spec.seed, 1, edge, spec.target_count)) as u32;
        weights[position] = clipped_weight(&spec, edge);
        delay_ticks[position] = clipped_delay(&spec, edge);
    }
    Ok(CompactProjection {
        offsets,
        targets,
        weights,
        delay_ticks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(seed: u64) -> FixedTotalSpec {
        FixedTotalSpec {
            source_count: 5,
            target_count: 7,
            target_offset: 11,
            edge_count: 10_000,
            seed,
            weight_mean: -4.0,
            weight_std: 0.4,
            excitatory: false,
            delay_mean_ticks: 7.5,
            delay_std_ticks: 3.75,
            minimum_delay_ticks: 1,
        }
    }

    #[test]
    fn fixed_total_is_compact_bounded_and_reproducible() {
        let first = build_fixed_total(spec(123)).unwrap();
        let second = build_fixed_total(spec(123)).unwrap();
        assert_eq!(first.offsets, second.offsets);
        assert_eq!(first.targets, second.targets);
        assert_eq!(first.weights, second.weights);
        assert_eq!(first.delay_ticks, second.delay_ticks);
        assert_eq!(first.offsets[0], 0);
        assert_eq!(first.offsets[5], 10_000);
        assert!(first.targets.iter().all(|&value| (11..18).contains(&value)));
        assert!(first.weights.iter().all(|&value| value <= 0.0));
        assert!(first.delay_ticks.iter().all(|&value| value >= 1));
        assert_eq!(
            first.resident_bytes(),
            estimated_resident_bytes(5, 10_000).unwrap()
        );
        assert_ne!(first.targets, build_fixed_total(spec(124)).unwrap().targets);
    }

    #[test]
    fn endpoint_arrays_are_source_major_and_reproducible() {
        let first = build_fixed_total_indices(5, 7, 10_000, 123).unwrap();
        let second = build_fixed_total_indices(5, 7, 10_000, 123).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.0.len(), 10_000);
        assert!(first.0.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(first.0.iter().all(|&source| source < 5));
        assert!(first.1.iter().all(|&target| target < 7));
        assert_ne!(first, build_fixed_total_indices(5, 7, 10_000, 124).unwrap());
    }

    #[test]
    fn fixed_indegree_is_distinct_per_target_and_reproducible() {
        let first = build_fixed_indegree_indices(11, 7, 5, 123).unwrap();
        let second = build_fixed_indegree_indices(11, 7, 5, 123).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.0.len(), 35);
        assert!(first.0.windows(2).all(|pair| pair[0] <= pair[1]));
        for target in 0..7 {
            let mut sources: Vec<_> = first
                .0
                .iter()
                .zip(&first.1)
                .filter_map(|(&source, &actual)| (actual == target).then_some(source))
                .collect();
            sources.sort_unstable();
            sources.dedup();
            assert_eq!(sources.len(), 5);
        }
        assert_ne!(first, build_fixed_indegree_indices(11, 7, 5, 124).unwrap());
        assert!(build_fixed_indegree_indices(4, 2, 5, 1).is_err());
    }
}
