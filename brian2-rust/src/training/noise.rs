//! Counter-addressed Gaussian samples. No mutable draw order or rank identity:
//! forward, VJP, sequence chunks and MPI partitions address the same sample.
//! Version 1: nested SplitMix64 finalizers and Box-Muller (cosine branch).
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

pub(super) fn normal(seed: u64, sequence: u64, batch: u64, layer: u64, neuron: u64, tick: u64, stream: u64) -> f64 {
    let mut key = mix(seed ^ 0x42325344454e3031);
    for value in [sequence, batch, layer, neuron, tick, stream] {
        key = mix(key ^ mix(value.wrapping_add(0x9e3779b97f4a7c15)));
    }
    let unit = |x: u64| ((x >> 11) as f64 + 0.5) * (1.0 / 9007199254740992.0);
    let u = unit(mix(key ^ 0xa0761d6478bd642f));
    let v = unit(mix(key ^ 0xe7037ed1a0b428db));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// Version 1 uniform stream, domain-separated from the existing Gaussian RNG.
/// Exactly 53 random bits map to [0, 1); zero is allowed, one is not.
pub(super) fn uniform(seed: u64, sequence: u64, batch: u64, layer: u64, neuron: u64, tick: u64, stream: u64) -> f64 {
    let mut key = mix(seed ^ 0x4232554e49463031);
    for value in [sequence, batch, layer, neuron, tick, stream] {
        key = mix(key ^ mix(value.wrapping_add(0x9e3779b97f4a7c15)));
    }
    (mix(key ^ 0xa0761d6478bd642f) >> 11) as f64 * (1.0 / 9007199254740992.0)
}

pub(super) fn sample(uniform_draw: bool, seed: u64, sequence: u64, batch: u64, layer: u64, neuron: u64, tick: u64, stream: u64) -> f64 {
    if uniform_draw { uniform(seed, sequence, batch, layer, neuron, tick, stream) }
    else { normal(seed, sequence, batch, layer, neuron, tick, stream) }
}

pub(super) fn device_sample(value: f64, uniform_draw: bool) -> f32 {
    let rounded = value as f32;
    if uniform_draw { rounded.min(f32::from_bits(1.0f32.to_bits() - 1)) } else { rounded }
}

/// Fresh events are addressed by emission tick, independently of their arrival
/// generation. Imported Brian queue entries have persistent snapshot-local IDs.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventAddress {
    pub delay: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending: Option<u64>,
}

pub(super) fn event_sample(address: Option<&EventAddress>, uniform_draw: bool,
    seed: u64, sequence: u64, batch: u64, domain: u64, entity: u64, tick: u64, stream: u64) -> f64 {
    let Some(address) = address else { return sample(uniform_draw, seed, sequence, batch, domain, entity, tick, stream); };
    let mut key = mix(seed ^ if uniform_draw { 0x4232455655303031 } else { 0x423245564e303031 });
    // Wrapping encodes emissions before sequence tick zero without colliding
    // with the supported nonnegative simulation tick range (at most 2^53).
    let emitted = if address.pending.is_some() { 0 } else { tick.wrapping_sub(address.delay) };
    for value in [sequence, batch, domain, entity, emitted,
                  u64::from(address.pending.is_some()), address.pending.unwrap_or(0), stream] {
        key = mix(key ^ mix(value.wrapping_add(0x9e3779b97f4a7c15)));
    }
    if uniform_draw {
        (mix(key ^ 0xa0761d6478bd642f) >> 11) as f64 * (1.0 / 9007199254740992.0)
    } else {
        let unit = |x: u64| ((x >> 11) as f64 + 0.5) * (1.0 / 9007199254740992.0);
        let u = unit(mix(key ^ 0xa0761d6478bd642f));
        let v = unit(mix(key ^ 0xe7037ed1a0b428db));
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn event_identity_follows_emission_or_imported_entry() {
        for uniform in [false, true] {
            let draw = |address: Option<&EventAddress>, tick| event_sample(address, uniform, 731, 9, 2, 5, 3, tick, 0);
            let three = EventAddress { delay: 3, pending: None };
            let five = EventAddress { delay: 5, pending: None };
            // The same emission arriving by a different route has the same draw.
            assert_eq!(draw(Some(&three), 10), draw(Some(&five), 12));
            // Distinct emissions that coincide at arrival stay independent.
            assert_ne!(draw(Some(&three), 10), draw(Some(&five), 10));
            let imported = EventAddress { delay: 0, pending: Some(1) };
            let other = EventAddress { delay: 0, pending: Some(2) };
            assert_eq!(draw(Some(&imported), 0), draw(Some(&imported), 100));
            assert_ne!(draw(Some(&imported), 0), draw(Some(&other), 0));
            assert_ne!(draw(Some(&imported), 0), draw(Some(&three), 3));
            assert_ne!(draw(Some(&three), 0), draw(Some(&three), 1u64 << 53));
            assert_eq!(draw(None, 10), sample(uniform, 731, 9, 2, 5, 3, 10, 0));
        }
    }
    #[test]
    fn uniform_device_range_keeps_half_open_endpoint() {
        let below_one = f64::from_bits(1.0f64.to_bits()-1);
        assert_eq!(below_one as f32, 1.0);
        assert_eq!(device_sample(below_one, true), f32::from_bits(1.0f32.to_bits()-1));
        assert_eq!(device_sample(0.0, true), 0.0);
        assert_eq!(device_sample(0.5, true), 0.5);
        assert_eq!(device_sample(below_one, false), 1.0);
    }
    #[test]
    fn uniform_is_reproducible_half_open_and_domain_separated() {
        let mut sum = 0.; let mut squares = 0.;
        for tick in 0..20000 {
            let u = uniform(u64::MAX, u64::MAX-1, 7, 11, 13, tick, 15);
            assert!(u >= 0. && u < 1.);
            assert_eq!(u, sample(true, u64::MAX, u64::MAX-1, 7, 11, 13, tick, 15));
            assert_ne!(u, normal(u64::MAX, u64::MAX-1, 7, 11, 13, tick, 15));
            sum += u; squares += u*u;
        }
        let mean = sum/20000.;
        assert!((mean-0.5).abs() < 0.01);
        assert!((squares/20000.-mean*mean-1./12.).abs() < 0.003);
    }
}
