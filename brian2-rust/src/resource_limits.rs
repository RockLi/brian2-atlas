//! Opt-in process budgets, independent of model semantics and host admission.
use crate::Result;

fn parse(name: &str, text: &str, maximum: usize) -> Result<usize> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("{name} requires decimal digits in 1..{maximum}").into());
    }
    let value: usize = text.parse()?;
    if !(1..=maximum).contains(&value) {
        return Err(format!("{name} requires decimal digits in 1..{maximum}").into());
    }
    Ok(value)
}

fn environment(name: &str, default: usize, maximum: u64) -> Result<usize> {
    // Native process ceilings can exceed a wasm32 address space. Keep the
    // declared ceiling wide, then constrain it to this target's index width.
    let maximum = usize::try_from(maximum).unwrap_or(usize::MAX);
    match std::env::var(name) {
        Ok(text) => parse(name, &text, maximum),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

pub fn initial_values() -> Result<usize> {
    environment("B2_MAX_INITIAL_VALUES", 100_000_000, 8_000_000_000)
}

pub fn explicit_synapses() -> Result<usize> {
    environment("B2_MAX_EXPLICIT_SYNAPSES", 50_000_000, 1_000_000_000)
}

pub fn timed_array_values() -> Result<usize> {
    environment("B2_MAX_TIMED_ARRAY_VALUES", 10_000_000, 1_000_000_000)
}

pub fn ir_bytes() -> Result<usize> {
    environment("B2_MAX_IR_BYTES", 2048 * 1_048_576, 64 * 1_073_741_824)
}

pub fn neurons() -> Result<usize> {
    environment("B2_MAX_NEURONS", 1_000_000, 16_000_000)
}

pub fn population_steps() -> Result<usize> {
    environment("B2_MAX_POPULATION_STEPS", 10_000_000, 10_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_limits_are_finite_positive_decimal_integers() {
        for invalid in [
            "",
            "0",
            "-1",
            "+1",
            "1.5",
            " 1",
            "１",
            "64000001",
            "99999999999999999999999999999",
        ] {
            assert!(parse("budget", invalid, 64_000_000).is_err(), "{invalid}");
        }
        assert_eq!(parse("budget", "1", 64_000_000).unwrap(), 1);
        assert_eq!(parse("budget", "64000000", 64_000_000).unwrap(), 64_000_000);
    }
}
