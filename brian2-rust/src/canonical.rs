//! Frozen Python-compatible canonical JSON, including JSON dimension numbers.
use serde_json::Value;
use std::io::{self, Write};

fn float_text(number: &serde_json::Number) -> String {
    // serde_json supplies shortest round-tripping digits; Python's repr uses
    // fixed notation for exponents -4..15 and signed, two-digit exponents.
    let text = number.to_string();
    let (sign, unsigned) = text
        .strip_prefix('-')
        .map_or(("", text.as_str()), |value| ("-", value));
    let (mantissa, exponent) = unsigned
        .split_once('e')
        .map_or((unsigned, 0), |(mantissa, exponent)| {
            (mantissa, exponent.parse::<i32>().expect("JSON exponent"))
        });
    let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let digits = mantissa.replace('.', "");
    let leading = digits.bytes().take_while(|byte| *byte == b'0').count();
    let digits = digits[leading..].trim_end_matches('0');
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let exponent = exponent + decimal - leading as i32 - 1;
    if !(-4..16).contains(&exponent) {
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        format!("{sign}{}{fraction}e{exponent:+03}", &digits[..1])
    } else {
        let decimal = exponent + 1;
        if decimal <= 0 {
            format!("{sign}0.{}{digits}", "0".repeat((-decimal) as usize))
        } else if decimal as usize >= digits.len() {
            format!(
                "{sign}{digits}{}.0",
                "0".repeat(decimal as usize - digits.len())
            )
        } else {
            let (whole, fraction) = digits.split_at(decimal as usize);
            format!("{sign}{whole}.{fraction}")
        }
    }
}

pub fn write(value: &Value, writer: &mut impl Write) -> io::Result<()> {
    match value {
        Value::Number(number) if number.is_f64() => writer.write_all(float_text(number).as_bytes()),
        Value::Array(values) => {
            writer.write_all(b"[")?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    writer.write_all(b",")?;
                }
                write(value, writer)?;
            }
            writer.write_all(b"]")
        }
        Value::Object(values) => {
            writer.write_all(b"{")?;
            // serde_json's default Map is ordered by UTF-8 keys, which is also
            // Unicode code point order for valid strings.
            for (index, (key, value)) in values.iter().enumerate() {
                if index != 0 {
                    writer.write_all(b",")?;
                }
                serde_json::to_writer(&mut *writer, key)?;
                writer.write_all(b":")?;
                write(value, writer)?;
            }
            writer.write_all(b"}")
        }
        _ => serde_json::to_writer(writer, value).map_err(Into::into),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn python_float_notation() {
        for text in [
            "0.0",
            "-0.0",
            "1e-07",
            "-1e-05",
            "0.0001",
            "1.0",
            "1000000000000000.0",
            "1e+16",
            "1e+23",
            "5e-324",
            "1.7976931348623157e+308",
            "1.2345678901234567",
        ] {
            let value = serde_json::from_str(text).unwrap();
            let mut actual = Vec::new();
            super::write(&value, &mut actual).unwrap();
            assert_eq!(String::from_utf8(actual).unwrap(), text);
        }
    }
}
