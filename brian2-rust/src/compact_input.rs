use super::*;
use serde::de::{self, Visitor};
use std::fmt;
use std::ops::Deref;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct EncodedBits {
    bytes: [u8; 16],
    len: u8,
}
impl EncodedBits {
    pub(crate) fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len as usize]).unwrap()
    }
}
impl Deref for EncodedBits {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl Serialize for EncodedBits {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}
impl<'de> Deserialize<'de> for EncodedBits {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct BitsVisitor;
        impl Visitor<'_> for BitsVisitor {
            type Value = EncodedBits;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a canonical encoded scalar")
            }
            fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<EncodedBits, E> {
                if !matches!(value.len(), 2 | 8 | 16)
                    || !value
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err(E::custom("invalid encoded scalar"));
                }
                let mut bytes = [0u8; 16];
                bytes[..value.len()].copy_from_slice(value.as_bytes());
                Ok(EncodedBits {
                    bytes,
                    len: value.len() as u8,
                })
            }
        }
        deserializer.deserialize_str(BitsVisitor)
    }
}

pub(crate) enum EncodedArray {
    Uniform {
        value: EncodedBits,
        len: usize,
    },
    Dense(Vec<EncodedBits>),
    Indexed {
        values: Vec<EncodedBits>,
        indices: Vec<u32>,
    },
}
impl EncodedArray {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Uniform { len, .. } => *len,
            Self::Dense(v) => v.len(),
            Self::Indexed { indices, .. } => indices.len(),
        }
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(crate) fn unique_iter(&self) -> EncodedIter<'_> {
        EncodedIter {
            array: self,
            position: 0,
            limit: match self {
                Self::Uniform { len, .. } => usize::from(*len > 0),
                Self::Dense(v) => v.len(),
                Self::Indexed { values, .. } => values.len(),
            },
            unique: true,
        }
    }
    pub(crate) fn iter(&self) -> EncodedIter<'_> {
        EncodedIter {
            array: self,
            position: 0,
            limit: self.len(),
            unique: false,
        }
    }
}
pub(crate) struct EncodedIter<'a> {
    array: &'a EncodedArray,
    position: usize,
    limit: usize,
    unique: bool,
}
impl<'a> Iterator for EncodedIter<'a> {
    type Item = &'a EncodedBits;
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.limit {
            return None;
        }
        let index = self.position;
        self.position += 1;
        Some(match self.array {
            EncodedArray::Uniform { value, .. } => value,
            EncodedArray::Dense(v) => &v[index],
            EncodedArray::Indexed { values, indices } => {
                &values[if self.unique {
                    index
                } else {
                    indices[index] as usize
                }]
            }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.limit - self.position;
        (n, Some(n))
    }
}
impl ExactSizeIterator for EncodedIter<'_> {}
impl<'a> IntoIterator for &'a EncodedArray {
    type Item = &'a EncodedBits;
    type IntoIter = EncodedIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl Serialize for EncodedArray {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(self.len()))?;
        for value in self {
            seq.serialize_element(value)?;
        }
        seq.end()
    }
}
impl<'de> Deserialize<'de> for EncodedArray {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct ArrayVisitor;
        impl<'de> Visitor<'de> for ArrayVisitor {
            type Value = EncodedArray;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an encoded scalar array")
            }
            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let Some(first) = seq.next_element::<EncodedBits>()? else {
                    return Ok(EncodedArray::Dense(Vec::new()));
                };
                let mut len = 1;
                while let Some(value) = seq.next_element::<EncodedBits>()? {
                    if value == first {
                        len += 1;
                        continue;
                    }
                    // Event traces often repeat across many edges even after
                    // warmup. Bound the palette; arbitrary dense state falls
                    // back once its distinct-value count exceeds the cap.
                    let mut lookup = std::collections::HashMap::new();
                    lookup.insert(first.clone(), 0u32);
                    lookup.insert(value.clone(), 1u32);
                    let mut values = vec![first, value];
                    let mut indices = vec![0u32; len];
                    indices.push(1);
                    while let Some(value) = seq.next_element::<EncodedBits>()? {
                        if let Some(&index) = lookup.get(&value) {
                            indices.push(index);
                        } else if values.len() < 65536 {
                            let index = values.len() as u32;
                            lookup.insert(value.clone(), index);
                            values.push(value);
                            indices.push(index);
                        } else {
                            let mut dense: Vec<_> = indices
                                .iter()
                                .map(|&index| values[index as usize].clone())
                                .collect();
                            dense.push(value);
                            drop(lookup);
                            drop(values);
                            drop(indices);
                            while let Some(value) = seq.next_element::<EncodedBits>()? {
                                dense.push(value);
                            }
                            return Ok(EncodedArray::Dense(dense));
                        }
                    }
                    return Ok(EncodedArray::Indexed { values, indices });
                }
                Ok(EncodedArray::Uniform { value: first, len })
            }
        }
        d.deserialize_seq(ArrayVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: String,
    definition: serde_json::Value,
    instance: Instance,
    run: serde_json::Value,
    protocol: serde_json::Value,
}
struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.update(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn json<W: Write, T: Serialize>(w: &mut W, v: &T) -> Result<()> {
    serde_json::to_writer(w, v)?;
    Ok(())
}
fn small<W: Write, T: Serialize>(w: &mut W, v: &T) -> Result<()> {
    canonical::write(&serde_json::to_value(v)?, w)?;
    Ok(())
}
fn encoded<W: Write>(w: &mut W, array: &EncodedArray) -> Result<()> {
    w.write_all(b"[")?;
    match array {
        EncodedArray::Uniform { value, len } if *len > 0 => {
            let quoted = format!("\"{}\"", value.as_str());
            w.write_all(quoted.as_bytes())?;
            let item = format!(",{quoted}");
            let block = item.repeat(2048);
            let mut remaining = len - 1;
            while remaining >= 2048 {
                w.write_all(block.as_bytes())?;
                remaining -= 2048;
            }
            w.write_all(&block.as_bytes()[..remaining * item.len()])?;
        }
        _ => {
            for (i, value) in array.iter().enumerate() {
                if i > 0 {
                    w.write_all(b",")?;
                }
                w.write_all(b"\"")?;
                w.write_all(value.as_str().as_bytes())?;
                w.write_all(b"\"")?;
            }
        }
    }
    w.write_all(b"]")?;
    Ok(())
}
fn columns<W: Write>(w: &mut W, values: &BTreeMap<String, EncodedArray>) -> Result<()> {
    w.write_all(b"{")?;
    for (i, (name, array)) in values.iter().enumerate() {
        if i > 0 {
            w.write_all(b",")?;
        }
        json(w, name)?;
        w.write_all(b":")?;
        encoded(w, array)?;
    }
    w.write_all(b"}")?;
    Ok(())
}
fn instance_hash(instance: &Instance) -> Result<String> {
    let mut hash = HashWriter(Sha256::new());
    {
        let w = &mut BufWriter::with_capacity(65536, &mut hash);
        w.write_all(b"{\"neuron_count\":")?;
        json(w, &instance.neuron_count)?;
        w.write_all(b",\"populations\":[")?;
        for (i, p) in instance.populations.iter().enumerate() {
            if i > 0 {
                w.write_all(b",")?;
            }
            w.write_all(b"{\"initial_state\":")?;
            columns(w, &p.initial_state)?;
            w.write_all(b",\"parameters\":")?;
            columns(w, &p.parameters)?;
            w.write_all(b",\"refractory\":")?;
            if let Some(r) = &p.refractory {
                w.write_all(b"{\"initial_lastspike\":")?;
                encoded(w, &r.initial_lastspike)?;
                w.write_all(b",\"initial_not_refractory\":")?;
                json(w, &r.initial_not_refractory)?;
                w.write_all(b",\"period\":")?;
                json(w, &r.period)?;
                w.write_all(b",\"period_ticks\":")?;
                json(w, &r.period_ticks)?;
                w.write_all(b"}")?;
            } else {
                w.write_all(b"null")?;
            }
            w.write_all(b",\"spike_generator\":")?;
            if let Some(g) = &p.spike_generator {
                w.write_all(b"{\"spike_indices\":")?;
                json(w, &g.spike_indices)?;
                w.write_all(b",\"spike_ticks\":")?;
                json(w, &g.spike_ticks)?;
                w.write_all(b"}")?;
            } else {
                w.write_all(b"null")?;
            }
            w.write_all(b"}")?;
        }
        w.write_all(b"],\"rng_seed\":")?;
        json(w, &instance.rng_seed)?;
        w.write_all(b",\"synapses\":[")?;
        for (i, s) in instance.synapses.iter().enumerate() {
            if i > 0 {
                w.write_all(b",")?;
            }
            w.write_all(b"{\"initial_state\":")?;
            columns(w, &s.initial_state)?;
            w.write_all(b",\"parameters\":")?;
            columns(w, &s.parameters)?;
            w.write_all(b",\"pathways\":[")?;
            for (j, p) in s.pathways.iter().enumerate() {
                if j > 0 {
                    w.write_all(b",")?;
                }
                w.write_all(b"{\"delay\":")?;
                encoded(w, &p.delay)?;
                w.write_all(b",\"delay_initializer\":")?;
                small(w, &p.delay_initializer)?;
                w.write_all(b",\"delay_ticks\":")?;
                json(w, &p.delay_ticks)?;
                w.write_all(b",\"event\":")?;
                json(w, &p.event)?;
                w.write_all(b",\"kind\":")?;
                json(w, &p.kind)?;
                w.write_all(b",\"name\":")?;
                json(w, &p.name)?;
                w.write_all(b",\"pending\":[")?;
                for (k, e) in p.pending.iter().enumerate() {
                    if k > 0 {
                        w.write_all(b",")?;
                    }
                    small(w, e)?;
                }
                w.write_all(b"]}")?;
            }
            w.write_all(b"],\"source\":")?;
            json(w, &s.source)?;
            w.write_all(b",\"target\":")?;
            json(w, &s.target)?;
            w.write_all(b",\"topology\":")?;
            small(w, &s.topology)?;
            w.write_all(b"}")?;
        }
        w.write_all(b"]}")?;
        w.flush()?;
    }
    Ok(format!("{:x}", hash.0.finalize()))
}
pub(crate) fn validate(reader: impl Read) -> Result<()> {
    let doc: Document = serde_json::from_reader(reader)?;
    check(
        doc.schema == "b2ir-v1",
        "compact validation requires current schema",
    )?;
    let expected = serde_json::json!({"name":"b2ir","version":{"major":1,"minor":0},
        "canonical_encoding":"b2ir-canonical-json-v1","hash_algorithm":"sha256",
        "layers":{"definition":canonical_hash(&doc.definition)?,"instance":instance_hash(&doc.instance)?,"run":canonical_hash(&doc.run)?}});
    check(
        expected == doc.protocol,
        "AtlasIR canonical layer hash mismatch",
    )?;
    let model = Model {
        schema: doc.schema,
        protocol: serde_json::from_value(doc.protocol)?,
        definition: serde_json::from_value(doc.definition)?,
        instance: doc.instance,
        run: serde_json::from_value(doc.run)?,
    };
    model.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_arrays_preserve_order_hash_bytes_and_dense_fallback() {
        for values in [
            vec!["8000000000000000".to_owned(); 65539],
            (0..65539).map(|i| format!("{:016x}", i % 3)).collect(),
            (0..65539).map(|i| format!("{i:016x}")).collect(),
        ] {
            let wire = serde_json::to_vec(&values).unwrap();
            let parsed: EncodedArray = serde_json::from_slice(&wire).unwrap();
            assert_eq!(parsed.len(), values.len());
            assert_eq!(serde_json::to_vec(&parsed).unwrap(), wire);
            let mut canonical = Vec::new();
            encoded(&mut canonical, &parsed).unwrap();
            assert_eq!(canonical, wire);
            assert!(parsed.iter().zip(&values).all(|(a, b)| a.as_str() == b));
            let distinct: BTreeSet<_> = parsed.unique_iter().map(|v| v.as_str()).collect();
            assert_eq!(distinct, values.iter().map(String::as_str).collect());
            if values[0] == values[3] && values[0] != values[1] {
                assert!(matches!(parsed, EncodedArray::Indexed { .. }));
            }
        }
    }

    #[test]
    fn compact_document_checks_wire_hash_and_original_semantics() {
        let wire = include_bytes!("../tests/golden/b2ir-v1/minimal-v1.json");
        validate(wire.as_slice()).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(wire).unwrap();
        value["instance"]["neuron_count"] = serde_json::json!(123);
        assert!(validate(serde_json::to_vec(&value).unwrap().as_slice()).is_err());
        value["protocol"]["layers"] = protocol_layers(&value).unwrap();
        assert!(validate(serde_json::to_vec(&value).unwrap().as_slice()).is_err());
    }
}
