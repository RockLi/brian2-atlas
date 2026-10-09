# AtlasIR v1 — Frozen Intermediate Representation

Status: **frozen**

AtlasIR is the intermediate representation of the Atlas execution architecture.
Brian2 is the currently implemented modeling frontend. This naming does not
claim another implemented frontend.

The `b2ir-*` schema, canonical-encoding and ABI identifiers below are retained
wire compatibility identifiers. Model bytes, layer hashes, existing caches and
supported predecessor migrations are unchanged by the public rename.

Python callers can use `brian2_atlas.ir` for canonical serialization, integrity
checks and migration. Native function registration accepts `atlasir-c-abi-v1`,
`atlasir-cuda-device-v1`, `atlasir-metal-v1` and `atlasir-wgsl-v1`; their `b2ir-*`
spellings remain accepted and produce identical serialized descriptors.
Conflicting registrations under both spellings are rejected.

Schema identifier: `b2ir-v1`

Protocol version: `1.0`

Canonical encoding: `b2ir-canonical-json-v1`
Normative implementation: the Python writer in `python/brian2_rust/protocol.py`
and the independent Rust reader/validator in `src/model.rs`.

This document defines the compatibility contract between a Brian2 frontend and
standalone execution backends. “MUST”, “MUST NOT”, “SHOULD” and “MAY” are used
in their RFC 2119 sense.

## 1. Freeze contract

`b2ir-v1` is immutable. A producer or consumer MUST NOT change the meaning,
default, ordering, numeric encoding or validation rule of an existing v1 field.
An incompatible change requires a new schema identifier and an explicit,
tested migration. Unknown fields are rejected.

A backend MAY support only a declared subset of valid v1 models. It MUST reject
an unsupported valid model before creating simulation result files; it MUST NOT
silently change schedule, dtype, units, RNG, event, indexing or assignment
semantics. Backend capability is therefore separate from IR validity.

The frozen contract covers the input IR and its semantic identity. Native AOT
artifact manifests and result dump formats have their own version identifiers
and may evolve independently.

## 2. Root and integrity envelope

A document is a JSON object with exactly these keys:

- `schema`: exactly `b2ir-v1`;
- `protocol`: the integrity envelope;
- `definition`: immutable model structure and executable semantics;
- `instance`: initial arrays, topology and RNG seed;
- `run`: absolute time interval and clock tick ranges.

The protocol envelope is:

```json
{
  "name": "b2ir",
  "version": {"major": 1, "minor": 0},
  "canonical_encoding": "b2ir-canonical-json-v1",
  "hash_algorithm": "sha256",
  "layers": {
    "definition": "<64 lower-case hex digits>",
    "instance": "<64 lower-case hex digits>",
    "run": "<64 lower-case hex digits>"
  }
}
```

Each layer hash is SHA-256 over that layer’s canonical bytes. The envelope is
not included in any layer hash. A consumer MUST verify the envelope before
validation or execution.

## 3. Canonical JSON

`b2ir-canonical-json-v1` is UTF-8 JSON with:

- object keys sorted by Unicode code point;
- no whitespace between tokens;
- `,` and `:` as compact separators;
- non-ASCII characters emitted as UTF-8, not `\u` escapes;
- JSON strings escaped by the standard JSON rules;
- no NaN or infinity numeric literals.

The existing Python writer defines JSON number formatting (in particular for
SI dimension exponents). Finite binary64 JSON numbers use shortest round-trip
digits, fixed notation for decimal exponents -4 through 15, and otherwise a
lower-case `e` with an explicit sign and at least two exponent digits.
Integral floating-point numbers in fixed notation retain `.0`; negative zero
is `-0.0`. JSON integers remain integers. Consumers MUST parse binary64 JSON
numbers with correct rounding and MUST NOT substitute a host serializer's
different exponent formatting before hashing.

Model floating-point values are not JSON numbers. An f64 is exactly 16
lower-case hexadecimal digits containing its IEEE-754 binary64 bits in network
byte order; an f32 is exactly 8 digits containing binary32 bits. Signed and
unsigned integers use fixed-width two’s-complement hexadecimal strings. Bool
instance values are `00` or `01`. This prevents decimal formatting from
changing identity or results.

## 4. Definition layer

`definition` contains exactly:

- `clocks[]`;
- `populations[]`;
- `synapses[]`;
- `functions[]`;
- `schedule`;
- `numeric_profile`;
- `rng_algorithm`.

For v1, `numeric_profile` is `reference-f64` and `rng_algorithm` is
`splitmix64-counter-v1`.

### 4.1 Types, dimensions and domains

Public storage dtypes are `f32`, `f64`, `i32`, `i64`, `u32`, `u64` and `bool`.
Expressions additionally use logical `index` and `tick` types. Every symbol and
assignment carries a dtype, a seven-element SI dimension vector and an index
domain (`scalar`, `neuron` or `synapse`).

The Rust validator independently infers dtype and dimension for every
expression. In particular:

- addition/subtraction/comparison require equal dimensions, except literal
  zero where explicitly permitted;
- multiplication/division combine dimensions;
- powers apply the scalar exponent to dimensions;
- exponential, logarithmic and trigonometric operations require dimensionless
  arguments;
- assignment requires the target dtype and dimensions;
- threshold and logical operators require bool/dimensionless results;
- time, dt, delay and rate sites are checked against their required dimensions;
- `index` and `tick` remain logical values until an explicit conversion.

Unit and type validation occurs once while loading/compiling and adds no hot
loop work.

### 4.2 Clocks and time

Each clock has a stable index, name and positive finite dt. Executable objects
refer to clocks by index. `run.clocks[]` supplies an exact `start_tick` and
`steps` for every clock. `run.start` and `run.duration` are absolute f64 times.

Backends MUST choose the next active time across all clocks and execute active
objects in canonical schedule order. Tick conversion, refractory deadlines and
delay quantisation use their explicitly versioned rules; they are not inferred
from host floating-point rounding modes.

### 4.3 Population schema

Every population independently defines its count, global offset, clock,
state/parameter tables, linked-variable table, CodeObject table, named event
table, refractory descriptor and monitor descriptors. No two populations are
required to have the same equations, dtype layout, dt, threshold, reset,
refractory policy or CodeObjectSpec.

A linked variable names a source population/state and one index policy:
`identity`, fixed `constant` indices, local integer `state`, or local integer
`parameter`. Reads belong to the source resource; dynamic index reads also
belong to the local mapping resource. Out-of-range dynamic indices are errors.

### 4.4 Synapse schema

Every Synapses definition independently names source/target populations and
local endpoint ranges, state/parameter tables, endpoint aliases and
CodeObjects. Instance topology is either:

- `explicit`: u32 source/target edge arrays in creation order; or
- `fixed_total`: edge count, seed and procedural initializers; or
- `binary_csr`: an immutable external `B2CSR001` source-major graph, declared
  endpoint/edge/column counts, SHA-256 and column-to-parameter mapping. Mapped
  per-edge parameters must be constant float64. The loader validates both the
  contents and checksum. AOT copies the graph into its self-contained instance;
  this storage extension preserves the existing v1 envelope and canonical hashes
  for models using the original topology forms.

Each named pre/post pathway binds an EventStream and carries an independent
scalar, per-edge or procedural delay plus pending events. Edge creation order is
semantic: event expansion and non-commutative assignments preserve it.

### 4.5 Expressions and assignments

Expression nodes are tagged objects. V1 includes literals, typed integers,
bools, loads, casts, Function calls, counter RNG draws, TimedArray access,
logical index/tick conversions, unary mathematical operations, arithmetic,
comparison, boolean operations, `clip`, `timestep` and `tick_offset`.

CodeObjectSpec contains `name`, `kind`, `when`, `order`, `clock`, iteration
domain, scalar statements, vector statements, conditional-write metadata and
declared read/write effects. Temporaries are local to one CodeObject. Statement
order and old-state snapshots are semantic; a backend may fuse or eliminate a
temporary only after proving observational equivalence.

### 4.6 Events and refractory state

`spike` is a conventional EventStream, not a special wire format. Custom named
events use the same protocol. Threshold producers, reset consumers, pre/post
pathways and EventMonitor objects bind streams by name.

`lastspike` and `not_refractory` are explicit state resources. Fixed-time and
expression refractory policies are validated state machines. `(unless
refractory)` is represented per assignment; reset is not implicitly guarded.

### 4.7 Function Contract and backend ABI

Every user Function declares:

- stable name and semantic version;
- `b2ir-function-v1` ABI;
- ordered argument names, dtypes and dimensions;
- return dtype and dimensions;
- stateful/deterministic/thread-safe/RNG effects;
- an optional portable `b2ir-expression-v1` body and its canonical hash;
- optional backend implementations keyed by `cpu`, `cuda`, `metal` or `wgsl`.

Backend implementation ABI names are respectively `b2ir-c-abi-v1`,
`b2ir-cuda-device-v1`, `b2ir-metal-v1` and `b2ir-wgsl-v1`. Each descriptor
contains an entry point, source text and SHA-256. Entries are unique within a
backend.

The CPU C ABI currently admits f64, i64 and bool signatures. They map to C
`double`, `int64_t` and `uint8_t`. The AOT builder compiles an independent C11
translation unit, injects a compile-time signature check and statically links
the object. No Rust ABI or Python callback crosses the hot path.

The opt-in GPU float32 profile maps these Function dtypes to Metal/CUDA
`float`, signed 64-bit `long` and `bool`, with an exact native signature check.
Native-only Functions use the selected backend descriptor; portable bodies keep
their established lowering. See [GPU Functions](GPU_FUNCTIONS.md) for source
scope, call checks, frontend usage and current verification status.

Portable bodies are structurally verified as pure and independently type/unit
checked. Python closure/global capture, control flow, hidden RNG and unavailable
source fail closed unless the user explicitly supplies a backend
implementation. Explicit native source is trusted executable code: hashes prove
identity and integrity, not semantic purity.

## 5. Schedule and Effect Algebra

`definition.schedule` contains:

- ordered Brian base slots;
- their canonical `before_<slot>`, `<slot>`, `after_<slot>` expansion;
- every executable node sorted by `(slot, order, name, id)`;
- node owner, item index, clock and operation;
- sorted unique resource reads/writes;
- the exact RAW/WAR/WAW dependency set induced by earlier nodes.

The Rust validator reconstructs metadata, effects and dependencies from the
referenced object and rejects omissions, duplicates, forged effects or wrong
ordering. The reference backend executes this list directly.

The frozen v1 wire graph records explicit CodeObject accesses and event
bindings. Before using it as an execution/optimization graph, consumers MUST
also derive the implicit refractory state-machine accesses from each operation:

- A `spike` threshold on a refractory population reads `not_refractory` and
  writes both `lastspike` and `not_refractory`.
- A fixed-refractory state updater reads `lastspike` and writes
  `not_refractory` before evaluating its explicit statements.

These accesses are unconditional conservative effects, even if a particular
lane does not spike. Execution dependencies must include them. They are
completed after wire validation and hashing; the frozen serialized graph and
its existing hashes are preserved. Python AOT ordering and fusion proofs and
the Rust execution graph apply this completion independently. A custom schedule
that cannot safely be implemented by the AOT phase order is rejected before
compilation; the reference backend continues to execute the canonical order.

An optimizing backend MAY reorder, fuse, parallelise or delay nodes only when
its Effect Algebra proves that every inverted or crossed pair is free of
observable RAW, WAR and WAW conflicts. Failure to prove safety requires the
canonical order or an explicit unsupported-model error.

## 6. Instance and run layers

`instance` contains neuron count, u64 RNG seed, population arrays, synapse
arrays/topologies/pathways and pending event state. State and parameter arrays
are SoA and their order is the Definition table order.

`run` contains absolute start/duration and the per-clock tick interval. A
segmented run is equivalent to the corresponding continuous run when Definition
and compatible Instance identity are preserved. Pending delays, RNG counters,
refractory state and monitor continuation are part of this contract.

Definition, Instance and Run have independent hashes so a compiled Definition
may be reused with a compatible Instance/Run without accepting semantic drift.

## 7. Migration and conformance

The v1 readers accept probe v34, v35, v36 and v37 only through explicit
migration. V35–v37 envelopes are verified before mutation; v34 predates the
envelope and MUST NOT carry one. Migration fills only fields whose old meaning
has a single v1 representation, then emits a new v1 envelope. All older or
unknown schemas are rejected.

A conforming producer must pass the Python/Rust canonical hash vectors. A
conforming consumer must pass the frozen golden corpus, tamper tests, migration
corpus and semantic differential tests. The authoritative fixtures live in
`tests/golden/b2ir-v1/`.

## 8. Deliberately separate versioned protocols

The following are not silently coupled to `b2ir-v1`:

- native AOT manifest (`b2-native-probe-v0` until separately frozen);
- binary result dump (`b2-result-dump-v3`);
- binary event dump (`B2EVT001`);
- backend capability report (`b2-capability-report-v1`).

Changing one of these protocols does not change valid AtlasIR semantics, but every
consumer must still validate its own identifier and integrity fields.

AOT manifests used for instance replacement record `definition_sha256` and
`run_sha256` in addition to the generated Rust source hash. Replacement requires
both semantic layers and the generated source to match, so changes to separate
C Function translation units cannot silently reuse old linked code. Artifacts
without these semantic hashes must be rebuilt before instance replacement.
