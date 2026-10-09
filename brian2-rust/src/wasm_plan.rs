//! Cross-language contract for the browser physical plan. AtlasIR remains frozen.
use super::*;

// Browser input remains bounded independently of the native large-model limit.
const MAX_WASM_IR_BYTES: u64 = 128 * 1_048_576;

/// An independently validated, resumable execution of a Python WasmPlan.
/// All model data, compiled programs and event queues are owned by this object.
pub struct WasmExecution {
    runtime: executor::Runtime,
    plan_sha256: String,
}

impl WasmExecution {
    pub fn new(model_json: &str, plan_json: &str) -> Result<Self> {
        check(
            model_json.len() as u64 <= MAX_WASM_IR_BYTES,
            "WASM model exceeds 128 MiB",
        )?;
        check(
            plan_json.len() <= 16 * 1_048_576,
            "WASM plan exceeds 16 MiB",
        )?;
        let mut value: serde_json::Value = serde_json::from_str(model_json)?;
        prepare_protocol(&mut value)?;
        verify_protocol(&value)?;
        let (model, expected) = validated_plan(value)?;
        let plan: serde_json::Value = serde_json::from_str(plan_json)?;
        check(
            plan == expected,
            "WASM execution plan does not match model, effects or compiler policy",
        )?;
        // The verified plan supplies the actual dispatch order; it is not an
        // explain-only sidecar. No CPU fusion/parallel choices are reused.
        let dispatch = plan["logical"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n["ordinal"].as_u64().unwrap() as usize)
            .collect();
        let plan_sha256 = canonical_hash(&plan)?;
        let runtime = executor::Runtime::new(model, dispatch)?;
        Ok(Self {
            runtime,
            plan_sha256,
        })
    }

    pub fn step(&mut self, max_ticks: usize) -> Result<bool> {
        self.runtime.step(max_ticks)
    }
    pub fn spike_count(&self) -> usize {
        self.runtime.spike_count()
    }
    pub fn is_finished(&self) -> bool {
        self.runtime.is_finished()
    }
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }

    /// Native-compatible little-endian dumps, with metadata bound to the plan.
    pub fn results(&self) -> Result<(Vec<u8>, Vec<u8>, serde_json::Value)> {
        let mut results = Vec::new();
        let mut events = Vec::new();
        let mut summary = self.runtime.write_results(&mut results, &mut events)?;
        check(
            results.len() as u64 == summary["dump_bytes"].as_u64().unwrap(),
            "result dump size mismatch",
        )?;
        check(
            events.len() as u64 == summary["event_dump_bytes"].as_u64().unwrap(),
            "event monitor dump size mismatch",
        )?;
        summary["backend"] = "wasm".into();
        summary["numeric_profile"] = "reference-f64".into();
        summary["plan_sha256"] = self.plan_sha256.clone().into();
        Ok((results, events, summary))
    }
}

fn validated_plan(value: serde_json::Value) -> Result<(Model, serde_json::Value)> {
    let layers = protocol_layers(&value)?;
    let mut model: Model = serde_json::from_value(value.clone())?;
    check(
        !model
            .instance
            .synapses
            .iter()
            .any(|s| matches!(s.topology, TopologyInstance::BinaryCsr { .. })),
        "WASM requires inline or procedural topology; binary CSR file paths are unsupported",
    )?;
    check(
        model.definition.functions.iter().all(|f| f.body.is_some()),
        "WASM requires portable Function bodies; native-only Functions are unsupported",
    )?;
    model.validate()?;
    model.complete_execution_effects()?;
    let nodes: Vec<_> = model.definition.schedule.nodes.iter().enumerate().map(|(ordinal, n)| {
            serde_json::json!({"id": n.id, "ordinal": ordinal, "operation": n.operation,
                "owner_kind": n.owner_kind, "owner_index": n.owner_index, "item_index": n.item_index,
                "clock": n.clock, "reads": n.effects.reads, "writes": n.effects.writes,
                "dependencies": n.dependencies})
        }).collect();
    let clocks: Vec<_> = model
        .definition
        .clocks
        .iter()
        .zip(&model.run.clocks)
        .enumerate()
        .map(|(clock, (d, r))| {
            serde_json::json!({"clock": clock, "dt": d.dt,
                "start_tick": r.start_tick, "steps": r.steps})
        })
        .collect();
    let expected = serde_json::json!({
        "schema": "b2-wasm-plan-v0", "planner_version": "wasm-plan-1",
        "definition_sha256": layers["definition"], "instance_sha256": layers["instance"],
        "run_sha256": layers["run"], "numeric_profile": "reference-f64",
        "strategy": "serial-tiled-v1", "logical": {"nodes": nodes, "clocks": clocks}
    });
    Ok((model, expected))
}

/// Authoring boundary: explicitly accept an edited AtlasIR draft, validate its
/// semantics and produce new wire hashes and a new browser execution plan.
/// Loading an existing bundle still uses the strict constructor above.
pub fn compile_browser_bundle(draft_json: &str) -> Result<String> {
    check(
        draft_json.len() as u64 <= MAX_WASM_IR_BYTES,
        "WASM model exceeds 128 MiB",
    )?;
    let mut value: serde_json::Value = serde_json::from_str(draft_json)?;
    check(value["schema"] == "b2ir-v1", "authoring requires AtlasIR v1")?;
    value["protocol"] = expected_protocol(&value)?;
    let (_, plan) = validated_plan(value.clone())?;
    Ok(serde_json::to_string(&serde_json::json!({
        "schema": "b2-wasm-bundle-v0",
        "model_json": serde_json::to_string(&value)?,
        "plan_json": serde_json::to_string(&plan)?,
        "plan_sha256": canonical_hash(&plan)?,
    }))?)
}
