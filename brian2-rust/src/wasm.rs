//! Browser bindings. No filesystem, system clock, threads, Python or WASI.
use crate::WasmExecution;
use wasm_bindgen::prelude::*;

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[wasm_bindgen]
pub struct BrowserExecutor {
    execution: WasmExecution,
    output: Option<(Vec<u8>, Vec<u8>, serde_json::Value)>,
}

#[wasm_bindgen]
impl BrowserExecutor {
    #[wasm_bindgen(constructor)]
    pub fn new(model_json: &str, plan_json: &str) -> std::result::Result<BrowserExecutor, JsValue> {
        Ok(Self {
            execution: WasmExecution::new(model_json, plan_json).map_err(js_error)?,
            output: None,
        })
    }

    /// Advance at most max_ticks scheduler instants (all coincident clocks).
    /// Returns true when complete. Yield to the browser between calls.
    pub fn step(&mut self, max_ticks: u32) -> std::result::Result<bool, JsValue> {
        self.execution.step(max_ticks as usize).map_err(js_error)
    }

    #[wasm_bindgen(getter)]
    pub fn spike_count(&self) -> usize {
        self.execution.spike_count()
    }

    #[wasm_bindgen(getter)]
    pub fn finished(&self) -> bool {
        self.execution.is_finished()
    }

    #[wasm_bindgen(getter)]
    pub fn plan_sha256(&self) -> String {
        self.execution.plan_sha256().to_owned()
    }

    fn collect_output(&mut self) -> std::result::Result<(), JsValue> {
        if self.output.is_none() {
            self.output = Some(self.execution.results().map_err(js_error)?);
        }
        Ok(())
    }

    pub fn results(&mut self) -> std::result::Result<Vec<u8>, JsValue> {
        self.collect_output()?;
        Ok(self.output.as_ref().unwrap().0.clone())
    }
    pub fn events(&mut self) -> std::result::Result<Vec<u8>, JsValue> {
        self.collect_output()?;
        Ok(self.output.as_ref().unwrap().1.clone())
    }
    pub fn summary(&mut self) -> std::result::Result<String, JsValue> {
        self.collect_output()?;
        serde_json::to_string(&self.output.as_ref().unwrap().2).map_err(js_error)
    }
}

#[wasm_bindgen]
pub fn compile_model(draft_json: &str) -> std::result::Result<String, JsValue> {
    crate::compile_browser_bundle(draft_json).map_err(js_error)
}
