//! Validated B2IR execution shared by native and browser backends.
include!("model.rs");
mod binary_topology;
mod canonical;
mod compact_input;
mod executor;
mod gpu_initialization;
mod resource_limits;
pub mod large_topology;
mod wasm_plan;
pub use wasm_plan::{compile_browser_bundle, WasmExecution};
