//! Native command-line adapter for the shared execution engine.
fn main() {
    if let Err(error) = b2_runner::run_cli() {
        eprintln!("b2-runner: {error}");
        std::process::exit(1);
    }
}
