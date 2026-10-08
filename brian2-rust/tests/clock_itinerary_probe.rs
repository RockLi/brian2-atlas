//! Standalone probe of the same scheduler compiled into b2-train.
#[path = "../src/training/clock.rs"]
mod clock;
fn main() {
    if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let start: f64 = args[1].parse()?; let end: f64 = args[2].parse()?;
    let dts = args[3].split(',').map(str::parse).collect::<Result<Vec<f64>, _>>()?;
    let order = args[4].split(',').filter(|v| !v.is_empty()).map(str::parse).collect::<Result<Vec<usize>, _>>()?;
    let s = clock::Schedule { start, dts: &dts, epsilon: 1e-4, order: &order };
    let mut cursor = if args.len() > 5 {
        let mut fields = args[5].split(':'); let visits = fields.next().unwrap().parse()?;
        let ticks = fields.next().unwrap().split(',').map(str::parse).collect::<Result<Vec<u64>, _>>()?;
        let calls = fields.next().unwrap().split(',').map(str::parse).collect::<Result<Vec<u64>, _>>()?;
        let initial_calls = fields.next().unwrap().split(',').map(str::parse).collect::<Result<Vec<u64>, _>>()?;
        let start = fields.next().unwrap().parse()?;
        s.restore(&clock::State { start, visits, ticks, calls, initial_calls })?
    } else { s.cursor()? };
    if args.len() > 6 { cursor.restart(args[6].parse()?)?; }
    let mut trace = Vec::new(); let mut draws = Vec::new();
    cursor.run_until(end, |v| {
        if trace.len() > 100_000 { return Err("probe trace budget exceeded"); }
        for &i in v.active {
            draws.push(format!("[{},{},{}]", i, v.ticks[i], v.calls[i]));
            trace.push(format!("[{:?},{},{:?},{:?}]", v.time, i, v.ticks, v.times));
        }
        Ok(())
    })?;
    let state = cursor.snapshot();
    println!("{{\"trace\":[{}],\"ticks\":{:?},\"times\":{:?},\"visits\":{},\"start\":{:?},\"calls\":{:?},\"initial_calls\":{:?},\"draws\":[{}]}}",
        trace.join(","), state.ticks, cursor.visit().times, state.visits, state.start, state.calls, state.initial_calls, draws.join(","));
    Ok(())
}
