//! Isolated native supervised-training CLI. Never invokes another ML runtime.
fn report_error(error: &dyn std::error::Error, rank: usize) {
    let mut message = error.to_string();
    if message.len() > 4096 {
        let mut end = 4096 - " [truncated]".len();
        while !message.is_char_boundary(end) { end -= 1; }
        message.truncate(end);
        message.push_str(" [truncated]");
    }
    // The coordinator opts in. Persist before MPI_Abort or a broken stderr
    // pipe can discard the cause. A per-rank temporary and atomic rename leave
    // one complete error result even when several owners fail simultaneously.
    if std::env::var("B2_TRAIN_ERROR_RESULT").as_deref() == Ok("1") {
        if let Some(output) = std::env::args_os().nth(2) {
            let output = std::path::PathBuf::from(output);
            let temporary = output.with_extension(format!("error-{rank}.tmp"));
            let payload = serde_json::json!({"schema":"b2-native-training-error-v1","message":message});
            if let Ok(bytes) = serde_json::to_vec(&payload) {
                if std::fs::write(&temporary, bytes).is_ok() {
                    let _ = std::fs::rename(&temporary, &output);
                }
            }
        }
    }
    eprintln!("b2-train: {message}");
}
fn main() {
    let mut mpi = match b2_runner::training::mpi::Context::from_environment() {
        Ok(context) => context,
        Err(error) => {
            let rank = ["OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"].iter()
                .find_map(|name| std::env::var(name).ok()?.parse::<usize>().ok()).unwrap_or(0);
            report_error(error.as_ref(), rank);
            std::process::exit(1);
        }
    };
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<_> = std::env::args_os().collect();
        if args.len() != 3 {
            return Err("usage: b2-train REQUEST.json RESULT.json".into());
        }
        if std::fs::metadata(&args[1])?.len() > 64 * 1024 * 1024 {
            return Err("training request exceeds 64 MiB input limit".into());
        }
        let bytes = std::fs::read(&args[1])?;
        if let Some(context) = &mpi {
            use sha2::{Digest, Sha256};
            let mut hash = Sha256::new();
            hash.update(&bytes);
            hash.update(std::fs::read(std::env::current_exe()?)?);
            hash.update(std::fs::read(std::env::var("B2_TRAIN_MPI_LIB")?)?);
            context.agree(&hash.finalize().into())?;
        }
        let request = serde_json::from_slice(&bytes)?;
        let result = b2_runner::training::execute_distributed(request, mpi.as_ref())?;
        if mpi.as_ref().is_none_or(|m| m.rank == 0) {
            std::fs::write(&args[2], serde_json::to_vec(&result)?)?;
        }
        if let Some(context) = &mut mpi {
            context.finish()?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        if error.is::<b2_runner::training::PeerFailure>() {
            // This rank participated in the failed collective but has no
            // original cause. MPI_Finalize waits for the owner to persist its
            // error and abort, preventing a fast peer from killing that owner.
            eprintln!("b2-train: {error}");
            if let Some(context) = &mut mpi { let _ = context.finish(); }
        } else {
            report_error(error.as_ref(), mpi.as_ref().map_or(0, |m| m.rank));
        }
        drop(mpi); // abort peers before exiting on a rank-local failure
        std::process::exit(1);
    }
}
